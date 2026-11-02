import type { Api, Context, Model } from '@earendil-works/pi-ai';
import type { ExtensionContext } from '@earendil-works/pi-coding-agent';
import type { ThinkingLevel } from '@earendil-works/pi-agent-core';
import type {
  RouterConfig,
  RoutingDecision,
  RouterTier,
  RouterPinByProfile,
  RouterThinkingByProfile,
  RouterModelRegistry,
  RouterModelCapabilityLookup,
  VirtualRouterState,
  RouterVirtualModelDefinition,
  ModelRouteRequest,
  ModelRoute,
} from './types';
import {
  profileNames,
  parseCanonicalModelRef,
  findModelByRef,
  isObjectRecord,
  ROUTER_TIERS,
  THINKING_LEVELS,
  DEFAULT_THINKING_LEVELS,
  resolveContextWindow,
  resolveMaxTokens,
  resolveThinkingLevels,
  collectProfileThinkingLevels,
  clampThinkingLevel,
} from './config';
import {
  phaseForTier,
  resolveAvailableTier,
  buildRoutingDecision,
  decideRouting,
  runClassifier,
  hasImageAttachment,
} from './routing';
import { DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_TOKENS } from './constants';
import {
  classifyLimitError,
  isValidEventTimestamp,
  type CooldownStore,
} from './cooldown';
import {
  collectEncryptedReasoningExclusions,
  collectUnrecordedEncryptedRejections,
  type EncryptedRejectionRecord,
} from './encrypted-reasoning';

const countedMessages = new WeakSet<object>();

export const markMessageCostCounted = (message: unknown): void => {
  if (isObjectRecord(message)) {
    countedMessages.add(message);
  }
};

const extractSingleMessageCost = (message: unknown): number => {
  if (
    !isObjectRecord(message) ||
    (message.role !== 'assistant' && message.role !== 'toolResult') ||
    countedMessages.has(message)
  ) {
    return 0;
  }
  countedMessages.add(message);
  const total =
    isObjectRecord(message.usage) && isObjectRecord(message.usage.cost)
      ? message.usage.cost.total
      : undefined;
  return typeof total === 'number' && total > 0 ? total : 0;
};

export const extractUncountedMessageCost = (input: unknown): number => {
  if (!isObjectRecord(input)) return 0;
  if (input.role === 'assistant' || input.role === 'toolResult') {
    return extractSingleMessageCost(input);
  }
  let cost = extractSingleMessageCost(input.message);
  if (Array.isArray(input.toolResults)) {
    for (const toolResult of input.toolResults) {
      cost += extractSingleMessageCost(toolResult);
    }
  }
  return cost;
};

const supportsReasoning = (
  profile: RouterConfig['profiles'][string],
  modelRegistry: RouterModelCapabilityLookup | undefined,
): boolean =>
  ROUTER_TIERS.some((tier) => {
    const tierConfig = profile[tier];
    if (!tierConfig) return false;
    const modelRefs = [
      tierConfig.model,
      ...(tierConfig.fallbacks?.map((fallback) => fallback.model) ?? []),
    ];
    return modelRegistry
      ? modelRefs.some(
          (modelRef) => findModelByRef(modelRef, modelRegistry)?.reasoning,
        )
      : tierConfig.reasoning !== false;
  });

// Exported so the `agent_before_settle` auto-continue guard
// (extensions/index.ts) can enumerate the same tier's configured model +
// fallback refs it needs to check for a usable replacement, without a
// second, drifting reimplementation of "this tier's candidate refs".
export const getTierModels = (
  profile: RouterConfig['profiles'][string],
  tier: RouterTier,
  preferredModel?: string,
): string[] => {
  const tierConfig = profile[tier];
  if (!tierConfig) return preferredModel ? [preferredModel] : [];
  const base = [
    ...new Set([
      tierConfig.model,
      ...(tierConfig.fallbacks?.map((fallback) => fallback.model) ?? []),
    ]),
  ];
  if (!preferredModel) return base;
  const preferredIdx = base.indexOf(preferredModel);
  if (preferredIdx === -1) return [preferredModel, ...base];
  if (preferredIdx === 0) return base;
  return [...base.slice(preferredIdx), ...base.slice(0, preferredIdx)];
};

const inferDecisionFromPrevious = (
  profileName: string,
  profile: RouterConfig['profiles'][string],
  previous: ModelRouteRequest<VirtualRouterState>['previous'],
  thinkingOverrides?: RouterThinkingByProfile[string],
): RoutingDecision | undefined => {
  if (!previous) return undefined;
  const targetRef = `${previous.model.provider}/${previous.model.id}`;
  const tier = ROUTER_TIERS.find((t) =>
    getTierModels(profile, t).includes(targetRef),
  );
  if (!tier) return undefined;
  const decision = buildRoutingDecision(
    profileName,
    profile,
    tier,
    phaseForTier(tier),
    `Inferred ${tier} tier from previous response (${targetRef}).`,
    thinkingOverrides,
  );
  return {
    ...decision,
    targetProvider: previous.model.provider,
    targetModelId: previous.model.id,
    targetLabel: targetRef,
    thinking:
      thinkingOverrides?.[tier] ?? previous.thinkingLevel ?? decision.thinking,
    ...(profile[tier]?.model !== targetRef ? { isFallback: true } : {}),
  };
};

export interface RouterProviderState<TContext = ExtensionContext> {
  lastRegisteredModels: string;
  readonly currentConfig: RouterConfig;
  readonly currentModelRegistry: RouterModelRegistry | undefined;
  readonly lastExtensionContext: TContext | undefined;
  selectedProfile: string | undefined;
  routerEnabled: boolean;
  lastDecision: RoutingDecision | undefined;
  readonly thinkingByProfile: RouterThinkingByProfile;
  readonly pinnedTierByProfile: RouterPinByProfile;
  accumulatedCost: number;
  readonly cooldownStore?: CooldownStore;
  // When this process started, and which transcript-observed limit-error
  // message timestamps it has already turned into a cooldown. Both are
  // in-memory only (never persisted): a historical error from before this
  // process started, or one already processed this run, is never re-recorded.
  readonly processStartTime: number;
  readonly processedTranscriptFailures: Set<string>;
  // Set by extensions/index.ts's turn_end handler for exactly one route()
  // call, right after it decides to fire the one-shot auto-continue: names
  // the model that just hit a non-retryable usage-limit error. route() below
  // fires the single "continuing on <next>" notice once the replacement
  // model is actually resolved (not in turn_end, which cannot know it yet),
  // then always clears this so no later call can re-fire it.
  pendingAutoContinueNotice?: { failedRef: string };
  // Same one-shot notice channel for the encrypted-reasoning recovery
  // (extensions/encrypted-reasoning.ts): set by extensions/index.ts's
  // agent_before_settle handler when it requests a continuation away from a
  // model that rejected this conversation's replayed encrypted reasoning.
  pendingEncryptedReasoningNotice?: { failedRef: string };
  // Durable, signature-bound rejection metadata for this session
  // (extensions/encrypted-reasoning.ts's `EncryptedRejectionRecord`), loaded
  // from session custom entries by extensions/index.ts and extended whenever
  // route() or the agent_before_settle handler observes a new rejection.
  readonly encryptedRejections?: readonly EncryptedRejectionRecord[];
}

export interface RouterProviderActions<TContext = ExtensionContext> {
  persistState: () => void;
  recordDebugDecision: (decision: RoutingDecision) => void;
  getThinkingOverride: (
    profileName: string,
    tier: RouterTier,
  ) => ThinkingLevel | undefined;
  updateStatus: (ctx: TContext) => void;
  syncPiThinkingLevel: (level: ThinkingLevel | undefined) => void;
  // Optional so existing action mocks that don't exercise auto-continue need
  // no change. Implementations must themselves be advisory (never throw).
  notify?: (
    ctx: TContext,
    message: string,
    type?: 'info' | 'warning' | 'error',
  ) => void;
  // Persists one encrypted-reasoning rejection as session metadata (an
  // idempotent custom entry) and adds it to the in-memory record list route()
  // reads. Advisory like everything else here: a failure must never break the
  // request.
  recordEncryptedRejection?: (record: EncryptedRejectionRecord) => void;
}

// Cooldowns are an advisory routing hint, never a correctness mechanism: a
// turn must never fail because the (possibly caller-supplied, possibly
// malformed) CooldownStore threw. Every failure here is logged at most once
// and degrades to "route as if no cooldown existed" rather than throwing out
// of route().
const loggedCooldownErrors = new Set<string>();
const logCooldownErrorOnce = (context: string, error: unknown): void => {
  const message = `pi-model-router: cooldown ${context} failed: ${String(error)}`;
  if (loggedCooldownErrors.has(message)) return;
  loggedCooldownErrors.add(message);
  console.error(message);
};

// Shared by every candidates()-filtering call site in route(): a throwing
// CooldownStore (e.g. a misbehaving stub, not just our own createCooldownStore)
// must fall back to the unfiltered candidate list instead of breaking the turn.
const filterCooldownCandidates = (
  store: CooldownStore,
  modelRefs: string[],
  now: number,
  isUsable: (ref: string) => boolean,
): string[] => {
  try {
    return store.candidates(modelRefs, now, isUsable);
  } catch (error) {
    logCooldownErrorOnce('candidates', error);
    return modelRefs;
  }
};

/**
 * Resolves a canonical "provider/model" ref against a model registry: not
 * the virtual `router` provider itself, present in the registry, and
 * (when the registry reports it) authenticated. Exported so the
 * `agent_before_settle` auto-continue guard (extensions/index.ts,
 * extensions/autocontinue.ts) can ask "is there a usable replacement for the
 * failed model?" with the exact same resolvability check `routeVirtualModelRequest`
 * itself uses to pick a real model, instead of a second, drifting
 * reimplementation of "resolvable + authenticated".
 */
export const resolveRoutedCandidate = (
  modelRef: string,
  registry: RouterModelRegistry,
):
  | {
      ok: true;
      model: Model<Api>;
      provider: string;
      modelId: string;
    }
  | {
      ok: false;
      error: string;
    } => {
  const { provider, modelId } = parseCanonicalModelRef(modelRef);
  if (provider === 'router') {
    return {
      ok: false,
      error: `Invalid routed target "${modelRef}": cannot route to virtual router provider.`,
    };
  }
  const targetModel = registry.find(provider, modelId);
  if (!targetModel) {
    return {
      ok: false,
      error: `Routed model not found: ${provider}/${modelId}`,
    };
  }
  if (registry.hasConfiguredAuth && !registry.hasConfiguredAuth(targetModel)) {
    return {
      ok: false,
      error: `No configured credentials for routed model: ${provider}/${modelId}`,
    };
  }
  return {
    ok: true,
    model: targetModel,
    provider,
    modelId,
  };
};

/**
 * Inputs of the ONE candidate calculation used both by route() and by the
 * agent_before_settle replacement check (extensions/index.ts). Sharing it is
 * what makes the continuation decision and routing agree: the decision asks
 * "will route() find a model?" with the same profile, tier, image
 * requirement, exclusion set, and cooldown state route() will use.
 */
const lastResolutionError = (errors: readonly string[]): Error =>
  new Error(
    errors.length > 0
      ? errors[errors.length - 1]
      : 'Failed to resolve any model in the routing chain.',
  );

export interface RoutedCandidateSelectionInput {
  profile: RouterConfig['profiles'][string];
  tier: RouterTier;
  preferredModel?: string;
  requiresImage: boolean;
  registry: RouterModelRegistry;
  // Models excluded by the encrypted-reasoning recovery. Exclusion is
  // mandatory: an excluded ref is never returned, and never restored when
  // everything else is filtered out.
  excludedRefs: ReadonlySet<string>;
  cooldownStore?: CooldownStore;
  now: number;
  // The model that just failed, when route() is serving a retry: candidates
  // configured after it are tried first.
  failedRef?: string;
}

export interface RoutedCandidateSelection {
  // Ordered refs route() will actually try: resolvable + authenticated,
  // image-capable when the conversation needs it, not excluded, and
  // cooldown-filtered. Empty means "stop" -- never a resurrected model.
  refs: string[];
  // Refs dropped by the encrypted-reasoning exclusions, for the stop message.
  excludedRefs: string[];
  // Resolution errors for configured refs that are not routable at all.
  resolutionErrors: string[];
}

const rotateAfterFailed = (
  refs: string[],
  base: readonly string[],
  failedRef: string,
): string[] => {
  const failedIdx = base.indexOf(failedRef);
  if (failedIdx === -1 || refs.length <= 1) return refs;
  const baseIndex = (ref: string): number => {
    const index = base.indexOf(ref);
    return index === -1 ? Number.MAX_SAFE_INTEGER : index;
  };
  const group = (ref: string): number => (baseIndex(ref) > failedIdx ? 0 : 1);
  return [...refs].sort(
    (a, b) => group(a) - group(b) || baseIndex(a) - baseIndex(b),
  );
};

export const selectRoutedCandidates = (
  input: RoutedCandidateSelectionInput,
): RoutedCandidateSelection => {
  const base = getTierModels(input.profile, input.tier, input.preferredModel);
  const resolved = base.map((modelRef) => ({
    modelRef,
    result: resolveRoutedCandidate(modelRef, input.registry),
  }));
  const resolutionErrors = resolved
    .filter((entry) => !entry.result.ok)
    .map((entry) => (entry.result.ok ? '' : entry.result.error))
    .filter((error) => error !== '');
  const routable = resolved.filter(
    (
      entry,
    ): entry is {
      modelRef: string;
      result: {
        ok: true;
        model: Model<Api>;
        provider: string;
        modelId: string;
      };
    } => entry.result.ok,
  );

  // Image compatibility behaves exactly as before: only image-capable models
  // may replay an image-bearing conversation, and when none is configured the
  // preferred (first) candidate remains the last resort.
  const imageEligible = input.requiresImage
    ? routable.filter((entry) => entry.result.model.input.includes('image'))
    : routable;
  const considered =
    imageEligible.length > 0 ? imageEligible : routable.slice(0, 1);

  const excludedRefs = considered
    .filter((entry) => input.excludedRefs.has(entry.modelRef))
    .map((entry) => entry.modelRef);
  const eligible = considered.filter(
    (entry) => !input.excludedRefs.has(entry.modelRef),
  );
  if (eligible.length === 0) {
    return { refs: [], excludedRefs, resolutionErrors };
  }
  const refs = eligible.map((entry) => entry.modelRef);

  // Exclusion is applied BEFORE cooldown selection, and the cooldown store's
  // "keep the soonest-expiring usable model" fallback only ever chooses among
  // the refs it is given -- so a cooling model can never resurrect an
  // excluded one (the P1 this ordering fixes).
  const cooled = input.cooldownStore
    ? filterCooldownCandidates(
        input.cooldownStore,
        refs,
        input.now,
        (ref) => resolveRoutedCandidate(ref, input.registry).ok,
      )
    : refs;
  const withoutExcluded = cooled.filter((ref) => !input.excludedRefs.has(ref));
  const ordered = input.failedRef
    ? rotateAfterFailed(withoutExcluded, base, input.failedRef)
    : withoutExcluded;
  return { refs: ordered, excludedRefs, resolutionErrors };
};

/**
 * The clear stop message for the one case routing must never paper over: every
 * model that could serve this tier is excluded by the encrypted-reasoning
 * recovery. Replaying one of them can only reproduce the rejection, so route()
 * rejects and Pi ends the run with this error instead.
 */
export const encryptedReasoningExhaustedMessage = (
  profileName: string,
  tier: RouterTier,
  excludedRefs: readonly string[],
): string =>
  `pi-model-router: stopping instead of replaying rejected encrypted reasoning. ` +
  `Every model configured for profile "${profileName}" tier "${tier}" rejected this conversation's replayed ` +
  `encrypted reasoning (${excludedRefs.join(', ')}), so no remaining model can serve it. ` +
  `Switch models with /model, or add another model to this tier's fallbacks.`;

export const routeVirtualModelRequest = async <TContext = ExtensionContext>(
  request: ModelRouteRequest<VirtualRouterState>,
  ctx: TContext | undefined,
  state: RouterProviderState<TContext>,
  actions: RouterProviderActions<TContext>,
): Promise<ModelRoute<VirtualRouterState>> => {
  try {
    const ctxRegistry =
      ctx &&
      typeof ctx === 'object' &&
      'modelRegistry' in ctx &&
      ctx.modelRegistry
        ? (ctx.modelRegistry as RouterModelRegistry)
        : undefined;
    const registry = ctxRegistry ?? state.currentModelRegistry;
    if (!registry) {
      throw new Error('Router model registry is unavailable.');
    }

    const profileName = request.model.id;
    const profile = state.currentConfig.profiles[profileName];
    if (!profile) {
      throw new Error(`Unknown router profile: ${profileName}`);
    }

    const resolveCandidate = (modelRef: string) =>
      resolveRoutedCandidate(modelRef, registry);

    // Encrypted-reasoning exclusions for THIS request, from both sources
    // (extensions/encrypted-reasoning.ts):
    //  - durable, signature-bound rejection metadata persisted in this session
    //    (survives restarts and branch summaries that move the rejecting error
    //    off the active path while the signed response remains), and
    //  - the rejection messages Pi surfaces for this request: persisted
    //    assistant errors in the projection AND `request.failed`, which is how
    //    a retryable wrapper around the same rejection reaches routing before
    //    Pi's retry loop re-selects a model.
    const rejectionRecords = state.encryptedRejections ?? [];
    // The API the registered target for a ref currently speaks: part of the
    // replay identity (provider + API + model), so a record bound under one API
    // never bans a target whose API changed.
    const resolveModelApi = (ref: string): string | undefined => {
      const resolved = resolveRoutedCandidate(ref, registry);
      return resolved.ok ? resolved.model.api : undefined;
    };
    const encryptedReasoningExclusions = collectEncryptedReasoningExclusions({
      messages: request.messages,
      failed: request.failed?.message,
      records: rejectionRecords,
      resolveModelApi,
    });
    // Persist metadata for any rejection visible here that has none yet, so a
    // retryable (request.failed) or foreign (projection-only) rejection becomes
    // signature-bound session state for later requests. Advisory: a session
    // write failure must never break the request.
    try {
      for (const record of collectUnrecordedEncryptedRejections({
        messages: request.messages,
        failed: request.failed?.message,
        records: rejectionRecords,
        at: Date.now(),
      })) {
        actions.recordEncryptedRejection?.(record);
      }
    } catch {
      // Ignore: advisory bookkeeping only.
    }

    // The single candidate calculation shared with the agent_before_settle
    // replacement check (extensions/index.ts). Exclusion is mandatory and is
    // applied before cooldown selection and retry rotation, and when nothing
    // survives it the request stops with a clear error instead of replaying a
    // rejected model.
    const selectCandidates = (input: {
      tier: RouterTier;
      preferredModel?: string;
      requiresImage: boolean;
      failedRef?: string;
    }) =>
      selectRoutedCandidates({
        profile,
        tier: input.tier,
        preferredModel: input.preferredModel,
        requiresImage: input.requiresImage,
        registry,
        excludedRefs: encryptedReasoningExclusions,
        cooldownStore,
        now: Date.now(),
        failedRef: input.failedRef,
      });

    const resolveSelectedModel = (
      selection: RoutedCandidateSelection,
      tier: RouterTier,
    ) => {
      if (selection.refs.length > 0) {
        return resolveFirstAvailableModel([selection.refs[0]]);
      }
      if (selection.excludedRefs.length > 0) {
        throw new Error(
          encryptedReasoningExhaustedMessage(
            profileName,
            tier,
            selection.excludedRefs,
          ),
        );
      }
      throw lastResolutionError(selection.resolutionErrors);
    };

    const resolveFirstAvailableModel = (candidates: string[]) => {
      let lastError: unknown;
      for (let i = 0; i < candidates.length; i++) {
        const modelRef = candidates[i];
        const resolved = resolveCandidate(modelRef);
        if (!resolved.ok) {
          lastError = new Error(resolved.error);
          continue;
        }
        return { ...resolved, modelRef, index: i };
      }
      throw lastError instanceof Error
        ? lastError
        : new Error(
            typeof lastError === 'string'
              ? lastError
              : 'Failed to resolve any model in the routing chain.',
          );
    };

    const resolveDelegatedThinking = (
      tier: RouterTier,
      targetRef: string,
      targetModel: Model<Api>,
    ): {
      effectiveThinking: ThinkingLevel;
      delegatedThinking: ThinkingLevel;
    } => {
      const tierConfig = profile[tier];
      const fallbackThinking =
        targetRef === tierConfig?.model
          ? undefined
          : tierConfig?.fallbacks?.find(
              (fallback) => fallback.model === targetRef,
            )?.thinking;
      const effectiveThinking =
        actions.getThinkingOverride(profileName, tier) ??
        fallbackThinking ??
        tierConfig?.thinking ??
        tier;
      const clamped = clampThinkingLevel(
        effectiveThinking,
        resolveThinkingLevels(tier, profile, registry, targetModel),
        targetModel.thinkingLevelMap?.off !== null,
      );
      const delegatedThinking: ThinkingLevel =
        targetModel.reasoning && clamped !== 'off' ? clamped : 'off';
      return { effectiveThinking, delegatedThinking };
    };

    const cooldownStore = state.cooldownStore;
    const cooldownNow = Date.now();
    if (cooldownStore) {
      // Every cooldown touchpoint below (the retry record, the transcript
      // failure scan, and the transcript success scan) is wrapped in one
      // try/catch: a throw from any of them -- including from a malformed or
      // deliberately hostile CooldownStore, not just our own
      // createCooldownStore -- must degrade to "no cooldown recorded this
      // turn" instead of breaking the turn.
      try {
        const failedText = request.failed?.message.errorMessage ?? '';
        if (
          request.reason === 'retry' &&
          request.failed &&
          classifyLimitError(failedText)
        ) {
          cooldownStore.record(
            `${request.failed.model.provider}/${request.failed.model.id}`,
            failedText,
            'retry',
            state.currentConfig,
            cooldownNow,
          );
        }

        // Transcript scan: only a NEW assistant error (timestamped after this
        // process started, and not already recorded by this process) is turned
        // into a cooldown. Dedup is a plain in-memory set, never persisted, so a
        // historical failure can never be re-armed after a restart or an
        // explicit `/router cooldown clear` — there's nothing on disk to replay.
        for (const message of request.messages) {
          if (!isObjectRecord(message)) continue;
          const candidate = message as unknown as Record<string, unknown>;
          if (
            candidate.role !== 'assistant' ||
            candidate.stopReason !== 'error'
          )
            continue;
          // A malformed numeric timestamp (NaN, Infinity, 1e20, or anything
          // outside Date's representable range) must never reach record():
          // ignore that transcript message for cooldown purposes instead of
          // letting an invalid recorded_at escape to iso().
          const timestamp =
            typeof candidate.timestamp === 'number' &&
            isValidEventTimestamp(candidate.timestamp)
              ? candidate.timestamp
              : undefined;
          if (timestamp === undefined || timestamp <= state.processStartTime)
            continue;
          const identity = String(timestamp);
          if (state.processedTranscriptFailures.has(identity)) continue;
          const provider =
            typeof candidate.provider === 'string' ? candidate.provider : '';
          const model =
            typeof candidate.model === 'string' ? candidate.model : '';
          const errorMessage =
            typeof candidate.errorMessage === 'string'
              ? candidate.errorMessage
              : '';
          if (!provider || !model || !classifyLimitError(errorMessage))
            continue;
          state.processedTranscriptFailures.add(identity);
          // recordedAt is the failed message's own (event) timestamp, not
          // cooldownNow (scan time): noteSuccess below compares a success
          // message's timestamp against this recorded_at, and a success that
          // is genuinely newer than the failure must clear it even though
          // both timestamps predate the scan that observed them. Only the
          // expiry math (cooldownNow) stays anchored to real wall-clock time.
          cooldownStore.record(
            `${provider}/${model}`,
            errorMessage,
            'transcript',
            state.currentConfig,
            cooldownNow,
            timestamp,
          );
        }

        // A success only clears a model's cooldown when it's actually newer
        // than the failure that armed it: an older success already sitting in
        // the transcript (processed again on every turn) must never erase a
        // cooldown recorded from a later failure. noteSuccess enforces this
        // against the model's own recorded_at; the processStartTime check here
        // additionally drops successes that predate this process, matching the
        // same rule already applied to transcript-sourced failures above.
        for (const message of request.messages) {
          if (!isObjectRecord(message)) continue;
          const candidate = message as unknown as Record<string, unknown>;
          if (candidate.role !== 'assistant') continue;
          if (
            candidate.stopReason === 'error' ||
            candidate.stopReason === 'aborted'
          )
            continue;
          // Same malformed-timestamp guard as the failure scan above.
          const successTimestamp =
            typeof candidate.timestamp === 'number' &&
            isValidEventTimestamp(candidate.timestamp)
              ? candidate.timestamp
              : undefined;
          if (
            successTimestamp === undefined ||
            successTimestamp <= state.processStartTime
          )
            continue;
          if (
            typeof candidate.provider === 'string' &&
            typeof candidate.model === 'string'
          ) {
            cooldownStore.noteSuccess(
              `${candidate.provider}/${candidate.model}`,
              successTimestamp,
            );
          }
        }
      } catch (error) {
        logCooldownErrorOnce('record/transcript-scan', error);
      }
    }

    const pinnedTier = state.pinnedTierByProfile[profileName];
    const previousDecision =
      request.state?.lastDecision?.profile === profileName
        ? request.state.lastDecision
        : state.lastDecision?.profile === profileName
          ? state.lastDecision
          : inferDecisionFromPrevious(
              profileName,
              profile,
              request.previous,
              state.thinkingByProfile[profileName],
            );

    // Direct requests (e.g. compaction summaries) route cleanly without mutating session state.
    // Prefer the currently active model (from previous decision) to preserve the session's
    // prompt cache, falling back to pinned tier or medium only if no active model exists.
    if (request.reason === 'direct') {
      const directTier = resolveAvailableTier(
        profile,
        pinnedTier ?? previousDecision?.tier ?? 'medium',
      );
      const selection = selectCandidates({
        tier: directTier,
        preferredModel:
          previousDecision?.tier === directTier
            ? previousDecision.targetLabel
            : undefined,
        // Compaction/summary requests never carried image filtering; keep it
        // that way.
        requiresImage: false,
      });
      const resolved = resolveSelectedModel(selection, directTier);
      const { delegatedThinking } = resolveDelegatedThinking(
        directTier,
        resolved.modelRef,
        resolved.model,
      );
      return {
        model: resolved.model,
        thinkingLevel: delegatedThinking,
      };
    }

    // Account for any completed assistant or toolResult (e.g. codemode) usage not yet recorded.
    for (const msg of request.messages) {
      state.accumulatedCost += extractUncountedMessageCost(msg);
    }
    if (request.failed?.message) {
      state.accumulatedCost += extractUncountedMessageCost(
        request.failed.message,
      );
    }

    state.selectedProfile = profileName;
    state.routerEnabled = true;

    const isBudgetExceeded =
      state.currentConfig.maxSessionBudget !== undefined &&
      state.accumulatedCost >= state.currentConfig.maxSessionBudget;

    const context: Context = { messages: [...request.messages] };
    const lastMessage = request.messages.findLast((m) => m.role !== 'system');
    const isContinuation =
      request.reason === 'continuation' ||
      (request.reason !== 'retry' && lastMessage?.role === 'toolResult');
    const hasChangedPinOnContinuation =
      isContinuation &&
      pinnedTier !== undefined &&
      pinnedTier !== previousDecision?.tier;

    let decision: RoutingDecision;

    if (
      (isContinuation || request.reason === 'retry') &&
      previousDecision &&
      !hasChangedPinOnContinuation
    ) {
      if (isBudgetExceeded && previousDecision.tier === 'high') {
        const downgradedTier = resolveAvailableTier(profile, 'medium');
        decision = buildRoutingDecision(
          profileName,
          profile,
          downgradedTier,
          phaseForTier(downgradedTier),
          `Budget exceeded. Downgraded from high to ${downgradedTier} tier. (Original: ${previousDecision.reasoning})`,
          state.thinkingByProfile[profileName],
          previousDecision.isClassifier,
        );
        decision.isBudgetForced = true;
      } else {
        decision = {
          ...previousDecision,
          timestamp: Date.now(),
          reasoning:
            isContinuation &&
            !previousDecision.reasoning.startsWith('Preserved ')
              ? `Preserved ${previousDecision.targetLabel} for tool-result continuation ` +
                `to maintain prompt cache and thinking state. (Original: ${previousDecision.reasoning})`
              : previousDecision.reasoning,
        };
      }
    } else {
      decision = decideRouting(
        context,
        profileName,
        profile,
        previousDecision,
        pinnedTier,
        state.thinkingByProfile[profileName],
        state.currentConfig.phaseBias,
        state.currentConfig.rules,
        isBudgetExceeded,
      );

      // Classifier Override — skip when pinned, rule-matched, or budget exceeded.
      if (
        !isContinuation &&
        state.currentConfig.classifierModel &&
        !pinnedTier &&
        !decision.isRuleMatched &&
        !isBudgetExceeded
      ) {
        const classifierResult = await runClassifier(
          state.currentConfig.classifierModel.model,
          registry,
          context,
          previousDecision?.phase,
          state.currentConfig.classifierModel.thinking,
          state.currentConfig.phaseBias,
          request.signal,
        );
        if (classifierResult?.cost) {
          state.accumulatedCost += classifierResult.cost;
        }
        if (classifierResult) {
          const classifiedTier = resolveAvailableTier(
            profile,
            classifierResult.tier,
          );
          decision = buildRoutingDecision(
            profileName,
            profile,
            classifiedTier,
            phaseForTier(classifiedTier),
            `Classifier: ${classifierResult.reasoning}`,
            state.thinkingByProfile[profileName],
            true,
          );
        }
      }
    }

    const imageAttached = hasImageAttachment(context);
    const checkModelSupportsImage = (modelRef: string) =>
      findModelByRef(modelRef, registry)?.input?.includes('image') ?? false;

    const preferredForImageCheck =
      previousDecision?.tier === decision.tier
        ? previousDecision.targetLabel
        : decision.targetLabel;

    if (
      imageAttached &&
      !getTierModels(profile, decision.tier, preferredForImageCheck).some(
        checkModelSupportsImage,
      )
    ) {
      const higherTiers = ROUTER_TIERS.slice(
        0,
        ROUTER_TIERS.indexOf(decision.tier),
      ).reverse() as RouterTier[];
      const foundTier = higherTiers.find((t) =>
        getTierModels(profile, t).some(checkModelSupportsImage),
      );
      if (foundTier) {
        decision = buildRoutingDecision(
          profileName,
          profile,
          foundTier,
          phaseForTier(foundTier),
          `Forced ${foundTier} tier because the originally routed ${decision.tier} tier does not support image attachments.`,
          state.thinkingByProfile[profileName],
          false,
        );
      }
    }

    const isFirstTurnForProfile =
      !isContinuation &&
      request.reason !== 'retry' &&
      !previousDecision &&
      !pinnedTier;
    if (state.currentConfig.autoPinFirstTurnDecision && isFirstTurnForProfile) {
      state.pinnedTierByProfile[profileName] = decision.tier;
      decision = {
        ...decision,
        reasoning: `${decision.reasoning} (Auto-pinned to ${decision.tier} tier from first turn.)`,
      };
    }

    const preferredModel =
      previousDecision?.tier === decision.tier
        ? previousDecision.targetLabel
        : decision.targetLabel;

    // One shared calculation for image compatibility, resolvability, the
    // mandatory encrypted-reasoning exclusions, cooldown selection and retry
    // rotation -- the same one the agent_before_settle replacement check uses
    // (extensions/index.ts). Never re-derives candidates here.
    const failedRef =
      request.reason === 'retry' && request.failed?.model
        ? `${request.failed.model.provider}/${request.failed.model.id}`
        : undefined;
    const selection = selectCandidates({
      tier: decision.tier,
      preferredModel,
      requiresImage: imageAttached,
      failedRef,
    });

    const resolved = resolveSelectedModel(selection, decision.tier);
    const selectedModel = resolved.model;
    if (resolved.modelRef !== decision.targetLabel || resolved.index > 0) {
      const originalTarget = decision.targetLabel;
      decision = {
        ...decision,
        targetProvider: resolved.provider,
        targetModelId: resolved.modelId,
        targetLabel: resolved.modelRef,
        isFallback: true,
        reasoning: encryptedReasoningExclusions.has(originalTarget)
          ? `Fallback to ${resolved.modelRef} because ${originalTarget} rejected this conversation's replayed encrypted reasoning. (Original: ${decision.reasoning})`
          : request.reason === 'retry' && request.failed
            ? `Fallback to ${resolved.modelRef} after ${request.failed.model.provider}/${request.failed.model.id} failed${request.failed.message.errorMessage ? ` (${request.failed.message.errorMessage})` : ''}. (Original: ${decision.reasoning})`
            : resolved.modelRef !== originalTarget
              ? preferredModel === resolved.modelRef
                ? `Preserved fallback ${resolved.modelRef} for ${decision.tier} tier. (Original: ${decision.reasoning})`
                : `Fallback to ${resolved.modelRef} because ${originalTarget} was unavailable. (Original: ${decision.reasoning})`
              : decision.reasoning,
      };
    }

    const { delegatedThinking } = resolveDelegatedThinking(
      decision.tier,
      resolved.modelRef,
      selectedModel,
    );

    decision = {
      ...decision,
      thinking: delegatedThinking,
    };

    state.lastDecision = decision;
    actions.recordDebugDecision(decision);

    const effectiveCtx = ctx ?? state.lastExtensionContext;
    // Consume the pending auto-continue notice before any other UI
    // operation below: `syncPiThinkingLevel`/`updateStatus` can throw (stale
    // extension context), and if the notice were cleared after them, that
    // throw would leave it pending into a later, unrelated route() call,
    // which would then fire a notice naming the wrong failed model. Clearing
    // it here, unconditionally and first, guarantees it is used at most once
    // no matter what happens next in this route() call.
    const pendingAutoContinueNotice = state.pendingAutoContinueNotice;
    state.pendingAutoContinueNotice = undefined;
    // Same one-shot rule for the encrypted-reasoning recovery's notice.
    const pendingEncryptedReasoningNotice =
      state.pendingEncryptedReasoningNotice;
    state.pendingEncryptedReasoningNotice = undefined;
    try {
      actions.syncPiThinkingLevel(delegatedThinking);
      if (effectiveCtx) {
        actions.updateStatus(effectiveCtx);
      }
      // Fire the one auto-continue notice agent_before_settle armed, now
      // that the actual replacement model is known.
      if (pendingAutoContinueNotice) {
        const { failedRef } = pendingAutoContinueNotice;
        if (effectiveCtx) {
          actions.notify?.(
            effectiveCtx,
            `${failedRef} hit its usage limit; continuing on ${resolved.modelRef}`,
            'info',
          );
        }
      }
      if (pendingEncryptedReasoningNotice) {
        const { failedRef } = pendingEncryptedReasoningNotice;
        if (effectiveCtx) {
          actions.notify?.(
            effectiveCtx,
            `${failedRef} rejected this conversation's encrypted reasoning; continuing on ${resolved.modelRef}`,
            'info',
          );
        }
      }
    } catch {
      // Stale extension context — skip non-critical UI updates.
    }

    return {
      model: selectedModel,
      thinkingLevel: delegatedThinking,
      state: {
        phase: decision.phase,
        tier: decision.tier,
        lastDecision: decision,
      },
    };
  } finally {
    try {
      actions.persistState();
    } catch {
      // Ignore: extension context may be stale after session teardown.
    }
  }
};

export const registerRouterProvider = <TContext = ExtensionContext>(
  pi: {
    registerVirtualModel: (
      model: RouterVirtualModelDefinition<VirtualRouterState, TContext>,
    ) => void;
    unregisterVirtualModel?: (provider: string, id: string) => void;
  },
  state: RouterProviderState<TContext>,
  actions: RouterProviderActions<TContext>,
) => {
  const profileList = profileNames(state.currentConfig);

  const modelDefinitions = profileList.map((name) => {
    const profile = state.currentConfig.profiles[name];

    // Report the MAX context window and max output tokens across all tiers
    // before the first response. Afterwards, Pi uses the physical model's limits
    // and auto-compacts before sending if the routed model's context is smaller.
    let maxContextWindow = DEFAULT_CONTEXT_WINDOW;
    let maxMaxTokens = DEFAULT_MAX_TOKENS;
    for (const tier of ROUTER_TIERS) {
      if (!profile[tier]) continue;
      const cw = resolveContextWindow(
        tier,
        profile,
        state.currentModelRegistry,
      );
      const mot = resolveMaxTokens(tier, profile, state.currentModelRegistry);
      if (cw > maxContextWindow) maxContextWindow = cw;
      if (mot > maxMaxTokens) maxMaxTokens = mot;
    }

    const hasReasoning = supportsReasoning(profile, state.currentModelRegistry);
    const profileLevels = collectProfileThinkingLevels(
      profile,
      state.currentModelRegistry,
    );
    const thinkingLevels: ThinkingLevel[] = hasReasoning
      ? THINKING_LEVELS.filter(
          (level) =>
            level === 'off' ||
            profileLevels.has(level) ||
            (profileLevels.size === 0 &&
              DEFAULT_THINKING_LEVELS.includes(level)),
        )
      : ['off'];

    return {
      id: name,
      name: `Router ${name}`,
      thinkingLevels,
      input: ['text', 'image'] as ('text' | 'image')[],
      contextWindow: maxContextWindow,
      maxTokens: maxMaxTokens,
    };
  });

  const modelsKey = modelDefinitions
    .map(
      (m) =>
        `${m.id}:${m.contextWindow}:${m.maxTokens}:${m.thinkingLevels.join('|')}`,
    )
    .join(',');
  if (state.lastRegisteredModels === modelsKey) return;

  const previousIds = state.lastRegisteredModels
    ? state.lastRegisteredModels
        .split(',')
        .map((entry) => entry.split(':')[0])
        .filter(Boolean)
    : [];
  for (const prevId of previousIds) {
    if (!profileList.includes(prevId)) {
      pi.unregisterVirtualModel?.('router', prevId);
    }
  }

  for (const def of modelDefinitions) {
    pi.registerVirtualModel({
      provider: 'router',
      id: def.id,
      name: def.name,
      thinkingLevels: def.thinkingLevels,
      contextWindow: def.contextWindow,
      maxTokens: def.maxTokens,
      input: def.input,
      route: (request, ctx) =>
        routeVirtualModelRequest(request, ctx, state, actions),
    });
  }

  state.lastRegisteredModels = modelsKey;
};
