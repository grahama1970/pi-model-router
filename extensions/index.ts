import type { ExtensionContext } from '@earendil-works/pi-coding-agent';
import type { ThinkingLevel } from '@earendil-works/pi-agent-core';
import {
  type RouterConfig,
  type RouterExtensionAPI,
  type RouterExtensionContext,
  type RouterModelRegistry,
  type RoutingDecision,
  type RouterPinByProfile,
  type RouterThinkingByProfile,
  type RouterTier,
  type CustomSessionEntry,
} from './types';
import {
  isObjectRecord,
  loadRouterConfig,
  profileNames,
  resolveProfileName,
  applyThinkingOverride,
  getUnsupportedTiers,
} from './config';
import { MAX_DEBUG_HISTORY } from './constants';
import { isRouterPersistedState, buildPersistedState } from './state';
import {
  updateStatus,
  formatDecision,
  formatUnsupportedThinkingWarning,
} from './ui';
import { registerCommands } from './commands';
import {
  registerRouterProvider,
  markMessageCostCounted,
  extractUncountedMessageCost,
  resolveRoutedCandidate,
  getTierModels,
  selectRoutedCandidates,
} from './provider';
import { createCooldownStore } from './cooldown';
import { loadAccountRegistry, registerPersistedAccounts } from './accounts';
import type { AccountRegistry } from './accounts';
import {
  createAutoContinueGuardState,
  evaluateBeforeSettleForAutoContinue,
  logAutoContinueErrorOnce,
  resetAutoContinueGuardOnSettle,
} from './autocontinue';
import {
  conversationHasImage,
  createEncryptedReasoningRecoveryState,
  ENCRYPTED_REJECTION_ENTRY_TYPE,
  evaluateBeforeSettleForEncryptedReasoning,
  isEncryptedRejectionRecord,
  observeEncryptedReasoningFailure,
  rejectionRecordKey,
  resetEncryptedReasoningRecoveryOnSettle,
  type EncryptedRejectionRecord,
} from './encrypted-reasoning';

const routerExtension = <
  TCtx extends RouterExtensionContext = ExtensionContext,
>(
  pi: RouterExtensionAPI<TCtx>,
) => {
  const state = {
    currentConfig: { profiles: {} } as RouterConfig,
    currentModelRegistry: undefined as RouterModelRegistry | undefined,
    currentCwd: process.cwd(),
    lastDecision: undefined as RoutingDecision | undefined,
    debugEnabled: false,
    routerEnabled: false,
    selectedProfile: undefined as string | undefined,
    widgetEnabled: false,
    lastRegisteredModels: '',
    pinnedTierByProfile: {} as RouterPinByProfile,
    thinkingByProfile: {} as RouterThinkingByProfile,
    debugHistory: [] as RoutingDecision[],
    lastNonRouterModel: undefined as string | undefined,
    accumulatedCost: 0,
    cooldownStore: createCooldownStore(),
    accountRegistry: loadAccountRegistry() as AccountRegistry,
    // Provider ids this process itself registered via accounts.ts's
    // registerAccountProvider/registerPersistedAccounts -- never persisted
    // (pi.registerProvider() is runtime-only). Used to refuse overwriting a
    // provider id this feature does not own, and to refuse unregistering one
    // it never actually registered (extensions/accounts.ts's
    // registerAccountProvider/unregisterAccountProvider).
    ownedAccountProviderIds: new Set<string>(),
    processStartTime: Date.now(),
    processedTranscriptFailures: new Set<string>(),
    // In-memory only, never persisted "never chain" / "at most once per
    // provider+model per cooldown period" guard state for auto-continue
    // (extensions/autocontinue.ts) -- same session-scoped lifetime as
    // processedTranscriptFailures above.
    autoContinueGuard: createAutoContinueGuardState(),
    pendingAutoContinueNotice: undefined as { failedRef: string } | undefined,
    // Encrypted-reasoning recovery (extensions/encrypted-reasoning.ts):
    // same in-memory, session-scoped lifetime as the auto-continue guard.
    // The conversation-scoped model exclusion is NOT stored here -- route()
    // derives it from the request's own history on every call.
    encryptedReasoningRecovery: createEncryptedReasoningRecoveryState(),
    // Durable, signature-bound encrypted-reasoning rejections for THIS
    // session (extensions/encrypted-reasoning.ts's
    // `EncryptedRejectionRecord`). Seeded from the session's custom entries on
    // session_start/session_tree and extended whenever a rejection is observed,
    // so the exclusion outlives the projectional error message and survives a
    // restart or a branch summary that keeps the signed response.
    encryptedRejections: [] as EncryptedRejectionRecord[],
    pendingEncryptedReasoningNotice: undefined as
      | { failedRef: string }
      | undefined,
    lastExtensionContext: undefined as TCtx | undefined,
    lastConfigWarnings: [] as string[],
  };
  let lastPersistedSnapshot: string | undefined;
  let isInitialized = false;
  let isInternalModelSwitch = false;
  let internalThinkingChangeDepth = 0;

  const setModelInternally = async (
    model: NonNullable<ExtensionContext['model']>,
    ctx: TCtx,
  ) => {
    if (ctx.model?.provider === model.provider && ctx.model?.id === model.id) {
      return true;
    }
    isInternalModelSwitch = true;
    try {
      return await pi.setModel(model);
    } catch {
      // Extension context may be stale after session teardown.
      return false;
    } finally {
      isInternalModelSwitch = false;
    }
  };

  const syncPiThinkingLevel = (level: ThinkingLevel | undefined) => {
    if (!level) return;
    internalThinkingChangeDepth++;
    try {
      pi.setThinkingLevel(level);
    } catch {
      // Extension context may be stale after session teardown.
    } finally {
      setTimeout(() => {
        internalThinkingChangeDepth--;
      }, 0);
    }
  };

  const recordDebugDecision = (decision: RoutingDecision) => {
    state.debugHistory = [...state.debugHistory, decision].slice(
      -MAX_DEBUG_HISTORY,
    );
    if (state.debugEnabled && state.lastExtensionContext) {
      try {
        state.lastExtensionContext.ui.notify(
          `[router] ${formatDecision(decision)}`,
          'info',
        );
      } catch {
        // Ignore notification errors on stale context.
      }
    }
  };

  const getThinkingOverride = (profileName: string, tier: RouterTier) => {
    return state.thinkingByProfile[profileName]?.[tier];
  };

  const persistState = () => {
    const persisted = buildPersistedState(
      state.routerEnabled,
      state.selectedProfile,
      state.pinnedTierByProfile,
      state.thinkingByProfile,
      state.debugEnabled,
      state.widgetEnabled,
      state.debugHistory,
      state.lastDecision,
      state.lastNonRouterModel,
      state.accumulatedCost,
    );
    const snapshot = JSON.stringify({
      ...persisted,
      timestamp: 0,
      lastDecision: persisted.lastDecision
        ? { ...persisted.lastDecision, timestamp: 0 }
        : undefined,
      debugHistory: persisted.debugHistory?.map((decision) => ({
        ...decision,
        timestamp: 0,
      })),
    });
    if (snapshot === lastPersistedSnapshot) {
      return;
    }
    try {
      pi.appendEntry('router-state', persisted);
    } catch {
      // Defensive fallback: the session_shutdown event may fire after this
      // code runs (due to event loop ordering), so isActive can still be
      // true even though the runtime is already stale.
      return;
    }
    lastPersistedSnapshot = snapshot;
  };

  const actions = {
    persistState,
    recordDebugDecision,
    getThinkingOverride,
    syncPiThinkingLevel,
    notify: (
      ctx: TCtx,
      message: string,
      type: 'info' | 'warning' | 'error' = 'info',
    ) => {
      try {
        ctx.ui.notify(message, type);
      } catch {
        // Extension context may be stale after session teardown.
      }
    },
    updateStatus: (ctx: TCtx) =>
      updateStatus(
        ctx,
        state.routerEnabled,
        state.selectedProfile,
        state.pinnedTierByProfile,
        state.thinkingByProfile,
        state.lastDecision,
        state.lastNonRouterModel,
        state.accumulatedCost,
        state.widgetEnabled,
        state.currentConfig,
      ),
    reloadConfig: (ctx?: TCtx, options?: { preserveDebug?: boolean }) => {
      const loaded = loadRouterConfig(state.currentCwd);
      state.currentConfig = loaded.config;
      state.lastConfigWarnings = loaded.warnings;
      if (!options?.preserveDebug) {
        state.debugEnabled = state.currentConfig.debug ?? false;
      }
      state.selectedProfile = resolveProfileName(
        state.currentConfig,
        state.selectedProfile,
      );
      actions.registerRouterProvider();
      if (ctx) {
        actions.updateStatus(ctx);
        if (state.lastConfigWarnings.length > 0) {
          ctx.ui.notify(
            `Router Configuration Warnings:\n${state.lastConfigWarnings.join('\n')}`,
            'warning',
          );
        }
      }
    },
    ensureValidActiveRouterProfile: async (ctx: TCtx) => {
      if (ctx.model?.provider !== 'router') {
        return;
      }
      if (state.currentConfig.profiles[ctx.model.id]) {
        state.selectedProfile = ctx.model.id;
        state.routerEnabled = true;
        return;
      }

      // The active router model's profile no longer exists in config
      ctx.ui.notify(
        `Router profile "${ctx.model.id}" is no longer configured.`,
        'warning',
      );
      state.routerEnabled = false;
      state.selectedProfile = undefined;
    },
    switchToRouterProfile: async (
      profileName: string,
      ctx: TCtx,
      strict = true,
    ) => {
      if (!state.currentConfig.profiles[profileName]) {
        if (strict) {
          ctx.ui.notify(`Unknown router profile: ${profileName}`, 'error');
        }
        return false;
      }

      // Ensure the provider is registered with current capacities for this profile
      actions.registerRouterProvider();

      const routerModel = ctx.modelRegistry.find('router', profileName);
      if (!routerModel) {
        ctx.ui.notify(`Unknown router profile: ${profileName}`, 'error');
        return false;
      }
      if (ctx.model && ctx.model.provider !== 'router') {
        state.lastNonRouterModel = `${ctx.model.provider}/${ctx.model.id}`;
      }
      const success = await setModelInternally(routerModel, ctx);
      if (!success) {
        ctx.ui.notify(`Failed to switch to router/${profileName}`, 'error');
        return false;
      }
      state.selectedProfile = profileName;
      state.routerEnabled = true;
      persistState();
      actions.updateStatus(ctx);
      return true;
    },
    registerRouterProvider: () => {
      registerRouterProvider(pi, state, actions);
    },
    // Reads every rejection record this SESSION holds, from Pi's own
    // `sessionManager.getEntries()` (all entries, not just the active branch):
    // a branch summary that abandons the rejecting assistant message keeps the
    // record readable, while the fingerprint rule in
    // `collectEncryptedReasoningExclusions` decides whether it still applies.
    refreshEncryptedRejections: (ctx: TCtx) => {
      try {
        const entries = ctx.sessionManager.getEntries?.() ?? [];
        const records: EncryptedRejectionRecord[] = [];
        const seen = new Set<string>();
        for (const entry of entries) {
          if (
            entry.type !== 'custom' ||
            entry.customType !== ENCRYPTED_REJECTION_ENTRY_TYPE
          ) {
            continue;
          }
          if (!isEncryptedRejectionRecord(entry.data)) continue;
          const key = rejectionRecordKey(entry.data);
          if (seen.has(key)) continue;
          seen.add(key);
          records.push(entry.data);
        }
        state.encryptedRejections = records;
      } catch {
        // Advisory: keep the records already in memory on any failure.
      }
    },
    // Persists one rejection as idempotent session metadata. Called by the
    // agent_before_settle decision and by route() (for a rejection that only
    // `request.failed` carried), never throws.
    recordEncryptedRejection: (record: EncryptedRejectionRecord) => {
      try {
        const key = rejectionRecordKey(record);
        if (
          state.encryptedRejections.some(
            (existing) => rejectionRecordKey(existing) === key,
          )
        ) {
          return;
        }
        state.encryptedRejections = [...state.encryptedRejections, record];
        pi.appendEntry(ENCRYPTED_REJECTION_ENTRY_TYPE, record);
      } catch {
        // Advisory: routing already excludes the model for this request from
        // the rejection message itself.
      }
    },
    reregisterAccounts: (ctx: TCtx) => {
      registerPersistedAccounts(
        state.accountRegistry,
        ctx.modelRegistry,
        pi,
        state.ownedAccountProviderIds,
      );
    },
  };

  actions.reloadConfig();

  const restoreStateFromSession = async (ctx: TCtx) => {
    state.lastExtensionContext = ctx;
    state.currentModelRegistry = ctx.modelRegistry;
    state.currentCwd = ctx.cwd;
    actions.reloadConfig(ctx);
    // Re-read the registry (another process may have added an account since
    // this one started) and re-register every clone: pi.registerProvider()
    // is a runtime call, never persisted by Pi itself, so a fresh process
    // must redo it on every session_start, same as the router's own virtual
    // models via actions.registerRouterProvider() above.
    state.accountRegistry = loadAccountRegistry();
    actions.reregisterAccounts(ctx);
    actions.refreshEncryptedRejections(ctx);

    state.routerEnabled = ctx.model?.provider === 'router';
    state.selectedProfile =
      ctx.model?.provider === 'router'
        ? resolveProfileName(state.currentConfig, ctx.model.id)
        : resolveProfileName(state.currentConfig, state.selectedProfile);
    // Clear in-place to keep references intact
    for (const key of Object.keys(state.pinnedTierByProfile)) {
      delete state.pinnedTierByProfile[key];
    }
    for (const key of Object.keys(state.thinkingByProfile)) {
      delete state.thinkingByProfile[key];
    }
    state.widgetEnabled = false;
    state.debugHistory = [];
    state.accumulatedCost = 0;
    state.lastNonRouterModel =
      ctx.model && ctx.model.provider !== 'router'
        ? `${ctx.model.provider}/${ctx.model.id}`
        : state.lastNonRouterModel;
    state.lastDecision = undefined;

    const entries: readonly CustomSessionEntry[] =
      ctx.sessionManager.getBranch();
    for (const entry of entries) {
      if (entry.type === 'message' && entry.message !== undefined) {
        markMessageCostCounted(entry.message);
      }
    }
    const savedState = entries
      .filter(
        (entry) =>
          entry.type === 'custom' && entry.customType === 'router-state',
      )
      .map((entry) => entry.data)
      .findLast((data) => isRouterPersistedState(data));

    if (isRouterPersistedState(savedState)) {
      state.selectedProfile = resolveProfileName(
        state.currentConfig,
        savedState.selectedProfile,
      );
      state.routerEnabled = savedState.enabled;
      if (savedState.pinByProfile) {
        Object.assign(state.pinnedTierByProfile, savedState.pinByProfile);
      }
      if (savedState.thinkingByProfile) {
        Object.assign(state.thinkingByProfile, savedState.thinkingByProfile);
      }
      if (savedState.pinTier && state.selectedProfile) {
        state.pinnedTierByProfile[state.selectedProfile] = savedState.pinTier;
      }
      state.debugEnabled = savedState.debugEnabled ?? state.debugEnabled;
      state.widgetEnabled = savedState.widgetEnabled ?? state.widgetEnabled;
      state.debugHistory = savedState.debugHistory
        ? [...savedState.debugHistory].slice(-MAX_DEBUG_HISTORY)
        : [];
      state.lastNonRouterModel =
        savedState.lastNonRouterModel ?? state.lastNonRouterModel;
      state.accumulatedCost = savedState.accumulatedCost ?? 0;
      state.lastDecision = savedState.lastDecision;
    }

    await actions.ensureValidActiveRouterProfile(ctx);

    if (state.routerEnabled && state.selectedProfile) {
      const routerModel = ctx.modelRegistry.find(
        'router',
        state.selectedProfile,
      );
      if (routerModel) {
        const success = await setModelInternally(routerModel, ctx);
        if (!success) {
          ctx.ui.notify(
            `Failed to restore router/${state.selectedProfile} after relaunch.`,
            'warning',
          );
          state.routerEnabled = false;
        } else {
          // Sync pi's thinking level display with the router's last decision
          syncPiThinkingLevel(state.lastDecision?.thinking);
        }
      } else {
        ctx.ui.notify(
          `Unable to restore router/${state.selectedProfile}; model is unavailable.`,
          'warning',
        );
        state.routerEnabled = false;
      }
    }

    persistState();
    actions.updateStatus(ctx);
  };

  registerCommands(pi, state, actions);

  pi.on('session_start', async (_event, ctx) => {
    isInitialized = false;
    await restoreStateFromSession(ctx);
    isInitialized = true;
    if (state.debugEnabled) {
      ctx.ui.notify(
        `Router initialized with profiles: ${profileNames(state.currentConfig).join(', ')}`,
        'info',
      );
    }
  });

  pi.on('session_tree', async (_event, ctx) => {
    await restoreStateFromSession(ctx);
  });

  // Eagerly initialize the model registry from any event that provides
  // ExtensionContext. In subagent contexts (e.g. pi-dynamic-workflows),
  // session_start may never fire, but turn_start/model_select fire before every LLM
  // call — including the first call to the router provider's streamSimple.
  // Only set when not already initialized: if extensions share instances across
  // parent/subagent sessions, always overwriting would replace the parent's valid
  // registry with the subagent's — which goes stale when the subagent ends.
  const ensureInitializedFromContext = (ctx: TCtx) => {
    if (!state.currentModelRegistry) {
      state.currentModelRegistry = ctx.modelRegistry;
      state.lastExtensionContext = ctx;
      state.currentCwd = ctx.cwd;
      actions.reloadConfig(ctx);
      state.accountRegistry = loadAccountRegistry();
      actions.reregisterAccounts(ctx);
      actions.refreshEncryptedRejections(ctx);
    }
  };

  pi.on('turn_start', async (_event, ctx) => {
    ensureInitializedFromContext(ctx);
    actions.registerRouterProvider();
  });

  pi.on('model_select', async (event, ctx) => {
    ensureInitializedFromContext(ctx);
    if (!isInitialized || isInternalModelSwitch) return;
    if (event.model.provider === 'router') {
      const profileName = resolveProfileName(
        state.currentConfig,
        event.model.id,
      );
      if (!profileName) {
        ctx.ui.notify(`Unknown router profile: ${event.model.id}`, 'error');
        return;
      }

      state.routerEnabled = true;
      state.selectedProfile = profileName;
    } else {
      state.routerEnabled = false;
      state.lastNonRouterModel = `${event.model.provider}/${event.model.id}`;
    }
    persistState();
    actions.updateStatus(ctx);
  });

  // turn_end only does cost-accounting bookkeeping now: auto-continue's own
  // decision moved to agent_before_settle below, because Pi's `finishTurn`
  // discards turn_end's `continue` whenever its own retry/fallback decision
  // was already "end" -- always true for a non-retryable error, which is
  // exactly the case this mechanism exists to handle. See
  // extensions/autocontinue.ts's `evaluateBeforeSettleForAutoContinue` doc
  // comment for the full root-cause trace into Pi's source.
  pi.on('turn_end', async (event, ctx) => {
    ensureInitializedFromContext(ctx);
    if (state.routerEnabled) {
      state.accumulatedCost += extractUncountedMessageCost(event);
    }
    persistState();
    actions.updateStatus(ctx);
  });

  // Shared by the auto-continue decision below: generic in `modelRef` so it
  // answers "is this ref currently cooling" for the failed model (the
  // per-model dedup guard) and for any other candidate ref (the
  // has-a-replacement guard), off the same on-disk cooldown state.
  const isAutoContinueRefStillCooling = (modelRef: string): boolean => {
    try {
      const entry = state.cooldownStore.load(Date.now()).entries[modelRef];
      return entry !== undefined && Date.parse(entry.until) > Date.now();
    } catch {
      return false;
    }
  };

  // Requires at least one OTHER usable (resolvable + authenticated),
  // non-cooling candidate in the failed model's current profile/tier before
  // auto-continue is allowed to fire -- otherwise the continuation would
  // just re-hit the same dead end. Uses the failed message's own routed
  // profile/tier, i.e. the last routing decision in effect when it failed,
  // never a value inferred from unrelated state.
  const hasUsableAutoContinueReplacement = (failedRef: string): boolean => {
    try {
      const registry = state.currentModelRegistry;
      const profileName = state.selectedProfile;
      const tier = state.lastDecision?.tier;
      if (!registry || !profileName || !tier) return false;
      const profile = state.currentConfig.profiles[profileName];
      if (!profile) return false;
      const candidateRefs = getTierModels(profile, tier).filter(
        (ref) => ref !== failedRef,
      );
      if (candidateRefs.length === 0) return false;
      const usableRefs = candidateRefs.filter(
        (ref) => resolveRoutedCandidate(ref, registry).ok,
      );
      if (usableRefs.length === 0) return false;
      const filtered = state.cooldownStore.candidates(
        usableRefs,
        Date.now(),
        () => true,
      );
      return filtered.some((ref) => !isAutoContinueRefStillCooling(ref));
    } catch {
      return false;
    }
  };

  // Requires at least one remaining ELIGIBLE candidate, computed with the exact
  // same function route() uses (`selectRoutedCandidates`): same profile, tier,
  // image requirement, encrypted-reasoning exclusion set, and cooldown state.
  // Sharing the calculation is what guarantees the continuation can never be
  // requested for a chain that routing would then refuse to serve, and that a
  // cooling model's "keep the soonest-expiring one" fallback can never bring a
  // rejected model back.
  // API identity for the encrypted-reasoning exclusions: Pi replays a signed
  // block only to the same provider + API + model, so a record bound under one
  // API must not exclude a target whose API changed.
  const resolveRoutedModelApi = (modelRef: string): string | undefined => {
    try {
      const registry = state.currentModelRegistry;
      if (!registry) return undefined;
      const resolved = resolveRoutedCandidate(modelRef, registry);
      return resolved.ok ? resolved.model.api : undefined;
    } catch {
      return undefined;
    }
  };

  const hasUsableEncryptedReasoningReplacement = (
    failedRef: string,
    excludedRefs: ReadonlySet<string>,
    contextMessages: readonly unknown[],
  ): boolean => {
    try {
      const registry = state.currentModelRegistry;
      const profileName = state.selectedProfile;
      const tier = state.lastDecision?.tier;
      if (!registry || !profileName || !tier) return false;
      const profile = state.currentConfig.profiles[profileName];
      if (!profile) return false;
      const selection = selectRoutedCandidates({
        profile,
        tier,
        preferredModel: state.lastDecision?.targetLabel,
        requiresImage: conversationHasImage(contextMessages),
        registry,
        excludedRefs,
        cooldownStore: state.cooldownStore,
        now: Date.now(),
        failedRef,
      });
      return selection.refs.length > 0 && selection.refs[0] !== failedRef;
    } catch {
      return false;
    }
  };

  pi.on('agent_before_settle', async (event, ctx) => {
    ensureInitializedFromContext(ctx);

    // Auto-continue is advisory, exactly like cooldowns: any failure here
    // degrades to "do nothing" -- it must never throw into Pi's
    // agent_before_settle dispatch and never block the run from settling.
    // Scoped to router-routed turns only ("a turn routed by router/*").
    if (!state.routerEnabled) return undefined;
    try {
      const decision = evaluateBeforeSettleForAutoContinue(
        event,
        state.autoContinueGuard,
        isAutoContinueRefStillCooling,
        hasUsableAutoContinueReplacement,
      );
      if (!decision) {
        // Independent second recovery path: a model that rejected this
        // conversation's replayed encrypted reasoning (see
        // extensions/encrypted-reasoning.ts) has no chance of succeeding on
        // a retry, so continue on a different model the same way a user
        // would switch models by hand. The usage-limit path above keeps its
        // exact behavior; an encrypted-reasoning error is never
        // limit-classified, so the two never contend for the same failure.
        //
        // The observed rejection is recorded BEFORE the continuation decision
        // and independently of it: at final exhaustion (every candidate has
        // rejected) no continuation is requested, yet this failure's signature
        // binding must still be persisted -- otherwise a later branch summary
        // that keeps this model's signed response while dropping the error
        // would leave it selectable again, replaying the rejected payload.
        try {
          const contextMessages =
            isObjectRecord(event) &&
            isObjectRecord(event.context) &&
            Array.isArray(event.context.contextMessages)
              ? (event.context.contextMessages as readonly unknown[])
              : [];
          const observation = observeEncryptedReasoningFailure({
            messages: contextMessages,
            at: Date.now(),
          });
          if (observation?.record) {
            actions.recordEncryptedRejection(observation.record);
          }
        } catch {
          // Advisory: a metadata write failure must never block settling.
        }

        const reasoningRecovery = evaluateBeforeSettleForEncryptedReasoning(
          event,
          state.encryptedReasoningRecovery,
          hasUsableEncryptedReasoningReplacement,
          state.encryptedRejections,
          resolveRoutedModelApi,
        );
        if (!reasoningRecovery) return undefined;
        if (reasoningRecovery.armNotice) {
          state.pendingEncryptedReasoningNotice = {
            failedRef: reasoningRecovery.failedRef,
          };
        }
        return {
          entries: reasoningRecovery.entries,
          continue: reasoningRecovery.continue,
        };
      }

      // Record the cooldown for the failed model immediately, at decision
      // time, rather than relying on route()'s own transcript scan to catch
      // it when the continuation's route() runs next: that scan only turns
      // a message newer than THIS process's own start time
      // (state.processStartTime) into a cooldown, so a later, separate `pi`
      // invocation that resumes this session would never see a failure that
      // happened entirely within this process's lifetime -- it would retry
      // the same already-failed model. Writing the cooldown to disk here,
      // before that gap matters, is what makes it visible to that next
      // process. Cooldown recording is itself advisory: a write failure must
      // never stop the auto-continue from firing.
      try {
        state.cooldownStore.record(
          decision.failedRef,
          decision.errorMessage,
          'transcript',
          state.currentConfig,
          Date.now(),
          decision.timestamp,
        );
      } catch (error) {
        logAutoContinueErrorOnce(error);
      }

      state.pendingAutoContinueNotice = { failedRef: decision.failedRef };
      return { entries: decision.entries, continue: decision.continue };
    } catch (error) {
      logAutoContinueErrorOnce(error);
      return undefined;
    }
  });

  // Fires exactly once per run, after the `agent_before_settle` loop above
  // has fully stopped requesting continuations -- the only point at which
  // the "never chain" guard may safely reset for a later, genuinely
  // independent run. Also drops any pending auto-continue notice that
  // route() never got to consume (e.g. the continuation was cancelled
  // before its own route() call, or a prior route() threw before reaching
  // the notice), so a later, unrelated run's route() can never fire it.
  pi.on('agent_settled', (_event, ctx) => {
    ensureInitializedFromContext(ctx);
    if (!state.routerEnabled) return;
    try {
      resetAutoContinueGuardOnSettle(state.autoContinueGuard);
      state.pendingAutoContinueNotice = undefined;
      resetEncryptedReasoningRecoveryOnSettle(state.encryptedReasoningRecovery);
      state.pendingEncryptedReasoningNotice = undefined;
    } catch (error) {
      logAutoContinueErrorOnce(error);
    }
  });

  pi.on('thinking_level_select', (event, ctx) => {
    ensureInitializedFromContext(ctx);
    if (!isInitialized || !state.routerEnabled || !state.selectedProfile)
      return;
    if (internalThinkingChangeDepth > 0 || isInternalModelSwitch) return;
    if (
      ctx.model?.provider !== 'router' ||
      ctx.model?.id !== state.selectedProfile
    ) {
      return;
    }

    // User changed pi's thinking level (e.g. via shift+tab).
    // Apply as an all-tier thinking override for the active router profile.
    applyThinkingOverride(
      state.thinkingByProfile,
      state.selectedProfile,
      'all',
      event.level,
    );
    persistState();
    actions.updateStatus(ctx);
    const unsupported = getUnsupportedTiers(
      state.currentConfig.profiles[state.selectedProfile],
      event.level,
      ctx.modelRegistry,
    );
    if (unsupported.length > 0) {
      ctx.ui.notify(
        formatUnsupportedThinkingWarning('all', event.level, unsupported),
        'warning',
      );
    }
  });
};

export default routerExtension;
