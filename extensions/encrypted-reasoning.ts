import { createHash } from 'node:crypto';
import { isObjectRecord } from './config';

// Encrypted-reasoning recovery: the second, independent `agent_before_settle`
// recovery path next to extensions/autocontinue.ts (which owns usage limits).
//
// Codex (and other OpenAI Responses transports) answer with a reasoning item
// whose opaque payload Pi stores in the thinking block's `thinkingSignature`.
// Pi's own `transform-messages.js` drops that block when the request targets a
// different provider/API/model, but keeps it -- correctly, it is required for
// replay -- when the request goes back to the same model that produced it. If
// the provider then refuses its own payload ("invalid/undecryptable encrypted
// content"), every later request to that model replays the identical rejected
// payload, so retrying the same model can never succeed. The only recovery is
// the one a user performs by hand: switch to a different model and continue.
//
// This module holds that decision as pure functions. It deliberately does not
// rewrite history or strip signatures from the session (transform-messages and
// the provider transports own message conversion); it only (a) excludes the
// physical model that rejected the payload from this conversation's routing
// candidates, and (b) requests one more model request through Pi's
// `agent_before_settle` boundary, exactly as the usage-limit path does.
//
// The exclusion has two independent sources, because the rejecting assistant
// message alone is not a durable record of what was rejected:
//
//  1. Durable, signature-bound metadata: when a rejection is observed, the
//     rejected model is bound to fingerprints of the opaque signed history
//     present at that moment and persisted as a session custom entry
//     (`ENCRYPTED_REJECTION_ENTRY_TYPE`). The exclusion is active exactly
//     while one of those fingerprints is still in the request, which keeps it
//     across restarts and across branch summaries that move the rejecting
//     error off the active path while keeping the signed response, and retires
//     it once the implicated signed history is gone.
//  2. The rejection messages Pi surfaces per request -- persisted assistant
//     errors in the projection AND `request.failed` (a retryable failure need
//     never be in the projection). This is the conservative fallback for a
//     rejection that has no metadata yet (e.g. another process's failure seen
//     only through its projected error).
//
// Only a digest of a signature is ever persisted (`signatureFingerprint`),
// never the opaque payload itself or anything derived from it beyond the
// digest.
//
// Advisory like cooldowns and auto-continue: every exported function returns
// undefined/empty on unexpected input and never throws.

export const ENCRYPTED_REASONING_CUSTOM_TYPE =
  'router-encrypted-reasoning-recovery';

// Session custom-entry type for durable rejection metadata. Pi's
// `pi.appendEntry()` writes it on the active branch, and
// `ctx.sessionManager.getEntries()` reads every entry of the session, so a
// branch summary that abandons the error entry does not erase the record.
export const ENCRYPTED_REJECTION_ENTRY_TYPE = 'router-encrypted-rejection';

export interface EncryptedReasoningEntryDraft {
  type: 'custom_message';
  customType: typeof ENCRYPTED_REASONING_CUSTOM_TYPE;
  content: string;
  display: boolean;
}

export const buildEncryptedReasoningEntry =
  (): EncryptedReasoningEntryDraft => ({
    type: 'custom_message',
    customType: ENCRYPTED_REASONING_CUSTOM_TYPE,
    content:
      '[pi-model-router] The previous model rejected the encrypted reasoning this conversation replays. Continue where you left off; a different model has been selected automatically.',
    display: false,
  });

// Both patterns must match, which is what keeps this classifier narrow: a
// quota error ("insufficient_quota"), a transport error, or a generic
// request-validation error never mentions encrypted/decrypted reasoning
// content or a thinking signature, so it stays with the path that owns it.
//
// The observed shape (OpenAI Responses and compatible gateways) is a 400
// whose body names the offending input item, e.g.
// `400: {"error":{"message":"Invalid 'input[3].encrypted_content': Encrypted
// content could not be verified.","code":"invalid_encrypted_content"}}`.
// A retryable wrapper around the same rejection (e.g. a gateway prefixing
// `500:` or `service unavailable`) still matches and is handled here too --
// Pi's own retry cannot help, because the replay is deterministic.
const ENCRYPTED_REASONING_SUBJECT_PATTERN =
  /encrypted[\s_-]*(?:reasoning|content)|reasoning[\s_-]*content|thinking[\s_-]*signature|decrypt/i;
const ENCRYPTED_REASONING_REJECTION_PATTERN =
  /invalid|malformed|corrupt|could\s*not|cannot|can't|unable|mismatch|not\s+valid|fail|reject|expire|verif/i;

/**
 * True only for a provider response that rejects a *replayed encrypted
 * reasoning payload* (or the thinking signature that carries it). Requires
 * both the subject (encrypted/decrypted reasoning content, thinking
 * signature) and a rejection term, so unrelated 400s stay untouched.
 */
export const isEncryptedReasoningReplayError = (text: string): boolean =>
  typeof text === 'string' &&
  text.length > 0 &&
  ENCRYPTED_REASONING_SUBJECT_PATTERN.test(text) &&
  ENCRYPTED_REASONING_REJECTION_PATTERN.test(text);

export interface EncryptedReasoningFailure {
  failedRef: string;
  provider: string;
  modelId: string;
  // The API the failing request was made with. Pi's `transform-messages.js`
  // replays a signed thinking block only for the SAME provider + API + model
  // (`isSameModel = provider === model.provider && api === model.api &&
  // model === model.id`), so API belongs to the replay identity exactly like
  // provider and model do.
  api?: string;
  errorMessage: string;
  // The failed message's own event time: the STABLE identity of this failure,
  // independent of when it is observed or which signatures exist later.
  timestamp?: number;
}

/**
 * A failed message is eligible only when it is an assistant error (never an
 * abort: `stopReason` must be exactly `"error"`), carries the
 * provider/model/errorMessage Pi recorded on the message itself, and that
 * errorMessage is an encrypted-reasoning replay rejection.
 *
 * There is no `isRetryableAssistantError` gate: a rejected signature is
 * deterministic for the model that produced it, so a retryable-looking
 * wrapper around the same rejection is still unrecoverable on that model.
 * `extensions/provider.ts` applies this same classifier to `request.failed`,
 * which is how a retryable rejection reaches routing before Pi's retry
 * loop runs again.
 */
export const classifyEncryptedReasoningFailure = (
  message: unknown,
): EncryptedReasoningFailure | undefined => {
  if (!isObjectRecord(message)) return undefined;
  if (message.role !== 'assistant' || message.stopReason !== 'error') {
    return undefined;
  }
  const { errorMessage, provider, model } = message;
  if (
    typeof errorMessage !== 'string' ||
    !isEncryptedReasoningReplayError(errorMessage)
  ) {
    return undefined;
  }
  if (typeof provider !== 'string' || provider === '') return undefined;
  if (typeof model !== 'string' || model === '') return undefined;
  const api =
    typeof message.api === 'string' && message.api !== ''
      ? message.api
      : undefined;
  const timestamp =
    typeof message.timestamp === 'number' && Number.isFinite(message.timestamp)
      ? message.timestamp
      : undefined;
  return {
    failedRef: `${provider}/${model}`,
    provider,
    modelId: model,
    api,
    errorMessage,
    timestamp,
  };
};

/**
 * Opaque fingerprint of one signed thinking block: `sha256` over the
 * producing model and the signature string, truncated to 16 hex chars.
 *
 * The signature itself is never stored or logged -- only this digest -- so
 * session metadata never carries the provider's encrypted payload. Two
 * blocks that carry the same replayable payload necessarily share the same
 * digest, which is what lets the exclusion survive a branch summary or a
 * restart while retiring once that payload leaves the conversation.
 */
export const signatureFingerprint = (
  provider: unknown,
  api: string,
  model: unknown,
  signature: string,
): string =>
  createHash('sha256')
    .update(
      `${String(provider)}\u0000${api}\u0000${String(model)}\u0000${signature}`,
    )
    .digest('hex')
    .slice(0, 16);

/** The replay identity Pi itself uses: provider + API + model. */
export interface ReplayScope {
  provider: string;
  modelId: string;
  // Part of the replay identity, exactly as in Pi's `isSameModel`. Never
  // optional: a missing API is missing identity, not a wildcard.
  api: string;
}

/**
 * Fingerprints of every signed thinking block in this message list, sorted and
 * deduplicated so persisted metadata is stable across restarts. Only
 * `thinkingSignature` blocks carry a replayable payload; text/tool signatures
 * are not reasoning and are ignored.
 */
export const signedHistoryFingerprints = (
  messages: readonly unknown[],
  scope?: ReplayScope,
): string[] => {
  const fingerprints = new Set<string>();
  for (const message of messages) {
    if (!isObjectRecord(message) || message.role !== 'assistant') continue;
    // An errored or aborted assistant turn carries no replayable payload.
    if (message.stopReason === 'error' || message.stopReason === 'aborted') {
      continue;
    }
    // Pi's `transform-messages.js` (isSameModel) replays a signed thinking block
    // only when the request targets the SAME provider, API and model that
    // produced it; every other block is dropped before the request is sent, so
    // it can never be the payload a model rejects.
    // Without an explicit API there is no replay identity to fingerprint, so
    // an API-less message never contributes a fingerprint (never a wildcard).
    const api =
      typeof message.api === 'string' && message.api !== ''
        ? message.api
        : undefined;
    if (api === undefined) continue;
    if (scope) {
      if (message.provider !== scope.provider) continue;
      if (message.model !== scope.modelId) continue;
      if (api !== scope.api) continue;
    }
    if (!Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (!isObjectRecord(block) || block.type !== 'thinking') continue;
      const signature = block.thinkingSignature;
      if (typeof signature !== 'string' || signature.length === 0) continue;
      fingerprints.add(
        signatureFingerprint(message.provider, api, message.model, signature),
      );
    }
  }
  return [...fingerprints].sort();
};

/** Splits a canonical `provider/model` ref; the model id may contain `/`. */
export const parseModelRef = (
  ref: string,
): { provider: string; modelId: string } => {
  const index = ref.indexOf('/');
  if (index <= 0) return { provider: '', modelId: '' };
  return { provider: ref.slice(0, index), modelId: ref.slice(index + 1) };
};

/**
 * Durable rejection metadata: which model rejected the payload, and which
 * signed history was present when it did.
 */
export interface EncryptedRejectionRecord {
  customType: typeof ENCRYPTED_REJECTION_ENTRY_TYPE;
  failedRef: string;
  // API of the failing call: part of the replay identity. Records without it
  // are rejected by `isEncryptedRejectionRecord` (no wildcard, no migration).
  api: string;
  fingerprints: string[];
  // When the rejection was observed.
  at: number;
  // The failed message's own event time: stable identity of the failure, never
  // rewritten when the same failure is observed again with a different payload.
  failureAt?: number;
}

export const isEncryptedRejectionRecord = (
  value: unknown,
): value is EncryptedRejectionRecord => {
  if (!isObjectRecord(value)) return false;
  if (value.customType !== ENCRYPTED_REJECTION_ENTRY_TYPE) return false;
  const { failedRef, fingerprints, at, api, failureAt } = value;
  if (typeof failedRef !== 'string' || failedRef === '') return false;
  if (typeof at !== 'number' || !Number.isFinite(at)) return false;
  // Missing identity is not a wildcard: a record without an explicit API is
  // ignored outright.
  if (typeof api !== 'string' || api === '') return false;
  if (
    failureAt !== undefined &&
    (typeof failureAt !== 'number' || !Number.isFinite(failureAt))
  ) {
    return false;
  }
  return (
    Array.isArray(fingerprints) &&
    fingerprints.every(
      (fingerprint) =>
        typeof fingerprint === 'string' && /^[0-9a-f]{16}$/.test(fingerprint),
    )
  );
};

/**
 * Builds the metadata record for an observed rejection.
 *
 * Bound to *every* signed history block present at failure time, because the
 * exact rejected item is only named inside the provider's error body, which
 * is not a reliable identifier across gateways. That is deliberately the
 * conservative direction: the exclusion stays active while any implicated
 * payload could still be replayed.
 *
 * Returns undefined when the conversation carries no signed history at all:
 * with nothing identifiable to bind to, a durable exclusion would be a
 * blanket per-model ban with no evidence, so the caller keeps only the
 * per-request rejection-message exclusion (documented boundary in
 * docs/ARCHITECTURE.md). It is never a global cooldown either way.
 */
export const buildEncryptedRejectionRecord = (
  failure: EncryptedReasoningFailure,
  messages: readonly unknown[],
  at: number,
): EncryptedRejectionRecord | undefined => {
  // Only signed blocks the rejecting call itself could have replayed count:
  // same provider, same API, same model -- exactly Pi's `isSameModel`. Pi drops
  // every other block before sending, so binding those would keep this target
  // excluded after its own payload is gone, and would misread another model's
  // (or another API's) payload as the rejected one.
  const { provider, modelId } = parseModelRef(failure.failedRef);
  if (modelId === '') return undefined;
  // No API, no replay identity: never create durable metadata from an API-less
  // failure (and never downgrade it to a provider+model wildcard).
  if (failure.api === undefined || failure.api === '') return undefined;
  const fingerprints = signedHistoryFingerprints(messages, {
    provider,
    modelId,
    api: failure.api,
  });
  if (fingerprints.length === 0) return undefined;
  const record: EncryptedRejectionRecord = {
    customType: ENCRYPTED_REJECTION_ENTRY_TYPE,
    failedRef: failure.failedRef,
    api: failure.api,
    fingerprints,
    at,
  };
  if (failure.timestamp !== undefined) record.failureAt = failure.timestamp;
  return record;
};

/**
 * Identity of one recorded failure. When the failure's own event time is known
 * it is the identity -- observing the same failure again (even after new
 * signatures appear) must never create a second record bound to a different
 * payload. Without an event time the bound payload set is the identity, which
 * is what allows a genuinely NEW payload of the same model to be recorded.
 */
export const rejectionRecordKey = (
  record: Pick<
    EncryptedRejectionRecord,
    'failedRef' | 'fingerprints' | 'api' | 'failureAt'
  >,
): string => {
  const head = `${record.failedRef}\u0000${record.api ?? ''}`;
  if (record.failureAt !== undefined)
    return `${head}\u0000#${record.failureAt}`;
  return `${head}\u0000${[...record.fingerprints].sort().join(',')}`;
};

/**
 * The exclusion set for one request.
 *
 * Three sources, each with its own rule -- deduplicating by model alone is
 * what previously let an OLD record for model A suppress a NEW rejection of A,
 * because A's fresh failure looked "already recorded":
 *
 * - Durable records: a record excludes its model while ANY of its bound
 *   fingerprints is still present in this request. Once the payload is gone
 *   the record retires (by evidence, not because the error message vanished).
 * - `request.failed`: a FRESH failure Pi hands over out-of-band (retry path).
 *   Always honored, whatever records exist for that model -- the model just
 *   rejected THIS request, so re-selecting it would replay the same payload.
 * - Projected assistant errors: a message is either the failure an existing
 *   record already describes (stale: its timestamp is not newer than that
 *   record, so the fingerprint rule above governs and retirement still works)
 *   or a NEW failure of the same model (newer than every record for it), which
 *   is excluded even when an older, retired record exists.
 */
export const collectEncryptedReasoningExclusions = (input: {
  messages: readonly unknown[];
  failed?: unknown;
  records?: readonly EncryptedRejectionRecord[];
  // Resolves the API the registered target for a ref currently speaks: the API
  // half of the replay identity. When it is absent (pure callers/tests) or the
  // target cannot be resolved, nothing applies -- an unknown API is never
  // treated as a wildcard.
  resolveModelApi?: (failedRef: string) => string | undefined;
}): Set<string> => {
  const excluded = new Set<string>();
  const records = (input.records ?? []).filter(isEncryptedRejectionRecord);
  const presentByApi = new Map<string, Set<string>>();
  const presentFor = (api: string): Set<string> => {
    const key = api;
    const cached = presentByApi.get(key);
    if (cached) return cached;
    const fingerprints = new Set<string>();
    for (const message of input.messages) {
      if (!isObjectRecord(message) || message.role !== 'assistant') continue;
      if (message.stopReason === 'error' || message.stopReason === 'aborted') {
        continue;
      }
      if (message.api !== api) continue;
      if (!Array.isArray(message.content)) continue;
      for (const block of message.content) {
        if (!isObjectRecord(block) || block.type !== 'thinking') continue;
        const signature = block.thinkingSignature;
        if (typeof signature !== 'string' || signature.length === 0) continue;
        fingerprints.add(
          signatureFingerprint(message.provider, api, message.model, signature),
        );
      }
    }
    presentByApi.set(key, fingerprints);
    return fingerprints;
  };

  // A record applies only to a target that still speaks the API it was
  // recorded under (the replay identity includes the API).
  const recordAppliesToTarget = (record: EncryptedRejectionRecord): boolean => {
    const targetApi = input.resolveModelApi?.(record.failedRef);
    return targetApi !== undefined && targetApi === record.api;
  };

  // 1. Payload-bound records: active only while a bound payload of the SAME
  //    replay identity (provider + API + model) is present.
  for (const record of records) {
    if (record.fingerprints.length === 0) continue;
    if (!recordAppliesToTarget(record)) continue;
    const present = presentFor(record.api);
    if (record.fingerprints.some((fingerprint) => present.has(fingerprint))) {
      excluded.add(record.failedRef);
    }
  }

  // 2. Fresh out-of-band failure (retry path): always honored, never gated by
  //    an older record for the same model.
  const failedFailure = classifyEncryptedReasoningFailure(input.failed);
  if (
    failedFailure &&
    failureAppliesToTarget(failedFailure, input.resolveModelApi)
  ) {
    excluded.add(failedFailure.failedRef);
  }

  // 3. Projected assistant errors: a message an existing record already
  //    describes defers to that record (retirement still works), a newer
  //    message is a NEW failure and is excluded.
  for (const message of input.messages) {
    const failure = classifyEncryptedReasoningFailure(message);
    if (!failure) continue;
    if (!failureAppliesToTarget(failure, input.resolveModelApi)) continue;
    const recordPayloadPresent = records
      .filter((record) => record.failedRef === failure.failedRef)
      .filter(recordAppliesToTarget)
      .some((record) =>
        record.fingerprints.some((fingerprint) =>
          presentFor(record.api).has(fingerprint),
        ),
      );
    if (recordPayloadPresent) {
      excluded.add(failure.failedRef);
      continue;
    }
    if (isStaleFailureForRecord(failure, records)) continue;
    excluded.add(failure.failedRef);
  }

  return excluded;
};

/**
 * A failure itself came from a specific API; when the registered target now
 * speaks a different one, the replayed signature would not even be sent there.
 */
const failureAppliesToTarget = (
  failure: EncryptedReasoningFailure,
  resolveModelApi?: (failedRef: string) => string | undefined,
): boolean => {
  // Missing identity is never a wildcard: an API-less failure applies to no
  // target, and an unresolvable target API matches nothing.
  if (failure.api === undefined || failure.api === '') return false;
  const targetApi = resolveModelApi?.(failure.failedRef);
  return targetApi !== undefined && targetApi === failure.api;
};

/**
 * True when this projected error is the failure an existing record for the
 * same model already describes. Identity first: the failed message's own event
 * time against the record's recorded failure time. Fallback for messages or
 * records without an event time: a message not newer than the record is the
 * recorded observation. A message newer than every record for the model is a
 * NEW failure and must be honored on its own.
 */
const isStaleFailureForRecord = (
  failure: EncryptedReasoningFailure,
  records: readonly EncryptedRejectionRecord[],
): boolean => {
  const matching = records.filter(
    (record) => record.failedRef === failure.failedRef,
  );
  if (matching.length === 0) return false;
  if (failure.timestamp === undefined) return true;
  return matching.some(
    (record) => (record.failureAt ?? record.at) >= failure.timestamp!,
  );
};

/**
 * Rejection messages visible in this request whose exact failure (model AND
 * bound payload set) has no record yet. `extensions/provider.ts` persists
 * these, so a rejection observed only through `request.failed` (retryable
 * wrapper) or through another process's projected error becomes
 * signature-bound metadata for later requests. A model that rejects a NEW
 * payload therefore gets a second record instead of being treated as already
 * known.
 */
export const collectUnrecordedEncryptedRejections = (input: {
  messages: readonly unknown[];
  failed?: unknown;
  records?: readonly EncryptedRejectionRecord[];
  at: number;
}): EncryptedRejectionRecord[] => {
  const records = (input.records ?? []).filter(isEncryptedRejectionRecord);
  const observations: Array<{
    failure: EncryptedReasoningFailure;
    // History that PRECEDES the failure. A signature produced after it can
    // never have been replayed to it, so it must never be bound.
    preceding: readonly unknown[];
    fresh: boolean;
  }> = [];
  input.messages.forEach((message, index) => {
    const failure = classifyEncryptedReasoningFailure(message);
    if (!failure) return;
    observations.push({
      failure,
      preceding: input.messages.slice(0, index),
      fresh: false,
    });
  });
  const failedFailure = classifyEncryptedReasoningFailure(input.failed);
  if (failedFailure) {
    // `request.failed` is a fresh failure of THIS request; there is no later
    // history yet, so the whole request history precedes it.
    observations.push({
      failure: failedFailure,
      preceding: input.messages,
      fresh: true,
    });
  }

  const emitted: EncryptedRejectionRecord[] = [];
  const seenKeys = new Set(records.map((record) => rejectionRecordKey(record)));
  for (const { failure, preceding, fresh } of observations) {
    // A stale projected error is an already-recorded failure seen again; it
    // must never fabricate a ban, whatever signatures appeared since.
    if (!fresh && isStaleFailureForRecord(failure, [...records, ...emitted])) {
      continue;
    }
    const record = buildEncryptedRejectionRecord(failure, preceding, input.at);
    // No identifiable signed history: no durable record (documented boundary),
    // the per-request exclusion still applies.
    if (!record) continue;
    const key = rejectionRecordKey(record);
    // Same observed failure (stable event time) or same payload identity: never
    // re-record it with a different payload set.
    if (seenKeys.has(key)) continue;
    seenKeys.add(key);
    emitted.push(record);
  }
  return emitted;
};

/**
 * The rejection Pi's `agent_before_settle` boundary just observed, with its
 * durable record (undefined when the conversation carries no replayable signed
 * history). Called BEFORE any continuation decision so the metadata is
 * persisted even when there is no replacement left and the run is about to
 * stop.
 */
export const observeEncryptedReasoningFailure = (input: {
  messages: readonly unknown[];
  at: number;
}): { failedRef: string; record?: EncryptedRejectionRecord } | undefined => {
  if (input.messages.length === 0) return undefined;
  const failure = classifyEncryptedReasoningFailure(
    input.messages[input.messages.length - 1],
  );
  if (!failure) return undefined;
  return {
    failedRef: failure.failedRef,
    record: buildEncryptedRejectionRecord(failure, input.messages, input.at),
  };
};

/**
 * Whether this conversation carries an image part. Needed at decision time
 * because the replacement must be able to replay the whole conversation, and
 * `agent_before_settle` hands over raw session-projection messages rather
 * than a pi-ai `Context`, so routing.ts's `hasImageAttachment()` (typed on
 * `Context`) cannot be reused here.
 */
export const conversationHasImage = (messages: readonly unknown[]): boolean =>
  messages.some(
    (message) =>
      isObjectRecord(message) &&
      Array.isArray(message.content) &&
      message.content.some(
        (part) => isObjectRecord(part) && part.type === 'image',
      ),
  );

export interface EncryptedReasoningRecoveryState {
  // Failed refs this episode already continued away from. Together with the
  // conversation's exclusion set this bounds the recovery to distinct
  // remaining candidates: a model that already failed is excluded from
  // routing, and can never be the ref that fires twice.
  attemptedRefs: Set<string>;
  // One user-visible notice per recovery episode, no matter how many
  // distinct models the episode walks through. Cleared only by
  // `resetEncryptedReasoningRecoveryOnSettle`.
  noticeArmed: boolean;
}

export const createEncryptedReasoningRecoveryState =
  (): EncryptedReasoningRecoveryState => ({
    attemptedRefs: new Set<string>(),
    noticeArmed: false,
  });

export interface EncryptedReasoningRecoveryResult {
  failedRef: string;
  entries: EncryptedReasoningEntryDraft[];
  continue: true;
  // True for the first fire of this episode only; the caller arms the
  // pending notice with it and route() fires that notice once when the
  // replacement model is actually known.
  armNotice: boolean;
}

/**
 * Full `agent_before_settle` decision for the encrypted-reasoning path.
 * Mirrors extensions/autocontinue.ts's `evaluateBeforeSettleForAutoContinue`:
 * reads the untyped event defensively, mutates `state` in place only when it
 * decides to fire, and returns `undefined` for every case that must stay
 * silent (a successful or aborted run, an unrelated error, a model already
 * replaced in this episode, or no remaining eligible model).
 *
 * `hasUsableReplacement` is supplied by the caller (extensions/index.ts) and
 * must answer with the exact candidate calculation `route()` uses
 * (`selectRoutedCandidates` in extensions/provider.ts), so the decision can
 * never fire for a continuation that routing would then fail to serve.
 * Requiring a replacement is what makes exhaustion stop instead of continuing
 * into certain failure: once every candidate is excluded, no continuation is
 * requested and the run stops on the provider error.
 */
export const evaluateBeforeSettleForEncryptedReasoning = (
  event: unknown,
  state: EncryptedReasoningRecoveryState,
  hasUsableReplacement: (
    failedRef: string,
    excludedRefs: ReadonlySet<string>,
    contextMessages: readonly unknown[],
  ) => boolean,
  records: readonly EncryptedRejectionRecord[] = [],
  resolveModelApi?: (failedRef: string) => string | undefined,
): EncryptedReasoningRecoveryResult | undefined => {
  if (!isObjectRecord(event)) return undefined;
  if (event.outcome !== 'error') return undefined;

  const context = isObjectRecord(event.context) ? event.context : undefined;
  const contextMessages = context ? context.contextMessages : undefined;
  if (!Array.isArray(contextMessages) || contextMessages.length === 0) {
    return undefined;
  }
  const failure = classifyEncryptedReasoningFailure(
    contextMessages[contextMessages.length - 1],
  );
  if (!failure) return undefined;
  if (state.attemptedRefs.has(failure.failedRef)) return undefined;

  const excludedRefs = collectEncryptedReasoningExclusions({
    messages: contextMessages,
    records,
    resolveModelApi,
  });
  if (!hasUsableReplacement(failure.failedRef, excludedRefs, contextMessages)) {
    return undefined;
  }

  state.attemptedRefs.add(failure.failedRef);
  const armNotice = !state.noticeArmed;
  state.noticeArmed = true;

  return {
    failedRef: failure.failedRef,
    entries: [buildEncryptedReasoningEntry()],
    continue: true,
    armNotice,
  };
};

/**
 * Clears the episode on Pi's actual settlement (`agent_settled`, dispatched
 * once per run after the `agent_before_settle` loop stopped requesting
 * continuations) so a later, genuinely independent run gets its own fresh
 * budget and its own single notice. Durable rejection records are session
 * state, not episode state, and are untouched.
 */
export const resetEncryptedReasoningRecoveryOnSettle = (
  state: EncryptedReasoningRecoveryState,
): void => {
  state.attemptedRefs.clear();
  state.noticeArmed = false;
};
