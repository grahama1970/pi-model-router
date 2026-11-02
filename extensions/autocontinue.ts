import type { AssistantMessage } from '@earendil-works/pi-ai';
import { isRetryableAssistantError } from '@earendil-works/pi-ai';
import { classifyLimitError, isValidEventTimestamp } from './cooldown';
import { isObjectRecord } from './config';

// Auto-continue is advisory, exactly like extensions/cooldown.ts: it fires at
// most one extra model request after a turn that ends with a non-retryable
// usage-limit error Pi itself chose not to retry, so a session never sits
// stuck on a dead model when another configured model could have answered.
// Every path here must degrade to "do nothing" on any unexpected input or
// failure -- never throw, never block the turn, never chain.

// Tag on the session-persisted entry so it is never mistaken for literal
// user input -- in the session file it is `role: "custom"` with this
// customType, not `role: "user"`; only Pi's own `convertToLlm()` turns it
// into a user-role message for the provider request, the same mechanism
// compaction summaries and other extension-injected content use.
export const AUTO_CONTINUE_CUSTOM_TYPE = 'router-auto-continue';

export interface AutoContinueEntryDraft {
  type: 'custom_message';
  customType: typeof AUTO_CONTINUE_CUSTOM_TYPE;
  content: string;
  display: boolean;
}

export const buildAutoContinueEntry = (): AutoContinueEntryDraft => ({
  type: 'custom_message',
  customType: AUTO_CONTINUE_CUSTOM_TYPE,
  content:
    '[pi-model-router] Your previous response stopped because the model hit its usage limit. Continue where you left off; a different model has been selected automatically.',
  display: false,
});

export interface AutoContinueFailure {
  failedRef: string;
  errorMessage: string;
  // The failed message's own event timestamp, when present and within
  // Date's representable range -- becomes the cooldown entry's
  // `recorded_at` so a later process reading the cooldown file compares
  // against when the failure actually happened, not when it was recorded.
  // `undefined` when missing or malformed; the caller then lets
  // `CooldownStore.record()` default `recordedAt` to its own `now`.
  timestamp: number | undefined;
}

/**
 * A failed turn is eligible for the one-shot auto-continue only when its
 * final assistant message:
 *  - actually failed (`stopReason === "error"`, never "aborted" -- an abort
 *    is a deliberate user action and must never trigger auto-continue);
 *  - carries the provider/model/errorMessage Pi recorded on the message
 *    itself (never inferred from routing state);
 *  - is classified as a rate/quota/usage-limit error by the same
 *    categorizer `extensions/cooldown.ts` uses to record the cooldown --
 *    never a generic transport or server error; and
 *  - is NOT retryable per Pi's own `isRetryableAssistantError` (the exact
 *    predicate `agent-session.js`'s `_isRetryableError` uses to decide
 *    whether to retry across fallback models before ending the turn), so a
 *    retryable 429 Pi's own retry/fallback chain already owns is left
 *    completely untouched.
 */
export const classifyAutoContinueFailure = (
  message: unknown,
): AutoContinueFailure | undefined => {
  if (!isObjectRecord(message)) return undefined;
  if (message.role !== 'assistant' || message.stopReason !== 'error') {
    return undefined;
  }
  const { errorMessage, provider, model, timestamp } = message;
  if (typeof errorMessage !== 'string' || errorMessage === '') {
    return undefined;
  }
  if (typeof provider !== 'string' || provider === '') return undefined;
  if (typeof model !== 'string' || model === '') return undefined;
  if (!classifyLimitError(errorMessage)) return undefined;

  // isRetryableAssistantError only reads stopReason/errorMessage; the rest of
  // AssistantMessage's shape is irrelevant to it, so this minimal cast calls
  // the exact same predicate Pi's own agent-level retry uses, never a
  // reimplementation of it.
  const retryable = isRetryableAssistantError({
    stopReason: 'error',
    errorMessage,
  } as unknown as AssistantMessage);
  if (retryable) return undefined;

  const validTimestamp =
    typeof timestamp === 'number' && isValidEventTimestamp(timestamp)
      ? timestamp
      : undefined;

  return {
    failedRef: `${provider}/${model}`,
    errorMessage,
    timestamp: validTimestamp,
  };
};

/**
 * Per-chain and per-(provider/model, cooldown period) gating, kept separate
 * from `classifyAutoContinueFailure` so each guard is testable on its own.
 *
 * `chainArmed` enforces "never chain": once an auto-continue has fired for
 * the current unbroken run of consecutive failed turns, no later
 * `agent_before_settle` error in that same run -- even one that fails on a
 * different model -- fires another one. Only Pi's actual settlement
 * (`agent_settled`, dispatched exactly once per run, in a `finally` block,
 * after the `agent_before_settle` loop has fully stopped requesting
 * continuations -- see `resetAutoContinueGuardOnSettle`) clears it. A
 * non-error `agent_before_settle` call must NOT clear it: Pi's run loop
 * (`agent-session.js`'s `while (!this._agentRunAbortRequested) { ... }`)
 * can dispatch `agent_before_settle` more than once within the SAME run --
 * a continuation that itself succeeds, followed by a later turn in that
 * same run that fails on a different model -- and clearing on that
 * in-between success would let the second failure fire again.
 *
 * `stillCoolingFromEarlierFailure` + `alreadyFiredForActiveModel` enforce "at
 * most once per provider/model per cooldown period": a model already
 * auto-continued for while its cooldown is still active is skipped silently;
 * once that cooldown expires or clears, a later failure of the same model
 * gets a fresh one-shot budget.
 */
export const shouldFireAutoContinue = (input: {
  chainArmed: boolean;
  stillCoolingFromEarlierFailure: boolean;
  alreadyFiredForActiveModel: boolean;
}): boolean => {
  if (input.chainArmed) return false;
  if (
    input.stillCoolingFromEarlierFailure &&
    input.alreadyFiredForActiveModel
  ) {
    return false;
  }
  return true;
};

export interface AutoContinueGuardState {
  chainArmed: boolean;
  firedModels: Set<string>;
}

export const createAutoContinueGuardState = (): AutoContinueGuardState => ({
  chainArmed: false,
  firedModels: new Set<string>(),
});

export interface AutoContinueResult {
  failedRef: string;
  // Forwarded so the caller (extensions/index.ts) can record a cooldown for
  // the failed model immediately, at decision time -- see
  // `evaluateBeforeSettleForAutoContinue` below for why this cannot wait for
  // route()'s own transcript scan on the continuation.
  errorMessage: string;
  timestamp: number | undefined;
  entries: AutoContinueEntryDraft[];
  continue: true;
}

/**
 * Full `agent_before_settle` decision: extracts `outcome` and the last
 * message of `event.context.contextMessages` defensively from an untyped
 * event, applies every guard above, and mutates `guard` in place (the same
 * session-scoped, non-persisted pattern `processedTranscriptFailures` uses
 * in extensions/index.ts) when it decides to fire. Returns `undefined` for
 * every case that must stay silent: a successful turn, an abort, a
 * retryable error, a non-limit error, or a guard rejection.
 *
 * This boundary -- not `turn_end` -- is where the decision must actually
 * fire. Pi's `finishTurn` dispatches `turn_end`, then checks its own
 * retry/fallback decision; when that decision is `action: "end"` (always
 * true for a non-retryable error, since Pi itself already gave up), it
 * discards whatever `turn_end` returned:
 * `agent-session.js`'s `_installAgentBoundaryHooks()` --
 * `if (previousDecision?.action === "end") return previousDecision;` --
 * runs unconditionally before the `extensionContinue` from `turn_end` is
 * even consulted. `agent_before_settle` is dispatched separately, by
 * `_runBeforeSettleBoundary()`, right before the run loop would actually
 * stop, independent of that discarded decision, so a `continue: true`
 * returned here is honored unconditionally (as long as the resulting
 * context can still continue).
 */
export const evaluateBeforeSettleForAutoContinue = (
  event: unknown,
  guard: AutoContinueGuardState,
  isStillCooling: (failedRef: string) => boolean,
  hasUsableReplacement: (failedRef: string) => boolean,
): AutoContinueResult | undefined => {
  if (!isObjectRecord(event)) return undefined;
  const outcome = typeof event.outcome === 'string' ? event.outcome : undefined;
  if (outcome !== 'error') {
    // A successful or aborted `agent_before_settle` call does not end the
    // "never chain" guard's lifetime -- see `resetAutoContinueGuardOnSettle`
    // for where it actually clears.
    return undefined;
  }

  const context = isObjectRecord(event.context) ? event.context : undefined;
  const contextMessages = context ? context.contextMessages : undefined;
  const lastMessage =
    Array.isArray(contextMessages) && contextMessages.length > 0
      ? contextMessages[contextMessages.length - 1]
      : undefined;

  const failure = classifyAutoContinueFailure(lastMessage);
  if (!failure) return undefined;

  const fire = shouldFireAutoContinue({
    chainArmed: guard.chainArmed,
    stillCoolingFromEarlierFailure: isStillCooling(failure.failedRef),
    alreadyFiredForActiveModel: guard.firedModels.has(failure.failedRef),
  });
  if (!fire) return undefined;

  // Firing without an actual replacement just re-requests the same failed
  // model (or another unusable/still-cooling one): the continuation would
  // only hit the identical limit error again, burning the one-shot budget
  // for nothing visible to change. Require at least one other usable
  // (resolvable + authenticated), non-cooling candidate in the current
  // profile/tier before committing; leave the error visible otherwise.
  if (!hasUsableReplacement(failure.failedRef)) return undefined;

  guard.chainArmed = true;
  guard.firedModels.add(failure.failedRef);

  return {
    failedRef: failure.failedRef,
    errorMessage: failure.errorMessage,
    timestamp: failure.timestamp,
    entries: [buildAutoContinueEntry()],
    continue: true,
  };
};

/**
 * Clears the per-run "never chain" guard on Pi's actual settlement. Must be
 * wired to the `agent_settled` event (not `agent_before_settle`'s non-error
 * outcome -- see the doc comment above `shouldFireAutoContinue`): Pi's run
 * loop dispatches `agent_settled` from `_emitAgentSettled()` in a `finally`
 * block exactly once, only after the `agent_before_settle` loop has fully
 * stopped requesting continuations, so this is the first point at which a
 * later, genuinely independent run may safely get its own fresh budget.
 * `firedModels` is untouched here: it survives for the life of the session,
 * gated separately by cooldown state in `shouldFireAutoContinue`.
 */
export const resetAutoContinueGuardOnSettle = (
  guard: AutoContinueGuardState,
): void => {
  guard.chainArmed = false;
};

const loggedAutoContinueErrors = new Set<string>();

export const logAutoContinueErrorOnce = (error: unknown): void => {
  const message = `pi-model-router: auto-continue failed: ${String(error)}`;
  if (loggedAutoContinueErrors.has(message)) return;
  loggedAutoContinueErrors.add(message);
  console.error(message);
};
