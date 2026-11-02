import { describe, it, expect } from 'vitest';
import {
  AUTO_CONTINUE_CUSTOM_TYPE,
  buildAutoContinueEntry,
  classifyAutoContinueFailure,
  createAutoContinueGuardState,
  evaluateBeforeSettleForAutoContinue,
  resetAutoContinueGuardOnSettle,
  shouldFireAutoContinue,
} from './autocontinue';

const assistantError = (overrides: Record<string, unknown> = {}) => ({
  role: 'assistant',
  stopReason: 'error',
  provider: 'openai',
  model: 'gpt-4o-mini',
  errorMessage: 'insufficient_quota',
  ...overrides,
});

// Mirrors the real `AgentBeforeSettleEvent` shape
// (`event.context.contextMessages`), narrowed to only the field
// `evaluateBeforeSettleForAutoContinue` reads.
const beforeSettleEvent = (outcome: string, contextMessages: unknown[]) => ({
  outcome,
  context: { contextMessages },
});

describe('autocontinue.ts', () => {
  describe('classifyAutoContinueFailure', () => {
    it('classifies a non-retryable usage-limit error as eligible', () => {
      const result = classifyAutoContinueFailure(assistantError());
      expect(result).toEqual({
        failedRef: 'openai/gpt-4o-mini',
        errorMessage: 'insufficient_quota',
        timestamp: undefined,
      });
    });

    it('carries a valid message timestamp through for the cooldown recordedAt', () => {
      const result = classifyAutoContinueFailure(
        assistantError({ timestamp: 1700000000000 }),
      );
      expect(result?.timestamp).toBe(1700000000000);
    });

    it('drops an out-of-range timestamp rather than forwarding a malformed one', () => {
      const result = classifyAutoContinueFailure(
        assistantError({ timestamp: Number.POSITIVE_INFINITY }),
      );
      expect(result?.timestamp).toBeUndefined();
    });

    // The regression this guards: without the isRetryableAssistantError check,
    // every 429 (most of which Pi's own retry/fallback chain already handles)
    // would also trigger auto-continue, double-handling errors Pi already owns.
    it('does not classify a retryable 429 as eligible, even though it is limit-classified', () => {
      expect(
        classifyAutoContinueFailure(
          assistantError({ errorMessage: '429 too many requests' }),
        ),
      ).toBeUndefined();
    });

    it('does not classify a non-limit error as eligible', () => {
      expect(
        classifyAutoContinueFailure(
          assistantError({ errorMessage: '500 internal server error' }),
        ),
      ).toBeUndefined();
    });

    // The regression this guards: without the stopReason check, an aborted
    // turn (role assistant, no error) could be misread as a failure.
    it('never classifies an aborted turn as eligible', () => {
      expect(
        classifyAutoContinueFailure(
          assistantError({ stopReason: 'aborted', errorMessage: undefined }),
        ),
      ).toBeUndefined();
    });

    it('does not classify a successful turn as eligible', () => {
      expect(
        classifyAutoContinueFailure(
          assistantError({ stopReason: 'stop', errorMessage: undefined }),
        ),
      ).toBeUndefined();
    });

    it('does not classify a non-assistant message as eligible', () => {
      expect(
        classifyAutoContinueFailure(assistantError({ role: 'toolResult' })),
      ).toBeUndefined();
    });

    it('does not classify a message missing provider or model as eligible', () => {
      expect(
        classifyAutoContinueFailure(assistantError({ provider: undefined })),
      ).toBeUndefined();
      expect(
        classifyAutoContinueFailure(assistantError({ model: undefined })),
      ).toBeUndefined();
    });

    it('does not throw on non-object input', () => {
      expect(classifyAutoContinueFailure(undefined)).toBeUndefined();
      expect(classifyAutoContinueFailure('error')).toBeUndefined();
      expect(classifyAutoContinueFailure(null)).toBeUndefined();
    });
  });

  describe('buildAutoContinueEntry', () => {
    it('builds a custom_message draft, never a literal user message', () => {
      const entry = buildAutoContinueEntry();
      expect(entry.type).toBe('custom_message');
      expect(entry.customType).toBe(AUTO_CONTINUE_CUSTOM_TYPE);
      expect(entry.customType).not.toBe('user');
      expect(typeof entry.content).toBe('string');
      expect(entry.content.length).toBeGreaterThan(0);
    });
  });

  describe('shouldFireAutoContinue', () => {
    it('fires for a first-time failure with nothing cooling and no chain armed', () => {
      expect(
        shouldFireAutoContinue({
          chainArmed: false,
          stillCoolingFromEarlierFailure: false,
          alreadyFiredForActiveModel: false,
        }),
      ).toBe(true);
    });

    it('never fires when the chain is already armed', () => {
      expect(
        shouldFireAutoContinue({
          chainArmed: true,
          stillCoolingFromEarlierFailure: false,
          alreadyFiredForActiveModel: false,
        }),
      ).toBe(false);
    });

    it('does not fire a second time while the same cooldown period is still active', () => {
      expect(
        shouldFireAutoContinue({
          chainArmed: false,
          stillCoolingFromEarlierFailure: true,
          alreadyFiredForActiveModel: true,
        }),
      ).toBe(false);
    });

    it('fires again once the earlier cooldown is no longer active, even if fired before', () => {
      expect(
        shouldFireAutoContinue({
          chainArmed: false,
          stillCoolingFromEarlierFailure: false,
          alreadyFiredForActiveModel: true,
        }),
      ).toBe(true);
    });
  });

  describe('evaluateBeforeSettleForAutoContinue', () => {
    const notCooling = () => false;
    const hasReplacement = () => true;

    it('fires once on a non-retryable usage-limit error and requests exactly one continuation', () => {
      const guard = createAutoContinueGuardState();
      const result = evaluateBeforeSettleForAutoContinue(
        beforeSettleEvent('error', [assistantError()]),
        guard,
        notCooling,
        hasReplacement,
      );
      expect(result).toBeDefined();
      expect(result?.failedRef).toBe('openai/gpt-4o-mini');
      expect(result?.errorMessage).toBe('insufficient_quota');
      expect(result?.continue).toBe(true);
      expect(result?.entries).toHaveLength(1);
      expect(result?.entries[0].customType).toBe(AUTO_CONTINUE_CUSTOM_TYPE);
      expect(guard.chainArmed).toBe(true);
      expect(guard.firedModels.has('openai/gpt-4o-mini')).toBe(true);
    });

    // The regression this guards: `turn_end`'s `continue` is discarded by Pi
    // whenever its own finishTurn decision was already "end" (always true for
    // a non-retryable error), so only the LAST message of
    // `contextMessages` -- the real persisted session projection
    // `agent_before_settle` observes -- may ever be classified, never a
    // same-shaped `event.message` field turn_end used to carry.
    it('only inspects the last context message, never an earlier one', () => {
      const guard = createAutoContinueGuardState();
      const result = evaluateBeforeSettleForAutoContinue(
        beforeSettleEvent('error', [
          assistantError({ provider: 'google', model: 'gemini-1.5-flash' }),
          { role: 'user', content: 'hi' },
        ]),
        guard,
        notCooling,
        hasReplacement,
      );
      expect(result).toBeUndefined();
    });

    it('never fires for a retryable error (leaves it to Pi\u2019s own retry/fallback chain)', () => {
      const guard = createAutoContinueGuardState();
      const result = evaluateBeforeSettleForAutoContinue(
        beforeSettleEvent('error', [
          assistantError({ errorMessage: '429 too many requests' }),
        ]),
        guard,
        notCooling,
        hasReplacement,
      );
      expect(result).toBeUndefined();
      expect(guard.chainArmed).toBe(false);
    });

    it('never fires for an aborted turn', () => {
      const guard = createAutoContinueGuardState();
      const result = evaluateBeforeSettleForAutoContinue(
        beforeSettleEvent('aborted', [assistantError()]),
        guard,
        notCooling,
        hasReplacement,
      );
      expect(result).toBeUndefined();
      expect(guard.chainArmed).toBe(false);
    });

    it('never chains: a second consecutive error does not fire again, even on a different model', () => {
      const guard = createAutoContinueGuardState();
      const first = evaluateBeforeSettleForAutoContinue(
        beforeSettleEvent('error', [assistantError()]),
        guard,
        notCooling,
        hasReplacement,
      );
      expect(first).toBeDefined();

      // The continuation itself also fails, now on a different fallback model.
      const second = evaluateBeforeSettleForAutoContinue(
        beforeSettleEvent('error', [
          assistantError({ provider: 'google', model: 'gemini-1.5-flash' }),
        ]),
        guard,
        notCooling,
        hasReplacement,
      );
      expect(second).toBeUndefined();
    });

    // The regression this guards: Pi's run loop can dispatch
    // `agent_before_settle` more than once within the SAME run (a
    // continuation that itself succeeds, followed by a later turn in that
    // same run that fails on a different model). Only `agent_settled` --
    // never a non-error `agent_before_settle` -- may reset the chain guard.
    it('a non-error agent_before_settle within the same run does not reset the chain guard', () => {
      const guard = createAutoContinueGuardState();
      const first = evaluateBeforeSettleForAutoContinue(
        beforeSettleEvent('error', [assistantError()]),
        guard,
        notCooling,
        hasReplacement,
      );
      expect(first).toBeDefined();

      // The continuation succeeds, but the run is not over yet (Pi's loop
      // calls agent_before_settle again before a later turn in this run).
      evaluateBeforeSettleForAutoContinue(
        beforeSettleEvent('completed', [
          assistantError({ stopReason: 'stop' }),
        ]),
        guard,
        notCooling,
        hasReplacement,
      );
      expect(guard.chainArmed).toBe(true);

      // A later turn in the SAME run fails on a different model.
      const second = evaluateBeforeSettleForAutoContinue(
        beforeSettleEvent('error', [
          assistantError({ provider: 'google', model: 'gemini-1.5-flash' }),
        ]),
        guard,
        notCooling,
        hasReplacement,
      );
      expect(second).toBeUndefined();
    });

    it('is silent on a later run while the same model is still cooling (no notice, no auto-continue)', () => {
      const guard = createAutoContinueGuardState();
      const isStillCooling = () => true;
      const first = evaluateBeforeSettleForAutoContinue(
        beforeSettleEvent('error', [assistantError()]),
        guard,
        isStillCooling,
        hasReplacement,
      );
      expect(first).toBeDefined();

      // The run settles, isolating this case to the per-(provider/model,
      // cooldown period) dedup alone.
      resetAutoContinueGuardOnSettle(guard);
      expect(guard.chainArmed).toBe(false);

      const second = evaluateBeforeSettleForAutoContinue(
        beforeSettleEvent('error', [assistantError()]),
        guard,
        isStillCooling,
        hasReplacement,
      );
      expect(second).toBeUndefined();
    });

    it('fires again for the same model once its cooldown period has expired', () => {
      const guard = createAutoContinueGuardState();
      let cooling = true;
      const isStillCooling = () => cooling;

      const first = evaluateBeforeSettleForAutoContinue(
        beforeSettleEvent('error', [assistantError()]),
        guard,
        isStillCooling,
        hasReplacement,
      );
      expect(first).toBeDefined();

      resetAutoContinueGuardOnSettle(guard);

      cooling = false; // cooldown has expired by the time of this later failure
      const second = evaluateBeforeSettleForAutoContinue(
        beforeSettleEvent('error', [assistantError()]),
        guard,
        isStillCooling,
        hasReplacement,
      );
      expect(second).toBeDefined();
    });

    // The regression this guards: firing without a real replacement just
    // re-requests the same failed (or another unusable/still-cooling) model,
    // burning the one-shot budget for nothing visible to change.
    it('does not fire when no usable replacement candidate exists', () => {
      const guard = createAutoContinueGuardState();
      const noReplacement = () => false;
      const result = evaluateBeforeSettleForAutoContinue(
        beforeSettleEvent('error', [assistantError()]),
        guard,
        notCooling,
        noReplacement,
      );
      expect(result).toBeUndefined();
      expect(guard.chainArmed).toBe(false);
      expect(guard.firedModels.has('openai/gpt-4o-mini')).toBe(false);
    });

    // A failure skipped for lack of a replacement must not burn the one-shot
    // budget: once a replacement becomes available (e.g. another model's
    // cooldown expires), the same failed model can still fire.
    it('a skipped no-replacement failure does not block a later fire for the same model', () => {
      const guard = createAutoContinueGuardState();
      let replacementAvailable = false;
      const hasReplacementNow = () => replacementAvailable;

      const first = evaluateBeforeSettleForAutoContinue(
        beforeSettleEvent('error', [assistantError()]),
        guard,
        notCooling,
        hasReplacementNow,
      );
      expect(first).toBeUndefined();

      replacementAvailable = true;
      const second = evaluateBeforeSettleForAutoContinue(
        beforeSettleEvent('error', [assistantError()]),
        guard,
        notCooling,
        hasReplacementNow,
      );
      expect(second).toBeDefined();
    });

    it('does not throw on malformed event input', () => {
      const guard = createAutoContinueGuardState();
      expect(
        evaluateBeforeSettleForAutoContinue(
          undefined,
          guard,
          notCooling,
          hasReplacement,
        ),
      ).toBeUndefined();
      expect(
        evaluateBeforeSettleForAutoContinue(
          'oops',
          guard,
          notCooling,
          hasReplacement,
        ),
      ).toBeUndefined();
      expect(
        evaluateBeforeSettleForAutoContinue(
          {},
          guard,
          notCooling,
          hasReplacement,
        ),
      ).toBeUndefined();
      expect(
        evaluateBeforeSettleForAutoContinue(
          { outcome: 'error', context: {} },
          guard,
          notCooling,
          hasReplacement,
        ),
      ).toBeUndefined();
      expect(
        evaluateBeforeSettleForAutoContinue(
          { outcome: 'error', context: { contextMessages: [] } },
          guard,
          notCooling,
          hasReplacement,
        ),
      ).toBeUndefined();
    });
  });

  describe('resetAutoContinueGuardOnSettle', () => {
    it('clears chainArmed without touching firedModels', () => {
      const guard = createAutoContinueGuardState();
      guard.chainArmed = true;
      guard.firedModels.add('openai/gpt-4o-mini');

      resetAutoContinueGuardOnSettle(guard);

      expect(guard.chainArmed).toBe(false);
      expect(guard.firedModels.has('openai/gpt-4o-mini')).toBe(true);
    });
  });
});
