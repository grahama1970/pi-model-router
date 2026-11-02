import { describe, it, expect } from 'vitest';
import {
  ENCRYPTED_REASONING_CUSTOM_TYPE,
  ENCRYPTED_REJECTION_ENTRY_TYPE,
  buildEncryptedReasoningEntry,
  buildEncryptedRejectionRecord,
  classifyEncryptedReasoningFailure,
  collectEncryptedReasoningExclusions,
  collectUnrecordedEncryptedRejections,
  conversationHasImage,
  createEncryptedReasoningRecoveryState,
  evaluateBeforeSettleForEncryptedReasoning,
  isEncryptedReasoningReplayError,
  isEncryptedRejectionRecord,
  observeEncryptedReasoningFailure,
  rejectionRecordKey,
  resetEncryptedReasoningRecoveryOnSettle,
  signatureFingerprint,
  signedHistoryFingerprints,
} from './encrypted-reasoning';

// The exact shape Pi produces for a rejected replay: the provider body,
// prefixed with the HTTP status by pi-ai's formatProviderError().
const CODEX_REJECTION =
  '400: {"error":{"message":"Invalid \'input[3].encrypted_content\': Encrypted content could not be verified.","type":"invalid_request_error","param":"input[3].encrypted_content","code":"invalid_encrypted_content"}}';

const encryptedError = (overrides: Record<string, unknown> = {}) => ({
  role: 'assistant',
  stopReason: 'error',
  provider: 'openai-codex',
  model: 'gpt-5.5-codex',
  api: 'openai-responses',
  errorMessage: CODEX_REJECTION,
  ...overrides,
});

// A successful assistant message that carries a replayable signed thinking
// block, the shape Pi persists for OpenAI Responses reasoning items.
const signedAssistant = (
  signature: string,
  overrides: Record<string, unknown> = {},
) => ({
  role: 'assistant',
  stopReason: 'stop',
  provider: 'openai-codex',
  model: 'gpt-5.5-codex',
  api: 'openai-responses',
  content: [
    { type: 'thinking', thinking: 'thought', thinkingSignature: signature },
  ],
  ...overrides,
});

// Mirrors the real `AgentBeforeSettleEvent` shape
// (`event.context.contextMessages`), narrowed to what
// `evaluateBeforeSettleForEncryptedReasoning` reads.
const beforeSettleEvent = (outcome: string, contextMessages: unknown[]) => ({
  outcome,
  context: { contextMessages },
});

// Fixtures are openai-responses throughout; the resolver makes the explicit
// API identity available to the exclusion logic (absence is never a wildcard).
const RESPONSES_API = 'openai-responses';
const exclusionsFor = (
  input: Omit<
    Parameters<typeof collectEncryptedReasoningExclusions>[0],
    'resolveModelApi'
  > & { resolveModelApi?: (ref: string) => string | undefined },
) =>
  collectEncryptedReasoningExclusions({
    resolveModelApi: () => RESPONSES_API,
    ...input,
  });

// Same for the decision: fixtures are openai-responses, so the API resolver is
// explicit (never a wildcard).
const evaluateRecovery = (
  ...args: Parameters<typeof evaluateBeforeSettleForEncryptedReasoning>
) =>
  evaluateBeforeSettleForEncryptedReasoning(
    args[0],
    args[1],
    args[2],
    args[3] ?? [],
    args[4] ?? (() => RESPONSES_API),
  );

describe('encrypted-reasoning.ts', () => {
  describe('isEncryptedReasoningReplayError', () => {
    it('matches the observed Codex encrypted_content rejection', () => {
      expect(isEncryptedReasoningReplayError(CODEX_REJECTION)).toBe(true);
    });

    it('matches an undecryptable extra body with reasoning wording', () => {
      expect(
        isEncryptedReasoningReplayError(
          'Encrypted reasoning content could not be decrypted',
        ),
      ).toBe(true);
      expect(
        isEncryptedReasoningReplayError(
          '400: {"error":{"message":"Invalid encrypted reasoning content"}}',
        ),
      ).toBe(true);
    });

    it('matches a thinking signature verification failure', () => {
      expect(
        isEncryptedReasoningReplayError(
          'thinkingSignature verification failed: signature mismatch',
        ),
      ).toBe(true);
    });

    // Narrowness: every other error the router already owns must stay with
    // its own path -- a quota error belongs to auto-continue/cooldowns, and an
    // unrelated provider rejection must simply surface.
    it('does not match usage-limit, aborted, transport, or unrelated validation errors', () => {
      for (const text of [
        'insufficient_quota',
        '429 Too Many Requests',
        'subscription_sharing_usage_limit_exceeded',
        'Request was aborted',
        '503 Service Unavailable',
        '400: {"error":{"message":"Unknown parameter: \'foo\'"}}',
        'invalid api key',
        'The model does not support image input',
      ]) {
        expect(isEncryptedReasoningReplayError(text)).toBe(false);
      }
    });

    it('does not match an empty error message', () => {
      expect(isEncryptedReasoningReplayError('')).toBe(false);
    });
  });

  describe('classifyEncryptedReasoningFailure', () => {
    it('classifies an assistant error naming the model that rejected the payload', () => {
      expect(classifyEncryptedReasoningFailure(encryptedError())).toEqual({
        failedRef: 'openai-codex/gpt-5.5-codex',
        provider: 'openai-codex',
        modelId: 'gpt-5.5-codex',
        api: 'openai-responses',
        errorMessage: CODEX_REJECTION,
        timestamp: undefined,
      });
    });

    it('carries the failing API and the message event time for the replay identity', () => {
      const failure = classifyEncryptedReasoningFailure(
        encryptedError({ api: 'openai-responses', timestamp: 1700000000000 }),
      );
      expect(failure?.api).toBe('openai-responses');
      expect(failure?.timestamp).toBe(1700000000000);
    });

    it('never classifies an abort', () => {
      expect(
        classifyEncryptedReasoningFailure(
          encryptedError({ stopReason: 'aborted' }),
        ),
      ).toBeUndefined();
    });

    it('never classifies a successful message or one without provider/model identity', () => {
      expect(
        classifyEncryptedReasoningFailure(
          encryptedError({ stopReason: 'stop' }),
        ),
      ).toBeUndefined();
      expect(
        classifyEncryptedReasoningFailure(encryptedError({ provider: '' })),
      ).toBeUndefined();
      expect(
        classifyEncryptedReasoningFailure(encryptedError({ model: undefined })),
      ).toBeUndefined();
      expect(classifyEncryptedReasoningFailure(undefined)).toBeUndefined();
      expect(classifyEncryptedReasoningFailure('boom')).toBeUndefined();
    });

    it('does not classify a limit error even when it mentions a model', () => {
      expect(
        classifyEncryptedReasoningFailure(
          encryptedError({ errorMessage: 'insufficient_quota' }),
        ),
      ).toBeUndefined();
    });
  });

  describe('collectEncryptedReasoningExclusions', () => {
    it('collects every model in the conversation that rejected the payload', () => {
      const excluded = exclusionsFor({
        messages: [
          { role: 'user', content: 'do it', timestamp: 1 },
          encryptedError(),
          encryptedError({
            provider: 'openai',
            model: 'gpt-5.5',
          }),
          { role: 'user', content: 'continue', timestamp: 2 },
        ],
      });
      expect([...excluded]).toEqual([
        'openai-codex/gpt-5.5-codex',
        'openai/gpt-5.5',
      ]);
    });

    it('ignores every other kind of message', () => {
      const excluded = exclusionsFor({
        messages: [
          { role: 'user', content: 'hi', timestamp: 1 },
          {
            role: 'assistant',
            stopReason: 'error',
            provider: 'openai',
            model: 'gpt-4o-mini',
            errorMessage: 'insufficient_quota',
          },
          {
            role: 'assistant',
            stopReason: 'aborted',
            provider: 'openai',
            model: 'gpt-4o',
            errorMessage: CODEX_REJECTION,
          },
        ],
      });
      expect(excluded.size).toBe(0);
    });
  });

  describe('signed history fingerprints', () => {
    it('fingerprints signed thinking blocks deterministically and opaquely', () => {
      const messages = [signedAssistant('sig-alpha')];
      const first = signedHistoryFingerprints(messages);
      const second = signedHistoryFingerprints(messages);
      expect(first).toEqual(second);
      expect(first).toHaveLength(1);
      expect(first[0]).toMatch(/^[0-9a-f]{16}$/);
      // The digest must not carry the signature itself.
      expect(first[0]).not.toContain('sig-alpha');
      expect(first[0]).not.toContain('alpha');
    });

    it('changes when the signed payload changes and ignores text/tool signatures', () => {
      expect(signedHistoryFingerprints([signedAssistant('sig-a')])).not.toEqual(
        signedHistoryFingerprints([signedAssistant('sig-b')]),
      );
      expect(
        signedHistoryFingerprints([
          {
            role: 'assistant',
            provider: 'openai-codex',
            model: 'gpt-5.5-codex',
            content: [
              {
                type: 'text',
                text: 'hi',
                textSignature: 'not-a-reasoning-sig',
              },
              { type: 'toolCall', id: 'c|fc', name: 'bash', arguments: {} },
            ],
          },
        ]),
      ).toEqual([]);
    });

    it('deduplicates identical payloads and binds the model that produced them', () => {
      const same = signedHistoryFingerprints([
        signedAssistant('sig-x'),
        signedAssistant('sig-x'),
      ]);
      expect(same).toHaveLength(1);
      expect(
        signatureFingerprint(
          'openai-codex',
          'openai-responses',
          'gpt-5.5-codex',
          'sig-x',
        ),
      ).not.toEqual(
        signatureFingerprint('openai', 'openai-responses', 'gpt-5.5', 'sig-x'),
      );
    });
  });

  describe('durable rejection records', () => {
    // P1 (over-binding): only blocks the rejecting model could itself replay
    // are bound. Pi drops every other model's thinking before sending, so
    // binding those would keep this model excluded after its own payload is
    // gone, and would misattribute another model's payload.
    it('binds only the signed blocks replayable to the rejecting model', () => {
      const record = buildEncryptedRejectionRecord(
        classifyEncryptedReasoningFailure(
          encryptedError({ provider: 'openai-codex', model: 'gpt-5.5-codex' }),
        )!,
        [
          signedAssistant('sig-a'),
          signedAssistant('sig-b', { model: 'other-model' }),
          signedAssistant('sig-c', { provider: 'other-provider' }),
        ],
        1700000000000,
      );
      expect(record).toMatchObject({
        customType: ENCRYPTED_REJECTION_ENTRY_TYPE,
        failedRef: 'openai-codex/gpt-5.5-codex',
        at: 1700000000000,
      });
      expect(record?.fingerprints).toEqual([
        signatureFingerprint(
          'openai-codex',
          'openai-responses',
          'gpt-5.5-codex',
          'sig-a',
        ),
      ]);
      // Never the raw payload.
      expect(JSON.stringify(record)).not.toContain('sig-a');
    });

    it('ignores errored and aborted assistant turns when binding payloads', () => {
      const record = buildEncryptedRejectionRecord(
        classifyEncryptedReasoningFailure(
          encryptedError({ provider: 'openai-codex', model: 'gpt-5.5-codex' }),
        )!,
        [
          signedAssistant('sig-a'),
          signedAssistant('sig-err', { stopReason: 'error' }),
          signedAssistant('sig-abort', { stopReason: 'aborted' }),
        ],
        1700000000000,
      );
      expect(record?.fingerprints).toEqual([
        signatureFingerprint(
          'openai-codex',
          'openai-responses',
          'gpt-5.5-codex',
          'sig-a',
        ),
      ]);
    });

    it('creates no record when the rejecting model has no replayable signed block', () => {
      expect(
        buildEncryptedRejectionRecord(
          classifyEncryptedReasoningFailure(
            encryptedError({
              provider: 'openai-codex',
              model: 'gpt-5.5-codex',
            }),
          )!,
          [signedAssistant('sig-foreign', { model: 'other-model' })],
          1700000000000,
        ),
      ).toBeUndefined();
    });

    // Documented boundary: with nothing identifiable to bind to, no durable
    // record is created (never a blanket per-model ban, never a cooldown).
    it('creates no record when the conversation carries no signed history', () => {
      expect(
        buildEncryptedRejectionRecord(
          classifyEncryptedReasoningFailure(
            encryptedError({
              provider: 'openai-codex',
              model: 'gpt-5.5-codex',
            }),
          )!,
          [{ role: 'user', content: 'hi', timestamp: 1 }],
          1700000000000,
        ),
      ).toBeUndefined();
    });

    it('validates persisted record shape and rejects malformed fingerprints', () => {
      const record = buildEncryptedRejectionRecord(
        classifyEncryptedReasoningFailure(
          encryptedError({ provider: 'openai-codex', model: 'gpt-5.5-codex' }),
        )!,
        [signedAssistant('sig-a')],
        1700000000000,
      );
      expect(isEncryptedRejectionRecord(record)).toBe(true);
      expect(
        isEncryptedRejectionRecord({ ...record, fingerprints: ['nope'] }),
      ).toBe(false);
      expect(isEncryptedRejectionRecord({ ...record, failedRef: '' })).toBe(
        false,
      );
      expect(isEncryptedRejectionRecord({ failedRef: 'a/b' })).toBe(false);
      expect(
        isEncryptedRejectionRecord({ ...record, customType: 'other' }),
      ).toBe(false);
    });

    it('keys records by failed model and fingerprint set', () => {
      const a = buildEncryptedRejectionRecord(
        classifyEncryptedReasoningFailure(
          encryptedError({ provider: 'openai-codex', model: 'gpt-5.5-codex' }),
        )!,
        [signedAssistant('sig-a')],
        1,
      )!;
      const b = buildEncryptedRejectionRecord(
        classifyEncryptedReasoningFailure(
          encryptedError({ provider: 'openai-codex', model: 'gpt-5.5-codex' }),
        )!,
        [signedAssistant('sig-b')],
        2,
      )!;
      expect(rejectionRecordKey(a)).toBe(
        rejectionRecordKey({
          failedRef: a.failedRef,
          api: a.api,
          fingerprints: a.fingerprints,
        }),
      );
      expect(rejectionRecordKey(a)).not.toBe(rejectionRecordKey(b));
    });
  });

  describe('collectEncryptedReasoningExclusions (durable metadata)', () => {
    const record = buildEncryptedRejectionRecord(
      classifyEncryptedReasoningFailure(
        encryptedError({ provider: 'openai-codex', model: 'gpt-5.5-codex' }),
      )!,
      [signedAssistant('sig-a')],
      1700000000000,
    )!;

    // The P1-2 regression: a branch summary can keep the signed response and
    // move the rejecting error off the active path, so the message-based scan
    // sees nothing. The fingerprint-bound record must still exclude.
    it('excludes a model whose bound signed history is still present, even with no error message in the projection', () => {
      const messages = [signedAssistant('sig-a')];
      expect(classifyEncryptedReasoningFailure(messages[0])).toBeUndefined();
      const excluded = exclusionsFor({
        messages,
        records: [record],
      });
      expect([...excluded]).toEqual(['openai-codex/gpt-5.5-codex']);
    });

    // "Retire when the implicated signed history disappears, not merely when
    // the error is projected away": with the payload gone, the record no
    // longer excludes even if a stale rejection message is still projected.
    it('retires the record once the implicated signed history is gone', () => {
      const excluded = exclusionsFor({
        messages: [
          { role: 'user', content: 'hi', timestamp: 1 },
          encryptedError(),
        ],
        records: [record],
      });
      expect([...excluded]).toEqual([]);
    });

    it('keeps excluding a model with no record while its rejection is projected', () => {
      const excluded = exclusionsFor({
        messages: [signedAssistant('sig-a'), encryptedError()],
      });
      expect([...excluded]).toEqual(['openai-codex/gpt-5.5-codex']);
    });

    // Retryable wrapper: Pi hands the failure over as `request.failed`, which
    // need not be in the projection at all.
    it('excludes the failed model carried only by request.failed', () => {
      const excluded = exclusionsFor({
        messages: [signedAssistant('sig-a')],
        failed: encryptedError({
          errorMessage: `500 service unavailable: ${CODEX_REJECTION}`,
        }),
      });
      expect([...excluded]).toEqual(['openai-codex/gpt-5.5-codex']);
    });

    it('does not exclude when every fingerprint is absent and nothing is projected', () => {
      expect(
        exclusionsFor({
          messages: [signedAssistant('sig-other')],
          records: [record],
        }).size,
      ).toBe(0);
    });
  });

  describe('mixed A/B signed history (P1: over-binding)', () => {
    // A = another model whose signed history stays in the conversation,
    // B = the rejecting model. Pi drops A's thinking before sending to B.
    const historyWithA = [
      signedAssistant('sig-a', { provider: 'openai', model: 'gpt-5.5' }),
      signedAssistant('sig-b'),
      encryptedError(),
    ];

    it("keeps B excluded while B's own signed payload remains", () => {
      const record = buildEncryptedRejectionRecord(
        classifyEncryptedReasoningFailure(
          encryptedError({ provider: 'openai-codex', model: 'gpt-5.5-codex' }),
        )!,
        historyWithA,
        1700000000000,
      )!;
      expect(record.fingerprints).toEqual([
        signatureFingerprint(
          'openai-codex',
          'openai-responses',
          'gpt-5.5-codex',
          'sig-b',
        ),
      ]);

      const excluded = exclusionsFor({
        messages: [
          signedAssistant('sig-a', { provider: 'openai', model: 'gpt-5.5' }),
          signedAssistant('sig-b'),
          // The rejection message itself is off the active path (branch
          // summary shape).
        ],
        records: [record],
      });
      expect([...excluded]).toEqual(['openai-codex/gpt-5.5-codex']);
    });

    it("makes B eligible again once B's payload is gone, even though A's stays", () => {
      const record = buildEncryptedRejectionRecord(
        classifyEncryptedReasoningFailure(
          encryptedError({ provider: 'openai-codex', model: 'gpt-5.5-codex' }),
        )!,
        historyWithA,
        1700000000000,
      )!;
      const excluded = exclusionsFor({
        messages: [
          signedAssistant('sig-a', { provider: 'openai', model: 'gpt-5.5' }),
        ],
        records: [record],
      });
      expect([...excluded]).toEqual([]);
    });
  });

  describe('fresh failures are never suppressed by an older record (P1)', () => {
    // Record for model A bound to payload X, taken at time T.
    const at = 1700000000000;
    const recordX = buildEncryptedRejectionRecord(
      classifyEncryptedReasoningFailure(
        encryptedError({ provider: 'openai-codex', model: 'gpt-5.5-codex' }),
      )!,
      [signedAssistant('sig-x')],
      at,
    )!;

    it('honors request.failed for a model that already has an older record', () => {
      // Payload X is gone; the model now rejects a FRESH request (payload Y).
      const excluded = exclusionsFor({
        messages: [signedAssistant('sig-y')],
        failed: encryptedError(),
        records: [recordX],
      });
      expect([...excluded]).toEqual(['openai-codex/gpt-5.5-codex']);
    });

    it('persists a new record for a new payload of the same model', () => {
      const records = collectUnrecordedEncryptedRejections({
        messages: [signedAssistant('sig-y')],
        failed: encryptedError(),
        records: [recordX],
        at: at + 1000,
      });
      expect(records).toHaveLength(1);
      expect(records[0].failedRef).toBe('openai-codex/gpt-5.5-codex');
      expect(records[0].fingerprints).not.toEqual(recordX.fingerprints);
      expect(rejectionRecordKey(records[0])).not.toBe(
        rejectionRecordKey(recordX),
      );
    });

    it('does not duplicate the record for the same failure identity', () => {
      expect(
        collectUnrecordedEncryptedRejections({
          messages: [signedAssistant('sig-x')],
          failed: encryptedError(),
          records: [recordX],
          at: at + 1000,
        }),
      ).toEqual([]);
    });

    it('treats a projected error newer than the record as a new failure', () => {
      const excluded = exclusionsFor({
        messages: [
          signedAssistant('sig-y'),
          encryptedError({ timestamp: at + 5000 }),
        ],
        records: [recordX],
      });
      expect([...excluded]).toEqual(['openai-codex/gpt-5.5-codex']);
    });

    it('treats a projected error older than the record as the recorded failure, so retirement still applies', () => {
      const excluded = exclusionsFor({
        messages: [
          signedAssistant('sig-y'),
          encryptedError({ timestamp: at - 5000 }),
        ],
        records: [recordX],
      });
      expect([...excluded]).toEqual([]);
    });

    it('keeps new-failure exclusion per model, independently', () => {
      const recordOther = buildEncryptedRejectionRecord(
        classifyEncryptedReasoningFailure(
          encryptedError({ provider: 'openai', model: 'gpt-5.5' }),
        )!,
        [signedAssistant('sig-z', { provider: 'openai', model: 'gpt-5.5' })],
        at,
      )!;
      const excluded = exclusionsFor({
        messages: [signedAssistant('sig-y')],
        failed: encryptedError({
          provider: 'openai-codex',
          model: 'gpt-5.5-codex',
        }),
        records: [recordOther],
      });
      expect([...excluded]).toEqual(['openai-codex/gpt-5.5-codex']);
    });
  });

  describe('observeEncryptedReasoningFailure', () => {
    it('reports the rejected model with a payload-bound record', () => {
      const observation = observeEncryptedReasoningFailure({
        messages: [signedAssistant('sig-x'), encryptedError()],
        at: 1700000000000,
      });
      expect(observation?.failedRef).toBe('openai-codex/gpt-5.5-codex');
      expect(observation?.record?.fingerprints).toHaveLength(1);
    });

    it('reports the failure without a record when there is no signed history', () => {
      const observation = observeEncryptedReasoningFailure({
        messages: [encryptedError()],
        at: 1700000000000,
      });
      expect(observation).toEqual({
        failedRef: 'openai-codex/gpt-5.5-codex',
        record: undefined,
      });
    });

    it('ignores a non-rejection last message and an empty transcript', () => {
      expect(
        observeEncryptedReasoningFailure({ messages: [], at: 1 }),
      ).toBeUndefined();
      expect(
        observeEncryptedReasoningFailure({
          messages: [signedAssistant('sig-x')],
          at: 1,
        }),
      ).toBeUndefined();
    });
  });

  describe('collectUnrecordedEncryptedRejections', () => {
    it('returns one record per unrecorded rejected model that has its own replayable payload', () => {
      const messages = [
        signedAssistant('sig-a'),
        signedAssistant('sig-foreign', {
          provider: 'openai',
          model: 'gpt-5.5',
        }),
        encryptedError(),
      ];
      const records = collectUnrecordedEncryptedRejections({
        messages,
        failed: encryptedError({ provider: 'openai', model: 'gpt-5.5' }),
        at: 1700000000000,
      });
      expect(records.map((entry) => entry.failedRef).sort()).toEqual([
        'openai-codex/gpt-5.5-codex',
        'openai/gpt-5.5',
      ]);
      // Each record binds only its own model's payload.
      for (const record of records) {
        expect(record.fingerprints).toHaveLength(1);
      }
    });

    it('creates no record for a rejected model that has no replayable payload of its own', () => {
      const records = collectUnrecordedEncryptedRejections({
        messages: [signedAssistant('sig-a'), encryptedError()],
        failed: encryptedError({ provider: 'openai', model: 'gpt-5.5' }),
        at: 1700000000000,
      });
      expect(records.map((entry) => entry.failedRef)).toEqual([
        'openai-codex/gpt-5.5-codex',
      ]);
    });

    it('skips models that already have a durable record', () => {
      const record = buildEncryptedRejectionRecord(
        classifyEncryptedReasoningFailure(
          encryptedError({ provider: 'openai-codex', model: 'gpt-5.5-codex' }),
        )!,
        [signedAssistant('sig-a')],
        1,
      )!;
      expect(
        collectUnrecordedEncryptedRejections({
          messages: [signedAssistant('sig-a'), encryptedError()],
          records: [record],
          at: 2,
        }),
      ).toEqual([]);
    });

    it('creates nothing when there is no signed history to bind to', () => {
      expect(
        collectUnrecordedEncryptedRejections({
          messages: [encryptedError()],
          at: 1,
        }),
      ).toEqual([]);
    });
  });

  // Review v3 regressions. These were written first and failed on the
  // pre-repair implementation (evidence: 02-before-fix-unit.txt in the retained
  // proof); they now pin the corrected semantics.
  describe('review v3 counterexamples', () => {
    const failureWith = (
      api: string,
      overrides: Record<string, unknown> = {},
    ) =>
      classifyEncryptedReasoningFailure(encryptedError({ api, ...overrides }))!;

    // P1-1: the replay identity is provider + API + model (transform-messages
    // isSameModel), so a signature produced under another API is not bound.
    it('P1-1: does not bind a signature produced under a different API', () => {
      const record = buildEncryptedRejectionRecord(
        failureWith('openai-responses'),
        [
          signedAssistant('sig-responses', { api: 'openai-responses' }),
          signedAssistant('sig-completions', { api: 'openai-completions' }),
        ],
        1700000000000,
      );
      expect(record?.api).toBe('openai-responses');
      expect(record?.fingerprints).toEqual([
        signatureFingerprint(
          'openai-codex',
          'openai-responses',
          'gpt-5.5-codex',
          'sig-responses',
        ),
      ]);
    });

    // P1-1: "old-API signature must not ban changed-API target".
    it('P1-1: an old-API record does not ban a target whose API changed', () => {
      const record = buildEncryptedRejectionRecord(
        failureWith('openai-responses'),
        [signedAssistant('sig-responses', { api: 'openai-responses' })],
        1700000000000,
      )!;
      const messages = [
        signedAssistant('sig-responses', { api: 'openai-responses' }),
      ];
      expect([
        ...exclusionsFor({
          messages,
          records: [record],
          resolveModelApi: () => 'openai-completions',
        }),
      ]).toEqual([]);
      // Same record, same payload, same API as the target: still excluded.
      expect([
        ...exclusionsFor({
          messages,
          records: [record],
          resolveModelApi: () => 'openai-responses',
        }),
      ]).toEqual(['openai-codex/gpt-5.5-codex']);
    });

    // P1-2: a stale projected error must not fabricate a ban bound to a
    // signature that appeared after it.
    it('P1-2: a stale projected error does not fabricate a ban on newer history', () => {
      const at = 1700000000000;
      const failureAt = at - 5000;
      const recordX = buildEncryptedRejectionRecord(
        failureWith('openai-responses', { timestamp: failureAt }),
        [signedAssistant('sig-x', { api: 'openai-responses' })],
        at,
      )!;
      expect(recordX.failureAt).toBe(failureAt);

      expect(
        collectUnrecordedEncryptedRejections({
          messages: [
            signedAssistant('sig-x', { api: 'openai-responses' }),
            encryptedError({
              api: 'openai-responses',
              timestamp: failureAt,
            }),
            // A NEW valid payload produced after the failure.
            signedAssistant('sig-y', { api: 'openai-responses' }),
          ],
          records: [recordX],
          at,
        }),
      ).toEqual([]);
    });

    // Contrast: a genuinely NEWER projected failure is still recorded, bound
    // only to the history that precedes it.
    it('P1-2: a newer projected failure is still recorded against its preceding history', () => {
      const at = 1700000000000;
      const recordX = buildEncryptedRejectionRecord(
        failureWith('openai-responses', { timestamp: at - 9000 }),
        [signedAssistant('sig-x', { api: 'openai-responses' })],
        at,
      )!;
      const records = collectUnrecordedEncryptedRejections({
        messages: [
          signedAssistant('sig-x', { api: 'openai-responses' }),
          encryptedError({ api: 'openai-responses', timestamp: at - 9000 }),
          signedAssistant('sig-y', { api: 'openai-responses' }),
          encryptedError({ api: 'openai-responses', timestamp: at - 1000 }),
          // Produced AFTER the new failure; must not be bound to it.
          signedAssistant('sig-z', { api: 'openai-responses' }),
        ],
        records: [recordX],
        at,
      });
      expect(records).toHaveLength(1);
      expect(records[0].failureAt).toBe(at - 1000);
      expect(records[0].fingerprints).toEqual([
        signatureFingerprint(
          'openai-codex',
          'openai-responses',
          'gpt-5.5-codex',
          'sig-x',
        ),
        signatureFingerprint(
          'openai-codex',
          'openai-responses',
          'gpt-5.5-codex',
          'sig-y',
        ),
      ]);
    });

    // P1-2: an already observed failure is never re-recorded with a new payload.
    it('P1-2: re-observing the same failure does not rewrite its payload set', () => {
      const at = 1700000000000;
      const failureAt = at - 5000;
      const observed = buildEncryptedRejectionRecord(
        failureWith('openai-responses', { timestamp: failureAt }),
        [signedAssistant('sig-x', { api: 'openai-responses' })],
        at,
      )!;
      const again = collectUnrecordedEncryptedRejections({
        messages: [
          signedAssistant('sig-x', { api: 'openai-responses' }),
          encryptedError({ api: 'openai-responses', timestamp: failureAt }),
          signedAssistant('sig-y', { api: 'openai-responses' }),
        ],
        records: [observed],
        at: at + 1000,
      });
      expect(again).toEqual([]);
    });
  });

  // Final review: there is no API-less compatibility. A record without an
  // explicit API, and a failure without one, carry no replay identity at all --
  // never a wildcard.
  describe('API identity is never optional (final review)', () => {
    const apiLessRecord = {
      customType: 'router-encrypted-rejection',
      failedRef: 'openai-codex/gpt-5.5-codex',
      // No api: the legacy/API-less shape that must now be ignored.
      fingerprints: [
        signatureFingerprint(
          'openai-codex',
          'openai-responses',
          'gpt-5.5-codex',
          'sig-responses',
        ),
      ],
      at: 1700000000000,
    };

    it('ignores an API-less record even when its fingerprint matches', () => {
      expect(isEncryptedRejectionRecord(apiLessRecord)).toBe(false);
      // Old responses signature present, target now speaks completions.
      expect([
        ...exclusionsFor({
          messages: [signedAssistant('sig-responses')],
          records: [apiLessRecord] as never,
          resolveModelApi: () => 'openai-completions',
        }),
      ]).toEqual([]);
      // ... and it does not apply through a matching-API wildcard either.
      expect([
        ...exclusionsFor({
          messages: [signedAssistant('sig-responses')],
          records: [apiLessRecord] as never,
          resolveModelApi: () => 'openai-responses',
        }),
      ]).toEqual([]);
    });

    it('still excludes for an explicit same-API record (contrast)', () => {
      const explicit = {
        ...apiLessRecord,
        customType: 'router-encrypted-rejection' as const,
        api: 'openai-responses',
      };
      expect(isEncryptedRejectionRecord(explicit)).toBe(true);
      expect([
        ...exclusionsFor({
          messages: [signedAssistant('sig-responses')],
          records: [explicit],
          resolveModelApi: () => 'openai-responses',
        }),
      ]).toEqual(['openai-codex/gpt-5.5-codex']);
    });

    it('creates no durable record from a failure without an API', () => {
      const failure = classifyEncryptedReasoningFailure(
        encryptedError({ api: undefined }),
      )!;
      expect(failure.api).toBeUndefined();
      expect(
        buildEncryptedRejectionRecord(
          failure,
          [signedAssistant('sig-x')],
          1700000000000,
        ),
      ).toBeUndefined();
      expect(
        collectUnrecordedEncryptedRejections({
          messages: [
            signedAssistant('sig-x'),
            encryptedError({ api: undefined }),
          ],
          at: 1700000000000,
        }),
      ).toEqual([]);
      expect(
        observeEncryptedReasoningFailure({
          messages: [
            signedAssistant('sig-x'),
            encryptedError({ api: undefined }),
          ],
          at: 1700000000000,
        })?.record,
      ).toBeUndefined();
    });

    it('never fingerprints a signed block whose message carries no API', () => {
      expect(
        signedHistoryFingerprints([
          { ...signedAssistant('sig-x'), api: undefined },
        ]),
      ).toEqual([]);
      expect(
        signedHistoryFingerprints([signedAssistant('sig-x')]),
      ).toHaveLength(1);
    });
  });

  describe('conversationHasImage', () => {
    it('detects an image part anywhere in the conversation', () => {
      expect(
        conversationHasImage([
          { role: 'user', content: 'hi', timestamp: 1 },
          {
            role: 'user',
            content: [{ type: 'image', data: 'AAA', mimeType: 'image/png' }],
            timestamp: 2,
          },
        ]),
      ).toBe(true);
    });

    it('is false for text-only or malformed content', () => {
      expect(
        conversationHasImage([
          { role: 'user', content: 'hi', timestamp: 1 },
          { role: 'user', content: null },
          'nope',
        ]),
      ).toBe(false);
    });
  });

  describe('buildEncryptedReasoningEntry', () => {
    it('is a custom_message draft, never a literal user message', () => {
      expect(buildEncryptedReasoningEntry()).toMatchObject({
        type: 'custom_message',
        customType: ENCRYPTED_REASONING_CUSTOM_TYPE,
        display: false,
      });
    });
  });

  describe('evaluateBeforeSettleForEncryptedReasoning', () => {
    it('requests one continuation on a usable replacement, and arms the single notice', () => {
      const state = createEncryptedReasoningRecoveryState();
      const result = evaluateRecovery(
        beforeSettleEvent('error', [encryptedError()]),
        state,
        () => true,
      );

      expect(result).toMatchObject({
        failedRef: 'openai-codex/gpt-5.5-codex',
        continue: true,
        armNotice: true,
        entries: [
          {
            type: 'custom_message',
            customType: ENCRYPTED_REASONING_CUSTOM_TYPE,
          },
        ],
      });
      expect(state.attemptedRefs.has('openai-codex/gpt-5.5-codex')).toBe(true);
      expect(state.noticeArmed).toBe(true);
    });

    it('passes the conversation exclusions and messages to the replacement check', () => {
      const messages = [encryptedError()];
      let seen: {
        failedRef: string;
        excluded: string[];
        messages: unknown;
      } | null = null;
      evaluateRecovery(
        beforeSettleEvent('error', messages),
        createEncryptedReasoningRecoveryState(),
        (failedRef, excludedRefs, contextMessages) => {
          seen = {
            failedRef,
            excluded: [...excludedRefs],
            messages: contextMessages,
          };
          return true;
        },
      );
      expect(seen).not.toBeNull();
      expect(seen).toMatchObject({
        failedRef: 'openai-codex/gpt-5.5-codex',
        excluded: ['openai-codex/gpt-5.5-codex'],
        messages,
      });
    });

    // Bounded attempts through distinct remaining options: an episode may
    // continue away from several different models, but each fire is a
    // different failed model, and only the first one shows the notice.
    it('continues again after a second, different model fails, without a second notice', () => {
      const state = createEncryptedReasoningRecoveryState();
      const first = evaluateRecovery(
        beforeSettleEvent('error', [encryptedError()]),
        state,
        () => true,
      );
      expect(first?.armNotice).toBe(true);

      const second = evaluateRecovery(
        beforeSettleEvent('error', [
          encryptedError(),
          encryptedError({ provider: 'openai', model: 'gpt-5.5' }),
        ]),
        state,
        () => true,
      );
      expect(second).toMatchObject({
        failedRef: 'openai/gpt-5.5',
        continue: true,
        armNotice: false,
      });
    });

    it('never fires twice for the same failed model in one episode', () => {
      const state = createEncryptedReasoningRecoveryState();
      expect(
        evaluateRecovery(
          beforeSettleEvent('error', [encryptedError()]),
          state,
          () => true,
        ),
      ).toBeDefined();
      expect(
        evaluateRecovery(
          beforeSettleEvent('error', [encryptedError()]),
          state,
          () => true,
        ),
      ).toBeUndefined();
    });

    // Exhaustion: with no usable replacement left the recovery must stop and
    // leave the provider error visible rather than continue into a certain
    // repeat failure.
    it('does not fire when no usable replacement remains', () => {
      const state = createEncryptedReasoningRecoveryState();
      let calls = 0;
      const result = evaluateRecovery(
        beforeSettleEvent('error', [encryptedError()]),
        state,
        () => {
          calls++;
          return false;
        },
      );
      expect(result).toBeUndefined();
      expect(calls).toBe(1);
      expect(state.attemptedRefs.size).toBe(0);
      expect(state.noticeArmed).toBe(false);
    });

    it('never fires for a completed or aborted run, or an unrelated error', () => {
      for (const [outcome, message] of [
        ['completed', encryptedError()],
        ['aborted', encryptedError()],
        ['error', encryptedError({ errorMessage: 'insufficient_quota' })],
        ['error', encryptedError({ stopReason: 'aborted' })],
      ] as const) {
        expect(
          evaluateRecovery(
            beforeSettleEvent(outcome, [message]),
            createEncryptedReasoningRecoveryState(),
            () => true,
          ),
        ).toBeUndefined();
      }
    });

    it('stays silent on a malformed event or an empty transcript', () => {
      for (const event of [
        undefined,
        'boom',
        {},
        { outcome: 'error' },
        { outcome: 'error', context: {} },
        { outcome: 'error', context: { contextMessages: [] } },
      ]) {
        expect(
          evaluateRecovery(
            event,
            createEncryptedReasoningRecoveryState(),
            () => true,
          ),
        ).toBeUndefined();
      }
    });
  });

  describe('resetEncryptedReasoningRecoveryOnSettle', () => {
    it('clears the episode so a later run gets its own attempts and notice', () => {
      const state = createEncryptedReasoningRecoveryState();
      evaluateRecovery(
        beforeSettleEvent('error', [encryptedError()]),
        state,
        () => true,
      );
      expect(state.noticeArmed).toBe(true);

      resetEncryptedReasoningRecoveryOnSettle(state);
      expect(state.attemptedRefs.size).toBe(0);
      expect(state.noticeArmed).toBe(false);

      const next = evaluateRecovery(
        beforeSettleEvent('error', [encryptedError()]),
        state,
        () => true,
      );
      expect(next?.armNotice).toBe(true);
    });
  });
});
