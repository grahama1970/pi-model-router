import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createCooldownStore } from './cooldown';
import type { ThinkingLevel } from '@earendil-works/pi-agent-core';
import type { Api, Model } from '@earendil-works/pi-ai';
import type { ExtensionFactory } from '@earendil-works/pi-coding-agent';
import type {
  CustomSessionEntry,
  RouterExtensionAPI,
  RouterModelSelectEvent,
  RouterThinkingLevelSelectEvent,
  RouterTurnEndResult,
} from './types';
import routerExtension from './index';
import { signatureFingerprint as signatureFingerprintForTest } from './encrypted-reasoning';

const _extensionFactoryTypeCheck: ExtensionFactory = routerExtension;
void _extensionFactoryTypeCheck;

vi.mock('./config', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./config')>();
  return {
    ...actual,
    loadRouterConfig: () => ({
      config: {
        profiles: {
          balanced: {
            high: { model: 'openai/gpt-4o' },
            // A fallback is present so the auto-continue tests below have a
            // real usable replacement candidate to find in this tier.
            medium: {
              model: 'openai/gpt-4o-mini',
              fallbacks: [{ model: 'openai/gpt-4o-mini-fallback' }],
            },
          },
        },
      },
      warnings: [],
    }),
    profileNames: () => ['balanced'],
    resolveProfileName: (_config: unknown, name: unknown) =>
      name === 'balanced' ? 'balanced' : undefined,
    resolveContextWindow: () => 100000,
    resolveMaxTokens: () => 4000,
    resolveThinkingLevels: () => undefined,
    collectProfileThinkingLevels: () => new Set<string>(),
    getUnsupportedTiers: (): string[] => [],
  };
});

const createTestModel = (overrides: Partial<Model<Api>> = {}): Model<Api> => ({
  id: 'balanced',
  name: 'Balanced',
  api: 'openai-completions',
  provider: 'router',
  baseUrl: 'https://router.local',
  input: ['text', 'image'],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  reasoning: true,
  contextWindow: 100000,
  maxTokens: 4000,
  ...overrides,
});

const buildMockCtx = () => ({
  cwd: '/mock/cwd',
  modelRegistry: {
    find: vi
      .fn<(provider: string, modelId: string) => Model<Api> | undefined>()
      .mockReturnValue(createTestModel({ provider: 'router', id: 'balanced' })),
  },
  model: {
    provider: 'router',
    id: 'balanced',
  } as RouterModelSelectEvent['model'] | undefined,
  sessionManager: {
    getBranch: (): CustomSessionEntry[] => [],
    getEntries: (): CustomSessionEntry[] => [],
  },
  ui: {
    setStatus: vi.fn((_key: string, _text: string | undefined) => {}),
    setWidget: vi.fn((_key: string, _content: string[] | undefined) => {}),
    theme: { fg: (_c: 'dim', text: string) => text },
    notify: vi.fn(
      (_message: string, _type?: 'info' | 'warning' | 'error') => {},
    ),
  },
});

type MockCtx = ReturnType<typeof buildMockCtx>;

type RouterTestMessage = {
  role?: string;
  stopReason?: string;
  errorMessage?: string;
  provider?: string;
  model?: string;
  timestamp?: number;
  usage?: { cost?: { total?: number } };
};

type FullRouterEvent = RouterModelSelectEvent &
  RouterThinkingLevelSelectEvent & {
    outcome?: string;
    message?: RouterTestMessage;
    // Mirrors the real `AgentBeforeSettleEvent`'s `context.contextMessages`
    // -- the shape `evaluateBeforeSettleForAutoContinue` actually reads.
    context?: { contextMessages?: RouterTestMessage[] };
    // Mirrors the real `AgentSettledEvent`'s `aborted` field.
    aborted?: boolean;
  };

type RegisteredHandler = (
  event: FullRouterEvent,
  ctx: MockCtx,
) => Promise<void | RouterTurnEndResult> | void | RouterTurnEndResult;

type TriggerHandler = (
  event: Partial<FullRouterEvent>,
  ctx: MockCtx,
) => Promise<void | RouterTurnEndResult> | void | RouterTurnEndResult;

const createMockPi = (eventListeners: Record<string, TriggerHandler[]>) => ({
  registerVirtualModel: vi.fn<
    RouterExtensionAPI<MockCtx>['registerVirtualModel']
  >(() => {}),
  unregisterVirtualModel: vi.fn(() => {}),
  registerProvider: vi.fn(() => {}),
  registerCommand: vi.fn(() => {}),
  setModel: vi.fn(async () => true),
  setThinkingLevel: vi.fn((_level: ThinkingLevel) => {}),
  appendEntry: vi.fn(() => {}),
  on: vi.fn((event: string, handler: RegisteredHandler) => {
    if (!eventListeners[event]) {
      eventListeners[event] = [];
    }
    eventListeners[event].push((partialEvent, ctx) =>
      handler(
        {
          model: { provider: 'router', id: 'balanced' },
          level: 'off',
          ...partialEvent,
        },
        ctx,
      ),
    );
  }),
});

describe('index.ts (orchestrator)', () => {
  let mockPi: ReturnType<typeof createMockPi> & RouterExtensionAPI<MockCtx>;
  let eventListeners: Record<string, TriggerHandler[]> = {};

  beforeEach(() => {
    eventListeners = {};
    mockPi = createMockPi(eventListeners);
  });

  it('should initialize and register commands, virtual models, and event hooks', () => {
    routerExtension(mockPi);

    expect(mockPi.registerVirtualModel).toHaveBeenCalledWith(
      expect.objectContaining({
        provider: 'router',
        id: 'balanced',
      }),
    );
    expect(mockPi.registerCommand).toHaveBeenCalledWith(
      'router',
      expect.any(Object),
    );
    expect(mockPi.on).toHaveBeenCalledWith(
      'session_start',
      expect.any(Function),
    );
    expect(mockPi.on).toHaveBeenCalledWith(
      'session_tree',
      expect.any(Function),
    );
    expect(mockPi.on).toHaveBeenCalledWith(
      'model_select',
      expect.any(Function),
    );
    expect(mockPi.on).toHaveBeenCalledWith('turn_end', expect.any(Function));
    expect(mockPi.on).toHaveBeenCalledWith(
      'agent_before_settle',
      expect.any(Function),
    );
  });

  it('should restore state from session on session_start hook', async () => {
    routerExtension(mockPi);

    const mockCtx = buildMockCtx();
    mockCtx.model = { provider: 'openai', id: 'gpt-4o' };
    mockCtx.sessionManager.getBranch = () => [
      {
        type: 'custom',
        customType: 'router-state',
        data: {
          enabled: true,
          selectedProfile: 'balanced',
          pinByProfile: { balanced: 'high' },
          thinkingByProfile: {},
          debugEnabled: true,
          widgetEnabled: true,
          accumulatedCost: 0.012,
          timestamp: Date.now(),
        },
      },
    ];

    // Trigger session_start
    const sessionStartHandlers = eventListeners['session_start'] || [];
    for (const handler of sessionStartHandlers) {
      await handler({}, mockCtx);
    }

    expect(mockCtx.ui.setStatus).toHaveBeenCalled();
    expect(mockPi.setModel).toHaveBeenCalledWith(
      expect.objectContaining({ provider: 'router', id: 'balanced' }),
    );
  });

  it('should handle model select hook', async () => {
    routerExtension(mockPi);

    const mockCtx = buildMockCtx();

    // Trigger session_start to initialize first
    const sessionStartHandlers = eventListeners['session_start'] || [];
    for (const handler of sessionStartHandlers) {
      await handler({}, mockCtx);
    }

    const modelSelectHandlers = eventListeners['model_select'] || [];
    for (const handler of modelSelectHandlers) {
      await handler({ model: { provider: 'router', id: 'balanced' } }, mockCtx);
    }

    expect(mockCtx.ui.setStatus).toHaveBeenCalled();
  });

  describe('model_select event', () => {
    it('should set routerEnabled=false and record lastNonRouterModel for non-router model', async () => {
      routerExtension(mockPi);

      const mockCtx = buildMockCtx();

      // Initialize via session_start
      const sessionStartHandlers = eventListeners['session_start'] || [];
      for (const handler of sessionStartHandlers) {
        await handler({}, mockCtx);
      }

      // Clear appendEntry calls from initialization
      mockPi.appendEntry.mockClear();

      // Select a non-router model
      const modelSelectHandlers = eventListeners['model_select'] || [];
      for (const handler of modelSelectHandlers) {
        await handler(
          { model: { provider: 'anthropic', id: 'claude-3-5-sonnet' } },
          mockCtx,
        );
      }

      // Should have persisted state (routerEnabled=false, lastNonRouterModel set)
      expect(mockPi.appendEntry).toHaveBeenCalledWith(
        'router-state',
        expect.objectContaining({
          enabled: false,
          lastNonRouterModel: 'anthropic/claude-3-5-sonnet',
        }),
      );
    });

    it('should be a no-op before session_start (isInitialized=false)', async () => {
      routerExtension(mockPi);

      const mockCtx = buildMockCtx();

      // Clear calls from constructor
      mockPi.appendEntry.mockClear();
      mockCtx.ui.setStatus.mockClear();

      // Trigger model_select WITHOUT session_start first
      const modelSelectHandlers = eventListeners['model_select'] || [];
      for (const handler of modelSelectHandlers) {
        await handler(
          { model: { provider: 'anthropic', id: 'claude-3-5-sonnet' } },
          mockCtx,
        );
      }

      // Should NOT have persisted state or updated status
      expect(mockPi.appendEntry).not.toHaveBeenCalled();
    });
  });

  describe('thinking_level_select event', () => {
    it('should apply thinking level as all-tier override for active profile', async () => {
      routerExtension(mockPi);

      const mockCtx = buildMockCtx();

      // Initialize via session_start (sets routerEnabled=true, selectedProfile='balanced')
      const sessionStartHandlers = eventListeners['session_start'] || [];
      for (const handler of sessionStartHandlers) {
        await handler({}, mockCtx);
      }

      // Clear appendEntry calls from initialization
      mockPi.appendEntry.mockClear();

      // Trigger thinking_level_select
      const thinkingHandlers = eventListeners['thinking_level_select'] || [];
      for (const handler of thinkingHandlers) {
        handler({ level: 'high' }, mockCtx);
      }

      // Should persist state with thinking overrides for all tiers
      expect(mockPi.appendEntry).toHaveBeenCalledWith(
        'router-state',
        expect.objectContaining({
          thinkingByProfile: {
            balanced: { high: 'high', medium: 'high', low: 'high' },
          },
        }),
      );
    });

    it('should be ignored when router is not enabled', async () => {
      routerExtension(mockPi);

      const mockCtx = buildMockCtx();
      // Set model to non-router so routerEnabled stays false after restore
      mockCtx.model = { provider: 'openai', id: 'gpt-4o' };

      // Initialize via session_start with non-router model
      const sessionStartHandlers = eventListeners['session_start'] || [];
      for (const handler of sessionStartHandlers) {
        await handler({}, mockCtx);
      }

      mockPi.appendEntry.mockClear();

      // Trigger thinking_level_select
      const thinkingHandlers = eventListeners['thinking_level_select'] || [];
      for (const handler of thinkingHandlers) {
        handler({ level: 'medium' }, mockCtx);
      }

      // Should NOT have called appendEntry (no persist because early return)
      expect(mockPi.appendEntry).not.toHaveBeenCalled();
    });

    it('should be ignored before initialization', () => {
      routerExtension(mockPi);

      const mockCtx = buildMockCtx();

      mockPi.appendEntry.mockClear();

      // Trigger thinking_level_select without session_start
      const thinkingHandlers = eventListeners['thinking_level_select'] || [];
      for (const handler of thinkingHandlers) {
        handler({ level: 'low' }, mockCtx);
      }

      expect(mockPi.appendEntry).not.toHaveBeenCalled();
    });

    it('should ignore thinking_level_select emitted by pi.setModel during session_start (defaultThinkingLevel in settings.json)', async () => {
      const mockCtx = buildMockCtx();

      // Simulate Pi >= 0.85.0 behavior where pi.setModel emits thinking_level_select
      // with defaultThinkingLevel from settings.json
      mockPi.setModel = vi.fn(async () => {
        const thinkingHandlers = eventListeners['thinking_level_select'] || [];
        for (const handler of thinkingHandlers) {
          await handler({ level: 'medium' }, mockCtx);
        }
        return true;
      });

      routerExtension(mockPi);

      const sessionStartHandlers = eventListeners['session_start'] || [];
      for (const handler of sessionStartHandlers) {
        await handler({}, mockCtx);
      }

      expect(mockPi.appendEntry).toHaveBeenCalledWith(
        'router-state',
        expect.objectContaining({
          enabled: true,
          selectedProfile: 'balanced',
          thinkingByProfile: {},
        }),
      );
    });

    it('should ignore thinking_level_select when ctx.model does not match the active router profile', async () => {
      routerExtension(mockPi);

      const mockCtx = buildMockCtx();

      const sessionStartHandlers = eventListeners['session_start'] || [];
      for (const handler of sessionStartHandlers) {
        await handler({}, mockCtx);
      }

      mockPi.appendEntry.mockClear();

      // Simulate external model switch where Pi updates ctx.model and emits
      // thinking_level_select before emitting model_select
      mockCtx.model = { provider: 'anthropic', id: 'claude-3-5-sonnet' };
      const thinkingHandlers = eventListeners['thinking_level_select'] || [];
      for (const handler of thinkingHandlers) {
        await handler({ level: 'medium' }, mockCtx);
      }

      expect(mockPi.appendEntry).not.toHaveBeenCalled();
    });

    it('should ignore async thinking_level_select echo emitted by syncPiThinkingLevel during routing (issue #29)', async () => {
      const mockCtx = buildMockCtx();

      // Simulate Pi's fire-and-forget `void this._extensionRunner.emit(...)`
      // delayed by a microtask/timer (e.g. when another extension is loaded before pi-model-router)
      mockPi.setThinkingLevel = vi.fn((level: ThinkingLevel) => {
        setTimeout(() => {
          const thinkingHandlers =
            eventListeners['thinking_level_select'] || [];
          for (const handler of thinkingHandlers) {
            void handler({ level }, mockCtx);
          }
        }, 0);
      });

      routerExtension(mockPi);

      const sessionStartHandlers = eventListeners['session_start'] || [];
      for (const handler of sessionStartHandlers) {
        await handler({}, mockCtx);
      }

      const vmDef = mockPi.registerVirtualModel.mock.lastCall?.[0];
      expect(vmDef).toBeDefined();

      mockPi.appendEntry.mockClear();

      await vmDef!.route(
        {
          model: { id: 'balanced', provider: 'router' } as Model<Api>,
          thinkingLevel: 'high',
          reason: 'user',
          messages: [{ role: 'user', content: 'just reply yes', timestamp: 0 }],
        },
        mockCtx,
      );

      // Wait for async setTimeout(..., 0) echo to drain
      await new Promise((resolve) => setTimeout(resolve, 10));

      expect(mockPi.setThinkingLevel).toHaveBeenCalled();
      expect(mockPi.appendEntry).toHaveBeenLastCalledWith(
        'router-state',
        expect.objectContaining({ thinkingByProfile: {} }),
      );
    });
  });

  describe('restoreStateFromSession edge cases', () => {
    it('should handle fresh session with no saved router-state entries without calling setModel redundantly', async () => {
      routerExtension(mockPi);

      const mockCtx = buildMockCtx();
      // Empty session, ctx.model is already router/balanced
      mockCtx.sessionManager.getBranch = () => [];

      const sessionStartHandlers = eventListeners['session_start'] || [];
      for (const handler of sessionStartHandlers) {
        await handler({}, mockCtx);
      }

      // Should NOT call setModel when ctx.model is already router/balanced
      expect(mockPi.setModel).not.toHaveBeenCalled();
      // Should persist initial state
      expect(mockPi.appendEntry).toHaveBeenCalledWith(
        'router-state',
        expect.objectContaining({
          enabled: true,
          selectedProfile: 'balanced',
          thinkingByProfile: {},
        }),
      );
    });

    it('should handle failed model restoration (setModel returns false)', async () => {
      mockPi.setModel = vi.fn().mockResolvedValue(false);
      routerExtension(mockPi);

      const mockCtx = buildMockCtx();
      mockCtx.model = { provider: 'openai', id: 'gpt-4o' };
      mockCtx.sessionManager.getBranch = () => [
        {
          type: 'custom',
          customType: 'router-state',
          data: {
            enabled: true,
            selectedProfile: 'balanced',
            pinByProfile: {},
            thinkingByProfile: {},
            timestamp: Date.now(),
          },
        },
      ];

      const sessionStartHandlers = eventListeners['session_start'] || [];
      for (const handler of sessionStartHandlers) {
        await handler({}, mockCtx);
      }

      // Should notify about failure
      expect(mockCtx.ui.notify).toHaveBeenCalledWith(
        expect.stringContaining('Failed to restore router/balanced'),
        'warning',
      );

      // routerEnabled should be set to false
      expect(mockPi.appendEntry).toHaveBeenCalledWith(
        'router-state',
        expect.objectContaining({
          enabled: false,
        }),
      );
    });

    it('should handle router model unavailable in registry', async () => {
      routerExtension(mockPi);

      const mockCtx = buildMockCtx();
      // Registry returns undefined for router model
      mockCtx.modelRegistry.find = vi.fn().mockReturnValue(undefined);
      mockCtx.sessionManager.getBranch = () => [
        {
          type: 'custom',
          customType: 'router-state',
          data: {
            enabled: true,
            selectedProfile: 'balanced',
            pinByProfile: {},
            thinkingByProfile: {},
            timestamp: Date.now(),
          },
        },
      ];

      const sessionStartHandlers = eventListeners['session_start'] || [];
      for (const handler of sessionStartHandlers) {
        await handler({}, mockCtx);
      }

      // Should notify about unavailability
      expect(mockCtx.ui.notify).toHaveBeenCalledWith(
        expect.stringContaining('Unable to restore router/balanced'),
        'warning',
      );

      // routerEnabled should be false
      expect(mockPi.appendEntry).toHaveBeenCalledWith(
        'router-state',
        expect.objectContaining({
          enabled: false,
        }),
      );
    });

    it('should migrate legacy pinTier field to pinByProfile', async () => {
      routerExtension(mockPi);

      const mockCtx = buildMockCtx();
      mockCtx.sessionManager.getBranch = () => [
        {
          type: 'custom',
          customType: 'router-state',
          data: {
            enabled: true,
            selectedProfile: 'balanced',
            pinTier: 'medium',
            pinByProfile: {},
            thinkingByProfile: {},
            timestamp: Date.now(),
          },
        },
      ];

      const sessionStartHandlers = eventListeners['session_start'] || [];
      for (const handler of sessionStartHandlers) {
        await handler({}, mockCtx);
      }

      // Should persist with the legacy pinTier migrated into pinByProfile
      expect(mockPi.appendEntry).toHaveBeenCalledWith(
        'router-state',
        expect.objectContaining({
          pinByProfile: expect.objectContaining({ balanced: 'medium' }),
        }),
      );
    });
  });

  describe('turn_end event', () => {
    it('should persist state and update status but NOT restore model when router is not enabled', async () => {
      routerExtension(mockPi);

      const mockCtx = buildMockCtx();
      // Set model to non-router
      mockCtx.model = { provider: 'openai', id: 'gpt-4o' };

      // Initialize via session_start with non-router model
      const sessionStartHandlers = eventListeners['session_start'] || [];
      for (const handler of sessionStartHandlers) {
        await handler({}, mockCtx);
      }

      // Clear mocks after initialization
      mockPi.setModel.mockClear();
      mockPi.appendEntry.mockClear();
      mockCtx.ui.setStatus.mockClear();

      // Trigger turn_end
      const turnEndHandlers = eventListeners['turn_end'] || [];
      for (const handler of turnEndHandlers) {
        await handler({}, mockCtx);
      }

      // Should NOT call setModel (router is not enabled)
      expect(mockPi.setModel).not.toHaveBeenCalled();

      // Should still update status
      expect(mockCtx.ui.setStatus).toHaveBeenCalled();

      // persistState is called, but snapshot deduplication may skip appendEntry
      // since state hasn't changed since session_start. The key assertion is
      // that setModel was NOT called (router is not enabled).
    });
  });

  describe('agent_before_settle event: auto-continue', () => {
    // record() now writes to disk the moment auto-continue fires (see
    // extensions/index.ts), so every test in this block gets its own tmp
    // cooldown dir via PI_MODEL_ROUTER_COOLDOWNS -- createCooldownStore()'s
    // default dir (`defaultCooldownDir()`) is read lazily at
    // routerExtension(mockPi) call time, so setting the env var before that
    // call is enough; nothing in this file ever touches the real
    // ~/.local/state/pi-model-router/cooldowns directory.
    let cooldownDir: string;
    let previousCooldownEnv: string | undefined;

    beforeEach(() => {
      previousCooldownEnv = process.env.PI_MODEL_ROUTER_COOLDOWNS;
      cooldownDir = mkdtempSync(join(tmpdir(), 'pmr-index-cooldown-'));
      process.env.PI_MODEL_ROUTER_COOLDOWNS = cooldownDir;
    });

    afterEach(() => {
      if (previousCooldownEnv === undefined) {
        delete process.env.PI_MODEL_ROUTER_COOLDOWNS;
      } else {
        process.env.PI_MODEL_ROUTER_COOLDOWNS = previousCooldownEnv;
      }
      rmSync(cooldownDir, { recursive: true, force: true });
    });

    const assistantLimitError = (overrides: Record<string, unknown> = {}) => ({
      role: 'assistant',
      stopReason: 'error',
      provider: 'openai',
      model: 'gpt-4o-mini',
      errorMessage: 'insufficient_quota',
      ...overrides,
    });

    const beforeSettleEvent = (
      outcome: string,
      message: RouterTestMessage,
    ): Partial<FullRouterEvent> => ({
      outcome,
      context: { contextMessages: [message] },
    });

    const initRouterEnabled = async (mockCtx: MockCtx) => {
      // buildMockCtx() defaults to model router/balanced, so session_start
      // alone enables the router -- matching "a turn routed by router/*".
      // Seeds a persisted `lastDecision` on the medium tier (the tier
      // `assistantLimitError`'s failed model belongs to): the real
      // `agent_before_settle` handler only knows which profile/tier to check
      // for a usable replacement from `state.lastDecision`, which a real
      // session always has by the time a routed turn can fail.
      mockCtx.sessionManager.getBranch = () => [
        {
          type: 'custom',
          customType: 'router-state',
          data: {
            enabled: true,
            selectedProfile: 'balanced',
            lastDecision: {
              profile: 'balanced',
              tier: 'medium',
              phase: 'implementation',
              targetProvider: 'openai',
              targetModelId: 'gpt-4o-mini',
              targetLabel: 'openai/gpt-4o-mini',
              reasoning: 'test',
              thinking: 'off',
              timestamp: Date.now(),
            },
            timestamp: Date.now(),
          },
        },
      ];
      const sessionStartHandlers = eventListeners['session_start'] || [];
      for (const handler of sessionStartHandlers) {
        await handler({}, mockCtx);
      }
    };

    // Regression: without the agent_before_settle wiring in index.ts, this
    // returns undefined and the run simply ends with the error shown to the
    // user -- no continuation, no notice. (A handler on turn_end alone
    // cannot fix this: Pi's finishTurn discards turn_end's `continue`
    // whenever its own retry/fallback decision was already "end", which it
    // always is for a non-retryable error -- see
    // extensions/autocontinue.ts's module doc comment.)
    it('requests exactly one continuation with a custom_message entry on a non-retryable usage-limit error', async () => {
      routerExtension(mockPi);
      const mockCtx = buildMockCtx();
      await initRouterEnabled(mockCtx);

      const handlers = eventListeners['agent_before_settle'] || [];
      let result: unknown;
      for (const handler of handlers) {
        result = await handler(
          beforeSettleEvent('error', assistantLimitError()),
          mockCtx,
        );
      }

      expect(result).toMatchObject({
        continue: true,
        entries: [
          { type: 'custom_message', customType: 'router-auto-continue' },
        ],
      });
    });

    // The actual fix under test: the failed model's cooldown is written to
    // disk at fire time, not left for route()'s own transcript scan on the
    // continuation -- which would miss it entirely in a separate process
    // that resumes this session later (see extensions/index.ts's
    // agent_before_settle handler comment).
    it('records a cooldown for the failed model on disk when it fires', async () => {
      routerExtension(mockPi);
      const mockCtx = buildMockCtx();
      await initRouterEnabled(mockCtx);

      const handlers = eventListeners['agent_before_settle'] || [];
      for (const handler of handlers) {
        await handler(
          beforeSettleEvent(
            'error',
            assistantLimitError({ timestamp: 1700000000000 }),
          ),
          mockCtx,
        );
      }

      // A fresh store pointed at the same directory sees it as a foreign
      // file, proving the write actually reached disk rather than staying
      // only in this process's in-memory state.
      const verifyStore = createCooldownStore(cooldownDir);
      const entry = verifyStore.load(Date.now()).entries['openai/gpt-4o-mini'];
      expect(entry).toBeDefined();
      expect(entry?.source).toBe('transcript');
      expect(entry?.recorded_at).toBe(new Date(1700000000000).toISOString());
    });

    it('never chains: a second consecutive error does not request another continuation', async () => {
      routerExtension(mockPi);
      const mockCtx = buildMockCtx();
      await initRouterEnabled(mockCtx);

      const handlers = eventListeners['agent_before_settle'] || [];
      let first: unknown;
      let second: unknown;
      for (const handler of handlers) {
        first = await handler(
          beforeSettleEvent('error', assistantLimitError()),
          mockCtx,
        );
        second = await handler(
          beforeSettleEvent(
            'error',
            assistantLimitError({
              provider: 'google',
              model: 'gemini-1.5-flash',
            }),
          ),
          mockCtx,
        );
      }

      expect(first).toMatchObject({ continue: true });
      expect(second).toBeUndefined();
    });

    it("does not fire for a retryable 429, leaving it to Pi's own retry/fallback chain", async () => {
      routerExtension(mockPi);
      const mockCtx = buildMockCtx();
      await initRouterEnabled(mockCtx);

      const handlers = eventListeners['agent_before_settle'] || [];
      let result: unknown;
      for (const handler of handlers) {
        result = await handler(
          beforeSettleEvent(
            'error',
            assistantLimitError({ errorMessage: '429 too many requests' }),
          ),
          mockCtx,
        );
      }

      expect(result).toBeUndefined();
    });

    it('never fires for a user abort', async () => {
      routerExtension(mockPi);
      const mockCtx = buildMockCtx();
      await initRouterEnabled(mockCtx);

      const handlers = eventListeners['agent_before_settle'] || [];
      let result: unknown;
      for (const handler of handlers) {
        result = await handler(
          beforeSettleEvent(
            'aborted',
            assistantLimitError({ stopReason: 'aborted' }),
          ),
          mockCtx,
        );
      }

      expect(result).toBeUndefined();
    });

    it('never fires when the turn was not routed by router/*', async () => {
      routerExtension(mockPi);
      const mockCtx = buildMockCtx();
      mockCtx.model = { provider: 'openai', id: 'gpt-4o' };

      const sessionStartHandlers = eventListeners['session_start'] || [];
      for (const handler of sessionStartHandlers) {
        await handler({}, mockCtx);
      }

      const handlers = eventListeners['agent_before_settle'] || [];
      let result: unknown;
      for (const handler of handlers) {
        result = await handler(
          beforeSettleEvent('error', assistantLimitError()),
          mockCtx,
        );
      }

      expect(result).toBeUndefined();
    });

    // The regression this guards: resetting the chain guard on a non-error
    // agent_before_settle (instead of agent_settled) would let a later
    // failure in the SAME run re-fire; see extensions/autocontinue.ts's
    // `resetAutoContinueGuardOnSettle` doc comment.
    it('agent_settled resets the chain guard so a new run can auto-continue again', async () => {
      routerExtension(mockPi);
      const mockCtx = buildMockCtx();
      await initRouterEnabled(mockCtx);

      const beforeSettleHandlers = eventListeners['agent_before_settle'] || [];
      const settledHandlers = eventListeners['agent_settled'] || [];

      let first: unknown;
      for (const handler of beforeSettleHandlers) {
        first = await handler(
          beforeSettleEvent('error', assistantLimitError()),
          mockCtx,
        );
      }
      expect(first).toMatchObject({ continue: true });

      // The continuation succeeds, but Pi's run loop can still dispatch
      // agent_before_settle again within the same run.
      let midRun: unknown;
      for (const handler of beforeSettleHandlers) {
        midRun = await handler(
          beforeSettleEvent(
            'completed',
            assistantLimitError({ stopReason: 'stop' }),
          ),
          mockCtx,
        );
      }
      expect(midRun).toBeUndefined();

      // A later turn in the SAME run fails on a different model: still
      // chained, no second fire.
      let sameRunSecond: unknown;
      for (const handler of beforeSettleHandlers) {
        sameRunSecond = await handler(
          beforeSettleEvent(
            'error',
            assistantLimitError({
              provider: 'google',
              model: 'gemini-1.5-flash',
            }),
          ),
          mockCtx,
        );
      }
      expect(sameRunSecond).toBeUndefined();

      // The run actually settles.
      for (const handler of settledHandlers) {
        await handler({ aborted: false }, mockCtx);
      }

      // A later, independent run's failure gets its own fresh budget.
      let newRun: unknown;
      for (const handler of beforeSettleHandlers) {
        newRun = await handler(
          beforeSettleEvent(
            'error',
            assistantLimitError({
              provider: 'google',
              model: 'gemini-1.5-flash',
            }),
          ),
          mockCtx,
        );
      }
      expect(newRun).toMatchObject({ continue: true });
    });

    // The regression this guards: without a usable replacement check, this
    // would fire and re-request the SAME sole model for the 'high' tier
    // (single model, no fallback), hitting the identical limit error again.
    it('does not fire when the failed model has no usable replacement in its tier', async () => {
      routerExtension(mockPi);
      const mockCtx = buildMockCtx();
      mockCtx.sessionManager.getBranch = () => [
        {
          type: 'custom',
          customType: 'router-state',
          data: {
            enabled: true,
            selectedProfile: 'balanced',
            lastDecision: {
              profile: 'balanced',
              tier: 'high',
              phase: 'implementation',
              targetProvider: 'openai',
              targetModelId: 'gpt-4o',
              targetLabel: 'openai/gpt-4o',
              reasoning: 'test',
              thinking: 'off',
              timestamp: Date.now(),
            },
            timestamp: Date.now(),
          },
        },
      ];
      const sessionStartHandlers = eventListeners['session_start'] || [];
      for (const handler of sessionStartHandlers) {
        await handler({}, mockCtx);
      }

      const handlers = eventListeners['agent_before_settle'] || [];
      let result: unknown;
      for (const handler of handlers) {
        result = await handler(
          beforeSettleEvent(
            'error',
            assistantLimitError({ provider: 'openai', model: 'gpt-4o' }),
          ),
          mockCtx,
        );
      }

      expect(result).toBeUndefined();
    });

    // The regression this guards: `updateStatus`/`syncPiThinkingLevel`
    // throwing in route() (stale extension context) used to leave the
    // pending notice unconsumed into a later, unrelated route() call, which
    // would then fire a notice naming the wrong failed model. agent_settled
    // now drops any notice route() never got to consume at all -- e.g. the
    // continuation itself is cancelled before its own route() call runs.
    it('agent_settled clears a pending auto-continue notice route() never consumed', async () => {
      routerExtension(mockPi);
      const mockCtx = buildMockCtx();
      await initRouterEnabled(mockCtx);

      const beforeSettleHandlers = eventListeners['agent_before_settle'] || [];
      const settledHandlers = eventListeners['agent_settled'] || [];

      for (const handler of beforeSettleHandlers) {
        await handler(
          beforeSettleEvent('error', assistantLimitError()),
          mockCtx,
        );
      }

      // The continuation never reaches route() (e.g. cancelled) -- the run
      // settles with the pending notice still unconsumed.
      for (const handler of settledHandlers) {
        await handler({ aborted: true }, mockCtx);
      }

      const vmDef = mockPi.registerVirtualModel.mock.lastCall?.[0];
      expect(vmDef).toBeDefined();
      (mockCtx.ui.notify as ReturnType<typeof vi.fn>).mockClear();

      // A later, unrelated route() call must never fire a stale notice naming
      // the wrong failed model.
      await vmDef!.route(
        {
          model: { id: 'balanced', provider: 'router' } as Model<Api>,
          thinkingLevel: 'high',
          reason: 'user',
          messages: [{ role: 'user', content: 'hello', timestamp: 0 }],
        },
        mockCtx,
      );

      expect(mockCtx.ui.notify).not.toHaveBeenCalled();
    });
  });

  describe('agent_before_settle event: encrypted reasoning recovery', () => {
    const CODEX_REJECTION =
      '400: {"error":{"message":"Invalid \'input[3].encrypted_content\': Encrypted content could not be verified.","code":"invalid_encrypted_content"}}';

    const assistantEncryptedError = (
      overrides: Record<string, unknown> = {},
    ) => ({
      role: 'assistant',
      stopReason: 'error',
      provider: 'openai',
      model: 'gpt-4o-mini',
      api: 'openai-completions',
      errorMessage: CODEX_REJECTION,
      ...overrides,
    });

    const beforeSettleEvent = (
      outcome: string,
      contextMessages: RouterTestMessage[],
    ): Partial<FullRouterEvent> => ({
      outcome,
      context: { contextMessages },
    });

    const triggerBeforeSettle = async (
      event: Partial<FullRouterEvent>,
      mockCtx: MockCtx,
    ) => {
      const handlers = eventListeners['agent_before_settle'] || [];
      let result: unknown;
      for (const handler of handlers) {
        result = await handler(event, mockCtx);
      }
      return result;
    };

    const triggerSettled = async (mockCtx: MockCtx) => {
      for (const handler of eventListeners['agent_settled'] || []) {
        await handler({}, mockCtx);
      }
    };

    // Seeds the persisted lastDecision on the medium tier (the tier the
    // failed model belongs to) exactly like the auto-continue block above:
    // the handler only knows which profile/tier to check from
    // state.lastDecision.
    const initRouterEnabled = async (mockCtx: MockCtx) => {
      mockCtx.sessionManager.getBranch = () => [
        {
          type: 'custom',
          customType: 'router-state',
          data: {
            enabled: true,
            selectedProfile: 'balanced',
            lastDecision: {
              profile: 'balanced',
              tier: 'medium',
              phase: 'implementation',
              targetProvider: 'openai',
              targetModelId: 'gpt-4o-mini',
              targetLabel: 'openai/gpt-4o-mini',
              reasoning: 'test',
              thinking: 'off',
              timestamp: Date.now(),
            },
            timestamp: Date.now(),
          },
        },
      ];
      for (const handler of eventListeners['session_start'] || []) {
        await handler({}, mockCtx);
      }
    };

    // The regression: without this second recovery path the run simply ends
    // with the rejection shown to the user, even though another configured
    // model can answer (the semantic equivalent of switching models by hand).
    it('requests one continuation with an encrypted-reasoning entry on a rejected replay', async () => {
      routerExtension(mockPi);
      const mockCtx = buildMockCtx();
      await initRouterEnabled(mockCtx);

      const result = await triggerBeforeSettle(
        beforeSettleEvent('error', [assistantEncryptedError()]),
        mockCtx,
      );

      expect(result).toMatchObject({
        continue: true,
        entries: [
          {
            type: 'custom_message',
            customType: 'router-encrypted-reasoning-recovery',
          },
        ],
      });
    });

    it('never fires twice for the same failed model within one run', async () => {
      routerExtension(mockPi);
      const mockCtx = buildMockCtx();
      await initRouterEnabled(mockCtx);

      const first = await triggerBeforeSettle(
        beforeSettleEvent('error', [assistantEncryptedError()]),
        mockCtx,
      );
      expect(first).toBeDefined();

      const second = await triggerBeforeSettle(
        beforeSettleEvent('error', [assistantEncryptedError()]),
        mockCtx,
      );
      expect(second).toBeUndefined();
    });

    it('continues away from a second, different model that also rejects the payload', async () => {
      routerExtension(mockPi);
      const mockCtx = buildMockCtx();
      await initRouterEnabled(mockCtx);

      await triggerBeforeSettle(
        beforeSettleEvent('error', [assistantEncryptedError()]),
        mockCtx,
      );
      const result = await triggerBeforeSettle(
        beforeSettleEvent('error', [
          assistantEncryptedError(),
          assistantEncryptedError({ model: 'gpt-4o' }),
        ]),
        mockCtx,
      );

      expect(result).toMatchObject({ continue: true });
    });

    // Exhaustion: once no other usable candidate is left, the run must end
    // with the rejection visible instead of looping.
    it('stays silent when no usable replacement remains', async () => {
      routerExtension(mockPi);
      const mockCtx = buildMockCtx();
      await initRouterEnabled(mockCtx);
      mockCtx.modelRegistry.find = vi.fn(() => undefined);

      const result = await triggerBeforeSettle(
        beforeSettleEvent('error', [assistantEncryptedError()]),
        mockCtx,
      );

      expect(result).toBeUndefined();
    });

    it('stays silent for an unrelated error and for an abort', async () => {
      routerExtension(mockPi);
      const mockCtx = buildMockCtx();
      await initRouterEnabled(mockCtx);

      expect(
        await triggerBeforeSettle(
          beforeSettleEvent('error', [
            assistantEncryptedError({
              errorMessage: '400: {"error":{"message":"Unknown parameter"}}',
            }),
          ]),
          mockCtx,
        ),
      ).toBeUndefined();

      expect(
        await triggerBeforeSettle(
          beforeSettleEvent('aborted', [assistantEncryptedError()]),
          mockCtx,
        ),
      ).toBeUndefined();
    });

    // The usage-limit path keeps its exact behavior and its own entry type;
    // an encrypted-reasoning failure is never limit-classified, so the two
    // never contend for the same failure.
    it('leaves a usage-limit failure on the auto-continue path', async () => {
      routerExtension(mockPi);
      const mockCtx = buildMockCtx();
      await initRouterEnabled(mockCtx);

      const result = await triggerBeforeSettle(
        beforeSettleEvent('error', [
          assistantEncryptedError({ errorMessage: 'insufficient_quota' }),
        ]),
        mockCtx,
      );

      expect(result).toMatchObject({
        continue: true,
        entries: [
          { type: 'custom_message', customType: 'router-auto-continue' },
        ],
      });
    });

    // P1-2 regression, extension side: the rejection is persisted as durable
    // session metadata bound to fingerprints of the signed history, so the
    // exclusion no longer depends on the rejecting message staying in the
    // projection (branch summary / restart / compaction).
    it('persists a fingerprint-bound rejection record and never the raw signature', async () => {
      routerExtension(mockPi);
      const mockCtx = buildMockCtx();
      await initRouterEnabled(mockCtx);

      const signed = {
        role: 'assistant',
        stopReason: 'stop',
        provider: 'openai',
        model: 'gpt-4o-mini',
        api: 'openai-completions',
        content: [
          {
            type: 'thinking',
            thinking: 'thought',
            thinkingSignature: 'opaque-signature-payload',
          },
        ],
      };

      const result = await triggerBeforeSettle(
        beforeSettleEvent('error', [signed, assistantEncryptedError()]),
        mockCtx,
      );
      expect(result).toMatchObject({ continue: true });

      const appendEntryMock = mockPi.appendEntry as unknown as ReturnType<
        typeof vi.fn
      >;
      const call = appendEntryMock.mock.calls.find(
        (args: unknown[]) => args[0] === 'router-encrypted-rejection',
      );
      expect(call).toBeDefined();
      const record = call?.[1] as {
        failedRef: string;
        fingerprints: string[];
      };
      expect(record.failedRef).toBe('openai/gpt-4o-mini');
      expect(record.fingerprints).toHaveLength(1);
      expect(record.fingerprints[0]).toMatch(/^[0-9a-f]{16}$/);
      expect(JSON.stringify(record)).not.toContain('opaque-signature-payload');
    });

    // P1-2 regression: metadata is loaded from ALL session entries (not only
    // the active branch), so a branch summary that abandoned the rejecting
    // assistant message keeps the record available; the fingerprint rule then
    // decides whether it still applies.
    it('loads rejection records from every session entry on session_start', async () => {
      routerExtension(mockPi);
      const mockCtx = buildMockCtx();
      const signed = {
        role: 'assistant',
        stopReason: 'stop',
        provider: 'openai',
        model: 'gpt-4o-mini',
        api: 'openai-completions',
        content: [
          {
            type: 'thinking',
            thinking: 'thought',
            thinkingSignature: 'opaque-signature-payload',
          },
        ],
      };
      const fingerprint = signatureFingerprintForTest(
        'openai',
        'openai-completions',
        'gpt-4o-mini',
        'opaque-signature-payload',
      );
      mockCtx.sessionManager.getEntries = () => [
        {
          type: 'custom',
          customType: 'router-encrypted-rejection',
          data: {
            customType: 'router-encrypted-rejection',
            failedRef: 'openai/gpt-4o-mini',
            api: 'openai-completions',
            fingerprints: [fingerprint],
            at: Date.now(),
          },
        },
      ];
      await initRouterEnabled(mockCtx);

      // The projection still carries a rejection message, but the point of
      // this regression is that the RECORD alone excludes the model: the only
      // remaining candidate is the configured fallback.
      const result = await triggerBeforeSettle(
        beforeSettleEvent('error', [signed, assistantEncryptedError()]),
        mockCtx,
      );
      expect(result).toMatchObject({ continue: true });
    });

    // P1-1 regression, decision side: the continuation check uses the same
    // candidate calculation as route(), so it never fires when nothing is left.
    it('does not continue when every candidate is excluded by recorded metadata', async () => {
      routerExtension(mockPi);
      const mockCtx = buildMockCtx();
      const signed = {
        role: 'assistant',
        stopReason: 'stop',
        provider: 'openai',
        model: 'gpt-4o-mini',
        api: 'openai-completions',
        content: [
          {
            type: 'thinking',
            thinking: 'thought',
            thinkingSignature: 'opaque-signature-payload',
          },
        ],
      };
      const fingerprint = signatureFingerprintForTest(
        'openai',
        'openai-completions',
        'gpt-4o-mini',
        'opaque-signature-payload',
      );
      mockCtx.sessionManager.getEntries = () => [
        {
          type: 'custom',
          customType: 'router-encrypted-rejection',
          data: {
            customType: 'router-encrypted-rejection',
            failedRef: 'openai/gpt-4o-mini',
            api: 'openai-completions',
            fingerprints: [fingerprint],
            at: Date.now(),
          },
        },
        {
          type: 'custom',
          customType: 'router-encrypted-rejection',
          data: {
            customType: 'router-encrypted-rejection',
            failedRef: 'openai/gpt-4o-mini-fallback',
            api: 'openai-completions',
            fingerprints: [fingerprint],
            at: Date.now(),
          },
        },
      ];
      await initRouterEnabled(mockCtx);

      const result = await triggerBeforeSettle(
        beforeSettleEvent('error', [signed, assistantEncryptedError()]),
        mockCtx,
      );
      expect(result).toBeUndefined();
    });

    // P1 regression: the rejection is persisted BEFORE (and independently of)
    // the continuation decision. At final exhaustion there is no replacement,
    // so no continuation is requested -- but this model's signature binding
    // must still be saved, otherwise a later branch summary that keeps its
    // signed response would make it selectable again.
    it('persists the rejection even when no continuation is possible', async () => {
      routerExtension(mockPi);
      const mockCtx = buildMockCtx();
      await initRouterEnabled(mockCtx);
      // No candidate resolves -> no usable replacement -> no continuation.
      mockCtx.modelRegistry.find = vi.fn(() => undefined);

      const signed = {
        role: 'assistant',
        stopReason: 'stop',
        provider: 'openai',
        model: 'gpt-4o-mini',
        api: 'openai-completions',
        content: [
          {
            type: 'thinking',
            thinking: 'thought',
            thinkingSignature: 'opaque-signature-payload',
          },
        ],
      };
      const result = await triggerBeforeSettle(
        beforeSettleEvent('error', [signed, assistantEncryptedError()]),
        mockCtx,
      );

      expect(result).toBeUndefined();
      const appendEntryMock = mockPi.appendEntry as unknown as ReturnType<
        typeof vi.fn
      >;
      const call = appendEntryMock.mock.calls.find(
        (args: unknown[]) => args[0] === 'router-encrypted-rejection',
      );
      expect(call).toBeDefined();
      const record = call?.[1] as { failedRef: string; fingerprints: string[] };
      expect(record.failedRef).toBe('openai/gpt-4o-mini');
      expect(record.fingerprints).toHaveLength(1);
    });

    // P1 regression (over-binding): a rejection is bound only to the rejecting
    // model's own replayable signatures, so another model's payload staying in
    // the conversation cannot keep it excluded.
    it("persists only the rejecting model's own signature fingerprints", async () => {
      routerExtension(mockPi);
      const mockCtx = buildMockCtx();
      await initRouterEnabled(mockCtx);

      const foreign = {
        role: 'assistant',
        stopReason: 'stop',
        provider: 'google',
        model: 'gemini-1.5-flash',
        api: 'openai-completions',
        content: [
          {
            type: 'thinking',
            thinking: 'thought',
            thinkingSignature: 'foreign-signature',
          },
        ],
      };
      const own = {
        role: 'assistant',
        stopReason: 'stop',
        provider: 'openai',
        model: 'gpt-4o-mini',
        api: 'openai-completions',
        content: [
          {
            type: 'thinking',
            thinking: 'thought',
            thinkingSignature: 'own-signature',
          },
        ],
      };
      await triggerBeforeSettle(
        beforeSettleEvent('error', [foreign, own, assistantEncryptedError()]),
        mockCtx,
      );

      const appendEntryMock = mockPi.appendEntry as unknown as ReturnType<
        typeof vi.fn
      >;
      const record = appendEntryMock.mock.calls.find(
        (args: unknown[]) => args[0] === 'router-encrypted-rejection',
      )?.[1] as { fingerprints: string[] };
      expect(record.fingerprints).toEqual([
        signatureFingerprintForTest(
          'openai',
          'openai-completions',
          'gpt-4o-mini',
          'own-signature',
        ),
      ]);
    });

    it('agent_settled ends the episode so a later run fires again', async () => {
      routerExtension(mockPi);
      const mockCtx = buildMockCtx();
      await initRouterEnabled(mockCtx);

      expect(
        await triggerBeforeSettle(
          beforeSettleEvent('error', [assistantEncryptedError()]),
          mockCtx,
        ),
      ).toBeDefined();

      await triggerSettled(mockCtx);

      expect(
        await triggerBeforeSettle(
          beforeSettleEvent('error', [assistantEncryptedError()]),
          mockCtx,
        ),
      ).toBeDefined();
    });
  });

  describe('persistState deduplication', () => {
    it('should only call appendEntry once when state has not changed between turn_end calls', async () => {
      routerExtension(mockPi);

      const mockCtx = buildMockCtx();

      // Initialize with router enabled (default: model is router/balanced)
      const sessionStartHandlers = eventListeners['session_start'] || [];
      for (const handler of sessionStartHandlers) {
        await handler({}, mockCtx);
      }

      // Select router model to ensure routerEnabled=true
      const modelSelectHandlers = eventListeners['model_select'] || [];
      for (const handler of modelSelectHandlers) {
        await handler(
          { model: { provider: 'router', id: 'balanced' } },
          mockCtx,
        );
      }

      // Clear mocks after initialization and model_select
      mockPi.appendEntry.mockClear();

      // Trigger turn_end — first call may persist if snapshot differs
      const turnEndHandlers = eventListeners['turn_end'] || [];
      for (const handler of turnEndHandlers) {
        await handler({}, mockCtx);
      }
      const callsAfterFirst = mockPi.appendEntry.mock.calls.length;

      // Trigger turn_end again — state is identical, snapshot dedup should skip
      for (const handler of turnEndHandlers) {
        await handler({}, mockCtx);
      }
      const callsAfterSecond = mockPi.appendEntry.mock.calls.length;

      // No additional appendEntry calls on the second turn_end
      expect(callsAfterSecond).toBe(callsAfterFirst);
    });
  });

  describe('ensureInitializedFromContext', () => {
    it('should initialize registry and context on first turn_start, but not overwrite on subsequent events', async () => {
      routerExtension(mockPi);

      const mockCtx1 = buildMockCtx();
      mockCtx1.cwd = '/mock/cwd1';

      // Trigger turn_start event with mockCtx1
      const turnStartHandlers = eventListeners['turn_start'] || [];
      for (const handler of turnStartHandlers) {
        await handler({}, mockCtx1);
      }

      // Should have reloaded config and updated status because registry was undefined
      expect(mockCtx1.ui.setStatus).toHaveBeenCalled();
      mockCtx1.ui.setStatus.mockClear();

      // Trigger turn_start with a DIFFERENT CWD — should NOT reinitialize because
      // registry is already set (guards against subagent overwriting parent state)
      const mockCtx2 = buildMockCtx();
      mockCtx2.cwd = '/mock/cwd2';
      await turnStartHandlers[0]({}, mockCtx2);
      expect(mockCtx2.ui.setStatus).not.toHaveBeenCalled();
    });

    it('should refresh router virtual model registration on turn_start when dynamic models (e.g. llama.cpp) are added after session_start', async () => {
      let dynamicModelLoaded = false;
      routerExtension(mockPi);

      const mockCtx = buildMockCtx();
      mockCtx.modelRegistry.find = vi.fn(
        (provider: string, modelId: string) => {
          if (provider === 'openai' && modelId === 'gpt-4o') {
            return dynamicModelLoaded
              ? createTestModel({
                  provider,
                  id: modelId,
                  reasoning: true,
                })
              : undefined;
          }
          return createTestModel({
            provider,
            id: modelId,
            reasoning: false,
          });
        },
      );

      const sessionStartHandlers = eventListeners['session_start'] || [];
      for (const handler of sessionStartHandlers) {
        await handler({}, mockCtx);
      }

      mockPi.registerVirtualModel.mockClear();

      // Simulate dynamic model registration (e.g. via /llama command after session_start)
      dynamicModelLoaded = true;

      const turnStartHandlers = eventListeners['turn_start'] || [];
      for (const handler of turnStartHandlers) {
        await handler({}, mockCtx);
      }

      expect(mockPi.registerVirtualModel).toHaveBeenCalledWith(
        expect.objectContaining({
          provider: 'router',
          id: 'balanced',
          thinkingLevels: ['off', 'low', 'medium', 'high'],
        }),
      );
    });

    it('should accumulate assistant message cost on turn_end when router is enabled', async () => {
      routerExtension(mockPi);
      const mockCtx = buildMockCtx();

      const sessionStartHandlers = eventListeners['session_start'] || [];
      for (const handler of sessionStartHandlers) {
        await handler({}, mockCtx);
      }

      mockPi.appendEntry.mockClear();

      const turnEndHandlers = eventListeners['turn_end'] || [];
      const assistantMessage = {
        role: 'assistant',
        usage: { cost: { total: 0.0042 } },
      };
      for (const handler of turnEndHandlers) {
        await handler({ message: assistantMessage }, mockCtx);
      }

      expect(mockPi.appendEntry).toHaveBeenCalledWith(
        'router-state',
        expect.objectContaining({
          accumulatedCost: 0.0042,
        }),
      );
    });
  });
});
