import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { chmodSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { Api, Model, Provider } from '@earendil-works/pi-ai';
import type { AutocompleteItem } from '@earendil-works/pi-tui';
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';
import { registerCommands } from './commands';
import { createCooldownStore } from './cooldown';
import {
  claimNextAccountSlot,
  loadAccountRegistry,
  markAccountRemoved,
  readAccountEntry,
} from './accounts';
import type {
  RouterConfig,
  RoutingDecision,
  RouterPinByProfile,
  RouterThinkingByProfile,
  RouterCommandContext,
} from './types';

interface RegisteredRouterCommand {
  description?: string;
  getArgumentCompletions?: (
    prefix: string,
  ) => AutocompleteItem[] | null | Promise<AutocompleteItem[] | null>;
  handler: (args: string, ctx: RouterCommandContext) => Promise<void>;
}

const createTestModel = (provider: string, id: string): Model<Api> => ({
  id,
  name: id,
  api: 'openai-completions',
  provider,
  baseUrl: 'https://api.openai.com/v1',
  input: ['text', 'image'],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  reasoning: false,
  contextWindow: 128000,
  maxTokens: 4096,
});

describe('commands.ts', () => {
  // extensions/accounts.ts's claimNextAccountSlot/loadAccountRegistry/
  // markAccountRemoved/readAccountEntry all read the account-alias
  // directory via `PI_MODEL_ROUTER_ACCOUNTS`, resolved lazily at call time
  // (same pattern as cooldown.ts's `PI_MODEL_ROUTER_COOLDOWNS`, see
  // index.test.ts). Every test in this file gets its own tmp directory so
  // none of them ever touch the real ~/.local/state/pi-model-router/accounts.
  let accountsDir: string;
  let previousAccountsEnv: string | undefined;

  beforeEach(() => {
    previousAccountsEnv = process.env.PI_MODEL_ROUTER_ACCOUNTS;
    accountsDir = mkdtempSync(join(tmpdir(), 'pmr-commands-accounts-'));
    process.env.PI_MODEL_ROUTER_ACCOUNTS = accountsDir;
  });

  afterEach(() => {
    if (previousAccountsEnv === undefined) {
      delete process.env.PI_MODEL_ROUTER_ACCOUNTS;
    } else {
      process.env.PI_MODEL_ROUTER_ACCOUNTS = previousAccountsEnv;
    }
    chmodSync(accountsDir, 0o700); // undo any permission-failure test's chmod
    rmSync(accountsDir, { recursive: true, force: true });
  });

  const buildMockPi = () => {
    let registeredCommand: RegisteredRouterCommand | null = null;
    return {
      registerCommand: (name: string, cmd: RegisteredRouterCommand) => {
        if (name === 'router') {
          registeredCommand = cmd;
        }
      },
      setModel: vi
        .fn<(model: Model<Api>) => Promise<boolean>>()
        .mockResolvedValue(true),
      registerProvider: vi.fn() as unknown as ExtensionAPI['registerProvider'],
      unregisterProvider:
        vi.fn() as unknown as ExtensionAPI['unregisterProvider'],
      getRegisteredCommand: () => {
        if (!registeredCommand || !registeredCommand.getArgumentCompletions) {
          throw new Error('Command router was not registered');
        }
        return {
          ...registeredCommand,
          getArgumentCompletions: (
            prefix: string,
          ): AutocompleteItem[] | null => {
            const res = registeredCommand!.getArgumentCompletions!(prefix);
            return res instanceof Promise ? null : res;
          },
        };
      },
    };
  };

  const buildMockCtx = () => ({
    ui: {
      notify:
        vi.fn<(message: string, type?: 'info' | 'warning' | 'error') => void>(),
      setStatus: vi.fn<(key: string, text: string | undefined) => void>(),
      setWidget: vi.fn<(key: string, content: string[] | undefined) => void>(),
    },
    modelRegistry: {
      find: vi
        .fn<(provider: string, modelId: string) => Model<Api> | undefined>()
        .mockImplementation((provider: string, modelId: string) => {
          if (provider === 'router' || provider === 'openai') {
            return createTestModel(provider, modelId);
          }
          return undefined;
        }),
      getProvider: vi
        .fn<(id: string) => Provider | undefined>()
        .mockImplementation((id: string) =>
          id === 'zai' || id === 'anthropic'
            ? ({
                id,
                name: id === 'zai' ? 'ZAI' : 'Anthropic',
                getModels: () => [],
                stream: vi.fn(),
                streamSimple: vi.fn(),
                auth: { apiKey: undefined, oauth: undefined },
              } as unknown as Provider)
            : undefined,
        ),
    },
    model: { provider: 'router', id: 'balanced' },
  });

  const buildDefaultState = () => {
    const config: RouterConfig = {
      phaseBias: 0.5,
      profiles: {
        balanced: {
          high: { model: 'openai/gpt-4o' },
          medium: { model: 'openai/gpt-4o-mini' },
        },
        cheap: {
          low: { model: 'openai/gpt-4o-micro' },
        },
      },
    };

    const lastDecision: RoutingDecision = {
      profile: 'balanced',
      tier: 'medium',
      phase: 'implementation',
      targetProvider: 'openai',
      targetModelId: 'gpt-4o-mini',
      targetLabel: 'openai/gpt-4o-mini',
      reasoning: 'Default reasoning',
      thinking: 'medium',
      timestamp: Date.now(),
    };

    return {
      currentConfig: config,
      routerEnabled: true,
      selectedProfile: 'balanced' as string | undefined,
      pinnedTierByProfile: {} as RouterPinByProfile,
      thinkingByProfile: {} as RouterThinkingByProfile,
      lastDecision: lastDecision as RoutingDecision | undefined,
      lastNonRouterModel: 'openai/gpt-4o' as string | undefined,
      accumulatedCost: 0.05,
      debugEnabled: false,
      widgetEnabled: false,
      debugHistory: [lastDecision],
      lastConfigWarnings: [] as string[],
      ownedAccountProviderIds: new Set<string>(),
    };
  };

  const buildMockActions = () => ({
    persistState: vi.fn(),
    updateStatus: vi.fn(),
    reloadConfig: vi.fn(),
    ensureValidActiveRouterProfile: vi.fn(),
    switchToRouterProfile: vi.fn().mockResolvedValue(true),
    syncPiThinkingLevel: vi.fn(),
  });

  describe('Registration & Subcommand Completion', () => {
    it('should register router command', () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();

      registerCommands(pi, state, actions);
      expect(pi.getRegisteredCommand()).toBeDefined();
    });

    it('should autocomplete subcommands', () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      const completions = cmd.getArgumentCompletions('');
      expect(completions).toBeDefined();
      const names = completions!.map((c) => c.value);
      expect(names).toContain('status');
      expect(names).toContain('profile');
      expect(names).toContain('pin');
    });

    it('should autocomplete profile names', () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      const completions = cmd.getArgumentCompletions('profile ');
      expect(completions).toBeDefined();
      const values = completions!.map((c) => c.value);
      expect(values).toContain('profile balanced');
      expect(values).toContain('profile cheap');
    });

    it('should autocomplete pin arguments', () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      const completions = cmd.getArgumentCompletions('pin ');
      expect(completions).toBeDefined();
      const values = completions!.map((c) => c.value);
      expect(values).toContain('pin auto');
      expect(values).toContain('pin high');
      expect(values).not.toContain('pin balanced');
    });
  });

  describe('Handler Subcommands', () => {
    it('should handle /router status', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('status', ctx);
      expect(ctx.ui.notify).toHaveBeenCalled();
      const notifyMessage = ctx.ui.notify.mock.calls[0][0];
      expect(notifyMessage).toContain('Model Router Status:');
      expect(notifyMessage).toContain('Selected profile: balanced');
      expect(actions.updateStatus).toHaveBeenCalledWith(ctx);
    });

    it('should handle /router profile switch', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('profile cheap', ctx);
      expect(actions.switchToRouterProfile).toHaveBeenCalledWith('cheap', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        expect.stringContaining('Switched to router profile'),
        'info',
      );
    });

    it('should handle /router pin', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('pin high', ctx);
      expect(state.pinnedTierByProfile.balanced).toBe('high');
      expect(actions.persistState).toHaveBeenCalled();
      expect(actions.updateStatus).toHaveBeenCalledWith(ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        'Router pinned to high',
        'info',
      );

      // Clear pin
      await cmd.handler('pin auto', ctx);
      expect(state.pinnedTierByProfile.balanced).toBeUndefined();
    });

    it('should handle /router thinking', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('thinking high xhigh', ctx);
      expect(state.thinkingByProfile.balanced?.high).toBe('xhigh');
      expect(actions.persistState).toHaveBeenCalled();
      expect(actions.updateStatus).toHaveBeenCalledWith(ctx);
    });

    it('should handle /router disable', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('disable', ctx);
      expect(pi.setModel).toHaveBeenCalledWith(
        expect.objectContaining({
          provider: 'openai',
          id: 'gpt-4o',
        }),
      );
      expect(state.routerEnabled).toBe(false);
      expect(actions.persistState).toHaveBeenCalled();
      expect(actions.updateStatus).toHaveBeenCalledWith(ctx);
    });

    it('should handle /router fix', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('fix low', ctx);
      expect(state.pinnedTierByProfile.balanced).toBe('low');
      expect(actions.persistState).toHaveBeenCalled();
      expect(actions.updateStatus).toHaveBeenCalledWith(ctx);
    });

    it('should handle /router widget toggles', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('widget on', ctx);
      expect(state.widgetEnabled).toBe(true);

      await cmd.handler('widget off', ctx);
      expect(state.widgetEnabled).toBe(false);
    });

    it('should handle /router debug history control', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('debug show', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        expect.stringContaining('Recent Routing Decisions'),
        'info',
      );

      await cmd.handler('debug clear', ctx);
      expect(state.debugHistory.length).toBe(0);
    });

    it('should handle /router reload config', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('reload', ctx);
      expect(actions.reloadConfig).toHaveBeenCalledWith(ctx, {
        preserveDebug: true,
      });
      expect(actions.ensureValidActiveRouterProfile).toHaveBeenCalledWith(ctx);
    });
  });

  describe('handleStatus edge cases', () => {
    it('should show error when status has extra args', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('status extra', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        'Usage: /router status (no arguments)',
        'error',
      );
    });

    it('should display Auto-pin first turn status', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      state.currentConfig.autoPinFirstTurnDecision = true;
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('status', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        expect.stringContaining('Auto-pin first turn: on'),
        'info',
      );
    });

    it('should handle status without lastDecision', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      state.lastDecision = undefined;
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('status', ctx);
      const notifyMessage = ctx.ui.notify.mock.calls[0][0];
      expect(notifyMessage).toContain('Model Router Status:');
      expect(notifyMessage).not.toContain('Last routed tier:');
    });

    it('should show config warnings in status', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      state.lastConfigWarnings = ['Warning 1', 'Warning 2'];
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('status', ctx);
      const notifyMessage = ctx.ui.notify.mock.calls[0][0];
      expect(notifyMessage).toContain('⚠️ Configuration Warnings:');
      expect(notifyMessage).toContain('Warning 1');
      expect(notifyMessage).toContain('Warning 2');
    });

    it('should show maxSessionBudget in status', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      state.currentConfig.maxSessionBudget = 10.0;
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('status', ctx);
      const notifyMessage = ctx.ui.notify.mock.calls[0][0];
      expect(notifyMessage).toContain('$10.00');
    });
  });

  describe('handleProfile edge cases', () => {
    it('should show current profile when no argument given', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('profile', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        expect.stringContaining('Current profile: balanced'),
        'info',
      );
    });

    it('should show error when profile has too many arguments', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('profile one two', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        'Usage: /router profile [name]',
        'error',
      );
    });

    it('should not notify on profile switch failure', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      actions.switchToRouterProfile.mockResolvedValue(false);
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('profile nonexistent', ctx);
      expect(actions.switchToRouterProfile).toHaveBeenCalledWith(
        'nonexistent',
        ctx,
      );
      // When switchToRouterProfile returns false, no success notification
      expect(ctx.ui.notify).not.toHaveBeenCalledWith(
        expect.stringContaining('Switched to router profile'),
        'info',
      );
    });
  });

  describe('handlePin edge cases', () => {
    it('should show error when no active profile', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      state.selectedProfile = undefined;
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('pin high', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        'No router profile is active. Select a router model first.',
        'error',
      );
    });

    it('should show current pin when no arguments', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('pin', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        expect.stringContaining('Pinned tier: auto'),
        'info',
      );
      expect(actions.updateStatus).toHaveBeenCalledWith(ctx);
    });

    it('should show error when pin has too many arguments', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('pin high extra', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        'Usage: /router pin <high|medium|low|auto>',
        'error',
      );
    });

    it('should show error when pin value is invalid', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('pin invalid', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        expect.stringContaining('Invalid router pin: invalid'),
        'error',
      );
    });
  });

  describe('handleThinking branches', () => {
    it('should show error when no active profile', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      state.selectedProfile = undefined;
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('thinking high', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        'No router profile is active. Select a router model first.',
        'error',
      );
    });

    it('should show current thinking when no arguments', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('thinking', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        expect.stringContaining('Thinking overrides:'),
        'info',
      );
    });

    it('should show error with too many arguments', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('thinking high medium low', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        'Too many arguments for /router thinking.',
        'error',
      );
    });

    it('should show error with invalid tier', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('thinking badtier high', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        expect.stringContaining('Invalid tier: badtier'),
        'error',
      );
    });

    it('should show error with invalid level', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('thinking badlevel', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        expect.stringContaining('Invalid thinking level: badlevel'),
        'error',
      );
    });

    it('should apply auto to all tiers and clear overrides', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      state.thinkingByProfile.balanced = { high: 'xhigh', medium: 'medium' };
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('thinking auto', ctx);
      // All tier overrides should be cleared, and the profile entry deleted
      expect(state.thinkingByProfile.balanced).toBeUndefined();
      expect(actions.persistState).toHaveBeenCalled();
      expect(actions.updateStatus).toHaveBeenCalledWith(ctx);
    });

    it('should apply thinking level to specific tier', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('thinking low minimal', ctx);
      expect(state.thinkingByProfile.balanced?.low).toBe('minimal');
      expect(actions.persistState).toHaveBeenCalled();
    });

    it('should clear specific tier with auto', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      state.thinkingByProfile.balanced = { high: 'xhigh' };
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('thinking high auto', ctx);
      // The profile entry should be cleaned up since it's empty
      expect(state.thinkingByProfile.balanced).toBeUndefined();
      expect(actions.persistState).toHaveBeenCalled();
    });

    it('should sync pi thinking level when setting a level', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('thinking high', ctx);
      expect(actions.syncPiThinkingLevel).toHaveBeenCalledWith('high');
    });

    it('should restore last decision thinking when setting auto', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('thinking auto', ctx);
      // lastDecision.thinking is 'medium'
      expect(actions.syncPiThinkingLevel).toHaveBeenCalledWith('medium');
    });

    it('should warn about unsupported tiers', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      // Set up profile with resolvedThinkingLevels that exclude 'xhigh'
      state.currentConfig.profiles.balanced = {
        high: {
          model: 'openai/gpt-4o',
          resolvedThinkingLevels: ['high', 'medium', 'low'],
        },
        medium: {
          model: 'openai/gpt-4o-mini',
          resolvedThinkingLevels: ['high', 'medium', 'low'],
        },
      };
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('thinking xhigh', ctx);
      // Should warn that tiers don't support xhigh
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        expect.stringContaining("may not support 'xhigh'"),
        'warning',
      );
    });

    it('should not warn for off level', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      state.currentConfig.profiles.balanced = {
        high: {
          model: 'openai/gpt-4o',
          resolvedThinkingLevels: ['high', 'medium'],
        },
      };
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('thinking off', ctx);
      // Should NOT warn for 'off'
      const warnCalls = ctx.ui.notify.mock.calls.filter(
        (c: unknown[]) => c[1] === 'warning',
      );
      expect(warnCalls.length).toBe(0);
    });

    it('should accept "all" as explicit tier arg', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('thinking all high', ctx);
      expect(state.thinkingByProfile.balanced?.high).toBe('high');
      expect(state.thinkingByProfile.balanced?.medium).toBe('high');
      expect(state.thinkingByProfile.balanced?.low).toBe('high');
      expect(actions.persistState).toHaveBeenCalled();
    });
  });

  describe('handleDisable edge cases', () => {
    it('should show error with extra args', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('disable extra', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        'Usage: /router disable (no arguments)',
        'error',
      );
    });

    it('should warn when no lastNonRouterModel', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      state.lastNonRouterModel = undefined;
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('disable', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        expect.stringContaining('No previous non-router model recorded'),
        'warning',
      );
    });

    it('should show error when model not found in registry', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      state.lastNonRouterModel = 'unknown/model-x';
      const actions = buildMockActions();
      const ctx = buildMockCtx();
      ctx.modelRegistry.find.mockReturnValue(undefined);

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('disable', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        expect.stringContaining('Recorded non-router model is unavailable'),
        'error',
      );
    });

    it('should show error when setModel fails', async () => {
      const pi = buildMockPi();
      pi.setModel.mockResolvedValue(false);
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('disable', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        expect.stringContaining('Failed to switch to'),
        'error',
      );
      // State should NOT be changed on failure
      expect(state.routerEnabled).toBe(true);
    });
  });

  describe('handleFix edge cases', () => {
    it('should show error with wrong number of args', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('fix', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        'Usage: /router fix <high|medium|low>',
        'error',
      );
    });

    it('should show error with too many args', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('fix high extra', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        'Usage: /router fix <high|medium|low>',
        'error',
      );
    });

    it('should show error with invalid tier', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('fix badtier', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        'Usage: /router fix <high|medium|low>',
        'error',
      );
    });

    it('should warn when no last decision', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      state.lastDecision = undefined;
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('fix high', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        'No recent routing decision to fix.',
        'warning',
      );
    });
  });

  describe('handleWidget edge cases', () => {
    it('should show error with too many args', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('widget on extra', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        'Usage: /router widget <on|off|toggle>',
        'error',
      );
    });

    it('should toggle widget when no arg given', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      state.widgetEnabled = false;
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('widget', ctx);
      expect(state.widgetEnabled).toBe(true);
      expect(actions.persistState).toHaveBeenCalled();

      await cmd.handler('widget', ctx);
      expect(state.widgetEnabled).toBe(false);
    });
  });

  describe('handleDebug edge cases', () => {
    it('should enable debug explicitly', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      state.debugEnabled = false;
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('debug on', ctx);
      expect(state.debugEnabled).toBe(true);
      expect(actions.persistState).toHaveBeenCalled();
    });

    it('should disable debug explicitly', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      state.debugEnabled = true;
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('debug off', ctx);
      expect(state.debugEnabled).toBe(false);
      expect(actions.persistState).toHaveBeenCalled();
    });

    it('should toggle debug when no arg given', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      state.debugEnabled = false;
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('debug', ctx);
      expect(state.debugEnabled).toBe(true);

      await cmd.handler('debug', ctx);
      expect(state.debugEnabled).toBe(false);
    });

    it('should show message when debug history is empty', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      state.debugHistory.length = 0;
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('debug show', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        'No recent routing decisions.',
        'info',
      );
    });

    it('should show error with too many args', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('debug on extra', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        'Usage: /router debug <on|off|show|clear>',
        'error',
      );
    });
  });

  describe('handleCooldowns edge cases', () => {
    let cooldownDir: string;

    beforeEach(() => {
      cooldownDir = mkdtempSync(join(tmpdir(), 'pmr-commands-cooldown-'));
    });

    afterEach(() => {
      rmSync(cooldownDir, { recursive: true, force: true });
    });

    const buildStateWithCooldownStore = () => ({
      ...buildDefaultState(),
      cooldownStore: createCooldownStore(
        join(cooldownDir, 'cooldowns'),
        () => Date.parse('2026-11-01T00:00:00Z'),
        join(cooldownDir, 'provider-switch.json'),
      ),
    });

    it('should notify there are no active cooldowns', async () => {
      const pi = buildMockPi();
      const state = buildStateWithCooldownStore();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('cooldowns', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        'No active cooldowns.',
        'info',
      );
    });

    it('should list an active cooldown', async () => {
      const pi = buildMockPi();
      const state = buildStateWithCooldownStore();
      const actions = buildMockActions();
      const ctx = buildMockCtx();
      state.cooldownStore.record(
        'zai/glm-5.3',
        'retry in 10 min',
        'retry',
        state.currentConfig,
      );

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('cooldowns', ctx);
      const [message] = ctx.ui.notify.mock.calls[0];
      expect(message).toContain('zai/glm-5.3');
    });

    it('should clear a single cooldown with "clear <provider/model>"', async () => {
      const pi = buildMockPi();
      const state = buildStateWithCooldownStore();
      const actions = buildMockActions();
      const ctx = buildMockCtx();
      state.cooldownStore.record(
        'zai/glm-5.3',
        'retry in 10 min',
        'retry',
        state.currentConfig,
      );

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('cooldown clear zai/glm-5.3', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        'Cooldown cleared: zai/glm-5.3',
        'info',
      );
      expect(state.cooldownStore.load().entries['zai/glm-5.3']).toBeUndefined();
      expect(actions.updateStatus).toHaveBeenCalled();
    });

    it('should clear every cooldown with "clear all"', async () => {
      const pi = buildMockPi();
      const state = buildStateWithCooldownStore();
      const actions = buildMockActions();
      const ctx = buildMockCtx();
      state.cooldownStore.record(
        'zai/glm-5.3',
        'retry in 10 min',
        'retry',
        state.currentConfig,
      );
      state.cooldownStore.record(
        'moonshot/kimi',
        'retry in 10 min',
        'retry',
        state.currentConfig,
      );

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('cooldown clear all', ctx);
      expect(state.cooldownStore.load().entries).toEqual({});
    });

    it('should show usage on invalid cooldown subcommand', async () => {
      const pi = buildMockPi();
      const state = buildStateWithCooldownStore();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('cooldown bogus', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        'Usage: /router cooldowns [clear <provider/model|all>]',
        'error',
      );
    });

    it('should include a Cooldowns section in /router status', async () => {
      const pi = buildMockPi();
      const state = buildStateWithCooldownStore();
      const actions = buildMockActions();
      const ctx = buildMockCtx();
      state.cooldownStore.record(
        'zai/glm-5.3',
        'retry in 10 min',
        'retry',
        state.currentConfig,
      );

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('', ctx);
      const [message] = ctx.ui.notify.mock.calls[0];
      expect(message).toContain('Cooldowns:');
      expect(message).toContain('zai/glm-5.3');
    });
  });

  describe('handleReload edge cases', () => {
    it('should show error with extra args', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('reload extra', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        'Usage: /router reload (no arguments)',
        'error',
      );
    });
  });

  describe('handleAccount', () => {
    // These tests exercise the real filesystem persistence layer
    // (extensions/accounts.ts), not a mocked registry: handleAccount's
    // add/list/remove now read/write per-account files directly under
    // accountsDir (see the outer beforeEach above) rather than mutating an
    // in-memory "registry" object that a separate save action persists.

    it('registers an account clone by claiming zai-2 on disk', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('account add zai', ctx);

      expect(pi.registerProvider).toHaveBeenCalledTimes(1);
      const registered = (pi.registerProvider as ReturnType<typeof vi.fn>).mock
        .calls[0][0] as Provider;
      expect(registered.id).toBe('zai-2');
      expect(registered.name).toContain('account 2');
      expect(loadAccountRegistry(accountsDir).accounts.zai).toEqual([2]);
      expect(readAccountEntry('zai-2', accountsDir)?.removed).toBe(false);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        expect.stringContaining('Registered account "zai-2"'),
        'info',
      );
    });

    it('assigns increasing account numbers across repeated adds for the same base', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('account add zai', ctx);
      await cmd.handler('account add zai', ctx);

      expect(loadAccountRegistry(accountsDir).accounts.zai).toEqual([2, 3]);
      const ids = (
        pi.registerProvider as ReturnType<typeof vi.fn>
      ).mock.calls.map((call) => (call[0] as Provider).id);
      expect(ids).toEqual(['zai-2', 'zai-3']);
    });

    it('rejects anthropic with a clear message and registers/claims nothing', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('account add anthropic', ctx);

      expect(pi.registerProvider).not.toHaveBeenCalled();
      expect(
        loadAccountRegistry(accountsDir).accounts.anthropic,
      ).toBeUndefined();
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        expect.stringContaining('Pi Black'),
        'error',
      );
    });

    it('rejects an unknown base provider', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('account add not-a-real-provider', ctx);

      expect(pi.registerProvider).not.toHaveBeenCalled();
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        expect.stringContaining('Unknown provider'),
        'error',
      );
    });

    it('shows usage error for add with no provider argument', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('account add', ctx);

      expect(ctx.ui.notify).toHaveBeenCalledWith(
        'Usage: /router account add <provider>',
        'error',
      );
    });

    it('P1 regression: rolls back (tombstones the already-claimed slot, never deletes it) when provider registration fails on add', async () => {
      // Order is claim-then-register (the slot's file is persisted, atomically,
      // BEFORE pi.registerProvider is ever called) -- so a registration
      // failure can never leave a provider registered with nothing on disk
      // claiming its slot. It can leave a claimed-but-never-registered file,
      // which must be tombstoned (not deleted) so the number is never reused.
      // This is the "add-save-failure rollback" case flagged as missing by the
      // review: the old design's failure mode was a save() call failing after
      // registration; here the equivalent failure is registration itself
      // failing after the (always-persisted-on-success) claim.
      const pi = buildMockPi();
      pi.registerProvider = vi.fn().mockImplementation(() => {
        throw new Error('registerProvider boom');
      }) as unknown as ExtensionAPI['registerProvider'];
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('account add zai', ctx);

      expect(ctx.ui.notify).toHaveBeenCalledWith(
        expect.stringContaining('Failed to register account "zai-2"'),
        'error',
      );
      const entry = readAccountEntry('zai-2', accountsDir);
      expect(entry).toBeDefined();
      expect(entry?.removed).toBe(true); // tombstoned, not deleted
      expect(loadAccountRegistry(accountsDir).accounts.zai).toBeUndefined(); // not active
      expect(state.ownedAccountProviderIds.has('zai-2')).toBe(false);
    });

    it('reports an error and never calls pi.registerProvider when claiming an account slot fails', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      chmodSync(accountsDir, 0o500); // directory not writable -> claim fails
      await cmd.handler('account add zai', ctx);
      chmodSync(accountsDir, 0o700);

      expect(pi.registerProvider).not.toHaveBeenCalled();
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        expect.stringContaining('Failed to register account for "zai"'),
        'error',
      );
    });

    it('lists registered accounts grouped by base', async () => {
      claimNextAccountSlot('zai', accountsDir); // zai-2
      claimNextAccountSlot('zai', accountsDir); // zai-3
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('account list', ctx);

      expect(ctx.ui.notify).toHaveBeenCalledWith(
        expect.stringContaining('zai: zai-2, zai-3'),
        'info',
      );
    });

    it('reports no accounts when none are registered', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('account list', ctx);

      expect(ctx.ui.notify).toHaveBeenCalledWith(
        'No extra accounts registered.',
        'info',
      );
    });

    it('account list always re-scans the directory fresh (never a cached view)', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();
      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('account list', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        'No extra accounts registered.',
        'info',
      );

      // Simulates a DIFFERENT process adding an account for the same base
      // directly on disk, with no in-memory state in common with this `state`.
      claimNextAccountSlot('zai', accountsDir);

      await cmd.handler('account list', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        expect.stringContaining('zai: zai-2'),
        'info',
      );
    });

    it('removes a registered account, unregisters the provider, and tombstones its file', async () => {
      claimNextAccountSlot('zai', accountsDir); // zai-2
      const pi = buildMockPi();
      const state = buildDefaultState();
      // Ownership of "zai-2" is normally established by registerPersistedAccounts
      // on session_start/turn_start before any command can run (see
      // extensions/index.ts's restoreStateFromSession/ensureInitializedFromContext).
      state.ownedAccountProviderIds.add('zai-2');
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('account remove zai-2', ctx);

      expect(pi.unregisterProvider).toHaveBeenCalledWith('zai-2');
      expect(loadAccountRegistry(accountsDir).accounts.zai).toBeUndefined();
      const entry = readAccountEntry('zai-2', accountsDir);
      expect(entry?.removed).toBe(true);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        'Removed account "zai-2".',
        'info',
      );
    });

    it("tombstones the file even when this process never actually registered the provider (not owned -- never unregisters someone else's id)", async () => {
      claimNextAccountSlot('zai', accountsDir); // zai-2
      const pi = buildMockPi();
      const state = buildDefaultState();
      // state.ownedAccountProviderIds deliberately left empty.
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('account remove zai-2', ctx);

      expect(pi.unregisterProvider).not.toHaveBeenCalled();
      expect(loadAccountRegistry(accountsDir).accounts.zai).toBeUndefined();
      expect(readAccountEntry('zai-2', accountsDir)?.removed).toBe(true);
    });

    it("P1 regression: removing one session's account never erases a different base's accounts that only exist on disk", async () => {
      // Simulates a second, independent session's account that this `state`
      // never knew about (no shared in-memory registry) -- it must survive an
      // unrelated remove untouched.
      claimNextAccountSlot('zai', accountsDir); // zai-2, about to be removed
      claimNextAccountSlot('deepseek', accountsDir); // deepseek-2, a different session's account
      const pi = buildMockPi();
      const state = buildDefaultState();
      state.ownedAccountProviderIds.add('zai-2');
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('account remove zai-2', ctx);

      expect(loadAccountRegistry(accountsDir).accounts.zai).toBeUndefined();
      expect(loadAccountRegistry(accountsDir).accounts.deepseek).toEqual([2]);
    });

    it('rolls back (re-registers the clone) when persisting the removal fails', async () => {
      claimNextAccountSlot('zai', accountsDir); // zai-2
      const pi = buildMockPi();
      const state = buildDefaultState();
      state.ownedAccountProviderIds.add('zai-2');
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      chmodSync(accountsDir, 0o500); // directory not writable -> the tombstone rewrite fails
      await cmd.handler('account remove zai-2', ctx);
      chmodSync(accountsDir, 0o700);

      expect(pi.unregisterProvider).toHaveBeenCalledWith('zai-2');
      // The on-disk file is unchanged (rewrite failed) -- still active.
      expect(readAccountEntry('zai-2', accountsDir)?.removed).toBe(false);
      // ...and the clone is re-registered under the same alias id.
      const registerCalls = (
        pi.registerProvider as ReturnType<typeof vi.fn>
      ).mock.calls.map((call) => (call[0] as Provider).id);
      expect(registerCalls).toContain('zai-2');
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        expect.stringContaining('rolled back'),
        'error',
      );
    });

    it('reports an error when removing an unknown account', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('account remove zai-9', ctx);

      expect(pi.unregisterProvider).not.toHaveBeenCalled();
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        'No such account: zai-9',
        'error',
      );
    });

    it('reports an error when removing an already-removed (tombstoned) account', async () => {
      const claim = claimNextAccountSlot('zai', accountsDir);
      markAccountRemoved(claim.aliasId!, accountsDir);
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('account remove zai-2', ctx);

      expect(pi.unregisterProvider).not.toHaveBeenCalled();
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        'No such account: zai-2',
        'error',
      );
    });

    it('shows usage error for an unknown account action', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('account', ctx);

      expect(ctx.ui.notify).toHaveBeenCalledWith(
        'Usage: /router account <add <provider>|list|remove <alias>>',
        'error',
      );
    });
  });

  describe('Autocomplete completions', () => {
    it('should autocomplete the account subcommand name', () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      const completions = cmd.getArgumentCompletions('');
      const names = completions!.map((c) => c.value);
      expect(names).toContain('account');
    });

    it('should autocomplete account actions', () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      const completions = cmd.getArgumentCompletions('account ');
      expect(completions).toBeDefined();
      const values = completions!.map((c) => c.value);
      expect(values).toEqual(['account add', 'account list', 'account remove']);
    });

    it('should autocomplete known aliases for account remove', () => {
      claimNextAccountSlot('zai', accountsDir); // zai-2
      claimNextAccountSlot('zai', accountsDir); // zai-3
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      const completions = cmd.getArgumentCompletions('account remove ');
      expect(completions).toBeDefined();
      const values = completions!.map((c) => c.value);
      expect(values).toEqual(['account remove zai-2', 'account remove zai-3']);
    });

    it('should return null for account add (no provider completions offered)', () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      expect(cmd.getArgumentCompletions('account add ')).toBeNull();
    });

    it('should return thinking completions for first arg', () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      const completions = cmd.getArgumentCompletions('thinking ');
      expect(completions).toBeDefined();
      const values = completions!.map((c) => c.value);
      // Should have levels and tiers
      expect(values).toContain('thinking auto');
      expect(values).toContain('thinking high');
    });

    it('should return null when thinking first arg is a level-tier overlap', () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      // 'high' is both a tier and a level; level check comes first, so no further completions
      const completions = cmd.getArgumentCompletions('thinking high a');
      expect(completions).toBeNull();
    });

    it('should return null for thinking completions after level arg', () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      // 'auto' is a level, so no further completions
      const completions = cmd.getArgumentCompletions('thinking auto ');
      expect(completions).toBeNull();
    });

    it('should return fix completions', () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      const completions = cmd.getArgumentCompletions('fix ');
      expect(completions).toBeDefined();
      const values = completions!.map((c) => c.value);
      expect(values).toContain('fix high');
      expect(values).toContain('fix medium');
      expect(values).toContain('fix low');
    });

    it('should return widget completions', () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      const completions = cmd.getArgumentCompletions('widget ');
      expect(completions).toBeDefined();
      const values = completions!.map((c) => c.value);
      expect(values).toContain('widget on');
      expect(values).toContain('widget off');
      expect(values).toContain('widget toggle');
    });

    it('should return debug completions', () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      const completions = cmd.getArgumentCompletions('debug ');
      expect(completions).toBeDefined();
      const values = completions!.map((c) => c.value);
      expect(values).toContain('debug on');
      expect(values).toContain('debug off');
      expect(values).toContain('debug show');
      expect(values).toContain('debug clear');
    });

    it('should return null for unknown subcommand completions', () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      const completions = cmd.getArgumentCompletions('unknown ');
      expect(completions).toBeNull();
    });

    it('should filter subcommand completions by prefix', () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      const completions = cmd.getArgumentCompletions('st');
      expect(completions).toBeDefined();
      const values = completions!.map((c) => c.value);
      expect(values).toContain('status');
      expect(values).not.toContain('profile');
    });
  });

  describe('Default handler branch', () => {
    it('should show error for unknown subcommand', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('nonexistent', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        expect.stringContaining('Unknown router subcommand: nonexistent'),
        'error',
      );
    });

    it('should treat profile name as subcommand (backward compat)', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('cheap', ctx);
      expect(actions.switchToRouterProfile).toHaveBeenCalledWith('cheap', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        expect.stringContaining('Router enabled with profile'),
        'info',
      );
    });

    it('should show error when profile name has extra args', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('balanced extra', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        expect.stringContaining('no extra arguments allowed'),
        'error',
      );
    });

    it('should fall through to status on empty args', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        expect.stringContaining('Model Router Status:'),
        'info',
      );
    });

    it('should show help with /router help', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('help', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        expect.stringContaining('Router Subcommands:'),
        'info',
      );
    });

    it('should show help with /router ?', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('?', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        expect.stringContaining('Router Subcommands:'),
        'info',
      );
    });

    it('should show error when help has extra args', async () => {
      const pi = buildMockPi();
      const state = buildDefaultState();
      const actions = buildMockActions();
      const ctx = buildMockCtx();

      registerCommands(pi, state, actions);
      const cmd = pi.getRegisteredCommand();

      await cmd.handler('help extra', ctx);
      expect(ctx.ui.notify).toHaveBeenCalledWith(
        'Usage: /router help (no arguments)',
        'error',
      );
    });
  });
});
