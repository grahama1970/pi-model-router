import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  registerRouterProvider,
  extractUncountedMessageCost,
  markMessageCostCounted,
  type RouterProviderState,
  type RouterProviderActions,
} from './provider';
import { createCooldownStore, type CooldownStore } from './cooldown';
import type { Api, AssistantMessage, Model } from '@earendil-works/pi-ai';
import type { ThinkingLevel } from '@earendil-works/pi-agent-core';
import type {
  RouterConfig,
  RouterModelRegistry,
  RouterVirtualModelDefinition,
  VirtualRouterState,
  ModelRouteRequest,
} from './types';

const createTestModel = (overrides: Partial<Model<Api>> = {}): Model<Api> => ({
  id: 'gpt-4o',
  name: 'GPT-4o',
  api: 'openai-completions',
  provider: 'openai',
  baseUrl: 'https://api.openai.com/v1',
  input: ['text', 'image'],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  reasoning: false,
  contextWindow: 128000,
  maxTokens: 4096,
  ...overrides,
});

const createAssistantMsg = (text: string, totalCost = 0): AssistantMessage => ({
  role: 'assistant',
  content: [{ type: 'text', text }],
  api: 'openai-completions',
  provider: 'openai',
  model: 'gpt-4o',
  usage: {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 0,
    cost: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      total: totalCost,
    },
  },
  stopReason: 'stop',
  timestamp: Date.now(),
});

interface TestExtensionContext {
  modelRegistry?: RouterModelRegistry;
}

interface TestProviderState extends RouterProviderState<TestExtensionContext> {
  currentConfig: RouterConfig;
  currentModelRegistry: RouterModelRegistry | undefined;
  lastExtensionContext: TestExtensionContext | undefined;
}

describe('provider.ts (virtual models)', () => {
  let registeredVirtualModels: Map<
    string,
    RouterVirtualModelDefinition<VirtualRouterState, TestExtensionContext>
  >;
  let unregisteredVirtualModels: Array<{ provider: string; id: string }>;
  let mockPi: {
    registerVirtualModel: (
      model: RouterVirtualModelDefinition<
        VirtualRouterState,
        TestExtensionContext
      >,
    ) => void;
    unregisterVirtualModel: (provider: string, id: string) => void;
  };
  let mockState: TestProviderState;
  let mockActions: RouterProviderActions<TestExtensionContext>;

  beforeEach(() => {
    vi.clearAllMocks();
    registeredVirtualModels = new Map();
    unregisteredVirtualModels = [];

    mockPi = {
      registerVirtualModel: (definition) => {
        registeredVirtualModels.set(
          `${definition.provider}/${definition.id}`,
          definition,
        );
      },
      unregisterVirtualModel: (provider, id) => {
        unregisteredVirtualModels.push({ provider, id });
        registeredVirtualModels.delete(`${provider}/${id}`);
      },
    };

    const config: RouterConfig = {
      profiles: {
        balanced: {
          high: {
            model: 'openai/gpt-4o',
            thinking: 'high',
            resolvedContextWindow: 10000,
            resolvedThinkingLevels: ['low', 'medium', 'high'],
          },
          medium: {
            model: 'openai/gpt-4o-mini',
            thinking: 'medium',
            resolvedContextWindow: 5000,
            fallbacks: [{ model: 'google/gemini-1.5-flash' }],
            resolvedThinkingLevels: ['low', 'medium', 'high'],
          },
        },
      },
    };

    const mockRegistry: RouterModelRegistry = {
      find: (provider: string, modelId: string) => {
        if (provider === 'openai' || provider === 'google') {
          return createTestModel({
            provider,
            id: modelId,
            input: ['text', 'image'],
            reasoning: true,
            contextWindow: modelId === 'gpt-4o-mini' ? 5000 : 10000,
          });
        }
        return undefined;
      },
      hasConfiguredAuth: () => true,
    };

    mockState = {
      lastRegisteredModels: '',
      currentConfig: config,
      currentModelRegistry: mockRegistry,
      lastExtensionContext: {
        modelRegistry: mockRegistry,
      },
      selectedProfile: undefined,
      routerEnabled: false,
      lastDecision: undefined,
      thinkingByProfile: {},
      pinnedTierByProfile: {},
      accumulatedCost: 0,
      processStartTime: 0,
      processedTranscriptFailures: new Set<string>(),
    };

    mockActions = {
      persistState: vi.fn(),
      recordDebugDecision: vi.fn(),
      getThinkingOverride: vi.fn().mockReturnValue(undefined),
      updateStatus: vi.fn(),
      syncPiThinkingLevel: vi.fn(),
    };
  });

  const buildRequest = (
    overrides: Partial<ModelRouteRequest<VirtualRouterState>> = {},
  ): ModelRouteRequest<VirtualRouterState> => ({
    model: createTestModel({
      id: 'balanced',
      provider: 'router',
      api: 'pi-virtual' as Api,
    }),
    thinkingLevel: 'medium',
    reason: 'user',
    messages: [{ role: 'user', content: 'hello', timestamp: Date.now() }],
    ...overrides,
  });

  describe('registerRouterProvider', () => {
    it('should register each profile as a virtual model under router provider', () => {
      registerRouterProvider(mockPi, mockState, mockActions);

      const vm = registeredVirtualModels.get('router/balanced');
      expect(vm).toBeDefined();
      expect(vm!.provider).toBe('router');
      expect(vm!.id).toBe('balanced');
      expect(vm!.name).toBe('Router balanced');
      expect(vm!.thinkingLevels).toEqual(['off', 'low', 'medium', 'high']);
    });

    it('should register configured thinkingLevels on initial load before currentModelRegistry is bound', () => {
      mockState.currentModelRegistry = undefined;
      registerRouterProvider(mockPi, mockState, mockActions);

      const vm = registeredVirtualModels.get('router/balanced');
      expect(vm).toBeDefined();
      expect(vm!.thinkingLevels).toEqual(['off', 'low', 'medium', 'high']);
    });

    it('should unregister removed profiles when config is reloaded', () => {
      mockState.currentConfig.profiles.cheap = {
        low: { model: 'openai/gpt-4o-mini' },
      };
      registerRouterProvider(mockPi, mockState, mockActions);
      expect(registeredVirtualModels.has('router/balanced')).toBe(true);
      expect(registeredVirtualModels.has('router/cheap')).toBe(true);

      // Remove cheap profile and re-register
      delete mockState.currentConfig.profiles.cheap;
      registerRouterProvider(mockPi, mockState, mockActions);

      expect(unregisteredVirtualModels).toEqual([
        { provider: 'router', id: 'cheap' },
      ]);
      expect(registeredVirtualModels.has('router/cheap')).toBe(false);
    });

    it('should route user requests and return physical model, clamped thinkingLevel, and branch state', async () => {
      registerRouterProvider(mockPi, mockState, mockActions);
      const vm = registeredVirtualModels.get('router/balanced')!;

      const prevAssistant = createAssistantMsg('Previous turn', 0.0025);
      const route = await vm.route(
        buildRequest({
          messages: [
            prevAssistant,
            { role: 'user', content: 'fix this bug', timestamp: Date.now() },
          ],
        }),
        mockState.lastExtensionContext!,
      );

      expect(route.model.provider).toBe('openai');
      expect(route.model.id).toBe('gpt-4o-mini');
      expect(route.thinkingLevel).toBe('medium');
      expect(route.state?.tier).toBe('medium');
      expect(route.state?.phase).toBe('implementation');
      expect(mockState.selectedProfile).toBe('balanced');
      expect(mockState.routerEnabled).toBe(true);
      expect(mockState.accumulatedCost).toBe(0.0025);
      expect(mockActions.persistState).toHaveBeenCalled();
    });

    it('should route direct requests (e.g. compaction summaries) using active model without mutating session routing state', async () => {
      registerRouterProvider(mockPi, mockState, mockActions);
      const vm = registeredVirtualModels.get('router/balanced')!;

      // 1. Without active decision: falls back to medium tier default
      const defaultRoute = await vm.route(
        buildRequest({ reason: 'direct' }),
        mockState.lastExtensionContext!,
      );

      expect(defaultRoute.model.provider).toBe('openai');
      expect(defaultRoute.model.id).toBe('gpt-4o-mini');
      expect(defaultRoute.state).toBeUndefined();
      expect(mockState.lastDecision).toBeUndefined();
      expect(mockActions.recordDebugDecision).not.toHaveBeenCalled();

      // 2. With active high-tier decision: reuses the active model to preserve prompt cache
      mockState.lastDecision = {
        profile: 'balanced',
        tier: 'high',
        phase: 'planning',
        targetProvider: 'openai',
        targetModelId: 'gpt-4o',
        targetLabel: 'openai/gpt-4o',
        thinking: 'high',
        reasoning: 'Active planning turn',
        timestamp: Date.now(),
      };

      const activeRoute = await vm.route(
        buildRequest({ reason: 'direct' }),
        mockState.lastExtensionContext!,
      );

      expect(activeRoute.model.provider).toBe('openai');
      expect(activeRoute.model.id).toBe('gpt-4o');
      expect(activeRoute.thinkingLevel).toBe('high');
      expect(activeRoute.state).toBeUndefined();
    });

    it('should preserve previous model on tool-result continuation across any provider', async () => {
      registerRouterProvider(mockPi, mockState, mockActions);
      const vm = registeredVirtualModels.get('router/balanced')!;

      mockState.lastDecision = {
        profile: 'balanced',
        tier: 'high',
        phase: 'planning',
        targetProvider: 'openai',
        targetModelId: 'gpt-4o',
        targetLabel: 'openai/gpt-4o',
        thinking: 'high',
        reasoning: 'Initial planning decision',
        timestamp: Date.now(),
      };

      const route = await vm.route(
        buildRequest({
          reason: 'continuation',
          messages: [
            {
              role: 'user',
              content: 'plan the architecture',
              timestamp: Date.now(),
            },
            {
              role: 'toolResult',
              toolCallId: 'c1',
              toolName: 'read',
              content: [{ type: 'text', text: 'file contents' }],
              isError: false,
              timestamp: Date.now(),
            },
          ],
        }),
        mockState.lastExtensionContext!,
      );

      expect(route.model.id).toBe('gpt-4o');
      expect(route.thinkingLevel).toBe('high');
      expect(mockState.lastDecision!.targetModelId).toBe('gpt-4o');
      expect(mockState.lastDecision!.reasoning).toContain(
        'Preserved openai/gpt-4o for tool-result continuation',
      );
    });

    it('should advance to fallback model on retry when primary model failed', async () => {
      registerRouterProvider(mockPi, mockState, mockActions);
      const vm = registeredVirtualModels.get('router/balanced')!;

      mockState.pinnedTierByProfile['balanced'] = 'medium';
      mockState.lastDecision = {
        profile: 'balanced',
        tier: 'medium',
        phase: 'implementation',
        targetProvider: 'openai',
        targetModelId: 'gpt-4o-mini',
        targetLabel: 'openai/gpt-4o-mini',
        thinking: 'medium',
        reasoning: 'Pinned medium',
        timestamp: Date.now(),
      };

      const failedModel = createTestModel({
        provider: 'openai',
        id: 'gpt-4o-mini',
      });
      const failedMessage: AssistantMessage = {
        ...createAssistantMsg('', 0),
        provider: 'openai',
        model: 'gpt-4o-mini',
        stopReason: 'error',
        errorMessage: '429 Too Many Requests',
      };

      const route = await vm.route(
        buildRequest({
          reason: 'retry',
          failed: {
            model: failedModel,
            thinkingLevel: 'medium',
            message: failedMessage,
          },
        }),
        mockState.lastExtensionContext!,
      );

      expect(route.model.provider).toBe('google');
      expect(route.model.id).toBe('gemini-1.5-flash');
      expect(mockState.lastDecision!.isFallback).toBe(true);
      expect(mockState.lastDecision!.reasoning).toContain(
        'Fallback to google/gemini-1.5-flash after openai/gpt-4o-mini failed (429 Too Many Requests)',
      );
    });

    it('should fall back immediately when primary model is missing from registry or lacks credentials', async () => {
      mockState.currentModelRegistry!.hasConfiguredAuth = (model: Model<Api>) =>
        model.id !== 'gpt-4o-mini';

      registerRouterProvider(mockPi, mockState, mockActions);
      const vm = registeredVirtualModels.get('router/balanced')!;
      mockState.pinnedTierByProfile['balanced'] = 'medium';

      const route = await vm.route(
        buildRequest(),
        mockState.lastExtensionContext!,
      );

      expect(route.model.provider).toBe('google');
      expect(route.model.id).toBe('gemini-1.5-flash');
      expect(mockState.lastDecision!.isFallback).toBe(true);
    });

    it('should force higher tier if current tier does not support image attachments', async () => {
      mockState.currentModelRegistry!.find = (
        provider: string,
        modelId: string,
      ) => {
        if (modelId === 'gpt-4o') {
          return createTestModel({
            provider,
            id: modelId,
            input: ['text', 'image'],
            reasoning: true,
          });
        }
        return createTestModel({
          provider,
          id: modelId,
          input: ['text'],
          reasoning: true,
        });
      };

      registerRouterProvider(mockPi, mockState, mockActions);
      const vm = registeredVirtualModels.get('router/balanced')!;
      mockState.pinnedTierByProfile['balanced'] = 'medium';

      const route = await vm.route(
        buildRequest({
          messages: [
            {
              role: 'user',
              content: [{ type: 'image', mimeType: 'image/png', data: 'data' }],
              timestamp: Date.now(),
            },
          ],
        }),
        mockState.lastExtensionContext!,
      );

      expect(route.model.id).toBe('gpt-4o');
      expect(mockState.lastDecision!.tier).toBe('high');
      expect(mockState.lastDecision!.reasoning).toContain(
        'Forced high tier because the originally routed medium tier does not support image attachments',
      );
    });

    it('should throw when registry is unavailable', async () => {
      mockState.currentModelRegistry = undefined;
      mockState.lastExtensionContext = {};

      registerRouterProvider(mockPi, mockState, mockActions);
      const vm = registeredVirtualModels.get('router/balanced')!;

      await expect(
        vm.route(buildRequest(), mockState.lastExtensionContext),
      ).rejects.toThrow('unavailable');
      expect(mockActions.persistState).toHaveBeenCalled();
    });

    it('should throw when profile is unknown', async () => {
      registerRouterProvider(mockPi, mockState, mockActions);
      const vm = registeredVirtualModels.get('router/balanced')!;

      await expect(
        vm.route(
          buildRequest({
            model: createTestModel({
              id: 'nonexistent',
              provider: 'router',
            }),
          }),
          mockState.lastExtensionContext!,
        ),
      ).rejects.toThrow('Unknown router profile');
    });

    it('should clamp thinkingLevel to off when target model does not support reasoning', async () => {
      mockState.currentModelRegistry!.find = (
        provider: string,
        modelId: string,
      ) =>
        createTestModel({
          provider,
          id: modelId,
          reasoning: false,
        });

      registerRouterProvider(mockPi, mockState, mockActions);
      const vm = registeredVirtualModels.get('router/balanced')!;
      mockState.pinnedTierByProfile['balanced'] = 'high';

      const route = await vm.route(
        buildRequest({ thinkingLevel: 'high' as ThinkingLevel }),
        mockState.lastExtensionContext!,
      );

      expect(route.thinkingLevel).toBe('off');
    });

    it('should invoke native classifier when classifierModel is configured and accumulate classifier cost', async () => {
      mockState.currentConfig.classifierModel = { model: 'auto' };
      mockState.currentModelRegistry!.getAvailableOfType = vi
        .fn()
        .mockResolvedValue([
          {
            type: 'classifier',
            id: 'jev-latest',
            name: 'Jev Latest',
            api: 'typesafe-system-one',
            provider: 'typesafe',
            baseUrl: 'https://api.typesafe.ai',
            input: ['text'],
            contextWindow: 32000,
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
          },
        ]);
      mockState.currentModelRegistry!.classify = vi.fn().mockResolvedValue({
        api: 'typesafe-system-one',
        provider: 'typesafe',
        model: 'jev-latest',
        stopReason: 'stop',
        timestamp: Date.now(),
        usage: {
          input: 50,
          output: 1,
          cacheRead: 0,
          cacheWrite: 0,
          totalTokens: 51,
          cost: {
            input: 0.0003,
            output: 0,
            cacheRead: 0,
            cacheWrite: 0,
            total: 0.0003,
          },
        },
        answers: {
          tier: {
            type: 'choice',
            choice: 'high',
            probabilities: { high: 0.9, medium: 0.08, low: 0.02 },
            confidence: 0.85,
          },
        },
      });

      registerRouterProvider(mockPi, mockState, mockActions);
      const vm = registeredVirtualModels.get('router/balanced')!;

      const route = await vm.route(
        buildRequest(),
        mockState.lastExtensionContext!,
      );

      expect(route.model.id).toBe('gpt-4o');
      expect(mockState.lastDecision?.tier).toBe('high');
      expect(mockState.lastDecision?.isClassifier).toBe(true);
      expect(mockState.accumulatedCost).toBeCloseTo(0.0003);
    });

    it('should downgrade high tier to medium on continuation when session budget is exceeded', async () => {
      mockState.currentConfig.maxSessionBudget = 0.01;
      mockState.accumulatedCost = 0.02;

      const branchDecision = {
        profile: 'balanced',
        tier: 'high' as const,
        phase: 'planning' as const,
        targetProvider: 'openai',
        targetModelId: 'gpt-4o',
        targetLabel: 'openai/gpt-4o',
        thinking: 'high' as ThinkingLevel,
        reasoning: 'Planning turn',
        timestamp: Date.now(),
      };

      registerRouterProvider(mockPi, mockState, mockActions);
      const vm = registeredVirtualModels.get('router/balanced')!;

      const route = await vm.route(
        buildRequest({
          reason: 'continuation',
          state: {
            phase: 'planning',
            tier: 'high',
            lastDecision: branchDecision,
          },
        }),
        mockState.lastExtensionContext!,
      );

      expect(route.model.id).toBe('gpt-4o-mini');
      expect(mockState.lastDecision?.tier).toBe('medium');
      expect(mockState.lastDecision?.isBudgetForced).toBe(true);
    });

    it('should infer previous decision from request.previous when branch state is absent', async () => {
      registerRouterProvider(mockPi, mockState, mockActions);
      const vm = registeredVirtualModels.get('router/balanced')!;

      const prevModel = createTestModel({
        provider: 'openai',
        id: 'gpt-4o',
        reasoning: true,
      });

      const directRoute = await vm.route(
        buildRequest({
          reason: 'direct',
          previous: { model: prevModel, thinkingLevel: 'high' },
        }),
        mockState.lastExtensionContext!,
      );

      expect(directRoute.model.id).toBe('gpt-4o');
      expect(directRoute.thinkingLevel).toBe('high');
    });

    it('should rotate through multiple fallbacks in order on consecutive retries and preserve fallback across same-tier user turns', async () => {
      mockState.currentConfig.profiles.balanced.medium!.fallbacks = [
        { model: 'google/gemini-1.5-flash' },
        { model: 'openai/gpt-4o-fallback-2' },
      ];

      registerRouterProvider(mockPi, mockState, mockActions);
      const vm = registeredVirtualModels.get('router/balanced')!;

      // Turn 1 initial user request -> primary model (openai/gpt-4o-mini)
      const initialRoute = await vm.route(
        buildRequest({
          messages: [
            { role: 'user', content: 'fix this bug', timestamp: Date.now() },
          ],
        }),
        mockState.lastExtensionContext!,
      );
      expect(initialRoute.model.id).toBe('gpt-4o-mini');

      // Retry 1 after primary failed -> first fallback (google/gemini-1.5-flash)
      const retry1 = await vm.route(
        buildRequest({
          reason: 'retry',
          state: initialRoute.state,
          failed: {
            model: createTestModel({ provider: 'openai', id: 'gpt-4o-mini' }),
            thinkingLevel: 'medium',
            message: {
              ...createAssistantMsg('', 0),
              provider: 'openai',
              model: 'gpt-4o-mini',
              stopReason: 'error',
              errorMessage: '429 Too Many Requests',
            },
          },
        }),
        mockState.lastExtensionContext!,
      );
      expect(retry1.model.provider).toBe('google');
      expect(retry1.model.id).toBe('gemini-1.5-flash');

      // Turn 2 user request on same tier (medium, unpinned) -> sticks to working fallback (gemini-1.5-flash)
      const turn2Route = await vm.route(
        buildRequest({
          reason: 'user',
          state: retry1.state,
          messages: [
            { role: 'user', content: 'fix this bug', timestamp: Date.now() },
            createAssistantMsg('fixed', 0.001),
            {
              role: 'user',
              content: 'also update the test',
              timestamp: Date.now(),
            },
          ],
        }),
        mockState.lastExtensionContext!,
      );
      expect(turn2Route.model.provider).toBe('google');
      expect(turn2Route.model.id).toBe('gemini-1.5-flash');
      expect(mockState.lastDecision?.isFallback).toBe(true);
      expect(mockState.lastDecision?.reasoning).toContain(
        'Preserved fallback google/gemini-1.5-flash for medium tier.',
      );

      // If gemini-1.5-flash now fails on Turn 2 -> advances to 2nd fallback (gpt-4o-fallback-2), NOT back to broken primary
      const retry2 = await vm.route(
        buildRequest({
          reason: 'retry',
          state: turn2Route.state,
          failed: {
            model: createTestModel({
              provider: 'google',
              id: 'gemini-1.5-flash',
            }),
            thinkingLevel: 'medium',
            message: {
              ...createAssistantMsg('', 0),
              provider: 'google',
              model: 'gemini-1.5-flash',
              stopReason: 'error',
              errorMessage: '503 Service Unavailable',
            },
          },
        }),
        mockState.lastExtensionContext!,
      );
      expect(retry2.model.provider).toBe('openai');
      expect(retry2.model.id).toBe('gpt-4o-fallback-2');
    });

    it('should use per-fallback thinking, tier thinking when omitted, and clamp to the selected model', async () => {
      mockState.currentConfig.profiles.balanced.medium!.thinking = 'max';
      mockState.currentConfig.profiles.balanced.medium!.resolvedThinkingLevels =
        ['low', 'high'];
      mockState.currentConfig.profiles.balanced.medium!.fallbacks = [
        { model: 'google/gemini-1.5-flash', thinking: 'low' },
        { model: 'openai/gpt-4o-fallback-2' },
      ];

      registerRouterProvider(mockPi, mockState, mockActions);
      const vm = registeredVirtualModels.get('router/balanced')!;
      const initialRoute = await vm.route(
        buildRequest({
          messages: [
            { role: 'user', content: 'fix this bug', timestamp: Date.now() },
          ],
        }),
        mockState.lastExtensionContext!,
      );
      expect(initialRoute.thinkingLevel).toBe('high');

      const retry1 = await vm.route(
        buildRequest({
          reason: 'retry',
          state: initialRoute.state,
          failed: {
            model: createTestModel({ provider: 'openai', id: 'gpt-4o-mini' }),
            thinkingLevel: 'high',
            message: {
              ...createAssistantMsg('', 0),
              provider: 'openai',
              model: 'gpt-4o-mini',
              stopReason: 'error',
              errorMessage: '429 Too Many Requests',
            },
          },
        }),
        mockState.lastExtensionContext!,
      );
      expect(retry1.model.id).toBe('gemini-1.5-flash');
      expect(retry1.thinkingLevel).toBe('low');
      expect(mockState.lastDecision?.thinking).toBe('low');

      const retry2 = await vm.route(
        buildRequest({
          reason: 'retry',
          state: retry1.state,
          failed: {
            model: createTestModel({
              provider: 'google',
              id: 'gemini-1.5-flash',
            }),
            thinkingLevel: 'low',
            message: {
              ...createAssistantMsg('', 0),
              provider: 'google',
              model: 'gemini-1.5-flash',
              stopReason: 'error',
              errorMessage: '503 Service Unavailable',
            },
          },
        }),
        mockState.lastExtensionContext!,
      );
      expect(retry2.model.id).toBe('gpt-4o-fallback-2');
      expect(retry2.thinkingLevel).toBe('high');
      expect(mockState.lastDecision?.thinking).toBe('high');
    });

    it('should let the explicit thinking override win over a per-fallback level', async () => {
      mockState.currentConfig.profiles.balanced.medium!.fallbacks = [
        { model: 'google/gemini-1.5-flash', thinking: 'low' },
      ];
      mockActions.getThinkingOverride = vi.fn().mockReturnValue('max');

      registerRouterProvider(mockPi, mockState, mockActions);
      const vm = registeredVirtualModels.get('router/balanced')!;
      const initialRoute = await vm.route(
        buildRequest({
          messages: [
            { role: 'user', content: 'fix this bug', timestamp: Date.now() },
          ],
        }),
        mockState.lastExtensionContext!,
      );
      const retry = await vm.route(
        buildRequest({
          reason: 'retry',
          state: initialRoute.state,
          failed: {
            model: createTestModel({ provider: 'openai', id: 'gpt-4o-mini' }),
            thinkingLevel: 'medium',
            message: {
              ...createAssistantMsg('', 0),
              provider: 'openai',
              model: 'gpt-4o-mini',
              stopReason: 'error',
              errorMessage: '429 Too Many Requests',
            },
          },
        }),
        mockState.lastExtensionContext!,
      );
      expect(retry.model.id).toBe('gemini-1.5-flash');
      expect(retry.thinkingLevel).toBe('high');
      expect(mockState.lastDecision?.thinking).toBe('high');
    });

    it('should auto-pin on first turn when autoPinFirstTurnDecision is enabled and allow unpinning back to dynamic routing', async () => {
      mockState.currentConfig.autoPinFirstTurnDecision = true;
      mockState.currentConfig.profiles.balanced.low = {
        model: 'openai/gpt-4o-nano',
        thinking: 'low',
        resolvedContextWindow: 5000,
        resolvedThinkingLevels: ['low', 'medium', 'high'],
      };
      mockState.currentConfig.classifierModel = { model: 'auto' };
      const classifySpy = vi.fn().mockResolvedValue({
        api: 'typesafe-system-one',
        provider: 'typesafe',
        model: 'jev-latest',
        stopReason: 'stop',
        timestamp: Date.now(),
        answers: {
          tier: {
            type: 'choice',
            choice: 'low',
            probabilities: { high: 0.05, medium: 0.05, low: 0.9 },
            confidence: 0.9,
          },
        },
      });
      mockState.currentModelRegistry!.getAvailableOfType = vi
        .fn()
        .mockResolvedValue([
          {
            type: 'classifier',
            id: 'jev-latest',
            name: 'Jev Latest',
            api: 'typesafe-system-one',
            provider: 'typesafe',
            baseUrl: 'https://api.typesafe.ai',
            input: ['text'],
            contextWindow: 32000,
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
          },
        ]);
      mockState.currentModelRegistry!.classify = classifySpy;

      registerRouterProvider(mockPi, mockState, mockActions);
      const vm = registeredVirtualModels.get('router/balanced')!;

      // Turn 1: classifier selects low tier -> auto-pins balanced to low
      const turn1 = await vm.route(
        buildRequest({
          messages: [
            {
              role: 'user',
              content: 'summarize this file',
              timestamp: Date.now(),
            },
          ],
        }),
        mockState.lastExtensionContext!,
      );

      expect(turn1.model.id).toBe('gpt-4o-nano');
      expect(mockState.pinnedTierByProfile.balanced).toBe('low');
      expect(mockState.lastDecision?.reasoning).toContain(
        '(Auto-pinned to low tier from first turn.)',
      );
      expect(classifySpy).toHaveBeenCalledTimes(1);

      // Turn 2: complex architecture prompt stays pinned to low without calling classifier again
      const turn2 = await vm.route(
        buildRequest({
          state: turn1.state,
          messages: [
            {
              role: 'user',
              content: 'summarize this file',
              timestamp: Date.now(),
            },
            createAssistantMsg('summary'),
            {
              role: 'user',
              content:
                'design a comprehensive multi-region architecture migration plan',
              timestamp: Date.now(),
            },
          ],
        }),
        mockState.lastExtensionContext!,
      );

      expect(turn2.model.id).toBe('gpt-4o-nano');
      expect(mockState.pinnedTierByProfile.balanced).toBe('low');
      expect(classifySpy).toHaveBeenCalledTimes(1);

      // User unpins (/router pin auto) -> Turn 3 dynamically routes to high and does NOT re-auto-pin
      delete mockState.pinnedTierByProfile.balanced;
      classifySpy.mockResolvedValueOnce({
        api: 'typesafe-system-one',
        provider: 'typesafe',
        model: 'jev-latest',
        stopReason: 'stop',
        timestamp: Date.now(),
        answers: {
          tier: {
            type: 'choice',
            choice: 'high',
            probabilities: { high: 0.95, medium: 0.03, low: 0.02 },
            confidence: 0.95,
          },
        },
      });

      const turn3 = await vm.route(
        buildRequest({
          state: turn2.state,
          messages: [
            {
              role: 'user',
              content: 'summarize this file',
              timestamp: Date.now(),
            },
            createAssistantMsg('summary'),
            {
              role: 'user',
              content:
                'design a comprehensive multi-region architecture migration plan',
              timestamp: Date.now(),
            },
            createAssistantMsg('draft'),
            {
              role: 'user',
              content: 'now analyze the deep architectural tradeoffs',
              timestamp: Date.now(),
            },
          ],
        }),
        mockState.lastExtensionContext!,
      );

      expect(turn3.model.id).toBe('gpt-4o');
      expect(mockState.pinnedTierByProfile.balanced).toBeUndefined();
      expect(classifySpy).toHaveBeenCalledTimes(2);
    });
  });

  describe('cooldowns (integration)', () => {
    let cooldownDir: string;

    beforeEach(() => {
      cooldownDir = mkdtempSync(join(tmpdir(), 'pmr-provider-cooldown-'));
      Object.assign(mockState, {
        cooldownStore: createCooldownStore(
          join(cooldownDir, 'cooldowns'),
          () => Date.now(),
          join(cooldownDir, 'provider-switch.json'),
        ),
      });
    });

    afterEach(() => {
      rmSync(cooldownDir, { recursive: true, force: true });
    });

    it('records a cooldown on a retryable failure and skips the model on the very next route call', async () => {
      registerRouterProvider(mockPi, mockState, mockActions);
      const vm = registeredVirtualModels.get('router/balanced')!;

      const initialRoute = await vm.route(
        buildRequest({
          messages: [
            { role: 'user', content: 'fix this bug', timestamp: Date.now() },
          ],
        }),
        mockState.lastExtensionContext!,
      );
      expect(initialRoute.model.id).toBe('gpt-4o-mini');

      const retry = await vm.route(
        buildRequest({
          reason: 'retry',
          state: initialRoute.state,
          failed: {
            model: createTestModel({ provider: 'openai', id: 'gpt-4o-mini' }),
            thinkingLevel: 'medium',
            message: {
              ...createAssistantMsg('', 0),
              provider: 'openai',
              model: 'gpt-4o-mini',
              stopReason: 'error',
              errorMessage: '429 Too Many Requests',
            },
          },
        }),
        mockState.lastExtensionContext!,
      );
      expect(retry.model.id).toBe('gemini-1.5-flash');

      // A brand-new conversation (no carried-over state) must also skip the
      // cooling-down primary model, proving the cooldown is shared across sessions.
      const freshTurn = await vm.route(
        buildRequest({
          messages: [
            { role: 'user', content: 'fix this bug', timestamp: Date.now() },
          ],
        }),
        mockState.lastExtensionContext!,
      );
      expect(freshTurn.model.id).toBe('gemini-1.5-flash');
      expect(
        mockState.cooldownStore!.load().entries['openai/gpt-4o-mini'],
      ).toBeDefined();
    });

    it('records a non-retryable transcript error and skips the model on the next route call', async () => {
      registerRouterProvider(mockPi, mockState, mockActions);
      const vm = registeredVirtualModels.get('router/balanced')!;

      const route = await vm.route(
        buildRequest({
          reason: 'user',
          messages: [
            { role: 'user', content: 'fix this bug', timestamp: Date.now() },
            {
              ...createAssistantMsg('', 0),
              provider: 'openai',
              model: 'gpt-4o-mini',
              stopReason: 'error',
              errorMessage: 'insufficient_quota',
            },
            {
              role: 'user',
              content: 'please continue',
              timestamp: Date.now(),
            },
          ],
        }),
        mockState.lastExtensionContext!,
      );

      expect(route.model.id).toBe('gemini-1.5-flash');
      const entry =
        mockState.cooldownStore!.load().entries['openai/gpt-4o-mini'];
      expect(entry?.source).toBe('transcript');
    });

    it('keeps the soonest-to-expire model when every candidate for the tier is cooling down', async () => {
      const now = Date.now();
      mockState.cooldownStore!.record(
        'openai/gpt-4o-mini',
        'retry in 10 min',
        'retry',
        mockState.currentConfig,
        now,
      );
      mockState.cooldownStore!.record(
        'google/gemini-1.5-flash',
        'retry in 60 min',
        'retry',
        mockState.currentConfig,
        now,
      );

      registerRouterProvider(mockPi, mockState, mockActions);
      const vm = registeredVirtualModels.get('router/balanced')!;
      const route = await vm.route(
        buildRequest({
          messages: [
            { role: 'user', content: 'fix this bug', timestamp: Date.now() },
          ],
        }),
        mockState.lastExtensionContext!,
      );
      expect(route.model.id).toBe('gpt-4o-mini');
    });

    it('review finding 2: never re-arms a transcript-observed cooldown that was already cleared, even on a later turn', async () => {
      registerRouterProvider(mockPi, mockState, mockActions);
      const vm = registeredVirtualModels.get('router/balanced')!;
      const errorTimestamp = Date.now();
      const transcriptWithError: ModelRouteRequest<VirtualRouterState>['messages'] =
        [
          { role: 'user', content: 'fix this bug', timestamp: Date.now() },
          {
            ...createAssistantMsg('', 0),
            provider: 'openai',
            model: 'gpt-4o-mini',
            stopReason: 'error',
            errorMessage: 'insufficient_quota',
            timestamp: errorTimestamp,
          },
          { role: 'user', content: 'please continue', timestamp: Date.now() },
        ];

      const first = await vm.route(
        buildRequest({ reason: 'user', messages: transcriptWithError }),
        mockState.lastExtensionContext!,
      );
      expect(first.model.id).toBe('gemini-1.5-flash');
      expect(
        mockState.cooldownStore!.load().entries['openai/gpt-4o-mini'],
      ).toBeDefined();

      // An operator (or the model succeeding elsewhere) clears the cooldown to
      // force an immediate retry of the primary model. A brand-new conversation
      // (no carried-over sticky decision) proves the clear took effect rather
      // than riding leftover fallback stickiness from the first call.
      mockState.cooldownStore!.clear('openai/gpt-4o-mini');
      mockState.lastDecision = undefined;
      expect(
        mockState.cooldownStore!.load().entries['openai/gpt-4o-mini'],
      ).toBeUndefined();

      // The SAME historical error message is still in the transcript on the next
      // turn. Re-scanning it must not re-arm the cooldown the clear just lifted.
      const second = await vm.route(
        buildRequest({ reason: 'user', messages: transcriptWithError }),
        mockState.lastExtensionContext!,
      );
      expect(second.model.id).toBe('gpt-4o-mini');
      expect(
        mockState.cooldownStore!.load().entries['openai/gpt-4o-mini'],
      ).toBeUndefined();
    });

    it('review finding 7: keeps a cooling but usable model over an unusable (unresolvable) one, instead of throwing with zero usable candidates', async () => {
      mockState.currentConfig.profiles.balanced.medium!.fallbacks = [
        { model: 'google/gemini-1.5-flash' },
        // Not recognized by the mock registry's find() — simulates an
        // unresolvable/unauthenticated candidate.
        { model: 'untrusted/blocked' },
      ];
      const now = Date.now();
      // Every USABLE candidate is cooling; only the unusable one is not cooling.
      mockState.cooldownStore!.record(
        'openai/gpt-4o-mini',
        'retry in 5 min',
        'retry',
        mockState.currentConfig,
        now,
      );
      mockState.cooldownStore!.record(
        'google/gemini-1.5-flash',
        'retry in 10 min',
        'retry',
        mockState.currentConfig,
        now,
      );

      registerRouterProvider(mockPi, mockState, mockActions);
      const vm = registeredVirtualModels.get('router/balanced')!;

      // Pre-fix behavior would have filtered to only 'untrusted/blocked' (the
      // sole non-cooling candidate) and thrown "Failed to resolve any model",
      // breaking the turn. The fix keeps the soonest-expiring USABLE candidate.
      const route = await vm.route(
        buildRequest({
          messages: [
            { role: 'user', content: 'fix this bug', timestamp: Date.now() },
          ],
        }),
        mockState.lastExtensionContext!,
      );
      expect(route.model.id).toBe('gpt-4o-mini');
    });

    it('never records a transcript error timestamped before this process started', async () => {
      const historicalTimestamp = Date.now() - 60 * 60 * 1000;
      Object.assign(mockState, { processStartTime: Date.now() });
      registerRouterProvider(mockPi, mockState, mockActions);
      const vm = registeredVirtualModels.get('router/balanced')!;

      const route = await vm.route(
        buildRequest({
          reason: 'user',
          messages: [
            { role: 'user', content: 'fix this bug', timestamp: Date.now() },
            {
              ...createAssistantMsg('', 0),
              provider: 'openai',
              model: 'gpt-4o-mini',
              stopReason: 'error',
              errorMessage: 'insufficient_quota',
              timestamp: historicalTimestamp,
            },
            {
              role: 'user',
              content: 'please continue',
              timestamp: Date.now(),
            },
          ],
        }),
        mockState.lastExtensionContext!,
      );

      // The failure predates this process; it must be ignored, not re-recorded.
      expect(route.model.id).toBe('gpt-4o-mini');
      expect(
        mockState.cooldownStore!.load().entries['openai/gpt-4o-mini'],
      ).toBeUndefined();
    });

    it('review 3 finding 2: an older success already in the transcript never erases a cooldown recorded from a newer failure', async () => {
      registerRouterProvider(mockPi, mockState, mockActions);
      const vm = registeredVirtualModels.get('router/balanced')!;

      const successTimestamp = Date.now() - 2000;
      const errorTimestamp = Date.now() - 1000;
      const route = await vm.route(
        buildRequest({
          reason: 'user',
          messages: [
            { role: 'user', content: 'fix this bug', timestamp: Date.now() },
            {
              ...createAssistantMsg('', 0),
              provider: 'openai',
              model: 'gpt-4o-mini',
              stopReason: 'stop',
              timestamp: successTimestamp,
            },
            {
              ...createAssistantMsg('', 0),
              provider: 'openai',
              model: 'gpt-4o-mini',
              stopReason: 'error',
              errorMessage: 'insufficient_quota',
              timestamp: errorTimestamp,
            },
          ],
        }),
        mockState.lastExtensionContext!,
      );

      // The success predates the failure's recorded_at (both predate this
      // real route() call); it must not clear what the error just armed.
      expect(route.model.id).toBe('gemini-1.5-flash');
      expect(
        mockState.cooldownStore!.load().entries['openai/gpt-4o-mini'],
      ).toBeDefined();
    });

    it('review 3 finding 2: a success newer than the recorded failure clears the cooldown', async () => {
      registerRouterProvider(mockPi, mockState, mockActions);
      const vm = registeredVirtualModels.get('router/balanced')!;

      const errorTimestamp = Date.now();
      const firstTranscript: ModelRouteRequest<VirtualRouterState>['messages'] =
        [
          { role: 'user', content: 'fix this bug', timestamp: Date.now() },
          {
            ...createAssistantMsg('', 0),
            provider: 'openai',
            model: 'gpt-4o-mini',
            stopReason: 'error',
            errorMessage: 'insufficient_quota',
            timestamp: errorTimestamp,
          },
        ];

      await vm.route(
        buildRequest({ reason: 'user', messages: firstTranscript }),
        mockState.lastExtensionContext!,
      );
      expect(
        mockState.cooldownStore!.load().entries['openai/gpt-4o-mini'],
      ).toBeDefined();

      // A later success for the same model, timestamped strictly after the
      // failure's recorded_at (set during the call above), must clear it.
      const successTimestamp = Date.now() + 1000;
      mockState.lastDecision = undefined;
      await vm.route(
        buildRequest({
          reason: 'user',
          messages: [
            ...firstTranscript,
            {
              ...createAssistantMsg('', 0),
              provider: 'openai',
              model: 'gpt-4o-mini',
              stopReason: 'stop',
              timestamp: successTimestamp,
            },
          ],
        }),
        mockState.lastExtensionContext!,
      );
      expect(
        mockState.cooldownStore!.load().entries['openai/gpt-4o-mini'],
      ).toBeUndefined();
    });

    it('review 4 finding 2: a single scan seeing [failure t1, success t2] (t1 < t2 < scan) is NOT left cooling', async () => {
      registerRouterProvider(mockPi, mockState, mockActions);
      const vm = registeredVirtualModels.get('router/balanced')!;

      const failureTimestamp = Date.now() - 2000;
      const successTimestamp = Date.now() - 1000;
      const route = await vm.route(
        buildRequest({
          reason: 'user',
          messages: [
            { role: 'user', content: 'fix this bug', timestamp: Date.now() },
            {
              ...createAssistantMsg('', 0),
              provider: 'openai',
              model: 'gpt-4o-mini',
              stopReason: 'error',
              errorMessage: 'insufficient_quota',
              timestamp: failureTimestamp,
            },
            {
              ...createAssistantMsg('', 0),
              provider: 'openai',
              model: 'gpt-4o-mini',
              stopReason: 'stop',
              timestamp: successTimestamp,
            },
          ],
        }),
        mockState.lastExtensionContext!,
      );

      // recorded_at must come from the failure's OWN timestamp (t1), not the
      // scan time: the success (t2), though both predate this scan, is still
      // newer than t1 and must clear the cooldown the failure just armed.
      expect(route.model.id).toBe('gpt-4o-mini');
      expect(
        mockState.cooldownStore!.load().entries['openai/gpt-4o-mini'],
      ).toBeUndefined();
    });

    it('review 4 finding 2: a single scan seeing [success t1, failure t2] (t1 < t2 < scan) leaves the model cooling', async () => {
      registerRouterProvider(mockPi, mockState, mockActions);
      const vm = registeredVirtualModels.get('router/balanced')!;

      const successTimestamp = Date.now() - 2000;
      const failureTimestamp = Date.now() - 1000;
      const route = await vm.route(
        buildRequest({
          reason: 'user',
          messages: [
            { role: 'user', content: 'fix this bug', timestamp: Date.now() },
            {
              ...createAssistantMsg('', 0),
              provider: 'openai',
              model: 'gpt-4o-mini',
              stopReason: 'stop',
              timestamp: successTimestamp,
            },
            {
              ...createAssistantMsg('', 0),
              provider: 'openai',
              model: 'gpt-4o-mini',
              stopReason: 'error',
              errorMessage: 'insufficient_quota',
              timestamp: failureTimestamp,
            },
          ],
        }),
        mockState.lastExtensionContext!,
      );

      // The success (t1) predates the failure's own event time (t2), so it
      // must not clear the cooldown the failure arms.
      expect(route.model.id).toBe('gemini-1.5-flash');
      expect(
        mockState.cooldownStore!.load().entries['openai/gpt-4o-mini'],
      ).toBeDefined();
    });

    describe('review 5: malformed numeric transcript timestamps never escape route()', () => {
      // Pre-fix, provider.ts accepted any numeric candidate.timestamp as-is
      // and forwarded it straight to cooldownStore.record() as recordedAt;
      // record() built the cooldown entry (iso(recordedAt)) outside its own
      // try/catch, so a non-finite or out-of-Date-range transcript timestamp
      // threw a RangeError out of route() instead of being ignored.
      it.each([
        ['1e20', 1e20],
        ['NaN', NaN],
        ['Infinity', Infinity],
        ['-1', -1],
      ])(
        'route() still returns a routed model for a transcript failure timestamped %s',
        async (_label, timestamp) => {
          // processStartTime far below every malformed value (including -1)
          // so none of them are filtered out by the "before this process
          // started" check before ever reaching the timestamp validity guard.
          Object.assign(mockState, { processStartTime: -1e21 });
          registerRouterProvider(mockPi, mockState, mockActions);
          const vm = registeredVirtualModels.get('router/balanced')!;

          const route = await vm.route(
            buildRequest({
              reason: 'user',
              messages: [
                {
                  role: 'user',
                  content: 'fix this bug',
                  timestamp: Date.now(),
                },
                {
                  ...createAssistantMsg('', 0),
                  provider: 'openai',
                  model: 'gpt-4o-mini',
                  stopReason: 'error',
                  errorMessage: 'insufficient_quota',
                  timestamp,
                },
              ],
            }),
            mockState.lastExtensionContext!,
          );

          expect(route.model).toBeDefined();
        },
      );

      it('route() still returns a routed model for a transcript success timestamped 1e20', async () => {
        Object.assign(mockState, { processStartTime: -1e21 });
        registerRouterProvider(mockPi, mockState, mockActions);
        const vm = registeredVirtualModels.get('router/balanced')!;

        const route = await vm.route(
          buildRequest({
            reason: 'user',
            messages: [
              {
                role: 'user',
                content: 'fix this bug',
                timestamp: Date.now(),
              },
              {
                ...createAssistantMsg('', 0),
                provider: 'openai',
                model: 'gpt-4o-mini',
                stopReason: 'stop',
                timestamp: 1e20,
              },
            ],
          }),
          mockState.lastExtensionContext!,
        );

        expect(route.model).toBeDefined();
      });
    });

    describe('review 5: structural guard -- a throwing CooldownStore never breaks route()', () => {
      it('route() still returns a routed model when every CooldownStore method throws', async () => {
        const throwingStore: CooldownStore = {
          load: () => {
            throw new Error('boom: load');
          },
          record: () => {
            throw new Error('boom: record');
          },
          noteSuccess: () => {
            throw new Error('boom: noteSuccess');
          },
          candidates: () => {
            throw new Error('boom: candidates');
          },
          clear: () => {
            throw new Error('boom: clear');
          },
          formatStatus: () => {
            throw new Error('boom: formatStatus');
          },
        };
        Object.assign(mockState, { cooldownStore: throwingStore });
        const errorSpy = vi
          .spyOn(console, 'error')
          .mockImplementation(() => {});
        registerRouterProvider(mockPi, mockState, mockActions);
        const vm = registeredVirtualModels.get('router/balanced')!;

        const route = await vm.route(
          buildRequest({
            messages: [
              { role: 'user', content: 'fix this bug', timestamp: Date.now() },
            ],
          }),
          mockState.lastExtensionContext!,
        );

        expect(route.model).toBeDefined();
        errorSpy.mockRestore();
      });

      it('a throwing CooldownStore still lets a retry fall back to another model', async () => {
        const throwingStore: CooldownStore = {
          load: () => {
            throw new Error('boom: load');
          },
          record: () => {
            throw new Error('boom: record');
          },
          noteSuccess: () => {
            throw new Error('boom: noteSuccess');
          },
          candidates: () => {
            throw new Error('boom: candidates');
          },
          clear: () => {
            throw new Error('boom: clear');
          },
          formatStatus: () => {
            throw new Error('boom: formatStatus');
          },
        };
        Object.assign(mockState, { cooldownStore: throwingStore });
        const errorSpy = vi
          .spyOn(console, 'error')
          .mockImplementation(() => {});
        registerRouterProvider(mockPi, mockState, mockActions);
        const vm = registeredVirtualModels.get('router/balanced')!;

        const initialRoute = await vm.route(
          buildRequest({
            messages: [
              { role: 'user', content: 'fix this bug', timestamp: Date.now() },
            ],
          }),
          mockState.lastExtensionContext!,
        );

        const retry = await vm.route(
          buildRequest({
            reason: 'retry',
            state: initialRoute.state,
            failed: {
              model: createTestModel({ provider: 'openai', id: 'gpt-4o-mini' }),
              thinkingLevel: 'medium',
              message: {
                ...createAssistantMsg('', 0),
                provider: 'openai',
                model: 'gpt-4o-mini',
                stopReason: 'error',
                errorMessage: '429 Too Many Requests',
              },
            },
          }),
          mockState.lastExtensionContext!,
        );

        expect(retry.model).toBeDefined();
        errorSpy.mockRestore();
      });
    });
  });

  describe('cost tracking helpers', () => {
    it('should count each assistant and toolResult message object only once, including turn_end events', () => {
      const msg = createAssistantMsg('resp', 0.012);
      expect(extractUncountedMessageCost(msg)).toBe(0.012);
      expect(extractUncountedMessageCost(msg)).toBe(0);

      const preCounted = createAssistantMsg('old', 0.05);
      markMessageCostCounted(preCounted);
      expect(extractUncountedMessageCost(preCounted)).toBe(0);

      const toolResultWithUsage = {
        role: 'toolResult',
        toolCallId: 'c1',
        toolName: 'codemode',
        content: [{ type: 'text', text: 'classified' }],
        isError: false,
        usage: {
          input: 1000,
          output: 10,
          cacheRead: 0,
          cacheWrite: 0,
          totalTokens: 1010,
          cost: {
            input: 0.004,
            output: 0,
            cacheRead: 0,
            cacheWrite: 0,
            total: 0.004,
          },
        },
        timestamp: Date.now(),
      };
      const turnEndMsg = createAssistantMsg('turn-end', 0.01);
      expect(
        extractUncountedMessageCost({
          message: turnEndMsg,
          toolResults: [toolResultWithUsage],
        }),
      ).toBeCloseTo(0.014);
      // Subsequent check should not double-count either message or toolResult
      expect(extractUncountedMessageCost(turnEndMsg)).toBe(0);
      expect(extractUncountedMessageCost(toolResultWithUsage)).toBe(0);
    });
  });
});
