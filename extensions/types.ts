import type { ThinkingLevel } from '@earendil-works/pi-agent-core';
import type {
  Api,
  ClassifierApi,
  ClassifierContext,
  ClassifierModel,
  ClassifierResult,
  Model,
} from '@earendil-works/pi-ai';
import type {
  ExtensionAPI,
  ExtensionContext,
  ModelRoute,
  ModelRouteReason,
  ModelRouteRequest,
  VirtualModelDefinition,
} from '@earendil-works/pi-coding-agent';

export type {
  ClassifierApi,
  ClassifierContext,
  ClassifierModel,
  ClassifierResult,
  ModelRoute,
  ModelRouteReason,
  ModelRouteRequest,
};

export type RouterTier = 'high' | 'medium' | 'low';
export type RouterPin = RouterTier | 'auto';
export type RouterPhase = 'planning' | 'implementation' | 'lightweight';
export type RouterPinByProfile = Partial<Record<string, RouterTier>>;
export type RouterThinkingByTier = Partial<Record<RouterTier, ThinkingLevel>>;
export type RouterThinkingByProfile = Record<string, RouterThinkingByTier>;

export interface RouterModelCapabilityLookup {
  find: (
    provider: string,
    modelId: string,
  ) =>
    | {
        contextWindow?: number;
        maxTokens?: number;
        reasoning?: boolean;
        thinkingLevelMap?: Partial<Record<ThinkingLevel, string | null>>;
        input?: readonly string[];
      }
    | undefined;
}

export type RouterModelLookup = Pick<ExtensionContext['modelRegistry'], 'find'>;

export type RouterModelRegistry = Pick<
  ExtensionContext['modelRegistry'],
  'find'
> &
  Partial<
    Pick<
      ExtensionContext['modelRegistry'],
      | 'hasConfiguredAuth'
      | 'findOfType'
      | 'getAvailableOfType'
      | 'classify'
      | 'streamSimple'
      // Used by the account-alias feature (extensions/accounts.ts) to clone
      // a base provider's complete native registration under a new id.
      | 'getProvider'
    >
  >;

export interface ClassifierDecisionResult {
  tier: RouterTier;
  reasoning: string;
  cost?: number;
}

export interface RouterUIContext {
  ui: {
    setStatus: (key: string, text: string | undefined) => void;
    setWidget: (key: string, content: string[] | undefined) => void;
    theme: {
      fg: (color: 'dim', text: string) => string;
    };
  };
}

export interface RouterCommandContext {
  ui: {
    notify: (message: string, type?: 'info' | 'warning' | 'error') => void;
    setStatus: (key: string, text: string | undefined) => void;
    setWidget: (key: string, content: string[] | undefined) => void;
  };
  // RouterModelRegistry (not the narrower RouterModelLookup) so command
  // handlers can reach the optional `getProvider` the account-alias feature
  // (extensions/accounts.ts) needs; every other pick stays optional so
  // existing fakes that only implement `find` still satisfy this type.
  modelRegistry: RouterModelRegistry;
  model: Pick<Model<Api>, 'provider' | 'id'> | undefined;
}

export interface RouterExtensionContext extends RouterUIContext {
  cwd: string;
  modelRegistry: RouterModelRegistry;
  model:
    | {
        provider: string;
        id: string;
        contextWindow?: number;
        maxTokens?: number;
      }
    | undefined;
  sessionManager: {
    getBranch: () => readonly CustomSessionEntry[];
    // Pi 1.4.0 `SessionManager.getEntries(): SessionEntry[]` -- every entry in
    // the session, not just the active branch -- narrowed to the fields this
    // extension reads.
    getEntries: () => readonly CustomSessionEntry[];
  };
  ui: RouterUIContext['ui'] & RouterCommandContext['ui'];
}

export interface RouterModelSelectEvent {
  model: {
    provider: string;
    id: string;
    contextWindow?: number;
    maxTokens?: number;
  };
}

export interface RouterThinkingLevelSelectEvent {
  level: ThinkingLevel;
}

// Mirrors Pi's real `CustomMessageEntryDraft` (a `SessionBoundaryDraft`
// member): a session-persisted `role: "custom"` entry, tagged with
// `customType`, never a literal `role: "user"` message. Narrowed to exactly
// the fields extensions/autocontinue.ts produces.
export interface RouterTurnEndEntryDraft {
  type: 'custom_message';
  customType: string;
  content: string;
  display: boolean;
}

// Mirrors Pi's real `BoundaryResult` (`TurnEndEventResult`): returning
// `continue: true` from a `turn_end` handler requests exactly one more model
// request for the same turn.
export interface RouterTurnEndResult {
  entries?: RouterTurnEndEntryDraft[];
  continue?: boolean;
}

export interface VirtualRouterState {
  phase: RouterPhase;
  tier: RouterTier;
  lastDecision: RoutingDecision;
}

export interface RouterVirtualModelDefinition<
  TState = VirtualRouterState,
  TCtx = ExtensionContext,
> extends Omit<VirtualModelDefinition<TState>, 'route'> {
  route(
    request: ModelRouteRequest<TState>,
    ctx: TCtx,
  ): ModelRoute<TState> | Promise<ModelRoute<TState>>;
}

export interface RouterExtensionAPI<
  TCtx extends RouterExtensionContext = ExtensionContext,
> {
  registerVirtualModel: (
    model: RouterVirtualModelDefinition<VirtualRouterState, TCtx>,
  ) => void;
  unregisterVirtualModel?: (provider: string, id: string) => void;
  registerProvider?: ExtensionAPI['registerProvider'];
  unregisterProvider?: ExtensionAPI['unregisterProvider'];
  registerCommand: (
    name: string,
    options: {
      description?: string;
      getArgumentCompletions?: (argumentPrefix: string) =>
        | Array<{ value: string; label: string; description?: string }>
        | null
        | Promise<Array<{
            value: string;
            label: string;
            description?: string;
          }> | null>;
      handler: (args: string, ctx: TCtx) => Promise<void>;
    },
  ) => void;
  setModel: ExtensionAPI['setModel'];
  setThinkingLevel: ExtensionAPI['setThinkingLevel'];
  appendEntry: ExtensionAPI['appendEntry'];
  on(
    event: 'session_start' | 'session_tree' | 'turn_start',
    handler: (event: unknown, ctx: TCtx) => Promise<void> | void,
  ): void;
  // turn_end is Pi's "actionable boundary" for the active turn. A handler's
  // `{ entries, continue: true }` return is only honored here when Pi's own
  // `finishTurn` decision was NOT already "end" -- which it always is for a
  // non-retryable error, since Pi itself gave up on that turn. That makes
  // turn_end unusable for auto-continue's actual use case; it is kept purely
  // for non-continuation bookkeeping (cost accounting).
  on(
    event: 'turn_end',
    handler: (
      event: unknown,
      ctx: TCtx,
    ) => Promise<RouterTurnEndResult | void> | RouterTurnEndResult | void,
  ): void;
  // agent_before_settle fires once per run, right before the run loop would
  // actually stop, independent of turn_end's (possibly discarded) decision.
  // A handler's `{ entries, continue: true }` here is honored unconditionally
  // (as long as the resulting context can still continue -- i.e. the last
  // message is not itself an unanswered assistant message), which is why
  // `extensions/autocontinue.ts` makes its actual continuation decision here,
  // not in turn_end. Reuses `RouterTurnEndResult`'s shape: Pi's real
  // `BoundaryResult` is identical for both boundaries.
  on(
    event: 'agent_before_settle',
    handler: (
      event: unknown,
      ctx: TCtx,
    ) => Promise<RouterTurnEndResult | void> | RouterTurnEndResult | void,
  ): void;
  // agent_settled fires exactly once per run, after the agent_before_settle
  // loop above has fully stopped requesting continuations -- the only point
  // at which extensions/autocontinue.ts's per-run "never chain" guard may
  // safely reset for a later, genuinely independent run. Never fires more
  // than once per run, unlike agent_before_settle.
  on(
    event: 'agent_settled',
    handler: (event: unknown, ctx: TCtx) => Promise<void> | void,
  ): void;
  on(
    event: 'model_select',
    handler: (event: RouterModelSelectEvent, ctx: TCtx) => Promise<void> | void,
  ): void;
  on(
    event: 'thinking_level_select',
    handler: (
      event: RouterThinkingLevelSelectEvent,
      ctx: TCtx,
    ) => Promise<void> | void,
  ): void;
}

export interface RoutingRule {
  matches: string | string[];
  tier: RouterTier;
  reason?: string;
}

export interface ModelDefinition {
  model: string;
  contextWindow?: number;
  maxTokens?: number;
  reasoning?: boolean;
  thinkingLevels?: ThinkingLevel[];
}

export interface ClassifierConfig {
  model: string;
  thinking?: ThinkingLevel;
}

export interface FallbackModel {
  model: string;
  thinking?: ThinkingLevel;
}

export interface RoutedTierConfig {
  model: string;
  thinking?: ThinkingLevel;
  fallbacks?: FallbackModel[];
  contextWindow?: number;
  maxTokens?: number;
  reasoning?: boolean;
  thinkingLevels?: ThinkingLevel[];
  resolvedContextWindow?: number;
  resolvedMaxTokens?: number;
  resolvedThinkingLevels?: ThinkingLevel[];
}

export interface RouterProfile {
  high?: RoutedTierConfig;
  medium?: RoutedTierConfig;
  low?: RoutedTierConfig;
}

export interface RouterConfig {
  debug?: boolean;
  autoPinFirstTurnDecision?: boolean;
  classifierModel?: ClassifierConfig;
  phaseBias?: number;
  maxSessionBudget?: number;
  rules?: RoutingRule[];
  profiles: Record<string, RouterProfile>;
  models?: Record<string, ModelDefinition>;
  providerTimezones?: Record<string, string>;
}

export interface RoutingDecision {
  profile: string;
  tier: RouterTier;
  phase: RouterPhase;
  targetProvider: string;
  targetModelId: string;
  targetLabel: string;
  reasoning: string;
  thinking: ThinkingLevel;
  timestamp: number;
  isClassifier?: boolean;
  isFallback?: boolean;
  isBudgetForced?: boolean;
  isRuleMatched?: boolean;
}

export interface RouterPersistedState {
  enabled: boolean;
  selectedProfile: string;
  pinTier?: RouterTier;
  pinByProfile?: RouterPinByProfile;
  thinkingByProfile?: RouterThinkingByProfile;
  debugEnabled?: boolean;
  widgetEnabled?: boolean;
  debugHistory?: RoutingDecision[];
  lastPhase?: RouterPhase;
  lastDecision?: RoutingDecision;
  lastNonRouterModel?: string;
  accumulatedCost?: number;
  timestamp: number;
}

export interface ConfigLoadResult {
  config: RouterConfig;
  warnings: string[];
}

export interface ParsedConfigFile {
  config: Partial<RouterConfig>;
  warnings: string[];
}

export interface CustomSessionEntry {
  type: string;
  customType?: string;
  data?: unknown;
  message?: unknown;
}
