import type {
  ExtensionAPI,
  ExtensionCommandContext,
} from '@earendil-works/pi-coding-agent';
import type { ThinkingLevel } from '@earendil-works/pi-agent-core';
import type { AutocompleteItem } from '@earendil-works/pi-tui';
import type {
  RouterConfig,
  RouterPinByProfile,
  RouterThinkingByProfile,
  RoutingDecision,
  RouterTier,
  RouterCommandContext,
} from './types';
import type { CooldownStore } from './cooldown';
import {
  ACCOUNT_BASE_ALLOWLIST,
  ACCOUNT_BASE_DENYLIST,
  claimNextAccountSlot,
  cloneProviderAsAccount,
  listAliasesForBase,
  loadAccountRegistry,
  markAccountRemoved,
  parseAliasId,
  readAccountEntry,
  registerAccountProvider,
  unregisterAccountProvider,
} from './accounts';
import {
  profileNames,
  THINKING_LEVELS,
  ROUTER_PIN_VALUES,
  ROUTER_TIERS,
  isRouterTier,
  isThinkingLevel,
  findModelByRef,
  applyThinkingOverride,
  getUnsupportedTiers,
} from './config';
import {
  formatPinSummary,
  formatThinkingSummary,
  formatModelRef,
  formatSessionCost,
  formatDecision,
  formatUnsupportedThinkingWarning,
} from './ui';

export const registerCommands = <
  TCtx extends RouterCommandContext = ExtensionCommandContext,
>(
  pi: {
    registerCommand: (
      name: string,
      options: {
        description?: string;
        getArgumentCompletions?: (
          argumentPrefix: string,
        ) => AutocompleteItem[] | null | Promise<AutocompleteItem[] | null>;
        handler: (args: string, ctx: TCtx) => Promise<void>;
      },
    ) => void;
    setModel: ExtensionAPI['setModel'];
    registerProvider?: ExtensionAPI['registerProvider'];
    unregisterProvider?: ExtensionAPI['unregisterProvider'];
  },
  state: {
    readonly currentConfig: RouterConfig;
    routerEnabled: boolean;
    selectedProfile: string | undefined;
    readonly pinnedTierByProfile: RouterPinByProfile;
    readonly thinkingByProfile: RouterThinkingByProfile;
    readonly lastDecision: RoutingDecision | undefined;
    lastNonRouterModel: string | undefined;
    readonly accumulatedCost: number;
    debugEnabled: boolean;
    widgetEnabled: boolean;
    readonly debugHistory: RoutingDecision[];
    readonly lastConfigWarnings: string[];
    readonly cooldownStore?: CooldownStore;
    // Provider ids this process has itself registered for the account-alias
    // feature (see extensions/accounts.ts's registerAccountProvider /
    // unregisterAccountProvider). Never persisted.
    readonly ownedAccountProviderIds: Set<string>;
  },
  actions: {
    persistState: () => void;
    updateStatus: (ctx: TCtx) => void;
    reloadConfig: (ctx?: TCtx, options?: { preserveDebug?: boolean }) => void;
    ensureValidActiveRouterProfile: (ctx: TCtx) => Promise<void>;
    switchToRouterProfile: (
      profileName: string,
      ctx: TCtx,
      strict?: boolean,
    ) => Promise<boolean>;
    syncPiThinkingLevel: (level: ThinkingLevel | undefined) => void;
  },
) => {
  const SUBCOMMAND_DETAILS = [
    { name: 'status', desc: 'Show current router status' },
    { name: 'profile', desc: 'Switch to a different router profile' },
    { name: 'pin', desc: 'Pin routing for a profile to a specific tier' },
    { name: 'thinking', desc: 'Override thinking level for a tier or profile' },
    { name: 'disable', desc: 'Disable the router and restore last model' },
    {
      name: 'account',
      desc: 'Manage extra provider accounts (add|list|remove)',
    },
    {
      name: 'fix',
      desc: 'Correct the last routing decision and pin that tier',
    },
    { name: 'widget', desc: 'Toggle the router status widget' },
    { name: 'debug', desc: 'Toggle or clear router debug history' },
    { name: 'cooldowns', desc: 'Show shared model cooldowns' },
    { name: 'cooldown', desc: 'Clear shared model cooldowns' },
    { name: 'reload', desc: 'Reload the model router configuration' },
    { name: 'help', desc: 'Show usage help for subcommands' },
  ];

  const getSubcommandCompletions = (
    prefix: string,
  ): AutocompleteItem[] | null => {
    const items = SUBCOMMAND_DETAILS.filter((s) =>
      s.name.startsWith(prefix),
    ).map((s) => ({
      value: s.name,
      label: s.name,
      description: s.desc,
    }));
    return items.length > 0 ? items : null;
  };

  const getPinCompletions = (args: string[]): AutocompleteItem[] | null => {
    // pin <tier|auto>
    if (args.length <= 1) {
      const token = args[0] ?? '';
      const items = ROUTER_PIN_VALUES.filter((value) =>
        value.startsWith(token),
      ).map((value) => ({
        value,
        label: value,
        description:
          value === 'auto'
            ? 'Restore auto-routing (clear pin) for the active profile'
            : `Pin active profile to ${value} tier`,
      }));
      return items.length > 0 ? items : null;
    }
    return null;
  };

  const getThinkingCompletions = (
    args: string[],
  ): AutocompleteItem[] | null => {
    // thinking [tier] <level|auto>
    const tierValues = [...ROUTER_TIERS];
    const levelValues = ['auto', ...THINKING_LEVELS];

    if (args.length <= 1) {
      const token = args[0] ?? '';
      return [
        ...levelValues
          .filter((v) => v.startsWith(token))
          .map((v) => ({
            value: v,
            label: v,
            description:
              v === 'auto'
                ? 'Restore default thinking level'
                : `Set thinking level to ${v}`,
          })),
        ...tierValues
          .filter((v) => v.startsWith(token))
          .map((v) => ({
            value: v,
            label: v,
            description: `Override thinking for ${v} tier`,
          })),
      ];
    }

    if (levelValues.includes(args[0])) {
      return null;
    }

    if ((tierValues as string[]).includes(args[0])) {
      const tier = args[0];
      const levelPrefix = args[1] ?? '';
      return levelValues
        .filter((v) => v.startsWith(levelPrefix))
        .map((v) => ({
          value: `${tier} ${v}`,
          label: `${tier} ${v}`,
          description:
            v === 'auto'
              ? `Restore default thinking level for ${tier} tier`
              : `Set thinking level to ${v} for ${tier} tier`,
        }));
    }

    return null;
  };

  const handleStatus = async (args: string[], ctx: TCtx) => {
    if (args.length > 0) {
      ctx.ui.notify('Usage: /router status (no arguments)', 'error');
      return;
    }
    const names = profileNames(state.currentConfig).join(', ');
    const lines = [
      'Model Router Status:',
      `Router enabled: ${state.routerEnabled ? 'yes' : 'off'}`,
      `Selected profile: ${state.selectedProfile ?? 'none'}`,
      `Selected profile pin: ${state.selectedProfile ? (state.pinnedTierByProfile[state.selectedProfile] ?? 'auto') : 'none'}`,
      `Auto-pin first turn: ${state.currentConfig.autoPinFirstTurnDecision ? 'on' : 'off'}`,
      `Pins by profile: ${formatPinSummary(state.pinnedTierByProfile)}`,
      `Thinking overrides: ${formatThinkingSummary(state.thinkingByProfile)}`,
      `Widget: ${state.widgetEnabled ? 'on' : 'off'}`,
      `Phase bias: ${state.currentConfig.phaseBias}`,
      `Session cost: ${formatSessionCost(state.accumulatedCost, state.currentConfig.maxSessionBudget)}`,
      `Available profiles: ${names}`,
      `Last non-router model: ${formatModelRef(state.lastNonRouterModel)}`,
      `Debug: ${state.debugEnabled ? 'on' : 'off'}`,
      `Debug history: ${state.debugHistory.length} decisions`,
    ];
    const cooldownLines = state.cooldownStore?.formatStatus() ?? [];
    lines.push(
      '',
      'Cooldowns:',
      ...(cooldownLines.length > 0 ? cooldownLines : ['none']),
    );
    if (state.lastDecision) {
      lines.push(
        `Last routed tier: ${state.lastDecision.tier}`,
        `Last phase: ${state.lastDecision.phase}`,
        `Last model: ${state.lastDecision.targetProvider}/${state.lastDecision.targetModelId} (${state.lastDecision.thinking})`,
        `Reason: ${state.lastDecision.reasoning}`,
      );
    }
    if (state.lastConfigWarnings && state.lastConfigWarnings.length > 0) {
      lines.push(
        '',
        '⚠️ Configuration Warnings:',
        ...state.lastConfigWarnings.map((w) => `  - ${w}`),
      );
    }
    ctx.ui.notify(lines.join('\n'), 'info');
    actions.updateStatus(ctx);
  };

  const handleProfile = async (args: string[], ctx: TCtx) => {
    if (args.length > 1) {
      ctx.ui.notify('Usage: /router profile [name]', 'error');
      return;
    }
    const profileName = args[0];
    if (!profileName) {
      ctx.ui.notify(
        `Current profile: ${state.selectedProfile}. Available: ${profileNames(state.currentConfig).join(', ')}`,
        'info',
      );
      return;
    }
    const success = await actions.switchToRouterProfile(profileName, ctx);
    if (success) {
      ctx.ui.notify(
        `Switched to router profile: ${state.selectedProfile}`,
        'info',
      );
    }
  };

  const handlePin = async (args: string[], ctx: TCtx) => {
    const currentProfile = state.selectedProfile;
    if (!currentProfile) {
      ctx.ui.notify(
        'No router profile is active. Select a router model first.',
        'error',
      );
      return;
    }
    if (args.length === 0) {
      ctx.ui.notify(
        [
          `Profile: ${currentProfile}`,
          `Pinned tier: ${state.pinnedTierByProfile[currentProfile] ?? 'auto'}`,
          `Usage: /router pin <high|medium|low|auto>`,
        ].join('\n'),
        'info',
      );
      actions.updateStatus(ctx);
      return;
    }

    if (args.length > 1) {
      ctx.ui.notify('Usage: /router pin <high|medium|low|auto>', 'error');
      return;
    }

    const pinValue = args[0];

    if (
      !ROUTER_PIN_VALUES.includes(
        pinValue as (typeof ROUTER_PIN_VALUES)[number],
      )
    ) {
      ctx.ui.notify(
        `Invalid router pin: ${pinValue}. Use one of: ${ROUTER_PIN_VALUES.join(', ')}`,
        'error',
      );
      return;
    }

    const nextTier = pinValue === 'auto' ? undefined : (pinValue as RouterTier);
    if (nextTier) {
      state.pinnedTierByProfile[currentProfile] = nextTier;
    } else {
      delete state.pinnedTierByProfile[currentProfile];
    }
    actions.persistState();
    actions.updateStatus(ctx);
    ctx.ui.notify(
      nextTier
        ? `Router pinned to ${nextTier}`
        : `Router pin cleared; heuristic routing restored`,
      'info',
    );
  };

  const handleThinking = async (args: string[], ctx: TCtx) => {
    const currentProfile = state.selectedProfile;
    if (!currentProfile) {
      ctx.ui.notify(
        'No router profile is active. Select a router model first.',
        'error',
      );
      return;
    }
    if (args.length === 0) {
      ctx.ui.notify(
        [
          `Profile: ${currentProfile}`,
          `Thinking overrides: ${JSON.stringify(state.thinkingByProfile[currentProfile] ?? {})}`,
          'Usage: /router thinking <level|auto>           (applies to all tiers)',
          '   or: /router thinking <tier> <level|auto>    (applies to one tier)',
          'Note: not all tier models may support every thinking level.',
        ].join('\n'),
        'info',
      );
      return;
    }

    if (args.length > 2) {
      ctx.ui.notify('Too many arguments for /router thinking.', 'error');
      return;
    }

    const tierArg = args.length === 1 ? 'all' : args[0];
    if (tierArg !== 'all' && !isRouterTier(tierArg)) {
      ctx.ui.notify(
        `Invalid tier: ${tierArg}. Use high, medium, or low.`,
        'error',
      );
      return;
    }
    const tier: RouterTier | 'all' = tierArg;
    const levelValue = args.length === 1 ? args[0] : args[1];
    if (levelValue !== 'auto' && !isThinkingLevel(levelValue)) {
      ctx.ui.notify(
        `Invalid thinking level: ${levelValue}. Use auto or: ${THINKING_LEVELS.join(', ')}`,
        'error',
      );
      return;
    }

    const nextLevel = levelValue === 'auto' ? undefined : levelValue;
    applyThinkingOverride(
      state.thinkingByProfile,
      currentProfile,
      tier,
      nextLevel,
    );

    actions.persistState();
    actions.updateStatus(ctx);
    actions.syncPiThinkingLevel(nextLevel ?? state.lastDecision?.thinking);
    if (nextLevel) {
      const unsupported = getUnsupportedTiers(
        state.currentConfig.profiles[currentProfile],
        nextLevel,
        ctx.modelRegistry,
      );
      if (unsupported.length > 0) {
        ctx.ui.notify(
          formatUnsupportedThinkingWarning(tier, nextLevel, unsupported),
          'warning',
        );
      }
    }
  };

  const handleDisable = async (args: string[], ctx: TCtx) => {
    if (args.length > 0) {
      ctx.ui.notify('Usage: /router disable (no arguments)', 'error');
      return;
    }
    if (!state.lastNonRouterModel) {
      ctx.ui.notify(
        'No previous non-router model recorded. Use /model to pick a concrete model.',
        'warning',
      );
      return;
    }
    const targetModel = findModelByRef(
      state.lastNonRouterModel,
      ctx.modelRegistry,
    );
    if (!targetModel) {
      ctx.ui.notify(
        `Recorded non-router model is unavailable: ${state.lastNonRouterModel}`,
        'error',
      );
      return;
    }
    const success = await pi.setModel(targetModel);
    if (!success) {
      ctx.ui.notify(`Failed to switch to ${state.lastNonRouterModel}`, 'error');
      return;
    }
    state.routerEnabled = false;
    actions.persistState();
    actions.updateStatus(ctx);
    ctx.ui.notify(
      `Router disabled. Restored ${state.lastNonRouterModel}`,
      'info',
    );
  };

  const handleFix = async (args: string[], ctx: TCtx) => {
    if (args.length !== 1) {
      ctx.ui.notify('Usage: /router fix <high|medium|low>', 'error');
      return;
    }
    const tier = args[0]?.toLowerCase();
    if (!ROUTER_TIERS.includes(tier as RouterTier)) {
      ctx.ui.notify('Usage: /router fix <high|medium|low>', 'error');
      return;
    }
    if (!state.lastDecision) {
      ctx.ui.notify('No recent routing decision to fix.', 'warning');
      return;
    }
    state.pinnedTierByProfile[state.lastDecision.profile] = tier as RouterTier;
    actions.persistState();
    actions.updateStatus(ctx);
    ctx.ui.notify(
      `Router decision corrected. ${state.lastDecision.profile} is now pinned to ${tier}.`,
      'info',
    );
  };

  const handleWidget = async (args: string[], ctx: TCtx) => {
    if (args.length > 1) {
      ctx.ui.notify('Usage: /router widget <on|off|toggle>', 'error');
      return;
    }
    const cmd = args[0]?.toLowerCase();
    if (cmd === 'on') state.widgetEnabled = true;
    else if (cmd === 'off') state.widgetEnabled = false;
    else state.widgetEnabled = !state.widgetEnabled;
    actions.persistState();
    actions.updateStatus(ctx);
    ctx.ui.notify(
      `Router widget ${state.widgetEnabled ? 'enabled' : 'disabled'}.`,
      'info',
    );
  };

  const handleDebug = async (args: string[], ctx: TCtx) => {
    if (args.length > 1) {
      ctx.ui.notify('Usage: /router debug <on|off|show|clear>', 'error');
      return;
    }
    const cmd = args[0]?.toLowerCase();
    if (cmd === 'on') state.debugEnabled = true;
    else if (cmd === 'off') state.debugEnabled = false;
    else if (cmd === 'clear') state.debugHistory.length = 0;
    else if (cmd === 'show') {
      if (state.debugHistory.length === 0) {
        ctx.ui.notify('No recent routing decisions.', 'info');
      } else {
        const history = state.debugHistory
          .map(
            (d) =>
              `[${new Date(d.timestamp).toLocaleTimeString()}] ${formatDecision(d)}`,
          )
          .join('\n');
        ctx.ui.notify(`Recent Routing Decisions:\n${history}`, 'info');
      }
      return;
    } else {
      state.debugEnabled = !state.debugEnabled;
    }
    actions.persistState();
    ctx.ui.notify(
      `Router debug ${state.debugEnabled ? 'enabled' : 'disabled'}.`,
      'info',
    );
  };

  const handleCooldowns = async (args: string[], ctx: TCtx) => {
    if (args.length === 0) {
      const lines = state.cooldownStore?.formatStatus() ?? [];
      ctx.ui.notify(
        lines.length > 0
          ? `Cooldowns:\n${lines.join('\n')}`
          : 'No active cooldowns.',
        'info',
      );
      return;
    }
    if (args[0] === 'clear' && args.length === 2 && state.cooldownStore) {
      state.cooldownStore.clear(args[1]);
      ctx.ui.notify(`Cooldown cleared: ${args[1]}`, 'info');
      actions.updateStatus(ctx);
      return;
    }
    ctx.ui.notify(
      'Usage: /router cooldowns [clear <provider/model|all>]',
      'error',
    );
  };

  const handleReload = async (args: string[], ctx: TCtx) => {
    if (args.length > 0) {
      ctx.ui.notify('Usage: /router reload (no arguments)', 'error');
      return;
    }
    actions.reloadConfig(ctx, { preserveDebug: true });
    await actions.ensureValidActiveRouterProfile(ctx);
    ctx.ui.notify(
      `Router config reloaded. Profiles: ${profileNames(state.currentConfig).join(', ')}`,
      'info',
    );
  };

  // `undefined` means "unknown" (no model to check, or the registry pick
  // doesn't implement hasConfiguredAuth in this context): callers render
  // that as no label at all rather than guessing.
  const accountAuthLabel = (ctx: TCtx, aliasId: string): string => {
    try {
      const provider = ctx.modelRegistry.getProvider?.(aliasId);
      const model = provider?.getModels()[0];
      if (!model || !ctx.modelRegistry.hasConfiguredAuth) return '';
      return ctx.modelRegistry.hasConfiguredAuth(model)
        ? ''
        : ' (not logged in)';
    } catch {
      return '';
    }
  };

  const handleAccount = async (args: string[], ctx: TCtx) => {
    const action = args[0];

    if (action === 'add') {
      const base = args[1];
      if (!base || args.length > 2) {
        ctx.ui.notify('Usage: /router account add <provider>', 'error');
        return;
      }
      if (ACCOUNT_BASE_DENYLIST.has(base)) {
        ctx.ui.notify(
          `Extra accounts for "${base}" are not supported: Pi Black's Claude-subscription request transform only patches the built-in "anthropic" provider, and its implementation is private and unpublished, so this router cannot safely reuse it for "${base}-N" without depending on another extension's internals. See docs/ARCHITECTURE.md "Accounts".`,
          'error',
        );
        return;
      }
      if (!pi.registerProvider) {
        ctx.ui.notify(
          'This Pi build does not support registering providers from an extension.',
          'error',
        );
        return;
      }
      const baseProvider = ctx.modelRegistry.getProvider?.(base);
      if (!baseProvider) {
        ctx.ui.notify(
          `Unknown provider "${base}". It must already be registered (built-in, via models.json, or by another extension) before you can add an extra account for it.`,
          'error',
        );
        return;
      }
      if (!ACCOUNT_BASE_ALLOWLIST.has(base)) {
        ctx.ui.notify(
          `Extra accounts for "${base}" are not supported: this router only clones providers with a static model catalog and no credential-dependent refresh (currently: ${[...ACCOUNT_BASE_ALLOWLIST].join(', ')}). See docs/ARCHITECTURE.md "Accounts".`,
          'error',
        );
        return;
      }

      // Claims the next free "<base>-<n>" slot by atomically O_EXCL-creating
      // that id's own file -- no shared registry to read-modify-write, so a
      // concurrent /router account add for the same base (another session,
      // another process) always lands on a DIFFERENT number instead of racing
      // onto the same one. See extensions/accounts.ts's module doc comment.
      const claim = claimNextAccountSlot(base);
      if (!claim.ok || !claim.aliasId || claim.accountNumber === undefined) {
        ctx.ui.notify(
          `Failed to register account for "${base}": ${claim.error}`,
          'error',
        );
        return;
      }
      const { aliasId, accountNumber } = claim;

      const registration = registerAccountProvider(
        cloneProviderAsAccount(baseProvider, accountNumber),
        ctx.modelRegistry,
        pi,
        state.ownedAccountProviderIds,
      );
      if (!registration.ok) {
        // The slot file is already claimed and persisted (ok by design --
        // claim-then-register, not register-then-persist, so there is never a
        // window where a provider is registered but nothing on disk claims
        // its slot). Tombstone it rather than deleting it: the number must
        // never be reused even though registration never actually succeeded.
        markAccountRemoved(aliasId);
        ctx.ui.notify(
          `Failed to register account "${aliasId}": ${registration.error}`,
          'error',
        );
        return;
      }

      ctx.ui.notify(
        `Registered account "${aliasId}" (clone of "${base}"). Run /login ${aliasId} to add its credential, then reference "${aliasId}/<model>" in a profile's model or fallbacks.`,
        'info',
      );
      return;
    }

    if (action === 'list') {
      if (args.length > 1) {
        ctx.ui.notify('Usage: /router account list', 'error');
        return;
      }
      // Fresh scan of the accounts directory on every call -- never a cached
      // registry -- so this always reflects every process's adds/removes.
      const registry = loadAccountRegistry();
      const bases = Object.keys(registry.accounts).sort();
      if (bases.length === 0) {
        ctx.ui.notify('No extra accounts registered.', 'info');
        return;
      }
      const lines = bases.map((base) => {
        const aliases = listAliasesForBase(registry, base).map(
          (alias) => `${alias}${accountAuthLabel(ctx, alias)}`,
        );
        return `${base}: ${aliases.join(', ')}`;
      });
      ctx.ui.notify(`Accounts:\n${lines.join('\n')}`, 'info');
      return;
    }

    if (action === 'remove') {
      const alias = args[1];
      if (!alias || args.length > 2) {
        ctx.ui.notify('Usage: /router account remove <base-n>', 'error');
        return;
      }
      const parsed = parseAliasId(alias);
      // Fresh read of just this alias's own file -- never a cached registry --
      // so "does this account exist and is it active" always reflects every
      // process's adds/removes, not just this session's view at startup.
      const entry = parsed ? readAccountEntry(alias) : undefined;
      if (!parsed || !entry || entry.removed) {
        ctx.ui.notify(`No such account: ${alias}`, 'error');
        return;
      }

      unregisterAccountProvider(alias, pi, state.ownedAccountProviderIds);

      if (markAccountRemoved(alias)) {
        ctx.ui.notify(`Removed account "${alias}".`, 'info');
        return;
      }

      // Roll back: re-register the clone so the runtime registration matches
      // what the (unchanged) on-disk file still says -- active -- rather than
      // leaving the alias unregistered but still "active" on disk.
      const baseProvider = ctx.modelRegistry.getProvider?.(parsed.base);
      if (baseProvider) {
        registerAccountProvider(
          cloneProviderAsAccount(baseProvider, parsed.accountNumber),
          ctx.modelRegistry,
          pi,
          state.ownedAccountProviderIds,
        );
      }
      ctx.ui.notify(
        `Failed to persist removal of "${alias}"; rolled back.`,
        'error',
      );
      return;
    }

    ctx.ui.notify(
      'Usage: /router account <add <provider>|list|remove <alias>>',
      'error',
    );
  };

  pi.registerCommand('router', {
    description: 'Model router control center',
    getArgumentCompletions: (prefix) => {
      const trimmedLeft = prefix.trimStart();
      const hasTrailingSpace = /\s$/.test(prefix);
      const parts = trimmedLeft.length > 0 ? trimmedLeft.split(/\s+/) : [];

      if (parts.length === 0) {
        return getSubcommandCompletions('');
      }

      if (parts.length === 1 && !hasTrailingSpace) {
        return getSubcommandCompletions(parts[0]);
      }

      const subcommand = parts[0];
      const subArgs = parts.slice(1);
      if (hasTrailingSpace && parts.length === 1) {
        subArgs.push('');
      }

      switch (subcommand) {
        case 'profile': {
          const profilePrefix = subArgs[0] ?? '';
          const items = profileNames(state.currentConfig)
            .filter((name) => name.startsWith(profilePrefix))
            .map((name) => ({
              value: `profile ${name}`,
              label: `router/${name}`,
              description: `Switch to router profile "${name}"`,
            }));
          return items.length > 0 ? items : null;
        }
        case 'pin': {
          const completions = getPinCompletions(subArgs);
          return (
            completions?.map((c) => ({
              ...c,
              value: `pin ${c.value}`,
              description: c.description ?? `Pin routing to ${c.label}`,
            })) ?? null
          );
        }
        case 'thinking': {
          const completions = getThinkingCompletions(subArgs);
          return (
            completions?.map((c) => ({
              ...c,
              value: `thinking ${c.value}`,
              description: c.description ?? `Set thinking level to ${c.label}`,
            })) ?? null
          );
        }
        case 'fix': {
          const fixPrefix = subArgs[0] ?? '';
          const items = ['high', 'medium', 'low']
            .filter((t) => t.startsWith(fixPrefix.toLowerCase()))
            .map((t) => ({
              value: `fix ${t}`,
              label: t,
              description: `Correct decision and pin to ${t} tier`,
            }));
          return items.length > 0 ? items : null;
        }
        case 'widget': {
          const widgetPrefix = subArgs[0] ?? '';
          const items = ['on', 'off', 'toggle']
            .filter((v) => v.startsWith(widgetPrefix))
            .map((v) => ({
              value: `widget ${v}`,
              label: v,
              description: `Set widget to ${v}`,
            }));
          return items.length > 0 ? items : null;
        }
        case 'debug': {
          const debugPrefix = subArgs[0] ?? '';
          const items = ['on', 'off', 'toggle', 'clear', 'show']
            .filter((v) => v.startsWith(debugPrefix))
            .map((v) => ({
              value: `debug ${v}`,
              label: v,
              description: `Router debug: ${v}`,
            }));
          return items.length > 0 ? items : null;
        }
        case 'account': {
          const actionPrefix = subArgs[0] ?? '';
          if (subArgs.length <= 1) {
            const items = ['add', 'list', 'remove']
              .filter((v) => v.startsWith(actionPrefix))
              .map((v) => ({
                value: `account ${v}`,
                label: v,
                description:
                  v === 'add'
                    ? 'Register an extra account for a provider'
                    : v === 'list'
                      ? 'List registered extra accounts'
                      : 'Remove an extra account',
              }));
            return items.length > 0 ? items : null;
          }
          if (subArgs[0] === 'remove') {
            const aliasPrefix = subArgs[1] ?? '';
            // Fresh scan, same reasoning as handleAccount's 'list'/'remove'.
            const registry = loadAccountRegistry();
            const aliases = Object.keys(registry.accounts).flatMap((base) =>
              listAliasesForBase(registry, base),
            );
            const items = aliases
              .filter((alias) => alias.startsWith(aliasPrefix))
              .map((alias) => ({
                value: `account remove ${alias}`,
                label: alias,
                description: `Remove extra account "${alias}"`,
              }));
            return items.length > 0 ? items : null;
          }
          return null;
        }
      }

      return null;
    },
    handler: async (args, ctx) => {
      const parts = args?.trim().split(/\s+/) ?? [];
      const subcommand = parts[0];
      const subArgs = parts.slice(1);

      switch (subcommand) {
        case 'profile':
          await handleProfile(subArgs, ctx);
          break;
        case 'pin':
          await handlePin(subArgs, ctx);
          break;
        case 'thinking':
          await handleThinking(subArgs, ctx);
          break;
        case 'disable':
          await handleDisable(subArgs, ctx);
          break;
        case 'fix':
          await handleFix(subArgs, ctx);
          break;
        case 'widget':
          await handleWidget(subArgs, ctx);
          break;
        case 'debug':
          await handleDebug(subArgs, ctx);
          break;
        case 'cooldowns':
        case 'cooldown':
          await handleCooldowns(subArgs, ctx);
          break;
        case 'account':
          await handleAccount(subArgs, ctx);
          break;
        case 'reload':
          await handleReload(subArgs, ctx);
          break;
        case 'status':
          await handleStatus(subArgs, ctx);
          break;
        case 'help':
        case '?':
          if (subArgs.length > 0) {
            ctx.ui.notify('Usage: /router help (no arguments)', 'error');
            return;
          }
          ctx.ui.notify(
            [
              'Router Subcommands:',
              '  status                      Show current status, profile, pin, cost, and last decision.',
              '  profile [name]              Switch to a profile (enables router if off). Lists available if no name.',
              '  pin <tier|auto>             Force a tier (high|medium|low) or set to auto.',
              '  thinking [tier] <level>     Override thinking level (off|minimal|...|max|auto). Not all tier models may support every level.',
              '  disable                     Disable the router and restore the last used non-router model.',
              '  fix <tier>                  Correct the last routing decision and pin that tier for the current profile.',
              '  widget <on|off|toggle>      Control the persistent status widget visibility.',
              '  debug <on|off|show|clear>   Control routing debug logging to notifications and history.',
              '  account add <provider>      Register an extra account (<provider>-N) cloned from an already-registered provider; run /login <provider>-N next.',
              '  account list                List registered extra accounts.',
              '  account remove <alias>      Remove an extra account.',
              '  reload                      Hot-reload the configuration JSON from .pi/model-router.json.',
              '  help, ?                     Show this help message.',
            ].join('\n'),
            'info',
          );
          break;
        default:
          if (subcommand) {
            // Check if subcommand is actually a profile name (backwards compatible-ish with /router-on)
            if (state.currentConfig.profiles[subcommand]) {
              if (subArgs.length > 0) {
                ctx.ui.notify(
                  `Usage: /router ${subcommand} (no extra arguments allowed)`,
                  'error',
                );
                return;
              }
              await actions.switchToRouterProfile(subcommand, ctx);
              ctx.ui.notify(
                `Router enabled with profile: ${state.selectedProfile}`,
                'info',
              );
            } else {
              ctx.ui.notify(
                `Unknown router subcommand: ${subcommand}. Try /router help`,
                'error',
              );
            }
          } else {
            await handleStatus(subArgs, ctx);
          }
          break;
      }
    },
  });
};
