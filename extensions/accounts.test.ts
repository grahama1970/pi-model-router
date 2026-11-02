import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { spawn } from 'node:child_process';
import {
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { ApiKeyAuth, Provider } from '@earendil-works/pi-ai';
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
  registerPersistedAccounts,
  unregisterAccountProvider,
  aliasProviderId,
  type AccountRegistry,
} from './accounts';

const makeModel = (provider: string) => ({
  id: 'glm-5.3',
  name: 'GLM 5.3',
  api: 'openai-completions',
  provider,
  baseUrl: `https://api.${provider}.example/v1`,
  input: ['text'],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 128000,
  maxTokens: 8192,
  reasoning: false,
});

const makeProvider = (id: string, overrides?: Partial<Provider>): Provider => ({
  id,
  name: `${id} display name`,
  baseUrl: `https://api.${id}.example/v1`,
  getModels: () =>
    [makeModel(id)] as unknown as ReturnType<Provider['getModels']>,
  stream: () => {
    throw new Error('not used in these tests');
  },
  streamSimple: () => {
    throw new Error('not used in these tests');
  },
  auth: {
    apiKey: { name: 'API key', login: async () => ({ type: 'api_key' }) },
  } as unknown as Provider['auth'],
  ...overrides,
});

describe('accounts.ts', () => {
  let dir: string;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'pmr-accounts-test-'));
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  describe('aliasProviderId / parseAliasId', () => {
    it('builds "<base>-<n>" and parses it back', () => {
      expect(aliasProviderId('zai', 2)).toBe('zai-2');
      expect(parseAliasId('zai-2')).toEqual({ base: 'zai', accountNumber: 2 });
    });

    it('round-trips a base id that itself contains a hyphen', () => {
      expect(aliasProviderId('openai-codex', 3)).toBe('openai-codex-3');
      expect(parseAliasId('openai-codex-3')).toEqual({
        base: 'openai-codex',
        accountNumber: 3,
      });
    });

    it('rejects a bare base id (no numeric suffix)', () => {
      expect(parseAliasId('zai')).toBeUndefined();
    });

    it('rejects a suffix below 2', () => {
      expect(parseAliasId('zai-1')).toBeUndefined();
      expect(parseAliasId('zai-0')).toBeUndefined();
    });

    it('rejects a non-numeric suffix', () => {
      expect(parseAliasId('zai-flash')).toBeUndefined();
    });
  });

  describe('claimNextAccountSlot / markAccountRemoved / readAccountEntry (per-file persistence, no shared registry)', () => {
    it('claims account number 2 for the first extra account of a base', () => {
      const result = claimNextAccountSlot('zai', dir);
      expect(result).toEqual({ ok: true, accountNumber: 2, aliasId: 'zai-2' });
    });

    it('claims increasing numbers for repeated adds to the same base', () => {
      claimNextAccountSlot('zai', dir);
      claimNextAccountSlot('zai', dir);
      const third = claimNextAccountSlot('zai', dir);
      expect(third).toEqual({ ok: true, accountNumber: 4, aliasId: 'zai-4' });
      expect(loadAccountRegistry(dir).accounts.zai).toEqual([2, 3, 4]);
    });

    it('keeps separate bases independent', () => {
      claimNextAccountSlot('zai', dir);
      claimNextAccountSlot('openai-codex', dir);
      const registry = loadAccountRegistry(dir);
      expect(listAliasesForBase(registry, 'zai')).toEqual(['zai-2']);
      expect(listAliasesForBase(registry, 'openai-codex')).toEqual([
        'openai-codex-2',
      ]);
    });

    it('removing the highest account does NOT free its number -- the next claim is never reused', () => {
      // Regression for a review finding: an earlier version computed the
      // next number as max(active accounts) + 1, so removing the highest
      // active account made its number immediately reusable. A removed
      // suffix must never be reused, or a stale reference to it (a fallback
      // chain entry, a stored credential under the old id, a log line) could
      // later resolve to a completely different account.
      const a = claimNextAccountSlot('zai', dir); // zai-2
      const b = claimNextAccountSlot('zai', dir); // zai-3
      expect(markAccountRemoved(b.aliasId!, dir)).toBe(true);
      expect(loadAccountRegistry(dir).accounts.zai).toEqual([a.accountNumber]);
      const next = claimNextAccountSlot('zai', dir);
      expect(next.accountNumber).toBe(4); // not 3
    });

    it('a claimed-then-removed number survives removing every account for a base', () => {
      const first = claimNextAccountSlot('zai', dir); // zai-2
      expect(markAccountRemoved(first.aliasId!, dir)).toBe(true);
      expect(loadAccountRegistry(dir).accounts.zai).toBeUndefined();
      const next = claimNextAccountSlot('zai', dir);
      expect(next.accountNumber).toBe(3);
    });

    it('removing a non-highest account does not free its number for reuse', () => {
      const a = claimNextAccountSlot('zai', dir); // zai-2
      const b = claimNextAccountSlot('zai', dir); // zai-3
      expect(markAccountRemoved(a.aliasId!, dir)).toBe(true);
      expect(loadAccountRegistry(dir).accounts.zai).toEqual([b.accountNumber]);
      const next = claimNextAccountSlot('zai', dir);
      expect(next.accountNumber).toBe(4); // not 2
    });

    it('markAccountRemoved never deletes the file -- the removed id stays present as a tombstone', () => {
      const claim = claimNextAccountSlot('zai', dir);
      expect(markAccountRemoved(claim.aliasId!, dir)).toBe(true);
      expect(readdirSync(dir)).toEqual([`${claim.aliasId}.json`]);
      const entry = readAccountEntry(claim.aliasId!, dir);
      expect(entry).toMatchObject({
        base: 'zai',
        accountNumber: 2,
        removed: true,
      });
      expect(entry?.removedAt).toBeTruthy();
    });

    it('readAccountEntry returns undefined for an alias with no file yet', () => {
      expect(readAccountEntry('zai-2', dir)).toBeUndefined();
    });

    it('markAccountRemoved returns false for a malformed alias id', () => {
      expect(markAccountRemoved('zai', dir)).toBe(false); // no numeric suffix
      expect(readdirSync(dir)).toEqual([]);
    });

    it('fails after MAX_ACCOUNT_SLOT_ATTEMPTS (99) exhausted slots, never looping forever', () => {
      for (let n = 2; n <= 100; n++) {
        writeFileSync(
          join(dir, `zai-${n}.json`),
          JSON.stringify({
            base: 'zai',
            accountNumber: n,
            removed: true,
            createdAt: new Date(0).toISOString(),
          }),
        );
      }
      const result = claimNextAccountSlot('zai', dir);
      expect(result.ok).toBe(false);
      expect(result.error).toMatch(/99 attempts/);
    });

    it('writes only base provider ids, small integers, and timestamps -- never a credential-shaped value', () => {
      const claim = claimNextAccountSlot('zai', dir);
      const onDisk = readFileSync(join(dir, `${claim.aliasId}.json`), 'utf-8');
      expect(onDisk).not.toMatch(/key|token|secret|credential/i);
      expect(JSON.parse(onDisk)).toMatchObject({
        base: 'zai',
        accountNumber: 2,
        removed: false,
      });
    });

    it("P1 regression: A add -> A remove -> B add (two independent callers sharing one directory) gets B-3, never A's removed B-2", () => {
      // "A" and "B" here stand for two separate Pi processes/sessions that
      // only share the on-disk accounts directory -- there is no shared
      // in-memory registry object between them, which is exactly the shape
      // of the real cross-process scenario this replaces (a single
      // model-router-accounts.json both processes read-modify-wrote).
      const a = claimNextAccountSlot('zai', dir);
      expect(a).toMatchObject({ ok: true, accountNumber: 2 });
      expect(markAccountRemoved(a.aliasId!, dir)).toBe(true);
      const b = claimNextAccountSlot('zai', dir);
      expect(b).toMatchObject({ ok: true, accountNumber: 3 });
    });

    it("P1 regression: B's remove of its own account never erases A's account for the same base", () => {
      const a = claimNextAccountSlot('zai', dir); // zai-2, "owned" by A
      const b = claimNextAccountSlot('zai', dir); // zai-3, "owned" by B
      expect(markAccountRemoved(b.aliasId!, dir)).toBe(true);
      const registry = loadAccountRegistry(dir);
      expect(registry.accounts.zai).toEqual([a.accountNumber]);
    });

    it("P1 regression: B's add/remove for a different base never erases A's accounts", () => {
      const a = claimNextAccountSlot('zai', dir); // A's account
      const bAdd = claimNextAccountSlot('deepseek', dir); // B's account
      expect(markAccountRemoved(bAdd.aliasId!, dir)).toBe(true); // B removes its own
      const registry = loadAccountRegistry(dir);
      expect(registry.accounts.zai).toEqual([a.accountNumber]);
      expect(registry.accounts.deepseek).toBeUndefined();
    });

    it('P1 regression: two real child processes calling claimNextAccountSlot concurrently for the same base get distinct ids', async () => {
      // Simulated in-process interleaving (the tests above) cannot prove
      // cross-process atomicity -- two separate OS processes are the actual
      // failure mode a shared-registry read-modify-write races under. This
      // spawns two real `node` processes that both import accounts.ts
      // directly (Node's native TypeScript stripping, same technique as
      // cooldown.test.ts's FIFO regressions) and race claimNextAccountSlot
      // against the same directory.
      const accountsModulePath = new URL('./accounts.ts', import.meta.url)
        .pathname;
      const scriptPath = join(dir, 'claim.mts');
      writeFileSync(
        scriptPath,
        [
          `import { claimNextAccountSlot } from ${JSON.stringify(accountsModulePath)};`,
          `const result = claimNextAccountSlot('zai', ${JSON.stringify(dir)});`,
          `process.stdout.write(JSON.stringify(result));`,
          `process.exit(0);`,
        ].join('\n'),
      );

      const runChild = (): Promise<{
        ok: boolean;
        accountNumber?: number;
        aliasId?: string;
      }> =>
        new Promise((resolvePromise, reject) => {
          const child = spawn(
            process.execPath,
            ['--experimental-strip-types', scriptPath],
            { timeout: 5000 },
          );
          let out = '';
          child.stdout.on('data', (chunk) => {
            out += chunk;
          });
          child.on('error', reject);
          child.on('exit', (code) => {
            if (code !== 0) {
              reject(new Error(`child exited with code ${code}: ${out}`));
              return;
            }
            try {
              resolvePromise(JSON.parse(out));
            } catch (error) {
              reject(error);
            }
          });
        });

      const [resultA, resultB] = await Promise.all([runChild(), runChild()]);
      expect(resultA.ok).toBe(true);
      expect(resultB.ok).toBe(true);
      expect(resultA.accountNumber).not.toBe(resultB.accountNumber);
      expect(
        [resultA.accountNumber, resultB.accountNumber].sort((x, y) => x! - y!),
      ).toEqual([2, 3]);
      // Both claims actually landed on disk, as distinct active accounts.
      const registry = loadAccountRegistry(dir);
      expect(registry.accounts.zai).toEqual([2, 3]);
    }, 10000);
  });

  describe('loadAccountRegistry', () => {
    it('returns an empty registry when the directory does not exist', () => {
      expect(loadAccountRegistry(dir)).toEqual({ accounts: {} });
    });

    it('ignores a corrupt account file and degrades to skipping it', () => {
      writeFileSync(join(dir, 'zai-2.json'), '{not valid json');
      expect(loadAccountRegistry(dir)).toEqual({ accounts: {} });
    });

    it('ignores a well-formed JSON file with the wrong shape', () => {
      writeFileSync(join(dir, 'zai-2.json'), JSON.stringify({ foo: 'bar' }));
      expect(loadAccountRegistry(dir)).toEqual({ accounts: {} });
    });

    it("ignores a file whose own declared base/accountNumber doesn't match its filename", () => {
      writeFileSync(
        join(dir, 'zai-2.json'),
        JSON.stringify({
          base: 'deepseek', // mismatched -- filename says zai-2
          accountNumber: 2,
          removed: false,
          createdAt: new Date().toISOString(),
        }),
      );
      expect(loadAccountRegistry(dir)).toEqual({ accounts: {} });
    });

    it('ignores a non-account file in the directory (e.g. a stray .json file)', () => {
      writeFileSync(join(dir, 'not-an-alias.json'), '{}');
      expect(loadAccountRegistry(dir)).toEqual({ accounts: {} });
    });

    it('excludes removed (tombstoned) accounts from the active view', () => {
      const claim = claimNextAccountSlot('zai', dir);
      markAccountRemoved(claim.aliasId!, dir);
      expect(loadAccountRegistry(dir)).toEqual({ accounts: {} });
    });
  });

  describe('cloneProviderAsAccount', () => {
    it("clones id and name, carries baseUrl/stream/streamSimple through unchanged, and rewrites each model's own provider field to the alias id", () => {
      // Regression for a live defect found 2026-11-01 against a local fake
      // 429 provider: Model objects carry their own `provider` field,
      // separate from `Provider.id` (see `BaseModel` in
      // `@earendil-works/pi-ai`'s `dist/types.d.ts`). Pi's dispatcher resolves
      // which Provider to stream through, which credential to resolve, and
      // which ref to report on failure from a model's *own* `provider` field.
      // Inheriting `getModels` unchanged left every model's `provider` field
      // pointing at the base id, so requests "routed" to the clone actually
      // dispatched through the base provider, silently never using the
      // clone's own credential slot, and cooldowns recorded under the base's
      // ref instead of the clone's -- defeating independent cooldowns too.
      const base = makeProvider('zai');
      const clone = cloneProviderAsAccount(base, 2);
      expect(clone.id).toBe('zai-2');
      expect(clone.name).toBe('zai display name (account 2)');
      expect(clone.baseUrl).toBe(base.baseUrl);
      expect(clone.stream).toBe(base.stream);
      expect(clone.streamSimple).toBe(base.streamSimple);

      // getModels() must be a new function -- it cannot be the inherited
      // reference, or the provider rewrite below would be impossible.
      expect(clone.getModels).not.toBe(base.getModels);
      const baseModels = base.getModels();
      const cloneModels = clone.getModels();
      expect(cloneModels).toHaveLength(baseModels.length);
      cloneModels.forEach((model, i) => {
        expect(model.provider).toBe('zai-2');
        expect({ ...model, provider: undefined }).toEqual({
          ...baseModels[i],
          provider: undefined,
        });
      });
    });

    it('supports a base id that already contains a hyphen', () => {
      const base = makeProvider('openai-codex');
      const clone = cloneProviderAsAccount(base, 2);
      expect(clone.id).toBe('openai-codex-2');
    });

    it("never falls back to the base's ambient env var when the alias has no stored credential of its own", async () => {
      // P1 regression: an alias with no `/login <alias>` credential yet must
      // be treated as unauthenticated, never silently resolved through the
      // base's env var. `credential: undefined` is exactly the shape Pi's
      // auth resolver (`resolveProviderAuthWithSignal` in
      // `@earendil-works/pi-ai`'s `dist/auth/resolve.js`) passes when nothing
      // is stored for this provider id.
      let calls = 0;
      const apiKey: ApiKeyAuth = {
        name: 'API key',
        login: async () => ({ type: 'api_key' }),
        resolve: async ({ credential }) => {
          calls++;
          if (credential?.key) {
            return {
              auth: { apiKey: credential.key },
              source: 'stored credential',
            };
          }
          // Simulates envApiKeyAuth's ambient env-var fallback.
          return { auth: { apiKey: 'env-value' }, source: 'ZAI_API_KEY' };
        },
      };
      const base = makeProvider('zai', { auth: { apiKey } });
      const clone = cloneProviderAsAccount(base, 2);

      const ctx = { env: async () => undefined, fileExists: async () => false };
      const withoutCredential = await clone.auth.apiKey!.resolve({
        ctx,
        credential: undefined,
        signal: new AbortController().signal,
      });
      expect(withoutCredential).toBeUndefined();
      expect(calls).toBe(0); // never delegated to the base's resolve

      const withCredential = await clone.auth.apiKey!.resolve({
        ctx,
        credential: { type: 'api_key', key: 'alias-own-key' },
        signal: new AbortController().signal,
      });
      expect(withCredential?.auth.apiKey).toBe('alias-own-key');
      expect(calls).toBe(1); // delegated once a credential IS stored

      // The base provider's own auth is untouched: resolving with no
      // credential still falls back to the ambient env var for the base.
      const baseResolve = await base.auth.apiKey!.resolve({
        ctx,
        credential: undefined,
        signal: new AbortController().signal,
      });
      expect(baseResolve?.auth.apiKey).toBe('env-value');
    });

    it("never falls back to a models.json-configured key either (composeApiKeyAuth's check path)", async () => {
      // Live-verified scenario: a provider declared in models.json with a
      // baked-in `apiKey` (e.g. the fake429 fixture) gets a `check` method
      // from pi-coding-agent's `composeApiKeyAuth` that independently falls
      // back to the configured key when `credential` is absent -- `resolve`
      // alone is not the only ambient-fallback path to block.
      let checkCalls = 0;
      const apiKey: ApiKeyAuth = {
        name: 'API key',
        login: async () => ({ type: 'api_key' }),
        resolve: async ({ credential }) =>
          credential?.key
            ? { auth: { apiKey: credential.key }, source: 'stored credential' }
            : {
                auth: { apiKey: 'configured-key' },
                source: 'configured API key',
              },
        check: async () => {
          checkCalls++;
          return { type: 'api_key', source: 'configured API key' };
        },
      };
      const base = makeProvider('zai', { auth: { apiKey } });
      const clone = cloneProviderAsAccount(base, 2);
      const ctx = { env: async () => undefined, fileExists: async () => false };
      const result = await clone.auth.apiKey!.check!({
        ctx,
        credential: undefined,
        signal: new AbortController().signal,
      });
      expect(result).toBeUndefined();
      expect(checkCalls).toBe(0);
    });

    it('treats a present-but-empty/whitespace api_key credential as absent, proven against the real native envApiKeyAuth + composeApiKeyAuth chain (no fakes)', async () => {
      // P1 regression (reviewer finding 07f36115, accounts.ts:535,542). Pi's
      // own `/login` dialog accepts blank input (`login-dialog.js:44-48`), so
      // a stored credential can be `{type:"api_key", key:""}`. The guards in
      // `wrapApiKeyAuthForAlias` previously checked only `!input.credential`,
      // so a *present but empty* credential passed straight through to the
      // base auth. This test proves the fallback it must block by chaining
      // the actual installed `@earendil-works/pi-ai` `envApiKeyAuth`
      // (`dist/auth/helpers.js:18-25`) through the actual installed
      // `@earendil-works/pi-coding-agent` `composeApiKeyAuth`
      // (`dist/core/provider-composer.js`, used for every `models.json`
      // provider) -- both imported by their real `dist/` file, since neither
      // package's public entry point re-exports `composeModelProvider` or
      // `ModelConfig` (internal, not part of the documented extension
      // surface; importing the concrete file is how this suite proves the
      // fix against the shipped fallback behavior instead of a hand-rolled
      // stand-in for it).
      const { composeModelProvider } =
        (await import('../node_modules/@earendil-works/pi-coding-agent/dist/core/provider-composer.js')) as {
          composeModelProvider: (
            providerId: string,
            base: Provider | undefined,
            modelConfig: unknown,
            extension: undefined,
          ) => Provider;
        };
      const { ModelConfig } =
        (await import('../node_modules/@earendil-works/pi-coding-agent/dist/core/model-config.js')) as {
          ModelConfig: { load: (path: undefined) => Promise<unknown> };
        };
      const { envApiKeyAuth } = await import('@earendil-works/pi-ai');

      const envVarName = 'ROUTER_TEST_DUMMY_ENV_VAR';
      const envDummyValue = 'dummy-env-fallback-value';
      const aliasDummyValue = 'dummy-alias-own-value';
      const originalEnvValue = process.env[envVarName];
      process.env[envVarName] = envDummyValue;
      try {
        const modelConfig = await ModelConfig.load(undefined);
        const rawBase = makeProvider('zai', {
          auth: {
            apiKey: envApiKeyAuth('TEST env key', [envVarName]),
          } as unknown as Provider['auth'],
        });
        // The real composed provider: native `envApiKeyAuth` wrapped by native
        // `composeApiKeyAuth`, with no `models.json` entry (`modelConfig` has
        // no providers) and no extension layer -- the thinnest real
        // composition that still exercises both native fallbacks.
        const nativeBase = composeModelProvider(
          'zai',
          rawBase,
          modelConfig,
          undefined,
        );
        const clone = cloneProviderAsAccount(nativeBase, 2);
        const ctx = {
          env: async (name: string) => process.env[name],
          fileExists: async () => false,
        };
        const signal = new AbortController().signal;

        // Sanity check: proves the native chain itself (pre-wrapper) really
        // does fall back to the env var for an empty key -- the exact defect
        // the alias wrapper must block.
        const nativeResolveEmpty = await nativeBase.auth.apiKey!.resolve({
          ctx,
          credential: { type: 'api_key', key: '' },
          signal,
        });
        expect(nativeResolveEmpty?.auth.apiKey).toBe(envDummyValue);
        const nativeCheckEmpty = await nativeBase.auth.apiKey!.check!({
          ctx,
          credential: { type: 'api_key', key: '' },
          signal,
        });
        expect(nativeCheckEmpty?.source).toBe(envVarName);

        for (const blankKey of ['', '   ']) {
          const resolved = await clone.auth.apiKey!.resolve({
            ctx,
            credential: { type: 'api_key', key: blankKey },
            signal,
          });
          expect(resolved).toBeUndefined();

          const checked = await clone.auth.apiKey!.check!({
            ctx,
            credential: { type: 'api_key', key: blankKey },
            signal,
          });
          expect(checked).toBeUndefined();
        }

        const resolvedWithOwnKey = await clone.auth.apiKey!.resolve({
          ctx,
          credential: { type: 'api_key', key: aliasDummyValue },
          signal,
        });
        expect(resolvedWithOwnKey?.auth.apiKey).toBe(aliasDummyValue);
        expect(resolvedWithOwnKey?.auth.apiKey).not.toBe(envDummyValue);
      } finally {
        if (originalEnvValue === undefined) delete process.env[envVarName];
        else process.env[envVarName] = originalEnvValue;
      }
    });

    it('carries oauth auth through unchanged (reference equality)', () => {
      const oauth = {
        name: 'OAuth',
        login: async () => ({
          type: 'oauth' as const,
          refresh: '',
          access: '',
          expires: 0,
        }),
        refresh: async (c: unknown) => c,
        toAuth: async () => ({}),
      };
      const base = makeProvider('openai-codex', {
        auth: { oauth } as unknown as Provider['auth'],
      });
      const clone = cloneProviderAsAccount(base, 2);
      expect(clone.auth.oauth).toBe(oauth);
      expect(clone.auth.apiKey).toBeUndefined();
    });

    it('never copies refreshModels, even when the base defines one (dynamic-catalog guard, defense in depth)', () => {
      const base = makeProvider('zai', {
        refreshModels: async () => {
          throw new Error('should never be reachable through the clone');
        },
      });
      const clone = cloneProviderAsAccount(base, 2);
      expect(clone.refreshModels).toBeUndefined();
      expect(base.refreshModels).toBeDefined();
    });
  });

  describe('ACCOUNT_BASE_DENYLIST', () => {
    it('denies anthropic', () => {
      expect(ACCOUNT_BASE_DENYLIST.has('anthropic')).toBe(true);
    });

    it('does not deny zai or openai-codex', () => {
      expect(ACCOUNT_BASE_DENYLIST.has('zai')).toBe(false);
      expect(ACCOUNT_BASE_DENYLIST.has('openai-codex')).toBe(false);
    });
  });

  describe('registerPersistedAccounts', () => {
    it('re-registers every persisted alias by cloning its currently-resolvable base', () => {
      const registry: AccountRegistry = {
        accounts: { zai: [2, 3], 'openai-codex': [2] },
      };
      const zai = makeProvider('zai');
      const codex = makeProvider('openai-codex');
      const providers = new Map([
        ['zai', zai],
        ['openai-codex', codex],
      ]);
      const registered: string[] = [];
      registerPersistedAccounts(
        registry,
        { getProvider: (id) => providers.get(id) },
        {
          registerProvider: (provider) => registered.push(provider.id),
        },
        new Set<string>(),
      );
      expect(registered.sort()).toEqual(['openai-codex-2', 'zai-2', 'zai-3']);
    });

    it('skips a base that no longer resolves instead of throwing', () => {
      const registry: AccountRegistry = { accounts: { zai: [2] } };
      const registered: string[] = [];
      expect(() =>
        registerPersistedAccounts(
          registry,
          { getProvider: () => undefined },
          { registerProvider: (provider) => registered.push(provider.id) },
          new Set<string>(),
        ),
      ).not.toThrow();
      expect(registered).toEqual([]);
    });

    it('is a no-op when getProvider or registerProvider is missing', () => {
      const registry: AccountRegistry = { accounts: { zai: [2] } };
      expect(() =>
        registerPersistedAccounts(registry, {}, {}, new Set<string>()),
      ).not.toThrow();
    });

    it('never throws out of a per-account registration failure', () => {
      const registry: AccountRegistry = { accounts: { zai: [2, 3] } };
      const zai = makeProvider('zai');
      const registered: string[] = [];
      expect(() =>
        registerPersistedAccounts(
          registry,
          // Only the base id resolves -- neither alias is registered yet, so
          // registerAccountProvider's collision check never refuses them.
          { getProvider: (id) => (id === 'zai' ? zai : undefined) },
          {
            registerProvider: (provider) => {
              if (provider.id === 'zai-2') throw new Error('boom');
              registered.push(provider.id);
            },
          },
          new Set<string>(),
        ),
      ).not.toThrow();
      expect(registered).toEqual(['zai-3']);
    });

    it('skips a base not on ACCOUNT_BASE_ALLOWLIST even if persisted (e.g. a hand-edited registry file)', () => {
      const registry: AccountRegistry = {
        accounts: { 'not-allowlisted': [2] },
      };
      const provider = makeProvider('not-allowlisted');
      const registered: string[] = [];
      registerPersistedAccounts(
        registry,
        { getProvider: () => provider },
        { registerProvider: (p) => registered.push(p.id) },
        new Set<string>(),
      );
      expect(registered).toEqual([]);
    });

    it('skips re-registering an alias id already claimed by something this process does not own', () => {
      const registry: AccountRegistry = { accounts: { zai: [2] } };
      const zai = makeProvider('zai');
      const foreignZai2 = makeProvider('zai-2');
      const registered: string[] = [];
      registerPersistedAccounts(
        registry,
        {
          getProvider: (id) =>
            id === 'zai' ? zai : id === 'zai-2' ? foreignZai2 : undefined,
        },
        { registerProvider: (p) => registered.push(p.id) },
        new Set<string>(), // empty: this process has not claimed zai-2 itself
      );
      expect(registered).toEqual([]);
    });

    it('allows re-registering an alias id this process already owns (e.g. session_tree running again)', () => {
      const registry: AccountRegistry = { accounts: { zai: [2] } };
      const zai = makeProvider('zai');
      const ownedAlready = makeProvider('zai-2');
      const registered: string[] = [];
      registerPersistedAccounts(
        registry,
        {
          getProvider: (id) =>
            id === 'zai' ? zai : id === 'zai-2' ? ownedAlready : undefined,
        },
        { registerProvider: (p) => registered.push(p.id) },
        new Set<string>(['zai-2']),
      );
      expect(registered).toEqual(['zai-2']);
    });
  });

  describe('registerAccountProvider / unregisterAccountProvider (ownership)', () => {
    it('registers when no provider exists at that id yet, and records ownership', () => {
      const owned = new Set<string>();
      const provider = makeProvider('zai-2');
      const registered: string[] = [];
      const result = registerAccountProvider(
        provider,
        { getProvider: () => undefined },
        { registerProvider: (p) => registered.push(p.id) },
        owned,
      );
      expect(result.ok).toBe(true);
      expect(registered).toEqual(['zai-2']);
      expect(owned.has('zai-2')).toBe(true);
    });

    it('refuses to overwrite an id already registered by something this feature does not own', () => {
      const owned = new Set<string>();
      const foreign = makeProvider('zai-2');
      const registered: string[] = [];
      const result = registerAccountProvider(
        makeProvider('zai-2'),
        { getProvider: () => foreign },
        { registerProvider: (p) => registered.push(p.id) },
        owned,
      );
      expect(result.ok).toBe(false);
      expect(result.error).toContain('zai-2');
      expect(registered).toEqual([]);
      expect(owned.has('zai-2')).toBe(false);
    });

    it('allows overwriting an id this process already owns', () => {
      const owned = new Set<string>(['zai-2']);
      const existing = makeProvider('zai-2');
      const registered: string[] = [];
      const result = registerAccountProvider(
        makeProvider('zai-2'),
        { getProvider: () => existing },
        { registerProvider: (p) => registered.push(p.id) },
        owned,
      );
      expect(result.ok).toBe(true);
      expect(registered).toEqual(['zai-2']);
    });

    it('reports an error instead of throwing when pi.registerProvider is missing', () => {
      const result = registerAccountProvider(
        makeProvider('zai-2'),
        { getProvider: () => undefined },
        {},
        new Set<string>(),
      );
      expect(result.ok).toBe(false);
      expect(result.error).toBeTruthy();
    });

    it('unregisterAccountProvider only unregisters an id this process owns', () => {
      const owned = new Set<string>(['zai-2']);
      const unregistered: string[] = [];
      unregisterAccountProvider(
        'zai-3', // not owned
        { unregisterProvider: (id) => unregistered.push(id) },
        owned,
      );
      expect(unregistered).toEqual([]);

      unregisterAccountProvider(
        'zai-2', // owned
        { unregisterProvider: (id) => unregistered.push(id) },
        owned,
      );
      expect(unregistered).toEqual(['zai-2']);
      expect(owned.has('zai-2')).toBe(false);
    });
  });

  describe('ACCOUNT_BASE_ALLOWLIST', () => {
    it('allows the five providers with a static catalog and no credential-dependent refresh', () => {
      expect([...ACCOUNT_BASE_ALLOWLIST].sort()).toEqual(
        ['deepseek', 'kimi-coding', 'moonshotai', 'openai-codex', 'zai'].sort(),
      );
    });

    it('does not allow anthropic or an arbitrary unlisted provider', () => {
      expect(ACCOUNT_BASE_ALLOWLIST.has('anthropic')).toBe(false);
      expect(ACCOUNT_BASE_ALLOWLIST.has('groq')).toBe(false);
    });
  });
});
