import {
  closeSync,
  constants as fsConstants,
  existsSync,
  fstatSync,
  mkdirSync,
  openSync,
  readFileSync,
  readdirSync,
  renameSync,
  unlinkSync,
  writeSync,
} from 'node:fs';
import { randomBytes } from 'node:crypto';
import { homedir } from 'node:os';
import { join, resolve, sep } from 'node:path';
import type {
  ApiKeyAuth,
  ApiKeyCredential,
  Provider,
} from '@earendil-works/pi-ai';

// Extra provider "accounts": a second (third, ...) credential for a
// provider Pi only supports one login for. `/router account add <base>`
// registers `<base>-<n>` as a clone of the base provider's already-
// registered native `Provider` object -- same `api`, `baseUrl`, model
// catalog, and OAuth login/refresh, but with its own `apiKey` auth (see
// `wrapApiKeyAuthForAlias` below) -- under a new provider id. Pi's
// credential store is keyed by `Provider.id`, one credential per provider
// (see `CredentialStore` in `@earendil-works/pi-ai`'s `dist/auth/types.d.ts`),
// so the clone's `/login` writes to its own `auth.json` entry without
// touching the base provider's. This file never reads, constructs, prints,
// or persists a credential value: registering the clone is the entire
// mechanism, and the credential itself always goes through Pi's own
// interactive login flow for the new provider id.
//
// Only providers on `ACCOUNT_BASE_ALLOWLIST` can be added (static model
// catalog, no credential-dependent `refreshModels`; see that constant's
// comment). `cloneProviderAsAccount` rewraps the clone's `apiKey` auth so an
// alias with no stored credential of its own is never silently
// authenticated through the base's ambient env var or configured key (see
// `wrapApiKeyAuthForAlias`).
//
// Idea borrowed from hjanuschka/pi-multi-pass (MIT license;
// extensions/multi-sub.ts ~L250-363 registers one extra provider per
// account via `pi.registerProvider` with hand-written provider templates).
// Adapted here to clone whatever native `Provider` object Pi (or another
// extension) already registered for the base id instead of re-declaring
// provider templates -- the same code path then covers zai, deepseek,
// moonshotai, and kimi-coding (which pi-multi-pass does not support) and
// openai-codex.
//
// PERSISTENCE (P1 fix, cross-process races): there is no single shared
// registry file anymore. Each account gets its own file,
// `<accountsDir>/<base>-<n>.json`, and that file's existence (active OR
// tombstoned) is the sole source of truth for whether `<n>` is taken --
// there is no separate in-memory "registry" that a second process could
// read stale, mutate, and overwrite. `/router account add` claims the next
// free id by trying `<base>-2`, `<base>-3`, ... and creating each
// candidate's file with `O_EXCL` (atomic create-if-absent at the
// filesystem level: exactly one of any number of racing processes can
// `O_EXCL`-create the same path); the first one that succeeds is this
// process's id, with no read-modify-write of anything shared and nothing
// to lock. `/router account remove` never deletes a file -- it rewrites
// (temp-file-then-rename, like `cooldown.ts`'s `persistOwn`) only that
// alias's own file to `{ removed: true, ... }` -- so a removed id's file
// keeps existing as a tombstone forever and can never be reclaimed by a
// later `O_EXCL` attempt from any process, which is what makes account
// numbers never reuse without needing a persisted counter. Like
// `cooldown.ts`, there are no locks, leases, or waits anywhere in this
// file: a lock-based design previously hung Pi (see `docs/ARCHITECTURE.md`
// "Shared cooldowns" for that history) and every filesystem call here is
// bounded and non-blocking (`O_NONBLOCK`, regular-file-only reads).

// `anthropic` is refused here. Pi Black (git:github.com/paoloanzn/pi-black,
// not modified by this change) rewrites only the built-in `anthropic`
// provider's requests to satisfy Claude subscription billing
// (`wrapAnthropicProvider` in its `src/anthropic-provider.ts`, invoked from
// `extensions/pi-black.ts`). That module has no `exports`/`main` entry and
// `"private": true` in pi-black's `package.json` -- it is not a published
// dependency, so importing it from this package would mean reaching into
// another extension's internal, unversioned file layout instead of a
// documented integration boundary (`registerProvider`'s `ProviderConfig`/
// `Provider` surface). A second Anthropic account registered without that
// transform is rejected upstream the first time it streams with
// "Third-party apps now draw from your extra usage", so `/router account
// add anthropic` is rejected here instead, before any request is made.
export const ACCOUNT_BASE_DENYLIST: ReadonlySet<string> = new Set([
  'anthropic',
]);

// Only providers with a static model catalog and no credential-dependent
// `refreshModels` hook are safe to clone: `cloneProviderAsAccount` shares
// every non-overridden field (including any `refreshModels` closure) by
// reference between the base and every alias, so a dynamic provider's
// refresh would read/write catalog state shared across accounts instead of
// per-credential state -- the base and every alias would silently show
// whichever account refreshed last. Verified 2026-11 against the installed
// `@earendil-works/pi-ai` build (`node_modules/@earendil-works/pi-ai/dist/
// providers/{zai,openai-codex,deepseek,moonshotai,kimi-coding}.js`): none of
// these five pass `fetchModels` to `createProvider`, so none define
// `refreshModels` -- all five return a fixed, bundled model list. Any other
// provider is refused by `/router account add` even if it happens to be
// registered, and `cloneProviderAsAccount` drops `refreshModels` outright as
// a second, defense-in-depth guard (see below).
export const ACCOUNT_BASE_ALLOWLIST: ReadonlySet<string> = new Set([
  'zai',
  'openai-codex',
  'deepseek',
  'moonshotai',
  'kimi-coding',
]);

export interface AccountRegistry {
  // base provider id -> active extra-account suffix numbers (each >= 2).
  // The n-th account for `base` is registered as provider id `${base}-${n}`.
  // Always a point-in-time snapshot from `loadAccountRegistry()` scanning
  // `defaultAccountsDir()` -- call it again for a fresh view. There is no
  // "the" in-memory registry to mutate and save anymore; each account file
  // in that directory is independently authoritative (see module doc
  // comment above).
  accounts: Record<string, number[]>;
}

// One JSON file per account, keyed by its own alias id as the filename
// (`<base>-<n>.json`). `removed` is a tombstone, never a deletion: once a
// file exists for `<base>-<n>` -- active or removed -- that id can never be
// claimed again by any process's `claimNextAccountSlot`.
export interface AccountFileRecord {
  base: string;
  accountNumber: number;
  removed: boolean;
  createdAt: string; // ISO UTC
  removedAt?: string; // ISO UTC, present only once removed
}

export const defaultAccountsDir = (): string =>
  process.env.PI_MODEL_ROUTER_ACCOUNTS ??
  join(homedir(), '.local', 'state', 'pi-model-router', 'accounts');

// node:fs mkdirSync(path, {recursive:true}) can hang indefinitely on Node v22
// when path is under /proc (verified in cooldown.ts: non-recursive mkdir on a
// /proc path returns ENOENT instantly; the recursive variant never returns).
// Account persistence must never block the process, so directory creation
// walks the path itself with bounded, non-recursive mkdirSync calls instead
// of trusting the recursive flag. Same pattern as `cooldown.ts`'s
// `mkdirBoundedSync`, duplicated here rather than shared so this module has
// no internal dependency on cooldown.ts's private helpers.
const MAX_MKDIR_COMPONENTS = 32;

const mkdirBoundedSync = (targetDir: string): void => {
  const resolved = resolve(targetDir);
  const segments = resolved.split(sep).filter((segment) => segment.length > 0);
  if (segments.length > MAX_MKDIR_COMPONENTS) {
    throw new Error(
      `accounts dir exceeds ${MAX_MKDIR_COMPONENTS} path components: ${resolved}`,
    );
  }
  let current: string = sep;
  for (const segment of segments) {
    current =
      current === sep ? `${sep}${segment}` : `${current}${sep}${segment}`;
    try {
      mkdirSync(current, { mode: 0o700 });
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'EEXIST') continue;
      throw error;
    }
  }
};

const MAX_ACCOUNT_FILE_BYTES = 4 * 1024;

// Reads a path only if it's a REGULAR file no larger than 4 KiB, via a
// non-blocking, non-following open so a FIFO, socket, device, or symlink
// left (deliberately or accidentally) in the accounts dir can never hang a
// read: O_NONBLOCK makes both the open() and the subsequent read() on that
// fd return immediately instead of waiting for a writer, and the fstatSync
// check on the already-open fd (not the path, closing a TOCTOU window)
// rejects anything that isn't a plain file before its content is read.
// Returns undefined for anything that isn't eligible -- callers treat that
// exactly like "unreadable or malformed", i.e. skip, never throw.
const readRegularFileBounded = (path: string): string | undefined => {
  let fd: number | undefined;
  try {
    fd = openSync(
      path,
      fsConstants.O_RDONLY | fsConstants.O_NONBLOCK | fsConstants.O_NOFOLLOW,
    );
    const stat = fstatSync(fd);
    if (!stat.isFile() || stat.size > MAX_ACCOUNT_FILE_BYTES) return undefined;
    return readFileSync(fd, 'utf8');
  } catch {
    return undefined;
  } finally {
    if (fd !== undefined) {
      try {
        closeSync(fd);
      } catch {
        // Best-effort close only; nothing we can do if this fails.
      }
    }
  }
};

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null;

const parseAccountRecord = (
  content: string | undefined,
): AccountFileRecord | undefined => {
  if (content === undefined) return undefined;
  try {
    const raw: unknown = JSON.parse(content);
    if (!isObject(raw)) return undefined;
    const { base, accountNumber, removed, createdAt, removedAt } = raw;
    if (typeof base !== 'string' || base.length === 0) return undefined;
    if (
      typeof accountNumber !== 'number' ||
      !Number.isInteger(accountNumber) ||
      accountNumber < 2
    )
      return undefined;
    if (typeof removed !== 'boolean') return undefined;
    return {
      base,
      accountNumber,
      removed,
      createdAt:
        typeof createdAt === 'string' ? createdAt : new Date(0).toISOString(),
      removedAt: typeof removedAt === 'string' ? removedAt : undefined,
    };
  } catch {
    return undefined;
  }
};

const ACCOUNT_FILE_SUFFIX = '.json';

/**
 * Reads every `<base>-<n>.json` in `dir`, skipping anything that isn't a
 * regular, bounded-size, well-formed account file -- a `.tmp` file from an
 * in-progress `markAccountRemoved` rename, a stray non-account file, or a
 * file whose own declared `base`/`accountNumber` doesn't match its
 * filename, are all silently skipped rather than throwing. Never throws.
 */
const scanAccountsDir = (dir: string): Map<string, AccountFileRecord> => {
  const result = new Map<string, AccountFileRecord>();
  if (!existsSync(dir)) return result;
  let names: string[];
  try {
    names = readdirSync(dir);
  } catch {
    return result;
  }
  for (const name of names) {
    if (!name.endsWith(ACCOUNT_FILE_SUFFIX)) continue;
    const aliasId = name.slice(0, -ACCOUNT_FILE_SUFFIX.length);
    const parsedAlias = parseAliasId(aliasId);
    if (!parsedAlias) continue;
    const record = parseAccountRecord(readRegularFileBounded(join(dir, name)));
    if (!record) continue;
    if (
      record.base !== parsedAlias.base ||
      record.accountNumber !== parsedAlias.accountNumber
    )
      continue;
    result.set(aliasId, record);
  }
  return result;
};

const createAccountFileExclusive = (
  filePath: string,
  record: AccountFileRecord,
): boolean => {
  let fd: number | undefined;
  try {
    fd = openSync(
      filePath,
      fsConstants.O_WRONLY |
        fsConstants.O_CREAT |
        fsConstants.O_EXCL |
        fsConstants.O_NOFOLLOW |
        fsConstants.O_NONBLOCK,
      0o600,
    );
    writeSync(fd, JSON.stringify(record, null, 2));
    return true;
  } catch {
    // EEXIST (another process already owns this id, active or tombstoned),
    // a FIFO/symlink refused by O_NOFOLLOW, or any other failure: the slot
    // isn't ours. Never thrown -- callers try the next candidate id.
    return false;
  } finally {
    if (fd !== undefined) {
      try {
        closeSync(fd);
      } catch {
        // Best-effort close only.
      }
    }
  }
};

/**
 * Fresh, point-in-time view of every active account, built by scanning
 * `dir` -- there is no cached/in-memory registry to go stale here, so every
 * call (command handler, `registerPersistedAccounts` at session start,
 * autocomplete) sees whatever every process has written so far. A missing,
 * corrupt, or unreadable directory or file degrades to an empty/partial
 * registry rather than throwing.
 */
export const loadAccountRegistry = (
  dir: string = defaultAccountsDir(),
): AccountRegistry => {
  const accounts: Record<string, number[]> = {};
  try {
    for (const record of scanAccountsDir(dir).values()) {
      if (record.removed) continue;
      const list = accounts[record.base] ?? [];
      list.push(record.accountNumber);
      accounts[record.base] = list;
    }
    for (const base of Object.keys(accounts)) {
      accounts[base] = [...new Set(accounts[base])].sort((a, b) => a - b);
    }
  } catch {
    return { accounts: {} };
  }
  return { accounts };
};

export const aliasProviderId = (base: string, accountNumber: number): string =>
  `${base}-${accountNumber}`;

/**
 * Parse `<base>-<n>` back into its base provider id and account number.
 * Returns `undefined` for anything not shaped like an account alias
 * (no trailing `-<integer >= 2>`), including a bare base id.
 */
export const parseAliasId = (
  alias: string,
): { base: string; accountNumber: number } | undefined => {
  const match = /^(.+)-(\d+)$/.exec(alias);
  if (!match) return undefined;
  const accountNumber = Number(match[2]);
  if (!Number.isInteger(accountNumber) || accountNumber < 2) return undefined;
  return { base: match[1], accountNumber };
};

export const listAliasesForBase = (
  registry: AccountRegistry,
  base: string,
): string[] =>
  (registry.accounts[base] ?? []).map((n) => aliasProviderId(base, n));

const MAX_ACCOUNT_SLOT_ATTEMPTS = 99; // tries <base>-2 through <base>-100

export interface ClaimAccountSlotResult {
  ok: boolean;
  accountNumber?: number;
  aliasId?: string;
  error?: string;
}

/**
 * Atomically claims the next free account slot for `base`: tries
 * `<base>-2`, `<base>-3`, ... up to `MAX_ACCOUNT_SLOT_ATTEMPTS` times,
 * `O_EXCL`-creating each candidate's file until one succeeds. `O_EXCL`
 * create-if-absent is atomic at the filesystem level, so when two processes
 * race to add an account for the same base, each lands on a DIFFERENT
 * number -- there is no read-then-write window for them to collide in (the
 * cross-process race this replaces: a shared `model-router-accounts.json`
 * read by both, both computing the same "next" number, both writing a
 * clone registered under the identical alias id for two different
 * accounts). No locks, leases, or waits: a losing attempt just means "try
 * the next number", never "wait and retry this one". The caller is
 * responsible for registering the actual provider clone after a successful
 * claim, and for calling `markAccountRemoved` to tombstone the file (never
 * delete it) if that registration then fails -- see `handleAccount`'s
 * `add` in `extensions/commands.ts`.
 */
export const claimNextAccountSlot = (
  base: string,
  dir: string = defaultAccountsDir(),
): ClaimAccountSlotResult => {
  try {
    mkdirBoundedSync(dir);
  } catch (error) {
    return {
      ok: false,
      error: `Failed to create accounts directory: ${error instanceof Error ? error.message : String(error)}`,
    };
  }
  const createdAt = new Date().toISOString();
  for (let n = 2; n < 2 + MAX_ACCOUNT_SLOT_ATTEMPTS; n++) {
    const aliasId = aliasProviderId(base, n);
    const filePath = join(dir, `${aliasId}.json`);
    if (
      createAccountFileExclusive(filePath, {
        base,
        accountNumber: n,
        removed: false,
        createdAt,
      })
    ) {
      return { ok: true, accountNumber: n, aliasId };
    }
  }
  return {
    ok: false,
    error: `No available account slot for "${base}" after ${MAX_ACCOUNT_SLOT_ATTEMPTS} attempts.`,
  };
};

/** Fresh read of a single alias's own file. Never throws. */
export const readAccountEntry = (
  aliasId: string,
  dir: string = defaultAccountsDir(),
): AccountFileRecord | undefined =>
  parseAccountRecord(readRegularFileBounded(join(dir, `${aliasId}.json`)));

/**
 * Tombstones (never deletes) `aliasId`'s own file: rewrites it, via
 * temp-file-then-rename with an `O_EXCL`-created, unpredictable temp name
 * (same pattern as `cooldown.ts`'s `persistOwn`), to `{ removed: true,
 * ... }`. Only ever touches this one alias's file -- there is no shared
 * registry file for a second process's remove/add to race against, so a
 * `/router account remove` in one session can never erase another
 * session's accounts for the same or a different base. Because the file
 * keeps existing (now tombstoned) rather than being deleted, no later
 * `claimNextAccountSlot` call -- from this or any other process -- can ever
 * `O_EXCL`-create this id again: account numbers are never reused. Returns
 * whether the rewrite succeeded; callers treat a `false` as advisory (the
 * caller should report the failure and may roll back an in-memory/runtime
 * change it already made) rather than throwing.
 */
export const markAccountRemoved = (
  aliasId: string,
  dir: string = defaultAccountsDir(),
): boolean => {
  const parsed = parseAliasId(aliasId);
  if (!parsed) return false;
  const filePath = join(dir, `${aliasId}.json`);
  const existing = readAccountEntry(aliasId, dir);
  const record: AccountFileRecord = {
    base: parsed.base,
    accountNumber: parsed.accountNumber,
    removed: true,
    createdAt: existing?.createdAt ?? new Date().toISOString(),
    removedAt: new Date().toISOString(),
  };
  let temp: string | undefined;
  let createdTemp = false;
  try {
    mkdirBoundedSync(dir);
    temp = `${filePath}.${process.pid}.${randomBytes(4).toString('hex')}.tmp`;
    let fd: number | undefined;
    try {
      fd = openSync(
        temp,
        fsConstants.O_WRONLY |
          fsConstants.O_CREAT |
          fsConstants.O_EXCL |
          fsConstants.O_NOFOLLOW |
          fsConstants.O_NONBLOCK,
        0o600,
      );
      createdTemp = true;
      writeSync(fd, JSON.stringify(record, null, 2));
    } finally {
      if (fd !== undefined) closeSync(fd);
    }
    renameSync(temp, filePath);
    return true;
  } catch {
    if (temp !== undefined && createdTemp) {
      try {
        unlinkSync(temp);
      } catch {
        // Best-effort cleanup only; nothing we can do if this also fails.
      }
    }
    return false;
  }
};

// Every catalog entry a Provider returns carries its own `provider` field
// (see `BaseModel` in `@earendil-works/pi-ai`'s `dist/types.d.ts`), separate
// from the containing `Provider.id`. Pi's dispatcher resolves which
// `Provider` to stream through, which credential to resolve, and which
// `provider/model` ref to report on failure from a model's *own* `provider`
// field -- not from whichever `Provider.id` happened to return the model.
// Live-verified 2026-11 against a local fake 429 provider: cloning only
// `id`/`name` and leaving `getModels()` inherited unchanged left every
// returned model's `provider` field pointing at the base id, so requests
// "routed" to the clone actually dispatched through the base provider's own
// registration -- the clone's separate credential slot was silently never
// used, and failures were reported (and cooled down) under the base's
// `provider/model` ref instead of the clone's, defeating independent
// cooldowns too. `rewriteModelProvider` is what fixes that: every model this
// clone returns gets its own `provider` field rewritten to the alias id.
const rewriteModelProvider = <T extends { provider: string }>(
  aliasId: string,
  model: T,
): T => ({ ...model, provider: aliasId });

/**
 * Wrap a base provider's apiKey auth so the alias never falls back to
 * ambient resolution (an env var, or a models.json-configured key) when it
 * has no stored credential of its own. Pi's credential store is keyed by
 * `Provider.id` (one entry per provider id), so an alias with no
 * `/login <alias>` entry yet has `credential === undefined` on every
 * `resolve`/`check` call -- exactly the same shape as a provider that was
 * never configured at all. Both `@earendil-works/pi-ai`'s `envApiKeyAuth`
 * (`dist/auth/helpers.js`) and pi-coding-agent's `composeApiKeyAuth`
 * (`dist/core/provider-composer.js`, used for every `models.json`-declared
 * provider) treat that as "fall back to the env var / configured key"
 * instead of "unauthenticated" -- live-verified 2026-11 against a local fake
 * 429 provider declared in `models.json` with a baked-in `apiKey`: an alias
 * cloned without this wrapper silently authenticated through the base's
 * configured key and was dispatched as if logged in. Returning `undefined`
 * whenever `credential` is absent is what Pi's own auth resolver
 * (`@earendil-works/pi-ai`'s `dist/auth/resolve.js`,
 * `resolveProviderAuthWithSignal`) treats as "provider not configured": no
 * other supported hook exists to disable only the ambient-fallback half of
 * `ApiKeyAuth.resolve`/`check` while keeping stored-credential resolution
 * (and `login`, which writes to the alias's own store entry) working
 * unchanged -- replacing `resolve`/`check` entirely is the documented
 * `ApiKeyAuth` extension point (`dist/auth/types.d.ts`). The base
 * provider's own `auth` object is left untouched; only the alias's clone
 * gets the wrapped copy.
 */
// A `{type:"api_key"}` credential with a blank/whitespace-only `key` is
// exactly what Pi's own `/login` writes when the user submits empty input
// (`login-dialog.js:44-48` accepts a blank string without rejecting it).
// `input.credential` being merely *present* with such a key was treated as
// "this alias has its own credential" and passed straight through to the
// base provider's `resolve`/`check` -- but native `envApiKeyAuth`
// (`dist/auth/helpers.js:18-25`) treats a falsy/empty `credential.key` the
// same as no credential and falls back to the provider's ambient env var
// (e.g. `ZAI_API_KEY`), so the alias silently authenticated through the base
// account's env credential instead of reporting unauthenticated. Requiring a
// non-empty, trimmed `key` closes that gap; `undefined`/`check` failure here
// is exactly what Pi's auth resolver treats as "not configured" (see the
// `wrapApiKeyAuthForAlias` doc comment above).
const hasNonEmptyApiKey = (credential?: ApiKeyCredential): boolean =>
  typeof credential?.key === 'string' && credential.key.trim().length > 0;

const wrapApiKeyAuthForAlias = (apiKey: ApiKeyAuth): ApiKeyAuth => {
  const wrapped: ApiKeyAuth = {
    ...apiKey,
    resolve: async (input) => {
      if (!hasNonEmptyApiKey(input.credential)) return undefined;
      return apiKey.resolve(input);
    },
  };
  if (apiKey.check) {
    const check = apiKey.check;
    wrapped.check = async (input) => {
      if (!hasNonEmptyApiKey(input.credential)) return undefined;
      return check(input);
    };
  }
  return wrapped;
};

/**
 * Clone a base provider's complete native registration under a new id.
 * `api`, `baseUrl`, `headers`, `stream`/`streamSimple`, and `oauth` auth
 * carry over unchanged. `apiKey` auth is rewrapped (see
 * `wrapApiKeyAuthForAlias`) so the alias never authenticates via the base's
 * ambient env var or configured key. `id`, `name`, and the `provider` field
 * on every model `getModels()`/`getAllModels()` return are rewritten to the
 * new id (see `rewriteModelProvider` above for why that rewrite is
 * required, not cosmetic). `refreshModels` is dropped outright, never
 * copied: this clone is only ever called for a base on
 * `ACCOUNT_BASE_ALLOWLIST` (static catalog, no credential-dependent
 * refresh), and a dynamic provider's `refreshModels` closure would read/
 * write catalog state shared by reference across every clone and the base.
 */
export const cloneProviderAsAccount = (
  base: Provider,
  accountNumber: number,
): Provider => {
  const aliasId = aliasProviderId(base.id, accountNumber);
  // `refreshModels` is never carried over -- see the doc comment above.
  const { refreshModels: _baseRefreshModels, ...baseWithoutRefresh } = base;
  const newApiKeyAuth = base.auth.apiKey
    ? wrapApiKeyAuthForAlias(base.auth.apiKey)
    : undefined;
  const clone: Provider = {
    ...baseWithoutRefresh,
    id: aliasId,
    name: `${base.name} (account ${accountNumber})`,
    auth: newApiKeyAuth ? { ...base.auth, apiKey: newApiKeyAuth } : base.auth,
    getModels: () =>
      base.getModels().map((model) => rewriteModelProvider(aliasId, model)),
  };
  if (base.getAllModels) {
    clone.getAllModels = () =>
      base.getAllModels!().map((model) => rewriteModelProvider(aliasId, model));
  }
  return clone;
};

export interface AccountProviderLookup {
  getProvider?: (id: string) => Provider | undefined;
}

export interface AccountProviderRegistrar {
  registerProvider?: (provider: Provider) => void;
  unregisterProvider?: (id: string) => void;
}

export interface AccountRegistrationResult {
  ok: boolean;
  error?: string;
}

/**
 * Register one account provider, refusing a collision with an id this
 * feature does not own (P1: a provider id already claimed by something
 * else -- another extension, a `models.json` entry, or a genuine id clash
 * -- must never be silently overwritten). `ownedProviderIds` is this
 * process's own record of which provider ids it has itself registered
 * (never persisted -- `pi.registerProvider()` is a runtime-only call, so
 * every process starts with an empty set and earns ownership by
 * registering); an id already present in `modelRegistry` that this process
 * has not itself claimed is refused rather than overwritten. On success,
 * `provider.id` is added to `ownedProviderIds` so a later re-registration
 * of the same id (e.g. `registerPersistedAccounts` running again on
 * `session_tree`) is recognized as this feature's own and allowed to
 * refresh it. Never throws: a `pi.registerProvider` failure is caught and
 * reported in the result.
 */
export const registerAccountProvider = (
  provider: Provider,
  modelRegistry: AccountProviderLookup,
  pi: AccountProviderRegistrar,
  ownedProviderIds: Set<string>,
): AccountRegistrationResult => {
  if (!pi.registerProvider) {
    return {
      ok: false,
      error:
        'This Pi build does not support registering providers from an extension.',
    };
  }
  let existing: Provider | undefined;
  try {
    existing = modelRegistry.getProvider?.(provider.id);
  } catch {
    existing = undefined;
  }
  if (existing && !ownedProviderIds.has(provider.id)) {
    return {
      ok: false,
      error:
        'Provider id "' +
        provider.id +
        '" is already registered by something other than this feature; refusing to overwrite it.',
    };
  }
  try {
    pi.registerProvider(provider);
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
  ownedProviderIds.add(provider.id);
  return { ok: true };
};

/**
 * Unregister an account provider, but only if this process itself owns
 * that id (P1/P2: "remove only unregisters aliases this feature owns" --
 * never unregister a provider id this feature never actually registered,
 * even if the persisted registry names it).
 */
export const unregisterAccountProvider = (
  aliasId: string,
  pi: AccountProviderRegistrar,
  ownedProviderIds: Set<string>,
): void => {
  if (!ownedProviderIds.has(aliasId)) return;
  try {
    pi.unregisterProvider?.(aliasId);
  } finally {
    ownedProviderIds.delete(aliasId);
  }
};

/**
 * Re-registers every persisted account as a clone of its (currently
 * resolvable) base provider. `pi.registerProvider()` is a runtime
 * registration, never persisted by Pi itself, so every process restart must
 * redo it -- the same reason `extensions/provider.ts`'s
 * `registerRouterProvider()` re-registers the router's own virtual models
 * on every `session_start`/`turn_start`. A base provider that no longer
 * resolves (for example, an extension that provided it was removed), a
 * base no longer on `ACCOUNT_BASE_ALLOWLIST` (e.g. after a hand-edited
 * registry file, or a narrower allowlist shipped later), or a collision
 * with a provider id this process does not own (see
 * `registerAccountProvider`) is skipped silently rather than throwing: a
 * stale or inconsistent registry entry must never block session startup.
 * `registry` should come from a fresh `loadAccountRegistry()` call made
 * right before this -- see `extensions/index.ts`'s
 * `restoreStateFromSession`/`ensureInitializedFromContext`.
 */
export const registerPersistedAccounts = (
  registry: AccountRegistry,
  modelRegistry: AccountProviderLookup,
  pi: AccountProviderRegistrar,
  ownedProviderIds: Set<string>,
): void => {
  if (!modelRegistry.getProvider || !pi.registerProvider) return;
  for (const [base, numbers] of Object.entries(registry.accounts)) {
    if (!ACCOUNT_BASE_ALLOWLIST.has(base)) continue;
    let baseProvider: Provider | undefined;
    try {
      baseProvider = modelRegistry.getProvider(base);
    } catch {
      baseProvider = undefined;
    }
    if (!baseProvider) continue;
    for (const accountNumber of numbers) {
      registerAccountProvider(
        cloneProviderAsAccount(baseProvider, accountNumber),
        modelRegistry,
        pi,
        ownedProviderIds,
      );
    }
  }
};
