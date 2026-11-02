import {
  chmodSync,
  closeSync,
  constants as fsConstants,
  existsSync,
  fstatSync,
  mkdirSync,
  openSync,
  readFileSync,
  readdirSync,
  renameSync,
  statSync,
  unlinkSync,
  writeSync,
} from 'node:fs';
import { randomBytes } from 'node:crypto';
import { join, resolve, sep } from 'node:path';
import { homedir } from 'node:os';
import type { RouterConfig } from './types';

// Cooldowns are advisory, best-effort routing hints, never a correctness
// mechanism: a Pi session must never become unresponsive because of a rate
// limit, and this subsystem must never cause that itself. Every code path
// here is wrapped so a read/write/parse failure degrades to "route as if no
// cooldown existed" instead of throwing into route(). There are no locks,
// no leases, and no waits anywhere in this file — nothing here can spin.

export type CooldownSource = 'retry' | 'transcript' | 'factory';

// Category only — never the raw provider error text, which can carry account
// identifiers or other message-specific detail we must not persist to disk.
export type CooldownReasonCategory =
  | 'rate_limit'
  | 'quota'
  | 'usage_limit'
  | 'too_many_requests'
  | 'unknown';

export interface CooldownEntry {
  until: string; // ISO UTC
  reason: CooldownReasonCategory;
  // Only the substring the reset-time regex itself matched (≤40 chars), e.g.
  // "retry in 10 min" — never the surrounding provider error message.
  raw_reset_text: string;
  assumed_tz: string;
  source: CooldownSource;
  recorded_at: string; // ISO UTC
}

export interface CooldownState {
  entries: Record<string, CooldownEntry>;
}

export interface CooldownStore {
  // Merged view across this process's own cooldowns, every other process's
  // on-disk cooldown file, and the read-only agent-factory state.
  load(now?: number): CooldownState;
  // `now` drives expiry math only (relative "retry in N min" parsing and the
  // default cooldown length). `recordedAt`, when given, becomes the entry's
  // `recorded_at` -- the event time a later success is compared against in
  // noteSuccess -- and defaults to `now`. Transcript-sourced failures must
  // pass the failed message's own timestamp here so recorded_at reflects
  // when the failure actually happened, not when the transcript was scanned.
  record(
    modelRef: string,
    errorText: string,
    source: Exclude<CooldownSource, 'factory'>,
    config: RouterConfig,
    now?: number,
    recordedAt?: number,
  ): CooldownEntry;
  // Removes modelRef from THIS process's own file only. When successTimestamp
  // is given, the removal is skipped unless it's strictly newer than that
  // model's own recorded_at, so a success message that is older than (or
  // concurrent with) the model's latest recorded failure can never erase it;
  // omitting the timestamp keeps the old unconditional-clear behavior.
  noteSuccess(modelRef: string, successTimestamp?: number): void;
  // isUsable, when provided, restricts cooldown filtering to resolvable,
  // authenticated candidates: unusable refs always pass through untouched, and
  // "keep the soonest-expiring one when everything is cooling" only ever picks
  // a usable ref, so filtering never leaves zero usable models.
  candidates(
    modelRefs: readonly string[],
    now?: number,
    isUsable?: (modelRef: string) => boolean,
  ): string[];
  // Removes modelRef (or everything, for 'all'/undefined) from this process's
  // own file, and suppresses every other process's (and the factory's)
  // current entry for each cleared model independently, until that specific
  // entry changes -- so an unrelated, unchanged foreign entry for the same
  // model can never resurface just because a different entry for it expired
  // or was cleared.
  clear(modelRef?: string): void;
  formatStatus(now?: number): string[];
}

// node:fs mkdirSync(path, {recursive:true}) can hang indefinitely on Node v22
// when path is under /proc (verified: non-recursive mkdir on a /proc path
// returns ENOENT instantly; the recursive variant never returns). Cooldowns
// are advisory and must never block the process, so directory creation walks
// the path itself with bounded, non-recursive mkdirSync calls instead of
// trusting the recursive flag.
const MAX_MKDIR_COMPONENTS = 32;

const mkdirBoundedSync = (targetDir: string): void => {
  const resolved = resolve(targetDir);
  const segments = resolved.split(sep).filter((segment) => segment.length > 0);
  if (segments.length > MAX_MKDIR_COMPONENTS) {
    throw new Error(
      `cooldown dir exceeds ${MAX_MKDIR_COMPONENTS} path components: ${resolved}`,
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

const DEFAULT_COOLDOWN_MS = 30 * 60 * 1000;
const SEVEN_DAYS_MS = 7 * 24 * 60 * 60 * 1000;
const STALE_FILE_MS = 24 * 60 * 60 * 1000;
const READ_CACHE_MS = 2000;
const RAW_RESET_TEXT_MAX = 40;
const MAX_COOLDOWN_FILE_BYTES = 64 * 1024;
const FACTORY_STATE_PATH =
  '/mnt/storage12tb/agent-factory/state/provider-switch.json';

// Reads a path only if it's a REGULAR file no larger than 64 KiB, via a
// non-blocking, non-following open so a FIFO, socket, device, or symlink
// left (deliberately or accidentally) in the cooldown dir can never hang
// this read: O_NONBLOCK makes both the open() and the subsequent read() on
// that fd return immediately instead of waiting for a writer, and the
// fstatSync check on the already-open fd (not the path, closing a TOCTOU
// window) rejects anything that isn't a plain file before we read its
// content. Returns undefined for anything that isn't eligible — callers
// treat that exactly like "unreadable or malformed", i.e. skip, never throw.
const readRegularFileBounded = (path: string): string | undefined => {
  let fd: number | undefined;
  try {
    fd = openSync(
      path,
      fsConstants.O_RDONLY | fsConstants.O_NONBLOCK | fsConstants.O_NOFOLLOW,
    );
    const stat = fstatSync(fd);
    if (!stat.isFile() || stat.size > MAX_COOLDOWN_FILE_BYTES) return undefined;
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
const DEFAULT_TIMEZONES: Record<string, string> = {
  zai: 'Asia/Shanghai',
  moonshot: 'Asia/Shanghai',
  'kimi-coding': 'Asia/Shanghai',
  deepseek: 'Asia/Shanghai',
  qwen: 'Asia/Shanghai',
  alibaba: 'Asia/Shanghai',
  minimax: 'Asia/Shanghai',
};

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const iso = (time: number): string => new Date(time).toISOString();

const parseTime = (value: unknown): number | undefined => {
  if (typeof value !== 'string') return undefined;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : undefined;
};

// Date can only represent values within +/-8,640,000,000,000,000ms of the
// epoch; anything outside that range -- or non-finite (NaN/Infinity) --
// makes `new Date(value).toISOString()` throw a RangeError. A transcript
// message's `timestamp` field is attacker/provider-controlled input, so
// every timestamp taken from a transcript must pass this check before it is
// ever used to build a cooldown entry.
const MAX_VALID_TIMESTAMP_MS = 8_640_000_000_000_000;
export const isValidEventTimestamp = (value: number): boolean =>
  Number.isFinite(value) &&
  Math.abs(value) <= MAX_VALID_TIMESTAMP_MS &&
  !Number.isNaN(new Date(value).getTime());

const REASON_CATEGORIES: readonly CooldownReasonCategory[] = [
  'rate_limit',
  'quota',
  'usage_limit',
  'too_many_requests',
  'unknown',
];

const isReasonCategory = (value: unknown): value is CooldownReasonCategory =>
  typeof value === 'string' &&
  (REASON_CATEGORIES as readonly string[]).includes(value);

const SOURCES: readonly CooldownSource[] = ['retry', 'transcript', 'factory'];
const isSource = (value: unknown): value is CooldownSource =>
  typeof value === 'string' && (SOURCES as readonly string[]).includes(value);

// Categorize a provider error into a fixed, non-identifying reason bucket.
// Order matters: more specific categories are checked before the generic
// rate_limit catch-all (e.g. a "limit exhausted" message is usage_limit, not
// just a bare rate_limit, even though it also matches the 429-ish pattern).
export const categorizeReason = (text: string): CooldownReasonCategory => {
  if (/insufficient[_ ]quota|quota\s*(?:exceeded|exhausted)/i.test(text))
    return 'quota';
  if (
    /usage\s*limit|limit\s*(?:exhausted|reached)|subscription(?:[_ ]sharing)?[_ ]usage[_ ]limit/i.test(
      text,
    )
  )
    return 'usage_limit';
  if (/too\s*many\s*requests/i.test(text)) return 'too_many_requests';
  if (/429|\b1308\b|\b1310\b|rate\s*limit/i.test(text)) return 'rate_limit';
  return 'unknown';
};

export const classifyLimitError = (text: string): boolean =>
  /(?:429|\b1308\b|\b1310\b|rate\s*limit|usage\s*limit|limit\s*(?:exhausted|reached)|quota\s*(?:exceeded|exhausted)|too\s*many\s*requests|insufficient[_ ]quota|subscription(?:[_ ]sharing)?[_ ]usage[_ ]limit)/i.test(
    text,
  );

export const getCooldownDefaultMs = (): number => DEFAULT_COOLDOWN_MS;

const safeTimeZone = (value: string): boolean => {
  try {
    new Intl.DateTimeFormat('en-US', { timeZone: value }).format();
    return true;
  } catch {
    return false;
  }
};

const timezoneOffset = (time: number, timeZone: string): number => {
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone,
    hour12: false,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  }).formatToParts(new Date(time));
  const values: Record<string, number> = {};
  for (const part of parts) {
    if (part.type !== 'literal') values[part.type] = Number(part.value);
  }
  return (
    Date.UTC(
      values.year,
      values.month - 1,
      values.day,
      values.hour === 24 ? 0 : values.hour,
      values.minute,
      values.second,
    ) - time
  );
};

const localDateCandidates = (
  year: number,
  month: number,
  day: number,
  hour: number,
  minute: number,
  second: number,
  timeZone: string,
): number[] => {
  const naive = Date.UTC(year, month - 1, day, hour, minute, second);
  const offsets = new Set<number>();
  for (let delta = -48; delta <= 48; delta += 6) {
    offsets.add(timezoneOffset(naive + delta * 60 * 60 * 1000, timeZone));
  }
  const candidates = [...offsets]
    .map((offset) => naive - offset)
    .filter((candidate) => {
      const parts = new Intl.DateTimeFormat('en-US', {
        timeZone,
        hour12: false,
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      }).formatToParts(new Date(candidate));
      const values: Record<string, number> = {};
      for (const part of parts) {
        if (part.type !== 'literal') values[part.type] = Number(part.value);
      }
      return (
        values.year === year &&
        values.month === month &&
        values.day === day &&
        (values.hour === 24 ? 0 : values.hour) === hour &&
        values.minute === minute &&
        values.second === second
      );
    })
    .sort((a, b) => a - b);
  return candidates.length > 0
    ? candidates
    : [naive - timezoneOffset(naive, timeZone)];
};

const truncateRaw = (text: string): string => text.slice(0, RAW_RESET_TEXT_MAX);

// Any computed reset time that is non-finite, unconstructible, already past,
// or more than 7 days out is untrustworthy (e.g. a malformed "retry in
// 9999999999999999 hours" that would otherwise overflow Date's valid range
// and throw out of toISOString()). Fall back to a fixed, safe 30-minute
// cooldown instead — this is what prevents a bad provider message from ever
// throwing out of route().
const clampUntil = (
  candidate: number,
  now: number,
): { until: number; untrusted: boolean } => {
  const valid =
    Number.isFinite(candidate) && !Number.isNaN(new Date(candidate).getTime());
  if (!valid || candidate <= now || candidate - now > SEVEN_DAYS_MS) {
    return { until: now + DEFAULT_COOLDOWN_MS, untrusted: true };
  }
  return { until: candidate, untrusted: false };
};

export const parseResetTime = (
  text: string,
  provider: string,
  config: RouterConfig,
  now: number,
): { until: number; raw: string; assumedTz: string } => {
  const explicit = text.match(
    /\b(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?\s*(?:Z|[+-]\d{2}:?\d{2}))\b/i,
  );
  if (explicit) {
    const parsed = Date.parse(explicit[1]);
    if (Number.isFinite(parsed)) {
      const clamped = clampUntil(parsed, now);
      return {
        until: clamped.until,
        raw: truncateRaw(explicit[1]),
        assumedTz: clamped.untrusted ? 'untrusted' : 'explicit',
      };
    }
  }

  const retry = text.match(
    /\bretry\s+in\s+(\d+(?:\.\d+)?)\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)\b/i,
  );
  if (retry) {
    const amount = Number(retry[1]);
    const unit = retry[2].toLowerCase();
    const multiplier = unit.startsWith('h')
      ? 60 * 60 * 1000
      : unit.startsWith('m')
        ? 60 * 1000
        : 1000;
    const clamped = clampUntil(now + amount * multiplier, now);
    return {
      until: clamped.until,
      raw: truncateRaw(retry[0]),
      assumedTz: clamped.untrusted ? 'untrusted' : 'relative',
    };
  }

  const naive = text.match(
    /\b(?:reset(?:s|ting)?\s+at|reset\s*:?)\s*(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})\b/i,
  );
  const timeZone =
    config.providerTimezones?.[provider] ??
    DEFAULT_TIMEZONES[provider] ??
    'UTC';
  if (naive && safeTimeZone(timeZone)) {
    const [, year, month, day, hour, minute, second] = naive;
    const candidate = localDateCandidates(
      Number(year),
      Number(month),
      Number(day),
      Number(hour),
      Number(minute),
      Number(second),
      timeZone,
    )[0];
    const clamped = clampUntil(candidate, now);
    return {
      until: clamped.until,
      raw: truncateRaw(naive[0]),
      assumedTz: clamped.untrusted ? 'untrusted' : timeZone,
    };
  }

  return {
    until: now + DEFAULT_COOLDOWN_MS,
    raw: '',
    assumedTz: 'UTC',
  };
};

export const defaultCooldownDir = (): string =>
  process.env.PI_MODEL_ROUTER_COOLDOWNS ??
  join(homedir(), '.local', 'state', 'pi-model-router', 'cooldowns');

const entrySignature = (entry: CooldownEntry): string =>
  `${entry.until}|${entry.recorded_at}|${entry.source}`;

const parseEntries = (value: unknown): Record<string, CooldownEntry> => {
  if (!isObject(value)) throw new Error('invalid cooldown file');
  const entries: Record<string, CooldownEntry> = {};
  for (const [ref, raw] of Object.entries(value)) {
    if (!isObject(raw)) continue;
    const until = parseTime(raw.until);
    if (until === undefined) continue;
    if (!isSource(raw.source)) continue;
    entries[ref] = {
      until: raw.until as string,
      reason: isReasonCategory(raw.reason) ? raw.reason : 'unknown',
      raw_reset_text:
        typeof raw.raw_reset_text === 'string'
          ? truncateRaw(raw.raw_reset_text)
          : '',
      assumed_tz: typeof raw.assumed_tz === 'string' ? raw.assumed_tz : 'UTC',
      source: raw.source,
      recorded_at:
        typeof raw.recorded_at === 'string' ? raw.recorded_at : iso(until),
    };
  }
  return entries;
};

const maxMergeInto = (
  target: Record<string, CooldownEntry>,
  source: Record<string, CooldownEntry>,
): void => {
  for (const [ref, entry] of Object.entries(source)) {
    const existing = target[ref];
    if (
      !existing ||
      (parseTime(existing.until) ?? 0) <= (parseTime(entry.until) ?? 0)
    ) {
      target[ref] = entry;
    }
  }
};

export const createCooldownStore = (
  dir = defaultCooldownDir(),
  nowFn: () => number = Date.now,
  factoryStatePath: string = FACTORY_STATE_PATH,
  ownId: string = `${process.pid}-${nowFn()}`,
): CooldownStore => {
  const ownFile = join(dir, `${ownId}.json`);
  let ownEntries: Record<string, CooldownEntry> = {};

  const loggedErrors = new Set<string>();
  // Every failure here must be logged at most once and never thrown into the
  // caller: route() must always get a safe fallback instead of a crash.
  const logOnce = (context: string, error: unknown): void => {
    const message = `pi-model-router: cooldown ${context} failed: ${String(error)}`;
    if (loggedErrors.has(message)) return;
    loggedErrors.add(message);
    console.error(message);
  };

  const persistOwn = (): void => {
    // The temp name must be unpredictable: a fixed/time-derived name let an
    // attacker pre-create a FIFO at the exact path this process will write,
    // and a plain writeFileSync() blocks forever opening it for writing with
    // no reader. pid + 8 random hex bytes makes the path unguessable, and
    // O_EXCL|O_NOFOLLOW|O_NONBLOCK on top means even a guessed/pre-existing
    // path (FIFO, symlink, or otherwise) is refused instead of opened.
    let temporary: string | undefined;
    // Only set once openSync has actually created `temporary`: if O_EXCL
    // rejects a colliding pre-existing path (FIFO, symlink, attacker file,
    // or -- astronomically unlikely but not provably impossible -- a random
    // collision with another of our own in-flight writes), that path was
    // never ours, and unlinking it on cleanup would delete someone else's
    // file instead of our own failed write.
    let createdTemporary = false;
    try {
      mkdirBoundedSync(dir);
      chmodSync(dir, 0o700);
      temporary = `${ownFile}.${process.pid}.${randomBytes(4).toString('hex')}.tmp`;
      let fd: number | undefined;
      try {
        fd = openSync(
          temporary,
          fsConstants.O_WRONLY |
            fsConstants.O_CREAT |
            fsConstants.O_EXCL |
            fsConstants.O_NOFOLLOW |
            fsConstants.O_NONBLOCK,
          0o600,
        );
        createdTemporary = true;
        writeSync(fd, `${JSON.stringify({ entries: ownEntries }, null, 2)}\n`);
      } finally {
        if (fd !== undefined) closeSync(fd);
      }
      chmodSync(temporary, 0o600);
      renameSync(temporary, ownFile);
      chmodSync(ownFile, 0o600);
    } catch (error) {
      if (temporary !== undefined && createdTemporary) {
        try {
          unlinkSync(temporary);
        } catch {
          // Best-effort cleanup only; nothing we can do if this also fails.
        }
      }
      // Write failures are ignored: this turn's decision already has the
      // in-memory entry; persistence to disk is best-effort only.
      logOnce('write', error);
    }
  };

  // Reads every OTHER process's cooldown file in the directory, merging by
  // max(until) per model, skipping unreadable/malformed files, and ignoring
  // (then best-effort deleting) files older than 24h.
  // Returns one {ref, entry} pair per (foreign file, model) -- deliberately
  // UNMERGED across files, unlike the old max-per-model dict, so clear() can
  // snapshot and independently suppress every foreign process's entry for a
  // model, not just whichever one currently has the largest `until`.
  const readForeignFiles = (
    now: number,
  ): Array<{ ref: string; entry: CooldownEntry }> => {
    const raw: Array<{ ref: string; entry: CooldownEntry }> = [];
    if (!existsSync(dir)) return raw; // Nothing has cooled down yet; not an error.
    let names: string[];
    try {
      names = readdirSync(dir);
    } catch (error) {
      logOnce('readdir', error);
      return raw;
    }
    for (const name of names) {
      if (!name.endsWith('.json') || name === `${ownId}.json`) continue;
      const filePath = join(dir, name);
      try {
        const stat = statSync(filePath);
        if (now - stat.mtimeMs > STALE_FILE_MS) {
          try {
            unlinkSync(filePath);
          } catch {
            // Best-effort cleanup only; another process may have raced us.
          }
          continue;
        }
        const content = readRegularFileBounded(filePath);
        if (content === undefined) continue; // Not a regular file, too large, or unreadable.
        const parsed: unknown = JSON.parse(content);
        if (!isObject(parsed)) continue;
        for (const [ref, entry] of Object.entries(
          parseEntries(parsed.entries),
        )) {
          raw.push({ ref, entry });
        }
      } catch {
        continue; // Unreadable or malformed file: skip it, never throw.
      }
    }
    return raw;
  };

  // Computed deterministically from the factory file's own observation
  // timestamp — never from "now" — so re-reading an unchanged file always
  // yields the same until and never renews a cooldown.
  const readFactoryEntries = (now: number): Record<string, CooldownEntry> => {
    try {
      if (!existsSync(factoryStatePath)) return {};
      const stat = statSync(factoryStatePath);
      const content = readRegularFileBounded(factoryStatePath);
      if (content === undefined) return {};
      const parsed: unknown = JSON.parse(content);
      if (!isObject(parsed)) return {};
      const anchor = parseTime(parsed.probed_at) ?? parseTime(parsed.since);
      if (anchor === undefined) return {}; // No reliable observation time.
      const stamp = Math.min(anchor, stat.mtimeMs);
      if (now - stamp > STALE_FILE_MS) return {};
      const exhausted = parsed.exhausted_models;
      if (!Array.isArray(exhausted)) return {};
      const until = anchor + DEFAULT_COOLDOWN_MS;
      const entries: Record<string, CooldownEntry> = {};
      for (const model of exhausted) {
        if (typeof model !== 'string' || !model.includes('/')) continue;
        entries[model] = {
          until: iso(until),
          reason: 'unknown',
          raw_reset_text: '',
          assumed_tz: 'UTC',
          source: 'factory',
          recorded_at: iso(anchor),
        };
      }
      return entries;
    } catch (error) {
      logOnce('factory', error);
      return {};
    }
  };

  // Keyed off the caller-supplied `now` (the same value route() reuses across
  // every cooldownStore call in one turn), not the real wall clock: this both
  // satisfies "at most once per route() call" and "cache for 2s" without a
  // second, independent notion of time.
  //
  // Deliberately returns every (ref, entry) pair UNMERGED, one per foreign
  // file plus at most one from the factory file: a merged max-per-model dict
  // here would let a second foreign entry for the same model hide behind
  // whichever one currently wins, so clear() could only ever snapshot and
  // suppress the winner -- the exact bug this shape avoids (see clear()).
  let foreignCache:
    | {
        expiresAt: number;
        entries: Array<{ ref: string; entry: CooldownEntry }>;
      }
    | undefined;
  const getForeignAndFactory = (
    now: number,
  ): Array<{ ref: string; entry: CooldownEntry }> => {
    if (foreignCache && now < foreignCache.expiresAt) {
      return foreignCache.entries;
    }
    const entries: Array<{ ref: string; entry: CooldownEntry }> = [
      ...readForeignFiles(now),
      ...Object.entries(readFactoryEntries(now)).map(([ref, entry]) => ({
        ref,
        entry,
      })),
    ];
    foreignCache = { expiresAt: now + READ_CACHE_MS, entries };
    return entries;
  };

  // ref -> signatures of every foreign/factory entry that was present for
  // that model when clear() last ran, snapshotted per-entry (not just for
  // whichever entry currently has the largest `until`). Each signature is
  // suppressed independently: an unchanged foreign entry stays hidden across
  // repeated merges, while any entry whose signature doesn't match one in the
  // set (a fresh write, or one that was never cleared) is shown normally.
  const suppressed = new Map<string, Set<string>>();

  const mergedView = (now: number): Record<string, CooldownEntry> => {
    const byRef = new Map<string, CooldownEntry[]>();
    for (const { ref, entry } of getForeignAndFactory(now)) {
      const suppressedSignatures = suppressed.get(ref);
      if (suppressedSignatures?.has(entrySignature(entry))) continue;
      const bucket = byRef.get(ref);
      if (bucket) bucket.push(entry);
      else byRef.set(ref, [entry]);
    }
    const visible: Record<string, CooldownEntry> = {};
    for (const [ref, entries] of byRef) {
      visible[ref] = entries.reduce((best, candidate) =>
        (parseTime(candidate.until) ?? 0) > (parseTime(best.until) ?? 0)
          ? candidate
          : best,
      );
    }
    maxMergeInto(visible, ownEntries); // Own state is always authoritative/live.
    return visible;
  };

  // `now` and `recordedAt` are deliberately separate: `now` drives expiry
  // math only (relative "retry in N min" parsing and the default cooldown
  // length, both inherently about real wall-clock time), while `recordedAt`
  // becomes the entry's `recorded_at` -- the event time noteSuccess compares
  // a later success against. Route-time ('retry') failures have no earlier
  // event time than the moment they're observed, so recordedAt defaults to
  // now; transcript-scanned failures pass the failed message's own
  // timestamp so a success recorded in the same scan (whose own timestamp
  // is always <= scan time) compares against when the failure actually
  // happened, not when this scan happened to run.
  const buildEntry = (
    errorText: string,
    source: CooldownSource,
    provider: string,
    config: RouterConfig,
    now: number,
    recordedAt: number,
  ): CooldownEntry => {
    const parsed = parseResetTime(errorText, provider, config, now);
    return {
      until: iso(parsed.until),
      reason: categorizeReason(errorText),
      raw_reset_text: parsed.raw,
      assumed_tz: parsed.assumedTz,
      source,
      recorded_at: iso(recordedAt),
    };
  };

  return {
    load: (now = nowFn()) => {
      try {
        return { entries: mergedView(now) };
      } catch (error) {
        logOnce('load', error);
        return { entries: {} };
      }
    },

    // Every input here -- including a malformed/out-of-range `recordedAt`
    // forwarded from a transcript message's own timestamp -- must be able to
    // flow through buildEntry() without ever throwing out of record(): it is
    // called directly from route()'s hot path, and a thrown RangeError there
    // (e.g. from iso() on a non-finite or out-of-range recordedAt) would
    // break the turn instead of degrading to "no cooldown recorded".
    record: (
      modelRef,
      errorText,
      source,
      config,
      now = nowFn(),
      recordedAt = now,
    ) => {
      try {
        const provider = modelRef.slice(0, modelRef.indexOf('/'));
        const entry = buildEntry(
          errorText,
          source,
          provider,
          config,
          now,
          recordedAt,
        );
        ownEntries = { ...ownEntries, [modelRef]: entry };
        persistOwn();
        return entry;
      } catch (error) {
        logOnce('record', error);
        // Date.now() (not nowFn()) here deliberately: nowFn is caller-supplied
        // and must not be trusted to produce a valid fallback given we just
        // caught a failure that may have originated from a bad `now`/`recordedAt`.
        const fallbackNow = Date.now();
        return {
          until: iso(fallbackNow + DEFAULT_COOLDOWN_MS),
          reason: 'unknown',
          raw_reset_text: '',
          assumed_tz: 'UTC',
          source,
          recorded_at: iso(fallbackNow),
        };
      }
    },

    noteSuccess: (modelRef, successTimestamp) => {
      try {
        const existing = ownEntries[modelRef];
        if (!existing) return;
        if (successTimestamp !== undefined) {
          const recordedAt = parseTime(existing.recorded_at);
          // A success older than (or exactly at) the latest recorded failure
          // must never clear it -- only a strictly newer success may.
          if (recordedAt !== undefined && successTimestamp <= recordedAt)
            return;
        }
        const { [modelRef]: _removed, ...rest } = ownEntries;
        ownEntries = rest;
        persistOwn();
      } catch (error) {
        logOnce('noteSuccess', error);
      }
    },

    candidates: (modelRefs, now = nowFn(), isUsable) => {
      try {
        const usable = (ref: string): boolean =>
          isUsable ? isUsable(ref) : true;
        const usableRefs = modelRefs.filter(usable);
        if (usableRefs.length === 0) return [...modelRefs];

        const view = mergedView(now);
        const isCooling = (ref: string): boolean => {
          const entry = view[ref];
          return entry !== undefined && (parseTime(entry.until) ?? 0) > now;
        };

        const coolingUsable = usableRefs.filter(isCooling);
        if (coolingUsable.length === 0) return [...modelRefs];
        const keptUsable = usableRefs.filter((ref) => !isCooling(ref));
        if (keptUsable.length > 0) {
          return modelRefs.filter((ref) => !coolingUsable.includes(ref));
        }
        // Every usable candidate is cooling: keep only the soonest-expiring
        // usable one so filtering never leaves zero usable models.
        const soonestRef = [...coolingUsable].sort(
          (a, b) =>
            (parseTime(view[a]?.until) ?? 0) - (parseTime(view[b]?.until) ?? 0),
        )[0];
        return modelRefs.filter(
          (ref) => !coolingUsable.includes(ref) || ref === soonestRef,
        );
      } catch (error) {
        logOnce('candidates', error);
        return [...modelRefs];
      }
    },

    clear: (modelRef) => {
      try {
        const now = nowFn();
        // Snapshot EVERY foreign/factory entry for each cleared model
        // individually -- not just the one mergedView would currently pick as
        // the max -- so a lower-`until` foreign entry that was merely masked
        // by a larger own or foreign entry doesn't resurface, unsuppressed,
        // the moment that larger entry is cleared or expires.
        const foreignEntries = getForeignAndFactory(now);
        const suppressRef = (ref: string): void => {
          const matches = foreignEntries.filter((item) => item.ref === ref);
          if (matches.length === 0) return;
          const signatures = suppressed.get(ref) ?? new Set<string>();
          for (const { entry } of matches)
            signatures.add(entrySignature(entry));
          suppressed.set(ref, signatures);
        };
        if (!modelRef || modelRef === 'all') {
          const refs = new Set<string>([
            ...Object.keys(ownEntries),
            ...foreignEntries.map((item) => item.ref),
          ]);
          for (const ref of refs) suppressRef(ref);
          ownEntries = {};
        } else {
          delete ownEntries[modelRef];
          suppressRef(modelRef);
        }
        persistOwn();
      } catch (error) {
        logOnce('clear', error);
      }
    },

    formatStatus: (now = nowFn()) => {
      try {
        const view = mergedView(now);
        return Object.entries(view)
          .filter(([, entry]) => (parseTime(entry.until) ?? 0) > now)
          .sort(([a], [b]) => a.localeCompare(b))
          .map(([ref, entry]) => {
            const until = parseTime(entry.until) ?? now;
            return `${ref}: until ${new Date(until).toLocaleString()} (UTC ${new Date(until).toISOString()}); reset=${entry.raw_reset_text || 'none'}; assumed_tz=${entry.assumed_tz}; source=${entry.source}`;
          });
      } catch (error) {
        logOnce('formatStatus', error);
        return [];
      }
    },
  };
};
