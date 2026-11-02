import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  statSync,
  utimesSync,
  writeFileSync,
} from 'node:fs';
import { execFileSync, spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  createCooldownStore,
  classifyLimitError,
  categorizeReason,
  parseResetTime,
  getCooldownDefaultMs,
} from './cooldown';
import type { RouterConfig } from './types';

const baseConfig: RouterConfig = { profiles: {} };

describe('cooldown.ts', () => {
  let dir: string;
  let factoryPath: string;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'pmr-cooldown-test-'));
    factoryPath = join(dir, 'provider-switch.json');
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  describe('classifyLimitError', () => {
    it.each([
      '429 Too Many Requests',
      '{"code":"1308","message":"limit"}',
      '{"code":"1310","message":"Weekly/Monthly Limit Exhausted. Your limit will reset at 2026-10-05 04:08:27"}',
      'rate limit exceeded',
      'usage limit reached',
      'limit exhausted',
      'quota exceeded',
      'quota exhausted',
      'too many requests',
      'insufficient_quota',
      'subscription_sharing_usage_limit_exceeded',
    ])('classifies "%s" as a limit error', (text) => {
      expect(classifyLimitError(text)).toBe(true);
    });

    it('does not classify unrelated errors', () => {
      expect(classifyLimitError('500 internal server error')).toBe(false);
      expect(classifyLimitError('invalid api key')).toBe(false);
      expect(classifyLimitError('context length exceeded')).toBe(false);
    });
  });

  describe('categorizeReason', () => {
    it.each([
      ['429 rate limit exceeded', 'rate_limit'],
      ['429 Too Many Requests', 'too_many_requests'],
      ['insufficient_quota', 'quota'],
      ['quota exceeded', 'quota'],
      ['usage limit reached', 'usage_limit'],
      ['too many requests', 'too_many_requests'],
      ['some unrecognized failure text', 'unknown'],
    ])('categorizes "%s" as "%s"', (text, category) => {
      expect(categorizeReason(text)).toBe(category);
    });
  });

  describe('parseResetTime', () => {
    const now = Date.parse('2026-11-01T00:00:00Z');

    it('parses an explicit offset timestamp exactly', () => {
      const result = parseResetTime(
        '429 {"message":"reset at 2026-11-02T04:08:27+08:00"}',
        'zai',
        baseConfig,
        now,
      );
      expect(result.until).toBe(Date.parse('2026-11-02T04:08:27+08:00'));
      expect(result.assumedTz).toBe('explicit');
    });

    it('parses an explicit Z timestamp exactly', () => {
      const result = parseResetTime(
        'limit resets at 2026-11-02T04:08:27Z',
        'openai',
        baseConfig,
        now,
      );
      expect(result.until).toBe(Date.parse('2026-11-02T04:08:27Z'));
      expect(result.assumedTz).toBe('explicit');
    });

    it.each([
      ['retry in 30 s', 30 * 1000],
      ['retry in 5 min', 5 * 60 * 1000],
      ['retry in 2 hours', 2 * 60 * 60 * 1000],
    ])('parses relative "%s" as now + duration', (text, deltaMs) => {
      const result = parseResetTime(text, 'openai', baseConfig, now);
      expect(result.until).toBe(now + deltaMs);
      expect(result.assumedTz).toBe('relative');
    });

    it('interprets a naive reset time in the zai provider default timezone (Asia/Shanghai)', () => {
      const result = parseResetTime(
        'Your limit will reset at 2026-11-02 04:08:27',
        'zai',
        baseConfig,
        now,
      );
      // 2026-11-02 04:08:27 Asia/Shanghai (UTC+8, no DST) == 2026-11-01 20:08:27 UTC.
      expect(result.until).toBe(Date.parse('2026-11-01T20:08:27Z'));
      expect(result.assumedTz).toBe('Asia/Shanghai');
    });

    it('defaults unknown providers to UTC for naive reset times', () => {
      const result = parseResetTime(
        'reset at 2026-11-02 04:08:27',
        'openai',
        baseConfig,
        now,
      );
      expect(result.until).toBe(Date.parse('2026-11-02T04:08:27Z'));
      expect(result.assumedTz).toBe('UTC');
    });

    it('honors a configured providerTimezones override', () => {
      const config: RouterConfig = {
        profiles: {},
        providerTimezones: { zai: 'UTC' },
      };
      const result = parseResetTime(
        'reset at 2026-11-02 04:08:27',
        'zai',
        config,
        now,
      );
      expect(result.until).toBe(Date.parse('2026-11-02T04:08:27Z'));
      expect(result.assumedTz).toBe('UTC');
    });

    it('falls back to now + 30 min and marks untrusted when the naive time is already past', () => {
      const result = parseResetTime(
        'reset at 2026-10-01 00:00:00',
        'zai',
        baseConfig,
        now,
      );
      expect(result.until).toBe(now + getCooldownDefaultMs());
      expect(result.assumedTz).toBe('untrusted');
    });

    it('falls back to now + 30 min and marks untrusted when the naive time is more than 7 days out', () => {
      const result = parseResetTime(
        'reset at 2026-11-20 00:00:00',
        'zai',
        baseConfig,
        now,
      );
      expect(result.until).toBe(now + getCooldownDefaultMs());
      expect(result.assumedTz).toBe('untrusted');
    });

    it('falls back to now + 30 min when the error text names no reset time', () => {
      const result = parseResetTime(
        '429 rate limit',
        'openai',
        baseConfig,
        now,
      );
      expect(result.until).toBe(now + getCooldownDefaultMs());
      expect(result.raw).toBe('');
    });

    it('prefers the earlier plausible interpretation for an ambiguous DST fall-back local time', () => {
      const config: RouterConfig = {
        profiles: {},
        providerTimezones: { acme: 'America/New_York' },
      };
      // 2026-11-01 is the US DST fall-back day; 01:30 local occurs twice
      // (05:30Z while still EDT, then 06:30Z once EST takes effect).
      const result = parseResetTime(
        'reset at 2026-11-01 01:30:00',
        'acme',
        config,
        now,
      );
      expect(result.until).toBe(Date.parse('2026-11-01T05:30:00Z'));
      expect(result.assumedTz).toBe('America/New_York');
    });

    // Regression: review 2 found a huge relative duration ("retry in N hours")
    // could overflow Date's representable range and throw out of
    // toISOString(). It must instead degrade to a safe, bounded cooldown.
    it('never throws on an absurdly large relative duration; falls back to untrusted 30 min', () => {
      expect(() => {
        const result = parseResetTime(
          'retry in 9999999999999999 hours',
          'openai',
          baseConfig,
          now,
        );
        expect(result.until).toBe(now + getCooldownDefaultMs());
        expect(result.assumedTz).toBe('untrusted');
      }).not.toThrow();
    });

    it('truncates raw_reset_text to at most 40 characters', () => {
      const result = parseResetTime(
        'retry in 9999999999999999999999999999999999999999 hours',
        'openai',
        baseConfig,
        now,
      );
      expect(result.raw.length).toBeLessThanOrEqual(40);
    });
  });

  describe('createCooldownStore', () => {
    const makeStore = (nowFn: () => number, ownId = 'proc-a') =>
      createCooldownStore(dir, nowFn, factoryPath, ownId);

    it('records a cooldown and filters it out of candidates until it expires', () => {
      const store = makeStore(() => Date.parse('2026-11-01T00:00:00Z'));
      store.record('zai/glm-5.3', '429 rate limit', 'retry', baseConfig);
      expect(
        store.candidates(['zai/glm-5.3', 'anthropic/claude-opus-5-5']),
      ).toEqual(['anthropic/claude-opus-5-5']);
    });

    it('keeps the soonest-expiring model when every candidate is cooling down', () => {
      const now = Date.parse('2026-11-01T00:00:00Z');
      const store = makeStore(() => now);
      store.record('a/one', 'retry in 10 min', 'retry', baseConfig, now);
      store.record('a/two', 'retry in 60 min', 'retry', baseConfig, now);
      expect(store.candidates(['a/one', 'a/two'], now)).toEqual(['a/one']);
    });

    it('leaves fallbacks not in the candidate list untouched by filtering', () => {
      const now = Date.parse('2026-11-01T00:00:00Z');
      const store = makeStore(() => now);
      store.record('a/one', 'retry in 10 min', 'retry', baseConfig, now);
      expect(store.candidates(['a/one', 'a/two', 'a/three'], now)).toEqual([
        'a/two',
        'a/three',
      ]);
    });

    it('makes an expired entry immediately available again, with no probe/lease claiming', () => {
      let now = Date.parse('2026-11-01T00:00:00Z');
      const store = makeStore(() => now);
      store.record('a/one', 'retry in 1 min', 'retry', baseConfig, now);
      now += 2 * 60 * 1000; // past the cooldown
      // Every caller sees it as available once expired — no lease, no claim.
      expect(store.candidates(['a/one', 'a/two'], now)).toEqual([
        'a/one',
        'a/two',
      ]);
      expect(store.candidates(['a/one', 'a/two'], now)).toEqual([
        'a/one',
        'a/two',
      ]);
    });

    it('clears the entry once a later response succeeds', () => {
      let now = Date.parse('2026-11-01T00:00:00Z');
      const store = makeStore(() => now);
      store.record('a/one', 'retry in 1 min', 'retry', baseConfig, now);
      now += 2 * 60 * 1000;
      store.noteSuccess('a/one');
      expect(store.load(now).entries['a/one']).toBeUndefined();
    });

    it('re-arms a new cooldown on a second limit error after the first expires', () => {
      let now = Date.parse('2026-11-01T00:00:00Z');
      const store = makeStore(() => now);
      store.record('a/one', 'retry in 1 min', 'retry', baseConfig, now);
      now += 2 * 60 * 1000;
      store.record('a/one', 'retry in 30 min', 'retry', baseConfig, now);
      const entry = store.load(now).entries['a/one'];
      expect(Date.parse(entry.until)).toBe(now + 30 * 60 * 1000);
      expect(store.candidates(['a/one', 'a/two'], now)).toEqual(['a/two']);
    });

    it('clears a single model, and "all" clears every cooldown', () => {
      const store = makeStore(() => Date.now());
      store.record('a/one', '429', 'retry', baseConfig);
      store.record('a/two', '429', 'retry', baseConfig);
      store.clear('a/one');
      expect(Object.keys(store.load().entries)).toEqual(['a/two']);
      store.clear('all');
      expect(store.load().entries).toEqual({});
    });

    it('writes the store file and directory with restrictive permissions', () => {
      const store = makeStore(() => Date.now());
      store.record('a/one', '429', 'retry', baseConfig);
      expect(statSync(dir).mode & 0o777).toBe(0o700);
      expect(statSync(join(dir, 'proc-a.json')).mode & 0o777).toBe(0o600);
    });

    it('formats status with local time, UTC time, raw reset text, assumed tz, and source', () => {
      const now = Date.parse('2026-11-01T00:00:00Z');
      const store = makeStore(() => now);
      store.record('zai/glm-5.3', 'retry in 10 min', 'retry', baseConfig, now);
      const lines = store.formatStatus(now);
      expect(lines).toHaveLength(1);
      expect(lines[0]).toContain('zai/glm-5.3');
      expect(lines[0]).toContain('UTC 2026-11-01T00:10:00.000Z');
      expect(lines[0]).toContain('reset=retry in 10 min');
      expect(lines[0]).toContain('assumed_tz=relative');
      expect(lines[0]).toContain('source=retry');
    });

    describe('per-process files merge across processes', () => {
      it("merges two processes' own files, keeping max(until) per model", () => {
        const now = Date.parse('2026-11-01T00:00:00Z');
        const storeA = makeStore(() => now, 'proc-a');
        const storeB = makeStore(() => now, 'proc-b');
        storeA.record('a/one', 'retry in 10 min', 'retry', baseConfig, now);
        storeB.record('a/one', 'retry in 60 min', 'retry', baseConfig, now);

        expect(Date.parse(storeA.load(now).entries['a/one'].until)).toBe(
          now + 60 * 60 * 1000,
        );
        expect(Date.parse(storeB.load(now).entries['a/one'].until)).toBe(
          now + 60 * 60 * 1000,
        );
      });

      it("noteSuccess only removes the entry from the calling process's own file", () => {
        const now = Date.parse('2026-11-01T00:00:00Z');
        const storeA = makeStore(() => now, 'proc-a');
        const storeB = makeStore(() => now, 'proc-b');
        storeB.record('a/one', 'retry in 60 min', 'retry', baseConfig, now);

        storeA.noteSuccess('a/one'); // storeA never had this entry; no-op.
        // storeB's file still cools; a fresh reader must still see it.
        const storeC = makeStore(() => now, 'proc-c');
        expect(storeC.load(now).entries['a/one']).toBeDefined();
      });

      it('skips unreadable or malformed files and still returns the valid entries', () => {
        const now = Date.parse('2026-11-01T00:00:00Z');
        mkdirSync(dir, { recursive: true });
        writeFileSync(join(dir, 'proc-corrupt.json'), '{not json', {
          mode: 0o600,
        });
        const store = makeStore(() => now, 'proc-a');
        store.record('a/one', 'retry in 10 min', 'retry', baseConfig, now);
        expect(() => store.load(now)).not.toThrow();
        expect(store.load(now).entries['a/one']).toBeDefined();
      });

      it('ignores and best-effort deletes files older than 24h', () => {
        const now = Date.parse('2026-11-02T00:00:00Z');
        const stalePath = join(dir, 'proc-stale.json');
        mkdirSync(dir, { recursive: true });
        writeFileSync(
          stalePath,
          JSON.stringify({
            entries: {
              'a/one': {
                until: new Date(now + 60 * 60 * 1000).toISOString(),
                reason: 'rate_limit',
                raw_reset_text: '',
                assumed_tz: 'UTC',
                source: 'retry',
                recorded_at: new Date(now).toISOString(),
              },
            },
          }),
        );
        const oldMtime = new Date(now - 25 * 60 * 60 * 1000);
        utimesSync(stalePath, oldMtime, oldMtime);

        const store = makeStore(() => now, 'proc-a');
        expect(store.load(now).entries['a/one']).toBeUndefined();
        expect(existsSync(stalePath)).toBe(false);
      });
    });

    describe("clear() suppresses other processes' entries until they change", () => {
      it('ignores a still-present foreign entry after clear(), until it changes', () => {
        const now = Date.parse('2026-11-01T00:00:00Z');
        const storeA = makeStore(() => now, 'proc-a');
        const storeB = makeStore(() => now, 'proc-b');
        storeB.record('a/one', 'retry in 60 min', 'retry', baseConfig, now);
        expect(storeA.load(now).entries['a/one']).toBeDefined();

        storeA.clear('a/one');
        expect(storeA.load(now).entries['a/one']).toBeUndefined();
        // Unaffected: storeB (and a fresh reader) still see the cooldown.
        const storeC = makeStore(() => now, 'proc-c');
        expect(storeC.load(now).entries['a/one']).toBeDefined();

        // storeB re-records (a changed entry) — storeA must see it again
        // once its 2s foreign-read cache has expired.
        const later = now + 3000; // beyond the 2s foreign-read cache window
        storeB.record('a/one', 'retry in 90 min', 'retry', baseConfig, later);
        expect(storeA.load(later).entries['a/one']).toBeDefined();
      });

      it('"clear all" suppresses every currently-known entry, own and foreign', () => {
        const now = Date.parse('2026-11-01T00:00:00Z');
        const storeA = makeStore(() => now, 'proc-a');
        const storeB = makeStore(() => now, 'proc-b');
        storeA.record('a/one', 'retry in 10 min', 'retry', baseConfig, now);
        storeB.record('a/two', 'retry in 10 min', 'retry', baseConfig, now);

        storeA.clear('all');
        expect(storeA.load(now).entries).toEqual({});
      });

      // Review 3 finding 3: clear() used to snapshot only the merged WINNER
      // (whichever of own/foreign currently had the largest `until`) and
      // suppress just that one signature. When the winner was the own entry,
      // a smaller but genuinely UNCHANGED foreign entry for the same model was
      // never snapshotted at all, so deleting the own entry let that foreign
      // entry resurface unsuppressed -- clear() appeared to do nothing. clear()
      // must snapshot and suppress every foreign entry for the model
      // independently of which one currently wins the merge.
      it('clear(model) suppresses a smaller foreign entry that an own entry was masking, not just the merged winner', () => {
        const now = Date.parse('2026-11-01T00:00:00Z');
        const storeOwn = makeStore(() => now, 'proc-own');
        const storeForeign = makeStore(() => now, 'proc-foreign');
        storeOwn.record('a/one', 'retry in 60 min', 'retry', baseConfig, now);
        storeForeign.record(
          'a/one',
          'retry in 10 min',
          'retry',
          baseConfig,
          now,
        );

        // The own entry (larger until) currently wins the merge.
        expect(storeOwn.candidates(['a/one', 'a/two'], now)).toEqual(['a/two']);

        storeOwn.clear('a/one');
        expect(storeOwn.load(now).entries['a/one']).toBeUndefined();
        // The foreign +10 min entry must stay suppressed: it was only ever
        // masked by the larger own entry, it never itself changed.
        expect(storeOwn.candidates(['a/one', 'a/two'], now)).toEqual([
          'a/one',
          'a/two',
        ]);
      });

      it('clear("all") suppresses a smaller foreign entry for each cleared model, not just the merged winner', () => {
        const now = Date.parse('2026-11-01T00:00:00Z');
        const storeOwn = makeStore(() => now, 'proc-own');
        const storeForeign = makeStore(() => now, 'proc-foreign');
        storeOwn.record('a/one', 'retry in 60 min', 'retry', baseConfig, now);
        storeForeign.record(
          'a/one',
          'retry in 10 min',
          'retry',
          baseConfig,
          now,
        );

        storeOwn.clear('all');
        expect(storeOwn.load(now).entries['a/one']).toBeUndefined();
        expect(storeOwn.candidates(['a/one', 'a/two'], now)).toEqual([
          'a/one',
          'a/two',
        ]);
      });

      it('a foreign entry that actually changes after clear() is honored again', () => {
        const now = Date.parse('2026-11-01T00:00:00Z');
        const storeOwn = makeStore(() => now, 'proc-own');
        const storeForeign = makeStore(() => now, 'proc-foreign');
        storeOwn.record('a/one', 'retry in 60 min', 'retry', baseConfig, now);
        storeForeign.record(
          'a/one',
          'retry in 10 min',
          'retry',
          baseConfig,
          now,
        );

        storeOwn.clear('a/one');
        expect(storeOwn.load(now).entries['a/one']).toBeUndefined();

        // storeForeign re-records (a genuinely changed entry) beyond the 2s
        // foreign-read cache window: that specific entry's signature no
        // longer matches what clear() snapshotted, so it must be honored.
        const later = now + 3000;
        storeForeign.record(
          'a/one',
          'retry in 20 min',
          'retry',
          baseConfig,
          later,
        );
        expect(storeOwn.load(later).entries['a/one']).toBeDefined();
      });
    });

    describe('ingestion of the read-only factory state', () => {
      it('ignores a missing factory file', () => {
        const store = makeStore(() => Date.now());
        expect(store.load().entries).toEqual({});
      });

      it('computes until = probed_at + 30 min deterministically, and is visible without any explicit ingest call', () => {
        const now = Date.parse('2026-11-02T01:30:00Z');
        writeFileSync(
          factoryPath,
          JSON.stringify({
            provider: 'zai',
            exhausted_models: ['zai/glm-5.3', 'zai/glm-5.3-flash'],
            since: '2026-11-01T16:30:48Z',
            probed_at: '2026-11-02T01:13:03Z',
          }),
        );
        const store = makeStore(() => now);
        const state = store.load(now);
        expect(state.entries['zai/glm-5.3'].source).toBe('factory');
        expect(Date.parse(state.entries['zai/glm-5.3'].until)).toBe(
          Date.parse('2026-11-02T01:13:03Z') + 30 * 60 * 1000,
        );
        expect(state.entries['zai/glm-5.3-flash'].source).toBe('factory');
      });

      it('uses since when probed_at is absent, never "now"', () => {
        const now = Date.parse('2026-11-02T01:30:00Z');
        writeFileSync(
          factoryPath,
          JSON.stringify({
            provider: 'zai',
            exhausted_models: ['zai/glm-5.3'],
            since: '2026-11-01T16:30:48Z',
          }),
        );
        const store = makeStore(() => now);
        const entry = store.load(now).entries['zai/glm-5.3'];
        expect(Date.parse(entry.until)).toBe(
          Date.parse('2026-11-01T16:30:48Z') + getCooldownDefaultMs(),
        );
      });

      it('ignores a factory file with neither probed_at nor since (cannot be deterministic)', () => {
        writeFileSync(
          factoryPath,
          JSON.stringify({ exhausted_models: ['zai/glm-5.3'] }),
        );
        const store = makeStore(() => Date.parse('2026-11-02T01:30:00Z'));
        expect(store.load().entries).toEqual({});
      });

      it('ignores a factory file older than 24h', () => {
        const now = Date.parse('2026-11-03T02:00:00Z');
        writeFileSync(
          factoryPath,
          JSON.stringify({
            provider: 'zai',
            exhausted_models: ['zai/glm-5.3'],
            probed_at: '2026-11-01T01:13:03Z',
          }),
        );
        const store = makeStore(() => now);
        expect(store.load(now).entries).toEqual({});
      });

      // Regression: review 2 found unchanged factory input kept renewing the
      // cooldown (recorded_at/until advancing on every route()). Re-reading the
      // same file at different `now` values must yield an identical result.
      it('re-reading an unchanged factory file yields the same until every time (never "now"-based)', () => {
        writeFileSync(
          factoryPath,
          JSON.stringify({
            exhausted_models: ['zai/glm-5.3'],
            probed_at: '2026-11-02T01:13:03Z',
          }),
        );
        const store = makeStore(() => Date.parse('2026-11-02T01:30:00Z'));
        const first = store.load(Date.parse('2026-11-02T01:30:00Z')).entries[
          'zai/glm-5.3'
        ];
        const second = store.load(Date.parse('2026-11-02T01:40:00Z')).entries[
          'zai/glm-5.3'
        ];
        expect(second.until).toBe(first.until);
        expect(second.recorded_at).toBe(first.recorded_at);
      });
    });

    describe('only a reason category and the regex-matched reset text are persisted', () => {
      it('never persists raw error text: only the reason category and the regex-matched reset text land on disk', () => {
        const sentinel = 'acct_SuperSecretCustomerID123';
        const identifying = `429 {"account":"${sentinel}","message":"rate limit exceeded, retry in 10 min"}`;
        const store = makeStore(() => Date.now());
        const entry = store.record('a/one', identifying, 'retry', baseConfig);

        expect(entry.reason).toBe('rate_limit');
        expect(entry.raw_reset_text).toBe('retry in 10 min');

        const onDisk = readFileSync(join(dir, 'proc-a.json'), 'utf8');
        expect(onDisk).not.toContain(sentinel);
        expect(onDisk).not.toContain(identifying);
      });
    });

    describe('filtering only applies to resolvable, authenticated candidates', () => {
      it('never leaves zero usable candidates: a cooling usable model is kept when it is the only usable one', () => {
        const now = Date.parse('2026-11-01T00:00:00Z');
        const store = makeStore(() => now);
        store.record('x/good', 'retry in 5 min', 'retry', baseConfig, now);
        const isUsable = (ref: string) => ref === 'x/good';

        const result = store.candidates(
          ['x/good', 'y/unusable'],
          now,
          isUsable,
        );
        expect(result).toContain('x/good');
      });

      it('prefers the soonest-expiring USABLE candidate, never an unusable one with an earlier raw expiry', () => {
        const now = Date.parse('2026-11-01T00:00:00Z');
        const store = makeStore(() => now);
        store.record('b/usable', 'retry in 5 min', 'retry', baseConfig, now);
        store.record('c/unusable', 'retry in 1 min', 'retry', baseConfig, now);
        const isUsable = (ref: string) => ref !== 'c/unusable';

        const result = store.candidates(
          ['b/usable', 'c/unusable'],
          now,
          isUsable,
        );
        expect(result).toContain('b/usable');
      });
    });

    describe('explicit regressions for review 2 (no locks, no leases, no spinning)', () => {
      it('an unwritable directory never throws; record() still returns a usable entry and candidates() stays fast', () => {
        const blockerFile = join(dir, 'blocker');
        writeFileSync(blockerFile, 'not a directory');
        const unwritableDir = join(blockerFile, 'nested', 'cooldowns');
        const store = createCooldownStore(
          unwritableDir,
          () => Date.now(),
          join(dir, 'no-such-factory.json'),
          'proc-a',
        );
        const errorSpy = vi
          .spyOn(console, 'error')
          .mockImplementation(() => {});

        let entry: ReturnType<typeof store.record> | undefined;
        expect(() => {
          entry = store.record('a/one', '429 rate limit', 'retry', baseConfig);
        }).not.toThrow();
        expect(entry?.reason).toBe('rate_limit');

        const start = Date.now();
        expect(() => store.candidates(['a/one'])).not.toThrow();
        expect(Date.now() - start).toBeLessThan(100);
        expect(() => store.noteSuccess('a/one')).not.toThrow();
        expect(() => store.clear('a/one')).not.toThrow();
        expect(() => store.load()).not.toThrow();
        expect(() => store.formatStatus()).not.toThrow();
        expect(errorSpy).toHaveBeenCalled();
        errorSpy.mockRestore();
      });

      it('ENOTDIR (a path component is a file, not a directory) never throws and candidates() stays fast', () => {
        const blockerFile = join(dir, 'blocker-file');
        writeFileSync(blockerFile, 'i am a file');
        const enotdirPath = join(blockerFile, 'cooldowns');
        const store = createCooldownStore(
          enotdirPath,
          () => Date.now(),
          join(dir, 'no-such-factory.json'),
          'proc-a',
        );
        const errorSpy = vi
          .spyOn(console, 'error')
          .mockImplementation(() => {});

        const start = Date.now();
        expect(() => store.candidates(['a/one', 'a/two'])).not.toThrow();
        expect(Date.now() - start).toBeLessThan(100);
        expect(store.candidates(['a/one', 'a/two'])).toEqual([
          'a/one',
          'a/two',
        ]);
        errorSpy.mockRestore();
      });

      it('a directory sitting where the own cooldown file should be never throws', () => {
        mkdirSync(join(dir, 'proc-a.json'), { recursive: true });
        const store = createCooldownStore(
          dir,
          () => Date.now(),
          factoryPath,
          'proc-a',
        );
        const errorSpy = vi
          .spyOn(console, 'error')
          .mockImplementation(() => {});

        let entry: ReturnType<typeof store.record> | undefined;
        expect(() => {
          entry = store.record('a/one', '429 rate limit', 'retry', baseConfig);
        }).not.toThrow();
        expect(entry?.reason).toBe('rate_limit');
        errorSpy.mockRestore();
      });

      it('stale and corrupt files in the directory never throw and candidates() stays fast', () => {
        mkdirSync(dir, { recursive: true });
        writeFileSync(join(dir, 'proc-corrupt.json'), 'not json at all', {
          mode: 0o600,
        });
        const now = Date.now();
        const store = createCooldownStore(
          dir,
          () => now,
          factoryPath,
          'proc-a',
        );

        const start = Date.now();
        expect(() => store.candidates(['a/one'])).not.toThrow();
        expect(Date.now() - start).toBeLessThan(100);
      });

      // Regression for a real incident: node:fs mkdirSync(p, {recursive:true})
      // never returns when p is under /proc on Linux (non-recursive mkdir on
      // the same path returns ENOENT instantly). A cooldown dir resolving
      // under /proc must never hang record()/candidates() or the whole Pi
      // process hangs with it.
      it.skipIf(process.platform !== 'linux')(
        'a cooldown dir under /proc never hangs record() or candidates()',
        () => {
          const procDir = join('/proc', 'pmr-nonexistent', 'cdd');
          const store = createCooldownStore(
            procDir,
            () => Date.now(),
            join(dir, 'no-such-factory.json'),
            'proc-a',
          );
          const errorSpy = vi
            .spyOn(console, 'error')
            .mockImplementation(() => {});

          const start = Date.now();
          let entry: ReturnType<typeof store.record> | undefined;
          expect(() => {
            entry = store.record(
              'a/one',
              '429 rate limit',
              'retry',
              baseConfig,
            );
          }).not.toThrow();
          expect(entry?.reason).toBe('rate_limit');
          expect(() => store.candidates(['a/one'])).not.toThrow();
          expect(Date.now() - start).toBeLessThan(1000);

          errorSpy.mockRestore();
        },
        3000,
      );

      // Regression for review 3 finding 1: a FIFO (or any non-regular file --
      // socket, device) in the cooldown dir used to be opened with a plain
      // blocking readFileSync(path), which hangs forever reading a FIFO with
      // no writer. Every read now opens O_NONBLOCK|O_NOFOLLOW, fstats the fd,
      // and skips anything that isn't a plain file, so this must return
      // well within the subprocess's own timeout instead of hanging it.
      it.skipIf(process.platform !== 'linux')(
        'a FIFO left in the cooldown dir never blocks load()/candidates()',
        () => {
          const fifoDir = mkdtempSync(join(tmpdir(), 'pmr-cooldown-fifo-'));
          try {
            execFileSync('mkfifo', [join(fifoDir, 'foreign.json')]);
            const cooldownModulePath = new URL('./cooldown.ts', import.meta.url)
              .pathname;
            const scriptPath = join(fifoDir, 'probe.mts');
            writeFileSync(
              scriptPath,
              [
                `import { createCooldownStore } from ${JSON.stringify(cooldownModulePath)};`,
                `const store = createCooldownStore(${JSON.stringify(fifoDir)}, () => Date.now(), ${JSON.stringify(join(fifoDir, 'no-such-factory.json'))}, 'own-proc');`,
                `store.load();`,
                `store.candidates(['a/one', 'a/two']);`,
                `process.exit(0);`,
              ].join('\n'),
            );

            const result = spawnSync(
              process.execPath,
              ['--experimental-strip-types', scriptPath],
              { timeout: 3000 },
            );

            expect(result.signal).toBeNull();
            expect(result.status).toBe(0);
          } finally {
            rmSync(fifoDir, { recursive: true, force: true });
          }
        },
        5000,
      );

      it.skipIf(process.platform !== 'linux')(
        'a FIFO at the factory state path never blocks load()/candidates()',
        () => {
          const fifoDir = mkdtempSync(
            join(tmpdir(), 'pmr-cooldown-factory-fifo-'),
          );
          try {
            const factoryFifoPath = join(fifoDir, 'provider-switch.json');
            execFileSync('mkfifo', [factoryFifoPath]);
            const cooldownModulePath = new URL('./cooldown.ts', import.meta.url)
              .pathname;
            const scriptPath = join(fifoDir, 'probe.mts');
            writeFileSync(
              scriptPath,
              [
                `import { createCooldownStore } from ${JSON.stringify(cooldownModulePath)};`,
                `const store = createCooldownStore(${JSON.stringify(join(fifoDir, 'cooldowns'))}, () => Date.now(), ${JSON.stringify(factoryFifoPath)}, 'own-proc');`,
                `store.load();`,
                `store.candidates(['a/one', 'a/two']);`,
                `process.exit(0);`,
              ].join('\n'),
            );

            const result = spawnSync(
              process.execPath,
              ['--experimental-strip-types', scriptPath],
              { timeout: 3000 },
            );

            expect(result.signal).toBeNull();
            expect(result.status).toBe(0);
          } finally {
            rmSync(fifoDir, { recursive: true, force: true });
          }
        },
        5000,
      );

      // Regression for review 4 finding 1: persistOwn() used to name its temp
      // file `${ownFile}.${nowFn()}.tmp` -- fully predictable from outside the
      // process -- and write it with a plain writeFileSync(), which blocks
      // forever opening a FIFO for writing with no reader. A FIFO pre-created
      // at that exact old, guessable path must never hang record() now that
      // the temp name includes pid + random bytes and the open path never
      // opens a path it didn't just exclusively create.
      it.skipIf(process.platform !== 'linux')(
        'a FIFO pre-created at the old predictable temp-file pattern never blocks record()',
        () => {
          const fifoDir = mkdtempSync(
            join(tmpdir(), 'pmr-cooldown-predictable-fifo-'),
          );
          try {
            // Fix `now` so the OLD `${ownFile}.${nowFn()}.tmp` pattern is
            // fully deterministic and can be pre-created ahead of time.
            const fixedNow = 1700000000000;
            const ownFile = join(fifoDir, 'own-proc.json');
            const oldPredictableTemp = `${ownFile}.${fixedNow}.tmp`;
            execFileSync('mkfifo', [oldPredictableTemp]);

            const cooldownModulePath = new URL('./cooldown.ts', import.meta.url)
              .pathname;
            const scriptPath = join(fifoDir, 'probe.mts');
            writeFileSync(
              scriptPath,
              [
                `import { createCooldownStore } from ${JSON.stringify(cooldownModulePath)};`,
                `const store = createCooldownStore(${JSON.stringify(fifoDir)}, () => ${fixedNow}, ${JSON.stringify(join(fifoDir, 'no-such-factory.json'))}, 'own-proc');`,
                `store.record('a/one', '429 rate limit', 'retry', { profiles: {} });`,
                `process.exit(0);`,
              ].join('\n'),
            );

            const result = spawnSync(
              process.execPath,
              ['--experimental-strip-types', scriptPath],
              { timeout: 3000 },
            );

            expect(result.signal).toBeNull();
            expect(result.status).toBe(0);
          } finally {
            rmSync(fifoDir, { recursive: true, force: true });
          }
        },
        5000,
      );

      // Regression for review 4 finding 1 (O_EXCL half): the temp file open
      // must use O_EXCL so a colliding pre-existing path at the new pid+random
      // temp name is never opened/overwritten/followed -- it's refused, the
      // pre-existing file is left untouched, and the write fails safely
      // instead of silently corrupting or truncating something this process
      // never created.
      it('never opens or overwrites a pre-existing path at the temp-file name (O_EXCL)', async () => {
        const excDir = mkdtempSync(join(tmpdir(), 'pmr-cooldown-excl-'));
        try {
          vi.resetModules();
          vi.doMock('node:crypto', async (importOriginal) => {
            const actual = await importOriginal<typeof import('node:crypto')>();
            return {
              ...actual,
              randomBytes: (n: number) =>
                Buffer.from('cafebabe'.slice(0, n * 2), 'hex'),
            };
          });
          const { createCooldownStore: createCooldownStoreMocked } =
            await import('./cooldown');

          const ownFile = join(excDir, 'own-excl.json');
          const expectedTemp = `${ownFile}.${process.pid}.cafebabe.tmp`;
          const sentinel = 'pre-existing-content-must-survive';
          writeFileSync(expectedTemp, sentinel);

          const store = createCooldownStoreMocked(
            excDir,
            () => Date.now(),
            join(excDir, 'no-such-factory.json'),
            'own-excl',
          );
          const errorSpy = vi
            .spyOn(console, 'error')
            .mockImplementation(() => {});

          expect(() => {
            store.record('a/one', '429 rate limit', 'retry', baseConfig);
          }).not.toThrow();

          // The pre-existing file at the colliding path was never opened or
          // truncated by our write path (O_EXCL refused it), so its content
          // is byte-for-byte unchanged, and the real target file was never
          // produced since the write attempt failed.
          expect(readFileSync(expectedTemp, 'utf8')).toBe(sentinel);
          expect(existsSync(ownFile)).toBe(false);
          expect(errorSpy).toHaveBeenCalled();

          errorSpy.mockRestore();
        } finally {
          rmSync(excDir, { recursive: true, force: true });
          vi.doUnmock('node:crypto');
          vi.resetModules();
        }
      });
    });

    describe('explicit regressions for review 5 (malformed numeric timestamps never throw out of record())', () => {
      // Pre-fix, record() called buildEntry() -- and therefore iso(recordedAt)
      // -- OUTSIDE its own try/catch, so any recordedAt that made
      // `new Date(recordedAt).toISOString()` throw (non-finite, or outside
      // Date's +/-8.64e15ms representable range) escaped record() as an
      // uncaught RangeError instead of degrading to a safe default entry.
      it.each([
        ['1e20 (far outside Date range)', 1e20],
        ['NaN', NaN],
        ['Infinity', Infinity],
        ['-Infinity', -Infinity],
        ['beyond the +8.64e15ms Date limit', 8.65e15],
        ['beyond the -8.64e15ms Date limit', -8.65e15],
      ])(
        'record() never throws and still returns a usable entry when recordedAt is %s',
        (_label, recordedAt) => {
          const store = createCooldownStore(
            dir,
            () => Date.now(),
            factoryPath,
            'proc-a',
          );
          const errorSpy = vi
            .spyOn(console, 'error')
            .mockImplementation(() => {});

          let entry: ReturnType<typeof store.record> | undefined;
          expect(() => {
            entry = store.record(
              'a/one',
              '429 rate limit',
              'transcript',
              baseConfig,
              Date.now(),
              recordedAt,
            );
          }).not.toThrow();
          expect(entry).toBeDefined();
          // The fallback entry's own fields must themselves be valid ISO
          // strings -- not just "didn't throw" -- since callers (load(),
          // noteSuccess()) parse recorded_at/until right back out.
          expect(() =>
            new Date(entry!.recorded_at).toISOString(),
          ).not.toThrow();
          expect(() => new Date(entry!.until).toISOString()).not.toThrow();

          errorSpy.mockRestore();
        },
      );

      it('record() never throws when `now` itself is malformed', () => {
        const store = createCooldownStore(
          dir,
          () => Date.now(),
          factoryPath,
          'proc-a',
        );
        const errorSpy = vi
          .spyOn(console, 'error')
          .mockImplementation(() => {});

        let entry: ReturnType<typeof store.record> | undefined;
        expect(() => {
          entry = store.record(
            'a/one',
            '429 rate limit',
            'retry',
            baseConfig,
            NaN,
          );
        }).not.toThrow();
        expect(entry).toBeDefined();
        expect(() => new Date(entry!.recorded_at).toISOString()).not.toThrow();

        errorSpy.mockRestore();
      });

      it('load(), candidates(), noteSuccess(), clear(), and formatStatus() never throw given a foreign file with a malformed entry on disk', () => {
        mkdirSync(dir, { recursive: true });
        writeFileSync(
          join(dir, 'proc-b.json'),
          JSON.stringify({
            entries: {
              'a/one': {
                until: 'not-a-date',
                reason: 'unknown',
                raw_reset_text: '',
                assumed_tz: 'UTC',
                source: 'retry',
                recorded_at: 'also-not-a-date',
              },
            },
          }),
          { mode: 0o600 },
        );
        const store = createCooldownStore(
          dir,
          () => Date.now(),
          factoryPath,
          'proc-a',
        );

        expect(() => store.load()).not.toThrow();
        expect(() => store.candidates(['a/one'])).not.toThrow();
        expect(() => store.noteSuccess('a/one', Date.now())).not.toThrow();
        expect(() => store.clear('a/one')).not.toThrow();
        expect(() => store.formatStatus()).not.toThrow();
      });
    });
  });
});
