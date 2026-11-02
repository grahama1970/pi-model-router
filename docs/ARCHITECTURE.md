# Architecture: Pi Model Router Extension

The `pi-model-router` is an extension-first model router for the `pi` coding agent. It registers a custom logical provider (`router`) that exposes "profiles" as models (e.g., `router/balanced`). For every turn, the router intelligently selects an underlying concrete model based on task complexity, conversation phase, and user-defined rules.

## Core Concepts

### 1. Profiles & Tiers

The router is organized into **Profiles** (e.g., `balanced`, `cheap`, `deep`). Each profile defines up to three **Tiers** (at least one required):

- **High**: Reserved for architecture, design, complex debugging, and planning. Uses high-reasoning models.
- **Medium**: The default for standard implementation, multi-file edits, and focused fixes.
- **Low**: Used for summaries, changelogs, formatting, and simple read-only lookups.

### 2. Virtual Model Implementation

The extension uses `pi.registerVirtualModel` (and `pi.unregisterVirtualModel`) to register each configured profile under the `router` provider (`router/<profile>`). Pi cleanly separates **Selection** (`model_change` entries keep `router/<profile>`) from **Dispatch** (each assistant message records the physical `provider`, `model`, and `thinkingLevel`). This provides:

- Native footer display (`profile • level → physical-model • level`) and per-physical-model cost breakdown in `/session`.
- Safe conversation replay across different physical models and providers.
- Native pre-request auto-compaction when routing to a physical model whose context window is smaller than the current transcript.

## Routing Decision Flow

For every request sent to a `router/*` virtual model, `route(request, ctx)` inspects `request.reason`:

- **`direct` requests** (e.g., compaction summaries): Routed directly to the `medium` (or pinned) tier without mutating session routing state or running the classifier.
- **`continuation` requests** (tool-result continuations within a turn): Preserves the previous routed model for the turn to maintain provider prompt caches and thinking signatures (unless a new pin, budget cap, or image attachment requires switching).
- **`retry` requests** (automatic retry after a failed request): Advances to the next available model in the tier's `fallbacks` chain after `request.failed.model`.
- **`user` requests** (new user prompt, steer, or follow-up): Executes the full tier selection pipeline:
  1. **Budget Check**: If `maxSessionBudget` is configured and session spend exceeds it, downgrades `high` tier requests to `medium`.
  2. **Manual Pin**: Uses the tier pinned via `/router pin` or `/router fix`.
  3. **Custom Rules**: Matches keyword-based rules defined in config against the user prompt.
  4. **Classifier (Optional)**: When `classifierModel` is configured, prefers Pi's native classifier models (`ctx.modelRegistry.classify()` with `"auto"`, TypeSafe Jev, Cloudflare Clef / Clef Omni, OpenAI GPT-6 Luna, or `llama.cpp` classifiers — with image filtering and text-only fallback) and falls back to chat LLM classification (`ctx.modelRegistry.streamSimple()`).
  5. **Heuristics (Fallback)**: Uses fast local heuristics (keyword/length/tool-use analysis) if the classifier is off or fails.
  6. **Biased Stickiness**: The `phaseBias` setting modulates heuristic thresholds and low-confidence classifier probabilities to keep the router in a consistent phase.
  7. **First-Turn Auto-Pin (Optional)**: When `autoPinFirstTurnDecision` is enabled, the router automatically pins the active profile to the tier selected on the first turn of a conversation. Users can run `/router pin auto` at any time to unpin and return to dynamic per-turn routing.

## Cooldowns

The operator goal this subsystem serves: a Pi session must never become unresponsive because of a provider rate limit, and the router itself must never be the cause. Every cooldown code path is advisory and best-effort: any exception or bad input logs once and falls back to routing as if no cooldown existed. There are no locks, no leases, and no waits anywhere in `extensions/cooldown.ts` — nothing in this subsystem can spin or block a turn.

### Per-process files, no contention

Each Pi process writes only its own cooldown file, `~/.local/state/pi-model-router/cooldowns/<pid>-<startTimeMs>.json` (override the directory with `PI_MODEL_ROUTER_COOLDOWNS`; dir `0700`, file `0600`, atomic temp-file-then-rename, write failures logged once and otherwise ignored). There is nothing to lock: a process only ever appends to or deletes from its own file in memory, then best-effort persists it.

The temp file's name is `<ownFile>.<pid>.<8 random hex bytes>.tmp`, never a predictable name derived from the clock: it's opened with `openSync(tmp, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_NONBLOCK, 0o600)`, written via the returned fd, and the fd is always closed in a `finally` before `renameSync`. `O_EXCL` means the open fails instead of silently writing through or hanging on any pre-existing path at that name — a FIFO, symlink, or attacker-planted file is refused, never opened; a predictable, clock-derived name previously let a FIFO pre-created at the exact path the process was about to write block `persistOwn()` (and therefore `record()`) forever. On any failure the temp path is unlinked best-effort — but only if this call actually created it (tracked separately from merely computing the path), so a failed `O_EXCL` open on someone else's pre-existing file never deletes that file.

Every read (`load()`, `candidates()`, `formatStatus()`) merges its own in-memory state with every other process's on-disk file in the directory, taking `max(until)` per `provider/model`, skipping unreadable or malformed files, and ignoring (then best-effort deleting) files whose mtime is older than 24h. Other processes' files and the factory state (below) are read at most once per `route()` call and cached for 2 seconds, keyed off the caller-supplied `now` rather than a separate wall clock, so repeated calls within one turn don't re-hit disk.

Because the cooldown directory (and the factory state path) can contain anything another process, a misconfigured mount, or an attacker leaves there, every file's _content_ is read via `openSync(path, O_RDONLY | O_NONBLOCK | O_NOFOLLOW)` followed by `fstatSync` on the open fd, keeping only entries where `stat.isFile()` is true and `stat.size <= 64 KiB`; the fd is always closed in a `finally`. `O_NONBLOCK` makes both the `open()` and the subsequent read on that fd return immediately instead of blocking forever, so a FIFO, socket, or device file dropped into the directory (e.g. `foreign.json` as a named pipe with no writer) is skipped, never hangs the read. `O_NOFOLLOW` rejects symlinks via the open call itself rather than trusting a separate `lstat`. Anything that isn't eligible is treated exactly like an unreadable or malformed file: skipped, never thrown.

Foreign files and the factory file are read **unmerged** into a flat list of `{ ref, entry }` pairs (one entry per file per model), not collapsed to a single `max(until)` winner per model, before the per-process merge and suppression pass below — this is what lets `clear()` track every source's entry for a model independently (see "Success and `/router cooldown clear`").

### Entry shape: category and bounded reset text only

A `CooldownEntry` is `{ until, reason, raw_reset_text, assumed_tz, source, recorded_at }`. `reason` is one of a fixed category set from `categorizeReason()`: `rate_limit | quota | usage_limit | too_many_requests | unknown` — never raw provider error text. `raw_reset_text` is limited to the substring the reset-time regex itself matched (e.g. `"retry in 10 min"`), truncated to 40 characters, never the surrounding message — so account identifiers or other message detail in a provider error can never land on disk. `source` is `retry | transcript | factory`.

### Recording: retry failures and new transcript errors only

`extensions/provider.ts` records a cooldown in two cases: (a) `request.reason === 'retry'` with a limit-classified `request.failed.message`, and (b) a transcript scan of `request.messages` for assistant messages with `stopReason: "error"` whose `timestamp` is newer than this process's start time (`state.processStartTime`, captured once at extension load) and not already in an in-memory `Set` of processed message timestamps (`state.processedTranscriptFailures`). Both are plain session state, never persisted — a historical failure from before this process started, or one already turned into a cooldown this run, is never reprocessed, including after `/router cooldown clear` or a restart.

`CooldownStore.record()` takes `now` and an optional `recordedAt` (defaulting to `now`) as separate parameters: `now` drives expiry math only — relative `"retry in N min"` parsing and the default cooldown length, both inherently about real wall-clock time — while `recordedAt` becomes the entry's `recorded_at`. Route-time (`'retry'`) failures have no earlier event time than the moment they're observed, so the provider passes only `now` and `recordedAt` defaults to it. Transcript-sourced failures pass the failed message's own `timestamp` as `recordedAt`, separate from the scan's `cooldownNow`: this is what lets `noteSuccess` below compare a success against when the failure actually happened rather than when the transcript scan that found it happened to run.

### Reset parsing, with a hard safety clamp

`parseResetTime()` prefers explicit offsets, then relative retry durations (`"retry in N s/m/h"`), then a naive local time in the provider's timezone (`RouterConfig.providerTimezones`, defaulting `zai`/`moonshot`/`kimi-coding`/`deepseek`/`qwen`/`alibaba`/`minimax` to `Asia/Shanghai` and everything else to UTC; the earlier interpretation wins on an ambiguous DST transition). Every computed date — regardless of which branch produced it — is validated before use: anything non-finite, unconstructible, already past, or more than 7 days out falls back to `now + 30 min` with `assumed_tz: "untrusted"`. This is what stops a malformed message like `"retry in 9999999999999999 hours"` from overflowing `Date`'s representable range and throwing out of `toISOString()`.

### Selection: usable candidates only, no leases

`extensions/provider.ts` passes `isUsableCandidate` (resolvable in the model registry _and_ authenticated) into `cooldownStore.candidates(modelsToTry, now, isUsableCandidate)`. Unusable refs always pass through untouched, and "keep the soonest-expiring candidate when everything is cooling" only ever picks among usable refs — filtering can never leave a tier with zero usable candidates. Once a cooldown's `until` has passed, every caller sees the model as available immediately; there is no probe lease or claim, so more than one process may retry a just-expired model at the same time — a new limit error simply re-records (one 429 per process per expiry is an accepted cost of removing the lease).

### Success and `/router cooldown clear`

When a model answers without error, `noteSuccess(modelRef, successTimestamp?)` removes it from the calling process's own file only — and only if that success is actually newer than the failure that armed it. `extensions/provider.ts` scans the whole transcript on every turn, so an old success message sitting earlier in the same transcript as a newer quota error must not erase the cooldown the newer error just recorded: `noteSuccess` compares `successTimestamp` against the model's own `recorded_at` and is a no-op unless the success is strictly newer, and the provider-side scan additionally ignores any success message timestamped at or before `state.processStartTime`, matching the rule already applied to transcript-sourced failures.

Because `recorded_at` for a transcript failure is the failure message's own timestamp (not scan time), this comparison is correct even when a single scan observes both a failure and a success for the same model in one pass: `[failure t1, success t2]` with `t1 < t2` clears the cooldown (the success is genuinely newer than the failure), while `[success t1, failure t2]` with `t1 < t2` leaves it cooling (the success predates the failure's own event time), regardless of how far in the past both timestamps are relative to the scan itself.

`clear(ref)` / `clear('all')` removes the entry from this process's own file and, for each cleared model, snapshots and suppresses the signature (`until`/`recorded_at`/`source`) of **every** foreign/factory entry for that model individually — not just whichever one currently wins the `max(until)` merge. This matters because a smaller foreign entry can be masked by a larger own (or other foreign) entry in the merged view; suppressing only the merged winner would let that smaller, genuinely unchanged foreign entry resurface the instant the larger entry is cleared or expires, making `clear()` look like a no-op. Each suppressed signature is independent: a ref is hidden from this process's merged view only while a foreign entry with that exact signature is still present, and lifts automatically (for that entry only) once the owning process writes a new one.

### Factory state: read-only, computed deterministically

`readFactoryEntries()` reads `/mnt/storage12tb/agent-factory/state/provider-switch.json` (path overridable for tests) but never writes to it. `until = probed_at (or since) + 30 min`, computed from the file's own observation timestamp — never from "now" — so re-reading an unchanged file always yields the same `until` and never renews a cooldown. A missing, malformed, or (`now - stamp > 24h`) stale file is ignored. Factory entries are computed live on every merged read rather than persisted to any process's file.

### `/router cooldowns` and `/router cooldown clear`

`/router cooldowns` and `/router status` show local and UTC reset times, source, raw reset text, and assumed timezone via `formatStatus()`. `/router cooldown clear <provider/model|all>` clears as described above.

See `extensions/cooldown.test.ts` for the regression coverage of the failure modes above: an unwritable directory, `ENOTDIR`, a directory sitting where the own file should be, stale/corrupt files, and a FIFO left in the cooldown directory or at the factory state path (`candidates()` must return within its timeout in every case, never throw, and never block); a FIFO pre-created at the old predictable temp-file name (`record()` must still return well within the subprocess timeout); a pre-existing file colliding with the new pid+random temp name (the write path must never open it, via `O_EXCL`, and must leave it byte-for-byte unchanged); and per-entry `clear()` suppression of a foreign entry masked by a larger own entry. See `extensions/provider.test.ts` (`cooldowns (integration)`) for the success-ordering regressions: an older success already in the transcript must never erase a cooldown recorded from a newer failure, a success genuinely newer than the recorded failure does clear it, and a single scan that observes both a failure and a success for the same model clears or keeps the cooldown based on which message's own timestamp is newer, not on scan order or scan time.

## Module Architecture

The extension is modularized for maintainability:

- `extensions/index.ts`: Orchestrator. Manages state, hooks into `pi` events, and wires modules together.
- `extensions/provider.ts`: Implements the `router` provider, cooldown detection, and delegation/retry loop.
- `extensions/cooldown.ts`: Shared cooldown persistence, reset parsing, factory-state ingestion, and candidate filtering.
- `extensions/routing.ts`: Core decision logic, heuristics, and the LLM classifier.
- `extensions/config.ts`: Loads, merges, and normalizes the JSON configuration.
- `extensions/commands.ts`: Registers all `/router` subcommands and their autocompletions.
- `extensions/ui.ts`: Manages the status line and the optional state widget.
- `extensions/state.ts`: Handles session-persisted state and snapshots.
- `extensions/types.ts`: Centralized interface and type definitions.

## Thinking Resolution

For the model actually selected, thinking resolves in `extensions/provider.ts:276-304` (`resolveDelegatedThinking`): an explicit `/router thinking` override wins first, then the selected fallback's optional `thinking`, then the tier's `thinking` (the primary model always uses the tier value). The result is clamped against the selected model's capabilities with `clampThinkingLevel` in `extensions/config.ts:752-774` before dispatch and status/debug reporting.

## State & Persistence

The router state is persisted using `pi.appendEntry` with a custom type `router-state`. This allows the router to:

- Restore the active profile and pins across agent relaunches.
- Maintain independent pins and state for different conversation branches.
- Track accumulated session costs safely.

## Reliability: Fallback Chains

Each tier in a profile can define an optional `fallbacks` list. If the primary model fails (e.g., due to rate limits or provider downtime), the router automatically retries the next model in the chain before surfacing an error to the user.
