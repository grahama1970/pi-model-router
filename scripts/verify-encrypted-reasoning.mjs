// Runtime verification for the encrypted-reasoning recovery
// (extensions/encrypted-reasoning.ts), through the real `pi` runtime with
// synthetic streaming providers. No credentials, no paid model calls: the
// providers are a local HTTP server (verify-encrypted-reasoning-server.mjs)
// registered through a throwaway agent dir's models.json.
//
//   node scripts/verify-encrypted-reasoning.mjs
//   PI_BIN=/path/to/pi node scripts/verify-encrypted-reasoning.mjs
//   PMR_EXTENSION=/path/to/older/extensions/index.ts node scripts/verify-encrypted-reasoning.mjs
//
// Each scenario gets its own agent dir, project dir, session dir, cooldown dir
// and provider process under the OS temp dir, then runs `pi` with the router
// extension loaded and asserts against BOTH the provider's recorded request
// bodies and the session transcript Pi wrote. Prints PASS/FAIL per assertion,
// exits non-zero on any failure, and kills every process it spawned.
import { spawn, spawnSync } from 'node:child_process';
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = dirname(HERE);
const SERVER = join(HERE, 'verify-encrypted-reasoning-server.mjs');
const EXTENSION =
  process.env.PMR_EXTENSION ?? join(REPO, 'extensions/index.ts');
const PI = process.env.PI_BIN ?? 'pi';

const results = [];
const preparedDirs = [];
const createdRoots = [];
const check = (scenario, name, ok, detail) => {
  results.push({ scenario, name, ok: Boolean(ok), detail });
  const status = ok ? 'PASS' : 'FAIL';
  process.stdout.write(
    `${status} [${scenario}] ${name}${ok || !detail ? '' : ` -- ${detail}`}\n`,
  );
};

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const writeModels = (agentDir) => {
  const model = (id, reasoning) => ({
    id,
    name: id,
    reasoning,
    input: ['text'],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 128000,
    maxTokens: 4096,
  });
  mkdirSync(agentDir, { recursive: true });
  writeFileSync(
    join(agentDir, 'models.json'),
    JSON.stringify({
      providers: {
        cxr: {
          baseUrl: '__CXR_URL__',
          api: 'openai-responses',
          apiKey: 'fake-key-cxr',
          models: [
            model('cxr-coder', true),
            model('cxr-coder-2', true),
            model('cxr-flaky', true),
            model('cxr-quota', false),
            model('cxr-broken', false),
            model('cxr-slow', false),
          ],
        },
        alt: {
          baseUrl: '__ALT_URL__',
          api: 'openai-responses',
          apiKey: 'fake-key-alt',
          models: [model('alt-model', false)],
        },
      },
    }),
  );
};

const startServer = async (logDir) => {
  mkdirSync(logDir, { recursive: true });
  const child = spawn(process.execPath, [SERVER], {
    env: { ...process.env, PORT: '0', LOG_DIR: logDir },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  const port = await new Promise((resolve, reject) => {
    let out = '';
    const timer = setTimeout(
      () => reject(new Error(`server did not start: ${out}`)),
      10000,
    );
    child.stdout.on('data', (chunk) => {
      out += chunk.toString();
      const match = out.match(/listening on (\d+)/);
      if (match) {
        clearTimeout(timer);
        resolve(Number(match[1]));
      }
    });
    child.on('exit', (code) => {
      clearTimeout(timer);
      reject(new Error(`server exited early rc=${code}: ${out}`));
    });
  });
  return { child, port };
};

const stopServer = (child) => {
  if (!child || child.exitCode !== null) return;
  child.kill('SIGKILL');
};

const readRequests = (logDir) => {
  try {
    return readFileSync(join(logDir, 'requests.jsonl'), 'utf8')
      .trim()
      .split('\n')
      .filter(Boolean)
      .map((line) => JSON.parse(line));
  } catch {
    return [];
  }
};

const readSession = (sessionDir) => {
  let files = [];
  try {
    files = readdirSync(sessionDir)
      .filter((name) => name.endsWith('.jsonl'))
      .sort();
  } catch {
    return { path: undefined, entries: [] };
  }
  const path =
    files.length > 0 ? join(sessionDir, files[files.length - 1]) : '';
  if (!path) return { path: undefined, entries: [] };
  const entries = readFileSync(path, 'utf8')
    .trim()
    .split('\n')
    .filter(Boolean)
    .map((line) => JSON.parse(line));
  return { path, entries };
};

const messagesOf = (entries) =>
  entries
    .filter((entry) => entry.type === 'message')
    .map((entry) => entry.message);

const customTypesOf = (entries) =>
  entries
    .filter(
      (entry) => entry.type === 'custom' || entry.type === 'custom_message',
    )
    .map((entry) => entry.customType);

const replayItems = (request) => {
  const body = JSON.parse(request.bodyText);
  const input = Array.isArray(body.input) ? body.input : [];
  return input.filter(
    (item) =>
      item && item.type === 'reasoning' && item.encrypted_content !== undefined,
  );
};

const profileConfig = (primary, fallbacks) => {
  const chain = fallbacks.map((model) => ({ model }));
  return {
    // A rule-matched prompt pins the tier deterministically, so a later run can
    // force a FRESH routing decision (not the sticky same-tier preference) and
    // thereby expose whether an exclusion is still active.
    rules: [{ matches: 'probe high', tier: 'high', reason: 'probe' }],
    profiles: {
      test: {
        high: { model: primary, fallbacks: chain },
        medium: { model: primary, fallbacks: chain },
        low: { model: primary, fallbacks: chain },
      },
    },
  };
};

// Synthetic, test-only extension used by the branch-summary scenarios. It
// performs a real `pi.navigateTree(..., { summarize: true })` on session start
// and supplies the branch summary from `session_before_tree`, so the run makes
// no extra model call. `keep-signed` branches at the signed assistant response
// (rejection error drops off the active path, replayable payload stays);
// `drop-signed` branches at the first user message (payload gone too).
const TEST_EXTENSION_SOURCE = `export default function (pi) {
  // pi.navigateTree is only available on the COMMAND context in this Pi
  // version, so the synthetic branch summary is triggered by the /pmr-branch
  // command (RPC prompt with a slash command executes it immediately).
  pi.on('session_before_tree', async () => ({
    summary: { summary: 'synthetic branch summary', details: {} },
  }));
  pi.registerCommand('pmr-branch', {
    description: 'synthetic branch summary for the router verification',
    handler: async (_args, ctx) => {
      const mode = process.env.PMR_TEST_BRANCH;
      const entries = ctx.sessionManager
        .getBranch()
        .filter((entry) => entry.type === 'message');
      const signed = (entry) =>
        entry.message &&
        entry.message.role === 'assistant' &&
        Array.isArray(entry.message.content) &&
        entry.message.content.some(
          (block) => block && block.type === 'thinking' && block.thinkingSignature,
        );
      const signedEntries = entries.filter(signed);
      const target =
        mode === 'keep-signed'
          ? signedEntries[0]
          : mode === 'keep-last-signed'
            ? signedEntries[signedEntries.length - 1]
            : entries.find((entry) => entry.message && entry.message.role === 'user');
      if (!target) return;
      await ctx.navigateTree(target.id, { summarize: true });
    },
  });
}
`;

const writeTestExtension = (context) => {
  const path = join(context.dir, 'branch-summary-extension.mjs');
  writeFileSync(path, TEST_EXTENSION_SOURCE);
  return path;
};

const piArgs = (context, extra) => [
  '--no-extensions',
  '--no-mcp',
  '--no-skills',
  '--no-prompt-templates',
  '-e',
  EXTENSION,
  '--approve',
  '--session-dir',
  context.sessionDir,
  '--model',
  'router/test',
  ...(context.testExtension ? ['-e', context.testExtension] : []),
  ...extra,
];

const piEnv = (context) => ({
  ...process.env,
  PI_CODING_AGENT_DIR: context.agentDir,
  PI_MODEL_ROUTER_COOLDOWNS: context.cooldownDir,
  ...(context.runEnv ?? {}),
});

const runPi = (context, args, timeout = 120000) =>
  spawnSync(PI, piArgs(context, args), {
    cwd: context.projectDir,
    env: piEnv(context),
    encoding: 'utf8',
    timeout,
  });

const prepareScenario = async (name, primary, fallbacks) => {
  let root = process.env.PMR_WORK_DIR;
  if (!root) {
    root = mkdtempSync(join(tmpdir(), 'pmr-cxr-'));
    createdRoots.push(root);
  }
  const dir = join(root, name);
  rmSync(dir, { recursive: true, force: true });
  preparedDirs.push(dir);
  const context = {
    name,
    dir,
    agentDir: join(dir, 'agent'),
    projectDir: join(dir, 'project'),
    sessionDir: join(dir, 'sessions'),
    cooldownDir: join(dir, 'cooldowns'),
    logDir: join(dir, 'logs'),
  };
  mkdirSync(join(context.projectDir, '.pi'), { recursive: true });
  mkdirSync(context.sessionDir, { recursive: true });
  mkdirSync(context.cooldownDir, { recursive: true });
  writeFileSync(join(context.projectDir, 'marker.txt'), 'marker contents\n');
  writeFileSync(
    join(context.projectDir, '.pi/model-router.json'),
    JSON.stringify(profileConfig(primary, fallbacks)),
  );

  const server = await startServer(context.logDir);
  context.server = server.child;
  writeModels(context.agentDir);
  const modelsPath = join(context.agentDir, 'models.json');
  writeFileSync(
    modelsPath,
    readFileSync(modelsPath, 'utf8')
      .replace('__CXR_URL__', `http://127.0.0.1:${server.port}/cxr`)
      .replace('__ALT_URL__', `http://127.0.0.1:${server.port}/alt`),
  );
  return context;
};

const scenarioRejectThenFallback = async () => {
  const name = 'reject-then-fallback';
  const context = await prepareScenario(name, 'cxr/cxr-coder', [
    'alt/alt-model',
  ]);
  try {
    const run = runPi(context, [
      '--tools',
      'bash',
      '-p',
      'read the marker file',
    ]);
    check(
      name,
      'pi run exits 0',
      run.status === 0,
      `status=${run.status} ${run.stderr}`,
    );
    const requests = readRequests(context.logDir);
    const sequence = requests.map((r) => `${r.url}:${r.model}`).join(' | ');
    check(
      name,
      'the rejected model answers first, rejects the replay, then another model answers',
      sequence ===
        '/cxr/responses:cxr-coder | /cxr/responses:cxr-coder | /alt/responses:alt-model',
      sequence,
    );

    const replay = requests[1] ? replayItems(requests[1]) : [];
    check(
      name,
      'the second request to the same model actually replays the encrypted reasoning item',
      replay.length === 1 && replay[0].encrypted_content === 'enc_cxr-coder_1',
      JSON.stringify(replay),
    );

    const replacement = requests[2];
    const replacementJson = replacement
      ? JSON.stringify(JSON.parse(replacement.bodyText))
      : '';
    check(
      name,
      'the replacement request omits the incompatible encrypted reasoning payload',
      Boolean(replacement) && replayItems(replacement).length === 0,
      `replayed items=${replacement ? replayItems(replacement).length : 'n/a'}`,
    );
    check(
      name,
      'the replacement request still carries the ordinary user message',
      replacementJson.includes('read the marker file'),
    );
    check(
      name,
      'the replacement request still carries the tool result',
      replacementJson.includes('function_call_output') &&
        replacementJson.includes('marker contents'),
    );

    const { entries } = readSession(context.sessionDir);
    const messages = messagesOf(entries);
    const errorMessage = messages.find(
      (message) =>
        message.role === 'assistant' && message.stopReason === 'error',
    );
    check(
      name,
      'the session records the encrypted-content rejection',
      Boolean(
        errorMessage &&
        errorMessage.provider === 'cxr' &&
        errorMessage.model === 'cxr-coder' &&
        /Invalid 'input\[\d+\]\.encrypted_content'/.test(
          errorMessage.errorMessage ?? '',
        ),
      ),
      JSON.stringify(errorMessage?.errorMessage ?? null),
    );
    check(
      name,
      'the session has exactly one encrypted-reasoning recovery entry',
      customTypesOf(entries).filter(
        (type) => type === 'router-encrypted-reasoning-recovery',
      ).length === 1,
      JSON.stringify(customTypesOf(entries)),
    );
    const lastAssistant = messages.filter((m) => m.role === 'assistant').at(-1);
    check(
      name,
      'the run finishes with an answer from the replacement model',
      lastAssistant?.provider === 'alt' &&
        lastAssistant?.model === 'alt-model' &&
        lastAssistant?.stopReason === 'stop',
      JSON.stringify({
        provider: lastAssistant?.provider,
        stopReason: lastAssistant?.stopReason,
      }),
    );
    check(
      name,
      'pi printed the replacement answer',
      run.stdout.includes('FALLBACK ANSWER'),
      run.stdout,
    );

    const before = readRequests(context.logDir).length;
    const second = runPi(context, [
      '--tools',
      'bash',
      '-c',
      '-p',
      'second question',
    ]);
    check(
      name,
      'the follow-up run in a new process exits 0',
      second.status === 0,
      `status=${second.status} ${second.stderr}`,
    );
    const after = readRequests(context.logDir).slice(before);
    check(
      name,
      'a later turn in a new process never re-selects the rejected model',
      after.length > 0 && after.every((r) => r.model === 'alt-model'),
      after.map((r) => r.model).join(','),
    );
    check(
      name,
      'the later turn is answered by the replacement model',
      second.stdout.includes('FALLBACK ANSWER'),
      second.stdout,
    );

    // Runtime shape of the persisted failure identity: the record carries the
    // failing API and the failed message's own event time, and binds only the
    // rejected payload. (The stale-error fabrication counterexample itself is
    // pinned at unit level -- see the retained 02-before-fix-unit.txt -- because
    // a linear synthetic history cannot both keep a stale rejection and drop the
    // payload it refers to while adding a newer same-model signature.)
    const { entries: finalEntries } = readSession(context.sessionDir);
    const rejectionRecords = finalEntries.filter(
      (entry) =>
        (entry.type === 'custom' || entry.type === 'custom_message') &&
        entry.customType === 'router-encrypted-rejection',
    );
    check(
      name,
      'the later turn adds no duplicate record for the already-recorded rejection',
      rejectionRecords.length === 1,
      `records=${rejectionRecords.length}`,
    );
    const record = rejectionRecords[0]?.data ?? {};
    check(
      name,
      'the record carries the failing API and event time and binds only the rejected payload',
      record.api === 'openai-responses' &&
        Array.isArray(record.fingerprints) &&
        record.fingerprints.length === 1 &&
        typeof record.failureAt === 'number',
      JSON.stringify({
        api: record.api,
        fingerprints: record.fingerprints?.length,
        failureAt: typeof record.failureAt,
      }),
    );
  } finally {
    stopServer(context.server);
  }
};

const scenarioChainedRecovery = async () => {
  const name = 'chained-recovery';
  const context = await prepareScenario(name, 'cxr/cxr-coder', [
    'cxr/cxr-coder-2',
    'alt/alt-model',
  ]);
  try {
    const run = runPi(context, [
      '--tools',
      'bash',
      '-p',
      'read the marker file',
    ]);
    const sequence = readRequests(context.logDir)
      .map((r) => r.model)
      .join(' | ');
    check(
      name,
      'two distinct models reject the replay and the third one answers',
      sequence ===
        'cxr-coder | cxr-coder | cxr-coder-2 | cxr-coder-2 | alt-model',
      sequence,
    );
    check(
      name,
      'the run exits 0 on the third model answer',
      run.status === 0 && run.stdout.includes('FALLBACK ANSWER'),
      `status=${run.status} stdout=${JSON.stringify(run.stdout)}`,
    );
    const { entries } = readSession(context.sessionDir);
    check(
      name,
      'each distinct failed model gets its own recovery continuation',
      customTypesOf(entries).filter(
        (type) => type === 'router-encrypted-reasoning-recovery',
      ).length === 2,
      JSON.stringify(customTypesOf(entries)),
    );
    // Notices are not observable in print mode (Pi's headless UI notify is a
    // no-op there); see the rpc-notice scenario below for the runtime proof.
  } finally {
    stopServer(context.server);
  }
};

// Runs one RPC-mode pi process through a script of steps. A step with
// `expectSettle` waits for the run's `agent_settled`; an extension-command step
// waits for its (handled) response instead, because commands do not start a
// run. All protocol records are collected for the caller's assertions.
const runPiRpc = (context, steps, timeoutMs = 120000) =>
  new Promise((resolve, reject) => {
    const child = spawn(
      PI,
      piArgs(context, ['--tools', 'bash', '--mode', 'rpc']),
      {
        cwd: context.projectDir,
        env: piEnv(context),
        stdio: ['pipe', 'pipe', 'pipe'],
      },
    );
    const records = [];
    let buffer = '';
    let stepIndex = 0;
    let settled = false;
    let stderr = '';
    let responseId = 0;
    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      reject(new Error(`rpc run did not settle: stderr=${stderr}`));
    }, timeoutMs);

    const sendNext = () => {
      if (stepIndex >= steps.length) return;
      const step = steps[stepIndex];
      stepIndex += 1;
      if (step.waitMs) {
        setTimeout(sendNext, step.waitMs);
        return;
      }
      responseId += 1;
      currentId = `step-${responseId}`;
      currentExpectsSettle = Boolean(step.expectSettle);
      child.stdin.write(
        `${JSON.stringify({ id: currentId, type: 'prompt', message: step.message })}\n`,
      );
    };
    let currentId;
    let currentExpectsSettle = false;

    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });
    child.stdout.on('data', (chunk) => {
      buffer += chunk.toString();
      let index = buffer.indexOf('\n');
      while (index >= 0) {
        const line = buffer.slice(0, index).trim();
        buffer = buffer.slice(index + 1);
        index = buffer.indexOf('\n');
        if (!line) continue;
        let record;
        try {
          record = JSON.parse(line);
        } catch {
          continue;
        }
        records.push(record);
        const isSettle = record.type === 'agent_settled';
        const isResponse =
          record.type === 'response' &&
          record.command === 'prompt' &&
          record.id === currentId;
        if (isSettle && currentExpectsSettle) {
          currentExpectsSettle = false;
          if (stepIndex >= steps.length) {
            settled = true;
            clearTimeout(timer);
            child.kill('SIGKILL');
            resolve({ records, childPid: child.pid, stderr });
            return;
          }
          sendNext();
        } else if (isResponse && !currentExpectsSettle) {
          if (stepIndex >= steps.length) {
            settled = true;
            clearTimeout(timer);
            child.kill('SIGKILL');
            resolve({ records, childPid: child.pid, stderr });
            return;
          }
          sendNext();
        }
      }
    });
    child.stdin.write(
      `${JSON.stringify({ id: 'warmup', type: 'get_state' })}\n`,
    );
    sendNext();
  });

const scenarioRpcNotice = async () => {
  const name = 'rpc-notice';
  const context = await prepareScenario(name, 'cxr/cxr-coder', [
    'cxr/cxr-coder-2',
    'alt/alt-model',
  ]);
  try {
    const { records, childPid } = await runPiRpc(context, [
      { message: 'read the marker file', expectSettle: true },
    ]);
    process.stdout.write(`     rpc pi pid ${childPid} reaped\n`);
    const notices = records
      .filter(
        (record) =>
          record.type === 'extension_ui_request' && record.method === 'notify',
      )
      .map((record) => String(record.message));
    const recoveryNotices = notices.filter((message) =>
      message.includes("rejected this conversation's encrypted reasoning"),
    );
    check(
      name,
      'the RPC client sees exactly one recovery notice for a two-model episode',
      recoveryNotices.length === 1,
      JSON.stringify(notices),
    );
    check(
      name,
      'the notice names the rejected model and its replacement',
      recoveryNotices.length === 1 &&
        recoveryNotices[0].includes('cxr/cxr-coder rejected') &&
        recoveryNotices[0].includes('continuing on cxr/cxr-coder-2'),
      JSON.stringify(recoveryNotices),
    );
    const { entries } = readSession(context.sessionDir);
    check(
      name,
      'the same episode really recovered twice',
      customTypesOf(entries).filter(
        (type) => type === 'router-encrypted-reasoning-recovery',
      ).length === 2,
      JSON.stringify(customTypesOf(entries)),
    );
  } finally {
    stopServer(context.server);
  }
};

const scenarioExhaustion = async () => {
  const name = 'exhaustion';
  const context = await prepareScenario(name, 'cxr/cxr-coder', [
    'cxr/cxr-coder-2',
  ]);
  try {
    const run = runPi(context, [
      '--tools',
      'bash',
      '-p',
      'read the marker file',
    ]);
    const requests = readRequests(context.logDir);
    const counts = {};
    for (const request of requests) {
      counts[request.model] = (counts[request.model] ?? 0) + 1;
    }
    check(
      name,
      'the run terminates instead of looping',
      run.status !== null,
      `status=${run.status} signal=${run.signal}`,
    );
    check(
      name,
      'each model is asked exactly twice (one answer, one rejected replay)',
      (counts['cxr-coder'] ?? 0) === 2 && (counts['cxr-coder-2'] ?? 0) === 2,
      JSON.stringify(counts),
    );
    const { entries } = readSession(context.sessionDir);
    check(
      name,
      'the single distinct recovery is attempted once, then the run stops',
      customTypesOf(entries).filter(
        (type) => type === 'router-encrypted-reasoning-recovery',
      ).length === 1,
      JSON.stringify(customTypesOf(entries)),
    );
    const messages = messagesOf(entries);
    const lastAssistant = messages.filter((m) => m.role === 'assistant').at(-1);
    check(
      name,
      'the exhausted run ends on the provider error, not on a fabricated answer',
      lastAssistant?.stopReason === 'error',
      JSON.stringify({
        provider: lastAssistant?.provider,
        stopReason: lastAssistant?.stopReason,
      }),
    );
  } finally {
    stopServer(context.server);
  }
};

const scenarioUnrelatedError = async () => {
  const name = 'unrelated-error';
  const context = await prepareScenario(name, 'cxr/cxr-broken', [
    'alt/alt-model',
  ]);
  try {
    const run = runPi(context, ['-p', 'read the marker file']);
    const requests = readRequests(context.logDir);
    check(
      name,
      'an unrelated provider rejection makes exactly one request',
      requests.length === 1 && requests[0].model === 'cxr-broken',
      requests.map((r) => r.model).join(','),
    );
    check(
      name,
      'the replacement model is never asked',
      requests.every((r) => r.model !== 'alt-model'),
    );
    const { entries } = readSession(context.sessionDir);
    check(
      name,
      'no recovery entry is appended',
      customTypesOf(entries).every(
        (type) => type !== 'router-encrypted-reasoning-recovery',
      ),
      JSON.stringify(customTypesOf(entries)),
    );
    const messages = messagesOf(entries);
    const lastAssistant = messages.filter((m) => m.role === 'assistant').at(-1);
    check(
      name,
      'the unrelated error stays visible',
      lastAssistant?.stopReason === 'error' &&
        /Unknown parameter/.test(lastAssistant?.errorMessage ?? ''),
      JSON.stringify(lastAssistant?.errorMessage ?? null),
    );
    check(
      name,
      'pi terminates on the unrelated error instead of continuing',
      run.status !== null,
      `status=${run.status} signal=${run.signal}`,
    );
  } finally {
    stopServer(context.server);
  }
};

const scenarioQuota = async () => {
  const name = 'quota-path';
  const context = await prepareScenario(name, 'cxr/cxr-quota', [
    'alt/alt-model',
  ]);
  try {
    const run = runPi(context, ['-p', 'read the marker file']);
    const requests = readRequests(context.logDir);
    check(
      name,
      'the quota model is tried once and the replacement answers',
      requests.length === 2 &&
        requests[0].model === 'cxr-quota' &&
        requests[1].model === 'alt-model',
      requests.map((r) => r.model).join(','),
    );
    const { entries } = readSession(context.sessionDir);
    check(
      name,
      'the existing auto-continue path still fires',
      customTypesOf(entries).filter((type) => type === 'router-auto-continue')
        .length === 1,
      JSON.stringify(customTypesOf(entries)),
    );
    check(
      name,
      'no encrypted-reasoning recovery entry appears',
      customTypesOf(entries).every(
        (type) => type !== 'router-encrypted-reasoning-recovery',
      ),
    );
    check(
      name,
      'the replacement answer reaches the user',
      run.stdout.includes('FALLBACK ANSWER'),
      run.stdout,
    );
    const messages = messagesOf(entries);
    const lastAssistant = messages.filter((m) => m.role === 'assistant').at(-1);
    check(
      name,
      'the run ends on the replacement model',
      lastAssistant?.provider === 'alt' && lastAssistant?.stopReason === 'stop',
      JSON.stringify({ provider: lastAssistant?.provider }),
    );
  } finally {
    stopServer(context.server);
  }
};

const scenarioAbort = async () => {
  const name = 'abort';
  const context = await prepareScenario(name, 'cxr/cxr-slow', [
    'alt/alt-model',
  ]);
  let child;
  try {
    child = spawn(PI, piArgs(context, ['-p', 'read the marker file']), {
      cwd: context.projectDir,
      env: piEnv(context),
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    process.stdout.write(`     spawned pi (abort scenario) pid=${child.pid}\n`);
    await sleep(4000);
    child.kill('SIGINT');
    const exit = await new Promise((resolve) => {
      const timer = setTimeout(() => resolve({ timedOut: true }), 30000);
      child.on('exit', (code, signal) => {
        clearTimeout(timer);
        resolve({ code, signal, timedOut: false });
      });
    });
    check(
      name,
      'an aborted run terminates instead of requesting a continuation',
      !exit.timedOut,
      JSON.stringify(exit),
    );
    await sleep(500);
    const requests = readRequests(context.logDir);
    check(
      name,
      'the replacement model is never asked after an abort',
      requests.every((r) => r.model !== 'alt-model'),
      requests.map((r) => r.model).join(','),
    );
    const { entries } = readSession(context.sessionDir);
    check(
      name,
      'no recovery entry is appended for an abort',
      customTypesOf(entries).every(
        (type) => type !== 'router-encrypted-reasoning-recovery',
      ),
      JSON.stringify(customTypesOf(entries)),
    );
    const messages = messagesOf(entries);
    check(
      name,
      'no assistant error was recorded (the run ended on the user abort)',
      messages.every((m) => m.role !== 'assistant' || m.stopReason !== 'error'),
      JSON.stringify(
        messages.filter((m) => m.role === 'assistant').map((m) => m.stopReason),
      ),
    );
  } finally {
    if (child && child.exitCode === null) child.kill('SIGKILL');
    if (child) process.stdout.write(`     pi pid ${child.pid} reaped\n`);
    stopServer(context.server);
  }
};

const scenarioMixedQuotaEncrypted = async () => {
  const name = 'mixed-quota-encrypted';
  const context = await prepareScenario(name, 'cxr/cxr-coder', [
    'cxr/cxr-quota',
    'alt/alt-model',
  ]);
  try {
    const run = runPi(context, [
      '--tools',
      'bash',
      '-p',
      'read the marker file',
    ]);
    const requests = readRequests(context.logDir);
    const sequence = requests.map((r) => r.model).join(' | ');
    check(
      name,
      'encrypted rejection recovers into a quota-limited model, which recovers into the healthy one',
      sequence === 'cxr-coder | cxr-coder | cxr-quota | alt-model',
      sequence,
    );
    check(
      name,
      'the mixed episode terminates on the healthy model',
      run.status === 0 && run.stdout.includes('FALLBACK ANSWER'),
      `status=${run.status} stdout=${JSON.stringify(run.stdout)}`,
    );
    const { entries } = readSession(context.sessionDir);
    const types = customTypesOf(entries);
    check(
      name,
      'both recovery kinds appear exactly once',
      types.filter((t) => t === 'router-encrypted-reasoning-recovery')
        .length === 1 &&
        types.filter((t) => t === 'router-auto-continue').length === 1,
      JSON.stringify(types),
    );
    check(
      name,
      'the encrypted rejection is persisted as durable session metadata',
      types.filter((t) => t === 'router-encrypted-rejection').length === 1,
      JSON.stringify(types),
    );
  } finally {
    stopServer(context.server);
  }
};

const scenarioAllExcludedLaterTurn = async () => {
  const name = 'all-excluded-later-turn';
  const context = await prepareScenario(name, 'cxr/cxr-coder', [
    'cxr/cxr-coder-2',
  ]);
  try {
    const first = runPi(context, [
      '--tools',
      'bash',
      '-p',
      'read the marker file',
    ]);
    check(
      name,
      'the first run stops on the provider error once every candidate rejected',
      first.status !== null,
      `status=${first.status}`,
    );
    const before = readRequests(context.logDir).length;

    const second = runPi(context, ['--tools', 'bash', '-c', '-p', 'try again']);
    const after = readRequests(context.logDir).slice(before);
    check(
      name,
      'the later turn makes no model request at all (no replay of a rejected model)',
      after.length === 0,
      after.map((r) => `${r.model}`).join(','),
    );
    const text = `${second.stdout ?? ''}${second.stderr ?? ''}`;
    check(
      name,
      'the later turn stops with the clear routing message',
      text.includes(
        'stopping instead of replaying rejected encrypted reasoning',
      ),
      JSON.stringify(text.slice(0, 300)),
    );
  } finally {
    stopServer(context.server);
  }
};

const scenarioRetryableWrappedRejection = async () => {
  const name = 'retryable-wrapped-rejection';
  const context = await prepareScenario(name, 'cxr/cxr-flaky', [
    'alt/alt-model',
  ]);
  try {
    const run = runPi(context, [
      '--tools',
      'bash',
      '-p',
      'read the marker file',
    ]);
    const requests = readRequests(context.logDir);
    const sequence = requests.map((r) => r.model).join(' | ');
    check(
      name,
      'a retryable 500 wrapper around the encrypted rejection still switches models on the retry',
      sequence === 'cxr-flaky | cxr-flaky | alt-model',
      sequence,
    );
    check(
      name,
      'the run finishes on the replacement model',
      run.status === 0 && run.stdout.includes('FALLBACK ANSWER'),
      `status=${run.status} stdout=${JSON.stringify(run.stdout)}`,
    );
    const { entries } = readSession(context.sessionDir);
    check(
      name,
      'the retry-path rejection is persisted as durable metadata',
      customTypesOf(entries).filter((t) => t === 'router-encrypted-rejection')
        .length === 1,
      JSON.stringify(customTypesOf(entries)),
    );
  } finally {
    stopServer(context.server);
  }
};

const scenarioBranchSummaryKeepsSigned = async () => {
  const name = 'branch-summary-keeps-signed';
  const context = await prepareScenario(name, 'cxr/cxr-coder', [
    'alt/alt-model',
  ]);
  context.testExtension = writeTestExtension(context);
  context.runEnv = { PMR_TEST_BRANCH: 'keep-signed' };
  try {
    // Turn 1: signed answer -> rejected replay -> recovery to the replacement.
    // Then a real branch summary at the signed assistant (the rejection error
    // leaves the active path, the replayable payload stays), then a fresh
    // rule-matched prompt that forces a new routing decision.
    const { records, childPid, stderr } = await runPiRpc(context, [
      { message: 'read the marker file', expectSettle: true },
      { message: '/pmr-branch' },
      { message: 'probe high: next task', expectSettle: true },
    ]);
    process.stdout.write(`     rpc pi pid ${childPid} reaped\n`);
    check(
      name,
      'pi reported no extension error while branching',
      !stderr.includes('pmr-branch'),
      JSON.stringify(stderr.slice(0, 400)),
    );

    const requests = readRequests(context.logDir);
    const afterNavigation = requests.filter((request) =>
      request.bodyText.includes('probe high'),
    );
    check(
      name,
      'the branch summary really replaced the abandoned path (the recovery note is gone from the request)',
      afterNavigation.length > 0 &&
        !afterNavigation.some((request) =>
          request.bodyText.includes(
            'rejected the encrypted reasoning this conversation',
          ),
        ),
      afterNavigation.length > 0
        ? JSON.stringify(Object.keys(JSON.parse(afterNavigation[0].bodyText)))
        : 'no post-navigation request',
    );
    check(
      name,
      'after the summary keeps the signed response, the rejected model is still excluded',
      afterNavigation.length > 0 &&
        afterNavigation.every((request) => request.model === 'alt-model'),
      afterNavigation.map((request) => request.model).join(','),
    );
    const { entries } = readSession(context.sessionDir);
    check(
      name,
      'the durable rejection record survives the branch summary',
      customTypesOf(entries).filter((t) => t === 'router-encrypted-rejection')
        .length === 1,
      JSON.stringify(customTypesOf(entries)),
    );
    check(
      name,
      'the post-summary turn completes on the replacement model',
      records.some(
        (record) =>
          record.type === 'message_end' || record.type === 'agent_settled',
      ),
      JSON.stringify(records.slice(-1)),
    );
  } finally {
    stopServer(context.server);
  }
};

const scenarioBranchSummaryDropsSigned = async () => {
  const name = 'branch-summary-drops-signed';
  const context = await prepareScenario(name, 'cxr/cxr-coder', [
    'alt/alt-model',
  ]);
  context.testExtension = writeTestExtension(context);
  context.runEnv = { PMR_TEST_BRANCH: 'drop-signed' };
  try {
    // 1. first turn: payload X rejected, recovery to the replacement.
    // 2. branch summary at the first user message: X is gone, so the record
    //    retires and the model is eligible again.
    // 3. next turn: the model answers with a NEW payload Y (the synthetic
    //    server issues a distinct payload per answered turn); the following
    //    request replays Y and is rejected. That is a FRESH failure of a model
    //    that still has the retired record for X -- it must switch models and
    //    persist a second record, not be suppressed by the old one.
    const { records, childPid, stderr } = await runPiRpc(context, [
      { message: 'read the marker file', expectSettle: true },
      { message: '/pmr-branch' },
      { message: 'probe high: next task', expectSettle: true },
    ]);
    process.stdout.write(`     rpc pi pid ${childPid} reaped\n`);
    check(
      name,
      'pi reported no extension error while branching',
      !stderr.includes('pmr-branch'),
      JSON.stringify(stderr.slice(0, 400)),
    );

    const requests = readRequests(context.logDir);
    const afterNavigation = requests.filter((request) =>
      request.bodyText.includes('probe high'),
    );
    check(
      name,
      'the branch summary really removed the signed history from the request',
      afterNavigation.length > 0 &&
        replayItems(afterNavigation[0]).length === 0,
      afterNavigation.length > 0
        ? `replayed items=${replayItems(afterNavigation[0]).length}`
        : 'no post-navigation request',
    );
    check(
      name,
      'once the implicated signed history is gone, the exclusion retires and the model is used again',
      afterNavigation.length > 0 && afterNavigation[0].model === 'cxr-coder',
      afterNavigation.map((request) => request.model).join(','),
    );
    // The re-rejection: the model answered with a NEW payload and then rejected
    // its replay. The fresh failure must switch models instead of re-selecting
    // it (the old record is bound to the removed payload and must not suppress).
    check(
      name,
      'a fresh rejection of the NEW payload switches models instead of replaying the old record',
      afterNavigation.some((request) => request.model === 'alt-model'),
      afterNavigation.map((request) => request.model).join(','),
    );
    const postNavigationTail = afterNavigation
      .slice(
        afterNavigation.findIndex((request) => request.model === 'alt-model'),
      )
      .map((request) => request.model);
    check(
      name,
      'the post-rejection turn ends on the replacement model, not on the rejected one',
      postNavigationTail.length > 0 &&
        postNavigationTail.every((model) => model === 'alt-model'),
      postNavigationTail.join(','),
    );
    const { entries } = readSession(context.sessionDir);
    const rejectionRecords = entries.filter(
      (entry) =>
        (entry.type === 'custom' || entry.type === 'custom_message') &&
        entry.customType === 'router-encrypted-rejection',
    );
    check(
      name,
      'the NEW payload gets its own durable record (payload identity, not model identity)',
      rejectionRecords.length === 2,
      `records=${rejectionRecords.length}`,
    );
    check(
      name,
      'pi completed the post-summary turn',
      records.some((record) => record.type === 'agent_settled'),
      JSON.stringify(records.slice(-1)),
    );

    // 4. restart: the second record keeps the model excluded while payload Y is
    //    still in the conversation.
    const before = readRequests(context.logDir).length;
    context.runEnv = {};
    const restarted = runPi(context, [
      '--tools',
      'bash',
      '-c',
      '-p',
      'probe high: after restart',
    ]);
    const afterRestart = readRequests(context.logDir).slice(before);
    check(
      name,
      'after a restart the model stays excluded while its new payload remains',
      afterRestart.length > 0 &&
        afterRestart.every((request) => request.model === 'alt-model'),
      afterRestart.map((request) => request.model).join(','),
    );
    check(
      name,
      'the restarted turn is answered by the replacement model',
      restarted.status === 0 && restarted.stdout.includes('FALLBACK ANSWER'),
      `status=${restarted.status} stdout=${JSON.stringify(restarted.stdout.slice(0, 200))}`,
    );
  } finally {
    stopServer(context.server);
  }
};

const scenarioExhaustionThenBranch = async () => {
  const name = 'exhaustion-then-branch';
  const context = await prepareScenario(name, 'cxr/cxr-coder', [
    'cxr/cxr-coder-2',
  ]);
  context.testExtension = writeTestExtension(context);
  context.runEnv = { PMR_TEST_BRANCH: 'keep-last-signed' };
  try {
    // A rejects X, the recovery moves to B, B rejects Y, and there is no
    // remaining candidate: the run stops. The rejection of B must still be
    // persisted BEFORE that no-replacement return.
    const { childPid, stderr } = await runPiRpc(context, [
      { message: 'read the marker file', expectSettle: true },
      { message: '/pmr-branch' },
    ]);
    process.stdout.write(`     rpc pi pid ${childPid} reaped\n`);
    check(
      name,
      'pi reported no extension error while branching',
      !stderr.includes('pmr-branch'),
      JSON.stringify(stderr.slice(0, 400)),
    );

    const firstRunRequests = readRequests(context.logDir);
    check(
      name,
      'the first run exhausts both candidates',
      firstRunRequests.map((request) => request.model).join(' | ') ===
        'cxr-coder | cxr-coder | cxr-coder-2 | cxr-coder-2',
      firstRunRequests.map((request) => request.model).join(' | '),
    );

    const { entries } = readSession(context.sessionDir);
    const rejectionRecords = entries.filter(
      (entry) =>
        (entry.type === 'custom' || entry.type === 'custom_message') &&
        entry.customType === 'router-encrypted-rejection',
    );
    check(
      name,
      'the final exhausted rejection was persisted even though no continuation was requested',
      rejectionRecords.length === 2,
      `records=${rejectionRecords.length}`,
    );
    // Over-binding regression (P1): at the second model's failure the session
    // still carries the FIRST model's signed payload, but that payload is never
    // replayed to the second model (Pi drops foreign thinking). Each record must
    // therefore bind exactly its own model's payload, not both.
    const fingerprintCounts = rejectionRecords.map(
      (entry) => (entry.data || {}).fingerprints?.length,
    );
    check(
      name,
      "each record binds only its own model's payload (the other model's signature is not bound)",
      fingerprintCounts.length === 2 &&
        fingerprintCounts.every((count) => count === 1),
      `fingerprints per record=${JSON.stringify(fingerprintCounts)}`,
    );

    // The branch summary now keeps B's successful signed response and drops B's
    // rejection error (A's error stays in the branch).
    const before = readRequests(context.logDir).length;
    context.runEnv = {};
    const restarted = runPi(context, [
      '--tools',
      'bash',
      '-c',
      '-p',
      'probe high: after branch',
    ]);
    const afterBranch = readRequests(context.logDir).slice(before);
    check(
      name,
      'after the branch + restart B stays excluded by its signed payload (no replay request)',
      afterBranch.length === 0,
      afterBranch.map((request) => request.model).join(','),
    );
    const text = `${restarted.stdout ?? ''}${restarted.stderr ?? ''}`;
    check(
      name,
      'the restarted turn stops with the clear routing message',
      text.includes(
        'stopping instead of replaying rejected encrypted reasoning',
      ),
      JSON.stringify(text.slice(0, 300)),
    );
  } finally {
    stopServer(context.server);
  }
};

const main = async () => {
  process.stdout.write(
    `pi binary: ${PI}\nextension: ${EXTENSION}\nwork dir: ${process.env.PMR_WORK_DIR ?? 'fresh temp dir per scenario'}\n`,
  );
  const scenarios = [
    scenarioRejectThenFallback,
    scenarioChainedRecovery,
    scenarioRpcNotice,
    scenarioMixedQuotaEncrypted,
    scenarioAllExcludedLaterTurn,
    scenarioRetryableWrappedRejection,
    scenarioBranchSummaryKeepsSigned,
    scenarioBranchSummaryDropsSigned,
    scenarioExhaustionThenBranch,
    scenarioExhaustion,
    scenarioUnrelatedError,
    scenarioQuota,
    scenarioAbort,
  ];
  const only = process.env.PMR_ONLY;
  for (const scenario of scenarios) {
    const normalize = (value) => value.toLowerCase().replace(/[^a-z0-9]/g, '');
    if (only && !normalize(scenario.name).includes(normalize(only))) {
      continue;
    }
    try {
      await scenario();
    } catch (error) {
      check('harness', `scenario ${scenario.name} threw`, false, String(error));
    }
  }
  // Leave the session transcripts and recorded provider requests behind only
  // when the caller chose the work dir; otherwise the temp dirs go away.
  if (!process.env.PMR_WORK_DIR) {
    for (const dir of preparedDirs)
      rmSync(dir, { recursive: true, force: true });
    for (const root of createdRoots)
      rmSync(root, { recursive: true, force: true });
  } else {
    process.stdout.write(`kept work dirs under ${process.env.PMR_WORK_DIR}\n`);
  }
  const failed = results.filter((result) => !result.ok);
  process.stdout.write(
    `\n${results.length - failed.length}/${results.length} assertions passed\n`,
  );
  if (failed.length > 0) {
    for (const failure of failed) {
      process.stdout.write(
        `FAILED [${failure.scenario}] ${failure.name}: ${failure.detail ?? ''}\n`,
      );
    }
    process.exitCode = 1;
  }
};

await main();
