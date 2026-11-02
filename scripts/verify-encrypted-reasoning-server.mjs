// Synthetic streaming provider for the pi-model-router encrypted-reasoning
// runtime verification (scripts/verify-encrypted-reasoning.mjs starts and stops
// it). No credentials, no network: two provider base paths on 127.0.0.1
// speaking the OpenAI Responses SSE protocol Pi's openai-responses transport
// understands.
//
// Model behaviours (all on provider `cxr` unless noted):
//   cxr-coder     reasoning item with encrypted_content + a bash tool call;
//                 rejects any request whose input replays encrypted_content
//                 with a non-retryable 400
//   cxr-coder-2   same, with its own encrypted payload id
//   cxr-flaky     same first answer, but rejects a replayed payload with a
//                 RETRYABLE 500 wrapper around the same encrypted-content
//                 rejection (exercises Pi's retry -> request.failed path)
//   cxr-quota     always 429 insufficient_quota
//   cxr-broken    always 400 unrelated validation error
//   cxr-slow      streams response.created, then never completes (abort test)
//   alt/alt-model (provider `alt`) plain text answer, no reasoning item
//
// Every request body is appended to $LOG_DIR/requests.jsonl for read-back.
import { createServer } from 'node:http';
import { appendFileSync } from 'node:fs';
import { join } from 'node:path';

const PORT = Number(process.env.PORT);
const LOG_DIR = process.env.LOG_DIR;

const log = (entry) => {
  appendFileSync(join(LOG_DIR, 'requests.jsonl'), `${JSON.stringify(entry)}\n`);
};

const jsonError = (res, status, body) => {
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(JSON.stringify(body));
};

const sse = (res, events) => {
  res.writeHead(200, {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
  });
  for (const event of events) {
    res.write(`event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`);
  }
  res.end();
};

const usage = {
  input_tokens: 20,
  output_tokens: 5,
  input_tokens_details: { cached_tokens: 0 },
  output_tokens_details: { reasoning_tokens: 0 },
  total_tokens: 25,
};

// Distinct payload per answered turn, so a model that answered twice (e.g.
// after its first payload was removed from the conversation) provably rejects a
// DIFFERENT payload the second time. `enc_<model>_<n>` / `rs_<model>_<n>`.
const payloadCounters = new Map();
const nextPayloadId = (model) => {
  const n = (payloadCounters.get(model) ?? 0) + 1;
  payloadCounters.set(model, n);
  return `${model}_${n}`;
};

const reasoningItem = (model, payloadId) => ({
  id: `rs_${payloadId}`,
  type: 'reasoning',
  status: 'completed',
  summary: [{ type: 'summary_text', text: `thought for ${model}` }],
  encrypted_content: `enc_${payloadId}`,
});

const messageItem = (text, id = 'msg_1') => ({
  id,
  type: 'message',
  role: 'assistant',
  status: 'completed',
  content: [{ type: 'output_text', text, annotations: [] }],
});

const toolCallItem = (id = 'call_1') => ({
  id: 'fc_1',
  type: 'function_call',
  call_id: id,
  name: 'bash',
  arguments: JSON.stringify({ command: 'cat marker.txt' }),
  status: 'completed',
});

const completed = (output, id = 'resp_1') => ({
  type: 'response.completed',
  response: { id, status: 'completed', output, usage },
});

const textAnswer = (res, text) =>
  sse(res, [
    {
      type: 'response.created',
      response: { id: 'resp_t', status: 'in_progress' },
    },
    {
      type: 'response.output_item.done',
      output_index: 0,
      item: messageItem(text),
    },
    completed([messageItem(text)]),
  ]);

const coderAnswer = (res, model) => {
  const payloadId = nextPayloadId(model);
  const reasoning = reasoningItem(model, payloadId);
  return sse(res, [
    {
      type: 'response.created',
      response: { id: 'resp_c', status: 'in_progress' },
    },
    {
      type: 'response.output_item.done',
      output_index: 0,
      item: reasoning,
    },
    {
      type: 'response.output_item.done',
      output_index: 1,
      item: toolCallItem(),
    },
    completed([reasoning, toolCallItem()]),
  ]);
};

const inputText = (input) => JSON.stringify(input ?? null);

const replaysEncryptedContent = (input) =>
  Array.isArray(input) && JSON.stringify(input).includes('"encrypted_content"');

const handle = (req, res, body) => {
  const url = req.url ?? '/';
  const payload = body ? JSON.parse(body) : {};
  const model = payload.model ?? 'unknown';
  const entry = {
    at: Date.now(),
    url,
    model,
    inputText: inputText(payload.input),
    bodyText: body ?? '',
  };
  log(entry);

  if (url === '/alt/responses') {
    // The replacement carries its OWN signed reasoning payload (a distinct
    // provider id, so Pi never replays the rejected model's payload here). This
    // is what a later turn's stale rejection + newer signature scenario needs.
    // Replays of the replacement's own payload are answered normally.
    const payloadId = nextPayloadId(model);
    const reasoning = reasoningItem(model, payloadId);
    const text = messageItem('FALLBACK ANSWER');
    return sse(res, [
      {
        type: 'response.created',
        response: { id: 'resp_alt', status: 'in_progress' },
      },
      { type: 'response.output_item.done', output_index: 0, item: reasoning },
      { type: 'response.output_item.done', output_index: 1, item: text },
      completed([reasoning, text]),
    ]);
  }

  if (model === 'cxr-quota') {
    return jsonError(res, 429, {
      error: { message: 'insufficient_quota', code: 'insufficient_quota' },
    });
  }
  if (model === 'cxr-flaky') {
    if (replaysEncryptedContent(payload.input)) {
      return jsonError(res, 500, {
        error: {
          message:
            "Invalid 'input[3].encrypted_content': Encrypted content could not be verified.",
          type: 'server_error',
          code: 'invalid_encrypted_content',
        },
      });
    }
    return coderAnswer(res, model);
  }
  if (model === 'cxr-broken') {
    return jsonError(res, 400, {
      error: { message: "Unknown parameter: 'foo'", code: 'unknown_parameter' },
    });
  }
  if (model === 'cxr-slow') {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
    });
    res.write(
      `event: response.created\ndata: ${JSON.stringify({ type: 'response.created', response: { id: 'resp_slow', status: 'in_progress' } })}\n\n`,
    );
    return undefined;
  }
  if (model === 'cxr-coder' || model === 'cxr-coder-2') {
    if (replaysEncryptedContent(payload.input)) {
      return jsonError(res, 400, {
        error: {
          message:
            "Invalid 'input[3].encrypted_content': Encrypted content could not be verified.",
          type: 'invalid_request_error',
          param: 'input[3].encrypted_content',
          code: 'invalid_encrypted_content',
        },
      });
    }
    return coderAnswer(res, model);
  }

  return jsonError(res, 500, {
    error: { message: `unhandled model ${model}` },
  });
};

const server = createServer((req, res) => {
  let body = '';
  req.on('data', (chunk) => {
    body += chunk;
  });
  req.on('end', () => {
    try {
      handle(req, res, body);
    } catch (error) {
      jsonError(res, 500, { error: { message: String(error) } });
    }
  });
});

server.listen(PORT, '127.0.0.1', () => {
  process.stdout.write(`fake provider listening on ${server.address().port}\n`);
});
