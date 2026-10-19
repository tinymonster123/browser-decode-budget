// The shared page session: transformers.js environment, one model load, and
// one deterministic greedy decode.
//
// Pages keep their instrumentation (counters, timestamps, paired rounds) and
// call this module for the parts that must not drift between pages. The
// browser harness stays untested by design; this module is untested for the
// same reason.

import { env, LogLevel, pipeline, TextStreamer } from '/vendor/transformers.web.js';

export const SYSTEM =
  'You are a professional translator. Translate the text the user sends into ' +
  'Chinese (Simplified). Reply with only the translation — no explanation, no notes, ' +
  'no original text, no quotes.';

export const DEFAULT_TEXT =
  'The end of Dennard scaling meant that shrinking a transistor no longer reduced its ' +
  'power density. Chipmakers kept the clock flat and spent the transistor budget on ' +
  'parallelism instead. Four cores became sixteen, then sixty-four, and the software ' +
  'world slowly discovered that it did not know how to use them.';

export function configureEnv() {
  env.allowLocalModels = false;
  env.allowRemoteModels = true;
  env.remoteHost = location.origin + '/';
  env.remotePathTemplate = '{model}/';
  env.useBrowserCache = false;
  env.useWasmCache = true;
  env.logLevel = LogLevel.ERROR;
  env.backends.onnx.wasm.wasmPaths = {
    mjs: '/vendor/ort/ort-wasm-simd-threaded.asyncify.mjs',
    wasm: '/vendor/ort/ort-wasm-simd-threaded.asyncify.wasm',
  };
}

export function status(text) {
  document.querySelector('#status').textContent = text;
  console.log(text);
  window.__reportProgress?.(text);
}

export async function loadModel(id) {
  status(`loading ${id}…`);
  const t0 = performance.now();
  const pipe = await pipeline('text-generation', id, {
    device: 'webgpu',
    dtype: 'q4f16',
    progress_callback: (p) => {
      if (p.status === 'progress' && p.file) {
        const pct = p.total ? Math.round((100 * p.loaded) / p.total) : 0;
        document.querySelector('#status').textContent =
          `loading ${id}: ${p.file} ${pct}%`;
      }
    },
  });
  return { pipe, loadMs: Math.round(performance.now() - t0) };
}

// `onToken` receives the 1-based token index; `startedAt` lets callers time
// the first and last token without opening a second clock.
export async function generate(pipe, maxNewTokens, text, onToken) {
  const messages = [
    { role: 'system', content: SYSTEM },
    { role: 'user', content: text },
  ];
  let tokens = 0;
  const streamer = new TextStreamer(pipe.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    token_callback_function: (generated) => {
      for (let i = 0; i < generated.length; i++) {
        tokens += 1;
        onToken?.(tokens);
      }
    },
    callback_function: () => {},
  });
  const startedAt = performance.now();
  const out = await pipe(messages, {
    max_new_tokens: maxNewTokens,
    do_sample: false,
    return_full_text: false,
    streamer,
  });
  return { out, totalMs: performance.now() - startedAt, tokens, startedAt };
}

export function generatedText(out) {
  const value = out?.[0]?.generated_text;
  if (Array.isArray(value)) {
    const last = value.at(-1);
    return typeof last === 'string' ? last : (last?.content ?? JSON.stringify(last));
  }
  return value ?? '';
}
