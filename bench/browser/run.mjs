// Gate 3 driver: serve the two model roots, drive Chrome, collect paired A/B
// decode timings and the generated-text equivalence check.
//
//   node bench/browser/prepare.mjs   # once
//   node bench/browser/run.mjs
//
// Env:
//   REPEATS=4 MAX_NEW=64 HEADLESS=1 PORT=8123 MODEL_BASE=... MODEL_GATE2=...

import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { artifact } from './artifacts.mjs';
import {
  announceLoad,
  openPage,
  startServer,
  stampLedger,
  writeLedger,
} from './runtime.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '../..');
const PORT = Number(process.env.PORT ?? 8123);
const REPEATS = Math.max(1, Number(process.env.REPEATS ?? 4));
const MAX_NEW = Math.max(1, Number(process.env.MAX_NEW ?? 64));
const HEADLESS = process.env.HEADLESS === '1';
const A_MODEL = process.env.A_MODEL ?? 'base';
const A_NAME = process.env.A_NAME ?? A_MODEL;
const B_MODEL = process.env.B_MODEL ?? 'gate2';
const B_NAME = process.env.B_NAME ?? B_MODEL;
const OUT =
  process.env.OUT ??
  process.env.GATE3_OUT ??
  join(root, 'runs', `2026-10-19-gate3-${A_NAME}-vs-${B_NAME}.json`);

// Names must exist in the registry: the server only routes registered roots.
artifact(A_MODEL);
artifact(B_MODEL);

const load = announceLoad();

const server = startServer(PORT);

function median(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function summarize(rounds) {
  const byName = new Map();
  for (const run of rounds) {
    if (!run.tokens || run.firstTokenMs == null || run.lastTokenMs == null) continue;
    const decodeMs = run.lastTokenMs - run.firstTokenMs;
    const msPerToken = run.tokens > 1 ? decodeMs / (run.tokens - 1) : null;
    const entry = byName.get(run.name) ?? { runs: [], msPerToken: [], totalMs: [], tokens: [] };
    entry.runs.push(run);
    if (msPerToken != null) entry.msPerToken.push(msPerToken);
    entry.totalMs.push(run.ms);
    entry.tokens.push(run.tokens);
    byName.set(run.name, entry);
  }
  const summary = {};
  for (const [name, entry] of byName) {
    summary[name] = {
      runs: entry.runs.length,
      medianMsPerToken: median(entry.msPerToken),
      medianTotalMs: median(entry.totalMs),
      medianTokens: median(entry.tokens),
      samplesMsPerToken: entry.msPerToken.map((value) => Number(value.toFixed(3))),
    };
  }
  return summary;
}

let context;
try {
  await server.ready;

  console.log('opening page…');
  const opened = await openPage({
    url: `http://127.0.0.1:${PORT}/`,
    headless: HEADLESS,
    label: 'fused-onnx-bench',
    prefix: 'page',
    onResponse: true,
    onReadyTimeout: async (page) => {
      console.log('[page] __ready never became true');
      await page.screenshot({ path: '/tmp/fused-onnx-bench-failed.png', fullPage: true });
    },
  });
  context = opened.context;
  const page = opened.page;
  console.log('page ready');

  console.log(
    `launching: repeats=${REPEATS} max_new_tokens=${MAX_NEW} headless=${HEADLESS} ` +
      `a=${A_MODEL}/${A_NAME} b=${B_MODEL}/${B_NAME}`,
  );
  const result = await page.evaluate(
    (options) => window.__bench(options),
    {
      repeats: REPEATS,
      maxNewTokens: MAX_NEW,
      aModel: A_MODEL,
      aName: A_NAME,
      bModel: B_MODEL,
      bName: B_NAME,
    },
  );

  if (result.error) {
    console.error(`page error: ${result.error}`);
    if (result.stack) console.error(result.stack);
  }

  const summary = summarize(result.rounds ?? []);
  const a = summary[A_NAME]?.medianMsPerToken;
  const b = summary[B_NAME]?.medianMsPerToken;
  const delta = a && b ? (100 * (b - a)) / a : null;

  console.log('\nvariant   median ms/token   median total ms   tokens');
  for (const name of [A_NAME, B_NAME]) {
    const row = summary[name];
    if (!row) continue;
    console.log(
      `${name.padEnd(9)} ${String(row.medianMsPerToken?.toFixed(2)).padStart(15)} ` +
        `${String(row.medianTotalMs).padStart(16)} ${String(row.medianTokens).padStart(8)}`,
    );
  }
  if (delta != null) console.log(`${B_NAME} vs ${A_NAME}: ${delta >= 0 ? '+' : ''}${delta.toFixed(1)}%`);

  if (result.equivalence) {
    console.log(`\nequivalence: exact=${result.equivalence.exact}`);
    if (!result.equivalence.exact) {
      console.log(`  ${A_NAME}: ${JSON.stringify(result.equivalence.aText.slice(0, 400))}`);
      console.log(`  ${B_NAME}: ${JSON.stringify(result.equivalence.bText.slice(0, 400))}`);
    }
  }

  const ledger = stampLedger({
    root,
    load,
    config: {
      repeats: REPEATS,
      maxNewTokens: MAX_NEW,
      port: PORT,
      headless: HEADLESS,
      aModel: A_MODEL,
      aName: A_NAME,
      bModel: B_MODEL,
      bName: B_NAME,
    },
    summary,
    result,
  });
  writeLedger(OUT, ledger);
  console.log(`\nwrote ${OUT}`);

  if (result.error) process.exitCode = 1;
} finally {
  await context?.close();
  server.stop();
}
