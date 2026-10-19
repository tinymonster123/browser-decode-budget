// Gate 3 driver: serve the two model roots, drive Chrome, collect paired A/B
// decode timings and the generated-text equivalence check.
//
//   node bench/browser/prepare.mjs   # once
//   node bench/browser/run.mjs
//
// Env:
//   REPEATS=4 MAX_NEW=64 HEADLESS=1 PORT=8123 MODEL_BASE=... MODEL_GATE2=...

import { execFileSync, spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import { cpus, loadavg, tmpdir } from 'node:os';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { makeLedger } from '../analysis/ledger.mjs';

const require = createRequire(
  process.env.PLAYWRIGHT_REQUIRE ?? '/Users/fenghanli/Projects/map-tile-lab/package.json',
);
const { chromium } = require('playwright');

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
  process.env.GATE3_OUT ??
  join(root, 'runs', `2026-10-19-gate3-${A_NAME}-vs-${B_NAME}.json`);

const load = loadavg().map((value) => Number(value.toFixed(2)));
console.log(`system load: ${load.join(' / ')} (${cpus().length} cores)`);

const server = spawn(process.execPath, [join(here, 'serve.mjs')], {
  env: { ...process.env, PORT: String(PORT) },
  stdio: ['ignore', 'pipe', 'pipe'],
});
server.stdout.on('data', (data) => process.stdout.write(`[server] ${data}`));
server.stderr.on('data', (data) => process.stderr.write(`[server] ${data}`));

async function waitForServer() {
  for (let i = 0; i < 100; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${PORT}/`);
      if (res.ok) return;
    } catch {}
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error('server did not start');
}

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
  await waitForServer();
  const userDataDir = mkdtempSync(join(tmpdir(), 'fused-onnx-bench-'));
  context = await chromium.launchPersistentContext(userDataDir, {
    channel: 'chrome',
    headless: HEADLESS,
    viewport: { width: 1100, height: 900 },
    args: [
      '--no-proxy-server',
      '--enable-unsafe-webgpu',
      '--disable-background-timer-throttling',
      '--disable-renderer-backgrounding',
    ],
  });

  const page = await context.newPage();
  page.setDefaultTimeout(0);
  await page.exposeFunction('__reportProgress', (message) => {
    console.log(`[page] ${message}`);
  });
  page.on('pageerror', (error) => console.log('[pageerror]', String(error).slice(0, 300)));
  page.on('console', (message) => {
    if (message.type() === 'error' || message.type() === 'warning') {
      console.log(`[console:${message.type()}]`, message.text().slice(0, 300));
    }
  });
  page.on('requestfailed', (request) => {
    const error = request.failure()?.errorText;
    // transformers.js aborts duplicate in-flight fetches after a cache hit.
    if (error !== 'net::ERR_ABORTED') console.log('[requestfailed]', request.url(), error);
  });
  page.on('response', (response) => {
    if (response.status() >= 400) console.log('[http]', response.status(), response.url());
  });
  page.on('crash', () => console.log('[page] CRASHED'));

  console.log('opening page…');
  await page.goto(`http://127.0.0.1:${PORT}/`, { waitUntil: 'load' });
  try {
    await page.waitForFunction(() => window.__ready === true, { timeout: 60000 });
  } catch (error) {
    console.log('[page] __ready never became true');
    await page.screenshot({ path: '/tmp/fused-onnx-bench-failed.png', fullPage: true });
    throw error;
  }
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

  const ledger = makeLedger({
    commit: execFileSync('git', ['rev-parse', '--short', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim(),
    loadavg: load,
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
  mkdirSync(join(root, 'runs'), { recursive: true });
  writeFileSync(OUT, JSON.stringify(ledger, null, 2));
  console.log(`\nwrote ${OUT}`);

  if (result.error) process.exitCode = 1;
} finally {
  await context?.close();
  server.kill('SIGTERM');
}
