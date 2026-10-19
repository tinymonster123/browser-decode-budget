// Experiment 3: measure what one trivial WebGPU dispatch / pass / submit
// costs on this machine, so the dispatch deltas from experiment 2 can be
// converted into milliseconds.
//
//   node bench/browser/gpu-microbench.mjs
//
// Env: REPEATS=5 HEADLESS=1 PORT=8125 OUT=...

import { execFileSync } from 'node:child_process';
import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import { cpus, loadavg, tmpdir } from 'node:os';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(
  process.env.PLAYWRIGHT_REQUIRE ?? '/Users/fenghanli/Projects/map-tile-lab/package.json',
);
const { chromium } = require('playwright');

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '../..');
const PORT = Number(process.env.PORT ?? 8125);
const REPEATS = Math.max(1, Number(process.env.REPEATS ?? 5));
const HEADLESS = process.env.HEADLESS === '1';
const OUT =
  process.env.OUT ?? join(root, 'runs', '2026-10-19-exp3-dispatch-cost.json');

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

function fmt(value, digits = 2) {
  if (value == null || !Number.isFinite(value)) return '—';
  return value.toFixed(digits);
}

try {
  await waitForServer();

  const userDataDir = mkdtempSync(join(tmpdir(), 'fused-onnx-microbench-'));
  const context = await chromium.launchPersistentContext(userDataDir, {
    channel: 'chrome',
    headless: HEADLESS,
    viewport: { width: 900, height: 700 },
    args: [
      '--no-proxy-server',
      '--enable-unsafe-webgpu',
      '--disable-background-timer-throttling',
      '--disable-renderer-backgrounding',
    ],
  });

  let result;
  try {
    const page = await context.newPage();
    page.setDefaultTimeout(0);
    await page.exposeFunction('__reportProgress', (message) => console.log(`[page] ${message}`));
    page.on('pageerror', (error) => console.log('[pageerror]', String(error).slice(0, 300)));
    page.on('console', (message) => {
      if (message.type() === 'error' || message.type() === 'warning') {
        console.log(`[console:${message.type()}]`, message.text().slice(0, 300));
      }
    });

    await page.goto(`http://127.0.0.1:${PORT}/gpu-microbench.html`, { waitUntil: 'load' });
    await page.waitForFunction(() => window.__ready === true, { timeout: 30000 });
    result = await page.evaluate(
      (options) => window.__microbench(options),
      { counts: [0, 100, 500, 1000, 2000], repeats: REPEATS },
    );
  } finally {
    await context.close();
  }

  if (result.error) {
    console.error(`error: ${result.error}`);
    if (result.stack) console.error(result.stack);
  }

  console.log(`adapter: ${JSON.stringify(result.info)}`);

  const byMode = new Map();
  for (const row of result.rows) {
    const list = byMode.get(row.mode) ?? [];
    list.push(row);
    byMode.set(row.mode, list);
  }

  console.log('\nwall = encode + submit + GPU completion');
  console.log(
    [
      'mode'.padEnd(10),
      'count'.padStart(7),
      'wall ms'.padStart(10),
      'cpu ms'.padStart(10),
      'gpu ms'.padStart(10),
      'µs/dispatch'.padStart(13),
    ].join(' '),
  );

  const summary = {};
  for (const [mode, rows] of byMode) {
    rows.sort((a, b) => a.count - b.count);
    const zero = rows.find((row) => row.count === 0);
    summary[mode] = [];
    for (const row of rows) {
      const base = zero?.wallMs ?? 0;
      const perDispatch =
        row.count > 0 ? ((row.wallMs - base) * 1000) / row.count : null;
      summary[mode].push({
        count: row.count,
        wallMs: row.wallMs,
        cpuMs: row.cpuMs,
        gpuMs: row.gpuMs,
        usPerDispatch: perDispatch,
        samples: row.samples,
      });
      console.log(
        [
          mode.padEnd(10),
          String(row.count).padStart(7),
          fmt(row.wallMs).padStart(10),
          fmt(row.cpuMs).padStart(10),
          fmt(row.gpuMs).padStart(10),
          fmt(perDispatch, 3).padStart(13),
        ].join(' '),
      );
    }
  }

  console.log(
    `\ncreateBindGroup: ${fmt(result.bindGroup?.usPerCall, 3)} µs/call ` +
      `(${result.bindGroup?.count} calls × ${REPEATS} repeats)`,
  );

  // Slopes over the largest counts are the most robust estimate.
  console.log('\nper-dispatch cost at the two largest counts');
  for (const [mode, rows] of Object.entries(summary)) {
    const tail = rows.filter((row) => row.count > 0).slice(-2);
    const parts = tail.map(
      (row) => `${row.count}: ${fmt(row.usPerDispatch, 3)} µs`,
    );
    console.log(`  ${mode.padEnd(10)} ${parts.join('   ')}`);
  }

  const ledger = {
    timestamp: new Date().toISOString(),
    commit: execFileSync('git', ['rev-parse', '--short', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim(),
    loadavg: load,
    config: { repeats: REPEATS, port: PORT, headless: HEADLESS },
    result,
    summary,
  };
  mkdirSync(join(root, 'runs'), { recursive: true });
  writeFileSync(OUT, JSON.stringify(ledger, null, 2));
  console.log(`\nwrote ${OUT}`);

  if (result.error) process.exitCode = 1;
} finally {
  server.kill('SIGTERM');
}
