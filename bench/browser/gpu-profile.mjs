// Gate 5: per-compute-pass GPU timestamps, to find where decode time actually
// goes. Reuses the same single-model setup as count.mjs.
//
//   node bench/browser/gpu-profile.mjs
//
// Env: MODELS=base,gate1,gate2 STEPS=4 HEADLESS=1 PORT=8126 OUT=...

import { execFileSync, spawn } from 'node:child_process';
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
const PORT = Number(process.env.PORT ?? 8126);
const STEPS = Math.max(1, Number(process.env.STEPS ?? 4));
const HEADLESS = process.env.HEADLESS === '1';
const MODELS = (process.env.MODELS ?? 'base,gate1,gate2').split(',').filter(Boolean);
const OUT = process.env.OUT ?? join(root, 'runs', '2026-10-19-exp4-pass-profile.json');

const NODES = { base: 1289, gate1: 1233, gate2: 1078 };

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

async function measure(model) {
  const userDataDir = mkdtempSync(join(tmpdir(), `fused-onnx-profile-${model}-`));
  const context = await chromium.launchPersistentContext(userDataDir, {
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

  try {
    const page = await context.newPage();
    page.setDefaultTimeout(0);
    await page.exposeFunction('__reportProgress', (message) => {
      console.log(`[${model}] ${message}`);
    });
    page.on('pageerror', (error) => console.log('[pageerror]', String(error).slice(0, 300)));
    page.on('console', (message) => {
      if (message.type() === 'error' || message.type() === 'warning') {
        console.log(`[console:${message.type()}]`, message.text().slice(0, 300));
      }
    });
    page.on('requestfailed', (request) => {
      const error = request.failure()?.errorText;
      if (error !== 'net::ERR_ABORTED') console.log('[requestfailed]', request.url(), error);
    });
    page.on('crash', () => console.log('[page] CRASHED'));

    await page.goto(`http://127.0.0.1:${PORT}/gpu-profile.html`, { waitUntil: 'load' });
    await page.waitForFunction(() => window.__ready === true, { timeout: 60000 });

    return await page.evaluate(
      (options) => window.__profile(options),
      { model, profileTokens: STEPS },
    );
  } finally {
    await context.close();
  }
}

function mean(values) {
  if (!values.length) return null;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function fmt(value, digits = 1) {
  if (value == null || !Number.isFinite(value)) return '—';
  return value.toFixed(digits);
}

function sparkline(values, width = 48) {
  const chars = '▁▂▃▄▅▆▇█';
  if (!values.length) return '';
  const max = Math.max(...values);
  if (!max) return ' '.repeat(width);
  const buckets = [];
  const size = values.length / width;
  for (let i = 0; i < width; i++) {
    const slice = values.slice(Math.floor(i * size), Math.max(Math.floor((i + 1) * size), Math.floor(i * size) + 1));
    buckets.push(mean(slice) ?? 0);
  }
  return buckets
    .map((value) => chars[Math.min(chars.length - 1, Math.floor((value / max) * (chars.length - 1)))])
    .join('');
}

try {
  await waitForServer();

  const results = [];
  for (const model of MODELS) {
    console.log(`\n=== ${model} ===`);
    results.push(await measure(model));
  }

  for (const result of results) {
    if (result.error) {
      console.error(`\n[${result.model}] error: ${result.error}`);
      if (result.stack) console.error(result.stack);
    }
  }

  console.log('\nPer decode step (GPU timestamps, µs)');
  console.log(
    [
      'model'.padEnd(8),
      'step'.padStart(5),
      'passes'.padStart(7),
      'dispatch'.padStart(9),
      'sum µs'.padStart(9),
      'span µs'.padStart(9),
      'wall µs'.padStart(9),
      'sum/wall'.padStart(9),
      'median'.padStart(8),
      'p90'.padStart(8),
      'max'.padStart(8),
    ].join(' '),
  );

  for (const result of results) {
    if (result.error) continue;
    for (const run of result.runs) {
      console.log(
        [
          result.model.padEnd(8),
          String(run.step).padStart(5),
          String(run.passes).padStart(7),
          String(run.totalDispatches).padStart(9),
          fmt(run.sumUs).padStart(9),
          fmt(run.spanUs).padStart(9),
          fmt(run.stepWallMs == null ? null : run.stepWallMs * 1000).padStart(9),
          fmt(run.stepWallMs ? run.sumUs / (run.stepWallMs * 1000) : null, 2).padStart(9),
          fmt(run.medianUs, 2).padStart(8),
          fmt(run.p90Us, 2).padStart(8),
          fmt(run.maxUs, 2).padStart(8),
        ].join(' '),
      );
    }
  }

  console.log('\nGPU timeline of one decode step (each bar = one compute pass, mean over steps)');
  for (const result of results) {
    if (result.error || !result.runs.length) continue;
    const width = Math.min(...result.runs.map((r) => r.segments.length));
    const profile = [];
    for (let i = 0; i < width; i++) {
      profile.push(mean(result.runs.map((r) => r.segments[i]?.us ?? 0)) ?? 0);
    }
    const maxUs = Math.max(...profile);
    console.log(
      `${result.model.padEnd(8)} ${sparkline(profile, 64)}  ` +
        `passes=${width} max=${fmt(maxUs, 1)}µs`,
    );
    const top = profile
      .map((us, index) => ({ index, us }))
      .sort((a, b) => b.us - a.us)
      .slice(0, 6);
    console.log(
      `         top passes: ${top
        .map((entry) => `#${entry.index}=${fmt(entry.us, 1)}µs`)
        .join('  ')}`,
    );
  }

  console.log('\nPass-duration distribution (mean over steps)');
  for (const result of results) {
    if (result.error || !result.runs.length) continue;
    const all = result.runs.flatMap((r) => r.segments.map((s) => s.us)).sort((a, b) => a - b);
    const pick = (p) => all[Math.min(all.length - 1, Math.floor(p * all.length))];
    const sum = all.reduce((a, b) => a + b, 0);
    console.log(
      `${result.model.padEnd(8)} passes=${all.length} sum=${fmt(sum / result.runs.length)}µs ` +
        `p50=${fmt(pick(0.5), 2)} p75=${fmt(pick(0.75), 2)} p90=${fmt(pick(0.9), 2)} ` +
        `p99=${fmt(pick(0.99), 2)} max=${fmt(all.at(-1), 1)}`,
    );
  }

  console.log('\nDispatches vs pass duration (mean over steps)');
  for (const result of results) {
    if (result.error || !result.runs.length) continue;
    const byCount = new Map();
    for (const run of result.runs) {
      for (const seg of run.segments) {
        const entry = byCount.get(seg.dispatches) ?? { n: 0, us: 0 };
        entry.n += 1;
        entry.us += seg.us;
        byCount.set(seg.dispatches, entry);
      }
    }
    const rows = [...byCount.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([count, entry]) =>
        `${count}d:${fmt(entry.us / entry.n, 2)}µs×${entry.n}`,
      );
    console.log(`${result.model.padEnd(8)} ${rows.join('  ')}`);
  }

  const ledger = {
    timestamp: new Date().toISOString(),
    commit: execFileSync('git', ['rev-parse', '--short', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim(),
    loadavg: load,
    config: { models: MODELS, steps: STEPS, port: PORT, headless: HEADLESS },
    nodeCounts: NODES,
    results,
  };
  mkdirSync(join(root, 'runs'), { recursive: true });
  writeFileSync(OUT, JSON.stringify(ledger, null, 2));
  console.log(`\nwrote ${OUT}`);

  if (results.some((r) => r.error)) process.exitCode = 1;
} finally {
  server.kill('SIGTERM');
}
