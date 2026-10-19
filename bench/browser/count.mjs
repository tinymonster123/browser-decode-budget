// Experiment 2: count real WebGPU calls per prefill and per decode token for
// each artifact, on a single loaded model (no A/B, no dual-model memory bias).
//
//   node bench/browser/prepare.mjs        # once
//   node bench/browser/count.mjs
//
// Env:
//   MODELS=base,gate1,gate2 MAX_NEW=32 HEADLESS=1 PORT=8124 OUT=...

import { execFileSync, spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import { cpus, loadavg, tmpdir } from 'node:os';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  DISPATCH_COUNTER,
  compareModels,
  countsRun,
  timingRun,
  makeLedger,
} from '../analysis/ledger.mjs';

const require = createRequire(
  process.env.PLAYWRIGHT_REQUIRE ?? '/Users/fenghanli/Projects/map-tile-lab/package.json',
);
const { chromium } = require('playwright');

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '../..');
const PORT = Number(process.env.PORT ?? 8124);
const MAX_NEW = Math.max(2, Number(process.env.MAX_NEW ?? 32));
const HEADLESS = process.env.HEADLESS === '1';
const MODELS = (process.env.MODELS ?? 'base,gate1,gate2').split(',').filter(Boolean);
const OUT =
  process.env.OUT ?? join(root, 'runs', '2026-10-19-exp2-dispatch-count.json');

// From docs/experiments/2026-10-19-gate0-census.md and the gate reports.
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
  const userDataDir = mkdtempSync(join(tmpdir(), `fused-onnx-count-${model}-`));
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

    await page.goto(`http://127.0.0.1:${PORT}/count.html`, { waitUntil: 'load' });
    await page.waitForFunction(() => window.__ready === true, { timeout: 60000 });

    return await page.evaluate(
      (options) => window.__count(options),
      { model, maxNewTokens: MAX_NEW, runCpu: [false, true] },
    );
  } finally {
    await context.close();
  }
}

function fmt(value, digits = 1) {
  if (value == null || !Number.isFinite(value)) return '—';
  return value.toFixed(digits);
}

function pad(value, width) {
  return String(value).padEnd(width);
}

function padStart(value, width) {
  return String(value).padStart(width);
}

function pick(run, key) {
  return run?.perToken?.[key] ?? 0;
}

function pickCpu(run, key) {
  return run?.cpuMsPerToken?.[key] ?? 0;
}

try {
  await waitForServer();

  const results = [];
  for (const model of MODELS) {
    console.log(`\n=== ${model} ===`);
    results.push(await measure(model));
  }

  const errors = results.filter((r) => r.error);
  // Harness self-checks become assertions: a run whose patch did not take, or
  // that had no WebGPU, cannot be used as a measurement.
  const checkFailures = results.flatMap((result) => {
    if (result.error) return [];
    const failures = [];
    if (!result.hasWebGpu) failures.push(`${result.model}: WebGPU unavailable`);
    for (const [name, ok] of Object.entries(result.patched ?? {})) {
      if (!ok) failures.push(`${result.model}: ${name} patch failed`);
    }
    return failures;
  });
  for (const result of errors) {
    console.error(`\n[${result.model}] error: ${result.error}`);
    if (result.stack) console.error(result.stack);
  }

  console.log('\nWebGPU calls per decode token (median of decode steps)');
  console.log(
    [
      pad('model', 8),
      padStart('nodes', 6),
      padStart('dispatch', 9),
      padStart('indirect', 9),
      padStart('passes', 7),
      padStart('submit', 7),
      padStart('bindGrp', 8),
      padStart('writeBuf', 9),
      padStart('ms/token', 9),
    ].join(' '),
  );

  for (const result of results) {
    if (result.error) continue;
    // Roles come from the analysis seam, so a missing run is an error.
    const counts = countsRun(result);
    const timed = timingRun(result);
    console.log(
      [
        pad(result.model, 8),
        padStart(NODES[result.model] ?? '?', 6),
        padStart(fmt(pick(counts, DISPATCH_COUNTER), 1), 9),
        padStart(fmt(pick(counts, 'pass.dispatchWorkgroupsIndirect'), 1), 9),
        padStart(fmt(pick(counts, 'enc.beginComputePass'), 1), 7),
        padStart(fmt(pick(counts, 'queue.submit'), 1), 7),
        padStart(fmt(pick(counts, 'dev.createBindGroup'), 1), 8),
        padStart(fmt(pick(counts, 'queue.writeBuffer'), 1), 9),
        padStart(fmt(timed.decodeMsPerStep, 2), 9),
      ].join(' '),
    );
  }

  console.log('\nPrefill (one pass over the prompt)');
  console.log(
    [
      pad('model', 8),
      padStart('dispatch', 9),
      padStart('passes', 7),
      padStart('submit', 7),
      padStart('bindGrp', 8),
      padStart('writeBuf', 9),
      padStart('ms', 9),
    ].join(' '),
  );
  for (const result of results) {
    if (result.error) continue;
    const counts = countsRun(result);
    const prefill = counts.prefill ?? {};
    console.log(
      [
        pad(result.model, 8),
        padStart(fmt(prefill[DISPATCH_COUNTER], 1), 9),
        padStart(fmt(prefill['enc.beginComputePass'], 1), 7),
        padStart(fmt(prefill['queue.submit'], 1), 7),
        padStart(fmt(prefill['dev.createBindGroup'], 1), 8),
        padStart(fmt(prefill['queue.writeBuffer'], 1), 9),
        padStart(fmt(counts.prefillMs, 2), 9),
      ].join(' '),
    );
  }

  console.log('\nCPU time inside the wrappers, µs per decode token');
  for (const result of results) {
    if (result.error) continue;
    const timed = timingRun(result);
    const rows = Object.entries(timed.cpuMsPerToken ?? {})
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8)
      .map(([key, ms]) => `${key}=${(ms * 1000).toFixed(1)}µs`);
    console.log(`${pad(result.model, 8)} ${rows.join('  ') || '—'}`);
  }

  const ledger = makeLedger({
    commit: execFileSync('git', ['rev-parse', '--short', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim(),
    loadavg: load,
    config: { models: MODELS, maxNewTokens: MAX_NEW, port: PORT, headless: HEADLESS },
    nodeCounts: NODES,
    results,
  });

  // Write before analysing: a failed comparison must not discard measurements.
  mkdirSync(join(root, 'runs'), { recursive: true });
  writeFileSync(OUT, JSON.stringify(ledger, null, 2));

  // ---- The part that actually answers "why" -------------------------
  console.log('\nNode reduction vs real dispatch reduction');
  const pairs = [
    ['base', 'gate1'],
    ['gate1', 'gate2'],
    ['base', 'gate2'],
  ];
  const comparisons = compareModels(ledger, { pairs });
  console.log(
    [
      pad('pair', 14),
      padStart('Δnodes', 8),
      padStart('Δdispatch', 11),
      padStart('node:dispatch', 14),
      padStart('Δms/token', 11),
    ].join(' '),
  );
  for (const comparison of comparisons) {
    if (comparison.error) {
      console.log(`${pad(`${comparison.a}→${comparison.b}`, 14)} ${comparison.error}`);
      continue;
    }
    console.log(
      [
        pad(`${comparison.a}→${comparison.b}`, 14),
        padStart(comparison.deltaNodes ?? '—', 8),
        padStart(comparison.deltaDispatches.toFixed(1), 11),
        padStart(
          comparison.nodeToDispatchRatio == null
            ? '—'
            : comparison.nodeToDispatchRatio.toFixed(2),
          14,
        ),
        padStart(
          `${comparison.deltaMsPerStep >= 0 ? '+' : ''}${comparison.deltaMsPerStep.toFixed(2)}`,
          11,
        ),
      ].join(' '),
    );
  }

  console.log('\nIf one dispatch costs C µs, the measured Δdispatch predicts:');
  const cValues = [3, 5, 10, 20, 30];
  console.log(
    [pad('pair', 14), ...cValues.map((c) => padStart(`${c}µs`, 9))].join(' '),
  );
  for (const comparison of comparisons) {
    if (comparison.error) continue;
    const cells = cValues.map((c) => {
      const ms = (comparison.deltaDispatches * c) / 1000;
      return padStart(`${ms >= 0 ? '+' : ''}${ms.toFixed(2)}ms`, 9);
    });
    console.log([pad(`${comparison.a}→${comparison.b}`, 14), ...cells].join(' '));
  }

  for (const failure of checkFailures) console.error(`[check] ${failure}`);
  console.log(`\nwrote ${OUT}`);

  if (errors.length || checkFailures.length) process.exitCode = 1;
} finally {
  server.kill('SIGTERM');
}
