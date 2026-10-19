// Experiment 2: count real WebGPU calls per prefill and per decode token for
// each artifact, on a single loaded model (no A/B, no dual-model memory bias).
//
//   node bench/browser/prepare.mjs        # once
//   node bench/browser/count.mjs
//
// Env:
//   MODELS=base,gate1,gate2 MAX_NEW=32 HEADLESS=1 PORT=8124 OUT=...

import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  DISPATCH_COUNTER,
  compareModels,
  countsRun,
  timingRun,
} from '../analysis/ledger.mjs';
import {
  announceLoad,
  instrumentPage,
  launchChrome,
  startServer,
  stampLedger,
  writeLedger,
} from './runtime.mjs';

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

const load = announceLoad();

const server = startServer(PORT);

async function measure(model) {
  const context = await launchChrome({
    headless: HEADLESS,
    label: `fused-onnx-count-${model}`,
  });

  try {
    const page = await context.newPage();
    await instrumentPage(page, { prefix: model });

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
  await server.ready;

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

  const ledger = stampLedger({
    root,
    config: { models: MODELS, maxNewTokens: MAX_NEW, port: PORT, headless: HEADLESS },
    load,
    nodeCounts: NODES,
    results,
  });

  // Write before analysing: a failed comparison must not discard measurements.
  writeLedger(OUT, ledger);

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
  server.stop();
}
