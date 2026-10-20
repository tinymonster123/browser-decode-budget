// Experiment 2: count real WebGPU calls per prefill and per decode token for
// each artifact, on a single loaded model (no A/B, no dual-model memory bias).
//
//   node bench/browser/prepare.mjs        # once
//   node bench/browser/count.mjs
//
// Env:
//   MODELS=base,gate1,gate2 MAX_NEW=32 HEADLESS=1 PORT=8124 OUT=...

import {
  DISPATCH_COUNTER,
  compareModels,
  countsRun,
  timingRun,
} from '../analysis/ledger.mjs';
import { nodeCounts } from './artifacts.mjs';
import {
  envList,
  envNumber,
  fmt,
  ledgerPath,
  runDriver,
} from './driver.mjs';

const MODELS = envList('MODELS', ['base', 'gate1', 'gate2']);
const MAX_NEW = Math.max(2, envNumber('MAX_NEW', 32));
const NODES = nodeCounts();

function pad(value, width) {
  return String(value).padEnd(width);
}

function padStart(value, width) {
  return String(value).padStart(width);
}

function pick(run, key) {
  return run?.perToken?.[key] ?? 0;
}

const { ledger } = await runDriver({
  name: 'count',
  port: envNumber('PORT', 8124),
  out: ledgerPath('OUT', 'runs/2026-10-19-exp2-dispatch-count.json'),
  config: { models: MODELS, maxNewTokens: MAX_NEW },

  async measure(ctx) {
    const results = [];
    for (const model of MODELS) {
      ctx.log(`\n=== ${model} ===`);
      results.push(
        await ctx.measureModel(model, {
          path: '/count.html',
          call: '__count',
          options: { model, maxNewTokens: MAX_NEW, runCpu: [false, true] },
        }),
      );
    }
    return { results };
  },

  // Harness self-checks become assertions: a run whose patch did not take, or
  // that had no WebGPU, cannot be used as a measurement.
  checks({ results }) {
    return results.map((result) => {
      const failures = [];
      if (result.error) failures.push(result.error);
      if (!result.hasWebGpu) failures.push('WebGPU unavailable');
      for (const [name, ok] of Object.entries(result.patched ?? {})) {
        if (!ok) failures.push(`${name} patch failed`);
      }
      return {
        name: `${result.model}: harness`,
        ok: failures.length === 0,
        detail: failures.join('; ') || 'patches applied, WebGPU present',
      };
    });
  },

  report({ results }) {
    for (const result of results) {
      if (!result.error) continue;
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
  },

  ledger({ results }) {
    return { nodeCounts: NODES, results };
  },
});

// ---- The part that actually answers "why" ------------------------------
// The ledger is already written, so a failed comparison cannot discard the
// measurements that produced it.

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
