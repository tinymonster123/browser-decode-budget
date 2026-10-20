// Gate 8: why 242 MatMulNBits dispatches do not share one bandwidth.
//
// The attribution page records, per dispatch, the sizes of the storage
// buffers its bind groups reference. A size that matches exactly one known
// MatMulNBits initializer identifies the weight that dispatch reads, so time
// divided by bytes gives an effective bandwidth per weight size.
//
//   node bench/browser/prepare.mjs        # once
//   make weight-bytes                     # records the known sizes
//   node bench/browser/matmul-efficiency.mjs
//
// Env: MODELS=base STEPS=4 HEADLESS=1 PORT=8129 OUT=... WEIGHTS=...

import { join } from 'node:path';

import { attributeMatmuls } from '../analysis/ledger.mjs';
import { nodeCounts } from './artifacts.mjs';
import {
  envList,
  envNumber,
  envText,
  fmt,
  knownFingerprints,
  ledgerPath,
  root,
  runDriver,
  timelineFailures,
} from './driver.mjs';

const MODELS = envList('MODELS', ['base']);
const STEPS = Math.max(2, envNumber('STEPS', 4));
const NODES = nodeCounts();
const OP_KEY = 'op:MatMulNBits';

// Kill criterion, fixed before the run: if less than 90% of the matmul time
// can be tied to a known weight size, the per-size bandwidth table cannot
// answer the question and the gate fails instead of averaging the gap away.
const MIN_COVERAGE = 0.9;

const WEIGHTS = envText(
  'WEIGHTS',
  join(root, 'runs/2026-10-20-exp7-weight-bytes-by-op.json'),
);

const FINGERPRINTS = knownFingerprints(WEIGHTS, 'MatMulNBits');

await runDriver({
  name: 'matmul-efficiency',
  port: envNumber('PORT', 8129),
  out: ledgerPath('OUT', 'runs/2026-10-20-exp8-matmul-efficiency.json'),
  config: {
    models: MODELS,
    steps: STEPS,
    weights: WEIGHTS,
    opKey: OP_KEY,
    minCoverage: MIN_COVERAGE,
  },

  async measure(ctx) {
    const results = [];
    const analyses = {};
    for (const model of MODELS) {
      ctx.log(`\n=== ${model} ===`);
      const result = await ctx.measureModel(model, {
        path: '/op-attribution.html',
        call: '__attribute',
        options: { model, profileTokens: STEPS },
      });
      results.push(result);
      if (!result.error && result.runs?.length) {
        analyses[model] = attributeMatmuls(result, {
          opKey: OP_KEY,
          fingerprints: FINGERPRINTS,
        });
      }
    }
    return { results, analyses };
  },

  checks({ results, analyses }) {
    const base = results.map((result) => {
      const failures = timelineFailures(result);
      return {
        name: `${result.model}: harness`,
        ok: failures.length === 0,
        detail:
          failures.join('; ') ||
          `${result.runs?.length ?? 0} steps covered`,
      };
    });

    const coverage = results
      .filter((result) => !result.error && result.runs?.length)
      .map((result) => {
        const analysis = analyses[result.model];
        return {
          name: `${result.model}: weight-size coverage`,
          ok:
            analysis.coverage != null &&
            analysis.coverage >= MIN_COVERAGE &&
            analysis.invalidSegments === 0,
          detail:
            analysis.coverage == null
              ? 'no analysis'
              : `${fmt(analysis.coverage * 100, 1)}% of matmul time matched ` +
                `(minimum ${fmt(MIN_COVERAGE * 100, 0)}%), ` +
                `${analysis.unmatchedSegments} unmatched / ` +
                `${analysis.ambiguousSegments} ambiguous / ` +
                `${analysis.invalidSegments} invalid segments`,
        };
      });

    return base.concat(coverage);
  },

  report({ results, analyses }) {
    for (const result of results) {
      if (!result.error) continue;
      console.error(`\n[${result.model}] error: ${result.error}`);
      if (result.stack) console.error(result.stack);
    }

    for (const result of results) {
      if (result.error) continue;
      const analysis = analyses[result.model];
      if (!analysis) continue;

      console.log(
        `\n${result.model}: matmul ${fmt(analysis.totalUsPerStep / 1000)} ms/step, ` +
          `${fmt(analysis.dispatchesPerStep, 0)} dispatches/step, ` +
          `weight-size coverage ${fmt(analysis.coverage * 100, 1)}% ` +
          `(${analysis.classes.length} sizes, ${analysis.ambiguousSegments} ambiguous)`,
      );

      console.log(
        [
          'weight MiB'.padStart(10),
          'dispatch'.padStart(9),
          'µs/step'.padStart(10),
          'GB/s'.padStart(8),
          'share'.padStart(8),
        ].join(' '),
      );
      for (const row of analysis.classes) {
        console.log(
          [
            fmt(row.bytes / 1024 / 1024, 2).padStart(10),
            fmt(row.dispatchesPerStep, 1).padStart(9),
            fmt(row.usPerStep, 1).padStart(10),
            fmt(row.gbps, 1).padStart(8),
            `${fmt((row.shareOfMatmul ?? 0) * 100, 1)}%`.padStart(8),
          ].join(' '),
        );
      }
    }
  },

  ledger({ results, analyses }) {
    return { nodeCounts: NODES, results, analyses };
  },
});
