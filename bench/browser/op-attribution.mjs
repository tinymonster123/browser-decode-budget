// Gate 7: which kernels own the decode time above the weight floor.
//
// Each compute pass is timed with timestamp-query and tagged with the pipeline
// identity that issued it. The identity comes from the descriptor labels ORT's
// WebGPU EP passes at pipeline creation (with a shader-code hash as fallback),
// so this driver does not guess an op name.
//
//   node bench/browser/prepare.mjs        # once
//   node bench/browser/op-attribution.mjs
//
// Env: MODELS=base STEPS=4 HEADLESS=1 PORT=8128 OUT=...

import { attributeOps } from '../analysis/ledger.mjs';
import { nodeCounts } from './artifacts.mjs';
import {
  envList,
  envNumber,
  fmt,
  ledgerPath,
  runDriver,
  timelineFailures,
} from './driver.mjs';

const MODELS = envList('MODELS', ['base']);
const STEPS = Math.max(2, envNumber('STEPS', 4));
const NODES = nodeCounts();

// Splitting one pass per dispatch adds a pass boundary to every dispatch.
// Gate 4 measured that boundary at 0.88 µs (runs/2026-10-19-exp3-dispatch-cost.json);
// the attribution subtracts it so cheap kernels are not credited with it.
const PER_DISPATCH_OVERHEAD_US = 0.88;

// The gate's kill criterion: an attribution that leaves more than 5% of the
// pass time unlabeled cannot answer the question and must be reported as a
// failed attribution, not rounded into a conclusion.
const MIN_IDENTITY_COVERAGE = 0.95;

await runDriver({
  name: 'op-attribution',
  port: envNumber('PORT', 8128),
  out: ledgerPath('OUT', 'runs/2026-10-20-exp7-op-attribution.json'),
  config: {
    models: MODELS,
    steps: STEPS,
    minIdentityCoverage: MIN_IDENTITY_COVERAGE,
    perDispatchOverheadUs: PER_DISPATCH_OVERHEAD_US,
  },

  async measure(ctx) {
    const results = [];
    const attributions = {};
    for (const model of MODELS) {
      ctx.log(`\n=== ${model} ===`);
      const result = await ctx.measureModel(model, {
        path: '/op-attribution.html',
        call: '__attribute',
        options: { model, profileTokens: STEPS },
      });
      results.push(result);
      if (!result.error && result.runs?.length) {
        attributions[model] = attributeOps(result, {
          perDispatchOverheadUs: PER_DISPATCH_OVERHEAD_US,
        });
      }
    }
    return { results, attributions };
  },

  checks({ results, attributions }) {
    return results.map((result) => {
      const failures = timelineFailures(result);
      return {
        name: `${result.model}: timestamps`,
        ok: failures.length === 0,
        detail:
          failures.join('; ') ||
          `${result.runs?.length ?? 0} steps covered`,
      };
    }).concat(
      results
        .filter((result) => !result.error && result.runs?.length)
        .map((result) => {
          const attribution = attributions[result.model];
          const coverage = attribution?.identityCoverage;
          return {
            name: `${result.model}: identity coverage`,
            ok: coverage != null && coverage >= MIN_IDENTITY_COVERAGE,
            detail:
              coverage == null
                ? 'no attribution'
                : `${fmt(coverage * 100, 1)}% of pass time has a named op ` +
                  `(minimum ${fmt(MIN_IDENTITY_COVERAGE * 100, 0)}%)`,
          };
        }),
    );
  },

  report({ results, attributions }) {
    for (const result of results) {
      if (!result.error) continue;
      console.error(`\n[${result.model}] error: ${result.error}`);
      if (result.stack) console.error(result.stack);
    }

    for (const result of results) {
      if (result.error) continue;
      const attribution = attributions[result.model];
      if (!attribution) continue;

      const meanPasses =
        attribution.steps.reduce((sum, step) => sum + step.passes, 0) /
        attribution.steps.length;
      console.log(
        `\n${result.model}: ${fmt(meanPasses, 0)} passes/step, ` +
          `raw ${fmt(attribution.totalUsPerStep / 1000)} ms/step, ` +
          `corrected ${fmt(attribution.totalCorrectedUsPerStep / 1000)} ms/step ` +
          `(−${fmt(PER_DISPATCH_OVERHEAD_US)} µs per dispatch), ` +
          `identity coverage ${fmt(attribution.identityCoverage * 100, 1)}% ` +
          `(${attribution.classes.length} identities, ${attribution.mixedPasses} mixed passes)`,
      );
      if (result.identities?.length) {
        console.log(`  identities: ${result.identities.slice(0, 12).join(', ')}`);
      }
      const details = result.identityDetails ?? {};
      for (const row of attribution.classes.slice(0, 8)) {
        const detail = details[row.key];
        if (!detail) continue;
        console.log(
          `  ${row.label} len=${detail.codeLength} ` +
            `pipelineLabel="${detail.label}" moduleLabel="${detail.moduleLabel}" ` +
            `${detail.preview.slice(0, 140)}`,
        );
      }
      if (attribution.identityCoverage < MIN_IDENTITY_COVERAGE) {
        console.log(
          '  [cannot attribute] more than 5% of pass time has no named op; ' +
            'the rows below are incomplete',
        );
      }

      console.log(
        [
          'class'.padEnd(40),
          'µs/step'.padStart(10),
          'share'.padStart(8),
          'step range'.padStart(12),
          'corrected'.padStart(10),
          'dispatch'.padStart(9),
        ].join(' '),
      );
      for (const row of attribution.classes.slice(0, 20)) {
        const range = row.shareStability
          ? `${fmt(row.shareStability.min * 100, 1)}–${fmt(
              row.shareStability.max * 100,
              1,
            )}%`
          : '—';
        console.log(
          [
            row.label.slice(0, 40).padEnd(40),
            fmt(row.usPerStep, 1).padStart(10),
            `${fmt((row.share ?? 0) * 100, 1)}%`.padStart(8),
            range.padStart(12),
            fmt(row.correctedUsPerStep, 1).padStart(10),
            fmt(row.dispatchesPerStep, 1).padStart(9),
          ].join(' '),
        );
      }
    }
  },

  ledger({ results, attributions }) {
    return { nodeCounts: NODES, results, attributions };
  },
});
