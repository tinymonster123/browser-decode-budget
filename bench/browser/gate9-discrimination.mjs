// Gate 9: which matmul gap is real, and is the split-pass measurement
// inflating it?
//
// Gate 8 found effective bandwidth falling with the bytes one dispatch reads:
// the 0.22 MiB k_proj runs at 10.7–15.2 GB/s while classes above 3.47 MiB run
// at 55.9–72.7. Two candidates survived that report:
//
//   A. small matmuls are parallelism/latency limited (few workgroups);
//   B. large matmuls leave memory efficiency on the table (62 vs 86 GB/s).
//
// Gate 7/8 measured both inside a one-pass-per-dispatch encoding, which adds a
// pass boundary per dispatch and removes the overlap a shared pass allows.
// Before spending on either candidate, this gate replays one recorded dispatch
// per weight class in both encodings and records the launched workgroup count.
//
// Kill criteria, fixed before the run:
//
//   1. At least 90% of MatMulNBits time and 90% of replayed dispatches must
//      match a known weight fingerprint; below that this is a negative result.
//   2. Workgroup counts must be present for at least 95% of matched segments;
//      otherwise the parallelism question is unanswered.
//   3. For each class, split→batched is reported. If split exceeds batched by
//      more than 15% for a class, the Gate 7/8 absolute time for that class is
//      inflated and the corrected value is the one the report uses.
//   4. A candidate enters an optimization gate only if its corrected upper
//      bound is at least 10% of the per-token time. A candidate whose lever is
//      a custom WebGPU kernel stays out of scope (repository non-goal).
//
//   node bench/browser/prepare.mjs        # once
//   make weight-bytes                     # records the known sizes
//   node bench/browser/gate9-discrimination.mjs
//
// Env: MODELS=base STEPS=4 HEADLESS=1 PORT=8130 OUT=... WEIGHTS=... TOKEN=...
//      COUNTS=1,4,16,64 REPEATS=7

import { join } from 'node:path';

import {
  attributeMatmuls,
  compareEncodings,
  decideEncodings,
} from '../analysis/ledger.mjs';
import { nodeCounts } from './artifacts.mjs';
import {
  envList,
  envNumber,
  envText,
  fmt,
  knownFingerprints,
  ledgerPath,
  readJson,
  root,
  runDriver,
  timelineFailures,
} from './driver.mjs';

const MODELS = envList('MODELS', ['base']);
const STEPS = Math.max(2, envNumber('STEPS', 4));
const NODES = nodeCounts();
const OP_KEY = 'op:MatMulNBits';

const MIN_COVERAGE = 0.9;
const MIN_WORKGROUP_COVERAGE = 0.95;
const SPLIT_INFLATION_LIMIT = 0.15;
const SMALL_CLASS_BYTES = 2 * 1024 * 1024;
const PRACTICAL_CEILING_GBPS = 86.18; // Gate 5 LM head, batched pass
const DECISION_MIN_SHARE = 0.1;

// The fourth kill criterion is not numeric: a candidate only enters an
// optimization gate if the lever is in stack. Both candidates' only lever is
// a custom or grouped WebGPU kernel, which this repository excludes.
const LEVER_IN_STACK = { small: false, large: false };

const REPLAY_COUNTS = envText('COUNTS', '1,4,16,64')
  .split(',')
  .map((value) => Number(value.trim()))
  .filter((value) => Number.isInteger(value) && value >= 1);
const REPLAY_REPEATS = Math.max(2, envNumber('REPEATS', 7));

const WEIGHTS = envText(
  'WEIGHTS',
  join(root, 'runs/2026-10-20-exp7-weight-bytes-by-op.json'),
);
const TOKEN_LEDGER = envText(
  'TOKEN',
  join(root, 'runs/2026-10-19-exp2-dispatch-count.json'),
);

// The per-token wall time comes from the uninstrumented count run. Split-pass
// wall time is 2–3x larger and must not be used as the denominator.
function tokenMsFrom(path) {
  const ledger = readJson(path);
  for (const result of ledger.results ?? []) {
    for (const variant of result.variants ?? []) {
      for (const run of variant.runs ?? []) {
        if (run.cpu === false && Number.isFinite(run.decodeMsPerStep)) {
          return run.decodeMsPerStep;
        }
      }
    }
  }
  throw new Error(`no cpu=false decodeMsPerStep in ${path}`);
}

const FINGERPRINTS = knownFingerprints(WEIGHTS, 'MatMulNBits');
const TOKEN_MS = tokenMsFrom(TOKEN_LEDGER);

function sum(values) {
  return values.reduce((total, value) => total + value, 0);
}

await runDriver({
  name: 'gate9-discrimination',
  port: envNumber('PORT', 8130),
  out: ledgerPath('OUT', 'runs/2026-10-20-exp9-gate9-discrimination.json'),
  config: {
    models: MODELS,
    steps: STEPS,
    weights: WEIGHTS,
    tokenLedger: TOKEN_LEDGER,
    opKey: OP_KEY,
    counts: REPLAY_COUNTS,
    repeats: REPLAY_REPEATS,
    minCoverage: MIN_COVERAGE,
    minWorkgroupCoverage: MIN_WORKGROUP_COVERAGE,
    splitInflationLimit: SPLIT_INFLATION_LIMIT,
    smallClassBytes: SMALL_CLASS_BYTES,
    practicalCeilingGbps: PRACTICAL_CEILING_GBPS,
    decisionMinShare: DECISION_MIN_SHARE,
  },

  async measure(ctx) {
    const results = [];
    const analyses = {};
    for (const model of MODELS) {
      ctx.log(`\n=== ${model} ===`);
      const result = await ctx.measureModel(model, {
        path: '/op-attribution.html',
        call: '__gate9',
        options: {
          model,
          profileTokens: STEPS,
          replay: {
            opKey: OP_KEY,
            counts: REPLAY_COUNTS,
            repeats: REPLAY_REPEATS,
          },
        },
      });
      results.push({ model, ...result });
      if (!result.attribution?.error && !result.replay?.error) {
        const attribution = attributeMatmuls(result.attribution, {
          opKey: OP_KEY,
          fingerprints: FINGERPRINTS,
        });
        const encodings = compareEncodings(result.replay, {
          fingerprints: FINGERPRINTS,
        });
        const decision = decideEncodings(
          { attribution, encodings, tokenMs: TOKEN_MS },
          {
            smallClassBytes: SMALL_CLASS_BYTES,
            practicalCeilingGbps: PRACTICAL_CEILING_GBPS,
            minShare: DECISION_MIN_SHARE,
          },
        );
        analyses[model] = { attribution, encodings, decision };
      }
    }
    return { results, analyses };
  },

  checks({ results, analyses }) {
    const checks = [];

    for (const result of results) {
      const attribution = result.attribution ?? {};
      const failures = timelineFailures({
        ...attribution,
        error: result.error ?? attribution.error ?? null,
      });
      checks.push({
        name: `${result.model}: harness`,
        ok: failures.length === 0,
        detail:
          failures.join('; ') ||
          `${attribution.runs?.length ?? 0} steps, ${result.replay?.classes?.length ?? 0} replay classes`,
      });

      const analysis = analyses[result.model];
      if (!analysis) {
        checks.push({
          name: `${result.model}: analysis`,
          ok: false,
          detail: result.replay?.error ?? 'no analysis',
        });
        continue;
      }

      checks.push({
        name: `${result.model}: weight-size coverage`,
        ok: analysis.attribution.coverage >= MIN_COVERAGE,
        detail:
          `${fmt(analysis.attribution.coverage * 100, 1)}% of matmul time matched ` +
          `(minimum ${fmt(MIN_COVERAGE * 100, 0)}%), ` +
          `${analysis.attribution.unmatchedSegments} unmatched / ` +
          `${analysis.attribution.ambiguousSegments} ambiguous`,
      });

      checks.push({
        name: `${result.model}: workgroup coverage`,
        ok:
          analysis.attribution.workgroupCoverage != null &&
          analysis.attribution.workgroupCoverage >= MIN_WORKGROUP_COVERAGE,
        detail:
          analysis.attribution.workgroupCoverage == null
            ? 'no workgroup data'
            : `${fmt(analysis.attribution.workgroupCoverage * 100, 1)}% of matched ` +
              `segments carried a workgroup count ` +
              `(minimum ${fmt(MIN_WORKGROUP_COVERAGE * 100, 0)}%)`,
      });

      checks.push({
        name: `${result.model}: replay coverage`,
        ok:
          analysis.encodings.coverage != null &&
          analysis.encodings.coverage >= MIN_COVERAGE,
        detail:
          `${fmt((analysis.encodings.coverage ?? 0) * 100, 1)}% of replayed ` +
          `dispatches matched a fingerprint, ` +
          `${analysis.encodings.unmatchedClasses} unmatched classes, ` +
          `${result.replay?.skipped ?? 0} indirect dispatches skipped`,
      });

      // The decision joins attribution rows to replay rows on
      // (bytes, workgroups). A silent join miss would shrink the headroom
      // table without failing anything else, so count the joined dispatches.
      const joinedDispatches = sum(
        analysis.decision.rows.map((row) => row.dispatchesPerStep),
      );
      checks.push({
        name: `${result.model}: decision coverage`,
        ok:
          analysis.decision.unmatchedKeys.length === 0 &&
          Math.abs(joinedDispatches - analysis.attribution.dispatchesPerStep) <= 1,
        detail:
          `${fmt(joinedDispatches, 0)} of ` +
          `${fmt(analysis.attribution.dispatchesPerStep, 0)} dispatches per step ` +
          `joined a replay class; unmatched keys: ` +
          `${analysis.decision.unmatchedKeys.join(', ') || 'none'}`,
      });
    }

    return checks;
  },

  report({ results, analyses }) {
    for (const result of results) {
      if (result.attribution?.error) {
        console.error(`\n[${result.model}] attribution error: ${result.attribution.error}`);
        if (result.attribution.stack) console.error(result.attribution.stack);
      }
      if (result.replay?.error) {
        console.error(`\n[${result.model}] replay error: ${result.replay.error}`);
        if (result.replay.stack) console.error(result.replay.stack);
      }
    }

    for (const result of results) {
      const analysis = analyses[result.model];
      if (!analysis) continue;
      const { attribution, encodings, decision } = analysis;

      console.log(
        `\n${result.model}: matmul ${fmt(attribution.totalUsPerStep / 1000)} ms/step, ` +
          `${fmt(attribution.dispatchesPerStep, 0)} dispatches/step, ` +
          `coverage ${fmt(attribution.coverage * 100, 1)}%, ` +
          `workgroups on ${fmt(attribution.workgroupCoverage * 100, 1)}% of segments`,
      );
      console.log(
        `uniform writes: ${result.replay?.uniformWrites?.calls ?? 0} calls, ` +
          `${result.replay?.uniformWrites?.distinctBuffers ?? 0} distinct buffers, ` +
          `${result.replay?.uniformWrites?.restored ?? 0} restored`,
      );

      const byKey = new Map(encodings.rows.map((row) => [row.key, row]));
      const decisionByKey = new Map(decision.rows.map((row) => [row.key, row]));
      console.log(
        [
          'weight MiB'.padStart(10),
          'dispatch'.padStart(8),
          'wg'.padStart(8),
          'invoc'.padStart(10),
          'cold µs'.padStart(9),
          'hot split'.padStart(9),
          'hot batch'.padStart(9),
          's/b'.padStart(6),
          'cold GB/s'.padStart(9),
        ].join(' '),
      );
      for (const row of attribution.classes) {
        const encoding = byKey.get(row.key);
        const corrected = decisionByKey.get(row.key);
        const coldUs = row.dispatchesPerStep
          ? row.usPerStep / row.dispatchesPerStep
          : null;
        console.log(
          [
            fmt(row.bytes / 1024 / 1024, 2).padStart(10),
            fmt(row.dispatchesPerStep, 1).padStart(8),
            (row.workgroupStats ? fmt(row.workgroupStats.median, 0) : '—').padStart(8),
            (row.invocationStats ? fmt(row.invocationStats.median, 0) : '—').padStart(10),
            fmt(coldUs, 1).padStart(9),
            (encoding ? fmt(encoding.splitUsPerDispatch, 1) : '—').padStart(9),
            (encoding ? fmt(encoding.batchedUsPerDispatch, 1) : '—').padStart(9),
            (encoding ? fmt(encoding.splitToBatched, 2) : '—').padStart(6),
            (corrected ? fmt(corrected.coldGbps, 1) : '—').padStart(9),
          ].join(' '),
        );
      }
      console.log(
        'cold = this run\'s one-pass-per-dispatch attribution; ' +
          'hot split / hot batch = the same recorded dispatch replayed N times ' +
          '(the weight is cache-resident after the first repeat)',
      );

      const share = (value) => (value == null ? '—' : `${fmt(value * 100, 1)}%`);
      console.log(
        `\nmeasurement mode: per-dispatch split−batched ${fmt(decision.passOverheadUs, 2)} µs ` +
          `(median of positive differences on cache-resident classes); ` +
          `ratio median ${fmt(decision.splitRatios.median, 2)}, ` +
          `max ${fmt(decision.splitRatios.max, 2)} ` +
          `(the ratio is cache-confounded for weights larger than cache)`,
      );
      console.log(
        `small classes (< ${fmt(SMALL_CLASS_BYTES / 1024 / 1024, 1)} MiB): ` +
          `corrected ${fmt(decision.small.correctedMs)} ms, ` +
          `ideal at ${fmt(decision.referenceGbps, 1)} GB/s ${fmt(decision.small.idealMs)} ms, ` +
          `headroom ${fmt(decision.small.headroomMs)} ms ` +
          `(${share(decision.small.share)} of token)`,
      );
      console.log(
        `large classes at this run's best class (${fmt(decision.bestGbps, 1)} GB/s): ` +
          `corrected ${fmt(decision.large.correctedMs)} ms, ` +
          `ideal ${fmt(decision.large.idealRunBestMs)} ms, ` +
          `headroom ${fmt(decision.large.headroomRunBestMs)} ms ` +
          `(${share(decision.large.shareRunBest)} of token)`,
      );
      console.log(
        `large classes at the Gate 5 ceiling (${fmt(decision.practicalCeilingGbps, 2)} GB/s): ` +
          `headroom ${fmt(decision.large.headroomMs)} ms ` +
          `(${share(decision.large.share)} of token) — cross-load sensitivity, ` +
          `not a same-run bound`,
      );
      // The 10% bar uses the same-run best class because the Gate 5 ceiling was
      // measured at a different system load, and the project forbids comparing
      // absolute GB/s across loads.
      const idealSmallBest =
        decision.small.bytesPerStep / (decision.bestGbps * 1e6);
      const headroomSmallBest = decision.small.correctedMs - idealSmallBest;
      console.log(
        `small classes at this run's best class: ideal ${fmt(idealSmallBest)} ms, ` +
          `headroom ${fmt(headroomSmallBest)} ms ` +
          `(${share(headroomSmallBest / decision.tokenMs)} of token)`,
      );

      const inflated = decision.rows.filter(
        (row) => row.splitToBatched - 1 > SPLIT_INFLATION_LIMIT,
      );
      console.log(
        `split/batched > ${fmt(SPLIT_INFLATION_LIMIT * 100, 0)}%: ` +
          (inflated.length
            ? inflated
                .map(
                  (row) =>
                    `${fmt(row.bytes / 1024 / 1024, 2)} MiB/${row.workgroups} wg ` +
                    `${fmt(row.splitToBatched, 2)}x`,
                )
                .join(', ')
            : 'none') +
          ` — these ratios use hot weights and are not applied to cold time`,
      );

      console.log(
        `token ${fmt(decision.tokenMs, 3)} ms; bar ${fmt(decision.minShare * 100, 0)}% ` +
          `→ small ${decision.small.share >= decision.minShare ? 'passes' : 'fails'}, ` +
          `large (same-run best) ${decision.large.shareRunBest >= decision.minShare ? 'passes' : 'fails'}; ` +
          `lever in stack: small ${LEVER_IN_STACK.small ? 'yes' : 'no'}, ` +
          `large ${LEVER_IN_STACK.large ? 'yes' : 'no'}`,
      );
    }
  },

  ledger({ results, analyses }) {
    return { nodeCounts: NODES, tokenMs: TOKEN_MS, results, analyses };
  },
});
