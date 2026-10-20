// The ledger seam.
//
// Plain JSON in, plain numbers out. Nothing here touches a browser, a GPU or
// a model, so the numbers used for a decision can be re-derived from the
// ledgers in `runs/` and checked with `node --test`.
//
// Ledger shapes this module reads:
//
//   microbench  { result: { rows: [{ mode, sweep, count, wallMs, gpuMs, ... }] } }
//   count       { nodeCounts: { model: n }, results: [{ model, variants: [{ runs }] }] }
//   profile     { results: [{ model, patchInfo: { featureInjected }, runs: [{ coverage, segments }] }] }
//   attribution { results: [{ model, runs: [{ step, segments: [{ us, dispatches, pipelines }] }] }] }
//   bandwidth   [{ bytes, ms }]
//
// A microbench ledger may hold several `sweep`s of the same mode; callers
// select one sweep explicitly so two repetitions never mix in one fit.
//
// `makeLedger` is the one constructor for the envelope every driver writes;
// its timestamp is injectable so tests stay deterministic.

export const DISPATCH_COUNTER = 'pass.dispatchWorkgroups';
const PLATEAU_TOLERANCE = 0.1;

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

function assertPositive(value, name) {
  if (!(Number.isFinite(value) && value > 0)) {
    throw new Error(`${name} must be a positive number, got ${value}`);
  }
}

// Slope between the two largest dispatch counts, in microseconds per
// dispatch. Rows with count 0 carry the fixed pass setup and are excluded.
//
// `sweep` selects one repetition of the whole sweep when a ledger contains
// several; without it two repetitions would contribute rows with the same
// count and the fit would either be ambiguous or fail.
// `field` selects the timing channel: wall (encode + submit + GPU) or gpu
// (submit-to-completion only).
export function dispatchSlope(ledger, { mode, sweep, field = 'wallMs' } = {}) {
  const rows = (ledger?.result?.rows ?? ledger?.rows ?? []).filter(
    (row) =>
      row.mode === mode &&
      (sweep == null || row.sweep === sweep) &&
      Number.isFinite(row.count) &&
      row.count > 0 &&
      Number.isFinite(row[field]),
  );

  if (rows.length < 2) {
    throw new Error(
      `dispatch slope needs at least two usable points for mode "${mode}"` +
        `${sweep == null ? '' : ` sweep ${sweep}`} in field "${field}", ` +
        `got ${rows.length}`,
    );
  }

  const [first, second] = [...rows]
    .sort((a, b) => a.count - b.count)
    .slice(-2);

  if (second.count === first.count) {
    throw new Error(`dispatch slope cannot be fit: duplicate count ${first.count}`);
  }

  return {
    mode,
    sweep: sweep ?? null,
    field,
    usPerDispatch:
      ((second[field] - first[field]) * 1000) / (second.count - first.count),
    points: [
      { count: first.count, [field]: first[field] },
      { count: second.count, [field]: second[field] },
    ],
  };
}

// Two repetitions of the same measurement. A number that moves more than
// `tolerance` (a fraction) between them cannot be used for a decision.
export function agreement(a, b, { tolerance = 0.1 } = {}) {
  assertPositive(a, 'first measurement');
  assertPositive(b, 'second measurement');

  const mean = (a + b) / 2;
  const deltaPercent = (Math.abs(a - b) / mean) * 100;

  return {
    a,
    b,
    mean,
    deltaPercent,
    tolerance,
    within: deltaPercent / 100 <= tolerance,
  };
}

// What removing N dispatches would save at a measured per-dispatch cost.
// The result is in milliseconds because the decode budget is in milliseconds.
export function fusionCeiling({ usPerDispatch, dispatchCounts }) {
  assertPositive(usPerDispatch, 'usPerDispatch');
  if (!Array.isArray(dispatchCounts) || dispatchCounts.length === 0) {
    throw new Error('fusion ceiling needs at least one dispatch count');
  }

  return dispatchCounts.map((dispatches) => {
    assertPositive(dispatches, 'dispatch count');
    return { dispatches, ms: (usPerDispatch * dispatches) / 1000 };
  });
}

// The serialized and independent slopes from the same harness. `ratio` is how
// many times more expensive a dispatch is when the next one waits for it.
export function dispatchModeComparison({ dependentUs, independentUs }) {
  assertPositive(dependentUs, 'dependent dispatch cost');
  assertPositive(independentUs, 'independent dispatch cost');

  return {
    dependentUs,
    independentUs,
    ratio: dependentUs / independentUs,
    extraUs: dependentUs - independentUs,
    dependentExceedsIndependent: dependentUs > independentUs,
  };
}

// Bandwidth plateau from a size sweep.
//
// Throughput is bytes / ms / 1e6 GB/s. The plateau is the first size from
// which every remaining sample stays within 10% of it, so a fast
// cache-resident region at small sizes cannot be mistaken for the plateau.
export function bandwidthPlateau(samples) {
  if (!Array.isArray(samples) || samples.length < 2) {
    throw new Error('bandwidth plateau needs at least two size samples');
  }

  const sorted = [...samples].sort((a, b) => a.bytes - b.bytes);
  for (const sample of sorted) {
    assertPositive(sample.bytes, 'bandwidth sample bytes');
    assertPositive(sample.ms, 'bandwidth sample ms');
  }

  const gbps = sorted.map((sample) => sample.bytes / sample.ms / 1e6);

  for (let start = 0; start <= sorted.length - 2; start++) {
    const base = gbps[start];
    const tail = gbps.slice(start);
    if (tail.every((value) => Math.abs(value - base) / base <= PLATEAU_TOLERANCE)) {
      return {
        gbps: median(tail),
        fromBytes: sorted[start].bytes,
        samples: sorted.slice(start),
      };
    }
  }

  throw new Error(
    `bandwidth probe has no plateau within ${(PLATEAU_TOLERANCE * 100).toFixed(0)}%`,
  );
}

// The two sweep plateaus from one probe run. The ceiling is their mean; the
// reported start is the later of the two sizes, so a sweep that needs a larger
// buffer before it plateaus is not overruled by the other.
export function combinePlateaus(plateaus) {
  if (!Array.isArray(plateaus) || plateaus.length !== 2) {
    throw new Error(
      `combining plateaus needs exactly two sweeps, got ${plateaus?.length ?? 0}`,
    );
  }
  for (const plateau of plateaus) {
    assertPositive(plateau?.gbps, 'plateau gbps');
    assertPositive(plateau?.fromBytes, 'plateau fromBytes');
  }

  return {
    gbps: (plateaus[0].gbps + plateaus[1].gbps) / 2,
    fromBytes: Math.max(plateaus[0].fromBytes, plateaus[1].fromBytes),
    perSweep: plateaus,
    agreement: agreement(plateaus[0].gbps, plateaus[1].gbps),
  };
}

// How much of the measured ceiling one transfer uses.
export function utilization({ bytes, ms }, ceilingGbps) {
  assertPositive(ceilingGbps, 'bandwidth ceiling');
  assertPositive(bytes, 'transfer bytes');
  assertPositive(ms, 'transfer ms');

  const gbps = bytes / ms / 1e6;
  return { gbps, percent: (gbps / ceilingGbps) * 100 };
}

// The Gate 5 profile is only usable for utilization when its per-pass times
// came from timestamp-query. If the feature was unavailable, the page says so
// in `patchInfo` and the run has no coverage; refuse rather than substitute a
// wall-time estimate.
export function requireTimestampQuery(profile) {
  const results = profile?.results;
  if (!Array.isArray(results) || results.length === 0) {
    throw new Error('profile ledger has no results');
  }

  for (const result of results) {
    if (result?.patchInfo?.featureInjected !== true) {
      throw new Error(
        `${result?.model ?? 'model'}: timestamp-query was not available; ` +
          'refusing to compute utilization',
      );
    }
    const runs = result.runs ?? [];
    if (runs.length === 0 || runs.some((run) => !(run.coverage > 0))) {
      throw new Error(
        `${result.model}: no timestamp coverage; refusing to compute utilization`,
      );
    }
  }

  return results;
}

// The LM head is the third-from-last compute pass: one MatMulNBits dispatch
// between the final norm and the softcap tail. The Gate 5 report identified it
// by position; the selection is validated against the largest single-dispatch
// pass so a graph change cannot silently pick a different pass. The per-model
// value is the median over the recorded decode steps.
export function lmHeadPasses(profile) {
  return requireTimestampQuery(profile).map((result) => {
    const steps = (result.runs ?? []).map((run) => {
      const segments = run.segments ?? [];
      if (segments.length < 3) {
        throw new Error(
          `${result.model} step ${run.step}: fewer than three compute passes`,
        );
      }

      const index = segments.length - 3;
      const lm = segments[index];
      if (lm.dispatches !== 1) {
        throw new Error(
          `${result.model} step ${run.step}: expected one dispatch at pass ` +
            `${index}, got ${lm.dispatches}`,
        );
      }

      const largest = segments
        .filter((segment) => segment.dispatches === 1)
        .reduce((best, segment) => (segment.us > best.us ? segment : best));
      if (largest !== lm) {
        throw new Error(
          `${result.model} step ${run.step}: the third-from-last pass is not ` +
            'the largest single-dispatch pass',
        );
      }

      return { step: run.step, index, us: lm.us, dispatches: lm.dispatches };
    });

    return {
      model: result.model,
      medianUs: median(steps.map((step) => step.us)),
      steps,
    };
  });
}

// What fraction of the measured bandwidth ceiling the LM head reaches, using
// the recorded per-pass GPU time and each model's recorded weight bytes. A
// model without a recorded byte count is refused rather than borrowing
// another model's number.
export function lmHeadUtilization(profile, { bytesByModel, ceilingGbps } = {}) {
  if (bytesByModel == null) {
    throw new Error('LM head utilization needs bytesByModel');
  }

  return lmHeadPasses(profile).map((pass) => {
    const bytes = bytesByModel[pass.model];
    if (!Number.isFinite(bytes)) {
      throw new Error(`no recorded weight bytes for model "${pass.model}"`);
    }

    return {
      model: pass.model,
      medianUs: pass.medianUs,
      bytes,
      steps: pass.steps,
      ...utilization({ bytes, ms: pass.medianUs / 1000 }, ceilingGbps),
    };
  });
}

function runsOf(result) {
  return result?.variants?.[0]?.runs ?? [];
}

const RUN_ROLES = {
  counts: { cpu: false, label: 'counts-only' },
  timing: { cpu: true, label: 'cpu-timed' },
};

function requireRun(result, role) {
  const model = result?.model ?? 'model';
  const run = runsOf(result).find((candidate) => candidate.cpu === role.cpu);
  if (!run) {
    throw new Error(`ledger has no ${role.label} run for ${model}`);
  }
  return run;
}

// The two run roles every count ledger writes. Callers select by role, never
// by position, so a missing run cannot silently become the wrong run.
export function countsRun(result) {
  return requireRun(result, RUN_ROLES.counts);
}

export function timingRun(result) {
  return requireRun(result, RUN_ROLES.timing);
}

// The envelope every driver writes: when, from which commit, under what load,
// and with what configuration. Extra payload fields are copied through. This
// is the seam's only exported constructor; everything else reads.
export function makeLedger({
  timestamp = new Date().toISOString(),
  commit,
  loadavg,
  config,
  ...fields
}) {
  return { timestamp, commit, loadavg, config, ...fields };
}

// Per-model comparison of a count ledger: dispatch and node deltas, the
// measured milliseconds per decode step, and the microseconds attributable to
// each removed dispatch. `b - a` is the direction of every delta, so a
// positive `usPerRemovedDispatch` means the removal saved time.
export function compareModels(ledger, { pairs = [] } = {}) {
  return pairs.map(([a, b]) => {
    const missing = !ledger?.results?.find((result) => result.model === a)
      ? a
      : !ledger?.results?.find((result) => result.model === b)
        ? b
        : null;
    if (missing) {
      return { a, b, error: `${missing} not found in ledger` };
    }

    const resultA = ledger.results.find((result) => result.model === a);
    const resultB = ledger.results.find((result) => result.model === b);

    const countsA = countsRun(resultA).perToken?.[DISPATCH_COUNTER];
    const countsB = countsRun(resultB).perToken?.[DISPATCH_COUNTER];
    if (!Number.isFinite(countsA) || !Number.isFinite(countsB)) {
      throw new Error(`ledger has no "${DISPATCH_COUNTER}" counter for ${a} or ${b}`);
    }

    const msA = timingRun(resultA).decodeMsPerStep;
    const msB = timingRun(resultB).decodeMsPerStep;
    if (!Number.isFinite(msA) || !Number.isFinite(msB)) {
      throw new Error(`ledger has no decodeMsPerStep for ${a} or ${b}`);
    }

    const nodesA = ledger?.nodeCounts?.[a];
    const nodesB = ledger?.nodeCounts?.[b];
    const deltaNodes =
      Number.isFinite(nodesA) && Number.isFinite(nodesB) ? nodesB - nodesA : null;
    const deltaDispatches = countsB - countsA;
    const deltaMsPerStep = msB - msA;

    return {
      a,
      b,
      deltaNodes,
      deltaDispatches,
      deltaMsPerStep,
      nodeToDispatchRatio:
        deltaNodes != null && deltaDispatches !== 0
          ? deltaNodes / deltaDispatches
          : null,
      usPerRemovedDispatch:
        deltaDispatches !== 0 ? (deltaMsPerStep * 1000) / deltaDispatches : null,
    };
  });
}

// Gate 7: attribute each compute pass to the pipeline identity that issued it.
//
// A pass that binds several pipelines (a loop inside one encoder) is split
// evenly, because the GPU timestamps bound the whole pass, not each dispatch.
// Identity keys are opaque: "label:<ORT label>", "shader:<hash>:<entry>",
// "pipeline:<n>" or "unlabeled". This function never guesses an op name; it
// groups by whatever identity the page could observe.
export function attributeOps(profile, { perDispatchOverheadUs = 0 } = {}) {
  const runs = [...(profile?.runs ?? [])];
  if (!runs.length) throw new Error('attributeOps: no usable decode steps');

  const classes = new Map();
  const stepsOut = [];
  let mixedPasses = 0;

  const add = (target, us, passes, dispatches) => {
    target.us += us;
    target.passes += passes;
    target.dispatches += dispatches;
  };

  for (const run of runs) {
    const step = run.step;
    let totalUs = 0;
    let passes = 0;

    for (const segment of run.segments ?? []) {
      const us = Number(segment.us);
      if (!Number.isFinite(us) || us < 0) continue;
      const keys = [...new Set((segment.pipelines ?? []).filter(Boolean))];
      const divisor = keys.length || 1;
      const share = us / divisor;
      const dispatchShare = (segment.dispatches ?? 0) / divisor;

      totalUs += us;
      passes += 1;
      if (keys.length > 1) mixedPasses += 1;

      for (const key of keys.length ? keys : ['unlabeled']) {
        let entry = classes.get(key);
        if (!entry) {
          entry = {
            key,
            label: labelOf(key),
            us: 0,
            passes: 0,
            dispatches: 0,
            perStep: new Map(),
          };
          classes.set(key, entry);
        }
        add(entry, share, 1 / divisor, dispatchShare);

        const row = entry.perStep.get(step) ?? {
          step,
          us: 0,
          passes: 0,
          dispatches: 0,
        };
        add(row, share, 1 / divisor, dispatchShare);
        entry.perStep.set(step, row);
      }
    }

    stepsOut.push({ step, totalUs, passes });
  }

  const stepTotals = new Map(stepsOut.map((row) => [row.step, row.totalUs]));
  const totalUs = stepsOut.reduce((sum, row) => sum + row.totalUs, 0);
  const stepCount = runs.length;
  const unlabeledUs = classes.get('unlabeled')?.us ?? 0;

  // Only a label or op key names the kernel. A shader hash or a pipeline
  // ordinal is an identity, not a name, so it does not count as coverage.
  const namedUs = [...classes.values()]
    .filter((entry) => entry.key.startsWith('label:') || entry.key.startsWith('op:'))
    .reduce((sum, entry) => sum + entry.us, 0);

  const rows = [...classes.values()]
    .map((entry) => {
      const perStep = [...entry.perStep.values()]
        .map((row) => ({
          ...row,
          share: stepTotals.get(row.step) ? row.us / stepTotals.get(row.step) : null,
        }))
        .sort((a, b) => a.step - b.step);

      const values = perStep.map((row) => row.us);
      const shares = perStep.map((row) => row.share ?? 0);
      const meanUs = entry.us / stepCount;
      const dispatchesPerStep = entry.dispatches / stepCount;
      // One pass now holds one dispatch, so every dispatch also pays a pass
      // boundary. This is an estimate built on the Gate 4 wall-clock slope;
      // the raw value stays the measured fact.
      const correctedUsPerStep = Math.max(
        0,
        meanUs - perDispatchOverheadUs * dispatchesPerStep,
      );
      const minUs = values.length ? Math.min(...values) : 0;
      const maxUs = values.length ? Math.max(...values) : 0;
      const minShare = shares.length ? Math.min(...shares) : 0;
      const maxShare = shares.length ? Math.max(...shares) : 0;
      return {
        key: entry.key,
        label: entry.label,
        usPerStep: meanUs,
        correctedUsPerStep,
        passesPerStep: entry.passes / stepCount,
        dispatchesPerStep,
        share: totalUs ? entry.us / totalUs : null,
        stability:
          values.length > 1
            ? {
                minUs,
                maxUs,
                deltaPercent: meanUs ? (100 * (maxUs - minUs)) / meanUs : null,
              }
            : null,
        shareStability:
          shares.length > 1
            ? {
                min: minShare,
                max: maxShare,
                deltaPoints: 100 * (maxShare - minShare),
              }
            : null,
        perStep,
      };
    })
    .sort((a, b) => b.usPerStep - a.usPerStep);

  const totalCorrectedUs = rows.reduce((sum, row) => sum + row.correctedUsPerStep, 0);
  for (const row of rows) {
    row.correctedShare = totalCorrectedUs
      ? row.correctedUsPerStep / totalCorrectedUs
      : null;
  }

  return {
    steps: stepsOut,
    classes: rows,
    perDispatchOverheadUs,
    totalUsPerStep: totalUs / stepCount,
    totalCorrectedUsPerStep: totalCorrectedUs,
    identityCoverage: totalUs ? Math.min(1, namedUs / totalUs) : null,
    unlabeledShare: totalUs ? unlabeledUs / totalUs : null,
    mixedPasses,
  };
}

// Identity grammar, produced by the browser pages and parsed here:
//   label:<ORT pipeline label>      (empty in this ORT build)
//   op:<shader module label>         the op name, e.g. "MatMulNBits"
//   shader:<code hash>:<entryPoint>  unnamed fallback
//   pipeline:<ordinal>               unnamed fallback
//   unlabeled                        no identity at all
function labelOf(key) {
  if (key.startsWith('label:')) return key.slice('label:'.length);
  if (key.startsWith('op:')) return key.slice('op:'.length);
  if (key.startsWith('shader:')) return key.split(':')[1] ?? key;
  return key;
}
