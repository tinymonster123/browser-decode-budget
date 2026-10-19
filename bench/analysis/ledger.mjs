// The ledger analysis seam for Gate 6.
//
// Plain JSON in, plain numbers out. Nothing here touches a browser, a GPU or
// a model, so the numbers used for the decision can be re-derived from the
// ledgers in `runs/` and checked with `node --test`.
//
// Ledger shapes this module reads:
//
//   microbench  { result: { rows: [{ mode, count, wallMs, ... }] } }
//   count       { nodeCounts: { model: n }, results: [{ model, variants: [{ runs }] }] }
//   bandwidth   [{ bytes, ms }]   (the probe in ticket #4 will report this array)

const DISPATCH_COUNTER = 'pass.dispatchWorkgroups';
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
export function dispatchSlope(ledger, { mode } = {}) {
  const rows = (ledger?.result?.rows ?? ledger?.rows ?? []).filter(
    (row) =>
      row.mode === mode &&
      Number.isFinite(row.count) &&
      row.count > 0 &&
      Number.isFinite(row.wallMs),
  );

  if (rows.length < 2) {
    throw new Error(
      `dispatch slope needs at least two usable points for mode "${mode}", got ${rows.length}`,
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
    usPerDispatch:
      ((second.wallMs - first.wallMs) * 1000) / (second.count - first.count),
    points: [
      { count: first.count, wallMs: first.wallMs },
      { count: second.count, wallMs: second.wallMs },
    ],
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

// How much of the measured ceiling one transfer uses.
export function utilization({ bytes, ms }, ceilingGbps) {
  assertPositive(ceilingGbps, 'bandwidth ceiling');
  assertPositive(bytes, 'transfer bytes');
  assertPositive(ms, 'transfer ms');

  const gbps = bytes / ms / 1e6;
  return { gbps, percent: (gbps / ceilingGbps) * 100 };
}

function runsOf(result) {
  return result?.variants?.[0]?.runs ?? [];
}

function requireRun(result, predicate, model, kind) {
  const run = runsOf(result).find(predicate);
  if (!run) {
    throw new Error(`ledger has no ${kind} run for ${model}`);
  }
  return run;
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

    const countsA = requireRun(resultA, (run) => run.cpu === false, a, 'counts-only')
      .perToken?.[DISPATCH_COUNTER];
    const countsB = requireRun(resultB, (run) => run.cpu === false, b, 'counts-only')
      .perToken?.[DISPATCH_COUNTER];
    if (!Number.isFinite(countsA) || !Number.isFinite(countsB)) {
      throw new Error(`ledger has no "${DISPATCH_COUNTER}" counter for ${a} or ${b}`);
    }

    const msA = requireRun(resultA, (run) => run.cpu === true, a, 'cpu-timed')
      .decodeMsPerStep;
    const msB = requireRun(resultB, (run) => run.cpu === true, b, 'cpu-timed')
      .decodeMsPerStep;
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
