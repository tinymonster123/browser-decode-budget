// Tests for the ledger analysis seam. Plain JSON in, plain numbers out:
// no browser, no GPU, no model.
//
//   node --test bench/analysis/*.test.mjs

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  dispatchSlope,
  bandwidthPlateau,
  utilization,
  compareModels,
  countsRun,
  timingRun,
  makeLedger,
  agreement,
  fusionCeiling,
  dispatchModeComparison,
} from './ledger.mjs';

function row(mode, count, wallMs, gpuMs = wallMs) {
  return { mode, count, wallMs, gpuMs, cpuMs: 0, samples: [] };
}

function modelResult(model, dispatch, ms) {
  return {
    model,
    variants: [
      {
        runs: [
          { cpu: false, perToken: { 'pass.dispatchWorkgroups': dispatch } },
          { cpu: true, decodeMsPerStep: ms },
        ],
      },
    ],
  };
}

test('dispatch slope: a synthetic linear sweep returns the slope exactly', () => {
  // 1 µs per dispatch, plus a fixed 0.25 ms of pass setup. The two largest
  // counts differ by exactly 1000 dispatches and 1 ms, and all four values
  // are exactly representable, so the slope is exact.
  const ledger = {
    result: {
      rows: [
        row('independent', 0, 0.25),
        row('independent', 100, 0.35),
        row('independent', 500, 0.75),
        row('independent', 1000, 1.25),
        row('independent', 2000, 2.25),
      ],
    },
  };

  const slope = dispatchSlope(ledger, { mode: 'independent' });

  assert.equal(slope.usPerDispatch, 1);
  assert.equal(slope.mode, 'independent');
  assert.equal(slope.points.length, 2);
  assert.deepEqual(
    slope.points.map((point) => point.count),
    [1000, 2000],
  );
});

test('dispatch slope: modes stay separate', () => {
  const ledger = {
    result: {
      rows: [
        row('independent', 500, 0.5),
        row('independent', 2000, 2),
        row('dependent', 500, 2.5),
        row('dependent', 2000, 10),
      ],
    },
  };

  assert.equal(dispatchSlope(ledger, { mode: 'independent' }).usPerDispatch, 1);
  assert.equal(dispatchSlope(ledger, { mode: 'dependent' }).usPerDispatch, 5);
});

test('dispatch slope: refuses a sweep with fewer than two usable points', () => {
  const ledger = { result: { rows: [row('dependent', 0, 0.1)] } };

  assert.throws(
    () => dispatchSlope(ledger, { mode: 'dependent' }),
    /two/i,
  );
});

test('bandwidth plateau: a cache effect then a flat tail returns the plateau', () => {
  // 400 GB/s while the buffer fits in cache, then a flat 100 GB/s plateau.
  const samples = [
    { bytes: 1e6, ms: 0.0025 },
    { bytes: 1e7, ms: 0.025 },
    { bytes: 1e8, ms: 1 },
    { bytes: 4e8, ms: 4 },
  ];

  const plateau = bandwidthPlateau(samples);

  assert.equal(plateau.gbps, 100);
  assert.equal(plateau.fromBytes, 1e8);
  assert.equal(plateau.samples.length, 2);
});

test('bandwidth plateau: a known constant plateau returns that plateau exactly', () => {
  const samples = [
    { bytes: 1e8, ms: 2 },
    { bytes: 2e8, ms: 4 },
    { bytes: 4e8, ms: 8 },
    { bytes: 8e8, ms: 16 },
  ];

  const plateau = bandwidthPlateau(samples);

  assert.equal(plateau.gbps, 50);
  assert.equal(plateau.fromBytes, 1e8);
});

test('bandwidth plateau: refuses a probe with fewer than two sizes', () => {
  assert.throws(
    () => bandwidthPlateau([{ bytes: 1e8, ms: 1 }]),
    /two/i,
  );
});

test('utilization: known bytes and GPU time return the measured percentage', () => {
  const measured = utilization({ bytes: 1e9, ms: 10 }, 200);

  assert.equal(measured.gbps, 100);
  assert.equal(measured.percent, 50);
});

test('utilization: refuses a zero ceiling instead of dividing by it', () => {
  assert.throws(
    () => utilization({ bytes: 1e9, ms: 10 }, 0),
    /ceiling/i,
  );
});

test('run roles: counts and timing are selected by role, not position', () => {
  const result = {
    model: 'base',
    variants: [
      {
        runs: [
          { cpu: true, decodeMsPerStep: 10 },
          { cpu: false, perToken: { 'pass.dispatchWorkgroups': 100 } },
        ],
      },
    ],
  };

  assert.equal(countsRun(result).perToken['pass.dispatchWorkgroups'], 100);
  assert.equal(timingRun(result).decodeMsPerStep, 10);
  assert.throws(() => countsRun({ model: 'gate2', variants: [{ runs: [] }] }), /counts-only run for gate2/);
});

test('makeLedger: one envelope shape, payload copied through', () => {
  const ledger = makeLedger({
    timestamp: '2026-10-19T00:00:00.000Z',
    commit: 'abc1234',
    loadavg: [1, 2, 3],
    config: { port: 8124 },
    results: [{ model: 'base' }],
  });

  assert.deepEqual(ledger, {
    timestamp: '2026-10-19T00:00:00.000Z',
    commit: 'abc1234',
    loadavg: [1, 2, 3],
    config: { port: 8124 },
    results: [{ model: 'base' }],
  });
});

test('makeLedger: a built ledger feeds the comparison seam', () => {
  const ledger = makeLedger({
    timestamp: '2026-10-19T00:00:00.000Z',
    commit: 'abc1234',
    loadavg: [1],
    config: {},
    nodeCounts: { base: 100, gate2: 50 },
    results: [modelResult('base', 100, 10), modelResult('gate2', 60, 8)],
  });

  const [comparison] = compareModels(ledger, { pairs: [['base', 'gate2']] });

  assert.equal(comparison.deltaDispatches, -40);
  assert.equal(comparison.usPerRemovedDispatch, 50);
});

test('compareModels: reports the dispatch and time deltas for a pair', () => {
  const ledger = {
    nodeCounts: { base: 100, gate2: 50 },
    results: [modelResult('base', 100, 10), modelResult('gate2', 60, 8)],
  };

  const [comparison] = compareModels(ledger, { pairs: [['base', 'gate2']] });

  assert.equal(comparison.deltaNodes, -50);
  assert.equal(comparison.deltaDispatches, -40);
  assert.equal(comparison.deltaMsPerStep, -2);
  assert.equal(comparison.nodeToDispatchRatio, 1.25);
  assert.equal(comparison.usPerRemovedDispatch, 50);
});

test('compareModels: a missing cpu-timed run is an error, not a fallback', () => {
  const ledger = {
    nodeCounts: { base: 100, gate2: 50 },
    results: [
      {
        model: 'base',
        variants: [
          {
            runs: [
              { cpu: false, perToken: { 'pass.dispatchWorkgroups': 100 } },
            ],
          },
        ],
      },
      {
        model: 'gate2',
        variants: [
          {
            runs: [
              { cpu: false, perToken: { 'pass.dispatchWorkgroups': 60 } },
              { cpu: true, decodeMsPerStep: 8 },
            ],
          },
        ],
      },
    ],
  };

  assert.throws(
    () => compareModels(ledger, { pairs: [['base', 'gate2']] }),
    /cpu-timed run for base/,
  );
});

test('compareModels: a model missing from the ledger is reported, not guessed', () => {
  const ledger = {
    nodeCounts: { base: 100 },
    results: [{ model: 'base', variants: [{ runs: [{ cpu: false, perToken: {} }] }] }],
  };

  const [comparison] = compareModels(ledger, { pairs: [['base', 'gate2']] });

  assert.equal(comparison.error, 'gate2 not found in ledger');
});

test('dispatch slope: a sweep filter keeps two runs of the same sweep apart', () => {
  // Two sweeps of an identical dependent chain. If the filter were ignored,
  // the last two rows would share count 2000 and the slope would refuse.
  const ledger = {
    result: {
      rows: [
        { ...row('dependent', 1000, 3), sweep: 0 },
        { ...row('dependent', 2000, 5.5), sweep: 0 },
        { ...row('dependent', 1000, 3.5), sweep: 1 },
        { ...row('dependent', 2000, 6.5), sweep: 1 },
      ],
    },
  };

  assert.equal(
    dispatchSlope(ledger, { mode: 'dependent', sweep: 0 }).usPerDispatch,
    2.5,
  );
  assert.equal(
    dispatchSlope(ledger, { mode: 'dependent', sweep: 1 }).usPerDispatch,
    3,
  );
});

test('dispatch slope: the timing field is selectable', () => {
  const ledger = {
    result: {
      rows: [
        { ...row('dependent', 1000, 3), gpuMs: 2.5, sweep: 0 },
        { ...row('dependent', 2000, 5.5), gpuMs: 4.5, sweep: 0 },
      ],
    },
  };

  assert.equal(
    dispatchSlope(ledger, { mode: 'dependent', field: 'gpuMs' }).usPerDispatch,
    2,
  );
  assert.equal(
    dispatchSlope(ledger, { mode: 'dependent' }).usPerDispatch,
    2.5,
  );
});

test('agreement: 2.0 vs 2.2 µs is within 10%, 2.0 vs 2.4 is not', () => {
  const close = agreement(2.0, 2.2);
  assert.equal(close.within, true);
  assert.ok(Math.abs(close.deltaPercent - 9.5238) < 0.001);
  assert.ok(Math.abs(close.mean - 2.1) < 1e-12);

  const far = agreement(2.0, 2.4);
  assert.equal(far.within, false);
  assert.ok(Math.abs(far.deltaPercent - 18.1818) < 0.001);
});

test('agreement: refuses a zero measurement instead of reporting it', () => {
  assert.throws(() => agreement(0, 2), /positive/i);
  assert.throws(() => agreement(2, Number.NaN), /positive/i);
});

test('fusion ceiling: a known serialized slope multiplied by the dispatch counts', () => {
  // 2.5 µs per dispatch: 211 dispatches = 0.5275 ms, 1112 = 2.78 ms.
  const ceilings = fusionCeiling({
    usPerDispatch: 2.5,
    dispatchCounts: [211, 1112],
  });

  assert.deepEqual(ceilings, [
    { dispatches: 211, ms: 0.5275 },
    { dispatches: 1112, ms: 2.78 },
  ]);
});

test('fusion ceiling: refuses a non-positive slope or an empty count list', () => {
  assert.throws(
    () => fusionCeiling({ usPerDispatch: 0, dispatchCounts: [211] }),
    /positive/i,
  );
  assert.throws(
    () => fusionCeiling({ usPerDispatch: 2, dispatchCounts: [] }),
    /count/i,
  );
});

test('dispatch mode comparison: dependent over independent ratio', () => {
  // The machine's measured shape: 1.31 µs independent, 2.50 µs serialized.
  const comparison = dispatchModeComparison({
    dependentUs: 2.5,
    independentUs: 1.31,
  });

  assert.equal(comparison.ratio, 2.5 / 1.31);
  assert.equal(comparison.extraUs, 1.19);
  assert.equal(comparison.dependentExceedsIndependent, true);
});

test('dispatch mode comparison: a dependent chain that is not slower is visible', () => {
  const comparison = dispatchModeComparison({
    dependentUs: 1.2,
    independentUs: 1.31,
  });

  assert.equal(comparison.dependentExceedsIndependent, false);
  assert.ok(comparison.ratio < 1);
});

test('the existing experiment ledgers still fit the seam unchanged', async () => {
  const { readFile } = await import('node:fs/promises');

  const count = JSON.parse(
    await readFile(
      new URL('../../runs/2026-10-19-exp2-dispatch-count.json', import.meta.url),
      'utf8',
    ),
  );
  const microbench = JSON.parse(
    await readFile(
      new URL('../../runs/2026-10-19-exp3-dispatch-cost.json', import.meta.url),
      'utf8',
    ),
  );

  const [comparison] = compareModels(count, { pairs: [['base', 'gate2']] });

  assert.equal(comparison.error, undefined);
  assert.ok(Number.isFinite(comparison.deltaDispatches));
  assert.ok(Number.isFinite(comparison.deltaMsPerStep));

  const slope = dispatchSlope(microbench, { mode: 'one-pass' });
  assert.ok(slope.usPerDispatch > 0);
});
