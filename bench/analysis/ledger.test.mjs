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
  requireTimestampQuery,
  lmHeadPasses,
  lmHeadUtilization,
  combinePlateaus,
  attributeOps,
  attributeMatmuls,
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

function profileLedger({
  lmUs = [2600, 2700, 3600],
  featureInjected = true,
  coverage = 0.5,
} = {}) {
  return {
    results: [
      {
        model: 'base',
        patchInfo: { featureInjected },
        runs: lmUs.map((us, step) => ({
          step,
          coverage,
          segments: [
            { seq: step * 5, us: 40, dispatches: 1 },
            { seq: step * 5 + 1, us: 30, dispatches: 3 },
            // third from last: the single-dispatch LM head
            { seq: step * 5 + 2, us, dispatches: 1 },
            { seq: step * 5 + 3, us: 13, dispatches: 2 },
            { seq: step * 5 + 4, us: 4, dispatches: 1 },
          ],
        })),
      },
    ],
  };
}

test('timestamp gate: a profile without timestamp-query is refused', () => {
  assert.throws(
    () => requireTimestampQuery(profileLedger({ featureInjected: false })),
    /timestamp-query/,
  );
  assert.throws(
    () => requireTimestampQuery({ results: [] }),
    /no results/i,
  );
});

test('timestamp gate: a profile with zero coverage is refused', () => {
  assert.throws(
    () => requireTimestampQuery(profileLedger({ coverage: 0 })),
    /coverage/,
  );
});

test('lm head pass: the third-from-last single dispatch is selected per step', () => {
  const passes = lmHeadPasses(profileLedger());

  assert.equal(passes.length, 1);
  assert.equal(passes[0].model, 'base');
  assert.equal(passes[0].medianUs, 2700);
  assert.deepEqual(
    passes[0].steps.map((step) => step.us),
    [2600, 2700, 3600],
  );
  assert.ok(passes[0].steps.every((step) => step.index === 2));
});

test('lm head pass: a third-from-last pass with two dispatches is refused', () => {
  const ledger = profileLedger();
  ledger.results[0].runs[0].segments[2].dispatches = 2;

  assert.throws(() => lmHeadPasses(ledger), /one dispatch/);
});

test('lm head pass: a larger single-dispatch earlier in the step is refused', () => {
  const ledger = profileLedger();
  ledger.results[0].runs[0].segments[0].us = 9999;

  assert.throws(() => lmHeadPasses(ledger), /largest single-dispatch/);
});

test('lm head utilization: bytes over measured GPU time, against the ceiling', () => {
  // 232,783,872 bytes in 2.7 ms = 86.2 GB/s; against a 120 GB/s ceiling.
  const [entry] = lmHeadUtilization(profileLedger(), {
    bytesByModel: { base: 232783872 },
    ceilingGbps: 120,
  });

  assert.equal(entry.model, 'base');
  assert.equal(entry.medianUs, 2700);
  assert.equal(entry.bytes, 232783872);
  assert.deepEqual(
    entry.steps.map((step) => step.us),
    [2600, 2700, 3600],
  );
  assert.ok(Math.abs(entry.gbps - 86.216249) < 0.001);
  assert.ok(Math.abs(entry.percent - 71.846874) < 0.001);
});

test('lm head utilization: refuses to estimate when timestamps are missing', () => {
  assert.throws(
    () =>
      lmHeadUtilization(profileLedger({ featureInjected: false }), {
        bytesByModel: { base: 232783872 },
        ceilingGbps: 120,
      }),
    /timestamp-query/,
  );
});

test('lm head utilization: refuses a model with no recorded weight bytes', () => {
  assert.throws(
    () =>
      lmHeadUtilization(profileLedger(), {
        bytesByModel: { gate1: 1 },
        ceilingGbps: 120,
      }),
    /no recorded weight bytes for model "base"/,
  );
});

test('combine plateaus: the mean ceiling and the later start', () => {
  const combined = combinePlateaus([
    { gbps: 109.01, fromBytes: 8 * 1024 * 1024 },
    { gbps: 109.44, fromBytes: 16 * 1024 * 1024 },
  ]);

  assert.ok(Math.abs(combined.gbps - 109.225) < 0.001);
  assert.equal(combined.fromBytes, 16 * 1024 * 1024);
  assert.equal(combined.agreement.within, true);
  assert.equal(combined.perSweep.length, 2);
});

test('combine plateaus: exactly two sweeps, both positive', () => {
  assert.throws(() => combinePlateaus([]), /two/i);
  assert.throws(
    () => combinePlateaus([{ gbps: 1, fromBytes: 1 }]),
    /two/i,
  );
  assert.throws(
    () =>
      combinePlateaus([
        { gbps: 0, fromBytes: 1 },
        { gbps: 1, fromBytes: 1 },
      ]),
    /positive/i,
  );
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

// Gate 7: pass → identity attribution. The identity is whatever the page
// observed; the analysis only groups and shares time.
test('attributeOps: groups by identity and reports unlabeled coverage', () => {
  const profile = {
    runs: [
      {
        step: 0,
        segments: [
          { us: 1000, dispatches: 10, pipelines: ['label:MatMulNBits'] },
          { us: 500, dispatches: 1, pipelines: [] },
        ],
      },
    ],
  };

  const out = attributeOps(profile);

  assert.equal(out.steps[0].totalUs, 1500);
  assert.equal(out.steps[0].passes, 2);
  assert.equal(out.totalUsPerStep, 1500);
  assert.equal(out.classes[0].key, 'label:MatMulNBits');
  assert.equal(out.classes[0].label, 'MatMulNBits');
  assert.equal(out.classes[0].usPerStep, 1000);
  assert.equal(out.classes[1].key, 'unlabeled');
  assert.equal(out.identityCoverage, 1000 / 1500);
  assert.equal(out.unlabeledShare, 500 / 1500);
  assert.equal(out.mixedPasses, 0);
});

test('attributeOps: a pass binding several pipelines is split evenly', () => {
  const profile = {
    runs: [
      {
        step: 0,
        segments: [
          { us: 100, dispatches: 4, pipelines: ['label:A', 'label:B', 'label:A'] },
        ],
      },
    ],
  };

  const out = attributeOps(profile);
  const byKey = Object.fromEntries(out.classes.map((row) => [row.key, row]));

  assert.equal(byKey['label:A'].usPerStep, 50);
  assert.equal(byKey['label:B'].usPerStep, 50);
  assert.equal(byKey['label:A'].dispatchesPerStep, 2);
  assert.equal(byKey['label:B'].passesPerStep, 0.5);
  assert.equal(out.mixedPasses, 1);
  assert.equal(out.identityCoverage, 1);
});

test('attributeOps: per-step means and stability across steps', () => {
  const profile = {
    runs: [
      {
        step: 0,
        segments: [
          { us: 100, dispatches: 1, pipelines: ['label:A'] },
          { us: 300, dispatches: 3, pipelines: ['label:B'] },
        ],
      },
      {
        step: 1,
        segments: [
          { us: 200, dispatches: 1, pipelines: ['label:A'] },
          { us: 300, dispatches: 3, pipelines: ['label:B'] },
        ],
      },
    ],
  };

  const out = attributeOps(profile);
  const byKey = Object.fromEntries(out.classes.map((row) => [row.key, row]));

  assert.equal(byKey['label:A'].usPerStep, 150);
  assert.ok(Math.abs(byKey['label:A'].stability.deltaPercent - (100 / 150) * 100) < 1e-9);
  assert.equal(byKey['label:B'].usPerStep, 300);
  assert.equal(byKey['label:B'].stability.deltaPercent, 0);
  assert.deepEqual(byKey['label:A'].perStep.map((row) => row.step), [0, 1]);
  assert.deepEqual(
    byKey['label:A'].perStep.map((row) => row.share),
    [0.25, 0.4],
  );
  assert.equal(byKey['label:A'].shareStability.min, 0.25);
  assert.equal(byKey['label:A'].shareStability.max, 0.4);
});

test('attributeOps: refuses an empty profile instead of returning zero rows', () => {
  assert.throws(() => attributeOps({ runs: [] }), /no usable decode steps/);
});

test('attributeOps: subtracts the per-dispatch pass boundary, not below zero', () => {
  const profile = {
    runs: [
      {
        step: 0,
        segments: [
          { us: 10, dispatches: 1, pipelines: ['op:A'] },
          { us: 1, dispatches: 1, pipelines: ['op:B'] },
        ],
      },
    ],
  };

  const out = attributeOps(profile, { perDispatchOverheadUs: 2 });
  const byKey = Object.fromEntries(out.classes.map((row) => [row.key, row]));

  assert.equal(out.perDispatchOverheadUs, 2);
  assert.equal(byKey['op:A'].correctedUsPerStep, 8);
  assert.equal(byKey['op:B'].correctedUsPerStep, 0);
  assert.equal(out.totalCorrectedUsPerStep, 8);
  assert.ok(Math.abs(byKey['op:A'].share - 10 / 11) < 1e-9);
  assert.equal(byKey['op:A'].correctedShare, 1);
});

test('attributeOps: a shader hash is an identity but not a named op', () => {
  const profile = {
    runs: [
      {
        step: 0,
        segments: [
          { us: 50, dispatches: 1, pipelines: ['op:MatMulNBits'] },
          { us: 25, dispatches: 1, pipelines: ['shader:abcd1234:main'] },
          { us: 25, dispatches: 1, pipelines: [] },
        ],
      },
    ],
  };

  const out = attributeOps(profile);

  assert.equal(out.identityCoverage, 0.5);
  assert.equal(out.unlabeledShare, 0.25);
});

test('attributeMatmuls: groups matmul time by the weight fingerprint it binds', () => {
  const profile = {
    runs: [
      {
        step: 0,
        segments: [
          { us: 10000, dispatches: 1, pipelines: ['op:MatMulNBits'], buffers: [1e9, 4096] },
          { us: 2000, dispatches: 1, pipelines: ['op:MatMulNBits'], buffers: [2e6, 4096] },
          { us: 500, dispatches: 1, pipelines: ['op:MatMulNBits'], buffers: [123, 4096] },
          { us: 700, dispatches: 1, pipelines: ['op:MatMulNBits'], buffers: [1000, 500, 4096] },
          { us: Number.NaN, dispatches: 1, pipelines: ['op:MatMulNBits'], buffers: [1e9] },
          { us: 900, dispatches: 1, pipelines: ['op:LayerNorm'], buffers: [1e9] },
        ],
      },
    ],
  };
  const fingerprints = [
    { sizes: [1e9], bytes: 1e9, nodes: 1 },
    { sizes: [2e6], bytes: 2e6, nodes: 1 },
    { sizes: [500, 1000], bytes: 1500, nodes: 2 },
  ];

  const out = attributeMatmuls(profile, { fingerprints });
  const byBytes = Object.fromEntries(out.classes.map((row) => [row.bytes, row]));

  assert.equal(out.dispatchesPerStep, 4);
  assert.equal(out.matchedDispatchesPerStep, 3);
  assert.equal(out.totalUsPerStep, 13200);
  assert.ok(Math.abs(out.coverage - 12700 / 13200) < 1e-9);
  assert.equal(byBytes[1e9].gbps, 100);
  assert.equal(byBytes[2e6].gbps, 1);
  assert.equal(out.unmatchedSegments, 1);
  assert.equal(out.ambiguousSegments, 0);
  assert.equal(out.invalidSegments, 1);
});

test('attributeMatmuls: two matching fingerprints with different bytes is ambiguous', () => {
  const profile = {
    runs: [
      {
        step: 0,
        segments: [
          { us: 10, dispatches: 1, pipelines: ['op:MatMulNBits'], buffers: [1e9, 2e6] },
        ],
      },
    ],
  };
  const fingerprints = [
    { sizes: [1e9], bytes: 1e9, nodes: 1 },
    { sizes: [2e6], bytes: 2e6, nodes: 1 },
  ];

  const out = attributeMatmuls(profile, { fingerprints });

  assert.equal(out.coverage, 0);
  assert.equal(out.ambiguousSegments, 1);
  assert.deepEqual(out.classes, []);
});

test('attributeMatmuls: refuses a profile without known fingerprints', () => {
  const profile = { runs: [{ step: 0, segments: [] }] };

  assert.throws(() => attributeMatmuls(profile, { fingerprints: [] }), /no known weight fingerprints/);
  assert.throws(() => attributeMatmuls({ runs: [] }, { fingerprints: [{ sizes: [1], bytes: 1 }] }), /no usable decode steps/);
});
