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
} from './ledger.mjs';

function row(mode, count, wallMs, gpuMs = wallMs) {
  return { mode, count, wallMs, gpuMs, cpuMs: 0, samples: [] };
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

test('compareModels: reports the dispatch and time deltas for a pair', () => {
  const ledger = {
    nodeCounts: { base: 100, gate2: 50 },
    results: [
      {
        model: 'base',
        variants: [
          {
            runs: [
              { cpu: false, perToken: { 'pass.dispatchWorkgroups': 100 } },
              { cpu: true, decodeMsPerStep: 10 },
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
