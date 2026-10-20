// Gate 6: how much of the measured bandwidth ceiling the LM head uses.
//
// No browser and no model run. The numerator comes from the Gate 5 per-pass
// timestamps, the denominator from the bandwidth probe, and each model's
// weight bytes from its own recorded LM head initializer sizes. If the
// profile was not taken with timestamp-query, the analysis refuses instead of
// substituting a guess.
//
//   node bench/browser/lm-head-utilization.mjs
//
// Env: PROFILE=... BANDWIDTH=... WEIGHTS=path1,path2,... OUT=...

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { lmHeadUtilization } from '../analysis/ledger.mjs';
import {
  envList,
  envText,
  fmt,
  ledgerPath,
  root,
  runDriver,
} from './driver.mjs';

const PROFILE = envText('PROFILE', join(root, 'runs/2026-10-19-exp4-pass-profile.json'));
const BANDWIDTH = envText(
  'BANDWIDTH',
  join(root, 'runs/2026-10-20-exp6-bandwidth.json'),
);
const WEIGHTS = envList('WEIGHTS', [
  join(root, 'runs/2026-10-20-lm-head-weights.json'),
  join(root, 'runs/2026-10-20-lm-head-weights-gate1.json'),
  join(root, 'runs/2026-10-20-lm-head-weights-gate2.json'),
]);

function readJson(path) {
  return JSON.parse(readFileSync(path, 'utf8'));
}

await runDriver({
  name: 'lm-head-utilization',
  out: ledgerPath('OUT', 'runs/2026-10-20-exp6-lm-head-utilization.json'),
  server: false,
  config: { profile: PROFILE, bandwidth: BANDWIDTH, weights: WEIGHTS },

  async measure() {
    const profile = readJson(PROFILE);
    const bandwidth = readJson(BANDWIDTH);
    const weightRecords = WEIGHTS.map(readJson);

    if (bandwidth.result?.timestampQuery !== true) {
      throw new Error(
        'the bandwidth ledger was not taken with timestamp-query; ' +
          'refusing to compute utilization',
      );
    }
    if (bandwidth.usable !== true) {
      throw new Error(
        'the bandwidth ledger did not pass its own checks; refusing to use its ceiling',
      );
    }
    if (bandwidth.plateau == null) {
      throw new Error('the bandwidth ledger has no agreeing plateau');
    }

    const names = weightRecords.map((record) => record.name);
    if (new Set(names).size !== names.length) {
      throw new Error(`duplicate weight artifact names: ${names.join(', ')}`);
    }

    const bytesByModel = Object.fromEntries(
      weightRecords.map((record) => [record.name, record.totalBytes]),
    );
    const missing = (profile.results ?? [])
      .map((result) => result.model)
      .filter((model) => !Number.isFinite(bytesByModel[model]));
    if (missing.length) {
      throw new Error(`no recorded weight bytes for: ${missing.join(', ')}`);
    }

    const ceilingGbps = bandwidth.plateau.gbps;
    const results = lmHeadUtilization(profile, { bytesByModel, ceilingGbps });

    return { bandwidth, weightRecords, ceilingGbps, results };
  },

  report({ bandwidth, weightRecords, ceilingGbps, results }) {
    console.log(`profile:   ${PROFILE}`);
    console.log(`bandwidth: ${BANDWIDTH}`);
    console.log(
      `ceiling:   ${fmt(ceilingGbps)} GB/s from ${bandwidth.plateau.fromBytes / 1024 / 1024} MB`,
    );
    console.log(
      `weights:   ${weightRecords
        .map((record) => `${record.name}=${record.totalBytes}`)
        .join(', ')}`,
    );

    console.log('\nLM head utilization');
    console.log(
      [
        'model'.padEnd(8),
        'LM head µs'.padStart(11),
        'bytes'.padStart(11),
        'GB/s'.padStart(9),
        '% ceiling'.padStart(10),
      ].join(' '),
    );
    for (const result of results) {
      console.log(
        [
          result.model.padEnd(8),
          fmt(result.medianUs, 1).padStart(11),
          String(result.bytes).padStart(11),
          fmt(result.gbps).padStart(9),
          fmt(result.percent, 1).padStart(10),
        ].join(' '),
      );
    }
  },

  ledger({ bandwidth, weightRecords, ceilingGbps, results }) {
    return {
      inputs: {
        ceilingGbps,
        fromBytes: bandwidth.plateau.fromBytes,
        weights: weightRecords.map((record) => ({
          model: record.name,
          bytes: record.totalBytes,
          graphInitializerBytes: record.graphInitializerBytes,
          node: record.node,
          selection: record.selection,
          tensors: record.tensors,
        })),
      },
      results,
    };
  },
});
