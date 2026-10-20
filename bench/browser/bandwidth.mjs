// Gate 6 experiment 2: the achievable read bandwidth of this GPU.
//
// The page runs one read-only compute pass over a buffer at several sizes and
// times the pass with timestamp-query. The plateau is the first size from
// which every remaining sample stays within 10%, so a cache-resident region
// at small sizes cannot be mistaken for the ceiling.
//
//   node bench/browser/bandwidth.mjs
//
// Env: SIZES_MB=4,8,16,32,64,128,256 REPEATS=30 SWEEPS=2 HEADLESS=1 PORT=8127
//      OUT=...

import { bandwidthPlateau, combinePlateaus } from '../analysis/ledger.mjs';
import {
  envList,
  envNumber,
  fmt,
  ledgerPath,
  runDriver,
} from './driver.mjs';

const REPEATS = Math.max(1, envNumber('REPEATS', 30));
const SWEEPS = envNumber('SWEEPS', 2);

if (SWEEPS !== 2) {
  throw new Error(`SWEEPS must be exactly 2, got ${SWEEPS}`);
}

const SIZES_MB = envList('SIZES_MB', ['4', '8', '16', '32', '64', '128', '256']).map(
  Number,
);
if (SIZES_MB.length < 4) {
  throw new Error(`SIZES_MB needs at least four sizes, got ${SIZES_MB.length}`);
}

function plateauFor(result, sweep) {
  const rows = (result.rows ?? [])
    .filter((row) => row.sweep === sweep)
    .sort((a, b) => a.bytes - b.bytes);

  try {
    return {
      sweep,
      ...bandwidthPlateau(
        rows.map((row) => ({ bytes: row.bytes, ms: row.p05Ns / 1e6 })),
      ),
    };
  } catch (error) {
    return { sweep, error: error.message };
  }
}

await runDriver({
  name: 'bandwidth',
  port: envNumber('PORT', 8127),
  out: ledgerPath('OUT', 'runs/2026-10-20-exp6-bandwidth.json'),
  config: { sizesMb: SIZES_MB, repeats: REPEATS, sweeps: SWEEPS },

  async measure(ctx) {
    const result = await ctx.withPage(
      {
        path: '/gpu-microbench.html',
        viewport: { width: 900, height: 700 },
        label: `${ctx.name}-page`,
        readyTimeout: 30000,
      },
      (page) =>
        page.evaluate(
          (options) => window.__bandwidth(options),
          { sizesMb: SIZES_MB, repeats: REPEATS, sweeps: SWEEPS },
        ),
    );

    const plateaus = [plateauFor(result, 0), plateauFor(result, 1)];
    const bothPlateaus = plateaus.every((plateau) => !plateau.error);
    const plateau = bothPlateaus ? combinePlateaus(plateaus) : null;
    const checks = [
      {
        name: 'timestamp-query-available',
        ok: result.timestampQuery === true,
        detail: result.timestampQuery
          ? 'pass timestamps were used'
          : 'timestamp-query is unavailable; no bandwidth is reported',
      },
      {
        name: 'at-least-four-sizes',
        ok: SIZES_MB.length >= 4,
        detail: `${SIZES_MB.length} sizes`,
      },
      {
        name: 'plateau-found-in-both-sweeps',
        ok: bothPlateaus,
        detail: bothPlateaus
          ? 'both sweeps have a 10% plateau'
          : plateaus.map((entry) => entry.error ?? 'ok').join(' | '),
      },
      {
        name: 'sweeps-agree',
        ok: plateau?.agreement.within === true,
        detail: `${fmt(plateau?.agreement.deltaPercent, 1)}% between sweeps`,
      },
      {
        name: 'plateau-is-hardware-plausible',
        ok:
          plateau != null &&
          plateau.gbps >= 10 &&
          plateau.gbps <= 1000,
        detail: plateau ? `${fmt(plateau.gbps)} GB/s` : 'no plateau',
      },
    ];

    return { result, plateaus, plateau, checks };
  },

  checks({ checks }) {
    return checks;
  },

  report({ result, plateaus, plateau }) {
    if (result.error) {
      console.error(`error: ${result.error}`);
      if (result.stack) console.error(result.stack);
    }

    console.log(`adapter: ${JSON.stringify(result.info)}`);
    console.log(`timestamp-query: ${result.timestampQuery}`);

    if (!result.timestampQuery) {
      console.error(
        '\ntimestamp-query is not available; refusing to report bandwidth.',
      );
    }

    console.log('\nread bandwidth per size');
    console.log(
      [
        'size MB'.padStart(8),
        'sweep'.padStart(5),
        'p05 GB/s'.padStart(10),
        'median'.padStart(9),
        'min ns'.padStart(9),
        'p05 ms'.padStart(9),
      ].join(' '),
    );

    for (const row of [...(result.rows ?? [])].sort(
      (a, b) => a.bytes - b.bytes || a.sweep - b.sweep,
    )) {
      console.log(
        [
          String(row.bytes / 1024 / 1024).padStart(8),
          String(row.sweep).padStart(5),
          fmt(row.p05Gbps).padStart(10),
          fmt(row.medianGbps).padStart(9),
          fmt(row.minNs, 0).padStart(9),
          fmt(row.p05Ns / 1e6, 3).padStart(9),
        ].join(' '),
      );
    }

    console.log('\nplateau within 10%');
    for (const entry of plateaus) {
      if (entry.error) {
        console.log(`  sweep ${entry.sweep}: ${entry.error}`);
      } else {
        console.log(
          `  sweep ${entry.sweep}: ${fmt(entry.gbps)} GB/s ` +
            `from ${entry.fromBytes / 1024 / 1024} MB`,
        );
      }
    }

    if (plateau) {
      console.log(
        `  ceiling: ${fmt(plateau.gbps)} GB/s from ${plateau.fromBytes / 1024 / 1024} MB ` +
          `(sweeps differ ${fmt(plateau.agreement.deltaPercent, 1)}%)`,
      );
    }
  },

  ledger({ result, plateaus, plateau, checks }) {
    return {
      result,
      plateaus,
      plateau,
      agreement: plateau?.agreement ?? null,
      checks,
      usable: checks.every((check) => check.ok),
    };
  },
});
