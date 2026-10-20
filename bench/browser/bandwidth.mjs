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

import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { bandwidthPlateau, combinePlateaus } from '../analysis/ledger.mjs';
import {
  announceLoad,
  openPage,
  startServer,
  stampLedger,
  writeLedger,
} from './runtime.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '../..');
const PORT = Number(process.env.PORT ?? 8127);
const REPEATS = Math.max(1, Number(process.env.REPEATS ?? 30));
const SWEEPS = Number(process.env.SWEEPS ?? 2);
const HEADLESS = process.env.HEADLESS === '1';

if (SWEEPS !== 2) {
  throw new Error(`SWEEPS must be exactly 2, got ${SWEEPS}`);
}

function csv(name, fallback) {
  const raw = process.env[name];
  return raw ? raw.split(',').map((value) => value.trim()).filter(Boolean) : fallback;
}

const SIZES_MB = csv('SIZES_MB', [4, 8, 16, 32, 64, 128, 256]).map(Number);
if (SIZES_MB.length < 4) {
  throw new Error(`SIZES_MB needs at least four sizes, got ${SIZES_MB.length}`);
}

const OUT =
  process.env.OUT ?? join(root, 'runs', '2026-10-20-exp6-bandwidth.json');

const load = announceLoad();
const server = startServer(PORT);

function fmt(value, digits = 2) {
  if (value == null || !Number.isFinite(value)) return '—';
  return value.toFixed(digits);
}

function plateauFor(result, sweep) {
  const rows = result.rows
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

try {
  await server.ready;

  let context;
  let result;
  try {
    const opened = await openPage({
      url: `http://127.0.0.1:${PORT}/gpu-microbench.html`,
      headless: HEADLESS,
      viewport: { width: 900, height: 700 },
      label: 'fused-onnx-bandwidth',
      prefix: 'page',
      readyTimeout: 30000,
    });
    context = opened.context;
    result = await opened.page.evaluate(
      (options) => window.__bandwidth(options),
      { sizesMb: SIZES_MB, repeats: REPEATS, sweeps: SWEEPS },
    );
  } finally {
    await context?.close();
  }

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

  for (const row of [...result.rows].sort(
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

  const plateaus = [plateauFor(result, 0), plateauFor(result, 1)];
  console.log('\nplateau within 10%');
  for (const plateau of plateaus) {
    if (plateau.error) {
      console.log(`  sweep ${plateau.sweep}: ${plateau.error}`);
    } else {
      console.log(
        `  sweep ${plateau.sweep}: ${fmt(plateau.gbps)} GB/s ` +
          `from ${plateau.fromBytes / 1024 / 1024} MB`,
      );
    }
  }

  const bothPlateaus = plateaus.every((plateau) => !plateau.error);
  const plateau = bothPlateaus ? combinePlateaus(plateaus) : null;

  if (plateau) {
    console.log(
      `  ceiling: ${fmt(plateau.gbps)} GB/s from ${plateau.fromBytes / 1024 / 1024} MB ` +
        `(sweeps differ ${fmt(plateau.agreement.deltaPercent, 1)}%)`,
    );
  }

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
        : plateaus.map((p) => p.error ?? 'ok').join(' | '),
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

  console.log('\nself-checks');
  for (const check of checks) {
    console.log(`  [${check.ok ? 'ok' : 'FAIL'}] ${check.name}: ${check.detail}`);
  }

  const ledger = stampLedger({
    root,
    config: {
      sizesMb: SIZES_MB,
      repeats: REPEATS,
      sweeps: SWEEPS,
      port: PORT,
      headless: HEADLESS,
    },
    load,
    result,
    plateaus,
    plateau,
    agreement: plateau?.agreement ?? null,
    checks,
    usable: checks.every((check) => check.ok),
  });
  writeLedger(OUT, ledger);
  console.log(`\nwrote ${OUT}`);

  if (!ledger.usable) process.exitCode = 1;
} finally {
  server.stop();
}
