// Experiment 3: measure what one trivial WebGPU dispatch / pass / submit
// costs on this machine, so the dispatch deltas from experiment 2 can be
// converted into milliseconds.
//
//   node bench/browser/gpu-microbench.mjs
//
// Env: REPEATS=5 HEADLESS=1 PORT=8125 OUT=...

import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { dispatchSlope } from '../analysis/ledger.mjs';
import {
  announceLoad,
  instrumentPage,
  launchChrome,
  startServer,
  stampLedger,
  writeLedger,
} from './runtime.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '../..');
const PORT = Number(process.env.PORT ?? 8125);
const REPEATS = Math.max(1, Number(process.env.REPEATS ?? 5));
const HEADLESS = process.env.HEADLESS === '1';
const OUT =
  process.env.OUT ?? join(root, 'runs', '2026-10-19-exp3-dispatch-cost.json');

const load = announceLoad();

const server = startServer(PORT);

function fmt(value, digits = 2) {
  if (value == null || !Number.isFinite(value)) return '—';
  return value.toFixed(digits);
}

try {
  await server.ready;

  const context = await launchChrome({
    headless: HEADLESS,
    viewport: { width: 900, height: 700 },
    label: 'fused-onnx-microbench',
  });

  let result;
  try {
    const page = await context.newPage();
    await instrumentPage(page);

    await page.goto(`http://127.0.0.1:${PORT}/gpu-microbench.html`, { waitUntil: 'load' });
    await page.waitForFunction(() => window.__ready === true, { timeout: 30000 });
    result = await page.evaluate(
      (options) => window.__microbench(options),
      { counts: [0, 100, 500, 1000, 2000], repeats: REPEATS },
    );
  } finally {
    await context.close();
  }

  if (result.error) {
    console.error(`error: ${result.error}`);
    if (result.stack) console.error(result.stack);
  }

  console.log(`adapter: ${JSON.stringify(result.info)}`);

  const byMode = new Map();
  for (const row of result.rows) {
    const list = byMode.get(row.mode) ?? [];
    list.push(row);
    byMode.set(row.mode, list);
  }

  console.log('\nwall = encode + submit + GPU completion');
  console.log(
    [
      'mode'.padEnd(10),
      'count'.padStart(7),
      'wall ms'.padStart(10),
      'cpu ms'.padStart(10),
      'gpu ms'.padStart(10),
      'µs/dispatch vs 0'.padStart(17),
    ].join(' '),
  );

  const summary = {};
  for (const [mode, rows] of byMode) {
    rows.sort((a, b) => a.count - b.count);
    const zero = rows.find((row) => row.count === 0);
    summary[mode] = [];
    for (const row of rows) {
      const base = zero?.wallMs ?? 0;
      const perDispatch =
        row.count > 0 ? ((row.wallMs - base) * 1000) / row.count : null;
      summary[mode].push({
        count: row.count,
        wallMs: row.wallMs,
        cpuMs: row.cpuMs,
        gpuMs: row.gpuMs,
        usPerDispatch: perDispatch,
        samples: row.samples,
      });
      console.log(
        [
          mode.padEnd(10),
          String(row.count).padStart(7),
          fmt(row.wallMs).padStart(10),
          fmt(row.cpuMs).padStart(10),
          fmt(row.gpuMs).padStart(10),
          fmt(perDispatch, 3).padStart(17),
        ].join(' '),
      );
    }
  }

  console.log(
    `\ncreateBindGroup: ${fmt(result.bindGroup?.usPerCall, 3)} µs/call ` +
      `(${result.bindGroup?.count} calls × ${REPEATS} repeats)`,
  );

  // The slope between the two largest counts is the robust estimate; it comes
  // from the same analysis seam the decision report uses.
  console.log('\nslope between the two largest counts');
  for (const mode of Object.keys(summary)) {
    try {
      const slope = dispatchSlope({ result }, { mode });
      const [first, second] = slope.points;
      console.log(
        `  ${mode.padEnd(10)} ${fmt(slope.usPerDispatch, 3)} µs/dispatch ` +
          `(${first.count}→${second.count})`,
      );
    } catch (error) {
      console.log(`  ${mode.padEnd(10)} — ${error.message}`);
    }
  }

  const ledger = stampLedger({
    root,
    config: { repeats: REPEATS, port: PORT, headless: HEADLESS },
    load,
    result,
    summary,
  });
  writeLedger(OUT, ledger);
  console.log(`\nwrote ${OUT}`);

  if (result.error) process.exitCode = 1;
} finally {
  server.stop();
}
