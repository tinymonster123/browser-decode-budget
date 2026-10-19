// Gate 5: per-compute-pass GPU timestamps, to find where decode time actually
// goes. Reuses the same single-model setup as count.mjs.
//
//   node bench/browser/gpu-profile.mjs
//
// Env: MODELS=base,gate1,gate2 STEPS=4 HEADLESS=1 PORT=8126 OUT=...

import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { nodeCounts } from './artifacts.mjs';
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
const PORT = Number(process.env.PORT ?? 8126);
const STEPS = Math.max(1, Number(process.env.STEPS ?? 4));
const HEADLESS = process.env.HEADLESS === '1';
const MODELS = (process.env.MODELS ?? 'base,gate1,gate2').split(',').filter(Boolean);
const OUT = process.env.OUT ?? join(root, 'runs', '2026-10-19-exp4-pass-profile.json');

const NODES = nodeCounts();

const load = announceLoad();

const server = startServer(PORT);

async function measure(model) {
  const context = await launchChrome({
    headless: HEADLESS,
    label: `fused-onnx-profile-${model}`,
  });

  try {
    const page = await context.newPage();
    await instrumentPage(page, { prefix: model });

    await page.goto(`http://127.0.0.1:${PORT}/gpu-profile.html`, { waitUntil: 'load' });
    await page.waitForFunction(() => window.__ready === true, { timeout: 60000 });

    return await page.evaluate(
      (options) => window.__profile(options),
      { model, profileTokens: STEPS },
    );
  } finally {
    await context.close();
  }
}

function mean(values) {
  if (!values.length) return null;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function fmt(value, digits = 1) {
  if (value == null || !Number.isFinite(value)) return '—';
  return value.toFixed(digits);
}

function sparkline(values, width = 48) {
  const chars = '▁▂▃▄▅▆▇█';
  if (!values.length) return '';
  const max = Math.max(...values);
  if (!max) return ' '.repeat(width);
  const buckets = [];
  const size = values.length / width;
  for (let i = 0; i < width; i++) {
    const slice = values.slice(Math.floor(i * size), Math.max(Math.floor((i + 1) * size), Math.floor(i * size) + 1));
    buckets.push(mean(slice) ?? 0);
  }
  return buckets
    .map((value) => chars[Math.min(chars.length - 1, Math.floor((value / max) * (chars.length - 1)))])
    .join('');
}

try {
  await server.ready;

  const results = [];
  for (const model of MODELS) {
    console.log(`\n=== ${model} ===`);
    results.push(await measure(model));
  }

  // Harness self-checks become assertions: the timestamp patch must have
  // taken, every step must have timestamp coverage, and no encoder may have
  // overflowed its query slots.
  const checkFailures = results.flatMap((result) => {
    if (result.error) return [];
    const failures = [];
    for (const [name, ok] of Object.entries(result.patchInfo ?? {})) {
      if (!ok) failures.push(`${result.model}: ${name} patch failed`);
    }
    const runs = result.runs ?? [];
    if (!runs.length) failures.push(`${result.model}: no decode steps`);
    for (const run of runs) {
      if (!(run.coverage > 0)) {
        failures.push(`${result.model} step ${run.step}: no timestamp coverage`);
      }
      if (run.overflowEncoders > 0) {
        failures.push(`${result.model} step ${run.step}: encoder overflow`);
      }
    }
    return failures;
  });

  for (const result of results) {
    if (result.error) {
      console.error(`\n[${result.model}] error: ${result.error}`);
      if (result.stack) console.error(result.stack);
    }
  }

  console.log('\nPer decode step (GPU timestamps, µs)');
  console.log(
    [
      'model'.padEnd(8),
      'step'.padStart(5),
      'passes'.padStart(7),
      'dispatch'.padStart(9),
      'sum µs'.padStart(9),
      'span µs'.padStart(9),
      'wall µs'.padStart(9),
      'sum/wall'.padStart(9),
      'median'.padStart(8),
      'p90'.padStart(8),
      'max'.padStart(8),
    ].join(' '),
  );

  for (const result of results) {
    if (result.error) continue;
    for (const run of result.runs) {
      console.log(
        [
          result.model.padEnd(8),
          String(run.step).padStart(5),
          String(run.passes).padStart(7),
          String(run.totalDispatches).padStart(9),
          fmt(run.sumUs).padStart(9),
          fmt(run.spanUs).padStart(9),
          fmt(run.stepWallMs == null ? null : run.stepWallMs * 1000).padStart(9),
          fmt(run.stepWallMs ? run.sumUs / (run.stepWallMs * 1000) : null, 2).padStart(9),
          fmt(run.medianUs, 2).padStart(8),
          fmt(run.p90Us, 2).padStart(8),
          fmt(run.maxUs, 2).padStart(8),
        ].join(' '),
      );
    }
  }

  console.log('\nGPU timeline of one decode step (each bar = one compute pass, mean over steps)');
  for (const result of results) {
    if (result.error || !result.runs.length) continue;
    const width = Math.min(...result.runs.map((r) => r.segments.length));
    const profile = [];
    for (let i = 0; i < width; i++) {
      profile.push(mean(result.runs.map((r) => r.segments[i]?.us ?? 0)) ?? 0);
    }
    const maxUs = Math.max(...profile);
    console.log(
      `${result.model.padEnd(8)} ${sparkline(profile, 64)}  ` +
        `passes=${width} max=${fmt(maxUs, 1)}µs`,
    );
    const top = profile
      .map((us, index) => ({ index, us }))
      .sort((a, b) => b.us - a.us)
      .slice(0, 6);
    console.log(
      `         top passes: ${top
        .map((entry) => `#${entry.index}=${fmt(entry.us, 1)}µs`)
        .join('  ')}`,
    );
  }

  console.log('\nPass-duration distribution (mean over steps)');
  for (const result of results) {
    if (result.error || !result.runs.length) continue;
    const all = result.runs.flatMap((r) => r.segments.map((s) => s.us)).sort((a, b) => a - b);
    const pick = (p) => all[Math.min(all.length - 1, Math.floor(p * all.length))];
    const sum = all.reduce((a, b) => a + b, 0);
    console.log(
      `${result.model.padEnd(8)} passes=${all.length} sum=${fmt(sum / result.runs.length)}µs ` +
        `p50=${fmt(pick(0.5), 2)} p75=${fmt(pick(0.75), 2)} p90=${fmt(pick(0.9), 2)} ` +
        `p99=${fmt(pick(0.99), 2)} max=${fmt(all.at(-1), 1)}`,
    );
  }

  console.log('\nDispatches vs pass duration (mean over steps)');
  for (const result of results) {
    if (result.error || !result.runs.length) continue;
    const byCount = new Map();
    for (const run of result.runs) {
      for (const seg of run.segments) {
        const entry = byCount.get(seg.dispatches) ?? { n: 0, us: 0 };
        entry.n += 1;
        entry.us += seg.us;
        byCount.set(seg.dispatches, entry);
      }
    }
    const rows = [...byCount.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([count, entry]) =>
        `${count}d:${fmt(entry.us / entry.n, 2)}µs×${entry.n}`,
      );
    console.log(`${result.model.padEnd(8)} ${rows.join('  ')}`);
  }

  const ledger = stampLedger({
    root,
    config: { models: MODELS, steps: STEPS, port: PORT, headless: HEADLESS },
    load,
    nodeCounts: NODES,
    results,
  });
  writeLedger(OUT, ledger);
  console.log(`\nwrote ${OUT}`);

  for (const failure of checkFailures) console.error(`[check] ${failure}`);
  if (results.some((r) => r.error) || checkFailures.length) process.exitCode = 1;
} finally {
  server.stop();
}
