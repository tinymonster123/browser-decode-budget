// Gate 5: per-compute-pass GPU timestamps, to find where decode time actually
// goes. Reuses the same single-model setup as count.mjs.
//
//   node bench/browser/gpu-profile.mjs
//
// Env: MODELS=base,gate1,gate2 STEPS=4 HEADLESS=1 PORT=8126 OUT=...

import { nodeCounts } from './artifacts.mjs';
import {
  envList,
  envNumber,
  fmt,
  ledgerPath,
  mean,
  quantile,
  runDriver,
  timelineFailures,
} from './driver.mjs';

const MODELS = envList('MODELS', ['base', 'gate1', 'gate2']);
const STEPS = Math.max(1, envNumber('STEPS', 4));
const NODES = nodeCounts();

function sparkline(values, width = 48) {
  const chars = '▁▂▃▄▅▆▇█';
  if (!values.length) return '';
  const max = Math.max(...values);
  if (!max) return ' '.repeat(width);
  const buckets = [];
  const size = values.length / width;
  for (let i = 0; i < width; i++) {
    const slice = values.slice(
      Math.floor(i * size),
      Math.max(Math.floor((i + 1) * size), Math.floor(i * size) + 1),
    );
    buckets.push(mean(slice) ?? 0);
  }
  return buckets
    .map(
      (value) =>
        chars[
          Math.min(
            chars.length - 1,
            Math.floor((value / max) * (chars.length - 1)),
          )
        ],
    )
    .join('');
}

await runDriver({
  name: 'profile',
  port: envNumber('PORT', 8126),
  out: ledgerPath('OUT', 'runs/2026-10-19-exp4-pass-profile.json'),
  config: { models: MODELS, steps: STEPS },

  async measure(ctx) {
    const results = [];
    for (const model of MODELS) {
      ctx.log(`\n=== ${model} ===`);
      results.push(
        await ctx.measureModel(model, {
          path: '/gpu-profile.html',
          call: '__profile',
          options: { model, profileTokens: STEPS },
        }),
      );
    }
    return { results };
  },

  // The timestamp patch must have taken, every step must have timestamp
  // coverage, and no encoder may have overflowed its query slots.
  checks({ results }) {
    return results.map((result) => {
      const failures = timelineFailures(result);
      return {
        name: `${result.model}: timestamps`,
        ok: failures.length === 0,
        detail:
          failures.join('; ') ||
          `${result.runs?.length ?? 0} steps covered`,
      };
    });
  },

  report({ results }) {
    for (const result of results) {
      if (!result.error) continue;
      console.error(`\n[${result.model}] error: ${result.error}`);
      if (result.stack) console.error(result.stack);
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
            fmt(run.stepWallMs ? run.sumUs / (run.stepWallMs * 1000) : null, 2)
              .padStart(9),
            fmt(run.medianUs, 2).padStart(8),
            fmt(run.p90Us, 2).padStart(8),
            fmt(run.maxUs, 2).padStart(8),
          ].join(' '),
        );
      }
    }

    console.log(
      '\nGPU timeline of one decode step (each bar = one compute pass, mean over steps)',
    );
    for (const result of results) {
      if (result.error || !result.runs.length) continue;
      const width = Math.min(...result.runs.map((run) => run.segments.length));
      const profile = [];
      for (let i = 0; i < width; i++) {
        profile.push(mean(result.runs.map((run) => run.segments[i]?.us ?? 0)) ?? 0);
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
      const all = result.runs.flatMap((run) => run.segments.map((s) => s.us));
      const sum = all.reduce((a, b) => a + b, 0);
      console.log(
        `${result.model.padEnd(8)} passes=${all.length} sum=${fmt(sum / result.runs.length)}µs ` +
          `p50=${fmt(quantile(all, 0.5), 2)} p75=${fmt(quantile(all, 0.75), 2)} ` +
          `p90=${fmt(quantile(all, 0.9), 2)} p99=${fmt(quantile(all, 0.99), 2)} ` +
          `max=${fmt(Math.max(...all), 1)}`,
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
        .map(
          ([count, entry]) => `${count}d:${fmt(entry.us / entry.n, 2)}µs×${entry.n}`,
        );
      console.log(`${result.model.padEnd(8)} ${rows.join('  ')}`);
    }
  },

  ledger({ results }) {
    return { nodeCounts: NODES, results };
  },
});
