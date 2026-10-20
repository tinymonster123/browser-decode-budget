// Gate 6 experiment 1: what one serialized WebGPU dispatch costs.
//
// The same page measures the independent encoding (N dispatches in one pass,
// which the GPU can overlap) and the dependent encoding (one dispatch per
// pass, each pass reading what the previous pass wrote, which the GPU cannot
// overlap). Both run on one device with one timing method, twice, and the
// dependent chain is verified against its expected value.
//
//   node bench/browser/gpu-microbench.mjs
//
// Env: COUNTS=0,1000,2000,4000,8000 MODES=one-pass,n-pass,dependent
//      REPEATS=60 SWEEPS=2 HEADLESS=1 PORT=8125 OUT=...

import {
  agreement,
  dispatchModeComparison,
  dispatchSlope,
  fusionCeiling,
} from '../analysis/ledger.mjs';
import {
  envList,
  envNumber,
  fmt,
  ledgerPath,
  runDriver,
} from './driver.mjs';

const REPEATS = Math.max(1, envNumber('REPEATS', 60));
const SWEEPS = envNumber('SWEEPS', 2);

// The agreement check compares the two repetitions. More than two would be
// silently ignored by that comparison, so refuse rather than mislead.
if (SWEEPS !== 2) {
  throw new Error(`SWEEPS must be exactly 2, got ${SWEEPS}`);
}

const COUNTS = envList('COUNTS', ['0', '1000', '2000', '4000', '8000']).map(Number);
const MODES = envList('MODES', ['one-pass', 'n-pass', 'dependent']);

// The two ceiling questions from the Gate 6 spec: 211 nodes removed by Gate 2
// (what fusion would be worth if one node were one dispatch), and every
// dispatch in a base decode step (1112).
const FUSION_DISPATCH_COUNTS = [211, 1112];
const INDEPENDENT_MODE = 'one-pass';
const DEPENDENT_MODE = 'dependent';

for (const required of [INDEPENDENT_MODE, DEPENDENT_MODE]) {
  if (!MODES.includes(required)) {
    throw new Error(`MODES must include "${required}" to compare the two costs`);
  }
}

function slopeOf(result, mode, sweep, field = 'wallMs') {
  try {
    return dispatchSlope(result, { mode, sweep, field });
  } catch {
    return null;
  }
}

await runDriver({
  name: 'microbench',
  port: envNumber('PORT', 8125),
  out: ledgerPath('OUT', 'runs/2026-10-20-exp5-serialized-dispatch.json'),
  config: { counts: COUNTS, modes: MODES, repeats: REPEATS, sweeps: SWEEPS },

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
          (options) => window.__microbench(options),
          { counts: COUNTS, modes: MODES, repeats: REPEATS, sweeps: SWEEPS },
        ),
    );

    const slopes = {};
    for (const mode of MODES) {
      slopes[mode] = [];
      for (let sweep = 0; sweep < SWEEPS; sweep++) {
        const p05Wall = slopeOf(result, mode, sweep, 'p05WallMs');
        const p05Gpu = slopeOf(result, mode, sweep, 'p05GpuMs');
        const medianWall = slopeOf(result, mode, sweep, 'wallMs');
        const minWall = slopeOf(result, mode, sweep, 'minWallMs');
        slopes[mode].push({
          sweep,
          p05WallUsPerDispatch: p05Wall?.usPerDispatch ?? null,
          p05GpuUsPerDispatch: p05Gpu?.usPerDispatch ?? null,
          medianWallUsPerDispatch: medianWall?.usPerDispatch ?? null,
          minWallUsPerDispatch: minWall?.usPerDispatch ?? null,
          wallPoints: p05Wall?.points ?? null,
          gpuPoints: p05Gpu?.points ?? null,
        });
      }
    }

    const slopeValues = (mode, field) =>
      slopes[mode].map((entry) => entry[field]).filter(Number.isFinite);

    // Two repetitions of the same measurement must agree within 10% before
    // any number is used for a decision.
    const agreementByMode = {};
    for (const mode of MODES) {
      const walls = slopeValues(mode, 'p05WallUsPerDispatch');
      const gpus = slopeValues(mode, 'p05GpuUsPerDispatch');
      agreementByMode[mode] = {
        wall:
          walls.length === 2
            ? agreement(walls[0], walls[1])
            : { within: false, reason: 'missing sweep' },
        gpu:
          gpus.length === 2
            ? agreement(gpus[0], gpus[1])
            : { within: false, reason: 'missing sweep' },
      };
    }

    const meanSlope = (mode) => {
      const values = slopeValues(mode, 'p05WallUsPerDispatch');
      return values.reduce((sum, value) => sum + value, 0) / values.length;
    };

    const independentUs = meanSlope(INDEPENDENT_MODE);
    const dependentUs = meanSlope(DEPENDENT_MODE);
    const haveBoth = Number.isFinite(independentUs) && Number.isFinite(dependentUs);
    const comparison = haveBoth
      ? dispatchModeComparison({ dependentUs, independentUs })
      : null;
    const ceilings = haveBoth
      ? fusionCeiling({
          usPerDispatch: dependentUs,
          dispatchCounts: FUSION_DISPATCH_COUNTS,
        })
      : [];

    const dependentAgrees = agreementByMode[DEPENDENT_MODE]?.wall?.within === true;
    const independentAgrees = agreementByMode[INDEPENDENT_MODE]?.wall?.within === true;
    const chain = Array.isArray(result.chain) ? result.chain : [];
    const chainVerified = chain.length > 0 && chain.every((entry) => entry.ok);
    const chainFailures = chain.filter((entry) => !entry.ok);

    const checks = [
      {
        name: 'dependent-chain-verifies',
        ok: chainVerified,
        detail:
          chain.length === 0
            ? 'no chain samples were read back'
            : chainFailures.length === 0
              ? `${chain.length} chain points match the expected recurrence`
              : `${chainFailures.length} chain points diverged: ` +
                JSON.stringify(chainFailures.slice(0, 3)),
      },
      {
        name: 'dependent-exceeds-independent',
        ok: comparison?.dependentExceedsIndependent === true,
        detail: comparison
          ? `${fmt(comparison.ratio, 2)}× serialized/independent`
          : 'dependent or independent slope is missing',
      },
      {
        name: 'dependent-sweeps-agree',
        ok: dependentAgrees,
        detail: `${fmt(agreementByMode[DEPENDENT_MODE]?.wall?.deltaPercent, 1)}% between sweeps`,
      },
      {
        name: 'independent-sweeps-agree',
        ok: independentAgrees,
        detail: `${fmt(agreementByMode[INDEPENDENT_MODE]?.wall?.deltaPercent, 1)}% between sweeps`,
      },
    ];

    return {
      result,
      slopes,
      agreementByMode,
      comparison,
      ceilings,
      independentUs,
      dependentUs,
      checks,
      usable:
        chainVerified &&
        dependentAgrees &&
        independentAgrees &&
        comparison?.dependentExceedsIndependent === true,
    };
  },

  checks({ checks }) {
    return checks;
  },

  report(payload) {
    const {
      result,
      slopes,
      agreementByMode,
      comparison,
      ceilings,
      independentUs,
      dependentUs,
    } = payload;

    if (result.error) {
      console.error(`error: ${result.error}`);
      if (result.stack) console.error(result.stack);
    }

    console.log(`adapter: ${JSON.stringify(result.info)}`);

    const byModeSweep = new Map();
    for (const row of result.rows ?? []) {
      const key = `${row.mode}#${row.sweep}`;
      const list = byModeSweep.get(key) ?? [];
      list.push(row);
      byModeSweep.set(key, list);
    }

    console.log(
      '\nwall = encode + submit + GPU completion; gpu = submit → completion; ' +
        'med = median of repeats, p05 = 5th percentile, min = fastest repeat',
    );
    console.log(
      [
        'mode'.padEnd(10),
        'sweep'.padStart(5),
        'count'.padStart(7),
        'wall med'.padStart(10),
        'wall p05'.padStart(10),
        'wall min'.padStart(10),
        'gpu p05'.padStart(10),
        'cpu ms'.padStart(9),
      ].join(' '),
    );

    for (const [key, rows] of byModeSweep) {
      rows.sort((a, b) => a.count - b.count);
      const [mode, sweep] = key.split('#');
      for (const row of rows) {
        console.log(
          [
            mode.padEnd(10),
            sweep.padStart(5),
            String(row.count).padStart(7),
            fmt(row.wallMs).padStart(10),
            fmt(row.p05WallMs).padStart(10),
            fmt(row.minWallMs).padStart(10),
            fmt(row.p05GpuMs).padStart(10),
            fmt(row.cpuMs).padStart(9),
          ].join(' '),
        );
      }
    }

    console.log('\nslope between the two largest counts (p05 / median / min wall, p05 gpu)');
    console.log(
      [
        'mode'.padEnd(10),
        'sweep'.padStart(5),
        'wall p05'.padStart(9),
        'wall med'.padStart(9),
        'wall min'.padStart(9),
        'gpu p05'.padStart(9),
      ].join(' '),
    );
    for (const mode of MODES) {
      for (const entry of slopes[mode]) {
        console.log(
          [
            mode.padEnd(10),
            String(entry.sweep).padStart(5),
            fmt(entry.p05WallUsPerDispatch, 3).padStart(9),
            fmt(entry.medianWallUsPerDispatch, 3).padStart(9),
            fmt(entry.minWallUsPerDispatch, 3).padStart(9),
            fmt(entry.p05GpuUsPerDispatch, 3).padStart(9),
          ].join(' '),
        );
      }
    }

    console.log('\nagreement between the two sweeps');
    for (const mode of MODES) {
      const wall = agreementByMode[mode].wall;
      const gpu = agreementByMode[mode].gpu;
      console.log(
        `  ${mode.padEnd(10)} wall ${fmt(wall.a, 3)} vs ${fmt(wall.b, 3)} µs ` +
          `→ ${fmt(wall.deltaPercent, 1)}% ${wall.within ? 'OK' : 'TOO WIDE'}   ` +
          `gpu ${fmt(gpu.a, 3)} vs ${fmt(gpu.b, 3)} µs ` +
          `→ ${fmt(gpu.deltaPercent, 1)}% ${gpu.within ? 'OK' : 'TOO WIDE'}`,
      );
    }

    if (comparison) {
      console.log('\nserialized vs independent');
      console.log(
        `  ${fmt(dependentUs, 3)} µs vs ${fmt(independentUs, 3)} µs = ` +
          `${fmt(comparison.ratio, 2)}× ` +
          `(+${fmt(comparison.extraUs, 3)} µs per dispatch)`,
      );
    } else {
      console.log('\nserialized vs independent: could not be obtained');
    }

    console.log('\nrecomputed node-fusion ceiling at the serialized cost');
    if (ceilings.length) {
      for (const { dispatches, ms } of ceilings) {
        console.log(`  ${String(dispatches).padStart(4)} dispatches → ${fmt(ms, 3)} ms`);
      }
    } else {
      console.log('  — no usable dependent slope');
    }
  },

  ledger({ result, slopes, agreementByMode, comparison, ceilings, checks, usable }) {
    return {
      result,
      slopes,
      agreement: agreementByMode,
      comparison,
      ceilings,
      checks,
      usable,
    };
  },
});
