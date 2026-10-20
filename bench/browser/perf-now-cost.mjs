// Why the cpu=true pass of count.mjs is ~4.7% slower: measure what
// performance.now() and the counter update actually cost in Chrome, then check
// the arithmetic against the two count runs.
//
//   node bench/browser/perf-now-cost.mjs
//
// Env: OUT=... LEDGER=runs/2026-10-19-exp2-dispatch-count.json

import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { countsRun, timingRun } from '../analysis/ledger.mjs';
import {
  envText,
  ledgerPath,
  root,
  runDriver,
} from './driver.mjs';

const LEDGER = envText(
  'LEDGER',
  join(root, 'runs/2026-10-19-exp2-dispatch-count.json'),
);

await runDriver({
  name: 'perf-now',
  out: ledgerPath('OUT', 'runs/2026-10-19-exp2b-timestamp-overhead.json'),
  server: false,
  config: { ledger: LEDGER },

  async measure(ctx) {
    const costs = await ctx.withPage(
      { content: '<html><body>perf-now probe</body></html>', headless: true },
      (page) =>
        page.evaluate(() => {
          const N = 2_000_000;
          let sink = 0;
          const bench = (fn) => {
            fn(1000);
            const t0 = performance.now();
            for (let i = 0; i < N; i++) fn(i);
            return ((performance.now() - t0) / N) * 1000; // µs/op
          };

          const dict = Object.create(null);
          const arr = new Float64Array(8);

          const now = bench(() => {
            sink += performance.now();
          });
          const twoNow = bench(() => {
            sink += performance.now();
            sink += performance.now();
          });
          const dictUpdate = bench((i) => {
            dict['k' + (i & 7)] = (dict['k' + (i & 7)] ?? 0) + 1;
          });
          const typedUpdate = bench((i) => {
            arr[i & 7] = (arr[i & 7] ?? 0) + 1;
          });

          return { now, twoNow, dictUpdate, typedUpdate, iterations: N, sink };
        }),
    );

    // Rebuild the per-token call count from the counting ledger and compare the
    // predicted increment with the measured difference between the two runs.
    let arithmetic = null;
    try {
      const ledger = JSON.parse(readFileSync(LEDGER, 'utf8'));
      const rows = [];
      for (const result of ledger.results) {
        const counts = countsRun(result);
        const timed = timingRun(result);
        const callsPerToken = Object.values(counts.perToken ?? {}).reduce(
          (a, b) => a + b,
          0,
        );
        // cpu=true adds: two performance.now() plus one cpuMs dictionary write.
        const perCallUs = costs.twoNow + costs.dictUpdate;
        const predictedMs = (callsPerToken * perCallUs) / 1000;
        const measuredMs = timed.decodeMsPerStep - counts.decodeMsPerStep;
        rows.push({
          model: result.model,
          callsPerToken,
          perCallUs,
          predictedMs,
          measuredMs,
          errorPct: predictedMs
            ? (100 * (measuredMs - predictedMs)) / predictedMs
            : null,
          relativePct: counts.decodeMsPerStep
            ? (100 * measuredMs) / counts.decodeMsPerStep
            : null,
        });
      }
      arithmetic = {
        ledger: LEDGER,
        rows,
        medianCallsPerToken: rows.length
          ? [...rows.map((row) => row.callsPerToken)].sort((a, b) => a - b)[
              Math.floor(rows.length / 2)
            ]
          : null,
      };
    } catch (error) {
      arithmetic = { error: String(error) };
    }

    return { costs, arithmetic };
  },

  report({ costs, arithmetic }) {
    console.log('Chrome cost per operation');
    console.log(`  performance.now() x1     ${costs.now.toFixed(3)} µs`);
    console.log(`  performance.now() x2     ${costs.twoNow.toFixed(3)} µs`);
    console.log(`  dict counter update      ${costs.dictUpdate.toFixed(3)} µs`);
    console.log(`  typed-array update       ${costs.typedUpdate.toFixed(3)} µs`);

    if (arithmetic?.rows?.length) {
      console.log('\nThe two count.mjs runs, reconciled');
      console.log(
        [
          'model'.padEnd(8),
          'calls/token'.padStart(12),
          'predicted'.padStart(11),
          'measured'.padStart(10),
          'error'.padStart(8),
          'share'.padStart(8),
        ].join(' '),
      );
      for (const row of arithmetic.rows) {
        console.log(
          [
            row.model.padEnd(8),
            String(row.callsPerToken).padStart(12),
            `${row.predictedMs.toFixed(2)}ms`.padStart(11),
            `${row.measuredMs.toFixed(2)}ms`.padStart(10),
            `${row.errorPct >= 0 ? '+' : ''}${row.errorPct.toFixed(0)}%`.padStart(8),
            `${row.relativePct.toFixed(1)}%`.padStart(8),
          ].join(' '),
        );
      }
    } else if (arithmetic?.error) {
      console.log(`\n[could not read ${LEDGER}: ${arithmetic.error}]`);
    }
  },

  ledger({ costs, arithmetic }) {
    return { costsUs: costs, arithmetic };
  },
});
