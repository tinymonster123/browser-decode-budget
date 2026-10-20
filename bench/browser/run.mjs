// Gate 3 driver: serve the two model roots, drive Chrome, collect paired A/B
// decode timings and the generated-text equivalence check.
//
//   node bench/browser/prepare.mjs   # once
//   node bench/browser/run.mjs
//
// Env:
//   REPEATS=4 MAX_NEW=64 HEADLESS=1 PORT=8123 MODEL_BASE=... MODEL_GATE2=...

import { artifact } from './artifacts.mjs';
import {
  envNumber,
  envText,
  ledgerPath,
  median,
  runDriver,
} from './driver.mjs';

const REPEATS = Math.max(1, envNumber('REPEATS', 4));
const MAX_NEW = Math.max(1, envNumber('MAX_NEW', 64));
const A_MODEL = envText('A_MODEL', 'base');
const A_NAME = envText('A_NAME', A_MODEL);
const B_MODEL = envText('B_MODEL', 'gate2');
const B_NAME = envText('B_NAME', B_MODEL);

// Names must exist in the registry: the server only routes registered roots.
artifact(A_MODEL);
artifact(B_MODEL);

function summarize(rounds) {
  const byName = new Map();
  for (const run of rounds) {
    if (!run.tokens || run.firstTokenMs == null || run.lastTokenMs == null) continue;
    const decodeMs = run.lastTokenMs - run.firstTokenMs;
    const msPerToken = run.tokens > 1 ? decodeMs / (run.tokens - 1) : null;
    const entry = byName.get(run.name) ?? {
      runs: [],
      msPerToken: [],
      totalMs: [],
      tokens: [],
    };
    entry.runs.push(run);
    if (msPerToken != null) entry.msPerToken.push(msPerToken);
    entry.totalMs.push(run.ms);
    entry.tokens.push(run.tokens);
    byName.set(run.name, entry);
  }
  const summary = {};
  for (const [name, entry] of byName) {
    summary[name] = {
      runs: entry.runs.length,
      medianMsPerToken: median(entry.msPerToken),
      medianTotalMs: median(entry.totalMs),
      medianTokens: median(entry.tokens),
      samplesMsPerToken: entry.msPerToken.map((value) => Number(value.toFixed(3))),
    };
  }
  return summary;
}

await runDriver({
  name: 'bench',
  port: envNumber('PORT', 8123),
  out: ledgerPath('OUT', 'runs/2026-10-19-gate3-base-vs-gate2.json'),
  config: {
    repeats: REPEATS,
    maxNewTokens: MAX_NEW,
    aModel: A_MODEL,
    aName: A_NAME,
    bModel: B_MODEL,
    bName: B_NAME,
  },

  async measure(ctx) {
    ctx.log(
      `launching: repeats=${REPEATS} max_new_tokens=${MAX_NEW} ` +
        `headless=${ctx.headless} a=${A_MODEL}/${A_NAME} b=${B_MODEL}/${B_NAME}`,
    );
    const result = await ctx.withPage(
      {
        path: '/',
        label: `${ctx.name}-ab`,
        prefix: 'page',
        onResponse: true,
        onReadyTimeout: async (page) => {
          console.log('[page] __ready never became true');
          await page.screenshot({
            path: '/tmp/browser-decode-budget-bench-failed.png',
            fullPage: true,
          });
        },
      },
      (page) =>
        page.evaluate(
          (options) => window.__bench(options),
          {
            repeats: REPEATS,
            maxNewTokens: MAX_NEW,
            aModel: A_MODEL,
            aName: A_NAME,
            bModel: B_MODEL,
            bName: B_NAME,
          },
        ),
    );
    return { result, summary: summarize(result.rounds ?? []) };
  },

  checks({ result }) {
    return [
      {
        name: 'page',
        ok: !result.error,
        detail: result.error ?? 'bench completed',
      },
    ];
  },

  report({ result, summary }) {
    if (result.error) {
      console.error(`page error: ${result.error}`);
      if (result.stack) console.error(result.stack);
    }

    const a = summary[A_NAME]?.medianMsPerToken;
    const b = summary[B_NAME]?.medianMsPerToken;
    const delta = a && b ? (100 * (b - a)) / a : null;

    console.log('\nvariant   median ms/token   median total ms   tokens');
    for (const name of [A_NAME, B_NAME]) {
      const row = summary[name];
      if (!row) continue;
      console.log(
        `${name.padEnd(9)} ${String(row.medianMsPerToken?.toFixed(2)).padStart(15)} ` +
          `${String(row.medianTotalMs).padStart(16)} ${String(row.medianTokens).padStart(8)}`,
      );
    }
    if (delta != null) {
      console.log(
        `${B_NAME} vs ${A_NAME}: ${delta >= 0 ? '+' : ''}${delta.toFixed(1)}%`,
      );
    }

    if (result.equivalence) {
      console.log(`\nequivalence: exact=${result.equivalence.exact}`);
      if (!result.equivalence.exact) {
        console.log(`  ${A_NAME}: ${JSON.stringify(result.equivalence.aText.slice(0, 400))}`);
        console.log(`  ${B_NAME}: ${JSON.stringify(result.equivalence.bText.slice(0, 400))}`);
      }
    }
  },

  ledger({ result, summary }) {
    return { summary, result };
  },
});
