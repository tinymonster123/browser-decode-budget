// The one entry point every measurement driver calls.
//
// A driver supplies what to measure, how to print it, its self-checks and its
// extra ledger fields. This module owns everything else: environment parsing,
// the static server, the page lifecycle, the ledger envelope and the exit
// code. Drivers never import `runtime.mjs` directly, so the browser plumbing
// has one caller-facing seam.

import { dirname, isAbsolute, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  announceLoad,
  launchChrome,
  openPage,
  startServer,
  stampLedger,
  writeLedger,
} from './runtime.mjs';

const here = dirname(fileURLToPath(import.meta.url));
export const root = resolve(here, '../..');

function env(name) {
  const value = process.env[name];
  return value == null || value === '' ? null : value;
}

export function envNumber(name, fallback) {
  const raw = env(name);
  if (raw == null) return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value)) {
    throw new Error(`${name} must be a number, got "${raw}"`);
  }
  return value;
}

export function envFlag(name, fallback = false) {
  const raw = env(name);
  return raw == null ? fallback : raw === '1' || raw === 'true';
}

export function envText(name, fallback) {
  return env(name) ?? fallback;
}

export function envList(name, fallback) {
  const raw = env(name);
  return raw
    ? raw
        .split(',')
        .map((value) => value.trim())
        .filter(Boolean)
    : fallback;
}

// OUT may be absolute or relative to the caller's working directory, like any
// other CLI path; the default is relative to the repository root.
export function ledgerPath(name, fallback) {
  const raw = env(name);
  if (!raw) return join(root, fallback);
  return isAbsolute(raw) ? raw : resolve(raw);
}

export function fmt(value, digits = 2) {
  if (value == null || !Number.isFinite(value)) return '—';
  return value.toFixed(digits);
}

export function mean(values) {
  if (!values.length) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export function median(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

export function quantile(values, fraction) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(
    sorted.length - 1,
    Math.floor(fraction * sorted.length),
  );
  return sorted[index];
}

function printChecks(checks) {
  if (!checks.length) return;
  console.log('\nself-checks');
  for (const check of checks) {
    console.log(
      `  [${check.ok ? 'ok' : 'FAIL'}] ${check.name}: ${check.detail ?? ''}`,
    );
  }
}

// `checks` returns `[{ name, ok, detail }]`; a single failed check fails the
// run after the ledger is written, so a broken harness cannot pass silently.
export async function runDriver({
  name,
  port = 8123,
  out,
  config = {},
  server = true,
  measure,
  report = () => {},
  checks = () => [],
  ledger = () => ({}),
}) {
  const load = announceLoad();
  const headless = envFlag('HEADLESS');
  const service = server ? startServer(port) : null;

  const ctx = {
    root,
    name,
    port,
    headless,
    log: (...args) => console.log(...args),

    // Open a fresh context and page, run `fn` against it, always close it.
    // A fresh context per model is what keeps one model's GPU allocations
    // from biasing the next one.
    async withPage(options, fn) {
      // A self-contained page (no server, no model) uses a throwaway browser.
      if (options.content != null) {
        const browser = await launchChrome({
          headless: options.headless ?? headless,
          persistent: false,
        });
        try {
          const page = await browser.newPage();
          await page.setContent(options.content);
          return await fn(page);
        } finally {
          await browser.close();
        }
      }

      const url = options.path
        ? `http://127.0.0.1:${port}${options.path}`
        : options.url;
      const opened = await openPage({
        headless,
        label: name,
        prefix: 'page',
        ...options,
        url,
      });
      try {
        return await fn(opened.page);
      } finally {
        await opened.context.close();
      }
    },

    // The count/profile shape: one model, one page function, one result.
    async measureModel(model, { path, call, options = {}, ...pageOptions } = {}) {
      return ctx.withPage(
        { path, label: `${name}-${model}`, prefix: model, ...pageOptions },
        (page) =>
          page.evaluate(
            (input) => window[input.call](input.options),
            { call, options },
          ),
      );
    },
  };

  try {
    await service?.ready;

    const payload = await measure(ctx);
    report(payload, ctx);

    const checkResults = checks(payload, ctx) ?? [];
    printChecks(checkResults);

    const stamped = stampLedger({
      root,
      config: server ? { ...config, port, headless } : config,
      load,
      ...ledger(payload, ctx),
    });
    writeLedger(out, stamped);
    console.log(`\nwrote ${out}`);

    if (checkResults.some((check) => !check.ok)) process.exitCode = 1;
    return { payload, ledger: stamped, checks: checkResults };
  } finally {
    service?.stop();
  }
}
