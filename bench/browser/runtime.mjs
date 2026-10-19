// The Node-side runtime shared by every browser driver.
//
// Each experiment supplies its URL, page evaluation and output path. This
// module owns the rest: Playwright resolution, the static server, Chrome
// flags, page listeners, cleanup, and the ledger envelope.

import { execFileSync, spawn } from 'node:child_process';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { cpus, loadavg, tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { makeLedger } from '../analysis/ledger.mjs';

const here = dirname(fileURLToPath(import.meta.url));

export function loadPlaywright() {
  const require = createRequire(
    process.env.PLAYWRIGHT_REQUIRE ?? '/Users/fenghanli/Projects/map-tile-lab/package.json',
  );
  return require('playwright');
}

export function systemLoad() {
  return loadavg().map((value) => Number(value.toFixed(2)));
}

export function announceLoad() {
  const load = systemLoad();
  console.log(`system load: ${load.join(' / ')} (${cpus().length} cores)`);
  return load;
}

export function gitCommit(root) {
  return execFileSync('git', ['rev-parse', '--short', 'HEAD'], {
    cwd: root,
    encoding: 'utf8',
  }).trim();
}

// One envelope for every driver. `load` is captured at startup, not here, so
// the ledger records the condition the run began under.
export function stampLedger({ root, config, load = systemLoad(), ...fields }) {
  return makeLedger({ commit: gitCommit(root), loadavg: load, config, ...fields });
}

export function writeLedger(out, ledger) {
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, JSON.stringify(ledger, null, 2));
}

async function waitForServer(port) {
  for (let i = 0; i < 100; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${port}/`);
      if (res.ok) return;
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error('server did not start');
}

export function startServer(port) {
  const server = spawn(process.execPath, [join(here, 'serve.mjs')], {
    env: { ...process.env, PORT: String(port) },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  server.stdout.on('data', (data) => process.stdout.write(`[server] ${data}`));
  server.stderr.on('data', (data) => process.stderr.write(`[server] ${data}`));
  return {
    process: server,
    ready: waitForServer(port),
    stop: () => server.kill('SIGTERM'),
  };
}

const CHROME_ARGS = [
  '--no-proxy-server',
  '--enable-unsafe-webgpu',
  '--disable-background-timer-throttling',
  '--disable-renderer-backgrounding',
];

// persistent=true gives the run its own profile directory, which is how the
// model-loading drivers isolate one model per context.
export async function launchChrome({
  headless = false,
  viewport = { width: 1100, height: 900 },
  label = 'fused-onnx',
  persistent = true,
} = {}) {
  const { chromium } = loadPlaywright();
  if (!persistent) {
    return chromium.launch({ channel: 'chrome', headless, args: CHROME_ARGS });
  }
  const userDataDir = mkdtempSync(join(tmpdir(), `${label}-`));
  return chromium.launchPersistentContext(userDataDir, {
    channel: 'chrome',
    headless,
    viewport,
    args: CHROME_ARGS,
  });
}

export async function instrumentPage(page, { prefix = 'page', onResponse = false } = {}) {
  page.setDefaultTimeout(0);
  await page.exposeFunction('__reportProgress', (message) => {
    console.log(`[${prefix}] ${message}`);
  });
  page.on('pageerror', (error) => console.log('[pageerror]', String(error).slice(0, 300)));
  page.on('console', (message) => {
    if (message.type() === 'error' || message.type() === 'warning') {
      console.log(`[console:${message.type()}]`, message.text().slice(0, 300));
    }
  });
  page.on('requestfailed', (request) => {
    const error = request.failure()?.errorText;
    // transformers.js aborts duplicate in-flight fetches after a cache hit.
    if (error !== 'net::ERR_ABORTED') console.log('[requestfailed]', request.url(), error);
  });
  if (onResponse) {
    page.on('response', (response) => {
      if (response.status() >= 400) console.log('[http]', response.status(), response.url());
    });
  }
  page.on('crash', () => console.log('[page] CRASHED'));
  return page;
}
