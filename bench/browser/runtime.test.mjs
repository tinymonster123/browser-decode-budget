// Tests for the Node-side driver runtime. No browser, no server, no GPU.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { gitCommit, stampLedger, systemLoad, writeLedger } from './runtime.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '../..');

test('systemLoad: three finite numbers', () => {
  const load = systemLoad();

  assert.equal(load.length, 3);
  assert.ok(load.every((value) => Number.isFinite(value)));
});

test('stampLedger: one envelope, caller load preserved', () => {
  const ledger = stampLedger({
    root,
    config: { port: 8124 },
    load: [1, 2, 3],
    results: [{ model: 'base' }],
  });

  assert.equal(ledger.commit, gitCommit(root));
  assert.deepEqual(ledger.loadavg, [1, 2, 3]);
  assert.deepEqual(ledger.config, { port: 8124 });
  assert.deepEqual(ledger.results, [{ model: 'base' }]);
  assert.match(ledger.timestamp, /^\d{4}-\d{2}-\d{2}T/);
});

test('writeLedger: creates the parent directory and writes JSON', () => {
  const dir = mkdtempSync(join(tmpdir(), 'fused-onnx-ledger-'));
  try {
    const out = join(dir, 'nested', 'run.json');
    writeLedger(out, { commit: 'abc1234', results: [] });

    assert.deepEqual(JSON.parse(readFileSync(out, 'utf8')), {
      commit: 'abc1234',
      results: [],
    });
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
