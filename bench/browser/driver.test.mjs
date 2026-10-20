// Tests for the driver seam. No browser, no server, no GPU: every driver is
// one `runDriver` call, so the plumbing can be exercised with a fake measure.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import {
  envFlag,
  envList,
  envNumber,
  fmt,
  ledgerPath,
  mean,
  median,
  quantile,
  root,
  runDriver,
} from './driver.mjs';

function withEnv(values, fn) {
  const saved = {};
  for (const [key, value] of Object.entries(values)) {
    saved[key] = process.env[key];
    if (value == null) delete process.env[key];
    else process.env[key] = value;
  }
  try {
    return fn();
  } finally {
    for (const [key, value] of Object.entries(saved)) {
      if (value == null) delete process.env[key];
      else process.env[key] = value;
    }
  }
}

test('envNumber: default, parse, and refuse a non-number', () => {
  withEnv({ TEST_NUMBER: null }, () => {
    assert.equal(envNumber('TEST_NUMBER', 7), 7);
  });
  withEnv({ TEST_NUMBER: '12' }, () => {
    assert.equal(envNumber('TEST_NUMBER', 7), 12);
  });
  withEnv({ TEST_NUMBER: 'twelve' }, () => {
    assert.throws(() => envNumber('TEST_NUMBER', 7), /must be a number/);
  });
});

test('envFlag and envList: the CLI conventions every driver shares', () => {
  withEnv({ TEST_FLAG: '1' }, () => assert.equal(envFlag('TEST_FLAG'), true));
  withEnv({ TEST_FLAG: '0' }, () => assert.equal(envFlag('TEST_FLAG'), false));
  withEnv({ TEST_FLAG: null }, () => assert.equal(envFlag('TEST_FLAG', true), true, 'fallback'));
  withEnv({ TEST_LIST: 'base, gate1,,gate2 ' }, () => {
    assert.deepEqual(envList('TEST_LIST', ['x']), ['base', 'gate1', 'gate2']);
  });
  withEnv({ TEST_LIST: null }, () => {
    assert.deepEqual(envList('TEST_LIST', ['x']), ['x']);
  });
});

test('ledgerPath: default under the repo root, explicit paths resolved', () => {
  withEnv({ TEST_OUT: null }, () => {
    assert.equal(ledgerPath('TEST_OUT', 'runs/x.json'), join(root, 'runs/x.json'));
  });
  withEnv({ TEST_OUT: 'runs/y.json' }, () => {
    assert.equal(ledgerPath('TEST_OUT', 'runs/x.json'), join(process.cwd(), 'runs/y.json'));
  });
});

test('summary statistics: mean, median, quantile', () => {
  assert.equal(mean([1, 2, 3]), 2);
  assert.equal(mean([]), null);
  assert.equal(median([3, 1, 2]), 2);
  assert.equal(median([4, 1, 2, 3]), 2.5);
  assert.equal(quantile([5, 1, 3, 2, 4], 0.5), 3);
  assert.equal(quantile([1], 0.9), 1);
});

test('fmt: em dash for missing values, fixed digits otherwise', () => {
  assert.equal(fmt(null), '—');
  assert.equal(fmt(Number.NaN), '—');
  assert.equal(fmt(1.2345, 2), '1.23');
});

test('runDriver: one call writes the ledger and returns the payload', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'browser-decode-budget-driver-'));
  const out = join(dir, 'nested', 'run.json');
  try {
    const returned = await runDriver({
      name: 'test',
      out,
      server: false,
      config: { answer: 42 },
      async measure() {
        return { results: [{ model: 'base' }] };
      },
      checks() {
        return [{ name: 'fake', ok: true, detail: 'ok' }];
      },
      ledger({ results }) {
        return { results };
      },
    });

    const written = JSON.parse(readFileSync(out, 'utf8'));
    assert.equal(written.config.answer, 42);
    assert.deepEqual(written.results, [{ model: 'base' }]);
    assert.deepEqual(returned.payload.results, [{ model: 'base' }]);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('runDriver: a failed check fails the run after writing the ledger', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'browser-decode-budget-driver-'));
  const out = join(dir, 'run.json');
  const savedExitCode = process.exitCode;
  try {
    process.exitCode = 0;
    await runDriver({
      name: 'test',
      out,
      server: false,
      async measure() {
        return {};
      },
      checks() {
        return [{ name: 'broken', ok: false, detail: 'the harness lied' }];
      },
    });
    assert.equal(process.exitCode, 1);
    assert.ok(readFileSync(out, 'utf8').length > 0, 'ledger written before failing');
  } finally {
    process.exitCode = savedExitCode;
    rmSync(dir, { recursive: true, force: true });
  }
});
