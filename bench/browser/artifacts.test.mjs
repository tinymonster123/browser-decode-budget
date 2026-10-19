// Tests for the artifact registry. No browser, no model files.

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { artifact, artifactNames, modelRoute, nodeCounts } from './artifacts.mjs';

test('artifact: names resolve to one root, node count and dtype', () => {
  assert.deepEqual(artifactNames(), ['base', 'gate1', 'gate2']);

  const gate2 = artifact('gate2');
  assert.equal(gate2.nodes, 1078);
  assert.equal(gate2.dtype, 'q4f16');
  assert.match(gate2.root, /gemma4-e2b-q4f16-gate2$/);
});

test('artifact: an unknown name fails loudly', () => {
  assert.throws(() => artifact('fp16'), /unknown artifact "fp16"/);
});

test('artifact: MODEL_<NAME> overrides the default root', () => {
  const previous = process.env.MODEL_GATE1;
  process.env.MODEL_GATE1 = '/tmp/some-artifact';
  try {
    assert.equal(artifact('gate1').root, '/tmp/some-artifact');
  } finally {
    if (previous === undefined) delete process.env.MODEL_GATE1;
    else process.env.MODEL_GATE1 = previous;
  }
});

test('nodeCounts: every artifact has a count', () => {
  assert.deepEqual(nodeCounts(), { base: 1289, gate1: 1233, gate2: 1078 });
});

test('modelRoute: URL prefix to artifact and relative path', () => {
  assert.deepEqual(modelRoute('/gate2/onnx/decoder.onnx'), {
    name: 'gate2',
    relative: 'onnx/decoder.onnx',
  });
  assert.equal(modelRoute('/vendor/transformers.web.js'), null);
  assert.equal(modelRoute('/gate2'), null);
});
