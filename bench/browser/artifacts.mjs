// The artifact registry: one name → one root, node count and dtype.
//
// The name is what the pages and drivers pass around; the server turns it into
// a URL prefix and the drivers read its node count. Adding a variant (a new
// dtype, a new fusion stage) is one entry here.

import { homedir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '../..');

const DEFINITIONS = {
  base: {
    nodes: 1289,
    dtype: 'q4f16',
    defaultRoot: join(
      homedir(),
      'Projects/gemma-browser-lab/models/onnx-community/gemma-4-E2B-it-ONNX',
    ),
  },
  gate1: {
    nodes: 1233,
    dtype: 'q4f16',
    defaultRoot: join(root, 'models/derived/gemma4-e2b-q4f16-gate1'),
  },
  gate2: {
    nodes: 1078,
    dtype: 'q4f16',
    defaultRoot: join(root, 'models/derived/gemma4-e2b-q4f16-gate2'),
  },
};

export function artifactNames() {
  return Object.keys(DEFINITIONS);
}

export function artifact(name) {
  const definition = DEFINITIONS[name];
  if (!definition) {
    throw new Error(`unknown artifact "${name}"; known: ${artifactNames().join(', ')}`);
  }
  const override = process.env[`MODEL_${name.toUpperCase()}`];
  return {
    name,
    root: resolve(override ?? definition.defaultRoot),
    nodes: definition.nodes,
    dtype: definition.dtype,
  };
}

export function nodeCounts() {
  return Object.fromEntries(
    artifactNames().map((name) => [name, DEFINITIONS[name].nodes]),
  );
}

// Turn a request pathname into an artifact and a path relative to its root.
export function modelRoute(pathname) {
  for (const name of artifactNames()) {
    const prefix = `/${name}/`;
    if (pathname.startsWith(prefix)) {
      return { name, relative: pathname.slice(prefix.length) };
    }
  }
  return null;
}
