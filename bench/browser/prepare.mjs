// Copy the browser runtime assets from the gemma-translate checkout.
//
//   node bench/browser/prepare.mjs
//
// The assets are build inputs, not vendored source: transformers.web.js comes
// from the installed @huggingface/transformers package, and the ORT wasm files
// are the same ones gemma-translate copies into public/ort/.

import { copyFileSync, cpSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const gemma = resolve(here, '../../../gemma-translate');
const vendor = join(here, 'vendor');
const ort = join(vendor, 'ort');
const ortWeb = join(vendor, 'ort-web');

if (!existsSync(gemma)) {
  throw new Error(`gemma-translate checkout not found: ${gemma}`);
}

mkdirSync(ort, { recursive: true });
mkdirSync(ortWeb, { recursive: true });

const sources = [
  [
    join(gemma, 'node_modules/@huggingface/transformers/dist/transformers.web.js'),
    join(vendor, 'transformers.web.js'),
  ],
  [
    join(gemma, 'public/ort/ort-wasm-simd-threaded.asyncify.mjs'),
    join(ort, 'ort-wasm-simd-threaded.asyncify.mjs'),
  ],
  [
    join(gemma, 'public/ort/ort-wasm-simd-threaded.asyncify.wasm'),
    join(ort, 'ort-wasm-simd-threaded.asyncify.wasm'),
  ],
  // transformers.web.js keeps `onnxruntime-web/webgpu` as a bare import;
  // the import map in index.html points it at this bundle.
  [
    join(
      gemma,
      'node_modules/onnxruntime-web/dist/ort.webgpu.bundle.min.mjs',
    ),
    join(ortWeb, 'ort.webgpu.bundle.min.mjs'),
  ],
];

for (const [from, to] of sources) {
  if (!existsSync(from)) throw new Error(`missing asset: ${from}`);
  copyFileSync(from, to);
  console.log(`${from} -> ${to}`);
}

// transformers.web.js also imports `Tensor` from the bare specifier
// `onnxruntime-common`; serve its ESM build behind the import map.
const commonFrom = join(gemma, 'node_modules/onnxruntime-common/dist/esm');
const commonTo = join(ortWeb, 'onnxruntime-common');
cpSync(commonFrom, commonTo, { recursive: true });
console.log(`${commonFrom} -> ${commonTo}`);
