// Static server for the Gate 3 page and the two model roots.
//
//   node bench/browser/serve.mjs
//
// Routes:
//   /                 -> index.html
//   /vendor/...       -> prepared runtime assets
//   /base/...         -> original onnx-community model root
//   /gate2/...        -> fused Gate 2 artifact root

import { createReadStream, existsSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { homedir } from 'node:os';
import { dirname, extname, join, normalize, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT ?? 8123);

const BASE = resolve(
  process.env.MODEL_BASE ??
    join(homedir(), 'Projects/gemma-browser-lab/models/onnx-community/gemma-4-E2B-it-ONNX'),
);
const GATE1 = resolve(
  process.env.MODEL_GATE1 ??
    join(here, '../../models/derived/gemma4-e2b-q4f16-gate1'),
);
const GATE2 = resolve(
  process.env.MODEL_GATE2 ??
    join(here, '../../models/derived/gemma4-e2b-q4f16-gate2'),
);
const VENDOR = join(here, 'vendor');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.wasm': 'application/wasm',
  '.json': 'application/json; charset=utf-8',
  '.jinja': 'text/plain; charset=utf-8',
  '.onnx': 'application/octet-stream',
  '.onnx_data': 'application/octet-stream',
  '.bin': 'application/octet-stream',
};

function safeJoin(root, relative) {
  const target = resolve(root, '.' + sep + normalize(relative));
  if (target !== root && !target.startsWith(root + sep)) return null;
  return target;
}

function sendFile(req, res, path) {
  const stat = statSync(path);
  const type = MIME[extname(path)] ?? 'application/octet-stream';
  const range = req.headers.range;

  const headers = {
    'Content-Type': type,
    'Accept-Ranges': 'bytes',
    'Cache-Control': 'no-store',
    'Access-Control-Allow-Origin': '*',
    'Cross-Origin-Resource-Policy': 'cross-origin',
  };

  if (range) {
    const match = /^bytes=(\d*)-(\d*)$/.exec(range);
    if (match) {
      const start = match[1] ? Number(match[1]) : 0;
      const end = match[2] ? Number(match[2]) : stat.size - 1;
      if (start <= end && end < stat.size) {
        res.writeHead(206, {
          ...headers,
          'Content-Range': `bytes ${start}-${end}/${stat.size}`,
          'Content-Length': end - start + 1,
        });
        createReadStream(path, { start, end }).pipe(res);
        return;
      }
    }
  }

  res.writeHead(200, { ...headers, 'Content-Length': stat.size });
  createReadStream(path).pipe(res);
}

const server = createServer((req, res) => {
  // The wasm runtime uses SharedArrayBuffer; COOP/COEP keeps it cross-origin isolated.
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');

  const url = new URL(req.url ?? '/', `http://127.0.0.1:${PORT}`);
  const pathname = decodeURIComponent(url.pathname);

  let file = null;
  if (pathname === '/' || pathname === '/index.html') {
    file = join(here, 'index.html');
  } else if (/^\/[A-Za-z0-9_-]+\.html$/.test(pathname)) {
    file = safeJoin(here, pathname.slice(1));
  } else if (pathname.startsWith('/vendor/')) {
    file = safeJoin(VENDOR, pathname.slice('/vendor/'.length));
  } else if (pathname.startsWith('/base/')) {
    file = safeJoin(BASE, pathname.slice('/base/'.length));
  } else if (pathname.startsWith('/gate1/')) {
    file = safeJoin(GATE1, pathname.slice('/gate1/'.length));
  } else if (pathname.startsWith('/gate2/')) {
    file = safeJoin(GATE2, pathname.slice('/gate2/'.length));
  }

  if (!file || !existsSync(file) || !statSync(file).isFile()) {
    res.writeHead(404, { 'Content-Type': 'text/plain' });
    res.end(`not found: ${pathname}\n`);
    return;
  }

  try {
    sendFile(req, res, file);
  } catch (error) {
    res.writeHead(500, { 'Content-Type': 'text/plain' });
    res.end(String(error));
  }
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`serving on http://127.0.0.1:${PORT}/`);
  console.log(`  base  -> ${BASE}`);
  console.log(`  gate1 -> ${GATE1}`);
  console.log(`  gate2 -> ${GATE2}`);
});
