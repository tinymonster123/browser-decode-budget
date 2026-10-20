// Shared encoder/timestamp scaffolding for the pages that time compute passes.
//
// Both gpu-profile.html (one pass per ORT pass) and op-attribution.html (one
// pass per dispatch) create the same query set, resolve buffer and read buffer
// per encoder, then map the timestamps back after the run. That fragile part
// lives here so the two pages only choose when a pass starts and ends.

// An encoder holds ~13 dispatches; op-attribution gives each its own pass, so
// 26 slots suffice. 128 leaves headroom for encoders that batch more.
export const MAX_QUERIES = 128;

export function newEncoderRecord(device) {
  return {
    qs: device.createQuerySet({ type: 'timestamp', count: MAX_QUERIES }),
    resolveBuf: device.createBuffer({
      size: MAX_QUERIES * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    }),
    readBuf: device.createBuffer({
      size: MAX_QUERIES * 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    }),
    n: 0,
    passes: [],
    overflow: false,
  };
}

// `rec.passes` holds plain records with at least { seq, slot }. The caller
// decides what else to attach; `rec` itself must never be attached, because
// the page returns these objects across the Playwright boundary.
export async function collectSegments(records, device) {
  if (!device) return Object.assign([], { overflowEncoders: 0 });
  await device.queue.onSubmittedWorkDone();
  const segments = [];
  let overflow = 0;

  for (const rec of records) {
    if (rec.overflow) overflow += 1;
    if (!rec.n) continue;
    try {
      await rec.readBuf.mapAsync(GPUMapMode.READ, 0, rec.n * 8);
      const view = new BigInt64Array(rec.readBuf.getMappedRange(0, rec.n * 8));
      for (const meta of rec.passes) {
        const startNs = Number(view[meta.slot]);
        const endNs = Number(view[meta.slot + 1]);
        segments.push({
          ...meta,
          startNs,
          endNs,
          us: (endNs - startNs) / 1000,
        });
      }
      rec.readBuf.unmap();
    } catch (error) {
      console.warn('[timeline] map failed', error);
    }
  }

  segments.sort((a, b) => a.seq - b.seq);
  segments.overflowEncoders = overflow;
  return segments;
}
