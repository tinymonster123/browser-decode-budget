// The WebGPU patch protocol shared by count.html and gpu-profile.html.
//
// Both pages need the same fragile part: intercept the adapter ORT will use,
// wrap the prototype methods idempotently, and also reach the prototypes from
// a throwaway device. What they do inside the wrappers (count calls, inject
// timestamps) stays in each page.
//
// Untested by design, like the rest of the browser harness.

export const WRAPPED = Symbol('browser-decode-budget.wrapped');

export function wrapMethod(proto, name, factory) {
  if (!proto) return false;
  let desc;
  try {
    desc = Object.getOwnPropertyDescriptor(proto, name);
  } catch {
    return false;
  }
  if (!desc || typeof desc.value !== 'function') return false;
  if (desc.value[WRAPPED]) return true;

  const wrapped = factory(desc.value);
  wrapped[WRAPPED] = true;
  try {
    Object.defineProperty(proto, name, { ...desc, value: wrapped });
  } catch {
    try {
      proto[name] = wrapped;
    } catch {
      return false;
    }
  }
  return true;
}

export function prototypesOf(device) {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.end();
  return {
    device: Object.getPrototypeOf(device),
    encoder: Object.getPrototypeOf(encoder),
    pass: Object.getPrototypeOf(pass),
    queue: Object.getPrototypeOf(device.queue),
  };
}

// Intercept navigator.gpu.requestAdapter so `onDevice` runs as soon as the
// real device exists. `extendDescriptor` may add required features.
export function interceptDevice({ onDevice, extendDescriptor }) {
  const gpu = navigator.gpu;
  if (!gpu || !gpu.requestAdapter) return false;

  const origRequestAdapter = gpu.requestAdapter.bind(gpu);
  gpu.requestAdapter = async function (...args) {
    const adapter = await origRequestAdapter(...args);
    if (!adapter || adapter[WRAPPED]) return adapter;

    const origRequestDevice = adapter.requestDevice.bind(adapter);
    adapter.requestDevice = async function (desc = {}) {
      const finalDesc = extendDescriptor ? extendDescriptor(desc, adapter) : desc;
      const device = await origRequestDevice(finalDesc);
      try {
        onDevice(device);
      } catch (error) {
        console.warn('WebGPU patch on requestDevice failed', error);
      }
      return device;
    };
    adapter[WRAPPED] = true;
    return adapter;
  };
  return true;
}

// A second route to the prototypes, in case ORT grabbed its adapter before
// the hook was installed. The device is destroyed after patching.
export async function withThrowawayDevice(patch) {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return false;
  const device = await adapter.requestDevice();
  try {
    patch(device);
  } finally {
    device.destroy();
  }
  return true;
}
