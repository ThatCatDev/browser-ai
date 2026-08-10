// src/device.ts
var adapter;
function gpuAvailable() {
  if (!adapter) {
    adapter = (async () => {
      const gpu = navigator.gpu;
      if (typeof gpu?.requestAdapter !== "function") return false;
      try {
        return !!await gpu.requestAdapter();
      } catch {
        return false;
      }
    })();
  }
  return adapter;
}
function forgetDevice() {
  adapter = void 0;
}
function bestDevice() {
  return navigator.gpu ? "webgpu" : "wasm";
}
async function resolveDevice(preference) {
  if (preference === "wasm") return { device: "wasm", fellBack: false };
  if (await gpuAvailable()) return { device: "webgpu", fellBack: false };
  return { device: "wasm", fellBack: preference === "webgpu" };
}
var Downloads = class {
  constructor() {
    this.files = /* @__PURE__ */ new Map();
  }
  record(file, at, of) {
    if (!of) return;
    this.files.set(file, { at, of });
  }
  /** Everything known so far, in bytes and as a fraction of itself. */
  state() {
    let loaded = 0;
    let total = 0;
    this.files.forEach((file) => {
      loaded += file.at;
      total += file.of;
    });
    return {
      loaded,
      total,
      fraction: total ? Math.min(1, loaded / total) : 0
    };
  }
  fraction() {
    return this.state().fraction;
  }
  /** Everything, arrived. What "ready" means in bytes. */
  finished() {
    const { total } = this.state();
    return { loaded: total, total, fraction: 1 };
  }
};
async function transformers() {
  const mod = await import("@huggingface/transformers");
  mod.env.allowLocalModels = false;
  if (mod.env.backends?.onnx?.wasm) mod.env.backends.onnx.wasm.numThreads = 1;
  return mod;
}

export {
  gpuAvailable,
  forgetDevice,
  bestDevice,
  resolveDevice,
  Downloads,
  transformers
};
//# sourceMappingURL=chunk-SUA6R43P.js.map