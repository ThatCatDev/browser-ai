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
  fraction() {
    let at = 0;
    let of = 0;
    this.files.forEach((file) => {
      at += file.at;
      of += file.of;
    });
    return of ? Math.min(1, at / of) : 0;
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
//# sourceMappingURL=chunk-B5A4XJ7T.js.map