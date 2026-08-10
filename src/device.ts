/**
 * Where the arithmetic happens, and how to find out honestly.
 *
 * Two backends, one set of weights: WebGPU where the machine has it, WASM where
 * it does not. That is the whole reason this uses `transformers.js` rather than
 * pairing a GPU-only runtime with a CPU-only one — two runtimes means two model
 * formats, two caches, and twice the bandwidth for the same feature.
 *
 * Threads are deliberately never enabled. Multi-threaded WASM needs
 * `SharedArrayBuffer`, which needs `Cross-Origin-Embedder-Policy` on the
 * document — and that policy refuses to embed any cross-origin frame whose
 * origin does not opt in. A library should not be able to break its host's
 * iframes in exchange for being faster.
 */

export type Device = "webgpu" | "wasm";

/** What was asked for, which is not always what the machine can give. */
export type DevicePreference = "auto" | Device;

/** Fraction of a download that has arrived, 0 to 1. */
export type Progress = (fraction: number) => void;

/**
 * Whether this browser can *actually* run on the GPU.
 *
 * Not the same question as whether `navigator.gpu` exists, which is the trap
 * worth knowing about: Chrome ships the API and then refuses an adapter on a
 * long list of older Intel parts and Linux drivers. Checking only for the
 * object means the model fails on its way up, and the failure surfaces as
 * something the app can only report as a total loss — on machines where the CPU
 * path would have worked perfectly.
 *
 * Asked once, and kept: it is a negotiation with the driver, not a property
 * lookup, and the answer does not change while the page is open.
 */
let adapter: Promise<boolean> | undefined;

export function gpuAvailable(): Promise<boolean> {
  if (!adapter) {
    adapter = (async () => {
      const gpu = (
        navigator as Navigator & { gpu?: { requestAdapter?: () => Promise<unknown> } }
      ).gpu;
      if (typeof gpu?.requestAdapter !== "function") return false;
      try {
        return !!(await gpu.requestAdapter());
      } catch {
        return false;
      }
    })();
  }
  return adapter;
}

/** Forget the cached answer. For tests, and for anything that reloads a page. */
export function forgetDevice(): void {
  adapter = undefined;
}

/**
 * What this browser claims, cheaply and synchronously.
 *
 * Only ever a claim — anything about to load a model should wait for
 * `resolveDevice`. Useful for drawing a control before the driver has answered.
 */
export function bestDevice(): Device {
  return (navigator as Navigator & { gpu?: unknown }).gpu ? "webgpu" : "wasm";
}

/**
 * The device to use, and whether the answer disappointed anybody.
 *
 * Falling back is right; falling back silently is not. Somebody who asked for
 * the GPU and got the CPU should be able to be told why it is slow.
 */
export async function resolveDevice(
  preference: DevicePreference
): Promise<{ device: Device; fellBack: boolean }> {
  if (preference === "wasm") return { device: "wasm", fellBack: false };
  if (await gpuAvailable()) return { device: "webgpu", fellBack: false };
  return { device: "wasm", fellBack: preference === "webgpu" };
}

/**
 * A running total across however many files a model turns out to be.
 *
 * The library reports progress per file, and a bar that restarts for each one
 * is worse than no bar at all.
 */
export class Downloads {
  private readonly files = new Map<string, { at: number; of: number }>();

  record(file: string, at: number, of: number): void {
    if (!of) return;
    this.files.set(file, { at, of });
  }

  fraction(): number {
    let at = 0;
    let of = 0;
    this.files.forEach((file) => {
      at += file.at;
      of += file.of;
    });
    return of ? Math.min(1, at / of) : 0;
  }
}

/** Everything the pipelines need from `transformers.js`, loaded on demand. */
export async function transformers() {
  /*
   * Imported here rather than at the top of a module: the library is a few
   * hundred kilobytes of JavaScript before a single weight is fetched, and an
   * app that never runs a model should never pay for it.
   */
  const mod = await import("@huggingface/transformers");
  mod.env.allowLocalModels = false;
  if (mod.env.backends?.onnx?.wasm) mod.env.backends.onnx.wasm.numThreads = 1;
  return mod;
}
