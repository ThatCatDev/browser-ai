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

/**
 * How much of a download has arrived.
 *
 * The fraction on its own is not enough to draw a bar with, and the reason is
 * worth stating: `total` is the bytes of the files *discovered so far*, and
 * files are discovered in the order they are needed. A model is a handful of
 * small JSON files and one enormous one, so for the first second the fraction
 * is a fraction of a few hundred kilobytes — it climbs to nearly 1, and then
 * collapses when the weights are finally announced. Only the bytes say whether
 * 90% means ninety per cent of the model or ninety per cent of its tokenizer.
 */
export interface Loading {
  /** Bytes that have arrived, across every file seen so far. */
  loaded: number;
  /** Bytes known to be coming. Grows as files are discovered. */
  total: number;
  /** `loaded / total` — of what is known, which is not always the whole. */
  fraction: number;
}

/**
 * Told how a download is going, as often as the runtime says so.
 *
 * The fraction comes first because it is what most callers want, and what this
 * was before the bytes were there to be had.
 */
export type Progress = (fraction: number, detail: Loading) => void;

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

  /** Everything known so far, in bytes and as a fraction of itself. */
  state(): Loading {
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

  fraction(): number {
    return this.state().fraction;
  }

  /** Everything, arrived. What "ready" means in bytes. */
  finished(): Loading {
    const { total } = this.state();
    return { loaded: total, total, fraction: 1 };
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
