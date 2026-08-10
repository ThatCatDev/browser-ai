import {
  Downloads,
  resolveDevice,
  transformers,
  type Device,
  type Progress
} from "./device";

/**
 * Turning text into vectors, which is most of what retrieval is.
 *
 * MiniLM at eight bits is about 23MB — small enough that a first search is not
 * an event, and the reason retrieval is worth doing in a browser at all. It is
 * eight bits rather than four because the accuracy of the ranking is the entire
 * point: a search that confidently returns the wrong thing is not a feature.
 */
export const EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";

export interface Embedder {
  /** The device it settled on, once loaded. */
  readonly device: Device | undefined;
  /** Bring the model in. Safe to call twice; the second call waits on the first. */
  load(onProgress?: Progress): Promise<void>;
  /** One vector per string, normalised, so similarity is a dot product. */
  embed(texts: string[]): Promise<Float32Array[]>;
  dispose(): void;
}

class TransformersEmbedder implements Embedder {
  device: Device | undefined;
  private pipeline?: (
    texts: string[],
    options: { pooling: "mean"; normalize: boolean }
  ) => Promise<{ tolist(): number[][] }>;
  private loading?: Promise<void>;

  constructor(private readonly model: string = EMBEDDING_MODEL) {}

  load(onProgress?: Progress): Promise<void> {
    if (!this.loading) this.loading = this.bring(onProgress);
    return this.loading;
  }

  private async bring(onProgress?: Progress): Promise<void> {
    const downloads = new Downloads();
    const { pipeline } = await transformers();
    const { device } = await resolveDevice("auto");

    const extractor = await pipeline("feature-extraction", this.model, {
      device,
      dtype: "q8",
      progress_callback: (event: {
        status: string;
        file?: string;
        loaded?: number;
        total?: number;
      }) => {
        if (!onProgress) return;
        if (event.status === "progress" && event.file) {
          downloads.record(event.file, event.loaded ?? 0, event.total ?? 0);
          const state = downloads.state();
          onProgress(state.fraction, state);
        }
        if (event.status === "ready") onProgress(1, downloads.finished());
      }
    });

    this.device = device;
    this.pipeline = extractor as unknown as typeof this.pipeline;
  }

  async embed(texts: string[]): Promise<Float32Array[]> {
    if (!texts.length) return [];
    if (!this.pipeline) throw new Error("Embedder used before it was loaded");

    // Pooled and normalised inside the graph, so what comes back is ready to
    // compare with a dot product.
    const output = await this.pipeline(texts, { pooling: "mean", normalize: true });
    return output.tolist().map((row) => Float32Array.from(row));
  }

  dispose(): void {
    this.pipeline = undefined;
    this.loading = undefined;
    this.device = undefined;
  }
}

let shared: Embedder | undefined;

/** The embedder for this page. One per page; the weights are not cheap. */
export function embedder(): Embedder {
  if (!shared) shared = new TransformersEmbedder();
  return shared;
}

/** Swap the implementation — for tests, and for whatever replaces this one. */
export function setEmbedder(replacement: Embedder | undefined): void {
  shared = replacement;
}

/**
 * How alike two vectors are, from -1 to 1.
 *
 * A plain dot product, because everything here is normalised on the way out of
 * the model — dividing by two lengths that are both 1 is work for nothing.
 */
export function similarity(a: Float32Array, b: Float32Array): number {
  const length = Math.min(a.length, b.length);
  let total = 0;
  for (let i = 0; i < length; i++) total += a[i] * b[i];
  return total;
}
