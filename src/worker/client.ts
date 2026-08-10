import type { ChatEngine, ChatOptions, Message } from "../chat";
import type { Embedder } from "../embed";
import type { Device, DevicePreference, Progress } from "../device";
import type { Request, Response } from "./protocol";

/**
 * `Omit` over a union, which the built-in one is not: applied directly it
 * collapses the four request shapes into their common fields and nothing else
 * type-checks.
 */
type Ask<T> = T extends unknown ? Omit<T, "id"> : never;

/**
 * The same engines, with the work happening somewhere else.
 *
 * These implement `ChatEngine` and `Embedder` exactly, so an application swaps
 * one for the other and nothing above it changes. What it gains is an interface
 * that still responds while a model is coming up — see `./worker` for why that
 * is worth a message protocol.
 *
 * The `Worker` is passed in rather than constructed here. Every bundler has its
 * own idea of how a worker's URL should be written, and a library that guesses
 * wrong breaks the build rather than degrading; the application owns one small
 * file and is never surprised.
 */

interface Pending {
  resolve: (value: Extract<Response, { kind: "done" }>) => void;
  reject: (error: Error) => void;
  onProgress?: Progress;
  onToken?: (text: string) => void;
}

/**
 * One channel per worker, however many engines are talking through it.
 *
 * A worker has a single `onmessage` slot, so two clients sharing a thread meant
 * the second silently took the first's replies: the chat engine kept answering
 * and the embedder waited forever for messages that were being delivered
 * somewhere else. Sharing the channel — one listener, one id counter — is what
 * makes "one worker serves both" true rather than merely intended.
 */
const channels = new WeakMap<Worker, Channel>();

function channelFor(worker: Worker): Channel {
  let channel = channels.get(worker);
  if (!channel) {
    channel = new Channel(worker);
    channels.set(worker, channel);
  }
  channel.retain();
  return channel;
}

class Channel {
  private next = 1;
  private users = 0;
  private readonly pending = new Map<number, Pending>();

  constructor(private readonly worker: Worker) {
    worker.onmessage = (event: MessageEvent<Response>) => {
      const message = event.data;
      const waiting = this.pending.get(message.id);
      if (!waiting) return;

      switch (message.kind) {
        case "progress":
          waiting.onProgress?.(message.fraction, {
            loaded: message.loaded,
            total: message.total,
            fraction: message.fraction
          });
          break;
        case "token":
          waiting.onToken?.(message.text);
          break;
        case "done":
          this.pending.delete(message.id);
          waiting.resolve(message);
          break;
        case "error":
          this.pending.delete(message.id);
          waiting.reject(new Error(message.message));
          break;
      }
    };

    /*
     * A worker that dies takes every outstanding request with it.
     *
     * Without this they simply never settle, and the caller waits forever on a
     * thread that is gone — which looks exactly like a model that is very slow.
     */
    worker.onerror = (event) => {
      const error = new Error(
        (event as ErrorEvent).message || "The model worker stopped"
      );
      this.pending.forEach((waiting) => waiting.reject(error));
      this.pending.clear();
    };
  }

  retain(): void {
    this.users++;
  }

  send(
    request: Ask<Request>,
    handlers: Omit<Pending, "resolve" | "reject"> = {}
  ): Promise<Extract<Response, { kind: "done" }>> {
    const id = this.next++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, ...handlers });
      this.worker.postMessage({ ...request, id } as Request);
    });
  }

  /** Fire and forget, for anything with no answer worth waiting for. */
  tell(request: Ask<Request>): void {
    this.worker.postMessage({ ...request, id: 0 } as Request);
  }

  /**
   * Let go of it, and stop the thread when nobody is left.
   *
   * A chat window closing must not take the embedder's model down with it: the
   * launcher is still searching, and the weights are the expensive part.
   */
  close(): void {
    this.users = Math.max(0, this.users - 1);
    if (this.users > 0) return;

    this.pending.forEach((waiting) =>
      waiting.reject(new Error("The model worker was closed"))
    );
    this.pending.clear();
    channels.delete(this.worker);
    this.worker.terminate();
  }
}

class WorkerChat implements ChatEngine {
  device: Device | undefined;
  fellBackToCpu = false;
  private readonly channel: Channel;
  private loading?: Promise<void>;

  constructor(
    private readonly worker: Worker,
    readonly model: string,
    private readonly preference: DevicePreference = "auto",
    private readonly options?: ChatOptions
  ) {
    this.channel = channelFor(worker);
  }

  load(onProgress?: Progress): Promise<void> {
    if (!this.loading) {
      this.loading = this.channel
        .send(
          {
            kind: "load-chat",
            model: this.model,
            device: this.preference,
            options: this.options
          },
          { onProgress }
        )
        .then((done) => {
          this.device = done.device;
          this.fellBackToCpu = !!done.fellBack;
        });
    }
    return this.loading;
  }

  async reply(
    messages: Message[],
    onToken: (text: string) => void,
    signal?: AbortSignal
  ): Promise<string> {
    // Abandoning is a message, not a rejection: the thread keeps the model.
    signal?.addEventListener("abort", () => this.channel.tell({ kind: "abort" }), {
      once: true
    });

    const done = await this.channel.send({ kind: "generate", messages }, { onToken });
    return done.text ?? "";
  }

  dispose(): void {
    this.channel.close();
    this.loading = undefined;
    this.device = undefined;
  }
}

class WorkerEmbedder implements Embedder {
  device: Device | undefined;
  private readonly channel: Channel;
  private loading?: Promise<void>;

  constructor(
    worker: Worker,
    private readonly model?: string
  ) {
    this.channel = channelFor(worker);
  }

  load(onProgress?: Progress): Promise<void> {
    if (!this.loading) {
      this.loading = this.channel
        .send({ kind: "load-embed", model: this.model }, { onProgress })
        .then((done) => {
          this.device = done.device;
        });
    }
    return this.loading;
  }

  async embed(texts: string[]): Promise<Float32Array[]> {
    if (!texts.length) return [];
    const done = await this.channel.send({ kind: "embed", texts });
    return (done.vectors ?? []).map((row) => Float32Array.from(row));
  }

  dispose(): void {
    this.channel.close();
    this.loading = undefined;
    this.device = undefined;
  }
}

/**
 * A chat engine that runs somewhere else.
 *
 * The application makes the worker, because only it knows how its bundler
 * writes one:
 *
 *   // model.worker.ts — the whole file
 *   import "@thatcatdev/browser-ai/worker";
 *
 *   const worker = new Worker(new URL("./model.worker.ts", import.meta.url), {
 *     type: "module"
 *   });
 *   const engine = workerChat(worker, CHAT_MODEL, "auto");
 */
export function workerChat(
  worker: Worker,
  model: string,
  preference: DevicePreference = "auto",
  options?: ChatOptions
): ChatEngine {
  return new WorkerChat(worker, model, preference, options);
}

/** An embedder that runs somewhere else. One worker can serve both. */
export function workerEmbedder(worker: Worker, model?: string): Embedder {
  return new WorkerEmbedder(worker, model);
}
