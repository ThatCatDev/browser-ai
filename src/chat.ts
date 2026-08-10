import {
  Downloads,
  resolveDevice,
  transformers,
  type Device,
  type DevicePreference,
  type Progress
} from "./device";

/**
 * A conversation with a model on the machine that is reading the page.
 *
 * Nothing leaves the browser: no key, no endpoint, nobody's training data. The
 * price is the download, and a hard ceiling on how good the answers can be.
 *
 * Be clear about that ceiling, because it decides how these are worth using.
 * At 135M a model writes fluent English and cannot hold a conversation — asked
 * what it was doing it will write a scene, invent a colleague and give her
 * dialogue. Half a billion parameters is where it starts answering the question
 * it was asked. Even then it is a paraphraser rather than an authority: hand it
 * text and it will summarise it well, ask it to recall and it will invent.
 *
 * Which is why the sampling here is cooler and narrower than a chat app's
 * usual. A small model given room does not get more creative; it gets further
 * from the question.
 */

export interface ChatModel {
  id: string;
  label: string;
  /** Roughly what it costs to fetch, in the words somebody would use. */
  size: string;
  /**
   * The same figure in bytes, near enough to draw a bar against.
   *
   * Needed because the runtime only ever reports the files it has met, and the
   * weights are announced last — without something to measure against, a bar
   * reads 98% while the actual model has not started arriving. Approximate on
   * purpose: it is a denominator until the real total is known, not a promise.
   */
  bytes: number;
  /** What it is like to talk to. No marketing. */
  note: string;
}

/**
 * Three rungs, smallest first.
 *
 * Nothing past a billion: a 2B model is another gigabyte and a half, does not
 * fit in the WASM address space at all, and would mean offering a window that
 * cannot work to every visitor without WebGPU.
 */
/*
 * The sizes are what these actually weigh at `q4` on the hub, measured rather
 * than estimated from the parameter count — which is what they were, and which
 * was wrong by a factor of two in both directions. Four-bit quantisation
 * applies to the weight matrices and not to the embedding table, so a model
 * with a 150,000-token vocabulary carries a great deal of full-precision
 * baggage: Qwen at half a billion parameters is a larger download than
 * SmolLM2's file size would lead anybody to guess.
 *
 * They matter more than a label. This is the number somebody decides on, and
 * it is the denominator a progress bar is drawn against.
 */
export const CHAT_MODELS: ChatModel[] = [
  {
    id: "HuggingFaceTB/SmolLM2-135M-Instruct",
    label: "SmolLM2 135M",
    size: "~185MB",
    bytes: 185_000_000,
    note: "Fastest to arrive. Writes fluently and wanders off the question."
  },
  {
    id: "onnx-community/Qwen2.5-0.5B-Instruct",
    label: "Qwen2.5 0.5B",
    size: "~800MB",
    bytes: 800_000_000,
    note: "The default. Answers what was asked, briefly."
  },
  {
    id: "onnx-community/Llama-3.2-1B-Instruct-ONNX",
    label: "Llama 3.2 1B",
    size: "~1.7GB",
    bytes: 1_700_000_000,
    note: "Conversational. Worth it on a GPU, painful without one."
  }
];

/** The first rung that answers the question it was asked. */
export const CHAT_MODEL = CHAT_MODELS[1].id;

export interface Message {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface ChatOptions {
  /** A short paragraph. Generous ceilings invite drift rather than detail. */
  maxTokens?: number;
  temperature?: number;
  topK?: number;
  topP?: number;
  repetitionPenalty?: number;
}

export interface ChatEngine {
  readonly device: Device | undefined;
  readonly model: string;
  /** True when the GPU was asked for and this browser could not give it. */
  readonly fellBackToCpu: boolean;
  load(onProgress?: Progress): Promise<void>;
  /**
   * Answer, a token at a time.
   *
   * Streamed because on a CPU this writes at about reading speed: waiting for
   * the last word before showing the first turns a working answer into a
   * frozen window.
   */
  reply(
    messages: Message[],
    onToken: (text: string) => void,
    signal?: AbortSignal
  ): Promise<string>;
  dispose(): void;
}

const SAMPLING: Required<ChatOptions> = {
  maxTokens: 160,
  temperature: 0.3,
  topK: 40,
  topP: 0.9,
  repetitionPenalty: 1.1
};

class TransformersChat implements ChatEngine {
  device: Device | undefined;
  fellBackToCpu = false;
  private generate?: (
    input: unknown,
    options: Record<string, unknown>
  ) => Promise<Array<{ generated_text: unknown }>>;
  private tokenizer?: unknown;
  private loading?: Promise<void>;

  constructor(
    readonly model: string = CHAT_MODEL,
    private readonly preference: DevicePreference = "auto",
    private readonly sampling: Required<ChatOptions> = SAMPLING
  ) {}

  matches(preference: DevicePreference): boolean {
    return this.preference === preference;
  }

  load(onProgress?: Progress): Promise<void> {
    if (!this.loading) this.loading = this.bring(onProgress);
    return this.loading;
  }

  private async bring(onProgress?: Progress): Promise<void> {
    const downloads = new Downloads();
    const { pipeline } = await transformers();
    const { device, fellBack } = await resolveDevice(this.preference);
    this.fellBackToCpu = fellBack;

    const build = (on: Device) =>
      pipeline("text-generation", this.model, {
        device: on,
        /*
         * Four bits, where the embedder takes eight. The trade runs the other
         * way for generation: the file is what somebody waits for, and slightly
         * more repetition costs less than twice the wait.
         */
        dtype: "q4",
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

    /*
     * One retry, on the CPU.
     *
     * An adapter that exists is not a promise that a model will run on it:
     * shaders fail to compile, buffers are refused, and an integrated GPU
     * sharing system memory is where a large model runs out of room. The honest
     * answer to all of those is the slower path, not a dead window.
     */
    let pipe;
    let ran = device;
    try {
      pipe = await build(device);
    } catch (error) {
      if (device !== "webgpu") throw error;
      ran = "wasm";
      this.fellBackToCpu = true;
      pipe = await build("wasm");
    }

    this.device = ran;
    this.tokenizer = (pipe as unknown as { tokenizer: unknown }).tokenizer;
    this.generate = pipe as unknown as typeof this.generate;
  }

  async reply(
    messages: Message[],
    onToken: (text: string) => void,
    signal?: AbortSignal
  ): Promise<string> {
    if (!this.generate) throw new Error("Chat used before it was loaded");

    const { TextStreamer } = await transformers();
    let answer = "";

    const streamer = new TextStreamer(this.tokenizer as never, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (text: string) => {
        if (signal?.aborted) return;
        answer += text;
        onToken(text);
      }
    });

    const output = await this.generate(messages, {
      max_new_tokens: this.sampling.maxTokens,
      temperature: this.sampling.temperature,
      top_k: this.sampling.topK,
      top_p: this.sampling.topP,
      do_sample: true,
      repetition_penalty: this.sampling.repetitionPenalty,
      return_full_text: false,
      streamer
    });

    if (answer) return answer.trim();

    // An implementation without a streamer, or one that finished in a single
    // chunk. The reply is in the output either way.
    const generated = output?.[0]?.generated_text;
    if (Array.isArray(generated)) {
      const last = generated[generated.length - 1] as { content?: string };
      return (last?.content ?? "").trim();
    }
    return typeof generated === "string" ? generated.trim() : "";
  }

  dispose(): void {
    this.generate = undefined;
    this.tokenizer = undefined;
    this.loading = undefined;
    this.device = undefined;
    this.fellBackToCpu = false;
  }
}

let shared: ChatEngine | undefined;

/**
 * The chat model for this page, for whatever has been chosen.
 *
 * One at a time: two of these is a gigabyte of the same idea held in memory, so
 * changing the model or the device lets go of the last one first. The weights
 * stay in the browser's cache, so going back to one already used is a load
 * rather than a download.
 */
export function chatEngine(
  model: string = CHAT_MODEL,
  preference: DevicePreference = "auto",
  options?: ChatOptions
): ChatEngine {
  const current = shared as TransformersChat | undefined;
  if (current?.model === model && current.matches(preference) && !options) {
    return current;
  }

  shared?.dispose();
  shared = new TransformersChat(model, preference, { ...SAMPLING, ...options });
  return shared;
}

/** Swap the implementation — for tests, and for whatever replaces this one. */
export function setChatEngine(replacement: ChatEngine | undefined): void {
  shared = replacement;
}
