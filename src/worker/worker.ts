/**
 * The models, on a thread that is not the one drawing the page.
 *
 * This is not an optimisation. Bringing a 0.5B model up costs about five
 * seconds of solid arithmetic, and on the main thread that is five seconds in
 * which nothing responds: no menu opens, no window drags, no key registers. The
 * failure that produces is worse than the delay — the interface appears dead,
 * so the person presses again, and both presses arrive at once when it thaws.
 * Every generation after that janks the page for its duration too.
 *
 * Importing this module *is* starting the worker: it attaches the handler and
 * waits. Applications reach it through `workerChat` and `workerEmbedder` in
 * `./client`, which speak the protocol next door and present the same
 * interfaces as the main-thread implementations.
 */
import { transformers, resolveDevice, Downloads, type Device } from "../device";
import type { Request, Response } from "./protocol";

const reply = (message: Response) => (self as unknown as Worker).postMessage(message);

/** What the pipelines are, once they exist. */
let chat: { generate?: unknown; tokenizer?: unknown } = {};
let embed: unknown;
/** The generation in flight, so it can be abandoned when a window closes. */
let abandoned = false;

async function loadChat(request: Extract<Request, { kind: "load-chat" }>) {
  const downloads = new Downloads();
  const { pipeline } = await transformers();
  const { device, fellBack } = await resolveDevice(request.device);

  const build = (on: Device) =>
    pipeline("text-generation", request.model, {
      device: on,
      dtype: "q4",
      progress_callback: (event: {
        status: string;
        file?: string;
        loaded?: number;
        total?: number;
      }) => {
        if (event.status === "progress" && event.file) {
          downloads.record(event.file, event.loaded ?? 0, event.total ?? 0);
          reply({ id: request.id, kind: "progress", ...downloads.state() });
        }
        if (event.status === "ready") {
          reply({ id: request.id, kind: "progress", ...downloads.finished() });
        }
      }
    });

  /*
   * The same retry the main-thread engine does, for the same reason: an adapter
   * that exists is not a promise that a model will run on it, and an integrated
   * GPU sharing system memory is where a large one runs out of room.
   */
  let pipe;
  let ran = device;
  let fell = fellBack;
  try {
    pipe = await build(device);
  } catch (error) {
    if (device !== "webgpu") throw error;
    ran = "wasm";
    fell = true;
    pipe = await build("wasm");
  }

  chat = {
    generate: pipe,
    tokenizer: (pipe as unknown as { tokenizer: unknown }).tokenizer
  };
  reply({ id: request.id, kind: "done", device: ran, fellBack: fell });
}

async function generate(
  request: Extract<Request, { kind: "generate" }>,
  options: Record<string, unknown>
) {
  if (!chat.generate) throw new Error("Chat used before it was loaded");
  const { TextStreamer } = await transformers();

  abandoned = false;
  let answer = "";

  const streamer = new TextStreamer(chat.tokenizer as never, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (text: string) => {
      if (abandoned) return;
      answer += text;
      reply({ id: request.id, kind: "token", text });
    }
  });

  const generateFn = chat.generate as (
    input: unknown,
    options: Record<string, unknown>
  ) => Promise<Array<{ generated_text: unknown }>>;

  const output = await generateFn(request.messages, { ...options, streamer });

  if (!answer) {
    const generated = output?.[0]?.generated_text;
    if (Array.isArray(generated)) {
      const last = generated[generated.length - 1] as { content?: string };
      answer = last?.content ?? "";
    } else if (typeof generated === "string") {
      answer = generated;
    }
  }

  reply({ id: request.id, kind: "done", text: answer.trim() });
}

async function loadEmbed(request: Extract<Request, { kind: "load-embed" }>) {
  const downloads = new Downloads();
  const { pipeline } = await transformers();
  const { device } = await resolveDevice("auto");

  embed = await pipeline(
    "feature-extraction",
    request.model ?? "Xenova/all-MiniLM-L6-v2",
    {
      device,
      dtype: "q8",
      progress_callback: (event: {
        status: string;
        file?: string;
        loaded?: number;
        total?: number;
      }) => {
        if (event.status === "progress" && event.file) {
          downloads.record(event.file, event.loaded ?? 0, event.total ?? 0);
          reply({ id: request.id, kind: "progress", ...downloads.state() });
        }
        if (event.status === "ready") {
          reply({ id: request.id, kind: "progress", ...downloads.finished() });
        }
      }
    }
  );

  reply({ id: request.id, kind: "done", device });
}

async function runEmbed(request: Extract<Request, { kind: "embed" }>) {
  if (!embed) throw new Error("Embedder used before it was loaded");
  const extractor = embed as (
    texts: string[],
    options: { pooling: "mean"; normalize: boolean }
  ) => Promise<{ tolist(): number[][] }>;

  const output = await extractor(request.texts, { pooling: "mean", normalize: true });
  reply({ id: request.id, kind: "done", vectors: output.tolist() });
}

/** Sampling lives here so the numbers travel with the model, not the message. */
const SAMPLING = {
  max_new_tokens: 160,
  temperature: 0.3,
  top_k: 40,
  top_p: 0.9,
  do_sample: true,
  repetition_penalty: 1.1,
  return_full_text: false
};

let sampling: Record<string, unknown> = SAMPLING;

self.onmessage = async (event: MessageEvent<Request>) => {
  const request = event.data;

  if (request.kind === "abort") {
    // The tokens already in flight stop being anybody's business. The model is
    // left loaded: the next question should not pay for it again.
    abandoned = true;
    return;
  }

  try {
    switch (request.kind) {
      case "load-chat":
        sampling = {
          ...SAMPLING,
          ...(request.options?.maxTokens ? { max_new_tokens: request.options.maxTokens } : {}),
          ...(request.options?.temperature !== undefined
            ? { temperature: request.options.temperature }
            : {}),
          ...(request.options?.topK !== undefined ? { top_k: request.options.topK } : {}),
          ...(request.options?.topP !== undefined ? { top_p: request.options.topP } : {}),
          ...(request.options?.repetitionPenalty !== undefined
            ? { repetition_penalty: request.options.repetitionPenalty }
            : {})
        };
        await loadChat(request);
        break;
      case "generate":
        await generate(request, sampling);
        break;
      case "load-embed":
        await loadEmbed(request);
        break;
      case "embed":
        await runEmbed(request);
        break;
    }
  } catch (error) {
    reply({
      id: request.id,
      kind: "error",
      message: error instanceof Error ? error.message : String(error)
    });
  }
};
