import {
  Downloads,
  resolveDevice,
  transformers
} from "./chunk-B5A4XJ7T.js";

// src/worker/worker.ts
var reply = (message) => self.postMessage(message);
var chat = {};
var embed;
var abandoned = false;
async function loadChat(request) {
  const downloads = new Downloads();
  const { pipeline } = await transformers();
  const { device, fellBack } = await resolveDevice(request.device);
  const build = (on) => pipeline("text-generation", request.model, {
    device: on,
    dtype: "q4",
    progress_callback: (event) => {
      if (event.status === "progress" && event.file) {
        downloads.record(event.file, event.loaded ?? 0, event.total ?? 0);
        reply({ id: request.id, kind: "progress", fraction: downloads.fraction() });
      }
      if (event.status === "ready") {
        reply({ id: request.id, kind: "progress", fraction: 1 });
      }
    }
  });
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
    tokenizer: pipe.tokenizer
  };
  reply({ id: request.id, kind: "done", device: ran, fellBack: fell });
}
async function generate(request, options) {
  if (!chat.generate) throw new Error("Chat used before it was loaded");
  const { TextStreamer } = await transformers();
  abandoned = false;
  let answer = "";
  const streamer = new TextStreamer(chat.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (text) => {
      if (abandoned) return;
      answer += text;
      reply({ id: request.id, kind: "token", text });
    }
  });
  const generateFn = chat.generate;
  const output = await generateFn(request.messages, { ...options, streamer });
  if (!answer) {
    const generated = output?.[0]?.generated_text;
    if (Array.isArray(generated)) {
      const last = generated[generated.length - 1];
      answer = last?.content ?? "";
    } else if (typeof generated === "string") {
      answer = generated;
    }
  }
  reply({ id: request.id, kind: "done", text: answer.trim() });
}
async function loadEmbed(request) {
  const downloads = new Downloads();
  const { pipeline } = await transformers();
  const { device } = await resolveDevice("auto");
  embed = await pipeline(
    "feature-extraction",
    request.model ?? "Xenova/all-MiniLM-L6-v2",
    {
      device,
      dtype: "q8",
      progress_callback: (event) => {
        if (event.status === "progress" && event.file) {
          downloads.record(event.file, event.loaded ?? 0, event.total ?? 0);
          reply({ id: request.id, kind: "progress", fraction: downloads.fraction() });
        }
        if (event.status === "ready") {
          reply({ id: request.id, kind: "progress", fraction: 1 });
        }
      }
    }
  );
  reply({ id: request.id, kind: "done", device });
}
async function runEmbed(request) {
  if (!embed) throw new Error("Embedder used before it was loaded");
  const extractor = embed;
  const output = await extractor(request.texts, { pooling: "mean", normalize: true });
  reply({ id: request.id, kind: "done", vectors: output.tolist() });
}
var SAMPLING = {
  max_new_tokens: 160,
  temperature: 0.3,
  top_k: 40,
  top_p: 0.9,
  do_sample: true,
  repetition_penalty: 1.1,
  return_full_text: false
};
var sampling = SAMPLING;
self.onmessage = async (event) => {
  const request = event.data;
  if (request.kind === "abort") {
    abandoned = true;
    return;
  }
  try {
    switch (request.kind) {
      case "load-chat":
        sampling = {
          ...SAMPLING,
          ...request.options?.maxTokens ? { max_new_tokens: request.options.maxTokens } : {},
          ...request.options?.temperature !== void 0 ? { temperature: request.options.temperature } : {},
          ...request.options?.topK !== void 0 ? { top_k: request.options.topK } : {},
          ...request.options?.topP !== void 0 ? { top_p: request.options.topP } : {},
          ...request.options?.repetitionPenalty !== void 0 ? { repetition_penalty: request.options.repetitionPenalty } : {}
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
//# sourceMappingURL=worker.js.map