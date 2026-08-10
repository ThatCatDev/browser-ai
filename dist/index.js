import {
  Downloads,
  bestDevice,
  forgetDevice,
  gpuAvailable,
  resolveDevice,
  transformers
} from "./chunk-B5A4XJ7T.js";

// src/embed.ts
var EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";
var TransformersEmbedder = class {
  constructor(model = EMBEDDING_MODEL) {
    this.model = model;
  }
  load(onProgress) {
    if (!this.loading) this.loading = this.bring(onProgress);
    return this.loading;
  }
  async bring(onProgress) {
    const downloads = new Downloads();
    const { pipeline } = await transformers();
    const { device } = await resolveDevice("auto");
    const extractor = await pipeline("feature-extraction", this.model, {
      device,
      dtype: "q8",
      progress_callback: (event) => {
        if (!onProgress) return;
        if (event.status === "progress" && event.file) {
          downloads.record(event.file, event.loaded ?? 0, event.total ?? 0);
          onProgress(downloads.fraction());
        }
        if (event.status === "ready") onProgress(1);
      }
    });
    this.device = device;
    this.pipeline = extractor;
  }
  async embed(texts) {
    if (!texts.length) return [];
    if (!this.pipeline) throw new Error("Embedder used before it was loaded");
    const output = await this.pipeline(texts, { pooling: "mean", normalize: true });
    return output.tolist().map((row) => Float32Array.from(row));
  }
  dispose() {
    this.pipeline = void 0;
    this.loading = void 0;
    this.device = void 0;
  }
};
var shared;
function embedder() {
  if (!shared) shared = new TransformersEmbedder();
  return shared;
}
function setEmbedder(replacement) {
  shared = replacement;
}
function similarity(a, b) {
  const length = Math.min(a.length, b.length);
  let total = 0;
  for (let i = 0; i < length; i++) total += a[i] * b[i];
  return total;
}

// src/chat.ts
var CHAT_MODELS = [
  {
    id: "HuggingFaceTB/SmolLM2-135M-Instruct",
    label: "SmolLM2 135M",
    size: "~100MB",
    note: "Fastest to arrive. Writes fluently and wanders off the question."
  },
  {
    id: "onnx-community/Qwen2.5-0.5B-Instruct",
    label: "Qwen2.5 0.5B",
    size: "~350MB",
    note: "The default. Answers what was asked, briefly."
  },
  {
    id: "onnx-community/Llama-3.2-1B-Instruct-ONNX",
    label: "Llama 3.2 1B",
    size: "~900MB",
    note: "Conversational. Worth it on a GPU, painful without one."
  }
];
var CHAT_MODEL = CHAT_MODELS[1].id;
var SAMPLING = {
  maxTokens: 160,
  temperature: 0.3,
  topK: 40,
  topP: 0.9,
  repetitionPenalty: 1.1
};
var TransformersChat = class {
  constructor(model = CHAT_MODEL, preference = "auto", sampling = SAMPLING) {
    this.model = model;
    this.preference = preference;
    this.sampling = sampling;
    this.fellBackToCpu = false;
  }
  matches(preference) {
    return this.preference === preference;
  }
  load(onProgress) {
    if (!this.loading) this.loading = this.bring(onProgress);
    return this.loading;
  }
  async bring(onProgress) {
    const downloads = new Downloads();
    const { pipeline } = await transformers();
    const { device, fellBack } = await resolveDevice(this.preference);
    this.fellBackToCpu = fellBack;
    const build = (on) => pipeline("text-generation", this.model, {
      device: on,
      /*
       * Four bits, where the embedder takes eight. The trade runs the other
       * way for generation: the file is what somebody waits for, and slightly
       * more repetition costs less than twice the wait.
       */
      dtype: "q4",
      progress_callback: (event) => {
        if (!onProgress) return;
        if (event.status === "progress" && event.file) {
          downloads.record(event.file, event.loaded ?? 0, event.total ?? 0);
          onProgress(downloads.fraction());
        }
        if (event.status === "ready") onProgress(1);
      }
    });
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
    this.tokenizer = pipe.tokenizer;
    this.generate = pipe;
  }
  async reply(messages, onToken, signal) {
    if (!this.generate) throw new Error("Chat used before it was loaded");
    const { TextStreamer } = await transformers();
    let answer = "";
    const streamer = new TextStreamer(this.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (text) => {
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
    const generated = output?.[0]?.generated_text;
    if (Array.isArray(generated)) {
      const last = generated[generated.length - 1];
      return (last?.content ?? "").trim();
    }
    return typeof generated === "string" ? generated.trim() : "";
  }
  dispose() {
    this.generate = void 0;
    this.tokenizer = void 0;
    this.loading = void 0;
    this.device = void 0;
    this.fellBackToCpu = false;
  }
};
var shared2;
function chatEngine(model = CHAT_MODEL, preference = "auto", options) {
  const current = shared2;
  if (current?.model === model && current.matches(preference) && !options) {
    return current;
  }
  shared2?.dispose();
  shared2 = new TransformersChat(model, preference, { ...SAMPLING, ...options });
  return shared2;
}
function setChatEngine(replacement) {
  shared2 = replacement;
}

// src/retrieval.ts
var DEFAULTS = {
  floor: 0.3,
  lexicalBoost: 0.25,
  cacheKey: "browser-ai:vectors"
};
var VectorIndex = class {
  constructor(options = {}) {
    this.vectors = /* @__PURE__ */ new Map();
    this.documents = [];
    this.fingerprint = "";
    /** Query vectors, because people ask the same thing twice. */
    this.queries = /* @__PURE__ */ new Map();
    this.model = options.model ?? embedder();
    this.store = options.store;
    this.cacheKey = options.cacheKey ?? DEFAULTS.cacheKey;
    this.floor = options.floor ?? DEFAULTS.floor;
    this.lexicalBoost = options.lexicalBoost ?? DEFAULTS.lexicalBoost;
    this.stopWords = new Set(options.stopWords ?? []);
  }
  get ready() {
    return this.vectors.size > 0;
  }
  /**
   * Learn these documents, or notice they are already learned.
   *
   * Safe to call on every open: identical input returns immediately, and two
   * callers arriving together share the work. Failure leaves the index empty
   * rather than half-built, so `search` answers nothing and whatever this was
   * improving carries on without it.
   */
  build(documents, onProgress) {
    const fingerprint = documents.map((doc) => `${doc.id}:${doc.text.length}`).join("|");
    if (this.fingerprint === fingerprint && this.vectors.size) return Promise.resolve();
    if (this.building && this.fingerprint === fingerprint) return this.building;
    this.fingerprint = fingerprint;
    this.documents = documents;
    this.building = this.fill(fingerprint, onProgress).catch(() => {
      this.vectors.clear();
      this.building = void 0;
    });
    return this.building;
  }
  async fill(fingerprint, onProgress) {
    await this.model.load(onProgress);
    const stored = await this.store?.read(this.cacheKey);
    if (stored?.fingerprint === fingerprint) {
      Object.entries(stored.vectors).forEach(
        ([id, vector]) => this.vectors.set(id, Float32Array.from(vector))
      );
      return;
    }
    const vectors = await this.model.embed(this.documents.map((doc) => doc.text));
    this.documents.forEach((doc, i) => this.vectors.set(doc.id, vectors[i]));
    await this.store?.write(this.cacheKey, {
      fingerprint,
      vectors: Object.fromEntries(
        this.documents.map((doc, i) => [doc.id, Array.from(vectors[i])])
      )
    });
  }
  /**
   * What this query is about, best first.
   *
   * Nothing at all until the index is built, and nothing for a query too short
   * to mean anything — two letters are a prefix, not a subject.
   */
  async search(query, limit2 = 4) {
    const q = query.trim();
    if (!this.ready || q.length < 3) return [];
    let vector = this.queries.get(q);
    if (!vector) {
      try {
        [vector] = await this.model.embed([q]);
      } catch {
        return [];
      }
      this.queries.set(q, vector);
    }
    const terms = this.lexicalBoost ? q.toLowerCase().split(/[^a-z0-9.+#-]+/).filter((term) => term.length >= 4 && !this.stopWords.has(term)) : [];
    return this.documents.map((document) => {
      const lower = document.text.toLowerCase();
      const hits = terms.filter((term) => lower.includes(term)).length;
      const lexical = terms.length ? hits / terms.length * this.lexicalBoost : 0;
      return {
        document,
        score: similarity(vector, this.vectors.get(document.id) ?? new Float32Array()) + lexical
      };
    }).filter((match) => match.score >= this.floor).sort((a, b) => b.score - a.score).slice(0, limit2);
  }
};
function contextual(question, previous, words = 6) {
  const q = question.trim();
  if (!previous) return q;
  return q.split(/\s+/).length <= words ? `${previous.trim()} ${q}` : q;
}

// src/prompt.ts
function ground(question, found, options = {}) {
  const heading = options.heading ?? "Notes:";
  const instruction = options.instruction ?? "Using the notes above, answer briefly:";
  if (!found.length) {
    if (options.whenEmpty !== "say-unknown") return question;
    return [
      "There are no notes about this.",
      "",
      `Say plainly that you do not know, in one sentence. Question: ${question}`
    ].join("\n");
  }
  return [
    heading,
    ...found.map((document) => `- ${document.text}`),
    "",
    `${instruction} ${question}`
  ].join("\n");
}

// src/limits.ts
var LOOKUP = /\b(search|google|look (it |this |that )?up|browse|internet access|the web|online)\b/i;
var PRESENT = /\b(today|tonight|right now|currently|this (week|month|year)|latest|newest|recent|news|weather|price of|stock|airing|out now|release[ds]? (this|last)?)\b/i;
var CLOCK = /\b(what (day|date|time)|todays? date|what year)\b/i;
var SOURCES = /\b(based on|where (did|do) (you|that) (get|come)|what.{0,12}(source|sources)|how do you know|says who)\b/i;
function limit(question) {
  const q = question.trim();
  if (!q) return void 0;
  if (SOURCES.test(q)) return "asks-for-sources";
  if (CLOCK.test(q)) return "no-clock";
  if (LOOKUP.test(q)) return "no-internet";
  if (PRESENT.test(q)) return "no-present";
  return void 0;
}

// src/worker/client.ts
var channels = /* @__PURE__ */ new WeakMap();
function channelFor(worker) {
  let channel = channels.get(worker);
  if (!channel) {
    channel = new Channel(worker);
    channels.set(worker, channel);
  }
  channel.retain();
  return channel;
}
var Channel = class {
  constructor(worker) {
    this.worker = worker;
    this.next = 1;
    this.users = 0;
    this.pending = /* @__PURE__ */ new Map();
    worker.onmessage = (event) => {
      const message = event.data;
      const waiting = this.pending.get(message.id);
      if (!waiting) return;
      switch (message.kind) {
        case "progress":
          waiting.onProgress?.(message.fraction);
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
    worker.onerror = (event) => {
      const error = new Error(
        event.message || "The model worker stopped"
      );
      this.pending.forEach((waiting) => waiting.reject(error));
      this.pending.clear();
    };
  }
  retain() {
    this.users++;
  }
  send(request, handlers = {}) {
    const id = this.next++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, ...handlers });
      this.worker.postMessage({ ...request, id });
    });
  }
  /** Fire and forget, for anything with no answer worth waiting for. */
  tell(request) {
    this.worker.postMessage({ ...request, id: 0 });
  }
  /**
   * Let go of it, and stop the thread when nobody is left.
   *
   * A chat window closing must not take the embedder's model down with it: the
   * launcher is still searching, and the weights are the expensive part.
   */
  close() {
    this.users = Math.max(0, this.users - 1);
    if (this.users > 0) return;
    this.pending.forEach(
      (waiting) => waiting.reject(new Error("The model worker was closed"))
    );
    this.pending.clear();
    channels.delete(this.worker);
    this.worker.terminate();
  }
};
var WorkerChat = class {
  constructor(worker, model, preference = "auto", options) {
    this.worker = worker;
    this.model = model;
    this.preference = preference;
    this.options = options;
    this.fellBackToCpu = false;
    this.channel = channelFor(worker);
  }
  load(onProgress) {
    if (!this.loading) {
      this.loading = this.channel.send(
        {
          kind: "load-chat",
          model: this.model,
          device: this.preference,
          options: this.options
        },
        { onProgress }
      ).then((done) => {
        this.device = done.device;
        this.fellBackToCpu = !!done.fellBack;
      });
    }
    return this.loading;
  }
  async reply(messages, onToken, signal) {
    signal?.addEventListener("abort", () => this.channel.tell({ kind: "abort" }), {
      once: true
    });
    const done = await this.channel.send({ kind: "generate", messages }, { onToken });
    return done.text ?? "";
  }
  dispose() {
    this.channel.close();
    this.loading = void 0;
    this.device = void 0;
  }
};
var WorkerEmbedder = class {
  constructor(worker, model) {
    this.model = model;
    this.channel = channelFor(worker);
  }
  load(onProgress) {
    if (!this.loading) {
      this.loading = this.channel.send({ kind: "load-embed", model: this.model }, { onProgress }).then((done) => {
        this.device = done.device;
      });
    }
    return this.loading;
  }
  async embed(texts) {
    if (!texts.length) return [];
    const done = await this.channel.send({ kind: "embed", texts });
    return (done.vectors ?? []).map((row) => Float32Array.from(row));
  }
  dispose() {
    this.channel.close();
    this.loading = void 0;
    this.device = void 0;
  }
};
function workerChat(worker, model, preference = "auto", options) {
  return new WorkerChat(worker, model, preference, options);
}
function workerEmbedder(worker, model) {
  return new WorkerEmbedder(worker, model);
}
export {
  CHAT_MODEL,
  CHAT_MODELS,
  EMBEDDING_MODEL,
  VectorIndex,
  bestDevice,
  chatEngine,
  contextual,
  embedder,
  forgetDevice,
  gpuAvailable,
  ground,
  limit,
  resolveDevice,
  setChatEngine,
  setEmbedder,
  similarity,
  workerChat,
  workerEmbedder
};
//# sourceMappingURL=index.js.map