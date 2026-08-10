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
type Device = "webgpu" | "wasm";
/** What was asked for, which is not always what the machine can give. */
type DevicePreference = "auto" | Device;
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
interface Loading {
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
type Progress = (fraction: number, detail: Loading) => void;
declare function gpuAvailable(): Promise<boolean>;
/** Forget the cached answer. For tests, and for anything that reloads a page. */
declare function forgetDevice(): void;
/**
 * What this browser claims, cheaply and synchronously.
 *
 * Only ever a claim — anything about to load a model should wait for
 * `resolveDevice`. Useful for drawing a control before the driver has answered.
 */
declare function bestDevice(): Device;
/**
 * The device to use, and whether the answer disappointed anybody.
 *
 * Falling back is right; falling back silently is not. Somebody who asked for
 * the GPU and got the CPU should be able to be told why it is slow.
 */
declare function resolveDevice(preference: DevicePreference): Promise<{
    device: Device;
    fellBack: boolean;
}>;

/**
 * Turning text into vectors, which is most of what retrieval is.
 *
 * MiniLM at eight bits is about 23MB — small enough that a first search is not
 * an event, and the reason retrieval is worth doing in a browser at all. It is
 * eight bits rather than four because the accuracy of the ranking is the entire
 * point: a search that confidently returns the wrong thing is not a feature.
 */
declare const EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";
interface Embedder {
    /** The device it settled on, once loaded. */
    readonly device: Device | undefined;
    /** Bring the model in. Safe to call twice; the second call waits on the first. */
    load(onProgress?: Progress): Promise<void>;
    /** One vector per string, normalised, so similarity is a dot product. */
    embed(texts: string[]): Promise<Float32Array[]>;
    dispose(): void;
}
/** The embedder for this page. One per page; the weights are not cheap. */
declare function embedder(): Embedder;
/** Swap the implementation — for tests, and for whatever replaces this one. */
declare function setEmbedder(replacement: Embedder | undefined): void;
/**
 * How alike two vectors are, from -1 to 1.
 *
 * A plain dot product, because everything here is normalised on the way out of
 * the model — dividing by two lengths that are both 1 is work for nothing.
 */
declare function similarity(a: Float32Array, b: Float32Array): number;

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
interface ChatModel {
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
declare const CHAT_MODELS: ChatModel[];
/** The first rung that answers the question it was asked. */
declare const CHAT_MODEL: string;
interface Message {
    role: "system" | "user" | "assistant";
    content: string;
}
interface ChatOptions {
    /** A short paragraph. Generous ceilings invite drift rather than detail. */
    maxTokens?: number;
    temperature?: number;
    topK?: number;
    topP?: number;
    repetitionPenalty?: number;
}
interface ChatEngine {
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
    reply(messages: Message[], onToken: (text: string) => void, signal?: AbortSignal): Promise<string>;
    dispose(): void;
}
/**
 * The chat model for this page, for whatever has been chosen.
 *
 * One at a time: two of these is a gigabyte of the same idea held in memory, so
 * changing the model or the device lets go of the last one first. The weights
 * stay in the browser's cache, so going back to one already used is a load
 * rather than a download.
 */
declare function chatEngine(model?: string, preference?: DevicePreference, options?: ChatOptions): ChatEngine;
/** Swap the implementation — for tests, and for whatever replaces this one. */
declare function setChatEngine(replacement: ChatEngine | undefined): void;

/**
 * Retrieval, for corpora small enough that a `for` loop is the right index.
 *
 * A few thousand vectors is a few milliseconds of arithmetic per query, so
 * there is no approximate index here and no database: an ANN structure would
 * cost more to build than the search it saves, and a vector store would be a
 * dependency in exchange for nothing. Past roughly ten thousand passages this
 * is the wrong tool and sqlite-vec or pgvector in WASM is the right one.
 *
 * What it does do is the part that actually decides whether retrieval is any
 * good: chunking is the caller's business, but scoring blends meaning with a
 * nudge for the words themselves — because sentence embeddings are weak on
 * proper nouns, and a question naming a company or a library should find the
 * passage that names it back.
 */
interface Document {
    id: string;
    /** Where it came from, for an answer that has to cite itself. */
    source: string;
    text: string;
}
interface Match {
    document: Document;
    score: number;
}
/**
 * Somewhere to keep vectors between visits.
 *
 * An interface rather than an implementation: a library has no business
 * choosing IndexedDB, a wrapper around it, or a server. Anything that can hold
 * a JSON-shaped value against a key will do.
 */
interface VectorStore {
    read(key: string): Promise<unknown | undefined>;
    write(key: string, value: unknown): Promise<void>;
}
interface IndexOptions {
    model?: Embedder;
    /** Where to keep the vectors. Without one, every visit re-embeds. */
    store?: VectorStore;
    /** The key those vectors are kept under. */
    cacheKey?: string;
    /**
     * How alike is alike enough.
     *
     * Cosine similarity always has a best answer, however wrong: without a floor,
     * "asdfgh" returns three passages with great confidence. Lower it for a
     * search box, where a weak match costs a glance; raise it for a prompt, where
     * a weak match becomes something a model treats as true.
     */
    floor?: number;
    /**
     * How much the query's own uncommon words are worth when they appear.
     *
     * Zero is pure semantics. A little — a fifth or so — is what makes questions
     * naming a rare token find the passage naming it back, without turning the
     * ranking into keyword search.
     */
    lexicalBoost?: number;
    /** Words too ordinary to be evidence. Only matters if `lexicalBoost` is set. */
    stopWords?: Iterable<string>;
}
declare class VectorIndex {
    private readonly model;
    private readonly store?;
    private readonly cacheKey;
    private readonly floor;
    private readonly lexicalBoost;
    private readonly stopWords;
    private readonly vectors;
    private documents;
    private building?;
    private fingerprint;
    /** Query vectors, because people ask the same thing twice. */
    private readonly queries;
    constructor(options?: IndexOptions);
    get ready(): boolean;
    /**
     * Learn these documents, or notice they are already learned.
     *
     * Safe to call on every open: identical input returns immediately, and two
     * callers arriving together share the work. Failure leaves the index empty
     * rather than half-built, so `search` answers nothing and whatever this was
     * improving carries on without it.
     */
    build(documents: Document[], onProgress?: Progress): Promise<void>;
    private fill;
    /**
     * What this query is about, best first.
     *
     * Nothing at all until the index is built, and nothing for a query too short
     * to mean anything — two letters are a prefix, not a subject.
     */
    search(query: string, limit?: number): Promise<Match[]>;
}
/**
 * The question a follow-up is really asking.
 *
 * "what about weaknesses?" means nothing on its own: embedded alone it matches
 * whatever the question before it matched, and a model handed those passages
 * will answer the wrong question convincingly. A short question after another
 * is almost always about the same subject, so the previous one is carried
 * forward for the *search* — never for what the model is asked.
 *
 * Length is the test rather than a list of opening words, because "and infra?"
 * and "why" are follow-ups too and nothing about them looks like one.
 */
declare function contextual(question: string, previous?: string, words?: number): string;

/**
 * Putting what is known in front of the question.
 *
 * The format is not a detail. The first version of this opened with "Use only
 * the notes below to answer. If they do not cover it, say you do not know." —
 * and a small model, with the answer sitting directly above the question,
 * replied "I'm sorry, but I can't assist with that request." It reads a
 * prohibition as a policy it is being tested on rather than as material.
 *
 * What works is stating what the notes are and then asking. The
 * staying-inside-them half is carried by them being the only relevant thing in
 * the window, which at this size is enough.
 */
interface GroundOptions {
    /** What the notes are, in the app's own words. */
    heading?: string;
    /** How to end: the instruction the question is attached to. */
    instruction?: string;
    /**
     * What to do when nothing was retrieved.
     *
     * `"ask-anyway"` passes the question through untouched, which is right when
     * the model may reasonably know — general knowledge, definitions, code.
     * `"say-unknown"` tells it outright that there is nothing, which is the one
     * instruction that reliably produces "I do not know" rather than fiction. Use
     * it when the corpus is the only thing that could have answered.
     */
    whenEmpty?: "ask-anyway" | "say-unknown";
}
declare function ground(question: string, found: Document[], options?: GroundOptions): string;

/**
 * Questions a browser-local model must not be allowed to answer.
 *
 * Every one of these was learned from a model doing it. Told plainly in its
 * system prompt that it could not search, a 0.5B model still answered "Yes, I
 * can search the web. What would you like to search for?" — and then, asked
 * what was airing, invented a title and a release date. At this size an
 * instruction is a suggestion and agreement is the likeliest next token.
 *
 * So this is not left to the model. The rules are deliberately blunt: a false
 * positive costs somebody one honest sentence, a false negative is a machine
 * lying fluently about something nobody in a browser tab can check.
 *
 * Reasons rather than sentences, because the words belong to the application.
 * A portfolio, a docs site and a support widget should each say this in their
 * own voice.
 */
type Limit = 
/** Asked to search, browse, or otherwise reach the internet. */
"no-internet"
/** Asked about the present: news, prices, weather, what is out now. */
 | "no-present"
/** Asked the date or time. */
 | "no-clock"
/** Asked where the last answer came from — about the conversation, not the subject. */
 | "asks-for-sources";
/**
 * What, if anything, stops this question reaching the model.
 *
 * Order matters: a question can be about both the clock and the present, and
 * "what is today's date" deserves the answer about having no clock rather than
 * an offer to search for it.
 */
declare function limit(question: string): Limit | undefined;

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
declare function workerChat(worker: Worker, model: string, preference?: DevicePreference, options?: ChatOptions): ChatEngine;
/** An embedder that runs somewhere else. One worker can serve both. */
declare function workerEmbedder(worker: Worker, model?: string): Embedder;

/**
 * What the two sides say to each other.
 *
 * Deliberately small and explicit: a worker boundary is the one place in a
 * library where a shape mismatch turns into silence rather than a type error,
 * so every message is one of these and every reply carries the id it answers.
 */
type Request = {
    id: number;
    kind: "load-chat";
    model: string;
    device: DevicePreference;
    options?: ChatOptions;
} | {
    id: number;
    kind: "generate";
    messages: Message[];
} | {
    id: number;
    kind: "load-embed";
    model?: string;
} | {
    id: number;
    kind: "embed";
    texts: string[];
} | {
    id: number;
    kind: "abort";
};
type Response = 
/**
 * A download, as a fraction and in bytes. Sent many times against one
 * request. The bytes travel too because the fraction alone cannot be drawn
 * honestly — see `Loading`.
 */
{
    id: number;
    kind: "progress";
    fraction: number;
    loaded: number;
    total: number;
}
/** A piece of an answer, as it is generated. */
 | {
    id: number;
    kind: "token";
    text: string;
}
/** The request is finished, with whatever it produced. */
 | {
    id: number;
    kind: "done";
    text?: string;
    vectors?: number[][];
    device?: Device;
    fellBack?: boolean;
} | {
    id: number;
    kind: "error";
    message: string;
};

export { CHAT_MODEL, CHAT_MODELS, type ChatEngine, type ChatModel, type ChatOptions, type Device, type DevicePreference, type Document, EMBEDDING_MODEL, type Embedder, type GroundOptions, type IndexOptions, type Limit, type Loading, type Match, type Message, type Progress, VectorIndex, type VectorStore, type Request as WorkerRequest, type Response as WorkerResponse, bestDevice, chatEngine, contextual, embedder, forgetDevice, gpuAvailable, ground, limit, resolveDevice, setChatEngine, setEmbedder, similarity, workerChat, workerEmbedder };
