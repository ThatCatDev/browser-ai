/**
 * Small language models and retrieval, running on the machine reading the page.
 *
 * Everything here is browser-only and model-agnostic about its subject: it
 * knows how to load a model, turn text into vectors, find the passages a
 * question is about, and refuse the questions a local model has no business
 * answering. It knows nothing about where the text came from or what the
 * interface looks like — that is the application's, and deliberately so.
 *
 *   import { VectorIndex, chatEngine, ground, limit } from "@thatcatdev/browser-ai";
 *
 * `@huggingface/transformers` is a peer dependency, reached through a dynamic
 * import at the point of use: an app that never runs a model never downloads
 * the library, let alone any weights.
 */

export {
  bestDevice,
  forgetDevice,
  gpuAvailable,
  resolveDevice,
  type Device,
  type DevicePreference,
  type Progress
} from "./device";

export {
  EMBEDDING_MODEL,
  embedder,
  setEmbedder,
  similarity,
  type Embedder
} from "./embed";

export {
  CHAT_MODEL,
  CHAT_MODELS,
  chatEngine,
  setChatEngine,
  type ChatEngine,
  type ChatModel,
  type ChatOptions,
  type Message
} from "./chat";

export {
  VectorIndex,
  contextual,
  type Document,
  type IndexOptions,
  type Match,
  type VectorStore
} from "./retrieval";

export { ground, type GroundOptions } from "./prompt";
export { limit, type Limit } from "./limits";
