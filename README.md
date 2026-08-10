# @thatcatdev/browser-ai

Small language models and retrieval, running in the browser on the visitor's own
machine. No server, no key, no request leaving the tab.

```ts
import { VectorIndex, chatEngine, ground, limit } from "@thatcatdev/browser-ai";
```

## Installing

Not on a registry — install it from the tag you want:

```sh
bun add github:ThatCatDev/browser-ai#v1.0.0
```

`dist/` is committed at every release, so there is nothing to build on install.
`@huggingface/transformers` is a **peer dependency**: the host app owns the
version, and there is never a second copy of a large library or a second WASM
runtime in the bundle.

## What it does, and what it deliberately does not

It knows how to load a model, turn text into vectors, find the passages a
question is about, and refuse the questions a local model has no business
answering. It knows nothing about where your text comes from, how it is stored,
or what your interface looks like — you pass documents in and get matches back.

## Retrieval

```ts
const index = new VectorIndex({
  store,               // anything with read(key) / write(key, value)
  cacheKey: "notes",
  floor: 0.32,         // higher for a prompt than for a search box
  lexicalBoost: 0.25   // helps questions that name a rare word find it
});

await index.build(documents);          // { id, source, text }[]
const found = await index.search("message queue");
```

Brute-force cosine over the whole corpus, on purpose. A few thousand vectors is
a few milliseconds a query, and an approximate index would cost more to build
than the search it saves. Past ten thousand or so, this is the wrong tool and
sqlite-vec or pgvector in WASM is the right one.

Two details that decide whether retrieval is any good:

- **`lexicalBoost`** — sentence embeddings are weak on proper nouns. A question
  naming a company or a library scores no better against the passage naming it
  back unless the words themselves count for something.
- **`contextual(question, previous)`** — "what about weaknesses?" means nothing
  embedded on its own. Short follow-ups carry the previous question into the
  *search* only.

## Off the main thread

Bringing a 0.5B model up costs about **five seconds of solid arithmetic**, and
on the main thread that is five seconds in which nothing responds — no menu
opens, no window drags, no key registers. The failure that produces is worse
than the delay: the interface looks dead, so the person presses again, and both
presses arrive at once when it thaws.

So run it in a worker. Same interfaces, same behaviour, a page that still
answers:

```ts
// model.worker.ts — the whole file
import "@thatcatdev/browser-ai/worker";
```

```ts
import { workerChat, workerEmbedder, CHAT_MODEL } from "@thatcatdev/browser-ai";

const worker = new Worker(new URL("./model.worker.ts", import.meta.url), {
  type: "module"
});

const engine = workerChat(worker, CHAT_MODEL, "auto");   // implements ChatEngine
const model  = workerEmbedder(worker);                    // implements Embedder
```

The `Worker` is constructed by the application rather than by this package,
because every bundler has its own idea of how a worker URL should be written and
a library that guesses wrong breaks the build rather than degrading. One small
file, and no surprises.

One worker can serve both. Aborting a generation is a message, not a teardown:
the window that asked has gone, but the thread keeps the model, so the next
question does not pay for it again.

## Chat

```ts
const engine = chatEngine(CHAT_MODEL, "auto");
await engine.load((fraction) => show(fraction));
await engine.reply(messages, (token) => append(token), signal);
```

Three models are described in `CHAT_MODELS`, with sizes, because the size is
what a visitor is really choosing. The default is Qwen2.5 0.5B — the first rung
that answers the question it was asked rather than writing a scene around it.

## Grounding, and knowing when not to answer

```ts
const asked = ground(question, found, { whenEmpty: "say-unknown" });
```

The format is not a detail. Opening with "use only the notes below, and say you
do not know otherwise" makes a small model refuse outright — it reads a
prohibition as a policy it is being tested on. Notes, then the question, and
nothing in between.

```ts
switch (limit(question)) {
  case "no-internet":      // asked to search or browse
  case "no-present":       // asked about news, prices, what is out now
  case "no-clock":         // asked the date or the time
  case "asks-for-sources": // asked where the last answer came from
}
```

`limit` returns a reason, never a sentence: the words belong to your app. Told
plainly in its system prompt that it could not search, a 0.5B model still
answered "Yes, I can search the web" and then invented a release date. At this
size an instruction is a suggestion, so these are decided in code before a token
is generated.

## Devices

WebGPU where the machine has it, WASM where it does not, one set of ONNX weights
either way. `resolveDevice` asks the driver for an adapter rather than checking
that `navigator.gpu` exists — Chrome ships the API and then refuses an adapter
on a range of older Intel parts and Linux drivers, and checking only for the
object means the model fails deep in the pipeline on machines where the CPU path
would have worked.

Threads are never enabled. Multi-threaded WASM needs `SharedArrayBuffer`, which
needs `Cross-Origin-Embedder-Policy`, which refuses to embed any cross-origin
frame that has not opted in — a library should not be able to break its host's
iframes in exchange for being faster.

## Releases

Conventional commits on `main`, `semantic-release` for the tag. `fix:` is a
patch, `feat:` a minor, `!` or `BREAKING CHANGE:` a major.

## Licence

MIT

> **Note on installing from git**: there is no `prepare` script on purpose.
> `dist/` is committed at every release, so a consumer never has to install this
> package's build tooling — `bun add github:ThatCatDev/browser-ai#v1.0.1` fetches
> a tag that already contains what it needs.
