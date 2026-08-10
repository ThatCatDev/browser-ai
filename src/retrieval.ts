import { embedder, similarity, type Embedder } from "./embed";
import type { Progress } from "./device";

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

export interface Document {
  id: string;
  /** Where it came from, for an answer that has to cite itself. */
  source: string;
  text: string;
}

export interface Match {
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
export interface VectorStore {
  read(key: string): Promise<unknown | undefined>;
  write(key: string, value: unknown): Promise<void>;
}

export interface IndexOptions {
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

interface Stored {
  fingerprint: string;
  vectors: Record<string, number[]>;
}

const DEFAULTS = {
  floor: 0.3,
  lexicalBoost: 0.25,
  cacheKey: "browser-ai:vectors"
};

export class VectorIndex {
  private readonly model: Embedder;
  private readonly store?: VectorStore;
  private readonly cacheKey: string;
  private readonly floor: number;
  private readonly lexicalBoost: number;
  private readonly stopWords: Set<string>;

  private readonly vectors = new Map<string, Float32Array>();
  private documents: Document[] = [];
  private building?: Promise<void>;
  private fingerprint = "";
  /** Query vectors, because people ask the same thing twice. */
  private readonly queries = new Map<string, Float32Array>();

  constructor(options: IndexOptions = {}) {
    this.model = options.model ?? embedder();
    this.store = options.store;
    this.cacheKey = options.cacheKey ?? DEFAULTS.cacheKey;
    this.floor = options.floor ?? DEFAULTS.floor;
    this.lexicalBoost = options.lexicalBoost ?? DEFAULTS.lexicalBoost;
    this.stopWords = new Set(options.stopWords ?? []);
  }

  get ready(): boolean {
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
  build(documents: Document[], onProgress?: Progress): Promise<void> {
    const fingerprint = documents.map((doc) => `${doc.id}:${doc.text.length}`).join("|");
    if (this.fingerprint === fingerprint && this.vectors.size) return Promise.resolve();
    if (this.building && this.fingerprint === fingerprint) return this.building;

    this.fingerprint = fingerprint;
    this.documents = documents;
    this.building = this.fill(fingerprint, onProgress).catch(() => {
      this.vectors.clear();
      this.building = undefined;
    });
    return this.building;
  }

  private async fill(fingerprint: string, onProgress?: Progress) {
    /*
     * The model comes in even when the vectors do not.
     *
     * Cached vectors answer "what does the corpus mean"; they cannot answer
     * "what does this question mean", and that needs the model at query time.
     * Loading it only on a cache miss leaves every visit after the first with a
     * full index and no way to ask it anything.
     */
    await this.model.load(onProgress);

    const stored = (await this.store?.read(this.cacheKey)) as Stored | undefined;
    if (stored?.fingerprint === fingerprint) {
      Object.entries(stored.vectors).forEach(([id, vector]) =>
        this.vectors.set(id, Float32Array.from(vector))
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
    } satisfies Stored);
  }

  /**
   * What this query is about, best first.
   *
   * Nothing at all until the index is built, and nothing for a query too short
   * to mean anything — two letters are a prefix, not a subject.
   */
  async search(query: string, limit = 4): Promise<Match[]> {
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

    const terms = this.lexicalBoost
      ? q
          .toLowerCase()
          .split(/[^a-z0-9.+#-]+/)
          .filter((term) => term.length >= 4 && !this.stopWords.has(term))
      : [];

    return this.documents
      .map((document) => {
        const lower = document.text.toLowerCase();
        const hits = terms.filter((term) => lower.includes(term)).length;
        const lexical = terms.length ? (hits / terms.length) * this.lexicalBoost : 0;
        return {
          document,
          score:
            similarity(vector!, this.vectors.get(document.id) ?? new Float32Array()) +
            lexical
        };
      })
      .filter((match) => match.score >= this.floor)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
  }
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
export function contextual(question: string, previous?: string, words = 6): string {
  const q = question.trim();
  if (!previous) return q;
  return q.split(/\s+/).length <= words ? `${previous.trim()} ${q}` : q;
}
