import { describe, it, expect, vi } from 'vitest'
import { VectorIndex, contextual, type Document, type VectorStore } from '../src/retrieval'
import { similarity, type Embedder } from '../src/embed'

const docs: Document[] = [
  { id: 'a', source: 'ep', text: 'A library for handling events with Kafka, Pulsar and RabbitMQ' },
  { id: 'b', source: 'anime-api', text: 'An anime catalogue service' },
  { id: 'c', source: 'GoTu', text: 'As Engineering Manager at GoTu, James led the platform migration' },
]

/** An embedder with opinions we control: one axis per topic. */
const fake = (topics: Record<string, string[]>): Embedder => {
  const axes = Object.keys(topics)
  const vector = (text: string) => {
    const lower = text.toLowerCase()
    const values = axes.map((a) => (topics[a].some((w) => lower.includes(w)) ? 1 : 0))
    const length = Math.hypot(...values) || 1
    return Float32Array.from(values.map((v) => v / length))
  }
  return {
    device: 'wasm',
    load: vi.fn().mockResolvedValue(undefined),
    embed: vi.fn(async (texts: string[]) => texts.map(vector)),
    dispose: vi.fn(),
  }
}

const model = () =>
  fake({
    queues: ['kafka', 'pulsar', 'rabbitmq', 'message queue', 'events'],
    anime: ['anime'],
    manage: ['engineering manager', 'migration'],
  })

const memoryStore = (): VectorStore & { data: Map<string, unknown> } => {
  const data = new Map<string, unknown>()
  return {
    data,
    read: async (key) => data.get(key),
    write: async (key, value) => void data.set(key, value),
  }
}

describe('similarity', () => {
  it('is the cosine, since everything is normalised on the way out', () => {
    const a = Float32Array.from([1, 0])
    expect(similarity(a, a)).toBeCloseTo(1)
    expect(similarity(a, Float32Array.from([-1, 0]))).toBeCloseTo(-1)
    expect(similarity(a, Float32Array.from([0, 1]))).toBeCloseTo(0)
  })
})

describe('VectorIndex', () => {
  it('finds what a query is about, with none of its letters', async () => {
    const index = new VectorIndex({ model: model() })
    await index.build(docs)

    const [best] = await index.search('message queue')
    expect(best.document.id).toBe('a')
  })

  // Cosine always has a best answer, however wrong.
  it('answers nothing for a query about nothing', async () => {
    const index = new VectorIndex({ model: model() })
    await index.build(docs)
    expect(await index.search('asdfghjkl')).toEqual([])
  })

  it('leaves a two-letter query to whatever else the app does', async () => {
    const index = new VectorIndex({ model: model() })
    await index.build(docs)
    expect(await index.search('ka')).toEqual([])
  })

  /*
   * Sentence embeddings are weak on proper nouns: "GoTu" is a rare token
   * carrying almost no meaning, so a question naming it scored no better
   * against the passage naming it back.
   */
  it('nudges a passage that contains the query’s own rare words', async () => {
    const blind = fake({ nothing: ['zzzz'] })   // no semantic signal at all
    const index = new VectorIndex({ model: blind, floor: 0.1, lexicalBoost: 0.5 })
    await index.build(docs)

    const [best] = await index.search('what happened at GoTu')
    expect(best.document.id).toBe('c')
  })

  it('can be told to ignore the words entirely', async () => {
    const blind = fake({ nothing: ['zzzz'] })
    // No floor, so the scores themselves can be inspected rather than filtered.
    const index = new VectorIndex({ model: blind, floor: -1, lexicalBoost: 0 })
    await index.build(docs)

    const found = await index.search('what happened at GoTu')
    // Pure semantics from a model with no signal: nothing is *about* anything,
    // and naming GoTu buys the passage that names it back exactly nothing.
    expect(new Set(found.map((m) => m.score)).size).toBe(1)
  })

  it('embeds a corpus once', async () => {
    const embedder = model()
    const index = new VectorIndex({ model: embedder })
    await index.build(docs)
    await index.build(docs)
    expect((embedder.embed as any).mock.calls).toHaveLength(1)
  })

  it('keeps the vectors in the store it was given', async () => {
    const store = memoryStore()
    await new VectorIndex({ model: model(), store, cacheKey: 'k' }).build(docs)
    expect(store.data.has('k')).toBe(true)
  })

  /*
   * Regression: cached vectors answer "what does the corpus mean" and cannot
   * answer "what does this question mean". Skipping the model load on a cache
   * hit left every later visit with a full index and no way to query it.
   */
  it('takes vectors from the store but still loads the model', async () => {
    const store = memoryStore()
    await new VectorIndex({ model: model(), store, cacheKey: 'k' }).build(docs)

    const second = model()
    const index = new VectorIndex({ model: second, store, cacheKey: 'k' })
    await index.build(docs)

    expect((second.embed as any).mock.calls).toHaveLength(0)
    expect(second.load).toHaveBeenCalled()
    expect((await index.search('message queue'))[0].document.id).toBe('a')
  })

  it('rebuilds when the documents have changed underneath it', async () => {
    const store = memoryStore()
    await new VectorIndex({ model: model(), store, cacheKey: 'k' }).build(docs)

    const second = model()
    await new VectorIndex({ model: second, store, cacheKey: 'k' }).build([
      ...docs,
      { id: 'd', source: 'new', text: 'something else entirely' },
    ])
    expect((second.embed as any).mock.calls).toHaveLength(1)
  })

  // This improves something that already works; it must never break it.
  it('stays empty and silent when the model will not load', async () => {
    const broken: Embedder = {
      device: undefined,
      load: vi.fn().mockRejectedValue(new Error('no adapter, no wasm, no luck')),
      embed: vi.fn(),
      dispose: vi.fn(),
    }
    const index = new VectorIndex({ model: broken })

    await expect(index.build(docs)).resolves.toBeUndefined()
    expect(index.ready).toBe(false)
    expect(await index.search('message queue')).toEqual([])
  })
})

describe('contextual', () => {
  it('carries the previous question into a short follow-up', () => {
    expect(contextual('what about weaknesses?', 'what are his strengths?'))
      .toBe('what are his strengths? what about weaknesses?')
  })

  it('leaves a question that stands on its own', () => {
    const asked = 'what did he build at Taskworld with Docker and Kubernetes'
    expect(contextual(asked, 'tell me about GoTu')).toBe(asked)
  })

  it('has nothing to carry on the first question', () => {
    expect(contextual('hello')).toBe('hello')
  })
})
