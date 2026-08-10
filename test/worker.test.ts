import { describe, it, expect, vi } from 'vitest'
import { workerChat, workerEmbedder } from '../src/worker/client'
import type { Request, Response } from '../src/worker/protocol'

/**
 * A worker that is not one: it records what it was sent and lets a test decide
 * what comes back and when. The real one is exercised in a browser; what
 * matters here is that the client speaks the protocol correctly and survives
 * the thread going away.
 */
const fakeWorker = () => {
  const sent: Request[] = []
  const worker = {
    postMessage: (message: Request) => void sent.push(message),
    terminate: vi.fn(),
    onmessage: null as ((event: MessageEvent<Response>) => void) | null,
    onerror: null as ((event: Event) => void) | null,
  }
  return {
    sent,
    worker: worker as unknown as Worker,
    reply: (message: Response) =>
      worker.onmessage?.({ data: message } as MessageEvent<Response>),
    die: (message = 'worker died') =>
      worker.onerror?.({ message } as unknown as Event),
  }
}

describe('the worker chat client', () => {
  it('asks for a model and reports where it ended up running', async () => {
    const { worker, sent, reply } = fakeWorker()
    const engine = workerChat(worker, 'some/model', 'auto')

    const seen: number[] = []
    const loading = engine.load((fraction) => seen.push(fraction))

    expect(sent[0]).toMatchObject({ kind: 'load-chat', model: 'some/model' })
    reply({ id: sent[0].id, kind: 'progress', fraction: 0.5 })
    reply({ id: sent[0].id, kind: 'done', device: 'webgpu', fellBack: false })

    await loading
    expect(seen).toEqual([0.5])
    expect(engine.device).toBe('webgpu')
    expect(engine.fellBackToCpu).toBe(false)
  })

  it('passes on a fall back to the CPU, so the app can say so', async () => {
    const { worker, sent, reply } = fakeWorker()
    const engine = workerChat(worker, 'some/model', 'webgpu')

    const loading = engine.load()
    reply({ id: sent[0].id, kind: 'done', device: 'wasm', fellBack: true })
    await loading

    expect(engine.device).toBe('wasm')
    expect(engine.fellBackToCpu).toBe(true)
  })

  it('streams tokens as they arrive and returns the whole answer', async () => {
    const { worker, sent, reply } = fakeWorker()
    const engine = workerChat(worker, 'some/model')

    const tokens: string[] = []
    const answering = engine.reply(
      [{ role: 'user', content: 'hello' }],
      (token) => tokens.push(token),
    )

    const id = sent[0].id
    reply({ id, kind: 'token', text: 'Hel' })
    reply({ id, kind: 'token', text: 'lo.' })
    reply({ id, kind: 'done', text: 'Hello.' })

    expect(await answering).toBe('Hello.')
    expect(tokens).toEqual(['Hel', 'lo.'])
  })

  /*
   * Abandoning is a message rather than a rejection: the window that asked has
   * gone, but the thread keeps the model — the next question should not pay a
   * third of a gigabyte for it again.
   */
  it('abandons an answer without tearing the model down', async () => {
    const { worker, sent } = fakeWorker()
    const engine = workerChat(worker, 'some/model')
    const stop = new AbortController()

    void engine.reply([{ role: 'user', content: 'go' }], () => {}, stop.signal)
    stop.abort()

    expect(sent.map((m) => m.kind)).toContain('abort')
    expect((worker as unknown as { terminate: unknown }).terminate).not.toHaveBeenCalled()
  })

  /*
   * A worker that dies takes every outstanding request with it. Without this
   * they never settle, and waiting forever on a thread that is gone looks
   * exactly like a model that is merely slow.
   */
  it('fails what is outstanding when the thread stops', async () => {
    const { worker, sent, die } = fakeWorker()
    const engine = workerChat(worker, 'some/model')

    const loading = engine.load()
    expect(sent).toHaveLength(1)
    die('out of memory')

    await expect(loading).rejects.toThrow(/out of memory/)
  })

  it('lets go of the thread when disposed', () => {
    const { worker } = fakeWorker()
    workerChat(worker, 'some/model').dispose()
    expect((worker as unknown as { terminate: unknown }).terminate).toHaveBeenCalled()
  })
})

describe('the worker embedder client', () => {
  it('sends text and rebuilds the vectors it gets back', async () => {
    const { worker, sent, reply } = fakeWorker()
    const model = workerEmbedder(worker)

    const loading = model.load()
    reply({ id: sent[0].id, kind: 'done', device: 'wasm' })
    await loading

    const embedding = model.embed(['one', 'two'])
    const id = sent[1].id
    reply({ id, kind: 'done', vectors: [[1, 0], [0, 1]] })

    const vectors = await embedding
    expect(vectors[0]).toBeInstanceOf(Float32Array)
    expect(Array.from(vectors[1])).toEqual([0, 1])
  })

  it('does not trouble the thread with nothing to embed', async () => {
    const { worker, sent, reply } = fakeWorker()
    const model = workerEmbedder(worker)
    const loading = model.load()
    reply({ id: sent[0].id, kind: 'done', device: 'wasm' })
    await loading

    expect(await model.embed([])).toEqual([])
    expect(sent).toHaveLength(1)
  })
})

/*
 * Regression: a worker has one `onmessage` slot, so two clients sharing a
 * thread meant the second silently took the first's replies. In the desktop
 * this came out of, the chat model answered normally while the embedder waited
 * forever for messages that were being delivered somewhere else — so the notes
 * never built and every answer came back ungrounded.
 */
describe('two engines, one thread', () => {
  it('delivers to both of them', async () => {
    const { worker, sent, reply } = fakeWorker()
    const chat = workerChat(worker, 'some/model')
    const model = workerEmbedder(worker)

    const loadingChat = chat.load()
    const loadingModel = model.load()

    // Two requests, two different ids, one listener.
    expect(sent.map((m) => m.kind)).toEqual(['load-chat', 'load-embed'])
    expect(new Set(sent.map((m) => m.id)).size).toBe(2)

    reply({ id: sent[1].id, kind: 'done', device: 'wasm' })
    reply({ id: sent[0].id, kind: 'done', device: 'webgpu' })

    await Promise.all([loadingChat, loadingModel])
    expect(chat.device).toBe('webgpu')
    expect(model.device).toBe('wasm')
  })

  // A chat window closing must not take the launcher's embedder down with it.
  it('keeps the thread while anything is still using it', async () => {
    const { worker } = fakeWorker()
    const chat = workerChat(worker, 'some/model')
    const model = workerEmbedder(worker)

    chat.dispose()
    expect((worker as unknown as { terminate: unknown }).terminate).not.toHaveBeenCalled()

    model.dispose()
    expect((worker as unknown as { terminate: unknown }).terminate).toHaveBeenCalled()
  })
})
