import { describe, it, expect, beforeEach, vi } from 'vitest'
import { forgetDevice, bestDevice, resolveDevice, Downloads } from '../src/device'

const nav = navigator as unknown as Record<string, unknown>

beforeEach(() => {
  delete nav.gpu
  forgetDevice()
})

describe('choosing a device', () => {
  it('uses the GPU when the driver hands one over', async () => {
    nav.gpu = { requestAdapter: async () => ({}) }
    await expect(resolveDevice('auto')).resolves.toEqual({ device: 'webgpu', fellBack: false })
  })

  /*
   * The case worth having a library for: Chrome ships `navigator.gpu` and then
   * refuses an adapter on a long list of older Intel parts and Linux drivers.
   * Checking only for the object means the model fails deep inside the pipeline,
   * on machines where the CPU path would have worked.
   */
  it('falls back when the API is there but the adapter is refused', async () => {
    nav.gpu = { requestAdapter: async () => null }
    await expect(resolveDevice('auto')).resolves.toEqual({ device: 'wasm', fellBack: false })

    forgetDevice()
    nav.gpu = { requestAdapter: async () => null }
    // Asked for outright, it still falls back — and says it had to.
    await expect(resolveDevice('webgpu')).resolves.toEqual({ device: 'wasm', fellBack: true })
  })

  it('falls back when asking for an adapter throws', async () => {
    nav.gpu = {
      requestAdapter: async () => {
        throw new Error('driver blocklisted')
      },
    }
    await expect(resolveDevice('webgpu')).resolves.toEqual({ device: 'wasm', fellBack: true })
  })

  it('takes the CPU at its word, without asking the driver', async () => {
    let asked = false
    nav.gpu = {
      requestAdapter: async () => {
        asked = true
        return {}
      },
    }
    await expect(resolveDevice('wasm')).resolves.toEqual({ device: 'wasm', fellBack: false })
    expect(asked).toBe(false)
  })

  it('answers cheaply for anything that has to draw before the driver replies', () => {
    expect(bestDevice()).toBe('wasm')
    nav.gpu = {}
    expect(bestDevice()).toBe('webgpu')
  })
})

describe('Downloads', () => {
  // A bar that restarts for every file is worse than no bar.
  it('totals across however many files a model turns out to be', () => {
    const downloads = new Downloads()
    downloads.record('model.onnx', 50, 100)
    downloads.record('tokenizer.json', 0, 100)
    expect(downloads.fraction()).toBeCloseTo(0.25)

    downloads.record('tokenizer.json', 100, 100)
    expect(downloads.fraction()).toBeCloseTo(0.75)
  })

  it('is zero before anything with a known size has arrived', () => {
    expect(new Downloads().fraction()).toBe(0)
  })

  /*
   * The bytes are the part a bar can be drawn from. A fraction of the files
   * met so far says 90% while the weights have not been announced; the bytes
   * say 3MB, which anything holding the model's real size can act on.
   */
  it('gives the bytes, not only the ratio between them', () => {
    const downloads = new Downloads()
    downloads.record('tokenizer.json', 2_000_000, 2_000_000)
    expect(downloads.state()).toEqual({
      loaded: 2_000_000,
      total: 2_000_000,
      fraction: 1,
    })

    downloads.record('model.onnx', 0, 800_000_000)
    const { loaded, total, fraction } = downloads.state()
    expect(loaded).toBe(2_000_000)
    expect(total).toBe(802_000_000)
    expect(fraction).toBeCloseTo(0.0025)
  })

  it('calls everything it knows about arrived, when it is', () => {
    const downloads = new Downloads()
    downloads.record('model.onnx', 10, 100)
    expect(downloads.finished()).toEqual({ loaded: 100, total: 100, fraction: 1 })
  })
})
