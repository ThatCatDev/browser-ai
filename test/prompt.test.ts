import { describe, it, expect } from 'vitest'
import { ground } from '../src/prompt'
import { limit } from '../src/limits'

const notes = [
  { id: '1', source: 'GoTu', text: 'James led the platform migration at GoTu.' },
]

describe('ground', () => {
  /*
   * The first version opened with "Use only the notes below to answer. If they
   * do not cover it, say you do not know." — and a small model, with the answer
   * directly above the question, replied "I'm sorry, but I can't assist with
   * that request." A prohibition reads as a policy it is being tested on.
   */
  it('states what the notes are and then asks, with nothing in between', () => {
    const asked = ground('what did he do there?', notes)
    expect(asked).toContain('Notes:')
    expect(asked).toContain('- James led the platform migration at GoTu.')
    expect(asked).not.toMatch(/only|do not know/i)
    expect(asked.trimEnd().endsWith('answer briefly: what did he do there?')).toBe(true)
  })

  it('lets the application choose the words around them', () => {
    const asked = ground('and after that?', notes, {
      heading: 'From the handbook:',
      instruction: 'Answer in one line:',
    })
    expect(asked).toContain('From the handbook:')
    expect(asked).toContain('Answer in one line: and after that?')
  })

  // Right when the model may reasonably know: definitions, code, general things.
  it('passes a question through untouched when nothing was found', () => {
    expect(ground('what is a monolith?', [])).toBe('what is a monolith?')
  })

  /*
   * And the opposite, when the corpus was the only thing that could have
   * answered: saying outright that there is nothing is what produces "I do not
   * know" rather than invention.
   */
  it('says outright there is nothing, when asked to', () => {
    const asked = ground('what are his weaknesses?', [], { whenEmpty: 'say-unknown' })
    expect(asked).toContain('There are no notes about this')
    expect(asked).toContain('you do not know')
  })
})

describe('limit', () => {
  it('catches a request to search', () => {
    expect(limit('can you search the web?')).toBe('no-internet')
    expect(limit('google it for me')).toBe('no-internet')
  })

  it('catches anything about the present', () => {
    expect(limit('whats the newest anime airing?')).toBe('no-present')
    expect(limit('what is the weather today')).toBe('no-present')
    expect(limit('latest news')).toBe('no-present')
  })

  // A question can be about both; the clock deserves the clock's answer.
  it('answers the clock before offering to search for the date', () => {
    expect(limit('what day is it?')).toBe('no-clock')
  })

  it('catches somebody asking where an answer came from', () => {
    expect(limit('what is that based on?')).toBe('asks-for-sources')
    expect(limit('how do you know?')).toBe('asks-for-sources')
  })

  // Most things are still the model's to attempt.
  it('lets an ordinary question through', () => {
    expect(limit('what did James do at GoTu?')).toBeUndefined()
    expect(limit('explain what a monolith is')).toBeUndefined()
  })
})
