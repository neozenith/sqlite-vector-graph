import { describe, it, expect } from 'vitest'
import { DEFAULT_FCOSE_PARAMS, toLayoutOptions } from '../fcose'

describe('DEFAULT_FCOSE_PARAMS', () => {
  it('has expected default values', () => {
    expect(DEFAULT_FCOSE_PARAMS).toEqual({
      nodeRepulsion: 4500,
      idealEdgeLength: 50,
      gravity: 0.25,
    })
  })
})

describe('toLayoutOptions', () => {
  it('wraps nodeRepulsion and idealEdgeLength in functions', () => {
    const opts = toLayoutOptions({ nodeRepulsion: 1000, idealEdgeLength: 80, gravity: 0.5 })
    expect(typeof opts.nodeRepulsion).toBe('function')
    expect(typeof opts.idealEdgeLength).toBe('function')
    expect((opts.nodeRepulsion as () => number)()).toBe(1000)
    expect((opts.idealEdgeLength as () => number)()).toBe(80)
  })

  it('passes gravity as a plain number', () => {
    const opts = toLayoutOptions(DEFAULT_FCOSE_PARAMS)
    expect(opts.gravity).toBe(0.25)
  })
})
