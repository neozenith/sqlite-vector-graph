import { describe, it, expect, vi, beforeEach } from 'vitest'
import {
  fetchPipeline,
  fetchStage,
  queryGraphRAG,
  fetchStageItems,
  fetchEntitiesGrouped,
  fetchEntitiesByChunk,
} from '../services/kg-service'

const mockFetch = vi.fn()
vi.stubGlobal('fetch', mockFetch)

beforeEach(() => {
  mockFetch.mockReset()
})

function mockOk(data: unknown) {
  mockFetch.mockResolvedValueOnce({
    ok: true,
    json: () => Promise.resolve(data),
  })
}

describe('fetchPipeline', () => {
  it('calls /api/kg/pipeline', async () => {
    mockOk({ stages: [] })
    const result = await fetchPipeline()
    expect(result.stages).toEqual([])
    expect(mockFetch).toHaveBeenCalledWith('/api/kg/pipeline', expect.anything())
  })
})

describe('fetchStage', () => {
  it('calls /api/kg/stage/{n}', async () => {
    mockOk({ stage: 3, name: 'Entity Extraction', count: 100 })
    const result = await fetchStage(3)
    expect(result.stage).toBe(3)
    expect(mockFetch).toHaveBeenCalledWith('/api/kg/stage/3', expect.anything())
  })
})

describe('queryGraphRAG', () => {
  it('sends POST with query body', async () => {
    mockOk({ query: 'test', stages: {} })
    const result = await queryGraphRAG('division of labor', 5, 3)
    expect(result.query).toBe('test')
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/kg/query',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ query: 'division of labor', k: 5, max_depth: 3 }),
      }),
    )
  })

  it('uses default k and maxDepth', async () => {
    mockOk({ query: 'test', stages: {} })
    await queryGraphRAG('test')
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/kg/query',
      expect.objectContaining({
        body: JSON.stringify({ query: 'test', k: 10, max_depth: 2 }),
      }),
    )
  })
})

describe('fetchStageItems', () => {
  it('calls /api/kg/stage/{n}/items with pagination params', async () => {
    mockOk({ stage: 1, items: [], total: 0, page: 1, page_size: 20 })
    await fetchStageItems(1, 2, 10, 'foo')
    expect(mockFetch).toHaveBeenCalledWith('/api/kg/stage/1/items?page=2&page_size=10&q=foo', expect.anything())
  })

  it('omits q when empty', async () => {
    mockOk({ stage: 3, items: [], total: 0, page: 1, page_size: 20 })
    await fetchStageItems(3)
    expect(mockFetch).toHaveBeenCalledWith('/api/kg/stage/3/items?page=1&page_size=20', expect.anything())
  })
})

describe('fetchEntitiesGrouped', () => {
  it('calls entities-grouped with pagination params', async () => {
    mockOk({ items: [], total: 0, page: 2, page_size: 10 })
    await fetchEntitiesGrouped(2, 10, 'alice')
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/kg/stage/3/entities-grouped?page=2&page_size=10&q=alice',
      expect.anything(),
    )
  })

  it('uses defaults and omits q when empty', async () => {
    mockOk({ items: [], total: 0, page: 1, page_size: 20 })
    await fetchEntitiesGrouped()
    expect(mockFetch).toHaveBeenCalledWith('/api/kg/stage/3/entities-grouped?page=1&page_size=20', expect.anything())
  })
})

describe('fetchEntitiesByChunk', () => {
  it('calls entities-by-chunk with pagination params', async () => {
    mockOk({ items: [], total: 0, page: 3, page_size: 5 })
    await fetchEntitiesByChunk(3, 5, 'bob')
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/kg/stage/3/entities-by-chunk?page=3&page_size=5&q=bob',
      expect.anything(),
    )
  })

  it('uses defaults and omits q when empty', async () => {
    mockOk({ items: [], total: 0, page: 1, page_size: 20 })
    await fetchEntitiesByChunk()
    expect(mockFetch).toHaveBeenCalledWith('/api/kg/stage/3/entities-by-chunk?page=1&page_size=20', expect.anything())
  })
})
