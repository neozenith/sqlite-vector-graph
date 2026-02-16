import { describe, it, expect, vi, beforeEach } from 'vitest'
import { fetchJSON, ApiError } from '../services/api-client'

const mockFetch = vi.fn()
vi.stubGlobal('fetch', mockFetch)

beforeEach(() => {
  mockFetch.mockReset()
})

describe('fetchJSON', () => {
  it('returns parsed JSON on success', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ data: 'test' }),
    })

    const result = await fetchJSON<{ data: string }>('/api/test')
    expect(result).toEqual({ data: 'test' })
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/test',
      expect.objectContaining({
        headers: expect.objectContaining({ 'Content-Type': 'application/json' }),
      }),
    )
  })

  it('throws ApiError on non-ok response with JSON body', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 404,
      statusText: 'Not Found',
      json: () => Promise.resolve({ detail: 'not found' }),
    })

    await expect(fetchJSON('/api/missing')).rejects.toThrow(ApiError)
    try {
      await fetchJSON('/api/missing')
    } catch {
      // First call already consumed the mock, re-setup for clarity
    }
  })

  it('throws ApiError with text body when JSON parsing fails', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
      json: () => Promise.reject(new Error('not json')),
      text: () => Promise.resolve('plain error text'),
    })

    await expect(fetchJSON('/api/broken')).rejects.toThrow(ApiError)
  })

  it('passes custom options through', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({}),
    })

    await fetchJSON('/api/test', { method: 'POST', body: '{}' })
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/test',
      expect.objectContaining({
        method: 'POST',
        body: '{}',
      }),
    )
  })
})

describe('ApiError', () => {
  it('has correct properties', () => {
    const err = new ApiError(400, 'Bad Request', { detail: 'oops' })
    expect(err.status).toBe(400)
    expect(err.statusText).toBe('Bad Request')
    expect(err.body).toEqual({ detail: 'oops' })
    expect(err.name).toBe('ApiError')
    expect(err.message).toContain('400')
  })
})
