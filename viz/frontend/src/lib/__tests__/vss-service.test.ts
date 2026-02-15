import { describe, it, expect, vi, beforeEach } from 'vitest';
import { fetchIndexes, fetchEmbeddings, searchVSS, searchVSSByText } from '../services/vss-service';

const mockFetch = vi.fn();
vi.stubGlobal('fetch', mockFetch);

beforeEach(() => {
  mockFetch.mockReset();
});

function mockOk(data: unknown) {
  mockFetch.mockResolvedValueOnce({
    ok: true,
    json: () => Promise.resolve(data),
  });
}

describe('fetchIndexes', () => {
  it('calls /api/indexes', async () => {
    const indexes = [{ name: 'test', dimensions: 4, metric: 'cosine', m: 16, ef_construction: 200, node_count: 10 }];
    mockOk(indexes);
    const result = await fetchIndexes();
    expect(result).toEqual(indexes);
    expect(mockFetch).toHaveBeenCalledWith('/api/indexes', expect.anything());
  });
});

describe('fetchEmbeddings', () => {
  it('calls /api/vss/{name}/embeddings with dimensions', async () => {
    mockOk({ index: 'test', count: 5, points: [] });
    await fetchEmbeddings('test', 3);
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/vss/test/embeddings?dimensions=3',
      expect.anything(),
    );
  });

  it('defaults to 2 dimensions', async () => {
    mockOk({ index: 'test', count: 0, points: [] });
    await fetchEmbeddings('test');
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/vss/test/embeddings?dimensions=2',
      expect.anything(),
    );
  });

  it('encodes special characters in index name', async () => {
    mockOk({ index: 'my index', count: 0, points: [] });
    await fetchEmbeddings('my index');
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/vss/my%20index/embeddings?dimensions=2',
      expect.anything(),
    );
  });
});

describe('searchVSS', () => {
  it('calls /api/vss/{name}/search with params', async () => {
    mockOk({ index: 'test', query_id: 1, k: 10, count: 5, neighbors: [] });
    await searchVSS('test', 1, 10, 32);
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/vss/test/search?query_id=1&k=10&ef_search=32',
      expect.anything(),
    );
  });

  it('uses default k and ef_search', async () => {
    mockOk({ index: 'test', query_id: 1, k: 20, count: 0, neighbors: [] });
    await searchVSS('test', 1);
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/vss/test/search?query_id=1&k=20&ef_search=64',
      expect.anything(),
    );
  });
});

describe('searchVSSByText', () => {
  it('calls /api/vss/{name}/search_text with query', async () => {
    mockOk({ index: 'test', query: 'hello', k: 10, count: 0, neighbors: [] });
    await searchVSSByText('test', 'hello', 10, 32);
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/vss/test/search_text?q=hello&k=10&ef_search=32',
      expect.anything(),
    );
  });

  it('encodes special characters in query', async () => {
    mockOk({ index: 'test', query: 'hello world', k: 20, count: 0, neighbors: [] });
    await searchVSSByText('test', 'hello world');
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/vss/test/search_text?q=hello%20world&k=20&ef_search=64',
      expect.anything(),
    );
  });
});
