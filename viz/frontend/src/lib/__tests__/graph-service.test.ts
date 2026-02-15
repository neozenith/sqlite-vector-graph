import { describe, it, expect, vi, beforeEach } from 'vitest';
import { fetchGraphs, fetchSubgraph, fetchBFS, fetchCommunities, fetchCentrality } from '../services/graph-service';

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

describe('fetchGraphs', () => {
  it('calls /api/graphs', async () => {
    mockOk([{ table_name: 'edges', edge_count: 100 }]);
    const result = await fetchGraphs();
    expect(result).toHaveLength(1);
    expect(mockFetch).toHaveBeenCalledWith('/api/graphs', expect.anything());
  });
});

describe('fetchSubgraph', () => {
  it('calls with limit parameter', async () => {
    mockOk({ nodes: [], edges: [] });
    await fetchSubgraph('edges', 200);
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/graph/edges/subgraph?limit=200',
      expect.anything(),
    );
  });

  it('defaults to limit=500', async () => {
    mockOk({ nodes: [], edges: [] });
    await fetchSubgraph('edges');
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/graph/edges/subgraph?limit=500',
      expect.anything(),
    );
  });
});

describe('fetchBFS', () => {
  it('calls with all parameters', async () => {
    mockOk({ nodes: [] });
    await fetchBFS('edges', 'alice', 2, 'out');
    const url = mockFetch.mock.calls[0][0] as string;
    expect(url).toContain('/api/graph/edges/bfs?');
    expect(url).toContain('start=alice');
    expect(url).toContain('max_depth=2');
    expect(url).toContain('direction=out');
  });

  it('defaults to max_depth=3 and direction=both', async () => {
    mockOk({ nodes: [] });
    await fetchBFS('edges', 'bob');
    const url = mockFetch.mock.calls[0][0] as string;
    expect(url).toContain('max_depth=3');
    expect(url).toContain('direction=both');
  });
});

describe('fetchCommunities', () => {
  it('calls with resolution', async () => {
    mockOk({ communities: {} });
    await fetchCommunities('edges', 1.5);
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/graph/edges/communities?resolution=1.5',
      expect.anything(),
    );
  });
});

describe('fetchCentrality', () => {
  it('calls with measure and direction', async () => {
    mockOk({ scores: [] });
    await fetchCentrality('edges', 'betweenness', 'out');
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/graph/edges/centrality?measure=betweenness&direction=out',
      expect.anything(),
    );
  });

  it('defaults to degree and both', async () => {
    mockOk({ scores: [] });
    await fetchCentrality('edges');
    expect(mockFetch).toHaveBeenCalledWith(
      '/api/graph/edges/centrality?measure=degree&direction=both',
      expect.anything(),
    );
  });
});
