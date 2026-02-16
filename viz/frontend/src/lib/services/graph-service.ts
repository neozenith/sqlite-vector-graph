/** Graph exploration API service. */

import type {
  BfsResponse,
  CentralityMeasure,
  CentralityResponse,
  CommunitiesResponse,
  Direction,
  EdgeTableInfo,
  SubgraphResponse,
} from '../types'
import { fetchJSON } from './api-client'

export function fetchGraphs(): Promise<EdgeTableInfo[]> {
  return fetchJSON<EdgeTableInfo[]>('/api/graphs')
}

export function fetchSubgraph(edgeTable: string, limit: number = 500): Promise<SubgraphResponse> {
  return fetchJSON<SubgraphResponse>(`/api/graph/${encodeURIComponent(edgeTable)}/subgraph?limit=${limit}`)
}

export function fetchBFS(
  edgeTable: string,
  startNode: string,
  maxDepth: number = 3,
  direction: Direction = 'both',
): Promise<BfsResponse> {
  const params = new URLSearchParams({
    start: startNode,
    max_depth: String(maxDepth),
    direction,
  })
  return fetchJSON<BfsResponse>(`/api/graph/${encodeURIComponent(edgeTable)}/bfs?${params}`)
}

export function fetchCommunities(edgeTable: string, resolution: number = 1.0): Promise<CommunitiesResponse> {
  return fetchJSON<CommunitiesResponse>(
    `/api/graph/${encodeURIComponent(edgeTable)}/communities?resolution=${resolution}`,
  )
}

export function fetchCentrality(
  edgeTable: string,
  measure: CentralityMeasure = 'degree',
  direction: Direction = 'both',
): Promise<CentralityResponse> {
  return fetchJSON<CentralityResponse>(
    `/api/graph/${encodeURIComponent(edgeTable)}/centrality?measure=${measure}&direction=${direction}`,
  )
}
