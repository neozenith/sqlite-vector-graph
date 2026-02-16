/** TanStack Query hooks for graph exploration data. */

import { useQuery } from '@tanstack/react-query'
import * as graphService from '@/lib/services/graph-service'
import type { CentralityMeasure, Direction } from '@/lib/types'

export const graphKeys = {
  graphs: ['graphs'] as const,
  subgraph: (table: string, limit: number) => ['graph', table, 'subgraph', limit] as const,
  bfs: (table: string, start: string, depth: number, dir: string) =>
    ['graph', table, 'bfs', start, depth, dir] as const,
  communities: (table: string, resolution: number) => ['graph', table, 'communities', resolution] as const,
  centrality: (table: string, measure: string, dir: string) => ['graph', table, 'centrality', measure, dir] as const,
}

export function useGraphs() {
  return useQuery({
    queryKey: graphKeys.graphs,
    queryFn: graphService.fetchGraphs,
    staleTime: 60_000,
  })
}

export function useSubgraph(edgeTable: string | null, limit: number = 500) {
  return useQuery({
    queryKey: graphKeys.subgraph(edgeTable ?? '', limit),
    queryFn: () => graphService.fetchSubgraph(edgeTable!, limit),
    enabled: !!edgeTable,
    staleTime: 5 * 60_000,
  })
}

export function useBFS(
  edgeTable: string | null,
  startNode: string | null,
  maxDepth: number = 3,
  direction: Direction = 'both',
) {
  return useQuery({
    queryKey: graphKeys.bfs(edgeTable ?? '', startNode ?? '', maxDepth, direction),
    queryFn: () => graphService.fetchBFS(edgeTable!, startNode!, maxDepth, direction),
    enabled: !!edgeTable && !!startNode,
  })
}

export function useCommunities(edgeTable: string | null, resolution: number = 1.0) {
  return useQuery({
    queryKey: graphKeys.communities(edgeTable ?? '', resolution),
    queryFn: () => graphService.fetchCommunities(edgeTable!, resolution),
    enabled: !!edgeTable,
    staleTime: 5 * 60_000,
  })
}

export function useCentrality(
  edgeTable: string | null,
  measure: CentralityMeasure = 'degree',
  direction: Direction = 'both',
) {
  return useQuery({
    queryKey: graphKeys.centrality(edgeTable ?? '', measure, direction),
    queryFn: () => graphService.fetchCentrality(edgeTable!, measure, direction),
    enabled: !!edgeTable,
    staleTime: 5 * 60_000,
  })
}
