/** TanStack Query hooks for VSS (Vector Similarity Search) data. */

import { useQuery } from '@tanstack/react-query'
import * as vssService from '@/lib/services/vss-service'

/** Query key factory for cache management. */
export const vssKeys = {
  indexes: ['indexes'] as const,
  embeddings: (name: string, dims: number) => ['vss', name, 'embeddings', dims] as const,
  search: (name: string, queryId: number, k: number) => ['vss', name, 'search', queryId, k] as const,
  textSearch: (name: string, query: string, k: number) => ['vss', name, 'text-search', query, k] as const,
}

export function useIndexes() {
  return useQuery({
    queryKey: vssKeys.indexes,
    queryFn: vssService.fetchIndexes,
    staleTime: 60_000,
  })
}

export function useEmbeddings(indexName: string | null, dimensions: 2 | 3 = 2) {
  return useQuery({
    queryKey: vssKeys.embeddings(indexName ?? '', dimensions),
    queryFn: () => vssService.fetchEmbeddings(indexName!, dimensions),
    enabled: !!indexName,
    staleTime: 5 * 60_000, // UMAP projections are stable
  })
}

export function useVSSSearch(indexName: string | null, queryId: number | null, k: number = 20) {
  return useQuery({
    queryKey: vssKeys.search(indexName ?? '', queryId ?? 0, k),
    queryFn: () => vssService.searchVSS(indexName!, queryId!, k),
    enabled: !!indexName && queryId != null,
  })
}

export function useVSSTextSearch(indexName: string | null, query: string, k: number = 20) {
  const trimmed = query.trim()
  return useQuery({
    queryKey: vssKeys.textSearch(indexName ?? '', trimmed, k),
    queryFn: () => vssService.searchVSSByText(indexName!, trimmed, k),
    enabled: !!indexName && trimmed.length >= 2,
    staleTime: 30_000,
  })
}
