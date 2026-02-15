/** VSS (Vector Similarity Search) API service. */

import type {
  EmbeddingsResponse,
  HnswIndexInfo,
  SearchResponse,
} from '../types';
import { fetchJSON } from './api-client';

export function fetchIndexes(): Promise<HnswIndexInfo[]> {
  return fetchJSON<HnswIndexInfo[]>('/api/indexes');
}

export function fetchEmbeddings(
  indexName: string,
  dimensions: 2 | 3 = 2,
): Promise<EmbeddingsResponse> {
  return fetchJSON<EmbeddingsResponse>(
    `/api/vss/${encodeURIComponent(indexName)}/embeddings?dimensions=${dimensions}`,
  );
}

export function searchVSS(
  indexName: string,
  queryId: number,
  k: number = 20,
  efSearch: number = 64,
): Promise<SearchResponse> {
  return fetchJSON<SearchResponse>(
    `/api/vss/${encodeURIComponent(indexName)}/search?query_id=${queryId}&k=${k}&ef_search=${efSearch}`,
  );
}

export function searchVSSByText(
  indexName: string,
  query: string,
  k: number = 20,
  efSearch: number = 64,
): Promise<SearchResponse> {
  return fetchJSON<SearchResponse>(
    `/api/vss/${encodeURIComponent(indexName)}/search_text?q=${encodeURIComponent(query)}&k=${k}&ef_search=${efSearch}`,
  );
}
