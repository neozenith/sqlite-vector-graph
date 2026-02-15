/** KG Pipeline API service. */

import type {
  GraphRAGResult,
  PipelineSummary,
  StageDetail,
  StageItemsResponse,
} from '../types';
import { fetchJSON } from './api-client';

export function fetchPipeline(): Promise<PipelineSummary> {
  return fetchJSON<PipelineSummary>('/api/kg/pipeline');
}

export function fetchStage(stageNum: number): Promise<StageDetail> {
  return fetchJSON<StageDetail>(`/api/kg/stage/${stageNum}`);
}

export function queryGraphRAG(
  query: string,
  k: number = 10,
  maxDepth: number = 2,
): Promise<GraphRAGResult> {
  return fetchJSON<GraphRAGResult>('/api/kg/query', {
    method: 'POST',
    body: JSON.stringify({ query, k, max_depth: maxDepth }),
  });
}

export function fetchStageItems(
  stage: number,
  page: number = 1,
  pageSize: number = 20,
  query: string = '',
): Promise<StageItemsResponse> {
  const params = new URLSearchParams({
    page: String(page),
    page_size: String(pageSize),
  });
  if (query) params.set('q', query);
  return fetchJSON<StageItemsResponse>(`/api/kg/stage/${stage}/items?${params}`);
}
