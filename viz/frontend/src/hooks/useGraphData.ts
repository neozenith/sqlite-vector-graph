/** TanStack Query hooks for graph exploration data. */

import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import { useAppStore } from '@/stores/app-store';
import * as graphService from '@/lib/services/graph-service';
import type { CentralityMeasure, Direction } from '@/lib/types';

export const graphKeys = {
  graphs: ['graphs'] as const,
  subgraph: (table: string, limit: number) =>
    ['graph', table, 'subgraph', limit] as const,
  bfs: (table: string, start: string, depth: number, dir: string) =>
    ['graph', table, 'bfs', start, depth, dir] as const,
  communities: (table: string, resolution: number) =>
    ['graph', table, 'communities', resolution] as const,
  centrality: (table: string, measure: string, dir: string) =>
    ['graph', table, 'centrality', measure, dir] as const,
};

export function useGraphs() {
  return useQuery({
    queryKey: graphKeys.graphs,
    queryFn: graphService.fetchGraphs,
    staleTime: 60_000,
  });
}

export function useSubgraph(edgeTable: string | null, limit: number = 500) {
  return useQuery({
    queryKey: graphKeys.subgraph(edgeTable ?? '', limit),
    queryFn: () => graphService.fetchSubgraph(edgeTable!, limit),
    enabled: !!edgeTable,
    staleTime: 5 * 60_000,
  });
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
  });
}

export function useCommunities(
  edgeTable: string | null,
  resolution: number = 1.0,
) {
  return useQuery({
    queryKey: graphKeys.communities(edgeTable ?? '', resolution),
    queryFn: () => graphService.fetchCommunities(edgeTable!, resolution),
    enabled: !!edgeTable,
    staleTime: 5 * 60_000,
  });
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
  });
}

/** Combined hook for the Graph Explorer component — reads from Zustand store. */
export function useGraphExplorer() {
  const selectedNode = useAppStore((s) => s.selectedNode);
  const setSelectedNode = useAppStore((s) => s.setSelectedNode);
  const measure = useAppStore((s) => s.measure);
  const setMeasure = useAppStore((s) => s.setMeasure);
  const layout = useAppStore((s) => s.layout);
  const setLayout = useAppStore((s) => s.setLayout);
  const communityGrouping = useAppStore((s) => s.communityGrouping);
  const setCommunityGrouping = useAppStore((s) => s.setCommunityGrouping);

  // These remain as local useState since they're not part of deep-linkable state
  const [resolution, setResolution] = useState(1.0);
  const [maxDepth, setMaxDepth] = useState(3);

  return {
    selectedNode,
    setSelectedNode,
    measure,
    setMeasure,
    resolution,
    setResolution,
    maxDepth,
    setMaxDepth,
    layout,
    setLayout,
    communityGrouping,
    setCommunityGrouping,
  };
}
