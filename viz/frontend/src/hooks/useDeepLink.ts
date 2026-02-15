/** Sync URL search params ↔ Zustand store for deep linking. */

import { useEffect, useRef } from 'react';
import { useAppStore } from '@/stores/app-store';
import type { TabId, CentralityMeasure } from '@/lib/types';
import type { LayoutName } from '@/lib/constants';

const VALID_TABS = new Set<string>(['vss', 'graph', 'kg']);
const VALID_MEASURES = new Set<string>(['degree', 'betweenness', 'closeness']);
const VALID_LAYOUTS = new Set<string>(['fcose', 'cose', 'circle', 'grid', 'concentric', 'breadthfirst', 'random']);

/** Parse URL search params and hydrate the Zustand store. */
export function hydrateFromURL(search: string): void {
  const params = new URLSearchParams(search);
  const store = useAppStore.getState();

  const tab = params.get('tab');
  if (tab && VALID_TABS.has(tab)) store.setActiveTab(tab as TabId);

  const index = params.get('index');
  if (index) store.setSelectedIndex(index);

  const dim = params.get('dim');
  if (dim === '2' || dim === '3') store.setDimensions(Number(dim) as 2 | 3);

  const point = params.get('point');
  if (point) {
    const n = Number(point);
    if (Number.isFinite(n)) store.setSelectedPointId(n);
  }

  const k = params.get('k');
  if (k) {
    const n = Number(k);
    if (Number.isInteger(n) && n >= 1 && n <= 100) store.setK(n);
  }

  const q = params.get('q');
  if (q != null) store.setSearchText(q);

  const graph = params.get('graph');
  if (graph) store.setSelectedGraph(graph);

  const layout = params.get('layout');
  if (layout && VALID_LAYOUTS.has(layout)) store.setLayout(layout as LayoutName);

  const measure = params.get('measure');
  if (measure && VALID_MEASURES.has(measure)) store.setMeasure(measure as CentralityMeasure);

  const group = params.get('group');
  if (group === '1' || group === 'true') store.setCommunityGrouping(true);
  else if (group === '0' || group === 'false') store.setCommunityGrouping(false);

  const node = params.get('node');
  if (node) store.setSelectedNode(node);

  const stage = params.get('stage');
  if (stage) {
    const n = Number(stage);
    if (Number.isInteger(n) && n >= 1 && n <= 7) store.setActiveStage(n);
  }
}

/** Serialize the current Zustand state to URL search params string. */
export function serializeToURL(): string {
  const s = useAppStore.getState();
  const params = new URLSearchParams();

  // Only include non-default values to keep URLs clean
  if (s.activeTab !== 'vss') params.set('tab', s.activeTab);
  if (s.selectedIndex) params.set('index', s.selectedIndex);
  if (s.dimensions !== 3) params.set('dim', String(s.dimensions));
  if (s.selectedPointId != null) params.set('point', String(s.selectedPointId));
  if (s.k !== 20) params.set('k', String(s.k));
  if (s.searchText) params.set('q', s.searchText);
  if (s.selectedGraph) params.set('graph', s.selectedGraph);
  if (s.layout !== 'fcose') params.set('layout', s.layout);
  if (s.measure !== 'betweenness') params.set('measure', s.measure);
  if (!s.communityGrouping) params.set('group', '0');
  if (s.selectedNode) params.set('node', s.selectedNode);
  if (s.activeStage !== 1) params.set('stage', String(s.activeStage));

  return params.toString();
}

/**
 * Hook that syncs URL search params with the Zustand store.
 * - On mount: reads URL and hydrates state.
 * - On state change: debounced replaceState to update URL.
 * Call once in a top-level component (e.g. AppContent).
 */
export function useDeepLink(): void {
  const hydrated = useRef(false);

  // Hydrate from URL on mount (once)
  useEffect(() => {
    if (!hydrated.current) {
      hydrated.current = true;
      hydrateFromURL(window.location.search);
    }
  }, []);

  // Subscribe to store changes and update URL
  useEffect(() => {
    let timeoutId: ReturnType<typeof setTimeout>;

    const unsubscribe = useAppStore.subscribe(() => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => {
        const search = serializeToURL();
        const url = search ? `?${search}` : window.location.pathname;
        window.history.replaceState(null, '', url);
      }, 100);
    });

    return () => {
      unsubscribe();
      clearTimeout(timeoutId);
    };
  }, []);
}
