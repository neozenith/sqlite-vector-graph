/** Zustand store for shared client state. */

import { create } from 'zustand';
import type { TabId, CentralityMeasure } from '../lib/types';
import type { LayoutName } from '../lib/constants';

interface AppState {
  /** Currently active tab. */
  activeTab: TabId;
  setActiveTab: (tab: TabId) => void;

  /** Selected HNSW index name for VSS explorer. */
  selectedIndex: string | null;
  setSelectedIndex: (name: string | null) => void;

  /** Selected edge table for graph explorer. */
  selectedGraph: string | null;
  setSelectedGraph: (name: string | null) => void;

  /** Active KG pipeline stage number (1-7). */
  activeStage: number;
  setActiveStage: (stage: number) => void;

  // ── VSS Explorer state ──────────────────────────────────────────────

  /** UMAP projection dimensions (2D or 3D). */
  dimensions: 2 | 3;
  setDimensions: (d: 2 | 3) => void;

  /** Currently selected point ID for neighbor search. */
  selectedPointId: number | null;
  setSelectedPointId: (id: number | null) => void;

  /** Number of nearest neighbors to search. */
  k: number;
  setK: (k: number) => void;

  /** Text search query for VSS. */
  searchText: string;
  setSearchText: (text: string) => void;

  // ── Graph Explorer state ────────────────────────────────────────────

  /** Selected node in the graph. */
  selectedNode: string | null;
  setSelectedNode: (node: string | null) => void;

  /** Centrality measure to display. */
  measure: CentralityMeasure;
  setMeasure: (m: CentralityMeasure) => void;

  /** Cytoscape layout algorithm. */
  layout: LayoutName;
  setLayout: (l: LayoutName) => void;

  /** Whether community grouping is enabled. */
  communityGrouping: boolean;
  setCommunityGrouping: (on: boolean) => void;

  // ── KG Explorer state ───────────────────────────────────────────────

  /** GraphRAG query text. */
  queryText: string;
  setQueryText: (text: string) => void;
}

export const useAppStore = create<AppState>((set) => ({
  activeTab: 'vss',
  setActiveTab: (tab) => set({ activeTab: tab }),

  selectedIndex: null,
  setSelectedIndex: (name) => set({ selectedIndex: name }),

  selectedGraph: null,
  setSelectedGraph: (name) => set({ selectedGraph: name }),

  activeStage: 1,
  setActiveStage: (stage) => set({ activeStage: stage }),

  // VSS Explorer
  dimensions: 3,
  setDimensions: (d) => set({ dimensions: d }),

  selectedPointId: null,
  setSelectedPointId: (id) => set({ selectedPointId: id }),

  k: 20,
  setK: (k) => set({ k }),

  searchText: '',
  setSearchText: (text) => set({ searchText: text }),

  // Graph Explorer
  selectedNode: null,
  setSelectedNode: (node) => set({ selectedNode: node }),

  measure: 'betweenness',
  setMeasure: (m) => set({ measure: m }),

  layout: 'fcose',
  setLayout: (l) => set({ layout: l }),

  communityGrouping: true,
  setCommunityGrouping: (on) => set({ communityGrouping: on }),

  // KG Explorer
  queryText: '',
  setQueryText: (text) => set({ queryText: text }),
}));
