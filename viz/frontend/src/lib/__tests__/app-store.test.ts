import { describe, it, expect, beforeEach } from 'vitest';
import { useAppStore } from '../../stores/app-store';

// Reset store between tests
beforeEach(() => {
  useAppStore.setState({
    activeTab: 'vss',
    selectedIndex: null,
    selectedGraph: null,
    activeStage: 1,
    dimensions: 3,
    selectedPointId: null,
    k: 20,
    searchText: '',
    selectedNode: null,
    measure: 'betweenness',
    layout: 'fcose',
    communityGrouping: true,
    queryText: '',
  });
});

describe('useAppStore', () => {
  it('has correct initial state', () => {
    const state = useAppStore.getState();
    expect(state.activeTab).toBe('vss');
    expect(state.selectedIndex).toBeNull();
    expect(state.selectedGraph).toBeNull();
    expect(state.activeStage).toBe(1);
  });

  it('sets active tab', () => {
    useAppStore.getState().setActiveTab('graph');
    expect(useAppStore.getState().activeTab).toBe('graph');
  });

  it('sets selected index', () => {
    useAppStore.getState().setSelectedIndex('chunks_vec');
    expect(useAppStore.getState().selectedIndex).toBe('chunks_vec');
  });

  it('clears selected index', () => {
    useAppStore.getState().setSelectedIndex('test');
    useAppStore.getState().setSelectedIndex(null);
    expect(useAppStore.getState().selectedIndex).toBeNull();
  });

  it('sets selected graph', () => {
    useAppStore.getState().setSelectedGraph('edges');
    expect(useAppStore.getState().selectedGraph).toBe('edges');
  });

  it('sets active stage', () => {
    useAppStore.getState().setActiveStage(5);
    expect(useAppStore.getState().activeStage).toBe(5);
  });
});
