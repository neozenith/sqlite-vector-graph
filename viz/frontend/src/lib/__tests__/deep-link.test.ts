import { describe, it, expect, beforeEach } from 'vitest';
import { useAppStore } from '../../stores/app-store';
import { hydrateFromURL, serializeToURL } from '../../hooks/useDeepLink';

/** Reset store to defaults between tests. */
function resetStore() {
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
}

beforeEach(resetStore);

describe('hydrateFromURL', () => {
  it('sets tab from URL', () => {
    hydrateFromURL('?tab=graph');
    expect(useAppStore.getState().activeTab).toBe('graph');
  });

  it('ignores invalid tab values', () => {
    hydrateFromURL('?tab=invalid');
    expect(useAppStore.getState().activeTab).toBe('vss');
  });

  it('sets VSS index', () => {
    hydrateFromURL('?index=chunks_vec');
    expect(useAppStore.getState().selectedIndex).toBe('chunks_vec');
  });

  it('sets dimensions to 3', () => {
    hydrateFromURL('?dim=3');
    expect(useAppStore.getState().dimensions).toBe(3);
  });

  it('ignores invalid dimension values', () => {
    hydrateFromURL('?dim=4');
    expect(useAppStore.getState().dimensions).toBe(3); // stays at default
  });

  it('sets selected point ID', () => {
    hydrateFromURL('?point=42');
    expect(useAppStore.getState().selectedPointId).toBe(42);
  });

  it('ignores non-numeric point', () => {
    hydrateFromURL('?point=abc');
    expect(useAppStore.getState().selectedPointId).toBeNull();
  });

  it('sets k value', () => {
    hydrateFromURL('?k=50');
    expect(useAppStore.getState().k).toBe(50);
  });

  it('ignores out-of-range k', () => {
    hydrateFromURL('?k=200');
    expect(useAppStore.getState().k).toBe(20);
  });

  it('sets search text', () => {
    hydrateFromURL('?q=hello+world');
    expect(useAppStore.getState().searchText).toBe('hello world');
  });

  it('sets graph table', () => {
    hydrateFromURL('?graph=relations');
    expect(useAppStore.getState().selectedGraph).toBe('relations');
  });

  it('sets layout', () => {
    hydrateFromURL('?layout=circle');
    expect(useAppStore.getState().layout).toBe('circle');
  });

  it('ignores invalid layout', () => {
    hydrateFromURL('?layout=invalid');
    expect(useAppStore.getState().layout).toBe('fcose'); // stays at default
  });

  it('sets measure', () => {
    hydrateFromURL('?measure=betweenness');
    expect(useAppStore.getState().measure).toBe('betweenness');
  });

  it('ignores invalid measure', () => {
    hydrateFromURL('?measure=invalid');
    expect(useAppStore.getState().measure).toBe('betweenness'); // stays at default
  });

  it('sets community grouping on', () => {
    useAppStore.setState({ communityGrouping: false });
    hydrateFromURL('?group=1');
    expect(useAppStore.getState().communityGrouping).toBe(true);
  });

  it('sets community grouping off', () => {
    hydrateFromURL('?group=0');
    expect(useAppStore.getState().communityGrouping).toBe(false);
  });

  it('sets selected node', () => {
    hydrateFromURL('?node=Alice');
    expect(useAppStore.getState().selectedNode).toBe('Alice');
  });

  it('sets active stage', () => {
    hydrateFromURL('?stage=3');
    expect(useAppStore.getState().activeStage).toBe(3);
  });

  it('ignores out-of-range stage', () => {
    hydrateFromURL('?stage=99');
    expect(useAppStore.getState().activeStage).toBe(1);
  });

  it('handles multiple params at once', () => {
    hydrateFromURL('?tab=graph&graph=relations&layout=circle&measure=betweenness&node=Bob');
    const s = useAppStore.getState();
    expect(s.activeTab).toBe('graph');
    expect(s.selectedGraph).toBe('relations');
    expect(s.layout).toBe('circle');
    expect(s.measure).toBe('betweenness');
    expect(s.selectedNode).toBe('Bob');
  });

  it('handles empty search string', () => {
    hydrateFromURL('');
    // Everything stays at defaults
    expect(useAppStore.getState().activeTab).toBe('vss');
    expect(useAppStore.getState().dimensions).toBe(3);
  });
});

describe('serializeToURL', () => {
  it('returns empty string for default state', () => {
    expect(serializeToURL()).toBe('');
  });

  it('includes tab when not default', () => {
    useAppStore.setState({ activeTab: 'graph' });
    expect(serializeToURL()).toContain('tab=graph');
  });

  it('omits tab when default (vss)', () => {
    expect(serializeToURL()).not.toContain('tab=');
  });

  it('includes index when set', () => {
    useAppStore.setState({ selectedIndex: 'chunks_vec' });
    expect(serializeToURL()).toContain('index=chunks_vec');
  });

  it('includes dim when 2D (non-default)', () => {
    useAppStore.setState({ dimensions: 2 });
    expect(serializeToURL()).toContain('dim=2');
  });

  it('omits dim when 3D (default)', () => {
    expect(serializeToURL()).not.toContain('dim=');
  });

  it('includes point when selected', () => {
    useAppStore.setState({ selectedPointId: 42 });
    expect(serializeToURL()).toContain('point=42');
  });

  it('includes k when not default', () => {
    useAppStore.setState({ k: 50 });
    expect(serializeToURL()).toContain('k=50');
  });

  it('omits k when default (20)', () => {
    expect(serializeToURL()).not.toContain('k=');
  });

  it('includes search text', () => {
    useAppStore.setState({ searchText: 'hello world' });
    const url = serializeToURL();
    expect(url).toContain('q=hello');
  });

  it('includes graph when set', () => {
    useAppStore.setState({ selectedGraph: 'relations' });
    expect(serializeToURL()).toContain('graph=relations');
  });

  it('includes layout when not default', () => {
    useAppStore.setState({ layout: 'circle' });
    expect(serializeToURL()).toContain('layout=circle');
  });

  it('omits layout when fcose (default)', () => {
    expect(serializeToURL()).not.toContain('layout=');
  });

  it('includes measure when not default', () => {
    useAppStore.setState({ measure: 'degree' });
    expect(serializeToURL()).toContain('measure=degree');
  });

  it('omits measure when betweenness (default)', () => {
    expect(serializeToURL()).not.toContain('measure=');
  });

  it('includes group=0 when disabled (non-default)', () => {
    useAppStore.setState({ communityGrouping: false });
    expect(serializeToURL()).toContain('group=0');
  });

  it('omits group when enabled (default)', () => {
    expect(serializeToURL()).not.toContain('group=');
  });

  it('includes node when selected', () => {
    useAppStore.setState({ selectedNode: 'Alice' });
    expect(serializeToURL()).toContain('node=Alice');
  });

  it('includes stage when not default', () => {
    useAppStore.setState({ activeStage: 5 });
    expect(serializeToURL()).toContain('stage=5');
  });
});

describe('round-trip: serialize → hydrate', () => {
  it('preserves all non-default state through round-trip', () => {
    useAppStore.setState({
      activeTab: 'graph',
      selectedGraph: 'relations',
      layout: 'circle',
      measure: 'betweenness',
      communityGrouping: true,
      selectedNode: 'Alice',
    });

    const url = serializeToURL();
    resetStore();
    hydrateFromURL(`?${url}`);

    const s = useAppStore.getState();
    expect(s.activeTab).toBe('graph');
    expect(s.selectedGraph).toBe('relations');
    expect(s.layout).toBe('circle');
    expect(s.measure).toBe('betweenness');
    expect(s.communityGrouping).toBe(true);
    expect(s.selectedNode).toBe('Alice');
  });

  it('preserves VSS state through round-trip', () => {
    useAppStore.setState({
      selectedIndex: 'chunks_vec',
      dimensions: 3,
      selectedPointId: 42,
      k: 50,
      searchText: 'neural networks',
    });

    const url = serializeToURL();
    resetStore();
    hydrateFromURL(`?${url}`);

    const s = useAppStore.getState();
    expect(s.selectedIndex).toBe('chunks_vec');
    expect(s.dimensions).toBe(3);
    expect(s.selectedPointId).toBe(42);
    expect(s.k).toBe(50);
    expect(s.searchText).toBe('neural networks');
  });
});
