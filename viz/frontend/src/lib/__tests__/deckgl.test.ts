import { describe, it, expect } from 'vitest';
import { toScatterplotData, searchResultIds, computeViewState } from '../transforms/deckgl';
import type { ScatterplotDatum, ViewState3D } from '../transforms/deckgl';
import { BACKGROUND_POINT_COLOR, SEARCH_HIGHLIGHT_COLOR, SELECTED_POINT_COLOR } from '../constants';
import type { EmbeddingPoint, SearchNeighbor } from '../types';

const points: EmbeddingPoint[] = [
  { id: 1, x: 0, y: 0, label: 'a', metadata: {} },
  { id: 2, x: 1, y: 1, label: 'b', metadata: {} },
  { id: 3, x: 2, y: 2, label: 'c', metadata: {} },
];

describe('toScatterplotData', () => {
  it('converts points to scatterplot data', () => {
    const data = toScatterplotData(points);
    expect(data).toHaveLength(3);
    expect(data[0].position).toEqual([0, 0]);
    expect(data[0].id).toBe(1);
    expect(data[0].label).toBe('a');
    expect(data[0].color).toEqual(BACKGROUND_POINT_COLOR);
    expect(data[0].radius).toBe(5);
    expect(data[0].opacity).toBe(128); // 50% for background
  });

  it('highlights search results in orange with high opacity', () => {
    const results: SearchNeighbor[] = [
      { id: 2, distance: 0.1, label: 'b', metadata: {} },
    ];
    const data = toScatterplotData(points, results);
    expect(data[1].color).toEqual(SEARCH_HIGHLIGHT_COLOR);
    expect(data[1].opacity).toBe(220);
    expect(data[0].color).toEqual(BACKGROUND_POINT_COLOR); // non-result
    expect(data[0].opacity).toBe(64); // 25% when search active
  });

  it('highlights selected point in red with full opacity', () => {
    const data = toScatterplotData(points, undefined, 1);
    expect(data[0].color).toEqual(SELECTED_POINT_COLOR);
    expect(data[0].opacity).toBe(255);
    // unselected points fade to 25% when a point is selected
    expect(data[1].opacity).toBe(64);
  });

  it('selected takes priority over search highlight', () => {
    const results: SearchNeighbor[] = [
      { id: 1, distance: 0.05, label: 'a', metadata: {} },
    ];
    const data = toScatterplotData(points, results, 1);
    expect(data[0].color).toEqual(SELECTED_POINT_COLOR);
  });

  it('all points have uniform radius', () => {
    const results: SearchNeighbor[] = [
      { id: 2, distance: 0.1, label: 'b', metadata: {} },
      { id: 3, distance: 0.5, label: 'c', metadata: {} },
    ];
    const data = toScatterplotData(points, results, 1);
    // selected, search result, and background all have the same radius
    expect(data[0].radius).toBe(5);
    expect(data[1].radius).toBe(5);
    expect(data[2].radius).toBe(5);
  });

  it('includes z coordinate when present', () => {
    const points3d: EmbeddingPoint[] = [
      { id: 1, x: 0, y: 0, z: 5, label: 'a', metadata: {} },
      { id: 2, x: 1, y: 1, label: 'b', metadata: {} },
    ];
    const data = toScatterplotData(points3d);
    expect(data[0].position).toEqual([0, 0, 5]);
    expect(data[1].position).toEqual([1, 1]);
  });
});

describe('searchResultIds', () => {
  it('returns Set of IDs', () => {
    const results: SearchNeighbor[] = [
      { id: 1, distance: 0.1, label: 'a', metadata: {} },
      { id: 3, distance: 0.2, label: 'c', metadata: {} },
    ];
    const ids = searchResultIds(results);
    expect(ids.has(1)).toBe(true);
    expect(ids.has(3)).toBe(true);
    expect(ids.has(2)).toBe(false);
  });
});

describe('computeViewState', () => {
  it('returns origin with zoom 0 for empty data', () => {
    const vs = computeViewState([]);
    expect(vs.target).toEqual([0, 0, 0]);
    expect(vs.zoom).toBe(0);
  });

  it('centers on the bounding box midpoint', () => {
    const data: ScatterplotDatum[] = [
      { position: [-10, -10], id: 1, label: 'a', color: [0, 0, 0], radius: 3, opacity: 128 },
      { position: [10, 10], id: 2, label: 'b', color: [0, 0, 0], radius: 3, opacity: 128 },
    ];
    const vs = computeViewState(data, 800, 600);
    expect(vs.target[0]).toBeCloseTo(0);
    expect(vs.target[1]).toBeCloseTo(0);
    expect(vs.target[2]).toBe(0);
  });

  it('computes positive zoom for small data range in large viewport', () => {
    const data: ScatterplotDatum[] = [
      { position: [0, 0], id: 1, label: 'a', color: [0, 0, 0], radius: 3, opacity: 128 },
      { position: [10, 10], id: 2, label: 'b', color: [0, 0, 0], radius: 3, opacity: 128 },
    ];
    // viewport 800x600, data range 10x10 → zoom should be positive (scaling up)
    const vs = computeViewState(data, 800, 600);
    expect(vs.zoom).toBeGreaterThan(0);
  });

  it('computes negative zoom for large data range in small viewport', () => {
    const data: ScatterplotDatum[] = [
      { position: [-500, -500], id: 1, label: 'a', color: [0, 0, 0], radius: 3, opacity: 128 },
      { position: [500, 500], id: 2, label: 'b', color: [0, 0, 0], radius: 3, opacity: 128 },
    ];
    // viewport 200x200, data range 1000x1000 → zoom should be negative (scaling down)
    const vs = computeViewState(data, 200, 200);
    expect(vs.zoom).toBeLessThan(0);
  });

  it('handles single point without crashing', () => {
    const data: ScatterplotDatum[] = [
      { position: [5, 5], id: 1, label: 'a', color: [0, 0, 0], radius: 3, opacity: 128 },
    ];
    const vs = computeViewState(data, 800, 600);
    expect(vs.target[0]).toBeCloseTo(5);
    expect(vs.target[1]).toBeCloseTo(5);
    // rangeX/Y fallback to 1 → zoom should be large and positive
    expect(vs.zoom).toBeGreaterThan(5);
  });

  it('returns ViewState3D with rotation props when is3D=true', () => {
    const data: ScatterplotDatum[] = [
      { position: [0, 0, 0], id: 1, label: 'a', color: [0, 0, 0], radius: 3, opacity: 128 },
      { position: [10, 10, 10], id: 2, label: 'b', color: [0, 0, 0], radius: 3, opacity: 128 },
    ];
    const vs = computeViewState(data, 800, 600, true);
    expect(vs.rotationX).toBe(30);
    expect(vs.rotationOrbit).toBe(-30);
    expect(vs.orbitAxis).toBe('Y');
  });

  it('centers target[2] on Z midpoint for 3D data', () => {
    const data: ScatterplotDatum[] = [
      { position: [-5, 0, -20], id: 1, label: 'a', color: [0, 0, 0], radius: 3, opacity: 128 },
      { position: [5, 0, 40], id: 2, label: 'b', color: [0, 0, 0], radius: 3, opacity: 128 },
    ];
    const vs = computeViewState(data, 800, 600, true);
    expect(vs.target[2]).toBeCloseTo(10); // (-20+40)/2
  });

  it('accounts for Z range in 3D zoom calculation', () => {
    // Tall Z range should reduce zoom compared to flat 2D
    const data: ScatterplotDatum[] = [
      { position: [0, 0, -500], id: 1, label: 'a', color: [0, 0, 0], radius: 3, opacity: 128 },
      { position: [10, 10, 500], id: 2, label: 'b', color: [0, 0, 0], radius: 3, opacity: 128 },
    ];
    const vs3D = computeViewState(data, 800, 600, true);
    const vs2D = computeViewState(data, 800, 600);
    // 3D zoom must be smaller (more zoomed out) due to large Z range
    expect(vs3D.zoom).toBeLessThan(vs2D.zoom);
  });

  it('returns ViewState3D for empty data when is3D=true', () => {
    const vs = computeViewState([], 800, 600, true);
    expect(vs.target).toEqual([0, 0, 0]);
    expect(vs.zoom).toBe(0);
    expect(vs.rotationX).toBe(30);
    expect(vs.rotationOrbit).toBe(-30);
    expect(vs.orbitAxis).toBe('Y');
  });
});
