import { useEffect, useMemo } from 'react';
import { DeckGL } from '@deck.gl/react';
import { OrthographicView, OrbitView } from '@deck.gl/core';
import { ScatterplotLayer } from '@deck.gl/layers';
import { SimpleMeshLayer } from '@deck.gl/mesh-layers';
import { SphereGeometry } from '@luma.gl/engine';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { useAppStore } from '@/stores/app-store';
import { useIndexes, useEmbeddings, useVSSSearch, useVSSTextSearch, useVSSExplorer } from '@/hooks/useVSSData';
import { toScatterplotData, computeViewState } from '@/lib/transforms/deckgl';
import type { ScatterplotDatum } from '@/lib/transforms/deckgl';
import type { HnswIndexInfo, SearchNeighbor } from '@/lib/types';
import type { Color } from '@deck.gl/core';

const SPHERE_GEOMETRY = new SphereGeometry({ radius: 1, nlat: 16, nlong: 16 });

export function VSSExplorer() {
  const { selectedIndex, setSelectedIndex } = useAppStore();
  const {
    selectedPointId, setSelectedPointId, k, setK, dimensions, setDimensions,
    searchText, setSearchText,
  } = useVSSExplorer();

  const { data: indexes } = useIndexes();
  const { data: embeddings, isLoading: embeddingsLoading } = useEmbeddings(selectedIndex, dimensions);
  const { data: searchResults } = useVSSSearch(selectedIndex, selectedPointId, k);
  const textSearch = useVSSTextSearch(selectedIndex, searchText, k);

  // Auto-select first index (via effect to avoid setState during render)
  useEffect(() => {
    if (indexes?.length && !selectedIndex) {
      setSelectedIndex(indexes[0].name);
    }
  }, [indexes, selectedIndex, setSelectedIndex]);

  // Use text search results when available, otherwise fall back to point-click search
  const activeResults = textSearch.data ?? searchResults;

  const scatterData = embeddings
    ? toScatterplotData(
        embeddings.points,
        activeResults?.neighbors,
        selectedPointId ?? undefined,
      )
    : [];

  const is3D = dimensions === 3;
  const initialViewState = useMemo(
    () => is3D ? computeViewState(scatterData, 800, 600, true) : computeViewState(scatterData),
    [scatterData, is3D],
  );

  const view = is3D ? new OrbitView({ orbitAxis: 'Y' }) : new OrthographicView({});

  // Compute world-space sphere scale: adapts to data spread and point count
  const sphereScale = useMemo(() => {
    if (!is3D || scatterData.length === 0) return 1;
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;
    for (const d of scatterData) {
      const [x, y] = d.position;
      const z = d.position[2] ?? 0;
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    }
    const avgRange = ((maxX - minX) + (maxY - minY) + (maxZ - minZ)) / 3 || 1;
    return avgRange / Math.sqrt(scatterData.length) / 16;
  }, [scatterData, is3D]);

  const getPos = (d: ScatterplotDatum): [number, number, number] =>
    d.position.length === 3 ? d.position as [number, number, number] : [d.position[0], d.position[1], 0];

  const handleClick = (info: { object?: unknown }) => {
    if (info.object) {
      setSelectedPointId((info.object as ScatterplotDatum).id);
    }
  };

  const layer = is3D
    ? new SimpleMeshLayer<ScatterplotDatum>({
        id: 'embeddings-3d',
        data: scatterData,
        mesh: SPHERE_GEOMETRY,
        getPosition: getPos,
        getColor: (d: ScatterplotDatum): Color => [d.color[0], d.color[1], d.color[2], d.opacity],
        getScale: (d: ScatterplotDatum) => {
          const s = d.radius * sphereScale;
          return [s, s, s];
        },
        pickable: true,
        onClick: handleClick,
      })
    : new ScatterplotLayer<ScatterplotDatum>({
        id: 'embeddings',
        data: scatterData,
        getPosition: getPos,
        getFillColor: (d: ScatterplotDatum): Color => [d.color[0], d.color[1], d.color[2], d.opacity],
        getRadius: (d: ScatterplotDatum) => d.radius,
        radiusMinPixels: 2,
        radiusMaxPixels: 15,
        pickable: true,
        onClick: handleClick,
      });

  const selectedNeighbor = activeResults?.neighbors.find(
    (n: SearchNeighbor) => n.id === selectedPointId,
  );

  return (
    <div className="flex h-full gap-4">
      {/* Sidebar */}
      <div className="w-64 flex flex-col gap-3 shrink-0">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Index</CardTitle>
          </CardHeader>
          <CardContent>
            <select
              className="w-full border rounded px-2 py-1 text-sm bg-background"
              value={selectedIndex ?? ''}
              onChange={(e: React.ChangeEvent<HTMLSelectElement>) => {
                setSelectedIndex(e.target.value || null);
                setSelectedPointId(null);
              }}
            >
              <option value="">Select index...</option>
              {indexes?.map((idx: HnswIndexInfo) => (
                <option key={idx.name} value={idx.name}>
                  {idx.name} ({idx.node_count} pts, {idx.dimensions}D {idx.metric})
                </option>
              ))}
            </select>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">View</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex gap-1">
              <Button
                variant={!is3D ? 'default' : 'outline'}
                size="sm"
                className="flex-1 h-7 text-xs"
                onClick={() => setDimensions(2)}
              >
                2D
              </Button>
              <Button
                variant={is3D ? 'default' : 'outline'}
                size="sm"
                className="flex-1 h-7 text-xs"
                onClick={() => setDimensions(3)}
              >
                3D
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Text Search</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex gap-1">
              <Input
                placeholder="Search text..."
                value={searchText}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSearchText(e.target.value)}
                onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
                  if (e.key === 'Enter' && searchText.trim()) {
                    // Trigger is automatic via the hook debounce
                  }
                }}
                className="h-8 text-xs"
              />
            </div>
            {textSearch.isLoading && (
              <div className="text-xs text-muted-foreground mt-1">Searching...</div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Search K</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <Input
                type="number"
                min={1}
                max={100}
                value={k}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setK(Number(e.target.value))}
                className="w-20 h-8"
              />
              <span className="text-xs text-muted-foreground">neighbors</span>
            </div>
          </CardContent>
        </Card>

        {embeddings && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Stats</CardTitle>
            </CardHeader>
            <CardContent className="space-y-1 text-xs">
              <div>Points: <Badge variant="secondary">{embeddings.count}</Badge></div>
              <div>Original dim: <Badge variant="secondary">{embeddings.original_dimensions}</Badge></div>
              <div>Projected: <Badge variant="secondary">{embeddings.projected_dimensions}D</Badge></div>
            </CardContent>
          </Card>
        )}

        {activeResults && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Search Results</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-1 max-h-64 overflow-y-auto">
                {activeResults.neighbors.slice(0, 10).map((n: SearchNeighbor) => (
                  <div
                    key={n.id}
                    className="text-xs p-1 rounded cursor-pointer hover:bg-muted truncate"
                    onClick={() => setSelectedPointId(n.id)}
                  >
                    <span className="font-mono">{n.distance.toFixed(4)}</span>{' '}
                    <span className="text-muted-foreground">{n.label || `#${n.id}`}</span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Canvas */}
      <div className="flex-1 relative bg-muted/30 rounded-lg overflow-hidden">
        {embeddingsLoading ? (
          <div className="absolute inset-0 flex items-center justify-center text-muted-foreground">
            Projecting embeddings with UMAP...
          </div>
        ) : (
          <DeckGL
            views={view}
            initialViewState={initialViewState}
            layers={[layer]}
            controller={true}
            getTooltip={({ object }: { object?: ScatterplotDatum }) =>
              object ? `${object.label || `Point #${object.id}`}` : null
            }
          />
        )}
      </div>

      {/* Detail Panel */}
      {selectedNeighbor && (
        <div className="w-56 shrink-0">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Selected Point</CardTitle>
            </CardHeader>
            <CardContent className="text-xs space-y-1">
              <div>ID: {selectedNeighbor.id}</div>
              <div>Distance: {selectedNeighbor.distance.toFixed(6)}</div>
              {selectedNeighbor.label && <div>Label: {selectedNeighbor.label}</div>}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
