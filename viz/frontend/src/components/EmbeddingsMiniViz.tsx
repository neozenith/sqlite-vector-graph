/** Props-driven Deck.GL scatter plot for embedding visualizations.
 * Fixed 300px height, read-only, no sidebar. */

import { useMemo } from 'react';
import { DeckGL } from '@deck.gl/react';
import { OrthographicView } from '@deck.gl/core';
import { ScatterplotLayer } from '@deck.gl/layers';
import { toScatterplotData, computeViewState } from '@/lib/transforms/deckgl';
import type { ScatterplotDatum } from '@/lib/transforms/deckgl';
import type { EmbeddingPoint } from '@/lib/types';
import type { Color } from '@deck.gl/core';

interface Props {
  points: EmbeddingPoint[];
}

export function EmbeddingsMiniViz({ points }: Props) {
  const scatterData = useMemo(() => toScatterplotData(points), [points]);
  const viewState = useMemo(() => computeViewState(scatterData), [scatterData]);

  const layer = new ScatterplotLayer<ScatterplotDatum>({
    id: 'mini-embeddings',
    data: scatterData,
    getPosition: (d: ScatterplotDatum) =>
      d.position.length === 3 ? d.position as [number, number, number] : [d.position[0], d.position[1], 0],
    getFillColor: (d: ScatterplotDatum): Color => [d.color[0], d.color[1], d.color[2], d.opacity],
    getRadius: (d: ScatterplotDatum) => d.radius,
    radiusMinPixels: 2,
    radiusMaxPixels: 10,
    pickable: false,
  });

  if (points.length === 0) {
    return (
      <div className="h-[300px] flex items-center justify-center text-muted-foreground text-xs">
        No embedding data
      </div>
    );
  }

  return (
    <div className="h-[300px] relative bg-muted/30 rounded overflow-hidden">
      <DeckGL
        views={new OrthographicView({})}
        initialViewState={viewState}
        layers={[layer]}
        controller={true}
        getTooltip={({ object }: { object?: ScatterplotDatum }) =>
          object ? `${object.label || `#${object.id}`}` : null
        }
      />
    </div>
  );
}
