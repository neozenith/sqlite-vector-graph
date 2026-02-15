/** Props-driven Cytoscape graph for inline graph visualizations.
 * Fixed 300px height, read-only, cose layout. */

// @ts-expect-error react-cytoscapejs has no type declarations
import CytoscapeComponent from 'react-cytoscapejs';
import { useMemo } from 'react';
import type cytoscape from 'cytoscape';
import { toCytoscapeElements, applyCommunityColors } from '@/lib/transforms/cytoscape';
import type { GraphNode, GraphEdge } from '@/lib/types';

interface Props {
  nodes: GraphNode[];
  edges: GraphEdge[];
  communities?: Record<string, number>;
}

const stylesheet: cytoscape.StylesheetStyle[] = [
  {
    selector: 'node',
    style: {
      'background-color': 'data(color)' as any,
      'label': 'data(label)',
      'width': 'data(size)' as any,
      'height': 'data(size)' as any,
      'font-size': '6px',
      'text-valign': 'bottom',
      'text-margin-y': 3,
      'color': '#888',
    },
  },
  {
    selector: 'edge',
    style: {
      'width': 1,
      'line-color': '#ddd',
      'curve-style': 'bezier',
      'target-arrow-shape': 'triangle',
      'target-arrow-color': '#ddd',
      'arrow-scale': 0.4,
    },
  },
];

export function GraphMiniViz({ nodes, edges, communities }: Props) {
  const elements = useMemo(() => {
    let els = toCytoscapeElements(nodes, edges);
    if (communities) {
      els = applyCommunityColors(els, communities);
    }
    return els;
  }, [nodes, edges, communities]);

  if (nodes.length === 0) {
    return (
      <div className="h-[300px] flex items-center justify-center text-muted-foreground text-xs">
        No graph data
      </div>
    );
  }

  return (
    <div className="h-[300px] relative bg-muted/30 rounded overflow-hidden">
      <CytoscapeComponent
        elements={elements as any}
        stylesheet={stylesheet}
        layout={{ name: 'cose', animate: false } as any}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
}
