// @ts-expect-error react-cytoscapejs has no type declarations
import CytoscapeComponent from 'react-cytoscapejs';
import { useEffect, useRef, useCallback } from 'react';
import cytoscape from 'cytoscape';
// @ts-expect-error cytoscape-fcose has no type declarations
import fcose from 'cytoscape-fcose';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { useAppStore } from '@/stores/app-store';
import {
  useGraphs,
  useSubgraph,
  useBFS,
  useCommunities,
  useCentrality,
  useGraphExplorer,
} from '@/hooks/useGraphData';
import {
  toCytoscapeElements,
  applyCommunityColors,
  applyCentralitySizing,
  highlightBfsNodes,
  applyCommunityGrouping,
} from '@/lib/transforms/cytoscape';
import { LAYOUT_OPTIONS } from '@/lib/constants';
import type { LayoutName } from '@/lib/constants';

// Register fCoSE layout extension once
cytoscape.use(fcose);

export function GraphExplorer() {
  const { selectedGraph, setSelectedGraph } = useAppStore();
  const {
    selectedNode, setSelectedNode, measure, setMeasure, resolution, maxDepth,
    layout, setLayout, communityGrouping, setCommunityGrouping,
  } = useGraphExplorer();

  const cyRef = useRef<cytoscape.Core | null>(null);

  const { data: graphs } = useGraphs();
  const { data: subgraph, isLoading: subgraphLoading } = useSubgraph(selectedGraph);
  const { data: bfsData } = useBFS(selectedGraph, selectedNode, maxDepth);
  const { data: communities } = useCommunities(selectedGraph, resolution);
  const { data: centrality } = useCentrality(selectedGraph, measure);

  // Auto-select first graph (via effect to avoid setState during render)
  useEffect(() => {
    if (graphs?.length && !selectedGraph) {
      setSelectedGraph(graphs[0].table_name);
    }
  }, [graphs, selectedGraph, setSelectedGraph]);

  // Build Cytoscape elements
  let elements = subgraph
    ? toCytoscapeElements(subgraph.nodes, subgraph.edges)
    : [];

  // Apply community colors
  if (communities?.node_community) {
    elements = applyCommunityColors(elements, communities.node_community);
  }

  // Apply centrality sizing
  if (centrality?.scores) {
    elements = applyCentralitySizing(elements, centrality.scores);
  }

  // Apply community grouping (compound parent nodes)
  if (communityGrouping && communities?.node_community) {
    elements = applyCommunityGrouping(elements, communities.node_community);
  }

  // Highlight BFS results
  if (bfsData?.nodes) {
    const bfsNodeIds = new Set(bfsData.nodes.map((n) => n.node));
    elements = highlightBfsNodes(elements, bfsNodeIds);
  }

  const runLayout = useCallback(() => {
    if (cyRef.current) {
      cyRef.current.layout({ name: layout, animate: true, animationDuration: 300 } as any).run();
    }
  }, [layout]);

  const cytoscapeStylesheet: cytoscape.StylesheetStyle[] = [
    {
      selector: 'node',
      style: {
        'background-color': 'data(color)' as any,
        'label': 'data(label)',
        'width': 'data(size)' as any,
        'height': 'data(size)' as any,
        'font-size': '8px',
        'text-valign': 'bottom',
        'text-margin-y': 4,
        'color': '#666',
      },
    },
    {
      selector: 'node[?highlighted]',
      style: {
        'border-width': 3,
        'border-color': '#ff6600',
      },
    },
    {
      selector: 'edge',
      style: {
        'width': 1,
        'line-color': '#ccc',
        'curve-style': 'bezier',
        'target-arrow-shape': 'triangle',
        'target-arrow-color': '#ccc',
        'arrow-scale': 0.5,
        'label': 'data(rel_type)' as any,
        'font-size': '6px',
        'text-rotation': 'autorotate' as any,
        'color': '#999',
        'text-margin-y': -6,
      },
    },
    {
      selector: ':parent',
      style: {
        'background-opacity': 0.05,
        'border-width': 1,
        'border-color': '#aaa',
        'border-opacity': 0.4,
        'label': 'data(label)',
        'font-size': '10px',
        'text-valign': 'top',
        'text-margin-y': -4,
        'color': '#888',
      } as any,
    },
  ];

  return (
    <div className="flex h-full gap-4">
      {/* Sidebar */}
      <div className="w-64 flex flex-col gap-3 shrink-0">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Edge Table</CardTitle>
          </CardHeader>
          <CardContent>
            <select
              className="w-full border rounded px-2 py-1 text-sm bg-background"
              value={selectedGraph ?? ''}
              onChange={(e) => {
                setSelectedGraph(e.target.value || null);
                setSelectedNode(null);
              }}
            >
              <option value="">Select table...</option>
              {graphs?.map((g) => (
                <option key={g.table_name} value={g.table_name}>
                  {g.table_name} ({g.edge_count} edges)
                </option>
              ))}
            </select>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Centrality</CardTitle>
          </CardHeader>
          <CardContent>
            <select
              className="w-full border rounded px-2 py-1 text-sm bg-background"
              value={measure}
              onChange={(e) => setMeasure(e.target.value as any)}
            >
              <option value="degree">Degree</option>
              <option value="betweenness">Betweenness</option>
              <option value="closeness">Closeness</option>
            </select>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Layout</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <select
              className="w-full border rounded px-2 py-1 text-sm bg-background"
              value={layout}
              onChange={(e) => setLayout(e.target.value as LayoutName)}
            >
              {LAYOUT_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
            <label className="flex items-center gap-2 text-xs">
              <input
                type="checkbox"
                checked={communityGrouping}
                onChange={(e) => setCommunityGrouping(e.target.checked)}
              />
              Group by community
            </label>
            <Button
              variant="outline"
              size="sm"
              className="w-full h-7 text-xs"
              onClick={runLayout}
            >
              Re-run layout
            </Button>
          </CardContent>
        </Card>

        {subgraph && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Graph Stats</CardTitle>
            </CardHeader>
            <CardContent className="space-y-1 text-xs">
              <div>Nodes: <Badge variant="secondary">{subgraph.node_count}</Badge></div>
              <div>Edges: <Badge variant="secondary">{subgraph.edge_count}</Badge></div>
              {communities && (
                <div>Communities: <Badge variant="secondary">{communities.community_count}</Badge></div>
              )}
            </CardContent>
          </Card>
        )}

        {selectedNode && bfsData && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">BFS from {selectedNode}</CardTitle>
            </CardHeader>
            <CardContent className="text-xs">
              <div>Reachable: {bfsData.count} nodes</div>
              <div className="mt-1 space-y-0.5 max-h-40 overflow-y-auto">
                {bfsData.nodes.slice(0, 15).map((n) => (
                  <div key={n.node} className="flex justify-between">
                    <span className="truncate">{n.node}</span>
                    <Badge variant="outline" className="text-[10px] ml-1">d={n.depth}</Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {centrality?.scores && (
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Top Centrality</CardTitle>
            </CardHeader>
            <CardContent className="text-xs space-y-0.5 max-h-40 overflow-y-auto">
              {centrality.scores.slice(0, 10).map((s) => (
                <div key={s.node} className="flex justify-between">
                  <span className="truncate">{s.node}</span>
                  <span className="font-mono text-muted-foreground">{s.centrality.toFixed(4)}</span>
                </div>
              ))}
            </CardContent>
          </Card>
        )}
      </div>

      {/* Canvas */}
      <div className="flex-1 relative bg-muted/30 rounded-lg overflow-hidden">
        {subgraphLoading ? (
          <div className="absolute inset-0 flex items-center justify-center text-muted-foreground">
            Loading graph...
          </div>
        ) : elements.length > 0 ? (
          <CytoscapeComponent
            elements={elements as any}
            stylesheet={cytoscapeStylesheet}
            layout={{ name: layout, animate: false } as any}
            style={{ width: '100%', height: '100%' }}
            cy={(cy: cytoscape.Core) => {
              cyRef.current = cy;
              cy.on('tap', 'node', (evt: cytoscape.EventObject) => {
                setSelectedNode(evt.target.id());
              });
            }}
          />
        ) : (
          <div className="absolute inset-0 flex items-center justify-center text-muted-foreground">
            Select an edge table to visualize
          </div>
        )}
      </div>
    </div>
  );
}
