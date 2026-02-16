/** Node2Vec stage: 3D embedding + graph + node table with cross-selection. */

import { useState, useRef, useCallback, useMemo } from 'react'
import cytoscape from 'cytoscape'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { EmbeddingCanvas } from '@/components/EmbeddingCanvas'
import { GraphCanvas } from '@/components/GraphCanvas'
import { GraphControls } from '@/components/GraphControls'
import { DEFAULT_FCOSE_PARAMS, toLayoutOptions } from '@/lib/fcose'
import type { FcoseParams } from '@/lib/fcose'
import { useKGEmbeddings, useKGSubgraph } from '@/hooks/useKGPipeline'
import { toCytoscapeElements } from '@/lib/transforms/cytoscape'
import type { GraphNode, GraphEdge, EmbeddingPoint } from '@/lib/types'

export function Node2VecStageView() {
  const { data: embeddingData, isLoading: embLoading } = useKGEmbeddings(7)
  const { data: subgraphData, isLoading: graphLoading } = useKGSubgraph(6)

  const [selectedNodeName, setSelectedNodeName] = useState<string | null>(null)
  const [tableFilter, setTableFilter] = useState('')
  const [fcoseParams, setFcoseParams] = useState<FcoseParams>(DEFAULT_FCOSE_PARAMS)
  const cyRef = useRef<cytoscape.Core | null>(null)

  // Build point ID → node name lookup from embedding labels
  const pointIdToName = useMemo(() => {
    const map = new Map<number, string>()
    if (embeddingData?.points) {
      for (const p of embeddingData.points) {
        if (p.label) map.set(p.id, p.label)
      }
    }
    return map
  }, [embeddingData])

  const nodes = useMemo(() => subgraphData?.nodes ?? [], [subgraphData])
  const edges = useMemo(() => subgraphData?.edges ?? [], [subgraphData])
  const elements = subgraphData ? toCytoscapeElements(subgraphData.nodes, subgraphData.edges) : []

  // Edges connected to the selected node
  const connectedEdges = useMemo(() => {
    if (!selectedNodeName) return []
    return edges.filter((e: GraphEdge) => e.source === selectedNodeName || e.target === selectedNodeName)
  }, [edges, selectedNodeName])

  // Filtered node list for the table
  const filteredNodes = useMemo(() => {
    if (!tableFilter) return nodes
    const q = tableFilter.toLowerCase()
    return nodes.filter(
      (n: GraphNode) => n.label.toLowerCase().includes(q) || (n.entity_type && n.entity_type.toLowerCase().includes(q)),
    )
  }, [nodes, tableFilter])

  // Find the embedding point ID for the selected node (for highlighting in canvas)
  const selectedEmbeddingId = useMemo(() => {
    if (!selectedNodeName) return undefined
    for (const [id, name] of pointIdToName) {
      if (name === selectedNodeName) return id
    }
    return undefined
  }, [selectedNodeName, pointIdToName])

  const highlightGraphNode = useCallback((nodeName: string) => {
    const cy = cyRef.current
    if (!cy) return
    // Clear previous highlights
    cy.nodes().removeClass('selected')
    cy.edges().removeClass('highlighted')
    // Highlight selected node and its edges
    const node = cy.$(`#${CSS.escape(nodeName)}`)
    if (node.length > 0) {
      node.addClass('selected')
      node.connectedEdges().addClass('highlighted')
      cy.animate({ center: { eles: node }, duration: 300 })
    }
  }, [])

  const handleEmbeddingSelect = useCallback(
    (pointId: number) => {
      const name = pointIdToName.get(pointId)
      if (name) {
        setSelectedNodeName(name)
        highlightGraphNode(name)
      }
    },
    [pointIdToName, highlightGraphNode],
  )

  const handleGraphNodeSelect = useCallback(
    (nodeId: string) => {
      setSelectedNodeName(nodeId)
      highlightGraphNode(nodeId)
    },
    [highlightGraphNode],
  )

  const handleTableRowClick = useCallback(
    (nodeName: string) => {
      setSelectedNodeName(nodeName)
      highlightGraphNode(nodeName)
    },
    [highlightGraphNode],
  )

  const runLayout = useCallback(() => {
    if (cyRef.current) {
      cyRef.current
        .layout({ name: 'fcose', animate: true, animationDuration: 300, ...toLayoutOptions(fcoseParams) } as any)
        .run()
    }
  }, [fcoseParams])

  const isLoading = embLoading || graphLoading

  if (isLoading) {
    return (
      <Card>
        <CardContent className="py-4">
          <div className="text-muted-foreground text-xs text-center">Loading Node2Vec data...</div>
        </CardContent>
      </Card>
    )
  }

  const hasEmbeddings = embeddingData && embeddingData.points.length > 0
  const hasGraph = elements.length > 0

  if (!hasEmbeddings && !hasGraph) {
    return (
      <Card>
        <CardContent className="py-4">
          <div className="text-muted-foreground text-xs text-center">
            No Node2Vec data available. Run the pipeline first.
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-4">
      {/* Top: Embedding + Graph side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* 3D Embedding canvas */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">
              Node2Vec Embeddings (3D)
              {embeddingData && (
                <Badge variant="secondary" className="ml-2 text-[10px]">
                  {embeddingData.count} points
                </Badge>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {hasEmbeddings ? (
              <EmbeddingCanvas
                points={embeddingData.points as EmbeddingPoint[]}
                dimensions={3}
                onSelect={handleEmbeddingSelect}
                selectedId={selectedEmbeddingId}
                className="h-[350px]"
              />
            ) : (
              <div className="h-[350px] flex items-center justify-center text-muted-foreground text-xs">
                No embedding data
              </div>
            )}
          </CardContent>
        </Card>

        {/* Graph canvas + controls */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Knowledge Graph</CardTitle>
          </CardHeader>
          <CardContent>
            {hasGraph ? (
              <div className="flex gap-3">
                <div className="flex-1 min-w-0">
                  <GraphCanvas
                    elements={elements}
                    layout="fcose"
                    layoutOptions={toLayoutOptions(fcoseParams)}
                    onNodeSelect={handleGraphNodeSelect}
                    cyRef={cyRef}
                    className="h-[350px]"
                  />
                </div>
                <div className="w-44 shrink-0">
                  <GraphControls params={fcoseParams} onChange={setFcoseParams} onRunLayout={runLayout} />
                </div>
              </div>
            ) : (
              <div className="h-[350px] flex items-center justify-center text-muted-foreground text-xs">
                No graph data
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Selected node detail */}
      {selectedNodeName && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">
              Selected: <span className="text-primary">{selectedNodeName}</span>
              <Badge variant="outline" className="ml-2 text-[10px]">
                {connectedEdges.length} edges
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {connectedEdges.length > 0 ? (
              <div className="border rounded overflow-hidden">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="bg-muted">
                      <th className="text-left p-2 font-medium">Source</th>
                      <th className="text-left p-2 font-medium">Relation</th>
                      <th className="text-left p-2 font-medium">Target</th>
                      <th className="text-right p-2 font-medium">Weight</th>
                    </tr>
                  </thead>
                  <tbody>
                    {connectedEdges.map((edge: GraphEdge, i: number) => (
                      <tr key={i} className="hover:bg-muted/50">
                        <td className={`p-2 ${edge.source === selectedNodeName ? 'font-medium text-primary' : ''}`}>
                          {edge.source}
                        </td>
                        <td className="p-2 text-muted-foreground">{edge.rel_type ?? '—'}</td>
                        <td className={`p-2 ${edge.target === selectedNodeName ? 'font-medium text-primary' : ''}`}>
                          {edge.target}
                        </td>
                        <td className="p-2 text-right font-mono">{edge.weight.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="text-muted-foreground text-xs">No edges connected to this node.</div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Node table */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Nodes ({nodes.length})</CardTitle>
        </CardHeader>
        <CardContent>
          <Input
            placeholder="Filter nodes..."
            value={tableFilter}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setTableFilter(e.target.value)}
            className="h-8 text-xs mb-2"
          />
          <div className="border rounded overflow-hidden max-h-[400px] overflow-y-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0">
                <tr className="bg-muted">
                  <th className="text-left p-2 font-medium">Node</th>
                  <th className="text-left p-2 font-medium">Type</th>
                  <th className="text-right p-2 font-medium">Mentions</th>
                </tr>
              </thead>
              <tbody>
                {filteredNodes.map((node: GraphNode) => {
                  const isSelected = selectedNodeName === node.id
                  return (
                    <tr
                      key={node.id}
                      className={`cursor-pointer transition-colors ${
                        isSelected ? 'bg-orange-100 dark:bg-orange-900/30' : 'hover:bg-muted/50'
                      }`}
                      onClick={() => handleTableRowClick(node.id)}
                    >
                      <td className="p-2 break-words">{node.label}</td>
                      <td className="p-2">
                        {node.entity_type && (
                          <Badge variant="outline" className="text-[10px]">
                            {node.entity_type}
                          </Badge>
                        )}
                      </td>
                      <td className="p-2 text-right font-mono">{node.mention_count ?? '—'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
