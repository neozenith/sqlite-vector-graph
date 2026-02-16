/** Relation Extraction stage: interactive graph + table with bidirectional selection. */

import { useState, useRef, useCallback, useMemo } from 'react'
import cytoscape from 'cytoscape'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { GraphCanvas } from '@/components/GraphCanvas'
import { GraphControls } from '@/components/GraphControls'
import { DEFAULT_FCOSE_PARAMS, toLayoutOptions } from '@/lib/fcose'
import type { FcoseParams } from '@/lib/fcose'
import { useKGSubgraph } from '@/hooks/useKGPipeline'
import { toCytoscapeElements } from '@/lib/transforms/cytoscape'
import type { GraphEdge } from '@/lib/types'

export function RelationStageView() {
  const { data: subgraphData, isLoading } = useKGSubgraph(4)
  const cyRef = useRef<cytoscape.Core | null>(null)
  const [selectedEdgeIdx, setSelectedEdgeIdx] = useState<number | null>(null)
  const [tableFilter, setTableFilter] = useState('')
  const [fcoseParams, setFcoseParams] = useState<FcoseParams>(DEFAULT_FCOSE_PARAMS)

  const edges = useMemo(() => subgraphData?.edges ?? [], [subgraphData])
  const elements = subgraphData ? toCytoscapeElements(subgraphData.nodes, subgraphData.edges) : []

  const filteredEdges = useMemo(() => {
    if (!tableFilter) return edges
    const q = tableFilter.toLowerCase()
    return edges.filter(
      (e: GraphEdge) =>
        e.source.toLowerCase().includes(q) ||
        e.target.toLowerCase().includes(q) ||
        (e.rel_type && e.rel_type.toLowerCase().includes(q)),
    )
  }, [edges, tableFilter])

  const handleEdgeSelect = useCallback((edgeId: string) => {
    const match = edgeId.match(/^e-(\d+)$/)
    if (!match) return
    const idx = Number(match[1])
    setSelectedEdgeIdx(idx)
    const cy = cyRef.current
    if (cy) {
      cy.edges().removeClass('highlighted')
      cy.$(`#${edgeId}`).addClass('highlighted')
    }
  }, [])

  const handleRowClick = useCallback((idx: number) => {
    setSelectedEdgeIdx(idx)
    const cy = cyRef.current
    if (cy) {
      cy.edges().removeClass('highlighted')
      cy.$(`#e-${idx}`).addClass('highlighted')
      const edge = cy.$(`#e-${idx}`)
      if (edge.length > 0) {
        cy.animate({ center: { eles: edge }, duration: 300 })
      }
    }
  }, [])

  const runLayout = useCallback(() => {
    if (cyRef.current) {
      cyRef.current
        .layout({ name: 'fcose', animate: true, animationDuration: 300, ...toLayoutOptions(fcoseParams) } as any)
        .run()
    }
  }, [fcoseParams])

  if (isLoading) {
    return (
      <Card>
        <CardContent className="py-4">
          <div className="text-muted-foreground text-xs text-center">Loading relation graph...</div>
        </CardContent>
      </Card>
    )
  }

  if (elements.length === 0) {
    return (
      <Card>
        <CardContent className="py-4">
          <div className="text-muted-foreground text-xs text-center">No relation data available</div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-4">
      {/* Graph + controls side by side */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Relation Graph</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4">
            <div className="flex-1 min-w-0">
              <GraphCanvas
                elements={elements}
                layout="fcose"
                layoutOptions={toLayoutOptions(fcoseParams)}
                onEdgeSelect={handleEdgeSelect}
                onNodeSelect={() => {}}
                cyRef={cyRef}
                className="h-[400px]"
              />
            </div>
            <div className="w-48 shrink-0">
              <GraphControls params={fcoseParams} onChange={setFcoseParams} onRunLayout={runLayout} />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Table */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Relations ({edges.length})</CardTitle>
        </CardHeader>
        <CardContent>
          <Input
            placeholder="Filter relations..."
            value={tableFilter}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setTableFilter(e.target.value)}
            className="h-8 text-xs mb-2"
          />
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
                {filteredEdges.map((edge: GraphEdge, displayIdx: number) => {
                  const originalIdx = edges.indexOf(edge)
                  const isSelected = selectedEdgeIdx === originalIdx
                  return (
                    <tr
                      key={displayIdx}
                      className={`cursor-pointer transition-colors ${
                        isSelected ? 'bg-orange-100 dark:bg-orange-900/30' : 'hover:bg-muted/50'
                      }`}
                      onClick={() => handleRowClick(originalIdx)}
                    >
                      <td className="p-2 break-words">{edge.source}</td>
                      <td className="p-2 break-words text-muted-foreground">{edge.rel_type ?? '—'}</td>
                      <td className="p-2 break-words">{edge.target}</td>
                      <td className="p-2 text-right font-mono">{edge.weight.toFixed(2)}</td>
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
