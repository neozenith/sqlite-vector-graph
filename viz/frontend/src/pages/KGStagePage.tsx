import { useState, useCallback, useRef } from 'react'
import { useParams, Link } from 'react-router-dom'
import cytoscape from 'cytoscape'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { EmbeddingCanvas } from '@/components/EmbeddingCanvas'
import { GraphCanvas } from '@/components/GraphCanvas'
import { GraphControls } from '@/components/GraphControls'
import { DEFAULT_FCOSE_PARAMS, toLayoutOptions } from '@/lib/fcose'
import type { FcoseParams } from '@/lib/fcose'
import { InfiniteList } from '@/components/InfiniteList'
import { EntityStageView } from '@/components/EntityStageView'
import { RelationStageView } from '@/components/RelationStageView'
import { Node2VecStageView } from '@/components/Node2VecStageView'
import { useStageDetail, useKGEmbeddings, useKGSubgraph, useInfiniteStageItems } from '@/hooks/useKGPipeline'
import { toCytoscapeElements } from '@/lib/transforms/cytoscape'
import { STAGE_NAMES, KG_STAGE_ROUTES } from '@/lib/constants'

/** Reverse map: slug → stage number */
const SLUG_TO_STAGE: Record<string, number> = Object.fromEntries(
  Object.entries(KG_STAGE_ROUTES).map(([num, slug]) => [slug, Number(num)]),
)

export function KGStagePage() {
  const { stageName } = useParams<{ stageName: string }>()
  const stageNum = SLUG_TO_STAGE[stageName ?? ''] ?? null

  const { data: stageDetail, isLoading: stageLoading } = useStageDetail(stageNum)
  const { data: embeddingData, isLoading: embLoading } = useKGEmbeddings(stageNum ?? 0)
  const { data: subgraphData, isLoading: graphLoading } = useKGSubgraph(stageNum ?? 0)

  const [filter, setFilter] = useState('')
  const [selectedEmbeddingId, setSelectedEmbeddingId] = useState<number | null>(null)
  const pageSize = 20

  // FCoSE params for stage 6 graph
  const [fcoseParams, setFcoseParams] = useState<FcoseParams>(DEFAULT_FCOSE_PARAMS)
  const cyRef = useRef<cytoscape.Core | null>(null)

  const runLayout = useCallback(() => {
    if (cyRef.current) {
      cyRef.current
        .layout({ name: 'fcose', animate: true, animationDuration: 300, ...toLayoutOptions(fcoseParams) } as any)
        .run()
    }
  }, [fcoseParams])

  // Generic infinite scroll for stages 1 and 5 (stages 3, 4, 7 get custom views)
  const useGenericItems = stageNum != null && [1, 5].includes(stageNum)
  const infiniteItems = useInfiniteStageItems(useGenericItems ? stageNum : null, pageSize, filter)

  const handleFilterChange = useCallback((q: string) => {
    setFilter(q)
  }, [])

  if (stageNum == null) {
    return (
      <div className="text-muted-foreground text-center py-8">
        Unknown stage.{' '}
        <Link to="/kg" className="underline">
          Back to pipeline overview
        </Link>
      </div>
    )
  }

  const title = STAGE_NAMES[stageNum] ?? `Stage ${stageNum}`
  const showEmbeddings = stageNum === 2 && (embLoading || embeddingData)
  // Stage 6 gets graph with controls; stage 7 gets Node2VecStageView
  const showGraph = stageNum === 6 && (graphLoading || subgraphData)

  const graphElements = subgraphData ? toCytoscapeElements(subgraphData.nodes, subgraphData.edges) : []

  const allItems = infiniteItems.data?.pages.flatMap((p) => p.items) ?? []
  const total = infiniteItems.data?.pages[0]?.total ?? 0

  return (
    <div className="max-w-4xl mx-auto space-y-4">
      <h2 className="text-lg font-semibold">
        {stageNum}. {title}
      </h2>

      {/* Stage Detail */}
      <Card>
        <CardHeader>
          <CardTitle>{title}</CardTitle>
        </CardHeader>
        <CardContent>
          {stageLoading ? (
            <div className="text-muted-foreground">Loading stage data...</div>
          ) : stageDetail ? (
            <StageContent stage={stageNum} detail={stageDetail} />
          ) : (
            <div className="text-muted-foreground">No data available for this stage.</div>
          )}
        </CardContent>
      </Card>

      {/* Embedding preview for stage 2 — 3D + pickable */}
      {showEmbeddings && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Embeddings Preview (3D)</CardTitle>
          </CardHeader>
          <CardContent>
            {embLoading ? (
              <div className="text-muted-foreground text-xs">Loading embeddings...</div>
            ) : embeddingData ? (
              <EmbeddingCanvas
                points={embeddingData.points}
                dimensions={3}
                onSelect={setSelectedEmbeddingId}
                selectedId={selectedEmbeddingId ?? undefined}
                className="h-[400px]"
              />
            ) : null}
          </CardContent>
        </Card>
      )}

      {/* Graph preview for stage 6 with FCoSE controls */}
      {showGraph && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Graph Preview</CardTitle>
          </CardHeader>
          <CardContent>
            {graphLoading ? (
              <div className="text-muted-foreground text-xs">Loading graph...</div>
            ) : graphElements.length > 0 ? (
              <div className="flex gap-4">
                <div className="flex-1 min-w-0">
                  <GraphCanvas
                    elements={graphElements}
                    layout="fcose"
                    layoutOptions={toLayoutOptions(fcoseParams)}
                    onNodeSelect={() => {}}
                    cyRef={cyRef}
                    className="h-[400px]"
                  />
                </div>
                <div className="w-48 shrink-0">
                  <GraphControls params={fcoseParams} onChange={setFcoseParams} onRunLayout={runLayout} />
                </div>
              </div>
            ) : null}
          </CardContent>
        </Card>
      )}

      {/* Stage 3: Entity extraction custom views */}
      {stageNum === 3 && <EntityStageView />}

      {/* Stage 4: Relation extraction interactive graph-table */}
      {stageNum === 4 && <RelationStageView />}

      {/* Stage 7: Node2Vec — embedding + graph + node table */}
      {stageNum === 7 && <Node2VecStageView />}

      {/* Generic infinite-scroll data browser for stages 1, 5 */}
      {useGenericItems && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Data Browser</CardTitle>
          </CardHeader>
          <CardContent>
            <InfiniteList
              items={allItems}
              total={total}
              hasNextPage={infiniteItems.hasNextPage}
              isFetchingNextPage={infiniteItems.isFetchingNextPage}
              isLoading={infiniteItems.isLoading}
              filter={filter}
              onFilterChange={handleFilterChange}
              onLoadMore={() => infiniteItems.fetchNextPage()}
            />
          </CardContent>
        </Card>
      )}
    </div>
  )
}

/** Render stage-specific content based on the detail data. */
function StageContent({ stage, detail }: { stage: number; detail: Record<string, unknown> }) {
  if (detail.available === false) {
    return <div className="text-muted-foreground text-sm">This stage has not been run yet.</div>
  }

  const count = typeof detail.count === 'number' ? detail.count : null
  const textLengths = detail.text_lengths as Record<string, number> | undefined
  const samples = Array.isArray(detail.samples) ? (detail.samples as Array<{ text_preview: string }>) : null

  type TypeCount = { type: string; count: number }
  const byType = Array.isArray(detail.by_type) ? (detail.by_type as TypeCount[]) : null

  const canonicalCount = typeof detail.canonical_count === 'number' ? detail.canonical_count : null
  const mergedCount = typeof detail.merged_count === 'number' ? detail.merged_count : null
  const mergeRatio = typeof detail.merge_ratio === 'number' ? detail.merge_ratio : null

  type ClusterInfo = { canonical: string; size: number; members: string[] }
  const largestClusters = Array.isArray(detail.largest_clusters) ? (detail.largest_clusters as ClusterInfo[]) : null

  const nodeCount = typeof detail.node_count === 'number' ? detail.node_count : null
  const edgeCount = typeof detail.edge_count === 'number' ? detail.edge_count : null
  const config = detail.config as Record<string, string> | undefined

  return (
    <div className="space-y-2 text-sm">
      {count !== null && (
        <div>
          Total: <Badge variant="secondary">{count.toLocaleString()}</Badge>
        </div>
      )}

      {stage === 1 && textLengths && (
        <div className="text-xs text-muted-foreground">
          Text lengths: min {textLengths.min}, max {textLengths.max}, mean {textLengths.mean}
        </div>
      )}
      {stage === 1 && samples && (
        <div className="space-y-1">
          {samples.map((s, i: number) => (
            <div key={i} className="text-xs p-2 bg-muted rounded break-words whitespace-pre-wrap">
              {s.text_preview}
            </div>
          ))}
        </div>
      )}

      {stage === 3 && byType && byType.length > 0 && (
        <div>
          <div className="text-xs font-medium mb-1">By Type:</div>
          <div className="flex flex-wrap gap-1">
            {byType.map((t) => (
              <Badge key={t.type} variant="outline">
                {t.type}: {t.count}
              </Badge>
            ))}
          </div>
        </div>
      )}

      {stage === 4 && byType && (
        <div className="flex flex-wrap gap-1">
          {byType.slice(0, 10).map((t) => (
            <Badge key={t.type} variant="outline">
              {t.type}: {t.count}
            </Badge>
          ))}
        </div>
      )}

      {stage === 5 && canonicalCount !== null && (
        <div className="space-y-1 text-xs">
          <div>Canonical entities: {canonicalCount.toLocaleString()}</div>
          {mergedCount !== null && mergeRatio !== null && (
            <div>
              Merged: {mergedCount.toLocaleString()} ({(mergeRatio * 100).toFixed(1)}%)
            </div>
          )}
          {largestClusters && (
            <div className="mt-2">
              <div className="font-medium">Largest clusters:</div>
              {largestClusters.map((c) => (
                <div key={c.canonical} className="p-1 bg-muted rounded mt-1">
                  <span className="font-medium">{c.canonical}</span> ({c.size} members)
                  <div className="text-muted-foreground">{c.members.join(', ')}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {stage === 6 && nodeCount !== null && (
        <div className="text-xs space-y-1">
          <div>
            Nodes: {nodeCount.toLocaleString()}, Edges: {(edgeCount ?? 0).toLocaleString()}
          </div>
        </div>
      )}

      {(stage === 2 || stage === 7) && config && (
        <div className="text-xs">
          <div className="font-medium mb-1">HNSW Config:</div>
          {Object.entries(config).map(([k, v]) => (
            <div key={k}>
              {k}: <Badge variant="outline">{v}</Badge>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
