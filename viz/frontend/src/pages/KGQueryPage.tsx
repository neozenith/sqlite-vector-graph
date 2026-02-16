/** GraphRAG Query page: search bar + side-by-side embedding/graph + result cards. */

import { useState, useRef, useCallback, useMemo } from 'react'
import type cytoscape from 'cytoscape'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { EmbeddingCanvas } from '@/components/EmbeddingCanvas'
import { GraphCanvas } from '@/components/GraphCanvas'
import { GraphControls } from '@/components/GraphControls'
import { DEFAULT_FCOSE_PARAMS, toLayoutOptions } from '@/lib/fcose'
import type { FcoseParams } from '@/lib/fcose'
import { useGraphRAGQuery, useKGEmbeddings, useKGSubgraph } from '@/hooks/useKGPipeline'
import { toCytoscapeElements, highlightBfsNodes } from '@/lib/transforms/cytoscape'
import type { SearchNeighbor, GraphRAGResult } from '@/lib/types'

/** Extract chunk IDs from GraphRAG results as SearchNeighbor-compatible objects. */
function extractChunkNeighbors(result: GraphRAGResult): SearchNeighbor[] {
  const fts = result.stages['1_fts_chunks'] as { chunks?: Array<{ chunk_id: number; text_preview?: string }> }
  if (!fts?.chunks) return []
  return fts.chunks.map((c, i) => ({
    id: c.chunk_id,
    distance: i * 0.1, // Rank-based pseudo-distance
    label: c.text_preview ?? `Chunk ${c.chunk_id}`,
    metadata: {},
  }))
}

/** Extract all entity names from GraphRAG results across all stages. */
function extractEntityNames(result: GraphRAGResult): Set<string> {
  const names = new Set<string>()

  const seeds = result.stages['2_seed_entities'] as { entities?: string[] }
  if (seeds?.entities) seeds.entities.forEach((e) => names.add(e))

  const expansion = result.stages['3_bfs_expansion'] as { newly_discovered?: string[] }
  if (expansion?.newly_discovered) expansion.newly_discovered.forEach((e) => names.add(e))

  const centrality = result.stages['4_centrality'] as { top_bridges?: Array<{ node: string }> }
  if (centrality?.top_bridges) centrality.top_bridges.forEach((b) => names.add(b.node))

  const leiden = result.stages['5_leiden_communities'] as { newly_added?: string[] }
  if (leiden?.newly_added) leiden.newly_added.forEach((e) => names.add(e))

  const n2v = result.stages['6_node2vec'] as { newly_added?: string[] }
  if (n2v?.newly_added) n2v.newly_added.forEach((e) => names.add(e))

  return names
}

/** Extract result passages from assembly stage. */
function extractPassages(result: GraphRAGResult): Array<{ chunk_id: number; text_preview: string; entity: string }> {
  const assembly = result.stages['7_assembly'] as {
    top_passages?: Array<{ chunk_id: number; text_preview: string; entity: string }>
  }
  return assembly?.top_passages ?? []
}

/** Extract stage summary counts for the progress badges. */
function extractStageSummary(result: GraphRAGResult): Array<{ label: string; count: number }> {
  const stages = result.stages
  const items: Array<{ label: string; count: number }> = []

  const fts = stages['1_fts_chunks'] as { count?: number }
  if (fts?.count != null) items.push({ label: 'FTS chunks', count: fts.count })

  const seeds = stages['2_seed_entities'] as { count?: number }
  if (seeds?.count != null) items.push({ label: 'Seed entities', count: seeds.count })

  const exp = stages['3_bfs_expansion'] as { expanded_count?: number }
  if (exp?.expanded_count != null) items.push({ label: 'Expanded', count: exp.expanded_count })

  const cent = stages['4_centrality'] as { count?: number }
  if (cent?.count != null) items.push({ label: 'Centrality', count: cent.count })

  const leiden = stages['5_leiden_communities'] as { community_entities?: number }
  if (leiden?.community_entities != null) items.push({ label: 'Communities', count: leiden.community_entities })

  const assembly = stages['7_assembly'] as { total_passages?: number }
  if (assembly?.total_passages != null) items.push({ label: 'Passages', count: assembly.total_passages })

  return items
}

export function KGQueryPage() {
  const [queryText, setQueryText] = useState('')
  const graphRAG = useGraphRAGQuery()

  // Pre-load embedding and graph data
  const { data: embeddingData, isLoading: embLoading } = useKGEmbeddings(2)
  const { data: subgraphData, isLoading: graphLoading } = useKGSubgraph(6)

  const [fcoseParams, setFcoseParams] = useState<FcoseParams>(DEFAULT_FCOSE_PARAMS)
  const cyRef = useRef<cytoscape.Core | null>(null)

  const handleQuery = useCallback(() => {
    if (queryText.trim()) {
      graphRAG.mutate({ query: queryText.trim() })
    }
  }, [queryText, graphRAG])

  const runLayout = useCallback(() => {
    if (cyRef.current) {
      cyRef.current
        .layout({ name: 'fcose', animate: true, animationDuration: 300, ...toLayoutOptions(fcoseParams) } as any)
        .run()
    }
  }, [fcoseParams])

  // Derived data from GraphRAG results
  const chunkNeighbors = useMemo(
    () => (graphRAG.data ? extractChunkNeighbors(graphRAG.data) : undefined),
    [graphRAG.data],
  )

  const entityNames = useMemo(
    () => (graphRAG.data ? extractEntityNames(graphRAG.data) : new Set<string>()),
    [graphRAG.data],
  )

  const passages = useMemo(() => (graphRAG.data ? extractPassages(graphRAG.data) : []), [graphRAG.data])

  const stageSummary = useMemo(() => (graphRAG.data ? extractStageSummary(graphRAG.data) : []), [graphRAG.data])

  // Build Cytoscape elements with highlighted entities
  const graphElements = useMemo(() => {
    if (!subgraphData) return []
    const base = toCytoscapeElements(subgraphData.nodes, subgraphData.edges)
    if (entityNames.size === 0) return base
    return highlightBfsNodes(base, entityNames)
  }, [subgraphData, entityNames])

  const hasEmbeddings = !embLoading && embeddingData && embeddingData.points.length > 0
  const hasGraph = !graphLoading && graphElements.length > 0
  const hasResults = graphRAG.data != null

  return (
    <div className="max-w-5xl mx-auto space-y-4">
      {/* Search bar */}
      <div className="flex gap-2">
        <Input
          placeholder="Search the knowledge graph... e.g., division of labor and trade"
          value={queryText}
          onChange={(e: React.ChangeEvent<HTMLInputElement>) => setQueryText(e.target.value)}
          onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => e.key === 'Enter' && handleQuery()}
          className="flex-1"
        />
        <Button onClick={handleQuery} disabled={graphRAG.isPending || !queryText.trim()}>
          {graphRAG.isPending ? 'Searching...' : 'Search'}
        </Button>
      </div>

      {/* Error display */}
      {graphRAG.error && (
        <div className="text-destructive text-sm p-3 bg-destructive/10 rounded">
          Query failed: {(graphRAG.error as Error).message}
        </div>
      )}

      {/* Pipeline stage progress badges */}
      {hasResults && stageSummary.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {stageSummary.map((s) => (
            <Badge key={s.label} variant="secondary" className="text-xs">
              {s.label}: {s.count}
            </Badge>
          ))}
        </div>
      )}

      {/* Side-by-side: 3D Embeddings + Knowledge Graph */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Left: 3D Embedding Space */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center justify-between">
              Embedding Space (3D)
              {embeddingData && (
                <span className="text-xs text-muted-foreground font-normal">
                  {embeddingData.count} points
                  {chunkNeighbors && chunkNeighbors.length > 0 && <> · {chunkNeighbors.length} matched</>}
                </span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {embLoading ? (
              <div className="h-[400px] flex items-center justify-center text-muted-foreground text-xs">
                Loading embeddings...
              </div>
            ) : hasEmbeddings ? (
              <EmbeddingCanvas
                points={embeddingData.points}
                searchResults={chunkNeighbors}
                dimensions={3}
                className="h-[400px]"
              />
            ) : (
              <div className="h-[400px] flex items-center justify-center text-muted-foreground text-xs">
                No embedding data. Run the pipeline first.
              </div>
            )}
          </CardContent>
        </Card>

        {/* Right: Knowledge Graph + controls */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm flex items-center justify-between">
              Knowledge Graph
              {subgraphData && (
                <span className="text-xs text-muted-foreground font-normal">
                  {subgraphData.node_count} nodes
                  {entityNames.size > 0 && <> · {entityNames.size} highlighted</>}
                </span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {graphLoading ? (
              <div className="h-[400px] flex items-center justify-center text-muted-foreground text-xs">
                Loading graph...
              </div>
            ) : hasGraph ? (
              <div className="flex gap-3">
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
                <div className="w-44 shrink-0">
                  <GraphControls params={fcoseParams} onChange={setFcoseParams} onRunLayout={runLayout} />
                </div>
              </div>
            ) : (
              <div className="h-[400px] flex items-center justify-center text-muted-foreground text-xs">
                No graph data. Run the pipeline first.
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Search Result Cards */}
      {passages.length > 0 && (
        <div>
          <h3 className="text-sm font-medium text-muted-foreground mb-3">Search Results ({passages.length})</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {passages.map((p, i) => (
              <Card key={i} className="text-xs">
                <CardContent className="p-3 space-y-2">
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="text-[10px] shrink-0">
                      Chunk #{p.chunk_id}
                    </Badge>
                    <Badge variant="secondary" className="text-[10px] truncate">
                      {p.entity}
                    </Badge>
                  </div>
                  <p className="text-muted-foreground break-words whitespace-pre-wrap leading-relaxed">
                    {p.text_preview}
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* No results message */}
      {hasResults && passages.length === 0 && (
        <div className="text-muted-foreground text-sm text-center py-8">
          No passages found for this query. Try different search terms.
        </div>
      )}

      {/* Prompt before first search */}
      {!hasResults && !graphRAG.isPending && (
        <div className="text-muted-foreground text-sm text-center py-8">
          Enter a search query to explore the knowledge graph.
        </div>
      )}
    </div>
  )
}
