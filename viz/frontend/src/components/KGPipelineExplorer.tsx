import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { useAppStore } from '@/stores/app-store';
import {
  usePipeline, useStageDetail, useGraphRAGQuery, useKGExplorer,
  useKGEmbeddings, useKGSubgraph, useStageItems,
} from '@/hooks/useKGPipeline';
import { toFunnelData, completedStageCount, entityReductionRatio } from '@/lib/transforms/kg-pipeline';
import { STAGE_NAMES } from '@/lib/constants';
import { EmbeddingsMiniViz } from './EmbeddingsMiniViz';
import { GraphMiniViz } from './GraphMiniViz';
import { PaginatedList } from './PaginatedList';
import type { PipelineStage, StageItemsResponse } from '@/lib/types';

interface FunnelItem {
  stage: number;
  name: string;
  percentage: number;
}

export function KGPipelineExplorer() {
  const { activeStage, setActiveStage } = useAppStore();
  const { queryText, setQueryText } = useKGExplorer();

  const { data: pipeline } = usePipeline();
  const { data: stageDetail, isLoading: stageLoading } = useStageDetail(activeStage);
  const graphRAG = useGraphRAGQuery();

  const stages = pipeline?.stages ?? [];
  const funnel = toFunnelData(stages);
  const completed = completedStageCount(stages);
  const reduction = entityReductionRatio(stages);

  const handleQuery = () => {
    if (queryText.trim()) {
      graphRAG.mutate({ query: queryText.trim() });
    }
  };

  return (
    <div className="flex h-full gap-4">
      {/* Stage Navigator */}
      <div className="w-56 flex flex-col gap-2 shrink-0">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Pipeline ({completed}/7)</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1">
            {stages.map((s: PipelineStage) => (
              <button
                key={s.stage}
                className={`w-full text-left px-2 py-1.5 rounded text-xs transition-colors ${
                  activeStage === s.stage
                    ? 'bg-primary text-primary-foreground'
                    : s.available
                      ? 'hover:bg-muted'
                      : 'text-muted-foreground opacity-50'
                }`}
                onClick={() => setActiveStage(s.stage)}
              >
                <div className="flex justify-between items-center">
                  <span>{s.stage}. {s.name}</span>
                  <Badge
                    variant={s.available ? 'secondary' : 'outline'}
                    className="text-[10px]"
                  >
                    {s.count.toLocaleString()}
                  </Badge>
                </div>
              </button>
            ))}
          </CardContent>
        </Card>

        {reduction > 0 && (
          <Card>
            <CardContent className="pt-4 text-xs text-center">
              Entity reduction: <Badge variant="secondary">{reduction}%</Badge>
              <div className="text-muted-foreground mt-1">
                {stages.find((s: PipelineStage) => s.stage === 3)?.count.toLocaleString()} entities
                {' → '}
                {stages.find((s: PipelineStage) => s.stage === 6)?.count.toLocaleString()} nodes
              </div>
            </CardContent>
          </Card>
        )}

        <Separator />

        {/* Funnel visualization */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Data Funnel</CardTitle>
          </CardHeader>
          <CardContent>
            {funnel.map((f: FunnelItem) => (
              <div key={f.stage} className="mb-1">
                <div className="text-[10px] text-muted-foreground">{f.name}</div>
                <div className="h-3 bg-muted rounded overflow-hidden">
                  <div
                    className="h-full bg-primary/60 rounded transition-all"
                    style={{ width: `${Math.max(f.percentage, 2)}%` }}
                  />
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      {/* Stage Detail */}
      <div className="flex-1 overflow-y-auto">
        <Card className="mb-4">
          <CardHeader>
            <CardTitle>{STAGE_NAMES[activeStage] ?? `Stage ${activeStage}`}</CardTitle>
          </CardHeader>
          <CardContent>
            {stageLoading ? (
              <div className="text-muted-foreground">Loading stage data...</div>
            ) : stageDetail ? (
              <StageContent stage={activeStage} detail={stageDetail} />
            ) : (
              <div className="text-muted-foreground">No data available for this stage.</div>
            )}
          </CardContent>
        </Card>

        {/* Inline visualizations for relevant stages */}
        <StageViz stage={activeStage} />

        {/* Paginated lists for data stages */}
        <StageItems stage={activeStage} />

        {/* GraphRAG Query */}
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">GraphRAG Query</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex gap-2 mb-3">
              <Input
                placeholder="e.g., division of labor and trade"
                value={queryText}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => setQueryText(e.target.value)}
                onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => e.key === 'Enter' && handleQuery()}
              />
              <Button
                onClick={handleQuery}
                disabled={graphRAG.isPending || !queryText.trim()}
              >
                {graphRAG.isPending ? 'Querying...' : 'Query'}
              </Button>
            </div>

            {graphRAG.data && (
              <div className="space-y-2 text-xs">
                {Object.entries(graphRAG.data.stages).map(([key, value]) => (
                  <div key={key} className="p-2 bg-muted rounded">
                    <div className="font-medium mb-1">{key}</div>
                    <pre className="text-[10px] overflow-x-auto whitespace-pre-wrap">
                      {JSON.stringify(value, null, 2)}
                    </pre>
                  </div>
                ))}
              </div>
            )}

            {graphRAG.error && (
              <div className="text-destructive text-xs mt-2">
                Query failed: {(graphRAG.error as Error).message}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

/** Inline visualizations for embedding/graph stages. */
function StageViz({ stage }: { stage: number }) {
  const { data: embeddingData, isLoading: embLoading } = useKGEmbeddings(stage);
  const { data: subgraphData, isLoading: graphLoading } = useKGSubgraph(stage);

  // Embedding stages: 2 and 7
  if ((stage === 2 || stage === 7) && (embLoading || embeddingData)) {
    return (
      <Card className="mb-4">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Embeddings Preview</CardTitle>
        </CardHeader>
        <CardContent>
          {embLoading ? (
            <div className="text-muted-foreground text-xs">Loading embeddings...</div>
          ) : embeddingData ? (
            <EmbeddingsMiniViz points={embeddingData.points} />
          ) : null}
        </CardContent>
      </Card>
    );
  }

  // Graph stages: 4 and 6
  if ((stage === 4 || stage === 6) && (graphLoading || subgraphData)) {
    return (
      <Card className="mb-4">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Graph Preview</CardTitle>
        </CardHeader>
        <CardContent>
          {graphLoading ? (
            <div className="text-muted-foreground text-xs">Loading graph...</div>
          ) : subgraphData ? (
            <GraphMiniViz nodes={subgraphData.nodes} edges={subgraphData.edges} />
          ) : null}
        </CardContent>
      </Card>
    );
  }

  return null;
}

/** Paginated searchable lists for data stages (1, 3, 4, 5). */
function StageItems({ stage }: { stage: number }) {
  const [page, setPage] = useState(1);
  const [filter, setFilter] = useState('');
  const pageSize = 20;

  const { data, isLoading } = useStageItems(
    [1, 3, 4, 5].includes(stage) ? stage : null,
    page,
    pageSize,
    filter,
  );

  if (![1, 3, 4, 5].includes(stage)) return null;

  const itemsData: StageItemsResponse | undefined = data;

  return (
    <Card className="mb-4">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">Data Browser</CardTitle>
      </CardHeader>
      <CardContent>
        <PaginatedList
          items={itemsData?.items ?? []}
          total={itemsData?.total ?? 0}
          page={page}
          pageSize={pageSize}
          isLoading={isLoading}
          filter={filter}
          onFilterChange={(q) => { setFilter(q); setPage(1); }}
          onPageChange={setPage}
        />
      </CardContent>
    </Card>
  );
}

/** Render stage-specific content based on the detail data. */
function StageContent({ stage, detail }: { stage: number; detail: Record<string, unknown> }) {
  if (detail.available === false) {
    return <div className="text-muted-foreground text-sm">This stage has not been run yet.</div>;
  }

  // Extract typed values from the untyped detail record to satisfy TS 5.9 strict JSX checks.
  const count = typeof detail.count === 'number' ? detail.count : null;
  const textLengths = detail.text_lengths as Record<string, number> | undefined;
  const samples = Array.isArray(detail.samples) ? detail.samples as Array<{ text_preview: string }> : null;

  type TypeCount = { type: string; count: number };
  const byType = Array.isArray(detail.by_type) ? detail.by_type as TypeCount[] : null;

  const canonicalCount = typeof detail.canonical_count === 'number' ? detail.canonical_count : null;
  const mergedCount = typeof detail.merged_count === 'number' ? detail.merged_count : null;
  const mergeRatio = typeof detail.merge_ratio === 'number' ? detail.merge_ratio : null;

  type ClusterInfo = { canonical: string; size: number; members: string[] };
  const largestClusters = Array.isArray(detail.largest_clusters) ? detail.largest_clusters as ClusterInfo[] : null;

  const nodeCount = typeof detail.node_count === 'number' ? detail.node_count : null;
  const edgeCount = typeof detail.edge_count === 'number' ? detail.edge_count : null;
  const config = detail.config as Record<string, string> | undefined;

  return (
    <div className="space-y-2 text-sm">
      {/* Count */}
      {count !== null && (
        <div>Total: <Badge variant="secondary">{count.toLocaleString()}</Badge></div>
      )}

      {/* Stage 1: Chunks */}
      {stage === 1 && textLengths && (
        <div className="text-xs text-muted-foreground">
          Text lengths: min {textLengths.min}, max {textLengths.max}, mean {textLengths.mean}
        </div>
      )}
      {stage === 1 && samples && (
        <div className="space-y-1">
          {samples.map((s, i: number) => (
            <div key={i} className="text-xs p-2 bg-muted rounded truncate">{s.text_preview}</div>
          ))}
        </div>
      )}

      {/* Stage 3: Entities by type */}
      {stage === 3 && byType && byType.length > 0 && (
        <div>
          <div className="text-xs font-medium mb-1">By Type:</div>
          <div className="flex flex-wrap gap-1">
            {byType.map((t) => (
              <Badge key={t.type} variant="outline">{t.type}: {t.count}</Badge>
            ))}
          </div>
        </div>
      )}

      {/* Stage 4: Relations */}
      {stage === 4 && byType && (
        <div className="flex flex-wrap gap-1">
          {byType.slice(0, 10).map((t) => (
            <Badge key={t.type} variant="outline">{t.type}: {t.count}</Badge>
          ))}
        </div>
      )}

      {/* Stage 5: Resolution */}
      {stage === 5 && canonicalCount !== null && (
        <div className="space-y-1 text-xs">
          <div>Canonical entities: {canonicalCount.toLocaleString()}</div>
          {mergedCount !== null && mergeRatio !== null && (
            <div>Merged: {mergedCount.toLocaleString()} ({(mergeRatio * 100).toFixed(1)}%)</div>
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

      {/* Stage 6: Graph */}
      {stage === 6 && nodeCount !== null && (
        <div className="text-xs space-y-1">
          <div>Nodes: {nodeCount.toLocaleString()}, Edges: {(edgeCount ?? 0).toLocaleString()}</div>
        </div>
      )}

      {/* Stage 2 & 7: Config */}
      {(stage === 2 || stage === 7) && config && (
        <div className="text-xs">
          <div className="font-medium mb-1">HNSW Config:</div>
          {Object.entries(config).map(([k, v]) => (
            <div key={k}>{k}: <Badge variant="outline">{v}</Badge></div>
          ))}
        </div>
      )}
    </div>
  );
}
