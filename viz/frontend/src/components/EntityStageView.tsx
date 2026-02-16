/** Entity Extraction stage: By-Entity / By-Chunk toggle views with infinite scroll. */

import { useState, useEffect, useRef, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { useInfiniteEntitiesGrouped, useInfiniteEntitiesByChunk } from '@/hooks/useKGPipeline'
import type { GroupedEntity, ChunkEntities } from '@/lib/types'

export function EntityStageView() {
  const [tab, setTab] = useState('by-entity')
  const [entityFilter, setEntityFilter] = useState('')
  const [chunkFilter, setChunkFilter] = useState('')
  const pageSize = 20

  const grouped = useInfiniteEntitiesGrouped(pageSize, entityFilter)
  const byChunk = useInfiniteEntitiesByChunk(pageSize, chunkFilter)

  const allGrouped = grouped.data?.pages.flatMap((p) => p.items) ?? []
  const groupedTotal = grouped.data?.pages[0]?.total ?? 0

  const allChunks = byChunk.data?.pages.flatMap((p) => p.items) ?? []
  const chunksTotal = byChunk.data?.pages[0]?.total ?? 0

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">Entity Browser</CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs value={tab} onValueChange={setTab}>
          <TabsList>
            <TabsTrigger value="by-entity">By Entity</TabsTrigger>
            <TabsTrigger value="by-chunk">By Chunk</TabsTrigger>
          </TabsList>

          <TabsContent value="by-entity" className="mt-3">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Input
                  placeholder="Filter entities..."
                  value={entityFilter}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setEntityFilter(e.target.value)}
                  className="h-8 text-xs"
                />
                <span className="text-xs text-muted-foreground whitespace-nowrap">
                  {groupedTotal.toLocaleString()} groups
                </span>
              </div>

              {grouped.isLoading ? (
                <div className="text-xs text-muted-foreground py-4 text-center">Loading...</div>
              ) : allGrouped.length === 0 ? (
                <div className="text-xs text-muted-foreground py-4 text-center">No entities found</div>
              ) : (
                <EntityGroupedList
                  items={allGrouped}
                  hasNextPage={grouped.hasNextPage}
                  isFetchingNextPage={grouped.isFetchingNextPage}
                  onLoadMore={() => grouped.fetchNextPage()}
                />
              )}
            </div>
          </TabsContent>

          <TabsContent value="by-chunk" className="mt-3">
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Input
                  placeholder="Filter by entity name..."
                  value={chunkFilter}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setChunkFilter(e.target.value)}
                  className="h-8 text-xs"
                />
                <span className="text-xs text-muted-foreground whitespace-nowrap">
                  {chunksTotal.toLocaleString()} chunks
                </span>
              </div>

              {byChunk.isLoading ? (
                <div className="text-xs text-muted-foreground py-4 text-center">Loading...</div>
              ) : allChunks.length === 0 ? (
                <div className="text-xs text-muted-foreground py-4 text-center">No chunks found</div>
              ) : (
                <ChunkEntityList
                  items={allChunks}
                  hasNextPage={byChunk.hasNextPage}
                  isFetchingNextPage={byChunk.isFetchingNextPage}
                  onLoadMore={() => byChunk.fetchNextPage()}
                />
              )}
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}

function EntityGroupedList({
  items,
  hasNextPage,
  isFetchingNextPage,
  onLoadMore,
}: {
  items: GroupedEntity[]
  hasNextPage: boolean
  isFetchingNextPage: boolean
  onLoadMore: () => void
}) {
  const sentinelRef = useRef<HTMLDivElement>(null)
  const stableOnLoadMore = useCallback(() => onLoadMore(), [onLoadMore])

  useEffect(() => {
    const el = sentinelRef.current
    if (!el) return
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && hasNextPage && !isFetchingNextPage) {
          stableOnLoadMore()
        }
      },
      { threshold: 0.1 },
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [hasNextPage, isFetchingNextPage, stableOnLoadMore])

  return (
    <div className="space-y-1">
      {items.map((entity, i) => (
        <div key={`${entity.name}-${entity.entity_type}-${i}`} className="text-xs p-2 bg-muted rounded">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="font-medium">{entity.name}</span>
            <Badge variant="outline" className="text-[10px]">
              {entity.entity_type}
            </Badge>
            <Badge variant="secondary" className="text-[10px]">
              {entity.mention_count}x
            </Badge>
          </div>
          <div className="text-muted-foreground mt-1 break-words whitespace-pre-wrap">
            chunks: {entity.chunk_ids.join(', ')}
          </div>
        </div>
      ))}
      <div ref={sentinelRef} className="h-4" />
      {isFetchingNextPage && <div className="text-xs text-muted-foreground text-center py-2">Loading more...</div>}
    </div>
  )
}

function ChunkEntityList({
  items,
  hasNextPage,
  isFetchingNextPage,
  onLoadMore,
}: {
  items: ChunkEntities[]
  hasNextPage: boolean
  isFetchingNextPage: boolean
  onLoadMore: () => void
}) {
  const sentinelRef = useRef<HTMLDivElement>(null)
  const stableOnLoadMore = useCallback(() => onLoadMore(), [onLoadMore])

  useEffect(() => {
    const el = sentinelRef.current
    if (!el) return
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && hasNextPage && !isFetchingNextPage) {
          stableOnLoadMore()
        }
      },
      { threshold: 0.1 },
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [hasNextPage, isFetchingNextPage, stableOnLoadMore])

  return (
    <div className="space-y-2">
      {items.map((chunk) => (
        <div key={chunk.chunk_id} className="text-xs p-2 bg-muted rounded">
          <div className="flex items-center gap-2 mb-1">
            <span className="font-mono text-muted-foreground">chunk {chunk.chunk_id}</span>
            <Badge variant="secondary" className="text-[10px]">
              {chunk.entity_count} entities
            </Badge>
          </div>
          {chunk.text && <div className="text-muted-foreground mb-1 break-words whitespace-pre-wrap">{chunk.text}</div>}
          <div className="flex flex-wrap gap-1">
            {chunk.entities.map((e, j) => (
              <Badge key={`${e.name}-${j}`} variant="outline" className="text-[10px]">
                {e.name} ({e.entity_type})
              </Badge>
            ))}
          </div>
        </div>
      ))}
      <div ref={sentinelRef} className="h-4" />
      {isFetchingNextPage && <div className="text-xs text-muted-foreground text-center py-2">Loading more...</div>}
    </div>
  )
}
