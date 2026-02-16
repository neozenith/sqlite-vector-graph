import { Link } from 'react-router-dom'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { useIndexes } from '@/hooks/useVSSData'
import type { HnswIndexInfo } from '@/lib/types'

export function EmbeddingsIndexPage() {
  const { data: indexes, isLoading } = useIndexes()

  return (
    <div className="max-w-2xl mx-auto space-y-4">
      <h2 className="text-lg font-semibold">HNSW Indexes</h2>
      <p className="text-sm text-muted-foreground">Select an index to explore its embeddings.</p>

      {isLoading ? (
        <div className="text-muted-foreground">Loading indexes...</div>
      ) : !indexes?.length ? (
        <div className="text-muted-foreground">No HNSW indexes found in this database.</div>
      ) : (
        <div className="grid gap-3">
          {indexes.map((idx: HnswIndexInfo) => (
            <Link key={idx.name} to={`/embeddings/${encodeURIComponent(idx.name)}`}>
              <Card className="hover:border-primary/50 transition-colors cursor-pointer">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">{idx.name}</CardTitle>
                </CardHeader>
                <CardContent className="flex gap-2">
                  <Badge variant="secondary">{idx.node_count} points</Badge>
                  <Badge variant="outline">{idx.dimensions}D</Badge>
                  <Badge variant="outline">{idx.metric}</Badge>
                  <Badge variant="outline">M={idx.m}</Badge>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}
