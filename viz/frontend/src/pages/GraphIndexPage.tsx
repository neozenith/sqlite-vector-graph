import { Link } from 'react-router-dom'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { useGraphs } from '@/hooks/useGraphData'
import type { EdgeTableInfo } from '@/lib/types'

export function GraphIndexPage() {
  const { data: graphs, isLoading } = useGraphs()

  return (
    <div className="max-w-2xl mx-auto space-y-4">
      <h2 className="text-lg font-semibold">Edge Tables</h2>
      <p className="text-sm text-muted-foreground">Select an edge table to explore its graph.</p>

      {isLoading ? (
        <div className="text-muted-foreground">Loading edge tables...</div>
      ) : !graphs?.length ? (
        <div className="text-muted-foreground">No edge tables found in this database.</div>
      ) : (
        <div className="grid gap-3">
          {graphs.map((g: EdgeTableInfo) => (
            <Link key={g.table_name} to={`/graph/${encodeURIComponent(g.table_name)}`}>
              <Card className="hover:border-primary/50 transition-colors cursor-pointer">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">{g.table_name}</CardTitle>
                </CardHeader>
                <CardContent className="flex gap-2">
                  <Badge variant="secondary">{g.edge_count} edges</Badge>
                  <Badge variant="outline">
                    {g.src_col} → {g.dst_col}
                  </Badge>
                  {g.weight_col && <Badge variant="outline">weight: {g.weight_col}</Badge>}
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}
