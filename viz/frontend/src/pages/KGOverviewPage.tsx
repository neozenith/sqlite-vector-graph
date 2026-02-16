import { Link } from 'react-router-dom'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { usePipeline } from '@/hooks/useKGPipeline'
import { toFunnelData, completedStageCount, entityReductionRatio } from '@/lib/transforms/kg-pipeline'
import { KG_STAGE_ROUTES } from '@/lib/constants'
import type { PipelineStage } from '@/lib/types'

interface FunnelItem {
  stage: number
  name: string
  percentage: number
}

export function KGOverviewPage() {
  const { data: pipeline } = usePipeline()
  const stages = pipeline?.stages ?? []
  const funnel = toFunnelData(stages)
  const completed = completedStageCount(stages)
  const reduction = entityReductionRatio(stages)

  return (
    <div className="max-w-3xl mx-auto space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">KG Pipeline ({completed}/7)</h2>
        <Link
          to="/kg/query"
          className="px-3 py-1.5 rounded-md text-xs font-medium bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
        >
          GraphRAG Query
        </Link>
      </div>

      {/* Stage List */}
      <div className="grid gap-3">
        {stages.map((s: PipelineStage) => (
          <Link key={s.stage} to={`/kg/${KG_STAGE_ROUTES[s.stage]}/`}>
            <Card
              className={`hover:border-primary/50 transition-colors cursor-pointer ${!s.available ? 'opacity-50' : ''}`}
            >
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex justify-between items-center">
                  <span>
                    {s.stage}. {s.name}
                  </span>
                  <Badge variant={s.available ? 'secondary' : 'outline'}>{s.count.toLocaleString()}</Badge>
                </CardTitle>
              </CardHeader>
              <CardContent className="text-xs text-muted-foreground">{s.description}</CardContent>
            </Card>
          </Link>
        ))}
      </div>

      {reduction > 0 && (
        <>
          <Separator />
          <Card>
            <CardContent className="pt-4 text-xs text-center">
              Entity reduction: <Badge variant="secondary">{reduction}%</Badge>
              <div className="text-muted-foreground mt-1">
                {stages.find((s: PipelineStage) => s.stage === 3)?.count.toLocaleString()} entities{' → '}
                {stages.find((s: PipelineStage) => s.stage === 6)?.count.toLocaleString()} nodes
              </div>
            </CardContent>
          </Card>
        </>
      )}

      {/* Funnel */}
      {funnel.length > 0 && (
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
      )}
    </div>
  )
}
