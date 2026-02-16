/** KG Pipeline data transforms. */

import type { PipelineStage } from '../types'

export interface FunnelDatum {
  stage: number
  name: string
  count: number
  percentage: number
}

/**
 * Convert pipeline stages to funnel chart data.
 * Normalizes counts as percentage of the max count.
 */
export function toFunnelData(stages: PipelineStage[]): FunnelDatum[] {
  const maxCount = Math.max(...stages.map((s) => s.count), 1)
  return stages
    .filter((s) => s.available)
    .map((s) => ({
      stage: s.stage,
      name: s.name,
      count: s.count,
      percentage: Math.round((s.count / maxCount) * 100),
    }))
}

/**
 * Count completed (available) stages.
 */
export function completedStageCount(stages: PipelineStage[]): number {
  return stages.filter((s) => s.available).length
}

/**
 * Get the entity reduction ratio from stage 3 (raw entities) to stage 6 (nodes).
 */
export function entityReductionRatio(stages: PipelineStage[]): number {
  const entityStage = stages.find((s) => s.stage === 3)
  const graphStage = stages.find((s) => s.stage === 6)
  if (!entityStage || !graphStage || entityStage.count === 0) return 0
  return Math.round((1 - graphStage.count / entityStage.count) * 100)
}
