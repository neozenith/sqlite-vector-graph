import { describe, it, expect } from 'vitest';
import {
  toFunnelData,
  completedStageCount,
  entityReductionRatio,
} from '../transforms/kg-pipeline';
import type { PipelineStage } from '../types';

const stages: PipelineStage[] = [
  { stage: 1, name: 'Chunking', description: '', count: 1850, available: true },
  { stage: 2, name: 'Embedding', description: '', count: 1850, available: true },
  { stage: 3, name: 'Entity Extraction', description: '', count: 11641, available: true },
  { stage: 4, name: 'Relation Extraction', description: '', count: 3048, available: true },
  { stage: 5, name: 'Entity Resolution', description: '', count: 100, available: true },
  { stage: 6, name: 'Graph Construction', description: '', count: 56, available: true },
  { stage: 7, name: 'Node2Vec', description: '', count: 0, available: false },
];

describe('toFunnelData', () => {
  it('converts stages to funnel data', () => {
    const funnel = toFunnelData(stages);
    // Only available stages
    expect(funnel).toHaveLength(6);
    expect(funnel[0].name).toBe('Chunking');
    expect(funnel[0].count).toBe(1850);
  });

  it('calculates percentage relative to max count', () => {
    const funnel = toFunnelData(stages);
    // Max count is 11641 (entities)
    const entityDatum = funnel.find((f) => f.stage === 3)!;
    expect(entityDatum.percentage).toBe(100);

    const graphDatum = funnel.find((f) => f.stage === 6)!;
    expect(graphDatum.percentage).toBeLessThan(5); // 56/11641 ~ 0.5%
  });

  it('excludes unavailable stages', () => {
    const funnel = toFunnelData(stages);
    expect(funnel.find((f) => f.stage === 7)).toBeUndefined();
  });
});

describe('completedStageCount', () => {
  it('counts available stages', () => {
    expect(completedStageCount(stages)).toBe(6);
  });

  it('returns 0 for empty array', () => {
    expect(completedStageCount([])).toBe(0);
  });
});

describe('entityReductionRatio', () => {
  it('calculates reduction from entities to nodes', () => {
    const ratio = entityReductionRatio(stages);
    // 11641 -> 56 = ~99.5% reduction
    expect(ratio).toBeGreaterThan(99);
  });

  it('returns 0 when entity count is 0', () => {
    const emptyStages: PipelineStage[] = [
      { stage: 3, name: 'Entities', description: '', count: 0, available: true },
      { stage: 6, name: 'Graph', description: '', count: 0, available: true },
    ];
    expect(entityReductionRatio(emptyStages)).toBe(0);
  });

  it('returns 0 when stages are missing', () => {
    expect(entityReductionRatio([])).toBe(0);
  });
});
