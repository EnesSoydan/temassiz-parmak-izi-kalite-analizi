import type { CaptureSample, SearchResult } from '@/types/biometrics';

export function rankSamplesByQuality(samples: CaptureSample[]): SearchResult[] {
  return samples
    .map((sample) => ({
      personId: sample.sessionId,
      score: sample.qualityMetrics?.blurScore ?? 0,
      rank: 0,
    }))
    .sort((left, right) => right.score - left.score)
    .slice(0, 3)
    .map((result, index) => ({
      ...result,
      rank: index + 1,
    }));
}
