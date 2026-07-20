export type FingerLabel = 'right_index' | 'left_index' | 'unknown';

export type QualityStatus = 'unknown' | 'poor' | 'usable' | 'good';

export type QualityMetrics = {
  blurScore: number;
  glareRatio: number;
  brightnessMean: number;
  roiCoverage: number;
  status: QualityStatus;
};

export type CaptureSample = {
  id: string;
  createdAt: string;
  rawImageUri: string;
  roiImageUri?: string;
  qualityMetrics?: QualityMetrics;
  deviceModel?: string;
  fingerLabel: FingerLabel;
  sessionId: string;
  qualityStatus: QualityStatus;
  accepted: boolean;
};

export type Person = {
  id: string;
  displayName: string;
  createdAt: string;
};

export type Enrollment = {
  id: string;
  personId: string;
  fingerLabel: FingerLabel;
  samples: string[];
  templateVersion: string;
};

export type SearchResult = {
  personId: string;
  score: number;
  rank: number;
};
