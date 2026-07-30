// İlk prototipte desteklenen parmak etiketi kümesi; çoklu parmak aşamasında genişletilecek.
export type FingerLabel = 'right_index' | 'left_index' | 'unknown';

// Kalite durumları UI, kayıt ve enrollment kararlarında ortak dil olarak kullanılır.
export type QualityStatus = 'unknown' | 'poor' | 'usable' | 'good';

// Capture sonrası hesaplanan temel görüntü kalite metrikleri.
export type QualityMetrics = {
  blurScore: number;
  glareRatio: number;
  brightnessMean: number;
  roiCoverage: number;
  status: QualityStatus;
};

// Biyometrik kabulün hangi sıkı kontrolde kaldığını kalibrasyon için makinece okunabilir biçimde taşır.
export type BiometricRejectionReason =
  | 'segmentation'
  | 'global-score'
  | 'orientation'
  | 'periodicity'
  | 'frequency-consistency'
  | 'evidence-ratio'
  | 'evidence-count'
  | 'source-resolution';

// Her parmak ROI'si için cihaz üzerinde hesaplanan ayrıntılı kalite sonucunu taşır.
export type FingerprintQuality = {
  globalScore: number;
  captureStatus: 'good' | 'medium' | 'poor';
  biometricStatus: 'sufficient' | 'insufficient';
  sourceResolutionScore: number;
  validEvidenceRatio: number;
  blurScore: number;
  contrastScore: number;
  brightnessScore: number;
  foregroundCoverage: number;
  textureVisibility: number;
  orientationCoherence: number;
  orientationReliableBlockRatio: number;
  orientationMedianCorrectionDegrees: number;
  ridgePeriodicity: number;
  ridgeFrequencyConsistency: number;
  ridgeValidBlockRatio: number;
  ridgeMedianPeriodPixels: number;
  ridgeOrientationBlockCount: number;
  ridgeInteriorBlockCount: number;
  ridgeCandidateBlockCount: number;
  ridgeValidBlockCount: number;
  ridgeAnalysisBlockSize?: number;
  ridgeAnalyzedScaleCount?: number;
  ridgeOrientationCandidateRatio?: number;
  biometricRequiredValidBlockCount?: number;
  biometricRejectionReasons?: BiometricRejectionReason[];
  ridgePeriodHistogram: string;
  ridgeRejectionSummary: string;
  ridgeEnhancementGainPercent: number;
  ridgeEnhancementSupportedAreaRatio: number;
  status: 'good' | 'medium' | 'poor';
  message: string;
};

// Kalibrasyon çekimlerini ham fotoğrafı paylaşmadan elle sınıflandırmak için kullanılan etiketler.
export type QualityCalibrationLabel = 'good' | 'borderline' | 'bad';

// OBB modelinin desteklediği parmak ucu sınıf adları.
export type DetectionClassName = 'index' | 'middle' | 'pinky' | 'ring' | 'unknown';

// Modelden gelen açılı kutuyu görüntüye oranlı dört köşe noktasıyla taşır.
export type DetectedObbBox = {
  id: string;
  classId: number;
  className: DetectionClassName;
  confidence: number;
  center: {
    x: number;
    y: number;
  };
  size: {
    width: number;
    height: number;
  };
  angle: number;
  points: {
    x: number;
    y: number;
  }[];
};

// Model kutusundan çıkarılan tek parmak ROI'sinin dosya ve koordinat bilgisini taşır.
export type FingerRoi = {
  id: string;
  detectionId: string;
  className: DetectionClassName;
  confidence: number;
  imageUri?: string;
  canonicalImageUri?: string;
  segmentedImageUri?: string;
  enhancedImageUri?: string;
  orientationImageUri?: string;
  sourcePixelWidth?: number;
  canonicalRotationDegrees?: number;
  silhouetteAxisDegrees?: number;
  canonicalResidualDegrees?: number;
  quality?: FingerprintQuality;
  maskPolygon?: {
    x: number;
    y: number;
  }[];
  normalizedMaskPolygon?: {
    x: number;
    y: number;
  }[];
  crop: {
    originX: number;
    originY: number;
    width: number;
    height: number;
    normalized: {
      x: number;
      y: number;
      width: number;
      height: number;
    };
  };
};

// Tek bir kamera yakalamasının ham görsel, ROI ve kalite metadata'sını taşır.
export type CaptureSample = {
  id: string;
  createdAt: string;
  rawImageUri: string;
  roiImageUri?: string;
  processedRoiImageUri?: string;
  roiCrop?: {
    originX: number;
    originY: number;
    width: number;
    height: number;
  };
  rawImageSize?: {
    width: number;
    height: number;
  };
  qualityMetrics?: QualityMetrics;
  processedQualityMetrics?: QualityMetrics;
  detections?: DetectedObbBox[];
  fingerRois?: FingerRoi[];
  deviceModel?: string;
  fingerLabel: FingerLabel;
  sessionId: string;
  qualityStatus: QualityStatus;
  calibrationLabel?: QualityCalibrationLabel;
  accepted: boolean;
};

// Yerel galeriye eklenecek kişi kaydının temel modeli.
export type Person = {
  id: string;
  displayName: string;
  createdAt: string;
};

// Bir kişinin belirli parmağı için kabul edilmiş capture örneklerini gruplayan enrollment modeli.
export type Enrollment = {
  id: string;
  personId: string;
  fingerLabel: FingerLabel;
  samples: string[];
  templateVersion: string;
};

// 1:N arama sonucunda aday kişiyi skor ve sıralama bilgisiyle döndürecek tip.
export type SearchResult = {
  personId: string;
  score: number;
  rank: number;
};
