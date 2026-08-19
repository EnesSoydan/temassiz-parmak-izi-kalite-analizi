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

// Minutiae template kabulünün görüntü kalitesinden sonraki ayrı ret nedenini taşır.
export type MinutiaeRejectionReason =
  | 'capture-quality'
  | 'search-area'
  | 'candidate-count'
  | 'candidate-overflow';

// Ridge iskeletinin hangi temizlik aşamasında parçalandığını karşılaştırmak için topoloji özeti.
export type MinutiaeTopologyStageDiagnostics = {
  pixelCount: number;
  componentCount: number;
  endingPixelCount: number;
  bifurcationPixelCount: number;
};

export type MinutiaeTopologyDiagnostics = {
  binary: MinutiaeTopologyStageDiagnostics;
  opened?: MinutiaeTopologyStageDiagnostics;
  orientationCleaned?: MinutiaeTopologyStageDiagnostics;
  thinned: MinutiaeTopologyStageDiagnostics;
  bridged: MinutiaeTopologyStageDiagnostics;
  componentFiltered: MinutiaeTopologyStageDiagnostics;
  pruned: MinutiaeTopologyStageDiagnostics;
  thresholdLoopCleaned?: MinutiaeTopologyStageDiagnostics;
};

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
  orientationDetailBlockSize?: number;
  orientationDetailVerifiedRatio?: number;
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
  minutiaeStatus?: 'sufficient' | 'insufficient';
  minutiaeRejectionReason?: MinutiaeRejectionReason;
  minutiaeCandidateCount?: number;
  minutiaeCrossingNumberCandidateCount?: number;
  minutiaeBranchValidatedCandidateCount?: number;
  minutiaeSuppressionCandidateCount?: number;
  minutiaeEndingCandidateCount?: number;
  minutiaeBifurcationCandidateCount?: number;
  minutiaeConfidenceHistogram?: number[];
  minutiaeThresholdPrimaryCount?: number;
  minutiaeThresholdLocationStableCount?: number;
  minutiaeThresholdTypeStableCount?: number;
  minutiaeThresholdLoopCandidateCount?: number;
  minutiaeThresholdLoopStableCount?: number;
  minutiaeThresholdLoopRemovedCount?: number;
  minutiaeThresholdLoopRemovedPixelCount?: number;
  minutiaeOrientationBridgeRemovedPixelCount?: number;
  minutiaeSearchableAreaRatio?: number;
  minutiaeLargestRegionRatio?: number;
  minutiaeTopology?: MinutiaeTopologyDiagnostics;
  status: 'good' | 'medium' | 'poor';
  message: string;
};

// Kalibrasyon çekimlerini ham fotoğrafı paylaşmadan elle sınıflandırmak için kullanılan etiketler.
export type QualityCalibrationLabel = 'good' | 'borderline' | 'bad';

// Gerçek giriş çekiminin yalnızca değerlendirmede kullanılan biyometrik gerçeğini taşır.
// Bu etiket matcher'a veya enrollment şablonlarına hiçbir zaman geri beslenmez.
export type ProbeEvaluationTruth =
  | {
      version: 1;
      relation: 'genuine';
      expectedPersonId: string;
      evaluationSessionId: string;
      labeledAt: string;
    }
  | {
      version: 1;
      relation: 'impostor';
      evaluationSessionId: string;
      labeledAt: string;
    };

// OBB modelinin desteklediği parmak ucu sınıf adları.
export type DetectionClassName = 'index' | 'middle' | 'pinky' | 'ring' | 'unknown';

// Enrollment ve 1:N eşleştirmede modelin gerçekten ayırt ettiği dört parmak konumu.
export const FINGERPRINT_POSITIONS = ['index', 'middle', 'ring', 'pinky'] as const;
export type FingerprintPosition = (typeof FINGERPRINT_POSITIONS)[number];

export function isFingerprintPosition(
  className: DetectionClassName
): className is FingerprintPosition {
  return className !== 'unknown';
}

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

// İskelet üzerinde tespit edilen ridge sonu ve çatallanma noktalarının desteklenen türleri.
export type MinutiaType = 'ending' | 'bifurcation';

// Tek bir minutia noktasını kanonik ROI'ye göre normalize koordinat, yön ve güvenle taşır.
export type FingerprintMinutia = {
  x: number;
  y: number;
  angleDegrees: number;
  type: MinutiaType;
  confidence: number;
};

// Enrollment öncesi ilk yerel biyometrik şablon, görüntüden bağımsız normalize minutiae listesini saklar.
export type FingerprintTemplate = {
  version: 'minutiae-v1' | 'minutiae-v2';
  width: number;
  height: number;
  ridgePeriodPixels: number;
  minutiae: FingerprintMinutia[];
  coordinateFrame?: 'source-roi' | 'homography-canonical';
  ridgeScaleFactor?: number;
  textureDescriptor?: number[];
};

// Eski çift çekim kayıtlarının metadata'sını okuyabilmek için geriye dönük tip.
export type ExposurePairComparison = {
  alignmentConfidence: number;
  ambientBetterBlockRatio: number;
  flashBetterBlockRatio: number;
  comparableBlockCount: number;
  ambientGlareRatio: number;
  flashGlareRatio: number;
  centerFlashBetterBlockRatio: number;
  outerAmbientBetterBlockRatio: number;
  ambientMeanScore: number;
  flashMeanScore: number;
  alignmentShiftX: number;
  alignmentShiftY: number;
  alignmentScale: number;
  alignmentRotationDegrees: number;
  recommendation: 'ambient' | 'flash' | 'complementary' | 'unaligned';
  processingSource?: 'ambient' | 'flash' | 'local';
};

// Model kutusundan çıkarılan tek parmak ROI'sinin dosya ve koordinat bilgisini taşır.
export type FingerRoi = {
  id: string;
  detectionId: string;
  className: DetectionClassName;
  confidence: number;
  imageUri?: string;
  canonicalImageUri?: string;
  // Eski çift çekim kayıtlarında bulunabilir; yeni kayıtlar yalnızca canonicalImageUri yazar.
  ambientCanonicalImageUri?: string;
  flashCanonicalImageUri?: string;
  alignedCanonicalImageUri?: string;
  canonicalSource?: 'ambient' | 'flash' | 'single';
  segmentedImageUri?: string;
  maskImageUri?: string;
  binaryImageUri?: string;
  openedBinaryImageUri?: string;
  minutiaeOverlayImageUri?: string;
  enhancedImageUri?: string;
  orientationImageUri?: string;
  minutiaeImageUri?: string;
  minutiaeTemplate?: FingerprintTemplate;
  exposureComparison?: ExposurePairComparison;
  sourcePixelWidth?: number;
  canonicalRotationDegrees?: number;
  homography?: [
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
  ];
  homographySourceSize?: { width: number; height: number };
  homographyTargetSize?: { width: number; height: number };
  ridgeScaleFactor?: number;
  coordinateFrame?: 'source-roi' | 'homography-canonical';
  perspectiveCorrected?: boolean;
  silhouetteAxisDegrees?: number;
  canonicalResidualDegrees?: number;
  silhouetteCorrectionDegrees?: number;
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
  // Eski çift çekim kayıtlarını silebilmek için korunur; yeni kayıtlar bu alanı yazmaz.
  exposurePair?: {
    ambientImageUri: string;
    flashImageUri: string;
  };
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
  pipelineVersion?: 'roi-homography-ridge-v2';
  captureConditions?: {
    flash: 'on' | 'off';
    estimatedDistanceCm?: number;
    estimatedHandAngleDegrees?: number;
  };
  fingerLabel: FingerLabel;
  sessionId: string;
  qualityStatus: QualityStatus;
  calibrationLabel?: QualityCalibrationLabel;
  probeEvaluation?: ProbeEvaluationTruth;
  accepted: boolean;
};

// Yerel galeriye eklenecek kişi kaydının temel modeli.
export type Person = {
  id: string;
  displayName: string;
  createdAt: string;
  schemaVersion: 1;
};

// Bir kişinin belirli parmağı için kabul edilmiş minutiae şablonunu taşıyan enrollment modeli.
export type Enrollment = {
  id: string;
  personId: string;
  fingerPosition: FingerprintPosition;
  template: FingerprintTemplate;
  qualitySnapshot: {
    overallScore: number;
    biometricStatus: 'sufficient';
    ridgeScore: number;
    orientationScore: number;
  };
  sourceCaptureId: string;
  createdAt: string;
  sampleIndex?: 0 | 1 | 2;
  coordinateFrame?: 'source-roi' | 'homography-canonical';
  templateVersion: FingerprintTemplate['version'];
};

// Şifreli yerel veritabanının düz metin içeriği; dosyaya yazılmadan önce AES-GCM ile korunur.
export type BiometricDatabase = {
  schemaVersion: 1;
  people: Person[];
  enrollments: Enrollment[];
  updatedAt: string;
};

export type FingerMatchStatus =
  | 'matched'
  | 'no-match'
  | 'insufficient'
  | 'missing';

export type FingerMatchFailureReason =
  | 'probe-missing'
  | 'probe-quality-insufficient'
  | 'enrollment-missing'
  | 'not-enough-probe-minutiae'
  | 'not-enough-enrollment-minutiae'
  | 'matched-minutiae-below-threshold'
  | 'coverage-below-threshold'
  | 'template-version-mismatch';

// Tek bir parmak karşılaştırmasının karar ve hata ayıklama için güvenli özeti.
export type FingerMatchResult = {
  fingerPosition: FingerprintPosition;
  status: FingerMatchStatus;
  score: number;
  matchedMinutiae: number;
  matchedEndingCount?: number;
  matchedBifurcationCount?: number;
  coverage: number;
  probeUsableMinutiae: number;
  enrollmentUsableMinutiae: number;
  minimumUsableMinutiae: number;
  minimumMatchedMinutiae: number;
  minimumCoverage: number;
  failureReasons: FingerMatchFailureReason[];
  transform?: {
    rotationDegrees: number;
    scale: number;
    translationX: number;
    translationY: number;
  };
  graphRelationScore?: number;
  confidenceScore?: number;
  distanceScore?: number;
  textureScore?: number;
};

export type PersonCandidateResult = {
  personId: string;
  displayName: string;
  score: number;
  matchedFingerCount: number;
  totalMatchedMinutiae: number;
  averageCoverage: number;
  fingerResults: FingerMatchResult[];
};

// 1:N arama sonucunda aday kişiyi skor ve parmak kanıtlarıyla döndüren tip.
export type PersonMatchResult = {
  personId: string;
  displayName: string;
  accepted: boolean;
  score: number;
  matchedFingerCount: number;
  totalMatchedMinutiae: number;
  averageCoverage: number;
  fingerResults: FingerMatchResult[];
  candidateResults: PersonCandidateResult[];
  validProbeFingerPositions: FingerprintPosition[];
  reason:
    | 'accepted'
    | 'no-people'
    | 'no-valid-probe'
    | 'no-match'
    | 'insufficient-multi-finger-evidence'
    | 'ambiguous'
    | 'conflicting-evidence';
};

// Eski kalite sıralama prototipinin geriye dönük sonucu.
export type SearchResult = {
  personId: string;
  score: number;
  rank: number;
};
