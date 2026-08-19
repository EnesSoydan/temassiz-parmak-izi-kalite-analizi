import type {
  BiometricRejectionReason,
  DetectionClassName,
  FingerRoi,
  FingerprintQuality,
  QualityStatus,
} from '@/types/biometrics';
import {
  estimateOrientationField,
  selectVerifiedFineOrientationBlocks,
  type OrientationBlock,
} from '@/lib/orientation-field';
import {
  FINGERPRINT_QUALITY_THRESHOLDS as QUALITY_THRESHOLDS,
  getScaleAwareEvidenceMinimums,
  hasStrongLocalOrientationEvidence,
  shouldTrySecondaryRidgeScale,
} from '@/lib/fingerprint-quality-config';
import { createOrientationVisualization } from '@/lib/orientation-visualization';
import { createMaskedClaheGrayscale as createMaskedClaheGrayscaleCore } from '@/lib/clahe';
import { createControlledRidgeEnhancement } from '@/lib/ridge-enhancement';
import {
  estimateRidgeFrequency,
  selectPreferredRidgeScale,
} from '@/lib/ridge-frequency';

type QualityAnalysisInput = {
  pixels: Uint8Array;
  mask: Uint8Array;
  width: number;
  height: number;
  coverage: number;
  sourcePixelWidth?: number;
  fingerClass?: DetectionClassName;
};

type BlockQuality = {
  blurScore: number;
  contrastScore: number;
  brightnessScore: number;
  textureScore: number;
  orientationScore: number;
  meanBrightness: number;
  shadowRatio: number;
  glareRatio: number;
};

// Blokların parmak alanı sayılması için gereken minimum maske kapsaması.
const MIN_BLOCK_MASK_COVERAGE = 0.6;

// Kitaptaki yerel kalite yaklaşımına uygun, ayarlanabilir blok taban genişliği.
const BASE_BLOCK_SIZE = 16;

// Araştırma aşamasında kaba yön kararını bozmadan daha yerel ridge akışını sınayan ince ızgara.
const EXPERIMENTAL_FINE_ORIENTATION_BLOCK_SIZE = 8;

// Segmentasyon maskesi içindeki ROI'yi bloklara ayırarak parmak bazında kalite sonucu üretir.
export function analyzeFingerprintQuality({
  ...input
}: QualityAnalysisInput): FingerprintQuality {
  return analyzeFingerprintQualityDetailed(input).quality;
}

// Kalite kararını ham veriden üretirken aynı orientation/frequency hesabından enhancement çıktısı da oluşturur.
export function analyzeFingerprintQualityDetailed({
  pixels,
  mask,
  width,
  height,
  coverage,
  sourcePixelWidth = width,
  fingerClass,
}: QualityAnalysisInput) {
  const rawGrayscale = createGrayscaleImage(pixels, width, height);
  const rawIntegral = createIntegralImage(rawGrayscale, width, height);
  const featureGrayscale = preprocessFingerprintGrayscale(
    rawGrayscale,
    mask,
    width,
    height,
    rawIntegral
  );
  const frequencyGrayscale = preprocessRidgeFrequencyGrayscale(
    rawGrayscale,
    mask,
    width,
    height,
    rawIntegral
  );
  const blockSize = getBlockSize(width, height);
  const primaryScale = analyzeRidgeScale({
    featureGrayscale,
    frequencyGrayscale,
    mask,
    width,
    height,
    blockSize,
  });
  // Serçede daima, diğer parmaklarda yalnızca ana kanıt yetersizse ikinci ölçeği kontrollü biçimde deneriz.
  const shouldAnalyzeSecondaryScale =
    fingerClass === 'pinky' ||
    shouldTrySecondaryRidgeScale({
      ridgeValidBlockRatio: primaryScale.ridgeFrequency.validBlockRatio,
      ridgeValidBlockCount: primaryScale.ridgeFrequency.validBlockCount,
      ridgeCandidateBlockCount:
        primaryScale.ridgeFrequency.candidateBlockCount,
    });
  const selectedScale =
    shouldAnalyzeSecondaryScale
      ? selectPreferredRidgeScale(
          primaryScale,
          analyzeRidgeScale({
            featureGrayscale,
            frequencyGrayscale,
            mask,
            width,
            height,
            blockSize: getSecondaryOrientationBlockSize(blockSize),
          })
        )
      : primaryScale;
  const analyzedScaleCount = shouldAnalyzeSecondaryScale ? 2 : 1;
  const { orientationBlocks, ridgeFrequency } = selectedScale;
  const fineOrientationBlocks = estimateOrientationField({
    grayscale: featureGrayscale,
    mask,
    width,
    height,
    blockSize: EXPERIMENTAL_FINE_ORIENTATION_BLOCK_SIZE,
    minMaskCoverage: MIN_BLOCK_MASK_COVERAGE,
  });
  const verifiedFineOrientationBlocks = selectVerifiedFineOrientationBlocks({
    fineBlocks: fineOrientationBlocks,
    referenceBlocks: orientationBlocks,
  });
  const finePrimaryBlockCount = fineOrientationBlocks.filter(
    (block) =>
      (block.gridOffset ?? 0) === 0 && block.maskCoverage >= 0.85
  ).length;
  const fineOrientationVerifiedRatio = Math.round(
    (verifiedFineOrientationBlocks.length /
      Math.max(finePrimaryBlockCount, 1)) *
      100
  );
  const orientationVisualization = createOrientationVisualization({
    grayscale: featureGrayscale,
    mask,
    width,
    height,
    orientationBlocks,
    detailCandidateBlocks: fineOrientationBlocks,
    detailOrientationBlocks: verifiedFineOrientationBlocks,
    frequencyBlocks: ridgeFrequency.blocks,
  });
  const blocks: BlockQuality[] = [];

  for (const orientationBlock of orientationBlocks) {
    const block = analyzeBlock(rawGrayscale, featureGrayscale, mask, width, orientationBlock);
    if (block) blocks.push(block);
  }

  if (blocks.length === 0) {
    return {
      quality: createSegmentationFailureQuality(coverage),
      enhancedPixels: undefined,
      enhancementSupportedAreaRatio: 0,
      orientationPixels: orientationVisualization.pixels,
      minutiaeSupportMask: undefined,
      minutiaeOrientationMask: undefined,
      minutiaeOrientationAngles: undefined,
      minutiaeOrientationFieldMask: undefined,
    };
  }

  const blurScore = summarizeBlockScores(blocks.map((block) => block.blurScore));
  const contrastScore = summarizeBlockScores(blocks.map((block) => block.contrastScore));
  const brightnessScore = summarizeBlockScores(blocks.map((block) => block.brightnessScore));
  const textureVisibility = summarizeBlockScores(blocks.map((block) => block.textureScore));
  const orientationCoherence = summarizeBlockScores(blocks.map((block) => block.orientationScore));
  const {
    ridgePeriodicity,
    ridgeFrequencyConsistency,
    medianPeriodPixels: ridgeMedianPeriodPixels,
  } = ridgeFrequency;
  const ridgeValidBlockCount = ridgeFrequency.validBlockCount;
  const ridgeValidBlockRatio = Math.round(ridgeFrequency.validBlockRatio * 100);
  const evidenceMinimums = getScaleAwareEvidenceMinimums(ridgeFrequency.candidateBlockCount);
  const sourceResolutionScore = scoreRange(sourcePixelWidth, 48, 150);
  const foregroundCoverage = Math.round(clamp01(coverage) * 100);
  const coverageScore = calculateCoverageScore(coverage);
  const weightedGlobalScore = Math.round(
    blurScore * 0.16 +
      textureVisibility * 0.12 +
      contrastScore * 0.1 +
      orientationCoherence * 0.16 +
      ridgePeriodicity * 0.2 +
      ridgeFrequencyConsistency * 0.1 +
      ridgeValidBlockRatio * 0.08 +
      brightnessScore * 0.05 +
      coverageScore * 0.03
  );
  // Ridge alanı çok küçükse ağırlıklı skorun nihai kaliteyi olduğundan yüksek göstermesini sınırlar.
  const globalScore =
    ridgeValidBlockCount === 0
      ? Math.min(weightedGlobalScore, 49)
      : ridgeValidBlockCount < evidenceMinimums.medium ||
          ridgeValidBlockRatio < QUALITY_THRESHOLDS.weakRidgeValidBlockRatio
        ? Math.min(weightedGlobalScore, 59)
        : weightedGlobalScore;
  const meanBrightness = getMedian(blocks.map((block) => block.meanBrightness));
  const glareRatio = getPercentile(blocks.map((block) => block.glareRatio), 0.75);
  const message = createQualityMessage({
    coverageScore,
    blurScore,
    contrastScore,
    brightnessScore,
    textureVisibility,
    orientationCoherence,
    ridgePeriodicity,
    ridgeFrequencyConsistency,
    ridgeValidBlockRatio,
    ridgeValidBlockCount,
    ridgeCandidateBlockCount: ridgeFrequency.candidateBlockCount,
    meanBrightness,
    glareRatio,
  });
  const captureStatus = getFingerprintCaptureStatus({
    globalScore,
    orientationCoherence,
    ridgePeriodicity,
    ridgeFrequencyConsistency,
    ridgeValidBlockRatio,
    ridgeValidBlockCount,
    ridgeCandidateBlockCount: ridgeFrequency.candidateBlockCount,
  });
  const biometricAssessment = getFingerprintBiometricAssessment({
    fingerClass,
    globalScore,
    orientationCoherence,
    ridgePeriodicity,
    ridgeFrequencyConsistency,
    ridgeValidBlockRatio,
    ridgeValidBlockCount,
    ridgeCandidateBlockCount: ridgeFrequency.candidateBlockCount,
    sourceResolutionScore,
  });
  const enhancement = createControlledRidgeEnhancement({
    grayscale: featureGrayscale,
    mask,
    width,
    height,
    frequencyBlocks: ridgeFrequency.blocks,
  });
  const minutiaeOrientationMask = createReliableOrientationMask({
    mask,
    width,
    height,
    orientationBlocks,
  });
  const minutiaeOrientationField = createMinutiaeOrientationField({
    mask,
    width,
    height,
    orientationBlocks: verifiedFineOrientationBlocks,
  });

  const quality: FingerprintQuality = {
    globalScore: clampScore(globalScore),
    captureStatus,
    biometricStatus: biometricAssessment.status,
    sourceResolutionScore,
    validEvidenceRatio: ridgeValidBlockRatio,
    blurScore,
    contrastScore,
    brightnessScore,
    foregroundCoverage,
    textureVisibility,
    orientationCoherence,
    orientationReliableBlockRatio: Math.round(
      orientationVisualization.reliableBlockRatio * 100
    ),
    orientationMedianCorrectionDegrees:
      orientationVisualization.medianAngularCorrectionDegrees,
    orientationDetailBlockSize: EXPERIMENTAL_FINE_ORIENTATION_BLOCK_SIZE,
    orientationDetailVerifiedRatio: fineOrientationVerifiedRatio,
    ridgePeriodicity,
    ridgeFrequencyConsistency,
    ridgeValidBlockRatio,
    ridgeMedianPeriodPixels,
    ridgeOrientationBlockCount: ridgeFrequency.orientationBlockCount,
    ridgeInteriorBlockCount: ridgeFrequency.interiorBlockCount,
    ridgeCandidateBlockCount: ridgeFrequency.candidateBlockCount,
    ridgeValidBlockCount: ridgeFrequency.validBlockCount,
    ridgeAnalysisBlockSize: selectedScale.blockSize,
    ridgeAnalyzedScaleCount: analyzedScaleCount,
    ridgeOrientationCandidateRatio: Math.round(
      (ridgeFrequency.candidateBlockCount /
        Math.max(ridgeFrequency.interiorBlockCount, 1)) *
        100
    ),
    biometricRequiredValidBlockCount: evidenceMinimums.biometric,
    biometricRejectionReasons: biometricAssessment.reasons,
    ridgePeriodHistogram: ridgeFrequency.periodHistogram,
    ridgeRejectionSummary: ridgeFrequency.rejectionSummary,
    ridgeEnhancementGainPercent: enhancement.contrastGainPercent,
    ridgeEnhancementSupportedAreaRatio: Math.round(
      enhancement.supportedAreaRatio * 100
    ),
    // Eski kayıt ve arayüz kodları için status, çekim kararının geriye uyumlu karşılığıdır.
    status: captureStatus,
    message,
  };

  return {
    quality,
    enhancedPixels: enhancement.pixels,
    enhancementSupportedAreaRatio: enhancement.supportedAreaRatio,
    orientationPixels: orientationVisualization.pixels,
    minutiaeSupportMask: enhancement.minutiaeSupportMask,
    minutiaeOrientationMask,
    minutiaeOrientationAngles: minutiaeOrientationField.angles,
    minutiaeOrientationFieldMask: minutiaeOrientationField.mask,
  };
}

// Kaba alanla doğrulanmış 8 px yön bloklarını, yalnızca güvenilir oldukları piksellerde
// ridge eksenini taşıyan sürekli bir alana dönüştürür. Doğrulanmamış boşluklar NaN kalır.
function createMinutiaeOrientationField({
  mask,
  width,
  height,
  orientationBlocks,
}: {
  mask: Uint8Array;
  width: number;
  height: number;
  orientationBlocks: OrientationBlock[];
}) {
  const angles = new Float32Array(mask.length);
  angles.fill(Number.NaN);
  const fieldMask = new Uint8Array(mask.length);

  for (const block of orientationBlocks) {
    const right = Math.min(width, block.left + block.size);
    const bottom = Math.min(height, block.top + block.size);
    for (let y = Math.max(0, block.top); y < bottom; y += 1) {
      for (let x = Math.max(0, block.left); x < right; x += 1) {
        const index = y * width + x;
        if (!mask[index]) continue;
        angles[index] = block.angleRadians;
        fieldMask[index] = 1;
      }
    }
  }

  return { angles, mask: fieldMask };
}

// Minutiae aramasının Gabor adacıklarına hapsolmaması için güvenilir yön bloklarından sürekli bir izin alanı üretir.
function createReliableOrientationMask({
  mask,
  width,
  height,
  orientationBlocks,
}: {
  mask: Uint8Array;
  width: number;
  height: number;
  orientationBlocks: OrientationBlock[];
}) {
  const orientationMask = new Uint8Array(mask.length);

  for (const block of orientationBlocks) {
    const smoothedCoherence = block.smoothedCoherence ?? block.coherence;
    if (
      block.maskCoverage < 0.72 ||
      block.coherence < 0.18 ||
      smoothedCoherence < 0.34 ||
      block.neighborhoodConsistency < 0.34
    ) {
      continue;
    }

    const right = Math.min(width, block.left + block.size);
    const bottom = Math.min(height, block.top + block.size);
    for (let y = Math.max(0, block.top); y < bottom; y += 1) {
      for (let x = Math.max(0, block.left); x < right; x += 1) {
        const index = y * width + x;
        if (mask[index]) orientationMask[index] = 1;
      }
    }
  }

  return orientationMask;
}

// Segmentasyon geçersizse kaydı korurken kalite sonucunu açıkça başarısız işaretler.
export function createSegmentationFailureQuality(foregroundCoverage = 0): FingerprintQuality {
  return {
    globalScore: 0,
    captureStatus: 'poor',
    biometricStatus: 'insufficient',
    sourceResolutionScore: 0,
    validEvidenceRatio: 0,
    blurScore: 0,
    contrastScore: 0,
    brightnessScore: 0,
    foregroundCoverage: Math.round(clamp01(foregroundCoverage) * 100),
    textureVisibility: 0,
    orientationCoherence: 0,
    orientationReliableBlockRatio: 0,
    orientationMedianCorrectionDegrees: 0,
    ridgePeriodicity: 0,
    ridgeFrequencyConsistency: 0,
    ridgeValidBlockRatio: 0,
    ridgeMedianPeriodPixels: 0,
    ridgeOrientationBlockCount: 0,
    ridgeInteriorBlockCount: 0,
    ridgeCandidateBlockCount: 0,
    ridgeValidBlockCount: 0,
    ridgeAnalysisBlockSize: 0,
    ridgeAnalyzedScaleCount: 0,
    ridgeOrientationCandidateRatio: 0,
    biometricRequiredValidBlockCount: 0,
    biometricRejectionReasons: ['segmentation'],
    ridgePeriodHistogram: 'yok',
    ridgeRejectionSummary: 'segmentasyon',
    ridgeEnhancementGainPercent: 0,
    ridgeEnhancementSupportedAreaRatio: 0,
    status: 'poor',
    message: 'Parmak alanı ayrılamadı. Elini daha düz ve net göster.',
  };
}

// Çoklu parmak kaydının genel durumunu en düşük ROI kalitesinden türetir.
export function getCaptureQualityStatus(fingerRois: FingerRoi[]): QualityStatus {
  const qualities = fingerRois
    .map((fingerRoi) => fingerRoi.quality)
    .filter((quality): quality is FingerprintQuality => quality !== undefined);

  if (qualities.length === 0) return 'unknown';
  if (qualities.some((quality) => getCaptureStatus(quality) === 'poor')) return 'poor';
  if (qualities.some((quality) => getCaptureStatus(quality) === 'medium')) return 'usable';
  return 'good';
}

// Kamera ekranında en zayıf parmağı önceleyen kısa ve anlaşılır geri bildirim üretir.
export function createCaptureQualityFeedback(fingerRois: FingerRoi[]) {
  const poorQualityRois = fingerRois
    .filter((fingerRoi) => fingerRoi.quality && getCaptureStatus(fingerRoi.quality) === 'poor')
    .sort((first, second) => (first.quality?.globalScore ?? 0) - (second.quality?.globalScore ?? 0));
  const mediumQualityCount = fingerRois.filter(
    (fingerRoi) =>
      fingerRoi.quality && getCaptureStatus(fingerRoi.quality) === 'medium'
  ).length;

  if (poorQualityRois.length === 0 && mediumQualityCount === 0) {
    return "Parmak ROI'leri uygun kalitede kaydedildi.";
  }

  if (poorQualityRois.length === 0) {
    return "Parmak ROI'leri orta kalitede kaydedildi.";
  }

  if (poorQualityRois.length > 1) {
    // Işık ve netlik yeterliyken ridge alanı birkaç parmakta birden düşüyorsa çekim açısını düzeltmeyi önerir.
    if (shouldRecommendParallelCapture(fingerRois)) {
      return 'Telefonu parmak yüzeyine paralel tutup yeniden çek.';
    }

    return `${poorQualityRois.length} parmak ROI'si için yeniden çekim öneriliyor.`;
  }

  const fingerRoi = poorQualityRois[0];
  return `${formatFingerClass(fingerRoi.className)} parmak: ${fingerRoi.quality?.message ?? 'kalite düşük.'}`;
}

// Birden fazla ROI'deki ortak ridge kaybının düşük ışık veya bulanıklıktan gelmediğini doğrular.
function shouldRecommendParallelCapture(fingerRois: FingerRoi[]) {
  const qualities = fingerRois
    .map((fingerRoi) => fingerRoi.quality)
    .filter((quality): quality is FingerprintQuality => quality !== undefined);

  if (qualities.length < 3) return false;

  const lowRidgeAreaCount = qualities.filter(
    (quality) => {
      const minimums = getScaleAwareEvidenceMinimums(quality.ridgeCandidateBlockCount);
      return (
        quality.ridgeValidBlockCount < minimums.medium ||
        quality.ridgeValidBlockRatio < QUALITY_THRESHOLDS.weakRidgeValidBlockRatio
      );
    }
  ).length;
  const otherwiseUsableCount = qualities.filter(
    (quality) =>
      quality.blurScore >= QUALITY_THRESHOLDS.weakBlur &&
      quality.brightnessScore >= 45 &&
      quality.contrastScore >= QUALITY_THRESHOLDS.weakContrast &&
      quality.foregroundCoverage >= QUALITY_THRESHOLDS.minCoverage * 100 &&
      quality.foregroundCoverage <= QUALITY_THRESHOLDS.maxCoverage * 100
  ).length;

  return lowRidgeAreaCount >= 2 && otherwiseUsableCount >= 2;
}

// Kayıtlar ekranında kullanılacak kısa Türkçe kalite etiketini döndürür.
export function formatFingerprintQualityStatus(quality?: FingerprintQuality) {
  if (!quality) return 'Analiz yok';
  if (getCaptureStatus(quality) === 'good') return 'Uygun';
  if (getCaptureStatus(quality) === 'medium') return 'Orta';
  return 'Tekrar çek';
}

// ROI boyutuna göre küçük görüntülerde aşırı ince, büyüklerde aşırı iri blok kullanılmasını önler.
function getBlockSize(width: number, height: number) {
  const shortestEdge = Math.min(width, height);
  return Math.max(BASE_BLOCK_SIZE, Math.min(32, Math.round(shortestEdge / 8)));
}

// Tek bir blok ölçeğinde orientation alanı ve ona bağlı ridge frekans kanıtını birlikte üretir.
function analyzeRidgeScale({
  featureGrayscale,
  frequencyGrayscale,
  mask,
  width,
  height,
  blockSize,
}: {
  featureGrayscale: Uint8Array;
  frequencyGrayscale: Uint8Array;
  mask: Uint8Array;
  width: number;
  height: number;
  blockSize: number;
}) {
  const orientationBlocks = estimateOrientationField({
    grayscale: featureGrayscale,
    mask,
    width,
    height,
    blockSize,
    minMaskCoverage: MIN_BLOCK_MASK_COVERAGE,
  });
  const ridgeFrequency = estimateRidgeFrequency({
    grayscale: frequencyGrayscale,
    mask,
    width,
    height,
    orientationBlocks,
  });

  return { blockSize, orientationBlocks, ridgeFrequency };
}

// Serçenin kıvrımlı ridge akışını izlemek için ana ölçekten daha ince fakat kararlı bir ikinci ızgara seçer.
function getSecondaryOrientationBlockSize(primaryBlockSize: number) {
  return Math.max(12, Math.min(primaryBlockSize - 2, Math.round(primaryBlockSize * 0.75)));
}

// Bir blokta yalnızca maske içindeki pikselleri kullanarak ışık, kontrast, keskinlik ve doku ölçer.
function analyzeBlock(
  rawGrayscale: Uint8Array,
  featureGrayscale: Uint8Array,
  mask: Uint8Array,
  width: number,
  orientationBlock: OrientationBlock
): BlockQuality | null {
  const featureHistogram = new Uint32Array(256);
  let maskedCount = 0;
  let pixelCount = 0;
  let rawTotal = 0;
  let shadowCount = 0;
  let glareCount = 0;
  const { left, top, size: blockSize } = orientationBlock;

  for (let y = top; y < top + blockSize; y += 1) {
    for (let x = left; x < left + blockSize; x += 1) {
      pixelCount += 1;
      const index = y * width + x;
      if (!mask[index]) continue;

      const rawValue = rawGrayscale[index];
      const featureValue = featureGrayscale[index];
      featureHistogram[featureValue] += 1;
      rawTotal += rawValue;
      maskedCount += 1;
      if (rawValue < 45) shadowCount += 1;
      if (rawValue > 235) glareCount += 1;
    }
  }

  if (maskedCount / Math.max(pixelCount, 1) < MIN_BLOCK_MASK_COVERAGE || maskedCount < 12) {
    return null;
  }

  const meanBrightness = rawTotal / maskedCount;
  const contrastRange =
    getHistogramPercentile(featureHistogram, maskedCount, 0.9) -
    getHistogramPercentile(featureHistogram, maskedCount, 0.1);
  const orientationScore = clampScore(
    orientationBlock.coherence * 70 + orientationBlock.neighborhoodConsistency * 30
  );
  const blurScore = scoreRange(orientationBlock.gradientEnergy, 80, 900);
  const contrastScore = scoreRange(contrastRange, 18, 62);
  const brightnessScore = calculateBrightnessScore(
    meanBrightness,
    shadowCount / maskedCount,
    glareCount / maskedCount
  );
  const textureScore = clampScore(blurScore * 0.5 + orientationScore * 0.35 + contrastScore * 0.15);

  return {
    blurScore,
    contrastScore,
    brightnessScore,
    textureScore,
    orientationScore,
    meanBrightness,
    shadowRatio: shadowCount / maskedCount,
    glareRatio: glareCount / maskedCount,
  };
}

// Ridge frekansında yapay kontrast üretmeden yalnızca yavaş ışık değişimini dengeler.
function preprocessRidgeFrequencyGrayscale(
  grayscale: Uint8Array,
  mask: Uint8Array,
  width: number,
  height: number,
  integral: Float64Array
) {
  const radius = Math.max(6, Math.min(16, Math.round(Math.min(width, height) / 9)));
  const normalized = new Uint8Array(grayscale.length);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      if (!mask[index]) continue;

      const left = Math.max(0, x - radius);
      const top = Math.max(0, y - radius);
      const right = Math.min(width - 1, x + radius);
      const bottom = Math.min(height - 1, y + radius);
      const area = (right - left + 1) * (bottom - top + 1);
      const localMean = readIntegralSum(integral, width, left, top, right, bottom) / area;
      normalized[index] = clampByte(128 + grayscale[index] - localMean);
    }
  }

  return normalized;
}

// Kalite ölçümünde çizgi dokusunu daha adil görmek için lokal ışık normalizasyonu uygular.
function preprocessFingerprintGrayscale(
  grayscale: Uint8Array,
  mask: Uint8Array,
  width: number,
  height: number,
  integral: Float64Array
) {
  const radius = Math.max(5, Math.min(12, Math.round(Math.min(width, height) / 12)));
  const normalized = new Uint8Array(grayscale.length);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      if (!mask[index]) continue;

      const left = Math.max(0, x - radius);
      const top = Math.max(0, y - radius);
      const right = Math.min(width - 1, x + radius);
      const bottom = Math.min(height - 1, y + radius);
      const area = (right - left + 1) * (bottom - top + 1);
      const localMean = readIntegralSum(integral, width, left, top, right, bottom) / area;
      const value = clampByte(128 + (grayscale[index] - localMean) * 1.3);

      normalized[index] = value;
    }
  }

  const adaptive = createMaskedClaheGrayscaleCore(normalized, mask, width, height);

  for (let index = 0; index < normalized.length; index += 1) {
    if (!mask[index]) continue;
    normalized[index] = clampByte(
      normalized[index] * 0.68 + adaptive[index] * 0.32
    );
  }

  return normalized;
}

// Yerel kontrastı artırır; clip limit parlama bölgelerinin histogramı domine etmesini engeller.
// 8-bit histogramdan sıralı diziyle aynı enterpolasyonlu yüzdelik değerini hesaplar.
function getHistogramPercentile(histogram: Uint32Array, count: number, percentile: number) {
  if (count === 0) return 0;
  const index = (count - 1) * percentile;
  const lowerRank = Math.floor(index);
  const upperRank = Math.ceil(index);
  const fraction = index - lowerRank;
  const lowerValue = getHistogramValueAtRank(histogram, lowerRank);
  const upperValue = getHistogramValueAtRank(histogram, upperRank);

  return lowerValue * (1 - fraction) + upperValue * fraction;
}

// Histogramda sıfır tabanlı hedef sıraya karşılık gelen gri değeri döndürür.
function getHistogramValueAtRank(histogram: Uint32Array, targetRank: number) {
  let cumulative = 0;

  for (let value = 0; value < histogram.length; value += 1) {
    cumulative += histogram[value];
    if (cumulative > targetRank) return value;
  }

  return 255;
}

// RGBA görüntüyü tekrar eden renk hesabını önlemek için bir kez gri tona dönüştürür.
function createGrayscaleImage(pixels: Uint8Array, width: number, height: number) {
  const grayscale = new Uint8Array(width * height);

  for (let index = 0; index < grayscale.length; index += 1) {
    const pixelIndex = index * 4;
    grayscale[index] = Math.round(
      (pixels[pixelIndex] ?? 0) * 0.299 +
        (pixels[pixelIndex + 1] ?? 0) * 0.587 +
        (pixels[pixelIndex + 2] ?? 0) * 0.114
    );
  }

  return grayscale;
}

// Yerel blok skorlarını medyan ve zayıf çeyrekle birleştirerek bozuk alanları görünür kılar.
function summarizeBlockScores(scores: number[]) {
  const median = getMedian(scores);
  const lowerQuartile = getPercentile(scores, 0.25);
  return clampScore(median * 0.7 + lowerQuartile * 0.3);
}

// Segmentasyon maskesinin çok küçük veya neredeyse tüm ROI'yi kaplaması durumunda puan düşürür.
function calculateCoverageScore(coverage: number) {
  if (coverage < QUALITY_THRESHOLDS.minCoverage) {
    return scoreRange(coverage, 0.03, QUALITY_THRESHOLDS.minCoverage);
  }

  if (coverage > QUALITY_THRESHOLDS.maxCoverage) {
    return scoreRange(1 - coverage, 0.01, 1 - QUALITY_THRESHOLDS.maxCoverage);
  }

  return 100;
}

// Ortalama ışık, gölge ve parlama oranını tek parlaklık puanına dönüştürür.
function calculateBrightnessScore(meanBrightness: number, shadowRatio: number, glareRatio: number) {
  const meanScore =
    meanBrightness < QUALITY_THRESHOLDS.dimBrightness
      ? scoreRange(meanBrightness, 20, QUALITY_THRESHOLDS.dimBrightness)
      : meanBrightness > QUALITY_THRESHOLDS.brightBrightness
        ? scoreRange(255 - meanBrightness, 20, 255 - QUALITY_THRESHOLDS.brightBrightness)
        : 100;
  const shadowPenalty = clamp01(shadowRatio / 0.28) * 25;
  const glarePenalty = clamp01(glareRatio / QUALITY_THRESHOLDS.glareRatio) * 35;

  return clampScore(meanScore - shadowPenalty - glarePenalty);
}

// En baskın sorun üzerinden kullanıcının hemen uygulayabileceği tek mesajı belirler.
function createQualityMessage({
  coverageScore,
  blurScore,
  contrastScore,
  brightnessScore,
  textureVisibility,
  orientationCoherence,
  ridgePeriodicity,
  ridgeFrequencyConsistency,
  ridgeValidBlockRatio,
  ridgeValidBlockCount,
  ridgeCandidateBlockCount,
  meanBrightness,
  glareRatio,
}: {
  coverageScore: number;
  blurScore: number;
  contrastScore: number;
  brightnessScore: number;
  textureVisibility: number;
  orientationCoherence: number;
  ridgePeriodicity: number;
  ridgeFrequencyConsistency: number;
  ridgeValidBlockRatio: number;
  ridgeValidBlockCount: number;
  ridgeCandidateBlockCount: number;
  meanBrightness: number;
  glareRatio: number;
}) {
  if (coverageScore < 45) return 'Parmak alanı ayrılamadı. Elini daha düz ve net göster.';
  if (brightnessScore < 45 && meanBrightness < QUALITY_THRESHOLDS.dimBrightness) {
    return 'Işık yetersiz. Daha aydınlık bir ortam dene.';
  }
  if (brightnessScore < 45 && glareRatio > QUALITY_THRESHOLDS.glareRatio) {
    return 'Parlama fazla. Işığı doğrudan parmağına tutma.';
  }
  if (blurScore < QUALITY_THRESHOLDS.weakBlur) return 'Görüntü bulanık. Parmağını sabit tut.';
  if (textureVisibility < QUALITY_THRESHOLDS.weakTexture) {
    return 'Parmak izi dokusu yeterince görünmüyor. Parmağını biraz yaklaştır.';
  }
  if (orientationCoherence < QUALITY_THRESHOLDS.weakOrientation) {
    return 'Çizgi yönü yeterince tutarlı değil. Parmağını daha sabit tut.';
  }
  if (ridgePeriodicity < QUALITY_THRESHOLDS.weakRidgePeriodicity) {
    return 'Parmak izi çizgileri düzenli seçilemiyor. Parmağını yaklaştır ve sabit tut.';
  }
  const evidenceMinimums = getScaleAwareEvidenceMinimums(ridgeCandidateBlockCount);
  if (
    ridgeValidBlockCount < evidenceMinimums.medium ||
    ridgeValidBlockRatio < QUALITY_THRESHOLDS.weakRidgeValidBlockRatio
  ) {
    return 'Parmak izi çizgileri yeterli alanda seçilemiyor. Parmağını kameraya yaklaştır.';
  }
  if (ridgeFrequencyConsistency < QUALITY_THRESHOLDS.weakRidgeFrequencyConsistency) {
    return 'Parmak izi çizgi aralıkları yeterince tutarlı değil. Daha net bir çekim dene.';
  }
  if (contrastScore < QUALITY_THRESHOLDS.weakContrast) {
    return 'Kontrast düşük. Daha dengeli ışıkta tekrar dene.';
  }
  return 'Görüntü uygun.';
}

// Kullanıcı yönlendirmesini ROI ölçeğine göre değişen kanıt miktarıyla belirler.
function getFingerprintCaptureStatus({
  globalScore,
  orientationCoherence,
  ridgePeriodicity,
  ridgeFrequencyConsistency,
  ridgeValidBlockRatio,
  ridgeValidBlockCount,
  ridgeCandidateBlockCount,
}: {
  globalScore: number;
  orientationCoherence: number;
  ridgePeriodicity: number;
  ridgeFrequencyConsistency: number;
  ridgeValidBlockRatio: number;
  ridgeValidBlockCount: number;
  ridgeCandidateBlockCount: number;
}): FingerprintQuality['captureStatus'] {
  const minimums = getScaleAwareEvidenceMinimums(ridgeCandidateBlockCount);

  if (
    globalScore >= QUALITY_THRESHOLDS.goodScore &&
    orientationCoherence >= 32 &&
    ridgePeriodicity >= 34 &&
    ridgeFrequencyConsistency >= 28 &&
    ridgeValidBlockRatio >= QUALITY_THRESHOLDS.captureGoodValidRatio &&
    ridgeValidBlockCount >= minimums.good
  ) {
    return 'good';
  }

  // Az sayıdaki geçerli blokta frekans komşuluğu ölçülemeyebilir; bu eksiklik orta çekimi tek başına reddetmez.
  const hasEnoughFrequencyNeighbors =
    ridgeValidBlockCount < 4 || ridgeFrequencyConsistency >= 12;
  if (
    globalScore >= QUALITY_THRESHOLDS.mediumScore &&
    orientationCoherence >= 20 &&
    ridgePeriodicity >= 16 &&
    hasEnoughFrequencyNeighbors &&
    ridgeValidBlockRatio >= QUALITY_THRESHOLDS.captureMediumValidRatio &&
    ridgeValidBlockCount >= minimums.medium
  ) {
    return 'medium';
  }
  return 'poor';
}

// Gelecekte enrollment için kullanılacak sıkı karar, ham ridge kanıtı ve kaynak çözünürlüğünü birlikte ister.
export function getFingerprintBiometricAssessment({
  fingerClass,
  globalScore,
  orientationCoherence,
  ridgePeriodicity,
  ridgeFrequencyConsistency,
  ridgeValidBlockRatio,
  ridgeValidBlockCount,
  ridgeCandidateBlockCount,
  sourceResolutionScore,
}: {
  fingerClass?: DetectionClassName;
  globalScore: number;
  orientationCoherence: number;
  ridgePeriodicity: number;
  ridgeFrequencyConsistency: number;
  ridgeValidBlockRatio: number;
  ridgeValidBlockCount: number;
  ridgeCandidateBlockCount: number;
  sourceResolutionScore: number;
}): {
  status: FingerprintQuality['biometricStatus'];
  reasons: BiometricRejectionReason[];
} {
  const minimums = getScaleAwareEvidenceMinimums(ridgeCandidateBlockCount);
  const reasons: BiometricRejectionReason[] = [];
  const minimumBiometricScore =
    fingerClass === 'pinky'
      ? QUALITY_THRESHOLDS.biometricPinkyScore
      : QUALITY_THRESHOLDS.biometricScore;
  // Yaygın ve doğrulanmış yerel ridge akışı, kıvrımlı parmak ucunda düşük kalan global yön özetini telafi edebilir.
  const hasStrongLocalEvidence = hasStrongLocalOrientationEvidence({
    ridgePeriodicity,
    ridgeFrequencyConsistency,
    ridgeValidBlockRatio,
    ridgeValidBlockCount,
    ridgeCandidateBlockCount,
  });

  // Her başarısız koşulu ayrı saklayarak tek bir "insufficient" sonucunun nedenini görünür kılarız.
  if (globalScore < minimumBiometricScore) reasons.push('global-score');
  if (
    orientationCoherence < QUALITY_THRESHOLDS.biometricOrientation &&
    !hasStrongLocalEvidence
  ) {
    reasons.push('orientation');
  }
  if (ridgePeriodicity < QUALITY_THRESHOLDS.biometricPeriodicity) {
    reasons.push('periodicity');
  }
  if (
    ridgeFrequencyConsistency <
    QUALITY_THRESHOLDS.biometricFrequencyConsistency
  ) {
    reasons.push('frequency-consistency');
  }
  if (ridgeValidBlockRatio < QUALITY_THRESHOLDS.biometricValidRatio) {
    reasons.push('evidence-ratio');
  }
  if (ridgeValidBlockCount < minimums.biometric) {
    reasons.push('evidence-count');
  }
  if (sourceResolutionScore < QUALITY_THRESHOLDS.biometricResolution) {
    reasons.push('source-resolution');
  }

  return {
    status: reasons.length === 0 ? 'sufficient' : 'insufficient',
    reasons,
  };
}

// Eski kayıtlarda captureStatus bulunmadığında status alanına güvenli biçimde geri düşer.
function getCaptureStatus(quality: FingerprintQuality) {
  return quality.captureStatus ?? quality.status;
}

// Parmak sınıfını kısa Türkçe etikete dönüştürür.
function formatFingerClass(className: FingerRoi['className']) {
  if (className === 'index') return 'İşaret';
  if (className === 'middle') return 'Orta';
  if (className === 'ring') return 'Yüzük';
  if (className === 'pinky') return 'Serçe';
  return 'Parmak';
}

// Lokal ortalama hesabını hızlı yapmak için gri görüntünün integral toplamını çıkarır.
function createIntegralImage(grayscale: Uint8Array, width: number, height: number) {
  const integral = new Float64Array((width + 1) * (height + 1));

  for (let y = 1; y <= height; y += 1) {
    let rowTotal = 0;

    for (let x = 1; x <= width; x += 1) {
      rowTotal += grayscale[(y - 1) * width + (x - 1)];
      integral[y * (width + 1) + x] = integral[(y - 1) * (width + 1) + x] + rowTotal;
    }
  }

  return integral;
}

// Integral görüntüden dikdörtgen toplamını okuyarak lokal parlaklık ortalamasını destekler.
function readIntegralSum(
  integral: Float64Array,
  width: number,
  left: number,
  top: number,
  right: number,
  bottom: number
) {
  const stride = width + 1;
  const x1 = left;
  const y1 = top;
  const x2 = right + 1;
  const y2 = bottom + 1;

  return (
    integral[y2 * stride + x2] -
    integral[y1 * stride + x2] -
    integral[y2 * stride + x1] +
    integral[y1 * stride + x1]
  );
}

// Sıralı kopya üzerinden istenen yüzdelik değeri döndürür.
function getPercentile(values: number[], percentile: number) {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((first, second) => first - second);
  const index = (sorted.length - 1) * percentile;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const fraction = index - lower;

  return (sorted[lower] ?? 0) * (1 - fraction) + (sorted[upper] ?? 0) * fraction;
}

// Ortanca değeri yüzdelik yardımcısıyla hesaplar.
function getMedian(values: number[]) {
  return getPercentile(values, 0.5);
}

// Ham ölçümü 0-100 arası kalibre edilmiş puana çevirir.
function scoreRange(value: number, minimum: number, maximum: number) {
  return clampScore(((value - minimum) / Math.max(maximum - minimum, 1)) * 100);
}

// Skoru güvenli 0-100 aralığında tutar.
function clampScore(value: number) {
  return Math.round(Math.min(Math.max(value, 0), 100));
}

// Oranı güvenli 0-1 aralığında tutar.
function clamp01(value: number) {
  return Math.min(Math.max(value, 0), 1);
}

// Piksel değerini güvenli 8-bit gri ton aralığında tutar.
function clampByte(value: number) {
  return Math.round(Math.min(Math.max(value, 0), 255));
}
