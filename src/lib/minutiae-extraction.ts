import type {
  FingerprintMinutia,
  FingerprintTemplate,
  MinutiaeTopologyDiagnostics,
  MinutiaeTopologyStageDiagnostics,
  MinutiaType,
} from '@/types/biometrics';

type MinutiaeExtractionInput = {
  pixels: Uint8Array;
  mask: Uint8Array;
  candidateMask?: Uint8Array;
  orientationMask?: Uint8Array;
  orientationAngles?: Float32Array;
  orientationFieldMask?: Uint8Array;
  width: number;
  height: number;
  ridgePeriodPixels: number;
  minimumSupportConfidence?: number;
};

type PixelPoint = {
  x: number;
  y: number;
};

export type MinutiaCandidate = PixelPoint & {
  type: MinutiaType;
  angleDegrees: number;
  confidence: number;
};

export type MinutiaeExtractionResult = {
  template: FingerprintTemplate;
  visualizationPixels: Uint8Array;
  binaryVisualizationPixels: Uint8Array;
  openedBinaryVisualizationPixels: Uint8Array;
  skeletonOverlayVisualizationPixels: Uint8Array;
  binaryRidgeRatio: number;
  skeletonPixelCount: number;
  rawCandidateCount: number;
  thinningIterations: number;
  prunedPixelCount: number;
  bridgedPixelCount: number;
  removedComponentPixelCount: number;
  filledHolePixelCount: number;
  supportCoverage: number;
  searchableAreaRatio: number;
  largestSearchableRegionRatio: number;
  binaryNoisePixelCount: number;
  morphologicalOpeningStatus: MorphologicalOpeningStatus;
  morphologicalOpeningBeforePixelCount: number;
  morphologicalOpeningAfterPixelCount: number;
  morphologicalOpeningBeforeComponentCount: number;
  morphologicalOpeningAfterComponentCount: number;
  orientationBridgeRemovedPixelCount: number;
  crossingNumberCandidateCount: number;
  bifurcationCrossingNumberCandidateCount: number;
  bifurcationBranchValidatedCount: number;
  bifurcationRingValidatedCount: number;
  bifurcationMicroCycleValidatedCount: number;
  bifurcationStabilityValidatedCount: number;
  bifurcationSuppressionCount: number;
  branchValidatedCandidateCount: number;
  suppressionCandidateCount: number;
  endingCandidateCount: number;
  bifurcationCandidateCount: number;
  confidenceHistogram: number[];
  thresholdConsensusPrimaryCount: number;
  thresholdConsensusLocationCount: number;
  thresholdConsensusTypeCount: number;
  thresholdLoopCandidateCount: number;
  thresholdLoopStableCount: number;
  thresholdLoopRemovedCount: number;
  thresholdLoopRemovedPixelCount: number;
  topologyDiagnostics: MinutiaeTopologyDiagnostics;
};

export type MinutiaeDetectionDiagnostics = {
  crossingNumberCandidateCount: number;
  bifurcationCrossingNumberCandidateCount: number;
  bifurcationBranchValidatedCount: number;
  bifurcationRingValidatedCount: number;
  bifurcationMicroCycleValidatedCount: number;
  bifurcationStabilityValidatedCount: number;
  bifurcationSuppressionCount: number;
  branchValidatedCandidateCount: number;
  suppressionCandidateCount: number;
  endingCandidateCount: number;
  bifurcationCandidateCount: number;
  confidenceHistogram: number[];
};

// Zhang-Suen inceltmesinin bozuk görüntüde sınırsız dönmesini engeller.
const MAX_THINNING_ITERATIONS = 64;

// Tek bir ROI'nin metadata ve görselini aşırı sayıda sahte adayla doldurmayı sınırlar.
const MAX_MINUTIAE_COUNT = 80;

// Çatallanmanın üç dalını 3x3 komşuluk yerine yaklaşık üç ridge periyodu boyunca doğrular.
const BIFURCATION_BRANCH_LENGTH_PERIOD_FACTOR = 2.8;

// Aynı yuvarlak/paralel ridge bozulmasından çıkan komşu mavi adayları daha geniş alanda tekilleştirir.
const BIFURCATION_SUPPRESSION_PERIOD_FACTOR = 2;

// Gerçek çatallanmanın geniş çevrede tam üç ayrı ridge çıkışı vermesini sınayan halka yarıçapı.
const BIFURCATION_RING_RADIUS_PERIOD_FACTOR = 2.4;

// Tek eşikte oluşan kopma ve birleşmeleri ayırt etmek için ana eşik çevresinde iki küçük perturbasyon kullanılır.
const RIDGE_THRESHOLD_MULTIPLIERS = [0.5, 1, 1.5] as const;

// Doğrulanmış 8 px yön bloğunun 28° kabul payına örnekleme toleransı eklenir;
// bu sınırı aşan kopukluk ridge boyunca değil, ridge'lere enine ilerliyor sayılır.
const MAX_ORIENTATION_GAP_AXIS_DIFFERENCE_DEGREES = 32;

// Beklenen ridge genişliği 3x3 çapraz çekirdeği taşıyamayacak kadar küçükse opening uygulanmaz.
const MIN_MORPHOLOGICAL_OPENING_RIDGE_PERIOD = 6;
const MIN_MORPHOLOGICAL_OPENING_PIXEL_RETENTION = 0.82;

export type MorphologicalOpeningStatus =
  | 'applied'
  | 'skipped-small-period'
  | 'rejected-pixel-loss'
  | 'rejected-fragmentation'
  | 'no-change';

export type MorphologicalOpeningResult = {
  status: MorphologicalOpeningStatus;
  beforePixelCount: number;
  afterPixelCount: number;
  beforeComponentCount: number;
  afterComponentCount: number;
};

type ThresholdPipelineResult = {
  binary: Uint8Array;
  referenceBinary?: Uint8Array;
  skeleton: Uint8Array;
  detection: ReturnType<typeof detectMinutiaeFromSkeletonDetailed>;
  binaryNoisePixelCount: number;
  morphologicalOpening: MorphologicalOpeningResult;
  orientationBridgeRemovedPixelCount: number;
  filledHolePixelCount: number;
  thinningIterations: number;
  bridgedPixelCount: number;
  removedComponentPixelCount: number;
  prunedPixelCount: number;
  topologyDiagnostics?: MinutiaeTopologyDiagnostics;
};

export type ThresholdConsensusResult = {
  candidates: MinutiaCandidate[];
  primaryCount: number;
  locationStableCount: number;
  typeStableCount: number;
};

export type ThresholdLoopValidationResult = {
  candidateCount: number;
  stableCount: number;
  unstableCount: number;
  removedLoopCount: number;
  removedPixelCount: number;
};

type SkeletonLoop = {
  centerX: number;
  centerY: number;
  area: number;
  left: number;
  top: number;
  right: number;
  bottom: number;
  boundaryIndexes: number[];
};

// Gri ROI'den adaptif ikili ridge haritası, iskelet ve filtrelenmiş minutiae şablonu üretir.
export function extractFingerprintMinutiae({
  pixels,
  mask,
  candidateMask = mask,
  orientationMask = mask,
  orientationAngles,
  orientationFieldMask,
  width,
  height,
  ridgePeriodPixels,
  minimumSupportConfidence = 76,
}: MinutiaeExtractionInput): MinutiaeExtractionResult {
  const grayscale = rgbaToGrayscale(pixels, width, height);
  const foregroundDistance = createMaskDistanceMap(mask, width, height);
  const period = clamp(ridgePeriodPixels || 8, 5, 14);
  // Yön görselindeki yeşil hücreleri oluşturan doğrulanmış ince alan varsa,
  // daha gevşek kaba yön maskesi yerine doğrudan bu alanı zorunlu sınır yaparız.
  const verifiedOrientationMask = orientationFieldMask ?? orientationMask;
  const stableSearchArea = createStableMinutiaeSearchArea({
    supportMask: candidateMask,
    orientationMask: verifiedOrientationMask,
    foregroundMask: mask,
    width,
    height,
    ridgePeriodPixels: period,
  });
  const candidateDistance = createMaskDistanceMap(
    stableSearchArea.mask,
    width,
    height
  );
  const candidateBoundaryMargin = getCandidateBoundaryMargin(period);
  const searchableRegion = calculateSearchableRegionMetrics(
    candidateDistance,
    foregroundDistance,
    mask,
    width,
    height,
    candidateBoundaryMargin,
    getForegroundBoundaryMargin(period)
  );
  const commonPipelineInput = {
    grayscale,
    mask,
    foregroundDistance,
    stableSearchMask: stableSearchArea.mask,
    candidateDistance,
    supportConfidence: stableSearchArea.supportConfidence,
    width,
    height,
    ridgePeriodPixels: period,
    minimumSupportConfidence,
    orientationAngles,
    orientationFieldMask,
  };
  const primaryPipeline = runMinutiaeThresholdPipeline({
    ...commonPipelineInput,
    thresholdMultiplier: RIDGE_THRESHOLD_MULTIPLIERS[1],
    collectTopologyDiagnostics: true,
  });
  const relaxedPipeline = runMinutiaeThresholdPipeline({
    ...commonPipelineInput,
    thresholdMultiplier: RIDGE_THRESHOLD_MULTIPLIERS[0],
    collectTopologyDiagnostics: false,
  });
  const strictPipeline = runMinutiaeThresholdPipeline({
    ...commonPipelineInput,
    thresholdMultiplier: RIDGE_THRESHOLD_MULTIPLIERS[2],
    collectTopologyDiagnostics: false,
  });
  const thresholdLoopValidation = removeThresholdUnstableSkeletonLoops({
    skeleton: primaryPipeline.skeleton,
    variantSkeletons: [relaxedPipeline.skeleton, strictPipeline.skeleton],
    width,
    height,
    ridgePeriodPixels: period,
    orientationAngles,
    orientationMask: orientationFieldMask,
  });
  if (thresholdLoopValidation.removedPixelCount > 0) {
    primaryPipeline.removedComponentPixelCount += removeSmallSkeletonComponents(
      primaryPipeline.skeleton,
      width,
      height,
      Math.max(18, Math.round(period * 3.5))
    );
    primaryPipeline.prunedPixelCount += pruneShortSkeletonBranches(
      primaryPipeline.skeleton,
      width,
      height,
      Math.max(8, Math.round(period * 1.8))
    );
    primaryPipeline.detection = detectMinutiaeFromSkeletonDetailed({
      skeleton: primaryPipeline.skeleton,
      maskDistance: candidateDistance,
      foregroundDistance,
      width,
      height,
      ridgePeriodPixels: period,
      supportConfidence: stableSearchArea.supportConfidence,
      minimumSupportConfidence,
    });
  }
  if (primaryPipeline.topologyDiagnostics) {
    primaryPipeline.topologyDiagnostics.thresholdLoopCleaned =
      summarizeRidgeTopology(primaryPipeline.skeleton, width, height);
  }
  const thresholdConsensus = selectThresholdStableMinutiaeCandidates({
    primaryCandidates: primaryPipeline.detection.candidates,
    variantCandidates: [
      relaxedPipeline.detection.candidates,
      strictPipeline.detection.candidates,
    ],
    ridgePeriodPixels: period,
  });
  const rawCandidates = thresholdConsensus.candidates;
  const binaryVisualizationPixels = createBinaryRidgeVisualization(
    primaryPipeline.referenceBinary ?? primaryPipeline.binary,
    stableSearchArea.mask,
    width,
    height
  );
  const openedBinaryVisualizationPixels = createBinaryRidgeVisualization(
    primaryPipeline.binary,
    stableSearchArea.mask,
    width,
    height
  );
  const minutiae = rawCandidates.slice(0, MAX_MINUTIAE_COUNT).map(
    (candidate): FingerprintMinutia => ({
      x: round(candidate.x / Math.max(width - 1, 1), 5),
      y: round(candidate.y / Math.max(height - 1, 1), 5),
      angleDegrees: round(candidate.angleDegrees, 1),
      type: candidate.type,
      confidence: candidate.confidence,
    })
  );
  const template: FingerprintTemplate = {
    version: 'minutiae-v2',
    width,
    height,
    ridgePeriodPixels: round(period, 1),
    minutiae,
    coordinateFrame: 'homography-canonical',
  };

  return {
    template,
    visualizationPixels: createMinutiaeVisualization(
      primaryPipeline.skeleton,
      rawCandidates.slice(0, MAX_MINUTIAE_COUNT),
      stableSearchArea.mask,
      width,
      height
    ),
    binaryVisualizationPixels,
    openedBinaryVisualizationPixels,
    skeletonOverlayVisualizationPixels: createMinutiaeSkeletonOverlay(
      primaryPipeline.skeleton,
      stableSearchArea.mask,
      width,
      height
    ),
    binaryRidgeRatio: calculateBinaryRidgeRatio(
      primaryPipeline.binary,
      stableSearchArea.mask
    ),
    skeletonPixelCount: countEnabledPixels(primaryPipeline.skeleton),
    rawCandidateCount: rawCandidates.length,
    thinningIterations: primaryPipeline.thinningIterations,
    prunedPixelCount: primaryPipeline.prunedPixelCount,
    bridgedPixelCount: primaryPipeline.bridgedPixelCount,
    removedComponentPixelCount: primaryPipeline.removedComponentPixelCount,
    filledHolePixelCount: primaryPipeline.filledHolePixelCount,
    supportCoverage: calculateMaskCoverage(candidateMask, mask),
    searchableAreaRatio: searchableRegion.totalRatio,
    largestSearchableRegionRatio: searchableRegion.largestRegionRatio,
    binaryNoisePixelCount: primaryPipeline.binaryNoisePixelCount,
    morphologicalOpeningStatus: primaryPipeline.morphologicalOpening.status,
    morphologicalOpeningBeforePixelCount:
      primaryPipeline.morphologicalOpening.beforePixelCount,
    morphologicalOpeningAfterPixelCount:
      primaryPipeline.morphologicalOpening.afterPixelCount,
    morphologicalOpeningBeforeComponentCount:
      primaryPipeline.morphologicalOpening.beforeComponentCount,
    morphologicalOpeningAfterComponentCount:
      primaryPipeline.morphologicalOpening.afterComponentCount,
    orientationBridgeRemovedPixelCount:
      primaryPipeline.orientationBridgeRemovedPixelCount,
    topologyDiagnostics: primaryPipeline.topologyDiagnostics!,
    ...primaryPipeline.detection.diagnostics,
    confidenceHistogram: createConfidenceHistogram(rawCandidates),
    thresholdConsensusPrimaryCount: thresholdConsensus.primaryCount,
    thresholdConsensusLocationCount: thresholdConsensus.locationStableCount,
    thresholdConsensusTypeCount: thresholdConsensus.typeStableCount,
    thresholdLoopCandidateCount: thresholdLoopValidation.candidateCount,
    thresholdLoopStableCount: thresholdLoopValidation.stableCount,
    thresholdLoopRemovedCount: thresholdLoopValidation.removedLoopCount,
    thresholdLoopRemovedPixelCount: thresholdLoopValidation.removedPixelCount,
  };
}

// Aynı ROI'yi belirli bir adaptif eşik perturbasyonuyla baştan sona işleyerek karşılaştırılabilir adaylar üretir.
function runMinutiaeThresholdPipeline({
  grayscale,
  mask,
  foregroundDistance,
  stableSearchMask,
  candidateDistance,
  supportConfidence,
  width,
  height,
  ridgePeriodPixels,
  minimumSupportConfidence,
  orientationAngles,
  orientationFieldMask,
  thresholdMultiplier,
  collectTopologyDiagnostics,
}: {
  grayscale: Uint8Array;
  mask: Uint8Array;
  foregroundDistance: Uint16Array;
  stableSearchMask: Uint8Array;
  candidateDistance: Uint16Array;
  supportConfidence: Uint8Array;
  width: number;
  height: number;
  ridgePeriodPixels: number;
  minimumSupportConfidence: number;
  orientationAngles?: Float32Array;
  orientationFieldMask?: Uint8Array;
  thresholdMultiplier: number;
  collectTopologyDiagnostics: boolean;
}): ThresholdPipelineResult {
  const binary = createAdaptiveRidgeBinary(
    grayscale,
    mask,
    foregroundDistance,
    width,
    height,
    ridgePeriodPixels,
    thresholdMultiplier
  );
  // Güvensiz yön bölgelerindeki koyu lekeleri topolojiye hiç sokmayız. Yalnızca
  // görseli kırpmak yeterli değildir; aksi halde inceltme bu lekeleri bağlayabilir.
  restrictBinaryToMask(binary, stableSearchMask);
  const binaryNoisePixelCount = removeSmallBinaryComponents(
    binary,
    width,
    height,
    Math.max(12, Math.round(ridgePeriodPixels * 2))
  );
  cleanBinaryRidges(binary, candidateDistance, width, height);
  restrictBinaryToMask(binary, stableSearchMask);
  const filledHolePixelCount = fillSmallBinaryRidgeHoles(
    binary,
    candidateDistance,
    width,
    height,
    ridgePeriodPixels
  );
  restrictBinaryToMask(binary, stableSearchMask);
  const binaryTopology = collectTopologyDiagnostics
    ? summarizeRidgeTopology(binary, width, height)
    : undefined;
  const referenceBinary = collectTopologyDiagnostics
    ? new Uint8Array(binary)
    : undefined;
  const morphologicalOpening = applyControlledBinaryOpening({
    binary,
    width,
    height,
    ridgePeriodPixels,
    allowedMask: stableSearchMask,
  });
  const openedTopology = collectTopologyDiagnostics
    ? summarizeRidgeTopology(binary, width, height)
    : undefined;
  const orientationBridgeRemovedPixelCount =
    orientationAngles && orientationFieldMask
      ? removeOrientationInconsistentBinaryBridges({
          binary,
          orientationAngles,
          orientationMask: orientationFieldMask,
          maskDistance: foregroundDistance,
          width,
          height,
          ridgePeriodPixels,
        })
      : 0;
  if (referenceBinary && orientationAngles && orientationFieldMask) {
    removeOrientationInconsistentBinaryBridges({
      binary: referenceBinary,
      orientationAngles,
      orientationMask: orientationFieldMask,
      maskDistance: foregroundDistance,
      width,
      height,
      ridgePeriodPixels,
    });
  }
  const orientationCleanedTopology = collectTopologyDiagnostics
    ? summarizeRidgeTopology(binary, width, height)
    : undefined;
  const thinning = thinRidgesZhangSuen(binary, width, height);
  const thinnedTopology = collectTopologyDiagnostics
    ? summarizeRidgeTopology(thinning.skeleton, width, height)
    : undefined;
  const bridgedPixelCount = bridgeShortSkeletonGaps(
    thinning.skeleton,
    stableSearchMask,
    width,
    height,
    ridgePeriodPixels,
    orientationAngles,
    orientationFieldMask
  );
  const bridgedTopology = collectTopologyDiagnostics
    ? summarizeRidgeTopology(thinning.skeleton, width, height)
    : undefined;
  const removedComponentPixelCount = removeSmallSkeletonComponents(
    thinning.skeleton,
    width,
    height,
    Math.max(18, Math.round(ridgePeriodPixels * 3.5))
  );
  const componentFilteredTopology = collectTopologyDiagnostics
    ? summarizeRidgeTopology(thinning.skeleton, width, height)
    : undefined;
  const prunedPixelCount = pruneShortSkeletonBranches(
    thinning.skeleton,
    width,
    height,
    Math.max(8, Math.round(ridgePeriodPixels * 1.8))
  );
  const prunedTopology = collectTopologyDiagnostics
    ? summarizeRidgeTopology(thinning.skeleton, width, height)
    : undefined;
  const detection = detectMinutiaeFromSkeletonDetailed({
    skeleton: thinning.skeleton,
    maskDistance: candidateDistance,
    foregroundDistance,
    width,
    height,
    ridgePeriodPixels,
    supportConfidence,
    minimumSupportConfidence,
  });

  return {
    binary,
    referenceBinary,
    skeleton: thinning.skeleton,
    detection,
    binaryNoisePixelCount,
    morphologicalOpening,
    orientationBridgeRemovedPixelCount,
    filledHolePixelCount,
    thinningIterations: thinning.iterations,
    bridgedPixelCount,
    removedComponentPixelCount,
    prunedPixelCount,
    topologyDiagnostics:
      binaryTopology &&
      openedTopology &&
      orientationCleanedTopology &&
      thinnedTopology &&
      bridgedTopology &&
      componentFilteredTopology &&
      prunedTopology
          ? {
            binary: binaryTopology,
            opened: openedTopology,
            orientationCleaned: orientationCleanedTopology,
            thinned: thinnedTopology,
            bridged: bridgedTopology,
            componentFiltered: componentFilteredTopology,
            pruned: prunedTopology,
          }
        : undefined,
  };
}

// Tek geçiş 3x3 çapraz erosion+dilation uygular; gerçek ridge alanını aşırı
// azaltan veya görüntüyü çok fazla parçaya bölen sonucu otomatik olarak geri alır.
export function applyControlledBinaryOpening({
  binary,
  width,
  height,
  ridgePeriodPixels,
  allowedMask,
}: {
  binary: Uint8Array;
  width: number;
  height: number;
  ridgePeriodPixels: number;
  allowedMask?: Uint8Array;
}): MorphologicalOpeningResult {
  const beforePixelCount = countEnabledPixels(binary);
  const beforeComponentCount = countEnabledComponents(binary, width, height);
  const unchangedResult = (
    status: MorphologicalOpeningStatus
  ): MorphologicalOpeningResult => ({
    status,
    beforePixelCount,
    afterPixelCount: beforePixelCount,
    beforeComponentCount,
    afterComponentCount: beforeComponentCount,
  });

  if (ridgePeriodPixels < MIN_MORPHOLOGICAL_OPENING_RIDGE_PERIOD) {
    return unchangedResult('skipped-small-period');
  }

  const eroded = erodeBinaryWithCrossKernel(binary, width, height);
  const opened = dilateBinaryWithCrossKernel(eroded, width, height);
  if (allowedMask) restrictBinaryToMask(opened, allowedMask);
  const afterPixelCount = countEnabledPixels(opened);
  const afterComponentCount = countEnabledComponents(opened, width, height);
  if (afterPixelCount === beforePixelCount) {
    return unchangedResult('no-change');
  }

  const retainedRatio = afterPixelCount / Math.max(beforePixelCount, 1);
  if (retainedRatio < MIN_MORPHOLOGICAL_OPENING_PIXEL_RETENTION) {
    return unchangedResult('rejected-pixel-loss');
  }
  const maximumComponentCount = Math.max(
    beforeComponentCount * 3,
    beforeComponentCount + 32
  );
  if (afterComponentCount > maximumComponentCount) {
    return unchangedResult('rejected-fragmentation');
  }

  binary.set(opened);
  return {
    status: 'applied',
    beforePixelCount,
    afterPixelCount,
    beforeComponentCount,
    afterComponentCount,
  };
}

// Binary ridge tamponunu doğrulanmış yön alanıyla kesiştirir; maske sınırı
// dışında kalan pikseller daha sonraki morfoloji ve inceltme adımlarına ulaşmaz.
function restrictBinaryToMask(binary: Uint8Array, allowedMask: Uint8Array) {
  if (binary.length !== allowedMask.length) {
    throw new RangeError('Binary ridge ve yön güven maskesi boyutları uyuşmuyor.');
  }

  for (let index = 0; index < binary.length; index += 1) {
    if (!allowedMask[index]) binary[index] = 0;
  }
}

function erodeBinaryWithCrossKernel(
  binary: Uint8Array,
  width: number,
  height: number
) {
  const eroded = new Uint8Array(binary.length);
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const index = y * width + x;
      if (
        binary[index] &&
        binary[index - 1] &&
        binary[index + 1] &&
        binary[index - width] &&
        binary[index + width]
      ) {
        eroded[index] = 1;
      }
    }
  }
  return eroded;
}

function dilateBinaryWithCrossKernel(
  binary: Uint8Array,
  width: number,
  height: number
) {
  const dilated = new Uint8Array(binary.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      if (!binary[index]) continue;
      dilated[index] = 1;
      if (x > 0) dilated[index - 1] = 1;
      if (x + 1 < width) dilated[index + 1] = 1;
      if (y > 0) dilated[index - width] = 1;
      if (y + 1 < height) dilated[index + width] = 1;
    }
  }
  return dilated;
}

// Tek bir eşikte görünen topoloji yerine, yakın eşiklerde aynı konum ve tipte tekrar eden adayları template'e alır.
export function selectThresholdStableMinutiaeCandidates({
  primaryCandidates,
  variantCandidates,
  ridgePeriodPixels,
}: {
  primaryCandidates: MinutiaCandidate[];
  variantCandidates: MinutiaCandidate[][];
  ridgePeriodPixels: number;
}): ThresholdConsensusResult {
  const locationHits = new Uint8Array(primaryCandidates.length);
  const typeHits = new Uint8Array(primaryCandidates.length);
  locationHits.fill(1);
  typeHits.fill(1);
  const maximumDistance = Math.max(3, ridgePeriodPixels * 0.7);

  for (const candidates of variantCandidates) {
    for (const index of matchConsensusCandidates(
      primaryCandidates,
      candidates,
      maximumDistance,
      false
    )) {
      locationHits[index] += 1;
    }
    for (const index of matchConsensusCandidates(
      primaryCandidates,
      candidates,
      maximumDistance,
      true
    )) {
      typeHits[index] += 1;
    }
  }

  let locationStableCount = 0;
  let typeStableCount = 0;
  const requiredFullConsensus = variantCandidates.length + 1;
  const candidates = primaryCandidates.flatMap((candidate, index) => {
    const locationStable = locationHits[index] >= 2;
    if (locationStable) locationStableCount += 1;
    const requiredTypeHits =
      candidate.type === 'bifurcation' ? requiredFullConsensus : 2;
    const typeStable = locationStable && typeHits[index] >= requiredTypeHits;
    if (!typeStable) return [];
    typeStableCount += 1;
    return [{
      ...candidate,
      confidence: Math.round(
        clamp(
          candidate.confidence +
            (typeHits[index] === requiredFullConsensus ? 4 : -2),
          0,
          100
        )
      ),
    }];
  });

  return {
    candidates,
    primaryCount: primaryCandidates.length,
    locationStableCount,
    typeStableCount,
  };
}

// Ana eşikte oluşan küçük kapalı çevrimleri komşu eşiklerde arar. En az iki
// eşikte tekrarlanan çevrim korunur; yalnızca sıkı eşikte desteklenmeyen ve
// güvenilir ridge yönüne belirgin biçimde aykırı bağlantılar yerel olarak açılır.
export function removeThresholdUnstableSkeletonLoops({
  skeleton,
  variantSkeletons,
  width,
  height,
  ridgePeriodPixels,
  orientationAngles,
  orientationMask,
}: {
  skeleton: Uint8Array;
  variantSkeletons: Uint8Array[];
  width: number;
  height: number;
  ridgePeriodPixels: number;
  orientationAngles?: Float32Array;
  orientationMask?: Uint8Array;
}): ThresholdLoopValidationResult {
  const emptyResult: ThresholdLoopValidationResult = {
    candidateCount: 0,
    stableCount: 0,
    unstableCount: 0,
    removedLoopCount: 0,
    removedPixelCount: 0,
  };
  const pixelCount = width * height;
  if (
    width < 3 ||
    height < 3 ||
    skeleton.length !== pixelCount ||
    variantSkeletons.some((variant) => variant.length !== pixelCount)
  ) {
    return emptyResult;
  }

  const period = clamp(ridgePeriodPixels, 5, 14);
  const primaryLoops = findSmallSkeletonLoops(
    skeleton,
    width,
    height,
    period
  );
  const variantLoops = variantSkeletons.map((variant) =>
    findSmallSkeletonLoops(variant, width, height, period)
  );
  const stableLoops = primaryLoops.filter((loop) =>
    variantLoops.some((loops) =>
      loops.some((candidate) => skeletonLoopsMatch(loop, candidate, period))
    )
  );
  const unstableLoops = primaryLoops.filter(
    (loop) => !stableLoops.includes(loop)
  );

  if (
    unstableLoops.length === 0 ||
    !orientationAngles ||
    !orientationMask ||
    orientationAngles.length !== pixelCount ||
    orientationMask.length !== pixelCount ||
    variantSkeletons.length === 0
  ) {
    return {
      candidateCount: primaryLoops.length,
      stableCount: stableLoops.length,
      unstableCount: unstableLoops.length,
      removedLoopCount: 0,
      removedPixelCount: 0,
    };
  }

  const strictSkeleton = variantSkeletons[variantSkeletons.length - 1];
  const relaxedSkeleton = variantSkeletons[0];
  const maximumCutsPerLoop = Math.max(1, Math.round(period * 0.35));
  let removedLoopCount = 0;
  let removedPixelCount = 0;

  for (const originalLoop of unstableLoops) {
    let currentLoop = findMatchingSkeletonLoop(
      skeleton,
      width,
      height,
      period,
      originalLoop
    );
    if (!currentLoop) continue;

    const tentativeRemovals: number[] = [];
    for (let attempt = 0; attempt < maximumCutsPerLoop; attempt += 1) {
      const cutIndex = selectUnstableLoopCutPixel({
        loop: currentLoop,
        skeleton,
        relaxedSkeleton,
        strictSkeleton,
        orientationAngles,
        orientationMask,
        width,
        height,
        ridgePeriodPixels: period,
      });
      if (cutIndex === undefined) break;

      skeleton[cutIndex] = 0;
      tentativeRemovals.push(cutIndex);
      currentLoop = findMatchingSkeletonLoop(
        skeleton,
        width,
        height,
        period,
        originalLoop
      );
      if (!currentLoop) break;
    }

    if (currentLoop) {
      // Tek bir pikseli kesip topolojiyi değiştiremediysek yarım müdahaleyi geri alırız.
      for (const index of tentativeRemovals) skeleton[index] = 1;
      continue;
    }

    removedLoopCount += 1;
    removedPixelCount += tentativeRemovals.length;
  }

  return {
    candidateCount: primaryLoops.length,
    stableCount: stableLoops.length,
    unstableCount: unstableLoops.length,
    removedLoopCount,
    removedPixelCount,
  };
}

// İskeletin dört-komşulukla dışarı ulaşamayan arka plan bileşenlerini küçük
// çevrim içleri olarak bulur. Büyük gerçek enclosure'lar bu yerel filtreden çıkarılır.
function findSmallSkeletonLoops(
  skeleton: Uint8Array,
  width: number,
  height: number,
  ridgePeriodPixels: number
) {
  const visited = new Uint8Array(skeleton.length);
  const queue = new Int32Array(skeleton.length);
  const loops: SkeletonLoop[] = [];
  const minimumArea = Math.max(
    2,
    Math.round(ridgePeriodPixels * ridgePeriodPixels * 0.04)
  );
  const maximumArea = Math.max(
    20,
    Math.round(ridgePeriodPixels * ridgePeriodPixels * 2.5)
  );
  const maximumSpan = Math.max(8, Math.round(ridgePeriodPixels * 2.75));
  const minimumBoundaryPixels = Math.max(6, Math.round(ridgePeriodPixels * 0.75));
  const cardinalOffsets = [
    [0, -1],
    [1, 0],
    [0, 1],
    [-1, 0],
  ] as const;

  for (let startIndex = 0; startIndex < skeleton.length; startIndex += 1) {
    if (skeleton[startIndex] || visited[startIndex]) continue;
    let queueStart = 0;
    let queueEnd = 1;
    queue[0] = startIndex;
    visited[startIndex] = 1;
    let enclosed = true;
    let minimumX = width;
    let maximumX = -1;
    let minimumY = height;
    let maximumY = -1;
    let coordinateTotalX = 0;
    let coordinateTotalY = 0;

    while (queueStart < queueEnd) {
      const index = queue[queueStart];
      queueStart += 1;
      const x = index % width;
      const y = Math.floor(index / width);
      minimumX = Math.min(minimumX, x);
      maximumX = Math.max(maximumX, x);
      minimumY = Math.min(minimumY, y);
      maximumY = Math.max(maximumY, y);
      coordinateTotalX += x;
      coordinateTotalY += y;
      if (x === 0 || x === width - 1 || y === 0 || y === height - 1) {
        enclosed = false;
      }

      for (const [deltaX, deltaY] of cardinalOffsets) {
        const nextX = x + deltaX;
        const nextY = y + deltaY;
        if (
          nextX < 0 ||
          nextX >= width ||
          nextY < 0 ||
          nextY >= height
        ) {
          continue;
        }
        const nextIndex = nextY * width + nextX;
        if (skeleton[nextIndex] || visited[nextIndex]) continue;
        visited[nextIndex] = 1;
        queue[queueEnd] = nextIndex;
        queueEnd += 1;
      }
    }

    const componentWidth = maximumX - minimumX + 1;
    const componentHeight = maximumY - minimumY + 1;
    if (
      !enclosed ||
      queueEnd < minimumArea ||
      queueEnd > maximumArea ||
      componentWidth > maximumSpan ||
      componentHeight > maximumSpan
    ) {
      continue;
    }

    const boundary = new Set<number>();
    for (let componentIndex = 0; componentIndex < queueEnd; componentIndex += 1) {
      const index = queue[componentIndex];
      const x = index % width;
      const y = Math.floor(index / width);
      for (let deltaY = -1; deltaY <= 1; deltaY += 1) {
        for (let deltaX = -1; deltaX <= 1; deltaX += 1) {
          if (deltaX === 0 && deltaY === 0) continue;
          const nextX = x + deltaX;
          const nextY = y + deltaY;
          if (
            nextX < 0 ||
            nextX >= width ||
            nextY < 0 ||
            nextY >= height
          ) {
            continue;
          }
          const nextIndex = nextY * width + nextX;
          if (skeleton[nextIndex]) boundary.add(nextIndex);
        }
      }
    }
    if (boundary.size < minimumBoundaryPixels) continue;

    loops.push({
      centerX: coordinateTotalX / queueEnd,
      centerY: coordinateTotalY / queueEnd,
      area: queueEnd,
      left: minimumX,
      top: minimumY,
      right: maximumX,
      bottom: maximumY,
      boundaryIndexes: [...boundary],
    });
  }

  return loops;
}

function skeletonLoopsMatch(
  first: SkeletonLoop,
  second: SkeletonLoop,
  ridgePeriodPixels: number
) {
  const centerDistance = Math.hypot(
    first.centerX - second.centerX,
    first.centerY - second.centerY
  );
  const areaRatio =
    Math.max(first.area, second.area) / Math.max(1, Math.min(first.area, second.area));
  const firstWidth = first.right - first.left + 1;
  const firstHeight = first.bottom - first.top + 1;
  const secondWidth = second.right - second.left + 1;
  const secondHeight = second.bottom - second.top + 1;
  const maximumSpanDifference = Math.max(4, ridgePeriodPixels * 1.25);
  return (
    centerDistance <= Math.max(3, ridgePeriodPixels * 0.85) &&
    areaRatio <= 4 &&
    Math.abs(firstWidth - secondWidth) <= maximumSpanDifference &&
    Math.abs(firstHeight - secondHeight) <= maximumSpanDifference
  );
}

function findMatchingSkeletonLoop(
  skeleton: Uint8Array,
  width: number,
  height: number,
  ridgePeriodPixels: number,
  target: SkeletonLoop
) {
  return findSmallSkeletonLoops(
    skeleton,
    width,
    height,
    ridgePeriodPixels
  ).find((loop) => skeletonLoopsMatch(loop, target, ridgePeriodPixels));
}

function selectUnstableLoopCutPixel({
  loop,
  skeleton,
  relaxedSkeleton,
  strictSkeleton,
  orientationAngles,
  orientationMask,
  width,
  height,
  ridgePeriodPixels,
}: {
  loop: SkeletonLoop;
  skeleton: Uint8Array;
  relaxedSkeleton: Uint8Array;
  strictSkeleton: Uint8Array;
  orientationAngles: Float32Array;
  orientationMask: Uint8Array;
  width: number;
  height: number;
  ridgePeriodPixels: number;
}) {
  const candidates: { index: number; score: number }[] = [];
  const localRadius = Math.max(2, Math.round(ridgePeriodPixels * 0.35));

  for (const index of loop.boundaryIndexes) {
    if (!skeleton[index] || !orientationMask[index]) continue;
    const expectedRadians = orientationAngles[index];
    if (!Number.isFinite(expectedRadians)) continue;
    const x = index % width;
    const y = Math.floor(index / width);
    if (hasSkeletonPixelNear(strictSkeleton, width, height, x, y, 1)) {
      continue;
    }

    const neighborCount = countNeighbors(skeleton, width, x, y);
    if (neighborCount < 2 || neighborCount > 4) continue;
    const localAxis = estimateSkeletonAxis(
      skeleton,
      width,
      height,
      x,
      y,
      localRadius
    );
    const expectedAxis = normalizeAxisDegrees(
      (expectedRadians * 180) / Math.PI
    );
    const orientationMismatch = axisDifferenceDegrees(localAxis, expectedAxis);
    if (orientationMismatch < 40) continue;

    const relaxedUnsupported = !hasSkeletonPixelNear(
      relaxedSkeleton,
      width,
      height,
      x,
      y,
      1
    );
    candidates.push({
      index,
      score:
        orientationMismatch +
        (relaxedUnsupported ? 20 : 0) -
        Math.abs(neighborCount - 2) * 8,
    });
  }

  candidates.sort((first, second) => second.score - first.score);
  return candidates[0]?.index;
}

function hasSkeletonPixelNear(
  skeleton: Uint8Array,
  width: number,
  height: number,
  centerX: number,
  centerY: number,
  radius: number
) {
  for (
    let y = Math.max(0, centerY - radius);
    y <= Math.min(height - 1, centerY + radius);
    y += 1
  ) {
    for (
      let x = Math.max(0, centerX - radius);
      x <= Math.min(width - 1, centerX + radius);
      x += 1
    ) {
      if (skeleton[y * width + x]) return true;
    }
  }
  return false;
}

// Bir perturbasyon adayının birden fazla ana adayı desteklememesi için en yakın çiftleri açgözlü biçimde tekilleştirir.
function matchConsensusCandidates(
  primaryCandidates: MinutiaCandidate[],
  candidates: MinutiaCandidate[],
  maximumDistance: number,
  requireSameType: boolean
) {
  const pairs: { primaryIndex: number; candidateIndex: number; distance: number }[] = [];
  for (let primaryIndex = 0; primaryIndex < primaryCandidates.length; primaryIndex += 1) {
    const primary = primaryCandidates[primaryIndex];
    for (let candidateIndex = 0; candidateIndex < candidates.length; candidateIndex += 1) {
      const candidate = candidates[candidateIndex];
      if (requireSameType && primary.type !== candidate.type) continue;
      const candidateDistance = Math.hypot(
        primary.x - candidate.x,
        primary.y - candidate.y
      );
      if (candidateDistance <= maximumDistance) {
        pairs.push({ primaryIndex, candidateIndex, distance: candidateDistance });
      }
    }
  }
  pairs.sort((first, second) => first.distance - second.distance);

  const matchedPrimary = new Set<number>();
  const matchedCandidates = new Set<number>();
  for (const pair of pairs) {
    if (
      matchedPrimary.has(pair.primaryIndex) ||
      matchedCandidates.has(pair.candidateIndex)
    ) {
      continue;
    }
    matchedPrimary.add(pair.primaryIndex);
    matchedCandidates.add(pair.candidateIndex);
  }
  return matchedPrimary;
}

// Template'e gerçekten giren kararlı adayların güven dağılımını raporlar.
function createConfidenceHistogram(candidates: MinutiaCandidate[]) {
  const histogram = new Array<number>(10).fill(0);
  for (const candidate of candidates) {
    histogram[Math.min(9, Math.floor(candidate.confidence / 10))] += 1;
  }
  return histogram;
}

// Thinning öncesi ridge bantlarını referans görüntüyle aynı siyah ridge/beyaz zemin biçiminde gösterir.
function createBinaryRidgeVisualization(
  binary: Uint8Array,
  mask: Uint8Array,
  width: number,
  height: number
) {
  const pixels = new Uint8Array(width * height);
  pixels.fill(255);
  for (let index = 0; index < pixels.length; index += 1) {
    if (mask[index] && binary[index]) pixels[index] = 0;
  }
  return pixels;
}

// Nokta görüntüsündeki inceltilmiş ridge iskeletini kanonik ROI üzerine bindirmek için
// arka planı tamamen şeffaf, koordinatları değişmemiş beyaz bir RGBA katman üretir.
function createMinutiaeSkeletonOverlay(
  skeleton: Uint8Array,
  mask: Uint8Array,
  width: number,
  height: number
) {
  const pixels = new Uint8Array(width * height * 4);
  for (let index = 0; index < skeleton.length; index += 1) {
    if (!mask[index] || !skeleton[index]) continue;
    const offset = index * 4;
    pixels[offset] = 255;
    pixels[offset + 1] = 255;
    pixels[offset + 2] = 255;
    pixels[offset + 3] = 255;
  }
  return pixels;
}

// Parmak tabanındaki eklem/kıvrım bölgesini minutiae aramasından ayıran geçici distal maske.
export function createDistalMinutiaeMask(
  mask: Uint8Array,
  width: number,
  height: number,
  retainedHeightRatio = 0.84
) {
  const distalMask = new Uint8Array(mask);
  let minimumY = height;
  let maximumY = -1;

  for (let index = 0; index < mask.length; index += 1) {
    if (!mask[index]) continue;
    const y = Math.floor(index / width);
    minimumY = Math.min(minimumY, y);
    maximumY = Math.max(maximumY, y);
  }

  if (maximumY < minimumY) return distalMask;
  const retainedMaximumY = Math.floor(
    minimumY + (maximumY - minimumY) * retainedHeightRatio
  );

  for (let y = retainedMaximumY + 1; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      distalMask[y * width + x] = 0;
    }
  }

  return distalMask;
}

// Ridge bantları ve iskelet için bileşen, uç ve çatallanma piksel sayılarını aynı ölçekte raporlar.
function summarizeRidgeTopology(
  image: Uint8Array,
  width: number,
  height: number
): MinutiaeTopologyStageDiagnostics {
  let pixelCount = 0;
  let endingPixelCount = 0;
  let bifurcationPixelCount = 0;

  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const index = y * width + x;
      if (!image[index]) continue;
      pixelCount += 1;
      const crossingNumber = calculateCrossingNumber(
        getClockwiseNeighbors(image, width, x, y)
      );
      if (crossingNumber === 1) endingPixelCount += 1;
      if (crossingNumber === 3) bifurcationPixelCount += 1;
    }
  }

  return {
    pixelCount,
    componentCount: countEnabledComponents(image, width, height),
    endingPixelCount,
    bifurcationPixelCount,
  };
}

// Sekizli komşuluk bileşen sayısı, ridge kopukluğunu yalnızca piksel oranından bağımsız gösterir.
function countEnabledComponents(
  image: Uint8Array,
  width: number,
  height: number
) {
  const visited = new Uint8Array(image.length);
  const queue = new Int32Array(image.length);
  let componentCount = 0;

  for (let startIndex = 0; startIndex < image.length; startIndex += 1) {
    if (!image[startIndex] || visited[startIndex]) continue;
    componentCount += 1;
    let queueStart = 0;
    let queueEnd = 1;
    queue[0] = startIndex;
    visited[startIndex] = 1;

    while (queueStart < queueEnd) {
      const index = queue[queueStart];
      queueStart += 1;
      const x = index % width;
      const y = Math.floor(index / width);

      for (let deltaY = -1; deltaY <= 1; deltaY += 1) {
        for (let deltaX = -1; deltaX <= 1; deltaX += 1) {
          if (deltaX === 0 && deltaY === 0) continue;
          const nextX = x + deltaX;
          const nextY = y + deltaY;
          if (
            nextX < 0 ||
            nextX >= width ||
            nextY < 0 ||
            nextY >= height
          ) {
            continue;
          }
          const nextIndex = nextY * width + nextX;
          if (!image[nextIndex] || visited[nextIndex]) continue;
          visited[nextIndex] = 1;
          queue[queueEnd] = nextIndex;
          queueEnd += 1;
        }
      }
    }
  }

  return componentCount;
}

// Testlerde ve ileride template doğrulamasında kullanılmak üzere hazır iskeletten aday çıkarır.
export function detectMinutiaeFromSkeleton({
  skeleton,
  maskDistance,
  foregroundDistance = maskDistance,
  supportConfidence,
  width,
  height,
  ridgePeriodPixels,
  minimumSupportConfidence = 76,
}: {
  skeleton: Uint8Array;
  maskDistance: Uint16Array;
  foregroundDistance?: Uint16Array;
  supportConfidence?: Uint8Array;
  width: number;
  height: number;
  ridgePeriodPixels: number;
  minimumSupportConfidence?: number;
}) {
  return detectMinutiaeFromSkeletonDetailed({
    skeleton,
    maskDistance,
    foregroundDistance,
    supportConfidence,
    width,
    height,
    ridgePeriodPixels,
    minimumSupportConfidence,
  }).candidates;
}

export function detectMinutiaeFromSkeletonDetailed({
  skeleton,
  maskDistance,
  foregroundDistance = maskDistance,
  supportConfidence,
  width,
  height,
  ridgePeriodPixels,
  minimumSupportConfidence = 76,
}: {
  skeleton: Uint8Array;
  maskDistance: Uint16Array;
  foregroundDistance?: Uint16Array;
  supportConfidence?: Uint8Array;
  width: number;
  height: number;
  ridgePeriodPixels: number;
  minimumSupportConfidence?: number;
}) {
  const candidates: MinutiaCandidate[] = [];
  let crossingNumberCandidateCount = 0;
  let bifurcationCrossingNumberCandidateCount = 0;
  let bifurcationBranchValidatedCount = 0;
  let bifurcationRingValidatedCount = 0;
  let branchValidatedCandidateCount = 0;
  let endingCandidateCount = 0;
  let bifurcationCandidateCount = 0;
  const boundaryMargin = getCandidateBoundaryMargin(ridgePeriodPixels);
  const foregroundBoundaryMargin =
    getForegroundBoundaryMargin(ridgePeriodPixels);

  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const index = y * width + x;
      if (
        !skeleton[index] ||
        maskDistance[index] < boundaryMargin ||
        foregroundDistance[index] < foregroundBoundaryMargin
      ) {
        continue;
      }

      const neighbors = getClockwiseNeighbors(skeleton, width, x, y);
      const crossingNumber = calculateCrossingNumber(neighbors);
      const type =
        crossingNumber === 1
          ? 'ending'
          : crossingNumber === 3
            ? 'bifurcation'
            : null;
      if (!type) continue;
      crossingNumberCandidateCount += 1;
      if (type === 'bifurcation') {
        bifurcationCrossingNumberCandidateCount += 1;
      }

      const minimumBranchLength =
        type === 'bifurcation'
          ? Math.max(
              14,
              Math.round(
                ridgePeriodPixels *
                  BIFURCATION_BRANCH_LENGTH_PERIOD_FACTOR
              )
            )
          : Math.max(9, Math.round(ridgePeriodPixels * 1.7));
      const branchCount = countSustainedBranches(
        skeleton,
        width,
        height,
        x,
        y,
        minimumBranchLength
      );
      if (
        (type === 'ending' && branchCount < 1) ||
        (type === 'bifurcation' && branchCount < 3)
      ) {
        continue;
      }
      if (type === 'bifurcation') bifurcationBranchValidatedCount += 1;
      if (
        type === 'bifurcation' &&
        !hasStableBifurcationRingTopology(
          skeleton,
          width,
          height,
          x,
          y,
          ridgePeriodPixels
        )
      ) {
        continue;
      }
      const localSupportConfidence =
        (supportConfidence?.[index] ?? 255) / 255;
      const confidence = Math.round(
        clamp(
          44 +
            Math.min(maskDistance[index] - boundaryMargin, 8) * 1.2 +
            Math.min(branchCount, 3) * 4 +
            localSupportConfidence * 32,
          0,
          100
        )
      );
      // Genişletilmiş arama alanının dış kuşağındaki zayıf topoloji, template'e aday olarak taşınmaz.
      if (
        supportConfidence &&
        confidence < minimumSupportConfidence
      ) {
        continue;
      }
      if (type === 'bifurcation') bifurcationRingValidatedCount += 1;
      branchValidatedCandidateCount += 1;
      if (type === 'ending') endingCandidateCount += 1;
      if (type === 'bifurcation') bifurcationCandidateCount += 1;

      candidates.push({
        x,
        y,
        type,
        angleDegrees: estimateSkeletonAxis(
          skeleton,
          width,
          height,
          x,
          y,
          Math.max(5, Math.round(ridgePeriodPixels))
        ),
        confidence,
      });
    }
  }

  const microCycleValidatedCandidates = rejectShortBifurcationCycles({
    candidates,
    skeleton,
    width,
    height,
    ridgePeriodPixels,
  });
  const bifurcationMicroCycleValidatedCount =
    microCycleValidatedCandidates.filter(
      (candidate) => candidate.type === 'bifurcation'
    ).length;
  const stabilityValidatedCandidates = rejectUnstableCandidatePairs(
    microCycleValidatedCandidates,
    ridgePeriodPixels
  );
  const bifurcationStabilityValidatedCount =
    stabilityValidatedCandidates.filter(
      (candidate) => candidate.type === 'bifurcation'
    ).length;
  const suppressedCandidates = suppressNearbyCandidates(
    stabilityValidatedCandidates,
    ridgePeriodPixels
  );
  const bifurcationSuppressionCount = suppressedCandidates.filter(
    (candidate) => candidate.type === 'bifurcation'
  ).length;
  const confidenceHistogram = new Array<number>(10).fill(0);
  for (const candidate of suppressedCandidates) {
    confidenceHistogram[Math.min(9, Math.floor(candidate.confidence / 10))] += 1;
  }

  return {
    candidates: suppressedCandidates,
    diagnostics: {
      crossingNumberCandidateCount,
      bifurcationCrossingNumberCandidateCount,
      bifurcationBranchValidatedCount,
      bifurcationRingValidatedCount,
      bifurcationMicroCycleValidatedCount,
      bifurcationStabilityValidatedCount,
      bifurcationSuppressionCount,
      branchValidatedCandidateCount,
      suppressionCandidateCount: suppressedCandidates.length,
      endingCandidateCount,
      bifurcationCandidateCount,
      confidenceHistogram,
    },
  };
}

// Parçalı kalite maskesinin sınırında kesilmiş ridge'lerin sahte son sayılmasını önleyen güvenlik payını hesaplar.
function getCandidateBoundaryMargin(ridgePeriodPixels: number) {
  return Math.max(4, Math.round(ridgePeriodPixels * 0.75));
}

// Parmak silüeti dışında oluşan kontur ve parlama kenarlarını aday merkezinden uzak tutar.
function getForegroundBoundaryMargin(ridgePeriodPixels: number) {
  return Math.max(5, Math.round(ridgePeriodPixels * 0.75));
}

// Sert Gabor adacıklarını güvenilir yön alanı içinde genişletip daha kararlı bir minutiae arama bölgesi üretir.
export function createStableMinutiaeSearchArea({
  supportMask,
  orientationMask,
  foregroundMask,
  width,
  height,
  ridgePeriodPixels,
}: {
  supportMask: Uint8Array;
  orientationMask: Uint8Array;
  foregroundMask: Uint8Array;
  width: number;
  height: number;
  ridgePeriodPixels: number;
}) {
  const pixelCount = width * height;
  const maximumDistance = Math.max(
    6,
    Math.min(18, Math.round(ridgePeriodPixels * 1.5))
  );
  const unvisited = 65535;
  const distance = new Uint16Array(pixelCount);
  distance.fill(unvisited);
  const queue = new Int32Array(pixelCount);
  let queueStart = 0;
  let queueEnd = 0;

  // Gerçek Gabor desteğini büyümenin çekirdeği olarak kullanırız; yönü güvensiz alana taşmayız.
  for (let index = 0; index < pixelCount; index += 1) {
    if (
      supportMask[index] &&
      orientationMask[index] &&
      foregroundMask[index]
    ) {
      distance[index] = 0;
      queue[queueEnd] = index;
      queueEnd += 1;
    }
  }

  while (queueStart < queueEnd) {
    const index = queue[queueStart];
    queueStart += 1;
    const currentDistance = distance[index];
    if (currentDistance >= maximumDistance) continue;
    const x = index % width;
    const y = Math.floor(index / width);

    for (let deltaY = -1; deltaY <= 1; deltaY += 1) {
      for (let deltaX = -1; deltaX <= 1; deltaX += 1) {
        if (deltaX === 0 && deltaY === 0) continue;
        const nextX = x + deltaX;
        const nextY = y + deltaY;
        if (
          nextX < 0 ||
          nextX >= width ||
          nextY < 0 ||
          nextY >= height
        ) {
          continue;
        }
        const nextIndex = nextY * width + nextX;
        if (
          distance[nextIndex] !== unvisited ||
          !orientationMask[nextIndex] ||
          !foregroundMask[nextIndex]
        ) {
          continue;
        }
        distance[nextIndex] = currentDistance + 1;
        queue[queueEnd] = nextIndex;
        queueEnd += 1;
      }
    }
  }

  const searchMask = new Uint8Array(pixelCount);
  const supportConfidence = new Uint8Array(pixelCount);
  for (let index = 0; index < pixelCount; index += 1) {
    const currentDistance = distance[index];
    if (currentDistance === unvisited || currentDistance > maximumDistance) {
      continue;
    }
    searchMask[index] = 1;
    supportConfidence[index] = Math.round(
      72 + (1 - currentDistance / Math.max(maximumDistance, 1)) * 183
    );
  }

  removeSmallSearchMaskComponents(
    searchMask,
    supportConfidence,
    width,
    height,
    Math.max(96, Math.round(ridgePeriodPixels ** 2 * 2))
  );

  return { mask: searchMask, supportConfidence };
}

// Genişleme sonrasında tek başına kalan küçük arama adacıklarını ve güven değerlerini birlikte temizler.
function removeSmallSearchMaskComponents(
  mask: Uint8Array,
  supportConfidence: Uint8Array,
  width: number,
  height: number,
  minimumPixelCount: number
) {
  const visited = new Uint8Array(mask.length);

  for (let startIndex = 0; startIndex < mask.length; startIndex += 1) {
    if (!mask[startIndex] || visited[startIndex]) continue;
    const queue = [startIndex];
    visited[startIndex] = 1;

    for (let queueIndex = 0; queueIndex < queue.length; queueIndex += 1) {
      const index = queue[queueIndex];
      const x = index % width;
      const y = Math.floor(index / width);
      for (let deltaY = -1; deltaY <= 1; deltaY += 1) {
        for (let deltaX = -1; deltaX <= 1; deltaX += 1) {
          if (deltaX === 0 && deltaY === 0) continue;
          const nextX = x + deltaX;
          const nextY = y + deltaY;
          if (
            nextX < 0 ||
            nextX >= width ||
            nextY < 0 ||
            nextY >= height
          ) {
            continue;
          }
          const nextIndex = nextY * width + nextX;
          if (!mask[nextIndex] || visited[nextIndex]) continue;
          visited[nextIndex] = 1;
          queue.push(nextIndex);
        }
      }
    }

    if (queue.length >= minimumPixelCount) continue;
    for (const index of queue) {
      mask[index] = 0;
      supportConfidence[index] = 0;
    }
  }
}

// RGBA piksel tamponunu lokal eşik hesabında kullanılacak luminance görüntüsüne çevirir.
function rgbaToGrayscale(pixels: Uint8Array, width: number, height: number) {
  const grayscale = new Uint8Array(width * height);
  for (let index = 0; index < grayscale.length; index += 1) {
    const offset = index * 4;
    grayscale[index] = Math.round(
      (pixels[offset] ?? 0) * 0.299 +
        (pixels[offset + 1] ?? 0) * 0.587 +
        (pixels[offset + 2] ?? 0) * 0.114
    );
  }
  return grayscale;
}

// Her parmak pikselinin maske sınırına yaklaşık Manhattan uzaklığını iki geçişte hesaplar.
export function createMaskDistanceMap(
  mask: Uint8Array,
  width: number,
  height: number
) {
  const distance = new Uint16Array(mask.length);
  const maximumDistance = width + height;

  for (let index = 0; index < mask.length; index += 1) {
    distance[index] = mask[index] ? maximumDistance : 0;
  }

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      if (!mask[index]) continue;
      if (x > 0) distance[index] = Math.min(distance[index], distance[index - 1] + 1);
      if (y > 0) distance[index] = Math.min(distance[index], distance[index - width] + 1);
    }
  }

  for (let y = height - 1; y >= 0; y -= 1) {
    for (let x = width - 1; x >= 0; x -= 1) {
      const index = y * width + x;
      if (!mask[index]) continue;
      if (x < width - 1) {
        distance[index] = Math.min(distance[index], distance[index + 1] + 1);
      }
      if (y < height - 1) {
        distance[index] = Math.min(distance[index], distance[index + width] + 1);
      }
    }
  }

  return distance;
}

// Lokal maskeli ortalamanın altında kalan koyu çizgileri ridge olarak işaretler.
function createAdaptiveRidgeBinary(
  grayscale: Uint8Array,
  mask: Uint8Array,
  maskDistance: Uint16Array,
  width: number,
  height: number,
  ridgePeriodPixels: number,
  thresholdMultiplier = 1
) {
  const binary = new Uint8Array(mask.length);
  const integral = createIntegralImage(grayscale, width, height);
  const maskIntegral = createIntegralImage(mask, width, height);
  const radius = Math.max(6, Math.round(ridgePeriodPixels * 1.5));

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      if (!mask[index] || maskDistance[index] < 2) continue;
      const left = Math.max(0, x - radius);
      const top = Math.max(0, y - radius);
      const right = Math.min(width - 1, x + radius);
      const bottom = Math.min(height - 1, y + radius);
      const foregroundCount = readIntegralSum(
        maskIntegral,
        width,
        left,
        top,
        right,
        bottom
      );
      if (foregroundCount < radius * radius * 0.5) continue;
      const localMean =
        readIntegralSum(integral, width, left, top, right, bottom) /
        foregroundCount;
      const thresholdOffset =
        Math.max(2, ridgePeriodPixels * 0.22) * thresholdMultiplier;
      if (grayscale[index] < localMean - thresholdOffset) binary[index] = 1;
    }
  }

  return binary;
}

// Adaptif eşikten kalan çok küçük, gerçek ridge oluşturamayacak ikili bileşenleri kaldırır.
// Eşik ridge periyoduna bağlıdır; ana ridge bantlarını koparmamak için bilinçli olarak düşüktür.
export function removeSmallBinaryComponents(
  binary: Uint8Array,
  width: number,
  height: number,
  minimumComponentPixels: number
) {
  const visited = new Uint8Array(binary.length);
  let removedPixelCount = 0;

  for (let startIndex = 0; startIndex < binary.length; startIndex += 1) {
    if (!binary[startIndex] || visited[startIndex]) continue;

    const component = [startIndex];
    visited[startIndex] = 1;

    for (
      let componentIndex = 0;
      componentIndex < component.length;
      componentIndex += 1
    ) {
      const index = component[componentIndex];
      const x = index % width;
      const y = Math.floor(index / width);

      for (let deltaY = -1; deltaY <= 1; deltaY += 1) {
        for (let deltaX = -1; deltaX <= 1; deltaX += 1) {
          if (deltaX === 0 && deltaY === 0) continue;
          const nextX = x + deltaX;
          const nextY = y + deltaY;
          if (
            nextX < 0 ||
            nextY < 0 ||
            nextX >= width ||
            nextY >= height
          ) {
            continue;
          }
          const nextIndex = nextY * width + nextX;
          if (binary[nextIndex] && !visited[nextIndex]) {
            visited[nextIndex] = 1;
            component.push(nextIndex);
          }
        }
      }
    }

    if (component.length >= minimumComponentPixels) continue;
    for (const index of component) binary[index] = 0;
    removedPixelCount += component.length;
  }

  return removedPixelCount;
}

// Tek piksellik gürültüyü kaldırıp küçük ridge boşluklarını kapatmadan önce yalnızca güçlü komşulukları korur.
function cleanBinaryRidges(
  binary: Uint8Array,
  maskDistance: Uint16Array,
  width: number,
  height: number
) {
  for (let pass = 0; pass < 2; pass += 1) {
    const next = new Uint8Array(binary);
    for (let y = 1; y < height - 1; y += 1) {
      for (let x = 1; x < width - 1; x += 1) {
        const index = y * width + x;
        if (maskDistance[index] < 2) {
          next[index] = 0;
          continue;
        }
        const neighborCount = countNeighbors(binary, width, x, y);
        if (binary[index] && neighborCount <= 1) next[index] = 0;
        if (!binary[index] && neighborCount >= 7) next[index] = 1;
      }
    }
    binary.set(next);
  }
}

// Ridge bandındaki mikro parlaklık deliklerini doldurup inceltme sırasında küçük halka oluşmasını önler.
export function fillSmallBinaryRidgeHoles(
  binary: Uint8Array,
  maskDistance: Uint16Array,
  width: number,
  height: number,
  ridgePeriodPixels: number
) {
  const visited = new Uint8Array(binary.length);
  const maximumArea = Math.max(
    8,
    Math.round(ridgePeriodPixels ** 2 * 0.6)
  );
  const maximumSpan = Math.max(4, Math.round(ridgePeriodPixels * 0.9));
  let filledPixelCount = 0;

  for (let startIndex = 0; startIndex < binary.length; startIndex += 1) {
    if (
      binary[startIndex] ||
      visited[startIndex] ||
      maskDistance[startIndex] < 2
    ) {
      continue;
    }

    const component = [startIndex];
    visited[startIndex] = 1;
    let enclosed = true;
    let minimumX = startIndex % width;
    let maximumX = minimumX;
    let minimumY = Math.floor(startIndex / width);
    let maximumY = minimumY;

    for (
      let componentIndex = 0;
      componentIndex < component.length;
      componentIndex += 1
    ) {
      const index = component[componentIndex];
      const x = index % width;
      const y = Math.floor(index / width);
      minimumX = Math.min(minimumX, x);
      maximumX = Math.max(maximumX, x);
      minimumY = Math.min(minimumY, y);
      maximumY = Math.max(maximumY, y);

      for (let deltaY = -1; deltaY <= 1; deltaY += 1) {
        for (let deltaX = -1; deltaX <= 1; deltaX += 1) {
          if (deltaX === 0 && deltaY === 0) continue;
          const nextX = x + deltaX;
          const nextY = y + deltaY;
          if (
            nextX < 0 ||
            nextX >= width ||
            nextY < 0 ||
            nextY >= height
          ) {
            enclosed = false;
            continue;
          }
          const nextIndex = nextY * width + nextX;
          if (maskDistance[nextIndex] < 2) {
            enclosed = false;
            continue;
          }
          if (binary[nextIndex] || visited[nextIndex]) continue;
          visited[nextIndex] = 1;
          component.push(nextIndex);
        }
      }
    }

    const componentWidth = maximumX - minimumX + 1;
    const componentHeight = maximumY - minimumY + 1;
    if (
      !enclosed ||
      component.length > maximumArea ||
      componentWidth > maximumSpan ||
      componentHeight > maximumSpan
    ) {
      continue;
    }

    for (const index of component) binary[index] = 1;
    filledPixelCount += component.length;
  }

  return filledPixelCount;
}

// Güvenilir yerel ridge eksenine yaklaşık dik ilerleyen kısa ikili bağlantıları,
// inceltme bu bağlantıları kalıcı iskelet çatallarına çevirmeden önce keser.
export function removeOrientationInconsistentBinaryBridges({
  binary,
  orientationAngles,
  orientationMask,
  maskDistance,
  width,
  height,
  ridgePeriodPixels,
}: {
  binary: Uint8Array;
  orientationAngles: Float32Array;
  orientationMask: Uint8Array;
  maskDistance: Uint16Array;
  width: number;
  height: number;
  ridgePeriodPixels: number;
}) {
  if (
    binary.length !== width * height ||
    orientationAngles.length !== binary.length ||
    orientationMask.length !== binary.length ||
    maskDistance.length !== binary.length
  ) {
    return 0;
  }

  const period = clamp(ridgePeriodPixels, 5, 14);
  const probeLength = Math.max(5, Math.round(period * 0.95));
  const maximumTangentRun = Math.max(2, Math.round(period * 0.35));
  const minimumNormalRun = Math.max(2, Math.round(period * 0.35));
  const minimumNormalTotal = Math.max(5, Math.round(period * 0.75));
  const minimumAxisAdvantage = Math.max(2, Math.round(period * 0.35));
  const boundaryMargin = Math.max(3, Math.round(period * 0.55));
  const candidateMask = new Uint8Array(binary.length);

  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const index = y * width + x;
      const angle = orientationAngles[index];
      if (
        !binary[index] ||
        !orientationMask[index] ||
        maskDistance[index] < boundaryMargin ||
        !Number.isFinite(angle)
      ) {
        continue;
      }

      const tangentX = Math.cos(angle);
      const tangentY = Math.sin(angle);
      const normalX = -tangentY;
      const normalY = tangentX;
      const tangentForward = countDirectionalBinaryRun(
        binary,
        width,
        height,
        x,
        y,
        tangentX,
        tangentY,
        probeLength
      );
      const tangentBackward = countDirectionalBinaryRun(
        binary,
        width,
        height,
        x,
        y,
        -tangentX,
        -tangentY,
        probeLength
      );
      if (
        tangentForward > maximumTangentRun ||
        tangentBackward > maximumTangentRun
      ) {
        continue;
      }

      const normalForward = countDirectionalBinaryRun(
        binary,
        width,
        height,
        x,
        y,
        normalX,
        normalY,
        probeLength
      );
      const normalBackward = countDirectionalBinaryRun(
        binary,
        width,
        height,
        x,
        y,
        -normalX,
        -normalY,
        probeLength
      );
      const tangentTotal = tangentForward + tangentBackward;
      const normalTotal = normalForward + normalBackward;
      if (
        normalForward < minimumNormalRun ||
        normalBackward < minimumNormalRun ||
        normalTotal < minimumNormalTotal ||
        normalTotal < tangentTotal + minimumAxisAdvantage
      ) {
        continue;
      }

      candidateMask[index] = 1;
    }
  }

  const visited = new Uint8Array(binary.length);
  const maximumCandidateSpan = Math.max(4, Math.round(period * 0.9));
  const maximumCandidatePixels = Math.max(
    10,
    Math.round(period * period * 0.55)
  );
  let removedPixelCount = 0;

  for (let startIndex = 0; startIndex < candidateMask.length; startIndex += 1) {
    if (!candidateMask[startIndex] || visited[startIndex]) continue;
    const component = [startIndex];
    visited[startIndex] = 1;
    let minimumX = startIndex % width;
    let maximumX = minimumX;
    let minimumY = Math.floor(startIndex / width);
    let maximumY = minimumY;

    for (
      let componentIndex = 0;
      componentIndex < component.length;
      componentIndex += 1
    ) {
      const index = component[componentIndex];
      const x = index % width;
      const y = Math.floor(index / width);
      minimumX = Math.min(minimumX, x);
      maximumX = Math.max(maximumX, x);
      minimumY = Math.min(minimumY, y);
      maximumY = Math.max(maximumY, y);

      for (let deltaY = -1; deltaY <= 1; deltaY += 1) {
        for (let deltaX = -1; deltaX <= 1; deltaX += 1) {
          if (deltaX === 0 && deltaY === 0) continue;
          const nextX = x + deltaX;
          const nextY = y + deltaY;
          if (
            nextX < 0 ||
            nextX >= width ||
            nextY < 0 ||
            nextY >= height
          ) {
            continue;
          }
          const nextIndex = nextY * width + nextX;
          if (!candidateMask[nextIndex] || visited[nextIndex]) continue;
          visited[nextIndex] = 1;
          component.push(nextIndex);
        }
      }
    }

    const componentSpan = Math.max(
      maximumX - minimumX + 1,
      maximumY - minimumY + 1
    );
    if (
      componentSpan > maximumCandidateSpan ||
      component.length > maximumCandidatePixels
    ) {
      continue;
    }
    for (const index of component) binary[index] = 0;
    removedPixelCount += component.length;
  }

  return removedPixelCount;
}

// Bir eksende merkezden itibaren kesintisiz ikili ridge uzunluğunu, çapraz açılarda
// aynı pikseli iki kez saymadan ölçer.
function countDirectionalBinaryRun(
  binary: Uint8Array,
  width: number,
  height: number,
  startX: number,
  startY: number,
  directionX: number,
  directionY: number,
  maximumLength: number
) {
  let count = 0;
  let previousIndex = startY * width + startX;
  for (let distance = 1; distance <= maximumLength; distance += 1) {
    const x = Math.round(startX + directionX * distance);
    const y = Math.round(startY + directionY * distance);
    if (x < 0 || x >= width || y < 0 || y >= height) break;
    const index = y * width + x;
    if (index === previousIndex) continue;
    previousIndex = index;
    if (!binary[index]) break;
    count += 1;
  }
  return count;
}

// İkili ridge bantlarını topolojiyi koruyarak tek piksel kalınlığında iskelete indirir.
function thinRidgesZhangSuen(
  binary: Uint8Array,
  width: number,
  height: number
) {
  const skeleton = new Uint8Array(binary);
  let iterations = 0;
  let changed = true;

  while (changed && iterations < MAX_THINNING_ITERATIONS) {
    changed = false;
    for (let subIteration = 0; subIteration < 2; subIteration += 1) {
      const removals: number[] = [];

      for (let y = 1; y < height - 1; y += 1) {
        for (let x = 1; x < width - 1; x += 1) {
          const index = y * width + x;
          if (!skeleton[index]) continue;
          const neighbors = getClockwiseNeighbors(skeleton, width, x, y);
          const enabledCount = neighbors.reduce((total, value) => total + value, 0);
          if (enabledCount < 2 || enabledCount > 6) continue;
          if (calculateCrossingNumber(neighbors) !== 1) continue;

          const p2 = neighbors[0];
          const p4 = neighbors[2];
          const p6 = neighbors[4];
          const p8 = neighbors[6];
          const firstCondition =
            subIteration === 0
              ? p2 * p4 * p6 === 0 && p4 * p6 * p8 === 0
              : p2 * p4 * p8 === 0 && p2 * p6 * p8 === 0;
          if (firstCondition) removals.push(index);
        }
      }

      if (removals.length > 0) {
        changed = true;
        for (const index of removals) skeleton[index] = 0;
      }
    }
    iterations += 1;
  }

  return { skeleton, iterations };
}

// Birbirine bakan, aynı eksendeki çok kısa iskelet kopukluklarını bağlayarak iki sahte ridge sonunu önler.
export function bridgeShortSkeletonGaps(
  skeleton: Uint8Array,
  searchMask: Uint8Array,
  width: number,
  height: number,
  ridgePeriodPixels: number,
  orientationAngles?: Float32Array,
  orientationMask?: Uint8Array
) {
  const endpoints: (PixelPoint & { angleDegrees: number })[] = [];
  const maximumGap = Math.max(4, Math.round(ridgePeriodPixels * 0.7));

  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const index = y * width + x;
      if (
        skeleton[index] &&
        searchMask[index] &&
        countNeighbors(skeleton, width, x, y) === 1
      ) {
        endpoints.push({
          x,
          y,
          angleDegrees: estimateEndpointAxis(
            skeleton,
            width,
            height,
            x,
            y,
            maximumGap
          ),
        });
      }
    }
  }

  const possibleBridges: {
    firstIndex: number;
    secondIndex: number;
    distance: number;
    points: PixelPoint[];
  }[] = [];

  for (let firstIndex = 0; firstIndex < endpoints.length; firstIndex += 1) {
    for (
      let secondIndex = firstIndex + 1;
      secondIndex < endpoints.length;
      secondIndex += 1
    ) {
      const first = endpoints[firstIndex];
      const second = endpoints[secondIndex];
      const distance = Math.hypot(second.x - first.x, second.y - first.y);
      if (distance < 2 || distance > maximumGap) continue;
      const connectionAxis = normalizeAxisDegrees(
        (Math.atan2(second.y - first.y, second.x - first.x) * 180) / Math.PI
      );
      if (
        axisDifferenceDegrees(first.angleDegrees, connectionAxis) > 18 ||
        axisDifferenceDegrees(second.angleDegrees, connectionAxis) > 18 ||
        axisDifferenceDegrees(first.angleDegrees, second.angleDegrees) > 20
      ) {
        continue;
      }
      if (
        !isGapBridgeConsistentWithOrientation({
          first,
          second,
          connectionAxis,
          orientationAngles,
          orientationMask,
          width,
          height,
        })
      ) {
        continue;
      }

      const points = createLinePoints(first.x, first.y, second.x, second.y);
      if (
        points.some(
          (point) => !searchMask[point.y * width + point.x]
        ) ||
        !isBridgePathClear(
          skeleton,
          width,
          height,
          points,
          first,
          second
        )
      ) {
        continue;
      }
      possibleBridges.push({ firstIndex, secondIndex, distance, points });
    }
  }

  possibleBridges.sort((left, right) => left.distance - right.distance);
  const usedEndpoints = new Set<number>();
  let bridgedPixelCount = 0;

  for (const bridge of possibleBridges) {
    if (
      usedEndpoints.has(bridge.firstIndex) ||
      usedEndpoints.has(bridge.secondIndex)
    ) {
      continue;
    }
    usedEndpoints.add(bridge.firstIndex);
    usedEndpoints.add(bridge.secondIndex);
    for (const point of bridge.points) {
      const index = point.y * width + point.x;
      if (!skeleton[index]) {
        skeleton[index] = 1;
        bridgedPixelCount += 1;
      }
    }
  }

  return bridgedPixelCount;
}

// İnceltme öncesinde kesilen enine bağların, uçlar birbirine bakıyor diye yeniden
// kurulmasını engeller; yön alanı yoksa eski güvenli davranışı korur.
function isGapBridgeConsistentWithOrientation({
  first,
  second,
  connectionAxis,
  orientationAngles,
  orientationMask,
  width,
  height,
}: {
  first: PixelPoint;
  second: PixelPoint;
  connectionAxis: number;
  orientationAngles?: Float32Array;
  orientationMask?: Uint8Array;
  width: number;
  height: number;
}) {
  if (
    !orientationAngles ||
    !orientationMask ||
    orientationAngles.length !== width * height ||
    orientationMask.length !== width * height
  ) {
    return true;
  }

  const samples = [
    first,
    {
      x: Math.round((first.x + second.x) / 2),
      y: Math.round((first.y + second.y) / 2),
    },
    second,
  ];
  for (const sample of samples) {
    const index = sample.y * width + sample.x;
    const angleRadians = orientationAngles[index];
    if (!orientationMask[index] || !Number.isFinite(angleRadians)) continue;
    const expectedAxis = normalizeAxisDegrees(
      (angleRadians * 180) / Math.PI
    );
    if (
      axisDifferenceDegrees(expectedAxis, connectionAxis) >
      MAX_ORIENTATION_GAP_AXIS_DIFFERENCE_DEGREES
    ) {
      return false;
    }
  }
  return true;
}

// Köprünün başka bir ridge'e değmesini veya onu kesmesini engelleyerek sahte çatallanma oluşmasını önler.
function isBridgePathClear(
  skeleton: Uint8Array,
  width: number,
  height: number,
  points: PixelPoint[],
  first: PixelPoint,
  second: PixelPoint
) {
  const endpointIndexes = new Set([
    first.y * width + first.x,
    second.y * width + second.x,
  ]);

  for (let pointIndex = 1; pointIndex < points.length - 1; pointIndex += 1) {
    const point = points[pointIndex];
    for (let deltaY = -1; deltaY <= 1; deltaY += 1) {
      for (let deltaX = -1; deltaX <= 1; deltaX += 1) {
        const nextX = point.x + deltaX;
        const nextY = point.y + deltaY;
        if (
          nextX < 0 ||
          nextX >= width ||
          nextY < 0 ||
          nextY >= height
        ) {
          continue;
        }
        const nextIndex = nextY * width + nextX;
        if (skeleton[nextIndex] && !endpointIndexes.has(nextIndex)) {
          return false;
        }
      }
    }
  }

  return true;
}

// Ridge ucundan içeri doğru tek dalı izleyip kısa kopukluk hizalamasında kullanılacak ekseni hesaplar.
function estimateEndpointAxis(
  skeleton: Uint8Array,
  width: number,
  height: number,
  startX: number,
  startY: number,
  maximumLength: number
) {
  let previousIndex = -1;
  let current = { x: startX, y: startY };
  let last = current;

  for (let length = 0; length < maximumLength; length += 1) {
    const currentIndex = current.y * width + current.x;
    const nextPoints = getNeighborPoints(
      skeleton,
      width,
      height,
      current.x,
      current.y
    ).filter((point) => point.y * width + point.x !== previousIndex);
    if (nextPoints.length !== 1) break;
    previousIndex = currentIndex;
    current = nextPoints[0];
    last = current;
  }

  return normalizeAxisDegrees(
    (Math.atan2(last.y - startY, last.x - startX) * 180) / Math.PI
  );
}

// İki iskelet ucu arasındaki Bresenham piksellerini bağlantı maskesi kontrolü için listeler.
function createLinePoints(
  startX: number,
  startY: number,
  endX: number,
  endY: number
) {
  const points: PixelPoint[] = [];
  const deltaX = Math.abs(endX - startX);
  const deltaY = Math.abs(endY - startY);
  const stepX = startX < endX ? 1 : -1;
  const stepY = startY < endY ? 1 : -1;
  let error = deltaX - deltaY;
  let x = startX;
  let y = startY;

  while (true) {
    points.push({ x, y });
    if (x === endX && y === endY) break;
    const doubledError = error * 2;
    if (doubledError > -deltaY) {
      error -= deltaY;
      x += stepX;
    }
    if (doubledError < deltaX) {
      error += deltaX;
      y += stepY;
    }
  }

  return points;
}

// Merkez piksel çevresindeki 0->1 geçiş sayısı ridge sonu ve çatallanmayı ayırır.
function calculateCrossingNumber(neighbors: number[]) {
  let transitions = 0;
  for (let index = 0; index < neighbors.length; index += 1) {
    transitions += Math.abs(neighbors[index] - neighbors[(index + 1) % neighbors.length]);
  }
  return transitions / 2;
}

// Kısa spur parçalarının minutiae sayılmaması için merkezden çıkan sürdürülebilir dalları izler.
function countSustainedBranches(
  skeleton: Uint8Array,
  width: number,
  height: number,
  centerX: number,
  centerY: number,
  minimumLength: number
) {
  const centerIndex = centerY * width + centerX;
  const starts = getDistinctBranchStarts(skeleton, width, centerX, centerY);
  let sustained = 0;

  for (const start of starts) {
    let previousIndex = centerIndex;
    let current = start;
    const visited = new Set<number>([centerIndex]);

    for (let length = 1; length <= minimumLength; length += 1) {
      const currentIndex = current.y * width + current.x;
      visited.add(currentIndex);
      if (length >= minimumLength) {
        sustained += 1;
        break;
      }
      const nextPoints = getNeighborPoints(
        skeleton,
        width,
        height,
        current.x,
        current.y
      ).filter((point) => {
        const index = point.y * width + point.x;
        return index !== previousIndex && !visited.has(index);
      });
      if (nextPoints.length === 0) break;
      previousIndex = currentIndex;
      current = nextPoints[0];
    }
  }

  return sustained;
}

// Merkezden çıkan üç dalı ayrı izleyerek paralel ridge'lerin halkadaki piksellerinin
// gerçek çatallanma dalı sanılmasını önler.
function hasStableBifurcationRingTopology(
  skeleton: Uint8Array,
  width: number,
  height: number,
  centerX: number,
  centerY: number,
  ridgePeriodPixels: number
) {
  const starts = getDistinctBranchStarts(
    skeleton,
    width,
    centerX,
    centerY
  );
  if (starts.length !== 3) return false;

  const requiredLength = Math.max(
    12,
    Math.round(ridgePeriodPixels * 1.8)
  );
  const traceLength = Math.max(
    requiredLength,
    Math.round(ridgePeriodPixels * BIFURCATION_RING_RADIUS_PERIOD_FACTOR)
  );
  const traces = starts.map((start) =>
    traceBifurcationBranch(
      skeleton,
      width,
      height,
      centerX,
      centerY,
      start,
      traceLength
    )
  );
  if (traces.some((trace) => !trace || trace.length < requiredLength)) {
    return false;
  }

  const branchAngles = traces.map((trace) => {
    const endpoint = trace![trace!.length - 1];
    return normalizeDirectedDegrees(
      (Math.atan2(endpoint.y - centerY, endpoint.x - centerX) * 180) /
        Math.PI
    );
  });
  const minimumAngleSeparation = 26;
  for (let firstIndex = 0; firstIndex < branchAngles.length; firstIndex += 1) {
    for (
      let secondIndex = firstIndex + 1;
      secondIndex < branchAngles.length;
      secondIndex += 1
    ) {
      if (
        directedAngleDifference(
          branchAngles[firstIndex],
          branchAngles[secondIndex]
        ) < minimumAngleSeparation
      ) {
        return false;
      }
    }
  }

  const occupiedPixels = new Set<number>();
  for (const trace of traces) {
    for (const point of trace!) {
      const index = point.y * width + point.x;
      if (occupiedPixels.has(index)) return false;
      occupiedPixels.add(index);
    }
  }

  return true;
}

// Çatallanma dalını merkezden dışarı doğru, yönünü koruyarak izler.
function traceBifurcationBranch(
  skeleton: Uint8Array,
  width: number,
  height: number,
  centerX: number,
  centerY: number,
  start: PixelPoint,
  maximumLength: number
) {
  const trace: PixelPoint[] = [];
  const visited = new Set<number>([centerY * width + centerX]);
  let previous = { x: centerX, y: centerY };
  let current = start;

  while (trace.length < maximumLength) {
    if (
      current.x < 0 ||
      current.x >= width ||
      current.y < 0 ||
      current.y >= height
    ) {
      return null;
    }
    const currentIndex = current.y * width + current.x;
    if (visited.has(currentIndex)) return null;
    visited.add(currentIndex);
    trace.push(current);

    const nextPoints = getNeighborPoints(
      skeleton,
      width,
      height,
      current.x,
      current.y
    ).filter((point) => {
      const index = point.y * width + point.x;
      return index !== previous.y * width + previous.x && !visited.has(index);
    });
    if (nextPoints.length === 0) break;

    const incomingX = current.x - previous.x;
    const incomingY = current.y - previous.y;
    nextPoints.sort((left, right) => {
      const leftX = left.x - current.x;
      const leftY = left.y - current.y;
      const rightX = right.x - current.x;
      const rightY = right.y - current.y;
      return (
        rightX * incomingX + rightY * incomingY -
        (leftX * incomingX + leftY * incomingY)
      );
    });
    previous = current;
    current = nextPoints[0];
  }

  return trace;
}

// Crossing Number çevresindeki her 0->1 geçişinden tek başlangıç alarak aynı dalın çapraz komşularını iki kez saymayı önler.
function getDistinctBranchStarts(
  skeleton: Uint8Array,
  width: number,
  centerX: number,
  centerY: number
) {
  const neighbors = getClockwiseNeighbors(skeleton, width, centerX, centerY);
  const offsets = [
    [0, -1],
    [1, -1],
    [1, 0],
    [1, 1],
    [0, 1],
    [-1, 1],
    [-1, 0],
    [-1, -1],
  ] as const;
  const starts: PixelPoint[] = [];

  for (let index = 0; index < neighbors.length; index += 1) {
    const previous =
      neighbors[(index + neighbors.length - 1) % neighbors.length];
    if (!neighbors[index] || previous) continue;
    starts.push({
      x: centerX + offsets[index][0],
      y: centerY + offsets[index][1],
    });
  }

  return starts;
}

// Çok küçük bağımsız iskelet parçalarını kaldırarak gürültü lekelerinin ridge sonu üretmesini önler.
export function removeSmallSkeletonComponents(
  skeleton: Uint8Array,
  width: number,
  height: number,
  minimumPixelCount: number
) {
  const visited = new Uint8Array(skeleton.length);
  let removedPixelCount = 0;

  for (let startIndex = 0; startIndex < skeleton.length; startIndex += 1) {
    if (!skeleton[startIndex] || visited[startIndex]) continue;
    const component: number[] = [];
    const queue = [startIndex];
    visited[startIndex] = 1;

    for (let queueIndex = 0; queueIndex < queue.length; queueIndex += 1) {
      const index = queue[queueIndex];
      component.push(index);
      const x = index % width;
      const y = Math.floor(index / width);

      for (let deltaY = -1; deltaY <= 1; deltaY += 1) {
        for (let deltaX = -1; deltaX <= 1; deltaX += 1) {
          if (deltaX === 0 && deltaY === 0) continue;
          const nextX = x + deltaX;
          const nextY = y + deltaY;
          if (
            nextX < 0 ||
            nextX >= width ||
            nextY < 0 ||
            nextY >= height
          ) {
            continue;
          }
          const nextIndex = nextY * width + nextX;
          if (!skeleton[nextIndex] || visited[nextIndex]) continue;
          visited[nextIndex] = 1;
          queue.push(nextIndex);
        }
      }
    }

    if (component.length >= minimumPixelCount) continue;
    for (const index of component) skeleton[index] = 0;
    removedPixelCount += component.length;
  }

  return removedPixelCount;
}

// Ridge periyodundan kısa uç dalları iskeletten kaldırarak bunların sahte son veya çatallanma üretmesini engeller.
export function pruneShortSkeletonBranches(
  skeleton: Uint8Array,
  width: number,
  height: number,
  minimumLength: number
) {
  let removedPixelCount = 0;

  for (let pass = 0; pass < 3; pass += 1) {
    const removals = new Set<number>();

    for (let y = 1; y < height - 1; y += 1) {
      for (let x = 1; x < width - 1; x += 1) {
        const endpointIndex = y * width + x;
        if (
          !skeleton[endpointIndex] ||
          countNeighbors(skeleton, width, x, y) !== 1
        ) {
          continue;
        }

        const path = traceShortEndpointBranch(
          skeleton,
          width,
          height,
          x,
          y,
          minimumLength
        );
        if (!path) continue;
        for (const index of path) removals.add(index);
      }
    }

    if (removals.size === 0) break;
    for (const index of removals) skeleton[index] = 0;
    removedPixelCount += removals.size;
  }

  return removedPixelCount;
}

// Bir uçtan kavşağa veya diğer uca kadar ilerleyip yalnızca eşikten kısa kalan dalın piksellerini döndürür.
function traceShortEndpointBranch(
  skeleton: Uint8Array,
  width: number,
  height: number,
  startX: number,
  startY: number,
  minimumLength: number
) {
  const path: number[] = [];
  let previousIndex = -1;
  let current = { x: startX, y: startY };

  while (path.length <= minimumLength) {
    const currentIndex = current.y * width + current.x;
    path.push(currentIndex);
    const nextPoints = getNeighborPoints(
      skeleton,
      width,
      height,
      current.x,
      current.y
    ).filter((point) => point.y * width + point.x !== previousIndex);

    if (nextPoints.length === 0) {
      return path.length < minimumLength ? path : null;
    }
    if (nextPoints.length > 1) {
      return path.length < minimumLength ? path : null;
    }
    if (path.length >= minimumLength) return null;

    previousIndex = currentIndex;
    current = nextPoints[0];
  }

  return null;
}

// Yakın iskelet piksellerinin ana eksenini minutia yönü olarak 0-180 derece aralığında hesaplar.
function estimateSkeletonAxis(
  skeleton: Uint8Array,
  width: number,
  height: number,
  centerX: number,
  centerY: number,
  radius: number
) {
  const points: PixelPoint[] = [];
  for (let y = Math.max(0, centerY - radius); y <= Math.min(height - 1, centerY + radius); y += 1) {
    for (let x = Math.max(0, centerX - radius); x <= Math.min(width - 1, centerX + radius); x += 1) {
      if (!skeleton[y * width + x]) continue;
      if ((x - centerX) ** 2 + (y - centerY) ** 2 <= radius ** 2) {
        points.push({ x, y });
      }
    }
  }
  if (points.length < 2) return 0;

  const meanX = points.reduce((total, point) => total + point.x, 0) / points.length;
  const meanY = points.reduce((total, point) => total + point.y, 0) / points.length;
  let covarianceXX = 0;
  let covarianceXY = 0;
  let covarianceYY = 0;
  for (const point of points) {
    const dx = point.x - meanX;
    const dy = point.y - meanY;
    covarianceXX += dx * dx;
    covarianceXY += dx * dy;
    covarianceYY += dy * dy;
  }
  let angle =
    (0.5 * Math.atan2(2 * covarianceXY, covarianceXX - covarianceYY) * 180) /
    Math.PI;
  if (angle < 0) angle += 180;
  return angle;
}

// Aynı fiziksel noktada oluşan komşu adaylardan daha güvenilir olanı tutar.
function suppressNearbyCandidates(
  candidates: MinutiaCandidate[],
  ridgePeriodPixels: number
) {
  const selected: MinutiaCandidate[] = [];
  const ordered = [...candidates].sort(
    (left, right) => right.confidence - left.confidence
  );

  for (const candidate of ordered) {
    const minimumDistance =
      candidate.type === 'bifurcation'
        ? Math.max(
            9,
            ridgePeriodPixels * BIFURCATION_SUPPRESSION_PERIOD_FACTOR
          )
        : Math.max(5, ridgePeriodPixels);
    const overlaps = selected.some(
      (existing) =>
        existing.type === candidate.type &&
        Math.hypot(existing.x - candidate.x, existing.y - candidate.y) <
        minimumDistance
    );
    if (!overlaps) selected.push(candidate);
  }
  return selected;
}

// Kısa ridge köprüleri ve küçük enclosure'lar iki yakın çatallanma üretir.
// Aynı iskelet yolu üzerinde birkaç ridge periyodu içinde bağlanan çiftler,
// temasız görüntüde güvenilir minutiae kabul edilmeyecek kadar kararsızdır.
function rejectShortBifurcationCycles({
  candidates,
  skeleton,
  width,
  height,
  ridgePeriodPixels,
}: {
  candidates: MinutiaCandidate[];
  skeleton: Uint8Array;
  width: number;
  height: number;
  ridgePeriodPixels: number;
}) {
  const rejected = new Set<number>();
  const maximumPairDistance = ridgePeriodPixels * 3.1;
  const maximumPathLength = Math.max(
    18,
    Math.round(ridgePeriodPixels * 4.5)
  );

  for (let firstIndex = 0; firstIndex < candidates.length; firstIndex += 1) {
    const first = candidates[firstIndex];
    if (first.type !== 'bifurcation' || rejected.has(firstIndex)) continue;

    for (
      let secondIndex = firstIndex + 1;
      secondIndex < candidates.length;
      secondIndex += 1
    ) {
      const second = candidates[secondIndex];
      if (
        second.type !== 'bifurcation' ||
        rejected.has(secondIndex) ||
        Math.hypot(second.x - first.x, second.y - first.y) >
          maximumPairDistance
      ) {
        continue;
      }
      if (
        !hasSkeletonConnectionWithinSteps(
          skeleton,
          width,
          height,
          first,
          second,
          maximumPathLength
        )
      ) {
        continue;
      }
      rejected.add(firstIndex);
      rejected.add(secondIndex);
      break;
    }
  }

  return candidates.filter((_, index) => !rejected.has(index));
}

// İki aday arasındaki iskelet bağlantısını sınırlı BFS ile doğrular.
function hasSkeletonConnectionWithinSteps(
  skeleton: Uint8Array,
  width: number,
  height: number,
  start: PixelPoint,
  target: PixelPoint,
  maximumSteps: number
) {
  const startIndex = start.y * width + start.x;
  const targetIndex = target.y * width + target.x;
  const queue = [startIndex];
  const depths = [0];
  const visited = new Set<number>([startIndex]);

  for (let queueIndex = 0; queueIndex < queue.length; queueIndex += 1) {
    const index = queue[queueIndex];
    const depth = depths[queueIndex];
    if (index === targetIndex) return true;
    if (depth >= maximumSteps) continue;
    const x = index % width;
    const y = Math.floor(index / width);

    for (let deltaY = -1; deltaY <= 1; deltaY += 1) {
      for (let deltaX = -1; deltaX <= 1; deltaX += 1) {
        if (deltaX === 0 && deltaY === 0) continue;
        const nextX = x + deltaX;
        const nextY = y + deltaY;
        if (
          nextX < 0 ||
          nextX >= width ||
          nextY < 0 ||
          nextY >= height
        ) {
          continue;
        }
        const nextIndex = nextY * width + nextX;
        if (!skeleton[nextIndex] || visited.has(nextIndex)) continue;
        if (nextIndex === targetIndex) return true;
        visited.add(nextIndex);
        queue.push(nextIndex);
        depths.push(depth + 1);
      }
    }
  }

  return false;
}

// Yakın zıt tipleri ve aynı eksende birbirine bakan kopuk-ridge sonlarını kararsız topoloji olarak eler.
export function rejectUnstableCandidatePairs(
  candidates: MinutiaCandidate[],
  ridgePeriodPixels: number
) {
  const rejected = new Set<number>();
  const oppositeTypeDistance = ridgePeriodPixels * 1.35;
  const brokenRidgeDistance = ridgePeriodPixels * 2.2;
  const crossLinkDistance = ridgePeriodPixels * 1.8;

  for (let firstIndex = 0; firstIndex < candidates.length; firstIndex += 1) {
    for (
      let secondIndex = firstIndex + 1;
      secondIndex < candidates.length;
      secondIndex += 1
    ) {
      const first = candidates[firstIndex];
      const second = candidates[secondIndex];
      const deltaX = second.x - first.x;
      const deltaY = second.y - first.y;
      const distance = Math.hypot(deltaX, deltaY);
      const connectionAxis = normalizeAxisDegrees(
        (Math.atan2(deltaY, deltaX) * 180) / Math.PI
      );

      // Paralel iki ridge arasındaki ince gürültü köprüsü, birbirine bakan iki sahte çatallanma üretir.
      if (
        first.type === 'bifurcation' &&
        second.type === 'bifurcation' &&
        distance <= crossLinkDistance &&
        axisDifferenceDegrees(first.angleDegrees, second.angleDegrees) <= 28 &&
        axisDifferenceDegrees(first.angleDegrees, connectionAxis) >= 48 &&
        axisDifferenceDegrees(second.angleDegrees, connectionAxis) >= 48
      ) {
        rejected.add(firstIndex);
        rejected.add(secondIndex);
        continue;
      }

      if (
        first.type !== second.type &&
        distance <= oppositeTypeDistance
      ) {
        rejected.add(firstIndex);
        rejected.add(secondIndex);
        continue;
      }

      if (
        first.type !== 'ending' ||
        second.type !== 'ending' ||
        distance > brokenRidgeDistance
      ) {
        continue;
      }

      const firstAlignment = axisDifferenceDegrees(
        first.angleDegrees,
        connectionAxis
      );
      const secondAlignment = axisDifferenceDegrees(
        second.angleDegrees,
        connectionAxis
      );
      const mutualAlignment = axisDifferenceDegrees(
        first.angleDegrees,
        second.angleDegrees
      );

      if (
        firstAlignment <= 24 &&
        secondAlignment <= 24 &&
        mutualAlignment <= 24
      ) {
        rejected.add(firstIndex);
        rejected.add(secondIndex);
      }
    }
  }

  return candidates.filter((_, index) => !rejected.has(index));
}

// Ridge eksenlerini yönsüz 0-180 derece aralığına normalize eder.
function normalizeAxisDegrees(value: number) {
  const normalized = value % 180;
  return normalized < 0 ? normalized + 180 : normalized;
}

// Dal yönünü merkezden dışarı doğru yönlü 0-360 derece aralığına normalize eder.
function normalizeDirectedDegrees(value: number) {
  const normalized = value % 360;
  return normalized < 0 ? normalized + 360 : normalized;
}

// İki yönlü dal arasındaki en kısa açısal farkı hesaplar.
function directedAngleDifference(first: number, second: number) {
  const difference = Math.abs(
    normalizeDirectedDegrees(first) - normalizeDirectedDegrees(second)
  );
  return Math.min(difference, 360 - difference);
}

// Yönsüz iki ridge ekseni arasındaki en küçük açısal farkı hesaplar.
function axisDifferenceDegrees(first: number, second: number) {
  const difference = Math.abs(
    normalizeAxisDegrees(first) - normalizeAxisDegrees(second)
  );
  return Math.min(difference, 180 - difference);
}

// Siyah zeminde beyaz iskelet, turuncu ridge sonu ve mavi çatallanma işaretleri çizer.
function createMinutiaeVisualization(
  skeleton: Uint8Array,
  minutiae: MinutiaCandidate[],
  mask: Uint8Array,
  width: number,
  height: number
) {
  const pixels = new Uint8Array(width * height * 4);
  for (let index = 0; index < mask.length; index += 1) {
    const offset = index * 4;
    const value = mask[index] ? (skeleton[index] ? 210 : 20) : 0;
    pixels[offset] = value;
    pixels[offset + 1] = value;
    pixels[offset + 2] = value;
    pixels[offset + 3] = 255;
  }

  for (const minutia of minutiae) {
    const color =
      minutia.type === 'ending'
        ? ([255, 145, 45] as const)
        : ([55, 180, 255] as const);
    drawMarker(pixels, width, height, minutia.x, minutia.y, color);
  }
  return pixels;
}

// Minutia merkezini küçük, boşluk bırakmayan renkli artı işaretiyle görünür kılar.
function drawMarker(
  pixels: Uint8Array,
  width: number,
  height: number,
  centerX: number,
  centerY: number,
  color: readonly [number, number, number]
) {
  for (let offset = -3; offset <= 3; offset += 1) {
    setPixelColor(pixels, width, height, centerX + offset, centerY, color);
    setPixelColor(pixels, width, height, centerX, centerY + offset, color);
  }
}

// Görselleştirme işaretini görüntü sınırları içinde RGBA tampona yazar.
function setPixelColor(
  pixels: Uint8Array,
  width: number,
  height: number,
  x: number,
  y: number,
  color: readonly [number, number, number]
) {
  if (x < 0 || x >= width || y < 0 || y >= height) return;
  const offset = (y * width + x) * 4;
  pixels[offset] = color[0];
  pixels[offset + 1] = color[1];
  pixels[offset + 2] = color[2];
  pixels[offset + 3] = 255;
}

// İskelet komşularını kuzeyden başlayıp saat yönünde Zhang-Suen sırasıyla döndürür.
function getClockwiseNeighbors(
  image: Uint8Array,
  width: number,
  x: number,
  y: number
) {
  return [
    image[(y - 1) * width + x],
    image[(y - 1) * width + x + 1],
    image[y * width + x + 1],
    image[(y + 1) * width + x + 1],
    image[(y + 1) * width + x],
    image[(y + 1) * width + x - 1],
    image[y * width + x - 1],
    image[(y - 1) * width + x - 1],
  ];
}

// Merkez çevresindeki etkin iskelet piksellerini koordinat listesine dönüştürür.
function getNeighborPoints(
  image: Uint8Array,
  width: number,
  height: number,
  x: number,
  y: number
) {
  const points: PixelPoint[] = [];
  for (let deltaY = -1; deltaY <= 1; deltaY += 1) {
    for (let deltaX = -1; deltaX <= 1; deltaX += 1) {
      if (deltaX === 0 && deltaY === 0) continue;
      const nextX = x + deltaX;
      const nextY = y + deltaY;
      if (
        nextX >= 0 &&
        nextX < width &&
        nextY >= 0 &&
        nextY < height &&
        image[nextY * width + nextX]
      ) {
        points.push({ x: nextX, y: nextY });
      }
    }
  }
  return points;
}

// Bir iskelet pikselinin sekizli komşuluğundaki etkin piksel sayısını döndürür.
function countNeighbors(
  image: Uint8Array,
  width: number,
  x: number,
  y: number
) {
  return getClockwiseNeighbors(image, width, x, y).reduce(
    (total, value) => total + value,
    0
  );
}

// Lokal ortalama sorgularını sabit maliyete indirmek için bir piksel fazlalıklı integral görüntü oluşturur.
function createIntegralImage(
  values: Uint8Array,
  width: number,
  height: number
) {
  const stride = width + 1;
  const integral = new Float64Array(stride * (height + 1));

  for (let y = 1; y <= height; y += 1) {
    let rowTotal = 0;
    for (let x = 1; x <= width; x += 1) {
      rowTotal += values[(y - 1) * width + (x - 1)];
      integral[y * stride + x] = integral[(y - 1) * stride + x] + rowTotal;
    }
  }
  return integral;
}

// İntegral görüntüden kapalı dikdörtgen toplamını okur.
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

// İkili ridge piksellerinin parmak maskesine oranını ölçer.
function calculateBinaryRidgeRatio(binary: Uint8Array, mask: Uint8Array) {
  let foregroundCount = 0;
  let ridgeCount = 0;
  for (let index = 0; index < mask.length; index += 1) {
    if (!mask[index]) continue;
    foregroundCount += 1;
    if (binary[index]) ridgeCount += 1;
  }
  return round(ridgeCount / Math.max(foregroundCount, 1), 4);
}

// Minutiae aramasına açılan doğrulanmış alanın parmak silüetine oranını tanılama için hesaplar.
function calculateMaskCoverage(
  candidateMask: Uint8Array,
  foregroundMask: Uint8Array
) {
  let foregroundCount = 0;
  let supportedCount = 0;

  for (let index = 0; index < foregroundMask.length; index += 1) {
    if (!foregroundMask[index]) continue;
    foregroundCount += 1;
    if (candidateMask[index]) supportedCount += 1;
  }

  return round(supportedCount / Math.max(foregroundCount, 1), 4);
}

// Sınır payından sonra kalan toplam ve en büyük kesintisiz minutiae arama alanını ölçer.
function calculateSearchableRegionMetrics(
  candidateDistance: Uint16Array,
  foregroundDistance: Uint16Array,
  foregroundMask: Uint8Array,
  width: number,
  height: number,
  candidateBoundaryMargin: number,
  foregroundBoundaryMargin: number
) {
  const searchable = new Uint8Array(foregroundMask.length);
  const visited = new Uint8Array(foregroundMask.length);
  let foregroundCount = 0;
  let searchableCount = 0;
  let largestRegionCount = 0;

  for (let index = 0; index < foregroundMask.length; index += 1) {
    if (!foregroundMask[index]) continue;
    foregroundCount += 1;
    if (
      candidateDistance[index] >= candidateBoundaryMargin &&
      foregroundDistance[index] >= foregroundBoundaryMargin
    ) {
      searchable[index] = 1;
      searchableCount += 1;
    }
  }

  for (let startIndex = 0; startIndex < searchable.length; startIndex += 1) {
    if (!searchable[startIndex] || visited[startIndex]) continue;
    const queue = [startIndex];
    visited[startIndex] = 1;

    for (let queueIndex = 0; queueIndex < queue.length; queueIndex += 1) {
      const index = queue[queueIndex];
      const x = index % width;
      const y = Math.floor(index / width);

      for (let deltaY = -1; deltaY <= 1; deltaY += 1) {
        for (let deltaX = -1; deltaX <= 1; deltaX += 1) {
          if (deltaX === 0 && deltaY === 0) continue;
          const nextX = x + deltaX;
          const nextY = y + deltaY;
          if (
            nextX < 0 ||
            nextX >= width ||
            nextY < 0 ||
            nextY >= height
          ) {
            continue;
          }
          const nextIndex = nextY * width + nextX;
          if (!searchable[nextIndex] || visited[nextIndex]) continue;
          visited[nextIndex] = 1;
          queue.push(nextIndex);
        }
      }
    }

    largestRegionCount = Math.max(largestRegionCount, queue.length);
  }

  return {
    totalRatio: round(searchableCount / Math.max(foregroundCount, 1), 4),
    largestRegionRatio: round(
      largestRegionCount / Math.max(foregroundCount, 1),
      4
    ),
  };
}

// Etkin piksel sayısını tanılama ve sentetik testlerde kullanmak üzere hesaplar.
function countEnabledPixels(image: Uint8Array) {
  let count = 0;
  for (const value of image) if (value) count += 1;
  return count;
}

// Sayıyı belirtilen alt ve üst sınırlar içinde tutar.
function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(Math.max(value, minimum), maximum);
}

// Template metadata'sını gereksiz kayan nokta hassasiyetinden arındırır.
function round(value: number, precision: number) {
  const multiplier = 10 ** precision;
  return Math.round(value * multiplier) / multiplier;
}
