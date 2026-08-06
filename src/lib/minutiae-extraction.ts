import type {
  FingerprintMinutia,
  FingerprintTemplate,
  MinutiaType,
} from '@/types/biometrics';

type MinutiaeExtractionInput = {
  pixels: Uint8Array;
  mask: Uint8Array;
  candidateMask?: Uint8Array;
  orientationMask?: Uint8Array;
  width: number;
  height: number;
  ridgePeriodPixels: number;
  minimumSupportConfidence?: number;
};

type PixelPoint = {
  x: number;
  y: number;
};

type MinutiaCandidate = PixelPoint & {
  type: MinutiaType;
  angleDegrees: number;
  confidence: number;
};

export type MinutiaeExtractionResult = {
  template: FingerprintTemplate;
  visualizationPixels: Uint8Array;
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
  crossingNumberCandidateCount: number;
  branchValidatedCandidateCount: number;
  suppressionCandidateCount: number;
  endingCandidateCount: number;
  bifurcationCandidateCount: number;
  confidenceHistogram: number[];
};

export type MinutiaeDetectionDiagnostics = {
  crossingNumberCandidateCount: number;
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

// Gri ROI'den adaptif ikili ridge haritası, iskelet ve filtrelenmiş minutiae şablonu üretir.
export function extractFingerprintMinutiae({
  pixels,
  mask,
  candidateMask = mask,
  orientationMask = mask,
  width,
  height,
  ridgePeriodPixels,
  minimumSupportConfidence = 76,
}: MinutiaeExtractionInput): MinutiaeExtractionResult {
  const grayscale = rgbaToGrayscale(pixels, width, height);
  const foregroundDistance = createMaskDistanceMap(mask, width, height);
  const period = clamp(ridgePeriodPixels || 8, 5, 14);
  const stableSearchArea = createStableMinutiaeSearchArea({
    supportMask: candidateMask,
    orientationMask,
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
  const binary = createAdaptiveRidgeBinary(
    grayscale,
    mask,
    foregroundDistance,
    width,
    height,
    period
  );
  cleanBinaryRidges(binary, foregroundDistance, width, height);
  const filledHolePixelCount = fillSmallBinaryRidgeHoles(
    binary,
    foregroundDistance,
    width,
    height,
    period
  );
  const thinning = thinRidgesZhangSuen(binary, width, height);
  const bridgedPixelCount = bridgeShortSkeletonGaps(
    thinning.skeleton,
    stableSearchArea.mask,
    width,
    height,
    period
  );
  const removedComponentPixelCount = removeSmallSkeletonComponents(
    thinning.skeleton,
    width,
    height,
    Math.max(18, Math.round(period * 3.5))
  );
  const prunedPixelCount = pruneShortSkeletonBranches(
    thinning.skeleton,
    width,
    height,
    Math.max(8, Math.round(period * 1.8))
  );
  const detection = detectMinutiaeFromSkeletonDetailed({
    skeleton: thinning.skeleton,
    maskDistance: candidateDistance,
    foregroundDistance,
    width,
    height,
    ridgePeriodPixels: period,
    supportConfidence: stableSearchArea.supportConfidence,
    minimumSupportConfidence,
  });
  const rawCandidates = detection.candidates;
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
      thinning.skeleton,
      rawCandidates.slice(0, MAX_MINUTIAE_COUNT),
      mask,
      width,
      height
    ),
    binaryRidgeRatio: calculateBinaryRidgeRatio(binary, mask),
    skeletonPixelCount: countEnabledPixels(thinning.skeleton),
    rawCandidateCount: rawCandidates.length,
    thinningIterations: thinning.iterations,
    prunedPixelCount,
    bridgedPixelCount,
    removedComponentPixelCount,
    filledHolePixelCount,
    supportCoverage: calculateMaskCoverage(candidateMask, mask),
    searchableAreaRatio: searchableRegion.totalRatio,
    largestSearchableRegionRatio: searchableRegion.largestRegionRatio,
    ...detection.diagnostics,
  };
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

  const stabilityValidatedCandidates = rejectUnstableCandidatePairs(
    candidates,
    ridgePeriodPixels
  );
  const suppressedCandidates = suppressNearbyCandidates(
    stabilityValidatedCandidates,
    ridgePeriodPixels
  );
  const confidenceHistogram = new Array<number>(10).fill(0);
  for (const candidate of suppressedCandidates) {
    confidenceHistogram[Math.min(9, Math.floor(candidate.confidence / 10))] += 1;
  }

  return {
    candidates: suppressedCandidates,
    diagnostics: {
      crossingNumberCandidateCount,
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
  ridgePeriodPixels: number
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
      const thresholdOffset = Math.max(2, ridgePeriodPixels * 0.22);
      if (grayscale[index] < localMean - thresholdOffset) binary[index] = 1;
    }
  }

  return binary;
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
  ridgePeriodPixels: number
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

// Merkezden uzaktaki halkada tam üç iskelet çıkışı arayarak kısa köprü ve ağ birleşmelerini eler.
function hasStableBifurcationRingTopology(
  skeleton: Uint8Array,
  width: number,
  height: number,
  centerX: number,
  centerY: number,
  ridgePeriodPixels: number
) {
  const radius = Math.max(
    8,
    Math.round(
      ridgePeriodPixels * BIFURCATION_RING_RADIUS_PERIOD_FACTOR
    )
  );
  const thickness = Math.max(2, Math.round(ridgePeriodPixels * 0.3));
  const innerRadiusSquared = (radius - thickness) ** 2;
  const outerRadiusSquared = (radius + thickness) ** 2;
  const ringPixels = new Set<number>();

  for (
    let y = Math.max(1, centerY - radius - thickness);
    y <= Math.min(height - 2, centerY + radius + thickness);
    y += 1
  ) {
    for (
      let x = Math.max(1, centerX - radius - thickness);
      x <= Math.min(width - 2, centerX + radius + thickness);
      x += 1
    ) {
      const distanceSquared =
        (x - centerX) ** 2 + (y - centerY) ** 2;
      const index = y * width + x;
      if (
        skeleton[index] &&
        distanceSquared >= innerRadiusSquared &&
        distanceSquared <= outerRadiusSquared
      ) {
        ringPixels.add(index);
      }
    }
  }

  const visited = new Set<number>();
  let stableExitCount = 0;
  const minimumExitPixels = Math.max(2, thickness);

  for (const startIndex of ringPixels) {
    if (visited.has(startIndex)) continue;
    const queue = [startIndex];
    visited.add(startIndex);
    let componentPixelCount = 0;

    for (let queueIndex = 0; queueIndex < queue.length; queueIndex += 1) {
      const index = queue[queueIndex];
      componentPixelCount += 1;
      const x = index % width;
      const y = Math.floor(index / width);

      for (let deltaY = -1; deltaY <= 1; deltaY += 1) {
        for (let deltaX = -1; deltaX <= 1; deltaX += 1) {
          if (deltaX === 0 && deltaY === 0) continue;
          const nextIndex = (y + deltaY) * width + x + deltaX;
          if (
            ringPixels.has(nextIndex) &&
            !visited.has(nextIndex)
          ) {
            visited.add(nextIndex);
            queue.push(nextIndex);
          }
        }
      }
    }

    if (componentPixelCount >= minimumExitPixels) {
      stableExitCount += 1;
      if (stableExitCount > 3) return false;
    }
  }

  return stableExitCount === 3;
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
