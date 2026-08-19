import assert from 'node:assert/strict';

import {
  applyControlledBinaryOpening,
  bridgeShortSkeletonGaps,
  createMaskDistanceMap,
  createDistalMinutiaeMask,
  createStableMinutiaeSearchArea,
  detectMinutiaeFromSkeleton,
  detectMinutiaeFromSkeletonDetailed,
  extractFingerprintMinutiae,
  fillSmallBinaryRidgeHoles,
  pruneShortSkeletonBranches,
  removeOrientationInconsistentBinaryBridges,
  removeSmallBinaryComponents,
  removeSmallSkeletonComponents,
  removeThresholdUnstableSkeletonLoops,
  selectThresholdStableMinutiaeCandidates,
} from '../src/lib/minutiae-extraction.ts';

const WIDTH = 96;
const HEIGHT = 96;
const RIDGE_PERIOD = 8;

// İki kalın ridge arasındaki tek piksellik yanlış bağlantı opening ile ayrılırken
// ana ridge gövdelerinin korunmasını doğrular.
const openingBridgeBinary = new Uint8Array(WIDTH * HEIGHT);
for (let y = 28; y <= 34; y += 1) {
  for (let x = 12; x <= 83; x += 1) {
    openingBridgeBinary[y * WIDTH + x] = 1;
  }
}
for (let y = 42; y <= 48; y += 1) {
  for (let x = 12; x <= 83; x += 1) {
    openingBridgeBinary[y * WIDTH + x] = 1;
  }
}
for (let y = 35; y <= 41; y += 1) {
  openingBridgeBinary[y * WIDTH + 48] = 1;
}
const openingResult = applyControlledBinaryOpening({
  binary: openingBridgeBinary,
  width: WIDTH,
  height: HEIGHT,
  ridgePeriodPixels: RIDGE_PERIOD,
});
assert.equal(openingResult.status, 'applied');
assert.equal(openingBridgeBinary[38 * WIDTH + 48], 0);
assert.equal(openingBridgeBinary[31 * WIDTH + 48], 1);
assert.equal(openingBridgeBinary[45 * WIDTH + 48], 1);
assert.equal(countMaskComponents(openingBridgeBinary), 2);

// Tek piksellik gerçek ridge opening tarafından silinecekse güvenlik kapısı sonucu geri alır.
const fragileRidgeBinary = new Uint8Array(WIDTH * HEIGHT);
drawLine(fragileRidgeBinary, 20, 18, 75, 18);
const fragileRidgeBefore = new Uint8Array(fragileRidgeBinary);
const fragileOpeningResult = applyControlledBinaryOpening({
  binary: fragileRidgeBinary,
  width: WIDTH,
  height: HEIGHT,
  ridgePeriodPixels: RIDGE_PERIOD,
});
assert.equal(fragileOpeningResult.status, 'rejected-pixel-loss');
assert.deepEqual(fragileRidgeBinary, fragileRidgeBefore);

// Sekizli komşulukla bağlı maske bileşenlerini sayarak arama adacıklarının birleşmesini doğrular.
function countMaskComponents(mask) {
  const visited = new Uint8Array(mask.length);
  let componentCount = 0;

  for (let startIndex = 0; startIndex < mask.length; startIndex += 1) {
    if (!mask[startIndex] || visited[startIndex]) continue;
    componentCount += 1;
    const queue = [startIndex];
    visited[startIndex] = 1;

    for (let queueIndex = 0; queueIndex < queue.length; queueIndex += 1) {
      const index = queue[queueIndex];
      const x = index % WIDTH;
      const y = Math.floor(index / WIDTH);
      for (let deltaY = -1; deltaY <= 1; deltaY += 1) {
        for (let deltaX = -1; deltaX <= 1; deltaX += 1) {
          if (deltaX === 0 && deltaY === 0) continue;
          const nextX = x + deltaX;
          const nextY = y + deltaY;
          if (
            nextX < 0 ||
            nextX >= WIDTH ||
            nextY < 0 ||
            nextY >= HEIGHT
          ) {
            continue;
          }
          const nextIndex = nextY * WIDTH + nextX;
          if (!mask[nextIndex] || visited[nextIndex]) continue;
          visited[nextIndex] = 1;
          queue.push(nextIndex);
        }
      }
    }
  }

  return componentCount;
}

// Sentetik çizgiyi sekizli bağlı iskelet olarak piksel tamponuna çizer.
function drawLine(image, startX, startY, endX, endY) {
  const deltaX = Math.abs(endX - startX);
  const deltaY = Math.abs(endY - startY);
  const stepX = startX < endX ? 1 : -1;
  const stepY = startY < endY ? 1 : -1;
  let error = deltaX - deltaY;
  let x = startX;
  let y = startY;

  while (true) {
    image[y * WIDTH + x] = 1;
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
}

// Tam dolu maske, sentetik minutiae noktalarının görüntü sınırından güvenle uzak kalmasını sağlar.
function createFullMaskDistance() {
  const mask = new Uint8Array(WIDTH * HEIGHT);
  mask.fill(1);
  return createMaskDistanceMap(mask, WIDTH, HEIGHT);
}

// Ayrık iki Gabor çekirdeğinin güvenilir yön alanı içinde tek arama bölgesine dönüşmesini sınar.
const supportMask = new Uint8Array(WIDTH * HEIGHT);
const orientationMask = new Uint8Array(WIDTH * HEIGHT);
const foregroundMask = new Uint8Array(WIDTH * HEIGHT);
foregroundMask.fill(1);
for (let y = 20; y < 76; y += 1) {
  for (let x = 12; x < 84; x += 1) {
    orientationMask[y * WIDTH + x] = 1;
  }
}
for (let y = 36; y < 60; y += 1) {
  for (let x = 24; x < 39; x += 1) supportMask[y * WIDTH + x] = 1;
  for (let x = 49; x < 64; x += 1) supportMask[y * WIDTH + x] = 1;
}
const stableSearchArea = createStableMinutiaeSearchArea({
  supportMask,
  orientationMask,
  foregroundMask,
  width: WIDTH,
  height: HEIGHT,
  ridgePeriodPixels: RIDGE_PERIOD,
});
if (
  countMaskComponents(stableSearchArea.mask) !== 1 ||
  stableSearchArea.mask[48 * WIDTH + 44] !== 1 ||
  stableSearchArea.mask[10 * WIDTH + 48] !== 0
) {
  throw new Error(
    'Gabor adacıkları güvenilir yön alanı içinde kontrollü biçimde birleştirilemedi.'
  );
}

// Düz ridge çizgisinin iki ucunun ridge ending olarak bulunmasını doğrular.
const straight = new Uint8Array(WIDTH * HEIGHT);
drawLine(straight, 20, 48, 75, 48);
const straightMinutiae = detectMinutiaeFromSkeleton({
  skeleton: straight,
  maskDistance: createFullMaskDistance(),
  width: WIDTH,
  height: HEIGHT,
  ridgePeriodPixels: RIDGE_PERIOD,
});
const straightEndings = straightMinutiae.filter(
  (item) => item.type === 'ending'
).length;
if (straightEndings !== 2) {
  throw new Error(`Düz ridge için 2 son bekleniyordu, ${straightEndings} bulundu.`);
}

// Arama alanının dış kuşağındaki düşük Gabor güveninin tek başına minutiae üretmediğini doğrular.
const weakSupportConfidence = new Uint8Array(WIDTH * HEIGHT);
weakSupportConfidence.fill(72);
const weakSupportMinutiae = detectMinutiaeFromSkeleton({
  skeleton: straight,
  maskDistance: createFullMaskDistance(),
  supportConfidence: weakSupportConfidence,
  width: WIDTH,
  height: HEIGHT,
  ridgePeriodPixels: RIDGE_PERIOD,
});
if (weakSupportMinutiae.length !== 0) {
  throw new Error(
    `Düşük Gabor güveninde 0 aday bekleniyordu, ${weakSupportMinutiae.length} bulundu.`
  );
}

// Arama maskesi sınırında kesilen ridge ucunun minutiae, içerideki gerçek ucun ise aday kalmasını sınar.
const boundedSearchMask = new Uint8Array(WIDTH * HEIGHT);
for (let y = 10; y < 86; y += 1) {
  for (let x = 10; x < 86; x += 1) {
    boundedSearchMask[y * WIDTH + x] = 1;
  }
}
const boundaryEnding = new Uint8Array(WIDTH * HEIGHT);
drawLine(boundaryEnding, 14, 48, 70, 48);
const boundaryFilteredMinutiae = detectMinutiaeFromSkeleton({
  skeleton: boundaryEnding,
  maskDistance: createMaskDistanceMap(
    boundedSearchMask,
    WIDTH,
    HEIGHT
  ),
  width: WIDTH,
  height: HEIGHT,
  ridgePeriodPixels: RIDGE_PERIOD,
});
if (
  boundaryFilteredMinutiae.length !== 1 ||
  boundaryFilteredMinutiae[0].x !== 70
) {
  throw new Error(
    `Arama sınırı filtresinde yalnızca iç ridge sonu bekleniyordu, ${boundaryFilteredMinutiae.length} aday bulundu.`
  );
}

// Ridge içindeki küçük kapalı deliğin temizlenip uzun valley açıklığının korunmasını doğrular.
const holeBinary = new Uint8Array(WIDTH * HEIGHT);
for (let y = 20; y <= 76; y += 1) {
  for (let x = 16; x <= 80; x += 1) {
    holeBinary[y * WIDTH + x] = 1;
  }
}
for (let y = 38; y <= 41; y += 1) {
  for (let x = 38; x <= 41; x += 1) {
    holeBinary[y * WIDTH + x] = 0;
  }
}
for (let x = 30; x <= 54; x += 1) {
  holeBinary[58 * WIDTH + x] = 0;
}
const filledHolePixels = fillSmallBinaryRidgeHoles(
  holeBinary,
  createFullMaskDistance(),
  WIDTH,
  HEIGHT,
  RIDGE_PERIOD
);
if (
  filledHolePixels !== 16 ||
  holeBinary[39 * WIDTH + 39] !== 1 ||
  holeBinary[58 * WIDTH + 42] !== 0
) {
  throw new Error(
    `Mikro delik temizliğinde 16 piksel bekleniyordu, ${filledHolePixels} dolduruldu.`
  );
}

// Y biçimli iskeletin bir çatallanma ve üç ridge sonu üretmesini doğrular.
const branching = new Uint8Array(WIDTH * HEIGHT);
drawLine(branching, 48, 48, 48, 20);
drawLine(branching, 48, 48, 24, 72);
drawLine(branching, 48, 48, 72, 72);
const branchMinutiae = detectMinutiaeFromSkeleton({
  skeleton: branching,
  maskDistance: createFullMaskDistance(),
  width: WIDTH,
  height: HEIGHT,
  ridgePeriodPixels: RIDGE_PERIOD,
});
const branchEndings = branchMinutiae.filter(
  (item) => item.type === 'ending'
).length;
const bifurcations = branchMinutiae.filter(
  (item) => item.type === 'bifurcation'
).length;
const branchDiagnostics = detectMinutiaeFromSkeletonDetailed({
  skeleton: branching,
  maskDistance: createFullMaskDistance(),
  width: WIDTH,
  height: HEIGHT,
  ridgePeriodPixels: RIDGE_PERIOD,
}).diagnostics;

if (branchEndings !== 3 || bifurcations !== 1) {
  throw new Error(
    `Y ridge için 3 son/1 çatallanma bekleniyordu, ${branchEndings}/${bifurcations} bulundu.`
  );
}

// Birkaç ridge periyodu içinde kapanan küçük enclosure güvenilir çatallanma sayılmaz.
const microEnclosure = new Uint8Array(WIDTH * HEIGHT);
drawLine(microEnclosure, 16, 48, 40, 48);
drawLine(microEnclosure, 40, 48, 48, 40);
drawLine(microEnclosure, 48, 40, 56, 48);
drawLine(microEnclosure, 40, 48, 48, 56);
drawLine(microEnclosure, 48, 56, 56, 48);
drawLine(microEnclosure, 56, 48, 80, 48);
const microEnclosureMinutiae = detectMinutiaeFromSkeleton({
  skeleton: microEnclosure,
  maskDistance: createFullMaskDistance(),
  width: WIDTH,
  height: HEIGHT,
  ridgePeriodPixels: RIDGE_PERIOD,
});
const microEnclosureBifurcations = microEnclosureMinutiae.filter(
  (item) => item.type === 'bifurcation'
).length;
assert.equal(microEnclosureBifurcations, 0);

// Geniş ve üç dalı kararlı enclosure, mikro-halka filtresinde korunur.
const wideEnclosure = new Uint8Array(WIDTH * HEIGHT);
drawLine(wideEnclosure, 2, 48, 26, 48);
drawLine(wideEnclosure, 26, 48, 48, 28);
drawLine(wideEnclosure, 48, 28, 70, 48);
drawLine(wideEnclosure, 26, 48, 48, 68);
drawLine(wideEnclosure, 48, 68, 70, 48);
drawLine(wideEnclosure, 70, 48, 94, 48);
const wideEnclosureMinutiae = detectMinutiaeFromSkeleton({
  skeleton: wideEnclosure,
  maskDistance: createFullMaskDistance(),
  width: WIDTH,
  height: HEIGHT,
  ridgePeriodPixels: RIDGE_PERIOD,
});
const wideEnclosureBifurcations = wideEnclosureMinutiae.filter(
  (item) => item.type === 'bifurcation'
).length;
assert.equal(wideEnclosureBifurcations, 2);

// Paralel ridge'ler arasındaki ince çapraz bağın iki gerçek çatallanma gibi saklanmadığını doğrular.
const crossLinkedRidges = new Uint8Array(WIDTH * HEIGHT);
drawLine(crossLinkedRidges, 16, 40, 80, 40);
drawLine(crossLinkedRidges, 16, 48, 80, 48);
drawLine(crossLinkedRidges, 48, 40, 48, 48);
const crossLinkedMinutiae = detectMinutiaeFromSkeleton({
  skeleton: crossLinkedRidges,
  maskDistance: createFullMaskDistance(),
  width: WIDTH,
  height: HEIGHT,
  ridgePeriodPixels: RIDGE_PERIOD,
});
const crossLinkedBifurcations = crossLinkedMinutiae.filter(
  (item) => item.type === 'bifurcation'
).length;
if (crossLinkedBifurcations !== 0) {
  throw new Error(
    `Çapraz ridge köprüsünde 0 çatallanma bekleniyordu, ${crossLinkedBifurcations} bulundu.`
  );
}

// Birden fazla paralel ridge arasındaki merdiven biçimli gürültü bağlarının gerçek çatallanma sayılmadığını doğrular.
const ladderRidges = new Uint8Array(WIDTH * HEIGHT);
drawLine(ladderRidges, 14, 32, 82, 32);
drawLine(ladderRidges, 14, 40, 82, 40);
drawLine(ladderRidges, 14, 48, 82, 48);
drawLine(ladderRidges, 38, 32, 38, 40);
drawLine(ladderRidges, 58, 40, 58, 48);
const ladderMinutiae = detectMinutiaeFromSkeleton({
  skeleton: ladderRidges,
  maskDistance: createFullMaskDistance(),
  width: WIDTH,
  height: HEIGHT,
  ridgePeriodPixels: RIDGE_PERIOD,
});
const ladderBifurcations = ladderMinutiae.filter(
  (item) => item.type === 'bifurcation'
).length;
if (ladderBifurcations !== 0) {
  throw new Error(
    `Paralel ridge merdiveninde 0 çatallanma bekleniyordu, ${ladderBifurcations} bulundu.`
  );
}

// Uzun ridge üzerindeki kısa yan çıkıntının budanıp sahte çatallanma üretmediğini doğrular.
const shortSpur = new Uint8Array(WIDTH * HEIGHT);
drawLine(shortSpur, 20, 48, 75, 48);
drawLine(shortSpur, 48, 48, 48, 43);
const prunedSpurPixels = pruneShortSkeletonBranches(
  shortSpur,
  WIDTH,
  HEIGHT,
  Math.round(RIDGE_PERIOD * 1.5)
);
const shortSpurMinutiae = detectMinutiaeFromSkeleton({
  skeleton: shortSpur,
  maskDistance: createFullMaskDistance(),
  width: WIDTH,
  height: HEIGHT,
  ridgePeriodPixels: RIDGE_PERIOD,
});
const shortSpurEndings = shortSpurMinutiae.filter(
  (item) => item.type === 'ending'
).length;
const shortSpurBifurcations = shortSpurMinutiae.filter(
  (item) => item.type === 'bifurcation'
).length;
if (
  prunedSpurPixels === 0 ||
  shortSpurEndings !== 2 ||
  shortSpurBifurcations !== 0
) {
  throw new Error(
    `Kısa spur budamasında 2 son/0 çatallanma bekleniyordu, ${shortSpurEndings}/${shortSpurBifurcations} bulundu.`
  );
}

// Aynı eksendeki kısa ridge kopmasının iki sahte iç uç yerine yalnızca fiziksel dış uçları bırakmasını doğrular.
const brokenRidge = new Uint8Array(WIDTH * HEIGHT);
drawLine(brokenRidge, 16, 48, 44, 48);
drawLine(brokenRidge, 49, 48, 80, 48);
const bridgedRidge = new Uint8Array(brokenRidge);
const bridgeMask = new Uint8Array(WIDTH * HEIGHT);
bridgeMask.fill(1);
const bridgedPixelCount = bridgeShortSkeletonGaps(
  bridgedRidge,
  bridgeMask,
  WIDTH,
  HEIGHT,
  RIDGE_PERIOD
);
if (bridgedPixelCount === 0 || bridgedRidge[48 * WIDTH + 47] !== 1) {
  throw new Error('Kısa ve aynı eksendeki ridge kopukluğu bağlanamadı.');
}
const brokenRidgeMinutiae = detectMinutiaeFromSkeleton({
  skeleton: brokenRidge,
  maskDistance: createFullMaskDistance(),
  width: WIDTH,
  height: HEIGHT,
  ridgePeriodPixels: RIDGE_PERIOD,
});
const brokenRidgeEndings = brokenRidgeMinutiae.filter(
  (item) => item.type === 'ending'
).length;
if (brokenRidgeEndings !== 2) {
  throw new Error(
    `Kopuk ridge filtresinde 2 dış son bekleniyordu, ${brokenRidgeEndings} bulundu.`
  );
}

// Uzun ridge yanındaki küçük bağımsız çizgi parçasının minutiae üretmeden kaldırılmasını doğrular.
const componentNoise = new Uint8Array(WIDTH * HEIGHT);
drawLine(componentNoise, 16, 48, 80, 48);
drawLine(componentNoise, 30, 25, 35, 25);
const removedComponentPixels = removeSmallSkeletonComponents(
  componentNoise,
  WIDTH,
  HEIGHT,
  Math.round(RIDGE_PERIOD * 3.5)
);
const componentMinutiae = detectMinutiaeFromSkeleton({
  skeleton: componentNoise,
  maskDistance: createFullMaskDistance(),
  width: WIDTH,
  height: HEIGHT,
  ridgePeriodPixels: RIDGE_PERIOD,
});
if (
  removedComponentPixels === 0 ||
  componentMinutiae.filter((item) => item.type === 'ending').length !== 2
) {
  throw new Error('Küçük iskelet bileşeni güvenli biçimde kaldırılamadı.');
}

// Yatay ridge yönüne dik, iki komşu ridge arasındaki kısa ve kalın ikili bağ
// inceltmeden önce kesilirken ana ridge bantlarının korunmasını doğrular.
const orientationBridgeBinary = new Uint8Array(WIDTH * HEIGHT);
for (let y = 39; y <= 41; y += 1) {
  for (let x = 16; x <= 80; x += 1) {
    orientationBridgeBinary[y * WIDTH + x] = 1;
  }
}
for (let y = 47; y <= 49; y += 1) {
  for (let x = 16; x <= 80; x += 1) {
    orientationBridgeBinary[y * WIDTH + x] = 1;
  }
}
for (let y = 42; y <= 46; y += 1) {
  for (let x = 47; x <= 49; x += 1) {
    orientationBridgeBinary[y * WIDTH + x] = 1;
  }
}
const horizontalOrientationAngles = new Float32Array(WIDTH * HEIGHT);
const reliableOrientationMask = new Uint8Array(WIDTH * HEIGHT);
reliableOrientationMask.fill(1);
const orientationBridgeRemovedPixels =
  removeOrientationInconsistentBinaryBridges({
    binary: orientationBridgeBinary,
    orientationAngles: horizontalOrientationAngles,
    orientationMask: reliableOrientationMask,
    maskDistance: createFullMaskDistance(),
    width: WIDTH,
    height: HEIGHT,
    ridgePeriodPixels: RIDGE_PERIOD,
  });
assert.equal(orientationBridgeRemovedPixels > 0, true);
assert.equal(orientationBridgeBinary[44 * WIDTH + 48], 0);
assert.equal(orientationBridgeBinary[40 * WIDTH + 48], 1);
assert.equal(orientationBridgeBinary[48 * WIDTH + 48], 1);

// Aynı yöne aykırı görünen fakat bir ridge periyodundan uzun olan gerçek dal,
// kısa köprü filtresinin kapsamı dışında kalır.
const longBranchBinary = new Uint8Array(WIDTH * HEIGHT);
for (let y = 47; y <= 49; y += 1) {
  for (let x = 16; x <= 80; x += 1) {
    longBranchBinary[y * WIDTH + x] = 1;
  }
}
for (let y = 20; y <= 46; y += 1) {
  for (let x = 47; x <= 49; x += 1) {
    longBranchBinary[y * WIDTH + x] = 1;
  }
}
const longBranchRemovedPixels = removeOrientationInconsistentBinaryBridges({
  binary: longBranchBinary,
  orientationAngles: horizontalOrientationAngles,
  orientationMask: reliableOrientationMask,
  maskDistance: createFullMaskDistance(),
  width: WIDTH,
  height: HEIGHT,
  ridgePeriodPixels: RIDGE_PERIOD,
});
assert.equal(longBranchRemovedPixels, 0);
assert.equal(longBranchBinary[32 * WIDTH + 48], 1);
assert.equal(branchDiagnostics.bifurcationCrossingNumberCandidateCount > 0, true);
assert.equal(branchDiagnostics.bifurcationBranchValidatedCount > 0, true);
assert.equal(branchDiagnostics.bifurcationRingValidatedCount > 0, true);
assert.equal(branchDiagnostics.bifurcationCandidateCount, 1);
assert.equal(branchDiagnostics.bifurcationMicroCycleValidatedCount, 1);
assert.equal(branchDiagnostics.bifurcationStabilityValidatedCount, 1);
assert.equal(branchDiagnostics.bifurcationSuppressionCount, 1);

// Adaptif binary haritada noktasal gürültü silinirken gerçek ridge bileşeni korunur.
const binaryNoise = new Uint8Array(WIDTH * HEIGHT);
for (let x = 20; x <= 45; x += 1) binaryNoise[30 * WIDTH + x] = 1;
binaryNoise[10 * WIDTH + 10] = 1;
binaryNoise[10 * WIDTH + 11] = 1;
binaryNoise[10 * WIDTH + 12] = 1;
const removedBinaryNoise = removeSmallBinaryComponents(
  binaryNoise,
  WIDTH,
  HEIGHT,
  8
);
assert.equal(removedBinaryNoise, 3);
assert.equal(binaryNoise[30 * WIDTH + 20], 1);
assert.equal(binaryNoise[10 * WIDTH + 11], 0);

// Kalın gri ridge bandının adaptif eşik ve inceltme sonrasında iki uç verdiğini uçtan uca doğrular.
const thickPixels = new Uint8Array(WIDTH * HEIGHT * 4);
const thickMask = new Uint8Array(WIDTH * HEIGHT);
thickMask.fill(1);
for (let index = 0; index < WIDTH * HEIGHT; index += 1) {
  const offset = index * 4;
  thickPixels[offset] = 220;
  thickPixels[offset + 1] = 220;
  thickPixels[offset + 2] = 220;
  thickPixels[offset + 3] = 255;
}
for (let y = 46; y <= 50; y += 1) {
  for (let x = 20; x <= 75; x += 1) {
    const offset = (y * WIDTH + x) * 4;
    thickPixels[offset] = 35;
    thickPixels[offset + 1] = 35;
    thickPixels[offset + 2] = 35;
  }
}
const extracted = extractFingerprintMinutiae({
  pixels: thickPixels,
  mask: thickMask,
  width: WIDTH,
  height: HEIGHT,
  ridgePeriodPixels: RIDGE_PERIOD,
});
const extractedEndings = extracted.template.minutiae.filter(
  (item) => item.type === 'ending'
).length;
assert.equal(extracted.binaryVisualizationPixels[0], 255);
assert.equal(extracted.binaryVisualizationPixels[48 * WIDTH + 40], 0);
assert.equal(
  extracted.skeletonOverlayVisualizationPixels.length,
  WIDTH * HEIGHT * 4
);
assert.deepEqual(
  [...extracted.skeletonOverlayVisualizationPixels.slice(0, 4)],
  [0, 0, 0, 0]
);
const skeletonCenterOffset = (48 * WIDTH + 40) * 4;
assert.deepEqual(
  [
    ...extracted.skeletonOverlayVisualizationPixels.slice(
      skeletonCenterOffset,
      skeletonCenterOffset + 4
    ),
  ],
  [255, 255, 255, 255]
);

// Kaba yön maskesi bütün ROI'yi kapsasa bile yeşil hücreleri temsil eden ince
// doğrulama maskesinin dışındaki ridge, binary ve iskelet çıktısına taşınmaz.
const verifiedFineMask = new Uint8Array(WIDTH * HEIGHT);
for (let y = 12; y < 84; y += 1) {
  for (let x = 12; x < 60; x += 1) {
    verifiedFineMask[y * WIDTH + x] = 1;
  }
}
const orientationLimitedExtraction = extractFingerprintMinutiae({
  pixels: thickPixels,
  mask: thickMask,
  candidateMask: thickMask,
  orientationMask: thickMask,
  orientationFieldMask: verifiedFineMask,
  width: WIDTH,
  height: HEIGHT,
  ridgePeriodPixels: RIDGE_PERIOD,
});
const reliableSkeletonOffset = (48 * WIDTH + 40) * 4;
const unreliableSkeletonOffset = (48 * WIDTH + 70) * 4;
assert.equal(
  orientationLimitedExtraction.binaryVisualizationPixels[48 * WIDTH + 40],
  0
);
assert.equal(
  orientationLimitedExtraction.binaryVisualizationPixels[48 * WIDTH + 70],
  255
);
assert.equal(
  orientationLimitedExtraction.skeletonOverlayVisualizationPixels[
    reliableSkeletonOffset + 3
  ],
  255
);
assert.equal(
  orientationLimitedExtraction.skeletonOverlayVisualizationPixels[
    unreliableSkeletonOffset + 3
  ],
  0
);
const distalMask = createDistalMinutiaeMask(
  new Uint8Array(WIDTH * HEIGHT).fill(1),
  WIDTH,
  HEIGHT,
  0.75
);
assert.equal(distalMask[10 * WIDTH + 10], 1);
assert.equal(distalMask[90 * WIDTH + 10], 0);
if (extractedEndings !== 2) {
  throw new Error(
    `Kalın ridge uçtan uca testinde 2 son bekleniyordu, ${extractedEndings} bulundu.`
  );
}

// Yatay ridge alanında dikey iki uç birbirine baksa bile enine köprü yeniden kurulmaz.
const perpendicularGap = new Uint8Array(WIDTH * HEIGHT);
drawLine(perpendicularGap, 16, 40, 80, 40);
drawLine(perpendicularGap, 16, 48, 80, 48);
drawLine(perpendicularGap, 48, 40, 48, 42);
drawLine(perpendicularGap, 48, 46, 48, 48);
const perpendicularBridgePixels = bridgeShortSkeletonGaps(
  perpendicularGap,
  bridgeMask,
  WIDTH,
  HEIGHT,
  RIDGE_PERIOD,
  horizontalOrientationAngles,
  reliableOrientationMask
);
assert.equal(perpendicularBridgePixels, 0);
assert.equal(perpendicularGap[44 * WIDTH + 48], 0);
assert.equal(extracted.topologyDiagnostics.binary.componentCount, 1);
assert.equal(extracted.topologyDiagnostics.thinned.endingPixelCount, 2);
assert.equal(extracted.topologyDiagnostics.pruned.componentCount, 1);
assert.equal(extracted.topologyDiagnostics.pruned.endingPixelCount, 2);
assert.equal(extracted.topologyDiagnostics.pruned.bifurcationPixelCount, 0);

// Yakın eşiklerde konumu ve tipi tekrarlanmayan adaylar template'e taşınmaz.
const thresholdConsensus = selectThresholdStableMinutiaeCandidates({
  primaryCandidates: [
    { x: 20, y: 20, type: 'ending', angleDegrees: 20, confidence: 86 },
    { x: 40, y: 40, type: 'bifurcation', angleDegrees: 50, confidence: 88 },
    { x: 60, y: 60, type: 'ending', angleDegrees: 80, confidence: 84 },
    { x: 75, y: 30, type: 'bifurcation', angleDegrees: 110, confidence: 90 },
  ],
  variantCandidates: [
    [
      { x: 22, y: 19, type: 'ending', angleDegrees: 22, confidence: 83 },
      { x: 42, y: 39, type: 'bifurcation', angleDegrees: 48, confidence: 85 },
      { x: 61, y: 61, type: 'bifurcation', angleDegrees: 79, confidence: 82 },
      { x: 76, y: 31, type: 'bifurcation', angleDegrees: 108, confidence: 87 },
    ],
    [
      { x: 19, y: 21, type: 'ending', angleDegrees: 19, confidence: 85 },
      { x: 39, y: 42, type: 'bifurcation', angleDegrees: 52, confidence: 87 },
      { x: 74, y: 29, type: 'ending', angleDegrees: 111, confidence: 86 },
    ],
  ],
  ridgePeriodPixels: RIDGE_PERIOD,
});
assert.equal(thresholdConsensus.primaryCount, 4);
assert.equal(thresholdConsensus.locationStableCount, 4);
assert.equal(thresholdConsensus.typeStableCount, 2);
assert.deepEqual(
  thresholdConsensus.candidates.map((candidate) => candidate.type),
  ['ending', 'bifurcation']
);

// Normal ve relaxed eşiklerde aynı kalan küçük kapalı çevrim gerçek topoloji
// adayı sayılır; sıkı eşikte açılmış olsa bile iki eşik desteğiyle korunur.
const stableLoopSkeleton = new Uint8Array(WIDTH * HEIGHT);
drawLine(stableLoopSkeleton, 20, 40, 76, 40);
drawLine(stableLoopSkeleton, 20, 48, 76, 48);
drawLine(stableLoopSkeleton, 46, 40, 46, 48);
drawLine(stableLoopSkeleton, 56, 40, 56, 48);
const openedLoopVariant = new Uint8Array(stableLoopSkeleton);
for (let y = 41; y < 48; y += 1) {
  openedLoopVariant[y * WIDTH + 56] = 0;
}
const preservedLoopSkeleton = new Uint8Array(stableLoopSkeleton);
const preservedLoopResult = removeThresholdUnstableSkeletonLoops({
  skeleton: preservedLoopSkeleton,
  variantSkeletons: [stableLoopSkeleton, openedLoopVariant],
  width: WIDTH,
  height: HEIGHT,
  ridgePeriodPixels: RIDGE_PERIOD,
  orientationAngles: horizontalOrientationAngles,
  orientationMask: reliableOrientationMask,
});
assert.equal(preservedLoopResult.candidateCount, 1);
assert.equal(preservedLoopResult.stableCount, 1);
assert.equal(preservedLoopResult.removedLoopCount, 0);
assert.deepEqual(preservedLoopSkeleton, stableLoopSkeleton);

// Yalnızca normal eşikte oluşan ve yatay yön alanına dik ilerleyen bağlantı
// kesilir; çevrim açıldıktan sonra aynı bölge yeniden halka sayılmamalıdır.
const unstableLoopSkeleton = new Uint8Array(stableLoopSkeleton);
const unstableLoopResult = removeThresholdUnstableSkeletonLoops({
  skeleton: unstableLoopSkeleton,
  variantSkeletons: [openedLoopVariant, openedLoopVariant],
  width: WIDTH,
  height: HEIGHT,
  ridgePeriodPixels: RIDGE_PERIOD,
  orientationAngles: horizontalOrientationAngles,
  orientationMask: reliableOrientationMask,
});
assert.equal(unstableLoopResult.candidateCount, 1);
assert.equal(unstableLoopResult.stableCount, 0);
assert.equal(unstableLoopResult.unstableCount, 1);
assert.equal(unstableLoopResult.removedLoopCount, 1);
assert.ok(unstableLoopResult.removedPixelCount > 0);
const recheckedLoopResult = removeThresholdUnstableSkeletonLoops({
  skeleton: unstableLoopSkeleton,
  variantSkeletons: [openedLoopVariant, openedLoopVariant],
  width: WIDTH,
  height: HEIGHT,
  ridgePeriodPixels: RIDGE_PERIOD,
  orientationAngles: horizontalOrientationAngles,
  orientationMask: reliableOrientationMask,
});
assert.equal(recheckedLoopResult.candidateCount, 0);

console.info(
  `[Minutiae testi] arama_bileşeni=${countMaskComponents(stableSearchArea.mask)}, düşük_destek=${weakSupportMinutiae.length}, sınır_sonu=${boundaryFilteredMinutiae.length}, mikro_delik=${filledHolePixels}, düz_son=${straightEndings}, y_son=${branchEndings}, y_çatallanma=${bifurcations}, mikro_enclosure=${microEnclosureBifurcations}, geniş_enclosure=${wideEnclosureBifurcations}, çapraz_çatallanma=${crossLinkedBifurcations}, yön_köprüsü=${orientationBridgeRemovedPixels}, uzun_dal=${longBranchRemovedPixels}, enine_yeniden_bağ=${perpendicularBridgePixels}, merdiven_çatallanma=${ladderBifurcations}, spur_budanan=${prunedSpurPixels}, köprü=${bridgedPixelCount}, kopuk_son=${brokenRidgeEndings}, küçük_bileşen=${removedComponentPixels}, kalın_son=${extractedEndings}, eşik=${thresholdConsensus.primaryCount}/${thresholdConsensus.locationStableCount}/${thresholdConsensus.typeStableCount}, eşik_halkası=${unstableLoopResult.candidateCount}/${unstableLoopResult.stableCount}/${unstableLoopResult.removedLoopCount}/${unstableLoopResult.removedPixelCount}px, inceltme=${extracted.thinningIterations}`
);
