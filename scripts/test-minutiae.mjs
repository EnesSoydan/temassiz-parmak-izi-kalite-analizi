import {
  bridgeShortSkeletonGaps,
  createMaskDistanceMap,
  createStableMinutiaeSearchArea,
  detectMinutiaeFromSkeleton,
  extractFingerprintMinutiae,
  fillSmallBinaryRidgeHoles,
  pruneShortSkeletonBranches,
  removeSmallSkeletonComponents,
} from '../src/lib/minutiae-extraction.ts';

const WIDTH = 96;
const HEIGHT = 96;
const RIDGE_PERIOD = 8;

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

if (branchEndings !== 3 || bifurcations !== 1) {
  throw new Error(
    `Y ridge için 3 son/1 çatallanma bekleniyordu, ${branchEndings}/${bifurcations} bulundu.`
  );
}

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
if (extractedEndings !== 2) {
  throw new Error(
    `Kalın ridge uçtan uca testinde 2 son bekleniyordu, ${extractedEndings} bulundu.`
  );
}

console.info(
  `[Minutiae testi] arama_bileşeni=${countMaskComponents(stableSearchArea.mask)}, düşük_destek=${weakSupportMinutiae.length}, sınır_sonu=${boundaryFilteredMinutiae.length}, mikro_delik=${filledHolePixels}, düz_son=${straightEndings}, y_son=${branchEndings}, y_çatallanma=${bifurcations}, çapraz_çatallanma=${crossLinkedBifurcations}, merdiven_çatallanma=${ladderBifurcations}, spur_budanan=${prunedSpurPixels}, köprü=${bridgedPixelCount}, kopuk_son=${brokenRidgeEndings}, küçük_bileşen=${removedComponentPixels}, kalın_son=${extractedEndings}, inceltme=${extracted.thinningIterations}`
);
