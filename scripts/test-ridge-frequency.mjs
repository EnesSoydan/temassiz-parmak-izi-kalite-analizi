import assert from 'node:assert/strict';

import { getScaleAwareEvidenceMinimums } from '../src/lib/fingerprint-quality-config.ts';
import { estimateOrientationField } from '../src/lib/orientation-field.ts';
import { createOrientationVisualization } from '../src/lib/orientation-visualization.ts';
import { estimateRidgeFrequency } from '../src/lib/ridge-frequency.ts';

const WIDTH = 96;
const HEIGHT = 96;
const RIDGE_PERIOD = 7;

// Sentetik görüntülerin tamamını geçerli parmak alanı kabul eden test maskesi üretir.
function createFullMask() {
  return new Uint8Array(WIDTH * HEIGHT).fill(1);
}

// Dikey ridge çizgileriyle uyumlu, komşuluk devamlılığı yüksek test blokları üretir.
function createOrientationBlocks() {
  return [20, 44].flatMap((top) =>
    [20, 44].map((left) => ({
      left,
      top,
      size: 24,
      angleRadians: Math.PI / 2,
      coherence: 0.9,
      neighborhoodConsistency: 0.95,
      gradientEnergy: 500,
      maskCoverage: 1,
    }))
  );
}

// Bilinen periyotta düzenli dikey ridge/valley sinyali üretir.
function createPeriodicRidges(amplitude = 48) {
  const image = new Uint8Array(WIDTH * HEIGHT);

  for (let y = 0; y < HEIGHT; y += 1) {
    for (let x = 0; x < WIDTH; x += 1) {
      image[y * WIDTH + x] = Math.round(
        128 + Math.sin((Math.PI * 2 * x) / RIDGE_PERIOD) * amplitude
      );
    }
  }

  return image;
}

// Perspektif etkisini taklit etmek için periyodu görüntü boyunca yavaşça değişen ridge sinyali üretir.
function createGraduallyVaryingRidges() {
  const image = new Uint8Array(WIDTH * HEIGHT);
  const phases = new Float64Array(WIDTH);
  let phase = 0;

  for (let x = 0; x < WIDTH; x += 1) {
    const period = 7 + (x / (WIDTH - 1)) * 2;
    phase += (Math.PI * 2) / period;
    phases[x] = phase;
  }

  for (let y = 0; y < HEIGHT; y += 1) {
    for (let x = 0; x < WIDTH; x += 1) {
      image[y * WIDTH + x] = Math.round(128 + Math.sin(phases[x]) * 40);
    }
  }

  return image;
}

// Periyodik yapı içermeyen düz görüntü üretir.
function createFlatImage() {
  return new Uint8Array(WIDTH * HEIGHT).fill(128);
}

// Yalnızca yavaş parlaklık değişimi içeren görüntü üretir.
function createGradientImage() {
  const image = new Uint8Array(WIDTH * HEIGHT);

  for (let y = 0; y < HEIGHT; y += 1) {
    for (let x = 0; x < WIDTH; x += 1) {
      image[y * WIDTH + x] = Math.round(50 + (x / (WIDTH - 1)) * 150);
    }
  }

  return image;
}

// Her çalıştırmada aynı sonucu veren periyotsuz gürültü görüntüsü üretir.
function createDeterministicNoise() {
  const image = new Uint8Array(WIDTH * HEIGHT);
  let state = 123456789;

  for (let index = 0; index < image.length; index += 1) {
    state = (state * 1664525 + 1013904223) >>> 0;
    image[index] = state >>> 24;
  }

  return image;
}

// Verilen sentetik görüntünün ridge frekans özetini ortak ayarlarla hesaplar.
function analyze(grayscale) {
  return estimateRidgeFrequency({
    grayscale,
    mask: createFullMask(),
    width: WIDTH,
    height: HEIGHT,
    orientationBlocks: createOrientationBlocks(),
  });
}

const periodic = analyze(createPeriodicRidges());
const lowContrastPeriodic = analyze(createPeriodicRidges(4));
const varyingPeriodic = analyze(createGraduallyVaryingRidges());
const flat = analyze(createFlatImage());
const gradient = analyze(createGradientImage());
const noise = analyze(createDeterministicNoise());
const scaleAwareSmallRoi = getScaleAwareEvidenceMinimums(47);
const orientationField = estimateOrientationField({
  grayscale: createPeriodicRidges(),
  mask: createFullMask(),
  width: WIDTH,
  height: HEIGHT,
  blockSize: 24,
  minMaskCoverage: 0.6,
});
const orientationVisualization = createOrientationVisualization({
  grayscale: createPeriodicRidges(),
  mask: createFullMask(),
  width: WIDTH,
  height: HEIGHT,
  orientationBlocks: orientationField,
});

assert.ok(periodic.ridgePeriodicity >= 55, 'Düzenli ridge sinyali yüksek periyodiklik vermeli.');
assert.ok(periodic.validBlockRatio >= 0.75, 'Düzenli ridge bloklarının çoğu geçerli olmalı.');
assert.ok(
  Math.abs(periodic.medianPeriodPixels - RIDGE_PERIOD) <= 1,
  'Bulunan ridge periyodu sentetik periyoda yakın olmalı.'
);
assert.ok(
  lowContrastPeriodic.ridgePeriodicity >= 35,
  'Düşük kontrastlı düzenli ridge sinyali algılanabilmeli.'
);
assert.ok(
  lowContrastPeriodic.validBlockRatio >= 0.5,
  'Düşük kontrastlı ridge bloklarının çoğu geçerli olmalı.'
);
assert.ok(
  varyingPeriodic.ridgeFrequencyConsistency >= 45,
  'Yavaş değişen ridge periyodu yerel frekans haritasında tutarlı sayılmalı.'
);
assert.equal(flat.validBlockCount, 0, 'Düz görüntüde ridge bloğu bulunmamalı.');
assert.ok(gradient.ridgePeriodicity <= 15, 'Yavaş gradyan ridge olarak kabul edilmemeli.');
assert.ok(noise.validBlockRatio <= 0.25, 'Rastgele gürültü ridge olarak kabul edilmemeli.');
assert.equal(
  scaleAwareSmallRoi.medium,
  4,
  'Küçük ROI orta kalite için büyük parmakla aynı sabit blok sayısına zorlanmamalı.'
);
assert.ok(
  orientationField.some((block) => block.gridOffset === 0) &&
    orientationField.some((block) => block.gridOffset !== 0),
  'Orientation alanı ana ve yarım-blok kaydırılmış iki ızgarayı kullanmalı.'
);
assert.ok(
  orientationField.every(
    (block) =>
      Number.isFinite(block.angleRadians) &&
      (block.smoothedCoherence ?? 0) >= 0 &&
      (block.smoothedCoherence ?? 0) <= 1
  ),
  'Yumuşatılmış orientation değerleri geçerli aralıkta kalmalı.'
);
assert.ok(
  orientationVisualization.reliableBlockRatio >= 0.75,
  'Düzenli sentetik ridge alanında yön bloklarının çoğu güvenilir olmalı.'
);
assert.ok(
  countColoredOrientationPixels(orientationVisualization.pixels) > 0,
  'Orientation haritası ridge eksenlerini renkli çizgilerle göstermeli.'
);

// Orientation görselindeki gri tabandan farklı renkli yön çizgilerini sayar.
function countColoredOrientationPixels(pixels) {
  let count = 0;
  for (let index = 0; index < pixels.length; index += 4) {
    if (pixels[index] !== pixels[index + 1] || pixels[index + 1] !== pixels[index + 2]) {
      count += 1;
    }
  }
  return count;
}

console.info(
  `[Ridge testi] düzenli=${periodic.ridgePeriodicity}/${Math.round(periodic.validBlockRatio * 100)}%, düşük_kontrast=${lowContrastPeriodic.ridgePeriodicity}/${Math.round(lowContrastPeriodic.validBlockRatio * 100)}%, değişken_frekans=${varyingPeriodic.ridgeFrequencyConsistency}, düz=${flat.ridgePeriodicity}, gradyan=${gradient.ridgePeriodicity}, gürültü=${noise.ridgePeriodicity}/${Math.round(noise.validBlockRatio * 100)}%`
);
