import assert from 'node:assert/strict';

import {
  compareExposurePair,
  selectExposureProcessingSource,
} from '../src/lib/exposure-pair-analysis.ts';
import { convertNitroPixelsToRgba } from '../src/lib/nitro-pixel-data.ts';

const WIDTH = 96;
const HEIGHT = 128;

// Merkezdeki sıcak ten rengi yanlışlıkla BGRA diye etiketlenirse kırmızı-mavi düzeltmesini sınar.
function createMislabeledSkinPixels() {
  const pixels = new Uint8Array(24 * 32 * 4);
  for (let index = 0; index < 24 * 32; index += 1) {
    const offset = index * 4;
    pixels[offset] = 184;
    pixels[offset + 1] = 118;
    pixels[offset + 2] = 96;
    pixels[offset + 3] = 255;
  }
  return pixels;
}

// Düzenli çizgili sentetik ROI üretip ikinci pozdaki küçük kaymayı ve parlamayı sınar.
function createExposure({ shiftX = 0, glareCenter = false }) {
  const pixels = new Uint8Array(WIDTH * HEIGHT * 4);
  const mask = new Uint8Array(WIDTH * HEIGHT).fill(1);

  for (let y = 0; y < HEIGHT; y += 1) {
    for (let x = 0; x < WIDTH; x += 1) {
      const index = y * WIDTH + x;
      const pixelIndex = index * 4;
      const shiftedX = x - shiftX;
      const ridge = Math.sin((shiftedX / 8) * Math.PI * 2) >= 0 ? 72 : 172;
      const isGlare =
        glareCenter && x >= 30 && x < 66 && y >= 36 && y < 92;
      const value = isGlare ? 248 : ridge;
      pixels[pixelIndex] = value;
      pixels[pixelIndex + 1] = value;
      pixels[pixelIndex + 2] = value;
      pixels[pixelIndex + 3] = 255;
    }
  }

  return {
    imageUri: 'test.jpg',
    pixels,
    mask,
    width: WIDTH,
    height: HEIGHT,
    coverage: 1,
    silhouetteAxisDegrees: 90,
    timings: { decodeMs: 0, maskMs: 0, enhancementMs: 0, encodeMs: 0, writeMs: 0 },
  };
}

// Sol ve sağ yarıda farklı ridge kontrastı üreterek iki pozun birbirini tamamladığı durumu sınar.
function createSplitExposure(leftContrast, rightContrast) {
  const pixels = new Uint8Array(WIDTH * HEIGHT * 4);
  const mask = new Uint8Array(WIDTH * HEIGHT).fill(1);

  for (let y = 0; y < HEIGHT; y += 1) {
    for (let x = 0; x < WIDTH; x += 1) {
      const contrast = x < WIDTH / 2 ? leftContrast : rightContrast;
      const ridgeSign = Math.sin((x / 8) * Math.PI * 2) >= 0 ? 1 : -1;
      const value = Math.round(128 + ridgeSign * contrast);
      const dataIndex = (y * WIDTH + x) * 4;
      pixels[dataIndex] = value;
      pixels[dataIndex + 1] = value;
      pixels[dataIndex + 2] = value;
      pixels[dataIndex + 3] = 255;
    }
  }

  return { pixels, mask, width: WIDTH, height: HEIGHT };
}

const ambient = createExposure({});
const shiftedFlash = createExposure({ shiftX: 4, glareCenter: true });
const comparison = compareExposurePair(ambient, shiftedFlash);

assert.ok(comparison.alignmentConfidence >= 70);
assert.ok(comparison.comparableBlockCount >= 100);
assert.ok(comparison.flashGlareRatio > comparison.ambientGlareRatio);
assert.ok(comparison.ambientBetterBlockRatio > 0);

const correctedSkin = convertNitroPixelsToRgba({
  source: createMislabeledSkinPixels(),
  width: 24,
  height: 32,
  pixelFormat: 'BGRA',
});
assert.equal(correctedSkin.redBlueCorrected, true);
assert.deepEqual(Array.from(correctedSkin.pixels.slice(0, 4)), [184, 118, 96, 255]);

const correctlyLabeledBgra = createMislabeledSkinPixels();
for (let index = 0; index < correctlyLabeledBgra.length; index += 4) {
  correctlyLabeledBgra[index] = 96;
  correctlyLabeledBgra[index + 2] = 184;
}
const unchangedSkin = convertNitroPixelsToRgba({
  source: correctlyLabeledBgra,
  width: 24,
  height: 32,
  pixelFormat: 'BGRA',
});
assert.equal(unchangedSkin.redBlueCorrected, false);
assert.deepEqual(Array.from(unchangedSkin.pixels.slice(0, 4)), [184, 118, 96, 255]);

const complementarySelection = selectExposureProcessingSource(
  createSplitExposure(62, 26),
  createSplitExposure(26, 62)
);
assert.equal(complementarySelection.processingSource, 'local');
assert.ok(complementarySelection.flashPixelRatio >= 20);
assert.ok(complementarySelection.flashPixelRatio <= 80);

const flashSelection = selectExposureProcessingSource(
  createSplitExposure(8, 8),
  createSplitExposure(62, 62)
);
assert.equal(flashSelection.processingSource, 'flash');
assert.ok(flashSelection.flashPixelRatio >= 70);

const selectedFlash = selectExposureProcessingSource(
  createSplitExposure(8, 8),
  createSplitExposure(62, 62),
  compareExposurePair(createSplitExposure(8, 8), createSplitExposure(62, 62))
);
assert.equal(selectedFlash.processingSource, 'flash');
assert.ok(selectedFlash.flashPixelRatio >= 70);

console.info(
  `[Poz testi] hizalama=${comparison.alignmentConfidence}%, blok=${comparison.comparableBlockCount}, flaşsız=${comparison.ambientBetterBlockRatio}%, flaşlı=${comparison.flashBetterBlockRatio}%, parlama=${comparison.ambientGlareRatio}%/${comparison.flashGlareRatio}%, öneri=${comparison.recommendation}, yerel_flaş=${complementarySelection.flashPixelRatio}%`
);
