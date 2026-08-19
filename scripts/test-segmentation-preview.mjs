import assert from 'node:assert/strict';

import { createSegmentedPreviewPixels } from '../src/lib/segmentation-preview.ts';

const WIDTH = 32;
const HEIGHT = 24;
const PIXEL_COUNT = WIDTH * HEIGHT;

function createSource(outsideValue) {
  const pixels = new Uint8Array(PIXEL_COUNT * 4);
  const mask = new Uint8Array(PIXEL_COUNT);

  for (let y = 0; y < HEIGHT; y += 1) {
    for (let x = 0; x < WIDTH; x += 1) {
      const index = y * WIDTH + x;
      const inside = x >= 5 && x < 27 && y >= 4 && y < 20;
      const ridge = inside && x >= 14 && x <= 16;
      const value = inside ? (ridge ? 92 : 148) : outsideValue;
      const dataIndex = index * 4;
      mask[index] = inside ? 1 : 0;
      pixels[dataIndex] = value;
      pixels[dataIndex + 1] = value;
      pixels[dataIndex + 2] = value;
      pixels[dataIndex + 3] = 255;
    }
  }

  return { pixels, mask };
}

const darkOutside = createSource(0);
const brightOutside = createSource(255);
const first = createSegmentedPreviewPixels(
  darkOutside.pixels,
  darkOutside.mask,
  WIDTH,
  HEIGHT
);
const second = createSegmentedPreviewPixels(
  brightOutside.pixels,
  brightOutside.mask,
  WIDTH,
  HEIGHT
);
const byteMask = new Uint8Array(brightOutside.mask);
for (let index = 0; index < byteMask.length; index += 1) {
  if (byteMask[index]) byteMask[index] = 255;
}
const byteMaskPreview = createSegmentedPreviewPixels(
  brightOutside.pixels,
  byteMask,
  WIDTH,
  HEIGHT
);

for (let index = 0; index < PIXEL_COUNT; index += 1) {
  const dataIndex = index * 4;
  if (!darkOutside.mask[index]) {
    assert.equal(first[dataIndex], 0);
    continue;
  }

  assert.equal(first[dataIndex], second[dataIndex]);
  assert.equal(first[dataIndex], byteMaskPreview[dataIndex]);
  assert.ok(first[dataIndex] >= 32 && first[dataIndex] <= 224);
  assert.equal(first[dataIndex], first[dataIndex + 1]);
  assert.equal(first[dataIndex], first[dataIndex + 2]);
  assert.equal(first[dataIndex + 3], 255);
}

const ridgeValue = first[(12 * WIDTH + 15) * 4];
const valleyValue = first[(12 * WIDTH + 20) * 4];
assert.ok(ridgeValue < valleyValue, 'Koyu ridge çizgisi önizlemede korunamadı.');

console.info(
  `[Seg önizleme testi] ridge=${ridgeValue}, valley=${valleyValue}, aralık=32-224`
);
