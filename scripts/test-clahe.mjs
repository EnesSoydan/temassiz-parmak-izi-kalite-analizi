import assert from 'node:assert/strict';
import { createMaskedClaheGrayscale } from '../src/lib/clahe.ts';

const width = 64;
const height = 64;
const grayscale = new Uint8Array(width * height);
const mask = new Uint8Array(width * height).fill(1);

for (let y = 0; y < height; y += 1) {
  for (let x = 0; x < width; x += 1) {
    grayscale[y * width + x] = 90 + ((x + y) % 8) * 3;
  }
}

const enhanced = createMaskedClaheGrayscale(grayscale, mask, width, height);
assert.equal(enhanced.length, grayscale.length);
assert.ok(enhanced.some((value) => value !== grayscale[0]));
assert.ok(enhanced.every((value) => value >= 0 && value <= 255));

console.log('Maskeli CLAHE testleri başarılı.');
