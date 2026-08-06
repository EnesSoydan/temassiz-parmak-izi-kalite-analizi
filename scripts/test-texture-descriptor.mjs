import assert from 'node:assert/strict';
import {
  cosineSimilarity,
  createRidgeTextureDescriptor,
} from '../src/lib/texture-descriptor.ts';

const width = 32;
const height = 32;
const pixels = new Uint8Array(width * height * 4);
const mask = new Uint8Array(width * height).fill(1);
for (let index = 0; index < width * height; index += 1) {
  const value = index % 2 === 0 ? 70 : 190;
  pixels[index * 4] = value;
  pixels[index * 4 + 1] = value;
  pixels[index * 4 + 2] = value;
  pixels[index * 4 + 3] = 255;
}
const descriptor = createRidgeTextureDescriptor({ pixels, mask, width, height });
assert.equal(descriptor.length, 64);
assert.ok(Math.abs(cosineSimilarity(descriptor, descriptor) - 1) < 1e-6);
assert.equal(cosineSimilarity(descriptor, descriptor.slice(0, 32)), null);

console.log('Ridge texture descriptor testleri başarılı.');
