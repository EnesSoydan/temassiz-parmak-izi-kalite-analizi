import assert from 'node:assert/strict';
import {
  calculateRidgeScaleFactor,
  rescaleRgbaAndMask,
} from '../src/lib/ridge-scale.ts';

assert.equal(calculateRidgeScaleFactor(8), 1);
assert.ok(calculateRidgeScaleFactor(10) < 1);
assert.ok(calculateRidgeScaleFactor(6) > 1);
assert.equal(calculateRidgeScaleFactor(0), 1);

const pixels = new Uint8Array([
  0, 0, 0, 255,
  100, 100, 100, 255,
  200, 200, 200, 255,
  255, 255, 255, 255,
]);
const mask = new Uint8Array([1, 0, 0, 1]);
const scaled = rescaleRgbaAndMask({
  pixels,
  mask,
  width: 2,
  height: 2,
  scaleFactor: 2,
});
assert.deepEqual(
  { width: scaled.width, height: scaled.height },
  { width: 4, height: 4 }
);
assert.equal(scaled.pixels.length, 4 * 4 * 4);
assert.equal(scaled.mask[0], 1);
assert.equal(scaled.mask[scaled.mask.length - 1], 1);

console.log('Ridge ölçekleme testleri başarılı.');
