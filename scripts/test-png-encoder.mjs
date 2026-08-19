import assert from 'node:assert/strict';
import {
  encodeGrayscalePng,
  encodeRgbaPng,
} from '../src/lib/png-encoder.ts';

const png = encodeGrayscalePng(new Uint8Array([1, 0, 1, 1]), 2, 2);
assert.deepEqual([...png.slice(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10]);
assert.equal(String.fromCharCode(...png.slice(12, 16)), 'IHDR');
assert.equal(String.fromCharCode(...png.slice(-8, -4)), 'IEND');

const rgbaPng = encodeRgbaPng(
  new Uint8Array([
    255, 145, 45, 255,
    55, 180, 255, 255,
  ]),
  2,
  1
);
assert.deepEqual([...rgbaPng.slice(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10]);
assert.equal(rgbaPng[24], 8);
assert.equal(rgbaPng[25], 6);
assert.equal(String.fromCharCode(...rgbaPng.slice(-8, -4)), 'IEND');

console.log('PNG maske encoder testleri başarılı.');
