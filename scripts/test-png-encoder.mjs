import assert from 'node:assert/strict';
import { encodeGrayscalePng } from '../src/lib/png-encoder.ts';

const png = encodeGrayscalePng(new Uint8Array([1, 0, 1, 1]), 2, 2);
assert.deepEqual([...png.slice(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10]);
assert.equal(String.fromCharCode(...png.slice(12, 16)), 'IHDR');
assert.equal(String.fromCharCode(...png.slice(-8, -4)), 'IEND');

console.log('PNG maske encoder testleri başarılı.');
