import assert from 'node:assert/strict';
import {
  createLetterboxTransform,
  getAspectPreservingSize,
  getCanonicalGeometry,
  mapModelPointToNormalizedImage,
} from '../src/lib/roi-geometry.ts';

const landscape = createLetterboxTransform(4000, 3000, 480);
assert.deepEqual(
  {
    scaledWidth: landscape.scaledWidth,
    scaledHeight: landscape.scaledHeight,
    padX: landscape.padX,
    padY: landscape.padY,
  },
  { scaledWidth: 480, scaledHeight: 360, padX: 0, padY: 60 }
);
assert.deepEqual(
  mapModelPointToNormalizedImage(240, 240, landscape),
  { x: 0.5, y: 0.5 }
);
assert.deepEqual(
  mapModelPointToNormalizedImage(0, 60, landscape),
  { x: 0, y: 0 }
);
assert.deepEqual(
  mapModelPointToNormalizedImage(480, 420, landscape),
  { x: 1, y: 1 }
);

const portrait = createLetterboxTransform(3000, 4000, 480);
assert.deepEqual(
  mapModelPointToNormalizedImage(60, 0, portrait),
  { x: 0, y: 0 }
);
assert.deepEqual(
  mapModelPointToNormalizedImage(420, 480, portrait),
  { x: 1, y: 1 }
);

assert.deepEqual(getAspectPreservingSize(640, 480, 480), {
  width: 480,
  height: 360,
});

const verticalGeometry = getCanonicalGeometry([
  { x: 0.45, y: 0.2 },
  { x: 0.55, y: 0.2 },
  { x: 0.55, y: 0.8 },
  { x: 0.45, y: 0.8 },
]);
assert.ok(verticalGeometry);
assert.equal(verticalGeometry.rotationDegrees, 0);
assert.ok(verticalGeometry.longEdge > verticalGeometry.shortEdge);

const diagonalGeometry = getCanonicalGeometry([
  { x: 0, y: 0 },
  { x: 1, y: 1 },
  { x: 0.75, y: 1.25 },
  { x: -0.25, y: 0.25 },
]);
assert.ok(diagonalGeometry);
assert.ok(Math.abs(diagonalGeometry.rotationDegrees - 45) < 0.001);

console.log('ROI hizalama geometrisi testleri başarılı.');
