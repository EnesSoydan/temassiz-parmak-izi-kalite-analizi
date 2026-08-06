import assert from 'node:assert/strict';
import {
  createQuadToRectangleHomography,
  createLetterboxTransform,
  getAspectPreservingSize,
  getCanonicalGeometry,
  mapModelPointToNormalizedImage,
  orderQuadrilateralPoints,
  transformPointByHomography,
  warpRgbaImageByHomography,
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

function assertPointClose(actual, expected, tolerance = 0.001) {
  assert.ok(Math.abs(actual.x - expected.x) <= tolerance);
  assert.ok(Math.abs(actual.y - expected.y) <= tolerance);
}

function assertQuadMapsToTarget(
  sourcePoints,
  targetWidth,
  targetHeight,
  longEdgeAxis = 'horizontal'
) {
  const homography = createQuadToRectangleHomography(
    sourcePoints,
    targetWidth,
    targetHeight,
    longEdgeAxis
  );
  const orderedSource = orderQuadrilateralPoints(sourcePoints);
  const edgeLengths = orderedSource.map((point, index) => {
    const next = orderedSource[(index + 1) % orderedSource.length];
    return Math.hypot(next.x - point.x, next.y - point.y);
  });
  let canonicalSource;
  if (longEdgeAxis === 'vertical') {
    const firstShortEdgeIndex = edgeLengths.indexOf(Math.min(...edgeLengths));
    const oppositeShortEdgeIndex = (firstShortEdgeIndex + 2) % 4;
    const centerY = (edgeIndex) => {
      const first = orderedSource[edgeIndex];
      const second = orderedSource[(edgeIndex + 1) % 4];
      return (first.y + second.y) / 2;
    };
    const topEdgeIndex =
      centerY(firstShortEdgeIndex) <= centerY(oppositeShortEdgeIndex)
        ? firstShortEdgeIndex
        : oppositeShortEdgeIndex;
    const bottomEdgeIndex = (topEdgeIndex + 2) % 4;
    const topEdge = [
      orderedSource[topEdgeIndex],
      orderedSource[(topEdgeIndex + 1) % 4],
    ].sort((first, second) => first.x - second.x);
    const bottomEdge = [
      orderedSource[bottomEdgeIndex],
      orderedSource[(bottomEdgeIndex + 1) % 4],
    ].sort((first, second) => first.x - second.x);
    canonicalSource = [topEdge[0], topEdge[1], bottomEdge[1], bottomEdge[0]];
  } else {
    const longEdgeIndex = edgeLengths.indexOf(Math.max(...edgeLengths));
    canonicalSource = orderedSource
      .slice(longEdgeIndex)
      .concat(orderedSource.slice(0, longEdgeIndex));
  }
  assert.equal(homography.length, 9);
  const targetCorners = [
    { x: 0, y: 0 },
    { x: targetWidth - 1, y: 0 },
    { x: targetWidth - 1, y: targetHeight - 1 },
    { x: 0, y: targetHeight - 1 },
  ];
  for (const [index, sourcePoint] of canonicalSource.entries()) {
    const mapped = transformPointByHomography(sourcePoint, homography);
    assertPointClose(mapped, targetCorners[index]);
  }
  assert.equal(orderedSource.length, 4);
  return homography;
}

const perspectiveQuad = [
  { x: 28, y: 20 },
  { x: 190, y: 5 },
  { x: 176, y: 82 },
  { x: 12, y: 96 },
];
const perspectiveHomography = assertQuadMapsToTarget(
  perspectiveQuad,
  256,
  128
);
assert.ok(perspectiveHomography.some((value) => Math.abs(value) > 0.001));

const verticalQuad = [
  { x: 40, y: 10 },
  { x: 70, y: 14 },
  { x: 62, y: 210 },
  { x: 32, y: 206 },
];
assertQuadMapsToTarget(verticalQuad, 128, 256, 'vertical');

// Dikey ve eksen hizalı ROI, homografi sonrası 180 derece dönmemelidir.
const verticalIdentity = createQuadToRectangleHomography(
  [
    { x: 0, y: 0 },
    { x: 3, y: 0 },
    { x: 3, y: 5 },
    { x: 0, y: 5 },
  ],
  4,
  6,
  'vertical'
);
assertPointClose(
  transformPointByHomography({ x: 0, y: 0 }, verticalIdentity),
  { x: 0, y: 0 }
);
assertPointClose(
  transformPointByHomography({ x: 3, y: 5 }, verticalIdentity),
  { x: 3, y: 5 }
);

// Sol ve sağ uzun kenarların çok küçük uzunluk farkı kanonik yönü ters çevirmemelidir.
const unequalLongEdges = createQuadToRectangleHomography(
  [
    { x: 0.2, y: 0 },
    { x: 3, y: 0.1 },
    { x: 3.1, y: 5 },
    { x: 0, y: 5.2 },
  ],
  4,
  6,
  'vertical'
);
assert.ok(
  transformPointByHomography({ x: 0.2, y: 0 }, unequalLongEdges).y < 0.001
);
assert.ok(
  transformPointByHomography({ x: 0, y: 5.2 }, unequalLongEdges).y > 4.999
);

const sourcePixels = new Uint8Array(4 * 4 * 4);
for (let index = 0; index < 16; index += 1) {
  sourcePixels[index * 4] = index;
  sourcePixels[index * 4 + 1] = index;
  sourcePixels[index * 4 + 2] = index;
  sourcePixels[index * 4 + 3] = 255;
}
const identityHomography = createQuadToRectangleHomography(
  [
    { x: 0, y: 0 },
    { x: 3, y: 0 },
    { x: 3, y: 3 },
    { x: 0, y: 3 },
  ],
  4,
  4
);
const warped = warpRgbaImageByHomography({
  pixels: sourcePixels,
  width: 4,
  height: 4,
  matrix: identityHomography,
  targetWidth: 4,
  targetHeight: 4,
});
assert.deepEqual([...warped.slice(0, 4)], [0, 0, 0, 255]);
assert.deepEqual([...warped.slice(-4)], [15, 15, 15, 255]);

console.log('ROI hizalama geometrisi testleri başarılı.');
