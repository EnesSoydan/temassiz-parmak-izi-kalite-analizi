import assert from 'node:assert/strict';

import {
  analyzeLiveRoiPixels,
  createInitialLiveCaptureGateState,
  updateLiveCaptureGate,
} from '../src/lib/live-capture-quality.ts';

const WIDTH = 160;
const HEIGHT = 120;
const FINGERS = ['index', 'middle', 'ring', 'pinky'];
const FRAME_SIZE = { width: WIDTH, height: HEIGHT };

function createDetection(className, centerX, overrides = {}) {
  const centerY = overrides.centerY ?? 0.52;
  const frameShortEdge = Math.min(WIDTH, HEIGHT);
  const shortEdge = (overrides.shortEdgeRatio ?? 0.12) * frameShortEdge;
  const longEdge = (overrides.longEdgeRatio ?? 0.19) * frameShortEdge;
  const confidence = overrides.confidence ?? 0.82;
  const angle = overrides.angle ?? Math.PI / 2;
  const centerPixels = { x: centerX * WIDTH, y: centerY * HEIGHT };
  const longAxis = {
    x: Math.cos(angle) * longEdge * 0.5,
    y: Math.sin(angle) * longEdge * 0.5,
  };
  const shortAxis = {
    x: -Math.sin(angle) * shortEdge * 0.5,
    y: Math.cos(angle) * shortEdge * 0.5,
  };
  const pixelCorners = [
    {
      x: centerPixels.x - longAxis.x - shortAxis.x,
      y: centerPixels.y - longAxis.y - shortAxis.y,
    },
    {
      x: centerPixels.x + longAxis.x - shortAxis.x,
      y: centerPixels.y + longAxis.y - shortAxis.y,
    },
    {
      x: centerPixels.x + longAxis.x + shortAxis.x,
      y: centerPixels.y + longAxis.y + shortAxis.y,
    },
    {
      x: centerPixels.x - longAxis.x + shortAxis.x,
      y: centerPixels.y - longAxis.y + shortAxis.y,
    },
  ];
  const points = pixelCorners.map((point) => ({
    x: point.x / WIDTH,
    y: point.y / HEIGHT,
  }));
  const classIds = { index: 0, middle: 1, pinky: 2, ring: 3 };

  return {
    id: `live-${className}`,
    classId: classIds[className],
    className,
    confidence,
    center: { x: centerX, y: centerY },
    size: {
      width:
        Math.max(...points.map((point) => point.x)) -
        Math.min(...points.map((point) => point.x)),
      height:
        Math.max(...points.map((point) => point.y)) -
        Math.min(...points.map((point) => point.y)),
    },
    angle,
    points,
  };
}

function createDetections(overrides = {}) {
  const centers = [0.17, 0.39, 0.61, 0.83];
  return FINGERS.map((finger, index) =>
    createDetection(finger, centers[index], overrides[finger] ?? {})
  );
}

function createDetailedFrame() {
  const pixels = new Uint8Array(WIDTH * HEIGHT * 4);
  for (let y = 0; y < HEIGHT; y += 1) {
    for (let x = 0; x < WIDTH; x += 1) {
      const value = 90 + ((x + y) % 8) * 10;
      const index = (y * WIDTH + x) * 4;
      pixels[index] = value;
      pixels[index + 1] = value;
      pixels[index + 2] = value;
      pixels[index + 3] = 255;
    }
  }
  return pixels;
}

function createFlatFrame(value) {
  const pixels = new Uint8Array(WIDTH * HEIGHT * 4);
  for (let index = 0; index < pixels.length; index += 4) {
    pixels[index] = value;
    pixels[index + 1] = value;
    pixels[index + 2] = value;
    pixels[index + 3] = 255;
  }
  return pixels;
}

function measure(buffer, detections) {
  return analyzeLiveRoiPixels({
    buffer: buffer.buffer,
    width: WIDTH,
    height: HEIGHT,
    detections,
  });
}

function updateGate(previousState, detections, pixels) {
  return updateLiveCaptureGate({
    previousState,
    detections,
    pixelMetrics: measure(pixels, detections),
    frameSize: FRAME_SIZE,
  });
}

function getFinger(result, fingerPosition) {
  return result.fingerResults.find(
    (finger) => finger.fingerPosition === fingerPosition
  );
}

const detailedFrame = createDetailedFrame();
const stableDetections = createDetections();
const detailedMetrics = measure(detailedFrame, stableDetections);
assert.equal(Object.keys(detailedMetrics).length, 4);
assert.ok(detailedMetrics.index.contrastDeviation > 14);
assert.ok(detailedMetrics.index.edgeEnergy > 4.5);

let gateState = createInitialLiveCaptureGateState();
let gateResult;
for (let updateIndex = 0; updateIndex < 4; updateIndex += 1) {
  const update = updateGate(gateState, stableDetections, detailedFrame);
  gateState = update.state;
  gateResult = update.result;
}
assert.equal(gateResult.ready, true, 'Dört kararlı ROI çekim kapısını açmalı.');
assert.equal(gateResult.readyFingerCount, 4);
assert.ok(
  Math.abs(
    getFinger(gateResult, 'index').geometryMetrics.shortEdgeRatio - 0.12
  ) < 1e-8
);

const movedDetections = createDetections({ index: { centerY: 0.58 } });
const movementUpdate = updateGate(gateState, movedDetections, detailedFrame);
assert.equal(movementUpdate.result.ready, false);
assert.equal(getFinger(movementUpdate.result, 'index').reason, 'moving');

const missingDetections = stableDetections.filter(
  (detection) => detection.className !== 'pinky'
);
const missingUpdate = updateGate(gateState, missingDetections, detailedFrame);
assert.equal(missingUpdate.result.ready, false);
assert.equal(getFinger(missingUpdate.result, 'pinky').reason, 'missing');

const flatMetrics = measure(createFlatFrame(130), stableDetections);
const flatUpdate = updateLiveCaptureGate({
  previousState: createInitialLiveCaptureGateState(),
  detections: stableDetections,
  pixelMetrics: flatMetrics,
  frameSize: FRAME_SIZE,
});
assert.equal(getFinger(flatUpdate.result, 'middle').reason, 'low-contrast');

const edgeDetections = createDetections({ index: { centerY: 0.02 } });
const edgeUpdate = updateGate(
  createInitialLiveCaptureGateState(),
  edgeDetections,
  detailedFrame
);
assert.equal(getFinger(edgeUpdate.result, 'index').reason, 'outside-frame');

const farDetections = createDetections({
  index: { shortEdgeRatio: 0.09 },
  middle: { shortEdgeRatio: 0.09 },
});
const farUpdate = updateGate(
  createInitialLiveCaptureGateState(),
  farDetections,
  detailedFrame
);
assert.equal(getFinger(farUpdate.result, 'index').reason, 'too-small');
assert.equal(getFinger(farUpdate.result, 'middle').reason, 'too-small');

const closeDetections = createDetections({
  index: { shortEdgeRatio: 0.2 },
});
const closeUpdate = updateGate(
  createInitialLiveCaptureGateState(),
  closeDetections,
  detailedFrame
);
assert.equal(getFinger(closeUpdate.result, 'index').reason, 'too-large');

const angledDetections = createDetections({
  index: { angle: (66 * Math.PI) / 180 },
  middle: { angle: (66 * Math.PI) / 180 },
  ring: { angle: (66 * Math.PI) / 180 },
  pinky: { angle: (66 * Math.PI) / 180 },
});
const angledUpdate = updateGate(
  createInitialLiveCaptureGateState(),
  angledDetections,
  detailedFrame
);
assert.equal(getFinger(angledUpdate.result, 'index').reason, 'hand-angle');

const inconsistentAngleDetections = createDetections({
  pinky: { angle: (115 * Math.PI) / 180 },
});
const inconsistentAngleUpdate = updateGate(
  createInitialLiveCaptureGateState(),
  inconsistentAngleDetections,
  detailedFrame
);
assert.equal(
  getFinger(inconsistentAngleUpdate.result, 'index').reason,
  'angle-inconsistent'
);

const inconsistentSizeDetections = createDetections({
  pinky: { shortEdgeRatio: 0.187, longEdgeRatio: 0.23 },
});
const inconsistentSizeUpdate = updateGate(
  createInitialLiveCaptureGateState(),
  inconsistentSizeDetections,
  detailedFrame
);
assert.equal(
  getFinger(inconsistentSizeUpdate.result, 'index').reason,
  'size-inconsistent'
);

let driftState = createInitialLiveCaptureGateState();
let driftResult;
for (let updateIndex = 0; updateIndex < 4; updateIndex += 1) {
  const centerY = 0.52 + updateIndex * 0.012;
  const driftDetections = createDetections({
    index: { centerY },
    middle: { centerY },
    ring: { centerY },
    pinky: { centerY },
  });
  const driftUpdate = updateGate(driftState, driftDetections, detailedFrame);
  driftState = driftUpdate.state;
  driftResult = driftUpdate.result;
}
assert.equal(driftResult.ready, false, 'Yavaş sürüklenme hazır sayılmamalı.');
assert.equal(getFinger(driftResult, 'index').reason, 'moving');

console.info(
  `[Canlı çekim kapısı testi] hazır=${gateResult.readyFingerCount}/4, kısa=${(getFinger(gateResult, 'index').geometryMetrics.shortEdgeRatio * 100).toFixed(1)}%, parlaklık=${detailedMetrics.index.brightnessMean.toFixed(1)}, kontrast=${detailedMetrics.index.contrastDeviation.toFixed(1)}, detay=${detailedMetrics.index.edgeEnergy.toFixed(1)}`
);
