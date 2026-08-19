import type {
  DetectedObbBox,
  FingerprintPosition,
} from '@/types/biometrics';

type Point = DetectedObbBox['points'][number];

const FINGERPRINT_POSITIONS: FingerprintPosition[] = [
  'index',
  'middle',
  'ring',
  'pinky',
];

export type LiveCaptureGateReason =
  | 'ready'
  | 'missing'
  | 'low-confidence'
  | 'outside-frame'
  | 'too-small'
  | 'too-large'
  | 'invalid-shape'
  | 'hand-angle'
  | 'angle-inconsistent'
  | 'size-inconsistent'
  | 'pixel-data-missing'
  | 'too-dark'
  | 'too-bright'
  | 'glare'
  | 'low-contrast'
  | 'low-detail'
  | 'stabilizing'
  | 'moving';

export type LiveRoiPixelMetrics = {
  brightnessMean: number;
  contrastDeviation: number;
  highlightRatio: number;
  edgeEnergy: number;
  sampledPixelCount: number;
};

export type LiveRoiGeometryMetrics = {
  areaRatio: number;
  shortEdgeRatio: number;
  longEdgeRatio: number;
  aspectRatio: number;
  verticalDeviationDegrees: number;
};

export type LiveCaptureFingerResult = {
  fingerPosition: FingerprintPosition;
  ready: boolean;
  reason: LiveCaptureGateReason;
  confidence: number;
  stableUpdateCount: number;
  pixelMetrics?: LiveRoiPixelMetrics;
  geometryMetrics?: LiveRoiGeometryMetrics;
};

export type LiveCaptureGateResult = {
  calibrationVersion: string;
  ready: boolean;
  readyFingerCount: number;
  fingerResults: LiveCaptureFingerResult[];
};

type DetectionSnapshot = {
  center: Point;
  area: number;
  angle: number;
  geometry: LiveRoiGeometryMetrics;
};

export type LiveCaptureGateState = {
  previousDetections: Partial<Record<FingerprintPosition, DetectionSnapshot>>;
  stabilityAnchors: Partial<Record<FingerprintPosition, DetectionSnapshot>>;
  stableUpdateCounts: Partial<Record<FingerprintPosition, number>>;
};

export const LIVE_CAPTURE_GATE_CONFIG = {
  // Normal/yakın/uzak/açılı gerçek probe oturumlarının 2026-08-07 dağılımıyla başlatılan saha kalibrasyonu.
  calibrationVersion: 'probe-20260807-v1',
  minimumConfidence: 0.32,
  pinkyMinimumConfidence: 0.24,
  minimumShortEdgeRatio: {
    index: 0.105,
    middle: 0.105,
    ring: 0.1,
    pinky: 0.095,
  },
  maximumShortEdgeRatio: 0.19,
  minimumAspectRatio: 1.08,
  maximumAspectRatio: 3.2,
  maximumMedianVerticalDeviationRadians: (15 * Math.PI) / 180,
  maximumFingerAngleSpreadRadians: (16 * Math.PI) / 180,
  maximumShortEdgeSpreadRatio: 1.55,
  frameMarginRatio: 0.012,
  pixelAnalysisInsetRatio: 0.82,
  minimumBrightnessMean: 55,
  maximumBrightnessMean: 225,
  maximumHighlightRatio: 0.22,
  minimumContrastDeviation: 14,
  minimumEdgeEnergy: 4.5,
  maximumCenterShift: 0.02,
  maximumAreaChangeRatio: 1.2,
  maximumAngleChangeRadians: (10 * Math.PI) / 180,
  maximumAnchorCenterShift: 0.03,
  maximumAnchorAreaChangeRatio: 1.28,
  maximumAnchorAngleChangeRadians: (14 * Math.PI) / 180,
  requiredStableUpdates: 4,
} as const;

export function createInitialLiveCaptureGateState(): LiveCaptureGateState {
  return {
    previousDetections: {},
    stabilityAnchors: {},
    stableUpdateCounts: {},
  };
}

// Canlı RGBA frame'de yalnızca OBB içindeki pikselleri örnekleyerek hafif pozlama ve detay ölçümü yapar.
export function analyzeLiveRoiPixels({
  buffer,
  width,
  height,
  detections,
}: {
  buffer: ArrayBuffer;
  width: number;
  height: number;
  detections: DetectedObbBox[];
}) {
  const metrics: Partial<Record<FingerprintPosition, LiveRoiPixelMetrics>> = {};
  const pixels = new Uint8Array(buffer);
  if (width <= 0 || height <= 0 || pixels.length < width * height * 4) {
    return metrics;
  }

  const bestDetections = selectBestFingerDetections(detections);
  for (const fingerPosition of FINGERPRINT_POSITIONS) {
    const detection = bestDetections.get(fingerPosition);
    if (!detection) continue;

    const result = analyzePolygonPixels({ pixels, width, height, detection });
    if (result) metrics[fingerPosition] = result;
  }

  return metrics;
}

// Her parmağın görüntü ve geometri koşullarını ayrı izler; yalnızca tümü ardışık olarak kararlıysa kapıyı açar.
export function updateLiveCaptureGate({
  previousState,
  detections,
  pixelMetrics,
  frameSize,
}: {
  previousState: LiveCaptureGateState;
  detections: DetectedObbBox[];
  pixelMetrics: Partial<Record<FingerprintPosition, LiveRoiPixelMetrics>>;
  frameSize: { width: number; height: number };
}): { state: LiveCaptureGateState; result: LiveCaptureGateResult } {
  const bestDetections = selectBestFingerDetections(detections);
  const nextState = createInitialLiveCaptureGateState();
  const snapshots: Partial<Record<FingerprintPosition, DetectionSnapshot>> = {};
  const frameReasons: Partial<Record<FingerprintPosition, LiveCaptureGateReason>> = {};
  const fingerResults: LiveCaptureFingerResult[] = [];

  for (const fingerPosition of FINGERPRINT_POSITIONS) {
    const detection = bestDetections.get(fingerPosition);
    const metrics = pixelMetrics[fingerPosition];

    if (!detection) {
      frameReasons[fingerPosition] = 'missing';
      continue;
    }

    const snapshot = createDetectionSnapshot(detection, frameSize);
    snapshots[fingerPosition] = snapshot;
    nextState.previousDetections[fingerPosition] = snapshot;
    frameReasons[fingerPosition] = getStaticFrameRejectionReason({
      fingerPosition,
      detection,
      metrics,
      snapshot,
    });
  }

  if (
    FINGERPRINT_POSITIONS.every(
      (fingerPosition) => frameReasons[fingerPosition] === 'ready'
    )
  ) {
    const handReason = getHandGeometryRejectionReason(snapshots);
    if (handReason) {
      for (const fingerPosition of FINGERPRINT_POSITIONS) {
        frameReasons[fingerPosition] = handReason;
      }
    }
  }

  for (const fingerPosition of FINGERPRINT_POSITIONS) {
    const detection = bestDetections.get(fingerPosition);
    const metrics = pixelMetrics[fingerPosition];
    const snapshot = snapshots[fingerPosition];
    const staticReason = frameReasons[fingerPosition] ?? 'missing';

    if (!detection || !snapshot) {
      fingerResults.push({
        fingerPosition,
        ready: false,
        reason: 'missing',
        confidence: 0,
        stableUpdateCount: 0,
      });
      continue;
    }

    let currentReason = staticReason;
    let stableUpdateCount = 0;

    if (staticReason === 'ready') {
      const previousSnapshot = previousState.previousDetections[fingerPosition];
      const previousAnchor = previousState.stabilityAnchors[fingerPosition];

      if (!previousSnapshot || !previousAnchor) {
        currentReason = 'stabilizing';
        stableUpdateCount = 1;
        nextState.stabilityAnchors[fingerPosition] = snapshot;
      } else if (
        !isStableDetection(previousSnapshot, snapshot) ||
        !isStableAgainstAnchor(previousAnchor, snapshot)
      ) {
        currentReason = 'moving';
        stableUpdateCount = 1;
        nextState.stabilityAnchors[fingerPosition] = snapshot;
      } else {
        stableUpdateCount =
          (previousState.stableUpdateCounts[fingerPosition] ?? 1) + 1;
        nextState.stabilityAnchors[fingerPosition] = previousAnchor;
        currentReason =
          stableUpdateCount >= LIVE_CAPTURE_GATE_CONFIG.requiredStableUpdates
            ? 'ready'
            : 'stabilizing';
      }
    }

    nextState.stableUpdateCounts[fingerPosition] = stableUpdateCount;
    const ready = currentReason === 'ready';

    fingerResults.push({
      fingerPosition,
      ready,
      reason: ready ? 'ready' : currentReason,
      confidence: detection.confidence,
      stableUpdateCount,
      pixelMetrics: metrics,
      geometryMetrics: snapshot.geometry,
    });
  }

  const readyFingerCount = fingerResults.filter((result) => result.ready).length;
  return {
    state: nextState,
    result: {
      calibrationVersion: LIVE_CAPTURE_GATE_CONFIG.calibrationVersion,
      ready: readyFingerCount === FINGERPRINT_POSITIONS.length,
      readyFingerCount,
      fingerResults,
    },
  };
}

function getStaticFrameRejectionReason({
  fingerPosition,
  detection,
  metrics,
  snapshot,
}: {
  fingerPosition: FingerprintPosition;
  detection: DetectedObbBox;
  metrics?: LiveRoiPixelMetrics;
  snapshot: DetectionSnapshot;
}): LiveCaptureGateReason {
  const minimumConfidence =
    fingerPosition === 'pinky'
      ? LIVE_CAPTURE_GATE_CONFIG.pinkyMinimumConfidence
      : LIVE_CAPTURE_GATE_CONFIG.minimumConfidence;
  if (detection.confidence < minimumConfidence) return 'low-confidence';
  if (!isPolygonInsideFrame(detection.points)) return 'outside-frame';
  if (
    snapshot.geometry.shortEdgeRatio <
    LIVE_CAPTURE_GATE_CONFIG.minimumShortEdgeRatio[fingerPosition]
  ) {
    return 'too-small';
  }
  if (
    snapshot.geometry.shortEdgeRatio >
    LIVE_CAPTURE_GATE_CONFIG.maximumShortEdgeRatio
  ) {
    return 'too-large';
  }
  if (
    snapshot.geometry.aspectRatio < LIVE_CAPTURE_GATE_CONFIG.minimumAspectRatio ||
    snapshot.geometry.aspectRatio > LIVE_CAPTURE_GATE_CONFIG.maximumAspectRatio
  ) {
    return 'invalid-shape';
  }
  if (!metrics) return 'pixel-data-missing';
  if (metrics.brightnessMean < LIVE_CAPTURE_GATE_CONFIG.minimumBrightnessMean) {
    return 'too-dark';
  }
  if (metrics.brightnessMean > LIVE_CAPTURE_GATE_CONFIG.maximumBrightnessMean) {
    return 'too-bright';
  }
  if (metrics.highlightRatio > LIVE_CAPTURE_GATE_CONFIG.maximumHighlightRatio) {
    return 'glare';
  }
  if (
    metrics.contrastDeviation <
    LIVE_CAPTURE_GATE_CONFIG.minimumContrastDeviation
  ) {
    return 'low-contrast';
  }
  if (metrics.edgeEnergy < LIVE_CAPTURE_GATE_CONFIG.minimumEdgeEnergy) {
    return 'low-detail';
  }
  return 'ready';
}

function analyzePolygonPixels({
  pixels,
  width,
  height,
  detection,
}: {
  pixels: Uint8Array;
  width: number;
  height: number;
  detection: DetectedObbBox;
}): LiveRoiPixelMetrics | undefined {
  if (detection.points.length < 3) return undefined;

  const polygon = insetPolygon(
    detection.points.map((point) => ({
      x: point.x * width,
      y: point.y * height,
    })),
    LIVE_CAPTURE_GATE_CONFIG.pixelAnalysisInsetRatio
  );
  const bounds = getPolygonBounds(polygon, width, height);
  const sampleStep = Math.max(1, Math.round(Math.max(width, height) / 240));
  let sampledPixelCount = 0;
  let luminanceSum = 0;
  let luminanceSquaredSum = 0;
  let highlightedPixelCount = 0;
  let edgeSum = 0;
  let edgeSampleCount = 0;

  for (let y = bounds.top; y <= bounds.bottom; y += sampleStep) {
    for (let x = bounds.left; x <= bounds.right; x += sampleStep) {
      const point = { x: x + 0.5, y: y + 0.5 };
      if (!isPointInsidePolygon(point, polygon)) continue;

      const luminance = getPixelLuminance(pixels, width, x, y);
      sampledPixelCount += 1;
      luminanceSum += luminance;
      luminanceSquaredSum += luminance * luminance;
      if (luminance >= 245) highlightedPixelCount += 1;

      const neighborX = x + sampleStep;
      if (
        neighborX <= bounds.right &&
        isPointInsidePolygon({ x: neighborX + 0.5, y: y + 0.5 }, polygon)
      ) {
        edgeSum += Math.abs(
          luminance - getPixelLuminance(pixels, width, neighborX, y)
        );
        edgeSampleCount += 1;
      }

      const neighborY = y + sampleStep;
      if (
        neighborY <= bounds.bottom &&
        isPointInsidePolygon({ x: x + 0.5, y: neighborY + 0.5 }, polygon)
      ) {
        edgeSum += Math.abs(
          luminance - getPixelLuminance(pixels, width, x, neighborY)
        );
        edgeSampleCount += 1;
      }
    }
  }

  if (sampledPixelCount < 16) return undefined;
  const brightnessMean = luminanceSum / sampledPixelCount;
  const variance = Math.max(
    luminanceSquaredSum / sampledPixelCount - brightnessMean * brightnessMean,
    0
  );

  return {
    brightnessMean,
    contrastDeviation: Math.sqrt(variance),
    highlightRatio: highlightedPixelCount / sampledPixelCount,
    edgeEnergy: edgeSampleCount > 0 ? edgeSum / edgeSampleCount : 0,
    sampledPixelCount,
  };
}

function selectBestFingerDetections(detections: DetectedObbBox[]) {
  const bestDetections = new Map<FingerprintPosition, DetectedObbBox>();
  for (const detection of detections) {
    if (detection.className === 'unknown') continue;
    const current = bestDetections.get(detection.className);
    if (!current || detection.confidence > current.confidence) {
      bestDetections.set(detection.className, detection);
    }
  }
  return bestDetections;
}

function createDetectionSnapshot(
  detection: DetectedObbBox,
  frameSize: { width: number; height: number }
): DetectionSnapshot {
  const pixelPoints = detection.points.map((point) => ({
    x: point.x * frameSize.width,
    y: point.y * frameSize.height,
  }));
  const edgeLengths = pixelPoints.map((point, index) => {
    const next = pixelPoints[(index + 1) % pixelPoints.length] ?? point;
    return Math.hypot(next.x - point.x, next.y - point.y);
  });
  const shortEdge = Math.min(...edgeLengths);
  const longEdge = Math.max(...edgeLengths);
  const frameShortEdge = Math.max(Math.min(frameSize.width, frameSize.height), 1);
  const verticalDeviationRadians = Math.abs(
    getHalfTurnAngleDifference(Math.PI / 2, detection.angle)
  );

  return {
    center: getPolygonCenter(detection.points),
    area: getPolygonArea(detection.points),
    angle: detection.angle,
    geometry: {
      areaRatio: getPolygonArea(detection.points),
      shortEdgeRatio: Number.isFinite(shortEdge) ? shortEdge / frameShortEdge : 0,
      longEdgeRatio: Number.isFinite(longEdge) ? longEdge / frameShortEdge : 0,
      aspectRatio: shortEdge > 1e-8 ? longEdge / shortEdge : Number.POSITIVE_INFINITY,
      verticalDeviationDegrees: (verticalDeviationRadians * 180) / Math.PI,
    },
  };
}

function getHandGeometryRejectionReason(
  snapshots: Partial<Record<FingerprintPosition, DetectionSnapshot>>
): LiveCaptureGateReason | null {
  const completeSnapshots = FINGERPRINT_POSITIONS.map(
    (fingerPosition) => snapshots[fingerPosition]
  ).filter((snapshot): snapshot is DetectionSnapshot => Boolean(snapshot));
  if (completeSnapshots.length !== FINGERPRINT_POSITIONS.length) return null;

  const referenceAngle = getMedianAxisAngle(
    completeSnapshots.map((snapshot) => snapshot.angle)
  );
  const medianVerticalDeviation = Math.abs(
    getHalfTurnAngleDifference(Math.PI / 2, referenceAngle)
  );
  if (
    medianVerticalDeviation >
    LIVE_CAPTURE_GATE_CONFIG.maximumMedianVerticalDeviationRadians
  ) {
    return 'hand-angle';
  }

  const maximumAngleSpread = Math.max(
    ...completeSnapshots.map((snapshot) =>
      Math.abs(getHalfTurnAngleDifference(referenceAngle, snapshot.angle))
    )
  );
  if (
    maximumAngleSpread > LIVE_CAPTURE_GATE_CONFIG.maximumFingerAngleSpreadRadians
  ) {
    return 'angle-inconsistent';
  }

  const shortEdges = completeSnapshots.map(
    (snapshot) => snapshot.geometry.shortEdgeRatio
  );
  const smallestShortEdge = Math.max(Math.min(...shortEdges), 1e-8);
  if (
    Math.max(...shortEdges) / smallestShortEdge >
    LIVE_CAPTURE_GATE_CONFIG.maximumShortEdgeSpreadRatio
  ) {
    return 'size-inconsistent';
  }

  return null;
}

function isStableDetection(first: DetectionSnapshot, second: DetectionSnapshot) {
  const centerShift = Math.hypot(
    first.center.x - second.center.x,
    first.center.y - second.center.y
  );
  const smallerArea = Math.max(Math.min(first.area, second.area), 1e-8);
  const areaChangeRatio = Math.max(first.area, second.area) / smallerArea;
  const angleChange = Math.abs(
    getHalfTurnAngleDifference(first.angle, second.angle)
  );

  return (
    centerShift <= LIVE_CAPTURE_GATE_CONFIG.maximumCenterShift &&
    areaChangeRatio <= LIVE_CAPTURE_GATE_CONFIG.maximumAreaChangeRatio &&
    angleChange <= LIVE_CAPTURE_GATE_CONFIG.maximumAngleChangeRadians
  );
}

function isStableAgainstAnchor(
  anchor: DetectionSnapshot,
  current: DetectionSnapshot
) {
  const centerShift = Math.hypot(
    anchor.center.x - current.center.x,
    anchor.center.y - current.center.y
  );
  const smallerArea = Math.max(Math.min(anchor.area, current.area), 1e-8);
  const areaChangeRatio = Math.max(anchor.area, current.area) / smallerArea;
  const angleChange = Math.abs(
    getHalfTurnAngleDifference(anchor.angle, current.angle)
  );

  return (
    centerShift <= LIVE_CAPTURE_GATE_CONFIG.maximumAnchorCenterShift &&
    areaChangeRatio <= LIVE_CAPTURE_GATE_CONFIG.maximumAnchorAreaChangeRatio &&
    angleChange <= LIVE_CAPTURE_GATE_CONFIG.maximumAnchorAngleChangeRadians
  );
}

function getHalfTurnAngleDifference(from: number, to: number) {
  const halfTurn = Math.PI;
  let difference = (to - from) % halfTurn;
  if (difference > halfTurn / 2) difference -= halfTurn;
  if (difference < -halfTurn / 2) difference += halfTurn;
  return difference;
}

function getMedianAxisAngle(angles: number[]) {
  const doubledX = angles.reduce((sum, angle) => sum + Math.cos(angle * 2), 0);
  const doubledY = angles.reduce((sum, angle) => sum + Math.sin(angle * 2), 0);
  return Math.atan2(doubledY, doubledX) / 2;
}

function isPolygonInsideFrame(points: Point[]) {
  const margin = LIVE_CAPTURE_GATE_CONFIG.frameMarginRatio;
  return (
    points.length >= 3 &&
    points.every(
      (point) =>
        point.x >= margin &&
        point.x <= 1 - margin &&
        point.y >= margin &&
        point.y <= 1 - margin
    )
  );
}

function getPolygonCenter(points: Point[]) {
  const total = points.reduce(
    (sum, point) => ({ x: sum.x + point.x, y: sum.y + point.y }),
    { x: 0, y: 0 }
  );
  const divisor = Math.max(points.length, 1);
  return { x: total.x / divisor, y: total.y / divisor };
}

function insetPolygon(points: Point[], ratio: number) {
  const center = getPolygonCenter(points);
  return points.map((point) => ({
    x: center.x + (point.x - center.x) * ratio,
    y: center.y + (point.y - center.y) * ratio,
  }));
}

function getPolygonArea(points: Point[]) {
  if (points.length < 3) return 0;
  let twiceArea = 0;
  for (let index = 0; index < points.length; index += 1) {
    const current = points[index];
    const next = points[(index + 1) % points.length];
    twiceArea += current.x * next.y - next.x * current.y;
  }
  return Math.abs(twiceArea) / 2;
}

function getPolygonBounds(points: Point[], width: number, height: number) {
  const xValues = points.map((point) => point.x);
  const yValues = points.map((point) => point.y);
  return {
    left: clampInteger(Math.floor(Math.min(...xValues)), 0, width - 1),
    top: clampInteger(Math.floor(Math.min(...yValues)), 0, height - 1),
    right: clampInteger(Math.ceil(Math.max(...xValues)), 0, width - 1),
    bottom: clampInteger(Math.ceil(Math.max(...yValues)), 0, height - 1),
  };
}

function isPointInsidePolygon(point: Point, polygon: Point[]) {
  let inside = false;
  for (
    let currentIndex = 0, previousIndex = polygon.length - 1;
    currentIndex < polygon.length;
    previousIndex = currentIndex, currentIndex += 1
  ) {
    const current = polygon[currentIndex];
    const previous = polygon[previousIndex];
    const intersects =
      current.y > point.y !== previous.y > point.y &&
      point.x <
        ((previous.x - current.x) * (point.y - current.y)) /
          (previous.y - current.y || Number.EPSILON) +
          current.x;
    if (intersects) inside = !inside;
  }
  return inside;
}

function getPixelLuminance(
  pixels: Uint8Array,
  width: number,
  x: number,
  y: number
) {
  const index = (y * width + x) * 4;
  const red = pixels[index] ?? 0;
  const green = pixels[index + 1] ?? 0;
  const blue = pixels[index + 2] ?? 0;
  return red * 0.2126 + green * 0.7152 + blue * 0.0722;
}

function clampInteger(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(value, maximum));
}
