import type {
  DetectedObbBox,
  DetectionClassName,
} from '@/types/biometrics';

type Point = DetectedObbBox['points'][number];

export type LiveDetectionTrack = {
  detection: DetectedObbBox;
  missedUpdates: number;
  stableUpdates: number;
  confirmed: boolean;
};

const TRACK_ACQUIRE_CONFIDENCE = 0.3;
const PINKY_TRACK_ACQUIRE_CONFIDENCE = 0.24;
const TRACK_CONTINUE_CONFIDENCE = 0.2;
const TRACK_CONFIRM_UPDATE_COUNT = 2;
const TRACK_MAX_MISSED_UPDATES = 2;
const HIGH_CONFIDENCE_OUTLIER_OVERRIDE = 0.78;

const MIN_CENTER_CURRENT_WEIGHT = 0.3;
const MAX_CENTER_CURRENT_WEIGHT = 0.65;
const MIN_SHAPE_CURRENT_WEIGHT = 0.18;
const MAX_SHAPE_CURRENT_WEIGHT = 0.42;
const MIN_ANGLE_CURRENT_WEIGHT = 0.2;
const MAX_ANGLE_CURRENT_WEIGHT = 0.48;
const FINGER_DETECTION_CLASSES: DetectionClassName[] = [
  'index',
  'middle',
  'ring',
  'pinky',
];

export type LiveDetectionFallbackMergeResult = {
  detections: DetectedObbBox[];
  fallbackClasses: DetectionClassName[];
};

export type LiveDetectionCaptureMergeResult = LiveDetectionFallbackMergeResult & {
  snapshotClasses: DetectionClassName[];
};

// Fotoğraf modeli bir sınıfı kaçırırsa, yalnızca o sınıfı son kararlı canlı OBB ile tamamlar.
// Noktalar normalize edilmiş ve yönü düzeltilmiş görüntü koordinatında tutulduğu için
// fotoğraf ROI hattı aynı kutuyu doğrudan kullanabilir; fotoğraf tarafından bulunan kutular ezilmez.
export function mergeMissingLiveDetections(
  photoDetections: DetectedObbBox[],
  liveFallbackDetections: DetectedObbBox[]
): LiveDetectionFallbackMergeResult {
  const photoClasses = new Set(
    photoDetections
      .map((detection) => detection.className)
      .filter((className): className is DetectionClassName =>
        FINGER_DETECTION_CLASSES.includes(className)
      )
  );
  const bestFallbackByClass = new Map<DetectionClassName, DetectedObbBox>();

  for (const detection of liveFallbackDetections) {
    if (!FINGER_DETECTION_CLASSES.includes(detection.className)) continue;
    const current = bestFallbackByClass.get(detection.className);
    if (!current || detection.confidence > current.confidence) {
      bestFallbackByClass.set(detection.className, detection);
    }
  }

  const detections = photoDetections.map((detection) => cloneDetection(detection));
  const fallbackClasses: DetectionClassName[] = [];

  for (const fingerClass of FINGER_DETECTION_CLASSES) {
    if (photoClasses.has(fingerClass)) continue;

    const fallback = bestFallbackByClass.get(fingerClass);
    if (!fallback) continue;

    detections.push({
      ...cloneDetection(fallback),
      id: `live-fallback-${fingerClass}`,
    });
    fallbackClasses.push(fingerClass);
  }

  return { detections, fallbackClasses };
}

// Kalite kapısının çekimi tetiklediği karedeki OBB'leri ROI kaynağı olarak dondurur.
// Fotoğraf modeli yine tanı amacıyla çalışır; ancak aynı sınıfı farklı veya dar bir kutuyla
// bulsa bile çekim anında kullanıcıya gösterilen alan değiştirilmez. Snapshot'ta bulunmayan
// bir sınıf olursa fotoğraf modeli güvenli yedek olarak kullanılır.
export function mergeCaptureSnapshotDetections(
  photoDetections: DetectedObbBox[],
  captureSnapshotDetections: DetectedObbBox[]
): LiveDetectionCaptureMergeResult {
  const bestPhotoByClass = getBestFingerDetectionsByClass(photoDetections);
  const bestSnapshotByClass = getBestFingerDetectionsByClass(
    captureSnapshotDetections
  );
  const photoClasses = new Set(bestPhotoByClass.keys());
  const detections: DetectedObbBox[] = [];
  const snapshotClasses: DetectionClassName[] = [];
  const fallbackClasses: DetectionClassName[] = [];

  for (const fingerClass of FINGER_DETECTION_CLASSES) {
    const snapshot = bestSnapshotByClass.get(fingerClass);
    if (snapshot) {
      detections.push({
        ...cloneDetection(snapshot),
        id: `live-capture-${fingerClass}`,
      });
      snapshotClasses.push(fingerClass);
      if (!photoClasses.has(fingerClass)) fallbackClasses.push(fingerClass);
      continue;
    }

    const photoDetection = bestPhotoByClass.get(fingerClass);
    if (photoDetection) detections.push(cloneDetection(photoDetection));
  }

  return { detections, fallbackClasses, snapshotClasses };
}

// Aynı parmak sınıfındaki kutuları önceki geometrisine göre ilişkilendirip kararlı canlı takip üretir.
export function updateLiveDetectionTracks(
  previousTracks: Map<DetectionClassName, LiveDetectionTrack>,
  currentDetections: DetectedObbBox[]
) {
  const candidatesByClass = groupDetectionsByClass(currentDetections);
  const nextTracks = new Map<DetectionClassName, LiveDetectionTrack>();

  for (const [className, candidates] of candidatesByClass) {
    const previousTrack = previousTracks.get(className);

    if (!previousTrack) {
      const strongestCandidate = [...candidates].sort(
        (first, second) => second.confidence - first.confidence
      )[0];

      if (
        strongestCandidate &&
        strongestCandidate.confidence >= getTrackAcquireConfidence(className)
      ) {
        nextTracks.set(className, createNewTrack(strongestCandidate));
      }
      continue;
    }

    const matchedCandidate = selectBestTrackCandidate(
      previousTrack.detection,
      candidates
    );

    if (
      !matchedCandidate ||
      matchedCandidate.confidence < TRACK_CONTINUE_CONFIDENCE ||
      !isPlausibleTrackUpdate(previousTrack.detection, matchedCandidate)
    ) {
      retainMissedTrack(nextTracks, className, previousTrack);
      continue;
    }

    const stableUpdates = previousTrack.stableUpdates + 1;
    nextTracks.set(className, {
      detection: smoothTrackedDetection(
        previousTrack.detection,
        matchedCandidate,
        previousTrack.missedUpdates
      ),
      missedUpdates: 0,
      stableUpdates,
      confirmed:
        previousTrack.confirmed ||
        stableUpdates >= TRACK_CONFIRM_UPDATE_COUNT,
    });
  }

  // O an modelin göremediği doğrulanmış kutuyu kısa süre koruyarak tek karelik kaybolmaları gizler.
  for (const [className, previousTrack] of previousTracks) {
    if (nextTracks.has(className) || candidatesByClass.has(className)) continue;
    retainMissedTrack(nextTracks, className, previousTrack);
  }

  return nextTracks;
}

// Henüz ikinci tutarlı ölçümü almamış adayları canlı arayüze ve odak döngüsüne göndermez.
export function getVisibleLiveDetections(
  tracks: Map<DetectionClassName, LiveDetectionTrack>
) {
  return [...tracks.values()]
    .filter((track) => track.confirmed)
    .map((track) => track.detection)
    .sort((first, second) => second.confidence - first.confidence);
}

// Kalite kapısına yalnızca bu inference turunda model tarafından yeniden görülen doğrulanmış kutuları verir.
export function getCurrentConfirmedLiveDetections(
  tracks: Map<DetectionClassName, LiveDetectionTrack>
) {
  return [...tracks.values()]
    .filter((track) => track.confirmed && track.missedUpdates === 0)
    .map((track) => track.detection)
    .sort((first, second) => second.confidence - first.confidence);
}

// OBB yönü 180 derece periyodiktir; iki eşdeğer yön arasındaki en kısa açısal farkı döndürür.
export function getHalfTurnAngleDifference(from: number, to: number) {
  const halfTurn = Math.PI;
  let difference = (to - from) % halfTurn;

  if (difference > halfTurn / 2) difference -= halfTurn;
  if (difference < -halfTurn / 2) difference += halfTurn;
  return difference;
}

// Aynı fiziksel köşelerin indisleri export gösterimiyle değişse bile en yakın köşe sırasını seçer.
export function alignObbPointsToReference(
  referencePoints: Point[],
  candidatePoints: Point[]
) {
  if (
    referencePoints.length !== candidatePoints.length ||
    candidatePoints.length < 2
  ) {
    return candidatePoints.map((point) => ({ ...point }));
  }

  let bestPoints = candidatePoints.map((point) => ({ ...point }));
  let bestDistance = Number.POSITIVE_INFINITY;

  for (const shouldReverse of [false, true]) {
    const orderedPoints = shouldReverse
      ? [...candidatePoints].reverse()
      : [...candidatePoints];

    for (let shift = 0; shift < orderedPoints.length; shift += 1) {
      const shiftedPoints = referencePoints.map(
        (_, index) => orderedPoints[(index + shift) % orderedPoints.length]
      );
      const distance = getMeanSquaredPointDistance(
        referencePoints,
        shiftedPoints
      );

      if (distance < bestDistance) {
        bestDistance = distance;
        bestPoints = shiftedPoints.map((point) => ({ ...point }));
      }
    }
  }

  return bestPoints;
}

function groupDetectionsByClass(detections: DetectedObbBox[]) {
  const groups = new Map<DetectionClassName, DetectedObbBox[]>();

  for (const detection of detections) {
    const classCandidates = groups.get(detection.className) ?? [];
    classCandidates.push(detection);
    groups.set(detection.className, classCandidates);
  }

  return groups;
}

function getBestFingerDetectionsByClass(detections: DetectedObbBox[]) {
  const bestByClass = new Map<DetectionClassName, DetectedObbBox>();

  for (const detection of detections) {
    if (!FINGER_DETECTION_CLASSES.includes(detection.className)) continue;
    const current = bestByClass.get(detection.className);
    if (!current || detection.confidence > current.confidence) {
      bestByClass.set(detection.className, detection);
    }
  }

  return bestByClass;
}

function getTrackAcquireConfidence(className: DetectionClassName) {
  return className === 'pinky'
    ? PINKY_TRACK_ACQUIRE_CONFIDENCE
    : TRACK_ACQUIRE_CONFIDENCE;
}

function createNewTrack(detection: DetectedObbBox): LiveDetectionTrack {
  return {
    detection: {
      ...detection,
      id: `live-${detection.className}`,
      points: detection.points.map((point) => ({ ...point })),
    },
    missedUpdates: 0,
    stableUpdates: 1,
    confirmed: false,
  };
}

function retainMissedTrack(
  tracks: Map<DetectionClassName, LiveDetectionTrack>,
  className: DetectionClassName,
  previousTrack: LiveDetectionTrack
) {
  const missedUpdates = previousTrack.missedUpdates + 1;
  if (missedUpdates > TRACK_MAX_MISSED_UPDATES) return;

  tracks.set(className, {
    ...previousTrack,
    missedUpdates,
  });
}

function selectBestTrackCandidate(
  previous: DetectedObbBox,
  candidates: DetectedObbBox[]
) {
  let bestCandidate: DetectedObbBox | null = null;
  let bestScore = Number.NEGATIVE_INFINITY;

  for (const candidate of candidates) {
    const alignedCandidate = alignDetectionToPrevious(previous, candidate);
    const centerDistance = getPointDistance(
      getPolygonCenter(previous.points),
      getPolygonCenter(alignedCandidate.points)
    );
    const cornerDistance = Math.sqrt(
      getMeanSquaredPointDistance(previous.points, alignedCandidate.points)
    );
    const areaSimilarity = getAreaSimilarity(
      previous.points,
      alignedCandidate.points
    );
    const angleSimilarity =
      1 -
      Math.min(
        Math.abs(
          getHalfTurnAngleDifference(previous.angle, alignedCandidate.angle)
        ) /
          (Math.PI / 2),
        1
      );
    const continuityScore =
      clamp01(1 - centerDistance / 0.18) * 0.3 +
      clamp01(1 - cornerDistance / 0.2) * 0.3 +
      areaSimilarity * 0.2 +
      angleSimilarity * 0.1 +
      alignedCandidate.confidence * 0.1;

    if (continuityScore > bestScore) {
      bestScore = continuityScore;
      bestCandidate = alignedCandidate;
    }
  }

  return bestCandidate;
}

function alignDetectionToPrevious(
  previous: DetectedObbBox,
  current: DetectedObbBox
): DetectedObbBox {
  return {
    ...current,
    points: alignObbPointsToReference(previous.points, current.points),
  };
}

function isPlausibleTrackUpdate(
  previous: DetectedObbBox,
  current: DetectedObbBox
) {
  const centerDistance = getPointDistance(
    getPolygonCenter(previous.points),
    getPolygonCenter(current.points)
  );
  const cornerDistance = Math.sqrt(
    getMeanSquaredPointDistance(previous.points, current.points)
  );
  const areaRatio = getSymmetricAreaRatio(previous.points, current.points);
  const angleDifference = Math.abs(
    getHalfTurnAngleDifference(previous.angle, current.angle)
  );
  const outlierSignals = [
    centerDistance > 0.14,
    cornerDistance > 0.16,
    areaRatio > 2,
    angleDifference > Math.PI * 0.3,
  ].filter(Boolean).length;
  const isHighConfidence =
    current.confidence >= HIGH_CONFIDENCE_OUTLIER_OVERRIDE;
  const maximumCornerDistance = isHighConfidence ? 0.32 : 0.24;
  const maximumOutlierSignals = isHighConfidence ? 3 : 2;

  return (
    cornerDistance <= maximumCornerDistance &&
    outlierSignals < maximumOutlierSignals
  );
}

function smoothTrackedDetection(
  previous: DetectedObbBox,
  current: DetectedObbBox,
  missedUpdates: number
): DetectedObbBox {
  const previousCenter = getPolygonCenter(previous.points);
  const currentCenter = getPolygonCenter(current.points);
  const centerDistance = getPointDistance(previousCenter, currentCenter);
  const movementFactor = clamp01(centerDistance / 0.08);
  const recoveryBoost = Math.min(missedUpdates * 0.12, 0.24);
  const centerWeight = clamp01(
    interpolate(
      MIN_CENTER_CURRENT_WEIGHT,
      MAX_CENTER_CURRENT_WEIGHT,
      movementFactor
    ) + recoveryBoost
  );
  const shapeWeight = clamp01(
    interpolate(
      MIN_SHAPE_CURRENT_WEIGHT,
      MAX_SHAPE_CURRENT_WEIGHT,
      movementFactor
    ) + recoveryBoost * 0.5
  );
  const angleWeight = clamp01(
    interpolate(
      MIN_ANGLE_CURRENT_WEIGHT,
      MAX_ANGLE_CURRENT_WEIGHT,
      movementFactor
    ) + recoveryBoost * 0.5
  );
  const center = {
    x: interpolate(previousCenter.x, currentCenter.x, centerWeight),
    y: interpolate(previousCenter.y, currentCenter.y, centerWeight),
  };
  const points = previous.points.map((previousPoint, index) => {
    const currentPoint = current.points[index] ?? previousPoint;
    const previousOffset = {
      x: previousPoint.x - previousCenter.x,
      y: previousPoint.y - previousCenter.y,
    };
    const currentOffset = {
      x: currentPoint.x - currentCenter.x,
      y: currentPoint.y - currentCenter.y,
    };

    return {
      x:
        center.x +
        interpolate(previousOffset.x, currentOffset.x, shapeWeight),
      y:
        center.y +
        interpolate(previousOffset.y, currentOffset.y, shapeWeight),
    };
  });
  const axisAlignedSize = getAxisAlignedSize(points);

  return {
    ...current,
    id: `live-${current.className}`,
    confidence: interpolate(previous.confidence, current.confidence, 0.35),
    center,
    size: axisAlignedSize,
    angle:
      previous.angle +
      getHalfTurnAngleDifference(previous.angle, current.angle) * angleWeight,
    points,
  };
}

function getPolygonCenter(points: Point[]) {
  if (points.length === 0) return { x: 0, y: 0 };

  const total = points.reduce(
    (sum, point) => ({ x: sum.x + point.x, y: sum.y + point.y }),
    { x: 0, y: 0 }
  );
  return {
    x: total.x / points.length,
    y: total.y / points.length,
  };
}

function getAxisAlignedSize(points: Point[]) {
  if (points.length === 0) return { width: 0, height: 0 };

  const xValues = points.map((point) => point.x);
  const yValues = points.map((point) => point.y);
  return {
    width: Math.max(...xValues) - Math.min(...xValues),
    height: Math.max(...yValues) - Math.min(...yValues),
  };
}

function getMeanSquaredPointDistance(first: Point[], second: Point[]) {
  const pointCount = Math.min(first.length, second.length);
  if (pointCount === 0) return Number.POSITIVE_INFINITY;

  let total = 0;
  for (let index = 0; index < pointCount; index += 1) {
    const deltaX = (first[index]?.x ?? 0) - (second[index]?.x ?? 0);
    const deltaY = (first[index]?.y ?? 0) - (second[index]?.y ?? 0);
    total += deltaX * deltaX + deltaY * deltaY;
  }
  return total / pointCount;
}

function getPolygonArea(points: Point[]) {
  if (points.length < 3) return 0;

  let twiceArea = 0;
  for (let index = 0; index < points.length; index += 1) {
    const current = points[index];
    const next = points[(index + 1) % points.length];
    twiceArea +=
      (current?.x ?? 0) * (next?.y ?? 0) -
      (next?.x ?? 0) * (current?.y ?? 0);
  }
  return Math.abs(twiceArea) / 2;
}

function getAreaSimilarity(first: Point[], second: Point[]) {
  const firstArea = getPolygonArea(first);
  const secondArea = getPolygonArea(second);
  const largestArea = Math.max(firstArea, secondArea);
  if (largestArea <= 1e-8) return 0;
  return Math.min(firstArea, secondArea) / largestArea;
}

function getSymmetricAreaRatio(first: Point[], second: Point[]) {
  const firstArea = Math.max(getPolygonArea(first), 1e-8);
  const secondArea = Math.max(getPolygonArea(second), 1e-8);
  return Math.max(firstArea / secondArea, secondArea / firstArea);
}

function getPointDistance(first: Point, second: Point) {
  return Math.hypot(first.x - second.x, first.y - second.y);
}

function cloneDetection(detection: DetectedObbBox): DetectedObbBox {
  return {
    ...detection,
    center: { ...detection.center },
    size: { ...detection.size },
    points: detection.points.map((point) => ({ ...point })),
  };
}

function interpolate(from: number, to: number, weight: number) {
  return from * (1 - weight) + to * weight;
}

function clamp01(value: number) {
  return Math.max(0, Math.min(value, 1));
}
