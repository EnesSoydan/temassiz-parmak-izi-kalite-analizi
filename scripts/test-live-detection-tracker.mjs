import assert from 'node:assert/strict';

import {
  getHalfTurnAngleDifference,
  getVisibleLiveDetections,
  updateLiveDetectionTracks,
} from '../src/lib/live-detection-tracker.ts';

function createDetection({
  className = 'index',
  confidence = 0.8,
  centerX = 0.5,
  centerY = 0.5,
  width = 0.18,
  height = 0.08,
  angle = 0,
  pointShift = 0,
}) {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const halfWidth = width / 2;
  const halfHeight = height / 2;
  const localPoints = [
    { x: -halfWidth, y: -halfHeight },
    { x: halfWidth, y: -halfHeight },
    { x: halfWidth, y: halfHeight },
    { x: -halfWidth, y: halfHeight },
  ];
  const points = localPoints.map((point) => ({
    x: centerX + point.x * cos - point.y * sin,
    y: centerY + point.x * sin + point.y * cos,
  }));
  const shiftedPoints = points.map(
    (_, index) => points[(index + pointShift) % points.length]
  );
  const xValues = shiftedPoints.map((point) => point.x);
  const yValues = shiftedPoints.map((point) => point.y);
  const classIds = { index: 0, middle: 1, pinky: 2, ring: 3 };

  return {
    id: `test-${className}`,
    classId: classIds[className] ?? -1,
    className,
    confidence,
    center: { x: centerX, y: centerY },
    size: {
      width: Math.max(...xValues) - Math.min(...xValues),
      height: Math.max(...yValues) - Math.min(...yValues),
    },
    angle,
    points: shiftedPoints,
  };
}

function getPolygonArea(points) {
  let twiceArea = 0;
  for (let index = 0; index < points.length; index += 1) {
    const current = points[index];
    const next = points[(index + 1) % points.length];
    twiceArea += current.x * next.y - next.x * current.y;
  }
  return Math.abs(twiceArea) / 2;
}

function degrees(value) {
  return (value * Math.PI) / 180;
}

// OBB'nin 179° ve 1° gösterimleri arasında 178° yerine kısa 2° fark seçilmelidir.
assert.ok(
  Math.abs(getHalfTurnAngleDifference(degrees(179), degrees(1)) - degrees(2)) <
    1e-8,
  '180° periyodik OBB açısı en kısa yönde çözülmeli.'
);

const initialDetection = createDetection({ angle: degrees(14) });
let tracks = updateLiveDetectionTracks(new Map(), [initialDetection]);
assert.equal(
  getVisibleLiveDetections(tracks).length,
  0,
  'İlk tek ölçüm yanlış pozitif titreşimini önlemek için henüz gösterilmemeli.'
);

// Aynı dörtgen farklı başlangıç köşesi ve 180° eşdeğer açıyla gelirse kutu çökmemelidir.
const equivalentDetection = createDetection({
  angle: degrees(194),
  pointShift: 2,
});
tracks = updateLiveDetectionTracks(tracks, [equivalentDetection]);
const confirmedDetection = getVisibleLiveDetections(tracks)[0];
assert.ok(confirmedDetection, 'İkinci tutarlı ölçüm canlı kutuyu doğrulamalı.');
assert.ok(
  Math.abs(
    getPolygonArea(confirmedDetection.points) -
      getPolygonArea(initialDetection.points)
  ) < 1e-8,
  'Eşdeğer köşe sırası geçişinde OBB alanı korunmalı.'
);

// Küçük gerçek hareket izlenirken merkez doğrudan gürültülü yeni konuma sıçramamalıdır.
const movedDetection = createDetection({
  centerX: 0.52,
  centerY: 0.495,
  width: 0.19,
  height: 0.082,
  angle: degrees(16),
});
tracks = updateLiveDetectionTracks(tracks, [movedDetection]);
const smoothedDetection = getVisibleLiveDetections(tracks)[0];
assert.ok(
  smoothedDetection.center.x > confirmedDetection.center.x &&
    smoothedDetection.center.x < movedDetection.center.x,
  'Küçük merkez hareketi gecikmesiz ama yumuşatılmış biçimde izlenmeli.'
);
assert.ok(
  smoothedDetection.size.width < movedDetection.size.width,
  'Boyut değişimi yeni ölçüme tek karede tamamen atlamamalı.'
);

// Aynı sınıfta aniden uzakta ve çok büyük gelen orta güvenli aday mevcut track'i bozmamalıdır.
const outlierDetection = createDetection({
  confidence: 0.55,
  centerX: 0.82,
  centerY: 0.2,
  width: 0.48,
  height: 0.3,
  angle: degrees(82),
});
const centerBeforeOutlier = smoothedDetection.center;
tracks = updateLiveDetectionTracks(tracks, [outlierDetection]);
const retainedTrack = tracks.get('index');
assert.ok(retainedTrack, 'Ani aykırı ölçümde önceki track kısa süre korunmalı.');
assert.equal(retainedTrack.missedUpdates, 1);
assert.deepEqual(
  retainedTrack.detection.center,
  centerBeforeOutlier,
  'Aykırı kutu doğrulanmış canlı geometriyi değiştirmemeli.'
);

tracks = updateLiveDetectionTracks(tracks, [
  { ...outlierDetection, confidence: 0.92 },
]);
const retainedAgainstConfidentOutlier = tracks.get('index');
assert.ok(
  retainedAgainstConfidentOutlier,
  'Çok güvenli fakat birden fazla geometrik sınırı aşan aykırı kutu da doğrudan kabul edilmemeli.'
);
assert.deepEqual(
  retainedAgainstConfidentOutlier.detection.center,
  centerBeforeOutlier,
  'Yüksek güven skoru tek başına fiziksel olarak tutarsız sıçramayı geçirmemeli.'
);

// Düşük güvenli index yeni track başlatamazken küçük serçe için kontrollü tolerans korunur.
const weakIndexTracks = updateLiveDetectionTracks(new Map(), [
  createDetection({ confidence: 0.25 }),
]);
assert.equal(weakIndexTracks.size, 0, 'Düşük güvenli index track başlatmamalı.');

const pinkyTracks = updateLiveDetectionTracks(new Map(), [
  createDetection({ className: 'pinky', confidence: 0.25 }),
]);
assert.equal(pinkyTracks.size, 1, 'Serçe için düşük başlangıç eşiği korunmalı.');
assert.equal(
  getVisibleLiveDetections(pinkyTracks).length,
  0,
  'Düşük güvenli serçe de ikinci tutarlı ölçümden önce çizilmemeli.'
);

console.info(
  `[Canlı takip testi] açı_farkı=${(
    (getHalfTurnAngleDifference(degrees(179), degrees(1)) * 180) /
    Math.PI
  ).toFixed(1)}°, merkez=${smoothedDetection.center.x.toFixed(3)}, alan=${getPolygonArea(
    smoothedDetection.points
  ).toFixed(4)}, aykırı_koruma=${retainedAgainstConfidentOutlier.missedUpdates}`
);
