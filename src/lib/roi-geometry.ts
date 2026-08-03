export type NormalizedPoint = {
  x: number;
  y: number;
};

export type LetterboxTransform = {
  sourceWidth: number;
  sourceHeight: number;
  targetSize: number;
  scaledWidth: number;
  scaledHeight: number;
  padX: number;
  padY: number;
  scaleX: number;
  scaleY: number;
};

// Model girdisini oranı bozmadan kare tuvale yerleştirmek için kullanılan dönüşümü üretir.
export function createLetterboxTransform(
  sourceWidth: number,
  sourceHeight: number,
  targetSize: number
): LetterboxTransform {
  if (sourceWidth <= 0 || sourceHeight <= 0 || targetSize <= 0) {
    throw new Error('Letterbox boyutları pozitif olmalı.');
  }

  const scale = Math.min(targetSize / sourceWidth, targetSize / sourceHeight);
  const scaledWidth = Math.max(1, Math.round(sourceWidth * scale));
  const scaledHeight = Math.max(1, Math.round(sourceHeight * scale));
  const padX = Math.floor((targetSize - scaledWidth) / 2);
  const padY = Math.floor((targetSize - scaledHeight) / 2);

  return {
    sourceWidth,
    sourceHeight,
    targetSize,
    scaledWidth,
    scaledHeight,
    padX,
    padY,
    scaleX: scaledWidth / sourceWidth,
    scaleY: scaledHeight / sourceHeight,
  };
}

// Canlı frame'i gereksiz yere büyütmeden en fazla hedef kenara sığdırır.
export function getAspectPreservingSize(
  sourceWidth: number,
  sourceHeight: number,
  maximumEdge: number
) {
  if (sourceWidth <= 0 || sourceHeight <= 0 || maximumEdge <= 0) {
    throw new Error('Görüntü boyutları pozitif olmalı.');
  }

  const scale = Math.min(1, maximumEdge / Math.max(sourceWidth, sourceHeight));
  return {
    width: Math.max(1, Math.round(sourceWidth * scale)),
    height: Math.max(1, Math.round(sourceHeight * scale)),
  };
}

// Model karesindeki koordinatı, padding ve oran dönüşümünü geri alarak kaynak görüntüye taşır.
export function mapModelPointToNormalizedImage(
  modelX: number,
  modelY: number,
  transform: LetterboxTransform
): NormalizedPoint {
  return {
    x: clamp01(
      ((modelX - transform.padX) / transform.scaleX) /
        transform.sourceWidth
    ),
    y: clamp01(
      ((modelY - transform.padY) / transform.scaleY) /
        transform.sourceHeight
    ),
  };
}

// Çizgi ekseni 180 derece simetrik olduğu için en kısa eşdeğer dönüşü seçer.
export function normalizeAxisRotation(degrees: number) {
  let normalized = degrees;
  while (normalized > 90) normalized -= 180;
  while (normalized < -90) normalized += 180;
  return normalized;
}

// OBB köşelerinden uzun ekseni ve dikleştirme dönüşünü hesaplar.
export function getCanonicalGeometry(points?: NormalizedPoint[]) {
  if (!points || points.length !== 4) return null;

  const edges = points.map((point, index) => {
    const next = points[(index + 1) % points.length];
    const deltaX = next.x - point.x;
    const deltaY = next.y - point.y;
    return {
      length: Math.hypot(deltaX, deltaY),
      angleDegrees: (Math.atan2(deltaY, deltaX) * 180) / Math.PI,
    };
  });
  const longEdge = [...edges].sort((first, second) => second.length - first.length)[0];
  const shortEdge = [...edges].sort((first, second) => first.length - second.length)[0];

  if (!longEdge || !shortEdge || shortEdge.length <= Number.EPSILON) return null;

  return {
    longEdge: longEdge.length,
    shortEdge: shortEdge.length,
    rotationDegrees: normalizeAxisRotation(90 - longEdge.angleDegrees),
  };
}

function clamp01(value: number) {
  return Math.min(Math.max(value, 0), 1);
}
