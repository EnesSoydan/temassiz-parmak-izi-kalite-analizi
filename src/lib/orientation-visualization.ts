import type { OrientationBlock } from '@/lib/orientation-field';

type OrientationVisualizationInput = {
  grayscale: Uint8Array;
  mask: Uint8Array;
  width: number;
  height: number;
  orientationBlocks: OrientationBlock[];
};

// Yumuşatılmış ridge eksenlerini segmentli gri görüntü üzerinde güven renkleriyle gösterir.
export function createOrientationVisualization({
  grayscale,
  mask,
  width,
  height,
  orientationBlocks,
}: OrientationVisualizationInput) {
  const pixels = new Uint8Array(mask.length * 4);
  const primaryBlocks = orientationBlocks.filter(
    (block) => (block.gridOffset ?? 0) === 0
  );

  // Koyu gri taban, hem parmak dokusunu korur hem de renkli yön çizgilerinin seçilmesini sağlar.
  for (let index = 0; index < mask.length; index += 1) {
    const outputIndex = index * 4;
    const value = mask[index] ? Math.round(grayscale[index] * 0.38) : 0;
    pixels[outputIndex] = value;
    pixels[outputIndex + 1] = value;
    pixels[outputIndex + 2] = value;
    pixels[outputIndex + 3] = 255;
  }

  for (const block of primaryBlocks) {
    const confidence = getOrientationBlockConfidence(block);
    const color = getConfidenceColor(confidence);
    const centerX = block.left + block.size / 2;
    const centerY = block.top + block.size / 2;
    const halfLength = block.size * 0.34;
    const deltaX = Math.cos(block.angleRadians) * halfLength;
    const deltaY = Math.sin(block.angleRadians) * halfLength;

    drawThickLine(
      pixels,
      mask,
      width,
      height,
      centerX - deltaX,
      centerY - deltaY,
      centerX + deltaX,
      centerY + deltaY,
      color
    );
  }

  return {
    pixels,
    reliableBlockRatio:
      primaryBlocks.filter(
        (block) => getOrientationBlockConfidence(block) >= 0.58
      ).length / Math.max(primaryBlocks.length, 1),
    medianAngularCorrectionDegrees: getMedian(
      primaryBlocks.map((block) =>
        getAxisAngleDifferenceDegrees(
          block.rawAngleRadians ?? block.angleRadians,
          block.angleRadians
        )
      )
    ),
  };
}

// Ham coherence ve komşuluk devamlılığını tek görsel güven değerinde birleştirir.
function getOrientationBlockConfidence(block: OrientationBlock) {
  return clamp01(
    block.coherence * 0.5 +
      block.neighborhoodConsistency * 0.3 +
      (block.smoothedCoherence ?? block.coherence) * 0.2
  );
}

// Yüksek güveni yeşil, orta güveni sarı, düşük güveni kırmızı olarak kodlar.
function getConfidenceColor(confidence: number): [number, number, number] {
  if (confidence >= 0.58) return [47, 209, 107];
  if (confidence >= 0.38) return [245, 184, 68];
  return [229, 72, 77];
}

// Görüntü koordinatlarında iki piksel kalınlığında yön ekseni çizer.
function drawThickLine(
  pixels: Uint8Array,
  mask: Uint8Array,
  width: number,
  height: number,
  startX: number,
  startY: number,
  endX: number,
  endY: number,
  color: [number, number, number]
) {
  const steps = Math.max(
    1,
    Math.ceil(Math.max(Math.abs(endX - startX), Math.abs(endY - startY)))
  );

  for (let step = 0; step <= steps; step += 1) {
    const ratio = step / steps;
    const x = Math.round(startX + (endX - startX) * ratio);
    const y = Math.round(startY + (endY - startY) * ratio);

    for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
      for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
        if (Math.abs(offsetX) + Math.abs(offsetY) > 1) continue;
        writeColorPixel(
          pixels,
          mask,
          width,
          height,
          x + offsetX,
          y + offsetY,
          color
        );
      }
    }
  }
}

// Yön çizgisini yalnızca geçerli parmak maskesinin içinde kalan piksele yazar.
function writeColorPixel(
  pixels: Uint8Array,
  mask: Uint8Array,
  width: number,
  height: number,
  x: number,
  y: number,
  color: [number, number, number]
) {
  if (x < 0 || x >= width || y < 0 || y >= height) return;
  const index = y * width + x;
  if (!mask[index]) return;
  const outputIndex = index * 4;
  pixels[outputIndex] = color[0];
  pixels[outputIndex + 1] = color[1];
  pixels[outputIndex + 2] = color[2];
  pixels[outputIndex + 3] = 255;
}

// 180 derece simetrik iki ridge ekseni arasındaki en küçük açı farkını derece olarak hesaplar.
function getAxisAngleDifferenceDegrees(first: number, second: number) {
  const rawDifference = Math.abs(first - second) % Math.PI;
  const axisDifference = Math.min(rawDifference, Math.PI - rawDifference);
  return (axisDifference * 180) / Math.PI;
}

// Sayı dizisinin ortanca değerini açısal düzeltme özetinde kullanır.
function getMedian(values: number[]) {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((first, second) => first - second);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? ((sorted[middle - 1] ?? 0) + (sorted[middle] ?? 0)) / 2
    : (sorted[middle] ?? 0);
}

// Güven değerini geçerli oran aralığında tutar.
function clamp01(value: number) {
  return Math.min(Math.max(value, 0), 1);
}
