import type { OrientationBlock } from '@/lib/orientation-field';
import type { RidgeFrequencyBlock } from '@/lib/ridge-frequency';

type OrientationVisualizationInput = {
  grayscale: Uint8Array;
  mask: Uint8Array;
  width: number;
  height: number;
  orientationBlocks: OrientationBlock[];
  detailCandidateBlocks?: OrientationBlock[];
  detailOrientationBlocks?: OrientationBlock[];
  frequencyBlocks?: RidgeFrequencyBlock[];
};

// Yumuşatılmış ridge eksenlerini ve frekans kanıt kararlarını aynı doğrulama haritasında gösterir.
export function createOrientationVisualization({
  grayscale,
  mask,
  width,
  height,
  orientationBlocks,
  detailCandidateBlocks = [],
  detailOrientationBlocks = [],
  frequencyBlocks = [],
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

  const fineCandidateBlocks = detailCandidateBlocks.filter(
    (block) => (block.gridOffset ?? 0) === 0 && block.maskCoverage >= 0.85
  );
  const verifiedFineBlockKeys = new Set(
    detailOrientationBlocks.map(getOrientationBlockKey)
  );

  if (fineCandidateBlocks.length > 0) {
    // Her 8x8 hücreyi kendi yerel yön güveni ve kaba yönle uyumuna göre ayrı çerçeveleriz.
    for (const block of fineCandidateBlocks) {
      const confidence = getOrientationBlockConfidence(block);
      const color: [number, number, number] = verifiedFineBlockKeys.has(
        getOrientationBlockKey(block)
      )
        ? [47, 209, 107]
        : confidence >= 0.45
          ? [229, 72, 77]
          : [245, 184, 68];
      drawBlockOutline(pixels, mask, width, height, block, color);
    }
  } else {
    // İnce deney kapalıysa eski frekans kanıt çerçevelerini kaba yön ölçeğinde gösteririz.
    for (const block of primaryBlocks.filter(
      (item) => item.maskCoverage >= 0.85
    )) {
      const frequencyBlock = findFrequencyBlockForOrientationBlock(
        block,
        frequencyBlocks
      );
      const color: [number, number, number] = frequencyBlock
        ? frequencyBlock.valid
          ? [47, 209, 107]
          : [229, 72, 77]
        : [245, 184, 68];
      drawBlockOutline(pixels, mask, width, height, block, color);
    }
  }

  const directionBlocks =
    fineCandidateBlocks.length > 0 ? fineCandidateBlocks : primaryBlocks;

  // İnce ızgarada her hücrenin yönünü gösterir; doğrulanmayan çizgileri gri tutarak sonuçtan ayırırız.
  for (const block of directionBlocks) {
    const confidence = getOrientationBlockConfidence(block);
    const isVerifiedFineBlock = verifiedFineBlockKeys.has(
      getOrientationBlockKey(block)
    );
    const color =
      fineCandidateBlocks.length > 0 && !isVerifiedFineBlock
        ? ([120, 130, 140] as [number, number, number])
        : getOrientationColor(confidence);
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
      color,
      block.size <= 10 ? 0 : 1
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

// Aynı fiziksel ince hücreyi aday ve doğrulanmış listeler arasında kararlı biçimde eşleştirir.
function getOrientationBlockKey(block: OrientationBlock) {
  return `${block.left}:${block.top}:${block.size}:${block.gridOffset ?? 0}`;
}

// İki kaydırılmış ızgaradaki frekans ölçümünü aynı fiziksel ana yön hücresiyle eşleştirir.
function findFrequencyBlockForOrientationBlock(
  orientationBlock: OrientationBlock,
  frequencyBlocks: RidgeFrequencyBlock[]
) {
  const centerX = orientationBlock.left + orientationBlock.size / 2;
  const centerY = orientationBlock.top + orientationBlock.size / 2;
  let nearest: RidgeFrequencyBlock | undefined;
  let nearestDistance = Number.POSITIVE_INFINITY;

  for (const block of frequencyBlocks) {
    const distance = Math.hypot(
      block.left + block.size / 2 - centerX,
      block.top + block.size / 2 - centerY
    );
    if (
      distance <= orientationBlock.size * 0.8 &&
      distance < nearestDistance
    ) {
      nearest = block;
      nearestDistance = distance;
    }
  }

  return nearest;
}

// Ham coherence ve komşuluk devamlılığını tek görsel güven değerinde birleştirir.
function getOrientationBlockConfidence(block: OrientationBlock) {
  return clamp01(
    block.coherence * 0.5 +
      block.neighborhoodConsistency * 0.3 +
      (block.smoothedCoherence ?? block.coherence) * 0.2
  );
}

// Kanıt çerçevelerinden ayrılması için ridge yönünü güvene göre açık mavi ve gri tonlarda çizer.
function getOrientationColor(confidence: number): [number, number, number] {
  if (confidence >= 0.58) return [90, 190, 255];
  if (confidence >= 0.38) return [190, 220, 240];
  return [120, 130, 140];
}

// Blok kararını doku görüntüsünü kapatmadan tek piksel çerçeve olarak çizer.
function drawBlockOutline(
  pixels: Uint8Array,
  mask: Uint8Array,
  width: number,
  height: number,
  block: OrientationBlock,
  color: [number, number, number]
) {
  const left = Math.round(block.left);
  const top = Math.round(block.top);
  const right = Math.round(block.left + block.size - 1);
  const bottom = Math.round(block.top + block.size - 1);

  for (let x = left; x <= right; x += 1) {
    writeColorPixel(pixels, mask, width, height, x, top, color);
    writeColorPixel(pixels, mask, width, height, x, bottom, color);
  }
  for (let y = top; y <= bottom; y += 1) {
    writeColorPixel(pixels, mask, width, height, left, y, color);
    writeColorPixel(pixels, mask, width, height, right, y, color);
  }
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
  color: [number, number, number],
  thickness: number
) {
  const steps = Math.max(
    1,
    Math.ceil(Math.max(Math.abs(endX - startX), Math.abs(endY - startY)))
  );

  for (let step = 0; step <= steps; step += 1) {
    const ratio = step / steps;
    const x = Math.round(startX + (endX - startX) * ratio);
    const y = Math.round(startY + (endY - startY) * ratio);

    for (let offsetY = -thickness; offsetY <= thickness; offsetY += 1) {
      for (let offsetX = -thickness; offsetX <= thickness; offsetX += 1) {
        if (Math.abs(offsetX) + Math.abs(offsetY) > thickness) continue;
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
