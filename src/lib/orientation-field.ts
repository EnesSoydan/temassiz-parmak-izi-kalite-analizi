export type OrientationBlock = {
  left: number;
  top: number;
  size: number;
  angleRadians: number;
  rawAngleRadians?: number;
  coherence: number;
  smoothedCoherence?: number;
  neighborhoodConsistency: number;
  gradientEnergy: number;
  maskCoverage: number;
  gridOffset?: number;
};

type EstimateOrientationFieldInput = {
  grayscale: Uint8Array;
  mask: Uint8Array;
  width: number;
  height: number;
  blockSize: number;
  minMaskCoverage: number;
};

// Blok tabanlı ridge orientation alanını gradient tensor formülüyle tahmin eder.
export function estimateOrientationField({
  grayscale,
  mask,
  width,
  height,
  blockSize,
  minMaskCoverage,
}: EstimateOrientationFieldInput) {
  const blocks: OrientationBlock[] = [];
  const gridOffsets = [0, Math.max(1, Math.floor(blockSize / 2))];

  // İkinci yarım-blok ızgara, parmak çizgisinin blok sınırına denk gelmesine bağlı kararsızlığı azaltır.
  for (const gridOffset of gridOffsets) {
    for (let top = 1 + gridOffset; top + blockSize < height - 1; top += blockSize) {
      for (let left = 1 + gridOffset; left + blockSize < width - 1; left += blockSize) {
        const block = estimateOrientationBlock(
          grayscale,
          mask,
          width,
          left,
          top,
          blockSize,
          minMaskCoverage
        );

        if (block) blocks.push({ ...block, gridOffset });
      }
    }
  }

  return smoothOrientationField(blocks);
}

// Tek blokta Sobel benzeri türevlerden 0-180 derece eksenli ridge yönü ve tutarlılık hesaplar.
function estimateOrientationBlock(
  grayscale: Uint8Array,
  mask: Uint8Array,
  width: number,
  left: number,
  top: number,
  blockSize: number,
  minMaskCoverage: number
): OrientationBlock | null {
  let maskedCount = 0;
  let pixelCount = 0;
  let tensorX = 0;
  let tensorY = 0;
  let tensorMagnitude = 0;

  for (let y = top; y < top + blockSize; y += 1) {
    for (let x = left; x < left + blockSize; x += 1) {
      pixelCount += 1;
      const index = y * width + x;
      if (!mask[index]) continue;

      const value = grayscale[index];
      const gradientX = (grayscale[index + 1] ?? value) - (grayscale[index - 1] ?? value);
      const gradientY = (grayscale[index + width] ?? value) - (grayscale[index - width] ?? value);
      const energy = gradientX * gradientX + gradientY * gradientY;

      maskedCount += 1;
      tensorX += gradientX * gradientX - gradientY * gradientY;
      tensorY += 2 * gradientX * gradientY;
      tensorMagnitude += energy;
    }
  }

  const maskCoverage = maskedCount / Math.max(pixelCount, 1);
  if (maskCoverage < minMaskCoverage || maskedCount < 12) return null;

  const coherence = Math.hypot(tensorX, tensorY) / Math.max(tensorMagnitude, 1);

  return {
    left,
    top,
    size: blockSize,
    // Gradient yönünün dik açısı ridge akış yönünü verir; sonuç 0-180 derece eksenlidir.
    angleRadians: normalizeHalfCircleAngle(0.5 * Math.atan2(tensorY, tensorX) + Math.PI / 2),
    coherence,
    neighborhoodConsistency: 0,
    gradientEnergy: tensorMagnitude / maskedCount,
    maskCoverage,
  };
}

// Yönleri 180 derece eksenli yapıya uygun çift-açı vektörleriyle komşuluk içinde yumuşatır.
function smoothOrientationField(blocks: OrientationBlock[]) {
  return blocks.map((block) => {
    const neighbors = blocks.filter(
      (candidate) =>
        Math.abs(candidate.left - block.left) <= block.size * 1.25 &&
        Math.abs(candidate.top - block.top) <= block.size * 1.25
    );

    let vectorX = 0;
    let vectorY = 0;
    let totalWeight = 0;

    for (const neighbor of neighbors) {
      const centerDistance = Math.hypot(
        neighbor.left - block.left,
        neighbor.top - block.top
      );
      const distanceWeight = 1 / (1 + centerDistance / Math.max(block.size, 1));
      const weight = Math.max(neighbor.coherence, 0.05) * distanceWeight;
      vectorX += Math.cos(neighbor.angleRadians * 2) * weight;
      vectorY += Math.sin(neighbor.angleRadians * 2) * weight;
      totalWeight += weight;
    }

    const smoothedAngle = normalizeHalfCircleAngle(
      0.5 * Math.atan2(vectorY, vectorX)
    );
    const smoothedCoherence = Math.hypot(vectorX, vectorY) / Math.max(totalWeight, 0.0001);
    const comparableNeighbors = neighbors.filter((candidate) => candidate !== block);
    const neighborhoodConsistency =
      comparableNeighbors.length === 0
        ? block.coherence
        : comparableNeighbors.reduce((total, neighbor) => {
            const axisSimilarity =
              (Math.cos((smoothedAngle - neighbor.angleRadians) * 2) + 1) / 2;
            return total + axisSimilarity;
          }, 0) / comparableNeighbors.length;

    return {
      ...block,
      rawAngleRadians: block.angleRadians,
      angleRadians: smoothedAngle,
      smoothedCoherence,
      neighborhoodConsistency,
    };
  });
}

// Ridge yönünü 0 ile pi arasında tutarak sağ/sol yön değil çizgi ekseni olarak saklar.
function normalizeHalfCircleAngle(angleRadians: number) {
  let angle = angleRadians % Math.PI;
  if (angle < 0) angle += Math.PI;
  return angle;
}
