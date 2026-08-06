export type RidgeScaledRgbaMask = {
  pixels: Uint8Array;
  mask: Uint8Array;
  width: number;
  height: number;
  scaleFactor: number;
};

export const DEFAULT_TARGET_RIDGE_PERIOD_PIXELS = 8;

// Temassız ROI'leri fiziksel ppi yerine aynı cihazdaki ridge periyoduna göre ortak alana taşır.
export function calculateRidgeScaleFactor(
  measuredPeriodPixels: number,
  targetPeriodPixels = DEFAULT_TARGET_RIDGE_PERIOD_PIXELS
) {
  if (
    !Number.isFinite(measuredPeriodPixels) ||
    measuredPeriodPixels <= 0 ||
    !Number.isFinite(targetPeriodPixels) ||
    targetPeriodPixels <= 0
  ) {
    return 1;
  }

  // Aşırı ölçekleme gürültüyü büyütür; gerçek kalibrasyon bu aralığı veriyle güncelleyebilir.
  return clamp(targetPeriodPixels / measuredPeriodPixels, 0.7, 1.4);
}

// Gri/RGBA veriyi bilinear, maskeyi nearest-neighbor örneklemeyle aynı dönüşümde ölçekler.
export function rescaleRgbaAndMask({
  pixels,
  mask,
  width,
  height,
  scaleFactor,
}: {
  pixels: Uint8Array;
  mask: Uint8Array;
  width: number;
  height: number;
  scaleFactor: number;
}): RidgeScaledRgbaMask {
  if (
    width <= 0 ||
    height <= 0 ||
    pixels.length < width * height * 4 ||
    mask.length < width * height
  ) {
    throw new Error('Ridge ölçekleme için geçersiz görüntü tamponu.');
  }

  const safeScale = Number.isFinite(scaleFactor) && scaleFactor > 0
    ? scaleFactor
    : 1;
  const targetWidth = Math.max(1, Math.round(width * safeScale));
  const targetHeight = Math.max(1, Math.round(height * safeScale));
  const outputPixels = new Uint8Array(targetWidth * targetHeight * 4);
  const outputMask = new Uint8Array(targetWidth * targetHeight);

  for (let targetY = 0; targetY < targetHeight; targetY += 1) {
    const sourceY = Math.min(
      height - 1,
      Math.max(0, ((targetY + 0.5) / safeScale) - 0.5)
    );
    const top = Math.floor(sourceY);
    const bottom = Math.min(top + 1, height - 1);
    const yWeight = sourceY - top;

    for (let targetX = 0; targetX < targetWidth; targetX += 1) {
      const sourceX = Math.min(
        width - 1,
        Math.max(0, ((targetX + 0.5) / safeScale) - 0.5)
      );
      const left = Math.floor(sourceX);
      const right = Math.min(left + 1, width - 1);
      const xWeight = sourceX - left;
      const outputIndex = (targetY * targetWidth + targetX) * 4;
      const topLeft = (top * width + left) * 4;
      const topRight = (top * width + right) * 4;
      const bottomLeft = (bottom * width + left) * 4;
      const bottomRight = (bottom * width + right) * 4;

      for (let channel = 0; channel < 4; channel += 1) {
        const topValue =
          pixels[topLeft + channel] * (1 - xWeight) +
          pixels[topRight + channel] * xWeight;
        const bottomValue =
          pixels[bottomLeft + channel] * (1 - xWeight) +
          pixels[bottomRight + channel] * xWeight;
        outputPixels[outputIndex + channel] = Math.round(
          topValue * (1 - yWeight) + bottomValue * yWeight
        );
      }

      const nearestX = Math.min(width - 1, Math.max(0, Math.round(sourceX)));
      const nearestY = Math.min(height - 1, Math.max(0, Math.round(sourceY)));
      outputMask[targetY * targetWidth + targetX] =
        mask[nearestY * width + nearestX] ?? 0;
    }
  }

  return {
    pixels: outputPixels,
    mask: outputMask,
    width: targetWidth,
    height: targetHeight,
    scaleFactor: safeScale,
  };
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(Math.max(value, minimum), maximum);
}
