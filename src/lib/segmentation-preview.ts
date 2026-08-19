const PREVIEW_MINIMUM = 32;
const PREVIEW_MAXIMUM = 224;
const LOCAL_CONTRAST_GAIN = 1.2;

// Segmentasyon önizlemesini analiz tamponuna dokunmadan, maskeli ve doygunluğu sınırlı gri tonda üretir.
export function createSegmentedPreviewPixels(
  sourcePixels: Uint8Array,
  mask: Uint8Array,
  width: number,
  height: number
) {
  const pixelCount = width * height;
  if (sourcePixels.length < pixelCount * 4 || mask.length < pixelCount) {
    throw new RangeError('Segmentasyon önizleme tamponunun boyutu geçersiz.');
  }

  const grayscale = createGrayscaleImage(sourcePixels, pixelCount);
  const normalized = normalizeMaskedLocalBrightness(
    grayscale,
    mask,
    width,
    height
  );
  const toned = compressMaskedContrast(normalized, mask);
  const outputPixels = new Uint8Array(pixelCount * 4);

  for (let index = 0; index < pixelCount; index += 1) {
    const dataIndex = index * 4;
    const value = mask[index] ? toned[index] : 0;
    outputPixels[dataIndex] = value;
    outputPixels[dataIndex + 1] = value;
    outputPixels[dataIndex + 2] = value;
    outputPixels[dataIndex + 3] = 255;
  }

  return outputPixels;
}

function createGrayscaleImage(pixels: Uint8Array, pixelCount: number) {
  const grayscale = new Uint8Array(pixelCount);

  for (let index = 0; index < pixelCount; index += 1) {
    const dataIndex = index * 4;
    grayscale[index] = Math.round(
      pixels[dataIndex] * 0.299 +
        pixels[dataIndex + 1] * 0.587 +
        pixels[dataIndex + 2] * 0.114
    );
  }

  return grayscale;
}

// Maske dışındaki arka planın lokal ortalamayı bozmasını engelleyerek parlama bölgelerini yumuşakça merkezler.
function normalizeMaskedLocalBrightness(
  grayscale: Uint8Array,
  mask: Uint8Array,
  width: number,
  height: number
) {
  const radius = Math.max(
    5,
    Math.min(12, Math.round(Math.min(width, height) / 12))
  );
  const binaryMask = normalizeBinaryMask(mask);
  const valueIntegral = createMaskedIntegralImage(
    grayscale,
    binaryMask,
    width,
    height
  );
  const maskIntegral = createIntegralImage(binaryMask, width, height);
  const normalized = new Uint8Array(grayscale.length);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      if (!mask[index]) continue;

      const left = Math.max(0, x - radius);
      const top = Math.max(0, y - radius);
      const right = Math.min(width - 1, x + radius);
      const bottom = Math.min(height - 1, y + radius);
      const foregroundCount = readIntegralSum(
        maskIntegral,
        width,
        left,
        top,
        right,
        bottom
      );
      const localMean =
        foregroundCount > 0
          ? readIntegralSum(
              valueIntegral,
              width,
              left,
              top,
              right,
              bottom
            ) / foregroundCount
          : grayscale[index];

      normalized[index] = clampByte(
        128 + (grayscale[index] - localMean) * LOCAL_CONTRAST_GAIN
      );
    }
  }

  return normalized;
}

// Yüzdelik kontrastı tam siyah-beyaza taşımak yerine güvenli gri aralıkta tutar.
function compressMaskedContrast(grayscale: Uint8Array, mask: Uint8Array) {
  const output = new Uint8Array(grayscale.length);
  const { low, high } = getMaskedContrastBounds(grayscale, mask);
  const midpoint = (low + high) / 2;
  const range = Math.max(high - low, 48);
  const lowerBound = midpoint - range / 2;

  for (let index = 0; index < grayscale.length; index += 1) {
    if (!mask[index]) continue;
    const normalized = clamp((grayscale[index] - lowerBound) / range, 0, 1);
    output[index] = Math.round(
      PREVIEW_MINIMUM + normalized * (PREVIEW_MAXIMUM - PREVIEW_MINIMUM)
    );
  }

  return output;
}

function normalizeBinaryMask(mask: Uint8Array) {
  const binaryMask = new Uint8Array(mask.length);
  for (let index = 0; index < mask.length; index += 1) {
    if (mask[index]) binaryMask[index] = 1;
  }
  return binaryMask;
}

function createMaskedIntegralImage(
  values: Uint8Array,
  mask: Uint8Array,
  width: number,
  height: number
) {
  const maskedValues = new Uint8Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    if (mask[index]) maskedValues[index] = values[index];
  }
  return createIntegralImage(maskedValues, width, height);
}

function createIntegralImage(
  values: Uint8Array,
  width: number,
  height: number
) {
  const integral = new Float64Array((width + 1) * (height + 1));

  for (let y = 1; y <= height; y += 1) {
    let rowTotal = 0;
    for (let x = 1; x <= width; x += 1) {
      rowTotal += values[(y - 1) * width + (x - 1)];
      integral[y * (width + 1) + x] =
        integral[(y - 1) * (width + 1) + x] + rowTotal;
    }
  }

  return integral;
}

function readIntegralSum(
  integral: Float64Array,
  width: number,
  left: number,
  top: number,
  right: number,
  bottom: number
) {
  const stride = width + 1;
  const x2 = right + 1;
  const y2 = bottom + 1;
  return (
    integral[y2 * stride + x2] -
    integral[top * stride + x2] -
    integral[y2 * stride + left] +
    integral[top * stride + left]
  );
}

function getMaskedContrastBounds(grayscale: Uint8Array, mask: Uint8Array) {
  const histogram = new Uint32Array(256);
  let count = 0;

  for (let index = 0; index < grayscale.length; index += 1) {
    if (!mask[index]) continue;
    histogram[grayscale[index]] += 1;
    count += 1;
  }

  return {
    low: getHistogramPercentile(histogram, count, 0.03),
    high: getHistogramPercentile(histogram, count, 0.97),
  };
}

function getHistogramPercentile(
  histogram: Uint32Array,
  count: number,
  percentile: number
) {
  if (count === 0) return 128;
  const target = Math.floor((count - 1) * percentile);
  let cumulative = 0;

  for (let value = 0; value < histogram.length; value += 1) {
    cumulative += histogram[value];
    if (cumulative > target) return value;
  }

  return 255;
}

function clampByte(value: number) {
  return Math.round(clamp(value, 0, 255));
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(Math.max(value, minimum), maximum);
}
