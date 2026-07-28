import * as FileSystem from 'expo-file-system/legacy';
import { manipulateAsync, SaveFormat } from 'expo-image-manipulator';
import { Buffer } from 'buffer';
import { decode, encode } from 'jpeg-js';

export type SegmentationResult = {
  imageUri: string;
  pixels: Uint8Array;
  mask: Uint8Array;
  width: number;
  height: number;
  coverage: number;
};

type YCrCbColor = {
  y: number;
  cr: number;
  cb: number;
};

// Saf JavaScript piksel işlemini kısa tutmak için analiz çözünürlüğünü sınırlarız.
const MAX_ANALYSIS_WIDTH = 384;

// Parmak silüeti için ten rengi aralığını merkezdeki örnekle birlikte kullanırız.
const SKIN_RANGE = {
  yMin: 20,
  crMin: 125,
  crMax: 185,
  cbMin: 65,
  cbMax: 150,
};

// Maske kapsaması bu aralığın dışındaysa parmak alanı güvenilir kabul edilmez.
const MIN_MASK_COVERAGE = 0.18;
const MAX_MASK_COVERAGE = 0.96;

// ROI JPEG dosyasını küçültüp çözer; aynı piksel verisi hem segmentasyon hem kalite için paylaşılır.
export async function segmentFingerRoiImage({
  roiImageUri,
  outputImageUri,
  sourceWidth,
}: {
  roiImageUri: string;
  outputImageUri: string;
  sourceWidth: number;
}): Promise<SegmentationResult | null> {
  ensureJpegBufferShim();

  const image = await loadAnalysisImage(roiImageUri, sourceWidth);
  const sourcePixels = new Uint8Array(image.data);
  const targetColor = sampleCenterSkinColor(sourcePixels, image.width, image.height);
  const initialMask = createSkinMask(sourcePixels, image.width, image.height, targetColor);
  const closedMask = erode(
    dilate(initialMask, image.width, image.height, 2),
    image.width,
    image.height,
    1
  );
  const cleanedMask = dilate(
    erode(closedMask, image.width, image.height, 1),
    image.width,
    image.height,
    1
  );
  const mask = keepCenterConnectedComponent(cleanedMask, image.width, image.height);
  const coverage = countMaskPixels(mask) / Math.max(image.width * image.height, 1);

  if (coverage < MIN_MASK_COVERAGE || coverage > MAX_MASK_COVERAGE) {
    return null;
  }

  const segmentedPixels = createEnhancedSegmentedPixels(sourcePixels, mask, image.width, image.height);

  const jpeg = encode({ data: segmentedPixels, width: image.width, height: image.height }, 95);
  await FileSystem.writeAsStringAsync(outputImageUri, bytesToBase64(jpeg.data), {
    encoding: FileSystem.EncodingType.Base64,
  });

  return {
    imageUri: outputImageUri,
    pixels: sourcePixels,
    mask,
    width: image.width,
    height: image.height,
    coverage,
  };
}

// Segmentasyon çıktısını yalnızca silüet değil, ridge dokusunu daha görünür yapan gri görüntü olarak üretir.
function createEnhancedSegmentedPixels(
  sourcePixels: Uint8Array,
  mask: Uint8Array,
  width: number,
  height: number
) {
  const grayscale = createGrayscaleImage(sourcePixels, width, height);
  const normalized = normalizeLocalBrightness(grayscale, mask, width, height);
  const stretched = stretchMaskedContrast(normalized, mask);
  const sharpened = sharpenMaskedGrayscale(stretched, mask, width, height);
  const outputPixels = new Uint8Array(sourcePixels.length);

  for (let pixelIndex = 0; pixelIndex < mask.length; pixelIndex += 1) {
    const dataIndex = pixelIndex * 4;
    const value = mask[pixelIndex] ? sharpened[pixelIndex] : 0;

    outputPixels[dataIndex] = value;
    outputPixels[dataIndex + 1] = value;
    outputPixels[dataIndex + 2] = value;
    outputPixels[dataIndex + 3] = 255;
  }

  return outputPixels;
}

// RGB piksel dizisini luminance tabanlı gri tona çevirir.
function createGrayscaleImage(pixels: Uint8Array, width: number, height: number) {
  const grayscale = new Uint8Array(width * height);

  for (let index = 0; index < grayscale.length; index += 1) {
    const pixelIndex = index * 4;
    grayscale[index] = Math.round(
      (pixels[pixelIndex] ?? 0) * 0.299 +
        (pixels[pixelIndex + 1] ?? 0) * 0.587 +
        (pixels[pixelIndex + 2] ?? 0) * 0.114
    );
  }

  return grayscale;
}

// Parmak üzerindeki yavaş ışık değişimini bastırıp lokal çizgi farklarını öne çıkarır.
function normalizeLocalBrightness(
  grayscale: Uint8Array,
  mask: Uint8Array,
  width: number,
  height: number
) {
  const radius = Math.max(5, Math.min(12, Math.round(Math.min(width, height) / 12)));
  const integral = createIntegralImage(grayscale, width, height);
  const normalized = new Uint8Array(grayscale.length);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      if (!mask[index]) continue;

      const left = Math.max(0, x - radius);
      const top = Math.max(0, y - radius);
      const right = Math.min(width - 1, x + radius);
      const bottom = Math.min(height - 1, y + radius);
      const area = (right - left + 1) * (bottom - top + 1);
      const localMean = readIntegralSum(integral, width, left, top, right, bottom) / area;

      normalized[index] = clampByte(128 + (grayscale[index] - localMean) * 1.65);
    }
  }

  return normalized;
}

// Maskeli alanın yüzde değerlerine göre kontrastı güvenli şekilde genişletir.
function stretchMaskedContrast(grayscale: Uint8Array, mask: Uint8Array) {
  const values: number[] = [];
  const stretched = new Uint8Array(grayscale.length);

  for (let index = 0; index < grayscale.length; index += 1) {
    if (mask[index]) values.push(grayscale[index]);
  }

  const low = getPercentile(values, 0.04);
  const high = getPercentile(values, 0.96);
  const range = Math.max(high - low, 12);

  for (let index = 0; index < grayscale.length; index += 1) {
    if (!mask[index]) continue;
    stretched[index] = clampByte(((grayscale[index] - low) / range) * 255);
  }

  return stretched;
}

// Hafif unsharp mask ile ridge benzeri ince geçişleri abartmadan belirginleştirir.
function sharpenMaskedGrayscale(
  grayscale: Uint8Array,
  mask: Uint8Array,
  width: number,
  height: number
) {
  const blurred = blurGrayscale(grayscale, width, height);
  const sharpened = new Uint8Array(grayscale.length);

  for (let index = 0; index < grayscale.length; index += 1) {
    if (!mask[index]) continue;

    sharpened[index] = clampByte(grayscale[index] + (grayscale[index] - blurred[index]) * 0.85);
  }

  return sharpened;
}

// Küçük 3x3 bulanıklaştırma, keskinleştirme için düşük frekans referansı üretir.
function blurGrayscale(grayscale: Uint8Array, width: number, height: number) {
  const blurred = new Uint8Array(grayscale.length);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let total = 0;
      let count = 0;

      for (let dy = -1; dy <= 1; dy += 1) {
        for (let dx = -1; dx <= 1; dx += 1) {
          const nx = x + dx;
          const ny = y + dy;

          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
          total += grayscale[ny * width + nx];
          count += 1;
        }
      }

      blurred[y * width + x] = Math.round(total / Math.max(count, 1));
    }
  }

  return blurred;
}

// Lokal ortalama hesabını hızlandırmak için klasik integral görüntü üretir.
function createIntegralImage(grayscale: Uint8Array, width: number, height: number) {
  const integral = new Float64Array((width + 1) * (height + 1));

  for (let y = 1; y <= height; y += 1) {
    let rowTotal = 0;

    for (let x = 1; x <= width; x += 1) {
      rowTotal += grayscale[(y - 1) * width + (x - 1)];
      integral[y * (width + 1) + x] = integral[(y - 1) * (width + 1) + x] + rowTotal;
    }
  }

  return integral;
}

// Integral görüntüden verilen dikdörtgenin toplam parlaklık değerini okur.
function readIntegralSum(
  integral: Float64Array,
  width: number,
  left: number,
  top: number,
  right: number,
  bottom: number
) {
  const stride = width + 1;
  const x1 = left;
  const y1 = top;
  const x2 = right + 1;
  const y2 = bottom + 1;

  return (
    integral[y2 * stride + x2] -
    integral[y1 * stride + x2] -
    integral[y2 * stride + x1] +
    integral[y1 * stride + x1]
  );
}

// Kaynak ROI büyükse native ImageManipulator ile küçültür, küçük ROI'yi ise olduğu gibi çözer.
async function loadAnalysisImage(roiImageUri: string, sourceWidth: number) {
  if (sourceWidth > MAX_ANALYSIS_WIDTH) {
    const resized = await manipulateAsync(
      roiImageUri,
      [{ resize: { width: MAX_ANALYSIS_WIDTH } }],
      { base64: true, compress: 0.95, format: SaveFormat.JPEG }
    );

    if (!resized.base64) {
      throw new Error('Kalite analizi için ROI görüntüsü hazırlanamadı.');
    }

    return decode(base64ToBytes(resized.base64), { useTArray: true });
  }

  const base64 = await FileSystem.readAsStringAsync(roiImageUri, {
    encoding: FileSystem.EncodingType.Base64,
  });
  return decode(base64ToBytes(base64), { useTArray: true });
}

// ROI merkezindeki ten benzeri piksellerden ışığa uyarlanmış renk referansı çıkarır.
function sampleCenterSkinColor(pixels: Uint8Array, width: number, height: number): YCrCbColor {
  const left = Math.floor(width * 0.28);
  const right = Math.ceil(width * 0.72);
  const top = Math.floor(height * 0.18);
  const bottom = Math.ceil(height * 0.82);
  let yTotal = 0;
  let crTotal = 0;
  let cbTotal = 0;
  let count = 0;

  for (let y = top; y < bottom; y += 1) {
    for (let x = left; x < right; x += 1) {
      const color = readPixelYCrCb(pixels, width, x, y);

      if (!isBroadSkinColor(color)) continue;

      yTotal += color.y;
      crTotal += color.cr;
      cbTotal += color.cb;
      count += 1;
    }
  }

  if (count === 0) {
    return readPixelYCrCb(pixels, width, Math.floor(width / 2), Math.floor(height / 2));
  }

  return { y: yTotal / count, cr: crTotal / count, cb: cbTotal / count };
}

// Geniş ten eşiği ve merkez renk yakınlığıyla ilk parmak maskesini oluşturur.
function createSkinMask(
  pixels: Uint8Array,
  width: number,
  height: number,
  targetColor: YCrCbColor
) {
  const mask = new Uint8Array(width * height);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const color = readPixelYCrCb(pixels, width, x, y);
      const chromaDistance = Math.abs(color.cr - targetColor.cr) + Math.abs(color.cb - targetColor.cb);
      const brightnessDistance = Math.abs(color.y - targetColor.y);

      if (isBroadSkinColor(color) && chromaDistance <= 42 && brightnessDistance <= 95) {
        mask[y * width + x] = 1;
      }
    }
  }

  return mask;
}

// Aynı renkteki arka plan parçalarını elemek için merkeze en yakın büyük bağlı alanı saklar.
function keepCenterConnectedComponent(mask: Uint8Array, width: number, height: number) {
  const labels = new Int32Array(mask.length);
  const queue = new Int32Array(mask.length);
  const centerX = (width - 1) / 2;
  const centerY = (height - 1) / 2;
  let bestLabel = 0;
  let bestScore = 0;
  let nextLabel = 0;

  for (let start = 0; start < mask.length; start += 1) {
    if (!mask[start] || labels[start]) continue;

    nextLabel += 1;
    let head = 0;
    let tail = 0;
    let area = 0;
    let xTotal = 0;
    let yTotal = 0;
    queue[tail] = start;
    tail += 1;
    labels[start] = nextLabel;

    while (head < tail) {
      const index = queue[head];
      head += 1;
      const x = index % width;
      const y = Math.floor(index / width);
      area += 1;
      xTotal += x;
      yTotal += y;

      if (x > 0 && !labels[index - 1] && mask[index - 1]) {
        labels[index - 1] = nextLabel;
        queue[tail] = index - 1;
        tail += 1;
      }
      if (x + 1 < width && !labels[index + 1] && mask[index + 1]) {
        labels[index + 1] = nextLabel;
        queue[tail] = index + 1;
        tail += 1;
      }
      if (y > 0 && !labels[index - width] && mask[index - width]) {
        labels[index - width] = nextLabel;
        queue[tail] = index - width;
        tail += 1;
      }
      if (y + 1 < height && !labels[index + width] && mask[index + width]) {
        labels[index + width] = nextLabel;
        queue[tail] = index + width;
        tail += 1;
      }
    }

    const componentX = xTotal / Math.max(area, 1);
    const componentY = yTotal / Math.max(area, 1);
    const centerDistance = Math.hypot(componentX - centerX, componentY - centerY) /
      Math.max(Math.hypot(centerX, centerY), 1);
    const score = area * (1.25 - Math.min(centerDistance, 1) * 0.5);

    if (score > bestScore) {
      bestScore = score;
      bestLabel = nextLabel;
    }
  }

  const selectedMask = new Uint8Array(mask.length);
  if (!bestLabel) return selectedMask;

  for (let index = 0; index < labels.length; index += 1) {
    selectedMask[index] = labels[index] === bestLabel ? 1 : 0;
  }

  return selectedMask;
}

// Pikselin geniş ten rengi aralığında kalıp kalmadığını döndürür.
function isBroadSkinColor(color: YCrCbColor) {
  return (
    color.y >= SKIN_RANGE.yMin &&
    color.cr >= SKIN_RANGE.crMin &&
    color.cr <= SKIN_RANGE.crMax &&
    color.cb >= SKIN_RANGE.cbMin &&
    color.cb <= SKIN_RANGE.cbMax
  );
}

// RGBA pikseli YCrCb uzayına çevirir.
function readPixelYCrCb(pixels: Uint8Array, width: number, x: number, y: number): YCrCbColor {
  const index = (y * width + x) * 4;
  const red = pixels[index] ?? 0;
  const green = pixels[index + 1] ?? 0;
  const blue = pixels[index + 2] ?? 0;
  const luminance = 0.299 * red + 0.587 * green + 0.114 * blue;

  return {
    y: luminance,
    cr: (red - luminance) * 0.713 + 128,
    cb: (blue - luminance) * 0.564 + 128,
  };
}

// Maskeyi genişleterek küçük boşlukları kapatır.
function dilate(mask: Uint8Array, width: number, height: number, radius: number) {
  const nextMask = new Uint8Array(mask.length);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let hasFilledNeighbor = false;

      for (let dy = -radius; dy <= radius && !hasFilledNeighbor; dy += 1) {
        for (let dx = -radius; dx <= radius; dx += 1) {
          const nx = x + dx;
          const ny = y + dy;

          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
          if (mask[ny * width + nx]) {
            hasFilledNeighbor = true;
            break;
          }
        }
      }

      nextMask[y * width + x] = hasFilledNeighbor ? 1 : 0;
    }
  }

  return nextMask;
}

// Maskeyi daraltarak tekil arka plan gürültüsünü temizler.
function erode(mask: Uint8Array, width: number, height: number, radius: number) {
  const nextMask = new Uint8Array(mask.length);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let allFilled = true;

      for (let dy = -radius; dy <= radius && allFilled; dy += 1) {
        for (let dx = -radius; dx <= radius; dx += 1) {
          const nx = x + dx;
          const ny = y + dy;

          if (nx < 0 || ny < 0 || nx >= width || ny >= height || !mask[ny * width + nx]) {
            allFilled = false;
            break;
          }
        }
      }

      nextMask[y * width + x] = allFilled ? 1 : 0;
    }
  }

  return nextMask;
}

// Maskedeki dolu piksel sayısını hesaplar.
function countMaskPixels(mask: Uint8Array) {
  let count = 0;

  for (let index = 0; index < mask.length; index += 1) {
    count += mask[index] ? 1 : 0;
  }

  return count;
}

// jpeg-js encoder React Native'de gerçek Buffer API'sine ihtiyaç duyduğu için MIT lisanslı polyfill'i globale bağlar.
function ensureJpegBufferShim() {
  const globalScope = globalThis as typeof globalThis & { Buffer?: typeof Buffer };

  globalScope.Buffer = Buffer;
}

// React Native tarafında Buffer'a yaslanmadan base64 string'i byte dizisine çevirir.
function base64ToBytes(base64: string) {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  const lookup = new Uint8Array(256);

  for (let i = 0; i < chars.length; i += 1) {
    lookup[chars.charCodeAt(i)] = i;
  }

  const clean = base64.replace(/[^A-Za-z0-9+/=]/g, '');
  const padding = clean.endsWith('==') ? 2 : clean.endsWith('=') ? 1 : 0;
  const length = Math.floor((clean.length * 3) / 4) - padding;
  const bytes = new Uint8Array(length);
  let byteIndex = 0;

  for (let i = 0; i < clean.length; i += 4) {
    const encoded =
      (lookup[clean.charCodeAt(i)] << 18) |
      (lookup[clean.charCodeAt(i + 1)] << 12) |
      (lookup[clean.charCodeAt(i + 2)] << 6) |
      lookup[clean.charCodeAt(i + 3)];

    if (byteIndex < length) bytes[byteIndex++] = (encoded >> 16) & 255;
    if (byteIndex < length) bytes[byteIndex++] = (encoded >> 8) & 255;
    if (byteIndex < length) bytes[byteIndex++] = encoded & 255;
  }

  return bytes;
}

// JPEG encoder çıktısını dosyaya yazılabilecek base64 string'e dönüştürür.
function bytesToBase64(bytes: Uint8Array) {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  let result = '';

  for (let i = 0; i < bytes.length; i += 3) {
    const byte1 = bytes[i] ?? 0;
    const byte2 = bytes[i + 1] ?? 0;
    const byte3 = bytes[i + 2] ?? 0;
    const encoded = (byte1 << 16) | (byte2 << 8) | byte3;

    result += chars[(encoded >> 18) & 63];
    result += chars[(encoded >> 12) & 63];
    result += i + 1 < bytes.length ? chars[(encoded >> 6) & 63] : '=';
    result += i + 2 < bytes.length ? chars[encoded & 63] : '=';
  }

  return result;
}

// Yüzdelik değerleri kontrast germe sırasında uç parlaklıkları bastırmak için kullanırız.
function getPercentile(values: number[], percentile: number) {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((first, second) => first - second);
  const index = (sorted.length - 1) * percentile;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const fraction = index - lower;

  return (sorted[lower] ?? 0) * (1 - fraction) + (sorted[upper] ?? 0) * fraction;
}

// Görüntü işlemlerinde değerleri güvenli 8-bit piksel aralığında tutar.
function clampByte(value: number) {
  return Math.round(Math.min(Math.max(value, 0), 255));
}
