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
  silhouetteAxisDegrees: number;
  timings: {
    decodeMs: number;
    maskMs: number;
    enhancementMs: number;
    encodeMs: number;
    writeMs: number;
  };
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

// Sıkı OBB kırpımlarında parmak ROI'nin çoğunu kaplayabilir; yalnızca neredeyse tam taşmayı reddederiz.
const MIN_MASK_COVERAGE = 0.12;
const MAX_MASK_COVERAGE = 0.995;

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

  const decodeStartedAt = Date.now();
  const image = await loadAnalysisImage(roiImageUri, sourceWidth);
  const decodeMs = Date.now() - decodeStartedAt;
  const sourcePixels = new Uint8Array(image.data);
  const maskStartedAt = Date.now();
  const targetColor = sampleCenterSkinColor(sourcePixels, image.width, image.height);
  const skinMask = cleanAndSelectMask(
    createSkinMask(sourcePixels, image.width, image.height, targetColor),
    image.width,
    image.height
  );
  const skinCoverage = calculateMaskCoverage(skinMask, image.width, image.height);
  let mask = skinMask;
  let coverage = skinCoverage;
  let method = 'ten';

  // Sabit YCrCb ten aralığı beyaz dengesi nedeniyle kaçırırsa merkez rengine göre ikinci maske deneriz.
  if (!isValidMaskCoverage(skinCoverage)) {
    const adaptiveMask = cleanAndSelectMask(
      createAdaptiveCenterMask(sourcePixels, image.width, image.height, targetColor),
      image.width,
      image.height
    );
    const adaptiveCoverage = calculateMaskCoverage(adaptiveMask, image.width, image.height);

    if (isValidMaskCoverage(adaptiveCoverage)) {
      mask = adaptiveMask;
      coverage = adaptiveCoverage;
      method = 'uyarlanabilir';
    } else {
      console.info(
        `[ROI segmentasyon] sonuç=yok, ten=${formatCoverage(skinCoverage)}, uyarlanabilir=${formatCoverage(adaptiveCoverage)}`
      );
      return null;
    }
  }

  if (!isValidMaskCoverage(coverage)) {
    return null;
  }

  const silhouetteAxisDegrees = estimateMaskPrincipalAxis(mask, image.width, image.height);
  const maskMs = Date.now() - maskStartedAt;
  console.info(`[ROI segmentasyon] yöntem=${method}, kapsam=${formatCoverage(coverage)}`);
  const enhancementStartedAt = Date.now();
  const segmentedPixels = createEnhancedSegmentedPixels(sourcePixels, mask, image.width, image.height);
  const enhancementMs = Date.now() - enhancementStartedAt;

  const encodeStartedAt = Date.now();
  const jpeg = encode({ data: segmentedPixels, width: image.width, height: image.height }, 95);
  const encodeMs = Date.now() - encodeStartedAt;
  const writeStartedAt = Date.now();
  await FileSystem.writeAsStringAsync(outputImageUri, bytesToBase64(jpeg.data), {
    encoding: FileSystem.EncodingType.Base64,
  });
  const writeMs = Date.now() - writeStartedAt;

  return {
    imageUri: outputImageUri,
    pixels: sourcePixels,
    mask,
    width: image.width,
    height: image.height,
    coverage,
    silhouetteAxisDegrees,
    timings: {
      decodeMs,
      maskMs,
      enhancementMs,
      encodeMs,
      writeMs,
    },
  };
}

// Segmentasyon maskesinin ikinci momentlerinden parmağın 0-180 derece aralığındaki uzun eksenini ölçer.
function estimateMaskPrincipalAxis(
  mask: Uint8Array,
  width: number,
  height: number
) {
  let count = 0;
  let xTotal = 0;
  let yTotal = 0;

  for (let index = 0; index < mask.length; index += 1) {
    if (!mask[index]) continue;
    xTotal += index % width;
    yTotal += Math.floor(index / width);
    count += 1;
  }

  if (count < 12) return 90;
  const centerX = xTotal / count;
  const centerY = yTotal / count;
  let covarianceXX = 0;
  let covarianceXY = 0;
  let covarianceYY = 0;

  for (let index = 0; index < mask.length; index += 1) {
    if (!mask[index]) continue;
    const deltaX = (index % width) - centerX;
    const deltaY = Math.floor(index / width) - centerY;
    covarianceXX += deltaX * deltaX;
    covarianceXY += deltaX * deltaY;
    covarianceYY += deltaY * deltaY;
  }

  let axisDegrees =
    (0.5 * Math.atan2(2 * covarianceXY, covarianceXX - covarianceYY) * 180) /
    Math.PI;
  if (axisDegrees < 0) axisDegrees += 180;
  return Math.round(axisDegrees * 10) / 10;
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
  const stretched = new Uint8Array(grayscale.length);
  const { low, high } = getMaskedContrastBounds(grayscale, mask);
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
  const integral = createIntegralImage(grayscale, width, height);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const left = Math.max(0, x - 1);
      const top = Math.max(0, y - 1);
      const right = Math.min(width - 1, x + 1);
      const bottom = Math.min(height - 1, y + 1);
      const area = (right - left + 1) * (bottom - top + 1);
      const total = readIntegralSum(integral, width, left, top, right, bottom);

      blurred[y * width + x] = Math.round(total / area);
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

// Sabit ten aralığına girmeyen ışık koşullarında ROI merkezinin renk yakınlığıyla yedek maske üretir.
function createAdaptiveCenterMask(
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

      if (chromaDistance <= 52 && brightnessDistance <= 110) {
        mask[y * width + x] = 1;
      }
    }
  }

  return mask;
}

// İlk maskenin küçük boşluklarını kapatıp yalnızca merkeze yakın büyük bileşenini saklar.
function cleanAndSelectMask(mask: Uint8Array, width: number, height: number) {
  const closedMask = erode(dilate(mask, width, height, 2), width, height, 1);
  const cleanedMask = dilate(erode(closedMask, width, height, 1), width, height, 1);
  return keepCenterConnectedComponent(cleanedMask, width, height);
}

// Maske kapsamasını ROI piksel sayısına oranlar.
function calculateMaskCoverage(mask: Uint8Array, width: number, height: number) {
  return countMaskPixels(mask) / Math.max(width * height, 1);
}

// Boş, çok küçük veya görüntünün tamamına taşmış maskeleri geçersiz sayar.
function isValidMaskCoverage(coverage: number) {
  return coverage >= MIN_MASK_COVERAGE && coverage <= MAX_MASK_COVERAGE;
}

// Segmentasyon kalibrasyon loglarında kapsama oranını okunabilir yüzdeye dönüştürür.
function formatCoverage(coverage: number) {
  return `${Math.round(coverage * 1000) / 10}%`;
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
  const integral = createBinaryIntegralImage(mask, width, height);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const left = Math.max(0, x - radius);
      const top = Math.max(0, y - radius);
      const right = Math.min(width - 1, x + radius);
      const bottom = Math.min(height - 1, y + radius);
      const filledCount = readBinaryIntegralSum(integral, width, left, top, right, bottom);

      nextMask[y * width + x] = filledCount > 0 ? 1 : 0;
    }
  }

  return nextMask;
}

// Maskeyi daraltarak tekil arka plan gürültüsünü temizler.
function erode(mask: Uint8Array, width: number, height: number, radius: number) {
  const nextMask = new Uint8Array(mask.length);
  const integral = createBinaryIntegralImage(mask, width, height);
  const kernelWidth = radius * 2 + 1;
  const requiredCount = kernelWidth * kernelWidth;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const left = x - radius;
      const top = y - radius;
      const right = x + radius;
      const bottom = y + radius;

      // Önceki davranışta görüntü dışındaki komşular sıfır kabul edildiği için sınırı doğrudan boş bırakırız.
      if (left < 0 || top < 0 || right >= width || bottom >= height) {
        continue;
      }

      const filledCount = readBinaryIntegralSum(integral, width, left, top, right, bottom);
      nextMask[y * width + x] = filledCount === requiredCount ? 1 : 0;
    }
  }

  return nextMask;
}

// İkili maskenin dikdörtgen toplamlarını sabit zamanda okuyabilmek için integral görüntüsünü üretir.
function createBinaryIntegralImage(mask: Uint8Array, width: number, height: number) {
  const integral = new Int32Array((width + 1) * (height + 1));

  for (let y = 1; y <= height; y += 1) {
    let rowTotal = 0;

    for (let x = 1; x <= width; x += 1) {
      rowTotal += mask[(y - 1) * width + (x - 1)] ? 1 : 0;
      integral[y * (width + 1) + x] = integral[(y - 1) * (width + 1) + x] + rowTotal;
    }
  }

  return integral;
}

// İkili integral görüntüden verilen kapalı dikdörtgenin dolu piksel sayısını okur.
function readBinaryIntegralSum(
  integral: Int32Array,
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

// 8-bit histogram üzerinden sıralama yapmadan maskeli alanın %4-%96 kontrast sınırlarını bulur.
function getMaskedContrastBounds(grayscale: Uint8Array, mask: Uint8Array) {
  const histogram = new Uint32Array(256);
  let count = 0;

  for (let index = 0; index < grayscale.length; index += 1) {
    if (!mask[index]) continue;
    histogram[grayscale[index]] += 1;
    count += 1;
  }

  return {
    low: getHistogramPercentile(histogram, count, 0.04),
    high: getHistogramPercentile(histogram, count, 0.96),
  };
}

// Histogramda hedef yüzdelik sırasına ulaşan ilk 8-bit değeri döndürür.
function getHistogramPercentile(histogram: Uint32Array, count: number, percentile: number) {
  if (count === 0) return 0;
  const target = Math.floor((count - 1) * percentile);
  let cumulative = 0;

  for (let value = 0; value < histogram.length; value += 1) {
    cumulative += histogram[value];
    if (cumulative > target) return value;
  }

  return 255;
}

// Görüntü işlemlerinde değerleri güvenli 8-bit piksel aralığında tutar.
function clampByte(value: number) {
  return Math.round(Math.min(Math.max(value, 0), 255));
}
