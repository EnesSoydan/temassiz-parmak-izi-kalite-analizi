import * as FileSystem from 'expo-file-system/legacy';
import { Buffer } from 'buffer';
import { encode } from 'jpeg-js';

import type { RidgeFrequencyBlock } from '@/lib/ridge-frequency';

type RidgeEnhancementInput = {
  grayscale: Uint8Array;
  mask: Uint8Array;
  width: number;
  height: number;
  frequencyBlocks: RidgeFrequencyBlock[];
};

// Minutiae için yalnızca Gabor katkısının blok geçişlerinden yeterince uzakta olduğu alanı kabul ederiz.
const MIN_MINUTIAE_ENHANCEMENT_WEIGHT = 0.08;

// Yalnızca ham veride doğrulanmış ridge bloklarında yön ve periyoda uyarlanmış Gabor yanıtı üretir.
export function createControlledRidgeEnhancement({
  grayscale,
  mask,
  width,
  height,
  frequencyBlocks,
}: RidgeEnhancementInput) {
  const enhancedValueTotal = new Float32Array(mask.length);
  const enhancementWeightTotal = new Float32Array(mask.length);

  for (const block of frequencyBlocks) {
    if (!block.valid || block.periodPixels <= 0) continue;
    applyDirectionalGabor(
      grayscale,
      enhancedValueTotal,
      enhancementWeightTotal,
      mask,
      width,
      height,
      block
    );
  }

  const pixels = new Uint8Array(mask.length * 4);
  const enhancedGrayscale = new Uint8Array(mask.length);
  const minutiaeSupportMask = new Uint8Array(mask.length);
  let supportedPixelCount = 0;
  let foregroundPixelCount = 0;

  for (let index = 0; index < mask.length; index += 1) {
    const outputIndex = index * 4;
    const isForeground = mask[index] !== 0;
    const enhancementWeight = enhancementWeightTotal[index];
    const enhancedValue =
      enhancementWeight > 0
        ? enhancedValueTotal[index] / enhancementWeight
        : grayscale[index];
    // Filtrelenmiş alanı ham normalize görüntüye yumuşakça karıştırarak kare blok sınırlarını gizler.
    const blendStrength = Math.min(enhancementWeight, 0.88);
    const value = isForeground
      ? clampByte(
          grayscale[index] * (1 - blendStrength) +
            enhancedValue * blendStrength
        )
      : 0;
    if (isForeground) foregroundPixelCount += 1;
    if (enhancementWeight > 0.08) supportedPixelCount += 1;
    if (
      isForeground &&
      enhancementWeight >= MIN_MINUTIAE_ENHANCEMENT_WEIGHT
    ) {
      minutiaeSupportMask[index] = 1;
    }
    enhancedGrayscale[index] = value;
    pixels[outputIndex] = value;
    pixels[outputIndex + 1] = value;
    pixels[outputIndex + 2] = value;
    pixels[outputIndex + 3] = 255;
  }

  const contrastMeasurement = measureSupportedRidgeContrast({
    source: grayscale,
    enhanced: enhancedGrayscale,
    mask,
    enhancementWeightTotal,
    width,
    height,
    frequencyBlocks,
  });

  return {
    pixels,
    supportedAreaRatio: supportedPixelCount / Math.max(foregroundPixelCount, 1),
    contrastBefore: contrastMeasurement.before,
    contrastAfter: contrastMeasurement.after,
    contrastGainPercent: contrastMeasurement.gainPercent,
    minutiaeSupportMask,
  };
}

// Doğrulanmış bloğun çizgilerine dik yönde gerçek piksel değişimlerini süzen bir Gabor çekirdeği uygular.
function applyDirectionalGabor(
  source: Uint8Array,
  enhancedValueTotal: Float32Array,
  enhancementWeightTotal: Float32Array,
  mask: Uint8Array,
  width: number,
  height: number,
  block: RidgeFrequencyBlock
) {
  const tangentX = Math.cos(block.angleRadians);
  const tangentY = Math.sin(block.angleRadians);
  const normalX = -Math.sin(block.angleRadians);
  const normalY = Math.cos(block.angleRadians);
  const normalRadius = Math.max(
    3,
    Math.min(7, Math.round(block.periodPixels * 0.8))
  );
  const tangentRadius = Math.max(2, Math.min(4, Math.round(block.size * 0.2)));
  const sigmaNormal = Math.max(block.periodPixels * 0.55, 2);
  const sigmaTangent = Math.max(block.size * 0.28, 2);
  const kernel: {
    normalOffset: number;
    tangentOffset: number;
    weight: number;
  }[] = [];
  const tangentOffsets = [-tangentRadius, 0, tangentRadius];
  let weightTotal = 0;

  for (
    let normalOffset = -normalRadius;
    normalOffset <= normalRadius;
    normalOffset += 1
  ) {
    for (const tangentOffset of tangentOffsets) {
      const gaussian = Math.exp(
        -(normalOffset * normalOffset) /
          (2 * sigmaNormal * sigmaNormal) -
          (tangentOffset * tangentOffset) /
            (2 * sigmaTangent * sigmaTangent)
      );
      const wave = Math.cos(
        (Math.PI * 2 * normalOffset) / block.periodPixels
      );
      const weight = gaussian * wave;
      kernel.push({ normalOffset, tangentOffset, weight });
      weightTotal += weight;
    }
  }

  // Sabit parlaklığın filtre yanıtına dönüşmemesi için çekirdek ortalamasını sıfırlarız.
  const meanWeight = weightTotal / Math.max(kernel.length, 1);
  let absoluteWeightTotal = 0;
  for (const sample of kernel) {
    sample.weight -= meanWeight;
    absoluteWeightTotal += Math.abs(sample.weight);
  }

  const right = Math.min(block.left + block.size, width);
  const bottom = Math.min(block.top + block.size, height);
  const confidenceWeight = Math.max(0.2, Math.min(block.periodicity / 100, 1));

  for (let y = Math.max(block.top, 0); y < bottom; y += 1) {
    for (let x = Math.max(block.left, 0); x < right; x += 1) {
      const index = y * width + x;
      if (!mask[index]) continue;

      let response = 0;
      let usedWeight = 0;

      for (const sample of kernel) {
        const sampleX = Math.round(
          x +
            normalX * sample.normalOffset +
            tangentX * sample.tangentOffset
        );
        const sampleY = Math.round(
          y +
            normalY * sample.normalOffset +
            tangentY * sample.tangentOffset
        );
        if (sampleX < 0 || sampleX >= width || sampleY < 0 || sampleY >= height) continue;
        const sampleIndex = sampleY * width + sampleX;
        if (!mask[sampleIndex]) continue;

        response += source[sampleIndex] * sample.weight;
        usedWeight += Math.abs(sample.weight);
      }

      if (usedWeight < absoluteWeightTotal * 0.68) continue;
      const rawEnhancedValue =
        128 + (response / Math.max(usedWeight, 0.0001)) * 3;
      // Gabor çıktısının kamerada bulunmayan aşırı açık/koyu çizgiler üretmesini sınırlarız.
      const enhancedValue = clamp(
        rawEnhancedValue,
        source[index] - 64,
        source[index] + 64
      );
      const windowWeight =
        createRaisedCosineWeight(x, block.left, block.size) *
        createRaisedCosineWeight(y, block.top, block.size);
      const contributionWeight = windowWeight * confidenceWeight;

      if (contributionWeight <= 0.001) continue;
      enhancedValueTotal[index] += enhancedValue * contributionWeight;
      enhancementWeightTotal[index] += contributionWeight;
    }
  }
}

// Yalnızca doğrulanmış ve gerçekten filtrelenmiş bölgelerde ridge-valley kontrast değişimini ölçer.
function measureSupportedRidgeContrast({
  source,
  enhanced,
  mask,
  enhancementWeightTotal,
  width,
  height,
  frequencyBlocks,
}: {
  source: Uint8Array;
  enhanced: Uint8Array;
  mask: Uint8Array;
  enhancementWeightTotal: Float32Array;
  width: number;
  height: number;
  frequencyBlocks: RidgeFrequencyBlock[];
}) {
  const measured = new Uint8Array(mask.length);
  let beforeTotal = 0;
  let afterTotal = 0;
  let sampleCount = 0;

  for (const block of frequencyBlocks) {
    if (!block.valid || block.periodPixels <= 0) continue;
    const normalX = -Math.sin(block.angleRadians);
    const normalY = Math.cos(block.angleRadians);
    const halfPeriod = Math.max(2, Math.round(block.periodPixels / 2));
    const right = Math.min(block.left + block.size, width);
    const bottom = Math.min(block.top + block.size, height);

    for (let y = Math.max(block.top, 0); y < bottom; y += 1) {
      for (let x = Math.max(block.left, 0); x < right; x += 1) {
        const index = y * width + x;
        if (
          measured[index] ||
          !mask[index] ||
          enhancementWeightTotal[index] <= 0.08
        ) {
          continue;
        }

        const firstX = Math.round(x - normalX * halfPeriod);
        const firstY = Math.round(y - normalY * halfPeriod);
        const secondX = Math.round(x + normalX * halfPeriod);
        const secondY = Math.round(y + normalY * halfPeriod);
        if (
          firstX < 0 ||
          firstX >= width ||
          firstY < 0 ||
          firstY >= height ||
          secondX < 0 ||
          secondX >= width ||
          secondY < 0 ||
          secondY >= height
        ) {
          continue;
        }

        const firstIndex = firstY * width + firstX;
        const secondIndex = secondY * width + secondX;
        if (!mask[firstIndex] || !mask[secondIndex]) continue;

        beforeTotal += Math.abs(
          source[index] - (source[firstIndex] + source[secondIndex]) / 2
        );
        afterTotal += Math.abs(
          enhanced[index] -
            (enhanced[firstIndex] + enhanced[secondIndex]) / 2
        );
        measured[index] = 1;
        sampleCount += 1;
      }
    }
  }

  const before = beforeTotal / Math.max(sampleCount, 1);
  const after = afterTotal / Math.max(sampleCount, 1);
  const gainPercent =
    sampleCount > 0
      ? clamp(((after - before) / Math.max(before, 1)) * 100, -100, 300)
      : 0;

  return {
    before: roundToOneDecimal(before),
    after: roundToOneDecimal(after),
    gainPercent: roundToOneDecimal(gainPercent),
  };
}

// Blok merkezini güçlü, kenarlarını sıfır yapan pencere ile komşu blok geçişlerini yumuşatır.
function createRaisedCosineWeight(position: number, start: number, size: number) {
  const normalized = Math.min(Math.max((position - start + 0.5) / Math.max(size, 1), 0), 1);
  return Math.sin(Math.PI * normalized) ** 2;
}

// RGBA enhancement çıktısını kalıcı JPEG dosyasına yazar.
export async function saveRidgeEnhancedImage({
  pixels,
  width,
  height,
  outputImageUri,
  maximumWidth = 256,
  quality = 88,
}: {
  pixels: Uint8Array;
  width: number;
  height: number;
  outputImageUri: string;
  maximumWidth?: number;
  quality?: number;
}) {
  ensureJpegBufferShim();

  // Teknik galeri görsellerini küçültür; kalite ve minutiae hesapları bundan önce tam analiz tamponunda yapılmıştır.
  const preview = resizeTechnicalPreview(pixels, width, height, maximumWidth);
  const jpeg = encode(
    { data: preview.pixels, width: preview.width, height: preview.height },
    quality
  );
  await FileSystem.writeAsStringAsync(outputImageUri, bytesToBase64(jpeg.data), {
    encoding: FileSystem.EncodingType.Base64,
  });
  return outputImageUri;
}

// jpeg-js React Native ortamında global Buffer beklediği için kodlamadan önce polyfill'i hazırlar.
function ensureJpegBufferShim() {
  const globalScope = globalThis as typeof globalThis & { Buffer?: typeof Buffer };
  if (globalScope.Buffer) return;
  globalScope.Buffer = Buffer;
}

// RGBA teknik çıktıyı en-boy oranını koruyarak en fazla hedef genişliğe örnekler.
function resizeTechnicalPreview(
  pixels: Uint8Array,
  width: number,
  height: number,
  maximumWidth: number
) {
  if (width <= maximumWidth) return { pixels, width, height };
  const targetWidth = maximumWidth;
  const targetHeight = Math.max(1, Math.round((height * targetWidth) / width));
  const resized = new Uint8Array(targetWidth * targetHeight * 4);

  for (let y = 0; y < targetHeight; y += 1) {
    const sourceY = Math.min(Math.floor((y * height) / targetHeight), height - 1);
    for (let x = 0; x < targetWidth; x += 1) {
      const sourceX = Math.min(Math.floor((x * width) / targetWidth), width - 1);
      const sourceIndex = (sourceY * width + sourceX) * 4;
      const targetIndex = (y * targetWidth + x) * 4;
      resized[targetIndex] = pixels[sourceIndex];
      resized[targetIndex + 1] = pixels[sourceIndex + 1];
      resized[targetIndex + 2] = pixels[sourceIndex + 2];
      resized[targetIndex + 3] = pixels[sourceIndex + 3];
    }
  }

  return { pixels: resized, width: targetWidth, height: targetHeight };
}

// jpeg-js çıktısını React Native ve Node ortamlarında güvenli base64 metnine çevirir.
function bytesToBase64(bytes: Uint8Array) {
  return Buffer.from(bytes).toString('base64');
}

// Filtre yanıtını geçerli 8-bit gri ton aralığında tutar.
function clampByte(value: number) {
  return Math.round(Math.min(Math.max(value, 0), 255));
}

// Sayısal filtre çıktısını verilen güvenli alt ve üst sınır arasında tutar.
function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(Math.max(value, minimum), maximum);
}

// Tanılama metriklerini okunabilir tek ondalık hassasiyette tutar.
function roundToOneDecimal(value: number) {
  return Math.round(value * 10) / 10;
}
