import { decode } from 'jpeg-js';

import type { QualityMetrics } from '@/types/biometrics';

type EstimateQualityInput = {
  imageWidth: number;
  imageHeight: number;
  roiWidth: number;
  roiHeight: number;
};

type AnalyzeFrameInput = {
  base64: string;
};

const ROI_WIDTH_RATIO = 0.42;
const ROI_ASPECT_RATIO = 0.62;

export function estimateInitialQuality({
  imageWidth,
  imageHeight,
  roiWidth,
  roiHeight,
}: EstimateQualityInput): QualityMetrics {
  const roiCoverage = (roiWidth * roiHeight) / Math.max(imageWidth * imageHeight, 1);

  return {
    blurScore: 0,
    glareRatio: 0,
    brightnessMean: 0,
    roiCoverage,
    status: 'unknown',
  };
}

export function analyzeFrameQuality({ base64 }: AnalyzeFrameInput): QualityMetrics {
  const image = decode(base64ToBytes(base64), { useTArray: true });
  const roi = getCenteredRoi(image.width, image.height);
  const step = Math.max(1, Math.floor(Math.min(roi.width, roi.height) / 96));
  const grayWidth = Math.ceil(roi.width / step);
  const grayHeight = Math.ceil(roi.height / step);
  const gray = new Float32Array(grayWidth * grayHeight);

  let index = 0;
  let brightPixels = 0;
  let totalBrightness = 0;

  for (let y = roi.originY; y < roi.originY + roi.height; y += step) {
    for (let x = roi.originX; x < roi.originX + roi.width; x += step) {
      const pixelIndex = (y * image.width + x) * 4;
      const red = image.data[pixelIndex] ?? 0;
      const green = image.data[pixelIndex + 1] ?? 0;
      const blue = image.data[pixelIndex + 2] ?? 0;
      const value = 0.299 * red + 0.587 * green + 0.114 * blue;

      gray[index] = value;
      totalBrightness += value;

      if (red > 245 && green > 245 && blue > 245) {
        brightPixels += 1;
      }

      index += 1;
    }
  }

  const sampleCount = Math.max(index, 1);
  const brightnessMean = Math.round(totalBrightness / sampleCount);
  const glareRatio = brightPixels / sampleCount;
  const blurScore = calculateSharpnessScore(gray, grayWidth, grayHeight);
  const roiCoverage = (roi.width * roi.height) / Math.max(image.width * image.height, 1);

  return {
    blurScore,
    glareRatio,
    brightnessMean,
    roiCoverage,
    status: getQualityStatus({ blurScore, glareRatio, brightnessMean }),
  };
}

export function formatQualityStatus(status: QualityMetrics['status']) {
  if (status === 'good') {
    return 'uygun';
  }

  if (status === 'usable') {
    return 'kullanılabilir';
  }

  if (status === 'poor') {
    return 'düşük';
  }

  return 'ölçülmedi';
}

function getCenteredRoi(imageWidth: number, imageHeight: number) {
  const width = Math.round(imageWidth * ROI_WIDTH_RATIO);
  const height = Math.min(Math.round(width / ROI_ASPECT_RATIO), Math.round(imageHeight * 0.82));

  return {
    originX: Math.max(0, Math.round((imageWidth - width) / 2)),
    originY: Math.max(0, Math.round((imageHeight - height) / 2)),
    width,
    height,
  };
}

function calculateSharpnessScore(gray: Float32Array, width: number, height: number) {
  if (width < 3 || height < 3) {
    return 0;
  }

  let sum = 0;
  let sumSquares = 0;
  let count = 0;

  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const center = gray[y * width + x] ?? 0;
      const laplacian =
        -4 * center +
        (gray[y * width + x - 1] ?? 0) +
        (gray[y * width + x + 1] ?? 0) +
        (gray[(y - 1) * width + x] ?? 0) +
        (gray[(y + 1) * width + x] ?? 0);

      sum += laplacian;
      sumSquares += laplacian * laplacian;
      count += 1;
    }
  }

  const mean = sum / Math.max(count, 1);
  const variance = sumSquares / Math.max(count, 1) - mean * mean;

  return Math.max(0, Math.min(100, Math.round(variance / 18)));
}

function getQualityStatus({
  blurScore,
  glareRatio,
  brightnessMean,
}: Pick<QualityMetrics, 'blurScore' | 'glareRatio' | 'brightnessMean'>): QualityMetrics['status'] {
  if (brightnessMean < 45 || brightnessMean > 225 || glareRatio > 0.08 || blurScore < 18) {
    return 'poor';
  }

  if (glareRatio > 0.035 || blurScore < 35) {
    return 'usable';
  }

  return 'good';
}

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

    if (byteIndex < length) {
      bytes[byteIndex] = (encoded >> 16) & 255;
      byteIndex += 1;
    }

    if (byteIndex < length) {
      bytes[byteIndex] = (encoded >> 8) & 255;
      byteIndex += 1;
    }

    if (byteIndex < length) {
      bytes[byteIndex] = encoded & 255;
      byteIndex += 1;
    }
  }

  return bytes;
}
