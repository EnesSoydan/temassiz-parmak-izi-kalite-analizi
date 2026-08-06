import { getFingertipSegmentationRuntime } from '@/lib/onnx-model';

const MODEL_SIZE = 480;
const PROTOTYPE_SIZE = 120;
const MASK_COEFFICIENT_COUNT = 32;
const MIN_CONFIDENCE = 0.25;

type TensorLike = {
  data: ArrayLike<number>;
  dims: readonly number[];
};

type LetterboxInfo = {
  resizedWidth: number;
  resizedHeight: number;
  padLeft: number;
  padTop: number;
};

export type LearnedSegmentationMask = {
  mask: Uint8Array;
  coverage: number;
  confidence: number;
};

let hasReportedFailure = false;

/**
 * YOLO segmentation ONNX çıktısını mevcut ROI koordinat çerçevesine geri taşır.
 * Model tek sınıf ve tek parmak ROI'si için eğitildiği için en güvenilir adayı seçer.
 */
export async function inferLearnedFingerMask({
  pixels,
  width,
  height,
}: {
  pixels: Uint8Array;
  width: number;
  height: number;
}): Promise<LearnedSegmentationMask | null> {
  if (width < 2 || height < 2 || pixels.length < width * height * 4) {
    return null;
  }

  try {
    const { ort, session } = await getFingertipSegmentationRuntime();
    const { tensor, letterbox } = createLetterboxTensor(pixels, width, height);
    const inputName = session.inputNames[0] ?? 'images';
    const input = new ort.Tensor('float32', tensor, [
      1,
      3,
      MODEL_SIZE,
      MODEL_SIZE,
    ]);
    const outputs = await session.run({ [inputName]: input });
    const detectionOutput = findOutput(outputs, session.outputNames, 0);
    const prototypeOutput = findOutput(outputs, session.outputNames, 1);

    if (!detectionOutput || !prototypeOutput) {
      throw new Error('Segmentasyon ONNX çıktıları bulunamadı.');
    }

    const candidate = selectBestCandidate(detectionOutput);
    if (!candidate || candidate.confidence < MIN_CONFIDENCE) {
      return null;
    }

    const prototypeMask = createPrototypeMask(
      prototypeOutput,
      candidate.coefficients,
      candidate.box
    );
    const mask = restoreMaskToSource(
      prototypeMask,
      width,
      height,
      letterbox
    );
    const maskPixels = mask.reduce((sum, value) => sum + value, 0);

    return {
      mask,
      coverage: maskPixels / Math.max(mask.length, 1),
      confidence: candidate.confidence,
    };
  } catch (error) {
    if (!hasReportedFailure) {
      hasReportedFailure = true;
      console.warn(
        `[Segmentation ONNX] kullanılmadı; ten maskesine geri dönülüyor: ${String(error)}`
      );
    }
    return null;
  }
}

function createLetterboxTensor(
  pixels: Uint8Array,
  width: number,
  height: number
): { tensor: Float32Array; letterbox: LetterboxInfo } {
  const scale = Math.min(MODEL_SIZE / width, MODEL_SIZE / height);
  const resizedWidth = Math.max(1, Math.min(MODEL_SIZE, Math.round(width * scale)));
  const resizedHeight = Math.max(1, Math.min(MODEL_SIZE, Math.round(height * scale)));
  const padLeft = Math.floor((MODEL_SIZE - resizedWidth) / 2);
  const padTop = Math.floor((MODEL_SIZE - resizedHeight) / 2);
  const tensor = new Float32Array(3 * MODEL_SIZE * MODEL_SIZE);
  tensor.fill(114 / 255);
  const planeSize = MODEL_SIZE * MODEL_SIZE;
  const resizedScaleX = resizedWidth / width;
  const resizedScaleY = resizedHeight / height;

  for (let targetY = 0; targetY < resizedHeight; targetY += 1) {
    const sourceY = clamp(
      (targetY + 0.5) / resizedScaleY - 0.5,
      0,
      height - 1
    );
    const top = Math.floor(sourceY);
    const bottom = Math.min(top + 1, height - 1);
    const verticalWeight = sourceY - top;

    for (let targetX = 0; targetX < resizedWidth; targetX += 1) {
      const sourceX = clamp(
        (targetX + 0.5) / resizedScaleX - 0.5,
        0,
        width - 1
      );
      const left = Math.floor(sourceX);
      const right = Math.min(left + 1, width - 1);
      const horizontalWeight = sourceX - left;
      const topLeft = (top * width + left) * 4;
      const topRight = (top * width + right) * 4;
      const bottomLeft = (bottom * width + left) * 4;
      const bottomRight = (bottom * width + right) * 4;
      const canvasIndex = (padTop + targetY) * MODEL_SIZE + padLeft + targetX;
      const weights = [
        (1 - horizontalWeight) * (1 - verticalWeight),
        horizontalWeight * (1 - verticalWeight),
        (1 - horizontalWeight) * verticalWeight,
        horizontalWeight * verticalWeight,
      ];
      const sourceIndices = [topLeft, topRight, bottomLeft, bottomRight];

      for (let channel = 0; channel < 3; channel += 1) {
        let value = 0;
        for (let sample = 0; sample < sourceIndices.length; sample += 1) {
          value +=
            (pixels[sourceIndices[sample] + channel] ?? 0) * weights[sample];
        }
        tensor[channel * planeSize + canvasIndex] = value / 255;
      }
    }
  }

  return {
    tensor,
    letterbox: {
      resizedWidth,
      resizedHeight,
      padLeft,
      padTop,
    },
  };
}

function findOutput(
  outputs: Record<string, unknown>,
  outputNames: readonly string[],
  index: number
) {
  const outputName = outputNames[index];
  const namedOutput = outputName ? outputs[outputName] : undefined;
  const fallbackOutput = outputs[`output${index}`];
  const output = namedOutput ?? fallbackOutput;

  return isTensorLike(output) ? output : null;
}

function selectBestCandidate(output: TensorLike) {
  const [batch, channels, candidateCount] = output.dims;
  if (
    batch !== 1 ||
    channels < 5 + MASK_COEFFICIENT_COUNT ||
    !candidateCount ||
    output.data.length < channels * candidateCount
  ) {
    return null;
  }

  let bestIndex = -1;
  let bestConfidence = -Infinity;
  for (let candidateIndex = 0; candidateIndex < candidateCount; candidateIndex += 1) {
    const confidence = Number(output.data[4 * candidateCount + candidateIndex] ?? 0);
    if (confidence > bestConfidence) {
      bestConfidence = confidence;
      bestIndex = candidateIndex;
    }
  }

  if (bestIndex < 0) {
    return null;
  }

  const readChannel = (channel: number) =>
    Number(output.data[channel * candidateCount + bestIndex] ?? 0);
  const centerX = readChannel(0);
  const centerY = readChannel(1);
  const boxWidth = Math.max(1, readChannel(2));
  const boxHeight = Math.max(1, readChannel(3));
  const coefficients = new Float32Array(MASK_COEFFICIENT_COUNT);

  for (
    let coefficientIndex = 0;
    coefficientIndex < MASK_COEFFICIENT_COUNT;
    coefficientIndex += 1
  ) {
    coefficients[coefficientIndex] = readChannel(5 + coefficientIndex);
  }

  return {
    confidence: bestConfidence,
    coefficients,
    box: {
      x1: clamp(centerX - boxWidth / 2, 0, MODEL_SIZE - 1),
      y1: clamp(centerY - boxHeight / 2, 0, MODEL_SIZE - 1),
      x2: clamp(centerX + boxWidth / 2, 0, MODEL_SIZE - 1),
      y2: clamp(centerY + boxHeight / 2, 0, MODEL_SIZE - 1),
    },
  };
}

function createPrototypeMask(
  output: TensorLike,
  coefficients: Float32Array,
  box: { x1: number; y1: number; x2: number; y2: number }
) {
  const [batch, channels, prototypeHeight, prototypeWidth] = output.dims;
  if (
    batch !== 1 ||
    channels !== MASK_COEFFICIENT_COUNT ||
    prototypeWidth !== PROTOTYPE_SIZE ||
    prototypeHeight !== PROTOTYPE_SIZE ||
    output.data.length < channels * prototypeWidth * prototypeHeight
  ) {
    throw new Error('Segmentasyon prototip boyutu beklenen 120x120 değil.');
  }

  const prototypeMask = new Float32Array(PROTOTYPE_SIZE * PROTOTYPE_SIZE);
  const boxX1 = (box.x1 / MODEL_SIZE) * PROTOTYPE_SIZE;
  const boxY1 = (box.y1 / MODEL_SIZE) * PROTOTYPE_SIZE;
  const boxX2 = (box.x2 / MODEL_SIZE) * PROTOTYPE_SIZE;
  const boxY2 = (box.y2 / MODEL_SIZE) * PROTOTYPE_SIZE;
  const prototypePlaneSize = PROTOTYPE_SIZE * PROTOTYPE_SIZE;

  for (let y = 0; y < PROTOTYPE_SIZE; y += 1) {
    for (let x = 0; x < PROTOTYPE_SIZE; x += 1) {
      const index = y * PROTOTYPE_SIZE + x;
      if (x < boxX1 || x > boxX2 || y < boxY1 || y > boxY2) {
        continue;
      }

      let logit = 0;
      for (
        let coefficientIndex = 0;
        coefficientIndex < MASK_COEFFICIENT_COUNT;
        coefficientIndex += 1
      ) {
        const prototypeIndex = coefficientIndex * prototypePlaneSize + index;
        logit +=
          coefficients[coefficientIndex] * Number(output.data[prototypeIndex] ?? 0);
      }
      prototypeMask[index] = sigmoid(logit);
    }
  }

  return prototypeMask;
}

function restoreMaskToSource(
  prototypeMask: Float32Array,
  width: number,
  height: number,
  letterbox: LetterboxInfo
) {
  const mask = new Uint8Array(width * height);
  const resizedScaleX = letterbox.resizedWidth / width;
  const resizedScaleY = letterbox.resizedHeight / height;

  for (let y = 0; y < height; y += 1) {
    const modelY = letterbox.padTop + (y + 0.5) * resizedScaleY - 0.5;
    const prototypeY = (modelY / (MODEL_SIZE - 1)) * (PROTOTYPE_SIZE - 1);

    for (let x = 0; x < width; x += 1) {
      const modelX = letterbox.padLeft + (x + 0.5) * resizedScaleX - 0.5;
      const prototypeX = (modelX / (MODEL_SIZE - 1)) * (PROTOTYPE_SIZE - 1);
      const probability = sampleFloatBilinear(
        prototypeMask,
        PROTOTYPE_SIZE,
        PROTOTYPE_SIZE,
        prototypeX,
        prototypeY
      );
      mask[y * width + x] = probability >= 0.5 ? 1 : 0;
    }
  }

  return mask;
}

function sampleFloatBilinear(
  pixels: Float32Array,
  width: number,
  height: number,
  x: number,
  y: number
) {
  const sampleX = clamp(x, 0, width - 1);
  const sampleY = clamp(y, 0, height - 1);
  const left = Math.floor(sampleX);
  const top = Math.floor(sampleY);
  const right = Math.min(left + 1, width - 1);
  const bottom = Math.min(top + 1, height - 1);
  const horizontalWeight = sampleX - left;
  const verticalWeight = sampleY - top;
  const topValue =
    pixels[top * width + left] * (1 - horizontalWeight) +
    pixels[top * width + right] * horizontalWeight;
  const bottomValue =
    pixels[bottom * width + left] * (1 - horizontalWeight) +
    pixels[bottom * width + right] * horizontalWeight;

  return topValue * (1 - verticalWeight) + bottomValue * verticalWeight;
}

function isTensorLike(value: unknown): value is TensorLike {
  if (!value || typeof value !== 'object') {
    return false;
  }

  const tensor = value as Partial<TensorLike>;
  return Array.isArray(tensor.dims) && tensor.data != null;
}

function sigmoid(value: number) {
  if (value >= 0) {
    const exponent = Math.exp(-value);
    return 1 / (1 + exponent);
  }

  const exponent = Math.exp(value);
  return exponent / (1 + exponent);
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(Math.max(value, minimum), maximum);
}
