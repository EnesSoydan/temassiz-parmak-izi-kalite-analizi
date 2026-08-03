import type { ExposurePairComparison } from '@/types/biometrics';

const ANALYSIS_WIDTH = 96;
const ANALYSIS_HEIGHT = 128;
const BLOCK_SIZE = 8;
const MIN_LOCAL_ALIGNMENT = 0.55;
const ALIGNMENT_ANGLES = [-6, -3, 0, 3, 6];
const ALIGNMENT_SCALES = [0.96, 1, 1.04];

type AlignmentTransform = {
  shiftX: number;
  shiftY: number;
  scale: number;
  rotationDegrees: number;
  confidence: number;
};

type NormalizedExposure = {
  grayscale: Uint8Array;
  highlightMask: Uint8Array;
  mask: Uint8Array;
  gradientX: Float32Array;
  gradientY: Float32Array;
  gradientMagnitude: Float32Array;
};

export type ExposureAnalysisInput = {
  pixels: Uint8Array;
  mask: Uint8Array;
  width: number;
  height: number;
};

export type ExposureProcessingResult = {
  comparison: ExposurePairComparison;
  pixels: Uint8Array;
  width: number;
  height: number;
  processingSource: 'ambient' | 'flash' | 'local';
  flashPixelRatio: number;
};

// İki kanonik ROI'yi ortak analiz ızgarasına taşıyıp dönüşüm adaylarını karşılaştırır.
export function compareExposurePair(
  ambient: ExposureAnalysisInput,
  flash: ExposureAnalysisInput
): ExposurePairComparison {
  const ambientNormalized = normalizeExposure(ambient);
  const flashNormalized = normalizeExposure(flash);
  const alignment = findBestAlignment(ambientNormalized, flashNormalized);

  let ambientBetterCount = 0;
  let flashBetterCount = 0;
  let comparableBlockCount = 0;
  let ambientGlarePixels = 0;
  let flashGlarePixels = 0;
  let comparedPixels = 0;
  let centerComparableCount = 0;
  let centerFlashBetterCount = 0;
  let outerComparableCount = 0;
  let outerAmbientBetterCount = 0;
  let ambientScoreTotal = 0;
  let flashScoreTotal = 0;

  for (let blockY = 0; blockY < ANALYSIS_HEIGHT; blockY += BLOCK_SIZE) {
    for (let blockX = 0; blockX < ANALYSIS_WIDTH; blockX += BLOCK_SIZE) {
      const metrics = compareBlock({
        ambient: ambientNormalized,
        flash: flashNormalized,
        blockX,
        blockY,
        alignment,
      });
      if (!metrics) continue;

      comparableBlockCount += 1;
      ambientScoreTotal += metrics.ambientScore;
      flashScoreTotal += metrics.flashScore;
      ambientGlarePixels += metrics.ambientGlarePixels;
      flashGlarePixels += metrics.flashGlarePixels;
      comparedPixels += metrics.pixelCount;
      if (metrics.ambientScore > metrics.flashScore + 5) ambientBetterCount += 1;
      if (metrics.flashScore > metrics.ambientScore + 5) flashBetterCount += 1;

      const isCenter =
        blockX >= ANALYSIS_WIDTH * 0.25 &&
        blockX + BLOCK_SIZE <= ANALYSIS_WIDTH * 0.75 &&
        blockY >= ANALYSIS_HEIGHT * 0.15 &&
        blockY + BLOCK_SIZE <= ANALYSIS_HEIGHT * 0.75;
      if (isCenter) {
        centerComparableCount += 1;
        if (metrics.flashScore > metrics.ambientScore + 5) {
          centerFlashBetterCount += 1;
        }
      } else {
        outerComparableCount += 1;
        if (metrics.ambientScore > metrics.flashScore + 5) {
          outerAmbientBetterCount += 1;
        }
      }
    }
  }

  const ambientBetterBlockRatio = ratio(ambientBetterCount, comparableBlockCount);
  const flashBetterBlockRatio = ratio(flashBetterCount, comparableBlockCount);
  const alignmentConfidence = Math.round(alignment.confidence * 100);
  let recommendation: ExposurePairComparison['recommendation'] = 'ambient';

  if (alignmentConfidence < 30 || comparableBlockCount < 8) {
    recommendation = 'unaligned';
  } else if (ambientBetterBlockRatio >= 20 && flashBetterBlockRatio >= 20) {
    recommendation = 'complementary';
  } else if (flashBetterBlockRatio > ambientBetterBlockRatio + 10) {
    recommendation = 'flash';
  }

  return {
    alignmentConfidence,
    ambientBetterBlockRatio,
    flashBetterBlockRatio,
    comparableBlockCount,
    ambientGlareRatio: ratio(ambientGlarePixels, comparedPixels),
    flashGlareRatio: ratio(flashGlarePixels, comparedPixels),
    centerFlashBetterBlockRatio: ratio(
      centerFlashBetterCount,
      centerComparableCount
    ),
    outerAmbientBetterBlockRatio: ratio(
      outerAmbientBetterCount,
      outerComparableCount
    ),
    ambientMeanScore: Math.round(
      ambientScoreTotal / Math.max(comparableBlockCount, 1)
    ),
    flashMeanScore: Math.round(
      flashScoreTotal / Math.max(comparableBlockCount, 1)
    ),
    alignmentShiftX: alignment.shiftX,
    alignmentShiftY: alignment.shiftY,
    alignmentScale: alignment.scale,
    alignmentRotationDegrees: alignment.rotationDegrees,
    recommendation,
  };
}

// Güvenilir hizalamada her blok için daha güçlü poz kaynağını seçer; birleşik dosya üretmez.
export function selectExposureProcessingSource(
  ambient: ExposureAnalysisInput,
  flash: ExposureAnalysisInput,
  existingComparison?: ExposurePairComparison
): ExposureProcessingResult {
  if (ambient.width !== flash.width || ambient.height !== flash.height) {
    throw new Error('Poz seçimi girdileri aynı boyutta olmalıdır.');
  }

  const comparison = existingComparison ?? compareExposurePair(ambient, flash);
  if (comparison.alignmentConfidence < MIN_LOCAL_ALIGNMENT * 100) {
    const useFlash = shouldUseWholeFlash(comparison);
    return createWholeSourceResult(
      useFlash ? flash : ambient,
      { ...comparison, processingSource: useFlash ? 'flash' : 'ambient' },
      useFlash ? 'flash' : 'ambient',
      useFlash ? 100 : 0
    );
  }

  if (comparison.recommendation === 'flash' && shouldUseWholeFlash(comparison)) {
    return createWholeSourceResult(
      flash,
      { ...comparison, processingSource: 'flash' },
      'flash',
      100
    );
  }

  if (comparison.recommendation === 'ambient') {
    return createWholeSourceResult(
      ambient,
      { ...comparison, processingSource: 'ambient' },
      'ambient',
      0
    );
  }

  const ambientNormalized = normalizeExposure(ambient);
  const flashNormalized = normalizeExposure(flash);
  const blockChoices = createLocalFlashChoices(
    ambientNormalized,
    flashNormalized,
    comparison
  );
  const hasFlashBlocks = blockChoices.some((choice) => choice === 1);
  const hasAmbientBlocks = blockChoices.some((choice) => choice === 0);

  if (!hasFlashBlocks || !hasAmbientBlocks) {
    const useFlash = hasFlashBlocks;
    return createWholeSourceResult(
      useFlash ? flash : ambient,
      { ...comparison, processingSource: useFlash ? 'flash' : 'ambient' },
      useFlash ? 'flash' : 'ambient',
      useFlash ? 100 : 0
    );
  }

  const pixels = new Uint8Array(ambient.pixels.length);
  let selectedFlashPixelCount = 0;
  let selectedPixelCount = 0;

  for (let y = 0; y < ambient.height; y += 1) {
    for (let x = 0; x < ambient.width; x += 1) {
      const pixelIndex = y * ambient.width + x;
      const dataIndex = pixelIndex * 4;
      const analysisPoint = mapOutputPixelToAnalysis(
        x,
        y,
        ambient.width,
        ambient.height
      );
      const flashPoint = mapAmbientToFlash(
        analysisPoint.x,
        analysisPoint.y,
        comparison
      );
      const flashX = Math.round(
        ((flashPoint.x + 0.5) * flash.width) / ANALYSIS_WIDTH - 0.5
      );
      const flashY = Math.round(
        ((flashPoint.y + 0.5) * flash.height) / ANALYSIS_HEIGHT - 0.5
      );
      const flashInBounds =
        flashX >= 0 && flashX < flash.width && flashY >= 0 && flashY < flash.height;
      const flashIndex = flashY * flash.width + flashX;
      const canUseFlash =
        flashInBounds && ambient.mask[pixelIndex] > 0 && flash.mask[flashIndex] > 0;
      const flashChoice = canUseFlash
        ? sampleFlashChoice(blockChoices, analysisPoint.x, analysisPoint.y)
        : 0;
      const flashDataIndex = flashIndex * 4;
      const useFlash = canUseFlash && flashChoice === 1;

      for (let channel = 0; channel < 3; channel += 1) {
        pixels[dataIndex + channel] = useFlash
          ? flash.pixels[flashDataIndex + channel]
          : ambient.pixels[dataIndex + channel];
      }
      pixels[dataIndex + 3] = 255;
      if (useFlash) selectedFlashPixelCount += 1;
      selectedPixelCount += 1;
    }
  }

  const flashPixelRatio = Math.round(
    (selectedFlashPixelCount / Math.max(selectedPixelCount, 1)) * 100
  );
  return {
    comparison: { ...comparison, processingSource: 'local' },
    pixels,
    width: ambient.width,
    height: ambient.height,
    processingSource: 'local',
    flashPixelRatio,
  };
}

// Hizalama başarısızsa bir kaynağın tamamını seçerek yanlış eşleşme üretmez.
function createWholeSourceResult(
  source: ExposureAnalysisInput,
  comparison: ExposurePairComparison,
  processingSource: 'ambient' | 'flash',
  flashPixelRatio: number
): ExposureProcessingResult {
  return {
    comparison: { ...comparison, processingSource },
    pixels: new Uint8Array(source.pixels),
    width: source.width,
    height: source.height,
    processingSource,
    flashPixelRatio,
  };
}

// Her analiz bloğunda flaşın yerel kalite puanına göre katkısını belirler.
function createLocalFlashChoices(
  ambient: NormalizedExposure,
  flash: NormalizedExposure,
  comparison: ExposurePairComparison
) {
  const alignment = comparisonToAlignment(comparison);
  const columns = ANALYSIS_WIDTH / BLOCK_SIZE;
  const rows = ANALYSIS_HEIGHT / BLOCK_SIZE;
  const choices = new Uint8Array(columns * rows);
  const preferFlashGlobally = shouldUseWholeFlash(comparison);

  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const metrics = compareBlock({
        ambient,
        flash,
        blockX: column * BLOCK_SIZE,
        blockY: row * BLOCK_SIZE,
        alignment,
      });
      if (!metrics) continue;

      const scoreDifference = metrics.flashScore - metrics.ambientScore;
      const flashGlareRatio = metrics.flashGlarePixels / Math.max(metrics.pixelCount, 1);
      const ambientGlareRatio = metrics.ambientGlarePixels / Math.max(metrics.pixelCount, 1);
      const index = row * columns + column;

      if (flashGlareRatio > ambientGlareRatio + 0.08) {
        choices[index] = 0;
      } else if (scoreDifference >= 4) {
        choices[index] = 1;
      } else if (scoreDifference <= -4) {
        choices[index] = 0;
      } else {
        choices[index] = preferFlashGlobally ? 1 : 0;
      }
    }
  }

  return choices;
}

// Blok geçişlerini sertleştirmemek için komşu kaynak ağırlıklarını iki doğrusal örnekler.
function sampleFlashChoice(choices: Uint8Array, analysisX: number, analysisY: number) {
  const columns = ANALYSIS_WIDTH / BLOCK_SIZE;
  const rows = ANALYSIS_HEIGHT / BLOCK_SIZE;
  const column = clamp(Math.floor(analysisX / BLOCK_SIZE), 0, columns - 1);
  const row = clamp(Math.floor(analysisY / BLOCK_SIZE), 0, rows - 1);
  return choices[row * columns + column] ?? 0;
}

// Hizalama zayÄ±f olsa bile daha net ve parlama yapmayan tÃ¼m kaynaÄŸÄ± seÃ§er.
function shouldUseWholeFlash(comparison: ExposurePairComparison) {
  const flashNotMoreGlared =
    comparison.flashGlareRatio <= comparison.ambientGlareRatio + 12;
  const flashHasHigherMean =
    comparison.flashMeanScore >= comparison.ambientMeanScore + 3;
  const flashWinsMoreBlocks =
    comparison.flashBetterBlockRatio >= comparison.ambientBetterBlockRatio + 10;
  return flashNotMoreGlared && (flashHasHigherMean || flashWinsMoreBlocks);
}

// Çıktı tamponundaki pikseli ortak analiz ızgarasına çevirir.
function mapOutputPixelToAnalysis(x: number, y: number, width: number, height: number) {
  return {
    x: ((x + 0.5) * ANALYSIS_WIDTH) / Math.max(width, 1) - 0.5,
    y: ((y + 0.5) * ANALYSIS_HEIGHT) / Math.max(height, 1) - 0.5,
  };
}

// Karşılaştırmadan gelen kayıt parametrelerini dönüşüm nesnesine çevirir.
function comparisonToAlignment(comparison: ExposurePairComparison): AlignmentTransform {
  return {
    shiftX: comparison.alignmentShiftX,
    shiftY: comparison.alignmentShiftY,
    scale: comparison.alignmentScale,
    rotationDegrees: comparison.alignmentRotationDegrees,
    confidence: comparison.alignmentConfidence / 100,
  };
}

// Ambient koordinatını flaşlı ROI koordinatına ölçek ve dönme ile taşır.
function mapAmbientToFlash(x: number, y: number, comparison: ExposurePairComparison) {
  const alignment = comparisonToAlignment(comparison);
  return mapPoint(x, y, alignment);
}

// Analiz merkezini koruyarak bir nokta için affine benzeri küçük dönüşüm uygular.
function mapPoint(x: number, y: number, alignment: AlignmentTransform) {
  const centerX = ANALYSIS_WIDTH / 2;
  const centerY = ANALYSIS_HEIGHT / 2;
  const radians = (alignment.rotationDegrees * Math.PI) / 180;
  const deltaX = (x - centerX) * alignment.scale;
  const deltaY = (y - centerY) * alignment.scale;
  return {
    x: centerX + deltaX * Math.cos(radians) - deltaY * Math.sin(radians) + alignment.shiftX,
    y: centerY + deltaX * Math.sin(radians) + deltaY * Math.cos(radians) + alignment.shiftY,
  };
}

// Gradyan korelasyonu ile ölçek, dönme ve kayma adayları arasından en iyi hizalamayı seçer.
function findBestAlignment(
  ambient: NormalizedExposure,
  flash: NormalizedExposure
): AlignmentTransform {
  let best: AlignmentTransform = {
    shiftX: 0,
    shiftY: 0,
    scale: 1,
    rotationDegrees: 0,
    confidence: 0,
  };

  for (const scale of ALIGNMENT_SCALES) {
    for (const rotationDegrees of ALIGNMENT_ANGLES) {
      for (let shiftY = -8; shiftY <= 8; shiftY += 4) {
        for (let shiftX = -8; shiftX <= 8; shiftX += 4) {
          const candidate = {
            shiftX,
            shiftY,
            scale,
            rotationDegrees,
            confidence: 0,
          };
          candidate.confidence = scoreAlignment(ambient, flash, candidate, 4);
          if (candidate.confidence > best.confidence) best = candidate;
        }
      }
    }
  }

  const fineAngles = [best.rotationDegrees - 1.5, best.rotationDegrees, best.rotationDegrees + 1.5];
  const fineScales = [best.scale - 0.02, best.scale, best.scale + 0.02];
  for (const scale of fineScales) {
    for (const rotationDegrees of fineAngles) {
      for (let shiftY = best.shiftY - 2; shiftY <= best.shiftY + 2; shiftY += 1) {
        for (let shiftX = best.shiftX - 2; shiftX <= best.shiftX + 2; shiftX += 1) {
          const candidate = {
            shiftX,
            shiftY,
            scale,
            rotationDegrees,
            confidence: 0,
          };
          candidate.confidence = scoreAlignment(ambient, flash, candidate, 4);
          if (candidate.confidence > best.confidence) best = candidate;
        }
      }
    }
  }

  return best;
}

// Bir dönüşüm adayının gradyan büyüklükleri arasındaki normalize korelasyonunu hesaplar.
function scoreAlignment(
  ambient: NormalizedExposure,
  flash: NormalizedExposure,
  alignment: AlignmentTransform,
  step: number
) {
  let sumAmbientSquared = 0;
  let sumFlashSquared = 0;
  let sumProduct = 0;
  let count = 0;

  for (let y = 10; y < ANALYSIS_HEIGHT - 10; y += step) {
    for (let x = 8; x < ANALYSIS_WIDTH - 8; x += step) {
      const ambientIndex = y * ANALYSIS_WIDTH + x;
      if (!ambient.mask[ambientIndex]) continue;
      const flashPoint = mapPoint(x, y, alignment);
      const flashX = Math.round(flashPoint.x);
      const flashY = Math.round(flashPoint.y);
      if (
        flashX < 1 ||
        flashX >= ANALYSIS_WIDTH - 1 ||
        flashY < 1 ||
        flashY >= ANALYSIS_HEIGHT - 1
      ) {
        continue;
      }
      const flashIndex = flashY * ANALYSIS_WIDTH + flashX;
      if (!flash.mask[flashIndex]) continue;
      const ambientValue = ambient.gradientMagnitude[ambientIndex];
      const flashValue = flash.gradientMagnitude[flashIndex];
      sumAmbientSquared += ambientValue * ambientValue;
      sumFlashSquared += flashValue * flashValue;
      sumProduct += ambientValue * flashValue;
      count += 1;
    }
  }

  if (count < 100) return 0;
  return sumProduct / Math.sqrt(Math.max(sumAmbientSquared * sumFlashSquared, 1));
}

// Normalize edilmiş ROI'yi gri ton, maske ve gradyan tamponlarına taşır.
function normalizeExposure(result: ExposureAnalysisInput): NormalizedExposure {
  const grayscale = new Uint8Array(ANALYSIS_WIDTH * ANALYSIS_HEIGHT);
  const highlightMask = new Uint8Array(grayscale.length);
  const mask = new Uint8Array(grayscale.length);

  for (let y = 0; y < ANALYSIS_HEIGHT; y += 1) {
    const sourceY = Math.min(
      Math.floor(((y + 0.5) * result.height) / ANALYSIS_HEIGHT),
      result.height - 1
    );
    for (let x = 0; x < ANALYSIS_WIDTH; x += 1) {
      const sourceX = Math.min(
        Math.floor(((x + 0.5) * result.width) / ANALYSIS_WIDTH),
        result.width - 1
      );
      const targetIndex = y * ANALYSIS_WIDTH + x;
      const sourceIndex = sourceY * result.width + sourceX;
      const pixelIndex = sourceIndex * 4;
      const red = result.pixels[pixelIndex];
      const green = result.pixels[pixelIndex + 1];
      const blue = result.pixels[pixelIndex + 2];
      const gray = Math.round(red * 0.299 + green * 0.587 + blue * 0.114);
      grayscale[targetIndex] = gray;
      highlightMask[targetIndex] =
        gray >= 218 || (Math.max(red, green, blue) >= 248 && gray >= 195) ? 1 : 0;
      mask[targetIndex] = result.mask[sourceIndex] ? 1 : 0;
    }
  }

  const gradients = createGradients(grayscale, mask);
  return { grayscale, highlightMask, mask, ...gradients };
}

// Sobel gradyanlarını ridge kalitesi ve yön uyumu için tek sefer hesaplar.
function createGradients(grayscale: Uint8Array, mask: Uint8Array) {
  const gradientX = new Float32Array(grayscale.length);
  const gradientY = new Float32Array(grayscale.length);
  const gradientMagnitude = new Float32Array(grayscale.length);

  for (let y = 1; y < ANALYSIS_HEIGHT - 1; y += 1) {
    for (let x = 1; x < ANALYSIS_WIDTH - 1; x += 1) {
      const index = y * ANALYSIS_WIDTH + x;
      if (!mask[index]) continue;
      const top = index - ANALYSIS_WIDTH;
      const bottom = index + ANALYSIS_WIDTH;
      const gx =
        -grayscale[top - 1] + grayscale[top + 1] -
        2 * grayscale[index - 1] + 2 * grayscale[index + 1] -
        grayscale[bottom - 1] + grayscale[bottom + 1];
      const gy =
        -grayscale[top - 1] - 2 * grayscale[top] - grayscale[top + 1] +
        grayscale[bottom - 1] + 2 * grayscale[bottom] + grayscale[bottom + 1];
      gradientX[index] = gx;
      gradientY[index] = gy;
      gradientMagnitude[index] = Math.hypot(gx, gy);
    }
  }

  return { gradientX, gradientY, gradientMagnitude };
}

// Aynı blokta parlama, kontrast, gradyan ve yön tutarlılığını ortak puana çevirir.
function compareBlock({
  ambient,
  flash,
  blockX,
  blockY,
  alignment,
}: {
  ambient: NormalizedExposure;
  flash: NormalizedExposure;
  blockX: number;
  blockY: number;
  alignment: AlignmentTransform;
}) {
  const ambientValues: number[] = [];
  const flashValues: number[] = [];
  let ambientGradient = 0;
  let flashGradient = 0;
  let ambientGxxMinusGyy = 0;
  let ambientTwoGxy = 0;
  let ambientGradientEnergy = 0;
  let flashGxxMinusGyy = 0;
  let flashTwoGxy = 0;
  let flashGradientEnergy = 0;
  let ambientGlarePixels = 0;
  let flashGlarePixels = 0;

  for (let y = blockY; y < blockY + BLOCK_SIZE; y += 1) {
    for (let x = blockX; x < blockX + BLOCK_SIZE; x += 1) {
      const flashPoint = mapPoint(x, y, alignment);
      const flashX = Math.round(flashPoint.x);
      const flashY = Math.round(flashPoint.y);
      if (
        flashX < 1 ||
        flashX >= ANALYSIS_WIDTH - 1 ||
        flashY < 1 ||
        flashY >= ANALYSIS_HEIGHT - 1
      ) {
        continue;
      }

      const ambientIndex = y * ANALYSIS_WIDTH + x;
      const flashIndex = flashY * ANALYSIS_WIDTH + flashX;
      if (!ambient.mask[ambientIndex] || !flash.mask[flashIndex]) continue;

      const ambientGray = ambient.grayscale[ambientIndex];
      const flashGray = flash.grayscale[flashIndex];
      ambientValues.push(ambientGray);
      flashValues.push(flashGray);
      if (ambient.highlightMask[ambientIndex]) ambientGlarePixels += 1;
      if (flash.highlightMask[flashIndex]) flashGlarePixels += 1;

      const ambientGx = ambient.gradientX[ambientIndex];
      const ambientGy = ambient.gradientY[ambientIndex];
      const flashGx = flash.gradientX[flashIndex];
      const flashGy = flash.gradientY[flashIndex];
      ambientGradient += ambient.gradientMagnitude[ambientIndex];
      flashGradient += flash.gradientMagnitude[flashIndex];
      ambientGxxMinusGyy += ambientGx * ambientGx - ambientGy * ambientGy;
      ambientTwoGxy += 2 * ambientGx * ambientGy;
      ambientGradientEnergy += ambientGx * ambientGx + ambientGy * ambientGy;
      flashGxxMinusGyy += flashGx * flashGx - flashGy * flashGy;
      flashTwoGxy += 2 * flashGx * flashGy;
      flashGradientEnergy += flashGx * flashGx + flashGy * flashGy;
    }
  }

  if (ambientValues.length < BLOCK_SIZE * BLOCK_SIZE * 0.45) return null;
  const pixelCount = ambientValues.length;
  return {
    ambientScore: createBlockScore({
      values: ambientValues,
      gradientMean: ambientGradient / pixelCount,
      coherence: Math.hypot(ambientGxxMinusGyy, ambientTwoGxy) /
        Math.max(ambientGradientEnergy, 1),
      glareRatio: ambientGlarePixels / pixelCount,
    }),
    flashScore: createBlockScore({
      values: flashValues,
      gradientMean: flashGradient / pixelCount,
      coherence: Math.hypot(flashGxxMinusGyy, flashTwoGxy) /
        Math.max(flashGradientEnergy, 1),
      glareRatio: flashGlarePixels / pixelCount,
    }),
    ambientGlarePixels,
    flashGlarePixels,
    pixelCount,
  };
}

// Blok parlaklığı, ridge gradyanı, yön tutarlılığı ve parlama riskini puanlar.
function createBlockScore({
  values,
  gradientMean,
  coherence,
  glareRatio,
}: {
  values: number[];
  gradientMean: number;
  coherence: number;
  glareRatio: number;
}) {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance =
    values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  const contrastScore = Math.min(Math.sqrt(variance) / 42, 1);
  const ridgeScore = Math.min(gradientMean / 180, 1);
  const illuminationScore = Math.max(0, 1 - glareRatio * 5);
  const brightFlatPenalty = mean > 190 && contrastScore < 0.28 ? 18 : 0;
  return (
    contrastScore * 25 +
    ridgeScore * 30 +
    Math.min(coherence, 1) * 25 +
    illuminationScore * 20 -
    brightFlatPenalty
  );
}

function ratio(value: number, total: number) {
  return Math.round((value / Math.max(total, 1)) * 100);
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(Math.max(value, minimum), maximum);
}
