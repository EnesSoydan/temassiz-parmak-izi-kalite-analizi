import type { OrientationBlock } from '@/lib/orientation-field';

export type RidgeFrequencyBlock = {
  left: number;
  top: number;
  size: number;
  periodPixels: number;
  periodicity: number;
  autocorrelation: number;
  halfPeriodCorrelation: number;
  cycleCount: number;
  angleRadians: number;
  rejectionReason:
    | 'none'
    | 'signature'
    | 'amplitude'
    | 'peak'
    | 'period-boundary'
    | 'correlation'
    | 'half-period'
    | 'ridge-valley'
    | 'cycles'
    | 'score';
  valid: boolean;
};

export type RidgeFrequencySummary = {
  ridgePeriodicity: number;
  ridgeFrequencyConsistency: number;
  validBlockRatio: number;
  validBlockCount: number;
  candidateBlockCount: number;
  interiorBlockCount: number;
  orientationBlockCount: number;
  medianPeriodPixels: number;
  periodHistogram: string;
  rejectionSummary: string;
  blocks: RidgeFrequencyBlock[];
};

// Aynı ROI'nin farklı blok ölçeklerinde ölçülen sonuçlarından fiziksel iç alanda daha güçlü kanıt üreteni seçer.
export function selectPreferredRidgeScale<
  Analysis extends {
    ridgeFrequency: RidgeFrequencySummary;
  },
>(primary: Analysis, secondary: Analysis) {
  const primaryScore = scoreRidgeScale(primary.ridgeFrequency);
  const secondaryScore = scoreRidgeScale(secondary.ridgeFrequency);
  return secondaryScore > primaryScore + 2 ? secondary : primary;
}

// Küçük blokların salt adet avantajını önlemek için geçerli kanıtı tüm iç blok alanına oranlar.
function scoreRidgeScale(summary: RidgeFrequencySummary) {
  const physicalEvidenceRatio =
    summary.validBlockCount / Math.max(summary.interiorBlockCount, 1);
  return (
    physicalEvidenceRatio * 100 * 0.55 +
    summary.ridgePeriodicity * 0.25 +
    summary.ridgeFrequencyConsistency * 0.2
  );
}

type EstimateRidgeFrequencyInput = {
  grayscale: Uint8Array;
  mask: Uint8Array;
  width: number;
  height: number;
  orientationBlocks: OrientationBlock[];
};

// Telefon ROI'lerinde görülen düşük kontrastlı ridge profilini düz görüntüden ayıran alt genlik sınırı.
const MIN_SIGNATURE_STANDARD_DEVIATION = 1.2;

// Güvenilir ve komşularıyla uyumlu yönü bulunan bloklarda gerçek ridge tekrarını ölçer.
export function estimateRidgeFrequency({
  grayscale,
  mask,
  width,
  height,
  orientationBlocks,
}: EstimateRidgeFrequencyInput): RidgeFrequencySummary {
  const primaryInteriorBlocks = orientationBlocks.filter(
    (block) => (block.gridOffset ?? 0) === 0 && block.maskCoverage >= 0.85
  );
  const allInteriorBlocks = orientationBlocks.filter((block) => block.maskCoverage >= 0.85);
  const rawCandidateBlocks = allInteriorBlocks.filter(
    (block) =>
      block.coherence >= 0.24 &&
      block.neighborhoodConsistency >= 0.4 &&
      (block.smoothedCoherence ?? block.coherence) >= 0.42
  );
  const rawBlocks = rawCandidateBlocks.map((block) =>
    estimateBlockFrequency(grayscale, mask, width, height, block)
  );
  const blocks = selectAreaNormalizedBlocks(
    primaryInteriorBlocks,
    rawCandidateBlocks,
    rawBlocks
  );
  const validBlocks = blocks.filter((block) => block.valid);
  const validBlockRatio = validBlocks.length / Math.max(blocks.length, 1);
  const validPeriods = validBlocks.map((block) => block.periodPixels);
  // Ridge gücü yalnızca doğrulanmış bloklardan gelir; ne kadar alan bulunduğunu validBlockRatio ayrı ölçer.
  const ridgePeriodicity = summarizeScores(
    validBlocks.map((block) => block.periodicity)
  );
  const ridgeFrequencyConsistency = calculateFrequencyConsistency(validBlocks);

  return {
    ridgePeriodicity,
    ridgeFrequencyConsistency,
    validBlockRatio,
    validBlockCount: validBlocks.length,
    candidateBlockCount: blocks.length,
    interiorBlockCount: primaryInteriorBlocks.length,
    orientationBlockCount: orientationBlocks.filter(
      (block) => (block.gridOffset ?? 0) === 0
    ).length,
    medianPeriodPixels: getMedian(validPeriods),
    periodHistogram: createPeriodHistogram(validPeriods),
    rejectionSummary: createRejectionSummary(blocks),
    blocks,
  };
}

// İki kaydırılmış ızgaradan aynı fiziksel alanı temsil eden ölçümler içinde en güvenilir olanı tutar.
function selectAreaNormalizedBlocks(
  primaryBlocks: OrientationBlock[],
  candidateBlocks: OrientationBlock[],
  measuredBlocks: RidgeFrequencyBlock[]
) {
  const bestByPrimaryIndex = new Map<number, RidgeFrequencyBlock>();

  measuredBlocks.forEach((measurement, measurementIndex) => {
    const candidate = candidateBlocks[measurementIndex];
    const centerX = candidate.left + candidate.size / 2;
    const centerY = candidate.top + candidate.size / 2;
    let nearestIndex = -1;
    let nearestDistance = Number.POSITIVE_INFINITY;

    primaryBlocks.forEach((primaryBlock, primaryIndex) => {
      const primaryCenterX = primaryBlock.left + primaryBlock.size / 2;
      const primaryCenterY = primaryBlock.top + primaryBlock.size / 2;
      const distance = Math.hypot(centerX - primaryCenterX, centerY - primaryCenterY);

      if (distance < nearestDistance && distance <= primaryBlock.size * 0.76) {
        nearestIndex = primaryIndex;
        nearestDistance = distance;
      }
    });

    if (nearestIndex < 0) return;
    const previous = bestByPrimaryIndex.get(nearestIndex);
    if (!previous || isStrongerMeasurement(measurement, previous)) {
      bestByPrimaryIndex.set(nearestIndex, measurement);
    }
  });

  return [...bestByPrimaryIndex.values()];
}

// Aynı ana hücreye düşen ölçümlerde geçerli ve daha periyodik kanıtı tercih eder.
function isStrongerMeasurement(
  candidate: RidgeFrequencyBlock,
  reference: RidgeFrequencyBlock
) {
  if (candidate.valid !== reference.valid) return candidate.valid;
  return candidate.periodicity > reference.periodicity;
}

// Bir blokta x-signature, tam/yarım periyot karşıtlığı ve çevrim sayısıyla ridge tekrarını doğrular.
function estimateBlockFrequency(
  grayscale: Uint8Array,
  mask: Uint8Array,
  width: number,
  height: number,
  block: OrientationBlock
): RidgeFrequencyBlock {
  const signature = createXSignature(grayscale, mask, width, height, block);
  if (!signature) return createInvalidBlock(block, 'signature');

  const signal = detrendAndSmoothSignal(signature);
  const standardDeviation = getStandardDeviation(signal);
  if (standardDeviation < MIN_SIGNATURE_STANDARD_DEVIATION) {
    return createInvalidBlock(block, 'amplitude');
  }

  // Dört pikselin altındaki tekrar telefon görüntüsünde güvenilir ridge çözünürlüğü sayılmaz.
  const minimumPeriod = 4;
  const maximumPeriod = Math.min(13, Math.floor(signal.length / 3));
  const peak = findAutocorrelationPeak(signal, minimumPeriod, maximumPeriod);
  if (!peak) return createInvalidBlock(block, 'peak');
  if (peak.period === minimumPeriod || peak.period === maximumPeriod) {
    return createInvalidBlock(block, 'period-boundary', {
      periodPixels: peak.period,
      autocorrelation: peak.correlation,
    });
  }

  const halfPeriod = Math.max(2, Math.round(peak.period / 2));
  const halfPeriodCorrelation = calculateNormalizedAutocorrelation(signal, halfPeriod);
  const ridgeValleyContrast = peak.correlation - halfPeriodCorrelation;
  const cycleCount = countAlternatingCycles(signal);
  const correlationScore = scoreRange(peak.correlation, 0.3, 0.72);
  const ridgeValleyScore = scoreRange(ridgeValleyContrast, 0.35, 1.25);
  const amplitudeScore = scoreRange(standardDeviation, MIN_SIGNATURE_STANDARD_DEVIATION, 10);
  const cycleScore = scoreRange(cycleCount, 2.5, 4.5);
  const periodicity = clampScore(
    correlationScore * 0.35 +
      ridgeValleyScore * 0.35 +
      amplitudeScore * 0.15 +
      cycleScore * 0.15
  );
  const valid =
    peak.correlation >= 0.32 &&
    halfPeriodCorrelation <= 0.18 &&
    ridgeValleyContrast >= 0.38 &&
    cycleCount >= 2.5 &&
    periodicity >= 28;
  const rejectionReason =
    peak.correlation < 0.32
      ? 'correlation'
      : halfPeriodCorrelation > 0.18
        ? 'half-period'
        : ridgeValleyContrast < 0.38
          ? 'ridge-valley'
          : cycleCount < 2.5
            ? 'cycles'
            : periodicity < 28
              ? 'score'
              : 'none';

  return {
    left: block.left,
    top: block.top,
    size: block.size,
    periodPixels: peak.period,
    periodicity,
    autocorrelation: peak.correlation,
    halfPeriodCorrelation,
    cycleCount,
    angleRadians: block.angleRadians,
    rejectionReason,
    valid,
  };
}

// Ridge yönüne paralel pikselleri ortalayıp çizgilere dik tek boyutlu x-signature üretir.
function createXSignature(
  grayscale: Uint8Array,
  mask: Uint8Array,
  width: number,
  height: number,
  block: OrientationBlock
) {
  const centerX = block.left + block.size / 2;
  const centerY = block.top + block.size / 2;
  const tangentX = Math.cos(block.angleRadians);
  const tangentY = Math.sin(block.angleRadians);
  const normalX = -tangentY;
  const normalY = tangentX;
  // Küçük orientation hücresinde de 9-12 px ridge periyotlarını görebilecek kadar uzun profil örnekleriz.
  const normalRadius = Math.max(18, block.size);
  const tangentRadius = Math.max(4, Math.round(block.size * 0.55));
  const requiredTangentSamples = Math.ceil((tangentRadius * 2 + 1) * 0.65);
  const signature: (number | null)[] = [];

  for (let normalOffset = -normalRadius; normalOffset <= normalRadius; normalOffset += 1) {
    let total = 0;
    let count = 0;

    for (
      let tangentOffset = -tangentRadius;
      tangentOffset <= tangentRadius;
      tangentOffset += 1
    ) {
      const x = Math.round(
        centerX + normalX * normalOffset + tangentX * tangentOffset
      );
      const y = Math.round(
        centerY + normalY * normalOffset + tangentY * tangentOffset
      );

      if (x < 0 || x >= width || y < 0 || y >= height) continue;
      const index = y * width + x;
      if (!mask[index]) continue;

      total += grayscale[index];
      count += 1;
    }

    signature.push(count >= requiredTangentSamples ? total / count : null);
  }

  const validCount = signature.filter((value) => value !== null).length;
  if (validCount / signature.length < 0.82) return null;

  return fillMissingSamples(signature);
}

// Maske sınırındaki az sayıdaki boş profil örneğini en yakın geçerli değerlerle tamamlar.
function fillMissingSamples(signature: (number | null)[]) {
  const completed = [...signature];
  let lastValue: number | null = null;

  for (let index = 0; index < completed.length; index += 1) {
    if (completed[index] !== null) {
      lastValue = completed[index];
    } else if (lastValue !== null) {
      completed[index] = lastValue;
    }
  }

  let nextValue: number | null = null;
  for (let index = completed.length - 1; index >= 0; index -= 1) {
    if (completed[index] !== null) {
      nextValue = completed[index];
    } else if (nextValue !== null) {
      completed[index] = nextValue;
    }
  }

  if (completed.some((value) => value === null)) return null;
  return completed as number[];
}

// Yavaş ışık değişimini hareketli ortalamayla çıkarıp yalnızca yerel ridge tekrarını bırakır.
function detrendAndSmoothSignal(signal: number[]) {
  const trendRadius = Math.max(3, Math.round(signal.length / 8));
  const detrended = signal.map((value, index) => {
    let total = 0;
    let count = 0;

    for (
      let sampleIndex = Math.max(0, index - trendRadius);
      sampleIndex <= Math.min(signal.length - 1, index + trendRadius);
      sampleIndex += 1
    ) {
      total += signal[sampleIndex];
      count += 1;
    }

    return value - total / Math.max(count, 1);
  });

  return detrended.map((value, index) => {
    const previous = detrended[index - 1] ?? value;
    const next = detrended[index + 1] ?? value;
    return (previous + value * 2 + next) / 4;
  });
}

// Sıfır geçişlerini sayarak profilde en az üç ridge/valley çevrimi bulunmasını doğrular.
function countAlternatingCycles(signal: number[]) {
  const threshold = Math.max(getStandardDeviation(signal) * 0.18, 1);
  let previousSign = 0;
  let crossings = 0;

  for (const value of signal) {
    const sign = value > threshold ? 1 : value < -threshold ? -1 : 0;
    if (sign === 0) continue;
    if (previousSign !== 0 && sign !== previousSign) crossings += 1;
    previousSign = sign;
  }

  return crossings / 2;
}

// Beklenen ridge aralığında yalnızca iç bölgede kalan yerel otokorelasyon tepesini bulur.
function findAutocorrelationPeak(signal: number[], minimumPeriod: number, maximumPeriod: number) {
  if (maximumPeriod < minimumPeriod) return null;

  const correlations: { period: number; correlation: number }[] = [];
  for (let period = minimumPeriod; period <= maximumPeriod; period += 1) {
    correlations.push({
      period,
      correlation: calculateNormalizedAutocorrelation(signal, period),
    });
  }

  const localPeaks = correlations.filter((candidate, index) => {
    if (index === 0 || index === correlations.length - 1) return false;
    const previous = correlations[index - 1].correlation;
    const next = correlations[index + 1].correlation;
    return candidate.correlation > previous && candidate.correlation >= next;
  });

  return localPeaks.reduce<{ period: number; correlation: number } | null>(
    (best, candidate) =>
      !best || candidate.correlation > best.correlation ? candidate : best,
    null
  );
}

// İki kaydırılmış profil arasındaki benzerliği -1 ile 1 arasında hesaplar.
function calculateNormalizedAutocorrelation(signal: number[], period: number) {
  let product = 0;
  let firstEnergy = 0;
  let secondEnergy = 0;

  for (let index = 0; index + period < signal.length; index += 1) {
    const first = signal[index];
    const second = signal[index + period];
    product += first * second;
    firstEnergy += first * first;
    secondEnergy += second * second;
  }

  return product / Math.max(Math.sqrt(firstEnergy * secondEnergy), 0.0001);
}

// Frekans haritasında yalnızca yakın blokları karşılaştırarak perspektif kaynaklı yavaş değişime izin verir.
function calculateFrequencyConsistency(blocks: RidgeFrequencyBlock[]) {
  if (blocks.length < 4) return 0;

  const localScores = blocks.flatMap((block) => {
    const neighbors = blocks.filter((candidate) => {
      if (candidate === block) return false;
      const neighborhoodRadius = Math.max(block.size, candidate.size) * 1.6;
      return (
        Math.abs(candidate.left - block.left) <= neighborhoodRadius &&
        Math.abs(candidate.top - block.top) <= neighborhoodRadius
      );
    });

    if (neighbors.length === 0) return [];

    const similarities = neighbors.map((neighbor) => {
      const referencePeriod = Math.max((block.periodPixels + neighbor.periodPixels) / 2, 1);
      const relativeDifference =
        Math.abs(block.periodPixels - neighbor.periodPixels) / referencePeriod;
      return clampScore((1 - relativeDifference / 0.45) * 100);
    });

    return [getMedian(similarities)];
  });

  if (localScores.length < Math.max(2, Math.ceil(blocks.length * 0.35))) return 0;
  return summarizeScores(localScores);
}

// Geçersiz ölçümü sıfır puanla saklayarak başarısız blokların genel sonucu etkilemesini sağlar.
function createInvalidBlock(
  block: OrientationBlock,
  rejectionReason: RidgeFrequencyBlock['rejectionReason'],
  values: Partial<RidgeFrequencyBlock> = {}
): RidgeFrequencyBlock {
  return {
    left: block.left,
    top: block.top,
    size: block.size,
    periodPixels: 0,
    periodicity: 0,
    autocorrelation: 0,
    halfPeriodCorrelation: 0,
    cycleCount: 0,
    angleRadians: block.angleRadians,
    rejectionReason,
    valid: false,
    ...values,
  };
}

// Blok ret nedenlerini terminalde okunabilir kısa bir sayaç metnine dönüştürür.
function createRejectionSummary(blocks: RidgeFrequencyBlock[]) {
  const counts = new Map<RidgeFrequencyBlock['rejectionReason'], number>();

  for (const block of blocks) {
    if (block.rejectionReason === 'none') continue;
    counts.set(block.rejectionReason, (counts.get(block.rejectionReason) ?? 0) + 1);
  }

  if (counts.size === 0) return 'yok';
  return [...counts.entries()]
    .sort((first, second) => second[1] - first[1])
    .map(([reason, count]) => `${reason}:${count}`)
    .join(',');
}

// Geçerli blokların piksel periyotlarını kısa bir dağılım metnine dönüştürür.
function createPeriodHistogram(periods: number[]) {
  if (periods.length === 0) return 'yok';
  const counts = new Map<number, number>();

  for (const period of periods) {
    const roundedPeriod = Math.round(period);
    counts.set(roundedPeriod, (counts.get(roundedPeriod) ?? 0) + 1);
  }

  return [...counts.entries()]
    .sort((first, second) => first[0] - second[0])
    .map(([period, count]) => `${period}:${count}`)
    .join(',');
}

// Blok puanlarını medyan ve zayıf çeyrekle birleştirir.
function summarizeScores(scores: number[]) {
  const median = getMedian(scores);
  const lowerQuartile = getPercentile(scores, 0.25);
  return clampScore(median * 0.7 + lowerQuartile * 0.3);
}

// Dizinin standart sapmasını profil genliği için hesaplar.
function getStandardDeviation(values: number[]) {
  const mean = getMean(values);
  const variance =
    values.reduce((total, value) => total + (value - mean) ** 2, 0) /
    Math.max(values.length, 1);
  return Math.sqrt(variance);
}

// Sayı dizisinin ortalamasını hesaplar.
function getMean(values: number[]) {
  return values.reduce((total, value) => total + value, 0) / Math.max(values.length, 1);
}

// Sıralı kopya üzerinden istenen yüzdelik değeri döndürür.
function getPercentile(values: number[], percentile: number) {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((first, second) => first - second);
  const index = (sorted.length - 1) * percentile;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const fraction = index - lower;

  return (sorted[lower] ?? 0) * (1 - fraction) + (sorted[upper] ?? 0) * fraction;
}

// Ortanca değeri yüzdelik yardımcısıyla hesaplar.
function getMedian(values: number[]) {
  return getPercentile(values, 0.5);
}

// Ham ölçümü ayarlanabilir aralıktan 0-100 puana çevirir.
function scoreRange(value: number, minimum: number, maximum: number) {
  return clampScore(((value - minimum) / Math.max(maximum - minimum, 0.0001)) * 100);
}

// Skoru güvenli 0-100 aralığında tutar.
function clampScore(value: number) {
  return Math.round(Math.min(Math.max(value, 0), 100));
}
