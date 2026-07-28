import type { FingerRoi, FingerprintQuality, QualityStatus } from '@/types/biometrics';

type QualityAnalysisInput = {
  pixels: Uint8Array;
  mask: Uint8Array;
  width: number;
  height: number;
  coverage: number;
};

type BlockQuality = {
  blurScore: number;
  contrastScore: number;
  brightnessScore: number;
  textureScore: number;
  meanBrightness: number;
  shadowRatio: number;
  glareRatio: number;
};

// Blokların parmak alanı sayılması için gereken minimum maske kapsaması.
const MIN_BLOCK_MASK_COVERAGE = 0.6;

// Kitaptaki yerel kalite yaklaşımına uygun, ayarlanabilir blok taban genişliği.
const BASE_BLOCK_SIZE = 16;

// İlk saha testinde tek noktadan kalibrasyon yapılabilmesi için geçici eşikler burada tutulur.
const QUALITY_THRESHOLDS = {
  minCoverage: 0.18,
  maxCoverage: 0.96,
  goodScore: 62,
  mediumScore: 34,
  dimBrightness: 58,
  brightBrightness: 210,
  glareRatio: 0.035,
  weakBlur: 32,
  weakTexture: 28,
  weakContrast: 30,
};

// Segmentasyon maskesi içindeki ROI'yi bloklara ayırarak parmak bazında kalite sonucu üretir.
export function analyzeFingerprintQuality({
  pixels,
  mask,
  width,
  height,
  coverage,
}: QualityAnalysisInput): FingerprintQuality {
  const rawGrayscale = createGrayscaleImage(pixels, width, height);
  const featureGrayscale = preprocessFingerprintGrayscale(rawGrayscale, mask, width, height);
  const blockSize = getBlockSize(width, height);
  const blocks: BlockQuality[] = [];

  for (let top = 1; top + blockSize < height - 1; top += blockSize) {
    for (let left = 1; left + blockSize < width - 1; left += blockSize) {
      const block = analyzeBlock(rawGrayscale, featureGrayscale, mask, width, left, top, blockSize);
      if (block) blocks.push(block);
    }
  }

  if (blocks.length === 0) {
    return createSegmentationFailureQuality(coverage);
  }

  const blurScore = summarizeBlockScores(blocks.map((block) => block.blurScore));
  const contrastScore = summarizeBlockScores(blocks.map((block) => block.contrastScore));
  const brightnessScore = summarizeBlockScores(blocks.map((block) => block.brightnessScore));
  const textureVisibility = summarizeBlockScores(blocks.map((block) => block.textureScore));
  const foregroundCoverage = Math.round(clamp01(coverage) * 100);
  const coverageScore = calculateCoverageScore(coverage);
  const globalScore = Math.round(
    blurScore * 0.25 +
      textureVisibility * 0.25 +
      contrastScore * 0.2 +
      brightnessScore * 0.15 +
      coverageScore * 0.15
  );
  const meanBrightness = getMedian(blocks.map((block) => block.meanBrightness));
  const glareRatio = getPercentile(blocks.map((block) => block.glareRatio), 0.75);
  const message = createQualityMessage({
    coverageScore,
    blurScore,
    contrastScore,
    brightnessScore,
    textureVisibility,
    meanBrightness,
    glareRatio,
  });

  return {
    globalScore: clampScore(globalScore),
    blurScore,
    contrastScore,
    brightnessScore,
    foregroundCoverage,
    textureVisibility,
    status: getFingerprintQualityStatus(globalScore),
    message,
  };
}

// Segmentasyon geçersizse kaydı korurken kalite sonucunu açıkça başarısız işaretler.
export function createSegmentationFailureQuality(foregroundCoverage = 0): FingerprintQuality {
  return {
    globalScore: 0,
    blurScore: 0,
    contrastScore: 0,
    brightnessScore: 0,
    foregroundCoverage: Math.round(clamp01(foregroundCoverage) * 100),
    textureVisibility: 0,
    status: 'poor',
    message: 'Parmak alanı ayrılamadı. Elini daha düz ve net göster.',
  };
}

// Çoklu parmak kaydının genel durumunu en düşük ROI kalitesinden türetir.
export function getCaptureQualityStatus(fingerRois: FingerRoi[]): QualityStatus {
  const qualities = fingerRois
    .map((fingerRoi) => fingerRoi.quality)
    .filter((quality): quality is FingerprintQuality => quality !== undefined);

  if (qualities.length === 0) return 'unknown';
  if (qualities.some((quality) => quality.status === 'poor')) return 'poor';
  if (qualities.some((quality) => quality.status === 'medium')) return 'usable';
  return 'good';
}

// Kamera ekranında en zayıf parmağı önceleyen kısa ve anlaşılır geri bildirim üretir.
export function createCaptureQualityFeedback(fingerRois: FingerRoi[]) {
  const poorQualityRois = fingerRois
    .filter((fingerRoi) => fingerRoi.quality?.status === 'poor')
    .sort((first, second) => (first.quality?.globalScore ?? 0) - (second.quality?.globalScore ?? 0));
  const mediumQualityCount = fingerRois.filter(
    (fingerRoi) => fingerRoi.quality?.status === 'medium'
  ).length;

  if (poorQualityRois.length === 0 && mediumQualityCount === 0) {
    return "Parmak ROI'leri uygun kalitede kaydedildi.";
  }

  if (poorQualityRois.length === 0) {
    return "Parmak ROI'leri orta kalitede kaydedildi.";
  }

  if (poorQualityRois.length > 1) {
    return `${poorQualityRois.length} parmak ROI'si için yeniden çekim öneriliyor.`;
  }

  const fingerRoi = poorQualityRois[0];
  return `${formatFingerClass(fingerRoi.className)} parmak: ${fingerRoi.quality?.message ?? 'kalite düşük.'}`;
}

// Kayıtlar ekranında kullanılacak kısa Türkçe kalite etiketini döndürür.
export function formatFingerprintQualityStatus(quality?: FingerprintQuality) {
  if (!quality) return 'Analiz yok';
  if (quality.status === 'good') return 'Uygun';
  if (quality.status === 'medium') return 'Orta';
  return 'Tekrar çek';
}

// ROI boyutuna göre küçük görüntülerde aşırı ince, büyüklerde aşırı iri blok kullanılmasını önler.
function getBlockSize(width: number, height: number) {
  const shortestEdge = Math.min(width, height);
  return Math.max(BASE_BLOCK_SIZE, Math.min(32, Math.round(shortestEdge / 8)));
}

// Bir blokta yalnızca maske içindeki pikselleri kullanarak ışık, kontrast, keskinlik ve doku ölçer.
function analyzeBlock(
  rawGrayscale: Uint8Array,
  featureGrayscale: Uint8Array,
  mask: Uint8Array,
  width: number,
  left: number,
  top: number,
  blockSize: number
): BlockQuality | null {
  const values: number[] = [];
  const rawValues: number[] = [];
  let maskedCount = 0;
  let pixelCount = 0;
  let shadowCount = 0;
  let glareCount = 0;
  let gradientEnergy = 0;
  let tensorX = 0;
  let tensorY = 0;
  let tensorMagnitude = 0;

  for (let y = top; y < top + blockSize; y += 1) {
    for (let x = left; x < left + blockSize; x += 1) {
      pixelCount += 1;
      const index = y * width + x;
      if (!mask[index]) continue;

      const rawValue = rawGrayscale[index];
      const featureValue = featureGrayscale[index];
      values.push(featureValue);
      rawValues.push(rawValue);
      maskedCount += 1;
      if (rawValue < 45) shadowCount += 1;
      if (rawValue > 235) glareCount += 1;

      const gradientX = (featureGrayscale[index + 1] ?? featureValue) -
        (featureGrayscale[index - 1] ?? featureValue);
      const gradientY = (featureGrayscale[index + width] ?? featureValue) -
        (featureGrayscale[index - width] ?? featureValue);
      const energy = gradientX * gradientX + gradientY * gradientY;
      gradientEnergy += energy;
      tensorX += gradientX * gradientX - gradientY * gradientY;
      tensorY += 2 * gradientX * gradientY;
      tensorMagnitude += energy;
    }
  }

  if (maskedCount / Math.max(pixelCount, 1) < MIN_BLOCK_MASK_COVERAGE || values.length < 12) {
    return null;
  }

  const meanBrightness = getMean(rawValues);
  const contrastRange = getPercentile(values, 0.9) - getPercentile(values, 0.1);
  const averageGradientEnergy = gradientEnergy / maskedCount;
  const coherence = Math.hypot(tensorX, tensorY) / Math.max(tensorMagnitude, 1);
  const blurScore = scoreRange(averageGradientEnergy, 80, 900);
  const contrastScore = scoreRange(contrastRange, 18, 62);
  const brightnessScore = calculateBrightnessScore(
    meanBrightness,
    shadowCount / maskedCount,
    glareCount / maskedCount
  );
  const textureScore = clampScore(blurScore * 0.55 + coherence * 100 * 0.45);

  return {
    blurScore,
    contrastScore,
    brightnessScore,
    textureScore,
    meanBrightness,
    shadowRatio: shadowCount / maskedCount,
    glareRatio: glareCount / maskedCount,
  };
}

// Kalite ölçümünde çizgi dokusunu daha adil görmek için lokal ışık normalizasyonu uygular.
function preprocessFingerprintGrayscale(
  grayscale: Uint8Array,
  mask: Uint8Array,
  width: number,
  height: number
) {
  const radius = Math.max(5, Math.min(12, Math.round(Math.min(width, height) / 12)));
  const integral = createIntegralImage(grayscale, width, height);
  const normalized = new Uint8Array(grayscale.length);
  const values: number[] = [];

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
      const value = clampByte(128 + (grayscale[index] - localMean) * 1.65);

      normalized[index] = value;
      values.push(value);
    }
  }

  const low = getPercentile(values, 0.04);
  const high = getPercentile(values, 0.96);
  const range = Math.max(high - low, 12);

  for (let index = 0; index < normalized.length; index += 1) {
    if (!mask[index]) continue;
    normalized[index] = clampByte(((normalized[index] - low) / range) * 255);
  }

  return normalized;
}

// RGBA görüntüyü tekrar eden renk hesabını önlemek için bir kez gri tona dönüştürür.
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

// Yerel blok skorlarını medyan ve zayıf çeyrekle birleştirerek bozuk alanları görünür kılar.
function summarizeBlockScores(scores: number[]) {
  const median = getMedian(scores);
  const lowerQuartile = getPercentile(scores, 0.25);
  return clampScore(median * 0.7 + lowerQuartile * 0.3);
}

// Segmentasyon maskesinin çok küçük veya neredeyse tüm ROI'yi kaplaması durumunda puan düşürür.
function calculateCoverageScore(coverage: number) {
  if (coverage < QUALITY_THRESHOLDS.minCoverage) {
    return scoreRange(coverage, 0.03, QUALITY_THRESHOLDS.minCoverage);
  }

  if (coverage > QUALITY_THRESHOLDS.maxCoverage) {
    return scoreRange(1 - coverage, 0.01, 1 - QUALITY_THRESHOLDS.maxCoverage);
  }

  return 100;
}

// Ortalama ışık, gölge ve parlama oranını tek parlaklık puanına dönüştürür.
function calculateBrightnessScore(meanBrightness: number, shadowRatio: number, glareRatio: number) {
  const meanScore =
    meanBrightness < QUALITY_THRESHOLDS.dimBrightness
      ? scoreRange(meanBrightness, 20, QUALITY_THRESHOLDS.dimBrightness)
      : meanBrightness > QUALITY_THRESHOLDS.brightBrightness
        ? scoreRange(255 - meanBrightness, 20, 255 - QUALITY_THRESHOLDS.brightBrightness)
        : 100;
  const shadowPenalty = clamp01(shadowRatio / 0.28) * 25;
  const glarePenalty = clamp01(glareRatio / QUALITY_THRESHOLDS.glareRatio) * 35;

  return clampScore(meanScore - shadowPenalty - glarePenalty);
}

// En baskın sorun üzerinden kullanıcının hemen uygulayabileceği tek mesajı belirler.
function createQualityMessage({
  coverageScore,
  blurScore,
  contrastScore,
  brightnessScore,
  textureVisibility,
  meanBrightness,
  glareRatio,
}: {
  coverageScore: number;
  blurScore: number;
  contrastScore: number;
  brightnessScore: number;
  textureVisibility: number;
  meanBrightness: number;
  glareRatio: number;
}) {
  if (coverageScore < 45) return 'Parmak alanı ayrılamadı. Elini daha düz ve net göster.';
  if (brightnessScore < 45 && meanBrightness < QUALITY_THRESHOLDS.dimBrightness) {
    return 'Işık yetersiz. Daha aydınlık bir ortam dene.';
  }
  if (brightnessScore < 45 && glareRatio > QUALITY_THRESHOLDS.glareRatio) {
    return 'Parlama fazla. Işığı doğrudan parmağına tutma.';
  }
  if (blurScore < QUALITY_THRESHOLDS.weakBlur) return 'Görüntü bulanık. Parmağını sabit tut.';
  if (textureVisibility < QUALITY_THRESHOLDS.weakTexture) {
    return 'Parmak izi dokusu yeterince görünmüyor. Parmağını biraz yaklaştır.';
  }
  if (contrastScore < QUALITY_THRESHOLDS.weakContrast) {
    return 'Kontrast düşük. Daha dengeli ışıkta tekrar dene.';
  }
  return 'Görüntü uygun.';
}

// Genel skorun kullanıcıya gösterilecek üç seviyeli durumunu belirler.
function getFingerprintQualityStatus(globalScore: number): FingerprintQuality['status'] {
  if (globalScore >= QUALITY_THRESHOLDS.goodScore) return 'good';
  if (globalScore >= QUALITY_THRESHOLDS.mediumScore) return 'medium';
  return 'poor';
}

// Parmak sınıfını kısa Türkçe etikete dönüştürür.
function formatFingerClass(className: FingerRoi['className']) {
  if (className === 'index') return 'İşaret';
  if (className === 'middle') return 'Orta';
  if (className === 'ring') return 'Yüzük';
  if (className === 'pinky') return 'Serçe';
  return 'Parmak';
}

// Lokal ortalama hesabını hızlı yapmak için gri görüntünün integral toplamını çıkarır.
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

// Integral görüntüden dikdörtgen toplamını okuyarak lokal parlaklık ortalamasını destekler.
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

// Ham ölçümü 0-100 arası kalibre edilmiş puana çevirir.
function scoreRange(value: number, minimum: number, maximum: number) {
  return clampScore(((value - minimum) / Math.max(maximum - minimum, 1)) * 100);
}

// Skoru güvenli 0-100 aralığında tutar.
function clampScore(value: number) {
  return Math.round(Math.min(Math.max(value, 0), 100));
}

// Oranı güvenli 0-1 aralığında tutar.
function clamp01(value: number) {
  return Math.min(Math.max(value, 0), 1);
}

// Piksel değerini güvenli 8-bit gri ton aralığında tutar.
function clampByte(value: number) {
  return Math.round(Math.min(Math.max(value, 0), 255));
}
