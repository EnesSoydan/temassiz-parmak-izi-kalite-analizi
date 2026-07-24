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
  useFullImage?: boolean;
};

const ROI_WIDTH_RATIO = 0.42;
const ROI_ASPECT_RATIO = 0.62;

// Görüntü pikseli okunamazsa kaliteyi uydurmadan yalnızca ROI doluluk bilgisini döndürür.
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

// JPEG base64 görüntüden temel kalite sinyallerini hesaplar: netlik, parlama, parlaklık ve ROI doluluk.
export function analyzeFrameQuality({ base64, useFullImage = false }: AnalyzeFrameInput): QualityMetrics {
  const image = decode(base64ToBytes(base64), { useTArray: true });

  // ROI zaten kırpılmış geldiyse tüm görüntüyü, ham frame geldiyse merkezi ROI alanını analiz ederiz.
  const roi = useFullImage
    ? { originX: 0, originY: 0, width: image.width, height: image.height }
    : getCenteredRoi(image.width, image.height);

  // Büyük görsellerde ölçümü hızlandırmak için ROI içinde seyrek örnekleme yapılır.
  const step = Math.max(1, Math.floor(Math.min(roi.width, roi.height) / 96));
  const grayWidth = Math.ceil(roi.width / step);
  const grayHeight = Math.ceil(roi.height / step);
  const gray = new Float32Array(grayWidth * grayHeight);

  let index = 0;
  let brightPixels = 0;
  let totalBrightness = 0;

  // RGB pikselleri gri tona çevirirken parlama ve ortalama ışık bilgisini aynı geçişte toplarız.
  for (let y = roi.originY; y < roi.originY + roi.height; y += step) {
    for (let x = roi.originX; x < roi.originX + roi.width; x += step) {
      const pixelIndex = (y * image.width + x) * 4;
      const red = image.data[pixelIndex] ?? 0;
      const green = image.data[pixelIndex + 1] ?? 0;
      const blue = image.data[pixelIndex + 2] ?? 0;
      const value = 0.299 * red + 0.587 * green + 0.114 * blue;

      gray[index] = value;
      totalBrightness += value;

      // Üç kanal da çok yüksekse bu pikseli parlama adayı sayıyoruz.
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

// Kalite durumunu kullanıcıya gösterilecek kısa Türkçe metne çevirir.
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

// Kalite metriklerini kullanıcıya doğrudan aksiyon veren kısa bir öneri metnine çevirir.
export function getQualityAdvice(metrics: QualityMetrics) {
  // Ölçüm üretilemediyse sonucu iyiymiş gibi göstermeyip tekrar çekim isteriz.
  if (metrics.status === 'unknown') {
    return 'Kalite ölçümü tamamlanamadı. Kareyi tekrar yakala.';
  }

  // Netlik eşiği düşükse ilk öneri mesafe ve sabitleme olur; parmak izi çizgileri için bu en kritik sinyal.
  if (metrics.blurScore < 18) {
    return 'Netlik düşük. Telefonu biraz uzaklaştırıp sabit tut.';
  }

  // Aşırı parlama ridge bilgisini örtebilir; ışık açısını değiştirmek gerekir.
  if (metrics.glareRatio > 0.08) {
    return 'Parlama fazla. Işığın geliş açısını değiştir.';
  }

  // Çok karanlık görüntüde çizgi yapısı ölçülemez; kullanıcıyı daha aydınlık ortama yönlendiririz.
  if (metrics.brightnessMean < 45) {
    return 'Görüntü karanlık. Ortam ışığını artır.';
  }

  // Çok parlak görüntüde parmak yüzeyi patlar; ışığı azaltmak veya açı değiştirmek gerekir.
  if (metrics.brightnessMean > 225) {
    return 'Görüntü çok parlak. Işığı azalt veya açı değiştir.';
  }

  // Orta kalite örnekleri demo için saklanabilir ama enrollment için daha iyi kare hedeflenir.
  if (metrics.status === 'usable') {
    return 'Kullanılabilir ama daha net bir kare tercih edilir.';
  }

  return 'Kalite uygun. Bu kare enrollment adayı olabilir.';
}

// Enrollment'a girecek örneklerde daha sıkı eşikler kullanır; her saklanan kare eğitim/galeri adayı olmak zorunda değil.
export function isEnrollmentQualityAcceptable(metrics: QualityMetrics) {
  // Bilinmeyen, düşük veya sadece kullanılabilir kaliteyi enrollment dışı bırakırız.
  if (metrics.status !== 'good') {
    return false;
  }

  // Matcher denemeleri için netliği daha yüksek tutuyoruz; bulanık çizgi yapısı yanlış eşleşme üretir.
  if (metrics.blurScore < 45) {
    return false;
  }

  // Parlak yüzey patlamaları ridge bilgisini kapattığı için enrollment eşiği genel kalite eşiğinden daha katıdır.
  if (metrics.glareRatio > 0.025) {
    return false;
  }

  // Çok karanlık veya çok parlak ama "good" sınırında kalmış örnekleri de galeriye sokmayız.
  if (metrics.brightnessMean < 85 || metrics.brightnessMean > 200) {
    return false;
  }

  return true;
}

// Enrollment kararını kayıt ekranlarında kısa ve anlaşılır Türkçe metne dönüştürür.
export function formatEnrollmentDecision(metrics?: QualityMetrics) {
  if (!metrics) {
    return 'ölçüm yok';
  }

  return isEnrollmentQualityAcceptable(metrics) ? 'enrollment adayı' : 'tekrar çekilmeli';
}

// Ham frame analizi gerektiğinde ekran rehberiyle uyumlu merkezi ROI alanını hesaplar.
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

// Laplacian varyansına benzer basit bir keskinlik skoru üretir; skor düştükçe bulanıklık artar.
function calculateSharpnessScore(gray: Float32Array, width: number, height: number) {
  // Laplacian komşuluk hesabı için en az 3x3 piksel gerekir.
  if (width < 3 || height < 3) {
    return 0;
  }

  let sum = 0;
  let sumSquares = 0;
  let count = 0;

  // Her pikseli dört komşusuyla karşılaştırarak kenar/çizgi enerjisini ölçeriz.
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

// Metrik eşiklerini tek bir genel kalite sınıfına indirger.
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

  // Base64 her dört karakterde üç byte üretir; padding varsa son byte'ları atlarız.
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
