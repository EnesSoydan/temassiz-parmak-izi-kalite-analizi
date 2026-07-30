import * as FileSystem from 'expo-file-system/legacy';

import type {
  CaptureSample,
  DetectionClassName,
  FingerRoi,
  QualityCalibrationLabel,
} from '@/types/biometrics';

// Capture verilerini Expo'nun uygulama içi documentDirectory alanında tutuyoruz.
const ROOT_DIR = `${FileSystem.documentDirectory ?? ''}fingerprint-captures/`;
const RAW_DIR = `${ROOT_DIR}raw/`;
const ROI_DIR = `${ROOT_DIR}roi/`;
const CANONICAL_ROI_DIR = `${ROOT_DIR}canonical-roi/`;
const SEGMENTED_ROI_DIR = `${ROOT_DIR}segmented-roi/`;
const ENHANCED_ROI_DIR = `${ROOT_DIR}enhanced-roi/`;
const ORIENTATION_ROI_DIR = `${ROOT_DIR}orientation-roi/`;
const PROCESSED_ROI_DIR = `${ROOT_DIR}processed-roi/`;
const INDEX_FILE = `${ROOT_DIR}captures.json`;
const CALIBRATION_SUMMARY_FILE = `${ROOT_DIR}quality-calibration-summary.json`;

// Ham görüntü, ROI ve index dosyası için gerekli klasörleri hazırlar.
export async function ensureCaptureStorage() {
  await ensureDirectory(ROOT_DIR);
  await ensureDirectory(RAW_DIR);
  await ensureDirectory(ROI_DIR);
  await ensureDirectory(CANONICAL_ROI_DIR);
  await ensureDirectory(SEGMENTED_ROI_DIR);
  await ensureDirectory(ENHANCED_ROI_DIR);
  await ensureDirectory(ORIENTATION_ROI_DIR);
  await ensureDirectory(PROCESSED_ROI_DIR);
}

// Normalize edilmiş ham kamera karesini kalıcı uygulama dizinine kopyalar.
export async function saveRawImage(sourceUri: string, sampleId: string) {
  await ensureCaptureStorage();
  const targetUri = `${RAW_DIR}${sampleId}.jpg`;
  await FileSystem.copyAsync({ from: sourceUri, to: targetUri });
  return targetUri;
}

// Kırpılmış parmak ROI görüntüsünü ham kareden ayrı dosya olarak saklar.
export async function saveRoiImage(sourceUri: string, sampleId: string) {
  await ensureCaptureStorage();
  const targetUri = `${ROI_DIR}${sampleId}.jpg`;
  await FileSystem.copyAsync({ from: sourceUri, to: targetUri });
  return targetUri;
}

// Modelin bulduğu parmağa ait ROI görselini sınıf adıyla ayrı bir dosya olarak saklar.
export async function saveFingerRoiImage(
  sourceUri: string,
  sampleId: string,
  className: DetectionClassName
) {
  await ensureCaptureStorage();
  const targetUri = `${ROI_DIR}${sampleId}-${className}.jpg`;
  await FileSystem.copyAsync({ from: sourceUri, to: targetUri });
  return targetUri;
}

// OBB açısına göre dikleştirilmiş kanonik ROI görüntüsünü ayrı klasörde saklar.
export async function saveCanonicalFingerRoiImage(
  sourceUri: string,
  sampleId: string,
  className: DetectionClassName
) {
  await ensureCaptureStorage();
  const targetUri = `${CANONICAL_ROI_DIR}${sampleId}-${className}.jpg`;
  await FileSystem.copyAsync({ from: sourceUri, to: targetUri });
  return targetUri;
}

// Segmentasyon çıktısının yazılacağı kalıcı dosya yolunu hazırlar.
export async function createSegmentedFingerRoiImageUri(
  sampleId: string,
  className: DetectionClassName
) {
  await ensureCaptureStorage();
  return `${SEGMENTED_ROI_DIR}${sampleId}-${className}.jpg`;
}

// Yön ve frekans destekli ridge iyileştirme çıktısının kalıcı yolunu hazırlar.
export async function createEnhancedFingerRoiImageUri(
  sampleId: string,
  className: DetectionClassName
) {
  await ensureCaptureStorage();
  return `${ENHANCED_ROI_DIR}${sampleId}-${className}.jpg`;
}

// Blok tabanlı ridge yön haritasının kalıcı dosya yolunu hazırlar.
export async function createOrientationFingerRoiImageUri(
  sampleId: string,
  className: DetectionClassName
) {
  await ensureCaptureStorage();
  return `${ORIENTATION_ROI_DIR}${sampleId}-${className}.jpg`;
}

// Ön işlemden geçmiş ROI JPEG base64 verisini ayrı dosya olarak yazar.
export async function saveProcessedRoiImage(base64: string, sampleId: string) {
  await ensureCaptureStorage();
  const targetUri = `${PROCESSED_ROI_DIR}${sampleId}.jpg`;
  await FileSystem.writeAsStringAsync(targetUri, base64, { encoding: FileSystem.EncodingType.Base64 });
  return targetUri;
}

// captures.json index dosyasını okuyup kayıtları en güncel haliyle döndürür.
export async function loadCaptureSamples(): Promise<CaptureSample[]> {
  await ensureCaptureStorage();
  const info = await FileSystem.getInfoAsync(INDEX_FILE);

  // İlk açılışta index dosyası olmayabilir; bunu boş galeri olarak kabul ederiz.
  if (!info.exists) {
    return [];
  }

  const text = await FileSystem.readAsStringAsync(INDEX_FILE);
  return JSON.parse(text) as CaptureSample[];
}

// Yeni capture örneğini listenin başına ekler ve index dosyasını günceller.
export async function appendCaptureSample(sample: CaptureSample) {
  const samples = await loadCaptureSamples();
  const nextSamples = [sample, ...samples];
  await FileSystem.writeAsStringAsync(INDEX_FILE, JSON.stringify(nextSamples, null, 2));
  return nextSamples;
}

// Kullanıcının verdiği kalite etiketini yalnızca capture metadata'sında günceller.
export async function setCaptureCalibrationLabel(
  sampleId: string,
  calibrationLabel: QualityCalibrationLabel
) {
  const samples = await loadCaptureSamples();
  const nextSamples = samples.map((sample) =>
    sample.id === sampleId ? { ...sample, calibrationLabel } : sample
  );
  await FileSystem.writeAsStringAsync(INDEX_FILE, JSON.stringify(nextSamples, null, 2));
  return nextSamples;
}

// Etiketli örneklerin fotoğraf yollarını ve kimliklerini dışarıda bırakan kalibrasyon özeti üretir.
export async function exportQualityCalibrationSummary() {
  const samples = await loadCaptureSamples();
  const labeledSamples = samples
    .filter((sample) => sample.calibrationLabel)
    .map((sample) => ({
      label: sample.calibrationLabel,
      source: sample.deviceModel?.includes('Galeri') ? 'gallery' : 'camera',
      captureQualityStatus: sample.qualityStatus,
      fingers: (sample.fingerRois ?? []).map((fingerRoi) => ({
        className: fingerRoi.className,
        detectionConfidence: fingerRoi.confidence,
        sourcePixelWidth: fingerRoi.sourcePixelWidth ?? null,
        canonicalRotationDegrees:
          fingerRoi.canonicalRotationDegrees ?? null,
        silhouetteAxisDegrees: fingerRoi.silhouetteAxisDegrees ?? null,
        canonicalResidualDegrees:
          fingerRoi.canonicalResidualDegrees ?? null,
        quality: fingerRoi.quality
          ? {
              globalScore: fingerRoi.quality.globalScore,
              captureStatus:
                fingerRoi.quality.captureStatus ?? fingerRoi.quality.status,
              biometricStatus:
                fingerRoi.quality.biometricStatus ?? 'insufficient',
              sourceResolutionScore:
                fingerRoi.quality.sourceResolutionScore ?? 0,
              validEvidenceRatio:
                fingerRoi.quality.validEvidenceRatio ??
                fingerRoi.quality.ridgeValidBlockRatio,
              blurScore: fingerRoi.quality.blurScore,
              contrastScore: fingerRoi.quality.contrastScore,
              brightnessScore: fingerRoi.quality.brightnessScore,
              orientationCoherence: fingerRoi.quality.orientationCoherence,
              orientationReliableBlockRatio:
                fingerRoi.quality.orientationReliableBlockRatio ?? 0,
              orientationMedianCorrectionDegrees:
                fingerRoi.quality.orientationMedianCorrectionDegrees ?? 0,
              ridgePeriodicity: fingerRoi.quality.ridgePeriodicity,
              ridgeFrequencyConsistency:
                fingerRoi.quality.ridgeFrequencyConsistency,
              ridgeCandidateBlockCount:
                fingerRoi.quality.ridgeCandidateBlockCount,
              ridgeValidBlockCount: fingerRoi.quality.ridgeValidBlockCount,
              ridgeAnalysisBlockSize:
                fingerRoi.quality.ridgeAnalysisBlockSize ?? 0,
              ridgeAnalyzedScaleCount:
                fingerRoi.quality.ridgeAnalyzedScaleCount ?? 1,
              ridgeOrientationCandidateRatio:
                fingerRoi.quality.ridgeOrientationCandidateRatio ?? 0,
              biometricRequiredValidBlockCount:
                fingerRoi.quality.biometricRequiredValidBlockCount ?? 0,
              biometricRejectionReasons:
                fingerRoi.quality.biometricRejectionReasons ?? [],
              ridgeEnhancementGainPercent:
                fingerRoi.quality.ridgeEnhancementGainPercent ?? 0,
              ridgeEnhancementSupportedAreaRatio:
                fingerRoi.quality.ridgeEnhancementSupportedAreaRatio ?? 0,
            }
          : null,
      })),
    }));

  const summary = {
    schemaVersion: 1,
    createdAt: new Date().toISOString(),
    labeledSampleCount: labeledSamples.length,
    samples: labeledSamples,
  };
  await FileSystem.writeAsStringAsync(
    CALIBRATION_SUMMARY_FILE,
    JSON.stringify(summary, null, 2)
  );
  return {
    uri: CALIBRATION_SUMMARY_FILE,
    sampleCount: labeledSamples.length,
  };
}

// Bir kaydı hem metadata listesinden hem de ilişkili görüntü dosyalarından kaldırır.
export async function deleteCaptureSample(sampleId: string) {
  const samples = await loadCaptureSamples();
  const sample = samples.find((item) => item.id === sampleId);
  const nextSamples = samples.filter((item) => item.id !== sampleId);

  // Ham görüntü dosyası yoksa silme işlemini hata saymayız; metadata yine temizlenir.
  if (sample?.rawImageUri) {
    await deleteFileIfExists(sample.rawImageUri);
  }

  // ROI dosyası da aynı kayıtla birlikte temizlenir.
  if (sample?.roiImageUri) {
    await deleteFileIfExists(sample.roiImageUri);
  }

  // Çoklu parmak ROI dosyaları varsa ham kayıtla birlikte kaldırılır.
  if (sample?.fingerRois) {
    await Promise.all(
      sample.fingerRois.map(async (fingerRoi) => {
        if (fingerRoi.imageUri) {
          await deleteFileIfExists(fingerRoi.imageUri);
        }

        // Kanonik ROI varsa ham ROI ile birlikte temizleriz.
        if (fingerRoi.canonicalImageUri) {
          await deleteFileIfExists(fingerRoi.canonicalImageUri);
        }

        // Segmentasyonlu ROI varsa normal ROI ile birlikte temizleriz.
        if (fingerRoi.segmentedImageUri) {
          await deleteFileIfExists(fingerRoi.segmentedImageUri);
        }

        // Ridge iyileştirme çıktısı varsa aynı kaydın diğer ROI dosyalarıyla birlikte sileriz.
        if (fingerRoi.enhancedImageUri) {
          await deleteFileIfExists(fingerRoi.enhancedImageUri);
        }

        // Eski sürümlerde üretilmiş fark haritası varsa geriye dönük olarak temizleriz.
        const legacyDifferenceUri = (
          fingerRoi as FingerRoi & {
            enhancementDifferenceImageUri?: string;
          }
        ).enhancementDifferenceImageUri;
        if (legacyDifferenceUri) {
          await deleteFileIfExists(legacyDifferenceUri);
        }

        // Orientation doğrulama haritası varsa parmak kaydıyla birlikte sileriz.
        if (fingerRoi.orientationImageUri) {
          await deleteFileIfExists(fingerRoi.orientationImageUri);
        }
      })
    );
  }

  // İşlenmiş ROI varsa kayıtla birlikte kaldırılır; eski kayıtlarda bu alan olmayabilir.
  if (sample?.processedRoiImageUri) {
    await deleteFileIfExists(sample.processedRoiImageUri);
  }

  await FileSystem.writeAsStringAsync(INDEX_FILE, JSON.stringify(nextSamples, null, 2));
  return nextSamples;
}

// Eksik klasörleri idempotent şekilde oluşturur.
async function ensureDirectory(uri: string) {
  const info = await FileSystem.getInfoAsync(uri);

  // Klasör zaten varsa dokunmayız; yoksa ara dizinlerle birlikte oluştururuz.
  if (!info.exists) {
    await FileSystem.makeDirectoryAsync(uri, { intermediates: true });
  }
}

// Dosya varsa siler, yoksa işlemi sessizce tamamlar.
async function deleteFileIfExists(uri: string) {
  const info = await FileSystem.getInfoAsync(uri);

  // idempotent silme, aynı dosya daha önce kaldırılmış olsa bile akışı kırmaz.
  if (info.exists) {
    await FileSystem.deleteAsync(uri, { idempotent: true });
  }
}
