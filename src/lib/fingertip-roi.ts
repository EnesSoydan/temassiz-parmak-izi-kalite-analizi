import { Images, type Image as NitroImage } from 'react-native-nitro-image';

import {
  createEnhancedFingerRoiImageUri,
  createMinutiaeFingerRoiImageUri,
  createOrientationFingerRoiImageUri,
  createSegmentedFingerRoiImageUri,
  saveCanonicalFingerRoiImage,
  saveFingerRoiImage,
} from '@/lib/capture-storage';
import { segmentFingerRoiImage } from '@/lib/finger-segmentation';
import {
  analyzeFingerprintQualityDetailed,
  createSegmentationFailureQuality,
} from '@/lib/fingerprint-quality';
import { sortFingerRois } from '@/lib/finger-order';
import {
  extractFingerprintMinutiae,
  type MinutiaeExtractionResult,
} from '@/lib/minutiae-extraction';
import { saveRidgeEnhancedImage } from '@/lib/ridge-enhancement';
import type {
  CaptureSample,
  DetectedObbBox,
  FingerRoi,
  FingerprintQuality,
  MinutiaeRejectionReason,
} from '@/types/biometrics';

type ImageSize = NonNullable<CaptureSample['rawImageSize']>;

// Çok küçük hatalı kırpmaların dosya üretmesini engelleyen minimum piksel boyutu.
const MIN_ROI_PIXEL_SIZE = 16;

// Üç işçi native beklemeleri paylaşırken dördüncü büyük piksel çalışma kopyasını açmayarak belleği sınırlar.
const ROI_PROCESSING_CONCURRENCY = 2;

// Enrollment öncesi geçici template kapısı, çok az veya taşmış aday listesini biyometrik veri olarak saklamaz.
const MIN_TEMPLATE_MINUTIAE_COUNT = 8;
const MAX_TEMPLATE_MINUTIAE_COUNT = 60;
const MIN_SEARCHABLE_AREA_RATIO = 0.015;
const MIN_LARGEST_SEARCHABLE_REGION_RATIO = 0.008;

// Canlı veya fotoğraf model kutularından parmak başına normalize ROI planı üretir.
export function createFingerRoiPlans(
  detections: DetectedObbBox[],
  imageSize: ImageSize
): FingerRoi[] {
  return sortFingerRois(
    detections
      .map((detection) => createFingerRoiPlan(detection, imageSize))
      .filter((roi): roi is FingerRoi => roi !== null)
  );
}

// Ham görselden model tespitlerine göre her parmak için ayrı ROI JPEG dosyası çıkarır.
export async function extractFingerRoisFromImage({
  imageUri,
  imageSize,
  detections,
  sampleId,
  sourceImage,
}: {
  imageUri: string;
  imageSize: ImageSize;
  detections: DetectedObbBox[];
  sampleId: string;
  sourceImage?: NitroImage;
}) {
  const roiPlans = createFingerRoiPlans(detections, imageSize);
  const processingStartedAt = Date.now();
  // Kamera çekiminde bellekteki yönlü görüntüyü, galeri akışında ise dosyadan yüklenen tek native görüntüyü paylaşırız.
  const sharedSourceImage =
    sourceImage ?? (await Images.loadFromFileAsync(toNativeFilePath(imageUri)));

  // İki işçi native kırpma ve dosya erişimini üst üste bindirir; üç işçi cihazda JS sıkışmasına yol açıyordu.
  const fingerRois = await mapWithConcurrency(
    roiPlans,
    ROI_PROCESSING_CONCURRENCY,
    async (roiPlan) => {
      const fingerStartedAt = Date.now();
      const cropStartedAt = Date.now();

      // Nitro Image aynı çözülmüş fotoğraftan eksene hizalı ham ROI'yi native tarafta kırpar.
      const cropped = await sharedSourceImage.cropAsync(
        roiPlan.crop.originX,
        roiPlan.crop.originY,
        roiPlan.crop.originX + roiPlan.crop.width,
        roiPlan.crop.originY + roiPlan.crop.height
      );
      const croppedPath = await cropped.saveToTemporaryFileAsync('jpg', 95);
      const cropMs = Date.now() - cropStartedAt;

      // Geçici ROI dosyasını uygulamanın kalıcı capture klasörüne taşınabilir isimle kopyalar.
      const saveStartedAt = Date.now();
      const savedImageUri = await saveFingerRoiImage(
        toFileUri(croppedPath),
        sampleId,
        roiPlan.className
      );
      const canonical = await createCanonicalFingerRoi(cropped, roiPlan);
      const canonicalPath = await canonical.image.saveToTemporaryFileAsync('jpg', 95);
      const canonicalImageUri = await saveCanonicalFingerRoiImage(
        toFileUri(canonicalPath),
        sampleId,
        roiPlan.className
      );
      const saveMs = Date.now() - saveStartedAt;
      const processedRoi = await tryProcessFingerRoi({
        roiImageUri: canonicalImageUri,
        sampleId,
        className: roiPlan.className,
        sourceWidth: canonical.sourcePixelWidth,
        canonicalRotationDegrees: canonical.rotationDegrees,
      });

      const timings = processedRoi.timings;
      console.info(
        timings
          ? `[ROI süre] parmak=${roiPlan.className}, kırpma=${cropMs}ms, kayıt=${saveMs}ms, çözme=${timings.decodeMs}ms, maske=${timings.maskMs}ms, iyileştirme=${timings.enhancementMs}ms, jpeg=${timings.encodeMs}ms, seg_yazma=${timings.writeMs}ms, kalite=${timings.qualityMs}ms, teknik_yazma=${timings.technicalWriteMs}ms, toplam=${Date.now() - fingerStartedAt}ms`
          : `[ROI süre] parmak=${roiPlan.className}, kırpma=${cropMs}ms, kayıt=${saveMs}ms, segmentasyon=başarısız, toplam=${Date.now() - fingerStartedAt}ms`
      );

      return {
        ...roiPlan,
        imageUri: savedImageUri,
        canonicalImageUri,
        segmentedImageUri: processedRoi.segmentedImageUri,
        enhancedImageUri: processedRoi.enhancedImageUri,
        orientationImageUri: processedRoi.orientationImageUri,
        minutiaeImageUri: processedRoi.minutiaeImageUri,
        minutiaeTemplate: processedRoi.minutiaeTemplate,
        sourcePixelWidth: canonical.sourcePixelWidth,
        canonicalRotationDegrees: canonical.rotationDegrees,
        silhouetteAxisDegrees: processedRoi.silhouetteAxisDegrees,
        canonicalResidualDegrees: processedRoi.canonicalResidualDegrees,
        quality: processedRoi.quality,
      };
    }
  );

  console.info(
    `[ROI süre] parmak_sayısı=${fingerRois.length}, toplam=${Date.now() - processingStartedAt}ms`
  );

  // Paralel işçiler farklı zamanda bitse de kayıt metadata'sı her zaman kullanıcı sırasını korur.
  return sortFingerRois(fingerRois);
}

// Verilen işleri sabit sayıda işçiyle çalıştırıp sonuçların giriş sırasını korur.
async function mapWithConcurrency<Input, Output>(
  items: Input[],
  concurrency: number,
  worker: (item: Input) => Promise<Output>
) {
  const results = new Array<Output>(items.length);
  let nextIndex = 0;

  // Her işçi sıradaki boş indeksi alır; JavaScript eşzamanlı kalırken native işler bekleme süresini paylaşır.
  async function runWorker() {
    while (nextIndex < items.length) {
      const currentIndex = nextIndex;
      nextIndex += 1;
      results[currentIndex] = await worker(items[currentIndex]);
    }
  }

  const workerCount = Math.min(Math.max(concurrency, 1), items.length);
  await Promise.all(Array.from({ length: workerCount }, () => runWorker()));
  return results;
}

// Segmentasyon başarısız olursa normal ROI kaydını bozmadan boş sonuç döndürür.
async function tryProcessFingerRoi({
  roiImageUri,
  sampleId,
  className,
  sourceWidth,
  canonicalRotationDegrees,
}: {
  roiImageUri: string;
  sampleId: string;
  className: FingerRoi['className'];
  sourceWidth: number;
  canonicalRotationDegrees: number;
}) {
  try {
    const outputImageUri = await createSegmentedFingerRoiImageUri(sampleId, className);
    const enhancedImageUri = await createEnhancedFingerRoiImageUri(sampleId, className);
    const orientationImageUri = await createOrientationFingerRoiImageUri(
      sampleId,
      className
    );
    const minutiaeImageUri = await createMinutiaeFingerRoiImageUri(
      sampleId,
      className
    );
    const result = await segmentFingerRoiImage({ roiImageUri, outputImageUri, sourceWidth });

    if (!result) {
      return { quality: createSegmentationFailureQuality() };
    }

    const qualityStartedAt = Date.now();
    const analysis = analyzeFingerprintQualityDetailed({
      ...result,
      sourcePixelWidth: sourceWidth,
      fingerClass: className,
    });
    const quality = analysis.quality;
    const qualityMs = Date.now() - qualityStartedAt;
    let savedEnhancedImageUri: string | undefined;
    let savedOrientationImageUri: string | undefined;
    let savedMinutiaeImageUri: string | undefined;
    let minutiaeTemplate: FingerRoi['minutiaeTemplate'];
    let minutiaeVisualization: Uint8Array | undefined;

    // Teknik Nokta görüntüsü kalite kararından bağımsız üretilir; template kabulü aşağıdaki ayrı kapıda yapılır.
    if (
      analysis.enhancedPixels &&
      analysis.minutiaeSupportMask &&
      analysis.minutiaeOrientationMask &&
      quality.ridgeMedianPeriodPixels > 0
    ) {
      const minutiaeStartedAt = Date.now();
      const minutiae = extractFingerprintMinutiae({
        pixels: analysis.enhancedPixels,
        mask: result.mask,
        candidateMask: analysis.minutiaeSupportMask,
        orientationMask: analysis.minutiaeOrientationMask,
        width: result.width,
        height: result.height,
        ridgePeriodPixels: quality.ridgeMedianPeriodPixels,
      });
      minutiaeVisualization = minutiae.visualizationPixels;
      const minutiaeAssessment = assessMinutiaeTemplate(quality, minutiae);
      quality.minutiaeStatus = minutiaeAssessment.status;
      quality.minutiaeRejectionReason = minutiaeAssessment.reason;
      quality.minutiaeCandidateCount = minutiae.template.minutiae.length;
      quality.minutiaeSearchableAreaRatio = Math.round(
        minutiae.searchableAreaRatio * 100
      );
      quality.minutiaeLargestRegionRatio = Math.round(
        minutiae.largestSearchableRegionRatio * 100
      );
      if (minutiaeAssessment.status === 'sufficient') {
        minutiaeTemplate = minutiae.template;
      }
      const endingCount = minutiae.template.minutiae.filter(
        (item) => item.type === 'ending'
      ).length;
      const bifurcationCount =
        minutiae.template.minutiae.length - endingCount;
      console.info(
        `[Minutiae] parmak=${className}, durum=${minutiaeAssessment.status}, ret=${minutiaeAssessment.reason ?? 'yok'}, toplam=${minutiae.template.minutiae.length}, son=${endingCount}, çatallanma=${bifurcationCount}, ham_aday=${minutiae.rawCandidateCount}, iskelet=${minutiae.skeletonPixelCount}, mikro_delik=${minutiae.filledHolePixelCount}, küçük_bileşen=${minutiae.removedComponentPixelCount}, budanan=${minutiae.prunedPixelCount}, bağlanan=${minutiae.bridgedPixelCount}, destek=${Math.round(minutiae.supportCoverage * 100)}%, aranabilir=${Math.round(minutiae.searchableAreaRatio * 100)}%, büyük_bölge=${Math.round(minutiae.largestSearchableRegionRatio * 100)}%, ridge_oranı=${Math.round(minutiae.binaryRidgeRatio * 100)}%, inceltme=${minutiae.thinningIterations}, süre=${Date.now() - minutiaeStartedAt}ms`
      );
    } else {
      quality.minutiaeStatus = 'insufficient';
      quality.minutiaeRejectionReason = 'search-area';
    }

    const technicalWriteStartedAt = Date.now();
    const technicalImageWrites: Promise<unknown>[] = [];

    // Yön haritası ridge kanıtı çıkmasa da orientation hesabını doğrulayabilmek için saklanır.
    if (analysis.orientationPixels) {
      technicalImageWrites.push(
        saveRidgeEnhancedImage({
          pixels: analysis.orientationPixels,
          width: result.width,
          height: result.height,
          outputImageUri: orientationImageUri,
        }).then((savedUri) => {
          savedOrientationImageUri = savedUri;
        })
      );
    }

    // Enhancement yalnızca ridge kanıtı bulunan alan varsa yazılır; boş teknik çıktı galeriye eklenmez.
    if (analysis.enhancedPixels && analysis.enhancementSupportedAreaRatio > 0.02) {
      technicalImageWrites.push(
        saveRidgeEnhancedImage({
          pixels: analysis.enhancedPixels,
          width: result.width,
          height: result.height,
          outputImageUri: enhancedImageUri,
        }).then((savedUri) => {
          savedEnhancedImageUri = savedUri;
        })
      );
    }

    // Minutiae doğrulama görselini diğer teknik çıktılarla aynı anda kaydederiz.
    if (minutiaeVisualization) {
      technicalImageWrites.push(
        saveRidgeEnhancedImage({
          pixels: minutiaeVisualization,
          width: result.width,
          height: result.height,
          outputImageUri: minutiaeImageUri,
        }).then((savedUri) => {
          savedMinutiaeImageUri = savedUri;
        })
      );
    }
    // Native dosya yazımları birbirini beklemeden ilerler; Promise.all iki JPEG çıktısının tamamlandığını garanti eder.
    await Promise.all(technicalImageWrites);
    const technicalWriteMs = Date.now() - technicalWriteStartedAt;
    const canonicalResidualDegrees = normalizeAxisRotation(
      result.silhouetteAxisDegrees - 90
    );

    // Saha kalibrasyonunda farklı çekimleri karşılaştırmak için yalnızca terminale kısa kalite özeti yazar.
    const biometricRejections =
      quality.biometricRejectionReasons?.join('+') || 'yok';
    console.info(
      `[ROI kalite] parmak=${className}, çekim=${quality.captureStatus}, biyometri=${quality.biometricStatus}, biyometri_ret=${biometricRejections}, genel=${quality.globalScore}, çözünürlük=${quality.sourceResolutionScore}, ridge=${quality.ridgePeriodicity}, frekans=${quality.ridgeFrequencyConsistency}, yön=${quality.orientationCoherence}, güvenilir_yön=${quality.orientationReliableBlockRatio}%, ince_yön=${quality.orientationDetailBlockSize ?? 0}px/${quality.orientationDetailVerifiedRatio ?? 0}%, yön_düzeltme=${quality.orientationMedianCorrectionDegrees.toFixed(1)}°, obb_dönüş=${canonicalRotationDegrees.toFixed(1)}°, silüet_ekseni=${result.silhouetteAxisDegrees.toFixed(1)}°, dikey_sapma=${canonicalResidualDegrees.toFixed(1)}°, ölçek=${quality.ridgeAnalysisBlockSize ?? 0}px/${quality.ridgeAnalyzedScaleCount ?? 1}, iyileştirme=${quality.ridgeEnhancementGainPercent.toFixed(1)}%, gabor_alanı=${quality.ridgeEnhancementSupportedAreaRatio}%, yön_adayı=${quality.ridgeOrientationCandidateRatio ?? 0}%, kanıt=${quality.ridgeValidBlockCount}/${quality.ridgeCandidateBlockCount} (${quality.ridgeValidBlockRatio}%, gereken=${quality.biometricRequiredValidBlockCount ?? 0}/22%), iç_blok=${quality.ridgeInteriorBlockCount}/${quality.ridgeOrientationBlockCount}, periyot=${quality.ridgeMedianPeriodPixels.toFixed(1)}px [${quality.ridgePeriodHistogram}], frekans_ret=${quality.ridgeRejectionSummary}`
    );

    return {
      segmentedImageUri: result.imageUri,
      enhancedImageUri: savedEnhancedImageUri,
      orientationImageUri: savedOrientationImageUri,
      minutiaeImageUri: savedMinutiaeImageUri,
      minutiaeTemplate,
      silhouetteAxisDegrees: result.silhouetteAxisDegrees,
      canonicalResidualDegrees,
      quality,
      timings: {
        ...result.timings,
        qualityMs,
        technicalWriteMs,
      },
    };
  } catch (error) {
    console.warn('Parmak ROI segmentasyonu uygulanamadı.', error);
    return { quality: createSegmentationFailureQuality() };
  }
}

// Teknik aday listesini çekim kalitesi, kesintisiz arama alanı ve aday sayısıyla template kabulünden ayırır.
function assessMinutiaeTemplate(
  quality: FingerprintQuality,
  minutiae: MinutiaeExtractionResult
): {
  status: 'sufficient' | 'insufficient';
  reason?: MinutiaeRejectionReason;
} {
  if (quality.biometricStatus !== 'sufficient') {
    return { status: 'insufficient', reason: 'capture-quality' };
  }
  if (
    minutiae.searchableAreaRatio < MIN_SEARCHABLE_AREA_RATIO ||
    minutiae.largestSearchableRegionRatio <
      MIN_LARGEST_SEARCHABLE_REGION_RATIO
  ) {
    return { status: 'insufficient', reason: 'search-area' };
  }
  if (minutiae.template.minutiae.length < MIN_TEMPLATE_MINUTIAE_COUNT) {
    return { status: 'insufficient', reason: 'candidate-count' };
  }
  if (minutiae.template.minutiae.length > MAX_TEMPLATE_MINUTIAE_COUNT) {
    return { status: 'insufficient', reason: 'candidate-overflow' };
  }
  return { status: 'sufficient' };
}

// OBB'nin uzun kenarını dikey eksene getirip yalnızca gerçek açılı kutu boyutunu saklar.
async function createCanonicalFingerRoi(cropped: NitroImage, roiPlan: FingerRoi) {
  const geometry = getCanonicalGeometry(roiPlan.maskPolygon);
  if (!geometry) {
    return {
      image: cropped,
      sourcePixelWidth: Math.min(cropped.width, cropped.height),
      rotationDegrees: 0,
    };
  }

  const rotated = await cropped.rotateAsync(geometry.rotationDegrees, false);
  const targetWidth = Math.min(Math.max(Math.round(geometry.shortEdge), 1), rotated.width);
  const targetHeight = Math.min(Math.max(Math.round(geometry.longEdge), 1), rotated.height);
  const left = Math.max(0, Math.round((rotated.width - targetWidth) / 2));
  const top = Math.max(0, Math.round((rotated.height - targetHeight) / 2));
  const canonical = await rotated.cropAsync(
    left,
    top,
    Math.min(left + targetWidth, rotated.width),
    Math.min(top + targetHeight, rotated.height)
  );

  return {
    image: canonical,
    sourcePixelWidth: geometry.shortEdge,
    rotationDegrees: geometry.rotationDegrees,
  };
}

// OBB köşelerinden uzun parmak eksenini ve dikleştirme için gereken dönüş açısını hesaplar.
function getCanonicalGeometry(points?: { x: number; y: number }[]) {
  if (!points || points.length !== 4) return null;

  const edges = points.map((point, index) => {
    const next = points[(index + 1) % points.length];
    const deltaX = next.x - point.x;
    const deltaY = next.y - point.y;
    return {
      length: Math.hypot(deltaX, deltaY),
      angleDegrees: (Math.atan2(deltaY, deltaX) * 180) / Math.PI,
    };
  });
  const longEdge = [...edges].sort((first, second) => second.length - first.length)[0];
  const shortEdge = [...edges].sort((first, second) => first.length - second.length)[0];

  if (!longEdge || !shortEdge || shortEdge.length < 1) return null;

  return {
    longEdge: longEdge.length,
    shortEdge: shortEdge.length,
    rotationDegrees: normalizeAxisRotation(90 - longEdge.angleDegrees),
  };
}

// Çizgi ekseni 180 derece simetrik olduğu için en kısa eşdeğer dönüşü seçer.
function normalizeAxisRotation(degrees: number) {
  let normalized = degrees;
  while (normalized > 90) normalized -= 180;
  while (normalized < -90) normalized += 180;
  return normalized;
}

// Nitro Image'in beklediği yerel dosya yolundan file URI önekini kaldırır.
function toNativeFilePath(uri: string) {
  return decodeURIComponent(uri.replace(/^file:\/\//, ''));
}

// Nitro Image'in geçici dosya yolunu Expo FileSystem'in kullanacağı URI'ye dönüştürür.
function toFileUri(path: string) {
  return path.startsWith('file://') ? path : `file://${path}`;
}

// Tek tespitin OBB köşelerinden yalnızca model kutusunu çevreleyen ROI kırpma alanını hesaplar.
function createFingerRoiPlan(
  detection: DetectedObbBox,
  imageSize: ImageSize
): FingerRoi | null {
  const axisBox = getNormalizedAxisAlignedBox(detection);
  const boxWidth = axisBox.right - axisBox.left;
  const boxHeight = axisBox.bottom - axisBox.top;

  // Geçersiz veya sıfıra yakın kutuları kayda sokmadan atlarız.
  if (boxWidth <= 0 || boxHeight <= 0) {
    return null;
  }

  const normalizedCrop = normalizeCrop({
    x: axisBox.left,
    y: axisBox.top,
    width: boxWidth,
    height: boxHeight,
  });
  const pixelCrop = toPixelCrop(normalizedCrop, imageSize);

  // Dosya olarak anlamlı olmayacak kadar küçük ROI planlarını erken eleriz.
  if (pixelCrop.width < MIN_ROI_PIXEL_SIZE || pixelCrop.height < MIN_ROI_PIXEL_SIZE) {
    return null;
  }

  return {
    id: `roi-${detection.className}`,
    detectionId: detection.id,
    className: detection.className,
    confidence: detection.confidence,
    maskPolygon: createCropRelativePolygon(detection, pixelCrop, imageSize),
    normalizedMaskPolygon: createNormalizedCropRelativePolygon(detection, pixelCrop, imageSize),
    crop: {
      ...pixelCrop,
      normalized: normalizedCrop,
    },
  };
}

// OBB köşelerini kırpılmış ROI dosyasının kendi piksel koordinat sistemine taşır.
function createCropRelativePolygon(
  detection: DetectedObbBox,
  crop: { originX: number; originY: number; width: number; height: number },
  imageSize: ImageSize
) {
  return detection.points.map((point) => ({
    x: point.x * imageSize.width - crop.originX,
    y: point.y * imageSize.height - crop.originY,
  }));
}

// OBB köşelerini kırpılmış ROI dosyasının 0-1 oranlı koordinat sistemine taşır.
function createNormalizedCropRelativePolygon(
  detection: DetectedObbBox,
  crop: { originX: number; originY: number; width: number; height: number },
  imageSize: ImageSize
) {
  return detection.points.map((point) => ({
    x: clamp01((point.x * imageSize.width - crop.originX) / Math.max(crop.width, 1)),
    y: clamp01((point.y * imageSize.height - crop.originY) / Math.max(crop.height, 1)),
  }));
}

// Normalize OBB noktalarından kırpma hesabına uygun eksene hizalı çevre kutusunu çıkarır.
function getNormalizedAxisAlignedBox(detection: DetectedObbBox) {
  const xValues = detection.points.map((point) => point.x);
  const yValues = detection.points.map((point) => point.y);

  return {
    left: Math.min(...xValues),
    top: Math.min(...yValues),
    right: Math.max(...xValues),
    bottom: Math.max(...yValues),
  };
}

// ROI planını görüntü sınırları içinde kalacak şekilde 0-1 aralığına sıkıştırır.
function normalizeCrop(crop: { x: number; y: number; width: number; height: number }) {
  const left = clamp01(crop.x);
  const top = clamp01(crop.y);
  const right = clamp01(crop.x + crop.width);
  const bottom = clamp01(crop.y + crop.height);

  return {
    x: left,
    y: top,
    width: Math.max(0, right - left),
    height: Math.max(0, bottom - top),
  };
}

// Normalize kırpma alanını ImageManipulator'ın beklediği piksel koordinatlarına dönüştürür.
function toPixelCrop(
  crop: { x: number; y: number; width: number; height: number },
  imageSize: ImageSize
) {
  const originX = Math.round(crop.x * imageSize.width);
  const originY = Math.round(crop.y * imageSize.height);
  const right = Math.round((crop.x + crop.width) * imageSize.width);
  const bottom = Math.round((crop.y + crop.height) * imageSize.height);

  return {
    originX,
    originY,
    width: Math.max(0, Math.min(imageSize.width - originX, right - originX)),
    height: Math.max(0, Math.min(imageSize.height - originY, bottom - originY)),
  };
}

// Değeri görüntü oranı olarak güvenli 0-1 aralığında tutar.
function clamp01(value: number) {
  return Math.min(Math.max(value, 0), 1);
}
