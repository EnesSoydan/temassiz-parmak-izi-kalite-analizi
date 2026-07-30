import { Images, type Image as NitroImage } from 'react-native-nitro-image';

import {
  createEnhancedFingerRoiImageUri,
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
import { saveRidgeEnhancedImage } from '@/lib/ridge-enhancement';
import type { CaptureSample, DetectedObbBox, FingerRoi } from '@/types/biometrics';

type ImageSize = NonNullable<CaptureSample['rawImageSize']>;

// Çok küçük hatalı kırpmaların dosya üretmesini engelleyen minimum piksel boyutu.
const MIN_ROI_PIXEL_SIZE = 16;

// Üç işçi native beklemeleri paylaşırken dördüncü büyük piksel çalışma kopyasını açmayarak belleği sınırlar.
const ROI_PROCESSING_CONCURRENCY = 2;

// Canlı veya fotoğraf model kutularından parmak başına normalize ROI planı üretir.
export function createFingerRoiPlans(
  detections: DetectedObbBox[],
  imageSize: ImageSize
): FingerRoi[] {
  return detections
    .map((detection) => createFingerRoiPlan(detection, imageSize))
    .filter((roi): roi is FingerRoi => roi !== null);
}

// Ham görselden model tespitlerine göre her parmak için ayrı ROI JPEG dosyası çıkarır.
export async function extractFingerRoisFromImage({
  imageUri,
  imageSize,
  detections,
  sampleId,
}: {
  imageUri: string;
  imageSize: ImageSize;
  detections: DetectedObbBox[];
  sampleId: string;
}) {
  const roiPlans = createFingerRoiPlans(detections, imageSize);
  const processingStartedAt = Date.now();
  // Kaynak fotoğrafı her parmak için tekrar çözmek yerine tek native görüntü nesnesi paylaşılır.
  const sourceImage = await Images.loadFromFileAsync(toNativeFilePath(imageUri));

  // İki işçi native kırpma ve dosya erişimini üst üste bindirir; üç işçi cihazda JS sıkışmasına yol açıyordu.
  const fingerRois = await mapWithConcurrency(
    roiPlans,
    ROI_PROCESSING_CONCURRENCY,
    async (roiPlan) => {
      const fingerStartedAt = Date.now();
      const cropStartedAt = Date.now();

      // Nitro Image aynı çözülmüş fotoğraftan eksene hizalı ham ROI'yi native tarafta kırpar.
      const cropped = await sourceImage.cropAsync(
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
          ? `[ROI süre] parmak=${roiPlan.className}, kırpma=${cropMs}ms, kayıt=${saveMs}ms, çözme=${timings.decodeMs}ms, maske=${timings.maskMs}ms, iyileştirme=${timings.enhancementMs}ms, jpeg=${timings.encodeMs}ms, yazma=${timings.writeMs}ms, kalite=${timings.qualityMs}ms, toplam=${Date.now() - fingerStartedAt}ms`
          : `[ROI süre] parmak=${roiPlan.className}, kırpma=${cropMs}ms, kayıt=${saveMs}ms, segmentasyon=başarısız, toplam=${Date.now() - fingerStartedAt}ms`
      );

      return {
        ...roiPlan,
        imageUri: savedImageUri,
        canonicalImageUri,
        segmentedImageUri: processedRoi.segmentedImageUri,
        enhancedImageUri: processedRoi.enhancedImageUri,
        orientationImageUri: processedRoi.orientationImageUri,
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

  return fingerRois;
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
    let savedEnhancedImageUri: string | undefined;
    let savedOrientationImageUri: string | undefined;

    // Yön haritası ridge kanıtı çıkmasa da orientation hesabını doğrulayabilmek için saklanır.
    if (analysis.orientationPixels) {
      savedOrientationImageUri = await saveRidgeEnhancedImage({
        pixels: analysis.orientationPixels,
        width: result.width,
        height: result.height,
        outputImageUri: orientationImageUri,
      });
    }

    // Enhancement yalnızca ridge kanıtı bulunan alan varsa yazılır; boş teknik çıktı galeriye eklenmez.
    if (analysis.enhancedPixels && analysis.enhancementSupportedAreaRatio > 0.02) {
      await saveRidgeEnhancedImage({
        pixels: analysis.enhancedPixels,
        width: result.width,
        height: result.height,
        outputImageUri: enhancedImageUri,
      });
      savedEnhancedImageUri = enhancedImageUri;
    }
    const qualityMs = Date.now() - qualityStartedAt;
    const canonicalResidualDegrees = normalizeAxisRotation(
      result.silhouetteAxisDegrees - 90
    );

    // Saha kalibrasyonunda farklı çekimleri karşılaştırmak için yalnızca terminale kısa kalite özeti yazar.
    const biometricRejections =
      quality.biometricRejectionReasons?.join('+') || 'yok';
    console.info(
      `[ROI kalite] parmak=${className}, çekim=${quality.captureStatus}, biyometri=${quality.biometricStatus}, biyometri_ret=${biometricRejections}, genel=${quality.globalScore}, çözünürlük=${quality.sourceResolutionScore}, ridge=${quality.ridgePeriodicity}, frekans=${quality.ridgeFrequencyConsistency}, yön=${quality.orientationCoherence}, güvenilir_yön=${quality.orientationReliableBlockRatio}%, yön_düzeltme=${quality.orientationMedianCorrectionDegrees.toFixed(1)}°, obb_dönüş=${canonicalRotationDegrees.toFixed(1)}°, silüet_ekseni=${result.silhouetteAxisDegrees.toFixed(1)}°, dikey_sapma=${canonicalResidualDegrees.toFixed(1)}°, ölçek=${quality.ridgeAnalysisBlockSize ?? 0}px/${quality.ridgeAnalyzedScaleCount ?? 1}, iyileştirme=${quality.ridgeEnhancementGainPercent.toFixed(1)}%, gabor_alanı=${quality.ridgeEnhancementSupportedAreaRatio}%, yön_adayı=${quality.ridgeOrientationCandidateRatio ?? 0}%, kanıt=${quality.ridgeValidBlockCount}/${quality.ridgeCandidateBlockCount} (${quality.ridgeValidBlockRatio}%, gereken=${quality.biometricRequiredValidBlockCount ?? 0}/22%), iç_blok=${quality.ridgeInteriorBlockCount}/${quality.ridgeOrientationBlockCount}, periyot=${quality.ridgeMedianPeriodPixels.toFixed(1)}px [${quality.ridgePeriodHistogram}], frekans_ret=${quality.ridgeRejectionSummary}`
    );

    return {
      segmentedImageUri: result.imageUri,
      enhancedImageUri: savedEnhancedImageUri,
      orientationImageUri: savedOrientationImageUri,
      silhouetteAxisDegrees: result.silhouetteAxisDegrees,
      canonicalResidualDegrees,
      quality,
      timings: {
        ...result.timings,
        qualityMs,
      },
    };
  } catch (error) {
    console.warn('Parmak ROI segmentasyonu uygulanamadı.', error);
    return { quality: createSegmentationFailureQuality() };
  }
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
