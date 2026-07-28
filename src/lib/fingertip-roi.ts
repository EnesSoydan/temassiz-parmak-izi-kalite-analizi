import { manipulateAsync, SaveFormat } from 'expo-image-manipulator';

import {
  createSegmentedFingerRoiImageUri,
  saveFingerRoiImage,
} from '@/lib/capture-storage';
import { segmentFingerRoiImage } from '@/lib/finger-segmentation';
import {
  analyzeFingerprintQuality,
  createSegmentationFailureQuality,
} from '@/lib/fingerprint-quality';
import type { CaptureSample, DetectedObbBox, FingerRoi } from '@/types/biometrics';

type ImageSize = NonNullable<CaptureSample['rawImageSize']>;

// Çok küçük hatalı kırpmaların dosya üretmesini engelleyen minimum piksel boyutu.
const MIN_ROI_PIXEL_SIZE = 16;

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
  const fingerRois: FingerRoi[] = [];

  for (const roiPlan of roiPlans) {
    // Expo ImageManipulator ham fotoğrafı bozmadan geçici kırpılmış ROI dosyası üretir.
    const cropped = await manipulateAsync(
      imageUri,
      [
        {
          crop: {
            originX: roiPlan.crop.originX,
            originY: roiPlan.crop.originY,
            width: roiPlan.crop.width,
            height: roiPlan.crop.height,
          },
        },
      ],
      {
        compress: 0.95,
        format: SaveFormat.JPEG,
      }
    );

    // Geçici ROI dosyasını uygulamanın kalıcı capture klasörüne taşınabilir isimle kopyalar.
    const savedImageUri = await saveFingerRoiImage(cropped.uri, sampleId, roiPlan.className);
    const processedRoi = await tryProcessFingerRoi({
      roiImageUri: savedImageUri,
      sampleId,
      className: roiPlan.className,
      sourceWidth: cropped.width,
    });

    fingerRois.push({
      ...roiPlan,
      imageUri: savedImageUri,
      segmentedImageUri: processedRoi.segmentedImageUri,
      quality: processedRoi.quality,
    });
  }

  return fingerRois;
}

// Segmentasyon başarısız olursa normal ROI kaydını bozmadan boş sonuç döndürür.
async function tryProcessFingerRoi({
  roiImageUri,
  sampleId,
  className,
  sourceWidth,
}: {
  roiImageUri: string;
  sampleId: string;
  className: FingerRoi['className'];
  sourceWidth: number;
}) {
  try {
    const outputImageUri = await createSegmentedFingerRoiImageUri(sampleId, className);
    const result = await segmentFingerRoiImage({ roiImageUri, outputImageUri, sourceWidth });

    if (!result) {
      return { quality: createSegmentationFailureQuality() };
    }

    return {
      segmentedImageUri: result.imageUri,
      quality: analyzeFingerprintQuality(result),
    };
  } catch (error) {
    console.warn('Parmak ROI segmentasyonu uygulanamadı.', error);
    return { quality: createSegmentationFailureQuality() };
  }
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
