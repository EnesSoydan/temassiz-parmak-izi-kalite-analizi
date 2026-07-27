import { manipulateAsync, SaveFormat } from 'expo-image-manipulator';

import { saveFingerRoiImage } from '@/lib/capture-storage';
import type { CaptureSample, DetectedObbBox, FingerRoi } from '@/types/biometrics';

type ImageSize = NonNullable<CaptureSample['rawImageSize']>;

// Parmak ucu kutusunun çevresinde ridge bölgesini de içerecek güvenli yatay payı belirler.
const ROI_HORIZONTAL_PADDING_RATIO = 0.75;

// Kutunun üst tarafında az, alt tarafında daha fazla alan bırakarak parmak yüzeyini korur.
const ROI_TOP_PADDING_RATIO = 0.45;
const ROI_BOTTOM_PADDING_RATIO = 1.2;

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
    fingerRois.push({ ...roiPlan, imageUri: savedImageUri });
  }

  return fingerRois;
}

// Tek tespitin OBB köşelerinden eksene hizalı, genişletilmiş ROI kırpma alanı hesaplar.
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

  const centerX = axisBox.left + boxWidth / 2;
  const expandedWidth = boxWidth * (1 + ROI_HORIZONTAL_PADDING_RATIO * 2);
  const expandedHeight = boxHeight * (1 + ROI_TOP_PADDING_RATIO + ROI_BOTTOM_PADDING_RATIO);
  const normalizedCrop = normalizeCrop({
    x: centerX - expandedWidth / 2,
    y: axisBox.top - boxHeight * ROI_TOP_PADDING_RATIO,
    width: expandedWidth,
    height: expandedHeight,
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
    crop: {
      ...pixelCrop,
      normalized: normalizedCrop,
    },
  };
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
