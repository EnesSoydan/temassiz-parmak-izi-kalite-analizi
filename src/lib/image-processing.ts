import { manipulateAsync, SaveFormat } from 'expo-image-manipulator';

// Ekrandaki ROI rehberiyle dosyaya kırpılan ROI'nin oranlarını aynı tutan temel ayarlar.
const ROI_WIDTH_RATIO = 0.42;
const ROI_ASPECT_RATIO = 0.62;
const ROI_VERTICAL_CENTER_RATIO = 0.46;

export type CropResult = {
  uri: string;
  width: number;
  height: number;
  base64?: string;
  crop: {
    originX: number;
    originY: number;
    width: number;
    height: number;
  };
};

// Tek parmak prototipi için görüntünün orta bölgesinden sabit oranlı ROI kırpar ve kalite hesabı için base64 üretir.
export async function cropFingerRoi({
  imageUri,
  imageWidth,
  imageHeight,
}: {
  imageUri: string;
  imageWidth: number;
  imageHeight: number;
}): Promise<CropResult> {
  // ROI genişliği ekran rehberiyle aynı oranda tutulur; yükseklik parmak formuna göre daha dikeydir.
  const width = Math.round(imageWidth * ROI_WIDTH_RATIO);
  const height = Math.round(width / ROI_ASPECT_RATIO);
  const safeHeight = Math.min(height, Math.round(imageHeight * 0.82));
  const originX = Math.max(0, Math.round((imageWidth - width) / 2));

  // Dikey merkez biraz yukarı alınır; kullanıcı parmak ucunu genellikle kutunun üst-orta kısmına hizalar.
  const originY = clamp(
    Math.round(imageHeight * ROI_VERTICAL_CENTER_RATIO - safeHeight / 2),
    0,
    imageHeight - safeHeight
  );

  // ROI dosyasını küçük ama analiz edilebilir boyuta indirir; ham görüntü ayrıca saklandığı için bilgi kaybolmaz.
  const result = await manipulateAsync(
    imageUri,
    [
      {
        crop: {
          originX,
          originY,
          width,
          height: safeHeight,
        },
      },
      { resize: { width: 512 } },
    ],
    {
      compress: 0.95,
      format: SaveFormat.JPEG,
      base64: true,
    }
  );

  return {
    uri: result.uri,
    width: result.width,
    height: result.height,
    base64: result.base64,
    crop: {
      originX,
      originY,
      width,
      height: safeHeight,
    },
  };
}

// ROI koordinatının görüntü sınırlarının dışına taşmasını engeller.
function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}
