import { manipulateAsync, SaveFormat } from 'expo-image-manipulator';

const ROI_WIDTH_RATIO = 0.42;
const ROI_ASPECT_RATIO = 0.62;
const ROI_VERTICAL_CENTER_RATIO = 0.46;

export type CropResult = {
  uri: string;
  width: number;
  height: number;
  crop: {
    originX: number;
    originY: number;
    width: number;
    height: number;
  };
};

export async function cropFingerRoi({
  imageUri,
  imageWidth,
  imageHeight,
}: {
  imageUri: string;
  imageWidth: number;
  imageHeight: number;
}): Promise<CropResult> {
  const width = Math.round(imageWidth * ROI_WIDTH_RATIO);
  const height = Math.round(width / ROI_ASPECT_RATIO);
  const safeHeight = Math.min(height, Math.round(imageHeight * 0.82));
  const originX = Math.max(0, Math.round((imageWidth - width) / 2));
  const originY = clamp(
    Math.round(imageHeight * ROI_VERTICAL_CENTER_RATIO - safeHeight / 2),
    0,
    imageHeight - safeHeight
  );

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
    }
  );

  return {
    uri: result.uri,
    width: result.width,
    height: result.height,
    crop: {
      originX,
      originY,
      width,
      height: safeHeight,
    },
  };
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}
