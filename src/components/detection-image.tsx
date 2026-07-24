import { Image } from 'expo-image';
import { useState } from 'react';
import { LayoutChangeEvent, StyleSheet, View } from 'react-native';
import Svg, { G, Rect, Text as SvgText } from 'react-native-svg';

import type { CaptureSample, DetectedObbBox } from '@/types/biometrics';

type DetectionImageProps = {
  imageUri: string;
  imageSize?: CaptureSample['rawImageSize'];
  detections?: DetectedObbBox[];
  height?: number;
};

// Kayıtlı el görüntüsünü, modelin ürettiği kutularla birlikte gösterir.
export function DetectionImage({
  imageUri,
  imageSize,
  detections = [],
  height = 380,
}: DetectionImageProps) {
  const [layout, setLayout] = useState({ width: 0, height: 0 });
  const imageFrame = getContainedImageFrame(layout, imageSize);

  // Görsel alanı değiştiğinde kutuların ekrandaki konumunu yeniden hesaplamak için ölçüyü saklar.
  function handleLayout(event: LayoutChangeEvent) {
    setLayout({
      width: event.nativeEvent.layout.width,
      height: event.nativeEvent.layout.height,
    });
  }

  return (
    <View style={[styles.frame, { height }]} onLayout={handleLayout}>
      <Image source={{ uri: imageUri }} style={styles.image} contentFit="contain" />

      {layout.width > 0 && layout.height > 0 && (
        <Svg pointerEvents="none" style={StyleSheet.absoluteFill}>
          {detections.map((detection) => {
            const box = getAxisAlignedScreenBox(detection, imageFrame);

            return (
              <G key={detection.id}>
                <Rect
                  x={box.x}
                  y={box.y}
                  width={box.width}
                  height={box.height}
                  fill="rgba(47, 209, 107, 0.10)"
                  stroke="#2FD16B"
                  strokeWidth={2}
                />
                <SvgText
                  x={box.x + box.width / 2}
                  y={Math.max(box.y - 4, 12)}
                  fill="#ffffff"
                  fontSize={9}
                  fontWeight="700"
                  textAnchor="middle">
                  {`${formatDetectionClass(detection.className)} ${Math.round(detection.confidence * 100)}%`}
                </SvgText>
              </G>
            );
          })}
        </Svg>
      )}
    </View>
  );
}

// Normalize OBB noktalarını ekrana taşıyıp sade, eksene hizalı bir kutuya dönüştürür.
function getAxisAlignedScreenBox(
  detection: DetectedObbBox,
  imageFrame: { x: number; y: number; width: number; height: number }
) {
  const screenPoints = detection.points.map((point) => ({
    x: imageFrame.x + point.x * imageFrame.width,
    y: imageFrame.y + point.y * imageFrame.height,
  }));
  const xValues = screenPoints.map((point) => point.x);
  const yValues = screenPoints.map((point) => point.y);
  const left = Math.min(...xValues);
  const top = Math.min(...yValues);
  const right = Math.max(...xValues);
  const bottom = Math.max(...yValues);

  return {
    x: left,
    y: top,
    width: right - left,
    height: bottom - top,
  };
}

// contentFit="contain" davranışındaki boşlukları hesaba katarak gerçek görsel alanını bulur.
function getContainedImageFrame(
  layout: { width: number; height: number },
  imageSize?: CaptureSample['rawImageSize']
) {
  if (!imageSize || layout.width === 0 || layout.height === 0) {
    return { x: 0, y: 0, width: layout.width, height: layout.height };
  }

  const imageRatio = imageSize.width / imageSize.height;
  const layoutRatio = layout.width / layout.height;

  // Yatay görselde üst ve altta oluşan boşlukları çizim hesabına ekler.
  if (imageRatio > layoutRatio) {
    const height = layout.width / imageRatio;
    return {
      x: 0,
      y: (layout.height - height) / 2,
      width: layout.width,
      height,
    };
  }

  const width = layout.height * imageRatio;
  return {
    x: (layout.width - width) / 2,
    y: 0,
    width,
    height: layout.height,
  };
}

// Model sınıf adlarını kayıt ekranında kısa Türkçe etiketlere çevirir.
function formatDetectionClass(className: DetectedObbBox['className']) {
  if (className === 'index') return 'işaret';
  if (className === 'middle') return 'orta';
  if (className === 'pinky') return 'serçe';
  if (className === 'ring') return 'yüzük';
  return 'tespit';
}

const styles = StyleSheet.create({
  frame: {
    width: '100%',
    overflow: 'hidden',
    borderRadius: 6,
    backgroundColor: '#000000',
  },
  image: {
    width: '100%',
    height: '100%',
  },
});
