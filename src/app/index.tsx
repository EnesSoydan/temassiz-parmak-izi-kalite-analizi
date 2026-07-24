import * as Device from 'expo-device';
import { useFocusEffect } from 'expo-router';
import { useCallback, useEffect, useRef, useState } from 'react';
import { LayoutChangeEvent, Pressable, StyleSheet, View } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import Svg, { G, Rect, Text as SvgText } from 'react-native-svg';
import {
  Camera as VisionCamera,
  HybridFrameConverter,
  type CameraRef as VisionCameraRef,
  useCameraPermission,
  useFrameOutput,
  usePhotoOutput,
} from 'react-native-vision-camera';
import { scheduleOnRN } from 'react-native-worklets';

import { ThemedText } from '@/components/themed-text';
import { BottomTabInset, Spacing } from '@/constants/theme';
import { appendCaptureSample, saveRawImage } from '@/lib/capture-storage';
import {
  detectFingertipObbBoxes,
  detectFingertipObbBoxesFromRawPixels,
  FINGERTIP_MODEL_SIZE,
  type ModelPixelFormat,
} from '@/lib/fingertip-detection';
import { loadFingertipObbSession } from '@/lib/onnx-model';
import type { CaptureSample, DetectedObbBox } from '@/types/biometrics';

type CaptureMode = 'photo' | 'realtime';

// Canlı model kutularını kamera önizlemesine taşırken kaynak frame oranını korur.
type LiveDetectionFrame = {
  width: number;
  height: number;
};

// Kamera üstündeki canlı kutu katmanının ihtiyaç duyduğu verileri taşır.
type LiveDetectionOverlayProps = {
  detections: DetectedObbBox[];
  imageSize: LiveDetectionFrame | null;
};

// Canlı model sonucunu kareler arasında kısa süre koruyan takip kaydını tanımlar.
type LiveDetectionTrack = {
  detection: DetectedObbBox;
  missedUpdates: number;
};

// Mobil model için yeterli ayrıntıyı koruyan canlı kamera çözünürlüğü.
const LIVE_FRAME_RESOLUTION = { width: 640, height: 480 };

// Canlı model yükünü sınırlandırmak için her altıncı frame'i inference adayı yapar.
const LIVE_MODEL_FRAME_INTERVAL = 6;

// Güncel kutuya daha yüksek ağırlık vererek gecikmeyi artırmadan titreşimi azaltır.
const LIVE_TRACK_CURRENT_WEIGHT = 0.9;

// Bir tur kaçırılan parmak kutusunu kısa süre ekranda tutar.
const LIVE_TRACK_MAX_MISSED_UPDATES = 1;

// Güncel model sonuçlarını sınıf kimliğiyle önceki kutulara bağlar.
function updateLiveDetectionTracks(
  previousTracks: Map<DetectedObbBox['className'], LiveDetectionTrack>,
  currentDetections: DetectedObbBox[]
) {
  const nextTracks = new Map<DetectedObbBox['className'], LiveDetectionTrack>();

  // Her parmak sınıfını önceki konumuyla yumuşatıp kararlı bir çizim kimliği üretir.
  for (const currentDetection of currentDetections) {
    const previousTrack = previousTracks.get(currentDetection.className);
    const detection = previousTrack
      ? interpolateDetection(previousTrack.detection, currentDetection, LIVE_TRACK_CURRENT_WEIGHT)
      : {
          ...currentDetection,
          id: `live-${currentDetection.className}`,
        };

    nextTracks.set(currentDetection.className, {
      detection,
      missedUpdates: 0,
    });
  }

  // Modelin anlık kaçırdığı kutuları sınırlı sayıda güncelleme boyunca korur.
  for (const [className, previousTrack] of previousTracks) {
    if (nextTracks.has(className)) continue;

    const missedUpdates = previousTrack.missedUpdates + 1;
    if (missedUpdates <= LIVE_TRACK_MAX_MISSED_UPDATES) {
      nextTracks.set(className, {
        detection: previousTrack.detection,
        missedUpdates,
      });
    }
  }

  return nextTracks;
}

// Aynı parmağa ait eski ve yeni kutunun koordinatlarını birbirine yaklaştırır.
function interpolateDetection(
  previous: DetectedObbBox,
  current: DetectedObbBox,
  currentWeight: number
): DetectedObbBox {
  const previousWeight = 1 - currentWeight;

  return {
    ...current,
    id: `live-${current.className}`,
    confidence: previous.confidence * previousWeight + current.confidence * currentWeight,
    center: {
      x: previous.center.x * previousWeight + current.center.x * currentWeight,
      y: previous.center.y * previousWeight + current.center.y * currentWeight,
    },
    size: {
      width: previous.size.width * previousWeight + current.size.width * currentWeight,
      height: previous.size.height * previousWeight + current.size.height * currentWeight,
    },
    angle: previous.angle * previousWeight + current.angle * currentWeight,
    points: current.points.map((point, index) => ({
      x: (previous.points[index]?.x ?? point.x) * previousWeight + point.x * currentWeight,
      y: (previous.points[index]?.y ?? point.y) * previousWeight + point.y * currentWeight,
    })),
  };
}

// Canlı sonuçları kamera görüntüsü üzerinde eksene hizalı kutular olarak çizer.
function LiveDetectionOverlay({ detections, imageSize }: LiveDetectionOverlayProps) {
  const [layout, setLayout] = useState({ width: 0, height: 0 });
  const imageFrame = getCoveredImageFrame(layout, imageSize);

  // Kamera önizlemesinin gerçek boyutunu kutu koordinat hesabı için saklar.
  function handleLayout(event: LayoutChangeEvent) {
    setLayout({
      width: event.nativeEvent.layout.width,
      height: event.nativeEvent.layout.height,
    });
  }

  return (
    <View pointerEvents="none" style={styles.liveDetectionOverlay} onLayout={handleLayout}>
      {layout.width > 0 && layout.height > 0 && (
        <Svg width={layout.width} height={layout.height}>
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

// Normalize kutu noktalarını kamera alanındaki eksene hizalı koordinatlara çevirir.
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

// resizeMode="cover" ile taşan kamera görüntüsünün gerçek çizim alanını hesaplar.
function getCoveredImageFrame(
  layout: { width: number; height: number },
  imageSize: LiveDetectionFrame | null
) {
  if (!imageSize || layout.width === 0 || layout.height === 0) {
    return { x: 0, y: 0, width: layout.width, height: layout.height };
  }

  const imageRatio = imageSize.width / imageSize.height;
  const layoutRatio = layout.width / layout.height;

  // Yatay taşmada kameranın ekran dışında kalan sağ ve sol bölümünü hesaba katar.
  if (imageRatio > layoutRatio) {
    const width = layout.height * imageRatio;
    return {
      x: (layout.width - width) / 2,
      y: 0,
      width,
      height: layout.height,
    };
  }

  const height = layout.width / imageRatio;
  return {
    x: 0,
    y: (layout.height - height) / 2,
    width: layout.width,
    height,
  };
}

// Model sınıf adlarını kamera üzerinde kısa Türkçe etiketlere çevirir.
function formatDetectionClass(className: DetectedObbBox['className']) {
  if (className === 'index') return 'işaret';
  if (className === 'middle') return 'orta';
  if (className === 'pinky') return 'serçe';
  if (className === 'ring') return 'yüzük';
  return 'tespit';
}

// Yerel capture kayıtları için kısa ve çakışma ihtimali düşük kimlik üretir.
function createId(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

// Model hatasının fotoğraf dosyasının kaydedilmesini engellememesi için inference'ı güvenli çalıştırır.
async function tryDetectFingertips(imageUri: string) {
  try {
    return await detectFingertipObbBoxes(imageUri);
  } catch (error) {
    console.warn('Parmak ucu modeli çalıştırılamadı.', error);
    return [];
  }
}

export default function CameraScreen() {
  const cameraRef = useRef<VisionCameraRef>(null);
  const permissionRequestedRef = useRef(false);
  const feedbackTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const liveDetectionBusyRef = useRef(false);
  const liveDetectionPausedRef = useRef(true);
  const liveDetectionTracksRef = useRef(
    new Map<DetectedObbBox['className'], LiveDetectionTrack>()
  );
  const [captureMode, setCaptureMode] = useState<CaptureMode>('photo');
  const [isScreenFocused, setIsScreenFocused] = useState(true);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [isTakingPhoto, setIsTakingPhoto] = useState(false);
  const [liveDetections, setLiveDetections] = useState<DetectedObbBox[]>([]);
  const [liveDetectionFrame, setLiveDetectionFrame] = useState<LiveDetectionFrame | null>(null);
  const [modelStatus, setModelStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [feedback, setFeedback] = useState('');
  const { hasPermission, requestPermission } = useCameraPermission();
  const photoOutput = usePhotoOutput({ quality: 0.95, qualityPrioritization: 'speed' });
  const isRealtimeMode = captureMode === 'realtime';
  const canUseCamera = hasPermission && isScreenFocused;

  // Kısa durum mesajını gösterip başarılı işlemlerde otomatik olarak temizler.
  const showFeedback = useCallback((message: string, autoClear = true) => {
    if (feedbackTimerRef.current) {
      clearTimeout(feedbackTimerRef.current);
    }

    setFeedback(message);
    if (autoClear) {
      feedbackTimerRef.current = setTimeout(() => setFeedback(''), 1800);
    }
  }, []);

  // Worklet'ten gelen mobil boyutlu görüntüyü ONNX modelinde tek sefer çalıştırır.
  const processLiveModelFrame = useCallback(
    (
      buffer: ArrayBuffer,
      width: number,
      height: number,
      pixelFormat: ModelPixelFormat,
      sourceWidth: number,
      sourceHeight: number
    ) => {
      if (liveDetectionBusyRef.current || liveDetectionPausedRef.current) {
        return;
      }

      liveDetectionBusyRef.current = true;
      detectFingertipObbBoxesFromRawPixels(buffer, width, height, pixelFormat)
        .then((detections) => {
          const nextTracks = updateLiveDetectionTracks(liveDetectionTracksRef.current, detections);
          liveDetectionTracksRef.current = nextTracks;
          setLiveDetections([...nextTracks.values()].map((track) => track.detection));
          setLiveDetectionFrame({ width: sourceWidth, height: sourceHeight });
        })
        .catch((error) => {
          liveDetectionPausedRef.current = true;
          console.warn('Canlı parmak ucu modeli çalıştırılamadı.', error);
        })
        .finally(() => {
          liveDetectionBusyRef.current = false;
        });
    },
    []
  );

  // Frame dönüşüm hatasında canlı modeli durdurup tekrar eden uyarıları engeller.
  const reportLiveFrameError = useCallback((message: string) => {
    liveDetectionPausedRef.current = true;
    console.warn('Canlı kamera frame’i hazırlanamadı.', message);
  }, []);

  const frameOutput = useFrameOutput({
    targetResolution: LIVE_FRAME_RESOLUTION,
    pixelFormat: 'yuv',
    dropFramesWhileBusy: true,
    enablePreviewSizedOutputBuffers: true,
    enablePhysicalBufferRotation: true,
    onFrame(frame) {
      'worklet';

      // Canlı takip sayacını ve kalıcı dönüşüm hatasını worklet tarafında saklar.
      const globalFrameState = globalThis as typeof globalThis & {
        __fingerFrameCounter?: number;
        __fingerLiveFrameFailed?: boolean;
      };
      const nextCount = (globalFrameState.__fingerFrameCounter ?? 0) + 1;
      globalFrameState.__fingerFrameCounter = nextCount;

      try {
        // Belirlenen aralıktaki frame'i model boyutuna getirip JavaScript tarafına yollar.
        if (nextCount % LIVE_MODEL_FRAME_INTERVAL === 0 && !globalFrameState.__fingerLiveFrameFailed) {
          const sourceWidth = frame.width;
          const sourceHeight = frame.height;
          const image = HybridFrameConverter.convertFrameToImage(frame);

          try {
            const resizedImage = image.resize(FINGERTIP_MODEL_SIZE, FINGERTIP_MODEL_SIZE);

            try {
              const rawPixels = resizedImage.toRawPixelData();
              scheduleOnRN(
                processLiveModelFrame,
                rawPixels.buffer,
                rawPixels.width,
                rawPixels.height,
                rawPixels.pixelFormat as ModelPixelFormat,
                sourceWidth,
                sourceHeight
              );
            } finally {
              resizedImage.dispose();
            }
          } finally {
            image.dispose();
          }
        }
      } catch (error) {
        globalFrameState.__fingerLiveFrameFailed = true;
        scheduleOnRN(reportLiveFrameError, String(error));
      } finally {
        // GPU tampon havuzunun dolmaması için her frame'i mutlaka serbest bırakır.
        frame.dispose();
      }
    },
  });

  // Uygulama ilk açıldığında kamera iznini bir kez otomatik ister.
  useEffect(() => {
    if (hasPermission || permissionRequestedRef.current) return;

    permissionRequestedRef.current = true;
    requestPermission().then((isGranted) => {
      if (!isGranted) {
        showFeedback('Kamerayı kullanmak için izin vermelisin.', false);
      }
    });
  }, [hasPermission, requestPermission, showFeedback]);

  // ONNX modelini ekran açılınca yükleyip fotoğraf ve canlı mod için hazırlar.
  useEffect(() => {
    let isMounted = true;

    loadFingertipObbSession()
      .then(() => {
        if (isMounted) setModelStatus('ready');
      })
      .catch((error) => {
        console.warn('ONNX parmak ucu modeli yüklenemedi.', error);
        if (isMounted) {
          setModelStatus('error');
          showFeedback('Model yüklenemedi.', false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, [showFeedback]);

  // Kamera sekmesi görünürken kamerayı çalıştırır, sekmeden çıkınca kaynağı serbest bırakır.
  useFocusEffect(
    useCallback(() => {
      setIsScreenFocused(true);
      return () => {
        setIsScreenFocused(false);
        setIsCameraReady(false);
        liveDetectionTracksRef.current.clear();
        setLiveDetections([]);
        setLiveDetectionFrame(null);
      };
    }, [])
  );

  // Ekran kapanırken bekleyen geri bildirim zamanlayıcısını temizler.
  useEffect(() => {
    return () => {
      if (feedbackTimerRef.current) {
        clearTimeout(feedbackTimerRef.current);
      }
    };
  }, []);

  // Fotoğraf ve canlı takip modları arasında geçiş yapıp eski takip kutularını temizler.
  function handleModeChange() {
    const nextMode: CaptureMode = captureMode === 'photo' ? 'realtime' : 'photo';
    liveDetectionPausedRef.current = nextMode === 'photo';
    liveDetectionTracksRef.current.clear();
    setLiveDetections([]);
    setLiveDetectionFrame(null);
    setCaptureMode(nextMode);
    setFeedback('');
  }

  // Kameradan tam el karesi alır, modeli çalıştırır ve kutu verileriyle yerel kayda ekler.
  async function handleTakePhoto() {
    if (!canUseCamera || !isCameraReady || isTakingPhoto) return;

    // Model henüz hazır değilse kutusuz kayıt üretmemek için çekimi kısa bir mesajla durdurur.
    if (modelStatus !== 'ready') {
      showFeedback(
        modelStatus === 'loading' ? 'Model hazırlanıyor.' : 'Model kullanılamıyor.',
        modelStatus === 'loading'
      );
      return;
    }

    setIsTakingPhoto(true);
    showFeedback('Fotoğraf işleniyor...', false);

    try {
      const photo = await photoOutput.capturePhoto(
        { flashMode: 'off', enableShutterSound: false },
        {}
      );
      const picturePath = await photo.saveToTemporaryFileAsync();
      const capturedImage = {
        uri: `file://${picturePath}`,
        width: photo.width,
        height: photo.height,
      };
      photo.dispose();

      const sampleId = createId('sample');
      const rawImageUri = await saveRawImage(capturedImage.uri, sampleId);
      const detections = await tryDetectFingertips(rawImageUri);

      // Kayıt ekranı, saklanan normalize kutuları ham görselin üzerinde yeniden çizer.
      const sample: CaptureSample = {
        id: sampleId,
        createdAt: new Date().toISOString(),
        rawImageUri,
        rawImageSize: {
          width: capturedImage.width,
          height: capturedImage.height,
        },
        detections,
        deviceModel: Device.modelName ?? 'Bilinmeyen cihaz',
        fingerLabel: 'unknown',
        sessionId: createId('session'),
        qualityStatus: 'unknown',
        accepted: false,
      };

      await appendCaptureSample(sample);
      showFeedback(
        detections.length > 0
          ? 'Fotoğraf kutularıyla birlikte kaydedildi.'
          : 'Fotoğraf kaydedildi, model parmak ucu bulamadı.'
      );
    } catch (error) {
      console.warn('Fotoğraf kaydedilemedi.', error);
      showFeedback('Fotoğraf kaydedilemedi. Tekrar dene.', false);
    } finally {
      setIsTakingPhoto(false);
    }
  }

  // Kamera izni reddedildiyse kullanıcıya yeniden isteme olanağı verir.
  async function handlePermissionRetry() {
    const isGranted = await requestPermission();
    if (!isGranted) {
      showFeedback('Kamera izni verilmedi.', false);
    }
  }

  return (
    <View style={styles.container}>
      {hasPermission ? (
        <VisionCamera
          ref={cameraRef}
          style={StyleSheet.absoluteFill}
          device="back"
          outputs={[photoOutput, frameOutput]}
          isActive={canUseCamera}
          orientationSource="interface"
          resizeMode="cover"
          enableNativeTapToFocusGesture
          onStarted={() => {
            setIsCameraReady(true);
            setFeedback('');
          }}
          onPreviewStopped={() => setIsCameraReady(false)}
          onError={(error) => {
            console.warn('Kamera oturumu hata verdi.', error);
            setIsCameraReady(false);
            showFeedback(`Kamera hatası: ${error.message}`, false);
          }}
        />
      ) : (
        <View style={styles.permissionState}>
          <ThemedText style={styles.permissionText}>Kamera izni gerekli</ThemedText>
          <Pressable style={styles.permissionButton} onPress={handlePermissionRetry}>
            <ThemedText type="smallBold" style={styles.permissionButtonText}>
              İzin ver
            </ThemedText>
          </Pressable>
        </View>
      )}

      {isRealtimeMode && (
        <LiveDetectionOverlay detections={liveDetections} imageSize={liveDetectionFrame} />
      )}

      <SafeAreaView pointerEvents="box-none" style={styles.controls}>
        <View style={styles.topControl}>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel={
              isRealtimeMode ? 'Fotoğraf moduna geç' : 'Canlı takip moduna geç'
            }
            hitSlop={16}
            onPress={handleModeChange}>
            <ThemedText type="smallBold" style={styles.modeText}>
              {isRealtimeMode ? 'Fotoğraf modu' : 'Canlı takip'}
            </ThemedText>
          </Pressable>
        </View>

        <View style={styles.bottomControl}>
          {feedback ? (
            <ThemedText type="small" style={styles.feedbackText}>
              {feedback}
            </ThemedText>
          ) : null}

          {!isRealtimeMode && (
            <Pressable
              accessibilityLabel="Fotoğraf çek"
              style={[
                styles.shutterButton,
                (!isCameraReady || isTakingPhoto) && styles.disabledButton,
              ]}
              disabled={!isCameraReady || isTakingPhoto}
              onPress={handleTakePhoto}>
              <View style={styles.shutterInner} />
            </Pressable>
          )}
        </View>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
  liveDetectionOverlay: {
    position: 'absolute',
    top: 0,
    right: 0,
    bottom: 0,
    left: 0,
    zIndex: 2,
  },
  controls: {
    position: 'absolute',
    top: 0,
    right: 0,
    bottom: 0,
    left: 0,
    zIndex: 3,
    justifyContent: 'space-between',
  },
  topControl: {
    minHeight: 56,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.28)',
  },
  modeText: {
    color: '#ffffff',
    textShadowColor: 'rgba(0, 0, 0, 0.75)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 3,
  },
  bottomControl: {
    minHeight: 132,
    alignItems: 'center',
    justifyContent: 'flex-end',
    gap: Spacing.two,
    paddingBottom: BottomTabInset + Spacing.three,
    backgroundColor: 'rgba(0, 0, 0, 0.28)',
  },
  feedbackText: {
    color: '#ffffff',
    textAlign: 'center',
    textShadowColor: 'rgba(0, 0, 0, 0.8)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 3,
    paddingHorizontal: Spacing.four,
  },
  shutterButton: {
    width: 72,
    height: 72,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 36,
    borderWidth: 4,
    borderColor: '#ffffff',
    backgroundColor: 'rgba(0, 0, 0, 0.22)',
  },
  shutterInner: {
    width: 54,
    height: 54,
    borderRadius: 27,
    backgroundColor: '#ffffff',
  },
  disabledButton: {
    opacity: 0.5,
  },
  permissionState: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.three,
    paddingHorizontal: Spacing.four,
  },
  permissionText: {
    color: '#ffffff',
  },
  permissionButton: {
    minHeight: 44,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 6,
    backgroundColor: '#ffffff',
    paddingHorizontal: Spacing.four,
  },
  permissionButtonText: {
    color: '#111111',
  },
});
