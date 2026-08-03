import * as Device from 'expo-device';
import { useFocusEffect } from 'expo-router';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { LayoutChangeEvent, Pressable, StyleSheet, View } from 'react-native';
import type { Image as NitroImage } from 'react-native-nitro-image';
import { SafeAreaView } from 'react-native-safe-area-context';
import Svg, { G, Rect, Text as SvgText } from 'react-native-svg';
import {
  Camera as VisionCamera,
  HybridFrameConverter,
  type CameraDevice,
  type CameraRef as VisionCameraRef,
  type Constraint,
  type MeteringMode,
  type Photo,
  useCameraDevice,
  useCameraPermission,
  useFrameOutput,
  usePhotoOutput,
} from 'react-native-vision-camera';
import { createSynchronizable, scheduleOnRN } from 'react-native-worklets';

import { ThemedText } from '@/components/themed-text';
import { BottomTabInset, Spacing } from '@/constants/theme';
import { appendCaptureSample, saveRawImage } from '@/lib/capture-storage';
import {
  detectFingertipObbBoxes,
  detectFingertipObbBoxesFromImage,
  detectFingertipObbBoxesFromRawPixels,
  FINGERTIP_MODEL_SIZE,
} from '@/lib/fingertip-detection';
import { extractFingerRoisFromImage } from '@/lib/fingertip-roi';
import {
  createCaptureQualityFeedback,
  getCaptureQualityStatus,
} from '@/lib/fingerprint-quality';
import { loadFingertipObbSession } from '@/lib/onnx-model';
import type { CaptureSample, DetectedObbBox } from '@/types/biometrics';

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

type PreparedCapturePhoto = {
  image: NitroImage;
  temporaryPath: string;
  preparationDurationMs: number;
};

// Mobil model için yeterli ayrıntıyı koruyan canlı kamera çözünürlüğü.
const LIVE_FRAME_RESOLUTION = { width: 640, height: 480 };

// Canlı model yükünü sınırlandırmak için her altıncı frame'i inference adayı yapar.
const LIVE_MODEL_FRAME_INTERVAL = 6;

// Güncel kutuya daha yüksek ağırlık vererek gecikmeyi artırmadan titreşimi azaltır.
const LIVE_TRACK_CURRENT_WEIGHT = 0.8;

// Anlık kaçırılan kutuyu iki model güncellemesi koruyup eski konumun iz bırakmasını sınırlar.
const LIVE_TRACK_MAX_MISSED_UPDATES = 2;

// Canlı kutu odağını kamerayı sürekli yeniden taramaya zorlamayacak aralıkta yeniler.
const LIVE_FOCUS_MIN_INTERVAL_MS = 900;

// Kutu grubu çok az hareket ettiğinde gereksiz odak çağrılarını sınırlar.
const LIVE_FOCUS_MIN_POINT_DISTANCE = 24;

// Sabit duran elde bile mesafe değişimini yakalamak için odağı belirli aralıkla tazeler.
const LIVE_FOCUS_FORCE_REFRESH_MS = 1800;

// Canlı odak zaten sürerken deklanşör kilidinin oturması için yalnızca kısa bir ek bekleme kullanılır.
const PRE_CAPTURE_FOCUS_DELAY_MS = 350;

// Bölgesel AE ölçümünü korurken sabit pozlama telafisi uygulamayız.
const ROI_EXPOSURE_COMPENSATION_INDEX = 0;

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

// Canlı kutuların tamamını kapsayan alanı kamera önizlemesinde 3A ölçüm bölgesine çevirir.
function getLiveMeteringRegion({
  detections,
  imageSize,
  layout,
}: {
  detections: DetectedObbBox[];
  imageSize: LiveDetectionFrame | null;
  layout: { width: number; height: number };
}) {
  if (layout.width <= 0 || layout.height <= 0 || detections.length === 0) {
    return {
      x: layout.width / 2,
      y: layout.height / 2,
      size: Math.max(Math.min(layout.width, layout.height) * 0.18, 1),
    };
  }

  const imageFrame = getCoveredImageFrame(layout, imageSize);
  const screenPoints = detections.flatMap((detection) =>
    detection.points.map((point) => ({
      x: point.x * imageFrame.width + imageFrame.x,
      y: point.y * imageFrame.height + imageFrame.y,
    }))
  );
  const xValues = screenPoints.map((point) => point.x);
  const yValues = screenPoints.map((point) => point.y);
  const left = Math.min(...xValues);
  const top = Math.min(...yValues);
  const right = Math.max(...xValues);
  const bottom = Math.max(...yValues);
  const shortestLayoutEdge = Math.min(layout.width, layout.height);

  return {
    x: clamp((left + right) / 2, 0, layout.width),
    y: clamp((top + bottom) / 2, 0, layout.height),
    size: clamp(
      Math.max(right - left, bottom - top) * 1.12,
      shortestLayoutEdge * 0.12,
      shortestLayoutEdge * 0.48
    ),
  };
}

// Cihazın desteklediği odak, pozlama ve beyaz dengesi ölçüm modlarını güvenli sırayla seçer.
function getSupportedMeteringModes(device: CameraDevice) {
  const modes: MeteringMode[] = [];
  if (device.supportsFocusMetering) modes.push('AF');
  if (device.supportsExposureMetering) modes.push('AE');
  if (device.supportsWhiteBalanceMetering) modes.push('AWB');
  return modes;
}

// Box bölgesini boyutuyla birlikte CameraX 3A ölçümüne gönderir.
async function meterCameraAtRegion({
  camera,
  device,
  region,
  adaptiveness,
}: {
  camera: VisionCameraRef;
  device: CameraDevice;
  region: { x: number; y: number; size: number };
  adaptiveness: 'continuous' | 'locked';
}) {
  const controller = camera.controller;
  const modes = getSupportedMeteringModes(device);
  if (!controller || modes.length === 0) return;

  const meteringPoint = camera.createMeteringPoint(
    region.x,
    region.y,
    region.size
  );
  await controller.focusTo(meteringPoint, {
    modes,
    responsiveness: adaptiveness === 'locked' ? 'snappy' : 'steady',
    adaptiveness,
    autoResetAfter: null,
  });
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

// Fotoğraf işlem aşamalarını karşılaştırmak için duvar saati milisaniyesini okur.
function getCurrentTimeMs() {
  return Date.now();
}

// Canlı model sonucuna göre kullanıcıya teknik değer göstermeden çekim duruşunu hatırlatır.
function createLiveCaptureGuide({
  detections,
  modelStatus,
  isCameraReady,
}: {
  detections: DetectedObbBox[];
  modelStatus: 'loading' | 'ready' | 'error';
  isCameraReady: boolean;
}) {
  if (!isCameraReady) return 'Kamera hazırlanıyor...';
  if (modelStatus === 'loading') return 'Model hazırlanıyor...';
  if (modelStatus === 'error') return 'Model kullanılamıyor.';

  const detectedClasses = new Set(
    detections
      .map((detection) => detection.className)
      .filter((className) => className !== 'unknown')
  );

  if (detectedClasses.size < 4) {
    return 'Dört parmak ucunu kadraja al.';
  }

  return 'Telefonu parmak yüzeyine paralel tut ve sabit kal.';
}

// Kısa kamera işlemlerinde akışı bloklamadan beklemek için kullanılır.
function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// İki önizleme noktası arasındaki ekran mesafesini hesaplar.
function getPointDistance(
  first: { x: number; y: number },
  second: { x: number; y: number }
) {
  return Math.hypot(first.x - second.x, first.y - second.y);
}

// Yeni bir 3A isteğinin önceki isteği iptal etmesini beklenen yenileme davranışı olarak tanır.
function isMeteringCancellation(error: unknown) {
  return String(error).includes('OperationCanceledException');
}

// Sayıyı güvenli şekilde verilen aralıkta tutar.
function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

// Kamera fotoğrafını yön bilgisi uygulanmış native görüntüye ve geçici JPEG yoluna dönüştürür.
async function prepareCapturedPhoto(
  photo: Photo,
  lighting: 'flaşsız' | 'flaşlı'
): Promise<PreparedCapturePhoto> {
  const preparationStartedAt = getCurrentTimeMs();

  try {
    const image = await photo.toImageAsync();
    // EXIF yönünü dosya pikseline uygularız; JPEG fallback ve ROI koordinatları aynı ekseni kullanır.
    const temporaryPath = await image.saveToTemporaryFileAsync('jpg', 95);
    console.info(
      `[Fotoğraf kaynak] ışık=${lighting}, sensör=${photo.width}x${photo.height}, yön=${photo.orientation}, işleme=${image.width}x${image.height}, ham=${photo.isRawPhoto}, hat=native`
    );
    return {
      image,
      temporaryPath,
      preparationDurationMs: getCurrentTimeMs() - preparationStartedAt,
    };
  } finally {
    photo.dispose();
  }
}

// Native model bir sınıfı kaçırırsa aynı sabit JPEG'i de tarayıp sınıf başına en güvenilir kutuyu tutar.
async function tryDetectFingertipsFromImage(
  image: NitroImage,
  fallbackImageUri: string
) {
  let nativeDetections: DetectedObbBox[] = [];

  try {
    nativeDetections = await detectFingertipObbBoxesFromImage(image);
    if (nativeDetections.length >= 4) return nativeDetections;
  } catch (error) {
    console.warn('Parmak ucu modeli native görüntüyle çalıştırılamadı.', error);
  }

  try {
    const jpegDetections = await detectFingertipObbBoxes(fallbackImageUri);
    return mergeBestDetections(nativeDetections, jpegDetections);
  } catch (fallbackError) {
    console.warn('Parmak ucu modeli JPEG yedeğiyle de çalıştırılamadı.', fallbackError);
    return nativeDetections;
  }
}

// Birden fazla model sonucundan her parmak sınıfının en güvenilir kutusunu seçer.
function mergeBestDetections(...groups: DetectedObbBox[][]) {
  const bestByClass = new Map<DetectedObbBox['className'], DetectedObbBox>();
  for (const detection of groups.flat()) {
    const current = bestByClass.get(detection.className);
    if (!current || detection.confidence > current.confidence) {
      bestByClass.set(detection.className, detection);
    }
  }
  return [...bestByClass.values()];
}

// Her pozlamadaki aynı sınıfı karşılaştırıp daha güvenilir kutunun kaynağını taşır.
function selectBestDetectionsAcrossExposure(
  ambientDetections: DetectedObbBox[],
  flashDetections: DetectedObbBox[]
) {
  const ambientByClass = new Map(
    ambientDetections.map((detection) => [detection.className, detection])
  );
  const flashByClass = new Map(
    flashDetections.map((detection) => [detection.className, detection])
  );
  const allClasses = new Set([
    ...ambientByClass.keys(),
    ...flashByClass.keys(),
  ]);
  const detections: DetectedObbBox[] = [];
  const sourceByClassName = new Map<
    DetectedObbBox['className'],
    'ambient' | 'flash'
  >();
  const selectedFromFlash: DetectedObbBox['className'][] = [];

  // Her parmak için en yüksek güveni seçmek, flaşta açılan serçe gibi durumları korur.
  for (const className of allClasses) {
    const ambient = ambientByClass.get(className);
    const flash = flashByClass.get(className);
    const selected = !ambient || (flash && flash.confidence > ambient.confidence)
      ? flash
      : ambient;

    if (!selected) continue;
    detections.push(selected);
    const source = selected === flash ? 'flash' : 'ambient';
    sourceByClassName.set(className, source);
    if (source === 'flash') {
      selectedFromFlash.push(className);
    }
  }

  return { detections, sourceByClassName, selectedFromFlash };
}

// Model sonuçlarını çift çekim logunda kısa ve okunabilir biçimde gösterir.
function formatDetectionSummary(detections: DetectedObbBox[]) {
  if (detections.length === 0) return 'boş';
  return detections
    .map((detection) => `${detection.className}:${detection.confidence.toFixed(2)}`)
    .join(',');
}

// ROI çıkarma hatası fotoğraf kaydını iptal etmesin diye çoklu parmak kırpmayı güvenli çalıştırır.
async function tryExtractFingerRois({
  imageUri,
  imageSize,
  detections,
  sampleId,
  sourceImage,
  sourceImagesByClassName,
  comparisonImagesByClassName,
  comparisonDetectionsByClassName,
}: {
  imageUri: string;
  imageSize: NonNullable<CaptureSample['rawImageSize']>;
  detections: DetectedObbBox[];
  sampleId: string;
  sourceImage?: NitroImage;
  sourceImagesByClassName?: ReadonlyMap<DetectedObbBox['className'], NitroImage>;
  comparisonImagesByClassName?: ReadonlyMap<
    DetectedObbBox['className'],
    NitroImage
  >;
  comparisonDetectionsByClassName?: ReadonlyMap<
    DetectedObbBox['className'],
    DetectedObbBox
  >;
}) {
  try {
    return await extractFingerRoisFromImage({
      imageUri,
      imageSize,
      detections,
      sampleId,
      sourceImage,
      sourceImagesByClassName,
      comparisonImagesByClassName,
      comparisonDetectionsByClassName,
    });
  } catch (error) {
    console.warn('Parmak ROI görselleri çıkarılamadı.', error);
    return [];
  }
}

export default function CameraScreen() {
  const cameraRef = useRef<VisionCameraRef>(null);
  const permissionRequestedRef = useRef(false);
  const feedbackTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const liveDetectionBusyRef = useRef(false);
  const liveDetectionPausedRef = useRef(true);
  const liveFocusBusyRef = useRef(false);
  const captureMeteringActiveRef = useRef(false);
  const lastLiveFocusAtRef = useRef(0);
  const lastLiveFocusPointRef = useRef<{ x: number; y: number } | null>(null);
  const liveDetectionTracksRef = useRef(
    new Map<DetectedObbBox['className'], LiveDetectionTrack>()
  );
  const liveInferenceBusy = useMemo(() => createSynchronizable(false), []);
  const liveFrameFailed = useMemo(() => createSynchronizable(false), []);
  const photoProcessingActive = useMemo(() => createSynchronizable(false), []);
  const [isScreenFocused, setIsScreenFocused] = useState(true);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [isTakingPhoto, setIsTakingPhoto] = useState(false);
  // Çift çekim deneyi varsayılan olarak açıktır; kullanıcı üst anahtardan tek çekime dönebilir.
  const [isDualCaptureEnabled, setIsDualCaptureEnabled] = useState(true);
  const [liveDetections, setLiveDetections] = useState<DetectedObbBox[]>([]);
  const [liveDetectionFrame, setLiveDetectionFrame] = useState<LiveDetectionFrame | null>(null);
  const [modelStatus, setModelStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [feedback, setFeedback] = useState('');
  const { hasPermission, requestPermission } = useCameraPermission();
  const closeUpCameraDevice = useCameraDevice('back', {
    physicalDevices: ['ultra-wide-angle'],
  });
  const photoOutput = usePhotoOutput({ quality: 1, qualityPrioritization: 'quality' });
  // Fotoğraf çözünürlüğünü canlı akıştan önde tutup mümkünse piksel birleştirmeyen sensör formatını isteriz.
  const cameraConstraints = useMemo<Constraint[]>(
    () => [{ binned: false }, { resolutionBias: photoOutput }],
    [photoOutput]
  );
  const canUseCamera = hasPermission && isScreenFocused;
  const [cameraLayout, setCameraLayout] = useState({ width: 0, height: 0 });
  const closeUpZoom =
    closeUpCameraDevice?.isVirtualDevice &&
    closeUpCameraDevice.physicalDevices.some((device) => device.type === 'ultra-wide-angle')
      ? closeUpCameraDevice.minZoom
      : 1;
  const activeCameraHasFlash = closeUpCameraDevice?.hasFlash === true;
  const liveCaptureGuide = createLiveCaptureGuide({
    detections: liveDetections,
    modelStatus,
    isCameraReady,
  });

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
    async (
      rawBuffer: ArrayBuffer,
      width: number,
      height: number,
      sourceWidth: number,
      sourceHeight: number
    ) => {
      if (liveDetectionBusyRef.current || liveDetectionPausedRef.current) {
        liveInferenceBusy.setBlocking(false);
        return;
      }

      liveDetectionBusyRef.current = true;

      try {
        // Android cihaz testinde doğru sonuç veren ham kanal sırası RGBA olarak doğrulandı.
        const detections = await detectFingertipObbBoxesFromRawPixels(
          rawBuffer,
          width,
          height,
          'RGBA'
        );

        const nextTracks = updateLiveDetectionTracks(liveDetectionTracksRef.current, detections);
        const trackedDetections = [...nextTracks.values()].map((track) => track.detection);
        liveDetectionTracksRef.current = nextTracks;
        setLiveDetections(trackedDetections);
        setLiveDetectionFrame({ width: sourceWidth, height: sourceHeight });
      } catch (error) {
        liveDetectionPausedRef.current = true;
        console.warn('Canlı parmak ucu modeli çalıştırılamadı.', error);
      } finally {
        liveDetectionBusyRef.current = false;
        liveInferenceBusy.setBlocking(false);
      }
    },
    [liveInferenceBusy]
  );

  // Frame dönüşüm hatasında canlı modeli durdurup tekrar eden uyarıları engeller.
  const reportLiveFrameError = useCallback((message: string) => {
    liveDetectionPausedRef.current = true;
    console.warn('Canlı kamera frame’i hazırlanamadı.', message);
  }, []);

  const frameOutput = useFrameOutput({
    targetResolution: LIVE_FRAME_RESOLUTION,
    // Model RGB beklediği için renk dönüşümünü kamera hattında tutarlı biçimde yaptırır.
    pixelFormat: 'rgb',
    dropFramesWhileBusy: true,
    enablePreviewSizedOutputBuffers: true,
    enablePhysicalBufferRotation: true,
    onFrame(frame) {
      'worklet';

      // Canlı takip için yalnızca her altıncı kamera frame'ini inference adayı yapar.
      const globalFrameState = globalThis as typeof globalThis & {
        __fingerFrameCounter?: number;
      };
      const nextCount = (globalFrameState.__fingerFrameCounter ?? 0) + 1;
      globalFrameState.__fingerFrameCounter = nextCount;
      const isInferenceFrame = nextCount % LIVE_MODEL_FRAME_INTERVAL === 0;

      // Model çalışırken yeni frame'i JPEG'e dönüştürmeden doğrudan kamera tamponunu serbest bırakır.
      if (
        !isInferenceFrame ||
        liveFrameFailed.getBlocking() ||
        photoProcessingActive.getBlocking()
      ) {
        frame.dispose();
        return;
      }

      if (liveInferenceBusy.getBlocking()) {
        frame.dispose();
        return;
      }

      liveInferenceBusy.setBlocking(true);

      try {
        const sourceWidth = frame.width;
        const sourceHeight = frame.height;
        const image = HybridFrameConverter.convertFrameToImage(frame);

        try {
          // Worklet dışındaki JavaScript fonksiyonlarına senkron çağrı yapmamak için oran hesabı burada tutulur.
          const inferenceScale = Math.min(
            1,
            FINGERTIP_MODEL_SIZE / Math.max(sourceWidth, sourceHeight)
          );
          const inferenceSize = {
            width: Math.max(1, Math.round(sourceWidth * inferenceScale)),
            height: Math.max(1, Math.round(sourceHeight * inferenceScale)),
          };
          const resizedImage = image.resize(
            inferenceSize.width,
            inferenceSize.height
          );

          try {
            // Native ham tamponu görüntü serbest bırakılmadan önce bağımsız JavaScript belleğine kopyalar.
            const rawPixels = resizedImage.toRawPixelData();
            const sourcePixels = new Uint8Array(rawPixels.buffer);
            const copiedPixels = new Uint8Array(sourcePixels.length);
            copiedPixels.set(sourcePixels);

            scheduleOnRN(
              processLiveModelFrame,
              copiedPixels.buffer,
              inferenceSize.width,
              inferenceSize.height,
              sourceWidth,
              sourceHeight
            );
          } finally {
            resizedImage.dispose();
          }
        } finally {
          image.dispose();
        }
      } catch (error) {
        liveFrameFailed.setBlocking(true);
        liveInferenceBusy.setBlocking(false);
        scheduleOnRN(reportLiveFrameError, String(error));
      } finally {
        // GPU tampon havuzunun dolmaması için seçilen inference frame'ini her durumda serbest bırakır.
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

  // Kamera sekmesi açık ve model hazır olduğunda canlı kutu tespitini otomatik çalıştırır.
  useEffect(() => {
    liveDetectionPausedRef.current = modelStatus !== 'ready' || !isScreenFocused;
    if (modelStatus === 'ready' && isScreenFocused) {
      liveFrameFailed.setBlocking(false);
      liveInferenceBusy.setBlocking(false);
    }
  }, [isScreenFocused, liveFrameFailed, liveInferenceBusy, modelStatus]);

  // Canlı kutular kararlı biçimde görüldükçe kutu grubunun merkezinde yakın odağı sürekli tutar.
  useEffect(() => {
    const camera = cameraRef.current;
    if (
      !camera ||
      !canUseCamera ||
      !isCameraReady ||
      isTakingPhoto ||
      liveFocusBusyRef.current ||
      liveDetections.length === 0 ||
      cameraLayout.width <= 0 ||
      cameraLayout.height <= 0
    ) {
      return;
    }

    const region = getLiveMeteringRegion({
      detections: liveDetections,
      imageSize: liveDetectionFrame,
      layout: cameraLayout,
    });
    const now = Date.now();
    const elapsed = now - lastLiveFocusAtRef.current;
    const previousPoint = lastLiveFocusPointRef.current;
    const pointMoved =
      !previousPoint ||
      getPointDistance(previousPoint, region) >= LIVE_FOCUS_MIN_POINT_DISTANCE;

    if (
      elapsed < LIVE_FOCUS_MIN_INTERVAL_MS ||
      (!pointMoved && elapsed < LIVE_FOCUS_FORCE_REFRESH_MS)
    ) {
      return;
    }

    liveFocusBusyRef.current = true;
    lastLiveFocusAtRef.current = now;
    lastLiveFocusPointRef.current = region;

    const activeDevice = camera.controller?.device;
    if (!activeDevice) {
      liveFocusBusyRef.current = false;
      return;
    }

    meterCameraAtRegion({
      camera,
      device: activeDevice,
      region,
      adaptiveness: 'continuous',
      })
      .catch((error) => {
        if (
          !captureMeteringActiveRef.current &&
          !isMeteringCancellation(error)
        ) {
          console.warn('Canlı kutu odağı uygulanamadı.', error);
        }
      })
      .finally(() => {
        liveFocusBusyRef.current = false;
      });
  }, [
    cameraLayout,
    canUseCamera,
    isCameraReady,
    isTakingPhoto,
    liveDetectionFrame,
    liveDetections,
  ]);

  // Kamera sekmesi görünürken kamerayı çalıştırır, sekmeden çıkınca kaynağı serbest bırakır.
  useFocusEffect(
    useCallback(() => {
      setIsScreenFocused(true);
      return () => {
        setIsScreenFocused(false);
        setIsCameraReady(false);
        liveFocusBusyRef.current = false;
        lastLiveFocusAtRef.current = 0;
        lastLiveFocusPointRef.current = null;
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
    captureMeteringActiveRef.current = true;
    liveDetectionPausedRef.current = true;
    photoProcessingActive.setBlocking(true);
    // Odak ve gerçek deklanşör tamamlanana kadar kullanıcının elini kadrajda sabit tutmasını ister.
    showFeedback('Elini sabit tut, fotoğraf çekiliyor...', false);
    const processingStartedAt = getCurrentTimeMs();
    let ambientSourceImage: NitroImage | undefined;
    let flashSourceImage: NitroImage | undefined;

    try {
      const focusStartedAt = getCurrentTimeMs();
      await focusCameraBeforeCapture();
      const focusDuration = getCurrentTimeMs() - focusStartedAt;

      const shouldCaptureExposurePair =
        isDualCaptureEnabled &&
        (cameraRef.current?.controller?.device.hasFlash ?? false);

      // İlk kareyi ortam ışığında alırız; ana kayıt ve kutu koordinatları bu kareyi temel alır.
      const ambientCaptureStartedAt = getCurrentTimeMs();
      const ambientPhoto = await photoOutput.capturePhoto(
        {
          flashMode: 'off',
          enableRedEyeReduction: false,
          enableShutterSound: false,
        },
        {}
      );
      const ambientCaptureDuration = getCurrentTimeMs() - ambientCaptureStartedAt;
      const ambientPrepared = await prepareCapturedPhoto(ambientPhoto, 'flaşsız');
      ambientSourceImage = ambientPrepared.image;

      let flashPrepared: PreparedCapturePhoto | undefined;
      let flashCaptureDuration = 0;
      if (shouldCaptureExposurePair) {
        // İkinci deklanşör tamamlanana kadar elin çekilmemesi gerekir; flaş yalnızca bu fotoğrafta patlar.
        showFeedback('Elini sabit tut, ikinci fotoğraf çekiliyor...', false);
        const flashCaptureStartedAt = getCurrentTimeMs();
        const flashPhoto = await photoOutput.capturePhoto(
          {
            flashMode: 'on',
            enableRedEyeReduction: false,
            enableShutterSound: false,
          },
          {}
        );
        flashCaptureDuration = getCurrentTimeMs() - flashCaptureStartedAt;
        flashPrepared = await prepareCapturedPhoto(flashPhoto, 'flaşlı');
        flashSourceImage = flashPrepared.image;
      }

      // İki gerçek deklanşör de tamamlandı; bundan sonraki işlemler yalnızca sabit JPEG'lerde yapılır.
      showFeedback('Fotoğraflar alındı, işleniyor...', false);
      const sampleId = createId('sample');
      const rawSaveStartedAt = getCurrentTimeMs();
      const rawImageUri = await saveRawImage(
        `file://${ambientPrepared.temporaryPath}`,
        sampleId
      );
      const flashImageUri = flashPrepared
        ? await saveRawImage(
            `file://${flashPrepared.temporaryPath}`,
            `${sampleId}-flash`
          )
        : undefined;
      const rawSaveDuration = getCurrentTimeMs() - rawSaveStartedAt;

      const detectionStartedAt = getCurrentTimeMs();
      const ambientDetections = await tryDetectFingertipsFromImage(
        ambientSourceImage,
        rawImageUri
      );
      const flashDetections =
        flashSourceImage && flashImageUri
          ? await tryDetectFingertipsFromImage(flashSourceImage, flashImageUri)
          : [];
      const {
        detections,
        sourceByClassName,
        selectedFromFlash,
      } =
        selectBestDetectionsAcrossExposure(ambientDetections, flashDetections);
      const detectionDuration = getCurrentTimeMs() - detectionStartedAt;
      console.info(
        `[Çift çekim] flaşsız=${formatDetectionSummary(ambientDetections)}, flaşlı=${formatDetectionSummary(flashDetections)}, seçilen=${formatDetectionSummary(detections)}, flaş_kaynağı=${selectedFromFlash.join(',') || 'yok'}`
      );

      // Her tespiti kendi çekildiği native görüntüyle eşleştirerek hareketten doğan yanlış crop riskini azaltırız.
      const sourceImagesByClassName = new Map<
        DetectedObbBox['className'],
        NitroImage
      >();
      for (const detection of detections) {
        if (
          sourceByClassName.get(detection.className) === 'flash' &&
          flashSourceImage
        ) {
          sourceImagesByClassName.set(detection.className, flashSourceImage);
        } else {
          sourceImagesByClassName.set(detection.className, ambientSourceImage);
        }
      }

      // Her seçilen parmak için karşı pozlamadaki eş kutuyu ayrıca taşırız.
      const comparisonImagesByClassName = new Map<
        DetectedObbBox['className'],
        NitroImage
      >();
      const comparisonDetectionsByClassName = new Map<
        DetectedObbBox['className'],
        DetectedObbBox
      >();
      const ambientByClassName = new Map(
        ambientDetections.map((detection) => [detection.className, detection])
      );
      const flashByClassName = new Map(
        flashDetections.map((detection) => [detection.className, detection])
      );
      for (const detection of detections) {
        const selectedSource = sourceByClassName.get(detection.className);
        const comparisonDetection =
          selectedSource === 'flash'
            ? ambientByClassName.get(detection.className)
            : flashByClassName.get(detection.className);
        const comparisonImage =
          selectedSource === 'flash' ? ambientSourceImage : flashSourceImage;
        if (comparisonDetection && comparisonImage) {
          comparisonDetectionsByClassName.set(
            detection.className,
            comparisonDetection
          );
          comparisonImagesByClassName.set(detection.className, comparisonImage);
        }
      }

      const roiStartedAt = getCurrentTimeMs();
      const fingerRois = await tryExtractFingerRois({
        imageUri: rawImageUri,
        imageSize: {
          width: ambientSourceImage.width,
          height: ambientSourceImage.height,
        },
        detections,
        sampleId,
        sourceImage: ambientSourceImage,
        sourceImagesByClassName,
        comparisonImagesByClassName,
        comparisonDetectionsByClassName,
      });
      const roiDuration = getCurrentTimeMs() - roiStartedAt;

      // Ham çift korunur; bu aşamada pikseller birleştirilmeden yalnızca eksik sınıf tamamlanır.
      const sample: CaptureSample = {
        id: sampleId,
        createdAt: new Date().toISOString(),
        rawImageUri,
        exposurePair:
          flashImageUri
            ? { ambientImageUri: rawImageUri, flashImageUri }
            : undefined,
        rawImageSize: {
          width: ambientSourceImage.width,
          height: ambientSourceImage.height,
        },
        detections,
        fingerRois,
        deviceModel: Device.modelName ?? 'Bilinmeyen cihaz',
        fingerLabel: 'unknown',
        sessionId: createId('session'),
        qualityStatus: getCaptureQualityStatus(fingerRois),
        accepted: false,
      };

      const indexSaveStartedAt = getCurrentTimeMs();
      await appendCaptureSample(sample);
      const indexSaveDuration = getCurrentTimeMs() - indexSaveStartedAt;
      console.info(
        `[Fotoğraf süre] odak=${focusDuration}ms, flaşsız_çekim=${ambientCaptureDuration}ms, flaşlı_çekim=${flashCaptureDuration}ms, hazırlık=${ambientPrepared.preparationDurationMs + (flashPrepared?.preparationDurationMs ?? 0)}ms, ham_kayıt=${rawSaveDuration}ms, model=${detectionDuration}ms, roi=${roiDuration}ms, indeks=${indexSaveDuration}ms, toplam=${getCurrentTimeMs() - processingStartedAt}ms`
      );
      showFeedback(
        fingerRois.length > 0
          ? createCaptureQualityFeedback(fingerRois)
          : detections.length > 0
            ? 'Fotoğraf kutularıyla birlikte kaydedildi.'
            : 'Fotoğraf kaydedildi, model parmak ucu bulamadı.'
      );
    } catch (error) {
      console.warn('Fotoğraf kaydedilemedi.', error);
      showFeedback('Fotoğraf kaydedilemedi. Tekrar dene.', false);
    } finally {
      ambientSourceImage?.dispose();
      flashSourceImage?.dispose();
      captureMeteringActiveRef.current = false;
      photoProcessingActive.setBlocking(false);
      liveDetectionPausedRef.current = modelStatus !== 'ready' || !isScreenFocused;
      setIsTakingPhoto(false);
    }
  }

  // Deklanşör öncesinde canlı kutuların merkezine odak/pozlama ölçümü yaparak ROI netliğini artırır.
  async function focusCameraBeforeCapture() {
    const camera = cameraRef.current;
    if (!camera || cameraLayout.width <= 0 || cameraLayout.height <= 0) return;

    try {
      // Devam eden canlı 3A isteğini bitirip deklanşör ölçümünü tek etkin istek olarak başlatırız.
      await camera.resetFocus().catch((error) => {
        if (!isMeteringCancellation(error)) {
          console.warn('Canlı kamera ölçümü sıfırlanamadı.', error);
        }
      });
      const region = getLiveMeteringRegion({
        detections: liveDetections,
        imageSize: liveDetectionFrame,
        layout: cameraLayout,
      });
      const activeDevice = camera.controller?.device;
      if (!activeDevice) return;

      await meterCameraAtRegion({
        camera,
        device: activeDevice,
        region,
        adaptiveness: 'locked',
      });
      const controller = camera.controller;
      if (controller) {
        console.info(
          `[Çekim 3A] modlar=${getSupportedMeteringModes(activeDevice).join('+') || 'yok'}, bölge=${Math.round(region.size)}px, poz_kademesi=${controller.exposureBias.toFixed(0)}`
        );
      }
      await sleep(PRE_CAPTURE_FOCUS_DELAY_MS);
    } catch (error) {
      console.warn('Çekim öncesi kamera odağı uygulanamadı.', error);
    }
  }

  // Kamera izni reddedildiyse kullanıcıya yeniden isteme olanağı verir.
  async function handlePermissionRetry() {
    const isGranted = await requestPermission();
    if (!isGranted) {
      showFeedback('Kamera izni verilmedi.', false);
    }
  }

  // Kullanıcının aynı sahneyi flaşlı ve flaşsız çekerek kalite sonuçlarını karşılaştırmasını sağlar.
  function handleToggleDualCapture() {
    if (!activeCameraHasFlash) {
      showFeedback('Bu kamera flaşı desteklemiyor.');
      return;
    }

    setIsDualCaptureEnabled((currentValue) => !currentValue);
  }

  return (
    <View
      style={styles.container}
      onLayout={(event) =>
        setCameraLayout({
          width: event.nativeEvent.layout.width,
          height: event.nativeEvent.layout.height,
        })
      }>
      {hasPermission ? (
        <VisionCamera
          ref={cameraRef}
          style={StyleSheet.absoluteFill}
          device={closeUpCameraDevice ?? 'back'}
          outputs={[photoOutput, frameOutput]}
          constraints={cameraConstraints}
          isActive={canUseCamera}
          orientationSource="interface"
          resizeMode="cover"
          enableNativeTapToFocusGesture
          onSessionConfigSelected={(config) => {
            console.info(
              `[Kamera formatı] binned=${config.isBinned}, piksel=${config.nativePixelFormat}, odak=${config.autoFocusSystem}, foto_hdr=${config.isPhotoHDREnabled}`
            );
          }}
          onStarted={() => {
            setIsCameraReady(true);
            setFeedback('');
            const controller = cameraRef.current?.controller;
            const activeDevice = controller?.device;
            if (!controller || !activeDevice) return;

            // Kamera aktif olduktan sonra yakın lens zoomunu ve ROI için kontrollü EV telafisini sırayla uygularız.
            void (async () => {
              try {
                if (activeDevice.isVirtualDevice) {
                  await controller.setZoom(closeUpZoom);
                }

                const exposureCompensationIndex = activeDevice.supportsExposureBias
                  ? clamp(
                      ROI_EXPOSURE_COMPENSATION_INDEX,
                      activeDevice.minExposureBias,
                      activeDevice.maxExposureBias
                    )
                  : 0;
                if (activeDevice.supportsExposureBias) {
                  await controller.setExposureBias(exposureCompensationIndex);
                }

                console.info(
                  `[Yakın odak] lens=${activeDevice.type}, sanal=${activeDevice.isVirtualDevice}, zoom=${closeUpZoom.toFixed(2)}`
                );
                console.info(
                  `[Kamera 3A] AF=${activeDevice.supportsFocusMetering}, AE=${activeDevice.supportsExposureMetering}, AWB=${activeDevice.supportsWhiteBalanceMetering}, flaş=${activeDevice.hasFlash}, poz_kademesi=${exposureCompensationIndex.toFixed(0)} [${activeDevice.minExposureBias.toFixed(0)},${activeDevice.maxExposureBias.toFixed(0)}], düşük_ışık=${activeDevice.supportsLowLightBoost}`
                );
              } catch (error) {
                console.warn('Yakın çekim kamera ayarları uygulanamadı.', error);
              }
            })();
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

      <LiveDetectionOverlay detections={liveDetections} imageSize={liveDetectionFrame} />

      <SafeAreaView pointerEvents="box-none" style={styles.controls}>
        <View style={styles.topControl}>
          <ThemedText type="smallBold" style={styles.modeText}>
            Canlı takip
          </ThemedText>
          <Pressable
            accessibilityLabel="Çift pozlama çekimi"
            accessibilityRole="switch"
            accessibilityState={{
              checked: isDualCaptureEnabled,
              disabled: !activeCameraHasFlash || isTakingPhoto,
            }}
            disabled={!activeCameraHasFlash || isTakingPhoto}
            onPress={handleToggleDualCapture}
            style={[
              styles.flashToggle,
              isDualCaptureEnabled && styles.flashToggleActive,
              (!activeCameraHasFlash || isTakingPhoto) && styles.disabledButton,
            ]}>
            <ThemedText type="smallBold" style={styles.flashToggleText}>
              {activeCameraHasFlash
                ? isDualCaptureEnabled
                  ? 'Çift çekim açık'
                  : 'Tek çekim'
                : 'Çift çekim yok'}
            </ThemedText>
          </Pressable>
        </View>

        <View style={styles.bottomControl}>
          {feedback || liveCaptureGuide ? (
            <ThemedText type="small" style={styles.feedbackText}>
              {feedback || liveCaptureGuide}
            </ThemedText>
          ) : null}

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
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: Spacing.three,
    paddingHorizontal: Spacing.four,
    backgroundColor: 'rgba(0, 0, 0, 0.28)',
  },
  modeText: {
    color: '#ffffff',
    textShadowColor: 'rgba(0, 0, 0, 0.75)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 3,
  },
  flashToggle: {
    minHeight: 36,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.55)',
    borderRadius: 6,
    paddingHorizontal: Spacing.three,
    backgroundColor: 'rgba(0, 0, 0, 0.36)',
  },
  flashToggleActive: {
    borderColor: '#F5B844',
    backgroundColor: 'rgba(245, 184, 68, 0.22)',
  },
  flashToggleText: {
    color: '#ffffff',
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
