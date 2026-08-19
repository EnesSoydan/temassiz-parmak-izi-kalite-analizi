import * as Device from 'expo-device';
import { Image } from 'expo-image';
import * as ImagePicker from 'expo-image-picker';
import {
  manipulateAsync,
  FlipType,
  SaveFormat,
  type Action,
} from 'expo-image-manipulator';
import { useFocusEffect } from 'expo-router';
import { useCallback, useRef, useState } from 'react';
import {
  Gesture,
  GestureDetector,
  GestureHandlerRootView,
} from 'react-native-gesture-handler';
import Animated, {
  runOnJS,
  useAnimatedStyle,
  useSharedValue,
  withTiming,
} from 'react-native-reanimated';
import {
  FlatList,
  Alert,
  Modal,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  useWindowDimensions,
  View,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { DetectionImage } from '@/components/detection-image';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { WebBadge } from '@/components/web-badge';
import { BottomTabInset, MaxContentWidth, Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import {
  clearBiometricDatabase,
  deletePerson,
  loadBiometricDatabase,
} from '@/lib/biometric-database';
import {
  appendCaptureSample,
  deleteCaptureSample,
  exportQualityCalibrationSummary,
  loadCaptureSamples,
  saveRawImage,
  setCaptureCalibrationLabel,
  setCaptureProbeEvaluation,
} from '@/lib/capture-storage';
import {
  createFingerprintEnrollmentBaselineReport,
  createFingerprintProbeBaselineReport,
} from '@/lib/fingerprint-baseline-report';
import { detectFingertipObbBoxes } from '@/lib/fingertip-detection';
import { sortFingerRois } from '@/lib/finger-order';
import { extractFingerRoisFromImage } from '@/lib/fingertip-roi';
import {
  formatFingerprintQualityStatus,
  getCaptureQualityStatus,
} from '@/lib/fingerprint-quality';
import { FINGERPRINT_POSITIONS } from '@/types/biometrics';
import type {
  CaptureSample,
  DetectedObbBox,
  FingerRoi,
  Enrollment,
  Person,
  QualityCalibrationLabel,
} from '@/types/biometrics';

// Galeriden eklenen kayıtlar için kısa ve çakışma ihtimali düşük kimlik üretir.
function createId(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function formatRateForUi(rate: number | null) {
  return rate === null ? 'hesaplanamadı' : `${Math.round(rate * 100)}%`;
}

function getExifOrientation(asset: ImagePicker.ImagePickerAsset) {
  const exif = asset.exif as
    | Record<string, number | string | undefined>
    | undefined;
  return Number(exif?.Orientation ?? exif?.orientation ?? 1);
}

function getExifOrientationActions(orientation: number): Action[] {
  switch (orientation) {
    case 2:
      return [{ flip: FlipType.Horizontal }];
    case 3:
      return [{ rotate: 180 }];
    case 4:
      return [{ flip: FlipType.Vertical }];
    case 5:
      return [{ flip: FlipType.Horizontal }, { rotate: 270 }];
    case 6:
      return [{ rotate: 90 }];
    case 7:
      return [{ flip: FlipType.Horizontal }, { rotate: 90 }];
    case 8:
      return [{ rotate: 270 }];
    default:
      return [];
  }
}

async function normalizePickedImage(asset: ImagePicker.ImagePickerAsset) {
  const actions = getExifOrientationActions(getExifOrientation(asset));
  return manipulateAsync(asset.uri, actions, {
    compress: 1,
    format: SaveFormat.JPEG,
  });
}

// Parmak sınıf adlarını ROI önizlemelerinde kısa Türkçe etiketlere çevirir.
function formatFingerRoiClass(className: FingerRoi['className']) {
  if (className === 'index') return 'işaret';
  if (className === 'middle') return 'orta';
  if (className === 'pinky') return 'serçe';
  if (className === 'ring') return 'yüzük';
  return 'parmak';
}

type RoiInspectionVariant =
  | 'roi'
  | 'canonical'
  | 'alignedCanonical'
  | 'segmented'
  | 'binary'
  | 'openedBinary'
  | 'orientation'
  | 'enhanced'
  | 'minutiae';

// Tam ekran inceleyicide gösterilebilen ROI sürümlerini kullanıcı etiketleriyle tanımlar.
const ROI_INSPECTION_OPTIONS: {
  value: RoiInspectionVariant;
  label: string;
}[] = [
  { value: 'roi', label: 'ROI' },
  { value: 'canonical', label: 'Flaşlı' },
  { value: 'alignedCanonical', label: 'Hizalı' },
  { value: 'segmented', label: 'Seg' },
  { value: 'binary', label: 'Binary' },
  { value: 'openedBinary', label: 'Opening' },
  { value: 'orientation', label: 'Yön' },
  { value: 'enhanced', label: 'Ridge' },
  { value: 'minutiae', label: 'Minutiae' },
];

const ROI_THUMBNAIL_WIDTH = 52;
const ROI_VARIANT_COUNT = ROI_INSPECTION_OPTIONS.length;
const ROI_PREVIEW_WIDTH =
  ROI_VARIANT_COUNT * ROI_THUMBNAIL_WIDTH +
  (ROI_VARIANT_COUNT - 1) * Spacing.one;

// Seçilen inceleme sürümünün parmak ROI metadata'sındaki dosya yolunu döndürür.
function getRoiInspectionUri(
  fingerRoi: FingerRoi,
  variant: RoiInspectionVariant
) {
  if (variant === 'roi') return fingerRoi.imageUri;
  if (variant === 'canonical') return fingerRoi.canonicalImageUri;
  if (variant === 'alignedCanonical') return fingerRoi.alignedCanonicalImageUri;
  if (variant === 'segmented') return fingerRoi.segmentedImageUri;
  if (variant === 'binary') return fingerRoi.binaryImageUri;
  if (variant === 'openedBinary') return fingerRoi.openedBinaryImageUri;
  if (variant === 'orientation') return fingerRoi.orientationImageUri;
  if (variant === 'enhanced') return fingerRoi.enhancedImageUri;
  return fingerRoi.minutiaeImageUri;
}

const MIN_INSPECTION_SCALE = 1;
const MAX_INSPECTION_SCALE = 4;

// Tam ekran ROI sayfasına pinch zoom, zoom sırasında taşıma ve isteğe bağlı iskelet katmanı ekler.
function ZoomableInspectionImage({
  imageUri,
  overlayImageUri,
  onZoomStateChange,
}: {
  imageUri: string;
  overlayImageUri?: string;
  onZoomStateChange: (zoomed: boolean) => void;
}) {
  const [isZoomed, setIsZoomed] = useState(false);
  const scale = useSharedValue(MIN_INSPECTION_SCALE);
  const savedScale = useSharedValue(MIN_INSPECTION_SCALE);
  const translationX = useSharedValue(0);
  const translationY = useSharedValue(0);
  const savedTranslationX = useSharedValue(0);
  const savedTranslationY = useSharedValue(0);
  const containerWidth = useSharedValue(1);
  const containerHeight = useSharedValue(1);
  const overlayOpacity = useSharedValue(0);

  const reportZoomState = (zoomed: boolean) => {
    setIsZoomed(zoomed);
    onZoomStateChange(zoomed);
  };

  const pinchGesture = Gesture.Pinch()
    .onBegin(() => {
      runOnJS(reportZoomState)(true);
    })
    .onUpdate((event) => {
      scale.value = Math.min(
        MAX_INSPECTION_SCALE,
        Math.max(MIN_INSPECTION_SCALE, savedScale.value * event.scale)
      );
    })
    .onFinalize(() => {
      const nextScale = Math.min(
        MAX_INSPECTION_SCALE,
        Math.max(MIN_INSPECTION_SCALE, scale.value)
      );
      if (nextScale <= MIN_INSPECTION_SCALE + 0.01) {
        scale.value = withTiming(MIN_INSPECTION_SCALE);
        translationX.value = withTiming(0);
        translationY.value = withTiming(0);
        savedScale.value = MIN_INSPECTION_SCALE;
        savedTranslationX.value = 0;
        savedTranslationY.value = 0;
        runOnJS(reportZoomState)(false);
        return;
      }

      const maximumX = (containerWidth.value * (nextScale - 1)) / 2;
      const maximumY = (containerHeight.value * (nextScale - 1)) / 2;
      scale.value = nextScale;
      savedScale.value = nextScale;
      translationX.value = Math.min(
        maximumX,
        Math.max(-maximumX, translationX.value)
      );
      translationY.value = Math.min(
        maximumY,
        Math.max(-maximumY, translationY.value)
      );
      savedTranslationX.value = translationX.value;
      savedTranslationY.value = translationY.value;
      runOnJS(reportZoomState)(true);
    });

  const panGesture = Gesture.Pan()
    .enabled(isZoomed)
    .minDistance(2)
    .onUpdate((event) => {
      const maximumX = (containerWidth.value * (scale.value - 1)) / 2;
      const maximumY = (containerHeight.value * (scale.value - 1)) / 2;
      translationX.value = Math.min(
        maximumX,
        Math.max(-maximumX, savedTranslationX.value + event.translationX)
      );
      translationY.value = Math.min(
        maximumY,
        Math.max(-maximumY, savedTranslationY.value + event.translationY)
      );
    })
    .onFinalize(() => {
      savedTranslationX.value = translationX.value;
      savedTranslationY.value = translationY.value;
    });

  const tapGesture = Gesture.Tap()
    .enabled(Boolean(overlayImageUri))
    .maxDistance(8)
    .onEnd((_event, successful) => {
      if (!successful) return;
      overlayOpacity.value = withTiming(
        overlayOpacity.value > 0.5 ? 0 : 1,
        { duration: 140 }
      );
    });

  const imageGesture = Gesture.Simultaneous(
    pinchGesture,
    panGesture,
    tapGesture
  );
  const imageTransformStyle = useAnimatedStyle(() => ({
    transform: [
      { translateX: translationX.value },
      { translateY: translationY.value },
      { scale: scale.value },
    ],
  }));
  const overlayStyle = useAnimatedStyle(() => ({
    opacity: overlayOpacity.value,
  }));

  return (
    <GestureDetector gesture={imageGesture}>
      <Animated.View
        style={[styles.inspectionImageFrame, imageTransformStyle]}
        onLayout={(event) => {
          containerWidth.value = Math.max(event.nativeEvent.layout.width, 1);
          containerHeight.value = Math.max(event.nativeEvent.layout.height, 1);
        }}>
        <Image
          source={{ uri: imageUri }}
          style={styles.inspectionImage}
          contentFit="contain"
        />
        {overlayImageUri ? (
          <Animated.View
            pointerEvents="none"
            style={[styles.inspectionOverlay, overlayStyle]}>
            <Image
              source={{ uri: overlayImageUri }}
              style={styles.inspectionImage}
              contentFit="contain"
            />
          </Animated.View>
        ) : null}
      </Animated.View>
    </GestureDetector>
  );
}

// Tek bir kaydın flaşlı ham fotoğrafını, ROI çıktısını ve kalite durumlarını gösterir.
function CaptureCard({
  sample,
  people,
  isEnrollmentSource,
  onDelete,
  onCalibrationLabel,
  onProbeEvaluation,
}: {
  sample: CaptureSample;
  people: Person[];
  isEnrollmentSource: boolean;
  onDelete: (sampleId: string) => void;
  onCalibrationLabel: (sampleId: string, label: QualityCalibrationLabel) => void;
  onProbeEvaluation: (
    sampleId: string,
    target: { relation: 'genuine'; personId: string } | { relation: 'impostor' } | null
  ) => void;
}) {
  const inspectionScrollRef = useRef<ScrollView>(null);
  const { width: windowWidth } = useWindowDimensions();
  const [inspection, setInspection] = useState<{
    fingerRoi: FingerRoi;
    variant: RoiInspectionVariant;
  } | null>(null);
  const [inspectionImageZoomed, setInspectionImageZoomed] = useState(false);
  const [inspectionZoomRevision, setInspectionZoomRevision] = useState(0);
  const inspectionPageWidth = Math.max(
    windowWidth - Spacing.three * 2,
    1
  );
  const availableInspectionOptions = inspection
    ? ROI_INSPECTION_OPTIONS.filter((option) =>
        getRoiInspectionUri(inspection.fingerRoi, option.value)
      )
    : [];

  // Parmak önizlemesini seçilen sürümle tam ekran karşılaştırma görünümünde açar.
  function openInspection(
    fingerRoi: FingerRoi,
    variant: RoiInspectionVariant
  ) {
    setInspectionImageZoomed(false);
    setInspectionZoomRevision((revision) => revision + 1);
    setInspection({ fingerRoi, variant });
  }

  function closeInspection() {
    setInspectionImageZoomed(false);
    setInspection(null);
  }

  // Alt sekmeden seçilen ROI sürümünü işaretleyip yatay galeriyi aynı sayfaya taşır.
  function selectInspectionVariant(
    variant: RoiInspectionVariant,
    animated = true
  ) {
    if (!inspection) return;
    const index = availableInspectionOptions.findIndex(
      (option) => option.value === variant
    );
    if (index < 0) return;

    setInspectionImageZoomed(false);
    setInspectionZoomRevision((revision) => revision + 1);
    setInspection({ fingerRoi: inspection.fingerRoi, variant });
    inspectionScrollRef.current?.scrollTo({
      x: index * inspectionPageWidth,
      animated,
    });
  }

  // Parmakla sayfa kaydırıldığında alt sekmenin seçili durumunu yeni sayfayla eşleştirir.
  function handleInspectionSwipe(offsetX: number) {
    if (!inspection) return;
    const index = Math.min(
      Math.max(Math.round(offsetX / inspectionPageWidth), 0),
      availableInspectionOptions.length - 1
    );
    const option = availableInspectionOptions[index];
    if (!option || option.value === inspection.variant) return;
    setInspectionImageZoomed(false);
    setInspectionZoomRevision((revision) => revision + 1);
    setInspection({
      fingerRoi: inspection.fingerRoi,
      variant: option.value,
    });
  }

  // Modal açıldığında galeriyi dokunulan ilk önizlemenin sayfasına konumlandırır.
  function alignInspectionOnOpen() {
    if (!inspection) return;
    const index = availableInspectionOptions.findIndex(
      (option) => option.value === inspection.variant
    );
    inspectionScrollRef.current?.scrollTo({
      x: Math.max(index, 0) * inspectionPageWidth,
      animated: false,
    });
  }

  return (
    <ThemedView type="backgroundElement" style={styles.card}>
      <DetectionImage
        imageUri={sample.rawImageUri}
        imageSize={sample.rawImageSize}
        detections={sample.detections}
      />

      {sample.fingerRois?.length ? (
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.roiStrip}>
          {sortFingerRois(sample.fingerRois)
            .filter((fingerRoi) => fingerRoi.imageUri)
            .map((fingerRoi) => (
              <View key={fingerRoi.id} style={styles.roiPreview}>
                <View style={styles.roiImagePair}>
                  <View style={styles.roiImageColumn}>
                    {fingerRoi.binaryImageUri ? (
                      <Pressable
                        onPress={() => openInspection(fingerRoi, 'binary')}>
                        <Image
                          source={{ uri: fingerRoi.binaryImageUri }}
                          style={[styles.roiImage, styles.binaryRoiImage]}
                          contentFit="cover"
                        />
                      </Pressable>
                    ) : (
                      <View style={[styles.roiImage, styles.emptySegmentImage]}>
                        <ThemedText type="small" style={styles.emptySegmentText}>
                          Yok
                        </ThemedText>
                      </View>
                    )}
                    <ThemedText type="small" style={styles.roiVariantLabel}>
                      Binary
                    </ThemedText>
                  </View>

                  <View style={styles.roiImageColumn}>
                    {fingerRoi.openedBinaryImageUri ? (
                      <Pressable
                        onPress={() => openInspection(fingerRoi, 'openedBinary')}>
                        <Image
                          source={{ uri: fingerRoi.openedBinaryImageUri }}
                          style={[styles.roiImage, styles.binaryRoiImage]}
                          contentFit="cover"
                        />
                      </Pressable>
                    ) : (
                      <View style={[styles.roiImage, styles.emptySegmentImage]}>
                        <ThemedText type="small" style={styles.emptySegmentText}>
                          Yok
                        </ThemedText>
                      </View>
                    )}
                    <ThemedText type="small" style={styles.roiVariantLabel}>
                      Opening
                    </ThemedText>
                  </View>

                  <View style={styles.roiImageColumn}>
                    <Pressable onPress={() => openInspection(fingerRoi, 'roi')}>
                      <Image
                        source={{ uri: fingerRoi.imageUri }}
                        style={styles.roiImage}
                        contentFit="cover"
                      />
                    </Pressable>
                    <ThemedText type="small" style={styles.roiVariantLabel}>
                      ROI
                    </ThemedText>
                  </View>

                  <View style={styles.roiImageColumn}>
                    {getRoiInspectionUri(fingerRoi, 'canonical') ? (
                      <Pressable
                        onPress={() => openInspection(fingerRoi, 'canonical')}>
                        <Image
                          source={{
                            uri: getRoiInspectionUri(fingerRoi, 'canonical'),
                          }}
                          style={styles.roiImage}
                          contentFit="cover"
                        />
                      </Pressable>
                    ) : (
                      <View style={[styles.roiImage, styles.emptySegmentImage]}>
                        <ThemedText type="small" style={styles.emptySegmentText}>
                          Yok
                        </ThemedText>
                      </View>
                    )}
                    <ThemedText type="small" style={styles.roiVariantLabel}>
                      Flaşlı
                    </ThemedText>
                  </View>

                  <View style={styles.roiImageColumn}>
                    {fingerRoi.alignedCanonicalImageUri ? (
                      <Pressable
                        onPress={() =>
                          openInspection(fingerRoi, 'alignedCanonical')
                        }>
                        <Image
                          source={{ uri: fingerRoi.alignedCanonicalImageUri }}
                          style={styles.roiImage}
                          contentFit="cover"
                        />
                      </Pressable>
                    ) : (
                      <View style={[styles.roiImage, styles.emptySegmentImage]}>
                        <ThemedText type="small" style={styles.emptySegmentText}>
                          Yok
                        </ThemedText>
                      </View>
                    )}
                    <ThemedText type="small" style={styles.roiVariantLabel}>
                      Hizalı
                    </ThemedText>
                  </View>

                  <View style={styles.roiImageColumn}>
                    {fingerRoi.segmentedImageUri ? (
                      <Pressable
                        onPress={() => openInspection(fingerRoi, 'segmented')}>
                        <Image
                          source={{ uri: fingerRoi.segmentedImageUri }}
                          style={[styles.roiImage, styles.segmentedRoiImage]}
                          contentFit="cover"
                        />
                      </Pressable>
                    ) : (
                      <View style={[styles.roiImage, styles.emptySegmentImage]}>
                        <ThemedText type="small" style={styles.emptySegmentText}>
                          Yok
                        </ThemedText>
                      </View>
                    )}
                    <ThemedText type="small" style={styles.roiVariantLabel}>
                      Seg
                    </ThemedText>
                  </View>

                  <View style={styles.roiImageColumn}>
                    {fingerRoi.orientationImageUri ? (
                      <Pressable
                        onPress={() => openInspection(fingerRoi, 'orientation')}>
                        <Image
                          source={{ uri: fingerRoi.orientationImageUri }}
                          style={[styles.roiImage, styles.orientationRoiImage]}
                          contentFit="cover"
                        />
                      </Pressable>
                    ) : (
                      <View style={[styles.roiImage, styles.emptySegmentImage]}>
                        <ThemedText type="small" style={styles.emptySegmentText}>
                          Yok
                        </ThemedText>
                      </View>
                    )}
                    <ThemedText type="small" style={styles.roiVariantLabel}>
                      Yön
                    </ThemedText>
                  </View>

                  <View style={styles.roiImageColumn}>
                    {fingerRoi.enhancedImageUri ? (
                      <Pressable
                        onPress={() => openInspection(fingerRoi, 'enhanced')}>
                        <Image
                          source={{ uri: fingerRoi.enhancedImageUri }}
                          style={[styles.roiImage, styles.enhancedRoiImage]}
                          contentFit="cover"
                        />
                      </Pressable>
                    ) : (
                      <View style={[styles.roiImage, styles.emptySegmentImage]}>
                        <ThemedText type="small" style={styles.emptySegmentText}>
                          Yok
                        </ThemedText>
                      </View>
                    )}
                    <ThemedText type="small" style={styles.roiVariantLabel}>
                      Ridge
                    </ThemedText>
                  </View>

                  <View style={styles.roiImageColumn}>
                    {fingerRoi.minutiaeImageUri ? (
                      <Pressable
                        onPress={() => openInspection(fingerRoi, 'minutiae')}>
                        <Image
                          source={{ uri: fingerRoi.minutiaeImageUri }}
                          style={[styles.roiImage, styles.minutiaeRoiImage]}
                          contentFit="cover"
                        />
                      </Pressable>
                    ) : (
                      <View style={[styles.roiImage, styles.emptySegmentImage]}>
                        <ThemedText type="small" style={styles.emptySegmentText}>
                          Yok
                        </ThemedText>
                      </View>
                    )}
                    <ThemedText type="small" style={styles.roiVariantLabel}>
                      Nokta
                    </ThemedText>
                  </View>
                </View>
                <ThemedText type="small" style={styles.roiLabel}>
                  {formatFingerRoiClass(fingerRoi.className)}
                </ThemedText>
                <ThemedText
                  type="small"
                  style={[
                    styles.qualityLabel,
                    (fingerRoi.quality?.captureStatus ?? fingerRoi.quality?.status) === 'good' &&
                      styles.qualityGood,
                    (fingerRoi.quality?.captureStatus ?? fingerRoi.quality?.status) === 'medium' &&
                      styles.qualityMedium,
                    (fingerRoi.quality?.captureStatus ?? fingerRoi.quality?.status) === 'poor' &&
                      styles.qualityPoor,
                  ]}>
                  {formatFingerprintQualityStatus(fingerRoi.quality)}
                </ThemedText>
              </View>
            ))}
        </ScrollView>
      ) : null}

      <View style={styles.calibrationControl}>
        <ThemedText type="small" themeColor="textSecondary">
          Elle değerlendirme
        </ThemedText>
        <View style={styles.calibrationSegments}>
          {CALIBRATION_OPTIONS.map((option) => (
            <Pressable
              key={option.value}
              style={[
                styles.calibrationSegment,
                sample.calibrationLabel === option.value &&
                  styles.calibrationSegmentSelected,
              ]}
              onPress={() => onCalibrationLabel(sample.id, option.value)}>
              <ThemedText type="small">{option.label}</ThemedText>
            </Pressable>
          ))}
        </View>
      </View>

      {isEnrollmentSource ? (
        <ThemedText
          type="small"
          themeColor="textSecondary"
          style={styles.enrollmentSourceLabel}>
          Enrollment kaynağı · giriş probe’u olarak etiketlenmez
        </ThemedText>
      ) : (
        <View style={styles.probeControl}>
          <View style={styles.probeControlHeader}>
            <ThemedText type="small" themeColor="textSecondary">
              Giriş çekiminin gerçeği
            </ThemedText>
            {sample.probeEvaluation ? (
              <ThemedText type="small" themeColor="textSecondary">
                Oturum {sample.probeEvaluation.evaluationSessionId.slice(-6)}
              </ThemedText>
            ) : null}
          </View>
          <View style={styles.probeSegments}>
            <Pressable
              style={[
                styles.probeSegment,
                !sample.probeEvaluation && styles.probeSegmentSelected,
              ]}
              onPress={() => onProbeEvaluation(sample.id, null)}>
              <ThemedText type="small">Etiketsiz</ThemedText>
            </Pressable>
            {people.map((person) => (
              <Pressable
                key={person.id}
                style={[
                  styles.probeSegment,
                  sample.probeEvaluation?.relation === 'genuine' &&
                    sample.probeEvaluation.expectedPersonId === person.id &&
                    styles.probeSegmentSelected,
                ]}
                onPress={() =>
                  onProbeEvaluation(sample.id, {
                    relation: 'genuine',
                    personId: person.id,
                  })
                }>
                <ThemedText type="small">{person.displayName}</ThemedText>
              </Pressable>
            ))}
            <Pressable
              style={[
                styles.probeSegment,
                styles.impostorProbeSegment,
                sample.probeEvaluation?.relation === 'impostor' &&
                  styles.impostorProbeSegmentSelected,
              ]}
              onPress={() =>
                onProbeEvaluation(sample.id, { relation: 'impostor' })
              }>
              <ThemedText type="small">Kayıt dışı el</ThemedText>
            </Pressable>
          </View>
        </View>
      )}

      <View style={styles.cardFooter}>
        <ThemedText type="small" themeColor="textSecondary">
          {new Date(sample.createdAt).toLocaleString('tr-TR')}
        </ThemedText>
        <Pressable hitSlop={12} onPress={() => onDelete(sample.id)}>
          <ThemedText type="smallBold" style={styles.deleteText}>
            Sil
          </ThemedText>
        </Pressable>
      </View>

      <Modal
        visible={inspection !== null}
        animationType="fade"
        transparent
        onShow={alignInspectionOnOpen}
        onRequestClose={closeInspection}>
        <GestureHandlerRootView style={styles.inspectionGestureRoot}>
        <View style={styles.inspectionBackdrop}>
          <View style={styles.inspectionHeader}>
            <ThemedText type="smallBold" style={styles.inspectionTitle}>
              {inspection
                ? `${formatFingerRoiClass(inspection.fingerRoi.className)} parmak`
                : 'Parmak ROI'}
            </ThemedText>
            <Pressable
              style={styles.closeInspectionButton}
              onPress={closeInspection}>
              <ThemedText type="smallBold" style={styles.inspectionTitle}>
                Kapat
              </ThemedText>
            </Pressable>
          </View>

          <ScrollView
            ref={inspectionScrollRef}
            horizontal
            pagingEnabled
            bounces={false}
            scrollEnabled={!inspectionImageZoomed}
            showsHorizontalScrollIndicator={false}
            style={styles.inspectionImageArea}
            onContentSizeChange={alignInspectionOnOpen}
            onMomentumScrollEnd={(event) =>
              handleInspectionSwipe(event.nativeEvent.contentOffset.x)
            }>
            {inspection
              ? availableInspectionOptions.map((option) => {
                  const imageUri = getRoiInspectionUri(
                    inspection.fingerRoi,
                    option.value
                  );
                  return (
                    <View
                      key={option.value}
                      style={[
                        styles.inspectionPage,
                        { width: inspectionPageWidth },
                      ]}>
                      {imageUri ? (
                        <ZoomableInspectionImage
                          key={`${option.value}-${inspectionZoomRevision}`}
                          imageUri={imageUri}
                          overlayImageUri={
                            option.value === 'alignedCanonical'
                              ? inspection.fingerRoi.minutiaeOverlayImageUri
                              : undefined
                          }
                          onZoomStateChange={setInspectionImageZoomed}
                        />
                      ) : null}
                    </View>
                  );
                })
              : null}
          </ScrollView>

          {inspection?.variant === 'orientation' ? (
            <ThemedText type="small" style={styles.inspectionLegend}>
              8x8 hücre: yeşil doğrulandı, kırmızı kaba yönle uyuşmadı,
              sarı zayıf. Mavi çizgi ridge yönü, gri çizgi belirsiz yöndür.
            </ThemedText>
          ) : null}
          {inspection?.variant === 'minutiae' ? (
            <ThemedText type="small" style={styles.inspectionLegend}>
              Beyaz çizgi ridge iskeleti, turuncu işaret ridge sonu, mavi işaret çatallanmadır.
            </ThemedText>
          ) : null}
          <ThemedText type="small" style={styles.inspectionGestureHint}>
            {inspection?.variant === 'alignedCanonical'
              ? inspection.fingerRoi.minutiaeOverlayImageUri
                ? 'Tek dokun: iskeleti aç/kapat · İki parmak: yakınlaştır/uzaklaştır · Yakınken sürükle.'
                : 'İskelet katmanı bu eski kayıtta yok · İki parmakla yakınlaştır/uzaklaştır.'
              : 'İki parmakla yakınlaştır/uzaklaştır · Yakınken tek parmakla sürükle.'}
          </ThemedText>

          <View style={styles.inspectionTabs}>
            {inspection
              ? availableInspectionOptions.map((option) => (
                  <Pressable
                    key={option.value}
                    style={[
                      styles.inspectionTab,
                      inspection.variant === option.value &&
                        styles.inspectionTabSelected,
                    ]}
                    onPress={() => selectInspectionVariant(option.value)}>
                    <ThemedText
                      type="smallBold"
                      style={styles.inspectionTitle}>
                      {option.label}
                    </ThemedText>
                  </Pressable>
                ))
              : null}
          </View>
        </View>
        </GestureHandlerRootView>
      </Modal>
    </ThemedView>
  );
}

// Kalibrasyon etiketlerini kayıt kartlarında kısa ve anlaşılır seçeneklerle sunar.
const CALIBRATION_OPTIONS: {
  value: QualityCalibrationLabel;
  label: string;
}[] = [
  { value: 'good', label: 'İyi' },
  { value: 'borderline', label: 'Sınırda' },
  { value: 'bad', label: 'Kötü' },
];

// Galeri testinde ROI çıkarma hatası olursa ana kayıt ve kutu metadata'sı yine korunur.
async function tryExtractFingerRois({
  imageUri,
  imageSize,
  detections,
  sampleId,
}: {
  imageUri: string;
  imageSize: NonNullable<CaptureSample['rawImageSize']>;
  detections: DetectedObbBox[];
  sampleId: string;
}) {
  try {
    return await extractFingerRoisFromImage({ imageUri, imageSize, detections, sampleId });
  } catch (error) {
    console.warn('Galeri görselinden parmak ROI çıkarılamadı.', error);
    return [];
  }
}

function PersonRegistry({
  people,
  enrollments,
  onDelete,
  onClear,
  selectedPersonId,
  onSelect,
}: {
  people: Person[];
  enrollments: Enrollment[];
  onDelete: (person: Person) => void;
  onClear: () => void;
  selectedPersonId: string | null;
  onSelect: (personId: string) => void;
}) {
  return (
    <ThemedView style={styles.peopleCard}>
      <View style={styles.peopleHeader}>
        <View style={styles.headerText}>
          <ThemedText type="subtitle" style={styles.peopleTitle}>
            Kimlik kayıtları
          </ThemedText>
          <ThemedText type="small" themeColor="textSecondary">
            {people.length === 0
              ? 'Henüz kayıtlı kişi yok.'
              : `${people.length} kişi, şablonlar şifreli yerelde tutuluyor.`}
          </ThemedText>
        </View>
        {people.length > 0 ? (
          <Pressable style={styles.resetPeopleButton} onPress={onClear}>
            <ThemedText type="smallBold" style={styles.deleteText}>
              Tümünü sıfırla
            </ThemedText>
          </Pressable>
        ) : null}
      </View>

      {people.map((person) => {
        const personEnrollments = enrollments.filter(
          (enrollment) => enrollment.personId === person.id
        );
        return (
          <View
            key={person.id}
            style={[
              styles.personRow,
              selectedPersonId === person.id && styles.personRowSelected,
            ]}>
            <Pressable
              style={styles.personDetails}
              onPress={() => onSelect(person.id)}>
              <ThemedText type="smallBold">{person.displayName}</ThemedText>
              <ThemedText type="small" themeColor="textSecondary">
                {FINGERPRINT_POSITIONS.map((position) => {
                  const exists = personEnrollments.some(
                    (enrollment) => enrollment.fingerPosition === position
                  );
                  return `${formatFingerRoiClass(position)}: ${exists ? 'hazır' : 'eksik'}`;
                }).join(' · ')}
              </ThemedText>
              <ThemedText type="small" themeColor="textSecondary">
                Kayıt görüntülerini görmek için dokun
              </ThemedText>
            </Pressable>
            <Pressable hitSlop={12} onPress={() => onDelete(person)}>
              <ThemedText type="smallBold" style={styles.deleteText}>
                Sil
              </ThemedText>
            </Pressable>
          </View>
        );
      })}
    </ThemedView>
  );
}

export default function RecordsScreen() {
  const [samples, setSamples] = useState<CaptureSample[]>([]);
  const [people, setPeople] = useState<Person[]>([]);
  const [enrollments, setEnrollments] = useState<Enrollment[]>([]);
  const [selectedPersonId, setSelectedPersonId] = useState<string | null>(null);
  const [feedback, setFeedback] = useState('Kayıtlar yükleniyor...');
  const [isPickingImage, setIsPickingImage] = useState(false);
  const [isCreatingBaseline, setIsCreatingBaseline] = useState(false);
  const [isCreatingProbeReport, setIsCreatingProbeReport] = useState(false);
  const [activeProbeSessionId, setActiveProbeSessionId] = useState(() =>
    createId('probe-session')
  );
  const safeAreaInsets = useSafeAreaInsets();
  const theme = useTheme();
  const insets = {
    ...safeAreaInsets,
    bottom: safeAreaInsets.bottom + BottomTabInset + Spacing.three,
  };

  // Kayıtlar sekmesi açıldığında cihaz içindeki fotoğraf listesini yeniden okur.
  const loadSamples = useCallback(() => {
    Promise.all([loadCaptureSamples(), loadBiometricDatabase()])
      .then(([storedSamples, database]) => {
        setSamples(storedSamples);
        setPeople(database.people);
        setEnrollments(database.enrollments);
        setSelectedPersonId((currentId) =>
          currentId && database.people.some((person) => person.id === currentId)
            ? currentId
            : null
        );
        setFeedback(
          storedSamples.length === 0 ? 'Henüz kayıt yok.' : `${storedSamples.length} fotoğraf`
        );
      })
      .catch(() => {
        setFeedback('Kayıtlar okunamadı.');
      });
  }, []);

  useFocusEffect(loadSamples);

  function handleDeletePerson(person: Person) {
    Alert.alert(
      `${person.displayName} silinsin mi?`,
      'Kişiye ait dört biyometrik şablon da silinecek.',
      [
        { text: 'Vazgeç', style: 'cancel' },
        {
          text: 'Sil',
          style: 'destructive',
          onPress: async () => {
            try {
              const database = await deletePerson(person.id);
              setPeople(database.people);
              setEnrollments(database.enrollments);
              setFeedback(`${person.displayName} silindi.`);
            } catch {
              setFeedback('Kişi silinemedi.');
            }
          },
        },
      ]
    );
  }

  function handleClearPeople() {
    Alert.alert(
      'Tüm biyometrik kayıtlar silinsin mi?',
      'Bu işlem kayıtlı kişileri ve dört parmak şablonlarını kaldırır.',
      [
        { text: 'Vazgeç', style: 'cancel' },
        {
          text: 'Tümünü sil',
          style: 'destructive',
          onPress: async () => {
            try {
              await clearBiometricDatabase();
              setPeople([]);
              setEnrollments([]);
              setSelectedPersonId(null);
              setFeedback('Biyometrik kayıtlar sıfırlandı.');
            } catch {
              setFeedback('Biyometrik kayıtlar sıfırlanamadı.');
            }
          },
        },
      ]
    );
  }

  // Kaydı önce ekrandan kaldırır, dosya silme başarısız olursa eski listeyi geri yükler.
  async function handleDelete(sampleId: string) {
    const previousSamples = samples;
    const nextVisibleSamples = samples.filter((sample) => sample.id !== sampleId);

    // İyimser güncelleme silme sonucunu uygulama yeniden açılmadan gösterir.
    setSamples(nextVisibleSamples);
    setFeedback(
      nextVisibleSamples.length === 0 ? 'Henüz kayıt yok.' : `${nextVisibleSamples.length} fotoğraf`
    );

    try {
      const nextSamples = await deleteCaptureSample(sampleId);
      setSamples(nextSamples);
    } catch {
      setSamples(previousSamples);
      setFeedback('Kayıt silinemedi.');
    }
  }

  // Elle verilen kalite etiketini kaydedip kart listesini güncel metadata ile yeniler.
  async function handleCalibrationLabel(
    sampleId: string,
    label: QualityCalibrationLabel
  ) {
    try {
      const nextSamples = await setCaptureCalibrationLabel(sampleId, label);
      setSamples(nextSamples);
    } catch {
      setFeedback('Kalite etiketi kaydedilemedi.');
    }
  }

  // Gerçek kişi/farklı el bilgisini aktif test oturumuyla capture metadata'sına yazar.
  async function handleProbeEvaluation(
    sampleId: string,
    target:
      | { relation: 'genuine'; personId: string }
      | { relation: 'impostor' }
      | null
  ) {
    try {
      const probeEvaluation = target
        ? target.relation === 'genuine'
          ? {
              version: 1 as const,
              relation: 'genuine' as const,
              expectedPersonId: target.personId,
              evaluationSessionId: activeProbeSessionId,
              labeledAt: new Date().toISOString(),
            }
          : {
              version: 1 as const,
              relation: 'impostor' as const,
              evaluationSessionId: activeProbeSessionId,
              labeledAt: new Date().toISOString(),
            }
        : null;
      const nextSamples = await setCaptureProbeEvaluation(
        sampleId,
        probeEvaluation
      );
      setSamples(nextSamples);
      setFeedback(
        target
          ? 'Probe etiketi değerlendirme için kaydedildi.'
          : 'Probe etiketi kaldırıldı.'
      );
    } catch {
      setFeedback('Probe etiketi kaydedilemedi.');
    }
  }

  // Sonraki etiketlerin önceki tekrarlarla aynı bağımsız oturum sayılmasını önler.
  function handleStartNewProbeSession() {
    setActiveProbeSessionId(createId('probe-session'));
    setFeedback('Yeni test oturumu başlatıldı. Bundan sonraki etiketler bu oturuma yazılır.');
  }

  // Etiketli kayıtların fotoğraf içermeyen metrik özetini cihaz içinde oluşturur.
  async function handleExportCalibration() {
    try {
      const result = await exportQualityCalibrationSummary();
      setFeedback(
        result.sampleCount > 0
          ? `${result.sampleCount} etiketli çekimin özeti hazırlandı.`
          : 'Özet için önce çekimleri etiketle.'
      );
    } catch {
      setFeedback('Kalibrasyon özeti oluşturulamadı.');
    }
  }

  // Enrollment şablonlarını cihazdan çıkarmadan toplu genuine/impostor baseline metriklerini hesaplar.
  async function handleCreateBaseline() {
    if (isCreatingBaseline) return;
    setIsCreatingBaseline(true);
    setFeedback('Biyometrik baseline hesaplanıyor...');
    try {
      const { report } = await createFingerprintEnrollmentBaselineReport();
      if (report.dataset.personCount === 0) {
        setFeedback('Baseline için geçerli minutiae-v2 enrollment kaydı yok.');
      } else if (report.dataset.personCount < 2) {
        setFeedback(
          'Genuine baseline hazır. Impostor ve FAR için en az iki kayıtlı kişi gerekir.'
        );
      } else {
        const rank1 = report.identification.rank1Rate;
        setFeedback(
          `${report.dataset.personCount} kişilik baseline hazır · rank-1=${rank1 === null ? 'yok' : `${Math.round(rank1 * 100)}%`}`
        );
      }
    } catch (error) {
      console.warn('Biyometrik baseline raporu oluşturulamadı.', error);
      setFeedback('Biyometrik baseline raporu oluşturulamadı.');
    } finally {
      setIsCreatingBaseline(false);
    }
  }

  // Etiketli gerçek giriş çekimlerinde genuine kabul ve açık-küme yanlış kabul oranlarını ölçer.
  async function handleCreateProbeReport() {
    if (isCreatingProbeReport) return;
    setIsCreatingProbeReport(true);
    setFeedback('Etiketli giriş çekimleri ölçülüyor...');
    try {
      const { report } = await createFingerprintProbeBaselineReport();
      if (report.dataset.labeledProbeCount === 0) {
        setFeedback('Probe raporu için önce giriş çekimlerini etiketle.');
      } else if (report.dataset.evaluatedProbeCount === 0) {
        setFeedback(
          `${report.dataset.labeledProbeCount} etiket var; geçerli minutiae-v2 probe veya enrollment bulunamadı.`
        );
      } else {
        const trueIdentificationRate = report.identification.trueIdentificationRate;
        const falseAcceptRate = report.identification.openSetFalseAcceptRate;
        setFeedback(
          `${report.dataset.evaluatedProbeCount} probe ölçüldü · doğru kabul=${formatRateForUi(trueIdentificationRate)} · FAR=${formatRateForUi(falseAcceptRate)}`
        );
      }
    } catch (error) {
      console.warn('Probe baseline raporu oluşturulamadı.', error);
      setFeedback('Probe baseline raporu oluşturulamadı.');
    } finally {
      setIsCreatingProbeReport(false);
    }
  }

  // Galeriden seçilen el görselini modelden geçirip kutu verileriyle kayıt listesine ekler.
  async function handlePickImage() {
    if (isPickingImage) return;

    setIsPickingImage(true);
    setFeedback('Galeri açılıyor...');

    try {
      const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();

      // Galeri izni yoksa sistem seçicisini açmadan kullanıcıyı bilgilendirir.
      if (!permission.granted) {
        setFeedback('Galeriden seçmek için fotoğraf izni vermelisin.');
        return;
      }

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ['images'],
        allowsEditing: false,
        exif: true,
        quality: 1,
        selectionLimit: 1,
      });

      // Kullanıcı seçim ekranını kapattığında mevcut kayıtları değiştirmez.
      if (result.canceled || !result.assets[0]) {
        setFeedback(samples.length === 0 ? 'Henüz kayıt yok.' : `${samples.length} fotoğraf`);
        return;
      }

      const asset = result.assets[0];
      setFeedback('Görsel işleniyor...');
      const sampleId = createId('gallery');
      const normalizedAsset = await normalizePickedImage(asset);
      const rawImageUri = await saveRawImage(normalizedAsset.uri, sampleId);
      const detections = await detectFingertipObbBoxes(rawImageUri);
      const fingerRois = await tryExtractFingerRois({
        imageUri: rawImageUri,
        imageSize: {
          width: normalizedAsset.width,
          height: normalizedAsset.height,
        },
        detections,
        sampleId,
      });

      // Model kutularını ve ROI dosyalarını ham görselden ayrı metadata olarak saklar.
      const sample: CaptureSample = {
        id: sampleId,
        createdAt: new Date().toISOString(),
        rawImageUri,
        rawImageSize: {
          width: normalizedAsset.width,
          height: normalizedAsset.height,
        },
        detections,
        fingerRois,
        deviceModel: `${Device.modelName ?? 'Bilinmeyen cihaz'} · Galeri`,
        fingerLabel: 'unknown',
        sessionId: createId('gallery-session'),
        qualityStatus: getCaptureQualityStatus(fingerRois),
        accepted: false,
      };

      const nextSamples = await appendCaptureSample(sample);
      setSamples(nextSamples);
      setFeedback(`${nextSamples.length} fotoğraf`);
    } catch (error) {
      console.warn('Galeri görseli işlenemedi.', error);
      setFeedback('Görsel işlenemedi.');
    } finally {
      setIsPickingImage(false);
    }
  }

  const contentPlatformStyle = Platform.select({
    android: {
      paddingTop: insets.top + Spacing.four,
      paddingLeft: insets.left,
      paddingRight: insets.right,
      paddingBottom: insets.bottom,
    },
    web: {
      paddingTop: Spacing.six,
      paddingBottom: Spacing.four,
    },
  });

  const selectedPerson = people.find((person) => person.id === selectedPersonId);
  const enrollmentSourceCaptureIds = new Set(
    enrollments.map((enrollment) => enrollment.sourceCaptureId)
  );
  const selectedEnrollmentSamples = selectedPerson
    ? [...
        new Map(
          enrollments
            .filter((enrollment) => enrollment.personId === selectedPerson.id)
            .sort((first, second) => (first.sampleIndex ?? 0) - (second.sampleIndex ?? 0))
            .map((enrollment) => [
              enrollment.sourceCaptureId,
              samples.find((sample) => sample.id === enrollment.sourceCaptureId),
            ])
        ).values(),
      ].filter((sample): sample is CaptureSample => Boolean(sample))
    : [];

  return (
    <FlatList
      data={samples}
      keyExtractor={(sample) => sample.id}
      renderItem={({ item }) => (
        <CaptureCard
          sample={item}
          people={people}
          isEnrollmentSource={enrollmentSourceCaptureIds.has(item.id)}
          onDelete={handleDelete}
          onCalibrationLabel={handleCalibrationLabel}
          onProbeEvaluation={handleProbeEvaluation}
        />
      )}
      style={[styles.list, { backgroundColor: theme.background }]}
      contentInset={insets}
      contentContainerStyle={[styles.contentContainer, contentPlatformStyle]}
      showsVerticalScrollIndicator={false}
      initialNumToRender={2}
      maxToRenderPerBatch={2}
      windowSize={3}
      removeClippedSubviews={Platform.OS === 'android'}
      ListHeaderComponent={
        <>
          <PersonRegistry
            people={people}
            enrollments={enrollments}
            onDelete={handleDeletePerson}
            onClear={handleClearPeople}
            selectedPersonId={selectedPersonId}
            onSelect={setSelectedPersonId}
          />
          {selectedPerson ? (
            <View style={styles.enrollmentPreview}>
              <ThemedText type="subtitle">
                {selectedPerson.displayName} · enrollment çekimleri ({selectedEnrollmentSamples.length}/3)
              </ThemedText>
              {selectedEnrollmentSamples.length > 0 ? (
                selectedEnrollmentSamples.map((sample) => (
                  <CaptureCard
                    key={`enrollment-${sample.id}`}
                    sample={sample}
                    people={people}
                    isEnrollmentSource
                    onDelete={handleDelete}
                    onCalibrationLabel={handleCalibrationLabel}
                    onProbeEvaluation={handleProbeEvaluation}
                  />
                ))
              ) : (
                <ThemedText type="small" themeColor="textSecondary">
                  Kaynak çekim geçmişinde bulunamadı.
                </ThemedText>
              )}
            </View>
          ) : null}
          <View style={styles.header}>
            <View style={styles.headerText}>
              <ThemedText type="subtitle">Kayıtlar</ThemedText>
              <ThemedText type="small" themeColor="textSecondary">
                {feedback}
              </ThemedText>
            </View>

            <View style={styles.headerActions}>
              <Pressable
                style={[
                  styles.summaryButton,
                  isCreatingProbeReport && styles.disabledButton,
                ]}
                disabled={isCreatingProbeReport}
                onPress={handleCreateProbeReport}>
                <ThemedText type="smallBold">
                  {isCreatingProbeReport ? 'Ölçülüyor...' : 'Probe raporu'}
                </ThemedText>
              </Pressable>
              <Pressable
                style={[
                  styles.summaryButton,
                  isCreatingBaseline && styles.disabledButton,
                ]}
                disabled={isCreatingBaseline}
                onPress={handleCreateBaseline}>
                <ThemedText type="smallBold">
                  {isCreatingBaseline ? 'Ölçülüyor...' : 'Baseline'}
                </ThemedText>
              </Pressable>
              <Pressable style={styles.summaryButton} onPress={handleExportCalibration}>
                <ThemedText type="smallBold">Özet</ThemedText>
              </Pressable>
              <Pressable
                style={[styles.galleryButton, isPickingImage && styles.disabledButton]}
                disabled={isPickingImage}
                onPress={handlePickImage}>
                <ThemedText type="smallBold">
                  {isPickingImage ? 'İşleniyor...' : 'Galeriden seç'}
                </ThemedText>
              </Pressable>
            </View>
          </View>
          <View style={styles.probeSessionBar}>
            <View style={styles.headerText}>
              <ThemedText type="smallBold">Aktif test oturumu</ThemedText>
              <ThemedText type="small" themeColor="textSecondary">
                {activeProbeSessionId.slice(-6)} · aynı koşuldaki tekrarları bu oturumda etiketle
              </ThemedText>
            </View>
            <Pressable
              style={styles.summaryButton}
              onPress={handleStartNewProbeSession}>
              <ThemedText type="smallBold">Yeni oturum</ThemedText>
            </Pressable>
          </View>
        </>
      }
      ListFooterComponent={Platform.OS === 'web' ? <WebBadge /> : null}
    />
  );
}

const styles = StyleSheet.create({
  list: {
    flex: 1,
  },
  contentContainer: {
    width: '100%',
    maxWidth: MaxContentWidth,
    alignSelf: 'center',
    flexGrow: 1,
    gap: Spacing.four,
    paddingHorizontal: Spacing.four,
  },
  peopleCard: {
    gap: Spacing.two,
    borderRadius: 8,
    padding: Spacing.three,
  },
  peopleHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: Spacing.two,
  },
  peopleTitle: {
    fontSize: 22,
    lineHeight: 28,
  },
  resetPeopleButton: {
    minHeight: 36,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 6,
    backgroundColor: 'rgba(220, 60, 60, 0.12)',
    paddingHorizontal: Spacing.two,
  },
  personRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: Spacing.two,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: 'rgba(127, 127, 127, 0.25)',
    paddingTop: Spacing.two,
  },
  personRowSelected: {
    borderRadius: 6,
    backgroundColor: 'rgba(32, 138, 239, 0.12)',
    paddingHorizontal: Spacing.two,
  },
  personDetails: {
    flex: 1,
    gap: Spacing.half,
  },
  enrollmentPreview: {
    gap: Spacing.two,
  },
  header: {
    minHeight: 52,
    alignItems: 'stretch',
    gap: Spacing.three,
  },
  headerText: {
    flex: 1,
    minWidth: 0,
    gap: Spacing.half,
  },
  headerActions: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'flex-start',
    gap: Spacing.one,
  },
  probeSessionBar: {
    minHeight: 52,
    flexDirection: 'row',
    flexWrap: 'wrap',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: Spacing.two,
    borderRadius: 6,
    backgroundColor: 'rgba(95, 168, 255, 0.10)',
    padding: Spacing.two,
  },
  galleryButton: {
    minHeight: 40,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 6,
    backgroundColor: 'rgba(127, 127, 127, 0.18)',
    paddingHorizontal: Spacing.three,
  },
  summaryButton: {
    minHeight: 40,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 6,
    backgroundColor: 'rgba(127, 127, 127, 0.18)',
    paddingHorizontal: Spacing.two,
  },
  card: {
    overflow: 'hidden',
    gap: Spacing.two,
    borderRadius: 6,
    padding: Spacing.two,
  },
  roiStrip: {
    gap: Spacing.two,
    paddingRight: Spacing.two,
  },
  roiPreview: {
    width: ROI_PREVIEW_WIDTH,
    gap: Spacing.one,
  },
  roiImagePair: {
    flexDirection: 'row',
    gap: Spacing.one,
  },
  roiImageColumn: {
    width: ROI_THUMBNAIL_WIDTH,
    gap: Spacing.half,
  },
  roiImage: {
    width: ROI_THUMBNAIL_WIDTH,
    height: 72,
    overflow: 'hidden',
    borderRadius: 6,
    backgroundColor: '#000000',
  },
  emptySegmentImage: {
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(127, 127, 127, 0.14)',
    borderWidth: 1,
    borderColor: 'rgba(127, 127, 127, 0.45)',
    borderStyle: 'dashed',
  },
  segmentedRoiImage: {
    borderWidth: 1,
    borderColor: '#2FD16B',
  },
  binaryRoiImage: {
    borderWidth: 1,
    borderColor: '#FFFFFF',
    backgroundColor: '#FFFFFF',
  },
  enhancedRoiImage: {
    borderWidth: 1,
    borderColor: '#5FA8FF',
  },
  orientationRoiImage: {
    borderWidth: 1,
    borderColor: '#A78BFA',
  },
  minutiaeRoiImage: {
    borderWidth: 1,
    borderColor: '#FF912D',
  },
  emptySegmentText: {
    color: '#9CA3AF',
    fontSize: 10,
    lineHeight: 12,
  },
  roiVariantLabel: {
    textAlign: 'center',
    fontSize: 10,
    lineHeight: 12,
  },
  roiLabel: {
    textAlign: 'center',
  },
  qualityLabel: {
    textAlign: 'center',
    fontSize: 10,
    lineHeight: 12,
  },
  qualityGood: {
    color: '#2FD16B',
  },
  qualityMedium: {
    color: '#F5B844',
  },
  qualityPoor: {
    color: '#E5484D',
  },
  calibrationControl: {
    gap: Spacing.one,
    paddingHorizontal: Spacing.one,
  },
  calibrationSegments: {
    minHeight: 34,
    flexDirection: 'row',
    overflow: 'hidden',
    borderRadius: 6,
    borderWidth: 1,
    borderColor: 'rgba(127, 127, 127, 0.35)',
  },
  calibrationSegment: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: Spacing.one,
  },
  calibrationSegmentSelected: {
    backgroundColor: 'rgba(95, 168, 255, 0.25)',
  },
  probeControl: {
    gap: Spacing.one,
    paddingHorizontal: Spacing.one,
  },
  probeControlHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: Spacing.two,
  },
  probeSegments: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: Spacing.one,
  },
  probeSegment: {
    minHeight: 34,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 6,
    borderWidth: 1,
    borderColor: 'rgba(127, 127, 127, 0.35)',
    paddingHorizontal: Spacing.two,
  },
  probeSegmentSelected: {
    borderColor: '#5FA8FF',
    backgroundColor: 'rgba(95, 168, 255, 0.25)',
  },
  impostorProbeSegment: {
    borderColor: 'rgba(229, 72, 77, 0.55)',
  },
  impostorProbeSegmentSelected: {
    backgroundColor: 'rgba(229, 72, 77, 0.20)',
  },
  enrollmentSourceLabel: {
    paddingHorizontal: Spacing.one,
  },
  cardFooter: {
    minHeight: 36,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: Spacing.two,
    paddingHorizontal: Spacing.one,
  },
  deleteText: {
    color: '#E5484D',
  },
  disabledButton: {
    opacity: 0.5,
  },
  inspectionBackdrop: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.96)',
    paddingTop: 48,
    paddingBottom: 28,
    paddingHorizontal: Spacing.three,
  },
  inspectionGestureRoot: {
    flex: 1,
  },
  inspectionHeader: {
    minHeight: 44,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: Spacing.two,
  },
  inspectionTitle: {
    color: '#FFFFFF',
  },
  closeInspectionButton: {
    minHeight: 36,
    justifyContent: 'center',
    borderRadius: 6,
    backgroundColor: 'rgba(255, 255, 255, 0.14)',
    paddingHorizontal: Spacing.two,
  },
  inspectionImageArea: {
    flex: 1,
  },
  inspectionPage: {
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  inspectionImageFrame: {
    width: '100%',
    height: '100%',
  },
  inspectionImage: {
    width: '100%',
    height: '100%',
  },
  inspectionOverlay: {
    position: 'absolute',
    top: 0,
    right: 0,
    bottom: 0,
    left: 0,
  },
  inspectionLegend: {
    color: '#D1D5DB',
    textAlign: 'center',
    paddingVertical: Spacing.two,
  },
  inspectionGestureHint: {
    color: '#9CA3AF',
    textAlign: 'center',
    paddingVertical: Spacing.one,
  },
  inspectionTabs: {
    minHeight: 44,
    flexDirection: 'row',
    overflow: 'hidden',
    borderRadius: 6,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.25)',
  },
  inspectionTab: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  inspectionTabSelected: {
    backgroundColor: 'rgba(95, 168, 255, 0.35)',
  },
});
