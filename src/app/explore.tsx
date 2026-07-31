import * as Device from 'expo-device';
import { Image } from 'expo-image';
import * as ImagePicker from 'expo-image-picker';
import { useFocusEffect } from 'expo-router';
import { useCallback, useRef, useState } from 'react';
import {
  FlatList,
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
  appendCaptureSample,
  deleteCaptureSample,
  exportQualityCalibrationSummary,
  loadCaptureSamples,
  saveRawImage,
  setCaptureCalibrationLabel,
} from '@/lib/capture-storage';
import { detectFingertipObbBoxes } from '@/lib/fingertip-detection';
import { sortFingerRois } from '@/lib/finger-order';
import { extractFingerRoisFromImage } from '@/lib/fingertip-roi';
import {
  formatFingerprintQualityStatus,
  getCaptureQualityStatus,
} from '@/lib/fingerprint-quality';
import type {
  CaptureSample,
  DetectedObbBox,
  FingerRoi,
  QualityCalibrationLabel,
} from '@/types/biometrics';

// Galeriden eklenen kayıtlar için kısa ve çakışma ihtimali düşük kimlik üretir.
function createId(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
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
  | 'segmented'
  | 'orientation'
  | 'enhanced'
  | 'minutiae';

// Tam ekran inceleyicide gösterilebilen ROI sürümlerini kullanıcı etiketleriyle tanımlar.
const ROI_INSPECTION_OPTIONS: {
  value: RoiInspectionVariant;
  label: string;
}[] = [
  { value: 'roi', label: 'ROI' },
  { value: 'canonical', label: 'Kanonik' },
  { value: 'segmented', label: 'Seg' },
  { value: 'orientation', label: 'Yön' },
  { value: 'enhanced', label: 'Ridge' },
  { value: 'minutiae', label: 'Minutiae' },
];

// Seçilen inceleme sürümünün parmak ROI metadata'sındaki dosya yolunu döndürür.
function getRoiInspectionUri(
  fingerRoi: FingerRoi,
  variant: RoiInspectionVariant
) {
  if (variant === 'roi') return fingerRoi.imageUri;
  if (variant === 'canonical') return fingerRoi.canonicalImageUri;
  if (variant === 'segmented') return fingerRoi.segmentedImageUri;
  if (variant === 'orientation') return fingerRoi.orientationImageUri;
  if (variant === 'enhanced') return fingerRoi.enhancedImageUri;
  return fingerRoi.minutiaeImageUri;
}

// Tek bir kaydın ham fotoğrafını, ROI çiftlerini ve kalite durumlarını gösterir.
function CaptureCard({
  sample,
  onDelete,
  onCalibrationLabel,
}: {
  sample: CaptureSample;
  onDelete: (sampleId: string) => void;
  onCalibrationLabel: (sampleId: string, label: QualityCalibrationLabel) => void;
}) {
  const inspectionScrollRef = useRef<ScrollView>(null);
  const { width: windowWidth } = useWindowDimensions();
  const [inspection, setInspection] = useState<{
    fingerRoi: FingerRoi;
    variant: RoiInspectionVariant;
  } | null>(null);
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
    setInspection({ fingerRoi, variant });
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
                    {fingerRoi.canonicalImageUri ? (
                      <Pressable
                        onPress={() => openInspection(fingerRoi, 'canonical')}>
                        <Image
                          source={{ uri: fingerRoi.canonicalImageUri }}
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
                      Kanonik
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
        onRequestClose={() => setInspection(null)}>
        <View style={styles.inspectionBackdrop}>
          <View style={styles.inspectionHeader}>
            <ThemedText type="smallBold" style={styles.inspectionTitle}>
              {inspection
                ? `${formatFingerRoiClass(inspection.fingerRoi.className)} parmak`
                : 'Parmak ROI'}
            </ThemedText>
            <Pressable
              style={styles.closeInspectionButton}
              onPress={() => setInspection(null)}>
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
                        <Image
                          source={{ uri: imageUri }}
                          style={styles.inspectionImage}
                          contentFit="contain"
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

export default function RecordsScreen() {
  const [samples, setSamples] = useState<CaptureSample[]>([]);
  const [feedback, setFeedback] = useState('Kayıtlar yükleniyor...');
  const [isPickingImage, setIsPickingImage] = useState(false);
  const safeAreaInsets = useSafeAreaInsets();
  const theme = useTheme();
  const insets = {
    ...safeAreaInsets,
    bottom: safeAreaInsets.bottom + BottomTabInset + Spacing.three,
  };

  // Kayıtlar sekmesi açıldığında cihaz içindeki fotoğraf listesini yeniden okur.
  const loadSamples = useCallback(() => {
    loadCaptureSamples()
      .then((storedSamples) => {
        setSamples(storedSamples);
        setFeedback(
          storedSamples.length === 0 ? 'Henüz kayıt yok.' : `${storedSamples.length} fotoğraf`
        );
      })
      .catch(() => {
        setFeedback('Kayıtlar okunamadı.');
      });
  }, []);

  useFocusEffect(loadSamples);

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
      const rawImageUri = await saveRawImage(asset.uri, sampleId);
      const detections = await detectFingertipObbBoxes(rawImageUri);
      const fingerRois = await tryExtractFingerRois({
        imageUri: rawImageUri,
        imageSize: {
          width: asset.width,
          height: asset.height,
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
          width: asset.width,
          height: asset.height,
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

  return (
    <FlatList
      data={samples}
      keyExtractor={(sample) => sample.id}
      renderItem={({ item }) => (
        <CaptureCard
          sample={item}
          onDelete={handleDelete}
          onCalibrationLabel={handleCalibrationLabel}
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
        <View style={styles.header}>
          <View style={styles.headerText}>
            <ThemedText type="subtitle">Kayıtlar</ThemedText>
            <ThemedText type="small" themeColor="textSecondary">
              {feedback}
            </ThemedText>
          </View>

          <View style={styles.headerActions}>
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
  header: {
    minHeight: 52,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: Spacing.three,
  },
  headerText: {
    flex: 1,
    gap: Spacing.half,
  },
  headerActions: {
    flexDirection: 'row',
    gap: Spacing.one,
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
    width: 336,
    gap: Spacing.one,
  },
  roiImagePair: {
    flexDirection: 'row',
    gap: Spacing.one,
  },
  roiImageColumn: {
    width: 52,
    gap: Spacing.half,
  },
  roiImage: {
    width: 52,
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
  },
  inspectionImage: {
    width: '100%',
    height: '100%',
  },
  inspectionLegend: {
    color: '#D1D5DB',
    textAlign: 'center',
    paddingVertical: Spacing.two,
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
