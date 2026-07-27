import * as Device from 'expo-device';
import * as ImagePicker from 'expo-image-picker';
import { useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { Platform, Pressable, ScrollView, StyleSheet, View } from 'react-native';
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
  loadCaptureSamples,
  saveRawImage,
} from '@/lib/capture-storage';
import { detectFingertipObbBoxes } from '@/lib/fingertip-detection';
import { extractFingerRoisFromImage } from '@/lib/fingertip-roi';
import type { CaptureSample, DetectedObbBox } from '@/types/biometrics';

// Galeriden eklenen kayıtlar için kısa ve çakışma ihtimali düşük kimlik üretir.
function createId(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

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
        qualityStatus: 'unknown',
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
    <ScrollView
      style={[styles.scrollView, { backgroundColor: theme.background }]}
      contentInset={insets}
      contentContainerStyle={[styles.contentContainer, contentPlatformStyle]}
      showsVerticalScrollIndicator={false}>
      <ThemedView style={styles.container}>
        <View style={styles.header}>
          <View style={styles.headerText}>
            <ThemedText type="subtitle">Kayıtlar</ThemedText>
            <ThemedText type="small" themeColor="textSecondary">
              {feedback}
            </ThemedText>
          </View>

          <Pressable
            style={[styles.galleryButton, isPickingImage && styles.disabledButton]}
            disabled={isPickingImage}
            onPress={handlePickImage}>
            <ThemedText type="smallBold">
              {isPickingImage ? 'İşleniyor...' : 'Galeriden seç'}
            </ThemedText>
          </Pressable>
        </View>

        <View style={styles.list}>
          {samples.map((sample) => (
            <ThemedView key={sample.id} type="backgroundElement" style={styles.card}>
              <DetectionImage
                imageUri={sample.rawImageUri}
                imageSize={sample.rawImageSize}
                detections={sample.detections}
              />

              <View style={styles.cardFooter}>
                <ThemedText type="small" themeColor="textSecondary">
                  {new Date(sample.createdAt).toLocaleString('tr-TR')}
                </ThemedText>
                <Pressable hitSlop={12} onPress={() => handleDelete(sample.id)}>
                  <ThemedText type="smallBold" style={styles.deleteText}>
                    Sil
                  </ThemedText>
                </Pressable>
              </View>
            </ThemedView>
          ))}
        </View>

        {Platform.OS === 'web' && <WebBadge />}
      </ThemedView>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  scrollView: {
    flex: 1,
  },
  contentContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
  },
  container: {
    width: '100%',
    maxWidth: MaxContentWidth,
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
  galleryButton: {
    minHeight: 40,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 6,
    backgroundColor: 'rgba(127, 127, 127, 0.18)',
    paddingHorizontal: Spacing.three,
  },
  list: {
    gap: Spacing.three,
  },
  card: {
    overflow: 'hidden',
    gap: Spacing.two,
    borderRadius: 6,
    padding: Spacing.two,
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
});
