import { Image } from 'expo-image';
import { useFocusEffect } from 'expo-router';
import { useCallback, useState } from 'react';
import { Platform, Pressable, ScrollView, StyleSheet, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { WebBadge } from '@/components/web-badge';
import { BottomTabInset, MaxContentWidth, Spacing } from '@/constants/theme';
import { deleteCaptureSample, loadCaptureSamples } from '@/lib/capture-storage';
import { formatQualityStatus } from '@/lib/quality';
import { useTheme } from '@/hooks/use-theme';
import type { CaptureSample } from '@/types/biometrics';

export default function RecordsScreen() {
  const [samples, setSamples] = useState<CaptureSample[]>([]);
  const [feedback, setFeedback] = useState('Kayıtlar yükleniyor...');
  const safeAreaInsets = useSafeAreaInsets();
  const theme = useTheme();
  const insets = {
    ...safeAreaInsets,
    bottom: safeAreaInsets.bottom + BottomTabInset + Spacing.three,
  };

  const loadSamples = useCallback(() => {
    loadCaptureSamples()
      .then((storedSamples) => {
        setSamples(storedSamples);
        setFeedback(
          storedSamples.length === 0
            ? 'Henüz kayıt yok. Ana ekrandan kare yakalayabilirsin.'
            : `${storedSamples.length} kayıt cihaz içinde saklanıyor.`
        );
      })
      .catch(() => {
        setFeedback('Kayıtlar okunamadı.');
      });
  }, []);

  useFocusEffect(loadSamples);

  async function handleDelete(sampleId: string) {
    try {
      const nextSamples = await deleteCaptureSample(sampleId);
      setSamples(nextSamples);
      setFeedback(
        nextSamples.length === 0
          ? 'Kayıt silindi. Henüz kayıt yok.'
          : `Kayıt silindi. ${nextSamples.length} kayıt kaldı.`
      );
    } catch {
      setFeedback('Kayıt silinemedi.');
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
      contentContainerStyle={[styles.contentContainer, contentPlatformStyle]}>
      <ThemedView style={styles.container}>
        <ThemedView style={styles.header}>
          <ThemedText type="code" style={styles.eyebrow}>
            yerel capture arşivi
          </ThemedText>
          <ThemedText type="subtitle">Kayıtlar</ThemedText>
          <ThemedText themeColor="textSecondary">{feedback}</ThemedText>
        </ThemedView>

        <View style={styles.list}>
          {samples.map((sample) => (
            <ThemedView key={sample.id} type="backgroundElement" style={styles.card}>
              <View style={styles.imageGrid}>
                <View style={styles.imageColumn}>
                  <ThemedText type="smallBold">Ham görüntü</ThemedText>
                  <Image source={{ uri: sample.rawImageUri }} style={styles.rawImage} contentFit="cover" />
                </View>
                <View style={styles.imageColumn}>
                  <ThemedText type="smallBold">ROI</ThemedText>
                  <Image source={{ uri: sample.roiImageUri }} style={styles.roiImage} contentFit="contain" />
                </View>
              </View>

              <View style={styles.metaGrid}>
                <Metric label="Tarih" value={new Date(sample.createdAt).toLocaleString('tr-TR')} />
                <Metric label="Cihaz" value={sample.deviceModel ?? 'Bilinmeyen cihaz'} />
                <Metric label="Netlik" value={`${sample.qualityMetrics?.blurScore ?? 0}/100`} />
                <Metric
                  label="Parlama"
                  value={`%${Math.round((sample.qualityMetrics?.glareRatio ?? 0) * 100)}`}
                />
                <Metric label="Işık" value={`${sample.qualityMetrics?.brightnessMean ?? 0}/255`} />
                <Metric
                  label="Kalite"
                  value={sample.qualityMetrics ? formatQualityStatus(sample.qualityMetrics.status) : 'ölçülmedi'}
                />
              </View>

              <Pressable style={styles.deleteButton} onPress={() => handleDelete(sample.id)}>
                <ThemedText type="smallBold" style={styles.deleteText}>
                  Kaydı sil
                </ThemedText>
              </Pressable>
            </ThemedView>
          ))}
        </View>

        {Platform.OS === 'web' && <WebBadge />}
      </ThemedView>
    </ScrollView>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <ThemedView type="backgroundSelected" style={styles.metricRow}>
      <ThemedText type="smallBold">{label}</ThemedText>
      <ThemedText type="small" themeColor="textSecondary" style={styles.metricValue}>
        {value}
      </ThemedText>
    </ThemedView>
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
    maxWidth: MaxContentWidth,
    flexGrow: 1,
    gap: Spacing.three,
    paddingHorizontal: Spacing.four,
  },
  header: {
    gap: Spacing.two,
  },
  eyebrow: {
    textTransform: 'uppercase',
  },
  list: {
    gap: Spacing.three,
  },
  card: {
    gap: Spacing.three,
    padding: Spacing.three,
    borderRadius: Spacing.four,
  },
  imageGrid: {
    flexDirection: 'row',
    gap: Spacing.two,
  },
  imageColumn: {
    flex: 1,
    gap: Spacing.two,
  },
  rawImage: {
    width: '100%',
    height: 180,
    borderRadius: Spacing.three,
    backgroundColor: '#000000',
  },
  roiImage: {
    width: '100%',
    height: 180,
    borderRadius: Spacing.three,
    backgroundColor: '#000000',
  },
  metaGrid: {
    gap: Spacing.two,
  },
  metricRow: {
    minHeight: 40,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: Spacing.two,
    borderRadius: Spacing.two,
    paddingHorizontal: Spacing.three,
  },
  metricValue: {
    flex: 1,
    textAlign: 'right',
  },
  deleteButton: {
    minHeight: 44,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: Spacing.two,
    backgroundColor: '#B42318',
  },
  deleteText: {
    color: '#ffffff',
  },
});
