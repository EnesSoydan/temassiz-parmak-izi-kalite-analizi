import { CameraView, useCameraPermissions } from 'expo-camera';
import { useState } from 'react';
import { Platform, Pressable, ScrollView, StyleSheet, View } from 'react-native';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';

import { HintRow } from '@/components/hint-row';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { WebBadge } from '@/components/web-badge';
import { BottomTabInset, MaxContentWidth, Spacing } from '@/constants/theme';

export default function HomeScreen() {
  const [isCaptureReady, setIsCaptureReady] = useState(false);
  const [permission, requestPermission] = useCameraPermissions();
  const insets = useSafeAreaInsets();

  const canShowCamera = isCaptureReady && permission?.granted;

  async function handleCapturePress() {
    if (!isCaptureReady && !permission?.granted) {
      const nextPermission = await requestPermission();
      setIsCaptureReady(nextPermission.granted);
      return;
    }

    setIsCaptureReady((current) => !current);
  }

  return (
    <ThemedView style={styles.container}>
      <SafeAreaView style={[styles.safeArea, { paddingTop: insets.top + Spacing.four }]}>
        <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
          <ThemedView style={styles.heroSection}>
            <ThemedText type="code" style={styles.eyebrow}>
              capture quality prototype
            </ThemedText>
            <ThemedText type="title" style={styles.title}>
              Temassız parmak izi kalite analizi
            </ThemedText>
            <ThemedText themeColor="textSecondary" style={styles.description}>
              İlk hedefimiz doğrulama yapmak değil; kameradan alınan görüntünün
              netlik, ölçek, parlama ve kadraj açısından kullanılabilir olup
              olmadığını ölçmek.
            </ThemedText>
          </ThemedView>

          <ThemedText type="code" style={styles.code}>
            ilk deney zinciri
          </ThemedText>

          <ThemedView type="backgroundElement" style={styles.stepContainer}>
            <HintRow
              title="1. Kadraj"
              hint={<ThemedText type="small">parmak ROI içinde mi?</ThemedText>}
            />
            <HintRow
              title="2. Kalite"
              hint={<ThemedText type="small">netlik ve parlama</ThemedText>}
            />
            <HintRow
              title="3. Seçim"
              hint={<ThemedText type="small">en iyi kareyi sakla</ThemedText>}
            />
            <HintRow
              title="4. Sonra"
              hint={<ThemedText type="small">matcher/model deneyi</ThemedText>}
            />
          </ThemedView>

          <ThemedView type="backgroundElement" style={styles.captureCard}>
            {canShowCamera ? (
              <View style={styles.cameraFrame}>
                <CameraView style={styles.cameraPreview} facing="back" />
                <View pointerEvents="none" style={styles.cameraOverlay}>
                  <View style={styles.roiBox} />
                  <ThemedText type="smallBold" style={styles.overlayText}>
                    Parmağını kutunun içine hizala
                  </ThemedText>
                </View>
              </View>
            ) : (
              <ThemedView type="backgroundSelected" style={styles.fingerGuide}>
                <ThemedText type="code" themeColor="textSecondary">
                  ROI
                </ThemedText>
              </ThemedView>
            )}

            <ThemedView style={styles.captureContent}>
              <ThemedText type="smallBold">Capture alanı</ThemedText>
              <ThemedText type="small" themeColor="textSecondary">
                {canShowCamera
                  ? 'Canlı kamera açık. Şimdilik yalnızca kadraj alanını test ediyoruz.'
                  : 'Kamerayı açınca önce parmak ROI kadrajını kontrol edeceğiz.'}
              </ThemedText>
            </ThemedView>

            <Pressable style={styles.captureButton} onPress={handleCapturePress}>
              <ThemedText type="smallBold" style={styles.captureButtonText}>
                {canShowCamera ? 'Kamerayı kapat' : 'Kamerayı aç'}
              </ThemedText>
            </Pressable>
          </ThemedView>

          {Platform.OS === 'web' && <WebBadge />}
        </ScrollView>
      </SafeAreaView>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    flexDirection: 'row',
  },
  safeArea: {
    flex: 1,
    paddingHorizontal: Spacing.four,
    justifyContent: 'flex-start',
    maxWidth: MaxContentWidth,
  },
  scrollContent: {
    gap: Spacing.three,
    paddingBottom: BottomTabInset + Spacing.five,
  },
  heroSection: {
    alignItems: 'flex-start',
    justifyContent: 'center',
    paddingHorizontal: Spacing.four,
    gap: Spacing.three,
  },
  eyebrow: {
    textTransform: 'uppercase',
  },
  title: {
    maxWidth: 640,
  },
  description: {
    maxWidth: 640,
  },
  code: {
    textTransform: 'uppercase',
    paddingHorizontal: Spacing.four,
  },
  stepContainer: {
    gap: Spacing.three,
    alignSelf: 'stretch',
    paddingHorizontal: Spacing.three,
    paddingVertical: Spacing.four,
    borderRadius: Spacing.four,
  },
  captureCard: {
    alignSelf: 'stretch',
    gap: Spacing.three,
    paddingHorizontal: Spacing.three,
    paddingVertical: Spacing.three,
    borderRadius: Spacing.four,
  },
  fingerGuide: {
    minHeight: 220,
    borderRadius: Spacing.three,
    alignItems: 'center',
    justifyContent: 'center',
  },
  cameraFrame: {
    minHeight: 280,
    overflow: 'hidden',
    borderRadius: Spacing.three,
    backgroundColor: '#000000',
  },
  cameraPreview: {
    flex: 1,
  },
  cameraOverlay: {
    ...StyleSheet.absoluteFill,
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.three,
    padding: Spacing.three,
  },
  roiBox: {
    width: '42%',
    minWidth: 112,
    maxWidth: 160,
    aspectRatio: 0.62,
    borderWidth: 2,
    borderColor: '#ffffff',
    borderRadius: Spacing.three,
    backgroundColor: 'transparent',
  },
  overlayText: {
    color: '#ffffff',
    textAlign: 'center',
    textShadowColor: '#000000',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 4,
  },
  captureContent: {
    gap: Spacing.half,
  },
  captureButton: {
    minHeight: 48,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: Spacing.two,
    backgroundColor: '#208AEF',
    paddingHorizontal: Spacing.three,
  },
  captureButtonText: {
    color: '#ffffff',
  },
});
