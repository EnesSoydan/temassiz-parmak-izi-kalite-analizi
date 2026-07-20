import { CameraView, useCameraPermissions } from 'expo-camera';
import * as Device from 'expo-device';
import { Image } from 'expo-image';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Platform, Pressable, ScrollView, StyleSheet, View } from 'react-native';
import { SafeAreaView, useSafeAreaInsets } from 'react-native-safe-area-context';

import { HintRow } from '@/components/hint-row';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { WebBadge } from '@/components/web-badge';
import { BottomTabInset, MaxContentWidth, Spacing } from '@/constants/theme';
import { appendCaptureSample, loadCaptureSamples, saveRawImage, saveRoiImage } from '@/lib/capture-storage';
import { cropFingerRoi } from '@/lib/image-processing';
import { analyzeFrameQuality, estimateInitialQuality, formatQualityStatus } from '@/lib/quality';
import type { CaptureSample, QualityMetrics } from '@/types/biometrics';

type CheckStatus = 'idle' | 'watching' | 'good' | 'warning' | 'bad';

type QualityCheckProps = {
  label: string;
  value: string;
  status: CheckStatus;
};

function QualityCheck({ label, value, status }: QualityCheckProps) {
  return (
    <ThemedView type="backgroundSelected" style={styles.qualityRow}>
      <View
        style={[
          styles.statusDot,
          status === 'good' && styles.statusDotGood,
          status === 'watching' && styles.statusDotActive,
          status === 'warning' && styles.statusDotWarning,
          status === 'bad' && styles.statusDotBad,
          status === 'idle' && styles.statusDotIdle,
        ]}
      />
      <ThemedText type="smallBold" style={styles.qualityLabel}>
        {label}
      </ThemedText>
      <ThemedText type="small" themeColor="textSecondary" style={styles.qualityValue}>
        {value}
      </ThemedText>
    </ThemedView>
  );
}

function createId(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function withTimeout<T>(promise: Promise<T>, timeoutMs: number) {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) => {
      setTimeout(() => reject(new Error('timeout')), timeoutMs);
    }),
  ]);
}

function getMetricStatus(metrics: QualityMetrics | null, field: 'blur' | 'glare' | 'brightness'): CheckStatus {
  if (!metrics) {
    return 'idle';
  }

  if (field === 'blur') {
    if (metrics.blurScore < 18) return 'bad';
    if (metrics.blurScore < 35) return 'warning';
    return 'good';
  }

  if (field === 'glare') {
    if (metrics.glareRatio > 0.08) return 'bad';
    if (metrics.glareRatio > 0.035) return 'warning';
    return 'good';
  }

  if (metrics.brightnessMean < 45 || metrics.brightnessMean > 225) return 'bad';
  if (metrics.brightnessMean < 75 || metrics.brightnessMean > 205) return 'warning';
  return 'good';
}

function getQualityDotStatus(metrics: QualityMetrics | null): CheckStatus {
  if (!metrics) return 'idle';
  if (metrics.status === 'good') return 'good';
  if (metrics.status === 'usable') return 'warning';
  if (metrics.status === 'poor') return 'bad';
  return 'idle';
}

export default function HomeScreen() {
  const cameraRef = useRef<CameraView>(null);
  const isAnalyzingRef = useRef(false);
  const [isCaptureReady, setIsCaptureReady] = useState(false);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [isTakingPhoto, setIsTakingPhoto] = useState(false);
  const [samples, setSamples] = useState<CaptureSample[]>([]);
  const [selectedSample, setSelectedSample] = useState<CaptureSample | null>(null);
  const [liveMetrics, setLiveMetrics] = useState<QualityMetrics | null>(null);
  const [analysisStatus, setAnalysisStatus] = useState('Ölçüm bekleniyor.');
  const [feedback, setFeedback] = useState('Kamerayı açınca önce parmak ROI kadrajını kontrol edeceğiz.');
  const [permission, requestPermission] = useCameraPermissions();
  const insets = useSafeAreaInsets();

  const canShowCamera = isCaptureReady && permission?.granted;
  const canAnalyzeCamera = canShowCamera && isCameraReady;
  const visibleMetrics = canShowCamera ? liveMetrics : null;

  useEffect(() => {
    loadCaptureSamples()
      .then((storedSamples) => {
        setSamples(storedSamples);
        setSelectedSample(storedSamples[0] ?? null);
      })
      .catch(() => {
        setFeedback('Yerel kayıtlar okunamadı.');
      });
  }, []);

  const analyzeLiveFrame = useCallback(async () => {
    if (!cameraRef.current || !canAnalyzeCamera || isTakingPhoto || isAnalyzingRef.current) {
      return;
    }

    isAnalyzingRef.current = true;
    setAnalysisStatus('Örnek kare alınıyor...');

    try {
      const picture = await withTimeout(
        cameraRef.current.takePictureAsync({
          base64: true,
          quality: 0.2,
          exif: false,
          skipProcessing: false,
          shutterSound: false,
        }),
        5000
      );

      if (picture.base64) {
        const metrics = analyzeFrameQuality({ base64: picture.base64 });
        setLiveMetrics(metrics);
        setAnalysisStatus(`Son ölçüm: ${new Date().toLocaleTimeString('tr-TR')}`);
      } else {
        setAnalysisStatus('Örnek kare alındı ama analiz verisi gelmedi.');
      }
    } catch {
      setAnalysisStatus('Canlı ölçüm zaman aşımına düştü.');
    } finally {
      isAnalyzingRef.current = false;
    }
  }, [canAnalyzeCamera, isTakingPhoto]);

  useEffect(() => {
    if (!canAnalyzeCamera) {
      return;
    }

    void analyzeLiveFrame();
    const intervalId = setInterval(() => {
      void analyzeLiveFrame();
    }, 2500);

    return () => clearInterval(intervalId);
  }, [analyzeLiveFrame, canAnalyzeCamera]);

  async function handleCapturePress() {
    if (!isCaptureReady && !permission?.granted) {
      const nextPermission = await requestPermission();
      setIsCaptureReady(nextPermission.granted);
      setFeedback(
        nextPermission.granted
          ? 'Kamera açık. Parmağını ROI kutusuna hizala.'
          : 'Kamera izni verilmedi. Yakalama yapılamaz.'
      );
      return;
    }

    if (isCaptureReady) {
      setLiveMetrics(null);
      setIsCameraReady(false);
      setAnalysisStatus('Ölçüm bekleniyor.');
    }

    setIsCaptureReady((current) => !current);
    setFeedback(isCaptureReady ? 'Kamera kapatıldı.' : 'Kamera açık. Parmağını ROI kutusuna hizala.');
  }

  async function handleTakePhoto() {
    if (!cameraRef.current || !canShowCamera || isTakingPhoto) {
      return;
    }

    setIsTakingPhoto(true);
    setFeedback('Kare yakalanıyor...');

    try {
      const picture = await cameraRef.current.takePictureAsync({
        quality: 1,
        exif: false,
        skipProcessing: false,
        shutterSound: false,
      });

      const sampleId = createId('sample');
      const sessionId = createId('session');
      const roi = await cropFingerRoi({
        imageUri: picture.uri,
        imageWidth: picture.width,
        imageHeight: picture.height,
      });
      const rawImageUri = await saveRawImage(picture.uri, sampleId);
      const roiImageUri = await saveRoiImage(roi.uri, sampleId);
      const fallbackMetrics = estimateInitialQuality({
        imageWidth: picture.width,
        imageHeight: picture.height,
        roiWidth: roi.crop.width,
        roiHeight: roi.crop.height,
      });
      const qualityMetrics = liveMetrics ?? fallbackMetrics;

      const sample: CaptureSample = {
        id: sampleId,
        createdAt: new Date().toISOString(),
        rawImageUri,
        roiImageUri,
        qualityMetrics,
        deviceModel: Device.modelName ?? 'Bilinmeyen cihaz',
        fingerLabel: 'unknown',
        sessionId,
        qualityStatus: qualityMetrics.status,
        accepted: qualityMetrics.status !== 'poor' && qualityMetrics.status !== 'unknown',
      };

      const nextSamples = await appendCaptureSample(sample);
      setSamples(nextSamples);
      setSelectedSample(sample);
      setFeedback(
        sample.accepted
          ? 'Kare kaydedildi. Canlı kalite ölçümü kayda işlendi.'
          : 'Kare kaydedildi ancak kalite düşük veya ölçüm belirsiz.'
      );
    } catch {
      setFeedback('Kare yakalanamadı. Kamerayı sabitleyip tekrar dene.');
    } finally {
      setIsTakingPhoto(false);
    }
  }

  return (
    <ThemedView style={styles.container}>
      <SafeAreaView style={[styles.safeArea, { paddingTop: insets.top + Spacing.four }]}>
        <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
          <ThemedView style={styles.heroSection}>
            <ThemedText type="code" style={styles.eyebrow}>
              kamera kalite prototipi
            </ThemedText>
            <ThemedText type="title" style={styles.title}>
              Temassız parmak izi kalite analizi
            </ThemedText>
            <ThemedText themeColor="textSecondary" style={styles.description}>
              Kamera açıkken ROI bölgesinden düzenli örnek kare alıp netlik,
              parlama ve ışık durumunu canlı izliyoruz.
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
              hint={<ThemedText type="small">canlı netlik ve parlama</ThemedText>}
            />
            <HintRow
              title="3. Seçim"
              hint={<ThemedText type="small">en iyi kareyi sakla</ThemedText>}
            />
            <HintRow
              title="4. Sonra"
              hint={<ThemedText type="small">enrollment ve 1:N arama</ThemedText>}
            />
          </ThemedView>

          <ThemedView type="backgroundElement" style={styles.captureCard}>
            {canShowCamera ? (
              <View style={styles.cameraFrame}>
                <CameraView
                  ref={cameraRef}
                  style={styles.cameraPreview}
                  facing="back"
                  animateShutter={false}
                  onCameraReady={() => {
                    setIsCameraReady(true);
                    setAnalysisStatus('Kamera hazır. Canlı ölçüm başlıyor.');
                  }}
                  onMountError={() => {
                    setIsCameraReady(false);
                    setAnalysisStatus('Kamera başlatılamadı.');
                  }}
                />
                <View pointerEvents="none" style={styles.cameraOverlay}>
                  <View style={styles.roiBox} />
                  <ThemedText type="smallBold" style={styles.overlayText}>
                    Parmağını kutunun içine hizala
                  </ThemedText>
                  <ThemedText type="small" style={styles.overlayHint}>
                    Netlik kırmızıysa telefonu biraz uzaklaştır
                  </ThemedText>
                  <ThemedText type="small" style={styles.overlayHint}>
                    Parlama kırmızıysa ışığı değiştir ve telefonu sabit tut
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
              <ThemedText type="smallBold">Yakalama alanı</ThemedText>
              <ThemedText type="small" themeColor="textSecondary">
                {feedback}
              </ThemedText>
            </ThemedView>

            <View style={styles.qualityList}>
              <QualityCheck
                label="Kadraj"
                value={canAnalyzeCamera ? 'izleniyor' : canShowCamera ? 'kamera hazırlanıyor' : 'kamera bekleniyor'}
                status={canAnalyzeCamera ? 'watching' : 'idle'}
              />
              <QualityCheck
                label="Netlik"
                value={visibleMetrics ? `${visibleMetrics.blurScore}/100` : 'ölçüm bekleniyor'}
                status={getMetricStatus(visibleMetrics, 'blur')}
              />
              <QualityCheck
                label="Parlama"
                value={visibleMetrics ? `%${Math.round(visibleMetrics.glareRatio * 100)}` : 'ölçüm bekleniyor'}
                status={getMetricStatus(visibleMetrics, 'glare')}
              />
              <QualityCheck
                label="Işık"
                value={visibleMetrics ? `${visibleMetrics.brightnessMean}/255` : 'ölçüm bekleniyor'}
                status={getMetricStatus(visibleMetrics, 'brightness')}
              />
              <QualityCheck
                label="Kalite"
                value={visibleMetrics ? formatQualityStatus(visibleMetrics.status) : 'henüz yok'}
                status={getQualityDotStatus(visibleMetrics)}
              />
              <QualityCheck
                label="Ölçüm"
                value={analysisStatus}
                status={visibleMetrics ? 'good' : canAnalyzeCamera ? 'watching' : 'idle'}
              />
            </View>

            <View style={styles.buttonRow}>
              <Pressable style={styles.secondaryButton} onPress={handleCapturePress}>
                <ThemedText type="smallBold" style={styles.secondaryButtonText}>
                  {canShowCamera ? 'Kamerayı kapat' : 'Kamerayı aç'}
                </ThemedText>
              </Pressable>
              <Pressable
                style={[styles.captureButton, (!canShowCamera || isTakingPhoto) && styles.disabledButton]}
                disabled={!canShowCamera || isTakingPhoto}
                onPress={handleTakePhoto}>
                <ThemedText type="smallBold" style={styles.captureButtonText}>
                  {isTakingPhoto ? 'Kaydediliyor...' : 'Kare yakala'}
                </ThemedText>
              </Pressable>
            </View>
          </ThemedView>

          {selectedSample && (
            <ThemedView type="backgroundElement" style={styles.previewCard}>
              <ThemedText type="smallBold">Son yakalanan ROI</ThemedText>
              <Image source={{ uri: selectedSample.roiImageUri }} style={styles.roiPreview} contentFit="contain" />
              <ThemedText type="small" themeColor="textSecondary">
                {new Date(selectedSample.createdAt).toLocaleString('tr-TR')} · {selectedSample.deviceModel}
              </ThemedText>
            </ThemedView>
          )}

          <ThemedView type="backgroundElement" style={styles.previewCard}>
            <ThemedText type="smallBold">Yerel kayıtlar</ThemedText>
            <ThemedText type="small" themeColor="textSecondary">
              {samples.length === 0
                ? 'Henüz kayıt yok. Kamera açıp kare yakala.'
                : `${samples.length} kayıt cihaz içinde saklanıyor.`}
            </ThemedText>
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
    paddingBottom: BottomTabInset + Spacing.six,
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
    height: 340,
    overflow: 'hidden',
    borderRadius: Spacing.three,
    backgroundColor: '#000000',
  },
  cameraPreview: {
    width: '100%',
    height: '100%',
  },
  cameraOverlay: {
    ...StyleSheet.absoluteFill,
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.two,
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
  overlayHint: {
    color: '#ffffff',
    textAlign: 'center',
    textShadowColor: '#000000',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 4,
  },
  captureContent: {
    gap: Spacing.half,
  },
  qualityList: {
    gap: Spacing.two,
  },
  qualityRow: {
    minHeight: 44,
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.two,
    borderRadius: Spacing.two,
    paddingHorizontal: Spacing.three,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  statusDotIdle: {
    backgroundColor: '#8A8F98',
  },
  statusDotActive: {
    backgroundColor: '#2FD16B',
  },
  statusDotGood: {
    backgroundColor: '#2FD16B',
  },
  statusDotWarning: {
    backgroundColor: '#F6A623',
  },
  statusDotBad: {
    backgroundColor: '#EF4444',
  },
  qualityLabel: {
    minWidth: 72,
  },
  qualityValue: {
    flex: 1,
    textAlign: 'right',
  },
  buttonRow: {
    flexDirection: 'row',
    gap: Spacing.two,
  },
  secondaryButton: {
    minHeight: 48,
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: Spacing.two,
    backgroundColor: '#3A3D44',
    paddingHorizontal: Spacing.three,
  },
  secondaryButtonText: {
    color: '#ffffff',
  },
  captureButton: {
    minHeight: 48,
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: Spacing.two,
    backgroundColor: '#208AEF',
    paddingHorizontal: Spacing.three,
  },
  disabledButton: {
    opacity: 0.5,
  },
  captureButtonText: {
    color: '#ffffff',
  },
  previewCard: {
    alignSelf: 'stretch',
    gap: Spacing.three,
    paddingHorizontal: Spacing.three,
    paddingVertical: Spacing.three,
    borderRadius: Spacing.four,
  },
  roiPreview: {
    width: '100%',
    height: 360,
    borderRadius: Spacing.three,
    backgroundColor: '#000000',
  },
});
