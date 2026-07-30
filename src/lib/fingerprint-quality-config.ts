// Saha verileri geldikçe kalite kararlarının tek yerden kalibre edilmesini sağlayan eşikler.
export const FINGERPRINT_QUALITY_THRESHOLDS = {
  minCoverage: 0.18,
  maxCoverage: 0.985,
  goodScore: 62,
  mediumScore: 34,
  dimBrightness: 58,
  brightBrightness: 210,
  glareRatio: 0.035,
  weakBlur: 32,
  weakTexture: 28,
  weakOrientation: 30,
  weakRidgePeriodicity: 24,
  weakRidgeFrequencyConsistency: 20,
  weakRidgeValidBlockRatio: 8,
  weakContrast: 30,
  captureGoodValidRatio: 18,
  captureMediumValidRatio: 8,
  biometricValidRatio: 22,
  biometricScore: 70,
  biometricOrientation: 45,
  biometricPeriodicity: 55,
  biometricFrequencyConsistency: 50,
  biometricResolution: 45,
  biometricOrientationFallbackValidRatio: 26,
} as const;

// Küçük ROI'leri büyük parmaklarla aynı sabit blok sayısına zorlamadan gereken kanıtı hesaplar.
export function getScaleAwareEvidenceMinimums(candidateBlockCount: number) {
  return {
    medium: Math.max(4, Math.ceil(candidateBlockCount * 0.08)),
    good: Math.max(8, Math.ceil(candidateBlockCount * 0.18)),
    biometric: Math.max(12, Math.ceil(candidateBlockCount * 0.22)),
  };
}

// Doğrulanmış yerel ridge kanıtının kıvrımlı ROI'deki düşük global yön özetini telafi edip edemeyeceğini belirler.
export function hasStrongLocalOrientationEvidence({
  ridgePeriodicity,
  ridgeFrequencyConsistency,
  ridgeValidBlockRatio,
  ridgeValidBlockCount,
  ridgeCandidateBlockCount,
}: {
  ridgePeriodicity: number;
  ridgeFrequencyConsistency: number;
  ridgeValidBlockRatio: number;
  ridgeValidBlockCount: number;
  ridgeCandidateBlockCount: number;
}) {
  const minimums = getScaleAwareEvidenceMinimums(ridgeCandidateBlockCount);
  return (
    ridgePeriodicity >= FINGERPRINT_QUALITY_THRESHOLDS.biometricPeriodicity &&
    ridgeFrequencyConsistency >=
      FINGERPRINT_QUALITY_THRESHOLDS.biometricFrequencyConsistency &&
    ridgeValidBlockRatio >=
      FINGERPRINT_QUALITY_THRESHOLDS.biometricOrientationFallbackValidRatio &&
    ridgeValidBlockCount >= minimums.biometric
  );
}

// Ana ölçekte biyometrik kanıt sınıra ulaşmıyorsa ikinci orientation ölçeğinin denenmesini ister.
export function shouldTrySecondaryRidgeScale({
  ridgeValidBlockRatio,
  ridgeValidBlockCount,
  ridgeCandidateBlockCount,
}: {
  ridgeValidBlockRatio: number;
  ridgeValidBlockCount: number;
  ridgeCandidateBlockCount: number;
}) {
  const minimums = getScaleAwareEvidenceMinimums(ridgeCandidateBlockCount);
  return (
    ridgeValidBlockRatio * 100 <
      FINGERPRINT_QUALITY_THRESHOLDS.biometricOrientationFallbackValidRatio ||
    ridgeValidBlockCount < minimums.biometric
  );
}
