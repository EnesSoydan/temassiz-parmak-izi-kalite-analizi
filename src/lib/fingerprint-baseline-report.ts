import * as FileSystem from 'expo-file-system/legacy';

import { loadBiometricDatabase } from '@/lib/biometric-database';
import { ensureCaptureStorage, loadCaptureSamples } from '@/lib/capture-storage';
import {
  evaluateFingerprintEnrollmentBaseline,
  evaluateFingerprintLabeledProbes,
} from '@/lib/fingerprint-evaluation';
import {
  DEFAULT_FINGERPRINT_MATCHER_CONFIG,
  identifyPersonFromFingerRois,
  matchFingerprintTemplates,
} from '@/lib/fingerprint-matcher';

const BASELINE_REPORT_FILE = `${FileSystem.documentDirectory ?? ''}fingerprint-captures/fingerprint-baseline-report.json`;
const PROBE_REPORT_FILE = `${FileSystem.documentDirectory ?? ''}fingerprint-captures/fingerprint-probe-report.json`;

// Biyometrik şablonları dışarı çıkarmadan cihaz içinde leave-one-out baseline raporu üretir.
export async function createFingerprintEnrollmentBaselineReport() {
  const database = await loadBiometricDatabase();
  const report = await evaluateFingerprintEnrollmentBaseline({
    people: database.people,
    enrollments: database.enrollments,
    matcherConfig: DEFAULT_FINGERPRINT_MATCHER_CONFIG,
    matchTemplates: matchFingerprintTemplates,
    identifyPerson: identifyPersonFromFingerRois,
    // Uzun 1:N taramalarda JS iş parçacığını düzenli aralıklarla arayüze geri verir.
    yieldControl: () =>
      new Promise<void>((resolve) => {
        setTimeout(resolve, 0);
      }),
  });

  await ensureCaptureStorage();
  await FileSystem.writeAsStringAsync(
    BASELINE_REPORT_FILE,
    JSON.stringify(report, null, 2)
  );
  console.info(
    `[Baseline] kişi=${report.dataset.personCount}, çekim=${report.dataset.sourceCaptureCount}, şablon=${report.dataset.usableTemplateCount}, rank1=${formatRate(report.identification.rank1Rate)}, doğru_tanıma=${formatRate(report.identification.correctIdentificationRate)}, açık_küme_far=${formatRate(report.identification.openSetFalseAcceptRate)}`
  );

  return { uri: BASELINE_REPORT_FILE, report };
}

// Elle etiketlenmiş gerçek giriş çekimlerinden fotoğraf ve kimlik içermeyen saha raporu üretir.
export async function createFingerprintProbeBaselineReport() {
  const [database, samples] = await Promise.all([
    loadBiometricDatabase(),
    loadCaptureSamples(),
  ]);
  const report = await evaluateFingerprintLabeledProbes({
    samples,
    people: database.people,
    enrollments: database.enrollments,
    matcherConfig: DEFAULT_FINGERPRINT_MATCHER_CONFIG,
    identifyPerson: identifyPersonFromFingerRois,
    yieldControl: () =>
      new Promise<void>((resolve) => {
        setTimeout(resolve, 0);
      }),
  });

  await ensureCaptureStorage();
  await FileSystem.writeAsStringAsync(
    PROBE_REPORT_FILE,
    JSON.stringify(report, null, 2)
  );
  console.info(
    `[Probe baseline] etiketli=${report.dataset.labeledProbeCount}, değerlendirilen=${report.dataset.evaluatedProbeCount}, genuine=${report.identification.genuineProbeCount}, doğru_kabul=${formatRate(report.identification.trueIdentificationRate)}, impostor=${report.identification.impostorProbeCount}, far=${formatRate(report.identification.openSetFalseAcceptRate)}`
  );

  return { uri: PROBE_REPORT_FILE, report };
}

function formatRate(rate: number | null) {
  return rate === null ? 'hesaplanamadı' : `${(rate * 100).toFixed(1)}%`;
}
