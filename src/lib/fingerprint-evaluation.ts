import type { FingerprintMatcherConfig } from '@/lib/fingerprint-matcher';
import type {
  CaptureSample,
  Enrollment,
  FingerMatchFailureReason,
  FingerMatchResult,
  FingerRoi,
  FingerprintPosition,
  FingerprintTemplate,
  Person,
  PersonMatchResult,
} from '@/types/biometrics';

const FINGER_POSITIONS = ['index', 'middle', 'ring', 'pinky'] as const;

type MatchTemplates = (
  probeTemplate: FingerprintTemplate,
  enrollmentTemplate: FingerprintTemplate,
  config: FingerprintMatcherConfig
) => FingerMatchResult;

type IdentifyPerson = (input: {
  fingerRois: FingerRoi[];
  people: Person[];
  enrollments: Enrollment[];
  config: FingerprintMatcherConfig;
}) => PersonMatchResult;

type NumericSummary = {
  count: number;
  minimum: number;
  percentile10: number;
  median: number;
  percentile90: number;
  maximum: number;
  mean: number;
};

type PairObservation = {
  score: number;
  matchedMinutiae: number;
  coverage: number;
  acceptedByEvidenceGate: boolean;
  failureReasons: FingerMatchFailureReason[];
};

export type FingerprintPairEvaluation = {
  genuine: {
    score: NumericSummary | null;
    matchedMinutiae: NumericSummary | null;
    coverage: NumericSummary | null;
    evidenceAcceptedCount: number;
    evidenceTrueMatchRate: number | null;
  };
  impostor: {
    score: NumericSummary | null;
    matchedMinutiae: NumericSummary | null;
    coverage: NumericSummary | null;
    evidenceAcceptedCount: number;
    evidenceFalseMatchRate: number | null;
  };
  estimatedEqualErrorPoint: {
    scoreThreshold: number;
    falseMatchRate: number;
    falseNonMatchRate: number;
  } | null;
  failureReasonCounts: Partial<Record<FingerMatchFailureReason, number>>;
};

export type FingerprintIdentificationEvaluation = {
  evaluatedProbeCaptureCount: number;
  completeProbeCaptureCount: number;
  rank1CorrectCount: number;
  rank1Rate: number | null;
  acceptedCorrectCount: number;
  correctIdentificationRate: number | null;
  falseNonIdentificationCount: number;
  falseNonIdentificationRate: number | null;
  misidentificationCount: number;
  misidentificationRate: number | null;
  openSetProbeCount: number;
  openSetFalseAcceptCount: number;
  openSetFalseAcceptRate: number | null;
};

export type FingerprintBaselineReport = {
  schemaVersion: 1;
  evaluationVersion: 'enrollment-leave-one-out-v1';
  createdAt: string;
  templateVersion: 'minutiae-v2';
  coordinateFrame: 'homography-canonical';
  dataset: {
    personCount: number;
    sourceCaptureCount: number;
    completeSourceCaptureCount: number;
    usableTemplateCount: number;
    excludedTemplateCount: number;
    templateCountByFinger: Record<FingerprintPosition, number>;
  };
  matcher: {
    minimumUsableMinutiae: number;
    minimumMatchedMinutiae: number;
    minimumCoverage: number;
    minimumMatchedFingerCount: number;
    minimumTotalMatchedMinutiae: number;
    minimumPersonAverageCoverage: number;
  };
  pairEvaluation: {
    overall: FingerprintPairEvaluation;
    byFinger: Record<FingerprintPosition, FingerprintPairEvaluation>;
  };
  identification: FingerprintIdentificationEvaluation;
  limitations: string[];
};

export type FingerprintLabeledProbeIdentificationEvaluation = {
  genuineProbeCount: number;
  rank1CorrectCount: number;
  rank1Rate: number | null;
  acceptedCorrectCount: number;
  trueIdentificationRate: number | null;
  falseNonIdentificationCount: number;
  falseNonIdentificationRate: number | null;
  misidentificationCount: number;
  misidentificationRate: number | null;
  impostorProbeCount: number;
  openSetFalseAcceptCount: number;
  openSetFalseAcceptRate: number | null;
};

type ProbeExclusionReason =
  | 'no-usable-probe-fingers'
  | 'no-usable-gallery'
  | 'expected-person-unavailable';

type ProbeFingerUsabilityReason =
  | 'roi-missing'
  | 'biometric-insufficient'
  | 'template-missing'
  | 'template-version-mismatch'
  | 'roi-coordinate-frame-mismatch'
  | 'template-coordinate-frame-mismatch';

type ProbeDecisionOutcome =
  | 'true-accept'
  | 'false-non-identification'
  | 'misidentification'
  | 'correct-reject'
  | 'false-accept'
  | 'excluded';

type ProbeFingerDiagnostic = {
  fingerPosition: FingerprintPosition;
  roiPresent: boolean;
  usableForMatching: boolean;
  usabilityReasons: ProbeFingerUsabilityReason[];
  sourcePixelWidth: number | null;
  coordinateFrame: FingerRoi['coordinateFrame'] | null;
  templateMinutiaeCount: number | null;
  quality: {
    globalScore: number | null;
    captureStatus: 'good' | 'medium' | 'poor' | null;
    biometricStatus: 'sufficient' | 'insufficient' | null;
    biometricRejectionReasons: string[];
    foregroundCoverage: number | null;
    orientationCoherence: number | null;
    orientationReliableBlockRatio: number | null;
    ridgeValidBlockRatio: number | null;
    ridgeEnhancementSupportedAreaRatio: number | null;
    ridgeMedianPeriodPixels: number | null;
    minutiaeStatus: 'sufficient' | 'insufficient' | null;
    minutiaeRejectionReason: string | null;
    minutiaeCandidateCount: number | null;
    minutiaeBranchValidatedCandidateCount: number | null;
    minutiaeSuppressionCandidateCount: number | null;
    minutiaeEndingCandidateCount: number | null;
    minutiaeBifurcationCandidateCount: number | null;
    minutiaeSearchableAreaRatio: number | null;
  } | null;
  match: {
    status: FingerMatchResult['status'];
    score: number;
    matchedMinutiae: number;
    matchedEndingCount: number | null;
    matchedBifurcationCount: number | null;
    coverage: number;
    probeUsableMinutiae: number;
    enrollmentUsableMinutiae: number;
    failureReasons: FingerMatchFailureReason[];
    graphRelationScore: number | null;
    confidenceScore: number | null;
    distanceScore: number | null;
    textureScore: number | null;
    rotationDegrees: number | null;
    scale: number | null;
  } | null;
};

type ProbeCaptureDiagnostic = {
  probe: string;
  session: string;
  relation: 'genuine' | 'impostor';
  source: 'camera' | 'gallery';
  evaluationStatus: 'evaluated' | 'excluded';
  exclusionReason: ProbeExclusionReason | null;
  outcome: ProbeDecisionOutcome;
  matcherDecisionReason: PersonMatchResult['reason'] | null;
  accepted: boolean | null;
  rank1Correct: boolean | null;
  validProbeFingerCount: number;
  matchedFingerCount: number | null;
  totalMatchedMinutiae: number | null;
  averageCoverage: number | null;
  score: number | null;
  fingers: ProbeFingerDiagnostic[];
};

export type FingerprintLabeledProbeReport = {
  schemaVersion: 1;
  evaluationVersion: 'labeled-probe-v2';
  createdAt: string;
  templateVersion: 'minutiae-v2';
  coordinateFrame: 'homography-canonical';
  dataset: {
    registeredPersonCount: number;
    usableGalleryPersonCount: number;
    labeledProbeCount: number;
    genuineLabeledProbeCount: number;
    impostorLabeledProbeCount: number;
    evaluatedProbeCount: number;
    completeProbeCount: number;
    excludedProbeCount: number;
    evaluatedSessionCount: number;
  };
  matcher: FingerprintBaselineReport['matcher'];
  pairEvaluation: FingerprintBaselineReport['pairEvaluation'];
  identification: FingerprintLabeledProbeIdentificationEvaluation;
  sessionEvaluation: {
    genuineSessionCount: number;
    acceptedGenuineSessionCount: number;
    genuineSessionSuccessRate: number | null;
    impostorSessionCount: number;
    falseAcceptedImpostorSessionCount: number;
    impostorSessionFalseAcceptRate: number | null;
  };
  diagnostics: {
    exclusionReasonCounts: Partial<Record<ProbeExclusionReason, number>>;
    matcherDecisionReasonCounts: Partial<
      Record<PersonMatchResult['reason'], number>
    >;
    probes: ProbeCaptureDiagnostic[];
  };
  limitations: string[];
};

export async function evaluateFingerprintEnrollmentBaseline({
  people,
  enrollments,
  matcherConfig,
  matchTemplates,
  identifyPerson,
  yieldControl = () => Promise.resolve(),
  createdAt = new Date().toISOString(),
}: {
  people: Person[];
  enrollments: Enrollment[];
  matcherConfig: FingerprintMatcherConfig;
  matchTemplates: MatchTemplates;
  identifyPerson: IdentifyPerson;
  yieldControl?: () => Promise<void>;
  createdAt?: string;
}): Promise<FingerprintBaselineReport> {
  const usableEnrollments = enrollments.filter(isUsableEvaluationEnrollment);
  const usablePersonIds = new Set(
    usableEnrollments.map((enrollment) => enrollment.personId)
  );
  const usablePeople = people.filter((person) => usablePersonIds.has(person.id));
  const captureGroups = groupEnrollmentsByCapture(usableEnrollments);
  const observationsByFinger = createObservationMap();
  const overallObservations = { genuine: [], impostor: [] } as {
    genuine: PairObservation[];
    impostor: PairObservation[];
  };
  let processedPairCount = 0;

  for (const fingerPosition of FINGER_POSITIONS) {
    const fingerEnrollments = usableEnrollments.filter(
      (enrollment) => enrollment.fingerPosition === fingerPosition
    );
    for (const probe of fingerEnrollments) {
      for (const gallery of fingerEnrollments) {
        if (probe.id === gallery.id) continue;
        const result = matchTemplates(
          probe.template,
          gallery.template,
          matcherConfig
        );
        const observation = createPairObservation(result);
        const relation =
          probe.personId === gallery.personId ? 'genuine' : 'impostor';
        observationsByFinger[fingerPosition][relation].push(observation);
        overallObservations[relation].push(observation);
        processedPairCount += 1;
        if (processedPairCount % 32 === 0) await yieldControl();
      }
    }
  }

  const identification = await evaluateIdentification({
    people: usablePeople,
    enrollments: usableEnrollments,
    captureGroups,
    matcherConfig,
    identifyPerson,
    yieldControl,
  });

  return {
    schemaVersion: 1,
    evaluationVersion: 'enrollment-leave-one-out-v1',
    createdAt,
    templateVersion: 'minutiae-v2',
    coordinateFrame: 'homography-canonical',
    dataset: {
      personCount: usablePeople.length,
      sourceCaptureCount: captureGroups.length,
      completeSourceCaptureCount: captureGroups.filter(
        (group) => group.enrollments.length === FINGER_POSITIONS.length
      ).length,
      usableTemplateCount: usableEnrollments.length,
      excludedTemplateCount: enrollments.length - usableEnrollments.length,
      templateCountByFinger: Object.fromEntries(
        FINGER_POSITIONS.map((fingerPosition) => [
          fingerPosition,
          usableEnrollments.filter(
            (enrollment) => enrollment.fingerPosition === fingerPosition
          ).length,
        ])
      ) as Record<FingerprintPosition, number>,
    },
    matcher: {
      minimumUsableMinutiae: matcherConfig.minimumUsableMinutiae,
      minimumMatchedMinutiae: matcherConfig.minimumMatchedMinutiae,
      minimumCoverage: matcherConfig.minimumCoverage,
      minimumMatchedFingerCount: matcherConfig.minimumMatchedFingerCount,
      minimumTotalMatchedMinutiae: matcherConfig.minimumTotalMatchedMinutiae,
      minimumPersonAverageCoverage:
        matcherConfig.minimumPersonAverageCoverage,
    },
    pairEvaluation: {
      overall: evaluatePairObservations(overallObservations),
      byFinger: Object.fromEntries(
        FINGER_POSITIONS.map((fingerPosition) => [
          fingerPosition,
          evaluatePairObservations(observationsByFinger[fingerPosition]),
        ])
      ) as Record<FingerprintPosition, FingerprintPairEvaluation>,
    },
    identification,
    limitations: [
      'Bu rapor yalnızca kalite kapısından geçmiş enrollment çekimlerini leave-one-out yöntemiyle ölçer.',
      'Enrollment çekimleri gerçek kullanım probe dağılımını temsil etmez; saha çekimleri ayrıca etiketlenmelidir.',
      'Kişi sayısı ikiden azsa impostor ve açık-küme yanlış kabul oranı hesaplanamaz.',
      'Eşikler bu rapor tarafından otomatik olarak uygulamaya yazılmaz.',
    ],
  };
}

// Etiketli gerçek giriş çekimlerini, kayıtlı şablonları değiştirmeden kapalı/açık küme olarak ölçer.
export async function evaluateFingerprintLabeledProbes({
  samples,
  people,
  enrollments,
  matcherConfig,
  identifyPerson,
  yieldControl = () => Promise.resolve(),
  createdAt = new Date().toISOString(),
}: {
  samples: CaptureSample[];
  people: Person[];
  enrollments: Enrollment[];
  matcherConfig: FingerprintMatcherConfig;
  identifyPerson: IdentifyPerson;
  yieldControl?: () => Promise<void>;
  createdAt?: string;
}): Promise<FingerprintLabeledProbeReport> {
  const usableEnrollments = enrollments.filter(isUsableEvaluationEnrollment);
  const usablePersonIds = new Set(
    usableEnrollments.map((enrollment) => enrollment.personId)
  );
  const galleryPeople = people.filter((person) => usablePersonIds.has(person.id));
  const labeledSamples = samples.filter((sample) => sample.probeEvaluation);
  const observationsByFinger = createObservationMap();
  const overallObservations = { genuine: [], impostor: [] } as {
    genuine: PairObservation[];
    impostor: PairObservation[];
  };
  const sessionOutcomes = new Map<string, ProbeSessionOutcome>();
  const sessionAliases = new Map<string, string>();
  const probeDiagnostics: ProbeCaptureDiagnostic[] = [];
  const exclusionReasonCounts: Partial<Record<ProbeExclusionReason, number>> = {};
  const matcherDecisionReasonCounts: Partial<
    Record<PersonMatchResult['reason'], number>
  > = {};
  let evaluatedProbeCount = 0;
  let completeProbeCount = 0;
  let genuineProbeCount = 0;
  let rank1CorrectCount = 0;
  let acceptedCorrectCount = 0;
  let falseNonIdentificationCount = 0;
  let misidentificationCount = 0;
  let impostorProbeCount = 0;
  let openSetFalseAcceptCount = 0;

  for (const [probeIndex, sample] of labeledSamples.entries()) {
    const truth = sample.probeEvaluation;
    if (!truth) continue;
    const fingerRois = getUsableProbeFingerRois(sample);
    const expectedPersonExists =
      truth.relation === 'impostor' || usablePersonIds.has(truth.expectedPersonId);
    const exclusionReason = getProbeExclusionReason({
      usableFingerCount: fingerRois.length,
      usableGalleryPersonCount: galleryPeople.length,
      expectedPersonExists,
    });
    const probeAlias = `probe-${String(probeIndex + 1).padStart(3, '0')}`;
    const sessionAlias = getOrCreateAnonymousAlias(
      sessionAliases,
      truth.evaluationSessionId,
      'session'
    );

    if (exclusionReason) {
      incrementCount(exclusionReasonCounts, exclusionReason);
      probeDiagnostics.push({
        probe: probeAlias,
        session: sessionAlias,
        relation: truth.relation,
        source: getProbeSource(sample),
        evaluationStatus: 'excluded',
        exclusionReason,
        outcome: 'excluded',
        matcherDecisionReason: null,
        accepted: null,
        rank1Correct: null,
        validProbeFingerCount: fingerRois.length,
        matchedFingerCount: null,
        totalMatchedMinutiae: null,
        averageCoverage: null,
        score: null,
        fingers: createProbeFingerDiagnostics(sample, []),
      });
      continue;
    }

    const result = identifyPerson({
      fingerRois,
      people: galleryPeople,
      enrollments: usableEnrollments,
      config: matcherConfig,
    });
    const rank1PersonId = result.candidateResults[0]?.personId;
    const candidate =
      truth.relation === 'genuine'
        ? result.candidateResults.find(
            (item) => item.personId === truth.expectedPersonId
          )
        : result.candidateResults[0];
    const relation = truth.relation === 'genuine' ? 'genuine' : 'impostor';
    const rank1Correct =
      truth.relation === 'genuine'
        ? rank1PersonId === truth.expectedPersonId
        : null;
    const outcome = getProbeDecisionOutcome(truth, result);

    incrementCount(matcherDecisionReasonCounts, result.reason);
    probeDiagnostics.push({
      probe: probeAlias,
      session: sessionAlias,
      relation: truth.relation,
      source: getProbeSource(sample),
      evaluationStatus: 'evaluated',
      exclusionReason: null,
      outcome,
      matcherDecisionReason: result.reason,
      accepted: result.accepted,
      rank1Correct,
      validProbeFingerCount: fingerRois.length,
      matchedFingerCount: candidate?.matchedFingerCount ?? 0,
      totalMatchedMinutiae: candidate?.totalMatchedMinutiae ?? 0,
      averageCoverage: candidate?.averageCoverage ?? 0,
      score: candidate?.score ?? 0,
      fingers: createProbeFingerDiagnostics(
        sample,
        candidate?.fingerResults ?? []
      ),
    });

    for (const fingerResult of candidate?.fingerResults ?? []) {
      const observation = createPairObservation(fingerResult);
      observationsByFinger[fingerResult.fingerPosition][relation].push(
        observation
      );
      overallObservations[relation].push(observation);
    }

    evaluatedProbeCount += 1;
    if (fingerRois.length === FINGER_POSITIONS.length) completeProbeCount += 1;

    if (truth.relation === 'genuine') {
      const acceptedCorrect =
        result.accepted && result.personId === truth.expectedPersonId;
      const misidentified = result.accepted && !acceptedCorrect;
      genuineProbeCount += 1;
      if (rank1Correct) rank1CorrectCount += 1;
      if (acceptedCorrect) acceptedCorrectCount += 1;
      if (!acceptedCorrect) falseNonIdentificationCount += 1;
      if (misidentified) misidentificationCount += 1;
      updateProbeSessionOutcome(sessionOutcomes, {
        key: `genuine:${truth.expectedPersonId}:${truth.evaluationSessionId}`,
        relation: 'genuine',
        acceptedCorrect,
        falseAccepted: false,
      });
    } else {
      const falseAccepted = result.accepted;
      impostorProbeCount += 1;
      if (falseAccepted) openSetFalseAcceptCount += 1;
      updateProbeSessionOutcome(sessionOutcomes, {
        key: `impostor:${truth.evaluationSessionId}`,
        relation: 'impostor',
        acceptedCorrect: false,
        falseAccepted,
      });
    }
    await yieldControl();
  }

  const genuineSessions = [...sessionOutcomes.values()].filter(
    (session) => session.relation === 'genuine'
  );
  const impostorSessions = [...sessionOutcomes.values()].filter(
    (session) => session.relation === 'impostor'
  );
  const acceptedGenuineSessionCount = genuineSessions.filter(
    (session) => session.acceptedCorrect
  ).length;
  const falseAcceptedImpostorSessionCount = impostorSessions.filter(
    (session) => session.falseAccepted
  ).length;

  return {
    schemaVersion: 1,
    evaluationVersion: 'labeled-probe-v2',
    createdAt,
    templateVersion: 'minutiae-v2',
    coordinateFrame: 'homography-canonical',
    dataset: {
      registeredPersonCount: people.length,
      usableGalleryPersonCount: galleryPeople.length,
      labeledProbeCount: labeledSamples.length,
      genuineLabeledProbeCount: labeledSamples.filter(
        (sample) => sample.probeEvaluation?.relation === 'genuine'
      ).length,
      impostorLabeledProbeCount: labeledSamples.filter(
        (sample) => sample.probeEvaluation?.relation === 'impostor'
      ).length,
      evaluatedProbeCount,
      completeProbeCount,
      excludedProbeCount: labeledSamples.length - evaluatedProbeCount,
      evaluatedSessionCount: sessionOutcomes.size,
    },
    matcher: {
      minimumUsableMinutiae: matcherConfig.minimumUsableMinutiae,
      minimumMatchedMinutiae: matcherConfig.minimumMatchedMinutiae,
      minimumCoverage: matcherConfig.minimumCoverage,
      minimumMatchedFingerCount: matcherConfig.minimumMatchedFingerCount,
      minimumTotalMatchedMinutiae: matcherConfig.minimumTotalMatchedMinutiae,
      minimumPersonAverageCoverage:
        matcherConfig.minimumPersonAverageCoverage,
    },
    pairEvaluation: {
      overall: evaluatePairObservations(overallObservations),
      byFinger: Object.fromEntries(
        FINGER_POSITIONS.map((fingerPosition) => [
          fingerPosition,
          evaluatePairObservations(observationsByFinger[fingerPosition]),
        ])
      ) as Record<FingerprintPosition, FingerprintPairEvaluation>,
    },
    identification: {
      genuineProbeCount,
      rank1CorrectCount,
      rank1Rate: safeRate(rank1CorrectCount, genuineProbeCount),
      acceptedCorrectCount,
      trueIdentificationRate: safeRate(acceptedCorrectCount, genuineProbeCount),
      falseNonIdentificationCount,
      falseNonIdentificationRate: safeRate(
        falseNonIdentificationCount,
        genuineProbeCount
      ),
      misidentificationCount,
      misidentificationRate: safeRate(misidentificationCount, genuineProbeCount),
      impostorProbeCount,
      openSetFalseAcceptCount,
      openSetFalseAcceptRate: safeRate(
        openSetFalseAcceptCount,
        impostorProbeCount
      ),
    },
    sessionEvaluation: {
      genuineSessionCount: genuineSessions.length,
      acceptedGenuineSessionCount,
      genuineSessionSuccessRate: safeRate(
        acceptedGenuineSessionCount,
        genuineSessions.length
      ),
      impostorSessionCount: impostorSessions.length,
      falseAcceptedImpostorSessionCount,
      impostorSessionFalseAcceptRate: safeRate(
        falseAcceptedImpostorSessionCount,
        impostorSessions.length
      ),
    },
    diagnostics: {
      exclusionReasonCounts,
      matcherDecisionReasonCounts,
      probes: probeDiagnostics,
    },
    limitations: [
      'Probe etiketleri yalnızca ölçüm içindir; matcher kararına veya enrollment şablonlarına geri beslenmez.',
      'Aynı oturumdaki tekrarlar çekim bazında ayrıca raporlanır; oturum metriğinde genuine için en az bir doğru kabul başarı, impostor için herhangi bir kabul hata sayılır.',
      'Çekim ve oturum kimlikleri raporda sıralı anonim adlarla değiştirilir; ham görüntü yolları ve minutiae koordinatları dışarı yazılmaz.',
      'Az kişi ve az oturumla hesaplanan FAR/FRR güvenlik iddiası için yeterli değildir.',
      'Eşikler bu rapor tarafından otomatik olarak uygulamaya yazılmaz.',
    ],
  };
}

type ProbeSessionOutcome = {
  relation: 'genuine' | 'impostor';
  acceptedCorrect: boolean;
  falseAccepted: boolean;
};

function updateProbeSessionOutcome(
  sessions: Map<string, ProbeSessionOutcome>,
  outcome: ProbeSessionOutcome & { key: string }
) {
  const current = sessions.get(outcome.key);
  sessions.set(outcome.key, {
    relation: outcome.relation,
    acceptedCorrect: Boolean(current?.acceptedCorrect || outcome.acceptedCorrect),
    falseAccepted: Boolean(current?.falseAccepted || outcome.falseAccepted),
  });
}

function getProbeExclusionReason({
  usableFingerCount,
  usableGalleryPersonCount,
  expectedPersonExists,
}: {
  usableFingerCount: number;
  usableGalleryPersonCount: number;
  expectedPersonExists: boolean;
}): ProbeExclusionReason | null {
  if (usableFingerCount === 0) return 'no-usable-probe-fingers';
  if (usableGalleryPersonCount === 0) return 'no-usable-gallery';
  if (!expectedPersonExists) return 'expected-person-unavailable';
  return null;
}

function getOrCreateAnonymousAlias(
  aliases: Map<string, string>,
  sourceId: string,
  prefix: string
) {
  const existing = aliases.get(sourceId);
  if (existing) return existing;
  const alias = `${prefix}-${String(aliases.size + 1).padStart(3, '0')}`;
  aliases.set(sourceId, alias);
  return alias;
}

function incrementCount<Key extends string>(
  counts: Partial<Record<Key, number>>,
  key: Key
) {
  counts[key] = (counts[key] ?? 0) + 1;
}

function getProbeSource(sample: CaptureSample): 'camera' | 'gallery' {
  return sample.deviceModel?.includes('Galeri') ? 'gallery' : 'camera';
}

function getProbeDecisionOutcome(
  truth: NonNullable<CaptureSample['probeEvaluation']>,
  result: PersonMatchResult
): ProbeDecisionOutcome {
  if (truth.relation === 'impostor') {
    return result.accepted ? 'false-accept' : 'correct-reject';
  }
  if (result.accepted && result.personId === truth.expectedPersonId) {
    return 'true-accept';
  }
  if (result.accepted) return 'misidentification';
  return 'false-non-identification';
}

function createProbeFingerDiagnostics(
  sample: CaptureSample,
  fingerResults: FingerMatchResult[]
): ProbeFingerDiagnostic[] {
  const roiByPosition = new Map<FingerprintPosition, FingerRoi>();
  for (const fingerRoi of sample.fingerRois ?? []) {
    if (FINGER_POSITIONS.includes(fingerRoi.className as FingerprintPosition)) {
      roiByPosition.set(
        fingerRoi.className as FingerprintPosition,
        fingerRoi
      );
    }
  }
  const matchByPosition = new Map(
    fingerResults.map((result) => [result.fingerPosition, result])
  );

  return FINGER_POSITIONS.map((fingerPosition) => {
    const fingerRoi = roiByPosition.get(fingerPosition);
    const match = matchByPosition.get(fingerPosition);
    const usabilityReasons = getProbeFingerUsabilityReasons(fingerRoi);
    const quality = fingerRoi?.quality;
    return {
      fingerPosition,
      roiPresent: Boolean(fingerRoi),
      usableForMatching: usabilityReasons.length === 0,
      usabilityReasons,
      sourcePixelWidth: fingerRoi?.sourcePixelWidth ?? null,
      coordinateFrame: fingerRoi?.coordinateFrame ?? null,
      templateMinutiaeCount:
        fingerRoi?.minutiaeTemplate?.minutiae.length ?? null,
      quality: quality
        ? {
            globalScore: quality.globalScore ?? null,
            captureStatus:
              quality.captureStatus ?? quality.status ?? null,
            biometricStatus: quality.biometricStatus ?? null,
            biometricRejectionReasons:
              quality.biometricRejectionReasons ?? [],
            foregroundCoverage: quality.foregroundCoverage ?? null,
            orientationCoherence: quality.orientationCoherence ?? null,
            orientationReliableBlockRatio:
              quality.orientationReliableBlockRatio ?? null,
            ridgeValidBlockRatio: quality.ridgeValidBlockRatio ?? null,
            ridgeEnhancementSupportedAreaRatio:
              quality.ridgeEnhancementSupportedAreaRatio ?? null,
            ridgeMedianPeriodPixels:
              quality.ridgeMedianPeriodPixels ?? null,
            minutiaeStatus: quality.minutiaeStatus ?? null,
            minutiaeRejectionReason:
              quality.minutiaeRejectionReason ?? null,
            minutiaeCandidateCount:
              quality.minutiaeCandidateCount ?? null,
            minutiaeBranchValidatedCandidateCount:
              quality.minutiaeBranchValidatedCandidateCount ?? null,
            minutiaeSuppressionCandidateCount:
              quality.minutiaeSuppressionCandidateCount ?? null,
            minutiaeEndingCandidateCount:
              quality.minutiaeEndingCandidateCount ?? null,
            minutiaeBifurcationCandidateCount:
              quality.minutiaeBifurcationCandidateCount ?? null,
            minutiaeSearchableAreaRatio:
              quality.minutiaeSearchableAreaRatio ?? null,
          }
        : null,
      match: match
        ? {
            status: match.status,
            score: match.score,
            matchedMinutiae: match.matchedMinutiae,
            matchedEndingCount: match.matchedEndingCount ?? null,
            matchedBifurcationCount:
              match.matchedBifurcationCount ?? null,
            coverage: match.coverage,
            probeUsableMinutiae: match.probeUsableMinutiae,
            enrollmentUsableMinutiae: match.enrollmentUsableMinutiae,
            failureReasons: match.failureReasons,
            graphRelationScore: match.graphRelationScore ?? null,
            confidenceScore: match.confidenceScore ?? null,
            distanceScore: match.distanceScore ?? null,
            textureScore: match.textureScore ?? null,
            rotationDegrees: match.transform?.rotationDegrees ?? null,
            scale: match.transform?.scale ?? null,
          }
        : null,
    };
  });
}

function getProbeFingerUsabilityReasons(
  fingerRoi: FingerRoi | undefined
): ProbeFingerUsabilityReason[] {
  if (!fingerRoi) return ['roi-missing'];
  const reasons: ProbeFingerUsabilityReason[] = [];
  if (fingerRoi.quality?.biometricStatus !== 'sufficient') {
    reasons.push('biometric-insufficient');
  }
  if (!fingerRoi.minutiaeTemplate) {
    reasons.push('template-missing');
  } else {
    if (fingerRoi.minutiaeTemplate.version !== 'minutiae-v2') {
      reasons.push('template-version-mismatch');
    }
    if (
      fingerRoi.minutiaeTemplate.coordinateFrame !== 'homography-canonical'
    ) {
      reasons.push('template-coordinate-frame-mismatch');
    }
  }
  if (fingerRoi.coordinateFrame !== 'homography-canonical') {
    reasons.push('roi-coordinate-frame-mismatch');
  }
  return reasons;
}

function isUsableProbeFingerRoi(fingerRoi: FingerRoi) {
  return getProbeFingerUsabilityReasons(fingerRoi).length === 0;
}

function getUsableProbeFingerRois(sample: CaptureSample) {
  const bestByPosition = new Map<FingerprintPosition, FingerRoi>();
  for (const fingerRoi of sample.fingerRois ?? []) {
    if (!FINGER_POSITIONS.includes(fingerRoi.className as FingerprintPosition)) {
      continue;
    }
    if (!isUsableProbeFingerRoi(fingerRoi)) continue;
    bestByPosition.set(
      fingerRoi.className as FingerprintPosition,
      fingerRoi
    );
  }
  return FINGER_POSITIONS.flatMap((fingerPosition) => {
    const fingerRoi = bestByPosition.get(fingerPosition);
    return fingerRoi ? [fingerRoi] : [];
  });
}

function isUsableEvaluationEnrollment(enrollment: Enrollment) {
  return (
    enrollment.template.version === 'minutiae-v2' &&
    enrollment.templateVersion === 'minutiae-v2' &&
    enrollment.coordinateFrame === 'homography-canonical' &&
    enrollment.template.coordinateFrame === 'homography-canonical'
  );
}

function createObservationMap() {
  return Object.fromEntries(
    FINGER_POSITIONS.map((fingerPosition) => [
      fingerPosition,
      { genuine: [] as PairObservation[], impostor: [] as PairObservation[] },
    ])
  ) as Record<
    FingerprintPosition,
    { genuine: PairObservation[]; impostor: PairObservation[] }
  >;
}

function createPairObservation(result: FingerMatchResult): PairObservation {
  return {
    score: result.score,
    matchedMinutiae: result.matchedMinutiae,
    coverage: result.coverage,
    acceptedByEvidenceGate: result.status === 'matched',
    failureReasons: result.failureReasons,
  };
}

function evaluatePairObservations({
  genuine,
  impostor,
}: {
  genuine: PairObservation[];
  impostor: PairObservation[];
}): FingerprintPairEvaluation {
  const genuineAcceptedCount = genuine.filter(
    (observation) => observation.acceptedByEvidenceGate
  ).length;
  const impostorAcceptedCount = impostor.filter(
    (observation) => observation.acceptedByEvidenceGate
  ).length;
  const failureReasonCounts: Partial<
    Record<FingerMatchFailureReason, number>
  > = {};

  for (const observation of [...genuine, ...impostor]) {
    for (const reason of observation.failureReasons) {
      failureReasonCounts[reason] = (failureReasonCounts[reason] ?? 0) + 1;
    }
  }

  return {
    genuine: {
      score: summarizeNumbers(genuine.map((observation) => observation.score)),
      matchedMinutiae: summarizeNumbers(
        genuine.map((observation) => observation.matchedMinutiae)
      ),
      coverage: summarizeNumbers(
        genuine.map((observation) => observation.coverage)
      ),
      evidenceAcceptedCount: genuineAcceptedCount,
      evidenceTrueMatchRate: safeRate(genuineAcceptedCount, genuine.length),
    },
    impostor: {
      score: summarizeNumbers(impostor.map((observation) => observation.score)),
      matchedMinutiae: summarizeNumbers(
        impostor.map((observation) => observation.matchedMinutiae)
      ),
      coverage: summarizeNumbers(
        impostor.map((observation) => observation.coverage)
      ),
      evidenceAcceptedCount: impostorAcceptedCount,
      evidenceFalseMatchRate: safeRate(impostorAcceptedCount, impostor.length),
    },
    estimatedEqualErrorPoint: estimateEqualErrorPoint(genuine, impostor),
    failureReasonCounts,
  };
}

function estimateEqualErrorPoint(
  genuine: PairObservation[],
  impostor: PairObservation[]
) {
  if (genuine.length === 0 || impostor.length === 0) return null;
  let best:
    | {
        scoreThreshold: number;
        falseMatchRate: number;
        falseNonMatchRate: number;
        difference: number;
      }
    | undefined;

  for (let step = 0; step <= 200; step += 1) {
    const scoreThreshold = step / 2;
    const genuineAccepted = genuine.filter(
      (observation) =>
        observation.acceptedByEvidenceGate &&
        observation.score >= scoreThreshold
    ).length;
    const impostorAccepted = impostor.filter(
      (observation) =>
        observation.acceptedByEvidenceGate &&
        observation.score >= scoreThreshold
    ).length;
    const falseMatchRate = impostorAccepted / impostor.length;
    const falseNonMatchRate = 1 - genuineAccepted / genuine.length;
    const difference = Math.abs(falseMatchRate - falseNonMatchRate);

    if (!best || difference < best.difference) {
      best = {
        scoreThreshold,
        falseMatchRate,
        falseNonMatchRate,
        difference,
      };
    }
  }

  return best
    ? {
        scoreThreshold: best.scoreThreshold,
        falseMatchRate: round(best.falseMatchRate),
        falseNonMatchRate: round(best.falseNonMatchRate),
      }
    : null;
}

async function evaluateIdentification({
  people,
  enrollments,
  captureGroups,
  matcherConfig,
  identifyPerson,
  yieldControl,
}: {
  people: Person[];
  enrollments: Enrollment[];
  captureGroups: CaptureGroup[];
  matcherConfig: FingerprintMatcherConfig;
  identifyPerson: IdentifyPerson;
  yieldControl: () => Promise<void>;
}): Promise<FingerprintIdentificationEvaluation> {
  let evaluatedProbeCaptureCount = 0;
  let completeProbeCaptureCount = 0;
  let rank1CorrectCount = 0;
  let acceptedCorrectCount = 0;
  let falseNonIdentificationCount = 0;
  let misidentificationCount = 0;
  let openSetProbeCount = 0;
  let openSetFalseAcceptCount = 0;

  for (const captureGroup of captureGroups) {
    const galleryEnrollments = enrollments.filter(
      (enrollment) => enrollment.sourceCaptureId !== captureGroup.sourceCaptureId
    );
    if (
      !galleryEnrollments.some(
        (enrollment) => enrollment.personId === captureGroup.personId
      )
    ) {
      continue;
    }
    const galleryPersonIds = new Set(
      galleryEnrollments.map((enrollment) => enrollment.personId)
    );
    const galleryPeople = people.filter((person) =>
      galleryPersonIds.has(person.id)
    );
    const fingerRois = createEvaluationFingerRois(captureGroup.enrollments);
    const result = identifyPerson({
      fingerRois,
      people: galleryPeople,
      enrollments: galleryEnrollments,
      config: matcherConfig,
    });
    const rank1PersonId = result.candidateResults[0]?.personId;

    evaluatedProbeCaptureCount += 1;
    if (fingerRois.length === FINGER_POSITIONS.length) {
      completeProbeCaptureCount += 1;
    }
    if (rank1PersonId === captureGroup.personId) rank1CorrectCount += 1;
    if (result.accepted && result.personId === captureGroup.personId) {
      acceptedCorrectCount += 1;
    } else {
      falseNonIdentificationCount += 1;
      if (result.accepted && result.personId !== captureGroup.personId) {
        misidentificationCount += 1;
      }
    }
    await yieldControl();

    const impostorPeople = galleryPeople.filter(
      (person) => person.id !== captureGroup.personId
    );
    const impostorPersonIds = new Set(
      impostorPeople.map((person) => person.id)
    );
    const impostorEnrollments = galleryEnrollments.filter((enrollment) =>
      impostorPersonIds.has(enrollment.personId)
    );
    if (impostorPeople.length === 0 || impostorEnrollments.length === 0) {
      continue;
    }
    const openSetResult = identifyPerson({
      fingerRois,
      people: impostorPeople,
      enrollments: impostorEnrollments,
      config: matcherConfig,
    });
    openSetProbeCount += 1;
    if (openSetResult.accepted) openSetFalseAcceptCount += 1;
    await yieldControl();
  }

  return {
    evaluatedProbeCaptureCount,
    completeProbeCaptureCount,
    rank1CorrectCount,
    rank1Rate: safeRate(rank1CorrectCount, evaluatedProbeCaptureCount),
    acceptedCorrectCount,
    correctIdentificationRate: safeRate(
      acceptedCorrectCount,
      evaluatedProbeCaptureCount
    ),
    falseNonIdentificationCount,
    falseNonIdentificationRate: safeRate(
      falseNonIdentificationCount,
      evaluatedProbeCaptureCount
    ),
    misidentificationCount,
    misidentificationRate: safeRate(
      misidentificationCount,
      evaluatedProbeCaptureCount
    ),
    openSetProbeCount,
    openSetFalseAcceptCount,
    openSetFalseAcceptRate: safeRate(
      openSetFalseAcceptCount,
      openSetProbeCount
    ),
  };
}

type CaptureGroup = {
  personId: string;
  sourceCaptureId: string;
  enrollments: Enrollment[];
};

function groupEnrollmentsByCapture(enrollments: Enrollment[]) {
  const groups = new Map<string, CaptureGroup>();
  for (const enrollment of enrollments) {
    const key = `${enrollment.personId}:${enrollment.sourceCaptureId}`;
    const group = groups.get(key) ?? {
      personId: enrollment.personId,
      sourceCaptureId: enrollment.sourceCaptureId,
      enrollments: [],
    };
    group.enrollments.push(enrollment);
    groups.set(key, group);
  }
  return [...groups.values()];
}

function createEvaluationFingerRois(enrollments: Enrollment[]): FingerRoi[] {
  const bestByPosition = new Map<FingerprintPosition, Enrollment>();
  for (const enrollment of enrollments) {
    bestByPosition.set(enrollment.fingerPosition, enrollment);
  }

  return FINGER_POSITIONS.flatMap((fingerPosition) => {
    const enrollment = bestByPosition.get(fingerPosition);
    if (!enrollment) return [];
    return [
      {
        id: `evaluation-${enrollment.id}`,
        detectionId: `evaluation-${enrollment.id}`,
        className: fingerPosition,
        confidence: 1,
        minutiaeTemplate: enrollment.template,
        coordinateFrame: enrollment.coordinateFrame,
        quality: {
          biometricStatus: 'sufficient',
        } as FingerRoi['quality'],
        crop: {
          originX: 0,
          originY: 0,
          width: enrollment.template.width,
          height: enrollment.template.height,
          normalized: { x: 0, y: 0, width: 1, height: 1 },
        },
      },
    ];
  });
}

function summarizeNumbers(values: number[]): NumericSummary | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((first, second) => first - second);
  return {
    count: sorted.length,
    minimum: round(sorted[0]),
    percentile10: round(readPercentile(sorted, 0.1)),
    median: round(readPercentile(sorted, 0.5)),
    percentile90: round(readPercentile(sorted, 0.9)),
    maximum: round(sorted[sorted.length - 1]),
    mean: round(
      sorted.reduce((sum, value) => sum + value, 0) / sorted.length
    ),
  };
}

function readPercentile(sorted: number[], percentile: number) {
  const position = (sorted.length - 1) * percentile;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

function safeRate(numerator: number, denominator: number) {
  return denominator > 0 ? round(numerator / denominator) : null;
}

function round(value: number) {
  return Math.round(value * 10000) / 10000;
}
