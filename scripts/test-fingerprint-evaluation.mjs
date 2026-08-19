import assert from 'node:assert/strict';

import {
  DEFAULT_FINGERPRINT_MATCHER_CONFIG,
  identifyPersonFromFingerRois,
  matchFingerprintTemplates,
} from '../src/lib/fingerprint-matcher.ts';
import {
  evaluateFingerprintEnrollmentBaseline,
  evaluateFingerprintLabeledProbes,
} from '../src/lib/fingerprint-evaluation.ts';

const FINGER_POSITIONS = ['index', 'middle', 'ring', 'pinky'];
const people = ['Ali', 'Buse', 'Can'].map((displayName, index) => ({
  id: `person-${index}`,
  displayName,
  createdAt: '2026-01-01T00:00:00.000Z',
  schemaVersion: 1,
}));

function createRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function createBasePoints(personIndex, fingerIndex) {
  const random = createRandom(1000 + personIndex * 101 + fingerIndex * 17);
  return Array.from({ length: 18 }, () => ({
    x: 0.12 + random() * 0.76,
    y: 0.12 + random() * 0.76,
    angleDegrees: random() * 180,
    type: random() > 0.72 ? 'bifurcation' : 'ending',
    confidence: 82 + Math.round(random() * 14),
  }));
}

function createTemplate(personIndex, fingerIndex, sampleIndex) {
  const rotationDegrees = (sampleIndex - 1) * 2.4;
  const radians = (rotationDegrees * Math.PI) / 180;
  const cosine = Math.cos(radians);
  const sine = Math.sin(radians);
  const translationX = (sampleIndex - 1) * 0.006;
  const translationY = (1 - sampleIndex) * 0.004;
  const minutiae = createBasePoints(personIndex, fingerIndex).map((point) => {
    const centeredX = point.x - 0.5;
    const centeredY = point.y - 0.5;
    return {
      ...point,
      x: centeredX * cosine - centeredY * sine + 0.5 + translationX,
      y: centeredX * sine + centeredY * cosine + 0.5 + translationY,
      angleDegrees: point.angleDegrees + rotationDegrees,
    };
  });
  return {
    version: 'minutiae-v2',
    width: 224,
    height: 320,
    ridgePeriodPixels: 8 + (sampleIndex - 1) * 0.1,
    minutiae,
    coordinateFrame: 'homography-canonical',
  };
}

const enrollments = people.flatMap((person, personIndex) =>
  [0, 1, 2].flatMap((sampleIndex) =>
    FINGER_POSITIONS.map((fingerPosition, fingerIndex) => ({
      id: `${person.id}-${sampleIndex}-${fingerPosition}`,
      personId: person.id,
      fingerPosition,
      template: createTemplate(personIndex, fingerIndex, sampleIndex),
      qualitySnapshot: {
        overallScore: 85,
        biometricStatus: 'sufficient',
        ridgeScore: 80,
        orientationScore: 80,
      },
      sourceCaptureId: `${person.id}-capture-${sampleIndex}`,
      createdAt: '2026-01-01T00:00:00.000Z',
      sampleIndex,
      coordinateFrame: 'homography-canonical',
      templateVersion: 'minutiae-v2',
    }))
  )
);

const report = await evaluateFingerprintEnrollmentBaseline({
  people,
  enrollments,
  matcherConfig: DEFAULT_FINGERPRINT_MATCHER_CONFIG,
  matchTemplates: matchFingerprintTemplates,
  identifyPerson: identifyPersonFromFingerRois,
  createdAt: '2026-01-01T00:00:00.000Z',
});

assert.equal(report.dataset.personCount, 3);
assert.equal(report.dataset.sourceCaptureCount, 9);
assert.equal(report.dataset.completeSourceCaptureCount, 9);
assert.equal(report.dataset.usableTemplateCount, 36);
assert.equal(report.dataset.excludedTemplateCount, 0);
assert.equal(report.pairEvaluation.byFinger.index.genuine.score?.count, 18);
assert.equal(report.pairEvaluation.byFinger.index.impostor.score?.count, 54);
assert.equal(report.pairEvaluation.overall.genuine.score?.count, 72);
assert.equal(report.pairEvaluation.overall.impostor.score?.count, 216);
assert.ok(
  (report.pairEvaluation.overall.genuine.score?.mean ?? 0) >
    (report.pairEvaluation.overall.impostor.score?.mean ?? 100)
);
assert.equal(
  report.pairEvaluation.overall.genuine.evidenceTrueMatchRate,
  1
);
assert.equal(
  report.pairEvaluation.overall.impostor.evidenceFalseMatchRate,
  0
);
assert.ok(report.pairEvaluation.overall.estimatedEqualErrorPoint);
assert.equal(report.identification.evaluatedProbeCaptureCount, 9);
assert.equal(report.identification.completeProbeCaptureCount, 9);
assert.equal(report.identification.rank1Rate, 1);
assert.equal(report.identification.correctIdentificationRate, 1);
assert.equal(report.identification.openSetProbeCount, 9);
assert.equal(report.identification.openSetFalseAcceptRate, 0);
const serializedReport = JSON.stringify(report);
assert.ok(!serializedReport.includes('"personId"'));
assert.ok(!serializedReport.includes('"displayName"'));
assert.ok(!serializedReport.includes('"sourceCaptureId"'));
assert.ok(!serializedReport.includes('"template"'));

const excludedReport = await evaluateFingerprintEnrollmentBaseline({
  people,
  enrollments: [
    ...enrollments,
    {
      ...enrollments[0],
      id: 'legacy-template',
      template: { ...enrollments[0].template, version: 'minutiae-v1' },
      templateVersion: 'minutiae-v1',
    },
  ],
  matcherConfig: DEFAULT_FINGERPRINT_MATCHER_CONFIG,
  matchTemplates: matchFingerprintTemplates,
  identifyPerson: identifyPersonFromFingerRois,
});
assert.equal(excludedReport.dataset.excludedTemplateCount, 1);

const onePersonReport = await evaluateFingerprintEnrollmentBaseline({
  people: [people[0]],
  enrollments: enrollments.filter(
    (enrollment) => enrollment.personId === people[0].id
  ),
  matcherConfig: DEFAULT_FINGERPRINT_MATCHER_CONFIG,
  matchTemplates: matchFingerprintTemplates,
  identifyPerson: identifyPersonFromFingerRois,
});
assert.equal(onePersonReport.dataset.personCount, 1);
assert.equal(onePersonReport.pairEvaluation.overall.impostor.score, null);
assert.equal(onePersonReport.identification.openSetProbeCount, 0);
assert.equal(onePersonReport.identification.openSetFalseAcceptRate, null);

const emptyReport = await evaluateFingerprintEnrollmentBaseline({
  people: [],
  enrollments: [],
  matcherConfig: DEFAULT_FINGERPRINT_MATCHER_CONFIG,
  matchTemplates: matchFingerprintTemplates,
  identifyPerson: identifyPersonFromFingerRois,
});
assert.equal(emptyReport.dataset.personCount, 0);
assert.equal(emptyReport.identification.rank1Rate, null);

function createProbeFingerRois(personIndex, sampleIndex) {
  return FINGER_POSITIONS.map((fingerPosition, fingerIndex) => ({
    id: `probe-${personIndex}-${sampleIndex}-${fingerPosition}`,
    detectionId: `probe-detection-${fingerPosition}`,
    className: fingerPosition,
    confidence: 0.95,
    minutiaeTemplate: createTemplate(personIndex, fingerIndex, sampleIndex),
    coordinateFrame: 'homography-canonical',
    quality: {
      biometricStatus: 'sufficient',
    },
    crop: {
      originX: 0,
      originY: 0,
      width: 224,
      height: 320,
      normalized: { x: 0, y: 0, width: 1, height: 1 },
    },
  }));
}

function createProbeSample({
  id,
  personIndex,
  sampleIndex,
  probeEvaluation,
}) {
  return {
    id,
    createdAt: '2026-01-02T00:00:00.000Z',
    rawImageUri: `file:///${id}.jpg`,
    fingerRois: createProbeFingerRois(personIndex, sampleIndex),
    fingerLabel: 'unknown',
    sessionId: `capture-session-${id}`,
    qualityStatus: 'good',
    probeEvaluation,
    accepted: false,
  };
}

const probeSamples = [
  createProbeSample({
    id: 'genuine-probe-1',
    personIndex: 0,
    sampleIndex: 3,
    probeEvaluation: {
      version: 1,
      relation: 'genuine',
      expectedPersonId: people[0].id,
      evaluationSessionId: 'session-genuine-1',
      labeledAt: '2026-01-02T00:00:00.000Z',
    },
  }),
  createProbeSample({
    id: 'genuine-probe-2',
    personIndex: 0,
    sampleIndex: 2,
    probeEvaluation: {
      version: 1,
      relation: 'genuine',
      expectedPersonId: people[0].id,
      evaluationSessionId: 'session-genuine-1',
      labeledAt: '2026-01-02T00:01:00.000Z',
    },
  }),
  createProbeSample({
    id: 'open-set-impostor-probe',
    personIndex: 99,
    sampleIndex: 1,
    probeEvaluation: {
      version: 1,
      relation: 'impostor',
      evaluationSessionId: 'session-impostor-1',
      labeledAt: '2026-01-02T00:02:00.000Z',
    },
  }),
  createProbeSample({
    id: 'missing-person-probe',
    personIndex: 50,
    sampleIndex: 1,
    probeEvaluation: {
      version: 1,
      relation: 'genuine',
      expectedPersonId: 'deleted-person',
      evaluationSessionId: 'session-invalid-1',
      labeledAt: '2026-01-02T00:03:00.000Z',
    },
  }),
  {
    ...createProbeSample({
      id: 'no-usable-finger-probe',
      personIndex: 70,
      sampleIndex: 1,
      probeEvaluation: {
        version: 1,
        relation: 'impostor',
        evaluationSessionId: 'session-invalid-2',
        labeledAt: '2026-01-02T00:04:00.000Z',
      },
    }),
    fingerRois: [],
  },
  createProbeSample({
    id: 'unlabeled-probe',
    personIndex: 1,
    sampleIndex: 1,
  }),
];
const enrollmentsBeforeProbeEvaluation = JSON.stringify(enrollments);
const samplesBeforeProbeEvaluation = JSON.stringify(probeSamples);
const probeReport = await evaluateFingerprintLabeledProbes({
  samples: probeSamples,
  people,
  enrollments,
  matcherConfig: DEFAULT_FINGERPRINT_MATCHER_CONFIG,
  identifyPerson: identifyPersonFromFingerRois,
  createdAt: '2026-01-02T00:10:00.000Z',
});

assert.equal(probeReport.dataset.registeredPersonCount, 3);
assert.equal(probeReport.dataset.usableGalleryPersonCount, 3);
assert.equal(probeReport.evaluationVersion, 'labeled-probe-v2');
assert.equal(probeReport.dataset.labeledProbeCount, 5);
assert.equal(probeReport.dataset.genuineLabeledProbeCount, 3);
assert.equal(probeReport.dataset.impostorLabeledProbeCount, 2);
assert.equal(probeReport.dataset.evaluatedProbeCount, 3);
assert.equal(probeReport.dataset.completeProbeCount, 3);
assert.equal(probeReport.dataset.excludedProbeCount, 2);
assert.equal(probeReport.dataset.evaluatedSessionCount, 2);
assert.equal(probeReport.identification.genuineProbeCount, 2);
assert.equal(probeReport.identification.rank1Rate, 1);
assert.equal(probeReport.identification.trueIdentificationRate, 1);
assert.equal(probeReport.identification.impostorProbeCount, 1);
assert.equal(probeReport.identification.openSetFalseAcceptRate, 0);
assert.equal(probeReport.sessionEvaluation.genuineSessionCount, 1);
assert.equal(probeReport.sessionEvaluation.genuineSessionSuccessRate, 1);
assert.equal(probeReport.sessionEvaluation.impostorSessionCount, 1);
assert.equal(probeReport.sessionEvaluation.impostorSessionFalseAcceptRate, 0);
assert.equal(probeReport.pairEvaluation.overall.genuine.score?.count, 8);
assert.equal(probeReport.pairEvaluation.overall.impostor.score?.count, 4);
assert.equal(probeReport.diagnostics.probes.length, 5);
assert.deepEqual(probeReport.diagnostics.exclusionReasonCounts, {
  'expected-person-unavailable': 1,
  'no-usable-probe-fingers': 1,
});
assert.equal(
  Object.values(probeReport.diagnostics.matcherDecisionReasonCounts).reduce(
    (sum, count) => sum + count,
    0
  ),
  3
);
assert.equal(probeReport.diagnostics.probes[0].probe, 'probe-001');
assert.equal(probeReport.diagnostics.probes[0].session, 'session-001');
assert.equal(probeReport.diagnostics.probes[0].outcome, 'true-accept');
assert.equal(probeReport.diagnostics.probes[0].validProbeFingerCount, 4);
assert.equal(probeReport.diagnostics.probes[0].fingers.length, 4);
assert.equal(
  probeReport.diagnostics.probes[0].fingers[0].usableForMatching,
  true
);
assert.ok(probeReport.diagnostics.probes[0].fingers[0].match);
assert.equal(probeReport.diagnostics.probes[2].outcome, 'correct-reject');
assert.equal(
  probeReport.diagnostics.probes[3].exclusionReason,
  'expected-person-unavailable'
);
assert.equal(
  probeReport.diagnostics.probes[4].exclusionReason,
  'no-usable-probe-fingers'
);
assert.deepEqual(
  probeReport.diagnostics.probes[4].fingers.map(
    (finger) => finger.usabilityReasons
  ),
  [
    ['roi-missing'],
    ['roi-missing'],
    ['roi-missing'],
    ['roi-missing'],
  ]
);
assert.equal(JSON.stringify(enrollments), enrollmentsBeforeProbeEvaluation);
assert.equal(JSON.stringify(probeSamples), samplesBeforeProbeEvaluation);
const serializedProbeReport = JSON.stringify(probeReport);
assert.ok(!serializedProbeReport.includes('personId'));
assert.ok(!serializedProbeReport.includes('displayName'));
assert.ok(!serializedProbeReport.includes('sourceCaptureId'));
assert.ok(!serializedProbeReport.includes('"minutiae":'));
assert.ok(!serializedProbeReport.includes('rawImageUri'));
assert.ok(!serializedProbeReport.includes('genuine-probe-1'));
assert.ok(!serializedProbeReport.includes('session-genuine-1'));
assert.ok(!serializedProbeReport.includes(people[0].id));

console.info(
  `[Baseline testi] kişi=${report.dataset.personCount}, genuine=${report.pairEvaluation.overall.genuine.score?.count}, impostor=${report.pairEvaluation.overall.impostor.score?.count}, rank1=${report.identification.rank1Rate}, probe=${probeReport.dataset.evaluatedProbeCount}`
);
