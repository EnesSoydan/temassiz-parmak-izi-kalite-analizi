import assert from 'node:assert/strict';

import {
  assessPersonMatchEvidence,
  identifyPersonFromFingerRois,
  matchFingerprintTemplates,
} from '../src/lib/fingerprint-matcher.ts';

const basePoints = [
  [0.18, 0.2, 'ending'],
  [0.31, 0.18, 'bifurcation'],
  [0.46, 0.22, 'ending'],
  [0.62, 0.19, 'ending'],
  [0.78, 0.27, 'bifurcation'],
  [0.22, 0.42, 'bifurcation'],
  [0.38, 0.48, 'ending'],
  [0.55, 0.43, 'bifurcation'],
  [0.72, 0.51, 'ending'],
  [0.45, 0.72, 'ending'],
  [0.67, 0.74, 'bifurcation'],
].map(([x, y, type], index) => ({
  x,
  y,
  type,
  angleDegrees: 12 + index * 19,
  confidence: 88 + (index % 3),
}));

function makeTemplate(points = basePoints) {
  return {
    version: 'minutiae-v1',
    width: 224,
    height: 320,
    ridgePeriodPixels: 8,
    minutiae: points,
  };
}

function inverseTransformTemplate(template, rotationDegrees, translationX, translationY) {
  const radians = (-rotationDegrees * Math.PI) / 180;
  const cos = Math.cos(radians);
  const sin = Math.sin(radians);
  return makeTemplate(
    template.minutiae.map((point) => {
      const shiftedX = point.x - translationX;
      const shiftedY = point.y - translationY;
      return {
        ...point,
        x: shiftedX * cos - shiftedY * sin,
        y: shiftedX * sin + shiftedY * cos,
        angleDegrees: point.angleDegrees - rotationDegrees,
      };
    })
  );
}

const enrollmentTemplate = makeTemplate();
const identicalMatch = matchFingerprintTemplates(
  enrollmentTemplate,
  enrollmentTemplate
);
assert.equal(identicalMatch.status, 'matched');
assert.ok(identicalMatch.score >= 90);
assert.equal(identicalMatch.matchedMinutiae, basePoints.length);
assert.ok((identicalMatch.graphRelationScore ?? 0) > 0.5);

const transformedProbe = inverseTransformTemplate(
  enrollmentTemplate,
  7,
  0.035,
  -0.02
);
const transformedMatch = matchFingerprintTemplates(
  transformedProbe,
  enrollmentTemplate
);
assert.equal(transformedMatch.status, 'matched');
assert.ok(transformedMatch.matchedMinutiae >= 8);

const insufficientTemplate = makeTemplate(basePoints.slice(0, 4));
assert.equal(
  matchFingerprintTemplates(insufficientTemplate, enrollmentTemplate).status,
  'insufficient'
);

const versionMismatch = matchFingerprintTemplates(
  enrollmentTemplate,
  { ...enrollmentTemplate, version: 'minutiae-v2' }
);
assert.ok(versionMismatch.failureReasons.includes('template-version-mismatch'));

const personA = { id: 'person-a', displayName: 'Ali', createdAt: '', schemaVersion: 1 };
const personB = { id: 'person-b', displayName: 'Buse', createdAt: '', schemaVersion: 1 };
const allPositions = ['index', 'middle', 'ring', 'pinky'];
const enrollments = allPositions.map((fingerPosition) => ({
  id: `enrollment-a-${fingerPosition}`,
  personId: personA.id,
  fingerPosition,
  template: enrollmentTemplate,
  qualitySnapshot: {
    overallScore: 80,
    biometricStatus: 'sufficient',
    ridgeScore: 80,
    orientationScore: 80,
  },
  sourceCaptureId: 'capture-a',
  createdAt: '',
  templateVersion: 'minutiae-v1',
}));

const probeRoi = {
  id: 'probe-index',
  detectionId: 'detection-index',
  className: 'index',
  confidence: 0.95,
  minutiaeTemplate: transformedProbe,
  quality: { biometricStatus: 'sufficient' },
};
const singleFingerIdentification = identifyPersonFromFingerRois({
  fingerRois: [probeRoi],
  people: [personA, personB],
  enrollments,
});
assert.equal(singleFingerIdentification.accepted, false);
assert.equal(
  singleFingerIdentification.reason,
  'insufficient-multi-finger-evidence'
);
assert.equal(singleFingerIdentification.matchedFingerCount, 1);

const probeRois = allPositions.map((fingerPosition) => ({
  ...probeRoi,
  id: `probe-${fingerPosition}`,
  detectionId: `detection-${fingerPosition}`,
  className: fingerPosition,
}));
const identification = identifyPersonFromFingerRois({
  fingerRois: probeRois,
  people: [personA, personB],
  enrollments,
});
assert.equal(identification.accepted, true);
assert.equal(identification.personId, personA.id);
assert.equal(identification.matchedFingerCount, 4);
assert.equal(identification.totalMatchedMinutiae, basePoints.length * 4);

const threeSampleIdentification = identifyPersonFromFingerRois({
  fingerRois: probeRois,
  people: [personA],
  enrollments: enrollments.flatMap((enrollment) =>
    [0, 1, 2].map((sampleIndex) => ({ ...enrollment, sampleIndex }))
  ),
});
assert.equal(threeSampleIdentification.accepted, true);

function makeEvidenceResult(fingerPosition, status, matchedMinutiae, coverage) {
  return {
    fingerPosition,
    status,
    score: 70,
    matchedMinutiae,
    coverage,
    probeUsableMinutiae: 40,
    enrollmentUsableMinutiae: 40,
    minimumUsableMinutiae: 8,
    minimumMatchedMinutiae: 8,
    minimumCoverage: 0.25,
    failureReasons: [],
  };
}

// Sahadaki doğru/el-dışı log çifti için güvenlik regresyonu.
const genuineEvidence = assessPersonMatchEvidence([
  makeEvidenceResult('index', 'matched', 8, 0.3),
  makeEvidenceResult('middle', 'matched', 8, 0.3),
  makeEvidenceResult('ring', 'matched', 15, 0.35),
  makeEvidenceResult('pinky', 'matched', 21, 0.41),
]);
assert.equal(genuineEvidence.accepted, true);
assert.equal(genuineEvidence.totalMatchedMinutiae, 52);

const differentHandEvidence = assessPersonMatchEvidence([
  makeEvidenceResult('index', 'no-match', 5, 0.19),
  makeEvidenceResult('middle', 'no-match', 4, 0.15),
  makeEvidenceResult('ring', 'no-match', 8, 0.24),
  makeEvidenceResult('pinky', 'matched', 17, 0.35),
]);
assert.equal(differentHandEvidence.accepted, false);
assert.equal(differentHandEvidence.matchedFingerCount, 1);

const lowTotalEvidence = assessPersonMatchEvidence([
  makeEvidenceResult('index', 'matched', 8, 0.3),
  makeEvidenceResult('middle', 'matched', 8, 0.3),
  makeEvidenceResult('ring', 'matched', 8, 0.3),
]);
assert.equal(lowTotalEvidence.accepted, false);

const wrongPosition = identifyPersonFromFingerRois({
  fingerRois: [{ ...probeRoi, className: 'middle' }],
  people: [personA],
  enrollments: enrollments.filter((item) => item.fingerPosition === 'index'),
});
assert.equal(wrongPosition.accepted, false);

console.info(
  `[Matcher testi] aynı=${identicalMatch.score}, dönüşüm=${transformedMatch.score}, kişi=${identification.displayName}, farklı_el=${differentHandEvidence.accepted}`
);
