import type {
  Enrollment,
  FingerMatchResult,
  FingerMatchFailureReason,
  FingerRoi,
  FingerprintMinutia,
  FingerprintPosition,
  FingerprintTemplate,
  Person,
  PersonCandidateResult,
  PersonMatchResult,
} from '@/types/biometrics';

const FINGERPRINT_POSITIONS = ['index', 'middle', 'ring', 'pinky'] as const;

type MatchPoint = FingerprintMinutia & { index: number };

export type FingerprintMatcherConfig = {
  minimumUsableMinutiae: number;
  maximumAnchorCount: number;
  minimumMatchedMinutiae: number;
  distanceTolerance: number;
  angleToleranceDegrees: number;
  minimumCoverage: number;
  minimumMatchedFingerCount: number;
  minimumTotalMatchedMinutiae: number;
  minimumPersonAverageCoverage: number;
  ambiguityMargin: number;
  minimumConfidence: number;
  maximumRansacIterations: number;
  minimumTransformScale: number;
  maximumTransformScale: number;
  maximumTransformRotationDegrees: number;
};

export const DEFAULT_FINGERPRINT_MATCHER_CONFIG: FingerprintMatcherConfig = {
  minimumUsableMinutiae: 8,
  maximumAnchorCount: 18,
  minimumMatchedMinutiae: 8,
  distanceTolerance: 0.055,
  angleToleranceDegrees: 34,
  minimumCoverage: 0.25,
  minimumMatchedFingerCount: 3,
  minimumTotalMatchedMinutiae: 40,
  minimumPersonAverageCoverage: 0.28,
  ambiguityMargin: 7,
  minimumConfidence: 45,
  maximumRansacIterations: 128,
  minimumTransformScale: 0.7,
  maximumTransformScale: 1.4,
  // Canlı kapı en fazla 15° el sapmasına izin verir; kanonik hizalama sonrası 20° üzeri hipotez fiziksel kabul edilmez.
  maximumTransformRotationDegrees: 20,
};

type Transform = {
  rotationDegrees: number;
  scale: number;
  translationX: number;
  translationY: number;
};

type TemplateMatch = {
  score: number;
  matchedMinutiae: number;
  matchedEndingCount: number;
  matchedBifurcationCount: number;
  coverage: number;
  transform: Transform;
  graphRelationScore: number;
  confidenceScore: number;
  distanceScore: number;
  textureScore: number | null;
};

type GraphEdge = {
  fromIndex: number;
  toIndex: number;
  distance: number;
  angleDegrees: number;
};

function getFailureReasons({
  probeUsableMinutiae,
  enrollmentUsableMinutiae,
  matchedMinutiae,
  coverage,
  config,
}: {
  probeUsableMinutiae: number;
  enrollmentUsableMinutiae: number;
  matchedMinutiae: number;
  coverage: number;
  config: FingerprintMatcherConfig;
}): FingerMatchFailureReason[] {
  const reasons: FingerMatchFailureReason[] = [];
  if (probeUsableMinutiae < config.minimumUsableMinutiae) {
    reasons.push('not-enough-probe-minutiae');
  }
  if (enrollmentUsableMinutiae < config.minimumUsableMinutiae) {
    reasons.push('not-enough-enrollment-minutiae');
  }
  if (matchedMinutiae < config.minimumMatchedMinutiae) {
    reasons.push('matched-minutiae-below-threshold');
  }
  if (coverage < config.minimumCoverage) {
    reasons.push('coverage-below-threshold');
  }
  return reasons;
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

function cosineSimilarity(first: number[], second: number[]) {
  if (first.length === 0 || first.length !== second.length) return null;
  let dot = 0;
  let firstNorm = 0;
  let secondNorm = 0;
  for (let index = 0; index < first.length; index += 1) {
    dot += first[index] * second[index];
    firstNorm += first[index] ** 2;
    secondNorm += second[index] ** 2;
  }
  if (firstNorm <= 1e-8 || secondNorm <= 1e-8) return null;
  return clamp(
    (dot / Math.sqrt(firstNorm * secondNorm) + 1) / 2,
    0,
    1
  );
}

function normalizeAngle(angle: number) {
  let normalized = angle % 360;
  if (normalized > 180) normalized -= 360;
  if (normalized < -180) normalized += 360;
  return normalized;
}

function angleDifference(first: number, second: number) {
  return Math.abs(normalizeAngle(first - second));
}

function distance(first: MatchPoint, second: MatchPoint) {
  return Math.hypot(first.x - second.x, first.y - second.y);
}

function filterPoints(
  template: FingerprintTemplate,
  config: FingerprintMatcherConfig
) {
  return template.minutiae
    .map((point, index) => ({ ...point, index }))
    .filter((point) => point.confidence >= config.minimumConfidence)
    .sort((first, second) => second.confidence - first.confidence);
}

function createTransform(
  probeAnchor: MatchPoint,
  enrollmentAnchor: MatchPoint,
  probeTemplate: FingerprintTemplate,
  enrollmentTemplate: FingerprintTemplate
): Transform {
  const rotationDegrees = normalizeAngle(
    enrollmentAnchor.angleDegrees - probeAnchor.angleDegrees
  );
  const periodRatio =
    enrollmentTemplate.ridgePeriodPixels /
    Math.max(probeTemplate.ridgePeriodPixels, 0.1);
  const scale = clamp(periodRatio, 0.86, 1.16);
  const radians = (rotationDegrees * Math.PI) / 180;
  const cos = Math.cos(radians);
  const sin = Math.sin(radians);
  const rotatedX =
    (probeAnchor.x * cos - probeAnchor.y * sin) * scale;
  const rotatedY =
    (probeAnchor.x * sin + probeAnchor.y * cos) * scale;

  return {
    rotationDegrees,
    scale,
    translationX: enrollmentAnchor.x - rotatedX,
    translationY: enrollmentAnchor.y - rotatedY,
  };
}

function transformPoint(point: MatchPoint, transform: Transform): MatchPoint {
  const radians = (transform.rotationDegrees * Math.PI) / 180;
  const cos = Math.cos(radians);
  const sin = Math.sin(radians);
  return {
    ...point,
    x:
      (point.x * cos - point.y * sin) * transform.scale +
      transform.translationX,
    y:
      (point.x * sin + point.y * cos) * transform.scale +
      transform.translationY,
    angleDegrees: point.angleDegrees + transform.rotationDegrees,
  };
}

function getSpatialCell(point: MatchPoint) {
  return `${Math.floor(clamp(point.x, 0, 0.999) * 4)}:${Math.floor(
    clamp(point.y, 0, 0.999) * 4
  )}`;
}

function calculateCoverage(
  matchedProbePoints: MatchPoint[],
  matchedEnrollmentPoints: MatchPoint[],
  probeCount: number,
  enrollmentCount: number
) {
  const probeCells = new Set(matchedProbePoints.map(getSpatialCell));
  const enrollmentCells = new Set(matchedEnrollmentPoints.map(getSpatialCell));
  const cellCoverage = Math.min(probeCells.size, enrollmentCells.size) / 16;
  const pointCoverage =
    matchedProbePoints.length / Math.max(Math.min(probeCount, enrollmentCount), 1);
  return clamp(pointCoverage * 0.65 + cellCoverage * 0.35, 0, 1);
}

function getNeighbourDistances(point: MatchPoint, points: MatchPoint[]) {
  return points
    .filter((candidate) => candidate.index !== point.index)
    .map((candidate) => distance(point, candidate))
    .sort((first, second) => first - second)
    .slice(0, 4);
}

function createLocalGraph(points: MatchPoint[]): GraphEdge[] {
  const edges: GraphEdge[] = [];
  for (const point of points) {
    const neighbours = points
      .filter((candidate) => candidate.index !== point.index)
      .map((candidate) => ({
        candidate,
        distance: distance(point, candidate),
      }))
      .sort((first, second) => first.distance - second.distance)
      .slice(0, 4);
    for (const neighbour of neighbours) {
      edges.push({
        fromIndex: point.index,
        toIndex: neighbour.candidate.index,
        distance: neighbour.distance,
        angleDegrees: normalizeAngle(
          (Math.atan2(
            neighbour.candidate.y - point.y,
            neighbour.candidate.x - point.x
          ) *
            180) /
            Math.PI
        ),
      });
    }
  }
  return edges;
}

function calculateGraphRelationScore(
  matchedProbePoints: MatchPoint[],
  matchedEnrollmentPoints: MatchPoint[],
  transform: Transform,
  probePoints: MatchPoint[],
  enrollmentPoints: MatchPoint[]
) {
  const enrollmentByProbeIndex = new Map<number, MatchPoint>();
  matchedProbePoints.forEach((probePoint, index) => {
    enrollmentByProbeIndex.set(probePoint.index, matchedEnrollmentPoints[index]);
  });
  const enrollmentGraph = createLocalGraph(enrollmentPoints);
  const enrollmentEdges = new Map<string, GraphEdge>();
  for (const edge of enrollmentGraph) {
    enrollmentEdges.set(`${edge.fromIndex}:${edge.toIndex}`, edge);
  }

  const relationScores: number[] = [];
  for (const probeEdge of createLocalGraph(probePoints)) {
    const enrollmentFrom = enrollmentByProbeIndex.get(probeEdge.fromIndex);
    const enrollmentTo = enrollmentByProbeIndex.get(probeEdge.toIndex);
    if (!enrollmentFrom || !enrollmentTo) continue;
    const enrollmentEdge = enrollmentEdges.get(
      `${enrollmentFrom.index}:${enrollmentTo.index}`
    );
    if (!enrollmentEdge) continue;
    const expectedDistance = probeEdge.distance * transform.scale;
    const distanceScore = clamp(
      1 - Math.abs(Math.log((enrollmentEdge.distance + 0.001) / (expectedDistance + 0.001))) / 0.55,
      0,
      1
    );
    const transformedFrom = transformPoint(
      matchedProbePoints.find((point) => point.index === probeEdge.fromIndex)!,
      transform
    );
    const transformedTo = transformPoint(
      matchedProbePoints.find((point) => point.index === probeEdge.toIndex)!,
      transform
    );
    const transformedAngle = normalizeAngle(
      (Math.atan2(
        transformedTo.y - transformedFrom.y,
        transformedTo.x - transformedFrom.x
      ) *
        180) /
        Math.PI
    );
    const angleScore = clamp(
      1 - angleDifference(transformedAngle, enrollmentEdge.angleDegrees) / 90,
      0,
      1
    );
    relationScores.push(distanceScore * 0.65 + angleScore * 0.35);
  }
  if (relationScores.length === 0) return 0;
  return relationScores.reduce((sum, score) => sum + score, 0) / relationScores.length;
}

// Anchor adayının yakın çevresinin benzerliğini kontrol ederek tekil gürültü noktalarını eler.
function hasCompatibleNeighbourhood(
  probePoint: MatchPoint,
  enrollmentPoint: MatchPoint,
  probePoints: MatchPoint[],
  enrollmentPoints: MatchPoint[],
  scale: number
) {
  const probeNeighbours = getNeighbourDistances(probePoint, probePoints);
  const enrollmentNeighbours = getNeighbourDistances(
    enrollmentPoint,
    enrollmentPoints
  );
  const comparisonCount = Math.min(probeNeighbours.length, enrollmentNeighbours.length);
  if (comparisonCount < 2) return true;

  let relativeError = 0;
  for (let index = 0; index < comparisonCount; index += 1) {
    const expectedDistance = probeNeighbours[index] * scale;
    relativeError += Math.abs(
      Math.log((enrollmentNeighbours[index] + 0.001) / (expectedDistance + 0.001))
    );
  }
  return relativeError / comparisonCount <= 0.7;
}

function tryTransform(
  probePoints: MatchPoint[],
  enrollmentPoints: MatchPoint[],
  transform: Transform,
  config: FingerprintMatcherConfig,
  probeTextureDescriptor?: number[],
  enrollmentTextureDescriptor?: number[]
): TemplateMatch {
  const usedEnrollmentIndexes = new Set<number>();
  const matchedProbePoints: MatchPoint[] = [];
  const matchedEnrollmentPoints: MatchPoint[] = [];
  let weightedConfidence = 0;

  for (const probePoint of probePoints) {
    const transformed = transformPoint(probePoint, transform);
    let bestCandidate: MatchPoint | undefined;
    let bestDistance = Number.POSITIVE_INFINITY;

    for (const enrollmentPoint of enrollmentPoints) {
      if (usedEnrollmentIndexes.has(enrollmentPoint.index)) continue;
      if (enrollmentPoint.type !== probePoint.type) continue;
      const candidateDistance = distance(transformed, enrollmentPoint);
      if (candidateDistance > config.distanceTolerance) continue;
      if (
        angleDifference(transformed.angleDegrees, enrollmentPoint.angleDegrees) >
        config.angleToleranceDegrees
      ) {
        continue;
      }
      if (candidateDistance < bestDistance) {
        bestDistance = candidateDistance;
        bestCandidate = enrollmentPoint;
      }
    }

    if (!bestCandidate) continue;
    usedEnrollmentIndexes.add(bestCandidate.index);
    matchedProbePoints.push(probePoint);
    matchedEnrollmentPoints.push(bestCandidate);
    weightedConfidence +=
      (probePoint.confidence + bestCandidate.confidence) / 200;
  }

  const matchedMinutiae = matchedProbePoints.length;
  const matchedEndingCount = matchedProbePoints.filter(
    (point) => point.type === 'ending'
  ).length;
  const matchedBifurcationCount =
    matchedMinutiae - matchedEndingCount;
  const coverage = calculateCoverage(
    matchedProbePoints,
    matchedEnrollmentPoints,
    probePoints.length,
    enrollmentPoints.length
  );
  const countRatio =
    matchedMinutiae / Math.max(Math.min(probePoints.length, enrollmentPoints.length), 1);
  const confidenceRatio = weightedConfidence / Math.max(matchedMinutiae, 1);
  const distanceQuality = matchedProbePoints.length
    ? matchedProbePoints.reduce((total, probePoint, index) => {
        const enrollmentPoint = matchedEnrollmentPoints[index];
        const transformed = transformPoint(probePoint, transform);
        const normalizedDistance =
          distance(transformed, enrollmentPoint) / config.distanceTolerance;
        return total + clamp(1 - normalizedDistance, 0, 1);
      }, 0) / matchedProbePoints.length
    : 0;
  const graphRelationScore = calculateGraphRelationScore(
    matchedProbePoints,
    matchedEnrollmentPoints,
    transform,
    probePoints,
    enrollmentPoints
  );
  // Graph ve texture değerleri henüz gerçek/impostor veriyle kalibre edilmediği için
  // kabul skoruna katılmaz; yalnızca terminalde tanı amaçlı raporlanır.
  const score = clamp(
    100 *
      (countRatio * 0.38 +
        confidenceRatio * 0.18 +
        coverage * 0.24 +
        distanceQuality * 0.2),
    0,
    100
  );
  const textureScore = cosineSimilarity(
    probeTextureDescriptor ?? [],
    enrollmentTextureDescriptor ?? []
  );
  return {
    score,
    matchedMinutiae,
    matchedEndingCount,
    matchedBifurcationCount,
    coverage,
    transform,
    graphRelationScore,
    confidenceScore: confidenceRatio,
    distanceScore: distanceQuality,
    textureScore,
  };
}

function createPairTransform(
  probeFirst: MatchPoint,
  probeSecond: MatchPoint,
  enrollmentFirst: MatchPoint,
  enrollmentSecond: MatchPoint,
  config: FingerprintMatcherConfig
): Transform | null {
  const probeDeltaX = probeSecond.x - probeFirst.x;
  const probeDeltaY = probeSecond.y - probeFirst.y;
  const enrollmentDeltaX = enrollmentSecond.x - enrollmentFirst.x;
  const enrollmentDeltaY = enrollmentSecond.y - enrollmentFirst.y;
  const probeDistance = Math.hypot(probeDeltaX, probeDeltaY);
  const enrollmentDistance = Math.hypot(enrollmentDeltaX, enrollmentDeltaY);
  if (probeDistance < 0.02 || enrollmentDistance < 0.02) return null;

  const scale = enrollmentDistance / probeDistance;
  if (
    scale < config.minimumTransformScale ||
    scale > config.maximumTransformScale
  ) {
    return null;
  }
  const rotationDegrees = normalizeAngle(
    ((Math.atan2(enrollmentDeltaY, enrollmentDeltaX) -
      Math.atan2(probeDeltaY, probeDeltaX)) *
      180) /
      Math.PI
  );
  if (
    Math.abs(rotationDegrees) > config.maximumTransformRotationDegrees
  ) {
    return null;
  }
  const radians = (rotationDegrees * Math.PI) / 180;
  const rotatedX =
    (probeFirst.x * Math.cos(radians) - probeFirst.y * Math.sin(radians)) * scale;
  const rotatedY =
    (probeFirst.x * Math.sin(radians) + probeFirst.y * Math.cos(radians)) * scale;

  return {
    rotationDegrees,
    scale,
    translationX: enrollmentFirst.x - rotatedX,
    translationY: enrollmentFirst.y - rotatedY,
  };
}

function getAnchorPairs(
  probePoints: MatchPoint[],
  enrollmentPoints: MatchPoint[],
  config: FingerprintMatcherConfig,
  scale: number
) {
  const probeAnchors = probePoints.slice(0, config.maximumAnchorCount);
  const enrollmentAnchors = enrollmentPoints.slice(0, config.maximumAnchorCount);
  const pairs: { probe: MatchPoint; enrollment: MatchPoint }[] = [];

  for (const probe of probeAnchors) {
    for (const enrollment of enrollmentAnchors) {
      if (
        probe.type === enrollment.type &&
        hasCompatibleNeighbourhood(
          probe,
          enrollment,
          probePoints,
          enrollmentPoints,
          scale
        )
      ) {
        pairs.push({ probe, enrollment });
      }
    }
  }

  // Çok düşük minutiae sayısında komşuluk filtresi tüm tohumları elememelidir.
  if (pairs.length === 0) {
    return probeAnchors.flatMap((probe) =>
      enrollmentAnchors
        .filter((enrollment) => probe.type === enrollment.type)
        .map((enrollment) => ({ probe, enrollment }))
    );
  }
  return pairs;
}

function getRansacTransforms(
  probePoints: MatchPoint[],
  enrollmentPoints: MatchPoint[],
  probeTemplate: FingerprintTemplate,
  enrollmentTemplate: FingerprintTemplate,
  config: FingerprintMatcherConfig,
  scale: number
) {
  const anchorPairs = getAnchorPairs(
    probePoints,
    enrollmentPoints,
    config,
    scale
  );
  const transforms: Transform[] = [];
  const seen = new Set<string>();
  const addTransform = (transform: Transform | null) => {
    if (!transform) return;
    if (
      Math.abs(normalizeAngle(transform.rotationDegrees)) >
      config.maximumTransformRotationDegrees
    ) {
      return;
    }
    const key = [
      transform.rotationDegrees.toFixed(3),
      transform.scale.toFixed(4),
      transform.translationX.toFixed(4),
      transform.translationY.toFixed(4),
    ].join(':');
    if (seen.has(key)) return;
    seen.add(key);
    transforms.push(transform);
  };

  // İki correspondence aynı anda doğru yerel ölçek ve dönmeyi belirler.
  for (let first = 0; first < anchorPairs.length; first += 1) {
    for (
      let second = first + 1;
      second < anchorPairs.length;
      second += 1
    ) {
      const firstPair = anchorPairs[first];
      const secondPair = anchorPairs[second];
      if (
        firstPair.probe.index === secondPair.probe.index ||
        firstPair.enrollment.index === secondPair.enrollment.index
      ) {
        continue;
      }
      const probePairDistance = distance(firstPair.probe, secondPair.probe);
      const enrollmentPairDistance = distance(
        firstPair.enrollment,
        secondPair.enrollment
      );
      const pairScale = enrollmentPairDistance / Math.max(probePairDistance, 0.001);
      if (Math.abs(Math.log(pairScale / Math.max(scale, 0.001))) > 0.65) {
        continue;
      }
      addTransform(
        createPairTransform(
          firstPair.probe,
          secondPair.probe,
          firstPair.enrollment,
          secondPair.enrollment,
          config
        )
      );
      if (transforms.length >= config.maximumRansacIterations) return transforms;
    }
  }

  // İki noktalı hipotez oluşmazsa mevcut tek-anchor yolu yalnızca güvenli fallback olur.
  for (const pair of anchorPairs.slice(0, config.maximumAnchorCount)) {
    addTransform(
      createTransform(
        pair.probe,
        pair.enrollment,
        probeTemplate,
        enrollmentTemplate
      )
    );
    if (transforms.length >= config.maximumRansacIterations) break;
  }
  return transforms;
}

export function matchFingerprintTemplates(
  probeTemplate: FingerprintTemplate,
  enrollmentTemplate: FingerprintTemplate,
  config: FingerprintMatcherConfig = DEFAULT_FINGERPRINT_MATCHER_CONFIG
): FingerMatchResult {
  if (probeTemplate.version !== enrollmentTemplate.version) {
    return {
      fingerPosition: 'index',
      status: 'insufficient',
      score: 0,
      matchedMinutiae: 0,
      coverage: 0,
      probeUsableMinutiae: 0,
      enrollmentUsableMinutiae: 0,
      minimumUsableMinutiae: config.minimumUsableMinutiae,
      minimumMatchedMinutiae: config.minimumMatchedMinutiae,
      minimumCoverage: config.minimumCoverage,
      failureReasons: ['template-version-mismatch'],
    };
  }
  const probePoints = filterPoints(probeTemplate, config);
  const enrollmentPoints = filterPoints(enrollmentTemplate, config);

  if (
    probePoints.length < config.minimumUsableMinutiae ||
    enrollmentPoints.length < config.minimumUsableMinutiae
  ) {
    const failureReasons = getFailureReasons({
      probeUsableMinutiae: probePoints.length,
      enrollmentUsableMinutiae: enrollmentPoints.length,
      matchedMinutiae: 0,
      coverage: 0,
      config,
    });
    return {
      fingerPosition: 'index',
      status: 'insufficient',
      score: 0,
      matchedMinutiae: 0,
      coverage: 0,
      probeUsableMinutiae: probePoints.length,
      enrollmentUsableMinutiae: enrollmentPoints.length,
      minimumUsableMinutiae: config.minimumUsableMinutiae,
      minimumMatchedMinutiae: config.minimumMatchedMinutiae,
      minimumCoverage: config.minimumCoverage,
      failureReasons,
    };
  }

  let bestMatch: TemplateMatch | undefined;
  const scale = clamp(
    enrollmentTemplate.ridgePeriodPixels /
      Math.max(probeTemplate.ridgePeriodPixels, 0.1),
    0.86,
    1.16
  );
  for (const transform of getRansacTransforms(
    probePoints,
    enrollmentPoints,
    probeTemplate,
    enrollmentTemplate,
    config,
    scale
  )) {
    const candidate = tryTransform(
      probePoints,
      enrollmentPoints,
      transform,
      config,
      probeTemplate.textureDescriptor,
      enrollmentTemplate.textureDescriptor
    );
    if (!bestMatch || candidate.score > bestMatch.score) {
      bestMatch = candidate;
    }
  }

  const match = bestMatch ?? {
    score: 0,
    matchedMinutiae: 0,
    matchedEndingCount: 0,
    matchedBifurcationCount: 0,
    coverage: 0,
    transform: {
      rotationDegrees: 0,
      scale: 1,
      translationX: 0,
      translationY: 0,
    },
    graphRelationScore: 0,
    confidenceScore: 0,
    distanceScore: 0,
    textureScore: null,
  };
  const status: FingerMatchResult['status'] =
    match.matchedMinutiae >= config.minimumMatchedMinutiae &&
    match.coverage >= config.minimumCoverage
      ? 'matched'
      : 'no-match';

  return {
    fingerPosition: 'index',
    status,
    score: Math.round(match.score * 10) / 10,
    matchedMinutiae: match.matchedMinutiae,
    matchedEndingCount: match.matchedEndingCount,
    matchedBifurcationCount: match.matchedBifurcationCount,
    coverage: Math.round(match.coverage * 1000) / 1000,
    probeUsableMinutiae: probePoints.length,
    enrollmentUsableMinutiae: enrollmentPoints.length,
    minimumUsableMinutiae: config.minimumUsableMinutiae,
    minimumMatchedMinutiae: config.minimumMatchedMinutiae,
    minimumCoverage: config.minimumCoverage,
    failureReasons: getFailureReasons({
      probeUsableMinutiae: probePoints.length,
      enrollmentUsableMinutiae: enrollmentPoints.length,
      matchedMinutiae: match.matchedMinutiae,
      coverage: match.coverage,
      config,
    }),
    transform: match.transform,
    graphRelationScore: Math.round(match.graphRelationScore * 1000) / 1000,
    confidenceScore: Math.round(match.confidenceScore * 1000) / 1000,
    distanceScore: Math.round(match.distanceScore * 1000) / 1000,
    textureScore:
      match.textureScore === null
        ? undefined
        : Math.round(match.textureScore * 1000) / 1000,
  };
}

function withFingerPosition(
  result: FingerMatchResult,
  fingerPosition: FingerprintPosition
): FingerMatchResult {
  return { ...result, fingerPosition };
}

function createMissingFingerResult(
  fingerPosition: FingerprintPosition,
  status: 'missing' | 'insufficient',
  config: FingerprintMatcherConfig,
  failureReason: 'probe-missing' | 'enrollment-missing'
): FingerMatchResult {
  return {
    fingerPosition,
    status,
    score: 0,
    matchedMinutiae: 0,
    coverage: 0,
    probeUsableMinutiae: 0,
    enrollmentUsableMinutiae: 0,
    minimumUsableMinutiae: config.minimumUsableMinutiae,
    minimumMatchedMinutiae: config.minimumMatchedMinutiae,
    minimumCoverage: config.minimumCoverage,
    failureReasons: [failureReason],
  };
}

function createInsufficientProbeResult(
  fingerPosition: FingerprintPosition,
  probe: FingerRoi,
  config: FingerprintMatcherConfig
): FingerMatchResult {
  const probeUsableMinutiae = probe.minutiaeTemplate
    ? filterPoints(probe.minutiaeTemplate, config).length
    : 0;
  const failureReasons: FingerMatchFailureReason[] = [
    'probe-quality-insufficient',
  ];
  if (probeUsableMinutiae < config.minimumUsableMinutiae) {
    failureReasons.push('not-enough-probe-minutiae');
  }

  return {
    fingerPosition,
    status: 'insufficient',
    score: 0,
    matchedMinutiae: 0,
    coverage: 0,
    probeUsableMinutiae,
    enrollmentUsableMinutiae: 0,
    minimumUsableMinutiae: config.minimumUsableMinutiae,
    minimumMatchedMinutiae: config.minimumMatchedMinutiae,
    minimumCoverage: config.minimumCoverage,
    failureReasons,
  };
}

function aggregateEnrollmentMatches(
  matches: FingerMatchResult[],
  fingerPosition: FingerprintPosition,
  config: FingerprintMatcherConfig
) {
  const selected = [...matches].sort((first, second) => second.score - first.score).slice(0, 2);
  if (selected.length === 0) {
    return createMissingFingerResult(
      fingerPosition,
      'insufficient',
      config,
      'enrollment-missing'
    );
  }
  const score = selected.reduce((sum, result) => sum + result.score, 0) / selected.length;
  const matchedMinutiae = Math.round(
    selected.reduce((sum, result) => sum + result.matchedMinutiae, 0) / selected.length
  );
  const matchedEndingCount = Math.round(
    selected.reduce((sum, result) => sum + (result.matchedEndingCount ?? 0), 0) /
      selected.length
  );
  const matchedBifurcationCount = Math.round(
    selected.reduce(
      (sum, result) => sum + (result.matchedBifurcationCount ?? 0),
      0
    ) / selected.length
  );
  const coverage =
    selected.reduce((sum, result) => sum + result.coverage, 0) / selected.length;
  const best = selected[0];
  const status: FingerMatchResult['status'] =
    best.status !== 'insufficient' &&
    matchedMinutiae >= config.minimumMatchedMinutiae &&
    coverage >= config.minimumCoverage
      ? 'matched'
      : best.status === 'insufficient'
        ? 'insufficient'
        : 'no-match';
  return {
    ...best,
    fingerPosition,
    status,
    score: Math.round(score * 10) / 10,
    matchedMinutiae,
    matchedEndingCount,
    matchedBifurcationCount,
    coverage: Math.round(coverage * 1000) / 1000,
    enrollmentUsableMinutiae: Math.round(
      selected.reduce((sum, result) => sum + result.enrollmentUsableMinutiae, 0) /
        selected.length
    ),
    failureReasons:
      status === 'matched'
        ? []
        : [...new Set(selected.flatMap((result) => result.failureReasons))],
  };
}

function scorePerson(fingerResults: FingerMatchResult[]) {
  const scored = fingerResults
    .filter((result) => result.status === 'matched');
  if (scored.length === 0) return 0;
  return (
    Math.round(
      (scored.reduce((sum, result) => sum + result.score, 0) / scored.length) *
        10
    ) / 10
  );
}

export function assessPersonMatchEvidence(
  fingerResults: FingerMatchResult[],
  config: FingerprintMatcherConfig = DEFAULT_FINGERPRINT_MATCHER_CONFIG
) {
  const matchedResults = fingerResults.filter(
    (result) => result.status === 'matched'
  );
  const totalMatchedMinutiae = matchedResults.reduce(
    (sum, result) => sum + result.matchedMinutiae,
    0
  );
  const averageCoverage = matchedResults.length
    ? matchedResults.reduce((sum, result) => sum + result.coverage, 0) /
      matchedResults.length
    : 0;
  const matchedFingerCount = matchedResults.length;

  return {
    accepted:
      matchedFingerCount >= config.minimumMatchedFingerCount &&
      totalMatchedMinutiae >= config.minimumTotalMatchedMinutiae &&
      averageCoverage >= config.minimumPersonAverageCoverage,
    matchedFingerCount,
    totalMatchedMinutiae,
    averageCoverage: Math.round(averageCoverage * 1000) / 1000,
  };
}

export function identifyPersonFromFingerRois({
  fingerRois,
  people,
  enrollments,
  config = DEFAULT_FINGERPRINT_MATCHER_CONFIG,
}: {
  fingerRois: FingerRoi[];
  people: Person[];
  enrollments: Enrollment[];
  config?: FingerprintMatcherConfig;
}): PersonMatchResult {
  const capturedProbes = new Map<FingerprintPosition, FingerRoi>();
  const validProbes = new Map<FingerprintPosition, FingerRoi>();
  for (const fingerRoi of fingerRois) {
    if (FINGERPRINT_POSITIONS.includes(fingerRoi.className as FingerprintPosition)) {
      const fingerPosition = fingerRoi.className as FingerprintPosition;
      capturedProbes.set(fingerPosition, fingerRoi);
    }
    if (
      FINGERPRINT_POSITIONS.includes(fingerRoi.className as FingerprintPosition) &&
      fingerRoi.minutiaeTemplate &&
      fingerRoi.quality?.biometricStatus === 'sufficient'
    ) {
      validProbes.set(fingerRoi.className as FingerprintPosition, fingerRoi);
    }
  }

  if (people.length === 0) {
    return {
      personId: '',
      displayName: '',
      accepted: false,
      score: 0,
      matchedFingerCount: 0,
      totalMatchedMinutiae: 0,
      averageCoverage: 0,
      fingerResults: [],
      candidateResults: [],
      validProbeFingerPositions: [],
      reason: 'no-people',
    };
  }

  if (validProbes.size === 0) {
    return {
      personId: '',
      displayName: '',
      accepted: false,
      score: 0,
      matchedFingerCount: 0,
      totalMatchedMinutiae: 0,
      averageCoverage: 0,
      fingerResults: [],
      candidateResults: [],
      validProbeFingerPositions: [],
      reason: 'no-valid-probe',
    };
  }

  const candidates = people.map((person) => {
    const fingerResults = FINGERPRINT_POSITIONS.map((fingerPosition) => {
      const capturedProbe = capturedProbes.get(fingerPosition);
      const probe = validProbes.get(fingerPosition);
      const enrollmentTemplates = enrollments
        .filter(
          (item) =>
            item.personId === person.id &&
            item.fingerPosition === fingerPosition
        )
        .sort((first, second) => (first.sampleIndex ?? 0) - (second.sampleIndex ?? 0));
      if (!capturedProbe) {
        return createMissingFingerResult(
          fingerPosition,
          'missing',
          config,
          'probe-missing'
        );
      }
      if (!probe) {
        return createInsufficientProbeResult(
          fingerPosition,
          capturedProbe,
          config
        );
      }
      if (enrollmentTemplates.length === 0 || !probe.minutiaeTemplate) {
        return createMissingFingerResult(
          fingerPosition,
          'insufficient',
          config,
          'enrollment-missing'
        );
      }
      return aggregateEnrollmentMatches(
        enrollmentTemplates.map((enrollment) =>
          withFingerPosition(
            matchFingerprintTemplates(
              probe.minutiaeTemplate!,
              enrollment.template,
              config
            ),
            fingerPosition
          )
        ),
        fingerPosition,
        config
      );
    });
    const evidence = assessPersonMatchEvidence(fingerResults, config);
    return {
      person,
      fingerResults,
      ...evidence,
      score: scorePerson(fingerResults),
    };
  });

  candidates.sort(
    (first, second) =>
      Number(second.accepted) - Number(first.accepted) ||
      second.matchedFingerCount - first.matchedFingerCount ||
      second.totalMatchedMinutiae - first.totalMatchedMinutiae ||
      second.score - first.score
  );
  const candidateResults: PersonCandidateResult[] = candidates.map(
    (candidate) => ({
      personId: candidate.person.id,
      displayName: candidate.person.displayName,
      score: candidate.score,
      matchedFingerCount: candidate.matchedFingerCount,
      totalMatchedMinutiae: candidate.totalMatchedMinutiae,
      averageCoverage: candidate.averageCoverage,
      fingerResults: candidate.fingerResults,
    })
  );
  const best = candidates[0];
  if (!best || best.score <= 0) {
    return {
      personId: '',
      displayName: '',
      accepted: false,
      score: 0,
      matchedFingerCount: 0,
      totalMatchedMinutiae: 0,
      averageCoverage: 0,
      fingerResults: best?.fingerResults ?? [],
      candidateResults,
      validProbeFingerPositions: [...validProbes.keys()],
      reason: 'no-match',
    };
  }

  const second = candidates[1];
  const secondIsStrong = Boolean(second?.accepted);
  if (
    best.accepted &&
    secondIsStrong &&
    best.score - second.score < config.ambiguityMargin
  ) {
    return {
      personId: best.person.id,
      displayName: best.person.displayName,
      accepted: false,
      score: best.score,
      matchedFingerCount: best.matchedFingerCount,
      totalMatchedMinutiae: best.totalMatchedMinutiae,
      averageCoverage: best.averageCoverage,
      fingerResults: best.fingerResults,
      candidateResults,
      validProbeFingerPositions: [...validProbes.keys()],
      reason: 'ambiguous',
    };
  }

  const bestMatchedPositions = new Set(
    best.fingerResults
      .filter((result) => result.status === 'matched')
      .map((result) => result.fingerPosition)
  );
  const conflictingCandidate = candidates.slice(1).find((candidate) =>
    candidate.fingerResults.some(
      (result) =>
        result.status === 'matched' &&
        !bestMatchedPositions.has(result.fingerPosition)
    )
  );
  if (!best.accepted && conflictingCandidate) {
    return {
      personId: best.person.id,
      displayName: best.person.displayName,
      accepted: false,
      score: best.score,
      matchedFingerCount: best.matchedFingerCount,
      totalMatchedMinutiae: best.totalMatchedMinutiae,
      averageCoverage: best.averageCoverage,
      fingerResults: best.fingerResults,
      candidateResults,
      validProbeFingerPositions: [...validProbes.keys()],
      reason: 'conflicting-evidence',
    };
  }

  const accepted = best.accepted;
  return {
    personId: best.person.id,
    displayName: best.person.displayName,
    accepted,
    score: best.score,
    matchedFingerCount: best.matchedFingerCount,
    totalMatchedMinutiae: best.totalMatchedMinutiae,
    averageCoverage: best.averageCoverage,
    fingerResults: best.fingerResults,
    candidateResults,
    validProbeFingerPositions: [...validProbes.keys()],
    reason: accepted
      ? 'accepted'
      : best.matchedFingerCount > 0
        ? 'insufficient-multi-finger-evidence'
        : 'no-match',
  };
}
