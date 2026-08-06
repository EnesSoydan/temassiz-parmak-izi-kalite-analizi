export type NormalizedPoint = {
  x: number;
  y: number;
};

export type PixelPoint = {
  x: number;
  y: number;
};

export type HomographyMatrix = [
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
];

export type LetterboxTransform = {
  sourceWidth: number;
  sourceHeight: number;
  targetSize: number;
  scaledWidth: number;
  scaledHeight: number;
  padX: number;
  padY: number;
  scaleX: number;
  scaleY: number;
};

// Model girdisini oranı bozmadan kare tuvale yerleştirmek için kullanılan dönüşümü üretir.
export function createLetterboxTransform(
  sourceWidth: number,
  sourceHeight: number,
  targetSize: number
): LetterboxTransform {
  if (sourceWidth <= 0 || sourceHeight <= 0 || targetSize <= 0) {
    throw new Error('Letterbox boyutları pozitif olmalı.');
  }

  const scale = Math.min(targetSize / sourceWidth, targetSize / sourceHeight);
  const scaledWidth = Math.max(1, Math.round(sourceWidth * scale));
  const scaledHeight = Math.max(1, Math.round(sourceHeight * scale));
  const padX = Math.floor((targetSize - scaledWidth) / 2);
  const padY = Math.floor((targetSize - scaledHeight) / 2);

  return {
    sourceWidth,
    sourceHeight,
    targetSize,
    scaledWidth,
    scaledHeight,
    padX,
    padY,
    scaleX: scaledWidth / sourceWidth,
    scaleY: scaledHeight / sourceHeight,
  };
}

// Canlı frame'i gereksiz yere büyütmeden en fazla hedef kenara sığdırır.
export function getAspectPreservingSize(
  sourceWidth: number,
  sourceHeight: number,
  maximumEdge: number
) {
  if (sourceWidth <= 0 || sourceHeight <= 0 || maximumEdge <= 0) {
    throw new Error('Görüntü boyutları pozitif olmalı.');
  }

  const scale = Math.min(1, maximumEdge / Math.max(sourceWidth, sourceHeight));
  return {
    width: Math.max(1, Math.round(sourceWidth * scale)),
    height: Math.max(1, Math.round(sourceHeight * scale)),
  };
}

// Model karesindeki koordinatı, padding ve oran dönüşümünü geri alarak kaynak görüntüye taşır.
export function mapModelPointToNormalizedImage(
  modelX: number,
  modelY: number,
  transform: LetterboxTransform
): NormalizedPoint {
  return {
    x: clamp01(
      ((modelX - transform.padX) / transform.scaleX) /
        transform.sourceWidth
    ),
    y: clamp01(
      ((modelY - transform.padY) / transform.scaleY) /
        transform.sourceHeight
    ),
  };
}

// Çizgi ekseni 180 derece simetrik olduğu için en kısa eşdeğer dönüşü seçer.
export function normalizeAxisRotation(degrees: number) {
  let normalized = degrees;
  while (normalized > 90) normalized -= 180;
  while (normalized < -90) normalized += 180;
  return normalized;
}

// OBB köşelerinden uzun ekseni ve dikleştirme dönüşünü hesaplar.
export function getCanonicalGeometry(points?: NormalizedPoint[]) {
  if (!points || points.length !== 4) return null;

  const edges = points.map((point, index) => {
    const next = points[(index + 1) % points.length];
    const deltaX = next.x - point.x;
    const deltaY = next.y - point.y;
    return {
      length: Math.hypot(deltaX, deltaY),
      angleDegrees: (Math.atan2(deltaY, deltaX) * 180) / Math.PI,
    };
  });
  const longEdge = [...edges].sort((first, second) => second.length - first.length)[0];
  const shortEdge = [...edges].sort((first, second) => first.length - second.length)[0];

  if (!longEdge || !shortEdge || shortEdge.length <= Number.EPSILON) return null;

  return {
    longEdge: longEdge.length,
    shortEdge: shortEdge.length,
    rotationDegrees: normalizeAxisRotation(90 - longEdge.angleDegrees),
  };
}

// Dörtgen köşelerini hedef tuvalde sol üstten başlayıp saat yönünde sıralar.
export function orderQuadrilateralPoints(points: PixelPoint[]) {
  if (points.length !== 4) {
    throw new Error('Homografi için tam olarak dört köşe gerekir.');
  }

  const center = points.reduce(
    (result, point) => ({ x: result.x + point.x / 4, y: result.y + point.y / 4 }),
    { x: 0, y: 0 }
  );
  const ordered = [...points].sort(
    (first, second) =>
      Math.atan2(first.y - center.y, first.x - center.x) -
      Math.atan2(second.y - center.y, second.x - center.x)
  );
  const topLeftIndex = ordered.reduce(
    (bestIndex, point, index) =>
      point.x + point.y < ordered[bestIndex].x + ordered[bestIndex].y
        ? index
        : bestIndex,
    0
  );

  return ordered
    .slice(topLeftIndex)
    .concat(ordered.slice(0, topLeftIndex));
}

// OBB dörtgenini sabit dikdörtgen tuvale taşıyan projektif dönüşümü üretir.
export function createQuadToRectangleHomography(
  sourcePoints: PixelPoint[],
  targetWidth: number,
  targetHeight: number,
  longEdgeAxis: 'horizontal' | 'vertical' = 'horizontal'
) {
  if (targetWidth < 2 || targetHeight < 2) {
    throw new Error('Homografi hedef boyutu en az 2x2 olmalı.');
  }

  const orderedSource = orderQuadrilateralPoints(sourcePoints);
  const edgeLengths = orderedSource.map((point, index) => {
    const next = orderedSource[(index + 1) % orderedSource.length];
    return Math.hypot(next.x - point.x, next.y - point.y);
  });
  let source: PixelPoint[];
  let target: PixelPoint[];

  if (longEdgeAxis === 'vertical') {
    // OBB'nin iki uzun kenarı neredeyse eşittir; en uzunu başlangıç seçmek 180° yönü
    // küçük yuvarlama farklarına bırakır. Fiziksel ekran üstündeki kısa kenarı kanonik
    // görüntünün üstüne taşıyarak kaynak ROI yönünü deterministik biçimde koruruz.
    const firstShortEdgeIndex = edgeLengths.indexOf(Math.min(...edgeLengths));
    const oppositeShortEdgeIndex = (firstShortEdgeIndex + 2) % 4;
    const edgeCenterY = (edgeIndex: number) => {
      const first = orderedSource[edgeIndex];
      const second = orderedSource[(edgeIndex + 1) % 4];
      return (first.y + second.y) / 2;
    };
    const topEdgeIndex =
      edgeCenterY(firstShortEdgeIndex) <= edgeCenterY(oppositeShortEdgeIndex)
        ? firstShortEdgeIndex
        : oppositeShortEdgeIndex;
    const bottomEdgeIndex = (topEdgeIndex + 2) % 4;
    const topEdge = [
      orderedSource[topEdgeIndex],
      orderedSource[(topEdgeIndex + 1) % 4],
    ].sort((first, second) => first.x - second.x);
    const bottomEdge = [
      orderedSource[bottomEdgeIndex],
      orderedSource[(bottomEdgeIndex + 1) % 4],
    ].sort((first, second) => first.x - second.x);

    source = [topEdge[0], topEdge[1], bottomEdge[1], bottomEdge[0]];
    target = [
      { x: 0, y: 0 },
      { x: targetWidth - 1, y: 0 },
      { x: targetWidth - 1, y: targetHeight - 1 },
      { x: 0, y: targetHeight - 1 },
    ];
  } else {
    const longEdgeIndex = edgeLengths.indexOf(Math.max(...edgeLengths));
    source = orderedSource
      .slice(longEdgeIndex)
      .concat(orderedSource.slice(0, longEdgeIndex));
    target = [
      { x: 0, y: 0 },
      { x: targetWidth - 1, y: 0 },
      { x: targetWidth - 1, y: targetHeight - 1 },
      { x: 0, y: targetHeight - 1 },
    ];
  }
  const matrix = new Array<number[]>(8);
  const values = new Array<number>(8);

  for (let index = 0; index < 4; index += 1) {
    const sourcePoint = source[index];
    const targetPoint = target[index];
    const row = index * 2;
    matrix[row] = [
      sourcePoint.x,
      sourcePoint.y,
      1,
      0,
      0,
      0,
      -targetPoint.x * sourcePoint.x,
      -targetPoint.x * sourcePoint.y,
    ];
    values[row] = targetPoint.x;
    matrix[row + 1] = [
      0,
      0,
      0,
      sourcePoint.x,
      sourcePoint.y,
      1,
      -targetPoint.y * sourcePoint.x,
      -targetPoint.y * sourcePoint.y,
    ];
    values[row + 1] = targetPoint.y;
  }

  const solution = solveLinearSystem(matrix, values);
  return [
    solution[0],
    solution[1],
    solution[2],
    solution[3],
    solution[4],
    solution[5],
    solution[6],
    solution[7],
    1,
  ] as HomographyMatrix;
}

// Homografi ile bir noktayı yeni koordinat çerçevesine taşır.
export function transformPointByHomography(
  point: PixelPoint,
  matrix: HomographyMatrix
): PixelPoint {
  const denominator = matrix[6] * point.x + matrix[7] * point.y + matrix[8];
  if (Math.abs(denominator) < 1e-8) {
    return { x: Number.NaN, y: Number.NaN };
  }

  return {
    x: (matrix[0] * point.x + matrix[1] * point.y + matrix[2]) / denominator,
    y: (matrix[3] * point.x + matrix[4] * point.y + matrix[5]) / denominator,
  };
}

// Hedef pikseli ters örnekleyerek projektif dönüşüm sonrası RGBA görüntü üretir.
export function warpRgbaImageByHomography({
  pixels,
  width,
  height,
  matrix,
  targetWidth,
  targetHeight,
}: {
  pixels: Uint8Array;
  width: number;
  height: number;
  matrix: HomographyMatrix;
  targetWidth: number;
  targetHeight: number;
}) {
  const inverse = invertHomography(matrix);
  const output = new Uint8Array(targetWidth * targetHeight * 4);

  for (let y = 0; y < targetHeight; y += 1) {
    for (let x = 0; x < targetWidth; x += 1) {
      const source = transformPointByHomography({ x, y }, inverse);
      const outputIndex = (y * targetWidth + x) * 4;
      if (
        !Number.isFinite(source.x) ||
        !Number.isFinite(source.y) ||
        source.x < 0 ||
        source.y < 0 ||
        source.x > width - 1 ||
        source.y > height - 1
      ) {
        output[outputIndex + 3] = 255;
        continue;
      }

      const left = Math.floor(source.x);
      const top = Math.floor(source.y);
      const right = Math.min(left + 1, width - 1);
      const bottom = Math.min(top + 1, height - 1);
      const xWeight = source.x - left;
      const yWeight = source.y - top;
      const samples = [
        { index: (top * width + left) * 4, weight: (1 - xWeight) * (1 - yWeight) },
        { index: (top * width + right) * 4, weight: xWeight * (1 - yWeight) },
        { index: (bottom * width + left) * 4, weight: (1 - xWeight) * yWeight },
        { index: (bottom * width + right) * 4, weight: xWeight * yWeight },
      ];

      for (let channel = 0; channel < 3; channel += 1) {
        output[outputIndex + channel] = Math.round(
          samples.reduce(
            (sum, sample) => sum + pixels[sample.index + channel] * sample.weight,
            0
          )
        );
      }
      output[outputIndex + 3] = 255;
    }
  }

  return output;
}

function invertHomography(matrix: HomographyMatrix): HomographyMatrix {
  const [a, b, c, d, e, f, g, h, i] = matrix;
  const determinant =
    a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (Math.abs(determinant) < 1e-10) {
    throw new Error('Homografi matrisi terslenemedi.');
  }

  return [
    (e * i - f * h) / determinant,
    (c * h - b * i) / determinant,
    (b * f - c * e) / determinant,
    (f * g - d * i) / determinant,
    (a * i - c * g) / determinant,
    (c * d - a * f) / determinant,
    (d * h - e * g) / determinant,
    (b * g - a * h) / determinant,
    (a * e - b * d) / determinant,
  ];
}

function solveLinearSystem(matrix: number[][], values: number[]) {
  const augmented = matrix.map((row, index) => [...row, values[index]]);

  for (let column = 0; column < 8; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < 8; row += 1) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) {
        pivot = row;
      }
    }
    if (Math.abs(augmented[pivot][column]) < 1e-10) {
      throw new Error('Homografi için çözülemeyen köşe geometrisi.');
    }
    [augmented[column], augmented[pivot]] = [augmented[pivot], augmented[column]];

    const pivotValue = augmented[column][column];
    for (let item = column; item <= 8; item += 1) {
      augmented[column][item] /= pivotValue;
    }

    for (let row = 0; row < 8; row += 1) {
      if (row === column) continue;
      const factor = augmented[row][column];
      for (let item = column; item <= 8; item += 1) {
        augmented[row][item] -= factor * augmented[column][item];
      }
    }
  }

  return augmented.map((row) => row[8]);
}

function clamp01(value: number) {
  return Math.min(Math.max(value, 0), 1);
}
