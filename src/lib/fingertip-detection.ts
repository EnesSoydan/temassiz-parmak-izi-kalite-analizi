import { decode } from 'jpeg-js';
import {
  Images,
  type Image as NitroImage,
} from 'react-native-nitro-image';

import { getFingertipObbRuntime } from '@/lib/onnx-model';
import {
  createLetterboxTransform,
  mapModelPointToNormalizedImage,
  type LetterboxTransform,
} from '@/lib/roi-geometry';
import type { DetectedObbBox, DetectionClassName } from '@/types/biometrics';

export const FINGERTIP_MODEL_SIZE = 480;
const FOUR_CLASS_OUTPUT_CHANNELS = 9;
const FIVE_CLASS_OUTPUT_CHANNELS = 10;
// Küçük serçe kutusunun canlıda kabul edilip fotoğrafta elenmemesi için iki akış aynı güven eşiğini kullanır.
const PHOTO_DETECTION_THRESHOLD = 0.2;
const LIVE_DETECTION_THRESHOLD = 0.2;
const NMS_IOU_THRESHOLD = 0.45;
const MAX_DETECTIONS = 30;
const MIN_INNER_FINGER_PROJECTION_GAP = 0.15;
const EXPECTED_MIDDLE_PROJECTION = 1 / 3;
const EXPECTED_RING_PROJECTION = 2 / 3;
const CLASS_NAMES: DetectionClassName[] = ['index', 'middle', 'pinky', 'ring'];
const FINGER_CLASS_IDS = {
  middle: 1,
  ring: 3,
} as const;

type RawObbCandidate = DetectedObbBox & {
  axisAlignedBox: {
    left: number;
    top: number;
    right: number;
    bottom: number;
  };
};

type TensorImage = {
  tensorData: Float32Array;
  transform: LetterboxTransform;
};

// Nitro Image'in Android/iOS tarafında üretebileceği ham piksel sıralarını tanımlar.
export type ModelPixelFormat =
  | 'ARGB'
  | 'BGRA'
  | 'ABGR'
  | 'RGBA'
  | 'XRGB'
  | 'BGRX'
  | 'XBGR'
  | 'RGBX'
  | 'RGB'
  | 'BGR'
  | 'unknown';

// Ham el fotoğrafını mobil modele uygun kare RGB tensor'a çevirir.
async function createModelInputTensor(imageUri: string): Promise<TensorImage> {
  const image = await Images.loadFromFileAsync(toNativeFilePath(imageUri));
  try {
    return await createModelInputFromImage(image);
  } finally {
    image.dispose();
  }
}

// Fotoğrafı ONNX Runtime'a gönderir ve model çıktısını çizilebilir OBB kutularına dönüştürür.
export async function detectFingertipObbBoxes(imageUri: string): Promise<DetectedObbBox[]> {
  const tensorImage = await createModelInputTensor(imageUri);
  return runFingertipModel(
    tensorImage.tensorData,
    PHOTO_DETECTION_THRESHOLD,
    tensorImage.transform
  );
}

// Kameranın yönü düzeltilmiş native görüntüsünü JPEG'e çevirmeden model boyutuna indirip modele gönderir.
export async function detectFingertipObbBoxesFromImage(
  image: NitroImage
): Promise<DetectedObbBox[]> {
  const tensorImage = await createModelInputFromImage(image);
  return runFingertipModel(
    tensorImage.tensorData,
    PHOTO_DETECTION_THRESHOLD,
    tensorImage.transform
  );
}

// Canlı kameradan gelen kare ham RGB piksel verisini dosya oluşturmadan ONNX modeline gönderir.
export async function detectFingertipObbBoxesFromRawPixels(
  buffer: ArrayBuffer,
  width: number,
  height: number,
  pixelFormat: ModelPixelFormat
): Promise<DetectedObbBox[]> {
  const transform = createLetterboxTransform(width, height, FINGERTIP_MODEL_SIZE);
  const tensorData = createLetterboxedTensorFromRawPixels(
    buffer,
    width,
    height,
    pixelFormat,
    transform
  );
  return runFingertipModel(tensorData, LIVE_DETECTION_THRESHOLD, transform);
}

// Canlı frame'in JPEG verisini fotoğrafla aynı RGBA çözümleme yolundan geçirerek modele gönderir.
export async function detectFingertipObbBoxesFromEncodedJpeg(
  buffer: ArrayBuffer,
  width: number,
  height: number
): Promise<DetectedObbBox[]> {
  const image = decode(new Uint8Array(buffer), { useTArray: true });
  const transform = createLetterboxTransform(image.width, image.height, FINGERTIP_MODEL_SIZE);
  const tensorData = createLetterboxedTensorFromRgbaPixels(
    image.data,
    image.width,
    image.height,
    transform
  );
  return runFingertipModel(tensorData, LIVE_DETECTION_THRESHOLD, transform);
}

// Hazırlanmış CHW tensor verisini mevcut ONNX oturumunda çalıştırıp OBB sonuçlarını döndürür.
async function runFingertipModel(
  tensorData: Float32Array,
  detectionThreshold: number,
  transform: LetterboxTransform
) {
  const { ort, session } = await getFingertipObbRuntime();
  const inputTensor = new ort.Tensor('float32', tensorData, [
    1,
    3,
    FINGERTIP_MODEL_SIZE,
    FINGERTIP_MODEL_SIZE,
  ]);
  const outputs = await session.run({ images: inputTensor });
  const outputTensor = outputs.output0 ?? Object.values(outputs)[0];

  if (!outputTensor || !(outputTensor.data instanceof Float32Array)) {
    throw new Error('ONNX modeli beklenen float çıktı üretmedi.');
  }

  const outputChannels = outputTensor.dims[1];

  // Eski model 5 sınıf, yeni model 4 sınıf üretir; kanal sayısını model çıktısından doğrularız.
  if (
    outputChannels !== FOUR_CLASS_OUTPUT_CHANNELS &&
    outputChannels !== FIVE_CLASS_OUTPUT_CHANNELS
  ) {
    throw new Error(`Desteklenmeyen ONNX çıktı şekli: ${outputTensor.dims.join('x')}.`);
  }

  return parseObbOutput(
    outputTensor.data,
    outputChannels,
    detectionThreshold,
    transform
  );
}

// JPEG çözücünün ürettiği RGBA piksellerini modelin beklediği normalize CHW RGB dizisine çevirir.
function createLetterboxedTensorFromRgbaPixels(
  pixels: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
  transform: LetterboxTransform
) {
  const tensorData = createFilledModelTensor(114);
  for (let targetY = 0; targetY < transform.scaledHeight; targetY += 1) {
    const sourceY = Math.min(
      Math.floor((targetY * height) / transform.scaledHeight),
      height - 1
    );
    for (let targetX = 0; targetX < transform.scaledWidth; targetX += 1) {
      const sourceX = Math.min(
        Math.floor((targetX * width) / transform.scaledWidth),
        width - 1
      );
      const sourceIndex = (sourceY * width + sourceX) * 4;
      writeModelPixel(
        tensorData,
        transform.padX + targetX,
        transform.padY + targetY,
        pixels[sourceIndex] ?? 0,
        pixels[sourceIndex + 1] ?? 0,
        pixels[sourceIndex + 2] ?? 0
      );
    }
  }
  return tensorData;
}

// Nitro Image piksel sırasını okuyup modelin beklediği normalize CHW RGB dizisine dönüştürür.
function createLetterboxedTensorFromRawPixels(
  buffer: ArrayBuffer,
  width: number,
  height: number,
  pixelFormat: ModelPixelFormat,
  transform: LetterboxTransform
) {
  const channelLayout = getPixelChannelLayout(pixelFormat);
  const pixels = new Uint8Array(buffer);
  const pixelCount = width * height;
  const expectedLength = pixelCount * channelLayout.bytesPerPixel;

  if (pixels.length < expectedLength) {
    throw new Error(`Canlı piksel tamponu eksik: ${pixels.length}/${expectedLength} byte.`);
  }

  const tensorData = createFilledModelTensor(114);

  for (let targetY = 0; targetY < transform.scaledHeight; targetY += 1) {
    const sourceY = Math.min(
      Math.floor((targetY * height) / transform.scaledHeight),
      height - 1
    );
    for (let targetX = 0; targetX < transform.scaledWidth; targetX += 1) {
      const sourceX = Math.min(
        Math.floor((targetX * width) / transform.scaledWidth),
        width - 1
      );
      const sourceIndex = (sourceY * width + sourceX) * channelLayout.bytesPerPixel;
      writeModelPixel(
        tensorData,
        transform.padX + targetX,
        transform.padY + targetY,
        pixels[sourceIndex + channelLayout.red] ?? 0,
        pixels[sourceIndex + channelLayout.green] ?? 0,
        pixels[sourceIndex + channelLayout.blue] ?? 0
      );
    }
  }

  return tensorData;
}

async function createModelInputFromImage(image: NitroImage): Promise<TensorImage> {
  const transform = createLetterboxTransform(
    image.width,
    image.height,
    FINGERTIP_MODEL_SIZE
  );
  const resized = await image.resizeAsync(transform.scaledWidth, transform.scaledHeight);

  try {
    const pixels = await resized.toRawPixelDataAsync(false);
    return {
      tensorData: createLetterboxedTensorFromRawPixels(
        pixels.buffer,
        pixels.width,
        pixels.height,
        'RGBA',
        transform
      ),
      transform,
    };
  } finally {
    resized.dispose();
  }
}

function createFilledModelTensor(backgroundValue: number) {
  const pixelCount = FINGERTIP_MODEL_SIZE * FINGERTIP_MODEL_SIZE;
  const normalizedBackground = backgroundValue / 255;
  const tensorData = new Float32Array(pixelCount * 3);
  tensorData.fill(normalizedBackground);
  return tensorData;
}

function writeModelPixel(
  tensorData: Float32Array,
  x: number,
  y: number,
  red: number,
  green: number,
  blue: number
) {
  const pixel = y * FINGERTIP_MODEL_SIZE + x;
  const pixelCount = FINGERTIP_MODEL_SIZE * FINGERTIP_MODEL_SIZE;
  tensorData[pixel] = red / 255;
  tensorData[pixelCount + pixel] = green / 255;
  tensorData[pixelCount * 2 + pixel] = blue / 255;
}

function toNativeFilePath(uri: string) {
  return decodeURIComponent(uri.replace(/^file:\/\//, ''));
}

// Platforma göre değişebilen ARGB/BGRA benzeri piksel dizilimlerinin RGB indislerini belirler.
function getPixelChannelLayout(pixelFormat: ModelPixelFormat) {
  if (pixelFormat === 'ARGB' || pixelFormat === 'XRGB') {
    return { bytesPerPixel: 4, red: 1, green: 2, blue: 3 };
  }

  if (pixelFormat === 'BGRA' || pixelFormat === 'BGRX') {
    return { bytesPerPixel: 4, red: 2, green: 1, blue: 0 };
  }

  if (pixelFormat === 'ABGR' || pixelFormat === 'XBGR') {
    return { bytesPerPixel: 4, red: 3, green: 2, blue: 1 };
  }

  if (pixelFormat === 'RGBA' || pixelFormat === 'RGBX') {
    return { bytesPerPixel: 4, red: 0, green: 1, blue: 2 };
  }

  if (pixelFormat === 'RGB') {
    return { bytesPerPixel: 3, red: 0, green: 1, blue: 2 };
  }

  if (pixelFormat === 'BGR') {
    return { bytesPerPixel: 3, red: 2, green: 1, blue: 0 };
  }

  throw new Error(`Desteklenmeyen canlı piksel formatı: ${pixelFormat}.`);
}

// YOLO OBB ham çıktısını modelin sınıf sayısına göre normalize köşe noktalarına çevirir.
function parseObbOutput(
  output: Float32Array,
  outputChannels: number,
  detectionThreshold: number,
  transform: LetterboxTransform
): DetectedObbBox[] {
  const anchorCount = Math.floor(output.length / outputChannels);
  const modelClassCount = outputChannels - 5;
  const candidates: RawObbCandidate[] = [];

  for (let anchor = 0; anchor < anchorCount; anchor += 1) {
    const centerX = output[anchor] ?? 0;
    const centerY = output[anchorCount + anchor] ?? 0;
    const width = Math.max(output[anchorCount * 2 + anchor] ?? 0, 1);
    const height = Math.max(output[anchorCount * 3 + anchor] ?? 0, 1);
    const classResult = findBestClass(output, anchorCount, anchor, modelClassCount);
    const angle = output[anchorCount * (4 + modelClassCount) + anchor] ?? 0;

    // Beş sınıflı eski modeldeki hand sınıfını kutu ve sayaç sonuçlarına dahil etmeyiz.
    if (classResult.isHand || classResult.confidence < detectionThreshold) {
      continue;
    }

    const points = createNormalizedObbPoints(
      centerX,
      centerY,
      width,
      height,
      angle,
      transform
    );
    const axisAlignedBox = getAxisAlignedBox(points);
    const center = mapModelPointToNormalizedImage(centerX, centerY, transform);

    candidates.push({
      id: `obb-${anchor}`,
      classId: classResult.fingerClassId,
      className: CLASS_NAMES[classResult.fingerClassId] ?? 'unknown',
      confidence: classResult.confidence,
      center: {
        x: center.x,
        y: center.y,
      },
      size: {
        width: Math.max(0, axisAlignedBox.right - axisAlignedBox.left),
        height: Math.max(0, axisAlignedBox.bottom - axisAlignedBox.top),
      },
      angle: getPhysicalAngle(points, transform),
      points,
      axisAlignedBox,
    });
  }

  return keepHighestConfidencePerClass(correctMiddleAndRingLabels(applyNms(candidates)))
    .slice(0, MAX_DETECTIONS)
    .map(({ axisAlignedBox, ...detection }) => detection);
}

// Her anchor için en yüksek sınıf skorunu bulur ve eski hand sınıfının kaymasını düzeltir.
function findBestClass(
  output: Float32Array,
  anchorCount: number,
  anchor: number,
  modelClassCount: number
) {
  let modelClassId = 0;
  let confidence = 0;

  for (let classIndex = 0; classIndex < modelClassCount; classIndex += 1) {
    const rawScore = output[anchorCount * (4 + classIndex) + anchor] ?? 0;
    const score = normalizeModelScore(rawScore);

    if (score > confidence) {
      modelClassId = classIndex;
      confidence = score;
    }
  }

  const hasHandClass = modelClassCount === 5;

  return {
    confidence,
    isHand: hasHandClass && modelClassId === 0,
    fingerClassId: hasHandClass ? modelClassId - 1 : modelClassId,
  };
}

// Bazı exportlarda skor zaten 0-1 aralığında gelir; değilse sigmoid ile olasılığa yaklaştırırız.
function normalizeModelScore(score: number) {
  if (score >= 0 && score <= 1) {
    return score;
  }

  return 1 / (1 + Math.exp(-score));
}

// Merkez, boyut ve açı bilgisinden normalize edilmiş dört OBB köşesi üretir.
function createNormalizedObbPoints(
  centerX: number,
  centerY: number,
  width: number,
  height: number,
  angle: number,
  transform: LetterboxTransform
) {
  const halfWidth = width / 2;
  const halfHeight = height / 2;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const corners = [
    { x: -halfWidth, y: -halfHeight },
    { x: halfWidth, y: -halfHeight },
    { x: halfWidth, y: halfHeight },
    { x: -halfWidth, y: halfHeight },
  ];

  return corners.map((corner) =>
    mapModelPointToNormalizedImage(
      centerX + corner.x * cos - corner.y * sin,
      centerY + corner.x * sin + corner.y * cos,
      transform
    )
  );
}

function getPhysicalAngle(
  points: DetectedObbBox['points'],
  transform: LetterboxTransform
) {
  const edges = points.map((point, index) => {
    const next = points[(index + 1) % points.length];
    const deltaX = (next?.x ?? point.x) - point.x;
    const deltaY = (next?.y ?? point.y) - point.y;
    return {
      length: Math.hypot(
        deltaX * transform.sourceWidth,
        deltaY * transform.sourceHeight
      ),
      deltaX,
      deltaY,
    };
  });
  const longEdge = [...edges].sort((first, second) => second.length - first.length)[0];
  if (!longEdge) return 0;

  return Math.atan2(
    longEdge.deltaY * transform.sourceHeight,
    longEdge.deltaX * transform.sourceWidth
  );
}

// OBB kutuları arasında hızlı NMS yapmak için köşelerden eksene hizalı çevre kutusu çıkarır.
function getAxisAlignedBox(points: DetectedObbBox['points']) {
  const xValues = points.map((point) => point.x);
  const yValues = points.map((point) => point.y);

  return {
    left: Math.min(...xValues),
    top: Math.min(...yValues),
    right: Math.max(...xValues),
    bottom: Math.max(...yValues),
  };
}

// Aynı bölgeyi anlatan düşük skorlu kutuları temizler.
function applyNms(candidates: RawObbCandidate[]) {
  const selected: RawObbCandidate[] = [];
  const sortedCandidates = [...candidates].sort((a, b) => b.confidence - a.confidence);

  for (const candidate of sortedCandidates) {
    const hasOverlap = selected.some(
      (selectedCandidate) =>
        canSuppressEachOther(selectedCandidate, candidate) &&
        calculateIoU(selectedCandidate.axisAlignedBox, candidate.axisAlignedBox) > NMS_IOU_THRESHOLD
    );

    if (!hasOverlap) {
      selected.push(candidate);
    }
  }

  return selected;
}

// Farklı parmak sınıflarının yakın durdukları için birbirini NMS ile silmesini engeller.
function canSuppressEachOther(first: RawObbCandidate, second: RawObbCandidate) {
  return first.className === second.className;
}

// Güvenilir işaret-serçe ekseninde en iyi iki iç parmak adayını orta ve yüzük olarak birlikte atar.
function correctMiddleAndRingLabels(candidates: RawObbCandidate[]) {
  const indexFinger = findHighestConfidenceDetection(candidates, 'index');
  const pinkyFinger = findHighestConfidenceDetection(candidates, 'pinky');

  // İki dış parmak bulunmadan yön güvenilir olmayacağı için model sınıflarını değiştirmeyiz.
  if (!indexFinger || !pinkyFinger) {
    return candidates;
  }

  const axisX = pinkyFinger.center.x - indexFinger.center.x;
  const axisY = pinkyFinger.center.y - indexFinger.center.y;
  const axisLengthSquared = axisX * axisX + axisY * axisY;

  // İşaret ve serçe merkezleri neredeyse aynıysa geometrik düzeltme kararsız olur.
  if (axisLengthSquared < 0.0025) {
    return candidates;
  }

  const innerCandidates = candidates
    .filter((candidate) => candidate.className === 'middle' || candidate.className === 'ring')
    .map((candidate) => ({
      candidate,
      projection: projectCandidateOntoFingerAxis(candidate, indexFinger, axisX, axisY, axisLengthSquared),
    }))
    .filter(({ projection }) => projection > 0.05 && projection < 0.95)
    .sort((first, second) => first.projection - second.projection);

  if (innerCandidates.length === 0) {
    return candidates;
  }

  const selectedPair = findBestInnerFingerPair(innerCandidates);

  // İki ayrı iç parmak bulunduğunda işarete yakın olanı orta, serçeye yakın olanı yüzük yaparız.
  if (selectedPair) {
    const [middleCandidate, ringCandidate] = selectedPair;
    const otherCandidates = candidates.filter(
      (candidate) => candidate.className !== 'middle' && candidate.className !== 'ring'
    );

    return [
      ...otherCandidates,
      relabelInnerFinger(middleCandidate.candidate, 'middle'),
      relabelInnerFinger(ringCandidate.candidate, 'ring'),
    ];
  }

  // Yalnızca tek güvenilir iç kutu varsa anatomik olarak en yakın sınıfa atayıp kopya üretmeyiz.
  const singleCandidate = [...innerCandidates].sort(
    (first, second) => second.candidate.confidence - first.candidate.confidence
  )[0];
  const singleClass =
    Math.abs(singleCandidate.projection - EXPECTED_MIDDLE_PROJECTION) <=
    Math.abs(singleCandidate.projection - EXPECTED_RING_PROJECTION)
      ? 'middle'
      : 'ring';
  const otherCandidates = candidates.filter(
    (candidate) => candidate.className !== 'middle' && candidate.className !== 'ring'
  );

  return [...otherCandidates, relabelInnerFinger(singleCandidate.candidate, singleClass)];
}

// Bir iç parmak adayının işaret parmağından serçe parmağa uzanan eksendeki 0-1 konumunu hesaplar.
function projectCandidateOntoFingerAxis(
  candidate: RawObbCandidate,
  indexFinger: RawObbCandidate,
  axisX: number,
  axisY: number,
  axisLengthSquared: number
) {
  return (
    ((candidate.center.x - indexFinger.center.x) * axisX +
      (candidate.center.y - indexFinger.center.y) * axisY) /
    axisLengthSquared
  );
}

// Birbirinden ayrı iki iç parmak adayından güven ve anatomik konumu birlikte en iyi olan çifti seçer.
function findBestInnerFingerPair(
  innerCandidates: { candidate: RawObbCandidate; projection: number }[]
) {
  let bestPair:
    | [
        { candidate: RawObbCandidate; projection: number },
        { candidate: RawObbCandidate; projection: number },
      ]
    | null = null;
  let bestScore = Number.NEGATIVE_INFINITY;

  for (let firstIndex = 0; firstIndex < innerCandidates.length; firstIndex += 1) {
    for (let secondIndex = firstIndex + 1; secondIndex < innerCandidates.length; secondIndex += 1) {
      const first = innerCandidates[firstIndex];
      const second = innerCandidates[secondIndex];

      if (second.projection - first.projection < MIN_INNER_FINGER_PROJECTION_GAP) {
        continue;
      }

      const anatomyPenalty =
        Math.abs(first.projection - EXPECTED_MIDDLE_PROJECTION) +
        Math.abs(second.projection - EXPECTED_RING_PROJECTION);
      const score = first.candidate.confidence + second.candidate.confidence - anatomyPenalty * 0.5;

      if (score > bestScore) {
        bestScore = score;
        bestPair = [first, second];
      }
    }
  }

  return bestPair;
}

// Seçilen iç parmak kutusunun sınıf kimliğini geometrik karara göre günceller.
function relabelInnerFinger(
  candidate: RawObbCandidate,
  className: 'middle' | 'ring'
): RawObbCandidate {
  return {
    ...candidate,
    classId: FINGER_CLASS_IDS[className],
    className,
  };
}

// Tek el kuralına göre her sınıftan yalnızca en yüksek güvenli bir kutu bırakır.
function keepHighestConfidencePerClass(candidates: RawObbCandidate[]) {
  const bestByClass = new Map<DetectionClassName, RawObbCandidate>();

  for (const candidate of candidates) {
    const currentBest = bestByClass.get(candidate.className);

    if (!currentBest || candidate.confidence > currentBest.confidence) {
      bestByClass.set(candidate.className, candidate);
    }
  }

  return [...bestByClass.values()].sort((first, second) => second.confidence - first.confidence);
}

// Belirtilen sınıftaki en güvenilir kutuyu geometrik referans olarak seçer.
function findHighestConfidenceDetection(
  candidates: RawObbCandidate[],
  className: 'index' | 'pinky'
) {
  return candidates
    .filter((candidate) => candidate.className === className)
    .sort((a, b) => b.confidence - a.confidence)[0];
}

// NMS kararında kullanılan basit eksene hizalı IoU hesabını yapar.
function calculateIoU(
  first: RawObbCandidate['axisAlignedBox'],
  second: RawObbCandidate['axisAlignedBox']
) {
  const left = Math.max(first.left, second.left);
  const top = Math.max(first.top, second.top);
  const right = Math.min(first.right, second.right);
  const bottom = Math.min(first.bottom, second.bottom);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  const firstArea = Math.max(0, first.right - first.left) * Math.max(0, first.bottom - first.top);
  const secondArea = Math.max(0, second.right - second.left) * Math.max(0, second.bottom - second.top);
  const union = firstArea + secondArea - intersection;

  return union <= 0 ? 0 : intersection / union;
}
