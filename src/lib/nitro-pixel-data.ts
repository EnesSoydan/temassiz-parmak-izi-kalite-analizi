export type NitroPixelFormat =
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

export type NormalizedNitroPixels = {
  pixels: Uint8Array;
  redBlueCorrected: boolean;
};

// Nitro'nun bildirdiği kanal sırasını RGBA'ya çevirir ve cihazdaki olası kırmızı-mavi etiket uyuşmazlığını denetler.
export function convertNitroPixelsToRgba({
  source,
  width,
  height,
  pixelFormat,
}: {
  source: Uint8Array;
  width: number;
  height: number;
  pixelFormat: NitroPixelFormat;
}): NormalizedNitroPixels {
  const bytesPerPixel = pixelFormat === 'RGB' || pixelFormat === 'BGR' ? 3 : 4;
  const pixelCount = Math.min(width * height, Math.floor(source.length / bytesPerPixel));
  const redBlueCorrected = shouldCorrectRedBlueOrder(
    source,
    width,
    height,
    pixelFormat,
    bytesPerPixel
  );
  const pixels = new Uint8Array(width * height * 4);

  for (let index = 0; index < pixelCount; index += 1) {
    const sourceIndex = index * bytesPerPixel;
    const targetIndex = index * 4;
    const color = readNitroRgb(source, sourceIndex, pixelFormat);
    pixels[targetIndex] = redBlueCorrected ? color.blue : color.red;
    pixels[targetIndex + 1] = color.green;
    pixels[targetIndex + 2] = redBlueCorrected ? color.red : color.blue;
    pixels[targetIndex + 3] = 255;
  }

  return { pixels, redBlueCorrected };
}

// ROI merkezindeki ten rengi kanıtı ters okumada belirgin biçimde güçleniyorsa yalnızca kırmızı ve maviyi düzeltir.
function shouldCorrectRedBlueOrder(
  source: Uint8Array,
  width: number,
  height: number,
  pixelFormat: NitroPixelFormat,
  bytesPerPixel: number
) {
  if (!['BGRA', 'BGRX', 'BGR', 'RGBA', 'RGBX', 'RGB'].includes(pixelFormat)) {
    return false;
  }

  const startX = Math.floor(width * 0.18);
  const endX = Math.ceil(width * 0.82);
  const startY = Math.floor(height * 0.08);
  const endY = Math.ceil(height * 0.92);
  const sampleStep = Math.max(1, Math.floor(Math.min(width, height) / 96));
  let reportedEvidence = 0;
  let swappedEvidence = 0;
  let sampledPixels = 0;

  for (let y = startY; y < endY; y += sampleStep) {
    for (let x = startX; x < endX; x += sampleStep) {
      const index = (y * width + x) * bytesPerPixel;
      if (index + bytesPerPixel > source.length) continue;
      const color = readNitroRgb(source, index, pixelFormat);
      reportedEvidence += getSkinColorEvidence(color.red, color.green, color.blue);
      swappedEvidence += getSkinColorEvidence(color.blue, color.green, color.red);
      sampledPixels += 1;
    }
  }

  const minimumLead = Math.max(12, sampledPixels * 0.04);
  return swappedEvidence >= sampledPixels * 0.08 &&
    swappedEvidence > reportedEvidence + minimumLead;
}

// Beyaz arka planı dışarıda bırakıp tipik ten kromasını 0-1 arası kanıt olarak puanlar.
function getSkinColorEvidence(red: number, green: number, blue: number) {
  const y = red * 0.299 + green * 0.587 + blue * 0.114;
  const cr = (red - y) * 0.713 + 128;
  const cb = (blue - y) * 0.564 + 128;
  const hasWarmChannelOrder = red >= green - 18 && red >= blue + 7;
  const hasSkinChroma = cr >= 132 && cr <= 185 && cb >= 72 && cb <= 138;
  return y >= 24 && y <= 248 && hasWarmChannelOrder && hasSkinChroma ? 1 : 0;
}

// Platforma göre değişen Nitro piksel formatından bildirilen RGB kanallarını okur.
function readNitroRgb(
  source: Uint8Array,
  index: number,
  pixelFormat: NitroPixelFormat
) {
  switch (pixelFormat) {
    case 'RGBA':
    case 'RGBX':
    case 'RGB':
      return { red: source[index], green: source[index + 1], blue: source[index + 2] };
    case 'ARGB':
    case 'XRGB':
      return { red: source[index + 1], green: source[index + 2], blue: source[index + 3] };
    case 'ABGR':
    case 'XBGR':
      return { red: source[index + 3], green: source[index + 2], blue: source[index + 1] };
    case 'BGRA':
    case 'BGRX':
    case 'BGR':
    case 'unknown':
    default:
      return { red: source[index + 2], green: source[index + 1], blue: source[index] };
  }
}
