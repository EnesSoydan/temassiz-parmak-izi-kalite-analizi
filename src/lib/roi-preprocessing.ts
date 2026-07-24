import { decode, encode } from 'jpeg-js';

type BufferShim = {
  from: (input: ArrayLike<number> | ArrayBuffer | string) => Uint8Array;
  alloc: (size: number) => Uint8Array;
};

type PreprocessedRoi = {
  base64: string;
  width: number;
  height: number;
};

// Kırpılmış ROI görüntüsünü grayscale + kontrast normalizasyonu ile baseline analizine hazırlar.
export function preprocessRoiImage(base64: string): PreprocessedRoi {
  ensureJpegBufferShim();

  const image = decode(base64ToBytes(base64), { useTArray: true });
  const pixelCount = image.width * image.height;
  const grayValues = new Uint8Array(pixelCount);
  const histogram = new Uint32Array(256);

  // İlk geçişte RGB pikselleri gri tona indirger ve kontrast aralığı için histogram toplarız.
  for (let i = 0; i < pixelCount; i += 1) {
    const pixelIndex = i * 4;
    const red = image.data[pixelIndex] ?? 0;
    const green = image.data[pixelIndex + 1] ?? 0;
    const blue = image.data[pixelIndex + 2] ?? 0;
    const gray = Math.round(0.299 * red + 0.587 * green + 0.114 * blue);

    grayValues[i] = gray;
    histogram[gray] += 1;
  }

  const low = findPercentileFromHistogram(histogram, pixelCount, 0.02);
  const high = findPercentileFromHistogram(histogram, pixelCount, 0.98);
  const range = Math.max(high - low, 1);

  // İkinci geçişte gri değerleri kırpılmış kontrast aralığına yayarız; alpha kanalını opak tutarız.
  for (let i = 0; i < pixelCount; i += 1) {
    const pixelIndex = i * 4;
    const normalized = high - low < 10 ? grayValues[i] : ((grayValues[i] - low) / range) * 255;
    const value = clamp(Math.round(normalized), 0, 255);

    image.data[pixelIndex] = value;
    image.data[pixelIndex + 1] = value;
    image.data[pixelIndex + 2] = value;
    image.data[pixelIndex + 3] = 255;
  }

  const jpeg = encode(image, 92);

  return {
    base64: bytesToBase64(jpeg.data),
    width: image.width,
    height: image.height,
  };
}

// jpeg-js encoder React Native'de global Buffer bekler; sadece ihtiyaç duyduğu from/alloc çağrılarını Uint8Array ile karşılarız.
function ensureJpegBufferShim() {
  const globalScope = globalThis as typeof globalThis & { Buffer?: BufferShim };

  if (globalScope.Buffer) {
    return;
  }

  globalScope.Buffer = {
    from(input) {
      if (typeof input === 'string') {
        return stringToBytes(input);
      }

      if (input instanceof ArrayBuffer) {
        return new Uint8Array(input);
      }

      return Uint8Array.from(input);
    },
    alloc(size) {
      return new Uint8Array(size);
    },
  };
}

// Histogramdan verilen yüzdelik eşik değerini bulur; uç parlama/gölge değerlerini yumuşakça kırpmak için kullanılır.
function findPercentileFromHistogram(histogram: Uint32Array, pixelCount: number, percentile: number) {
  const target = Math.max(1, Math.round(pixelCount * percentile));
  let total = 0;

  for (let value = 0; value < histogram.length; value += 1) {
    total += histogram[value];

    if (total >= target) {
      return value;
    }
  }

  return histogram.length - 1;
}

// Değeri verilen aralık içinde tutar; kontrast normalizasyonunda taşmayı engeller.
function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

// Buffer shim string girdisi alırsa byte dizisine çevirmek için kullanılır.
function stringToBytes(value: string) {
  const bytes = new Uint8Array(value.length);

  for (let i = 0; i < value.length; i += 1) {
    bytes[i] = value.charCodeAt(i) & 255;
  }

  return bytes;
}

// React Native tarafında Buffer'a yaslanmadan base64 string'i byte dizisine çevirir.
function base64ToBytes(base64: string) {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  const lookup = new Uint8Array(256);

  for (let i = 0; i < chars.length; i += 1) {
    lookup[chars.charCodeAt(i)] = i;
  }

  const clean = base64.replace(/[^A-Za-z0-9+/=]/g, '');
  const padding = clean.endsWith('==') ? 2 : clean.endsWith('=') ? 1 : 0;
  const length = Math.floor((clean.length * 3) / 4) - padding;
  const bytes = new Uint8Array(length);

  let byteIndex = 0;

  // Base64 her dört karakterde üç byte üretir; padding varsa son byte'ları atlarız.
  for (let i = 0; i < clean.length; i += 4) {
    const encoded =
      (lookup[clean.charCodeAt(i)] << 18) |
      (lookup[clean.charCodeAt(i + 1)] << 12) |
      (lookup[clean.charCodeAt(i + 2)] << 6) |
      lookup[clean.charCodeAt(i + 3)];

    if (byteIndex < length) {
      bytes[byteIndex] = (encoded >> 16) & 255;
      byteIndex += 1;
    }

    if (byteIndex < length) {
      bytes[byteIndex] = (encoded >> 8) & 255;
      byteIndex += 1;
    }

    if (byteIndex < length) {
      bytes[byteIndex] = encoded & 255;
      byteIndex += 1;
    }
  }

  return bytes;
}

// JPEG encoder çıktısını dosyaya yazılabilecek base64 string'e dönüştürür.
function bytesToBase64(bytes: Uint8Array) {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  let result = '';

  for (let i = 0; i < bytes.length; i += 3) {
    const byte1 = bytes[i] ?? 0;
    const byte2 = bytes[i + 1] ?? 0;
    const byte3 = bytes[i + 2] ?? 0;
    const encoded = (byte1 << 16) | (byte2 << 8) | byte3;

    result += chars[(encoded >> 18) & 63];
    result += chars[(encoded >> 12) & 63];
    result += i + 1 < bytes.length ? chars[(encoded >> 6) & 63] : '=';
    result += i + 2 < bytes.length ? chars[encoded & 63] : '=';
  }

  return result;
}
