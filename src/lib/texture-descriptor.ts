// Canonical ridge görüntüsünü küçük, cihaz üzerinde hesaplanabilir bir texture vektörüne çevirir.
export function createRidgeTextureDescriptor({
  pixels,
  mask,
  width,
  height,
  columns = 8,
  rows = 4,
}: {
  pixels: Uint8Array;
  mask: Uint8Array;
  width: number;
  height: number;
  columns?: number;
  rows?: number;
}) {
  const descriptor = new Float32Array(columns * rows * 2);
  let descriptorIndex = 0;
  for (let row = 0; row < rows; row += 1) {
    const top = Math.floor((row * height) / rows);
    const bottom = Math.max(top + 1, Math.floor(((row + 1) * height) / rows));
    for (let column = 0; column < columns; column += 1) {
      const left = Math.floor((column * width) / columns);
      const right = Math.max(
        left + 1,
        Math.floor(((column + 1) * width) / columns)
      );
      let valueTotal = 0;
      let gradientTotal = 0;
      let count = 0;
      for (let y = top; y < Math.min(bottom, height); y += 1) {
        for (let x = left; x < Math.min(right, width); x += 1) {
          const index = y * width + x;
          if (!mask[index]) continue;
          const value = readGrayscale(pixels, index);
          const neighbour = x + 1 < width ? readGrayscale(pixels, index + 1) : value;
          valueTotal += value;
          gradientTotal += Math.abs(value - neighbour);
          count += 1;
        }
      }
      descriptor[descriptorIndex] = count > 0 ? valueTotal / count / 255 : 0;
      descriptor[descriptorIndex + 1] =
        count > 0 ? gradientTotal / count / 255 : 0;
      descriptorIndex += 2;
    }
  }

  const mean = descriptor.reduce((sum, value) => sum + value, 0) / descriptor.length;
  let norm = 0;
  for (let index = 0; index < descriptor.length; index += 1) {
    descriptor[index] -= mean;
    norm += descriptor[index] ** 2;
  }
  const inverseNorm = 1 / Math.max(Math.sqrt(norm), 1e-6);
  return Array.from(descriptor, (value) => value * inverseNorm);
}

export function cosineSimilarity(first: number[], second: number[]) {
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
  return Math.min(1, Math.max(0, (dot / Math.sqrt(firstNorm * secondNorm) + 1) / 2));
}

function readGrayscale(pixels: Uint8Array, pixelIndex: number) {
  const index = pixelIndex * 4;
  return (
    pixels[index] * 0.299 +
    pixels[index + 1] * 0.587 +
    pixels[index + 2] * 0.114
  );
}
