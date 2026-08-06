// Maskeli CLAHE-benzeri yerel kontrast artırımı; parlama histogramının baskın olmasını engeller.
export function createMaskedClaheGrayscale(
  grayscale: Uint8Array,
  mask: Uint8Array,
  width: number,
  height: number,
  tileSize = 32
) {
  const output = new Uint8Array(grayscale.length);
  const tilesX = Math.ceil(width / tileSize);
  const tilesY = Math.ceil(height / tileSize);
  const luts: Uint8Array[] = [];

  for (let tileY = 0; tileY < tilesY; tileY += 1) {
    for (let tileX = 0; tileX < tilesX; tileX += 1) {
      const histogram = new Uint32Array(256);
      const left = tileX * tileSize;
      const top = tileY * tileSize;
      const right = Math.min(width, left + tileSize);
      const bottom = Math.min(height, top + tileSize);
      let sampleCount = 0;

      for (let y = top; y < bottom; y += 1) {
        for (let x = left; x < right; x += 1) {
          const index = y * width + x;
          if (!mask[index]) continue;
          histogram[grayscale[index]] += 1;
          sampleCount += 1;
        }
      }

      if (sampleCount < 16) {
        const identity = new Uint8Array(256);
        for (let value = 0; value < 256; value += 1) identity[value] = value;
        luts.push(identity);
        continue;
      }

      const clipLimit = Math.max(2, Math.round(sampleCount * 0.025));
      let excess = 0;
      for (let value = 0; value < histogram.length; value += 1) {
        if (histogram[value] > clipLimit) {
          excess += histogram[value] - clipLimit;
          histogram[value] = clipLimit;
        }
      }
      const redistributed = Math.floor(excess / histogram.length);
      const remainder = excess % histogram.length;
      for (let value = 0; value < histogram.length; value += 1) {
        histogram[value] += redistributed + (value < remainder ? 1 : 0);
      }

      const low = getHistogramPercentile(histogram, sampleCount, 0.04);
      const high = getHistogramPercentile(histogram, sampleCount, 0.96);
      const lowCdf = getHistogramCdf(histogram, low);
      const highCdf = Math.max(lowCdf + 1, getHistogramCdf(histogram, high));
      const lut = new Uint8Array(256);
      let cumulative = 0;
      for (let value = 0; value < histogram.length; value += 1) {
        cumulative += histogram[value];
        lut[value] = clampByte(
          ((cumulative - lowCdf) / (highCdf - lowCdf)) * 255
        );
      }
      luts.push(lut);
    }
  }

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      if (!mask[index]) continue;
      const tileX = Math.min(tilesX - 1, Math.floor(x / tileSize));
      const tileY = Math.min(tilesY - 1, Math.floor(y / tileSize));
      output[index] = luts[tileY * tilesX + tileX][grayscale[index]];
    }
  }

  return output;
}

function getHistogramCdf(histogram: Uint32Array, value: number) {
  let total = 0;
  for (let index = 0; index <= value; index += 1) total += histogram[index];
  return total;
}

function getHistogramPercentile(
  histogram: Uint32Array,
  count: number,
  percentile: number
) {
  if (count === 0) return 0;
  const target = Math.floor((count - 1) * percentile);
  let cumulative = 0;
  for (let value = 0; value < histogram.length; value += 1) {
    cumulative += histogram[value];
    if (cumulative > target) return value;
  }
  return 255;
}

function clampByte(value: number) {
  return Math.round(Math.min(Math.max(value, 0), 255));
}
