// Harici native bağımlılık olmadan 8-bit maskeyi gri tonlu PNG'ye çevirir.
export function encodeGrayscalePng(mask: Uint8Array, width: number, height: number) {
  if (mask.length < width * height || width <= 0 || height <= 0) {
    throw new Error('PNG maskesi için geçersiz boyut.');
  }
  const scanlines = new Uint8Array(height * (width + 1));
  for (let y = 0; y < height; y += 1) {
    const rowStart = y * (width + 1);
    scanlines[rowStart] = 0;
    for (let x = 0; x < width; x += 1) {
      scanlines[rowStart + x + 1] = mask[y * width + x] ? 255 : 0;
    }
  }
  const compressed = createStoredZlibStream(scanlines);
  const header = new Uint8Array(13);
  writeUint32(header, 0, width);
  writeUint32(header, 4, height);
  header[8] = 8;
  return concatBytes(
    new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]),
    createChunk('IHDR', header),
    createChunk('IDAT', compressed),
    createChunk('IEND', new Uint8Array())
  );
}

function createStoredZlibStream(data: Uint8Array) {
  const chunks: Uint8Array[] = [new Uint8Array([0x78, 0x01])];
  let offset = 0;
  while (offset < data.length) {
    const length = Math.min(65535, data.length - offset);
    const block = new Uint8Array(5 + length);
    block[0] = offset + length >= data.length ? 1 : 0;
    block[1] = length & 255;
    block[2] = (length >>> 8) & 255;
    block[3] = (~length) & 255;
    block[4] = (~length >>> 8) & 255;
    block.set(data.subarray(offset, offset + length), 5);
    chunks.push(block);
    offset += length;
  }
  const checksum = new Uint8Array(4);
  writeUint32(checksum, 0, adler32(data));
  chunks.push(checksum);
  return concatBytes(...chunks);
}

function createChunk(type: string, data: Uint8Array) {
  const typeBytes = new Uint8Array(type.length);
  for (let index = 0; index < type.length; index += 1) {
    typeBytes[index] = type.charCodeAt(index);
  }
  const length = new Uint8Array(4);
  writeUint32(length, 0, data.length);
  const crc = new Uint8Array(4);
  writeUint32(crc, 0, crc32(concatBytes(typeBytes, data)));
  return concatBytes(length, typeBytes, data, crc);
}

function adler32(data: Uint8Array) {
  let sumA = 1;
  let sumB = 0;
  for (const value of data) {
    sumA = (sumA + value) % 65521;
    sumB = (sumB + sumA) % 65521;
  }
  return ((sumB << 16) | sumA) >>> 0;
}

function crc32(data: Uint8Array) {
  let crc = 0xffffffff;
  for (const value of data) {
    crc ^= value;
    for (let bit = 0; bit < 8; bit += 1) {
      crc = crc & 1 ? (crc >>> 1) ^ 0xedb88320 : crc >>> 1;
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function writeUint32(target: Uint8Array, offset: number, value: number) {
  target[offset] = (value >>> 24) & 255;
  target[offset + 1] = (value >>> 16) & 255;
  target[offset + 2] = (value >>> 8) & 255;
  target[offset + 3] = value & 255;
}

function concatBytes(...arrays: Uint8Array[]) {
  const output = new Uint8Array(arrays.reduce((sum, array) => sum + array.length, 0));
  let offset = 0;
  for (const array of arrays) {
    output.set(array, offset);
    offset += array.length;
  }
  return output;
}
