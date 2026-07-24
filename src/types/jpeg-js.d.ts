declare module 'jpeg-js' {
  export type RawImageData = {
    width: number;
    height: number;
    data: Uint8Array;
  };

  export function decode(buffer: Uint8Array, options?: { useTArray?: boolean }): RawImageData;
  export function encode(image: RawImageData, quality?: number): { data: Uint8Array; width: number; height: number };
}
