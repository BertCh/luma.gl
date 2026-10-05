// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** CPU oracles for the RGB elevation decode. Test-only. */

export type RGBEncoding = 'terrarium' | 'mapbox';

const fround = Math.fround;
const FLOAT32_TENTH = fround(0.1);

/** Exact Terrarium height (float64 arithmetic is exact for every triple): R*256 + G + B/256 - 32768. */
export function decodeTerrariumFloat64(red: number, green: number, blue: number): number {
  return red * 256 + green + blue / 256 - 32768;
}

/** Mapbox terrain-RGB height in float64: -10000 + 0.1 * (R*65536 + G*256 + B). */
export function decodeMapboxFloat64(red: number, green: number, blue: number): number {
  return -10000 + 0.1 * (red * 65536 + green * 256 + blue);
}

/** The WGSL float32 operation order of the Terrarium kernel, with Math.fround. */
export function emulateTerrariumFloat32(red: number, green: number, blue: number): number {
  const code = red * 65536 + green * 256 + blue;
  return fround(fround(code * 0.00390625) - 32768);
}

/** The WGSL float32 operation order of the Mapbox kernel: one rounded multiply. */
export function emulateMapboxFloat32(red: number, green: number, blue: number): number {
  const code = red * 65536 + green * 256 + blue;
  return fround((code - 100000) * FLOAT32_TENTH);
}

/** The float32 value the kernel must produce for one triple. Terrarium uses the independent exact oracle. */
export function getExpectedFloat32(
  encoding: RGBEncoding,
  red: number,
  green: number,
  blue: number
): number {
  return encoding === 'terrarium'
    ? fround(decodeTerrariumFloat64(red, green, blue))
    : emulateMapboxFloat32(red, green, blue);
}

/** Float32 bit patterns of the expected value for every code `N = R*65536 + G*256 + B`. */
export function buildExpectedBits(encoding: RGBEncoding): Uint32Array {
  const values = new Float32Array(1 << 24);
  for (let code = 0; code < values.length; code++) {
    values[code] = getExpectedFloat32(encoding, code >> 16, (code >> 8) & 255, code & 255);
  }
  return new Uint32Array(values.buffer);
}

/** Spacing of float32 values at the magnitude of `value` (value must be finite). */
export function getFloat32Ulp(value: number): number {
  const magnitude = Math.abs(value);
  if (magnitude < 2 ** -126) {
    return 2 ** -149;
  }
  return 2 ** (Math.floor(Math.log2(magnitude)) - 23);
}

/** Returns the float32 ulp spacing at `value`, computed from the float32 bit pattern (exact). */
export function getExactFloat32Ulp(value: number): number {
  const bits = new Uint32Array(new Float32Array([Math.abs(value)]).buffer)[0];
  const exponent = (bits >>> 23) & 0xff;
  return exponent === 0 ? 2 ** -149 : 2 ** (exponent - 127 - 23);
}

/** Result of {@link measureMapboxRounding}. */
export type MapboxRoundingMeasurement = {
  /** Codes whose kernel value differs from `fround(M / 10)`. */
  differingCodeCount: number;
  /** Largest `|kernel - fround(M/10)|` in ulps of the kernel value. */
  maximumUlpDifference: number;
  /** Largest `|kernel - M/10| / (0.5 ulp + |M| * |fl(0.1) - 0.1|)`; must be at most 1. */
  maximumBoundRatio: number;
  /** Largest absolute error against `M / 10`, metres. */
  maximumErrorMetres: number;
  /** Largest absolute error against `M / 10` where `|h| <= 9000`, metres. */
  maximumErrorMetresWithin9000: number;
};

/** Exhaustively compares the Mapbox kernel against the exact decimal value `M / 10` for all 2^24 codes. */
export function measureMapboxRounding(): MapboxRoundingMeasurement {
  const tenthError = Math.abs(FLOAT32_TENTH - 0.1);
  let differingCodeCount = 0;
  let maximumUlpDifference = 0;
  let maximumBoundRatio = 0;
  let maximumErrorMetres = 0;
  let maximumErrorMetresWithin9000 = 0;
  for (let code = 0; code < 1 << 24; code++) {
    const scaled = code - 100000;
    const kernel = fround(scaled * FLOAT32_TENTH);
    const exact = scaled / 10;
    const correct = fround(exact);
    const ulp = getExactFloat32Ulp(kernel === 0 ? correct : kernel);
    if (kernel !== correct) {
      differingCodeCount++;
      maximumUlpDifference = Math.max(maximumUlpDifference, Math.abs(kernel - correct) / ulp);
    }
    const error = Math.abs(kernel - exact);
    const bound = 0.5 * ulp + Math.abs(scaled) * tenthError;
    if (bound > 0) {
      maximumBoundRatio = Math.max(maximumBoundRatio, error / bound);
    } else if (error > 0) {
      maximumBoundRatio = Infinity;
    }
    maximumErrorMetres = Math.max(maximumErrorMetres, error);
    if (Math.abs(exact) <= 9000) {
      maximumErrorMetresWithin9000 = Math.max(maximumErrorMetresWithin9000, error);
    }
  }
  return {
    differingCodeCount,
    maximumUlpDifference,
    maximumBoundRatio,
    maximumErrorMetres,
    maximumErrorMetresWithin9000
  };
}

/** Float32 bit pattern of NaN written for nodata. */
export const INVALID_FLOAT_BITS = 0x7fc00000;

/** Float32 to uint32 bits. */
export function getFloat32Bits(value: number): number {
  return new Uint32Array(new Float32Array([value]).buffer)[0];
}

/**
 * Small-grid CPU decode with the full nodata policy of the contributor, for the focused GPU tests.
 * Returns `{values, validity}` where nodata pixels hold NaN bits.
 */
export function decodeWithPolicy(
  words: Uint32Array,
  encoding: RGBEncoding,
  options: {
    alphaNoData?: boolean;
    noDataRGB?: readonly [number, number, number];
    validRange?: readonly [number, number];
    clampBathymetry?: boolean;
    inputValidity?: Uint32Array;
  } = {}
): {values: Uint32Array; validity: Uint32Array} {
  const [rangeMinimum, rangeMaximum] = options.validRange ?? [-11000, 9000];
  const values = new Uint32Array(words.length);
  const validity = new Uint32Array(words.length);
  for (let index = 0; index < words.length; index++) {
    const word = words[index];
    const red = word & 255;
    const green = (word >>> 8) & 255;
    const blue = (word >>> 16) & 255;
    const alpha = word >>> 24;
    const height = getExpectedFloat32(encoding, red, green, blue);
    let isValid = true;
    if ((options.alphaNoData ?? true) && alpha === 0) {
      isValid = false;
    }
    const noData = options.noDataRGB;
    if (noData && red === noData[0] && green === noData[1] && blue === noData[2]) {
      isValid = false;
    }
    if (options.inputValidity && options.inputValidity[index] === 0) {
      isValid = false;
    }
    if (height < fround(rangeMinimum) || height > fround(rangeMaximum)) {
      isValid = false;
    }
    validity[index] = isValid ? 1 : 0;
    const clamped = options.clampBathymetry && height < 0 && height > -12000 ? 0 : height;
    values[index] = isValid ? getFloat32Bits(clamped) : INVALID_FLOAT_BITS;
  }
  return {values, validity};
}
