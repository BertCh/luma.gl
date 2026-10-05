// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPURasterBand} from '../../gpu-raster/index';
import {getWGSLFloatLiteral, type MapGraphKernelBinding} from '../map-graph-kernels';
import {
  createRasterExtremaPyramidNodes,
  getGPURasterExtremaPyramidLayout,
  getRasterExtremaPyramidWGSL,
  type GPURasterExtremaPyramidLayout
} from '../raster-pyramid/index';
import {getTerrainElevationNodes} from '../terrain-analysis/terrain-analysis-utils';
import {
  GPU_POINT_HORIZON_EARTH_RADIUS,
  type GPUPointHorizonHeightReference,
  type GPUPointHorizonProjection,
  type GPUPointHorizonRowDirection,
  type GPUPointHorizonTraversal
} from './point-horizon-parameters';

/** Largest supported number of lattice samples per ray (keeps every lattice index exact in f32). */
const MAXIMUM_LATTICE_SAMPLES = 1 << 24;

/** Smallest lattice step exponent: steps are never finer than `2^-8` m. */
const MINIMUM_STEP_EXPONENT = -8;

/**
 * Options of the exact power-of-two distance lattice, shared by both point-horizon recipes.
 *
 * The lattice follows mt-image's adaptive step rule
 * `step(d) = max(stepFactor * d, clamp(nearFactor * d, minimumStep, cellSize * cellSteps))`
 * but quantizes it: inside octave `k` (`2^k <= d < 2^(k+1)`) the step is the largest power of two
 * not above the rule evaluated at `2^k`. Every lattice distance `2^k + j * step_k` is then exactly
 * representable in float32 (an integer times a power of two below `2^24` ulps added to a power of
 * two), so the CPU and every GPU compute bit-identical distances from `(octave, j)` with no
 * accumulated rounding, Kahan compensation or tables.
 */
export type GPUPointHorizonDistanceLatticeOptions = {
  /** First sample distance in meters (>= 1). Defaults to 20, which is exactly on the lattice. */
  minimumDistance?: number;
  /** Last sample distance in meters. */
  maximumDistance: number;
  /** Nominal ground meters per pixel, used only by the step rule. */
  cellSize: number;
  /** Relative step floor (distance fraction). Defaults to 3.5e-4. */
  stepFactor?: number;
  /** Near-field step as a distance fraction before clamping. Defaults to 0.01. */
  nearFactor?: number;
  /** Cell fraction capping the near-field step. Defaults to 0.5. */
  cellSteps?: number;
  /** Lower bound of the near-field step in meters. Defaults to 0.25. */
  minimumStep?: number;
};

/** One octave of a {@link GPUPointHorizonDistanceLattice}. */
export type GPUPointHorizonLatticeOctave = {
  /** `k` with `2^k <= d < 2^(k+1)`. */
  exponent: number;
  /** `2^k`. */
  base: number;
  /** Power-of-two step in meters, `<= base`. */
  step: number;
  /** First `j` used (nonzero only in the first octave). */
  firstJ: number;
  /** One past the last `j` used (less than `base / step` only in the last octave). */
  endJ: number;
  /** Global index of the sample `j = firstJ`. */
  firstIndex: number;
};

/** The sample distances of a ray, see {@link GPUPointHorizonDistanceLatticeOptions}. */
export type GPUPointHorizonDistanceLattice = {
  /** Number of lattice samples. */
  sampleCount: number;
  /** Consecutive octaves holding the samples. */
  octaves: readonly GPUPointHorizonLatticeOctave[];
  /** Requested `minimumDistance`. */
  minimumDistance: number;
  /** Requested `maximumDistance`. */
  maximumDistance: number;
  /** Distance of sample 0 (first lattice point `>= minimumDistance`). */
  firstDistance: number;
  /** Distance of the last sample (last lattice point `<= maximumDistance`). */
  lastDistance: number;
  /** Exact float32 distance of sample `index`. @throws RangeError outside `[0, sampleCount)`. */
  getDistance(index: number): number;
  /** Index of the first sample with distance `>= distance`, or `sampleCount` if none. */
  getCeilIndex(distance: number): number;
  /** Index of the last sample with distance `<= distance`, or `-1` if none. */
  getFloorIndex(distance: number): number;
};

/** Returns `floor(log2(value))` exactly for positive finite values. */
function getExponent(value: number): number {
  let exponent = Math.floor(Math.log2(value));
  while (2 ** (exponent + 1) <= value) {
    exponent++;
  }
  while (2 ** exponent > value) {
    exponent--;
  }
  return exponent;
}

/**
 * Builds the exact power-of-two distance lattice.
 *
 * @throws If a distance, cell size or factor is not finite and positive, the range is empty, or the
 * lattice would have more than 2^24 samples.
 */
export function getGPUPointHorizonDistanceLattice(
  options: GPUPointHorizonDistanceLatticeOptions
): GPUPointHorizonDistanceLattice {
  const minimumDistance = options.minimumDistance ?? 20;
  const {maximumDistance, cellSize} = options;
  const stepFactor = options.stepFactor ?? 3.5e-4;
  const nearFactor = options.nearFactor ?? 0.01;
  const cellSteps = options.cellSteps ?? 0.5;
  const minimumStep = options.minimumStep ?? 0.25;
  for (const [name, value] of [
    ['minimumDistance', minimumDistance],
    ['maximumDistance', maximumDistance],
    ['cellSize', cellSize],
    ['stepFactor', stepFactor],
    ['nearFactor', nearFactor],
    ['cellSteps', cellSteps],
    ['minimumStep', minimumStep]
  ] as const) {
    if (!(Number.isFinite(value) && value > 0)) {
      throw new Error(`Point horizon ${name} must be finite and positive`);
    }
  }
  if (minimumDistance < 1) {
    throw new Error('Point horizon minimumDistance must be at least 1 meter');
  }
  if (maximumDistance < minimumDistance) {
    throw new Error('Point horizon maximumDistance must not be below minimumDistance');
  }
  const firstExponent = getExponent(minimumDistance);
  const lastExponent = getExponent(maximumDistance);
  const octaves: GPUPointHorizonLatticeOctave[] = [];
  let sampleCount = 0;
  for (let exponent = firstExponent; exponent <= lastExponent; exponent++) {
    const base = 2 ** exponent;
    const want = Math.max(
      stepFactor * base,
      Math.min(Math.max(nearFactor * base, minimumStep), cellSize * cellSteps)
    );
    const stepExponent = Math.min(Math.max(getExponent(want), MINIMUM_STEP_EXPONENT), exponent);
    if (exponent - stepExponent > 22) {
      throw new Error('Point horizon distance lattice is too fine; increase stepFactor');
    }
    const step = 2 ** stepExponent;
    const fullEnd = base / step;
    const firstJ = exponent === firstExponent ? Math.ceil((minimumDistance - base) / step) : 0;
    const endJ =
      exponent === lastExponent ? Math.floor((maximumDistance - base) / step) + 1 : fullEnd;
    const clampedEnd = Math.min(endJ, fullEnd);
    if (clampedEnd <= firstJ) {
      continue;
    }
    octaves.push({exponent, base, step, firstJ, endJ: clampedEnd, firstIndex: sampleCount});
    sampleCount += clampedEnd - firstJ;
    if (sampleCount > MAXIMUM_LATTICE_SAMPLES) {
      throw new Error('Point horizon distance lattice has too many samples');
    }
  }
  if (sampleCount === 0) {
    throw new Error('Point horizon distance lattice is empty');
  }
  const getDistance = (index: number): number => {
    if (!Number.isInteger(index) || index < 0 || index >= sampleCount) {
      throw new RangeError(`Point horizon lattice index ${index} out of range`);
    }
    let octave = octaves[0];
    for (const candidate of octaves) {
      if (index >= candidate.firstIndex) {
        octave = candidate;
      }
    }
    return octave.base + (index - octave.firstIndex + octave.firstJ) * octave.step;
  };
  const getCeilIndex = (distance: number): number => {
    let low = 0;
    let high = sampleCount;
    while (low < high) {
      const middle = (low + high) >>> 1;
      if (getDistance(middle) >= distance) {
        high = middle;
      } else {
        low = middle + 1;
      }
    }
    return low;
  };
  const getFloorIndex = (distance: number): number => {
    const ceil = getCeilIndex(distance);
    return ceil < sampleCount && getDistance(ceil) === distance ? ceil : ceil - 1;
  };
  return {
    sampleCount,
    octaves,
    minimumDistance,
    maximumDistance,
    firstDistance: getDistance(0),
    lastDistance: getDistance(sampleCount - 1),
    getDistance,
    getCeilIndex,
    getFloorIndex
  };
}

/** Options of {@link getGPUPointHorizonSegments}. */
export type GPUPointHorizonSegmentOptions = {
  /** Largest `|latitude|` in degrees the window covers. Defaults to 80. */
  maximumLatitude?: number;
  /** Maximum chord sagitta divided by distance, radians. Defaults to 2e-5. */
  segmentTolerance?: number;
};

/** Piecewise great-circle breakpoints of a Web Mercator ray (float32-rounded, see below). */
export type GPUPointHorizonSegments = {
  /** Breakpoint distances in meters; segment `s` spans `distances[s]` to `distances[s + 1]`. */
  distances: readonly number[];
  /** `sin(D)` of every breakpoint, `D = distance / R`, rounded to float32 from float64. */
  sines: readonly number[];
  /** `1 - cos(D) = 2 sin^2(D / 2)` of every breakpoint, rounded to float32 from float64. */
  oneMinusCosines: readonly number[];
  /** First lattice index NOT in each segment (a sample belongs to the segment with `d < distances[s + 1]`). */
  endIndices: readonly number[];
};

/**
 * Breakpoints of the piecewise great-circle ray, a port of mt-image `marchSegments` with a single
 * ring. The chord length is `max(200, sqrt(8 * tolerance * d / kappa))` with
 * `kappa = max(0.05, tan(min(80, maximumLatitude + 3 degrees))) / R` bounding the Mercator
 * curvature of a great circle. Unlike mt-image, `kappa` uses the window's maximum latitude, not the
 * eye's, so the breakpoints are observer independent and baked into the shader as constants.
 */
export function getGPUPointHorizonSegments(
  lattice: GPUPointHorizonDistanceLattice,
  options: GPUPointHorizonSegmentOptions = {}
): GPUPointHorizonSegments {
  const maximumLatitude = options.maximumLatitude ?? 80;
  const tolerance = options.segmentTolerance ?? 2e-5;
  if (!(Number.isFinite(maximumLatitude) && maximumLatitude >= 0 && maximumLatitude <= 90)) {
    throw new Error('Point horizon maximumLatitude must be in [0, 90]');
  }
  if (!(Number.isFinite(tolerance) && tolerance > 0)) {
    throw new Error('Point horizon segmentTolerance must be finite and positive');
  }
  const radius = GPU_POINT_HORIZON_EARTH_RADIUS;
  const kappa =
    Math.max(0.05, Math.tan((Math.min(80, maximumLatitude + 3) * Math.PI) / 180)) / radius;
  const maximum = Math.fround(lattice.maximumDistance);
  const distances = [Math.fround(lattice.minimumDistance)];
  let distance = distances[0];
  while (distance < maximum) {
    const length = Math.max(200, Math.sqrt((8 * tolerance * distance) / kappa));
    distance = Math.min(Math.fround(distance + length), maximum);
    distances.push(distance);
  }
  if (distances.length === 1) {
    distances.push(distances[0] + 1);
  }
  const sines = distances.map(value => Math.fround(Math.sin(value / radius)));
  const oneMinusCosines = distances.map(value =>
    Math.fround(2 * Math.sin(value / (2 * radius)) ** 2)
  );
  const endIndices = distances.slice(1).map((value, segment) =>
    segment === distances.length - 2 ? lattice.sampleCount : lattice.getCeilIndex(value)
  );
  return {distances, sines, oneMinusCosines, endIndices};
}

/** Model options shared by `GPUPointHorizonProfile` and `GPUPointHorizonVisibility`. */
export type GPUPointHorizonModelOptions = {
  /** Raster projection model. Defaults to `'planar'`. */
  projection?: GPUPointHorizonProjection;
  /** Row direction of a planar raster. Defaults to `'south'`. */
  rowDirection?: GPUPointHorizonRowDirection;
  /** Eye height model. Defaults to `'ground'`. */
  heightReference?: GPUPointHorizonHeightReference;
  /** Sample traversal. Defaults to `'pyramid'`. */
  traversal?: GPUPointHorizonTraversal;
  /** Divisions of 360 degrees. Defaults to 720. */
  azimuthCount?: number;
  /** First global azimuth index of the covered sector. Defaults to 0. */
  firstAzimuth?: number;
  /** Number of azimuth indices covered, wrapping past `azimuthCount`. Defaults to `azimuthCount`. */
  azimuthSpan?: number;
  /** Last sample distance in meters (required). */
  maximumDistance: number;
  /** First sample distance in meters. Defaults to 20. */
  minimumDistance?: number;
  /** Nominal ground meters per pixel for the step rule (required). */
  cellSize: number;
  /** See {@link GPUPointHorizonDistanceLatticeOptions}. */
  stepFactor?: number;
  /** See {@link GPUPointHorizonDistanceLatticeOptions}. */
  nearFactor?: number;
  /** See {@link GPUPointHorizonDistanceLatticeOptions}. */
  cellSteps?: number;
  /** See {@link GPUPointHorizonDistanceLatticeOptions}. */
  minimumStep?: number;
  /** Web Mercator only, see {@link getGPUPointHorizonSegments}. Defaults to 80. */
  maximumLatitude?: number;
  /** Web Mercator only, see {@link getGPUPointHorizonSegments}. Defaults to 2e-5. */
  segmentTolerance?: number;
};

/** Validated, defaulted model. @internal */
export type ResolvedPointHorizonModel = {
  width: number;
  height: number;
  projection: GPUPointHorizonProjection;
  rowDirection: GPUPointHorizonRowDirection;
  heightReference: GPUPointHorizonHeightReference;
  traversal: GPUPointHorizonTraversal;
  azimuthCount: number;
  firstAzimuth: number;
  azimuthSpan: number;
  lattice: GPUPointHorizonDistanceLattice;
  segments: GPUPointHorizonSegments | null;
  /** Pyramid layout, present for the `'pyramid'` traversal. */
  layout: GPURasterExtremaPyramidLayout | null;
};

/** First block size of the internal pyramid. */
const PYRAMID_FIRST_BLOCK_SIZE = 4;

/**
 * Validates and defaults the model options.
 *
 * @internal
 */
export function resolvePointHorizonModel(
  id: string,
  width: number,
  height: number,
  options: GPUPointHorizonModelOptions
): ResolvedPointHorizonModel {
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width < 2 || height < 2) {
    throw new Error(`${id} dimensions must be integers of at least 2`);
  }
  if (width * height > 0xffffffff) {
    throw new Error(`${id} dimensions must be positive integers`);
  }
  const projection = options.projection ?? 'planar';
  if (projection !== 'planar' && projection !== 'web-mercator') {
    throw new Error(`${id} projection must be planar or web-mercator`);
  }
  const rowDirection = options.rowDirection ?? 'south';
  if (rowDirection !== 'south' && rowDirection !== 'north') {
    throw new Error(`${id} rowDirection must be south or north`);
  }
  const heightReference = options.heightReference ?? 'ground';
  if (heightReference !== 'ground' && heightReference !== 'absolute') {
    throw new Error(`${id} heightReference must be ground or absolute`);
  }
  const traversal = options.traversal ?? 'pyramid';
  if (traversal !== 'march' && traversal !== 'pyramid') {
    throw new Error(`${id} traversal must be march or pyramid`);
  }
  const azimuthCount = options.azimuthCount ?? 720;
  if (!Number.isInteger(azimuthCount) || azimuthCount < 1 || azimuthCount > MAXIMUM_LATTICE_SAMPLES) {
    throw new Error(`${id} azimuthCount must be an integer in [1, 2^24]`);
  }
  const firstAzimuth = options.firstAzimuth ?? 0;
  if (!Number.isInteger(firstAzimuth) || firstAzimuth < 0 || firstAzimuth >= azimuthCount) {
    throw new Error(`${id} firstAzimuth must be an integer in [0, azimuthCount)`);
  }
  const azimuthSpan = options.azimuthSpan ?? azimuthCount;
  if (!Number.isInteger(azimuthSpan) || azimuthSpan < 1 || azimuthSpan > azimuthCount) {
    throw new Error(`${id} azimuthSpan must be an integer in [1, azimuthCount]`);
  }
  let lattice: GPUPointHorizonDistanceLattice;
  try {
    lattice = getGPUPointHorizonDistanceLattice(options);
  } catch (error) {
    throw new Error(`${id} ${(error as Error).message}`);
  }
  return {
    width,
    height,
    projection,
    rowDirection,
    heightReference,
    traversal,
    azimuthCount,
    firstAzimuth,
    azimuthSpan,
    lattice,
    segments: projection === 'web-mercator' ? getGPUPointHorizonSegments(lattice, options) : null,
    layout:
      traversal === 'pyramid'
        ? getGPURasterExtremaPyramidLayout(width, height, {
            firstBlockSize: PYRAMID_FIRST_BLOCK_SIZE,
            footprint: 'bilinear'
          })
        : null
  };
}

/** Throws unless `view` is a packed float32 view of the given vector format. @internal */
export function validatePointHorizonRows(
  id: string,
  name: string,
  view: GraphDataView,
  format: 'float32' | 'float32x2' | 'float32x4'
): void {
  validatePackedView(view, [format], `${id} ${name}`);
}

const lit = getWGSLFloatLiteral;
const FLOAT_MAX = 3.4028234663852886e38;

function f32Array(values: readonly number[]): string {
  return `array<f32, ${values.length}>(${values.map(lit).join(', ')})`;
}

function u32Array(values: readonly number[]): string {
  return `array<u32, ${values.length}>(${values.map(value => `${value}u`).join(', ')})`;
}

/**
 * WGSL math shared by the point-horizon kernels: opaque guard, accurate series, azimuth sin/cos
 * from exact integer octant reduction, and the Web Mercator breakpoint offsets.
 *
 * Accuracy notes. WGSL `sin`/`cos` are only specified to 2^-11 absolute and `atan` to about 4096
 * ULP, so none of them is used where the result matters: `atanAccurate` reduces the argument to
 * `|z| <= tan(pi/12)` and sums an odd series; azimuth sin/cos reduce the integer index exactly into
 * one octant; `bp` forms every Web Mercator offset from cancellation-free difference terms and
 * small-argument series (mt-image `gpu/horizon/horizon.wgsl.ts`).
 */
const PRECISION_WGSL = /* wgsl */ `
const PI: f32 = 3.14159265358979323846;
const HALF_PI: f32 = 1.57079632679489661923;
const QUARTER_PI: f32 = 0.78539816339744830962;
const DEGREES: f32 = 57.29577951308232;
const INV_TWO_PI: f32 = 0.15915494309189535;
const TWO_PI: f32 = 6.28318530717958647692;
const BIG: f32 = ${lit(FLOAT_MAX)};
const EARTH_RADIUS: f32 = ${GPU_POINT_HORIZON_EARTH_RADIUS};

// XOR with the always-zero settings word: the compiler cannot fold it, so Metal fast math can
// neither re-associate (v - hi) - lo nor fuse a product into the following add.
fn opaque(x: f32) -> f32 {
  return bitcast<f32>(bitcast<u32>(x) ^ bitcast<u32>(settings[settingsOffset + 4u]));
}

fn isFiniteValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) < 0x7f800000u; }
fn nanValue() -> f32 { return bitcast<f32>(0x7fc00000u); }

// Odd Taylor series of atan for |z| <= tan(pi/12) = 0.2679.
fn atanSeries(z: f32) -> f32 {
  let z2 = z * z;
  return z * (1.0 + z2 * (-1.0 / 3.0 + z2 * (1.0 / 5.0 + z2 * (-1.0 / 7.0 + z2 * (1.0 / 9.0 +
    z2 * (-1.0 / 11.0 + z2 * (1.0 / 13.0 + z2 * (-1.0 / 15.0))))))));
}

// atan on [0, 1]: pi/6 + atan((z sqrt(3) - 1) / (z + sqrt(3))) above tan(pi/12).
fn atanUnit(a: f32) -> f32 {
  if (a > 0.2679491924311227) {
    return 0.5235987755982988 + atanSeries((a * 1.7320508075688772 - 1.0) / (a + 1.7320508075688772));
  }
  return atanSeries(a);
}

// atan to about 2 ULP plus one division, without the builtin.
fn atanAccurate(z: f32) -> f32 {
  let a = abs(z);
  var r: f32;
  if (a > 1.0) {
    r = HALF_PI - atanUnit(1.0 / a);
  } else {
    r = atanUnit(a);
  }
  return select(-r, r, z >= 0.0);
}

// atan2(y, x) for x > 0 with the series branch of mt-image's atanS, builtin for x <= 0.
fn atanS(y: f32, x: f32) -> f32 {
  if (x > 0.0) {
    return atanAccurate(y / x);
  }
  return atan2(y, x);
}

// atanh(z) for |z| < 0.2 (odd series to z^13), log form otherwise.
fn atanhS(z: f32) -> f32 {
  if (abs(z) < 0.2) {
    let z2 = z * z;
    return z * (1.0 + z2 * (1.0 / 3.0 + z2 * (1.0 / 5.0 + z2 * (1.0 / 7.0 + z2 * (1.0 / 9.0 +
      z2 * (1.0 / 11.0 + z2 * (1.0 / 13.0)))))));
  }
  return 0.5 * log((1.0 + z) / (1.0 - z));
}

// sin and cos series for |x| <= pi / 2 (error below 1e-9).
fn sinSeries(x: f32) -> f32 {
  let x2 = x * x;
  return x * (1.0 + x2 * (-1.0 / 6.0 + x2 * (1.0 / 120.0 + x2 * (-1.0 / 5040.0 + x2 * (1.0 / 362880.0 +
    x2 * (-1.0 / 39916800.0 + x2 * (1.0 / 6227020800.0 + x2 * (-1.0 / 1307674368000.0))))))));
}
fn cosSeries(x: f32) -> f32 {
  let x2 = x * x;
  return 1.0 + x2 * (-1.0 / 2.0 + x2 * (1.0 / 24.0 + x2 * (-1.0 / 720.0 + x2 * (1.0 / 40320.0 +
    x2 * (-1.0 / 3628800.0 + x2 * (1.0 / 479001600.0 + x2 * (-1.0 / 87178291200.0)))))));
}

// Azimuth index i of N (angle i * 360 / N degrees clockwise from north) to (sin, cos) with no
// builtin trigonometry: 8 i = octant * N + r exactly in integers, the octant remainder is folded
// into [0, pi/4], series evaluate it, and a quadrant rotation finishes. (sin, cos) = (east, north).
fn azimuthSinCos(index: u32, count: u32) -> vec2<f32> {
  let eight = 8u * (index % count);
  let octant = eight / count;
  let remainder = eight - octant * count;
  let odd = (octant & 1u) == 1u;
  let folded = select(remainder, count - remainder, odd);
  let angle = (f32(folded) / f32(count)) * QUARTER_PI;
  let s0 = sinSeries(angle);
  let c0 = cosSeries(angle);
  let s = select(s0, c0, odd);
  let c = select(c0, s0, odd);
  let quadrant = octant >> 1u;
  if (quadrant == 0u) { return vec2<f32>(s, c); }
  if (quadrant == 1u) { return vec2<f32>(c, -s); }
  if (quadrant == 2u) { return vec2<f32>(-s, -c); }
  return vec2<f32>(-c, s);
}

// Normalised Mercator offset (dx, dy) of the great-circle point at angular distance D (sinD, omc =
// 1 - cos D) along azimuth (sinA, cosA) from the eye (sinP1, cosP1, c2 = cosP1^2).
fn bp(sinD: f32, omc: f32, sinA: f32, cosA: f32, sinP1: f32, cosP1: f32, c2: f32) -> vec2<f32> {
  let ds = cosP1 * sinD * cosA - sinP1 * omc; // sin(phi2) - sin(phi1)
  let den = c2 - sinP1 * ds; // 1 - sin(phi1) sin(phi2)
  let dl = atanS(sinA * sinD * cosP1, den - omc); // cos D - sin(phi1) sin(phi2)
  let dy = atanhS(ds / den); // atanh(sin phi2) - atanh(sin phi1)
  return vec2<f32>(dl * INV_TWO_PI, -dy * INV_TWO_PI);
}

// sinh with a series below 0.5 (the exp form cancels there).
fn sinhS(x: f32) -> f32 {
  if (abs(x) < 0.5) {
    let x2 = x * x;
    return x * (1.0 + x2 * (1.0 / 6.0 + x2 * (1.0 / 120.0 + x2 * (1.0 / 5040.0 + x2 * (1.0 / 362880.0 +
      x2 * (1.0 / 39916800.0 + x2 * (1.0 / 6227020800.0)))))));
  }
  return 0.5 * (exp(x) - exp(-x));
}
fn coshS(x: f32) -> f32 {
  let e = exp(abs(x));
  return 0.5 * (e + 1.0 / e);
}
// tanh with a series below 0.3, the exp form above.
fn tanhS(x: f32) -> f32 {
  let a = abs(x);
  var r: f32;
  if (a < 0.3) {
    let x2 = x * x;
    r = a * (1.0 + x2 * (-1.0 / 3.0 + x2 * (2.0 / 15.0 + x2 * (-17.0 / 315.0 + x2 * (62.0 / 2835.0 +
      x2 * (-1382.0 / 155925.0))))));
  } else {
    let e = exp(-2.0 * a);
    r = (1.0 - e) / (1.0 + e);
  }
  return select(-r, r, x >= 0.0);
}
// asin for z < 0.2 by series, builtin above.
fn asinS(z: f32) -> f32 {
  if (z < 0.2) {
    let z2 = z * z;
    return z * (1.0 + z2 * (1.0 / 6.0 + z2 * (3.0 / 40.0 + z2 * (5.0 / 112.0 + z2 * (35.0 / 1152.0 +
      z2 * (63.0 / 2816.0 + z2 * (231.0 / 13312.0)))))));
  }
  return asin(min(z, 1.0));
}
`;

/**
 * Generates the WGSL march shared by the profile and visibility kernels: lattice helpers, eye and
 * position helpers, the exact bilinear sample, `marchRay`, and (for `'pyramid'`) the exact skip.
 *
 * The caller's kernel must bind `elevationValues` (f32), `elevationValidity` (u32), `settings`
 * (f32, slots `[curvature, cellX | worldPixelSize, cellY | mercatorTop, maximumDistance, ZERO]`)
 * and, for the pyramid, `pyramid` (f32, combined maxima then minima), using the usual
 * `${name}Offset` constants.
 *
 * ## Why the pyramid traversal equals the march bit for bit
 *
 * Both traversals walk the same lattice samples `n = 0, 1, ...` and evaluate each with the same
 * WGSL expressions (`getPosition`, `bilinearRelative`, the update `!has || t > tBest`); the pyramid
 * only omits samples that provably cannot satisfy `t > tBest`, and tBest/dBest change only through
 * evaluated samples, so the surviving sequence of updates is identical.
 *
 * Skipping samples `i..j` at pyramid level `L` is allowed when all of the following hold:
 *
 * 1. Membership. Every sample `n` in `i..j` has its base pixel (`floor` of the position) inside the
 *    `S x S` block of the cell and inside the bilinear grid. Each axis of the position
 *    `uA + opaque((d - dA) * du)` is a monotone function of `n`: `d` is monotone in `n`
 *    (exact lattice), subtraction of a constant, multiplication by a constant and addition of a
 *    constant are monotone under round-to-nearest, and `opaque` only forbids fusion, it does not
 *    change a value. `floor` and the integer offset are monotone too. A box condition on two monotone
 *    coordinates holds for every `n` between two members, so testing the endpoints `i` and `j`,
 *    recomputed with the identical `getPosition`, is exact. `j` is derived from the analytic exit of
 *    the ray from the block, then clamped to the last sample of the segment (positions are only
 *    monotone inside one linear segment) and of the ray, then verified and decremented (at most 3
 *    times) until the endpoint test passes, so rounding in the analytic exit cannot matter.
 * 2. Value bound. A bilinear sample whose base pixel lies in the block reads only pixels of the
 *    cell's bilinear footprint, so with all four corners valid the interpolated height is within
 *    `[Hmin, Hmax]` (invalid-corner samples are ignored by both traversals; an empty cell is all
 *    invalid and always skippable). The relative height `hr` is therefore at most
 *    `Nup = (Hmax - eyeHi) - eyeLo + EPS * M`, with `M` the largest magnitude involved and
 *    `EPS = 2^-17` (over 100 ULP) absorbing every rounding of the corner subtractions and the
 *    interpolation, and fused multiply-add or reassociation under fast math. For distances in
 *    `[d_i, d_j]`, `Nup / d` is at most `Nup / d_i` when `Nup >= 0` and at most `Nup / d_j` when
 *    `Nup < 0`, and `-d * c` is at most `-min(d_i c, d_j c)`; widening by `EPS` again covers the
 *    division and the curvature product. If the resulting `Up <= tBest`, then every skipped sample
 *    has `t <= Up <= tBest` and could not have updated the running maximum.
 *
 * The `q` record of the visibility kernel (`tQ`) is the running maximum when the first sample with
 * `d >= q` is reached; skipped samples never change the running maximum, so it is unchanged.
 *
 * @internal
 */
export function getPointHorizonMarchWGSL(model: ResolvedPointHorizonModel): string {
  const {lattice, segments, layout} = model;
  const mercator = model.projection === 'web-mercator';
  const usePyramid = model.traversal === 'pyramid';
  const octaves = lattice.octaves;
  const segmentCount = segments ? segments.distances.length - 1 : 1;
  const segmentEnds = segments ? segments.endIndices : [lattice.sampleCount];
  const eyeHeight =
    model.heightReference === 'ground'
      ? `let ground = sampleGround(column, row);
  if (ground.y == 0.0 || !isFiniteValue(height)) {
    eye.valid = false;
    return eye;
  }
  eye.hi = ground.x;
  eye.lo = height;`
      : `if (!isFiniteValue(height)) {
    eye.valid = false;
    return eye;
  }
  eye.hi = height;
  eye.lo = 0.0;`;
  const eyeMercator = mercator
    ? `let m1 = settings[settingsOffset + 2u] - TWO_PI * (row + 0.5) / settings[settingsOffset + 1u];
  eye.sinP = tanhS(m1);
  eye.cosP = 1.0 / coshS(m1);
  eye.m = m1;`
    : '';
  const raySetup = mercator
    ? `let worldPixelSize = settings[settingsOffset + 1u];
  let c2 = eye.cosP * eye.cosP;
  var b0 = bp(SEGMENT_SINES[0], SEGMENT_OMC[0], sinA, cosA, eye.sinP, eye.cosP, c2);`
    : `let dirU = sinA / settings[settingsOffset + 1u];
  let dirV = ${model.rowDirection === 'south' ? '-' : ''}cosA / settings[settingsOffset + 2u];`;
  const segmentGeometry = mercator
    ? `let b1 = bp(SEGMENT_SINES[segment + 1u], SEGMENT_OMC[segment + 1u], sinA, cosA, eye.sinP, eye.cosP, c2);
    let dA = SEGMENT_DISTANCES[segment];
    let invLength = 1.0 / (SEGMENT_DISTANCES[segment + 1u] - dA);
    let uA = eye.fx + opaque(b0.x * worldPixelSize);
    let vA = eye.fy + opaque(b0.y * worldPixelSize);
    let du = opaque((b1.x - b0.x) * worldPixelSize) * invLength;
    let dv = opaque((b1.y - b0.y) * worldPixelSize) * invLength;`
    : `let dA = 0.0;
    let uA = eye.fx;
    let vA = eye.fy;
    let du = dirU;
    let dv = dirV;`;
  const segmentEnd = mercator ? 'b0 = b1;' : '';
  const skip = usePyramid
    ? `if (n >= noTest) {
        var skipTo = n;
        var skipped = false;
        for (var level = 0u; level < PYRAMID_LEVEL_COUNT; level++) {
          let cellIndex = pyramidLevelIndex(level, u32(p.xi), u32(p.yi));
          let hMax = pyramid[pyramidOffset + cellIndex];
          let empty = hMax <= PYRAMID_EMPTY_MAXIMUM;
          if (!empty && !has) { break; }
          let blockSize = i32(pyramidLevelBlockSize(level));
          let blockX = (p.xi / blockSize) * blockSize;
          let blockY = (p.yi / blockSize) * blockSize;
          // Analytic exit of the ray from the block (integer part exact), then clamp and verify.
          var exitX = BIG;
          if (du > 0.0) { exitX = (f32(blockX + blockSize - p.xi) - p.fx) / du; }
          else if (du < 0.0) { exitX = (f32(blockX - p.xi) - p.fx) / du; }
          var exitY = BIG;
          if (dv > 0.0) { exitY = (f32(blockY + blockSize - p.yi) - p.fy) / dv; }
          else if (dv < 0.0) { exitY = (f32(blockY - p.yi) - p.fy) / dv; }
          var candidate = latFloorIndex(d + min(exitX, exitY));
          candidate = min(candidate, min(i32(segmentEnd) - 1, rayLast));
          var verified = false;
          var dEnd = d;
          for (var attempt = 0u; attempt < 4u; attempt++) {
            if (candidate < i32(n)) { break; }
            let candidateOctave = latOctave(u32(candidate));
            let candidateDistance = latDistance(
              candidateOctave, u32(candidate) - LAT_FIRST[candidateOctave] + LAT_J0[candidateOctave]);
            let q2 = getPosition(candidateDistance, dA, uA, vA, du, dv, eye);
            if (inGrid(q2) && q2.xi >= blockX && q2.xi < blockX + blockSize &&
                q2.yi >= blockY && q2.yi < blockY + blockSize) {
              verified = true;
              dEnd = candidateDistance;
              break;
            }
            candidate = candidate - 1;
          }
          if (!verified) { break; }
          var skippable = empty;
          if (!empty) {
            let hMin = pyramid[pyramidOffset + PYRAMID_MINIMUM_OFFSET + cellIndex];
            let magnitude = max(abs(hMax - eye.hi), abs(hMin - eye.hi)) + abs(eye.lo);
            let numerator = (hMax - eye.hi) - eye.lo + SKIP_EPSILON * magnitude;
            let ratio = select((numerator / dEnd) * (1.0 - SKIP_EPSILON),
              (numerator / d) * (1.0 + SKIP_EPSILON), numerator >= 0.0);
            let drop = min(d * curvature, dEnd * curvature);
            let upper = ratio - drop + SKIP_EPSILON * abs(drop);
            skippable = upper <= tBest;
          }
          if (skippable) {
            skipped = true;
            skipTo = max(skipTo, u32(candidate) + 1u);
          } else {
            if (level == 0u) { noTest = u32(candidate) + 1u; }
            break;
          }
        }
        if (skipped) {
          n = skipTo;
          if (n < LAT_COUNT) {
            octave = latOctave(n);
            j = n - LAT_FIRST[octave] + LAT_J0[octave];
          }
          continue;
        }
      }`
    : '';
  return /* wgsl */ `
const WIDTH: u32 = ${model.width}u;
const HEIGHT: u32 = ${model.height}u;
const WIDTH_I: i32 = ${model.width};
const HEIGHT_I: i32 = ${model.height};
const AZIMUTH_COUNT: u32 = ${model.azimuthCount}u;
const FIRST_AZIMUTH: u32 = ${model.firstAzimuth}u;
const AZIMUTH_SPAN: u32 = ${model.azimuthSpan}u;
const SKIP_EPSILON: f32 = 7.62939453125e-6;
${PRECISION_WGSL}
// Exact power-of-two distance lattice: octave k holds 2^k + j * step_k, all exact in f32.
const LAT_OCTAVE_COUNT: u32 = ${octaves.length}u;
const LAT_K0: i32 = ${octaves[0].exponent};
const LAT_COUNT: u32 = ${lattice.sampleCount}u;
const LAT_FIRST_DISTANCE: f32 = ${lit(lattice.firstDistance)};
const LAT_LAST_DISTANCE: f32 = ${lit(lattice.lastDistance)};
const MAXIMUM_DISTANCE: f32 = ${lit(lattice.maximumDistance)};
const LAT_BASE = ${f32Array(octaves.map(octave => octave.base))};
const LAT_STEP = ${f32Array(octaves.map(octave => octave.step))};
const LAT_INV_STEP = ${f32Array(octaves.map(octave => 1 / octave.step))};
const LAT_J0 = ${u32Array(octaves.map(octave => octave.firstJ))};
const LAT_J1 = ${u32Array(octaves.map(octave => octave.endJ))};
const LAT_FIRST = ${u32Array(octaves.map(octave => octave.firstIndex))};
const SEGMENT_COUNT: u32 = ${segmentCount}u;
const SEGMENT_END = ${u32Array(segmentEnds)};
${
  segments
    ? `const SEGMENT_DISTANCES = ${f32Array(segments.distances)};
const SEGMENT_SINES = ${f32Array(segments.sines)};
const SEGMENT_OMC = ${f32Array(segments.oneMinusCosines)};`
    : ''
}
${layout ? getRasterExtremaPyramidWGSL(layout, 'pyramid') : ''}

fn latDistance(octave: u32, j: u32) -> f32 {
  return LAT_BASE[octave] + f32(j) * LAT_STEP[octave];
}

fn latOctave(index: u32) -> u32 {
  var octave = 0u;
  for (var i = 1u; i < LAT_OCTAVE_COUNT; i++) {
    if (index >= LAT_FIRST[i]) { octave = i; }
  }
  return octave;
}

// Index of the first lattice sample with distance >= x (LAT_COUNT if none). The octave comes from
// the f32 exponent of x; x - 2^k and the division by a power of two are exact.
fn latCeilIndex(x: f32) -> u32 {
  if (!(x > LAT_FIRST_DISTANCE)) { return 0u; }
  if (x > LAT_LAST_DISTANCE) { return LAT_COUNT; }
  let exponent = i32((bitcast<u32>(x) >> 23u) & 0xffu) - 127;
  let octave = u32(exponent - LAT_K0);
  let steps = ceil((x - LAT_BASE[octave]) * LAT_INV_STEP[octave]);
  let j = min(max(u32(steps), LAT_J0[octave]), LAT_J1[octave]);
  return LAT_FIRST[octave] + j - LAT_J0[octave];
}

// Index of the last lattice sample with distance <= x (-1 if none).
fn latFloorIndex(x: f32) -> i32 {
  if (!(x >= LAT_FIRST_DISTANCE)) { return -1; }
  if (x >= LAT_LAST_DISTANCE) { return i32(LAT_COUNT) - 1; }
  let exponent = i32((bitcast<u32>(x) >> 23u) & 0xffu) - 127;
  let octave = u32(exponent - LAT_K0);
  let steps = floor((x - LAT_BASE[octave]) * LAT_INV_STEP[octave]);
  let j = min(u32(steps), LAT_J1[octave] - 1u);
  return i32(LAT_FIRST[octave] + j - LAT_J0[octave]);
}

fn readCellValid(cell: u32) -> bool {
  return elevationValidity[elevationValidityOffset + cell] != 0u &&
    elevationValidity[elevationValidityOffset + cell + 1u] != 0u &&
    elevationValidity[elevationValidityOffset + cell + WIDTH] != 0u &&
    elevationValidity[elevationValidityOffset + cell + WIDTH + 1u] != 0u;
}

// Bilinear height relative to the eye. Each corner is (v - hi) - lo with the subtraction pinned by
// opaque(): near the camera an f32 height formed after interpolation would carry the rounding of a
// 4-digit height, about 1e-3 degrees at 2 m. Products are pinned too so the result does not depend
// on the compiler's fused multiply-add choices (the pyramid bound relies on a stable value).
fn bilinearRelative(cell: u32, fx: f32, fy: f32, hi: f32, lo: f32) -> f32 {
  let a0 = opaque(elevationValues[elevationValuesOffset + cell] - hi) - lo;
  let a1 = opaque(elevationValues[elevationValuesOffset + cell + 1u] - hi) - lo;
  let c0 = opaque(elevationValues[elevationValuesOffset + cell + WIDTH] - hi) - lo;
  let c1 = opaque(elevationValues[elevationValuesOffset + cell + WIDTH + 1u] - hi) - lo;
  let t1 = opaque(a0 - a1);
  let t2 = opaque(t1 - c0);
  let twist = t2 + c1;
  let rowSlope = (c0 - a0) + opaque(twist * fx);
  return (a0 + opaque((a1 - a0) * fx)) + opaque(rowSlope * fy);
}

// Bilinear height relative to (hi, lo) at a pixel-center position inside [0, W-1] x [0, H-1];
// .y is 0 when a corner is invalid.
fn sampleRelative(column: f32, row: f32, hi: f32, lo: f32) -> vec2<f32> {
  let x0 = min(i32(floor(column)), WIDTH_I - 2);
  let y0 = min(i32(floor(row)), HEIGHT_I - 2);
  let cell = u32(y0) * WIDTH + u32(x0);
  if (!readCellValid(cell)) { return vec2<f32>(0.0, 0.0); }
  return vec2<f32>(bilinearRelative(cell, column - f32(x0), row - f32(y0), hi, lo), 1.0);
}

fn sampleGround(column: f32, row: f32) -> vec2<f32> {
  return sampleRelative(column, row, 0.0, 0.0);
}

fn isInsideGrid(column: f32, row: f32) -> bool {
  return column >= 0.0 && row >= 0.0 && column <= f32(WIDTH - 1u) && row <= f32(HEIGHT - 1u);
}

struct EyeState {
  valid: bool,
  column: i32,
  row: i32,
  fx: f32,
  fy: f32,
  hi: f32,
  lo: f32,
  sinP: f32,
  cosP: f32,
  m: f32,
};

// Eye at (column, row) in pixel-center index space: integer pixel plus f32 fraction, and the eye
// height as an f32 (hi, lo) pair so (value - hi) - lo is exact to f32 near the eye.
fn makeEye(column: f32, row: f32, height: f32) -> EyeState {
  var eye: EyeState;
  eye.valid = isInsideGrid(column, row);
  if (!eye.valid) { return eye; }
  let floorColumn = floor(column);
  let floorRow = floor(row);
  eye.column = i32(floorColumn);
  eye.row = i32(floorRow);
  eye.fx = column - floorColumn;
  eye.fy = row - floorRow;
  ${eyeHeight}
  ${eyeMercator}
  return eye;
}

struct Position { xi: i32, yi: i32, fx: f32, fy: f32 };

// Position of the sample at distance d inside the segment starting at dA, relative to the eye's
// integer pixel; the same code serves the march, the skip verification and both kernels.
fn getPosition(d: f32, dA: f32, uA: f32, vA: f32, du: f32, dv: f32, eye: EyeState) -> Position {
  let f = d - dA;
  let uf = uA + opaque(f * du);
  let vf = vA + opaque(f * dv);
  let floorU = floor(uf);
  let floorV = floor(vf);
  return Position(eye.column + i32(floorU), eye.row + i32(floorV), uf - floorU, vf - floorV);
}

// Bilinear footprint of a sample must be inside the grid, otherwise the ray ends.
fn inGrid(p: Position) -> bool {
  return p.xi >= 0 && p.yi >= 0 && p.xi < WIDTH_I - 1 && p.yi < HEIGHT_I - 1;
}

struct RayResult { has: bool, t: f32, d: f32, tQ: f32 };

// Marches one ray. Returns the maximum tangent tBest = hr / d - d c over valid samples and its
// distance. tQ is the running maximum when the first sample with d >= q is reached (q <= 0 or
// !useQ: -BIG).
fn marchRay(eye: EyeState, sinA: f32, cosA: f32, q: f32, useQ: bool) -> RayResult {
  let curvature = settings[settingsOffset];
  let maximumFrame = settings[settingsOffset + 3u];
  let maximumDistance = select(LAT_LAST_DISTANCE, min(maximumFrame, LAT_LAST_DISTANCE), maximumFrame > 0.0);
  let rayLast = latFloorIndex(maximumDistance);
  ${raySetup}
  var has = false;
  var tBest = 0.0;
  var dBest = 0.0;
  var qDone = !useQ;
  var tQ = -BIG;
  var n = 0u;
  var octave = 0u;
  var j = LAT_J0[0];
  var ended = false;
  var noTest = 0u;
  for (var segment = 0u; segment < SEGMENT_COUNT; segment++) {
    if (ended || i32(n) > rayLast) { break; }
    let segmentEnd = SEGMENT_END[segment];
    ${segmentGeometry}
    loop {
      if (n >= segmentEnd || i32(n) > rayLast) { break; }
      let d = latDistance(octave, j);
      if (!qDone && d >= q) {
        tQ = select(-BIG, tBest, has);
        qDone = true;
      }
      let p = getPosition(d, dA, uA, vA, du, dv, eye);
      if (!inGrid(p)) {
        ended = true;
        break;
      }
      ${skip}
      let cell = u32(p.yi) * WIDTH + u32(p.xi);
      if (readCellValid(cell)) {
        let hr = bilinearRelative(cell, p.fx, p.fy, eye.hi, eye.lo);
        let t = hr / d - opaque(d * curvature);
        if (!has || t > tBest) {
          has = true;
          tBest = t;
          dBest = d;
        }
      }
      n = n + 1u;
      j = j + 1u;
      if (j >= LAT_J1[octave]) {
        octave = octave + 1u;
        j = 0u;
      }
    }
    ${segmentEnd}
  }
  if (!qDone) { tQ = select(-BIG, tBest, has); }
  return RayResult(has, tBest, dBest, tQ);
}
`;
}

/** Canonical elevation, optional combined pyramid, their nodes and kernel bindings. @internal */
export type PointHorizonSource<Parameters> = {
  nodes: GPUCommandNode<Parameters>[];
  /** Read-only bindings `elevationValues`, `elevationValidity` and, for `'pyramid'`, `pyramid`. */
  bindings: MapGraphKernelBinding[];
};

/**
 * Creates the canonicalization nodes, and for the `'pyramid'` traversal the min-max pyramid nodes
 * writing ONE transient combined view, plus the matching kernel bindings.
 *
 * @internal
 */
export function createPointHorizonSource<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  model: ResolvedPointHorizonModel,
  terrain: GPURasterBand
): PointHorizonSource<Parameters> {
  const source = getTerrainElevationNodes(graph, id, terrain, model.width, model.height, true);
  const nodes: GPUCommandNode<Parameters>[] = [...source.nodes];
  const values = source.band.storage.values as GraphDataView<'float32'>;
  const validity = source.band.validity as GraphDataView<'uint32'>;
  const bindings: MapGraphKernelBinding[] = [
    {name: 'elevationValues', view: values, type: 'f32', access: 'read'},
    {name: 'elevationValidity', view: validity, type: 'u32', access: 'read'}
  ];
  if (model.layout) {
    const combined = createTransientView(graph, `${id}-pyramid`, 'float32', 2 * model.layout.length);
    nodes.push(
      ...createRasterExtremaPyramidNodes(graph, {
        id: `${id}-pyramid`,
        layout: model.layout,
        values,
        validity,
        combined
      })
    );
    bindings.push({name: 'pyramid', view: combined, type: 'f32', access: 'read'});
  }
  return {nodes, bindings};
}
