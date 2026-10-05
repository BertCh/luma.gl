// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUPointHorizonDistanceLattice,
  GPUPointHorizonSegments
} from '../../../src/gpu-terrain/point-horizon';

const EARTH_RADIUS = 6371008.8;

/** Model of the float64 point-horizon oracle, mirroring the GPU topology and settings. */
export type PointHorizonOracleModel = {
  width: number;
  height: number;
  values: ArrayLike<number>;
  validity?: ArrayLike<number>;
  projection: 'planar' | 'web-mercator';
  rowDirection?: 'south' | 'north';
  heightReference?: 'ground' | 'absolute';
  azimuthCount: number;
  firstAzimuth: number;
  azimuthSpan: number;
  lattice: GPUPointHorizonDistanceLattice;
  segments?: GPUPointHorizonSegments;
  curvature: number;
  /** Per-frame ray cap; `<= 0` means the lattice maximum. */
  maximumDistance?: number;
  cellSize?: readonly [number, number];
  worldPixelSize?: number;
  originY?: number;
};

/** Observer `[column, row, height]`. */
export type PointHorizonOracleObserver = readonly [number, number, number];

/** Float64 oracle result, one entry per `observer * azimuthSpan + a`. */
export type PointHorizonOracleResult = {
  tangent: Float64Array;
  /** Degrees; -90 when no sample, NaN for an invalid observer. */
  elevation: Float64Array;
  distance: Float64Array;
};

function isValidCell(model: PointHorizonOracleModel, x: number, y: number): boolean {
  const {width, validity, values} = model;
  for (const index of [
    y * width + x,
    y * width + x + 1,
    (y + 1) * width + x,
    (y + 1) * width + x + 1
  ]) {
    if (validity ? validity[index] === 0 : false) {
      return false;
    }
    if (!Number.isFinite(values[index])) {
      return false;
    }
  }
  return true;
}

/** Bilinear height of a valid cell at the base pixel and fractions, in float64. */
function bilinear(model: PointHorizonOracleModel, x: number, y: number, fx: number, fy: number) {
  const {width, values} = model;
  const a0 = values[y * width + x];
  const a1 = values[y * width + x + 1];
  const c0 = values[(y + 1) * width + x];
  const c1 = values[(y + 1) * width + x + 1];
  return a0 + (a1 - a0) * fx + (c0 - a0 + (a0 - a1 - c0 + c1) * fx) * fy;
}

/** Bilinear sample at a pixel-center position, clamped to the last cell; `undefined` if invalid. */
export function sampleOracleHeight(
  model: PointHorizonOracleModel,
  column: number,
  row: number
): number | undefined {
  const x = Math.min(Math.floor(column), model.width - 2);
  const y = Math.min(Math.floor(row), model.height - 2);
  if (!isValidCell(model, x, y)) {
    return undefined;
  }
  return bilinear(model, x, y, column - x, row - y);
}

/** Maximum-tangent march of one ray in float64, no skipping. */
export function marchOracleRay(
  model: PointHorizonOracleModel,
  eyeColumn: number,
  eyeRow: number,
  eyeElevation: number,
  sinA: number,
  cosA: number,
  stop = 0
): {has: boolean; t: number; d: number; tQ: number} {
  const {lattice, segments, width, height, curvature} = model;
  const frameMaximum = model.maximumDistance ?? 0;
  const maximum =
    frameMaximum > 0 ? Math.min(frameMaximum, lattice.lastDistance) : lattice.lastDistance;
  const mercator = model.projection === 'web-mercator';
  // Segment offsets in pixels at each breakpoint (mercator) or one straight segment (planar).
  let offsets: {d: number; x: number; y: number}[] = [];
  if (mercator) {
    const worldPixelSize = model.worldPixelSize as number;
    const m1 = Math.PI * (1 - (2 * ((model.originY as number) + eyeRow + 0.5)) / worldPixelSize);
    const sinP1 = Math.tanh(m1);
    const cosP1 = 1 / Math.cosh(m1);
    offsets = (segments as GPUPointHorizonSegments).distances.map(d => {
      const D = d / EARTH_RADIUS;
      const sinP2 = sinP1 * Math.cos(D) + cosP1 * Math.sin(D) * cosA;
      const dl = Math.atan2(sinA * Math.sin(D) * cosP1, Math.cos(D) - sinP1 * sinP2);
      const dm = Math.atanh(sinP2) - Math.atanh(sinP1);
      return {
        d,
        x: (dl / (2 * Math.PI)) * worldPixelSize,
        y: (-dm / (2 * Math.PI)) * worldPixelSize
      };
    });
  }
  const [cellX, cellY] = model.cellSize ?? [1, 1];
  const rowSign = (model.rowDirection ?? 'south') === 'south' ? -1 : 1;
  let has = false;
  let best = 0;
  let bestDistance = 0;
  let tQ = -3.4028234663852886e38;
  let qDone = !(stop > 0);
  const last = lattice.getFloorIndex(maximum);
  for (let n = 0; n <= last; n++) {
    const d = lattice.getDistance(n);
    if (!qDone && d >= stop) {
      tQ = has ? best : -3.4028234663852886e38;
      qDone = true;
    }
    let u: number;
    let v: number;
    if (mercator) {
      const list = offsets;
      let s = 0;
      while (s < list.length - 2 && d >= list[s + 1].d) {
        s++;
      }
      const fraction = (d - list[s].d) / (list[s + 1].d - list[s].d);
      u = eyeColumn + list[s].x + fraction * (list[s + 1].x - list[s].x);
      v = eyeRow + list[s].y + fraction * (list[s + 1].y - list[s].y);
    } else {
      u = eyeColumn + (sinA * d) / cellX;
      v = eyeRow + (rowSign * cosA * d) / cellY;
    }
    const x = Math.floor(u);
    const y = Math.floor(v);
    if (x < 0 || y < 0 || x >= width - 1 || y >= height - 1) {
      break;
    }
    if (!isValidCell(model, x, y)) {
      continue;
    }
    const h = bilinear(model, x, y, u - x, v - y);
    const t = (h - eyeElevation) / d - d * curvature;
    if (!has || t > best) {
      has = true;
      best = t;
      bestDistance = d;
    }
  }
  if (!qDone) {
    tQ = has ? best : -3.4028234663852886e38;
  }
  return {has, t: best, d: bestDistance, tQ};
}

/** Returns `[sin, cos]` of the azimuth index in float64. */
export function getOracleAzimuth(index: number, count: number): [number, number] {
  const angle = ((index % count) * 2 * Math.PI) / count;
  return [Math.sin(angle), Math.cos(angle)];
}

/** Float64 mirror of `GPUPointHorizonProfile`. */
export function computePointHorizonProfile(
  model: PointHorizonOracleModel,
  observers: readonly PointHorizonOracleObserver[]
): PointHorizonOracleResult {
  const rayCount = observers.length * model.azimuthSpan;
  const tangent = new Float64Array(rayCount).fill(Number.NaN);
  const elevation = new Float64Array(rayCount).fill(Number.NaN);
  const distance = new Float64Array(rayCount).fill(Number.NaN);
  observers.forEach(([column, row, height], observerIndex) => {
    if (!(column >= 0 && row >= 0 && column <= model.width - 1 && row <= model.height - 1)) {
      return;
    }
    let eyeElevation = height;
    if ((model.heightReference ?? 'ground') === 'ground') {
      const ground = sampleOracleHeight(model, column, row);
      if (ground === undefined) {
        return;
      }
      eyeElevation = ground + height;
    }
    for (let a = 0; a < model.azimuthSpan; a++) {
      const [sinA, cosA] = getOracleAzimuth(model.firstAzimuth + a, model.azimuthCount);
      const result = marchOracleRay(model, column, row, eyeElevation, sinA, cosA);
      const index = observerIndex * model.azimuthSpan + a;
      tangent[index] = result.has ? result.t : -3.4028234663852886e38;
      elevation[index] = result.has ? (Math.atan(result.t) * 180) / Math.PI : -90;
      distance[index] = result.has ? result.d : 0;
    }
  });
  return {tangent, elevation, distance};
}

/** Seeded PRNG (mulberry32). */
export function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Fractal value-noise terrain in meters: octaves of smoothly interpolated random lattices with
 * wavelengths `baseWavelength / 2^k` pixels and amplitudes `amplitude * persistence^k`.
 */
export function createFractalTerrain(
  width: number,
  height: number,
  seed: number,
  amplitude = 400,
  baseWavelength = 48,
  octaves = 5,
  persistence = 0.55
): Float32Array {
  const random = createRandom(seed);
  const values = new Float32Array(width * height);
  let wavelength = baseWavelength;
  let scale = amplitude;
  for (let octave = 0; octave < octaves; octave++) {
    const columns = Math.ceil(width / wavelength) + 2;
    const rows = Math.ceil(height / wavelength) + 2;
    const lattice = Float64Array.from({length: columns * rows}, () => random());
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const gx = x / wavelength;
        const gy = y / wavelength;
        const x0 = Math.floor(gx);
        const y0 = Math.floor(gy);
        const sx = (gx - x0) ** 2 * (3 - 2 * (gx - x0));
        const sy = (gy - y0) ** 2 * (3 - 2 * (gy - y0));
        const at = (i: number, j: number) => lattice[j * columns + i];
        const top = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * sx;
        const bottom = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * sx;
        values[y * width + x] += scale * (top + (bottom - top) * sy);
      }
    }
    wavelength = Math.max(2, wavelength / 2);
    scale *= persistence;
  }
  return values;
}
