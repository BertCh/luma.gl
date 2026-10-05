// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_COLOR_SCALE_NO_CLASS,
  type GPUColorScaleInterpolation,
  type GPUColorScaleType
} from '../../../src/gpu-dataframe/column-classification/color-scale-parameters';

/** Input of {@link computeColorScaleOnCPU}. */
export type ColorScaleOracleInput = {
  /** Float32 values, or uint32 category codes for the ordinal scale. */
  values: ArrayLike<number>;
  mask?: ArrayLike<number>;
  domain: ArrayLike<number>;
  /** Resolved active domain length (already `k + 1` on the `domainCount` view path). */
  activeDomainCount: number;
  palette: ArrayLike<number>;
  scale: GPUColorScaleType;
  paletteCount: number;
  interpolation?: GPUColorScaleInterpolation;
  clamp?: boolean;
  noDataColor?: number;
  logFloor?: number;
  exponent?: number;
  maximumDomainCount: number;
  maximumPaletteCount: number;
};

/** Output of {@link computeColorScaleOnCPU}. */
export type ColorScaleOracleResult = {
  classIndices: Uint32Array;
  colors: Uint32Array;
  classCounts: Uint32Array;
  /** 1 where a class or colour sits within float32 error of a decision boundary. */
  ambiguous: Uint8Array;
};

function transform(x: number, scale: GPUColorScaleType, exponent: number, logFloor: number) {
  switch (scale) {
    case 'sqrt':
      return Math.sign(x) * Math.sqrt(Math.abs(x));
    case 'pow':
      return x === 0 ? 0 : Math.sign(x) * Math.abs(x) ** exponent;
    case 'log': {
      const positive = x > 0 ? x : logFloor;
      return positive > 0 ? Math.log(positive) : NaN;
    }
    case 'symlog':
      return Math.sign(x) * Math.log(1 + Math.abs(x));
    default:
      return x;
  }
}

function distanceToInteger(value: number): number {
  return Math.abs(value - Math.round(value));
}

/**
 * Brute-force double precision reference for `GPUColorScale`, written from the documented
 * semantics: linear scans only, no binary search.
 */
export function computeColorScaleOnCPU(input: ColorScaleOracleInput): ColorScaleOracleResult {
  const rows = input.values.length;
  const classIndices = new Uint32Array(rows).fill(GPU_COLOR_SCALE_NO_CLASS);
  const colors = new Uint32Array(rows);
  const ambiguous = new Uint8Array(rows);
  const classCounts = new Uint32Array(input.maximumPaletteCount);
  const noDataColor = (input.noDataColor ?? 0) >>> 0;
  const paletteCount = Math.min(input.paletteCount, input.maximumPaletteCount);
  const domainCount = Math.min(
    input.activeDomainCount,
    input.maximumDomainCount,
    input.domain.length
  );
  const exponent = input.exponent ?? 1;
  const logFloor = input.logFloor ?? 1e-5;
  const isClamped = Boolean(input.clamp);
  const isLinear = input.interpolation === 'linear';
  const channel = (color: number, index: number) => (color >>> (index * 8)) & 255;
  for (let row = 0; row < rows; row++) {
    colors[row] = noDataColor;
    if (input.mask && input.mask[row] === 0) {
      continue;
    }
    const value = input.values[row];
    if (Number.isNaN(value) || paletteCount === 0) {
      continue;
    }
    let classIndex = -1;
    let position = 0;
    const scale = input.scale;
    if (scale === 'ordinal') {
      classIndex = value < paletteCount ? value : -1;
    } else if (domainCount === 0) {
      continue;
    } else if (scale === 'threshold' || scale === 'quantile') {
      if (domainCount < 2) {
        continue;
      }
      let count = 0;
      for (let edge = 1; edge <= domainCount - 2; edge++) {
        if (input.domain[edge] <= value) {
          count++;
        }
      }
      classIndex = Math.min(count, paletteCount - 1);
    } else {
      if (domainCount < 2) {
        continue;
      }
      const isQuantize = scale === 'quantize';
      const first = transform(input.domain[0], scale, exponent, logFloor);
      const last = transform(input.domain[domainCount - 1], scale, exponent, logFloor);
      const transformed = transform(value, scale, exponent, logFloor);
      const span = last - first;
      const t = span !== 0 ? (transformed - first) / span : 0.5;
      if (Number.isNaN(t) || (!isClamped && (t < 0 || t > 1))) {
        continue;
      }
      const clampedT = Math.min(Math.max(t, 0), 1);
      let rawClass: number;
      if (!isQuantize && domainCount > 2 && paletteCount === domainCount) {
        const stops = Array.from({length: domainCount}, (_, stop) =>
          transform(input.domain[stop], scale, exponent, logFloor)
        );
        if (t <= 0) {
          position = 0;
        } else if (t >= 1) {
          position = paletteCount - 1;
        } else {
          let segment = 0;
          for (let stop = 1; stop < domainCount - 1; stop++) {
            if (stops[stop] <= transformed) {
              segment = stop;
            }
          }
          const width = stops[segment + 1] - stops[segment];
          const fraction =
            width > 0 ? Math.min(Math.max((transformed - stops[segment]) / width, 0), 1) : 0;
          position = segment + fraction;
        }
        rawClass = position;
        ambiguous[row] = distanceToInteger(position) < 1e-3 ? 1 : 0;
      } else {
        position = clampedT * (paletteCount - 1);
        rawClass = clampedT * paletteCount;
        ambiguous[row] = distanceToInteger(rawClass) < 1e-3 ? 1 : 0;
      }
      if (!isClamped && (Math.abs(t) < 1e-5 || Math.abs(t - 1) < 1e-5)) {
        ambiguous[row] = 1;
      }
      classIndex = Math.min(Math.floor(rawClass), paletteCount - 1);
    }
    if (classIndex < 0) {
      continue;
    }
    classIndices[row] = classIndex;
    classCounts[classIndex]++;
    const continuous = ['linear', 'sqrt', 'pow', 'log', 'symlog'].includes(scale);
    if (isLinear && continuous) {
      const lower = Math.min(Math.floor(position), paletteCount - 1);
      const upper = Math.min(lower + 1, paletteCount - 1);
      const fraction = position - lower;
      let packed = 0;
      for (let index = 0; index < 4; index++) {
        const a = channel(input.palette[lower], index);
        const b = channel(input.palette[upper], index);
        const blended = Math.min(Math.max(Math.floor(a + (b - a) * fraction + 0.5), 0), 255);
        packed |= blended << (index * 8);
      }
      colors[row] = packed >>> 0;
    } else {
      colors[row] = input.palette[classIndex] >>> 0;
    }
  }
  return {classIndices, colors, classCounts, ambiguous};
}
