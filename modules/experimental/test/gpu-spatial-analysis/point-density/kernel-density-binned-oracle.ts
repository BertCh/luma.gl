// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  finishPolygonKernelDensity,
  getPolygonKernelDensityWeights,
  type PolygonKernelDensityInput
} from './point-density-oracle';

import {
  GPU_KERNEL_DENSITY_BINNED_BOX_PASS_COUNT,
  GPU_KERNEL_DENSITY_BINNED_EXPONENTIAL_VARIANCE_SHARE,
  GPU_KERNEL_DENSITY_GAUSSIAN_TRUNCATION
} from '../../../src/gpu-spatial-analysis/point-density/kernel-density-math';

/**
 * float64 model of the binned estimate of `GPUKernelDensity` (`method: 'binned'`), mirroring the
 * WGSL: the plan formulas of `blur-plan`, linear binning on the padded grid, the running-sum box
 * blur with periodic refresh, the two-sided exponential pass, and the final 1 / cell area scale.
 * Test-only oracle; nothing in `src` calls it.
 */

/** Per-axis filter parameters written by the `blur-plan` kernel. */
export type BinnedAxisPlan = {
  radius: number;
  edge: number;
  normalization: number;
  interval: number;
  pole: number;
};

/** The `writeAxisPlan` WGSL for one axis, in float64. `sigma` is `h / cell`. */
export function getBinnedAxisPlan(sigma: number, lineLength: number): BinnedAxisPlan {
  const varianceTotal = sigma * sigma;
  const varianceExponential = GPU_KERNEL_DENSITY_BINNED_EXPONENTIAL_VARIANCE_SHARE * varianceTotal;
  const varianceBox =
    (varianceTotal - varianceExponential) / GPU_KERNEL_DENSITY_BINNED_BOX_PASS_COUNT;
  const radius = Math.floor((Math.sqrt(1 + 12 * varianceBox) - 1) * 0.5);
  const edge = Math.min(
    Math.max(
      ((2 * radius + 1) * (varianceBox - (radius * (radius + 1)) / 3)) /
        (2 * ((radius + 1) * (radius + 1) - varianceBox)),
      0
    ),
    1
  );
  return {
    radius,
    edge,
    normalization: 1 / (2 * radius + 1 + 2 * edge),
    interval: Math.max(32, Math.min(2 * radius + 3, lineLength)),
    pole: varianceExponential / (varianceExponential + 1 + Math.sqrt(2 * varianceExponential + 1))
  };
}

/** Variance of one box pass of a plan: the exact value the cascade matches. */
export function getBoxVariance(plan: BinnedAxisPlan): number {
  const {radius, edge} = plan;
  return (
    ((radius * (radius + 1) * (2 * radius + 1)) / 3 + 2 * edge * (radius + 1) * (radius + 1)) /
    (2 * radius + 1 + 2 * edge)
  );
}

/** Box blur with zero extension by the WGSL running-sum update and periodic direct refresh. */
export function applyBoxBlur(line: Float64Array, plan: BinnedAxisPlan): Float64Array {
  const {radius, edge, normalization, interval} = plan;
  const length = line.length;
  const read = (position: number) => (position < 0 || position >= length ? 0 : line[position]);
  const output = new Float64Array(length);
  let window = 0;
  for (let position = 0; position < length; position++) {
    if (position % interval === 0) {
      let sum = 0;
      const last = Math.min(position + radius, length - 1);
      for (let tap = Math.max(position - radius, 0); tap <= last; tap++) {
        sum += read(tap);
      }
      window = sum + edge * (read(position - radius - 1) + read(position + radius + 1));
    } else {
      window +=
        (1 - edge) * (read(position + radius) - read(position - radius - 1)) +
        edge * (read(position + radius + 1) - read(position - radius - 2));
    }
    output[position] = Math.max(window, 0) * normalization;
  }
  return output;
}

/** Two-sided exponential pass: causal, then anticausal from the closed-form tail start. */
export function applyExponentialBlur(line: Float64Array, pole: number): Float64Array {
  const length = line.length;
  const output = new Float64Array(length);
  let state = 0;
  for (let position = 0; position < length; position++) {
    state = (1 - pole) * line[position] + pole * state;
    output[position] = state;
  }
  let next = state / (1 + pole);
  output[length - 1] = next;
  for (let position = length - 1; position > 0; position--) {
    next = (1 - pole) * output[position - 1] + pole * next;
    output[position - 1] = next;
  }
  return output;
}

/** Applies the cascade along one axis of a row-major array. */
function applyCascade(
  values: Float64Array,
  width: number,
  height: number,
  axis: 'x' | 'y',
  plan: BinnedAxisPlan
): Float64Array {
  const output = new Float64Array(values.length);
  const lineCount = axis === 'x' ? height : width;
  const lineLength = axis === 'x' ? width : height;
  for (let lineIndex = 0; lineIndex < lineCount; lineIndex++) {
    let line: Float64Array = new Float64Array(lineLength);
    for (let position = 0; position < lineLength; position++) {
      line[position] =
        values[axis === 'x' ? lineIndex * width + position : position * width + lineIndex];
    }
    for (let pass = 0; pass < GPU_KERNEL_DENSITY_BINNED_BOX_PASS_COUNT; pass++) {
      line = applyBoxBlur(line, plan);
    }
    line = applyExponentialBlur(line, plan.pole);
    for (let position = 0; position < lineLength; position++) {
      output[axis === 'x' ? lineIndex * width + position : position * width + lineIndex] =
        line[position];
    }
  }
  return output;
}

/** Inputs of the density models. Cell sizes are signed like the grid's. */
export type BinnedDensityInput = {
  /** Interleaved x, y. */
  points: readonly number[];
  weights?: readonly number[];
  width: number;
  height: number;
  /** Outer corner of cell (0, 0). */
  start: readonly [number, number];
  cellSize: readonly [number, number];
  bandwidth: number;
  /** `binnedPadding`. */
  padding?: number;
};

/** `method: 'binned'` with `edgeCorrection: 'none'` and `'intensity'`, row-major `width * height`. */
export function getBinnedDensityModel(input: BinnedDensityInput): Float64Array {
  const {width, height, start, cellSize, bandwidth} = input;
  const margin = (input.padding ?? 0) + 1;
  const paddedWidth = width + 2 * margin;
  const paddedHeight = height + 2 * margin;
  let bins: Float64Array = new Float64Array(paddedWidth * paddedHeight);
  for (let index = 0; index < input.points.length / 2; index++) {
    const weight = input.weights ? input.weights[index] : 1;
    if (!(weight > 0)) {
      continue;
    }
    const x = (input.points[2 * index] - start[0]) / cellSize[0] + margin - 0.5;
    const y = (input.points[2 * index + 1] - start[1]) / cellSize[1] + margin - 0.5;
    if (!(x >= 0 && y >= 0 && x < paddedWidth - 1 && y < paddedHeight - 1)) {
      continue;
    }
    const column = Math.floor(x);
    const row = Math.floor(y);
    const fractionX = x - column;
    const fractionY = y - row;
    const cell = row * paddedWidth + column;
    bins[cell] += weight * (1 - fractionX) * (1 - fractionY);
    bins[cell + 1] += weight * fractionX * (1 - fractionY);
    bins[cell + paddedWidth] += weight * (1 - fractionX) * fractionY;
    bins[cell + paddedWidth + 1] += weight * fractionX * fractionY;
  }
  bins = applyCascade(
    bins,
    paddedWidth,
    paddedHeight,
    'x',
    getBinnedAxisPlan(bandwidth / Math.abs(cellSize[0]), paddedWidth)
  );
  bins = applyCascade(
    bins,
    paddedWidth,
    paddedHeight,
    'y',
    getBinnedAxisPlan(bandwidth / Math.abs(cellSize[1]), paddedHeight)
  );
  const areaInverse = 1 / Math.abs(cellSize[0] * cellSize[1]);
  const density = new Float64Array(width * height);
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      density[row * width + column] =
        bins[(row + margin) * paddedWidth + column + margin] * areaInverse;
    }
  }
  return density;
}

/** The exact estimate (`method: 'exact'`, no edge correction, intensity), gaussian truncated at 5 h. */
export function getExactDensityModel(input: BinnedDensityInput): Float64Array {
  const {width, height, start, cellSize, bandwidth} = input;
  const density = new Float64Array(width * height);
  const supportSquared = (GPU_KERNEL_DENSITY_GAUSSIAN_TRUNCATION * bandwidth) ** 2;
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const centerX = start[0] + (column + 0.5) * cellSize[0];
      const centerY = start[1] + (row + 0.5) * cellSize[1];
      let total = 0;
      for (let index = 0; index < input.points.length / 2; index++) {
        const weight = input.weights ? input.weights[index] : 1;
        if (!(weight > 0)) {
          continue;
        }
        const distanceSquared =
          (input.points[2 * index] - centerX) ** 2 + (input.points[2 * index + 1] - centerY) ** 2;
        if (distanceSquared < supportSquared) {
          total +=
            (weight * Math.exp((-0.5 * distanceSquared) / bandwidth ** 2)) /
            (2 * Math.PI * bandwidth ** 2);
        }
      }
      density[row * width + column] = total;
    }
  }
  return density;
}

/** Binned polygon KDE: independent binary64 cascade with the same fan correction as exact KDE. */
export function getPolygonBinnedDensityOracle(input: PolygonKernelDensityInput): Float64Array {
  const {weights, weightSum} = getPolygonKernelDensityWeights(input);
  return finishPolygonKernelDensity(input, getBinnedDensityModel({...input, weights}), weightSum);
}
