// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  evaluateGPUKernelDensityKernel,
  type GPUKernelDensityKernel
} from '../../../src/gpu-spatial-analysis/point-density/kernel-density-math';
import {
  getOracleKernelMassFan,
  isOracleInsideWindow,
  type OracleStudyWindow
} from '../../analysis-core/study-window-oracle';
import type {BinnedDensityInput} from './kernel-density-binned-oracle';

/** CPU reference results for one density computation. */
export type DensityOracleResult = {
  counts: number[];
  sums: number[];
  means: number[];
  overflow: number;
};

type Bounds = [number, number, number, number];

/** Square-grid density matching `GPUGridBinning` inclusive-bounds semantics. */
export function computeGridDensity(
  positions: number[],
  weights: number[] | undefined,
  bounds: Bounds,
  gridSize: [number, number]
): DensityOracleResult {
  const [columns, rows] = gridSize;
  const [minX, minY, maxX, maxY] = bounds;
  const result = createResult(columns * rows);
  for (let index = 0; index < positions.length / 2; index++) {
    const x = positions[index * 2];
    const y = positions[index * 2 + 1];
    if (
      !Number.isFinite(x) ||
      !Number.isFinite(y) ||
      x < minX ||
      x > maxX ||
      y < minY ||
      y > maxY
    ) {
      continue;
    }
    const column = Math.min(Math.floor(((x - minX) / (maxX - minX || 1)) * columns), columns - 1);
    const row = Math.min(Math.floor(((y - minY) / (maxY - minY || 1)) * rows), rows - 1);
    accumulate(result, row * columns + column, weights?.[index]);
  }
  return finalize(result);
}

/** Hexagon density found by brute-force nearest center, independent of the cube-rounding code. */
export function computeHexagonDensity(
  positions: number[],
  weights: number[] | undefined,
  bounds: Bounds,
  gridSize: [number, number],
  radius: number
): DensityOracleResult {
  const [columns, rows] = gridSize;
  const [minX, minY, maxX, maxY] = bounds;
  const result = createResult(columns * rows);
  for (let index = 0; index < positions.length / 2; index++) {
    const x = positions[index * 2];
    const y = positions[index * 2 + 1];
    if (
      !Number.isFinite(x) ||
      !Number.isFinite(y) ||
      x < minX ||
      x > maxX ||
      y < minY ||
      y > maxY
    ) {
      continue;
    }
    const [column, row] = findNearestHexagon(x, y, minX, minY, radius, columns, rows).cell;
    if (column < 0 || row < 0 || column >= columns || row >= rows) {
      result.overflow = 1;
      continue;
    }
    accumulate(result, row * columns + column, weights?.[index]);
  }
  return finalize(result);
}

/** Nearest and second-nearest hexagon center distances over a padded lattice. */
export function findNearestHexagon(
  x: number,
  y: number,
  originX: number,
  originY: number,
  radius: number,
  columns: number,
  rows: number
): {cell: [number, number]; nearest: number; secondNearest: number} {
  let cell: [number, number] = [0, 0];
  let nearest = Infinity;
  let secondNearest = Infinity;
  const padding = 3;
  for (let row = -padding; row < rows + padding; row++) {
    for (let column = -padding; column < columns + padding; column++) {
      const centerX = originX + radius * Math.sqrt(3) * (column + 0.5 * (row & 1));
      const centerY = originY + 1.5 * radius * row;
      const distance = Math.hypot(x - centerX, y - centerY);
      if (distance < nearest) {
        secondNearest = nearest;
        nearest = distance;
        cell = [column, row];
      } else if (distance < secondNearest) {
        secondNearest = distance;
      }
    }
  }
  return {cell, nearest, secondNearest};
}

/** `[min, max]` over finite values selected by a nonzero mask, `[0, 0]` when empty. */
export function computeExtent(values: number[], mask: number[] | undefined): [number, number] {
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const [index, value] of values.entries()) {
    if ((mask && !mask[index]) || !Number.isFinite(value)) {
      continue;
    }
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
  }
  return minimum <= maximum ? [minimum, maximum] : [0, 0];
}

/** Equal-width histogram over an inclusive extent. */
export function computeHistogram(
  values: number[],
  mask: number[] | undefined,
  extent: [number, number],
  binCount: number
): number[] {
  const bins = new Array<number>(binCount).fill(0);
  const [minimum, maximum] = extent;
  for (const [index, value] of values.entries()) {
    if ((mask && !mask[index]) || value < minimum || value > maximum) {
      continue;
    }
    const bin =
      maximum === minimum
        ? 0
        : Math.min(Math.floor(((value - minimum) / (maximum - minimum)) * binCount), binCount - 1);
    bins[bin]++;
  }
  return bins;
}

/** Zero-boundary 2D convolution of a row-major field. */
export function convolveZero(
  field: number[],
  width: number,
  height: number,
  kernel: ArrayLike<number>,
  kernelWidth: number,
  kernelHeight: number
): number[] {
  const output = new Array<number>(width * height).fill(0);
  const halfWidth = (kernelWidth - 1) / 2;
  const halfHeight = (kernelHeight - 1) / 2;
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      let sum = 0;
      for (let kernelRow = 0; kernelRow < kernelHeight; kernelRow++) {
        for (let kernelColumn = 0; kernelColumn < kernelWidth; kernelColumn++) {
          const sourceRow = row + kernelRow - halfHeight;
          const sourceColumn = column + kernelColumn - halfWidth;
          if (sourceRow >= 0 && sourceRow < height && sourceColumn >= 0 && sourceColumn < width) {
            sum +=
              field[sourceRow * width + sourceColumn] *
              kernel[kernelRow * kernelWidth + kernelColumn];
          }
        }
      }
      output[row * width + column] = sum;
    }
  }
  return output;
}

/** Deterministic LCG points inside `bounds`, as a flat `[x0, y0, x1, y1, ...]` array. */
export function createSeededPoints(seed: number, count: number, bounds: Bounds): number[] {
  let state = seed >>> 0;
  const next = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
  const points: number[] = [];
  for (let index = 0; index < count; index++) {
    points.push(
      Math.fround(bounds[0] + next() * (bounds[2] - bounds[0])),
      Math.fround(bounds[1] + next() * (bounds[3] - bounds[1]))
    );
  }
  return points;
}

function createResult(cellCount: number): DensityOracleResult {
  return {
    counts: new Array<number>(cellCount).fill(0),
    sums: new Array<number>(cellCount).fill(0),
    means: new Array<number>(cellCount).fill(0),
    overflow: 0
  };
}

function accumulate(result: DensityOracleResult, cell: number, weight: number | undefined): void {
  result.counts[cell]++;
  if (weight !== undefined && Number.isFinite(weight)) {
    result.sums[cell] += weight;
  }
}

function finalize(result: DensityOracleResult): DensityOracleResult {
  result.means = result.counts.map((count, cell) => (count ? result.sums[cell] / count : 0));
  return result;
}

/** Inputs of the polygon KDE oracle, using absolute planar coordinates. */
export type PolygonKernelDensityInput = BinnedDensityInput & {
  window: OracleStudyWindow;
  kernel: GPUKernelDensityKernel;
  edgeCorrection: 'uniform' | 'diggle';
  normalization?: 'intensity' | 'probability';
};

/** Included raw and corrected point weights, independent of the GPU implementation. */
export function getPolygonKernelDensityWeights(input: PolygonKernelDensityInput): {
  weights: number[];
  weightSum: number;
  invalidCount: number;
} {
  let weightSum = 0;
  let invalidCount = 0;
  const weights = Array.from({length: input.points.length / 2}, (_, index) => {
    const point = [input.points[2 * index], input.points[2 * index + 1]] as const;
    const weight = input.weights?.[index] ?? 1;
    if (
      !point.every(Number.isFinite) ||
      !Number.isFinite(weight) ||
      weight < 0 ||
      !isOracleInsideWindow(input.window, point)
    ) {
      invalidCount++;
      return 0;
    }
    weightSum += weight;
    if (input.edgeCorrection === 'diggle') {
      const mass = getOracleKernelMassFan(input.window, point, input.bandwidth, input.kernel);
      return mass > 1e-6 ? weight / mass : 0;
    }
    return weight;
  });
  return {weights, weightSum, invalidCount};
}

/** Polygon cell mask, uniform edge mass and optional probability normalization. */
export function finishPolygonKernelDensity(
  input: PolygonKernelDensityInput,
  values: Float64Array,
  weightSum: number
): Float64Array {
  return values.map((value, index) => {
    const center = [
      input.start[0] + ((index % input.width) + 0.5) * input.cellSize[0],
      input.start[1] + (Math.floor(index / input.width) + 0.5) * input.cellSize[1]
    ] as const;
    if (!isOracleInsideWindow(input.window, center)) {
      return NaN;
    }
    let result = value;
    if (input.edgeCorrection === 'uniform') {
      const mass = getOracleKernelMassFan(input.window, center, input.bandwidth, input.kernel);
      result = mass > 1e-6 ? result / mass : NaN;
    }
    return input.normalization === 'probability'
      ? weightSum > 0
        ? result / weightSum
        : NaN
      : result;
  });
}

/** Exact polygon-window KDE with binary64 fan integration and point inclusion. */
export function getPolygonKernelDensityOracle(input: PolygonKernelDensityInput): Float64Array {
  const {weights, weightSum} = getPolygonKernelDensityWeights(input);
  const values = Float64Array.from({length: input.width * input.height}, (_, index) => {
    const centerX = input.start[0] + ((index % input.width) + 0.5) * input.cellSize[0];
    const centerY = input.start[1] + (Math.floor(index / input.width) + 0.5) * input.cellSize[1];
    let total = 0;
    for (const [pointIndex, weight] of weights.entries()) {
      if (weight > 0) {
        const distance = Math.hypot(
          input.points[2 * pointIndex] - centerX,
          input.points[2 * pointIndex + 1] - centerY
        );
        total += weight * evaluateGPUKernelDensityKernel(input.kernel, input.bandwidth, distance);
      }
    }
    return total;
  });
  return finishPolygonKernelDensity(input, values, weightSum);
}
