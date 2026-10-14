// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_CLASS_BREAKS_BOX_PLOT_CLASS_COUNT,
  type GPUClassBreaksSettings
} from '../../../src/gpu-dataframe/column-classification/class-breaks-parameters';
import {getOrderedFloat32Key} from '../../../src/gpu-dataframe/column-classification/column-classification-shared';
import {computeColumnQuantilesOracle} from './column-quantiles-oracle';
import {computeNaturalBreaksMirror} from './natural-breaks-mirror';

const fround = Math.fround;

/** Inputs of {@link computeClassBreaksOracle}. */
export type ClassBreaksOracleInput = {
  values: Float32Array;
  mask?: Uint32Array;
  parameters: GPUClassBreaksSettings;
  maximumClassCount: number;
  naturalBreaksBinCount: number;
};

/** Result of {@link computeClassBreaksOracle}. */
export type ClassBreaksOracleResult = {
  /** `maximumClassCount + 1` edges, NaN past `classCount`. */
  breaks: Float32Array;
  classCount: number;
  /** Finite minimum and maximum, by ordered key. */
  minimum: number;
  maximum: number;
  /** Natural breaks only: coarse dynamic program start bin of every class after the first. */
  naturalStartBins?: number[];
  /** Natural breaks only: the bin histogram the GPU builds. */
  naturalHistogram?: Uint32Array;
  /** Natural breaks only: inner edges before refinement (bin edges of the coarse partition). */
  naturalCoarseEdges?: number[];
};

function isFinite32(value: number): boolean {
  return Number.isFinite(value);
}

/**
 * CPU reference of `GPUClassBreaks`, written from the method definitions rather than the GPU
 * kernels: sorts for quantiles and gaps, f64 two-pass moments, an f64 Fisher-Jenks dynamic program
 * over the same bins. Values the GPU computes in f32 arithmetic are compared with a tolerance.
 */
export function computeClassBreaksOracle(input: ClassBreaksOracleInput): ClassBreaksOracleResult {
  const {values, mask, parameters, maximumClassCount} = input;
  const {method} = parameters;
  const finite: number[] = [];
  const valid: number[] = [];
  for (let row = 0; row < values.length; row++) {
    const value = values[row];
    if (Number.isNaN(value) || (mask && mask[row] === 0)) {
      continue;
    }
    valid.push(value);
    if (isFinite32(value)) {
      finite.push(value);
    }
  }
  const byKey = (left: number, right: number) =>
    getOrderedFloat32Key(left) - getOrderedFloat32Key(right);
  const sortedFinite = finite.slice().sort(byKey);
  const minimum = sortedFinite.length ? sortedFinite[0] : NaN;
  const maximum = sortedFinite.length ? sortedFinite[sortedFinite.length - 1] : NaN;
  let classCount =
    method === 'box-plot'
      ? GPU_CLASS_BREAKS_BOX_PLOT_CLASS_COUNT
      : method === 'custom'
        ? parameters.customEdges!.length - 1
        : (parameters.classCount ?? 5);
  // The prepare kernel's clamp of a raw per-frame count (the CPU packer refuses these values).
  if (method !== 'box-plot' && method !== 'custom') {
    classCount = classCount >= 1 ? Math.min(Math.floor(classCount), maximumClassCount) : 1;
  }
  const inner: number[] = [];
  const result: ClassBreaksOracleResult = {
    breaks: new Float32Array(maximumClassCount + 1).fill(NaN),
    classCount: 0,
    minimum,
    maximum
  };

  switch (method) {
    case 'equal-interval': {
      const width = fround(fround(maximum - minimum) / classCount);
      for (let edge = 1; edge < classCount; edge++) {
        inner.push(fround(minimum + fround(width * edge)));
      }
      break;
    }
    case 'quantile': {
      const quantiles = [];
      for (let edge = 1; edge < classCount; edge++) {
        quantiles.push(fround(edge / classCount));
      }
      const oracle = computeColumnQuantilesOracle({
        values,
        mask,
        quantiles,
        interpolation: 'linear'
      });
      inner.push(...oracle.quantiles);
      break;
    }
    case 'box-plot': {
      const oracle = computeColumnQuantilesOracle({
        values,
        mask,
        quantiles: [0.25, 0.5, 0.75],
        interpolation: 'linear'
      });
      const [lower, median, upper] = oracle.quantiles;
      const hinge = parameters.boxPlotHinge ?? 1.5;
      const spread = upper - lower;
      inner.push(lower - hinge * spread, lower, median, upper, upper + hinge * spread);
      break;
    }
    case 'standard-deviation': {
      const mean = finite.reduce((sum, value) => sum + value, 0) / finite.length;
      // Sample variance (n - 1), as mapclassify StdMean; a single value has zero spread.
      const variance =
        finite.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
        Math.max(finite.length - 1, 1);
      const interval = (parameters.standardDeviationInterval ?? 1) * Math.sqrt(variance);
      const center = (classCount - 2) / 2;
      for (let edge = 1; edge < classCount; edge++) {
        inner.push(mean + (edge - 1 - center) * interval);
      }
      break;
    }
    case 'head-tail': {
      const ratio = parameters.headTailRatio ?? 0.4;
      let head = finite;
      let previousCount = 0;
      for (let round = 0; round + 1 < classCount; round++) {
        const count = head.length;
        if (count < 1) {
          break;
        }
        const headMinimum = head.reduce((least, value) => Math.min(least, value), Infinity);
        const headMaximum = head.reduce((most, value) => Math.max(most, value), -Infinity);
        if (!(headMinimum < headMaximum)) {
          break;
        }
        if (round > 0 && (count > fround(fround(ratio) * previousCount) || count <= 1)) {
          break;
        }
        const mean = head.reduce((sum, value) => sum + value, 0) / count;
        inner.push(mean);
        previousCount = count;
        const threshold = getOrderedFloat32Key(fround(mean));
        head = head.filter(value => getOrderedFloat32Key(value) > threshold);
      }
      classCount = inner.length + 1;
      break;
    }
    case 'maximum-breaks': {
      // Gaps between consecutive distinct sorted values, largest first, lower position on ties.
      const gaps: {gap: number; position: number}[] = [];
      for (let position = 1; position < sortedFinite.length; position++) {
        const lower = sortedFinite[position - 1];
        const upper = sortedFinite[position];
        if (getOrderedFloat32Key(lower) !== getOrderedFloat32Key(upper)) {
          gaps.push({gap: fround(upper - lower), position});
        }
      }
      gaps.sort((left, right) => right.gap - left.gap || left.position - right.position);
      const midpoints = gaps.slice(0, Math.max(classCount - 1, 0)).map(({position}) => {
        const lower = sortedFinite[position - 1];
        const upper = sortedFinite[position];
        return fround(lower + fround(fround(upper - lower) * 0.5));
      });
      midpoints.sort((left, right) => left - right);
      inner.push(...midpoints);
      classCount = midpoints.length + 1;
      break;
    }
    case 'natural-breaks': {
      const natural = computeNaturalBreaksMirror(
        finite,
        minimum,
        maximum,
        classCount,
        input.naturalBreaksBinCount
      );
      result.naturalStartBins = natural.coarseStarts;
      result.naturalHistogram = natural.histogram;
      result.naturalCoarseEdges = natural.coarseEdges;
      inner.push(...natural.edges);
      classCount = natural.classCount;
      break;
    }
    case 'custom':
      break;
    default:
      throw new Error(`unknown method ${String(method)}`);
  }

  if (method === 'custom') {
    // A NaN edge makes the classification undefined (0 classes, NaN edges), as the publish kernel.
    if (Array.from(parameters.customEdges!).some(edge => Number.isNaN(edge))) {
      return result;
    }
    result.classCount = classCount;
    Array.from(parameters.customEdges!).forEach((edge, index) => {
      result.breaks[index] = edge;
    });
    return result;
  }
  if (sortedFinite.length === 0) {
    return result;
  }
  if (minimum === maximum) {
    result.classCount = 1;
    result.breaks[0] = minimum;
    result.breaks[1] = maximum;
    return result;
  }
  if (inner.slice(0, classCount - 1).some(edge => Number.isNaN(edge))) {
    return result;
  }
  result.classCount = classCount;
  result.breaks[0] = minimum;
  for (let edge = 1; edge < classCount; edge++) {
    result.breaks[edge] = Math.min(Math.max(inner[edge - 1], minimum), maximum);
  }
  result.breaks[classCount] = maximum;
  return result;
}

/**
 * Total squared deviation of the values from their class means, in f64, for the partition cut by
 * the inner edges (the GPUClassBreaks class rule: a value at or above `e[i]` is in class `i` or
 * later). The Fisher-Jenks objective.
 */
export function getWithinClassCost(
  values: ArrayLike<number>,
  innerEdges: readonly number[]
): number {
  const finite = Array.from(values).filter(Number.isFinite);
  const sums = new Array<number>(innerEdges.length + 1).fill(0);
  const counts = new Array<number>(innerEdges.length + 1).fill(0);
  const classes = finite.map(value => {
    let index = 0;
    while (index < innerEdges.length && value >= innerEdges[index]) {
      index++;
    }
    sums[index] += value;
    counts[index]++;
    return index;
  });
  let total = 0;
  finite.forEach((value, row) => {
    total += (value - sums[classes[row]] / counts[classes[row]]) ** 2;
  });
  return total;
}

/**
 * Exact Fisher-Jenks lower edges `e[1..k-1]` of the finite values by the textbook O(k n^2) dynamic
 * program in f64 on mean-centred sorted values (the independent reference of the specs; use for
 * n up to a few thousand). `unique` is false when another partition costs within 1e-9 relative, or
 * when the optimum splits a run of equal values; the edges are then not a unique answer.
 */
export function computeExactNaturalBreaks(
  values: ArrayLike<number>,
  classCount: number
): {edges: number[]; unique: boolean} {
  const sorted = Array.from(values)
    .filter(Number.isFinite)
    .sort((left, right) => left - right);
  const count = sorted.length;
  const mean = sorted.reduce((sum, value) => sum + value, 0) / count;
  const prefixSum = new Float64Array(count + 1);
  const prefixSquares = new Float64Array(count + 1);
  for (let row = 0; row < count; row++) {
    const centered = sorted[row] - mean;
    prefixSum[row + 1] = prefixSum[row] + centered;
    prefixSquares[row + 1] = prefixSquares[row] + centered * centered;
  }
  const getCost = (start: number, end: number) => {
    const members = end - start + 1;
    const sum = prefixSum[end + 1] - prefixSum[start];
    return prefixSquares[end + 1] - prefixSquares[start] - (sum * sum) / members;
  };
  let previous = new Float64Array(count);
  for (let end = 0; end < count; end++) {
    previous[end] = getCost(0, end);
  }
  const lowest: Int32Array[] = [new Int32Array(count)];
  const highest: Int32Array[] = [new Int32Array(count)];
  for (let layer = 1; layer < classCount; layer++) {
    const current = new Float64Array(count).fill(Infinity);
    const lowestStarts = new Int32Array(count);
    const highestStarts = new Int32Array(count);
    for (let end = layer; end < count; end++) {
      let best = Infinity;
      for (let start = layer; start <= end; start++) {
        best = Math.min(best, previous[start - 1] + getCost(start, end));
      }
      const tolerance = 1e-9 * Math.max(Math.abs(best), 1);
      let first = -1;
      let last = -1;
      for (let start = layer; start <= end; start++) {
        if (previous[start - 1] + getCost(start, end) <= best + tolerance) {
          first = first < 0 ? start : first;
          last = start;
        }
      }
      current[end] = best;
      lowestStarts[end] = first;
      highestStarts[end] = last;
    }
    previous = current;
    lowest.push(lowestStarts);
    highest.push(highestStarts);
  }
  const backtrack = (starts: Int32Array[]) => {
    const result: number[] = [];
    let end = count - 1;
    for (let layer = classCount - 1; layer > 0; layer--) {
      const start = starts[layer][end];
      result.unshift(start);
      end = start - 1;
    }
    return result;
  };
  const low = backtrack(lowest);
  const high = backtrack(highest);
  return {
    edges: low.map(start => sorted[start]),
    unique:
      low.every((start, index) => start === high[index]) &&
      low.every(start => sorted[start] !== sorted[start - 1])
  };
}

/** Brute-force rows per class: class `i` holds values with `i` inner edge keys at or below theirs. */
export function countClassesOracle(
  values: Float32Array,
  mask: Uint32Array | undefined,
  breaks: ArrayLike<number>,
  classCount: number,
  maximumClassCount: number
): Uint32Array {
  const counts = new Uint32Array(maximumClassCount);
  if (classCount === 0) {
    return counts;
  }
  for (let row = 0; row < values.length; row++) {
    const value = values[row];
    if (Number.isNaN(value) || (mask && mask[row] === 0)) {
      continue;
    }
    const key = getOrderedFloat32Key(value);
    let classIndex = 0;
    for (let edge = 1; edge < classCount; edge++) {
      if (getOrderedFloat32Key(breaks[edge]) <= key) {
        classIndex++;
      }
    }
    counts[classIndex]++;
  }
  return counts;
}
