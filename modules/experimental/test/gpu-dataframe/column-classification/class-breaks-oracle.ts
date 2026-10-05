// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_CLASS_BREAKS_BOX_PLOT_CLASS_COUNT,
  type GPUClassBreaksParameters
} from '../../../src/gpu-dataframe/column-classification/class-breaks-parameters';
import {getOrderedFloat32Key} from '../../../src/gpu-dataframe/column-classification/column-classification-shared';
import {computeColumnQuantilesOracle} from './column-quantiles-oracle';

const fround = Math.fround;

/** Inputs of {@link computeClassBreaksOracle}. */
export type ClassBreaksOracleInput = {
  values: Float32Array;
  mask?: Uint32Array;
  parameters: GPUClassBreaksParameters;
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
  /** Natural breaks only: start bin of every class after the first. */
  naturalStartBins?: number[];
  /** Natural breaks only: the bin histogram the GPU builds. */
  naturalHistogram?: Uint32Array;
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
      const variance = finite.reduce((sum, value) => sum + (value - mean) ** 2, 0) / finite.length;
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
      const natural = computeNaturalBreaks(
        finite,
        minimum,
        maximum,
        Math.min(classCount, input.naturalBreaksBinCount),
        input.naturalBreaksBinCount
      );
      result.naturalStartBins = natural.startBins;
      result.naturalHistogram = natural.histogram;
      inner.push(...natural.edges);
      classCount = natural.startBins.length + 1;
      break;
    }
    case 'custom':
      break;
    default:
      throw new Error(`unknown method ${String(method)}`);
  }

  if (method === 'custom') {
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
  result.classCount = classCount;
  result.breaks[0] = minimum;
  for (let edge = 1; edge < classCount; edge++) {
    result.breaks[edge] = Math.min(Math.max(inner[edge - 1], minimum), maximum);
  }
  result.breaks[classCount] = maximum;
  return result;
}

/** Returns the bin of `value` exactly as the GPU kernel computes it. */
export function getNaturalBreaksBin(
  value: number,
  minimum: number,
  width: number,
  binCount: number
): number {
  const difference = fround(value - minimum);
  let bin = Math.floor(fround(difference / width));
  bin = Math.min(Math.max(bin, 0), binCount - 1);
  if (difference < fround(bin * width)) {
    bin -= 1;
  } else if (difference >= fround((bin + 1) * width)) {
    bin += 1;
  }
  return Math.min(Math.max(bin, 0), binCount - 1);
}

/** Total weighted squared deviation of bin centres for a partition given by class start bins. */
export function getNaturalBreaksCost(histogram: Uint32Array, startBins: number[]): number {
  const bounds = [0, ...startBins, histogram.length];
  let total = 0;
  for (let index = 0; index + 1 < bounds.length; index++) {
    let weight = 0;
    let sum = 0;
    for (let bin = bounds[index]; bin < bounds[index + 1]; bin++) {
      weight += histogram[bin];
      sum += histogram[bin] * bin;
    }
    const mean = weight > 0 ? sum / weight : 0;
    for (let bin = bounds[index]; bin < bounds[index + 1]; bin++) {
      total += histogram[bin] * (bin - mean) ** 2;
    }
  }
  return total;
}

function computeNaturalBreaks(
  finite: number[],
  minimum: number,
  maximum: number,
  classCount: number,
  binCount: number
): {edges: number[]; startBins: number[]; histogram: Uint32Array} {
  const histogram = new Uint32Array(binCount);
  const width = fround(fround(maximum - minimum) * fround(1 / binCount));
  for (const value of finite) {
    histogram[width > 0 ? getNaturalBreaksBin(value, minimum, width, binCount) : 0]++;
  }
  // Exact f64 Fisher-Jenks over weighted bin centres, lowest start on ties.
  // Class costs from running f64 sums; bin indices and counts keep them exact below 2^53.
  const costs: number[][] = [];
  for (let start = 0; start < binCount; start++) {
    costs.push([]);
    let weight = 0;
    let sum = 0;
    let squares = 0;
    for (let end = start; end < binCount; end++) {
      weight += histogram[end];
      sum += histogram[end] * end;
      squares += histogram[end] * end * end;
      costs[start][end] = weight > 0 ? squares - (sum * sum) / weight : 0;
    }
  }
  const totals: number[][] = [costs[0].slice()];
  const starts: number[][] = [new Array(binCount).fill(0)];
  for (let layer = 1; layer < classCount; layer++) {
    totals.push(new Array(binCount).fill(Infinity));
    starts.push(new Array(binCount).fill(layer));
    for (let end = layer; end < binCount; end++) {
      for (let start = layer; start <= end; start++) {
        const candidate = totals[layer - 1][start - 1] + costs[start][end];
        if (candidate < totals[layer][end]) {
          totals[layer][end] = candidate;
          starts[layer][end] = start;
        }
      }
    }
  }
  const startBins: number[] = [];
  let end = binCount - 1;
  for (let layer = classCount - 1; layer > 0; layer--) {
    const start = starts[layer][end];
    startBins.unshift(start);
    end = start - 1;
  }
  const edges = startBins.map(start => fround(minimum + fround(start * width)));
  return {edges, startBins, histogram};
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
