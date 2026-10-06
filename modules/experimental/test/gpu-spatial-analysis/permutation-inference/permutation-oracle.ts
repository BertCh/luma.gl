// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getFeistelPermutationIndex,
  getFeistelRoundKeys,
  PhiloxStream
} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-random';
import {LOCAL_PERMUTATION_RANDOM_TAG} from '../../../src/gpu-spatial-analysis/permutation-inference/gpu-local-permutation-test';
import {getPermutationSeedKey} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-parameters';

const f32 = Math.fround;

/** Oracle alternative; `'directed'` is the folded count of earlier releases. */
export type OracleAlternative = 'directed' | 'two-sided' | 'greater' | 'lesser' | 'folded';

/** Exceedance count `M` and p-value multiplier for the alternative (esda calculate_significance). */
export function getOracleExceedance(
  alternative: OracleAlternative,
  greater: number,
  lesser: number,
  permutations: number,
  folded = 0
): {count: number; multiplier: number} {
  switch (alternative) {
    case 'folded':
      return {count: folded, multiplier: 1};
    case 'greater':
      return {count: greater, multiplier: 1};
    case 'lesser':
      return {count: lesser, multiplier: 1};
    case 'two-sided':
      return {count: Math.min(greater, lesser), multiplier: 2};
    default:
      return {count: Math.min(greater, permutations - greater), multiplier: 1};
  }
}

/** Pseudo p-value `min(m (M + 1), R + 1) / (R + 1)`. */
export function getOraclePValue(count: number, multiplier: number, permutations: number): number {
  return Math.min(multiplier * (count + 1), permutations + 1) / (permutations + 1);
}

/** A CPU spatial-weights CSR (neighbor IDs ascending within each row). */
export type CPUSpatialWeights = {
  offsets: Uint32Array;
  neighbors: Uint32Array;
  weights: Float32Array;
};

type Compacted = {
  /** Compacted position per row, or -1. */
  positions: number[];
  /** Row of each compacted position. */
  rows: number[];
  count: number;
  mean: number;
  meanY: number;
};

/** mulberry32 generator with values in [0, 1). */
export function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Builds a CSR from per-row `[neighbor, weight]` lists, sorting each row by neighbor. */
export function createWeights(
  rows: readonly (readonly (readonly [number, number])[])[]
): CPUSpatialWeights {
  const offsets = new Uint32Array(rows.length + 1);
  const neighbors: number[] = [];
  const weights: number[] = [];
  for (const [row, entries] of rows.entries()) {
    for (const [neighbor, weight] of [...entries].sort((a, b) => a[0] - b[0])) {
      neighbors.push(neighbor);
      weights.push(weight);
    }
    offsets[row + 1] = neighbors.length;
  }
  return {offsets, neighbors: Uint32Array.from(neighbors), weights: Float32Array.from(weights)};
}

/** kNN weights over random points (binary or row standardized), `k` nearest by distance then ID. */
export function createKnnWeights(
  positions: Float64Array,
  k: number,
  rowStandardize: boolean
): CPUSpatialWeights {
  const count = positions.length / 2;
  const lists: [number, number][][] = [];
  for (let i = 0; i < count; i++) {
    const candidates: [number, number][] = [];
    for (let j = 0; j < count; j++) {
      if (j !== i) {
        candidates.push([
          j,
          Math.hypot(
            positions[j * 2] - positions[i * 2],
            positions[j * 2 + 1] - positions[i * 2 + 1]
          )
        ]);
      }
    }
    candidates.sort((a, b) => a[1] - b[1] || a[0] - b[0]);
    lists.push(
      candidates.slice(0, k).map(([j]): [number, number] => [j, rowStandardize ? 1 / k : 1])
    );
  }
  return createWeights(lists);
}

function compact(values: Float32Array, mask?: Uint32Array, secondValues?: Float32Array): Compacted {
  const positions: number[] = [];
  const rows: number[] = [];
  let sum = 0;
  let sumY = 0;
  for (let row = 0; row < values.length; row++) {
    const included =
      (!mask || mask[row] !== 0) &&
      Number.isFinite(values[row]) &&
      (!secondValues || Number.isFinite(secondValues[row]));
    positions.push(included ? rows.length : -1);
    if (included) {
      rows.push(row);
      sum += values[row];
      sumY += secondValues ? secondValues[row] : 0;
    }
  }
  return {
    positions,
    rows,
    count: rows.length,
    mean: f32(sum / rows.length),
    meanY: f32(sumY / rows.length)
  };
}

/** Valid neighbor slots of `row`: included, not the row itself, inside the row's CSR range. */
function getNeighborSlots(weights: CPUSpatialWeights, positions: number[], row: number): number[] {
  const slots: number[] = [];
  for (let slot = weights.offsets[row]; slot < weights.offsets[row + 1]; slot++) {
    const neighbor = weights.neighbors[slot];
    if (neighbor < positions.length && neighbor !== row && positions[neighbor] >= 0) {
      slots.push(slot);
    }
  }
  return slots;
}

/** Result of {@link computeLocalPermutationOracle}. */
export type LocalPermutationOracle = {
  /** Folded exceedance count, or 0xffffffff when not tested. */
  exceedances: number[];
  pseudoPValues: number[];
  observed: number[];
  significant: number[];
  significantFalseDiscoveryRate: number[];
  overflow: number;
  /** Unfolded simulated terms of row `focusRow` (diagnostics). */
  simulated?: number[];
};

/**
 * Reference of `GPULocalPermutationTest`: identical Philox draws and partial Fisher-Yates, f32
 * arithmetic emulated with `Math.fround` in the GPU's operation order.
 */
export function computeLocalPermutationOracle(input: {
  weights: CPUSpatialWeights;
  values: Float32Array;
  mask?: Uint32Array;
  statistic: 'localMoran' | 'localG' | 'localGStar';
  seed: number;
  permutations: number;
  significanceLevel: number;
  maximumNeighbors: number;
  alternative?: OracleAlternative;
}): LocalPermutationOracle {
  const {weights, values, statistic, permutations} = input;
  const rowCount = values.length;
  const compacted = compact(values, input.mask);
  const {positions, count} = compacted;
  const centered = statistic === 'localMoran';
  const compactX = compacted.rows.map(row =>
    centered ? f32(values[row] - compacted.mean) : values[row]
  );
  const sumX = compacted.rows.reduce((total, row) => total + values[row], 0);
  const sumSquares = compacted.rows.reduce(
    (total, row) => total + (values[row] - sumX / count) ** 2,
    0
  );
  const term = (focus: number, lag: number) =>
    statistic === 'localMoran' ? f32(focus * lag) : statistic === 'localG' ? lag : f32(lag + focus);
  const key = getPermutationSeedKey(input.seed);
  const result: LocalPermutationOracle = {
    exceedances: [],
    pseudoPValues: [],
    observed: [],
    significant: [],
    significantFalseDiscoveryRate: [],
    overflow: 0
  };
  const multipliers = getOracleExceedance(
    input.alternative ?? 'directed',
    0,
    0,
    permutations
  ).multiplier;
  for (let row = 0; row < rowCount; row++) {
    const position = positions[row];
    const slots = position >= 0 ? getNeighborSlots(weights, positions, row) : [];
    if (slots.length > input.maximumNeighbors) {
      result.overflow = 1;
    }
    if (position < 0 || slots.length === 0 || slots.length > input.maximumNeighbors) {
      result.exceedances.push(0xffffffff);
      result.observed.push(NaN);
      continue;
    }
    let observedLag = 0;
    for (const slot of slots) {
      observedLag = f32(
        observedLag + f32(weights.weights[slot] * compactX[positions[weights.neighbors[slot]]])
      );
    }
    const focus = compactX[position];
    const observed = term(focus, observedLag);
    const others = count - 1;
    let greater = 0;
    let lesser = 0;
    const simulatedValues: number[] = [];
    for (let permutation = 0; permutation < permutations; permutation++) {
      const stream = new PhiloxStream(key, row, permutation, LOCAL_PERMUTATION_RANDOM_TAG);
      const swaps = new Map<number, number>();
      let lag = 0;
      for (const [drawn, slot] of slots.entries()) {
        const pick = drawn + stream.nextBelow(others - drawn);
        const chosen = swaps.get(pick) ?? pick;
        if (pick !== drawn) {
          swaps.set(pick, swaps.get(drawn) ?? drawn);
        }
        const drawnPosition = chosen + (chosen >= position ? 1 : 0);
        lag = f32(lag + f32(weights.weights[slot] * compactX[drawnPosition]));
      }
      const simulated = term(focus, lag);
      greater += simulated >= observed ? 1 : 0;
      lesser += simulated <= observed ? 1 : 0;
      simulatedValues.push(simulated);
    }
    const simulatedMean = simulatedValues.reduce((a, b) => a + b, 0) / permutations;
    const folded = simulatedValues.filter(
      value => Math.abs(value - simulatedMean) >= Math.abs(observed - simulatedMean)
    ).length;
    const {count: larger} = getOracleExceedance(
      input.alternative ?? 'directed',
      greater,
      lesser,
      permutations,
      folded
    );
    result.exceedances.push(larger);
    const x = values[row];
    result.observed.push(
      statistic === 'localMoran'
        ? ((count - 1) * observed) / sumSquares
        : statistic === 'localG'
          ? observedLag / (sumX - x)
          : observed / sumX
    );
  }
  const level = f32(input.significanceLevel);
  const scale = f32(permutations + 1);
  for (const exceedance of result.exceedances) {
    const tested = exceedance !== 0xffffffff;
    result.pseudoPValues.push(
      tested ? getOraclePValue(exceedance, multipliers, permutations) : NaN
    );
    result.significant.push(
      tested &&
        f32(Math.min(multipliers * (exceedance + 1), permutations + 1)) <= f32(level * scale)
        ? 1
        : 0
    );
  }
  // Benjamini-Hochberg over the tested rows, sorted by count then row (a stable sort).
  const tested = result.exceedances
    .map((exceedance, row) => ({exceedance, row}))
    .filter(entry => entry.exceedance !== 0xffffffff)
    .sort((a, b) => a.exceedance - b.exceedance || a.row - b.row);
  const m = tested.length;
  let threshold = 0;
  for (const [index, entry] of tested.entries()) {
    const rank = index + 1;
    if (
      f32(f32(Math.min(multipliers * (entry.exceedance + 1), permutations + 1)) * f32(m)) <=
      f32(f32(f32(rank) * level) * scale)
    ) {
      threshold = rank;
    }
  }
  const ranks = new Map(tested.map((entry, index) => [entry.row, index + 1]));
  for (let row = 0; row < rowCount; row++) {
    const rank = ranks.get(row) ?? 0;
    result.significantFalseDiscoveryRate.push(rank > 0 && rank <= threshold ? 1 : 0);
  }
  return result;
}

/** Result of {@link computeGlobalPermutationOracle}. */
export type GlobalPermutationOracle = {
  observed: number;
  /** Simulated statistics of permutations 1 to P. */
  simulated: number[];
  /** Raw pair sums: index 0 is the observed (identity) one. */
  pairSums: number[];
  exceedances: number;
  pseudoPValue: number;
  /** Tail counts #{sim >= observed} and #{sim <= observed}. */
  greater: number;
  lesser: number;
  simulatedMean: number;
  simulatedStandardDeviation: number;
  zSimulated: number;
};

/**
 * Reference of `GPUGlobalPermutationTest` in double precision with the GPU's exact permutations
 * (the TypeScript Feistel bijection is bit-identical to the WGSL one).
 */
export function computeGlobalPermutationOracle(input: {
  weights: CPUSpatialWeights;
  values: Float32Array;
  secondValues?: Float32Array;
  mask?: Uint32Array;
  statistic: 'moran' | 'geary' | 'getisOrdG' | 'bivariateMoran';
  seed: number;
  permutations: number;
  alternative?: OracleAlternative;
}): GlobalPermutationOracle {
  const {weights, values, statistic, permutations} = input;
  const bivariate = statistic === 'bivariateMoran';
  const secondValues = bivariate ? input.secondValues : undefined;
  const compacted = compact(values, input.mask, secondValues);
  const {positions, rows, count} = compacted;
  const meanX = rows.reduce((total, row) => total + values[row], 0) / count;
  const meanY = secondValues
    ? rows.reduce((total, row) => total + secondValues[row], 0) / count
    : 0;
  const compactX = rows.map(row => (statistic === 'getisOrdG' ? values[row] : values[row] - meanX));
  const compactY = rows.map(row => (secondValues ? secondValues[row] - meanY : 0));
  const sumSquares = rows.reduce((total, row) => total + (values[row] - meanX) ** 2, 0);
  const sumSquaresY = compactY.reduce((total, value) => total + value * value, 0);
  const slotsByRow = rows.map(row => getNeighborSlots(weights, positions, row));
  let s0 = 0;
  for (const slots of slotsByRow) {
    for (const slot of slots) {
      s0 += weights.weights[slot];
    }
  }
  const key = getPermutationSeedKey(input.seed);
  const pairSums: number[] = [];
  for (let permutation = 0; permutation <= permutations; permutation++) {
    const roundKeys = getFeistelRoundKeys(key, permutation);
    const permute = (position: number) =>
      permutation === 0 ? position : getFeistelPermutationIndex(position, count, roundKeys);
    let sum = 0;
    for (let index = 0; index < rows.length; index++) {
      const focus = compactX[bivariate ? index : permute(index)];
      let lag = 0;
      for (const slot of slotsByRow[index]) {
        const neighborPosition = permute(positions[weights.neighbors[slot]]);
        const neighborValue = bivariate ? compactY[neighborPosition] : compactX[neighborPosition];
        const weight = weights.weights[slot];
        lag +=
          statistic === 'geary' ? weight * (focus - neighborValue) ** 2 : weight * neighborValue;
      }
      sum += statistic === 'geary' ? lag : focus * lag;
    }
    pairSums.push(sum);
  }
  const scale = {
    moran: count / (s0 * sumSquares),
    geary: (count - 1) / (2 * s0 * sumSquares),
    getisOrdG: 1 / (count * (count - 1) * meanX * meanX - sumSquares),
    bivariateMoran: count / (s0 * Math.sqrt(sumSquares * sumSquaresY))
  }[statistic];
  const simulated = pairSums.slice(1).map(sum => scale * sum);
  const greater = pairSums.slice(1).filter(sum => sum >= pairSums[0]).length;
  const lesser = pairSums.slice(1).filter(sum => sum <= pairSums[0]).length;
  const mean = simulated.reduce((a, b) => a + b, 0) / permutations;
  const observedStatistic = scale * pairSums[0];
  const folded = simulated.filter(
    value => Math.abs(value - mean) >= Math.abs(observedStatistic - mean)
  ).length;
  const {count: larger, multiplier} = getOracleExceedance(
    input.alternative ?? 'directed',
    greater,
    lesser,
    permutations,
    folded
  );
  const standardDeviation = Math.sqrt(
    simulated.reduce((total, value) => total + (value - mean) ** 2, 0) / permutations
  );
  const observed = scale * pairSums[0];
  return {
    observed,
    simulated,
    pairSums,
    exceedances: larger,
    pseudoPValue: getOraclePValue(larger, multiplier, permutations),
    greater,
    lesser,
    simulatedMean: mean,
    simulatedStandardDeviation: standardDeviation,
    zSimulated: (observed - mean) / standardDeviation
  };
}
