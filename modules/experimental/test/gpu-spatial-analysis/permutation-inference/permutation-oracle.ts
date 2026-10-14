// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUInferenceAlternative,
  GPUMultipleTestingMethod
} from '../../../src/gpu-spatial-analysis/contracts';
import {getMultipleTestingAdjustedPValues} from '../../../src/gpu-spatial-analysis/multiple-testing';
import {getPermutationPValue} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-alternative';
import {PhiloxStream} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-random';
import {
  getSwapOrNotPermutationIndex,
  getSwapOrNotRounds
} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-shuffle';
import {
  GPU_LOCAL_GEARY_CLASS,
  getLocalStatisticSpec,
  LOCAL_PERMUTATION_RANDOM_TAG,
  type GPULocalJoinCountCase,
  type GPULocalPermutationStatistic
} from '../../../src/gpu-spatial-analysis/permutation-inference/gpu-local-permutation-test';
import {getPermutationSeedKey} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-parameters';

const f32 = Math.fround;

/** Oracle alternative: the shared inference union; `'less'` is accepted as `'lesser'`. */
export type OracleAlternative = GPUInferenceAlternative;

/** Exceedance count `M` and p-value multiplier for the alternative (see `getPermutationPValue`). */
export function getOracleExceedance(
  alternative: OracleAlternative,
  greater: number,
  lesser: number,
  permutations: number,
  folded = 0
): {count: number; multiplier: number} {
  const {exceedances} = getPermutationPValue(alternative, {greater, lesser, folded}, permutations);
  return {count: exceedances, multiplier: alternative === 'two-sided' ? 2 : 1};
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
  /** Included rows without an active slot to another included row. */
  islandCount: number;
};

/** Island policy of the oracle (see `GPUIslandPolicy`). */
export type OracleIslands = 'exclude' | 'keep';

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

/**
 * Included rows (mask nonzero, finite values), islands (included rows without an active slot to
 * another included row) and the analysis rows compacted in ascending row order.
 */
function compact(
  weights: CPUSpatialWeights,
  values: Float32Array,
  islands: OracleIslands,
  mask?: Uint32Array,
  secondValues?: Float32Array
): Compacted {
  const rowCount = values.length;
  const included = Array.from(
    {length: rowCount},
    (_, row) =>
      (!mask || mask[row] !== 0) &&
      Number.isFinite(values[row]) &&
      (!secondValues || Number.isFinite(secondValues[row]))
  );
  const positions: number[] = [];
  const rows: number[] = [];
  let islandCount = 0;
  let sum = 0;
  let sumY = 0;
  for (let row = 0; row < rowCount; row++) {
    let hasNeighbor = false;
    if (included[row]) {
      for (let slot = weights.offsets[row]; slot < weights.offsets[row + 1]; slot++) {
        const neighbor = weights.neighbors[slot];
        if (
          neighbor < rowCount &&
          neighbor !== row &&
          weights.weights[slot] > 0 &&
          included[neighbor]
        ) {
          hasNeighbor = true;
          break;
        }
      }
    }
    islandCount += included[row] && !hasNeighbor ? 1 : 0;
    const analyzed = included[row] && (islands === 'keep' || hasNeighbor);
    positions.push(analyzed ? rows.length : -1);
    if (analyzed) {
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
    meanY: f32(sumY / rows.length),
    islandCount
  };
}

/** Active slots of `row`: valid neighbor other than the row, positive weight, analysis row. */
function getNeighborSlots(weights: CPUSpatialWeights, positions: number[], row: number): number[] {
  const slots: number[] = [];
  for (let slot = weights.offsets[row]; slot < weights.offsets[row + 1]; slot++) {
    const neighbor = weights.neighbors[slot];
    if (
      neighbor < positions.length &&
      neighbor !== row &&
      weights.weights[slot] > 0 &&
      positions[neighbor] >= 0
    ) {
      slots.push(slot);
    }
  }
  return slots;
}

/**
 * The weight the kernels read for each active slot: as given, or, for row-standardized weights,
 * `f32(w * f32(1 / sum))` with the f32 sum of the active weights in slot order.
 */
function getEffectiveWeights(
  weights: CPUSpatialWeights,
  slots: number[],
  standardize: boolean
): Map<number, number> {
  let total = 0;
  for (const slot of slots) {
    total = f32(total + weights.weights[slot]);
  }
  const scale = standardize ? (total > 0 ? f32(1 / total) : 0) : 1;
  return new Map(
    slots.map(slot => [
      slot,
      standardize ? f32(weights.weights[slot] * scale) : weights.weights[slot]
    ])
  );
}

/** Result of {@link computeLocalPermutationOracle}. */
export type LocalPermutationOracle = {
  /** Exceedance count `M`, or 0xffffffff when not tested. */
  exceedances: number[];
  /** Unadjusted pseudo p-values, NaN when not tested. */
  pseudoPValues: number[];
  /** Adjusted p-values for `multipleTesting`, NaN when not tested. */
  adjustedPValues: number[];
  observed: number[];
  /** Exact integer `p <= level` for `'none'`, else `adjustedPValue <= level`. */
  significant: number[];
  /** `localGeary` classes (`GPU_LOCAL_GEARY_CLASS`) from the significance rule of `significant`. */
  classes: number[];
  /** `localBivariateMoran` quadrants (PySAL scheme), 0 for rows that are not analyzed. */
  quadrants: number[];
  overflow: number;
  islandCount: number;
};

/**
 * Local Geary classes (esda `Geary_Local(labels=True)` rule, `0` for NaN): positive association
 * (`observed` below the mean of the finite local values) is 1 high-high or 2 low-low by the sign of
 * the centered `x`; above the mean is 3 (negative association); `p > level` is 4.
 */
export function classifyLocalGeary(input: {
  observed: readonly number[];
  centeredValues: readonly number[];
  pValues: readonly number[];
  level: number;
}): number[] {
  const finite = input.observed.filter(value => Number.isFinite(value));
  const mean = finite.reduce((total, value) => total + value, 0) / finite.length;
  return input.observed.map((value, row) => {
    const p = input.pValues[row];
    if (!Number.isFinite(value) || !Number.isFinite(p)) {
      return GPU_LOCAL_GEARY_CLASS.undefined;
    }
    if (!(p <= input.level)) {
      return GPU_LOCAL_GEARY_CLASS.notSignificant;
    }
    const x = input.centeredValues[row];
    if (value < mean) {
      return x > 0
        ? GPU_LOCAL_GEARY_CLASS.highHigh
        : x < 0
          ? GPU_LOCAL_GEARY_CLASS.lowLow
          : GPU_LOCAL_GEARY_CLASS.undefined;
    }
    return value > mean
      ? GPU_LOCAL_GEARY_CLASS.negativeAssociation
      : GPU_LOCAL_GEARY_CLASS.undefined;
  });
}

/**
 * Reference of `GPULocalPermutationTest`: identical Philox draws and partial Fisher-Yates, f32
 * arithmetic emulated with `Math.fround` in the GPU's operation order.
 */
export function computeLocalPermutationOracle(input: {
  weights: CPUSpatialWeights;
  values: Float32Array;
  secondValues?: Float32Array;
  mask?: Uint32Array;
  statistic: GPULocalPermutationStatistic;
  joinCase?: GPULocalJoinCountCase;
  seed: number;
  permutations: number;
  significanceLevel: number;
  maximumNeighbors: number;
  alternative?: OracleAlternative;
  islands?: OracleIslands;
  /** Weights are row standardized: re-standardize over the active slots. */
  rowStandardized?: boolean;
  multipleTesting?: GPUMultipleTestingMethod;
  /** The contributor's compile-time maximum; a larger `permutations` is invalid. */
  maximumPermutations?: number;
}): LocalPermutationOracle {
  const {weights, values, statistic, permutations} = input;
  const alternative = input.alternative ?? 'two-sided';
  const keep = input.islands === 'keep';
  const rowCount = values.length;
  const spec = getLocalStatisticSpec(statistic, input.joinCase);
  const secondValues = spec.bivariate ? input.secondValues : undefined;
  const compacted = compact(weights, values, input.islands ?? 'exclude', input.mask, secondValues);
  const {positions, count} = compacted;
  const compactX = compacted.rows.map(row =>
    spec.centerX ? f32(values[row] - compacted.mean) : values[row]
  );
  const compactY = compacted.rows.map(row =>
    secondValues ? (spec.centerY ? f32(secondValues[row] - compacted.meanY) : secondValues[row]) : 0
  );
  const sumX = compacted.rows.reduce((total, row) => total + values[row], 0);
  const sumSquares = compacted.rows.reduce(
    (total, row) => total + (values[row] - sumX / count) ** 2,
    0
  );
  const sumSquaresY = compactY.reduce((total, value) => total + value * value, 0);
  const one = (value: number) => (value === 1 ? 1 : 0);
  // The per-draw terms and the row term of getLocalStatisticSpec, in f32 and in the kernel's order.
  const drawnA = (fx: number, fy: number, dx: number, dy: number): number => {
    switch (statistic) {
      case 'localGeary':
        return f32(focusDifference(fx, dx) ** 2);
      case 'localBivariateMoran':
        return dy;
      case 'localJoinCount':
        return one(dx);
      case 'localBivariateJoinCount':
        return input.joinCase === 'BJC' ? (1 - one(dx)) * one(dy) : one(dx) * one(dy);
      default:
        return dx;
    }
  };
  const drawnB = (dy: number): number => dy;
  const term = (fx: number, fy: number, lagA: number, lagB: number): number => {
    switch (statistic) {
      case 'localMoran':
      case 'localBivariateMoran':
        return f32(fx * lagA);
      case 'localGStar':
        return f32(lagA + fx);
      case 'localLeeL':
        return f32(lagA * lagB);
      case 'localJoinCount':
        return f32(one(fx) * lagA);
      case 'localBivariateJoinCount':
        return input.joinCase === 'BJC'
          ? f32(one(fx) * (1 - one(fy)) * lagA)
          : f32(one(fx) * one(fy) * lagA);
      default:
        return lagA;
    }
  };
  const key = getPermutationSeedKey(input.seed);
  const countIsValid =
    Number.isInteger(permutations) &&
    permutations >= 1 &&
    (input.maximumPermutations === undefined || permutations <= input.maximumPermutations);
  const result: LocalPermutationOracle = {
    exceedances: [],
    pseudoPValues: [],
    adjustedPValues: [],
    observed: [],
    significant: [],
    classes: [],
    quadrants: [],
    overflow: countIsValid ? 0 : 1,
    islandCount: compacted.islandCount
  };
  const multiplier = getOracleExceedance(alternative, 0, 0, permutations).multiplier;
  for (let row = 0; row < rowCount; row++) {
    const position = positions[row];
    const slots = position >= 0 ? getNeighborSlots(weights, positions, row) : [];
    if (position >= 0 && countIsValid && slots.length > input.maximumNeighbors) {
      result.overflow = 1;
    }
    if (position < 0 || slots.length > input.maximumNeighbors) {
      result.exceedances.push(0xffffffff);
      result.observed.push(NaN);
      result.quadrants.push(0);
      continue;
    }
    if (slots.length === 0 && !keep) {
      result.exceedances.push(0xffffffff);
      result.observed.push(NaN);
      result.quadrants.push(0);
      continue;
    }
    const effective = getEffectiveWeights(weights, slots, input.rowStandardized === true);
    const slotWeight = (slot: number) => (spec.joinCount ? 1 : effective.get(slot)!);
    const fx = compactX[position];
    const fy = compactY[position];
    let lagA = 0;
    let lagB = 0;
    for (const slot of slots) {
      const neighborPosition = positions[weights.neighbors[slot]];
      const dx = compactX[neighborPosition];
      const dy = compactY[neighborPosition];
      lagA = f32(lagA + f32(slotWeight(slot) * drawnA(fx, fy, dx, dy)));
      lagB = f32(lagB + f32(slotWeight(slot) * drawnB(dy)));
    }
    const observed = term(fx, fy, lagA, lagB);
    const x = values[row];
    const scaled = {
      localMoran: ((count - 1) * observed) / sumSquares,
      localGeary: (observed * count) / sumSquares,
      localG: lagA / (sumX - x),
      localGStar: observed / sumX,
      localBivariateMoran: ((count - 1) * observed) / Math.sqrt(sumSquares * sumSquaresY),
      localLeeL: (count * observed) / Math.sqrt(sumSquares * sumSquaresY),
      localJoinCount: observed,
      localBivariateJoinCount: observed
    }[statistic];
    result.observed.push(scaled);
    result.quadrants.push(lagA > 0 ? (fx > 0 ? 1 : 2) : fx > 0 ? 4 : 3);
    const others = count - 1;
    if (
      !countIsValid ||
      slots.length === 0 ||
      slots.length > others ||
      (spec.joinCount && observed === 0)
    ) {
      // Kept islands and join counts of zero are never tested, nor is anything when the count is invalid.
      result.exceedances.push(0xffffffff);
      continue;
    }
    let greater = 0;
    let lesser = 0;
    const simulatedValues: number[] = [];
    for (let permutation = 0; permutation < permutations; permutation++) {
      const stream = new PhiloxStream(key, row, permutation, LOCAL_PERMUTATION_RANDOM_TAG);
      const swaps = new Map<number, number>();
      let lagSimulatedA = 0;
      let lagSimulatedB = 0;
      for (const [drawn, slot] of slots.entries()) {
        const pick = drawn + stream.nextBelow(others - drawn);
        const chosen = swaps.get(pick) ?? pick;
        if (pick !== drawn) {
          swaps.set(pick, swaps.get(drawn) ?? drawn);
        }
        const drawnPosition = chosen + (chosen >= position ? 1 : 0);
        const dx = compactX[drawnPosition];
        const dy = compactY[drawnPosition];
        lagSimulatedA = f32(lagSimulatedA + f32(slotWeight(slot) * drawnA(fx, fy, dx, dy)));
        lagSimulatedB = f32(lagSimulatedB + f32(slotWeight(slot) * drawnB(dy)));
      }
      const simulated = term(fx, fy, lagSimulatedA, lagSimulatedB);
      greater += simulated >= observed ? 1 : 0;
      lesser += simulated <= observed ? 1 : 0;
      simulatedValues.push(simulated);
    }
    const simulatedMean = simulatedValues.reduce((a, b) => a + b, 0) / permutations;
    const folded = simulatedValues.filter(
      value => Math.abs(value - simulatedMean) >= Math.abs(observed - simulatedMean)
    ).length;
    result.exceedances.push(
      getPermutationPValue(alternative, {greater, lesser, folded}, permutations).exceedances
    );
  }
  for (const exceedance of result.exceedances) {
    result.pseudoPValues.push(
      exceedance !== 0xffffffff ? getOraclePValue(exceedance, multiplier, permutations) : NaN
    );
  }
  const method = input.multipleTesting ?? 'none';
  const adjusted = getMultipleTestingAdjustedPValues(result.pseudoPValues, method);
  const level = f32(input.significanceLevel);
  const scale = f32(permutations + 1);
  for (const [row, exceedance] of result.exceedances.entries()) {
    const tested = exceedance !== 0xffffffff;
    result.adjustedPValues.push(tested ? adjusted[row] : NaN);
    result.significant.push(
      tested &&
        (method === 'none'
          ? f32(Math.min(multiplier * (exceedance + 1), permutations + 1)) <= f32(level * scale)
          : adjusted[row] <= level)
        ? 1
        : 0
    );
  }
  if (statistic === 'localGeary') {
    // The classification tests significance the way `significant` does (an exact integer rule for 'none').
    const classifiedP = result.exceedances.map((exceedance, row) =>
      exceedance === 0xffffffff ? NaN : result.significant[row] === 1 ? 0 : 1
    );
    result.classes = classifyLocalGeary({
      observed: result.observed,
      centeredValues: positions.map(position => (position >= 0 ? compactX[position] : 0)),
      pValues: classifiedP,
      level: 0.5
    });
  }
  return result;
}

function focusDifference(focus: number, other: number): number {
  return f32(focus - other);
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
  islandCount: number;
  /** Number `n` of analysis rows. */
  analysisCount: number;
};

/**
 * Reference of `GPUGlobalPermutationTest` in double precision with the GPU's exact permutations
 * (the TypeScript swap-or-not shuffle is bit-identical to the WGSL one).
 */
export function computeGlobalPermutationOracle(input: {
  weights: CPUSpatialWeights;
  values: Float32Array;
  secondValues?: Float32Array;
  mask?: Uint32Array;
  statistic: 'moran' | 'geary' | 'getisOrdG' | 'bivariateMoran' | 'leesL';
  seed: number;
  permutations: number;
  alternative?: OracleAlternative;
  islands?: OracleIslands;
  /** Weights are row standardized: re-standardize over the active slots. */
  rowStandardized?: boolean;
}): GlobalPermutationOracle {
  const {weights, values, statistic, permutations} = input;
  const alternative = input.alternative ?? 'two-sided';
  const lee = statistic === 'leesL';
  const bivariate = statistic === 'bivariateMoran' || lee;
  const secondValues = bivariate ? input.secondValues : undefined;
  const compacted = compact(weights, values, input.islands ?? 'exclude', input.mask, secondValues);
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
  const standardize = input.rowStandardized === true;
  // Row-standardized rows with any active slot sum to 1 after re-standardization. A kept island has
  // no active slot: its scale is 0 (the kernel's rule), not 1 / 0, which would turn 0 * Infinity into NaN.
  const rowScales = slotsByRow.map(slots =>
    standardize
      ? slots.length > 0
        ? 1 / slots.reduce((total, slot) => total + weights.weights[slot], 0)
        : 0
      : 1
  );
  let s0 = 0;
  let s0Squared = 0;
  for (const slots of slotsByRow) {
    if (standardize) {
      s0 += slots.length > 0 ? 1 : 0;
      s0Squared += slots.length > 0 ? 1 : 0;
    } else {
      let rowSum = 0;
      for (const slot of slots) {
        s0 += weights.weights[slot];
        rowSum += weights.weights[slot];
      }
      s0Squared += rowSum * rowSum;
    }
  }
  const key = getPermutationSeedKey(input.seed);
  const pairSums: number[] = [];
  for (let permutation = 0; permutation <= permutations; permutation++) {
    const shuffle = permutation === 0 ? undefined : getSwapOrNotRounds(key, permutation, count);
    const permute = (position: number) =>
      shuffle ? getSwapOrNotPermutationIndex(position, count, shuffle) : position;
    let sum = 0;
    for (let index = 0; index < rows.length; index++) {
      const focus = compactX[bivariate ? index : permute(index)];
      let lag = 0;
      let lagY = 0;
      for (const slot of slotsByRow[index]) {
        const neighborPosition = permute(positions[weights.neighbors[slot]]);
        const neighborValue = bivariate ? compactY[neighborPosition] : compactX[neighborPosition];
        const weight = weights.weights[slot];
        if (lee) {
          // Lee's L permutes x and y together: both lags use the same permuted neighbor.
          lag += weight * compactX[neighborPosition];
          lagY += weight * compactY[neighborPosition];
          continue;
        }
        lag +=
          statistic === 'geary' ? weight * (focus - neighborValue) ** 2 : weight * neighborValue;
      }
      lag *= rowScales[index];
      lagY *= rowScales[index];
      sum += lee ? lag * lagY : statistic === 'geary' ? lag : focus * lag;
    }
    pairSums.push(sum);
  }
  const scale = {
    moran: count / (s0 * sumSquares),
    geary: (count - 1) / (2 * s0 * sumSquares),
    getisOrdG: 1 / (count * (count - 1) * meanX * meanX - sumSquares),
    bivariateMoran: count / (s0 * Math.sqrt(sumSquares * sumSquaresY)),
    leesL: count / (s0Squared * Math.sqrt(sumSquares * sumSquaresY))
  }[statistic];
  const simulated = pairSums.slice(1).map(sum => scale * sum);
  const greater = pairSums.slice(1).filter(sum => sum >= pairSums[0]).length;
  const lesser = pairSums.slice(1).filter(sum => sum <= pairSums[0]).length;
  const mean = simulated.reduce((a, b) => a + b, 0) / permutations;
  const observedStatistic = scale * pairSums[0];
  const folded = simulated.filter(
    value => Math.abs(value - mean) >= Math.abs(observedStatistic - mean)
  ).length;
  const {exceedances: larger, pValue} = getPermutationPValue(
    alternative,
    {greater, lesser, folded},
    permutations
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
    pseudoPValue: pValue,
    greater,
    lesser,
    simulatedMean: mean,
    simulatedStandardDeviation: standardDeviation,
    zSimulated: (observed - mean) / standardDeviation,
    islandCount: compacted.islandCount,
    analysisCount: count
  };
}
