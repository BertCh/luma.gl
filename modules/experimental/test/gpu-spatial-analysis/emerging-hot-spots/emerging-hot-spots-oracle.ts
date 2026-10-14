// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUMultipleTestingMethod} from '../../../src/gpu-spatial-analysis/contracts/inference';
import {GPU_EMERGING_HOT_SPOT_CATEGORIES} from '../../../src/gpu-spatial-analysis/emerging-hot-spots';
import {getMultipleTestingAdjustedPValues} from '../../../src/gpu-spatial-analysis/multiple-testing';
import {getTwoSidedPValue} from '../spatial-autocorrelation/spatial-autocorrelation-oracle';

const fround = Math.fround;

/** Cube geometry shared by the oracle functions. */
export type EmergingHotSpotCubeShape = {
  gridWidth: number;
  gridHeight: number;
  sliceCount: number;
};

/** Space-time cube input of the oracle, indexed `cell * sliceCount + slice`. */
export type EmergingHotSpotCube = EmergingHotSpotCubeShape & {
  values: ArrayLike<number>;
  mask?: ArrayLike<number>;
};

/** Global moments over valid bins, in double precision. */
export type EmergingHotSpotMoments = {
  count: number;
  mean: number;
  variance: number;
  standardDeviation: number;
};

/** Whether a bin takes part: its cell is unmasked and its value is finite. */
export function isOracleBinValid(cube: EmergingHotSpotCube, bin: number): boolean {
  const cell = Math.floor(bin / cube.sliceCount);
  return (!cube.mask || cube.mask[cell] !== 0) && Number.isFinite(cube.values[bin]);
}

/** Pass 1: `n`, mean and population variance over valid bins. */
export function computeEmergingHotSpotMoments(cube: EmergingHotSpotCube): EmergingHotSpotMoments {
  let count = 0;
  let sum = 0;
  for (let bin = 0; bin < cube.values.length; bin++) {
    if (isOracleBinValid(cube, bin)) {
      count++;
      sum += cube.values[bin];
    }
  }
  const mean = count > 0 ? sum / count : 0;
  let squares = 0;
  for (let bin = 0; bin < cube.values.length; bin++) {
    if (isOracleBinValid(cube, bin)) {
      squares += (cube.values[bin] - mean) ** 2;
    }
  }
  const variance = count > 0 ? squares / count : 0;
  return {count, mean, variance, standardDeviation: Math.sqrt(variance)};
}

/**
 * Pass 2: space-time Gi* z per bin with binary weights over the lattice disc and the current and
 * `temporalWindow` previous slices. `packed` is the float32 parameter array.
 */
export function computeSpaceTimeGiStar(
  cube: EmergingHotSpotCube,
  packed: ArrayLike<number>,
  maximumRadius = 4
): Float64Array {
  const {gridWidth, gridHeight, sliceCount} = cube;
  const moments = computeEmergingHotSpotMoments(cube);
  const radius = Math.min(packed[0], maximumRadius);
  const radiusSquared = fround(radius * radius);
  const reach = Math.floor(radius);
  const windowSize = Math.min(Math.floor(packed[1]), sliceCount - 1);
  const result = new Float64Array(cube.values.length).fill(NaN);
  for (let bin = 0; bin < cube.values.length; bin++) {
    if (!isOracleBinValid(cube, bin)) {
      continue;
    }
    const cell = Math.floor(bin / sliceCount);
    const slice = bin % sliceCount;
    const column = cell % gridWidth;
    const row = Math.floor(cell / gridWidth);
    let sum = 0;
    let neighborCount = 0;
    for (let deltaRow = -reach; deltaRow <= reach; deltaRow++) {
      for (let deltaColumn = -reach; deltaColumn <= reach; deltaColumn++) {
        const neighborRow = row + deltaRow;
        const neighborColumn = column + deltaColumn;
        if (
          neighborRow < 0 ||
          neighborRow >= gridHeight ||
          neighborColumn < 0 ||
          neighborColumn >= gridWidth ||
          deltaRow * deltaRow + deltaColumn * deltaColumn > radiusSquared
        ) {
          continue;
        }
        const neighborCell = neighborRow * gridWidth + neighborColumn;
        for (let t = Math.max(slice - windowSize, 0); t <= slice; t++) {
          const neighborBin = neighborCell * sliceCount + t;
          if (isOracleBinValid(cube, neighborBin)) {
            sum += cube.values[neighborBin] - moments.mean;
            neighborCount++;
          }
        }
      }
    }
    const spread = (neighborCount * (moments.count - neighborCount)) / (moments.count - 1);
    if (moments.count >= 2 && moments.variance > 0 && spread > 0) {
      result[bin] = sum / (moments.standardDeviation * Math.sqrt(spread));
    }
  }
  return result;
}

/**
 * Pass 2 in weights mode: space-time Gi* z per bin with spatial weights `w_ij` over cells (the
 * cell's own bins weigh `selfWeight`), temporal weight 1 over the current and `temporalWindow`
 * previous slices, and the Ord-Getis form `sum w (x - X) / (S sqrt((n S1 - W^2) / (n - 1)))`.
 * `cube.gridWidth * cube.gridHeight` is the cell count.
 */
export function computeSpaceTimeGiStarWeighted(
  cube: EmergingHotSpotCube,
  weights: {offsets: ArrayLike<number>; neighbors: ArrayLike<number>; weights: ArrayLike<number>},
  selfWeight: number,
  packed: ArrayLike<number>
): Float64Array {
  const {sliceCount} = cube;
  const moments = computeEmergingHotSpotMoments(cube);
  const windowSize = Math.min(Math.floor(packed[1]), sliceCount - 1);
  const result = new Float64Array(cube.values.length).fill(NaN);
  for (let bin = 0; bin < cube.values.length; bin++) {
    if (!isOracleBinValid(cube, bin)) {
      continue;
    }
    const cell = Math.floor(bin / sliceCount);
    const slice = bin % sliceCount;
    const spatial = [{cell, weight: selfWeight}];
    for (let slot = weights.offsets[cell]; slot < weights.offsets[cell + 1]; slot++) {
      if (weights.neighbors[slot] !== cell) {
        spatial.push({cell: weights.neighbors[slot], weight: weights.weights[slot]});
      }
    }
    let weightedSum = 0;
    let weightSum = 0;
    let squareSum = 0;
    for (const {cell: neighborCell, weight} of spatial) {
      if (weight === 0) {
        continue;
      }
      for (let t = Math.max(slice - windowSize, 0); t <= slice; t++) {
        const neighborBin = neighborCell * sliceCount + t;
        if (isOracleBinValid(cube, neighborBin)) {
          weightedSum += weight * (cube.values[neighborBin] - moments.mean);
          weightSum += weight;
          squareSum += weight * weight;
        }
      }
    }
    const spread = (moments.count * squareSum - weightSum ** 2) / (moments.count - 1);
    if (moments.count >= 2 && moments.variance > 0 && spread > 0) {
      result[bin] = weightedSum / (moments.standardDeviation * Math.sqrt(spread));
    }
  }
  return result;
}

/** Result of {@link computeMannKendall}. */
export type MannKendallResult = {
  /** Finite sample count. */
  count: number;
  /** Exact integer `S`. */
  statistic: number;
  /** Integer tie correction `sum_t t (t - 1) (2 t + 5)`. */
  tieTerm: number;
  /** Variance numerator over 18; 0 when undefined. */
  variance: number;
  /** Continuity-corrected z. */
  z: number;
  /** Two-sided normal p-value (Numerical Recipes erfcc fit). */
  p: number;
};

/** Pass 3 for one series: Mann-Kendall over finite values with exact-equality tie groups. */
export function computeMannKendall(series: ArrayLike<number>): MannKendallResult {
  const values: number[] = [];
  for (let index = 0; index < series.length; index++) {
    if (Number.isFinite(series[index])) {
      values.push(series[index]);
    }
  }
  const count = values.length;
  let statistic = 0;
  for (let first = 0; first < count; first++) {
    for (let second = first + 1; second < count; second++) {
      statistic += Math.sign(values[second] - values[first]);
    }
  }
  const groups = new Map<number, number>();
  for (const value of values) {
    groups.set(value, (groups.get(value) ?? 0) + 1);
  }
  let tieTerm = 0;
  for (const size of groups.values()) {
    if (size > 1) {
      tieTerm += size * (size - 1) * (2 * size + 5);
    }
  }
  const numerator = count >= 2 ? count * (count - 1) * (2 * count + 5) - tieTerm : 0;
  let z = 0;
  let p = 1;
  if (numerator > 0) {
    const deviation = Math.sqrt(numerator / 18);
    z =
      statistic > 0 ? (statistic - 1) / deviation : statistic < 0 ? (statistic + 1) / deviation : 0;
    p = getTwoSidedPValue(z);
  }
  return {count, statistic, tieTerm, variance: numerator / 18, z, p};
}

/** Per-series inputs of {@link classifyEmergingHotSpotSeries} beyond the z series itself. */
export type EmergingHotSpotTrend = {z: number; p: number};

/** Options shared by the oracle's classification functions. */
export type EmergingHotSpotOracleOptions = {
  /** Correction method of the node under test. Defaults to `'benjamini-hochberg'`, like the node. */
  multipleTesting?: GPUMultipleTestingMethod;
  /** Final slice rule of the node under test. Defaults to `'require'`. */
  finalSlice?: 'require' | 'last-valid';
  /**
   * Adjusted p-value per bin (for example read back from the GPU) used instead of recomputing it
   * from z, so a bin whose p-value sits within float32 error of the level cannot flip a comparison.
   * Ignored with `'none'`.
   */
  adjustedPValues?: ArrayLike<number>;
};

/** Two-sided normal p-value of every finite z (the WGSL's erfcc fit in double precision); NaN otherwise. */
export function computeBinPValues(zScores: ArrayLike<number>): Float64Array {
  return Float64Array.from({length: zScores.length}, (_, bin) =>
    Number.isFinite(zScores[bin]) ? getTwoSidedPValue(zScores[bin]) : NaN
  );
}

/** Adjusted p-values over the family of all finite-z bins, using the stream A CPU reference. */
export function computeAdjustedBinPValues(
  zScores: ArrayLike<number>,
  method: GPUMultipleTestingMethod
): Float64Array {
  return getMultipleTestingAdjustedPValues(computeBinPValues(zScores), method);
}

/**
 * Hot (1) and cold (2) bins, 0 otherwise. Without a correction: `z >= criticalZ` and
 * `z <= -criticalZ`. With one: adjusted p at most `fround(1 - confidenceLevel)` (`packed[6]`) with the
 * sign of z.
 */
export function computeBinStates(
  zScores: ArrayLike<number>,
  packed: ArrayLike<number>,
  options: EmergingHotSpotOracleOptions = {}
): Uint8Array {
  const method = options.multipleTesting ?? 'benjamini-hochberg';
  const states = new Uint8Array(zScores.length);
  const adjusted =
    method === 'none'
      ? undefined
      : (options.adjustedPValues ?? computeAdjustedBinPValues(zScores, method));
  const criticalZ = fround(packed[2]);
  const level = fround(packed[6]);
  for (let bin = 0; bin < zScores.length; bin++) {
    const z = zScores[bin];
    if (!Number.isFinite(z)) {
      continue;
    }
    if (adjusted) {
      if (adjusted[bin] <= level) {
        states[bin] = z > 0 ? 1 : z < 0 ? 2 : 0;
      }
    } else if (z >= criticalZ) {
      states[bin] = 1;
    } else if (z <= -criticalZ) {
      states[bin] = 2;
    }
  }
  return states;
}

/**
 * Pass 4 for one z series (see `GPUEmergingHotSpots` for the rules). `packed` is the float32
 * parameter array; thresholds use the same f32 arithmetic as the WGSL. `options.states` gives the
 * hot (1) and cold (2) state of every slice; without it the pre-correction `|z| >= criticalZ` rule
 * applies. A `trend` with fewer finite slices than `packed[5]` is treated as unavailable.
 */
export function classifyEmergingHotSpotSeries(
  series: ArrayLike<number>,
  trend: EmergingHotSpotTrend,
  packed: ArrayLike<number>,
  options: {states?: ArrayLike<number>; finalSlice?: 'require' | 'last-valid'} = {}
): number {
  const criticalZ = fround(packed[2]);
  const trendLevel = fround(packed[3]);
  const fraction = fround(packed[4]);
  const minimumTrendSliceCount = fround(packed[5]);
  let valid = 0;
  let hot = 0;
  let cold = 0;
  let trailingHot = 0;
  let trailingCold = 0;
  let finalState = 0;
  let lastValid = -1;
  for (let index = 0; index < series.length; index++) {
    const z = series[index];
    if (!Number.isFinite(z)) {
      continue;
    }
    valid++;
    lastValid = index;
    const state = options.states
      ? options.states[index]
      : z >= criticalZ
        ? 1
        : z <= -criticalZ
          ? 2
          : 0;
    if (state === 1) {
      hot++;
      trailingHot++;
      trailingCold = 0;
      finalState = 1;
    } else if (state === 2) {
      cold++;
      trailingCold++;
      trailingHot = 0;
      finalState = 2;
    } else {
      trailingHot = 0;
      trailingCold = 0;
      finalState = 0;
    }
  }
  const categories = GPU_EMERGING_HOT_SPOT_CATEGORIES;
  if (valid === 0) {
    return categories.NO_PATTERN;
  }
  if ((options.finalSlice ?? 'require') === 'require' && lastValid !== series.length - 1) {
    return categories.UNDETERMINED;
  }
  const isTrendKnown = valid >= minimumTrendSliceCount;
  const threshold = fround(fraction * valid);
  const hotPersistent = hot >= threshold;
  const coldPersistent = cold >= threshold;
  const significant = trend.p <= trendLevel;
  const trendUp = significant && trend.z > 0;
  const trendDown = significant && trend.z < 0;
  if (finalState === 1) {
    if (hot === 1) {
      return categories.NEW_HOT;
    }
    if (trailingHot >= 2 && trailingHot === hot && !hotPersistent) {
      return categories.CONSECUTIVE_HOT;
    }
    if (hotPersistent) {
      if (!isTrendKnown) {
        return categories.UNDETERMINED;
      }
      return trendUp
        ? categories.INTENSIFYING_HOT
        : trendDown
          ? categories.DIMINISHING_HOT
          : categories.PERSISTENT_HOT;
    }
    return cold === 0 ? categories.SPORADIC_HOT : categories.OSCILLATING_HOT;
  }
  if (finalState === 2) {
    if (cold === 1) {
      return categories.NEW_COLD;
    }
    if (trailingCold >= 2 && trailingCold === cold && !coldPersistent) {
      return categories.CONSECUTIVE_COLD;
    }
    if (coldPersistent) {
      if (!isTrendKnown) {
        return categories.UNDETERMINED;
      }
      return trendDown
        ? categories.INTENSIFYING_COLD
        : trendUp
          ? categories.DIMINISHING_COLD
          : categories.PERSISTENT_COLD;
    }
    return hot === 0 ? categories.SPORADIC_COLD : categories.OSCILLATING_COLD;
  }
  if (hot > 0 && hotPersistent) {
    return categories.HISTORICAL_HOT;
  }
  if (cold > 0 && coldPersistent) {
    return categories.HISTORICAL_COLD;
  }
  if (hot > 0 && cold === 0) {
    return categories.SPORADIC_HOT;
  }
  if (cold > 0 && hot === 0) {
    return categories.SPORADIC_COLD;
  }
  return categories.NO_PATTERN;
}

/** Per-cell results of passes 3 and 4. */
export type EmergingHotSpotCellResults = {
  trendZ: number[];
  trendP: number[];
  trendS: number[];
  category: number[];
  hotSliceCount: number[];
  coldSliceCount: number[];
  /** Index of the last finite slice per cell, or `0xffffffff`. */
  lastValidSlice: number[];
};

/** Runs passes 3 and 4 over every cell of a z cube (for example the GPU's own `giZScores`). */
export function computeEmergingHotSpotCells(
  zScores: ArrayLike<number>,
  shape: EmergingHotSpotCubeShape,
  packed: ArrayLike<number>,
  options: EmergingHotSpotOracleOptions = {}
): EmergingHotSpotCellResults {
  const cellCount = shape.gridWidth * shape.gridHeight;
  const states = computeBinStates(zScores, packed, options);
  const minimumTrendSliceCount = fround(packed[5]);
  const results: EmergingHotSpotCellResults = {
    trendZ: [],
    trendP: [],
    trendS: [],
    category: [],
    hotSliceCount: [],
    coldSliceCount: [],
    lastValidSlice: []
  };
  for (let cell = 0; cell < cellCount; cell++) {
    const begin = cell * shape.sliceCount;
    const series = Array.from({length: shape.sliceCount}, (_, slice) => zScores[begin + slice]);
    const cellStates = Array.from(states.subarray(begin, begin + shape.sliceCount));
    const trend = computeMannKendall(series);
    const isTrendKnown = trend.count >= minimumTrendSliceCount;
    const reported = isTrendKnown ? trend : {...trend, z: NaN, p: NaN};
    results.trendZ.push(reported.z);
    results.trendP.push(reported.p);
    results.trendS.push(trend.statistic);
    results.category.push(
      classifyEmergingHotSpotSeries(series, reported, packed, {
        states: cellStates,
        finalSlice: options.finalSlice
      })
    );
    results.hotSliceCount.push(cellStates.filter(state => state === 1).length);
    results.coldSliceCount.push(cellStates.filter(state => state === 2).length);
    const lastFinite = series.findLastIndex(z => Number.isFinite(z));
    results.lastValidSlice.push(lastFinite < 0 ? 0xffffffff : lastFinite);
  }
  return results;
}

/** Deterministic mulberry32 generator in `[0, 1)`. */
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

const HOT = 30;
const COLD = -30;

/** One designed series per category with the category it must classify as (radius 0, window 0). */
export type DesignedSeries = {
  name: string;
  expected: number;
  values: number[];
};

/**
 * Designed 20-slice series that, at radius 0 and window 0 inside a cube with many near-zero
 * background cells, reach every one of the 17 categories plus no-pattern cases.
 */
export function createDesignedSeries(): DesignedSeries[] {
  const categories = GPU_EMERGING_HOT_SPOT_CATEGORIES;
  const length = 20;
  const zeros = (count: number) => new Array<number>(count).fill(0);
  const repeat = (count: number, value: number) => new Array<number>(count).fill(value);
  const ramp = (from: number, to: number) =>
    Array.from({length}, (_, index) => from + ((to - from) * index) / (length - 1));
  const zigzag = (value: number) =>
    Array.from({length}, (_, index) => value + (index % 2 === 0 ? 1 : -1));
  const series: DesignedSeries[] = [
    {
      name: 'new hot',
      expected: categories.NEW_HOT,
      values: [...zeros(19), HOT]
    },
    {
      name: 'consecutive hot',
      expected: categories.CONSECUTIVE_HOT,
      values: [...zeros(15), ...repeat(5, HOT)]
    },
    {
      name: 'intensifying hot',
      expected: categories.INTENSIFYING_HOT,
      values: ramp(24, 48)
    },
    {
      name: 'persistent hot',
      expected: categories.PERSISTENT_HOT,
      values: zigzag(HOT)
    },
    {
      name: 'diminishing hot',
      expected: categories.DIMINISHING_HOT,
      values: ramp(48, 24)
    },
    {
      name: 'sporadic hot',
      expected: categories.SPORADIC_HOT,
      values: [HOT, ...zeros(4), HOT, ...zeros(3), HOT, ...zeros(2), HOT, ...zeros(6), HOT, 0]
    },
    {
      name: 'sporadic hot final hot',
      expected: categories.SPORADIC_HOT,
      values: [HOT, ...zeros(4), HOT, ...zeros(3), HOT, ...zeros(2), HOT, ...zeros(6), 0, HOT]
    },
    {
      name: 'oscillating hot',
      expected: categories.OSCILLATING_HOT,
      values: [COLD, HOT, ...zeros(5), COLD, ...zeros(4), HOT, ...zeros(5), HOT, HOT]
    },
    {
      name: 'historical hot',
      expected: categories.HISTORICAL_HOT,
      values: [...zigzag(HOT).slice(0, 19), 0]
    },
    {
      name: 'new cold',
      expected: categories.NEW_COLD,
      values: [...zeros(19), COLD]
    },
    {
      name: 'consecutive cold',
      expected: categories.CONSECUTIVE_COLD,
      values: [...zeros(15), ...repeat(5, COLD)]
    },
    {
      name: 'intensifying cold',
      expected: categories.INTENSIFYING_COLD,
      values: ramp(-24, -48)
    },
    {
      name: 'persistent cold',
      expected: categories.PERSISTENT_COLD,
      values: zigzag(COLD)
    },
    {
      name: 'diminishing cold',
      expected: categories.DIMINISHING_COLD,
      values: ramp(-48, -24)
    },
    {
      name: 'sporadic cold',
      expected: categories.SPORADIC_COLD,
      values: [COLD, ...zeros(4), COLD, ...zeros(3), COLD, ...zeros(2), COLD, ...zeros(6), COLD, 0]
    },
    {
      name: 'oscillating cold',
      expected: categories.OSCILLATING_COLD,
      values: [HOT, COLD, ...zeros(5), HOT, ...zeros(4), COLD, ...zeros(5), COLD, COLD]
    },
    {
      name: 'historical cold',
      expected: categories.HISTORICAL_COLD,
      values: [...zigzag(COLD).slice(0, 19), 0]
    },
    {
      name: 'no pattern flat',
      expected: categories.NO_PATTERN,
      values: zeros(20)
    },
    {
      name: 'no pattern mixed history',
      expected: categories.NO_PATTERN,
      values: [HOT, ...zeros(8), COLD, ...zeros(10)]
    }
  ];
  return series;
}

/**
 * Designed 24 x 24 x 20 cube: one designed series per cell, then masked, all-NaN and many noise
 * cells. The large noise background keeps the designed bins' z far above the level a
 * Benjamini-Hochberg correction over all valid bins requires.
 */
export function createDesignedCube(): {
  cube: EmergingHotSpotCube & {mask: Uint32Array; values: Float32Array};
  designed: DesignedSeries[];
  expectedCategories: number[];
} {
  const designed = createDesignedSeries();
  const shape = {gridWidth: 24, gridHeight: 24, sliceCount: 20};
  const cellCount = shape.gridWidth * shape.gridHeight;
  const values = new Float32Array(cellCount * shape.sliceCount);
  const mask = new Uint32Array(cellCount).fill(1);
  const random = createSeededRandom(11);
  const expectedCategories: number[] = [];
  for (let cell = 0; cell < cellCount; cell++) {
    const series = designed[cell];
    for (let slice = 0; slice < shape.sliceCount; slice++) {
      values[cell * shape.sliceCount + slice] =
        series && cell < designed.length ? series.values[slice] : random() * 2 - 1;
    }
    expectedCategories.push(series?.expected ?? GPU_EMERGING_HOT_SPOT_CATEGORIES.NO_PATTERN);
  }
  // A hot series that the mask removes.
  mask[designed.length] = 0;
  for (let slice = 0; slice < shape.sliceCount; slice++) {
    values[designed.length * shape.sliceCount + slice] = 30;
  }
  // A cell whose bins are all missing.
  values.fill(
    NaN,
    (designed.length + 1) * shape.sliceCount,
    (designed.length + 2) * shape.sliceCount
  );
  return {cube: {...shape, values, mask}, designed, expectedCategories};
}

// Multiple-testing reference. 100 bins at radius 0 and window 0, so the Gi* z of a bin is its
// population standard score; bins 92..97 are the strong (92-94) and weak (95-97) outliers.
// p = 2 * scipy.stats.norm.sf(|z|); adjusted values from statsmodels 0.15.0 multipletests
// (bonferroni, holm, fdr_bh, fdr_by). Script: scratchpad/fix/work/B/b5_emerging_ref.py
// (numpy 2.5.4, scipy 1.18.1, statsmodels 0.15.0).
export const MULTIPLE_TESTING_VALUES = [
  0.0, 0.3, -0.3, -0.9, -0.5, -1.0, 0.1, 1.3, -0.5, -0.6, 0.5, 0.4, 0.1, -0.9, -0.0, 0.7, -1.3,
  -0.5, -1.9, -1.3, -1.8, -0.2, -1.3, 0.3, 0.2, -0.2, -2.5, -0.5, -0.0, 0.1, -1.5, -0.5, -1.0, -0.8,
  1.1, -0.8, -0.0, 0.9, -0.6, -0.1, 0.1, 0.1, -1.2, 0.1, 1.4, -1.5, 0.9, 0.1, -0.6, 2.0, 0.8, -1.2,
  0.1, 0.6, -0.2, 0.7, -0.1, 0.7, 1.4, -0.7, 0.2, -0.5, 0.1, -1.2, -0.6, -0.2, 0.9, 1.1, -1.3, -0.8,
  0.6, -2.0, -0.5, -0.1, 1.3, 0.7, -0.3, -0.4, -0.3, 1.5, -0.4, -0.3, 0.4, -0.1, -0.2, -1.1, -0.0,
  -0.4, 1.2, 0.7, -0.0, 0.7, 7.0, 6.0, -6.5, 3.4, -3.3, 3.1, 0.0, 0.0
];
export const MULTIPLE_TESTING_TOP_BINS = [92, 93, 94, 95, 96, 97];
export const MULTIPLE_TESTING_P = [
  3.2348898651709686e-6, 6.50581295177076e-5, 1.895532573171461e-5, 0.023001976038730457,
  0.03080369422046859, 0.03799494754735993
];
export const MULTIPLE_TESTING_ADJUSTED: Record<string, number[]> = {
  bonferroni: [0.0003234889865170969, 0.006505812951770759, 0.001895532573171461, 1.0, 1.0, 1.0],
  holm: [0.0003234889865170969, 0.006375696692735344, 0.0018765772474397461, 1.0, 1.0, 1.0],
  'benjamini-hochberg': [
    0.0003234889865170969, 0.0021686043172569197, 0.0009477662865857304, 0.5750494009682614,
    0.6160738844093718, 0.6332491257893322
  ],
  'benjamini-yekutieli': [
    0.0016780594958628148, 0.011249369279994764, 0.004916421527011608, 1.0, 1.0, 1.0
  ]
};
// Bins significant at an adjusted p of 0.05 (statsmodels).
export const MULTIPLE_TESTING_SIGNIFICANT: Record<string, number[]> = {
  none: [92, 93, 94, 95, 96, 97],
  bonferroni: [92, 93, 94],
  holm: [92, 93, 94],
  'benjamini-hochberg': [92, 93, 94],
  'benjamini-yekutieli': [92, 93, 94]
};

// Time-step slicing reference: python datetime / calendar arithmetic, independent of the
// days_from_civil formula. Script: scratchpad/fix/work/B/b5_emerging_ref.py.
export const SLICE_EVENTS = [
  [2022, 10, 31, 23],
  [2022, 11, 1, 0],
  [2022, 12, 25, 23],
  [2022, 12, 26, 0],
  [2022, 12, 31, 23],
  [2023, 1, 1, 0],
  [2023, 1, 1, 23],
  [2023, 1, 2, 0],
  [2023, 2, 19, 12],
  [2023, 2, 20, 0],
  [2023, 4, 30, 12],
  [2023, 5, 1, 0],
  [2023, 12, 29, 12],
  [2023, 12, 30, 0],
  [2023, 12, 31, 23],
  [2024, 1, 1, 0],
  [2024, 2, 28, 23],
  [2024, 2, 29, 5],
  [2024, 3, 1, 5]
];
export const SLICE_CASES = [
  {
    step: 'month',
    interval: 1,
    origin: [2022, 11],
    count: 6,
    expected: [-1, 0, 1, 1, 1, 2, 2, 2, 3, 3, 5, -1, -1, -1, -1, -1, -1, -1, -1]
  },
  {
    step: 'month',
    interval: 2,
    origin: [2022, 11],
    count: 3,
    expected: [-1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1]
  },
  {
    step: 'week',
    interval: 1,
    origin: [2022, 12, 26],
    count: 8,
    expected: [-1, -1, -1, 0, 0, 0, 0, 1, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
  },
  {
    step: 'week',
    interval: 2,
    origin: [2022, 12, 26],
    count: 4,
    expected: [-1, -1, -1, 0, 0, 0, 0, 0, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
  },
  {
    step: 'day',
    interval: 1,
    origin: [2023, 12, 30],
    count: 100,
    expected: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 60, 61, 62]
  },
  {
    step: 'quarter',
    interval: 1,
    origin: [2022, 10],
    count: 4,
    expected: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, -1, -1, -1, -1, -1, -1, -1]
  },
  {
    step: 'year',
    interval: 1,
    origin: [2022],
    count: 3,
    expected: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
  },
  {
    step: 'hour',
    interval: 6,
    origin: [2024, 2, 29, 0],
    count: 20,
    expected: [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4]
  }
] as {
  step: 'year' | 'quarter' | 'month' | 'week' | 'day' | 'hour';
  interval: number;
  origin: number[];
  count: number;
  expected: number[];
}[];
