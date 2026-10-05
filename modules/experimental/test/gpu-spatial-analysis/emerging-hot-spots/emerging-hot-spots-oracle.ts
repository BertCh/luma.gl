// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_EMERGING_HOT_SPOT_CATEGORIES} from '../../../src/gpu-spatial-analysis/emerging-hot-spots';
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

/**
 * Pass 4 for one z series (see `GPUEmergingHotSpots` for the rules). `packed` is the float32
 * parameter array; thresholds use the same f32 arithmetic as the WGSL.
 */
export function classifyEmergingHotSpotSeries(
  series: ArrayLike<number>,
  trend: EmergingHotSpotTrend,
  packed: ArrayLike<number>
): number {
  const criticalZ = fround(packed[2]);
  const trendLevel = fround(packed[3]);
  const fraction = fround(packed[4]);
  let valid = 0;
  let hot = 0;
  let cold = 0;
  let trailingHot = 0;
  let trailingCold = 0;
  let finalState = 0;
  for (let index = 0; index < series.length; index++) {
    const z = series[index];
    if (!Number.isFinite(z)) {
      continue;
    }
    valid++;
    if (z >= criticalZ) {
      hot++;
      trailingHot++;
      trailingCold = 0;
      finalState = 1;
    } else if (z <= -criticalZ) {
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
};

/** Runs passes 3 and 4 over every cell of a z cube (for example the GPU's own `giZScores`). */
export function computeEmergingHotSpotCells(
  zScores: ArrayLike<number>,
  shape: EmergingHotSpotCubeShape,
  packed: ArrayLike<number>
): EmergingHotSpotCellResults {
  const cellCount = shape.gridWidth * shape.gridHeight;
  const criticalZ = fround(packed[2]);
  const results: EmergingHotSpotCellResults = {
    trendZ: [],
    trendP: [],
    trendS: [],
    category: [],
    hotSliceCount: [],
    coldSliceCount: []
  };
  for (let cell = 0; cell < cellCount; cell++) {
    const series = Array.from(
      {length: shape.sliceCount},
      (_, slice) => zScores[cell * shape.sliceCount + slice]
    );
    const trend = computeMannKendall(series);
    results.trendZ.push(trend.z);
    results.trendP.push(trend.p);
    results.trendS.push(trend.statistic);
    results.category.push(classifyEmergingHotSpotSeries(series, trend, packed));
    results.hotSliceCount.push(series.filter(z => Number.isFinite(z) && z >= criticalZ).length);
    results.coldSliceCount.push(series.filter(z => Number.isFinite(z) && z <= -criticalZ).length);
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

/** Designed 8 x 8 x 20 cube: one designed series per cell, then masked, all-NaN and noise cells. */
export function createDesignedCube(): {
  cube: EmergingHotSpotCube & {mask: Uint32Array; values: Float32Array};
  designed: DesignedSeries[];
  expectedCategories: number[];
} {
  const designed = createDesignedSeries();
  const shape = {gridWidth: 8, gridHeight: 8, sliceCount: 20};
  const cellCount = 64;
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
