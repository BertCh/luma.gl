// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getClassBreaks} from '../../cartography/breaks';

/**
 * Pure CPU helpers of the group-statistics scene: the unfiltered reference table that freezes the
 * class breaks, the hour histogram of one area for the rose chart, Spearman's rank correlation and
 * the clock formatter. No luma.gl, no deck.gl.
 */

/** Key of a row that belongs to no area (the all-ones key `GPUGroupStatistics` skips). */
export const NO_KEY = 0xffffffff;

/** The three row filters of the scene, as the mask kernel receives them. */
export type RowFilter = {
  /** Category code, or {@link NO_KEY} for every group. */
  category: number;
  /** First hour kept. */
  hourLow: number;
  /** First hour not kept. */
  hourHigh: number;
  /** 0 every grade, 1 research grade only, 2 not yet confirmed only. */
  gradeMode: 0 | 1 | 2;
};

/** Observation columns the hour histogram reads (one value per row). */
export type ObservationColumns = {
  hour: ArrayLike<number>;
  category: ArrayLike<number>;
  grade: ArrayLike<number>;
  /** Area index `0..76`, or {@link NO_KEY}. */
  key: ArrayLike<number>;
};

/**
 * Observations per hour of day for one area (`zone >= 0`) or for every area (`zone = -1`), under
 * the same filter the GPU mask applies. This is the data of the rose chart; the statistics on the
 * map come from the GPU.
 */
export function getHourCounts(
  columns: ObservationColumns,
  filter: RowFilter,
  zone: number
): Float64Array {
  const counts = new Float64Array(24);
  const rowCount = columns.hour.length;
  for (let row = 0; row < rowCount; row++) {
    const key = columns.key[row];
    if (key === NO_KEY || (zone >= 0 && key !== zone)) continue;
    if (filter.category !== NO_KEY && columns.category[row] !== filter.category) continue;
    const hour = columns.hour[row];
    if (hour < filter.hourLow || hour >= filter.hourHigh) continue;
    const confirmed = columns.grade[row] > 0.5;
    if ((filter.gradeMode === 1 && !confirmed) || (filter.gradeMode === 2 && confirmed)) continue;
    counts[hour]++;
  }
  return counts;
}

/** Inputs of {@link getReferenceTable}. */
export type ReferenceInputs = {
  areaCount: number;
  /** Community area `1..77` of every observation, 0 when none. */
  observationArea: ArrayLike<number>;
  researchGrade: ArrayLike<number>;
  introduced: ArrayLike<number>;
  species: ArrayLike<number>;
  /** Community area `1..77` of every tract. */
  tractArea: ArrayLike<number>;
  population: ArrayLike<number>;
  perCapitaIncome: ArrayLike<number>;
};

/** Per-area values of the unfiltered table (one entry per area). */
export type ReferenceTable = {
  counts: Float64Array;
  population: Float64Array;
  perThousand: Float64Array;
  /** Percent of the records. */
  researchShare: Float64Array;
  introducedShare: Float64Array;
  richness: Float64Array;
  /** Population-weighted per-capita income in dollars. */
  income: Float64Array;
};

/**
 * The unfiltered per-area table on the CPU. It exists only to freeze the class breaks (quantile
 * and equal interval) once, so a filter or a swipe changes the colour of an area, never the
 * classes. Every number on the map comes from the GPU.
 */
export function getReferenceTable(inputs: ReferenceInputs): ReferenceTable {
  const {areaCount} = inputs;
  const counts = new Float64Array(areaCount);
  const research = new Float64Array(areaCount);
  const introducedCounts = new Float64Array(areaCount);
  const taxa = Array.from({length: areaCount}, () => new Set<number>());
  for (let row = 0; row < inputs.observationArea.length; row++) {
    const area = inputs.observationArea[row];
    if (area < 1) continue;
    const zone = area - 1;
    counts[zone]++;
    research[zone] += inputs.researchGrade[row];
    introducedCounts[zone] += inputs.introduced[row];
    taxa[zone].add(inputs.species[row]);
  }
  const population = new Float64Array(areaCount);
  const incomeTimesPopulation = new Float64Array(areaCount);
  const knownPopulation = new Float64Array(areaCount);
  for (let tract = 0; tract < inputs.tractArea.length; tract++) {
    const area = inputs.tractArea[tract];
    if (area < 1) continue;
    const zone = area - 1;
    const residents = inputs.population[tract];
    population[zone] += residents;
    const income = inputs.perCapitaIncome[tract];
    if (Number.isFinite(income) && residents > 0) {
      incomeTimesPopulation[zone] += income * residents;
      knownPopulation[zone] += residents;
    }
  }
  const perThousand = new Float64Array(areaCount);
  const researchShare = new Float64Array(areaCount);
  const introducedShare = new Float64Array(areaCount);
  const richness = new Float64Array(areaCount);
  const income = new Float64Array(areaCount);
  for (let zone = 0; zone < areaCount; zone++) {
    perThousand[zone] = population[zone] > 0 ? (counts[zone] * 1000) / population[zone] : NaN;
    researchShare[zone] = counts[zone] > 0 ? (100 * research[zone]) / counts[zone] : NaN;
    introducedShare[zone] = counts[zone] > 0 ? (100 * introducedCounts[zone]) / counts[zone] : NaN;
    richness[zone] = taxa[zone].size;
    income[zone] =
      knownPopulation[zone] > 0 ? incomeTimesPopulation[zone] / knownPopulation[zone] : NaN;
  }
  return {counts, population, perThousand, researchShare, introducedShare, richness, income};
}

/** How the rate classes are drawn. */
export type Classification = 'manual' | 'quantile' | 'equal';

/** Frozen interior class breaks of every metric (five classes each). */
export type FrozenBreaks = {
  perThousand: Record<Classification, number[]>;
  researchShare: number[];
  introducedShare: number[];
  richness: number[];
  income: number[];
  /** `[minimum, maximum]` of the unfiltered values, for the outer legend labels. */
  extents: {
    perThousand: [number, number];
    researchShare: [number, number];
    introducedShare: [number, number];
    richness: [number, number];
    income: [number, number];
  };
};

/** The manual breaks of the rate map: roughly logarithmic, because the rates are heavy-tailed. */
export const MANUAL_RATE_BREAKS: readonly number[] = [1, 3, 10, 30];

/** Rounds to two significant figures and drops duplicates, so legend labels stay short. */
export function roundBreaks(breaks: readonly number[]): number[] {
  const rounded = breaks.map(value => Number(value.toPrecision(2)));
  return rounded.filter((value, index) => index === 0 || value > rounded[index - 1]);
}

/** Computes every break set once from the unfiltered table. */
export function getFrozenBreaks(reference: ReferenceTable): FrozenBreaks {
  const quantile = (values: ArrayLike<number>) =>
    roundBreaks(getClassBreaks(values, 5, 'quantile'));
  return {
    perThousand: {
      manual: [...MANUAL_RATE_BREAKS],
      quantile: quantile(reference.perThousand),
      equal: roundBreaks(getClassBreaks(reference.perThousand, 5, 'equal-interval'))
    },
    researchShare: quantile(reference.researchShare),
    introducedShare: quantile(reference.introducedShare),
    richness: quantile(reference.richness),
    income: quantile(reference.income),
    extents: {
      perThousand: getFiniteExtent(reference.perThousand),
      researchShare: getFiniteExtent(reference.researchShare),
      introducedShare: getFiniteExtent(reference.introducedShare),
      richness: getFiniteExtent(reference.richness),
      income: getFiniteExtent(reference.income)
    }
  };
}

function getFiniteExtent(values: ArrayLike<number>): [number, number] {
  let minimum = Number.POSITIVE_INFINITY;
  let maximum = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < values.length; index++) {
    const value = values[index];
    if (!Number.isFinite(value)) continue;
    if (value < minimum) minimum = value;
    if (value > maximum) maximum = value;
  }
  return Number.isFinite(minimum) ? [minimum, maximum] : [0, 1];
}

/**
 * Spearman's rank correlation of two equally long arrays, with average ranks for ties. Returns
 * NaN for fewer than three pairs or a constant array.
 */
export function getSpearmanCorrelation(x: ArrayLike<number>, y: ArrayLike<number>): number {
  const n = Math.min(x.length, y.length);
  if (n < 3) return NaN;
  const xRanks = getAverageRanks(x, n);
  const yRanks = getAverageRanks(y, n);
  let meanX = 0;
  let meanY = 0;
  for (let i = 0; i < n; i++) {
    meanX += xRanks[i];
    meanY += yRanks[i];
  }
  meanX /= n;
  meanY /= n;
  let covariance = 0;
  let varianceX = 0;
  let varianceY = 0;
  for (let i = 0; i < n; i++) {
    const dx = xRanks[i] - meanX;
    const dy = yRanks[i] - meanY;
    covariance += dx * dy;
    varianceX += dx * dx;
    varianceY += dy * dy;
  }
  return varianceX > 0 && varianceY > 0 ? covariance / Math.sqrt(varianceX * varianceY) : NaN;
}

function getAverageRanks(values: ArrayLike<number>, n: number): Float64Array {
  const order = Array.from({length: n}, (_, index) => index).sort((a, b) => values[a] - values[b]);
  const ranks = new Float64Array(n);
  let start = 0;
  while (start < n) {
    let end = start;
    while (end + 1 < n && values[order[end + 1]] === values[order[start]]) end++;
    const rank = (start + end) / 2;
    for (let i = start; i <= end; i++) ranks[order[i]] = rank;
    start = end + 1;
  }
  return ranks;
}

/** Indexes of `values` sorted from the largest to the smallest; NaN sorts last. */
export function getDescendingOrder(values: ArrayLike<number>): number[] {
  return Array.from({length: values.length}, (_, index) => index).sort((a, b) => {
    const aFinite = Number.isFinite(values[a]);
    const bFinite = Number.isFinite(values[b]);
    if (!aFinite || !bFinite) return aFinite === bFinite ? a - b : aFinite ? -1 : 1;
    return values[b] - values[a] || a - b;
  });
}

/** Median of the finite entries of an array (NaN when there are none). */
export function getMedian(values: ArrayLike<number>): number {
  const finite = Array.from(values).filter(Number.isFinite);
  if (finite.length === 0) return NaN;
  finite.sort((a, b) => a - b);
  const middle = (finite.length - 1) / 2;
  return (finite[Math.floor(middle)] + finite[Math.ceil(middle)]) / 2;
}

/** Percentile rank of `value` among the sorted finite `values`, as a share in `[0, 1]`. */
export function getPercentileRank(sorted: ArrayLike<number>, value: number): number {
  if (sorted.length === 0 || !Number.isFinite(value)) return NaN;
  let low = 0;
  let high = sorted.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (sorted[middle] <= value) low = middle + 1;
    else high = middle;
  }
  return low / sorted.length;
}

/** Hour of day as a clock time: `13.4` gives `13:24`. */
export function formatClockHour(hour: number): string {
  if (!Number.isFinite(hour)) return 'n/a';
  const minutes = Math.round(hour * 60);
  const wrapped = ((minutes % 1440) + 1440) % 1440;
  return `${Math.floor(wrapped / 60)}:${String(wrapped % 60).padStart(2, '0')}`;
}
