// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Data classification for choropleths and graduated symbols. Pure TypeScript.
 *
 * Break convention (matches the GPU layers' `classBreaks` and the `classes` legend): breaks are
 * interior thresholds in ascending order, `n` classes have `n - 1` breaks, and the class of a value
 * is the number of breaks `<= value` (a value equal to a break falls in the upper class).
 */

/** Supported classification schemes. */
export type ClassificationMethod =
  | 'equal-interval'
  | 'quantile'
  | 'natural-breaks'
  | 'standard-deviation'
  | 'pretty'
  | 'head-tail';

/** Short label and one-sentence plain-language help per classification method. */
export const CLASSIFICATION_METHOD_INFO: Record<
  ClassificationMethod,
  {label: string; help: string}
> = {
  'equal-interval': {
    label: 'Equal interval',
    help: 'Splits the value range into equal-width steps; easy to read, but skewed data leaves most classes empty.'
  },
  quantile: {
    label: 'Quantile',
    help: 'Puts the same number of features in every class; the map looks balanced but hides how large the gaps between values are.'
  },
  'natural-breaks': {
    label: 'Natural breaks (Jenks)',
    help: 'Places breaks at the biggest gaps so values within a class are as alike as possible; best for clustered data, but breaks are hard to compare between maps.'
  },
  'standard-deviation': {
    label: 'Standard deviation',
    help: 'Classes are one standard deviation wide around the mean; shows who is above or below average, and misleads on strongly skewed data.'
  },
  pretty: {
    label: 'Pretty (round numbers)',
    help: 'Uses round break values such as 10, 20, 50 so the legend reads easily; the class sizes follow the numbers, not the data.'
  },
  'head-tail': {
    label: 'Head/tail breaks',
    help: 'Repeatedly splits at the mean to expose a long tail of few large values among many small ones; made for heavy-tailed data and poor for symmetric data.'
  }
};

/** Returns the finite values as an ascending Float64Array. */
function getSortedFinite(values: ArrayLike<number>): Float64Array {
  let count = 0;
  for (let i = 0; i < values.length; i++) {
    if (Number.isFinite(values[i])) count++;
  }
  const sorted = new Float64Array(count);
  let cursor = 0;
  for (let i = 0; i < values.length; i++) {
    if (Number.isFinite(values[i])) sorted[cursor++] = values[i];
  }
  return sorted.sort();
}

/** Sorted, strictly ascending, finite breaks. */
function cleanBreaks(breaks: number[]): number[] {
  const sorted = breaks.filter(Number.isFinite).sort((a, b) => a - b);
  return sorted.filter((value, index) => index === 0 || value > sorted[index - 1]);
}

/** Removes floating point dust (`0.30000000000000004`) from a computed break. */
function tidy(value: number): number {
  return Number(value.toPrecision(12));
}

/** Minimum and maximum of the finite values; `[NaN, NaN]` when there are none. */
export function getExtent(values: ArrayLike<number>): [number, number] {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const value = values[i];
    if (!Number.isFinite(value)) continue;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  return min > max ? [NaN, NaN] : [min, max];
}

/** Equal-width classes over `[min, max]`. Returns `[]` for constant or empty data. */
export function getEqualIntervalBreaks(values: ArrayLike<number>, classCount: number): number[] {
  const [min, max] = getExtent(values);
  if (!(max > min) || classCount < 2) return [];
  const step = (max - min) / classCount;
  const breaks: number[] = [];
  for (let i = 1; i < classCount; i++) breaks.push(tidy(min + i * step));
  return cleanBreaks(breaks);
}

/**
 * Quantile breaks with linear interpolation (R type 7). Ties collapse: duplicate breaks are removed,
 * so heavily tied data returns fewer than `classCount - 1` breaks (fewer classes).
 */
export function getQuantileBreaks(values: ArrayLike<number>, classCount: number): number[] {
  const sorted = getSortedFinite(values);
  const n = sorted.length;
  if (n === 0 || classCount < 2) return [];
  const breaks: number[] = [];
  for (let i = 1; i < classCount; i++) {
    const position = ((n - 1) * i) / classCount;
    const lower = Math.floor(position);
    const upper = Math.min(n - 1, lower + 1);
    breaks.push(tidy(sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower)));
  }
  return cleanBreaks(breaks);
}

/**
 * Fisher-Jenks optimal 1-D classification (dynamic programming, O(k n^2) time). For more than
 * `sampleSize` values (default 3000) it runs on a deterministic, evenly strided sample of the sorted
 * values (always including the minimum and maximum), so the breaks are approximate but repeatable.
 * Returns fewer breaks when there are fewer distinct values than classes.
 */
export function getNaturalBreaks(
  values: ArrayLike<number>,
  classCount: number,
  options: {sampleSize?: number} = {}
): number[] {
  let sorted = getSortedFinite(values);
  const sampleSize = Math.max(2, options.sampleSize ?? 3000);
  if (sorted.length > sampleSize) {
    const sample = new Float64Array(sampleSize);
    for (let i = 0; i < sampleSize; i++) {
      sample[i] = sorted[Math.round((i * (sorted.length - 1)) / (sampleSize - 1))];
    }
    sorted = sample;
  }
  const n = sorted.length;
  let distinct = 0;
  for (let i = 0; i < n; i++) if (i === 0 || sorted[i] !== sorted[i - 1]) distinct++;
  const k = Math.min(Math.floor(classCount), distinct);
  if (k < 2) return [];

  const width = k + 1;
  const lowerLimits = new Int32Array((n + 1) * width);
  const costs = new Float64Array((n + 1) * width).fill(Infinity);
  for (let j = 1; j <= k; j++) {
    lowerLimits[1 * width + j] = 1;
    costs[1 * width + j] = 0;
  }
  for (let l = 2; l <= n; l++) {
    let sum = 0;
    let sumSquares = 0;
    let count = 0;
    let variance = 0;
    for (let m = 1; m <= l; m++) {
      const lowerClassLimit = l - m + 1;
      const value = sorted[lowerClassLimit - 1];
      count++;
      sum += value;
      sumSquares += value * value;
      variance = sumSquares - (sum * sum) / count;
      const previous = lowerClassLimit - 1;
      if (previous !== 0) {
        for (let j = 2; j <= k; j++) {
          const candidate = variance + costs[previous * width + j - 1];
          if (costs[l * width + j] >= candidate) {
            lowerLimits[l * width + j] = lowerClassLimit;
            costs[l * width + j] = candidate;
          }
        }
      }
    }
    lowerLimits[l * width + 1] = 1;
    costs[l * width + 1] = variance;
  }

  const breaks: number[] = [];
  let end = n;
  for (let j = k; j >= 2; j--) {
    const lowerClassLimit = lowerLimits[end * width + j];
    breaks.push(sorted[lowerClassLimit - 1]);
    end = lowerClassLimit - 1;
  }
  return cleanBreaks(breaks);
}

/**
 * Standard-deviation classes, one sigma wide, centred on the mean. An odd `classCount` has a middle
 * class straddling the mean; an even count has the mean as a break. Breaks outside the data range
 * are dropped, so skewed data returns fewer breaks.
 */
export function getStandardDeviationBreaks(
  values: ArrayLike<number>,
  classCount: number
): number[] {
  const sorted = getSortedFinite(values);
  const n = sorted.length;
  if (n < 2 || classCount < 2) return [];
  let mean = 0;
  for (let i = 0; i < n; i++) mean += sorted[i];
  mean /= n;
  let squares = 0;
  for (let i = 0; i < n; i++) squares += (sorted[i] - mean) ** 2;
  const deviation = Math.sqrt(squares / n);
  if (!(deviation > 0)) return [];
  const min = sorted[0];
  const max = sorted[n - 1];
  const breaks: number[] = [];
  for (let i = 1; i < classCount; i++) {
    const value = mean + (i - classCount / 2) * deviation;
    if (value > min && value <= max) breaks.push(value);
  }
  return cleanBreaks(breaks);
}

/** Rounds a positive value onto the 1-2-2.5-5-10 series (nearest). */
function getPrettyStep(rough: number): number {
  const exponent = Math.floor(Math.log10(rough));
  const magnitude = 10 ** exponent;
  const fraction = rough / magnitude;
  let best = 1;
  for (const candidate of [1, 2, 2.5, 5, 10]) {
    if (Math.abs(Math.log(candidate / fraction)) < Math.abs(Math.log(best / fraction))) {
      best = candidate;
    }
  }
  return best * magnitude;
}

/**
 * "Pretty" breaks: multiples of a round step (1, 2, 2.5 or 5 times a power of ten) giving about
 * `classCount` classes over `[min, max]`. The actual class count can differ by one or two.
 */
export function getPrettyBreaks(values: ArrayLike<number>, classCount: number): number[] {
  const [min, max] = getExtent(values);
  if (!(max > min) || classCount < 2) return [];
  const step = getPrettyStep((max - min) / classCount);
  const breaks: number[] = [];
  for (let i = Math.floor(min / step) + 1; i * step <= max; i++) {
    const value = tidy(i * step);
    if (value > min) breaks.push(value);
  }
  return cleanBreaks(breaks);
}

/**
 * Head/tail breaks (Jiang 2013): split at the mean, keep the head (values above the mean) while it
 * holds under 40% of the remaining values, up to `classCount - 1` breaks. Suited to heavy-tailed data.
 */
export function getHeadTailBreaks(values: ArrayLike<number>, classCount: number): number[] {
  let current = Array.from(getSortedFinite(values));
  const breaks: number[] = [];
  while (breaks.length < classCount - 1 && current.length > 1) {
    let mean = 0;
    for (const value of current) mean += value;
    mean /= current.length;
    const head = current.filter(value => value > mean);
    if (head.length === 0) break;
    breaks.push(mean);
    if (head.length / current.length >= 0.4) break;
    current = head;
  }
  return cleanBreaks(breaks);
}

/**
 * Class breaks for `values` by `method`. Ignores NaN and infinities. May return fewer than
 * `classCount - 1` breaks when the method cannot produce that many distinct ones (ties, few distinct
 * values, or a method that picks its own count such as `pretty`).
 */
export function getClassBreaks(
  values: ArrayLike<number>,
  classCount: number,
  method: ClassificationMethod,
  options: {sampleSize?: number} = {}
): number[] {
  switch (method) {
    case 'equal-interval':
      return getEqualIntervalBreaks(values, classCount);
    case 'quantile':
      return getQuantileBreaks(values, classCount);
    case 'natural-breaks':
      return getNaturalBreaks(values, classCount, options);
    case 'standard-deviation':
      return getStandardDeviationBreaks(values, classCount);
    case 'pretty':
      return getPrettyBreaks(values, classCount);
    case 'head-tail':
      return getHeadTailBreaks(values, classCount);
    default:
      return [];
  }
}

/** Class index of `value`: the number of `breaks` that are `<= value` (binary search). */
export function getClassIndex(value: number, breaks: readonly number[]): number {
  let low = 0;
  let high = breaks.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (breaks[middle] <= value) low = middle + 1;
    else high = middle;
  }
  return low;
}

/** Number of values per class (length `breaks.length + 1`); non-finite values are ignored. */
export function getClassCounts(values: ArrayLike<number>, breaks: readonly number[]): number[] {
  const counts = new Array<number>(breaks.length + 1).fill(0);
  for (let i = 0; i < values.length; i++) {
    if (Number.isFinite(values[i])) counts[getClassIndex(values[i], breaks)]++;
  }
  return counts;
}

/** Histogram over `extent` with `binCount` equal bins; the maximum falls in the last bin, values outside are ignored. */
export function getHistogram(
  values: ArrayLike<number>,
  extent: readonly [number, number],
  binCount: number
): number[] {
  const bins = new Array<number>(Math.max(1, Math.floor(binCount))).fill(0);
  const [min, max] = extent;
  const span = max - min;
  for (let i = 0; i < values.length; i++) {
    const value = values[i];
    if (!Number.isFinite(value) || value < min || value > max) continue;
    const bin =
      span > 0 ? Math.min(bins.length - 1, Math.floor(((value - min) / span) * bins.length)) : 0;
    bins[bin]++;
  }
  return bins;
}

/**
 * Goodness of variance fit (GVF, 0 to 1): `1 - SDCM / SDAM`, the share of total variance explained by
 * the classes. 1 means classes capture all variation; Jenks maximises it for a given class count.
 */
export function getGoodnessOfVarianceFit(
  values: ArrayLike<number>,
  breaks: readonly number[]
): number {
  const classCount = breaks.length + 1;
  const sums = new Float64Array(classCount);
  const squares = new Float64Array(classCount);
  const counts = new Float64Array(classCount);
  let total = 0;
  let totalSquares = 0;
  let n = 0;
  for (let i = 0; i < values.length; i++) {
    const value = values[i];
    if (!Number.isFinite(value)) continue;
    const index = getClassIndex(value, breaks);
    sums[index] += value;
    squares[index] += value * value;
    counts[index]++;
    total += value;
    totalSquares += value * value;
    n++;
  }
  const sdam = totalSquares - (total * total) / n;
  if (!(sdam > 0)) return 1;
  let sdcm = 0;
  for (let c = 0; c < classCount; c++) {
    if (counts[c] > 0) sdcm += squares[c] - (sums[c] * sums[c]) / counts[c];
  }
  return Math.max(0, Math.min(1, 1 - sdcm / sdam));
}

/**
 * One label per class. With `extent`: `min–b0`, `b0–b1`, ..., `bN–max`. Without: `< b0`, `b0–b1`, ...,
 * `≥ bN`.
 */
export function formatBreakLabels(
  breaks: readonly number[],
  extent: readonly [number, number] | undefined,
  format: (value: number) => string
): string[] {
  if (breaks.length === 0) {
    return extent ? [`${format(extent[0])}–${format(extent[1])}`] : ['all values'];
  }
  const labels: string[] = [];
  labels.push(extent ? `${format(extent[0])}–${format(breaks[0])}` : `< ${format(breaks[0])}`);
  for (let i = 1; i < breaks.length; i++) {
    labels.push(`${format(breaks[i - 1])}–${format(breaks[i])}`);
  }
  const last = breaks[breaks.length - 1];
  labels.push(extent ? `${format(last)}–${format(extent[1])}` : `≥ ${format(last)}`);
  return labels;
}
