// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeClassBreaksOracle,
  computeExactNaturalBreaks,
  getWithinClassCost
} from './class-breaks-oracle';
import {
  addWide,
  computeNaturalBreaksMirror,
  fromWide,
  getClassCost,
  getExactClassCost,
  multiplyWide,
  multiplyWideLow,
  subtractWide,
  toWide
} from './natural-breaks-mirror';

// The CPU mirror of the GPU natural-breaks algorithm (natural-breaks-mirror.ts) against exact
// references: BigInt arithmetic for the integer helpers, and an f64 O(k n^2) Fisher-Jenks dynamic
// program for the breaks. The fixture comparison with mapclassify and jenkspy is in
// class-breaks-reference.node.spec.ts.

/** Deterministic xorshift in [0, 1). */
function createRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 4294967296;
  };
}

function getNormal(random: () => number): number {
  return Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
}

function getRange(values: Float32Array): [number, number] {
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const value of values) {
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
  }
  return [minimum, maximum];
}

type Kind =
  | 'normal'
  | 'lognormal'
  | 'uniform'
  | 'mixture'
  | 'ties'
  | 'pareto'
  | 'outliers'
  | 'bimodal-outlier';

function createColumn(kind: Kind, random: () => number, rows: number): Float32Array {
  const values = new Float32Array(rows);
  for (let row = 0; row < rows; row++) {
    switch (kind) {
      case 'normal':
        values[row] = getNormal(random) * 3 + 7;
        break;
      case 'lognormal':
        values[row] = Math.exp(getNormal(random));
        break;
      case 'uniform':
        values[row] = random() * 100;
        break;
      case 'mixture':
        values[row] = [5, 40, 90][Math.floor(random() * 3)] + getNormal(random) * 2;
        break;
      case 'ties':
        values[row] = Math.floor(random() * 12);
        break;
      case 'pareto':
        values[row] = (1 - random()) ** (-1 / 1.2);
        break;
      case 'outliers':
        values[row] = row < 3 ? 1e4 * (row + 1) * (random() + 0.5) : getNormal(random);
        break;
      default:
        values[row] = row === 0 ? 1e7 : (random() < 0.5 ? 0 : 50) + getNormal(random);
    }
  }
  return values;
}

it('wide integer helpers equal BigInt arithmetic', () => {
  const random = createRandom(7);
  const randomWord = () => Math.floor(random() * 4294967296) >>> 0;
  const specials = [0, 1, 0xffff, 0x10000, 0x7fffffff, 0x80000000, 0xffffffff];
  const words = [...specials, ...Array.from({length: 200}, randomWord)];
  for (const left of words) {
    for (const right of specials) {
      expect(fromWide(multiplyWide(left, right)), `${left} * ${right}`).toBe(
        BigInt(left) * BigInt(right)
      );
    }
  }
  for (let trial = 0; trial < 2000; trial++) {
    const left = toWide((BigInt(randomWord()) << 32n) | BigInt(randomWord()));
    const right = toWide((BigInt(randomWord()) << 32n) | BigInt(randomWord()));
    const modulus = 1n << 64n;
    expect(fromWide(addWide(left, right))).toBe((fromWide(left) + fromWide(right)) % modulus);
    expect(fromWide(subtractWide(left, right))).toBe(
      (fromWide(left) - fromWide(right) + modulus) % modulus
    );
    expect(fromWide(multiplyWideLow(left, right))).toBe(
      (fromWide(left) * fromWide(right)) % modulus
    );
    expect(fromWide(multiplyWide(left[0], right[0]))).toBe(BigInt(left[0]) * BigInt(right[0]));
  }
});

it('class cost keeps f32 precision where S2 - S1^2 / n cancels', () => {
  // Derived bound: the recentred sum of squares S2c converts to f32 in two roundings (hi * 2^32 +
  // lo), the correction (S1c^2 / n) has three, and cost >= 0.35 * S2c (S1c is within 0.65 n of
  // the recentred mean and (x - m)^2 >= |x - m| for integers), so the relative error is below
  // (2 + 3 * 0.65) / 0.35 * 2^-24 < 12 * 2^-24 = 7.2e-7.
  const tolerance = 12 * 2 ** -24;
  const random = createRandom(11);
  let worst = 0;
  for (let trial = 0; trial < 3000; trial++) {
    const count = 1 + Math.floor(random() * 2 ** (1 + Math.floor(random() * 23)));
    const offset = Math.floor(random() * 2 ** 18) - 2 ** 17;
    // Spread from one step to the whole 18-bit range, around an offset: S2 is huge and S2 - S1^2/n
    // tiny, the case that destroys an f32 evaluation.
    const spread = Math.max(1, Math.floor(2 ** (random() * 17)));
    let sum = 0n;
    let squares = 0n;
    const members = Math.min(count, 400);
    const weight = Math.floor(count / members);
    let total = 0;
    for (let member = 0; member < members; member++) {
      const position = BigInt(offset + Math.floor((random() - 0.5) * spread));
      sum += position * BigInt(weight);
      squares += position * position * BigInt(weight);
      total += weight;
    }
    const exact = getExactClassCost(total, sum, squares);
    const actual = getClassCost(total, toWide(sum), toWide(squares));
    if (exact > 0) {
      worst = Math.max(worst, Math.abs(actual - exact) / exact);
    } else {
      expect(actual).toBe(0);
    }
  }
  expect(worst).toBeLessThan(tolerance);
});

it('coarse dynamic program equals the brute-force optimum over the bins', () => {
  const random = createRandom(23);
  for (const kind of ['normal', 'lognormal', 'mixture', 'pareto', 'ties'] as const) {
    const values = createColumn(kind, random, 300);
    const [minimum, maximum] = getRange(values);
    const binCount = 96;
    for (const classCount of [2, 3, 5, 6]) {
      const mirror = computeNaturalBreaksMirror(values, minimum, maximum, classCount, binCount);
      // Exact positions sums per bin -> exact cost of any run of bins, in BigInt.
      const prefixCount = [0];
      const prefixSum = [0n];
      const prefixSquares = [0n];
      for (let bin = 0; bin < binCount; bin++) {
        prefixCount.push(prefixCount[bin] + mirror.histogram[bin]);
        prefixSum.push(prefixSum[bin] + mirror.binSums[bin]);
        prefixSquares.push(prefixSquares[bin] + mirror.binSquares[bin]);
      }
      const getCost = (first: number, last: number) =>
        getExactClassCost(
          prefixCount[last + 1] - prefixCount[first],
          prefixSum[last + 1] - prefixSum[first],
          prefixSquares[last + 1] - prefixSquares[first]
        );
      // Brute force: best[c][j] over all starts.
      let previous = Array.from({length: binCount}, (_, end) => getCost(0, end));
      for (let layer = 1; layer < classCount; layer++) {
        const current = new Array<number>(binCount).fill(Infinity);
        for (let end = layer; end < binCount; end++) {
          for (let start = layer; start <= end; start++) {
            current[end] = Math.min(current[end], previous[start - 1] + getCost(start, end));
          }
        }
        previous = current;
      }
      const optimum = previous[binCount - 1];
      const bounds = [0, ...mirror.coarseStarts, binCount];
      let chosen = 0;
      for (let index = 0; index + 1 < bounds.length; index++) {
        chosen += getCost(bounds[index], bounds[index + 1] - 1);
      }
      // f32 costs may pick a different partition only within their relative precision.
      expect(chosen, `${kind} k=${classCount}`).toBeLessThanOrEqual(optimum * (1 + 1e-5) + 1e-6);
    }
  }
});

it('natural breaks equal exact Fisher-Jenks on well-scaled random columns', () => {
  const random = createRandom(101);
  let compared = 0;
  const failures: string[] = [];
  for (let trial = 0; trial < 60; trial++) {
    const kind = (['normal', 'uniform', 'mixture', 'ties', 'lognormal'] as const)[trial % 5];
    const rows = 100 + Math.floor(random() * 500);
    const classCount = 2 + Math.floor(random() * 9);
    const values = createColumn(kind, random, rows);
    const exact = computeExactNaturalBreaks(values, classCount);
    if (!exact.unique || new Set(values).size < classCount) {
      continue;
    }
    const [minimum, maximum] = getRange(values);
    const mirror = computeNaturalBreaksMirror(values, minimum, maximum, classCount, 2048);
    compared++;
    const expected = exact.edges.map(Math.fround);
    if (
      mirror.edges.length !== expected.length ||
      mirror.edges.some((edge, index) => edge !== expected[index])
    ) {
      failures.push(`${kind} rows=${rows} k=${classCount}`);
    }
  }
  expect(compared).toBeGreaterThan(40);
  // Flat cost landscapes can leave the coordinate descent in a neighbouring local optimum; on
  // these well-scaled columns that happened for 0 of 60 seeds when this was written.
  expect(failures).toEqual([]);
});

it('a single extreme outlier collapses the bins but the bulk is still split exactly', () => {
  // With the outlier 1e5 or more times the spread, the whole bulk sits in one bin: the coarse
  // program leaves empty classes, and the coarse stage runs again over the range of the bulk.
  const random = createRandom(5);
  let compared = 0;
  for (const outlier of [1e5, 1e7]) {
    for (const classCount of [2, 3, 4, 5, 6, 7, 8]) {
      const values = createColumn('normal', random, 300);
      values[17] = outlier;
      const exact = computeExactNaturalBreaks(values, classCount);
      if (!exact.unique) {
        continue;
      }
      const [minimum, maximum] = getRange(values);
      const mirror = computeNaturalBreaksMirror(values, minimum, maximum, classCount, 2048);
      expect(mirror.edges, `outlier ${outlier} k=${classCount}`).toEqual(
        exact.edges.map(Math.fround)
      );
      compared++;
    }
  }
  expect(compared).toBeGreaterThan(10);
});

it('a moderate outlier keeps the goodness of fit of the exact partition', () => {
  const random = createRandom(6);
  let worst = 0;
  for (const outlier of [1e2, 1e3, 1e4]) {
    for (const classCount of [2, 3, 4, 5, 6, 7, 8]) {
      const values = createColumn('normal', random, 300);
      values[17] = outlier;
      const exact = computeExactNaturalBreaks(values, classCount);
      const [minimum, maximum] = getRange(values);
      const mirror = computeNaturalBreaksMirror(values, minimum, maximum, classCount, 2048);
      expect(mirror.classCount).toBe(classCount);
      const gap =
        (getWithinClassCost(values, mirror.edges) - getWithinClassCost(values, exact.edges)) /
        getWithinClassCost(values, []);
      worst = Math.max(worst, gap);
    }
  }
  expect(worst).toBeLessThan(1e-5);
});

it('multi-scale columns keep the goodness of fit within 1e-5 of exact', () => {
  // Several clusters inside one equal-width bin, a Pareto tail and several outliers. A sweep of
  // 180 random columns of these kinds (k from 3 to 9) gave 179 exact partitions and a largest
  // goodness of variance fit shortfall of 4e-8; 1e-5 leaves a wide margin for other seeds.
  const random = createRandom(303);
  let worst = 0;
  for (let trial = 0; trial < 36; trial++) {
    const kind = (['pareto', 'outliers', 'bimodal-outlier'] as const)[trial % 3];
    const values = createColumn(kind, random, 150 + Math.floor(random() * 350));
    const classCount = 3 + Math.floor(random() * 7);
    const exact = computeExactNaturalBreaks(values, classCount);
    const [minimum, maximum] = getRange(values);
    const mirror = computeNaturalBreaksMirror(values, minimum, maximum, classCount, 2048);
    const total = getWithinClassCost(values, []);
    const gap =
      (getWithinClassCost(values, mirror.edges) - getWithinClassCost(values, exact.edges)) / total;
    worst = Math.max(worst, gap);
    expect(mirror.classCount, kind).toBe(classCount);
  }
  expect(worst).toBeLessThan(1e-5);
});

it('natural breaks handle ties, constant columns, k above the distinct values and k = 2', () => {
  const run = (values: number[], classCount: number, mask?: number[]) => {
    const result = computeClassBreaksOracle({
      values: Float32Array.from(values),
      mask: mask ? Uint32Array.from(mask) : undefined,
      parameters: {method: 'natural-breaks', classCount},
      maximumClassCount: 10,
      naturalBreaksBinCount: 256
    });
    return {
      classCount: result.classCount,
      edges: Array.from(result.breaks.slice(0, result.classCount + 1))
    };
  };
  // One class per distinct value when there are fewer values than requested classes.
  expect(run([5, 1, 1, 1, 9, 9, 5, 5], 6)).toEqual({classCount: 3, edges: [1, 5, 9, 9]});
  expect(run([2, 2, 3, 3, 3], 4)).toEqual({classCount: 2, edges: [2, 3, 3]});
  // A constant column is one class, whatever was requested.
  expect(run([4, 4, 4, 4], 5)).toEqual({classCount: 1, edges: [4, 4]});
  expect(run([4], 3)).toEqual({classCount: 1, edges: [4, 4]});
  // k = 2 on ties: the exact split minimises the within-class cost.
  const tied = [1, 1, 1, 2, 2, 5, 5, 5, 5, 9];
  const two = run(tied, 2);
  expect(two.edges).toEqual([1, 5, 9]);
  const exact = computeExactNaturalBreaks(tied, 2);
  expect(exact.edges).toEqual([5]);
  // k = 1.
  expect(run(tied, 1)).toEqual({classCount: 1, edges: [1, 9]});
  // NaN, infinities and masked rows are missing: the result equals that of the remaining column.
  const clean = [3, 8, 8.5, 9, 30, 31, 33, 100, 101, 140];
  const noisy = [NaN, 3, 8, Infinity, 8.5, 9, -Infinity, 30, 31, 33, 7777, 100, 101, 140, NaN];
  const mask = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1];
  for (const classCount of [2, 3, 4]) {
    expect(run(noisy, classCount, mask)).toEqual(run(clean, classCount));
  }
});

it('natural breaks are deterministic and independent of row order', () => {
  const random = createRandom(77);
  const values = createColumn('mixture', random, 400);
  const [minimum, maximum] = getRange(values);
  const first = computeNaturalBreaksMirror(values, minimum, maximum, 6, 512);
  const reversed = Float32Array.from(values).reverse();
  const second = computeNaturalBreaksMirror(reversed, minimum, maximum, 6, 512);
  expect(second.edges).toEqual(first.edges);
  expect(second.coarseStarts).toEqual(first.coarseStarts);
});

it('refinement stops early on settled columns', () => {
  const random = createRandom(9);
  const values = createColumn('normal', random, 500);
  const [minimum, maximum] = getRange(values);
  const mirror = computeNaturalBreaksMirror(values, minimum, maximum, 5, 2048);
  // Extents pass plus two rounds is the least a settled run needs; the cap is eight rounds.
  expect(mirror.passCount).toBeGreaterThanOrEqual(5);
  expect(mirror.passCount).toBeLessThanOrEqual(1 + 2 * 8);
});
