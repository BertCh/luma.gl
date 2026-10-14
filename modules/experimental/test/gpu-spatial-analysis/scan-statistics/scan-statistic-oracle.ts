// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {PhiloxStream} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-random';

/** Philox counter tag of the multinomial case draws, equal to the contributor's. */
const SCAN_STREAM = 0x5ca75747;
const CASE_BLOCKS = 1024;

export type ScanOracleModel = 'poisson' | 'bernoulli' | 'space-time-permutation';

export type ScanOracleInput = {
  /** Interleaved x, y per zone. */
  positions: Float32Array;
  cases: Uint32Array;
  /** Poisson baseline. */
  baseline?: Float32Array;
  /** Bernoulli controls. */
  controls?: Uint32Array;
  model?: ScanOracleModel;
  timeBuckets: number;
  /** Compile-time capacity (default: every zone). */
  maximumWindowZones?: number;
  maximumPopulationFraction: number;
  /** Default: 50% of the study period, at least 1. */
  maximumTimeBuckets?: number;
  windowShape: 'circle' | 'nearest';
  rateDirection?: 'high' | 'low' | 'both';
  maximumClusters: number;
  seed: number;
  permutations: number;
};

export type ScanOracleCluster = {
  center: number;
  zoneCount: number;
  firstBucket: number;
  lastBucket: number;
  logLikelihoodRatio: number;
  observedCases: number;
  expectedCases: number;
  pValue: number;
  radius: number;
  truncatedByZoneCap: boolean;
};

function getTerm(observed: number, expected: number): number {
  return observed > 0 && expected > 0 ? observed * Math.log(observed / expected) : 0;
}

function isDirected(
  observed: number,
  expected: number,
  direction: 'high' | 'low' | 'both'
): boolean {
  if (direction === 'high') return observed > expected;
  if (direction === 'low') return observed < expected;
  return observed !== expected;
}

/**
 * Poisson LLR `c ln(c/e) + (C-c) ln((C-c)/(C-e))` of a window, in double precision. The
 * space-time permutation model uses the same form with the margin-based expectation.
 */
export function getOracleLogLikelihoodRatio(
  observed: number,
  baseline: number,
  totalCases: number,
  totalBaseline: number,
  direction: 'high' | 'low' | 'both' = 'high'
): number {
  const expected = (baseline * totalCases) / totalBaseline;
  if (!(expected > 0) || !isDirected(observed, expected, direction)) return 0;
  let ratio = getTerm(observed, expected);
  const remaining = totalCases - observed;
  if (remaining > 0 && totalCases - expected > 0) {
    ratio += remaining * Math.log(remaining / (totalCases - expected));
  }
  return Math.max(ratio, 0);
}

/**
 * Bernoulli LLR of a window of population `n` holding `c` of the `C` cases among `N`
 * (Kulldorff 1997), written in the textbook form with the null log-likelihood subtracted.
 */
export function getOracleBernoulliLogLikelihoodRatio(
  observed: number,
  windowPopulation: number,
  totalCases: number,
  totalPopulation: number,
  direction: 'high' | 'low' | 'both' = 'high'
): number {
  const n = windowPopulation;
  const N = totalPopulation;
  const C = totalCases;
  if (!(n > 0) || !(N - n > 0)) return 0;
  if (!isDirected(observed / n, (C - observed) / (N - n), direction)) return 0;
  const log = (value: number, base: number) => (value > 0 ? value * Math.log(base) : 0);
  const c = observed;
  const alternative =
    log(c, c / n) +
    log(n - c, (n - c) / n) +
    log(C - c, (C - c) / (N - n)) +
    log(N - C - (n - c), (N - C - (n - c)) / (N - n));
  const nul = log(C, C / N) + log(N - C, (N - C) / N);
  return Math.max(alternative - nul, 0);
}

type Best = {
  ratio: number;
  zoneCount: number;
  firstBucket: number;
  lastBucket: number;
  cases: number;
  expected: number;
};

/** Brute-force scan statistic with the contributor's window, tie and draw rules. */
export function computeScanOracle(input: ScanOracleInput) {
  const {positions, cases, timeBuckets: buckets} = input;
  const model = input.model ?? 'poisson';
  const direction = input.rateDirection ?? 'high';
  const zones = positions.length / 2;
  const cells = zones * buckets;
  const usable = Math.min(input.maximumWindowZones ?? zones, zones);
  const listLength = usable < zones ? usable + 1 : zones;
  const maximumBuckets = input.maximumTimeBuckets ?? Math.max(1, Math.floor(buckets / 2));
  const totalCases = cases.reduce((sum, value) => sum + value, 0);

  // Per-cell weight: Poisson baseline, Bernoulli population, or zone total * bucket total / C.
  const weights = new Float64Array(cells);
  const zoneTotals = new Array<number>(zones).fill(0);
  const bucketTotals = new Array<number>(buckets).fill(0);
  for (let cell = 0; cell < cells; cell++) {
    zoneTotals[Math.floor(cell / buckets)] += cases[cell];
    bucketTotals[cell % buckets] += cases[cell];
  }
  for (let cell = 0; cell < cells; cell++) {
    if (model === 'poisson') weights[cell] = input.baseline![cell];
    else if (model === 'bernoulli') weights[cell] = cases[cell] + input.controls![cell];
    else {
      weights[cell] =
        (zoneTotals[Math.floor(cell / buckets)] * bucketTotals[cell % buckets]) / totalCases;
    }
  }
  const totalWeight = weights.reduce((sum, value) => sum + value, 0);

  const lists: {zone: number; distance: number}[][] = [];
  const slotLimits: number[] = [];
  const truncatedCenters: boolean[] = [];
  for (let center = 0; center < zones; center++) {
    const all = [];
    for (let zone = 0; zone < zones; zone++) {
      const dx = positions[zone * 2] - positions[center * 2];
      const dy = positions[zone * 2 + 1] - positions[center * 2 + 1];
      all.push({zone, distance: Math.fround(dx * dx + dy * dy)});
    }
    all.sort((a, b) => a.distance - b.distance || a.zone - b.zone);
    const list = all.slice(0, listLength);
    lists.push(list);
    let populated = 0;
    let slots = 0;
    let exceeded = false;
    for (let slot = 0; slot < usable; slot++) {
      for (let bucket = 0; bucket < buckets; bucket++) {
        populated += weights[list[slot].zone * buckets + bucket];
      }
      if (populated > input.maximumPopulationFraction * totalWeight) {
        exceeded = true;
        break;
      }
      slots = slot + 1;
    }
    let truncated = false;
    if (!exceeded && usable < zones && slots === usable) {
      for (let bucket = 0; bucket < buckets; bucket++) {
        populated += weights[list[usable].zone * buckets + bucket];
      }
      truncated = populated <= input.maximumPopulationFraction * totalWeight;
    }
    slotLimits.push(slots);
    truncatedCenters.push(truncated);
  }

  const getRatio = (observed: number, windowWeight: number) =>
    model === 'bernoulli'
      ? getOracleBernoulliLogLikelihoodRatio(
          observed,
          windowWeight,
          totalCases,
          totalWeight,
          direction
        )
      : getOracleLogLikelihoodRatio(observed, windowWeight, totalCases, totalWeight, direction);

  const evaluateCenter = (center: number, counts: ArrayLike<number>): Best => {
    const caseSums = new Array(buckets).fill(0);
    const weightSums = new Array(buckets).fill(0);
    const best: Best = {
      ratio: 0,
      zoneCount: 0,
      firstBucket: 0,
      lastBucket: 0,
      cases: 0,
      expected: 0
    };
    for (let slot = 0; slot < slotLimits[center]; slot++) {
      const zone = lists[center][slot].zone;
      for (let bucket = 0; bucket < buckets; bucket++) {
        caseSums[bucket] += counts[zone * buckets + bucket];
        weightSums[bucket] += weights[zone * buckets + bucket];
      }
      const separated =
        input.windowShape === 'nearest' ||
        slot + 1 >= zones ||
        lists[center][slot].distance < lists[center][slot + 1].distance;
      if (!separated) continue;
      for (let first = 0; first < buckets; first++) {
        let windowCases = 0;
        let windowWeight = 0;
        for (let last = first; last < Math.min(first + maximumBuckets, buckets); last++) {
          windowCases += caseSums[last];
          windowWeight += weightSums[last];
          const ratio = getRatio(windowCases, windowWeight);
          if (ratio > best.ratio) {
            Object.assign(best, {
              ratio,
              zoneCount: slot + 1,
              firstBucket: first,
              lastBucket: last,
              cases: windowCases,
              expected: (windowWeight * totalCases) / totalWeight
            });
          }
        }
      }
    }
    return best;
  };

  const observed = Array.from({length: zones}, (_, center) => evaluateCenter(center, cases));

  const key: [number, number] = [input.seed % 2 ** 32, Math.floor(input.seed / 2 ** 32)];
  const maxima = [Math.max(0, ...observed.map(best => best.ratio))];
  let running = 0;
  let cumulative: number[] = [];
  if (model === 'poisson') {
    // Multinomial replicates from the integer cumulative table.
    // The contributor sums the baseline in f32 over fixed blocks and rounds weights in f32, and the
    // Lemire draw depends on the exact total weight, so the oracle repeats those f32 steps.
    const baseline = input.baseline!;
    const blocks = Math.min(256, cells);
    const cellsPerBlock = Math.ceil(cells / blocks);
    let float32Baseline = 0;
    for (let block = 0; block < blocks; block++) {
      let blockSum = 0;
      for (
        let cell = block * cellsPerBlock;
        cell < Math.min((block + 1) * cellsPerBlock, cells);
        cell++
      ) {
        blockSum = Math.fround(blockSum + baseline[cell]);
      }
      float32Baseline = Math.fround(float32Baseline + blockSum);
    }
    cumulative = new Array<number>(cells);
    for (let cell = 0; cell < cells; cell++) {
      const share = Math.fround(Math.fround(baseline[cell] / float32Baseline) * 2147483648);
      running += baseline[cell] > 0 ? Math.floor(share) : 0;
      cumulative[cell] = running;
    }
  } else if (model === 'bernoulli') {
    cumulative = new Array<number>(cells);
    for (let cell = 0; cell < cells; cell++) {
      running += weights[cell];
      cumulative[cell] = running;
    }
  }
  const mulHigh = (value: number, bound: number) => Number((BigInt(value) * BigInt(bound)) >> 32n);
  const findCell = (target: number) => {
    let low = 0;
    let high = cells;
    while (low < high) {
      const middle = low + ((high - low) >> 1);
      if (cumulative[middle] > target) high = middle;
      else low = middle + 1;
    }
    return low;
  };

  for (let replicate = 0; replicate < input.permutations; replicate++) {
    const counts = new Uint32Array(cells);
    if (model === 'poisson') {
      const chunk = Math.ceil(totalCases / CASE_BLOCKS);
      for (let block = 0; block < CASE_BLOCKS; block++) {
        const first = block * chunk;
        const end = Math.min(first + chunk, totalCases);
        if (first >= end) continue;
        const stream = new PhiloxStream(key, replicate, block, SCAN_STREAM);
        for (let draw = first; draw < end; draw++) {
          counts[findCell(mulHigh(stream.nextUint32(), running))]++;
        }
      }
    } else if (model === 'bernoulli') {
      // Draw individuals without replacement: a population-proportional cell, accepted with the
      // share of its individuals not yet chosen; the controls are drawn when cases are the majority.
      const complement = totalCases > running - totalCases;
      const drawCount = complement ? running - totalCases : totalCases;
      const stream = new PhiloxStream(key, replicate, 0, SCAN_STREAM);
      for (let drawn = 0; drawn < drawCount; ) {
        const cell = findCell(mulHigh(stream.nextUint32(), running));
        const unchosen = weights[cell] - counts[cell];
        if (mulHigh(stream.nextUint32(), weights[cell]) < unchosen) {
          counts[cell]++;
          drawn++;
        }
      }
      if (complement)
        for (let cell = 0; cell < cells; cell++) counts[cell] = weights[cell] - counts[cell];
    } else {
      // Case times permuted over the case locations: zones in order take their times without
      // replacement from the remaining multiset of times.
      const remaining = [...bucketTotals];
      let left = totalCases;
      const stream = new PhiloxStream(key, replicate, 0, SCAN_STREAM);
      for (let zone = 0; zone < zones; zone++) {
        for (let draw = 0; draw < zoneTotals[zone]; draw++) {
          const target = mulHigh(stream.nextUint32(), left);
          let accumulated = 0;
          let chosen = buckets - 1;
          for (let bucket = 0; bucket < buckets; bucket++) {
            accumulated += remaining[bucket];
            if (target < accumulated) {
              chosen = bucket;
              break;
            }
          }
          counts[zone * buckets + chosen]++;
          remaining[chosen]--;
          left--;
        }
      }
    }
    let maximum = 0;
    for (let center = 0; center < zones; center++) {
      maximum = Math.max(maximum, evaluateCenter(center, counts).ratio);
    }
    maxima.push(maximum);
  }

  const alive = observed.map(best => best.ratio > 0);
  const clusters: ScanOracleCluster[] = [];
  while (clusters.length < input.maximumClusters) {
    let bestCenter = -1;
    for (let center = 0; center < zones; center++) {
      if (
        alive[center] &&
        (bestCenter < 0 || observed[center].ratio > observed[bestCenter].ratio)
      ) {
        bestCenter = center;
      }
    }
    if (bestCenter < 0) break;
    const best = observed[bestCenter];
    const members = new Set(lists[bestCenter].slice(0, best.zoneCount).map(entry => entry.zone));
    let exceeding = 0;
    for (let replicate = 1; replicate <= input.permutations; replicate++) {
      if (maxima[replicate] >= best.ratio) exceeding++;
    }
    clusters.push({
      center: bestCenter,
      zoneCount: best.zoneCount,
      firstBucket: best.firstBucket,
      lastBucket: best.lastBucket,
      logLikelihoodRatio: best.ratio,
      observedCases: best.cases,
      expectedCases: best.expected,
      pValue: (exceeding + 1) / (input.permutations + 1),
      radius: Math.sqrt(lists[bestCenter][best.zoneCount - 1].distance),
      truncatedByZoneCap: truncatedCenters[bestCenter]
    });
    for (let center = 0; center < zones; center++) {
      if (!alive[center]) continue;
      const own = lists[center].slice(0, observed[center].zoneCount);
      if (own.some(entry => members.has(entry.zone))) alive[center] = false;
    }
  }
  return {
    clusters,
    maxima,
    observed,
    totalCases,
    totalWeight: running,
    overflow: truncatedCenters.filter(Boolean).length
  };
}

/** Deterministic uniform generator for scenes (mulberry32). */
function createUniform(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * A `side x side` lattice (spacing 10, exact in f32) of Poisson zones with a regional cluster: the
 * first `hotRows` rows (`hotRows * side` zones) have a high rate. One bucket.
 */
export function createRegionalClusterScene(
  side: number,
  hotRows: number,
  seed: number
): ScanOracleInput {
  const random = createUniform(seed);
  const zones = side * side;
  const positions = new Float32Array(zones * 2);
  const cases = new Uint32Array(zones);
  const baseline = new Float32Array(zones);
  for (let zone = 0; zone < zones; zone++) {
    positions[zone * 2] = (zone % side) * 10;
    positions[zone * 2 + 1] = Math.floor(zone / side) * 10;
    baseline[zone] = 100;
    const rate = Math.floor(zone / side) < hotRows ? 0.3 : 0.05;
    let count = 0;
    for (let person = 0; person < 100; person++) if (random() < rate) count++;
    cases[zone] = count;
  }
  return {
    positions,
    cases,
    baseline,
    timeBuckets: 1,
    maximumPopulationFraction: 0.5,
    windowShape: 'circle',
    maximumClusters: 4,
    seed: 777,
    permutations: 99
  };
}
