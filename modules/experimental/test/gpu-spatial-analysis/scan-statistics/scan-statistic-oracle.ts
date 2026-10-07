// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {PhiloxStream} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-random';

/** Philox counter tag of the multinomial case draws, equal to the contributor's. */
const SCAN_STREAM = 0x5ca75747;
const CASE_BLOCKS = 1024;

export type ScanOracleInput = {
  /** Interleaved x, y per zone. */
  positions: Float32Array;
  cases: Uint32Array;
  baseline: Float32Array;
  timeBuckets: number;
  maximumWindowZones: number;
  maximumPopulationFraction: number;
  maximumTimeBuckets: number;
  windowShape: 'circle' | 'nearest';
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
};

/** Poisson LLR of a window (high rates only), in double precision. */
export function getOracleLogLikelihoodRatio(
  observed: number,
  baseline: number,
  totalCases: number,
  totalBaseline: number
): number {
  const expected = (baseline * totalCases) / totalBaseline;
  if (!(expected > 0) || observed <= expected) return 0;
  let ratio = observed * Math.log(observed / expected);
  const remaining = totalCases - observed;
  if (remaining > 0) ratio += remaining * Math.log(remaining / (totalCases - expected));
  return ratio;
}

type Best = {
  ratio: number;
  zoneCount: number;
  firstBucket: number;
  lastBucket: number;
  cases: number;
  expected: number;
};

/** Brute-force scan statistic with the contributor's tie rules and Philox case draws. */
export function computeScanOracle(input: ScanOracleInput) {
  const {positions, cases, baseline, timeBuckets: buckets} = input;
  const zones = positions.length / 2;
  const cells = zones * buckets;
  const listLength = Math.min(input.maximumWindowZones, zones);
  const totalCases = cases.reduce((sum, value) => sum + value, 0);
  const totalBaseline = baseline.reduce((sum, value) => sum + value, 0);

  const lists: {zone: number; distance: number}[][] = [];
  const masks: boolean[][] = [];
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
    const mask: boolean[] = [];
    let populated = 0;
    let open = true;
    for (let slot = 0; slot < listLength; slot++) {
      for (let bucket = 0; bucket < buckets; bucket++) {
        populated += baseline[list[slot].zone * buckets + bucket];
      }
      if (populated > input.maximumPopulationFraction * totalBaseline) open = false;
      const separated =
        input.windowShape === 'nearest' ||
        slot + 1 === listLength ||
        list[slot].distance < list[slot + 1].distance;
      mask.push(open && separated);
    }
    masks.push(mask);
  }

  const evaluateCenter = (center: number, counts: ArrayLike<number>, offset: number): Best => {
    const caseSums = new Array(buckets).fill(0);
    const baselineSums = new Array(buckets).fill(0);
    const best: Best = {
      ratio: 0,
      zoneCount: 0,
      firstBucket: 0,
      lastBucket: 0,
      cases: 0,
      expected: 0
    };
    for (let slot = 0; slot < listLength; slot++) {
      if (!masks[center].slice(slot).some(Boolean)) break;
      const zone = lists[center][slot].zone;
      for (let bucket = 0; bucket < buckets; bucket++) {
        caseSums[bucket] += counts[offset + zone * buckets + bucket];
        baselineSums[bucket] += baseline[zone * buckets + bucket];
      }
      if (!masks[center][slot]) continue;
      for (let first = 0; first < buckets; first++) {
        let windowCases = 0;
        let windowBaseline = 0;
        for (let last = first; last < Math.min(first + input.maximumTimeBuckets, buckets); last++) {
          windowCases += caseSums[last];
          windowBaseline += baselineSums[last];
          const ratio = getOracleLogLikelihoodRatio(
            windowCases,
            windowBaseline,
            totalCases,
            totalBaseline
          );
          if (ratio > best.ratio) {
            Object.assign(best, {
              ratio,
              zoneCount: slot + 1,
              firstBucket: first,
              lastBucket: last,
              cases: windowCases,
              expected: (windowBaseline * totalCases) / totalBaseline
            });
          }
        }
      }
    }
    return best;
  };

  const observed = Array.from({length: zones}, (_, center) => evaluateCenter(center, cases, 0));

  // Multinomial replicates from the integer cumulative table.
  // The contributor sums the baseline in f32 over fixed blocks and rounds weights in f32, and the
  // Lemire draw depends on the exact total weight, so the oracle repeats those f32 steps.
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
  const cumulative = new Array<number>(cells);
  let running = 0;
  for (let cell = 0; cell < cells; cell++) {
    const share = Math.fround(Math.fround(baseline[cell] / float32Baseline) * 2147483648);
    running += baseline[cell] > 0 ? Math.floor(share) : 0;
    cumulative[cell] = running;
  }
  const key: [number, number] = [input.seed % 2 ** 32, Math.floor(input.seed / 2 ** 32)];
  const maxima = [Math.max(0, ...observed.map(best => best.ratio))];
  for (let replicate = 0; replicate < input.permutations; replicate++) {
    const counts = new Uint32Array(cells);
    const chunk = Math.ceil(totalCases / CASE_BLOCKS);
    for (let block = 0; block < CASE_BLOCKS; block++) {
      const first = block * chunk;
      const end = Math.min(first + chunk, totalCases);
      if (first >= end) continue;
      const stream = new PhiloxStream(key, replicate, block, SCAN_STREAM);
      for (let draw = first; draw < end; draw++) {
        const target = Number((BigInt(stream.nextUint32()) * BigInt(running)) >> 32n);
        let low = 0;
        let high = cells;
        while (low < high) {
          const middle = low + ((high - low) >> 1);
          if (cumulative[middle] > target) high = middle;
          else low = middle + 1;
        }
        counts[low]++;
      }
    }
    let maximum = 0;
    for (let center = 0; center < zones; center++) {
      maximum = Math.max(maximum, evaluateCenter(center, counts, 0).ratio);
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
      radius: Math.sqrt(lists[bestCenter][best.zoneCount - 1].distance)
    });
    for (let center = 0; center < zones; center++) {
      if (!alive[center]) continue;
      const own = lists[center].slice(0, observed[center].zoneCount);
      if (own.some(entry => members.has(entry.zone))) alive[center] = false;
    }
  }
  return {clusters, maxima, observed, totalCases, totalBaseline, totalWeight: running};
}
