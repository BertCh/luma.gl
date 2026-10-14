// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import fixtureJson from '../../oracles/fixtures/statistics-ward.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';

/** One cut of a `statistics-ward.py` scene. */
export type WardFixtureCut = {
  targetRegionCount: number;
  /** Canonical labels: the smallest row of each cluster. */
  labels: number[];
  objective: number;
  marginRatio: number;
};

/** One scene of `statistics-ward.py`. */
export type WardFixtureScene = {
  rowCount: number;
  columnCount: number;
  values: number[];
  cuts: WardFixtureCut[];
  width?: number;
};

/** The decoded `statistics-ward.json` fixture (scipy and scikit-learn references). */
export const WARD_FIXTURE = decodeOracleFixture<{
  tolerances: {marginRatio: number; objectiveRelative: number};
  separated: WardFixtureScene;
  random: WardFixtureScene;
  lattice: WardFixtureScene;
}>(fixtureJson);

/** CSR adjacency lists (ascending neighbors) of a complete graph on `count` rows. */
export function createCompleteAdjacency(count: number): number[][] {
  return Array.from({length: count}, (_, row) =>
    Array.from({length: count}, (_, other) => other).filter(other => other !== row)
  );
}

/** Rook-contiguity adjacency lists of a width x height lattice. */
export function createLatticeAdjacency(width: number, height: number): number[][] {
  return Array.from({length: width * height}, (_, row) => {
    const x = row % width;
    return [row - width, x > 0 ? row - 1 : -1, x < width - 1 ? row + 1 : -1, row + width].filter(
      candidate => candidate >= 0 && candidate < width * height
    );
  });
}

/** CSR arrays of adjacency lists, for the weights of a contributor. */
export function getCSR(adjacency: readonly (readonly number[])[]) {
  const offsets = [0];
  const neighbors: number[] = [];
  for (const list of adjacency) {
    neighbors.push(...list);
    offsets.push(neighbors.length);
  }
  return {
    offsets: Uint32Array.from(offsets),
    neighbors: Uint32Array.from(neighbors),
    weights: Float32Array.from(neighbors, () => 1)
  };
}

/** The seed-keyed pair order of `GPUWardRegions` (`pairKey` in its WGSL), in uint32 arithmetic. */
export function getWardPairKey(first: number, second: number, seed: number): number {
  let value = (Math.imul(first, 0x9e3779b9) ^ second ^ seed) >>> 0;
  value = Math.imul(value ^ (value >>> 16), 0x7feb352d) >>> 0;
  value = Math.imul(value ^ (value >>> 15), 0x846ca68b) >>> 0;
  return (value ^ (value >>> 16)) >>> 0;
}

/** Result of {@link runWardReference}. */
export type WardReferenceResult = {
  /** Smallest row of each row's region. */
  labels: number[];
  /** Sum of the merges' Ward increases. */
  objective: number;
  mergeCount: number;
  /** Merge iterations executed (one merge each for `'sequential'`). */
  roundCount: number;
};

/**
 * CPU float64 mirror of the `GPUWardRegions` merge schedules (test oracle only): the same total
 * pair order (increase, seed-keyed pair key, first, second) and the same round structure.
 */
export function runWardReference(props: {
  values: ArrayLike<number>;
  columnCount: number;
  adjacency: readonly (readonly number[])[];
  targetRegionCount: number;
  mergeSchedule: 'sequential' | 'mutual-nearest';
  seed?: number;
  maximumRounds?: number;
}): WardReferenceResult {
  const {values, columnCount, adjacency, targetRegionCount, mergeSchedule} = props;
  const seed = props.seed ?? 0;
  const rowCount = adjacency.length;
  const labels = Array.from({length: rowCount}, (_, row) => row);
  const size = new Map<number, number>();
  const total = new Map<number, number[]>();
  for (let row = 0; row < rowCount; row++) {
    size.set(row, 1);
    total.set(
      row,
      Array.from({length: columnCount}, (_, column) => values[row * columnCount + column])
    );
  }
  const getIncrease = (first: number, second: number) => {
    const firstSize = size.get(first)!;
    const secondSize = size.get(second)!;
    let increase = 0;
    for (let column = 0; column < columnCount; column++) {
      const difference =
        total.get(first)![column] / firstSize - total.get(second)![column] / secondSize;
      increase += ((firstSize * secondSize) / (firstSize + secondSize)) * difference * difference;
    }
    return increase;
  };
  // Adjacent region pairs, first < second, from the rows' labels.
  const getPairs = () => {
    const pairs = new Map<string, {first: number; second: number}>();
    for (let row = 0; row < rowCount; row++) {
      for (const neighbor of adjacency[row]) {
        const first = Math.min(labels[row], labels[neighbor]);
        const second = Math.max(labels[row], labels[neighbor]);
        if (first !== second) pairs.set(`${first},${second}`, {first, second});
      }
    }
    return [...pairs.values()].map(pair => ({
      ...pair,
      increase: getIncrease(pair.first, pair.second),
      key: getWardPairKey(pair.first, pair.second, seed)
    }));
  };
  type Pair = ReturnType<typeof getPairs>[number];
  const compare = (left: Pair, right: Pair) =>
    left.increase - right.increase ||
    left.key - right.key ||
    left.first - right.first ||
    left.second - right.second;
  const merge = (pair: Pair) => {
    size.set(pair.first, size.get(pair.first)! + size.get(pair.second)!);
    total.set(
      pair.first,
      total.get(pair.first)!.map((sum, column) => sum + total.get(pair.second)![column])
    );
    size.delete(pair.second);
    total.delete(pair.second);
    for (let row = 0; row < rowCount; row++) {
      if (labels[row] === pair.second) labels[row] = pair.first;
    }
  };
  let objective = 0;
  let mergeCount = 0;
  let roundCount = 0;
  const maximumRounds = props.maximumRounds ?? rowCount;
  while (size.size > targetRegionCount && roundCount < maximumRounds) {
    const pairs = getPairs();
    if (pairs.length === 0) break;
    roundCount++;
    let selected: Pair[];
    if (mergeSchedule === 'sequential') {
      selected = [pairs.sort(compare)[0]];
    } else {
      // Each region's cheapest adjacent pair; pairs chosen from both sides merge.
      const nearest = new Map<number, Pair>();
      for (const pair of pairs) {
        for (const region of [pair.first, pair.second]) {
          const best = nearest.get(region);
          if (!best || compare(pair, best) < 0) nearest.set(region, pair);
        }
      }
      selected = pairs
        .filter(pair => nearest.get(pair.first) === pair && nearest.get(pair.second) === pair)
        .sort(compare)
        .slice(0, size.size - targetRegionCount);
    }
    for (const pair of selected) {
      objective += pair.increase;
      mergeCount++;
      merge(pair);
    }
  }
  return {labels, objective, mergeCount, roundCount};
}
