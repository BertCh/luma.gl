// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {PhiloxStream} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-random';
import {
  NETWORK_K_FUNCTION_LENGTH_RANGE,
  NETWORK_K_FUNCTION_STREAM
} from '../../../src/gpu-network/network-k-function/index';
import {
  dijkstra,
  NONE,
  snapOracle,
  type AccessibilityCSR
} from '../network-accessibility/network-accessibility-oracle';

/** One event placed on the network: edge endpoints and the costs to them, or inactive. */
export type KFunctionEvent = {
  first: number;
  second: number;
  firstCost: number;
  secondCost: number;
};

const INACTIVE: KFunctionEvent = {first: NONE, second: NONE, firstCost: -1, secondCost: -1};

/** Snaps events like `GPUNetworkSnapping` with `seedDirection: 'both'`. */
export function snapEvents(
  points: ArrayLike<number>,
  positions: ArrayLike<number>,
  csr: AccessibilityCSR,
  maxSnapDistance: number = Infinity
): KFunctionEvent[] {
  return snapOracle(points, positions, csr.sources, csr.neighbors, {
    edgeCosts: csr.weights,
    maxSnapDistance
  }).map(snap =>
    snap.edge === NONE
      ? INACTIVE
      : {
          first: csr.sources[snap.edge],
          second: csr.neighbors[snap.edge],
          firstCost: Math.fround(snap.sourceCost),
          secondCost: Math.fround(snap.targetCost)
        }
  );
}

/**
 * Draws `eventCount` random events like the GPU `simulate` kernel: an edge with probability
 * proportional to its quantized length through a Philox stream keyed by `seed`, `(event, pattern)`
 * counters and the K function tag, then a uniform fraction along it.
 */
export function simulateEvents(
  csr: AccessibilityCSR,
  nodeCount: number,
  networkLength: number,
  seed: number,
  pattern: number,
  eventCount: number,
  validCount: number
): KFunctionEvent[] {
  const scale = Math.fround(NETWORK_K_FUNCTION_LENGTH_RANGE / Math.fround(networkLength));
  const prefix = new Uint32Array(csr.neighbors.length);
  let sum = 0;
  for (let edge = 0; edge < prefix.length; edge++) {
    const weight = csr.weights[edge];
    if (networkLength > 0 && weight >= 0 && csr.neighbors[edge] < nodeCount) {
      sum += Math.floor(Math.fround(weight * scale));
    }
    prefix[edge] = sum;
  }
  const key: [number, number] = [seed % 2 ** 32, Math.floor(seed / 2 ** 32)];
  const events: KFunctionEvent[] = [];
  for (let event = 0; event < eventCount; event++) {
    if (event >= validCount || sum === 0) {
      events.push(INACTIVE);
      continue;
    }
    const stream = new PhiloxStream(key, event, pattern, NETWORK_K_FUNCTION_STREAM);
    const draw = stream.nextBelow(sum);
    const fraction = (stream.nextUint32() >>> 8) / 16777216;
    const edge = prefix.findIndex(value => value > draw);
    const weight = csr.weights[edge];
    events.push({
      first: csr.sources[edge],
      second: csr.neighbors[edge],
      firstCost: Math.fround(fraction * weight),
      secondCost: Math.fround((1 - fraction) * weight)
    });
  }
  return events;
}

/** Cumulative unordered-pair counts below each of `bandCount` thresholds (f32 thresholds, strict `<`). */
export function getPairCounts(
  csr: AccessibilityCSR,
  nodeCount: number,
  events: readonly KFunctionEvent[],
  maxDistance: number,
  bandCount: number
): number[] {
  const thresholds = Array.from({length: bandCount}, (_, band) =>
    Math.fround(Math.fround(Math.fround(maxDistance) * band) / (bandCount - 1))
  );
  const counts = new Array<number>(bandCount).fill(0);
  const isActive = (event: KFunctionEvent) => event.first < nodeCount && event.second < nodeCount;
  events.forEach((row, rowIndex) => {
    if (!isActive(row)) {
      return;
    }
    const costs = dijkstra(
      csr,
      nodeCount,
      [
        {node: row.first, cost: row.firstCost},
        {node: row.second, cost: row.secondCost}
      ],
      maxDistance
    );
    for (let columnIndex = rowIndex + 1; columnIndex < events.length; columnIndex++) {
      const column = events[columnIndex];
      if (!isActive(column)) {
        continue;
      }
      let best = Infinity;
      for (const [node, cost] of [
        [column.first, column.firstCost],
        [column.second, column.secondCost]
      ]) {
        if (Number.isFinite(costs[node])) {
          best = Math.min(best, Math.fround(costs[node] + cost));
        }
      }
      if (row.first !== row.second) {
        if (row.first === column.first && row.second === column.second) {
          best = Math.min(best, Math.abs(Math.fround(row.firstCost - column.firstCost)));
        } else if (row.first === column.second && row.second === column.first) {
          best = Math.min(best, Math.abs(Math.fround(row.firstCost - column.secondCost)));
        }
      }
      thresholds.forEach((threshold, band) => {
        if (best < threshold) {
          counts[band]++;
        }
      });
    }
  });
  return counts;
}

/** `K = 2 * pairs * L / n^2`. */
export function getKValues(
  counts: readonly number[],
  validCount: number,
  networkLength: number
): number[] {
  return counts.map(count =>
    validCount >= 2 ? (2 * count * networkLength) / (validCount * validCount) : 0
  );
}
