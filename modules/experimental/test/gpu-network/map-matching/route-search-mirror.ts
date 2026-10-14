// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** CSR network with planar node positions, for the route-search mirror. */
export type RouteSearchNetwork = {
  offsets: readonly number[];
  neighbors: readonly number[];
  positions: readonly number[];
};

/** How a full node table treats a newly discovered node. */
export type RouteTablePolicy =
  /** Pre-fix GPU behavior: the new node is dropped, uncounted. */
  | 'reject'
  /** Fixed GPU behavior: the open entry with the worst key is evicted when the new key is better. */
  | 'evict-worst';

export type RouteSearchResult = {
  /** Settled network distance of every target, or -1 when the search did not settle it. */
  targetDistances: number[];
  /** Discovered nodes the full table could not keep (rejected, or evicted for a better key). */
  droppedCount: number;
};

/**
 * Float64 mirror of `searchNetwork` in `GPUMapMatching`'s route kernel: bounded A* (heuristic the
 * straight distance to the nearest pending target, times 0.999999) over a private table of at most
 * `budget` nodes. Settlement picks the open entry with the smallest key, ties to the lowest slot.
 * It mirrors the WGSL control flow line for line so a node spec can pin the table policy; the GPU
 * spec runs the real kernel.
 */
export function searchRouteMirror(
  network: RouteSearchNetwork,
  start: number,
  limit: number,
  targets: readonly number[],
  budget: number,
  policy: RouteTablePolicy
): RouteSearchResult {
  const position = (node: number) => [network.positions[2 * node], network.positions[2 * node + 1]];
  const distanceBetween = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const length = (edge: number, source: number) =>
    distanceBetween(position(source), position(network.neighbors[edge]));
  const targetPositions = targets.map(position);
  const heuristic = (node: number) =>
    targetPositions.length === 0
      ? 0
      : Math.min(...targetPositions.map(target => distanceBetween(position(node), target))) *
        0.999999;
  const tableNode: number[] = [start];
  const tableDistance: number[] = [0];
  const tableKey: number[] = [heuristic(start)];
  const tableDone: boolean[] = [false];
  const pending = new Set(targets);
  let droppedCount = 0;
  for (;;) {
    let best = -1;
    let bestKey = limit + 1;
    for (let slot = 0; slot < tableNode.length; slot++) {
      if (!tableDone[slot] && tableKey[slot] < bestKey) {
        best = slot;
        bestKey = tableKey[slot];
      }
    }
    if (best < 0 || bestKey > limit) {
      break;
    }
    tableDone[best] = true;
    const node = tableNode[best];
    pending.delete(node);
    if (pending.size === 0) {
      break;
    }
    for (let edge = network.offsets[node]; edge < network.offsets[node + 1]; edge++) {
      const candidateDistance = tableDistance[best] + length(edge, node);
      if (candidateDistance > limit) {
        continue;
      }
      const neighbor = network.neighbors[edge];
      const found = tableNode.indexOf(neighbor);
      if (found >= 0) {
        if (!tableDone[found] && candidateDistance < tableDistance[found]) {
          tableDistance[found] = candidateDistance;
          tableKey[found] = candidateDistance + heuristic(neighbor);
        }
        continue;
      }
      const key = candidateDistance + heuristic(neighbor);
      if (tableNode.length < budget) {
        tableNode.push(neighbor);
        tableDistance.push(candidateDistance);
        tableKey.push(key);
        tableDone.push(false);
        continue;
      }
      droppedCount++;
      if (policy === 'evict-worst') {
        // Worst open entry, ties to the highest slot.
        let worst = -1;
        for (let slot = 0; slot < tableNode.length; slot++) {
          if (!tableDone[slot] && (worst < 0 || tableKey[slot] >= tableKey[worst])) {
            worst = slot;
          }
        }
        if (worst >= 0 && key < tableKey[worst]) {
          tableNode[worst] = neighbor;
          tableDistance[worst] = candidateDistance;
          tableKey[worst] = key;
        }
      }
    }
  }
  return {
    targetDistances: targets.map(target => {
      const slot = tableNode.indexOf(target);
      return slot >= 0 && tableDone[slot] ? tableDistance[slot] : -1;
    }),
    droppedCount
  };
}

/** Exact bounded Dijkstra distance from `start` to `target`, or -1 beyond `limit`. */
export function getExactRouteDistance(
  network: RouteSearchNetwork,
  start: number,
  target: number,
  limit: number
): number {
  const result = searchRouteMirror(
    network,
    start,
    limit,
    [target],
    Number.MAX_SAFE_INTEGER,
    'reject'
  );
  return result.targetDistances[0];
}

/**
 * Bidirectional Manhattan grid of `width * width` nodes `spacing` apart (the 100 m street grid of
 * review finding B-02), with rows in node order and neighbors in (west, east, south, north) order.
 */
export function createManhattanGrid(width: number, spacing: number): RouteSearchNetwork {
  const offsets = [0];
  const neighbors: number[] = [];
  const positions: number[] = [];
  for (let y = 0; y < width; y++) {
    for (let x = 0; x < width; x++) {
      positions.push(x * spacing, y * spacing);
      for (const [dx, dy] of [
        [-1, 0],
        [1, 0],
        [0, -1],
        [0, 1]
      ]) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx >= 0 && nx < width && ny >= 0 && ny < width) {
          neighbors.push(ny * width + nx);
        }
      }
      offsets.push(neighbors.length);
    }
  }
  return {offsets, neighbors, positions};
}

/** One route query between consecutive GPS samples: start node, target node, A* bound. */
export type RouteQuery = {start: number; target: number; limit: number};

/**
 * `count` deterministic route queries on a `width` Manhattan grid with `blockLength` blocks:
 * sample pairs about `sampleSpacing` apart in a random direction, with the `GPUMapMatching` default
 * bound `3 * straight + 50`.
 */
export function createRouteQueries(
  width: number,
  blockLength: number,
  sampleSpacing: number,
  count: number,
  seed = 7
): RouteQuery[] {
  let state = seed >>> 0;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
  const queries: RouteQuery[] = [];
  const blocks = sampleSpacing / blockLength;
  const margin = Math.ceil(blocks / 2);
  for (let index = 0; index < count; index++) {
    const angle = random() * Math.PI * 2;
    const dx = Math.round(Math.cos(angle) * blocks);
    const dy = Math.round(Math.sin(angle) * blocks);
    const x0 = margin + Math.floor(random() * (width - 2 * margin));
    const y0 = margin + Math.floor(random() * (width - 2 * margin));
    const x1 = Math.min(width - 1, Math.max(0, x0 + dx));
    const y1 = Math.min(width - 1, Math.max(0, y0 + dy));
    const straight = Math.hypot(x1 - x0, y1 - y0) * blockLength;
    queries.push({start: y0 * width + x0, target: y1 * width + x1, limit: 3 * straight + 50});
  }
  return queries;
}
