// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {NONE, type NetworkCSR} from '../network-reachability/network-reachability-oracle';

/** Inputs of {@link neighborhoodOracle}. */
export type NeighborhoodOracleInput = {
  csr: NetworkCSR;
  nodeCount: number;
  seeds: readonly number[];
  /** Active seed count. Defaults to every seed. */
  seedCount?: number;
  hops: number;
  maxHops: number;
  nodeIds?: readonly number[];
  edgeIds?: readonly number[];
  nodeCapacity?: number;
  edgeCapacity?: number;
};

/** Result of {@link neighborhoodOracle}. */
export type NeighborhoodOracleResult = {
  hopDistances: number[];
  nodeMask: number[];
  edgeMask: number[];
  nodeIds: number[];
  nodeCount: number;
  nodeTotal: number;
  nodeOverflow: number;
  edgeIds: number[];
  edgeCount: number;
  edgeTotal: number;
  edgeOverflow: number;
};

/**
 * CPU reference: multi-seed outgoing BFS to depth `min(hops, maxHops)`, induced edge mask with rows
 * clamped to the edge count, and compact ID lists in row order clamped to their capacities.
 */
export function neighborhoodOracle(input: NeighborhoodOracleInput): NeighborhoodOracleResult {
  const {csr, nodeCount} = input;
  const edgeCount = csr.neighbors.length;
  const limit = Math.min(input.hops, input.maxHops);
  const hopDistances = new Array<number>(nodeCount).fill(NONE);
  let frontier: number[] = [];
  const activeSeedCount = Math.min(input.seedCount ?? input.seeds.length, input.seeds.length);
  for (let index = 0; index < activeSeedCount; index++) {
    const seed = input.seeds[index];
    if (seed < nodeCount && hopDistances[seed] === NONE) {
      hopDistances[seed] = 0;
      frontier.push(seed);
    }
  }
  for (let depth = 1; depth <= limit && frontier.length > 0; depth++) {
    const next: number[] = [];
    for (const node of frontier) {
      const rowEnd = Math.min(csr.offsets[node + 1], edgeCount);
      for (let edge = Math.min(csr.offsets[node], edgeCount); edge < rowEnd; edge++) {
        const target = csr.neighbors[edge];
        if (target < nodeCount && hopDistances[target] === NONE) {
          hopDistances[target] = depth;
          next.push(target);
        }
      }
    }
    frontier = next;
  }
  const nodeMask = hopDistances.map(hop => (hop === NONE ? 0 : 1));
  const edgeMask = new Array<number>(edgeCount).fill(0);
  for (let node = 0; node < nodeCount; node++) {
    if (!nodeMask[node]) {
      continue;
    }
    const rowEnd = Math.min(csr.offsets[node + 1], edgeCount);
    for (let edge = Math.min(csr.offsets[node], edgeCount); edge < rowEnd; edge++) {
      const target = csr.neighbors[edge];
      edgeMask[edge] = target < nodeCount && nodeMask[target] ? 1 : 0;
    }
  }
  const compact = (
    mask: number[],
    ids: readonly number[] | undefined,
    capacity: number | undefined
  ) => {
    const accepted: number[] = [];
    for (const [row, flag] of mask.entries()) {
      if (flag) {
        accepted.push(ids?.[row] ?? row);
      }
    }
    const total = accepted.length;
    const count = Math.min(total, capacity ?? total);
    return {ids: accepted.slice(0, count), count, total, overflow: total > count ? 1 : 0};
  };
  const nodes = compact(nodeMask, input.nodeIds, input.nodeCapacity);
  const edges = compact(edgeMask, input.edgeIds, input.edgeCapacity);
  return {
    hopDistances,
    nodeMask,
    edgeMask,
    nodeIds: nodes.ids,
    nodeCount: nodes.count,
    nodeTotal: nodes.total,
    nodeOverflow: nodes.overflow,
    edgeIds: edges.ids,
    edgeCount: edges.count,
    edgeTotal: edges.total,
    edgeOverflow: edges.overflow
  };
}
