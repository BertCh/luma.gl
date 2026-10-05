// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {dijkstra, NONE, type NetworkCSR} from '../network-reachability/network-reachability-oracle';

/** One facility seed: node index and starting cost. */
export type ServiceAreasFacility = {node: number; cost: number};

/** CPU reference result. */
export type ServiceAreasOracleResult = {
  /** Minimum cost per node, `Infinity` when unreached. */
  costs: Float32Array;
  /** Smallest facility row among facilities achieving the minimum cost, or `NONE`. */
  assignments: Uint32Array;
  /** Nodes per facility row. */
  nodeCounts: Uint32Array;
  /** Sum of node costs per facility row. */
  costSums: Float64Array;
};

/**
 * CPU reference for `GPUNetworkServiceAreas`.
 *
 * Costs come from Dijkstra unless `costs` is given (use the GPU costs to avoid f32 summation order
 * differences). Assignments are the min-label fixpoint over tight edges with the GPU tie rule.
 */
export function serviceAreasOracle(
  csr: NetworkCSR,
  nodeCount: number,
  facilities: readonly ServiceAreasFacility[],
  facilityCount: number = facilities.length,
  costLimit: number = Infinity,
  costsOverride?: Float32Array
): ServiceAreasOracleResult {
  const active = facilities.slice(0, Math.min(facilityCount, facilities.length));
  const costs = costsOverride ?? dijkstra(csr, nodeCount, active, costLimit);
  const assignments = new Uint32Array(nodeCount).fill(NONE);
  for (const [row, {node, cost}] of active.entries()) {
    if (node < nodeCount && cost >= 0 && Number.isFinite(cost) && cost <= costLimit) {
      if (Math.fround(cost) === costs[node]) {
        assignments[node] = Math.min(assignments[node], row);
      }
    }
  }
  let changed = true;
  while (changed) {
    changed = false;
    for (let node = 0; node < nodeCount; node++) {
      if (assignments[node] === NONE || !Number.isFinite(costs[node])) {
        continue;
      }
      for (let edge = csr.offsets[node]; edge < csr.offsets[node + 1]; edge++) {
        const target = csr.neighbors[edge];
        const weight = csr.weights[edge];
        if (target >= nodeCount || !(weight >= 0) || !Number.isFinite(costs[target])) {
          continue;
        }
        if (
          Math.fround(costs[node] + weight) === costs[target] &&
          assignments[node] < assignments[target]
        ) {
          assignments[target] = assignments[node];
          changed = true;
        }
      }
    }
  }
  const nodeCounts = new Uint32Array(facilities.length);
  const costSums = new Float64Array(facilities.length);
  for (let node = 0; node < nodeCount; node++) {
    const row = assignments[node];
    if (row < facilities.length && Number.isFinite(costs[node])) {
      nodeCounts[row]++;
      costSums[row] += costs[node];
    }
  }
  return {costs, assignments, nodeCounts, costSums};
}
