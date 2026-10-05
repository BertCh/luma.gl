// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {NetworkCSR} from '../network-reachability/network-reachability-oracle';

const NONE = 0xffffffff;

/** Inputs of {@link extractPathsOracle}. */
export type PathExtractionOracleInput = {
  predecessors: ArrayLike<number>;
  costs: ArrayLike<number>;
  targets: ArrayLike<number>;
  targetCount?: number;
  maxPathLength: number;
  nodeIds?: ArrayLike<number>;
  capacity: number;
  csr?: NetworkCSR;
  edgeIds?: ArrayLike<number>;
  edgeCapacity?: number;
};

/** Outputs of {@link extractPathsOracle}. */
export type PathExtractionOracleResult = {
  /** Clamped node prefix. */
  ids: number[];
  count: number;
  totalCount: number;
  overflow: number;
  /** `targets.length + 1` unclamped starts. */
  pathOffsets: number[];
  pathCosts: number[];
  found: number[];
  /** Clamped edge prefix. */
  edgeIds: number[];
  edgeCount: number;
  edgeTotalCount: number;
  edgeOverflow: number;
  edgePathOffsets: number[];
  truncated: boolean;
};

/** Walks one target; returns its root-to-target node list or `undefined` plus truncation. */
function walkPath(
  predecessors: ArrayLike<number>,
  costs: ArrayLike<number>,
  target: number,
  maxPathLength: number
): {nodes?: number[]; truncated: boolean} {
  const nodeCount = predecessors.length;
  if (target >= nodeCount || !Number.isFinite(Math.fround(costs[target]))) {
    return {truncated: false};
  }
  const reversed = [target];
  let node = target;
  for (;;) {
    const predecessor = predecessors[node] >>> 0;
    if (predecessor === NONE) {
      return {nodes: reversed.reverse(), truncated: false};
    }
    if (predecessor >= nodeCount) {
      return {truncated: false};
    }
    if (reversed.length >= maxPathLength) {
      return {truncated: true};
    }
    node = predecessor;
    reversed.push(node);
  }
}

/** Returns the CSR edge index (or stable ID) for the link tail -> head, or `NONE`. */
function resolveEdge(
  csr: NetworkCSR,
  costs: ArrayLike<number>,
  tail: number,
  head: number,
  edgeIds?: ArrayLike<number>
): number {
  const edgeCount = csr.neighbors.length;
  const start = Math.min(csr.offsets[tail], edgeCount);
  const end = Math.min(csr.offsets[tail + 1], edgeCount);
  let fallback = NONE;
  let result = NONE;
  for (let edge = start; edge < end; edge++) {
    if (csr.neighbors[edge] !== head) {
      continue;
    }
    if (fallback === NONE) {
      fallback = edge;
    }
    const weight = csr.weights[edge];
    if (weight >= 0 && Math.fround(costs[tail] + weight) === costs[head]) {
      result = edge;
      break;
    }
  }
  if (result === NONE) {
    result = fallback;
  }
  return result !== NONE && edgeIds ? edgeIds[result] : result;
}

/** CPU reference for `GPUNetworkPathExtraction` with identical tie-breaking and bounds. */
export function extractPathsOracle(input: PathExtractionOracleInput): PathExtractionOracleResult {
  const {predecessors, costs, targets, maxPathLength, nodeIds, capacity, csr, edgeIds} = input;
  const edgeCapacity = input.edgeCapacity ?? 0;
  const activeCount = Math.min(input.targetCount ?? targets.length, targets.length);
  const allIds: number[] = [];
  const allEdges: number[] = [];
  const pathOffsets: number[] = [];
  const edgePathOffsets: number[] = [];
  const pathCosts: number[] = [];
  const found: number[] = [];
  let truncated = false;
  for (let row = 0; row < targets.length; row++) {
    pathOffsets.push(allIds.length);
    edgePathOffsets.push(allEdges.length);
    const walk =
      row < activeCount
        ? walkPath(predecessors, costs, targets[row] >>> 0, maxPathLength)
        : {truncated: false, nodes: undefined};
    truncated ||= walk.truncated;
    found.push(walk.nodes ? 1 : 0);
    pathCosts.push(walk.nodes ? costs[targets[row]] : Infinity);
    if (!walk.nodes) {
      continue;
    }
    for (const node of walk.nodes) {
      allIds.push(nodeIds ? nodeIds[node] : node);
    }
    if (csr) {
      for (let link = 0; link + 1 < walk.nodes.length; link++) {
        allEdges.push(resolveEdge(csr, costs, walk.nodes[link], walk.nodes[link + 1], edgeIds));
      }
    }
  }
  pathOffsets.push(allIds.length);
  edgePathOffsets.push(allEdges.length);
  const count = Math.min(allIds.length, capacity);
  const edgeCount = Math.min(allEdges.length, edgeCapacity);
  return {
    ids: allIds.slice(0, count),
    count,
    totalCount: allIds.length,
    overflow: allIds.length > capacity || truncated ? 1 : 0,
    pathOffsets,
    pathCosts,
    found,
    edgeIds: allEdges.slice(0, edgeCount),
    edgeCount,
    edgeTotalCount: allEdges.length,
    edgeOverflow: allEdges.length > edgeCapacity || truncated ? 1 : 0,
    edgePathOffsets,
    truncated
  };
}
