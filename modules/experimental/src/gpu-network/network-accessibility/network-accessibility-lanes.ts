// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Default budget of lane-expanded nodes searched together in one batch. */
export const DEFAULT_LANE_EXPANDED_NODE_BUDGET = 1_000_000;
/** Default budget of lane-expansion scratch memory, in bytes. */
export const DEFAULT_LANE_SCRATCH_BYTE_BUDGET = 128 * 1024 * 1024;
/** Lane count used before the budget scaled it: never recommend fewer lanes than this. */
export const MINIMUM_RECOMMENDED_LANE_COUNT = 32;

/** Inputs of {@link recommendLaneCount}. */
export type RecommendLaneCountOptions = {
  /** Number of matrix rows, that is, shortest-path searches. */
  rowCount: number;
  /** Number of network nodes. */
  nodeCount: number;
  /** Number of CSR edges. */
  edgeCount: number;
  /** Expanded nodes per batch to aim for. Defaults to 1,000,000. */
  expandedNodeBudget?: number;
  /** Scratch bytes the expanded CSR may use. Defaults to 128 MB. */
  scratchByteBudget?: number;
};

/**
 * Recommends how many rows `GPUNetworkCostMatrix` searches per lane-expanded batch.
 *
 * Small graphs need many lanes to fill the GPU and to amortize the per-round dispatch cost of
 * every batch, so the target is `expandedNodeBudget / nodeCount` lanes (434 for a 2,304-node
 * grid). The result never drops below 32 for budget reasons alone, so large graphs behave as
 * before, and is then limited by the scratch budget (`lanes * (nodeCount + 2 * edgeCount)`
 * 4-byte words, which also keeps every expanded buffer under the default storage binding limit),
 * by `rowCount`, and by the uint32 index range of the expanded network. Pure; the matrix costs are
 * bit-identical for every lane count.
 */
export function recommendLaneCount(options: RecommendLaneCountOptions): number {
  const {
    rowCount,
    nodeCount,
    edgeCount,
    expandedNodeBudget = DEFAULT_LANE_EXPANDED_NODE_BUDGET,
    scratchByteBudget = DEFAULT_LANE_SCRATCH_BYTE_BUDGET
  } = options;
  const rows = Math.max(1, Math.floor(rowCount));
  const nodes = Math.max(1, nodeCount);
  const edges = Math.max(0, edgeCount);
  let lanes = Math.max(MINIMUM_RECOMMENDED_LANE_COUNT, Math.floor(expandedNodeBudget / nodes));
  lanes = Math.min(lanes, Math.floor(scratchByteBudget / 4 / (nodes + 2 * edges)));
  lanes = Math.min(lanes, Math.floor((2 ** 32 - 2) / nodes));
  if (edges > 0) {
    lanes = Math.min(lanes, Math.floor((2 ** 32 - 1) / edges));
  }
  return Math.max(1, Math.min(rows, lanes));
}
