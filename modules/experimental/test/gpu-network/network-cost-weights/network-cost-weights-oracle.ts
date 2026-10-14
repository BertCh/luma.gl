// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUNetworkCostWeightsSettings} from '../../../src/gpu-network/network-cost-weights';

/** CPU weights of `GPUNetworkCostWeights`, the reference for its GPU specs. */
export type NetworkCostWeightsReference = {
  /** `weightsRowCount + 1` offsets, clamped to the capacity. */
  offsets: number[];
  /** Neighbor per written slot (`min(requiredCount, capacity)` entries). */
  neighbors: number[];
  /** Decay weight per written slot. */
  weights: number[];
  /** Cost per written slot. */
  distances: number[];
  /** Unclamped number of in-range entries. */
  requiredCount: number;
  /** 1 when `requiredCount` exceeds the capacity. */
  overflow: number;
};

/** The decay of `GPUNetworkCostWeightsDecay`, in float64. */
export function getNetworkCostDecay(cost: number, settings: GPUNetworkCostWeightsSettings): number {
  const beta = settings.beta ?? 0;
  switch (settings.decay ?? 'none') {
    case 'exponential':
      return Math.exp(-beta * cost);
    case 'power':
      return Math.max(cost, settings.minimumCost ?? 1) ** -beta;
    case 'gaussian': {
      const edge = Math.exp(-0.5);
      const ratio = cost / settings.costLimit;
      return Math.max((Math.exp(-0.5 * ratio * ratio) - edge) / (1 - edge), 0);
    }
    default:
      return 1;
  }
}

/**
 * Builds the weights a cost matrix gives, with the documented rules: an entry is a neighbor when
 * its cost is finite, non-negative and `<= costLimit`; rows in order, neighbors ascending; slots past
 * the capacity dropped and offsets clamped.
 */
export function getNetworkCostWeightsReference(props: {
  costs: ArrayLike<number>;
  rowCount: number;
  columns?: ArrayLike<number>;
  orientation: 'rows' | 'columns';
  settings: GPUNetworkCostWeightsSettings;
  capacity: number;
}): NetworkCostWeightsReference {
  const {costs, rowCount, columns, orientation, settings, capacity} = props;
  const columnCount = costs.length / rowCount;
  const locationCount = columns ? columns.length : columnCount;
  const weightsRowCount = orientation === 'rows' ? rowCount : locationCount;
  const neighborSpace = orientation === 'rows' ? locationCount : rowCount;
  const offsets = [0];
  const neighbors: number[] = [];
  const weights: number[] = [];
  const distances: number[] = [];
  let slot = 0;
  for (let row = 0; row < weightsRowCount; row++) {
    for (let neighbor = 0; neighbor < neighborSpace; neighbor++) {
      const location = orientation === 'rows' ? neighbor : row;
      const matrixRow = orientation === 'rows' ? row : neighbor;
      const column = columns ? columns[location] : location;
      if (column >= columnCount) {
        continue;
      }
      // The GPU reads float32 costs; the fixtures store float32-exact values.
      const cost = Math.fround(costs[matrixRow * columnCount + column]);
      if (!Number.isFinite(cost) || cost < 0 || cost > Math.fround(settings.costLimit)) {
        continue;
      }
      if (slot < capacity) {
        neighbors.push(neighbor);
        weights.push(getNetworkCostDecay(cost, settings));
        distances.push(cost);
      }
      slot++;
    }
    offsets.push(slot);
  }
  return {
    offsets: offsets.map(offset => Math.min(offset, capacity)),
    neighbors,
    weights,
    distances,
    requiredCount: slot,
    overflow: slot > capacity ? 1 : 0
  };
}
