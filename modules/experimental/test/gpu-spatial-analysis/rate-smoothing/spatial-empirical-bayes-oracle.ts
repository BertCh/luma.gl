// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Result of {@link computeSpatialEmpiricalBayesOracle}, one entry per row. */
export type SpatialEmpiricalBayesOracle = {
  spatialRate: number[];
  smoothed: number[];
  priorMean: number[];
  priorVariance: number[];
};

/**
 * Double-precision transcription of esda 2.10 `Spatial_Rate` and `Spatial_Empirical_Bayes`
 * (neighborhood = self + included neighbors, binary weights, variance clamped at zero).
 */
export function computeSpatialEmpiricalBayesOracle(input: {
  events: ArrayLike<number>;
  populations: ArrayLike<number>;
  offsets: ArrayLike<number>;
  neighbors: ArrayLike<number>;
  mask?: ArrayLike<number>;
}): SpatialEmpiricalBayesOracle {
  const {events, populations, offsets, neighbors, mask} = input;
  const rows = events.length;
  const included = (row: number) =>
    (!mask || mask[row] !== 0) &&
    Number.isFinite(events[row]) &&
    Number.isFinite(populations[row]) &&
    populations[row] > 0;
  const result: SpatialEmpiricalBayesOracle = {
    spatialRate: [],
    smoothed: [],
    priorMean: [],
    priorVariance: []
  };
  for (let row = 0; row < rows; row++) {
    if (!included(row)) {
      for (const column of Object.values(result)) column.push(NaN);
      continue;
    }
    const members = [row];
    for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
      const neighbor = neighbors[slot];
      if (neighbor !== row && included(neighbor)) members.push(neighbor);
    }
    const eventSum = members.reduce((total, member) => total + events[member], 0);
    const populationSum = members.reduce((total, member) => total + populations[member], 0);
    const mean = eventSum / populationSum;
    const variance = Math.max(
      members.reduce(
        (total, member) =>
          total + populations[member] * (events[member] / populations[member] - mean) ** 2,
        0
      ) /
        populationSum -
        mean / (populationSum / members.length),
      0
    );
    const denominator = variance + mean / populations[row];
    const weight = denominator > 0 ? variance / denominator : 0;
    result.spatialRate.push(mean);
    result.priorMean.push(mean);
    result.priorVariance.push(variance);
    result.smoothed.push(weight * (events[row] / populations[row]) + (1 - weight) * mean);
  }
  return result;
}
