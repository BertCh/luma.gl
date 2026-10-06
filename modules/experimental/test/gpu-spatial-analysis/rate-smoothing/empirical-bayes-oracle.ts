// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Result of {@link computeEmpiricalBayesOracle}. */
export type EmpiricalBayesOracle = {
  standardized: number[];
  smoothed: number[];
  raw: number[];
  count: number;
  pooledRate: number;
  weightedRateVariance: number;
  priorVariance: number;
};

/**
 * Double-precision transcription of esda `smoothing.assuncao_rate` and `Empirical_Bayes`, over
 * the rows with finite events and a finite positive population (and a nonzero mask).
 */
export function computeEmpiricalBayesOracle(input: {
  events: ArrayLike<number>;
  populations: ArrayLike<number>;
  mask?: ArrayLike<number>;
}): EmpiricalBayesOracle {
  const {events, populations, mask} = input;
  const rows = events.length;
  const included = Array.from({length: rows}, (_, row) => {
    return (
      (!mask || mask[row] !== 0) &&
      Number.isFinite(events[row]) &&
      Number.isFinite(populations[row]) &&
      populations[row] > 0
    );
  });
  const indices = included.flatMap((flag, row) => (flag ? [row] : []));
  const eventSum = indices.reduce((total, row) => total + events[row], 0);
  const populationSum = indices.reduce((total, row) => total + populations[row], 0);
  const pooledRate = eventSum / populationSum;
  const weightedRateVariance =
    indices.reduce(
      (total, row) => total + populations[row] * (events[row] / populations[row] - pooledRate) ** 2,
      0
    ) / populationSum;
  const priorVariance = weightedRateVariance - pooledRate / (populationSum / indices.length);
  const standardized: number[] = [];
  const smoothed: number[] = [];
  const raw: number[] = [];
  for (let row = 0; row < rows; row++) {
    if (!included[row]) {
      standardized.push(NaN);
      smoothed.push(NaN);
      raw.push(NaN);
      continue;
    }
    const rate = events[row] / populations[row];
    const noise = pooledRate / populations[row];
    const rawVariance = priorVariance + noise;
    const variance = rawVariance < 0 ? noise : rawVariance;
    standardized.push((rate - pooledRate) / Math.sqrt(variance));
    const weight = priorVariance / rawVariance;
    smoothed.push(weight * rate + (1 - weight) * pooledRate);
    raw.push(rate);
  }
  return {
    standardized,
    smoothed,
    raw,
    count: indices.length,
    pooledRate,
    weightedRateVariance,
    priorVariance
  };
}
