// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {RATE_REFERENCE} from './gpu-rate-standardization-reference';

/** Rows of the esda rate fixture (the last row is an island of the neighbor graph). */
export const RATE_ROW_COUNT = RATE_REFERENCE.events.length;

/** Restores NaN from the `null` the generated tables use for it. */
export function toFloat64(values: readonly (number | null)[]): number[] {
  return values.map(value => (value === null ? NaN : value));
}

/** Datasets of the fixture: `full` is every row, `missing` has three invalid rows. */
export type RateDataset = 'full' | 'missing' | 'under';

/** The events and populations of a dataset as the f32 the GPU reads. */
export function getRateInputs(dataset: RateDataset): {
  events: Float32Array;
  populations: Float32Array;
} {
  const R = RATE_REFERENCE;
  switch (dataset) {
    case 'full':
      return {events: Float32Array.from(R.events), populations: Float32Array.from(R.population)};
    case 'missing':
      return {
        events: Float32Array.from(toFloat64(R.eventsMissing)),
        populations: Float32Array.from(R.populationMissing)
      };
    case 'under':
      return {
        events: Float32Array.from(R.eventsUnder),
        populations: Float32Array.from(R.populationUnder)
      };
  }
}

/** Esda's standardized rates of a dataset (`assuncao_rate`), NaN for the invalid rows. */
export function getRateReferenceZ(dataset: RateDataset): number[] {
  const R = RATE_REFERENCE;
  return toFloat64(dataset === 'full' ? R.fullZ : dataset === 'missing' ? R.missingZ : R.underZ);
}

/**
 * Double-precision implementation of `esda.smoothing.assuncao_rate` over the valid rows (unmasked,
 * finite events >= 0, finite population > 0), NaN elsewhere. Also returns the global quantities
 * `[n, b, s2, a, totalPopulation]`.
 */
export function computeAssuncaoReisOracle(
  events: ArrayLike<number>,
  populations: ArrayLike<number>
): {z: number[]; statistics: number[]; negativeVarianceRows: number} {
  const valid = Array.from(events, (event, row) => {
    const population = populations[row];
    return Number.isFinite(event) && Number.isFinite(population) && event >= 0 && population > 0;
  });
  let count = 0;
  let eventSum = 0;
  let populationSum = 0;
  for (const [row, isValid] of valid.entries()) {
    if (isValid) {
      count++;
      eventSum += events[row];
      populationSum += populations[row];
    }
  }
  const b = eventSum / populationSum;
  let weightedSquares = 0;
  for (const [row, isValid] of valid.entries()) {
    if (isValid) {
      weightedSquares += populations[row] * (events[row] / populations[row] - b) ** 2;
    }
  }
  const s2 = weightedSquares / populationSum;
  const a = s2 - b / (populationSum / count);
  let negativeVarianceRows = 0;
  const z = Array.from(events, (event, row) => {
    if (!valid[row]) {
      return NaN;
    }
    const population = populations[row];
    const raw = a + b / population;
    if (raw < 0) {
      negativeVarianceRows++;
    }
    const variance = raw < 0 ? b / population : raw;
    return variance > 0 ? (event / population - b) / Math.sqrt(variance) : NaN;
  });
  return {z, statistics: [count, b, s2, a, populationSum], negativeVarianceRows};
}
