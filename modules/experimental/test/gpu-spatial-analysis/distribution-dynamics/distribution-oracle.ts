// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  computeLocalMoranOracle,
  type OracleWeights
} from '../spatial-autocorrelation/spatial-autocorrelation-oracle';

const NO_CLASS = 0xffffffff;

/** Class of a value under the shared rule: inner edges `e[1..k-1]` at or below the value. */
export function getOracleClass(
  value: number,
  edges: readonly number[],
  classCount: number
): number {
  if (classCount < 1 || Number.isNaN(value)) {
    return 0xffffffff;
  }
  let below = 0;
  for (let edge = 1; edge < classCount; edge++) {
    if (edges[edge] <= value) below++;
  }
  return below;
}

/** Oracle transition result. */
export type OracleTransitions = {
  counts: number[];
  probabilities: number[];
  rowTotals: number[];
  ignored: number;
};

/** Brute-force transition counts, optionally conditioned. Layout `(c * K + from) * K + to`. */
export function computeTransitionOracle(options: {
  classes: ArrayLike<number>;
  conditions?: ArrayLike<number>;
  mask?: ArrayLike<number>;
  rows: number;
  periods: number;
  classCount: number;
  conditionCount?: number;
  periodLag?: number;
}): OracleTransitions {
  const {rows, periods, classCount} = options;
  const conditionCount = options.conditionCount ?? 1;
  const lag = options.periodLag ?? 1;
  const counts = new Array(conditionCount * classCount * classCount).fill(0);
  let ignored = 0;
  for (let period = 0; period + lag < periods; period++) {
    for (let row = 0; row < rows; row++) {
      const from = options.classes[period * rows + row];
      const to = options.classes[(period + lag) * rows + row];
      const condition = options.conditions ? options.conditions[period * rows + row] : 0;
      const selected = options.mask ? options.mask[row] !== 0 : true;
      if (selected && from < classCount && to < classCount && condition < conditionCount) {
        counts[(condition * classCount + from) * classCount + to]++;
      } else {
        ignored++;
      }
    }
  }
  const rowTotals: number[] = [];
  const probabilities = counts.map(() => 0);
  for (let state = 0; state < conditionCount * classCount; state++) {
    let total = 0;
    for (let to = 0; to < classCount; to++) total += counts[state * classCount + to];
    rowTotals.push(total);
    for (let to = 0; to < classCount; to++) {
      probabilities[state * classCount + to] =
        total > 0 ? counts[state * classCount + to] / total : 0;
    }
  }
  return {counts, probabilities, rowTotals, ignored};
}

/** R-7 linear-interpolation quantile of a sorted array. */
export function getOracleQuantile(sorted: readonly number[], probability: number): number {
  const position = (sorted.length - 1) * probability;
  const low = Math.floor(position);
  const high = Math.ceil(position);
  return sorted[low] + (sorted[high] - sorted[low]) * (position - low);
}

/**
 * LISA Markov states per row and period (period-major), in double precision on top of the local
 * Moran oracle. A row-period that is masked, not finite or without an included neighbor is
 * `0xffffffff`. Gated: the quadrant of an analytically significant row, else 0. Ungated: the
 * quadrant from the signs of the centered value and the lag, `0xffffffff` when either is zero.
 */
export function computeLisaStatesOracle(options: {
  weights: OracleWeights;
  values: Float32Array;
  rows: number;
  periods: number;
  gating: 'significance' | 'none';
  significanceLevel?: number;
  mask?: Uint32Array;
  rowStandardized?: boolean;
}): number[] {
  const {rows, periods} = options;
  const states: number[] = [];
  for (let period = 0; period < periods; period++) {
    const values = options.values.subarray(period * rows, (period + 1) * rows);
    const result = computeLocalMoranOracle({
      weights: options.weights,
      values,
      mask: options.mask,
      parameters: {significanceLevel: options.significanceLevel ?? 0.05},
      rowStandardized: options.rowStandardized
    });
    for (let row = 0; row < rows; row++) {
      if (!Number.isFinite(values[row]) || result.neighborCounts[row] === 0) {
        states.push(NO_CLASS);
      } else if (options.gating === 'significance') {
        states.push(result.quadrants[row]);
      } else {
        const centered = values[row] - result.moments.mean;
        const lag = result.spatialLag[row];
        states.push(
          centered === 0 || lag === 0
            ? NO_CLASS
            : centered > 0
              ? lag > 0
                ? 1
                : 4
              : lag > 0
                ? 2
                : 3
        );
      }
    }
  }
  return states;
}
