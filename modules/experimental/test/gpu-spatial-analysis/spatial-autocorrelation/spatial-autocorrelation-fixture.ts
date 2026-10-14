// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {REFERENCE} from './spatial-autocorrelation-reference';
import type {OracleWeights} from './spatial-autocorrelation-oracle';

/** Rows of the esda reference fixture. */
export const REFERENCE_ROW_COUNT = REFERENCE.values.length;

/** Rows removed in the `masked` configuration of the reference (the island is among them). */
export const REFERENCE_MASKED_ROWS: readonly number[] = REFERENCE.maskedRows;

/** The reference values as the f32 the GPU reads. */
export function getReferenceValues(): Float32Array {
  return Float32Array.from(REFERENCE.values);
}

/**
 * CSR weights of the reference fixture: binary neighbors `i +- 1, i +- 2` among rows
 * `0..island - 1`, and the island row without neighbors. Optionally row-standardized by the full
 * neighbor count (what a producer writes before any masking).
 */
export function getReferenceWeights(options: {rowStandardize?: boolean} = {}): OracleWeights {
  const island = REFERENCE.island;
  const offsets = [0];
  const neighbors: number[] = [];
  const weights: number[] = [];
  for (let row = 0; row < REFERENCE_ROW_COUNT; row++) {
    if (row !== island) {
      const rowNeighbors = [row - 2, row - 1, row + 1, row + 2].filter(
        neighbor => neighbor >= 0 && neighbor < island
      );
      for (const neighbor of rowNeighbors) {
        neighbors.push(neighbor);
        weights.push(options.rowStandardize ? 1 / rowNeighbors.length : 1);
      }
    }
    offsets.push(neighbors.length);
  }
  return {
    offsets: Uint32Array.from(offsets),
    neighbors: Uint32Array.from(neighbors),
    weights: Float32Array.from(weights)
  };
}

/** Mask that removes {@link REFERENCE_MASKED_ROWS}. */
export function getReferenceMask(): Uint32Array {
  const mask = new Uint32Array(REFERENCE_ROW_COUNT).fill(1);
  for (const row of REFERENCE_MASKED_ROWS) {
    mask[row] = 0;
  }
  return mask;
}
