// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validatePackedUint32View,
  validatePackedView,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUSpatialWeightsPort} from '../contracts/index';

/**
 * A spatial-weights matrix in CSR form: row `i` lists the neighbors `j` of focus row `i` and the
 * weight `w_ij` of each.
 *
 * This is the shared weights structure of the statistics contributors. `GPUNeighborSearch`
 * writes it on the GPU (kNN or distance band); `GPUGlobalSpatialStatistics`,
 * `GPULocalPermutationTest` and `GPUGlobalPermutationTest` read it. Applications can also upload
 * one built elsewhere, for example polygon contiguity computed once on the CPU.
 *
 * Invariants every producer guarantees and every consumer may rely on:
 * - `offsets` holds `rows + 1` non-decreasing entries with `offsets[0] = 0`; row `i` occupies
 *   slots `[offsets[i], offsets[i + 1])` of `neighbors`, `weights` and `distances`.
 * - `offsets[rows] <= neighbors.length` (capacity); slots past `offsets[rows]` are unspecified.
 * - Within a row, neighbor IDs are strictly ascending (no duplicates), so a consumer can
 *   binary-search `w_ji` in row `j`.
 * - Self-join weights never list a row as its own neighbor (`w_ii = 0`).
 * - Weights are finite and non-negative.
 *
 * The row and neighbor-ID spaces may differ for cross (query to target) weights; the statistics
 * contributors require square self-join weights where `neighbors` index the same rows as `offsets`.
 */
export type GPUSpatialWeights = GPUSpatialWeightsPort & {
  /** `rows + 1` exclusive row offsets into the neighbor slots. */
  offsets: GraphDataView<'uint32'>;
  /** Neighbor row IDs, ascending within each row. Its length is the slot capacity. */
  neighbors: GraphDataView<'uint32'>;
  /** Weight `w_ij` per slot, aligned with `neighbors`. */
  weights: GraphDataView<'float32'>;
  /** Optional distance `|p_i - p_j|` per slot, aligned with `neighbors`. */
  distances?: GraphDataView<'float32'>;
};

/**
 * Validates the packed formats and lengths of a {@link GPUSpatialWeights} and returns its row
 * count (`offsets.length - 1`).
 *
 * Only shapes are validated here; the ordering invariants are the producer's responsibility.
 *
 * @param id Contributor ID used in error messages.
 * @param weights Weights to validate.
 * @param name Prop name used in error messages. Defaults to `'weights'`.
 */
export function validateGPUSpatialWeights(
  id: string,
  weights: GPUSpatialWeights,
  name = 'weights'
): number {
  validatePackedUint32View(weights.offsets, `${id} ${name}.offsets`);
  validatePackedUint32View(weights.neighbors, `${id} ${name}.neighbors`);
  validatePackedView(weights.weights, ['float32'], `${id} ${name}.weights`);
  if (weights.offsets.length < 2) {
    throw new Error(`${id} ${name}.offsets must hold at least two entries`);
  }
  if (weights.neighbors.length < 1) {
    throw new Error(`${id} ${name}.neighbors must hold at least one slot`);
  }
  if (weights.weights.length !== weights.neighbors.length) {
    throw new Error(`${id} ${name}.weights length must equal ${name}.neighbors length`);
  }
  if (weights.distances) {
    validatePackedView(weights.distances, ['float32'], `${id} ${name}.distances`);
    if (weights.distances.length !== weights.neighbors.length) {
      throw new Error(`${id} ${name}.distances length must equal ${name}.neighbors length`);
    }
  }
  return weights.offsets.length - 1;
}
