// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validatePackedUint32View,
  validatePackedView,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {validateGPUStatusPort, type GPUSpatialWeightsPort} from '../contracts/index';

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
  /**
   * Optional CPU-side description of how the weights were built. Producers in this package attach
   * it to the weights they expose; uploaded weights may omit it, which consumers read as
   * `transform: 'unknown'`, `symmetric: 'unknown'`.
   */
  info?: GPUSpatialWeightsInfo;
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
 * Transform state of a weights matrix (spdep `style`, PySAL `transform`). `'row'` means every row
 * with a positive weight sums to 1; consumers re-standardize such rows over the neighbors that
 * remain after masking (see {@link getGPUSpatialWeightsNeighborLoopWGSL}).
 */
export type GPUSpatialWeightsTransformState =
  | 'unknown'
  | 'raw'
  | 'binary'
  | 'row'
  | 'double'
  | 'variance'
  | 'kernel'
  | 'symmetric';

/** How a weights matrix was built, for provenance and legends. */
export type GPUSpatialWeightsProvenance = {
  /** Producer, for example `'GPUNeighborSearch'` or `'upload'`. */
  source: string;
  /** Neighbor criterion, for example `'knn'`, `'distance-band'`, `'queen'`, `'rook'`. */
  criterion?: string;
  /** `k` of a kNN criterion. */
  k?: number;
  /** Distance band or kernel bandwidth, in the units of the spatial context. */
  radius?: number;
  /** Order of a higher-order or cumulative contiguity matrix. */
  order?: number;
};

/** CPU-side metadata carried with a {@link GPUSpatialWeights}. */
export type GPUSpatialWeightsInfo = {
  /** Number of focus rows, `offsets.length - 1`. */
  rowCount: number;
  /** Transform applied to the weights. */
  transform: GPUSpatialWeightsTransformState;
  /** Whether `w_ij = w_ji` for every pair, when known. */
  symmetric: boolean | 'unknown';
  /**
   * Whether rows may list themselves (`w_ii` slots, for example `K(0)` of a kernel). Consumers skip
   * self slots; see each consumer's `selfWeight` handling.
   */
  hasSelfWeights?: boolean;
  /** Producer and construction parameters. */
  provenance?: GPUSpatialWeightsProvenance;
};

/**
 * Treatment of islands: included rows with no included neighbor of positive weight other than
 * themselves. `'exclude'` (default) removes them from the analysis (NaN output, not counted in
 * `n` or the moments); `'keep'` keeps them in `n` and the moments, as esda does.
 */
export type GPUIslandPolicy = 'exclude' | 'keep';

/** Default {@link GPUIslandPolicy}. */
export const GPU_DEFAULT_ISLAND_POLICY: GPUIslandPolicy = 'exclude';

/** Validates an island policy and returns it, defaulting to {@link GPU_DEFAULT_ISLAND_POLICY}. */
export function resolveGPUIslandPolicy(
  id: string,
  islands: GPUIslandPolicy | undefined
): GPUIslandPolicy {
  const policy = islands ?? GPU_DEFAULT_ISLAND_POLICY;
  if (policy !== 'exclude' && policy !== 'keep') {
    throw new Error(`${id} islands must be 'exclude' or 'keep'`);
  }
  return policy;
}

/** Returns whether a consumer should re-standardize rows over the neighbors left after masking. */
export function isGPUSpatialWeightsRowStandardized(weights: GPUSpatialWeights): boolean {
  return weights.info?.transform === 'row';
}

/**
 * WGSL statements visiting the active neighbor slots of focus row `row`: the single neighbor
 * definition shared by the summary, global and local statistics. A slot is active when its
 * neighbor is a valid row other than the focus row, is included (`isIncluded`), and its weight is
 * positive. `action` runs with `slot`, `neighbor` (row index) and `weight` in scope. A row with no
 * active slot is an island.
 *
 * Requires bindings `offsets`, `neighbors`, `weights` and the constants `ROWS` and `CAPACITY`
 * (`neighbors.length`); offsets are clamped to the capacity. Slots are visited in CSR order, so
 * per-row sums are reproducible.
 *
 * @param isIncluded WGSL boolean expression of `neighbor`, for example
 * `rowPositions[rowPositionsOffset + neighbor] != INVALID_POSITION`.
 * @param action WGSL statements run once per active slot.
 * @param row WGSL u32 expression of the focus row. Defaults to `index`.
 * @internal
 */
export function getGPUSpatialWeightsNeighborLoopWGSL(
  isIncluded: string,
  action: string,
  row = 'index'
): string {
  return /* wgsl */ `{
    let activeRow = ${row};
    let activeBegin = min(offsets[offsetsOffset + activeRow], CAPACITY);
    let activeEnd = min(offsets[offsetsOffset + activeRow + 1u], CAPACITY);
    for (var slot = activeBegin; slot < activeEnd; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor >= ROWS || neighbor == activeRow) {
        continue;
      }
      let weight = weights[weightsOffset + slot];
      if (!(weight > 0.0) || !(${isIncluded})) {
        continue;
      }
      ${action}
    }
  }`;
}

/**
 * Validates the packed formats and lengths of a {@link GPUSpatialWeights} and returns its row
 * count (`offsets.length - 1`).
 *
 * Only shapes, status formats and declared metadata are validated here. The ordering invariants
 * (sorted-unique rows, finite non-negative weights) are the producer's responsibility; for CSR
 * built elsewhere and uploaded, run {@link GPUSpatialWeightsValidation} once after upload.
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
  const rowCount = weights.offsets.length - 1;
  if (weights.status) {
    validateGPUStatusPort(`${id} ${name}.status`, weights.status);
  }
  const {info} = weights;
  if (info) {
    if (info.rowCount !== rowCount) {
      throw new Error(`${id} ${name}.info.rowCount must equal ${name}.offsets.length - 1`);
    }
    if (!TRANSFORM_STATES.includes(info.transform)) {
      throw new Error(`${id} ${name}.info.transform is not a known transform state`);
    }
    if (info.symmetric !== true && info.symmetric !== false && info.symmetric !== 'unknown') {
      throw new Error(`${id} ${name}.info.symmetric must be true, false or 'unknown'`);
    }
  }
  return rowCount;
}

/** Transform states a {@link GPUSpatialWeightsInfo} may declare. */
const TRANSFORM_STATES: readonly GPUSpatialWeightsTransformState[] = [
  'unknown',
  'raw',
  'binary',
  'row',
  'double',
  'variance',
  'kernel',
  'symmetric'
];

/**
 * Returns `weights` with `info` attached and the given status views merged over its own. The
 * views are shared with `weights`; only the wrapper object is new. Producers use it to build
 * their `outputWeights`.
 *
 * @param weights Caller-owned views.
 * @param description Metadata and status to attach. `status` entries replace those of `weights`.
 */
export function describeGPUSpatialWeights(
  weights: GPUSpatialWeights,
  description: {info: GPUSpatialWeightsInfo; status?: GPUSpatialWeights['status']}
): GPUSpatialWeights {
  const status = {...weights.status, ...description.status};
  return {
    ...weights,
    info: description.info,
    status: Object.keys(status).length > 0 ? status : undefined
  };
}

/**
 * Suggests a new slot capacity for the grow loop of capacity-bounded producers
 * (`GPUNeighborSearch`, `GPUContiguityWeights`, `GPULatticeWeights`, `GPUSpatialWeightsAlgebra`).
 *
 * The loop: build with a guess, read `outputWeights.status.requiredCount` and
 * `status.overflow` after a frame, and when the required count exceeds the capacity allocate a
 * view of `getGPUSpatialWeightsCapacity(requiredCount, capacity)` slots and rebuild. Returns
 * `currentCapacity` when `requiredCount` already fits; otherwise `requiredCount` plus `headroom`
 * (a fraction, default 0.25) so a slowly growing input does not rebuild every frame.
 *
 * @param requiredCount Exact unclamped slot requirement read back from the producer.
 * @param currentCapacity Slot capacity the producer ran with.
 * @param headroom Extra fraction of `requiredCount` to allocate on growth. Defaults to 0.25.
 */
export function getGPUSpatialWeightsCapacity(
  requiredCount: number,
  currentCapacity: number,
  headroom = 0.25
): number {
  if (!(Number.isFinite(requiredCount) && requiredCount >= 0)) {
    throw new Error('getGPUSpatialWeightsCapacity requiredCount must be a non-negative number');
  }
  if (requiredCount <= currentCapacity) {
    return currentCapacity;
  }
  return Math.ceil(requiredCount * (1 + Math.max(headroom, 0)));
}
