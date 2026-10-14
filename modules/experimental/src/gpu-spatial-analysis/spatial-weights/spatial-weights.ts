// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {
  validateGPUStatusPort,
  type GPUSpatialWeightsPort
} from '../../analysis-core/contracts/index';
import type {GPUCapacityPlan} from '../../analysis-core/contracts/capacity-planning';
import {validateCapacity, validatePackedView} from '../../utils/gpu-contributor-utils';

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
  information?: GPUSpatialWeightsInformation;
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
export type GPUSpatialWeightsInformation = {
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
  return weights.information?.transform === 'row';
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
  validatePackedView(weights.offsets, ['uint32'], `${id} ${name}.offsets`);
  validatePackedView(weights.neighbors, ['uint32'], `${id} ${name}.neighbors`);
  validatePackedView(weights.weights, ['float32'], `${id} ${name}.weights`);
  validateCapacity(id, `${name}.offsets length`, weights.offsets.length, 2);
  validateCapacity(id, `${name}.neighbors length`, weights.neighbors.length);
  if (weights.weights.length !== weights.neighbors.length) {
    throw new Error(`${id} ${name}.weights length must equal neighbors length`);
  }
  if (weights.distances) {
    validatePackedView(weights.distances, ['float32'], `${id} ${name}.distances`);
    if (weights.distances.length !== weights.neighbors.length) {
      throw new Error(`${id} ${name}.distances length must equal neighbors length`);
    }
  }
  const rowCount = weights.offsets.length - 1;
  if (weights.status) {
    validateGPUStatusPort(`${id} ${name}.status`, weights.status);
  }
  const {information} = weights;
  if (information) {
    if (information.rowCount !== rowCount) {
      throw new Error(`${id} ${name}.information.rowCount must equal the row count`);
    }
    if (!TRANSFORM_STATES.includes(information.transform)) {
      throw new Error(`${id} ${name}.information.transform is unknown`);
    }
    if (![true, false, 'unknown'].includes(information.symmetric)) {
      throw new Error(`${id} ${name}.information.symmetric must be a boolean or 'unknown'`);
    }
  }
  return rowCount;
}

/** Transform states a {@link GPUSpatialWeightsInformation} may declare. */
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
 * Caller-owned output of a capacity-bounded weights producer (`GPUContiguityWeights`,
 * `GPULatticeWeights`, `GPUSpatialWeightsAlgebra`): the CSR views plus the one-row status scalars
 * the producer writes.
 */
export type GPUSpatialWeightsOutput = GPUSpatialWeights & {
  status: {
    /** One when the neighbors (or an intermediate stage) did not fit their capacity, else 0. */
    overflow: GraphDataView<'uint32'>;
    /**
     * Optional unclamped slot requirement: exact unless `candidateOverflow` (or the producer's
     * documented intermediate stage) overflowed, in which case it is a lower bound.
     */
    requiredCount?: GraphDataView<'uint32'>;
    /** Optional: one when an intermediate candidate stage overflowed (see each producer). */
    candidateOverflow?: GraphDataView<'uint32'>;
  };
};

/**
 * Validates a {@link GPUSpatialWeightsOutput} and returns its row count.
 *
 * @param id Contributor ID used in error messages.
 * @param output Output to validate.
 */
export function validateGPUSpatialWeightsOutput(
  id: string,
  output: GPUSpatialWeightsOutput
): number {
  const rowCount = validateGPUSpatialWeights(id, output, 'output');
  if (!output.status?.overflow) {
    throw new Error(`${id} output.status.overflow is required`);
  }
  return rowCount;
}

/** Returns the views of a CSR, for disjointness and graph-membership checks. @internal */
export function getGPUSpatialWeightsViews(
  weights: GPUSpatialWeights
): (GraphDataView<'uint32'> | GraphDataView<'float32'> | undefined)[] {
  return [weights.offsets, weights.neighbors, weights.weights, weights.distances];
}

/** Returns the status views of a {@link GPUSpatialWeightsOutput}. @internal */
export function getGPUSpatialWeightsStatusViews(
  output: GPUSpatialWeightsOutput
): (GraphDataView<'uint32'> | undefined)[] {
  return [output.status.overflow, output.status.requiredCount, output.status.candidateOverflow];
}

/**
 * Returns `weights` with `information` attached and the given status views merged over its own.
 * The views are shared with `weights`; only the wrapper object is new. Producers use it to build
 * their `outputWeights`.
 *
 * @param weights Caller-owned views.
 * @param description Metadata and status to attach. `status` entries replace those of `weights`.
 */
export function describeGPUSpatialWeights(
  weights: GPUSpatialWeights,
  description: {information: GPUSpatialWeightsInformation; status?: GPUSpatialWeights['status']}
): GPUSpatialWeights {
  const status = {...weights.status, ...description.status};
  return {
    ...weights,
    information: description.information,
    status: Object.keys(status).length > 0 ? status : undefined
  };
}

/** Bytes per CSR slot of the caller-owned views: a uint32 neighbor ID and a float32 weight. */
export const GPU_SPATIAL_WEIGHTS_SLOT_BYTE_LENGTH = 8;
/** Extra bytes per slot when the CSR carries float32 `distances`. */
export const GPU_SPATIAL_WEIGHTS_DISTANCE_BYTE_LENGTH = 4;
/** Default growth headroom of {@link getGPUSpatialWeightsCapacityPlan}: 25 percent. */
export const GPU_SPATIAL_WEIGHTS_CAPACITY_HEADROOM = 0.25;

/** Inputs to {@link getGPUSpatialWeightsCapacityPlan}. */
export type GPUSpatialWeightsCapacityPlanOptions = {
  /** Focus rows of the CSR (`offsets.length - 1`). */
  rowCount: number;
  /** Slot capacity the producer ran with, or the first guess. */
  capacity: number;
  /** `output.status.requiredCount` read back after a frame, when the producer wrote it. */
  requiredCount?: number;
  /** Whether `distances` are allocated. Defaults to false. */
  distances?: boolean;
  /** Extra fraction of `requiredCount` allocated on growth. Defaults to 0.25. */
  headroom?: number;
};

/** Slot plan and caller-owned output bytes of a {@link GPUSpatialWeightsOutput}. */
export type GPUSpatialWeightsCapacityPlan = GPUCapacityPlan & {
  /** Bytes of `offsets` (`(rowCount + 1) * 4`). */
  offsetsByteLength: number;
};

/**
 * The grow step of every capacity-bounded weights producer (`GPUNeighborSearch`,
 * `GPUContiguityWeights`, `GPULatticeWeights`, `GPUSpatialWeightsAlgebra`): plans the slot
 * capacity of the next allocation and the bytes of the caller-owned CSR views.
 *
 * The loop: build with a guess, read `output.status.requiredCount` (and `overflow`) after a
 * frame, and when the required count exceeds the capacity allocate `plan.capacity` slots and
 * rebuild. The capacity stays unchanged while `requiredCount` fits; otherwise it is
 * `requiredCount` plus `headroom`, so a slowly growing input does not rebuild every frame.
 * `estimatedOutputBytes` is the bytes of `neighbors`, `weights` (and `distances`) at that
 * capacity plus `offsets`, from the same per-slot constants the recipes allocate with.
 */
export function getGPUSpatialWeightsCapacityPlan(
  options: GPUSpatialWeightsCapacityPlanOptions
): GPUSpatialWeightsCapacityPlan {
  const {rowCount, capacity} = options;
  validateCapacity('getGPUSpatialWeightsCapacityPlan', 'rowCount', rowCount);
  validateCapacity('getGPUSpatialWeightsCapacityPlan', 'capacity', capacity);
  const headroom = Math.max(options.headroom ?? GPU_SPATIAL_WEIGHTS_CAPACITY_HEADROOM, 0);
  const requiredCount = options.requiredCount ?? capacity;
  if (!(Number.isSafeInteger(requiredCount) && requiredCount >= 0)) {
    throw new Error('getGPUSpatialWeightsCapacityPlan requiredCount must be >= 0');
  }
  const nextCapacity =
    requiredCount <= capacity ? capacity : Math.ceil(requiredCount * (1 + headroom));
  const slotByteLength =
    GPU_SPATIAL_WEIGHTS_SLOT_BYTE_LENGTH +
    (options.distances ? GPU_SPATIAL_WEIGHTS_DISTANCE_BYTE_LENGTH : 0);
  const offsetsByteLength = (rowCount + 1) * Uint32Array.BYTES_PER_ELEMENT;
  return {
    capacity: nextCapacity,
    estimatedRequiredCount: requiredCount,
    limited: nextCapacity < requiredCount,
    offsetsByteLength,
    estimatedOutputBytes: nextCapacity * slotByteLength + offsetsByteLength,
    estimatedPeakTransientBytes: 0
  };
}
