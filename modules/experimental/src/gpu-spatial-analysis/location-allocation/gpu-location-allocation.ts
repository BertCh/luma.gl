// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {IS_NAN_BITS_WGSL} from '../../utils/wgsl-common';
import {
  createWGSLDispatchGate,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  getStorageBindingElementLimit,
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  createSparseTransposeNodes,
  createSparseValidationNodes,
  getSparseColumnBindings,
  getSparseRowBindings,
  SPARSE_ROW_COST_WGSL,
  type GPULocationAllocationSparseCosts,
  type SparseTranspose
} from './location-allocation-sparse';

export type {GPULocationAllocationSparseCosts};

const OPERATION = 'GPULocationAllocation';

/** Workgroup size of the single-workgroup choose and finalize reductions. */
const CHOOSE_WORKGROUP_SIZE = 256;

/** Location-allocation objective implemented by {@link GPULocationAllocation}. */
export type GPULocationAllocationOperation = 'p-median' | 'maximum-coverage' | 'set-covering';

/**
 * Packed status words written by {@link GPULocationAllocation}.
 *
 * Words 4 to 6 describe the swap improvement phase and are all zero when it is disabled
 * (`maximumSwapIterations: 0`) or not applicable (`set-covering`).
 */
export const GPU_LOCATION_ALLOCATION_STATUS = {
  selectedFacilityCount: 0,
  uncoveredDemandCount: 1,
  complete: 2,
  invalidCount: 3,
  /** Number of improving swaps applied. */
  swapIterationCount: 4,
  /** `1` when the swap phase stopped because no strictly improving swap exists (local optimum). */
  converged: 5,
  /** `1` when `maximumSwapIterations` swaps were applied, so local optimality is not verified. */
  iterationLimitReached: 6,
  length: 7
} as const;

/** Default `maximumSwapIterations` of the p-median and maximum-coverage operations. */
export const GPU_LOCATION_ALLOCATION_DEFAULT_MAXIMUM_SWAP_ITERATIONS = 100;

/** Largest `maximumSwapIterations`: every iteration adds five dispatches to the graph. */
export const GPU_LOCATION_ALLOCATION_MAXIMUM_SWAP_ITERATIONS = 1000;

/**
 * A swap must improve the objective by more than this fraction of the objective scale (the sum of
 * the finite weighted nearest-facility costs for p-median, the total demand weight for maximum
 * coverage) to be applied; for p-median a gain in reachable demand weight larger than this fraction
 * of the total weight takes precedence over cost. It keeps f32 summation noise from applying zero-gain swaps, so the
 * phase cannot cycle.
 */
export const GPU_LOCATION_ALLOCATION_SWAP_TOLERANCE = 1e-6;

/**
 * Largest `facilityCount` of {@link GPULocationAllocation} on `device`: the facility score table
 * holds two float32 values (8 bytes) per candidate in one storage binding, so 16777216 on a default
 * device (128 MiB bindings).
 *
 * @param device Device whose `maxStorageBufferBindingSize` applies.
 */
export function getGPULocationAllocationMaximumFacilityCount(device: Device): number {
  return getStorageBindingElementLimit(device, 8);
}

/**
 * Largest `costs` row count (`facilityCount * demand rows`) of {@link GPULocationAllocation} on
 * `device`: the cost matrix holds 4 bytes per entry in one storage binding, so 33554432 on a default
 * device (128 MiB bindings).
 *
 * @param device Device whose `maxStorageBufferBindingSize` applies.
 */
export function getGPULocationAllocationMaximumCostCount(device: Device): number {
  return getStorageBindingElementLimit(device, 4);
}

/**
 * Largest demand row count (`costs.length / facilityCount`) of {@link GPULocationAllocation} on
 * `device`: the swap phase packs three uint32 words (12 bytes) per demand row in one storage
 * binding, so 11184810 on a default device (128 MiB bindings). It applies to every operation.
 *
 * @param device Device whose `maxStorageBufferBindingSize` applies.
 */
export function getGPULocationAllocationMaximumDemandCount(device: Device): number {
  return getStorageBindingElementLimit(device, 12);
}

/**
 * Largest `sparseCosts` entry count of {@link GPULocationAllocation} on `device`: the
 * facility-major transpose holds two uint32 words (8 bytes) per entry in one storage binding, so
 * 16777216 on a default device (128 MiB bindings).
 *
 * @param device Device whose `maxStorageBufferBindingSize` applies.
 */
export function getGPULocationAllocationMaximumSparseEntryCount(device: Device): number {
  return getStorageBindingElementLimit(device, 8);
}

/**
 * Largest number of (selected slot, candidate) pairs one swap iteration scores on `device`: two
 * float32 scores (8 bytes) per pair in one storage binding, so 2^24 on a default device.
 */
function getMaximumSwapPairCount(device: Device): number {
  return getStorageBindingElementLimit(device, 8);
}

/** Caller-owned result views for {@link GPULocationAllocation}. */
export type GPULocationAllocationOutput = {
  /** One flag per candidate facility. */
  selectedFacilities: GraphDataView<'uint32'>;
  /** Selected facility per demand row, or `0xffffffff` when no selected facility is reachable. */
  assignments: GraphDataView<'uint32'>;
  /** Optional minimum selected-facility cost per demand row. */
  assignedCosts?: GraphDataView<'float32'>;
  /** One objective value: weighted cost for p-median, covered weight for maximum coverage, selected count for set covering. */
  objective: GraphDataView<'float32'>;
  /** Seven words indexed by {@link GPU_LOCATION_ALLOCATION_STATUS}. */
  status: GraphDataView<'uint32'>;
};

/** Properties for {@link GPULocationAllocation}. */
export type GPULocationAllocationProps = {
  id?: string;
  /**
   * Row-major demand-by-facility cost matrix. Positive infinity means unreachable. At most
   * {@link getGPULocationAllocationMaximumCostCount} entries and
   * {@link getGPULocationAllocationMaximumDemandCount} demand rows. Exactly one of `costs` and
   * `sparseCosts` is required.
   */
  costs?: GraphDataView<'float32'>;
  /**
   * Sparse alternative to `costs`: CSR rows of (facility, cost) entries, where a pair that is not
   * listed is unreachable (`+Infinity`). It removes the dense `demand rows x facilityCount` limit;
   * the work and memory then scale with the entry count, at most
   * {@link getGPULocationAllocationMaximumSparseEntryCount} (at least one entry), and the demand
   * row count is `offsets.length - 1`, at most {@link getGPULocationAllocationMaximumDemandCount}.
   * Rows must list strictly ascending facilities below `facilityCount`; a violation counts in
   * `status` word 3 (`invalidCount`). Supported by every operation, with the same results as the
   * equivalent dense matrix for a finite `coverageDistance` (up to float32 summation order in
   * near-ties).
   */
  sparseCosts?: GPULocationAllocationSparseCosts;
  /**
   * Number of candidate facility columns in `costs` (or the facility index range of
   * `sparseCosts`). At most {@link getGPULocationAllocationMaximumFacilityCount}.
   */
  facilityCount: number;
  /** Optional non-negative demand weight per matrix row. Defaults to one. */
  demandWeights?: GraphDataView<'float32'>;
  /** Greedy objective. */
  operation: GPULocationAllocationOperation;
  /** Facility budget for p-median/MCLP and hard search cap for LSCP. */
  maximumFacilityCount: number;
  /**
   * Maximum number of Teitz-Bart vertex-substitution (swap) iterations run after the greedy
   * construction, an integer in `[0, 1000]`. Compile-time. Default 100 for `p-median` and
   * `maximum-coverage`; `0` disables the phase (pure greedy); `set-covering` has no swap phase and
   * rejects a positive value. Each iteration scores every (selected facility, unselected candidate)
   * swap in parallel with the Whitaker / Resende-Werneck gain decomposition (`O(facilityCount *
   * demandCount)` work, or the entry count with `sparseCosts`), applies the
   * single best strictly improving swap (ties go to the lowest selected facility, then the lowest
   * candidate), and stops when none improves. Requires
   * `maximumFacilityCount * facilityCount <= 2^24`.
   */
  maximumSwapIterations?: number;
  /** One per-frame distance threshold, required by maximum coverage and set covering. */
  coverageDistance?: GraphDataView<'float32'>;
  output: GPULocationAllocationOutput;
};

/** WGSL that flags an invalid demand weight or coverage distance in `status` word 3. */
function getWeightAndCoverageValidationWGSL(hasWeights: boolean, hasCoverage: boolean): string {
  return `${
    hasWeights
      ? `if (index < DEMAND_COUNT && isInvalid(demandWeights[demandWeightsOffset + index])) {
    atomicAdd(&status[statusOffset + 3u], 1u);
  }`
      : ''
  }
  ${
    hasCoverage
      ? `if (index == 0u && isInvalid(coverageDistance[coverageDistanceOffset])) {
    atomicAdd(&status[statusOffset + 3u], 1u);
  }`
      : ''
  }`;
}

/**
 * Deterministic bounded location-allocation over a caller-owned cost matrix.
 *
 * `p-median` greedily adds the facility with the largest reduction in weighted nearest-facility
 * cost. `maximum-coverage` greedily adds the facility covering the greatest remaining demand
 * weight within `coverageDistance`. `set-covering` uses the same deterministic set-cover step until
 * every reachable demand row is covered or `maximumFacilityCount` is reached. Ties select the
 * lowest facility row. All assignments and stopping evidence remain GPU-resident.
 *
 * Unreachable demand (a `+Infinity` cost to every selected facility) is never summed into the
 * p-median objective: the objective is the weighted cost of the reachable rows, unreachable rows
 * are counted in the `uncoveredDemandCount` status word (infeasibility is reported there, not in
 * `invalidCount`, which marks invalid input and stops the run), their assignment is `0xffffffff`
 * and their `assignedCosts` entry is `+Infinity`. Greedy and swap steps compare candidates
 * lexicographically: first the demand weight left unreachable, then the reachable cost, so one
 * unreachable row never swamps the float32 cost differences of the others.
 *
 * After the greedy construction, `p-median` and `maximum-coverage` run a Teitz-Bart
 * vertex-substitution improvement phase (see `maximumSwapIterations`): the best strictly improving
 * swap of a selected facility for an unselected candidate is applied until none exists. The result
 * is a swap-local optimum, not the integer-programming optimum. Greedy maximum coverage alone has
 * the `(1 - 1/e)` guarantee relative to the optimum (Nemhauser, Wolsey and Fisher); greedy
 * p-median has no constant-factor guarantee, which is why the swap phase is on by default. The swap
 * phase never makes the objective worse, so the greedy guarantee still holds. Swap progress is
 * reported in the `swapIterationCount`, `converged` and `iterationLimitReached` status words.
 *
 * Swaps are scored with the Whitaker / Resende-Werneck gain decomposition, one pass per candidate. The
 * costs are a dense matrix (`costs`) or sparse CSR rows (`sparseCosts`, unlisted pairs unreachable).
 *
 * Run them after `GPUNetworkCostMatrix` for network location-allocation or provide any dense cost matrix.
 */
export class GPULocationAllocation implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPULocationAllocationProps;
  readonly demandCount: number;
  /** Resolved `maximumSwapIterations` (0 when the phase is disabled or not applicable). */
  readonly maximumSwapIterations: number;

  constructor(props: GPULocationAllocationProps) {
    this.id = props.id ?? 'location-allocation';
    this.props = props;
    const {id} = this;
    const {output} = props;
    if (props.costs && props.sparseCosts) {
      throw new Error(`${id} costs and sparseCosts are mutually exclusive`);
    }
    if (!props.costs && !props.sparseCosts) {
      throw new Error(`${id} requires costs or sparseCosts`);
    }
    validateCapacity(id, 'facilityCount', props.facilityCount);
    if (props.costs) {
      validatePackedView(props.costs, ['float32'], `${id} costs`);
      if (
        props.costs.length < props.facilityCount ||
        props.costs.length % props.facilityCount !== 0
      ) {
        throw new Error(`${id} costs length must be a positive multiple of facilityCount`);
      }
      this.demandCount = props.costs.length / props.facilityCount;
    } else {
      const sparse = props.sparseCosts!;
      validatePackedView(sparse.offsets, ['uint32'], `${id} sparseCosts.offsets`);
      validatePackedView(sparse.facilities, ['uint32'], `${id} sparseCosts.facilities`);
      validatePackedView(sparse.costs, ['float32'], `${id} sparseCosts.costs`);
      if (sparse.offsets.length < 2) {
        throw new Error(`${id} sparseCosts.offsets must hold demand rows + 1 offsets`);
      }
      if (sparse.facilities.length < 1 || sparse.facilities.length !== sparse.costs.length) {
        throw new Error(
          `${id} sparseCosts.facilities and costs must be equally long and not empty`
        );
      }
      this.demandCount = sparse.offsets.length - 1;
    }
    validateCapacity(
      id,
      'maximumFacilityCount',
      props.maximumFacilityCount,
      1,
      props.facilityCount
    );
    const requestedSwaps = props.maximumSwapIterations;
    if (requestedSwaps !== undefined) {
      validateCapacity(
        id,
        'maximumSwapIterations',
        requestedSwaps,
        0,
        GPU_LOCATION_ALLOCATION_MAXIMUM_SWAP_ITERATIONS
      );
    }
    if (props.operation === 'set-covering') {
      if (requestedSwaps) {
        throw new Error(`${id} set-covering has no swap phase; omit maximumSwapIterations`);
      }
      this.maximumSwapIterations = 0;
    } else {
      this.maximumSwapIterations =
        requestedSwaps ?? GPU_LOCATION_ALLOCATION_DEFAULT_MAXIMUM_SWAP_ITERATIONS;
    }
    if (props.demandWeights) {
      validatePackedView(props.demandWeights, ['float32'], `${id} demandWeights`);
      if (props.demandWeights.length !== this.demandCount) {
        throw new Error(`${id} demandWeights length must equal the demand row count`);
      }
    }
    if (props.operation !== 'p-median') {
      if (!props.coverageDistance) {
        throw new Error(`${id} coverageDistance is required for ${props.operation}`);
      }
      validatePackedView(props.coverageDistance, ['float32'], `${id} coverageDistance`);
      if (props.coverageDistance.length < 1) {
        throw new Error(`${id} coverageDistance must hold one value`);
      }
    } else if (props.coverageDistance) {
      validatePackedView(props.coverageDistance, ['float32'], `${id} coverageDistance`);
    }
    validatePackedView(output.selectedFacilities, ['uint32'], `${id} output.selectedFacilities`);
    validatePackedView(output.assignments, ['uint32'], `${id} output.assignments`);
    validatePackedView(output.status, ['uint32'], `${id} output.status`);
    validatePackedView(output.objective, ['float32'], `${id} output.objective`);
    if (output.selectedFacilities.length < props.facilityCount) {
      throw new Error(`${id} output.selectedFacilities must hold one flag per facility`);
    }
    if (output.assignments.length < this.demandCount) {
      throw new Error(`${id} output.assignments must hold one row per demand`);
    }
    if (output.objective.length < 1) {
      throw new Error(`${id} output.objective must hold one value`);
    }
    if (output.status.length < GPU_LOCATION_ALLOCATION_STATUS.length) {
      throw new Error(
        `${id} output.status must hold ${GPU_LOCATION_ALLOCATION_STATUS.length} words`
      );
    }
    if (output.assignedCosts) {
      validatePackedView(output.assignedCosts, ['float32'], `${id} output.assignedCosts`);
      if (output.assignedCosts.length < this.demandCount) {
        throw new Error(`${id} output.assignedCosts must hold one row per demand`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.selectedFacilities,
        output.assignments,
        output.assignedCosts,
        output.objective,
        output.status
      ],
      [...this.getCostViews(), props.demandWeights, props.coverageDistance]
    );
  }

  /** The cost views read by the kernels: the dense matrix, or the three CSR views. */
  private getCostViews(): GraphDataView[] {
    const {costs, sparseCosts} = this.props;
    return costs ? [costs] : [sparseCosts!.offsets, sparseCosts!.facilities, sparseCosts!.costs];
  }

  /**
   * Teitz-Bart vertex substitution scored with the Whitaker / Resende-Werneck gain decomposition.
   *
   * For a swap that removes selected facility `r` and adds candidate `i`, every demand row `j`
   * keeps its nearest selected facility unless `j` is served by `r` (p-median) or covered only by
   * `r` (maximum coverage); the improvement is therefore `gain(i) - loss(r) + extra(i, r)`:
   * `gain(i)` sums over all rows what `i` would win if nothing were removed, `loss(r)` is what the
   * rows served by `r` lose when it goes (their second-nearest facility takes over) and `extra(i,
   * r)` corrects the rows served by `r` that `i` also reaches. `gain` and `extra` only involve the
   * entries `(j, i)` that beat the row's nearest (second-nearest) cost, so one pass over candidate
   * `i`'s costs scores all of its `maximumFacilityCount` pairs: `O(facilityCount * demandCount)`
   * work per iteration instead of `O(maximumFacilityCount * facilityCount * demandCount)`.
   *
   * Each iteration: the ascending list of selected facilities and each facility's slot; per demand
   * row the nearest and second-nearest selected cost and the nearest slot (or the covering count
   * and the sole covering slot); one workgroup per slot summing `loss(r)` (plus one workgroup for
   * the objective scale); one thread per candidate scoring its pairs, each in a fixed ascending
   * demand order and without atomics; a single-workgroup argmax (ties to the lowest pair index,
   * i.e. lowest selected facility then lowest candidate); and the swap itself. Once no swap
   * improves, the apply kernel zeroes the indirect dispatch of the five other kernels, so a
   * converged run costs one one-thread dispatch per remaining iteration (rule 13).
   */
  private getSwapNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    context: {
      declarations: string;
      swapState: GraphDataView<'uint32'>;
      weightBinding: WGSLKernelBinding[];
      coverageBinding: WGSLKernelBinding[];
      demandWeightExpression: string;
      transpose?: SparseTranspose;
    }
  ): GPUCommandNode<Parameters>[] {
    const {id, props, demandCount} = this;
    const {output, facilityCount} = props;
    const sparse = props.sparseCosts;
    const {declarations, swapState, weightBinding, coverageBinding, demandWeightExpression} =
      context;
    const slots = props.maximumFacilityCount;
    const pairCount = slots * facilityCount;
    const coverageMode = props.operation === 'maximum-coverage';
    const view = <Format extends 'uint32' | 'float32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-swap-${name}`, format, length);
    // Per demand, packed as three u32 words (a device allows only eight storage buffers per kernel).
    // p-median: nearest cost bits, second-nearest cost bits, slot of the nearest facility.
    // maximum coverage: number of selected facilities within the coverage distance, unused, and the
    // slot of the only covering facility (NONE unless the count is one).
    const demandData = view('demand-data', 'uint32', demandCount * 3);
    const selectedList = view('selected-list', 'uint32', slots);
    // Slot (rank among the selected facilities) per facility, NONE for an unselected candidate.
    const slotOfFacility = view('slot-of-facility', 'uint32', facilityCount);
    // Three values per slot: the loss(r) terms (reach weight, cost; coverage uses the first) and
    // 1 when the slot holds a facility; then the objective scale and the total demand weight.
    const swapTotals = view('totals', 'float32', 3 * slots + 2);
    // Two values per pair: first the extra(i, r) terms, then the final (primary, secondary)
    // improvement, -1 when the pair is not a swap.
    const pairScores = view('pair-scores', 'float32', 2 * pairCount);
    const threshold = coverageMode ? 'coverageDistance[coverageDistanceOffset]' : '0.0';
    // Sparse scoring cannot visit the rows a candidate does not list, so loss(r) is summed per slot
    // (one workgroup each, plus one for the totals). Dense scoring visits every row and adds each
    // row's net term directly, so one workgroup of totals is enough.
    const removalGroupCount = sparse ? slots + 1 : 1;
    // Slots: list, prepare, removal (totals), score, choose.
    const gate = createWGSLDispatchGate(graph, {
      id: `${id}-swap-dispatch`,
      operation: OPERATION,
      shapes: [
        [1],
        [demandCount],
        [removalGroupCount * CHOOSE_WORKGROUP_SIZE, CHOOSE_WORKGROUP_SIZE],
        [facilityCount],
        [CHOOSE_WORKGROUP_SIZE, CHOOSE_WORKGROUP_SIZE]
      ]
    });
    const gateBinding = gate.binding;
    // Rows of one demand row that are considered for the nearest / second-nearest selection.
    const considerSlot = `if (cost < nearestCost) {
      secondCost = nearestCost;
      nearestCost = cost;
      nearestSlot = slot;
    } else if (cost < secondCost) {
      secondCost = cost;
    }
    if (cost <= threshold) {
      coveringCount++;
      coveringSlot = slot;
    }`;
    const denseSelectedLoop = `for (var slot = 0u; slot < SWAP_SLOTS; slot++) {
    let facility = selectedList[selectedListOffset + slot];
    if (facility == NONE) { break; }
    let cost = min(${
      sparse
        ? 'getSparseRowCost(demand, facility)'
        : 'costs[costsOffset + demand * FACILITY_COUNT + facility]'
    }, LARGE_COST);
    ${considerSlot}
  }`;
    // Sparse rows are scanned when they are no longer than the slot list, otherwise the slots are
    // looked up in the row; both visit the selected facilities in ascending order.
    const prepareLoop = sparse
      ? `let rowStart = rowOffsets[rowOffsetsOffset + demand];
  let rowEnd = rowOffsets[rowOffsetsOffset + demand + 1u];
  if (rowEnd - rowStart <= SWAP_SLOTS) {
    for (var entry = rowStart; entry < rowEnd; entry++) {
      let slot = slotOfFacility[slotOfFacilityOffset + rowFacilities[rowFacilitiesOffset + entry]];
      if (slot == NONE) { continue; }
      let cost = min(rowCosts[rowCostsOffset + entry], LARGE_COST);
      ${considerSlot}
    }
  } else {
    ${denseSelectedLoop}
  }`
      : denseSelectedLoop;
    const prepareBindings: WGSLKernelBinding[] = [
      ...(sparse
        ? [
            ...getSparseRowBindings(sparse),
            {name: 'slotOfFacility', view: slotOfFacility, type: 'u32', access: 'read'} as const
          ]
        : [{name: 'costs', view: props.costs!, type: 'f32', access: 'read'} as const]),
      {name: 'selectedList', view: selectedList, type: 'u32', access: 'read'},
      {name: 'demandData', view: demandData, type: 'u32', access: 'read_write'},
      ...coverageBinding
    ];
    // One loop over the entries (j, c) of candidate `i`: all demand rows (dense), or the candidate's
    // column of the transpose (sparse, ascending demand rows).
    const entryLoop = (body: string) =>
      sparse
        ? `let columnEnd = columnOffsets[columnOffsetsOffset + candidate + 1u];
  for (var entry = columnOffsets[columnOffsetsOffset + candidate]; entry < columnEnd; entry++) {
    let demand = columnEntries[columnEntriesOffset + 2u * entry];
    let c = min(bitcast<f32>(columnEntries[columnEntriesOffset + 2u * entry + 1u]), LARGE_COST);
    ${body}
  }`
        : `for (var demand = 0u; demand < DEMAND_COUNT; demand++) {
    let c = min(costs[costsOffset + demand * FACILITY_COUNT + candidate], LARGE_COST);
    ${body}
  }`;
    const scoreBindings: WGSLKernelBinding[] = [
      ...(sparse
        ? getSparseColumnBindings(context.transpose!)
        : [{name: 'costs', view: props.costs!, type: 'f32', access: 'read'} as const]),
      {name: 'slotOfFacility', view: slotOfFacility, type: 'u32', access: 'read'},
      {name: 'demandData', view: demandData, type: 'u32', access: 'read'},
      {name: 'swapTotals', view: swapTotals, type: 'f32', access: 'read'},
      ...(sparse
        ? []
        : [{name: 'selectedList', view: selectedList, type: 'u32', access: 'read'} as const]),
      {name: 'pairScores', view: pairScores, type: 'f32', access: 'read_write'},
      ...weightBinding,
      ...coverageBinding
    ];
    const sparseScoreBody = coverageMode
      ? entryLoop(`let weight = ${demandWeightExpression};
    if (c <= threshold) {
      let count = demandData[demandDataOffset + demand * 3u];
      let sole = demandData[demandDataOffset + demand * 3u + 2u];
      if (count == 0u) {
        gainCover += weight;
      } else if (sole != NONE) {
        pairScores[pairScoresOffset + 2u * (sole * FACILITY_COUNT + candidate)] += weight;
      }
    }`)
      : entryLoop(`let weight = ${demandWeightExpression};
    let nearest = bitcast<f32>(demandData[demandDataOffset + demand * 3u]);
    let second = bitcast<f32>(demandData[demandDataOffset + demand * 3u + 1u]);
    let slot = demandData[demandDataOffset + demand * 3u + 2u];
    if (c < nearest) {
      // gain(i): the candidate beats the nearest selected facility (or reaches an unreached row).
      if (isReachable(nearest)) {
        gainCost += weight * (nearest - c);
      } else {
        gainReach += weight;
        gainCost -= weight * c;
      }
    }
    if (slot != NONE && c < second) {
      // extra(i, r): the row loses its nearest facility r but the candidate beats the second.
      var extraReach = 0.0;
      var extraCost = weight * (second - max(c, nearest));
      if (!isReachable(second)) {
        extraReach = weight;
        extraCost = weight * (min(nearest, c) - c - nearest);
      }
      let pair = pairScoresOffset + 2u * (slot * FACILITY_COUNT + candidate);
      pairScores[pair] += extraReach;
      pairScores[pair + 1u] += extraCost;
    }`);
    // Dense: every row is visited, so a row served by slot r adds its net term directly,
    // `ct(min(d1, c)) - ct(min(d2, c))` (cost) and the reach weight lost when neither d2 nor c
    // reaches it. Every term is a modest difference; no loss(r) and extra(i, r) cancel.
    const denseScoreBody = coverageMode
      ? entryLoop(`let weight = ${demandWeightExpression};
    let count = demandData[demandDataOffset + demand * 3u];
    let sole = demandData[demandDataOffset + demand * 3u + 2u];
    if (count == 0u) {
      if (c <= threshold) { gainCover += weight; }
    } else if (sole != NONE && c > threshold) {
      // The only covering facility leaves and the candidate does not cover the row.
      pairScores[pairScoresOffset + 2u * (sole * FACILITY_COUNT + candidate)] -= weight;
    }`)
      : entryLoop(`let weight = ${demandWeightExpression};
    let nearest = bitcast<f32>(demandData[demandDataOffset + demand * 3u]);
    let second = bitcast<f32>(demandData[demandDataOffset + demand * 3u + 1u]);
    let slot = demandData[demandDataOffset + demand * 3u + 2u];
    if (c < nearest) {
      if (isReachable(nearest)) {
        gainCost += weight * (nearest - c);
      } else {
        gainReach += weight;
        gainCost -= weight * c;
      }
    } else if (slot != NONE) {
      // The row loses its nearest facility and falls back to the cheaper of the second and the
      // candidate (c >= nearest here, so the candidate does not beat the nearest).
      let kept = min(second, c);
      if (kept > nearest) {
        let pair = pairScoresOffset + 2u * (slot * FACILITY_COUNT + candidate);
        if (isReachable(kept)) {
          pairScores[pair + 1u] += weight * (nearest - kept);
        } else {
          pairScores[pair] -= weight;
          pairScores[pair + 1u] += weight * nearest;
        }
      }
    }`);
    const scoreBody = sparse ? sparseScoreBody : denseScoreBody;
    const lossReach = sparse ? 'swapTotals[swapTotalsOffset + 3u * slot]' : '0.0';
    const slotIsEmpty = sparse
      ? 'swapTotals[swapTotalsOffset + 3u * slot + 2u] < 0.5'
      : 'selectedList[selectedListOffset + slot] == NONE';
    const lossCost = sparse ? 'swapTotals[swapTotalsOffset + 3u * slot + 1u]' : '0.0';
    const scoreFinalize = coverageMode
      ? `let improvement = gainCover + ${lossReach} + pairScores[pairScoresOffset + pair];
    if (improvement > SWAP_TOLERANCE * scale) {
      pairScores[pairScoresOffset + pair] = improvement;
      pairScores[pairScoresOffset + pair + 1u] = 0.0;
    } else {
      pairScores[pairScoresOffset + pair] = -1.0;
      pairScores[pairScoresOffset + pair + 1u] = -1.0;
    }`
      : `let reachGain = gainReach + ${lossReach} + pairScores[pairScoresOffset + pair];
    let improvement = gainCost + ${lossCost} + pairScores[pairScoresOffset + pair + 1u];
    var primary = -1.0;
    var secondary = -1.0;
    if (reachGain > SWAP_TOLERANCE * totalWeight) {
      primary = reachGain;
      secondary = improvement;
    } else if (abs(reachGain) <= SWAP_TOLERANCE * totalWeight && improvement > SWAP_TOLERANCE * scale) {
      primary = 0.0;
      secondary = improvement;
    }
    pairScores[pairScoresOffset + pair] = primary;
    pairScores[pairScoresOffset + pair + 1u] = secondary;`;
    const removalBody = coverageMode
      ? `let weight = ${demandWeightExpression};
    if (removalTarget == SWAP_SLOTS) {
      sumA += weight;
    } else if (demandData[demandDataOffset + demand * 3u + 2u] == removalTarget) {
      sumA -= weight;
    }`
      : `let weight = ${demandWeightExpression};
    let nearest = bitcast<f32>(demandData[demandDataOffset + demand * 3u]);
    let second = bitcast<f32>(demandData[demandDataOffset + demand * 3u + 1u]);
    if (removalTarget == SWAP_SLOTS) {
      sumA += weight;
      if (isReachable(nearest)) { sumB += weight * nearest; }
    } else if (demandData[demandDataOffset + demand * 3u + 2u] == removalTarget) {
      // loss(r): the row falls back to its second-nearest facility, or becomes unreachable.
      if (isReachable(second)) {
        sumB += weight * (nearest - second);
      } else {
        sumA -= weight;
        sumB += weight * nearest;
      }
    }`;
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode(graph, {
        id: `${id}-swap-init`,
        operation: OPERATION,
        variant: 'swap-init',
        invocationCount: 1,
        bindings: [
          {name: 'swapState', view: swapState, type: 'u32', access: 'read_write'},
          gateBinding
        ],
        declarations,
        body: `swapState[swapStateOffset] = 0u;
  swapState[swapStateOffset + 1u] = 0u;
  swapState[swapStateOffset + 2u] = NONE;
  ${gate.armWGSL}`
      })
    ];
    for (let iteration = 0; iteration < this.maximumSwapIterations; iteration++) {
      const iterationId = `${id}-swap-${iteration}`;
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${iterationId}-list`,
          operation: OPERATION,
          variant: 'swap-list',
          invocationCount: 1,
          ...gate.getGate(0),
          bindings: [
            {name: 'selected', view: output.selectedFacilities, type: 'u32', access: 'read'},
            {name: 'status', view: output.status, type: 'u32', access: 'read'},
            {name: 'swapState', view: swapState, type: 'u32', access: 'read'},
            {name: 'selectedList', view: selectedList, type: 'u32', access: 'read_write'},
            {name: 'slotOfFacility', view: slotOfFacility, type: 'u32', access: 'read_write'}
          ],
          declarations,
          // Stopped or invalid: leave every slot empty so the later kernels do no work.
          body: `for (var slot = 0u; slot < SWAP_SLOTS; slot++) {
  selectedList[selectedListOffset + slot] = NONE;
}
let isActive = swapState[swapStateOffset] == 0u && status[statusOffset + 3u] == 0u;
var slot = 0u;
for (var facility = 0u; facility < FACILITY_COUNT; facility++) {
  var facilitySlot = NONE;
  if (isActive && selected[selectedOffset + facility] != 0u && slot < SWAP_SLOTS) {
    selectedList[selectedListOffset + slot] = facility;
    facilitySlot = slot;
    slot++;
  }
  slotOfFacility[slotOfFacilityOffset + facility] = facilitySlot;
}`
        }),
        createWGSLKernelNode(graph, {
          id: `${iterationId}-prepare`,
          operation: OPERATION,
          variant: `${props.operation}-swap-prepare${sparse ? '-sparse' : ''}`,
          invocationCount: demandCount,
          ...gate.getGate(1),
          bindings: prepareBindings,
          declarations: sparse ? `${declarations}\n${SPARSE_ROW_COST_WGSL}` : declarations,
          body: `let demand = index;
  var nearestCost = LARGE_COST;
  var secondCost = LARGE_COST;
  var nearestSlot = NONE;
  var coveringCount = 0u;
  var coveringSlot = NONE;
  let threshold = ${threshold};
  ${prepareLoop}
  demandData[demandDataOffset + demand * 3u] = ${coverageMode ? 'coveringCount' : 'bitcast<u32>(nearestCost)'};
  demandData[demandDataOffset + demand * 3u + 1u] = ${coverageMode ? '0u' : 'bitcast<u32>(secondCost)'};
  demandData[demandDataOffset + demand * 3u + 2u] = ${coverageMode ? 'select(NONE, coveringSlot, coveringCount == 1u)' : 'nearestSlot'};`
        }),
        createWGSLKernelNode(graph, {
          id: `${iterationId}-${sparse ? 'removal' : 'totals'}`,
          operation: OPERATION,
          variant: `${props.operation}-swap-${sparse ? 'removal' : 'totals'}`,
          invocationCount: removalGroupCount * CHOOSE_WORKGROUP_SIZE,
          workgroupSize: CHOOSE_WORKGROUP_SIZE,
          guardIndex: false,
          ...gate.getGate(2),
          bindings: [
            {name: 'demandData', view: demandData, type: 'u32', access: 'read'},
            {name: 'selectedList', view: selectedList, type: 'u32', access: 'read'},
            {name: 'swapTotals', view: swapTotals, type: 'f32', access: 'read_write'},
            ...weightBinding
          ],
          declarations: `${declarations}
const SWAP_WORKGROUP_SIZE: u32 = ${CHOOSE_WORKGROUP_SIZE}u;
var<workgroup> sharedA: array<f32, ${CHOOSE_WORKGROUP_SIZE}>;
var<workgroup> sharedB: array<f32, ${CHOOSE_WORKGROUP_SIZE}>;`,
          // Workgroup r (< SWAP_SLOTS) sums loss(r) over the rows served by slot r; the last
          // workgroup sums the total demand weight (A) and the objective scale (B). Strided lane
          // partials and a fixed tree make the sums independent of scheduling. No early return:
          // every invocation must reach the barriers.
          body: `let removalTarget = ${sparse ? 'workgroupIndex' : 'SWAP_SLOTS'};
  var sumA = 0.0;
  var sumB = 0.0;
  for (var demand = localInvocationIndex; demand < DEMAND_COUNT; demand += SWAP_WORKGROUP_SIZE) {
    ${removalBody}
  }
  sharedA[localInvocationIndex] = sumA;
  sharedB[localInvocationIndex] = sumB;
  workgroupBarrier();
  for (var stride = SWAP_WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride) {
      sharedA[localInvocationIndex] += sharedA[localInvocationIndex + stride];
      sharedB[localInvocationIndex] += sharedB[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    if (removalTarget == SWAP_SLOTS) {
      swapTotals[swapTotalsOffset + 3u * SWAP_SLOTS] = ${coverageMode ? 'sharedA[0]' : 'sharedB[0]'};
      swapTotals[swapTotalsOffset + 3u * SWAP_SLOTS + 1u] = sharedA[0];
    } else {
      swapTotals[swapTotalsOffset + 3u * removalTarget] = sharedA[0];
      swapTotals[swapTotalsOffset + 3u * removalTarget + 1u] = sharedB[0];
      swapTotals[swapTotalsOffset + 3u * removalTarget + 2u] =
        select(0.0, 1.0, selectedList[selectedListOffset + removalTarget] != NONE);
    }
  }`
        }),
        createWGSLKernelNode(graph, {
          id: `${iterationId}-score`,
          operation: OPERATION,
          variant: `${props.operation}-swap-score${sparse ? '-sparse' : ''}`,
          invocationCount: facilityCount,
          ...gate.getGate(3),
          bindings: scoreBindings,
          declarations,
          // One thread scores every pair of one candidate. Improvement is positive when the swap
          // helps: cost removed for p-median (after the reachable demand weight gained), covered
          // weight gained for maximum coverage. A pair is only kept when it beats the noise
          // tolerance. Only this thread touches the candidate's pair entries, so no atomics.
          body: `let candidate = index;
  let isCandidate = slotOfFacility[slotOfFacilityOffset + candidate] == NONE;
  let threshold = ${threshold};
  for (var slot = 0u; slot < SWAP_SLOTS; slot++) {
    let pair = pairScoresOffset + 2u * (slot * FACILITY_COUNT + candidate);
    pairScores[pair] = select(-1.0, 0.0, isCandidate);
    pairScores[pair + 1u] = select(-1.0, 0.0, isCandidate);
  }
  if (!isCandidate) {
    return;
  }
  var gainReach = 0.0;
  var gainCost = 0.0;
  var gainCover = 0.0;
  ${scoreBody}
  let scale = swapTotals[swapTotalsOffset + 3u * SWAP_SLOTS];
  let totalWeight = swapTotals[swapTotalsOffset + 3u * SWAP_SLOTS + 1u];
  for (var slot = 0u; slot < SWAP_SLOTS; slot++) {
    let pair = 2u * (slot * FACILITY_COUNT + candidate);
    if (${slotIsEmpty}) {
      pairScores[pairScoresOffset + pair] = -1.0;
      pairScores[pairScoresOffset + pair + 1u] = -1.0;
      continue;
    }
    ${scoreFinalize}
  }`
        }),
        createWGSLKernelNode(graph, {
          id: `${iterationId}-choose`,
          operation: OPERATION,
          variant: 'swap-choose',
          invocationCount: CHOOSE_WORKGROUP_SIZE,
          workgroupSize: CHOOSE_WORKGROUP_SIZE,
          guardIndex: false,
          ...gate.getGate(4),
          bindings: [
            {name: 'pairScores', view: pairScores, type: 'f32', access: 'read'},
            {name: 'swapState', view: swapState, type: 'u32', access: 'read_write'}
          ],
          declarations: `${declarations}
const PAIR_COUNT: u32 = ${pairCount}u;
const SWAP_WORKGROUP_SIZE: u32 = ${CHOOSE_WORKGROUP_SIZE}u;
var<workgroup> sharedScore: array<f32, ${CHOOSE_WORKGROUP_SIZE}>;
var<workgroup> sharedSecondary: array<f32, ${CHOOSE_WORKGROUP_SIZE}>;
var<workgroup> sharedPair: array<u32, ${CHOOSE_WORKGROUP_SIZE}>;
fn isBetterPair(
  score: f32, secondary: f32, pair: u32, bestScore: f32, bestSecondary: f32, bestPair: u32
) -> bool {
  if (score != bestScore) { return score > bestScore; }
  if (secondary != bestSecondary) { return secondary > bestSecondary; }
  return pair < bestPair;
}`,
          // Total order ((primary, secondary) descending, pair ascending), so the result is
          // independent of scheduling. Qualifying pairs have a non-negative primary; -1 marks
          // "no swap". No early return: every invocation must reach the barriers.
          body: `var bestPair = NONE;
var bestScore = -1.0;
var bestSecondary = -1.0;
if (swapState[swapStateOffset] == 0u) {
  for (var pair = localInvocationIndex; pair < PAIR_COUNT; pair += SWAP_WORKGROUP_SIZE) {
    let score = pairScores[pairScoresOffset + 2u * pair];
    let secondary = pairScores[pairScoresOffset + 2u * pair + 1u];
    if (isBetterPair(score, secondary, pair, bestScore, bestSecondary, bestPair)) {
      bestPair = pair;
      bestScore = score;
      bestSecondary = secondary;
    }
  }
}
sharedScore[localInvocationIndex] = bestScore;
sharedSecondary[localInvocationIndex] = bestSecondary;
sharedPair[localInvocationIndex] = bestPair;
workgroupBarrier();
for (var stride = SWAP_WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
  if (localInvocationIndex < stride &&
      isBetterPair(
        sharedScore[localInvocationIndex + stride],
        sharedSecondary[localInvocationIndex + stride],
        sharedPair[localInvocationIndex + stride],
        sharedScore[localInvocationIndex],
        sharedSecondary[localInvocationIndex],
        sharedPair[localInvocationIndex])) {
    sharedScore[localInvocationIndex] = sharedScore[localInvocationIndex + stride];
    sharedSecondary[localInvocationIndex] = sharedSecondary[localInvocationIndex + stride];
    sharedPair[localInvocationIndex] = sharedPair[localInvocationIndex + stride];
  }
  workgroupBarrier();
}
if (localInvocationIndex == 0u) {
  // A qualifying pair has a non-negative primary score; -1 marks "no swap".
  swapState[swapStateOffset + 2u] = select(NONE, sharedPair[0], sharedScore[0] >= 0.0);
}`
        }),
        createWGSLKernelNode(graph, {
          id: `${iterationId}-apply`,
          operation: OPERATION,
          variant: 'swap-apply',
          invocationCount: 1,
          bindings: [
            {name: 'selected', view: output.selectedFacilities, type: 'u32', access: 'read_write'},
            {name: 'selectedList', view: selectedList, type: 'u32', access: 'read'},
            {name: 'swapState', view: swapState, type: 'u32', access: 'read_write'},
            gateBinding
          ],
          declarations,
          body: `if (swapState[swapStateOffset] != 0u) {
  return;
}
let pair = swapState[swapStateOffset + 2u];
if (pair == NONE) {
  swapState[swapStateOffset] = 1u;
  ${gate.disarmWGSL}
  return;
}
selected[selectedOffset + selectedList[selectedListOffset + pair / FACILITY_COUNT]] = 0u;
selected[selectedOffset + pair % FACILITY_COUNT] = 1u;
swapState[swapStateOffset + 1u] = swapState[swapStateOffset + 1u] + 1u;`
        })
      );
    }
    return nodes;
  }

  /**
   * Sparse input setup: validation of the CSR structure and of the weights and coverage distance,
   * then the facility-major transpose the facility-side loops (greedy scores, swap candidates) read.
   */
  private getSparseSetupNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    context: {
      declarations: string;
      weightBinding: WGSLKernelBinding[];
      coverageBinding: WGSLKernelBinding[];
      needsCoverage: boolean;
    }
  ): {nodes: GPUCommandNode<Parameters>[]; transpose: SparseTranspose} {
    const {id, props, demandCount} = this;
    const sparse = props.sparseCosts!;
    const {declarations, weightBinding, coverageBinding, needsCoverage} = context;
    const nodes: GPUCommandNode<Parameters>[] = [
      ...createSparseValidationNodes(graph, {
        id,
        operation: OPERATION,
        sparse,
        declarations,
        status: props.output.status
      })
    ];
    if (props.demandWeights || needsCoverage) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-validate-inputs`,
          operation: OPERATION,
          variant: 'validate-sparse-inputs',
          invocationCount: Math.max(demandCount, 1),
          bindings: [
            ...weightBinding,
            ...coverageBinding,
            {name: 'status', view: props.output.status, type: 'atomic<u32>', access: 'read_write'}
          ],
          declarations,
          body: getWeightAndCoverageValidationWGSL(props.demandWeights !== undefined, needsCoverage)
        })
      );
    }
    const {nodes: transposeNodes, transpose} = createSparseTransposeNodes(graph, {
      id,
      operation: OPERATION,
      sparse,
      facilityCount: props.facilityCount,
      declarations
    });
    nodes.push(...transposeNodes);
    return {nodes, transpose};
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, demandCount} = this;
    const {output, facilityCount} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      ...this.getCostViews(),
      props.demandWeights,
      props.coverageDistance,
      output.selectedFacilities,
      output.assignments,
      output.assignedCosts,
      output.objective,
      output.status
    ]);
    validateCapacity(
      id,
      'facilityCount',
      props.facilityCount,
      1,
      getGPULocationAllocationMaximumFacilityCount(graph.device)
    );
    if (props.costs) {
      validateMaximumCount(
        id,
        'costs rows',
        props.costs.length,
        getGPULocationAllocationMaximumCostCount(graph.device)
      );
    } else {
      validateMaximumCount(
        id,
        'sparseCosts entries',
        props.sparseCosts!.facilities.length,
        getGPULocationAllocationMaximumSparseEntryCount(graph.device)
      );
    }
    validateMaximumCount(
      id,
      'demand rows',
      demandCount,
      getGPULocationAllocationMaximumDemandCount(graph.device)
    );
    if (this.maximumSwapIterations > 0) {
      const maximumSwapPairs = getMaximumSwapPairCount(graph.device);
      if (props.maximumFacilityCount * props.facilityCount > maximumSwapPairs) {
        throw new Error(
          `${id} swap phase needs maximumFacilityCount * facilityCount <= ${maximumSwapPairs}; reduce them or set maximumSwapIterations: 0`
        );
      }
    }
    const bestCosts = createTransientView(graph, `${id}-best-costs`, 'float32', demandCount);
    const facilityScores = createTransientView(
      graph,
      `${id}-facility-scores`,
      'float32',
      2 * facilityCount
    );
    const chosenFacility = createTransientView(graph, `${id}-chosen-facility`, 'uint32', 1);
    const demandObjectives = createTransientView(
      graph,
      `${id}-demand-objectives`,
      'float32',
      demandCount
    );
    const uncovered = createTransientView(graph, `${id}-uncovered`, 'uint32', demandCount);
    const mode =
      props.operation === 'p-median' ? 0 : props.operation === 'maximum-coverage' ? 1 : 2;
    const needsCoverage = mode !== 0;
    const swaps = this.maximumSwapIterations;
    const sparse = props.sparseCosts;
    const costBindings: WGSLKernelBinding[] = sparse
      ? getSparseRowBindings(sparse)
      : [{name: 'costs', view: props.costs!, type: 'f32', access: 'read'}];
    const declarations = `const DEMAND_COUNT: u32 = ${demandCount}u;
const FACILITY_COUNT: u32 = ${facilityCount}u;${
      sparse ? `\nconst SPARSE_ENTRY_COUNT: u32 = ${sparse.facilities.length}u;` : ''
    }
const MAXIMUM_FACILITIES: u32 = ${props.maximumFacilityCount}u;
const MODE: u32 = ${mode}u;
const NONE: u32 = 0xffffffffu;
const LARGE_COST: f32 = 1e30;
${IS_NAN_BITS_WGSL}
fn isInvalid(value: f32) -> bool { return isNanBits(value) || value < 0.0; }
// Unreachable costs (+Infinity, or the LARGE_COST sentinel of an unassigned row) are never summed.
fn isReachable(cost: f32) -> bool { return cost < LARGE_COST; }`;
    const swapDeclarations = `${declarations}
const MAXIMUM_SWAPS: u32 = ${swaps}u;
const SWAP_SLOTS: u32 = ${props.maximumFacilityCount}u;
const SWAP_TOLERANCE: f32 = ${GPU_LOCATION_ALLOCATION_SWAP_TOLERANCE};`;
    // Word 0: stopped (no improving swap, or invalid input), word 1: swaps applied, word 2: chosen pair.
    const swapState =
      swaps > 0 ? createTransientView(graph, `${id}-swap-state`, 'uint32', 3) : undefined!;
    const weightBinding: WGSLKernelBinding[] = props.demandWeights
      ? [{name: 'demandWeights', view: props.demandWeights, type: 'f32', access: 'read'}]
      : [];
    const coverageBinding: WGSLKernelBinding[] = needsCoverage
      ? [
          {
            name: 'coverageDistance',
            view: props.coverageDistance!,
            type: 'f32',
            access: 'read'
          }
        ]
      : [];
    const demandWeightExpression = props.demandWeights
      ? 'demandWeights[demandWeightsOffset + demand]'
      : '1.0';
    // Coverage modes stop choosing once no facility improves the objective: the choose kernel then
    // zeroes the indirect dispatch of the score and update kernels (slots 0 and 1), so the remaining
    // steps skip the demand-by-facility scan. p-median only stops on invalid input.
    const greedyGate = needsCoverage
      ? createWGSLDispatchGate(graph, {
          id: `${id}-greedy-dispatch`,
          operation: OPERATION,
          shapes: [[facilityCount], [demandCount]]
        })
      : null;
    const greedyGateBinding: WGSLKernelBinding[] = greedyGate ? [greedyGate.binding] : [];
    const sparseSetup = sparse
      ? this.getSparseSetupNodes(graph, {
          declarations,
          weightBinding,
          coverageBinding,
          needsCoverage
        })
      : undefined;
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode(graph, {
        id: `${id}-initialize`,
        operation: OPERATION,
        variant: 'initialize',
        invocationCount: Math.max(
          facilityCount,
          demandCount,
          GPU_LOCATION_ALLOCATION_STATUS.length
        ),
        bindings: [
          {
            name: 'selected',
            view: output.selectedFacilities,
            type: 'u32',
            access: 'read_write'
          },
          {name: 'assignments', view: output.assignments, type: 'u32', access: 'read_write'},
          {name: 'bestCosts', view: bestCosts, type: 'f32', access: 'read_write'},
          {name: 'objective', view: output.objective, type: 'f32', access: 'read_write'},
          {name: 'status', view: output.status, type: 'u32', access: 'read_write'},
          {name: 'chosen', view: chosenFacility, type: 'u32', access: 'read_write'},
          ...greedyGateBinding
        ],
        declarations,
        body: `if (index < FACILITY_COUNT) {
    selected[selectedOffset + index] = 0u;
  }
  if (index < DEMAND_COUNT) {
    bestCosts[bestCostsOffset + index] = LARGE_COST;
    assignments[assignmentsOffset + index] = NONE;
  }
  if (index < ${GPU_LOCATION_ALLOCATION_STATUS.length}u) {
    status[statusOffset + index] = select(0u, DEMAND_COUNT, index == 1u);
  }
  if (index == 0u) {
    objective[objectiveOffset] = 0.0;
    // Any value but NONE: the first choose step is not stopped.
    chosen[chosenOffset] = 0u;
    ${greedyGate ? greedyGate.armWGSL : ''}
  }`
      }),
      ...(sparseSetup
        ? sparseSetup.nodes
        : [
            createWGSLKernelNode(graph, {
              id: `${id}-validate`,
              operation: OPERATION,
              variant: 'validate',
              invocationCount: Math.max(props.costs!.length, demandCount, 1),
              bindings: [
                {name: 'costs', view: props.costs!, type: 'f32', access: 'read'},
                ...weightBinding,
                ...coverageBinding,
                {name: 'status', view: output.status, type: 'atomic<u32>', access: 'read_write'}
              ],
              declarations,
              body: `if (index < DEMAND_COUNT * FACILITY_COUNT && isInvalid(costs[costsOffset + index])) {
    atomicAdd(&status[statusOffset + 3u], 1u);
  }
  ${getWeightAndCoverageValidationWGSL(props.demandWeights !== undefined, needsCoverage)}`
            })
          ])
    ];

    for (let step = 0; step < props.maximumFacilityCount; step++) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-score-${step}`,
          operation: OPERATION,
          variant: `${props.operation}-score`,
          invocationCount: facilityCount,
          ...(greedyGate ? greedyGate.getGate(0) : {}),
          bindings: [
            ...(sparseSetup
              ? getSparseColumnBindings(sparseSetup.transpose)
              : [{name: 'costs', view: props.costs!, type: 'f32', access: 'read'} as const]),
            {name: 'selected', view: output.selectedFacilities, type: 'u32', access: 'read'},
            {name: 'bestCosts', view: bestCosts, type: 'f32', access: 'read'},
            {name: 'scores', view: facilityScores, type: 'f32', access: 'read_write'},
            {name: 'status', view: output.status, type: 'u32', access: 'read'},
            ...weightBinding,
            ...coverageBinding
          ],
          declarations,
          body: `let facility = index;
  let unavailableScore = select(-1.0, LARGE_COST, MODE == 0u);
  if (status[statusOffset + 3u] != 0u || selected[selectedOffset + facility] != 0u) {
    scores[scoresOffset + 2u * facility] = unavailableScore;
    scores[scoresOffset + 2u * facility + 1u] = unavailableScore;
    return;
  }
  let threshold = ${needsCoverage ? 'coverageDistance[coverageDistanceOffset]' : '0.0'};
  // p-median scores are lexicographic: (demand weight left unreachable, weighted cost of the
  // reachable rows). Coverage modes use only the first value.
  var score = 0.0;
  var reachableCost = 0.0;
  ${
    sparseSetup
      ? `// Sparse: the facility only changes the rows it lists, so the score is the negated gain
  // (reachable weight won, cost saved), which orders candidates like the dense totals.
  let columnEnd = columnOffsets[columnOffsetsOffset + facility + 1u];
  for (var entry = columnOffsets[columnOffsetsOffset + facility]; entry < columnEnd; entry++) {
    let demand = columnEntries[columnEntriesOffset + 2u * entry];
    let candidate = min(bitcast<f32>(columnEntries[columnEntriesOffset + 2u * entry + 1u]), LARGE_COST);
    let current = bestCosts[bestCostsOffset + demand];
    if (MODE == 0u) {
      if (candidate < current) {
        if (isReachable(current)) {
          reachableCost += ${demandWeightExpression} * (current - candidate);
        } else {
          score += ${demandWeightExpression};
          reachableCost -= ${demandWeightExpression} * candidate;
        }
      }
    } else if (current > threshold && candidate <= threshold) {
      score += ${demandWeightExpression};
    }
  }
  if (MODE == 0u) {
    score = -score;
    reachableCost = -reachableCost;
  }`
      : `for (var demand = 0u; demand < DEMAND_COUNT; demand++) {
    let current = bestCosts[bestCostsOffset + demand];
    let candidate = costs[costsOffset + demand * FACILITY_COUNT + facility];
    if (MODE == 0u) {
      let next = min(current, candidate);
      if (isReachable(next)) {
        reachableCost += ${demandWeightExpression} * next;
      } else {
        score += ${demandWeightExpression};
      }
    } else if (current > threshold && candidate <= threshold) {
      score += ${demandWeightExpression};
    }
  }`
  }
  scores[scoresOffset + 2u * facility] = score;
  scores[scoresOffset + 2u * facility + 1u] = reachableCost;`
        }),
        createWGSLKernelNode(graph, {
          id: `${id}-choose-${step}`,
          operation: OPERATION,
          variant: `${props.operation}-choose`,
          invocationCount: CHOOSE_WORKGROUP_SIZE,
          workgroupSize: CHOOSE_WORKGROUP_SIZE,
          guardIndex: false,
          bindings: [
            {name: 'selected', view: output.selectedFacilities, type: 'u32', access: 'read_write'},
            {name: 'scores', view: facilityScores, type: 'f32', access: 'read'},
            {name: 'chosen', view: chosenFacility, type: 'u32', access: 'read_write'},
            {name: 'status', view: output.status, type: 'u32', access: 'read_write'},
            ...greedyGateBinding
          ],
          declarations: `${declarations}
const CHOOSE_WORKGROUP_SIZE: u32 = ${CHOOSE_WORKGROUP_SIZE}u;
var<workgroup> sharedScore: array<f32, ${CHOOSE_WORKGROUP_SIZE}>;
var<workgroup> sharedSecondary: array<f32, ${CHOOSE_WORKGROUP_SIZE}>;
var<workgroup> sharedFacility: array<u32, ${CHOOSE_WORKGROUP_SIZE}>;
// p-median: lowest (unreachable weight, reachable cost); coverage: highest score.
fn isBetterCandidate(
  score: f32, secondary: f32, facility: u32,
  bestScore: f32, bestSecondary: f32, bestFacility: u32
) -> bool {
  if (MODE == 0u) {
    if (score != bestScore) { return score < bestScore; }
    if (secondary != bestSecondary) { return secondary < bestSecondary; }
    return facility < bestFacility;
  }
  return score > bestScore || (score == bestScore && facility < bestFacility);
}`,
          // One workgroup scans the facilities with a strided loop, then tree-reduces. The
          // (score, facility) order is total, so the lowest-facility tie rule is unchanged.
          // No early return: every invocation must reach the barriers.
          // A coverage step after one that chose nothing is stopped: its score kernel was skipped,
          // so the scores are stale. Every lane reads chosen before the first barrier and lane 0
          // writes it after the last one.
          body: `var bestFacility = NONE;
  var bestScore = select(-1.0, LARGE_COST, MODE == 0u);
  var bestSecondary = select(-1.0, LARGE_COST, MODE == 0u);
  let stopped = MODE != 0u && chosen[chosenOffset] == NONE;
  if (status[statusOffset + 3u] == 0u && !stopped) {
    for (var facility = localInvocationIndex; facility < FACILITY_COUNT; facility += CHOOSE_WORKGROUP_SIZE) {
      if (selected[selectedOffset + facility] != 0u) { continue; }
      let score = scores[scoresOffset + 2u * facility];
      let secondary = scores[scoresOffset + 2u * facility + 1u];
      if (isBetterCandidate(score, secondary, facility, bestScore, bestSecondary, bestFacility)) {
        bestFacility = facility;
        bestScore = score;
        bestSecondary = secondary;
      }
    }
  }
  sharedScore[localInvocationIndex] = bestScore;
  sharedSecondary[localInvocationIndex] = bestSecondary;
  sharedFacility[localInvocationIndex] = bestFacility;
  workgroupBarrier();
  for (var stride = CHOOSE_WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride &&
        isBetterCandidate(
          sharedScore[localInvocationIndex + stride],
          sharedSecondary[localInvocationIndex + stride],
          sharedFacility[localInvocationIndex + stride],
          sharedScore[localInvocationIndex],
          sharedSecondary[localInvocationIndex],
          sharedFacility[localInvocationIndex])) {
      sharedScore[localInvocationIndex] = sharedScore[localInvocationIndex + stride];
      sharedSecondary[localInvocationIndex] = sharedSecondary[localInvocationIndex + stride];
      sharedFacility[localInvocationIndex] = sharedFacility[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u) {
    bestFacility = sharedFacility[0];
    if (bestFacility != NONE && (MODE == 0u || sharedScore[0] > 0.0)) {
      selected[selectedOffset + bestFacility] = 1u;
      status[statusOffset] += 1u;
    } else {
      bestFacility = NONE;${
        greedyGate
          ? `
      ${greedyGate.disarmWGSL}`
          : ''
      }
    }
    chosen[chosenOffset] = bestFacility;
  }`
        }),
        createWGSLKernelNode(graph, {
          id: `${id}-update-${step}`,
          operation: OPERATION,
          variant: `${props.operation}-update`,
          invocationCount: demandCount,
          ...(greedyGate ? greedyGate.getGate(1) : {}),
          bindings: [
            ...costBindings,
            {name: 'bestCosts', view: bestCosts, type: 'f32', access: 'read_write'},
            {name: 'chosen', view: chosenFacility, type: 'u32', access: 'read'}
          ],
          declarations: sparse ? `${declarations}\n${SPARSE_ROW_COST_WGSL}` : declarations,
          body: `let facility = chosen[chosenOffset];
  if (facility != NONE) {
    let candidate = ${
      sparse
        ? 'getSparseRowCost(index, facility)'
        : 'costs[costsOffset + index * FACILITY_COUNT + facility]'
    };
    bestCosts[bestCostsOffset + index] = min(bestCosts[bestCostsOffset + index], candidate);
  }`
        })
      );
    }

    if (swaps > 0) {
      nodes.push(
        ...this.getSwapNodes(graph, {
          declarations: swapDeclarations,
          swapState,
          weightBinding,
          coverageBinding,
          demandWeightExpression,
          transpose: sparseSetup?.transpose
        })
      );
    }

    const assignedCostsWGSL = output.assignedCosts
      ? 'assignedCosts[assignedCostsOffset + demand] = select(bitcast<f32>(0x7f800000u | (demand & 0u)), assignedCost, assigned != NONE);'
      : '';
    const objectiveWGSL = `let threshold = ${needsCoverage ? 'coverageDistance[coverageDistanceOffset]' : '0.0'};
  let isUncovered = assigned == NONE || (MODE != 0u && assignedCost > threshold);
  uncovered[uncoveredOffset + demand] = select(0u, 1u, isUncovered);
  let weight = ${demandWeightExpression};
  // p-median sums the reachable rows only; unreachable rows are counted in uncoveredDemandCount.
  demandObjectives[demandObjectivesOffset + demand] = select(
    select(0.0, weight, !isUncovered),
    select(weight * assignedCost, 0.0, isUncovered),
    MODE == 0u
  );`;
    const assignedCostsBinding: WGSLKernelBinding[] = output.assignedCosts
      ? [{name: 'assignedCosts', view: output.assignedCosts, type: 'f32', access: 'read_write'}]
      : [];
    const objectiveBindings: WGSLKernelBinding[] = [
      {name: 'demandObjectives', view: demandObjectives, type: 'f32', access: 'read_write'},
      {name: 'uncovered', view: uncovered, type: 'u32', access: 'read_write'},
      ...weightBinding,
      ...coverageBinding,
      ...assignedCostsBinding
    ];
    if (sparse) {
      // Sparse rows list their facilities in ascending order, so the first strictly cheaper
      // selected entry wins ties like the dense lowest-facility rule. Two kernels keep each
      // within the eight-binding limit.
      const assignedCostScratch = createTransientView(
        graph,
        `${id}-assigned-cost-scratch`,
        'float32',
        demandCount
      );
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-assign`,
          operation: OPERATION,
          variant: `${props.operation}-assign-sparse`,
          invocationCount: demandCount,
          bindings: [
            ...costBindings,
            {name: 'selected', view: output.selectedFacilities, type: 'u32', access: 'read'},
            {name: 'assignments', view: output.assignments, type: 'u32', access: 'read_write'},
            {
              name: 'assignedCostScratch',
              view: assignedCostScratch,
              type: 'f32',
              access: 'read_write'
            }
          ],
          declarations,
          body: `let demand = index;
  var assigned = NONE;
  var assignedCost = LARGE_COST;
  let rowEnd = min(rowOffsets[rowOffsetsOffset + demand + 1u], SPARSE_ENTRY_COUNT);
  for (var entry = rowOffsets[rowOffsetsOffset + demand]; entry < rowEnd; entry++) {
    let facility = rowFacilities[rowFacilitiesOffset + entry];
    if (facility >= FACILITY_COUNT || selected[selectedOffset + facility] == 0u) { continue; }
    let cost = rowCosts[rowCostsOffset + entry];
    if (cost < assignedCost) {
      assignedCost = cost;
      assigned = facility;
    }
  }
  assignments[assignmentsOffset + demand] = assigned;
  assignedCostScratch[assignedCostScratchOffset + demand] = assignedCost;`
        }),
        createWGSLKernelNode(graph, {
          id: `${id}-objective`,
          operation: OPERATION,
          variant: `${props.operation}-objective`,
          invocationCount: demandCount,
          bindings: [
            {name: 'assignments', view: output.assignments, type: 'u32', access: 'read'},
            {name: 'assignedCostScratch', view: assignedCostScratch, type: 'f32', access: 'read'},
            ...objectiveBindings
          ],
          declarations,
          body: `let demand = index;
  let assigned = assignments[assignmentsOffset + demand];
  let assignedCost = assignedCostScratch[assignedCostScratchOffset + demand];
  ${assignedCostsWGSL}
  ${objectiveWGSL}`
        })
      );
    } else {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-assign`,
          operation: OPERATION,
          variant: `${props.operation}-assign`,
          invocationCount: demandCount,
          bindings: [
            ...costBindings,
            {name: 'selected', view: output.selectedFacilities, type: 'u32', access: 'read'},
            {name: 'assignments', view: output.assignments, type: 'u32', access: 'read_write'},
            ...objectiveBindings
          ],
          declarations,
          body: `let demand = index;
  var assigned = NONE;
  var assignedCost = LARGE_COST;
  for (var facility = 0u; facility < FACILITY_COUNT; facility++) {
    if (selected[selectedOffset + facility] == 0u) { continue; }
    let cost = costs[costsOffset + demand * FACILITY_COUNT + facility];
    if (cost < assignedCost || (cost == assignedCost && facility < assigned)) {
      assignedCost = cost;
      assigned = facility;
    }
  }
  assignments[assignmentsOffset + demand] = assigned;
  ${assignedCostsWGSL}
  ${objectiveWGSL}`
        })
      );
    }
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-finalize`,
        operation: OPERATION,
        variant: `${props.operation}-finalize`,
        invocationCount: CHOOSE_WORKGROUP_SIZE,
        workgroupSize: CHOOSE_WORKGROUP_SIZE,
        guardIndex: false,
        bindings: [
          {name: 'demandObjectives', view: demandObjectives, type: 'f32', access: 'read'},
          {name: 'uncovered', view: uncovered, type: 'u32', access: 'read'},
          {name: 'objective', view: output.objective, type: 'f32', access: 'read_write'},
          {name: 'status', view: output.status, type: 'u32', access: 'read_write'},
          ...(swaps > 0
            ? [{name: 'swapState', view: swapState, type: 'u32' as const, access: 'read' as const}]
            : [])
        ],
        declarations: `${swapDeclarations}
const FINALIZE_WORKGROUP_SIZE: u32 = ${CHOOSE_WORKGROUP_SIZE}u;
var<workgroup> sharedObjective: array<f32, ${CHOOSE_WORKGROUP_SIZE}>;
var<workgroup> sharedUncovered: array<u32, ${CHOOSE_WORKGROUP_SIZE}>;`,
        // One workgroup sums strided partials, then tree-reduces in a fixed order. No early
        // return: every invocation must reach the barriers.
        body: `var uncoveredCount = 0u;
  var objectiveValue = 0.0;
  for (var demand = localInvocationIndex; demand < DEMAND_COUNT; demand += FINALIZE_WORKGROUP_SIZE) {
    uncoveredCount += uncovered[uncoveredOffset + demand];
    objectiveValue += demandObjectives[demandObjectivesOffset + demand];
  }
  sharedObjective[localInvocationIndex] = objectiveValue;
  sharedUncovered[localInvocationIndex] = uncoveredCount;
  workgroupBarrier();
  for (var stride = FINALIZE_WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride) {
      sharedObjective[localInvocationIndex] += sharedObjective[localInvocationIndex + stride];
      sharedUncovered[localInvocationIndex] += sharedUncovered[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (localInvocationIndex == 0u && status[statusOffset + 3u] == 0u) {
    let selectedCount = status[statusOffset];
    let totalUncovered = sharedUncovered[0];
    objective[objectiveOffset] = select(sharedObjective[0], f32(selectedCount), MODE == 2u);
    status[statusOffset + 1u] = totalUncovered;
    status[statusOffset + 2u] = select(
      0u,
      1u,
      select(selectedCount == MAXIMUM_FACILITIES, totalUncovered == 0u, MODE == 2u)
    );${
      swaps > 0
        ? `
    status[statusOffset + 4u] = swapState[swapStateOffset + 1u];
    status[statusOffset + 5u] = swapState[swapStateOffset];
    status[statusOffset + 6u] = select(
      0u,
      1u,
      swapState[swapStateOffset] == 0u && swapState[swapStateOffset + 1u] >= MAXIMUM_SWAPS
    );`
        : ''
    }
  }`
      })
    );
    return nodes;
  }
}
