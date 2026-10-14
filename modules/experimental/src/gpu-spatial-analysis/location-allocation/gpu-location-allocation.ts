// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {IS_NAN_BITS_WGSL} from '../../utils/wgsl-common';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';

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
  swapIterations: 4,
  /** `1` when the swap phase stopped because no strictly improving swap exists (local optimum). */
  converged: 5,
  /** `1` when `maximumSwapIterations` swaps were applied, so local optimality is not verified. */
  iterationLimitReached: 6,
  length: 7
} as const;

/** Default `maximumSwapIterations` of the p-median and maximum-coverage operations. */
export const GPU_LOCATION_ALLOCATION_DEFAULT_SWAP_ITERATIONS = 100;

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

/** Largest number of (selected slot, candidate) pairs one swap iteration scores. */
const MAXIMUM_SWAP_PAIRS = 2 ** 24;

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
  /** Row-major demand-by-facility cost matrix. Positive infinity means unreachable. */
  costs: GraphDataView<'float32'>;
  /** Number of candidate facility columns in `costs`. */
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
   * swap in parallel (`O(maximumFacilityCount * facilityCount * demandCount)` work), applies the
   * single best strictly improving swap (ties go to the lowest selected facility, then the lowest
   * candidate), and stops when none improves. Requires
   * `maximumFacilityCount * facilityCount <= 2^24`.
   */
  maximumSwapIterations?: number;
  /** One per-frame distance threshold, required by maximum coverage and set covering. */
  coverageDistance?: GraphDataView<'float32'>;
  output: GPULocationAllocationOutput;
};

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
 * reported in the `swapIterations`, `converged` and `iterationLimitReached` status words.
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
    validatePackedView(props.costs, ['float32'], `${id} costs`);
    if (!Number.isSafeInteger(props.facilityCount) || props.facilityCount < 1) {
      throw new Error(`${id} facilityCount must be a positive integer`);
    }
    if (
      props.costs.length < props.facilityCount ||
      props.costs.length % props.facilityCount !== 0
    ) {
      throw new Error(`${id} costs length must be a positive multiple of facilityCount`);
    }
    this.demandCount = props.costs.length / props.facilityCount;
    if (
      !Number.isSafeInteger(props.maximumFacilityCount) ||
      props.maximumFacilityCount < 1 ||
      props.maximumFacilityCount > props.facilityCount
    ) {
      throw new Error(`${id} maximumFacilityCount must be an integer in [1, facilityCount]`);
    }
    const requestedSwaps = props.maximumSwapIterations;
    if (
      requestedSwaps !== undefined &&
      (!Number.isInteger(requestedSwaps) ||
        requestedSwaps < 0 ||
        requestedSwaps > GPU_LOCATION_ALLOCATION_MAXIMUM_SWAP_ITERATIONS)
    ) {
      throw new Error(
        `${id} maximumSwapIterations must be an integer in [0, ${GPU_LOCATION_ALLOCATION_MAXIMUM_SWAP_ITERATIONS}]`
      );
    }
    if (props.operation === 'set-covering') {
      if (requestedSwaps) {
        throw new Error(`${id} set-covering has no swap phase; omit maximumSwapIterations`);
      }
      this.maximumSwapIterations = 0;
    } else {
      this.maximumSwapIterations =
        requestedSwaps ?? GPU_LOCATION_ALLOCATION_DEFAULT_SWAP_ITERATIONS;
    }
    if (
      this.maximumSwapIterations > 0 &&
      props.maximumFacilityCount * props.facilityCount > MAXIMUM_SWAP_PAIRS
    ) {
      throw new Error(
        `${id} swap phase needs maximumFacilityCount * facilityCount <= 2^24; reduce them or set maximumSwapIterations: 0`
      );
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
    validatePackedUint32View(output.selectedFacilities, `${id} output.selectedFacilities`);
    validatePackedUint32View(output.assignments, `${id} output.assignments`);
    validatePackedUint32View(output.status, `${id} output.status`);
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
      [props.costs, props.demandWeights, props.coverageDistance]
    );
  }

  /**
   * Teitz-Bart vertex substitution. Each iteration: per demand row the nearest and second-nearest
   * selected cost (or the covering count), the ascending list of selected facilities, the
   * improvement of every (selected slot, candidate) pair, a single-workgroup argmax (ties to the
   * lowest pair index, i.e. lowest selected facility then lowest candidate), and the swap itself.
   * A stopped flag makes the remaining iterations early-out once no swap improves.
   */
  private getSwapNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    context: {
      declarations: string;
      swapState: GraphDataView<'uint32'>;
      weightBinding: WGSLKernelBinding[];
      coverageBinding: WGSLKernelBinding[];
      demandWeightExpression: string;
    }
  ): GPUCommandNode<Parameters>[] {
    const {id, props, demandCount} = this;
    const {output, facilityCount} = props;
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
    // Per demand: nearest and second-nearest selected cost plus the nearest facility (p-median), or
    // the number of selected facilities within the coverage distance (stored in `first`).
    // Packed as three u32 words per demand row: nearest cost bits, second cost bits, first/count
    // (a device allows only eight storage buffers per kernel).
    const demandData = view('demand-data', 'uint32', demandCount * 3);
    const selectedList = view('selected-list', 'uint32', slots);
    // Two values per pair: (primary, secondary) improvement, -1 when the pair is not a swap.
    const pairScores = view('pair-scores', 'float32', 2 * pairCount);
    const threshold = coverageMode ? 'coverageDistance[coverageDistanceOffset]' : '0.0';
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode(graph, {
        id: `${id}-swap-init`,
        operation: OPERATION,
        variant: 'swap-init',
        invocationCount: 1,
        bindings: [{name: 'swapState', view: swapState, type: 'u32', access: 'read_write'}],
        declarations,
        body: `swapState[swapStateOffset] = 0u;
  swapState[swapStateOffset + 1u] = 0u;
  swapState[swapStateOffset + 2u] = NONE;`
      })
    ];
    for (let iteration = 0; iteration < this.maximumSwapIterations; iteration++) {
      const iterationId = `${id}-swap-${iteration}`;
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${iterationId}-prepare`,
          operation: OPERATION,
          variant: `${props.operation}-swap-prepare`,
          invocationCount: demandCount,
          bindings: [
            {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
            {name: 'selected', view: output.selectedFacilities, type: 'u32', access: 'read'},
            {name: 'swapState', view: swapState, type: 'u32', access: 'read'},
            {name: 'demandData', view: demandData, type: 'u32', access: 'read_write'},
            ...coverageBinding
          ],
          declarations,
          body: `if (swapState[swapStateOffset] != 0u) {
  return;
}
let demand = index;
var nearestCost = LARGE_COST;
var secondCost = LARGE_COST;
var nearestFacility = NONE;
var coveringCount = 0u;
let threshold = ${threshold};
for (var facility = 0u; facility < FACILITY_COUNT; facility++) {
  if (selected[selectedOffset + facility] == 0u) { continue; }
  let cost = min(costs[costsOffset + demand * FACILITY_COUNT + facility], LARGE_COST);
  if (cost < nearestCost) {
    secondCost = nearestCost;
    nearestCost = cost;
    nearestFacility = facility;
  } else if (cost < secondCost) {
    secondCost = cost;
  }
  if (cost <= threshold) { coveringCount++; }
}
demandData[demandDataOffset + demand * 3u] = bitcast<u32>(nearestCost);
demandData[demandDataOffset + demand * 3u + 1u] = bitcast<u32>(secondCost);
demandData[demandDataOffset + demand * 3u + 2u] = select(nearestFacility, coveringCount, MODE == 1u);`
        }),
        createWGSLKernelNode(graph, {
          id: `${iterationId}-list`,
          operation: OPERATION,
          variant: 'swap-list',
          invocationCount: 1,
          bindings: [
            {name: 'selected', view: output.selectedFacilities, type: 'u32', access: 'read'},
            {name: 'status', view: output.status, type: 'u32', access: 'read'},
            {name: 'swapState', view: swapState, type: 'u32', access: 'read'},
            {name: 'selectedList', view: selectedList, type: 'u32', access: 'read_write'}
          ],
          declarations,
          // Stopped or invalid: leave every slot empty so the score kernel does no work.
          body: `for (var slot = 0u; slot < SWAP_SLOTS; slot++) {
  selectedList[selectedListOffset + slot] = NONE;
}
if (swapState[swapStateOffset] != 0u || status[statusOffset + 3u] != 0u) {
  return;
}
var slot = 0u;
for (var facility = 0u; facility < FACILITY_COUNT; facility++) {
  if (selected[selectedOffset + facility] != 0u && slot < SWAP_SLOTS) {
    selectedList[selectedListOffset + slot] = facility;
    slot++;
  }
}`
        }),
        createWGSLKernelNode(graph, {
          id: `${iterationId}-score`,
          operation: OPERATION,
          variant: `${props.operation}-swap-score`,
          invocationCount: pairCount,
          bindings: [
            {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
            {name: 'selected', view: output.selectedFacilities, type: 'u32', access: 'read'},
            {name: 'demandData', view: demandData, type: 'u32', access: 'read'},
            {name: 'selectedList', view: selectedList, type: 'u32', access: 'read'},
            {name: 'pairScores', view: pairScores, type: 'f32', access: 'read_write'},
            ...weightBinding,
            ...coverageBinding
          ],
          declarations,
          // Improvement is positive when the swap helps: cost removed for p-median, covered weight
          // gained for maximum coverage. A pair is only kept when it beats the noise tolerance.
          body: `let slot = index / FACILITY_COUNT;
let candidate = index % FACILITY_COUNT;
let removed = selectedList[selectedListOffset + slot];
pairScores[pairScoresOffset + 2u * index] = -1.0;
pairScores[pairScoresOffset + 2u * index + 1u] = -1.0;
if (removed == NONE || selected[selectedOffset + candidate] != 0u) {
  return;
}
let threshold = ${threshold};
var improvement = 0.0;
var scale = 0.0;
// p-median: reachable demand weight gained first, then reachable cost removed (lexicographic).
var reachGain = 0.0;
var totalWeight = 0.0;
for (var demand = 0u; demand < DEMAND_COUNT; demand++) {
  let weight = ${demandWeightExpression};
  let candidateCost = costs[costsOffset + demand * FACILITY_COUNT + candidate];
  if (MODE == 0u) {
    let current = bitcast<f32>(demandData[demandDataOffset + demand * 3u]);
    let second = bitcast<f32>(demandData[demandDataOffset + demand * 3u + 1u]);
    let base = select(current, second, demandData[demandDataOffset + demand * 3u + 2u] == removed);
    let next = min(base, min(candidateCost, LARGE_COST));
    let currentReachable = isReachable(current);
    let nextReachable = isReachable(next);
    reachGain += weight * (select(0.0, 1.0, nextReachable) - select(0.0, 1.0, currentReachable));
    improvement += select(0.0, weight * current, currentReachable) - select(0.0, weight * next, nextReachable);
    totalWeight += weight;
    if (currentReachable) { scale += weight * current; }
  } else {
    let count = demandData[demandDataOffset + demand * 3u + 2u];
    let removedCovers = costs[costsOffset + demand * FACILITY_COUNT + removed] <= threshold;
    let candidateCovers = candidateCost <= threshold;
    let after = count - select(0u, 1u, removedCovers) + select(0u, 1u, candidateCovers);
    improvement += weight * (select(0.0, 1.0, after > 0u) - select(0.0, 1.0, count > 0u));
    scale += weight;
  }
}
if (MODE == 0u) {
  if (reachGain > SWAP_TOLERANCE * totalWeight) {
    pairScores[pairScoresOffset + 2u * index] = reachGain;
    pairScores[pairScoresOffset + 2u * index + 1u] = improvement;
  } else if (abs(reachGain) <= SWAP_TOLERANCE * totalWeight && improvement > SWAP_TOLERANCE * scale) {
    pairScores[pairScoresOffset + 2u * index] = 0.0;
    pairScores[pairScoresOffset + 2u * index + 1u] = improvement;
  }
} else if (improvement > SWAP_TOLERANCE * scale) {
  pairScores[pairScoresOffset + 2u * index] = improvement;
  pairScores[pairScoresOffset + 2u * index + 1u] = 0.0;
}`
        }),
        createWGSLKernelNode(graph, {
          id: `${iterationId}-choose`,
          operation: OPERATION,
          variant: 'swap-choose',
          invocationCount: CHOOSE_WORKGROUP_SIZE,
          workgroupSize: CHOOSE_WORKGROUP_SIZE,
          guardIndex: false,
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
            {name: 'swapState', view: swapState, type: 'u32', access: 'read_write'}
          ],
          declarations,
          body: `if (swapState[swapStateOffset] != 0u) {
  return;
}
let pair = swapState[swapStateOffset + 2u];
if (pair == NONE) {
  swapState[swapStateOffset] = 1u;
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

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, demandCount} = this;
    const {output, facilityCount} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.costs,
      props.demandWeights,
      props.coverageDistance,
      output.selectedFacilities,
      output.assignments,
      output.assignedCosts,
      output.objective,
      output.status
    ]);
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
    const declarations = `const DEMAND_COUNT: u32 = ${demandCount}u;
const FACILITY_COUNT: u32 = ${facilityCount}u;
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
          {name: 'status', view: output.status, type: 'u32', access: 'read_write'}
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
  }`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-validate`,
        operation: OPERATION,
        variant: 'validate',
        invocationCount: Math.max(props.costs.length, demandCount, 1),
        bindings: [
          {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
          ...weightBinding,
          ...coverageBinding,
          {name: 'status', view: output.status, type: 'atomic<u32>', access: 'read_write'}
        ],
        declarations,
        body: `if (index < DEMAND_COUNT * FACILITY_COUNT && isInvalid(costs[costsOffset + index])) {
    atomicAdd(&status[statusOffset + 3u], 1u);
  }
  ${
    props.demandWeights
      ? `if (index < DEMAND_COUNT && isInvalid(demandWeights[demandWeightsOffset + index])) {
    atomicAdd(&status[statusOffset + 3u], 1u);
  }`
      : ''
  }
  ${
    needsCoverage
      ? `if (index == 0u && isInvalid(coverageDistance[coverageDistanceOffset])) {
    atomicAdd(&status[statusOffset + 3u], 1u);
  }`
      : ''
  }`
      })
    ];

    for (let step = 0; step < props.maximumFacilityCount; step++) {
      // Coverage modes stop choosing once no facility improves the objective, so later steps skip
      // the full demand-by-facility scan. p-median only stops on invalid input, which the status
      // check already covers.
      const gateOnPrevious = step > 0 && needsCoverage;
      const previousChosenBinding: WGSLKernelBinding[] = gateOnPrevious
        ? [{name: 'previousChosen', view: chosenFacility, type: 'u32', access: 'read'}]
        : [];
      const stoppedCondition = gateOnPrevious
        ? ' || previousChosen[previousChosenOffset] == NONE'
        : '';
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-score-${step}`,
          operation: OPERATION,
          variant: `${props.operation}-score`,
          invocationCount: facilityCount,
          bindings: [
            {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
            {name: 'selected', view: output.selectedFacilities, type: 'u32', access: 'read'},
            {name: 'bestCosts', view: bestCosts, type: 'f32', access: 'read'},
            {name: 'scores', view: facilityScores, type: 'f32', access: 'read_write'},
            {name: 'status', view: output.status, type: 'u32', access: 'read'},
            ...weightBinding,
            ...coverageBinding,
            ...previousChosenBinding
          ],
          declarations,
          body: `let facility = index;
  let unavailableScore = select(-1.0, LARGE_COST, MODE == 0u);
  if (status[statusOffset + 3u] != 0u || selected[selectedOffset + facility] != 0u${stoppedCondition}) {
    scores[scoresOffset + 2u * facility] = unavailableScore;
    scores[scoresOffset + 2u * facility + 1u] = unavailableScore;
    return;
  }
  let threshold = ${needsCoverage ? 'coverageDistance[coverageDistanceOffset]' : '0.0'};
  // p-median scores are lexicographic: (demand weight left unreachable, weighted cost of the
  // reachable rows). Coverage modes use only the first value.
  var score = 0.0;
  var reachableCost = 0.0;
  for (var demand = 0u; demand < DEMAND_COUNT; demand++) {
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
            {name: 'status', view: output.status, type: 'u32', access: 'read_write'}
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
          body: `var bestFacility = NONE;
  var bestScore = select(-1.0, LARGE_COST, MODE == 0u);
  var bestSecondary = select(-1.0, LARGE_COST, MODE == 0u);
  if (status[statusOffset + 3u] == 0u) {
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
      bestFacility = NONE;
    }
    chosen[chosenOffset] = bestFacility;
  }`
        }),
        createWGSLKernelNode(graph, {
          id: `${id}-update-${step}`,
          operation: OPERATION,
          variant: `${props.operation}-update`,
          invocationCount: demandCount,
          bindings: [
            {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
            {name: 'bestCosts', view: bestCosts, type: 'f32', access: 'read_write'},
            {name: 'chosen', view: chosenFacility, type: 'u32', access: 'read'}
          ],
          declarations,
          body: `let facility = chosen[chosenOffset];
  if (facility != NONE) {
    let candidate = costs[costsOffset + index * FACILITY_COUNT + facility];
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
          demandWeightExpression
        })
      );
    }

    const assignmentBindings: WGSLKernelBinding[] = [
      {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
      {name: 'selected', view: output.selectedFacilities, type: 'u32', access: 'read'},
      {name: 'assignments', view: output.assignments, type: 'u32', access: 'read_write'},
      {name: 'demandObjectives', view: demandObjectives, type: 'f32', access: 'read_write'},
      {name: 'uncovered', view: uncovered, type: 'u32', access: 'read_write'},
      ...weightBinding,
      ...coverageBinding
    ];
    if (output.assignedCosts) {
      assignmentBindings.push({
        name: 'assignedCosts',
        view: output.assignedCosts,
        type: 'f32',
        access: 'read_write'
      });
    }
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-assign`,
        operation: OPERATION,
        variant: `${props.operation}-assign`,
        invocationCount: demandCount,
        bindings: assignmentBindings,
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
  ${output.assignedCosts ? 'assignedCosts[assignedCostsOffset + demand] = select(bitcast<f32>(0x7f800000u | (demand & 0u)), assignedCost, assigned != NONE);' : ''}
  let threshold = ${needsCoverage ? 'coverageDistance[coverageDistanceOffset]' : '0.0'};
  let isUncovered = assigned == NONE || (MODE != 0u && assignedCost > threshold);
  uncovered[uncoveredOffset + demand] = select(0u, 1u, isUncovered);
  let weight = ${demandWeightExpression};
  // p-median sums the reachable rows only; unreachable rows are counted in uncoveredDemandCount.
  demandObjectives[demandObjectivesOffset + demand] = select(
    select(0.0, weight, !isUncovered),
    select(weight * assignedCost, 0.0, isUncovered),
    MODE == 0u
  );`
      }),
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
