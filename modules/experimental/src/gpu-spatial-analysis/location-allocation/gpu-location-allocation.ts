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
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';

const OPERATION = 'GPULocationAllocation';

/** Location-allocation objective implemented by {@link GPULocationAllocation}. */
export type GPULocationAllocationOperation = 'p-median' | 'maximum-coverage' | 'set-covering';

/** Packed status words written by {@link GPULocationAllocation}. */
export const GPU_LOCATION_ALLOCATION_STATUS = {
  selectedFacilityCount: 0,
  uncoveredDemandCount: 1,
  complete: 2,
  invalidCount: 3,
  length: 4
} as const;

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
  /** Four words indexed by {@link GPU_LOCATION_ALLOCATION_STATUS}. */
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
 * These are the standard bounded greedy heuristics, not an integer-programming optimum. Run them
 * after `GPUNetworkCostMatrix` for network location-allocation or provide any dense cost matrix.
 */
export class GPULocationAllocation implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPULocationAllocationProps;
  readonly demandCount: number;

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
      facilityCount
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
    const declarations = `const DEMAND_COUNT: u32 = ${demandCount}u;
const FACILITY_COUNT: u32 = ${facilityCount}u;
const MAXIMUM_FACILITIES: u32 = ${props.maximumFacilityCount}u;
const MODE: u32 = ${mode}u;
const NONE: u32 = 0xffffffffu;
const LARGE_COST: f32 = 1e30;
fn isInvalid(value: f32) -> bool { return value != value || value < 0.0; }`;
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
            ...coverageBinding
          ],
          declarations,
          body: `let facility = index;
  let unavailableScore = select(-1.0, LARGE_COST, MODE == 0u);
  if (status[statusOffset + 3u] != 0u || selected[selectedOffset + facility] != 0u) {
    scores[scoresOffset + facility] = unavailableScore;
    return;
  }
  let threshold = ${needsCoverage ? 'coverageDistance[coverageDistanceOffset]' : '0.0'};
  var score = 0.0;
  for (var demand = 0u; demand < DEMAND_COUNT; demand++) {
    let current = bestCosts[bestCostsOffset + demand];
    let candidate = costs[costsOffset + demand * FACILITY_COUNT + facility];
    if (MODE == 0u) {
      score += ${demandWeightExpression} * min(current, candidate);
    } else if (current > threshold && candidate <= threshold) {
      score += ${demandWeightExpression};
    }
  }
  scores[scoresOffset + facility] = score;`
        }),
        createWGSLKernelNode(graph, {
          id: `${id}-choose-${step}`,
          operation: OPERATION,
          variant: `${props.operation}-choose`,
          invocationCount: 1,
          bindings: [
            {name: 'selected', view: output.selectedFacilities, type: 'u32', access: 'read_write'},
            {name: 'scores', view: facilityScores, type: 'f32', access: 'read'},
            {name: 'chosen', view: chosenFacility, type: 'u32', access: 'read_write'},
            {name: 'status', view: output.status, type: 'u32', access: 'read_write'}
          ],
          declarations,
          body: `var bestFacility = NONE;
  var bestScore = select(-1.0, LARGE_COST, MODE == 0u);
  if (status[statusOffset + 3u] == 0u) {
    for (var facility = 0u; facility < FACILITY_COUNT; facility++) {
      if (selected[selectedOffset + facility] != 0u) { continue; }
      let score = scores[scoresOffset + facility];
      let improves = select(score > bestScore, score < bestScore, MODE == 0u);
      if (improves || (score == bestScore && facility < bestFacility)) {
        bestFacility = facility;
        bestScore = score;
      }
    }
  }
  if (bestFacility != NONE && (MODE == 0u || bestScore > 0.0)) {
    selected[selectedOffset + bestFacility] = 1u;
    status[statusOffset] += 1u;
  } else {
    bestFacility = NONE;
  }
  chosen[chosenOffset] = bestFacility;`
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
  ${output.assignedCosts ? 'assignedCosts[assignedCostsOffset + demand] = assignedCost;' : ''}
  let threshold = ${needsCoverage ? 'coverageDistance[coverageDistanceOffset]' : '0.0'};
  let isUncovered = assigned == NONE || (MODE != 0u && assignedCost > threshold);
  uncovered[uncoveredOffset + demand] = select(0u, 1u, isUncovered);
  let weight = ${demandWeightExpression};
  demandObjectives[demandObjectivesOffset + demand] = select(
    select(0.0, weight, !isUncovered),
    weight * assignedCost,
    MODE == 0u
  );`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-finalize`,
        operation: OPERATION,
        variant: `${props.operation}-finalize`,
        invocationCount: 1,
        bindings: [
          {name: 'demandObjectives', view: demandObjectives, type: 'f32', access: 'read'},
          {name: 'uncovered', view: uncovered, type: 'u32', access: 'read'},
          {name: 'objective', view: output.objective, type: 'f32', access: 'read_write'},
          {name: 'status', view: output.status, type: 'u32', access: 'read_write'}
        ],
        declarations,
        body: `if (status[statusOffset + 3u] != 0u) { return; }
  var uncoveredCount = 0u;
  var objectiveValue = 0.0;
  for (var demand = 0u; demand < DEMAND_COUNT; demand++) {
    uncoveredCount += uncovered[uncoveredOffset + demand];
    objectiveValue += demandObjectives[demandObjectivesOffset + demand];
  }
  let selectedCount = status[statusOffset];
  objective[objectiveOffset] = select(objectiveValue, f32(selectedCount), MODE == 2u);
  status[statusOffset + 1u] = uncoveredCount;
  status[statusOffset + 2u] = select(
    0u,
    1u,
    select(selectedCount == MAXIMUM_FACILITIES, uncoveredCount == 0u, MODE == 2u)
  );`
      })
    );
    return nodes;
  }
}
