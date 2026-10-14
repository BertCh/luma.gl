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
import {
  createWGSLDispatchGate,
  createWGSLKernelNode,
  getAutomaticWGSLKernelWorkgroupSize,
  type WGSLDispatchGate,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  getStorageBindingElementLimit,
  validateCapacity,
  validateMaximumCount,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {type GPUSpatialWeights, validateGPUSpatialWeights} from '../spatial-weights/index';
import {IS_FINITE_FLOAT_WGSL} from '../../utils/wgsl-common';
import {
  createStandardizeColumnsNode,
  createSymmetricAdjacency,
  getRegionalizationSeedParameterValues,
  REGIONALIZATION_SEED_PARAMETER_LENGTH,
  validateRegionalizationParameters,
  getGPURegionalizationMaximumSlotCount,
  validateRegionalizationSlotCount
} from './regionalization-inputs';

export const GPU_WARD_MAXIMUM_COLUMNS = 8;
/**
 * Largest region table of {@link GPUWardRegions} on `device`, in float32 words:
 * `rows * (columnCount + 1)` (each region's size and column sums) in one storage binding, so
 * 33554432 on a default device (128 MiB bindings).
 *
 * @param device Device whose `maxStorageBufferBindingSize` applies.
 */
export function getGPUWardMaximumRegionTableWordCount(device: Device): number {
  return getStorageBindingElementLimit(device, 4);
}
const GPU_WARD_REDUCTION_WORKGROUP_SIZE = 256;

/** Number of `uint32` words of {@link GPUWardRegionsProps.parameters}: `[seed]`. */
export const GPU_WARD_REGIONS_PARAMETER_LENGTH = REGIONALIZATION_SEED_PARAMETER_LENGTH;

/** Per-frame settings of {@link GPUWardRegions}, packed by {@link getGPUWardRegionsParameterValues}. */
export type GPUWardRegionsSettings = {
  /** Reproducible tie-order seed, an integer in `[0, 2^32 - 1]`. Defaults to 0. */
  seed?: number;
};

/**
 * Packs {@link GPUWardRegionsSettings} into the {@link GPU_WARD_REGIONS_PARAMETER_LENGTH}
 * `uint32` words of {@link GPUWardRegionsProps.parameters}.
 */
export function getGPUWardRegionsParameterValues(
  settings: GPUWardRegionsSettings = {}
): Uint32Array {
  return getRegionalizationSeedParameterValues('GPUWardRegions', settings.seed);
}

/** Words of {@link GPUWardRegionsOutput.status}. */
export const GPU_WARD_STATUS = {
  iterationCount: 0,
  converged: 1,
  iterationLimitReached: 2,
  invalidCount: 3,
  length: 4
} as const;

/**
 * Word of {@link GPUWardRegionsOutput.status} that receives the number of merge rounds executed by
 * the `'mutual-nearest'` {@link GPUWardRegionsProps.mergeSchedule}. It is optional: it is written
 * only when `output.status` holds more than {@link GPU_WARD_STATUS}`.length` words, and never by the
 * `'sequential'` schedule (whose rounds are its merges).
 */
export const GPU_WARD_ROUND_COUNT_STATUS_WORD = GPU_WARD_STATUS.length;

/** Merge order of {@link GPUWardRegions}; see {@link GPUWardRegionsProps.mergeSchedule}. */
export type GPUWardMergeSchedule = 'sequential' | 'mutual-nearest';

export type GPUWardRegionsProps = {
  id?: string;
  /**
   * Spatial weights that constrain which regions may merge. Two rows are adjacent when either
   * lists the other, so directed (for example k-nearest-neighbor) weights are symmetrized inside
   * the contributor; weights that declare `information.symmetric === true` are used as they are. At most
   * {@link getGPURegionalizationMaximumSlotCount} `neighbors` slots (16777216 on a default
   * device).
   */
  weights: GPUSpatialWeights;
  /** Row-major feature values. */
  values: GraphDataView<'float32'>;
  columnCount?: number;
  /**
   * Standardize every column to mean 0 and population standard deviation 1 (divisor `n`, the
   * `StandardScaler` convention, over finite entries) before merging, so columns in different
   * units weigh equally. Defaults to true; pass false when `values` is already on a common scale.
   * `output.objective` is then in standardized units.
   */
  standardize?: boolean;
  targetRegionCount: number;
  /**
   * Maximum adjacent-region merges (`'sequential'`) or maximum merge rounds (`'mutual-nearest'`).
   * Must not exceed `rowCount - targetRegionCount`. Every iteration is unrolled at compile time.
   */
  maximumIterations: number;
  /**
   * Compile-time merge order. Defaults to `'sequential'`: one merge per iteration, always the
   * globally cheapest adjacent pair, so `n - k` iterations are needed.
   *
   * `'mutual-nearest'` merges, in one round, every pair of regions that are each other's cheapest
   * adjacent region (reciprocal nearest neighbors, ties by the same seed-keyed total pair order),
   * so a round roughly halves the region count and about `log2(n)` rounds replace `n - k`
   * iterations; `maximumIterations` is then an upper bound on rounds. When a round would pass
   * `targetRegionCount`, only the cheapest pairs in that total order are merged. Ward's linkage
   * is reducible without contiguity constraints (complete adjacency), so the set of merges of the
   * whole dendrogram is the same; but a round also merges expensive pairs before cheaper pairs
   * that only appear after other merges of the round, so stopping at `targetRegionCount` can give
   * a different partition than `'sequential'`. The two agree when the target is a natural cut
   * (every merge below the cut is cheaper than every merge above it, for example well separated
   * clusters), and the result is always a valid partition whose `objective` is its exact
   * within-region sum of squared deviations. With contiguity constraints Ward can also have
   * inversions (a merged region can be cheaper to join than its parts), which makes the two
   * schedules differ further. In this mode `status[iterationCount]` still counts merges applied.
   */
  mergeSchedule?: GPUWardMergeSchedule;
  /**
   * Per-frame `uint32` words ({@link GPU_WARD_REGIONS_PARAMETER_LENGTH}): the tie-order seed,
   * written with {@link getGPUWardRegionsParameterValues}.
   */
  parameters: GraphDataView<'uint32'>;
  /** Caller-owned outputs. */
  output: GPUWardRegionsOutput;
};

/** Caller-owned outputs of {@link GPUWardRegions}. */
export type GPUWardRegionsOutput = {
  /** Region label per row: the smallest row of its region. */
  labels: GraphDataView<'uint32'>;
  /** One value: the within-region sum of squared deviations of `labels`. */
  objective: GraphDataView<'float32'>;
  /**
   * Four words indexed by {@link GPU_WARD_STATUS}; `iterationCount` is the number of merges
   * applied in either schedule. A fifth word ({@link GPU_WARD_ROUND_COUNT_STATUS_WORD}) is
   * written when present and the schedule is `'mutual-nearest'`.
   */
  status: GraphDataView<'uint32'>;
};

/**
 * Spatially constrained Ward agglomeration with an explicit merge budget.
 *
 * `output.objective` is the within-region sum of squared deviations of the returned
 * partition (every row starts as its own region with zero, each merge adds its Ward increase),
 * the same quantity as `withinSumOfSquares` of `GPURegionPartitionEvaluation` on the same table.
 *
 * Default change: columns are standardized (`standardize`, default true); the previous raw-value
 * behavior is `standardize: false`. Adjacency is the undirected union of the weights and their
 * transpose, so a directed link is a link in both directions.
 *
 * Every row begins as its own region. Each iteration merges the adjacent region pair with the
 * smallest Ward increase in within-region squared error. Ties use a stable seed-keyed pair order.
 * With `mergeSchedule: 'mutual-nearest'` each iteration is a round that merges all mutually
 * cheapest pairs at once (see {@link GPUWardRegionsProps.mergeSchedule}).
 * Disconnected weights can prevent the requested region count; `output.status` reports
 * that condition without readback during execution. Once no merge is possible (no adjacent
 * region pair remains, the target is reached, or the input is invalid) the remaining unrolled
 * iterations are gated off on the GPU through indirect dispatches with zero workgroups.
 */
export class GPUWardRegions implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUWardRegionsProps;
  readonly rowCount: number;
  readonly columnCount: number;

  constructor(props: GPUWardRegionsProps) {
    this.id = props.id ?? 'ward-regions';
    this.props = props;
    this.rowCount = validateGPUSpatialWeights(this.id, props.weights);
    this.columnCount = props.columnCount ?? 1;
    const {id, rowCount, columnCount} = this;
    const {output} = props;
    validateCapacity(id, 'columnCount', columnCount, 1, GPU_WARD_MAXIMUM_COLUMNS);
    validateCapacity(id, 'targetRegionCount', props.targetRegionCount, 1, rowCount);
    validateCapacity(
      id,
      'maximumIterations',
      props.maximumIterations,
      0,
      rowCount - props.targetRegionCount
    );
    if (
      props.mergeSchedule !== undefined &&
      props.mergeSchedule !== 'sequential' &&
      props.mergeSchedule !== 'mutual-nearest'
    ) {
      throw new Error(`${id} mergeSchedule must be 'sequential' or 'mutual-nearest'`);
    }
    validateRegionalizationParameters(id, props.parameters, GPU_WARD_REGIONS_PARAMETER_LENGTH);
    validatePackedView(props.values, ['float32'], `${id} values`);
    if (props.values.length !== rowCount * columnCount) {
      throw new Error(`${id} values length must equal rows * columnCount`);
    }
    validatePackedView(output.labels, ['uint32'], `${id} output.labels`);
    if (output.labels.length !== rowCount) {
      throw new Error(`${id} output.labels length must equal the weights row count`);
    }
    validatePackedView(output.objective, ['float32'], `${id} output.objective`);
    validatePackedView(output.status, ['uint32'], `${id} output.status`);
    if (output.objective.length < 1) {
      throw new Error(`${id} output.objective must hold one value`);
    }
    if (output.status.length < GPU_WARD_STATUS.length) {
      throw new Error(`${id} output.status must hold ${GPU_WARD_STATUS.length} words`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.labels, output.objective, output.status],
      [
        props.weights.offsets,
        props.weights.neighbors,
        props.weights.weights,
        props.values,
        props.parameters
      ]
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rowCount, columnCount} = this;
    const {output, parameters} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.weights.offsets,
      props.weights.neighbors,
      props.weights.weights,
      props.values,
      parameters,
      output.labels,
      output.objective,
      output.status
    ]);
    validateRegionalizationSlotCount(
      id,
      props.weights,
      getGPURegionalizationMaximumSlotCount(graph.device)
    );
    validateMaximumCount(
      id,
      'rows x (columnCount + 1)',
      rowCount * (columnCount + 1),
      getGPUWardMaximumRegionTableWordCount(graph.device)
    );
    const regionStride = columnCount + 1;
    const regions = createTransientView(graph, `${id}-regions`, 'float32', rowCount * regionStride);
    const mutual = props.mergeSchedule === 'mutual-nearest';
    const hasRoundCountWord = output.status.length > GPU_WARD_ROUND_COUNT_STATUS_WORD;
    // Sequential: [regions left]. Mutual-nearest: [regions left, rounds, exhausted, round merges,
    // round cost bits].
    const state = createTransientView(graph, `${id}-state`, 'uint32', mutual ? 5 : 1);
    const standardize = props.standardize !== false;
    const table = standardize
      ? createTransientView(graph, `${id}-table`, 'float32', rowCount * columnCount)
      : props.values;
    const adjacency = createSymmetricAdjacency(graph, {
      id,
      operation: 'GPUWardRegions',
      weights: props.weights,
      rowCount
    });
    const mutualBuffers = mutual
      ? {
          rowCost: createTransientView(graph, `${id}-row-cost`, 'uint32', rowCount),
          rowOther: createTransientView(graph, `${id}-row-other`, 'uint32', rowCount),
          regionCost: createTransientView(graph, `${id}-region-cost`, 'uint32', rowCount),
          regionKey: createTransientView(graph, `${id}-region-key`, 'uint32', rowCount),
          regionOther: createTransientView(graph, `${id}-region-other`, 'uint32', rowCount),
          mergeFlag: createTransientView(graph, `${id}-merge-flag`, 'uint32', rowCount)
        }
      : null;
    const candidateCostA = createTransientView(
      graph,
      `${id}-candidate-cost-a`,
      'float32',
      rowCount
    );
    const candidateCostB = createTransientView(
      graph,
      `${id}-candidate-cost-b`,
      'float32',
      rowCount
    );
    const candidateFirstA = createTransientView(
      graph,
      `${id}-candidate-first-a`,
      'uint32',
      rowCount
    );
    const candidateFirstB = createTransientView(
      graph,
      `${id}-candidate-first-b`,
      'uint32',
      rowCount
    );
    const candidateSecondA = createTransientView(
      graph,
      `${id}-candidate-second-a`,
      'uint32',
      rowCount
    );
    const candidateSecondB = createTransientView(
      graph,
      `${id}-candidate-second-b`,
      'uint32',
      rowCount
    );
    const candidateKeyA = createTransientView(graph, `${id}-candidate-key-a`, 'uint32', rowCount);
    const candidateKeyB = createTransientView(graph, `${id}-candidate-key-b`, 'uint32', rowCount);
    // Workgroup-tree argmin stages over the (cost, key, first, second) total order. Each stage
    // shrinks the candidate list by the workgroup size instead of by two, so a 100k-row problem
    // needs three dispatches per merge instead of seventeen. The plan is the same every merge.
    const reductionStages: {activeCount: number; workgroupSize: number; nextCount: number}[] = [];
    for (let activeCount = mutual ? 1 : rowCount; activeCount > 1; ) {
      const workgroupSize = Math.min(
        GPU_WARD_REDUCTION_WORKGROUP_SIZE,
        Math.max(2, 2 ** Math.ceil(Math.log2(activeCount)))
      );
      const nextCount = Math.ceil(activeCount / workgroupSize);
      reductionStages.push({activeCount, workgroupSize, nextCount});
      activeCount = nextCount;
    }
    // Indirect dispatch triples of the gated per-merge kernels: slot 0 for the row-parallel
    // candidates and relabel kernels, slot 1 + s for reduction stage s. The apply kernel zeroes
    // every x once no further merge is possible, so the remaining iterations launch nothing. The
    // apply kernel itself stays ungated: one buffer must not be both an indirect argument and
    // storage-written in one dispatch.
    const rowWorkgroupSize = getAutomaticWGSLKernelWorkgroupSize(rowCount);
    const gate =
      props.maximumIterations > 0
        ? createWGSLDispatchGate(graph, {
            id: `${id}-merge-dispatch`,
            operation: 'GPUWardRegions',
            shapes: [
              [rowCount, rowWorkgroupSize],
              // Slot 1: the one-workgroup cooperative kernels of the mutual-nearest rounds.
              ...(mutual
                ? [[GPU_WARD_REDUCTION_WORKGROUP_SIZE, GPU_WARD_REDUCTION_WORKGROUP_SIZE] as const]
                : []),
              ...reductionStages.map(stage => [stage.activeCount, stage.workgroupSize] as const)
            ]
          })
        : null;
    const declarations = `const ROW_COUNT: u32 = ${rowCount}u;
const COLUMN_COUNT: u32 = ${columnCount}u;
const REGION_STRIDE: u32 = ${regionStride}u;
const TARGET_REGION_COUNT: u32 = ${props.targetRegionCount}u;
const NONE: u32 = 0xffffffffu;
const LARGE_INCREASE: f32 = 3.402823466e+38;
${IS_FINITE_FLOAT_WGSL}
fn regionCount(region: u32) -> f32 { return regions[regionsOffset + region * REGION_STRIDE]; }
fn regionSum(region: u32, column: u32) -> f32 { return regions[regionsOffset + region * REGION_STRIDE + 1u + column]; }`;
    // Stable seed-keyed pair order. The seed is the per-frame parameter word, so a new seed is a
    // buffer write; only kernels that bind `parameters` include this function.
    const pairKeyWGSL = `fn pairKey(first: u32, second: u32) -> u32 {
  var value = (first * 0x9e3779b9u) ^ second ^ parameters[parametersOffset];
  value = (value ^ (value >> 16u)) * 0x7feb352du;
  value = (value ^ (value >> 15u)) * 0x846ca68bu;
  return value ^ (value >> 16u);
}`;
    const parametersBinding: WGSLKernelBinding = {
      name: 'parameters',
      view: parameters,
      type: 'u32',
      access: 'read'
    };
    const nodes: GPUCommandNode<Parameters>[] = [
      ...adjacency.nodes,
      ...(standardize
        ? [
            createStandardizeColumnsNode<Parameters>(graph, {
              id: `${id}-standardize`,
              operation: 'GPUWardRegions',
              values: props.values,
              scaled: table,
              rowCount,
              columnCount
            })
          ]
        : []),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initialize-status`,
        operation: 'GPUWardRegions',
        variant: 'initialize-status',
        invocationCount: 1,
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read_write'},
          {
            name: 'objective',
            view: output.objective,
            type: 'f32',
            access: 'read_write'
          },
          {name: 'status', view: output.status, type: 'u32', access: 'read_write'},
          ...(gate ? [gate.binding] : [])
        ],
        body: `state[stateOffset] = ${rowCount}u;
  ${mutual ? 'for (var word = 1u; word < 5u; word++) { state[stateOffset + word] = 0u; }' : ''}
  objective[objectiveOffset] = 0.0;
  status[statusOffset] = 0u;
  status[statusOffset + 1u] = 0u;
  status[statusOffset + 2u] = 0u;
  status[statusOffset + 3u] = 0u;
  ${mutual && hasRoundCountWord ? `status[statusOffset + ${GPU_WARD_ROUND_COUNT_STATUS_WORD}u] = 0u;` : ''}
  ${gate ? gate.armWGSL : ''}`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initialize-regions`,
        operation: 'GPUWardRegions',
        variant: 'initialize-regions',
        invocationCount: rowCount,
        bindings: [
          {name: 'offsets', view: props.weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: props.weights.neighbors, type: 'u32', access: 'read'},
          {name: 'values', view: props.values, type: 'f32', access: 'read'},
          {name: 'table', view: table, type: 'f32', access: 'read'},
          {name: 'labels', view: output.labels, type: 'u32', access: 'read_write'},
          {name: 'regions', view: regions, type: 'f32', access: 'read_write'},
          {
            name: 'status',
            view: output.status,
            type: 'atomic<u32>',
            access: 'read_write'
          }
        ],
        declarations,
        body: `labels[labelsOffset + index] = index;
  let regionBase = regionsOffset + index * REGION_STRIDE;
  regions[regionBase] = 1.0;
  var invalidCount = 0u;
  for (var column = 0u; column < COLUMN_COUNT; column++) {
    invalidCount += select(0u, 1u, !isFiniteFloat(values[valuesOffset + index * COLUMN_COUNT + column]));
    regions[regionBase + 1u + column] = table[tableOffset + index * COLUMN_COUNT + column];
  }
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    invalidCount += select(0u, 1u, neighbors[neighborsOffset + slot] >= ROW_COUNT);
  }
  if (invalidCount > 0u) { atomicAdd(&status[statusOffset + 3u], invalidCount); }`
      })
    ];

    for (let iteration = 0; iteration < props.maximumIterations; iteration++) {
      if (mutualBuffers) {
        nodes.push(
          ...this.createMutualNearestRoundNodes(graph, {
            iteration,
            buffers: mutualBuffers,
            adjacency: adjacency.weights,
            regions,
            state,
            gate: gate!,
            rowWorkgroupSize,
            declarations,
            pairKeyWGSL,
            parametersBinding
          })
        );
        continue;
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-iteration-${iteration}-candidates`,
          operation: 'GPUWardRegions',
          variant: 'adjacent-candidates',
          invocationCount: rowCount,
          workgroupSize: rowWorkgroupSize,
          ...gate!.getGate(0),
          bindings: [
            {name: 'offsets', view: adjacency.weights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: adjacency.weights.neighbors, type: 'u32', access: 'read'},
            {name: 'labels', view: output.labels, type: 'u32', access: 'read'},
            {name: 'regions', view: regions, type: 'f32', access: 'read'},
            {name: 'costOut', view: candidateCostA, type: 'f32', access: 'read_write'},
            {name: 'firstOut', view: candidateFirstA, type: 'u32', access: 'read_write'},
            {name: 'secondOut', view: candidateSecondA, type: 'u32', access: 'read_write'},
            parametersBinding
          ],
          declarations: `${declarations}
${pairKeyWGSL}`,
          // The gate guarantees more than TARGET_REGION_COUNT regions remain: it is armed while
          // state > target and disarmed by the merge that reaches the target.
          body: `var bestFirst = NONE;
  var bestSecond = NONE;
  var bestIncrease = LARGE_INCREASE;
  var bestKey = NONE;
  {
    let rowRegion = labels[labelsOffset + index];
    for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor >= ROW_COUNT) { continue; }
      let neighborRegion = labels[labelsOffset + neighbor];
      let first = min(rowRegion, neighborRegion);
      let second = max(rowRegion, neighborRegion);
      if (first != rowRegion || first == second) { continue; }
      let firstCount = regionCount(first);
      let secondCount = regionCount(second);
      var increase = 0.0;
      for (var column = 0u; column < COLUMN_COUNT; column++) {
        let difference = regionSum(first, column) / firstCount - regionSum(second, column) / secondCount;
        increase += firstCount * secondCount / (firstCount + secondCount) * difference * difference;
      }
      let key = pairKey(first, second);
      if (increase < bestIncrease || (increase == bestIncrease &&
          (key < bestKey || (key == bestKey && (first < bestFirst || (first == bestFirst && second < bestSecond)))))) {
        bestFirst = first; bestSecond = second; bestIncrease = increase; bestKey = key;
      }
    }
  }
  costOut[costOutOffset + index] = bestIncrease;
  firstOut[firstOutOffset + index] = bestFirst;
  secondOut[secondOutOffset + index] = bestSecond;`
        })
      );

      // The order is total, so the winner does not depend on the tree shape. The first stage
      // derives pair keys from the candidates, which replaces a separate key kernel.
      let sourceCost = candidateCostA;
      let sourceFirst = candidateFirstA;
      let sourceSecond = candidateSecondA;
      let sourceKey: GraphDataView<'uint32'> | null = null;
      let destinationCost = candidateCostB;
      let destinationFirst = candidateFirstB;
      let destinationSecond = candidateSecondB;
      let destinationKey = candidateKeyB;
      let spareKey = candidateKeyA;
      for (const [
        reduction,
        {activeCount, workgroupSize, nextCount}
      ] of reductionStages.entries()) {
        const keyExpression = sourceKey
          ? 'keyIn[keyInOffset + index]'
          : 'select(pairKey(first, second), NONE, first == NONE)';
        // The first stage derives keys (and reads the seed); later stages read the key buffer.
        const keyBindings: WGSLKernelBinding[] = sourceKey
          ? [{name: 'keyIn', view: sourceKey, type: 'u32', access: 'read'}]
          : [parametersBinding];
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-iteration-${iteration}-reduce-${reduction}`,
            operation: 'GPUWardRegions',
            variant: 'workgroup-reduction',
            invocationCount: activeCount,
            workgroupSize,
            guardIndex: false,
            ...gate!.getGate(1 + reduction),
            bindings: [
              {name: 'costIn', view: sourceCost, type: 'f32', access: 'read'},
              {name: 'firstIn', view: sourceFirst, type: 'u32', access: 'read'},
              {name: 'secondIn', view: sourceSecond, type: 'u32', access: 'read'},
              ...keyBindings,
              {name: 'costOut', view: destinationCost, type: 'f32', access: 'read_write'},
              {name: 'firstOut', view: destinationFirst, type: 'u32', access: 'read_write'},
              {name: 'secondOut', view: destinationSecond, type: 'u32', access: 'read_write'},
              {name: 'keyOut', view: destinationKey, type: 'u32', access: 'read_write'}
            ],
            declarations: `const INPUT_COUNT: u32 = ${activeCount}u;
const OUTPUT_COUNT: u32 = ${nextCount}u;
const WORKGROUP_SIZE: u32 = ${workgroupSize}u;
const NONE: u32 = 0xffffffffu;
const LARGE_INCREASE: f32 = 3.402823466e+38;
var<workgroup> sharedCost: array<f32, ${workgroupSize}>;
var<workgroup> sharedKey: array<u32, ${workgroupSize}>;
var<workgroup> sharedFirst: array<u32, ${workgroupSize}>;
var<workgroup> sharedSecond: array<u32, ${workgroupSize}>;
${sourceKey ? '' : pairKeyWGSL}
fn isBetter(slot: u32, other: u32) -> bool {
  let cost = sharedCost[slot];
  let otherCost = sharedCost[other];
  if (cost != otherCost) { return cost < otherCost; }
  let key = sharedKey[slot];
  let otherKey = sharedKey[other];
  if (key != otherKey) { return key < otherKey; }
  let first = sharedFirst[slot];
  let otherFirst = sharedFirst[other];
  if (first != otherFirst) { return first < otherFirst; }
  return sharedSecond[slot] < sharedSecond[other];
}`,
            // No early return: every invocation of a workgroup must reach the barriers.
            body: `var cost = LARGE_INCREASE;
  var first = NONE;
  var second = NONE;
  var key = NONE;
  if (index < INPUT_COUNT) {
    cost = costIn[costInOffset + index];
    first = firstIn[firstInOffset + index];
    second = secondIn[secondInOffset + index];
    key = ${keyExpression};
  }
  sharedCost[localInvocationIndex] = cost;
  sharedKey[localInvocationIndex] = key;
  sharedFirst[localInvocationIndex] = first;
  sharedSecond[localInvocationIndex] = second;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
    if (localInvocationIndex < stride && isBetter(localInvocationIndex + stride, localInvocationIndex)) {
      let winner = localInvocationIndex + stride;
      sharedCost[localInvocationIndex] = sharedCost[winner];
      sharedKey[localInvocationIndex] = sharedKey[winner];
      sharedFirst[localInvocationIndex] = sharedFirst[winner];
      sharedSecond[localInvocationIndex] = sharedSecond[winner];
    }
    workgroupBarrier();
  }
  let group = index / WORKGROUP_SIZE;
  if (localInvocationIndex == 0u && group < OUTPUT_COUNT) {
    costOut[costOutOffset + group] = sharedCost[0];
    firstOut[firstOutOffset + group] = sharedFirst[0];
    secondOut[secondOutOffset + group] = sharedSecond[0];
    keyOut[keyOutOffset + group] = sharedKey[0];
  }`
          })
        );
        // Ping-pong: the stage output becomes the next input; the old key buffer is reused.
        const nextKey = destinationKey;
        [sourceCost, destinationCost] = [destinationCost, sourceCost];
        [sourceFirst, destinationFirst] = [destinationFirst, sourceFirst];
        [sourceSecond, destinationSecond] = [destinationSecond, sourceSecond];
        sourceKey = nextKey;
        destinationKey = spareKey;
        spareKey = nextKey;
      }

      // Relabel runs before apply: it only needs the winning pair, and apply may disarm the
      // gate when this merge reaches the target.
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-iteration-${iteration}-relabel`,
          operation: 'GPUWardRegions',
          variant: 'parallel-relabel',
          invocationCount: rowCount,
          workgroupSize: rowWorkgroupSize,
          ...gate!.getGate(0),
          bindings: [
            {name: 'first', view: sourceFirst, type: 'u32', access: 'read'},
            {name: 'second', view: sourceSecond, type: 'u32', access: 'read'},
            {name: 'labels', view: output.labels, type: 'u32', access: 'read_write'},
            {name: 'status', view: output.status, type: 'u32', access: 'read'}
          ],
          declarations: 'const NONE: u32 = 0xffffffffu;',
          body: `let firstRegion = first[firstOffset];
  let secondRegion = second[secondOffset];
  if (status[statusOffset + 3u] == 0u && firstRegion != NONE && labels[labelsOffset + index] == secondRegion) {
    labels[labelsOffset + index] = firstRegion;
  }`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-iteration-${iteration}-apply`,
          operation: 'GPUWardRegions',
          variant: 'apply-merge',
          invocationCount: 1,
          bindings: [
            {name: 'cost', view: sourceCost, type: 'f32', access: 'read'},
            {name: 'first', view: sourceFirst, type: 'u32', access: 'read'},
            {name: 'second', view: sourceSecond, type: 'u32', access: 'read'},
            {name: 'regions', view: regions, type: 'f32', access: 'read_write'},
            {name: 'state', view: state, type: 'u32', access: 'read_write'},
            {
              name: 'objective',
              view: output.objective,
              type: 'f32',
              access: 'read_write'
            },
            {name: 'status', view: output.status, type: 'u32', access: 'read_write'},
            gate!.binding
          ],
          declarations: `const COLUMN_COUNT: u32 = ${columnCount}u;
const REGION_STRIDE: u32 = ${regionStride}u;
const TARGET_REGION_COUNT: u32 = ${props.targetRegionCount}u;
const NONE: u32 = 0xffffffffu;`,
          // A zero x in slot 0 means an earlier iteration disarmed the gate; the candidate buffers
          // then hold a stale winner that must not be applied again.
          body: `if (dispatch[dispatchOffset] == 0u) { return; }
  let firstRegion = first[firstOffset];
  let secondRegion = second[secondOffset];
  if (firstRegion != NONE && state[stateOffset] > TARGET_REGION_COUNT && status[statusOffset + 3u] == 0u) {
    let firstBase = regionsOffset + firstRegion * REGION_STRIDE;
    let secondBase = regionsOffset + secondRegion * REGION_STRIDE;
    regions[firstBase] += regions[secondBase];
    regions[secondBase] = 0.0;
    for (var column = 0u; column < COLUMN_COUNT; column++) {
      regions[firstBase + 1u + column] += regions[secondBase + 1u + column];
      regions[secondBase + 1u + column] = 0.0;
    }
    objective[objectiveOffset] += cost[costOffset];
    state[stateOffset] -= 1u;
    status[statusOffset] += 1u;
  }
  // No adjacent pair, the target reached, or invalid input: no later merge can happen.
  if (firstRegion == NONE || state[stateOffset] <= TARGET_REGION_COUNT || status[statusOffset + 3u] != 0u) {
    ${gate!.disarmWGSL}
  }`
        })
      );
    }

    if (mutualBuffers) {
      nodes.push(this.createMutualNearestFinalizeNode(graph, state));
      return nodes;
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finalize`,
        operation: 'GPUWardRegions',
        variant: 'finalize',
        invocationCount: 1,
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'status', view: output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: `const TARGET_REGION_COUNT: u32 = ${props.targetRegionCount}u;
const MAXIMUM_ITERATIONS: u32 = ${props.maximumIterations}u;`,
        body: `let remaining = state[stateOffset];
  let completed = status[statusOffset];
  status[statusOffset + 1u] = select(0u, 1u, remaining == TARGET_REGION_COUNT && status[statusOffset + 3u] == 0u);
  status[statusOffset + 2u] = select(0u, 1u, remaining > TARGET_REGION_COUNT && completed == MAXIMUM_ITERATIONS);`
      })
    );
    return nodes;
  }

  /**
   * One round of the `'mutual-nearest'` schedule. Every region finds its cheapest adjacent region
   * with three `atomicMin` passes over rows (cost bits, pair key, region id: together the same
   * total order as the sequential candidate order), reciprocal pairs are selected within the
   * remaining merge budget, and the selected pairs merge at once. The pairs are disjoint, so no
   * pointer jumping is needed. The round index is never in the WGSL: the rounds are identical and
   * the counter lives in the `state` word.
   */
  private createMutualNearestRoundNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    round: {
      iteration: number;
      buffers: Record<
        'rowCost' | 'rowOther' | 'regionCost' | 'regionKey' | 'regionOther' | 'mergeFlag',
        GraphDataView<'uint32'>
      >;
      adjacency: GPUSpatialWeights;
      regions: GraphDataView<'float32'>;
      state: GraphDataView<'uint32'>;
      gate: WGSLDispatchGate<Parameters>;
      rowWorkgroupSize: number;
      declarations: string;
      pairKeyWGSL: string;
      parametersBinding: WGSLKernelBinding;
    }
  ): GPUCommandNode<Parameters>[] {
    const {id, props, rowCount, columnCount} = this;
    const {output} = props;
    const {iteration, buffers, adjacency, regions, state, gate, rowWorkgroupSize} = round;
    const {declarations, pairKeyWGSL, parametersBinding} = round;
    const prefix = `${id}-round-${iteration}`;
    const rowKernel = {
      operation: 'GPUWardRegions',
      invocationCount: rowCount,
      workgroupSize: rowWorkgroupSize,
      ...gate.getGate(0)
    };
    const cooperativeKernel = {
      operation: 'GPUWardRegions',
      invocationCount: GPU_WARD_REDUCTION_WORKGROUP_SIZE,
      workgroupSize: GPU_WARD_REDUCTION_WORKGROUP_SIZE,
      guardIndex: false,
      ...gate.getGate(1)
    };
    const noneDeclaration = 'const NONE: u32 = 0xffffffffu;';
    const rowsDeclaration = `${noneDeclaration}
const ROW_COUNT: u32 = ${rowCount}u;`;
    const labelsRead: WGSLKernelBinding = {
      name: 'labels',
      view: output.labels,
      type: 'u32',
      access: 'read'
    };
    const rowCostRead: WGSLKernelBinding = {
      name: 'rowCost',
      view: buffers.rowCost,
      type: 'u32',
      access: 'read'
    };
    const rowOtherRead: WGSLKernelBinding = {
      name: 'rowOther',
      view: buffers.rowOther,
      type: 'u32',
      access: 'read'
    };
    const regionCostRead: WGSLKernelBinding = {
      name: 'regionCost',
      view: buffers.regionCost,
      type: 'u32',
      access: 'read'
    };
    const regionKeyRead: WGSLKernelBinding = {
      name: 'regionKey',
      view: buffers.regionKey,
      type: 'u32',
      access: 'read'
    };
    const mergeFlagBinding = (access: 'read' | 'read_write'): WGSLKernelBinding => ({
      name: 'mergeFlag',
      view: buffers.mergeFlag,
      type: 'u32',
      access
    });
    const regionOtherBinding = (access: 'read' | 'read_write'): WGSLKernelBinding => ({
      name: 'regionOther',
      view: buffers.regionOther,
      type: 'u32',
      access
    });
    const hasRoundCountWord = output.status.length > GPU_WARD_ROUND_COUNT_STATUS_WORD;

    return [
      createWGSLKernelNode<Parameters>(graph, {
        ...rowKernel,
        id: `${prefix}-reset-nearest`,
        variant: 'mutual-reset-nearest',
        bindings: [
          {name: 'regionCost', view: buffers.regionCost, type: 'u32', access: 'read_write'},
          {name: 'regionKey', view: buffers.regionKey, type: 'u32', access: 'read_write'},
          regionOtherBinding('read_write')
        ],
        declarations: noneDeclaration,
        body: `regionCost[regionCostOffset + index] = NONE;
  regionKey[regionKeyOffset + index] = NONE;
  regionOther[regionOtherOffset + index] = NONE;`
      }),
      // The row's cheapest adjacent region over the (increase, pair key, region id) total order.
      // Costs are non-negative, so their float32 bit patterns order like the floats and let the
      // region minima use integer atomics.
      createWGSLKernelNode<Parameters>(graph, {
        ...rowKernel,
        id: `${prefix}-nearest-candidates`,
        variant: 'mutual-nearest-candidates',
        bindings: [
          {name: 'offsets', view: adjacency.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: adjacency.neighbors, type: 'u32', access: 'read'},
          labelsRead,
          {name: 'regions', view: regions, type: 'f32', access: 'read'},
          {name: 'rowCost', view: buffers.rowCost, type: 'u32', access: 'read_write'},
          {name: 'rowOther', view: buffers.rowOther, type: 'u32', access: 'read_write'},
          parametersBinding
        ],
        declarations: `${declarations}
${pairKeyWGSL}
fn getIncrease(first: u32, second: u32) -> f32 {
  let firstCount = regionCount(first);
  let secondCount = regionCount(second);
  var increase = 0.0;
  for (var column = 0u; column < COLUMN_COUNT; column++) {
    let difference = regionSum(first, column) / firstCount - regionSum(second, column) / secondCount;
    increase += firstCount * secondCount / (firstCount + secondCount) * difference * difference;
  }
  return increase;
}`,
        body: `let region = labels[labelsOffset + index];
  var bestCost = NONE;
  var bestKey = NONE;
  var bestOther = NONE;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= ROW_COUNT) { continue; }
    let other = labels[labelsOffset + neighbor];
    if (other == region) { continue; }
    let first = min(region, other);
    let second = max(region, other);
    let cost = bitcast<u32>(max(getIncrease(first, second), 0.0));
    let key = pairKey(first, second);
    if (cost < bestCost || (cost == bestCost &&
        (key < bestKey || (key == bestKey && other < bestOther)))) {
      bestCost = cost; bestKey = key; bestOther = other;
    }
  }
  rowCost[rowCostOffset + index] = bestCost;
  rowOther[rowOtherOffset + index] = bestOther;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        ...rowKernel,
        id: `${prefix}-nearest-cost`,
        variant: 'mutual-nearest-cost',
        bindings: [
          labelsRead,
          rowCostRead,
          {name: 'regionCost', view: buffers.regionCost, type: 'atomic<u32>', access: 'read_write'}
        ],
        declarations: noneDeclaration,
        body: `let cost = rowCost[rowCostOffset + index];
  if (cost != NONE) { atomicMin(&regionCost[regionCostOffset + labels[labelsOffset + index]], cost); }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        ...rowKernel,
        id: `${prefix}-nearest-key`,
        variant: 'mutual-nearest-key',
        bindings: [
          labelsRead,
          rowCostRead,
          rowOtherRead,
          regionCostRead,
          {name: 'regionKey', view: buffers.regionKey, type: 'atomic<u32>', access: 'read_write'},
          parametersBinding
        ],
        declarations: `${noneDeclaration}
${pairKeyWGSL}`,
        body: `let cost = rowCost[rowCostOffset + index];
  let region = labels[labelsOffset + index];
  if (cost != NONE && cost == regionCost[regionCostOffset + region]) {
    let other = rowOther[rowOtherOffset + index];
    atomicMin(&regionKey[regionKeyOffset + region], pairKey(min(region, other), max(region, other)));
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        ...rowKernel,
        id: `${prefix}-nearest-region`,
        variant: 'mutual-nearest-region',
        bindings: [
          labelsRead,
          rowCostRead,
          rowOtherRead,
          regionCostRead,
          regionKeyRead,
          {
            name: 'regionOther',
            view: buffers.regionOther,
            type: 'atomic<u32>',
            access: 'read_write'
          },
          parametersBinding
        ],
        declarations: `${noneDeclaration}
${pairKeyWGSL}`,
        body: `let cost = rowCost[rowCostOffset + index];
  let region = labels[labelsOffset + index];
  let other = rowOther[rowOtherOffset + index];
  if (cost != NONE && cost == regionCost[regionCostOffset + region] &&
      pairKey(min(region, other), max(region, other)) == regionKey[regionKeyOffset + region]) {
    atomicMin(&regionOther[regionOtherOffset + region], other);
  }`
      }),
      // One workgroup: counts the reciprocal pairs, and when they exceed the merge budget selects
      // the cheapest `budget` of them by a bitwise radix select over the (cost bits, pair key,
      // first region) tuple, which is a total order on disjoint pairs.
      createWGSLKernelNode<Parameters>(graph, {
        ...cooperativeKernel,
        id: `${prefix}-select-pairs`,
        variant: 'mutual-select-pairs',
        bindings: [
          regionCostRead,
          regionKeyRead,
          regionOtherBinding('read'),
          mergeFlagBinding('read_write'),
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'status', view: output.status, type: 'u32', access: 'read'}
        ],
        declarations: `${rowsDeclaration}
const TARGET_REGION_COUNT: u32 = ${props.targetRegionCount}u;
const WORKGROUP_SIZE: u32 = ${GPU_WARD_REDUCTION_WORKGROUP_SIZE}u;
var<workgroup> sharedCount: atomic<u32>;
var<workgroup> uniformValue: u32;
// First region of a reciprocal pair: it and its choice name each other.
fn isMutual(region: u32) -> bool {
  let other = regionOther[regionOtherOffset + region];
  return other != NONE && other > region && other < ROW_COUNT &&
    regionOther[regionOtherOffset + other] == region;
}
fn getWord(region: u32, word: u32) -> u32 {
  if (word == 0u) { return regionCost[regionCostOffset + region]; }
  if (word == 1u) { return regionKey[regionKeyOffset + region]; }
  return region;
}
// Sums the lanes' counts through the shared atomic and returns the total to every lane.
fn reduceCount(lane: u32, laneCount: u32) -> u32 {
  if (lane == 0u) { atomicStore(&sharedCount, 0u); }
  workgroupBarrier();
  atomicAdd(&sharedCount, laneCount);
  workgroupBarrier();
  if (lane == 0u) { uniformValue = atomicLoad(&sharedCount); }
  return workgroupUniformLoad(&uniformValue);
}`,
        // Every lane reaches every barrier: all branch conditions come from workgroupUniformLoad.
        body: `let lane = localInvocationIndex;
  var laneMutual = 0u;
  for (var region = lane; region < ROW_COUNT; region += WORKGROUP_SIZE) {
    if (isMutual(region)) { laneMutual += 1u; }
  }
  let mutualCount = reduceCount(lane, laneMutual);
  // A merge is allowed only down to the target, and never on invalid input.
  let regionsLeft = state[stateOffset];
  var budget = 0u;
  if (regionsLeft > TARGET_REGION_COUNT && status[statusOffset + 3u] == 0u) {
    budget = regionsLeft - TARGET_REGION_COUNT;
  }
  if (lane == 0u) { uniformValue = budget; }
  let uniformBudget = workgroupUniformLoad(&uniformValue);
  var remaining = uniformBudget;
  var threshold0 = 0u;
  var threshold1 = 0u;
  var threshold2 = 0u;
  if (uniformBudget > 0u && mutualCount > uniformBudget) {
    for (var wordIndex = 0u; wordIndex < 3u; wordIndex++) {
      var prefix = 0u;
      for (var bit = 32u; bit > 0u; bit--) {
        let shift = bit - 1u;
        var laneZero = 0u;
        for (var region = lane; region < ROW_COUNT; region += WORKGROUP_SIZE) {
          if (!isMutual(region)) { continue; }
          if (wordIndex > 0u && getWord(region, 0u) != threshold0) { continue; }
          if (wordIndex > 1u && getWord(region, 1u) != threshold1) { continue; }
          let word = getWord(region, wordIndex);
          if ((((word ^ prefix) >> shift) >> 1u) != 0u) { continue; }
          if (((word >> shift) & 1u) == 0u) { laneZero += 1u; }
        }
        let zeroCount = reduceCount(lane, laneZero);
        if (zeroCount < remaining) {
          remaining -= zeroCount;
          prefix |= 1u << shift;
        }
      }
      if (wordIndex == 0u) { threshold0 = prefix; }
      else if (wordIndex == 1u) { threshold1 = prefix; }
      else { threshold2 = prefix; }
    }
  }
  for (var region = lane; region < ROW_COUNT; region += WORKGROUP_SIZE) {
    var accepted = isMutual(region) && uniformBudget > 0u;
    if (accepted && mutualCount > uniformBudget) {
      let word0 = getWord(region, 0u);
      let word1 = getWord(region, 1u);
      accepted = word0 < threshold0 || (word0 == threshold0 &&
        (word1 < threshold1 || (word1 == threshold1 && region <= threshold2)));
    }
    mergeFlag[mergeFlagOffset + region] = select(0u, 1u, accepted);
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        ...rowKernel,
        id: `${prefix}-merge-regions`,
        variant: 'mutual-merge-regions',
        bindings: [
          {name: 'regions', view: regions, type: 'f32', access: 'read_write'},
          regionOtherBinding('read'),
          mergeFlagBinding('read')
        ],
        declarations: `const REGION_STRIDE: u32 = ${columnCount + 1}u;`,
        // The accepted pairs are disjoint, so no thread writes a row another thread reads.
        body: `if (mergeFlag[mergeFlagOffset + index] == 0u) { return; }
  let firstBase = regionsOffset + index * REGION_STRIDE;
  let secondBase = regionsOffset + regionOther[regionOtherOffset + index] * REGION_STRIDE;
  for (var word = 0u; word < REGION_STRIDE; word++) {
    regions[firstBase + word] += regions[secondBase + word];
  }`
      }),
      // Rows of a merged-away region take the first region's label, and that region's moments
      // are cleared. Index `i` plays both the row and the region of that id.
      createWGSLKernelNode<Parameters>(graph, {
        ...rowKernel,
        id: `${prefix}-relabel`,
        variant: 'mutual-relabel',
        bindings: [
          {name: 'labels', view: output.labels, type: 'u32', access: 'read_write'},
          {name: 'regions', view: regions, type: 'f32', access: 'read_write'},
          regionOtherBinding('read'),
          mergeFlagBinding('read')
        ],
        declarations: `${noneDeclaration}
const REGION_STRIDE: u32 = ${columnCount + 1}u;
// First region of the pair whose second region is \`region\`, or NONE.
fn getMergedInto(region: u32) -> u32 {
  let partner = regionOther[regionOtherOffset + region];
  if (partner != NONE && partner < region && mergeFlag[mergeFlagOffset + partner] != 0u &&
      regionOther[regionOtherOffset + partner] == region) {
    return partner;
  }
  return NONE;
}`,
        body: `let rowTarget = getMergedInto(labels[labelsOffset + index]);
  if (rowTarget != NONE) { labels[labelsOffset + index] = rowTarget; }
  if (getMergedInto(index) != NONE) {
    for (var word = 0u; word < REGION_STRIDE; word++) {
      regions[regionsOffset + index * REGION_STRIDE + word] = 0.0;
    }
  }`
      }),
      // One workgroup: the round's merge count and cost, summed in a fixed lane and tree order.
      createWGSLKernelNode<Parameters>(graph, {
        ...cooperativeKernel,
        id: `${prefix}-sum-merges`,
        variant: 'mutual-sum-merges',
        bindings: [
          regionCostRead,
          mergeFlagBinding('read'),
          {name: 'state', view: state, type: 'u32', access: 'read_write'}
        ],
        declarations: `const ROW_COUNT: u32 = ${rowCount}u;
const WORKGROUP_SIZE: u32 = ${GPU_WARD_REDUCTION_WORKGROUP_SIZE}u;
var<workgroup> sharedCost: array<f32, ${GPU_WARD_REDUCTION_WORKGROUP_SIZE}>;
var<workgroup> sharedCount: array<u32, ${GPU_WARD_REDUCTION_WORKGROUP_SIZE}>;`,
        body: `let lane = localInvocationIndex;
  var cost = 0.0;
  var count = 0u;
  for (var region = lane; region < ROW_COUNT; region += WORKGROUP_SIZE) {
    if (mergeFlag[mergeFlagOffset + region] != 0u) {
      cost += bitcast<f32>(regionCost[regionCostOffset + region]);
      count += 1u;
    }
  }
  sharedCost[lane] = cost;
  sharedCount[lane] = count;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride >> 1u) {
    if (lane < stride) {
      sharedCost[lane] += sharedCost[lane + stride];
      sharedCount[lane] += sharedCount[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    state[stateOffset + 3u] = sharedCount[0];
    state[stateOffset + 4u] = bitcast<u32>(sharedCost[0]);
  }`
      }),
      // One thread, ungated (it writes the gate): applies the round to the totals.
      createWGSLKernelNode<Parameters>(graph, {
        operation: 'GPUWardRegions',
        id: `${prefix}-finish`,
        variant: 'mutual-finish',
        invocationCount: 1,
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read_write'},
          {name: 'objective', view: output.objective, type: 'f32', access: 'read_write'},
          {name: 'status', view: output.status, type: 'u32', access: 'read_write'},
          gate.binding
        ],
        declarations: `const TARGET_REGION_COUNT: u32 = ${props.targetRegionCount}u;`,
        // A zero x in slot 0 means an earlier round disarmed the gate; the sums are then stale.
        body: `if (dispatch[dispatchOffset] == 0u) { return; }
  let merged = state[stateOffset + 3u];
  objective[objectiveOffset] += bitcast<f32>(state[stateOffset + 4u]);
  state[stateOffset] -= merged;
  state[stateOffset + 1u] += 1u;
  status[statusOffset] += merged;
  ${hasRoundCountWord ? `status[statusOffset + ${GPU_WARD_ROUND_COUNT_STATUS_WORD}u] = state[stateOffset + 1u];` : ''}
  // No reciprocal pair means no adjacent regions remain (the cheapest pair is always reciprocal).
  if (merged == 0u) { state[stateOffset + 2u] = 1u; }
  if (merged == 0u || state[stateOffset] <= TARGET_REGION_COUNT || status[statusOffset + 3u] != 0u) {
    ${gate.disarmWGSL}
  }`
      })
    ];
  }

  private createMutualNearestFinalizeNode<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    state: GraphDataView<'uint32'>
  ): GPUCommandNode<Parameters> {
    const {id, props} = this;
    return createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-finalize`,
      operation: 'GPUWardRegions',
      variant: 'mutual-finalize',
      invocationCount: 1,
      bindings: [
        {name: 'state', view: state, type: 'u32', access: 'read'},
        {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
      ],
      // The round budget is an override, not WGSL text, so builds that differ only in
      // `maximumIterations` share one shader module.
      declarations: `const TARGET_REGION_COUNT: u32 = ${props.targetRegionCount}u;
override MAXIMUM_ROUNDS: u32;`,
      constants: {MAXIMUM_ROUNDS: props.maximumIterations},
      body: `let remaining = state[stateOffset];
  let valid = status[statusOffset + 3u] == 0u;
  status[statusOffset + 1u] = select(0u, 1u, remaining == TARGET_REGION_COUNT && valid);
  status[statusOffset + 2u] = select(0u, 1u, remaining > TARGET_REGION_COUNT && valid &&
    state[stateOffset + 1u] == MAXIMUM_ROUNDS && state[stateOffset + 2u] == 0u);`
    });
  }
}
