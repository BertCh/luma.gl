// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  getBoundedDispatchLayout,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GPUCommandGraphNodeCondition,
  type GraphBufferUse,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  createWGSLKernelNode,
  getAutomaticWGSLKernelWorkgroupSize,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import type {GPUOptimizationStatusPort} from '../contracts/index';
import {type GPUSpatialWeights, validateGPUSpatialWeights} from '../spatial-weights/index';
import {IS_FINITE_FLOAT_WGSL} from '../../utils/wgsl-common';
import {createStandardizeColumnsNode, createSymmetricAdjacency} from './regionalization-inputs';

export const GPU_WARD_MAXIMUM_COLUMNS = 8;
const GPU_WARD_REDUCTION_WORKGROUP_SIZE = 256;

export const GPU_WARD_STATUS = {
  iterationCount: 0,
  converged: 1,
  iterationLimitReached: 2,
  invalidCount: 3,
  length: 4
} as const;

export type GPUWardRegionsProps = {
  id?: string;
  /**
   * Spatial weights that constrain which regions may merge. Two rows are adjacent when either
   * lists the other, so directed (for example k-nearest-neighbor) weights are symmetrized inside
   * the contributor; weights that declare `info.symmetric === true` are used as they are.
   */
  weights: GPUSpatialWeights;
  /** Row-major feature values. */
  values: GraphDataView<'float32'>;
  columnCount?: number;
  /**
   * Standardize every column to mean 0 and population standard deviation 1 (divisor `n`, the
   * `StandardScaler` convention, over finite entries) before merging, so columns in different
   * units weigh equally. Defaults to true; pass false when `values` is already on a common scale.
   * `optimization.objective` is then in standardized units.
   */
  standardize?: boolean;
  targetRegionCount: number;
  /** Maximum adjacent-region merges. Must not exceed `rowCount - targetRegionCount`. */
  maximumIterations: number;
  /** Reproducible tie-order seed. */
  seed: number;
  labels: GraphDataView<'uint32'>;
  optimization: GPUOptimizationStatusPort;
};

/**
 * Spatially constrained Ward agglomeration with an explicit merge budget.
 *
 * `optimization.objective` is the within-region sum of squared deviations of the returned
 * partition (every row starts as its own region with zero, each merge adds its Ward increase),
 * the same quantity as `withinSsd` of `GPURegionPartitionEvaluation` on the same table.
 *
 * Default change: columns are standardized (`standardize`, default true); the previous raw-value
 * behavior is `standardize: false`. Adjacency is the undirected union of the weights and their
 * transpose, so a directed link is a link in both directions.
 *
 * Every row begins as its own region. Each iteration merges the adjacent region pair with the
 * smallest Ward increase in within-region squared error. Ties use a stable seed-keyed pair order.
 * Disconnected weights can prevent the requested region count; the optimization status reports
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
    if (
      !Number.isSafeInteger(columnCount) ||
      columnCount < 1 ||
      columnCount > GPU_WARD_MAXIMUM_COLUMNS
    ) {
      throw new Error(`${id} columnCount must be an integer in [1, ${GPU_WARD_MAXIMUM_COLUMNS}]`);
    }
    if (
      !Number.isSafeInteger(props.targetRegionCount) ||
      props.targetRegionCount < 1 ||
      props.targetRegionCount > rowCount
    ) {
      throw new Error(`${id} targetRegionCount must be an integer in [1, rowCount]`);
    }
    const maximumMergeCount = rowCount - props.targetRegionCount;
    if (
      !Number.isSafeInteger(props.maximumIterations) ||
      props.maximumIterations < 0 ||
      props.maximumIterations > maximumMergeCount
    ) {
      throw new Error(`${id} maximumIterations must be an integer in [0, ${maximumMergeCount}]`);
    }
    if (!Number.isSafeInteger(props.seed) || props.seed < 0 || props.seed > 0xffffffff) {
      throw new Error(`${id} seed must be a uint32`);
    }
    validatePackedView(props.values, ['float32'], `${id} values`);
    if (props.values.length !== rowCount * columnCount) {
      throw new Error(`${id} values length must equal rows * columnCount`);
    }
    validatePackedUint32View(props.labels, `${id} labels`);
    if (props.labels.length !== rowCount) {
      throw new Error(`${id} labels length must equal the weights row count`);
    }
    validatePackedView(props.optimization.objective, ['float32'], `${id} optimization.objective`);
    validatePackedUint32View(props.optimization.status, `${id} optimization.status`);
    if (props.optimization.objective.length < 1) {
      throw new Error(`${id} optimization.objective must hold one value`);
    }
    if (props.optimization.status.length < GPU_WARD_STATUS.length) {
      throw new Error(`${id} optimization.status must hold ${GPU_WARD_STATUS.length} words`);
    }
    if (props.optimization.seed !== props.seed) {
      throw new Error(`${id} optimization.seed must equal seed`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.labels, props.optimization.objective, props.optimization.status],
      [props.weights.offsets, props.weights.neighbors, props.weights.weights, props.values]
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rowCount, columnCount} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.weights.offsets,
      props.weights.neighbors,
      props.weights.weights,
      props.values,
      props.labels,
      props.optimization.objective,
      props.optimization.status
    ]);
    const regionStride = columnCount + 1;
    const regions = createTransientView(graph, `${id}-regions`, 'float32', rowCount * regionStride);
    const state = createTransientView(graph, `${id}-state`, 'uint32', 1);
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
    for (let activeCount = rowCount; activeCount > 1; ) {
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
    const maximumWorkgroupsPerDimension = graph.device.limits.maxComputeWorkgroupsPerDimension;
    const gatedLayouts = [
      getBoundedDispatchLayout(
        'GPUWardRegions',
        rowCount,
        rowWorkgroupSize,
        maximumWorkgroupsPerDimension
      ),
      ...reductionStages.map(stage =>
        getBoundedDispatchLayout(
          'GPUWardRegions',
          stage.activeCount,
          stage.workgroupSize,
          maximumWorkgroupsPerDimension
        )
      )
    ];
    const gate =
      props.maximumIterations > 0
        ? createTransientView(
            graph,
            `${id}-merge-dispatch`,
            'uint32',
            3 * gatedLayouts.length,
            Buffer.STORAGE | Buffer.INDIRECT
          )
        : null;
    const getGate = (
      slot: number
    ): {condition: GPUCommandGraphNodeCondition<Parameters>; extraResources: GraphBufferUse[]} => ({
      condition: {
        id: `${id}-merge-gate-${slot}`,
        source: 'gpu',
        mode: 'indirect',
        buffer: gate!.buffer,
        byteOffset: gate!.byteOffset + slot * 12
      },
      extraResources: [{buffer: gate!, usage: 'indirect'}]
    });
    const armGateWGSL = gatedLayouts
      .map(
        (layout, slot) =>
          `dispatch[dispatchOffset + ${3 * slot}u] = ${layout.x}u;
  dispatch[dispatchOffset + ${3 * slot + 1}u] = ${layout.y}u;
  dispatch[dispatchOffset + ${3 * slot + 2}u] = ${layout.z}u;`
      )
      .join('\n  ');
    const disarmGateWGSL = gatedLayouts
      .map((_, slot) => `dispatch[dispatchOffset + ${3 * slot}u] = 0u;`)
      .join(' ');
    const declarations = `const ROW_COUNT: u32 = ${rowCount}u;
const COLUMN_COUNT: u32 = ${columnCount}u;
const REGION_STRIDE: u32 = ${regionStride}u;
const TARGET_REGION_COUNT: u32 = ${props.targetRegionCount}u;
const NONE: u32 = 0xffffffffu;
const LARGE_INCREASE: f32 = 3.402823466e+38;
${IS_FINITE_FLOAT_WGSL}
fn regionCount(region: u32) -> f32 { return regions[regionsOffset + region * REGION_STRIDE]; }
fn regionSum(region: u32, column: u32) -> f32 { return regions[regionsOffset + region * REGION_STRIDE + 1u + column]; }
fn pairKey(first: u32, second: u32) -> u32 {
  var value = (first * 0x9e3779b9u) ^ second ^ ${props.seed}u;
  value = (value ^ (value >> 16u)) * 0x7feb352du;
  value = (value ^ (value >> 15u)) * 0x846ca68bu;
  return value ^ (value >> 16u);
}`;
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
            view: props.optimization.objective,
            type: 'f32',
            access: 'read_write'
          },
          {name: 'status', view: props.optimization.status, type: 'u32', access: 'read_write'},
          ...(gate
            ? [{name: 'dispatch', view: gate, type: 'u32' as const, access: 'read_write' as const}]
            : [])
        ],
        body: `state[stateOffset] = ${rowCount}u;
  objective[objectiveOffset] = 0.0;
  status[statusOffset] = 0u;
  status[statusOffset + 1u] = 0u;
  status[statusOffset + 2u] = 0u;
  status[statusOffset + 3u] = 0u;
  ${gate ? armGateWGSL : ''}`
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
          {name: 'labels', view: props.labels, type: 'u32', access: 'read_write'},
          {name: 'regions', view: regions, type: 'f32', access: 'read_write'},
          {
            name: 'status',
            view: props.optimization.status,
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
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-iteration-${iteration}-candidates`,
          operation: 'GPUWardRegions',
          variant: 'adjacent-candidates',
          invocationCount: rowCount,
          workgroupSize: rowWorkgroupSize,
          ...getGate(0),
          bindings: [
            {name: 'offsets', view: adjacency.weights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: adjacency.weights.neighbors, type: 'u32', access: 'read'},
            {name: 'labels', view: props.labels, type: 'u32', access: 'read'},
            {name: 'regions', view: regions, type: 'f32', access: 'read'},
            {name: 'costOut', view: candidateCostA, type: 'f32', access: 'read_write'},
            {name: 'firstOut', view: candidateFirstA, type: 'u32', access: 'read_write'},
            {name: 'secondOut', view: candidateSecondA, type: 'u32', access: 'read_write'}
          ],
          declarations,
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
        const keyBindings: WGSLKernelBinding[] = sourceKey
          ? [{name: 'keyIn', view: sourceKey, type: 'u32', access: 'read'}]
          : [];
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-iteration-${iteration}-reduce-${reduction}`,
            operation: 'GPUWardRegions',
            variant: 'workgroup-reduction',
            invocationCount: activeCount,
            workgroupSize,
            guardIndex: false,
            ...getGate(1 + reduction),
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
fn pairKey(first: u32, second: u32) -> u32 {
  var value = (first * 0x9e3779b9u) ^ second ^ ${props.seed}u;
  value = (value ^ (value >> 16u)) * 0x7feb352du;
  value = (value ^ (value >> 15u)) * 0x846ca68bu;
  return value ^ (value >> 16u);
}
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
          ...getGate(0),
          bindings: [
            {name: 'first', view: sourceFirst, type: 'u32', access: 'read'},
            {name: 'second', view: sourceSecond, type: 'u32', access: 'read'},
            {name: 'labels', view: props.labels, type: 'u32', access: 'read_write'},
            {name: 'status', view: props.optimization.status, type: 'u32', access: 'read'}
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
              view: props.optimization.objective,
              type: 'f32',
              access: 'read_write'
            },
            {name: 'status', view: props.optimization.status, type: 'u32', access: 'read_write'},
            {name: 'dispatch', view: gate!, type: 'u32', access: 'read_write'}
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
    ${disarmGateWGSL}
  }`
        })
      );
    }

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finalize`,
        operation: 'GPUWardRegions',
        variant: 'finalize',
        invocationCount: 1,
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'status', view: props.optimization.status, type: 'u32', access: 'read_write'}
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
}
