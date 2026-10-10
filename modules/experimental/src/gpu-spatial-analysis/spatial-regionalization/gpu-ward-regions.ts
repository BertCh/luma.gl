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
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import type {GPUOptimizationStatusPort} from '../contracts/index';
import {type GPUSpatialWeights, validateGPUSpatialWeights} from '../spatial-weights/index';

export const GPU_WARD_MAXIMUM_COLUMNS = 8;
export const GPU_WARD_STATUS = {
  iterationCount: 0,
  converged: 1,
  iterationLimitReached: 2,
  invalidCount: 3,
  length: 4
} as const;

export type GPUWardRegionsProps = {
  id?: string;
  /** Symmetric spatial weights that constrain which regions may merge. */
  weights: GPUSpatialWeights;
  /** Row-major feature values. */
  values: GraphDataView<'float32'>;
  columnCount?: number;
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
 * Every row begins as its own region. Each iteration merges the adjacent region pair with the
 * smallest Ward increase in within-region squared error. Ties use a stable seed-keyed pair order.
 * Disconnected weights can prevent the requested region count; the optimization status reports
 * that condition without readback during execution.
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
    const declarations = `const ROW_COUNT: u32 = ${rowCount}u;
const COLUMN_COUNT: u32 = ${columnCount}u;
const REGION_STRIDE: u32 = ${regionStride}u;
const TARGET_REGION_COUNT: u32 = ${props.targetRegionCount}u;
const NONE: u32 = 0xffffffffu;
const LARGE_INCREASE: f32 = 3.402823466e+38;
fn isFiniteFloat(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
fn regionCount(region: u32) -> f32 { return regions[regionsOffset + region * REGION_STRIDE]; }
fn regionSum(region: u32, column: u32) -> f32 { return regions[regionsOffset + region * REGION_STRIDE + 1u + column]; }
fn pairKey(first: u32, second: u32) -> u32 {
  var value = (first * 0x9e3779b9u) ^ second ^ ${props.seed}u;
  value = (value ^ (value >> 16u)) * 0x7feb352du;
  value = (value ^ (value >> 15u)) * 0x846ca68bu;
  return value ^ (value >> 16u);
}`;
    const nodes: GPUCommandNode<Parameters>[] = [
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
          {name: 'status', view: props.optimization.status, type: 'u32', access: 'read_write'}
        ],
        body: `state[stateOffset] = ${rowCount}u;
  objective[objectiveOffset] = 0.0;
  status[statusOffset] = 0u;
  status[statusOffset + 1u] = 0u;
  status[statusOffset + 2u] = 0u;
  status[statusOffset + 3u] = 0u;`
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
    let value = values[valuesOffset + index * COLUMN_COUNT + column];
    invalidCount += select(0u, 1u, !isFiniteFloat(value));
    regions[regionBase + 1u + column] = value;
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
          bindings: [
            {name: 'offsets', view: props.weights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: props.weights.neighbors, type: 'u32', access: 'read'},
            {name: 'labels', view: props.labels, type: 'u32', access: 'read'},
            {name: 'regions', view: regions, type: 'f32', access: 'read'},
            {name: 'state', view: state, type: 'u32', access: 'read'},
            {name: 'costOut', view: candidateCostA, type: 'f32', access: 'read_write'},
            {name: 'firstOut', view: candidateFirstA, type: 'u32', access: 'read_write'},
            {name: 'secondOut', view: candidateSecondA, type: 'u32', access: 'read_write'}
          ],
          declarations,
          body: `var bestFirst = NONE;
  var bestSecond = NONE;
  var bestIncrease = LARGE_INCREASE;
  var bestKey = NONE;
  if (state[stateOffset] > TARGET_REGION_COUNT) {
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
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-iteration-${iteration}-candidate-keys`,
          operation: 'GPUWardRegions',
          variant: 'candidate-keys',
          invocationCount: rowCount,
          bindings: [
            {name: 'firstIn', view: candidateFirstA, type: 'u32', access: 'read'},
            {name: 'secondIn', view: candidateSecondA, type: 'u32', access: 'read'},
            {name: 'keyOut', view: candidateKeyA, type: 'u32', access: 'read_write'}
          ],
          declarations: `const NONE: u32 = 0xffffffffu;
fn pairKey(first: u32, second: u32) -> u32 {
  var value = (first * 0x9e3779b9u) ^ second ^ ${props.seed}u;
  value = (value ^ (value >> 16u)) * 0x7feb352du;
  value = (value ^ (value >> 15u)) * 0x846ca68bu;
  return value ^ (value >> 16u);
}`,
          body: `let first = firstIn[firstInOffset + index];
  keyOut[keyOutOffset + index] = select(pairKey(first, secondIn[secondInOffset + index]), NONE, first == NONE);`
        })
      );

      let activeCount = rowCount;
      let sourceCost = candidateCostA;
      let sourceFirst = candidateFirstA;
      let sourceSecond = candidateSecondA;
      let sourceKey = candidateKeyA;
      let destinationCost = candidateCostB;
      let destinationFirst = candidateFirstB;
      let destinationSecond = candidateSecondB;
      let destinationKey = candidateKeyB;
      let reduction = 0;
      while (activeCount > 1) {
        const nextCount = Math.ceil(activeCount / 2);
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-iteration-${iteration}-reduce-${reduction}`,
            operation: 'GPUWardRegions',
            variant: 'deterministic-reduction',
            invocationCount: nextCount,
            bindings: [
              {name: 'costIn', view: sourceCost, type: 'f32', access: 'read'},
              {name: 'firstIn', view: sourceFirst, type: 'u32', access: 'read'},
              {name: 'secondIn', view: sourceSecond, type: 'u32', access: 'read'},
              {name: 'keyIn', view: sourceKey, type: 'u32', access: 'read'},
              {name: 'costOut', view: destinationCost, type: 'f32', access: 'read_write'},
              {name: 'firstOut', view: destinationFirst, type: 'u32', access: 'read_write'},
              {name: 'secondOut', view: destinationSecond, type: 'u32', access: 'read_write'},
              {name: 'keyOut', view: destinationKey, type: 'u32', access: 'read_write'}
            ],
            declarations: `const INPUT_COUNT: u32 = ${activeCount}u;`,
            body: `let left = 2u * index;
  let right = left + 1u;
  var winner = left;
  if (right < INPUT_COUNT) {
    let leftCost = costIn[costInOffset + left];
    let rightCost = costIn[costInOffset + right];
    let leftKey = keyIn[keyInOffset + left];
    let rightKey = keyIn[keyInOffset + right];
    let leftFirst = firstIn[firstInOffset + left];
    let rightFirst = firstIn[firstInOffset + right];
    let leftSecond = secondIn[secondInOffset + left];
    let rightSecond = secondIn[secondInOffset + right];
    if (rightCost < leftCost || (rightCost == leftCost &&
        (rightKey < leftKey || (rightKey == leftKey &&
        (rightFirst < leftFirst || (rightFirst == leftFirst && rightSecond < leftSecond)))))) {
      winner = right;
    }
  }
  costOut[costOutOffset + index] = costIn[costInOffset + winner];
  firstOut[firstOutOffset + index] = firstIn[firstInOffset + winner];
  secondOut[secondOutOffset + index] = secondIn[secondInOffset + winner];
  keyOut[keyOutOffset + index] = keyIn[keyInOffset + winner];`
          })
        );
        activeCount = nextCount;
        [sourceCost, destinationCost] = [destinationCost, sourceCost];
        [sourceFirst, destinationFirst] = [destinationFirst, sourceFirst];
        [sourceSecond, destinationSecond] = [destinationSecond, sourceSecond];
        [sourceKey, destinationKey] = [destinationKey, sourceKey];
        reduction++;
      }

      nodes.push(
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
            {name: 'status', view: props.optimization.status, type: 'u32', access: 'read_write'}
          ],
          declarations: `const COLUMN_COUNT: u32 = ${columnCount}u;
const REGION_STRIDE: u32 = ${regionStride}u;
const TARGET_REGION_COUNT: u32 = ${props.targetRegionCount}u;
const NONE: u32 = 0xffffffffu;`,
          body: `let firstRegion = first[firstOffset];
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
  }`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-iteration-${iteration}-relabel`,
          operation: 'GPUWardRegions',
          variant: 'parallel-relabel',
          invocationCount: rowCount,
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
