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

export const GPU_AZP_MAXIMUM_COLUMNS = 8;
export const GPU_AZP_MAXIMUM_ITERATIONS = 256;
export const GPU_AZP_STATUS = {
  iterationCount: 0,
  converged: 1,
  iterationLimitReached: 2,
  invalidCount: 3,
  length: 4
} as const;

export type GPUAZPRegionsProps = {
  id?: string;
  weights: GPUSpatialWeights;
  values: GraphDataView<'float32'>;
  columnCount?: number;
  /** Contiguous initial labels over symmetric weights. Values must be below `regionCapacity`. */
  initialLabels: GraphDataView<'uint32'>;
  regionCapacity: number;
  minimumRegionSize?: number;
  maximumIterations: number;
  /** Reproducible tie-order seed; record the seed used to create `initialLabels` here too. */
  seed: number;
  labels: GraphDataView<'uint32'>;
  optimization: GPUOptimizationStatusPort;
};

/**
 * Bounded deterministic AZP local search. Starting from a caller-supplied contiguous partition,
 * each iteration selects the greatest reduction in total within-region squared error. Every
 * candidate removal runs an exact reachability search over its source region; requiring a
 * destination neighbor then preserves destination connectivity as well.
 * Ties use a stable seed-keyed row order and then the destination label. The search therefore
 * remains reproducible without random GPU state. It stops at a local optimum or the explicit
 * iteration budget and publishes objective and convergence status.
 */
export class GPUAZPRegions implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUAZPRegionsProps;
  readonly rowCount: number;
  readonly columnCount: number;

  constructor(props: GPUAZPRegionsProps) {
    this.id = props.id ?? 'azp-regions';
    this.props = props;
    this.rowCount = validateGPUSpatialWeights(this.id, props.weights);
    this.columnCount = props.columnCount ?? 1;
    const {id, rowCount, columnCount} = this;
    if (
      !Number.isSafeInteger(columnCount) ||
      columnCount < 1 ||
      columnCount > GPU_AZP_MAXIMUM_COLUMNS
    ) {
      throw new Error(`${id} columnCount must be an integer in [1, ${GPU_AZP_MAXIMUM_COLUMNS}]`);
    }
    if (
      !Number.isSafeInteger(props.regionCapacity) ||
      props.regionCapacity < 2 ||
      props.regionCapacity > rowCount
    ) {
      throw new Error(`${id} regionCapacity must be an integer in [2, rowCount]`);
    }
    if (
      !Number.isSafeInteger(props.maximumIterations) ||
      props.maximumIterations < 1 ||
      props.maximumIterations > GPU_AZP_MAXIMUM_ITERATIONS
    ) {
      throw new Error(
        `${id} maximumIterations must be an integer in [1, ${GPU_AZP_MAXIMUM_ITERATIONS}]`
      );
    }
    const minimumRegionSize = props.minimumRegionSize ?? 1;
    if (!Number.isSafeInteger(minimumRegionSize) || minimumRegionSize < 1)
      throw new Error(`${id} minimumRegionSize must be positive`);
    if (!Number.isSafeInteger(props.seed) || props.seed < 0 || props.seed > 0xffffffff)
      throw new Error(`${id} seed must be a uint32`);
    validatePackedView(props.values, ['float32'], `${id} values`);
    if (props.values.length !== rowCount * columnCount)
      throw new Error(`${id} values length must equal rows * columnCount`);
    for (const [name, view] of [
      ['initialLabels', props.initialLabels],
      ['labels', props.labels],
      ['optimization.status', props.optimization.status]
    ] as const) {
      validatePackedUint32View(view, `${id} ${name}`);
    }
    if (props.initialLabels.length !== rowCount || props.labels.length !== rowCount)
      throw new Error(`${id} label views must equal the weights row count`);
    validatePackedView(props.optimization.objective, ['float32'], `${id} optimization.objective`);
    if (props.optimization.objective.length < 1)
      throw new Error(`${id} optimization.objective must hold one value`);
    if (props.optimization.status.length < GPU_AZP_STATUS.length)
      throw new Error(`${id} optimization.status must hold ${GPU_AZP_STATUS.length} words`);
    if (props.optimization.seed !== props.seed)
      throw new Error(`${id} optimization.seed must equal seed`);
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.labels, props.optimization.objective, props.optimization.status],
      [
        props.weights.offsets,
        props.weights.neighbors,
        props.weights.weights,
        props.values,
        props.initialLabels
      ]
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
      props.initialLabels,
      props.labels,
      props.optimization.objective,
      props.optimization.status
    ]);
    const regionStride = columnCount + 1;
    const regions = createTransientView(
      graph,
      `${id}-regions`,
      'float32',
      props.regionCapacity * regionStride
    );
    const minimumRegionSize = props.minimumRegionSize ?? 1;
    const state = createTransientView(graph, `${id}-state`, 'uint32', 1);
    const candidateDeltaA = createTransientView(
      graph,
      `${id}-candidate-delta-a`,
      'float32',
      rowCount
    );
    const candidateDeltaB = createTransientView(
      graph,
      `${id}-candidate-delta-b`,
      'float32',
      rowCount
    );
    const candidateDataA = createTransientView(
      graph,
      `${id}-candidate-data-a`,
      'uint32',
      3 * rowCount
    );
    const candidateDataB = createTransientView(
      graph,
      `${id}-candidate-data-b`,
      'uint32',
      3 * rowCount
    );
    const declarations = `const ROWS: u32 = ${rowCount}u;
const COLUMNS: u32 = ${columnCount}u;
const REGION_CAPACITY: u32 = ${props.regionCapacity}u;
const REGION_STRIDE: u32 = ${regionStride}u;
const MINIMUM_SIZE: u32 = ${minimumRegionSize}u;
const SEED: u32 = ${props.seed}u;
const NONE: u32 = 0xffffffffu;
const LARGE_DELTA: f32 = 3.402823466e+38;
fn tieKey(row: u32) -> u32 {
  var value = row ^ SEED;
  value = (value ^ (value >> 16u)) * 0x7feb352du;
  value = (value ^ (value >> 15u)) * 0x846ca68bu;
  return value ^ (value >> 16u);
}
fn isFiniteFloat(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
fn regionCount(region: u32) -> u32 { return u32(regions[regionsOffset + region * REGION_STRIDE]); }
fn regionSum(region: u32, column: u32) -> f32 { return regions[regionsOffset + region * REGION_STRIDE + 1u + column]; }`;
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initialize-status`,
        operation: 'GPUAZPRegions',
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
        body: `state[stateOffset] = 0u;
  objective[objectiveOffset] = 0.0;
  status[statusOffset] = 0u;
  status[statusOffset + 1u] = 0u;
  status[statusOffset + 2u] = 0u;
  status[statusOffset + 3u] = 0u;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-validate-and-copy`,
        operation: 'GPUAZPRegions',
        variant: 'parallel-validation',
        invocationCount: rowCount,
        bindings: [
          {name: 'offsets', view: props.weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: props.weights.neighbors, type: 'u32', access: 'read'},
          {name: 'values', view: props.values, type: 'f32', access: 'read'},
          {name: 'initialLabels', view: props.initialLabels, type: 'u32', access: 'read'},
          {name: 'labels', view: props.labels, type: 'u32', access: 'read_write'},
          {
            name: 'status',
            view: props.optimization.status,
            type: 'atomic<u32>',
            access: 'read_write'
          }
        ],
        declarations: `const ROWS: u32 = ${rowCount}u;
const COLUMNS: u32 = ${columnCount}u;
const REGION_CAPACITY: u32 = ${props.regionCapacity}u;
fn isFiniteFloat(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }`,
        body: `let region = initialLabels[initialLabelsOffset + index];
  labels[labelsOffset + index] = region;
  var invalidCount = select(0u, 1u, region >= REGION_CAPACITY);
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    invalidCount += select(0u, 1u, neighbors[neighborsOffset + slot] >= ROWS);
  }
  for (var column = 0u; column < COLUMNS; column++) {
    invalidCount += select(0u, 1u, !isFiniteFloat(values[valuesOffset + index * COLUMNS + column]));
  }
  if (invalidCount > 0u) { atomicAdd(&status[statusOffset + 3u], invalidCount); }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-region-statistics`,
        operation: 'GPUAZPRegions',
        variant: 'parallel-region-statistics',
        invocationCount: props.regionCapacity,
        bindings: [
          {name: 'labels', view: props.labels, type: 'u32', access: 'read'},
          {name: 'values', view: props.values, type: 'f32', access: 'read'},
          {name: 'regions', view: regions, type: 'f32', access: 'read_write'}
        ],
        declarations,
        body: `let region = index;
  let base = regionsOffset + region * REGION_STRIDE;
  var count = 0u;
  for (var column = 0u; column < COLUMNS; column++) { regions[base + 1u + column] = 0.0; }
  for (var row = 0u; row < ROWS; row++) {
    if (labels[labelsOffset + row] != region) { continue; }
    count++;
    for (var column = 0u; column < COLUMNS; column++) {
      regions[base + 1u + column] += values[valuesOffset + row * COLUMNS + column];
    }
  }
  regions[base] = f32(count);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-objective-rows`,
        operation: 'GPUAZPRegions',
        variant: 'parallel-objective',
        invocationCount: rowCount,
        bindings: [
          {name: 'labels', view: props.labels, type: 'u32', access: 'read'},
          {name: 'values', view: props.values, type: 'f32', access: 'read'},
          {name: 'regions', view: regions, type: 'f32', access: 'read'},
          {name: 'deltaOut', view: candidateDeltaA, type: 'f32', access: 'read_write'},
          {name: 'status', view: props.optimization.status, type: 'u32', access: 'read'}
        ],
        declarations,
        body: `if (status[statusOffset + 3u] > 0u) { deltaOut[deltaOutOffset + index] = 0.0; return; }
    let region = labels[labelsOffset + index];
    let count = f32(regionCount(region));
    var contribution = 0.0;
    for (var column = 0u; column < COLUMNS; column++) {
      let difference = values[valuesOffset + index * COLUMNS + column] - regionSum(region, column) / count;
      contribution += difference * difference;
    }
  deltaOut[deltaOutOffset + index] = contribution;`
      })
    ];

    let objectiveCount = rowCount;
    let objectiveSource = candidateDeltaA;
    let objectiveDestination = candidateDeltaB;
    let objectiveLevel = 0;
    while (objectiveCount > 1) {
      const nextCount = Math.ceil(objectiveCount / 2);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-objective-reduce-${objectiveLevel}`,
          operation: 'GPUAZPRegions',
          variant: 'deterministic-objective-reduction',
          invocationCount: nextCount,
          bindings: [
            {name: 'valuesIn', view: objectiveSource, type: 'f32', access: 'read'},
            {name: 'valuesOut', view: objectiveDestination, type: 'f32', access: 'read_write'}
          ],
          declarations: `const INPUT_COUNT: u32 = ${objectiveCount}u;`,
          body: `let left = 2u * index;
  let right = left + 1u;
  var total = valuesIn[valuesInOffset + left];
  if (right < INPUT_COUNT) { total += valuesIn[valuesInOffset + right]; }
  valuesOut[valuesOutOffset + index] = total;`
        })
      );
      objectiveCount = nextCount;
      [objectiveSource, objectiveDestination] = [objectiveDestination, objectiveSource];
      objectiveLevel++;
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-objective-publish`,
        operation: 'GPUAZPRegions',
        variant: 'objective-publish',
        invocationCount: 1,
        bindings: [
          {name: 'objectiveIn', view: objectiveSource, type: 'f32', access: 'read'},
          {
            name: 'objectiveOut',
            view: props.optimization.objective,
            type: 'f32',
            access: 'read_write'
          },
          {name: 'status', view: props.optimization.status, type: 'u32', access: 'read'}
        ],
        body: `objectiveOut[objectiveOutOffset] = select(objectiveIn[objectiveInOffset], 0.0, status[statusOffset + 3u] > 0u);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initialize-state`,
        operation: 'GPUAZPRegions',
        variant: 'initialize-state',
        invocationCount: 1,
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read_write'},
          {name: 'status', view: props.optimization.status, type: 'u32', access: 'read'}
        ],
        body: 'state[stateOffset] = select(0u, 1u, status[statusOffset + 3u] > 0u);'
      })
    );

    for (let iteration = 0; iteration < props.maximumIterations; iteration++) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-iteration-${iteration}-candidates`,
          operation: 'GPUAZPRegions',
          variant: 'exact-connectivity-candidates',
          invocationCount: rowCount,
          bindings: [
            {name: 'offsets', view: props.weights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: props.weights.neighbors, type: 'u32', access: 'read'},
            {name: 'values', view: props.values, type: 'f32', access: 'read'},
            {name: 'labels', view: props.labels, type: 'u32', access: 'read'},
            {name: 'regions', view: regions, type: 'f32', access: 'read'},
            {name: 'state', view: state, type: 'u32', access: 'read'},
            {name: 'deltaOut', view: candidateDeltaA, type: 'f32', access: 'read_write'},
            {name: 'dataOut', view: candidateDataA, type: 'u32', access: 'read_write'}
          ],
          declarations: `${declarations}
var<private> reached: array<bool, ${rowCount}>;`,
          body: `let row = index;
  let source = labels[labelsOffset + row];
  var sourceCount = 0u;
  if (state[stateOffset] == 0u) { sourceCount = regionCount(source); }
  var connected = sourceCount > MINIMUM_SIZE && state[stateOffset] == 0u;
  var root = NONE;
  if (connected) {
    for (var vertex = 0u; vertex < ROWS; vertex++) {
      reached[vertex] = false;
      if (vertex != row && labels[labelsOffset + vertex] == source) { root = min(root, vertex); }
    }
    connected = root != NONE;
  }
  if (connected) {
    reached[root] = true;
    for (var propagation = 0u; propagation < ROWS; propagation++) {
      var changed = false;
      for (var vertex = 0u; vertex < ROWS; vertex++) {
        if (!reached[vertex] || vertex == row || labels[labelsOffset + vertex] != source) { continue; }
        for (var slot = offsets[offsetsOffset + vertex]; slot < offsets[offsetsOffset + vertex + 1u]; slot++) {
          let neighbor = neighbors[neighborsOffset + slot];
          if (neighbor != row && neighbor < ROWS && labels[labelsOffset + neighbor] == source && !reached[neighbor]) {
            reached[neighbor] = true; changed = true;
          }
        }
      }
      if (!changed) { break; }
    }
    for (var vertex = 0u; vertex < ROWS; vertex++) {
      if (vertex != row && labels[labelsOffset + vertex] == source && !reached[vertex]) { connected = false; }
    }
  }
  var bestDelta = LARGE_DELTA;
  var bestDestination = NONE;
  if (connected) {
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor >= ROWS) { continue; }
      let destination = labels[labelsOffset + neighbor];
      if (destination == source) { continue; }
      let destinationCount = regionCount(destination);
      var delta = 0.0;
      for (var column = 0u; column < COLUMNS; column++) {
        let value = values[valuesOffset + row * COLUMNS + column];
        let sourceSum = regionSum(source, column);
        let destinationSum = regionSum(destination, column);
        delta += sourceSum * sourceSum / f32(sourceCount) + destinationSum * destinationSum / f32(destinationCount)
          - (sourceSum - value) * (sourceSum - value) / f32(sourceCount - 1u)
          - (destinationSum + value) * (destinationSum + value) / f32(destinationCount + 1u);
      }
      if (delta < bestDelta || (delta == bestDelta && destination < bestDestination)) {
        bestDelta = delta; bestDestination = destination;
      }
    }
  }
  if (bestDelta >= 0.0) { bestDelta = LARGE_DELTA; bestDestination = NONE; }
  deltaOut[deltaOutOffset + row] = bestDelta;
  let base = dataOutOffset + 3u * row;
  dataOut[base] = select(row, NONE, bestDestination == NONE);
  dataOut[base + 1u] = bestDestination;
  dataOut[base + 2u] = select(tieKey(row), NONE, bestDestination == NONE);`
        })
      );

      let activeCount = rowCount;
      let sourceDelta = candidateDeltaA;
      let sourceData = candidateDataA;
      let destinationDelta = candidateDeltaB;
      let destinationData = candidateDataB;
      let reduction = 0;
      while (activeCount > 1) {
        const nextCount = Math.ceil(activeCount / 2);
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-iteration-${iteration}-reduce-${reduction}`,
            operation: 'GPUAZPRegions',
            variant: 'deterministic-candidate-reduction',
            invocationCount: nextCount,
            bindings: [
              {name: 'deltaIn', view: sourceDelta, type: 'f32', access: 'read'},
              {name: 'dataIn', view: sourceData, type: 'u32', access: 'read'},
              {name: 'deltaOut', view: destinationDelta, type: 'f32', access: 'read_write'},
              {name: 'dataOut', view: destinationData, type: 'u32', access: 'read_write'}
            ],
            declarations: `const INPUT_COUNT: u32 = ${activeCount}u;`,
            body: `let left = 2u * index;
  let right = left + 1u;
  var winner = left;
  if (right < INPUT_COUNT) {
    let leftDelta = deltaIn[deltaInOffset + left];
    let rightDelta = deltaIn[deltaInOffset + right];
    let leftBase = dataInOffset + 3u * left;
    let rightBase = dataInOffset + 3u * right;
    let leftKey = dataIn[leftBase + 2u];
    let rightKey = dataIn[rightBase + 2u];
    let leftRow = dataIn[leftBase];
    let rightRow = dataIn[rightBase];
    let leftDestination = dataIn[leftBase + 1u];
    let rightDestination = dataIn[rightBase + 1u];
    if (rightDelta < leftDelta || (rightDelta == leftDelta &&
        (rightKey < leftKey || (rightKey == leftKey &&
        (rightRow < leftRow || (rightRow == leftRow && rightDestination < leftDestination)))))) { winner = right; }
  }
  deltaOut[deltaOutOffset + index] = deltaIn[deltaInOffset + winner];
  let sourceBase = dataInOffset + 3u * winner;
  let destinationBase = dataOutOffset + 3u * index;
  dataOut[destinationBase] = dataIn[sourceBase];
  dataOut[destinationBase + 1u] = dataIn[sourceBase + 1u];
  dataOut[destinationBase + 2u] = dataIn[sourceBase + 2u];`
          })
        );
        activeCount = nextCount;
        [sourceDelta, destinationDelta] = [destinationDelta, sourceDelta];
        [sourceData, destinationData] = [destinationData, sourceData];
        reduction++;
      }

      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-iteration-${iteration}-apply`,
          operation: 'GPUAZPRegions',
          variant: 'apply-exact-move',
          invocationCount: 1,
          bindings: [
            {name: 'delta', view: sourceDelta, type: 'f32', access: 'read'},
            {name: 'data', view: sourceData, type: 'u32', access: 'read'},
            {name: 'values', view: props.values, type: 'f32', access: 'read'},
            {name: 'labels', view: props.labels, type: 'u32', access: 'read_write'},
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
          declarations: `${declarations}`,
          body: `let row = data[dataOffset];
  let destination = data[dataOffset + 1u];
  if (row == NONE || status[statusOffset + 3u] > 0u) { state[stateOffset] = 1u; return; }
  let source = labels[labelsOffset + row];
  labels[labelsOffset + row] = destination;
  let sourceBase = regionsOffset + source * REGION_STRIDE;
  let destinationBase = regionsOffset + destination * REGION_STRIDE;
  regions[sourceBase] -= 1.0;
  regions[destinationBase] += 1.0;
  for (var column = 0u; column < COLUMNS; column++) {
    let value = values[valuesOffset + row * COLUMNS + column];
    regions[sourceBase + 1u + column] -= value;
    regions[destinationBase + 1u + column] += value;
  }
  objective[objectiveOffset] += delta[deltaOffset];
  status[statusOffset] += 1u;`
        })
      );
    }

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finalize`,
        operation: 'GPUAZPRegions',
        variant: 'finalize',
        invocationCount: 1,
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'status', view: props.optimization.status, type: 'u32', access: 'read_write'}
        ],
        declarations: `const MAXIMUM_ITERATIONS: u32 = ${props.maximumIterations}u;`,
        body: `let invalid = status[statusOffset + 3u] > 0u;
  status[statusOffset + 1u] = select(0u, 1u, !invalid && state[stateOffset] != 0u);
  status[statusOffset + 2u] = select(0u, 1u, !invalid && state[stateOffset] == 0u && status[statusOffset] == MAXIMUM_ITERATIONS);`
      })
    );
    return nodes;
  }
}
