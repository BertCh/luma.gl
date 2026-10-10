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
import {type GPUSpatialWeights, validateGPUSpatialWeights} from '../spatial-weights/index';

export const GPU_MAX_P_MAXIMUM_COLUMNS = 8;
export const GPU_MAX_P_STATUS = {
  regionCount: 0,
  feasible: 1,
  infeasibleRegionCount: 2,
  invalidCount: 3,
  length: 4
} as const;

export type GPUMaxPRegionsOutput = {
  labels: GraphDataView<'uint32'>;
  /** Total within-region squared error for the returned partition. */
  objective: GraphDataView<'float32'>;
  /** Four words indexed by {@link GPU_MAX_P_STATUS}. */
  status: GraphDataView<'uint32'>;
};

export type GPUMaxPRegionsProps = {
  id?: string;
  /** Symmetric spatial weights used for contiguous region growth. */
  weights: GPUSpatialWeights;
  /** Row-major feature values used to choose the next region member. */
  values: GraphDataView<'float32'>;
  columnCount?: number;
  /** Non-negative extensive value per row. */
  extensiveValues: GraphDataView<'float32'>;
  /** One per-frame minimum extensive value for a feasible region. */
  threshold: GraphDataView<'float32'>;
  /** Reproducible seed and tie-order input. */
  seed: number;
  output: GPUMaxPRegionsOutput;
};

/**
 * Deterministic bounded greedy Max-P regionalization.
 *
 * The contributor grows one contiguous region at a time until its extensive sum reaches the
 * threshold, selecting the adjacent row nearest to the region mean. An under-threshold remainder
 * is merged into an adjacent completed region when possible. This is a reproducible construction
 * heuristic rather than the randomized multi-start search used by some CPU implementations.
 */
export class GPUMaxPRegions implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUMaxPRegionsProps;
  readonly rowCount: number;
  readonly columnCount: number;

  constructor(props: GPUMaxPRegionsProps) {
    this.id = props.id ?? 'max-p-regions';
    this.props = props;
    this.rowCount = validateGPUSpatialWeights(this.id, props.weights);
    this.columnCount = props.columnCount ?? 1;
    const {id, rowCount, columnCount} = this;
    const {output} = props;
    if (
      !Number.isSafeInteger(columnCount) ||
      columnCount < 1 ||
      columnCount > GPU_MAX_P_MAXIMUM_COLUMNS
    ) {
      throw new Error(`${id} columnCount must be an integer in [1, ${GPU_MAX_P_MAXIMUM_COLUMNS}]`);
    }
    if (!Number.isSafeInteger(props.seed) || props.seed < 0 || props.seed > 0xffffffff) {
      throw new Error(`${id} seed must be a uint32`);
    }
    validatePackedView(props.values, ['float32'], `${id} values`);
    validatePackedView(props.extensiveValues, ['float32'], `${id} extensiveValues`);
    validatePackedView(props.threshold, ['float32'], `${id} threshold`);
    if (props.values.length !== rowCount * columnCount) {
      throw new Error(`${id} values length must equal rows * columnCount`);
    }
    if (props.extensiveValues.length !== rowCount) {
      throw new Error(`${id} extensiveValues length must equal the weights row count`);
    }
    if (props.threshold.length < 1) {
      throw new Error(`${id} threshold must hold one value`);
    }
    validatePackedUint32View(output.labels, `${id} output.labels`);
    validatePackedUint32View(output.status, `${id} output.status`);
    validatePackedView(output.objective, ['float32'], `${id} output.objective`);
    if (output.labels.length !== rowCount) {
      throw new Error(`${id} output.labels length must equal the weights row count`);
    }
    if (output.objective.length < 1) {
      throw new Error(`${id} output.objective must hold one value`);
    }
    if (output.status.length < GPU_MAX_P_STATUS.length) {
      throw new Error(`${id} output.status must hold ${GPU_MAX_P_STATUS.length} words`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.labels, output.objective, output.status],
      [
        props.weights.offsets,
        props.weights.neighbors,
        props.weights.weights,
        props.values,
        props.extensiveValues,
        props.threshold
      ]
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rowCount, columnCount} = this;
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.weights.offsets,
      props.weights.neighbors,
      props.weights.weights,
      props.values,
      props.extensiveValues,
      props.threshold,
      output.labels,
      output.objective,
      output.status
    ]);
    const regionStride = columnCount + 2;
    const regions = createTransientView(graph, `${id}-regions`, 'float32', rowCount * regionStride);
    const state = createTransientView(graph, `${id}-state`, 'uint32', 5);
    const candidateScoreA = createTransientView(
      graph,
      `${id}-candidate-score-a`,
      'float32',
      rowCount
    );
    const candidateScoreB = createTransientView(
      graph,
      `${id}-candidate-score-b`,
      'float32',
      rowCount
    );
    const candidateDataA = createTransientView(
      graph,
      `${id}-candidate-data-a`,
      'uint32',
      2 * rowCount
    );
    const candidateDataB = createTransientView(
      graph,
      `${id}-candidate-data-b`,
      'uint32',
      2 * rowCount
    );
    const declarations = `const ROW_COUNT: u32 = ${rowCount}u;
const COLUMN_COUNT: u32 = ${columnCount}u;
const REGION_STRIDE: u32 = ${regionStride}u;
const SEED: u32 = ${props.seed}u;
const NONE: u32 = 0xffffffffu;
const LARGE_DISTANCE: f32 = 3.402823466e+38;
fn isFiniteFloat(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
fn tieKey(row: u32) -> u32 {
  var value = row ^ SEED;
  value = (value ^ (value >> 16u)) * 0x7feb352du;
  value = (value ^ (value >> 15u)) * 0x846ca68bu;
  return value ^ (value >> 16u);
}`;
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initialize`,
        operation: 'GPUMaxPRegions',
        variant: 'initialize',
        invocationCount: 1,
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read_write'},
          {name: 'objective', view: output.objective, type: 'f32', access: 'read_write'},
          {name: 'status', view: output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: 'const NONE: u32 = 0xffffffffu;',
        body: `for (var word = 0u; word < 5u; word++) { state[stateOffset + word] = 0u; }
  state[stateOffset + 4u] = NONE;
  objective[objectiveOffset] = 0.0;
  status[statusOffset] = 0u;
  status[statusOffset + 1u] = 0u;
  status[statusOffset + 2u] = 0u;
  status[statusOffset + 3u] = 0u;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-validate-threshold`,
        operation: 'GPUMaxPRegions',
        variant: 'validate-threshold',
        invocationCount: 1,
        bindings: [
          {name: 'threshold', view: props.threshold, type: 'f32', access: 'read'},
          {name: 'status', view: output.status, type: 'atomic<u32>', access: 'read_write'}
        ],
        declarations:
          'fn finite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }',
        body: `let value = threshold[thresholdOffset];
  if (!finite(value) || value < 0.0) { atomicAdd(&status[statusOffset + 3u], 1u); }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-validate-rows`,
        operation: 'GPUMaxPRegions',
        variant: 'parallel-validation',
        invocationCount: rowCount,
        bindings: [
          {name: 'offsets', view: props.weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: props.weights.neighbors, type: 'u32', access: 'read'},
          {name: 'values', view: props.values, type: 'f32', access: 'read'},
          {name: 'extensive', view: props.extensiveValues, type: 'f32', access: 'read'},
          {name: 'labels', view: output.labels, type: 'u32', access: 'read_write'},
          {name: 'regions', view: regions, type: 'f32', access: 'read_write'},
          {name: 'status', view: output.status, type: 'atomic<u32>', access: 'read_write'}
        ],
        declarations,
        body: `labels[labelsOffset + index] = NONE;
  let regionBase = regionsOffset + index * REGION_STRIDE;
  for (var slot = 0u; slot < REGION_STRIDE; slot++) { regions[regionBase + slot] = 0.0; }
  var invalid = 0u;
  let extensiveValue = extensive[extensiveOffset + index];
  invalid += select(0u, 1u, !isFiniteFloat(extensiveValue) || extensiveValue < 0.0);
  for (var column = 0u; column < COLUMN_COUNT; column++) {
    invalid += select(0u, 1u, !isFiniteFloat(values[valuesOffset + index * COLUMN_COUNT + column]));
  }
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    invalid += select(0u, 1u, neighbors[neighborsOffset + slot] >= ROW_COUNT);
  }
  if (invalid > 0u) { atomicAdd(&status[statusOffset + 3u], invalid); }`
      })
    ];

    const appendReduction = (prefix: string) => {
      let activeCount = rowCount;
      let sourceScore = candidateScoreA;
      let sourceData = candidateDataA;
      let destinationScore = candidateScoreB;
      let destinationData = candidateDataB;
      let level = 0;
      while (activeCount > 1) {
        const nextCount = Math.ceil(activeCount / 2);
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${prefix}-reduce-${level}`,
            operation: 'GPUMaxPRegions',
            variant: 'deterministic-reduction',
            invocationCount: nextCount,
            bindings: [
              {name: 'scoreIn', view: sourceScore, type: 'f32', access: 'read'},
              {name: 'dataIn', view: sourceData, type: 'u32', access: 'read'},
              {name: 'scoreOut', view: destinationScore, type: 'f32', access: 'read_write'},
              {name: 'dataOut', view: destinationData, type: 'u32', access: 'read_write'}
            ],
            declarations: `const INPUT_COUNT: u32 = ${activeCount}u;`,
            body: `let left = 2u * index;
  let right = left + 1u;
  var winner = left;
  if (right < INPUT_COUNT) {
    let leftScore = scoreIn[scoreInOffset + left];
    let rightScore = scoreIn[scoreInOffset + right];
    let leftBase = dataInOffset + 2u * left;
    let rightBase = dataInOffset + 2u * right;
    let leftKey = dataIn[leftBase]; let rightKey = dataIn[rightBase];
    let leftValue = dataIn[leftBase + 1u]; let rightValue = dataIn[rightBase + 1u];
    if (rightScore < leftScore || (rightScore == leftScore &&
        (rightKey < leftKey || (rightKey == leftKey && rightValue < leftValue)))) { winner = right; }
  }
  scoreOut[scoreOutOffset + index] = scoreIn[scoreInOffset + winner];
  let sourceBase = dataInOffset + 2u * winner;
  let destinationBase = dataOutOffset + 2u * index;
  dataOut[destinationBase] = dataIn[sourceBase];
  dataOut[destinationBase + 1u] = dataIn[sourceBase + 1u];`
          })
        );
        activeCount = nextCount;
        [sourceScore, destinationScore] = [destinationScore, sourceScore];
        [sourceData, destinationData] = [destinationData, sourceData];
        level++;
      }
      return {score: sourceScore, data: sourceData};
    };

    for (let construction = 0; construction < rowCount; construction++) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-construction-${construction}-seed-candidates`,
          operation: 'GPUMaxPRegions',
          variant: 'parallel-seed-candidates',
          invocationCount: rowCount,
          bindings: [
            {name: 'labels', view: output.labels, type: 'u32', access: 'read'},
            {name: 'status', view: output.status, type: 'u32', access: 'read'},
            {name: 'scoreOut', view: candidateScoreA, type: 'f32', access: 'read_write'},
            {name: 'dataOut', view: candidateDataA, type: 'u32', access: 'read_write'}
          ],
          declarations,
          body: `let accepted = labels[labelsOffset + index] == NONE && status[statusOffset + 3u] == 0u;
  scoreOut[scoreOutOffset + index] = select(LARGE_DISTANCE, 0.0, accepted);
  let base = dataOutOffset + 2u * index;
  dataOut[base] = select(NONE, tieKey(index), accepted);
  dataOut[base + 1u] = select(NONE, index, accepted);`
        })
      );
      const seed = appendReduction(`${id}-construction-${construction}-seed`);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-construction-${construction}-seed-apply`,
          operation: 'GPUMaxPRegions',
          variant: 'apply-seed',
          invocationCount: 1,
          bindings: [
            {name: 'values', view: props.values, type: 'f32', access: 'read'},
            {name: 'extensive', view: props.extensiveValues, type: 'f32', access: 'read'},
            {name: 'labels', view: output.labels, type: 'u32', access: 'read_write'},
            {name: 'regions', view: regions, type: 'f32', access: 'read_write'},
            {name: 'state', view: state, type: 'u32', access: 'read_write'},
            {name: 'threshold', view: props.threshold, type: 'f32', access: 'read'},
            {name: 'data', view: seed.data, type: 'u32', access: 'read'}
          ],
          declarations,
          body: `let row = data[dataOffset + 1u];
  state[stateOffset + 2u] = select(0u, 1u, row != NONE);
  state[stateOffset + 4u] = NONE;
  if (row == NONE) { return; }
  let region = state[stateOffset];
  state[stateOffset + 1u] = region;
  labels[labelsOffset + row] = region;
  let base = regionsOffset + region * REGION_STRIDE;
  regions[base] = 1.0;
  regions[base + 1u] = extensive[extensiveOffset + row];
  for (var column = 0u; column < COLUMN_COUNT; column++) {
    regions[base + 2u + column] = values[valuesOffset + row * COLUMN_COUNT + column];
  }
  if (regions[base + 1u] >= threshold[thresholdOffset]) { state[stateOffset + 2u] = 2u; }`
        })
      );

      for (let growth = 0; growth < rowCount - 1; growth++) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-construction-${construction}-growth-${growth}-candidates`,
            operation: 'GPUMaxPRegions',
            variant: 'parallel-growth-candidates',
            invocationCount: rowCount,
            bindings: [
              {name: 'offsets', view: props.weights.offsets, type: 'u32', access: 'read'},
              {name: 'neighbors', view: props.weights.neighbors, type: 'u32', access: 'read'},
              {name: 'values', view: props.values, type: 'f32', access: 'read'},
              {name: 'labels', view: output.labels, type: 'u32', access: 'read'},
              {name: 'regions', view: regions, type: 'f32', access: 'read'},
              {name: 'state', view: state, type: 'u32', access: 'read'},
              {name: 'scoreOut', view: candidateScoreA, type: 'f32', access: 'read_write'},
              {name: 'dataOut', view: candidateDataA, type: 'u32', access: 'read_write'}
            ],
            declarations,
            body: `let region = state[stateOffset + 1u];
  let regionBase = regionsOffset + region * REGION_STRIDE;
  var accepted = state[stateOffset + 2u] == 1u && labels[labelsOffset + index] == NONE;
  var adjacent = false;
  if (accepted) {
    for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
      adjacent = adjacent || labels[labelsOffset + neighbors[neighborsOffset + slot]] == region;
    }
  }
  accepted = accepted && adjacent;
  var distance = LARGE_DISTANCE;
  if (accepted) {
    distance = 0.0;
    for (var column = 0u; column < COLUMN_COUNT; column++) {
      let difference = values[valuesOffset + index * COLUMN_COUNT + column] - regions[regionBase + 2u + column] / regions[regionBase];
      distance += difference * difference;
    }
  }
  scoreOut[scoreOutOffset + index] = distance;
  let base = dataOutOffset + 2u * index;
  dataOut[base] = select(NONE, tieKey(index), accepted);
  dataOut[base + 1u] = select(NONE, index, accepted);`
          })
        );
        const growthCandidate = appendReduction(
          `${id}-construction-${construction}-growth-${growth}`
        );
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-construction-${construction}-growth-${growth}-apply`,
            operation: 'GPUMaxPRegions',
            variant: 'apply-growth',
            invocationCount: 1,
            bindings: [
              {name: 'values', view: props.values, type: 'f32', access: 'read'},
              {name: 'extensive', view: props.extensiveValues, type: 'f32', access: 'read'},
              {name: 'labels', view: output.labels, type: 'u32', access: 'read_write'},
              {name: 'regions', view: regions, type: 'f32', access: 'read_write'},
              {name: 'state', view: state, type: 'u32', access: 'read_write'},
              {name: 'threshold', view: props.threshold, type: 'f32', access: 'read'},
              {name: 'data', view: growthCandidate.data, type: 'u32', access: 'read'}
            ],
            declarations,
            body: `let row = data[dataOffset + 1u];
  if (row == NONE) { return; }
  let region = state[stateOffset + 1u];
  labels[labelsOffset + row] = region;
  let base = regionsOffset + region * REGION_STRIDE;
  regions[base] += 1.0;
  regions[base + 1u] += extensive[extensiveOffset + row];
  for (var column = 0u; column < COLUMN_COUNT; column++) {
    regions[base + 2u + column] += values[valuesOffset + row * COLUMN_COUNT + column];
  }
  if (regions[base + 1u] >= threshold[thresholdOffset]) { state[stateOffset + 2u] = 2u; }`
          })
        );
      }

      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-construction-${construction}-merge-candidates`,
          operation: 'GPUMaxPRegions',
          variant: 'parallel-enclave-candidates',
          invocationCount: rowCount,
          bindings: [
            {name: 'offsets', view: props.weights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: props.weights.neighbors, type: 'u32', access: 'read'},
            {name: 'labels', view: output.labels, type: 'u32', access: 'read'},
            {name: 'regions', view: regions, type: 'f32', access: 'read'},
            {name: 'state', view: state, type: 'u32', access: 'read'},
            {name: 'scoreOut', view: candidateScoreA, type: 'f32', access: 'read_write'},
            {name: 'dataOut', view: candidateDataA, type: 'u32', access: 'read_write'}
          ],
          declarations,
          body: `let current = state[stateOffset + 1u];
  let currentBase = regionsOffset + current * REGION_STRIDE;
  var bestRegion = NONE;
  var bestDistance = LARGE_DISTANCE;
  if (state[stateOffset + 2u] != 0u && labels[labelsOffset + index] == current) {
    for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
      let candidate = labels[labelsOffset + neighbors[neighborsOffset + slot]];
      if (candidate == NONE || candidate == current) { continue; }
      let candidateBase = regionsOffset + candidate * REGION_STRIDE;
      var distance = 0.0;
      for (var column = 0u; column < COLUMN_COUNT; column++) {
        let difference = regions[currentBase + 2u + column] / regions[currentBase] -
          regions[candidateBase + 2u + column] / regions[candidateBase];
        distance += difference * difference;
      }
      if (distance < bestDistance || (distance == bestDistance && candidate < bestRegion)) {
        bestDistance = distance; bestRegion = candidate;
      }
    }
  }
  scoreOut[scoreOutOffset + index] = bestDistance;
  let base = dataOutOffset + 2u * index;
  dataOut[base] = bestRegion;
  dataOut[base + 1u] = bestRegion;`
        })
      );
      const mergeCandidate = appendReduction(`${id}-construction-${construction}-merge`);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-construction-${construction}-finalize`,
          operation: 'GPUMaxPRegions',
          variant: 'finalize-region',
          invocationCount: 1,
          bindings: [
            {name: 'threshold', view: props.threshold, type: 'f32', access: 'read'},
            {name: 'regions', view: regions, type: 'f32', access: 'read_write'},
            {name: 'state', view: state, type: 'u32', access: 'read_write'},
            {name: 'data', view: mergeCandidate.data, type: 'u32', access: 'read'}
          ],
          declarations,
          body: `if (state[stateOffset + 2u] == 0u) { return; }
  let current = state[stateOffset + 1u];
  let currentBase = regionsOffset + current * REGION_STRIDE;
  let feasible = regions[currentBase + 1u] >= threshold[thresholdOffset];
  let mergeRegion = data[dataOffset + 1u];
  state[stateOffset + 4u] = NONE;
  if (!feasible && mergeRegion != NONE) {
    let mergeBase = regionsOffset + mergeRegion * REGION_STRIDE;
    for (var slot = 0u; slot < REGION_STRIDE; slot++) { regions[mergeBase + slot] += regions[currentBase + slot]; }
    state[stateOffset + 4u] = mergeRegion;
  } else {
    state[stateOffset] += 1u;
    if (!feasible) { state[stateOffset + 3u] += 1u; }
  }
  state[stateOffset + 2u] = 0u;`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-construction-${construction}-relabel-enclave`,
          operation: 'GPUMaxPRegions',
          variant: 'parallel-enclave-relabel',
          invocationCount: rowCount,
          bindings: [
            {name: 'labels', view: output.labels, type: 'u32', access: 'read_write'},
            {name: 'state', view: state, type: 'u32', access: 'read'}
          ],
          declarations: 'const NONE: u32 = 0xffffffffu;',
          body: `let mergeRegion = state[stateOffset + 4u];
  let current = state[stateOffset + 1u];
  if (mergeRegion != NONE && labels[labelsOffset + index] == current) { labels[labelsOffset + index] = mergeRegion; }`
        })
      );
    }

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-objective-rows`,
        operation: 'GPUMaxPRegions',
        variant: 'parallel-objective',
        invocationCount: rowCount,
        bindings: [
          {name: 'values', view: props.values, type: 'f32', access: 'read'},
          {name: 'labels', view: output.labels, type: 'u32', access: 'read'},
          {name: 'regions', view: regions, type: 'f32', access: 'read'},
          {name: 'scoreOut', view: candidateScoreA, type: 'f32', access: 'read_write'},
          {name: 'status', view: output.status, type: 'u32', access: 'read'}
        ],
        declarations,
        body: `var contribution = 0.0;
  if (status[statusOffset + 3u] == 0u) {
    let region = labels[labelsOffset + index];
    let base = regionsOffset + region * REGION_STRIDE;
    for (var column = 0u; column < COLUMN_COUNT; column++) {
      let difference = values[valuesOffset + index * COLUMN_COUNT + column] - regions[base + 2u + column] / regions[base];
      contribution += difference * difference;
    }
  }
  scoreOut[scoreOutOffset + index] = contribution;`
      })
    );
    let objectiveCount = rowCount;
    let objectiveScore = candidateScoreA;
    let objectiveDestination = candidateScoreB;
    let objectiveLevel = 0;
    while (objectiveCount > 1) {
      const nextCount = Math.ceil(objectiveCount / 2);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-objective-reduce-${objectiveLevel}`,
          operation: 'GPUMaxPRegions',
          variant: 'deterministic-objective-reduction',
          invocationCount: nextCount,
          bindings: [
            {name: 'scoreIn', view: objectiveScore, type: 'f32', access: 'read'},
            {name: 'scoreOut', view: objectiveDestination, type: 'f32', access: 'read_write'}
          ],
          declarations: `const INPUT_COUNT: u32 = ${objectiveCount}u;`,
          body: `let left = 2u * index;
  let right = left + 1u;
  var sum = scoreIn[scoreInOffset + left];
  if (right < INPUT_COUNT) { sum += scoreIn[scoreInOffset + right]; }
  scoreOut[scoreOutOffset + index] = sum;`
        })
      );
      objectiveCount = nextCount;
      [objectiveScore, objectiveDestination] = [objectiveDestination, objectiveScore];
      objectiveLevel++;
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-publish`,
        operation: 'GPUMaxPRegions',
        variant: 'publish',
        invocationCount: 1,
        bindings: [
          {name: 'score', view: objectiveScore, type: 'f32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'objective', view: output.objective, type: 'f32', access: 'read_write'},
          {name: 'status', view: output.status, type: 'u32', access: 'read_write'}
        ],
        body: `objective[objectiveOffset] = score[scoreOffset];
  status[statusOffset] = state[stateOffset];
  status[statusOffset + 1u] = select(0u, 1u, state[stateOffset + 3u] == 0u && status[statusOffset + 3u] == 0u);
  status[statusOffset + 2u] = state[stateOffset + 3u];`
      })
    );
    return nodes;
  }
}
