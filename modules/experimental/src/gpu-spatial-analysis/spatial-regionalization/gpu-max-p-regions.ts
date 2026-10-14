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
import {getWorkgroupArgminWGSL} from './regionalization-wgsl';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {type GPUSpatialWeights, validateGPUSpatialWeights} from '../spatial-weights/index';

export const GPU_MAX_P_MAXIMUM_COLUMNS = 8;
const MAX_P_WORKGROUP_SIZE = 256;
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
 *
 * Scale limit: the construction is sequential and runs as one dispatch of one 256-lane
 * workgroup. Every growth step rescans all rows and their neighbor lists, so the dispatch does
 * about `rowCount * (rowCount + neighborCount) / 256` steps per lane. That is fine for thousands
 * of rows, but from roughly 50k-100k rows one dispatch can run for seconds and trip the operating
 * system's GPU watchdog (device loss). Partition larger inputs first, or use SKATER.
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
    // Words indexed by MAX_P_STATE: region count, infeasible count, and the invalid-input copy.
    const state = createTransientView(graph, `${id}-state`, 'uint32', 3);
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
        body: `state[stateOffset] = 0u;
  state[stateOffset + 1u] = 0u;
  state[stateOffset + 2u] = 0u;
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
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-capture-invalid`,
        operation: 'GPUMaxPRegions',
        variant: 'capture-invalid',
        invocationCount: 1,
        bindings: [
          {name: 'status', view: output.status, type: 'u32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read_write'}
        ],
        body: 'state[stateOffset + 2u] = status[statusOffset + 3u];'
      }),
      // The construction is inherently sequential: every growth step depends on the previous
      // one. One persistent workgroup runs the whole seed / grow / merge loop with parallel
      // strided candidate scans and shared-memory argmin reductions, instead of unrolling
      // O(rowCount^2) graph nodes of mostly empty full-size dispatches.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-construct`,
        operation: 'GPUMaxPRegions',
        variant: 'persistent-workgroup-construction',
        invocationCount: MAX_P_WORKGROUP_SIZE,
        workgroupSize: MAX_P_WORKGROUP_SIZE,
        guardIndex: false,
        bindings: [
          {name: 'offsets', view: props.weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: props.weights.neighbors, type: 'u32', access: 'read'},
          {name: 'values', view: props.values, type: 'f32', access: 'read'},
          {name: 'extensive', view: props.extensiveValues, type: 'f32', access: 'read'},
          {name: 'threshold', view: props.threshold, type: 'f32', access: 'read'},
          {name: 'labels', view: output.labels, type: 'u32', access: 'read_write'},
          {name: 'regions', view: regions, type: 'f32', access: 'read_write'},
          {name: 'state', view: state, type: 'u32', access: 'read_write'}
        ],
        declarations: `${declarations}
${getWorkgroupArgminWGSL(MAX_P_WORKGROUP_SIZE)}
// Running count, extensive sum and column sums of the region under construction.
var<workgroup> currentStats: array<f32, REGION_STRIDE>;
var<workgroup> broadcastWord: u32;`,
        body: `let lane = localInvocationIndex;
  if (lane == 0u) { broadcastWord = state[stateOffset + 2u]; }
  let invalid = workgroupUniformLoad(&broadcastWord);
  var regionCount = 0u;
  var infeasibleCount = 0u;
  if (invalid == 0u) {
    let thresholdValue = threshold[thresholdOffset];
    for (var construction = 0u; construction < ROW_COUNT; construction++) {
      var bestScore = LARGE_DISTANCE;
      var bestKey = NONE;
      var bestValue = NONE;
      for (var row = lane; row < ROW_COUNT; row += ${MAX_P_WORKGROUP_SIZE}u) {
        let key = tieKey(row);
        if (labels[labelsOffset + row] == NONE && isArgminBetter(0.0, key, row, bestScore, bestKey, bestValue)) {
          bestScore = 0.0; bestKey = key; bestValue = row;
        }
      }
      let seed = reduceArgmin(lane, bestScore, bestKey, bestValue);
      if (seed == NONE) { break; }
      let region = regionCount;
      if (lane == 0u) {
        labels[labelsOffset + seed] = region;
        currentStats[0] = 1.0;
        currentStats[1] = extensive[extensiveOffset + seed];
        for (var column = 0u; column < COLUMN_COUNT; column++) {
          currentStats[2u + column] = values[valuesOffset + seed * COLUMN_COUNT + column];
        }
        broadcastWord = select(1u, 2u, currentStats[1] >= thresholdValue);
      }
      var phase = workgroupUniformLoad(&broadcastWord);
      storageBarrier();
      // Grow while under threshold: the unassigned row adjacent to the region nearest its mean.
      loop {
        if (phase != 1u) { break; }
        bestScore = LARGE_DISTANCE; bestKey = NONE; bestValue = NONE;
        for (var row = lane; row < ROW_COUNT; row += ${MAX_P_WORKGROUP_SIZE}u) {
          if (labels[labelsOffset + row] != NONE) { continue; }
          var adjacent = false;
          for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
            if (labels[labelsOffset + neighbors[neighborsOffset + slot]] == region) { adjacent = true; break; }
          }
          if (!adjacent) { continue; }
          var distance = 0.0;
          for (var column = 0u; column < COLUMN_COUNT; column++) {
            let difference = values[valuesOffset + row * COLUMN_COUNT + column] - currentStats[2u + column] / currentStats[0];
            distance += difference * difference;
          }
          let key = tieKey(row);
          if (isArgminBetter(distance, key, row, bestScore, bestKey, bestValue)) {
            bestScore = distance; bestKey = key; bestValue = row;
          }
        }
        let grown = reduceArgmin(lane, bestScore, bestKey, bestValue);
        if (grown == NONE) { break; }
        if (lane == 0u) {
          labels[labelsOffset + grown] = region;
          currentStats[0] += 1.0;
          currentStats[1] += extensive[extensiveOffset + grown];
          for (var column = 0u; column < COLUMN_COUNT; column++) {
            currentStats[2u + column] += values[valuesOffset + grown * COLUMN_COUNT + column];
          }
          broadcastWord = select(1u, 2u, currentStats[1] >= thresholdValue);
        }
        phase = workgroupUniformLoad(&broadcastWord);
        storageBarrier();
      }
      if (lane == 0u) {
        let base = regionsOffset + region * REGION_STRIDE;
        for (var slot = 0u; slot < REGION_STRIDE; slot++) { regions[base + slot] = currentStats[slot]; }
      }
      // Nearest adjacent completed region by mean distance, ties to the lowest region ID.
      bestScore = LARGE_DISTANCE; bestKey = NONE; bestValue = NONE;
      for (var row = lane; row < ROW_COUNT; row += ${MAX_P_WORKGROUP_SIZE}u) {
        if (labels[labelsOffset + row] != region) { continue; }
        for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
          let candidate = labels[labelsOffset + neighbors[neighborsOffset + slot]];
          if (candidate == NONE || candidate == region) { continue; }
          let candidateBase = regionsOffset + candidate * REGION_STRIDE;
          var distance = 0.0;
          for (var column = 0u; column < COLUMN_COUNT; column++) {
            let difference = currentStats[2u + column] / currentStats[0] -
              regions[candidateBase + 2u + column] / regions[candidateBase];
            distance += difference * difference;
          }
          if (isArgminBetter(distance, candidate, candidate, bestScore, bestKey, bestValue)) {
            bestScore = distance; bestKey = candidate; bestValue = candidate;
          }
        }
      }
      let mergeRegion = reduceArgmin(lane, bestScore, bestKey, bestValue);
      if (phase != 2u && mergeRegion != NONE) {
        // Enclave: fold the under-threshold remainder into its nearest neighbor region and
        // reuse this region ID for the next seed.
        if (lane == 0u) {
          let mergeBase = regionsOffset + mergeRegion * REGION_STRIDE;
          for (var slot = 0u; slot < REGION_STRIDE; slot++) { regions[mergeBase + slot] += currentStats[slot]; }
        }
        for (var row = lane; row < ROW_COUNT; row += ${MAX_P_WORKGROUP_SIZE}u) {
          if (labels[labelsOffset + row] == region) { labels[labelsOffset + row] = mergeRegion; }
        }
      } else {
        regionCount += 1u;
        if (phase != 2u) { infeasibleCount += 1u; }
      }
      storageBarrier();
    }
  }
  if (lane == 0u) {
    state[stateOffset] = regionCount;
    state[stateOffset + 1u] = infeasibleCount;
  }`
      })
    ];

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
  status[statusOffset + 1u] = select(0u, 1u, state[stateOffset + 1u] == 0u && status[statusOffset + 3u] == 0u);
  status[statusOffset + 2u] = state[stateOffset + 1u];`
      })
    );
    return nodes;
  }
}
