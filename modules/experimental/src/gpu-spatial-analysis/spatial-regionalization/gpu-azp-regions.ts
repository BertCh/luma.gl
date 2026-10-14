// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {getSortedSegmentOffsetNodes} from '../../utils/sorted-segment-offsets';
import {getSortKeyBits} from '../../utils/sorted-segment-sums';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import type {GPUOptimizationStatusPort} from '../contracts/index';
import {type GPUSpatialWeights, validateGPUSpatialWeights} from '../spatial-weights/index';
import {IS_FINITE_FLOAT_WGSL} from '../../utils/wgsl-common';
import {createStandardizeColumnsNode, createSymmetricAdjacency} from './regionalization-inputs';

export const GPU_AZP_MAXIMUM_COLUMNS = 8;
export const GPU_AZP_MAXIMUM_ITERATIONS = 256;
/**
 * Largest supported row count. Each candidate invocation keeps a one-bit-per-row reachability
 * set in private memory, and WGSL caps private memory at 8192 bytes per shader.
 */
export const GPU_AZP_MAXIMUM_ROWS = 8192 * 8;
const AZP_REDUCTION_WORKGROUP_SIZE = 256;
export const GPU_AZP_STATUS = {
  iterationCount: 0,
  converged: 1,
  iterationLimitReached: 2,
  invalidCount: 3,
  length: 4
} as const;

export type GPUAZPRegionsProps = {
  id?: string;
  /**
   * Spatial weights that define adjacency. Two rows are adjacent when either lists the other, so
   * directed (for example k-nearest-neighbor) weights are symmetrized inside the contributor;
   * weights that declare `info.symmetric === true` are used as they are.
   */
  weights: GPUSpatialWeights;
  values: GraphDataView<'float32'>;
  columnCount?: number;
  /**
   * Standardize every column to mean 0 and population standard deviation 1 (divisor `n`, the
   * `StandardScaler` convention, over finite entries) before the search, so columns in different
   * units weigh equally. Defaults to true; pass false when `values` is already on a common scale.
   * `optimization.objective` is then in standardized units.
   */
  standardize?: boolean;
  /**
   * Optional non-negative extensive value per row. With `extensiveThreshold`, a row may only leave
   * its region when the rest of that region keeps an extensive sum of at least the threshold (the
   * max-p constraint). A region that is already below the threshold can receive rows but not lose
   * any. Both props are required together.
   */
  extensiveValues?: GraphDataView<'float32'>;
  /** One per-frame minimum extensive sum a region must keep after losing a row. */
  extensiveThreshold?: GraphDataView<'float32'>;
  /**
   * Contiguous initial labels over the (symmetrized) weights. Values must be below
   * `regionCapacity`. A region that is not connected is reported through `invalidCount` (the
   * search then does not run).
   */
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
 * Bounded deterministic greedy steepest-descent AZP. This is the greedy variant of AZP (Openshaw
 * and Rao): it has no tabu list, simulated annealing or reactive escape, so it stops at the first
 * local optimum and the result depends entirely on `initialLabels`. For several starts, run
 * several instances from different initial partitions and keep the smallest `objective`.
 *
 * Starting from a caller-supplied contiguous partition (checked on the GPU: a disconnected region
 * is counted in `invalidCount`), each iteration selects the greatest reduction in total
 * within-region squared error. `optimization.objective` is that within-region sum of squares of
 * the returned labels, in standardized units unless `standardize` is false (the same quantity as
 * `withinSsd` of `GPURegionPartitionEvaluation` on the same table). Every
 * candidate removal runs an exact reachability search over its source region; requiring a
 * destination neighbor then preserves destination connectivity as well.
 * Ties use a stable seed-keyed row order and then the destination label. The search therefore
 * remains reproducible without random GPU state. It stops at a local optimum or the explicit
 * iteration budget and publishes objective and convergence status.
 *
 * Default change: columns are standardized (`standardize`, default true). Adjacency is the
 * undirected union of the weights and their transpose.
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
    if (rowCount > GPU_AZP_MAXIMUM_ROWS) {
      throw new Error(`${id} supports at most ${GPU_AZP_MAXIMUM_ROWS} rows`);
    }
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
    if (Boolean(props.extensiveValues) !== Boolean(props.extensiveThreshold))
      throw new Error(`${id} extensiveValues and extensiveThreshold are required together`);
    if (props.extensiveValues && props.extensiveThreshold) {
      validatePackedView(props.extensiveValues, ['float32'], `${id} extensiveValues`);
      validatePackedView(props.extensiveThreshold, ['float32'], `${id} extensiveThreshold`);
      if (props.extensiveValues.length !== rowCount)
        throw new Error(`${id} extensiveValues length must equal the weights row count`);
      if (props.extensiveThreshold.length < 1)
        throw new Error(`${id} extensiveThreshold must hold one value`);
    }
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
        props.initialLabels,
        props.extensiveValues,
        props.extensiveThreshold
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
      props.extensiveValues,
      props.extensiveThreshold,
      props.labels,
      props.optimization.objective,
      props.optimization.status
    ]);
    const constrained = Boolean(props.extensiveValues);
    // Region record: count, column sums and, under the extensive constraint, the extensive sum.
    const regionStride = columnCount + 1 + (constrained ? 1 : 0);
    const standardize = props.standardize !== false;
    const table = standardize
      ? createTransientView(graph, `${id}-table`, 'float32', rowCount * columnCount)
      : props.values;
    const adjacency = createSymmetricAdjacency(graph, {
      id,
      operation: 'GPUAZPRegions',
      weights: props.weights,
      rowCount
    });
    const componentA = createTransientView(graph, `${id}-component-a`, 'uint32', rowCount);
    const componentB = createTransientView(graph, `${id}-component-b`, 'uint32', rowCount);
    const reachedWordCount = Math.ceil(rowCount / 32);
    const regions = createTransientView(
      graph,
      `${id}-regions`,
      'float32',
      props.regionCapacity * regionStride
    );
    const minimumRegionSize = props.minimumRegionSize ?? 1;
    // Word 0 is the stop flag. Under the extensive constraint the next `rowCount` words hold the
    // row extensive values and the last word the threshold, as bit patterns of float32.
    const state = createTransientView(
      graph,
      `${id}-state`,
      'uint32',
      constrained ? rowCount + 2 : 1
    );
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
    // Rows stably sorted by initial label, so each region's members form one segment in ascending
    // row order. Labels at or above regionCapacity sort last and belong to no segment.
    const sortKeys = createTransientView(graph, `${id}-label-sort-keys`, 'uint32', rowCount);
    const sortIndices = createTransientView(graph, `${id}-label-sort-indices`, 'uint32', rowCount);
    const sortedKeys = createTransientView(graph, `${id}-label-sorted-keys`, 'uint32', rowCount);
    const sortedRows = createTransientView(graph, `${id}-label-sorted-rows`, 'uint32', rowCount);
    const regionOffsets = createTransientView(
      graph,
      `${id}-region-offsets`,
      'uint32',
      props.regionCapacity + 1
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
${IS_FINITE_FLOAT_WGSL}
fn regionCount(region: u32) -> u32 { return u32(regions[regionsOffset + region * REGION_STRIDE]); }
fn regionSum(region: u32, column: u32) -> f32 { return regions[regionsOffset + region * REGION_STRIDE + 1u + column]; }`;
    const nodes: GPUCommandNode<Parameters>[] = [
      ...adjacency.nodes,
      ...(standardize
        ? [
            createStandardizeColumnsNode<Parameters>(graph, {
              id: `${id}-standardize`,
              operation: 'GPUAZPRegions',
              values: props.values,
              scaled: table,
              rowCount,
              columnCount
            })
          ]
        : []),
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
      ...(constrained
        ? [
            createWGSLKernelNode<Parameters>(graph, {
              id: `${id}-pack-extensive`,
              operation: 'GPUAZPRegions',
              variant: 'pack-extensive',
              invocationCount: rowCount + 1,
              bindings: [
                {name: 'extensive', view: props.extensiveValues!, type: 'f32', access: 'read'},
                {name: 'threshold', view: props.extensiveThreshold!, type: 'f32', access: 'read'},
                {name: 'state', view: state, type: 'u32', access: 'read_write'},
                {
                  name: 'status',
                  view: props.optimization.status,
                  type: 'atomic<u32>',
                  access: 'read_write'
                }
              ],
              declarations: `const ROWS: u32 = ${rowCount}u;
${IS_FINITE_FLOAT_WGSL}`,
              body: `// Slots 1..ROWS hold row extensive values and slot ROWS + 1 the threshold, as float bits.
  var value = threshold[thresholdOffset];
  if (index < ROWS) { value = extensive[extensiveOffset + index]; }
  if (!isFiniteFloat(value) || value < 0.0) { atomicAdd(&status[statusOffset + 3u], 1u); }
  state[stateOffset + 1u + index] = bitcast<u32>(value);`
            })
          ]
        : []),
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
          },
          {name: 'sortKeys', view: sortKeys, type: 'u32', access: 'read_write'},
          {name: 'sortIndices', view: sortIndices, type: 'u32', access: 'read_write'}
        ],
        declarations: `const ROWS: u32 = ${rowCount}u;
const COLUMNS: u32 = ${columnCount}u;
const REGION_CAPACITY: u32 = ${props.regionCapacity}u;
${IS_FINITE_FLOAT_WGSL}`,
        body: `let region = initialLabels[initialLabelsOffset + index];
  labels[labelsOffset + index] = region;
  sortKeys[sortKeysOffset + index] = min(region, REGION_CAPACITY);
  sortIndices[sortIndicesOffset + index] = index;
  var invalidCount = select(0u, 1u, region >= REGION_CAPACITY);
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    invalidCount += select(0u, 1u, neighbors[neighborsOffset + slot] >= ROWS);
  }
  for (var column = 0u; column < COLUMNS; column++) {
    invalidCount += select(0u, 1u, !isFiniteFloat(values[valuesOffset + index * COLUMNS + column]));
  }
  if (invalidCount > 0u) { atomicAdd(&status[statusOffset + 3u], invalidCount); }`
      }),
      ...new GPUSort({
        id: `${id}-label-sort`,
        keys: sortKeys,
        values: sortIndices,
        outputKeys: sortedKeys,
        outputValues: sortedRows,
        keyBits: getSortKeyBits(props.regionCapacity)
      }).getCommandNodes(graph),
      ...getSortedSegmentOffsetNodes<Parameters>(graph, {
        id: `${id}-regions`,
        operation: 'GPUAZPRegions',
        groupCount: props.regionCapacity,
        sortedGroupKeys: sortedKeys,
        segmentOffsets: regionOffsets
      }),
      // Initial labels must be contiguous: connected components of each region by min-label
      // propagation (ping-pong, so a round never reads its own writes). A region with more than
      // one component counts its extra components as invalid input.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initial-contiguity`,
        operation: 'GPUAZPRegions',
        variant: 'initial-label-contiguity',
        invocationCount: AZP_REDUCTION_WORKGROUP_SIZE,
        workgroupSize: AZP_REDUCTION_WORKGROUP_SIZE,
        guardIndex: false,
        bindings: [
          {name: 'offsets', view: adjacency.weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: adjacency.weights.neighbors, type: 'u32', access: 'read'},
          {name: 'labels', view: props.labels, type: 'u32', access: 'read'},
          {name: 'componentA', view: componentA, type: 'u32', access: 'read_write'},
          {name: 'componentB', view: componentB, type: 'u32', access: 'read_write'},
          {name: 'sortedRows', view: sortedRows, type: 'u32', access: 'read'},
          {name: 'regionOffsets', view: regionOffsets, type: 'u32', access: 'read'},
          {
            name: 'status',
            view: props.optimization.status,
            type: 'atomic<u32>',
            access: 'read_write'
          }
        ],
        declarations: `const ROWS: u32 = ${rowCount}u;
const REGION_CAPACITY: u32 = ${props.regionCapacity}u;
const LANES: u32 = ${AZP_REDUCTION_WORKGROUP_SIZE}u;
var<workgroup> changedAtomic: atomic<u32>;
var<workgroup> changedFlag: u32;
fn readComponent(fromA: bool, row: u32) -> u32 {
  if (fromA) { return componentA[componentAOffset + row]; }
  return componentB[componentBOffset + row];
}`,
        body: `let lane = localInvocationIndex;
  for (var row = lane; row < ROWS; row += LANES) {
    componentA[componentAOffset + row] = row;
    componentB[componentBOffset + row] = row;
  }
  storageBarrier();
  workgroupBarrier();
  var fromA = true;
  for (var round = 0u; round <= ROWS; round++) {
    for (var row = lane; row < ROWS; row += LANES) {
      let own = readComponent(fromA, row);
      var best = own;
      let label = labels[labelsOffset + row];
      if (label < REGION_CAPACITY) {
        for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
          let neighbor = neighbors[neighborsOffset + slot];
          if (neighbor < ROWS && labels[labelsOffset + neighbor] == label) {
            best = min(best, readComponent(fromA, neighbor));
          }
        }
        // The value is the smallest row id seen in the component, so jump through it too.
        best = min(best, readComponent(fromA, best));
      }
      if (best != own) { atomicOr(&changedAtomic, 1u); }
      if (fromA) { componentB[componentBOffset + row] = best; } else { componentA[componentAOffset + row] = best; }
    }
    storageBarrier();
    workgroupBarrier();
    if (lane == 0u) { changedFlag = atomicLoad(&changedAtomic); atomicStore(&changedAtomic, 0u); }
    let anyChanged = workgroupUniformLoad(&changedFlag);
    storageBarrier();
    fromA = !fromA;
    if (anyChanged == 0u) { break; }
  }
  // Each region's first row (smallest) is a component root; any other root is an extra component.
  var extra = 0u;
  for (var row = lane; row < ROWS; row += LANES) {
    let label = labels[labelsOffset + row];
    if (label < REGION_CAPACITY && readComponent(fromA, row) == row) {
      let first = sortedRows[sortedRowsOffset + regionOffsets[regionOffsetsOffset + label]];
      extra += select(0u, 1u, first != row);
    }
  }
  if (extra > 0u) { atomicAdd(&status[statusOffset + 3u], extra); }`
      }),
      // One pass over the rows in total: each region walks only its own segment. The sort is
      // stable, so every region still sums its members in ascending row order, exactly the float
      // order of the former full scan per region (the search below depends on exact sums).
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-region-statistics`,
        operation: 'GPUAZPRegions',
        variant: 'segmented-region-statistics',
        invocationCount: props.regionCapacity,
        bindings: [
          {name: 'sortedRows', view: sortedRows, type: 'u32', access: 'read'},
          {name: 'regionOffsets', view: regionOffsets, type: 'u32', access: 'read'},
          {name: 'values', view: table, type: 'f32', access: 'read'},
          {name: 'regions', view: regions, type: 'f32', access: 'read_write'},
          ...(constrained
            ? [{name: 'state', view: state, type: 'u32' as const, access: 'read' as const}]
            : [])
        ],
        declarations,
        body: `let region = index;
  let base = regionsOffset + region * REGION_STRIDE;
  let begin = regionOffsets[regionOffsetsOffset + region];
  let end = regionOffsets[regionOffsetsOffset + region + 1u];
  for (var column = 0u; column < COLUMNS; column++) { regions[base + 1u + column] = 0.0; }
  ${constrained ? 'regions[base + 1u + COLUMNS] = 0.0;' : ''}
  for (var position = begin; position < end; position++) {
    let row = sortedRows[sortedRowsOffset + position];
    ${constrained ? 'regions[base + 1u + COLUMNS] += bitcast<f32>(state[stateOffset + 1u + row]);' : ''}
    for (var column = 0u; column < COLUMNS; column++) {
      regions[base + 1u + column] += values[valuesOffset + row * COLUMNS + column];
    }
  }
  regions[base] = f32(end - begin);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-objective-rows`,
        operation: 'GPUAZPRegions',
        variant: 'parallel-objective',
        invocationCount: rowCount,
        bindings: [
          {name: 'labels', view: props.labels, type: 'u32', access: 'read'},
          {name: 'values', view: table, type: 'f32', access: 'read'},
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
            {name: 'offsets', view: adjacency.weights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: adjacency.weights.neighbors, type: 'u32', access: 'read'},
            {name: 'values', view: table, type: 'f32', access: 'read'},
            {name: 'labels', view: props.labels, type: 'u32', access: 'read'},
            {name: 'regions', view: regions, type: 'f32', access: 'read'},
            {name: 'state', view: state, type: 'u32', access: 'read'},
            {name: 'deltaOut', view: candidateDeltaA, type: 'f32', access: 'read_write'},
            {name: 'dataOut', view: candidateDataA, type: 'u32', access: 'read_write'}
          ],
          declarations: `${declarations}
// One reachability bit per row: ${reachedWordCount * 4} bytes of private memory per invocation.
const REACHED_WORDS: u32 = ${reachedWordCount}u;
var<private> reached: array<u32, ${reachedWordCount}>;
fn isReached(vertex: u32) -> bool { return (reached[vertex >> 5u] & (1u << (vertex & 31u))) != 0u; }
fn markReached(vertex: u32) { reached[vertex >> 5u] |= 1u << (vertex & 31u); }`,
          body: `let row = index;
  let source = labels[labelsOffset + row];
  var sourceCount = 0u;
  if (state[stateOffset] == 0u) { sourceCount = regionCount(source); }
  var movable = sourceCount > MINIMUM_SIZE && state[stateOffset] == 0u;
  ${
    constrained
      ? `if (movable) {
    // The rest of the source region must still reach the extensive threshold.
    let rowExtensive = bitcast<f32>(state[stateOffset + 1u + row]);
    movable = regions[regionsOffset + source * REGION_STRIDE + 1u + COLUMNS] - rowExtensive >=
      bitcast<f32>(state[stateOffset + 1u + ROWS]);
  }`
      : ''
  }
  var bestDelta = LARGE_DELTA;
  var bestDestination = NONE;
  // Cheap exact delta first: only an improving move pays for the reachability search below.
  if (movable) {
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
  if (bestDelta < 0.0) {
    // Removing this row must leave the rest of the source region connected.
    var root = NONE;
    var last = 0u;
    for (var word = 0u; word < REACHED_WORDS; word++) { reached[word] = 0u; }
    for (var vertex = 0u; vertex < ROWS; vertex++) {
      if (vertex != row && labels[labelsOffset + vertex] == source) { root = min(root, vertex); last = vertex; }
    }
    var connected = root != NONE;
    if (connected) {
      markReached(root);
      for (var propagation = 0u; propagation < ROWS; propagation++) {
        var changed = false;
        for (var vertex = root; vertex <= last; vertex++) {
          if (!isReached(vertex) || vertex == row || labels[labelsOffset + vertex] != source) { continue; }
          for (var slot = offsets[offsetsOffset + vertex]; slot < offsets[offsetsOffset + vertex + 1u]; slot++) {
            let neighbor = neighbors[neighborsOffset + slot];
            if (neighbor != row && neighbor < ROWS && labels[labelsOffset + neighbor] == source && !isReached(neighbor)) {
              markReached(neighbor); changed = true;
            }
          }
        }
        if (!changed) { break; }
      }
      for (var vertex = root; vertex <= last; vertex++) {
        if (vertex != row && labels[labelsOffset + vertex] == source && !isReached(vertex)) { connected = false; }
      }
    }
    if (!connected) { bestDelta = LARGE_DELTA; bestDestination = NONE; }
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
        // One 256-lane workgroup argmin per block: log256 instead of log2 dispatches per move.
        const nextCount = Math.ceil(activeCount / AZP_REDUCTION_WORKGROUP_SIZE);
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-iteration-${iteration}-reduce-${reduction}`,
            operation: 'GPUAZPRegions',
            variant: 'deterministic-workgroup-candidate-reduction',
            invocationCount: nextCount * AZP_REDUCTION_WORKGROUP_SIZE,
            workgroupSize: AZP_REDUCTION_WORKGROUP_SIZE,
            guardIndex: false,
            bindings: [
              {name: 'deltaIn', view: sourceDelta, type: 'f32', access: 'read'},
              {name: 'dataIn', view: sourceData, type: 'u32', access: 'read'},
              {name: 'deltaOut', view: destinationDelta, type: 'f32', access: 'read_write'},
              {name: 'dataOut', view: destinationData, type: 'u32', access: 'read_write'}
            ],
            declarations: `const INPUT_COUNT: u32 = ${activeCount}u;
const WORKGROUP_SIZE: u32 = ${AZP_REDUCTION_WORKGROUP_SIZE}u;
const NONE: u32 = 0xffffffffu;
const LARGE_DELTA: f32 = 3.402823466e+38;
var<workgroup> sharedDelta: array<f32, ${AZP_REDUCTION_WORKGROUP_SIZE}>;
var<workgroup> sharedData: array<vec4<u32>, ${AZP_REDUCTION_WORKGROUP_SIZE}>;
// Candidate tuples are (row, destination, key) ordered by delta, key, row, then destination.
fn isBetter(delta: f32, data: vec4<u32>, bestDelta: f32, best: vec4<u32>) -> bool {
  return delta < bestDelta || (delta == bestDelta &&
    (data.z < best.z || (data.z == best.z &&
    (data.x < best.x || (data.x == best.x && data.y < best.y)))));
}`,
            // No early return: every invocation of a workgroup must reach the barriers.
            body: `let lane = localInvocationIndex;
  var delta = LARGE_DELTA;
  var data = vec4<u32>(NONE, NONE, NONE, 0u);
  if (index < INPUT_COUNT) {
    let base = dataInOffset + 3u * index;
    delta = deltaIn[deltaInOffset + index];
    data = vec4<u32>(dataIn[base], dataIn[base + 1u], dataIn[base + 2u], 0u);
  }
  sharedDelta[lane] = delta;
  sharedData[lane] = data;
  workgroupBarrier();
  for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (lane < stride && isBetter(sharedDelta[lane + stride], sharedData[lane + stride], sharedDelta[lane], sharedData[lane])) {
      sharedDelta[lane] = sharedDelta[lane + stride];
      sharedData[lane] = sharedData[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    let block = index / WORKGROUP_SIZE;
    let winner = sharedData[0];
    let destinationBase = dataOutOffset + 3u * block;
    deltaOut[deltaOutOffset + block] = sharedDelta[0];
    dataOut[destinationBase] = winner.x;
    dataOut[destinationBase + 1u] = winner.y;
    dataOut[destinationBase + 2u] = winner.z;
  }`
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
            {name: 'values', view: table, type: 'f32', access: 'read'},
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
  ${
    constrained
      ? `let rowExtensive = bitcast<f32>(state[stateOffset + 1u + row]);
  regions[sourceBase + 1u + COLUMNS] -= rowExtensive;
  regions[destinationBase + 1u + COLUMNS] += rowExtensive;`
      : ''
  }
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
