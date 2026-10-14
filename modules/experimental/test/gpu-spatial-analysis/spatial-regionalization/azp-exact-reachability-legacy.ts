// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLDispatchGate, createWGSLKernelNode} from '../../../src/utils/wgsl-kernel-nodes';
import {getSortedSegmentOffsetNodes} from '../../../src/utils/sorted-segment-offsets';
import {getSortKeyBits} from '../../../src/utils/sorted-segment-sums';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../../src/utils/gpu-contributor-utils';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {IS_FINITE_FLOAT_WGSL} from '../../../src/utils/wgsl-common';
import {
  createStandardizeColumnsNode,
  createSymmetricAdjacency,
  getRegionalizationSeedParameterValues,
  REGIONALIZATION_SEED_PARAMETER_LENGTH,
  validateRegionalizationParameters
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/regionalization-inputs';

/**
 * Frozen copy of the pre-articulation `GPUAZPRegions` (exact per-candidate reachability, one move
 * per iteration) kept only as the A/B baseline of the local-articulation rewrite.
 */
export const LEGACY_AZP_MAXIMUM_COLUMNS = 8;
export const LEGACY_AZP_MAXIMUM_ITERATIONS = 256;
/**
 * Largest supported row count. Each improving candidate keeps a one-bit-per-row reachability set
 * in private memory (4,096 bytes at this cap, half of the WGSL 8,192-byte private limit, so the
 * kernel's other private values fit beside it) and clears and scans it once per iteration. The
 * planned local articulation test (algorithms roadmap P3 #3) removes the set and this cap.
 */
export const LEGACY_AZP_MAXIMUM_ROWS = 32768;
const AZP_REDUCTION_WORKGROUP_SIZE = 256;

/** Number of `uint32` words of {@link GPULegacyAZPRegionsProps.parameters}: `[seed]`. */
export const LEGACY_AZP_REGIONS_PARAMETER_LENGTH = REGIONALIZATION_SEED_PARAMETER_LENGTH;

/** Per-frame settings of {@link GPULegacyAZPRegions}, packed by {@link getGPULegacyAZPRegionsParameterValues}. */
export type GPULegacyAZPRegionsSettings = {
  /**
   * Reproducible tie-order seed, an integer in `[0, 2^32 - 1]`. Defaults to 0. Record the seed
   * used to create `initialLabels` next to it.
   */
  seed?: number;
};

/**
 * Packs {@link GPULegacyAZPRegionsSettings} into the {@link LEGACY_AZP_REGIONS_PARAMETER_LENGTH} `uint32`
 * words of {@link GPULegacyAZPRegionsProps.parameters}.
 */
export function getGPULegacyAZPRegionsParameterValues(
  settings: GPULegacyAZPRegionsSettings = {}
): Uint32Array {
  return getRegionalizationSeedParameterValues('GPULegacyAZPRegions', settings.seed);
}

/** Words of {@link GPULegacyAZPRegionsOutput.status}. */
export const LEGACY_AZP_STATUS = {
  iterationCount: 0,
  converged: 1,
  iterationLimitReached: 2,
  invalidCount: 3,
  length: 4
} as const;

export type GPULegacyAZPRegionsProps = {
  id?: string;
  /**
   * Spatial weights that define adjacency. Two rows are adjacent when either lists the other, so
   * directed (for example k-nearest-neighbor) weights are symmetrized inside the contributor;
   * weights that declare `information.symmetric === true` are used as they are. At most
   * `GPU_REGIONALIZATION_MAXIMUM_SLOT_COUNT` `neighbors` slots.
   */
  weights: GPUSpatialWeights;
  values: GraphDataView<'float32'>;
  columnCount?: number;
  /**
   * Standardize every column to mean 0 and population standard deviation 1 (divisor `n`, the
   * `StandardScaler` convention, over finite entries) before the search, so columns in different
   * units weigh equally. Defaults to true; pass false when `values` is already on a common scale.
   * `output.objective` is then in standardized units.
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
  /**
   * Per-frame `uint32` words ({@link LEGACY_AZP_REGIONS_PARAMETER_LENGTH}): the tie-order seed,
   * written with {@link getGPULegacyAZPRegionsParameterValues}.
   */
  parameters: GraphDataView<'uint32'>;
  /** Caller-owned outputs. */
  output: GPULegacyAZPRegionsOutput;
};

/** Caller-owned outputs of {@link GPULegacyAZPRegions}. */
export type GPULegacyAZPRegionsOutput = {
  /** Region label per row, starting from `initialLabels`. */
  labels: GraphDataView<'uint32'>;
  /** One value: the within-region sum of squared deviations of `labels`. */
  objective: GraphDataView<'float32'>;
  /** Four words indexed by {@link LEGACY_AZP_STATUS}. */
  status: GraphDataView<'uint32'>;
};

/**
 * Bounded deterministic greedy steepest-descent AZP. This is the greedy variant of AZP (Openshaw
 * and Rao): it has no tabu list, simulated annealing or reactive escape, so it stops at the first
 * local optimum and the result depends entirely on `initialLabels`. For several starts, run
 * several instances from different initial partitions and keep the smallest `objective`.
 *
 * Starting from a caller-supplied contiguous partition (checked on the GPU: a disconnected region
 * is counted in `invalidCount`), each iteration selects the greatest reduction in total
 * within-region squared error. `output.objective` is that within-region sum of squares of
 * the returned labels, in standardized units unless `standardize` is false (the same quantity as
 * `withinSumOfSquares` of `GPURegionPartitionEvaluation` on the same table). Every
 * candidate removal runs an exact reachability search over its source region; requiring a
 * destination neighbor then preserves destination connectivity as well.
 * Ties use a stable seed-keyed row order and then the destination label. The search therefore
 * remains reproducible without random GPU state. It stops at a local optimum or the explicit
 * iteration budget and publishes objective and convergence status.
 *
 * Default change: columns are standardized (`standardize`, default true). Adjacency is the
 * undirected union of the weights and their transpose.
 */
export class GPULegacyAZPRegions implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPULegacyAZPRegionsProps;
  readonly rowCount: number;
  readonly columnCount: number;

  constructor(props: GPULegacyAZPRegionsProps) {
    this.id = props.id ?? 'azp-regions';
    this.props = props;
    this.rowCount = validateGPUSpatialWeights(this.id, props.weights);
    this.columnCount = props.columnCount ?? 1;
    const {id, rowCount, columnCount} = this;
    const {output} = props;
    if (rowCount > LEGACY_AZP_MAXIMUM_ROWS) {
      throw new Error(`${id} supports at most ${LEGACY_AZP_MAXIMUM_ROWS} rows`);
    }
    validateCapacity(id, 'columnCount', columnCount, 1, LEGACY_AZP_MAXIMUM_COLUMNS);
    validateCapacity(id, 'regionCapacity', props.regionCapacity, 2, rowCount);
    validateCapacity(
      id,
      'maximumIterations',
      props.maximumIterations,
      1,
      LEGACY_AZP_MAXIMUM_ITERATIONS
    );
    validateCapacity(id, 'minimumRegionSize', props.minimumRegionSize ?? 1);
    validateRegionalizationParameters(id, props.parameters, LEGACY_AZP_REGIONS_PARAMETER_LENGTH);
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
      ['output.labels', output.labels],
      ['output.status', output.status]
    ] as const) {
      validatePackedView(view, ['uint32'], `${id} ${name}`);
    }
    if (props.initialLabels.length !== rowCount || output.labels.length !== rowCount)
      throw new Error(`${id} label views must equal the weights row count`);
    validatePackedView(output.objective, ['float32'], `${id} output.objective`);
    if (output.objective.length < 1) throw new Error(`${id} output.objective must hold one value`);
    if (output.status.length < LEGACY_AZP_STATUS.length)
      throw new Error(`${id} output.status must hold ${LEGACY_AZP_STATUS.length} words`);
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.labels, output.objective, output.status],
      [
        props.parameters,
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
    const {output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.parameters,
      props.weights.offsets,
      props.weights.neighbors,
      props.weights.weights,
      props.values,
      props.initialLabels,
      props.extensiveValues,
      props.extensiveThreshold,
      output.labels,
      output.objective,
      output.status
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
      operation: 'GPULegacyAZPRegions',
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
    // Candidate tuples (row, destination, tie key, delta bits): one binding instead of two, so the
    // apply kernel has room for the iteration gate within eight storage buffers.
    const candidateDataA = createTransientView(
      graph,
      `${id}-candidate-data-a`,
      'uint32',
      4 * rowCount
    );
    const candidateDataB = createTransientView(
      graph,
      `${id}-candidate-data-b`,
      'uint32',
      4 * rowCount
    );
    // Indirect dispatch triples of the gated iteration kernels (rule 13): slot 0 for the
    // candidates, slot 1 + level for each reduction level. The one-thread apply kernel zeroes them
    // at the local optimum, so the remaining iterations launch only that kernel.
    const reductionCounts: number[] = [];
    for (let activeCount = rowCount; activeCount > 1; ) {
      activeCount = Math.ceil(activeCount / AZP_REDUCTION_WORKGROUP_SIZE);
      reductionCounts.push(activeCount);
    }
    const iterationGate = createWGSLDispatchGate(graph, {
      id: `${id}-iteration-dispatch`,
      operation: 'GPULegacyAZPRegions',
      shapes: [
        [rowCount],
        ...reductionCounts.map(
          nextCount =>
            [nextCount * AZP_REDUCTION_WORKGROUP_SIZE, AZP_REDUCTION_WORKGROUP_SIZE] as const
        )
      ]
    });
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
const NONE: u32 = 0xffffffffu;
const LARGE_DELTA: f32 = 3.402823466e+38;
${IS_FINITE_FLOAT_WGSL}
fn regionCount(region: u32) -> u32 { return u32(regions[regionsOffset + region * REGION_STRIDE]); }
fn regionSum(region: u32, column: u32) -> f32 { return regions[regionsOffset + region * REGION_STRIDE + 1u + column]; }`;
    const nodes: GPUCommandNode<Parameters>[] = [
      ...adjacency.nodes,
      ...(standardize
        ? [
            createStandardizeColumnsNode<Parameters>(graph, {
              id: `${id}-standardize`,
              operation: 'GPULegacyAZPRegions',
              values: props.values,
              scaled: table,
              rowCount,
              columnCount
            })
          ]
        : []),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initialize-status`,
        operation: 'GPULegacyAZPRegions',
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
          {name: 'status', view: output.status, type: 'u32', access: 'read_write'}
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
              operation: 'GPULegacyAZPRegions',
              variant: 'pack-extensive',
              invocationCount: rowCount + 1,
              bindings: [
                {name: 'extensive', view: props.extensiveValues!, type: 'f32', access: 'read'},
                {name: 'threshold', view: props.extensiveThreshold!, type: 'f32', access: 'read'},
                {name: 'state', view: state, type: 'u32', access: 'read_write'},
                {
                  name: 'status',
                  view: output.status,
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
        operation: 'GPULegacyAZPRegions',
        variant: 'parallel-validation',
        invocationCount: rowCount,
        bindings: [
          {name: 'offsets', view: props.weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: props.weights.neighbors, type: 'u32', access: 'read'},
          {name: 'values', view: props.values, type: 'f32', access: 'read'},
          {name: 'initialLabels', view: props.initialLabels, type: 'u32', access: 'read'},
          {name: 'labels', view: output.labels, type: 'u32', access: 'read_write'},
          {
            name: 'status',
            view: output.status,
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
        operation: 'GPULegacyAZPRegions',
        groupCount: props.regionCapacity,
        sortedGroupKeys: sortedKeys,
        segmentOffsets: regionOffsets
      }),
      // Initial labels must be contiguous: connected components of each region by min-label
      // propagation (ping-pong, so a round never reads its own writes). A region with more than
      // one component counts its extra components as invalid input.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initial-contiguity`,
        operation: 'GPULegacyAZPRegions',
        variant: 'initial-label-contiguity',
        invocationCount: AZP_REDUCTION_WORKGROUP_SIZE,
        workgroupSize: AZP_REDUCTION_WORKGROUP_SIZE,
        guardIndex: false,
        bindings: [
          {name: 'offsets', view: adjacency.weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: adjacency.weights.neighbors, type: 'u32', access: 'read'},
          {name: 'labels', view: output.labels, type: 'u32', access: 'read'},
          {name: 'componentA', view: componentA, type: 'u32', access: 'read_write'},
          {name: 'componentB', view: componentB, type: 'u32', access: 'read_write'},
          {name: 'sortedRows', view: sortedRows, type: 'u32', access: 'read'},
          {name: 'regionOffsets', view: regionOffsets, type: 'u32', access: 'read'},
          {
            name: 'status',
            view: output.status,
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
        operation: 'GPULegacyAZPRegions',
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
        operation: 'GPULegacyAZPRegions',
        variant: 'parallel-objective',
        invocationCount: rowCount,
        bindings: [
          {name: 'labels', view: output.labels, type: 'u32', access: 'read'},
          {name: 'values', view: table, type: 'f32', access: 'read'},
          {name: 'regions', view: regions, type: 'f32', access: 'read'},
          {name: 'deltaOut', view: candidateDeltaA, type: 'f32', access: 'read_write'},
          {name: 'status', view: output.status, type: 'u32', access: 'read'}
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
          operation: 'GPULegacyAZPRegions',
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
        operation: 'GPULegacyAZPRegions',
        variant: 'objective-publish',
        invocationCount: 1,
        bindings: [
          {name: 'objectiveIn', view: objectiveSource, type: 'f32', access: 'read'},
          {
            name: 'objectiveOut',
            view: output.objective,
            type: 'f32',
            access: 'read_write'
          },
          {name: 'status', view: output.status, type: 'u32', access: 'read'}
        ],
        body: `objectiveOut[objectiveOutOffset] = select(objectiveIn[objectiveInOffset], 0.0, status[statusOffset + 3u] > 0u);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initialize-state`,
        operation: 'GPULegacyAZPRegions',
        variant: 'initialize-state',
        invocationCount: 1,
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read_write'},
          {name: 'status', view: output.status, type: 'u32', access: 'read'},
          iterationGate.binding
        ],
        // Invalid input never searches, so its gate starts closed.
        body: `let invalid = status[statusOffset + 3u] > 0u;
  state[stateOffset] = select(0u, 1u, invalid);
  if (invalid) {
    ${iterationGate.disarmWGSL}
  } else {
    ${iterationGate.armWGSL}
  }`
      })
    );

    for (let iteration = 0; iteration < props.maximumIterations; iteration++) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-iteration-${iteration}-candidates`,
          operation: 'GPULegacyAZPRegions',
          variant: 'exact-connectivity-candidates',
          invocationCount: rowCount,
          ...iterationGate.getGate(0),
          bindings: [
            {name: 'offsets', view: adjacency.weights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: adjacency.weights.neighbors, type: 'u32', access: 'read'},
            {name: 'values', view: table, type: 'f32', access: 'read'},
            {name: 'labels', view: output.labels, type: 'u32', access: 'read'},
            {name: 'regions', view: regions, type: 'f32', access: 'read'},
            {name: 'state', view: state, type: 'u32', access: 'read'},
            {name: 'dataOut', view: candidateDataA, type: 'u32', access: 'read_write'},
            {name: 'parameters', view: props.parameters, type: 'u32', access: 'read'}
          ],
          declarations: `${declarations}
// Stable seed-keyed row order; the seed is the per-frame parameter word.
fn tieKey(row: u32) -> u32 {
  var value = row ^ parameters[parametersOffset];
  value = (value ^ (value >> 16u)) * 0x7feb352du;
  value = (value ^ (value >> 15u)) * 0x846ca68bu;
  return value ^ (value >> 16u);
}
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
  let base = dataOutOffset + 4u * row;
  dataOut[base] = select(row, NONE, bestDestination == NONE);
  dataOut[base + 1u] = bestDestination;
  dataOut[base + 2u] = select(tieKey(row), NONE, bestDestination == NONE);
  dataOut[base + 3u] = bitcast<u32>(bestDelta);`
        })
      );

      let activeCount = rowCount;
      let sourceData = candidateDataA;
      let destinationData = candidateDataB;
      let reduction = 0;
      while (activeCount > 1) {
        // One 256-lane workgroup argmin per block: log256 instead of log2 dispatches per move.
        const nextCount = Math.ceil(activeCount / AZP_REDUCTION_WORKGROUP_SIZE);
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-iteration-${iteration}-reduce-${reduction}`,
            operation: 'GPULegacyAZPRegions',
            variant: 'deterministic-workgroup-candidate-reduction',
            invocationCount: nextCount * AZP_REDUCTION_WORKGROUP_SIZE,
            workgroupSize: AZP_REDUCTION_WORKGROUP_SIZE,
            guardIndex: false,
            ...iterationGate.getGate(1 + reduction),
            bindings: [
              {name: 'dataIn', view: sourceData, type: 'u32', access: 'read'},
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
    let base = dataInOffset + 4u * index;
    delta = bitcast<f32>(dataIn[base + 3u]);
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
    let blockIndex = index / WORKGROUP_SIZE;
    let winner = sharedData[0];
    let destinationBase = dataOutOffset + 4u * blockIndex;
    dataOut[destinationBase] = winner.x;
    dataOut[destinationBase + 1u] = winner.y;
    dataOut[destinationBase + 2u] = winner.z;
    dataOut[destinationBase + 3u] = bitcast<u32>(sharedDelta[0]);
  }`
          })
        );
        activeCount = nextCount;
        [sourceData, destinationData] = [destinationData, sourceData];
        reduction++;
      }

      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-iteration-${iteration}-apply`,
          operation: 'GPULegacyAZPRegions',
          variant: 'apply-exact-move',
          invocationCount: 1,
          bindings: [
            {name: 'data', view: sourceData, type: 'u32', access: 'read'},
            {name: 'values', view: table, type: 'f32', access: 'read'},
            {name: 'labels', view: output.labels, type: 'u32', access: 'read_write'},
            {name: 'regions', view: regions, type: 'f32', access: 'read_write'},
            {name: 'state', view: state, type: 'u32', access: 'read_write'},
            {
              name: 'objective',
              view: output.objective,
              type: 'f32',
              access: 'read_write'
            },
            {name: 'status', view: output.status, type: 'u32', access: 'read_write'},
            iterationGate.binding
          ],
          declarations: `${declarations}`,
          // Ungated (it writes the gate). After the local optimum it only re-reads the stop flag.
          body: `if (state[stateOffset] != 0u) { return; }
  let row = data[dataOffset];
  let destination = data[dataOffset + 1u];
  if (row == NONE || status[statusOffset + 3u] > 0u) {
    state[stateOffset] = 1u;
    ${iterationGate.disarmWGSL}
    return;
  }
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
  objective[objectiveOffset] += bitcast<f32>(data[dataOffset + 3u]);
  status[statusOffset] += 1u;`
        })
      );
    }

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finalize`,
        operation: 'GPULegacyAZPRegions',
        variant: 'finalize',
        invocationCount: 1,
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'status', view: output.status, type: 'u32', access: 'read_write'}
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
