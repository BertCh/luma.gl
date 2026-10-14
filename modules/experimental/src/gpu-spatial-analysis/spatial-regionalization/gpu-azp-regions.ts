// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLDispatchGate, createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {getSortedSegmentOffsetNodes} from '../../utils/sorted-segment-offsets';
import {getSortKeyBits} from '../../utils/sorted-segment-sums';
import {
  getStorageBindingElementLimit,
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateMaximumCount,
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

export const GPU_AZP_MAXIMUM_COLUMNS = 8;
export const GPU_AZP_MAXIMUM_ITERATIONS = 256;
/**
 * Largest `rows * columnCount` of the attribute table and `regionCapacity * (columnCount + 1 [+ 1])`
 * of the region table of {@link GPUAZPRegions} on `device`: each is one storage binding of float32
 * words, so 33554432 on a default device (128 MiB bindings). Every other buffer is
 * `O(rows + slots)` words, so this and the slot count bound the problem; there is no separate row
 * cap.
 *
 * @param device Device whose `maxStorageBufferBindingSize` applies.
 */
export function getGPUAZPMaximumTableWordCount(device: Device): number {
  return getStorageBindingElementLimit(device, 4);
}
/**
 * Largest number of rows of one region's local neighborhood the removal test of a candidate move
 * explores (the row's same-region neighbors plus their same-region neighbors). A candidate whose
 * neighborhood is larger is conservatively not moved.
 */
export const GPU_AZP_LOCAL_NEIGHBORHOOD_CAPACITY = 128;
/**
 * Largest neighbor-list length of a row the removal test scans. A candidate whose neighborhood
 * contains a longer list (a hub) is conservatively not moved.
 */
export const GPU_AZP_MAXIMUM_SCAN_DEGREE = 64;
/**
 * A move is applied only when its computed gain exceeds this fraction of the magnitude of the four
 * float32 terms it is the difference of, so rounding noise cannot make two near-equal moves undo
 * each other forever. About 33 float32 ulps.
 */
const AZP_RELATIVE_GAIN_THRESHOLD = 4e-6;
const LOCAL_HASH_SIZE = 2 * GPU_AZP_LOCAL_NEIGHBORHOOD_CAPACITY;

/** Number of `uint32` words of {@link GPUAZPRegionsProps.parameters}: `[seed]`. */
export const GPU_AZP_REGIONS_PARAMETER_LENGTH = REGIONALIZATION_SEED_PARAMETER_LENGTH;

/** Per-frame settings of {@link GPUAZPRegions}, packed by {@link getGPUAZPRegionsParameterValues}. */
export type GPUAZPRegionsSettings = {
  /**
   * Reproducible tie-order seed, an integer in `[0, 2^32 - 1]`. Defaults to 0. Record the seed
   * used to create `initialLabels` next to it.
   */
  seed?: number;
};

/**
 * Packs {@link GPUAZPRegionsSettings} into the {@link GPU_AZP_REGIONS_PARAMETER_LENGTH} `uint32`
 * words of {@link GPUAZPRegionsProps.parameters}.
 */
export function getGPUAZPRegionsParameterValues(settings: GPUAZPRegionsSettings = {}): Uint32Array {
  return getRegionalizationSeedParameterValues('GPUAZPRegions', settings.seed);
}

/** Words of {@link GPUAZPRegionsOutput.status}. */
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
   * weights that declare `information.symmetric === true` are used as they are. At most
   * {@link getGPURegionalizationMaximumSlotCount} `neighbors` slots (16777216 on a default
   * device).
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
  /**
   * Rounds of batched moves, an integer in `[1, GPU_AZP_MAXIMUM_ITERATIONS]`. Each round applies
   * many moves at once, so far fewer rounds are needed than the single moves of earlier versions.
   */
  maximumIterations: number;
  /**
   * Per-frame `uint32` words ({@link GPU_AZP_REGIONS_PARAMETER_LENGTH}): the tie-order seed,
   * written with {@link getGPUAZPRegionsParameterValues}.
   */
  parameters: GraphDataView<'uint32'>;
  /** Caller-owned outputs. */
  output: GPUAZPRegionsOutput;
};

/** Caller-owned outputs of {@link GPUAZPRegions}. */
export type GPUAZPRegionsOutput = {
  /** Region label per row, starting from `initialLabels`. */
  labels: GraphDataView<'uint32'>;
  /** One value: the within-region sum of squared deviations of `labels`. */
  objective: GraphDataView<'float32'>;
  /** Four words indexed by {@link GPU_AZP_STATUS}. */
  status: GraphDataView<'uint32'>;
};

/**
 * Bounded deterministic greedy AZP with batched moves. This is the greedy variant of AZP (Openshaw
 * and Rao): it has no tabu list, simulated annealing or reactive escape, so it stops at the first
 * local optimum and the result depends entirely on `initialLabels`. For several starts, run
 * several instances from different initial partitions and keep the smallest `objective`.
 *
 * Starting from a caller-supplied contiguous partition (checked on the GPU: a disconnected region
 * is counted in `invalidCount`), each round every row finds its best improving move (the largest
 * reduction in total within-region squared error from leaving its region for an adjacent one) and
 * the round applies an independent set of them at once: every region takes part in at most one
 * selected move, chosen by gain and then by a stable seed-keyed row order, so the selected gains
 * add up exactly, the objective falls strictly each round, and the result is reproducible without
 * random GPU state. The best move of the round is always selected, so a round without any
 * improving move is the local optimum.
 *
 * Contiguity is kept by a local articulation test instead of a search of the whole source region:
 * a row may leave region A only when its A-neighbors stay connected to each other through rows of
 * A within two steps of it (the row's A-neighbors and their A-neighbors, at most
 * `GPU_AZP_LOCAL_NEIGHBORHOOD_CAPACITY` rows, scanning neighbor lists of at most
 * `GPU_AZP_MAXIMUM_SCAN_DEGREE` entries). The test is sufficient, never necessary: it can never
 * disconnect a region, but it may refuse a safe move whose reconnecting path is longer, so the
 * result is a local optimum with respect to the moves that pass it, not necessarily to every
 * contiguity-preserving move. Destination connectivity holds because a row only moves to the
 * region of one of its neighbors. Private memory and per-round work per row are bounded by those
 * two constants and the row's neighbor count, independent of the row count, so there is no row
 * cap: only the `rows x columnCount` attribute table and the region table must each fit one
 * storage binding (`getGPUAZPMaximumTableWordCount` words).
 *
 * `output.objective` is the within-region sum of squares of the returned labels, in standardized
 * units unless `standardize` is false (the same quantity as `withinSumOfSquares` of
 * `GPURegionPartitionEvaluation` on the same table). `status.iterationCount` counts the rounds
 * that applied at least one move (up to `maximumIterations`), not individual moves. Ties between
 * moves use a stable seed-keyed row order and then the destination label.
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
    const {output} = props;
    validateCapacity(id, 'columnCount', columnCount, 1, GPU_AZP_MAXIMUM_COLUMNS);
    validateCapacity(id, 'regionCapacity', props.regionCapacity, 2, rowCount);
    validateCapacity(
      id,
      'maximumIterations',
      props.maximumIterations,
      1,
      GPU_AZP_MAXIMUM_ITERATIONS
    );
    validateCapacity(id, 'minimumRegionSize', props.minimumRegionSize ?? 1);
    validateRegionalizationParameters(id, props.parameters, GPU_AZP_REGIONS_PARAMETER_LENGTH);
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
    if (output.status.length < GPU_AZP_STATUS.length)
      throw new Error(`${id} output.status must hold ${GPU_AZP_STATUS.length} words`);
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
    validateRegionalizationSlotCount(
      id,
      props.weights,
      getGPURegionalizationMaximumSlotCount(graph.device)
    );
    // The attribute table and the region record table are the widest bindings.
    const maximumTableWordCount = getGPUAZPMaximumTableWordCount(graph.device);
    validateMaximumCount(
      id,
      'values rows x columnCount',
      rowCount * columnCount,
      maximumTableWordCount
    );
    validateMaximumCount(
      id,
      'regionCapacity x region record words',
      props.regionCapacity * (columnCount + 1 + (props.extensiveValues ? 1 : 0)),
      maximumTableWordCount
    );
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
    const regions = createTransientView(
      graph,
      `${id}-regions`,
      'float32',
      props.regionCapacity * regionStride
    );
    const minimumRegionSize = props.minimumRegionSize ?? 1;
    // Word 0 is the stop flag and word 1 the round's progress flag. Under the extensive
    // constraint the next `rowCount` words hold the row extensive values and the last word the
    // threshold, as bit patterns of float32.
    const state = createTransientView(
      graph,
      `${id}-state`,
      'uint32',
      constrained ? rowCount + 3 : 2
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
    // One word of the best improving move of every row: [gain bits, destination]; gain 0 is none.
    const moves = createTransientView(graph, `${id}-moves`, 'uint32', 2 * rowCount);
    // Four per-region segments of `regionCapacity` words: the largest gain claiming the region, the
    // largest tie key among the moves holding that gain at both their regions, and the row of the
    // selected move that leaves (and of the one that joins) the region.
    const claims = createTransientView(graph, `${id}-claims`, 'uint32', 4 * props.regionCapacity);
    // Indirect dispatch triples of the gated round kernels (rule 13): slot 0 for the per-row
    // kernels, slot 1 for the per-region kernel. The one-thread round-finish kernel zeroes them at
    // the local optimum, so the remaining rounds launch only that kernel.
    const iterationGate = createWGSLDispatchGate(graph, {
      id: `${id}-iteration-dispatch`,
      operation: 'GPUAZPRegions',
      shapes: [[rowCount], [props.regionCapacity]]
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
    // Initial labels must be contiguous. Connected components of each region by hooking and
    // pointer jumping (Shiloach-Vishkin style) over the same-label edges: every dispatch is
    // multi-workgroup with O(rows + slots) work. A round hooks every root onto the smallest
    // adjacent root (an atomic min into a separate array, so the roots read in the round never
    // change under it), applies the hooks, and then halves the tree depths with
    // `ceil(log2(rows)) + 1` jumps, so all trees are stars again. Roots only hook onto smaller
    // roots, so a component's root is its smallest row. Rounds stop on the GPU when nothing
    // hooks. A final pass checks every same-label edge joins equal roots, so a run that did not
    // converge within the compile-time round bound counts as invalid input instead of passing;
    // each region's extra roots (components) count as invalid input as well.
    const contiguityRounds = 2 * Math.ceil(Math.log2(Math.max(rowCount, 2))) + 8;
    const contiguityJumps = Math.ceil(Math.log2(Math.max(rowCount, 2))) + 1;
    const contiguityHooks = createTransientView(
      graph,
      `${id}-contiguity-hooks`,
      'uint32',
      rowCount
    );
    const contiguityChanged = createTransientView(graph, `${id}-contiguity-changed`, 'uint32', 1);
    const contiguityGate = createWGSLDispatchGate(graph, {
      id: `${id}-contiguity-dispatch`,
      operation: 'GPUAZPRegions',
      shapes: [[rowCount]]
    });
    const contiguityDeclarations = `const ROWS: u32 = ${rowCount}u;
const REGION_CAPACITY: u32 = ${props.regionCapacity}u;`;
    const contiguityNodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-contiguity-initialize`,
        operation: 'GPUAZPRegions',
        variant: 'contiguity-initialize',
        invocationCount: rowCount,
        bindings: [
          {name: 'parent', view: componentA, type: 'u32', access: 'read_write'},
          {name: 'hooks', view: contiguityHooks, type: 'u32', access: 'read_write'}
        ],
        body: `parent[parentOffset + index] = index;
  hooks[hooksOffset + index] = index;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-contiguity-arm`,
        operation: 'GPUAZPRegions',
        variant: 'contiguity-arm',
        invocationCount: 1,
        bindings: [
          {name: 'changed', view: contiguityChanged, type: 'u32', access: 'read_write'},
          contiguityGate.binding
        ],
        body: `changed[changedOffset] = 0u;
  ${contiguityGate.armWGSL}`
      })
    ];
    for (let round = 0; round < contiguityRounds; round++) {
      contiguityNodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-contiguity-${round}-hook`,
          operation: 'GPUAZPRegions',
          variant: 'contiguity-hook',
          invocationCount: rowCount,
          ...contiguityGate.getGate(0),
          bindings: [
            {name: 'offsets', view: adjacency.weights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: adjacency.weights.neighbors, type: 'u32', access: 'read'},
            {name: 'labels', view: output.labels, type: 'u32', access: 'read'},
            {name: 'parent', view: componentA, type: 'u32', access: 'read'},
            {name: 'hooks', view: contiguityHooks, type: 'atomic<u32>', access: 'read_write'}
          ],
          declarations: contiguityDeclarations,
          body: `let label = labels[labelsOffset + index];
  if (label >= REGION_CAPACITY) { return; }
  let own = parent[parentOffset + index];
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= ROWS || labels[labelsOffset + neighbor] != label) { continue; }
    let other = parent[parentOffset + neighbor];
    if (other != own) {
      atomicMin(&hooks[hooksOffset + max(own, other)], min(own, other));
    }
  }`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-contiguity-${round}-apply`,
          operation: 'GPUAZPRegions',
          variant: 'contiguity-apply',
          invocationCount: rowCount,
          ...contiguityGate.getGate(0),
          bindings: [
            {name: 'parent', view: componentA, type: 'u32', access: 'read_write'},
            {name: 'hooks', view: contiguityHooks, type: 'u32', access: 'read_write'},
            {name: 'changed', view: contiguityChanged, type: 'atomic<u32>', access: 'read_write'}
          ],
          body: `let hookedRoot = hooks[hooksOffset + index];
  if (hookedRoot < index) {
    parent[parentOffset + index] = hookedRoot;
    hooks[hooksOffset + index] = index;
    atomicOr(&changed[changedOffset], 1u);
  }`
        }),
        // Ungated (it writes the gate): no hook means every region's trees are final.
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-contiguity-${round}-finish`,
          operation: 'GPUAZPRegions',
          variant: 'contiguity-finish-round',
          invocationCount: 1,
          bindings: [
            {name: 'changed', view: contiguityChanged, type: 'u32', access: 'read_write'},
            contiguityGate.binding
          ],
          body: `if (changed[changedOffset] == 0u) {
    ${contiguityGate.disarmWGSL}
  }
  changed[changedOffset] = 0u;`
        })
      );
      for (let jump = 0; jump < contiguityJumps; jump++) {
        contiguityNodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-contiguity-${round}-jump-${jump}`,
            operation: 'GPUAZPRegions',
            variant: 'contiguity-jump',
            invocationCount: rowCount,
            ...contiguityGate.getGate(0),
            bindings: [
              {name: 'parent', view: componentA, type: 'atomic<u32>', access: 'read_write'}
            ],
            // Every value a row reads or writes is an ancestor of the row, so racing jumps only
            // shorten the path.
            body: `let own = atomicLoad(&parent[parentOffset + index]);
  atomicStore(&parent[parentOffset + index], atomicLoad(&parent[parentOffset + own]));`
          })
        );
      }
    }
    contiguityNodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initial-contiguity`,
        operation: 'GPUAZPRegions',
        variant: 'initial-label-contiguity',
        invocationCount: rowCount,
        bindings: [
          {name: 'offsets', view: adjacency.weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: adjacency.weights.neighbors, type: 'u32', access: 'read'},
          {name: 'labels', view: output.labels, type: 'u32', access: 'read'},
          {name: 'parent', view: componentA, type: 'u32', access: 'read'},
          {name: 'sortedRows', view: sortedRows, type: 'u32', access: 'read'},
          {name: 'regionOffsets', view: regionOffsets, type: 'u32', access: 'read'},
          {name: 'status', view: output.status, type: 'atomic<u32>', access: 'read_write'}
        ],
        declarations: contiguityDeclarations,
        body: `let label = labels[labelsOffset + index];
  if (label >= REGION_CAPACITY) { return; }
  let root = parent[parentOffset + index];
  // A row is bad when the trees are not stars or a same-label edge joins different roots: the
  // round bound ran out before convergence, which is reported as invalid input.
  var invalid = select(0u, 1u, parent[parentOffset + root] != root);
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor < ROWS && labels[labelsOffset + neighbor] == label &&
        parent[parentOffset + neighbor] != root) {
      invalid = 1u;
    }
  }
  // Each region's smallest row is its component root; any other root is an extra component.
  if (root == index) {
    let first = sortedRows[sortedRowsOffset + regionOffsets[regionOffsetsOffset + label]];
    invalid += select(0u, 1u, first != index);
  }
  if (invalid > 0u) { atomicAdd(&status[statusOffset + 3u], invalid); }`
      })
    );

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
              operation: 'GPUAZPRegions',
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
              body: `// Words 2..ROWS + 1 hold row extensive values and word ROWS + 2 the threshold, as float bits.
  var value = threshold[thresholdOffset];
  if (index < ROWS) { value = extensive[extensiveOffset + index]; }
  if (!isFiniteFloat(value) || value < 0.0) { atomicAdd(&status[statusOffset + 3u], 1u); }
  state[stateOffset + 2u + index] = bitcast<u32>(value);`
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
        operation: 'GPUAZPRegions',
        groupCount: props.regionCapacity,
        sortedGroupKeys: sortedKeys,
        segmentOffsets: regionOffsets
      }),
      ...contiguityNodes,
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
    ${constrained ? 'regions[base + 1u + COLUMNS] += bitcast<f32>(state[stateOffset + 2u + row]);' : ''}
    for (var column = 0u; column < COLUMNS; column++) {
      regions[base + 1u + column] += values[valuesOffset + row * COLUMNS + column];
    }
  }
  regions[base] = f32(end - begin);`
      })
    ];

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initialize-state`,
        operation: 'GPUAZPRegions',
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

    const claimsDeclarations = `const ROWS: u32 = ${rowCount}u;
const REGION_CAPACITY: u32 = ${props.regionCapacity}u;
const NONE: u32 = 0xffffffffu;`;
    // Stable seed-keyed row order, a bijection of the rows (every step is invertible), so the keys
    // of distinct rows differ and the largest key of a set of moves picks exactly one of them.
    const tieKeyWGSL = `fn tieKey(row: u32) -> u32 {
  var value = row ^ parameters[parametersOffset];
  value = (value ^ (value >> 16u)) * 0x7feb352du;
  value = (value ^ (value >> 15u)) * 0x846ca68bu;
  return value ^ (value >> 16u);
}`;
    // WGSL of one claim stage of the move of the row `index`.
    const claimStageWGSL = (
      stage: 'gain' | 'key' | 'winner'
    ) => `let gain = moves[movesOffset + 2u * index];
  if (gain == 0u) { return; }
  let source = labels[labelsOffset + index];
  let destination = moves[movesOffset + 2u * index + 1u];
  ${
    stage === 'gain'
      ? `atomicMax(&claims[claimsOffset + source], gain);
  atomicMax(&claims[claimsOffset + destination], gain);`
      : `if (atomicLoad(&claims[claimsOffset + source]) != gain ||
      atomicLoad(&claims[claimsOffset + destination]) != gain) { return; }
  let key = tieKey(index);
  ${
    stage === 'key'
      ? `atomicMax(&claims[claimsOffset + REGION_CAPACITY + source], key);
  atomicMax(&claims[claimsOffset + REGION_CAPACITY + destination], key);`
      : `if (atomicLoad(&claims[claimsOffset + REGION_CAPACITY + source]) != key ||
      atomicLoad(&claims[claimsOffset + REGION_CAPACITY + destination]) != key) { return; }
  // The move holds the largest (gain, key) of both its regions, and the keys of distinct rows
  // differ, so no other move selected this round touches either region.
  atomicStore(&claims[claimsOffset + 2u * REGION_CAPACITY + source], index);
  atomicStore(&claims[claimsOffset + 3u * REGION_CAPACITY + destination], index);
  atomicOr(&state[stateOffset + 1u], 1u);`
  }`
  }`;
    const rowClaimBindings = [
      {name: 'moves', view: moves, type: 'u32' as const, access: 'read' as const},
      {name: 'labels', view: output.labels, type: 'u32' as const, access: 'read' as const},
      {name: 'claims', view: claims, type: 'atomic<u32>' as const, access: 'read_write' as const}
    ];

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initialize-claims`,
        operation: 'GPUAZPRegions',
        variant: 'initialize-claims',
        invocationCount: props.regionCapacity,
        bindings: [{name: 'claims', view: claims, type: 'u32', access: 'read_write'}],
        declarations: claimsDeclarations,
        body: `claims[claimsOffset + index] = 0u;
  claims[claimsOffset + REGION_CAPACITY + index] = 0u;
  claims[claimsOffset + 2u * REGION_CAPACITY + index] = NONE;
  claims[claimsOffset + 3u * REGION_CAPACITY + index] = NONE;`
      })
    );

    // Each round selects an independent set of improving moves and applies it at once.
    //
    // Why the batch is safe. Every move (row r leaves region A for the adjacent region B) is
    // evaluated against the round's start state and depends only on the members and moments of A
    // and B. The selection lets every region take part in at most one selected move, so when the
    // round is applied each region changes by exactly one row: (1) the gains of the selected moves
    // add up exactly, so the objective falls by their sum; (2) A minus r stays connected, because
    // the removal test below was run against the same A and A was not touched by another move;
    // (3) B plus r stays connected, because r has a neighbor in B and B was connected. Moved rows
    // need not be pairwise non-adjacent: a neighbor of r that moves leaves a region that r's own
    // regions do not include, so it changes none of the sets the two tests read.
    for (let iteration = 0; iteration < props.maximumIterations; iteration++) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-iteration-${iteration}-candidates`,
          operation: 'GPUAZPRegions',
          variant: 'local-articulation-candidates',
          invocationCount: rowCount,
          ...iterationGate.getGate(0),
          bindings: [
            {name: 'offsets', view: adjacency.weights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: adjacency.weights.neighbors, type: 'u32', access: 'read'},
            {name: 'values', view: table, type: 'f32', access: 'read'},
            {name: 'labels', view: output.labels, type: 'u32', access: 'read'},
            {name: 'regions', view: regions, type: 'f32', access: 'read'},
            {name: 'state', view: state, type: 'u32', access: 'read'},
            {name: 'moves', view: moves, type: 'u32', access: 'read_write'}
          ],
          declarations: `${declarations}
const NOISE_THRESHOLD: f32 = ${AZP_RELATIVE_GAIN_THRESHOLD};
const LOCAL_CAPACITY: u32 = ${GPU_AZP_LOCAL_NEIGHBORHOOD_CAPACITY}u;
const LOCAL_HASH_SIZE: u32 = ${LOCAL_HASH_SIZE}u;
const MAXIMUM_SCAN_DEGREE: u32 = ${GPU_AZP_MAXIMUM_SCAN_DEGREE}u;
// Local neighborhood of the candidate row: ${4 * (2 * GPU_AZP_LOCAL_NEIGHBORHOOD_CAPACITY + LOCAL_HASH_SIZE) + 16} bytes of private memory,
// independent of the row count. localRows lists the rows, localSlots is an open-addressing hash
// from row to its position in localRows, localStack and localVisited drive the search.
var<private> localRows: array<u32, ${GPU_AZP_LOCAL_NEIGHBORHOOD_CAPACITY}>;
var<private> localSlots: array<u32, ${LOCAL_HASH_SIZE}>;
var<private> localStack: array<u32, ${GPU_AZP_LOCAL_NEIGHBORHOOD_CAPACITY}>;
var<private> localVisited: array<u32, ${GPU_AZP_LOCAL_NEIGHBORHOOD_CAPACITY / 32}>;
var<private> localCount: u32;
fn getLocalBucket(vertex: u32) -> u32 { return (vertex * 2654435761u) >> 24u; }
// Position of \`vertex\` in localRows, or NONE.
fn findLocal(vertex: u32) -> u32 {
  var bucket = getLocalBucket(vertex);
  for (var probe = 0u; probe < LOCAL_HASH_SIZE; probe++) {
    let entry = localSlots[bucket];
    if (entry == NONE) { return NONE; }
    if (localRows[entry] == vertex) { return entry; }
    bucket = (bucket + 1u) & (LOCAL_HASH_SIZE - 1u);
  }
  return NONE;
}
// Adds \`vertex\` once. Returns false when the list is full.
fn addLocal(vertex: u32) -> bool {
  var bucket = getLocalBucket(vertex);
  for (var probe = 0u; probe < LOCAL_HASH_SIZE; probe++) {
    let entry = localSlots[bucket];
    if (entry == NONE) {
      if (localCount >= LOCAL_CAPACITY) { return false; }
      localRows[localCount] = vertex;
      localSlots[bucket] = localCount;
      localCount++;
      return true;
    }
    if (localRows[entry] == vertex) { return true; }
    bucket = (bucket + 1u) & (LOCAL_HASH_SIZE - 1u);
  }
  return false;
}
// Local articulation test. Returns true only when region \`source\` minus \`row\` is certainly
// connected; false means "not proven", never "disconnected".
//
// Proof. The source region A is connected (the initial labels are checked, and every applied move
// keeps it so). Let N be the rows of A adjacent to \`row\`. Take any v in A minus row: a path in A
// leads from v to row; its last step before the first visit to row lies in N, so v is connected
// inside A minus row to some member of N. Hence A minus row is connected when all members of N
// lie in one component of A minus row. The search below builds S, N plus the rows of A adjacent
// to a member of N (all different from \`row\`), then explores the subgraph induced by S, which is a
// subgraph of A minus row, from one member of N. Reaching every member of N through it proves that
// they share a component of A minus row. A search that runs out of capacity or meets a neighbor
// list longer than MAXIMUM_SCAN_DEGREE answers false.
fn isRemovalSafe(row: u32, source: u32) -> bool {
  localCount = 0u;
  for (var bucket = 0u; bucket < LOCAL_HASH_SIZE; bucket++) { localSlots[bucket] = NONE; }
  for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor < ROWS && neighbor != row && labels[labelsOffset + neighbor] == source) {
      if (!addLocal(neighbor)) { return false; }
    }
  }
  let ringCount = localCount;
  // A row with no same-region neighbor cannot leave a connected region of two or more rows
  // connected (and cannot exist in one); a single neighbor makes the row a leaf.
  if (ringCount < 2u) { return ringCount == 1u; }
  for (var position = 0u; position < ringCount; position++) {
    let member = localRows[position];
    let begin = offsets[offsetsOffset + member];
    let end = offsets[offsetsOffset + member + 1u];
    if (end - begin > MAXIMUM_SCAN_DEGREE) { return false; }
    for (var slot = begin; slot < end; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor < ROWS && neighbor != row && labels[labelsOffset + neighbor] == source) {
        if (!addLocal(neighbor)) { return false; }
      }
    }
  }
  for (var word = 0u; word < ${GPU_AZP_LOCAL_NEIGHBORHOOD_CAPACITY / 32}u; word++) { localVisited[word] = 0u; }
  localVisited[0] = 1u;
  localStack[0] = 0u;
  var stackSize = 1u;
  var reachedRing = 1u;
  while (stackSize > 0u) {
    stackSize--;
    let current = localRows[localStack[stackSize]];
    let begin = offsets[offsetsOffset + current];
    let end = offsets[offsetsOffset + current + 1u];
    if (end - begin > MAXIMUM_SCAN_DEGREE) { return false; }
    for (var slot = begin; slot < end; slot++) {
      let found = findLocal(neighbors[neighborsOffset + slot]);
      if (found != NONE && (localVisited[found >> 5u] & (1u << (found & 31u))) == 0u) {
        localVisited[found >> 5u] |= 1u << (found & 31u);
        localStack[stackSize] = found;
        stackSize++;
        if (found < ringCount) { reachedRing++; }
      }
    }
    if (reachedRing == ringCount) { return true; }
  }
  return false;
}`,
          body: `let row = index;
  let source = labels[labelsOffset + row];
  let sourceCount = regionCount(source);
  var movable = sourceCount > MINIMUM_SIZE && state[stateOffset] == 0u;
  ${
    constrained
      ? `if (movable) {
    // The rest of the source region must still reach the extensive threshold.
    let rowExtensive = bitcast<f32>(state[stateOffset + 2u + row]);
    movable = regions[regionsOffset + source * REGION_STRIDE + 1u + COLUMNS] - rowExtensive >=
      bitcast<f32>(state[stateOffset + 2u + ROWS]);
  }`
      : ''
  }
  var bestDelta = LARGE_DELTA;
  var bestScale = 0.0;
  var bestDestination = NONE;
  // Cheap exact delta first: only an improving move pays for the removal test below.
  if (movable) {
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor >= ROWS) { continue; }
      let destination = labels[labelsOffset + neighbor];
      if (destination == source) { continue; }
      let destinationCount = regionCount(destination);
      var delta = 0.0;
      var scale = 0.0;
      for (var column = 0u; column < COLUMNS; column++) {
        let value = values[valuesOffset + row * COLUMNS + column];
        let sourceSum = regionSum(source, column);
        let destinationSum = regionSum(destination, column);
        let sourceBefore = sourceSum * sourceSum / f32(sourceCount);
        let destinationBefore = destinationSum * destinationSum / f32(destinationCount);
        let sourceAfter = (sourceSum - value) * (sourceSum - value) / f32(sourceCount - 1u);
        let destinationAfter = (destinationSum + value) * (destinationSum + value) / f32(destinationCount + 1u);
        delta += sourceBefore + destinationBefore - sourceAfter - destinationAfter;
        scale += sourceBefore + destinationBefore + sourceAfter + destinationAfter;
      }
      if (delta < bestDelta || (delta == bestDelta && destination < bestDestination)) {
        bestDelta = delta; bestScale = scale; bestDestination = destination;
      }
    }
  }
  var gainBits = 0u;
  // delta is the gain's negative: the sum of squares after minus before.
  if (bestDestination != NONE && bestDelta < -NOISE_THRESHOLD * bestScale && isRemovalSafe(row, source)) {
    gainBits = bitcast<u32>(-bestDelta);
  } else {
    bestDestination = NONE;
  }
  // Positive float32 bit patterns order like the values, so the claim stages compare bits.
  moves[movesOffset + 2u * row] = gainBits;
  moves[movesOffset + 2u * row + 1u] = bestDestination;`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-iteration-${iteration}-claim-gain`,
          operation: 'GPUAZPRegions',
          variant: 'claim-gain',
          invocationCount: rowCount,
          ...iterationGate.getGate(0),
          bindings: rowClaimBindings,
          declarations: claimsDeclarations,
          body: claimStageWGSL('gain')
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-iteration-${iteration}-claim-key`,
          operation: 'GPUAZPRegions',
          variant: 'claim-key',
          invocationCount: rowCount,
          ...iterationGate.getGate(0),
          bindings: [
            ...rowClaimBindings,
            {name: 'parameters', view: props.parameters, type: 'u32', access: 'read'}
          ],
          declarations: `${claimsDeclarations}\n${tieKeyWGSL}`,
          body: claimStageWGSL('key')
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-iteration-${iteration}-select`,
          operation: 'GPUAZPRegions',
          variant: 'select-moves',
          invocationCount: rowCount,
          ...iterationGate.getGate(0),
          bindings: [
            ...rowClaimBindings,
            {name: 'parameters', view: props.parameters, type: 'u32', access: 'read'},
            {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'}
          ],
          declarations: `${claimsDeclarations}\n${tieKeyWGSL}`,
          body: claimStageWGSL('winner')
        }),
        // Each region applies its own selected move, so no region word has two writers and no
        // float atomics are needed. The same pass resets the region's claim words.
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-iteration-${iteration}-apply`,
          operation: 'GPUAZPRegions',
          variant: 'apply-selected-moves',
          invocationCount: props.regionCapacity,
          ...iterationGate.getGate(1),
          bindings: [
            {name: 'moves', view: moves, type: 'u32', access: 'read'},
            {name: 'claims', view: claims, type: 'atomic<u32>', access: 'read_write'},
            {name: 'values', view: table, type: 'f32', access: 'read'},
            {name: 'labels', view: output.labels, type: 'u32', access: 'read_write'},
            {name: 'regions', view: regions, type: 'f32', access: 'read_write'},
            ...(constrained
              ? [{name: 'state', view: state, type: 'u32' as const, access: 'read' as const}]
              : [])
          ],
          declarations,
          body: `let region = index;
  let leavingRow = atomicLoad(&claims[claimsOffset + 2u * REGION_CAPACITY + region]);
  let joiningRow = atomicLoad(&claims[claimsOffset + 3u * REGION_CAPACITY + region]);
  let base = regionsOffset + region * REGION_STRIDE;
  if (leavingRow != NONE) {
    regions[base] -= 1.0;
    ${constrained ? 'regions[base + 1u + COLUMNS] -= bitcast<f32>(state[stateOffset + 2u + leavingRow]);' : ''}
    for (var column = 0u; column < COLUMNS; column++) {
      regions[base + 1u + column] -= values[valuesOffset + leavingRow * COLUMNS + column];
    }
    labels[labelsOffset + leavingRow] = moves[movesOffset + 2u * leavingRow + 1u];
  }
  if (joiningRow != NONE) {
    regions[base] += 1.0;
    ${constrained ? 'regions[base + 1u + COLUMNS] += bitcast<f32>(state[stateOffset + 2u + joiningRow]);' : ''}
    for (var column = 0u; column < COLUMNS; column++) {
      regions[base + 1u + column] += values[valuesOffset + joiningRow * COLUMNS + column];
    }
  }
  atomicStore(&claims[claimsOffset + region], 0u);
  atomicStore(&claims[claimsOffset + REGION_CAPACITY + region], 0u);
  atomicStore(&claims[claimsOffset + 2u * REGION_CAPACITY + region], NONE);
  atomicStore(&claims[claimsOffset + 3u * REGION_CAPACITY + region], NONE);`
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-iteration-${iteration}-finish`,
          operation: 'GPUAZPRegions',
          variant: 'finish-round',
          invocationCount: 1,
          bindings: [
            {name: 'state', view: state, type: 'u32', access: 'read_write'},
            {name: 'status', view: output.status, type: 'u32', access: 'read_write'},
            iterationGate.binding
          ],
          // Ungated (it writes the gate). A round that selected no move found the local optimum;
          // after that it only re-reads the stop flag.
          body: `if (state[stateOffset] != 0u) { return; }
  if (state[stateOffset + 1u] == 0u) {
    state[stateOffset] = 1u;
    ${iterationGate.disarmWGSL}
    return;
  }
  state[stateOffset + 1u] = 0u;
  status[statusOffset] += 1u;`
        })
      );
    }

    // The objective is recomputed from the final labels and region moments rather than
    // accumulated from the move gains, so it carries no per-round rounding drift.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-objective-rows`,
        operation: 'GPUAZPRegions',
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
    );
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
            view: output.objective,
            type: 'f32',
            access: 'read_write'
          },
          {name: 'status', view: output.status, type: 'u32', access: 'read'}
        ],
        body: `objectiveOut[objectiveOutOffset] = select(objectiveIn[objectiveInOffset], 0.0, status[statusOffset + 3u] > 0u);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finalize`,
        operation: 'GPUAZPRegions',
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
