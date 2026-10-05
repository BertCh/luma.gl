// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUGroupAggregation,
  GPUScan,
  GPUSort,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {createMapGraphKernelNode, type MapGraphKernelBinding} from '../map-graph-kernels';
import {getMapGraphSortKeyBits} from '../map-graph-sorted-sums';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../map-graph-utils';
import {
  getInsertNeighborWGSL,
  getNearestNeighborSearchWGSL,
  getNeighborSearchLatticeWGSL,
  getRadiusNeighborLoopWGSL,
  NEIGHBOR_WEIGHT_WGSL
} from './neighbor-search-kernels';
import {GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH} from './neighbor-search-parameters';
import {type GPUSpatialWeights, validateGPUSpatialWeights} from './spatial-weights';

const OPERATION = 'GPUNeighborSearch';

/** Largest supported compile-time `k`. */
export const GPU_NEIGHBOR_SEARCH_MAXIMUM_K = 32;

/**
 * Properties for {@link GPUNeighborSearch}.
 *
 * Per-frame (no rebuild or recompile): the contents of `positions`, `queryPositions`, `mask`,
 * `queryMask` and `parameters` (bounds, radius, weight function, row standardization).
 * Compile-time: `mode`, `k`, `gridSize`, row counts, capacity and which optional views exist.
 */
export type GPUNeighborSearchProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'neighbor-search'`. */
  id?: string;
  /** `'knn'`: the `k` nearest targets per query. `'radius'`: every target within `radius`. */
  mode: 'knn' | 'radius';
  /** Packed planar target points. At least one and fewer than 2^31 rows. */
  positions: GraphDataView<'float32x2'>;
  /**
   * Optional packed planar query points (a cross join). When omitted the search is a self join of
   * `positions` that never lists a row as its own neighbor.
   */
  queryPositions?: GraphDataView<'float32x2'>;
  /** Optional target selection: nonzero includes the target. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Optional query selection: nonzero includes the query row; excluded rows are empty. A self join
   * without `queryMask` uses `mask` for the query rows too.
   */
  queryMask?: GraphDataView<'uint32'>;
  /**
   * Per-frame parameters: packed float32 view of at least `GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH`
   * elements written with `getGPUNeighborSearchParameterValues`.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Maximum `[columns, rows]` of the cell lattice. Compile-time. Results never depend on it, only
   * speed. kNN mode spreads exactly this many cells over the bounds.
   */
  gridSize: readonly [number, number];
  /** Neighbors per query in kNN mode, an integer in `[1, 32]`. Compile-time. Ignored in radius mode. */
  k?: number;
  /**
   * Caller-owned output CSR: `offsets` has `queryRows + 1` entries; the slot capacity is
   * `neighbors.length`. `distances` is optional. Row `i` lists neighbor target IDs ascending.
   */
  weights: GPUSpatialWeights;
  /** Caller-owned one-row flag: 1 when the neighbors did not fit the capacity, else 0. */
  overflow: GraphDataView<'uint32'>;
  /** Optional caller-owned one-row unclamped total neighbor count. */
  totalNeighbors?: GraphDataView<'uint32'>;
  /** Optional caller-owned unclamped neighbor count per query row. */
  neighborCounts?: GraphDataView<'uint32'>;
};

/**
 * Exact k-nearest-neighbor and distance-band queries between planar points, written as a
 * {@link GPUSpatialWeights} CSR: the reusable spatial-weights structure of the statistics recipes.
 *
 * Definition, which the GPU result matches exactly (IDs and offsets) or within f32 rounding
 * (distances, weights):
 * - A target is valid when its mask is nonzero, its coordinates are finite and it lies inside the
 *   per-frame bounds; queries likewise with `queryMask`. A self join skips the row itself.
 * - Distances are Euclidean, `d^2 = dx^2 + dy^2` in f32 with `dx = target - query`.
 * - kNN: the `k` valid targets with the smallest `(d^2, id)` (ties go to the lowest ID), optionally
 *   only those with `d <= radius`. Fewer valid targets give a shorter row.
 * - Radius: every valid target with `d <= radius`.
 * - Each row lists its neighbors by ascending ID, with `distances` and `weights` aligned.
 * - Weights: binary `1`; inverse distance `max(d, distanceFloor)^-power`; or a kernel `K(d / h)`
 *   with bandwidth `h = radius` (radius mode) or the row's k-th neighbor distance (kNN, PySAL's
 *   adaptive bandwidth). Non-finite weights are written as 0. Row standardization divides by the
 *   row sum (summed in slot order).
 * - Capacity: rows are laid out in query order; slots past `neighbors.length` are dropped. Offsets
 *   are clamped to the capacity, so a reader always sees consistent (possibly truncated) rows,
 *   and `overflow` is set. `totalNeighbors` and `neighborCounts` stay unclamped.
 *
 * Algorithm: targets are bucketed by a stable sort into a per-frame lattice (cells at least
 * `radius` wide in radius mode; `gridSize` cells over the bounds in kNN mode). kNN keeps a private
 * sorted top-k list per query and searches expanding cell rings until the k-th distance is
 * strictly inside the visited box. Radius mode counts, scans, emits and then insertion-sorts each
 * row by ID. No float atomics: every output is bitwise reproducible.
 */
export class GPUNeighborSearch implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'neighbor-search';
  /** Validated properties. */
  readonly props: GPUNeighborSearchProps;

  constructor(props: GPUNeighborSearchProps) {
    const id = props.id ?? 'neighbor-search';
    this.id = id;
    this.props = props;
    if (props.mode !== 'knn' && props.mode !== 'radius') {
      throw new Error(`${id} mode must be 'knn' or 'radius'`);
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    const targetRows = props.positions.length;
    if (targetRows < 1 || targetRows >= 2 ** 31) {
      throw new Error(`${id} positions must hold between 1 and 2^31 - 1 rows`);
    }
    if (props.queryPositions) {
      validatePackedView(props.queryPositions, ['float32x2'], `${id} queryPositions`);
      if (props.queryPositions.length < 1 || props.queryPositions.length >= 2 ** 31) {
        throw new Error(`${id} queryPositions must hold between 1 and 2^31 - 1 rows`);
      }
    }
    const queryRows = this.getQueryRows();
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH} float32 values`
      );
    }
    const [columns, rows] = props.gridSize;
    if (
      !Number.isInteger(columns) ||
      !Number.isInteger(rows) ||
      columns < 1 ||
      rows < 1 ||
      columns * rows >= 0xffffffff
    ) {
      throw new Error(
        `${id} gridSize must be two positive integers with columns * rows < 2^32 - 1`
      );
    }
    if (props.mode === 'knn') {
      const k = props.k;
      if (k === undefined || !Number.isInteger(k) || k < 1 || k > GPU_NEIGHBOR_SEARCH_MAXIMUM_K) {
        throw new Error(`${id} k must be an integer in [1, ${GPU_NEIGHBOR_SEARCH_MAXIMUM_K}]`);
      }
    }
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
      if (props.mask.length !== targetRows) {
        throw new Error(`${id} mask length must equal positions length`);
      }
    }
    if (props.queryMask) {
      validatePackedUint32View(props.queryMask, `${id} queryMask`);
      if (props.queryMask.length !== queryRows) {
        throw new Error(`${id} queryMask length must equal the query row count`);
      }
    }
    const weightRows = validateGPUSpatialWeights(id, props.weights);
    if (weightRows !== queryRows) {
      throw new Error(`${id} weights.offsets length must equal the query row count + 1`);
    }
    validatePackedUint32View(props.overflow, `${id} overflow`);
    if (props.overflow.length < 1) {
      throw new Error(`${id} overflow must hold one uint32`);
    }
    if (props.totalNeighbors) {
      validatePackedUint32View(props.totalNeighbors, `${id} totalNeighbors`);
      if (props.totalNeighbors.length < 1) {
        throw new Error(`${id} totalNeighbors must hold one uint32`);
      }
    }
    if (props.neighborCounts) {
      validatePackedUint32View(props.neighborCounts, `${id} neighborCounts`);
      if (props.neighborCounts.length !== queryRows) {
        throw new Error(`${id} neighborCounts length must equal the query row count`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        props.weights.offsets,
        props.weights.neighbors,
        props.weights.weights,
        props.weights.distances,
        props.overflow,
        props.totalNeighbors,
        props.neighborCounts
      ],
      [props.positions, props.queryPositions, props.mask, props.queryMask, props.parameters]
    );
  }

  /** Number of query rows: `queryPositions.length`, or `positions.length` for a self join. */
  getQueryRows(): number {
    return (this.props.queryPositions ?? this.props.positions).length;
  }

  /** Returns the neighbor-search nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {positions, queryPositions, parameters, gridSize, weights, mode} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      queryPositions,
      props.mask,
      props.queryMask,
      parameters,
      weights.offsets,
      weights.neighbors,
      weights.weights,
      weights.distances,
      props.overflow,
      props.totalNeighbors,
      props.neighborCounts
    ]);
    const targetRows = positions.length;
    const queryRows = this.getQueryRows();
    const capacity = weights.neighbors.length;
    const cellCount = gridSize[0] * gridSize[1];
    const k = props.k ?? 1;
    const latticeWGSL = getNeighborSearchLatticeWGSL(mode, gridSize);
    const crossJoin = Boolean(queryPositions);
    const queryBinding: MapGraphKernelBinding[] = queryPositions
      ? [{name: 'queryPositions', view: queryPositions, type: 'f32', access: 'read'}]
      : [];
    const queryName = crossJoin ? 'queryPositions' : 'positions';
    const readQueryWGSL = `let x = ${queryName}[${queryName}Offset + index * 2u];
  let y = ${queryName}[${queryName}Offset + index * 2u + 1u];`;
    const selfCondition = crossJoin ? 'true' : 'neighbor != index';
    // A self join applies the target mask to the query rows too unless a query mask is given.
    const queryMask = props.queryMask ?? (crossJoin ? undefined : props.mask);

    const cellKeys = createTransientView(graph, `${id}-cell-keys`, 'uint32', targetRows);
    const rowIds = createTransientView(graph, `${id}-row-ids`, 'uint32', targetRows);
    const cellCounts = createTransientView(graph, `${id}-cell-counts`, 'uint32', cellCount);
    const cellOffsets = createTransientView(graph, `${id}-cell-offsets`, 'uint32', cellCount + 1);
    const sortedKeys = createTransientView(graph, `${id}-sorted-keys`, 'uint32', targetRows);
    const sortedRows = createTransientView(graph, `${id}-sorted-rows`, 'uint32', targetRows);
    const counts =
      props.neighborCounts ?? createTransientView(graph, `${id}-counts`, 'uint32', queryRows);
    const starts = createTransientView(graph, `${id}-starts`, 'uint32', queryRows);
    const distances =
      weights.distances ?? createTransientView(graph, `${id}-distances`, 'float32', capacity);

    const nodes: GPUCommandNode<Parameters>[] = [
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-cell-keys`,
        operation: OPERATION,
        variant: 'cell-keys',
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
          ...(props.mask
            ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
            : []),
          {name: 'cellKeys', view: cellKeys, type: 'u32', access: 'read_write'},
          {name: 'rowIds', view: rowIds, type: 'u32', access: 'read_write'}
        ],
        invocationCount: targetRows,
        declarations: latticeWGSL,
        body: `let lattice = readLattice();
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let included = ${props.mask ? 'mask[maskOffset + index] != 0u' : 'true'};
  var key = CELL_COUNT;
  if (included && isPointValid(lattice, x, y)) {
    key = getCellRow(lattice, y) * lattice.columns + getCellColumn(lattice, x);
  }
  cellKeys[cellKeysOffset + index] = key;
  rowIds[rowIdsOffset + index] = index;`
      }),
      // Invalid targets carry the key CELL_COUNT, which the aggregation ignores and the sort puts last.
      ...new GPUGroupAggregation({
        id: `${id}-cell-counts`,
        keys: cellKeys,
        output: cellCounts
      }).getCommandNodes(graph),
      ...new GPUScan({
        id: `${id}-cell-scan`,
        input: cellCounts,
        output: cellOffsets,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-cell-total`,
        operation: OPERATION,
        variant: 'cell-total',
        bindings: [
          {name: 'counts', view: cellCounts, type: 'u32', access: 'read'},
          {name: 'cellOffsets', view: cellOffsets, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `const LAST_CELL: u32 = ${cellCount - 1}u;`,
        body: `cellOffsets[cellOffsetsOffset + LAST_CELL + 1u] =
    cellOffsets[cellOffsetsOffset + LAST_CELL] + counts[countsOffset + LAST_CELL];`
      }),
      ...new GPUSort({
        id: `${id}-cell-sort`,
        keys: cellKeys,
        values: rowIds,
        outputKeys: sortedKeys,
        outputValues: sortedRows,
        keyBits: getMapGraphSortKeyBits(cellCount)
      }).getCommandNodes(graph),
      // Query validity, stored in `counts` until the search pass overwrites it with the count.
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-query-valid`,
        operation: OPERATION,
        variant: 'query-valid',
        bindings: [
          ...(crossJoin
            ? queryBinding
            : [
                {name: 'positions', view: positions, type: 'f32' as const, access: 'read' as const}
              ]),
          {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
          ...(queryMask
            ? [
                {
                  name: 'queryMask',
                  view: queryMask,
                  type: 'u32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
        ],
        invocationCount: queryRows,
        declarations: latticeWGSL,
        body: `let lattice = readLattice();
  ${readQueryWGSL}
  let included = ${queryMask ? 'queryMask[queryMaskOffset + index] != 0u' : 'true'};
  counts[countsOffset + index] = select(0u, 1u, included && isPointValid(lattice, x, y));`
      })
    ];

    const searchBindings: MapGraphKernelBinding[] = [
      {name: 'positions', view: positions, type: 'f32', access: 'read'},
      ...queryBinding,
      {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
      {name: 'sortedRows', view: sortedRows, type: 'u32', access: 'read'},
      {name: 'cellOffsets', view: cellOffsets, type: 'u32', access: 'read'}
    ];

    let knnIds: GraphDataView<'uint32'> | undefined;
    let knnDistances: GraphDataView<'float32'> | undefined;
    if (mode === 'knn') {
      knnIds = createTransientView(graph, `${id}-knn-ids`, 'uint32', queryRows * k);
      knnDistances = createTransientView(graph, `${id}-knn-distances`, 'float32', queryRows * k);
      nodes.push(
        createMapGraphKernelNode<Parameters>(graph, {
          id: `${id}-knn`,
          operation: OPERATION,
          variant: 'knn',
          bindings: [
            ...searchBindings,
            {name: 'counts', view: counts, type: 'u32', access: 'read_write'},
            {name: 'knnIds', view: knnIds, type: 'u32', access: 'read_write'},
            {name: 'knnDistances', view: knnDistances, type: 'f32', access: 'read_write'}
          ],
          invocationCount: queryRows,
          declarations: `${latticeWGSL}
${getInsertNeighborWGSL(k)}`,
          body: `var found = 0u;
  var bestDistances: array<f32, ${k}>;
  var bestIds: array<u32, ${k}>;
  if (counts[countsOffset + index] != 0u) {
    let lattice = readLattice();
    ${readQueryWGSL}
    ${getNearestNeighborSearchWGSL(selfCondition)}
  }
  // Re-sort the selected neighbors by ID (insertion sort over at most K entries).
  for (var slot = 1u; slot < found; slot++) {
    let neighborId = bestIds[slot];
    let distanceSquared = bestDistances[slot];
    var destination = slot;
    while (destination > 0u && bestIds[destination - 1u] > neighborId) {
      bestIds[destination] = bestIds[destination - 1u];
      bestDistances[destination] = bestDistances[destination - 1u];
      destination--;
    }
    bestIds[destination] = neighborId;
    bestDistances[destination] = distanceSquared;
  }
  for (var slot = 0u; slot < found; slot++) {
    knnIds[knnIdsOffset + index * K + slot] = bestIds[slot];
    knnDistances[knnDistancesOffset + index * K + slot] = sqrt(bestDistances[slot]);
  }
  counts[countsOffset + index] = found;`
        })
      );
    } else {
      nodes.push(
        createMapGraphKernelNode<Parameters>(graph, {
          id: `${id}-count`,
          operation: OPERATION,
          variant: 'radius-count',
          bindings: [
            ...searchBindings,
            {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
          ],
          invocationCount: queryRows,
          declarations: latticeWGSL,
          body: `var count = 0u;
  if (counts[countsOffset + index] != 0u) {
    let lattice = readLattice();
    ${readQueryWGSL}
    ${getRadiusNeighborLoopWGSL(`if (${selfCondition}) {
          count++;
        }`)}
  }
  counts[countsOffset + index] = count;`
        })
      );
    }

    nodes.push(
      ...new GPUScan({
        id: `${id}-scan`,
        input: counts,
        output: starts,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-offsets`,
        operation: OPERATION,
        variant: 'offsets',
        bindings: [
          {name: 'counts', view: counts, type: 'u32', access: 'read'},
          {name: 'starts', view: starts, type: 'u32', access: 'read'},
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read_write'},
          {name: 'overflow', view: props.overflow, type: 'u32', access: 'read_write'},
          ...(props.totalNeighbors
            ? [
                {
                  name: 'totalNeighbors',
                  view: props.totalNeighbors,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: queryRows + 1,
        declarations: `const QUERY_ROWS: u32 = ${queryRows}u;
const CAPACITY: u32 = ${capacity}u;`,
        body: `if (index < QUERY_ROWS) {
    offsets[offsetsOffset + index] = min(starts[startsOffset + index], CAPACITY);
  } else {
    let total = starts[startsOffset + QUERY_ROWS - 1u] + counts[countsOffset + QUERY_ROWS - 1u];
    offsets[offsetsOffset + QUERY_ROWS] = min(total, CAPACITY);
    overflow[overflowOffset] = select(0u, 1u, total > CAPACITY);
    ${props.totalNeighbors ? 'totalNeighbors[totalNeighborsOffset] = total;' : ''}
  }`
      })
    );

    if (knnIds && knnDistances) {
      nodes.push(
        createMapGraphKernelNode<Parameters>(graph, {
          id: `${id}-emit`,
          operation: OPERATION,
          variant: 'knn-emit',
          bindings: [
            {name: 'knnIds', view: knnIds, type: 'u32', access: 'read'},
            {name: 'knnDistances', view: knnDistances, type: 'f32', access: 'read'},
            {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read_write'},
            {name: 'distances', view: distances, type: 'f32', access: 'read_write'}
          ],
          invocationCount: queryRows,
          declarations: `const K: u32 = ${k}u;`,
          body: `let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  for (var slot = begin; slot < end; slot++) {
    let source = index * K + slot - begin;
    neighbors[neighborsOffset + slot] = knnIds[knnIdsOffset + source];
    distances[distancesOffset + slot] = knnDistances[knnDistancesOffset + source];
  }`
        })
      );
    } else {
      nodes.push(
        createMapGraphKernelNode<Parameters>(graph, {
          id: `${id}-emit`,
          operation: OPERATION,
          variant: 'radius-emit',
          bindings: [
            ...searchBindings,
            {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read_write'},
            {name: 'distances', view: distances, type: 'f32', access: 'read_write'}
          ],
          invocationCount: queryRows,
          declarations: latticeWGSL,
          body: `let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  if (end > begin) {
    let lattice = readLattice();
    ${readQueryWGSL}
    var next = begin;
    ${getRadiusNeighborLoopWGSL(`if (${selfCondition} && next < end) {
          neighbors[neighborsOffset + next] = neighbor;
          distances[distancesOffset + next] = sqrt(distanceSquared);
          next++;
        }`)}
  }`
        }),
        createMapGraphKernelNode<Parameters>(graph, {
          id: `${id}-sort-rows`,
          operation: OPERATION,
          variant: 'radius-sort-rows',
          bindings: [
            {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read_write'},
            {name: 'distances', view: distances, type: 'f32', access: 'read_write'}
          ],
          invocationCount: queryRows,
          body: `let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  for (var slot = begin + 1u; slot < end; slot++) {
    let neighborId = neighbors[neighborsOffset + slot];
    let distance = distances[distancesOffset + slot];
    var destination = slot;
    while (destination > begin && neighbors[neighborsOffset + destination - 1u] > neighborId) {
      neighbors[neighborsOffset + destination] = neighbors[neighborsOffset + destination - 1u];
      distances[distancesOffset + destination] = distances[distancesOffset + destination - 1u];
      destination--;
    }
    neighbors[neighborsOffset + destination] = neighborId;
    distances[distancesOffset + destination] = distance;
  }`
        })
      );
    }

    const bandwidthWGSL =
      mode === 'radius'
        ? 'let bandwidth = readParameter(4u);'
        : `var bandwidth = 0.0;
  for (var slot = begin; slot < end; slot++) {
    bandwidth = max(bandwidth, distances[distancesOffset + slot]);
  }`;
    nodes.push(
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-weights`,
        operation: OPERATION,
        variant: 'weights',
        bindings: [
          {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
          {name: 'distances', view: distances, type: 'f32', access: 'read'},
          {name: 'weights', view: weights.weights, type: 'f32', access: 'read_write'}
        ],
        invocationCount: queryRows,
        declarations: `${latticeWGSL}
${NEIGHBOR_WEIGHT_WGSL}`,
        body: `let begin = offsets[offsetsOffset + index];
  let end = offsets[offsetsOffset + index + 1u];
  ${bandwidthWGSL}
  var sum = 0.0;
  for (var slot = begin; slot < end; slot++) {
    let weight = getNeighborWeight(distances[distancesOffset + slot], bandwidth);
    weights[weightsOffset + slot] = weight;
    sum += weight;
  }
  if (readParameter(9u) != 0.0 && sum > 0.0 && isFiniteFloat(sum)) {
    for (var slot = begin; slot < end; slot++) {
      weights[weightsOffset + slot] = weights[weightsOffset + slot] / sum;
    }
  }`
      })
    );
    return nodes;
  }
}
