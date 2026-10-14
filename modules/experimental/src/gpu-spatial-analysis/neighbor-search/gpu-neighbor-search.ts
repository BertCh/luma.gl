// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUGridIndex,
  GPUScan,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  getGeodesicNearestNeighborSearchWGSL,
  getGeodesicRadiusNeighborLoopWGSL,
  getInsertNeighborWGSL,
  getNearestNeighborSearchWGSL,
  getNeighborSearchLatticeWGSL,
  getRadiusNeighborLoopWGSL,
  NEIGHBOR_WEIGHT_WGSL
} from './neighbor-search-kernels';
import {GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH} from './neighbor-search-parameters';
import {
  describeGPUSpatialWeights,
  type GPUSpatialWeights,
  type GPUSpatialWeightsTransformState,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {GPU_SPATIAL_WEIGHTS_ADAPTIVE_BANDWIDTH_EPSILON} from '../spatial-weights/gpu-spatial-weights-transform';
import {
  GPU_EARTH_MEAN_RADIUS_METERS,
  resolveGPUSpatialContext,
  type GPUSpatialContext,
  type GPUSpatialContextSupport
} from '../contracts/index';

const OPERATION = 'GPUNeighborSearch';

/**
 * Spatial contexts {@link GPUNeighborSearch} accepts: native planar coordinates, or longitude/latitude
 * degrees with the `'great-circle'` metric. Pass it to `resolveGPUSpatialContext` to validate a
 * context before building the node.
 */
export const GPU_NEIGHBOR_SEARCH_GREAT_CIRCLE_SUPPORT: GPUSpatialContextSupport = Object.freeze({
  coordinateSpaces: Object.freeze(['planar', 'longitude-latitude'] as const),
  metrics: Object.freeze(['none', 'native', 'great-circle'] as const)
});

/** Weight transform states a caller can declare for {@link GPUNeighborSearchProps.weightTransform}. */
export type GPUNeighborSearchWeightTransform = Extract<
  GPUSpatialWeightsTransformState,
  'binary' | 'raw' | 'kernel' | 'row'
>;

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
  /**
   * Coordinate space and metric of `positions`. Compile-time. Omitted means native planar
   * coordinates with Euclidean distance (unchanged behavior). The great-circle path is
   * `{coordinateSpace: 'longitude-latitude', metric: 'great-circle', units?: 'meters' | 'kilometers',
   * sphereRadius?}`: positions are `[longitude, latitude]` degrees, `bounds` are
   * `[minLongitude, minLatitude, maxLongitude, maxLatitude]` degrees (longitude span at most 360),
   * `radius`, `distanceFloor` and the output `distances` are great-circle distances in `units`
   * (default meters), and `sphereRadius` is in meters (default `GPU_EARTH_MEAN_RADIUS_METERS`).
   * Any other space or metric throws; project to a planar frame first.
   */
  spatialContext?: GPUSpatialContext;
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
  /**
   * Optional caller-owned one-row count of rows that took part by mask but were dropped because a
   * coordinate is not finite or lies outside the per-frame `bounds` (or, for great-circle search,
   * has a latitude beyond +-90). Such rows are silently empty and never a neighbor, so a nonzero
   * count means `bounds` is too small or the data has missing coordinates. Counts target rows, plus
   * the query rows of a cross join (`queryPositions`). It is `outputWeights.status.invalidCount`.
   */
  invalidCount?: GraphDataView<'uint32'>;
  /**
   * Declares the weight function the per-frame parameters will select, so `outputWeights.info` can
   * say what the weights are. Weight kind and row standardization are per-frame parameters that
   * the node cannot read at construction, so this is metadata the caller keeps true (derive it from
   * the same parameters with `getGPUNeighborSearchWeightTransform`). Omitted means `'unknown'`.
   */
  weightTransform?: GPUNeighborSearchWeightTransform;
  /**
   * Compile-time row standardization. When given, it is the single source of truth: the weights
   * kernel uses it as a constant and ignores the per-frame `rowStandardize` parameter, and
   * `outputWeights.info.transform` is `'row'` for `true`, else `weightTransform` (or `'unknown'`).
   * Consumers re-standardize rows over the neighbors left after masking only when the transform is
   * `'row'`, so set this (rather than the per-frame field) for any row-standardized band. Passing
   * `true` with a `weightTransform` other than `'row'`, or `false` with `'row'`, throws. When
   * omitted, the per-frame `rowStandardize` parameter applies and `weightTransform` is the
   * caller's declaration.
   */
  rowStandardize?: boolean;
};

/**
 * Exact k-nearest-neighbor and distance-band queries between points, written as a
 * {@link GPUSpatialWeights} CSR: the reusable spatial-weights structure of the statistics contributors.
 * Points are planar (default) or longitude/latitude degrees with great-circle distances; see
 * {@link GPUNeighborSearchProps.spatialContext}.
 *
 * Definition, which the GPU result matches exactly (IDs and offsets) or within f32 rounding
 * (distances, weights):
 * - A target is valid when its mask is nonzero, its coordinates are finite and it lies inside the
 *   per-frame bounds; queries likewise with `queryMask`. A self join skips the row itself.
 * - Planar distances are Euclidean, `d^2 = dx^2 + dy^2` in f32 with `dx = target - query`. Planar
 *   f32 coordinates keep about 7 significant digits, so UTM northings (~4e6 m, ulp 0.5 m) or Web
 *   Mercator coordinates (~1e7, ulp 1 m) quantize `dx` and make ties and band edges noisy. Project
 *   to a local frame (`createGPULocalFrame`) or subtract an origin before upload; the node cannot
 *   recover precision lost in its f32 inputs.
 * - Great-circle distances (`spatialContext` longitude-latitude) are haversine/central-angle
 *   distances on a sphere (`GEODESIC_WGSL`, f32, the inputs' own f32 quantization is the limit:
 *   about 1 m at longitude 180) in `units`, and `radius` is in `units`. The lattice stays in
 *   degrees. Radius mode visits every cell that can hold a target in the cap: the latitude span is
 *   `radius / R`, the longitude span is the exact cap extent `asin(sin(a) / cos(latitude))` (the
 *   whole circle when the cap reaches a pole) and bands wrap across +-180. kNN visits expanding
 *   rings (wrapping columns when `bounds` span exactly 360 degrees) and stops only when a
 *   conservative spherical lower bound of the distance to every unvisited cell exceeds the K-th
 *   distance, so results do not depend on `gridSize`; a looser bound only costs speed (queries near
 *   the poles or narrow longitude spans scan more cells). Points with latitude beyond +-90 are invalid.
 * - Point and query rows that are not valid never take part. Report them with `invalidCount`.
 * - kNN: the `k` valid targets with the smallest `(d^2, id)` (ties go to the lowest ID), optionally
 *   only those with `d <= radius`. Fewer valid targets give a shorter row.
 * - Radius: every valid target with `d <= radius`.
 * - Each row lists its neighbors by ascending ID, with `distances` and `weights` aligned.
 * - Weights: binary `1`; inverse distance `max(d, distanceFloor)^-power`; or a kernel `K(d / h)`
 *   with bandwidth `h = radius` (radius mode) or the row's k-th neighbor distance (kNN, PySAL's
 *   adaptive bandwidth, `h_i` = the row's k-th neighbor distance times 1.0000001 =
 *   `GPU_SPATIAL_WEIGHTS_ADAPTIVE_BANDWIDTH_EPSILON`, as libpysal `Kernel(fixed=False)` after
 *   removing the self slot; a row with fewer than `k` neighbors uses its farthest listed neighbor).
 *   Non-finite weights are written as 0. Row standardization divides by the row sum (summed in
 *   slot order). Rows never list themselves: the kernel's self weight `K(0)` is the consumer's
 *   `selfWeight` (`getGPUSpatialWeightsKernelValue(kernel, 0)`). A slot whose weight is 0 (for
 *   example a triangular kernel at `d = radius`, or coincident points with inverse distance and no
 *   floor) is NOT a neighbor under the stream-wide definition: consumers skip zero-weight slots.
 * - `outputWeights` is `weights` with `info` (`rowCount`, `transform` from `weightTransform` or
 *   `'unknown'`; `'row'` when `rowStandardize: true`, `symmetric` true only for an unmasked-query
 *   planar radius self join with a declared unstandardized transform, else `'unknown'`, `provenance` criterion `'knn'` or `'distance-band'`
 *   with `k`) and `status` (`overflow`, `requiredCount` = `totalNeighbors`, `invalidCount`) attached.
 *   `'adaptive'` kernel transforms accept it because kNN output has criterion `'knn'`. For PySAL's
 *   `min_threshold_distance` (the smallest band keeping every unit connected), run kNN with `k = 1`
 *   and take the maximum of the returned `distances`.
 * - Capacity: rows are laid out in query order; slots past `neighbors.length` are dropped. Offsets
 *   are clamped to the capacity, so a reader always sees consistent (possibly truncated) rows,
 *   and `overflow` is set. `totalNeighbors` and `neighborCounts` stay unclamped.
 *
 * Algorithm: targets are bucketed by `GPUGridIndex` into a per-frame lattice (cells at least
 * `radius` wide in radius mode; `gridSize` cells over the bounds in kNN mode), whose domain is
 * read from a buffer so changing bounds or radius never recompiles. kNN keeps a private sorted
 * top-k list per query and searches expanding cell rings until the k-th distance is strictly
 * inside the visited box. Radius mode counts, scans, emits and then insertion-sorts each row by
 * ID. Order within a grid cell is unspecified, but every row is sorted by ID and no float atomics
 * are used, so results are bitwise reproducible, except which neighbors the one row cut by an
 * exceeded capacity keeps.
 */
export class GPUNeighborSearch implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUNeighborSearchProps;
  /** Resolved coordinate space and metric (native planar when `spatialContext` was omitted). */
  readonly spatialContext: GPUSpatialContext;
  /** The output CSR with `info` and `status` attached, for consumers. */
  readonly outputWeights: GPUSpatialWeights;

  constructor(props: GPUNeighborSearchProps) {
    const id = props.id ?? 'neighbor-search';
    this.id = id;
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      GPU_NEIGHBOR_SEARCH_GREAT_CIRCLE_SUPPORT
    );
    if (
      this.spatialContext.coordinateSpace === 'longitude-latitude' &&
      this.spatialContext.metric !== 'great-circle'
    ) {
      throw new Error(`${id} longitude-latitude positions need the great-circle metric`);
    }
    if (this.spatialContext.coordinateSpace === 'planar' && this.spatialContext.metric === 'none') {
      throw new Error(`${id} planar positions need the native metric`);
    }
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
    if (props.invalidCount) {
      validatePackedUint32View(props.invalidCount, `${id} invalidCount`);
      if (props.invalidCount.length < 1) {
        throw new Error(`${id} invalidCount must hold one uint32`);
      }
    }
    if (
      props.weightTransform !== undefined &&
      !['binary', 'raw', 'kernel', 'row'].includes(props.weightTransform)
    ) {
      throw new Error(`${id} weightTransform must be 'binary', 'raw', 'kernel' or 'row'`);
    }
    if (
      props.rowStandardize !== undefined &&
      props.weightTransform !== undefined &&
      props.rowStandardize !== (props.weightTransform === 'row')
    ) {
      throw new Error(
        `${id} rowStandardize ${props.rowStandardize} conflicts with weightTransform '${props.weightTransform}'`
      );
    }
    const transform = props.rowStandardize ? 'row' : (props.weightTransform ?? 'unknown');
    const planarSelfJoin =
      this.spatialContext.coordinateSpace === 'planar' &&
      props.mode === 'radius' &&
      !props.queryPositions &&
      !props.queryMask;
    this.outputWeights = describeGPUSpatialWeights(props.weights, {
      info: {
        rowCount: queryRows,
        transform,
        symmetric:
          planarSelfJoin && transform !== 'unknown' && transform !== 'row' ? true : 'unknown',
        hasSelfWeights: false,
        provenance: {
          source: OPERATION,
          criterion: props.mode === 'knn' ? 'knn' : 'distance-band',
          k: props.mode === 'knn' ? props.k : undefined
        }
      },
      status: {
        overflow: props.overflow,
        requiredCount: props.totalNeighbors,
        invalidCount: props.invalidCount
      }
    });
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        props.invalidCount,
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
      props.neighborCounts,
      props.invalidCount
    ]);
    const targetRows = positions.length;
    const queryRows = this.getQueryRows();
    const capacity = weights.neighbors.length;
    const cellCount = gridSize[0] * gridSize[1];
    const k = props.k ?? 1;
    const {spatialContext} = this;
    const geodesic = spatialContext.coordinateSpace === 'longitude-latitude';
    // Sphere radius in the context's length unit; `sphereRadius` is given in meters.
    const sphereRadius = geodesic
      ? (spatialContext.sphereRadius ?? GPU_EARTH_MEAN_RADIUS_METERS) /
        (spatialContext.units === 'kilometers' ? 1000 : 1)
      : undefined;
    const latticeWGSL = getNeighborSearchLatticeWGSL(mode, gridSize, sphereRadius);
    const radiusLoop = geodesic ? getGeodesicRadiusNeighborLoopWGSL : getRadiusNeighborLoopWGSL;
    const crossJoin = Boolean(queryPositions);
    const queryBinding: WGSLKernelBinding[] = queryPositions
      ? [{name: 'queryPositions', view: queryPositions, type: 'f32', access: 'read'}]
      : [];
    const queryName = crossJoin ? 'queryPositions' : 'positions';
    const readQueryWGSL = (row: string) => `let x = ${queryName}[${queryName}Offset + ${row} * 2u];
  let y = ${queryName}[${queryName}Offset + ${row} * 2u + 1u];`;
    const selfCondition = crossJoin ? 'true' : 'neighbor != queryIndex';
    // A self join applies the target mask to the query rows too unless a query mask is given.
    const queryMask = props.queryMask ?? (crossJoin ? undefined : props.mask);
    // Threads of the search kernels walk the valid queries in grid-cell order, so neighboring
    // threads search neighboring cells (coalesced reads, coherent control flow) however the
    // input is ordered. The thread's query row (`row`) is where its results are written, so
    // outputs are identical to launching in row order. A self join whose query set equals the
    // target set reuses the target index; otherwise the queries get an index of their own.
    const sharesTargetIndex = !crossJoin && !props.queryMask;

    const gridBounds = createTransientView(graph, `${id}-grid-bounds`, 'float32', 4);
    const gridPositions = createTransientView(
      graph,
      `${id}-grid-positions`,
      'float32x2',
      targetRows
    );
    const cellOffsets = createTransientView(graph, `${id}-cell-offsets`, 'uint32', cellCount + 1);
    const sortedRows = createTransientView(graph, `${id}-sorted-rows`, 'uint32', targetRows);
    // Target `[x, y, row]` triples (coordinates as float bits) in cell order: the search loops
    // read them sequentially instead of gathering, and one binding carries all three.
    const sortedTargets = createTransientView(
      graph,
      `${id}-sorted-targets`,
      'uint32',
      targetRows * 3
    );
    const gridCount = createTransientView(graph, `${id}-grid-count`, 'uint32', 1);
    const gridOverflow = createTransientView(graph, `${id}-grid-overflow`, 'uint32', 1);
    const counts =
      props.neighborCounts ?? createTransientView(graph, `${id}-counts`, 'uint32', queryRows);
    const starts = createTransientView(graph, `${id}-starts`, 'uint32', queryRows);
    const distances =
      weights.distances ?? createTransientView(graph, `${id}-distances`, 'float32', capacity);
    const queryOrder = sharesTargetIndex
      ? sortedRows
      : createTransientView(graph, `${id}-query-order`, 'uint32', queryRows);
    // Valid query `[x, y, row]` triples in cell order; a shared index reuses the target triples.
    const sortedQueries = sharesTargetIndex
      ? sortedTargets
      : createTransientView(graph, `${id}-sorted-queries`, 'uint32', queryRows * 3);
    const queryCount = sharesTargetIndex
      ? gridCount
      : createTransientView(graph, `${id}-query-count`, 'uint32', 1);

    const nodes: GPUCommandNode<Parameters>[] = [
      // The upstream grid index takes its domain from a buffer, so the per-frame lattice (cells at
      // least `radius` wide in radius mode) becomes bounds extended to COLUMNS x ROWS whole cells.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-grid-bounds`,
        operation: OPERATION,
        variant: 'grid-bounds',
        bindings: [
          {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
          {name: 'gridBounds', view: gridBounds, type: 'f32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: latticeWGSL,
        body: `let lattice = readLattice();
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  gridBounds[gridBoundsOffset] = select(invalid, lattice.minimumX, lattice.valid);
  gridBounds[gridBoundsOffset + 1u] = select(invalid, lattice.minimumY, lattice.valid);
  gridBounds[gridBoundsOffset + 2u] = select(invalid,
    max(lattice.maximumX, lattice.minimumX + lattice.cellWidth * f32(COLUMNS)), lattice.valid);
  gridBounds[gridBoundsOffset + 3u] = select(invalid,
    max(lattice.maximumY, lattice.minimumY + lattice.cellHeight * f32(ROWS)), lattice.valid);`
      }),
      // The grid index has no mask and its domain is wider than the valid extent, so excluded
      // targets become NaN points, which it ignores.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-grid-positions`,
        operation: OPERATION,
        variant: 'grid-positions',
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
          ...(props.mask
            ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
            : []),
          {name: 'gridPositions', view: gridPositions, type: 'f32', access: 'read_write'}
        ],
        invocationCount: targetRows,
        declarations: latticeWGSL,
        body: `let lattice = readLattice();
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let included = ${props.mask ? 'mask[maskOffset + index] != 0u' : 'true'} && isPointValid(lattice, x, y);
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  gridPositions[gridPositionsOffset + index * 2u] = select(invalid, x, included);
  gridPositions[gridPositionsOffset + index * 2u + 1u] = select(invalid, y, included);`
      }),
      // Buckets targets into the lattice: `cellOffsets` and `sortedRows` (target rows by cell, in
      // unspecified order within a cell; every consumer below is order independent or sorts).
      ...new GPUGridIndex({
        id: `${id}-grid-index`,
        positions: gridPositions,
        gridSize: [gridSize[0], gridSize[1]],
        bounds: [0, 0, 1, 1],
        boundsBuffer: gridBounds,
        cellOffsets,
        objectIds: sortedRows,
        count: gridCount,
        overflow: gridOverflow
      }).getCommandNodes(graph),
      // Query validity, stored in `counts` until the search pass overwrites it with the count.
      createWGSLKernelNode<Parameters>(graph, {
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
  ${readQueryWGSL('index')}
  let included = ${queryMask ? 'queryMask[queryMaskOffset + index] != 0u' : 'true'};
  counts[countsOffset + index] = select(0u, 1u, included && isPointValid(lattice, x, y));`
      })
    ];

    if (!sharesTargetIndex) {
      const queryGridPositions = createTransientView(
        graph,
        `${id}-query-grid-positions`,
        'float32x2',
        queryRows
      );
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-query-grid-positions`,
          operation: OPERATION,
          variant: 'query-grid-positions',
          bindings: [
            ...(crossJoin
              ? queryBinding
              : [
                  {
                    name: 'positions',
                    view: positions,
                    type: 'f32' as const,
                    access: 'read' as const
                  }
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
            {
              name: 'queryGridPositions',
              view: queryGridPositions,
              type: 'f32',
              access: 'read_write'
            }
          ],
          invocationCount: queryRows,
          declarations: latticeWGSL,
          body: `let lattice = readLattice();
  ${readQueryWGSL('index')}
  let included = ${queryMask ? 'queryMask[queryMaskOffset + index] != 0u' : 'true'} && isPointValid(lattice, x, y);
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  queryGridPositions[queryGridPositionsOffset + index * 2u] = select(invalid, x, included);
  queryGridPositions[queryGridPositionsOffset + index * 2u + 1u] = select(invalid, y, included);`
        }),
        // Only the cell order of the queries is used; the cell offsets are scratch.
        ...new GPUGridIndex({
          id: `${id}-query-grid-index`,
          positions: queryGridPositions,
          gridSize: [gridSize[0], gridSize[1]],
          bounds: [0, 0, 1, 1],
          boundsBuffer: gridBounds,
          cellOffsets: createTransientView(
            graph,
            `${id}-query-cell-offsets`,
            'uint32',
            cellCount + 1
          ),
          objectIds: queryOrder,
          count: queryCount,
          overflow: createTransientView(graph, `${id}-query-grid-overflow`, 'uint32', 1)
        }).getCommandNodes(graph)
      );
    }
    // Kernels bind at most 8 storage buffers, so the target and query triples are single buffers
    // and a shared index binds each of them once under the target names.
    const queryBufferName = sharesTargetIndex ? 'sortedTargets' : 'sortedQueries';
    const queryCountName = sharesTargetIndex ? 'gridCount' : 'queryCount';
    const searchBindings: WGSLKernelBinding[] = [
      {name: 'sortedTargets', view: sortedTargets, type: 'u32', access: 'read'},
      ...(sharesTargetIndex
        ? []
        : [
            {
              name: 'sortedQueries',
              view: sortedQueries,
              type: 'u32' as const,
              access: 'read' as const
            }
          ]),
      {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
      {name: 'cellOffsets', view: cellOffsets, type: 'u32', access: 'read'},
      {name: queryCountName, view: queryCount, type: 'u32', access: 'read'}
    ];
    // Maps the thread to the valid query at that cell-order slot and reads its point; surplus
    // threads (invalid queries have no work and a zero count from the query-valid pass) return.
    const threadToRowWGSL = `if (index >= ${queryCountName}[${queryCountName}Offset]) {
    return;
  }
  let queryIndex = ${queryBufferName}[${queryBufferName}Offset + index * 3u + 2u];
  let x = bitcast<f32>(${queryBufferName}[${queryBufferName}Offset + index * 3u]);
  let y = bitcast<f32>(${queryBufferName}[${queryBufferName}Offset + index * 3u + 1u]);`;

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-sorted-targets`,
        operation: OPERATION,
        variant: 'sorted-targets',
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {name: 'sortedRows', view: sortedRows, type: 'u32', access: 'read'},
          {name: 'gridCount', view: gridCount, type: 'u32', access: 'read'},
          {name: 'sortedTargets', view: sortedTargets, type: 'u32', access: 'read_write'}
        ],
        invocationCount: targetRows,
        body: `if (index >= gridCount[gridCountOffset]) {
    return;
  }
  let row = sortedRows[sortedRowsOffset + index];
  sortedTargets[sortedTargetsOffset + index * 3u] = bitcast<u32>(positions[positionsOffset + row * 2u]);
  sortedTargets[sortedTargetsOffset + index * 3u + 1u] = bitcast<u32>(positions[positionsOffset + row * 2u + 1u]);
  sortedTargets[sortedTargetsOffset + index * 3u + 2u] = row;`
      })
    );
    if (!sharesTargetIndex) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-sorted-queries`,
          operation: OPERATION,
          variant: 'sorted-queries',
          bindings: [
            {
              name: 'queryPositions',
              view: queryPositions ?? positions,
              type: 'f32',
              access: 'read'
            },
            {name: 'queryOrder', view: queryOrder, type: 'u32', access: 'read'},
            {name: 'queryCount', view: queryCount, type: 'u32', access: 'read'},
            {name: 'sortedQueries', view: sortedQueries, type: 'u32', access: 'read_write'}
          ],
          invocationCount: queryRows,
          body: `if (index >= queryCount[queryCountOffset]) {
    return;
  }
  let row = queryOrder[queryOrderOffset + index];
  sortedQueries[sortedQueriesOffset + index * 3u] = bitcast<u32>(queryPositions[queryPositionsOffset + row * 2u]);
  sortedQueries[sortedQueriesOffset + index * 3u + 1u] = bitcast<u32>(queryPositions[queryPositionsOffset + row * 2u + 1u]);
  sortedQueries[sortedQueriesOffset + index * 3u + 2u] = row;`
        })
      );
    }

    let knnIds: GraphDataView<'uint32'> | undefined;
    let knnDistances: GraphDataView<'float32'> | undefined;
    if (mode === 'knn') {
      knnIds = createTransientView(graph, `${id}-knn-ids`, 'uint32', queryRows * k);
      knnDistances = createTransientView(graph, `${id}-knn-distances`, 'float32', queryRows * k);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
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
          body: `${threadToRowWGSL}
  var found = 0u;
  var bestDistances: array<f32, ${k}>;
  var bestIds: array<u32, ${k}>;
  if (counts[countsOffset + queryIndex] != 0u) {
    let lattice = readLattice();
    ${
      geodesic
        ? getGeodesicNearestNeighborSearchWGSL(selfCondition)
        : getNearestNeighborSearchWGSL(selfCondition)
    }
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
    knnIds[knnIdsOffset + queryIndex * K + slot] = bestIds[slot];
    knnDistances[knnDistancesOffset + queryIndex * K + slot] = ${
      geodesic ? 'bestDistances[slot]' : 'sqrt(bestDistances[slot])'
    };
  }
  counts[countsOffset + queryIndex] = found;`
        })
      );
    } else {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-count`,
          operation: OPERATION,
          variant: 'radius-count',
          bindings: [
            ...searchBindings,
            {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
          ],
          invocationCount: queryRows,
          declarations: latticeWGSL,
          body: `${threadToRowWGSL}
  var count = 0u;
  if (counts[countsOffset + queryIndex] != 0u) {
    let lattice = readLattice();
    ${radiusLoop(`if (${selfCondition}) {
          count++;
        }`)}
  }
  counts[countsOffset + queryIndex] = count;`
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
      createWGSLKernelNode<Parameters>(graph, {
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
        createWGSLKernelNode<Parameters>(graph, {
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
        createWGSLKernelNode<Parameters>(graph, {
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
          body: `${threadToRowWGSL}
  let begin = offsets[offsetsOffset + queryIndex];
  let end = offsets[offsetsOffset + queryIndex + 1u];
  if (end > begin) {
    let lattice = readLattice();
    var next = begin;
    ${radiusLoop(`if (${selfCondition} && next < end) {
          neighbors[neighborsOffset + next] = neighbor;
          distances[distancesOffset + next] = ${geodesic ? 'distance' : 'sqrt(distanceSquared)'};
          next++;
        }`)}
  }`
        }),
        createWGSLKernelNode<Parameters>(graph, {
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

    if (props.invalidCount) {
      // Included rows (mask nonzero) with a non-finite coordinate or outside the bounds, summed
      // with a scan so no atomics are needed. Self joins count targets; cross joins add queries.
      const flagRows = targetRows + (crossJoin ? queryRows : 0);
      const invalidFlags = createTransientView(graph, `${id}-invalid-flags`, 'uint32', flagRows);
      const invalidStarts = createTransientView(graph, `${id}-invalid-starts`, 'uint32', flagRows);
      const readPoint = (
        name: string,
        row: string
      ) => `let x = ${name}[${name}Offset + ${row} * 2u];
    let y = ${name}[${name}Offset + ${row} * 2u + 1u];`;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-invalid-flags`,
          operation: OPERATION,
          variant: 'invalid-flags',
          bindings: [
            {name: 'positions', view: positions, type: 'f32', access: 'read'},
            ...queryBinding,
            {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
            ...(props.mask
              ? [{name: 'mask', view: props.mask, type: 'u32' as const, access: 'read' as const}]
              : []),
            ...(crossJoin && props.queryMask
              ? [
                  {
                    name: 'queryMask',
                    view: props.queryMask,
                    type: 'u32' as const,
                    access: 'read' as const
                  }
                ]
              : []),
            {name: 'invalidFlags', view: invalidFlags, type: 'u32', access: 'read_write'}
          ],
          invocationCount: flagRows,
          declarations: `${latticeWGSL}
const TARGET_ROWS: u32 = ${targetRows}u;`,
          body: `let lattice = readLattice();
  var included = true;
  var inside = false;
  if (index < TARGET_ROWS) {
    ${readPoint('positions', 'index')}
    included = ${props.mask ? 'mask[maskOffset + index] != 0u' : 'true'};
    inside = isPointInBounds(lattice, x, y);
  } else {
    ${
      crossJoin
        ? `let queryRow = index - TARGET_ROWS;
    ${readPoint('queryPositions', 'queryRow')}
    included = ${crossJoin && props.queryMask ? 'queryMask[queryMaskOffset + queryRow] != 0u' : 'true'};
    inside = isPointInBounds(lattice, x, y);`
        : ''
    }
  }
  invalidFlags[invalidFlagsOffset + index] = select(0u, 1u, included && !inside);`
        }),
        ...new GPUScan({
          id: `${id}-invalid-scan`,
          input: invalidFlags,
          output: invalidStarts,
          mode: 'exclusive'
        }).getCommandNodes(graph),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-invalid-count`,
          operation: OPERATION,
          variant: 'invalid-count',
          bindings: [
            {name: 'invalidFlags', view: invalidFlags, type: 'u32', access: 'read'},
            {name: 'invalidStarts', view: invalidStarts, type: 'u32', access: 'read'},
            {name: 'invalidCount', view: props.invalidCount, type: 'u32', access: 'read_write'}
          ],
          invocationCount: 1,
          body: `invalidCount[invalidCountOffset] =
    invalidStarts[invalidStartsOffset + ${flagRows - 1}u] + invalidFlags[invalidFlagsOffset + ${flagRows - 1}u];`
        })
      );
    }

    const bandwidthWGSL =
      mode === 'radius'
        ? 'let bandwidth = readParameter(4u);'
        : // Adaptive bandwidth: the k-th neighbor distance (the row maximum) times libpysal's epsilon.
          `var bandwidth = 0.0;
  for (var slot = begin; slot < end; slot++) {
    bandwidth = max(bandwidth, distances[distancesOffset + slot]);
  }
  bandwidth = bandwidth * ${getWGSLFloatLiteral(GPU_SPATIAL_WEIGHTS_ADAPTIVE_BANDWIDTH_EPSILON)};`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
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
  if (${
    props.rowStandardize === undefined ? 'readParameter(9u) != 0.0' : String(props.rowStandardize)
  } && sum > 0.0 && isFiniteFloat(sum)) {
    for (var slot = begin; slot < end; slot++) {
      weights[weightsOffset + slot] = weights[weightsOffset + slot] / sum;
    }
  }`
      })
    );
    return nodes;
  }
}
