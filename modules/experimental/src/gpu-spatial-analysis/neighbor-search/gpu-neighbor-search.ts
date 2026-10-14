// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGPUGreatCircleRadius} from '../../analysis-core/contracts/spatial-context';
import {
  createTransientView,
  GPUGridIndex,
  GPUScan,
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
  getGPUGridIndexMaximumCellCount,
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  getBVHNearestNeighborSearchWGSL,
  getBVHNeighborSearchDeclarationsWGSL,
  getBVHRadiusNeighborLoopWGSL,
  getBVHThreadToRowWGSL,
  getGeodesicNearestNeighborSearchWGSL,
  getGeodesicRadiusNeighborLoopWGSL,
  getInsertNeighborWGSL,
  getNearestNeighborSearchWGSL,
  getNeighborSearchLatticeWGSL,
  getRadiusNeighborLoopWGSL,
  NEIGHBOR_SEARCH_BVH_NODE_WORDS,
  NEIGHBOR_WEIGHT_WGSL
} from './neighbor-search-kernels';
import {getNearestBVHNodes} from '../spatial-join/nearest-nodes';
import {getNextPowerOfTwo} from '../spatial-join/spatial-join-passes';
import {GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH} from './neighbor-search-parameters';
import {
  describeGPUSpatialWeights,
  type GPUSpatialWeights,
  type GPUSpatialWeightsOutput,
  type GPUSpatialWeightsTransformState,
  validateGPUSpatialWeightsOutput
} from '../spatial-weights/spatial-weights';
import {GPU_SPATIAL_WEIGHTS_ADAPTIVE_BANDWIDTH_EPSILON} from '../spatial-weights/gpu-spatial-weights-transform';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext,
  type GPUSpatialContextSupport
} from '../../analysis-core/contracts/index';

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

/**
 * Search structure of {@link GPUNeighborSearch}: `'grid'` buckets targets into a cell lattice,
 * `'bvh'` traverses a Hilbert-packed bounding-volume hierarchy over the valid targets, and `'auto'`
 * uses the grid. The BVH remains an explicit opt-in until a measured workload justifies selection.
 */
export type GPUNeighborSearchStructure = 'auto' | 'grid' | 'bvh';

/** Largest supported compile-time `k`. */
export const GPU_NEIGHBOR_SEARCH_MAXIMUM_K = 32;

/**
 * Caller-owned outputs of {@link GPUNeighborSearch}: the CSR with its status scalars, plus an
 * optional unclamped neighbor count per query row.
 */
export type GPUNeighborSearchOutput = GPUSpatialWeightsOutput & {
  /** Optional unclamped neighbor count per query row. */
  neighborCounts?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUNeighborSearch}.
 *
 * Per-frame (no rebuild or recompile): the contents of `positions`, `queryPositions`, `mask`,
 * `queryMask` and `parameters` (bounds, radius, weight function, row standardization).
 * Compile-time: `mode`, `k`, `gridSize`, `searchStructure`, row counts, capacity and which optional
 * views exist.
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
   * speed. kNN mode spreads exactly this many cells over the bounds. `columns * rows` is at most
   * {@link getGPUGridIndexMaximumCellCount} (2^25 - 1 on a default device).
   */
  gridSize: readonly [number, number];
  /**
   * Search structure. Compile-time; results (IDs, offsets, distances, weights) never depend on it,
   * only speed. Default `'auto'`.
   *
   * - `'grid'`: the cell lattice only. Fastest on evenly spread points; on clustered points one
   *   cell holds most targets and each query degrades toward a scan of them.
   * - `'bvh'`: a Hilbert-packed point BVH only (near-first depth-first traversal with box pruning). Planar positions only; longitude-latitude throws.
   * - `'auto'`: uses exactly the grid graph, with no BVH allocation, build or occupancy pass.
   *   The 200k-point clustered and uniform kNN/radius benchmarks all favor the grid; occupancy
   *   alone did not predict traversal cost. Great-circle search also always uses the grid.
   *
   * The `gridSize` shapes the grid path.
   */
  searchStructure?: GPUNeighborSearchStructure;
  /** Neighbors per query in kNN mode, an integer in `[1, 32]`. Compile-time. Ignored in radius mode. */
  k?: number;
  /**
   * Caller-owned output CSR: `offsets` has `queryRows + 1` entries; the slot capacity is
   * `neighbors.length`. `distances` is optional. Row `i` lists neighbor target IDs ascending.
   * `status.overflow` is 1 when the neighbors did not fit the capacity, else 0.
   * `status.requiredCount` is the optional unclamped total neighbor count. `status.invalidCount` is
   * the optional count of rows that took part by mask but were dropped because a coordinate is not
   * finite or lies outside the per-frame `bounds` (or, for great-circle search, has a latitude
   * beyond +-90). Such rows are silently empty and never a neighbor, so a nonzero count means
   * `bounds` is too small or the data has missing coordinates. It counts target rows, plus the
   * query rows of a cross join (`queryPositions`).
   */
  output: GPUNeighborSearchOutput;
  /**
   * Declares the weight function the per-frame parameters will select, so `outputWeights.information` can
   * say what the weights are. Weight kind and row standardization are per-frame parameters that
   * the node cannot read at construction, so this is metadata the caller keeps true (derive it from
   * the same parameters with `getGPUNeighborSearchWeightTransform`). Omitted means `'unknown'`.
   */
  weightTransform?: GPUNeighborSearchWeightTransform;
  /**
   * Compile-time row standardization. When given, it is the single source of truth: the weights
   * kernel uses it as a constant and ignores the per-frame `rowStandardize` parameter, and
   * `outputWeights.information.transform` is `'row'` for `true`, else `weightTransform` (or `'unknown'`).
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
 * - `outputWeights` is `output` with `information` (`rowCount`, `transform` from `weightTransform` or
 *   `'unknown'`; `'row'` when `rowStandardize: true`, `symmetric` true only for an unmasked-query
 *   planar radius self join with a declared unstandardized transform, else `'unknown'`, `provenance` criterion `'knn'` or `'distance-band'`
 *   with `k`) and `status` (`overflow`, `requiredCount`, `invalidCount`) attached.
 *   `'adaptive'` kernel transforms accept it because kNN output has criterion `'knn'`. For PySAL's
 *   `min_threshold_distance` (the smallest band keeping every unit connected), run kNN with `k = 1`
 *   and take the maximum of the returned `distances`.
 * - Capacity: rows are laid out in query order; slots past `neighbors.length` are dropped. Offsets
 *   are clamped to the capacity, so a reader always sees consistent (possibly truncated) rows,
 *   and `status.overflow` is set. `status.requiredCount` and `neighborCounts` stay unclamped.
 *
 * Algorithm: `searchStructure` selects between the grid below and a point BVH (see
 * {@link GPUNeighborSearchProps.searchStructure}); both produce the identical definition above. The
 * BVH path builds a Hilbert-packed BVH over the valid targets (invalid targets get empty boxes),
 * walks it near-first depth-first per query with the same `(d^2, id)` top-k list and radius limit, and then
 * shares the grid path's scan, offsets, emit and sort-by-ID stages. Grid path: targets are bucketed by `GPUGridIndex` into a per-frame lattice (cells at least
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
  /**
   * Requested search structure: `props.searchStructure` (default `'auto'`), or `'grid'` for
   * great-circle search. `'auto'` resolves to the grid before allocating either structure.
   */
  readonly searchStructure: GPUNeighborSearchStructure;

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
    const searchStructure = props.searchStructure ?? 'auto';
    if (!['auto', 'grid', 'bvh'].includes(searchStructure)) {
      throw new Error(`${id} searchStructure must be 'auto', 'grid' or 'bvh'`);
    }
    if (searchStructure === 'bvh' && this.spatialContext.coordinateSpace === 'longitude-latitude') {
      throw new Error(`${id} searchStructure 'bvh' supports planar positions only`);
    }
    this.searchStructure =
      this.spatialContext.coordinateSpace === 'longitude-latitude' ? 'grid' : searchStructure;
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    const targetRows = props.positions.length;
    validateCapacity(id, 'positions length', targetRows, 1, 2 ** 31 - 1);
    if (props.queryPositions) {
      validatePackedView(props.queryPositions, ['float32x2'], `${id} queryPositions`);
      validateCapacity(id, 'queryPositions length', props.queryPositions.length, 1, 2 ** 31 - 1);
    }
    const queryRows = this.getQueryRows();
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH} float32 values`
      );
    }
    const [columns, rows] = props.gridSize;
    validateCapacity(id, 'gridSize columns', columns, 1, Number.MAX_SAFE_INTEGER);
    validateCapacity(id, 'gridSize rows', rows, 1, Number.MAX_SAFE_INTEGER);
    if (props.mode === 'knn') {
      validateCapacity(id, 'k', props.k ?? Number.NaN, 1, GPU_NEIGHBOR_SEARCH_MAXIMUM_K);
    }
    if (props.mask) {
      validatePackedView(props.mask, ['uint32'], `${id} mask`);
      if (props.mask.length !== targetRows) {
        throw new Error(`${id} mask length must equal positions length`);
      }
    }
    if (props.queryMask) {
      validatePackedView(props.queryMask, ['uint32'], `${id} queryMask`);
      if (props.queryMask.length !== queryRows) {
        throw new Error(`${id} queryMask length must equal the query row count`);
      }
    }
    const {output} = props;
    const weightRows = validateGPUSpatialWeightsOutput(id, output);
    if (weightRows !== queryRows) {
      throw new Error(`${id} output.offsets length must equal the query row count + 1`);
    }
    const {requiredCount, invalidCount} = output.status;
    for (const [name, view] of [
      ['output.status.overflow', output.status.overflow],
      ['output.status.requiredCount', requiredCount],
      ['output.status.invalidCount', invalidCount]
    ] as const) {
      if (view) {
        validatePackedView(view, ['uint32'], `${id} ${name}`);
        if (view.length < 1) {
          throw new Error(`${id} ${name} must hold one uint32`);
        }
      }
    }
    if (output.neighborCounts) {
      validatePackedView(output.neighborCounts, ['uint32'], `${id} output.neighborCounts`);
      if (output.neighborCounts.length !== queryRows) {
        throw new Error(`${id} output.neighborCounts length must equal the query row count`);
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
    this.outputWeights = describeGPUSpatialWeights(output, {
      information: {
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
      }
    });
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        invalidCount,
        output.offsets,
        output.neighbors,
        output.weights,
        output.distances,
        output.status.overflow,
        requiredCount,
        output.neighborCounts
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
    const {positions, queryPositions, parameters, gridSize, output: weights, mode} = props;
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
      weights.status.overflow,
      weights.status.requiredCount,
      weights.status.invalidCount,
      weights.neighborCounts
    ]);
    validateMaximumCount(
      id,
      'gridSize cells',
      gridSize[0] * gridSize[1],
      getGPUGridIndexMaximumCellCount(graph.device)
    );
    const targetRows = positions.length;
    const queryRows = this.getQueryRows();
    const capacity = weights.neighbors.length;
    const cellCount = gridSize[0] * gridSize[1];
    const k = props.k ?? 1;
    const {spatialContext} = this;
    const geodesic = spatialContext.coordinateSpace === 'longitude-latitude';
    // Sphere radius in the context's length unit.
    const sphereRadius = getGPUGreatCircleRadius(spatialContext);
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

    // Resolve before allocating/building: the measured clustered and uniform workloads favor
    // the grid. An unused BVH must not add a sort/build to every auto encoding.
    const useBvh = this.searchStructure === 'bvh';
    const useGrid = !useBvh;
    // Grid-only scratch is one element when the grid is not used.
    const gridSized = (length: number) => (useGrid ? length : 1);

    const gridBounds = createTransientView(graph, `${id}-grid-bounds`, 'float32', gridSized(4));
    const gridPositions = createTransientView(
      graph,
      `${id}-grid-positions`,
      'float32x2',
      targetRows
    );
    const cellOffsets = createTransientView(
      graph,
      `${id}-cell-offsets`,
      'uint32',
      gridSized(cellCount + 1)
    );
    const sortedRows = createTransientView(
      graph,
      `${id}-sorted-rows`,
      'uint32',
      gridSized(targetRows)
    );
    // Target `[x, y, row]` triples (coordinates as float bits) in cell order: the search loops
    // read them sequentially instead of gathering, and one binding carries all three.
    const sortedTargets = createTransientView(
      graph,
      `${id}-sorted-targets`,
      'uint32',
      gridSized(targetRows * 3)
    );
    const gridCount = createTransientView(graph, `${id}-grid-count`, 'uint32', 1);
    const gridOverflow = createTransientView(graph, `${id}-grid-overflow`, 'uint32', 1);
    const counts =
      weights.neighborCounts ?? createTransientView(graph, `${id}-counts`, 'uint32', queryRows);
    const starts = createTransientView(graph, `${id}-starts`, 'uint32', queryRows);
    const distances =
      weights.distances ?? createTransientView(graph, `${id}-distances`, 'float32', capacity);
    const queryOrder = sharesTargetIndex
      ? sortedRows
      : createTransientView(graph, `${id}-query-order`, 'uint32', gridSized(queryRows));
    // Valid query `[x, y, row]` triples in cell order; a shared index reuses the target triples.
    const sortedQueries = sharesTargetIndex
      ? sortedTargets
      : createTransientView(graph, `${id}-sorted-queries`, 'uint32', gridSized(queryRows * 3));
    const queryCount = sharesTargetIndex
      ? gridCount
      : createTransientView(graph, `${id}-query-count`, 'uint32', 1);

    const nodes: GPUCommandNode<Parameters>[] = [
      // The upstream grid index takes its domain from a buffer, so the per-frame lattice (cells at
      // least `radius` wide in radius mode) becomes bounds extended to COLUMNS x ROWS whole cells.
      ...(useGrid
        ? [
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
            })
          ]
        : []),
      // The grid index has no mask and its domain is wider than the valid extent, so excluded
      // targets become NaN points, which it ignores. The BVH build treats a NaN point as an empty
      // box, so this one array carries target validity (mask, finite, bounds) into both structures.
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
      ...(useGrid
        ? new GPUGridIndex({
            id: `${id}-grid-index`,
            positions: gridPositions,
            gridSize: [gridSize[0], gridSize[1]],
            bounds: [0, 0, 1, 1],
            boundsBuffer: gridBounds,
            cellOffsets,
            objectIds: sortedRows,
            count: gridCount,
            overflow: gridOverflow
          }).getCommandNodes(graph)
        : []),
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

    if (useGrid && !sharesTargetIndex) {
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

    if (useGrid) {
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
    }
    if (useGrid && !sharesTargetIndex) {
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

    // BVH over the valid targets (NaN points of `gridPositions` become empty boxes), packed with
    // the leaf rows into one node table so a search kernel binds the whole tree once.
    let bvhNodeData: GraphDataView<'uint32'> | undefined;
    let bvhDeclarations = '';
    if (useBvh) {
      const {bvh, nodes: bvhNodes} = getNearestBVHNodes(graph, {
        id: `${id}-bvh`,
        operation: OPERATION,
        features: {kind: 'points', positions: gridPositions},
        featureCount: targetRows,
        leafCapacity: Math.max(2, getNextPowerOfTwo(targetRows)),
        spatialSort: true
      });
      nodes.push(...bvhNodes);
      bvhNodeData = createTransientView(
        graph,
        `${id}-bvh-node-data`,
        'uint32',
        bvh.nodeCount * NEIGHBOR_SEARCH_BVH_NODE_WORDS
      );
      bvhDeclarations = getBVHNeighborSearchDeclarationsWGSL({
        targetRows,
        internalNodeCount: bvh.internalNodeCount,
        levelCount: bvh.levelCount
      });
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-bvh-node-data`,
          operation: OPERATION,
          variant: 'bvh-node-data',
          bindings: [
            {name: 'nodeMinima', view: bvh.nodeMinima, type: 'f32', access: 'read'},
            {name: 'nodeMaxima', view: bvh.nodeMaxima, type: 'f32', access: 'read'},
            {name: 'leafIds', view: bvh.leafIds, type: 'u32', access: 'read'},
            {name: 'nodeData', view: bvhNodeData, type: 'u32', access: 'read_write'}
          ],
          invocationCount: bvh.nodeCount,
          declarations: `const INTERNAL_NODE_COUNT: u32 = ${bvh.internalNodeCount}u;
const NO_ROW: u32 = 0xffffffffu;`,
          body: `let base = nodeDataOffset + index * ${NEIGHBOR_SEARCH_BVH_NODE_WORDS}u;
  nodeData[base] = bitcast<u32>(nodeMinima[nodeMinimaOffset + index * 2u]);
  nodeData[base + 1u] = bitcast<u32>(nodeMinima[nodeMinimaOffset + index * 2u + 1u]);
  nodeData[base + 2u] = bitcast<u32>(nodeMaxima[nodeMaximaOffset + index * 2u]);
  nodeData[base + 3u] = bitcast<u32>(nodeMaxima[nodeMaximaOffset + index * 2u + 1u]);
  nodeData[base + 4u] = select(NO_ROW, leafIds[leafIdsOffset + index - INTERNAL_NODE_COUNT], index >= INTERNAL_NODE_COUNT);`
        })
      );
    }

    // BVH kernels bind the tree, the query points and the parameters (and the validity in `counts`).
    const bvhSearchBindings: WGSLKernelBinding[] = bvhNodeData
      ? [
          {name: 'nodeData', view: bvhNodeData, type: 'u32', access: 'read'},
          {name: queryName, view: queryPositions ?? positions, type: 'f32', access: 'read'},
          {name: 'parameters', view: parameters, type: 'f32', access: 'read'}
        ]
      : [];
    const bvhThreadToRowWGSL = getBVHThreadToRowWGSL(crossJoin);

    let knnIds: GraphDataView<'uint32'> | undefined;
    let knnDistances: GraphDataView<'float32'> | undefined;
    if (mode === 'knn') {
      knnIds = createTransientView(graph, `${id}-knn-ids`, 'uint32', queryRows * k);
      knnDistances = createTransientView(graph, `${id}-knn-distances`, 'float32', queryRows * k);
      const knnTailWGSL = `// Re-sort the selected neighbors by ID (insertion sort over at most K entries).
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
  counts[countsOffset + queryIndex] = found;`;
      const knnColumns: WGSLKernelBinding[] = [
        {name: 'counts', view: counts, type: 'u32', access: 'read_write'},
        {name: 'knnIds', view: knnIds, type: 'u32', access: 'read_write'},
        {name: 'knnDistances', view: knnDistances, type: 'f32', access: 'read_write'}
      ];
      if (useGrid) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-knn`,
            operation: OPERATION,
            variant: 'knn',
            bindings: [...searchBindings, ...knnColumns],
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
  ${knnTailWGSL}`
          })
        );
      }
      if (useBvh) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-bvh-knn`,
            operation: OPERATION,
            variant: 'bvh-knn',
            bindings: [...bvhSearchBindings, ...knnColumns],
            invocationCount: queryRows,
            declarations: `${latticeWGSL}
${getInsertNeighborWGSL(k)}
${bvhDeclarations}`,
            body: `${bvhThreadToRowWGSL}
  var found = 0u;
  var bestDistances: array<f32, ${k}>;
  var bestIds: array<u32, ${k}>;
  if (counts[countsOffset + queryIndex] != 0u) {
    let lattice = readLattice();
    ${readQueryWGSL('queryIndex')}
    ${getBVHNearestNeighborSearchWGSL(selfCondition)}
  }
  ${knnTailWGSL}`
          })
        );
      }
    } else {
      if (useGrid) {
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
      if (useBvh) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-bvh-count`,
            operation: OPERATION,
            variant: 'bvh-radius-count',
            bindings: [
              ...bvhSearchBindings,
              {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
            ],
            invocationCount: queryRows,
            declarations: `${latticeWGSL}
${bvhDeclarations}`,
            body: `${bvhThreadToRowWGSL}
  var count = 0u;
  if (counts[countsOffset + queryIndex] != 0u) {
    let lattice = readLattice();
    ${readQueryWGSL('queryIndex')}
    ${getBVHRadiusNeighborLoopWGSL(`if (${selfCondition}) {
        count++;
      }`)}
  }
  counts[countsOffset + queryIndex] = count;`
          })
        );
      }
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
          {name: 'overflow', view: weights.status.overflow, type: 'u32', access: 'read_write'},
          ...(weights.status.requiredCount
            ? [
                {
                  name: 'totalNeighbors',
                  view: weights.status.requiredCount,
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
    ${weights.status.requiredCount ? 'totalNeighbors[totalNeighborsOffset] = total;' : ''}
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
      const emitColumns: WGSLKernelBinding[] = [
        {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
        {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read_write'},
        {name: 'distances', view: distances, type: 'f32', access: 'read_write'}
      ];
      if (useBvh) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-bvh-emit`,
            operation: OPERATION,
            variant: 'bvh-radius-emit',
            bindings: [...bvhSearchBindings, ...emitColumns],
            invocationCount: queryRows,
            declarations: `${latticeWGSL}
${bvhDeclarations}`,
            body: `${bvhThreadToRowWGSL}
  let begin = offsets[offsetsOffset + queryIndex];
  let end = offsets[offsetsOffset + queryIndex + 1u];
  if (end > begin) {
    let lattice = readLattice();
    ${readQueryWGSL('queryIndex')}
    var next = begin;
    ${getBVHRadiusNeighborLoopWGSL(`if (${selfCondition} && next < end) {
        neighbors[neighborsOffset + next] = neighbor;
        distances[distancesOffset + next] = sqrt(distanceSquared);
        next++;
      }`)}
  }`
          })
        );
      }
      if (useGrid) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-emit`,
            operation: OPERATION,
            variant: 'radius-emit',
            bindings: [...searchBindings, ...emitColumns],
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
          })
        );
      }
      nodes.push(
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

    if (weights.status.invalidCount) {
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
            {
              name: 'invalidCount',
              view: weights.status.invalidCount,
              type: 'u32',
              access: 'read_write'
            }
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
