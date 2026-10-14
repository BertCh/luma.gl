// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCompaction,
  GPUGroupAggregation,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createFillNode, createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {GPU_SPATIAL_CLUSTERING_NOISE} from './spatial-clustering-parameters';
import {
  GET_QUIET_NAN_SEEDED_WGSL,
  IS_FINITE_FLOAT_WGSL,
  IS_NAN_BITS_WGSL
} from '../../utils/wgsl-common';
import {resolveGPUSpatialContext, type GPUSpatialContext} from '../contracts/spatial-context';

const OPERATION = 'GPUKMeans';
const NO_ROW = '0xffffffffu';
/** Workgroup size of the per-iteration center reductions. */
const REDUCTION_WORKGROUP_SIZE = 256;
/** Largest number of lane slices that split one cluster's work inside a chunk. */
const MAXIMUM_SLICES = 16;
/** Largest number of point chunks (workgroups) the partial pass launches. */
const MAXIMUM_CHUNKS = 256;

/** Formats a finite JavaScript number as a WGSL `f32` literal. */
function formatWGSLFloat(value: number): string {
  const text = String(Math.fround(value));
  return /[.e]/.test(text) ? text : `${text}.0`;
}

/** Largest `k`. */
export const GPU_KMEANS_MAXIMUM_CLUSTERS = 256;

/** Largest `iterations`: every Lloyd iteration adds several dispatches to the graph. */
export const GPU_KMEANS_MAXIMUM_ITERATIONS = 64;

/** Largest `restartCount`: every restart adds a full seeding and iteration chain to the graph. */
export const GPU_KMEANS_MAXIMUM_RESTARTS = 32;

/**
 * Seed of one {@link GPUKMeans} restart, derived deterministically from the user `seed` and the
 * restart index. Restart 0 uses `seed` itself, so `restartCount: 1` is the single seeded run;
 * later restarts use a murmur3-finalized mix of `seed` and the index. Exported so tests (and
 * callers) can reproduce any single restart with `restartCount: 1` and this seed.
 */
export function getKMeansRestartSeed(seed: number, restart: number): number {
  if (restart === 0) {
    return seed >>> 0;
  }
  let hash = (seed + Math.imul(restart, 0x9e3779b9)) >>> 0;
  hash ^= hash >>> 16;
  hash = Math.imul(hash, 0x85ebca6b) >>> 0;
  hash ^= hash >>> 13;
  hash = Math.imul(hash, 0xc2b2ae35) >>> 0;
  hash ^= hash >>> 16;
  return hash >>> 0;
}

/** How {@link GPUKMeans} picks the initial centers. */
export type GPUKMeansInitialization = 'first-valid' | 'kmeans++';

/**
 * Properties for {@link GPUKMeans}.
 *
 * Per-frame (no recompile): the contents of `positions`. Compile-time: `positions.length`, `k`,
 * `iterations`, `initialization`, `seed`, `restartCount` and which optional views are present.
 */
export type GPUKMeansProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'kmeans'`. */
  id?: string;
  /**
   * Coordinate space and metric of `positions`. Omitted means native planar coordinates. Distances
   * are Euclidean in the position units and centers are arithmetic means, so only planar contexts
   * are accepted; project longitude/latitude to a planar frame first.
   */
  spatialContext?: GPUSpatialContext;
  /** Packed planar points, one row per point. Rows with a non-finite coordinate are noise. */
  positions: GraphDataView<'float32x2'>;
  /** Number of clusters, an integer in `[1, 256]`. Compile-time. */
  k: number;
  /**
   * Maximum number of Lloyd iterations (assign then update), an integer in `[1, 64]`.
   * Compile-time. The graph always holds this many iterations, but once an iteration moves no
   * center by more than `tolerance` the remaining ones early-out inside their kernels (see
   * `tolerance`), so the result is still a pure function of the inputs.
   */
  iterations: number;
  /**
   * Convergence tolerance: an iteration converges when the largest center shift (Euclidean, in
   * position units) is at most this value. Compile-time, finite and `>= 0`; default `0`, which
   * stops only at an exact fixed point and so gives the same result as running every iteration.
   * A positive value trades exactness for fewer iterations. After convergence the assign and
   * update kernels return immediately; the per-iteration dispatches remain in the graph (there is
   * no GPU loop exit), so the saving is in kernel work, not dispatch count.
   */
  tolerance?: number;
  /**
   * Optional caller-owned two-row `uint32` output: `[iterationsUsed, converged]`. `iterationsUsed`
   * is how many iterations moved centers (1 to `iterations`), `converged` is `1` when the last one
   * shifted every center by at most `tolerance`, and `0` when `iterations` ran out first. With
   * restarts these describe the returned (lowest-inertia) restart.
   */
  convergence?: GraphDataView<'uint32'>;
  /**
   * Optional caller-owned one-row `float32` output: the total inertia (sum of squared distances of
   * the valid points to their final centers) of the returned clustering, i.e. of the best restart
   * when `restartCount > 1`. NaN when there are no valid points.
   */
  inertia?: GraphDataView<'float32'>;
  /**
   * `'kmeans++'` (default) draws the initial centers with D-squared weighting from a seeded hash,
   * so the same `seed` always gives the same centers on one device. `'first-valid'` uses the `k`
   * lowest-index points with finite coordinates, which is order-dependent and can place centers
   * next to each other; it has no randomness, so it cannot be combined with `restartCount > 1`.
   */
  initialization?: GPUKMeansInitialization;
  /** Seed of the `'kmeans++'` hash, an integer in `[0, 2^32 - 1]`. Default 0. */
  seed?: number;
  /**
   * Number of independent seeded initializations, each followed by its own Lloyd iterations, an
   * integer in `[1, 32]`. Compile-time; default 1 (scikit-learn's `n_init='auto'` for k-means++).
   * The run with the lowest total inertia is returned (ties go to the lowest restart index).
   * Restart `r` is seeded with `getKMeansRestartSeed(seed, r)`. Each restart adds a full
   * seeding and iteration chain to the graph, so graph size and GPU work grow linearly with it.
   * Requires `initialization: 'kmeans++'` when above 1.
   */
  restartCount?: number;
  /** Caller-owned labels, one row per point: cluster `0..k-1`, or `0xffffffff` for noise. */
  labels: GraphDataView<'uint32'>;
  /**
   * Caller-owned centers, `k` rows. A center that receives no point in an iteration is relocated
   * like scikit-learn does (see {@link GPUKMeans}): it moves to one of the points farthest from
   * their assigned centers. Only when no such point exists (every point sits on its center) it
   * moves to the mean of the largest cluster. A center that could not be initialized (fewer than
   * `k` valid points with `'first-valid'`, or `'kmeans++'` running out of weight) is NaN, is never
   * relocated and stays empty.
   */
  centers: GraphDataView<'float32x2'>;
  /** Optional caller-owned member count per cluster, `k` rows. */
  sizes?: GraphDataView<'uint32'>;
  /**
   * Optional caller-owned squared distance of each point to its center, one row per point; NaN for
   * noise. An explain column: its sum is the within-cluster inertia.
   */
  squaredDistances?: GraphDataView<'float32'>;
};

/**
 * Deterministic k-means (Lloyd's algorithm) of planar points.
 *
 * See also the upstream `GPUKMeans` in `@luma.gl/gpgpu/gpu-vector-search`, which clusters
 * N-dimensional embedding vectors with evenly spaced seeding. This class is a different contract:
 * 2D points, `k-means++` seeding by default, best-of-`restartCount` restarts, a convergence tolerance, a noise label for invalid
 * points and a per-point squared distance (inertia) output.
 *
 * Every choice that could vary between runs is fixed:
 * - Initial centers are `k-means++` (default) with a seeded integer hash or, with
 *   `initialization: 'first-valid'`, the first `k` valid points (lowest index first). Each
 *   `k-means++` round samples the next center with probability proportional to the
 *   squared distance to the nearest chosen center, by an exponential race `-ln(u) / w` whose
 *   minimum is found with integer atomics, lowest row first on ties.
 * - Each of the `iterations` rounds assigns every point to the nearest center, the lowest center
 *   ID winning ties, and moves each center to the mean of its members. Means use a fixed-order chunked
 *   reduction (no sort, no atomics), so sums are bitwise reproducible on one device. A final assignment
 *   against the last centers produces `labels`, `sizes` and `squaredDistances`.
 *
 * Results are reproducible for identical inputs on one device. `log` is not exactly specified by
 * WGSL, so `'kmeans++'` draws may differ between devices; `'first-valid'` has no such caveat apart
 * from f32 distance rounding.
 *
 * Empty clusters are relocated after every assignment step, following scikit-learn's
 * `_relocate_empty_clusters_dense`: with `E` empty clusters (lowest cluster ID first) the `E`
 * points with the largest squared distance to their assigned center, farthest first, become the
 * new centers of those clusters, and the donor clusters' counts and sums lose those points before
 * the means are taken, so the means stay consistent. Ties in distance go to the lowest row. The
 * selection is a single-workgroup iterated argmax with a fixed reduction order, so it is
 * deterministic and adds no readback. If the farthest distance is 0 (every point sits on its
 * center) nothing is relocated, and clusters that are still empty (that case, or a donor that lost
 * its only member) move to the mean of the largest cluster, lowest ID on ties. Differences from
 * scikit-learn: it orders the `E` farthest points with `argpartition`, whose order among them is
 * unspecified, and for a still-empty cluster before the largest one it copies the largest
 * cluster's unaveraged sum; this class always uses descending distance and the averaged mean.
 * Clusters whose center is NaN (never initialized) are not relocated.
 *
 * Cost per iteration: an `O(n k)` assignment, an `O(n k)` shared-memory reduction over `n`
 * points and, when a cluster is empty, an `O(n E)` relocation scan in one workgroup (three
 * dispatches); no sort, so memory does not grow with `iterations`.
 * Non-goals: weighted points, 3D points, k-medoids.
 */
export class GPUKMeans implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUKMeansProps;
  /** Resolved initialization. */
  readonly initialization: GPUKMeansInitialization;
  /** Resolved restart count. */
  readonly restartCount: number;
  /** Resolved spatial context (native planar when none was given). */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUKMeansProps) {
    const id = props.id ?? 'kmeans';
    this.id = id;
    this.props = props;
    this.initialization = props.initialization ?? 'kmeans++';
    this.restartCount = props.restartCount ?? 1;
    this.spatialContext = resolveGPUSpatialContext(id, props.spatialContext);
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    const rows = props.positions.length;
    if (rows < 1) {
      throw new Error(`${id} positions must hold at least one row`);
    }
    if (!Number.isInteger(props.k) || props.k < 1 || props.k > GPU_KMEANS_MAXIMUM_CLUSTERS) {
      throw new Error(`${id} k must be an integer in [1, ${GPU_KMEANS_MAXIMUM_CLUSTERS}]`);
    }
    if (
      !Number.isInteger(props.iterations) ||
      props.iterations < 1 ||
      props.iterations > GPU_KMEANS_MAXIMUM_ITERATIONS
    ) {
      throw new Error(
        `${id} iterations must be an integer in [1, ${GPU_KMEANS_MAXIMUM_ITERATIONS}]`
      );
    }
    if (this.initialization !== 'first-valid' && this.initialization !== 'kmeans++') {
      throw new Error(`${id} initialization must be 'first-valid' or 'kmeans++'`);
    }
    if (
      !Number.isInteger(this.restartCount) ||
      this.restartCount < 1 ||
      this.restartCount > GPU_KMEANS_MAXIMUM_RESTARTS
    ) {
      throw new Error(
        `${id} restartCount must be an integer in [1, ${GPU_KMEANS_MAXIMUM_RESTARTS}]`
      );
    }
    if (this.restartCount > 1 && this.initialization !== 'kmeans++') {
      throw new Error(`${id} restartCount above 1 needs the 'kmeans++' initialization`);
    }
    if (
      props.seed !== undefined &&
      (!Number.isInteger(props.seed) || props.seed < 0 || props.seed > 0xffffffff)
    ) {
      throw new Error(`${id} seed must be an integer in [0, 2^32 - 1]`);
    }
    if (
      props.tolerance !== undefined &&
      (!Number.isFinite(props.tolerance) || props.tolerance < 0)
    ) {
      throw new Error(`${id} tolerance must be a finite number >= 0`);
    }
    if (props.convergence) {
      validatePackedUint32View(props.convergence, `${id} convergence`);
      if (props.convergence.length !== 2) {
        throw new Error(`${id} convergence must hold two uint32 rows`);
      }
    }
    if (props.inertia) {
      validatePackedView(props.inertia, ['float32'], `${id} inertia`);
      if (props.inertia.length !== 1) {
        throw new Error(`${id} inertia must hold one float32 row`);
      }
    }
    validatePackedUint32View(props.labels, `${id} labels`);
    if (props.labels.length !== rows) {
      throw new Error(`${id} labels length must equal positions length`);
    }
    validatePackedView(props.centers, ['float32x2'], `${id} centers`);
    if (props.centers.length !== props.k) {
      throw new Error(`${id} centers length must equal k`);
    }
    if (props.sizes) {
      validatePackedUint32View(props.sizes, `${id} sizes`);
      if (props.sizes.length !== props.k) {
        throw new Error(`${id} sizes length must equal k`);
      }
    }
    if (props.squaredDistances) {
      validatePackedView(props.squaredDistances, ['float32'], `${id} squaredDistances`);
      if (props.squaredDistances.length !== rows) {
        throw new Error(`${id} squaredDistances length must equal positions length`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        props.labels,
        props.centers,
        props.sizes,
        props.squaredDistances,
        props.convergence,
        props.inertia
      ],
      [props.positions]
    );
  }

  /** Returns the preparation, initialization, Lloyd iteration and final assignment nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {positions, k, iterations, labels, centers} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      labels,
      centers,
      props.sizes,
      props.squaredDistances,
      props.convergence,
      props.inertia
    ]);
    const rows = positions.length;
    const seed = props.seed ?? 0;
    const restartCount = this.restartCount;
    const multiStart = restartCount > 1;
    const view = <Format extends 'uint32' | 'float32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, length);

    const validFlags = view('valid-flags', 'uint32', rows);
    const rowIds = view('row-ids', 'uint32', rows);
    const xs = view('xs', 'float32', rows);
    const ys = view('ys', 'float32', rows);
    const sizes = props.sizes ?? view('sizes', 'uint32', k);
    // With restarts every Lloyd run works on scratch centers and sizes; the best restart's centers
    // are copied to the caller's views at the end.
    const workCenters = multiStart
      ? createTransientView(graph, `${id}-work-centers`, 'float32x2', k)
      : centers;
    const workSizes = multiStart ? view('work-sizes', 'uint32', k) : sizes;
    // Squared distances are always produced when inertia is needed (restart scoring or output).
    const finalDistances =
      props.squaredDistances ??
      (multiStart || props.inertia ? view('squared-distances', 'float32', rows) : undefined);
    // Word 0: converged flag, word 1: iterations used, word 2: largest center shift (f32 bits;
    // non-negative floats order like their bit patterns, so atomicMax works).
    const state = view('state', 'uint32', 3);
    const tolerance = props.tolerance ?? 0;
    // Update step: a sort-free, atomic-free deterministic reduction. Points are cut into
    // `chunkCount` contiguous chunks of `chunkSize` rows (a multiple of the tile width); each
    // workgroup keeps per-cluster partial counts and sums for its chunk, then one workgroup per
    // cluster folds the chunk partials in a fixed order.
    const chunkCount = Math.min(MAXIMUM_CHUNKS, Math.ceil(rows / REDUCTION_WORKGROUP_SIZE));
    const chunkSize =
      Math.ceil(rows / chunkCount / REDUCTION_WORKGROUP_SIZE) * REDUCTION_WORKGROUP_SIZE;
    const slices = Math.min(MAXIMUM_SLICES, Math.floor(REDUCTION_WORKGROUP_SIZE / k));
    const partialCounts = view('partial-counts', 'uint32', k * chunkCount);
    const partialSumsX = view('partial-sums-x', 'float32', k * chunkCount);
    const partialSumsY = view('partial-sums-y', 'float32', k * chunkCount);
    // Per-iteration scratch of the empty-cluster relocation: each point's squared distance to the
    // center it was assigned to, and the per-cluster coordinate sums before the means are taken.
    const iterationDistances = view('iteration-distances', 'float32', rows);
    const clusterSums = createTransientView(graph, `${id}-cluster-sums`, 'float32x2', k);
    const sharedWGSL = `const K: u32 = ${k}u;
const NOISE: u32 = ${GPU_SPATIAL_CLUSTERING_NOISE}u;

${IS_FINITE_FLOAT_WGSL}
${IS_NAN_BITS_WGSL}

${GET_QUIET_NAN_SEEDED_WGSL}`;

    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-prepare`,
        operation: OPERATION,
        variant: 'prepare',
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {name: 'validFlags', view: validFlags, type: 'u32', access: 'read_write'},
          {name: 'rowIds', view: rowIds, type: 'u32', access: 'read_write'},
          {name: 'xs', view: xs, type: 'f32', access: 'read_write'},
          {name: 'ys', view: ys, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: sharedWGSL,
        body: `let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let valid = isFiniteFloat(x) && isFiniteFloat(y);
  validFlags[validFlagsOffset + index] = select(0u, 1u, valid);
  rowIds[rowIdsOffset + index] = index;
  xs[xsOffset + index] = select(0.0, x, valid);
  ys[ysOffset + index] = select(0.0, y, valid);`
      })
    ];

    const seeding = this.initialization === 'kmeans++';
    const minDistance = seeding ? view('min-distance', 'float32', rows) : undefined!;
    const keys = seeding ? view('keys', 'uint32', rows) : undefined!;
    const bestKey = seeding ? view('best-key', 'uint32', 1) : undefined!;
    const bestRow = seeding ? view('best-row', 'uint32', 1) : undefined!;
    const bestInertia = props.inertia ?? view('best-inertia', 'float32', 1);
    const bestCenters = multiStart
      ? createTransientView(graph, `${id}-best-centers`, 'float32x2', k)
      : undefined;
    const bestState = multiStart ? view('best-state', 'uint32', 3) : undefined;
    const assignNode = (
      assignId: string,
      withDistances: boolean,
      centersView: GraphDataView<'float32x2'>,
      distancesView?: GraphDataView<'float32'>
    ): GPUCommandNode<Parameters> =>
      createWGSLKernelNode<Parameters>(graph, {
        id: assignId,
        operation: OPERATION,
        variant: withDistances ? 'assign-final' : 'assign',
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {name: 'centers', view: centersView, type: 'f32', access: 'read'},
          {name: 'labels', view: labels, type: 'u32', access: 'read_write'},
          ...(withDistances
            ? []
            : [{name: 'state', view: state, type: 'u32' as const, access: 'read' as const}]),
          ...(distancesView
            ? [
                {
                  name: 'squaredDistances',
                  view: distancesView,
                  type: 'f32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: rows,
        declarations: sharedWGSL,
        body: `${
          withDistances
            ? ''
            : `// Converged: centers no longer move, so labels are already final.
  if (state[stateOffset] != 0u) {
    return;
  }
  `
        }let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  var best = NOISE;
  var bestDistance = getQuietNaN(index);
  if (isFiniteFloat(x) && isFiniteFloat(y)) {
    // Strict comparison: the lowest center ID wins ties; NaN centers never win.
    for (var center = 0u; center < K; center++) {
      let deltaX = centers[centersOffset + center * 2u] - x;
      let deltaY = centers[centersOffset + center * 2u + 1u] - y;
      let distance = deltaX * deltaX + deltaY * deltaY;
      if (!isNanBits(distance) && (best == NOISE || distance < bestDistance)) {
        best = center;
        bestDistance = distance;
      }
    }
  }
  labels[labelsOffset + index] = best;
  ${distancesView ? 'squaredDistances[squaredDistancesOffset + index] = select(getQuietNaN(index), bestDistance, best != NOISE);' : ''}`
      });

    const addInitialization = (restartId: string, restartSeed: number) => {
      if (this.initialization === 'first-valid') {
        const compactIds = view('compact-ids', 'uint32', rows);
        const validCount = view('valid-count', 'uint32', 1);
        nodes.push(
          ...new GPUCompaction({
            id: `${restartId}-valid-rows`,
            input: rowIds,
            flags: validFlags,
            output: compactIds,
            count: validCount
          }).getCommandNodes(graph),
          createWGSLKernelNode<Parameters>(graph, {
            id: `${restartId}-initial-centers`,
            operation: OPERATION,
            variant: 'initial-centers',
            bindings: [
              {name: 'positions', view: positions, type: 'f32', access: 'read'},
              {name: 'compactIds', view: compactIds, type: 'u32', access: 'read'},
              {name: 'validCount', view: validCount, type: 'u32', access: 'read'},
              {name: 'centers', view: workCenters, type: 'f32', access: 'read_write'}
            ],
            invocationCount: k,
            declarations: sharedWGSL,
            body: `let nan = getQuietNaN(index);
    var x = nan;
    var y = nan;
    if (index < validCount[validCountOffset]) {
      let row = compactIds[compactIdsOffset + index];
      x = positions[positionsOffset + row * 2u];
      y = positions[positionsOffset + row * 2u + 1u];
    }
    centers[centersOffset + index * 2u] = x;
    centers[centersOffset + index * 2u + 1u] = y;`
          })
        );
      } else {
        for (let round = 0; round < k; round++) {
          const roundId = `${restartId}-seed-${round}`;
          const roundSeed =
            (Math.imul(restartSeed, 0x9e3779b1) + Math.imul(round, 0x85ebca6b)) >>> 0;
          nodes.push(
            createFillNode<Parameters>(graph, {
              id: `${roundId}-clear-key`,
              operation: OPERATION,
              view: bestKey,
              type: 'u32',
              value: NO_ROW,
              componentCount: 1
            }),
            createFillNode<Parameters>(graph, {
              id: `${roundId}-clear-row`,
              operation: OPERATION,
              view: bestRow,
              type: 'u32',
              value: NO_ROW,
              componentCount: 1
            }),
            createWGSLKernelNode<Parameters>(graph, {
              id: `${roundId}-key`,
              operation: OPERATION,
              variant: 'seed-key',
              bindings: [
                {name: 'validFlags', view: validFlags, type: 'u32', access: 'read'},
                {name: 'minDistance', view: minDistance, type: 'f32', access: 'read'},
                {name: 'keys', view: keys, type: 'u32', access: 'read_write'},
                {name: 'bestKey', view: bestKey, type: 'atomic<u32>', access: 'read_write'}
              ],
              invocationCount: rows,
              guardIndex: false,
              declarations: `${sharedWGSL}
  const ROUND_SEED: u32 = ${roundSeed}u;
  const FIRST_ROUND: bool = ${round === 0};
  var<workgroup> groupBestKey: atomic<u32>;

  fn hashRow(row: u32) -> u32 {
    var h = row ^ ROUND_SEED;
    h = h ^ (h >> 16u);
    h = h * 0x7feb352du;
    h = h ^ (h >> 15u);
    h = h * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return h;
  }`,
              // Each workgroup folds its minimum in shared memory first, so the global word sees one
              // atomic per workgroup instead of one per point.
              body: `if (localInvocationIndex == 0u) {
      atomicStore(&groupBestKey, 0xffffffffu);
    }
    workgroupBarrier();
    var key = 0xffffffffu;
    if (index < INVOCATION_COUNT && validFlags[validFlagsOffset + index] != 0u) {
      // Exponential race: the minimum of -ln(u) / w samples a row with probability proportional to w.
      let weight = select(minDistance[minDistanceOffset + index], 1.0, FIRST_ROUND);
      let uniform = (f32(hashRow(index) >> 8u) + 0.5) / 16777216.0;
      key = select(0x7f800000u, bitcast<u32>(-log(uniform) / weight), weight > 0.0);
      atomicMin(&groupBestKey, key);
      keys[keysOffset + index] = key;
    } else if (index < INVOCATION_COUNT) {
      keys[keysOffset + index] = key;
    }
    workgroupBarrier();
    if (localInvocationIndex == 0u) {
      atomicMin(&bestKey[bestKeyOffset], atomicLoad(&groupBestKey));
    }`
            }),
            createWGSLKernelNode<Parameters>(graph, {
              id: `${roundId}-row`,
              operation: OPERATION,
              variant: 'seed-row',
              bindings: [
                {name: 'keys', view: keys, type: 'u32', access: 'read'},
                {name: 'bestKey', view: bestKey, type: 'u32', access: 'read'},
                {name: 'bestRow', view: bestRow, type: 'atomic<u32>', access: 'read_write'}
              ],
              invocationCount: rows,
              body: `let best = bestKey[bestKeyOffset];
    if (best != 0xffffffffu && keys[keysOffset + index] == best) {
      atomicMin(&bestRow[bestRowOffset], index);
    }`
            }),
            createWGSLKernelNode<Parameters>(graph, {
              id: `${roundId}-update`,
              operation: OPERATION,
              variant: 'seed-update',
              bindings: [
                {name: 'positions', view: positions, type: 'f32', access: 'read'},
                {name: 'validFlags', view: validFlags, type: 'u32', access: 'read'},
                {name: 'bestRow', view: bestRow, type: 'u32', access: 'read'},
                {name: 'minDistance', view: minDistance, type: 'f32', access: 'read_write'},
                {name: 'centers', view: workCenters, type: 'f32', access: 'read_write'}
              ],
              invocationCount: rows,
              declarations: `${sharedWGSL}
  const ROUND: u32 = ${round}u;`,
              body: `let row = bestRow[bestRowOffset];
    let hasCenter = row != 0xffffffffu;
    var centerX = getQuietNaN(index);
    var centerY = centerX;
    if (hasCenter) {
      centerX = positions[positionsOffset + row * 2u];
      centerY = positions[positionsOffset + row * 2u + 1u];
    }
    if (index == 0u) {
      centers[centersOffset + ROUND * 2u] = centerX;
      centers[centersOffset + ROUND * 2u + 1u] = centerY;
    }
    if (hasCenter && validFlags[validFlagsOffset + index] != 0u) {
      let deltaX = positions[positionsOffset + index * 2u] - centerX;
      let deltaY = positions[positionsOffset + index * 2u + 1u] - centerY;
      let distance = deltaX * deltaX + deltaY * deltaY;
      minDistance[minDistanceOffset + index] =
        select(min(minDistance[minDistanceOffset + index], distance), distance, ROUND == 0u);
    }`
            })
          );
        }
      }
    };
    const addIterations = (restartId: string) => {
      for (let iteration = 0; iteration < iterations; iteration++) {
        const iterationId = `${restartId}-iteration-${iteration}`;
        nodes.push(
          assignNode(`${iterationId}-assign`, false, workCenters, iterationDistances),
          createWGSLKernelNode<Parameters>(graph, {
            id: `${iterationId}-partial`,
            operation: OPERATION,
            variant: 'partial',
            bindings: [
              {name: 'labels', view: labels, type: 'u32', access: 'read'},
              {name: 'xs', view: xs, type: 'f32', access: 'read'},
              {name: 'ys', view: ys, type: 'f32', access: 'read'},
              {name: 'state', view: state, type: 'u32', access: 'read'},
              {name: 'partialCounts', view: partialCounts, type: 'u32', access: 'read_write'},
              {name: 'partialSumsX', view: partialSumsX, type: 'f32', access: 'read_write'},
              {name: 'partialSumsY', view: partialSumsY, type: 'f32', access: 'read_write'}
            ],
            workgroupSize: REDUCTION_WORKGROUP_SIZE,
            invocationCount: chunkCount * REDUCTION_WORKGROUP_SIZE,
            guardIndex: false,
            declarations: `${sharedWGSL}
  const ROWS: u32 = ${rows}u;
  const CHUNK_SIZE: u32 = ${chunkSize}u;
  const CHUNK_COUNT: u32 = ${chunkCount}u;
  const TILE_COUNT: u32 = ${chunkSize / REDUCTION_WORKGROUP_SIZE}u;
  const SLICES: u32 = ${slices}u;
  const LANES: u32 = ${slices * k}u;
  var<workgroup> tileLabels: array<u32, ${REDUCTION_WORKGROUP_SIZE}>;
  var<workgroup> tileXs: array<f32, ${REDUCTION_WORKGROUP_SIZE}>;
  var<workgroup> tileYs: array<f32, ${REDUCTION_WORKGROUP_SIZE}>;
  var<workgroup> laneCounts: array<u32, ${REDUCTION_WORKGROUP_SIZE}>;
  var<workgroup> laneSumsX: array<f32, ${REDUCTION_WORKGROUP_SIZE}>;
  var<workgroup> laneSumsY: array<f32, ${REDUCTION_WORKGROUP_SIZE}>;`,
            // Every barrier sits in uniform control flow: the tile loop has a constant trip count and
            // the converged / out-of-range cases only change the data (noise labels match no cluster).
            body: `let chunk = index / ${REDUCTION_WORKGROUP_SIZE}u;
    let converged = state[stateOffset] != 0u;
    let begin = chunk * CHUNK_SIZE;
    let end = min(begin + CHUNK_SIZE, ROWS);
    let lane = localInvocationIndex;
    let cluster = lane % K;
    let slice = lane / K;
    var count = 0u;
    var sumX = 0.0;
    var sumY = 0.0;
    for (var tile = 0u; tile < TILE_COUNT; tile++) {
      let row = begin + tile * ${REDUCTION_WORKGROUP_SIZE}u + lane;
      var label = NOISE;
      var x = 0.0;
      var y = 0.0;
      if (!converged && row < end) {
        label = labels[labelsOffset + row];
        x = xs[xsOffset + row];
        y = ys[ysOffset + row];
      }
      tileLabels[lane] = label;
      tileXs[lane] = x;
      tileYs[lane] = y;
      workgroupBarrier();
      if (lane < LANES) {
        // Lane (slice, cluster) owns tile entries slice, slice + SLICES, ...: a fixed order.
        for (var entry = slice; entry < ${REDUCTION_WORKGROUP_SIZE}u; entry += SLICES) {
          if (tileLabels[entry] == cluster) {
            count++;
            sumX += tileXs[entry];
            sumY += tileYs[entry];
          }
        }
      }
      workgroupBarrier();
    }
    laneCounts[lane] = count;
    laneSumsX[lane] = sumX;
    laneSumsY[lane] = sumY;
    workgroupBarrier();
    if (lane < K) {
      var chunkCountTotal = 0u;
      var chunkSumX = 0.0;
      var chunkSumY = 0.0;
      for (var sliceIndex = 0u; sliceIndex < SLICES; sliceIndex++) {
        chunkCountTotal += laneCounts[sliceIndex * K + lane];
        chunkSumX += laneSumsX[sliceIndex * K + lane];
        chunkSumY += laneSumsY[sliceIndex * K + lane];
      }
      let slot = lane * CHUNK_COUNT + chunk;
      partialCounts[partialCountsOffset + slot] = chunkCountTotal;
      partialSumsX[partialSumsXOffset + slot] = chunkSumX;
      partialSumsY[partialSumsYOffset + slot] = chunkSumY;
    }`
          }),
          createWGSLKernelNode<Parameters>(graph, {
            id: `${iterationId}-update`,
            operation: OPERATION,
            variant: 'update',
            bindings: [
              {name: 'partialCounts', view: partialCounts, type: 'u32', access: 'read'},
              {name: 'partialSumsX', view: partialSumsX, type: 'f32', access: 'read'},
              {name: 'partialSumsY', view: partialSumsY, type: 'f32', access: 'read'},
              {name: 'sizes', view: workSizes, type: 'u32', access: 'read_write'},
              {name: 'clusterSums', view: clusterSums, type: 'f32', access: 'read_write'},
              {name: 'state', view: state, type: 'u32', access: 'read'}
            ],
            workgroupSize: REDUCTION_WORKGROUP_SIZE,
            invocationCount: k * REDUCTION_WORKGROUP_SIZE,
            guardIndex: false,
            declarations: `const CHUNK_COUNT: u32 = ${chunkCount}u;
  var<workgroup> reducedCounts: array<u32, ${REDUCTION_WORKGROUP_SIZE}>;
  var<workgroup> reducedSumsX: array<f32, ${REDUCTION_WORKGROUP_SIZE}>;
  var<workgroup> reducedSumsY: array<f32, ${REDUCTION_WORKGROUP_SIZE}>;`,
            // One workgroup per cluster: strided per-lane sums over the chunk partials, then a fixed
            // binary tree, so the sums are bitwise reproducible. Lane 0 publishes the member count
            // and coordinate sums; the relocate kernel turns them into means.
            body: `let cluster = index / ${REDUCTION_WORKGROUP_SIZE}u;
    let lane = localInvocationIndex;
    var count = 0u;
    var sumX = 0.0;
    var sumY = 0.0;
    for (var chunk = lane; chunk < CHUNK_COUNT; chunk += ${REDUCTION_WORKGROUP_SIZE}u) {
      let slot = cluster * CHUNK_COUNT + chunk;
      count += partialCounts[partialCountsOffset + slot];
      sumX += partialSumsX[partialSumsXOffset + slot];
      sumY += partialSumsY[partialSumsYOffset + slot];
    }
    reducedCounts[lane] = count;
    reducedSumsX[lane] = sumX;
    reducedSumsY[lane] = sumY;
    workgroupBarrier();
    for (var stride = ${REDUCTION_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride / 2u) {
      if (lane < stride) {
        reducedCounts[lane] += reducedCounts[lane + stride];
        reducedSumsX[lane] += reducedSumsX[lane + stride];
        reducedSumsY[lane] += reducedSumsY[lane + stride];
      }
      workgroupBarrier();
    }
    // Converged: centers no longer move; the partials are stale and the final sizes come later.
    if (lane == 0u && state[stateOffset] == 0u) {
      sizes[sizesOffset + cluster] = reducedCounts[0];
      clusterSums[clusterSumsOffset + cluster * 2u] = reducedSumsX[0];
      clusterSums[clusterSumsOffset + cluster * 2u + 1u] = reducedSumsY[0];
    }`
          }),
          // Empty-cluster relocation and the center update, in one workgroup (k <= 256 lanes).
          // scikit-learn's `_relocate_empty_clusters_dense`: the points farthest from the center they
          // were assigned to (largest squared distance first, lowest row on ties) become the centers
          // of the empty clusters (lowest cluster ID gets the farthest point), and the donor
          // clusters' counts and sums lose those points. Selection is an iterated argmax that
          // excludes the previous winner by (distance, row) rank, so it needs no atomics.
          createWGSLKernelNode<Parameters>(graph, {
            id: `${iterationId}-relocate`,
            operation: OPERATION,
            variant: 'relocate',
            bindings: [
              {name: 'positions', view: positions, type: 'f32', access: 'read'},
              {name: 'labels', view: labels, type: 'u32', access: 'read'},
              {name: 'distances', view: iterationDistances, type: 'f32', access: 'read'},
              {name: 'clusterSums', view: clusterSums, type: 'f32', access: 'read'},
              {name: 'sizes', view: workSizes, type: 'u32', access: 'read_write'},
              {name: 'centers', view: workCenters, type: 'f32', access: 'read_write'},
              {name: 'state', view: state, type: 'atomic<u32>', access: 'read_write'}
            ],
            workgroupSize: REDUCTION_WORKGROUP_SIZE,
            invocationCount: REDUCTION_WORKGROUP_SIZE,
            guardIndex: false,
            declarations: `${sharedWGSL}
  const ROWS: u32 = ${rows}u;
  var<workgroup> activeFlag: u32;
  var<workgroup> stopFlag: u32;
  var<workgroup> emptyCount: u32;
  var<workgroup> pickDistance: f32;
  var<workgroup> pickRow: u32;
  var<workgroup> clusterCounts: array<u32, ${REDUCTION_WORKGROUP_SIZE}>;
  var<workgroup> clusterSumsX: array<f32, ${REDUCTION_WORKGROUP_SIZE}>;
  var<workgroup> clusterSumsY: array<f32, ${REDUCTION_WORKGROUP_SIZE}>;
  var<workgroup> emptyClusters: array<u32, ${REDUCTION_WORKGROUP_SIZE}>;
  var<workgroup> laneDistances: array<f32, ${REDUCTION_WORKGROUP_SIZE}>;
  var<workgroup> laneRows: array<u32, ${REDUCTION_WORKGROUP_SIZE}>;`,
            // Barriers sit in uniform control flow: every branch that contains one is decided by
            // `workgroupUniformLoad` of a flag only lane 0 writes.
            body: `let lane = localInvocationIndex;
  if (lane == 0u) {
    activeFlag = select(0u, 1u, atomicLoad(&state[stateOffset]) == 0u);
  }
  // Converged: the partials are stale and the centers no longer move.
  if (workgroupUniformLoad(&activeFlag) == 0u) {
    return;
  }
  if (lane < K) {
    clusterCounts[lane] = sizes[sizesOffset + lane];
    clusterSumsX[lane] = clusterSums[clusterSumsOffset + lane * 2u];
    clusterSumsY[lane] = clusterSums[clusterSumsOffset + lane * 2u + 1u];
  }
  workgroupBarrier();
  if (lane == 0u) {
    // Clusters whose center could not be initialized (NaN) are not empty clusters to relocate.
    var count = 0u;
    for (var cluster = 0u; cluster < K; cluster++) {
      let centerX = centers[centersOffset + cluster * 2u];
      if (clusterCounts[cluster] == 0u && isFiniteFloat(centerX)) {
        emptyClusters[count] = cluster;
        count++;
      }
    }
    emptyCount = count;
    stopFlag = 0u;
    pickDistance = 0.0;
    pickRow = 0u;
  }
  let pickLimit = workgroupUniformLoad(&emptyCount);
  for (var pickIndex = 0u; pickIndex < pickLimit; pickIndex++) {
    // scikit-learn returns without relocating when every point sits on its center.
    if (workgroupUniformLoad(&stopFlag) != 0u) {
      break;
    }
    let previousDistance = pickDistance;
    let previousRow = pickRow;
    var bestDistance = -1.0;
    var bestRow = NOISE;
    for (var row = lane; row < ROWS; row += ${REDUCTION_WORKGROUP_SIZE}u) {
      let distance = distances[distancesOffset + row];
      if (isFiniteFloat(distance) && labels[labelsOffset + row] != NOISE) {
        // Rank (distance descending, row ascending) strictly after the previous winner.
        let ranked = pickIndex == 0u || distance < previousDistance ||
          (distance == previousDistance && row > previousRow);
        if (ranked && distance > bestDistance) {
          bestDistance = distance;
          bestRow = row;
        }
      }
    }
    laneDistances[lane] = bestDistance;
    laneRows[lane] = bestRow;
    workgroupBarrier();
    for (var stride = ${REDUCTION_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride / 2u) {
      if (lane < stride) {
        let otherDistance = laneDistances[lane + stride];
        let otherRow = laneRows[lane + stride];
        if (otherDistance > laneDistances[lane] ||
          (otherDistance == laneDistances[lane] && otherRow < laneRows[lane])) {
          laneDistances[lane] = otherDistance;
          laneRows[lane] = otherRow;
        }
      }
      workgroupBarrier();
    }
    if (lane == 0u) {
      let row = laneRows[0];
      let distance = laneDistances[0];
      if (row == NOISE || (pickIndex == 0u && distance == 0.0)) {
        stopFlag = 1u;
      } else {
        pickDistance = distance;
        pickRow = row;
        let emptyCluster = emptyClusters[pickIndex];
        let donor = labels[labelsOffset + row];
        let x = positions[positionsOffset + row * 2u];
        let y = positions[positionsOffset + row * 2u + 1u];
        clusterSumsX[donor] -= x;
        clusterSumsY[donor] -= y;
        clusterCounts[donor] -= 1u;
        clusterSumsX[emptyCluster] = x;
        clusterSumsY[emptyCluster] = y;
        clusterCounts[emptyCluster] = 1u;
      }
    }
    workgroupBarrier();
  }
  // A cluster still empty (every point on its center, or its only member was taken) moves to the
  // mean of the largest cluster, lowest ID on ties, like scikit-learn's \`_average_centers\`.
  var largest = 0u;
  for (var cluster = 1u; cluster < K; cluster++) {
    if (clusterCounts[cluster] > clusterCounts[largest]) {
      largest = cluster;
    }
  }
  if (lane < K) {
    let size = clusterCounts[lane];
    let previousX = centers[centersOffset + lane * 2u];
    let previousY = centers[centersOffset + lane * 2u + 1u];
    var nextX = previousX;
    var nextY = previousY;
    if (size > 0u) {
      nextX = clusterSumsX[lane] / f32(size);
      nextY = clusterSumsY[lane] / f32(size);
    } else if (isFiniteFloat(previousX) && clusterCounts[largest] > 0u) {
      nextX = clusterSumsX[largest] / f32(clusterCounts[largest]);
      nextY = clusterSumsY[largest] / f32(clusterCounts[largest]);
    }
    sizes[sizesOffset + lane] = size;
    let deltaX = nextX - previousX;
    let deltaY = nextY - previousY;
    let shift = sqrt(deltaX * deltaX + deltaY * deltaY);
    if (isFiniteFloat(shift)) {
      atomicMax(&state[stateOffset + 2u], bitcast<u32>(shift));
    }
    centers[centersOffset + lane * 2u] = nextX;
    centers[centersOffset + lane * 2u + 1u] = nextY;
  }`
          }),
          createWGSLKernelNode<Parameters>(graph, {
            id: `${iterationId}-converge`,
            operation: OPERATION,
            variant: 'converge',
            bindings: [{name: 'state', view: state, type: 'u32', access: 'read_write'}],
            invocationCount: 1,
            declarations: `const TOLERANCE: f32 = ${formatWGSLFloat(tolerance)};`,
            body: `if (state[stateOffset] != 0u) {
      return;
    }
    state[stateOffset + 1u] = state[stateOffset + 1u] + 1u;
    if (bitcast<f32>(state[stateOffset + 2u]) <= TOLERANCE) {
      state[stateOffset] = 1u;
    }
    state[stateOffset + 2u] = 0u;`
          })
        );
      }
    };
    // Scores one restart: a fixed-order compensated sum of the squared distances, then (with
    // restarts) a keep-best step that copies this restart's centers and state when its inertia is
    // strictly lower than the best so far (restart 0 always wins the empty comparison).
    const addScore = (scoreId: string, restart: number) => {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: scoreId,
          operation: OPERATION,
          variant: multiStart ? 'restart-score' : 'inertia',
          bindings: [
            {name: 'squaredDistances', view: finalDistances!, type: 'f32', access: 'read'},
            {name: 'inertia', view: bestInertia, type: 'f32', access: 'read_write'},
            ...(multiStart
              ? [
                  {
                    name: 'centers',
                    view: workCenters,
                    type: 'f32' as const,
                    access: 'read' as const
                  },
                  {
                    name: 'bestCenters',
                    view: bestCenters!,
                    type: 'f32' as const,
                    access: 'read_write' as const
                  },
                  {name: 'state', view: state, type: 'u32' as const, access: 'read' as const},
                  {
                    name: 'bestState',
                    view: bestState!,
                    type: 'u32' as const,
                    access: 'read_write' as const
                  }
                ]
              : [])
          ],
          workgroupSize: REDUCTION_WORKGROUP_SIZE,
          invocationCount: REDUCTION_WORKGROUP_SIZE,
          guardIndex: false,
          declarations: `${sharedWGSL}
const ROWS: u32 = ${rows}u;
const FIRST_RESTART: bool = ${restart === 0};
var<workgroup> laneSums: array<f32, ${REDUCTION_WORKGROUP_SIZE}>;
var<workgroup> laneCounts: array<u32, ${REDUCTION_WORKGROUP_SIZE}>;
var<workgroup> improves: u32;`,
          body: `let lane = localInvocationIndex;
  var sum = 0.0;
  var compensation = 0.0;
  var count = 0u;
  for (var row = lane; row < ROWS; row += ${REDUCTION_WORKGROUP_SIZE}u) {
    let value = squaredDistances[squaredDistancesOffset + row];
    if (isFiniteFloat(value)) {
      let corrected = value - compensation;
      let next = sum + corrected;
      compensation = (next - sum) - corrected;
      sum = next;
      count++;
    }
  }
  laneSums[lane] = sum;
  laneCounts[lane] = count;
  workgroupBarrier();
  for (var stride = ${REDUCTION_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      laneSums[lane] += laneSums[lane + stride];
      laneCounts[lane] += laneCounts[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u) {
    let total = select(getQuietNaN(0u), laneSums[0], laneCounts[0] > 0u);${
      multiStart
        ? `
    let previous = inertia[inertiaOffset];
    // Strictly lower wins; a finite score also replaces a NaN one.
    let better = FIRST_RESTART || total < previous || (!isFiniteFloat(previous) && isFiniteFloat(total));
    improves = select(0u, 1u, better);
    if (better) {
      inertia[inertiaOffset] = total;
      bestState[bestStateOffset] = state[stateOffset];
      bestState[bestStateOffset + 1u] = state[stateOffset + 1u];
    }`
        : `
    inertia[inertiaOffset] = total;`
    }
  }${
    multiStart
      ? `
  workgroupBarrier();
  if (improves != 0u) {
    for (var slot = lane; slot < ${2 * k}u; slot += ${REDUCTION_WORKGROUP_SIZE}u) {
      bestCenters[bestCentersOffset + slot] = centers[centersOffset + slot];
    }
  }`
      : ''
  }`
        })
      );
    };
    for (let restart = 0; restart < restartCount; restart++) {
      const restartId = multiStart ? `${id}-restart-${restart}` : id;
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${restartId}-state-clear`,
          operation: OPERATION,
          view: state,
          type: 'u32',
          value: '0u'
        })
      );
      addInitialization(restartId, getKMeansRestartSeed(seed, restart));
      addIterations(restartId);
      if (multiStart) {
        // Final assignment of this restart against its own centers, then score and keep the best.
        nodes.push(assignNode(`${restartId}-score-assign`, true, workCenters, finalDistances));
        addScore(`${restartId}-score`, restart);
      }
    }
    if (multiStart) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-publish-centers`,
          operation: OPERATION,
          variant: 'publish-centers',
          bindings: [
            {name: 'bestCenters', view: bestCenters!, type: 'f32', access: 'read'},
            {name: 'centers', view: centers, type: 'f32', access: 'read_write'}
          ],
          invocationCount: 2 * k,
          body: `centers[centersOffset + index] = bestCenters[bestCentersOffset + index];`
        })
      );
    }
    nodes.push(
      assignNode(`${id}-final-assign`, true, centers, finalDistances),
      ...new GPUGroupAggregation({
        id: `${id}-final-sizes`,
        keys: labels,
        output: sizes
      }).getCommandNodes(graph)
    );
    if (props.inertia && !multiStart) {
      addScore(`${id}-inertia`, 0);
    }
    if (props.convergence) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-publish-convergence`,
          operation: OPERATION,
          variant: 'publish-convergence',
          bindings: [
            {name: 'state', view: multiStart ? bestState! : state, type: 'u32', access: 'read'},
            {name: 'convergence', view: props.convergence, type: 'u32', access: 'read_write'}
          ],
          invocationCount: 2,
          body: `convergence[convergenceOffset + index] = state[stateOffset + 1u - index];`
        })
      );
    }
    return nodes;
  }
}
