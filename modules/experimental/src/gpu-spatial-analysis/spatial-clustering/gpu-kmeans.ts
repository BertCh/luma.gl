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
import {getSortedSegmentSumNodes} from '../../utils/sorted-segment-sums';
import {GPU_SPATIAL_CLUSTERING_NOISE} from './spatial-clustering-parameters';

const OPERATION = 'GPUKMeans';
const NO_ROW = '0xffffffffu';

/** Largest `k`. */
export const GPU_KMEANS_MAXIMUM_CLUSTERS = 256;

/** Largest `iterations`: every Lloyd iteration adds a sort and several scans to the graph. */
export const GPU_KMEANS_MAXIMUM_ITERATIONS = 64;

/** How {@link GPUKMeans} picks the initial centers. */
export type GPUKMeansInitialization = 'first-valid' | 'kmeans++';

/**
 * Properties for {@link GPUKMeans}.
 *
 * Per-frame (no recompile): the contents of `positions`. Compile-time: `positions.length`, `k`,
 * `iterations`, `initialization`, `seed` and which optional views are present.
 */
export type GPUKMeansProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'kmeans'`. */
  id?: string;
  /** Packed planar points, one row per point. Rows with a non-finite coordinate are noise. */
  positions: GraphDataView<'float32x2'>;
  /** Number of clusters, an integer in `[1, 256]`. Compile-time. */
  k: number;
  /**
   * Number of Lloyd iterations (assign then update), an integer in `[1, 64]`. Compile-time and
   * fixed: there is no convergence test, which keeps the result a pure function of the inputs.
   */
  iterations: number;
  /**
   * `'first-valid'` (default) uses the `k` lowest-index points with finite coordinates as the
   * initial centers. `'kmeans++'` draws them with D-squared weighting from a seeded hash, so
   * the same `seed` always gives the same centers on one device.
   */
  initialization?: GPUKMeansInitialization;
  /** Seed of the `'kmeans++'` hash, an integer in `[0, 2^32 - 1]`. Default 0. */
  seed?: number;
  /** Caller-owned labels, one row per point: cluster `0..k-1`, or `0xffffffff` for noise. */
  labels: GraphDataView<'uint32'>;
  /**
   * Caller-owned centers, `k` rows. A center that never received a point keeps its previous
   * position; a center that could not be initialized (fewer than `k` valid points with
   * `'first-valid'`) is NaN and stays empty.
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
 * Every choice that could vary between runs is fixed:
 * - Initial centers are the first `k` valid points (lowest index first) or `k-means++` with a
 *   seeded integer hash: each round samples the next center with probability proportional to the
 *   squared distance to the nearest chosen center, by an exponential race `-ln(u) / w` whose
 *   minimum is found with integer atomics, lowest row first on ties.
 * - Each of the `iterations` rounds assigns every point to the nearest center, the lowest center
 *   ID winning ties, and moves each center to the mean of its members. Means use the fixed-order
 *   sorted segmented sum, so sums are bitwise reproducible on one device. A final assignment
 *   against the last centers produces `labels`, `sizes` and `squaredDistances`.
 *
 * Results are reproducible for identical inputs on one device. `log` is not exactly specified by
 * WGSL, so `'kmeans++'` draws may differ between devices; `'first-valid'` has no such caveat apart
 * from f32 distance rounding.
 *
 * Cost: one sort of all points per iteration, so memory grows with `iterations * points`.
 * Non-goals: convergence tests, empty-cluster reseeding, weighted points, 3D points, k-medoids.
 */
export class GPUKMeans implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUKMeansProps;
  /** Resolved initialization. */
  readonly initialization: GPUKMeansInitialization;

  constructor(props: GPUKMeansProps) {
    const id = props.id ?? 'kmeans';
    this.id = id;
    this.props = props;
    this.initialization = props.initialization ?? 'first-valid';
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
      props.seed !== undefined &&
      (!Number.isInteger(props.seed) || props.seed < 0 || props.seed > 0xffffffff)
    ) {
      throw new Error(`${id} seed must be an integer in [0, 2^32 - 1]`);
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
      [props.labels, props.centers, props.sizes, props.squaredDistances],
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
      props.squaredDistances
    ]);
    const rows = positions.length;
    const seed = props.seed ?? 0;
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
    const sumsX = view('sums-x', 'float32', k);
    const sumsY = view('sums-y', 'float32', k);
    const sharedWGSL = `const K: u32 = ${k}u;
const NOISE: u32 = ${GPU_SPATIAL_CLUSTERING_NOISE}u;

fn isFiniteFloat(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

fn getQuietNaN(seed: u32) -> f32 {
  return bitcast<f32>(0x7fc00000u | (seed & 0u));
}`;

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

    if (this.initialization === 'first-valid') {
      const compactIds = view('compact-ids', 'uint32', rows);
      const validCount = view('valid-count', 'uint32', 1);
      nodes.push(
        ...new GPUCompaction({
          id: `${id}-valid-rows`,
          input: rowIds,
          flags: validFlags,
          output: compactIds,
          count: validCount
        }).getCommandNodes(graph),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-initial-centers`,
          operation: OPERATION,
          variant: 'initial-centers',
          bindings: [
            {name: 'positions', view: positions, type: 'f32', access: 'read'},
            {name: 'compactIds', view: compactIds, type: 'u32', access: 'read'},
            {name: 'validCount', view: validCount, type: 'u32', access: 'read'},
            {name: 'centers', view: centers, type: 'f32', access: 'read_write'}
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
      const minDistance = view('min-distance', 'float32', rows);
      const keys = view('keys', 'uint32', rows);
      const bestKey = view('best-key', 'uint32', 1);
      const bestRow = view('best-row', 'uint32', 1);
      for (let round = 0; round < k; round++) {
        const roundId = `${id}-seed-${round}`;
        const roundSeed = (Math.imul(seed, 0x9e3779b1) + Math.imul(round, 0x85ebca6b)) >>> 0;
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
            declarations: `${sharedWGSL}
const ROUND_SEED: u32 = ${roundSeed}u;
const FIRST_ROUND: bool = ${round === 0};

fn hashRow(row: u32) -> u32 {
  var h = row ^ ROUND_SEED;
  h = h ^ (h >> 16u);
  h = h * 0x7feb352du;
  h = h ^ (h >> 15u);
  h = h * 0x846ca68bu;
  h = h ^ (h >> 16u);
  return h;
}`,
            body: `var key = 0xffffffffu;
  if (validFlags[validFlagsOffset + index] != 0u) {
    // Exponential race: the minimum of -ln(u) / w samples a row with probability proportional to w.
    let weight = select(minDistance[minDistanceOffset + index], 1.0, FIRST_ROUND);
    let uniform = (f32(hashRow(index) >> 8u) + 0.5) / 16777216.0;
    key = select(0x7f800000u, bitcast<u32>(-log(uniform) / weight), weight > 0.0);
    atomicMin(&bestKey[bestKeyOffset], key);
  }
  keys[keysOffset + index] = key;`
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
              {name: 'centers', view: centers, type: 'f32', access: 'read_write'}
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

    const assignNode = (assignId: string, withDistances: boolean): GPUCommandNode<Parameters> =>
      createWGSLKernelNode<Parameters>(graph, {
        id: assignId,
        operation: OPERATION,
        variant: withDistances ? 'assign-final' : 'assign',
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {name: 'centers', view: centers, type: 'f32', access: 'read'},
          {name: 'labels', view: labels, type: 'u32', access: 'read_write'},
          ...(withDistances && props.squaredDistances
            ? [
                {
                  name: 'squaredDistances',
                  view: props.squaredDistances,
                  type: 'f32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: rows,
        declarations: sharedWGSL,
        body: `let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  var best = NOISE;
  var bestDistance = getQuietNaN(index);
  if (isFiniteFloat(x) && isFiniteFloat(y)) {
    // Strict comparison: the lowest center ID wins ties; NaN centers never win.
    for (var center = 0u; center < K; center++) {
      let deltaX = centers[centersOffset + center * 2u] - x;
      let deltaY = centers[centersOffset + center * 2u + 1u] - y;
      let distance = deltaX * deltaX + deltaY * deltaY;
      if (distance == distance && (best == NOISE || distance < bestDistance)) {
        best = center;
        bestDistance = distance;
      }
    }
  }
  labels[labelsOffset + index] = best;
  ${withDistances && props.squaredDistances ? 'squaredDistances[squaredDistancesOffset + index] = select(getQuietNaN(index), bestDistance, best != NOISE);' : ''}`
      });

    for (let iteration = 0; iteration < iterations; iteration++) {
      const iterationId = `${id}-iteration-${iteration}`;
      nodes.push(
        assignNode(`${iterationId}-assign`, false),
        ...new GPUGroupAggregation({
          id: `${iterationId}-sizes`,
          keys: labels,
          output: sizes
        }).getCommandNodes(graph),
        ...getSortedSegmentSumNodes<Parameters>(graph, {
          id: `${iterationId}-sums`,
          operation: OPERATION,
          segmentCount: k,
          segmentKeys: labels,
          segmentCounts: sizes,
          reductions: [
            {name: 'x', contributions: xs, output: sumsX},
            {name: 'y', contributions: ys, output: sumsY}
          ]
        }),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${iterationId}-update`,
          operation: OPERATION,
          variant: 'update',
          bindings: [
            {name: 'sizes', view: sizes, type: 'u32', access: 'read'},
            {name: 'sumsX', view: sumsX, type: 'f32', access: 'read'},
            {name: 'sumsY', view: sumsY, type: 'f32', access: 'read'},
            {name: 'centers', view: centers, type: 'f32', access: 'read_write'}
          ],
          invocationCount: k,
          body: `let size = sizes[sizesOffset + index];
  if (size > 0u) {
    centers[centersOffset + index * 2u] = sumsX[sumsXOffset + index] / f32(size);
    centers[centersOffset + index * 2u + 1u] = sumsY[sumsYOffset + index] / f32(size);
  }`
        })
      );
    }
    nodes.push(
      assignNode(`${id}-final-assign`, true),
      ...new GPUGroupAggregation({
        id: `${id}-final-sizes`,
        keys: labels,
        output: sizes
      }).getCommandNodes(graph)
    );
    return nodes;
  }
}
