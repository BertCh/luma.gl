// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  assertCompatibleGPUSpatialContexts,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {GPU_LOCAL_OUTLIER_FACTOR_PARAMETER_LENGTH} from './outlier-detection-parameters';

import {IS_NAN_BITS_WGSL} from '../../utils/wgsl-common';

const OPERATION = 'GPULocalOutlierFactor';

/** Caller-owned outputs of {@link GPULocalOutlierFactor}. */
export type GPULocalOutlierFactorOutput = {
  /** k-distance per row: the largest neighbor distance, 0 for an empty row. */
  kDistance: GraphDataView<'float32'>;
  /**
   * Local reachability density per row: the inverse of the row's mean reachability distance plus
   * `densityFloor`. 0 for an empty row.
   */
  localReachabilityDensity: GraphDataView<'float32'>;
  /** Local outlier factor per row. 1 for a row with no neighbors. */
  lof: GraphDataView<'float32'>;
  /**
   * Optional mask: 1 where `lof > threshold`, else 0. All 0 when no `threshold` was packed (there
   * is no default threshold).
   */
  outlier?: GraphDataView<'uint32'>;
  /** Optional one-row count of rows with `lof > threshold`. Requires `outlier`. */
  outlierCount?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPULocalOutlierFactor}.
 *
 * Per-frame (no recompile): the contents of `neighbors` (including its distances) and
 * `parameters`. Compile-time: the row count, view lengths and which optional outputs exist.
 */
export type GPULocalOutlierFactorProps = {
  /** Prefix for generated node IDs. Defaults to `'local-outlier-factor'`. */
  id?: string;
  /**
   * Square neighbor table with `distances`: row `i` lists the neighbors of point `i` and their
   * distances. Typically the k-nearest-neighbor CSR written by `GPUNeighborSearch` in `'knn'`
   * mode (self join, `weights.distances` set; the weight values are unused), or one uploaded
   * from a CPU index. The neighbor count of each row is its `k`; rows must come from kNN so the
   * row's largest distance is the k-distance. A radius-mode list is not a kNN list and gives a
   * different, density-dependent statistic. With ties at the k-th distance the k-distance
   * neighborhood has more than `k` members (Breunig et al. 2000); a list cut at exactly `k`
   * entries drops the tied points and changes the scores (scikit-learn does the same). Distances
   * are used as given; `GPUNeighborSearch` carries their `spatialContext` on the neighbors.
   * Neighbor IDs at or above the row count are skipped.
   */
  neighbors: GPUSpatialWeights;
  /** Context of the packed densityFloor, checked against neighbors.spatialContext when both exist. */
  spatialContext?: GPUSpatialContext;
  /**
   * Per-frame parameters: packed float32 view of at least
   * `GPU_LOCAL_OUTLIER_FACTOR_PARAMETER_LENGTH` elements written with
   * `getGPULocalOutlierFactorParameterValues`.
   */
  parameters: GraphDataView<'float32'>;
  /** Caller-owned outputs. */
  output: GPULocalOutlierFactorOutput;
};

/**
 * Local Outlier Factor (Breunig et al. 2000) over a kNN table: the geo crate's
 * `OutlierDetection` and scikit-learn's `LocalOutlierFactor`, whose `-negative_outlier_factor_`
 * the `lof` output matches.
 *
 * Definition, for row `i` with neighbors `N(i)` at distances `d(i, j)`:
 * - `kDistance(i)` is the largest `d(i, j)`.
 * - `reach(i, j) = max(kDistance(j), d(i, j))`.
 * - `localReachabilityDensity(i) = 1 / (mean_j reach(i, j) + densityFloor)`.
 * - `lof(i) = mean_j lrd(j) / lrd(i)`.
 * Scores near 1 are as dense as their neighbors; scores well above 1 are locally sparse.
 *
 * Exploratory, not inferential: LOF has no null distribution, so there is no p-value, no default
 * cut and no "significant" outlier. The optional `outlier` mask applies a threshold the caller
 * supplies in `getGPULocalOutlierFactorParameterValues` (without one the mask is all 0), and
 * `lof` itself is the result to inspect. Guidance on `k` (the neighbor count of the table):
 * use `k >= 10` so scores are stable, and larger than the smallest group of points that should
 * count as a cluster rather than as outliers (Breunig et al. 2000); the score scale shifts with
 * `k`, so compare scores only across runs with the same `k`.
 *
 * Coordinates: neighbors carry their own `spatialContext` from `GPUNeighborSearch`. LOF is a
 * unit-free ratio, but `densityFloor` is added to a distance: its optional `spatialContext`
 * is checked against the neighbors' context when both are supplied.
 *
 * Duplicates: `densityFloor` (default 1e-10, scikit-learn's value) keeps a cluster of coincident
 * points, whose mean reachability is 0, finite at density `1 / densityFloor`; points whose
 * neighbors are such duplicates then get enormous scores, as in scikit-learn. Use `k` larger
 * than the duplicate count or a larger floor to avoid that.
 *
 * A row with no neighbors (for example a masked-out point) gets `kDistance` 0, density 0 and
 * `lof` 1 (never an outlier), and contributes density 0 to rows that list it.
 *
 * Determinism: each row accumulates its neighbors in slot order (ascending ID) in f32, the
 * count uses integer atomics, and no float atomics are used, so results are reproducible and
 * match a float64 oracle within f32 rounding (about 1e-6 relative for modest `k`).
 */
export class GPULocalOutlierFactor implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPULocalOutlierFactorProps;

  constructor(props: GPULocalOutlierFactorProps) {
    const id = props.id ?? 'local-outlier-factor';
    this.id = id;
    this.props = props;
    if (props.neighbors.spatialContext && props.spatialContext) {
      assertCompatibleGPUSpatialContexts(id, props.neighbors.spatialContext, props.spatialContext);
    }
    const rows = validateGPUSpatialWeights(id, props.neighbors, 'neighbors');
    if (!props.neighbors.distances) {
      throw new Error(`${id} neighbors.distances is required`);
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_LOCAL_OUTLIER_FACTOR_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_LOCAL_OUTLIER_FACTOR_PARAMETER_LENGTH} float32 values`
      );
    }
    const {output} = props;
    for (const [name, view] of [
      ['kDistance', output.kDistance],
      ['localReachabilityDensity', output.localReachabilityDensity],
      ['lof', output.lof]
    ] as const) {
      validatePackedView(view, ['float32'], `${id} output.${name}`);
      if (view.length !== rows) {
        throw new Error(`${id} output.${name} length must equal the neighbors row count`);
      }
    }
    if (output.outlier) {
      validatePackedView(output.outlier, ['uint32'], `${id} output.outlier`);
      if (output.outlier.length !== rows) {
        throw new Error(`${id} output.outlier length must equal the neighbors row count`);
      }
    }
    if (output.outlierCount) {
      validatePackedView(output.outlierCount, ['uint32'], `${id} output.outlierCount`);
      if (output.outlierCount.length < 1) {
        throw new Error(`${id} output.outlierCount must hold one uint32`);
      }
      if (!output.outlier) {
        throw new Error(`${id} output.outlierCount requires output.outlier`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.kDistance,
        output.localReachabilityDensity,
        output.lof,
        output.outlier,
        output.outlierCount
      ],
      [
        props.neighbors.offsets,
        props.neighbors.neighbors,
        props.neighbors.distances,
        props.parameters
      ]
    );
  }

  /** Returns the k-distance, density, score and optional outlier nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {neighbors, parameters, output} = props;
    const {kDistance, localReachabilityDensity, lof} = output;
    const distances = neighbors.distances!;
    validateGraphViewsBelongToGraph(id, graph, [
      neighbors.offsets,
      neighbors.neighbors,
      distances,
      parameters,
      kDistance,
      localReachabilityDensity,
      lof,
      output.outlier,
      output.outlierCount
    ]);
    const rows = neighbors.offsets.length - 1;
    const tableBindings = [
      {name: 'offsets', view: neighbors.offsets, type: 'u32' as const, access: 'read' as const},
      {name: 'neighbors', view: neighbors.neighbors, type: 'u32' as const, access: 'read' as const},
      {name: 'distances', view: distances, type: 'f32' as const, access: 'read' as const}
    ];
    const declarations = `const ROWS: u32 = ${rows}u;`;
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-k-distance`,
        operation: OPERATION,
        variant: 'k-distance',
        bindings: [
          ...tableBindings,
          {name: 'kDistance', view: kDistance, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rows,
        body: `var largest = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    largest = max(largest, distances[distancesOffset + slot]);
  }
  kDistance[kDistanceOffset + index] = largest;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-density`,
        operation: OPERATION,
        variant: 'density',
        bindings: [
          ...tableBindings,
          {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
          {name: 'kDistance', view: kDistance, type: 'f32', access: 'read'},
          {
            name: 'density',
            view: localReachabilityDensity,
            type: 'f32',
            access: 'read_write'
          }
        ],
        invocationCount: rows,
        declarations,
        body: `var sum = 0.0;
  var count = 0u;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor < ROWS) {
      sum += max(kDistance[kDistanceOffset + neighbor], distances[distancesOffset + slot]);
      count++;
    }
  }
  density[densityOffset + index] =
    select(0.0, 1.0 / (sum / f32(max(count, 1u)) + parameters[parametersOffset + 1u]), count > 0u);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-score`,
        operation: OPERATION,
        variant: 'score',
        bindings: [
          {name: 'offsets', view: neighbors.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: neighbors.neighbors, type: 'u32', access: 'read'},
          {name: 'density', view: localReachabilityDensity, type: 'f32', access: 'read'},
          {name: 'lof', view: lof, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations,
        body: `var sum = 0.0;
  var count = 0u;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor < ROWS) {
      sum += density[densityOffset + neighbor];
      count++;
    }
  }
  let own = density[densityOffset + index];
  lof[lofOffset + index] =
    select(1.0, sum / f32(max(count, 1u)) / own, count > 0u && own > 0.0);`
      })
    ];
    if (output.outlier) {
      if (output.outlierCount) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-clear-count`,
            operation: OPERATION,
            variant: 'clear-count',
            bindings: [
              {name: 'outlierCount', view: output.outlierCount, type: 'u32', access: 'read_write'}
            ],
            invocationCount: 1,
            body: 'outlierCount[outlierCountOffset] = 0u;'
          })
        );
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-outlier`,
          operation: OPERATION,
          variant: output.outlierCount ? 'mask-count' : 'mask',
          bindings: [
            {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
            {name: 'lof', view: lof, type: 'f32', access: 'read'},
            {name: 'outlier', view: output.outlier, type: 'u32', access: 'read_write'},
            ...(output.outlierCount
              ? [
                  {
                    name: 'outlierCount',
                    view: output.outlierCount,
                    type: 'atomic<u32>' as const,
                    access: 'read_write' as const
                  }
                ]
              : [])
          ],
          invocationCount: rows,
          declarations: IS_NAN_BITS_WGSL,
          // A NaN threshold means none was given: nothing is flagged (checked on the bits, since
          // NaN comparisons may be folded away).
          body: `let threshold = parameters[parametersOffset];
  let hasThreshold = !isNanBits(threshold);
  let flagged = hasThreshold && lof[lofOffset + index] > threshold;
  outlier[outlierOffset + index] = select(0u, 1u, flagged);
  ${
    output.outlierCount
      ? 'if (flagged) {\n    atomicAdd(&outlierCount[outlierCountOffset], 1u);\n  }'
      : ''
  }`
        })
      );
    }
    return nodes;
  }
}
