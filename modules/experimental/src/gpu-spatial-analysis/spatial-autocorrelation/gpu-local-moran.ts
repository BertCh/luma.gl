// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  getFalseDiscoveryRateNodes,
  getSpatialAutocorrelationInputNodes,
  getSpatialAutocorrelationNeighborLoopWGSL,
  getSpatialAutocorrelationSharedWGSL,
  validateSpatialAutocorrelationInputs
} from './spatial-autocorrelation-kernels';
import {GPU_LOCAL_MORAN_QUADRANT} from './spatial-autocorrelation-parameters';

const OPERATION = 'GPULocalMoran';

/**
 * Properties for {@link GPULocalMoran}.
 *
 * Per-frame (no rebuild or recompile): the contents of `positions`, `values`, `mask` and
 * `parameters` (bounds, radius, significance level, weight transform, optional fixed moments).
 * Compile-time: the row count, `gridSize`, `falseDiscoveryRate`, and which optional views are
 * present.
 */
export type GPULocalMoranProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'local-moran'`. */
  id?: string;
  /** Packed planar points, one row per observation. At least one and fewer than 2^24 rows. */
  positions: GraphDataView<'float32x2'>;
  /** Packed analysis values, one per row. Rows with a non-finite value are excluded. */
  values: GraphDataView<'float32'>;
  /**
   * Per-frame parameters: packed float32 view of at least
   * `GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH` elements written with
   * `getGPUSpatialAutocorrelationParameterValues`. Invalid bounds or radius exclude every row.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Maximum `[columns, rows]` of the neighbor-search cell lattice. Compile-time. Results never
   * depend on it, only speed: each encoding uses cells at least `radius` wide.
   */
  gridSize: readonly [number, number];
  /** Optional row selection: nonzero includes the row. Excluded rows are neither foci nor neighbors. */
  mask?: GraphDataView<'uint32'>;
  /** Caller-owned analytic z-score of local Moran's I per row; quiet NaN when undefined. */
  zScores: GraphDataView<'float32'>;
  /** Optional caller-owned local Moran's I per row; quiet NaN for excluded rows, 0 for islands. */
  localI?: GraphDataView<'float32'>;
  /**
   * Optional caller-owned spatial lag per row: mean centered value `x_j - X` of the neighbors
   * (the row itself excluded), the y axis of a Moran scatterplot. 0 for islands, NaN if excluded.
   */
  spatialLag?: GraphDataView<'float32'>;
  /**
   * Optional caller-owned quadrant per row (`GPU_LOCAL_MORAN_QUADRANT`): 1 HH, 2 LH, 3 LL, 4 HL
   * when significant at the per-frame `significanceLevel`, otherwise 0.
   */
  quadrants?: GraphDataView<'uint32'>;
  /** Optional caller-owned two-sided normal p-value per row; quiet NaN where z is NaN. */
  pValues?: GraphDataView<'float32'>;
  /** Optional caller-owned neighbor count per row, the row itself excluded. */
  neighborCounts?: GraphDataView<'uint32'>;
  /** Optional caller-owned `[n, mean, variance, standardDeviation]` actually used this frame. */
  globalStatistics?: GraphDataView<'float32'>;
  /**
   * Compile-time. When true, quadrant significance uses Benjamini-Hochberg false discovery rate
   * correction at the per-frame `significanceLevel` over all rows with a finite z. Adds one
   * 31-bit sort. Defaults to false.
   */
  falseDiscoveryRate?: boolean;
};

/**
 * Local Moran's I (LISA) over planar points with a distance-band weight, with an analytic z-score
 * and HH/LL/HL/LH quadrants.
 *
 * Definition, which the GPU result matches within f32 rounding:
 * - Included rows, `n`, the mean `X` and centered values `c = x - X` are as in
 *   `GPUHotSpotAnalysis`; `M2 = sum c^2` over included rows.
 * - Neighbors exclude the row itself: `w_ij = 1` when `j != i` and `|p_i - p_j| <= radius`;
 *   `k_i` is their count and `L_i = sum_j w_ij c_j`.
 * - `I_i = (n - 1) * c_i * L_i' / M2` with `L_i' = L_i` (binary) or `L_i / k_i` (row
 *   standardized, the default), matching esda's `Moran_Local` scaling.
 * - z-score: exact mean and variance of `I_i` under conditional randomization (the null that
 *   esda's conditional permutation test samples): the other `n - 1` centered values are permuted
 *   over the other locations. With `N = n - 1`, `mu = -c_i / N` and
 *   `sigma^2 = (M2 - c_i^2) / N - mu^2`, `E[L_i] = k_i mu` and
 *   `Var[L_i] = sigma^2 k_i (N - k_i) / (N - 1)`, so `z_i = sign(c_i) (L_i - k_i mu) / sqrt(Var)`.
 *   The weight scaling and `M2` cancel. z is NaN when `n < 3`, `c_i = 0`, `k_i = 0`, `k_i = N` or
 *   the variance is not positive.
 * - Quadrant: sign of `c_i` and of the spatial lag, reported only when the two-sided normal p-value
 *   is at most the significance level (or passes BH-FDR at that level).
 *
 * Under this null `|z_i|` and the p-value do not depend on `|c_i|`, only the sign does: a row whose
 * value is near the mean can still be significant (as with esda's conditional permutations), and
 * for values within f32 rounding of the mean the sign of z and the quadrant may differ from a
 * double-precision evaluation while `|z|` and p agree. The normality approximation is weak for
 * small `k_i` or skewed values; permutation inference (pseudo p-values) is not implemented yet.
 *
 * Determinism and the viewport caveat are as in `GPUHotSpotAnalysis`: fixed-order sums only, and
 * moments follow the included rows unless `fixedMoments` pins them.
 */
export class GPULocalMoran implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPULocalMoranProps;

  constructor(props: GPULocalMoranProps) {
    this.id = props.id ?? 'local-moran';
    this.props = props;
    const id = this.id;
    const rows = validateSpatialAutocorrelationInputs({...props, id});
    for (const [name, view] of [
      ['zScores', props.zScores],
      ['localI', props.localI],
      ['spatialLag', props.spatialLag],
      ['pValues', props.pValues]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
      }
    }
    for (const [name, view] of [
      ['quadrants', props.quadrants],
      ['neighborCounts', props.neighborCounts]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
      }
    }
    for (const [name, view] of [
      ['zScores', props.zScores],
      ['localI', props.localI],
      ['spatialLag', props.spatialLag],
      ['quadrants', props.quadrants],
      ['pValues', props.pValues],
      ['neighborCounts', props.neighborCounts]
    ] as const) {
      if (view && view.length !== rows) {
        throw new Error(`${id} ${name} length must equal positions length`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        props.zScores,
        props.localI,
        props.spatialLag,
        props.quadrants,
        props.pValues,
        props.neighborCounts,
        props.globalStatistics
      ],
      [props.positions, props.values, props.parameters, props.mask]
    );
  }

  /** Returns the local Moran nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {positions, parameters, gridSize, zScores} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      props.values,
      parameters,
      props.mask,
      zScores,
      props.localI,
      props.spatialLag,
      props.quadrants,
      props.pValues,
      props.neighborCounts,
      props.globalStatistics
    ]);
    const rows = positions.length;
    const inputs = getSpatialAutocorrelationInputNodes<Parameters>(graph, {
      ...props,
      id,
      operation: OPERATION
    });
    const nodes = inputs.nodes;
    const sharedWGSL = `${getSpatialAutocorrelationSharedWGSL(gridSize)}
const MOMENTS: u32 = ${rows}u;`;
    const spatialLag =
      props.spatialLag ?? createTransientView(graph, `${id}-spatial-lag`, 'float32', rows);
    const neighborCounts =
      props.neighborCounts ?? createTransientView(graph, `${id}-neighbor-counts`, 'uint32', rows);

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-neighbors`,
        operation: OPERATION,
        variant: 'local-moran',
        bindings: [
          {name: 'positions', view: positions, type: 'f32', access: 'read'},
          {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
          {name: 'sortedRows', view: inputs.sortedRows, type: 'u32', access: 'read'},
          {name: 'cellOffsets', view: inputs.cellOffsets, type: 'u32', access: 'read'},
          {name: 'statistics', view: inputs.statistics, type: 'f32', access: 'read'},
          {name: 'zScores', view: zScores, type: 'f32', access: 'read_write'},
          {name: 'spatialLag', view: spatialLag, type: 'f32', access: 'read_write'},
          {name: 'neighborCounts', view: neighborCounts, type: 'u32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: sharedWGSL,
        body: `let centered = statistics[statisticsOffset + index];
  var zScore = getQuietNaN(index);
  var lag = getQuietNaN(index);
  var neighborCount = 0u;
  if (isFiniteFloat(centered)) {
    let lattice = readLattice();
    let x = positions[positionsOffset + index * 2u];
    let y = positions[positionsOffset + index * 2u + 1u];
    var neighborSum = 0.0;
    ${getSpatialAutocorrelationNeighborLoopWGSL(`if (neighbor != index) {
          neighborSum += statistics[statisticsOffset + neighbor];
          neighborCount++;
        }`)}
    let weightSum = f32(neighborCount);
    lag = select(0.0, neighborSum / weightSum, neighborCount > 0u);
    let others = statistics[statisticsOffset + MOMENTS] - 1.0;
    let sumOfSquares = statistics[statisticsOffset + MOMENTS + 3u];
    let otherMean = -centered / others;
    let otherVariance = (sumOfSquares - centered * centered) / others - otherMean * otherMean;
    let lagVariance = otherVariance * weightSum * (others - weightSum) / (others - 1.0);
    if (others >= 2.0 && centered != 0.0 && neighborCount > 0u && weightSum < others &&
        lagVariance > 0.0 && isFiniteFloat(lagVariance)) {
      let lagZ = (neighborSum - weightSum * otherMean) / sqrt(lagVariance);
      zScore = select(lagZ, -lagZ, centered < 0.0);
    }
  }
  zScores[zScoresOffset + index] = zScore;
  spatialLag[spatialLagOffset + index] = lag;
  neighborCounts[neighborCountsOffset + index] = neighborCount;`
      })
    );

    if (props.localI) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-local-i`,
          operation: OPERATION,
          variant: 'local-i',
          bindings: [
            {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
            {name: 'statistics', view: inputs.statistics, type: 'f32', access: 'read'},
            {name: 'spatialLag', view: spatialLag, type: 'f32', access: 'read'},
            {name: 'neighborCounts', view: neighborCounts, type: 'u32', access: 'read'},
            {name: 'localI', view: props.localI, type: 'f32', access: 'read_write'}
          ],
          invocationCount: rows,
          declarations: sharedWGSL,
          body: `let centered = statistics[statisticsOffset + index];
  let lag = spatialLag[spatialLagOffset + index];
  let rowStandardized = readParameter(6u) != 0.0;
  let weightedLag = select(lag * f32(neighborCounts[neighborCountsOffset + index]), lag, rowStandardized);
  let scale = (statistics[statisticsOffset + MOMENTS] - 1.0) / statistics[statisticsOffset + MOMENTS + 3u];
  // Excluded rows keep their NaN centered value; islands have a zero lag and so I = 0.
  localI[localIOffset + index] = select(centered * weightedLag * scale, getQuietNaN(index), !isFiniteFloat(centered));`
        })
      );
    }

    if (!props.quadrants && !props.pValues) {
      return nodes;
    }
    const falseDiscoveryRate =
      props.falseDiscoveryRate && props.quadrants
        ? getFalseDiscoveryRateNodes<Parameters>(graph, {
            id: `${id}-fdr`,
            operation: OPERATION,
            zScores,
            parameters,
            gridSize,
            levelExpressions: ['readParameter(5u)']
          })
        : undefined;
    if (falseDiscoveryRate) {
      nodes.push(...falseDiscoveryRate.nodes);
    }
    const classifyBindings: WGSLKernelBinding[] = [
      {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
      {name: 'statistics', view: inputs.statistics, type: 'f32', access: 'read'},
      {name: 'zScores', view: zScores, type: 'f32', access: 'read'},
      {name: 'spatialLag', view: spatialLag, type: 'f32', access: 'read'}
    ];
    if (falseDiscoveryRate) {
      classifyBindings.push(
        {name: 'ranks', view: falseDiscoveryRate.ranks, type: 'u32', access: 'read'},
        {name: 'counters', view: falseDiscoveryRate.counters, type: 'u32', access: 'read'}
      );
    }
    if (props.quadrants) {
      classifyBindings.push({
        name: 'quadrants',
        view: props.quadrants,
        type: 'u32',
        access: 'read_write'
      });
    }
    if (props.pValues) {
      classifyBindings.push({
        name: 'pValues',
        view: props.pValues,
        type: 'f32',
        access: 'read_write'
      });
    }
    const {HIGH_HIGH, LOW_HIGH, LOW_LOW, HIGH_LOW, NOT_SIGNIFICANT} = GPU_LOCAL_MORAN_QUADRANT;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-classify`,
        operation: OPERATION,
        variant: falseDiscoveryRate ? 'classify-fdr' : 'classify',
        bindings: classifyBindings,
        invocationCount: rows,
        declarations: sharedWGSL,
        body: `let zScore = zScores[zScoresOffset + index];
  let finite = isFiniteFloat(zScore);
  let pValue = select(getQuietNaN(index), getTwoSidedPValue(zScore), finite);
  ${
    falseDiscoveryRate
      ? `let rank = ranks[ranksOffset + index];
  let significant = finite && rank > 0u && rank <= counters[countersOffset + 1u];`
      : 'let significant = finite && pValue <= readParameter(5u);'
  }
  let centered = statistics[statisticsOffset + index];
  let lag = spatialLag[spatialLagOffset + index];
  var quadrant = ${NOT_SIGNIFICANT}u;
  if (significant) {
    if (centered > 0.0 && lag > 0.0) {
      quadrant = ${HIGH_HIGH}u;
    } else if (centered < 0.0 && lag > 0.0) {
      quadrant = ${LOW_HIGH}u;
    } else if (centered < 0.0 && lag < 0.0) {
      quadrant = ${LOW_LOW}u;
    } else if (centered > 0.0 && lag < 0.0) {
      quadrant = ${HIGH_LOW}u;
    }
  }
  ${props.quadrants ? 'quadrants[quadrantsOffset + index] = quadrant;' : ''}
  ${props.pValues ? 'pValues[pValuesOffset + index] = pValue;' : ''}`
      })
    );
    return nodes;
  }
}
