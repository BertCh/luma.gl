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
  getSpatialAutocorrelationSharedWGSL,
  getSpatialWeightsNeighborLoopWGSL,
  SPATIAL_AUTOCORRELATION_FLOAT_WGSL,
  validateSpatialAutocorrelationInputs
} from './spatial-autocorrelation-kernels';
import type {GPUSpatialWeights} from '../spatial-weights/index';
import {GPU_LOCAL_MORAN_QUADRANT} from './spatial-autocorrelation-parameters';

const OPERATION = 'GPULocalMoran';

/**
 * Properties for {@link GPULocalMoran}.
 *
 * Per-frame (no rebuild or recompile): the contents of `weights`, `values`, `mask` and
 * `parameters` (significance level, optional fixed moments). Compile-time: the row count,
 * `falseDiscoveryRate`, and which optional views are present.
 */
export type GPULocalMoranProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'local-moran'`. */
  id?: string;
  /**
   * Square self-join spatial weights, one row per observation (for example from
   * `GPUNeighborSearch`, optionally transformed). Used as given: the caller applies any
   * transform. Row-standardized weights reproduce esda's default `Moran_Local(transform='r')`;
   * binary weights give the `'b'` variant. Fewer than 2^24 rows. The focal row is never its own
   * neighbor.
   */
  weights: GPUSpatialWeights;
  /** Packed analysis values, one per row. Rows with a non-finite value are excluded. */
  values: GraphDataView<'float32'>;
  /**
   * Per-frame parameters: packed float32 view of at least
   * `GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH` elements written with
   * `getGPUSpatialAutocorrelationParameterValues`.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Optional row selection: nonzero includes the row. Excluded rows are neither foci nor
   * neighbors (their weight is dropped from every row sum).
   */
  mask?: GraphDataView<'uint32'>;
  /** Caller-owned analytic z-score of local Moran's I per row; quiet NaN when undefined. */
  zScores: GraphDataView<'float32'>;
  /** Optional caller-owned local Moran's I per row; quiet NaN for excluded rows, 0 for islands. */
  localI?: GraphDataView<'float32'>;
  /**
   * Optional caller-owned spatial lag per row: `sum_j w_ij (x_j - X)` over included neighbors
   * with the weights as given (the weighted mean of the neighbors when the weights are
   * row-standardized), the y axis of a Moran scatterplot. 0 for islands, NaN if excluded.
   */
  spatialLag?: GraphDataView<'float32'>;
  /**
   * Optional caller-owned quadrant per row (`GPU_LOCAL_MORAN_QUADRANT`): 1 HH, 2 LH, 3 LL, 4 HL
   * when significant at the per-frame `significanceLevel`, otherwise 0.
   */
  quadrants?: GraphDataView<'uint32'>;
  /** Optional caller-owned two-sided normal p-value per row; quiet NaN where z is NaN. */
  pValues?: GraphDataView<'float32'>;
  /** Optional caller-owned count of included neighbors per row (weight entries pointing at included rows), the row itself excluded. */
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
 * Local Moran's I (LISA) on caller-supplied spatial weights, with an analytic z-score and
 * HH/LL/HL/LH quadrants.
 *
 * Definition, which the GPU result matches within f32 rounding:
 * - Included rows are mask-selected rows with a finite value. `n`, the mean `X` and centered
 *   values `c = x - X` are as in `GPUHotSpotAnalysis`; `M2 = sum c^2` over included rows.
 * - Weights `w_ij` are the CSR entries of row `i` pointing at included rows other than `i`, used
 *   as given. `k_i` is their count, `W_i = sum_j w_ij`, `S1_i = sum_j w_ij^2` and
 *   `L_i = sum_j w_ij c_j` (the spatial lag).
 * - `I_i = (n - 1) * c_i * L_i / M2`, matching esda's `Moran_Local` scaling for the weights given
 *   (row-standardized weights give esda's default, binary weights its `'b'` transform).
 * - z-score: exact mean and variance of `I_i` under conditional randomization (the null that
 *   esda's conditional permutation test samples): the other `n - 1` centered values are permuted
 *   over the other locations. With `N = n - 1`, `mu = -c_i / N` and
 *   `sigma^2 = (M2 - c_i^2) / N - mu^2`, `E[L_i] = W_i mu` and
 *   `Var[L_i] = sigma^2 (N S1_i - W_i^2) / (N - 1)` (which is `sigma^2 k_i (N - k_i) / (N - 1)`
 *   for binary weights), so `z_i = sign(c_i) (L_i - W_i mu) / sqrt(Var)`. The weight scaling and
 *   `M2` cancel, so z is invariant to rescaling a row of weights. z is NaN when `n < 3`,
 *   `c_i = 0`, `k_i = 0` or the variance is not positive.
 * - Quadrant: sign of `c_i` and of the spatial lag, reported only when the two-sided normal p-value
 *   is at most the significance level (or passes BH-FDR at that level).
 *
 * Under this null `|z_i|` and the p-value do not depend on `|c_i|`, only the sign does: a row whose
 * value is near the mean can still be significant (as with esda's conditional permutations), and
 * for values within f32 rounding of the mean the sign of z and the quadrant may differ from a
 * double-precision evaluation while `|z|` and p agree. The normality approximation is weak for
 * small `k_i` or skewed values; permutation inference lives in `GPULocalPermutationTest`.
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
        throw new Error(`${id} ${name} length must equal the weights row count`);
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
      [
        props.weights.offsets,
        props.weights.neighbors,
        props.weights.weights,
        props.values,
        props.parameters,
        props.mask
      ]
    );
  }

  /** Returns the local Moran nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {weights, parameters, zScores} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      weights.offsets,
      weights.neighbors,
      weights.weights,
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
    const rows = props.values.length;
    const inputs = getSpatialAutocorrelationInputNodes<Parameters>(graph, {
      ...props,
      id,
      operation: OPERATION
    });
    const nodes = inputs.nodes;
    const sharedWGSL = `${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}
const MOMENTS: u32 = ${rows}u;
const ROWS: u32 = ${rows}u;`;
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
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
          {name: 'weights', view: weights.weights, type: 'f32', access: 'read'},
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
    var neighborSum = 0.0;
    var weightSum = 0.0;
    var squareSum = 0.0;
    ${getSpatialWeightsNeighborLoopWGSL(`neighborSum += weight * statistics[statisticsOffset + neighbor];
      weightSum += weight;
      squareSum += weight * weight;
      neighborCount++;`)}
    lag = neighborSum;
    let others = statistics[statisticsOffset + MOMENTS] - 1.0;
    let sumOfSquares = statistics[statisticsOffset + MOMENTS + 3u];
    let otherMean = -centered / others;
    let otherVariance = (sumOfSquares - centered * centered) / others - otherMean * otherMean;
    let lagVariance = otherVariance * (others * squareSum - weightSum * weightSum) / (others - 1.0);
    if (others >= 2.0 && centered != 0.0 && neighborCount > 0u &&
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
            {name: 'statistics', view: inputs.statistics, type: 'f32', access: 'read'},
            {name: 'spatialLag', view: spatialLag, type: 'f32', access: 'read'},
            {name: 'localI', view: props.localI, type: 'f32', access: 'read_write'}
          ],
          invocationCount: rows,
          declarations: sharedWGSL,
          body: `let centered = statistics[statisticsOffset + index];
  let lag = spatialLag[spatialLagOffset + index];
  let scale = (statistics[statisticsOffset + MOMENTS] - 1.0) / statistics[statisticsOffset + MOMENTS + 3u];
  // Excluded rows keep their NaN centered value; islands have a zero lag and so I = 0.
  localI[localIOffset + index] = select(centered * lag * scale, getQuietNaN(index), !isFiniteFloat(centered));`
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
            levelExpressions: ['readParameter(0u)']
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
        declarations: getSpatialAutocorrelationSharedWGSL(),
        body: `let zScore = zScores[zScoresOffset + index];
  let finite = isFiniteFloat(zScore);
  let pValue = select(getQuietNaN(index), getTwoSidedPValue(zScore), finite);
  ${
    falseDiscoveryRate
      ? `let rank = ranks[ranksOffset + index];
  let significant = finite && rank > 0u && rank <= counters[countersOffset + 1u];`
      : 'let significant = finite && pValue <= readParameter(0u);'
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
