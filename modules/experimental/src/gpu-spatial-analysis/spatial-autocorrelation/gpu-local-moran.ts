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
import type {
  GPUInferenceMetadata,
  GPUMultipleTestingMethod
} from '../../analysis-core/contracts/index';
import {
  GPUMultipleTestingAdjustment,
  resolveGPUMultipleTestingMethod
} from '../multiple-testing/index';
import {
  getSpatialAutocorrelationConstantsWGSL,
  getSpatialAutocorrelationInputNodes,
  getSpatialAutocorrelationSharedWGSL,
  getSpatialWeightsNeighborLoopWGSL,
  resolveSpatialAutocorrelationMinimumNeighbors,
  SPATIAL_AUTOCORRELATION_FLOAT_WGSL,
  validateSpatialAutocorrelationInputs
} from './spatial-autocorrelation-kernels';
import {
  isGPUSpatialWeightsRowStandardized,
  resolveGPUIslandPolicy,
  type GPUIslandPolicy,
  type GPUSpatialWeights
} from '../spatial-weights/index';
import type {GPUFloat32OrUint32Values} from '../spatial-weights/float32-values';
import {
  getGPUSpatialAutocorrelationParameterValues,
  GPU_LOCAL_MORAN_QUADRANT,
  type GPUSpatialAutocorrelationSettings
} from './spatial-autocorrelation-parameters';

const OPERATION = 'GPULocalMoran';

/**
 * How `GPULocalMoran` gates the `quadrants` output: `'analytic'` reports a quadrant only where
 * the analytic p-value (adjusted when `multipleTesting` is set) is significant, `'none'` reports
 * it for every included row.
 */
export type GPULocalMoranQuadrantGating = 'analytic' | 'none';

/** Caller-owned outputs of {@link GPULocalMoran}. */
export type GPULocalMoranOutput = {
  /** Caller-owned analytic z-score of local Moran's I per row; quiet NaN when undefined. */
  zScores: GraphDataView<'float32'>;
  /**
   * Optional caller-owned local Moran's I per row; quiet NaN for excluded rows (and for islands
   * under `'exclude'`), 0 for islands under `'keep'`, NaN for every row when the values are
   * constant (zero sum of squares).
   */
  localI?: GraphDataView<'float32'>;
  /**
   * Optional caller-owned spatial lag per row: `sum_j w_ij (x_j - X)` over active neighbors
   * with the weights as given (re-standardized when the weights are row-standardized, so the
   * weighted mean of the neighbors), the y axis of a Moran scatterplot. 0 for islands that
   * are kept, NaN if excluded.
   */
  spatialLag?: GraphDataView<'float32'>;
  /**
   * Optional caller-owned quadrant per row (`GPU_LOCAL_MORAN_QUADRANT`): 1 HH, 2 LH, 3 LL, 4 HL
   * when the (adjusted) p-value is at most the per-frame `significanceLevel`, otherwise 0. The
   * quadrant is the sign of the centered value and of the lag, whatever the correction method.
   */
  quadrants?: GraphDataView<'uint32'>;
  /** Optional caller-owned unadjusted two-sided normal p-value per row; quiet NaN where z is NaN. */
  pValues?: GraphDataView<'float32'>;
  /**
   * Optional caller-owned p-value adjusted by `multipleTesting` (equal to `pValues` for `'none'`);
   * quiet NaN where z is NaN. The test family is every row with a defined z-score, never a viewport.
   */
  adjustedPValues?: GraphDataView<'float32'>;
  /** Optional caller-owned count of active neighbors per row (the row itself excluded); 0 for rows outside the analysis. */
  neighborCounts?: GraphDataView<'uint32'>;
  /** Optional caller-owned `[n, mean, variance, standardDeviation, islandCount]` actually used this frame. */
  globalStatistics?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPULocalMoran}.
 *
 * Per-frame (no rebuild or recompile): the contents of `weights`, `values`, `mask` and
 * `parameters` (significance level, optional fixed moments). Compile-time: the row count,
 * `islands`, `minimumNeighbors`, `multipleTesting`, and which optional views are present.
 */
export type GPULocalMoranProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'local-moran'`. */
  id?: string;
  /**
   * Square self-join spatial weights, one row per observation (for example from
   * `GPUNeighborSearch`, optionally transformed). Used as given, except that when
   * `weights.information.transform` is `'row'` each row's weights are re-standardized over the neighbors
   * that remain after masking, so a row still sums to 1 instead of shrinking toward 0. Row-standardized
   * weights reproduce esda's default `Moran_Local(transform='r')`; binary weights give the `'b'`
   * variant. Fewer than 2^24 rows. A neighbor is a slot with a positive weight that points at
   * another included row. If `weights.status.overflow` is nonzero every output is NaN.
   */
  weights: GPUSpatialWeights;
  /**
   * Packed analysis values, one per row, `float32` or `uint32` (for example counts; converted to
   * f32 by one node, exact below 2^24). Rows with a non-finite value are excluded.
   */
  values: GPUFloat32OrUint32Values;
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
  /**
   * Compile-time. Treatment of islands (included rows without a positive-weight neighbor among the
   * included rows). `'exclude'` (default) removes them from `n`, the mean and the variance and
   * writes NaN for every output; `'keep'` keeps them in the moments as esda does, with local I
   * and the spatial lag 0 and NaN z-score and p-value. The count is reported in
   * `globalStatistics[4]`. An island is decided in one pass, with no cascade: a row whose only
   * positive neighbors were islands has no neighbor among the analysis rows, gets NaN, and stays in `n`.
   */
  islands?: GPUIslandPolicy;
  /**
   * Compile-time. Rows with fewer active neighbors (the row itself excluded) get a NaN z-score
   * and p-value and quadrant 0. Defaults to 1, because the analytic normal approximation is poor
   * for few neighbors; a value around 4 is a common choice for local Moran. Must be an integer
   * of at least 1.
   */
  minimumNeighbors?: number;
  /** Caller-owned outputs. */
  output: GPULocalMoranOutput;
  /**
   * Compile-time. `'analytic'` (default) writes `quadrants` only where the analytic test is
   * significant. `'none'` writes the ungated quadrant (the signs of the centered value and the
   * spatial lag) for every included row with a nonzero centered value and nonzero lag, so a
   * caller can gate by another test such as `GPULocalPermutationTest`. Rows that are excluded,
   * islands, or exactly at the mean get 0. `pValues` are unaffected. Cannot be combined with
   * `multipleTesting`.
   */
  quadrantGating?: GPULocalMoranQuadrantGating;
  /**
   * Compile-time. Multiple-testing correction of the two-sided p-values (`GPUMultipleTestingAdjustment`):
   * `'none'` (default), `'bonferroni'`, `'holm'`, `'benjamini-hochberg'` or `'benjamini-yekutieli'`
   * (valid under any dependence, which local statistics have). Quadrants are significant when the
   * adjusted p-value is at most the per-frame `significanceLevel`. The family is every row with a
   * defined z-score.
   */
  multipleTesting?: GPUMultipleTestingMethod;
};

/**
 * Local Moran's I (LISA) on caller-supplied spatial weights, with an analytic z-score and
 * HH/LL/HL/LH quadrants.
 *
 * Definition, which the GPU result matches within f32 rounding:
 * - Included rows are mask-selected rows with a finite value. Islands (included rows without a
 *   positive-weight neighbor among the included rows) are removed by default (`islands:
 *   'exclude'`). `n`, the mean `X` and centered values `c = x - X` are as in `GPUHotSpotAnalysis`;
 *   `M2 = sum c^2` over the analysis rows.
 * - Weights `w_ij` are the CSR entries of row `i` with a positive weight pointing at analysis
 *   rows other than `i`. They are used as given, or divided by their sum when the weights are
 *   row-standardized (`weights.information.transform === 'row'`), so masked neighbors cannot shrink the
 *   lag. `k_i` is their count, `W_i = sum_j w_ij`, `S1_i = sum_j w_ij^2` and `L_i = sum_j w_ij c_j`
 *   (the spatial lag).
 * - `I_i = (n - 1) * c_i * L_i / M2`, matching esda's `Moran_Local` scaling for the weights given
 *   (row-standardized weights give esda's default, binary weights its `'b'` transform).
 * - z-score: exact mean and variance of `I_i` under conditional randomization (the null that
 *   esda's conditional permutation test samples): the other `n - 1` centered values are permuted
 *   over the other locations. With `N = n - 1`, `mu = -c_i / N` and
 *   `sigma^2 = (M2 - c_i^2) / N - mu^2`, `E[L_i] = W_i mu` and
 *   `Var[L_i] = sigma^2 (N S1_i - W_i^2) / (N - 1)` (which is `sigma^2 k_i (N - k_i) / (N - 1)`
 *   for binary weights), so `z_i = sign(c_i) (L_i - W_i mu) / sqrt(Var)`. The weight scaling and
 *   `M2` cancel, so z is invariant to rescaling a row of weights. z is NaN when `n < 3`,
 *   `c_i = 0`, `k_i < minimumNeighbors` or the variance is not positive.
 * - Quadrant: sign of `c_i` and of the spatial lag, reported only when the (adjusted) two-sided
 *   normal p-value is at most the significance level. Legend contract: the quadrant meaning never
 *   depends on the correction method; only which rows are significant does.
 *
 * Under this null `|z_i|` and the p-value do not depend on `|c_i|`, only the sign does: a row whose
 * value is near the mean can still be significant (as with esda's conditional permutations), and
 * for values within f32 rounding of the mean the sign of z and the quadrant may differ from a
 * double-precision evaluation while `|z|` and p agree. The normality approximation is weak for
 * small `k_i` or skewed values; permutation inference lives in `GPULocalPermutationTest`.
 *
 * Precision: the mean gets a second-pass correction and the variance uses centered deviations.
 *
 * Determinism and the viewport caveat are as in `GPUHotSpotAnalysis`: fixed-order sums only, and
 * moments follow the included rows unless `fixedMoments` pins them.
 */
export class GPULocalMoran implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPULocalMoranProps;
  /** Resolved correction method. */
  readonly multipleTesting: GPUMultipleTestingMethod;
  /** Resolved island policy. */
  readonly islands: GPUIslandPolicy;
  /** Resolved minimum number of active neighbors. */
  readonly minimumNeighbors: number;

  constructor(props: GPULocalMoranProps) {
    this.id = props.id ?? 'local-moran';
    this.props = props;
    const id = this.id;
    this.islands = resolveGPUIslandPolicy(id, props.islands);
    this.minimumNeighbors = resolveSpatialAutocorrelationMinimumNeighbors(
      id,
      props.minimumNeighbors
    );
    this.multipleTesting = resolveGPUMultipleTestingMethod(id, props.multipleTesting);
    const rows = validateSpatialAutocorrelationInputs({
      ...props,
      id,
      globalStatistics: props.output.globalStatistics
    });
    for (const [name, view] of [
      ['zScores', props.output.zScores],
      ['localI', props.output.localI],
      ['spatialLag', props.output.spatialLag],
      ['pValues', props.output.pValues],
      ['adjustedPValues', props.output.adjustedPValues]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
      }
    }
    for (const [name, view] of [
      ['quadrants', props.output.quadrants],
      ['neighborCounts', props.output.neighborCounts]
    ] as const) {
      if (view) {
        validatePackedUint32View(view, `${id} ${name}`);
      }
    }
    for (const [name, view] of [
      ['zScores', props.output.zScores],
      ['localI', props.output.localI],
      ['spatialLag', props.output.spatialLag],
      ['quadrants', props.output.quadrants],
      ['pValues', props.output.pValues],
      ['adjustedPValues', props.output.adjustedPValues],
      ['neighborCounts', props.output.neighborCounts]
    ] as const) {
      if (view && view.length !== rows) {
        throw new Error(`${id} ${name} length must equal the weights row count`);
      }
    }
    if (props.quadrantGating === 'none' && this.multipleTesting !== 'none') {
      throw new Error(`${id} quadrantGating 'none' cannot be combined with multipleTesting`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        props.output.zScores,
        props.output.localI,
        props.output.spatialLag,
        props.output.quadrants,
        props.output.pValues,
        props.output.adjustedPValues,
        props.output.neighborCounts,
        props.output.globalStatistics
      ],
      [
        props.weights.offsets,
        props.weights.neighbors,
        props.weights.weights,
        props.weights.status?.overflow,
        props.values,
        props.parameters,
        props.mask
      ]
    );
  }

  /**
   * Describes how the p-values of a run with `settings` were produced, for building a legend:
   * two-sided analytic normal p-values (conditional randomization moments), the correction method,
   * the significance level of the quadrants and the tested family.
   *
   * @param settings The settings packed into `parameters` with
   * `getGPUSpatialAutocorrelationParameterValues`; validated the same way.
   */
  getInferenceMetadata(settings: GPUSpatialAutocorrelationSettings = {}): GPUInferenceMetadata {
    // Packing performs the canonical validation of the settings.
    getGPUSpatialAutocorrelationParameterValues(settings);
    const significanceLevel = settings.significanceLevel ?? 0.05;
    return {
      alternative: 'two-sided',
      permutationCount: 0,
      multipleTesting: this.multipleTesting,
      significanceLevel,
      testedRowsDefinition: `included rows with at least ${this.minimumNeighbors} active neighbor${
        this.minimumNeighbors === 1 ? '' : 's'
      }${this.islands === 'exclude' ? ' (islands excluded)' : ''}, a nonzero centered value and a defined z-score`
    };
  }

  /** Returns the local Moran nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {weights, parameters} = props;
    const {zScores} = props.output;
    validateGraphViewsBelongToGraph(id, graph, [
      weights.offsets,
      weights.neighbors,
      weights.weights,
      weights.status?.overflow,
      props.values,
      parameters,
      props.mask,
      zScores,
      props.output.localI,
      props.output.spatialLag,
      props.output.quadrants,
      props.output.pValues,
      props.output.adjustedPValues,
      props.output.neighborCounts,
      props.output.globalStatistics
    ]);
    const rows = props.values.length;
    const inputs = getSpatialAutocorrelationInputNodes<Parameters>(graph, {
      ...props,
      globalStatistics: props.output.globalStatistics,
      id,
      islands: this.islands,
      operation: OPERATION
    });
    const nodes = inputs.nodes;
    const constantsWGSL = `${getSpatialAutocorrelationConstantsWGSL(rows, weights.neighbors.length)}
const MINIMUM_NEIGHBORS: u32 = ${this.minimumNeighbors}u;
const STANDARDIZE: bool = ${isGPUSpatialWeightsRowStandardized(weights)};`;
    const floatWGSL = `${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}
${constantsWGSL}`;
    // `readParameter` needs the `parameters` binding, so only kernels that bind it include it.
    const parameterWGSL = `${getSpatialAutocorrelationSharedWGSL()}
${constantsWGSL}`;
    const {HIGH_HIGH, LOW_HIGH, LOW_LOW, HIGH_LOW, NOT_SIGNIFICANT} = GPU_LOCAL_MORAN_QUADRANT;
    const hasCorrection = this.multipleTesting !== 'none';
    // Quadrant gating 'none' is rejected with a correction, so a correction implies analytic gating.
    const wantsClassify = Boolean(
      props.output.quadrants || props.output.pValues || props.output.adjustedPValues
    );
    const isAdjusting =
      hasCorrection &&
      Boolean(props.output.adjustedPValues || props.output.quadrants) &&
      wantsClassify;
    // With a correction the neighbor kernel writes the unadjusted p-values and a later kernel
    // classifies from the adjusted ones.
    const unadjustedPValues = isAdjusting
      ? (props.output.pValues ?? createTransientView(graph, `${id}-p-values`, 'float32', rows))
      : undefined;
    const adjustedPValues = isAdjusting
      ? (props.output.adjustedPValues ??
        createTransientView(graph, `${id}-adjusted-p-values`, 'float32', rows))
      : undefined;
    const needsClassifyKernel = (fuseClassify: boolean) =>
      wantsClassify && !fuseClassify && (!isAdjusting || Boolean(props.output.quadrants));

    // The per-row epilogues (local I, quadrant and p-value) are pure functions of values the
    // neighbor kernel already holds, so they are fused into it when its bindings still fit the
    // device limit: this removes one read-modify pass over `statistics`, `spatialLag` and
    // `zScores` per epilogue. Without fusion (a correction needs the adjusted p-values first),
    // they run as separate passes over the stored lag.
    const storageBufferLimit = graph.device.limits.maxStorageBuffersPerShaderStage;
    const getBindingCount = (fuseLocalI: boolean, fuseClassify: boolean) => {
      const storesLag =
        Boolean(props.output.spatialLag) ||
        (Boolean(props.output.localI) && !fuseLocalI) ||
        needsClassifyKernel(fuseClassify);
      return (
        5 +
        (storesLag ? 1 : 0) +
        (props.output.neighborCounts ? 1 : 0) +
        (props.output.localI && fuseLocalI ? 1 : 0) +
        (isAdjusting ? 1 : 0) +
        (fuseClassify
          ? 1 +
            (props.output.quadrants ? 1 : 0) +
            (props.output.pValues ? 1 : 0) +
            (props.output.adjustedPValues ? 1 : 0)
          : 0)
      );
    };
    const canFuseClassify = wantsClassify && !isAdjusting;
    let fuseLocalI = Boolean(props.output.localI);
    let fuseClassify = canFuseClassify;
    if (getBindingCount(fuseLocalI, fuseClassify) > storageBufferLimit) {
      fuseLocalI = false;
    }
    if (getBindingCount(fuseLocalI, fuseClassify) > storageBufferLimit) {
      fuseClassify = false;
      fuseLocalI =
        Boolean(props.output.localI) && getBindingCount(true, false) <= storageBufferLimit;
    }
    const storesLag =
      Boolean(props.output.spatialLag) ||
      (Boolean(props.output.localI) && !fuseLocalI) ||
      needsClassifyKernel(fuseClassify);
    const spatialLag = storesLag
      ? (props.output.spatialLag ??
        createTransientView(graph, `${id}-spatial-lag`, 'float32', rows))
      : undefined;

    const quadrantWGSL = (centeredName: string, lagName: string, significant: string) => `
  var quadrant = ${NOT_SIGNIFICANT}u;
  if (${significant}) {
    if (${centeredName} > 0.0 && ${lagName} > 0.0) {
      quadrant = ${HIGH_HIGH}u;
    } else if (${centeredName} < 0.0 && ${lagName} > 0.0) {
      quadrant = ${LOW_HIGH}u;
    } else if (${centeredName} < 0.0 && ${lagName} < 0.0) {
      quadrant = ${LOW_LOW}u;
    } else if (${centeredName} > 0.0 && ${lagName} < 0.0) {
      quadrant = ${HIGH_LOW}u;
    }
  }
  quadrants[quadrantsOffset + index] = quadrant;`;
    const classifyWGSL = (
      zScoreName: string,
      centeredName: string,
      lagName: string
    ) => `let pValue = select(getQuietNaN(index), getTwoSidedPValue(${zScoreName}), isFiniteFloat(${zScoreName}));
  ${
    props.output.quadrants
      ? quadrantWGSL(
          centeredName,
          lagName,
          props.quadrantGating === 'none'
            ? `isFiniteFloat(${centeredName})`
            : `isFiniteFloat(${zScoreName}) && pValue <= readParameter(0u)`
        )
      : ''
  }
  ${props.output.pValues ? 'pValues[pValuesOffset + index] = pValue;' : ''}
  ${props.output.adjustedPValues ? 'adjustedPValues[adjustedPValuesOffset + index] = pValue;' : ''}`;

    const neighborBindings: WGSLKernelBinding[] = [
      {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
      {name: 'weights', view: weights.weights, type: 'f32', access: 'read'},
      {name: 'statistics', view: inputs.statistics, type: 'f32', access: 'read'},
      {name: 'zScores', view: zScores, type: 'f32', access: 'read_write'}
    ];
    if (spatialLag) {
      neighborBindings.push({
        name: 'spatialLag',
        view: spatialLag,
        type: 'f32',
        access: 'read_write'
      });
    }
    if (props.output.neighborCounts) {
      neighborBindings.push({
        name: 'neighborCounts',
        view: props.output.neighborCounts,
        type: 'u32',
        access: 'read_write'
      });
    }
    if (props.output.localI && fuseLocalI) {
      neighborBindings.push({
        name: 'localI',
        view: props.output.localI,
        type: 'f32',
        access: 'read_write'
      });
    }
    if (unadjustedPValues) {
      neighborBindings.push({
        name: 'pValues',
        view: unadjustedPValues,
        type: 'f32',
        access: 'read_write'
      });
    }
    if (fuseClassify) {
      neighborBindings.push({name: 'parameters', view: parameters, type: 'f32', access: 'read'});
      if (props.output.quadrants) {
        neighborBindings.push({
          name: 'quadrants',
          view: props.output.quadrants,
          type: 'u32',
          access: 'read_write'
        });
      }
      if (props.output.pValues) {
        neighborBindings.push({
          name: 'pValues',
          view: props.output.pValues,
          type: 'f32',
          access: 'read_write'
        });
      }
      if (props.output.adjustedPValues) {
        neighborBindings.push({
          name: 'adjustedPValues',
          view: props.output.adjustedPValues,
          type: 'f32',
          access: 'read_write'
        });
      }
    }

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-neighbors`,
        operation: OPERATION,
        variant: `local-moran${props.output.localI && fuseLocalI ? '-local-i' : ''}${fuseClassify ? '-classify' : ''}${isAdjusting ? '-p-value' : ''}`,
        bindings: neighborBindings,
        invocationCount: rows,
        declarations: fuseClassify ? parameterWGSL : floatWGSL,
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
    if (STANDARDIZE && weightSum > 0.0) {
      // Row-standardized weights are divided by what remains of their row sum.
      neighborSum = neighborSum / weightSum;
      squareSum = squareSum / (weightSum * weightSum);
      weightSum = 1.0;
    }
    lag = neighborSum;
    let others = statistics[statisticsOffset + MOMENTS] - 1.0;
    let sumOfSquares = statistics[statisticsOffset + MOMENTS + 3u];
    let otherMean = -centered / others;
    let otherVariance = (sumOfSquares - centered * centered) / others - otherMean * otherMean;
    let lagVariance = otherVariance * (others * squareSum - weightSum * weightSum) / (others - 1.0);
    if (others >= 2.0 && centered != 0.0 && neighborCount >= MINIMUM_NEIGHBORS &&
        lagVariance > 0.0 && isFiniteFloat(lagVariance)) {
      let lagZ = (neighborSum - weightSum * otherMean) / sqrt(lagVariance);
      zScore = select(lagZ, -lagZ, centered < 0.0);
    }
  }
  zScores[zScoresOffset + index] = zScore;
  ${spatialLag ? 'spatialLag[spatialLagOffset + index] = lag;' : ''}
  ${props.output.neighborCounts ? 'neighborCounts[neighborCountsOffset + index] = neighborCount;' : ''}
  ${unadjustedPValues ? 'pValues[pValuesOffset + index] = select(getQuietNaN(index), getTwoSidedPValue(zScore), isFiniteFloat(zScore));' : ''}
  ${
    props.output.localI && fuseLocalI
      ? `{
    let sumOfSquares = statistics[statisticsOffset + MOMENTS + 3u];
    // Constant values (M2 = 0) leave I undefined: NaN, not the 0 of a kept island.
    let scale = select(getQuietNaN(index), (statistics[statisticsOffset + MOMENTS] - 1.0) / sumOfSquares, sumOfSquares > 0.0);
    // Excluded rows keep their NaN centered value; kept islands have a zero lag and so I = 0.
    localI[localIOffset + index] = select(select(centered * lag * scale, 0.0, lag == 0.0 && isFiniteFloat(scale)), getQuietNaN(index), !isFiniteFloat(centered));
  }`
      : ''
  }
  ${fuseClassify ? classifyWGSL('zScore', 'centered', 'lag') : ''}`
      })
    );

    if (props.output.localI && !fuseLocalI) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-local-i`,
          operation: OPERATION,
          variant: 'local-i',
          bindings: [
            {name: 'statistics', view: inputs.statistics, type: 'f32', access: 'read'},
            {name: 'spatialLag', view: spatialLag!, type: 'f32', access: 'read'},
            {name: 'localI', view: props.output.localI, type: 'f32', access: 'read_write'}
          ],
          invocationCount: rows,
          declarations: floatWGSL,
          body: `let centered = statistics[statisticsOffset + index];
  let lag = spatialLag[spatialLagOffset + index];
  let sumOfSquares = statistics[statisticsOffset + MOMENTS + 3u];
  // Constant values (M2 = 0) leave I undefined: NaN, not the 0 of a kept island.
  let scale = select(getQuietNaN(index), (statistics[statisticsOffset + MOMENTS] - 1.0) / sumOfSquares, sumOfSquares > 0.0);
  // Excluded rows keep their NaN centered value; kept islands have a zero lag and so I = 0.
  localI[localIOffset + index] = select(select(centered * lag * scale, 0.0, lag == 0.0 && isFiniteFloat(scale)), getQuietNaN(index), !isFiniteFloat(centered));`
        })
      );
    }

    if (isAdjusting) {
      nodes.push(
        ...new GPUMultipleTestingAdjustment({
          id: `${id}-multiple-testing`,
          pValues: unadjustedPValues!,
          method: this.multipleTesting,
          output: {adjustedPValues: adjustedPValues!}
        }).getCommandNodes(graph)
      );
    }
    if (!needsClassifyKernel(fuseClassify)) {
      return nodes;
    }
    const classifyBindings: WGSLKernelBinding[] = [
      {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
      {name: 'statistics', view: inputs.statistics, type: 'f32', access: 'read'},
      {name: 'zScores', view: zScores, type: 'f32', access: 'read'},
      {name: 'spatialLag', view: spatialLag!, type: 'f32', access: 'read'}
    ];
    if (isAdjusting) {
      classifyBindings.push({
        name: 'adjustedPValues',
        view: adjustedPValues!,
        type: 'f32',
        access: 'read'
      });
    }
    if (props.output.quadrants) {
      classifyBindings.push({
        name: 'quadrants',
        view: props.output.quadrants,
        type: 'u32',
        access: 'read_write'
      });
    }
    if (!isAdjusting && props.output.pValues) {
      classifyBindings.push({
        name: 'pValues',
        view: props.output.pValues,
        type: 'f32',
        access: 'read_write'
      });
    }
    if (!isAdjusting && props.output.adjustedPValues) {
      classifyBindings.push({
        name: 'adjustedPValues',
        view: props.output.adjustedPValues,
        type: 'f32',
        access: 'read_write'
      });
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-classify`,
        operation: OPERATION,
        variant: isAdjusting ? 'classify-adjusted' : 'classify',
        bindings: classifyBindings,
        invocationCount: rows,
        declarations: getSpatialAutocorrelationSharedWGSL(),
        body: `let zScore = zScores[zScoresOffset + index];
  let centered = statistics[statisticsOffset + index];
  let lag = spatialLag[spatialLagOffset + index];
  ${
    isAdjusting
      ? `let adjusted = adjustedPValues[adjustedPValuesOffset + index];
  ${quadrantWGSL('centered', 'lag', 'isFiniteFloat(zScore) && isFiniteFloat(adjusted) && adjusted <= readParameter(0u)')}`
      : classifyWGSL('zScore', 'centered', 'lag')
  }`
      })
    );
    return nodes;
  }
}
