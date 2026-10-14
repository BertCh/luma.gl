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
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
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
  GPU_HOT_SPOT_CRITICAL_Z_SCORES,
  GPU_HOT_SPOT_SIGNIFICANCE_LEVELS,
  type GPUSpatialAutocorrelationSettings
} from './spatial-autocorrelation-parameters';

/** Storage buffers per stage on a core WebGPU device (`maxStorageBuffersPerShaderStage`). */
const CORE_STORAGE_BUFFER_LIMIT = 8;
const OPERATION = 'GPUHotSpotAnalysis';

/** Caller-owned outputs of {@link GPUHotSpotAnalysis}. */
export type GPUHotSpotAnalysisOutput = {
  /** Caller-owned Gi* z-score per row; quiet NaN for excluded or undefined rows. */
  zScores: GraphDataView<'float32'>;
  /**
   * Optional caller-owned confidence bin per row, `sint32` in `-3..3`: `±3`, `±2`, `±1` for
   * 99%, 95% and 90% confidence hot (`+`) or cold (`-`) spots, `0` otherwise. Legend contract:
   * level `L` means the adjusted p-value (the plain p-value for `multipleTesting: 'none'`) is at
   * most `0.01`, `0.05` or `0.10`, whatever the correction method; the sign is the sign of z.
   */
  bins?: GraphDataView<'sint32'>;
  /** Optional caller-owned unadjusted two-sided normal p-value per row; quiet NaN where z is NaN. */
  pValues?: GraphDataView<'float32'>;
  /**
   * Optional caller-owned p-value adjusted by `multipleTesting` (equal to `pValues` for `'none'`);
   * quiet NaN where z is NaN. The test family is every row with a defined z-score, never a viewport.
   */
  adjustedPValues?: GraphDataView<'float32'>;
  /**
   * Optional caller-owned count of active neighbors per row, the row itself not included even when
   * `selfWeight` is nonzero; 0 for rows outside the analysis.
   */
  neighborCounts?: GraphDataView<'uint32'>;
  /** Optional caller-owned `[n, mean, variance, standardDeviation, islandCount]` actually used this frame. */
  globalStatistics?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUHotSpotAnalysis}.
 *
 * Per-frame (no rebuild or recompile): the contents of `weights`, `values`, `mask` and
 * `parameters` (optional fixed moments). Compile-time: the row count, `selfWeight`, `islands`,
 * `minimumNeighbors`, `multipleTesting`, and which optional views are present.
 */
export type GPUHotSpotAnalysisProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'hot-spot-analysis'`. */
  id?: string;
  /**
   * Square self-join spatial weights, one row per observation (for example from
   * `GPUNeighborSearch`, optionally transformed). Used as given, except that when
   * `weights.information.transform` is `'row'` each row's weights are re-standardized over the neighbors
   * that remain after masking. The caller applies any other transform (binary, kernel). Fewer
   * than 2^24 rows. The focal row's own weight is not read from the matrix (`w_ii = 0` by
   * invariant); see `selfWeight`. A neighbor is a slot with a positive weight that points at
   * another included row. If `weights.status.overflow` is nonzero every output is NaN.
   */
  weights: GPUSpatialWeights;
  /**
   * Weight `w_ii` given to the focal row itself, the "star" in Gi*. Compile-time, finite and
   * non-negative. Defaults to `1`, which with binary weights is the classic Gi*. Use `0` for the
   * Ord-Getis Gi statistic (focal row excluded from both the sum and the moments, see the class
   * description). With row-standardized weights `1` makes the focal row as heavy as all its
   * neighbors together, so prefer `0` there or fold the focal row into the weights before
   * standardizing.
   */
  selfWeight?: number;
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
   * included rows). `'exclude'` (default) removes them from `n`, the mean and the standard
   * deviation and writes NaN; `'keep'` keeps them in the moments, with NaN z-score, p-value and
   * bin 0. Islands never get a z-score from their own value through `selfWeight`. The count is
   * reported in `globalStatistics[4]`. The decision is made in one pass, with no cascade: a row
   * whose only positive neighbors were islands has no neighbor among the analysis rows, gets
   * NaN, and stays in `n`.
   */
  islands?: GPUIslandPolicy;
  /**
   * Compile-time. Rows with fewer active neighbors (the row itself excluded) get a NaN z-score
   * and p-value and bin 0. Defaults to 1, so only rows with no neighbor are undefined. Esri
   * recommends at least 8 neighbors for a meaningful Gi* z-score; pass 8 for that behavior. Must
   * be an integer of at least 1.
   */
  minimumNeighbors?: number;
  /** Caller-owned outputs. */
  output: GPUHotSpotAnalysisOutput;
  /**
   * Compile-time. Multiple-testing correction of the two-sided p-values
   * (`GPUMultipleTestingAdjustment`): `'none'` (default; bins use the critical values `1.645,
   * 1.960, 2.576`), `'bonferroni'`, `'holm'`, `'benjamini-hochberg'` (the Esri "Apply False
   * Discovery Rate Correction" option) or `'benjamini-yekutieli'` (valid under any dependence).
   * With a correction a row gets level `L` when its adjusted p-value is at most `0.10, 0.05, 0.01`.
   */
  multipleTesting?: GPUMultipleTestingMethod;
};

/**
 * Getis-Ord Gi* local hot spot analysis on caller-supplied spatial weights.
 *
 * Definition, which the GPU result matches within f32 rounding:
 * - Included rows are mask-selected rows with a finite value. Islands (included rows without a
 *   positive-weight neighbor among the included rows) are removed by default. `n`, the mean `X`
 *   and the population standard deviation `S = sqrt(sum (x - X)^2 / n)` are computed over the
 *   analysis rows (or taken from fixed moments), with a second-pass mean correction.
 * - Weights `w_ij` are the CSR entries of row `i` with a positive weight pointing at analysis rows
 *   other than `i`. They are used as given, or divided by their sum when the weights are
 *   row-standardized (`weights.information.transform === 'row'`), plus the focal weight
 *   `w_ii = selfWeight` (default 1, the star of Gi*). `W_i = sum_j w_ij` and `S1_i = sum_j w_ij^2`.
 * - With `selfWeight > 0` (Gi*): `z_i = sum_j w_ij (x_j - X) / (S * sqrt((n * S1_i - W_i^2) / (n - 1)))`,
 *   the Ord-Getis Gi* z-score for arbitrary weights (esda `G_Local(star=True)` and ArcGIS Hot Spot
 *   Analysis), computed on centered values to avoid cancellation. With binary weights
 *   `W_i = S1_i = k_i`. z is NaN when `n < 2`, `S = 0` or `n * S1_i <= W_i^2`.
 * - With `selfWeight = 0` (Gi): the exact Ord-Getis form, which standardizes with the mean and
 *   variance of the `n - 1` other observations (`x_i` excluded): `mu_i` and `s_i` are those
 *   moments, `z_i = (sum_j w_ij x_j - W_i mu_i) / (s_i * sqrt(((n - 1) S1_i - W_i^2) / (n - 2)))`.
 *   It equals esda's `G_Local(star=False)` analytic `Zs` for binary weights. z is NaN when
 *   `n < 3` or `s_i = 0`.
 * - A row with fewer than `minimumNeighbors` active neighbors (zero by default) has no z-score:
 *   a row never gets a z-score from its own value alone, under either island policy.
 * - p-values are two-sided normal. Bins use the 90/95/99% critical values, or the adjusted
 *   p-value when `multipleTesting` is set. Legend contract: bin level `L` always means the
 *   adjusted p-value (the plain p-value for `'none'`) is at most `0.10`, `0.05` or `0.01`.
 *
 * Determinism: each row sums its neighbors in CSR slot order, global moments use fixed-order tree
 * sums, and the multiple-testing adjustment is deterministic, so repeated encodings on one device
 * are bitwise identical.
 *
 * Caveat: moments describe the rows included this frame. When the mask follows the viewport,
 * panning changes `n`, `X` and `S` and therefore every z-score. Pass `fixedMoments` (for example
 * the `globalStatistics` of a full-dataset frame) to pin the reference distribution.
 *
 * Composition: an island pass, fixed-order two-level tree sums for the moments, one gather
 * kernel per focus row over its CSR row, an optional multiple-testing adjustment, and a classify
 * kernel.
 *
 * Non-goals: permutation (pseudo) p-values, building the weights, geodesic distances, chunked
 * inputs, space-time neighborhoods (see `GPUEmergingHotSpots`).
 */
export class GPUHotSpotAnalysis implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUHotSpotAnalysisProps;
  /** Resolved correction method. */
  readonly multipleTesting: GPUMultipleTestingMethod;
  /** Resolved island policy. */
  readonly islands: GPUIslandPolicy;
  /** Resolved minimum number of active neighbors. */
  readonly minimumNeighbors: number;

  constructor(props: GPUHotSpotAnalysisProps) {
    this.id = props.id ?? 'hot-spot-analysis';
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
    const selfWeight = props.selfWeight ?? 1;
    if (!Number.isFinite(selfWeight) || selfWeight < 0) {
      throw new Error(`${id} selfWeight must be a finite number >= 0`);
    }
    validatePackedView(props.output.zScores, ['float32'], `${id} zScores`);
    if (props.output.bins) {
      validatePackedView(props.output.bins, ['sint32'], `${id} bins`);
    }
    if (props.output.pValues) {
      validatePackedView(props.output.pValues, ['float32'], `${id} pValues`);
    }
    if (props.output.adjustedPValues) {
      validatePackedView(props.output.adjustedPValues, ['float32'], `${id} adjustedPValues`);
    }
    if (props.output.neighborCounts) {
      validatePackedUint32View(props.output.neighborCounts, `${id} neighborCounts`);
    }
    for (const [name, view] of [
      ['zScores', props.output.zScores],
      ['bins', props.output.bins],
      ['pValues', props.output.pValues],
      ['adjustedPValues', props.output.adjustedPValues],
      ['neighborCounts', props.output.neighborCounts]
    ] as const) {
      if (view && view.length !== rows) {
        throw new Error(`${id} ${name} length must equal the weights row count`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        props.output.zScores,
        props.output.bins,
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
   * Describes how the p-values were produced, for building a legend: two-sided analytic normal
   * p-values, the correction method, and the tested family. Bins use the fixed levels
   * `0.10, 0.05, 0.01`, so no single significance level is reported.
   *
   * @param settings The settings packed into `parameters` with
   * `getGPUSpatialAutocorrelationParameterValues`; validated the same way.
   */
  getInferenceMetadata(settings: GPUSpatialAutocorrelationSettings = {}): GPUInferenceMetadata {
    // Packing performs the canonical validation of the settings.
    getGPUSpatialAutocorrelationParameterValues(settings);
    return {
      alternative: 'two-sided',
      permutationCount: 0,
      multipleTesting: this.multipleTesting,
      testedRowsDefinition: `included rows with at least ${this.minimumNeighbors} active neighbor${
        this.minimumNeighbors === 1 ? '' : 's'
      }${this.islands === 'exclude' ? ' (islands excluded)' : ''} and a defined z-score`
    };
  }

  /** Returns the hot-spot nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {weights, parameters} = props;
    const {zScores} = props.output;
    const selfWeight = props.selfWeight ?? 1;
    validateGraphViewsBelongToGraph(id, graph, [
      weights.offsets,
      weights.neighbors,
      weights.weights,
      weights.status?.overflow,
      props.values,
      parameters,
      props.mask,
      zScores,
      props.output.bins,
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
    const sharedWGSL = SPATIAL_AUTOCORRELATION_FLOAT_WGSL;

    const neighborBindings: WGSLKernelBinding[] = [
      {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
      {name: 'weights', view: weights.weights, type: 'f32', access: 'read'},
      {name: 'statistics', view: inputs.statistics, type: 'f32', access: 'read'},
      {name: 'zScores', view: zScores, type: 'f32', access: 'read_write'}
    ];
    if (props.output.neighborCounts) {
      neighborBindings.push({
        name: 'neighborCounts',
        view: props.output.neighborCounts,
        type: 'u32',
        access: 'read_write'
      });
    }
    const hasCorrection = this.multipleTesting !== 'none';
    const wantsClassify = Boolean(
      props.output.bins || props.output.pValues || props.output.adjustedPValues
    );
    const isAdjusting = hasCorrection && Boolean(props.output.bins || props.output.adjustedPValues);
    // Without a correction the bin and p-value are pure functions of the row's z-score, so they
    // are written by the neighbor kernel itself (one pass less over `zScores`). With a correction
    // the neighbor kernel writes the unadjusted p-values, the adjustment runs, and a classify
    // kernel assigns the bins from the adjusted p-values.
    // Fusing is skipped when it would bind more than the 8 storage buffers of a core device
    // (every optional output with `multipleTesting: 'none'`); a classify pass over `zScores` then
    // writes the same values.
    const fusedOutputCount =
      Number(Boolean(props.output.bins)) +
      Number(Boolean(props.output.pValues)) +
      Number(Boolean(props.output.adjustedPValues));
    const fuseClassify =
      wantsClassify &&
      !isAdjusting &&
      neighborBindings.length + fusedOutputCount <= CORE_STORAGE_BUFFER_LIMIT;
    const unadjustedPValues = isAdjusting
      ? (props.output.pValues ?? createTransientView(graph, `${id}-p-values`, 'float32', rows))
      : undefined;
    const adjustedPValues = isAdjusting
      ? (props.output.adjustedPValues ??
        createTransientView(graph, `${id}-adjusted-p-values`, 'float32', rows))
      : undefined;
    const [critical90, critical95, critical99] =
      GPU_HOT_SPOT_CRITICAL_Z_SCORES.map(getWGSLFloatLiteral);
    const [level90, level95, level99] = GPU_HOT_SPOT_SIGNIFICANCE_LEVELS.map(getWGSLFloatLiteral);
    const fusedOutputBindings: WGSLKernelBinding[] = [];
    if (props.output.bins) {
      fusedOutputBindings.push({
        name: 'bins',
        view: props.output.bins,
        type: 'i32',
        access: 'read_write'
      });
    }
    if (props.output.pValues) {
      fusedOutputBindings.push({
        name: 'pValues',
        view: props.output.pValues,
        type: 'f32',
        access: 'read_write'
      });
    }
    if (props.output.adjustedPValues) {
      fusedOutputBindings.push({
        name: 'adjustedPValues',
        view: props.output.adjustedPValues,
        type: 'f32',
        access: 'read_write'
      });
    }
    const classifyWGSL = `let finite = isFiniteFloat(zScore);
  var level = 0;
  if (finite) {
    let absoluteZ = abs(zScore);
    level = select(select(select(0, 1, absoluteZ >= ${critical90}), 2, absoluteZ >= ${critical95}), 3, absoluteZ >= ${critical99});
  }
  ${props.output.bins ? 'bins[binsOffset + index] = select(level, -level, zScore < 0.0);' : ''}
  ${
    props.output.pValues || props.output.adjustedPValues
      ? `let pValue = select(getQuietNaN(index), getTwoSidedPValue(zScore), finite);
  ${props.output.pValues ? 'pValues[pValuesOffset + index] = pValue;' : ''}
  ${props.output.adjustedPValues ? 'adjustedPValues[adjustedPValuesOffset + index] = pValue;' : ''}`
      : ''
  }`;
    if (fuseClassify) {
      neighborBindings.push(...fusedOutputBindings);
    }
    if (unadjustedPValues) {
      neighborBindings.push({
        name: 'pValues',
        view: unadjustedPValues,
        type: 'f32',
        access: 'read_write'
      });
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-neighbors`,
        operation: OPERATION,
        variant: `${selfWeight === 0 ? 'gi' : 'gi-star'}${fuseClassify ? '-classify' : ''}${isAdjusting ? '-p-value' : ''}`,
        bindings: neighborBindings,
        invocationCount: rows,
        declarations: `${sharedWGSL}
${getSpatialAutocorrelationConstantsWGSL(rows, weights.neighbors.length)}
const MINIMUM_NEIGHBORS: u32 = ${this.minimumNeighbors}u;
const STANDARDIZE: bool = ${isGPUSpatialWeightsRowStandardized(weights)};
const SELF_WEIGHT: f32 = ${getWGSLFloatLiteral(selfWeight)};`,
        body: `let centered = statistics[statisticsOffset + index];
  var zScore = getQuietNaN(index);
  var neighborCount = 0u;
  if (isFiniteFloat(centered)) {
    var weightedSum = 0.0;
    var weightSum = 0.0;
    var squareSum = 0.0;
    ${getSpatialWeightsNeighborLoopWGSL(`weightedSum += weight * statistics[statisticsOffset + neighbor];
      weightSum += weight;
      squareSum += weight * weight;
      neighborCount++;`)}
    if (STANDARDIZE && weightSum > 0.0) {
      // Row-standardized weights are divided by what remains of their row sum.
      weightedSum = weightedSum / weightSum;
      squareSum = squareSum / (weightSum * weightSum);
      weightSum = 1.0;
    }
    let count = statistics[statisticsOffset + MOMENTS];
    let variance = statistics[statisticsOffset + MOMENTS + 2u];
    if (neighborCount >= MINIMUM_NEIGHBORS) {
      ${
        selfWeight === 0
          ? `// Ord-Getis Gi: standardize with the n - 1 other observations. The centered values of the
      // population sum to zero, so the others sum to -centered.
      let others = count - 1.0;
      let sumOfSquares = statistics[statisticsOffset + MOMENTS + 3u];
      let otherMean = -centered / others;
      let otherVariance = (sumOfSquares - centered * centered) / others - otherMean * otherMean;
      let spread = (others * squareSum - weightSum * weightSum) / (others - 1.0);
      if (others >= 2.0 && otherVariance > 0.0 && spread > 0.0 && isFiniteFloat(spread)) {
        zScore = (weightedSum - weightSum * otherMean) / (sqrt(otherVariance) * sqrt(spread));
      }`
          : `weightedSum += SELF_WEIGHT * centered;
      weightSum += SELF_WEIGHT;
      squareSum += SELF_WEIGHT * SELF_WEIGHT;
      let spread = (count * squareSum - weightSum * weightSum) / (count - 1.0);
      if (count >= 2.0 && variance > 0.0 && spread > 0.0 && isFiniteFloat(spread)) {
        zScore = weightedSum / (sqrt(variance) * sqrt(spread));
      }`
      }
    }
  }
  zScores[zScoresOffset + index] = zScore;
  ${props.output.neighborCounts ? 'neighborCounts[neighborCountsOffset + index] = neighborCount;' : ''}
  ${unadjustedPValues ? 'pValues[pValuesOffset + index] = select(getQuietNaN(index), getTwoSidedPValue(zScore), isFiniteFloat(zScore));' : ''}
  ${fuseClassify ? classifyWGSL : ''}`
      })
    );

    if (wantsClassify && !isAdjusting && !fuseClassify) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-classify`,
          operation: OPERATION,
          variant: 'classify',
          bindings: [
            {name: 'zScores', view: zScores, type: 'f32', access: 'read'},
            ...fusedOutputBindings
          ],
          invocationCount: rows,
          declarations: sharedWGSL,
          body: `let zScore = zScores[zScoresOffset + index];
  ${classifyWGSL}`
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
    if (!isAdjusting || !props.output.bins) {
      return nodes;
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-classify`,
        operation: OPERATION,
        variant: 'classify-adjusted',
        bindings: [
          {name: 'zScores', view: zScores, type: 'f32', access: 'read'},
          {name: 'adjustedPValues', view: adjustedPValues!, type: 'f32', access: 'read'},
          {name: 'bins', view: props.output.bins, type: 'i32', access: 'read_write'}
        ],
        invocationCount: rows,
        declarations: SPATIAL_AUTOCORRELATION_FLOAT_WGSL,
        // Rows without a finite adjusted p-value were not tested and get bin 0.
        body: `let zScore = zScores[zScoresOffset + index];
  let adjusted = adjustedPValues[adjustedPValuesOffset + index];
  var level = 0;
  if (isFiniteFloat(zScore) && isFiniteFloat(adjusted)) {
    level = select(select(select(0, 1, adjusted <= ${level90}), 2, adjusted <= ${level95}), 3, adjusted <= ${level99});
  }
  bins[binsOffset + index] = select(level, -level, zScore < 0.0);`
      })
    );
    return nodes;
  }
}
