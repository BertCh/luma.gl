// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
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
  getFalseDiscoveryRateNodes,
  getSpatialAutocorrelationInputNodes,
  getSpatialWeightsNeighborLoopWGSL,
  SPATIAL_AUTOCORRELATION_FLOAT_WGSL,
  validateSpatialAutocorrelationInputs
} from './spatial-autocorrelation-kernels';
import type {GPUSpatialWeights} from '../spatial-weights/index';
import {
  GPU_HOT_SPOT_CRITICAL_Z_SCORES,
  GPU_HOT_SPOT_SIGNIFICANCE_LEVELS
} from './spatial-autocorrelation-parameters';

const OPERATION = 'GPUHotSpotAnalysis';

/**
 * Properties for {@link GPUHotSpotAnalysis}.
 *
 * Per-frame (no rebuild or recompile): the contents of `weights`, `values`, `mask` and
 * `parameters` (optional fixed moments). Compile-time: the row count, `selfWeight`,
 * `falseDiscoveryRate`, and which optional views are present.
 */
export type GPUHotSpotAnalysisProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'hot-spot-analysis'`. */
  id?: string;
  /**
   * Square self-join spatial weights, one row per observation (for example from
   * `GPUNeighborSearch`, optionally transformed). Used as given: the caller applies any
   * transform (binary, row standardization, kernel). Fewer than 2^24 rows. The focal row's own
   * weight is not read from the matrix (`w_ii = 0` by invariant); see `selfWeight`.
   */
  weights: GPUSpatialWeights;
  /**
   * Weight `w_ii` given to the focal row itself, the "star" in Gi*. Compile-time, finite and
   * non-negative. Defaults to `1`, which with binary weights is the classic Gi*. Use `0` for the
   * Gi statistic (focal row excluded). With row-standardized weights `1` makes the focal row as
   * heavy as all its neighbors together, so prefer `0` there or fold the focal row into the
   * weights before standardizing.
   */
  selfWeight?: number;
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
  /** Caller-owned Gi* z-score per row; quiet NaN for excluded or undefined rows. */
  zScores: GraphDataView<'float32'>;
  /**
   * Optional caller-owned confidence bin per row, `sint32` in `-3..3`: `±3`, `±2`, `±1` for
   * 99%, 95% and 90% confidence hot (`+`) or cold (`-`) spots, `0` otherwise.
   */
  bins?: GraphDataView<'sint32'>;
  /** Optional caller-owned two-sided normal p-value per row; quiet NaN where z is NaN. */
  pValues?: GraphDataView<'float32'>;
  /** Optional caller-owned count of included weight entries per row, plus the row itself when `selfWeight` is nonzero; 0 for excluded rows. */
  neighborCounts?: GraphDataView<'uint32'>;
  /** Optional caller-owned `[n, mean, variance, standardDeviation]` actually used this frame. */
  globalStatistics?: GraphDataView<'float32'>;
  /**
   * Compile-time. When true, bins use Benjamini-Hochberg false discovery rate correction over all
   * rows with a finite z (the ArcGIS "Apply False Discovery Rate Correction" option): a row gets
   * level `L` when its p-value passes the BH step-up test at `alpha = 0.10, 0.05, 0.01`. Adds one
   * 31-bit sort. Defaults to false (uncorrected critical values `1.645, 1.960, 2.576`).
   */
  falseDiscoveryRate?: boolean;
};

/**
 * Getis-Ord Gi* local hot spot analysis on caller-supplied spatial weights.
 *
 * Definition, which the GPU result matches within f32 rounding:
 * - Included rows are mask-selected rows with a finite value. `n`, the mean `X` and the
 *   population standard deviation `S = sqrt(sum (x - X)^2 / n)` are computed over included rows
 *   (or taken from fixed moments).
 * - Weights `w_ij` are the CSR entries of row `i` that point at included rows other than `i`,
 *   used as given, plus the focal weight `w_ii = selfWeight` (default 1, the star of Gi*).
 *   `W_i = sum_j w_ij` and `S1_i = sum_j w_ij^2`.
 * - `z_i = sum_j w_ij (x_j - X) / (S * sqrt((n * S1_i - W_i^2) / (n - 1)))`, the Ord-Getis Gi*
 *   z-score for arbitrary weights (esda `G_Local(star=True)` and ArcGIS Hot Spot Analysis),
 *   computed on centered values to avoid cancellation. With binary weights `W_i = S1_i = k_i`.
 *   z is NaN when `n < 2`, `S = 0` or `n * S1_i <= W_i^2`.
 * - p-values are two-sided normal; bins use the 90/95/99% critical values, or BH-FDR.
 *
 * Determinism: each row sums its neighbors in CSR slot order, global moments use fixed-order tree
 * sums, and FDR uses integer atomics only, so repeated encodings on one device are bitwise
 * identical.
 *
 * Caveat: moments describe the rows included this frame. When the mask follows the viewport,
 * panning changes `n`, `X` and `S` and therefore every z-score. Pass `fixedMoments` (for example
 * the `globalStatistics` of a full-dataset frame) to pin the reference distribution.
 *
 * Composition: a validity kernel, fixed-order two-level tree sums for the moments, one gather
 * kernel per focus row over its CSR row, optional FDR nodes (one 31-bit `GPUSort`), and a classify
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

  constructor(props: GPUHotSpotAnalysisProps) {
    this.id = props.id ?? 'hot-spot-analysis';
    this.props = props;
    const id = this.id;
    const rows = validateSpatialAutocorrelationInputs({...props, id});
    const selfWeight = props.selfWeight ?? 1;
    if (!Number.isFinite(selfWeight) || selfWeight < 0) {
      throw new Error(`${id} selfWeight must be a finite number >= 0`);
    }
    validatePackedView(props.zScores, ['float32'], `${id} zScores`);
    if (props.bins) {
      validatePackedView(props.bins, ['sint32'], `${id} bins`);
    }
    if (props.pValues) {
      validatePackedView(props.pValues, ['float32'], `${id} pValues`);
    }
    if (props.neighborCounts) {
      validatePackedUint32View(props.neighborCounts, `${id} neighborCounts`);
    }
    for (const [name, view] of [
      ['zScores', props.zScores],
      ['bins', props.bins],
      ['pValues', props.pValues],
      ['neighborCounts', props.neighborCounts]
    ] as const) {
      if (view && view.length !== rows) {
        throw new Error(`${id} ${name} length must equal the weights row count`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.zScores, props.bins, props.pValues, props.neighborCounts, props.globalStatistics],
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

  /** Returns the hot-spot nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {weights, parameters, zScores} = props;
    const selfWeight = props.selfWeight ?? 1;
    validateGraphViewsBelongToGraph(id, graph, [
      weights.offsets,
      weights.neighbors,
      weights.weights,
      props.values,
      parameters,
      props.mask,
      zScores,
      props.bins,
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
    const sharedWGSL = SPATIAL_AUTOCORRELATION_FLOAT_WGSL;

    const neighborBindings: WGSLKernelBinding[] = [
      {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
      {name: 'weights', view: weights.weights, type: 'f32', access: 'read'},
      {name: 'statistics', view: inputs.statistics, type: 'f32', access: 'read'},
      {name: 'zScores', view: zScores, type: 'f32', access: 'read_write'}
    ];
    if (props.neighborCounts) {
      neighborBindings.push({
        name: 'neighborCounts',
        view: props.neighborCounts,
        type: 'u32',
        access: 'read_write'
      });
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-neighbors`,
        operation: OPERATION,
        variant: 'gi-star',
        bindings: neighborBindings,
        invocationCount: rows,
        declarations: `${sharedWGSL}
const MOMENTS: u32 = ${rows}u;
const ROWS: u32 = ${rows}u;
const SELF_WEIGHT: f32 = ${getWGSLFloatLiteral(selfWeight)};`,
        body: `let centered = statistics[statisticsOffset + index];
  var zScore = getQuietNaN(index);
  var neighborCount = 0u;
  if (isFiniteFloat(centered)) {
    var weightedSum = SELF_WEIGHT * centered;
    var weightSum = SELF_WEIGHT;
    var squareSum = SELF_WEIGHT * SELF_WEIGHT;
    neighborCount = select(0u, 1u, SELF_WEIGHT != 0.0);
    ${getSpatialWeightsNeighborLoopWGSL(`weightedSum += weight * statistics[statisticsOffset + neighbor];
      weightSum += weight;
      squareSum += weight * weight;
      neighborCount++;`)}
    let count = statistics[statisticsOffset + MOMENTS];
    let variance = statistics[statisticsOffset + MOMENTS + 2u];
    let spread = (count * squareSum - weightSum * weightSum) / (count - 1.0);
    if (count >= 2.0 && variance > 0.0 && spread > 0.0 && isFiniteFloat(spread)) {
      zScore = weightedSum / (sqrt(variance) * sqrt(spread));
    }
  }
  zScores[zScoresOffset + index] = zScore;
  ${props.neighborCounts ? 'neighborCounts[neighborCountsOffset + index] = neighborCount;' : ''}`
      })
    );

    if (!props.bins && !props.pValues) {
      return nodes;
    }
    const falseDiscoveryRate =
      props.falseDiscoveryRate && props.bins
        ? getFalseDiscoveryRateNodes<Parameters>(graph, {
            id: `${id}-fdr`,
            operation: OPERATION,
            zScores,
            parameters,
            levelExpressions: GPU_HOT_SPOT_SIGNIFICANCE_LEVELS.map(getWGSLFloatLiteral)
          })
        : undefined;
    if (falseDiscoveryRate) {
      nodes.push(...falseDiscoveryRate.nodes);
    }
    const classifyBindings: WGSLKernelBinding[] = [
      {name: 'zScores', view: zScores, type: 'f32', access: 'read'}
    ];
    if (falseDiscoveryRate) {
      classifyBindings.push(
        {name: 'ranks', view: falseDiscoveryRate.ranks, type: 'u32', access: 'read'},
        {name: 'counters', view: falseDiscoveryRate.counters, type: 'u32', access: 'read'}
      );
    }
    if (props.bins) {
      classifyBindings.push({name: 'bins', view: props.bins, type: 'i32', access: 'read_write'});
    }
    if (props.pValues) {
      classifyBindings.push({
        name: 'pValues',
        view: props.pValues,
        type: 'f32',
        access: 'read_write'
      });
    }
    const [critical90, critical95, critical99] =
      GPU_HOT_SPOT_CRITICAL_Z_SCORES.map(getWGSLFloatLiteral);
    const levelSource = falseDiscoveryRate
      ? `let rank = ranks[ranksOffset + index];
    if (rank > 0u && rank <= counters[countersOffset + 3u]) {
      level = 3;
    } else if (rank > 0u && rank <= counters[countersOffset + 2u]) {
      level = 2;
    } else if (rank > 0u && rank <= counters[countersOffset + 1u]) {
      level = 1;
    }`
      : `let absoluteZ = abs(zScore);
    level = select(select(select(0, 1, absoluteZ >= ${critical90}), 2, absoluteZ >= ${critical95}), 3, absoluteZ >= ${critical99});`;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-classify`,
        operation: OPERATION,
        variant: falseDiscoveryRate ? 'classify-fdr' : 'classify',
        bindings: classifyBindings,
        invocationCount: rows,
        declarations: SPATIAL_AUTOCORRELATION_FLOAT_WGSL,
        body: `let zScore = zScores[zScoresOffset + index];
  let finite = isFiniteFloat(zScore);
  var level = 0;
  if (finite) {
    ${levelSource}
  }
  ${props.bins ? 'bins[binsOffset + index] = select(level, -level, zScore < 0.0);' : ''}
  ${
    props.pValues
      ? 'pValues[pValuesOffset + index] = select(getQuietNaN(index), getTwoSidedPValue(zScore), finite);'
      : ''
  }`
      })
    );
    return nodes;
  }
}
