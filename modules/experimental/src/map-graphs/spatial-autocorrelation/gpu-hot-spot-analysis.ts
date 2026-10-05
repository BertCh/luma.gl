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
import type {GPUMapGraphRecipe} from '../map-graph-types';
import {
  createMapGraphKernelNode,
  getWGSLFloatLiteral,
  type MapGraphKernelBinding
} from '../map-graph-kernels';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../map-graph-utils';
import {
  getFalseDiscoveryRateNodes,
  getSpatialAutocorrelationInputNodes,
  getSpatialAutocorrelationNeighborLoopWGSL,
  getSpatialAutocorrelationSharedWGSL,
  SPATIAL_AUTOCORRELATION_FLOAT_WGSL,
  validateSpatialAutocorrelationInputs
} from './spatial-autocorrelation-kernels';
import {
  GPU_HOT_SPOT_CRITICAL_Z_SCORES,
  GPU_HOT_SPOT_SIGNIFICANCE_LEVELS
} from './spatial-autocorrelation-parameters';

const OPERATION = 'GPUHotSpotAnalysis';

/**
 * Properties for {@link GPUHotSpotAnalysis}.
 *
 * Per-frame (no rebuild or recompile): the contents of `positions`, `values`, `mask` and
 * `parameters` (bounds, radius, optional fixed moments). Compile-time: the row count, `gridSize`,
 * `falseDiscoveryRate`, and which optional views are present.
 */
export type GPUHotSpotAnalysisProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'hot-spot-analysis'`. */
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
  /** Caller-owned Gi* z-score per row; quiet NaN for excluded or undefined rows. */
  zScores: GraphDataView<'float32'>;
  /**
   * Optional caller-owned confidence bin per row, `sint32` in `-3..3`: `±3`, `±2`, `±1` for
   * 99%, 95% and 90% confidence hot (`+`) or cold (`-`) spots, `0` otherwise.
   */
  bins?: GraphDataView<'sint32'>;
  /** Optional caller-owned two-sided normal p-value per row; quiet NaN where z is NaN. */
  pValues?: GraphDataView<'float32'>;
  /** Optional caller-owned neighbor count per row, the row itself included; 0 for excluded rows. */
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
 * Getis-Ord Gi* local hot spot analysis over planar points with a distance-band weight.
 *
 * Definition, which the GPU result matches within f32 rounding:
 * - Included rows are mask-selected rows with a finite value and a finite position inside the
 *   inclusive bounds. `n`, the mean `X` and the population standard deviation
 *   `S = sqrt(sum (x - X)^2 / n)` are computed over included rows (or taken from fixed moments).
 * - Binary weights: `w_ij = 1` when `|p_i - p_j| <= radius`, the row itself included (Gi*).
 * - `z_i = sum_j w_ij (x_j - X) / (S * sqrt((n * k_i - k_i^2) / (n - 1)))`, where `k_i` is the
 *   neighbor count. This equals the ArcGIS Hot Spot Analysis formula with binary weights, computed
 *   on centered values to avoid cancellation. z is NaN when `n < 2`, `S = 0` or `k_i >= n`.
 * - p-values are two-sided normal; bins use the 90/95/99% critical values, or BH-FDR.
 *
 * Determinism: each row sums its neighbors in a fixed cell and row order, global moments use
 * fixed-order tree sums, and FDR uses integer atomics only, so repeated encodings on one device
 * are bitwise identical.
 *
 * Caveat: moments describe the rows included this frame. When the mask follows the viewport,
 * panning changes `n`, `X` and `S` and therefore every z-score, and rows near the edge of the
 * included set lose neighbors. Pass `fixedMoments` (for example the `globalStatistics` of a
 * full-dataset frame) to pin the reference distribution.
 *
 * Composition: a cell-key kernel, `GPUGroupAggregation` cell counts, `GPUScan` offsets, stable
 * `GPUSort` of rows by cell, two-level fixed-order tree sums for the moments, one gather kernel per
 * focus row over the 3x3 cell neighborhood, optional FDR nodes, and a classify kernel.
 *
 * Non-goals: permutation (pseudo) p-values, k-nearest or inverse-distance weights, geodesic
 * distances, chunked inputs, space-time neighborhoods.
 */
export class GPUHotSpotAnalysis implements GPUMapGraphRecipe {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Stable recipe name. */
  readonly recipe = 'hot-spot-analysis';
  /** Validated properties. */
  readonly props: GPUHotSpotAnalysisProps;

  constructor(props: GPUHotSpotAnalysisProps) {
    this.id = props.id ?? 'hot-spot-analysis';
    this.props = props;
    const id = this.id;
    const rows = validateSpatialAutocorrelationInputs({...props, id});
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
        throw new Error(`${id} ${name} length must equal positions length`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.zScores, props.bins, props.pValues, props.neighborCounts, props.globalStatistics],
      [props.positions, props.values, props.parameters, props.mask]
    );
  }

  /** Returns the hot-spot nodes in dependency order. */
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
      props.bins,
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
    const sharedWGSL = getSpatialAutocorrelationSharedWGSL(gridSize);

    const neighborBindings: MapGraphKernelBinding[] = [
      {name: 'positions', view: positions, type: 'f32', access: 'read'},
      {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
      {name: 'sortedRows', view: inputs.sortedRows, type: 'u32', access: 'read'},
      {name: 'cellOffsets', view: inputs.cellOffsets, type: 'u32', access: 'read'},
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
      createMapGraphKernelNode<Parameters>(graph, {
        id: `${id}-neighbors`,
        operation: OPERATION,
        variant: 'gi-star',
        bindings: neighborBindings,
        invocationCount: rows,
        declarations: `${sharedWGSL}
const MOMENTS: u32 = ${rows}u;`,
        body: `let centered = statistics[statisticsOffset + index];
  var zScore = getQuietNaN(index);
  var neighborCount = 0u;
  if (isFiniteFloat(centered)) {
    let lattice = readLattice();
    let x = positions[positionsOffset + index * 2u];
    let y = positions[positionsOffset + index * 2u + 1u];
    var neighborSum = 0.0;
    ${getSpatialAutocorrelationNeighborLoopWGSL(`neighborSum += statistics[statisticsOffset + neighbor];
        neighborCount++;`)}
    let count = statistics[statisticsOffset + MOMENTS];
    let variance = statistics[statisticsOffset + MOMENTS + 2u];
    let weightSum = f32(neighborCount);
    let spread = weightSum * (count - weightSum) / (count - 1.0);
    if (count >= 2.0 && variance > 0.0 && spread > 0.0 && isFiniteFloat(spread)) {
      zScore = neighborSum / (sqrt(variance) * sqrt(spread));
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
            gridSize,
            levelExpressions: GPU_HOT_SPOT_SIGNIFICANCE_LEVELS.map(getWGSLFloatLiteral)
          })
        : undefined;
    if (falseDiscoveryRate) {
      nodes.push(...falseDiscoveryRate.nodes);
    }
    const classifyBindings: MapGraphKernelBinding[] = [
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
      createMapGraphKernelNode<Parameters>(graph, {
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
