// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedUint32View,
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
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {ERFC_VALUE_WGSL} from '../../utils/wgsl-common';
import type {
  GPUInferenceMetadata,
  GPUMultipleTestingMethod
} from '../../analysis-core/contracts/inference';
import {
  GPUMultipleTestingAdjustment,
  GPU_MULTIPLE_TESTING_MAXIMUM_ROWS,
  resolveGPUMultipleTestingMethod
} from '../multiple-testing/index';
import {SPATIAL_AUTOCORRELATION_FLOAT_WGSL} from '../spatial-autocorrelation/spatial-autocorrelation-kernels';
import {validateGPUSpatialWeights, type GPUSpatialWeights} from '../spatial-weights/index';
import {
  getGPUEmergingHotSpotsParameterValues,
  type GPUEmergingHotSpotsSettings,
  GPU_EMERGING_HOT_SPOTS_MAXIMUM_RADIUS,
  GPU_EMERGING_HOT_SPOTS_MAXIMUM_SLICE_COUNT,
  GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH,
  GPU_EMERGING_HOT_SPOTS_STATISTICS_LENGTH
} from './emerging-hot-spot-parameters';

const OPERATION = 'GPUEmergingHotSpots';
/** Bins reduced by one workgroup in the first level of the moment sums. */
const BIN_BLOCK = 4096;
/** Threads of the moment-sum workgroups. */
const MOMENT_WORKGROUP_SIZE = 256;
/** Threads per workgroup of the space-time Gi* kernel; each workgroup covers whole cells. */
const GI_WORKGROUP_SIZE = 256;
/** Threads per cell in the cooperative Mann-Kendall pass. */
const MANN_KENDALL_WORKGROUP_SIZE = 64;
/** Below this series length a single invocation is cheaper than workgroup setup and reduction. */
const MANN_KENDALL_COOPERATIVE_SLICE_COUNT = 64;
const MAXIMUM_BIN_COUNT = 2 ** 31 - 1;

/** Caller-owned outputs of {@link GPUEmergingHotSpots}. */
export type GPUEmergingHotSpotsOutput = {
  /** Caller-owned per-bin Gi* z-scores in cube layout; NaN for missing bins. */
  giZScores: GraphDataView<'float32'>;
  /** Optional caller-owned per-bin two-sided normal p-value of the Gi* z, cube layout, NaN for missing bins. */
  pValues?: GraphDataView<'float32'>;
  /**
   * Optional caller-owned per-bin `multipleTesting`-adjusted p-value in cube layout; NaN for
   * missing bins. Equals `pValues` for `'none'`.
   */
  adjustedPValues?: GraphDataView<'float32'>;
  /**
   * Optional caller-owned per-cell index of the last finite slice, or
   * `GPU_EMERGING_HOT_SPOTS_NO_SLICE` (`0xffffffff`) for a cell without any.
   */
  lastValidSlice?: GraphDataView<'uint32'>;
  /** Caller-owned per-cell Mann-Kendall z of the Gi* series (continuity corrected). */
  trendZ: GraphDataView<'float32'>;
  /** Caller-owned per-cell two-sided Mann-Kendall p-value. */
  trendP: GraphDataView<'float32'>;
  /** Caller-owned per-cell Mann-Kendall statistic `S`, an exact integer. */
  trendS: GraphDataView<'sint32'>;
  /** Caller-owned per-cell category code from `GPU_EMERGING_HOT_SPOTS_CATEGORIES` (0 to 17). */
  category: GraphDataView<'uint32'>;
  /** Caller-owned per-cell count of significant hot slices (`z > 0` and adjusted p at most `1 - confidenceLevel`). */
  hotSliceCount: GraphDataView<'uint32'>;
  /** Caller-owned per-cell count of significant cold slices (`z < 0` and adjusted p at most `1 - confidenceLevel`). */
  coldSliceCount: GraphDataView<'uint32'>;
  /** Optional caller-owned `[n, mean, variance, standardDeviation]` of the valid bins. */
  globalStatistics?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUEmergingHotSpots}.
 *
 * The spatial neighborhood is either a regular lattice (`gridWidth`, `gridHeight` and the
 * per-frame `radius`) or arbitrary spatial `weights` over the cells; exactly one of the two.
 *
 * Per-frame (no rebuild or recompile): the contents of `values`, `mask`, `weights` and `parameters`
 * (radius in lattice mode, temporal window, critical z, trend significance and persistence
 * thresholds, minimum trend length, confidence level). Compile-time: `gridWidth`, `gridHeight`,
 * `sliceCount`, `maximumRadius`, `selfWeight`, `multipleTesting`, `finalSlice`, the value format,
 * and which optional views are present.
 */
export type GPUEmergingHotSpotsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'emerging-hot-spots'`. */
  id?: string;
  /**
   * Dense space-time cube indexed `cell * sliceCount + slice` with `cell = row * gridWidth +
   * column` in lattice mode and `cell` = the weights row in weights mode, the layout of `GPUTemporalReduction` output. A NaN bin is missing and excluded. A
   * `uint32` view (for example `GPUTemporalReduction` `counts`) is read as `f32(count)`, every bin
   * valid.
   */
  values: GraphDataView<'float32'> | GraphDataView<'uint32'>;
  /** Lattice mode: lattice width in cells. Compile-time. Set together with `gridHeight`. */
  gridWidth?: number;
  /** Lattice mode: lattice height in cells. Compile-time. Set together with `gridWidth`. */
  gridHeight?: number;
  /**
   * Weights mode: square self-join spatial weights over the cells (rows of `weights` are cells),
   * for example `GPUNeighborSearch` output over H3 cell centers, optionally transformed. Used as
   * given. The spatial neighborhood of a bin is the cell's CSR row (weights `w_ij`) and the cell
   * itself (`selfWeight`), each over the current slice and the `temporalWindow` previous slices,
   * every bin with temporal weight 1. Fewer than 2^24 cells. The per-frame `radius` is ignored.
   * Exclusive with `gridWidth` and `gridHeight`.
   */
  weights?: GPUSpatialWeights;
  /**
   * Weights mode: weight `w_ii` of the focal cell's own bins (the star of Gi*). Compile-time,
   * finite and non-negative. Defaults to `1`. Lattice mode always uses the focal cell with weight 1.
   */
  selfWeight?: number;
  /** Slices per cell, at most `GPU_EMERGING_HOT_SPOTS_MAXIMUM_SLICE_COUNT`. Compile-time. */
  sliceCount: number;
  /**
   * Largest per-frame `radius` in cells, at most `GPU_EMERGING_HOT_SPOTS_MAXIMUM_RADIUS`. Larger
   * per-frame radii are clamped. Compile-time. Defaults to 4.
   */
  maximumRadius?: number;
  /** Optional per-cell mask (one row per cell): zero excludes every bin of the cell. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Per-frame parameters: a packed float32 view of at least
   * `GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH` elements written with
   * `getGPUEmergingHotSpotsParameterValues` or {@link GPUEmergingHotSpots.getParameterValues}.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Compile-time multiple-testing correction of the per-bin hot and cold decision. Defaults to
   * `'benjamini-hochberg'` (ESRI applies FDR in emerging hot spot analysis; it was uncorrected).
   *
   * The two-sided normal p-value of every valid Gi* bin is adjusted over the whole family of valid
   * bins of the cube (never a display mask), and a bin is hot or cold when its adjusted p-value is
   * at most `1 - confidenceLevel`, with the sign of z. Bins within a temporal window overlap and
   * neighboring cells share observations, so the tests are dependent: `'benjamini-yekutieli'` is
   * the method that controls the false discovery rate under arbitrary dependence, at a cost in
   * power of about `ln m`. `'none'` reproduces the pre-correction rule `|z| >= criticalZ` exactly
   * and is the only method that accepts the `criticalZ` parameter. The family size is limited to
   * fewer than 2^24 bins unless `'none'` is used without `pValues` and `adjustedPValues`.
   */
  multipleTesting?: GPUMultipleTestingMethod;
  /**
   * Compile-time. How the cell's final slice is chosen. `'require'` (default): the final slice is
   * the last slice of the cube, and a cell whose last slice is missing (NaN) gets
   * `GPU_EMERGING_HOT_SPOTS_CATEGORIES.UNDETERMINED`. `'last-valid'`: the final slice is the last
   * finite one, so a stale cell can be labelled New or Persistent for a period it has no data in
   * (the previous behavior).
   */
  finalSlice?: 'require' | 'last-valid';
  /** Caller-owned outputs. */
  output: GPUEmergingHotSpotsOutput;
};

/**
 * ArcGIS-style emerging hot spot analysis over a dense space-time cube on a regular lattice.
 *
 * Passes, each deterministic and free of float atomics:
 * 1. Global `n`, mean and population standard deviation of valid bins: per-block sums over fixed
 *    1024-bin blocks, then one invocation adding the block partials in order (mean first, then
 *    the sum of squared deviations).
 * 2. Space-time Gi* per bin, the focal bin included. Lattice mode uses binary weights:
 *    `z = sum_j (x_j - X) / (S * sqrt((n k - k^2) / (n - 1)))` where `j` covers valid bins of cells
 *    whose lattice offset satisfies `dx^2 + dy^2 <= radius^2` in the current and `temporalWindow`
 *    previous slices, and `k` counts them. Neighbors are visited in a fixed order (offset rows,
 *    offset columns, ascending slice). z is NaN for a missing bin, `n < 2`, `S = 0` or `k >= n`.
 *    Weights mode generalizes to `z = sum_j w_j (x_j - X) / (S * sqrt((n S1 - W^2) / (n - 1)))`
 *    with `W = sum_j w_j` and `S1 = sum_j w_j^2` over the same bins, where `w_j` is the spatial
 *    weight of the bin's cell (`selfWeight` for the focal cell, CSR weight for the others; the
 *    temporal weight is 1) and neighbors are visited in CSR slot order, then ascending slice.
 *    z is NaN for a missing bin, `n < 2`, `S = 0` or `n S1 <= W^2`. With binary weights the two
 *    modes agree.
 * 3. Per-cell Mann-Kendall test over the finite z series (NaN slices skipped): `S = sum_{i<j}
 *    sign(z_j - z_i)` in exact integers, `Var = [n(n-1)(2n+5) - sum_t t(t-1)(2t+5)] / 18` with tie
 *    groups of exactly equal f32 values, `z = (S - sign(S)) / sqrt(Var)` (0 when `S = 0` or
 *    `Var <= 0`), and the two-sided p-value `erfc(|z| / sqrt(2))` with the Numerical Recipes
 *    `erfcc` Chebyshev fit (relative error below 1.2e-7); `p = 1` when `Var <= 0`.
 * 3b. Between 2 and 3: two-sided normal p-values `erfc(|z| / sqrt(2))` of every valid Gi* bin,
 *    adjusted by {@link GPUMultipleTestingAdjustment} over all valid bins (`multipleTesting`).
 * 4. Classification into the 17 {@link GPU_EMERGING_HOT_SPOTS_CATEGORIES} plus `UNDETERMINED`.
 *
 * Classification rules, evaluated on the finite z series (NaN slices skipped). A slice is hot when
 * its adjusted p-value is at most `1 - confidenceLevel` and `z > 0`, and cold with `z < 0`
 * (without a correction, `multipleTesting: 'none'`, hot is `z >= criticalZ` and cold
 * `z <= -criticalZ`, the same two-sided critical values). `N` is the finite slice count, `H` and
 * `C` the hot and cold counts, and "90%" is `persistentFraction * N`. The trend is significant
 * when `trendP <= trendSignificanceLevel`. The final slice is the last slice of the cube
 * (`finalSlice: 'require'`, default) or the last finite one (`'last-valid'`).
 * The final slice picks the track, then rules apply in order:
 * - Final slice hot: New (`H = 1`), Consecutive (trailing hot run of at least 2, `H` equal to the
 *   run, `H < 90%`), Intensifying / Diminishing / Persistent (`H >= 90%`, significant upward /
 *   significant downward / no significant trend), Sporadic (`C = 0`), otherwise Oscillating.
 * - Final slice cold: the mirror with Intensifying cold on a significant downward trend (the cold
 *   intensifies) and Diminishing cold on a significant upward trend.
 * - Final slice neither: Historical hot (`H >= 90%`), Historical cold (`C >= 90%`), Sporadic hot
 *   (`H > 0`, `C = 0`, as ArcGIS defines it without requiring a hot final slice), Sporadic cold
 *   (`C > 0`, `H = 0`), otherwise no pattern.
 * Ordering decisions: New is tested before Oscillating, so a cell hot only in the final slice is New
 * even with cold history; a cell with a cold final slice and `C = 1` is New cold even when 90% of
 * its earlier slices were hot; mixed hot and cold history without a significant final slice has
 * no pattern.
 *
 * Undetermined cells (`UNDETERMINED`, 17): with `finalSlice: 'require'`, a cell with at least one
 * finite slice whose last slice is missing; and a cell that is persistent in one direction
 * (the Intensifying / Persistent / Diminishing branches) but has fewer finite slices than
 * `minimumTrendSliceCount` (default 10, ESRI guidance), because its trend class would rest on a
 * Mann-Kendall test that is not supported by enough data. Such cells have NaN `trendZ` and
 * `trendP`. They are never silently reported as Persistent. Cells without any finite slice are
 * `NO_PATTERN`.
 *
 * Statistical caveats. Consecutive Gi* z values share `temporalWindow` slices, so the series is
 * serially correlated and the Mann-Kendall p-value is anti-conservative (ESRI has the same
 * property); `trendSignificanceLevel` is a screening threshold, not an error rate. Bins within a
 * temporal window overlap and neighboring cells share observations, so the per-bin tests are
 * dependent: Benjamini-Hochberg is only guaranteed under positive dependence, and
 * `'benjamini-yekutieli'` is the method that is valid under arbitrary dependence.
 *
 * Legend contract: category codes never change meaning with `multipleTesting`, `confidenceLevel`
 * or `finalSlice`; those options only change which bins count as hot or cold and which cells are
 * undetermined. Use {@link GPUEmergingHotSpots.getInferenceMetadata} to label the legend.
 *
 * Determinism: all sums run in a fixed order and integer statistics are exact, so repeated
 * encodings on one device are bitwise identical.
 *
 * Non-goals: permutation p-values, building the weights, and time-step intervals other than one
 * slice per cube column (aggregate to the interval before building the cube, as the space-time
 * hot spots recipe does).
 */
export class GPUEmergingHotSpots implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUEmergingHotSpotsProps;
  /** Resolved correction method (`'benjamini-hochberg'` when the prop is omitted). */
  readonly multipleTesting: GPUMultipleTestingMethod;
  /** Resolved final slice rule (`'require'` when the prop is omitted). */
  readonly finalSlice: 'require' | 'last-valid';

  constructor(props: GPUEmergingHotSpotsProps) {
    this.id = props.id ?? 'emerging-hot-spots';
    this.props = props;
    const id = this.id;
    const {output} = props;
    this.multipleTesting = resolveGPUMultipleTestingMethod(
      id,
      props.multipleTesting ?? 'benjamini-hochberg'
    );
    this.finalSlice = props.finalSlice ?? 'require';
    if (this.finalSlice !== 'require' && this.finalSlice !== 'last-valid') {
      throw new Error(`${id} finalSlice must be 'require' or 'last-valid'`);
    }
    const {gridWidth, gridHeight, sliceCount} = props;
    const hasLattice = gridWidth !== undefined || gridHeight !== undefined;
    if (hasLattice === (props.weights !== undefined)) {
      throw new Error(`${id} needs exactly one of gridWidth and gridHeight, or weights`);
    }
    const selfWeight = props.selfWeight ?? 1;
    if (!Number.isFinite(selfWeight) || selfWeight < 0) {
      throw new Error(`${id} selfWeight must be a finite number >= 0`);
    }
    if (hasLattice) {
      validateCapacity(id, 'gridWidth', gridWidth ?? Number.NaN);
      validateCapacity(id, 'gridHeight', gridHeight ?? Number.NaN);
    }
    validateCapacity(id, 'sliceCount', sliceCount, 1, GPU_EMERGING_HOT_SPOTS_MAXIMUM_SLICE_COUNT);
    const cellCount = props.weights
      ? validateGPUSpatialWeights(id, props.weights)
      : gridWidth! * gridHeight!;
    if (cellCount >= 2 ** 24) {
      throw new Error(`${id} must have fewer than 2^24 cells`);
    }
    const binCount = cellCount * sliceCount;
    if (binCount > MAXIMUM_BIN_COUNT) {
      throw new Error(`${id} cellCount * sliceCount must be below 2^31`);
    }
    validateCapacity(
      id,
      'maximumRadius',
      props.maximumRadius ?? 4,
      0,
      GPU_EMERGING_HOT_SPOTS_MAXIMUM_RADIUS
    );
    validatePackedView(props.values, ['float32', 'uint32'], `${id} values`);
    if (props.values.length !== binCount) {
      throw new Error(`${id} values length must equal cellCount * sliceCount`);
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH) {
      throw new Error(`${id} parameters is too short`);
    }
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
      if (props.mask.length !== cellCount) {
        throw new Error(`${id} mask length must equal the cell count`);
      }
    }
    validatePackedView(output.giZScores, ['float32'], `${id} giZScores`);
    if (output.giZScores.length !== binCount) {
      throw new Error(`${id} giZScores length must equal values length`);
    }
    for (const [name, view] of [
      ['pValues', output.pValues],
      ['adjustedPValues', output.adjustedPValues]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} ${name}`);
        if (view.length !== binCount) {
          throw new Error(`${id} ${name} length must equal values length`);
        }
      }
    }
    if (
      (this.multipleTesting !== 'none' || output.pValues || output.adjustedPValues) &&
      binCount >= GPU_MULTIPLE_TESTING_MAXIMUM_ROWS
    ) {
      // The adjustment ranks bins as exact f32 integers.
      throw new Error(`${id} multiple testing needs fewer than 2^24 bins`);
    }
    if (output.lastValidSlice) {
      validatePackedUint32View(output.lastValidSlice, `${id} lastValidSlice`);
      if (output.lastValidSlice.length !== cellCount) {
        throw new Error(`${id} lastValidSlice length must equal the cell count`);
      }
    }
    validatePackedView(output.trendZ, ['float32'], `${id} trendZ`);
    validatePackedView(output.trendP, ['float32'], `${id} trendP`);
    validatePackedView(output.trendS, ['sint32'], `${id} trendS`);
    validatePackedUint32View(output.category, `${id} category`);
    validatePackedUint32View(output.hotSliceCount, `${id} hotSliceCount`);
    validatePackedUint32View(output.coldSliceCount, `${id} coldSliceCount`);
    for (const [name, view] of [
      ['trendZ', output.trendZ],
      ['trendP', output.trendP],
      ['trendS', output.trendS],
      ['category', output.category],
      ['hotSliceCount', output.hotSliceCount],
      ['coldSliceCount', output.coldSliceCount]
    ] as const) {
      if (view.length !== cellCount) {
        throw new Error(`${id} ${name} length must equal the cell count`);
      }
    }
    if (output.globalStatistics) {
      validatePackedView(output.globalStatistics, ['float32'], `${id} globalStatistics`);
      if (output.globalStatistics.length < GPU_EMERGING_HOT_SPOTS_STATISTICS_LENGTH) {
        throw new Error(`${id} globalStatistics is too short`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.giZScores,
        output.pValues,
        output.adjustedPValues,
        output.lastValidSlice,
        output.trendZ,
        output.trendP,
        output.trendS,
        output.category,
        output.hotSliceCount,
        output.coldSliceCount,
        output.globalStatistics
      ],
      [
        props.values,
        props.parameters,
        props.mask,
        props.weights?.offsets,
        props.weights?.neighbors,
        props.weights?.weights
      ]
    );
  }

  /**
   * Packs `settings` for this node's `parameters` view, with `settings.multipleTesting` taken
   * from the node. Throws on invalid settings or a `multipleTesting` that differs from the node.
   *
   * @param settings The per-frame settings to pack (the same object `getInferenceMetadata` reads).
   * @param target Optional destination of at least `GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH` elements.
   */
  getParameterValues(settings: GPUEmergingHotSpotsSettings, target?: Float32Array): Float32Array {
    return getGPUEmergingHotSpotsParameterValues(this._resolveSettings(settings), target);
  }

  /**
   * CPU-side description of the inference behind the hot and cold bins, for building a legend,
   * derived from the same `settings` that {@link getParameterValues} packs
   * (`significanceLevel = 1 - confidenceLevel`). Validates `settings` like the packer.
   */
  getInferenceMetadata(settings: GPUEmergingHotSpotsSettings): GPUInferenceMetadata {
    const resolved = this._resolveSettings(settings);
    getGPUEmergingHotSpotsParameterValues(resolved);
    return {
      alternative: 'two-sided',
      permutationCount: 0,
      multipleTesting: this.multipleTesting,
      significanceLevel: 1 - (resolved.confidenceLevel ?? 0.95),
      testedRowsDefinition: 'valid space-time bins'
    };
  }

  private _resolveSettings(settings: GPUEmergingHotSpotsSettings): GPUEmergingHotSpotsSettings {
    if (
      settings.multipleTesting !== undefined &&
      settings.multipleTesting !== this.multipleTesting
    ) {
      throw new Error(`${this.id} settings.multipleTesting must match the node`);
    }
    return {...settings, multipleTesting: this.multipleTesting};
  }

  /** Returns the emerging-hot-spot nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {values, mask, parameters, weights, sliceCount, output} = props;
    const {giZScores} = output;
    const gridWidth = props.gridWidth ?? 1;
    const gridHeight = props.gridHeight ?? 1;
    const selfWeight = props.selfWeight ?? 1;
    validateGraphViewsBelongToGraph(id, graph, [
      values,
      mask,
      parameters,
      weights?.offsets,
      weights?.neighbors,
      weights?.weights,
      giZScores,
      output.trendZ,
      output.trendP,
      output.pValues,
      output.adjustedPValues,
      output.lastValidSlice,
      output.trendS,
      output.category,
      output.hotSliceCount,
      output.coldSliceCount,
      output.globalStatistics
    ]);
    const {finalSlice, multipleTesting} = this;
    // Without a correction the classifier compares z with the critical z and needs no p-values.
    const usesAdjustment = multipleTesting !== 'none';
    const needsPValues =
      usesAdjustment || output.pValues !== undefined || output.adjustedPValues !== undefined;
    const maximumRadius = props.maximumRadius ?? 4;
    const cellCount = weights ? weights.offsets.length - 1 : gridWidth * gridHeight;
    const binCount = cellCount * sliceCount;
    // Rule 16: every per-bin view is one storage binding, so the cube must fit the device's
    // binding size (2^25 four-byte bins at the 128 MiB default), not only the 2^31 index range.
    const maximumBindingBins = Math.floor(graph.device.limits.maxStorageBufferBindingSize / 4);
    if (binCount > maximumBindingBins) {
      throw new Error(`${id} cube exceeds maxStorageBufferBindingSize`);
    }
    const cooperativeMannKendall = sliceCount >= MANN_KENDALL_COOPERATIVE_SLICE_COUNT;
    const blockCount = Math.ceil(binCount / BIN_BLOCK);
    const valueType = values.format === 'uint32' ? 'u32' : 'f32';
    const sumPartials = createTransientView(graph, `${id}-sum-partials`, 'float32', blockCount);
    const countPartials = createTransientView(graph, `${id}-count-partials`, 'uint32', blockCount);
    const squarePartials = createTransientView(
      graph,
      `${id}-square-partials`,
      'float32',
      blockCount
    );
    const statistics =
      output.globalStatistics ??
      createTransientView(
        graph,
        `${id}-statistics`,
        'float32',
        GPU_EMERGING_HOT_SPOTS_STATISTICS_LENGTH
      );

    const declarations = /* wgsl */ `
const GRID_WIDTH: u32 = ${gridWidth}u;
const GRID_HEIGHT: u32 = ${gridHeight}u;
const CELL_COUNT: u32 = ${cellCount}u;
const SLICE_COUNT: u32 = ${sliceCount}u;
const BIN_COUNT: u32 = ${binCount}u;
const BIN_BLOCK: u32 = ${BIN_BLOCK}u;
const MAXIMUM_RADIUS: f32 = ${maximumRadius}.0;
const SELF_WEIGHT: f32 = ${getWGSLFloatLiteral(selfWeight)};
${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}

fn readValue(bin: u32) -> f32 {
  return f32(values[valuesOffset + bin]);
}

fn isBinValid(bin: u32, cell: u32) -> bool {
  ${mask ? 'if (mask[maskOffset + cell] == 0u) { return false; }' : ''}
  return isFiniteFloat(readValue(bin));
}
`;
    const valuesBinding: WGSLKernelBinding = {
      name: 'values',
      view: values,
      type: valueType,
      access: 'read'
    };
    const maskBindings: WGSLKernelBinding[] = mask
      ? [{name: 'mask', view: mask, type: 'u32', access: 'read'}]
      : [];
    const parametersBinding: WGSLKernelBinding = {
      name: 'parameters',
      view: parameters,
      type: 'f32',
      access: 'read'
    };
    const statisticsBinding = (access: 'read' | 'read_write'): WGSLKernelBinding => ({
      name: 'statistics',
      view: statistics,
      type: 'f32',
      access
    });
    const weightsBindings: WGSLKernelBinding[] = weights
      ? [
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
          {name: 'weights', view: weights.weights, type: 'f32', access: 'read'}
        ]
      : [];
    // Space-time Gi* by temporal factoring. The window sum over slices [first, slice] of the
    // neighbor sums equals the sum over those slices of per-slice ("instant") neighbor sums, so
    // each workgroup computes the instant sums of its cells once (one thread per (cell, slice),
    // consecutive threads reading consecutive slices) into workgroup memory and every bin then adds
    // at most `window + 1` instants. The work is n * T * (k + window) instead of the
    // n * T * k * window of re-walking the neighbors for every window slice.
    const cellsPerGroup = Math.max(1, Math.floor(GI_WORKGROUP_SIZE / sliceCount));
    const instantWGSL = weights
      ? `var weighted = 0.0;
    var weightTotal = 0.0;
    var squareTotal = 0.0;
    let bin = cell * SLICE_COUNT + slice;
    // The focal cell has weight SELF_WEIGHT; its CSR row never lists the cell itself.
    if (SELF_WEIGHT != 0.0 && isBinValid(bin, cell)) {
      weighted += SELF_WEIGHT * (readValue(bin) - mean);
      weightTotal += SELF_WEIGHT;
      squareTotal += SELF_WEIGHT * SELF_WEIGHT;
    }
    for (var slot = offsets[offsetsOffset + cell]; slot < offsets[offsetsOffset + cell + 1u]; slot++) {
      let neighborCell = neighbors[neighborsOffset + slot];
      if (neighborCell >= CELL_COUNT || neighborCell == cell) {
        continue;
      }
      let weight = weights[weightsOffset + slot];
      let neighborBin = neighborCell * SLICE_COUNT + slice;
      if (isBinValid(neighborBin, neighborCell)) {
        weighted += weight * (readValue(neighborBin) - mean);
        weightTotal += weight;
        squareTotal += weight * weight;
      }
    }
    instantWeighted[local] = weighted;
    instantWeight[local] = weightTotal;
    instantSquare[local] = squareTotal;`
      : `var weighted = 0.0;
    var neighborCount = 0u;
    let column = i32(cell % GRID_WIDTH);
    let row = i32(cell / GRID_WIDTH);
    for (var deltaRow = -reach; deltaRow <= reach; deltaRow++) {
      let neighborRow = row + deltaRow;
      if (neighborRow < 0 || neighborRow >= i32(GRID_HEIGHT)) {
        continue;
      }
      for (var deltaColumn = -reach; deltaColumn <= reach; deltaColumn++) {
        let neighborColumn = column + deltaColumn;
        if (neighborColumn < 0 || neighborColumn >= i32(GRID_WIDTH) ||
            f32(deltaRow * deltaRow + deltaColumn * deltaColumn) > radiusSquared) {
          continue;
        }
        let neighborCell = u32(neighborRow) * GRID_WIDTH + u32(neighborColumn);
        let neighborBin = neighborCell * SLICE_COUNT + slice;
        if (isBinValid(neighborBin, neighborCell)) {
          weighted += readValue(neighborBin) - mean;
          neighborCount++;
        }
      }
    }
    instantWeighted[local] = weighted;
    instantWeight[local] = f32(neighborCount);
    instantSquare[local] = 0.0;`;
    const windowSumWGSL = weights
      ? `var weightedSum = 0.0;
    var weightSum = 0.0;
    var squareSum = 0.0;
    for (var windowSlice = firstSlice; windowSlice <= slice; windowSlice++) {
      let slot = cellInGroup * SLICE_COUNT + windowSlice;
      weightedSum += instantWeighted[slot];
      weightSum += instantWeight[slot];
      squareSum += instantSquare[slot];
    }`
      : `var weightedSum = 0.0;
    var weightSum = 0.0;
    for (var windowSlice = firstSlice; windowSlice <= slice; windowSlice++) {
      let slot = cellInGroup * SLICE_COUNT + windowSlice;
      weightedSum += instantWeighted[slot];
      weightSum += instantWeight[slot];
    }
    let squareSum = weightSum;`;
    // Lattice weights are 1, so the squared-weight sum equals the weight sum; the weights path
    // keeps the squared-weight sum separately.
    const giBody = `let group = index / ${GI_WORKGROUP_SIZE}u;
  let local = localInvocationIndex;
  let cellInGroup = local / SLICE_COUNT;
  let slice = local % SLICE_COUNT;
  let cell = group * CELLS_PER_GROUP + cellInGroup;
  let isActive = index < INVOCATION_COUNT && cellInGroup < CELLS_PER_GROUP && cell < CELL_COUNT;
  let rawRadius = ${weights ? '0.0' : 'parameters[parametersOffset]'};
  let rawWindow = parameters[parametersOffset + 1u];
  let count = statistics[statisticsOffset];
  let mean = statistics[statisticsOffset + 1u];
  let variance = statistics[statisticsOffset + 2u];
  let deviation = statistics[statisticsOffset + 3u];
  let parametersValid = isFiniteFloat(rawRadius) && isFiniteFloat(rawWindow) && rawRadius >= 0.0 && rawWindow >= 0.0;
  let radius = min(rawRadius, MAXIMUM_RADIUS);
  let radiusSquared = radius * radius;
  let reach = i32(floor(radius));
  instantWeighted[local] = 0.0;
  instantWeight[local] = 0.0;
  instantSquare[local] = 0.0;
  if (isActive && parametersValid) {
    ${instantWGSL}
  }
  // Every invocation reaches the barrier.
  workgroupBarrier();
  if (isActive) {
    var zScore = getQuietNaN(index);
    if (parametersValid && isBinValid(cell * SLICE_COUNT + slice, cell)) {
      let windowSize = u32(min(floor(rawWindow), f32(SLICE_COUNT - 1u)));
      let firstSlice = slice - min(windowSize, slice);
      ${windowSumWGSL}
      ${
        weights
          ? 'let spread = (count * squareSum - weightSum * weightSum) / (count - 1.0);'
          : 'let spread = weightSum * (count - weightSum) / (count - 1.0);'
      }
      if (count >= 2.0 && variance > 0.0 && spread > 0.0 && isFiniteFloat(spread)) {
        zScore = weightedSum / (deviation * sqrt(spread));
      }
    }
    giZScores[giZScoresOffset + cell * SLICE_COUNT + slice] = zScore;
  }`;
    const pValues =
      output.pValues ??
      (needsPValues
        ? createTransientView(graph, `${id}-p-values`, 'float32', binCount)
        : undefined);
    const adjustedPValues =
      output.adjustedPValues ??
      (needsPValues
        ? createTransientView(graph, `${id}-adjusted-p-values`, 'float32', binCount)
        : undefined);
    // Two-sided normal p-value of every valid Gi* bin, then the correction over all valid bins.
    const getPValueNodes = (): GPUCommandNode<Parameters>[] => [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-p-values`,
        operation: OPERATION,
        variant: 'two-sided-p-values',
        bindings: [
          {name: 'giZScores', view: giZScores, type: 'f32', access: 'read'},
          {name: 'pValues', view: pValues!, type: 'f32', access: 'read_write'}
        ],
        invocationCount: binCount,
        declarations: `${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}
${ERFC_VALUE_WGSL}`,
        body: `let zScore = giZScores[giZScoresOffset + index];
  // erfc(|z| / sqrt(2)); NaN z (missing bin) stays NaN and is excluded from the family.
  pValues[pValuesOffset + index] = select(
    getQuietNaN(index),
    min(erfcValue(abs(zScore) * 0.7071067811865476), 1.0),
    isFiniteFloat(zScore)
  );`
      }),
      ...new GPUMultipleTestingAdjustment({
        id: `${id}-adjustment`,
        pValues: pValues!,
        method: multipleTesting,
        output: {adjustedPValues: adjustedPValues!}
      }).getCommandNodes(graph)
    ];
    const nodes: GPUCommandNode<Parameters>[] = [];

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-block-sums`,
        operation: OPERATION,
        variant: 'block-sums',
        bindings: [
          valuesBinding,
          ...maskBindings,
          {
            name: 'sumPartials',
            view: sumPartials,
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'countPartials',
            view: countPartials,
            type: 'u32',
            access: 'read_write'
          }
        ],
        // One workgroup per block: strided per-thread partials (coalesced reads) and a fixed tree.
        workgroupSize: MOMENT_WORKGROUP_SIZE,
        invocationCount: blockCount * MOMENT_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `${declarations}
var<workgroup> partialSums: array<f32, ${MOMENT_WORKGROUP_SIZE}>;
var<workgroup> partialCounts: array<u32, ${MOMENT_WORKGROUP_SIZE}>;`,
        // No early return: every invocation of a workgroup must reach the barriers.
        body: `let block = index / ${MOMENT_WORKGROUP_SIZE}u;
  let isInRange = index < INVOCATION_COUNT;
  var sum = 0.0;
  var count = 0u;
  if (isInRange) {
    let begin = block * BIN_BLOCK;
    let end = min(begin + BIN_BLOCK, BIN_COUNT);
    for (var bin = begin + localInvocationIndex; bin < end; bin += ${MOMENT_WORKGROUP_SIZE}u) {
      if (isBinValid(bin, bin / SLICE_COUNT)) {
        sum += readValue(bin);
        count++;
      }
    }
  }
  partialSums[localInvocationIndex] = sum;
  partialCounts[localInvocationIndex] = count;
  workgroupBarrier();
  for (var stride = ${MOMENT_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
      partialCounts[localInvocationIndex] += partialCounts[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (isInRange && localInvocationIndex == 0u) {
    sumPartials[sumPartialsOffset + block] = partialSums[0];
    countPartials[countPartialsOffset + block] = partialCounts[0];
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-mean`,
        operation: OPERATION,
        variant: 'mean',
        bindings: [
          {
            name: 'sumPartials',
            view: sumPartials,
            type: 'f32',
            access: 'read'
          },
          {
            name: 'countPartials',
            view: countPartials,
            type: 'u32',
            access: 'read'
          },
          statisticsBinding('read_write')
        ],
        invocationCount: 1,
        declarations: `const BLOCK_COUNT: u32 = ${blockCount}u;`,
        body: `var sum = 0.0;
  var count = 0u;
  for (var block = 0u; block < BLOCK_COUNT; block++) {
    sum += sumPartials[sumPartialsOffset + block];
    count += countPartials[countPartialsOffset + block];
  }
  statistics[statisticsOffset] = f32(count);
  statistics[statisticsOffset + 1u] = select(0.0, sum / f32(count), count > 0u);
  statistics[statisticsOffset + 2u] = 0.0;
  statistics[statisticsOffset + 3u] = 0.0;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-block-squares`,
        operation: OPERATION,
        variant: 'block-squares',
        bindings: [
          valuesBinding,
          ...maskBindings,
          statisticsBinding('read'),
          {
            name: 'squarePartials',
            view: squarePartials,
            type: 'f32',
            access: 'read_write'
          }
        ],
        workgroupSize: MOMENT_WORKGROUP_SIZE,
        invocationCount: blockCount * MOMENT_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `${declarations}
var<workgroup> partialSums: array<f32, ${MOMENT_WORKGROUP_SIZE}>;`,
        body: `let block = index / ${MOMENT_WORKGROUP_SIZE}u;
  let isInRange = index < INVOCATION_COUNT;
  let mean = statistics[statisticsOffset + 1u];
  var sum = 0.0;
  if (isInRange) {
    let begin = block * BIN_BLOCK;
    let end = min(begin + BIN_BLOCK, BIN_COUNT);
    for (var bin = begin + localInvocationIndex; bin < end; bin += ${MOMENT_WORKGROUP_SIZE}u) {
      if (isBinValid(bin, bin / SLICE_COUNT)) {
        let centered = readValue(bin) - mean;
        sum += centered * centered;
      }
    }
  }
  partialSums[localInvocationIndex] = sum;
  workgroupBarrier();
  for (var stride = ${MOMENT_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      partialSums[localInvocationIndex] += partialSums[localInvocationIndex + stride];
    }
    workgroupBarrier();
  }
  if (isInRange && localInvocationIndex == 0u) {
    squarePartials[squarePartialsOffset + block] = partialSums[0];
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-deviation`,
        operation: OPERATION,
        variant: 'deviation',
        bindings: [
          {
            name: 'squarePartials',
            view: squarePartials,
            type: 'f32',
            access: 'read'
          },
          statisticsBinding('read_write')
        ],
        invocationCount: 1,
        declarations: `const BLOCK_COUNT: u32 = ${blockCount}u;`,
        body: `var sum = 0.0;
  for (var block = 0u; block < BLOCK_COUNT; block++) {
    sum += squarePartials[squarePartialsOffset + block];
  }
  let count = statistics[statisticsOffset];
  let variance = select(0.0, sum / count, count > 0.0);
  statistics[statisticsOffset + 2u] = variance;
  statistics[statisticsOffset + 3u] = sqrt(variance);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-gi-star`,
        operation: OPERATION,
        variant: weights ? 'gi-star-weights' : 'gi-star',
        bindings: [
          valuesBinding,
          ...maskBindings,
          parametersBinding,
          ...weightsBindings,
          statisticsBinding('read'),
          {
            name: 'giZScores',
            view: giZScores,
            type: 'f32',
            access: 'read_write'
          }
        ],
        workgroupSize: GI_WORKGROUP_SIZE,
        invocationCount: Math.ceil(cellCount / cellsPerGroup) * GI_WORKGROUP_SIZE,
        guardIndex: false,
        declarations: `${declarations}
const CELLS_PER_GROUP: u32 = ${cellsPerGroup}u;
var<workgroup> instantWeighted: array<f32, ${GI_WORKGROUP_SIZE}>;
var<workgroup> instantWeight: array<f32, ${GI_WORKGROUP_SIZE}>;
var<workgroup> instantSquare: array<f32, ${GI_WORKGROUP_SIZE}>;`,
        body: giBody
      }),
      ...(needsPValues ? getPValueNodes() : []),
      // Hot and cold slice counts and the last finite slice, per cell. A separate pass keeps the
      // Mann-Kendall kernel within the storage buffer limit.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-slice-counts`,
        operation: OPERATION,
        variant: 'slice-counts',
        bindings: [
          parametersBinding,
          {name: 'giZScores', view: giZScores, type: 'f32', access: 'read'},
          ...(usesAdjustment
            ? [
                {
                  name: 'adjustedPValues',
                  view: adjustedPValues!,
                  type: 'f32',
                  access: 'read'
                } satisfies WGSLKernelBinding
              ]
            : []),
          {name: 'hotSliceCount', view: output.hotSliceCount, type: 'u32', access: 'read_write'},
          {name: 'coldSliceCount', view: output.coldSliceCount, type: 'u32', access: 'read_write'},
          ...(output.lastValidSlice
            ? [
                {
                  name: 'lastValidSlice',
                  view: output.lastValidSlice,
                  type: 'u32',
                  access: 'read_write'
                } satisfies WGSLKernelBinding
              ]
            : [])
        ],
        invocationCount: cellCount,
        declarations: `const SLICE_COUNT: u32 = ${sliceCount}u;
${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}
${getSliceStateWGSL(usesAdjustment)}`,
        body: `let base = index * SLICE_COUNT;
  ${getSliceCountingWGSL({
    valueExpression: 'giZScores[giZScoresOffset + base + slice]',
    adjustedExpression: 'adjustedPValues[adjustedPValuesOffset + base + slice]',
    usesAdjustment,
    collectSeries: false
  })}
  hotSliceCount[hotSliceCountOffset + index] = hot;
  coldSliceCount[coldSliceCountOffset + index] = cold;
  ${output.lastValidSlice ? 'lastValidSlice[lastValidSliceOffset + index] = lastValid;' : ''}`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-mann-kendall`,
        operation: OPERATION,
        variant: cooperativeMannKendall ? 'mann-kendall-cooperative' : 'mann-kendall-classify',
        bindings: [
          parametersBinding,
          {name: 'giZScores', view: giZScores, type: 'f32', access: 'read'},
          ...(usesAdjustment
            ? [
                {
                  name: 'adjustedPValues',
                  view: adjustedPValues!,
                  type: 'f32',
                  access: 'read'
                } satisfies WGSLKernelBinding
              ]
            : []),
          {
            name: 'trendZ',
            view: output.trendZ,
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'trendP',
            view: output.trendP,
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'trendS',
            view: output.trendS,
            type: 'i32',
            access: 'read_write'
          },
          {
            name: 'category',
            view: output.category,
            type: 'u32',
            access: 'read_write'
          }
        ],
        // One workgroup owns one cell. Pair comparisons are striped across lanes instead of one
        // invocation holding a 256-value private array and serially visiting every pair. Apart
        // from dividing the long dependent chain across 64 lanes, this avoids private-array
        // register pressure and the driver-dependent scratch-memory spill it can cause.
        workgroupSize: cooperativeMannKendall ? MANN_KENDALL_WORKGROUP_SIZE : undefined,
        invocationCount: cooperativeMannKendall
          ? cellCount * MANN_KENDALL_WORKGROUP_SIZE
          : cellCount,
        guardIndex: !cooperativeMannKendall,
        declarations: cooperativeMannKendall
          ? `const CELL_COUNT: u32 = ${cellCount}u;
const SLICE_COUNT: u32 = ${sliceCount}u;
const MANN_KENDALL_LANES: u32 = ${MANN_KENDALL_WORKGROUP_SIZE}u;
var<workgroup> partialStatistics: array<i32, ${MANN_KENDALL_WORKGROUP_SIZE}>;
var<workgroup> partialTieTerms: array<u32, ${MANN_KENDALL_WORKGROUP_SIZE}>;
var<workgroup> sharedSeries: array<f32, ${sliceCount}>;
var<workgroup> tieCandidates: array<u32, ${sliceCount}>;
var<workgroup> equalBeforeBits: array<atomic<u32>, ${Math.ceil(sliceCount / 32)}>;
${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}
${getSliceStateWGSL(usesAdjustment)}`
          : `const SLICE_COUNT: u32 = ${sliceCount}u;
${SPATIAL_AUTOCORRELATION_FLOAT_WGSL}
${getSliceStateWGSL(usesAdjustment)}`,
        body: cooperativeMannKendall
          ? `let cell = index / MANN_KENDALL_LANES;
  let lane = localInvocationIndex;
  let isActiveCell = cell < CELL_COUNT;
  let base = cell * SLICE_COUNT;
  for (var slice = lane; slice < SLICE_COUNT; slice += MANN_KENDALL_LANES) {
    var value = getQuietNaN(index);
    if (isActiveCell) {
      value = giZScores[giZScoresOffset + base + slice];
    }
    sharedSeries[slice] = value;
    tieCandidates[slice] = 0u;
  }
  for (var word = lane; word < ${Math.ceil(sliceCount / 32)}u; word += MANN_KENDALL_LANES) {
    atomicStore(&equalBeforeBits[word], 0u);
  }
  workgroupBarrier();
  var laneStatistic = 0;
  for (var first = lane; first < SLICE_COUNT; first += MANN_KENDALL_LANES) {
    let firstValue = sharedSeries[first];
    if (isFiniteFloat(firstValue)) {
      var groupSize = 1u;
      for (var second = first + 1u; second < SLICE_COUNT; second++) {
        let secondValue = sharedSeries[second];
        if (isFiniteFloat(secondValue)) {
          laneStatistic += select(select(0, -1, secondValue < firstValue), 1, secondValue > firstValue);
          if (secondValue == firstValue) {
            groupSize++;
            atomicOr(&equalBeforeBits[second / 32u], 1u << (second % 32u));
          }
        }
      }
      if (groupSize > 1u) {
        tieCandidates[first] = groupSize * (groupSize - 1u) * (2u * groupSize + 5u);
      }
    }
  }
  partialStatistics[lane] = laneStatistic;
  workgroupBarrier();
  var laneTieTerm = 0u;
  for (var first = lane; first < SLICE_COUNT; first += MANN_KENDALL_LANES) {
    let hasEqualBefore = (atomicLoad(&equalBeforeBits[first / 32u]) & (1u << (first % 32u))) != 0u;
    if (!hasEqualBefore) {
      laneTieTerm += tieCandidates[first];
    }
  }
  partialTieTerms[lane] = laneTieTerm;
  workgroupBarrier();
  for (var stride = MANN_KENDALL_LANES / 2u; stride > 0u; stride /= 2u) {
    if (lane < stride) {
      partialStatistics[lane] += partialStatistics[lane + stride];
      partialTieTerms[lane] += partialTieTerms[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u && isActiveCell) {
  let statistic = partialStatistics[0];
  let tieTerm = partialTieTerms[0];
  ${getSliceCountingWGSL({
    valueExpression: 'sharedSeries[slice]',
    adjustedExpression: 'adjustedPValues[adjustedPValuesOffset + base + slice]',
    usesAdjustment,
    collectSeries: false
  })}
  ${getClassificationWGSL({cellName: 'cell', finalSlice})}
  }`
          : getSerialMannKendallBody({sliceCount, usesAdjustment, finalSlice})
      })
    );
    return nodes;
  }
}

/**
 * WGSL `getSliceState`: 1 for a hot bin, 2 for a cold bin, 0 otherwise. Without a correction it
 * compares z with the critical z (the pre-correction behavior); with one, a bin is hot or cold
 * when its adjusted p-value is at most the significance level, with the sign of z.
 */
function getSliceStateWGSL(usesAdjustment: boolean): string {
  return usesAdjustment
    ? `fn getSliceState(z: f32, adjustedP: f32, level: f32) -> u32 {
  if (adjustedP <= level) {
    if (z > 0.0) {
      return 1u;
    }
    if (z < 0.0) {
      return 2u;
    }
  }
  return 0u;
}`
    : `fn getSliceState(z: f32, criticalZ: f32) -> u32 {
  if (z >= criticalZ) {
    return 1u;
  }
  if (z <= -criticalZ) {
    return 2u;
  }
  return 0u;
}`;
}

/**
 * WGSL pass over one cell's slices that declares `valid`, `hot`, `cold`, `trailingHot`,
 * `trailingCold`, `finalState` and `lastValid` (the last finite slice, `0xffffffff` for none), and
 * `series` when `collectSeries` is set. Reads the parameters it needs itself.
 */
function getSliceCountingWGSL(options: {
  valueExpression: string;
  adjustedExpression: string;
  usesAdjustment: boolean;
  collectSeries: boolean;
}): string {
  const state = options.usesAdjustment
    ? `getSliceState(value, ${options.adjustedExpression}, significanceLevel)`
    : 'getSliceState(value, criticalZ)';
  return /* wgsl */ `let criticalZ = parameters[parametersOffset + 2u];
  let significanceLevel = parameters[parametersOffset + 6u];
  var valid = 0u;
  var hot = 0u;
  var cold = 0u;
  var trailingHot = 0u;
  var trailingCold = 0u;
  var finalState = 0u;
  var lastValid = 0xffffffffu;
  for (var slice = 0u; slice < SLICE_COUNT; slice++) {
    let value = ${options.valueExpression};
    if (isFiniteFloat(value)) {
      ${options.collectSeries ? 'series[valid] = value;' : ''}
      valid++;
      lastValid = slice;
      let state = ${state};
      if (state == 1u) {
        hot++;
        trailingHot++;
        trailingCold = 0u;
        finalState = 1u;
      } else if (state == 2u) {
        cold++;
        trailingCold++;
        trailingHot = 0u;
        finalState = 2u;
      } else {
        trailingHot = 0u;
        trailingCold = 0u;
        finalState = 0u;
      }
    }
  }`;
}

/**
 * WGSL tail shared by both Mann-Kendall paths: from `statistic`, `tieTerm` and the slice counts it
 * writes the trend outputs, the slice counts and the category. See {@link GPUEmergingHotSpots} for
 * the rules; `UNDETERMINED` (17) replaces Intensifying/Persistent/Diminishing for cells with fewer
 * than `minimumTrendSliceCount` finite slices, and every pattern of a cell whose last slice is
 * missing when `finalSlice` is `'require'`.
 */
function getClassificationWGSL(options: {
  cellName: string;
  finalSlice: 'require' | 'last-valid';
}): string {
  const {cellName} = options;
  return /* wgsl */ `let trendLevel = parameters[parametersOffset + 3u];
  let fraction = parameters[parametersOffset + 4u];
  let minimumTrendSliceCount = parameters[parametersOffset + 5u];
  var numerator = 0u;
  if (valid >= 2u) {
    numerator = valid * (valid - 1u) * (2u * valid + 5u) - tieTerm;
  }
  var trendZScore = 0.0;
  var trendPValue = 1.0;
  if (numerator > 0u) {
    let deviation = sqrt(f32(numerator) / 18.0);
    if (statistic > 0) {
      trendZScore = f32(statistic - 1) / deviation;
    } else if (statistic < 0) {
      trendZScore = f32(statistic + 1) / deviation;
    }
    trendPValue = getTwoSidedPValue(trendZScore);
  }
  // Too few finite slices for a Mann-Kendall decision: no trend z or p, and no trend category.
  let isTrendKnown = f32(valid) >= minimumTrendSliceCount;
  if (!isTrendKnown) {
    trendZScore = getQuietNaN(${cellName});
    trendPValue = getQuietNaN(${cellName});
  }
  trendZ[trendZOffset + ${cellName}] = trendZScore;
  trendP[trendPOffset + ${cellName}] = trendPValue;
  trendS[trendSOffset + ${cellName}] = statistic;
  let threshold = fraction * f32(valid);
  let hotPersistent = f32(hot) >= threshold;
  let coldPersistent = f32(cold) >= threshold;
  let trendSignificant = trendPValue <= trendLevel;
  let trendUp = trendSignificant && trendZScore > 0.0;
  let trendDown = trendSignificant && trendZScore < 0.0;
  var result = 0u;
  if (valid > 0u) {
    if (finalState == 1u) {
      if (hot == 1u) {
        result = 1u;
      } else if (trailingHot >= 2u && trailingHot == hot && !hotPersistent) {
        result = 2u;
      } else if (hotPersistent) {
        result = select(17u, select(select(4u, 5u, trendDown), 3u, trendUp), isTrendKnown);
      } else {
        result = select(7u, 6u, cold == 0u);
      }
    } else if (finalState == 2u) {
      if (cold == 1u) {
        result = 9u;
      } else if (trailingCold >= 2u && trailingCold == cold && !coldPersistent) {
        result = 10u;
      } else if (coldPersistent) {
        result = select(17u, select(select(12u, 13u, trendUp), 11u, trendDown), isTrendKnown);
      } else {
        result = select(15u, 14u, hot == 0u);
      }
    } else if (hot > 0u && hotPersistent) {
      result = 8u;
    } else if (cold > 0u && coldPersistent) {
      result = 16u;
    } else if (hot > 0u && cold == 0u) {
      result = 6u;
    } else if (cold > 0u && hot == 0u) {
      result = 14u;
    }
    ${
      options.finalSlice === 'require'
        ? `// A missing final slice makes the pattern undecidable instead of silently shifting it.
    if (lastValid != SLICE_COUNT - 1u) {
      result = 17u;
    }`
        : ''
    }
  }
  category[categoryOffset + ${cellName}] = result;`;
}

/** Serial exact pass retained for short series where cooperative setup would dominate. */
function getSerialMannKendallBody(options: {
  sliceCount: number;
  usesAdjustment: boolean;
  finalSlice: 'require' | 'last-valid';
}): string {
  const {sliceCount} = options;
  return /* wgsl */ `let base = index * SLICE_COUNT;
  var series: array<f32, SLICE_COUNT>;
  var hasEqualBefore: array<u32, ${Math.ceil(sliceCount / 32)}>;
  ${getSliceCountingWGSL({
    valueExpression: 'giZScores[giZScoresOffset + base + slice]',
    adjustedExpression: 'adjustedPValues[adjustedPValuesOffset + base + slice]',
    usesAdjustment: options.usesAdjustment,
    collectSeries: true
  })}
  var statistic = 0;
  var tieTerm = 0u;
  for (var first = 0u; first < valid; first++) {
    let firstValue = series[first];
    var groupSize = 1u;
    for (var second = first + 1u; second < valid; second++) {
      let secondValue = series[second];
      statistic += select(select(0, -1, secondValue < firstValue), 1, secondValue > firstValue);
      if (secondValue == firstValue) {
        groupSize++;
        hasEqualBefore[second / 32u] |= 1u << (second % 32u);
      }
    }
    let isFirstMember = (hasEqualBefore[first / 32u] & (1u << (first % 32u))) == 0u;
    if (isFirstMember && groupSize > 1u) {
      tieTerm += groupSize * (groupSize - 1u) * (2u * groupSize + 5u);
    }
  }
  ${getClassificationWGSL({
    cellName: 'index',
    finalSlice: options.finalSlice
  })}`;
}
