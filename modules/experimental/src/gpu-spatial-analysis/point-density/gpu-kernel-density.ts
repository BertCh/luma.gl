// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUGridIndex,
  GPUReduction,
  type GPUCommandGraph,
  type GPUCommandGraphNodeCondition,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphBufferUse,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {
  getGPUGridIndexMaximumCellCount,
  validateCapacity,
  validateMaximumCount,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/index';
import {
  resolveGPUStudyWindow,
  type GPUStudyWindow
} from '../../analysis-core/contracts/study-window';
import {getStudyWindowBinding} from '../../utils/study-window-wgsl';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../utils/wgsl-common';
import {validateRasterGrid} from '../../gpu-raster/raster-band-utils';
import type {GPURasterMetadata} from '../../gpu-raster/types';
import {getRasterGridCorners} from '../spatial-interpolation/spatial-interpolation-parameters';
import {getRectangleScanWGSL} from '../spatial-interpolation/grid-neighbor-scan';
import {GRID_COORDINATE_WGSL} from '../spatial-interpolation/gpu-inverse-distance-weighting';
import {
  GPU_KERNEL_DENSITY_GAUSSIAN_TRUNCATION,
  type GPUKernelDensityBandwidthRule,
  type GPUKernelDensityKernel,
  type GPUKernelDensityMethod,
  type GPUKernelDensityWindow
} from './kernel-density-math';
import {
  createGPUKernelDensityBinnedNodes,
  getGPUKernelDensityBinnedSize,
  getGPUKernelDensityDefaultBinnedPadding,
  GPU_KERNEL_DENSITY_BINNED_MAXIMUM_CELL_COUNT
} from './kernel-density-binned';
import {
  getKernelDensityResultWGSL,
  getKernelDensityStudyWindowWGSL,
  getKernelValueWGSL,
  getWindowMassWGSL
} from './kernel-density-wgsl';

/** Spatial contexts supported by this area-measuring operation. */
const SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  measure: 'area'
} as const);

const OPERATION = 'GPUKernelDensity';

/** WGSL constants `DOMAIN_MIN` and `DOMAIN_MAX` of the index domain. */
function getDomainConstantsWGSL(indexBounds: readonly [number, number, number, number]): string {
  return `const DOMAIN_MIN: vec2f = vec2f(${getWGSLFloatLiteral(indexBounds[0])}, ${getWGSLFloatLiteral(indexBounds[1])});
const DOMAIN_MAX: vec2f = vec2f(${getWGSLFloatLiteral(indexBounds[2])}, ${getWGSLFloatLiteral(indexBounds[3])});`;
}

/** Number of float32 elements in a {@link GPUKernelDensity} parameter view. */
export const GPU_KERNEL_DENSITY_PARAMETER_LENGTH = 9;

/** Edge treatment at the boundary of the study window. See {@link GPUKernelDensityProps.edgeCorrection}. */
export type GPUKernelDensityEdgeCorrection = 'none' | 'uniform' | 'diggle';

/** Output scaling. See {@link GPUKernelDensityProps.normalization}. */
export type GPUKernelDensityNormalization = 'intensity' | 'probability';

/** Per-frame settings of {@link GPUKernelDensity}. */
export type GPUKernelDensitySettings = {
  /**
   * Kernel bandwidth `h` in position units (ground units). Positive and finite. Ignored (the GPU
   * computes it) when {@link GPUKernelDensityProps.bandwidthRule} is set; pass any positive value.
   */
  bandwidth: number;
  /**
   * Study window `[minX, minY, maxX, maxY]` of the edge correction. Defaults to the ordered rectangle of the output grid. Ignored
   * without edge correction. Use `'study-window'` with a polygon `studyWindow`; it writes NaN
   * rectangle slots, so a rectangle-mode operator with edge correction yields NaN cells.
   */
  window?: GPUKernelDensityWindow | 'study-window';
};

/**
 * Packs per-frame {@link GPUKernelDensity} parameters.
 *
 * Layout (float32): `[startX, startY, endX, endY, bandwidth, windowMinX, windowMinY, windowMaxX,
 * windowMaxY]`. The first four values are the outer corners of the output grid in grid order:
 * `(startX, startY)` is the outer corner of cell `(0, 0)` and `(endX, endY)` the opposite corner,
 * so a north-up grid (`grid.affine[4] < 0`) has `endY < startY` and row 0 at the top, and a grid
 * whose rows advance north has row 0 at `minY`. The window defaults to the ordered rectangle
 * `[min(startX, endX), min(startY, endY), max(startX, endX), max(startY, endY)]`.
 *
 * @param settings Bandwidth and optional window.
 * @param grid The output grid: axis-aligned, with a nonzero finite cell size; its width and height
 *   must match the contributor's.
 * @param target Optional destination of at least {@link GPU_KERNEL_DENSITY_PARAMETER_LENGTH} elements.
 * @throws If the grid is rotated, sheared or invalid, the target is too short, the bandwidth is
 * not positive and finite, the cell size is zero, or the window is not finite and ordered.
 */
export function getGPUKernelDensityParameterValues(
  settings: GPUKernelDensitySettings,
  grid: GPURasterMetadata,
  target: Float32Array = new Float32Array(GPU_KERNEL_DENSITY_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_KERNEL_DENSITY_PARAMETER_LENGTH) {
    throw new Error(
      `Kernel density parameter target must hold ${GPU_KERNEL_DENSITY_PARAMETER_LENGTH} elements`
    );
  }
  if (!Number.isFinite(settings.bandwidth) || settings.bandwidth <= 0) {
    throw new Error('Kernel density bandwidth must be positive and finite');
  }
  const corners = getRasterGridCorners('getGPUKernelDensityParameterValues', grid);
  const [startX, startY, endX, endY] = corners;
  if (!corners.every(Number.isFinite) || startX === endX || startY === endY) {
    throw new Error('Kernel density grid must have a finite, nonzero cell size');
  }
  const window = settings.window ?? [
    Math.min(startX, endX),
    Math.min(startY, endY),
    Math.max(startX, endX),
    Math.max(startY, endY)
  ];
  if (
    window !== 'study-window' &&
    (window.length !== 4 ||
      !window.every(Number.isFinite) ||
      window[0] >= window[2] ||
      window[1] >= window[3])
  ) {
    throw new Error('Kernel density window must be finite [minX, minY, maxX, maxY]');
  }
  target.set(corners, 0);
  target[4] = settings.bandwidth;
  if (window === 'study-window') {
    target.fill(Number.NaN, 5, 9);
  } else {
    target.set(window, 5);
  }
  return target;
}

/** Caller-owned outputs of {@link GPUKernelDensity}. */
export type GPUKernelDensityOutput = {
  /**
   * Density per output cell, at least `grid.width * grid.height` float32 rows, row
   * `y * grid.width + x` in the grid's row order (row 0 is the grid's first row: the top row of a
   * north-up grid, `minY` when rows advance north). Units: `'intensity'` is the sum of weights per unit area, position units to the
   * power -2 (points per square meter for a meter frame); `'probability'` is the same divided by
   * the sum of included weights, a probability density that integrates to about one over the
   * window. NaN where the cell center is outside the edge-correction window, the grid or
   * bandwidth is invalid, or (probability) no weight is included.
   */
  values: GraphDataView<'float32'>;
  /**
   * Optional one-row float32 sum of the included weights (the point count without `weights`),
   * before edge correction: the divisor of `'probability'`.
   */
  weightSum?: GraphDataView<'float32'>;
  /**
   * Optional one-row float32 holding the bandwidth in position units that the density kernels
   * read. Only with {@link GPUKernelDensityProps.bandwidthRule}, which writes it on the GPU each
   * encoding (NaN when the rule is undefined, for example fewer than two included rows, an
   * effective sample size of one or less, or zero spread, in which case every cell is NaN).
   */
  bandwidth?: GraphDataView<'float32'>;
  /** Optional diagnostic scalars. */
  status?: {
    /**
     * Optional one-row uint32 count of input rows that were excluded: a non-finite position or
     * weight, a negative weight, a position outside `indexBounds`, or (with edge correction)
     * outside the window. Zero-weight rows are included.
     */
    invalidCount?: GraphDataView<'uint32'>;
    /**
     * Optional one-row uint32 for `method: 'binned'` and `'auto'`: 1 when `'binned'` ran although
     * `binnedPadding` is smaller than the margin of `2.5 h / cell` cells the blur needs (the
     * estimate then drops near the grid edges), 0 otherwise. Always 0 for `'auto'`, which falls
     * back to the exact gather instead.
     */
    insufficientPadding?: GraphDataView<'uint32'>;
  };
};

/**
 * Properties for {@link GPUKernelDensity}.
 *
 * Per-frame (no recompile): the contents of `parameters` (bandwidth, window) and of every
 * input buffer. Topology (needs a new graph): the `grid` size, point count, `indexGridSize`,
 * `indexBounds`, `kernel`, `edgeCorrection`, `normalization`, and which optional inputs and outputs
 * are present.
 */
export type GPUKernelDensityProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'kernel-density'`. */
  id?: string;
  /**
   * Coordinate and metric contract of positions, grid and bandwidth. Planar/native only
   * (omitted means planar/native): the bandwidth is a ground distance, so positions must be in a
   * planar frame with the same unit on both axes. Longitude/latitude throws at construction.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Compile-time study window, default `'rectangle'` (the per-frame parameter rectangle).
   * A prepared polygon must be valid, share this operator's planar frame, and be added to the
   * graph first. Requires `'uniform'` or `'diggle'` edge correction. Window queries use absolute
   * frame coordinates; outside points are excluded and outside cell centers produce NaN.
   */
  studyWindow?: GPUStudyWindow;
  /** Packed point positions in the same units as the grid and bandwidth. */
  positions: GraphDataView<'float32x2'>;
  /**
   * Optional per-point weights, for example populations or counts. Non-finite and negative
   * weights exclude the point. Without weights every point has weight one.
   */
  weights?: GraphDataView<'float32'>;
  /** Per-frame float32 view of at least 9 values written with {@link getGPUKernelDensityParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /**
   * Output raster grid: axis-aligned, its width and height size the kernels, and its cell centers
   * are evaluated. Pack the matching placement with `getGPUKernelDensityParameterValues`.
   */
  grid: GPURasterMetadata;
  /**
   * Dimensions `[columns, rows]` of the internal `GPUGridIndex` over the points. Aim for a few
   * points per index cell, and index cells not much larger than the bandwidth. At most
   * {@link getGPUGridIndexMaximumCellCount} cells (2^25 - 1 on a default device).
   */
  indexGridSize: readonly [number, number];
  /**
   * Inclusive point domain `[minX, minY, maxX, maxY]` of the internal grid index. It must cover the
   * output grid and every point you want counted: points outside it are ignored (counted in
   * `output.status.invalidCount`), so a point just outside the domain stops contributing to cells within one
   * support radius of it.
   */
  indexBounds: readonly [number, number, number, number];
  /**
   * Computes the bandwidth on the GPU every encoding instead of reading `parameters` slot 4, so
   * no read-back is needed. `'scott'` and `'silverman'` are the normal-reference rules of
   * `scipy.stats.gaussian_kde`; in two dimensions both give the factor `n_eff^(-1/6)` and so the
   * same value: `h = n_eff^(-1/6) * sqrt((varianceX + varianceY) / 2)`, with the unbiased weighted
   * covariance (`numpy.cov(aweights = w)`) and `n_eff = (sum w)^2 / sum w^2`. Rows with a
   * non-finite position, or a non-finite or negative weight, are skipped (the index domain and
   * the study window are not applied). A normal-reference rule oversmooths clustered data. The
   * moments use a two-stage hierarchical reduction (centered, relative to the grid start corner).
   * Topology: changing it needs a new graph.
   */
  bandwidthRule?: GPUKernelDensityBandwidthRule;
  /** Kernel, `K_h(d) = K(d / h) / h^2`, normalised to integrate to one. Defaults to `'gaussian'`. */
  kernel?: GPUKernelDensityKernel;
  /**
   * Treatment of the study window boundary, where an uncorrected estimate loses the kernel mass
   * that falls outside the data region and underestimates the intensity.
   * - `'uniform'` (default): divide each cell by `e(u)`, the kernel mass inside the window
   *   (Diggle 1985; spatstat `density.ppp(edge = TRUE, diggle = FALSE)`). Unbiased in expectation
   *   for a locally uniform intensity, deterministic, and cheap. It is the default because the
   *   uncorrected estimate is systematically low within one bandwidth of the window edge.
   * - `'diggle'`: weight each point by `1 / e(x_i)` (Diggle 1985; Jones 1993, `diggle = TRUE`):
   *   preserves the point total better but is noisier near the edge.
   * - `'none'`: the plain kernel sum; the window is ignored.
   *
   * With `'uniform'` or `'diggle'` the window is the polygon `studyWindow` when supplied,
   * otherwise `parameters.window` (the grid rectangle by default),
   * points outside it are excluded, and cells whose center is outside it are NaN.
   */
  edgeCorrection?: GPUKernelDensityEdgeCorrection;
  /**
   * `'intensity'` (default): `sum_i w_i K_h(u - x_i)`, weights per unit area. `'probability'`:
   * that divided by the sum of included weights.
   */
  normalization?: GPUKernelDensityNormalization;
  /**
   * How the sum is evaluated. Topology: changing it needs a new graph.
   * - `'exact'` (default): every output cell gathers the points inside the kernel support (the
   *   gaussian is truncated at 5 bandwidths). Cost grows with the number of cells inside the
   *   support, `(5 h / cell)^2` per output cell times the local point density.
   * - `'binned'` (gaussian kernel only): each included weight is split over the four nearest cell
   *   centers (linear, or cloud-in-cell, binning, Wand 1994) with the deterministic fixed-point
   *   accumulator, and the bins are blurred along x then y by an almost-gaussian filter: three
   *   running-sum box blurs and one two-sided exponential pass per axis, variance-matched to
   *   `(h / cell)^2` (Wells 1986; Kovesi 2010, with the exponential cancelling the boxes' fourth
   *   cumulant). The cost is a few passes over the grid whatever the bandwidth and does not depend
   *   on the number of points beyond the binning. Result error versus `'exact'`, for
   *   `h >= 3` cells, points inside the binned region and a margin of `2.5 h / cell` cells
   *   (`binnedPadding`), is at most
   *   `getGPUKernelDensityBinnedErrorBound(r) * W / (2 pi h^2)` per cell, where `W` is the sum of
   *   the included weights and `r = max(cellX, cellY) / h`: `r^2 / 4` from the linear binning, 0.016
   *   from the filter shape, and a float32 term, 2e-4 or less up to `h = 200` cells; for example
   *   0.0185 at `h = 10` cells and 0.0161 at `h = 64` cells, that is under 2% of the peak of one
   *   point's kernel. Smoothly varying fields err far less than a lone point. Other kernels throw.
   * - `'auto'`: both paths are built and the GPU chooses every encoding from the bandwidth: exact
   *   while the gaussian support radius `5 h` is at most 32 cells on the larger axis (the
   *   exact gather visits that many cells squared per point), binned above it, but only when
   *   `binnedPadding` gives the blur its margin of `2.5 h / cell` cells (otherwise the exact
   *   gather stays correct, only slower). A non-gaussian kernel resolves to `'exact'`. Both paths'
   *   buffers are allocated.
   */
  method?: GPUKernelDensityMethod;
  /**
   * Cells added on every side of the output grid to the binned array (plus one guard cell), in
   * cells. It sets two things: the blur needs a margin of `2.5 h / cell` cells around the cells you
   * read, because each pass of the cascade drops the mass that leaves the array, so the binned
   * path is exact-to-tolerance only for `h <= binnedPadding * cell / 2.5` (`'binned'` reports a
   * shortfall in `output.status.insufficientPadding`; `'auto'` uses the exact gather instead); and
   * points up to `binnedPadding + 0.5` cells outside the grid are counted (points beyond are not,
   * though they still count in `weightSum` and the exact path), which matters with a `window`
   * larger than the grid or `edgeCorrection: 'none'`. The array holds
   * `(width + 2 binnedPadding + 2) * (height + 2 binnedPadding + 2)` cells of 20 bytes, bound as
   * one storage buffer of 12 bytes per cell. Defaults to the larger grid side (bandwidths up to
   * 40% of it), reduced to fit the buffer. Ignored by `'exact'`.
   */
  binnedPadding?: number;
  /** Caller-owned outputs. */
  output: GPUKernelDensityOutput;
};

/**
 * Planar kernel density estimate on a raster, with the bandwidth in ground units.
 *
 * For each output cell center `u` it computes `lambda(u) = sum_i w_i K_h(|u - x_i|)` over the
 * points within the kernel support, where `K_h(d) = K(d / h) / h^2` is a radially symmetric kernel
 * normalised to integrate to one over the plane: `'gaussian'` `exp(-t / 2) / (2 pi h^2)` truncated
 * at 5 bandwidths (lost mass `exp(-12.5)`, 3.7e-6), `'quartic'` (biweight) `3 / pi (1 - t)^2 / h^2`
 * and `'epanechnikov'` `2 / pi (1 - t) / h^2`, with `t = d^2 / h^2 < 1` (Silverman 1986, Density
 * Estimation, Table 3.1). The output is therefore an intensity in weight per unit area (not a count
 * per cell), unchanged by the raster resolution, and `normalization: 'probability'` divides by the
 * included weight to give a probability density. Compare {@link GPUPointDensity}, which bins points
 * and smooths the bin counts with a kernel measured in cells.
 *
 * Edge correction (`edgeCorrection`) divides by, or reweights with, the kernel mass `e(u)` inside
 * the study window. For rectangles the mass is closed form: normal-CDF differences for the
 * gaussian, and an exact inclusion-exclusion of four disk-clipped quadrant integrals for the
 * compact kernels (no quadrature). A prepared polygon `studyWindow` uses an oriented fan
 * decomposition of the kernel integral (2e-5 relative accuracy). It must be valid and in the
 * same planar frame as the KDE; window queries use absolute frame coordinates.
 *
 * Implementation: every encoding rebuilds a `GPUGridIndex` over the points, ranks the IDs inside
 * each index cell ascending, computes per-point weights (exclusion, and `w_i / e(x_i)` for
 * `'diggle'`), and gathers one output cell per invocation by visiting the index rows overlapping the
 * support square in a fixed order (index row, index column, ascending point row). No float atomics,
 * so results are deterministic for a device. Distances are formed relative to the grid start corner to
 * limit f32 cancellation far from the origin, but the input positions themselves are f32.
 *
 * Large bandwidths: the gather costs `(5 h / cell)^2` cells per point, which explodes when the
 * bandwidth spans many cells. `method: 'binned'` (or `'auto'`) replaces it with linear binning and
 * a separable almost-gaussian blur whose cost does not depend on the bandwidth, at the documented
 * error (see {@link GPUKernelDensityProps.method}). Edge correction, normalization and the NaN
 * rules are unchanged.
 *
 * Bandwidth: choose it in ground units for the question being asked, or set `bandwidthRule` to
 * have the GPU compute Scott's/Silverman's rule of thumb from the points with no read-back.
 *
 * Output cell `(x, y)` is evaluated at its center `gridStart + (x + 0.5, y + 0.5) * cellSize`,
 * where `gridStart` is the outer corner of cell (0, 0) and `cellSize` is signed, so output rows
 * follow the grid's row order (row 0 at the top of a north-up grid, at `minY` when rows advance
 * north). The estimate is a cell-center sample, not a cell average.
 */
export class GPUKernelDensity implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUKernelDensityProps;
  /** Resolved coordinate and metric contract (planar/native). */
  readonly spatialContext: GPUSpatialContext;
  /** Resolved compile-time study window. */
  readonly studyWindow: GPUStudyWindow;
  /** Resolved kernel. */
  readonly kernel: GPUKernelDensityKernel;
  /** Resolved edge correction. */
  readonly edgeCorrection: GPUKernelDensityEdgeCorrection;
  /** Resolved output scaling. */
  readonly normalization: GPUKernelDensityNormalization;
  /** Resolved method: `'auto'` with a non-gaussian kernel resolves to `'exact'`. */
  readonly method: GPUKernelDensityMethod;
  /** Resolved `binnedPadding`. */
  readonly binnedPadding: number;
  /** `grid.width * grid.height`. */
  readonly cellCount: number;

  constructor(props: GPUKernelDensityProps) {
    this.id = props.id ?? 'kernel-density';
    this.props = props;
    const {id} = this;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      SPATIAL_CONTEXT_SUPPORT
    );
    this.studyWindow = resolveGPUStudyWindow(id, props.studyWindow, this.spatialContext);
    this.kernel = props.kernel ?? 'gaussian';
    this.edgeCorrection = props.edgeCorrection ?? 'uniform';
    if (this.studyWindow !== 'rectangle' && this.edgeCorrection === 'none') {
      throw new Error(
        `${id} studyWindow is ignored without edge correction; set edgeCorrection to 'uniform' or 'diggle'`
      );
    }
    this.normalization = props.normalization ?? 'intensity';
    this.binnedPadding =
      props.binnedPadding ??
      getGPUKernelDensityDefaultBinnedPadding(props.grid.width, props.grid.height);
    const requestedMethod = props.method ?? 'exact';
    if (!['gaussian', 'quartic', 'epanechnikov'].includes(this.kernel)) {
      throw new Error(`${id} kernel must be gaussian, quartic, or epanechnikov`);
    }
    if (!['none', 'uniform', 'diggle'].includes(this.edgeCorrection)) {
      throw new Error(`${id} edgeCorrection must be none, uniform, or diggle`);
    }
    if (!['intensity', 'probability'].includes(this.normalization)) {
      throw new Error(`${id} normalization must be intensity or probability`);
    }
    if (!['exact', 'binned', 'auto'].includes(requestedMethod)) {
      throw new Error(`${id} method must be exact, binned, or auto`);
    }
    if (requestedMethod === 'binned' && this.kernel !== 'gaussian') {
      // The filter cascade approximates a gaussian, so it cannot stand in for the other kernels.
      throw new Error(`${id} method binned needs the gaussian kernel`);
    }
    if (!Number.isSafeInteger(this.binnedPadding) || this.binnedPadding < 0) {
      throw new Error(`${id} binnedPadding must be a nonnegative integer`);
    }
    this.method =
      requestedMethod === 'auto' && this.kernel !== 'gaussian' ? 'exact' : requestedMethod;
    if (
      props.bandwidthRule !== undefined &&
      !['scott', 'silverman'].includes(props.bandwidthRule)
    ) {
      throw new Error(`${id} bandwidthRule must be scott or silverman`);
    }
    if (props.output.bandwidth && !props.bandwidthRule) {
      throw new Error(`${id} output.bandwidth needs bandwidthRule`);
    }
    this.cellCount = validateRasterGrid(id, props.grid);
    if (props.grid.affine[1] !== 0 || props.grid.affine[3] !== 0) {
      throw new Error(`${id} needs an axis-aligned grid`);
    }
    if (this.method !== 'exact') {
      const {paddedWidth, paddedHeight} = getGPUKernelDensityBinnedSize(
        props.grid.width,
        props.grid.height,
        this.binnedPadding
      );
      if (paddedWidth * paddedHeight > GPU_KERNEL_DENSITY_BINNED_MAXIMUM_CELL_COUNT) {
        throw new Error(
          `${id} binned grid of ${paddedWidth} by ${paddedHeight} cells is too large`
        );
      }
    }
    for (const [name, size] of [
      ['indexGridSize[0]', props.indexGridSize[0]],
      ['indexGridSize[1]', props.indexGridSize[1]]
    ] as const) {
      validateCapacity(id, name, size, 1, Number.MAX_SAFE_INTEGER);
    }
    const [minX, minY, maxX, maxY] = props.indexBounds;
    if (
      props.indexBounds.length !== 4 ||
      !props.indexBounds.every(Number.isFinite) ||
      minX > maxX ||
      minY > maxY
    ) {
      throw new Error(`${id} indexBounds must be finite and ordered`);
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    if (props.positions.length < 1) {
      throw new Error(`${id} needs at least one point`);
    }
    if (props.weights) {
      validatePackedView(props.weights, ['float32'], `${id} weights`);
      if (props.weights.length !== props.positions.length) {
        throw new Error(`${id} weights length must equal positions length`);
      }
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    if (props.parameters.length < GPU_KERNEL_DENSITY_PARAMETER_LENGTH) {
      throw new Error(
        `${id} parameters must hold ${GPU_KERNEL_DENSITY_PARAMETER_LENGTH} float32 values`
      );
    }
    validatePackedView(props.output.values, ['float32'], `${id} output.values`);
    if (props.output.values.length < this.cellCount) {
      throw new Error(`${id} output.values must hold grid.width * grid.height rows`);
    }
    if (props.output.weightSum) {
      validatePackedView(props.output.weightSum, ['float32'], `${id} output.weightSum`);
      if (props.output.weightSum.length !== 1) {
        throw new Error(`${id} output.weightSum must contain exactly one float32 row`);
      }
    }
    if (props.output.status?.invalidCount) {
      validatePackedView(
        props.output.status.invalidCount,
        ['uint32'],
        `${id} output.status.invalidCount`
      );
      if (props.output.status.invalidCount.length !== 1) {
        throw new Error(`${id} output.status.invalidCount must contain exactly one uint32 row`);
      }
    }
    if (props.output.status?.insufficientPadding) {
      if (this.method === 'exact') {
        throw new Error(`${id} output.status.insufficientPadding needs method binned or auto`);
      }
      validatePackedView(
        props.output.status.insufficientPadding,
        ['uint32'],
        `${id} output.status.insufficientPadding`
      );
      if (props.output.status.insufficientPadding.length !== 1) {
        throw new Error(
          `${id} output.status.insufficientPadding must contain exactly one uint32 row`
        );
      }
    }
    if (props.output.bandwidth) {
      validatePackedView(props.output.bandwidth, ['float32'], `${id} output.bandwidth`);
      if (props.output.bandwidth.length !== 1) {
        throw new Error(`${id} output.bandwidth must contain exactly one float32 row`);
      }
    }
    const outputBuffers = [
      props.output.values,
      props.output.weightSum,
      props.output.status?.invalidCount,
      props.output.status?.insufficientPadding,
      props.output.bandwidth
    ]
      .filter(view => view !== undefined)
      .map(view => view.buffer);
    if (new Set(outputBuffers).size !== outputBuffers.length) {
      throw new Error(`${id} outputs must use separate buffers`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        props.output.values,
        props.output.weightSum,
        props.output.status?.invalidCount,
        props.output.status?.insufficientPadding,
        props.output.bandwidth
      ],
      [props.positions, props.weights, props.parameters]
    );
  }

  /**
   * Two-stage moments for {@link GPUKernelDensityProps.bandwidthRule}: per-row terms, hierarchical
   * `GPUReduction` sums (256 rows per workgroup partial, then further levels; never one workgroup
   * over all rows), the weighted mean, centered second moments, and a one-invocation finalize that
   * writes the bandwidth. Positions are taken relative to the grid start corner.
   */
  private getBandwidthRuleNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    bandwidthState: GraphDataView<'float32'>
  ): GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const pointCount = props.positions.length;
    const terms = ['weight', 'weight-square', 'x', 'y'].map(name =>
      createTransientView(graph, `${id}-bandwidth-term-${name}`, 'float32', pointCount)
    );
    const [termWeight, termWeightSquare, termX, termY] = terms;
    const sums = ['weight', 'weight-square', 'x', 'y', 'spread-x', 'spread-y'].map(name =>
      createTransientView(graph, `${id}-bandwidth-sum-${name}`, 'float32', 1)
    );
    const [sumWeight, sumWeightSquare, sumX, sumY, sumSpreadX, sumSpreadY] = sums;
    const means = createTransientView(graph, `${id}-bandwidth-means`, 'float32', 2);
    const inputBindings: WGSLKernelBinding[] = [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      ...(props.weights
        ? [{name: 'inputWeights', view: props.weights, type: 'f32', access: 'read'} as const]
        : []),
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'}
    ];
    // Rows with a non-finite position or a non-finite or negative weight contribute nothing.
    const rowPrelude = `let origin = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let position = vec2f(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
  let rawWeight = ${props.weights ? 'inputWeights[inputWeightsOffset + index]' : '1.0'};
  let included = isFiniteValue(position.x) && isFiniteValue(position.y) &&
    isFiniteValue(rawWeight) && rawWeight >= 0.0;
  let weight = select(0.0, rawWeight, included);
  let relative = select(vec2f(0.0), position - origin, included);`;
    const variantSuffix = props.weights ? 'weighted' : 'unweighted';
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-bandwidth-terms`,
        operation: OPERATION,
        variant: `bandwidth-terms-${variantSuffix}`,
        bindings: [
          ...inputBindings,
          {name: 'termWeight', view: termWeight, type: 'f32', access: 'read_write'},
          {name: 'termWeightSquare', view: termWeightSquare, type: 'f32', access: 'read_write'},
          {name: 'termX', view: termX, type: 'f32', access: 'read_write'},
          {name: 'termY', view: termY, type: 'f32', access: 'read_write'}
        ],
        invocationCount: pointCount,
        declarations: IS_FINITE_VALUE_WGSL,
        body: `${rowPrelude}
  termWeight[termWeightOffset + index] = weight;
  termWeightSquare[termWeightSquareOffset + index] = weight * weight;
  termX[termXOffset + index] = weight * relative.x;
  termY[termYOffset + index] = weight * relative.y;`
      })
    ];
    for (const [name, input, output] of [
      ['weight', termWeight, sumWeight],
      ['weight-square', termWeightSquare, sumWeightSquare],
      ['x', termX, sumX],
      ['y', termY, sumY]
    ] as const) {
      nodes.push(
        ...new GPUReduction({
          id: `${id}-bandwidth-sum-${name}`,
          input,
          output,
          operation: 'sum'
        }).getCommandNodes(graph)
      );
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-bandwidth-means`,
        operation: OPERATION,
        variant: 'bandwidth-means',
        bindings: [
          {name: 'sumWeight', view: sumWeight, type: 'f32', access: 'read'},
          {name: 'sumX', view: sumX, type: 'f32', access: 'read'},
          {name: 'sumY', view: sumY, type: 'f32', access: 'read'},
          {name: 'means', view: means, type: 'f32', access: 'read_write'}
        ],
        invocationCount: 1,
        body: `let total = sumWeight[sumWeightOffset];
  let valid = total > 0.0;
  means[meansOffset] = select(0.0, sumX[sumXOffset] / total, valid);
  means[meansOffset + 1u] = select(0.0, sumY[sumYOffset] / total, valid);`
      }),
      // Second pass: centered terms overwrite the first-pass x and y term buffers, which the
      // reductions above have already consumed (nodes execute in order).
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-bandwidth-spread`,
        operation: OPERATION,
        variant: `bandwidth-spread-${variantSuffix}`,
        bindings: [
          ...inputBindings,
          {name: 'means', view: means, type: 'f32', access: 'read'},
          {name: 'termX', view: termX, type: 'f32', access: 'read_write'},
          {name: 'termY', view: termY, type: 'f32', access: 'read_write'}
        ],
        invocationCount: pointCount,
        declarations: IS_FINITE_VALUE_WGSL,
        body: `${rowPrelude}
  let centered = relative - vec2f(means[meansOffset], means[meansOffset + 1u]);
  termX[termXOffset + index] = select(0.0, weight * centered.x * centered.x, included);
  termY[termYOffset + index] = select(0.0, weight * centered.y * centered.y, included);`
      })
    );
    for (const [name, input, output] of [
      ['spread-x', termX, sumSpreadX],
      ['spread-y', termY, sumSpreadY]
    ] as const) {
      nodes.push(
        ...new GPUReduction({
          id: `${id}-bandwidth-sum-${name}`,
          input,
          output,
          operation: 'sum'
        }).getCommandNodes(graph)
      );
    }
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-bandwidth-rule`,
        operation: OPERATION,
        variant: 'bandwidth-rule',
        bindings: [
          {name: 'sumWeight', view: sumWeight, type: 'f32', access: 'read'},
          {name: 'sumWeightSquare', view: sumWeightSquare, type: 'f32', access: 'read'},
          {name: 'sumSpreadX', view: sumSpreadX, type: 'f32', access: 'read'},
          {name: 'sumSpreadY', view: sumSpreadY, type: 'f32', access: 'read'},
          {name: 'bandwidthState', view: bandwidthState, type: 'f32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `${IS_FINITE_VALUE_WGSL}${GET_NAN_WGSL}`,
        // scipy: scotts_factor = n_eff^(-1/(d+4)); silverman_factor = (n_eff (d+2)/4)^(-1/(d+4));
        // both n_eff^(-1/6) for d = 2. Covariance is the unbiased weighted one: the weighted
        // spread divided by 1 - sum(w^2) / sum(w)^2.
        body: `let total = sumWeight[sumWeightOffset];
  let totalSquare = sumWeightSquare[sumWeightSquareOffset];
  let effective = total * total / totalSquare;
  let correction = 1.0 - totalSquare / (total * total);
  let variance = 0.5 * (sumSpreadX[sumSpreadXOffset] + sumSpreadY[sumSpreadYOffset]) / (total * correction);
  let bandwidth = pow(effective, -1.0 / 6.0) * sqrt(variance);
  let valid = total > 0.0 && totalSquare > 0.0 && effective > 1.0 && correction > 0.0 &&
    isFiniteValue(bandwidth) && bandwidth > 0.0;
  bandwidthState[bandwidthStateOffset] = select(getNaN(), bandwidth, valid);`
      })
    );
    return nodes;
  }

  /**
   * Returns the grid index build and the in-cell sort of the exact gather, with the cell offsets
   * and sorted IDs it reads.
   */
  private getIndexNodes<Parameters>(graph: GPUCommandGraph<Parameters>): {
    nodes: GPUCommandNode<Parameters>[];
    cellOffsets: GraphDataView<'uint32'>;
    sortedIds: GraphDataView<'uint32'>;
  } {
    const {id, props} = this;
    const {indexGridSize, indexBounds} = props;
    const pointCount = props.positions.length;
    const indexCellCount = indexGridSize[0] * indexGridSize[1];
    const cellOffsets = createTransientView(
      graph,
      `${id}-cell-offsets`,
      'uint32',
      indexCellCount + 1
    );
    const objectIds = createTransientView(graph, `${id}-object-ids`, 'uint32', pointCount);
    const sortedIds = createTransientView(graph, `${id}-sorted-ids`, 'uint32', pointCount);
    const indexCount = createTransientView(graph, `${id}-index-count`, 'uint32', 1);
    const indexOverflow = createTransientView(graph, `${id}-index-overflow`, 'uint32', 1);
    const nodes: GPUCommandNode<Parameters>[] = [
      ...new GPUGridIndex({
        id: `${id}-index`,
        positions: props.positions,
        gridSize: indexGridSize,
        bounds: indexBounds,
        cellOffsets,
        objectIds,
        count: indexCount,
        overflow: indexOverflow
      }).getCommandNodes(graph)
    ];

    // GPUGridIndex scatters IDs with atomics, so in-cell order varies between runs. Rank every ID
    // within its cell to restore ascending row order before any floating-point sum depends on it.
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-sort-cells`,
        operation: OPERATION,
        variant: 'sort-cells',
        bindings: [
          {name: 'cellOffsets', view: cellOffsets, type: 'u32', access: 'read'},
          {name: 'objectIds', view: objectIds, type: 'u32', access: 'read'},
          {name: 'indexCount', view: indexCount, type: 'u32', access: 'read'},
          {name: 'sortedIds', view: sortedIds, type: 'u32', access: 'read_write'}
        ],
        invocationCount: pointCount,
        declarations: `const INDEX_CELL_COUNT: u32 = ${indexCellCount}u;
const POINT_COUNT: u32 = ${pointCount}u;`,
        body: `if (index >= min(indexCount[indexCountOffset], POINT_COUNT)) {
    return;
  }
  let point = objectIds[objectIdsOffset + index];
  // Largest cell whose first slot is at or before this slot.
  var low = 0u;
  var high = INDEX_CELL_COUNT - 1u;
  while (low < high) {
    let middle = low + (high - low + 1u) / 2u;
    if (cellOffsets[cellOffsetsOffset + middle] <= index) {
      low = middle;
    } else {
      high = middle - 1u;
    }
  }
  let start = cellOffsets[cellOffsetsOffset + low];
  let end = cellOffsets[cellOffsetsOffset + low + 1u];
  var rank = 0u;
  for (var slot = start; slot < end; slot++) {
    rank += select(0u, 1u, objectIds[objectIdsOffset + slot] < point);
  }
  sortedIds[sortedIdsOffset + start + rank] = point;`
      })
    );
    return {nodes, cellOffsets, sortedIds};
  }

  /**
   * Returns the grid index build and the in-cell sort (not for `method: 'binned'`), the per-point
   * weights, the optional weight sum and invalid count reductions, and the per-cell gather
   * (`'exact'`), the binned estimate (`'binned'`), or both behind a GPU gate (`'auto'`), in order.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, kernel, edgeCorrection, normalization, method} = this;
    const {output, grid, indexBounds} = props;
    // The handle only looks up graph identity; it never invokes parameter-dependent callbacks.
    const studyWindow =
      this.studyWindow === 'rectangle'
        ? undefined
        : this.studyWindow.getViews(graph as unknown as GPUCommandGraph<unknown>);
    if (studyWindow) {
      validateGraphViewsBelongToGraph(id, graph, [
        studyWindow.edges,
        studyWindow.summary,
        studyWindow.slabs.params,
        studyWindow.slabs.slabOffsets,
        studyWindow.slabs.slabEntries,
        studyWindow.sampleCells,
        studyWindow.sampleCellCount,
        studyWindow.packed
      ]);
    }
    const needsExact = method !== 'binned';
    const {width, height} = grid;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.weights,
      props.parameters,
      output.values,
      output.weightSum,
      output.status?.invalidCount,
      output.status?.insufficientPadding,
      output.bandwidth
    ]);
    validateMaximumCount(
      id,
      'indexGridSize cells',
      props.indexGridSize[0] * props.indexGridSize[1],
      getGPUGridIndexMaximumCellCount(graph.device)
    );
    const pointCount = props.positions.length;
    const nodes: GPUCommandNode<Parameters>[] = [];
    // The binned path never reads the index; only the exact gather does.
    const exactIndex = needsExact ? this.getIndexNodes(graph) : undefined;
    if (exactIndex) {
      nodes.push(...exactIndex.nodes);
    }

    // Bandwidth rule on the GPU: the density kernels read `bandwidthState` instead of parameters[4].
    const bandwidthState = props.bandwidthRule
      ? (output.bandwidth ?? createTransientView(graph, `${id}-bandwidth-state`, 'float32', 1))
      : undefined;
    if (bandwidthState) {
      nodes.push(...this.getBandwidthRuleNodes(graph, bandwidthState));
    }
    const bandwidthBindings: WGSLKernelBinding[] = bandwidthState
      ? [{name: 'bandwidthState', view: bandwidthState, type: 'f32', access: 'read'}]
      : [];
    const bandwidthExpression = bandwidthState
      ? 'bandwidthState[bandwidthStateOffset]'
      : 'params[paramsOffset + 4u]';

    // Per-point weights: 0 for excluded rows; w / e(x) under Diggle. The raw (uncorrected) weights
    // feed the weight sum, so they are separate only when Diggle changes them.
    const needsRawWeights =
      edgeCorrection === 'diggle' && (normalization === 'probability' || Boolean(output.weightSum));
    const needsWeightSum = normalization === 'probability' || Boolean(output.weightSum);
    const pointWeights = createTransientView(graph, `${id}-point-weights`, 'float32', pointCount);
    const rawWeights = needsRawWeights
      ? createTransientView(graph, `${id}-raw-weights`, 'float32', pointCount)
      : pointWeights;
    const excluded = output.status?.invalidCount
      ? createTransientView(graph, `${id}-excluded`, 'uint32', pointCount)
      : undefined;
    const domainConstants = getDomainConstantsWGSL(indexBounds);
    const hasEdgeCorrection = edgeCorrection !== 'none';
    const weightBindings: WGSLKernelBinding[] = [
      ...(studyWindow ? [getStudyWindowBinding(studyWindow)] : []),
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      ...(props.weights
        ? [{name: 'inputWeights', view: props.weights, type: 'f32', access: 'read'} as const]
        : []),
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
      ...bandwidthBindings,
      {name: 'pointWeights', view: pointWeights, type: 'f32', access: 'read_write'},
      ...(needsRawWeights
        ? [{name: 'rawWeights', view: rawWeights, type: 'f32', access: 'read_write'} as const]
        : []),
      ...(excluded
        ? [{name: 'excluded', view: excluded, type: 'u32', access: 'read_write'} as const]
        : [])
    ];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-point-weights`,
        operation: OPERATION,
        variant: `point-weights-${edgeCorrection}`,
        bindings: weightBindings,
        invocationCount: pointCount,
        declarations: `${domainConstants}
${GRID_COORDINATE_WGSL}
${studyWindow ? getKernelDensityStudyWindowWGSL(studyWindow, edgeCorrection === 'diggle' ? kernel : undefined) : edgeCorrection === 'diggle' ? getWindowMassWGSL(kernel) : ''}`,
        body: `let gridStart = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let bandwidth = ${bandwidthExpression};
  let windowMin = vec2f(params[paramsOffset + 5u], params[paramsOffset + 6u]) - gridStart;
  let windowMax = vec2f(params[paramsOffset + 7u], params[paramsOffset + 8u]) - gridStart;
  let position = vec2f(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
  let weight = ${props.weights ? 'inputWeights[inputWeightsOffset + index]' : '1.0'};
  var included = isFiniteValue(position.x) && isFiniteValue(position.y) &&
    all(position >= DOMAIN_MIN) && all(position <= DOMAIN_MAX) &&
    isFiniteValue(weight) && weight >= 0.0;
  let relative = position - gridStart;
  ${studyWindow ? 'included = included && isInsideStudyWindow(position);' : hasEdgeCorrection ? 'included = included && all(relative >= windowMin) && all(relative <= windowMax);' : ''}
  var corrected = select(0.0, weight, included);
  ${rawWeights !== pointWeights ? 'rawWeights[rawWeightsOffset + index] = corrected;' : ''}
  ${
    edgeCorrection === 'diggle'
      ? `if (included) {
    let mass = ${studyWindow ? 'getStudyWindowKernelMass(position, bandwidth)' : 'getWindowMass(relative, windowMin, windowMax, bandwidth)'};
    corrected = select(0.0, weight / mass, mass > 1e-6);
  }`
      : ''
  }
  pointWeights[pointWeightsOffset + index] = corrected;
  ${excluded ? 'excluded[excludedOffset + index] = select(1u, 0u, included);' : ''}`
      })
    );
    const weightSum = needsWeightSum
      ? (output.weightSum ?? createTransientView(graph, `${id}-weight-sum-scratch`, 'float32', 1))
      : undefined;
    if (weightSum) {
      nodes.push(
        ...new GPUReduction({
          id: `${id}-weight-sum`,
          input: rawWeights,
          output: weightSum,
          operation: 'sum'
        }).getCommandNodes(graph)
      );
    }
    if (excluded && output.status?.invalidCount) {
      nodes.push(
        ...new GPUReduction({
          id: `${id}-invalid-count`,
          input: excluded,
          output: output.status?.invalidCount,
          operation: 'sum'
        }).getCommandNodes(graph)
      );
    }

    // Binned path: the plan (and under 'auto' the gate) precede the gather, the blur nodes follow.
    const binned =
      method === 'exact'
        ? undefined
        : createGPUKernelDensityBinnedNodes(graph, {
            id,
            positions: props.positions,
            parameters: props.parameters,
            pointWeights,
            studyWindow,
            bandwidthBindings,
            bandwidthExpression,
            weightSum,
            values: output.values,
            width,
            height,
            padding: this.binnedPadding,
            insufficientPadding: output.status?.insufficientPadding,
            edgeCorrection,
            normalization,
            autoGate: method === 'auto'
          });
    if (binned) {
      nodes.push(...binned.planNodes);
    }
    if (exactIndex) {
      // Separate polygon finish keeps the gather within eight storage bindings even with a
      // GPU bandwidth rule and probability normalization.
      const gathered = studyWindow
        ? createTransientView(graph, `${id}-gathered`, 'float32', this.cellCount)
        : undefined;
      nodes.push(
        this.getGatherNode(graph, {
          exactIndex,
          gathered,
          pointWeights,
          bandwidthBindings,
          bandwidthExpression,
          weightSum,
          gate: binned?.exactGate
        })
      );
      if (studyWindow && gathered) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-polygon-finish`,
            operation: OPERATION,
            variant: `polygon-finish-${kernel}-${edgeCorrection}-${normalization}`,
            bindings: [
              {name: 'gathered', view: gathered, type: 'f32', access: 'read'},
              {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
              ...bandwidthBindings,
              getStudyWindowBinding(studyWindow),
              ...(normalization === 'probability' && weightSum
                ? [{name: 'weightSum', view: weightSum, type: 'f32', access: 'read'} as const]
                : []),
              {name: 'valuesOut', view: output.values, type: 'f32', access: 'read_write'}
            ],
            invocationCount: this.cellCount,
            workgroupSize: 64,
            ...binned?.exactGate,
            declarations: `${GRID_COORDINATE_WGSL}${getKernelDensityStudyWindowWGSL(studyWindow, edgeCorrection === 'uniform' ? kernel : undefined)}`,
            body: `let gridStart = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let gridEnd = vec2f(params[paramsOffset + 2u], params[paramsOffset + 3u]);
  let cellSize = (gridEnd - gridStart) / vec2f(${width}.0, ${height}.0);
  let center = gridStart + (vec2f(f32(index % ${width}u), f32(index / ${width}u)) + vec2f(0.5)) * cellSize;
  let bandwidth = ${bandwidthExpression};
  let total = gathered[gatheredOffset + index];
  let valid = isFiniteValue(total) && isStudyWindowUsable();
  ${getKernelDensityResultWGSL(edgeCorrection, normalization, true)}`
          })
        );
      }
    }
    if (binned) {
      nodes.push(...binned.bodyNodes);
    }
    return nodes;
  }

  /** Per-cell exact gather over the sorted grid index, optionally gated by `method: 'auto'`. */
  private getGatherNode<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    inputs: {
      exactIndex: {cellOffsets: GraphDataView<'uint32'>; sortedIds: GraphDataView<'uint32'>};
      pointWeights: GraphDataView<'float32'>;
      gathered?: GraphDataView<'float32'>;
      bandwidthBindings: WGSLKernelBinding[];
      bandwidthExpression: string;
      weightSum?: GraphDataView<'float32'>;
      gate?: {
        condition: GPUCommandGraphNodeCondition<Parameters>;
        extraResources: GraphBufferUse[];
      };
    }
  ): GPUCommandNode<Parameters> {
    const {id, props, cellCount, kernel} = this;
    const edgeCorrection = inputs.gathered ? 'none' : this.edgeCorrection;
    const normalization = inputs.gathered ? 'intensity' : this.normalization;
    const {output, grid, indexGridSize, indexBounds} = props;
    const {width, height} = grid;
    const {pointWeights, bandwidthBindings, bandwidthExpression, weightSum, gate} = inputs;
    const {cellOffsets, sortedIds} = inputs.exactIndex;
    const domainConstants = getDomainConstantsWGSL(indexBounds);
    const gatherBindings: WGSLKernelBinding[] = [
      {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
      {name: 'pointWeights', view: pointWeights, type: 'f32', access: 'read'},
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
      ...bandwidthBindings,
      {name: 'cellOffsets', view: cellOffsets, type: 'u32', access: 'read'},
      {name: 'sortedIds', view: sortedIds, type: 'u32', access: 'read'},
      ...(normalization === 'probability' && weightSum
        ? [{name: 'weightSum', view: weightSum, type: 'f32', access: 'read'} as const]
        : []),
      {name: 'valuesOut', view: inputs.gathered ?? output.values, type: 'f32', access: 'read_write'}
    ];
    const supportRadiusFactor = kernel === 'gaussian' ? GPU_KERNEL_DENSITY_GAUSSIAN_TRUNCATION : 1;
    const visit = `let pointWeight = pointWeights[pointWeightsOffset + sample];
        if (!(pointWeight > 0.0)) {
          continue;
        }
        let position = vec2f(positions[positionsOffset + 2u * sample], positions[positionsOffset + 2u * sample + 1u]);
        // Distances come from local-origin offsets so large absolute coordinates cancel exactly.
        let delta = (position - gridStart) - cellOffset;
        let distanceSquared = dot(delta, delta);
        if (!(distanceSquared < supportSquared)) {
          continue;
        }
        total += pointWeight * getKernelValue(distanceSquared, bandwidth);`;
    return createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-gather`,
      operation: OPERATION,
      variant: `gather-${kernel}-${edgeCorrection}-${normalization}`,
      bindings: gatherBindings,
      invocationCount: cellCount,
      workgroupSize: 64,
      ...gate,
      declarations: `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const INDEX_WIDTH: u32 = ${indexGridSize[0]}u;
const INDEX_HEIGHT: u32 = ${indexGridSize[1]}u;
const SUPPORT_FACTOR: f32 = ${getWGSLFloatLiteral(supportRadiusFactor)};
${domainConstants}
${GRID_COORDINATE_WGSL}
${getKernelValueWGSL(kernel)}
${edgeCorrection === 'none' ? '' : getWindowMassWGSL(kernel)}`,
      body: `let column = index % WIDTH;
  let rasterRow = index / WIDTH;
  let gridStart = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let gridEnd = vec2f(params[paramsOffset + 2u], params[paramsOffset + 3u]);
  let bandwidth = ${bandwidthExpression};
  let windowMin = vec2f(params[paramsOffset + 5u], params[paramsOffset + 6u]) - gridStart;
  let windowMax = vec2f(params[paramsOffset + 7u], params[paramsOffset + 8u]) - gridStart;
  let cellSize = (gridEnd - gridStart) / vec2f(f32(WIDTH), f32(HEIGHT));
  let cellOffset = (vec2f(f32(column), f32(rasterRow)) + vec2f(0.5)) * cellSize;
  let center = gridStart + cellOffset;
  let radius = SUPPORT_FACTOR * bandwidth;
  let supportSquared = radius * radius;
  let queryMin = center - vec2f(radius);
  let queryMax = center + vec2f(radius);
  let valid = isFiniteValue(center.x) && isFiniteValue(center.y) &&
    isFiniteValue(bandwidth) && bandwidth > 0.0 && isFiniteValue(radius) &&
    all(gridEnd != gridStart);
  var total = 0.0;
  if (valid && all(queryMax >= DOMAIN_MIN) && all(queryMin <= DOMAIN_MAX)) {
    let clampedMin = max(queryMin, DOMAIN_MIN);
    let clampedMax = min(queryMax, DOMAIN_MAX);
    // Widen by one index cell per side to absorb f32 rounding in the binning pass; the exact
    // distance test rejects the extra candidates.
    var columnLow = getCoordinate(clampedMin.x, DOMAIN_MIN.x, DOMAIN_MAX.x, INDEX_WIDTH);
    var rowLow = getCoordinate(clampedMin.y, DOMAIN_MIN.y, DOMAIN_MAX.y, INDEX_HEIGHT);
    columnLow = columnLow - min(columnLow, 1u);
    rowLow = rowLow - min(rowLow, 1u);
    let columnHigh = min(getCoordinate(clampedMax.x, DOMAIN_MIN.x, DOMAIN_MAX.x, INDEX_WIDTH) + 1u, INDEX_WIDTH - 1u);
    let rowHigh = min(getCoordinate(clampedMax.y, DOMAIN_MIN.y, DOMAIN_MAX.y, INDEX_HEIGHT) + 1u, INDEX_HEIGHT - 1u);
    ${getRectangleScanWGSL(visit)}
  }
  ${getKernelDensityResultWGSL(edgeCorrection, normalization)}`
    });
  }
}
