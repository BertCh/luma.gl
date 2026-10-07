// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GraphVectorView,
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
  createGridIndexNodes,
  getGridLookupWGSL,
  getRingNeighborDistancesWGSL
} from './gwr-grid-index';
import {getCholeskyWGSL} from './spatial-regression-solve';
import {
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_ADAPTIVE_BANDWIDTH_FACTOR,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_DEFAULT_GRID_DIMENSION,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_INDEXED_ROW_COUNT,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_LADDER_LENGTH,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_NEIGHBOR_COUNT,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MINIMUM_INDEXED_ROW_COUNT,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_PREDICTOR_COUNT,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_ROW_COUNT,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MINIMUM_VARIANCE,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_PARAMETER_HEADER_LENGTH,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH,
  getGPUGeographicallyWeightedRegressionParameterLength
} from './geographically-weighted-regression-parameters';

const OPERATION = 'GPUGeographicallyWeightedRegression';
/** Rows summed in fixed order by one thread of a tile pass. */
const TILE_ROWS = 256;
/** Float32 values per row of the packed local statistics scratch. */
const LOCAL_STRIDE = 6;
/** Default `maxStorageBufferBindingSize`; bounds the per-candidate scratch. */
const MAXIMUM_SCRATCH_BYTES = 134217728;

/** Caller-owned outputs of {@link GPUGeographicallyWeightedRegression}. */
export type GPUGeographicallyWeightedRegressionOutput = {
  /**
   * Local coefficients at the selected bandwidth, row-major `row * (predictorCount + 1) + column`;
   * column 0 is the intercept and column `c + 1` is predictor `c`. NaN for excluded or singular rows.
   */
  coefficients: GraphDataView<'float32'>;
  /** Optional local weighted R^2 per row (`1 - weighted RSS / weighted TSS`); NaN when undefined. */
  localR2?: GraphDataView<'float32'>;
  /** Optional fitted value `x_i^T beta_i` per row; NaN for excluded or singular rows. */
  fitted?: GraphDataView<'float32'>;
  /** Optional residual `y_i - fitted_i` per row. */
  residuals?: GraphDataView<'float32'>;
  /** Optional hat-matrix diagonal `S_ii` per row. */
  hatDiagonal?: GraphDataView<'float32'>;
  /** Optional per-row status, see `GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS`. */
  localStatus?: GraphDataView<'uint32'>;
  /**
   * Optional local condition number per row, as mgwr `GWRResults.local_collinearity` (its
   * `local_CN`): the ratio of the largest to the smallest singular value of the design `[1, X]`
   * with every row scaled by its kernel weight and every column scaled to unit length. Large
   * values (mgwr and the GWR literature flag above 30) mean the local design is nearly collinear.
   * NaN for excluded or singular rows. The extra work is one more pass over each neighborhood.
   */
  localConditionNumber?: GraphDataView<'float32'>;
  /** Optional AICc per ladder candidate (NaN for invalid or unused candidates), `maximumBandwidthCount` values. */
  bandwidthScores?: GraphDataView<'float32'>;
  /** Optional `[selected ladder index, selected ladder value]`. */
  selectedBandwidth?: GraphDataView<'float32'>;
  /** Optional global summary, see `GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY`. */
  summary?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUGeographicallyWeightedRegression}.
 *
 * Per-frame (no rebuild or recompile): the contents of `positions`, `predictors`, `response`,
 * `mask` and `parameters` (kernel, bandwidth mode, ladder). Compile-time: the row count,
 * `predictorCount`, `maximumBandwidthCount`, `maximumNeighborCount`, and which optional views are
 * present.
 */
export type GPUGeographicallyWeightedRegressionProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'geographically-weighted-regression'`. */
  id?: string;
  /** Packed planar positions, one per row, in the units of the bandwidths. */
  positions: GraphDataView<'float32x2'>;
  /** Packed float32 predictors, row-major `row * predictorCount + column`, without the intercept. */
  predictors: GraphDataView<'float32'>;
  /** Number of predictor columns, 1 to 7 (the intercept is added). Compile-time. */
  predictorCount: number;
  /** Packed float32 response, one per row. */
  response: GraphDataView<'float32'>;
  /** Optional packed `uint32` row mask; zero excludes the row. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Per-frame parameters: float32 view of at least
   * `getGPUGeographicallyWeightedRegressionParameterLength(maximumBandwidthCount)` elements written
   * with `getGPUGeographicallyWeightedRegressionParameterValues`.
   */
  parameters: GraphDataView<'float32'>;
  /** Ladder capacity, 1 to 32. Compile-time. Default 32. */
  maximumBandwidthCount?: number;
  /** Largest adaptive `k`, 1 to 128. Compile-time. Default 128. */
  maximumNeighborCount?: number;
  /**
   * Dimensions `[columns, rows]` of the internal `GPUGridIndex` over the included rows, or `false`
   * for none. Compile-time. Defaults to a square grid of about 8 rows per cell (at most 1024 per
   * axis) when there are at least 1024 rows, otherwise none. The domain is the extent of the
   * included rows, recomputed every encoding. The index is used by bisquare kernels (fixed or
   * adaptive bandwidths) and by the adaptive k-th neighbour search; Gaussian weights scan every row. Aim for a few
   * cells across the typical bandwidth.
   */
  indexGridSize?: readonly [number, number] | false;
  /** Caller-owned outputs. */
  output: GPUGeographicallyWeightedRegressionOutput;
};

/**
 * Geographically weighted regression (Brunsdon, Fotheringham and Charlton 1996; Fotheringham,
 * Brunsdon and Charlton 2002) with automatic bandwidth selection by AICc, as in mgwr and ArcGIS
 * "Geographically Weighted Regression".
 *
 * For every included location `i` (a row with unmasked, finite position, predictors and response)
 * the contributor fits a weighted least squares `y ~ 1 + x` over all included rows `j` with weights
 * `w_ij = K(d_ij / h_i)`, `d` the planar Euclidean distance. Kernels: `'gaussian'`
 * `exp(-0.5 (d/h)^2)`, `'bisquare'` `(1 - (d/h)^2)^2` for `d < h`. The bandwidth is fixed (`h` is
 * the ladder value) or adaptive (`h` is `1.00001` times the distance to the k-th nearest included
 * row, counting the location itself, as mgwr; ties in distance cannot change the k-th value).
 *
 * The parameter buffer carries a bandwidth ladder of up to `maximumBandwidthCount` candidates.
 * For every candidate the contributor computes `RSS = sum (y_i - yhat_i)^2`, the hat trace
 * `tr(S) = sum S_ii` with `S_ii = x_i^T (X'WX)^-1 x_i w_ii`, and
 * `AICc = n ln(RSS/n) + n ln(2 pi) + n (n + tr(S)) / (n - 2 - tr(S))` (n = included rows). A
 * candidate is invalid (score NaN) when any included location is singular, the bandwidth is not
 * positive (fixed), `k` is outside `[2, maximumNeighborCount]` or exceeds the included rows
 * (adaptive), or `n - 2 - tr(S) <= 0`. The argmin over valid candidates (ties to the lowest
 * index) selects the bandwidth for the final local coefficients. A ladder of one value is a
 * fixed, unselected bandwidth. When no candidate is valid, index 0 is used for the outputs and
 * `summary[HAS_VALID_CANDIDATE]` is 0.
 *
 * Numerics: the design is centred on the focal row, `z_j = (1, x_j - x_i)`, so the fitted value
 * is the centred intercept and `S_ii` is the `(0, 0)` entry of the inverse; the normal equations
 * are Jacobi-equilibrated (divided by the square root of their diagonal) before the shared
 * Cholesky solve. Original-space intercepts are recovered as `b0 - sum b_c x_ic`. A predictor
 * that is constant over a location's neighbourhood makes that location singular.
 *
 * Complexity: one thread per location evaluates every candidate (and once more to select k-th
 * nearest distances and to evaluate the final fit). Bisquare kernels have bounded support, so
 * with a `GPUGridIndex` (`indexGridSize`, built by default from 1024 rows) they visit only the rows
 * in the cells covering the bandwidth, for fixed and adaptive bandwidths alike: the work is
 * `O(n * neighbours * ladder * p^2)` and the 65,536-row limit of the scan is lifted to 1,048,576
 * rows (the scan limit existed because `n^2` work per candidate becomes unusable; the indexed
 * limit is the `rows * ladder * 8` byte per-candidate scratch fitting one 128 MiB storage
 * binding). An adaptive bandwidth finds its k-th nearest distance by walking grid cells in rings
 * around the location and stopping once the k nearest are inside the visited block, `O(k)` per
 * location instead of a scan of every row (any kernel, adaptive). Invocations take the included
 * rows in grid-cell order, so one workgroup fits neighboring locations and shares their cells.
 * Gaussian kernels (unbounded, no cutoff is assumed) still scan every row per candidate in
 * `O(n^2)`; a grid-enabled instance accepts them but the caller must keep `n` practical. The grid
 * path sums rows in cell order (ascending row ID inside each cell) instead of row order, so
 * results differ from the scan only by float rounding (about 1e-5 in coefficients); sums stay in a
 * fixed order and no float atomics are used, so results are bitwise reproducible on one adapter.
 * Candidate RSS and trace are reduced by 256-row tiles merged in tile order. Very clustered data
 * puts many rows in one cell and the in-cell ID ranking costs `O(cell size^2)`.
 */
export class GPUGeographicallyWeightedRegression implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUGeographicallyWeightedRegressionProps;
  /** Row count, `positions.length`. */
  readonly rowCount: number;
  /** Columns of the design including the intercept, `predictorCount + 1`. */
  readonly coefficientCount: number;
  /** Ladder capacity. */
  readonly maximumBandwidthCount: number;
  /** Largest adaptive `k`. */
  readonly maximumNeighborCount: number;
  /** Grid index dimensions `[columns, rows]`, or null when no index is built. */
  readonly indexGridSize: readonly [number, number] | null;

  constructor(props: GPUGeographicallyWeightedRegressionProps) {
    this.id = props.id ?? 'geographically-weighted-regression';
    this.props = props;
    const id = this.id;
    const {predictorCount, output} = props;
    if (
      !Number.isInteger(predictorCount) ||
      predictorCount < 1 ||
      predictorCount > GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_PREDICTOR_COUNT
    ) {
      throw new Error(
        `${id} predictorCount must be an integer in [1, ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_PREDICTOR_COUNT}]`
      );
    }
    this.coefficientCount = predictorCount + 1;
    this.maximumBandwidthCount =
      props.maximumBandwidthCount ?? GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_LADDER_LENGTH;
    if (
      !Number.isInteger(this.maximumBandwidthCount) ||
      this.maximumBandwidthCount < 1 ||
      this.maximumBandwidthCount > GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_LADDER_LENGTH
    ) {
      throw new Error(
        `${id} maximumBandwidthCount must be an integer in [1, ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_LADDER_LENGTH}]`
      );
    }
    this.maximumNeighborCount =
      props.maximumNeighborCount ?? GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_NEIGHBOR_COUNT;
    if (
      !Number.isInteger(this.maximumNeighborCount) ||
      this.maximumNeighborCount < 1 ||
      this.maximumNeighborCount > GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_NEIGHBOR_COUNT
    ) {
      throw new Error(
        `${id} maximumNeighborCount must be an integer in [1, ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_NEIGHBOR_COUNT}]`
      );
    }
    for (const [name, view] of [
      ['positions', props.positions],
      ['predictors', props.predictors],
      ['response', props.response],
      ['mask', props.mask],
      ['parameters', props.parameters],
      ...Object.entries(output)
    ] as const) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    this.rowCount = props.positions.length;
    if (this.rowCount < 1) {
      throw new Error(`${id} needs at least one row`);
    }
    if (props.indexGridSize === false) {
      this.indexGridSize = null;
    } else if (props.indexGridSize) {
      const [columns, gridRows] = props.indexGridSize;
      if (
        !Number.isInteger(columns) ||
        !Number.isInteger(gridRows) ||
        columns < 1 ||
        gridRows < 1 ||
        columns * gridRows >= 0xffffffff
      ) {
        throw new Error(
          `${id} indexGridSize must be two positive integers with columns * rows < 2^32 - 1`
        );
      }
      this.indexGridSize = [columns, gridRows];
    } else if (this.rowCount >= GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MINIMUM_INDEXED_ROW_COUNT) {
      const dimension = Math.min(
        GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_DEFAULT_GRID_DIMENSION,
        Math.max(1, Math.ceil(Math.sqrt(this.rowCount / 8)))
      );
      this.indexGridSize = [dimension, dimension];
    } else {
      this.indexGridSize = null;
    }
    if (
      !this.indexGridSize &&
      this.rowCount > GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_ROW_COUNT
    ) {
      throw new Error(
        `${id} supports at most ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_ROW_COUNT} rows without a grid index`
      );
    }
    if (
      this.indexGridSize &&
      (this.rowCount > GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_INDEXED_ROW_COUNT ||
        this.rowCount * this.maximumBandwidthCount * 8 > MAXIMUM_SCRATCH_BYTES)
    ) {
      throw new Error(
        `${id} supports at most ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_INDEXED_ROW_COUNT} rows ` +
          `and rows * maximumBandwidthCount * 8 <= ${MAXIMUM_SCRATCH_BYTES} with a grid index`
      );
    }
    validatePackedView(props.predictors, ['float32'], `${id} predictors`);
    if (props.predictors.length !== this.rowCount * predictorCount) {
      throw new Error(`${id} predictors length must equal rows * predictorCount`);
    }
    validatePackedView(props.response, ['float32'], `${id} response`);
    if (props.response.length !== this.rowCount) {
      throw new Error(`${id} response length must equal the row count`);
    }
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
      if (props.mask.length !== this.rowCount) {
        throw new Error(`${id} mask length must equal the row count`);
      }
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    const parameterLength = getGPUGeographicallyWeightedRegressionParameterLength(
      this.maximumBandwidthCount
    );
    if (props.parameters.length < parameterLength) {
      throw new Error(`${id} parameters must hold ${parameterLength} float32 values`);
    }
    const lengths: [keyof GPUGeographicallyWeightedRegressionOutput, number][] = [
      ['coefficients', this.rowCount * this.coefficientCount],
      ['localR2', this.rowCount],
      ['fitted', this.rowCount],
      ['residuals', this.rowCount],
      ['hatDiagonal', this.rowCount],
      ['localStatus', this.rowCount],
      ['localConditionNumber', this.rowCount],
      ['bandwidthScores', this.maximumBandwidthCount],
      ['selectedBandwidth', 2],
      ['summary', GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH]
    ];
    for (const [name, length] of lengths) {
      const view = output[name];
      if (!view) {
        if (name === 'coefficients') {
          throw new Error(`${id} output.coefficients is required`);
        }
        continue;
      }
      if (name === 'localStatus') {
        validatePackedUint32View(view as GraphDataView<'uint32'>, `${id} output.${name}`);
      } else {
        validatePackedView(view, ['float32'], `${id} output.${name}`);
      }
      if (view.length < length) {
        throw new Error(`${id} output.${name} must hold at least ${length} values`);
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      lengths.map(([name]) => output[name]),
      [props.positions, props.predictors, props.response, props.mask, props.parameters]
    );
  }

  /**
   * Returns validate, response tile sums, total-sum-of-squares tiles, per-location candidate
   * evaluation, candidate tile sums, bandwidth selection, final fit, and (when any per-row output
   * besides the coefficients is requested) publish nodes in order.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {
      props,
      id,
      rowCount,
      coefficientCount,
      maximumBandwidthCount,
      maximumNeighborCount,
      indexGridSize
    } = this;
    const {output, predictorCount, positions, predictors, response, parameters} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      predictors,
      response,
      props.mask,
      parameters,
      ...Object.values(output)
    ]);
    const tileCount = Math.ceil(rowCount / TILE_ROWS);
    const transient = <Format extends 'uint32' | 'float32' | 'float32x2'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, Math.max(length, 1));
    const rowValid = transient('row-valid', 'uint32', rowCount);
    const tileResponse = transient('tile-response', 'float32', tileCount * 2);
    const tileTotal = transient('tile-total', 'float32', tileCount);
    const candidateScratch = transient(
      'candidate-scratch',
      'float32',
      rowCount * maximumBandwidthCount * 2
    );
    const tileCandidates = transient(
      'tile-candidates',
      'float32',
      tileCount * maximumBandwidthCount * 3
    );
    const scores =
      output.bandwidthScores ?? transient('bandwidth-scores', 'float32', maximumBandwidthCount);
    const selection = output.selectedBandwidth ?? transient('selected-bandwidth', 'float32', 2);
    const summary =
      output.summary ??
      transient('summary', 'float32', GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH);
    const locals = transient('locals', 'uint32', rowCount * LOCAL_STRIDE);

    const gridPositions = indexGridSize ? transient('grid-positions', 'float32x2', rowCount) : null;
    const gridIndex =
      indexGridSize && gridPositions
        ? createGridIndexNodes(graph, {
            id,
            operation: OPERATION,
            positions: gridPositions,
            rowCount,
            gridSize: indexGridSize
          })
        : null;
    const cellOffsets = gridIndex?.cellOffsets ?? null;
    const sortedIds = gridIndex?.sortedIds ?? null;

    const read = (name: string, view: GraphDataView, type: 'u32' | 'f32'): WGSLKernelBinding => ({
      name,
      view,
      type,
      access: 'read'
    });
    const write = (name: string, view: GraphDataView, type: 'u32' | 'f32'): WGSLKernelBinding => ({
      name,
      view,
      type,
      access: 'read_write'
    });
    const P = coefficientCount;
    const common = `const ROW_COUNT: u32 = ${rowCount}u;
const PREDICTOR_COUNT: u32 = ${predictorCount}u;
const P: u32 = ${P}u;
const LADDER: u32 = ${maximumBandwidthCount}u;
const MAXIMUM_NEIGHBORS: u32 = ${maximumNeighborCount}u;
const TILE_ROWS: u32 = ${TILE_ROWS}u;
const TILE_COUNT: u32 = ${tileCount}u;
const LOCAL_STRIDE: u32 = ${LOCAL_STRIDE}u;
const HEADER: u32 = ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_PARAMETER_HEADER_LENGTH}u;
const SENTINEL: f32 = 3.0e38;
const ADAPTIVE_FACTOR: f32 = ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_ADAPTIVE_BANDWIDTH_FACTOR};
const MINIMUM_VARIANCE: f32 = ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MINIMUM_VARIANCE};
const LN_TWO_PI: f32 = 1.8378770664093453;
${GWR_COMMON_WGSL}`;
    const fitDeclarations = `${common}
${CANDIDATE_COUNT_WGSL}
${getCholeskyWGSL(P)}
${getFitWGSL(P, indexGridSize, Boolean(output.localConditionNumber))}`;
    // Bindings of the grid candidate path, shared by the candidate and final fit kernels.
    const gridBindings: WGSLKernelBinding[] = indexGridSize
      ? [read('cellOffsets', cellOffsets!, 'u32'), read('sortedIds', sortedIds!, 'u32')]
      : [];
    const mask = props.mask;
    // With a grid, invocations take the included rows in grid-cell order instead of row order, so
    // the invocations of one workgroup fit neighboring locations and read the same neighborhood
    // cells (row order is spatially random). Per-location results do not depend on the order.
    const focalRowWGSL = indexGridSize
      ? `if (index >= getFocalCount()) {
    return;
  }
  let focal = getFocalRow(index);`
      : 'let focal = index;';
    const nodes: GPUCommandNode<Parameters>[] = [];

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-validate`,
        operation: OPERATION,
        variant: 'validate',
        bindings: [
          read('positions', positions, 'f32'),
          read('predictors', predictors, 'f32'),
          read('response', response, 'f32'),
          ...(mask ? [read('rowMask', mask, 'u32')] : []),
          write('rowValid', rowValid, 'u32')
        ],
        invocationCount: rowCount,
        declarations: common,
        body: `var isValid = true;
  ${mask ? 'isValid = rowMask[rowMaskOffset + index] != 0u;' : ''}
  if (!isFiniteBits(positions[positionsOffset + 2u * index]) ||
    !isFiniteBits(positions[positionsOffset + 2u * index + 1u]) ||
    !isFiniteBits(response[responseOffset + index])) {
    isValid = false;
  }
  for (var column = 0u; column < PREDICTOR_COUNT; column++) {
    if (!isFiniteBits(predictors[predictorsOffset + index * PREDICTOR_COUNT + column])) {
      isValid = false;
    }
  }
  rowValid[rowValidOffset + index] = select(0u, 1u, isValid);`
      })
    );

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-tile-response`,
        operation: OPERATION,
        variant: 'tile-response',
        bindings: [
          read('response', response, 'f32'),
          read('rowValid', rowValid, 'u32'),
          write('tileResponse', tileResponse, 'f32')
        ],
        invocationCount: tileCount,
        declarations: common,
        body: `let firstRow = index * TILE_ROWS;
  let endRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  var count = 0.0;
  var sum = 0.0;
  for (var row = firstRow; row < endRow; row++) {
    if (rowValid[rowValidOffset + row] != 0u) {
      count = count + 1.0;
      sum = sum + response[responseOffset + row];
    }
  }
  tileResponse[tileResponseOffset + 2u * index] = count;
  tileResponse[tileResponseOffset + 2u * index + 1u] = sum;`
      })
    );

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-tile-total`,
        operation: OPERATION,
        variant: 'tile-total',
        bindings: [
          read('response', response, 'f32'),
          read('rowValid', rowValid, 'u32'),
          read('tileResponse', tileResponse, 'f32'),
          write('tileTotal', tileTotal, 'f32')
        ],
        invocationCount: tileCount,
        declarations: common,
        body: `var count = 0.0;
  var sum = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    count = count + tileResponse[tileResponseOffset + 2u * tile];
    sum = sum + tileResponse[tileResponseOffset + 2u * tile + 1u];
  }
  let mean = sum / max(count, 1.0);
  let firstRow = index * TILE_ROWS;
  let endRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  var total = 0.0;
  for (var row = firstRow; row < endRow; row++) {
    if (rowValid[rowValidOffset + row] != 0u) {
      let deviation = response[responseOffset + row] - mean;
      total = total + deviation * deviation;
    }
  }
  tileTotal[tileTotalOffset + index] = total;`
      })
    );

    if (gridIndex && gridPositions) {
      nodes.push(
        // The grid index has no mask, so excluded rows become NaN points, which it ignores.
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-grid-positions`,
          operation: OPERATION,
          variant: 'grid-positions',
          bindings: [
            read('positions', positions, 'f32'),
            read('rowValid', rowValid, 'u32'),
            write('gridPositions', gridPositions, 'f32')
          ],
          invocationCount: rowCount,
          declarations: common,
          body: `let included = rowValid[rowValidOffset + index] != 0u;
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  gridPositions[gridPositionsOffset + index * 2u] =
    select(invalid, positions[positionsOffset + index * 2u], included);
  gridPositions[gridPositionsOffset + index * 2u + 1u] =
    select(invalid, positions[positionsOffset + index * 2u + 1u], included);`
        }),
        ...gridIndex.nodes
      );
    }

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-candidates`,
        operation: OPERATION,
        variant: 'candidates',
        bindings: [
          read('positions', positions, 'f32'),
          read('predictors', predictors, 'f32'),
          read('response', response, 'f32'),
          read('rowValid', rowValid, 'u32'),
          read('params', parameters, 'f32'),
          ...gridBindings,
          write('candidateScratch', candidateScratch, 'f32')
        ],
        invocationCount: rowCount,
        declarations: fitDeclarations,
        body: `${focalRowWGSL}
  let candidateCount = getCandidateCount();
  if (rowValid[rowValidOffset + focal] == 0u) {
    // Excluded rows are skipped by the tile sums, so their scratch is never read.
    return;
  }
  var neighbors: array<f32, MAXIMUM_NEIGHBORS>;
  let isAdaptive = params[paramsOffset + 1u] > 0.5;
  if (isAdaptive) {
    selectNeighborDistances(focal, getMaximumK(candidateCount), &neighbors);
  }
  for (var candidate = 0u; candidate < candidateCount; candidate++) {
    let slot = candidateScratchOffset + (candidate * ROW_COUNT + focal) * 2u;
    candidateScratch[slot] = 0.0;
    candidateScratch[slot + 1u] = -1.0;
    let bandwidth = getCandidateBandwidth(candidate, isAdaptive, &neighbors);
    if (bandwidth <= 0.0) {
      continue;
    }
    let fit = fitLocation(focal, bandwidth);
    if (!fit.ok) {
      continue;
    }
    let residual = response[responseOffset + focal] - fit.beta[0];
    candidateScratch[slot] = residual * residual;
    candidateScratch[slot + 1u] = fit.hat;
  }`
      })
    );

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-tile-candidates`,
        operation: OPERATION,
        variant: 'tile-candidates',
        bindings: [
          read('candidateScratch', candidateScratch, 'f32'),
          read('rowValid', rowValid, 'u32'),
          write('tileCandidates', tileCandidates, 'f32')
        ],
        invocationCount: tileCount * maximumBandwidthCount,
        declarations: common,
        body: `let tile = index / LADDER;
  let candidate = index % LADDER;
  let firstRow = tile * TILE_ROWS;
  let endRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  var residualSum = 0.0;
  var traceSum = 0.0;
  var failed = 0.0;
  for (var row = firstRow; row < endRow; row++) {
    if (rowValid[rowValidOffset + row] == 0u) {
      continue;
    }
    let slot = candidateScratchOffset + (candidate * ROW_COUNT + row) * 2u;
    let hat = candidateScratch[slot + 1u];
    if (hat < 0.0) {
      failed = failed + 1.0;
    } else {
      residualSum = residualSum + candidateScratch[slot];
      traceSum = traceSum + hat;
    }
  }
  let base = tileCandidatesOffset + index * 3u;
  tileCandidates[base] = residualSum;
  tileCandidates[base + 1u] = traceSum;
  tileCandidates[base + 2u] = failed;`
      })
    );

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-select`,
        operation: OPERATION,
        variant: 'select',
        bindings: [
          read('tileCandidates', tileCandidates, 'f32'),
          read('tileResponse', tileResponse, 'f32'),
          read('tileTotal', tileTotal, 'f32'),
          read('params', parameters, 'f32'),
          write('scores', scores, 'f32'),
          write('selection', selection, 'f32'),
          write('summary', summary, 'f32')
        ],
        invocationCount: 1,
        declarations: `${common}
${CANDIDATE_COUNT_WGSL}`,
        body: `var count = 0.0;
  var total = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    count = count + tileResponse[tileResponseOffset + 2u * tile];
    total = total + tileTotal[tileTotalOffset + tile];
  }
  let candidateCount = getCandidateCount();
  let nan = getNaN();
  var bestIndex = 0u;
  var bestScore = SENTINEL;
  var bestResidual = nan;
  var bestTrace = nan;
  var hasValid = false;
  for (var candidate = 0u; candidate < LADDER; candidate++) {
    var score = nan;
    if (candidate < candidateCount) {
      var residualSum = 0.0;
      var traceSum = 0.0;
      var failed = 0.0;
      for (var tile = 0u; tile < TILE_COUNT; tile++) {
        let base = tileCandidatesOffset + (tile * LADDER + candidate) * 3u;
        residualSum = residualSum + tileCandidates[base];
        traceSum = traceSum + tileCandidates[base + 1u];
        failed = failed + tileCandidates[base + 2u];
      }
      let denominator = count - 2.0 - traceSum;
      if (count > 0.0 && failed == 0.0 && denominator > 0.0) {
        let variance = max(residualSum / count, MINIMUM_VARIANCE);
        score = count * log(variance) + count * LN_TWO_PI + count * (count + traceSum) / denominator;
        if (isFiniteBits(score) && (!hasValid || score < bestScore)) {
          hasValid = true;
          bestScore = score;
          bestIndex = candidate;
          bestResidual = residualSum;
          bestTrace = traceSum;
        }
      } else {
        score = nan;
      }
    }
    scores[scoresOffset + candidate] = score;
  }
  selection[selectionOffset] = f32(bestIndex);
  selection[selectionOffset + 1u] = params[paramsOffset + HEADER + bestIndex];
  summary[summaryOffset] = bestResidual;
  summary[summaryOffset + 1u] = bestTrace;
  summary[summaryOffset + 2u] = select(nan, bestScore, hasValid);
  summary[summaryOffset + 3u] = select(nan, 1.0 - bestResidual / total, hasValid && total > 0.0);
  summary[summaryOffset + 4u] = count;
  summary[summaryOffset + 5u] = select(0.0, 1.0, hasValid);`
      })
    );

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-fit`,
        operation: OPERATION,
        variant: 'fit',
        bindings: [
          read('positions', positions, 'f32'),
          read('predictors', predictors, 'f32'),
          read('response', response, 'f32'),
          read('rowValid', rowValid, 'u32'),
          read('params', parameters, 'f32'),
          read('selection', selection, 'f32'),
          ...gridBindings,
          write('coefficients', output.coefficients, 'f32'),
          write('locals', locals, 'u32')
        ],
        invocationCount: rowCount,
        declarations: fitDeclarations,
        body: `${focalRowWGSL}
  let nan = getNaN();
  let coefficientBase = coefficientsOffset + focal * P;
  let localBase = localsOffset + focal * LOCAL_STRIDE;
  var status = ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.SINGULAR}u;
  var localR2 = nan;
  var conditionNumber = nan;
  var fittedValue = nan;
  var residual = nan;
  var hat = nan;
  for (var column = 0u; column < P; column++) {
    coefficients[coefficientBase + column] = nan;
  }
  if (rowValid[rowValidOffset + focal] == 0u) {
    status = ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.EXCLUDED}u;
  } else {
    let candidateCount = getCandidateCount();
    let candidate = min(u32(max(selection[selectionOffset], 0.0)), LADDER - 1u);
    var neighbors: array<f32, MAXIMUM_NEIGHBORS>;
    let isAdaptive = params[paramsOffset + 1u] > 0.5;
    if (isAdaptive) {
      selectNeighborDistances(focal, getMaximumK(candidateCount), &neighbors);
    }
    let bandwidth = getCandidateBandwidth(candidate, isAdaptive, &neighbors);
    if (bandwidth > 0.0) {
      let fit = fitLocation(focal, bandwidth);
      if (fit.ok) {
        status = ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.OK}u;
        var intercept = fit.beta[0];
        for (var column = 1u; column < P; column++) {
          let slope = fit.beta[column];
          coefficients[coefficientBase + column] = slope;
          intercept = intercept - slope * predictors[predictorsOffset + focal * PREDICTOR_COUNT + column - 1u];
        }
        coefficients[coefficientBase] = intercept;
        fittedValue = fit.beta[0];
        residual = response[responseOffset + focal] - fittedValue;
        hat = fit.hat;
        localR2 = getLocalR2(focal, bandwidth, fit);
        ${output.localConditionNumber ? 'conditionNumber = getLocalConditionNumber(focal, bandwidth);' : ''}
      }
    }
  }
  locals[localBase] = bitcast<u32>(localR2);
  locals[localBase + 1u] = bitcast<u32>(fittedValue);
  locals[localBase + 2u] = bitcast<u32>(residual);
  locals[localBase + 3u] = bitcast<u32>(hat);
  locals[localBase + 4u] = status;
  locals[localBase + 5u] = bitcast<u32>(conditionNumber);`
      })
    );

    if (indexGridSize) {
      // The fit kernel visits only grid (included) rows, so excluded rows are written here.
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-fit-excluded`,
          operation: OPERATION,
          variant: 'fit-excluded',
          bindings: [
            read('rowValid', rowValid, 'u32'),
            write('coefficients', output.coefficients, 'f32'),
            write('locals', locals, 'u32')
          ],
          invocationCount: rowCount,
          declarations: common,
          body: `if (rowValid[rowValidOffset + index] != 0u) {
    return;
  }
  let nan = getNaN();
  for (var column = 0u; column < P; column++) {
    coefficients[coefficientsOffset + index * P + column] = nan;
  }
  let localBase = localsOffset + index * LOCAL_STRIDE;
  for (var word = 0u; word < LOCAL_STRIDE; word++) {
    locals[localBase + word] = bitcast<u32>(nan);
  }
  locals[localBase + 4u] = ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.EXCLUDED}u;`
        })
      );
    }

    const publishTargets: [string, GraphDataView | undefined, 'f32' | 'u32', number][] = [
      ['localR2', output.localR2, 'f32', 0],
      ['fitted', output.fitted, 'f32', 1],
      ['residuals', output.residuals, 'f32', 2],
      ['hatDiagonal', output.hatDiagonal, 'f32', 3],
      ['localStatus', output.localStatus, 'u32', 4],
      ['localConditionNumber', output.localConditionNumber, 'f32', 5]
    ];
    const present = publishTargets.filter(([, view]) => view);
    if (present.length > 0) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-publish`,
          operation: OPERATION,
          variant: 'publish',
          bindings: [
            read('locals', locals, 'u32'),
            ...present.map(([name, view, type]) => write(name, view!, type))
          ],
          invocationCount: rowCount,
          declarations: common,
          body: present
            .map(([name, , type, slot]) =>
              type === 'f32'
                ? `${name}[${name}Offset + index] = bitcast<f32>(locals[localsOffset + index * LOCAL_STRIDE + ${slot}u]);`
                : `${name}[${name}Offset + index] = locals[localsOffset + index * LOCAL_STRIDE + ${slot}u];`
            )
            .join('\n  ')
        })
      );
    }
    return nodes;
  }
}

// Needs the params binding.
const CANDIDATE_COUNT_WGSL = /* wgsl */ `
fn getCandidateCount() -> u32 {
  return min(u32(max(params[paramsOffset + 2u], 0.0)), LADDER);
}
`;

const GWR_COMMON_WGSL = /* wgsl */ `
fn getNaN() -> f32 {
  var bits = 0x7fc00000u;
  return bitcast<f32>(bits);
}

fn isFiniteBits(x: f32) -> bool {
  return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}
`;

// Needs the bindings positions, predictors, response, rowValid and params, plus the Cholesky helpers.
function getFitWGSL(
  coefficientCount: number,
  indexGridSize: readonly [number, number] | null,
  hasConditionNumber: boolean
): string {
  const P = coefficientCount;
  return /* wgsl */ `
${indexGridSize ? getGridWGSL(indexGridSize) : ''}
struct LocalFit {
  ok: bool,
  hat: f32,
  weightSum: f32,
  weightedResponse: f32,
  beta: array<f32, ${P}>
}

fn getPosition(row: u32) -> vec2f {
  return vec2f(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}

fn getWeight(distance: f32, bandwidth: f32) -> f32 {
  let ratio = distance / bandwidth;
  if (params[paramsOffset] > 0.5) {
    if (distance >= bandwidth) {
      return 0.0;
    }
    let t = 1.0 - ratio * ratio;
    return t * t;
  }
  return exp(-0.5 * ratio * ratio);
}

// Largest requested k among valid adaptive candidates, clamped to [1, MAXIMUM_NEIGHBORS].
fn getMaximumK(candidateCount: u32) -> u32 {
  var maximumK = 1u;
  for (var candidate = 0u; candidate < candidateCount; candidate++) {
    let k = params[paramsOffset + HEADER + candidate];
    if (k >= 1.0 && k <= f32(MAXIMUM_NEIGHBORS)) {
      maximumK = max(maximumK, u32(floor(k + 0.5)));
    }
  }
  return min(maximumK, MAXIMUM_NEIGHBORS);
}

${
  indexGridSize
    ? getRingNeighborDistancesWGSL()
    : `// Keeps the maximumK smallest squared distances from row i (itself included) in ascending order.
fn selectNeighborDistances(i: u32, maximumK: u32, list: ptr<function, array<f32, MAXIMUM_NEIGHBORS>>) {
  for (var slot = 0u; slot < MAXIMUM_NEIGHBORS; slot++) {
    (*list)[slot] = SENTINEL;
  }
  let origin = getPosition(i);
  for (var row = 0u; row < ROW_COUNT; row++) {
    if (rowValid[rowValidOffset + row] == 0u) {
      continue;
    }
    let delta = getPosition(row) - origin;
    let squared = dot(delta, delta);
    if (squared < (*list)[maximumK - 1u]) {
      var slot = maximumK - 1u;
      loop {
        if (slot == 0u || (*list)[slot - 1u] <= squared) {
          break;
        }
        (*list)[slot] = (*list)[slot - 1u];
        slot = slot - 1u;
      }
      (*list)[slot] = squared;
    }
  }
}`
}

// Positive bandwidth for a ladder candidate, or -1 when the candidate is invalid for this row.
fn getCandidateBandwidth(candidate: u32, isAdaptive: bool, list: ptr<function, array<f32, MAXIMUM_NEIGHBORS>>) -> f32 {
  let value = params[paramsOffset + HEADER + candidate];
  if (!isAdaptive) {
    return select(-1.0, value, value > 0.0);
  }
  if (!(value >= 2.0) || value > f32(MAXIMUM_NEIGHBORS)) {
    return -1.0;
  }
  let squared = (*list)[u32(floor(value + 0.5)) - 1u];
  if (squared >= SENTINEL) {
    return -1.0;
  }
  return sqrt(squared) * ADAPTIVE_FACTOR;
}

fn fitLocation(i: u32, bandwidth: f32) -> LocalFit {
  var a: array<f32, ${P * P}>;
  var b: array<f32, ${P}>;
  var z: array<f32, ${P}>;
  var fit: LocalFit;
  fit.ok = false;
  fit.hat = 0.0;
  fit.weightSum = 0.0;
  fit.weightedResponse = 0.0;
  for (var column = 0u; column < P; column++) {
    fit.beta[column] = 0.0;
    b[column] = 0.0;
  }
  for (var entry = 0u; entry < P * P; entry++) {
    a[entry] = 0.0;
  }
  let origin = getPosition(i);
  let originBase = predictorsOffset + i * PREDICTOR_COUNT;
  z[0] = 1.0;
  ${getCandidateLoopWGSL(
    `let delta = getPosition(row) - origin;
    let weight = getWeight(sqrt(dot(delta, delta)), bandwidth);
    if (!(weight > 0.0)) {
      continue;
    }
    let rowBase = predictorsOffset + row * PREDICTOR_COUNT;
    for (var column = 0u; column < PREDICTOR_COUNT; column++) {
      z[column + 1u] = predictors[rowBase + column] - predictors[originBase + column];
    }
    let y = response[responseOffset + row];
    for (var r = 0u; r < P; r++) {
      let weighted = weight * z[r];
      b[r] = b[r] + weighted * y;
      for (var c = 0u; c <= r; c++) {
        a[r * P + c] = a[r * P + c] + weighted * z[c];
      }
    }`,
    Boolean(indexGridSize)
  )}
  fit.weightSum = a[0];
  fit.weightedResponse = b[0];
  var scale: array<f32, ${P}>;
  for (var r = 0u; r < P; r++) {
    let diagonal = a[r * P + r];
    if (!(diagonal > 0.0) || !isFiniteBits(diagonal)) {
      return fit;
    }
    scale[r] = 1.0 / sqrt(diagonal);
  }
  for (var r = 0u; r < P; r++) {
    for (var c = 0u; c <= r; c++) {
      a[r * P + c] = a[r * P + c] * scale[r] * scale[c];
    }
    b[r] = b[r] * scale[r];
  }
  if (!choleskyFactor_${P}(&a)) {
    return fit;
  }
  let inverse00 = choleskyInverseDiagonal_${P}(&a, 0u);
  choleskySolve_${P}(&a, &b);
  var finite = isFiniteBits(inverse00);
  for (var r = 0u; r < P; r++) {
    fit.beta[r] = b[r] * scale[r];
    finite = finite && isFiniteBits(fit.beta[r]);
  }
  fit.hat = inverse00 * scale[0] * scale[0];
  fit.ok = finite && fit.hat >= 0.0;
  return fit;
}

// Weighted R^2 of the local fit over the neighborhood of row i.
fn getLocalR2(i: u32, bandwidth: f32, fit: LocalFit) -> f32 {
  let origin = getPosition(i);
  let originBase = predictorsOffset + i * PREDICTOR_COUNT;
  let weightedMean = fit.weightedResponse / fit.weightSum;
  var residualSum = 0.0;
  var totalSum = 0.0;
  ${getCandidateLoopWGSL(
    `let delta = getPosition(row) - origin;
    let weight = getWeight(sqrt(dot(delta, delta)), bandwidth);
    if (!(weight > 0.0)) {
      continue;
    }
    let rowBase = predictorsOffset + row * PREDICTOR_COUNT;
    var prediction = fit.beta[0];
    for (var column = 0u; column < PREDICTOR_COUNT; column++) {
      prediction = prediction + fit.beta[column + 1u] * (predictors[rowBase + column] - predictors[originBase + column]);
    }
    let y = response[responseOffset + row];
    residualSum = residualSum + weight * (y - prediction) * (y - prediction);
    totalSum = totalSum + weight * (y - weightedMean) * (y - weightedMean);`,
    Boolean(indexGridSize)
  )}
  return select(getNaN(), 1.0 - residualSum / totalSum, totalSum > 0.0);
}
${hasConditionNumber ? getConditionNumberWGSL(P, indexGridSize) : ''}
`;
}

/**
 * Loop over the candidate rows of a location (`origin`, `bandwidth` in scope) running `body` with
 * `row` bound. Without a grid every row is visited in order. With one, bounded fixed-bandwidth
 * fits (bisquare, fixed) visit only the sorted contents of the cells covering the bandwidth, one
 * contiguous slot range per grid row; other encodings still visit every row.
 */
function getCandidateLoopWGSL(body: string, hasGrid: boolean): string {
  if (!hasGrid) {
    return `for (var row = 0u; row < ROW_COUNT; row++) {
    if (rowValid[rowValidOffset + row] == 0u) {
      continue;
    }
    ${body}
  }`;
  }
  return `let useGrid = isGridBandwidth(bandwidth);
  let range = select(vec4u(0u, 0u, 0u, 0u), getGridRange(origin, bandwidth), useGrid);
  let gridRowCount = select(1u, range.w - range.z + 1u, useGrid);
  for (var gridRowStep = 0u; gridRowStep < gridRowCount; gridRowStep++) {
    var firstSlot = 0u;
    var endSlot = ROW_COUNT;
    if (useGrid) {
      let cellRow = (range.z + gridRowStep) * INDEX_WIDTH;
      firstSlot = cellOffsets[cellOffsetsOffset + cellRow + range.x];
      endSlot = cellOffsets[cellOffsetsOffset + cellRow + range.y + 1u];
    }
    for (var slot = firstSlot; slot < endSlot; slot++) {
      var row = slot;
      if (useGrid) {
        row = sortedIds[sortedIdsOffset + slot];
      }
      if (rowValid[rowValidOffset + row] == 0u) {
        continue;
      }
      ${body}
    }
  }`;
}

/**
 * `getLocalConditionNumber`: mgwr `local_collinearity` condition number. With weights `w_j`
 * (kernel weights, not their square roots, exactly as mgwr's `xw = x * wi`) and the raw design
 * `x_j = (1, X_j)`, the singular values of the column-normalized `diag(w) x` are the square roots
 * of the eigenvalues of `C = D^-1/2 (sum_j w_j^2 x_j x_j') D^-1/2`, `D = diag(sum_j w_j^2 x_jc^2)`.
 * The moment is accumulated in coordinates centered on the focal row (as the fit is) and shifted
 * to raw coordinates afterwards; the eigenvalues of the unit-diagonal `C` come from cyclic Jacobi
 * sweeps. A near-singular `C` returns a very large number rather than NaN.
 */
function getConditionNumberWGSL(
  coefficientCount: number,
  indexGridSize: readonly [number, number] | null
): string {
  const P = coefficientCount;
  return /* wgsl */ `
fn getLocalConditionNumber(i: u32, bandwidth: f32) -> f32 {
  var moment: array<f32, ${P * P}>;
  var z: array<f32, ${P}>;
  for (var entry = 0u; entry < P * P; entry++) {
    moment[entry] = 0.0;
  }
  let origin = getPosition(i);
  let originBase = predictorsOffset + i * PREDICTOR_COUNT;
  z[0] = 1.0;
  ${getCandidateLoopWGSL(
    `let delta = getPosition(row) - origin;
    let weight = getWeight(sqrt(dot(delta, delta)), bandwidth);
    if (!(weight > 0.0)) {
      continue;
    }
    let squaredWeight = weight * weight;
    let rowBase = predictorsOffset + row * PREDICTOR_COUNT;
    for (var column = 0u; column < PREDICTOR_COUNT; column++) {
      z[column + 1u] = predictors[rowBase + column] - predictors[originBase + column];
    }
    for (var r = 0u; r < P; r++) {
      for (var c = 0u; c <= r; c++) {
        moment[r * P + c] = moment[r * P + c] + squaredWeight * z[r] * z[c];
      }
    }`,
    Boolean(indexGridSize)
  )}
  // Shift the centered moment to raw coordinates: x = (1, x_i + z), so raw = T moment T'.
  var center: array<f32, ${P}>;
  center[0] = 0.0;
  for (var column = 0u; column < PREDICTOR_COUNT; column++) {
    center[column + 1u] = predictors[originBase + column];
  }
  var raw: array<f32, ${P * P}>;
  for (var r = 0u; r < P; r++) {
    for (var c = 0u; c <= r; c++) {
      // T = I + e_r' center, with center[0] = 0 and the intercept row of T equal to e_0.
      let m00 = moment[0];
      let mr0 = moment[r * P];
      let mc0 = moment[c * P];
      var value = moment[r * P + c];
      if (r > 0u && c > 0u) {
        value = value + center[r] * mc0 + center[c] * mr0 + center[r] * center[c] * m00;
      } else if (r > 0u) {
        value = value + center[r] * m00;
      }
      raw[r * P + c] = value;
      raw[c * P + r] = value;
    }
  }
  var scale: array<f32, ${P}>;
  for (var r = 0u; r < P; r++) {
    let diagonal = raw[r * P + r];
    if (!(diagonal > 0.0) || !isFiniteBits(diagonal)) {
      return getNaN();
    }
    scale[r] = 1.0 / sqrt(diagonal);
  }
  for (var r = 0u; r < P; r++) {
    for (var c = 0u; c < P; c++) {
      raw[r * P + c] = raw[r * P + c] * scale[r] * scale[c];
    }
  }
  // Cyclic Jacobi sweeps on the symmetric unit-diagonal matrix.
  for (var sweep = 0u; sweep < 12u; sweep++) {
    var off = 0.0;
    for (var p = 0u; p + 1u < P; p++) {
      for (var q = p + 1u; q < P; q++) {
        let apq = raw[p * P + q];
        off = off + apq * apq;
        if (abs(apq) < 1.0e-30) {
          continue;
        }
        let theta = (raw[q * P + q] - raw[p * P + p]) / (2.0 * apq);
        let tangent = select(-1.0, 1.0, theta >= 0.0) / (abs(theta) + sqrt(theta * theta + 1.0));
        let cosine = 1.0 / sqrt(tangent * tangent + 1.0);
        let sine = tangent * cosine;
        for (var k = 0u; k < P; k++) {
          let akp = raw[k * P + p];
          let akq = raw[k * P + q];
          raw[k * P + p] = cosine * akp - sine * akq;
          raw[k * P + q] = sine * akp + cosine * akq;
        }
        for (var k = 0u; k < P; k++) {
          let apk = raw[p * P + k];
          let aqk = raw[q * P + k];
          raw[p * P + k] = cosine * apk - sine * aqk;
          raw[q * P + k] = sine * apk + cosine * aqk;
        }
      }
    }
    if (off < 1.0e-14) {
      break;
    }
  }
  var largest = raw[0];
  var smallest = raw[0];
  for (var r = 1u; r < P; r++) {
    largest = max(largest, raw[r * P + r]);
    smallest = min(smallest, raw[r * P + r]);
  }
  return sqrt(largest / max(smallest, 1.0e-30));
}
`;
}

function getGridWGSL(indexGridSize: readonly [number, number]): string {
  return /* wgsl */ `
${getGridLookupWGSL(indexGridSize)}

// Bisquare has bounded support (fixed or adaptive), so the grid is exact for it.
fn isGridBandwidth(bandwidth: f32) -> bool {
  return params[paramsOffset] > 0.5 && bandwidth > 0.0 && isFiniteBits(bandwidth);
}

// Included rows in grid-cell order: invocation \`slot\` fits the slot-th indexed row.
fn getFocalCount() -> u32 {
  return min(cellOffsets[cellOffsetsOffset + INDEX_WIDTH * INDEX_HEIGHT], ROW_COUNT);
}

fn getFocalRow(slot: u32) -> u32 {
  return sortedIds[sortedIdsOffset + slot];
}
`;
}
