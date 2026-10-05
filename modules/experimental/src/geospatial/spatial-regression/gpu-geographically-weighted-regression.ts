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
import {getCholeskyWGSL} from './spatial-regression-solve';
import {
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_ADAPTIVE_BANDWIDTH_FACTOR,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_LADDER_LENGTH,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_NEIGHBOR_COUNT,
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
const LOCAL_STRIDE = 5;

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
 * Complexity: one thread per location scans every row once per candidate (and once more to select
 * k-th nearest distances and to evaluate the final fit), so the work is `O(n^2 * ladder * p^2)`
 * with `n <= 65536`. No grid is used; scans run in row order so every sum has a fixed order. It is
 * a first version meant for thousands of locations. Candidate RSS and trace are reduced by
 * 256-row tiles merged in tile order. No float atomics: results are bitwise reproducible on one
 * adapter.
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
    if (this.rowCount > GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_ROW_COUNT) {
      throw new Error(
        `${id} supports at most ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_MAXIMUM_ROW_COUNT} rows`
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
    const {props, id, rowCount, coefficientCount, maximumBandwidthCount, maximumNeighborCount} =
      this;
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
    const transient = (name: string, format: 'uint32' | 'float32', length: number) =>
      createTransientView(graph, `${id}-${name}`, format, Math.max(length, 1));
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
${getFitWGSL(P)}`;
    const mask = props.mask;
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
          write('candidateScratch', candidateScratch, 'f32')
        ],
        invocationCount: rowCount,
        declarations: fitDeclarations,
        body: `let candidateCount = getCandidateCount();
  if (rowValid[rowValidOffset + index] == 0u) {
    for (var candidate = 0u; candidate < candidateCount; candidate++) {
      let slot = candidateScratchOffset + (candidate * ROW_COUNT + index) * 2u;
      candidateScratch[slot] = 0.0;
      candidateScratch[slot + 1u] = 0.0;
    }
    return;
  }
  var neighbors: array<f32, MAXIMUM_NEIGHBORS>;
  let isAdaptive = params[paramsOffset + 1u] > 0.5;
  if (isAdaptive) {
    selectNeighborDistances(index, getMaximumK(candidateCount), &neighbors);
  }
  for (var candidate = 0u; candidate < candidateCount; candidate++) {
    let slot = candidateScratchOffset + (candidate * ROW_COUNT + index) * 2u;
    candidateScratch[slot] = 0.0;
    candidateScratch[slot + 1u] = -1.0;
    let bandwidth = getCandidateBandwidth(candidate, isAdaptive, &neighbors);
    if (bandwidth <= 0.0) {
      continue;
    }
    let fit = fitLocation(index, bandwidth);
    if (!fit.ok) {
      continue;
    }
    let residual = response[responseOffset + index] - fit.beta[0];
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
          write('coefficients', output.coefficients, 'f32'),
          write('locals', locals, 'u32')
        ],
        invocationCount: rowCount,
        declarations: fitDeclarations,
        body: `let nan = getNaN();
  let coefficientBase = coefficientsOffset + index * P;
  let localBase = localsOffset + index * LOCAL_STRIDE;
  var status = ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.SINGULAR}u;
  var localR2 = nan;
  var fittedValue = nan;
  var residual = nan;
  var hat = nan;
  for (var column = 0u; column < P; column++) {
    coefficients[coefficientBase + column] = nan;
  }
  if (rowValid[rowValidOffset + index] == 0u) {
    status = ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.EXCLUDED}u;
  } else {
    let candidateCount = getCandidateCount();
    let candidate = min(u32(max(selection[selectionOffset], 0.0)), LADDER - 1u);
    var neighbors: array<f32, MAXIMUM_NEIGHBORS>;
    let isAdaptive = params[paramsOffset + 1u] > 0.5;
    if (isAdaptive) {
      selectNeighborDistances(index, getMaximumK(candidateCount), &neighbors);
    }
    let bandwidth = getCandidateBandwidth(candidate, isAdaptive, &neighbors);
    if (bandwidth > 0.0) {
      let fit = fitLocation(index, bandwidth);
      if (fit.ok) {
        status = ${GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.OK}u;
        var intercept = fit.beta[0];
        for (var column = 1u; column < P; column++) {
          let slope = fit.beta[column];
          coefficients[coefficientBase + column] = slope;
          intercept = intercept - slope * predictors[predictorsOffset + index * PREDICTOR_COUNT + column - 1u];
        }
        coefficients[coefficientBase] = intercept;
        fittedValue = fit.beta[0];
        residual = response[responseOffset + index] - fittedValue;
        hat = fit.hat;
        localR2 = getLocalR2(index, bandwidth, fit);
      }
    }
  }
  locals[localBase] = bitcast<u32>(localR2);
  locals[localBase + 1u] = bitcast<u32>(fittedValue);
  locals[localBase + 2u] = bitcast<u32>(residual);
  locals[localBase + 3u] = bitcast<u32>(hat);
  locals[localBase + 4u] = status;`
      })
    );

    const publishTargets: [string, GraphDataView | undefined, 'f32' | 'u32', number][] = [
      ['localR2', output.localR2, 'f32', 0],
      ['fitted', output.fitted, 'f32', 1],
      ['residuals', output.residuals, 'f32', 2],
      ['hatDiagonal', output.hatDiagonal, 'f32', 3],
      ['localStatus', output.localStatus, 'u32', 4]
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
function getFitWGSL(coefficientCount: number): string {
  const P = coefficientCount;
  return /* wgsl */ `
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

// Keeps the maximumK smallest squared distances from row i (itself included) in ascending order.
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
  for (var row = 0u; row < ROW_COUNT; row++) {
    if (rowValid[rowValidOffset + row] == 0u) {
      continue;
    }
    let delta = getPosition(row) - origin;
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
    }
  }
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
  for (var row = 0u; row < ROW_COUNT; row++) {
    if (rowValid[rowValidOffset + row] == 0u) {
      continue;
    }
    let delta = getPosition(row) - origin;
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
    totalSum = totalSum + weight * (y - weightedMean) * (y - weightedMean);
  }
  return select(getNaN(), 1.0 - residualSum / totalSum, totalSum > 0.0);
}
`;
}
