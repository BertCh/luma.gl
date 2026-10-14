// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {getWGSLFloatLiteral, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {ERFC_VALUE_WGSL, GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../utils/wgsl-common';
import {type GPUSpatialWeights, validateGPUSpatialWeights} from '../spatial-weights/index';
import type {GPUSolverStatusPort} from '../../analysis-core/contracts/index';
import {GPU_REGRESSION_PRECISION_CONDITION_LIMIT} from '../../analysis-core/contracts/model-result';
import {
  GPUSpatialWeightsTraces,
  validateGPUSpatialWeightsTracesParameters
} from './gpu-spatial-weights-traces';
import {
  createKernelFactory,
  createSeriesProbeNodes,
  getF32Binding,
  getU32Binding,
  SPATIAL_MAXIMUM_LIKELIHOOD_TILE_WORKGROUP_SIZE
} from './spatial-maximum-likelihood-nodes';
import {
  GPU_SPATIAL_ERROR_MAXIMUM_LIKELIHOOD_MAXIMUM_PREDICTOR_COUNT,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_BOUND_FRACTION,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_DEFAULT_MAXIMUM_POWER,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_MINIMUM_POWER,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_AIC,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_APPROXIMATION,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_AT_BOUNDARY,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CONDITION_NUMBER,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CONVERGED,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CORRELATION_CONDITION_NUMBER,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOCAL_MAXIMUM_COUNT,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOG_DETERMINANT,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOG_LIKELIHOOD,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PARAMETER,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PARAMETER_BOUND,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PRECISION_WARNING,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PSEUDO_R_SQUARED,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PSEUDO_R_SQUARED_SPATIAL,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_RESIDUAL_SUM_OF_SQUARES,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_ROW_COUNT,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SCHWARZ,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SERIES_LIMITED,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SIGMA_SQUARED,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_TRUNCATION_BOUND,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_VARIANCE_DEFINED,
  GPU_SPATIAL_MAXIMUM_LIKELIHOOD_TABLE_STRIDE,
  getSpatialMaximumLikelihoodSeriesLimit
} from './spatial-maximum-likelihood-parameters';
import {
  getKahanSumWGSL,
  getParameterSearchWGSL,
  getTableRowWGSL,
  getTraceSeriesWGSL
} from './spatial-maximum-likelihood-wgsl';
import {
  getBelsleyConditionNumberWGSL,
  getCholeskyWGSL,
  getSymmetricEigenvalueRangeWGSL
} from './spatial-regression-solve';
import {
  GPU_SPATIAL_WEIGHTS_TRACES_DEFAULT_PROBE_COUNT,
  GPU_SPATIAL_WEIGHTS_TRACES_EXACT_ROW_LIMIT,
  GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_EXACT_ROW_COUNT,
  getSpatialLogDeterminantWGSL,
  type GPUSpatialWeightsTracesMode
} from './spatial-weights-traces-parameters';

const OPERATION = 'GPUSpatialErrorMaximumLikelihood';
const MAXIMUM_DEFAULT_TILE_COUNT = 4096;
const MINIMUM_DEFAULT_TILE_ROWS = 64;
const TILE_WORKGROUP_SIZE = SPATIAL_MAXIMUM_LIKELIHOOD_TILE_WORKGROUP_SIZE;
const LN_TWO_PI = Math.log(2 * Math.PI);
/** Residual sums: `u'u`, `sum u`, `sum (y - ybar) u`, `sum (y - ybar)^2`. */
const RESIDUAL_SUM_COUNT = 4;

/** Caller-owned outputs of {@link GPUSpatialErrorMaximumLikelihood}. */
export type GPUSpatialErrorMaximumLikelihoodOutput = {
  /**
   * One-row solver `code` (0 on success) and, optionally, `converged` (1 when the optimum is an
   * interior stationary point).
   */
  status: Pick<GPUSolverStatusPort, 'code'> & Partial<Pick<GPUSolverStatusPort, 'converged'>>;

  /**
   * `(predictorCount + 2) * 4` float32 values: one row per coefficient in spreg order (intercept,
   * one per predictor, then `lambda` last), each row holding the estimate, standard error, z
   * statistic and two-sided normal p-value.
   */
  table: GraphDataView<'float32'>;
  /** At least 22 float32 values; see the `GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_*` slots. */
  summary: GraphDataView<'float32'>;
  /** Optional per-row residuals `u = y - X b` (spreg `u`); NaN for a failed fit. */
  residuals?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUSpatialErrorMaximumLikelihood}.
 *
 * Per-frame (no recompile): the contents of every input buffer. Topology (needs a new graph):
 * `predictorCount`, the row count, the weights views, `maximumPower`, the trace options,
 * `tileRowCount` and whether `residuals` is present.
 */
export type GPUSpatialErrorMaximumLikelihoodProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'spatial-error-maximum-likelihood'`. */
  id?: string;
  /**
   * Square weights `W`. Any sparsity pattern works (no symmetry is required) and rows need not be
   * standardized: `lambda` is searched within `min(0.99, seriesLimit) / ||W||_inf`.
   */
  weights: GPUSpatialWeights;
  /** Row-major predictors, `predictors[row * predictorCount + column]`, without an intercept. */
  predictors: GraphDataView<'float32'>;
  /** Response `y`, one value per row. */
  response: GraphDataView<'float32'>;
  /** Number of predictor columns, compile-time, 1 to 15. The intercept is implicit. */
  predictorCount: number;
  /** Series length `Q` of the log-determinant and trace series, an integer in `[2, 256]`. Defaults to 128. */
  maximumPower?: number;
  /**
   * `'exact'` or `'stochastic'` (Hutchinson) traces. Defaults to exact up to 4096 rows and
   * stochastic above, where the likelihood, hence `lambda`, carries stochastic error and
   * `GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_APPROXIMATION` is 1.
   */
  traceMode?: GPUSpatialWeightsTracesMode;
  /** Hutchinson probe count of the stochastic mode (traces and the `tr3` pass). Defaults to 64. */
  traceProbeCount?: number;
  /**
   * Optional per-frame uint32 view of `GPU_SPATIAL_WEIGHTS_TRACES_PARAMETER_LENGTH` words written
   * with `getGPUSpatialWeightsTracesParameterValues({seed})`: the Philox seed of the stochastic
   * probes, read on the GPU. Omitted means seed 0.
   */
  traceParameters?: GraphDataView<'uint32'>;
  /**
   * Rows reduced by one invocation of the accumulation passes. Defaults to
   * `max(64, ceil(rows / 4096))`.
   */
  tileRowCount?: number;
  /** Caller-owned outputs. */
  output: GPUSpatialErrorMaximumLikelihoodOutput;
};

/**
 * Maximum-likelihood estimator of the spatial error model `y = X b + u`, `u = lambda W u + e`,
 * `e ~ N(0, s2 I)` (Anselin 1988; spreg `ML_Error` with `method='full'`, whose exact
 * log-determinant is replaced here by a truncated trace series). Nothing is read back.
 *
 * 1. With `Z = [1, X - mean, y - mean]` (`m = predictorCount + 2` columns), three tile passes with
 *    Kahan merges accumulate `G0 = Z'Z`, `G1 + G1' = Z'WZ + (Z'WZ)'` and `G2 = (WZ)'(WZ)` once.
 *    The Gram matrix of the filtered columns `Z - lambda W Z` is then exactly
 *    `G0 - lambda (G1 + G1') + lambda^2 G2` for any `lambda` (the filtered constant
 *    `1 - lambda * sum_j w_ij` is column 0, as in spreg), so no pass over the rows repeats.
 * 2. For each `lambda` one small unit-scaled Cholesky solve of the filtered normal equations gives
 *    `s2(lambda) = ee(lambda) / n`, `ee = yy - 2 b'r + b'Ab` and, by the envelope theorem,
 *    `ee'(lambda) = yy' - 2 b'r' + b'A'b`. With the log-determinant series of
 *    `GPUSpatialWeightsTraces` the concentrated likelihood `-n/2 ln(s2) + ln|I - lambda W|` and its
 *    derivative are scanned on 1025 grid points, each positive-to-non-positive sign change is
 *    refined by 32 bisection steps and the best candidate wins (the same deterministic search as
 *    `GPUSpatialLagMaximumLikelihood`).
 * 3. At the estimate the filtered regression gives the betas (centered form mapped back to the
 *    uncentered intercept), `s2 = ee / n` and `Var(b) = s2 (Xs'Xs)^-1`; `u = y - X b`.
 *    `Var(lambda) = 1 / (tr2 + tr3 - 2 tr1^2 / n)` (the Schur complement of spreg's 2 x 2
 *    information matrix of `(lambda, s2)`), with the traces of `W (I - lambda W)^-1`: `tr1`, `tr2`
 *    from the power traces and `tr3` from the probe pass at the estimated `lambda`
 *    (`createSeriesProbeNodes`). Like spreg, the coefficient standard errors treat `lambda` as
 *    known and the covariance of the betas with `lambda` is zero.
 * 4. The full log-likelihood is `-n/2 ln(s2) + ln|I - lambda W| - n/2 (ln 2 pi + 1)` (spreg
 *    `logll`); `pr2` is the squared correlation of `y` and `X b`. spreg's `ML_Error` counts the
 *    intercept and predictors only (not `lambda`) in AIC and Schwarz (`k = x.shape[1]`), so the
 *    summary AIC and Schwarz use that count: `2 (predictorCount + 1) - 2 LL`. A criterion that
 *    counts `lambda` and `s2` differs by a constant `2` / `ln(n)` per omitted parameter.
 *
 * Search interval, truncation and approximation behave exactly as in
 * `GPUSpatialLagMaximumLikelihood` (`min(0.99, seriesLimit) / ||W||_inf`, default `Q = 128`, exact
 * traces up to 4096 rows). Conditioning reported is that of the final filtered regression: the
 * Belsley number of its raw `[c, X - lambda W X]` Gram matrix, the correlation condition number of
 * the unit-scaled filtered centered system and its largest predictor variance inflation factor.
 *
 * The float32 traces `tr(W^q)` grow like `n ||W||_inf^q`: for a raw (unstandardized) `W` choose
 * `maximumPower` with `||W||_inf^maximumPower < 1e36`, else they overflow and the status is 3.
 * Inputs must be finite: a non-finite predictor, response or weight gives status 3 (NaN outputs).
 * At most 15 predictors (the 16-wide Cholesky); no kernel binds more than 8 storage buffers.
 */
export class GPUSpatialErrorMaximumLikelihood implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialErrorMaximumLikelihoodProps;
  /** Number of rows. */
  readonly rowCount: number;
  /** Resolved series length `Q`. */
  readonly maximumPower: number;
  /** Resolved trace mode. */
  readonly traceMode: GPUSpatialWeightsTracesMode;
  /** Rows reduced by one accumulation invocation. */
  readonly tileRowCount: number;
  /** Number of accumulation tiles. */
  readonly tileCount: number;

  constructor(props: GPUSpatialErrorMaximumLikelihoodProps) {
    const id = props.id ?? 'spatial-error-maximum-likelihood';
    this.id = id;
    this.props = props;
    const {predictorCount, output} = props;
    validateCapacity(
      id,
      'predictorCount',
      predictorCount,
      1,
      GPU_SPATIAL_ERROR_MAXIMUM_LIKELIHOOD_MAXIMUM_PREDICTOR_COUNT
    );
    const maximumPower = props.maximumPower ?? GPU_SPATIAL_MAXIMUM_LIKELIHOOD_DEFAULT_MAXIMUM_POWER;
    validateCapacity(
      id,
      'maximumPower',
      maximumPower,
      GPU_SPATIAL_MAXIMUM_LIKELIHOOD_MINIMUM_POWER,
      256
    );
    this.maximumPower = maximumPower;
    const rows = validateGPUSpatialWeights(id, props.weights);
    validatePackedView(props.predictors, ['float32'], `${id} predictors`);
    validatePackedView(props.response, ['float32'], `${id} response`);
    if (props.response.length !== rows) {
      throw new Error(`${id} response length must equal the weights row count`);
    }
    if (props.predictors.length !== rows * predictorCount) {
      throw new Error(`${id} predictors length must equal the weights row count * predictorCount`);
    }
    if (
      props.traceMode !== undefined &&
      props.traceMode !== 'exact' &&
      props.traceMode !== 'stochastic'
    ) {
      throw new Error(`${id} traceMode must be 'exact' or 'stochastic'`);
    }
    this.traceMode =
      props.traceMode ??
      (rows <= GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_EXACT_ROW_COUNT ? 'exact' : 'stochastic');
    if (this.traceMode === 'exact' && rows > GPU_SPATIAL_WEIGHTS_TRACES_EXACT_ROW_LIMIT) {
      throw new Error(`${id} exact traces support at most 16384 rows; use traceMode 'stochastic'`);
    }
    if (props.traceProbeCount !== undefined) {
      validateCapacity(id, 'traceProbeCount', props.traceProbeCount);
    }
    validateGPUSpatialWeightsTracesParameters(id, 'traceParameters', props.traceParameters);
    validatePackedView(output.table, ['float32'], `${id} output.table`);
    if (output.table.length < (predictorCount + 2) * GPU_SPATIAL_MAXIMUM_LIKELIHOOD_TABLE_STRIDE) {
      throw new Error(`${id} output.table must hold (predictorCount + 2) * 4 float32 values`);
    }
    validatePackedView(output.summary, ['float32'], `${id} output.summary`);
    if (output.summary.length < GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH) {
      throw new Error(
        `${id} output.summary must hold ${GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH} float32 values`
      );
    }
    validatePackedView(output.status.code, ['uint32'], `${id} output.status.code`);
    if (output.status.code.length < 1) {
      throw new Error(`${id} output.status.code must hold one uint32 row`);
    }
    if (output.status.converged) {
      validatePackedView(output.status.converged, ['uint32'], `${id} output.status.converged`);
      if (output.status.converged.length < 1) {
        throw new Error(`${id} output.status.converged must hold one uint32 row`);
      }
    }
    if (output.residuals) {
      validatePackedView(output.residuals, ['float32'], `${id} output.residuals`);
      if (output.residuals.length < rows) {
        throw new Error(`${id} output.residuals must hold one row per input row`);
      }
    }
    if (props.tileRowCount !== undefined) {
      validateCapacity(id, 'tileRowCount', props.tileRowCount);
    }
    this.rowCount = rows;
    this.tileRowCount =
      props.tileRowCount ??
      Math.max(MINIMUM_DEFAULT_TILE_ROWS, Math.ceil(rows / MAXIMUM_DEFAULT_TILE_COUNT));
    this.tileCount = Math.ceil(rows / this.tileRowCount);
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.table, output.summary, output.status.code, output.status.converged, output.residuals],
      [
        props.predictors,
        props.response,
        props.weights.offsets,
        props.weights.neighbors,
        props.weights.weights
      ]
    );
  }

  /**
   * Returns the weights-norm and means passes, `GPUSpatialWeightsTraces`, the three Gram passes
   * and merge, the `lambda` search, the filtered solve, the residual passes, the `tr3` probe pass,
   * the variance and the finish kernel.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rowCount, tileRowCount, tileCount, maximumPower} = this;
    const {weights, output, predictorCount: k} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.predictors,
      props.response,
      weights.offsets,
      weights.neighbors,
      weights.weights,
      output.table,
      output.summary,
      output.status.code,
      output.status.converged,
      output.residuals
    ]);
    const p = k + 1;
    const columns = k + 2;
    const packedCount = (columns * (columns + 1)) / 2;
    const stochastic = this.traceMode === 'stochastic';

    let slotCount = 0;
    const slots: Record<string, number> = {};
    const allocate = (name: string, count = 1) => {
      slots[name] = slotCount;
      slotCount += count;
    };
    allocate('NORM');
    allocate('BOUND');
    allocate('PARAMETER');
    allocate('VALID');
    allocate('CANDIDATES');
    allocate('AT_BOUNDARY');
    allocate('SERIES_LIMITED');
    allocate('STATUS');
    allocate('MEANS', k);
    allocate('YBAR');
    allocate('BETA', p);
    allocate('SIGMA2');
    allocate('CONDITIONING', 4);
    allocate('TRACES', 3);
    allocate('STANDARD_ERRORS', p + 1);
    allocate('VARIANCE_DEFINED');
    const slotConstants = Object.entries(slots)
      .map(([name, value]) => `const W_${name}: u32 = ${value}u;`)
      .join('\n');

    const transient = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'float32', length);
    const kernel = createKernelFactory(graph, id, OPERATION);
    const workspace = transient('workspace', slotCount);
    const constants = `const ROWS: u32 = ${rowCount}u;
const K: u32 = ${k}u;
const P: u32 = ${p}u;
const M: u32 = ${columns}u;
const PACK: u32 = ${packedCount}u;
const TILE_ROWS: u32 = ${tileRowCount}u;
const TILE_COUNT: u32 = ${tileCount}u;
${slotConstants}`;
    const finiteDeclarations = `${GET_NAN_WGSL}
${IS_FINITE_VALUE_WGSL}`;
    const f32 = getF32Binding;
    const weightBindings: WGSLKernelBinding[] = [
      getU32Binding('offsets', weights.offsets, 'read'),
      getU32Binding('neighbors', weights.neighbors, 'read'),
      f32('weights', weights.weights, 'read')
    ];

    const powerTraces = transient('power-traces', maximumPower);
    const powerSums = transient('power-sums', maximumPower);
    const infinityNorm = transient('traces-infinity-norm', 1);
    const partialNorm = transient('partial-weights-norm', tileCount);
    const partialMeans = transient('partial-means', tileCount * (k + 1));
    const partialGram = transient('partial-gram', 3 * tileCount * packedCount);
    const gram = transient('gram', 3 * packedCount);
    const residualView = output.residuals ?? transient('residuals', rowCount);
    const partialResiduals = transient('partial-residuals', tileCount * RESIDUAL_SUM_COUNT);
    const residualSums = transient('residual-sums', RESIDUAL_SUM_COUNT);

    const getRowWGSL = /* wgsl */ `
fn getRow(row: u32, z: ptr<function, array<f32, ${columns}>>) {
  (*z)[0] = 1.0;
  for (var column = 0u; column < K; column++) {
    (*z)[1u + column] = design[designOffset + row * K + column] - workspace[workspaceOffset + W_MEANS + column];
  }
  (*z)[K + 1u] = response[responseOffset + row] - workspace[workspaceOffset + W_YBAR];
}`;
    const gramTiles = (mode: 0 | 1 | 2) =>
      kernel(
        `gram-tiles-${mode}`,
        [
          ...weightBindings,
          f32('design', props.predictors, 'read'),
          f32('response', props.response, 'read'),
          f32('workspace', workspace, 'read'),
          f32('partial', partialGram, 'read_write')
        ],
        tileCount,
        `${constants}
${getRowWGSL}`,
        `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${packedCount}>;
  for (var row = firstRow; row < lastRow; row++) {
    var z: array<f32, ${columns}>;
    getRow(row, &z);
${
  mode === 0
    ? `    for (var first = 0u; first < M; first++) {
      for (var second = 0u; second <= first; second++) {
        sums[first * (first + 1u) / 2u + second] += z[first] * z[second];
      }
    }`
    : `    var wz: array<f32, ${columns}>;
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor >= ROWS) {
        continue;
      }
      let weight = weights[weightsOffset + slot];
      var zn: array<f32, ${columns}>;
      getRow(neighbor, &zn);
      for (var column = 0u; column < M; column++) {
        wz[column] += weight * zn[column];
      }
    }
    for (var first = 0u; first < M; first++) {
      for (var second = 0u; second <= first; second++) {
        sums[first * (first + 1u) / 2u + second] += ${mode === 1 ? 'z[first] * wz[second] + z[second] * wz[first]' : 'wz[first] * wz[second]'};
      }
    }`
}
  }
  for (var slot = 0u; slot < PACK; slot++) {
    partial[partialOffset + (${mode}u * TILE_COUNT + index) * PACK + slot] = sums[slot];
  }`,
        {workgroupSize: TILE_WORKGROUP_SIZE}
      );

    const nodes: GPUCommandNode<Parameters>[] = [
      kernel(
        'weights-norm-tiles',
        [...weightBindings, f32('partial', partialNorm, 'read_write')],
        tileCount,
        constants,
        `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var largest = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    var rowSum = 0.0;
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      if (neighbors[neighborsOffset + slot] < ROWS) {
        rowSum += abs(weights[weightsOffset + slot]);
      }
    }
    largest = max(largest, rowSum);
  }
  partial[partialOffset + index] = largest;`,
        {workgroupSize: TILE_WORKGROUP_SIZE}
      ),
      kernel(
        'means-tiles',
        [
          f32('design', props.predictors, 'read'),
          f32('response', props.response, 'read'),
          f32('partial', partialMeans, 'read_write')
        ],
        tileCount,
        constants,
        `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${k + 1}>;
  for (var row = firstRow; row < lastRow; row++) {
    for (var column = 0u; column < K; column++) {
      sums[column] = sums[column] + design[designOffset + row * K + column];
    }
    sums[K] = sums[K] + response[responseOffset + row];
  }
  for (var slot = 0u; slot < ${k + 1}u; slot++) {
    partial[partialOffset + index * ${k + 1}u + slot] = sums[slot];
  }`,
        {workgroupSize: TILE_WORKGROUP_SIZE}
      ),
      kernel(
        'means-merge',
        [f32('partial', partialMeans, 'read'), f32('workspace', workspace, 'read_write')],
        k + 1,
        constants,
        `${getKahanSumWGSL('partial', 'index', `${k + 1}u`, 'TILE_COUNT')}
  workspace[workspaceOffset + W_MEANS + index] = sum / f32(ROWS);`
      ),
      ...new GPUSpatialWeightsTraces({
        id: `${id}-traces`,
        weights,
        maximumPower,
        mode: this.traceMode,
        probeCount: props.traceProbeCount,
        parameters: props.traceParameters,
        output: {powerTraces, powerSums, infinityNorm}
      }).getCommandNodes(graph),
      gramTiles(0),
      gramTiles(1),
      gramTiles(2),
      kernel(
        'gram-merge',
        [f32('partial', partialGram, 'read'), f32('gram', gram, 'read_write')],
        3 * packedCount,
        constants,
        `let mode = index / PACK;
  let entry = index % PACK;
  ${getKahanSumWGSL('partial', 'mode * TILE_COUNT * PACK + entry', 'PACK', 'TILE_COUNT')}
  gram[gramOffset + index] = sum;`
      ),
      kernel(
        'lambda-search',
        [
          f32('partialNorm', partialNorm, 'read'),
          f32('gram', gram, 'read'),
          f32('powerTraces', powerTraces, 'read'),
          f32('infinityNorm', infinityNorm, 'read'),
          f32('workspace', workspace, 'read_write')
        ],
        1,
        `${constants}
${finiteDeclarations}
const ROWS_F: f32 = ${rowCount}.0;
${getSpatialLogDeterminantWGSL(maximumPower, 'powerTraces', 'infinityNorm[infinityNormOffset]')}
${getTraceSeriesWGSL(maximumPower, 'powerTraces', 'infinityNorm[infinityNormOffset]')}
${getCholeskyWGSL(p)}
${getGramWGSL()}

// (ee, dee/dlambda) of the filtered regression at lambda; NaN when its system is not positive definite.
fn evaluate(lambda: f32) -> vec2<f32> {
  let nan = getNaN();
  var a: array<f32, ${p * p}>;
  var rhs: array<f32, ${p}>;
  var scale: array<f32, ${p}>;
  for (var column = 0u; column < P; column++) {
    let diagonal = getGramValue(column, column, lambda);
    if (!(diagonal > 0.0) || !isFiniteValue(diagonal)) {
      return vec2<f32>(nan, nan);
    }
    scale[column] = sqrt(diagonal);
  }
  for (var row = 0u; row < P; row++) {
    for (var column = 0u; column < P; column++) {
      a[row * P + column] = getGramValue(row, column, lambda) / (scale[row] * scale[column]);
    }
    rhs[row] = getGramValue(row, K + 1u, lambda) / scale[row];
  }
  if (!choleskyFactor_${p}(&a)) {
    return vec2<f32>(nan, nan);
  }
  choleskySolve_${p}(&a, &rhs);
  var errorSumOfSquares = getGramValue(K + 1u, K + 1u, lambda);
  var derivative = getGramDerivative(K + 1u, K + 1u, lambda);
  for (var row = 0u; row < P; row++) {
    let beta = rhs[row] / scale[row];
    errorSumOfSquares -= 2.0 * beta * getGramValue(row, K + 1u, lambda);
    derivative -= 2.0 * beta * getGramDerivative(row, K + 1u, lambda);
    for (var column = 0u; column < P; column++) {
      let other = rhs[column] / scale[column];
      errorSumOfSquares += beta * other * getGramValue(row, column, lambda);
      derivative += beta * other * getGramDerivative(row, column, lambda);
    }
  }
  return vec2<f32>(errorSumOfSquares, derivative);
}

fn getObjective(lambda: f32) -> f32 {
  return -0.5 * ROWS_F * log(evaluate(lambda).x / ROWS_F) + getSpatialLogDeterminant(lambda);
}

fn getDerivative(lambda: f32) -> f32 {
  let value = evaluate(lambda);
  return -0.5 * ROWS_F * value.y / value.x + getLogDeterminantDerivative(lambda);
}
${getParameterSearchWGSL()}`,
        `var largest = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    largest = max(largest, partialNorm[partialNormOffset + tile]);
  }
  let validNorm = largest > 0.0 && isFiniteValue(largest);
  let seriesLimit = ${getWGSLFloatLiteral(getSpatialMaximumLikelihoodSeriesLimit(maximumPower))};
  let stationarity = ${getWGSLFloatLiteral(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_BOUND_FRACTION)};
  let bound = select(getNaN(), min(stationarity, seriesLimit) / largest, validNorm);
  workspace[workspaceOffset + W_NORM] = select(getNaN(), largest, validNorm);
  workspace[workspaceOffset + W_BOUND] = bound;
  workspace[workspaceOffset + W_SERIES_LIMITED] = select(0.0, 1.0, seriesLimit < stationarity);
  workspace[workspaceOffset + W_PARAMETER] = getNaN();
  workspace[workspaceOffset + W_VALID] = 0.0;
  workspace[workspaceOffset + W_CANDIDATES] = 0.0;
  workspace[workspaceOffset + W_AT_BOUNDARY] = 0.0;
  let count = gram[gramOffset];
  let responseSquares = gram[gramOffset + packedIndex(K + 1u, K + 1u)];
  if (validNorm && isFiniteValue(count) && isFiniteValue(responseSquares) && count > 0.0 && responseSquares > 0.0) {
    let result = searchParameter(bound);
    workspace[workspaceOffset + W_PARAMETER] = result.parameter;
    workspace[workspaceOffset + W_VALID] = select(0.0, 1.0, isFiniteValue(result.parameter));
    workspace[workspaceOffset + W_CANDIDATES] = f32(result.localMaximumCount);
    workspace[workspaceOffset + W_AT_BOUNDARY] = select(0.0, 1.0, result.atBoundary);
  }`
      ),
      kernel(
        'filtered-solve',
        [f32('gram', gram, 'read'), f32('workspace', workspace, 'read_write')],
        1,
        `${constants}
${finiteDeclarations}
${getCholeskyWGSL(p)}
${getSymmetricEigenvalueRangeWGSL(p)}
${getBelsleyConditionNumberWGSL(k)}
${getGramWGSL()}`,
        getFilteredSolveBodyWGSL(k, p)
      ),
      kernel(
        'residuals-tiles',
        [
          f32('design', props.predictors, 'read'),
          f32('response', props.response, 'read'),
          f32('workspace', workspace, 'read'),
          f32('residualOut', residualView, 'read_write'),
          f32('partial', partialResiduals, 'read_write')
        ],
        tileCount,
        constants,
        `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  let ybar = workspace[workspaceOffset + W_YBAR];
  var sums: array<f32, ${RESIDUAL_SUM_COUNT}>;
  for (var row = firstRow; row < lastRow; row++) {
    var fitted = workspace[workspaceOffset + W_BETA];
    for (var column = 0u; column < K; column++) {
      fitted += design[designOffset + row * K + column] * workspace[workspaceOffset + W_BETA + 1u + column];
    }
    let u = response[responseOffset + row] - fitted;
    let y = response[responseOffset + row] - ybar;
    residualOut[residualOutOffset + row] = u;
    sums[0] += u * u;
    sums[1] += u;
    sums[2] += y * u;
    sums[3] += y * y;
  }
  for (var slot = 0u; slot < ${RESIDUAL_SUM_COUNT}u; slot++) {
    partial[partialOffset + index * ${RESIDUAL_SUM_COUNT}u + slot] = sums[slot];
  }`,
        {workgroupSize: TILE_WORKGROUP_SIZE}
      ),
      kernel(
        'residuals-merge',
        [f32('partial', partialResiduals, 'read'), f32('residualSums', residualSums, 'read_write')],
        RESIDUAL_SUM_COUNT,
        constants,
        `${getKahanSumWGSL('partial', 'index', `${RESIDUAL_SUM_COUNT}u`, 'TILE_COUNT')}
  residualSums[residualSumsOffset + index] = sum;`
      )
    ];

    const probes = createSeriesProbeNodes(graph, {
      id: `${id}-tr3`,
      operation: OPERATION,
      weights,
      rowCount,
      maximumPower,
      mode: this.traceMode,
      probeCount: props.traceProbeCount ?? GPU_SPATIAL_WEIGHTS_TRACES_DEFAULT_PROBE_COUNT,
      seedParameters: props.traceParameters,
      tileRowCount,
      tileCount,
      workspace,
      parameterSlot: slots['PARAMETER']
    });
    nodes.push(...probes.nodes);
    const probeScale = stochastic ? `1.0 / f32(${probes.probeTotal}u)` : '1.0';

    nodes.push(
      kernel(
        'variance',
        [
          f32('dots', probes.dots, 'read'),
          f32('powerTraces', powerTraces, 'read'),
          f32('infinityNorm', infinityNorm, 'read'),
          f32('workspace', workspace, 'read_write')
        ],
        1,
        `${constants}
${finiteDeclarations}
${getTraceSeriesWGSL(maximumPower, 'powerTraces', 'infinityNorm[infinityNormOffset]')}`,
        `let n = f32(ROWS);
  let lambda = workspace[workspaceOffset + W_PARAMETER];
  let traceOne = getTraceOne(lambda);
  let traceTwo = getTraceTwo(lambda);
  ${probes.dotsSumWGSL}
  let traceThree = sum * ${probeScale};
  workspace[workspaceOffset + W_TRACES] = traceOne;
  workspace[workspaceOffset + W_TRACES + 1u] = traceTwo;
  workspace[workspaceOffset + W_TRACES + 2u] = traceThree;
  let information = traceTwo + traceThree - 2.0 * traceOne * traceOne / n;
  if (information > 0.0 && isFiniteValue(information)) {
    workspace[workspaceOffset + W_STANDARD_ERRORS + P] = sqrt(1.0 / information);
  } else {
    workspace[workspaceOffset + W_STANDARD_ERRORS + P] = getNaN();
    workspace[workspaceOffset + W_VARIANCE_DEFINED] = 0.0;
  }`
      ),
      kernel(
        'finish',
        [
          f32('residualSums', residualSums, 'read'),
          f32('powerTraces', powerTraces, 'read'),
          f32('infinityNorm', infinityNorm, 'read'),
          f32('workspace', workspace, 'read'),
          f32('tableOut', output.table, 'read_write'),
          f32('summaryOut', output.summary, 'read_write'),
          getU32Binding('statusOut', output.status.code, 'read_write'),
          ...(output.status.converged
            ? [getU32Binding('convergedOut', output.status.converged, 'read_write')]
            : [])
        ],
        1,
        `${constants}
${finiteDeclarations}
${ERFC_VALUE_WGSL}
${getSpatialLogDeterminantWGSL(maximumPower, 'powerTraces', 'infinityNorm[infinityNormOffset]')}
${getTableRowWGSL()}
${getFinishWGSL(k, maximumPower, stochastic, output.status.converged !== undefined)}`,
        'finish();'
      )
    );
    return nodes;
  }
}

/** Packed lower-triangle index and the quadratic-in-lambda Gram entries of the filtered columns. */
function getGramWGSL(): string {
  return /* wgsl */ `
fn packedIndex(first: u32, second: u32) -> u32 {
  let high = max(first, second);
  let low = min(first, second);
  return high * (high + 1u) / 2u + low;
}

// Gram of Z - lambda W Z: G0 - lambda (G1 + G1') + lambda^2 G2, G0 G1 G2 stored back to back.
fn getGramValue(first: u32, second: u32, lambda: f32) -> f32 {
  let entry = packedIndex(first, second);
  return gram[gramOffset + entry] - lambda * gram[gramOffset + PACK + entry]
    + lambda * lambda * gram[gramOffset + 2u * PACK + entry];
}

fn getGramDerivative(first: u32, second: u32, lambda: f32) -> f32 {
  let entry = packedIndex(first, second);
  return -gram[gramOffset + PACK + entry] + 2.0 * lambda * gram[gramOffset + 2u * PACK + entry];
}
`;
}

/**
 * The `filtered-solve` kernel body: the filtered regression at the estimated lambda. Writes the
 * status code, the raw coefficients, `s2`, the conditioning and the coefficient standard errors
 * (the lambda standard error is written by the variance kernel).
 */
function getFilteredSolveBodyWGSL(k: number, p: number): string {
  return `let n = f32(ROWS);
  let lambda = workspace[workspaceOffset + W_PARAMETER];
  let nan = getNaN();
  for (var slot = 0u; slot < P + 1u; slot++) {
    workspace[workspaceOffset + W_STANDARD_ERRORS + slot] = nan;
    if (slot < P) {
      workspace[workspaceOffset + W_BETA + slot] = nan;
    }
  }
  workspace[workspaceOffset + W_VARIANCE_DEFINED] = 0.0;
  workspace[workspaceOffset + W_SIGMA2] = nan;
  for (var slot = 0u; slot < 4u; slot++) {
    workspace[workspaceOffset + W_CONDITIONING + slot] = nan;
  }
  if (!(n > f32(K + 2u))) {
    workspace[workspaceOffset + W_STATUS] = 2.0;
    return;
  }
  var a: array<f32, ${p * p}>;
  var unit: array<f32, ${p * p}>;
  var rhs: array<f32, ${p}>;
  var scale: array<f32, ${p}>;
  if (workspace[workspaceOffset + W_VALID] != 1.0 || !isFiniteValue(lambda)) {
    // A rank-deficient design makes every filtered system indefinite, so the search finds no
    // maximum: classify that as singular (the unfiltered system at lambda = 0), else non-finite.
    var nonFinite = false;
    var singular = false;
    for (var column = 0u; column < P; column++) {
      let diagonal = getGramValue(column, column, 0.0);
      nonFinite = nonFinite || !isFiniteValue(diagonal);
      singular = singular || !(diagonal > 0.0);
      scale[column] = sqrt(max(diagonal, 1.0e-30));
    }
    for (var row = 0u; row < P; row++) {
      for (var column = 0u; column < P; column++) {
        let value = getGramValue(row, column, 0.0) / (scale[row] * scale[column]);
        nonFinite = nonFinite || !isFiniteValue(value);
        a[row * P + column] = value;
      }
    }
    singular = singular || !choleskyFactor_${p}(&a);
    workspace[workspaceOffset + W_STATUS] = select(3.0, 1.0, singular && !nonFinite);
    return;
  }
  for (var column = 0u; column < P; column++) {
    let diagonal = getGramValue(column, column, lambda);
    if (!(diagonal > 0.0) || !isFiniteValue(diagonal)) {
      workspace[workspaceOffset + W_STATUS] = 1.0;
      return;
    }
    scale[column] = sqrt(diagonal);
  }
  for (var row = 0u; row < P; row++) {
    for (var column = 0u; column < P; column++) {
      let value = getGramValue(row, column, lambda) / (scale[row] * scale[column]);
      a[row * P + column] = value;
      unit[row * P + column] = value;
    }
    rhs[row] = getGramValue(row, K + 1u, lambda) / scale[row];
  }
  if (!choleskyFactor_${p}(&a)) {
    workspace[workspaceOffset + W_STATUS] = 1.0;
    return;
  }
  choleskySolve_${p}(&a, &rhs);
  var filtered: array<f32, ${p}>;
  var errorSumOfSquares = getGramValue(K + 1u, K + 1u, lambda);
  for (var row = 0u; row < P; row++) {
    filtered[row] = rhs[row] / scale[row];
    errorSumOfSquares -= 2.0 * filtered[row] * getGramValue(row, K + 1u, lambda);
    for (var column = 0u; column < P; column++) {
      errorSumOfSquares += filtered[row] * (rhs[column] / scale[column]) * getGramValue(row, column, lambda);
    }
  }
  let sigma2 = errorSumOfSquares / n;
  if (!(sigma2 > 0.0) || !isFiniteValue(sigma2)) {
    workspace[workspaceOffset + W_STATUS] = 3.0;
    return;
  }
  workspace[workspaceOffset + W_STATUS] = 0.0;
  workspace[workspaceOffset + W_SIGMA2] = sigma2;
  // Raw coefficients: b = filtered slopes, b0 = a0 + ybar - m'b.
  var intercept = filtered[0] + workspace[workspaceOffset + W_YBAR];
  for (var column = 0u; column < K; column++) {
    intercept -= workspace[workspaceOffset + W_MEANS + column] * filtered[1u + column];
    workspace[workspaceOffset + W_BETA + 1u + column] = filtered[1u + column];
  }
  workspace[workspaceOffset + W_BETA] = intercept;
  // Covariance sigma2 (Xs'Xs)^-1 = sigma2 D^-1 (unit^-1) D^-1 from unit solves.
  var covariance: array<f32, ${p * p}>;
  var vifs = 0.0;
  for (var column = 0u; column < P; column++) {
    var unitVector: array<f32, ${p}>;
    unitVector[column] = 1.0;
    choleskySolve_${p}(&a, &unitVector);
    for (var row = 0u; row < P; row++) {
      covariance[row * P + column] = sigma2 * unitVector[row] / (scale[row] * scale[column]);
    }
    if (column > 0u) {
      vifs = max(vifs, unitVector[column]);
    }
  }
  var interceptVariance = covariance[0];
  for (var first = 0u; first < K; first++) {
    let meanFirst = workspace[workspaceOffset + W_MEANS + first];
    interceptVariance -= 2.0 * meanFirst * covariance[1u + first];
    for (var second = 0u; second < K; second++) {
      interceptVariance += meanFirst * workspace[workspaceOffset + W_MEANS + second] * covariance[(1u + first) * P + 1u + second];
    }
  }
  workspace[workspaceOffset + W_STANDARD_ERRORS] = sqrt(interceptVariance);
  for (var column = 1u; column < P; column++) {
    workspace[workspaceOffset + W_STANDARD_ERRORS + column] = sqrt(covariance[column * P + column]);
  }
  workspace[workspaceOffset + W_VARIANCE_DEFINED] = 1.0;
  // Conditioning: Belsley condition number of the raw filtered design [c, Xc_s + m c] (c is the
  // filtered constant, not orthogonal to the centered columns) without forming its Gram matrix.
  // With L the Cholesky factor of the unit-diagonal filtered Gram (held in a), the raw column j is
  // inPlane_j u0 + g_j sum_i L[j][i] e_i, so the helper's alpha_j = inPlane_j / N_j, beta_j =
  // g_j / N_j and factor = L without its first row and column.
  var belsleyFactor: array<f32, ${k * k}>;
  var belsleyAlpha: array<f32, ${k}>;
  var belsleyBeta: array<f32, ${k}>;
  for (var j = 0u; j < K; j++) {
    let column = scale[1u + j];
    let inPlane = workspace[workspaceOffset + W_MEANS + j] * scale[0] + column * a[(1u + j) * P];
    var tail = 0.0;
    for (var i = 0u; i <= j; i++) {
      belsleyFactor[j * K + i] = a[(1u + j) * P + 1u + i];
      tail = tail + a[(1u + j) * P + 1u + i] * a[(1u + j) * P + 1u + i];
    }
    let rawNorm = sqrt(inPlane * inPlane + column * column * tail);
    belsleyAlpha[j] = inPlane / rawNorm;
    belsleyBeta[j] = column / rawNorm;
  }
  let unitRange = getSymmetricEigenvalueRange_${p}(&unit);
  workspace[workspaceOffset + W_CONDITIONING] = getBelsleyConditionNumber_${k}(&belsleyFactor, &belsleyAlpha, &belsleyBeta);
  workspace[workspaceOffset + W_CONDITIONING + 1u] = unitRange.y / unitRange.x;
  workspace[workspaceOffset + W_CONDITIONING + 2u] = vifs;`;
}

function getFinishWGSL(
  k: number,
  maximumPower: number,
  stochastic: boolean,
  hasConverged: boolean
): string {
  const rows = k + 2;
  const summary = (slot: number) => `summaryOut[summaryOutOffset + ${slot}u]`;
  return /* wgsl */ `
fn fail(status: u32) {
  let nan = getNaN();
  for (var slot = 0u; slot < ${rows * GPU_SPATIAL_MAXIMUM_LIKELIHOOD_TABLE_STRIDE}u; slot++) {
    tableOut[tableOutOffset + slot] = nan;
  }
  for (var slot = 0u; slot < ${GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LENGTH}u; slot++) {
    summaryOut[summaryOutOffset + slot] = nan;
  }
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_ROW_COUNT)} = f32(ROWS);
  statusOut[statusOutOffset] = status;
${hasConverged ? '  convergedOut[convergedOutOffset] = 0u;' : ''}
}

fn finish() {
  let code = u32(workspace[workspaceOffset + W_STATUS]);
  if (code != 0u) {
    fail(code);
    return;
  }
  let n = f32(ROWS);
  let lambda = workspace[workspaceOffset + W_PARAMETER];
  let sigma2 = workspace[workspaceOffset + W_SIGMA2];
  let residualSquares = residualSums[residualSumsOffset];
  let residualTotal = residualSums[residualSumsOffset + 1u];
  let responseResidual = residualSums[residualSumsOffset + 2u];
  let responseSquares = residualSums[residualSumsOffset + 3u];
  if (!isFiniteValue(residualSquares) || !isFiniteValue(responseSquares) || !isFiniteValue(responseResidual)) {
    fail(3u);
    return;
  }
  for (var row = 0u; row < P; row++) {
    writeTableRow(row, workspace[workspaceOffset + W_BETA + row], workspace[workspaceOffset + W_STANDARD_ERRORS + row]);
  }
  writeTableRow(P, lambda, workspace[workspaceOffset + W_STANDARD_ERRORS + P]);
  let logDeterminant = getSpatialLogDeterminant(lambda);
  let logLikelihood = -0.5 * n * log(sigma2) + logDeterminant - 0.5 * n * (${getWGSLFloatLiteral(LN_TWO_PI)} + 1.0);
  // spreg ML_Error counts the intercept and the predictors only.
  let parameterCount = f32(P);
  let predictionCovariance = responseSquares - responseResidual;
  let predictionVariance = responseSquares - 2.0 * responseResidual + residualSquares - residualTotal * residualTotal / n;
  let correlationCondition = workspace[workspaceOffset + W_CONDITIONING + 1u];
  let bound = workspace[workspaceOffset + W_BOUND];
  let norm = workspace[workspaceOffset + W_NORM];
  let scaled = abs(lambda) * norm;
  let atBoundary = workspace[workspaceOffset + W_AT_BOUNDARY] == 1.0;
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_ROW_COUNT)} = n;
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PARAMETER)} = lambda;
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SIGMA_SQUARED)} = sigma2;
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOG_LIKELIHOOD)} = logLikelihood;
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_AIC)} = 2.0 * parameterCount - 2.0 * logLikelihood;
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SCHWARZ)} = parameterCount * log(n) - 2.0 * logLikelihood;
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PSEUDO_R_SQUARED)} = predictionCovariance * predictionCovariance / (responseSquares * predictionVariance);
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PSEUDO_R_SQUARED_SPATIAL)} = getNaN();
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_RESIDUAL_SUM_OF_SQUARES)} = residualSquares;
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CONDITION_NUMBER)} = workspace[workspaceOffset + W_CONDITIONING];
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CORRELATION_CONDITION_NUMBER)} = correlationCondition;
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR)} = workspace[workspaceOffset + W_CONDITIONING + 2u];
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PRECISION_WARNING)} = select(0.0, 1.0, correlationCondition > ${getWGSLFloatLiteral(GPU_REGRESSION_PRECISION_CONDITION_LIMIT)});
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PARAMETER_BOUND)} = bound;
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_APPROXIMATION)} = ${stochastic ? '1.0' : '0.0'};
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CONVERGED)} = select(0.0, 1.0, !atBoundary);
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_AT_BOUNDARY)} = select(0.0, 1.0, atBoundary);
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SERIES_LIMITED)} = workspace[workspaceOffset + W_SERIES_LIMITED];
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_TRUNCATION_BOUND)} = select(
    1.0e38,
    n * pow(scaled, ${maximumPower + 1}.0) / (${maximumPower + 1}.0 * (1.0 - scaled)),
    scaled < 1.0
  );
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_VARIANCE_DEFINED)} = workspace[workspaceOffset + W_VARIANCE_DEFINED];
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOCAL_MAXIMUM_COUNT)} = workspace[workspaceOffset + W_CANDIDATES];
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOG_DETERMINANT)} = logDeterminant;
  statusOut[statusOutOffset] = 0u;
${hasConverged ? '  convergedOut[convergedOutOffset] = select(0u, 1u, !atBoundary);' : ''}
}
`;
}
