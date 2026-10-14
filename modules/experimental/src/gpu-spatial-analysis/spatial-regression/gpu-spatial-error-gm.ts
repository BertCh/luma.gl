// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GraphVectorView,
  validatePackedView,
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
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {type GPUSpatialWeights, validateGPUSpatialWeights} from '../spatial-weights/index';
import {GPUOrdinaryLeastSquares} from './gpu-ordinary-least-squares';
import {GPUSpatialWeightsTraces} from './gpu-spatial-weights-traces';
import {GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH} from './ordinary-least-squares-parameters';
import type {GPUSpatialWeightsTracesMode} from './spatial-weights-traces-parameters';
import {
  GPU_SPATIAL_ERROR_GM_DEFAULT_ESTIMATOR,
  GPU_SPATIAL_ERROR_GM_LAMBDA_BOUND_FRACTION,
  GPU_SPATIAL_ERROR_GM_MAXIMUM_PREDICTOR_COUNT,
  GPU_SPATIAL_ERROR_GM_SUMMARY_CONDITION_NUMBER,
  GPU_SPATIAL_ERROR_GM_SUMMARY_CORRELATION_CONDITION_NUMBER,
  GPU_SPATIAL_ERROR_GM_SUMMARY_LAMBDA_AT_BOUND,
  GPU_SPATIAL_ERROR_GM_SUMMARY_LAMBDA_BOUND,
  GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH,
  GPU_SPATIAL_ERROR_GM_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR,
  GPU_SPATIAL_ERROR_GM_SUMMARY_APPROXIMATION,
  GPU_SPATIAL_ERROR_GM_SUMMARY_PRECISION_WARNING,
  GPU_SPATIAL_ERROR_GM_TABLE_STRIDE,
  type GPUSpatialErrorGMEstimator
} from './spatial-error-gm-parameters';
import {
  getBelsleyConditionNumberWGSL,
  getCholeskyWGSL,
  getSymmetricEigenvalueRangeWGSL
} from './spatial-regression-solve';
import type {GPUSolverStatusPort} from '../contracts/index';
import {GPU_REGRESSION_PRECISION_CONDITION_LIMIT} from '../contracts/model-result';
import {ERFC_VALUE_WGSL, GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../utils/wgsl-common';

const OPERATION = 'GPUSpatialErrorGM';
const TILE_WORKGROUP_SIZE = 64;
const MAXIMUM_DEFAULT_TILE_COUNT = 4096;
const MINIMUM_DEFAULT_TILE_ROWS = 64;
/** Moment sums: `u'u`, `u'Wu`, `(Wu)'Wu`, `u'WWu`, `(Wu)'WWu`, `(WWu)'WWu`, `tr W'W`. */
const MOMENT_SUM_COUNT = 7;
/** Residual sums: `e'e`, `sum u`, `u'u`, `sum (y - ybar) u`, `sum (y - ybar)^2`. */
const RESIDUAL_SUM_COUNT = 5;
/** Filtered residual power sums: `sum u_s^2`, `sum u_s^3`, `sum u_s^4`. */
const POWER_SUM_COUNT = 3;
/** Grid intervals scanned for stationary points of the moment objective on `[-bound, bound]`. */
const LAMBDA_SCAN_STEPS = 1024;
/** Bisection steps that refine each bracketed minimum. */
const LAMBDA_BISECTION_STEPS = 40;
/** Traces of `W` the homoskedastic variance needs: maximum power and Gram traces. */
const TRACE_POWER_COUNT = 2;

/** Caller-owned outputs of {@link GPUSpatialErrorGM}. */
export type GPUSpatialErrorGMOutput = GPUSolverStatusPort & {
  /**
   * `(predictorCount + 2) * 4` float32 values: one row per coefficient in spreg order (intercept,
   * one per predictor, then the spatial error parameter `lambda` last), each row holding the
   * coefficient, standard error, z statistic and two-sided normal p-value. The `lambda` row holds
   * its estimate and, for the `homoskedastic` estimator, its standard error, z statistic and
   * p-value; for `kelejian-prucha` the other three are NaN, as in spreg `GM_Error`.
   */
  table: GraphDataView<'float32'>;
  /** At least 13 float32 values; see the `GPU_SPATIAL_ERROR_GM_SUMMARY_*` slots. */
  summary: GraphDataView<'float32'>;
  /** Optional per-row residuals `u = y - X b` (spreg `u`); NaN for a failed fit. */
  residuals?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUSpatialErrorGM}.
 *
 * Per-frame (no recompile): the contents of every input buffer. Topology (needs a new graph):
 * `predictorCount`, the row count, the weights views, whether `residuals` is present,
 * `estimator`, the trace options and `tileRowCount`.
 */
export type GPUSpatialErrorGMProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'spatial-error-gm'`. */
  id?: string;
  /**
   * Square weights `W`; rows index the regression rows. Any sparsity pattern works (no symmetry
   * is required) and rows need not be standardized: `lambda` is searched within
   * `0.99 / ||W||_inf`, which keeps `I - lambda W` invertible for any `W`.
   */
  weights: GPUSpatialWeights;
  /** Row-major predictors, `predictors[row * predictorCount + column]`, without an intercept. */
  predictors: GraphDataView<'float32'>;
  /** Response `y`, one value per row. */
  response: GraphDataView<'float32'>;
  /** Number of predictor columns, compile-time, 1 to 15. The intercept is implicit. */
  predictorCount: number;
  /**
   * Estimator. `'homoskedastic'` (the default, changed from the previous Kelejian-Prucha default)
   * is spreg `GM_Error_Hom` (`A1 = 'hom_sc'`, one iteration) and gives `lambda` a standard error,
   * z statistic and p-value. `'kelejian-prucha'` is spreg `GM_Error`, whose `lambda` has no
   * variance estimate (NaN).
   */
  estimator?: GPUSpatialErrorGMEstimator;
  /**
   * Trace computation of the `homoskedastic` variance: `'exact'` or `'stochastic'` (Hutchinson),
   * see `GPUSpatialWeightsTraces`. Defaults to exact up to 4096 rows and stochastic above, where
   * the standard error of `lambda` is approximate (`GPU_SPATIAL_ERROR_GM_SUMMARY_APPROXIMATION`).
   * Ignored by `kelejian-prucha`.
   */
  traceMode?: GPUSpatialWeightsTracesMode;
  /** Hutchinson probe count of the stochastic traces. Defaults to 64. */
  traceProbeCount?: number;
  /** Philox seed of the stochastic traces. Defaults to 0. */
  traceSeed?: number;
  /**
   * Rows reduced by one invocation of the accumulation passes. Defaults to
   * `max(64, ceil(rows / 4096))`.
   */
  tileRowCount?: number;
  /** Caller-owned outputs. */
  output: GPUSpatialErrorGMOutput;
};

/**
 * Generalized-moments estimator of the spatial error model `y = X b + u`, `u = lambda W u + e`.
 *
 * Two estimators share the first stage and the spatially filtered regression:
 *
 * 1. Ordinary least squares of `y` on `[1, X]` (the `GPUOrdinaryLeastSquares` contributor) gives
 *    residuals `u`.
 * 2. `lambda` comes from sample moments of `u`, `Wu` and `WWu`, minimized globally on a grid over
 *    `[-bound, bound]` (bracketed stationary points are refined by bisection; spreg runs a local
 *    L-BFGS-B from zero, which agrees whenever the objective has one basin). `bound` is
 *    `0.99 / ||W||_inf` with `||W||_inf` the largest absolute row sum, computed on the GPU. It
 *    equals spreg's 0.99 for a row-standardized `W` and keeps `I - lambda W` invertible for any
 *    `W` (a fixed 0.99 admits explosive `lambda` for raw weights).
 * 3. The spatially filtered regression `y - lambda W y` on `[1, X] - lambda W [1, X]` is fit by OLS
 *    (the filtered constant `1 - lambda * sum_j w_ij` is a column, as spreg `get_spFilter`). Columns
 *    and response are centered before filtering, which keeps float32 normal equations well
 *    conditioned (exactly equivalent by shift equivariance); intercept and its variance are mapped
 *    back to the uncentered model.
 *
 * `estimator: 'kelejian-prucha'` is spreg `GM_Error`: the moments are `G [lambda, lambda^2,
 * sigma2]' = g` with `G = [[2 u'Wu, -(Wu)'Wu, 1], [2 (Wu)'WWu, -(WWu)'WWu, tr(W'W) / n],
 * [u'WWu + (Wu)'Wu, -(Wu)'WWu, 0]] / n` and `g = [u'u, (Wu)'Wu, u'Wu] / n`; `sigma2 >= 0` is
 * eliminated by least squares. `lambda` is reported without a standard error; the betas use
 * `sigma2 (Xs'Xs)^-1`, `sigma2 = e'e / n` of the filtered residuals, and ignore the estimation of
 * `lambda`.
 *
 * `estimator: 'homoskedastic'` (the default) is spreg `GM_Error_Hom` (Drukker, Egger and Prucha
 * 2013; Anselin 2011) with `A1 = 'hom_sc'` and one iteration, replicated step for step:
 * (a) `lambda1` minimizes the unweighted `|G p - g|^2` of the two moments `u'A1u`, `u'A2u` with
 * `A1 = (W'W - (t/n) I) / (1 + (t/n)^2)`, `t = tr(W'W)`, `A2 = (W + W') / 2`; (b) the FGLS betas at
 * `lambda1` are the reported betas and give `u1 = y - X b1`; (c) the moments are recomputed on
 * `u1`, `psi` (the variance of the moments from `sigma2`, `mu3`, `mu4` of `u_s = u1 - lambda1 W u1`
 * and traces of `W`) weights the second minimization `(G p - g)' psi^-1 (G p - g)` giving
 * `lambda`; (d) `Var(b) = sigma2 (Xs'Xs)^-1` with `Xs` and `u_s` filtered at `lambda`
 * (`sigma2 = u_s'u_s / n`) and `Var(lambda) = (j' psi^-1 j)^-1 / n` with `j = G [1, 2 lambda]'` and
 * `psi` at `lambda`, so z statistics and p-values (standard normal) exist for `lambda` too. The
 * betas-`lambda` covariance of spreg is not formed (it affects no reported number). Every moment
 * entry is a combination of sums the kernels accumulate (`u'u`, `u'Wu`, ...) and of the traces
 * `tr(W)`, `tr(W^2)`, `tr(W'W)`, `tr(W'WW)`, `tr((W'W)^2)`, `sum c_i^2` from
 * `GPUSpatialWeightsTraces`: exact up to 4096 rows, Hutchinson estimates above, in which case the
 * standard error of `lambda` is approximate and `GPU_SPATIAL_ERROR_GM_SUMMARY_APPROXIMATION` is 1
 * (betas, `lambda` and the betas' standard errors are unaffected).
 *
 * Conditioning of the final filtered regression is reported in `summary` (Belsley condition number,
 * correlation condition number, largest variance inflation factor, and a precision warning above
 * `GPU_REGRESSION_PRECISION_CONDITION_LIMIT`); a warning never changes `status`.
 *
 * All accumulations use the fixed-order tile and Kahan-merge scheme of `GPUOrdinaryLeastSquares`,
 * so results are deterministic; nothing is read back. Up to 15 predictors (the OLS limit, the
 * 16-wide Cholesky); no kernel binds more than 8 storage buffers. Not included: `GM_Error_Het`
 * and the Kelejian-Prucha weighted estimator.
 */
export class GPUSpatialErrorGM implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialErrorGMProps;
  /** Number of rows. */
  readonly rowCount: number;
  /** Rows reduced by one accumulation invocation. */
  readonly tileRowCount: number;
  /** Number of accumulation tiles. */
  readonly tileCount: number;
  /** Resolved estimator. */
  readonly estimator: GPUSpatialErrorGMEstimator;

  constructor(props: GPUSpatialErrorGMProps) {
    const id = props.id ?? 'spatial-error-gm';
    this.id = id;
    this.props = props;
    const {predictorCount, output} = props;
    if (
      !Number.isInteger(predictorCount) ||
      predictorCount < 1 ||
      predictorCount > GPU_SPATIAL_ERROR_GM_MAXIMUM_PREDICTOR_COUNT
    ) {
      throw new Error(
        `${id} predictorCount must be an integer in [1, ${GPU_SPATIAL_ERROR_GM_MAXIMUM_PREDICTOR_COUNT}]`
      );
    }
    const estimator = props.estimator ?? GPU_SPATIAL_ERROR_GM_DEFAULT_ESTIMATOR;
    if (estimator !== 'homoskedastic' && estimator !== 'kelejian-prucha') {
      throw new Error(`${id} estimator must be 'homoskedastic' or 'kelejian-prucha'`);
    }
    this.estimator = estimator;
    for (const [name, view] of [
      ['predictors', props.predictors],
      ['response', props.response]
    ] as const) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    const rows = validateGPUSpatialWeights(id, props.weights);
    validatePackedView(props.predictors, ['float32'], `${id} predictors`);
    validatePackedView(props.response, ['float32'], `${id} response`);
    if (props.response.length !== rows) {
      throw new Error(`${id} response length must equal the weights row count`);
    }
    if (props.predictors.length !== rows * predictorCount) {
      throw new Error(`${id} predictors length must equal the weights row count * predictorCount`);
    }
    validatePackedView(output.table, ['float32'], `${id} output.table`);
    if (output.table.length < (predictorCount + 2) * GPU_SPATIAL_ERROR_GM_TABLE_STRIDE) {
      throw new Error(`${id} output.table must hold (predictorCount + 2) * 4 float32 values`);
    }
    validatePackedView(output.summary, ['float32'], `${id} output.summary`);
    if (output.summary.length < GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH) {
      throw new Error(
        `${id} output.summary must hold ${GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH} float32 values`
      );
    }
    validatePackedUint32View(output.status, `${id} output.status`);
    if (output.status.length < 1) {
      throw new Error(`${id} output.status must hold one uint32 row`);
    }
    if (output.residuals) {
      validatePackedView(output.residuals, ['float32'], `${id} output.residuals`);
      if (output.residuals.length < rows) {
        throw new Error(`${id} output.residuals must hold one row per input row`);
      }
    }
    if (props.tileRowCount !== undefined) {
      if (!Number.isInteger(props.tileRowCount) || props.tileRowCount < 1) {
        throw new Error(`${id} tileRowCount must be a positive integer`);
      }
    }
    this.rowCount = rows;
    this.tileRowCount =
      props.tileRowCount ??
      Math.max(MINIMUM_DEFAULT_TILE_ROWS, Math.ceil(rows / MAXIMUM_DEFAULT_TILE_COUNT));
    this.tileCount = Math.ceil(rows / this.tileRowCount);
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.table, output.summary, output.status, output.residuals],
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
   * Returns the first-stage `GPUOrdinaryLeastSquares` nodes, the weights-norm and means nodes, for
   * `homoskedastic` the `GPUSpatialWeightsTraces` nodes, then the moment, `lambda`, filtered
   * regression, residual and finish kernels (`homoskedastic` repeats the moment step on the FGLS
   * residuals and adds the filtered-residual power sums and the final filtered moments).
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rowCount, tileRowCount, tileCount, estimator} = this;
    const homoskedastic = estimator === 'homoskedastic';
    const {weights, output, predictorCount: k} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.predictors,
      props.response,
      weights.offsets,
      weights.neighbors,
      weights.weights,
      output.table,
      output.summary,
      output.status,
      output.residuals
    ]);
    const q = k + 1;
    const filteredMomentCount = q * q + q + 1;
    const meanCount = k + 1;
    // Workspace slots: means (k), response mean, lambda, objective, intercept, betas (q), then the
    // bound, the final lambda and objective, the approximation flag, the final moment matrix
    // (A0 A1 B0 B1), the trace terms (tr11 tr12 tr22 diag), and the final power sums.
    const boundSlot = k + 4 + q;
    const workspaceLength = boundSlot + 15;
    const transient = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'float32', length);
    const olsResiduals = transient('ols-residuals', rowCount);
    const olsStatus = createTransientView(graph, `${id}-ols-status`, 'uint32', 1);
    const olsNodes = new GPUOrdinaryLeastSquares({
      id: `${id}-ols`,
      predictors: props.predictors,
      response: props.response,
      predictorCount: k,
      tileRowCount,
      output: {
        coefficients: transient('ols-coefficients', q),
        standardErrors: transient('ols-standard-errors', q),
        tStatistics: transient('ols-t-statistics', q),
        summary: transient('ols-summary', GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH),
        status: olsStatus,
        residuals: olsResiduals
      }
    }).getCommandNodes(graph);
    const partialMeans = transient('partial-means', tileCount * meanCount);
    const workspace = transient('workspace', workspaceLength);
    const partialNorm = transient('partial-weights-norm', tileCount);
    const lagResiduals = transient('lag-residuals', rowCount);
    const partialMoments = transient('partial-moments', tileCount * MOMENT_SUM_COUNT);
    const moments = transient('moments', MOMENT_SUM_COUNT);
    const partialFiltered = transient('partial-filtered-moments', tileCount * filteredMomentCount);
    const filteredMoments = transient('filtered-moments', filteredMomentCount);
    const partialResiduals = transient('partial-residuals', tileCount * RESIDUAL_SUM_COUNT);
    const residualSums = transient('residual-sums', RESIDUAL_SUM_COUNT);
    const residualView = output.residuals ?? transient('residuals', rowCount);
    const constants = `const ROWS: u32 = ${rowCount}u;
const K: u32 = ${k}u;
const Q: u32 = ${q}u;
const TILE_ROWS: u32 = ${tileRowCount}u;
const TILE_COUNT: u32 = ${tileCount}u;
const W_YBAR: u32 = ${k}u;
const W_LAMBDA: u32 = ${k + 1}u;
const W_OBJECTIVE: u32 = ${k + 2}u;
const W_INTERCEPT: u32 = ${k + 3}u;
const W_BETA: u32 = ${k + 4}u;
const W_BOUND: u32 = ${boundSlot}u;
const W_LAMBDA_FINAL: u32 = ${boundSlot + 1}u;
const W_OBJECTIVE_FINAL: u32 = ${boundSlot + 2}u;
const W_APPROXIMATION: u32 = ${boundSlot + 3}u;
const W_MOMENT_MATRIX: u32 = ${boundSlot + 4}u;
const W_TRACE_TERMS: u32 = ${boundSlot + 8}u;
const W_POWER_SUMS: u32 = ${boundSlot + 12}u;`;
    const mergeLoop = (stride: number) => `var sum = 0.0;
  var compensation = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    let term = partial[partialOffset + tile * ${stride}u + index] - compensation;
    let next = sum + term;
    compensation = (next - sum) - term;
    sum = next;
  }`;
    const f32 = (name: string, view: GraphDataView, access: 'read' | 'read_write') =>
      ({name, view, type: 'f32', access}) as WGSLKernelBinding;
    const weightBindings: WGSLKernelBinding[] = [
      {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
      f32('weights', weights.weights, 'read')
    ];
    const rowBindings: WGSLKernelBinding[] = [
      ...weightBindings,
      f32('design', props.predictors, 'read'),
      f32('response', props.response, 'read'),
      f32('workspace', workspace, 'read')
    ];
    const kernel = (
      name: string,
      bindings: WGSLKernelBinding[],
      invocationCount: number,
      declarations: string,
      body: string,
      workgroupSize?: number
    ) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${name}`,
        operation: OPERATION,
        variant: name,
        bindings,
        invocationCount,
        declarations,
        body,
        ...(workgroupSize ? {workgroupSize} : {})
      });
    const filteredDeclarations = `${constants}
${getFilteredRowWGSL()}`;
    const finiteDeclarations = `${GET_NAN_WGSL}

${IS_FINITE_VALUE_WGSL}`;
    const solveDeclarations = `${constants}
${finiteDeclarations}

${getCholeskyWGSL(q)}`;
    const momentsTiles = (
      suffix: string,
      residual: GraphDataView,
      lag: GraphDataView,
      partial: GraphDataView
    ) =>
      kernel(
        `moments-tiles${suffix}`,
        [
          ...weightBindings,
          f32('residual', residual, 'read'),
          f32('lag', lag, 'read'),
          f32('partial', partial, 'read_write')
        ],
        tileCount,
        constants,
        `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${MOMENT_SUM_COUNT}>;
  for (var row = firstRow; row < lastRow; row++) {
    let u = residual[residualOffset + row];
    let wu = lag[lagOffset + row];
    var wwu = 0.0;
    var squareSum = 0.0;
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor >= ROWS) {
        continue;
      }
      let weight = weights[weightsOffset + slot];
      wwu += weight * lag[lagOffset + neighbor];
      squareSum += weight * weight;
    }
    sums[0] += u * u;
    sums[1] += u * wu;
    sums[2] += wu * wu;
    sums[3] += u * wwu;
    sums[4] += wu * wwu;
    sums[5] += wwu * wwu;
    sums[6] += squareSum;
  }
  for (var slot = 0u; slot < ${MOMENT_SUM_COUNT}u; slot++) {
    partial[partialOffset + index * ${MOMENT_SUM_COUNT}u + slot] = sums[slot];
  }`,
        TILE_WORKGROUP_SIZE
      );
    const residualLag = (suffix: string, residual: GraphDataView, lag: GraphDataView) =>
      kernel(
        `residual-lag${suffix}`,
        [...weightBindings, f32('residual', residual, 'read'), f32('lag', lag, 'read_write')],
        rowCount,
        constants,
        `var sum = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= ROWS) {
      continue;
    }
    sum += weights[weightsOffset + slot] * residual[residualOffset + neighbor];
  }
  lag[lagOffset + index] = sum;`
      );
    const filteredTiles = (suffix: string, lambdaSlot: string, partial: GraphDataView) =>
      kernel(
        `filtered-moments-tiles${suffix}`,
        [...rowBindings, f32('partial', partial, 'read_write')],
        tileCount,
        filteredDeclarations,
        `let lambda = workspace[workspaceOffset + ${lambdaSlot}];
  let base = partialOffset + index * ${filteredMomentCount}u;
  for (var slot = 0u; slot < ${filteredMomentCount}u; slot++) {
    partial[base + slot] = 0.0;
  }
  let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  for (var row = firstRow; row < lastRow; row++) {
    var f: array<f32, ${q}>;
    var ys = 0.0;
    getFilteredRow(row, lambda, &f, &ys);
    for (var a = 0u; a < Q; a++) {
      for (var b = 0u; b < Q; b++) {
        partial[base + a * Q + b] += f[a] * f[b];
      }
      partial[base + Q * Q + a] += f[a] * ys;
    }
    partial[base + Q * Q + Q] += ys * ys;
  }`,
        TILE_WORKGROUP_SIZE
      );
    const filteredMerge = (suffix: string, partial: GraphDataView, merged: GraphDataView) =>
      kernel(
        `filtered-moments-merge${suffix}`,
        [f32('partial', partial, 'read'), f32('filteredMoments', merged, 'read_write')],
        filteredMomentCount,
        constants,
        `${mergeLoop(filteredMomentCount)}
  filteredMoments[filteredMomentsOffset + index] = sum;`
      );

    const nodes: GPUCommandNode<Parameters>[] = [
      ...olsNodes,
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
  var sums: array<f32, ${meanCount}>;
  for (var row = firstRow; row < lastRow; row++) {
    for (var column = 0u; column < K; column++) {
      sums[column] = sums[column] + design[designOffset + row * K + column];
    }
    sums[K] = sums[K] + response[responseOffset + row];
  }
  for (var slot = 0u; slot < ${meanCount}u; slot++) {
    partial[partialOffset + index * ${meanCount}u + slot] = sums[slot];
  }`,
        TILE_WORKGROUP_SIZE
      ),
      kernel(
        'means-merge',
        [f32('partial', partialMeans, 'read'), f32('workspace', workspace, 'read_write')],
        meanCount,
        constants,
        `${mergeLoop(meanCount)}
  workspace[workspaceOffset + index] = sum / f32(ROWS);`
      ),
      // ||W||_inf, the largest absolute row sum, bounds lambda by 0.99 / ||W||_inf.
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
        TILE_WORKGROUP_SIZE
      ),
      kernel(
        'weights-norm-merge',
        [f32('partial', partialNorm, 'read'), f32('workspace', workspace, 'read_write')],
        1,
        `${constants}
${finiteDeclarations}`,
        `var largest = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    largest = max(largest, partial[partialOffset + tile]);
  }
  let valid = largest > 0.0 && isFiniteValue(largest);
  workspace[workspaceOffset + W_BOUND] = select(getNaN(), ${getWGSLFloatLiteral(GPU_SPATIAL_ERROR_GM_LAMBDA_BOUND_FRACTION)} / largest, valid);`
      )
    ];

    // Traces of W for the homoskedastic variance.
    const tracePowerTraces = transient('trace-power-traces', TRACE_POWER_COUNT);
    const traceGram = transient('trace-gram-traces', 4);
    const traceNorm = transient('trace-infinity-norm', 1);
    const traceApproximation = createTransientView(graph, `${id}-trace-approximation`, 'uint32', 1);
    if (homoskedastic) {
      nodes.push(
        ...new GPUSpatialWeightsTraces({
          id: `${id}-traces`,
          weights,
          maximumPower: TRACE_POWER_COUNT,
          mode: props.traceMode,
          probeCount: props.traceProbeCount,
          seed: props.traceSeed,
          includeGramTraces: true,
          output: {
            powerTraces: tracePowerTraces,
            powerSums: transient('trace-power-sums', TRACE_POWER_COUNT),
            gramTraces: traceGram,
            infinityNorm: traceNorm,
            approximation: traceApproximation
          }
        }).getCommandNodes(graph)
      );
    }

    nodes.push(
      residualLag('', olsResiduals, lagResiduals),
      momentsTiles('', olsResiduals, lagResiduals, partialMoments),
      kernel(
        'moments-merge',
        [f32('partial', partialMoments, 'read'), f32('moments', moments, 'read_write')],
        MOMENT_SUM_COUNT,
        constants,
        `${mergeLoop(MOMENT_SUM_COUNT)}
  moments[momentsOffset + index] = sum;`
      ),
      kernel(
        'lambda-solve',
        [f32('moments', moments, 'read'), f32('workspace', workspace, 'read_write')],
        1,
        homoskedastic
          ? `${constants}
${finiteDeclarations}
${getHomoskedasticMomentsWGSL(false)}
${getLambdaScanWGSL()}`
          : `${constants}
${finiteDeclarations}
${getKelejianPruchaMomentsWGSL()}
${getLambdaScanWGSL()}`,
        homoskedastic ? 'solveFirstStepLambda();' : 'solveLambda();'
      ),
      filteredTiles('', 'W_LAMBDA', partialFiltered),
      filteredMerge('', partialFiltered, filteredMoments),
      kernel(
        'filtered-solve',
        [
          f32('filteredMoments', filteredMoments, 'read'),
          f32('workspace', workspace, 'read_write')
        ],
        1,
        `${solveDeclarations}
${getSolveWGSL(q, 'filteredMoments', 'solveFiltered', false)}`,
        `var beta: array<f32, ${q}>;
  var inverseDiagonal: array<f32, ${q}>;
  var interceptQuadratic = 0.0;
  var conditioning: array<f32, 3>;
  let ok = solveFiltered(&beta, &inverseDiagonal, &interceptQuadratic, &conditioning);
  var intercept = workspace[workspaceOffset + W_YBAR] + beta[0];
  for (var column = 0u; column < K; column++) {
    intercept -= workspace[workspaceOffset + column] * beta[1u + column];
  }
  workspace[workspaceOffset + W_INTERCEPT] = select(getNaN(), intercept, ok);
  for (var column = 0u; column < Q; column++) {
    workspace[workspaceOffset + W_BETA + column] = select(getNaN(), beta[column], ok);
  }`
      ),
      kernel(
        'residuals-tiles',
        [
          ...rowBindings,
          f32('residualsOut', residualView, 'read_write'),
          f32('partial', partialResiduals, 'read_write')
        ],
        tileCount,
        filteredDeclarations,
        `let lambda = workspace[workspaceOffset + W_LAMBDA];
  let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${RESIDUAL_SUM_COUNT}>;
  for (var row = firstRow; row < lastRow; row++) {
    var f: array<f32, ${q}>;
    var ys = 0.0;
    getFilteredRow(row, lambda, &f, &ys);
    var filtered = ys;
    var predicted = workspace[workspaceOffset + W_INTERCEPT];
    for (var column = 0u; column < Q; column++) {
      filtered -= f[column] * workspace[workspaceOffset + W_BETA + column];
    }
    for (var column = 0u; column < K; column++) {
      predicted += design[designOffset + row * K + column] * workspace[workspaceOffset + W_BETA + 1u + column];
    }
    let residual = response[responseOffset + row] - predicted;
    let centeredResponse = response[responseOffset + row] - workspace[workspaceOffset + W_YBAR];
    residualsOut[residualsOutOffset + row] = residual;
    sums[0] += filtered * filtered;
    sums[1] += residual;
    sums[2] += residual * residual;
    sums[3] += centeredResponse * residual;
    sums[4] += centeredResponse * centeredResponse;
  }
  for (var slot = 0u; slot < ${RESIDUAL_SUM_COUNT}u; slot++) {
    partial[partialOffset + index * ${RESIDUAL_SUM_COUNT}u + slot] = sums[slot];
  }`,
        TILE_WORKGROUP_SIZE
      ),
      kernel(
        'residuals-merge',
        [f32('partial', partialResiduals, 'read'), f32('residualSums', residualSums, 'read_write')],
        RESIDUAL_SUM_COUNT,
        constants,
        `${mergeLoop(RESIDUAL_SUM_COUNT)}
  residualSums[residualSumsOffset + index] = sum;`
      )
    );

    let finalFilteredMoments: GraphDataView = filteredMoments;
    if (homoskedastic) {
      // Second step: moments of the FGLS residuals u1, psi at lambda1, weighted lambda.
      const lagResiduals1 = transient('lag-residuals-fgls', rowCount);
      const partialMoments1 = transient('partial-moments-fgls', tileCount * MOMENT_SUM_COUNT);
      const moments1 = transient('moments-fgls', MOMENT_SUM_COUNT);
      const partialPowers1 = transient('partial-power-sums-first', tileCount * POWER_SUM_COUNT);
      const powers1 = transient('power-sums-first', POWER_SUM_COUNT);
      const partialPowers2 = transient('partial-power-sums-final', tileCount * POWER_SUM_COUNT);
      const partialFiltered2 = transient(
        'partial-filtered-moments-final',
        tileCount * filteredMomentCount
      );
      const filteredMoments2 = transient('filtered-moments-final', filteredMomentCount);
      finalFilteredMoments = filteredMoments2;
      const powerTiles = (suffix: string, lambdaSlot: string, partial: GraphDataView) =>
        kernel(
          `power-sums-tiles${suffix}`,
          [
            f32('residual', residualView, 'read'),
            f32('lag', lagResiduals1, 'read'),
            f32('workspace', workspace, 'read'),
            f32('partial', partial, 'read_write')
          ],
          tileCount,
          constants,
          `let lambda = workspace[workspaceOffset + ${lambdaSlot}];
  let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${POWER_SUM_COUNT}>;
  for (var row = firstRow; row < lastRow; row++) {
    let filtered = residual[residualOffset + row] - lambda * lag[lagOffset + row];
    let square = filtered * filtered;
    sums[0] += square;
    sums[1] += square * filtered;
    sums[2] += square * square;
  }
  for (var slot = 0u; slot < ${POWER_SUM_COUNT}u; slot++) {
    partial[partialOffset + index * ${POWER_SUM_COUNT}u + slot] = sums[slot];
  }`,
          TILE_WORKGROUP_SIZE
        );
      nodes.push(
        residualLag('-fgls', residualView, lagResiduals1),
        momentsTiles('-fgls', residualView, lagResiduals1, partialMoments1),
        kernel(
          'moments-merge-fgls',
          [f32('partial', partialMoments1, 'read'), f32('moments', moments1, 'read_write')],
          MOMENT_SUM_COUNT,
          constants,
          `${mergeLoop(MOMENT_SUM_COUNT)}
  moments[momentsOffset + index] = sum;`
        ),
        powerTiles('-first', 'W_LAMBDA', partialPowers1),
        kernel(
          'power-sums-merge-first',
          [f32('partial', partialPowers1, 'read'), f32('powers', powers1, 'read_write')],
          POWER_SUM_COUNT,
          constants,
          `${mergeLoop(POWER_SUM_COUNT)}
  powers[powersOffset + index] = sum;`
        ),
        kernel(
          'lambda-solve-weighted',
          [
            f32('moments', moments1, 'read'),
            f32('powers', powers1, 'read'),
            f32('powerTraces', tracePowerTraces, 'read'),
            f32('gramTraces', traceGram, 'read'),
            f32('infinityNorm', traceNorm, 'read'),
            {name: 'approximation', view: traceApproximation, type: 'u32', access: 'read'},
            f32('workspace', workspace, 'read_write')
          ],
          1,
          `${constants}
${finiteDeclarations}
${getHomoskedasticMomentsWGSL(true)}
${getLambdaScanWGSL()}`,
          'solveWeightedLambda();'
        ),
        powerTiles('-final', 'W_LAMBDA_FINAL', partialPowers2),
        kernel(
          'power-sums-merge-final',
          [f32('partial', partialPowers2, 'read'), f32('workspace', workspace, 'read_write')],
          POWER_SUM_COUNT,
          constants,
          `${mergeLoop(POWER_SUM_COUNT)}
  workspace[workspaceOffset + W_POWER_SUMS + index] = sum;`
        ),
        filteredTiles('-final', 'W_LAMBDA_FINAL', partialFiltered2),
        filteredMerge('-final', partialFiltered2, filteredMoments2)
      );
    }

    nodes.push(
      kernel(
        'finish',
        [
          f32('filteredMoments', filteredMoments, 'read'),
          ...(homoskedastic ? [f32('filteredMomentsFinal', finalFilteredMoments, 'read')] : []),
          f32('workspace', workspace, 'read'),
          f32('residualSums', residualSums, 'read'),
          {name: 'olsStatus', view: olsStatus, type: 'u32', access: 'read'},
          f32('tableOut', output.table, 'read_write'),
          f32('summaryOut', output.summary, 'read_write'),
          {name: 'statusOut', view: output.status, type: 'u32', access: 'read_write'}
        ],
        1,
        `${solveDeclarations}
${getSymmetricEigenvalueRangeWGSL(q)}
${getBelsleyConditionNumberWGSL(q - 1)}
${
  homoskedastic
    ? `${getSolveWGSL(q, 'filteredMoments', 'solveFiltered', false)}
${getSolveWGSL(q, 'filteredMomentsFinal', 'solveFilteredFinal', true)}
${getHomoskedasticVarianceWGSL()}`
    : getSolveWGSL(q, 'filteredMoments', 'solveFilteredFinal', true)
}
${getFinishWGSL(k, homoskedastic)}`,
        'finish();'
      )
    );
    return nodes;
  }
}

/**
 * `getFilteredRow`: the filtered, centered regressors `f = [1 - lambda s_i, xs]` and response
 * `ys` of one row at `lambda`. Kernels that include it bind `offsets`, `neighbors`, `weights`,
 * `design`, `response` and `workspace` (means).
 */
function getFilteredRowWGSL(): string {
  return /* wgsl */ `
fn getFilteredRow(row: u32, lambda: f32, f: ptr<function, array<f32, Q>>, ys: ptr<function, f32>) {
  let responseMean = workspace[workspaceOffset + W_YBAR];
  var rowSum = 0.0;
  var lagResponse = 0.0;
  var lagged: array<f32, K>;
  for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= ROWS) {
      continue;
    }
    let weight = weights[weightsOffset + slot];
    rowSum += weight;
    lagResponse += weight * (response[responseOffset + neighbor] - responseMean);
    for (var column = 0u; column < K; column++) {
      lagged[column] += weight * (design[designOffset + neighbor * K + column] - workspace[workspaceOffset + column]);
    }
  }
  (*f)[0] = 1.0 - lambda * rowSum;
  for (var column = 0u; column < K; column++) {
    let centered = design[designOffset + row * K + column] - workspace[workspaceOffset + column];
    (*f)[1u + column] = centered - lambda * lagged[column];
  }
  *ys = (response[responseOffset + row] - responseMean) - lambda * lagResponse;
}
`;
}

/**
 * `scanLambda(bound) -> vec2(lambda, objective)`: the global minimum of `getMomentObjective` on
 * `[-bound, bound]`. Both estimators supply `getMomentObjective(lambda)` and `getMomentSlope(lambda)`
 * (half the derivative of the objective). The scan finds sign changes of the slope on a fixed grid
 * (including both ends), refines each bracketed minimum by bisection and keeps the lowest
 * objective; ties go to the lowest `lambda`.
 */
function getLambdaScanWGSL(): string {
  return /* wgsl */ `
fn scanLambda(bound: f32) -> vec2<f32> {
  var bestLambda = -bound;
  var bestObjective = getMomentObjective(-bound);
  let endObjective = getMomentObjective(bound);
  if (endObjective < bestObjective) {
    bestLambda = bound;
    bestObjective = endObjective;
  }
  var previousLambda = -bound;
  var previousSlope = getMomentSlope(-bound);
  for (var scanStep = 1u; scanStep <= ${LAMBDA_SCAN_STEPS}u; scanStep++) {
    let lambda = bound * (-1.0 + 2.0 * f32(scanStep) / ${LAMBDA_SCAN_STEPS}.0);
    let slope = getMomentSlope(lambda);
    if (previousSlope < 0.0 && slope >= 0.0) {
      var low = previousLambda;
      var high = lambda;
      for (var iteration = 0u; iteration < ${LAMBDA_BISECTION_STEPS}u; iteration++) {
        let middle = 0.5 * (low + high);
        if (getMomentSlope(middle) < 0.0) {
          low = middle;
        } else {
          high = middle;
        }
      }
      let candidate = 0.5 * (low + high);
      let objective = getMomentObjective(candidate);
      if (objective < bestObjective) {
        bestLambda = candidate;
        bestObjective = objective;
      }
    }
    previousLambda = lambda;
    previousSlope = slope;
  }
  return vec2<f32>(bestLambda, bestObjective);
}
`;
}

/**
 * `solveLambda`: the Kelejian-Prucha moment system of `GM_Error` with `sigma2` projected out.
 * Writes `lambda` and the scaled objective, NaN for non-finite or empty moments or bound.
 */
function getKelejianPruchaMomentsWGSL(): string {
  return /* wgsl */ `
var<private> momentA: vec3<f32>;
var<private> momentB: vec3<f32>;
var<private> momentG: vec3<f32>;
var<private> momentSigma: vec3<f32>;

// Moment residual with sigma2 at its constrained optimum: least squares in sigma2, clamped at zero.
fn getMomentResidual(lambda: f32) -> vec3<f32> {
  let free = momentG - momentA * lambda - momentB * (lambda * lambda);
  let sigma = max(dot(momentSigma, free) / dot(momentSigma, momentSigma), 0.0);
  return free - momentSigma * sigma;
}

fn getMomentObjective(lambda: f32) -> f32 {
  let residual = getMomentResidual(lambda);
  return dot(residual, residual);
}

// Half the derivative of the objective (envelope theorem: sigma2 is optimal or fixed at zero).
fn getMomentSlope(lambda: f32) -> f32 {
  return -dot(momentA + momentB * (2.0 * lambda), getMomentResidual(lambda));
}

fn solveLambda() {
  let n = f32(ROWS);
  let u2 = moments[momentsOffset];
  var finite = u2 > 0.0;
  for (var slot = 0u; slot < ${MOMENT_SUM_COUNT}u; slot++) {
    finite = finite && isFiniteValue(moments[momentsOffset + slot]);
  }
  let bound = workspace[workspaceOffset + W_BOUND];
  if (!finite || !isFiniteValue(bound)) {
    workspace[workspaceOffset + W_LAMBDA] = getNaN();
    workspace[workspaceOffset + W_OBJECTIVE] = getNaN();
    return;
  }
  let uwu = moments[momentsOffset + 1u];
  let wu2 = moments[momentsOffset + 2u];
  let uwwu = moments[momentsOffset + 3u];
  let wuwwu = moments[momentsOffset + 4u];
  let wwu2 = moments[momentsOffset + 5u];
  let scale = 1.0 / u2;
  momentSigma = vec3<f32>(1.0, moments[momentsOffset + 6u] / n, 0.0);
  momentA = vec3<f32>(2.0 * uwu, 2.0 * wuwwu, uwwu + wu2) * scale;
  momentB = vec3<f32>(-wu2, -wwu2, -wuwwu) * scale;
  momentG = vec3<f32>(u2, wu2, uwu) * scale;
  let best = scanLambda(bound);
  workspace[workspaceOffset + W_LAMBDA] = best.x;
  workspace[workspaceOffset + W_OBJECTIVE] = best.y;
}
`;
}

/**
 * The two-moment system of `GM_Error_Hom`: `r(l) = G [l, l^2]' - g` with
 * `G = [[2 ((Wu)'WWu - (t/n) u'Wu) / s, -((WWu)'WWu - (t/n) (Wu)'Wu) / s], [u'WWu + (Wu)'Wu,
 * -(Wu)'WWu]] / n`, `g = [((Wu)'Wu - (t/n) u'u) / s, u'Wu] / n`, `t = tr(W'W)`, `s = 1 + (t/n)^2`,
 * minimized as `r' M r` with `M` the identity (first step) or `psi^-1` (weighted step). With
 * `weighted`, also declares the trace terms and the `psi` helpers; the weighted solve additionally
 * stores `G`, the trace terms, the final `lambda` and the approximation flag in the workspace.
 */
function getHomoskedasticMomentsWGSL(weighted: boolean): string {
  return /* wgsl */ `
var<private> momentA: vec2<f32>;
var<private> momentB: vec2<f32>;
var<private> momentTarget: vec2<f32>;
var<private> momentWeight: mat2x2<f32>;

fn getMomentResidual(lambda: f32) -> vec2<f32> {
  return momentA * lambda + momentB * (lambda * lambda) - momentTarget;
}

fn getMomentObjective(lambda: f32) -> f32 {
  let residual = getMomentResidual(lambda);
  return dot(residual, momentWeight * residual);
}

// Half the derivative of the objective: (dr/dlambda)' M r.
fn getMomentSlope(lambda: f32) -> f32 {
  return dot(momentA + momentB * (2.0 * lambda), momentWeight * getMomentResidual(lambda));
}

// Fills momentA, momentB, momentTarget from the seven moment sums; false if they are not usable.
fn setMomentSystem() -> bool {
  let n = f32(ROWS);
  let u2 = moments[momentsOffset];
  var finite = u2 > 0.0;
  for (var slot = 0u; slot < ${MOMENT_SUM_COUNT}u; slot++) {
    finite = finite && isFiniteValue(moments[momentsOffset + slot]);
  }
  if (!finite) {
    return false;
  }
  let uwu = moments[momentsOffset + 1u];
  let wu2 = moments[momentsOffset + 2u];
  let uwwu = moments[momentsOffset + 3u];
  let wuwwu = moments[momentsOffset + 4u];
  let wwu2 = moments[momentsOffset + 5u];
  let t = moments[momentsOffset + 6u];
  let ratio = t / n;
  let scale = 1.0 + ratio * ratio;
  momentA = vec2<f32>(2.0 * (wuwwu - ratio * uwu) / scale, uwwu + wu2) / n;
  momentB = vec2<f32>(-(wwu2 - ratio * wu2) / scale, -wuwwu) / n;
  momentTarget = vec2<f32>((wu2 - ratio * u2) / scale, uwu) / n;
  return true;
}
${
  weighted
    ? /* wgsl */ `
var<private> traceTerms: vec4<f32>;

// psi (including the 1 / n) from sigma2 and the fourth moment mu4 of the filtered residuals.
fn getPsi(n: f32, sigma: f32, mu4: f32) -> mat2x2<f32> {
  let psi11 = 2.0 * sigma * sigma * traceTerms.x + (mu4 - 3.0 * sigma * sigma) * traceTerms.w;
  let psi12 = sigma * sigma * traceTerms.y;
  let psi22 = sigma * sigma * traceTerms.z;
  return mat2x2<f32>(psi11 / n, psi12 / n, psi12 / n, psi22 / n);
}

fn getPsiDeterminant(psi: mat2x2<f32>) -> f32 {
  return psi[0][0] * psi[1][1] - psi[0][1] * psi[1][0];
}

fn invertPsi(psi: mat2x2<f32>) -> mat2x2<f32> {
  return mat2x2<f32>(psi[1][1], -psi[0][1], -psi[1][0], psi[0][0]) * (1.0 / getPsiDeterminant(psi));
}

fn solveWeightedLambda() {
  let n = f32(ROWS);
  let nan = getNaN();
  workspace[workspaceOffset + W_LAMBDA_FINAL] = nan;
  workspace[workspaceOffset + W_OBJECTIVE_FINAL] = nan;
  workspace[workspaceOffset + W_APPROXIMATION] = f32(approximation[approximationOffset]);
  let bound = workspace[workspaceOffset + W_BOUND];
  var finite = isFiniteValue(bound);
  for (var slot = 0u; slot < ${POWER_SUM_COUNT}u; slot++) {
    finite = finite && isFiniteValue(powers[powersOffset + slot]);
  }
  for (var slot = 0u; slot < 2u; slot++) {
    finite = finite && isFiniteValue(powerTraces[powerTracesOffset + slot]);
  }
  for (var slot = 0u; slot < 4u; slot++) {
    finite = finite && isFiniteValue(gramTraces[gramTracesOffset + slot]);
  }
  if (!finite || !setMomentSystem()) {
    return;
  }
  // powerTraces hold tr((W/s)^q): tr(W) = s c_1 and tr(W^2) = s^2 c_2.
  let norm = infinityNorm[infinityNormOffset];
  // Trace terms: tr(A1 A1), tr(A1 2 A2), tr(2 A2 A2) and diag(A1)'diag(A1).
  let t = moments[momentsOffset + 6u];
  let ratio = t / n;
  let scale = 1.0 + ratio * ratio;
  let scaleSquared = scale * scale;
  traceTerms = vec4<f32>(
    (gramTraces[gramTracesOffset + 2u] - t * ratio) / scaleSquared,
    (2.0 * gramTraces[gramTracesOffset + 1u] - 2.0 * ratio * powerTraces[powerTracesOffset] * norm) / scale,
    powerTraces[powerTracesOffset + 1u] * norm * norm + t,
    (gramTraces[gramTracesOffset + 3u] - t * ratio) / scaleSquared
  );
  // psi at lambda1 from the power sums of u_s = u1 - lambda1 W u1.
  let sigma = powers[powersOffset] / n;
  let psi = getPsi(n, sigma, powers[powersOffset + 2u] / n);
  let determinant = getPsiDeterminant(psi);
  if (!(determinant > 0.0) || !isFiniteValue(determinant)) {
    return;
  }
  var weight = invertPsi(psi);
  // Normalize the weight to unit trace: same minimizer, well-scaled objective.
  let weightScale = weight[0][0] + weight[1][1];
  weight = weight * (1.0 / weightScale);
  momentWeight = weight;
  let best = scanLambda(bound);
  workspace[workspaceOffset + W_LAMBDA_FINAL] = best.x;
  workspace[workspaceOffset + W_OBJECTIVE_FINAL] = best.y * weightScale;
  workspace[workspaceOffset + W_MOMENT_MATRIX] = momentA.x;
  workspace[workspaceOffset + W_MOMENT_MATRIX + 1u] = momentA.y;
  workspace[workspaceOffset + W_MOMENT_MATRIX + 2u] = momentB.x;
  workspace[workspaceOffset + W_MOMENT_MATRIX + 3u] = momentB.y;
  for (var slot = 0u; slot < 4u; slot++) {
    workspace[workspaceOffset + W_TRACE_TERMS + slot] = traceTerms[slot];
  }
}
`
    : /* wgsl */ `
fn solveFirstStepLambda() {
  let bound = workspace[workspaceOffset + W_BOUND];
  if (!isFiniteValue(bound) || !setMomentSystem()) {
    workspace[workspaceOffset + W_LAMBDA] = getNaN();
    workspace[workspaceOffset + W_OBJECTIVE] = getNaN();
    return;
  }
  momentWeight = mat2x2<f32>(1.0, 0.0, 0.0, 1.0);
  let best = scanLambda(bound);
  workspace[workspaceOffset + W_LAMBDA] = best.x;
  workspace[workspaceOffset + W_OBJECTIVE] = best.y;
}
`
}
`;
}

/**
 * `getSolveWGSL`: `functionName(beta, inverseDiagonal, interceptQuadratic, conditioning)`, the
 * scaled Cholesky solve of the filtered normal equations `Xs'Xs b = Xs'ys` read from the binding
 * `momentsName` (centered-model intercept first), the diagonal of the inverse and
 * `g' (Xs'Xs)^-1 g` with `g = (1, -means)`, the variance factor of the uncentered intercept.
 * With `withConditioning`, `conditioning` receives the Belsley condition number of the raw
 * filtered `[1, X]` design (from the Cholesky factor of the filtered Gram and the means,
 * `getBelsleyConditionNumberWGSL`; the raw-Gram eigenvalue route is not resolvable in float32), the correlation
 * condition number of the factored matrix and the largest variance inflation factor over the
 * predictor columns. Kernels that include it bind `momentsName` and `workspace`, and declare the
 * Cholesky and eigenvalue helpers.
 */
function getSolveWGSL(
  q: number,
  momentsName: string,
  functionName: string,
  withConditioning: boolean
): string {
  const moment = (index: string) => `${momentsName}[${momentsName}Offset + ${index}]`;
  return /* wgsl */ `
fn ${functionName}(
  beta: ptr<function, array<f32, ${q}>>,
  inverseDiagonal: ptr<function, array<f32, ${q}>>,
  interceptQuadratic: ptr<function, f32>,
  conditioning: ptr<function, array<f32, 3>>
) -> bool {
  var scale: array<f32, ${q}>;
  for (var a = 0u; a < Q; a++) {
    let diagonal = ${moment('a * Q + a')};
    if (!(diagonal > 0.0) || !isFiniteValue(diagonal)) {
      return false;
    }
    scale[a] = 1.0 / sqrt(diagonal);
  }
  var factor: array<f32, ${q * q}>;
  var rhs: array<f32, ${q}>;
  for (var a = 0u; a < Q; a++) {
    for (var b = 0u; b < Q; b++) {
      factor[a * Q + b] = ${moment('a * Q + b')} * scale[a] * scale[b];
    }
    rhs[a] = ${moment('Q * Q + a')} * scale[a];
  }
${
  withConditioning
    ? /* wgsl */ `
  // Correlation condition number of the unit-diagonal matrix that is factored.
  var correlation = factor;
  let correlationRange = getSymmetricEigenvalueRange_${q}(&correlation);
  (*conditioning)[1] = correlationRange.y / max(correlationRange.x, 1.0e-30);
`
    : ''
}
  if (!choleskyFactor_${q}(&factor)) {
    return false;
  }
${
  withConditioning
    ? /* wgsl */ `
  // Belsley condition number of the raw filtered design [f0, f_j + m_j f0] (f0 is the filtered
  // constant, not orthogonal to the centered columns), without forming its Gram matrix.
  // With the Cholesky factor L of the unit-diagonal Gram of [f0, f_1..f_K] the raw column j is
  // inPlane_j u0 + g_j sum_i L[j][i] e_i (u0 the unit f0, e_i orthonormal), so the helper's
  // alpha_j = inPlane_j / N_j, beta_j = g_j / N_j and factor = L without its first row and column.
  var belsleyFactor: array<f32, ${(q - 1) * (q - 1)}>;
  var belsleyAlpha: array<f32, ${q - 1}>;
  var belsleyBeta: array<f32, ${q - 1}>;
  for (var j = 0u; j < K; j++) {
    let column = 1.0 / scale[1u + j];
    let inPlane = workspace[workspaceOffset + j] / scale[0] + column * factor[(1u + j) * Q];
    var tail = 0.0;
    for (var i = 0u; i <= j; i++) {
      belsleyFactor[j * K + i] = factor[(1u + j) * Q + 1u + i];
      tail = tail + factor[(1u + j) * Q + 1u + i] * factor[(1u + j) * Q + 1u + i];
    }
    let rawNorm = sqrt(inPlane * inPlane + column * column * tail);
    belsleyAlpha[j] = inPlane / rawNorm;
    belsleyBeta[j] = column / rawNorm;
  }
  (*conditioning)[0] = getBelsleyConditionNumber_${q - 1}(&belsleyFactor, &belsleyAlpha, &belsleyBeta);
  var inflation = 0.0;
  for (var b = 1u; b < Q; b++) {
    inflation = max(inflation, choleskyInverseDiagonal_${q}(&factor, b));
  }
  (*conditioning)[2] = inflation;
`
    : ''
}
  choleskySolve_${q}(&factor, &rhs);
  var meansScaled: array<f32, ${q}>;
  meansScaled[0] = scale[0];
  for (var b = 0u; b < K; b++) {
    meansScaled[1u + b] = -workspace[workspaceOffset + b] * scale[1u + b];
  }
  var solved = meansScaled;
  choleskySolve_${q}(&factor, &solved);
  var quadratic = 0.0;
  for (var b = 0u; b < Q; b++) {
    (*beta)[b] = rhs[b] * scale[b];
    (*inverseDiagonal)[b] = choleskyInverseDiagonal_${q}(&factor, b) * scale[b] * scale[b];
    quadratic += meansScaled[b] * solved[b];
  }
  *interceptQuadratic = quadratic;
  return true;
}
`;
}

/**
 * `getHomoskedasticVarianceWGSL`: `psi` at the final `lambda` and the variance of `lambda`,
 * `(j' psi^-1 j)^-1 / n`, for the finish kernel. Reads the workspace slots the weighted solve wrote.
 */
function getHomoskedasticVarianceWGSL(): string {
  return /* wgsl */ `
var<private> traceTerms: vec4<f32>;

fn getPsi(n: f32, sigma: f32, mu4: f32) -> mat2x2<f32> {
  let psi11 = 2.0 * sigma * sigma * traceTerms.x + (mu4 - 3.0 * sigma * sigma) * traceTerms.w;
  let psi12 = sigma * sigma * traceTerms.y;
  let psi22 = sigma * sigma * traceTerms.z;
  return mat2x2<f32>(psi11 / n, psi12 / n, psi12 / n, psi22 / n);
}

// Standard error of lambda, NaN if psi is not positive definite.
fn getLambdaStandardError(n: f32, lambda: f32, sigma: f32, mu4: f32) -> f32 {
  for (var slot = 0u; slot < 4u; slot++) {
    traceTerms[slot] = workspace[workspaceOffset + W_TRACE_TERMS + slot];
  }
  let psi = getPsi(n, sigma, mu4);
  let determinant = psi[0][0] * psi[1][1] - psi[0][1] * psi[1][0];
  if (!(determinant > 0.0) || !isFiniteValue(determinant)) {
    return getNaN();
  }
  let inverse = mat2x2<f32>(psi[1][1], -psi[0][1], -psi[1][0], psi[0][0]) * (1.0 / determinant);
  let j = vec2<f32>(
    workspace[workspaceOffset + W_MOMENT_MATRIX] + 2.0 * lambda * workspace[workspaceOffset + W_MOMENT_MATRIX + 2u],
    workspace[workspaceOffset + W_MOMENT_MATRIX + 1u] + 2.0 * lambda * workspace[workspaceOffset + W_MOMENT_MATRIX + 3u]
  );
  let quadratic = dot(j, inverse * j);
  return sqrt(1.0 / (quadratic * n));
}
`;
}

/** The finish kernel: coefficient table, summary and status. */
function getFinishWGSL(k: number, homoskedastic: boolean): string {
  const summary = (slot: number) => `summaryOut[summaryOutOffset + ${slot}u]`;
  return /* wgsl */ `
${ERFC_VALUE_WGSL}

fn fail(status: u32) {
  let nan = getNaN();
  for (var slot = 0u; slot < ${(k + 2) * 4}u; slot++) {
    tableOut[tableOutOffset + slot] = nan;
  }
  for (var slot = 0u; slot < ${GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH}u; slot++) {
    summaryOut[summaryOutOffset + slot] = nan;
  }
  summaryOut[summaryOutOffset] = f32(ROWS);
  statusOut[statusOutOffset] = status;
}

fn finish() {
  let n = f32(ROWS);
  if (!(n > f32(Q + 1u))) {
    fail(2u);
    return;
  }
  if (olsStatus[olsStatusOffset] != 0u) {
    fail(select(1u, 2u, olsStatus[olsStatusOffset] == 2u));
    return;
  }
  let lambda = workspace[workspaceOffset + ${homoskedastic ? 'W_LAMBDA_FINAL' : 'W_LAMBDA'}];
  if (!isFiniteValue(lambda)) {
    fail(3u);
    return;
  }
  var beta: array<f32, ${k + 1}>;
  var inverseDiagonal: array<f32, ${k + 1}>;
  var interceptQuadratic = 0.0;
  var conditioning: array<f32, 3>;
${
  homoskedastic
    ? /* wgsl */ `
  // The reported betas are the FGLS fit at lambda1; their variance uses the system at lambda.
  var unusedInverse: array<f32, ${k + 1}>;
  var unusedQuadratic = 0.0;
  if (!solveFiltered(&beta, &unusedInverse, &unusedQuadratic, &conditioning)) {
    fail(1u);
    return;
  }
  var finalBeta: array<f32, ${k + 1}>;
  if (!solveFilteredFinal(&finalBeta, &inverseDiagonal, &interceptQuadratic, &conditioning)) {
    fail(1u);
    return;
  }
  let sigmaSquared = workspace[workspaceOffset + W_POWER_SUMS] / n;
  let fourthMoment = workspace[workspaceOffset + W_POWER_SUMS + 2u] / n;
  let lambdaStandardError = getLambdaStandardError(n, lambda, sigmaSquared, fourthMoment);`
    : /* wgsl */ `
  if (!solveFilteredFinal(&beta, &inverseDiagonal, &interceptQuadratic, &conditioning)) {
    fail(1u);
    return;
  }
  let sigmaSquared = residualSums[residualSumsOffset] / n;`
}
  let filteredSquares = residualSums[residualSumsOffset];
  let residualTotal = residualSums[residualSumsOffset + 1u];
  let residualSquares = residualSums[residualSumsOffset + 2u];
  let responseResidual = residualSums[residualSumsOffset + 3u];
  let responseSquares = residualSums[residualSumsOffset + 4u];
  if (!isFiniteValue(filteredSquares) || !isFiniteValue(residualSquares) || !isFiniteValue(responseSquares)) {
    fail(3u);
    return;
  }
  for (var row = 0u; row < Q; row++) {
    // Row 0 is the intercept, rows 1..K the predictors.
    let coefficient = select(beta[row], workspace[workspaceOffset + W_INTERCEPT], row == 0u);
    let variance = sigmaSquared * select(inverseDiagonal[row], interceptQuadratic, row == 0u);
    let standardError = sqrt(variance);
    let z = coefficient / standardError;
    tableOut[tableOutOffset + row * 4u] = coefficient;
    tableOut[tableOutOffset + row * 4u + 1u] = standardError;
    tableOut[tableOutOffset + row * 4u + 2u] = z;
    tableOut[tableOutOffset + row * 4u + 3u] = erfcValue(abs(z) * 0.70710678);
  }
  let nan = getNaN();
${
  homoskedastic
    ? /* wgsl */ `
  let lambdaZ = lambda / lambdaStandardError;
  tableOut[tableOutOffset + Q * 4u] = lambda;
  tableOut[tableOutOffset + Q * 4u + 1u] = lambdaStandardError;
  tableOut[tableOutOffset + Q * 4u + 2u] = lambdaZ;
  tableOut[tableOutOffset + Q * 4u + 3u] = erfcValue(abs(lambdaZ) * 0.70710678);`
    : /* wgsl */ `
  tableOut[tableOutOffset + Q * 4u] = lambda;
  tableOut[tableOutOffset + Q * 4u + 1u] = nan;
  tableOut[tableOutOffset + Q * 4u + 2u] = nan;
  tableOut[tableOutOffset + Q * 4u + 3u] = nan;`
}
  // Squared correlation of y and y - u: the response is centered, so Syy = responseSquares.
  let predictedVariance = responseSquares - 2.0 * responseResidual + residualSquares - residualTotal * residualTotal / n;
  let covariance = responseSquares - responseResidual;
  let bound = workspace[workspaceOffset + W_BOUND];
  summaryOut[summaryOutOffset] = n;
  ${summary(1)} = lambda;
  ${summary(2)} = sigmaSquared;
  ${summary(3)} = covariance * covariance / (responseSquares * predictedVariance);
  ${summary(4)} = residualSquares;
  ${summary(5)} = workspace[workspaceOffset + ${homoskedastic ? 'W_OBJECTIVE_FINAL' : 'W_OBJECTIVE'}];
  ${summary(GPU_SPATIAL_ERROR_GM_SUMMARY_CONDITION_NUMBER)} = conditioning[0];
  ${summary(GPU_SPATIAL_ERROR_GM_SUMMARY_CORRELATION_CONDITION_NUMBER)} = conditioning[1];
  ${summary(GPU_SPATIAL_ERROR_GM_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR)} = conditioning[2];
  ${summary(GPU_SPATIAL_ERROR_GM_SUMMARY_PRECISION_WARNING)} =
    select(0.0, 1.0, conditioning[1] > ${getWGSLFloatLiteral(GPU_REGRESSION_PRECISION_CONDITION_LIMIT)});
  ${summary(GPU_SPATIAL_ERROR_GM_SUMMARY_LAMBDA_BOUND)} = bound;
  ${summary(GPU_SPATIAL_ERROR_GM_SUMMARY_APPROXIMATION)} = ${homoskedastic ? 'workspace[workspaceOffset + W_APPROXIMATION]' : '0.0'};
  ${summary(GPU_SPATIAL_ERROR_GM_SUMMARY_LAMBDA_AT_BOUND)} = select(0.0, 1.0, abs(lambda) >= bound * 0.9999);
  statusOut[statusOutOffset] = 0u;
}
`;
}
