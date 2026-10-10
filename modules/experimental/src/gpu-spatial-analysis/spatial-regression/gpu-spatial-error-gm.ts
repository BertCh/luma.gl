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
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {type GPUSpatialWeights, validateGPUSpatialWeights} from '../spatial-weights/index';
import {GPUOrdinaryLeastSquares} from './gpu-ordinary-least-squares';
import {GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH} from './ordinary-least-squares-parameters';
import {
  GPU_SPATIAL_ERROR_GM_MAXIMUM_PREDICTOR_COUNT,
  GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH,
  GPU_SPATIAL_ERROR_GM_TABLE_STRIDE
} from './spatial-error-gm-parameters';
import {getCholeskyWGSL} from './spatial-regression-solve';
import type {GPUSolverStatusPort} from '../contracts/index';

const OPERATION = 'GPUSpatialErrorGM';
const TILE_WORKGROUP_SIZE = 64;
const MAXIMUM_DEFAULT_TILE_COUNT = 4096;
const MINIMUM_DEFAULT_TILE_ROWS = 64;
/** Moment sums: `u'u`, `u'Wu`, `(Wu)'Wu`, `u'WWu`, `(Wu)'WWu`, `(WWu)'WWu`, `tr W'W`. */
const MOMENT_SUM_COUNT = 7;
/** Residual sums: `e'e`, `sum u`, `u'u`, `sum (y - ybar) u`, `sum (y - ybar)^2`. */
const RESIDUAL_SUM_COUNT = 5;
/** Hard bound of `lambda`, as spreg `optim_moments` (`(-0.99, 0.99)`). */
const LAMBDA_BOUND = 0.99;
/** Grid intervals scanned for stationary points of the moment objective on `[-bound, bound]`. */
const LAMBDA_SCAN_STEPS = 1024;
/** Bisection steps that refine each bracketed minimum. */
const LAMBDA_BISECTION_STEPS = 40;

/** Caller-owned outputs of {@link GPUSpatialErrorGM}. */
export type GPUSpatialErrorGMOutput = GPUSolverStatusPort & {
  /**
   * `(predictorCount + 2) * 4` float32 values: one row per coefficient in spreg order (intercept,
   * one per predictor, then the spatial error parameter `lambda` last), each row holding the
   * coefficient, standard error, z statistic and two-sided normal p-value. The `lambda` row holds
   * its estimate with NaN for the other three, as spreg `GM_Error` does not estimate its variance.
   */
  table: GraphDataView<'float32'>;
  /** At least 6 float32 values; see the `GPU_SPATIAL_ERROR_GM_SUMMARY_*` slots. */
  summary: GraphDataView<'float32'>;
  /** Optional per-row residuals `u = y - X b` (spreg `u`); NaN for a failed fit. */
  residuals?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUSpatialErrorGM}.
 *
 * Per-frame (no recompile): the contents of every input buffer. Topology (needs a new graph):
 * `predictorCount`, the row count, the weights views, whether `residuals` is present and
 * `tileRowCount`.
 */
export type GPUSpatialErrorGMProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'spatial-error-gm'`. */
  id?: string;
  /**
   * Square weights `W`; rows index the regression rows. Any sparsity pattern works (no symmetry
   * is required; `tr(W'W)` is `sum w_ij^2`, so nothing reads `W'`), and rows need not be standardized.
   */
  weights: GPUSpatialWeights;
  /** Row-major predictors, `predictors[row * predictorCount + column]`, without an intercept. */
  predictors: GraphDataView<'float32'>;
  /** Response `y`, one value per row. */
  response: GraphDataView<'float32'>;
  /** Number of predictor columns, compile-time, 1 to 8. The intercept is implicit. */
  predictorCount: number;
  /**
   * Rows reduced by one invocation of the accumulation passes. Defaults to
   * `max(64, ceil(rows / 4096))`.
   */
  tileRowCount?: number;
  /** Caller-owned outputs. */
  output: GPUSpatialErrorGMOutput;
};

/**
 * Generalized-moments estimator of the spatial error model `y = X b + u`, `u = lambda W u + e`
 * (spreg 1.9.1 `GM_Error`: Kelejian and Prucha 1998, 1999, homoskedastic).
 *
 * 1. Ordinary least squares of `y` on `[1, X]` (the `GPUOrdinaryLeastSquares` contributor) gives
 *    residuals `u`.
 * 2. The three sample moments use `u`, `Wu` and `WWu`:
 *    `G [lambda, lambda^2, sigma2]' = g` with
 *    `G = [[2 u'Wu, -(Wu)'Wu, 1], [2 (Wu)'WWu, -(WWu)'WWu, tr(W'W) / n], [u'WWu + (Wu)'Wu, -(Wu)'WWu, 0]] / n`
 *    and `g = [u'u, (Wu)'Wu, u'Wu] / n` (spreg `_momentsGM_Error`). `lambda` minimizes
 *    `|G p - g|^2` over `[-0.99, 0.99]` with `sigma2 >= 0` (spreg `optim_moments` bounds). `sigma2` is
 *    linear in the objective, so it is eliminated (least squares in `sigma2`, clamped at zero); one
 *    single-invocation kernel scans the stationary points of that objective on a fixed grid, refines
 *    every bracketed minimum by bisection and takes the lowest objective (ties go to the lowest
 *    `lambda`). Deviation: spreg runs a local L-BFGS-B from `lambda = 0, sigma2 = 1`; the two agree whenever
 *    the objective has a single basin, and this solve returns the global minimum otherwise.
 * 3. The spatially filtered regression `y - lambda W y` on `X - lambda W X` is fit by OLS, where
 *    the intercept column is filtered too (`1 - lambda * sum_j w_ij`, as spreg `get_spFilter`
 *    does for the constant). Columns and the response are centered by their means before
 *    filtering (the model is shift-equivariant, so the centered fit is exactly equivalent), which
 *    keeps float32 normal equations well conditioned; the intercept and its variance are mapped
 *    back to the uncentered model.
 * 4. Reported: coefficients for the original `[1, X]`, residuals `u = y - X b`, `sigma2 = e'e / n`
 *    with `e = u - lambda W u` (spreg `sig2` = `sig2n`) and covariance `sigma2 (Xs'Xs)^-1` of the
 *    betas. z statistics and p-values are standard normal, as in spreg. `lambda` is reported
 *    without a standard error.
 *
 * All accumulations use the fixed-order tile and Kahan-merge scheme of `GPUOrdinaryLeastSquares`,
 * so results are deterministic; nothing is read back. Not included: the `GM_Error_Het` and
 * `GM_Error_Hom` variants (their `lambda` variance and the Kelejian-Prucha weighted estimator).
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
   * Returns the first-stage `GPUOrdinaryLeastSquares` nodes followed by twelve nodes: the mean
   * tiles and merge, the residual lag, the moment tiles and merge, the `lambda` solve, the
   * filtered-moment tiles and merge, the filtered solve, the residual tiles and merge, and the
   * finish that writes `table`, `summary` and `status`.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rowCount, tileRowCount, tileCount} = this;
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
    const workspaceLength = meanCount + 3 + q;
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
const W_BETA: u32 = ${k + 4}u;`;
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
    const solveDeclarations = `${constants}
${getSolveWGSL(q)}`;
    return [
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
      kernel(
        'residual-lag',
        [
          ...weightBindings,
          f32('residual', olsResiduals, 'read'),
          f32('lag', lagResiduals, 'read_write')
        ],
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
      ),
      kernel(
        'moments-tiles',
        [
          ...weightBindings,
          f32('residual', olsResiduals, 'read'),
          f32('lag', lagResiduals, 'read'),
          f32('partial', partialMoments, 'read_write')
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
      ),
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
        `${constants}
${getLambdaWGSL()}`,
        'solveLambda();'
      ),
      kernel(
        'filtered-moments-tiles',
        [...rowBindings, f32('partial', partialFiltered, 'read_write')],
        tileCount,
        filteredDeclarations,
        `let base = partialOffset + index * ${filteredMomentCount}u;
  for (var slot = 0u; slot < ${filteredMomentCount}u; slot++) {
    partial[base + slot] = 0.0;
  }
  let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  for (var row = firstRow; row < lastRow; row++) {
    var f: array<f32, ${q}>;
    var ys = 0.0;
    getFilteredRow(row, &f, &ys);
    for (var a = 0u; a < Q; a++) {
      for (var b = 0u; b < Q; b++) {
        partial[base + a * Q + b] += f[a] * f[b];
      }
      partial[base + Q * Q + a] += f[a] * ys;
    }
    partial[base + Q * Q + Q] += ys * ys;
  }`,
        TILE_WORKGROUP_SIZE
      ),
      kernel(
        'filtered-moments-merge',
        [
          f32('partial', partialFiltered, 'read'),
          f32('filteredMoments', filteredMoments, 'read_write')
        ],
        filteredMomentCount,
        constants,
        `${mergeLoop(filteredMomentCount)}
  filteredMoments[filteredMomentsOffset + index] = sum;`
      ),
      kernel(
        'filtered-solve',
        [
          f32('filteredMoments', filteredMoments, 'read'),
          f32('workspace', workspace, 'read_write')
        ],
        1,
        solveDeclarations,
        `var beta: array<f32, ${q}>;
  var inverseDiagonal: array<f32, ${q}>;
  var interceptQuadratic = 0.0;
  let ok = solveFiltered(&beta, &inverseDiagonal, &interceptQuadratic);
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
        `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${RESIDUAL_SUM_COUNT}>;
  for (var row = firstRow; row < lastRow; row++) {
    var f: array<f32, ${q}>;
    var ys = 0.0;
    getFilteredRow(row, &f, &ys);
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
      ),
      kernel(
        'finish',
        [
          f32('filteredMoments', filteredMoments, 'read'),
          f32('workspace', workspace, 'read'),
          f32('residualSums', residualSums, 'read'),
          {name: 'olsStatus', view: olsStatus, type: 'u32', access: 'read'},
          f32('tableOut', output.table, 'read_write'),
          f32('summaryOut', output.summary, 'read_write'),
          {name: 'statusOut', view: output.status, type: 'u32', access: 'read_write'}
        ],
        1,
        `${solveDeclarations}
${getFinishWGSL(k)}`,
        'finish();'
      )
    ];
  }
}

/**
 * `getFilteredRow`: the filtered, centered regressors `f = [1 - lambda s_i, xs]` and response
 * `ys` of one row. Kernels that include it bind `offsets`, `neighbors`, `weights`, `design`,
 * `response` and `workspace` (means and `lambda`).
 */
function getFilteredRowWGSL(): string {
  return /* wgsl */ `
fn getFilteredRow(row: u32, f: ptr<function, array<f32, Q>>, ys: ptr<function, f32>) {
  let lambda = workspace[workspaceOffset + W_LAMBDA];
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
 * `solveLambda`: the moment system of `GM_Error` with `sigma2` projected out, scanned for minima
 * on `[-1, 1]`. Writes `lambda` and the scaled objective, NaN for non-finite or empty moments.
 */
function getLambdaWGSL(): string {
  return /* wgsl */ `
fn getNaN() -> f32 {
  var bits = 0x7fc00000u;
  return bitcast<f32>(bits);
}

fn isFiniteValue(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

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
  if (!finite) {
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
  let bound = ${LAMBDA_BOUND};
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
  workspace[workspaceOffset + W_LAMBDA] = bestLambda;
  workspace[workspaceOffset + W_OBJECTIVE] = bestObjective;
}
`;
}

/**
 * `solveFiltered`: scaled Cholesky solve of the filtered normal equations `Xs'Xs b = Xs'ys` (the
 * centered-model intercept first), the diagonal of the inverse and `g' (Xs'Xs)^-1 g` with
 * `g = (1, -means)`, the variance factor of the uncentered intercept. Kernels that include it
 * bind `filteredMoments` and `workspace`.
 */
function getSolveWGSL(q: number): string {
  return /* wgsl */ `
fn getNaN() -> f32 {
  var bits = 0x7fc00000u;
  return bitcast<f32>(bits);
}

fn isFiniteValue(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

${getCholeskyWGSL(q)}

fn solveFiltered(
  beta: ptr<function, array<f32, ${q}>>,
  inverseDiagonal: ptr<function, array<f32, ${q}>>,
  interceptQuadratic: ptr<function, f32>
) -> bool {
  var scale: array<f32, ${q}>;
  for (var a = 0u; a < Q; a++) {
    let diagonal = filteredMoments[filteredMomentsOffset + a * Q + a];
    if (!(diagonal > 0.0) || !isFiniteValue(diagonal)) {
      return false;
    }
    scale[a] = 1.0 / sqrt(diagonal);
  }
  var factor: array<f32, ${q * q}>;
  var rhs: array<f32, ${q}>;
  for (var a = 0u; a < Q; a++) {
    for (var b = 0u; b < Q; b++) {
      factor[a * Q + b] = filteredMoments[filteredMomentsOffset + a * Q + b] * scale[a] * scale[b];
    }
    rhs[a] = filteredMoments[filteredMomentsOffset + Q * Q + a] * scale[a];
  }
  if (!choleskyFactor_${q}(&factor)) {
    return false;
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

/** The finish kernel: coefficient table, summary and status. */
function getFinishWGSL(k: number): string {
  return /* wgsl */ `
fn erfcValue(x: f32) -> f32 {
  let z = abs(x);
  let t = 1.0 / (1.0 + 0.5 * z);
  let r = t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
    t * (-0.82215223 + t * 0.17087277)))))))));
  return select(2.0 - r, r, x >= 0.0);
}

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
  let lambda = workspace[workspaceOffset + W_LAMBDA];
  if (!isFiniteValue(lambda)) {
    fail(3u);
    return;
  }
  var beta: array<f32, ${k + 1}>;
  var inverseDiagonal: array<f32, ${k + 1}>;
  var interceptQuadratic = 0.0;
  if (!solveFiltered(&beta, &inverseDiagonal, &interceptQuadratic)) {
    fail(1u);
    return;
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
  let sigmaSquared = filteredSquares / n;
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
  tableOut[tableOutOffset + Q * 4u] = lambda;
  tableOut[tableOutOffset + Q * 4u + 1u] = nan;
  tableOut[tableOutOffset + Q * 4u + 2u] = nan;
  tableOut[tableOutOffset + Q * 4u + 3u] = nan;
  // Squared correlation of y and y - u: the response is centered, so Syy = responseSquares.
  let predictedVariance = responseSquares - 2.0 * responseResidual + residualSquares - residualTotal * residualTotal / n;
  let covariance = responseSquares - responseResidual;
  summaryOut[summaryOutOffset] = n;
  summaryOut[summaryOutOffset + 1u] = lambda;
  summaryOut[summaryOutOffset + 2u] = sigmaSquared;
  summaryOut[summaryOutOffset + 3u] = covariance * covariance / (responseSquares * predictedVariance);
  summaryOut[summaryOutOffset + 4u] = residualSquares;
  summaryOut[summaryOutOffset + 5u] = workspace[workspaceOffset + W_OBJECTIVE];
  statusOut[statusOutOffset] = 0u;
}
`;
}
