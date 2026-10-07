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
import {GPUSpatialWeightsTranspose} from '../spatial-weights/gpu-spatial-weights-transpose';
import {
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_MAXIMUM_PREDICTOR_COUNT,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE
} from './spatial-two-stage-least-squares-parameters';
import {getCholeskyWGSL} from './spatial-regression-solve';

const OPERATION = 'GPUSpatialTwoStageLeastSquares';
const TILE_WORKGROUP_SIZE = 64;
const MAXIMUM_DEFAULT_TILE_COUNT = 4096;
const MINIMUM_DEFAULT_TILE_ROWS = 64;
/** Residual sums per tile: `u'u`, `sum u`, `sum (y - ybar) u`, `sum (y - ybar)^2`. */
const RESIDUAL_SUM_COUNT = 4;
/** Anselin-Kelejian sums besides the `k + 1` products `u'W z`: `u'Wu`, `u'W1`, `S0`, `tr W'W`, `tr WW`. */
const ANSELIN_KELEJIAN_FIXED_SUM_COUNT = 5;

/** Caller-owned outputs of {@link GPUSpatialTwoStageLeastSquares}. */
export type GPUSpatialTwoStageLeastSquaresOutput = {
  /**
   * `(predictorCount + 2) * 4` float32 values: one row per coefficient in spreg order (intercept,
   * one per predictor, then the spatial lag `rho` of the response last), each row holding the
   * coefficient, standard error, z statistic and two-sided normal p-value.
   */
  table: GraphDataView<'float32'>;
  /** At least 7 float32 values; see the `GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_*` slots. */
  summary: GraphDataView<'float32'>;
  /** One uint32; see the `GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_STATUS_*` constants. */
  status: GraphDataView<'uint32'>;
  /** Optional per-row structural residuals `u = y - rho W y - X b`; NaN for a failed fit. */
  residuals?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUSpatialTwoStageLeastSquares}.
 *
 * Per-frame (no recompile): the contents of every input buffer. Topology (needs a new graph):
 * `predictorCount`, the row count, the weights views, whether `residuals` is present and
 * `tileRowCount`.
 */
export type GPUSpatialTwoStageLeastSquaresProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'spatial-two-stage-least-squares'`. */
  id?: string;
  /**
   * Square weights `W`; rows index the regression rows. Any sparsity pattern is accepted,
   * including directed kNN weights; the Anselin-Kelejian test forms `W'` with
   * `GPUSpatialWeightsTranspose`.
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
  /**
   * Instrument order (spreg `w_lags`), compile-time: 1 (the default) instruments `W y` with
   * `[1, X, W X]`, 2 with `[1, X, W X, W^2 X]`, where `W^2 X = W (W X)` is evaluated per row from
   * the two-hop neighborhood, so no extra storage buffers are bound. The instrument count is
   * `(order + 1) * predictorCount` columns.
   */
  instrumentOrder?: 1 | 2;
  /** Caller-owned outputs. */
  output: GPUSpatialTwoStageLeastSquaresOutput;
};

/**
 * Spatial two-stage least squares for the lag model `y = rho W y + X b + e` (spreg `GM_Lag` with
 * `w_lags = 1`, no robust or heteroskedastic correction). The endogenous regressor `W y` is
 * instrumented with `[1, X, W X]` (the lag of every predictor, never of the intercept).
 *
 * With `Z = [1, X, W y]`, `H = [1, X, W X]` and `P = H (H'H)^-1 H'` the estimate is
 * `delta = (Z'PZ)^-1 Z'P y`, the structural residuals are `u = y - Z delta`, `sigma2 = u'u / n`
 * (spreg `sig2n`, its `GM_Lag` default), and the covariance is `sigma2 (Z'PZ)^-1`. z statistics and
 * p-values are standard normal (spreg reports z for `GM_Lag`).
 *
 * Because `1` is in both `Z` and `H`, the problem is solved on columns centered by their means
 * (Frisch-Waugh), which removes the cancellation of raw moments; the intercept is recovered as
 * `ybar - mean(Z)' delta` with variance `sigma2 / n + mean(Z)' Var(delta) mean(Z)`.
 * The accumulation is the same fixed-order tile and Kahan-merge scheme as
 * `GPUOrdinaryLeastSquares`: tile moments `H'H`, `H'Z`, `H'y` and `diag(Z'Z)`, a single-invocation
 * solve (Cholesky of the scaled `H'H`, then of the scaled `Z'PZ`), a per-row residual pass and a
 * finish kernel for the table. Nothing is read back.
 *
 * The Anselin-Kelejian test of the structural residuals (spreg `AKtest(case='gen')`) is written to
 * `summary`: `AK = n I^2 / phi2` with `I = (n / S0) u'Wu / u'u`,
 * `phi2 = (T + 4 a / sigma2) / ((S0 / n)^2 n)` and `a = (u'WZ) (Z'PZ)^-1 (u'WZ)'`. `W'u` is summed
 * over the rows of the transposed CSR, so the sparsity pattern of `weights` may be asymmetric
 * (`T = sum w_ij^2 + sum w_ij w_ji` only involves pairs present in both directions).
 *
 * `instrumentOrder: 2` adds `W^2 X` to the instruments (`w_lags = 2`). The robust and heteroskedastic variants, and spatial
 * error models (`GM_Error`) are not included here.
 */
export class GPUSpatialTwoStageLeastSquares implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialTwoStageLeastSquaresProps;
  /** Number of rows. */
  readonly rowCount: number;
  /** Rows reduced by one accumulation invocation. */
  readonly tileRowCount: number;
  /** Number of accumulation tiles. */
  readonly tileCount: number;
  /** Instrument order, 1 or 2. */
  readonly instrumentOrder: 1 | 2;

  constructor(props: GPUSpatialTwoStageLeastSquaresProps) {
    const id = props.id ?? 'spatial-two-stage-least-squares';
    this.id = id;
    this.props = props;
    const {predictorCount, output} = props;
    if (
      !Number.isInteger(predictorCount) ||
      predictorCount < 1 ||
      predictorCount > GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_MAXIMUM_PREDICTOR_COUNT
    ) {
      throw new Error(
        `${id} predictorCount must be an integer in [1, ${GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_MAXIMUM_PREDICTOR_COUNT}]`
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
    if (
      output.table.length <
      (predictorCount + 2) * GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE
    ) {
      throw new Error(`${id} output.table must hold (predictorCount + 2) * 4 float32 values`);
    }
    validatePackedView(output.summary, ['float32'], `${id} output.summary`);
    if (output.summary.length < GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH) {
      throw new Error(
        `${id} output.summary must hold ${GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH} float32 values`
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
    const instrumentOrder = props.instrumentOrder ?? 1;
    if (instrumentOrder !== 1 && instrumentOrder !== 2) {
      throw new Error(`${id} instrumentOrder must be 1 or 2`);
    }
    this.instrumentOrder = instrumentOrder;
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
   * Returns the nodes in order: mean tiles and merge, moment tiles and merge, the solve, the
   * residual tiles and merge, the transpose of `W`, the `W'u` pass, the two Anselin-Kelejian tile
   * kernels and merge, and the finish that writes `table`, `summary` and `status`.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rowCount, tileRowCount, tileCount, instrumentOrder} = this;
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
    const m = (instrumentOrder + 1) * k;
    const q = k + 1;
    const momentCount = m * m + m * q + m + q;
    // Means: x (k), W x (k), W y, y, then W^2 x (k) for instrument order 2.
    const meanCount = (instrumentOrder + 1) * k + 2;
    const workspaceLength = meanCount + q;
    const transient = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'float32', length);
    const partialMeans = transient('partial-means', tileCount * meanCount);
    const partialMoments = transient('partial-moments', tileCount * momentCount);
    const moments = transient('moments', momentCount);
    const workspace = transient('workspace', workspaceLength);
    const partialResiduals = transient('partial-residuals', tileCount * RESIDUAL_SUM_COUNT);
    const residualSums = transient('residual-sums', RESIDUAL_SUM_COUNT);
    const akCount = ANSELIN_KELEJIAN_FIXED_SUM_COUNT + q;
    const partialAk = transient('partial-anselin-kelejian', tileCount * akCount);
    const akSums = transient('anselin-kelejian-sums', akCount);
    const residualView = output.residuals ?? transient('residuals', rowCount);
    const transposed: GPUSpatialWeights = {
      offsets: createTransientView(graph, `${id}-transpose-offsets`, 'uint32', rowCount + 1),
      neighbors: createTransientView(
        graph,
        `${id}-transpose-neighbors`,
        'uint32',
        weights.neighbors.length
      ),
      weights: transient('transpose-weights', weights.neighbors.length)
    };
    const transposedLagResidual = transient('transposed-lag-residual', rowCount);
    const constants = `const ROWS: u32 = ${rowCount}u;
const K: u32 = ${k}u;
const ORDER: u32 = ${instrumentOrder}u;
const M: u32 = ${m}u;
const Q: u32 = ${q}u;
const TILE_ROWS: u32 = ${tileRowCount}u;
const TILE_COUNT: u32 = ${tileCount}u;
const MEAN_COUNT: u32 = ${meanCount}u;
const MOMENTS: u32 = ${momentCount}u;
const W_DELTA: u32 = ${meanCount}u;`;
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
    const rowBindings: WGSLKernelBinding[] = [
      {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
      f32('weights', weights.weights, 'read'),
      f32('design', props.predictors, 'read'),
      f32('response', props.response, 'read')
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
    const lagDeclarations = `${constants}
${getLagWGSL()}`;
    const rowDeclarations = `${lagDeclarations}
${getCenteredRowWGSL()}`;
    const transposeNodes = new GPUSpatialWeightsTranspose({
      id: `${id}-transpose`,
      weights,
      output: transposed
    }).getCommandNodes(graph);
    return [
      kernel(
        'means-tiles',
        [...rowBindings, f32('partial', partialMeans, 'read_write')],
        tileCount,
        lagDeclarations,
        `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${meanCount}>;
  for (var row = firstRow; row < lastRow; row++) {
    var lagged: array<f32, ${k}>;
    var twiceLagged: array<f32, ${k}>;
    let lagResponse = getLags(row, &lagged, &twiceLagged, ORDER == 2u);
    for (var column = 0u; column < K; column++) {
      sums[column] = sums[column] + design[designOffset + row * K + column];
      sums[K + column] = sums[K + column] + lagged[column];
      if (ORDER == 2u) {
        sums[2u * K + 2u + column] = sums[2u * K + 2u + column] + twiceLagged[column];
      }
    }
    sums[2u * K] = sums[2u * K] + lagResponse;
    sums[2u * K + 1u] = sums[2u * K + 1u] + response[responseOffset + row];
  }
  for (var slot = 0u; slot < MEAN_COUNT; slot++) {
    partial[partialOffset + index * MEAN_COUNT + slot] = sums[slot];
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
        'moments-tiles',
        [
          ...rowBindings,
          f32('workspace', workspace, 'read'),
          f32('partial', partialMoments, 'read_write')
        ],
        tileCount,
        rowDeclarations,
        `let base = partialOffset + index * MOMENTS;
  for (var slot = 0u; slot < MOMENTS; slot++) {
    partial[base + slot] = 0.0;
  }
  let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  for (var row = firstRow; row < lastRow; row++) {
    var h: array<f32, ${m}>;
    var z: array<f32, ${q}>;
    var centeredResponse = 0.0;
    getCenteredRow(row, &h, &z, &centeredResponse, true);
    for (var a = 0u; a < M; a++) {
      for (var b = 0u; b < M; b++) {
        partial[base + a * M + b] += h[a] * h[b];
      }
      for (var b = 0u; b < Q; b++) {
        partial[base + M * M + a * Q + b] += h[a] * z[b];
      }
      partial[base + M * M + M * Q + a] += h[a] * centeredResponse;
    }
    for (var b = 0u; b < Q; b++) {
      partial[base + M * M + M * Q + M + b] += z[b] * z[b];
    }
  }`,
        TILE_WORKGROUP_SIZE
      ),
      kernel(
        'moments-merge',
        [f32('partial', partialMoments, 'read'), f32('moments', moments, 'read_write')],
        momentCount,
        constants,
        `${mergeLoop(momentCount)}
  moments[momentsOffset + index] = sum;`
      ),
      kernel(
        'solve',
        [f32('moments', moments, 'read'), f32('workspace', workspace, 'read_write')],
        1,
        `${constants}
${getSolveWGSL(m, q)}`,
        `var delta: array<f32, ${q}>;
  var inverseDiagonal: array<f32, ${q}>;
  var meanQuadratic = 0.0;
  var unusedVector: array<f32, ${q}>;
  var unusedQuadratic = 0.0;
  let ok = solveSystem(&delta, &inverseDiagonal, &meanQuadratic, &unusedVector, &unusedQuadratic);
  for (var column = 0u; column < Q; column++) {
    workspace[workspaceOffset + W_DELTA + column] = select(getNaN(), delta[column], ok);
  }`
      ),
      kernel(
        'residuals-tiles',
        [
          ...rowBindings,
          f32('workspace', workspace, 'read'),
          f32('residualsOut', residualView, 'read_write'),
          f32('partial', partialResiduals, 'read_write')
        ],
        tileCount,
        rowDeclarations,
        `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${RESIDUAL_SUM_COUNT}>;
  for (var row = firstRow; row < lastRow; row++) {
    var h: array<f32, ${m}>;
    var z: array<f32, ${q}>;
    var centeredResponse = 0.0;
    getCenteredRow(row, &h, &z, &centeredResponse, false);
    var residual = centeredResponse;
    for (var b = 0u; b < Q; b++) {
      residual = residual - z[b] * workspace[workspaceOffset + W_DELTA + b];
    }
    residualsOut[residualsOutOffset + row] = residual;
    sums[0] += residual * residual;
    sums[1] += residual;
    sums[2] += centeredResponse * residual;
    sums[3] += centeredResponse * centeredResponse;
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
      ...transposeNodes,
      kernel(
        'anselin-kelejian-tiles',
        [
          ...rowBindings,
          f32('workspace', workspace, 'read'),
          f32('residuals', residualView, 'read'),
          f32('partial', partialAk, 'read_write')
        ],
        tileCount,
        lagDeclarations,
        `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${akCount}>;
  for (var row = firstRow; row < lastRow; row++) {
    let residual = residuals[residualsOffset + row];
    var rowSum = 0.0;
    var lagResidual = 0.0;
    var squareSum = 0.0;
    var crossSum = 0.0;
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor >= ROWS) {
        continue;
      }
      let weight = weights[weightsOffset + slot];
      // Reverse weight w_ji by binary search of this row in the neighbor's sorted row.
      var reverse = 0.0;
      var low = offsets[offsetsOffset + neighbor];
      var high = offsets[offsetsOffset + neighbor + 1u];
      while (low < high) {
        let middle = (low + high) / 2u;
        let candidate = neighbors[neighborsOffset + middle];
        if (candidate == row) {
          reverse = weights[weightsOffset + middle];
          break;
        }
        if (candidate < row) {
          low = middle + 1u;
        } else {
          high = middle;
        }
      }
      let neighborResidual = residuals[residualsOffset + neighbor];
      rowSum += weight;
      lagResidual += weight * neighborResidual;
      squareSum += weight * weight;
      crossSum += weight * reverse;
    }
    sums[0] += residual * lagResidual;
    sums[1] += residual * rowSum;
    sums[2] += rowSum;
    sums[3] += squareSum;
    sums[4] += crossSum;
  }
  // The u'W z products (slots after the fixed ones) come from the next kernel.
  for (var slot = 0u; slot < ${ANSELIN_KELEJIAN_FIXED_SUM_COUNT}u; slot++) {
    partial[partialOffset + index * ${akCount}u + slot] = sums[slot];
  }`,
        TILE_WORKGROUP_SIZE
      ),
      // (W'u)_i over the transposed CSR, so one-way links are included.
      kernel(
        'transposed-lag-residual',
        [
          {name: 'offsets', view: transposed.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: transposed.neighbors, type: 'u32', access: 'read'},
          f32('weights', transposed.weights, 'read'),
          f32('residuals', residualView, 'read'),
          f32('lagOut', transposedLagResidual, 'read_write')
        ],
        rowCount,
        constants,
        `var sum = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    sum += weights[weightsOffset + slot] * residuals[residualsOffset + neighbors[neighborsOffset + slot]];
  }
  lagOut[lagOutOffset + index] = sum;`
      ),
      kernel(
        'anselin-kelejian-products',
        [
          ...rowBindings,
          f32('workspace', workspace, 'read'),
          f32('transposedLagResidual', transposedLagResidual, 'read'),
          f32('partial', partialAk, 'read_write')
        ],
        tileCount,
        lagDeclarations,
        `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${q}>;
  for (var row = firstRow; row < lastRow; row++) {
    var lagged: array<f32, ${k}>;
    var twiceLagged: array<f32, ${k}>;
    let lagResponse = getLags(row, &lagged, &twiceLagged, false);
    let product = transposedLagResidual[transposedLagResidualOffset + row];
    for (var column = 0u; column < K; column++) {
      let centered = design[designOffset + row * K + column] - workspace[workspaceOffset + column];
      sums[column] += product * centered;
    }
    sums[K] += product * (lagResponse - workspace[workspaceOffset + 2u * K]);
  }
  for (var column = 0u; column < Q; column++) {
    partial[partialOffset + index * ${akCount}u + ${ANSELIN_KELEJIAN_FIXED_SUM_COUNT}u + column] = sums[column];
  }`,
        TILE_WORKGROUP_SIZE
      ),
      kernel(
        'anselin-kelejian-merge',
        [f32('partial', partialAk, 'read'), f32('akSums', akSums, 'read_write')],
        akCount,
        constants,
        `${mergeLoop(akCount)}
  akSums[akSumsOffset + index] = sum;`
      ),
      kernel(
        'finish',
        [
          f32('moments', moments, 'read'),
          f32('workspace', workspace, 'read'),
          f32('residualSums', residualSums, 'read'),
          f32('akSums', akSums, 'read'),
          f32('tableOut', output.table, 'read_write'),
          f32('summaryOut', output.summary, 'read_write'),
          {name: 'statusOut', view: output.status, type: 'u32', access: 'read_write'}
        ],
        1,
        `${constants}
${getSolveWGSL(m, q)}
${getFinishWGSL(k)}`,
        'finish();'
      )
    ];
  }
}

/**
 * `getLags`: `W x` for every predictor and `W y` in one slot-order loop over the row. With
 * `needSecond`, also `W^2 x = W (W x)`, summed over the two-hop neighborhood in slot order.
 */
function getLagWGSL(): string {
  return /* wgsl */ `
fn getLags(
  row: u32,
  lagged: ptr<function, array<f32, K>>,
  twiceLagged: ptr<function, array<f32, K>>,
  needSecond: bool
) -> f32 {
  var lagResponse = 0.0;
  for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= ROWS) {
      continue;
    }
    let weight = weights[weightsOffset + slot];
    lagResponse += weight * response[responseOffset + neighbor];
    for (var column = 0u; column < K; column++) {
      (*lagged)[column] += weight * design[designOffset + neighbor * K + column];
    }
    if (needSecond) {
      var inner: array<f32, K>;
      for (var second = offsets[offsetsOffset + neighbor]; second < offsets[offsetsOffset + neighbor + 1u]; second++) {
        let farther = neighbors[neighborsOffset + second];
        if (farther >= ROWS) {
          continue;
        }
        let farWeight = weights[weightsOffset + second];
        for (var column = 0u; column < K; column++) {
          inner[column] += farWeight * design[designOffset + farther * K + column];
        }
      }
      for (var column = 0u; column < K; column++) {
        (*twiceLagged)[column] += weight * inner[column];
      }
    }
  }
  return lagResponse;
}
`;
}

/**
 * `getCenteredRow`: centered instruments `h`, regressors `z` and response of one row. Kernels
 * that include it bind `workspace` (the means).
 */
function getCenteredRowWGSL(): string {
  return /* wgsl */ `
// h = [x - mean(x), W x - mean(W x)], z = [x - mean(x), W y - mean(W y)].
fn getCenteredRow(
  row: u32,
  h: ptr<function, array<f32, M>>,
  z: ptr<function, array<f32, Q>>,
  centeredResponse: ptr<function, f32>,
  needSecond: bool
) {
  var lagged: array<f32, K>;
  var twiceLagged: array<f32, K>;
  let lagResponse = getLags(row, &lagged, &twiceLagged, needSecond);
  for (var column = 0u; column < K; column++) {
    let centered = design[designOffset + row * K + column] - workspace[workspaceOffset + column];
    (*h)[column] = centered;
    (*z)[column] = centered;
    (*h)[K + column] = lagged[column] - workspace[workspaceOffset + K + column];
    if (ORDER == 2u && needSecond) {
      (*h)[2u * K + column] = twiceLagged[column] - workspace[workspaceOffset + 2u * K + 2u + column];
    }
  }
  (*z)[K] = lagResponse - workspace[workspaceOffset + 2u * K];
  *centeredResponse = response[responseOffset + row] - workspace[workspaceOffset + 2u * K + 1u];
}
`;
}

/**
 * `solveSystem` for the solve and finish kernels: scaled Cholesky of `H'H`, `P`-projected normal
 * matrix `Z'PZ = (H'Z)' (H'H)^-1 (H'Z)`, its solve for `delta`, the diagonal of its inverse and the
 * quadratic form `m' (Z'PZ)^-1 m` of the regressor means `m` used by the intercept variance.
 */
function getSolveWGSL(m: number, q: number): string {
  return /* wgsl */ `
fn getNaN() -> f32 {
  var bits = 0x7fc00000u;
  return bitcast<f32>(bits);
}

fn isFiniteValue(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

fn regressorMean(column: u32) -> f32 {
  return workspace[workspaceOffset + select(column, 2u * K, column == K)];
}

${getCholeskyWGSL(m)}
${q === m ? '' : getCholeskyWGSL(q)}

fn solveSystem(
  delta: ptr<function, array<f32, ${q}>>,
  inverseDiagonal: ptr<function, array<f32, ${q}>>,
  meanQuadratic: ptr<function, f32>,
  second: ptr<function, array<f32, ${q}>>,
  secondQuadratic: ptr<function, f32>
) -> bool {
  let instrumentScalars = M * M + M * Q;
  var instrumentScale: array<f32, ${m}>;
  var regressorScale: array<f32, ${q}>;
  for (var a = 0u; a < M; a++) {
    let diagonal = moments[momentsOffset + a * M + a];
    if (!(diagonal > 0.0) || !isFiniteValue(diagonal)) {
      return false;
    }
    instrumentScale[a] = 1.0 / sqrt(diagonal);
  }
  for (var b = 0u; b < Q; b++) {
    let diagonal = moments[momentsOffset + instrumentScalars + M + b];
    if (!(diagonal > 0.0) || !isFiniteValue(diagonal)) {
      return false;
    }
    regressorScale[b] = 1.0 / sqrt(diagonal);
  }
  var factor: array<f32, ${m * m}>;
  for (var a = 0u; a < M; a++) {
    for (var c = 0u; c < M; c++) {
      factor[a * M + c] = moments[momentsOffset + a * M + c] * instrumentScale[a] * instrumentScale[c];
    }
  }
  if (!choleskyFactor_${m}(&factor)) {
    return false;
  }
  // x1 = (H'H)^-1 H'Z column by column, x2 = (H'H)^-1 H'y (scaled instrument space).
  var projectedInstruments: array<f32, ${m * q}>;
  for (var b = 0u; b < Q; b++) {
    var rhs: array<f32, ${m}>;
    for (var a = 0u; a < M; a++) {
      rhs[a] = moments[momentsOffset + M * M + a * Q + b] * instrumentScale[a] * regressorScale[b];
    }
    choleskySolve_${m}(&factor, &rhs);
    for (var a = 0u; a < M; a++) {
      projectedInstruments[a * Q + b] = rhs[a];
    }
  }
  var projectedResponse: array<f32, ${m}>;
  for (var a = 0u; a < M; a++) {
    projectedResponse[a] = moments[momentsOffset + M * M + M * Q + a] * instrumentScale[a];
  }
  choleskySolve_${m}(&factor, &projectedResponse);
  var normal: array<f32, ${q * q}>;
  var rhs: array<f32, ${q}>;
  for (var b = 0u; b < Q; b++) {
    var rhsValue = 0.0;
    for (var a = 0u; a < M; a++) {
      rhsValue += moments[momentsOffset + M * M + a * Q + b] * instrumentScale[a] * regressorScale[b] * projectedResponse[a];
    }
    rhs[b] = rhsValue;
    for (var c = 0u; c < Q; c++) {
      var sum = 0.0;
      for (var a = 0u; a < M; a++) {
        sum += moments[momentsOffset + M * M + a * Q + b] * instrumentScale[a] * regressorScale[b] * projectedInstruments[a * Q + c];
      }
      normal[b * Q + c] = sum;
    }
  }
  if (!choleskyFactor_${q}(&normal)) {
    return false;
  }
  choleskySolve_${q}(&normal, &rhs);
  for (var b = 0u; b < Q; b++) {
    (*delta)[b] = rhs[b] * regressorScale[b];
    (*inverseDiagonal)[b] = choleskyInverseDiagonal_${q}(&normal, b) * regressorScale[b] * regressorScale[b];
  }
  // Quadratic form of the regressor means, in the scaled space.
  var meansScaled: array<f32, ${q}>;
  for (var b = 0u; b < Q; b++) {
    meansScaled[b] = regressorMean(b) * regressorScale[b];
  }
  var solved = meansScaled;
  choleskySolve_${q}(&normal, &solved);
  var quadratic = 0.0;
  for (var b = 0u; b < Q; b++) {
    quadratic += meansScaled[b] * solved[b];
  }
  *meanQuadratic = quadratic;
  var secondScaled: array<f32, ${q}>;
  for (var b = 0u; b < Q; b++) {
    secondScaled[b] = (*second)[b] * regressorScale[b];
  }
  var secondSolved = secondScaled;
  choleskySolve_${q}(&normal, &secondSolved);
  var secondSum = 0.0;
  for (var b = 0u; b < Q; b++) {
    secondSum += secondScaled[b] * secondSolved[b];
  }
  *secondQuadratic = secondSum;
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
  for (var slot = 0u; slot < 7u; slot++) {
    summaryOut[summaryOutOffset + slot] = nan;
  }
  summaryOut[summaryOutOffset] = f32(ROWS);
  statusOut[statusOutOffset] = status;
}

fn finish() {
  let n = f32(ROWS);
  let degrees = n - f32(Q + 1u);
  if (!(degrees > 0.0)) {
    fail(2u);
    return;
  }
  let residualSquares = residualSums[residualSumsOffset];
  let residualTotal = residualSums[residualSumsOffset + 1u];
  let responseResidual = residualSums[residualSumsOffset + 2u];
  let responseSquares = residualSums[residualSumsOffset + 3u];
  if (!isFiniteValue(residualSquares) || !isFiniteValue(responseResidual) || !isFiniteValue(responseSquares)) {
    fail(select(3u, 1u, !isFiniteValue(workspace[workspaceOffset + W_DELTA])));
    return;
  }
  var delta: array<f32, ${k + 1}>;
  var inverseDiagonal: array<f32, ${k + 1}>;
  var meanQuadratic = 0.0;
  var productVector: array<f32, ${k + 1}>;
  for (var b = 0u; b < Q; b++) {
    productVector[b] = akSums[akSumsOffset + ${ANSELIN_KELEJIAN_FIXED_SUM_COUNT}u + b];
  }
  var productQuadratic = 0.0;
  if (!solveSystem(&delta, &inverseDiagonal, &meanQuadratic, &productVector, &productQuadratic)) {
    fail(1u);
    return;
  }
  let sigmaSquared = residualSquares / n;
  // Intercept: ybar - mean(Z) delta. Its variance is sigma2 / n plus the delta part.
  var intercept = workspace[workspaceOffset + 2u * K + 1u];
  for (var b = 0u; b < Q; b++) {
    intercept -= regressorMean(b) * delta[b];
  }
  let interceptVariance = sigmaSquared * (1.0 / n + meanQuadratic);
  for (var row = 0u; row < Q + 1u; row++) {
    // Rows 0 is the intercept; rows 1..K are the predictors; row K + 1 is rho (delta index K).
    let index = max(row, 1u) - 1u;
    let coefficient = select(delta[index], intercept, row == 0u);
    let variance = select(sigmaSquared * inverseDiagonal[index], interceptVariance, row == 0u);
    let standardError = sqrt(variance);
    let z = coefficient / standardError;
    tableOut[tableOutOffset + row * 4u] = coefficient;
    tableOut[tableOutOffset + row * 4u + 1u] = standardError;
    tableOut[tableOutOffset + row * 4u + 2u] = z;
    tableOut[tableOutOffset + row * 4u + 3u] = erfcValue(abs(z) * 0.70710678);
  }
  // Squared correlation of y and y - u: the response is centered, so Syy = responseSquares.
  let predictedVariance = responseSquares - 2.0 * responseResidual + residualSquares - residualTotal * residualTotal / n;
  let covariance = responseSquares - responseResidual;
  summaryOut[summaryOutOffset] = n;
  summaryOut[summaryOutOffset + 1u] = sigmaSquared;
  summaryOut[summaryOutOffset + 2u] = residualSquares;
  summaryOut[summaryOutOffset + 3u] = covariance * covariance / (responseSquares * predictedVariance);
  // Anselin-Kelejian (1997) as in spreg akTest: Moran's I of u, corrected by the estimation variance.
  let residualLagProduct = akSums[akSumsOffset];
  let rowSumProduct = akSums[akSumsOffset + 1u];
  let weightsSum = akSums[akSumsOffset + 2u];
  let traceT = akSums[akSumsOffset + 3u] + akSums[akSumsOffset + 4u];
  let moranI = (n * residualLagProduct) / (weightsSum * residualSquares);
  let estimationTerm = rowSumProduct * rowSumProduct / n + productQuadratic;
  let weightScale = weightsSum / n;
  let phiSquared = (traceT + 4.0 * estimationTerm / sigmaSquared) / (weightScale * weightScale * n);
  let anselinKelejian = n * moranI * moranI / phiSquared;
  summaryOut[summaryOutOffset + 4u] = moranI;
  summaryOut[summaryOutOffset + 5u] = anselinKelejian;
  summaryOut[summaryOutOffset + 6u] = erfcValue(sqrt(max(anselinKelejian, 0.0) * 0.5));
  statusOut[statusOutOffset] = 0u;
}
`;
}
