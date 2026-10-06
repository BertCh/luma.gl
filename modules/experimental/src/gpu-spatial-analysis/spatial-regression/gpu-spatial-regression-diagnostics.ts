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
import {type GPUSpatialWeights, validateGPUSpatialWeights} from '../spatial-weights/index';
import {GPUSpatialWeightsTranspose} from '../spatial-weights/gpu-spatial-weights-transpose';
import {
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH
} from './spatial-regression-diagnostics-parameters';
import {
  getCholeskyWGSL,
  SPATIAL_REGRESSION_MAXIMUM_PREDICTOR_COUNT
} from './spatial-regression-solve';

const OPERATION = 'GPUSpatialRegressionDiagnostics';
const TILE_WORKGROUP_SIZE = 64;
const MAXIMUM_DEFAULT_TILE_COUNT = 4096;
const MINIMUM_DEFAULT_TILE_ROWS = 64;
/** Scalar moments after the five `p x p` matrices: S0, S2, Sww, e'e, e'We, e'Wy, v'v. */
const SCALAR_MOMENT_COUNT = 7;

/** Caller-owned outputs of {@link GPUSpatialRegressionDiagnostics}. */
export type GPUSpatialRegressionDiagnosticsOutput = {
  /**
   * At least 18 float32 values: six tests of three values each (statistic, degrees of freedom,
   * p-value); see the `GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_*` row constants.
   */
  tests: GraphDataView<'float32'>;
  /**
   * At least 12 float32 values: the intermediate quantities (`n`, sigma squared, `T`, `J`, `S0`,
   * Moran's I with its expectation, variance, z and p-value, `e'We`, `e'Wy`); see the
   * `GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_*` slot constants.
   */
  summary: GraphDataView<'float32'>;
  /** One uint32; see the `GPU_SPATIAL_REGRESSION_DIAGNOSTICS_STATUS_*` constants. */
  status: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUSpatialRegressionDiagnostics}.
 *
 * Per-frame (no recompile): the contents of every input buffer. Topology (needs a new graph):
 * `predictorCount`, the row count, the weights views and `tileRowCount`.
 */
export type GPUSpatialRegressionDiagnosticsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'spatial-regression-diagnostics'`. */
  id?: string;
  /**
   * Square weights `W` over the regression rows. Any sparsity pattern is accepted: contiguity,
   * lattice and distance-band weights, but also directed ones such as plain kNN weights and
   * row-standardized weights. `W'` is formed on the GPU with `GPUSpatialWeightsTranspose`.
   */
  weights: GPUSpatialWeights;
  /** Row-major predictors, `predictors[row * predictorCount + column]`, without an intercept. */
  predictors: GraphDataView<'float32'>;
  /** Response `y`, one value per row. */
  response: GraphDataView<'float32'>;
  /**
   * Residuals `e = y - Xb` of an ordinary least squares fit with an intercept, one per row, for
   * example `GPUOrdinaryLeastSquares` `output.residuals`. They must be finite for every row; if
   * the fit excluded rows (mask) or failed, the diagnostics report `NON_FINITE` and NaN.
   */
  residuals: GraphDataView<'float32'>;
  /** Number of predictor columns, compile-time, 1 to 15. The intercept is implicit. */
  predictorCount: number;
  /**
   * Rows reduced by one invocation of the accumulation pass. Defaults to
   * `max(64, ceil(rows / 4096))`. The tile layout fixes the summation order, so results are
   * reproducible for a given value but differ in the last bits between values.
   */
  tileRowCount?: number;
  /** Caller-owned outputs. */
  output: GPUSpatialRegressionDiagnosticsOutput;
};

/**
 * Spatial diagnostics for an ordinary least squares fit (spreg `diagnostics_sp`: `LMtests` and
 * `MoranRes`): LM-lag, LM-error, robust LM-lag, robust LM-error, LM-SARMA, and Moran's I of the
 * residuals with its regression-residual mean, variance, normal z and p-value. This is the GeoDa
 * "fit OLS, then which spatial model?" panel.
 *
 * With `n` rows, `p = predictorCount + 1` regressors (including the intercept), `Z = [1, X]`,
 * `M = I - Z (Z'Z)^-1 Z'`, residuals `e`, `sigma2 = e'e / n`, `T = tr(W'W + WW)`, the fitted
 * values `Zb = y - e` and `J = (W Zb)' M (W Zb) / sigma2 + T`:
 *
 * - `LM-error = (e'We / sigma2)^2 / T`, `LM-lag = (e'Wy / sigma2)^2 / J` (df 1 each)
 * - robust LM-error `= (d_e - T d_l / J)^2 / (T - T^2 / J)` and robust LM-lag
 *   `= (d_l - d_e)^2 / (J - T)` with `d_e = e'We / sigma2`, `d_l = e'Wy / sigma2` (df 1 each)
 * - `LM-SARMA = LM-lag + robust LM-error` (df 2)
 * - Moran's I `= (n / S0) e'We / e'e` with `E[I] = -(n / S0) tr((Z'Z)^-1 Z'WZ) / (n - p)` and
 *   `Var[I] = (n / S0)^2 (tr(MWMW') + tr(MWMW) + tr(MW)^2) / ((n - p)(n - p + 2)) - E[I]^2`.
 *   The traces expand into `tr((Z'Z)^-1 Q)` of the `p x p` Gram matrices `Q` of `Z`, `WZ` and `W'Z`.
 *
 * Every sum is a fixed-order reduction with no atomics: a tile pass (one invocation per
 * `tileRowCount` rows, rows in order) accumulates the Gram matrices and scalar moments for its
 * tile in f32, a merge pass sums the tiles in order with Kahan compensation, and a single
 * invocation finishes the small arithmetic and p-values (chi-square via the closed forms for df 1
 * and 2, normal via a 1e-7 accurate `erfc`). Columns are centered before accumulation and the
 * Gram matrices are scaled to unit diagonal before the Cholesky solve. Nothing is read back.
 *
 * `W` is read inline rather than through `GPUSpatialLag`, so the `W`-products of `y`, `e`, the
 * fitted values and every column of `Z` cost one pass; the arithmetic is the same as
 * `GPUSpatialLag` (slot order, f32). The sparsity pattern may be asymmetric: `T = sum w_ij^2 +
 * sum w_ij w_ji` only involves pairs present in both directions, and `W'Z` (needed by the Moran
 * variance) is accumulated over the rows of the transposed CSR, so a link `i -> j` without
 * `j -> i` is counted exactly as in the dense definition.
 *
 * Not included: the Anselin-Kelejian test, which is defined on two-stage least squares residuals
 * and belongs with the spatial 2SLS contributor.
 */
export class GPUSpatialRegressionDiagnostics implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialRegressionDiagnosticsProps;
  /** Number of rows. */
  readonly rowCount: number;
  /** Rows reduced by one accumulation invocation. */
  readonly tileRowCount: number;
  /** Number of accumulation tiles. */
  readonly tileCount: number;

  constructor(props: GPUSpatialRegressionDiagnosticsProps) {
    const id = props.id ?? 'spatial-regression-diagnostics';
    this.id = id;
    this.props = props;
    const {predictorCount, output} = props;
    if (
      !Number.isInteger(predictorCount) ||
      predictorCount < 1 ||
      predictorCount > SPATIAL_REGRESSION_MAXIMUM_PREDICTOR_COUNT - 1
    ) {
      throw new Error(
        `${id} predictorCount must be an integer in [1, ${SPATIAL_REGRESSION_MAXIMUM_PREDICTOR_COUNT - 1}]`
      );
    }
    for (const [name, view] of [
      ['predictors', props.predictors],
      ['response', props.response],
      ['residuals', props.residuals]
    ] as const) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    const rows = validateGPUSpatialWeights(id, props.weights);
    validatePackedView(props.predictors, ['float32'], `${id} predictors`);
    validatePackedView(props.response, ['float32'], `${id} response`);
    validatePackedView(props.residuals, ['float32'], `${id} residuals`);
    if (props.response.length !== rows) {
      throw new Error(`${id} response length must equal the weights row count`);
    }
    if (props.residuals.length !== rows) {
      throw new Error(`${id} residuals length must equal the weights row count`);
    }
    if (props.predictors.length !== rows * predictorCount) {
      throw new Error(`${id} predictors length must equal the weights row count * predictorCount`);
    }
    validatePackedView(output.tests, ['float32'], `${id} output.tests`);
    if (output.tests.length < GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH) {
      throw new Error(
        `${id} output.tests must hold ${GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH} float32 values`
      );
    }
    validatePackedView(output.summary, ['float32'], `${id} output.summary`);
    if (output.summary.length < GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH) {
      throw new Error(
        `${id} output.summary must hold ${GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH} float32 values`
      );
    }
    validatePackedUint32View(output.status, `${id} output.status`);
    if (output.status.length < 1) {
      throw new Error(`${id} output.status must hold one uint32 row`);
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
      [output.tests, output.summary, output.status],
      [
        props.predictors,
        props.response,
        props.residuals,
        props.weights.offsets,
        props.weights.neighbors,
        props.weights.weights
      ]
    );
  }

  /**
   * Returns the nodes in order: column-mean tiles, mean merge, the row table of centered `Z`, `y`
   * and `e`, the transpose of `W`, the `W'Z` table, moment tiles, moment merge and the
   * single-invocation finish that writes `tests`, `summary` and `status`.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rowCount, tileRowCount, tileCount} = this;
    const {weights, output, predictorCount: k} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.predictors,
      props.response,
      props.residuals,
      weights.offsets,
      weights.neighbors,
      weights.weights,
      output.tests,
      output.summary,
      output.status
    ]);
    const p = k + 1;
    const momentCount = 5 * p * p + SCALAR_MOMENT_COUNT + p;
    const transient = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'float32', length);
    const partialMeans = transient('partial-means', tileCount * k);
    const means = transient('means', k);
    const partialMoments = transient('partial-moments', tileCount * momentCount);
    const moments = transient('moments', momentCount);
    const rowTable = transient('row-table', rowCount * (p + 2));
    const transposedLagTable = transient('transposed-lag-table', rowCount * p);
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
    const transposeNodes = new GPUSpatialWeightsTranspose({
      id: `${id}-transpose`,
      weights,
      output: transposed
    }).getCommandNodes(graph);
    const constants = `const ROWS: u32 = ${rowCount}u;
const K: u32 = ${k}u;
const TABLE_WIDTH: u32 = ${p + 2}u;
const P: u32 = ${p}u;
const TILE_ROWS: u32 = ${tileRowCount}u;
const TILE_COUNT: u32 = ${tileCount}u;`;
    const mergeLoop = (stride: number) => `var sum = 0.0;
  var compensation = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    let term = partial[partialOffset + tile * ${stride}u + index] - compensation;
    let next = sum + term;
    compensation = (next - sum) - term;
    sum = next;
  }`;
    const weightBindings: WGSLKernelBinding[] = [
      {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
      {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
      {name: 'weights', view: weights.weights, type: 'f32', access: 'read'}
    ];
    const rowTableBinding: WGSLKernelBinding = {
      name: 'table',
      view: rowTable,
      type: 'f32',
      access: 'read'
    };
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-means-tiles`,
        operation: OPERATION,
        variant: 'means-tiles',
        workgroupSize: TILE_WORKGROUP_SIZE,
        bindings: [
          {name: 'design', view: props.predictors, type: 'f32', access: 'read'},
          {name: 'partial', view: partialMeans, type: 'f32', access: 'read_write'}
        ],
        invocationCount: tileCount,
        declarations: constants,
        body: `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sums: array<f32, ${k}>;
  for (var row = firstRow; row < lastRow; row++) {
    for (var column = 0u; column < K; column++) {
      sums[column] = sums[column] + design[designOffset + row * K + column];
    }
  }
  for (var column = 0u; column < K; column++) {
    partial[partialOffset + index * K + column] = sums[column];
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-means-merge`,
        operation: OPERATION,
        variant: 'means-merge',
        bindings: [
          {name: 'partial', view: partialMeans, type: 'f32', access: 'read'},
          {name: 'means', view: means, type: 'f32', access: 'read_write'}
        ],
        invocationCount: k,
        declarations: constants,
        body: `${mergeLoop(k)}
  means[meansOffset + index] = sum / f32(ROWS);`
      }),
      // Row table: [1, centered predictors, y, e] per row, so later passes read one buffer.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-row-table`,
        operation: OPERATION,
        variant: 'row-table',
        bindings: [
          {name: 'design', view: props.predictors, type: 'f32', access: 'read'},
          {name: 'response', view: props.response, type: 'f32', access: 'read'},
          {name: 'residuals', view: props.residuals, type: 'f32', access: 'read'},
          {name: 'means', view: means, type: 'f32', access: 'read'},
          {name: 'tableOut', view: rowTable, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rowCount,
        declarations: constants,
        body: `let base = tableOutOffset + index * TABLE_WIDTH;
  tableOut[base] = 1.0;
  for (var column = 0u; column < K; column++) {
    tableOut[base + 1u + column] =
      design[designOffset + index * K + column] - means[meansOffset + column];
  }
  tableOut[base + P] = response[responseOffset + index];
  tableOut[base + P + 1u] = residuals[residualsOffset + index];`
      }),
      ...transposeNodes,
      // W'Z row by row over the transposed CSR: (W'Z)_i = sum_j w_ji z_j, with j ascending.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-transposed-lag`,
        operation: OPERATION,
        variant: 'transposed-lag',
        bindings: [
          {name: 'offsets', view: transposed.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: transposed.neighbors, type: 'u32', access: 'read'},
          {name: 'weights', view: transposed.weights, type: 'f32', access: 'read'},
          rowTableBinding,
          {name: 'lagOut', view: transposedLagTable, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rowCount,
        declarations: constants,
        body: `var sums: array<f32, ${p}>;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    let weight = weights[weightsOffset + slot];
    for (var column = 0u; column < P; column++) {
      sums[column] += weight * table[tableOffset + neighbor * TABLE_WIDTH + column];
    }
  }
  for (var column = 0u; column < P; column++) {
    lagOut[lagOutOffset + index * P + column] = sums[column];
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-moments-tiles`,
        operation: OPERATION,
        variant: 'moments-tiles',
        workgroupSize: TILE_WORKGROUP_SIZE,
        bindings: [
          ...weightBindings,
          rowTableBinding,
          {name: 'transposedLag', view: transposedLagTable, type: 'f32', access: 'read'},
          {name: 'partial', view: partialMoments, type: 'f32', access: 'read_write'}
        ],
        invocationCount: tileCount,
        declarations: `${constants}
const PP: u32 = ${p * p}u;
const MOMENTS: u32 = ${momentCount}u;
${getRowAccumulationWGSL()}`,
        body: `let base = partialOffset + index * MOMENTS;
  for (var slot = 0u; slot < MOMENTS; slot++) {
    partial[base + slot] = 0.0;
  }
  let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  for (var row = firstRow; row < lastRow; row++) {
    accumulateRow(row, base);
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-moments-merge`,
        operation: OPERATION,
        variant: 'moments-merge',
        bindings: [
          {name: 'partial', view: partialMoments, type: 'f32', access: 'read'},
          {name: 'moments', view: moments, type: 'f32', access: 'read_write'}
        ],
        invocationCount: momentCount,
        declarations: constants,
        body: `${mergeLoop(momentCount)}
  moments[momentsOffset + index] = sum;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finish`,
        operation: OPERATION,
        variant: 'finish',
        bindings: [
          {name: 'moments', view: moments, type: 'f32', access: 'read'},
          {name: 'testsOut', view: output.tests, type: 'f32', access: 'read_write'},
          {name: 'summaryOut', view: output.summary, type: 'f32', access: 'read_write'},
          {name: 'statusOut', view: output.status, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `${constants}
const PP: u32 = ${p * p}u;
${getFinishWGSL(p)}`,
        body: 'finish();'
      })
    ];
  }
}

/** The per-row accumulation of the moment tile pass. Expects the bindings of that kernel. */
function getRowAccumulationWGSL(): string {
  return /* wgsl */ `
fn tableValue(row: u32, column: u32) -> f32 {
  return table[tableOffset + row * TABLE_WIDTH + column];
}

fn accumulateRow(row: u32, base: u32) {
  var z: array<f32, P>;
  var u: array<f32, P>;
  var v: array<f32, P>;
  for (var column = 0u; column < P; column++) {
    z[column] = tableValue(row, column);
    // (W'Z)_row, accumulated over the transposed CSR so one-way links are included.
    v[column] = transposedLag[transposedLagOffset + row * P + column];
  }
  var weightSum = 0.0;
  var squareSum = 0.0;
  var crossSum = 0.0;
  var lagResidual = 0.0;
  var lagResponse = 0.0;
  var lagFitted = 0.0;
  let residual = tableValue(row, P + 1u);
  for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= ROWS) {
      continue;
    }
    let weight = weights[weightsOffset + slot];
    // Reverse weight w_ji by binary search of this row in the neighbor's sorted row; it is 0 for
    // a one-way link, which then contributes nothing to tr(WW).
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
    weightSum += weight;
    squareSum += weight * weight;
    crossSum += weight * reverse;
    let neighborResponse = tableValue(neighbor, P);
    let neighborResidual = tableValue(neighbor, P + 1u);
    lagResponse += weight * neighborResponse;
    lagResidual += weight * neighborResidual;
    lagFitted += weight * (neighborResponse - neighborResidual);
    for (var column = 0u; column < P; column++) {
      u[column] += weight * tableValue(neighbor, column);
    }
  }
  for (var first = 0u; first < P; first++) {
    for (var second = 0u; second < P; second++) {
      let slot = first * P + second;
      partial[base + slot] += z[first] * z[second];
      partial[base + PP + slot] += z[first] * u[second];
      partial[base + 2u * PP + slot] += u[first] * u[second];
      partial[base + 3u * PP + slot] += v[first] * v[second];
      partial[base + 4u * PP + slot] += v[first] * u[second];
    }
  }
  let scalars = base + 5u * PP;
  partial[scalars] += weightSum;
  partial[scalars + 1u] += squareSum;
  partial[scalars + 2u] += crossSum;
  partial[scalars + 3u] += residual * residual;
  partial[scalars + 4u] += residual * lagResidual;
  partial[scalars + 5u] += residual * lagResponse;
  partial[scalars + 6u] += lagFitted * lagFitted;
  for (var column = 0u; column < P; column++) {
    partial[scalars + 7u + column] += z[column] * lagFitted;
  }
}
`;
}

/** The single-invocation finish: solves with the Gram matrix and evaluates the tests. */
function getFinishWGSL(p: number): string {
  return /* wgsl */ `
var<private> columnScale: array<f32, ${p}>;

fn getNaN() -> f32 {
  var bits = 0x7fc00000u;
  return bitcast<f32>(bits);
}

fn isFiniteValue(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

// Complementary error function, fractional error below 1.2e-7 (Numerical Recipes erfcc).
fn erfcValue(x: f32) -> f32 {
  let z = abs(x);
  let t = 1.0 / (1.0 + 0.5 * z);
  let r = t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
    t * (-0.82215223 + t * 0.17087277)))))))));
  return select(2.0 - r, r, x >= 0.0);
}

fn chiSquareSurvival(value: f32, degreesOfFreedom: u32) -> f32 {
  if (degreesOfFreedom == 1u) {
    return erfcValue(sqrt(max(value, 0.0) * 0.5));
  }
  return exp(-max(value, 0.0) * 0.5);
}

// Scaled (unit-diagonal Gram) entry of the p x p matrix stored at \`base\`.
fn scaledMoment(base: u32, row: u32, column: u32) -> f32 {
  return moments[momentsOffset + base + row * P + column] * columnScale[row] * columnScale[column];
}

${getCholeskyWGSL(p)}

// result = (Z'Z)^-1 Q (or its transpose version) for the matrix Q stored at \`base\`.
fn solveMatrix(
  factor: ptr<function, array<f32, ${p * p}>>,
  base: u32,
  transposed: bool,
  result: ptr<function, array<f32, ${p * p}>>
) {
  for (var column = 0u; column < P; column++) {
    var rhs: array<f32, ${p}>;
    for (var row = 0u; row < P; row++) {
      rhs[row] = select(
        scaledMoment(base, row, column),
        scaledMoment(base, column, row),
        transposed
      );
    }
    choleskySolve_${p}(factor, &rhs);
    for (var row = 0u; row < P; row++) {
      (*result)[row * P + column] = rhs[row];
    }
  }
}

fn writeTest(row: u32, statistic: f32, degreesOfFreedom: u32) {
  testsOut[testsOutOffset + row * 3u] = statistic;
  testsOut[testsOutOffset + row * 3u + 1u] = f32(degreesOfFreedom);
  testsOut[testsOutOffset + row * 3u + 2u] = select(
    getNaN(),
    chiSquareSurvival(statistic, degreesOfFreedom),
    isFiniteValue(statistic)
  );
}

fn fail(status: u32) {
  let nan = getNaN();
  for (var slot = 0u; slot < 18u; slot++) {
    testsOut[testsOutOffset + slot] = nan;
  }
  for (var slot = 0u; slot < 12u; slot++) {
    summaryOut[summaryOutOffset + slot] = nan;
  }
  summaryOut[summaryOutOffset] = f32(ROWS);
  for (var row = 0u; row < 5u; row++) {
    testsOut[testsOutOffset + row * 3u + 1u] = select(1.0, 2.0, row == 4u);
  }
  testsOut[testsOutOffset + 16u] = 0.0;
  statusOut[statusOutOffset] = status;
}

fn finish() {
  let n = f32(ROWS);
  let scalars = 5u * PP;
  let weightSum = moments[momentsOffset + scalars];
  let squareSum = moments[momentsOffset + scalars + 1u];
  let crossSum = moments[momentsOffset + scalars + 2u];
  let residualSquares = moments[momentsOffset + scalars + 3u];
  let residualLag = moments[momentsOffset + scalars + 4u];
  let responseLag = moments[momentsOffset + scalars + 5u];
  let fittedLagSquares = moments[momentsOffset + scalars + 6u];
  if (ROWS <= P + 1u) {
    fail(2u);
    return;
  }
  if (!isFiniteValue(residualSquares) || !isFiniteValue(residualLag) ||
      !isFiniteValue(responseLag) || !isFiniteValue(fittedLagSquares) ||
      !isFiniteValue(squareSum) || !isFiniteValue(crossSum) || !(residualSquares > 0.0)) {
    fail(4u);
    return;
  }
  let traceT = squareSum + crossSum;
  if (!(weightSum > 0.0) || !(traceT > 0.0)) {
    fail(3u);
    return;
  }
  for (var column = 0u; column < P; column++) {
    let diagonal = moments[momentsOffset + column * P + column];
    if (!(diagonal > 0.0) || !isFiniteValue(diagonal)) {
      fail(1u);
      return;
    }
    columnScale[column] = 1.0 / sqrt(diagonal);
  }
  var factor: array<f32, ${p * p}>;
  for (var row = 0u; row < P; row++) {
    for (var column = 0u; column < P; column++) {
      factor[row * P + column] = scaledMoment(0u, row, column);
    }
  }
  if (!choleskyFactor_${p}(&factor)) {
    fail(1u);
    return;
  }
  var inverseC: array<f32, ${p * p}>;
  var inverseCTransposed: array<f32, ${p * p}>;
  var inverseD: array<f32, ${p * p}>;
  var inverseE: array<f32, ${p * p}>;
  var inverseF: array<f32, ${p * p}>;
  solveMatrix(&factor, PP, false, &inverseC);
  solveMatrix(&factor, PP, true, &inverseCTransposed);
  solveMatrix(&factor, 2u * PP, false, &inverseD);
  solveMatrix(&factor, 3u * PP, false, &inverseE);
  solveMatrix(&factor, 4u * PP, false, &inverseF);
  var traceC = 0.0;
  var traceD = 0.0;
  var traceE = 0.0;
  var traceF = 0.0;
  var traceCC = 0.0;
  var traceCCTransposed = 0.0;
  for (var row = 0u; row < P; row++) {
    traceC += inverseC[row * P + row];
    traceD += inverseD[row * P + row];
    traceE += inverseE[row * P + row];
    traceF += inverseF[row * P + row];
    for (var column = 0u; column < P; column++) {
      traceCC += inverseC[row * P + column] * inverseC[column * P + row];
      traceCCTransposed += inverseC[row * P + column] * inverseCTransposed[column * P + row];
    }
  }
  // (W Zb)' M (W Zb) = v'v - (Z'v)' (Z'Z)^-1 (Z'v) with v = W (y - e).
  var projected: array<f32, ${p}>;
  var scaledProduct: array<f32, ${p}>;
  for (var column = 0u; column < P; column++) {
    scaledProduct[column] = moments[momentsOffset + scalars + 7u + column] * columnScale[column];
    projected[column] = scaledProduct[column];
  }
  choleskySolve_${p}(&factor, &projected);
  var quadratic = 0.0;
  for (var column = 0u; column < P; column++) {
    quadratic += scaledProduct[column] * projected[column];
  }
  let fittedLagResidualSquares = max(fittedLagSquares - quadratic, 0.0);

  let sigmaSquared = residualSquares / n;
  let errorGradient = residualLag / sigmaSquared;
  let lagGradient = responseLag / sigmaSquared;
  let information = fittedLagResidualSquares / sigmaSquared + traceT;
  let lagStatistic = lagGradient * lagGradient / information;
  let errorStatistic = errorGradient * errorGradient / traceT;
  let robustDenominator = traceT - traceT * traceT / information;
  var robustError = getNaN();
  var robustLag = getNaN();
  var sarma = getNaN();
  if (robustDenominator > 0.0 && information > traceT) {
    let numerator = errorGradient - traceT * lagGradient / information;
    robustError = numerator * numerator / robustDenominator;
    let difference = lagGradient - errorGradient;
    robustLag = difference * difference / (information - traceT);
    sarma = lagStatistic + robustError;
  }
  writeTest(0u, lagStatistic, 1u);
  writeTest(1u, errorStatistic, 1u);
  writeTest(2u, robustLag, 1u);
  writeTest(3u, robustError, 1u);
  writeTest(4u, sarma, 2u);

  let degrees = n - f32(P);
  let scaleFactor = n / weightSum;
  let moranI = scaleFactor * residualLag / residualSquares;
  let moranExpectation = -scaleFactor * traceC / degrees;
  let traceMWMWt = squareSum - traceE - traceD + traceCCTransposed;
  let traceMWMW = crossSum - 2.0 * traceF + traceCC;
  let traceMW = -traceC;
  let moranVariance = scaleFactor * scaleFactor * (traceMWMWt + traceMWMW + traceMW * traceMW) /
    (degrees * (degrees + 2.0)) - moranExpectation * moranExpectation;
  var moranZ = getNaN();
  var moranP = getNaN();
  if (moranVariance > 0.0) {
    moranZ = (moranI - moranExpectation) / sqrt(moranVariance);
    moranP = erfcValue(abs(moranZ) * 0.70710678);
  }
  testsOut[testsOutOffset + 15u] = moranZ;
  testsOut[testsOutOffset + 16u] = 0.0;
  testsOut[testsOutOffset + 17u] = moranP;

  summaryOut[summaryOutOffset] = n;
  summaryOut[summaryOutOffset + 1u] = sigmaSquared;
  summaryOut[summaryOutOffset + 2u] = traceT;
  summaryOut[summaryOutOffset + 3u] = information;
  summaryOut[summaryOutOffset + 4u] = weightSum;
  summaryOut[summaryOutOffset + 5u] = moranI;
  summaryOut[summaryOutOffset + 6u] = moranExpectation;
  summaryOut[summaryOutOffset + 7u] = moranVariance;
  summaryOut[summaryOutOffset + 8u] = moranZ;
  summaryOut[summaryOutOffset + 9u] = moranP;
  summaryOut[summaryOutOffset + 10u] = residualLag;
  summaryOut[summaryOutOffset + 11u] = responseLag;
  statusOut[statusOutOffset] = 0u;
}
`;
}
