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
import {GPUOrdinaryLeastSquares} from './gpu-ordinary-least-squares';
import {
  GPUSpatialWeightsTraces,
  validateGPUSpatialWeightsTracesParameters
} from './gpu-spatial-weights-traces';
import {
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_CONDITION_NUMBER,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH
} from './ordinary-least-squares-parameters';
import {
  createKernelFactory,
  createSeriesProbeNodes,
  getF32Binding,
  getU32Binding,
  SPATIAL_MAXIMUM_LIKELIHOOD_TILE_WORKGROUP_SIZE
} from './spatial-maximum-likelihood-nodes';
import {
  GPU_SPATIAL_LAG_MAXIMUM_LIKELIHOOD_MAXIMUM_PREDICTOR_COUNT,
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
import {getCholeskyWGSL} from './spatial-regression-solve';
import {
  GPU_SPATIAL_WEIGHTS_TRACES_DEFAULT_PROBE_COUNT,
  GPU_SPATIAL_WEIGHTS_TRACES_EXACT_ROW_LIMIT,
  GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_EXACT_ROW_COUNT,
  getSpatialLogDeterminantWGSL,
  type GPUSpatialWeightsTracesMode
} from './spatial-weights-traces-parameters';

const OPERATION = 'GPUSpatialLagMaximumLikelihood';
const MAXIMUM_DEFAULT_TILE_COUNT = 4096;
const MINIMUM_DEFAULT_TILE_ROWS = 64;
const TILE_WORKGROUP_SIZE = SPATIAL_MAXIMUM_LIKELIHOOD_TILE_WORKGROUP_SIZE;
const LN_TWO_PI = Math.log(2 * Math.PI);

/** Caller-owned outputs of {@link GPUSpatialLagMaximumLikelihood}. */
export type GPUSpatialLagMaximumLikelihoodOutput = {
  /**
   * One-row solver `code` (0 on success) and, optionally, `converged` (1 when the optimum is an
   * interior stationary point).
   */
  status: Pick<GPUSolverStatusPort, 'code'> & Partial<Pick<GPUSolverStatusPort, 'converged'>>;

  /**
   * `(predictorCount + 2) * 4` float32 values: one row per coefficient in spreg order (intercept,
   * one per predictor, then `rho` last), each row holding the estimate, standard error, z
   * statistic and two-sided normal p-value. The layout is the one `GPUSpatialLagImpacts` reads
   * with `coefficientStride: 4` and `rhoIndex: predictorCount + 1`.
   */
  table: GraphDataView<'float32'>;
  /** At least 22 float32 values; see the `GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_*` slots. */
  summary: GraphDataView<'float32'>;
  /** Optional per-row residuals `u = y - rho W y - X b` (spreg `u`); NaN for a failed fit. */
  residuals?: GraphDataView<'float32'>;
  /**
   * Optional `maximumPower` float32 values `tr((W/s)^q)` the fit used, so that a downstream
   * `GPUSpatialLagImpacts` can share them. Provide `powerTraces`, `powerSums` and `infinityNorm`
   * together or none.
   */
  powerTraces?: GraphDataView<'float32'>;
  /** Optional `maximumPower` float32 values `1' (W/s)^q 1`, see `powerTraces`. */
  powerSums?: GraphDataView<'float32'>;
  /** Optional one float32 value `s = ||W||_inf` of the traces; provided with `powerTraces`. */
  infinityNorm?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUSpatialLagMaximumLikelihood}.
 *
 * Per-frame (no recompile): the contents of every input buffer. Topology (needs a new graph):
 * `predictorCount`, the row count, the weights views, `maximumPower`, the trace options,
 * `tileRowCount` and which optional outputs are present.
 */
export type GPUSpatialLagMaximumLikelihoodProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'spatial-lag-maximum-likelihood'`. */
  id?: string;
  /**
   * Square weights `W`. Any sparsity pattern works (no symmetry is required) and rows need not be
   * standardized: `rho` is searched within `min(0.99, seriesLimit) / ||W||_inf`.
   */
  weights: GPUSpatialWeights;
  /** Row-major predictors, `predictors[row * predictorCount + column]`, without an intercept. */
  predictors: GraphDataView<'float32'>;
  /** Response `y`, one value per row. */
  response: GraphDataView<'float32'>;
  /** Number of predictor columns, compile-time, 1 to 14. The intercept is implicit. */
  predictorCount: number;
  /**
   * Series length `Q` of the log-determinant, `tr1`, `tr2` and the probe pass, an integer in
   * `[2, 256]`. Defaults to 128. See the class documentation for the truncation bound.
   */
  maximumPower?: number;
  /**
   * `'exact'` (unit probes, no sampling error) or `'stochastic'` (Hutchinson). Defaults to exact
   * up to 4096 rows and stochastic above, where the likelihood, hence `rho`, carries stochastic
   * error and `GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_APPROXIMATION` is 1.
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
  output: GPUSpatialLagMaximumLikelihoodOutput;
};

/**
 * Maximum-likelihood estimator of the spatial lag model `y = rho W y + X b + e`, `e ~ N(0, s2 I)`
 * (Anselin 1988; spreg `ML_Lag` with `method='full'`, whose exact log-determinant is replaced
 * here by a truncated trace series). Nothing is read back.
 *
 * 1. `e0` and `e1` are the OLS residuals of `y` and `W y` on `[1, X]` (two
 *    `GPUOrdinaryLeastSquares` compositions; `W y` comes from an inline sparse product), so
 *    `u(rho) = e0 - rho e1` and `s2(rho) = u'u / n` need only the sums `e0'e0`, `e0'e1`, `e1'e1`.
 * 2. The concentrated log-likelihood `-n/2 ln(s2(rho)) + ln|I - rho W|` has the log-determinant
 *    `-sum_{q=1}^{Q} rho^q tr(W^q) / q` from `GPUSpatialWeightsTraces` (exact traces up to 4096
 *    rows, Hutchinson estimates above), evaluated by Horner's rule.
 * 3. `rho` is the global maximum of the likelihood on `[-B, B]`. A single kernel evaluates the
 *    analytic derivative on 1025 grid points, refines every positive-to-non-positive sign change
 *    by 32 bisection steps on the derivative and keeps the best candidate (end points count when
 *    the derivative points outward). It is deterministic and needs no iteration budget.
 * 4. `b = b0 - rho b1`, `s2 = u'u / n`, the full log-likelihood
 *    `-n/2 ln(s2) + ln|I - rho W| - n/2 (ln 2 pi + 1)`, spreg `pr2` (squared correlation of `y`
 *    and `y - u`) and `pr2_e` (of `y` and `(I - rho W)^-1 X b`).
 * 5. The variance is the inverse of spreg's information matrix of `(b, rho, s2)`, with
 *    `tr1 = tr(W A)`, `tr2 = tr((W A)^2)` as series in `rho` of the power traces, and
 *    `tr3 = tr((W A)' W A)` from a second probe pass at the estimated `rho`
 *    (`createSeriesProbeNodes`), with `A = (I - rho W)^-1`. The `s2` row is eliminated by its Schur
 *    complement (the `rho` entry becomes `tr2 + tr3 + (W X b~)'(W X b~) / s2 - 2 tr1^2 / n`), the
 *    matrix is built on centered predictors and unit-scaled before the Cholesky factorization, and
 *    the intercept variance is mapped back to the uncentered model.
 *
 * Search interval. `B = min(0.99, seriesLimit) / ||W||_inf`, with `||W||_inf` the largest absolute
 * row sum computed on the GPU, which keeps `I - rho W` invertible for any `W` (spreg's fixed
 * `(-1, 1)` admits explosive `rho` for raw weights) and equals 0.99 for a row-standardized `W`
 * (spreg searches up to 1). The truncated series omits a tail below
 * `n s^(Q+1) / ((Q+1) (1 - s))` with `s = |rho| ||W||_inf` (the worst case, `|tr(W^q)| <= n s^q`),
 * and the omission biases the likelihood toward large positive `rho`, so the interval also stops
 * where this tail exceeds `1e-5 n` (`getSpatialMaximumLikelihoodSeriesLimit`): about `|rho| < 0.93`
 * for the default `Q = 128` and 0.96 for 256. Beyond it the fit reports a boundary solution
 * (`..._SUMMARY_AT_BOUNDARY`, `..._SERIES_LIMITED`). The default `Q = 128` keeps the worst-case
 * truncation below `1e-7 n` log-likelihood units at `|rho| <= 0.9` (below `1e-6 |LL|`), at the
 * price of about `Q * (1 + probe batches)` dispatches; raise it when `rho` is near the bound.
 * `..._SUMMARY_TRUNCATION_BOUND` reports the worst-case remainder at the estimate. The
 * hand-written spreg search uses an `xatol` of `1e-7`, so `rho` agrees with it to about that.
 *
 * The float32 traces `tr(W^q)` grow like `n ||W||_inf^q`: for a raw (unstandardized) `W` choose
 * `maximumPower` with `||W||_inf^maximumPower < 1e36`, else they overflow and the status is 3.
 * Inputs must be finite: a non-finite predictor, response or weight gives status 3 (NaN outputs).
 * At most 14 predictors (the 16-wide Cholesky of the `(predictorCount + 2)`-wide information
 * matrix); no kernel binds more than 8 storage buffers. The `table` feeds `GPUSpatialLagImpacts`
 * directly.
 */
export class GPUSpatialLagMaximumLikelihood implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialLagMaximumLikelihoodProps;
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

  constructor(props: GPUSpatialLagMaximumLikelihoodProps) {
    const id = props.id ?? 'spatial-lag-maximum-likelihood';
    this.id = id;
    this.props = props;
    const {predictorCount, output} = props;
    validateCapacity(
      id,
      'predictorCount',
      predictorCount,
      1,
      GPU_SPATIAL_LAG_MAXIMUM_LIKELIHOOD_MAXIMUM_PREDICTOR_COUNT
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
    if (
      (output.powerTraces === undefined) !== (output.powerSums === undefined) ||
      (output.powerTraces === undefined) !== (output.infinityNorm === undefined)
    ) {
      throw new Error(
        `${id} output.powerTraces, output.powerSums and output.infinityNorm must be provided together`
      );
    }
    for (const [name, view] of [
      ['powerTraces', output.powerTraces],
      ['powerSums', output.powerSums],
      ['infinityNorm', output.infinityNorm]
    ] as const) {
      if (view) {
        validatePackedView(view, ['float32'], `${id} output.${name}`);
        if (view.length < (name === 'infinityNorm' ? 1 : maximumPower)) {
          throw new Error(`${id} output.${name} must hold maximumPower float32 values`);
        }
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
      [
        output.table,
        output.summary,
        output.status.code,
        output.status.converged,
        output.residuals,
        output.powerTraces,
        output.powerSums,
        output.infinityNorm
      ],
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
   * Returns the lagged-response and weights-norm node, the two `GPUOrdinaryLeastSquares`
   * compositions, `GPUSpatialWeightsTraces`, the means and residual moment passes, the `rho`
   * search, the coefficient solve, the residual pass, the `Q - 1` Horner steps of
   * `(I - rho W)^-1 X b` and the final moment passes, the `tr3` probe pass (per batch `Q` steps
   * and a merge), the variance and the finish kernel.
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
      output.residuals,
      output.powerTraces,
      output.powerSums,
      output.infinityNorm
    ]);
    const p = k + 1;
    const informationSize = p + 1;
    const stochastic = this.traceMode === 'stochastic';
    const packedCount = (k * (k + 1)) / 2;
    // Final sums: [sum wp, sum xc_j wp (k), sum wp^2, Syy, Syu, Su, Suu, Sye, See, Se, gram (packed)].
    const sumWeighted = 0;
    const sumWeightedSquares = k + 1;
    const sumResponseSquares = k + 2;
    const sumResponseResidual = k + 3;
    const sumResidual = k + 4;
    const sumResidualSquares = k + 5;
    const sumResponsePrediction = k + 6;
    const sumPredictionSquares = k + 7;
    const sumPrediction = k + 8;
    const gramBase = k + 9;
    const finalSumCount = gramBase + packedCount;

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
    allocate('BETA', p);
    allocate('MEANS', k);
    allocate('YBAR');
    allocate('CONDITIONING', 4);
    allocate('TRACES', 3);
    allocate('STANDARD_ERRORS', p + 1);
    allocate('VARIANCE_DEFINED');
    const slotConstants = Object.entries(slots)
      .map(([name, value]) => `const W_${name}: u32 = ${value}u;`)
      .join('\n');

    const transient = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'float32', length);
    const transientUint = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);
    const kernel = createKernelFactory(graph, id, OPERATION);
    const workspace = transient('workspace', slotCount);
    const constants = `const ROWS: u32 = ${rowCount}u;
const K: u32 = ${k}u;
const P: u32 = ${p}u;
const TILE_ROWS: u32 = ${tileRowCount}u;
const TILE_COUNT: u32 = ${tileCount}u;
const MAXIMUM_POWER: u32 = ${maximumPower}u;
${slotConstants}`;
    const finiteDeclarations = `${GET_NAN_WGSL}
${IS_FINITE_VALUE_WGSL}`;
    const f32 = getF32Binding;
    const weightBindings: WGSLKernelBinding[] = [
      getU32Binding('offsets', weights.offsets, 'read'),
      getU32Binding('neighbors', weights.neighbors, 'read'),
      f32('weights', weights.weights, 'read')
    ];

    // ---- First stage: W y, ||W||_inf tiles and the two OLS fits. ----
    const lagged = transient('lagged-response', rowCount);
    const partialNorm = transient('partial-weights-norm', tileCount);
    const olsResiduals = [
      transient('ols-residuals-response', rowCount),
      transient('ols-residuals-lagged', rowCount)
    ];
    const olsCoefficients = [p, p].map((length, index) =>
      transient(`ols-coefficients-${index}`, length)
    );
    const olsSummaries = [0, 1].map(index =>
      transient(`ols-summary-${index}`, GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH)
    );
    const olsStatuses = [0, 1].map(index => transientUint(`ols-status-${index}`, 1));
    const olsNodes = [0, 1].map(index =>
      new GPUOrdinaryLeastSquares({
        id: `${id}-ols-${index === 0 ? 'response' : 'lagged'}`,
        predictors: props.predictors,
        response: index === 0 ? props.response : lagged,
        predictorCount: k,
        tileRowCount,
        output: {
          coefficients: olsCoefficients[index],
          standardErrors: transient(`ols-standard-errors-${index}`, p),
          tStatistics: transient(`ols-t-statistics-${index}`, p),
          summary: olsSummaries[index],
          status: {code: olsStatuses[index]},
          residuals: olsResiduals[index]
        }
      }).getCommandNodes(graph)
    );

    const powerTraces = output.powerTraces ?? transient('power-traces', maximumPower);
    const powerSums = output.powerSums ?? transient('power-sums', maximumPower);
    const infinityNorm = output.infinityNorm ?? transient('traces-infinity-norm', 1);

    const partialMeans = transient('partial-means', tileCount * (k + 1));
    const partialMoments = transient('partial-moments', tileCount * 3);
    const residualView = output.residuals ?? transient('residuals', rowCount);
    const reduced = transient('reduced-prediction', rowCount);
    const chain = [transient('chain-a', rowCount), transient('chain-b', rowCount)];
    const weightedPrediction = transient('weighted-prediction', rowCount);
    const partialFinal = transient('partial-final-sums', tileCount * finalSumCount);
    const finalSums = transient('final-sums', finalSumCount);

    const nodes: GPUCommandNode<Parameters>[] = [
      kernel(
        'lagged-response-tiles',
        [
          ...weightBindings,
          f32('response', props.response, 'read'),
          f32('lagged', lagged, 'read_write'),
          f32('partialNorm', partialNorm, 'read_write')
        ],
        tileCount,
        constants,
        `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var largest = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    var sum = 0.0;
    var absoluteSum = 0.0;
    for (var slot = offsets[offsetsOffset + row]; slot < offsets[offsetsOffset + row + 1u]; slot++) {
      let neighbor = neighbors[neighborsOffset + slot];
      if (neighbor >= ROWS) {
        continue;
      }
      let weight = weights[weightsOffset + slot];
      sum += weight * response[responseOffset + neighbor];
      absoluteSum += abs(weight);
    }
    lagged[laggedOffset + row] = sum;
    largest = max(largest, absoluteSum);
  }
  partialNorm[partialNormOffset + index] = largest;`,
        {workgroupSize: TILE_WORKGROUP_SIZE}
      ),
      ...olsNodes[0],
      ...olsNodes[1],
      ...new GPUSpatialWeightsTraces({
        id: `${id}-traces`,
        weights,
        maximumPower,
        mode: this.traceMode,
        probeCount: props.traceProbeCount,
        parameters: props.traceParameters,
        output: {powerTraces, powerSums, infinityNorm}
      }).getCommandNodes(graph),
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
      kernel(
        'moments-tiles',
        [
          f32('residual0', olsResiduals[0], 'read'),
          f32('residual1', olsResiduals[1], 'read'),
          f32('partial', partialMoments, 'read_write')
        ],
        tileCount,
        constants,
        `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  var sumA = 0.0;
  var sumB = 0.0;
  var sumC = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    let e0 = residual0[residual0Offset + row];
    let e1 = residual1[residual1Offset + row];
    sumA += e0 * e0;
    sumB += e0 * e1;
    sumC += e1 * e1;
  }
  partial[partialOffset + index * 3u] = sumA;
  partial[partialOffset + index * 3u + 1u] = sumB;
  partial[partialOffset + index * 3u + 2u] = sumC;`,
        {workgroupSize: TILE_WORKGROUP_SIZE}
      ),
      kernel(
        'rho-search',
        [
          f32('partialNorm', partialNorm, 'read'),
          f32('partialMoments', partialMoments, 'read'),
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
var<private> sumA: f32;
var<private> sumB: f32;
var<private> sumC: f32;

fn getSumOfSquares(rho: f32) -> f32 {
  return sumA - 2.0 * rho * sumB + rho * rho * sumC;
}

fn getObjective(rho: f32) -> f32 {
  return -0.5 * ROWS_F * log(getSumOfSquares(rho) / ROWS_F) + getSpatialLogDeterminant(rho);
}

fn getDerivative(rho: f32) -> f32 {
  return -0.5 * ROWS_F * (-2.0 * sumB + 2.0 * rho * sumC) / getSumOfSquares(rho) + getLogDeterminantDerivative(rho);
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
  ${getKahanSumWGSL('partialMoments', '0u', '3u', 'TILE_COUNT', 'momentA')}
  ${getKahanSumWGSL('partialMoments', '1u', '3u', 'TILE_COUNT', 'momentB')}
  ${getKahanSumWGSL('partialMoments', '2u', '3u', 'TILE_COUNT', 'momentC')}
  sumA = momentA;
  sumB = momentB;
  sumC = momentC;
  if (validNorm && isFiniteValue(sumA) && isFiniteValue(sumB) && isFiniteValue(sumC) && sumA > 0.0 && sumC > 0.0) {
    let result = searchParameter(bound);
    workspace[workspaceOffset + W_PARAMETER] = result.parameter;
    workspace[workspaceOffset + W_VALID] = select(0.0, 1.0, isFiniteValue(result.parameter));
    workspace[workspaceOffset + W_CANDIDATES] = f32(result.localMaximumCount);
    workspace[workspaceOffset + W_AT_BOUNDARY] = select(0.0, 1.0, result.atBoundary);
  }`
      ),
      kernel(
        'solve-coefficients',
        [
          f32('olsCoefficients0', olsCoefficients[0], 'read'),
          f32('olsCoefficients1', olsCoefficients[1], 'read'),
          f32('olsSummary0', olsSummaries[0], 'read'),
          f32('olsSummary1', olsSummaries[1], 'read'),
          getU32Binding('olsStatus0', olsStatuses[0], 'read'),
          getU32Binding('olsStatus1', olsStatuses[1], 'read'),
          f32('workspace', workspace, 'read_write')
        ],
        1,
        `${constants}
${finiteDeclarations}`,
        `let rho = workspace[workspaceOffset + W_PARAMETER];
  var code = 0u;
  if (!(f32(ROWS) > f32(K + 2u))) {
    code = 2u;
  } else if (olsStatus0[olsStatus0Offset] != 0u) {
    code = select(1u, 2u, olsStatus0[olsStatus0Offset] == 2u);
  } else if (olsStatus1[olsStatus1Offset] != 0u) {
    code = select(1u, 2u, olsStatus1[olsStatus1Offset] == 2u);
  } else if (olsSummary0[olsSummary0Offset] != f32(ROWS) || olsSummary1[olsSummary1Offset] != f32(ROWS)) {
    code = 3u;
  } else if (workspace[workspaceOffset + W_VALID] != 1.0 || !isFiniteValue(rho)) {
    code = 3u;
  }
  workspace[workspaceOffset + W_STATUS] = f32(code);
  for (var column = 0u; column < P; column++) {
    workspace[workspaceOffset + W_BETA + column] = olsCoefficients0[olsCoefficients0Offset + column] - rho * olsCoefficients1[olsCoefficients1Offset + column];
  }
  for (var slot = 0u; slot < 4u; slot++) {
    workspace[workspaceOffset + W_CONDITIONING + slot] = olsSummary0[olsSummary0Offset + ${GPU_ORDINARY_LEAST_SQUARES_SUMMARY_CONDITION_NUMBER}u + slot];
  }`
      ),
      kernel(
        'residuals',
        [
          f32('response', props.response, 'read'),
          f32('lagged', lagged, 'read'),
          f32('residual0', olsResiduals[0], 'read'),
          f32('residual1', olsResiduals[1], 'read'),
          f32('residualOut', residualView, 'read_write'),
          f32('reduced', reduced, 'read_write'),
          f32('workspace', workspace, 'read')
        ],
        rowCount,
        constants,
        `let rho = workspace[workspaceOffset + W_PARAMETER];
  residualOut[residualOutOffset + index] = residual0[residual0Offset + index] - rho * residual1[residual1Offset + index];
  // X b = (y - e0) - rho (W y - e1): the OLS fitted values of y and W y.
  reduced[reducedOffset + index] = (response[responseOffset + index] - residual0[residual0Offset + index])
    - rho * (lagged[laggedOffset + index] - residual1[residual1Offset + index]);`
      )
    ];

    // ---- (I - rho W)^-1 X b by Horner: u <- z + rho W u, Q - 1 times from u = z. ----
    for (let step = 1; step < maximumPower; step++) {
      const first = step === 1;
      nodes.push(
        kernel(
          `inverse-series-${step}`,
          [
            ...weightBindings,
            f32('reduced', reduced, 'read'),
            ...(first ? [] : [f32('vectorIn', chain[step % 2], 'read')]),
            f32('vectorOut', chain[(step - 1) % 2], 'read_write'),
            f32('workspace', workspace, 'read')
          ],
          rowCount,
          constants,
          `let rho = workspace[workspaceOffset + W_PARAMETER];
  var sum = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= ROWS) {
      continue;
    }
    sum += weights[weightsOffset + slot] * ${first ? 'reduced[reducedOffset + neighbor]' : 'vectorIn[vectorInOffset + neighbor]'};
  }
  vectorOut[vectorOutOffset + index] = reduced[reducedOffset + index] + rho * sum;`
        )
      );
    }
    const spatialPrediction = chain[(maximumPower - 2) % 2];
    nodes.push(
      kernel(
        'weighted-prediction',
        [
          ...weightBindings,
          f32('vectorIn', spatialPrediction, 'read'),
          f32('vectorOut', weightedPrediction, 'read_write')
        ],
        rowCount,
        constants,
        `var sum = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let neighbor = neighbors[neighborsOffset + slot];
    if (neighbor >= ROWS) {
      continue;
    }
    sum += weights[weightsOffset + slot] * vectorIn[vectorInOffset + neighbor];
  }
  vectorOut[vectorOutOffset + index] = sum;`
      ),
      kernel(
        'final-moments-tiles',
        [
          f32('design', props.predictors, 'read'),
          f32('response', props.response, 'read'),
          f32('residual', residualView, 'read'),
          f32('prediction', spatialPrediction, 'read'),
          f32('weighted', weightedPrediction, 'read'),
          f32('workspace', workspace, 'read'),
          f32('partial', partialFinal, 'read_write')
        ],
        tileCount,
        constants,
        `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROWS);
  let ybar = workspace[workspaceOffset + W_YBAR];
  var sums: array<f32, ${finalSumCount}>;
  var centered: array<f32, ${k}>;
  for (var row = firstRow; row < lastRow; row++) {
    let y = response[responseOffset + row] - ybar;
    let u = residual[residualOffset + row];
    let e = prediction[predictionOffset + row] - ybar;
    let wp = weighted[weightedOffset + row];
    sums[${sumWeighted}] += wp;
    sums[${sumWeightedSquares}] += wp * wp;
    sums[${sumResponseSquares}] += y * y;
    sums[${sumResponseResidual}] += y * u;
    sums[${sumResidual}] += u;
    sums[${sumResidualSquares}] += u * u;
    sums[${sumResponsePrediction}] += y * e;
    sums[${sumPredictionSquares}] += e * e;
    sums[${sumPrediction}] += e;
    for (var column = 0u; column < K; column++) {
      centered[column] = design[designOffset + row * K + column] - workspace[workspaceOffset + W_MEANS + column];
      sums[1u + column] += centered[column] * wp;
    }
    for (var first = 0u; first < K; first++) {
      for (var second = 0u; second <= first; second++) {
        sums[${gramBase}u + first * (first + 1u) / 2u + second] += centered[first] * centered[second];
      }
    }
  }
  for (var slot = 0u; slot < ${finalSumCount}u; slot++) {
    partial[partialOffset + index * ${finalSumCount}u + slot] = sums[slot];
  }`,
        {workgroupSize: TILE_WORKGROUP_SIZE}
      ),
      kernel(
        'final-merge',
        [f32('partial', partialFinal, 'read'), f32('finalSums', finalSums, 'read_write')],
        finalSumCount,
        constants,
        `${getKahanSumWGSL('partial', 'index', `${finalSumCount}u`, 'TILE_COUNT')}
  finalSums[finalSumsOffset + index] = sum;`
      )
    );

    // ---- tr3 = tr((W A)' W A) probe pass at the estimated rho. ----
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
          f32('finalSums', finalSums, 'read'),
          f32('dots', probes.dots, 'read'),
          f32('powerTraces', powerTraces, 'read'),
          f32('infinityNorm', infinityNorm, 'read'),
          f32('workspace', workspace, 'read_write')
        ],
        1,
        `${constants}
${finiteDeclarations}
${getTraceSeriesWGSL(maximumPower, 'powerTraces', 'infinityNorm[infinityNormOffset]')}
${getCholeskyWGSL(informationSize)}`,
        `let n = f32(ROWS);
  let rho = workspace[workspaceOffset + W_PARAMETER];
  let sigma2 = finalSums[finalSumsOffset + ${sumResidualSquares}u] / n;
  let nan = getNaN();
  for (var slot = 0u; slot < P + 1u; slot++) {
    workspace[workspaceOffset + W_STANDARD_ERRORS + slot] = nan;
  }
  workspace[workspaceOffset + W_VARIANCE_DEFINED] = 0.0;
  let traceOne = getTraceOne(rho);
  let traceTwo = getTraceTwo(rho);
  ${probes.dotsSumWGSL}
  let traceThree = sum * ${probeScale};
  workspace[workspaceOffset + W_TRACES] = traceOne;
  workspace[workspaceOffset + W_TRACES + 1u] = traceTwo;
  workspace[workspaceOffset + W_TRACES + 2u] = traceThree;
  if (!(sigma2 > 0.0) || !isFiniteValue(sigma2) || !isFiniteValue(traceThree) || !isFiniteValue(rho)) {
    return;
  }
  // Information matrix on [intercept (centered), b (K), rho]; the sigma2 row is eliminated.
  var a: array<f32, ${informationSize * informationSize}>;
  a[0] = n / sigma2;
  for (var first = 0u; first < K; first++) {
    for (var second = 0u; second <= first; second++) {
      let value = finalSums[finalSumsOffset + ${gramBase}u + first * (first + 1u) / 2u + second] / sigma2;
      a[(1u + first) * ${informationSize}u + 1u + second] = value;
      a[(1u + second) * ${informationSize}u + 1u + first] = value;
    }
  }
  for (var column = 0u; column < P; column++) {
    let value = finalSums[finalSumsOffset + ${sumWeighted}u + column] / sigma2;
    a[column * ${informationSize}u + P] = value;
    a[P * ${informationSize}u + column] = value;
  }
  a[P * ${informationSize}u + P] = traceTwo + traceThree + finalSums[finalSumsOffset + ${sumWeightedSquares}u] / sigma2 - 2.0 * traceOne * traceOne / n;
  var scale: array<f32, ${informationSize}>;
  for (var column = 0u; column < ${informationSize}u; column++) {
    let diagonal = a[column * ${informationSize}u + column];
    if (!(diagonal > 0.0) || !isFiniteValue(diagonal)) {
      return;
    }
    scale[column] = sqrt(diagonal);
  }
  for (var row = 0u; row < ${informationSize}u; row++) {
    for (var column = 0u; column < ${informationSize}u; column++) {
      a[row * ${informationSize}u + column] = a[row * ${informationSize}u + column] / (scale[row] * scale[column]);
    }
  }
  if (!choleskyFactor_${informationSize}(&a)) {
    return;
  }
  // Covariance V = D^-1 (scaled inverse) D^-1, one unit solve per column.
  var covariance: array<f32, ${informationSize * informationSize}>;
  for (var column = 0u; column < ${informationSize}u; column++) {
    var unit: array<f32, ${informationSize}>;
    unit[column] = 1.0;
    choleskySolve_${informationSize}(&a, &unit);
    for (var row = 0u; row < ${informationSize}u; row++) {
      covariance[row * ${informationSize}u + column] = unit[row] / (scale[row] * scale[column]);
    }
  }
  // Intercept of the uncentered model: b0 = a0 - m'b.
  var interceptVariance = covariance[0];
  for (var first = 0u; first < K; first++) {
    let meanFirst = workspace[workspaceOffset + W_MEANS + first];
    interceptVariance -= 2.0 * meanFirst * covariance[1u + first];
    for (var second = 0u; second < K; second++) {
      interceptVariance += meanFirst * workspace[workspaceOffset + W_MEANS + second] * covariance[(1u + first) * ${informationSize}u + 1u + second];
    }
  }
  workspace[workspaceOffset + W_STANDARD_ERRORS] = sqrt(interceptVariance);
  for (var column = 1u; column < P; column++) {
    workspace[workspaceOffset + W_STANDARD_ERRORS + column] = sqrt(covariance[column * ${informationSize}u + column]);
  }
  workspace[workspaceOffset + W_STANDARD_ERRORS + P] = sqrt(covariance[P * ${informationSize}u + P]);
  workspace[workspaceOffset + W_VARIANCE_DEFINED] = 1.0;`
      ),
      kernel(
        'finish',
        [
          f32('finalSums', finalSums, 'read'),
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
${getFinishWGSL({
  predictorCount: k,
  maximumPower,
  stochastic,
  sumResponseSquares,
  sumResponseResidual,
  sumResidual,
  sumResidualSquares,
  sumResponsePrediction,
  sumPredictionSquares,
  sumPrediction,
  hasConverged: output.status.converged !== undefined
})}`,
        'finish();'
      )
    );
    return nodes;
  }
}

function getFinishWGSL(options: {
  predictorCount: number;
  maximumPower: number;
  stochastic: boolean;
  sumResponseSquares: number;
  sumResponseResidual: number;
  sumResidual: number;
  sumResidualSquares: number;
  sumResponsePrediction: number;
  sumPredictionSquares: number;
  sumPrediction: number;
  hasConverged: boolean;
}): string {
  const {predictorCount: k} = options;
  const rows = k + 2;
  const summary = (slot: number) => `summaryOut[summaryOutOffset + ${slot}u]`;
  const sum = (slot: number) => `finalSums[finalSumsOffset + ${slot}u]`;
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
${options.hasConverged ? '  convergedOut[convergedOutOffset] = 0u;' : ''}
}

fn finish() {
  let code = u32(workspace[workspaceOffset + W_STATUS]);
  if (code != 0u) {
    fail(code);
    return;
  }
  let n = f32(ROWS);
  let rho = workspace[workspaceOffset + W_PARAMETER];
  let sigma2 = ${sum(options.sumResidualSquares)} / n;
  if (!(sigma2 > 0.0) || !isFiniteValue(sigma2)) {
    fail(3u);
    return;
  }
  let defined = workspace[workspaceOffset + W_VARIANCE_DEFINED] == 1.0;
  for (var row = 0u; row < P; row++) {
    writeTableRow(row, workspace[workspaceOffset + W_BETA + row], workspace[workspaceOffset + W_STANDARD_ERRORS + row]);
  }
  writeTableRow(P, rho, workspace[workspaceOffset + W_STANDARD_ERRORS + P]);
  let logDeterminant = getSpatialLogDeterminant(rho);
  let logLikelihood = -0.5 * n * log(sigma2) + logDeterminant - 0.5 * n * (${getWGSLFloatLiteral(LN_TWO_PI)} + 1.0);
  let parameterCount = f32(P + 1u);
  // Squared correlations of y with y - u and with (I - rho W)^-1 X b.
  let responseSquares = ${sum(options.sumResponseSquares)};
  let responseResidual = ${sum(options.sumResponseResidual)};
  let residualTotal = ${sum(options.sumResidual)};
  let predictionCovariance = responseSquares - responseResidual;
  let predictionVariance = responseSquares - 2.0 * responseResidual + sigma2 * n - residualTotal * residualTotal / n;
  let spatialCovariance = ${sum(options.sumResponsePrediction)};
  let spatialVariance = ${sum(options.sumPredictionSquares)} - ${sum(options.sumPrediction)} * ${sum(options.sumPrediction)} / n;
  let conditionNumber = workspace[workspaceOffset + W_CONDITIONING];
  let correlationCondition = workspace[workspaceOffset + W_CONDITIONING + 1u];
  let bound = workspace[workspaceOffset + W_BOUND];
  let norm = workspace[workspaceOffset + W_NORM];
  let scaled = abs(rho) * norm;
  let atBoundary = workspace[workspaceOffset + W_AT_BOUNDARY] == 1.0;
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_ROW_COUNT)} = n;
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PARAMETER)} = rho;
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SIGMA_SQUARED)} = sigma2;
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOG_LIKELIHOOD)} = logLikelihood;
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_AIC)} = 2.0 * parameterCount - 2.0 * logLikelihood;
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SCHWARZ)} = parameterCount * log(n) - 2.0 * logLikelihood;
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PSEUDO_R_SQUARED)} = predictionCovariance * predictionCovariance / (responseSquares * predictionVariance);
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PSEUDO_R_SQUARED_SPATIAL)} = spatialCovariance * spatialCovariance / (responseSquares * spatialVariance);
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_RESIDUAL_SUM_OF_SQUARES)} = sigma2 * n;
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CONDITION_NUMBER)} = conditionNumber;
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CORRELATION_CONDITION_NUMBER)} = correlationCondition;
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_MAXIMUM_VARIANCE_INFLATION_FACTOR)} = workspace[workspaceOffset + W_CONDITIONING + 2u];
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PRECISION_WARNING)} = select(0.0, 1.0, correlationCondition > ${getWGSLFloatLiteral(GPU_REGRESSION_PRECISION_CONDITION_LIMIT)});
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_PARAMETER_BOUND)} = bound;
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_APPROXIMATION)} = ${options.stochastic ? '1.0' : '0.0'};
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_CONVERGED)} = select(0.0, 1.0, !atBoundary);
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_AT_BOUNDARY)} = select(0.0, 1.0, atBoundary);
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_SERIES_LIMITED)} = workspace[workspaceOffset + W_SERIES_LIMITED];
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_TRUNCATION_BOUND)} = select(
    1.0e38,
    n * pow(scaled, ${options.maximumPower + 1}.0) / (${options.maximumPower + 1}.0 * (1.0 - scaled)),
    scaled < 1.0
  );
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_VARIANCE_DEFINED)} = select(0.0, 1.0, defined);
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOCAL_MAXIMUM_COUNT)} = workspace[workspaceOffset + W_CANDIDATES];
  ${summary(GPU_SPATIAL_MAXIMUM_LIKELIHOOD_SUMMARY_LOG_DETERMINANT)} = logDeterminant;
  statusOut[statusOutOffset] = 0u;
${options.hasConverged ? '  convergedOut[convergedOutOffset] = select(0u, 1u, !atBoundary);' : ''}
}
`;
}
