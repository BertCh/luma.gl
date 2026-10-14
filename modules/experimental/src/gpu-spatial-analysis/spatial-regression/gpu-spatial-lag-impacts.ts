// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, type WGSLKernelBinding} from '../../utils/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../utils/wgsl-common';
import {
  GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_POWER,
  getSpatialMultiplierTotalWGSL,
  getSpatialMultiplierTraceWGSL
} from './spatial-weights-traces-parameters';

const NORM = 'infinityNorm[infinityNormOffset]';
const OPERATION = 'GPUSpatialLagImpacts';

/** Number of float32 values per predictor in `output.impacts`: direct, indirect, total. */
export const GPU_SPATIAL_LAG_IMPACTS_STRIDE = 3;

/** Caller-owned outputs of {@link GPUSpatialLagImpacts}. */
export type GPUSpatialLagImpactsOutput = {
  /**
   * `predictorCount * 3` float32 values: `[direct, indirect, total]` per predictor in coefficient
   * order. NaN for every predictor when the fit failed or `rho` is outside the series range.
   */
  impacts: GraphDataView<'float32'>;
};

/** Properties for {@link GPUSpatialLagImpacts}. */
export type GPUSpatialLagImpactsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'spatial-lag-impacts'`. */
  id?: string;
  /**
   * Fit coefficients, one row per coefficient with the estimate in the first column of each
   * `coefficientStride` values. The `table` of `GPUSpatialTwoStageLeastSquares` (stride 4, rows
   * `[intercept, predictors..., rho]`) can be passed directly.
   */
  coefficients: GraphDataView<'float32'>;
  /** Float32 values per coefficient row; 4 for the 2SLS `table`. */
  coefficientStride: number;
  /** Row of the spatial autoregressive parameter `rho`; `predictorCount + 1` for the 2SLS table. */
  rhoIndex: number;
  /** Row of the first predictor (the intercept is excluded). Defaults to 1. */
  firstPredictorIndex?: number;
  /** Number of predictors whose impacts are written, compile-time. */
  predictorCount: number;
  /** Number of rows `n` of the weights the traces describe. */
  rowCount: number;
  /** Series length `Q` used, at most the length of each traces view. */
  maximumPower: number;
  /**
   * Traces of the weights, the `powerTraces`, `powerSums` and `infinityNorm` outputs of
   * `GPUSpatialWeightsTraces` (the series run in the scaled parameter `rho * infinityNorm`).
   */
  traces: {
    powerTraces: GraphDataView<'float32'>;
    powerSums: GraphDataView<'float32'>;
    infinityNorm: GraphDataView<'float32'>;
  };
  /** Optional fit status code (a model's `output.status.code`); any nonzero value turns every impact into NaN. */
  status?: GraphDataView<'uint32'>;
  /**
   * Largest `|rho| * ||W||_inf` (the scaled series parameter) for which the series is evaluated;
   * larger values give NaN. The series needs it below 1, which is the default and is correct for
   * any `W` (row-standardized weights have `||W||_inf = 1`).
   */
  rhoBound?: number;
  /** Caller-owned outputs. */
  output: GPUSpatialLagImpactsOutput;
};

/**
 * Average direct, indirect and total impacts of the predictors of a spatial lag model
 * `y = rho W y + X beta + e` (LeSage and Pace 2009), from a fit table and the power traces of `W`.
 *
 * In a lag model `beta_k` is not the marginal effect of `x_k`: the effect of `x_k` on `y` is
 * `S_k = beta_k (I - rho W)^-1`. The scalar summaries are
 * - direct: `beta_k tr((I - rho W)^-1) / n` (mean diagonal of `S_k`),
 * - total: `beta_k 1'(I - rho W)^-1 1 / n` (mean row sum of `S_k`),
 * - indirect: total minus direct (the effect of neighbors),
 *
 * with `tr((I - rho W)^-1) = n + sum_{q=1}^{Q} rho^q tr(W^q)` and
 * `1'(I - rho W)^-1 1 = n + sum_{q=1}^{Q} rho^q 1'W^q 1` truncated at `Q = maximumPower` (Horner
 * evaluation). The truncation error is below `n |rho|^(Q+1) / (1 - |rho|)` for a row-standardized
 * `W` (see `getSpatialMultiplierTruncationBound`); choose `Q` so that `|rho|^Q` is negligible.
 * Impacts are NaN when the fit `status` is nonzero, `rho` is not finite or `|rho| >= rhoBound`.
 *
 * Reference: spreg 1.9.1 `GM_Lag(..., spat_impacts='full')` uses the same direct definition
 * (`tr((I - rho W)^-1) / n`) and, for a row-standardized `W`, the same total `1 / (1 - rho)`. For a
 * `W` that is not row-standardized, spreg's total is wrong (it still uses `1 / (1 - rho)`) while
 * the total here uses the exact `1' W^q 1` sums. spreg's `'simple'` method sets the direct
 * multiplier to 1 and is not reproduced; its `'power'` method stops when the first omitted term
 * is below tolerance, which is wrong when odd traces vanish (bipartite lattices), so it is not
 * reproduced either.
 *
 * Standard errors of the impacts (simulation of `(rho, beta)` from the coefficient covariance)
 * are deferred: the 2SLS contributor does not output the covariance matrix today.
 */
export class GPUSpatialLagImpacts implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialLagImpactsProps;

  constructor(props: GPUSpatialLagImpactsProps) {
    const id = props.id ?? 'spatial-lag-impacts';
    this.id = id;
    this.props = props;
    const {predictorCount, maximumPower, coefficientStride, rhoIndex, rowCount} = props;
    const firstPredictorIndex = props.firstPredictorIndex ?? 1;
    validateCapacity(id, 'predictorCount', predictorCount);
    validateCapacity(id, 'rowCount', rowCount);
    validateCapacity(id, 'maximumPower', maximumPower, 1, GPU_SPATIAL_WEIGHTS_TRACES_MAXIMUM_POWER);
    validateCapacity(id, 'coefficientStride', coefficientStride);
    if (!Number.isInteger(rhoIndex) || rhoIndex < 0) {
      throw new Error(`${id} rhoIndex must be a non-negative integer`);
    }
    if (!Number.isInteger(firstPredictorIndex) || firstPredictorIndex < 0) {
      throw new Error(`${id} firstPredictorIndex must be a non-negative integer`);
    }
    if (rhoIndex >= firstPredictorIndex && rhoIndex < firstPredictorIndex + predictorCount) {
      throw new Error(`${id} rhoIndex must not be a predictor row`);
    }
    if (props.rhoBound !== undefined && !(props.rhoBound > 0)) {
      throw new Error(`${id} rhoBound must be positive`);
    }
    validatePackedView(props.coefficients, ['float32'], `${id} coefficients`);
    const lastRow = Math.max(rhoIndex, firstPredictorIndex + predictorCount - 1);
    if (props.coefficients.length < lastRow * coefficientStride + 1) {
      throw new Error(`${id} coefficients must hold every predictor row and the rho row`);
    }
    validatePackedView(props.traces.powerTraces, ['float32'], `${id} traces.powerTraces`);
    validatePackedView(props.traces.powerSums, ['float32'], `${id} traces.powerSums`);
    if (
      props.traces.powerTraces.length < maximumPower ||
      props.traces.powerSums.length < maximumPower
    ) {
      throw new Error(`${id} traces must hold maximumPower float32 values each`);
    }
    validatePackedView(props.traces.infinityNorm, ['float32'], `${id} traces.infinityNorm`);
    if (props.traces.infinityNorm.length < 1) {
      throw new Error(`${id} traces.infinityNorm must hold one float32 value`);
    }
    if (props.status) {
      validatePackedView(props.status, ['uint32'], `${id} status`);
      if (props.status.length < 1) {
        throw new Error(`${id} status must hold one uint32 row`);
      }
    }
    validatePackedView(props.output.impacts, ['float32'], `${id} output.impacts`);
    if (props.output.impacts.length < predictorCount * GPU_SPATIAL_LAG_IMPACTS_STRIDE) {
      throw new Error(`${id} output.impacts must hold predictorCount * 3 float32 values`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.impacts],
      [
        props.coefficients,
        props.traces.powerTraces,
        props.traces.powerSums,
        props.traces.infinityNorm,
        props.status
      ]
    );
  }

  /** Returns the single impacts node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {predictorCount, maximumPower, rowCount, coefficientStride, rhoIndex} = props;
    const firstPredictorIndex = props.firstPredictorIndex ?? 1;
    validateGraphViewsBelongToGraph(id, graph, [
      props.coefficients,
      props.traces.powerTraces,
      props.traces.powerSums,
      props.traces.infinityNorm,
      props.status,
      props.output.impacts
    ]);
    const bindings: WGSLKernelBinding[] = [
      {name: 'coefficients', view: props.coefficients, type: 'f32', access: 'read'},
      {name: 'powerTraces', view: props.traces.powerTraces, type: 'f32', access: 'read'},
      {name: 'powerSums', view: props.traces.powerSums, type: 'f32', access: 'read'},
      {name: 'infinityNorm', view: props.traces.infinityNorm, type: 'f32', access: 'read'},
      ...(props.status
        ? [{name: 'status', view: props.status, type: 'u32', access: 'read'} as WGSLKernelBinding]
        : []),
      {name: 'impacts', view: props.output.impacts, type: 'f32', access: 'read_write'}
    ];
    const rhoBound = Math.fround(props.rhoBound ?? 1);
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-impacts`,
        operation: OPERATION,
        variant: 'impacts',
        bindings,
        invocationCount: predictorCount,
        declarations: `${GET_NAN_WGSL}
${IS_FINITE_VALUE_WGSL}
const ROWS: f32 = ${rowCount}.0;
const STRIDE: u32 = ${coefficientStride}u;
const RHO_ROW: u32 = ${rhoIndex}u;
const FIRST_ROW: u32 = ${firstPredictorIndex}u;
const RHO_BOUND: f32 = ${Number.isInteger(rhoBound) ? `${rhoBound}.0` : rhoBound};
${getSpatialMultiplierTraceWGSL(maximumPower, 'powerTraces', rowCount, NORM)}
${getSpatialMultiplierTotalWGSL(maximumPower, 'powerSums', rowCount, NORM)}`,
        body: `let rho = coefficients[coefficientsOffset + RHO_ROW * STRIDE];
  let beta = coefficients[coefficientsOffset + (FIRST_ROW + index) * STRIDE];
  var usable = isFiniteValue(rho) && abs(rho) * infinityNorm[infinityNormOffset] < RHO_BOUND;
${props.status ? '  usable = usable && status[statusOffset] == 0u;\n' : ''}  var direct = getNaN();
  var total = getNaN();
  if (usable) {
    direct = beta * getSpatialMultiplierTrace(rho) / ROWS;
    total = beta * getSpatialMultiplierTotal(rho) / ROWS;
  }
  impacts[impactsOffset + index * 3u] = direct;
  impacts[impactsOffset + index * 3u + 1u] = total - direct;
  impacts[impactsOffset + index * 3u + 2u] = total;`
      })
    ];
  }
}
