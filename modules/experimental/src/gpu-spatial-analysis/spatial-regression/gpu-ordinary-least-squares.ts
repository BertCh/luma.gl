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
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {
  GPU_ORDINARY_LEAST_SQUARES_PARAMETER_LENGTH,
  GPU_ORDINARY_LEAST_SQUARES_PIVOT_TOLERANCE,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH
} from './ordinary-least-squares-parameters';
import {
  CHI_SQUARE_MAXIMUM_ITERATIONS,
  getLogGammaOfHalfInteger
} from './ordinary-least-squares-statistics';
import {
  getCholeskyWGSL,
  SPATIAL_REGRESSION_MAXIMUM_PREDICTOR_COUNT
} from './spatial-regression-solve';

const OPERATION = 'GPUOrdinaryLeastSquares';
const TILE_WORKGROUP_SIZE = 64;
const MAXIMUM_DEFAULT_TILE_COUNT = 4096;
const MINIMUM_DEFAULT_TILE_ROWS = 64;

/** Caller-owned outputs of {@link GPUOrdinaryLeastSquares}. */
export type GPUOrdinaryLeastSquaresOutput = {
  /** `predictorCount + 1` coefficients, intercept first, then one per predictor column. */
  coefficients: GraphDataView<'float32'>;
  /** Coefficient standard errors, same layout as `coefficients`. */
  standardErrors: GraphDataView<'float32'>;
  /** Coefficient t statistics, same layout as `coefficients`. */
  tStatistics: GraphDataView<'float32'>;
  /**
   * At least 16 float32 values; see the `GPU_ORDINARY_LEAST_SQUARES_SUMMARY_*` slot constants:
   * row count, R2, adjusted R2, sigma squared, log-likelihood, AIC, BIC, Jarque-Bera and p-value,
   * Breusch-Pagan and p-value, RSS, TSS, skewness, kurtosis, ridge lambda.
   */
  summary: GraphDataView<'float32'>;
  /** One uint32: 0 ok, 1 singular or ill-conditioned, 2 too few rows (`n <= predictorCount + 1`). */
  status: GraphDataView<'uint32'>;
  /** Optional per-row residuals `y - fitted`; NaN for excluded rows and for a failed fit. */
  residuals?: GraphDataView<'float32'>;
  /** Optional per-row fitted values; NaN for excluded rows and for a failed fit. */
  fitted?: GraphDataView<'float32'>;
};

/**
 * Properties for {@link GPUOrdinaryLeastSquares}.
 *
 * Per-frame (no recompile): the contents of `parameters` and of every input buffer. Topology (needs
 * a new graph): `predictorCount`, the row count, whether `mask`, `parameters`, `residuals` and
 * `fitted` are present, and `tileRowCount`.
 */
export type GPUOrdinaryLeastSquaresProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'ordinary-least-squares'`. */
  id?: string;
  /** Row-major predictors, `predictors[row * predictorCount + column]`, without an intercept column. */
  predictors: GraphDataView<'float32'>;
  /** Response, one value per row. */
  response: GraphDataView<'float32'>;
  /** Optional packed `uint32` row mask; zero excludes the row. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Optional per-frame parameters: a float32 view of at least 1 element written with
   * `getGPUOrdinaryLeastSquaresParameterValues` (ridge lambda). Omit for a plain OLS fit.
   */
  parameters?: GraphDataView<'float32'>;
  /** Number of predictor columns, compile-time, 1 to 15. The intercept is added internally. */
  predictorCount: number;
  /**
   * Rows reduced by one invocation in the accumulation passes. Defaults to
   * `max(64, ceil(rows / 4096))`. The tile layout fixes the summation order, so results are
   * bitwise reproducible for a given value but differ in the last bits between values.
   */
  tileRowCount?: number;
  /** Caller-owned outputs. */
  output: GPUOrdinaryLeastSquaresOutput;
};

/**
 * Ordinary least squares with an intercept and a regression report: coefficients, standard errors,
 * t statistics, R2, adjusted R2, sigma squared, log-likelihood, AIC, BIC, residual and fitted
 * columns, the Jarque-Bera normality test and the Breusch-Pagan heteroskedasticity test.
 *
 * Rows with a non-finite predictor or response, or with a zero `mask`, are excluded. An optional
 * per-frame ridge penalty `lambda >= 0` shrinks the slopes (never the intercept).
 *
 * Accumulation is deterministic and well-conditioned, with no float atomics:
 * 1. Tile pass: one invocation per tile of `tileRowCount` consecutive rows sums the used rows in
 *    row order into a per-tile partial (plain f32; a tile is at most a few hundred rows, and the
 *    hot loop avoids depending on the compiler preserving compensation code). A merge pass then
 *    sums the partials in tile order with Kahan compensation. This yields the column means and the
 *    row count.
 * 2. A second tile and merge pass accumulates the centered cross-products `X'X`, `X'y`, `y'y`
 *    about those means, which removes the cancellation of the raw-moment formulation. The
 *    residual error left is the rounding of each mean (second order, negligible unless the mean is
 *    above roughly `1e3` standard deviations).
 * 3. One single-invocation kernel scales the centered normal matrix to a correlation matrix (so
 *    columns of different units are comparable), adds the scaled ridge penalty, factors it with
 *    `choleskyFactor_p`, and solves for the slopes. A pivot below
 *    `GPU_ORDINARY_LEAST_SQUARES_PIVOT_TOLERANCE` reports status 1.
 * 4. A per-row pass writes residuals and fitted values from the centered form
 *    `e = (y - ybar) - sum(b (x - xbar))`, and a tile and merge pass sums `e^2`, `e^3`, `e^4`
 *    so the residual sum of squares is accurate even when `R2` is close to 1.
 * 5. A single-invocation finish kernel computes the report. Standard errors are
 *    `sqrt(sigma2 * diag((X'X + lambda I)^-1))` (for a ridge fit they ignore the shrinkage bias),
 *    the intercept variance is `sigma2 (1/n + xbar' (X'X)^-1 xbar)`. The log-likelihood is
 *    Gaussian with `sigma2 = RSS / n`, floored at the smallest normal f32 so an exact fit stays
 *    finite. AIC and BIC count `p = predictorCount + 1` parameters (the GeoDa convention).
 * 6. Jarque-Bera uses the residual moments about zero (the mean is zero with an intercept):
 *    `JB = n/6 (S^2 + (K - 3)^2 / 4)` with p-value `exp(-JB / 2)`. Breusch-Pagan is the Koenker
 *    studentized form: the same accumulation, solve and a small finish kernel run again with
 *    `e^2` as the response, and the statistic is `n R2` of that regression with a chi-square
 *    p-value on `predictorCount` degrees of freedom (regularized upper incomplete gamma with a
 *    fixed iteration cap, mirrored on the CPU by `getChiSquareSurvival`).
 *
 * Inputs must be single packed views. Skipped-row masking, NaN propagation and the status word
 * make a failed fit leave every statistic NaN, never partial garbage.
 */
export class GPUOrdinaryLeastSquares implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUOrdinaryLeastSquaresProps;
  /** Number of rows. */
  readonly rowCount: number;
  /** Rows reduced by one accumulation invocation. */
  readonly tileRowCount: number;
  /** Number of accumulation tiles. */
  readonly tileCount: number;

  constructor(props: GPUOrdinaryLeastSquaresProps) {
    this.id = props.id ?? 'ordinary-least-squares';
    this.props = props;
    const id = this.id;
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
      ['mask', props.mask],
      ['parameters', props.parameters]
    ] as const) {
      if ((view as unknown) instanceof GraphVectorView) {
        throw new Error(`${id} ${name} must be a single packed view, not a chunked vector`);
      }
    }
    validatePackedView(props.predictors, ['float32'], `${id} predictors`);
    validatePackedView(props.response, ['float32'], `${id} response`);
    const rows = props.response.length;
    if (rows < 1) {
      throw new Error(`${id} needs at least one row`);
    }
    if (props.predictors.length !== rows * predictorCount) {
      throw new Error(`${id} predictors length must equal response length * predictorCount`);
    }
    if (props.mask) {
      validatePackedUint32View(props.mask, `${id} mask`);
      if (props.mask.length !== rows) {
        throw new Error(`${id} mask length must equal response length`);
      }
    }
    if (props.parameters) {
      validatePackedView(props.parameters, ['float32'], `${id} parameters`);
      if (props.parameters.length < GPU_ORDINARY_LEAST_SQUARES_PARAMETER_LENGTH) {
        throw new Error(
          `${id} parameters must hold ${GPU_ORDINARY_LEAST_SQUARES_PARAMETER_LENGTH} float32 value`
        );
      }
    }
    const coefficientCount = predictorCount + 1;
    for (const name of ['coefficients', 'standardErrors', 'tStatistics'] as const) {
      validatePackedView(output[name], ['float32'], `${id} output.${name}`);
      if (output[name].length < coefficientCount) {
        throw new Error(`${id} output.${name} must hold predictorCount + 1 rows`);
      }
    }
    validatePackedView(output.summary, ['float32'], `${id} output.summary`);
    if (output.summary.length < GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH) {
      throw new Error(
        `${id} output.summary must hold ${GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH} float32 values`
      );
    }
    validatePackedUint32View(output.status, `${id} output.status`);
    if (output.status.length < 1) {
      throw new Error(`${id} output.status must hold one uint32 row`);
    }
    for (const name of ['residuals', 'fitted'] as const) {
      const view = output[name];
      if (view) {
        validatePackedView(view, ['float32'], `${id} output.${name}`);
        if (view.length < rows) {
          throw new Error(`${id} output.${name} must hold one row per input row`);
        }
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
      [
        output.coefficients,
        output.standardErrors,
        output.tStatistics,
        output.summary,
        output.status,
        output.residuals,
        output.fitted
      ],
      [props.predictors, props.response, props.mask, props.parameters]
    );
  }

  /**
   * Returns the main chain (tile and merge means, tile and merge moments, solve, residuals,
   * residual-moment tile and merge, finish) followed by the Breusch-Pagan chain (the same four
   * accumulation nodes on `e^2`, a solve, a finish), in order.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, rowCount, tileRowCount, tileCount} = this;
    const {output, predictorCount} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.predictors,
      props.response,
      props.mask,
      props.parameters,
      output.coefficients,
      output.standardErrors,
      output.tStatistics,
      output.summary,
      output.status,
      output.residuals,
      output.fitted
    ]);
    const k = predictorCount;
    const layout = getWorkspaceLayout(k);
    const u32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', length);
    const f32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'float32', length);
    const residualView = output.residuals ?? f32('residuals', rowCount);
    const squaredView = f32('squared-residuals', rowCount);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const accumulationProps = {
      id,
      operation: OPERATION,
      predictors: props.predictors,
      mask: props.mask,
      rowCount,
      k,
      tileRowCount,
      tileCount
    };

    const mainWorkspace = u32('workspace', layout.length);
    nodes.push(
      ...getAccumulationNodes(graph, {
        ...accumulationProps,
        prefix: id,
        response: props.response,
        workspace: mainWorkspace,
        partialMeans: u32('partial-means', tileCount * (k + 2)),
        partialMoments: u32('partial-moments', tileCount * layout.momentCount)
      })
    );
    nodes.push(
      createKernel(graph, {
        id: `${id}-solve`,
        variant: 'solve',
        bindings: [
          ...(props.parameters ? [read('params', props.parameters, 'f32')] : []),
          writeWorkspace(mainWorkspace)
        ],
        invocationCount: 1,
        declarations: `${getSharedWGSL(layout, true)}
${getCholeskyWGSL(k)}`,
        body: getSolveBody(k, Boolean(props.parameters))
      })
    );

    nodes.push(
      createKernel(graph, {
        id: `${id}-residuals`,
        variant: 'residuals',
        bindings: [
          read('design', props.predictors, 'f32'),
          read('response', props.response, 'f32'),
          ...(props.mask ? [read('rowMask', props.mask, 'u32')] : []),
          readWorkspace(mainWorkspace),
          write('residualOut', residualView, 'f32'),
          ...(output.fitted ? [write('fittedOut', output.fitted, 'f32')] : []),
          write('squaredOut', squaredView, 'f32')
        ],
        invocationCount: rowCount,
        declarations: `${getSharedWGSL(layout, false)}
${getRowWGSL(rowCount, k, tileRowCount, Boolean(props.mask))}`,
        body: `let nan = getNaN();
  residualOut[residualOutOffset + index] = nan;
  ${output.fitted ? 'fittedOut[fittedOutOffset + index] = nan;' : ''}
  squaredOut[squaredOutOffset + index] = nan;
  if (ws[wsOffset + W_STATUS] != 0u || !isRowValid(index)) {
    return;
  }
  var fit = 0.0;
  for (var j = 0u; j < K; j++) {
    fit = fit + wsGet(W_SLOPES + j) * (design[designOffset + index * K + j] - wsGet(W_MEANS + j));
  }
  let meanResponse = wsGet(W_MEANS + K);
  let residual = (response[responseOffset + index] - meanResponse) - fit;
  residualOut[residualOutOffset + index] = residual;
  ${output.fitted ? 'fittedOut[fittedOutOffset + index] = meanResponse + fit;' : ''}
  squaredOut[squaredOutOffset + index] = residual * residual;`
      })
    );

    const partialResiduals = u32('partial-residuals', tileCount * RESIDUAL_SUM_COUNT);
    nodes.push(
      createKernel(graph, {
        id: `${id}-residual-tiles`,
        variant: 'residual-tiles',
        workgroupSize: TILE_WORKGROUP_SIZE,
        bindings: [
          read('residualIn', residualView, 'f32'),
          write('partial', partialResiduals, 'u32')
        ],
        invocationCount: tileCount,
        declarations: `const ROW_COUNT: u32 = ${rowCount}u;
const TILE_ROWS: u32 = ${tileRowCount}u;
${FINITE_WGSL}`,
        body: `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  var sumSquares = 0.0;
  var sumCubes = 0.0;
  var sumFourth = 0.0;
  for (var row = firstRow; row < lastRow; row++) {
    let e = residualIn[residualInOffset + row];
    if (!isFiniteValue(e)) {
      continue;
    }
    let e2 = e * e;
    sumSquares = sumSquares + e2;
    sumCubes = sumCubes + e2 * e;
    sumFourth = sumFourth + e2 * e2;
  }
  let base = partialOffset + index * ${RESIDUAL_SUM_COUNT}u;
  partial[base] = bitcast<u32>(sumSquares);
  partial[base + 1u] = bitcast<u32>(sumCubes);
  partial[base + 2u] = bitcast<u32>(sumFourth);`
      })
    );
    nodes.push(
      createKernel(graph, {
        id: `${id}-residual-merge`,
        variant: 'residual-merge',
        bindings: [read('partial', partialResiduals, 'u32'), writeWorkspace(mainWorkspace)],
        invocationCount: RESIDUAL_SUM_COUNT,
        declarations: `${getSharedWGSL(layout, true)}
const TILE_COUNT: u32 = ${tileCount}u;`,
        body: `var sum = 0.0;
  var compensation = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    let term = bitcast<f32>(partial[partialOffset + tile * ${RESIDUAL_SUM_COUNT}u + index]) - compensation;
    let next = sum + term;
    compensation = (next - sum) - term;
    sum = next;
  }
  wsSet(W_RESIDUAL_SUMS + index, sum);`
      })
    );

    nodes.push(
      createKernel(graph, {
        id: `${id}-finish`,
        variant: 'finish',
        bindings: [
          readWorkspace(mainWorkspace),
          write('coefficientsOut', output.coefficients, 'f32'),
          write('standardErrorsOut', output.standardErrors, 'f32'),
          write('tStatisticsOut', output.tStatistics, 'f32'),
          write('summaryOut', output.summary, 'f32'),
          write('statusOut', output.status, 'u32')
        ],
        invocationCount: 1,
        declarations: `${getSharedWGSL(layout, false)}
${getCholeskyWGSL(k)}`,
        body: getFinishBody(k)
      })
    );

    // Breusch-Pagan: regress e^2 on the same predictors with no ridge, then n R2.
    const bpWorkspace = u32('bp-workspace', layout.length);
    nodes.push(
      ...getAccumulationNodes(graph, {
        ...accumulationProps,
        prefix: `${id}-bp`,
        response: squaredView,
        workspace: bpWorkspace,
        partialMeans: u32('bp-partial-means', tileCount * (k + 2)),
        partialMoments: u32('bp-partial-moments', tileCount * layout.momentCount)
      })
    );
    nodes.push(
      createKernel(graph, {
        id: `${id}-bp-solve`,
        variant: 'solve',
        bindings: [writeWorkspace(bpWorkspace)],
        invocationCount: 1,
        declarations: `${getSharedWGSL(layout, true)}
${getCholeskyWGSL(k)}`,
        body: getSolveBody(k, false)
      })
    );
    nodes.push(
      createKernel(graph, {
        id: `${id}-bp-finish`,
        variant: 'breusch-pagan',
        bindings: [readWorkspace(bpWorkspace), write('summaryOut', output.summary, 'f32')],
        invocationCount: 1,
        declarations: `${getSharedWGSL(layout, false)}
${getChiSquareWGSL(k)}`,
        body: getBreuschPaganBody(k)
      })
    );
    return nodes;
  }
}

const RESIDUAL_SUM_COUNT = 3;

type WorkspaceLayout = {
  /** Number of packed upper-triangular centered moments of `[predictors, response]`. */
  momentCount: number;
  means: number;
  count: number;
  status: number;
  lambda: number;
  residualSums: number;
  scale: number;
  slopes: number;
  factor: number;
  moments: number;
  /** Workspace length in 32-bit words. */
  length: number;
};

/**
 * Word layout of one chain's `uint32` workspace (float32 values are stored with bitcast so counts
 * and statuses survive denormal handling):
 * means (k + 1), count, status, lambda, residual sums (3), scale (k), slopes (k), factor (k*k),
 * packed upper-triangular centered moments (d (d + 1) / 2 with d = k + 1).
 */
function getWorkspaceLayout(k: number): WorkspaceLayout {
  const d = k + 1;
  const momentCount = (d * (d + 1)) / 2;
  const means = 0;
  const count = means + d;
  const status = count + 1;
  const lambda = status + 1;
  const residualSums = lambda + 1;
  const scale = residualSums + RESIDUAL_SUM_COUNT;
  const slopes = scale + k;
  const factor = slopes + k;
  const moments = factor + k * k;
  return {
    momentCount,
    means,
    count,
    status,
    lambda,
    residualSums,
    scale,
    slopes,
    factor,
    moments,
    length: moments + momentCount
  };
}

const FINITE_WGSL = /* wgsl */ `
fn getNaN() -> f32 {
  var bits = 0x7fc00000u;
  return bitcast<f32>(bits);
}

fn isFiniteValue(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}
`;

/** Layout constants, workspace accessors, and the packed moment index. */
function getSharedWGSL(layout: WorkspaceLayout, writable: boolean): string {
  const k = layout.slopes - layout.scale;
  const d = k + 1;
  return /* wgsl */ `
const K: u32 = ${k}u;
const D: u32 = ${d}u;
const KK: u32 = ${k * k}u;
const W_MEANS: u32 = ${layout.means}u;
const W_COUNT: u32 = ${layout.count}u;
const W_STATUS: u32 = ${layout.status}u;
const W_LAMBDA: u32 = ${layout.lambda}u;
const W_RESIDUAL_SUMS: u32 = ${layout.residualSums}u;
const W_SCALE: u32 = ${layout.scale}u;
const W_SLOPES: u32 = ${layout.slopes}u;
const W_FACTOR: u32 = ${layout.factor}u;
const W_MOMENTS: u32 = ${layout.moments}u;
${FINITE_WGSL}
fn wsGet(slot: u32) -> f32 {
  return bitcast<f32>(ws[wsOffset + slot]);
}
${
  writable
    ? `fn wsSet(slot: u32, value: f32) {
  ws[wsOffset + slot] = bitcast<u32>(value);
}`
    : ''
}
fn momentAt(first: u32, second: u32) -> f32 {
  let lo = min(first, second);
  let hi = max(first, second);
  return wsGet(W_MOMENTS + lo * (2u * D - lo + 1u) / 2u + (hi - lo));
}
`;
}

/** Row access: validity (mask and finiteness) for the `design`, `response` and optional `rowMask` bindings. */
function getRowWGSL(rowCount: number, k: number, tileRowCount: number, hasMask: boolean): string {
  return /* wgsl */ `
const ROW_COUNT: u32 = ${rowCount}u;
const TILE_ROWS: u32 = ${tileRowCount}u;
const ROW_K: u32 = ${k}u;

fn isRowValid(row: u32) -> bool {
  ${hasMask ? 'if (rowMask[rowMaskOffset + row] == 0u) {\n    return false;\n  }' : ''}
  if (!isFiniteValue(response[responseOffset + row])) {
    return false;
  }
  for (var j = 0u; j < ROW_K; j++) {
    if (!isFiniteValue(design[designOffset + row * ROW_K + j])) {
      return false;
    }
  }
  return true;
}
`;
}

type AccumulationProps = {
  id: string;
  operation: string;
  prefix: string;
  predictors: GraphDataView<'float32'>;
  response: GraphDataView<'float32'>;
  mask?: GraphDataView<'uint32'>;
  rowCount: number;
  k: number;
  tileRowCount: number;
  tileCount: number;
  workspace: GraphDataView<'uint32'>;
  partialMeans: GraphDataView<'uint32'>;
  partialMoments: GraphDataView<'uint32'>;
};

/** Tile and merge passes for means plus count, then for centered moments. */
function getAccumulationNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: AccumulationProps
): GPUCommandNode<Parameters>[] {
  const {prefix, k, rowCount, tileRowCount, tileCount, workspace} = props;
  const d = k + 1;
  const layout = getWorkspaceLayout(k);
  const momentCount = layout.momentCount;
  const hasMask = Boolean(props.mask);
  const rowBindings = [
    read('design', props.predictors, 'f32'),
    read('response', props.response, 'f32'),
    ...(props.mask ? [read('rowMask', props.mask, 'u32')] : [])
  ];
  const rowDeclarations = `${getSharedWGSL(layout, false)}
${getRowWGSL(rowCount, k, tileRowCount, hasMask)}`;
  const mergeDeclarations = `${getSharedWGSL(layout, true)}
const TILE_COUNT: u32 = ${tileCount}u;`;
  const mergeLoop = (stride: string) => `var sum = 0.0;
  var compensation = 0.0;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    let term = bitcast<f32>(partial[partialOffset + tile * ${stride} + index]) - compensation;
    let next = sum + term;
    compensation = (next - sum) - term;
    sum = next;
  }`;
  return [
    createKernel(graph, {
      id: `${prefix}-means-tiles`,
      variant: 'means-tiles',
      workgroupSize: TILE_WORKGROUP_SIZE,
      bindings: [...rowBindings, write('partial', props.partialMeans, 'u32')],
      invocationCount: tileCount,
      declarations: `${FINITE_WGSL}
${getRowWGSL(rowCount, k, tileRowCount, hasMask)}`,
      body: `let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  var sums: array<f32, ${d}>;
  var count = 0u;
  for (var row = firstRow; row < lastRow; row++) {
    if (!isRowValid(row)) {
      continue;
    }
    count = count + 1u;
    for (var j = 0u; j < ROW_K; j++) {
      sums[j] = sums[j] + design[designOffset + row * ROW_K + j];
    }
    sums[ROW_K] = sums[ROW_K] + response[responseOffset + row];
  }
  let base = partialOffset + index * ${d + 1}u;
  for (var column = 0u; column < ${d}u; column++) {
    partial[base + column] = bitcast<u32>(sums[column]);
  }
  partial[base + ${d}u] = count;`
    }),
    createKernel(graph, {
      id: `${prefix}-means-merge`,
      variant: 'means-merge',
      bindings: [read('partial', props.partialMeans, 'u32'), writeWorkspace(workspace)],
      invocationCount: d,
      declarations: mergeDeclarations,
      body: `var rows = 0u;
  for (var tile = 0u; tile < TILE_COUNT; tile++) {
    rows = rows + partial[partialOffset + tile * ${d + 1}u + ${d}u];
  }
  ${mergeLoop(`${d + 1}u`)}
  wsSet(W_MEANS + index, select(0.0, sum / f32(rows), rows > 0u));
  if (index == 0u) {
    ws[wsOffset + W_COUNT] = rows;
  }`
    }),
    createKernel(graph, {
      id: `${prefix}-moments-tiles`,
      variant: 'moments-tiles',
      workgroupSize: TILE_WORKGROUP_SIZE,
      bindings: [
        ...rowBindings,
        readWorkspace(workspace),
        write('partial', props.partialMoments, 'u32')
      ],
      invocationCount: tileCount,
      declarations: rowDeclarations,
      body: `var means: array<f32, ${d}>;
  for (var column = 0u; column < D; column++) {
    means[column] = wsGet(W_MEANS + column);
  }
  let firstRow = index * TILE_ROWS;
  let lastRow = min(firstRow + TILE_ROWS, ROW_COUNT);
  var accumulator: array<f32, ${momentCount}>;
  var centered: array<f32, ${d}>;
  for (var row = firstRow; row < lastRow; row++) {
    if (!isRowValid(row)) {
      continue;
    }
    for (var j = 0u; j < ROW_K; j++) {
      centered[j] = design[designOffset + row * ROW_K + j] - means[j];
    }
    centered[ROW_K] = response[responseOffset + row] - means[ROW_K];
    var slot = 0u;
    for (var first = 0u; first < D; first++) {
      for (var second = first; second < D; second++) {
        accumulator[slot] = accumulator[slot] + centered[first] * centered[second];
        slot = slot + 1u;
      }
    }
  }
  for (var slot = 0u; slot < ${momentCount}u; slot++) {
    partial[partialOffset + index * ${momentCount}u + slot] = bitcast<u32>(accumulator[slot]);
  }`
    }),
    createKernel(graph, {
      id: `${prefix}-moments-merge`,
      variant: 'moments-merge',
      bindings: [read('partial', props.partialMoments, 'u32'), writeWorkspace(workspace)],
      invocationCount: momentCount,
      declarations: mergeDeclarations,
      body: `${mergeLoop(`${momentCount}u`)}
  wsSet(W_MOMENTS + index, sum);`
    })
  ];
}

/** Single-invocation solve of the (optionally ridge-penalized) scaled normal equations. */
function getSolveBody(k: number, hasRidge: boolean): string {
  return /* wgsl */ `let rows = ws[wsOffset + W_COUNT];
  ${
    hasRidge
      ? `let requested = params[paramsOffset];
  let lambda = select(0.0, requested, requested > 0.0 && isFiniteValue(requested));`
      : 'let lambda = 0.0;'
  }
  wsSet(W_LAMBDA, lambda);
  if (rows <= K + 1u) {
    ws[wsOffset + W_STATUS] = 2u;
    return;
  }
  var a: array<f32, ${k * k}>;
  var rhs: array<f32, ${k}>;
  var scale: array<f32, ${k}>;
  for (var i = 0u; i < K; i++) {
    let variance = momentAt(i, i);
    if (!(variance > 0.0) || !isFiniteValue(variance)) {
      ws[wsOffset + W_STATUS] = 1u;
      return;
    }
    scale[i] = sqrt(variance);
  }
  for (var i = 0u; i < K; i++) {
    for (var j = 0u; j <= i; j++) {
      var entry = momentAt(i, j) / (scale[i] * scale[j]);
      if (i == j) {
        entry = entry + lambda / (scale[i] * scale[i]);
      }
      a[i * K + j] = entry;
    }
    rhs[i] = momentAt(i, K) / scale[i];
  }
  if (!choleskyFactor_${k}(&a)) {
    ws[wsOffset + W_STATUS] = 1u;
    return;
  }
  for (var i = 0u; i < K; i++) {
    if (!(a[i * K + i] >= ${getWGSLFloatLiteral(GPU_ORDINARY_LEAST_SQUARES_PIVOT_TOLERANCE)})) {
      ws[wsOffset + W_STATUS] = 1u;
      return;
    }
  }
  for (var i = 0u; i < KK; i++) {
    wsSet(W_FACTOR + i, a[i]);
  }
  choleskySolve_${k}(&a, &rhs);
  for (var i = 0u; i < K; i++) {
    wsSet(W_SCALE + i, scale[i]);
    wsSet(W_SLOPES + i, rhs[i] / scale[i]);
  }
  ws[wsOffset + W_STATUS] = 0u;`;
}

function getFinishBody(k: number): string {
  const p = k + 1;
  return /* wgsl */ `let rows = ws[wsOffset + W_COUNT];
  let status = ws[wsOffset + W_STATUS];
  let nan = getNaN();
  statusOut[statusOutOffset] = status;
  if (status != 0u) {
    for (var i = 0u; i < ${p}u; i++) {
      coefficientsOut[coefficientsOutOffset + i] = nan;
      standardErrorsOut[standardErrorsOutOffset + i] = nan;
      tStatisticsOut[tStatisticsOutOffset + i] = nan;
    }
    for (var slot = 0u; slot < ${GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH}u; slot++) {
      summaryOut[summaryOutOffset + slot] = nan;
    }
    summaryOut[summaryOutOffset] = f32(rows);
    return;
  }
  let n = f32(rows);
  let freedom = n - ${p}.0;
  let residualSumSquares = wsGet(W_RESIDUAL_SUMS);
  let totalSumSquares = momentAt(K, K);
  var a: array<f32, ${k * k}>;
  for (var i = 0u; i < KK; i++) {
    a[i] = wsGet(W_FACTOR + i);
  }
  let sigmaSquared = residualSumSquares / freedom;
  var intercept = wsGet(W_MEANS + K);
  var u: array<f32, ${k}>;
  for (var i = 0u; i < K; i++) {
    intercept = intercept - wsGet(W_SLOPES + i) * wsGet(W_MEANS + i);
    u[i] = wsGet(W_MEANS + i) / wsGet(W_SCALE + i);
  }
  coefficientsOut[coefficientsOutOffset] = intercept;
  for (var i = 0u; i < K; i++) {
    coefficientsOut[coefficientsOutOffset + 1u + i] = wsGet(W_SLOPES + i);
  }
  // Intercept variance: sigma2 (1 / n + u' R^-1 u) with u = xbar / scale.
  choleskySolve_${k}(&a, &u);
  var quadratic = 0.0;
  for (var i = 0u; i < K; i++) {
    quadratic = quadratic + (wsGet(W_MEANS + i) / wsGet(W_SCALE + i)) * u[i];
  }
  standardErrorsOut[standardErrorsOutOffset] = sqrt(max(sigmaSquared * (1.0 / n + quadratic), 0.0));
  for (var i = 0u; i < K; i++) {
    let scale = wsGet(W_SCALE + i);
    let inverse = choleskyInverseDiagonal_${k}(&a, i) / (scale * scale);
    standardErrorsOut[standardErrorsOutOffset + 1u + i] = sqrt(max(sigmaSquared * inverse, 0.0));
  }
  for (var i = 0u; i < ${p}u; i++) {
    let standardError = standardErrorsOut[standardErrorsOutOffset + i];
    tStatisticsOut[tStatisticsOutOffset + i] = select(
      nan,
      coefficientsOut[coefficientsOutOffset + i] / standardError,
      standardError > 0.0
    );
  }
  let rSquared = select(nan, 1.0 - residualSumSquares / totalSumSquares, totalSumSquares > 0.0);
  let meanSquare = max(residualSumSquares / n, 1.17549435e-38);
  let logLikelihood = -0.5 * n * (1.8378770664093453 + log(meanSquare) + 1.0);
  var jarqueBera = nan;
  var jarqueBeraP = nan;
  var skewness = nan;
  var kurtosis = nan;
  let m2 = residualSumSquares / n;
  if (m2 > 0.0) {
    let m3 = wsGet(W_RESIDUAL_SUMS + 1u) / n;
    let m4 = wsGet(W_RESIDUAL_SUMS + 2u) / n;
    skewness = m3 / (m2 * sqrt(m2));
    kurtosis = m4 / (m2 * m2);
    let excess = kurtosis - 3.0;
    jarqueBera = n / 6.0 * (skewness * skewness + 0.25 * excess * excess);
    jarqueBeraP = exp(-0.5 * jarqueBera);
  }
  for (var slot = 0u; slot < ${GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH}u; slot++) {
    summaryOut[summaryOutOffset + slot] = nan;
  }
  summaryOut[summaryOutOffset + 0u] = n;
  summaryOut[summaryOutOffset + 1u] = rSquared;
  summaryOut[summaryOutOffset + 2u] = 1.0 - (1.0 - rSquared) * (n - 1.0) / freedom;
  summaryOut[summaryOutOffset + 3u] = sigmaSquared;
  summaryOut[summaryOutOffset + 4u] = logLikelihood;
  summaryOut[summaryOutOffset + 5u] = 2.0 * ${p}.0 - 2.0 * logLikelihood;
  summaryOut[summaryOutOffset + 6u] = ${p}.0 * log(n) - 2.0 * logLikelihood;
  summaryOut[summaryOutOffset + 7u] = jarqueBera;
  summaryOut[summaryOutOffset + 8u] = jarqueBeraP;
  summaryOut[summaryOutOffset + 11u] = residualSumSquares;
  summaryOut[summaryOutOffset + 12u] = totalSumSquares;
  summaryOut[summaryOutOffset + 13u] = skewness;
  summaryOut[summaryOutOffset + 14u] = kurtosis;
  summaryOut[summaryOutOffset + 15u] = wsGet(W_LAMBDA);`;
}

/** Regularized upper incomplete gamma for the chi-square survival function, fixed iteration cap. */
function getChiSquareWGSL(degreesOfFreedom: number): string {
  return /* wgsl */ `
const DEGREES_OF_FREEDOM: u32 = ${degreesOfFreedom}u;
const LOG_GAMMA: f32 = ${getWGSLFloatLiteral(getLogGammaOfHalfInteger(degreesOfFreedom))};

fn getChiSquareSurvival(value: f32) -> f32 {
  if (!(value > 0.0)) {
    return 1.0;
  }
  let a = f32(DEGREES_OF_FREEDOM) * 0.5;
  let x = value * 0.5;
  let logPrefactor = -x + a * log(x) - LOG_GAMMA;
  if (x < a + 1.0) {
    var term = 1.0 / a;
    var sum = term;
    for (var iteration = 1u; iteration <= ${CHI_SQUARE_MAXIMUM_ITERATIONS}u; iteration++) {
      term = term * x / (a + f32(iteration));
      sum = sum + term;
      if (term < sum * 1.0e-7) {
        break;
      }
    }
    return clamp(1.0 - sum * exp(logPrefactor), 0.0, 1.0);
  }
  let tiny = 1.0e-30;
  var b = x + 1.0 - a;
  var c = 1.0 / tiny;
  var d = 1.0 / b;
  var h = d;
  for (var iteration = 1u; iteration <= ${CHI_SQUARE_MAXIMUM_ITERATIONS}u; iteration++) {
    let fi = f32(iteration);
    let an = -fi * (fi - a);
    b = b + 2.0;
    d = an * d + b;
    if (abs(d) < tiny) {
      d = tiny;
    }
    c = b + an / c;
    if (abs(c) < tiny) {
      c = tiny;
    }
    d = 1.0 / d;
    let delta = d * c;
    h = h * delta;
    if (abs(delta - 1.0) < 1.0e-7) {
      break;
    }
  }
  return clamp(exp(logPrefactor) * h, 0.0, 1.0);
}
`;
}

function getBreuschPaganBody(k: number): string {
  return /* wgsl */ `let status = ws[wsOffset + W_STATUS];
  let nan = getNaN();
  var statistic = nan;
  var pValue = nan;
  let totalSumSquares = momentAt(K, K);
  if (status == 0u && totalSumSquares > 0.0) {
    var explained = 0.0;
    for (var i = 0u; i < ${k}u; i++) {
      explained = explained + wsGet(W_SLOPES + i) * momentAt(i, K);
    }
    let rSquared = clamp(explained / totalSumSquares, 0.0, 1.0);
    statistic = f32(ws[wsOffset + W_COUNT]) * rSquared;
    pValue = getChiSquareSurvival(statistic);
  }
  summaryOut[summaryOutOffset + 9u] = statistic;
  summaryOut[summaryOutOffset + 10u] = pValue;`;
}

function read(name: string, view: GraphDataView, type: 'u32' | 'f32'): WGSLKernelBinding {
  return {name, view, type, access: 'read'};
}

function write(name: string, view: GraphDataView, type: 'u32' | 'f32'): WGSLKernelBinding {
  return {name, view, type, access: 'read_write'};
}

function readWorkspace(view: GraphDataView<'uint32'>): WGSLKernelBinding {
  return read('ws', view, 'u32');
}

function writeWorkspace(view: GraphDataView<'uint32'>): WGSLKernelBinding {
  return write('ws', view, 'u32');
}

type KernelProps = {
  id: string;
  variant: string;
  bindings: readonly WGSLKernelBinding[];
  invocationCount: number;
  declarations?: string;
  body: string;
  workgroupSize?: number;
};

function createKernel<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: KernelProps
): GPUCommandNode<Parameters> {
  return createWGSLKernelNode<Parameters>(graph, {
    operation: OPERATION,
    ...props
  });
}
