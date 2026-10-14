// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';
import {GPULocalMoran} from '../spatial-autocorrelation/index';
import {
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_LM_ERROR,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_LM_LAG,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_ROBUST_LM_ERROR,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_ROBUST_LM_LAG,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_STRIDE,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH,
  GPUGeographicallyWeightedRegression,
  GPUOrdinaryLeastSquares,
  GPUSpatialRegressionDiagnostics
} from '../spatial-regression/index';
import type {GPUSpatialWeights} from '../spatial-weights/index';
import {
  assertRecipe,
  getOrCreateView,
  RecipeBuilder,
  type GPURecipeOverrides,
  type GPURecipeResult
} from './recipe-utils';

/** Specification search outcome: no spatial dependence detected, OLS is adequate. */
export const GPU_SPATIAL_REGRESSION_SPECIFICATION_OLS = 0;
/** Specification search outcome: estimate a spatial lag model. */
export const GPU_SPATIAL_REGRESSION_SPECIFICATION_SPATIAL_LAG = 1;
/** Specification search outcome: estimate a spatial error model. */
export const GPU_SPATIAL_REGRESSION_SPECIFICATION_SPATIAL_ERROR = 2;
/** Specification search outcome: both robust tests are significant (SARMA or a richer model). */
export const GPU_SPATIAL_REGRESSION_SPECIFICATION_LAG_AND_ERROR = 3;
/** Specification search outcome: the diagnostics failed or a needed p-value is NaN. */
export const GPU_SPATIAL_REGRESSION_SPECIFICATION_UNDETERMINED = 4;

/**
 * Anselin's (2005) specification search rule, as a pure function. `tests` holds the six
 * `[statistic, degrees of freedom, p-value]` rows of `GPUSpatialRegressionDiagnostics`; the GPU
 * node of the recipe evaluates exactly this rule. With significance level `alpha` and
 * "significant" meaning `p <= alpha`:
 *
 * 1. If `status` is not 0 or a needed p-value is NaN: `UNDETERMINED` (4). A failed diagnostic is
 *    never reported as "OLS adequate".
 * 2. Neither LM-lag nor LM-error significant: `OLS` (0).
 * 3. Only LM-lag significant: `SPATIAL_LAG` (1). Only LM-error significant: `SPATIAL_ERROR` (2).
 * 4. Both significant: look at the robust tests. Only robust LM-lag significant: lag (1). Only
 *    robust LM-error significant: error (2). Both significant: `LAG_AND_ERROR` (3), where Anselin
 *    advises taking the model with the larger robust statistic or a SARMA specification. Neither
 *    significant: the model whose classic LM statistic is larger (lag on a tie).
 *
 * Source: L. Anselin (2005), "Exploring Spatial Data with GeoDa: A Workbook", section on the
 * spatial regression decision process (the decision tree from the LM tests).
 */
export function getSpatialRegressionSpecification(
  tests: ArrayLike<number>,
  status: number,
  significanceLevel = 0.05
): number {
  const stride = GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_STRIDE;
  const statistic = (row: number) => tests[row * stride];
  const pValue = (row: number) => tests[row * stride + 2];
  const lag = GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_LM_LAG;
  const error = GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_LM_ERROR;
  const robustLag = GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_ROBUST_LM_LAG;
  const robustError = GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_ROBUST_LM_ERROR;
  if (status !== 0 || Number.isNaN(pValue(lag)) || Number.isNaN(pValue(error))) {
    return GPU_SPATIAL_REGRESSION_SPECIFICATION_UNDETERMINED;
  }
  const lagSignificant = pValue(lag) <= significanceLevel;
  const errorSignificant = pValue(error) <= significanceLevel;
  if (!lagSignificant && !errorSignificant) {
    return GPU_SPATIAL_REGRESSION_SPECIFICATION_OLS;
  }
  if (lagSignificant && !errorSignificant) {
    return GPU_SPATIAL_REGRESSION_SPECIFICATION_SPATIAL_LAG;
  }
  if (!lagSignificant && errorSignificant) {
    return GPU_SPATIAL_REGRESSION_SPECIFICATION_SPATIAL_ERROR;
  }
  if (Number.isNaN(pValue(robustLag)) || Number.isNaN(pValue(robustError))) {
    return GPU_SPATIAL_REGRESSION_SPECIFICATION_UNDETERMINED;
  }
  const robustLagSignificant = pValue(robustLag) <= significanceLevel;
  const robustErrorSignificant = pValue(robustError) <= significanceLevel;
  if (robustLagSignificant && robustErrorSignificant) {
    return GPU_SPATIAL_REGRESSION_SPECIFICATION_LAG_AND_ERROR;
  }
  if (robustLagSignificant) {
    return GPU_SPATIAL_REGRESSION_SPECIFICATION_SPATIAL_LAG;
  }
  if (robustErrorSignificant) {
    return GPU_SPATIAL_REGRESSION_SPECIFICATION_SPATIAL_ERROR;
  }
  return statistic(lag) >= statistic(error)
    ? GPU_SPATIAL_REGRESSION_SPECIFICATION_SPATIAL_LAG
    : GPU_SPATIAL_REGRESSION_SPECIFICATION_SPATIAL_ERROR;
}

/** Single-invocation GPU evaluation of {@link getSpatialRegressionSpecification}. @internal */
class SpatialRegressionSpecificationSearch implements GPUCommandNodeProducer<never> {
  constructor(
    readonly props: {
      id: string;
      tests: GraphDataView<'float32'>;
      status: GraphDataView<'uint32'>;
      significanceLevel: number;
      decision: GraphDataView<'uint32'>;
    }
  ) {}

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props} = this;
    const stride = GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_STRIDE;
    const row = (name: string, test: number) =>
      `let ${name}Statistic = tests[testsOffset + ${test * stride}u];
  let ${name}P = tests[testsOffset + ${test * stride + 2}u];`;
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${props.id}-specification`,
        operation: 'GPUSpatialRegressionSpecificationSearch',
        bindings: [
          {name: 'tests', view: props.tests, type: 'f32', access: 'read'},
          {name: 'diagnosticsStatus', view: props.status, type: 'u32', access: 'read'},
          {name: 'decisionOut', view: props.decision, type: 'u32', access: 'read_write'}
        ],
        invocationCount: 1,
        declarations: `const ALPHA: f32 = ${getWGSLFloatLiteral(props.significanceLevel)};
fn isNaNValue(value: f32) -> bool { return (bitcast<u32>(value) & 0x7fffffffu) > 0x7f800000u; }`,
        body: `${row('lag', GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_LM_LAG)}
  ${row('error', GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_LM_ERROR)}
  ${row('robustLag', GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_ROBUST_LM_LAG)}
  ${row('robustError', GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_ROBUST_LM_ERROR)}
  var decision = ${GPU_SPATIAL_REGRESSION_SPECIFICATION_UNDETERMINED}u;
  if (diagnosticsStatus[diagnosticsStatusOffset] == 0u && !isNaNValue(lagP) && !isNaNValue(errorP)) {
    let lagSignificant = lagP <= ALPHA;
    let errorSignificant = errorP <= ALPHA;
    if (!lagSignificant && !errorSignificant) {
      decision = ${GPU_SPATIAL_REGRESSION_SPECIFICATION_OLS}u;
    } else if (lagSignificant && !errorSignificant) {
      decision = ${GPU_SPATIAL_REGRESSION_SPECIFICATION_SPATIAL_LAG}u;
    } else if (!lagSignificant && errorSignificant) {
      decision = ${GPU_SPATIAL_REGRESSION_SPECIFICATION_SPATIAL_ERROR}u;
    } else if (!isNaNValue(robustLagP) && !isNaNValue(robustErrorP)) {
      let robustLagSignificant = robustLagP <= ALPHA;
      let robustErrorSignificant = robustErrorP <= ALPHA;
      if (robustLagSignificant && robustErrorSignificant) {
        decision = ${GPU_SPATIAL_REGRESSION_SPECIFICATION_LAG_AND_ERROR}u;
      } else if (robustLagSignificant) {
        decision = ${GPU_SPATIAL_REGRESSION_SPECIFICATION_SPATIAL_LAG}u;
      } else if (robustErrorSignificant) {
        decision = ${GPU_SPATIAL_REGRESSION_SPECIFICATION_SPATIAL_ERROR}u;
      } else if (lagStatistic >= errorStatistic) {
        decision = ${GPU_SPATIAL_REGRESSION_SPECIFICATION_SPATIAL_LAG}u;
      } else {
        decision = ${GPU_SPATIAL_REGRESSION_SPECIFICATION_SPATIAL_ERROR}u;
      }
    }
  }
  decisionOut[decisionOutOffset] = decision;`
      })
    ];
  }
}

/** Optional geographically weighted fits added to the regression recipe. */
export type GPUSpatialRegressionLocalFitOptions = {
  /** Planar positions, one per row, in the units of the bandwidths. */
  positions: GraphDataView<'float32x2'>;
  /** `getGPUGeographicallyWeightedRegressionParameterValues` view. */
  parameters: GraphDataView<'float32'>;
  /** Compile-time ladder capacity (default of the contributor: 32). */
  maximumBandwidthCount?: number;
  /** Compile-time adaptive neighbor bound (default of the contributor: 128). */
  maximumNeighborCount?: number;
  /**
   * Compile-time bandwidth refinement passes after the caller ladder (`false` or 0 for a fixed
   * ladder). Defaults to the contributor's default, 2 passes.
   */
  refinement?: boolean | number;
};

/** Properties for {@link addSpatialRegressionRecipe}. */
export type GPUSpatialRegressionRecipeProps = GPURecipeOverrides<
  Record<never, never>,
  {
    ols?: {
      coefficients?: GraphDataView<'float32'>;
      standardErrors?: GraphDataView<'float32'>;
      tStatistics?: GraphDataView<'float32'>;
      summary?: GraphDataView<'float32'>;
      status?: GraphDataView<'uint32'>;
      residuals?: GraphDataView<'float32'>;
      fitted?: GraphDataView<'float32'>;
    };
    diagnostics?: {
      tests?: GraphDataView<'float32'>;
      summary?: GraphDataView<'float32'>;
      status?: GraphDataView<'uint32'>;
    };
    specification?: {decision?: GraphDataView<'uint32'>};
    residualMoran?: {
      zScores?: GraphDataView<'float32'>;
      localI?: GraphDataView<'float32'>;
      quadrants?: GraphDataView<'uint32'>;
      pValues?: GraphDataView<'float32'>;
    };
    localFits?: {
      coefficients?: GraphDataView<'float32'>;
      localR2?: GraphDataView<'float32'>;
      residuals?: GraphDataView<'float32'>;
      selectedBandwidth?: GraphDataView<'float32'>;
      localStatus?: GraphDataView<'uint32'>;
      summary?: GraphDataView<'float32'>;
    };
  }
> & {
  /** Prefix for every node and transient ID. Defaults to `'spatial-regression'`. */
  id?: string;
  /** Row-major `row * predictorCount + column` predictors, without intercept. */
  predictors: GraphDataView<'float32'>;
  /** Response, one per row. */
  response: GraphDataView<'float32'>;
  /**
   * Optional row mask of the OLS fit (zero excludes the row), also given to the diagnostics and
   * the residual local Moran, which analyze the fitted rows only (see
   * `GPUSpatialRegressionDiagnostics`). Not applied to `localFits`.
   */
  mask?: GraphDataView<'uint32'>;
  /**
   * Passed to the diagnostics: rescale each fitted row's remaining weights to sum to 1 (default
   * false). Use it after masking when `weights` were row-standardized.
   */
  restandardizeRows?: boolean;
  /**
   * Significance level of the specification search (default 0.05); a compile-time constant of the
   * generated kernel.
   */
  significanceLevel?: number;
  /** Number of predictor columns. At most 15 for the OLS and diagnostics, 7 with `localFits`. */
  predictorCount: number;
  /**
   * Square weights of the diagnostics and the residual local Moran. Any sparsity pattern is
   * accepted, including asymmetric ones such as plain k-nearest-neighbor weights (the diagnostics
   * count one-way links exactly as the dense definitions do). Built by the caller or by an earlier
   * contributor in the same graph.
   */
  weights: GPUSpatialWeights;
  /** `getGPUSpatialAutocorrelationParameterValues` view for the residual local Moran. */
  parameters: GraphDataView<'float32'>;
  /** Optional OLS ridge parameters (`getGPUOrdinaryLeastSquaresParameterValues`). */
  olsParameters?: GraphDataView<'float32'>;
  /** Rows per tile of the OLS and diagnostics reductions. */
  tileRowCount?: number;
  /** Geographically weighted local fits; skipped when absent. */
  localFits?: GPUSpatialRegressionLocalFitOptions;
};

/** Named outputs of {@link addSpatialRegressionRecipe}. */
export type GPUSpatialRegressionRecipeResult = GPURecipeResult & {
  rowCount: number;
  ols: {
    coefficients: GraphDataView<'float32'>;
    standardErrors: GraphDataView<'float32'>;
    tStatistics: GraphDataView<'float32'>;
    summary: GraphDataView<'float32'>;
    status: GraphDataView<'uint32'>;
    residuals: GraphDataView<'float32'>;
    fitted: GraphDataView<'float32'>;
  };
  diagnostics: {
    tests: GraphDataView<'float32'>;
    summary: GraphDataView<'float32'>;
    status: GraphDataView<'uint32'>;
  };
  residualMoran: {
    zScores: GraphDataView<'float32'>;
    localI: GraphDataView<'float32'>;
    quadrants: GraphDataView<'uint32'>;
    pValues: GraphDataView<'float32'>;
  };
  /**
   * GPU-written specification search outcome (one `uint32`): one of the
   * `GPU_SPATIAL_REGRESSION_SPECIFICATION_*` constants, from {@link getSpatialRegressionSpecification}.
   */
  specification: {decision: GraphDataView<'uint32'>};
  /** Present when `localFits` was requested. */
  localFits?: {
    coefficients: GraphDataView<'float32'>;
    localR2: GraphDataView<'float32'>;
    residuals: GraphDataView<'float32'>;
    selectedBandwidth: GraphDataView<'float32'>;
    localStatus: GraphDataView<'uint32'>;
    summary: GraphDataView<'float32'>;
  };
};

/**
 * Spatial regression recipe: OLS, spatial dependence diagnostics on its residuals, a GPU-written
 * specification decision, a residual local Moran map, and optionally geographically weighted local
 * fits.
 *
 * Chain: `GPUOrdinaryLeastSquares` -> `GPUSpatialRegressionDiagnostics` (LM tests, residual
 * Moran) -> specification search (Anselin's rule, see {@link getSpatialRegressionSpecification}:
 * 0 OLS adequate, 1 spatial lag, 2 spatial error, 3 both robust tests significant, 4 undetermined,
 * at `significanceLevel`) and `GPULocalMoran` of the OLS residuals;
 * `GPUGeographicallyWeightedRegression` runs on the same inputs. A `mask` (or non-finite rows)
 * is handled end to end: the diagnostics analyze the fitted rows with `W` restricted to them
 * (`restandardizeRows` rescales the remaining weights), and `diagnostics.summary` reports the
 * excluded row count. The recipe stops at the decision and does not estimate the lag or error
 * model; follow the decision with `GPUSpatialTwoStageLeastSquares` or `GPUSpatialErrorGM`.
 */
export function addSpatialRegressionRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUSpatialRegressionRecipeProps
): GPUSpatialRegressionRecipeResult {
  const id = props.id ?? 'spatial-regression';
  const builder = new RecipeBuilder(graph);
  const outputOverrides = props.outputs ?? {};
  const rowCount = props.response.length;
  const {predictorCount} = props;
  assertRecipe(
    'addSpatialRegressionRecipe',
    props.predictors.length === rowCount * predictorCount,
    'predictors must hold rowCount * predictorCount values'
  );

  const olsInput = outputOverrides.ols ?? {};
  const ols = {
    coefficients: getOrCreateView(
      graph,
      `${id}-coefficients`,
      'float32',
      predictorCount + 1,
      olsInput.coefficients
    ),
    standardErrors: getOrCreateView(
      graph,
      `${id}-standard-errors`,
      'float32',
      predictorCount + 1,
      olsInput.standardErrors
    ),
    tStatistics: getOrCreateView(
      graph,
      `${id}-t-statistics`,
      'float32',
      predictorCount + 1,
      olsInput.tStatistics
    ),
    summary: getOrCreateView(
      graph,
      `${id}-ols-summary`,
      'float32',
      GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH,
      olsInput.summary
    ),
    status: getOrCreateView(graph, `${id}-ols-status`, 'uint32', 1, olsInput.status),
    residuals: getOrCreateView(graph, `${id}-residuals`, 'float32', rowCount, olsInput.residuals),
    fitted: getOrCreateView(graph, `${id}-fitted`, 'float32', rowCount, olsInput.fitted)
  };
  builder.add(
    new GPUOrdinaryLeastSquares({
      id: `${id}-ols`,
      predictors: props.predictors,
      response: props.response,
      parameters: props.olsParameters,
      mask: props.mask,
      predictorCount,
      tileRowCount: props.tileRowCount,
      output: ols
    })
  );

  const diagnosticsInput = outputOverrides.diagnostics ?? {};
  const diagnostics = {
    tests: getOrCreateView(
      graph,
      `${id}-tests`,
      'float32',
      GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH,
      diagnosticsInput.tests
    ),
    summary: getOrCreateView(
      graph,
      `${id}-diagnostics-summary`,
      'float32',
      GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH,
      diagnosticsInput.summary
    ),
    status: getOrCreateView(graph, `${id}-diagnostics-status`, 'uint32', 1, diagnosticsInput.status)
  };
  builder.add(
    new GPUSpatialRegressionDiagnostics({
      id: `${id}-diagnostics`,
      weights: props.weights,
      predictors: props.predictors,
      response: props.response,
      residuals: ols.residuals,
      mask: props.mask,
      restandardizeRows: props.restandardizeRows,
      predictorCount,
      tileRowCount: props.tileRowCount,
      output: diagnostics
    })
  );

  const specification = {
    decision: getOrCreateView(
      graph,
      `${id}-specification-decision`,
      'uint32',
      1,
      outputOverrides.specification?.decision
    )
  };
  builder.add(
    new SpatialRegressionSpecificationSearch({
      id,
      tests: diagnostics.tests,
      status: diagnostics.status,
      significanceLevel: props.significanceLevel ?? 0.05,
      decision: specification.decision
    }) as unknown as GPUCommandNodeProducer<Parameters>
  );

  const moranInput = outputOverrides.residualMoran ?? {};
  const residualMoran = {
    zScores: getOrCreateView(graph, `${id}-moran-z`, 'float32', rowCount, moranInput.zScores),
    localI: getOrCreateView(graph, `${id}-moran-i`, 'float32', rowCount, moranInput.localI),
    quadrants: getOrCreateView(
      graph,
      `${id}-moran-quadrants`,
      'uint32',
      rowCount,
      moranInput.quadrants
    ),
    pValues: getOrCreateView(graph, `${id}-moran-p`, 'float32', rowCount, moranInput.pValues)
  };
  builder.add(
    new GPULocalMoran({
      id: `${id}-residual-moran`,
      weights: props.weights,
      values: ols.residuals,
      mask: props.mask,
      parameters: props.parameters,
      ...residualMoran
    })
  );

  const result: GPUSpatialRegressionRecipeResult = {
    contributors: builder.contributors,
    rowCount,
    ols,
    diagnostics,
    residualMoran,
    specification,
    outputs: {ols, diagnostics, specification, residualMoran},
    intermediates: {},
    status: {
      stages: [
        {stage: 'ordinary-least-squares', status: {invalidCount: ols.status}},
        {stage: 'spatial-diagnostics', status: {invalidCount: diagnostics.status}}
      ]
    }
  };

  const local = props.localFits;
  if (local) {
    const provided = outputOverrides.localFits ?? {};
    const outputs = {
      coefficients: getOrCreateView(
        graph,
        `${id}-local-coefficients`,
        'float32',
        rowCount * (predictorCount + 1),
        provided.coefficients
      ),
      localR2: getOrCreateView(graph, `${id}-local-r2`, 'float32', rowCount, provided.localR2),
      residuals: getOrCreateView(
        graph,
        `${id}-local-residuals`,
        'float32',
        rowCount,
        provided.residuals
      ),
      selectedBandwidth: getOrCreateView(
        graph,
        `${id}-selected-bandwidth`,
        'float32',
        2,
        provided.selectedBandwidth
      ),
      localStatus: getOrCreateView(
        graph,
        `${id}-local-status`,
        'uint32',
        rowCount,
        provided.localStatus
      ),
      summary: getOrCreateView(
        graph,
        `${id}-local-summary`,
        'float32',
        GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH,
        provided.summary
      )
    };
    builder.add(
      new GPUGeographicallyWeightedRegression({
        id: `${id}-gwr`,
        positions: local.positions,
        predictors: props.predictors,
        predictorCount,
        response: props.response,
        parameters: local.parameters,
        maximumBandwidthCount: local.maximumBandwidthCount,
        maximumNeighborCount: local.maximumNeighborCount,
        refinement: local.refinement,
        output: outputs
      })
    );
    result.localFits = outputs;
    result.outputs['localFits'] = outputs;
  }
  return result;
}
