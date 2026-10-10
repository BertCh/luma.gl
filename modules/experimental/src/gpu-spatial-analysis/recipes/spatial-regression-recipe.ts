// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {GPULocalMoran} from '../spatial-autocorrelation/index';
import {
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH,
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
  /** Number of predictor columns. At most 15 for the diagnostics, 7 with `localFits`. */
  predictorCount: number;
  /**
   * Square weights of the diagnostics and the residual local Moran, with a symmetric sparsity
   * pattern (for example row-standardized contiguity). Built by the caller or by an earlier
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
 * Spatial regression recipe: OLS, spatial dependence diagnostics on its residuals, a residual
 * local Moran map, and optionally geographically weighted local fits.
 *
 * Chain: `GPUOrdinaryLeastSquares` -> `GPUSpatialRegressionDiagnostics` (LM tests, residual
 * Moran) and `GPULocalMoran` of the OLS residuals; `GPUGeographicallyWeightedRegression` runs on
 * the same inputs. Every row must be finite (no mask): the diagnostics reject non-finite
 * residuals, which is reported through `diagnostics.status`.
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
      predictorCount,
      tileRowCount: props.tileRowCount,
      output: diagnostics
    })
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
    outputs: {ols, diagnostics, residualMoran},
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
        output: outputs
      })
    );
    result.localFits = outputs;
    result.outputs['localFits'] = outputs;
  }
  return result;
}
