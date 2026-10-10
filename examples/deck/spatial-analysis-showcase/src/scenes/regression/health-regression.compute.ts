// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {COORDINATE_SYSTEM} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  addSpatialRegressionRecipe,
  getGPUNeighborSearchParameterValues,
  getGPUOrdinaryLeastSquaresParameterValues,
  getGPUSpatialAutocorrelationParameterValues,
  GPUContiguityWeights,
  GPUNeighborSearch,
  GPUSpatialErrorGM,
  GPUSpatialTwoStageLeastSquares,
  GPUSpatialWeightsTransform,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  GPU_ORDINARY_LEAST_SQUARES_PARAMETER_LENGTH,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_ADJUSTED_R_SQUARED,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_AIC,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BIC,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BREUSCH_PAGAN_P_VALUE,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_JARQUE_BERA_P_VALUE,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_R_SQUARED,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_SIGMA_SQUARED,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  GPU_SPATIAL_ERROR_GM_STATUS_OK,
  GPU_SPATIAL_ERROR_GM_SUMMARY_LAMBDA,
  GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH,
  GPU_SPATIAL_ERROR_GM_SUMMARY_PSEUDO_R_SQUARED,
  GPU_SPATIAL_ERROR_GM_TABLE_STRIDE,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_MORAN_EXPECTATION,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_MORAN_I,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_LM_ERROR,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_LM_LAG,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_LM_SARMA,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_MORAN_RESIDUALS,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_ROBUST_LM_ERROR,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_ROBUST_LM_LAG,
  GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_STRIDE,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_ANSELIN_KELEJIAN,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_ANSELIN_KELEJIAN_P_VALUE,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_PSEUDO_R_SQUARED,
  GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {SpatialAnalysisResources} from '../../engine/resources';
import type {SceneContext, SceneInstance} from '../scene';
import {createNamedReader} from './b6-reader';
import {PolygonFillLayer} from './b6-fill-layer';
import {
  buildOutlineSegments,
  computeCentroids,
  computeQueenAdjacency,
  createFeatureLocator,
  createLocalProjector,
  formatInteger,
  formatNumber,
  formatP,
  getLargestComponent,
  getMoments,
  getQuantile,
  NO_ROW,
  projectRows,
  readPolygonGeometry,
  subsetPolygonGeometry,
  triangulatePolygonGeometry
} from './b6-polygons';

/** Option state of the health-regression scene. */
export type HealthRegressionOptions = {
  outcome: 'diabetes' | 'obesity' | 'asthma' | 'depression' | 'heartDisease' | 'smoking';
  model: 'income' | 'economic' | 'full';
  weights: 'queen' | 'rook' | 'knn4' | 'knn8' | 'knn12';
  instruments: '1' | '2';
  ridge: number;
  significance: number;
  map: 'residuals' | 'fitted' | 'observed' | 'clusters' | 'lag' | 'error';
};

type ModelId = HealthRegressionOptions['model'];
type WeightsKind = HealthRegressionOptions['weights'];
type MapKind = HealthRegressionOptions['map'];

/** Predictor catalogue: label and how to compute it from tract columns. */
const PREDICTORS = [
  {id: 'income', label: 'ln income per capita'},
  {id: 'poverty', label: 'Poverty rate'},
  {id: 'uninsured', label: 'Uninsured rate'},
  {id: 'unemployed', label: 'Unemployment rate'},
  {id: 'age65', label: 'Age 65+ share'},
  {id: 'disability', label: 'Disability rate'},
  {id: 'noVehicle', label: 'No-vehicle households'},
  {id: 'black', label: 'Black share'},
  {id: 'hispanic', label: 'Hispanic share'}
] as const;
type PredictorId = (typeof PREDICTORS)[number]['id'];

/** Predictor sets of the `model` option. */
export const MODEL_PREDICTORS: Record<ModelId, readonly PredictorId[]> = {
  income: ['income'],
  economic: ['income', 'poverty', 'uninsured', 'unemployed'],
  full: ['income', 'poverty', 'uninsured', 'age65', 'disability', 'noVehicle', 'black', 'hispanic']
};

const OUTCOME_LABELS: Record<HealthRegressionOptions['outcome'], string> = {
  diabetes: 'Diabetes prevalence',
  obesity: 'Obesity prevalence',
  asthma: 'Current asthma prevalence',
  depression: 'Depression prevalence',
  heartDisease: 'Coronary heart disease prevalence',
  smoking: 'Current smoking prevalence'
};

const SLOTS_PER_ROW = 32;
const MAXIMUM_PREDICTORS = 8;
const GRID_SIZE: readonly [number, number] = [64, 64];

const NEIGHBOR_COUNTS: Record<string, number> = {knn4: 4, knn8: 8, knn12: 12};

/** Quadrant colors of the residual local Moran (index = quadrant code). */
export const RESIDUAL_CLUSTER_COLORS = [
  [150, 150, 158, 120],
  [214, 69, 65, 235],
  [138, 176, 232, 235],
  [49, 104, 190, 235],
  [240, 160, 60, 235]
] as const;

type Variant = {
  predictorCount: number;
  names: string[];
  recipe: CompiledGPUCommandGraph<void>;
  lag: Map<'1' | '2', CompiledGPUCommandGraph<void>>;
  error: CompiledGPUCommandGraph<void>;
  buffers: {
    predictors: Buffer;
    olsResiduals: Buffer;
    fitted: Buffer;
    quadrants: Buffer;
    lagResiduals: Buffer;
    errorResiduals: Buffer;
  };
  addLag: (order: '1' | '2') => CompiledGPUCommandGraph<void>;
  reader: ReturnType<typeof createNamedReader>;
};

/**
 * Regression of a PLACES health outcome on tract covariates for Chicago. The recipe graph
 * (`addSpatialRegressionRecipe`) runs `GPUOrdinaryLeastSquares`, `GPUSpatialRegressionDiagnostics`
 * and a residual `GPULocalMoran` on the study tracts; `GPUSpatialTwoStageLeastSquares` and
 * `GPUSpatialErrorGM` fit the lag and error alternatives on the same weights. Weights come from a
 * compiled `GPUContiguityWeights` (queen or rook, row-standardized by `GPUSpatialWeightsTransform`)
 * or `GPUNeighborSearch` kNN. Predictor sets, weights and instrument order are compile-time
 * choices: each variant is compiled the first time it is chosen and kept. Ridge penalty, outcome,
 * significance level and the map are parameter or layer changes. Results are written to
 * caller-owned buffers that the fill layer reads directly; small summaries and per-row result
 * columns are read back once per change (never per frame) for the readouts and tooltips.
 */
export async function createHealthRegression(
  ctx: SceneContext<HealthRegressionOptions>
): Promise<SceneInstance<HealthRegressionOptions>> {
  const tracts = ctx.datasets.get('chicago-tracts');
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'health-regression');

  // ---- Study set: tracts with every column finite, in the largest queen-connected group. ----
  const fullGeometry = readPolygonGeometry(tracts);
  const featureCount = fullGeometry.featureCount;
  const column = (name: string) => tracts.column<Float32Array>(name);
  const population = column('population');
  const households = column('households');
  const income = column('perCapitaIncome');
  const share = (numerator: string, denominator: Float32Array) => {
    const values = column(numerator);
    return Float32Array.from(values, (value, index) =>
      denominator[index] > 0 ? (100 * value) / denominator[index] : Number.NaN
    );
  };
  const rawPredictors: Record<PredictorId, Float32Array> = {
    income: Float32Array.from(income, value => (value > 0 ? Math.log(value) : Number.NaN)),
    poverty: column('poverty150Pct'),
    uninsured: column('uninsuredPct'),
    unemployed: column('unemployedPct'),
    age65: column('age65Pct'),
    disability: column('disabilityPct'),
    noVehicle: column('noVehiclePct'),
    black: share('nhBlack', population),
    hispanic: share('hispanic', population)
  };
  const outcomes = Object.fromEntries(
    (Object.keys(OUTCOME_LABELS) as HealthRegressionOptions['outcome'][]).map(key => [
      key,
      column(key)
    ])
  ) as Record<HealthRegressionOptions['outcome'], Float32Array>;
  const candidates: number[] = [];
  for (let feature = 0; feature < featureCount; feature++) {
    let finite = population[feature] >= 300 && households[feature] > 0;
    for (const values of Object.values(rawPredictors)) finite &&= Number.isFinite(values[feature]);
    for (const values of Object.values(outcomes)) finite &&= Number.isFinite(values[feature]);
    if (finite) candidates.push(feature);
  }
  const fullAdjacency = computeQueenAdjacency(fullGeometry);
  const features = getLargestComponent(fullAdjacency, candidates);
  const rowCount = features.length;
  const rowOfFeature = new Uint32Array(featureCount).fill(NO_ROW);
  features.forEach((feature, row) => {
    rowOfFeature[feature] = row;
  });
  const geometry = subsetPolygonGeometry(fullGeometry, features);
  const lngLat = computeCentroids(geometry);
  const origin = tracts.defaultOrigin;
  const meters = projectRows(lngLat, createLocalProjector(origin));
  const geoids = (tracts.manifest as unknown as {geoid: string[]}).geoid;

  // Standardized predictors (z-scores over the study tracts), one array per predictor.
  const standardized = {} as Record<PredictorId, Float32Array>;
  for (const {id} of PREDICTORS) {
    const raw = Float32Array.from(features, feature => rawPredictors[id][feature]);
    const {mean, deviation} = getMoments(raw);
    standardized[id] = Float32Array.from(raw, value => (value - mean) / deviation);
  }
  const getResponse = (outcome: HealthRegressionOptions['outcome']) =>
    Float32Array.from(features, feature => outcomes[outcome][feature]);

  // ---- Geometry buffers. ----
  const triangles = triangulatePolygonGeometry(geometry);
  const excludedRows = new Uint32Array(featureCount).fill(NO_ROW);
  for (let feature = 0; feature < featureCount; feature++) {
    if (rowOfFeature[feature] === NO_ROW) excludedRows[feature] = 0;
  }
  const allTriangles = triangulatePolygonGeometry(fullGeometry, excludedRows);
  const trianglesBuffer = resources.createBuffer('triangles', triangles.corners);
  const ownersBuffer = resources.createBuffer('owners', triangles.owners);
  const excludedTrianglesBuffer = resources.createBuffer(
    'excluded-triangles',
    allTriangles.corners
  );
  const excludedOwnersBuffer = resources.createBuffer('excluded-owners', allTriangles.owners);
  const outlineSegments = buildOutlineSegments(fullGeometry);
  const outlineBuffer = resources.createBuffer('outline', outlineSegments);
  const locate = createFeatureLocator(geometry);

  // ---- Inputs and weights buffers. ----
  const positionsBuffer = resources.createBuffer('positions', meters);
  const verticesBuffer = resources.createBuffer('vertices', geometry.vertices);
  const ringOffsetsBuffer = resources.createBuffer('ring-offsets', geometry.ringOffsets);
  const polygonOffsetsBuffer = resources.createBuffer(
    'polygon-offsets',
    geometry.featureRingOffsets
  );
  const responseBuffer = resources.createBuffer('response', getResponse(ctx.options.outcome));
  const capacity = rowCount * SLOTS_PER_ROW;
  const offsetsBuffer = resources.createBuffer('offsets', (rowCount + 1) * 4);
  const neighborsBuffer = resources.createBuffer('neighbors', capacity * 4);
  const weightValuesBuffer = resources.createBuffer('weight-values', capacity * 4);
  const overflowBuffer = resources.createBuffer('overflow', 4);
  const totalNeighborsBuffer = resources.createBuffer('total-neighbors', 4);
  const searchParameters = resources.createParameterBuffer(
    'search-parameters',
    'float32',
    GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH
  );
  const olsParameters = resources.createParameterBuffer(
    'ols-parameters',
    'float32',
    GPU_ORDINARY_LEAST_SQUARES_PARAMETER_LENGTH
  );
  const moranParameters = resources.createParameterBuffer(
    'moran-parameters',
    'float32',
    GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
  );

  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  for (let row = 0; row < rowCount; row++) {
    minimumX = Math.min(minimumX, meters[row * 2]);
    maximumX = Math.max(maximumX, meters[row * 2]);
    minimumY = Math.min(minimumY, meters[row * 2 + 1]);
    maximumY = Math.max(maximumY, meters[row * 2 + 1]);
  }
  const bounds: [number, number, number, number] = [
    minimumX - 500,
    minimumY - 500,
    maximumX + 500,
    maximumY + 500
  ];

  // ---- Weights graphs (compiled when first chosen). ----
  const importWeights = (graph: GPUCommandGraph<void>) => ({
    offsets: importGraphBuffer(graph, 'offsets', offsetsBuffer, 'uint32', rowCount + 1),
    neighbors: importGraphBuffer(graph, 'neighbors', neighborsBuffer, 'uint32', capacity),
    weights: importGraphBuffer(graph, 'weight-values', weightValuesBuffer, 'float32', capacity)
  });
  const weightsGraphs = new Map<WeightsKind, CompiledGPUCommandGraph<void>>();
  const getWeightsGraph = (kind: WeightsKind): CompiledGPUCommandGraph<void> => {
    let compiled = weightsGraphs.get(kind);
    if (compiled) return compiled;
    const graph = new GPUCommandGraph<void>(device, {id: `health-weights-${kind}`});
    const weights = importWeights(graph);
    const overflow = importGraphBuffer(graph, 'overflow', overflowBuffer, 'uint32', 1);
    const totalNeighbors = importGraphBuffer(
      graph,
      'total-neighbors',
      totalNeighborsBuffer,
      'uint32',
      1
    );
    if (kind === 'queen' || kind === 'rook') {
      graph.add(
        new GPUContiguityWeights({
          id: 'contiguity',
          criterion: kind,
          positions: importGraphBuffer(
            graph,
            'vertices',
            verticesBuffer,
            'float32x2',
            geometry.vertices.length / 2
          ),
          ringOffsets: importGraphBuffer(
            graph,
            'ring-offsets',
            ringOffsetsBuffer,
            'uint32',
            geometry.ringOffsets.length
          ),
          polygonOffsets: importGraphBuffer(
            graph,
            'polygon-offsets',
            polygonOffsetsBuffer,
            'uint32',
            rowCount + 1
          ),
          weights,
          overflow,
          totalNeighbors
        })
      );
      graph.add(new GPUSpatialWeightsTransform({id: 'row-standardize', operation: 'row', weights}));
    } else {
      graph.add(
        new GPUNeighborSearch({
          id: 'knn',
          mode: 'knn',
          k: NEIGHBOR_COUNTS[kind],
          gridSize: GRID_SIZE,
          positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', rowCount),
          parameters: searchParameters.importToGraph(graph),
          weights,
          overflow,
          totalNeighbors
        })
      );
    }
    compiled = resources.track(graph.compile());
    weightsGraphs.set(kind, compiled);
    return compiled;
  };
  searchParameters.write(
    getGPUNeighborSearchParameterValues({
      bounds,
      radius: Infinity,
      weightKind: 'binary',
      rowStandardize: true
    })
  );

  // ---- Model variants (compiled when first chosen). ----
  const variants = new Map<ModelId, Variant>();
  const getVariant = (model: ModelId): Variant => {
    const existing = variants.get(model);
    if (existing) return existing;
    const ids = MODEL_PREDICTORS[model];
    const predictorCount = ids.length;
    if (predictorCount > MAXIMUM_PREDICTORS) throw new Error('Too many predictors');
    const predictorValues = new Float32Array(rowCount * predictorCount);
    ids.forEach((id, columnIndex) => {
      for (let row = 0; row < rowCount; row++) {
        predictorValues[row * predictorCount + columnIndex] = standardized[id][row];
      }
    });
    const tag = `${model}-`;
    const make = (name: string, bytes: number) => resources.createBuffer(`${tag}${name}`, bytes);
    const buffers = {
      predictors: resources.createBuffer(`${tag}predictors`, predictorValues),
      coefficients: make('coefficients', (predictorCount + 1) * 4),
      standardErrors: make('standard-errors', (predictorCount + 1) * 4),
      tStatistics: make('t-statistics', (predictorCount + 1) * 4),
      olsSummary: make('ols-summary', GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH * 4),
      olsStatus: make('ols-status', 4),
      olsResiduals: make('ols-residuals', rowCount * 4),
      fitted: make('fitted', rowCount * 4),
      tests: make('tests', GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH * 4),
      diagnosticsSummary: make(
        'diagnostics-summary',
        GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH * 4
      ),
      diagnosticsStatus: make('diagnostics-status', 4),
      zScores: make('moran-z', rowCount * 4),
      localI: make('moran-i', rowCount * 4),
      quadrants: make('quadrants', rowCount * 4),
      pValues: make('moran-p', rowCount * 4),
      lagTable: make(
        'lag-table',
        (predictorCount + 2) * GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE * 4
      ),
      lagSummary: make('lag-summary', GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH * 4),
      lagStatus: make('lag-status', 4),
      lagResiduals: make('lag-residuals', rowCount * 4),
      errorTable: make('error-table', (predictorCount + 2) * GPU_SPATIAL_ERROR_GM_TABLE_STRIDE * 4),
      errorSummary: make('error-summary', GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH * 4),
      errorStatus: make('error-status', 4),
      errorResiduals: make('error-residuals', rowCount * 4)
    };

    // Recipe graph: OLS, diagnostics and the residual local Moran in one compiled graph.
    const recipeGraph = new GPUCommandGraph<void>(device, {id: `health-recipe-${model}`});
    const view = <F extends 'float32' | 'uint32'>(
      graph: GPUCommandGraph<void>,
      name: string,
      buffer: Buffer,
      format: F,
      length: number
    ) => importGraphBuffer(graph, name, buffer, format, length) as GraphDataView<F>;
    const predictorView = (graph: GPUCommandGraph<void>) =>
      view(graph, 'predictors', buffers.predictors, 'float32', rowCount * predictorCount);
    const responseView = (graph: GPUCommandGraph<void>) =>
      view(graph, 'response', responseBuffer, 'float32', rowCount);
    addSpatialRegressionRecipe(recipeGraph, {
      id: 'health',
      predictors: predictorView(recipeGraph),
      response: responseView(recipeGraph),
      predictorCount,
      weights: importWeights(recipeGraph),
      parameters: moranParameters.importToGraph(recipeGraph),
      olsParameters: olsParameters.importToGraph(recipeGraph),
      outputs: {
        ols: {
          coefficients: view(
            recipeGraph,
            'coefficients',
            buffers.coefficients,
            'float32',
            predictorCount + 1
          ),
          standardErrors: view(
            recipeGraph,
            'standard-errors',
            buffers.standardErrors,
            'float32',
            predictorCount + 1
          ),
          tStatistics: view(
            recipeGraph,
            't-statistics',
            buffers.tStatistics,
            'float32',
            predictorCount + 1
          ),
          summary: view(
            recipeGraph,
            'ols-summary',
            buffers.olsSummary,
            'float32',
            GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH
          ),
          status: view(recipeGraph, 'ols-status', buffers.olsStatus, 'uint32', 1),
          residuals: view(recipeGraph, 'ols-residuals', buffers.olsResiduals, 'float32', rowCount),
          fitted: view(recipeGraph, 'fitted', buffers.fitted, 'float32', rowCount)
        },
        diagnostics: {
          tests: view(
            recipeGraph,
            'tests',
            buffers.tests,
            'float32',
            GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH
          ),
          summary: view(
            recipeGraph,
            'diagnostics-summary',
            buffers.diagnosticsSummary,
            'float32',
            GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH
          ),
          status: view(recipeGraph, 'diagnostics-status', buffers.diagnosticsStatus, 'uint32', 1)
        },
        residualMoran: {
          zScores: view(recipeGraph, 'moran-z', buffers.zScores, 'float32', rowCount),
          localI: view(recipeGraph, 'moran-i', buffers.localI, 'float32', rowCount),
          quadrants: view(recipeGraph, 'quadrants', buffers.quadrants, 'uint32', rowCount),
          pValues: view(recipeGraph, 'moran-p', buffers.pValues, 'float32', rowCount)
        }
      }
    });
    const recipe = resources.track(recipeGraph.compile());

    // Spatial error model graph.
    const errorGraph = new GPUCommandGraph<void>(device, {id: `health-error-${model}`});
    errorGraph.add(
      new GPUSpatialErrorGM({
        id: 'error-gm',
        weights: importWeights(errorGraph),
        predictors: predictorView(errorGraph),
        response: responseView(errorGraph),
        predictorCount,
        output: {
          table: view(
            errorGraph,
            'error-table',
            buffers.errorTable,
            'float32',
            (predictorCount + 2) * GPU_SPATIAL_ERROR_GM_TABLE_STRIDE
          ),
          summary: view(
            errorGraph,
            'error-summary',
            buffers.errorSummary,
            'float32',
            GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH
          ),
          status: view(errorGraph, 'error-status', buffers.errorStatus, 'uint32', 1),
          residuals: view(
            errorGraph,
            'error-residuals',
            buffers.errorResiduals,
            'float32',
            rowCount
          )
        }
      })
    );
    const error = resources.track(errorGraph.compile());

    const lag = new Map<'1' | '2', CompiledGPUCommandGraph<void>>();
    const addLag = (order: '1' | '2') => {
      let compiled = lag.get(order);
      if (compiled) return compiled;
      const lagGraph = new GPUCommandGraph<void>(device, {id: `health-lag-${model}-${order}`});
      lagGraph.add(
        new GPUSpatialTwoStageLeastSquares({
          id: 'two-stage',
          weights: importWeights(lagGraph),
          predictors: predictorView(lagGraph),
          response: responseView(lagGraph),
          predictorCount,
          instrumentOrder: order === '2' ? 2 : 1,
          output: {
            table: view(
              lagGraph,
              'lag-table',
              buffers.lagTable,
              'float32',
              (predictorCount + 2) * GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE
            ),
            summary: view(
              lagGraph,
              'lag-summary',
              buffers.lagSummary,
              'float32',
              GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH
            ),
            status: view(lagGraph, 'lag-status', buffers.lagStatus, 'uint32', 1),
            residuals: view(lagGraph, 'lag-residuals', buffers.lagResiduals, 'float32', rowCount)
          }
        })
      );
      compiled = resources.track(lagGraph.compile());
      lag.set(order, compiled);
      return compiled;
    };

    const reader = createNamedReader(
      resources,
      `health-${model}`,
      [
        {name: 'coefficients', buffer: buffers.coefficients, bytes: (predictorCount + 1) * 4},
        {name: 'standardErrors', buffer: buffers.standardErrors, bytes: (predictorCount + 1) * 4},
        {name: 'tStatistics', buffer: buffers.tStatistics, bytes: (predictorCount + 1) * 4},
        {
          name: 'olsSummary',
          buffer: buffers.olsSummary,
          bytes: GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH * 4
        },
        {name: 'olsStatus', buffer: buffers.olsStatus, bytes: 4},
        {
          name: 'tests',
          buffer: buffers.tests,
          bytes: GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TESTS_LENGTH * 4
        },
        {
          name: 'diagnosticsSummary',
          buffer: buffers.diagnosticsSummary,
          bytes: GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_LENGTH * 4
        },
        {name: 'diagnosticsStatus', buffer: buffers.diagnosticsStatus, bytes: 4},
        {
          name: 'lagTable',
          buffer: buffers.lagTable,
          bytes: (predictorCount + 2) * GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE * 4
        },
        {
          name: 'lagSummary',
          buffer: buffers.lagSummary,
          bytes: GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_LENGTH * 4
        },
        {name: 'lagStatus', buffer: buffers.lagStatus, bytes: 4},
        {
          name: 'errorTable',
          buffer: buffers.errorTable,
          bytes: (predictorCount + 2) * GPU_SPATIAL_ERROR_GM_TABLE_STRIDE * 4
        },
        {
          name: 'errorSummary',
          buffer: buffers.errorSummary,
          bytes: GPU_SPATIAL_ERROR_GM_SUMMARY_LENGTH * 4
        },
        {name: 'errorStatus', buffer: buffers.errorStatus, bytes: 4},
        {name: 'overflow', buffer: overflowBuffer, bytes: 4},
        {name: 'totalNeighbors', buffer: totalNeighborsBuffer, bytes: 4},
        {name: 'olsResiduals', buffer: buffers.olsResiduals, bytes: rowCount * 4},
        {name: 'fitted', buffer: buffers.fitted, bytes: rowCount * 4},
        {name: 'quadrants', buffer: buffers.quadrants, bytes: rowCount * 4},
        {name: 'lagResiduals', buffer: buffers.lagResiduals, bytes: rowCount * 4},
        {name: 'errorResiduals', buffer: buffers.errorResiduals, bytes: rowCount * 4}
      ],
      get => onResult(model, predictorCount, get)
    );
    const variant: Variant = {
      predictorCount,
      names: ['Intercept', ...ids.map(id => PREDICTORS.find(entry => entry.id === id)!.label)],
      recipe,
      lag,
      error,
      buffers: {
        predictors: buffers.predictors,
        olsResiduals: buffers.olsResiduals,
        fitted: buffers.fitted,
        quadrants: buffers.quadrants,
        lagResiduals: buffers.lagResiduals,
        errorResiduals: buffers.errorResiduals
      },
      addLag,
      reader
    };
    variants.set(model, variant);
    return variant;
  };

  // ---- Result state (CPU copies for tooltips, legends and ranges). ----
  let olsSigma = 1;
  let latest: {
    olsResiduals: Float32Array;
    fitted: Float32Array;
    lagResiduals: Float32Array;
    errorResiduals: Float32Array;
    quadrants: Uint32Array;
  } | null = null;
  let responseRange: [number, number] = [0, 1];
  let selectedRow = -1;

  const updateResponseRange = () => {
    const response = getResponse(ctx.options.outcome);
    responseRange = [getQuantile(response, 0.02), getQuantile(response, 0.98)];
    ctx.setLegendExtent('value', responseRange);
  };

  const onResult = (
    model: ModelId,
    predictorCount: number,
    get: (name: string) => {f32: Float32Array; u32: Uint32Array}
  ) => {
    if (model !== ctx.options.model) return;
    const variant = variants.get(model)!;
    const summary = get('olsSummary').f32;
    const coefficients = get('coefficients').f32;
    const standardErrors = get('standardErrors').f32;
    const tStatistics = get('tStatistics').f32;
    const tests = get('tests').f32;
    const diagnostics = get('diagnosticsSummary').f32;
    const lagTable = get('lagTable').f32;
    const lagSummary = get('lagSummary').f32;
    const errorTable = get('errorTable').f32;
    const errorSummary = get('errorSummary').f32;
    const statuses = [
      get('olsStatus').u32[0],
      get('diagnosticsStatus').u32[0],
      get('lagStatus').u32[0],
      get('errorStatus').u32[0]
    ];
    olsSigma = Math.sqrt(Math.max(0, summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_SIGMA_SQUARED]));
    ctx.setLegendExtent('residual', [-2 * olsSigma, 2 * olsSigma]);
    latest = {
      olsResiduals: get('olsResiduals').f32,
      fitted: get('fitted').f32,
      lagResiduals: get('lagResiduals').f32,
      errorResiduals: get('errorResiduals').f32,
      quadrants: get('quadrants').u32
    };

    ctx.setReadout('rows', `${formatInteger(rowCount)} of ${formatInteger(featureCount)} tracts`);
    const overflow = get('overflow').u32[0];
    const linkTotal = Math.min(get('totalNeighbors').u32[0], capacity);
    ctx.setReadout(
      'links',
      `${formatInteger(linkTotal)} (mean ${(linkTotal / rowCount).toFixed(1)} per tract)${overflow ? ', OVERFLOW' : ''}`
    );
    ctx.setReadout(
      'fit',
      `R² ${formatNumber(summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_R_SQUARED])}, adjusted ${formatNumber(summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_ADJUSTED_R_SQUARED])}`
    );
    ctx.setReadout(
      'criteria',
      `AIC ${formatNumber(summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_AIC], 1)}, BIC ${formatNumber(summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BIC], 1)}`
    );
    ctx.setReadout(
      'normality',
      formatP(summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_JARQUE_BERA_P_VALUE])
    );
    ctx.setReadout(
      'heteroskedasticity',
      formatP(summary[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BREUSCH_PAGAN_P_VALUE])
    );

    const stride = GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_STRIDE;
    const test = (row: number) => ({statistic: tests[row * stride], p: tests[row * stride + 2]});
    const moranRow = test(GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_MORAN_RESIDUALS);
    ctx.setReadout(
      'moran',
      `I ${formatNumber(diagnostics[GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_MORAN_I])} (expected ${formatNumber(diagnostics[GPU_SPATIAL_REGRESSION_DIAGNOSTICS_SUMMARY_MORAN_EXPECTATION], 4)}), z ${formatNumber(moranRow.statistic, 1)}, ${formatP(moranRow.p)}`
    );
    const lagTest = test(GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_LM_LAG);
    const errorTest = test(GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_LM_ERROR);
    const robustLag = test(GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_ROBUST_LM_LAG);
    const robustError = test(GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_ROBUST_LM_ERROR);
    const sarma = test(GPU_SPATIAL_REGRESSION_DIAGNOSTICS_TEST_LM_SARMA);
    const describe = (entry: {statistic: number; p: number}) =>
      `${formatNumber(entry.statistic, 1)}, ${formatP(entry.p)}`;
    ctx.setReadout('lmLag', describe(lagTest));
    ctx.setReadout('lmError', describe(errorTest));
    ctx.setReadout('robustLag', describe(robustLag));
    ctx.setReadout('robustError', describe(robustError));
    ctx.setReadout('sarma', describe(sarma));
    const alpha = 0.05;
    let verdict: string;
    if (!(lagTest.p <= alpha) && !(errorTest.p <= alpha)) {
      verdict = 'Neither LM test is significant: OLS is adequate.';
    } else if (lagTest.p <= alpha && !(errorTest.p <= alpha)) {
      verdict = 'Only LM-lag is significant: fit the spatial lag model.';
    } else if (!(lagTest.p <= alpha) && errorTest.p <= alpha) {
      verdict = 'Only LM-error is significant: fit the spatial error model.';
    } else {
      verdict =
        robustLag.p < robustError.p
          ? 'Both significant; the robust lag test is stronger: spatial lag model.'
          : 'Both significant; the robust error test is stronger: spatial error model.';
    }
    ctx.setReadout('verdict', verdict);

    const lagStride = GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_TABLE_STRIDE;
    const errorStride = GPU_SPATIAL_ERROR_GM_TABLE_STRIDE;
    for (let index = 0; index <= MAXIMUM_PREDICTORS; index++) {
      if (index > predictorCount) {
        ctx.setReadout(`coefficient${index}`, null);
        continue;
      }
      const name = variant.names[index];
      ctx.setReadout(
        `coefficient${index}`,
        `${name}: ${formatNumber(coefficients[index], 2)} (t ${formatNumber(tStatistics[index], 1)}, se ${formatNumber(standardErrors[index], 2)})  to lag ${formatNumber(lagTable[index * lagStride], 2)}, error ${formatNumber(errorTable[index * errorStride], 2)}`
      );
    }
    const rhoBase = (predictorCount + 1) * lagStride;
    ctx.setReadout(
      'rho',
      statuses[2] === 0
        ? `ρ ${formatNumber(lagTable[rhoBase])} ± ${formatNumber(lagTable[rhoBase + 1])} (z ${formatNumber(lagTable[rhoBase + 2], 1)}, ${formatP(lagTable[rhoBase + 3])})`
        : `fit failed (status ${statuses[2]})`
    );
    ctx.setReadout(
      'lagFit',
      `pseudo R² ${formatNumber(lagSummary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_PSEUDO_R_SQUARED])}; Anselin-Kelejian ${formatNumber(lagSummary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_ANSELIN_KELEJIAN], 1)}, ${formatP(lagSummary[GPU_SPATIAL_TWO_STAGE_LEAST_SQUARES_SUMMARY_ANSELIN_KELEJIAN_P_VALUE])}`
    );
    ctx.setReadout(
      'lambda',
      statuses[3] === GPU_SPATIAL_ERROR_GM_STATUS_OK
        ? `λ ${formatNumber(errorSummary[GPU_SPATIAL_ERROR_GM_SUMMARY_LAMBDA])}; pseudo R² ${formatNumber(errorSummary[GPU_SPATIAL_ERROR_GM_SUMMARY_PSEUDO_R_SQUARED])}`
        : `fit failed (status ${statuses[3]})`
    );
    ctx.setReadout('status', statuses.join(' / '));

    // Residual clusters.
    const counts = [0, 0, 0, 0, 0];
    for (const quadrant of latest.quadrants) if (quadrant < 5) counts[quadrant]++;
    ctx.setReadout(
      'clusters',
      `${counts[1]} high-high / ${counts[3]} low-low / ${counts[2]} low-high / ${counts[4]} high-low`
    );
    ctx.requestLayers();
  };

  // ---- Parameters and dirtiness. ----
  let weightsDirty = true;
  let modelDirty = true;
  let activeWeights: WeightsKind = ctx.options.weights;
  let activeModel: ModelId = ctx.options.model;
  let activeOrder: '1' | '2' = ctx.options.instruments;
  const writeRidge = () => {
    const ridge = ctx.options.ridge;
    olsParameters.write(getGPUOrdinaryLeastSquaresParameterValues(ridge <= -2 ? 0 : 10 ** ridge));
  };
  const writeSignificance = () => {
    moranParameters.write(
      getGPUSpatialAutocorrelationParameterValues({significanceLevel: ctx.options.significance})
    );
  };
  writeRidge();
  writeSignificance();
  updateResponseRange();
  const activate = () => {
    activeWeights = ctx.options.weights;
    activeModel = ctx.options.model;
    activeOrder = ctx.options.instruments;
    getWeightsGraph(activeWeights);
    getVariant(activeModel).addLag(activeOrder);
  };
  activate();
  ctx.setStatus(`${formatInteger(rowCount)} Chicago tracts`);

  const dark = () => ctx.theme() === 'dark';

  return {
    getCompiledGraphs: () => {
      const variant = getVariant(activeModel);
      return [
        getWeightsGraph(activeWeights),
        variant.recipe,
        variant.lag.get(activeOrder)!,
        variant.error
      ];
    },

    setOption(id) {
      if (id === 'map') {
        ctx.requestLayers();
        return;
      }
      if (id === 'ridge') writeRidge();
      if (id === 'significance') writeSignificance();
      if (id === 'outcome') {
        responseBuffer.write(getResponse(ctx.options.outcome));
        updateResponseRange();
      }
      if (id === 'weights') {
        activate();
        weightsDirty = true;
      }
      if (id === 'model' || id === 'instruments') activate();
      modelDirty = true;
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      if (frame.frameIndex < 2) {
        weightsDirty = true;
        modelDirty = true;
      }
      const variant = getVariant(activeModel);
      if (weightsDirty) {
        getWeightsGraph(activeWeights).encode(commandEncoder, {parameters: undefined});
        weightsDirty = false;
        modelDirty = true;
      }
      if (modelDirty) {
        variant.recipe.encode(commandEncoder, {parameters: undefined});
        variant.lag.get(activeOrder)!.encode(commandEncoder, {parameters: undefined});
        variant.error.encode(commandEncoder, {parameters: undefined});
        variant.reader.request(commandEncoder);
        modelDirty = false;
      } else {
        variant.reader.flush(commandEncoder);
      }
    },

    getLayers() {
      const variant = getVariant(activeModel);
      const map: MapKind = ctx.options.map;
      const layers: Layer[] = [];
      const isDark = dark();
      layers.push(
        new PolygonFillLayer({
          id: 'health-excluded',
          triangles: excludedTrianglesBuffer,
          owners: excludedOwnersBuffer,
          triangleCount: allTriangles.triangleCount,
          colormap: 'uniform',
          color: isDark ? [90, 94, 104, 150] : [170, 172, 178, 150]
        })
      );
      const common = {
        triangles: trianglesBuffer,
        owners: ownersBuffer,
        triangleCount: triangles.triangleCount,
        opacity: 0.86,
        selectedRow
      };
      if (map === 'clusters') {
        layers.push(
          new PolygonFillLayer({
            ...common,
            id: 'health-clusters',
            values: variant.buffers.quadrants,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: RESIDUAL_CLUSTER_COLORS,
            noDataValue: NO_ROW
          })
        );
      } else if (map === 'fitted' || map === 'observed') {
        layers.push(
          new PolygonFillLayer({
            ...common,
            id: `health-${map}`,
            values: map === 'fitted' ? variant.buffers.fitted : responseBuffer,
            valueFormat: 'float32',
            colormap: 'ylorrd',
            valueRange: responseRange,
            color: [255, 255, 255, 255],
            noDataColor: [0, 0, 0, 0]
          })
        );
      } else {
        layers.push(
          new PolygonFillLayer({
            ...common,
            id: `health-${map}`,
            values:
              map === 'residuals'
                ? variant.buffers.olsResiduals
                : map === 'lag'
                  ? variant.buffers.lagResiduals
                  : variant.buffers.errorResiduals,
            valueFormat: 'float32',
            colormap: 'diverging',
            valueRange: [-2 * olsSigma, 2 * olsSigma],
            color: [255, 255, 255, 255]
          })
        );
      }
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'health-outline',
          coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
          segments: outlineBuffer,
          instanceCount: outlineSegments.length / 4,
          widthPixels: 0.8,
          color: isDark ? [225, 228, 238, 70] : [40, 44, 56, 90]
        })
      );
      return layers;
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      const row = locate(event.coordinate[0], event.coordinate[1]);
      if (row < 0 || !latest) return null;
      const feature = features[row];
      const response = getResponse(ctx.options.outcome)[row];
      const lines = [
        `Tract ${geoids[feature]}`,
        `${OUTCOME_LABELS[ctx.options.outcome]}: ${response.toFixed(1)}%`,
        `Fitted (OLS): ${latest.fitted[row].toFixed(1)}%`,
        `Residual: ${latest.olsResiduals[row] >= 0 ? '+' : ''}${latest.olsResiduals[row].toFixed(1)} points`,
        `Income per capita: $${formatInteger(income[feature])}`,
        `Poverty: ${rawPredictors.poverty[feature].toFixed(0)}%`
      ];
      if (ctx.options.map === 'lag')
        lines.push(`Lag-model residual: ${latest.lagResiduals[row].toFixed(1)}`);
      if (ctx.options.map === 'error')
        lines.push(`Error-model residual: ${latest.errorResiduals[row].toFixed(1)}`);
      return lines.join('\n');
    },

    onClick(event) {
      if (!event.coordinate) return false;
      const row = locate(event.coordinate[0], event.coordinate[1]);
      selectedRow = row === selectedRow ? -1 : row;
      ctx.requestLayers();
      return true;
    },

    destroy() {
      for (const variant of variants.values()) variant.reader.stop();
      resources.destroy();
    }
  };
}
