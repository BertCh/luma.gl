// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {COORDINATE_SYSTEM} from '@deck.gl/core';
import {
  getGPUGeographicallyWeightedRegressionParameterLength,
  getGPUGeographicallyWeightedRegressionParameterValues,
  getGPUPermutationParameterValues,
  GPUGeographicallyWeightedRegression,
  GPUGeographicallyWeightedRegressionNonstationarityTest,
  GPUOrdinaryLeastSquares,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY,
  GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH,
  GPU_GWR_NONSTATIONARITY_SUMMARY,
  GPU_GWR_NONSTATIONARITY_TABLE,
  GPU_GWR_NONSTATIONARITY_TABLE_STRIDE,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_R_SQUARED,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_RESIDUAL_SUM_OF_SQUARES,
  GPU_ORDINARY_LEAST_SQUARES_SUMMARY_ROW_COUNT,
  GPU_PERMUTATION_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {SpatialAnalysisResources} from '../../engine/resources';
import type {SceneContext, SceneInstance} from '../scene';
import {createNamedReader} from './b6-reader';
import {PolygonFillLayer} from './b6-fill-layer';
import {
  buildGeoJsonOutline,
  buildOutlineSegments,
  createFeatureLocator,
  formatInteger,
  formatNumber,
  formatP,
  getMoments,
  getQuantile,
  getSymmetricRange,
  NO_ROW,
  projectAlbers,
  projectRows,
  readPolygonGeometry,
  triangulatePolygonGeometry
} from './b6-polygons';

/** Option state of the local-relationships scene. */
export type LocalRelationshipsOptions = {
  map:
    | 'income'
    | 'density'
    | 'age65'
    | 'education'
    | 'intercept'
    | 'localR2'
    | 'condition'
    | 'influence'
    | 'residuals'
    | 'olsResiduals';
  center: 'zero' | 'global';
  kernel: 'bisquare' | 'gaussian';
  mode: 'adaptive' | 'fixed';
  bandwidth: number;
  subset: 'all' | 'metro' | 'nonmetro' | 'south';
  spatialIndex: boolean;
  permutations: number;
  seed: number;
  borders: boolean;
};

type MapKind = LocalRelationshipsOptions['map'];

/** Coefficient column of each coefficient map. */
const COEFFICIENT_COLUMNS: Partial<Record<MapKind, number>> = {
  intercept: 0,
  income: 1,
  density: 2,
  age65: 3,
  education: 4
};

/** Predictor labels in model order (the intercept comes first in every table). */
export const COUNTY_PREDICTOR_LABELS = [
  'ln median household income',
  'ln population density',
  'Age 65+ share',
  'No high-school diploma share'
] as const;

const PREDICTOR_COUNT = COUNTY_PREDICTOR_LABELS.length;
const COEFFICIENT_COUNT = PREDICTOR_COUNT + 1;
const LADDER_LENGTH = 8;
const MAXIMUM_NEIGHBORS = 128;
const MAXIMUM_PERMUTATIONS = 99;
/** Bandwidth ladders: nearest counties (adaptive) or meters (fixed). */
export const ADAPTIVE_LADDER = [24, 36, 48, 64, 80, 96, 112, 128];
export const FIXED_LADDER = [150e3, 250e3, 350e3, 500e3, 700e3, 1000e3, 1400e3, 2000e3];
const CONDITION_NUMBER_WARNING = 30;
/** FIPS state codes of the Census South region. */
const SOUTH_STATES = new Set([1, 5, 10, 11, 12, 13, 21, 22, 24, 28, 37, 40, 45, 47, 48, 51, 54]);

/**
 * Geographically weighted regression of county diabetes prevalence. Everything the contributors
 * need is written once into storage buffers; a bandwidth, kernel, mode or subset change is a
 * parameter or mask write that re-encodes the same compiled graph (`GPUGeographicallyWeightedRegression`
 * for the local fits, `GPUOrdinaryLeastSquares` for the national baseline, and the Monte Carlo
 * `GPUGeographicallyWeightedRegressionNonstationarityTest` run on demand). Local fits are written
 * to caller-owned buffers that `PolygonFillLayer` colors from directly; the columns are read back
 * once per change for the readouts and tooltips.
 */
export async function createLocalRelationships(
  ctx: SceneContext<LocalRelationshipsOptions>
): Promise<SceneInstance<LocalRelationshipsOptions>> {
  const counties = ctx.datasets.get('us-counties');
  const states = ctx.datasets.get('us-states');
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'local-relationships');

  // ---- Study set: counties with every variable finite. ----
  const geometry = readPolygonGeometry(counties);
  const featureCount = geometry.featureCount;
  const column = (name: string) => counties.column<Float32Array>(name);
  const diabetes = column('places_diabetes_ageAdj');
  const incomeRaw = column('medianHouseholdIncome');
  const densityRaw = column('popDensity');
  const age65Raw = column('age65plus');
  const educationRaw = column('noHighSchool');
  const rucc = column('rucc2023');
  const stateFips = counties.column<Uint8Array>('stateFips');
  const rowFeatures: number[] = [];
  for (let feature = 0; feature < featureCount; feature++) {
    if (
      Number.isFinite(diabetes[feature]) &&
      incomeRaw[feature] > 0 &&
      densityRaw[feature] > 0 &&
      Number.isFinite(age65Raw[feature]) &&
      Number.isFinite(educationRaw[feature])
    ) {
      rowFeatures.push(feature);
    }
  }
  const rowCount = rowFeatures.length;
  const rowOfFeature = new Uint32Array(featureCount).fill(NO_ROW);
  rowFeatures.forEach((feature, row) => {
    rowOfFeature[feature] = row;
  });
  const standardize = (raw: Float32Array, transform: (value: number) => number) => {
    const values = Float32Array.from(rowFeatures, feature => transform(raw[feature]));
    const {mean, deviation} = getMoments(values);
    return Float32Array.from(values, value => (value - mean) / deviation);
  };
  const predictorColumns = [
    standardize(incomeRaw, Math.log),
    standardize(densityRaw, Math.log),
    standardize(age65Raw, value => value),
    standardize(educationRaw, value => value)
  ];
  const predictorValues = new Float32Array(rowCount * PREDICTOR_COUNT);
  for (let row = 0; row < rowCount; row++) {
    for (let index = 0; index < PREDICTOR_COUNT; index++) {
      predictorValues[row * PREDICTOR_COUNT + index] = predictorColumns[index][row];
    }
  }
  const responseValues = Float32Array.from(rowFeatures, feature => diabetes[feature]);
  const centroidLngLat = counties.column<Float32Array>('centroid');
  const rowLngLat = new Float32Array(rowCount * 2);
  rowFeatures.forEach((feature, row) => {
    rowLngLat[row * 2] = centroidLngLat[feature * 2];
    rowLngLat[row * 2 + 1] = centroidLngLat[feature * 2 + 1];
  });
  const positionValues = projectRows(rowLngLat, projectAlbers);

  const metroMask = Uint32Array.from(rowFeatures, feature => (rucc[feature] <= 3 ? 1 : 0));
  const getMask = (subset: LocalRelationshipsOptions['subset']) =>
    Uint32Array.from(rowFeatures, (feature, row) =>
      subset === 'all'
        ? 1
        : subset === 'metro'
          ? metroMask[row]
          : subset === 'nonmetro'
            ? 1 - metroMask[row]
            : SOUTH_STATES.has(stateFips[feature])
              ? 1
              : 0
    );

  // ---- Geometry. ----
  const featureProperties = (counties.geojson?.features ?? []).map(
    feature => feature.properties as {name: string; state: string} | undefined
  );
  const triangles = triangulatePolygonGeometry(geometry, rowOfFeature);
  const everyTriangles = triangulatePolygonGeometry(geometry);
  const trianglesBuffer = resources.createBuffer('triangles', triangles.corners);
  const ownersBuffer = resources.createBuffer('owners', triangles.owners);
  const everyTrianglesBuffer = resources.createBuffer('every-triangles', everyTriangles.corners);
  const everyOwnersBuffer = resources.createBuffer('every-owners', everyTriangles.owners);
  const countyOutline = buildOutlineSegments(geometry);
  const countyOutlineBuffer = resources.createBuffer('county-outline', countyOutline);
  const stateOutline = states.geojson ? buildGeoJsonOutline(states.geojson) : new Float32Array(4);
  const stateOutlineBuffer = resources.createBuffer('state-outline', stateOutline);
  const locate = createFeatureLocator(geometry);

  // ---- Inputs and outputs. ----
  const positionsBuffer = resources.createBuffer('positions', positionValues);
  const predictorsBuffer = resources.createBuffer('predictors', predictorValues);
  const responseBuffer = resources.createBuffer('response', responseValues);
  const maskBuffer = resources.createBuffer('mask', getMask(ctx.options.subset));
  const coefficientsBuffer = resources.createBuffer(
    'coefficients',
    rowCount * COEFFICIENT_COUNT * 4
  );
  const localR2Buffer = resources.createBuffer('local-r2', rowCount * 4);
  const residualsBuffer = resources.createBuffer('residuals', rowCount * 4);
  const hatBuffer = resources.createBuffer('hat-diagonal', rowCount * 4);
  const statusBuffer = resources.createBuffer('local-status', rowCount * 4);
  const conditionBuffer = resources.createBuffer('condition-number', rowCount * 4);
  const scoresBuffer = resources.createBuffer('bandwidth-scores', LADDER_LENGTH * 4);
  const selectedBuffer = resources.createBuffer('selected-bandwidth', 8);
  const summaryBuffer = resources.createBuffer(
    'summary',
    GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH * 4
  );
  const parameters = resources.createParameterBuffer(
    'gwr-parameters',
    'float32',
    getGPUGeographicallyWeightedRegressionParameterLength(LADDER_LENGTH)
  );
  const olsCoefficients = resources.createBuffer('ols-coefficients', COEFFICIENT_COUNT * 4);
  const olsStandardErrors = resources.createBuffer('ols-standard-errors', COEFFICIENT_COUNT * 4);
  const olsTStatistics = resources.createBuffer('ols-t-statistics', COEFFICIENT_COUNT * 4);
  const olsSummary = resources.createBuffer(
    'ols-summary',
    GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH * 4
  );
  const olsStatus = resources.createBuffer('ols-status', 4);
  const olsResiduals = resources.createBuffer('ols-residuals', rowCount * 4);
  const olsFitted = resources.createBuffer('ols-fitted', rowCount * 4);
  const testTableLength = COEFFICIENT_COUNT * GPU_GWR_NONSTATIONARITY_TABLE_STRIDE;
  const testTable = resources.createBuffer('test-table', testTableLength * 4);
  const testSummary = resources.createBuffer(
    'test-summary',
    GPU_GWR_NONSTATIONARITY_SUMMARY.length * 4
  );
  const permutationParameters = resources.createParameterBuffer(
    'permutation-parameters',
    'uint32',
    GPU_PERMUTATION_PARAMETER_LENGTH
  );

  // ---- Graphs. The grid-indexed variant is the default; the scan variant compiles on demand. ----
  const gwrGraphs = new Map<boolean, CompiledGPUCommandGraph<void>>();
  const getGwrGraph = (indexed: boolean) => {
    let compiled = gwrGraphs.get(indexed);
    if (compiled) return compiled;
    const graph = new GPUCommandGraph<void>(device, {id: `gwr-${indexed ? 'indexed' : 'scan'}`});
    graph.add(
      new GPUGeographicallyWeightedRegression({
        id: 'gwr',
        positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', rowCount),
        predictors: importGraphBuffer(
          graph,
          'predictors',
          predictorsBuffer,
          'float32',
          rowCount * PREDICTOR_COUNT
        ),
        predictorCount: PREDICTOR_COUNT,
        response: importGraphBuffer(graph, 'response', responseBuffer, 'float32', rowCount),
        mask: importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', rowCount),
        parameters: parameters.importToGraph(graph),
        maximumBandwidthCount: LADDER_LENGTH,
        maximumNeighborCount: MAXIMUM_NEIGHBORS,
        indexGridSize: indexed ? undefined : false,
        output: {
          coefficients: importGraphBuffer(
            graph,
            'coefficients',
            coefficientsBuffer,
            'float32',
            rowCount * COEFFICIENT_COUNT
          ),
          localR2: importGraphBuffer(graph, 'local-r2', localR2Buffer, 'float32', rowCount),
          residuals: importGraphBuffer(graph, 'residuals', residualsBuffer, 'float32', rowCount),
          hatDiagonal: importGraphBuffer(graph, 'hat-diagonal', hatBuffer, 'float32', rowCount),
          localStatus: importGraphBuffer(graph, 'local-status', statusBuffer, 'uint32', rowCount),
          localConditionNumber: importGraphBuffer(
            graph,
            'condition-number',
            conditionBuffer,
            'float32',
            rowCount
          ),
          bandwidthScores: importGraphBuffer(
            graph,
            'bandwidth-scores',
            scoresBuffer,
            'float32',
            LADDER_LENGTH
          ),
          selectedBandwidth: importGraphBuffer(
            graph,
            'selected-bandwidth',
            selectedBuffer,
            'float32',
            2
          ),
          summary: importGraphBuffer(
            graph,
            'summary',
            summaryBuffer,
            'float32',
            GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH
          )
        }
      })
    );
    compiled = resources.track(graph.compile());
    gwrGraphs.set(indexed, compiled);
    return compiled;
  };

  const olsGraph = new GPUCommandGraph<void>(device, {id: 'gwr-ols-baseline'});
  olsGraph.add(
    new GPUOrdinaryLeastSquares({
      id: 'ols',
      predictors: importGraphBuffer(
        olsGraph,
        'predictors',
        predictorsBuffer,
        'float32',
        rowCount * PREDICTOR_COUNT
      ),
      response: importGraphBuffer(olsGraph, 'response', responseBuffer, 'float32', rowCount),
      mask: importGraphBuffer(olsGraph, 'mask', maskBuffer, 'uint32', rowCount),
      predictorCount: PREDICTOR_COUNT,
      output: {
        coefficients: importGraphBuffer(
          olsGraph,
          'ols-coefficients',
          olsCoefficients,
          'float32',
          COEFFICIENT_COUNT
        ),
        standardErrors: importGraphBuffer(
          olsGraph,
          'ols-standard-errors',
          olsStandardErrors,
          'float32',
          COEFFICIENT_COUNT
        ),
        tStatistics: importGraphBuffer(
          olsGraph,
          'ols-t-statistics',
          olsTStatistics,
          'float32',
          COEFFICIENT_COUNT
        ),
        summary: importGraphBuffer(
          olsGraph,
          'ols-summary',
          olsSummary,
          'float32',
          GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH
        ),
        status: importGraphBuffer(olsGraph, 'ols-status', olsStatus, 'uint32', 1),
        residuals: importGraphBuffer(olsGraph, 'ols-residuals', olsResiduals, 'float32', rowCount),
        fitted: importGraphBuffer(olsGraph, 'ols-fitted', olsFitted, 'float32', rowCount)
      }
    })
  );
  const compiledOls = resources.track(olsGraph.compile());

  const testGraph = new GPUCommandGraph<void>(device, {id: 'gwr-nonstationarity-test'});
  let testError: string | null = null;
  try {
    testGraph.add(
      new GPUGeographicallyWeightedRegressionNonstationarityTest({
        id: 'gwr-test',
        positions: importGraphBuffer(
          testGraph,
          'positions',
          positionsBuffer,
          'float32x2',
          rowCount
        ),
        predictors: importGraphBuffer(
          testGraph,
          'predictors',
          predictorsBuffer,
          'float32',
          rowCount * PREDICTOR_COUNT
        ),
        predictorCount: PREDICTOR_COUNT,
        response: importGraphBuffer(testGraph, 'response', responseBuffer, 'float32', rowCount),
        mask: importGraphBuffer(testGraph, 'mask', maskBuffer, 'uint32', rowCount),
        bandwidthParameters: parameters.importToGraph(testGraph),
        selectedBandwidth: importGraphBuffer(
          testGraph,
          'selected-bandwidth',
          selectedBuffer,
          'float32',
          2
        ),
        coefficients: importGraphBuffer(
          testGraph,
          'coefficients',
          coefficientsBuffer,
          'float32',
          rowCount * COEFFICIENT_COUNT
        ),
        parameters: permutationParameters.importToGraph(testGraph),
        maximumPermutations: MAXIMUM_PERMUTATIONS,
        maximumBandwidthCount: LADDER_LENGTH,
        maximumNeighborCount: MAXIMUM_NEIGHBORS,
        output: {
          table: importGraphBuffer(testGraph, 'test-table', testTable, 'float32', testTableLength),
          summary: importGraphBuffer(
            testGraph,
            'test-summary',
            testSummary,
            'float32',
            GPU_GWR_NONSTATIONARITY_SUMMARY.length
          )
        }
      })
    );
  } catch (error) {
    testError = error instanceof Error ? error.message : String(error);
  }
  const compiledTest = testError ? null : resources.track(testGraph.compile());

  // ---- State. ----
  // The grid index adds storage bindings to the fit kernel; a default 8-binding device cannot run it.
  const indexSupported = (device.limits.maxStorageBuffersPerShaderStage ?? 8) >= 10;
  let activeIndexed = ctx.options.spatialIndex && indexSupported;
  let dirty = true;
  let olsDirty = true;
  let testRequested = false;
  let testValid = false;
  let center = 0;
  let rangeValues: [number, number] = [-1, 1];
  let localRange: [number, number] = [0, 1];
  let latest: {
    coefficients: Float32Array;
    localR2: Float32Array;
    residuals: Float32Array;
    olsResiduals: Float32Array;
    olsCoefficients: Float32Array;
    selected: number;
    mode: 'adaptive' | 'fixed';
  } | null = null;

  const getLadder = () => (ctx.options.mode === 'adaptive' ? ADAPTIVE_LADDER : FIXED_LADDER);
  const writeParameters = () => {
    const {kernel, mode, bandwidth} = ctx.options;
    const ladder = getLadder();
    parameters.write(
      getGPUGeographicallyWeightedRegressionParameterValues(
        {
          kernel,
          bandwidthMode: mode,
          bandwidths: bandwidth <= 0 ? ladder : [ladder[Math.min(bandwidth, LADDER_LENGTH) - 1]]
        },
        LADDER_LENGTH
      )
    );
    dirty = true;
    testValid = false;
  };
  const writePermutations = () => {
    permutationParameters.write(
      getGPUPermutationParameterValues({
        seed: ctx.options.seed,
        permutations: ctx.options.permutations
      })
    );
    testValid = false;
  };
  writeParameters();
  writePermutations();
  getGwrGraph(activeIndexed);
  ctx.setReadout(
    'index',
    activeIndexed ? 'grid-indexed (fixed bisquare visits nearby cells only)' : 'full scan'
  );
  ctx.setReadout('rows', `${formatInteger(rowCount)} of ${formatInteger(featureCount)} counties`);
  ctx.setReadout('test', 'press “Run the Monte Carlo test”');
  ctx.setStatus(`${formatInteger(rowCount)} counties`);

  const updateRanges = () => {
    if (!latest) return;
    const {map} = ctx.options;
    const columnIndex = COEFFICIENT_COLUMNS[map];
    if (columnIndex !== undefined) {
      const values = new Float32Array(rowCount);
      for (let row = 0; row < rowCount; row++)
        values[row] = latest.coefficients[row * COEFFICIENT_COUNT + columnIndex];
      center = ctx.options.center === 'global' ? latest.olsCoefficients[columnIndex] : 0;
      const lower = getQuantile(values, 0.03) - center;
      const upper = getQuantile(values, 0.97) - center;
      const bound = Math.max(1e-6, Math.abs(lower), Math.abs(upper));
      rangeValues = [center - bound, center + bound];
      ctx.setLegendExtent('coefficient', rangeValues);
      ctx.setReadout(
        'spread',
        `${formatNumber(getQuantile(values, 0.05), 2)} to ${formatNumber(getQuantile(values, 0.95), 2)} (national OLS ${formatNumber(latest.olsCoefficients[columnIndex], 2)})`
      );
    } else if (map === 'residuals' || map === 'olsResiduals') {
      const source = map === 'residuals' ? latest.residuals : latest.olsResiduals;
      rangeValues = getSymmetricRange(source, 0.03, 0.97);
      ctx.setLegendExtent('residual', rangeValues);
    }
  };

  const reader = createNamedReader(
    resources,
    'gwr',
    [
      {
        name: 'summary',
        buffer: summaryBuffer,
        bytes: GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH * 4
      },
      {name: 'selected', buffer: selectedBuffer, bytes: 8},
      {name: 'scores', buffer: scoresBuffer, bytes: LADDER_LENGTH * 4},
      {name: 'coefficients', buffer: coefficientsBuffer, bytes: rowCount * COEFFICIENT_COUNT * 4},
      {name: 'localR2', buffer: localR2Buffer, bytes: rowCount * 4},
      {name: 'residuals', buffer: residualsBuffer, bytes: rowCount * 4},
      {name: 'status', buffer: statusBuffer, bytes: rowCount * 4},
      {name: 'condition', buffer: conditionBuffer, bytes: rowCount * 4},
      {name: 'hat', buffer: hatBuffer, bytes: rowCount * 4},
      {name: 'olsCoefficients', buffer: olsCoefficients, bytes: COEFFICIENT_COUNT * 4},
      {name: 'olsTStatistics', buffer: olsTStatistics, bytes: COEFFICIENT_COUNT * 4},
      {
        name: 'olsSummary',
        buffer: olsSummary,
        bytes: GPU_ORDINARY_LEAST_SQUARES_SUMMARY_LENGTH * 4
      },
      {name: 'olsResiduals', buffer: olsResiduals, bytes: rowCount * 4},
      {name: 'testTable', buffer: testTable, bytes: testTableLength * 4},
      {name: 'testSummary', buffer: testSummary, bytes: GPU_GWR_NONSTATIONARITY_SUMMARY.length * 4}
    ],
    get => {
      const summary = get('summary').f32;
      const selected = get('selected').f32;
      const scores = get('scores').f32;
      const coefficients = get('coefficients').f32;
      const status = get('status').u32;
      const condition = get('condition').f32;
      const olsSummaryValues = get('olsSummary').f32;
      const olsCoefficientValues = get('olsCoefficients').f32;
      const olsT = get('olsTStatistics').f32;
      const slot = GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY;
      latest = {
        coefficients,
        localR2: get('localR2').f32,
        residuals: get('residuals').f32,
        olsResiduals: get('olsResiduals').f32,
        olsCoefficients: olsCoefficientValues,
        selected: selected[1],
        mode: ctx.options.mode
      };
      const included = olsSummaryValues[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_ROW_COUNT];
      let singular = 0;
      let excluded = 0;
      for (let row = 0; row < rowCount; row++) {
        if (status[row] === GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.SINGULAR) singular++;
        else if (status[row] === GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_STATUS.EXCLUDED) excluded++;
      }
      const valid = summary[slot.HAS_VALID_CANDIDATE] !== 0;
      const unit = ctx.options.mode === 'adaptive' ? 'nearest counties' : 'km';
      const shown = ctx.options.mode === 'adaptive' ? selected[1] : selected[1] / 1000;
      ctx.setReadout(
        'bandwidth',
        valid
          ? `${formatNumber(shown, 0)} ${unit} (ladder step ${selected[0] + 1})`
          : 'no valid candidate (all singular)'
      );
      const ladderCount = ctx.options.bandwidth <= 0 ? LADDER_LENGTH : 1;
      ctx.setReadout(
        'scores',
        Array.from(scores)
          .slice(0, ladderCount)
          .map(value => (Number.isFinite(value) ? value.toFixed(0) : 'invalid'))
          .join(' / ')
      );
      const aicc = summary[slot.AICC];
      // OLS AICc with the same formula as the GWR (its hat trace is the parameter count).
      const parameterCount = COEFFICIENT_COUNT;
      const rss = olsSummaryValues[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_RESIDUAL_SUM_OF_SQUARES];
      const olsAicc =
        included * Math.log(rss / included) +
        included * Math.log(2 * Math.PI) +
        (included * (included + parameterCount)) / (included - 2 - parameterCount);
      ctx.setReadout(
        'aicc',
        `GWR ${formatNumber(aicc, 0)} vs OLS ${formatNumber(olsAicc, 0)} (${aicc < olsAicc ? 'GWR better by' : 'OLS better by'} ${formatNumber(Math.abs(olsAicc - aicc), 0)})`
      );
      ctx.setReadout(
        'rSquared',
        `GWR ${formatNumber(summary[slot.R_SQUARED])} vs OLS ${formatNumber(olsSummaryValues[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_R_SQUARED])}`
      );
      ctx.setReadout(
        'trace',
        `${formatNumber(summary[slot.TRACE_OF_HAT], 1)} (OLS: ${COEFFICIENT_COUNT})`
      );
      ctx.setReadout(
        'singular',
        `${formatInteger(singular)} singular, ${formatInteger(excluded)} excluded of ${formatInteger(rowCount)}`
      );
      const finiteCondition = Array.from(condition).filter(Number.isFinite);
      const flagged = finiteCondition.filter(value => value > CONDITION_NUMBER_WARNING).length;
      ctx.setReadout(
        'condition',
        finiteCondition.length
          ? `median ${formatNumber(getQuantile(finiteCondition, 0.5), 1)}, max ${formatNumber(getQuantile(finiteCondition, 1), 0)}, ${formatInteger(flagged)} above ${CONDITION_NUMBER_WARNING}`
          : 'n/a'
      );
      ctx.setReadout(
        'globalCoefficients',
        COUNTY_PREDICTOR_LABELS.map(
          (label, index) =>
            `${label} ${formatNumber(olsCoefficientValues[index + 1], 2)} (t ${formatNumber(olsT[index + 1], 1)})`
        ).join('; ')
      );
      const hat = get('hat').f32;
      const finiteHat = Array.from(hat).filter(Number.isFinite);
      localRange = [0, Math.max(1e-6, getQuantile(finiteHat, 0.98))];
      // Monte Carlo test.
      const testSummaryValues = get('testSummary').f32;
      const testValues = get('testTable').f32;
      const permutationCount = testSummaryValues[GPU_GWR_NONSTATIONARITY_SUMMARY.permutations];
      if (testValid && permutationCount > 0) {
        const labels = ['Intercept', ...COUNTY_PREDICTOR_LABELS];
        const parts = labels.map((label, index) => {
          const base = index * GPU_GWR_NONSTATIONARITY_TABLE_STRIDE;
          return `${label}: ${formatP(testValues[base + GPU_GWR_NONSTATIONARITY_TABLE.pseudoPValue])} (spread ${formatNumber(testValues[base + GPU_GWR_NONSTATIONARITY_TABLE.observedStandardDeviation], 2)} vs ${formatNumber(testValues[base + GPU_GWR_NONSTATIONARITY_TABLE.simulatedMean], 2)} by chance)`;
        });
        ctx.setReadout(
          'test',
          `${permutationCount} permutations; ${formatInteger(testSummaryValues[GPU_GWR_NONSTATIONARITY_SUMMARY.failedFitCount])} singular refits`
        );
        parts.forEach((part, index) => ctx.setReadout(`test${index}`, part));
      }
      updateRanges();
      ctx.requestLayers();
    }
  );

  const dark = () => ctx.theme() === 'dark';

  return {
    getCompiledGraphs: () => [
      getGwrGraph(activeIndexed),
      compiledOls,
      ...(compiledTest ? [compiledTest] : [])
    ],

    setOption(id) {
      if (id === 'map' || id === 'center' || id === 'borders') {
        updateRanges();
        ctx.requestLayers();
        return;
      }
      if (id === 'subset') {
        maskBuffer.write(getMask(ctx.options.subset));
        olsDirty = true;
        writeParameters();
        return;
      }
      if (id === 'spatialIndex') {
        activeIndexed = ctx.options.spatialIndex && indexSupported;
        ctx.setReadout(
          'index',
          ctx.options.spatialIndex && !indexSupported
            ? `unavailable: this device allows ${device.limits.maxStorageBuffersPerShaderStage} storage buffers per stage, the indexed kernel needs 10`
            : activeIndexed
              ? 'grid-indexed (fixed bisquare visits nearby cells only)'
              : 'full scan'
        );
        getGwrGraph(activeIndexed);
        dirty = true;
        return;
      }
      if (id === 'permutations' || id === 'seed') {
        writePermutations();
        return;
      }
      writeParameters();
    },

    onAction(id) {
      if (id === 'runTest') {
        if (!compiledTest) {
          ctx.setReadout('test', `unavailable on this device: ${testError}`);
          return;
        }
        testRequested = true;
        ctx.setReadout('test', 'running…');
      }
    },

    encode(commandEncoder, frame) {
      if (frame.frameIndex < 2) {
        dirty = true;
        olsDirty = true;
      }
      if (dirty || olsDirty || testRequested) {
        if (olsDirty) {
          compiledOls.encode(commandEncoder, {parameters: undefined});
          olsDirty = false;
        }
        if (dirty) {
          getGwrGraph(activeIndexed).encode(commandEncoder, {parameters: undefined});
          dirty = false;
        }
        if (testRequested && compiledTest) {
          compiledTest.encode(commandEncoder, {parameters: undefined});
          testRequested = false;
          testValid = true;
        }
        reader.request(commandEncoder);
      } else {
        reader.flush(commandEncoder);
      }
    },

    getLayers() {
      const {map} = ctx.options;
      const isDark = dark();
      const layers: Layer[] = [
        new PolygonFillLayer({
          id: 'counties-base',
          triangles: everyTrianglesBuffer,
          owners: everyOwnersBuffer,
          triangleCount: everyTriangles.triangleCount,
          colormap: 'uniform',
          color: isDark ? [88, 92, 102, 170] : [182, 184, 190, 170]
        })
      ];
      const common = {
        triangles: trianglesBuffer,
        owners: ownersBuffer,
        triangleCount: triangles.triangleCount,
        opacity: 0.9,
        valueFormat: 'float32' as const,
        color: [255, 255, 255, 255] as const,
        noDataColor: [0, 0, 0, 0] as const
      };
      const columnIndex = COEFFICIENT_COLUMNS[map];
      if (columnIndex !== undefined) {
        layers.push(
          new PolygonFillLayer({
            ...common,
            id: 'counties-coefficient',
            values: coefficientsBuffer,
            valueStride: COEFFICIENT_COUNT,
            valueOffset: columnIndex,
            colormap: 'diverging',
            valueRange: rangeValues
          })
        );
      } else if (map === 'localR2') {
        layers.push(
          new PolygonFillLayer({
            ...common,
            id: 'counties-r2',
            values: localR2Buffer,
            colormap: 'blues',
            valueRange: [0, 1]
          })
        );
      } else if (map === 'condition') {
        layers.push(
          new PolygonFillLayer({
            ...common,
            id: 'counties-condition',
            values: conditionBuffer,
            colormap: 'magma',
            valueRange: [1, CONDITION_NUMBER_WARNING]
          })
        );
      } else if (map === 'influence') {
        layers.push(
          new PolygonFillLayer({
            ...common,
            id: 'counties-hat',
            values: hatBuffer,
            colormap: 'inferno',
            valueRange: localRange
          })
        );
      } else {
        layers.push(
          new PolygonFillLayer({
            ...common,
            id: `counties-${map}`,
            values: map === 'residuals' ? residualsBuffer : olsResiduals,
            colormap: 'diverging',
            valueRange: rangeValues
          })
        );
      }
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'counties-outline',
          coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
          segments: countyOutlineBuffer,
          instanceCount: countyOutline.length / 4,
          widthPixels: 0.5,
          color: isDark ? [225, 228, 238, 28] : [40, 44, 56, 40]
        })
      );
      if (ctx.options.borders) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'states-outline',
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            segments: stateOutlineBuffer,
            instanceCount: stateOutline.length / 4,
            widthPixels: 1.2,
            color: isDark ? [235, 238, 248, 150] : [30, 34, 46, 160]
          })
        );
      }
      return layers;
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip(event) {
      if (!event.coordinate || !latest) return null;
      const feature = locate(event.coordinate[0], event.coordinate[1]);
      if (feature < 0) return null;
      const properties = featureProperties[feature];
      const row = rowOfFeature[feature];
      const title = `${properties?.name ?? 'County'}, ${properties?.state ?? ''}`;
      if (row === NO_ROW) return `${title}\n(not in the model: missing data)`;
      const local = (index: number) => latest!.coefficients[row * COEFFICIENT_COUNT + index];
      const lines = [
        title,
        `Diabetes ${diabetes[feature].toFixed(1)}%, income $${formatInteger(incomeRaw[feature])}`,
        `Local effect of income: ${Number.isFinite(local(1)) ? local(1).toFixed(2) : 'n/a'} (national ${latest.olsCoefficients[1].toFixed(2)})`,
        `Local R²: ${Number.isFinite(latest.localR2[row]) ? latest.localR2[row].toFixed(2) : 'n/a'}`
      ];
      return lines.join('\n');
    },

    destroy() {
      reader.stop();
      resources.destroy();
    }
  };
}
