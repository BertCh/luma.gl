// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Spatial regression and composite indicators on a Manhattan grid, as four views of one mode.
 *
 * The study grid is built once on the CPU from the catalog datasets: 250 m cells that contain a
 * road. Per cell, the response is the log of the POI count and the explanatory columns are the
 * road length, the log of the taxi-trip vertex count and the distance to Times Square. Included
 * cells are compacted into rows, so every recipe sees dense float32 columns, and a cell-to-row map
 * lets the map layer draw per-row results without repacking.
 *
 * - OLS: `GPUOrdinaryLeastSquares` (coefficients, R2, AIC, Jarque-Bera, Breusch-Pagan, residual
 *   map); the ridge penalty is a per-frame parameter.
 * - GWR: `GPUGeographicallyWeightedRegression` with an AICc bandwidth ladder compiled once; kernel,
 *   bandwidth mode, the choice between "auto (AICc)" and one ladder value, and the plotted
 *   coefficient are buffer writes or layer props. A single-ladder-value run is encoded when an
 *   input changes, not every frame (brute force, `O(n^2 * ladder)`).
 * - Composite: `GPUCompositeScore` with live weight, direction, scaler and aggregation controls.
 * - Inequality: `GPUInequality` per latitude band, with a Lorenz curve chart drawn from the
 *   knots read back (a small summary).
 *
 * All four graphs of the active view are compiled once in `create`; the view select re-enters the
 * mode, so each view has its own controls (the footer rebuild counter restarts at 0). Results
 * persist in the output buffers, so graphs are encoded only when an input changed.
 */

import type {Layer} from '@deck.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getGPUCompositeScoreParameterValues, getGPUInequalityParameterValues, GPUCompositeScore, GPUInequality, GPU_COMPOSITE_SCORE_PARAMETER_LENGTH, GPU_INEQUALITY_GLOBAL_SUMMARY, GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH, GPU_INEQUALITY_PARAMETER_LENGTH, type GPUCompositeScoreAggregation, type GPUCompositeScoreScaler} from '@luma.gl/experimental/gpu-dataframe';
import {getGPUGeographicallyWeightedRegressionParameterValues, getGPUOrdinaryLeastSquaresParameterValues, GPUGeographicallyWeightedRegression, GPUOrdinaryLeastSquares, GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY, GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH, GPU_ORDINARY_LEAST_SQUARES_PARAMETER_LENGTH, GPU_ORDINARY_LEAST_SQUARES_SUMMARY_ADJUSTED_R_SQUARED, GPU_ORDINARY_LEAST_SQUARES_SUMMARY_AIC, GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BIC, GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BREUSCH_PAGAN, GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BREUSCH_PAGAN_P_VALUE, GPU_ORDINARY_LEAST_SQUARES_SUMMARY_JARQUE_BERA, GPU_ORDINARY_LEAST_SQUARES_SUMMARY_JARQUE_BERA_P_VALUE, GPU_ORDINARY_LEAST_SQUARES_SUMMARY_R_SQUARED, GPU_ORDINARY_LEAST_SQUARES_SUMMARY_RESIDUAL_SUM_OF_SQUARES, GPU_ORDINARY_LEAST_SQUARES_SUMMARY_ROW_COUNT, GPU_ORDINARY_LEAST_SQUARES_SUMMARY_SIGMA_SQUARED, getGPUGeographicallyWeightedRegressionParameterLength} from '@luma.gl/experimental/geospatial';
import {importGraphBuffer, type GPUGeographicallyWeightedRegressionBandwidthMode, type GPUGeographicallyWeightedRegressionKernel} from '@luma.gl/experimental/UNRESOLVED';
import {
  LocalMetricProjection,
  type MapGraphsPointsOfInterest,
  type MapGraphsRoadNetwork,
  type MapGraphsTrips
} from '../map-graphs-data';
import type {
  MapGraphsModeContext,
  MapGraphsModeDefinition,
  MapGraphsModeInstance,
  MapGraphsViewState
} from '../map-graphs-mode';
import {formatCount, MapGraphsResources} from '../map-graphs-resources';
import {CellGridLayer} from './regression-layers';
import {SummaryReader} from './summary-reader';

type RegressionView = 'ols' | 'gwr' | 'composite' | 'inequality';

const VIEWS: readonly {id: RegressionView; label: string}[] = [
  {id: 'ols', label: 'Ordinary least squares (GPUOrdinaryLeastSquares)'},
  {id: 'gwr', label: 'Geographically weighted (GPUGeographicallyWeightedRegression)'},
  {id: 'composite', label: 'Composite score (GPUCompositeScore)'},
  {id: 'inequality', label: 'Inequality by band (GPUInequality)'}
];

const VIEW_STATE: MapGraphsViewState = {longitude: -73.97, latitude: 40.775, zoom: 11.6};

/** Reads the `view` URL parameter once at load. */
function readViewFromUrl(): RegressionView {
  const parameter =
    typeof window === 'undefined' ? null : new URLSearchParams(window.location.search).get('view');
  return VIEWS.find(entry => entry.id === parameter)?.id ?? 'ols';
}

/** View chosen by the last navigation, or by `?view=` at load. */
let requestedView: RegressionView = readViewFromUrl();

/** Switches view by re-entering the mode, which rebuilds the controls for that view. */
function navigate(view: RegressionView): void {
  requestedView = view;
  globalThis.mapGraphsExplorer?.selectMode('regression');
}

/** Frames between summary readbacks when nothing changed. */
const READBACK_INTERVAL_FRAMES = 60;
const NO_ROW = 0xffffffff;
const CELL_SIZE_METERS = 250;
const MAXIMUM_CELL_COUNT = 5000;
const MINIMUM_ROAD_KILOMETERS = 0.03;
const BAND_COUNT = 6;
const LORENZ_KNOT_COUNT = 21;
const PREDICTOR_NAMES = ['Road length (km)', 'Taxi activity (log)', 'Distance to Times Sq (km)'];
const BAND_COLORS: readonly (readonly [number, number, number])[] = [
  [78, 201, 255],
  [255, 148, 72],
  [189, 122, 255],
  [87, 235, 168],
  [255, 105, 168],
  [245, 220, 87]
];

/**
 * Spatial regression, geographically weighted regression, composite indicators and inequality on
 * a grid of Manhattan cells. See the file comment.
 */
export const regressionMode: MapGraphsModeDefinition = {
  id: 'regression',
  title: 'Regression',
  recipes: [
    'GPUOrdinaryLeastSquares',
    'GPUGeographicallyWeightedRegression',
    'GPUCompositeScore',
    'GPUInequality'
  ],
  description:
    'Why do some Manhattan cells hold more points of interest? Global OLS, locally varying GWR ' +
    'coefficients, a weight-slider composite index and Lorenz curves, all computed on the GPU ' +
    'from road, taxi and POI columns. Every control rewrites a parameter or input buffer.',
  initialViewState: VIEW_STATE,

  async create(context) {
    const view = requestedView;
    const [roads, trips, pois] = await Promise.all([
      context.data.getNewYorkRoads(),
      context.data.getNewYorkTrips(),
      context.data.getNewYorkPointsOfInterest()
    ]);
    context.signal.throwIfAborted();
    const study = buildCellStudy(roads, trips, pois);
    const resources = new MapGraphsResources(context.device, `regression-${view}`);
    context.controls.addSelect<RegressionView>({
      label: 'View (re-enters the mode with this view’s controls)',
      options: VIEWS.map(entry => ({value: entry.id, label: entry.label})),
      value: view,
      onChange: navigate
    });
    let instance: Omit<MapGraphsModeInstance, 'destroy'>;
    switch (view) {
      case 'ols':
        instance = createOlsView(context, resources, study);
        break;
      case 'gwr':
        instance = createGwrView(context, resources, study);
        break;
      case 'composite':
        instance = createCompositeView(context, resources, study);
        break;
      case 'inequality':
        instance = createInequalityView(context, resources, study);
        break;
    }
    return {...instance, destroy: () => resources.destroy()};
  }
};

// ---------------------------------------------------------------------------------------------
// Study grid
// ---------------------------------------------------------------------------------------------

/** Compacted study grid: one row per included cell. */
type CellStudy = {
  attribution: string;
  origin: readonly [number, number];
  /** Grid south-west corner in meters. */
  gridOrigin: readonly [number, number];
  cellSize: number;
  columns: number;
  rows: number;
  cellCount: number;
  rowCount: number;
  /** Row of every cell, or `NO_ROW`. */
  rowOfCell: Uint32Array;
  /** Latitude band of every cell, or `NO_ROW`. */
  bandOfCell: Uint32Array;
  /** `x, y` meter centers per row. */
  positions: Float32Array;
  /** Row-major `row * 3 + column`: road km, log taxi vertices, distance to Times Square km. */
  predictors: Float32Array;
  /** log(1 + POI count) per row. */
  response: Float32Array;
  /** Counts and measures per row. */
  poiCounts: Float32Array;
  taxiCounts: Float32Array;
  roadKilometers: Float32Array;
  intersections: Float32Array;
  /** Latitude band per row. */
  bands: Uint32Array;
};

function buildCellStudy(
  roads: MapGraphsRoadNetwork,
  trips: MapGraphsTrips,
  pois: MapGraphsPointsOfInterest
): CellStudy {
  const projection = new LocalMetricProjection(roads.origin);
  const [centerX, centerY] = projection.project(-73.9855, 40.758);
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let node = 0; node < roads.nodePositions.length / 2; node++) {
    minX = Math.min(minX, roads.nodePositions[node * 2]);
    maxX = Math.max(maxX, roads.nodePositions[node * 2]);
    minY = Math.min(minY, roads.nodePositions[node * 2 + 1]);
    maxY = Math.max(maxY, roads.nodePositions[node * 2 + 1]);
  }
  let cellSize = CELL_SIZE_METERS;
  while (
    Math.ceil((maxX - minX) / cellSize) * Math.ceil((maxY - minY) / cellSize) >
    MAXIMUM_CELL_COUNT
  ) {
    cellSize += 50;
  }
  const columns = Math.max(1, Math.ceil((maxX - minX) / cellSize));
  const rows = Math.max(1, Math.ceil((maxY - minY) / cellSize));
  const cellCount = columns * rows;
  const getCell = (x: number, y: number) => {
    const column = Math.floor((x - minX) / cellSize);
    const row = Math.floor((y - minY) / cellSize);
    return column >= 0 && column < columns && row >= 0 && row < rows ? row * columns + column : -1;
  };
  const roadKilometers = new Float32Array(cellCount);
  const intersections = new Float32Array(cellCount);
  const poiCounts = new Float32Array(cellCount);
  const taxiCounts = new Float32Array(cellCount);
  for (let segment = 0; segment < roads.segments.length / 4; segment++) {
    const [x0, y0, x1, y1] = roads.segments.subarray(segment * 4, segment * 4 + 4);
    const cell = getCell((x0 + x1) / 2, (y0 + y1) / 2);
    if (cell >= 0) roadKilometers[cell] += Math.hypot(x1 - x0, y1 - y0) / 1000;
  }
  const degrees = new Uint32Array(roads.nodePositions.length / 2);
  for (const source of roads.edgeSources) degrees[source]++;
  for (let node = 0; node < degrees.length; node++) {
    if (degrees[node] >= 3) {
      const cell = getCell(roads.nodePositions[node * 2], roads.nodePositions[node * 2 + 1]);
      if (cell >= 0) intersections[cell]++;
    }
  }
  for (let point = 0; point < pois.positions.length / 2; point++) {
    const cell = getCell(pois.positions[point * 2], pois.positions[point * 2 + 1]);
    if (cell >= 0) poiCounts[cell]++;
  }
  for (let vertex = 0; vertex < trips.vertexPositions.length / 2; vertex++) {
    const cell = getCell(trips.vertexPositions[vertex * 2], trips.vertexPositions[vertex * 2 + 1]);
    if (cell >= 0) taxiCounts[cell]++;
  }

  const includedCells: number[] = [];
  for (let cell = 0; cell < cellCount; cell++) {
    if (roadKilometers[cell] >= MINIMUM_ROAD_KILOMETERS) includedCells.push(cell);
  }
  const rowCount = includedCells.length;
  const rowOfCell = new Uint32Array(cellCount).fill(NO_ROW);
  const bandOfCell = new Uint32Array(cellCount).fill(NO_ROW);
  const positions = new Float32Array(rowCount * 2);
  const predictors = new Float32Array(rowCount * 3);
  const response = new Float32Array(rowCount);
  const rowPoiCounts = new Float32Array(rowCount);
  const rowTaxiCounts = new Float32Array(rowCount);
  const rowRoadKilometers = new Float32Array(rowCount);
  const rowIntersections = new Float32Array(rowCount);
  const bands = new Uint32Array(rowCount);
  let minimumCenterY = Infinity;
  let maximumCenterY = -Infinity;
  includedCells.forEach(cell => {
    const y = minY + (Math.floor(cell / columns) + 0.5) * cellSize;
    minimumCenterY = Math.min(minimumCenterY, y);
    maximumCenterY = Math.max(maximumCenterY, y);
  });
  includedCells.forEach((cell, row) => {
    const x = minX + ((cell % columns) + 0.5) * cellSize;
    const y = minY + (Math.floor(cell / columns) + 0.5) * cellSize;
    rowOfCell[cell] = row;
    positions.set([x, y], row * 2);
    predictors.set(
      [
        roadKilometers[cell],
        Math.log1p(taxiCounts[cell]),
        Math.hypot(x - centerX, y - centerY) / 1000
      ],
      row * 3
    );
    response[row] = Math.log1p(poiCounts[cell]);
    rowPoiCounts[row] = poiCounts[cell];
    rowTaxiCounts[row] = taxiCounts[cell];
    rowRoadKilometers[row] = roadKilometers[cell];
    rowIntersections[row] = intersections[cell];
    const band = Math.min(
      BAND_COUNT - 1,
      Math.floor(((y - minimumCenterY) / Math.max(1, maximumCenterY - minimumCenterY)) * BAND_COUNT)
    );
    bands[row] = band;
    bandOfCell[cell] = band;
  });
  return {
    attribution: `${pois.attribution}; ${trips.attribution}; ${roads.attribution}`,
    origin: roads.origin,
    gridOrigin: [minX, minY],
    cellSize,
    columns,
    rows,
    cellCount,
    rowCount,
    rowOfCell,
    bandOfCell,
    positions,
    predictors,
    response,
    poiCounts: rowPoiCounts,
    taxiCounts: rowTaxiCounts,
    roadKilometers: rowRoadKilometers,
    intersections: rowIntersections,
    bands
  };
}

/** Value at a fraction of the sorted finite values of `values`. */
function getQuantile(values: ArrayLike<number>, fraction: number): number {
  const finite = Array.from(values as ArrayLike<number>).filter(Number.isFinite);
  if (finite.length === 0) return 0;
  finite.sort((left, right) => left - right);
  return finite[
    Math.min(finite.length - 1, Math.max(0, Math.floor(fraction * (finite.length - 1))))
  ];
}

function getSymmetricRange(values: ArrayLike<number>): [number, number] {
  const bound = Math.max(
    1e-6,
    Math.abs(getQuantile(values, 0.05)),
    Math.abs(getQuantile(values, 0.95))
  );
  return [-bound, bound];
}

function formatNumber(value: number, digits = 3): string {
  if (!Number.isFinite(value)) return 'n/a';
  return Math.abs(value) < 0.01 && value !== 0 ? value.toExponential(2) : value.toFixed(digits);
}

const DIVERGING_GRADIENT = {
  colors: [
    [59, 140, 255],
    [237, 237, 237],
    [255, 115, 26]
  ],
  minimumLabel: 'negative',
  maximumLabel: 'positive'
} as const;

const VIRIDIS_GRADIENT = {
  colors: [
    [68, 1, 84],
    [59, 82, 139],
    [33, 145, 140],
    [94, 201, 98],
    [253, 231, 37]
  ],
  minimumLabel: 'low',
  maximumLabel: 'high'
} as const;

/** Instance of a view. */
type ViewInstance = Omit<MapGraphsModeInstance, 'destroy'>;

// ---------------------------------------------------------------------------------------------
// View: OLS
// ---------------------------------------------------------------------------------------------

type OlsMap = 'residuals' | 'fitted' | 'response';

function createOlsView(
  context: MapGraphsModeContext,
  resources: MapGraphsResources,
  study: CellStudy
): ViewInstance {
  const {rowCount} = study;
  let logRidge = -2;
  let mapKind: OlsMap = 'residuals';
  let dirty = true;
  let sigma = 1;

  const predictors = resources.createBuffer('predictors', study.predictors);
  const response = resources.createBuffer('response', study.response);
  const rowOfCell = resources.createBuffer('row-of-cell', study.rowOfCell);
  const coefficients = resources.createBuffer('coefficients', 4 * 4);
  const standardErrors = resources.createBuffer('standard-errors', 4 * 4);
  const tStatistics = resources.createBuffer('t-statistics', 4 * 4);
  const summary = resources.createBuffer('summary', 16 * 4);
  const status = resources.createBuffer('status', 4);
  const residuals = resources.createBuffer('residuals', rowCount * 4);
  const fitted = resources.createBuffer('fitted', rowCount * 4);
  const parameters = resources.createParameterBuffer(
    'ridge',
    'float32',
    GPU_ORDINARY_LEAST_SQUARES_PARAMETER_LENGTH
  );

  const graph = new GPUCommandGraph<void>(context.device, {id: 'regression-ols'});
  graph.add(
    new GPUOrdinaryLeastSquares({
      id: 'ols',
      predictors: importGraphBuffer(graph, 'predictors', predictors, 'float32', rowCount * 3),
      response: importGraphBuffer(graph, 'response', response, 'float32', rowCount),
      predictorCount: 3,
      parameters: parameters.importToGraph(graph),
      output: {
        coefficients: importGraphBuffer(graph, 'coefficients', coefficients, 'float32', 4),
        standardErrors: importGraphBuffer(graph, 'standard-errors', standardErrors, 'float32', 4),
        tStatistics: importGraphBuffer(graph, 't-statistics', tStatistics, 'float32', 4),
        summary: importGraphBuffer(graph, 'summary', summary, 'float32', 16),
        status: importGraphBuffer(graph, 'status', status, 'uint32', 1),
        residuals: importGraphBuffer(graph, 'residuals', residuals, 'float32', rowCount),
        fitted: importGraphBuffer(graph, 'fitted', fitted, 'float32', rowCount)
      }
    })
  );
  const compiled = resources.track(graph.compile());

  const writeRidge = () => {
    parameters.write(
      getGPUOrdinaryLeastSquaresParameterValues(logRidge <= -2 ? 0 : 10 ** logRidge)
    );
    dirty = true;
  };
  context.controls.addSlider({
    label: 'Ridge penalty lambda (per-frame parameter)',
    min: -2,
    max: 4,
    step: 0.1,
    value: logRidge,
    format: value => (value <= -2 ? 'off' : (10 ** value).toPrecision(2)),
    onChange: value => {
      logRidge = value;
      writeRidge();
    }
  });
  context.controls.addSelect<OlsMap>({
    label: 'Map',
    options: [
      {value: 'residuals', label: 'Residuals (blue: over-predicted)'},
      {value: 'fitted', label: 'Fitted log POI count'},
      {value: 'response', label: 'Observed log POI count'}
    ],
    value: mapKind,
    onChange: value => {
      mapKind = value;
      context.updateLayers();
    }
  });
  context.controls.addLegend({
    title: 'Residual: observed - fitted (about +/- 2 sigma)',
    gradient: {...DIVERGING_GRADIENT, minimumLabel: '-2σ', maximumLabel: '+2σ'}
  });
  context.controls.addLegend({
    title: 'Fitted / observed log(1 + POIs)',
    gradient: VIRIDIS_GRADIENT
  });
  context.controls.addNote(
    `log(1 + POIs) ~ ${PREDICTOR_NAMES.join(' + ')} on ${study.cellSize} m cells that contain a road.`
  );
  context.controls.addReadout('Cells (rows)', formatCount(rowCount));
  const fitReadout = context.controls.addReadout('Fit');
  const r2Readout = context.controls.addReadout('R² / adjusted');
  const criteriaReadout = context.controls.addReadout('AIC / BIC');
  const jarqueReadout = context.controls.addReadout('Jarque-Bera p');
  const pagan = context.controls.addReadout('Breusch-Pagan p');
  const coefficientReadouts = ['Intercept', ...PREDICTOR_NAMES].map(name =>
    context.controls.addReadout(name)
  );
  context.controls.addReadout('Data', study.attribution);
  writeRidge();

  const reader = new SummaryReader(
    resources,
    'ols',
    [
      {buffer: coefficients, size: 16},
      {buffer: standardErrors, size: 16},
      {buffer: tStatistics, size: 16},
      {buffer: summary, size: 64},
      {buffer: status, size: 4}
    ],
    bytes => {
      const floats = new Float32Array(bytes);
      const statusWord = new Uint32Array(bytes)[28];
      const sum = floats.subarray(12, 28);
      fitReadout.setValue(statusWord === 0 ? 'ok' : statusWord === 1 ? 'singular' : 'too few rows');
      r2Readout.setValue(
        `${formatNumber(sum[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_R_SQUARED])} / ${formatNumber(sum[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_ADJUSTED_R_SQUARED])}`
      );
      criteriaReadout.setValue(
        `${formatNumber(sum[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_AIC], 1)} / ${formatNumber(sum[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BIC], 1)}`
      );
      jarqueReadout.setValue(
        `${formatNumber(sum[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_JARQUE_BERA_P_VALUE])} (JB ${formatNumber(sum[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_JARQUE_BERA], 1)})`
      );
      pagan.setValue(
        `${formatNumber(sum[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BREUSCH_PAGAN_P_VALUE])} (BP ${formatNumber(sum[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_BREUSCH_PAGAN], 1)})`
      );
      coefficientReadouts.forEach((readout, index) =>
        readout.setValue(
          `${formatNumber(floats[index])} ± ${formatNumber(floats[4 + index])} (t ${formatNumber(floats[8 + index], 1)})`
        )
      );
      sigma = Math.sqrt(Math.max(0, sum[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_SIGMA_SQUARED]));
      context.updateLayers();
    }
  );

  const responseRange: [number, number] = [
    getQuantile(study.response, 0),
    getQuantile(study.response, 1)
  ];
  return {
    getCompiledGraphs: () => [compiled],
    encode(commandEncoder, frame) {
      if (dirty || frame.frameIndex < 3) {
        compiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
        reader.request(commandEncoder);
      } else {
        if (frame.frameIndex % READBACK_INTERVAL_FRAMES === 0) reader.markStale();
        reader.flush(commandEncoder);
      }
    },
    getLayers(): Layer[] {
      const values = mapKind === 'residuals' ? residuals : mapKind === 'fitted' ? fitted : response;
      return [
        new CellGridLayer({
          id: 'ols-cells',
          coordinateOrigin: [study.origin[0], study.origin[1], 0],
          positionOffset: study.gridOrigin,
          cellSize: study.cellSize,
          columns: study.columns,
          cellCount: study.cellCount,
          values,
          indices: rowOfCell,
          colormap: mapKind === 'residuals' ? 'diverging' : 'viridis',
          valueRange: mapKind === 'residuals' ? [-2 * sigma, 2 * sigma] : responseRange,
          color: [255, 255, 255, 215]
        })
      ];
    }
  };
}

// ---------------------------------------------------------------------------------------------
// View: GWR
// ---------------------------------------------------------------------------------------------

const GWR_LADDER_LENGTH = 8;
const ADAPTIVE_LADDER = [12, 20, 30, 45, 65, 90, 110, 128];
const FIXED_LADDER = [350, 500, 700, 1000, 1400, 2000, 2800, 4000];
const COEFFICIENT_COLUMNS = ['Intercept', ...PREDICTOR_NAMES];

type GwrMap = 'coefficient' | 'localR2';

function createGwrView(
  context: MapGraphsModeContext,
  resources: MapGraphsResources,
  study: CellStudy
): ViewInstance {
  const {rowCount} = study;
  let kernel: GPUGeographicallyWeightedRegressionKernel = 'bisquare';
  let bandwidthMode: GPUGeographicallyWeightedRegressionBandwidthMode = 'adaptive';
  let ladderChoice = -1;
  let mapKind: GwrMap = 'coefficient';
  let coefficientColumn = 1;
  let dirty = true;
  let coefficientTable = new Float32Array(rowCount * 4);
  let coefficientRange: [number, number] = [-1, 1];

  const positions = resources.createBuffer('positions', study.positions);
  const predictors = resources.createBuffer('predictors', study.predictors);
  const response = resources.createBuffer('response', study.response);
  const rowOfCell = resources.createBuffer('row-of-cell', study.rowOfCell);
  const coefficients = resources.createBuffer('coefficients', rowCount * 4 * 4);
  const localR2Buffer = resources.createBuffer('local-r2', rowCount * 4);
  const localStatus = resources.createBuffer('local-status', rowCount * 4);
  const bandwidthScores = resources.createBuffer('bandwidth-scores', GWR_LADDER_LENGTH * 4);
  const selectedBandwidth = resources.createBuffer('selected-bandwidth', 8);
  const summary = resources.createBuffer(
    'summary',
    GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH * 4
  );
  const parameters = resources.createParameterBuffer(
    'gwr-parameters',
    'float32',
    getGPUGeographicallyWeightedRegressionParameterLength(GWR_LADDER_LENGTH)
  );
  // The OLS comparison shares the columns.
  const olsCoefficients = resources.createBuffer('ols-coefficients', 16);
  const olsStandardErrors = resources.createBuffer('ols-standard-errors', 16);
  const olsTStatistics = resources.createBuffer('ols-t-statistics', 16);
  const olsSummary = resources.createBuffer('ols-summary', 64);
  const olsStatus = resources.createBuffer('ols-status', 4);

  const graph = new GPUCommandGraph<void>(context.device, {id: 'regression-gwr'});
  graph.add(
    new GPUGeographicallyWeightedRegression({
      id: 'gwr',
      positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', rowCount),
      predictors: importGraphBuffer(graph, 'predictors', predictors, 'float32', rowCount * 3),
      predictorCount: 3,
      response: importGraphBuffer(graph, 'response', response, 'float32', rowCount),
      parameters: parameters.importToGraph(graph),
      maximumBandwidthCount: GWR_LADDER_LENGTH,
      maximumNeighborCount: 128,
      output: {
        coefficients: importGraphBuffer(
          graph,
          'coefficients',
          coefficients,
          'float32',
          rowCount * 4
        ),
        localR2: importGraphBuffer(graph, 'local-r2', localR2Buffer, 'float32', rowCount),
        localStatus: importGraphBuffer(graph, 'local-status', localStatus, 'uint32', rowCount),
        bandwidthScores: importGraphBuffer(
          graph,
          'bandwidth-scores',
          bandwidthScores,
          'float32',
          GWR_LADDER_LENGTH
        ),
        selectedBandwidth: importGraphBuffer(
          graph,
          'selected-bandwidth',
          selectedBandwidth,
          'float32',
          2
        ),
        summary: importGraphBuffer(
          graph,
          'summary',
          summary,
          'float32',
          GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH
        )
      }
    })
  );
  const compiled = resources.track(graph.compile());

  const olsGraph = new GPUCommandGraph<void>(context.device, {id: 'regression-gwr-ols'});
  olsGraph.add(
    new GPUOrdinaryLeastSquares({
      id: 'ols',
      predictors: importGraphBuffer(olsGraph, 'predictors', predictors, 'float32', rowCount * 3),
      response: importGraphBuffer(olsGraph, 'response', response, 'float32', rowCount),
      predictorCount: 3,
      output: {
        coefficients: importGraphBuffer(olsGraph, 'coefficients', olsCoefficients, 'float32', 4),
        standardErrors: importGraphBuffer(
          olsGraph,
          'standard-errors',
          olsStandardErrors,
          'float32',
          4
        ),
        tStatistics: importGraphBuffer(olsGraph, 't-statistics', olsTStatistics, 'float32', 4),
        summary: importGraphBuffer(olsGraph, 'summary', olsSummary, 'float32', 16),
        status: importGraphBuffer(olsGraph, 'status', olsStatus, 'uint32', 1)
      }
    })
  );
  const compiledOls = resources.track(olsGraph.compile());

  const getLadder = () => (bandwidthMode === 'adaptive' ? ADAPTIVE_LADDER : FIXED_LADDER);
  const writeParameters = () => {
    const ladder = getLadder();
    parameters.write(
      getGPUGeographicallyWeightedRegressionParameterValues(
        {
          kernel,
          bandwidthMode,
          bandwidths: ladderChoice < 0 ? ladder : [ladder[ladderChoice]]
        },
        GWR_LADDER_LENGTH
      )
    );
    dirty = true;
  };
  context.controls.addSelect<string>({
    label: 'Bandwidth (parameter: the whole ladder is searched by AICc, or one value)',
    options: [
      {value: '-1', label: 'Auto (min AICc over the ladder)'},
      ...ADAPTIVE_LADDER.map((value, index) => ({value: String(index), label: `Ladder ${index}`}))
    ],
    value: '-1',
    onChange: value => {
      ladderChoice = Number(value);
      writeParameters();
    }
  });
  const describeLadder = () => {
    // Relabel the ladder options for the current bandwidth mode.
    const selectElement = document.querySelector<HTMLSelectElement>(
      '[aria-label^="Bandwidth (parameter"]'
    );
    if (!selectElement) return;
    const ladder = getLadder();
    Array.from(selectElement.options).forEach(option => {
      const index = Number(option.value);
      if (index >= 0) {
        option.textContent =
          bandwidthMode === 'adaptive' ? `${ladder[index]} nearest cells` : `${ladder[index]} m`;
      }
    });
  };
  context.controls.addSelect<GPUGeographicallyWeightedRegressionBandwidthMode>({
    label: 'Bandwidth mode (parameter)',
    options: [
      {value: 'adaptive', label: 'Adaptive: k nearest cells'},
      {value: 'fixed', label: 'Fixed: distance in meters'}
    ],
    value: bandwidthMode,
    onChange: value => {
      bandwidthMode = value;
      describeLadder();
      writeParameters();
    }
  });
  context.controls.addSelect<GPUGeographicallyWeightedRegressionKernel>({
    label: 'Kernel (parameter)',
    options: [
      {value: 'bisquare', label: 'Bisquare'},
      {value: 'gaussian', label: 'Gaussian'}
    ],
    value: kernel,
    onChange: value => {
      kernel = value;
      writeParameters();
    }
  });
  context.controls.addSelect<string>({
    label: 'Map',
    options: [
      {value: 'coefficient:0', label: `Local ${COEFFICIENT_COLUMNS[0]}`},
      {value: 'coefficient:1', label: `Local coefficient: ${COEFFICIENT_COLUMNS[1]}`},
      {value: 'coefficient:2', label: `Local coefficient: ${COEFFICIENT_COLUMNS[2]}`},
      {value: 'coefficient:3', label: `Local coefficient: ${COEFFICIENT_COLUMNS[3]}`},
      {value: 'localR2', label: 'Local R²'}
    ],
    value: 'coefficient:1',
    onChange: value => {
      if (value === 'localR2') {
        mapKind = 'localR2';
      } else {
        mapKind = 'coefficient';
        coefficientColumn = Number(value.split(':')[1]);
      }
      updateRange();
      context.updateLayers();
    }
  });
  context.controls.addLegend({
    title: 'Local coefficient (blue negative, orange positive; 5th to 95th percentile)',
    gradient: DIVERGING_GRADIENT
  });
  context.controls.addLegend({title: 'Local R² (0 to 1)', gradient: VIRIDIS_GRADIENT});
  context.controls.addNote(
    'Brute force O(n² x ladder): the graph is encoded when an input changes, not every frame.'
  );
  context.controls.addReadout('Cells (rows)', formatCount(rowCount));
  const selectedReadout = context.controls.addReadout('Selected bandwidth');
  const scoreReadout = context.controls.addReadout('AICc per ladder value');
  const gwrAiccReadout = context.controls.addReadout('GWR AICc / trace S');
  const olsAiccReadout = context.controls.addReadout('OLS AICc (same formula)');
  const rSquaredReadout = context.controls.addReadout('R²: GWR / OLS');
  const singularReadout = context.controls.addReadout('Singular locations');
  const spreadReadout = context.controls.addReadout('Coefficient 5-95%');
  context.controls.addReadout('Data', study.attribution);
  writeParameters();
  describeLadder();

  const updateRange = () => {
    if (mapKind === 'localR2') return;
    const column = new Float32Array(rowCount);
    for (let row = 0; row < rowCount; row++)
      column[row] = coefficientTable[row * 4 + coefficientColumn];
    coefficientRange = getSymmetricRange(column);
    spreadReadout.setValue(
      `${formatNumber(getQuantile(column, 0.05))} to ${formatNumber(getQuantile(column, 0.95))} (global ${formatNumber(olsCoefficientValues[coefficientColumn])})`
    );
  };
  let olsCoefficientValues = new Float32Array(4);
  let olsAicc = Number.NaN;
  let olsR2 = Number.NaN;

  const summaryFloats =
    GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH + 2 + GWR_LADDER_LENGTH;
  const reader = new SummaryReader(
    resources,
    'gwr',
    [
      {buffer: summary, size: GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH * 4},
      {buffer: selectedBandwidth, size: 8},
      {buffer: bandwidthScores, size: GWR_LADDER_LENGTH * 4},
      {buffer: coefficients, size: rowCount * 16},
      {buffer: localR2Buffer, size: rowCount * 4},
      {buffer: localStatus, size: rowCount * 4},
      {buffer: olsCoefficients, size: 16},
      {buffer: olsSummary, size: 64}
    ],
    bytes => {
      const floats = new Float32Array(bytes);
      const words = new Uint32Array(bytes);
      const gwr = floats.subarray(0, GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH);
      const selected = floats.subarray(
        GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH,
        GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH + 2
      );
      const scores = floats.subarray(
        GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY_LENGTH + 2,
        summaryFloats
      );
      coefficientTable = floats.slice(summaryFloats, summaryFloats + rowCount * 4);
      const statusBase = summaryFloats + rowCount * 5;
      let singular = 0;
      for (let row = 0; row < rowCount; row++) if (words[statusBase + row] === 1) singular++;
      const olsBase = statusBase + rowCount;
      olsCoefficientValues = floats.slice(olsBase, olsBase + 4);
      const olsSummaryValues = floats.subarray(olsBase + 4, olsBase + 20);
      const count = olsSummaryValues[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_ROW_COUNT];
      const rss = olsSummaryValues[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_RESIDUAL_SUM_OF_SQUARES];
      olsR2 = olsSummaryValues[GPU_ORDINARY_LEAST_SQUARES_SUMMARY_R_SQUARED];
      // GWR's AICc with the OLS hat trace (4 parameters).
      olsAicc =
        count * Math.log(rss / count) +
        count * Math.log(2 * Math.PI) +
        (count * (count + 4)) / (count - 2 - 4);
      const getSummary = (slot: number) => gwr[slot];
      const valid = getSummary(GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY.HAS_VALID_CANDIDATE);
      const ladder = getLadder();
      selectedReadout.setValue(
        valid
          ? `${bandwidthMode === 'adaptive' ? `${selected[1]} nearest cells` : `${selected[1]} m`} (candidate ${selected[0]})`
          : 'no valid candidate (all singular)'
      );
      scoreReadout.setValue(
        Array.from(scores)
          .slice(0, ladderChoice < 0 ? ladder.length : 1)
          .map(value => (Number.isFinite(value) ? value.toFixed(0) : 'NaN'))
          .join(' / ')
      );
      const aicc = getSummary(GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY.AICC);
      gwrAiccReadout.setValue(
        `${formatNumber(aicc, 1)} / ${formatNumber(getSummary(GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY.TRACE_OF_HAT), 1)}`
      );
      olsAiccReadout.setValue(
        `${formatNumber(olsAicc, 1)} (GWR ${aicc < olsAicc ? 'better' : 'worse'} by ${formatNumber(Math.abs(olsAicc - aicc), 1)})`
      );
      rSquaredReadout.setValue(
        `${formatNumber(getSummary(GPU_GEOGRAPHICALLY_WEIGHTED_REGRESSION_SUMMARY.R_SQUARED))} / ${formatNumber(olsR2)}`
      );
      singularReadout.setValue(`${formatCount(singular)} of ${formatCount(rowCount)}`);
      updateRange();
      context.updateLayers();
    }
  );

  return {
    getCompiledGraphs: () => [compiled, compiledOls],
    encode(commandEncoder, frame) {
      if (dirty || frame.frameIndex < 3) {
        compiled.encode(commandEncoder, {parameters: undefined});
        if (frame.frameIndex < 3) compiledOls.encode(commandEncoder, {parameters: undefined});
        dirty = false;
        reader.request(commandEncoder);
      } else {
        if (frame.frameIndex % READBACK_INTERVAL_FRAMES === 0) reader.markStale();
        reader.flush(commandEncoder);
      }
    },
    getLayers(): Layer[] {
      const localMap = mapKind === 'localR2';
      return [
        new CellGridLayer({
          id: 'gwr-cells',
          coordinateOrigin: [study.origin[0], study.origin[1], 0],
          positionOffset: study.gridOrigin,
          cellSize: study.cellSize,
          columns: study.columns,
          cellCount: study.cellCount,
          values: localMap ? localR2Buffer : coefficients,
          valueStride: localMap ? 1 : 4,
          valueOffset: localMap ? 0 : coefficientColumn,
          indices: rowOfCell,
          colormap: localMap ? 'viridis' : 'diverging',
          valueRange: localMap ? [0, 1] : coefficientRange,
          color: [255, 255, 255, 215]
        })
      ];
    }
  };
}

// ---------------------------------------------------------------------------------------------
// View: composite score
// ---------------------------------------------------------------------------------------------

const INDICATOR_NAMES = ['POI density', 'Road length', 'Intersections', 'Taxi activity'];

function createCompositeView(
  context: MapGraphsModeContext,
  resources: MapGraphsResources,
  study: CellStudy
): ViewInstance {
  const {rowCount} = study;
  const indicatorCount = INDICATOR_NAMES.length;
  const weights = [1, 1, 1, 1];
  const directions = [1, 1, 1, 1];
  let scaler: GPUCompositeScoreScaler = 'min-max';
  let aggregation: GPUCompositeScoreAggregation = 'weighted-sum';
  let dirty = true;
  let scores = new Float32Array(rowCount);
  let scoreRange: [number, number] = [0, 1];

  const indicators = new Float32Array(rowCount * indicatorCount);
  for (let row = 0; row < rowCount; row++) {
    indicators.set(
      [
        Math.log1p(study.poiCounts[row]),
        study.roadKilometers[row],
        study.intersections[row],
        Math.log1p(study.taxiCounts[row])
      ],
      row * indicatorCount
    );
  }
  const indicatorBuffer = resources.createBuffer('indicators', indicators);
  const rowOfCell = resources.createBuffer('row-of-cell', study.rowOfCell);
  const scoreBuffer = resources.createBuffer('score', rowCount * 4);
  const columnStatistics = resources.createBuffer('column-statistics', indicatorCount * 4 * 4);
  const loadings = resources.createBuffer('loadings', indicatorCount * 4);
  const componentSummary = resources.createBuffer('component-summary', 3 * 4);
  const parameters = resources.createParameterBuffer(
    'composite-parameters',
    'float32',
    GPU_COMPOSITE_SCORE_PARAMETER_LENGTH
  );
  const graph = new GPUCommandGraph<void>(context.device, {id: 'regression-composite'});
  graph.add(
    new GPUCompositeScore({
      id: 'composite',
      indicators: importGraphBuffer(
        graph,
        'indicators',
        indicatorBuffer,
        'float32',
        rowCount * indicatorCount
      ),
      indicatorCount,
      parameters: parameters.importToGraph(graph),
      enableRank: true,
      enablePrincipalComponent: true,
      output: {
        score: importGraphBuffer(graph, 'score', scoreBuffer, 'float32', rowCount),
        columnStatistics: importGraphBuffer(
          graph,
          'column-statistics',
          columnStatistics,
          'float32',
          indicatorCount * 4
        ),
        loadings: importGraphBuffer(graph, 'loadings', loadings, 'float32', indicatorCount),
        principalComponentSummary: importGraphBuffer(
          graph,
          'component-summary',
          componentSummary,
          'float32',
          3
        )
      }
    })
  );
  const compiled = resources.track(graph.compile());

  const writeParameters = () => {
    parameters.write(
      getGPUCompositeScoreParameterValues({scaler, aggregation, weights, directions})
    );
    dirty = true;
  };
  context.controls.addSelect<GPUCompositeScoreScaler>({
    label: 'Scaler (parameter)',
    options: [
      {value: 'min-max', label: 'Min-max'},
      {value: 'z-score', label: 'Z-score'},
      {value: 'rank', label: 'Percentile rank'}
    ],
    value: scaler,
    onChange: value => {
      scaler = value;
      writeParameters();
    }
  });
  context.controls.addSelect<GPUCompositeScoreAggregation>({
    label: 'Aggregation (parameter)',
    options: [
      {value: 'weighted-sum', label: 'Weighted sum'},
      {value: 'weighted-geometric-mean', label: 'Weighted geometric mean'},
      {value: 'principal-component', label: 'First principal component'}
    ],
    value: aggregation,
    onChange: value => {
      aggregation = value;
      writeParameters();
    }
  });
  INDICATOR_NAMES.forEach((name, index) => {
    context.controls.addSlider({
      label: `Weight: ${name}`,
      min: 0,
      max: 5,
      step: 0.1,
      value: weights[index],
      format: value => value.toFixed(1),
      onChange: value => {
        weights[index] = value;
        writeParameters();
      }
    });
    context.controls.addToggle({
      label: `${name}: higher is worse (flips the scaled value)`,
      value: false,
      onChange: value => {
        directions[index] = value ? -1 : 1;
        writeParameters();
      }
    });
  });
  context.controls.addLegend({
    title: 'Composite score (5th to 95th percentile; min-max and rank are 0 to 1)',
    gradient: VIRIDIS_GRADIENT
  });
  context.controls.addReadout('Cells (rows)', formatCount(rowCount));
  const classReadout = context.controls.addReadout('Cells per fifth');
  const statisticsReadouts = INDICATOR_NAMES.map(name => context.controls.addReadout(name));
  const componentReadout = context.controls.addReadout('PC1 explained variance');
  const loadingReadout = context.controls.addReadout('PC1 loadings');
  context.controls.addReadout('Data', study.attribution);
  writeParameters();

  const reader = new SummaryReader(
    resources,
    'composite',
    [
      {buffer: scoreBuffer, size: rowCount * 4},
      {buffer: columnStatistics, size: indicatorCount * 16},
      {buffer: loadings, size: indicatorCount * 4},
      {buffer: componentSummary, size: 12}
    ],
    bytes => {
      const floats = new Float32Array(bytes);
      scores = floats.slice(0, rowCount);
      const statisticsBase = rowCount;
      statisticsReadouts.forEach((readout, index) => {
        const base = statisticsBase + index * 4;
        readout.setValue(
          `${formatNumber(floats[base], 2)} .. ${formatNumber(floats[base + 1], 2)}, mean ${formatNumber(floats[base + 2], 2)}, sd ${formatNumber(floats[base + 3], 2)}`
        );
      });
      const loadingBase = statisticsBase + indicatorCount * 4;
      loadingReadout.setValue(
        Array.from(floats.subarray(loadingBase, loadingBase + indicatorCount), value =>
          value.toFixed(2)
        ).join(' / ')
      );
      componentReadout.setValue(
        `${(floats[loadingBase + indicatorCount + 1] * 100).toFixed(1)}% (residual ${formatNumber(floats[loadingBase + indicatorCount + 2], 4)})`
      );
      const lowQuantile = getQuantile(scores, 0.05);
      const highQuantile = getQuantile(scores, 0.95);
      scoreRange =
        scaler === 'z-score' || aggregation === 'principal-component'
          ? [lowQuantile, highQuantile]
          : aggregation === 'weighted-geometric-mean'
            ? [lowQuantile, highQuantile]
            : [0, 1];
      const finite = Array.from(scores)
        .filter(Number.isFinite)
        .sort((a, b) => a - b);
      const counts = new Array(5).fill(0);
      for (const value of finite) {
        const fraction =
          (value - finite[0]) / Math.max(1e-9, finite[finite.length - 1] - finite[0]);
        counts[Math.min(4, Math.floor(fraction * 5))]++;
      }
      classReadout.setValue(counts.map(formatCount).join(' / ') + ` (${finite.length} scored)`);
      context.updateLayers();
    }
  );

  return {
    getCompiledGraphs: () => [compiled],
    encode(commandEncoder, frame) {
      if (dirty || frame.frameIndex < 3) {
        compiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
        reader.request(commandEncoder);
      } else {
        if (frame.frameIndex % READBACK_INTERVAL_FRAMES === 0) reader.markStale();
        reader.flush(commandEncoder);
      }
    },
    getLayers(): Layer[] {
      return [
        new CellGridLayer({
          id: 'composite-cells',
          coordinateOrigin: [study.origin[0], study.origin[1], 0],
          positionOffset: study.gridOrigin,
          cellSize: study.cellSize,
          columns: study.columns,
          cellCount: study.cellCount,
          values: scoreBuffer,
          indices: rowOfCell,
          colormap: 'viridis',
          valueRange: scoreRange,
          color: [255, 255, 255, 215]
        })
      ];
    }
  };
}

// ---------------------------------------------------------------------------------------------
// View: inequality
// ---------------------------------------------------------------------------------------------

type InequalityColumn = 'poi' | 'taxi' | 'road';
type InequalityMetric = 'gini' | 'theilT' | 'atkinson' | 'hoover' | 'palma';

function createInequalityView(
  context: MapGraphsModeContext,
  resources: MapGraphsResources,
  study: CellStudy
): ViewInstance {
  const {rowCount} = study;
  let column: InequalityColumn = 'poi';
  let metric: InequalityMetric = 'gini';
  let epsilon = 0.5;
  let palmaTop = 0.1;
  let dirty = true;
  const metricValues: Record<InequalityMetric, Float32Array> = {
    gini: new Float32Array(BAND_COUNT),
    theilT: new Float32Array(BAND_COUNT),
    atkinson: new Float32Array(BAND_COUNT),
    hoover: new Float32Array(BAND_COUNT),
    palma: new Float32Array(BAND_COUNT)
  };
  let metricRange: [number, number] = [0, 1];

  const getColumn = (): Float32Array =>
    column === 'poi'
      ? study.poiCounts
      : column === 'taxi'
        ? study.taxiCounts
        : study.roadKilometers;
  const valuesBuffer = resources.createBuffer('values', getColumn());
  const zoneBuffer = resources.createBuffer('zone-ids', study.bands);
  const bandOfCell = resources.createBuffer('band-of-cell', study.bandOfCell);
  const parameters = resources.createParameterBuffer(
    'inequality-parameters',
    'float32',
    GPU_INEQUALITY_PARAMETER_LENGTH
  );
  const outputs = {
    gini: resources.createBuffer('gini', BAND_COUNT * 4),
    theilT: resources.createBuffer('theil-t', BAND_COUNT * 4),
    atkinson: resources.createBuffer('atkinson', BAND_COUNT * 4),
    hoover: resources.createBuffer('hoover', BAND_COUNT * 4),
    palma: resources.createBuffer('palma', BAND_COUNT * 4),
    count: resources.createBuffer('count', BAND_COUNT * 4),
    lorenzKnots: resources.createBuffer('lorenz-knots', BAND_COUNT * LORENZ_KNOT_COUNT * 4),
    globalSummary: resources.createBuffer(
      'global-summary',
      GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH * 4
    )
  };
  const graph = new GPUCommandGraph<void>(context.device, {id: 'regression-inequality'});
  graph.add(
    new GPUInequality({
      id: 'inequality',
      values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', rowCount),
      zoneIds: importGraphBuffer(graph, 'zone-ids', zoneBuffer, 'uint32', rowCount),
      zoneCount: BAND_COUNT,
      lorenzKnotCount: LORENZ_KNOT_COUNT,
      parameters: parameters.importToGraph(graph),
      output: {
        gini: importGraphBuffer(graph, 'gini', outputs.gini, 'float32', BAND_COUNT),
        theilT: importGraphBuffer(graph, 'theil-t', outputs.theilT, 'float32', BAND_COUNT),
        atkinson: importGraphBuffer(graph, 'atkinson', outputs.atkinson, 'float32', BAND_COUNT),
        hoover: importGraphBuffer(graph, 'hoover', outputs.hoover, 'float32', BAND_COUNT),
        palma: importGraphBuffer(graph, 'palma', outputs.palma, 'float32', BAND_COUNT),
        count: importGraphBuffer(graph, 'count', outputs.count, 'uint32', BAND_COUNT),
        lorenzKnots: importGraphBuffer(
          graph,
          'lorenz-knots',
          outputs.lorenzKnots,
          'float32',
          BAND_COUNT * LORENZ_KNOT_COUNT
        ),
        globalSummary: importGraphBuffer(
          graph,
          'global-summary',
          outputs.globalSummary,
          'float32',
          GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH
        )
      }
    })
  );
  const compiled = resources.track(graph.compile());

  const writeParameters = () => {
    parameters.write(getGPUInequalityParameterValues({epsilon, palmaTopShare: palmaTop}));
    dirty = true;
  };
  context.controls.addSelect<InequalityColumn>({
    label: 'Distribution of (rewrites the values column)',
    options: [
      {value: 'poi', label: 'POI count per cell'},
      {value: 'taxi', label: 'Taxi trip vertices per cell'},
      {value: 'road', label: 'Road length per cell (km)'}
    ],
    value: column,
    onChange: value => {
      column = value;
      valuesBuffer.write(getColumn());
      dirty = true;
    }
  });
  context.controls.addSelect<InequalityMetric>({
    label: 'Map and readouts: band index',
    options: [
      {value: 'gini', label: 'Gini'},
      {value: 'theilT', label: 'Theil T'},
      {value: 'atkinson', label: 'Atkinson (epsilon below)'},
      {value: 'hoover', label: 'Hoover'},
      {value: 'palma', label: 'Palma ratio'}
    ],
    value: metric,
    onChange: value => {
      metric = value;
      updateRange();
      context.updateLayers();
    }
  });
  context.controls.addSlider({
    label: 'Atkinson inequality aversion epsilon (parameter)',
    min: 0,
    max: 0.95,
    step: 0.05,
    value: epsilon,
    format: value => value.toFixed(2),
    onChange: value => {
      epsilon = value;
      writeParameters();
    }
  });
  context.controls.addSlider({
    label: 'Palma top share (bottom is 40%; parameter)',
    min: 0.05,
    max: 0.3,
    step: 0.05,
    value: palmaTop,
    format: value => `${Math.round(value * 100)}%`,
    onChange: value => {
      palmaTop = value;
      writeParameters();
    }
  });
  context.controls.addLegend({
    title: 'Band index (south to north bands; low to high)',
    gradient: VIRIDIS_GRADIENT
  });
  context.controls.addNote(
    'Bands split the study area into six latitude slices, 1 in the south to 6 in the north. ' +
      'Atkinson needs epsilon below 1 and Theil L is not shown because many cells hold zero (a ' +
      'zero makes them NaN by design).'
  );
  context.controls.addReadout('Cells (rows)', formatCount(rowCount));
  const pooledReadout = context.controls.addReadout('Pooled Gini / Theil T');
  const decompositionReadout = context.controls.addReadout('Theil T between + within');
  const bandReadouts = Array.from({length: BAND_COUNT}, (_, band) =>
    context.controls.addReadout(`Band ${band + 1}`)
  );
  context.controls.addReadout('Data', study.attribution);
  const canvas = createChartCanvas();
  writeParameters();

  const updateRange = () => {
    const values = metricValues[metric];
    const low = Math.min(...Array.from(values).filter(Number.isFinite));
    const high = Math.max(...Array.from(values).filter(Number.isFinite));
    metricRange = Number.isFinite(low) && high > low ? [low, high] : [0, 1];
  };

  const knotBytes = BAND_COUNT * LORENZ_KNOT_COUNT * 4;
  const reader = new SummaryReader(
    resources,
    'inequality',
    [
      {buffer: outputs.gini, size: BAND_COUNT * 4},
      {buffer: outputs.theilT, size: BAND_COUNT * 4},
      {buffer: outputs.atkinson, size: BAND_COUNT * 4},
      {buffer: outputs.hoover, size: BAND_COUNT * 4},
      {buffer: outputs.palma, size: BAND_COUNT * 4},
      {buffer: outputs.count, size: BAND_COUNT * 4},
      {buffer: outputs.lorenzKnots, size: knotBytes},
      {buffer: outputs.globalSummary, size: GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH * 4}
    ],
    bytes => {
      const floats = new Float32Array(bytes);
      const words = new Uint32Array(bytes);
      const names: InequalityMetric[] = ['gini', 'theilT', 'atkinson', 'hoover', 'palma'];
      names.forEach((name, index) => {
        metricValues[name] = floats.slice(index * BAND_COUNT, (index + 1) * BAND_COUNT);
      });
      const counts = words.subarray(5 * BAND_COUNT, 6 * BAND_COUNT);
      const knots = floats.subarray(
        6 * BAND_COUNT,
        6 * BAND_COUNT + BAND_COUNT * LORENZ_KNOT_COUNT
      );
      const summaryBase = 6 * BAND_COUNT + BAND_COUNT * LORENZ_KNOT_COUNT;
      const global = floats.subarray(
        summaryBase,
        summaryBase + GPU_INEQUALITY_GLOBAL_SUMMARY_LENGTH
      );
      pooledReadout.setValue(
        `${formatNumber(global[GPU_INEQUALITY_GLOBAL_SUMMARY.GINI])} / ${formatNumber(global[GPU_INEQUALITY_GLOBAL_SUMMARY.TOTAL_THEIL_T])}`
      );
      decompositionReadout.setValue(
        `${formatNumber(global[GPU_INEQUALITY_GLOBAL_SUMMARY.BETWEEN_THEIL_T])} + ${formatNumber(global[GPU_INEQUALITY_GLOBAL_SUMMARY.WITHIN_THEIL_T])}`
      );
      bandReadouts.forEach((readout, band) =>
        readout.setValue(
          `${formatNumber(metricValues[metric][band])} (${formatCount(counts[band])} cells, Gini ${formatNumber(metricValues.gini[band], 2)})`
        )
      );
      updateRange();
      drawLorenzChart(canvas, knots, metricValues.gini);
      context.updateLayers();
    }
  );

  return {
    getCompiledGraphs: () => [compiled],
    encode(commandEncoder, frame) {
      if (dirty || frame.frameIndex < 3) {
        compiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
        reader.request(commandEncoder);
      } else {
        if (frame.frameIndex % READBACK_INTERVAL_FRAMES === 0) reader.markStale();
        reader.flush(commandEncoder);
      }
    },
    getLayers(): Layer[] {
      return [
        new CellGridLayer({
          id: 'inequality-cells',
          coordinateOrigin: [study.origin[0], study.origin[1], 0],
          positionOffset: study.gridOrigin,
          cellSize: study.cellSize,
          columns: study.columns,
          cellCount: study.cellCount,
          values: outputs[metric],
          indices: bandOfCell,
          colormap: 'viridis',
          valueRange: metricRange,
          inset: 0,
          color: [255, 255, 255, 200]
        })
      ];
    }
  };
}

// ---------------------------------------------------------------------------------------------
// Lorenz curve chart
// ---------------------------------------------------------------------------------------------

const CHART_SIZE = 220;

/** Appends a canvas under the readouts for the Lorenz chart; null outside a browser. */
function createChartCanvas(): HTMLCanvasElement | null {
  if (typeof document === 'undefined') return null;
  const readouts = document.querySelector('[data-mode-readouts]');
  if (!readouts) return null;
  const canvas = document.createElement('canvas');
  canvas.width = CHART_SIZE * 2;
  canvas.height = CHART_SIZE * 2;
  canvas.style.cssText = `width:${CHART_SIZE}px;height:${CHART_SIZE}px;margin-top:10px;border-radius:8px;background:rgba(19,32,63,.7)`;
  canvas.setAttribute('aria-label', 'Lorenz curves by band');
  readouts.after(canvas);
  return canvas;
}

/** Draws one Lorenz curve per band against the line of equality. */
function drawLorenzChart(
  canvas: HTMLCanvasElement | null,
  knots: Float32Array,
  gini: Float32Array
): void {
  const context = canvas?.getContext('2d');
  if (!canvas || !context) return;
  const size = canvas.width;
  const margin = 36;
  const plot = size - margin * 1.5;
  context.clearRect(0, 0, size, size);
  context.font = '20px system-ui, sans-serif';
  context.lineWidth = 2;
  context.strokeStyle = 'rgba(169,184,208,0.5)';
  context.strokeRect(margin, margin / 2, plot, plot);
  context.setLineDash([6, 6]);
  context.beginPath();
  context.moveTo(margin, margin / 2 + plot);
  context.lineTo(margin + plot, margin / 2);
  context.stroke();
  context.setLineDash([]);
  for (let band = 0; band < BAND_COUNT; band++) {
    const [red, green, blue] = BAND_COLORS[band];
    context.strokeStyle = `rgb(${red},${green},${blue})`;
    context.lineWidth = 3;
    context.beginPath();
    for (let knot = 0; knot < LORENZ_KNOT_COUNT; knot++) {
      const value = knots[band * LORENZ_KNOT_COUNT + knot];
      if (!Number.isFinite(value)) break;
      const x = margin + (knot / (LORENZ_KNOT_COUNT - 1)) * plot;
      const y = margin / 2 + plot - value * plot;
      if (knot === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }
    context.stroke();
    context.fillStyle = `rgb(${red},${green},${blue})`;
    context.fillText(
      `${band + 1}: ${Number.isFinite(gini[band]) ? gini[band].toFixed(2) : 'n/a'}`,
      margin + 8 + (band % 3) * 120,
      margin / 2 + 28 + Math.floor(band / 3) * 24
    );
  }
  context.fillStyle = 'rgba(169,184,208,0.9)';
  context.fillText('population share', margin + plot / 2 - 70, size - 6);
  context.save();
  context.translate(14, margin / 2 + plot / 2 + 50);
  context.rotate(-Math.PI / 2);
  context.fillText('income share (Lorenz)', 0, 0);
  context.restore();
}
