// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Distribution dynamics of Manhattan taxi activity. Trip vertices are counted per square cell and
 * per time period (the trips span about 40 minutes, so a period is a few minutes, not an hour).
 * One compiled graph then runs the whole giddy-style chain on the GPU:
 *
 * 1. `GPUNeighborSearch` writes queen-contiguity weights between active cells.
 * 2. One `GPUClassBreaks` over the pooled column of every period gives a single set of edges, and
 *    `GPUClassAssignment` classifies every period with them (one legend for the time slider).
 * 3. `GPUTransitionMatrix` counts class moves between consecutive periods.
 * 4. `GPUSpatialMarkov` conditions the transitions on the pooled class of the spatial lag.
 * 5. `GPULISAMarkov` counts moves between local Moran quadrants.
 *
 * The class method, class count and LISA significance level are per-frame parameter writes; the
 * period slider only changes a two-word parameter of a tiny display kernel that gathers one period
 * of the chosen classification. The matrices come back through one small ring-buffered readback.
 */

import type {Layer} from '@deck.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUClassBreaks,
  getGPUClassBreaksParameterLength,
  getGPUClassBreaksParameterValues,
  type GPUClassBreaksMethod
} from '@luma.gl/experimental/gpu-dataframe';
import {
  getGPUNeighborSearchParameterValues,
  getGPUSpatialAutocorrelationParameterValues,
  GPUClassAssignment,
  GPULISAMarkov,
  GPUNeighborSearch,
  GPUSpatialMarkov,
  GPUTransitionMatrix,
  GPU_LISA_MARKOV_STATE_COUNT,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-spatial-analysis';
import type {GPUVectorFormat} from '@luma.gl/gpgpu/gpu-data';
import {importGraphBuffer} from '../graph-buffers';
import {SpatialAnalysisRasterLayer, type SpatialAnalysisColor} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {addKernelPass} from './mode-kernels';
import {SummaryReader} from './summary-reader';

const MINIMUM_CELL_METERS = 200;
const MAXIMUM_CELLS_PER_SIDE = 110;
const PERIOD_COUNT = 12;
/** Cells with fewer trip vertices over the whole span are left out of the analysis. */
const MINIMUM_ACTIVITY = 16;
const MAXIMUM_CLASS_COUNT = 5;
const LAG_CLASS_COUNT = 3;
const STATE_COUNT = GPU_LISA_MARKOV_STATE_COUNT;
const NO_CLASS = 0xffffffff;
const PLAY_SECONDS_PER_PERIOD = 0.9;

const CLASS_COLORS: readonly SpatialAnalysisColor[] = [
  [68, 1, 84, 235],
  [59, 82, 139, 235],
  [33, 145, 140, 235],
  [94, 201, 98, 235],
  [253, 231, 37, 235]
];
/** LISA states: 0 not significant, 1 HH, 2 LH, 3 LL, 4 HL. */
const LISA_COLORS: readonly SpatialAnalysisColor[] = [
  [120, 130, 145, 150],
  [215, 48, 39, 245],
  [145, 191, 219, 245],
  [49, 54, 149, 245],
  [253, 174, 97, 245]
];
const LISA_LABELS = ['n.s.', 'HH', 'LH', 'LL', 'HL'] as const;
const NO_DATA_COLOR: SpatialAnalysisColor = [0, 0, 0, 0];

type MapView = 'class' | 'lag' | 'lisa';
type MatrixChoice = 'pooled' | 'lag0' | 'lag1' | 'lag2' | 'lisa';

const METHOD_LABELS: Record<string, string> = {
  quantile: 'Quantile',
  'equal-interval': 'Equal interval',
  'natural-breaks': 'Natural breaks (Jenks)'
};
const METHODS = ['quantile', 'equal-interval', 'natural-breaks'] as const;

type GridCells = {
  columns: number;
  rows: number;
  cellMeters: number;
  bounds: [number, number, number, number];
  /** `rows * periods` vertex counts, period-major. */
  counts: Float32Array;
  positions: Float32Array;
  mask: Uint32Array;
  activeCount: number;
  timeStart: number;
  periodSeconds: number;
};

/** Returns the 0.5th and 99.5th percentile of one coordinate of packed `x, y` pairs. */
function getCentralRange(positions: Float32Array, axis: 0 | 1): [number, number] {
  const values = new Float32Array(positions.length / 2);
  for (let index = 0; index < values.length; index++) values[index] = positions[index * 2 + axis];
  values.sort();
  return [
    values[Math.floor(values.length * 0.005)],
    values[Math.min(values.length - 1, Math.floor(values.length * 0.995))]
  ];
}

/** Counts trip vertices per (period, cell) on a square grid over the data extent. */
function binTripVertices(
  positions: Float32Array,
  timestamps: Float32Array,
  timeRange: readonly [number, number]
): GridCells {
  // A few trips have coordinates far from the city, so the grid covers the central 99% of the
  // vertices (per axis) and drops the rest.
  const [minimumX, maximumX] = getCentralRange(positions, 0);
  const [minimumY, maximumY] = getCentralRange(positions, 1);
  const cellMeters = Math.max(
    MINIMUM_CELL_METERS,
    (maximumX - minimumX) / MAXIMUM_CELLS_PER_SIDE,
    (maximumY - minimumY) / MAXIMUM_CELLS_PER_SIDE
  );
  const columns = Math.max(2, Math.ceil((maximumX - minimumX) / cellMeters));
  const rows = Math.max(2, Math.ceil((maximumY - minimumY) / cellMeters));
  const cellCount = columns * rows;
  const counts = new Float32Array(cellCount * PERIOD_COUNT);
  const periodSeconds = Math.max(1, (timeRange[1] - timeRange[0]) / PERIOD_COUNT);
  const totals = new Float32Array(cellCount);
  for (let vertex = 0; vertex < timestamps.length; vertex++) {
    const x = positions[vertex * 2];
    const y = positions[vertex * 2 + 1];
    if (x < minimumX || x > maximumX || y < minimumY || y > maximumY) continue;
    const column = Math.min(
      columns - 1,
      Math.floor((positions[vertex * 2] - minimumX) / cellMeters)
    );
    const row = Math.min(rows - 1, Math.floor((positions[vertex * 2 + 1] - minimumY) / cellMeters));
    const period = Math.min(
      PERIOD_COUNT - 1,
      Math.max(0, Math.floor((timestamps[vertex] - timeRange[0]) / periodSeconds))
    );
    const cell = row * columns + column;
    counts[period * cellCount + cell]++;
    totals[cell]++;
  }
  // Counts are heavily skewed with many zeros, so the analysis variable is log2(1 + count).
  for (let index = 0; index < counts.length; index++) counts[index] = Math.log2(1 + counts[index]);
  const mask = new Uint32Array(cellCount);
  const cellPositions = new Float32Array(cellCount * 2);
  let activeCount = 0;
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const cell = row * columns + column;
      cellPositions[cell * 2] = minimumX + (column + 0.5) * cellMeters;
      cellPositions[cell * 2 + 1] = minimumY + (row + 0.5) * cellMeters;
      if (totals[cell] >= MINIMUM_ACTIVITY) {
        mask[cell] = 1;
        activeCount++;
      }
    }
  }
  return {
    columns,
    rows,
    cellMeters,
    bounds: [minimumX, minimumY, minimumX + columns * cellMeters, minimumY + rows * cellMeters],
    counts,
    positions: cellPositions,
    mask,
    activeCount,
    timeStart: timeRange[0],
    periodSeconds
  };
}

/** Matrices decoded from one summary readback. */
type MatrixSummary = {
  counts: Uint32Array;
  probabilities: Float32Array;
  totals: Uint32Array;
  ignored: number;
  classEdges: Float32Array;
  classCount: number;
  lagEdges: Float32Array;
  lagClassCount: number;
  spatialCounts: Uint32Array;
  spatialProbabilities: Float32Array;
  lisaCounts: Uint32Array;
  lisaProbabilities: Float32Array;
  overflow: number;
};

function formatMinutes(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, '0')}`;
}

/** Formats a class edge of log2(1 + count) back as a vertex count. */
function formatEdge(value: number): string {
  if (!Number.isFinite(value)) return '–';
  const count = 2 ** value - 1;
  return count < 10 ? count.toFixed(1) : String(Math.round(count));
}

export const markovMode: SpatialAnalysisModeDefinition = {
  id: 'markov',
  title: 'Markov',
  contributors: [
    'GPUClassBreaks',
    'GPUClassAssignment',
    'GPUTransitionMatrix',
    'GPUSpatialMarkov',
    'GPULISAMarkov',
    'GPUNeighborSearch'
  ],
  description:
    'Do busy cells stay busy? Taxi vertices per cell and time period are classified with one ' +
    'pooled legend; scrub or play the periods, and read the transition matrix, the same matrix ' +
    'conditioned on the neighbors’ class (spatial Markov), and LISA-state transitions.',
  initialViewState: {longitude: -73.985, latitude: 40.735, zoom: 12.2},

  async create(context) {
    const trips = await context.data.getNewYorkTrips();
    context.signal.throwIfAborted();
    const {device} = context;
    const resources = new SpatialAnalysisResources(device, 'markov');
    const grid = binTripVertices(trips.vertexPositions, trips.vertexTimestamps, trips.timeRange);
    const rowCount = grid.columns * grid.rows;
    const pooledCount = rowCount * PERIOD_COUNT;
    const slotCapacity = rowCount * 8 + 8;
    const pooledMask = new Uint32Array(pooledCount);
    for (let period = 0; period < PERIOD_COUNT; period++)
      pooledMask.set(grid.mask, period * rowCount);

    let method: GPUClassBreaksMethod = 'natural-breaks';
    let classCount = MAXIMUM_CLASS_COUNT;
    let significance = 0.1;
    let period = 0;
    let mapView: MapView = 'class';
    let matrixChoice: MatrixChoice = 'pooled';
    let playing = false;
    let lastAdvance = 0;
    let dirty = true;
    let summary: MatrixSummary | null = null;

    // Inputs.
    const valuesBuffer = resources.createBuffer('values', grid.counts);
    const positionsBuffer = resources.createBuffer('positions', grid.positions);
    const cellMaskBuffer = resources.createBuffer('cell-mask', grid.mask);
    const pooledMaskBuffer = resources.createBuffer('pooled-mask', pooledMask);
    // Weights and outputs.
    const offsetsBuffer = resources.createBuffer('offsets', (rowCount + 1) * 4);
    const neighborsBuffer = resources.createBuffer('neighbors', slotCapacity * 4);
    const weightsBuffer = resources.createBuffer('weights', slotCapacity * 4);
    const overflowBuffer = resources.createBuffer('overflow', 4);
    const breaksBuffer = resources.createBuffer('breaks', (MAXIMUM_CLASS_COUNT + 1) * 4);
    const classCountBuffer = resources.createBuffer('class-count', 4);
    const classesBuffer = resources.createBuffer('classes', pooledCount * 4);
    const pooledCountsBuffer = resources.createBuffer(
      'pooled-counts',
      MAXIMUM_CLASS_COUNT ** 2 * 4
    );
    const pooledProbabilitiesBuffer = resources.createBuffer(
      'pooled-probabilities',
      MAXIMUM_CLASS_COUNT ** 2 * 4
    );
    const pooledTotalsBuffer = resources.createBuffer('pooled-totals', MAXIMUM_CLASS_COUNT * 4);
    const pooledIgnoredBuffer = resources.createBuffer('pooled-ignored', 4);
    const spatialCells = LAG_CLASS_COUNT * MAXIMUM_CLASS_COUNT ** 2;
    const spatialCountsBuffer = resources.createBuffer('spatial-counts', spatialCells * 4);
    const spatialProbabilitiesBuffer = resources.createBuffer(
      'spatial-probabilities',
      spatialCells * 4
    );
    const spatialTotalsBuffer = resources.createBuffer(
      'spatial-totals',
      LAG_CLASS_COUNT * MAXIMUM_CLASS_COUNT * 4
    );
    const lagBreaksBuffer = resources.createBuffer('lag-breaks', (LAG_CLASS_COUNT + 1) * 4);
    const lagClassCountBuffer = resources.createBuffer('lag-class-count', 4);
    const lagClassesBuffer = resources.createBuffer('lag-classes', pooledCount * 4);
    const lisaCountsBuffer = resources.createBuffer('lisa-counts', STATE_COUNT ** 2 * 4);
    const lisaProbabilitiesBuffer = resources.createBuffer(
      'lisa-probabilities',
      STATE_COUNT ** 2 * 4
    );
    const lisaTotalsBuffer = resources.createBuffer('lisa-totals', STATE_COUNT * 4);
    const quadrantsBuffer = resources.createBuffer('quadrants', pooledCount * 4);
    const displayBuffer = resources.createBuffer('display', rowCount * 4);

    // Parameters.
    const searchParameters = resources.createParameterBuffer(
      'search-parameters',
      'float32',
      GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
      getGPUNeighborSearchParameterValues({
        bounds: [
          grid.bounds[0] - grid.cellMeters,
          grid.bounds[1] - grid.cellMeters,
          grid.bounds[2] + grid.cellMeters,
          grid.bounds[3] + grid.cellMeters
        ],
        radius: grid.cellMeters * 1.5,
        weightKind: 'binary',
        rowStandardize: true
      })
    );
    const breaksParameters = resources.createParameterBuffer(
      'breaks-parameters',
      'float32',
      getGPUClassBreaksParameterLength(MAXIMUM_CLASS_COUNT)
    );
    const lagParameters = resources.createParameterBuffer(
      'lag-parameters',
      'float32',
      getGPUClassBreaksParameterLength(LAG_CLASS_COUNT),
      getGPUClassBreaksParameterValues(
        {method: 'equal-interval', classCount: LAG_CLASS_COUNT},
        LAG_CLASS_COUNT
      )
    );
    const lisaParameters = resources.createParameterBuffer(
      'lisa-parameters',
      'float32',
      GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
    );
    const displayParameters = resources.createParameterBuffer('display-parameters', 'uint32', 2);

    function writeParameters(): void {
      breaksParameters.write(
        getGPUClassBreaksParameterValues({method, classCount}, MAXIMUM_CLASS_COUNT)
      );
      lisaParameters.write(
        getGPUSpatialAutocorrelationParameterValues({significanceLevel: significance})
      );
    }
    writeParameters();

    // Analysis graph.
    const analysisGraph = new GPUCommandGraph<void>(device, {id: 'markov-analysis'});
    const view = <Format extends GPUVectorFormat>(
      name: string,
      buffer: ReturnType<SpatialAnalysisResources['createBuffer']>,
      format: Format,
      length: number
    ) => importGraphBuffer(analysisGraph, name, buffer, format, length);
    const cellMaskView = view('cell-mask', cellMaskBuffer, 'uint32', rowCount);
    const pooledMaskView = view('pooled-mask', pooledMaskBuffer, 'uint32', pooledCount);
    const valuesView = view('values', valuesBuffer, 'float32', pooledCount);
    const weights = {
      offsets: view('offsets', offsetsBuffer, 'uint32', rowCount + 1),
      neighbors: view('neighbors', neighborsBuffer, 'uint32', slotCapacity),
      weights: view('weights', weightsBuffer, 'float32', slotCapacity)
    };
    const breaksView = view('breaks', breaksBuffer, 'float32', MAXIMUM_CLASS_COUNT + 1);
    const classCountView = view('class-count', classCountBuffer, 'uint32', 1);
    const classesView = view('classes', classesBuffer, 'uint32', pooledCount);
    analysisGraph.add(
      new GPUNeighborSearch({
        id: 'markov-search',
        mode: 'radius',
        positions: view('positions', positionsBuffer, 'float32x2', rowCount),
        mask: cellMaskView,
        parameters: searchParameters.importToGraph(analysisGraph),
        gridSize: [64, 64],
        weights,
        overflow: view('overflow', overflowBuffer, 'uint32', 1)
      })
    );
    analysisGraph.add(
      new GPUClassBreaks({
        id: 'markov-breaks',
        values: valuesView,
        mask: pooledMaskView,
        parameters: breaksParameters.importToGraph(analysisGraph),
        maximumClassCount: MAXIMUM_CLASS_COUNT,
        methods: METHODS as unknown as GPUClassBreaksMethod[],
        output: {breaks: breaksView, classCount: classCountView}
      })
    );
    analysisGraph.add(
      new GPUClassAssignment({
        id: 'markov-assign',
        values: valuesView,
        breaks: breaksView,
        classCount: classCountView,
        mask: pooledMaskView,
        output: classesView
      })
    );
    analysisGraph.add(
      new GPUTransitionMatrix({
        id: 'markov-matrix',
        classes: classesView,
        rows: rowCount,
        periods: PERIOD_COUNT,
        classCount: MAXIMUM_CLASS_COUNT,
        mask: cellMaskView,
        output: {
          counts: view('pooled-counts', pooledCountsBuffer, 'uint32', MAXIMUM_CLASS_COUNT ** 2),
          probabilities: view(
            'pooled-probabilities',
            pooledProbabilitiesBuffer,
            'float32',
            MAXIMUM_CLASS_COUNT ** 2
          ),
          rowTotals: view('pooled-totals', pooledTotalsBuffer, 'uint32', MAXIMUM_CLASS_COUNT),
          ignored: view('pooled-ignored', pooledIgnoredBuffer, 'uint32', 1)
        }
      })
    );
    analysisGraph.add(
      new GPUSpatialMarkov({
        id: 'markov-spatial',
        values: valuesView,
        classes: classesView,
        classCount: MAXIMUM_CLASS_COUNT,
        weights,
        periods: PERIOD_COUNT,
        lagMaximumClassCount: LAG_CLASS_COUNT,
        lagParameters: lagParameters.importToGraph(analysisGraph),
        lagMethods: ['equal-interval'],
        mask: cellMaskView,
        output: {
          counts: view('spatial-counts', spatialCountsBuffer, 'uint32', spatialCells),
          probabilities: view(
            'spatial-probabilities',
            spatialProbabilitiesBuffer,
            'float32',
            spatialCells
          ),
          rowTotals: view(
            'spatial-totals',
            spatialTotalsBuffer,
            'uint32',
            LAG_CLASS_COUNT * MAXIMUM_CLASS_COUNT
          ),
          lagBreaks: view('lag-breaks', lagBreaksBuffer, 'float32', LAG_CLASS_COUNT + 1),
          lagClassCount: view('lag-class-count', lagClassCountBuffer, 'uint32', 1),
          lagClasses: view('lag-classes', lagClassesBuffer, 'uint32', pooledCount)
        }
      })
    );
    analysisGraph.add(
      new GPULISAMarkov({
        id: 'markov-lisa',
        values: valuesView,
        weights,
        periods: PERIOD_COUNT,
        parameters: lisaParameters.importToGraph(analysisGraph),
        mask: cellMaskView,
        output: {
          counts: view('lisa-counts', lisaCountsBuffer, 'uint32', STATE_COUNT ** 2),
          probabilities: view(
            'lisa-probabilities',
            lisaProbabilitiesBuffer,
            'float32',
            STATE_COUNT ** 2
          ),
          rowTotals: view('lisa-totals', lisaTotalsBuffer, 'uint32', STATE_COUNT),
          quadrants: view('quadrants', quadrantsBuffer, 'uint32', pooledCount)
        }
      })
    );
    const analysis = resources.track(analysisGraph.compile());

    // Display graph: gathers one period of the chosen classification, masked cells hidden.
    const displayGraph = new GPUCommandGraph<void>(device, {id: 'markov-display'});
    const display = <Format extends GPUVectorFormat>(
      name: string,
      buffer: ReturnType<SpatialAnalysisResources['createBuffer']>,
      format: Format,
      length: number
    ) => importGraphBuffer(displayGraph, name, buffer, format, length);
    addKernelPass(displayGraph, {
      id: 'markov-display',
      invocationCount: rowCount,
      declarations: `const ROW_COUNT: u32 = ${rowCount}u;`,
      bindings: [
        {
          name: 'parameters',
          view: displayParameters.importToGraph(displayGraph),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'cellMask',
          view: display('cell-mask', cellMaskBuffer, 'uint32', rowCount),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'classes',
          view: display('classes', classesBuffer, 'uint32', pooledCount),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'lagClasses',
          view: display('lag-classes', lagClassesBuffer, 'uint32', pooledCount),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'quadrants',
          view: display('quadrants', quadrantsBuffer, 'uint32', pooledCount),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'shown',
          view: display('display', displayBuffer, 'uint32', rowCount),
          type: 'u32',
          access: 'read_write'
        }
      ],
      body: /* wgsl */ `
  if (cellMask[cellMaskOffset + index] == 0u) {
    shown[shownOffset + index] = 0xffffffffu;
    return;
  }
  let source = parameters[parametersOffset] * ROW_COUNT + index;
  let mapView = parameters[parametersOffset + 1u];
  var value = classes[classesOffset + source];
  if (mapView == 1u) {
    value = lagClasses[lagClassesOffset + source];
  } else if (mapView == 2u) {
    value = quadrants[quadrantsOffset + source];
  }
  shown[shownOffset + index] = value;`
    });
    const displayCompiled = resources.track(displayGraph.compile());

    // Summary readback.
    const sources = [
      {buffer: pooledCountsBuffer, size: MAXIMUM_CLASS_COUNT ** 2 * 4},
      {buffer: pooledProbabilitiesBuffer, size: MAXIMUM_CLASS_COUNT ** 2 * 4},
      {buffer: pooledTotalsBuffer, size: MAXIMUM_CLASS_COUNT * 4},
      {buffer: pooledIgnoredBuffer, size: 4},
      {buffer: breaksBuffer, size: (MAXIMUM_CLASS_COUNT + 1) * 4},
      {buffer: classCountBuffer, size: 4},
      {buffer: lagBreaksBuffer, size: (LAG_CLASS_COUNT + 1) * 4},
      {buffer: lagClassCountBuffer, size: 4},
      {buffer: spatialCountsBuffer, size: spatialCells * 4},
      {buffer: spatialProbabilitiesBuffer, size: spatialCells * 4},
      {buffer: lisaCountsBuffer, size: STATE_COUNT ** 2 * 4},
      {buffer: lisaProbabilitiesBuffer, size: STATE_COUNT ** 2 * 4},
      {buffer: overflowBuffer, size: 4}
    ];
    const reader = new SummaryReader(resources, 'markov', sources, bytes => {
      let offset = 0;
      const take = (length: number) => {
        const slice = bytes.slice(offset, offset + length * 4);
        offset += length * 4;
        return slice;
      };
      const unsigned = (length: number) => new Uint32Array(take(length));
      const floating = (length: number) => new Float32Array(take(length));
      summary = {
        counts: unsigned(MAXIMUM_CLASS_COUNT ** 2),
        probabilities: floating(MAXIMUM_CLASS_COUNT ** 2),
        totals: unsigned(MAXIMUM_CLASS_COUNT),
        ignored: unsigned(1)[0],
        classEdges: floating(MAXIMUM_CLASS_COUNT + 1),
        classCount: unsigned(1)[0],
        lagEdges: floating(LAG_CLASS_COUNT + 1),
        lagClassCount: unsigned(1)[0],
        spatialCounts: unsigned(spatialCells),
        spatialProbabilities: floating(spatialCells),
        lisaCounts: unsigned(STATE_COUNT ** 2),
        lisaProbabilities: floating(STATE_COUNT ** 2),
        overflow: unsigned(1)[0]
      };
      showSummary();
    });

    // Controls.
    const periodControl = context.controls.addSlider({
      label: 'Time period',
      min: 0,
      max: PERIOD_COUNT - 1,
      step: 1,
      value: period,
      format: value => {
        const start = grid.timeStart + value * grid.periodSeconds;
        return `${value + 1} of ${PERIOD_COUNT} (${formatMinutes(start)}–${formatMinutes(start + grid.periodSeconds)})`;
      },
      onChange: value => {
        period = value;
      }
    });
    context.controls.addToggle({
      label: 'Play periods',
      value: playing,
      onChange: value => {
        playing = value;
      }
    });
    context.controls.addSelect<MapView>({
      label: 'Map shows',
      options: [
        {value: 'class', label: 'Activity class (pooled breaks)'},
        {value: 'lag', label: 'Neighbors’ class (spatial lag)'},
        {value: 'lisa', label: 'LISA state (local Moran)'}
      ],
      value: mapView,
      onChange: value => {
        mapView = value;
        context.updateLayers();
      }
    });
    context.controls.addSelect<GPUClassBreaksMethod>({
      label: 'Class method (per-frame)',
      options: METHODS.map(value => ({value, label: METHOD_LABELS[value]})),
      value: method,
      onChange: value => {
        method = value;
        writeParameters();
        dirty = true;
      }
    });
    context.controls.addSlider({
      label: 'Classes (per-frame, up to 5)',
      min: 2,
      max: MAXIMUM_CLASS_COUNT,
      step: 1,
      value: classCount,
      onChange: value => {
        classCount = value;
        writeParameters();
        dirty = true;
      }
    });
    context.controls.addSlider({
      label: 'LISA significance level',
      min: 0.01,
      max: 0.2,
      step: 0.01,
      value: significance,
      format: value => value.toFixed(2),
      onChange: value => {
        significance = value;
        writeParameters();
        dirty = true;
      }
    });
    context.controls.addSelect<MatrixChoice>({
      label: 'Transition matrix shown',
      options: [
        {value: 'pooled', label: 'All cells (Markov)'},
        {value: 'lag0', label: 'Spatial Markov: low-lag neighbors'},
        {value: 'lag1', label: 'Spatial Markov: mid-lag neighbors'},
        {value: 'lag2', label: 'Spatial Markov: high-lag neighbors'},
        {value: 'lisa', label: 'LISA Markov (local Moran states)'}
      ],
      value: matrixChoice,
      onChange: value => {
        matrixChoice = value;
        showSummary();
      }
    });
    context.controls.addLegend({
      title: 'Activity class, low to high (also spatial-lag class 1-3)',
      entries: CLASS_COLORS.map((color, index) => ({color, label: String(index + 1)}))
    });
    context.controls.addLegend({
      title: 'LISA state',
      entries: LISA_COLORS.map((color, index) => ({color, label: LISA_LABELS[index]}))
    });
    context.controls.addReadout(
      'Active cells',
      `${formatCount(grid.activeCount)} of ${formatCount(rowCount)}`
    );
    context.controls.addReadout('Cell size', `${grid.cellMeters.toFixed(0)} m`);
    const edgesReadout = context.controls.addReadout('Class edges (vertices/cell)');
    const lagEdgesReadout = context.controls.addReadout('Lag edges (geometric mean)');
    const transitionsReadout = context.controls.addReadout('Transitions counted');
    const stayReadout = context.controls.addReadout('Stay (all)');
    const stayLagReadout = context.controls.addReadout('Stay by lag L / M / H');
    const stayLisaReadout = context.controls.addReadout('Stay (LISA states)');
    const flagsReadout = context.controls.addReadout('Search overflow');
    context.controls.addReadout('Data', trips.attribution);

    // Heat table appended to the readout block (the panel has no table control).
    const matrixTitle = document.createElement('div');
    matrixTitle.style.cssText = 'margin-top:8px;color:#a9b8d0';
    const matrixTable = document.createElement('table');
    matrixTable.style.cssText =
      'border-collapse:collapse;margin-top:3px;width:100%;text-align:center;font:10px/1.3 ui-monospace,monospace';
    document.querySelector('[data-mode-readouts]')?.append(matrixTitle, matrixTable);

    function getStay(counts: Uint32Array, size: number, offset = 0): string {
      let diagonal = 0;
      let total = 0;
      for (let from = 0; from < size; from++) {
        for (let to = 0; to < size; to++) {
          const value = counts[offset + from * size + to];
          total += value;
          if (from === to) diagonal += value;
        }
      }
      return total > 0 ? (diagonal / total).toFixed(2) : '–';
    }

    function sum(counts: Uint32Array): number {
      let total = 0;
      for (const value of counts) total += value;
      return total;
    }

    function showSummary(): void {
      if (!summary) return;
      const data = summary;
      const activeClasses = Math.min(MAXIMUM_CLASS_COUNT, Math.max(1, data.classCount));
      edgesReadout.setValue(
        Array.from(data.classEdges.subarray(0, activeClasses + 1), formatEdge).join(' | ')
      );
      lagEdgesReadout.setValue(
        Array.from(data.lagEdges.subarray(0, Math.max(1, data.lagClassCount) + 1), formatEdge).join(
          ' | '
        )
      );
      transitionsReadout.setValue(
        `${formatCount(sum(data.counts))} (${formatCount(data.ignored)} cell-periods skipped)`
      );
      stayReadout.setValue(getStay(data.counts, MAXIMUM_CLASS_COUNT));
      stayLagReadout.setValue(
        [0, 1, 2]
          .map(lag =>
            getStay(data.spatialCounts, MAXIMUM_CLASS_COUNT, lag * MAXIMUM_CLASS_COUNT ** 2)
          )
          .join(' / ')
      );
      stayLisaReadout.setValue(getStay(data.lisaCounts, STATE_COUNT));
      flagsReadout.setValue(data.overflow ? 'yes (weights truncated)' : 'no');

      let shownCounts = data.counts;
      let shownProbabilities = data.probabilities;
      let offset = 0;
      let labels: readonly string[] = Array.from(
        {length: MAXIMUM_CLASS_COUNT},
        (_, index) => `c${index + 1}`
      );
      let limit = activeClasses;
      let title = 'P(class at t+1 | class at t), all cells';
      if (matrixChoice === 'lisa') {
        shownCounts = data.lisaCounts;
        shownProbabilities = data.lisaProbabilities;
        labels = LISA_LABELS;
        limit = STATE_COUNT;
        title = 'P(LISA state at t+1 | state at t)';
      } else if (matrixChoice !== 'pooled') {
        const lag = Number(matrixChoice.slice(3));
        offset = lag * MAXIMUM_CLASS_COUNT ** 2;
        shownCounts = data.spatialCounts;
        shownProbabilities = data.spatialProbabilities;
        title = `P(class at t+1 | class at t), neighbors in lag class ${lag + 1} of ${LAG_CLASS_COUNT}`;
      }
      matrixTitle.textContent = title;
      // Every matrix is stored with a stride of five classes or states.
      const stride = MAXIMUM_CLASS_COUNT;
      const headerStyle = 'color:#7f90ad;font-weight:400';
      const rows: string[] = [
        `<tr><th style="${headerStyle}">from\\to</th>${labels
          .slice(0, limit)
          .map(label => `<th style="${headerStyle}">${label}</th>`)
          .join('')}<th style="${headerStyle}">n</th></tr>`
      ];
      for (let from = 0; from < limit; from++) {
        let rowTotal = 0;
        const cells: string[] = [];
        for (let to = 0; to < limit; to++) {
          const index = offset + from * stride + to;
          const count = shownCounts[index];
          rowTotal += count;
          const probability = shownProbabilities[index];
          cells.push(
            `<td title="${count} transitions" style="padding:2px 1px;border:1px solid #0b1226;background:rgba(77,163,255,${(0.08 + 0.85 * probability).toFixed(2)});color:#fff">${probability.toFixed(2)}</td>`
          );
        }
        rows.push(
          `<tr><th style="${headerStyle};text-align:right;padding-right:3px">${labels[from]}</th>${cells.join('')}<td style="color:#7f90ad;padding-left:3px">${formatCount(rowTotal)}</td></tr>`
        );
      }
      matrixTable.innerHTML = rows.join('');
    }

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [analysis, displayCompiled],
      encode(commandEncoder, frame) {
        if (playing && frame.timeSeconds - lastAdvance > PLAY_SECONDS_PER_PERIOD) {
          lastAdvance = frame.timeSeconds;
          period = (period + 1) % PERIOD_COUNT;
          periodControl.setValue(period);
        }
        if (dirty || frame.frameIndex < 2) {
          analysis.encode(commandEncoder, {parameters: undefined});
          dirty = false;
          reader.request(commandEncoder);
        }
        displayParameters.write(
          Uint32Array.of(period, mapView === 'class' ? 0 : mapView === 'lag' ? 1 : 2)
        );
        displayCompiled.encode(commandEncoder, {parameters: undefined});
        reader.flush(commandEncoder);
      },
      getLayers(): Layer[] {
        return [
          new SpatialAnalysisRasterLayer({
            id: `markov-${mapView}`,
            coordinateOrigin: [trips.origin[0], trips.origin[1], 0],
            gridSize: [grid.columns, grid.rows],
            bounds: grid.bounds,
            values: displayBuffer,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: mapView === 'lisa' ? LISA_COLORS : CLASS_COLORS,
            noDataValue: NO_CLASS,
            noDataColor: NO_DATA_COLOR,
            color: [255, 255, 255, 255]
          })
        ];
      },
      destroy() {
        reader.stop();
        matrixTitle.remove();
        matrixTable.remove();
        resources.destroy();
      }
    };
    return instance;
  }
};
