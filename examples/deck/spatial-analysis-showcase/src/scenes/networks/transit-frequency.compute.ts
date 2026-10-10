// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {GPUGroupStatistics} from '@luma.gl/experimental/gpu-dataframe';
import {
  getGPULineDensityParameterValues,
  GPULineDensity,
  GPUTrajectoryMetrics,
  GPU_LINE_DENSITY_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {
  getServiceClassColors,
  getServiceClassIndex,
  getServiceClassLabel,
  formatServiceHeadway,
  SERVICE_LABELS,
  SERVICE_BREAKS
} from './randstad-network-cartography';
import {SpatialAnalysisRasterLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {binValues, histogramChart} from '../movement/f-chart-helpers';
import type {SceneContext, SceneInstance} from '../scene';
import {
  loadTransitTrips,
  TRANSIT_MODE_LABELS,
  TRANSIT_MODES,
  TRANSIT_WINDOW_SECONDS
} from './transit-data';

/** Option state of the transit frequency scene. */
export type TransitFrequencyOptions = {
  modeFilter: string;
  cellMeters: number;
  representation: 'cells' | 'routes' | 'both';
  frequentOnly: boolean;
  serviceStyle: 'continuous' | 'classes';
  showRoutes: boolean;
};

const COLUMNS = 384;
const ROWS = 384;
const CELL_COUNT = COLUMNS * ROWS;
const WINDOW_HOURS = TRANSIT_WINDOW_SECONDS / 3600;
const TOP_LINES = 8;
const CONTINUOUS_SERVICE_MAXIMUM = 60;

type Variant = {
  key: string;
  compiled: CompiledGPUCommandGraph<void>;
};

/** Imports every buffer once per graph (a graph rejects two imports of one buffer). */
function createImporter(graph: GPUCommandGraph<void>) {
  const cache = new Map<Buffer, GraphDataView>();
  return <Format extends 'uint32' | 'float32' | 'float32x2'>(
    buffer: Buffer,
    format: Format,
    length: number
  ) => {
    let view = cache.get(buffer);
    if (!view) {
      view = importGraphBuffer(graph, buffer.id, buffer, format, length) as GraphDataView;
      cache.set(buffer, view);
    }
    return view as unknown as GraphDataView<Format>;
  };
}

/**
 * Transit frequency. `GPULineDensity` clips every trip segment to a grid and sums the length per
 * cell; length divided by the cell width is about the number of vehicle passes through the cell,
 * and divided by the window it is vehicles per hour. One graph per mode filter is compiled the
 * first time the mode is chosen. `GPUTrajectoryMetrics` measures every trip once and
 * `GPUGroupStatistics` summarises trips, distance and speed per line.
 */
export async function createTransitFrequency(
  ctx: SceneContext<TransitFrequencyOptions>
): Promise<SceneInstance<TransitFrequencyOptions>> {
  const trips = loadTransitTrips(ctx.datasets.get('poopdeck-gtfs-nl'));
  const {device} = ctx;
  const {trackCount, vertexCount, segmentCount} = trips;
  const resources = new SpatialAnalysisResources(device, 'frequency');
  const coordinateOrigin: [number, number, number] = [trips.origin[0], trips.origin[1], 0];

  // Grid center: the middle of the trips' planar bounds.
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    minX = Math.min(minX, trips.positions[vertex * 2]);
    maxX = Math.max(maxX, trips.positions[vertex * 2]);
    minY = Math.min(minY, trips.positions[vertex * 2 + 1]);
    maxY = Math.max(maxY, trips.positions[vertex * 2 + 1]);
  }
  const center: [number, number] = [(minX + maxX) / 2, (minY + maxY) / 2];

  // ---- Static inputs ----------------------------------------------------------------------------
  const positionsBuffer = resources.createBuffer('positions', trips.positions);
  const timestampsBuffer = resources.createBuffer('timestamps', trips.timestamps);
  const offsetsBuffer = resources.createBuffer('trip-offsets', trips.offsets);
  const segmentsBuffer = resources.createBuffer('segments', trips.segments);
  const lengthsBuffer = resources.createBuffer('lengths', CELL_COUNT * 4);
  const overflowBuffer = resources.createBuffer('overflow', 4);
  const totalRecordsBuffer = resources.createBuffer('total-records', 4);
  const gridParameters = resources.createParameterBuffer(
    'grid',
    'float32',
    GPU_LINE_DENSITY_PARAMETER_LENGTH
  );
  const dataFrameBuffer = resources.createBuffer('data-frame', 4 * 4 * 4);
  const hoverCellBuffer = resources.createBuffer('hover-cell', 4 * 4 * 4);

  // ---- Line density graphs: one per mode filter, compiled on first use --------------------------
  const variants = new Map<string, Variant>();

  function createVariant(filter: string): Variant {
    let positions = positionsBuffer;
    let offsets = offsetsBuffer;
    let rows = vertexCount;
    let paths = trackCount;
    if (filter !== 'all') {
      const wanted = TRANSIT_MODES.indexOf(filter as never);
      const selected: number[] = [];
      for (let trip = 0; trip < trackCount; trip++)
        if (trips.mode[trip] === wanted) selected.push(trip);
      let total = 0;
      for (const trip of selected) total += trips.offsets[trip + 1] - trips.offsets[trip];
      const subsetPositions = new Float32Array(total * 2);
      const subsetOffsets = new Uint32Array(selected.length + 1);
      let cursor = 0;
      selected.forEach((trip, index) => {
        subsetOffsets[index] = cursor;
        const from = trips.offsets[trip];
        const to = trips.offsets[trip + 1];
        subsetPositions.set(trips.positions.subarray(from * 2, to * 2), cursor * 2);
        cursor += to - from;
      });
      subsetOffsets[selected.length] = cursor;
      positions = resources.createBuffer(`positions-${filter}`, subsetPositions);
      offsets = resources.createBuffer(`offsets-${filter}`, subsetOffsets);
      rows = total;
      paths = selected.length;
    }
    const graph = new GPUCommandGraph<void>(device, {id: `frequency-density-${filter}`});
    const view = createImporter(graph);
    graph.add(
      new GPULineDensity({
        id: 'line-density',
        positions: view(positions, 'float32x2', rows),
        pathOffsets: view(offsets, 'uint32', paths + 1),
        columns: COLUMNS,
        rows: ROWS,
        spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'native'},
        parameters: gridParameters.importToGraph(graph),
        output: {
          lengths: view(lengthsBuffer, 'float32', CELL_COUNT),
          overflow: view(overflowBuffer, 'uint32', 1),
          totalRecords: view(totalRecordsBuffer, 'uint32', 1)
        }
      })
    );
    const variant = {key: filter, compiled: resources.track(graph.compile())};
    variants.set(filter, variant);
    return variant;
  }

  // ---- Per-trip metrics and per-line statistics (computed once) ---------------------------------
  const lineCount = trips.lines.length;
  const lineKeys = resources.createBuffer('line-keys', trips.line);
  const trackLengths = resources.createBuffer('track-lengths', trackCount * 4);
  const trackDurations = resources.createBuffer('track-durations', trackCount * 4);
  const averageSpeeds = resources.createBuffer('average-speeds', trackCount * 4);
  const outputKeys = resources.createBuffer('stat-keys', lineCount * 4);
  const lineTrips = resources.createBuffer('stat-trips', lineCount * 4);
  const lineDistance = resources.createBuffer('stat-distance', lineCount * 4);
  const lineMeanSpeed = resources.createBuffer('stat-mean-speed', lineCount * 4);
  const lineMedianSpeed = resources.createBuffer('stat-median-speed', lineCount * 4);
  const lineMeanDuration = resources.createBuffer('stat-mean-duration', lineCount * 4);
  const statCount = resources.createBuffer('stat-count', 4);
  const statOverflow = resources.createBuffer('stat-overflow', 4);
  const statisticsGraph = new GPUCommandGraph<void>(device, {id: 'frequency-statistics'});
  {
    const view = createImporter(statisticsGraph);
    const lengths = view(trackLengths, 'float32', trackCount);
    const durations = view(trackDurations, 'float32', trackCount);
    const speeds = view(averageSpeeds, 'float32', trackCount);
    statisticsGraph.add(
      new GPUTrajectoryMetrics({
        spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'native'},
        id: 'trip-metrics',
        positions: view(positionsBuffer, 'float32x2', vertexCount),
        timestamps: view(timestampsBuffer, 'float32', vertexCount),
        trackOffsets: view(offsetsBuffer, 'uint32', trackCount + 1),
        trackLengths: lengths,
        trackDurations: durations,
        averageSpeeds: speeds
      })
    );
    statisticsGraph.add(
      new GPUGroupStatistics({
        id: 'line-statistics',
        keys: view(lineKeys, 'uint32', trackCount),
        keyCount: lineCount,
        columns: [
          {
            values: lengths,
            statistics: ['sum'],
            output: {sumValues: view(lineDistance, 'float32', lineCount)}
          },
          {
            values: speeds,
            statistics: ['mean', 'median'],
            output: {
              means: view(lineMeanSpeed, 'float32', lineCount),
              medians: view(lineMedianSpeed, 'float32', lineCount)
            }
          },
          {
            values: durations,
            statistics: ['mean'],
            output: {means: view(lineMeanDuration, 'float32', lineCount)}
          }
        ],
        output: {
          keys: view(outputKeys, 'uint32', lineCount),
          counts: view(lineTrips, 'uint32', lineCount),
          count: view(statCount, 'uint32', 1),
          overflow: view(statOverflow, 'uint32', 1)
        }
      })
    );
  }
  const statisticsCompiled = resources.track(statisticsGraph.compile());

  // ---- State ------------------------------------------------------------------------------------
  let destroyed = false;
  let densityDirty = true;
  let statisticsPending = true;
  let activeFilter = ctx.options.modeFilter;
  createVariant('all');
  if (activeFilter !== 'all') createVariant(activeFilter);
  let cells = new Float32Array(CELL_COUNT);
  let gridMin: [number, number] = [0, 0];
  let hoveredCell = -1;
  let legendGround: 'light' | 'dark' | null = null;

  /** A deterministic, scene-local comparison: each segment contributes to its midpoint cell. */
  function getBusiestCellSummary(cellMeters: number): number {
    const minX = center[0] - (COLUMNS * cellMeters) / 2;
    const minY = center[1] - (ROWS * cellMeters) / 2;
    const totals = new Map<number, number>();
    for (let segment = 0; segment < segmentCount; segment++) {
      const offset = segment * 4;
      const x0 = trips.segments[offset];
      const y0 = trips.segments[offset + 1];
      const x1 = trips.segments[offset + 2];
      const y1 = trips.segments[offset + 3];
      const column = Math.floor(((x0 + x1) / 2 - minX) / cellMeters);
      const row = Math.floor(((y0 + y1) / 2 - minY) / cellMeters);
      if (column < 0 || column >= COLUMNS || row < 0 || row >= ROWS) continue;
      const key = row * COLUMNS + column;
      const length = Math.hypot(x1 - x0, y1 - y0);
      totals.set(key, (totals.get(key) ?? 0) + length);
    }
    let busiest = 0;
    for (const length of totals.values()) busiest = Math.max(busiest, length);
    return busiest / cellMeters / WINDOW_HOURS;
  }

  const busiestByCellSize = [200, 400, 800].map(getBusiestCellSummary);
  ctx.setChart('cellComparison', {
    kind: 'bars',
    height: 112,
    values: busiestByCellSize,
    labels: ['200 m', '400 m', '800 m'],
    yLabel: 'busiest veh/h',
    formatY: value => value.toFixed(0),
    description:
      'Busiest data-derived grid cell at each size. This deterministic midpoint summary uses the same scheduled segments and fixed two-hour normalisation as the map; the map itself clips every segment exactly.'
  });

  function getGrid() {
    const cellMeters = ctx.options.cellMeters;
    const min: [number, number] = [
      center[0] - (COLUMNS * cellMeters) / 2,
      center[1] - (ROWS * cellMeters) / 2
    ];
    gridMin = min;
    return {
      min,
      max: [min[0] + COLUMNS * cellMeters, min[1] + ROWS * cellMeters] as [number, number]
    };
  }

  function writeDataFrame(): void {
    const frame = new Float32Array([
      minX,
      minY,
      maxX,
      minY,
      maxX,
      minY,
      maxX,
      maxY,
      maxX,
      maxY,
      minX,
      maxY,
      minX,
      maxY,
      minX,
      minY
    ]);
    dataFrameBuffer.write(frame);
  }

  function publishFurniture(): void {
    ctx.setFurniture({
      title: {
        subtitle: `Scheduled service · 07:00–09:00 CEST · ${ctx.options.cellMeters} m cells`,
        sample: '3 July 2026 · timetable, not observed vehicles'
      },
      scaleBar: {units: 'metric'}
    });
  }

  function publishLegendGround(): void {
    const ground = ctx.ground();
    if (ground === legendGround) return;
    legendGround = ground;
    ctx.setLegendData('ground', ground);
  }

  function updateHoverCell(cell: number): void {
    if (cell === hoveredCell) return;
    hoveredCell = cell;
    if (cell >= 0) {
      const column = cell % COLUMNS;
      const row = Math.floor(cell / COLUMNS);
      const west = gridMin[0] + column * ctx.options.cellMeters;
      const south = gridMin[1] + row * ctx.options.cellMeters;
      const east = west + ctx.options.cellMeters;
      const north = south + ctx.options.cellMeters;
      hoverCellBuffer.write(
        new Float32Array([
          west,
          south,
          east,
          south,
          east,
          south,
          east,
          north,
          east,
          north,
          west,
          north,
          west,
          north,
          west,
          south
        ])
      );
    }
    ctx.requestLayers();
  }

  function writeGridParameters(): void {
    const grid = getGrid();
    gridParameters.write(
      getGPULineDensityParameterValues({
        minX: grid.min[0],
        minY: grid.min[1],
        cellWidth: ctx.options.cellMeters,
        cellHeight: ctx.options.cellMeters
      })
    );
    densityDirty = true;
  }

  /** Vehicles per hour implied by a cell's summed line length. */
  const toVehiclesPerHour = (length: number) => length / ctx.options.cellMeters / WINDOW_HOURS;

  // ---- Readbacks --------------------------------------------------------------------------------
  const densityReader = new SummaryReader(
    resources,
    'frequency-density',
    [
      {buffer: lengthsBuffer, size: CELL_COUNT * 4},
      {buffer: overflowBuffer, size: 4},
      {buffer: totalRecordsBuffer, size: 4}
    ],
    bytes => {
      if (destroyed) return;
      const floats = new Float32Array(bytes);
      const words = new Uint32Array(bytes);
      cells = floats.slice(0, CELL_COUNT);
      const overflow = words[CELL_COUNT];
      const totalRecords = words[CELL_COUNT + 1];
      let totalMeters = 0;
      let occupied = 0;
      let busiest = 0;
      for (let cell = 0; cell < CELL_COUNT; cell++) {
        const length = cells[cell];
        if (length > 0) {
          totalMeters += length;
          occupied++;
          busiest = Math.max(busiest, length);
        }
      }
      ctx.setReadout('vehicleKilometers', `${formatCount(totalMeters / 1000)} vehicle-km`);
      ctx.setReadout('occupied', `${formatCount(occupied)} of ${formatCount(CELL_COUNT)} cells`);
      ctx.setReadout('busiest', `${formatCount(toVehiclesPerHour(busiest))} vehicles / h`);
      ctx.setReadout(
        'pieces',
        `${formatCount(totalRecords)} segment-cell pieces${overflow ? ' (OVERFLOW: lengths are low)' : ''}`
      );
      const values = new Float32Array(occupied);
      let cursor = 0;
      for (let cell = 0; cell < CELL_COUNT; cell++) {
        if (cells[cell] > 0) values[cursor++] = toVehiclesPerHour(cells[cell]);
      }
      if (ctx.options.serviceStyle === 'classes') {
        const classCounts = new Float64Array(SERVICE_LABELS.length);
        for (const value of values) classCounts[getServiceClassIndex(value)]++;
        ctx.setChart('concentration', {
          kind: 'bars',
          height: 120,
          values: classCounts,
          labels: SERVICE_LABELS,
          colors: getServiceClassColors(ctx.ground() === 'dark').map((color, index) =>
            ctx.options.frequentOnly && index < 3
              ? ([color[0], color[1], color[2], 70] as const)
              : color
          ),
          highlight: ctx.options.frequentOnly ? [3, 4, 5] : undefined,
          yLabel: 'cells',
          description: ctx.options.frequentOnly
            ? 'Cells by the six fixed scheduled-service classes. The highlighted retained classes begin at 8 both-direction vehicles per hour, the approximate 15-minute-per-direction analytical threshold.'
            : 'Cells by the six fixed scheduled-service classes used by the raster, tooltip and legend.'
        });
      } else {
        const maximum = CONTINUOUS_SERVICE_MAXIMUM;
        ctx.setChart(
          'concentration',
          histogramChart(binValues(values, 0, maximum, 20), 0, maximum, {
            xLabel: `vehicles per hour through a ${ctx.options.cellMeters} m cell (last bin: and above)`,
            yLabel: 'cells',
            formatX: value => `${Math.round(value)}`,
            description:
              'Continuous square-root 0–60 veh/h view of occupied cells; the final bin contains values at or above 60.'
          })
        );
      }
    }
  );

  const statisticsReader = new SummaryReader(
    resources,
    'frequency-statistics',
    [
      {buffer: lineTrips, size: lineCount * 4},
      {buffer: lineDistance, size: lineCount * 4},
      {buffer: lineMeanSpeed, size: lineCount * 4},
      {buffer: lineMedianSpeed, size: lineCount * 4},
      {buffer: lineMeanDuration, size: lineCount * 4},
      {buffer: statOverflow, size: 4}
    ],
    bytes => {
      if (destroyed) return;
      const words = new Uint32Array(bytes);
      const floats = new Float32Array(bytes);
      const tripsOf = words.subarray(0, lineCount);
      const distance = floats.subarray(lineCount, lineCount * 2);
      const medianSpeed = floats.subarray(lineCount * 3, lineCount * 4);
      const meanDuration = floats.subarray(lineCount * 4, lineCount * 5);
      // Per mode: trips, distance and speed from the per-line statistics.
      const modeTrips = new Float64Array(TRANSIT_MODES.length);
      const modeMeters = new Float64Array(TRANSIT_MODES.length);
      const modeSeconds = new Float64Array(TRANSIT_MODES.length);
      for (let line = 0; line < lineCount; line++) {
        const mode = trips.lines[line].mode;
        modeTrips[mode] += tripsOf[line];
        modeMeters[mode] += distance[line];
        modeSeconds[mode] += meanDuration[line] * tripsOf[line];
      }
      const modeSpeed = TRANSIT_MODES.map((_, mode) =>
        modeSeconds[mode] > 0 ? (modeMeters[mode] / modeSeconds[mode]) * 3.6 : 0
      );
      ctx.setChart('modeSpeed', {
        kind: 'bars',
        height: 110,
        values: modeSpeed,
        labels: TRANSIT_MODES.map(mode => TRANSIT_MODE_LABELS[mode]),
        yLabel: 'km/h',
        formatY: value => value.toFixed(0),
        description:
          'Mean scheduled speed of the trips of each mode inside the window, total distance over total time, dwell at stops included.'
      });
      ctx.setReadout(
        'modeTable',
        TRANSIT_MODES.map(
          (mode, index) =>
            `${TRANSIT_MODE_LABELS[mode].padEnd(6)} ${formatCount(modeTrips[index]).padStart(5)} trips ${formatCount(modeMeters[index] / 1000).padStart(7)} km ${modeSpeed[index].toFixed(0).padStart(3)} km/h`
        ).join('\n')
      );
      // Busiest lines by trips in the window.
      const order = Array.from({length: lineCount}, (_, line) => line)
        .filter(line => trips.lines[line].name !== 'Flex')
        .sort((a, b) => tripsOf[b] - tripsOf[a])
        .slice(0, TOP_LINES);
      ctx.setChart('busiestLines', {
        kind: 'bars',
        height: 130,
        values: order.map(line => tripsOf[line] / WINDOW_HOURS),
        labels: order.map(
          line =>
            `${TRANSIT_MODE_LABELS[TRANSIT_MODES[trips.lines[line].mode]]} ${trips.lines[line].name}`
        ),
        yLabel: 'trips per hour',
        formatY: value => value.toFixed(0),
        description:
          'The lines with most scheduled trips in the window, both directions and every city that uses the same line number counted together. On-demand Flex services are left out.'
      });
      const top = order[0];
      ctx.setReadout(
        'topLine',
        top === undefined
          ? null
          : `${TRANSIT_MODE_LABELS[TRANSIT_MODES[trips.lines[top].mode]]} ${trips.lines[top].name}: ${formatCount(tripsOf[top])} trips, median ${(medianSpeed[top] * 3.6).toFixed(0)} km/h`
      );
      const flex = trips.lines.findIndex(line => line.name === 'Flex');
      ctx.setReadout(
        'flex',
        flex < 0
          ? 'none'
          : `${formatCount(tripsOf[flex])} trips (${((tripsOf[flex] / trackCount) * 100).toFixed(0)}% of all)`
      );
    }
  );

  writeGridParameters();
  writeDataFrame();
  publishFurniture();
  publishLegendGround();
  ctx.setReadout(
    'trips',
    `${formatCount(trackCount)} trips, ${formatCount(segmentCount)} segments`
  );

  function cellAt(coordinate: readonly [number, number] | null): number {
    if (!coordinate) return -1;
    const [x, y] = trips.project(coordinate[0], coordinate[1]);
    const cellMeters = ctx.options.cellMeters;
    const column = Math.floor((x - gridMin[0]) / cellMeters);
    const row = Math.floor((y - gridMin[1]) / cellMeters);
    if (column < 0 || column >= COLUMNS || row < 0 || row >= ROWS) return -1;
    return row * COLUMNS + column;
  }

  // ---- Instance ---------------------------------------------------------------------------------
  return {
    getCompiledGraphs: () => [
      ...[...variants.values()].map(variant => variant.compiled),
      statisticsCompiled
    ],

    setOption(id, _value, state) {
      switch (id) {
        case 'modeFilter':
          activeFilter = state.modeFilter;
          if (!variants.has(activeFilter)) createVariant(activeFilter);
          densityDirty = true;
          ctx.requestLayers();
          break;
        case 'cellMeters':
          writeGridParameters();
          publishFurniture();
          updateHoverCell(-1);
          ctx.requestLayers();
          break;
        case 'serviceStyle':
        case 'frequentOnly':
          densityDirty = true;
          ctx.requestLayers();
          break;
        default:
          ctx.requestLayers();
      }
    },

    onThemeChange() {
      publishLegendGround();
      densityDirty = true;
      ctx.requestLayers();
    },

    encode(commandEncoder) {
      if (statisticsPending) {
        statisticsCompiled.encode(commandEncoder, {parameters: undefined});
        statisticsPending = false;
        statisticsReader.request(commandEncoder);
      } else {
        statisticsReader.flush(commandEncoder);
      }
      const variant = variants.get(activeFilter);
      if (densityDirty && variant) {
        variant.compiled.encode(commandEncoder, {parameters: undefined});
        densityDirty = false;
        densityReader.request(commandEncoder);
      } else {
        densityReader.flush(commandEncoder);
      }
    },

    getLayers() {
      const options = ctx.options;
      const dark = ctx.ground() === 'dark';
      publishLegendGround();
      const grid = getGrid();
      const layers: Layer[] = [];
      const showRoutes = options.showRoutes && options.representation !== 'cells';
      if (showRoutes) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'frequency-routes',
            coordinateOrigin,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            widthPixels: options.representation === 'routes' ? 1.4 : 0.7,
            color: dark
              ? [225, 229, 240, options.representation === 'routes' ? 150 : 52]
              : [44, 50, 62, options.representation === 'routes' ? 160 : 58]
          })
        );
      }
      if (options.representation !== 'routes')
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: 'frequency-density',
            coordinateOrigin,
            gridSize: [COLUMNS, ROWS],
            bounds: [grid.min[0], grid.min[1], grid.max[0], grid.max[1]],
            values: lengthsBuffer,
            valueFormat: 'float32',
            colormap: 'magma',
            valueScale: 1 / (options.cellMeters * WINDOW_HOURS),
            valueRange: [0, CONTINUOUS_SERVICE_MAXIMUM],
            sqrtScale: true,
            discardAtOrBelow: 0,
            ...(options.serviceStyle === 'classes'
              ? {
                  classBreaks: SERVICE_BREAKS,
                  classColors: getServiceClassColors(dark).map((color, index) =>
                    options.frequentOnly && index < 3
                      ? ([color[0], color[1], color[2], 42] as const)
                      : color
                  )
                }
              : {}),
            outlineClasses: {
              color: dark ? [14, 17, 22, 90] : [255, 255, 255, 100],
              widthPixels: 0.35
            },
            opacity: 0.88
          })
        );
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'frequency-data-frame',
          coordinateOrigin,
          segments: dataFrameBuffer,
          instanceCount: 4,
          widthPixels: 0.9,
          dashArray: [5, 4],
          color: dark ? [235, 240, 255, 150] : [42, 52, 72, 145]
        })
      );
      if (hoveredCell >= 0) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'frequency-hover-cell',
            coordinateOrigin,
            segments: hoverCellBuffer,
            instanceCount: 4,
            widthPixels: 2,
            color: dark ? [255, 255, 255, 245] : [20, 28, 42, 235]
          })
        );
      }
      return layers;
    },

    getTooltip(event) {
      const cell = cellAt(event.coordinate);
      if (cell < 0) {
        updateHoverCell(-1);
        return null;
      }
      updateHoverCell(cell);
      const perHour = toVehiclesPerHour(cells[cell]);
      return cells[cell] > 0
        ? `${getServiceClassLabel(perHour)} · ${perHour.toFixed(perHour < 10 ? 1 : 0)} veh/h through this ${ctx.options.cellMeters} m cell · ${formatServiceHeadway(perHour)}`
        : null;
    },

    onClick(event) {
      const cell = cellAt(event.coordinate);
      if (cell < 0) return false;
      const perHour = toVehiclesPerHour(cells[cell]);
      ctx.setReadout(
        'cellValue',
        `${perHour.toFixed(perHour < 10 ? 1 : 0)} vehicles / h (${formatCount(cells[cell])} m of line in the cell)`
      );
      return true;
    },

    destroy() {
      destroyed = true;
      densityReader.stop();
      statisticsReader.stop();
      resources.destroy();
    }
  };
}
