// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Space-time analysis of the New York trip vertices in one compiled graph:
 *
 * `GPUCalendarBuckets` (UTC offset -> local hour and an hour by weekday matrix) ->
 * `GPUCrossfilter` (hour-of-day brush -> per-vertex selection mask) ->
 * `GPUTemporalReduction` (vertices x cells x 24 time slices: counts and peak speed) ->
 * `GPUEmergingHotSpots` (17-category map of the count cube) and
 * `GPUChangeDetection` (two-slice difference / log ratio / percent change, Mann-Kendall and Sen's
 * slope over the peak-speed stack).
 *
 * Honest data note: trips-v7 carries RELATIVE seconds (about 41 minutes), not calendar times. A
 * real hour-of-day or weekday analysis needs a calendar, so the mode anchors the recording to a
 * nominal Monday and offers a second, clearly labelled clock that stretches the 41 minutes by
 * 240 to cover about seven days. The stretched clock exists to exercise the calendar contributor; the
 * space-time patterns it produces are a property of the 41 minutes, not of a real week. The 24
 * slices are equal absolute time bins over the whole recording (not hour-of-day bins), because
 * `GPUTemporalReduction` bins by time and no contributor converts the calendar hour column to the
 * float times it would need.
 *
 * Every control is a buffer write: UTC offset, clock (timestamp words are rewritten), hour brush,
 * neighbor radius, time window, confidence, and the before and after slices. The graph (34 nodes)
 * measures about 0.8 ms CPU encode and 2 to 3 ms submit-to-idle on a laptop GPU, so it could run
 * every frame; it is encoded only when a control changed (plus warm-up) because the vertex data is
 * static and re-running it would only burn power. The rebuild counter stays 0.
 */

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  GPUCommandGraph,
  GPUHistogram,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {GPUCrossfilter} from '@luma.gl/experimental/gpu-crossfilter';
import {
  getGPUCalendarBucketsParameterValues,
  getGPUTemporalReductionWordParameterValues,
  getInt64TimeWords,
  GPUCalendarBuckets,
  GPUTemporalReduction,
  GPU_CALENDAR_BUCKETS_MATRIX_LENGTH,
  GPU_CALENDAR_BUCKETS_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-dataframe';
import {
  getGPUChangeDetectionParameterValues,
  GPUChangeDetection,
  GPU_CHANGE_DETECTION_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-raster';
import {
  getGPUEmergingHotSpotParameterValues,
  GPUEmergingHotSpots,
  GPU_EMERGING_HOT_SPOT_PARAMETER_LENGTH,
  GPU_EMERGING_HOT_SPOT_STATISTICS_LENGTH
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {importGraphBuffer} from '../graph-buffers';
import {SpatialAnalysisPointLayer} from '../spatial-analysis-layers';
import type {
  SpatialAnalysisModeDefinition,
  SpatialAnalysisModeInstance
} from '../spatial-analysis-mode';
import {formatCount, SpatialAnalysisResources} from '../spatial-analysis-resources';
import {SpaceTimeRasterLayer} from './space-time-layers';

/** Time slices of the cube. Compile-time. Sen's slope limits change detection to 64. */
const SLICE_COUNT = 24;
/** Longest axis of the lattice in cells. Compile-time (derived from the data extent). */
const LATTICE_CELLS = 48;
/** Largest per-frame neighbor radius in cells. Compile-time. */
const MAXIMUM_RADIUS = 4;
/** `GPUTemporalReduction` cell id that skips a row. */
const NO_CELL = 0xffffffff;
const CATEGORY_COUNT = 17;
const SIGNIFICANCE_CLASS_COUNT = 3;
const READBACK_INTERVAL_FRAMES = 10;
const WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'] as const;
/** Monday 2019-04-08 00:00 UTC. A nominal anchor: the dataset has no calendar. */
const BASE_EPOCH_MS = Date.UTC(2019, 3, 8);
const HOUR_MS = 3_600_000;
const RECORDED_START_MS = BASE_EPOCH_MS + 17 * HOUR_MS;
const STRETCH_FACTOR = 240;

const CATEGORY_NAMES = [
  'No pattern',
  'New hot',
  'Consecutive hot',
  'Intensifying hot',
  'Persistent hot',
  'Diminishing hot',
  'Sporadic hot',
  'Oscillating hot',
  'Historical hot',
  'New cold',
  'Consecutive cold',
  'Intensifying cold',
  'Persistent cold',
  'Diminishing cold',
  'Sporadic cold',
  'Oscillating cold',
  'Historical cold'
] as const;
/** Mirrors `spaceTimeCategoryColor` in space-time-layers.ts. */
const CATEGORY_COLORS: readonly (readonly [number, number, number])[] = [
  [60, 66, 80],
  [255, 224, 102],
  [255, 160, 60],
  [165, 0, 38],
  [215, 48, 39],
  [244, 160, 150],
  [253, 208, 162],
  [200, 100, 200],
  [170, 140, 140],
  [160, 230, 255],
  [90, 170, 255],
  [8, 48, 107],
  [33, 102, 172],
  [150, 190, 230],
  [190, 230, 215],
  [120, 100, 220],
  [130, 150, 170]
];
const DIVERGING_COLORS = [
  [33, 102, 172],
  [146, 197, 222],
  [247, 247, 247],
  [244, 165, 130],
  [178, 24, 43]
] as const;
const VIRIDIS_COLORS = [
  [68, 1, 84],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 37]
] as const;

type MapView =
  | 'hot-spots'
  | 'activity'
  | 'difference'
  | 'log-ratio'
  | 'percent-change'
  | 'mann-kendall'
  | 'sen-slope';
type ClockKind = 'recorded' | 'stretched';

/** Value range, colormap and legend of a scalar map view. */
const SCALAR_VIEWS: Record<
  Exclude<MapView, 'hot-spots'>,
  {range: readonly [number, number]; diverging: boolean; unit: string}
> = {
  activity: {range: [0, 16], diverging: false, unit: 'vertices per cell and slice (sqrt scale)'},
  difference: {range: [-10, 10], diverging: true, unit: 'm/s, after minus before'},
  'log-ratio': {range: [-1.5, 1.5], diverging: true, unit: 'ln(after / before)'},
  'percent-change': {range: [-150, 150], diverging: true, unit: '% of the before slice'},
  'mann-kendall': {range: [-3, 3], diverging: true, unit: 'Mann-Kendall Z of peak speed'},
  'sen-slope': {range: [-0.6, 0.6], diverging: true, unit: 'Sen slope, m/s per slice'}
};

export const spaceTimeMode: SpatialAnalysisModeDefinition = {
  id: 'space-time',
  title: 'Space-time',
  contributors: [
    'GPUCalendarBuckets',
    'GPUCrossfilter',
    'GPUTemporalReduction',
    'GPUEmergingHotSpots',
    'GPUChangeDetection',
    'GPUHistogram'
  ],
  description:
    'Taxi vertices binned into a space-time cube (cells x 24 time slices) on the GPU: a calendar ' +
    'hour brush, 17-category emerging hot spots over counts, and two-slice change detection with ' +
    'Mann-Kendall and Sen trends over peak speed. The trips carry only 41 minutes of relative ' +
    'time, so the calendar is a labelled nominal anchor.',
  initialViewState: {longitude: -73.985, latitude: 40.735, zoom: 11.6},

  async create(context) {
    const trips = await context.data.getNewYorkTrips();
    context.signal.throwIfAborted();
    const {device} = context;
    const resources = new SpatialAnalysisResources(device, 'space-time');

    // ---- Static per-vertex columns (built once on the CPU; the cube itself is GPU only). ----
    const vertexCount = trips.vertexTimestamps.length;
    const [timeMinimum, timeMaximum] = trips.timeRange;
    const recordedSeconds = Math.max(timeMaximum - timeMinimum, 1);
    // trips-v7 has a few vertices far from Manhattan (a lattice over the full extent would be
    // millions of meters wide), so the lattice covers the 0.5% to 99.5% quantiles plus a margin and
    // vertices outside it get no cell.
    const getQuantileRange = (axis: 0 | 1): [number, number] => {
      const sorted = new Float32Array(vertexCount);
      for (let vertex = 0; vertex < vertexCount; vertex++) {
        sorted[vertex] = trips.vertexPositions[vertex * 2 + axis];
      }
      sorted.sort();
      const low = sorted[Math.floor(vertexCount * 0.005)];
      const high = sorted[Math.min(vertexCount - 1, Math.floor(vertexCount * 0.995))];
      const margin = Math.max((high - low) * 0.03, 50);
      return [low - margin, high + margin];
    };
    const [minimumX, maximumX] = getQuantileRange(0);
    const [minimumY, maximumY] = getQuantileRange(1);
    const cellSize = Math.max(maximumX - minimumX, maximumY - minimumY, 1) / LATTICE_CELLS;
    const gridWidth = Math.max(2, Math.ceil((maximumX - minimumX) / cellSize + 1e-6));
    const gridHeight = Math.max(2, Math.ceil((maximumY - minimumY) / cellSize + 1e-6));
    const cellCount = gridWidth * gridHeight;
    const binCount = cellCount * SLICE_COUNT;
    const bounds = [
      minimumX,
      minimumY,
      minimumX + gridWidth * cellSize,
      minimumY + gridHeight * cellSize
    ] as const;

    // Cell index and speed (m/s to the next vertex of the same trip) per vertex.
    const cellIds = new Uint32Array(vertexCount);
    const speeds = new Float32Array(vertexCount);
    for (let vertex = 0; vertex < vertexCount; vertex++) {
      const x = trips.vertexPositions[vertex * 2];
      const y = trips.vertexPositions[vertex * 2 + 1];
      if (x < minimumX || x >= bounds[2] || y < minimumY || y >= bounds[3]) {
        cellIds[vertex] = NO_CELL;
        continue;
      }
      const column = Math.min(gridWidth - 1, Math.floor((x - minimumX) / cellSize));
      const row = Math.min(gridHeight - 1, Math.floor((y - minimumY) / cellSize));
      cellIds[vertex] = row * gridWidth + column;
    }
    for (let trip = 0; trip + 1 < trips.tripOffsets.length; trip++) {
      const first = trips.tripOffsets[trip];
      const last = trips.tripOffsets[trip + 1] - 1;
      for (let vertex = first; vertex <= last; vertex++) {
        const from = Math.min(vertex, last - 1);
        if (last <= first) {
          speeds[vertex] = 0;
          continue;
        }
        const dx = trips.vertexPositions[(from + 1) * 2] - trips.vertexPositions[from * 2];
        const dy = trips.vertexPositions[(from + 1) * 2 + 1] - trips.vertexPositions[from * 2 + 1];
        const dt = trips.vertexTimestamps[from + 1] - trips.vertexTimestamps[from];
        speeds[vertex] = dt > 0 ? Math.hypot(dx, dy) / dt : 0;
      }
    }

    let clock: ClockKind = 'recorded';
    const getClockStartMs = () => (clock === 'recorded' ? RECORDED_START_MS : BASE_EPOCH_MS);
    const getClockScale = () => (clock === 'recorded' ? 1 : STRETCH_FACTOR);
    const getSpanMs = () => Math.round(recordedSeconds * 1000 * getClockScale());
    const timestampValues = new BigInt64Array(vertexCount);
    const writeTimestamps = () => {
      const start = getClockStartMs();
      const scale = getClockScale();
      for (let vertex = 0; vertex < vertexCount; vertex++) {
        timestampValues[vertex] = BigInt(
          start + Math.round((trips.vertexTimestamps[vertex] - timeMinimum) * 1000 * scale)
        );
      }
      timestampsBuffer.write(getInt64TimeWords(timestampValues));
    };

    // ---- Buffers. ----
    const positionsBuffer = resources.createBuffer('positions', trips.vertexPositions);
    const timestampsBuffer = resources.createBuffer('timestamps', vertexCount * 8);
    const cellIdsBuffer = resources.createBuffer('cell-ids', cellIds);
    const speedsBuffer = resources.createBuffer('speeds', speeds);
    const hourBuffer = resources.createBuffer('hour', vertexCount * 4);
    const matrixBuffer = resources.createBuffer('hour-weekday-matrix', 168 * 4);
    const hourHistogramBuffer = resources.createBuffer('hour-histogram', 24 * 4);
    const selectionBuffer = resources.createBuffer('selection', vertexCount * 4);
    const selectedCountBuffer = resources.createBuffer('selected-count', 4);
    const countsBuffer = resources.createBuffer('cube-counts', binCount * 4);
    const minimumBuffer = resources.createBuffer('cube-min', binCount * 4);
    const maximumBuffer = resources.createBuffer('cube-max', binCount * 4);
    const firstBuffer = resources.createBuffer('cube-first', binCount * 4);
    const lastBuffer = resources.createBuffer('cube-last', binCount * 4);
    const occupiedIdsBuffer = resources.createBuffer('occupied-ids', binCount * 4);
    const occupiedCountBuffer = resources.createBuffer('occupied-count', 4);
    const occupiedOverflowBuffer = resources.createBuffer('occupied-overflow', 4);
    const giScoresBuffer = resources.createBuffer('gi-z-scores', binCount * 4);
    const trendZBuffer = resources.createBuffer('trend-z', cellCount * 4);
    const trendPBuffer = resources.createBuffer('trend-p', cellCount * 4);
    const trendSBuffer = resources.createBuffer('trend-s', cellCount * 4);
    const categoryBuffer = resources.createBuffer('category', cellCount * 4);
    const hotSliceBuffer = resources.createBuffer('hot-slices', cellCount * 4);
    const coldSliceBuffer = resources.createBuffer('cold-slices', cellCount * 4);
    const globalStatisticsBuffer = resources.createBuffer(
      'global-statistics',
      GPU_EMERGING_HOT_SPOT_STATISTICS_LENGTH * 4
    );
    const categoryCountsBuffer = resources.createBuffer('category-counts', CATEGORY_COUNT * 4);
    const differenceBuffer = resources.createBuffer('difference', cellCount * 4);
    const logRatioBuffer = resources.createBuffer('log-ratio', cellCount * 4);
    const percentChangeBuffer = resources.createBuffer('percent-change', cellCount * 4);
    const senSlopeBuffer = resources.createBuffer('sen-slope', cellCount * 4);
    const mannKendallSBuffer = resources.createBuffer('mann-kendall-s', cellCount * 4);
    const mannKendallZBuffer = resources.createBuffer('mann-kendall-z', cellCount * 4);
    const mannKendallPBuffer = resources.createBuffer('mann-kendall-p', cellCount * 4);
    const significanceBuffer = resources.createBuffer('significance', cellCount * 4);
    const significanceCountsBuffer = resources.createBuffer(
      'significance-counts',
      SIGNIFICANCE_CLASS_COUNT * 4
    );
    // `valueIndices` of the activity view: cell -> bin of the selected slice.
    const sliceIndexBuffer = resources.createBuffer('slice-indices', cellCount * 4);

    const calendarParameters = resources.createParameterBuffer(
      'calendar-parameters',
      'sint32',
      GPU_CALENDAR_BUCKETS_PARAMETER_LENGTH
    );
    const sliceParameters = resources.createParameterBuffer('slice-parameters', 'uint32', 4);
    const hotSpotParameters = resources.createParameterBuffer(
      'hot-spot-parameters',
      'float32',
      GPU_EMERGING_HOT_SPOT_PARAMETER_LENGTH
    );
    const changeParameters = resources.createParameterBuffer(
      'change-parameters',
      'float32',
      GPU_CHANGE_DETECTION_PARAMETER_LENGTH
    );

    // ---- One graph. ----
    const graph = new GPUCommandGraph<void>(device, {id: 'space-time'});
    const timestampsView = importGraphBuffer(
      graph,
      'timestamps',
      timestampsBuffer,
      'uint32x2',
      vertexCount
    );
    const hourView = importGraphBuffer(graph, 'hour', hourBuffer, 'uint32', vertexCount);
    const matrixView = importGraphBuffer(
      graph,
      'matrix',
      matrixBuffer,
      'uint32',
      GPU_CALENDAR_BUCKETS_MATRIX_LENGTH
    );
    graph.add(
      new GPUCalendarBuckets({
        id: 'calendar',
        timestamps: timestampsView,
        parameters: calendarParameters.importToGraph(graph),
        output: {hour: hourView, hourWeekdayCounts: matrixView}
      })
    );
    const selectionView = importGraphBuffer(
      graph,
      'selection',
      selectionBuffer,
      'uint32',
      vertexCount
    );
    const hourFilter = new GPUCrossfilter(graph, {
      id: 'hour-brush',
      dimensions: [
        {id: 'hour', kind: 'range', input: hourView, rejectNonFinite: true, exclusiveMaximum: true}
      ],
      views: [
        {
          id: 'hour-histogram',
          kind: 'histogram',
          dimension: 'hour',
          input: hourView,
          domain: [0, 24],
          output: importGraphBuffer(graph, 'hour-histogram', hourHistogramBuffer, 'uint32', 24)
        },
        {id: 'selection', kind: 'mask', output: selectionView},
        {
          id: 'selected-count',
          kind: 'count',
          output: importGraphBuffer(graph, 'selected-count', selectedCountBuffer, 'uint32', 1)
        }
      ]
    });
    hourFilter.addToGraph(graph);
    const countsView = importGraphBuffer(graph, 'counts', countsBuffer, 'uint32', binCount);
    const maximumView = importGraphBuffer(graph, 'maximum', maximumBuffer, 'float32', binCount);
    graph.add(
      new GPUTemporalReduction({
        id: 'cube',
        cellIds: importGraphBuffer(graph, 'cell-ids', cellIdsBuffer, 'uint32', vertexCount),
        timestamps: timestampsView,
        values: importGraphBuffer(graph, 'speeds', speedsBuffer, 'float32', vertexCount),
        mask: selectionView,
        parameters: sliceParameters.importToGraph(graph),
        cellCount,
        bucketCount: SLICE_COUNT,
        output: {
          counts: countsView,
          min: importGraphBuffer(graph, 'minimum', minimumBuffer, 'float32', binCount),
          max: maximumView,
          first: importGraphBuffer(graph, 'first', firstBuffer, 'float32', binCount),
          last: importGraphBuffer(graph, 'last', lastBuffer, 'float32', binCount),
          occupiedSlots: {
            ids: importGraphBuffer(graph, 'occupied-ids', occupiedIdsBuffer, 'uint32', binCount),
            count: importGraphBuffer(graph, 'occupied-count', occupiedCountBuffer, 'uint32', 1),
            overflow: importGraphBuffer(
              graph,
              'occupied-overflow',
              occupiedOverflowBuffer,
              'uint32',
              1
            )
          }
        }
      })
    );
    const categoryView = importGraphBuffer(graph, 'category', categoryBuffer, 'uint32', cellCount);
    graph.add(
      new GPUEmergingHotSpots({
        id: 'emerging',
        values: countsView,
        gridWidth,
        gridHeight,
        sliceCount: SLICE_COUNT,
        maximumRadius: MAXIMUM_RADIUS,
        parameters: hotSpotParameters.importToGraph(graph),
        giZScores: importGraphBuffer(graph, 'gi-z-scores', giScoresBuffer, 'float32', binCount),
        trendZ: importGraphBuffer(graph, 'trend-z', trendZBuffer, 'float32', cellCount),
        trendP: importGraphBuffer(graph, 'trend-p', trendPBuffer, 'float32', cellCount),
        trendS: importGraphBuffer(graph, 'trend-s', trendSBuffer, 'sint32', cellCount),
        category: categoryView,
        hotSliceCount: importGraphBuffer(graph, 'hot-slices', hotSliceBuffer, 'uint32', cellCount),
        coldSliceCount: importGraphBuffer(
          graph,
          'cold-slices',
          coldSliceBuffer,
          'uint32',
          cellCount
        ),
        globalStatistics: importGraphBuffer(
          graph,
          'global-statistics',
          globalStatisticsBuffer,
          'float32',
          GPU_EMERGING_HOT_SPOT_STATISTICS_LENGTH
        )
      })
    );
    graph.add(
      new GPUHistogram({
        id: 'category-counts',
        input: categoryView,
        output: importGraphBuffer(
          graph,
          'category-counts',
          categoryCountsBuffer,
          'uint32',
          CATEGORY_COUNT
        ),
        edges: Array.from({length: CATEGORY_COUNT + 1}, (_, index) => index)
      })
    );
    const significanceView = importGraphBuffer(
      graph,
      'significance',
      significanceBuffer,
      'uint32',
      cellCount
    );
    graph.add(
      new GPUChangeDetection({
        id: 'change',
        slices: maximumView,
        parameters: changeParameters.importToGraph(graph),
        cellCount,
        sliceCount: SLICE_COUNT,
        significanceSource: 'mann-kendall',
        output: {
          difference: importGraphBuffer(
            graph,
            'difference',
            differenceBuffer,
            'float32',
            cellCount
          ),
          logRatio: importGraphBuffer(graph, 'log-ratio', logRatioBuffer, 'float32', cellCount),
          percentChange: importGraphBuffer(
            graph,
            'percent-change',
            percentChangeBuffer,
            'float32',
            cellCount
          ),
          senSlope: importGraphBuffer(graph, 'sen-slope', senSlopeBuffer, 'float32', cellCount),
          mannKendallS: importGraphBuffer(
            graph,
            'mann-kendall-s',
            mannKendallSBuffer,
            'sint32',
            cellCount
          ),
          mannKendallZ: importGraphBuffer(
            graph,
            'mann-kendall-z',
            mannKendallZBuffer,
            'float32',
            cellCount
          ),
          mannKendallP: importGraphBuffer(
            graph,
            'mann-kendall-p',
            mannKendallPBuffer,
            'float32',
            cellCount
          ),
          significance: significanceView
        }
      })
    );
    graph.add(
      new GPUHistogram({
        id: 'significance-counts',
        input: significanceView,
        output: importGraphBuffer(
          graph,
          'significance-counts',
          significanceCountsBuffer,
          'uint32',
          SIGNIFICANCE_CLASS_COUNT
        ),
        edges: Array.from({length: SIGNIFICANCE_CLASS_COUNT + 1}, (_, index) => index)
      })
    );
    const compiled: CompiledGPUCommandGraph<void> = resources.track(graph.compile());

    // Summary readback layout in 32-bit words.
    const MATRIX_WORD = 0;
    const CATEGORY_WORD = MATRIX_WORD + GPU_CALENDAR_BUCKETS_MATRIX_LENGTH;
    const SIGNIFICANCE_WORD = CATEGORY_WORD + CATEGORY_COUNT;
    const CUBE_WORD = SIGNIFICANCE_WORD + SIGNIFICANCE_CLASS_COUNT; // occupied, overflow, selected
    const STATISTICS_WORD = CUBE_WORD + 3;
    const SUMMARY_WORDS = STATISTICS_WORD + GPU_EMERGING_HOT_SPOT_STATISTICS_LENGTH;
    const readbackRing = resources.track(
      new GPUReadbackRing(device, {id: 'space-time-summary', byteLength: SUMMARY_WORDS * 4})
    );

    // ---- State. ----
    let utcOffsetMinutes = 0;
    let hourFrom = 0;
    let hourTo = 23;
    let radius = 2;
    let temporalWindow = 2;
    let confidenceLevel: 0.9 | 0.95 | 0.99 = 0.9;
    let beforeSlice = 8;
    let afterSlice = 16;
    let view: MapView = 'hot-spots';
    let dirty = true;
    let readbackPending = false;
    let needsReadback = true;
    let destroyed = false;

    const getSliceWidthMs = () => Math.floor(getSpanMs() / SLICE_COUNT) + 1;
    const formatSlice = (slice: number) => {
      const width = getSliceWidthMs();
      const startMs = getClockStartMs() + slice * width + utcOffsetMinutes * 60_000;
      const date = new Date(startMs);
      const weekday = WEEKDAYS[(date.getUTCDay() + 6) % 7];
      const clockText = `${String(date.getUTCHours()).padStart(2, '0')}:${String(date.getUTCMinutes()).padStart(2, '0')}`;
      return width < 600_000
        ? `${clockText}:${String(date.getUTCSeconds()).padStart(2, '0')} ${weekday}`
        : `${weekday} ${clockText}`;
    };

    const writeParameters = () => {
      calendarParameters.write(getGPUCalendarBucketsParameterValues(utcOffsetMinutes, 0));
      hourFilter.setRange('hour', [hourFrom, hourTo + 1]);
      sliceParameters.write(
        getGPUTemporalReductionWordParameterValues(getClockStartMs(), getSliceWidthMs())
      );
      hotSpotParameters.write(
        getGPUEmergingHotSpotParameterValues({radius, temporalWindow, confidenceLevel})
      );
      changeParameters.write(getGPUChangeDetectionParameterValues({beforeSlice, afterSlice}));
      const indices = new Uint32Array(cellCount);
      for (let cell = 0; cell < cellCount; cell++) indices[cell] = cell * SLICE_COUNT + afterSlice;
      sliceIndexBuffer.write(indices);
      dirty = true;
      needsReadback = true;
    };

    // ---- Controls. ----
    context.controls.addSelect<MapView>({
      label: 'Map layer (buffer choice, no recompile)',
      options: [
        {value: 'hot-spots', label: 'Emerging hot spots (17 categories, count cube)'},
        {value: 'activity', label: 'Activity: vertices in the after slice'},
        {value: 'difference', label: 'Change: difference of peak speed'},
        {value: 'log-ratio', label: 'Change: log ratio of peak speed'},
        {value: 'percent-change', label: 'Change: percent change of peak speed'},
        {value: 'mann-kendall', label: 'Trend: Mann-Kendall Z of peak speed'},
        {value: 'sen-slope', label: "Trend: Sen's slope of peak speed"}
      ],
      value: view,
      onChange: value => {
        view = value;
        updateScaleLegend();
        context.updateLayers();
      }
    });
    context.controls.addSelect<ClockKind>({
      label: 'Clock (rewrites the timestamp buffer)',
      options: [
        {value: 'recorded', label: 'Recorded: 41 minutes, Monday 17:00 UTC (nominal)'},
        {value: 'stretched', label: `Stretched x${STRETCH_FACTOR}: about 7 days (demo only)`}
      ],
      value: clock,
      onChange: value => {
        clock = value;
        writeTimestamps();
        writeParameters();
        updateSliceLabels();
      }
    });
    context.controls.addSelect({
      label: 'UTC offset (calendar parameter)',
      options: [
        {value: '0', label: 'UTC'},
        {value: '-300', label: 'UTC-5 (New York, standard time)'},
        {value: '-240', label: 'UTC-4 (New York, daylight time)'},
        {value: '-480', label: 'UTC-8'},
        {value: '60', label: 'UTC+1'},
        {value: '330', label: 'UTC+5:30'},
        {value: '540', label: 'UTC+9'}
      ],
      value: '0',
      onChange: value => {
        utcOffsetMinutes = Number(value);
        writeParameters();
        updateSliceLabels();
      }
    });
    const hourFromControl = context.controls.addSlider({
      label: 'Local hour filter: from',
      min: 0,
      max: 23,
      step: 1,
      value: hourFrom,
      format: value => `${String(value).padStart(2, '0')}:00`,
      onChange: value => {
        hourFrom = value;
        if (hourTo < hourFrom) {
          hourTo = hourFrom;
          hourToControl.setValue(hourTo);
        }
        writeParameters();
      }
    });
    void hourFromControl;
    const hourToControl = context.controls.addSlider({
      label: 'Local hour filter: to (inclusive)',
      min: 0,
      max: 23,
      step: 1,
      value: hourTo,
      format: value => `${String(value).padStart(2, '0')}:59`,
      onChange: value => {
        hourTo = value;
        if (hourFrom > hourTo) {
          hourFrom = hourTo;
          hourFromControl.setValue(hourFrom);
        }
        writeParameters();
      }
    });
    context.controls.addSlider({
      label: 'Gi* neighbor distance (cells; per-frame parameter)',
      min: 0,
      max: MAXIMUM_RADIUS,
      step: 1,
      value: radius,
      format: value => `${value} cells (${Math.round(value * cellSize)} m)`,
      onChange: value => {
        radius = value;
        writeParameters();
      }
    });
    context.controls.addSlider({
      label: 'Gi* time window (previous slices; per-frame parameter)',
      min: 0,
      max: 6,
      step: 1,
      value: temporalWindow,
      format: value => `${value} slices`,
      onChange: value => {
        temporalWindow = value;
        writeParameters();
      }
    });
    context.controls.addSelect({
      label: 'Hot spot confidence',
      options: [
        {value: '0.9', label: '90%'},
        {value: '0.95', label: '95%'},
        {value: '0.99', label: '99%'}
      ],
      value: '0.9',
      onChange: value => {
        confidenceLevel = Number(value) as typeof confidenceLevel;
        writeParameters();
      }
    });
    const beforeControl = context.controls.addSlider({
      label: 'Change detection: before slice',
      min: 0,
      max: SLICE_COUNT - 1,
      step: 1,
      value: beforeSlice,
      format: value => `${value}: ${formatSlice(value)}`,
      onChange: value => {
        beforeSlice = value;
        writeParameters();
      }
    });
    const afterControl = context.controls.addSlider({
      label: 'Change detection: after slice',
      min: 0,
      max: SLICE_COUNT - 1,
      step: 1,
      value: afterSlice,
      format: value => `${value}: ${formatSlice(value)}`,
      onChange: value => {
        afterSlice = value;
        writeParameters();
      }
    });
    const updateSliceLabels = () => {
      beforeControl.setValue(beforeSlice);
      afterControl.setValue(afterSlice);
    };
    context.controls.addNote(
      'Clock: trips-v7 has relative seconds only (about 41 min). The calendar is a nominal ' +
        'anchor; the stretched clock only exercises GPUCalendarBuckets. Slices are 24 equal ' +
        'absolute time bins of the recording, not hours of the day.'
    );
    context.controls.addNote(
      `Cube: ${gridWidth} x ${gridHeight} cells of ${Math.round(cellSize)} m x ${SLICE_COUNT} ` +
        'slices. Counts feed the hot spot map; the change maps read the peak speed per cell and ' +
        'slice because no contributor converts uint32 counts to the float32 stack change detection reads.'
    );
    const selectedReadout = context.controls.addReadout('Vertices selected / live');
    const cubeReadout = context.controls.addReadout('Occupied bins / cube bins');
    const statisticsReadout = context.controls.addReadout('Bin count mean / std. deviation');
    const hotReadout = context.controls.addReadout('Hot / cold cells');
    const trendReadout = context.controls.addReadout('Significant speed trend up / down');
    context.controls.addReadout('Cube size', `${formatCount(binCount)} bins`);
    context.controls.addReadout('Vertices', formatCount(vertexCount));
    context.controls.addReadout('Data', trips.attribution);

    // ---- Panel extras (a heat matrix chart and live legends), appended to the mode area. ----
    const root = document.createElement('div');
    root.style.cssText = 'margin-top:10px;color:#a9b8d0;font-size:11px';
    const matrixTitle = document.createElement('div');
    matrixTitle.textContent = 'Vertices by local hour and weekday (GPUCalendarBuckets matrix)';
    const matrixCanvas = document.createElement('canvas');
    const matrixCellWidth = 11;
    const matrixCellHeight = 13;
    const matrixLeft = 26;
    const matrixTop = 12;
    matrixCanvas.width = matrixLeft + 24 * matrixCellWidth + 2;
    matrixCanvas.height = matrixTop + 7 * matrixCellHeight + 2;
    matrixCanvas.style.cssText = 'margin-top:3px;display:block;max-width:100%';
    matrixCanvas.setAttribute('aria-label', 'Hour by weekday vertex count heat matrix');
    const scaleLegend = document.createElement('div');
    scaleLegend.style.marginTop = '8px';
    const categoryLegend = document.createElement('div');
    categoryLegend.style.cssText = 'margin-top:8px';
    root.append(matrixTitle, matrixCanvas, scaleLegend, categoryLegend);
    const modeArea = document.querySelector('[data-spatial-analysis-panel] [data-mode]');
    modeArea?.appendChild(root);

    const toCss = (color: readonly number[]) => `rgb(${color[0]},${color[1]},${color[2]})`;
    const categoryCountElements: HTMLSpanElement[] = [];
    {
      const heading = document.createElement('div');
      heading.textContent = 'Emerging hot spot categories (cells)';
      const list = document.createElement('div');
      list.style.cssText = 'display:grid;grid-template-columns:1fr 1fr;gap:2px 10px;margin-top:3px';
      for (let category = 0; category < CATEGORY_COUNT; category++) {
        const item = document.createElement('div');
        item.style.cssText = 'display:flex;align-items:center;gap:4px;white-space:nowrap';
        const swatch = document.createElement('span');
        swatch.style.cssText = `display:inline-block;width:9px;height:9px;border-radius:2px;background:${toCss(CATEGORY_COLORS[category])}`;
        const name = document.createElement('span');
        name.textContent = CATEGORY_NAMES[category];
        const count = document.createElement('span');
        count.style.cssText = 'margin-left:auto;color:#edf4ff;font-family:ui-monospace,monospace';
        count.textContent = '-';
        categoryCountElements.push(count);
        item.append(swatch, name, count);
        list.appendChild(item);
      }
      categoryLegend.append(heading, list);
    }
    const updateScaleLegend = () => {
      scaleLegend.innerHTML = '';
      if (view === 'hot-spots') {
        scaleLegend.textContent = 'Colors: see the category list below.';
        return;
      }
      const spec = SCALAR_VIEWS[view];
      const colors = spec.diverging ? DIVERGING_COLORS : VIRIDIS_COLORS;
      const title = document.createElement('div');
      title.textContent = spec.unit;
      const bar = document.createElement('div');
      bar.style.cssText = `height:8px;border-radius:4px;margin-top:3px;background:linear-gradient(90deg,${colors.map(toCss).join(',')})`;
      const labels = document.createElement('div');
      labels.style.cssText = 'display:flex;justify-content:space-between';
      labels.innerHTML = `<span>${spec.range[0]}</span><span>${spec.range[1]}</span>`;
      scaleLegend.append(title, bar, labels);
    };
    updateScaleLegend();

    const drawMatrix = (counts: Uint32Array) => {
      const canvasContext = matrixCanvas.getContext('2d');
      if (!canvasContext) return;
      canvasContext.clearRect(0, 0, matrixCanvas.width, matrixCanvas.height);
      let largest = 1;
      for (const count of counts) largest = Math.max(largest, count);
      canvasContext.font = '9px system-ui, sans-serif';
      canvasContext.fillStyle = '#7f90ad';
      for (let hour = 0; hour < 24; hour += 3) {
        canvasContext.fillText(String(hour), matrixLeft + hour * matrixCellWidth, 9);
      }
      for (let weekday = 0; weekday < 7; weekday++) {
        canvasContext.fillStyle = '#7f90ad';
        canvasContext.fillText(WEEKDAYS[weekday], 0, matrixTop + weekday * matrixCellHeight + 10);
        for (let hour = 0; hour < 24; hour++) {
          const count = counts[weekday * 24 + hour];
          const inBrush = hour >= hourFrom && hour <= hourTo;
          const t = Math.sqrt(count / largest);
          const alpha = count === 0 ? 0.12 : 0.25 + 0.75 * t;
          canvasContext.fillStyle = inBrush
            ? `rgba(128, 234, 219, ${alpha})`
            : `rgba(150, 160, 175, ${alpha * 0.45})`;
          canvasContext.fillRect(
            matrixLeft + hour * matrixCellWidth,
            matrixTop + weekday * matrixCellHeight,
            matrixCellWidth - 1,
            matrixCellHeight - 1
          );
        }
      }
    };

    writeTimestamps();
    writeParameters();

    const readSummary = async (
      commandEncoder: Parameters<SpatialAnalysisModeInstance['encode']>[0]
    ) => {
      const ticket = readbackRing.tryAcquire();
      if (!ticket) return;
      const copies: [Buffer, number, number][] = [
        [matrixBuffer, MATRIX_WORD, GPU_CALENDAR_BUCKETS_MATRIX_LENGTH],
        [categoryCountsBuffer, CATEGORY_WORD, CATEGORY_COUNT],
        [significanceCountsBuffer, SIGNIFICANCE_WORD, SIGNIFICANCE_CLASS_COUNT],
        [occupiedCountBuffer, CUBE_WORD, 1],
        [occupiedOverflowBuffer, CUBE_WORD + 1, 1],
        [selectedCountBuffer, CUBE_WORD + 2, 1],
        [globalStatisticsBuffer, STATISTICS_WORD, GPU_EMERGING_HOT_SPOT_STATISTICS_LENGTH]
      ];
      for (const [sourceBuffer, word, length] of copies) {
        commandEncoder.copyBufferToBuffer({
          sourceBuffer,
          destinationBuffer: ticket.buffer,
          destinationOffset: word * 4,
          size: length * 4
        });
      }
      ticket.markEncoded({byteOffset: 0, byteLength: SUMMARY_WORDS * 4});
      readbackPending = true;
      needsReadback = false;
      try {
        const bytes = await ticket.read();
        if (destroyed) return;
        const words = new Uint32Array(bytes.buffer, bytes.byteOffset, SUMMARY_WORDS);
        const floats = new Float32Array(bytes.buffer, bytes.byteOffset, SUMMARY_WORDS);
        drawMatrix(words.subarray(MATRIX_WORD, MATRIX_WORD + GPU_CALENDAR_BUCKETS_MATRIX_LENGTH));
        let hotCells = 0;
        let coldCells = 0;
        for (let category = 0; category < CATEGORY_COUNT; category++) {
          const count = words[CATEGORY_WORD + category];
          categoryCountElements[category].textContent = formatCount(count);
          if (category >= 1 && category <= 8) hotCells += count;
          if (category >= 9) coldCells += count;
        }
        hotReadout.setValue(`${formatCount(hotCells)} / ${formatCount(coldCells)}`);
        trendReadout.setValue(
          `${formatCount(words[SIGNIFICANCE_WORD + 1])} / ${formatCount(words[SIGNIFICANCE_WORD + 2])} cells`
        );
        selectedReadout.setValue(
          `${formatCount(words[CUBE_WORD + 2])} / ${formatCount(vertexCount)}`
        );
        cubeReadout.setValue(
          `${formatCount(words[CUBE_WORD])}${words[CUBE_WORD + 1] ? ' (overflow)' : ''} / ${formatCount(binCount)}`
        );
        statisticsReadout.setValue(
          `${floats[STATISTICS_WORD + 1].toFixed(3)} / ${floats[STATISTICS_WORD + 3].toFixed(3)}`
        );
      } catch {
        // The ring or device was destroyed while the read was in flight.
        needsReadback = true;
      } finally {
        readbackPending = false;
      }
    };

    const instance: SpatialAnalysisModeInstance = {
      getCompiledGraphs: () => [compiled],
      encode(commandEncoder, frame) {
        // The vertex data is static: results only change with a control, so the graph is encoded
        // only then (plus two warm-up frames).
        if (dirty || frame.frameIndex < 2) {
          compiled.encode(commandEncoder, {parameters: undefined});
          dirty = false;
          needsReadback = true;
        }
        if (
          needsReadback &&
          !readbackPending &&
          frame.frameIndex >= 1 &&
          frame.frameIndex % READBACK_INTERVAL_FRAMES === 0
        ) {
          void readSummary(commandEncoder);
        }
      },
      getLayers() {
        const coordinateOrigin: [number, number, number] = [trips.origin[0], trips.origin[1], 0];
        const layers: Layer[] = [
          new SpatialAnalysisPointLayer({
            id: 'space-time-vertices',
            coordinateOrigin,
            positions: positionsBuffer,
            instanceCount: vertexCount,
            values: selectionBuffer,
            valueFormat: 'uint32',
            colormap: 'mask',
            color: [128, 234, 219, 60],
            noDataColor: [0, 0, 0, 0],
            radiusPixels: 1.4
          })
        ];
        const common = {
          coordinateOrigin,
          gridSize: [gridWidth, gridHeight] as const,
          bounds,
          opacity: 1
        };
        if (view === 'hot-spots') {
          layers.push(
            new SpaceTimeRasterLayer({
              ...common,
              id: 'space-time-categories',
              values: categoryBuffer,
              valueFormat: 'uint32',
              colormap: 'category',
              noDataColor: [0, 0, 0, 0]
            })
          );
        } else if (view === 'activity') {
          layers.push(
            new SpaceTimeRasterLayer({
              ...common,
              id: 'space-time-activity',
              values: countsBuffer,
              valueFormat: 'uint32',
              valueIndices: sliceIndexBuffer,
              colormap: 'viridis',
              valueRange: SCALAR_VIEWS.activity.range,
              sqrtScale: true,
              discardAtOrBelow: 0,
              color: [255, 255, 255, 225]
            })
          );
        } else {
          const valuesByView: Record<string, Buffer> = {
            difference: differenceBuffer,
            'log-ratio': logRatioBuffer,
            'percent-change': percentChangeBuffer,
            'mann-kendall': mannKendallZBuffer,
            'sen-slope': senSlopeBuffer
          };
          layers.push(
            new SpaceTimeRasterLayer({
              ...common,
              id: `space-time-${view}`,
              values: valuesByView[view],
              valueFormat: 'float32',
              colormap: 'grayscale',
              valueRange: SCALAR_VIEWS[view].range,
              noDataColor: [0, 0, 0, 0],
              color: [255, 255, 255, 225]
            })
          );
        }
        return layers;
      },
      destroy() {
        destroyed = true;
        root.remove();
        hourFilter.destroy();
        resources.destroy();
      }
    };
    return instance;
  }
};
