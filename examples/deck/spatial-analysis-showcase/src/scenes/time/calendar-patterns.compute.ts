// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {
  getGPUCalendarBucketsParameterValues,
  getGPUTimeWindowWordParameterValues,
  GPU_CALENDAR_BUCKETS_MATRIX_LENGTH,
  GPU_CALENDAR_BUCKETS_PARAMETER_LENGTH,
  GPU_TIME_WORD_WINDOW_PARAMETER_LENGTH,
  GPUCalendarBuckets,
  GPUGroupStatistics,
  GPUTimeWindowFilter
} from '@luma.gl/experimental/gpu-dataframe';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {addKernelPass} from '../../engine/mode-kernels';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer,
  type SpatialAnalysisColor
} from '../../engine/layers';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {RampName} from '../../engine/ramps';
import type {SceneContext, SceneInstance} from '../scene';
import {fillCategoryMask, readNatureEvents, type B13NatureEvents} from './b13-nature-events';
import {
  getGeoJsonOutlineSegments,
  getPolygonSource,
  lookupFeature,
  rasterizePolygons,
  type B13PolygonRaster
} from './b13-polygon-raster';
import {createViewImporter} from './b13-views';

export type CalendarStatistic =
  | 'count'
  | 'density'
  | 'mean-hour'
  | 'median-hour'
  | 'hour-spread'
  | 'percentile-hour'
  | 'mode-hour'
  | 'active-days'
  | 'research-grade-share'
  | 'introduced-share';

/** Option state of the calendar-patterns scene. */
export type CalendarPatternsOptions = {
  groupType: string;
  dateRange: readonly [number, number];
  playWindow: boolean;
  windowDays: number;
  utcOffset: number;
  firstDay: string;
  daylightSaving: boolean;
  hours: readonly [number, number];
  invertHours: boolean;
  weekdays: string;
  statistic: CalendarStatistic;
  percentile: number;
  variance: 'sample' | 'population';
  minimumEvents: number;
  showAreas: boolean;
  showPoints: boolean;
  showMatrix: boolean;
  ramp: RampName;
  opacity: number;
};

const AREA_COUNT = 77;
const _SECONDS_PER_DAY = 86400;
const YEAR_START_MS = Date.UTC(2023, 0, 1);
const PLAY_DAYS_PER_SECOND = 45;
const WEEKDAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'] as const;
const STATISTIC_CODES: Record<CalendarStatistic, number> = {
  count: 0,
  density: 1,
  'mean-hour': 2,
  'median-hour': 3,
  'hour-spread': 4,
  'percentile-hour': 5,
  'mode-hour': 6,
  'active-days': 7,
  'research-grade-share': 8,
  'introduced-share': 9
};

/** Weekday presets: Monday = 0 to Sunday = 6. */
const WEEKDAY_SETS: Record<string, readonly number[]> = {
  all: [0, 1, 2, 3, 4, 5, 6],
  weekdays: [0, 1, 2, 3, 4],
  weekend: [5, 6],
  'friday-saturday': [4, 5],
  mon: [0],
  tue: [1],
  wed: [2],
  thu: [3],
  fri: [4],
  sat: [5],
  sun: [6]
};

/** Unit and display name of each statistic, used by the legend and readouts. */
export const CALENDAR_STATISTIC_LABELS: Record<CalendarStatistic, {title: string; unit: string}> = {
  count: {title: 'Observations in the brush', unit: 'observations'},
  density: {title: 'Observations per square kilometer', unit: 'per km2'},
  'mean-hour': {title: 'Mean hour of day', unit: 'hour'},
  'median-hour': {title: 'Median hour of day', unit: 'hour'},
  'hour-spread': {title: 'Spread of the hour (standard deviation)', unit: 'hours'},
  'percentile-hour': {title: 'Percentile of the hour of day', unit: 'hour'},
  'mode-hour': {title: 'Most common hour', unit: 'hour'},
  'active-days': {title: 'Days with at least one observation', unit: 'days'},
  'research-grade-share': {title: 'Share research grade', unit: 'share'},
  'introduced-share': {title: 'Share introduced species', unit: 'share'}
};

const formatDate = (day: number) => {
  const date = new Date(YEAR_START_MS + day * 86400000);
  return `${['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][date.getUTCMonth()]} ${date.getUTCDate()}`;
};

/** Chicago's UTC offset in minutes for an instant, 2023 US daylight saving rules. */
function getChicagoOffsetMinutes(utcMilliseconds: number): number {
  const start = Date.UTC(2023, 2, 12, 8);
  const end = Date.UTC(2023, 10, 5, 7);
  return utcMilliseconds >= start && utcMilliseconds < end ? -300 : -360;
}

type Variant = {
  key: string;
  compiled: CompiledGPUCommandGraph<void>;
  reader: SummaryReader;
};

/**
 * Calendar patterns of Chicago nature observations. In one compiled graph: `GPUTimeWindowFilter` (exact Int64
 * time) selects the events inside a sliding date window, `GPUCalendarBuckets` decodes them into
 * hour, weekday and day of the year plus an hour by weekday count matrix, a small kernel applies
 * the hour and weekday brush, and `GPUGroupStatistics` computes per-community-area statistics
 * of the brushed events (dense keys, one row per area). Two small kernels pick the chosen
 * statistic into one metric buffer that a raster layer colors. The window, calendar parameters,
 * brush, statistic and percentile are parameter writes; daylight saving (a per-row offset
 * column) and the variance kind are compile-time, so four graph variants are compiled on demand.
 */
export async function createCalendarPatterns(
  ctx: SceneContext<CalendarPatternsOptions>
): Promise<SceneInstance<CalendarPatternsOptions>> {
  const {device} = ctx;
  const natureDataset = ctx.datasets.get('chicago-nature');
  const areasDataset = ctx.datasets.get('chicago-community-areas');
  const events: B13NatureEvents = readNatureEvents(natureDataset);
  const origin = events.origin;
  const eventCount = events.count;
  const areaProjection = areasDataset.getProjection(origin);
  const areaSource = getPolygonSource(areasDataset);
  const raster: B13PolygonRaster = rasterizePolygons(
    areaSource,
    areaProjection,
    areasDataset.manifest.bbox,
    1400
  );
  const areaNames = (areasDataset.geojson?.features ?? []).map(feature =>
    String((feature.properties as {name?: string})?.name ?? '')
  );
  const areaKm2 = areasDataset.column<Float32Array>('areaKm2');
  const outlineSegments = areasDataset.geojson
    ? getGeoJsonOutlineSegments(
        areasDataset.geojson as unknown as Parameters<typeof getGeoJsonOutlineSegments>[0],
        areaProjection
      )
    : new Float32Array(4);

  // Static per-event columns.
  const communityArea = natureDataset.column<Uint8Array>('communityArea');
  const researchGrade = natureDataset.column<Uint8Array>('researchGrade');
  const introduced = natureDataset.column<Uint8Array>('introduced');
  const areaKeys = new Uint32Array(eventCount);
  const researchGradeFlags = new Float32Array(eventCount);
  const introducedFlags = new Float32Array(eventCount);
  const chicagoOffsets = new Int32Array(eventCount);
  for (let index = 0; index < eventCount; index++) {
    areaKeys[index] =
      communityArea[index] > 0 && communityArea[index] <= AREA_COUNT
        ? communityArea[index] - 1
        : 0xffffffff;
    researchGradeFlags[index] = researchGrade[index] ? 1 : 0;
    introducedFlags[index] = introduced[index] ? 1 : 0;
    chicagoOffsets[index] = getChicagoOffsetMinutes(YEAR_START_MS + events.seconds[index] * 1000);
  }

  const resources = new SpatialAnalysisResources(device, 'calendar');
  const positionsBuffer = resources.createBuffer('positions', events.positions);
  const timeWordsBuffer = resources.createBuffer('time-words', events.timeWords as Uint32Array);
  const offsetsBuffer = resources.createBuffer('utc-offsets', chicagoOffsets);
  const categoryMaskBuffer = resources.createBuffer('category-mask', eventCount * 4);
  const areaKeyBuffer = resources.createBuffer('area-keys', areaKeys);
  const researchGradeBuffer = resources.createBuffer('researchGrade', researchGradeFlags);
  const introducedBuffer = resources.createBuffer('introduced', introducedFlags);
  const areaKm2Buffer = resources.createBuffer('area-km2', areaKm2);
  const cellFeatureBuffer = resources.createBuffer('cell-feature', raster.cellFeature);
  const outlineBuffer = resources.createBuffer('outlines', outlineSegments);
  // Shared outputs of every variant (only the current variant runs).
  const idsBuffer = resources.createBuffer('ids', eventCount * 4);
  const countBuffer = resources.createBuffer('window-count', 4);
  const overflowBuffer = resources.createBuffer('overflow', 4);
  const windowMaskBuffer = resources.createBuffer('window-mask', eventCount * 4);
  const hourBuffer = resources.createBuffer('hour', eventCount * 4);
  const weekdayBuffer = resources.createBuffer('weekday', eventCount * 4);
  const dayOfYearBuffer = resources.createBuffer('day-of-year', eventCount * 4);
  const matrixBuffer = resources.createBuffer('matrix', GPU_CALENDAR_BUCKETS_MATRIX_LENGTH * 4);
  const hourFloatBuffer = resources.createBuffer('hour-float', eventCount * 4);
  const dayFloatBuffer = resources.createBuffer('day-float', eventCount * 4);
  const brushBuffer = resources.createBuffer('brush', eventCount * 4);
  const groupKeysBuffer = resources.createBuffer('group-keys', AREA_COUNT * 4);
  const groupCountsBuffer = resources.createBuffer('group-counts', AREA_COUNT * 4);
  const groupCountBuffer = resources.createBuffer('group-count', 4);
  const groupOverflowBuffer = resources.createBuffer('group-overflow', 4);
  const hourCountsBuffer = resources.createBuffer('hour-counts', AREA_COUNT * 4);
  const hourMeansBuffer = resources.createBuffer('hour-means', AREA_COUNT * 4);
  const hourMediansBuffer = resources.createBuffer('hour-medians', AREA_COUNT * 4);
  const hourDeviationsBuffer = resources.createBuffer('hour-deviations', AREA_COUNT * 4);
  const hourPercentilesBuffer = resources.createBuffer('hour-percentiles', AREA_COUNT * 4);
  const hourModesBuffer = resources.createBuffer('hour-modes', AREA_COUNT * 4);
  const activeDaysBuffer = resources.createBuffer('active-days', AREA_COUNT * 4);
  const researchGradeMeansBuffer = resources.createBuffer('research-grade-means', AREA_COUNT * 4);
  const introducedMeansBuffer = resources.createBuffer('introduced-means', AREA_COUNT * 4);
  const metricBuffer = resources.createBuffer('metric', (AREA_COUNT + 1) * 4);
  const matrixOverlayBuffer = resources.createBuffer(
    'matrix-overlay',
    GPU_CALENDAR_BUCKETS_MATRIX_LENGTH * 4
  );

  const drawCommands = resources.track(
    new DrawCommandBuffer(device, {
      id: 'calendar-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const windowParameters = resources.createParameterBuffer(
    'window-parameters',
    'uint32',
    GPU_TIME_WORD_WINDOW_PARAMETER_LENGTH
  );
  const calendarParameters = resources.createParameterBuffer(
    'calendar-parameters',
    'sint32',
    GPU_CALENDAR_BUCKETS_PARAMETER_LENGTH
  );
  const brushParameters = resources.createParameterBuffer('brush-parameters', 'uint32', 4);
  const selectionParameters = resources.createParameterBuffer('selection-parameters', 'uint32', 4);
  const percentileParameters = resources.createParameterBuffer(
    'percentile-parameters',
    'float32',
    1
  );

  // Heat matrix inset in Lake Michigan, east of the Loop: 24 hours by 7 weekdays.
  const matrixSouthWest = areaProjection.project(-87.585, 41.805);
  const matrixNorthEast = areaProjection.project(-87.285, 41.895);
  const matrixBounds: [number, number, number, number] = [
    matrixSouthWest[0],
    matrixSouthWest[1],
    matrixNorthEast[0],
    matrixNorthEast[1]
  ];
  const gridSegments: number[] = [];
  const cellWidth = (matrixBounds[2] - matrixBounds[0]) / 24;
  const cellHeight = (matrixBounds[3] - matrixBounds[1]) / 7;
  for (const hour of [0, 6, 12, 18, 24]) {
    const x = matrixBounds[0] + hour * cellWidth;
    gridSegments.push(x, matrixBounds[1], x, matrixBounds[3]);
  }
  for (let row = 0; row <= 7; row++) {
    const y = matrixBounds[1] + row * cellHeight;
    gridSegments.push(matrixBounds[0], y, matrixBounds[2], y);
  }
  const gridBuffer = resources.createBuffer('matrix-grid', Float32Array.from(gridSegments));

  let destroyed = false;
  let dirty = true;
  let playStart = 0;
  let lastWindowKey = '';
  let current: Variant | null = null;
  let metricMaximum = 1;
  let metricMinimum = 0;
  let matrixMaximum = 1;
  let cpuMetric: Float32Array | null = null;
  let cpuCounts: Uint32Array | null = null;
  let cpuMatrix: Uint32Array | null = null;
  let windowEvents = 0;
  const variants = new Map<string, Variant>();

  function buildVariant(daylightSaving: boolean, variance: 'sample' | 'population'): Variant {
    const key = `${daylightSaving ? 'dst' : 'fixed'}-${variance}`;
    const graph = new GPUCommandGraph<void>(device, {id: `calendar-${key}`});
    const v = createViewImporter(graph, key);
    const timeWords = v('time-words', timeWordsBuffer, 'uint32x2', eventCount);
    const windowMask = v('window-mask', windowMaskBuffer, 'uint32', eventCount);
    graph.add(
      new GPUTimeWindowFilter({
        id: `${key}-window`,
        timestamps: timeWords,
        window: windowParameters.importToGraph(graph),
        additionalPredicates: [
          {kind: 'selection', mask: v('category-mask', categoryMaskBuffer, 'uint32', eventCount)}
        ],
        output: {
          ids: v('ids', idsBuffer, 'uint32', eventCount),
          count: v('window-count', countBuffer, 'uint32', 1),
          overflow: v('overflow', overflowBuffer, 'uint32', 1)
        },
        outputMask: windowMask,
        drawInstanceCount: graph.importGPUData(
          `${key}-draw-count`,
          drawCommands.getInstanceCountData(0)
        )
      })
    );
    const hour = v('hour', hourBuffer, 'uint32', eventCount);
    const weekday = v('weekday', weekdayBuffer, 'uint32', eventCount);
    const dayOfYear = v('day-of-year', dayOfYearBuffer, 'uint32', eventCount);
    graph.add(
      new GPUCalendarBuckets({
        id: `${key}-calendar`,
        timestamps: timeWords,
        mask: windowMask,
        utcOffsets: daylightSaving
          ? v('utc-offsets', offsetsBuffer, 'sint32', eventCount)
          : undefined,
        parameters: calendarParameters.importToGraph(graph),
        output: {
          hour,
          weekday,
          dayOfYear,
          hourWeekdayCounts: v('matrix', matrixBuffer, 'uint32', GPU_CALENDAR_BUCKETS_MATRIX_LENGTH)
        }
      })
    );
    const hourFloat = v('hour-float', hourFloatBuffer, 'float32', eventCount);
    const dayFloat = v('day-float', dayFloatBuffer, 'float32', eventCount);
    const brush = v('brush', brushBuffer, 'uint32', eventCount);
    // Converts the calendar columns to float32 and applies the hour and weekday brush.
    addKernelPass(graph, {
      id: `${key}-brush`,
      invocationCount: eventCount,
      bindings: [
        {name: 'windowMask', view: windowMask, type: 'u32', access: 'read'},
        {name: 'hours', view: hour, type: 'u32', access: 'read'},
        {name: 'weekdays', view: weekday, type: 'u32', access: 'read'},
        {name: 'days', view: dayOfYear, type: 'u32', access: 'read'},
        {
          name: 'brushParameters',
          view: brushParameters.importToGraph(graph),
          type: 'u32',
          access: 'read'
        },
        {name: 'hourFloat', view: hourFloat, type: 'f32', access: 'read_write'},
        {name: 'dayFloat', view: dayFloat, type: 'f32', access: 'read_write'},
        {name: 'brush', view: brush, type: 'u32', access: 'read_write'}
      ],
      body: /* wgsl */ `
  let valid = windowMask[windowMaskOffset + index] != 0u && hours[hoursOffset + index] < 24u;
  if (!valid) {
    hourFloat[hourFloatOffset + index] = bitcast<f32>(0x7fc00000u | (brushParameters[brushParametersOffset + 2u] & 0u));
    dayFloat[dayFloatOffset + index] = bitcast<f32>(0x7fc00000u | (brushParameters[brushParametersOffset + 2u] & 0u));
    brush[brushOffset + index] = 0u;
    return;
  }
  let hour = hours[hoursOffset + index];
  hourFloat[hourFloatOffset + index] = f32(hour);
  dayFloat[dayFloatOffset + index] = f32(days[daysOffset + index]);
  let hourFrom = brushParameters[brushParametersOffset];
  let hourTo = brushParameters[brushParametersOffset + 1u];
  var inside = hour >= hourFrom && hour <= hourTo;
  if (brushParameters[brushParametersOffset + 3u] != 0u) {
    inside = !inside;
  }
  let weekdayBit = 1u << weekdays[weekdaysOffset + index];
  brush[brushOffset + index] = select(0u, 1u, inside && (brushParameters[brushParametersOffset + 2u] & weekdayBit) != 0u);`
    });

    const hourStatistics = [
      'count',
      'mean',
      'median',
      'standardDeviation',
      'percentiles',
      'mode'
    ] as const;
    graph.add(
      new GPUGroupStatistics({
        id: `${key}-areas`,
        keys: v('area-keys', areaKeyBuffer, 'uint32', eventCount),
        mask: brush,
        keyCount: AREA_COUNT,
        variance,
        percentiles: percentileParameters.importToGraph(graph),
        columns: [
          {
            values: hourFloat,
            statistics: hourStatistics,
            output: {
              counts: v('hour-counts', hourCountsBuffer, 'uint32', AREA_COUNT),
              means: v('hour-means', hourMeansBuffer, 'float32', AREA_COUNT),
              medians: v('hour-medians', hourMediansBuffer, 'float32', AREA_COUNT),
              standardDeviations: v('hour-deviations', hourDeviationsBuffer, 'float32', AREA_COUNT),
              percentiles: v('hour-percentiles', hourPercentilesBuffer, 'float32', AREA_COUNT),
              modes: v('hour-modes', hourModesBuffer, 'float32', AREA_COUNT)
            }
          },
          {
            values: dayFloat,
            statistics: ['uniqueCount'],
            output: {uniqueCounts: v('active-days', activeDaysBuffer, 'uint32', AREA_COUNT)}
          },
          {
            values: v('researchGrade', researchGradeBuffer, 'float32', eventCount),
            statistics: ['mean'],
            output: {
              means: v('research-grade-means', researchGradeMeansBuffer, 'float32', AREA_COUNT)
            }
          },
          {
            values: v('introduced', introducedBuffer, 'float32', eventCount),
            statistics: ['mean'],
            output: {means: v('introduced-means', introducedMeansBuffer, 'float32', AREA_COUNT)}
          }
        ],
        output: {
          keys: v('group-keys', groupKeysBuffer, 'uint32', AREA_COUNT),
          counts: v('group-counts', groupCountsBuffer, 'uint32', AREA_COUNT),
          count: v('group-count', groupCountBuffer, 'uint32', 1),
          overflow: v('group-overflow', groupOverflowBuffer, 'uint32', 1)
        }
      })
    );

    // Picks the chosen statistic into one metric buffer (three kernels keep each under the
    // storage-binding limit). Entry AREA_COUNT is the "no area" sentinel (NaN).
    const metric = v('metric', metricBuffer, 'float32', AREA_COUNT + 1);
    const selection = selectionParameters.importToGraph(graph);
    addKernelPass(graph, {
      id: `${key}-metric-primary`,
      invocationCount: AREA_COUNT + 1,
      declarations: `const AREAS: u32 = ${AREA_COUNT}u;`,
      bindings: [
        {
          name: 'counts',
          view: v('hour-counts', hourCountsBuffer, 'uint32', AREA_COUNT),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'means',
          view: v('hour-means', hourMeansBuffer, 'float32', AREA_COUNT),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'medians',
          view: v('hour-medians', hourMediansBuffer, 'float32', AREA_COUNT),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'areaKm2',
          view: v('area-km2', areaKm2Buffer, 'float32', AREA_COUNT),
          type: 'f32',
          access: 'read'
        },
        {name: 'selection', view: selection, type: 'u32', access: 'read'},
        {name: 'metric', view: metric, type: 'f32', access: 'read_write'}
      ],
      body: /* wgsl */ `
  let nan = bitcast<f32>(0x7fc00000u | (selection[selectionOffset] & 0u));
  if (index >= AREAS) {
    metric[metricOffset + index] = nan;
    return;
  }
  let statistic = selection[selectionOffset];
  let count = counts[countsOffset + index];
  var value = nan;
  if (count >= selection[selectionOffset + 1u] && count > 0u) {
    if (statistic == 0u) {
      value = f32(count);
    } else if (statistic == 1u) {
      value = f32(count) / max(areaKm2[areaKm2Offset + index], 0.01);
    } else if (statistic == 2u) {
      value = means[meansOffset + index];
    } else if (statistic == 3u) {
      value = medians[mediansOffset + index];
    }
  }
  metric[metricOffset + index] = value;`
    });
    addKernelPass(graph, {
      id: `${key}-metric-hours`,
      invocationCount: AREA_COUNT,
      declarations: `const AREAS: u32 = ${AREA_COUNT}u;`,
      bindings: [
        {
          name: 'counts',
          view: v('hour-counts', hourCountsBuffer, 'uint32', AREA_COUNT),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'deviations',
          view: v('hour-deviations', hourDeviationsBuffer, 'float32', AREA_COUNT),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'percentiles',
          view: v('hour-percentiles', hourPercentilesBuffer, 'float32', AREA_COUNT),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'modes',
          view: v('hour-modes', hourModesBuffer, 'float32', AREA_COUNT),
          type: 'f32',
          access: 'read'
        },
        {name: 'selection', view: selection, type: 'u32', access: 'read'},
        {name: 'metric', view: metric, type: 'f32', access: 'read_write'}
      ],
      body: /* wgsl */ `
  let statistic = selection[selectionOffset];
  if (statistic < 4u || statistic > 6u) {
    return;
  }
  var value = bitcast<f32>(0x7fc00000u | (selection[selectionOffset] & 0u));
  if (counts[countsOffset + index] >= max(selection[selectionOffset + 1u], 1u)) {
    if (statistic == 4u) {
      value = deviations[deviationsOffset + index];
    } else if (statistic == 5u) {
      value = percentiles[percentilesOffset + index];
    } else {
      value = modes[modesOffset + index];
    }
  }
  metric[metricOffset + index] = value;`
    });
    addKernelPass(graph, {
      id: `${key}-metric-rates`,
      invocationCount: AREA_COUNT,
      declarations: `const AREAS: u32 = ${AREA_COUNT}u;`,
      bindings: [
        {
          name: 'counts',
          view: v('hour-counts', hourCountsBuffer, 'uint32', AREA_COUNT),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'activeDays',
          view: v('active-days', activeDaysBuffer, 'uint32', AREA_COUNT),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'researchGradeShares',
          view: v('research-grade-means', researchGradeMeansBuffer, 'float32', AREA_COUNT),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'introducedShares',
          view: v('introduced-means', introducedMeansBuffer, 'float32', AREA_COUNT),
          type: 'f32',
          access: 'read'
        },
        {name: 'selection', view: selection, type: 'u32', access: 'read'},
        {name: 'metric', view: metric, type: 'f32', access: 'read_write'}
      ],
      body: /* wgsl */ `
  let statistic = selection[selectionOffset];
  if (statistic < 7u) {
    return;
  }
  var value = bitcast<f32>(0x7fc00000u | (selection[selectionOffset] & 0u));
  if (counts[countsOffset + index] >= max(selection[selectionOffset + 1u], 1u)) {
    if (statistic == 7u) {
      value = f32(activeDays[activeDaysOffset + index]);
    } else if (statistic == 8u) {
      value = researchGradeShares[researchGradeSharesOffset + index];
    } else {
      value = introducedShares[introducedSharesOffset + index];
    }
  }
  metric[metricOffset + index] = value;`
    });
    const compiled = resources.track(graph.compile());
    const reader = new SummaryReader(
      resources,
      key,
      [
        {buffer: metricBuffer, size: (AREA_COUNT + 1) * 4},
        {buffer: groupCountsBuffer, size: AREA_COUNT * 4},
        {buffer: matrixBuffer, size: GPU_CALENDAR_BUCKETS_MATRIX_LENGTH * 4},
        {buffer: countBuffer, size: 4},
        {buffer: overflowBuffer, size: 4}
      ],
      bytes => {
        if (destroyed || current?.key !== key) return;
        showSummary(bytes);
      }
    );
    return {key, compiled, reader};
  }

  function getVariant(): Variant {
    const {daylightSaving, variance} = ctx.options;
    const key = `${daylightSaving ? 'dst' : 'fixed'}-${variance}`;
    let variant = variants.get(key);
    if (!variant) {
      variant = buildVariant(daylightSaving, variance);
      variants.set(key, variant);
    }
    return variant;
  }

  function showSummary(bytes: ArrayBuffer): void {
    const metric = new Float32Array(bytes, 0, AREA_COUNT + 1);
    cpuMetric = metric;
    cpuCounts = new Uint32Array(bytes, (AREA_COUNT + 1) * 4, AREA_COUNT);
    cpuMatrix = new Uint32Array(
      bytes,
      (AREA_COUNT + 1) * 4 + AREA_COUNT * 4,
      GPU_CALENDAR_BUCKETS_MATRIX_LENGTH
    );
    const tail = new Uint32Array(
      bytes,
      (AREA_COUNT + 1) * 4 + AREA_COUNT * 4 + GPU_CALENDAR_BUCKETS_MATRIX_LENGTH * 4,
      2
    );
    windowEvents = tail[0];
    let brushed = 0;
    for (const count of cpuCounts) brushed += count;
    ctx.setReadout('windowEvents', `${formatCount(windowEvents)} of ${formatCount(eventCount)}`);
    ctx.setReadout(
      'brushEvents',
      `${formatCount(brushed)} (${windowEvents > 0 ? ((brushed / windowEvents) * 100).toFixed(1) : '0'}% of the window)`
    );
    ctx.setReadout('overflow', tail[1] ? 'yes: capacity exceeded' : 'no');
    let minimum = Infinity;
    let maximum = -Infinity;
    let best = -1;
    for (let area = 0; area < AREA_COUNT; area++) {
      const value = metric[area];
      if (!Number.isFinite(value)) continue;
      minimum = Math.min(minimum, value);
      if (value > maximum) {
        maximum = value;
        best = area;
      }
    }
    if (Number.isFinite(minimum)) {
      metricMinimum = ctx.options.statistic.endsWith('hour') ? 0 : Math.min(0, minimum);
      metricMaximum = ctx.options.statistic.endsWith('hour')
        ? 24
        : Math.max(maximum, metricMinimum + 1e-6);
      if (
        ctx.options.statistic === 'research-grade-share' ||
        ctx.options.statistic === 'introduced-share'
      ) {
        metricMinimum = 0;
      }
      if (
        ctx.options.statistic === 'mean-hour' ||
        ctx.options.statistic === 'median-hour' ||
        ctx.options.statistic === 'percentile-hour' ||
        ctx.options.statistic === 'mode-hour'
      ) {
        metricMinimum = Math.max(0, Math.floor(minimum) - 1);
        metricMaximum = Math.min(24, Math.ceil(maximum) + 1);
      }
      if (ctx.options.statistic === 'hour-spread') {
        metricMinimum = Math.max(0, Math.floor(minimum));
        metricMaximum = Math.ceil(maximum) + 0.5;
      }
      ctx.setLegendExtent('metric', [metricMinimum, metricMaximum]);
    }
    ctx.setReadout(
      'topArea',
      best >= 0 ? `${areaNames[best] ?? `Area ${best + 1}`}: ${formatMetric(metric[best])}` : '-'
    );
    let matrixBest = 0;
    matrixMaximum = 1;
    for (let cell = 0; cell < cpuMatrix.length; cell++) {
      matrixMaximum = Math.max(matrixMaximum, cpuMatrix[cell]);
      if (cpuMatrix[cell] > cpuMatrix[matrixBest]) matrixBest = cell;
    }
    const first = Number(ctx.options.firstDay);
    ctx.setReadout(
      'busiestCell',
      `${WEEKDAY_NAMES[(first + Math.floor(matrixBest / 24)) % 7]} ${String(matrixBest % 24).padStart(2, '0')}:00 (${formatCount(cpuMatrix[matrixBest])} observations)`
    );
    ctx.setLegendExtent('matrix', [0, matrixMaximum]);
    ctx.requestLayers();
  }

  function formatMetric(value: number): string {
    const statistic = ctx.options.statistic;
    if (!Number.isFinite(value)) return 'n/a';
    if (statistic === 'research-grade-share' || statistic === 'introduced-share')
      return `${(value * 100).toFixed(1)}%`;
    if (statistic.endsWith('hour') || statistic === 'hour-spread') {
      const hour = Math.floor(value);
      return statistic === 'hour-spread'
        ? `${value.toFixed(1)} h`
        : `${String(hour).padStart(2, '0')}:${String(Math.round((value - hour) * 60)).padStart(2, '0')}`;
    }
    return value >= 100 ? formatCount(value) : value.toFixed(1);
  }

  function writeWindow(): void {
    const options = ctx.options;
    let startDay = options.dateRange[0];
    let endDay = options.dateRange[1];
    if (options.playWindow) {
      endDay = Math.min(365, playStart + options.windowDays);
      startDay = playStart;
    }
    const start = YEAR_START_MS + startDay * 86400000;
    const end = YEAR_START_MS + endDay * 86400000 - 1;
    windowParameters.write(getGPUTimeWindowWordParameterValues({start, end}));
    ctx.setReadout(
      'windowDates',
      `${formatDate(startDay)} to ${formatDate(Math.max(startDay, endDay - 1))}`
    );
    dirty = true;
  }

  function writeCategory(): void {
    const mask = new Uint32Array(eventCount);
    fillCategoryMask(events, ctx.options.groupType, mask);
    categoryMaskBuffer.write(mask);
    dirty = true;
  }

  function writeCalendar(): void {
    const options = ctx.options;
    calendarParameters.write(getGPUCalendarBucketsParametersFor(options));
    dirty = true;
  }

  function getGPUCalendarBucketsParametersFor(options: CalendarPatternsOptions): Int32Array {
    return getGPUCalendarBucketsParameterValues(options.utcOffset * 60, Number(options.firstDay));
  }

  function writeBrush(): void {
    const options = ctx.options;
    const first = Number(options.firstDay);
    let mask = 0;
    for (const day of WEEKDAY_SETS[options.weekdays] ?? WEEKDAY_SETS.all) {
      mask |= 1 << ((day - first + 7) % 7);
    }
    brushParameters.write(
      Uint32Array.of(options.hours[0], options.hours[1], mask, options.invertHours ? 1 : 0)
    );
    // Overlay of excluded matrix cells (1 = excluded), laid out weekday * 24 + hour.
    const overlay = new Uint32Array(GPU_CALENDAR_BUCKETS_MATRIX_LENGTH);
    for (let weekday = 0; weekday < 7; weekday++) {
      for (let hour = 0; hour < 24; hour++) {
        let inside = hour >= options.hours[0] && hour <= options.hours[1];
        if (options.invertHours) inside = !inside;
        const weekdayIn = (mask & (1 << weekday)) !== 0;
        overlay[weekday * 24 + hour] = inside && weekdayIn ? 0 : 1;
      }
    }
    matrixOverlayBuffer.write(overlay);
    dirty = true;
  }

  function writeSelection(): void {
    const options = ctx.options;
    selectionParameters.write(
      Uint32Array.of(STATISTIC_CODES[options.statistic], options.minimumEvents, 0, 0)
    );
    percentileParameters.write(Float32Array.of(options.percentile));
    dirty = true;
  }

  function adopt(): void {
    current = getVariant();
    writeWindow();
    writeCategory();
    writeCalendar();
    writeBrush();
    writeSelection();
    ctx.requestLayers();
  }

  adopt();

  return {
    getCompiledGraphs: () => (current ? [current.compiled as CompiledGPUCommandGraph<never>] : []),

    setOption(id) {
      if (id === 'daylightSaving' || id === 'variance') {
        adopt();
      } else if (id === 'dateRange' || id === 'windowDays' || id === 'playWindow') {
        writeWindow();
      } else if (id === 'groupType') {
        writeCategory();
      } else if (id === 'utcOffset') {
        writeCalendar();
      } else if (id === 'firstDay') {
        writeCalendar();
        writeBrush();
      } else if (id === 'hours' || id === 'invertHours' || id === 'weekdays') {
        writeBrush();
      } else if (id === 'statistic' || id === 'percentile' || id === 'minimumEvents') {
        writeSelection();
        if (id === 'statistic') ctx.requestLayers();
      } else {
        ctx.requestLayers();
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip({coordinate}) {
      if (!coordinate || !cpuMetric || !cpuCounts) return null;
      const feature = lookupFeature(raster, coordinate[0], coordinate[1]);
      if (feature < 0 || feature >= AREA_COUNT) return null;
      return [
        areaNames[feature] ?? `Area ${feature + 1}`,
        `${CALENDAR_STATISTIC_LABELS[ctx.options.statistic].title}: ${formatMetric(cpuMetric[feature])}`,
        `${formatCount(cpuCounts[feature])} observations in the brush`
      ].join('\n');
    },

    encode(commandEncoder, frame) {
      if (!current) return;
      const options = ctx.options;
      if (options.playWindow) {
        const span = Math.max(1, 365 - options.windowDays);
        playStart = (frame.timeSeconds * PLAY_DAYS_PER_SECOND) % span;
        const key = `${Math.round(playStart)}`;
        if (key !== lastWindowKey) {
          lastWindowKey = key;
          playStart = Math.round(playStart);
          writeWindow();
        }
      }
      if (dirty || frame.frameIndex < 3) {
        current.compiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
        current.reader.request(commandEncoder);
      }
      current.reader.flush(commandEncoder);
    },

    getLayers() {
      if (!current) return [];
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
      const layers: Layer[] = [];
      if (options.showAreas) {
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: 'calendar-areas',
            coordinateOrigin,
            gridSize: [raster.width, raster.height],
            bounds: raster.bounds,
            valueIndices: cellFeatureBuffer,
            values: metricBuffer,
            valueFormat: 'float32',
            colormap: options.ramp,
            valueRange: [metricMinimum, metricMaximum],
            sqrtScale:
              options.statistic === 'count' ||
              options.statistic === 'density' ||
              options.statistic === 'active-days',
            noDataColor: [0, 0, 0, 0],
            opacity: options.opacity,
            color: [255, 255, 255, 255]
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'calendar-outlines',
            coordinateOrigin,
            segments: outlineBuffer,
            instanceCount: outlineSegments.length / 4,
            widthPixels: 0.9,
            colormap: 'uniform',
            color: dark ? [235, 240, 250, 90] : [30, 40, 60, 100]
          })
        );
      }
      if (options.showPoints) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'calendar-points',
            coordinateOrigin,
            positions: positionsBuffer,
            ids: idsBuffer,
            drawCommands,
            radiusPixels: 1.6,
            values: hourFloatBuffer,
            valueFormat: 'float32',
            colormap: 'cividis',
            valueRange: [0, 24],
            color: [255, 255, 255, 200]
          })
        );
      }
      if (options.showMatrix) {
        const common = {
          coordinateOrigin,
          gridSize: [24, 7] as const,
          bounds: matrixBounds,
          rowOrigin: 'north' as const
        };
        layers.push(
          new SpatialAnalysisRasterLayer({
            ...common,
            id: 'calendar-matrix',
            values: matrixBuffer,
            valueFormat: 'uint32',
            colormap: 'inferno',
            valueRange: [0, matrixMaximum],
            sqrtScale: true,
            noDataColor: [0, 0, 0, 0],
            color: [255, 255, 255, 255]
          }),
          new SpatialAnalysisRasterLayer({
            ...common,
            id: 'calendar-matrix-excluded',
            values: matrixOverlayBuffer,
            valueFormat: 'uint32',
            colormap: 'mask',
            color: dark ? [10, 14, 24, 170] : [240, 244, 250, 190],
            noDataColor: [0, 0, 0, 0]
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'calendar-matrix-grid',
            coordinateOrigin,
            segments: gridBuffer,
            instanceCount: gridSegments.length / 4,
            widthPixels: 1,
            colormap: 'uniform',
            color: dark ? [255, 255, 255, 140] : [30, 40, 60, 140]
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      for (const variant of variants.values()) variant.reader.stop();
      resources.destroy();
    }
  };
}

export type {SpatialAnalysisColor};
