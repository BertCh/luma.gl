// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUTemporalReductionParameterValues,
  GPU_TEMPORAL_REDUCTION_PARAMETER_LENGTH,
  GPUTemporalReduction
} from '@luma.gl/experimental/gpu-dataframe';
import {
  getGPUNeighborSearchParameterValues,
  getGPUSpaceTimeParameterValues,
  getKnoxPoissonPValue,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  GPU_SPACE_TIME_PARAMETER_LENGTH,
  GPU_SPACE_TIME_SUMMARY,
  GPU_SPACE_TIME_SUMMARY_LENGTH,
  GPUKnoxTest,
  GPUNeighborSearch
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {SpatialAnalysisRasterLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {createPlaybackClock} from '../../engine/playback';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {createViewImporter} from './b13-views';
import {QuakeEventLayer} from './quake-event-layer';
import {readQuakeEvents, type QuakeEvents} from './quake-data';
import {
  formatQuakeDateTime,
  QUAKE_DAY_COUNT,
  QUAKE_DAY_ZERO_MS,
  QUAKE_REGIONS,
  type QuakeRegionId
} from './quake-regions';

/** Option state of the quake-aftershocks scene. */
export type QuakeAftershocksOptions = {
  playing: boolean;
  time: number;
  playbackSpeed: string;
  loop: boolean;
  region: QuakeRegionId;
  minimumMagnitude: '4' | '4.5' | '5';
  fadeDays: number;
  sizeScale: number;
  colorBy: 'depth' | 'age';
  showPast: boolean;
  spatialRadius: number;
  timeThreshold: number;
  permutations: number;
  showPairs: boolean;
  gridView: 'off' | 'month-count' | 'month-max' | 'burst';
  gridCells: number;
  minimumCellEvents: number;
  gridOpacity: number;
};

const MAXIMUM_PERMUTATIONS = 999;
const SLOTS_PER_EVENT = 256;
const MAXIMUM_GRID = 48;
const MAXIMUM_CELLS = MAXIMUM_GRID * MAXIMUM_GRID;
const MONTH_DAYS = 365.25 / 12;
const MONTH_COUNT = 60;
const NO_CELL = 0xffffffff;
const SUMMARY = GPU_SPACE_TIME_SUMMARY;
const HISTOGRAM_BINS = 40;

/** `Feb 2023` for a day count since 2020-01-01. */
function formatMonth(days: number): string {
  const date = new Date(QUAKE_DAY_ZERO_MS + days * 86400000);
  const month = date.toLocaleString('en-GB', {month: 'short', timeZone: 'UTC'});
  return `${month} ${date.getUTCFullYear()}`;
}

function formatNumber(value: number, digits = 2): string {
  return Number.isFinite(value) ? value.toFixed(digits) : 'n/a';
}

function formatProbability(value: number): string {
  if (!Number.isFinite(value)) return 'n/a';
  return value < 0.001 ? '< 0.001' : value.toFixed(3);
}

type KnoxPart = {
  compiled: CompiledGPUCommandGraph<void>;
  segments: Buffer;
  fade: Buffer;
  slotCapacity: number;
  searchBounds: [number, number, number, number];
  reader: SummaryReader;
};

type MonthlyPart = {
  compiled: CompiledGPUCommandGraph<void>;
  counts: Buffer;
  max: Buffer;
  burst: Buffer;
  reader: SummaryReader;
};

type Variant = {
  key: string;
  events: QuakeEvents;
  positions: Buffer;
  days: Buffer;
  magnitude: Buffer;
  depth: Buffer;
  knox: KnoxPart;
  monthly: MonthlyPart;
};

/**
 * Aftershock playback for one region of the 2020 to 2024 catalog. Two compiled graphs per
 * (region, minimum magnitude): `GPUNeighborSearch` lists the pairs close in space and
 * `GPUKnoxTest` counts how many are also close in time against a permutation reference;
 * `GPUTemporalReduction` bins magnitude by (grid cell, 30-day month) and a kernel turns the
 * counts into a burstiness map. The playback layer reads the catalog buffers directly. Radius,
 * time threshold, permutations, seed, grid size and the playhead are parameter writes.
 */
export async function createQuakeAftershocks(
  ctx: SceneContext<QuakeAftershocksOptions>
): Promise<SceneInstance<QuakeAftershocksOptions>> {
  const {device} = ctx;
  const catalog = ctx.datasets.get('poopdeck-earthquakes');
  const resources = new SpatialAnalysisResources(device, 'quake-aftershocks');

  const searchParameters = resources.createParameterBuffer(
    'search-parameters',
    'float32',
    GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH
  );
  const testParameters = resources.createParameterBuffer(
    'test-parameters',
    'uint32',
    GPU_SPACE_TIME_PARAMETER_LENGTH
  );
  const drawParameters = resources.createParameterBuffer('draw-parameters', 'float32', 2);
  const reductionParameters = resources.createParameterBuffer(
    'reduction-parameters',
    'float32',
    GPU_TEMPORAL_REDUCTION_PARAMETER_LENGTH,
    getGPUTemporalReductionParameterValues(0, MONTH_DAYS)
  );
  // Cell index of every month slice row for the map: cell c reads slot c * MONTH_COUNT + month.
  const sliceIndices = resources.createBuffer('slice-indices', MAXIMUM_CELLS * 4);
  // [minX, minY, 1 / cellWidth, 1 / cellHeight, cells per side, minimum events, 0, 0]
  const gridParameters = resources.createParameterBuffer('grid-parameters', 'float32', 8);

  const clock = createPlaybackClock(
    ctx,
    {time: 'time', play: 'playing', speed: 'playbackSpeed', loop: 'loop'},
    {range: [0, QUAKE_DAY_COUNT], rate: 1, step: 0.25}
  );

  let destroyed = false;
  let seed = 11;
  let knoxDirty = true;
  let monthlyDirty = true;
  let current: Variant | null = null;
  let cpuCounts: Uint32Array | null = null;
  let playheadDay = ctx.options.time;
  let lastChartMonth = -1;
  const variants = new Map<string, Variant>();

  function buildVariant(key: string, events: QuakeEvents): Variant {
    const count = events.count;
    const buffer = (name: string, data: Float32Array | Uint32Array | number) =>
      resources.createBuffer(`${key}-${name}`, data);
    const positions = buffer('positions', events.positions);
    const days = buffer('days', events.days);
    const magnitude = buffer('magnitude', events.magnitude);
    const depth = buffer('depth', events.depth);

    // ---- Knox: pairs close in space, then the space-time interaction test -------------------
    const slotCapacity = count * SLOTS_PER_EVENT;
    const offsets = buffer('offsets', (count + 1) * 4);
    const neighbors = buffer('neighbors', slotCapacity * 4);
    const weights = buffer('weights', slotCapacity * 4);
    const distances = buffer('distances', slotCapacity * 4);
    const overflow = buffer('overflow', 4);
    const knoxStatistics = buffer('knox-statistics', (MAXIMUM_PERMUTATIONS + 1) * 4);
    const knoxSummary = buffer('knox-summary', GPU_SPACE_TIME_SUMMARY_LENGTH * 4);
    const segments = buffer('segments', slotCapacity * 16);
    const fade = buffer('fade', slotCapacity * 4);
    const knoxGraph = new GPUCommandGraph<void>(device, {id: `quake-knox-${key}`});
    const v = createViewImporter(knoxGraph, `${key}-knox`);
    const positionsView = v('positions', positions, 'float32x2', count);
    const timesView = v('days', days, 'float32', count);
    const pairs = {
      offsets: v('offsets', offsets, 'uint32', count + 1),
      neighbors: v('neighbors', neighbors, 'uint32', slotCapacity),
      weights: v('weights', weights, 'float32', slotCapacity),
      distances: v('distances', distances, 'float32', slotCapacity)
    };
    knoxGraph.add(
      new GPUNeighborSearch({
        id: `${key}-search`,
        mode: 'radius',
        positions: positionsView,
        parameters: searchParameters.importToGraph(knoxGraph),
        gridSize: [64, 64],
        weights: pairs,
        overflow: v('overflow', overflow, 'uint32', 1)
      })
    );
    knoxGraph.add(
      new GPUKnoxTest({
        id: `${key}-knox`,
        pairs,
        times: timesView,
        parameters: testParameters.importToGraph(knoxGraph),
        maximumPermutations: MAXIMUM_PERMUTATIONS,
        statistics: v('knox-statistics', knoxStatistics, 'uint32', MAXIMUM_PERMUTATIONS + 1),
        summary: v('knox-summary', knoxSummary, 'float32', GPU_SPACE_TIME_SUMMARY_LENGTH)
      })
    );
    // One thread per pair slot finds its row by binary search and writes the segment of each
    // unordered pair (entries j > i) that is close in time; every other slot stays empty.
    addKernelPass(knoxGraph, {
      id: `${key}-segments`,
      invocationCount: slotCapacity,
      declarations: `const EVENT_COUNT: u32 = ${count}u;`,
      bindings: [
        {name: 'offsets', view: pairs.offsets, type: 'u32', access: 'read'},
        {name: 'neighbors', view: pairs.neighbors, type: 'u32', access: 'read'},
        {name: 'positions', view: positionsView, type: 'f32', access: 'read'},
        {name: 'times', view: timesView, type: 'f32', access: 'read'},
        {
          name: 'drawParameters',
          view: drawParameters.importToGraph(knoxGraph),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'segments',
          view: v('segments', segments, 'float32', slotCapacity * 4),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'fade',
          view: v('fade', fade, 'float32', slotCapacity),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: /* wgsl */ `
  segments[segmentsOffset + index * 4u] = 0.0;
  segments[segmentsOffset + index * 4u + 1u] = 0.0;
  segments[segmentsOffset + index * 4u + 2u] = 0.0;
  segments[segmentsOffset + index * 4u + 3u] = 0.0;
  fade[fadeOffset + index] = 0.0;
  if (index >= offsets[offsetsOffset + EVENT_COUNT]) {
    return;
  }
  var low = 0u;
  var high = EVENT_COUNT;
  while (low < high) {
    let middle = (low + high + 1u) / 2u;
    if (offsets[offsetsOffset + middle] <= index) {
      low = middle;
    } else {
      high = middle - 1u;
    }
  }
  let neighbor = neighbors[neighborsOffset + index];
  if (neighbor <= low) {
    return;
  }
  let timeGap = abs(times[timesOffset + low] - times[timesOffset + neighbor]);
  if (timeGap > drawParameters[drawParametersOffset]) {
    return;
  }
  segments[segmentsOffset + index * 4u] = positions[positionsOffset + low * 2u];
  segments[segmentsOffset + index * 4u + 1u] = positions[positionsOffset + low * 2u + 1u];
  segments[segmentsOffset + index * 4u + 2u] = positions[positionsOffset + neighbor * 2u];
  segments[segmentsOffset + index * 4u + 3u] = positions[positionsOffset + neighbor * 2u + 1u];
  fade[fadeOffset + index] = drawParameters[drawParametersOffset + 1u];`
    });
    const knoxCompiled = resources.track(knoxGraph.compile());
    const knoxReader = new SummaryReader(
      resources,
      `${key}-knox`,
      [
        {buffer: knoxSummary, size: GPU_SPACE_TIME_SUMMARY_LENGTH * 4},
        {buffer: overflow, size: 4},
        {buffer: knoxStatistics, size: (MAXIMUM_PERMUTATIONS + 1) * 4}
      ],
      bytes => {
        if (destroyed || current?.key !== key) return;
        showKnox(bytes);
      }
    );
    const [west, south, east, north] = events.windowMeters;
    const knox: KnoxPart = {
      compiled: knoxCompiled,
      segments,
      fade,
      slotCapacity,
      searchBounds: [west - 5000, south - 5000, east + 5000, north + 5000],
      reader: knoxReader
    };

    // ---- Month x cell reduction and the burstiness map --------------------------------------
    const slotCount = MAXIMUM_CELLS * MONTH_COUNT;
    const cellIds = buffer('cell-ids', count * 4);
    const counts = buffer('month-counts', slotCount * 4);
    const monthMin = buffer('month-min', slotCount * 4);
    const monthMax = buffer('month-max', slotCount * 4);
    const monthFirst = buffer('month-first', slotCount * 4);
    const monthLast = buffer('month-last', slotCount * 4);
    const burst = buffer('burst', MAXIMUM_CELLS * 4);
    const monthlyGraph = new GPUCommandGraph<void>(device, {id: `quake-monthly-${key}`});
    const m = createViewImporter(monthlyGraph, `${key}-monthly`);
    const monthlyPositions = m('positions', positions, 'float32x2', count);
    const cellIdView = m('cell-ids', cellIds, 'uint32', count);
    const countsView = m('month-counts', counts, 'uint32', slotCount);
    const gridView = gridParameters.importToGraph(monthlyGraph);
    // Each event's grid cell, from the window and the cells-per-side parameter.
    addKernelPass(monthlyGraph, {
      id: `${key}-cell-ids`,
      invocationCount: count,
      bindings: [
        {name: 'positions', view: monthlyPositions, type: 'f32', access: 'read'},
        {name: 'grid', view: gridView, type: 'f32', access: 'read'},
        {name: 'cellIds', view: cellIdView, type: 'u32', access: 'read_write'}
      ],
      body: /* wgsl */ `
  let sideCount = grid[gridOffset + 4u];
  let column = floor((positions[positionsOffset + index * 2u] - grid[gridOffset]) * grid[gridOffset + 2u]);
  let row = floor((positions[positionsOffset + index * 2u + 1u] - grid[gridOffset + 1u]) * grid[gridOffset + 3u]);
  if (column < 0.0 || row < 0.0 || column >= sideCount || row >= sideCount) {
    cellIds[cellIdsOffset + index] = ${NO_CELL}u;
    return;
  }
  cellIds[cellIdsOffset + index] = u32(row) * u32(sideCount) + u32(column);`
    });
    monthlyGraph.add(
      new GPUTemporalReduction({
        id: `${key}-reduction`,
        cellIds: cellIdView,
        timestamps: m('days', days, 'float32', count),
        values: m('magnitude', magnitude, 'float32', count),
        parameters: reductionParameters.importToGraph(monthlyGraph),
        cellCount: MAXIMUM_CELLS,
        bucketCount: MONTH_COUNT,
        output: {
          counts: countsView,
          min: m('month-min', monthMin, 'float32', slotCount),
          max: m('month-max', monthMax, 'float32', slotCount),
          first: m('month-first', monthFirst, 'float32', slotCount),
          last: m('month-last', monthLast, 'float32', slotCount),
          occupiedSlots: {
            ids: m('occupied-ids', buffer('occupied-ids', slotCount * 4), 'uint32', slotCount),
            count: m('occupied-count', buffer('occupied-count', 4), 'uint32', 1),
            overflow: m('occupied-overflow', buffer('occupied-overflow', 4), 'uint32', 1)
          }
        }
      })
    );
    // Burstiness of a cell: its busiest month over its average month, -1 when it has too few events.
    addKernelPass(monthlyGraph, {
      id: `${key}-burst`,
      invocationCount: MAXIMUM_CELLS,
      declarations: `const MONTHS: u32 = ${MONTH_COUNT}u;`,
      bindings: [
        {name: 'counts', view: countsView, type: 'u32', access: 'read'},
        {name: 'grid', view: gridView, type: 'f32', access: 'read'},
        {
          name: 'burst',
          view: m('burst', burst, 'float32', MAXIMUM_CELLS),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: /* wgsl */ `
  let sideCount = u32(grid[gridOffset + 4u]);
  var total = 0u;
  var peak = 0u;
  if (index < sideCount * sideCount) {
    for (var month = 0u; month < MONTHS; month++) {
      let monthCount = counts[countsOffset + index * MONTHS + month];
      total += monthCount;
      peak = max(peak, monthCount);
    }
  }
  var value = -1.0;
  if (total > 0u && f32(total) >= grid[gridOffset + 5u]) {
    value = f32(peak) * f32(MONTHS) / f32(total);
  }
  burst[burstOffset + index] = value;`
    });
    const monthlyCompiled = resources.track(monthlyGraph.compile());
    const monthlyReader = new SummaryReader(
      resources,
      `${key}-monthly`,
      [{buffer: counts, size: slotCount * 4}],
      bytes => {
        if (destroyed || current?.key !== key) return;
        cpuCounts = new Uint32Array(bytes);
        lastChartMonth = -1;
        showMonthly();
      }
    );
    return {
      key,
      events,
      positions,
      days,
      magnitude,
      depth,
      knox,
      monthly: {compiled: monthlyCompiled, counts, max: monthMax, burst, reader: monthlyReader}
    };
  }

  function getVariant(): Variant {
    const options = ctx.options;
    const key = `${options.region}|${options.minimumMagnitude}`;
    let variant = variants.get(key);
    if (!variant) {
      const events = readQuakeEvents(catalog, options.region, Number(options.minimumMagnitude));
      variant = buildVariant(key, events);
      variants.set(key, variant);
    }
    return variant;
  }

  function showKnox(bytes: ArrayBuffer): void {
    const words = GPU_SPACE_TIME_SUMMARY_LENGTH;
    const summary = new Float32Array(bytes, 0, words);
    const overflowFlag = new Uint32Array(bytes, words * 4, 1)[0];
    const permuted = new Uint32Array(bytes, words * 4 + 4, MAXIMUM_PERMUTATIONS + 1);
    const permutations = ctx.options.permutations;
    const observed = summary[SUMMARY.observed];
    const expected = summary[SUMMARY.expected];
    const mean = summary[SUMMARY.permutationMean];
    const sd = Math.sqrt(summary[SUMMARY.permutationVariance]);
    ctx.setReadout('pairs', summary[SUMMARY.pairCount]);
    ctx.setReadout('timeClose', summary[SUMMARY.timeClosePairs]);
    ctx.setReadout('knoxObserved', observed);
    ctx.setReadout('knoxExpected', formatNumber(expected, 1));
    ctx.setReadout('knoxRatio', `${formatNumber(observed / Math.max(expected, 1e-9), 1)}x`);
    ctx.setReadout('knoxPermuted', `${formatNumber(mean, 1)} ± ${formatNumber(sd, 1)}`);
    ctx.setReadout('knoxZ', formatNumber(summary[SUMMARY.zScore], 1));
    ctx.setReadout('knoxP', formatProbability(summary[SUMMARY.pseudoPGreater]));
    ctx.setReadout('knoxPoisson', formatProbability(getKnoxPoissonPValue(observed, expected)));
    ctx.setReadout('pairOverflow', overflowFlag ? 'yes: lower the radius' : 'no');
    // Permutation histogram over mean +- 4 sd; the observed count is pinned to the right edge when
    // it is further out than 12 sd.
    const low = Math.max(0, mean - 4 * sd);
    const reach = Math.max(mean + 4 * sd, Math.min(observed, mean + 12 * sd));
    const high = Math.max(reach, low + 1);
    const bins = new Array<number>(HISTOGRAM_BINS).fill(0);
    for (let permutation = 1; permutation <= permutations; permutation++) {
      const value = permuted[permutation];
      const bin = Math.floor(((value - low) / (high - low)) * HISTOGRAM_BINS);
      if (bin >= 0 && bin < HISTOGRAM_BINS) bins[bin]++;
    }
    const offScale = observed > high;
    ctx.setChart('knoxHistogram', {
      kind: 'histogram',
      values: bins,
      xDomain: [low, high],
      xLabel: 'time-close pairs among the spatial pairs',
      yLabel: 'permutations',
      markers: [
        {
          x: Math.min(observed, high),
          label: offScale ? `observed ${formatCount(observed)} (off the scale)` : 'observed'
        }
      ],
      description:
        'Histogram of the Knox count under random reassignment of the event times, with the observed count marked.'
    });
    ctx.requestLayers();
  }

  function showMonthly(): void {
    if (!current || !cpuCounts) return;
    const monthTotals = new Array<number>(MONTH_COUNT).fill(0);
    const cells = Math.min(MAXIMUM_CELLS, ctx.options.gridCells ** 2);
    for (let cell = 0; cell < cells; cell++) {
      for (let month = 0; month < MONTH_COUNT; month++) {
        monthTotals[month] += cpuCounts[cell * MONTH_COUNT + month];
      }
    }
    const month = Math.min(MONTH_COUNT - 1, Math.floor(playheadDay / MONTH_DAYS));
    lastChartMonth = month;
    let busiest = 0;
    for (let index = 1; index < MONTH_COUNT; index++) {
      if (monthTotals[index] > monthTotals[busiest]) busiest = index;
    }
    ctx.setReadout(
      'busiestMonth',
      `${formatMonth(busiest * MONTH_DAYS)}: ${formatCount(monthTotals[busiest])} events`
    );
    ctx.setChart('monthly', {
      kind: 'line',
      xLabel: 'month (30.4-day buckets from 1 Jan 2020)',
      yLabel: 'events per month',
      series: [
        {label: 'events', x: monthTotals.map((_, index) => index), y: monthTotals, area: true}
      ],
      formatX: value => formatMonth(value * MONTH_DAYS),
      markers: [{x: month, label: 'playhead'}],
      description:
        'Events per month in the region, summed over the grid cells by GPUTemporalReduction.'
    });
  }

  function writeKnoxParameters(): void {
    if (!current) return;
    const options = ctx.options;
    searchParameters.write(
      getGPUNeighborSearchParameterValues({
        bounds: current.knox.searchBounds,
        radius: options.spatialRadius * 1000,
        weightKind: 'binary'
      })
    );
    testParameters.write(
      getGPUSpaceTimeParameterValues({
        seed,
        permutations: options.permutations,
        timeThreshold: options.timeThreshold
      })
    );
    drawParameters.write(Float32Array.of(options.timeThreshold, options.showPairs ? 0.5 : 0));
    knoxDirty = true;
  }

  function writeGridParameters(): void {
    if (!current) return;
    const options = ctx.options;
    const [west, south, east, north] = current.events.windowMeters;
    const side = options.gridCells;
    gridParameters.write(
      Float32Array.of(
        west,
        south,
        side / (east - west),
        side / (north - south),
        side,
        options.minimumCellEvents,
        0,
        0
      )
    );
    monthlyDirty = true;
  }

  function showPlayheadReadouts(): void {
    if (!current) return;
    const {days, magnitude, count} = current.events;
    const fade = ctx.options.fadeDays;
    // First event after the window start (the catalog is in time order).
    let low = 0;
    let high = count;
    const start = playheadDay - fade;
    while (low < high) {
      const middle = (low + high) >> 1;
      if (days[middle] < start) low = middle + 1;
      else high = middle;
    }
    let visible = 0;
    let largest = 0;
    for (let index = low; index < count && days[index] <= playheadDay; index++) {
      visible++;
      largest = Math.max(largest, magnitude[index]);
    }
    ctx.setReadout('clock', formatQuakeDateTime(playheadDay));
    ctx.setReadout('windowCount', visible);
    ctx.setReadout('windowLargest', largest > 0 ? `M${largest.toFixed(1)}` : '-');
    const month = Math.min(MONTH_COUNT - 1, Math.floor(playheadDay / MONTH_DAYS));
    if (month !== lastChartMonth && cpuCounts) showMonthly();
  }

  function adoptVariant(): void {
    current = getVariant();
    cpuCounts = null;
    lastChartMonth = -1;
    ctx.setReadout(
      'events',
      `${formatCount(current.events.count)} M${ctx.options.minimumMagnitude}+ earthquakes in ${QUAKE_REGIONS[ctx.options.region].label}`
    );
    writeKnoxParameters();
    writeGridParameters();
    showPlayheadReadouts();
    ctx.requestLayers();
  }

  adoptVariant();

  return {
    getCompiledGraphs: () =>
      current
        ? ([current.knox.compiled, current.monthly.compiled] as CompiledGPUCommandGraph<never>[])
        : [],

    setOption(id) {
      if (id === 'region' || id === 'minimumMagnitude') {
        adoptVariant();
        if (id === 'region') {
          // Follow the region unless the camera is already looking at it (a story step sets both).
          const [west, south, east, north] = QUAKE_REGIONS[ctx.options.region].bbox;
          const view = ctx.getViewState();
          const inside =
            view.longitude >= west &&
            view.longitude <= east &&
            view.latitude >= south &&
            view.latitude <= north;
          if (!inside) ctx.flyTo(QUAKE_REGIONS[ctx.options.region].view, {transitionMs: 1000});
        }
      } else if (
        id === 'spatialRadius' ||
        id === 'timeThreshold' ||
        id === 'permutations' ||
        id === 'showPairs'
      ) {
        writeKnoxParameters();
      } else if (id === 'gridCells' || id === 'minimumCellEvents') {
        writeGridParameters();
      } else if (id === 'fadeDays') {
        showPlayheadReadouts();
      }
      ctx.requestLayers();
    },

    onAction(id) {
      if (id === 'reseed') {
        seed++;
        writeKnoxParameters();
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip({coordinate}) {
      if (!current || !coordinate || !cpuCounts || ctx.options.gridView === 'off') return null;
      const [west, south, east, north] = current.events.windowMeters;
      const [x, y] = current.events.projection.project(coordinate[0], coordinate[1]);
      const side = ctx.options.gridCells;
      const column = Math.floor(((x - west) / (east - west)) * side);
      const row = Math.floor(((y - south) / (north - south)) * side);
      if (column < 0 || row < 0 || column >= side || row >= side) return null;
      const cell = row * side + column;
      let total = 0;
      let peak = 0;
      let peakMonth = 0;
      for (let month = 0; month < MONTH_COUNT; month++) {
        const monthCount = cpuCounts[cell * MONTH_COUNT + month];
        total += monthCount;
        if (monthCount > peak) {
          peak = monthCount;
          peakMonth = month;
        }
      }
      if (total === 0) return 'No events in this cell';
      return `${formatCount(total)} events in five years\nbusiest month: ${formatMonth(peakMonth * MONTH_DAYS)} with ${formatCount(peak)}`;
    },

    encode(commandEncoder, frame) {
      if (!current) return;
      playheadDay = clock.advance(frame);
      if (clock.moved) {
        showPlayheadReadouts();
        ctx.requestLayers();
      }
      if (knoxDirty || frame.frameIndex < 2) {
        current.knox.compiled.encode(commandEncoder, {parameters: undefined});
        knoxDirty = false;
        current.knox.reader.request(commandEncoder);
      }
      if (monthlyDirty || frame.frameIndex < 2) {
        current.monthly.compiled.encode(commandEncoder, {parameters: undefined});
        monthlyDirty = false;
        current.monthly.reader.request(commandEncoder);
        ctx.requestLayers();
      }
      current.knox.reader.flush(commandEncoder);
      current.monthly.reader.flush(commandEncoder);
    },

    getLayers() {
      if (!current) return [];
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const origin = current.events.origin;
      const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
      const layers: Layer[] = [];
      if (options.gridView !== 'off') {
        const [west, south, east, north] = current.events.windowMeters;
        const side = options.gridCells;
        const month = Math.min(MONTH_COUNT - 1, Math.floor(playheadDay / MONTH_DAYS));
        const common = {
          coordinateOrigin,
          gridSize: [side, side] as const,
          bounds: [west, south, east, north] as const,
          tessellation: 32,
          opacity: options.gridOpacity,
          color: [255, 255, 255, 255] as const,
          colormap: 'cividis' as const
        };
        if (options.gridView === 'burst') {
          layers.push(
            new SpatialAnalysisRasterLayer({
              ...common,
              id: 'quake-grid-burst',
              values: current.monthly.burst,
              valueFormat: 'float32',
              valueRange: [1, 40],
              sqrtScale: true,
              discardAtOrBelow: 0
            })
          );
        } else {
          // Cell c of the month slice reads slot c * MONTH_COUNT + month of the reduction output.
          const indices = new Uint32Array(side * side);
          for (let cell = 0; cell < indices.length; cell++)
            indices[cell] = cell * MONTH_COUNT + month;
          sliceIndices.write(indices);
          layers.push(
            new SpatialAnalysisRasterLayer({
              ...common,
              id: `quake-grid-${options.gridView}`,
              values:
                options.gridView === 'month-count' ? current.monthly.counts : current.monthly.max,
              valueFormat: options.gridView === 'month-count' ? 'uint32' : 'float32',
              valueIndices: sliceIndices,
              valueRange: options.gridView === 'month-count' ? [0, 20] : [4, 8],
              sqrtScale: options.gridView === 'month-count',
              discardAtOrBelow: options.gridView === 'month-count' ? 0 : 3.9
            })
          );
        }
      }
      if (options.showPairs) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'quake-pairs',
            coordinateOrigin,
            segments: current.knox.segments,
            weights: current.knox.fade,
            instanceCount: current.knox.slotCapacity,
            widthPixels: 1.1,
            colormap: 'uniform',
            color: dark ? [255, 200, 90, 200] : [200, 90, 10, 200]
          })
        );
      }
      layers.push(
        new QuakeEventLayer({
          id: 'quake-events',
          coordinateOrigin,
          positions: current.positions,
          times: current.days,
          magnitudes: current.magnitude,
          depths: current.depth,
          instanceCount: current.events.count,
          playhead: playheadDay,
          fadeDays: options.fadeDays,
          sizePixels: 2.5 * options.sizeScale,
          ghostAlpha: options.showPast ? 0.1 : 0,
          colorMode: options.colorBy,
          ramp: 'magma',
          depthMax: 300,
          outlineColor: dark ? [255, 255, 255, 90] : [20, 25, 40, 200]
        })
      );
      return layers;
    },

    destroy() {
      destroyed = true;
      for (const variant of variants.values()) {
        variant.knox.reader.stop();
        variant.monthly.reader.stop();
      }
      resources.destroy();
    }
  };
}
