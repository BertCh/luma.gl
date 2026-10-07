// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUCalendarBucketsParameterValues,
  getGPUTemporalReductionParameterValues,
  GPU_CALENDAR_BUCKETS_PARAMETER_LENGTH,
  GPU_TEMPORAL_REDUCTION_PARAMETER_LENGTH,
  GPUTemporalReduction
} from '@luma.gl/experimental/gpu-dataframe';
import {
  addSpaceTimeHotSpotsRecipe,
  getGPUEmergingHotSpotParameterValues,
  getGPUNeighborSearchParameterValues,
  GPU_EMERGING_HOT_SPOT_PARAMETER_LENGTH,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  GPUEmergingHotSpots,
  GPUNeighborSearch,
  type GPUSpatialWeights
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/gpgpu/gpu-data';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisRasterLayer} from '../../engine/layers';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {EMERGING_CATEGORY_NAMES, EmergingCategoryRasterLayer} from './b13-category-layer';
import {
  NATURE_YEAR_SECONDS,
  fillCategoryMask,
  readNatureEvents,
  type B13NatureEvents
} from './b13-nature-events';

/** Option state of the emerging-hot-spots scene. */
export type EmergingHotSpotsOptions = {
  groupType: string;
  cube: 'months' | 'weeks' | 'hours';
  neighborhood: 'lattice' | 'weights';
  radius: number;
  weightKind: 'binary' | 'inverse-distance' | 'kernel';
  temporalWindow: number;
  confidence: '0.9' | '0.95' | '0.99';
  tieTrend: boolean;
  trendLevel: number;
  persistentFraction: number;
  minimumEvents: number;
  mapView: 'category' | 'gi' | 'trend' | 'hot-count' | 'cold-count' | 'events';
  slice: number;
  play: boolean;
  showEvents: boolean;
  opacity: number;
};

/** Cell edge in meters. */
const CELL_METERS = 500;
const MAXIMUM_RADIUS_CELLS = 4;
const WEEK_SECONDS = 7 * 86400;
const WEEK_COUNT = 52;
const _NO_CELL = 0xffffffff;
const PLAY_SECONDS_PER_SLICE = 0.45;
const MONTH_NAMES = [
  'Jan',
  'Feb',
  'Mar',
  'Apr',
  'May',
  'Jun',
  'Jul',
  'Aug',
  'Sep',
  'Oct',
  'Nov',
  'Dec'
];

const CUBES = {
  months: {sliceCount: 12, field: 'month' as const, firstValue: 1},
  hours: {sliceCount: 24, field: 'hour' as const, firstValue: 0},
  weeks: {sliceCount: WEEK_COUNT, field: null, firstValue: 0}
};

/** Label of one slice of a cube. */
export function formatSliceLabel(cube: EmergingHotSpotsOptions['cube'], slice: number): string {
  if (cube === 'months') return `${MONTH_NAMES[slice] ?? '?'} 2023`;
  if (cube === 'hours') {
    return `${String(slice).padStart(2, '0')}:00 to ${String((slice + 1) % 24).padStart(2, '0')}:00`;
  }
  const start = new Date(Date.UTC(2023, 0, 1 + slice * 7));
  const end = new Date(Date.UTC(2023, 0, 7 + slice * 7));
  const format = (date: Date) => `${MONTH_NAMES[date.getUTCMonth()]} ${date.getUTCDate()}`;
  return `Week ${slice + 1}: ${format(start)} to ${format(end)}`;
}

type Variant = {
  compiled: CompiledGPUCommandGraph<void>;
  sliceCount: number;
  counts: Buffer;
  giZ: Buffer;
  trendZ: Buffer;
  trendP: Buffer;
  category: Buffer;
  hotSlices: Buffer;
  coldSlices: Buffer;
  overflow: Buffer;
  occupiedCount: Buffer | null;
  reader: SummaryReader;
};

/**
 * Emerging hot spot analysis of Chicago nature observations. One cube (cells x time slices) of event counts is
 * built on the GPU three ways: `addSpaceTimeHotSpotsRecipe` over calendar months or hours, or
 * `GPUTemporalReduction` over seven-day buckets. `GPUEmergingHotSpots` then runs space-time Gi*,
 * the Mann-Kendall trend and the 17-category classification, with the neighborhood either a
 * lattice radius or a `GPUNeighborSearch` weights table. Cube source and neighborhood are
 * compile-time, so each of the six combinations is compiled on first use and cached; every other
 * option is a parameter-buffer write.
 */
export async function createEmergingHotSpots(
  ctx: SceneContext<EmergingHotSpotsOptions>
): Promise<SceneInstance<EmergingHotSpotsOptions>> {
  const {device} = ctx;
  const events: B13NatureEvents = readNatureEvents(ctx.datasets.get('chicago-nature'));
  const origin = events.origin;
  const eventCount = events.count;

  // The lattice covers the events; cells are CELL_METERS squares anchored at the south-west corner.
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  for (let index = 0; index < eventCount; index++) {
    const x = events.positions[index * 2];
    const y = events.positions[index * 2 + 1];
    minimumX = Math.min(minimumX, x);
    maximumX = Math.max(maximumX, x);
    minimumY = Math.min(minimumY, y);
    maximumY = Math.max(maximumY, y);
  }
  const gridWidth = Math.ceil((maximumX - minimumX) / CELL_METERS + 1e-6);
  const gridHeight = Math.ceil((maximumY - minimumY) / CELL_METERS + 1e-6);
  const cellCount = gridWidth * gridHeight;
  const bounds: [number, number, number, number] = [
    minimumX,
    minimumY,
    minimumX + gridWidth * CELL_METERS,
    minimumY + gridHeight * CELL_METERS
  ];

  // Static per-event cell and per-cell activity (all groups of life), computed once.
  const cellIds = new Uint32Array(eventCount);
  const yearTotals = new Uint32Array(cellCount);
  for (let index = 0; index < eventCount; index++) {
    const column = Math.min(
      gridWidth - 1,
      Math.floor((events.positions[index * 2] - minimumX) / CELL_METERS)
    );
    const row = Math.min(
      gridHeight - 1,
      Math.floor((events.positions[index * 2 + 1] - minimumY) / CELL_METERS)
    );
    const cell = row * gridWidth + column;
    cellIds[index] = cell;
    yearTotals[cell]++;
  }
  const cellCenters = new Float32Array(cellCount * 2);
  for (let row = 0; row < gridHeight; row++) {
    for (let column = 0; column < gridWidth; column++) {
      const cell = row * gridWidth + column;
      cellCenters[cell * 2] = minimumX + (column + 0.5) * CELL_METERS;
      cellCenters[cell * 2 + 1] = minimumY + (row + 0.5) * CELL_METERS;
    }
  }

  const resources = new SpatialAnalysisResources(device, 'emerging');
  const positionsBuffer = resources.createBuffer('positions', events.positions);
  const timeWordsBuffer = resources.createBuffer('time-words', events.timeWords as Uint32Array);
  const secondsBuffer = resources.createBuffer('seconds', events.seconds);
  const onesBuffer = resources.createBuffer('ones', new Float32Array(eventCount).fill(1));
  const cellIdsBuffer = resources.createBuffer('cell-ids', cellIds);
  const cellCentersBuffer = resources.createBuffer('cell-centers', cellCenters);
  const eventMaskBuffer = resources.createBuffer('event-mask', eventCount * 4);
  const cellMaskBuffer = resources.createBuffer('cell-mask', cellCount * 4);
  const sliceIndexBuffer = resources.createBuffer('slice-index', cellCount * 4);
  const maximumNeighbors = Math.ceil(Math.PI * MAXIMUM_RADIUS_CELLS ** 2) + 8;
  const slotCapacity = cellCount * maximumNeighbors;
  const offsetsBuffer = resources.createBuffer('offsets', (cellCount + 1) * 4);
  const neighborsBuffer = resources.createBuffer('neighbors', slotCapacity * 4);
  const weightsBuffer = resources.createBuffer('weights', slotCapacity * 4);
  const distancesBuffer = resources.createBuffer('distances', slotCapacity * 4);

  const calendarParameters = resources.createParameterBuffer(
    'calendar-parameters',
    'sint32',
    GPU_CALENDAR_BUCKETS_PARAMETER_LENGTH,
    getGPUCalendarBucketsParameterValues(0, 0)
  );
  const reductionParameters = resources.createParameterBuffer(
    'reduction-parameters',
    'float32',
    GPU_TEMPORAL_REDUCTION_PARAMETER_LENGTH,
    getGPUTemporalReductionParameterValues(0, WEEK_SECONDS)
  );
  const hotSpotParameters = resources.createParameterBuffer(
    'hot-spot-parameters',
    'float32',
    GPU_EMERGING_HOT_SPOT_PARAMETER_LENGTH
  );
  const searchParameters = resources.createParameterBuffer(
    'search-parameters',
    'float32',
    GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH
  );

  let destroyed = false;
  let dirty = true;
  let shownSlice = 0;
  let lastAdvance = 0;
  let activeCells = 0;
  let included = 0;
  let current: Variant | null = null;
  let cpuCategory: Uint32Array | null = null;
  let cpuTrendZ: Float32Array | null = null;
  let cpuHot: Uint32Array | null = null;
  let cpuCold: Uint32Array | null = null;
  let cpuCounts: Uint32Array | null = null;
  let countMaximum = 1;
  const variants = new Map<string, Variant>();

  const view = <Format extends GPUVectorFormat>(
    graph: GPUCommandGraph<void>,
    name: string,
    buffer: Buffer,
    format: Format,
    length?: number
  ) => importGraphBuffer(graph, name, buffer, format, length);

  function buildVariant(
    cube: EmergingHotSpotsOptions['cube'],
    neighborhood: EmergingHotSpotsOptions['neighborhood']
  ): Variant {
    const spec = CUBES[cube];
    const sliceCount = spec.sliceCount;
    const binCount = cellCount * sliceCount;
    const id = `${cube}-${neighborhood}`;
    const buffer = (name: string, bytes: number) => resources.createBuffer(`${id}-${name}`, bytes);
    const counts = buffer('counts', binCount * 4);
    const giZ = buffer('gi-z', binCount * 4);
    const trendZ = buffer('trend-z', cellCount * 4);
    const trendP = buffer('trend-p', cellCount * 4);
    const trendS = buffer('trend-s', cellCount * 4);
    const category = buffer('category', cellCount * 4);
    const hotSlices = buffer('hot-slices', cellCount * 4);
    const coldSlices = buffer('cold-slices', cellCount * 4);
    const overflow = buffer('overflow', 4);
    const graph = new GPUCommandGraph<void>(device, {id: `emerging-${id}`});
    // A buffer is imported once per graph; later lookups return the same view.
    const imported = new Map<string, unknown>();
    const v = <Format extends GPUVectorFormat>(
      name: string,
      source: Buffer,
      format: Format,
      length?: number
    ) => {
      let existing = imported.get(name);
      if (!existing) {
        existing = view(graph, `${id}-${name}`, source, format, length);
        imported.set(name, existing);
      }
      return existing as ReturnType<typeof view<Format>>;
    };

    // Spatial neighborhood: a lattice radius, or a neighbor-search weights table over cell centers.
    let weights: GPUSpatialWeights | undefined;
    if (neighborhood === 'weights') {
      weights = {
        offsets: v('offsets', offsetsBuffer, 'uint32', cellCount + 1),
        neighbors: v('neighbors', neighborsBuffer, 'uint32', slotCapacity),
        weights: v('weights', weightsBuffer, 'float32', slotCapacity),
        distances: v('distances', distancesBuffer, 'float32', slotCapacity)
      };
      graph.add(
        new GPUNeighborSearch({
          id: `${id}-search`,
          mode: 'radius',
          positions: v('cell-centers', cellCentersBuffer, 'float32x2', cellCount),
          mask: v('cell-mask', cellMaskBuffer, 'uint32', cellCount),
          parameters: searchParameters.importToGraph(graph),
          gridSize: [64, 64],
          weights,
          overflow: v('overflow', overflow, 'uint32', 1)
        })
      );
    }

    let occupiedCount: Buffer | null = null;
    if (cube === 'weeks') {
      const occupiedIds = buffer('occupied-ids', binCount * 4);
      occupiedCount = buffer('occupied-count', 4);
      const occupiedOverflow = buffer('occupied-overflow', 4);
      graph.add(
        new GPUTemporalReduction({
          id: `${id}-reduction`,
          cellIds: v('cell-ids', cellIdsBuffer, 'uint32', eventCount),
          timestamps: v('seconds', secondsBuffer, 'float32', eventCount),
          values: v('ones', onesBuffer, 'float32', eventCount),
          mask: v('event-mask', eventMaskBuffer, 'uint32', eventCount),
          parameters: reductionParameters.importToGraph(graph),
          cellCount,
          bucketCount: sliceCount,
          output: {
            counts: v('counts', counts, 'uint32', binCount),
            min: v('min', buffer('min', binCount * 4), 'float32', binCount),
            max: v('max', buffer('max', binCount * 4), 'float32', binCount),
            first: v('first', buffer('first', binCount * 4), 'float32', binCount),
            last: v('last', buffer('last', binCount * 4), 'float32', binCount),
            occupiedSlots: {
              ids: v('occupied-ids', occupiedIds, 'uint32', binCount),
              count: v('occupied-count', occupiedCount, 'uint32', 1),
              overflow: v('occupied-overflow', occupiedOverflow, 'uint32', 1)
            }
          }
        })
      );
      graph.add(
        new GPUEmergingHotSpots({
          id: `${id}-emerging`,
          values: v('counts', counts, 'uint32', binCount),
          ...(weights ? {weights} : {gridWidth, gridHeight, maximumRadius: MAXIMUM_RADIUS_CELLS}),
          sliceCount,
          mask: v('cell-mask', cellMaskBuffer, 'uint32', cellCount),
          parameters: hotSpotParameters.importToGraph(graph),
          giZScores: v('gi-z', giZ, 'float32', binCount),
          trendZ: v('trend-z', trendZ, 'float32', cellCount),
          trendP: v('trend-p', trendP, 'float32', cellCount),
          trendS: v('trend-s', trendS, 'sint32', cellCount),
          category: v('category', category, 'uint32', cellCount),
          hotSliceCount: v('hot-slices', hotSlices, 'uint32', cellCount),
          coldSliceCount: v('cold-slices', coldSlices, 'uint32', cellCount)
        })
      );
    } else {
      addSpaceTimeHotSpotsRecipe(graph, {
        id: `${id}-recipe`,
        timestamps: v('time-words', timeWordsBuffer, 'uint32x2', eventCount),
        mask: v('event-mask', eventMaskBuffer, 'uint32', eventCount),
        calendarParameters: calendarParameters.importToGraph(graph),
        slices: {
          field: spec.field!,
          firstValue: spec.firstValue,
          count: sliceCount
        },
        cells: weights
          ? {
              kind: 'ids',
              cellIds: v('cell-ids', cellIdsBuffer, 'uint32', eventCount),
              cellCount,
              weights,
              selfWeight: 1
            }
          : {
              kind: 'lattice',
              positions: v('positions', positionsBuffer, 'float32x2', eventCount),
              width: gridWidth,
              height: gridHeight,
              bounds,
              maximumRadius: MAXIMUM_RADIUS_CELLS
            },
        cellMask: v('cell-mask', cellMaskBuffer, 'uint32', cellCount),
        parameters: hotSpotParameters.importToGraph(graph),
        cube: v('counts', counts, 'uint32', binCount),
        giZScores: v('gi-z', giZ, 'float32', binCount),
        trendZ: v('trend-z', trendZ, 'float32', cellCount),
        trendP: v('trend-p', trendP, 'float32', cellCount),
        trendS: v('trend-s', trendS, 'sint32', cellCount),
        category: v('category', category, 'uint32', cellCount),
        hotSliceCount: v('hot-slices', hotSlices, 'uint32', cellCount),
        coldSliceCount: v('cold-slices', coldSlices, 'uint32', cellCount)
      });
    }
    const compiled = resources.track(graph.compile());
    const reader = new SummaryReader(
      resources,
      id,
      [
        {buffer: category, size: cellCount * 4},
        {buffer: trendZ, size: cellCount * 4},
        {buffer: hotSlices, size: cellCount * 4},
        {buffer: coldSlices, size: cellCount * 4},
        {buffer: counts, size: binCount * 4}
      ],
      bytes => {
        if (destroyed || current?.counts !== counts) return;
        const cells = cellCount * 4;
        cpuCategory = new Uint32Array(bytes, 0, cellCount);
        cpuTrendZ = new Float32Array(bytes, cells, cellCount);
        cpuHot = new Uint32Array(bytes, cells * 2, cellCount);
        cpuCold = new Uint32Array(bytes, cells * 3, cellCount);
        cpuCounts = new Uint32Array(bytes, cells * 4, binCount);
        showSummary();
      }
    );
    return {
      compiled,
      sliceCount,
      counts,
      giZ,
      trendZ,
      trendP,
      category,
      hotSlices,
      coldSlices,
      overflow,
      occupiedCount,
      reader
    };
  }

  function getVariant(): Variant {
    const {cube, neighborhood} = ctx.options;
    const key = `${cube}-${neighborhood}`;
    let variant = variants.get(key);
    if (!variant) {
      variant = buildVariant(cube, neighborhood);
      variants.set(key, variant);
    }
    return variant;
  }

  function writeEventMask(): void {
    const mask = new Uint32Array(eventCount);
    included = fillCategoryMask(events, ctx.options.groupType, mask);
    eventMaskBuffer.write(mask);
    ctx.setReadout('events', `${formatCount(included)} of ${formatCount(eventCount)}`);
    dirty = true;
  }

  function writeCellMask(): void {
    const mask = new Uint32Array(cellCount);
    activeCells = 0;
    for (let cell = 0; cell < cellCount; cell++) {
      const active = yearTotals[cell] >= ctx.options.minimumEvents ? 1 : 0;
      mask[cell] = active;
      activeCells += active;
    }
    cellMaskBuffer.write(mask);
    ctx.setReadout('cells', `${formatCount(activeCells)} of ${formatCount(cellCount)}`);
    dirty = true;
  }

  function writeParameters(): void {
    const options = ctx.options;
    const confidenceLevel = Number(options.confidence) as 0.9 | 0.95 | 0.99;
    hotSpotParameters.write(
      getGPUEmergingHotSpotParameterValues({
        radius: options.radius,
        temporalWindow: options.temporalWindow,
        confidenceLevel,
        trendSignificanceLevel: options.tieTrend ? 1 - confidenceLevel : options.trendLevel,
        persistentFraction: options.persistentFraction
      })
    );
    searchParameters.write(
      getGPUNeighborSearchParameterValues({
        bounds: [
          bounds[0] - CELL_METERS,
          bounds[1] - CELL_METERS,
          bounds[2] + CELL_METERS,
          bounds[3] + CELL_METERS
        ],
        radius: options.radius * CELL_METERS,
        weightKind:
          options.weightKind === 'inverse-distance'
            ? 'inverseDistance'
            : options.weightKind === 'kernel'
              ? 'kernel'
              : 'binary',
        power: 1,
        distanceFloor: CELL_METERS / 2,
        kernel: 'triangular'
      })
    );
    dirty = true;
  }

  function writeSliceIndices(slice: number): void {
    const sliceCount = current?.sliceCount ?? 1;
    const clamped = Math.min(slice, sliceCount - 1);
    const indices = new Uint32Array(cellCount);
    for (let cell = 0; cell < cellCount; cell++) indices[cell] = cell * sliceCount + clamped;
    sliceIndexBuffer.write(indices);
    shownSlice = clamped;
    ctx.setReadout('sliceLabel', formatSliceLabel(ctx.options.cube, clamped));
  }

  function showSummary(): void {
    if (!cpuCategory || !cpuCounts || !current) return;
    const categoryCounts = new Array<number>(17).fill(0);
    for (let cell = 0; cell < cellCount; cell++) {
      if (cpuCategory[cell] < 17) categoryCounts[cpuCategory[cell]]++;
    }
    const sum = (from: number, to: number) =>
      categoryCounts.slice(from, to + 1).reduce((total, value) => total + value, 0);
    ctx.setReadout('hot', sum(1, 8));
    ctx.setReadout('cold', sum(9, 16));
    ctx.setReadout('new', categoryCounts[1]);
    ctx.setReadout('intensifying', categoryCounts[3]);
    ctx.setReadout('persistent', categoryCounts[4]);
    ctx.setReadout('diminishing', categoryCounts[5]);
    ctx.setReadout('sporadic', categoryCounts[6]);
    const slices = current.sliceCount;
    const perSlice = new Array<number>(slices).fill(0);
    let total = 0;
    countMaximum = 1;
    for (let bin = 0; bin < cpuCounts.length; bin++) {
      perSlice[bin % slices] += cpuCounts[bin];
      total += cpuCounts[bin];
      countMaximum = Math.max(countMaximum, cpuCounts[bin]);
    }
    ctx.setReadout(
      'cubeEvents',
      `${formatCount(total)} observations in ${formatCount(cpuCounts.length)} bins`
    );
    let busiest = 0;
    for (let slice = 1; slice < slices; slice++)
      if (perSlice[slice] > perSlice[busiest]) busiest = slice;
    ctx.setReadout(
      'busiest',
      `${formatSliceLabel(ctx.options.cube, busiest)} (${formatCount(perSlice[busiest])})`
    );
    ctx.setLegendExtent('count', [0, countMaximum]);
    ctx.requestLayers();
  }

  writeEventMask();
  writeCellMask();
  writeParameters();
  current = getVariant();
  writeSliceIndices(ctx.options.slice - 1);
  ctx.setReadout('grid', `${gridWidth} x ${gridHeight} cells of ${CELL_METERS} m`);

  return {
    getCompiledGraphs: () => (current ? [current.compiled as CompiledGPUCommandGraph<never>] : []),

    setOption(id, _value, state) {
      if (id === 'cube' || id === 'neighborhood') {
        current = getVariant();
        writeSliceIndices(state.slice - 1);
        cpuCategory = null;
        dirty = true;
        ctx.requestLayers();
      } else if (id === 'groupType') {
        writeEventMask();
      } else if (id === 'minimumEvents') {
        writeCellMask();
      } else if (id === 'slice') {
        writeSliceIndices(state.slice - 1);
      } else if (
        id === 'radius' ||
        id === 'weightKind' ||
        id === 'temporalWindow' ||
        id === 'confidence' ||
        id === 'tieTrend' ||
        id === 'trendLevel' ||
        id === 'persistentFraction'
      ) {
        writeParameters();
      } else {
        ctx.requestLayers();
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip({coordinate}) {
      if (!coordinate || !cpuCategory || !cpuCounts || !current) return null;
      const projection = ctx.datasets.get('chicago-nature').getProjection(origin);
      const [x, y] = projection.project(coordinate[0], coordinate[1]);
      const column = Math.floor((x - minimumX) / CELL_METERS);
      const row = Math.floor((y - minimumY) / CELL_METERS);
      if (column < 0 || row < 0 || column >= gridWidth || row >= gridHeight) return null;
      const cell = row * gridWidth + column;
      if (!activeCellsMask(cell)) return null;
      const slices = current.sliceCount;
      let total = 0;
      for (let slice = 0; slice < slices; slice++) total += cpuCounts[cell * slices + slice];
      const here = cpuCounts[cell * slices + shownSlice];
      return [
        EMERGING_CATEGORY_NAMES[cpuCategory[cell]] ?? 'Unknown',
        `${formatCount(total)} observations in 2023, ${formatCount(here)} in ${formatSliceLabel(ctx.options.cube, shownSlice)}`,
        `Hot in ${cpuHot?.[cell] ?? 0} and cold in ${cpuCold?.[cell] ?? 0} of ${slices} slices`,
        `Mann-Kendall trend z = ${(cpuTrendZ?.[cell] ?? 0).toFixed(2)}`
      ].join('\n');
    },

    encode(commandEncoder, frame) {
      if (!current) return;
      const options = ctx.options;
      if (options.play) {
        if (frame.timeSeconds - lastAdvance > PLAY_SECONDS_PER_SLICE) {
          lastAdvance = frame.timeSeconds;
          writeSliceIndices((shownSlice + 1) % current.sliceCount);
        }
      } else if (shownSlice !== Math.min(options.slice - 1, current.sliceCount - 1)) {
        writeSliceIndices(options.slice - 1);
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
      const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      if (options.showEvents) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'emerging-events',
            coordinateOrigin,
            positions: positionsBuffer,
            instanceCount: eventCount,
            values: eventMaskBuffer,
            valueFormat: 'uint32',
            colormap: 'mask',
            color: dark ? [190, 215, 255, 70] : [40, 60, 110, 70],
            noDataColor: [0, 0, 0, 0],
            radiusPixels: 1
          })
        );
      }
      const common = {
        coordinateOrigin,
        gridSize: [gridWidth, gridHeight] as const,
        bounds,
        opacity: options.opacity
      };
      switch (options.mapView) {
        case 'category':
          layers.push(
            new EmergingCategoryRasterLayer({
              ...common,
              id: 'emerging-category',
              values: current.category,
              valueFormat: 'uint32',
              colormap: 'category',
              color: [255, 255, 255, 255]
            })
          );
          break;
        case 'gi':
          layers.push(
            new SpatialAnalysisRasterLayer({
              ...common,
              id: 'emerging-gi',
              values: current.giZ,
              valueIndices: sliceIndexBuffer,
              valueFormat: 'float32',
              colormap: 'diverging',
              valueRange: [-4, 4],
              color: [255, 255, 255, 255]
            })
          );
          break;
        case 'trend':
          layers.push(
            new SpatialAnalysisRasterLayer({
              ...common,
              id: 'emerging-trend',
              values: current.trendZ,
              valueFormat: 'float32',
              colormap: 'diverging',
              valueRange: [-4, 4],
              color: [255, 255, 255, 255]
            })
          );
          break;
        case 'hot-count':
        case 'cold-count':
          layers.push(
            new SpatialAnalysisRasterLayer({
              ...common,
              id: `emerging-${options.mapView}`,
              values: options.mapView === 'hot-count' ? current.hotSlices : current.coldSlices,
              valueFormat: 'uint32',
              colormap: options.mapView === 'hot-count' ? 'inferno' : 'viridis',
              valueRange: [0, current.sliceCount],
              discardAtOrBelow: 0,
              color: [255, 255, 255, 255]
            })
          );
          break;
        case 'events':
          layers.push(
            new SpatialAnalysisRasterLayer({
              ...common,
              id: 'emerging-events-cube',
              values: current.counts,
              valueIndices: sliceIndexBuffer,
              valueFormat: 'uint32',
              colormap: 'magma',
              valueRange: [0, countMaximum],
              sqrtScale: true,
              discardAtOrBelow: 0,
              color: [255, 255, 255, 255]
            })
          );
          break;
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      for (const variant of variants.values()) variant.reader.stop();
      resources.destroy();
    }
  };

  function activeCellsMask(cell: number): boolean {
    return yearTotals[cell] >= ctx.options.minimumEvents;
  }
}

// Exported for the scene's code snippet text.
export const EMERGING_CUBE_SLICE_COUNTS = {
  months: CUBES.months.sliceCount,
  weeks: CUBES.weeks.sliceCount,
  hours: CUBES.hours.sliceCount
};
export const EMERGING_YEAR_SECONDS = NATURE_YEAR_SECONDS;
