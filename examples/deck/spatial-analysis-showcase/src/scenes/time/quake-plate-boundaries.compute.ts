// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {GPUGroupStatistics} from '@luma.gl/experimental/gpu-dataframe';
import {
  getGPUDistanceFieldParameterValues,
  GPU_DISTANCE_FIELD_PARAMETER_LENGTH,
  GPUDistanceField
} from '@luma.gl/experimental/gpu-raster';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {SpatialAnalysisRasterLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {ChartSeries} from '../chart-types';
import type {SceneContext, SceneInstance} from '../scene';
import {createViewImporter} from './b13-views';
import {
  BOUNDARY_FAMILY_OF_CLASS,
  QUAKE_DEPTH_CLASS_COLORS,
  QUAKE_DEPTH_CLASS_NAMES,
  QUAKE_REGIONS,
  type QuakeRegionId
} from './quake-regions';
import {
  densifyBoundarySeeds,
  fitGutenbergRichter,
  getHistogramMedian,
  readBoundarySteps,
  readQuakeEvents,
  type QuakeBoundarySteps,
  type QuakeEvents
} from './quake-data';
import {QuakeEventLayer} from './quake-event-layer';

/** Option state of the quake-plate-boundaries scene. */
export type QuakePlateBoundariesOptions = {
  region: QuakeRegionId;
  boundaryClasses: 'all' | 'convergent' | 'subduction' | 'divergent' | 'transform';
  algorithm: 'exact' | 'jump-flood-0' | 'jump-flood-1' | 'jump-flood-2';
  maximumDistance: number;
  mapShows: 'distance' | 'allocation' | 'events';
  mapOpacity: number;
  minimumMagnitude: number;
  binWidth: '5' | '10' | '20' | '25';
  shallowLimit: number;
  deepLimit: number;
  lowerFraction: number;
  upperFraction: number;
  variance: 'sample' | 'population';
  eventColor: 'class' | 'depth';
  sizeScale: number;
  showLinks: boolean;
  linkMinimumDepth: number;
  grSubset: 'all' | 'near' | 'far';
  grDistance: number;
  completeness: number;
};

/** Family colors of boundaries and nearest-boundary zones: convergent, divergent, transform. */
export const BOUNDARY_FAMILY_COLORS = [
  [230, 57, 70],
  [66, 135, 245],
  [240, 200, 40]
] as const;

const GRID = 1024;
const CELL_COUNT = GRID * GRID;
const SEED_CAPACITY = 20000;
const BIN_COUNT = 64;
const CLASS_COUNT = 3;
const MAGNITUDE_BIN_COUNT = 50;
const MAGNITUDE_START = 4;
const PERCENTILE_COUNT = 3;
const NO_KEY = 0xffffffff;
const GRID_PADDING_METERS = 450000;
const NO_DISTANCE_CAP = 2000;

function formatNumber(value: number, digits = 2): string {
  return Number.isFinite(value) ? value.toFixed(digits) : 'n/a';
}

type Grid = {origin: [number, number]; cellSize: number};

type DistanceVariant = {
  compiled: CompiledGPUCommandGraph<void>;
};

type RegionVariant = {
  key: string;
  region: QuakeRegionId;
  events: QuakeEvents;
  steps: QuakeBoundarySteps;
  grid: Grid;
  positions: Buffer;
  depth: Buffer;
  magnitude: Buffer;
  distances: Buffer;
  allocation: Buffer;
  nearestCells: Buffer;
  distanceKm: Buffer;
  keyDistance: Buffer;
  keyClass: Buffer;
  keyMagnitude: Buffer;
  maskMagnitude: Buffer;
  stepSegments: Buffer;
  stepFamily: Buffer;
  selectedSteps: Buffer;
  selectedStepCount: number;
  seedPositions: Buffer;
  seedIds: Buffer;
  seedCount: Buffer;
  seedInfo: string;
  linkSegments: Buffer;
  linkFade: Buffer;
  distanceVariants: Map<string, DistanceVariant>;
  statistics: CompiledGPUCommandGraph<void>;
  reader: SummaryReader;
};

/**
 * Distance to the nearest plate boundary and what it explains, for one region of the 2020 to 2024
 * catalog. `GPUDistanceField` computes the exact (or jump-flooded) distance and nearest-boundary
 * class on a 1024 by 1024 grid from seed points along the PB2002 steps; a kernel samples it at
 * every event; three `GPUGroupStatistics` tables then give depth statistics per distance bin,
 * event counts per (depth class, distance bin) and counts per magnitude bin for the
 * Gutenberg-Richter fit. Seeds, distance cap, bin width, depth limits, percentiles and the
 * magnitude cut are parameter writes; region, variance and the distance algorithm compile.
 */
export async function createQuakePlateBoundaries(
  ctx: SceneContext<QuakePlateBoundariesOptions>
): Promise<SceneInstance<QuakePlateBoundariesOptions>> {
  const {device} = ctx;
  const catalog = ctx.datasets.get('poopdeck-earthquakes');
  const boundaries = ctx.datasets.get('plate-boundaries');
  const resources = new SpatialAnalysisResources(device, 'quake-plates');

  const distanceSettings = resources.createParameterBuffer(
    'distance-settings',
    'float32',
    GPU_DISTANCE_FIELD_PARAMETER_LENGTH
  );
  // [binKm, minimumMagnitude, grSubset, grDistanceKm, shallowLimit, deepLimit, linkEnabled,
  //  linkMinimumDepth, originX, originY, 1 / cellSize, cellSize, width, height, 0, 0]
  const eventParameters = resources.createParameterBuffer('event-parameters', 'float32', 16);
  const percentileParameters = resources.createParameterBuffer(
    'percentile-parameters',
    'float32',
    PERCENTILE_COUNT
  );

  let destroyed = false;
  let dirty = true;
  let current: RegionVariant | null = null;
  const regionVariants = new Map<string, RegionVariant>();
  // Last statistics read back, kept so the completeness slider can refit without a GPU pass.
  let magnitudeCounts: Uint32Array | null = null;

  function getGrid(events: QuakeEvents): Grid {
    const [west, south, east, north] = events.windowMeters;
    const extent = Math.max(east - west, north - south) + 2 * GRID_PADDING_METERS;
    return {
      origin: [(west + east) / 2 - extent / 2, (south + north) / 2 - extent / 2],
      cellSize: extent / GRID
    };
  }

  function buildRegion(
    key: string,
    region: QuakeRegionId,
    variance: 'sample' | 'population'
  ): RegionVariant {
    // The whole M4+ catalog of the window; the magnitude cut is a parameter.
    const events = readQuakeEvents(catalog, region, 4);
    const steps = readBoundarySteps(boundaries, events.projection, region, 7);
    const grid = getGrid(events);
    const count = events.count;
    const buffer = (name: string, data: Float32Array | Uint32Array | number) =>
      resources.createBuffer(`${key}-${name}`, data);
    const positions = buffer('positions', events.positions);
    const depth = buffer('depth', events.depth);
    const magnitude = buffer('magnitude', events.magnitude);
    const distances = buffer('distances', CELL_COUNT * 4);
    const allocation = buffer('allocation', CELL_COUNT * 4);
    const nearestCells = buffer('nearest-cells', CELL_COUNT * 4);
    const distanceKm = buffer('distance-km', count * 4);
    const keyDistance = buffer('key-distance', count * 4);
    const keyClass = buffer('key-class', count * 4);
    const keyMagnitude = buffer('key-magnitude', count * 4);
    const maskMagnitude = buffer('mask-magnitude', count * 4);
    const seedPositions = buffer('seed-positions', SEED_CAPACITY * 8);
    const seedIds = buffer('seed-ids', SEED_CAPACITY * 4);
    const seedCount = buffer('seed-count', 4);
    const linkSegments = buffer('link-segments', count * 16);
    const linkFade = buffer('link-fade', count * 4);
    const stepSegments = buffer('step-segments', steps.segments);
    const stepFamily = buffer(
      'step-family',
      Uint32Array.from(steps.stepClass, code => BOUNDARY_FAMILY_OF_CLASS[code])
    );
    const selectedSteps = buffer('selected-steps', Math.max(4, steps.count * 4));

    // ---- Statistics graph: three dense group tables over the event keys ---------------------
    const statisticsGraph = new GPUCommandGraph<void>(device, {id: `quake-statistics-${key}`});
    const s = createViewImporter(statisticsGraph, `${key}-statistics`);
    const mask = s('mask-magnitude', maskMagnitude, 'uint32', count);
    const table = (name: string, rows: number) => {
      const countsBuffer = buffer(`${name}-counts`, rows * 4);
      return {
        countsBuffer,
        views: {
          keys: s(`${name}-keys`, buffer(`${name}-keys`, rows * 4), 'uint32', rows),
          counts: s(`${name}-counts`, countsBuffer, 'uint32', rows),
          count: s(`${name}-count`, buffer(`${name}-count`, 4), 'uint32', 1),
          overflow: s(`${name}-overflow`, buffer(`${name}-overflow`, 4), 'uint32', 1)
        }
      };
    };
    const distanceTable = table('distance', BIN_COUNT);
    const classTable = table('class', CLASS_COUNT * BIN_COUNT);
    const magnitudeTable = table('magnitude-table', MAGNITUDE_BIN_COUNT);
    const depthMeans = buffer('depth-means', BIN_COUNT * 4);
    const depthDeviations = buffer('depth-deviations', BIN_COUNT * 4);
    const depthPercentiles = buffer('depth-percentiles', BIN_COUNT * PERCENTILE_COUNT * 4);
    statisticsGraph.add(
      new GPUGroupStatistics({
        id: `${key}-depth-by-distance`,
        keys: s('key-distance', keyDistance, 'uint32', count),
        mask,
        keyCount: BIN_COUNT,
        variance,
        percentiles: percentileParameters.importToGraph(statisticsGraph),
        columns: [
          {
            values: s('depth', depth, 'float32', count),
            statistics: ['mean', 'standardDeviation', 'percentiles'],
            output: {
              means: s('depth-means', depthMeans, 'float32', BIN_COUNT),
              standardDeviations: s('depth-deviations', depthDeviations, 'float32', BIN_COUNT),
              percentiles: s(
                'depth-percentiles',
                depthPercentiles,
                'float32',
                BIN_COUNT * PERCENTILE_COUNT
              )
            }
          }
        ],
        output: distanceTable.views
      })
    );
    statisticsGraph.add(
      new GPUGroupStatistics({
        id: `${key}-class-by-distance`,
        keys: s('key-class', keyClass, 'uint32', count),
        mask,
        keyCount: CLASS_COUNT * BIN_COUNT,
        columns: [],
        output: classTable.views
      })
    );
    statisticsGraph.add(
      new GPUGroupStatistics({
        id: `${key}-magnitude-counts`,
        keys: s('key-magnitude', keyMagnitude, 'uint32', count),
        keyCount: MAGNITUDE_BIN_COUNT,
        columns: [],
        output: magnitudeTable.views
      })
    );
    const statistics = resources.track(statisticsGraph.compile());
    const reader = new SummaryReader(
      resources,
      `${key}-statistics`,
      [
        {buffer: distanceTable.countsBuffer, size: BIN_COUNT * 4},
        {buffer: depthMeans, size: BIN_COUNT * 4},
        {buffer: depthDeviations, size: BIN_COUNT * 4},
        {buffer: depthPercentiles, size: BIN_COUNT * PERCENTILE_COUNT * 4},
        {buffer: classTable.countsBuffer, size: CLASS_COUNT * BIN_COUNT * 4},
        {buffer: magnitudeTable.countsBuffer, size: MAGNITUDE_BIN_COUNT * 4}
      ],
      bytes => {
        if (destroyed || current?.key !== key) return;
        showStatistics(bytes);
      }
    );
    return {
      key,
      region,
      events,
      steps,
      grid,
      positions,
      depth,
      magnitude,
      distances,
      allocation,
      nearestCells,
      distanceKm,
      keyDistance,
      keyClass,
      keyMagnitude,
      maskMagnitude,
      stepSegments,
      stepFamily,
      selectedSteps,
      selectedStepCount: 0,
      seedPositions,
      seedIds,
      seedCount,
      seedInfo: '',
      linkSegments,
      linkFade,
      distanceVariants: new Map(),
      statistics,
      reader
    };
  }

  function buildDistanceVariant(variant: RegionVariant, algorithm: string): DistanceVariant {
    const {events} = variant;
    const count = events.count;
    const graph = new GPUCommandGraph<void>(device, {
      id: `quake-distance-${variant.key}-${algorithm}`
    });
    const d = createViewImporter(graph, `${variant.key}-${algorithm}`);
    const settingsView = distanceSettings.importToGraph(graph);
    const parametersView = eventParameters.importToGraph(graph);
    const distancesView = d('distances', variant.distances, 'float32', CELL_COUNT);
    const nearestView = d('nearest-cells', variant.nearestCells, 'uint32', CELL_COUNT);
    const mode = algorithm === 'exact' ? 'exact' : 'jump-flood';
    graph.add(
      new GPUDistanceField({
        id: `${variant.key}-${algorithm}-field`,
        width: GRID,
        height: GRID,
        mode,
        jumpFloodRefinementPasses:
          algorithm === 'jump-flood-0' ? 0 : algorithm === 'jump-flood-2' ? 2 : 1,
        settings: settingsView,
        seedPositions: d('seed-positions', variant.seedPositions, 'float32x2', SEED_CAPACITY),
        seedIds: d('seed-ids', variant.seedIds, 'uint32', SEED_CAPACITY),
        seedCount: d('seed-count', variant.seedCount, 'uint32', 1),
        output: {
          distances: distancesView,
          allocation: d('allocation', variant.allocation, 'uint32', CELL_COUNT),
          nearestCells: nearestView
        }
      })
    );
    const positionsView = d('positions', variant.positions, 'float32x2', count);
    const distanceKmView = d('distance-km', variant.distanceKm, 'float32', count);
    const depthView = d('depth', variant.depth, 'float32', count);
    const magnitudeView = d('magnitude', variant.magnitude, 'float32', count);
    // Sample the distance grid at every event: kilometers, or -1 outside the grid or the cap.
    addKernelPass(graph, {
      id: `${variant.key}-event-distance`,
      invocationCount: count,
      bindings: [
        {name: 'positions', view: positionsView, type: 'f32', access: 'read'},
        {name: 'parameters', view: parametersView, type: 'f32', access: 'read'},
        {name: 'distances', view: distancesView, type: 'f32', access: 'read'},
        {name: 'distanceKm', view: distanceKmView, type: 'f32', access: 'read_write'}
      ],
      body: /* wgsl */ `
  let cellX = floor((positions[positionsOffset + index * 2u] - parameters[parametersOffset + 8u]) * parameters[parametersOffset + 10u]);
  let cellY = floor((positions[positionsOffset + index * 2u + 1u] - parameters[parametersOffset + 9u]) * parameters[parametersOffset + 10u]);
  var kilometers = -1.0;
  if (cellX >= 0.0 && cellY >= 0.0 && cellX < parameters[parametersOffset + 12u] && cellY < parameters[parametersOffset + 13u]) {
    let meters = distances[distancesOffset + u32(cellY) * u32(parameters[parametersOffset + 12u]) + u32(cellX)];
    if (meters < 3.0e38) {
      kilometers = meters * 0.001;
    }
  }
  distanceKm[distanceKmOffset + index] = kilometers;`
    });
    // Group keys from the distance: bin, (depth class, bin), magnitude bin for the chosen subset.
    addKernelPass(graph, {
      id: `${variant.key}-event-keys`,
      invocationCount: count,
      declarations: `const NO_KEY: u32 = ${NO_KEY}u; const BINS: u32 = ${BIN_COUNT}u;`,
      bindings: [
        {name: 'distanceKm', view: distanceKmView, type: 'f32', access: 'read'},
        {name: 'depth', view: depthView, type: 'f32', access: 'read'},
        {name: 'magnitude', view: magnitudeView, type: 'f32', access: 'read'},
        {name: 'parameters', view: parametersView, type: 'f32', access: 'read'},
        {
          name: 'keyDistance',
          view: d('key-distance', variant.keyDistance, 'uint32', count),
          type: 'u32',
          access: 'read_write'
        },
        {
          name: 'keyClass',
          view: d('key-class', variant.keyClass, 'uint32', count),
          type: 'u32',
          access: 'read_write'
        },
        {
          name: 'keyMagnitude',
          view: d('key-magnitude', variant.keyMagnitude, 'uint32', count),
          type: 'u32',
          access: 'read_write'
        },
        {
          name: 'maskMagnitude',
          view: d('mask-magnitude', variant.maskMagnitude, 'uint32', count),
          type: 'u32',
          access: 'read_write'
        }
      ],
      body: /* wgsl */ `
  let kilometers = distanceKm[distanceKmOffset + index];
  let eventDepth = depth[depthOffset + index];
  let eventMagnitude = magnitude[magnitudeOffset + index];
  var distanceKey = NO_KEY;
  var classKey = NO_KEY;
  if (kilometers >= 0.0) {
    let bin = u32(floor(kilometers / parameters[parametersOffset]));
    if (bin < BINS) {
      distanceKey = bin;
      var depthClass = 0u;
      if (eventDepth >= parameters[parametersOffset + 5u]) {
        depthClass = 2u;
      } else if (eventDepth >= parameters[parametersOffset + 4u]) {
        depthClass = 1u;
      }
      classKey = depthClass * BINS + bin;
    }
  }
  keyDistance[keyDistanceOffset + index] = distanceKey;
  keyClass[keyClassOffset + index] = classKey;
  maskMagnitude[maskMagnitudeOffset + index] = select(0u, 1u, eventMagnitude >= parameters[parametersOffset + 1u] - 0.0001);
  // Gutenberg-Richter subset: all events, only those within the distance, or only the others.
  let subset = u32(parameters[parametersOffset + 2u]);
  let isNear = kilometers >= 0.0 && kilometers <= parameters[parametersOffset + 3u];
  var included = true;
  if (subset == 1u) {
    included = isNear;
  } else if (subset == 2u) {
    included = !isNear;
  }
  let magnitudeBin = clamp(floor(eventMagnitude * 10.0 + 0.5) - ${MAGNITUDE_START * 10}.0, 0.0, ${MAGNITUDE_BIN_COUNT - 1}.0);
  keyMagnitude[keyMagnitudeOffset + index] = select(NO_KEY, u32(magnitudeBin), included);`
    });
    // Link each deep event to the center of the grid cell of its nearest boundary seed.
    addKernelPass(graph, {
      id: `${variant.key}-links`,
      invocationCount: count,
      declarations: `const NONE: u32 = ${NO_KEY}u;`,
      bindings: [
        {name: 'positions', view: positionsView, type: 'f32', access: 'read'},
        {name: 'distanceKm', view: distanceKmView, type: 'f32', access: 'read'},
        {name: 'depth', view: depthView, type: 'f32', access: 'read'},
        {name: 'parameters', view: parametersView, type: 'f32', access: 'read'},
        {name: 'nearestCells', view: nearestView, type: 'u32', access: 'read'},
        {
          name: 'linkSegments',
          view: d('link-segments', variant.linkSegments, 'float32', count * 4),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'linkFade',
          view: d('link-fade', variant.linkFade, 'float32', count),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: /* wgsl */ `
  linkFade[linkFadeOffset + index] = 0.0;
  linkSegments[linkSegmentsOffset + index * 4u] = 0.0;
  linkSegments[linkSegmentsOffset + index * 4u + 1u] = 0.0;
  linkSegments[linkSegmentsOffset + index * 4u + 2u] = 0.0;
  linkSegments[linkSegmentsOffset + index * 4u + 3u] = 0.0;
  if (parameters[parametersOffset + 6u] < 0.5 || distanceKm[distanceKmOffset + index] < 0.0) {
    return;
  }
  if (depth[depthOffset + index] < parameters[parametersOffset + 7u]) {
    return;
  }
  let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let cellX = u32(floor((x - parameters[parametersOffset + 8u]) * parameters[parametersOffset + 10u]));
  let cellY = u32(floor((y - parameters[parametersOffset + 9u]) * parameters[parametersOffset + 10u]));
  let width = u32(parameters[parametersOffset + 12u]);
  let nearest = nearestCells[nearestCellsOffset + cellY * width + cellX];
  if (nearest == NONE) {
    return;
  }
  let size = parameters[parametersOffset + 11u];
  linkSegments[linkSegmentsOffset + index * 4u] = x;
  linkSegments[linkSegmentsOffset + index * 4u + 1u] = y;
  linkSegments[linkSegmentsOffset + index * 4u + 2u] = parameters[parametersOffset + 8u] + (f32(nearest % width) + 0.5) * size;
  linkSegments[linkSegmentsOffset + index * 4u + 3u] = parameters[parametersOffset + 9u] + (f32(nearest / width) + 0.5) * size;
  linkFade[linkFadeOffset + index] = 1.0;`
    });
    return {compiled: resources.track(graph.compile())};
  }

  function getRegion(): RegionVariant {
    const options = ctx.options;
    const key = `${options.region}|${options.variance}`;
    let variant = regionVariants.get(key);
    if (!variant) {
      variant = buildRegion(key, options.region, options.variance);
      regionVariants.set(key, variant);
    }
    return variant;
  }

  function getDistanceVariant(variant: RegionVariant): DistanceVariant {
    const algorithm = ctx.options.algorithm;
    let result = variant.distanceVariants.get(algorithm);
    if (!result) {
      result = buildDistanceVariant(variant, algorithm);
      variant.distanceVariants.set(algorithm, result);
    }
    return result;
  }

  function acceptClass(code: number): boolean {
    const family = BOUNDARY_FAMILY_OF_CLASS[code];
    switch (ctx.options.boundaryClasses) {
      case 'convergent':
        return family === 0;
      case 'subduction':
        return code === 0;
      case 'divergent':
        return family === 1;
      case 'transform':
        return family === 2;
      default:
        return true;
    }
  }

  function writeSeeds(): void {
    if (!current) return;
    const seeds = densifyBoundarySeeds(
      current.steps,
      acceptClass,
      current.grid.cellSize * 0.8,
      SEED_CAPACITY
    );
    current.seedPositions.write(seeds.positions);
    current.seedIds.write(seeds.families);
    current.seedCount.write(Uint32Array.of(seeds.count));
    const selected = new Uint32Array(current.steps.count);
    let selectedCount = 0;
    for (let step = 0; step < current.steps.count; step++) {
      if (acceptClass(current.steps.stepClass[step])) selected[selectedCount++] = step;
    }
    if (selectedCount > 0) current.selectedSteps.write(selected.subarray(0, selectedCount));
    current.selectedStepCount = selectedCount;
    current.seedInfo = `${formatCount(seeds.count)} seeds on ${formatCount(selectedCount)} boundary steps, ${(current.grid.cellSize / 1000).toFixed(1)} km cells${seeds.overflow ? ' (seed list full)' : ''}`;
    ctx.setReadout('seeds', current.seedInfo);
  }

  function writeParameters(): void {
    if (!current) return;
    const options = ctx.options;
    const {grid} = current;
    const cap =
      options.maximumDistance >= NO_DISTANCE_CAP ? Infinity : options.maximumDistance * 1000;
    distanceSettings.write(
      getGPUDistanceFieldParameterValues({
        origin: grid.origin,
        cellSize: [grid.cellSize, grid.cellSize],
        maxDistance: cap
      })
    );
    eventParameters.write(
      Float32Array.of(
        Number(options.binWidth),
        options.minimumMagnitude,
        options.grSubset === 'all' ? 0 : options.grSubset === 'near' ? 1 : 2,
        options.grDistance,
        options.shallowLimit,
        options.deepLimit,
        options.showLinks ? 1 : 0,
        options.linkMinimumDepth,
        grid.origin[0],
        grid.origin[1],
        1 / grid.cellSize,
        grid.cellSize,
        GRID,
        GRID,
        0,
        0
      )
    );
    percentileParameters.write(Float32Array.of(options.lowerFraction, 0.5, options.upperFraction));
    dirty = true;
  }

  function showStatistics(bytes: ArrayBuffer): void {
    const options = ctx.options;
    let offset = 0;
    const binCounts = new Uint32Array(bytes, offset, BIN_COUNT);
    offset += BIN_COUNT * 4;
    const means = new Float32Array(bytes, offset, BIN_COUNT);
    offset += BIN_COUNT * 4;
    const deviations = new Float32Array(bytes, offset, BIN_COUNT);
    offset += BIN_COUNT * 4;
    const percentiles = new Float32Array(bytes, offset, BIN_COUNT * PERCENTILE_COUNT);
    offset += BIN_COUNT * PERCENTILE_COUNT * 4;
    const classCounts = new Uint32Array(bytes, offset, CLASS_COUNT * BIN_COUNT);
    offset += CLASS_COUNT * BIN_COUNT * 4;
    magnitudeCounts = new Uint32Array(bytes.slice(offset, offset + MAGNITUDE_BIN_COUNT * 4));
    const binKm = Number(options.binWidth);

    // Depth against distance: median line, mean line and a band between the two percentiles.
    const centers: number[] = [];
    const median: number[] = [];
    const mean: number[] = [];
    const low: number[] = [];
    const high: number[] = [];
    let total = 0;
    let within100 = 0;
    let slopeN = 0;
    let slopeX = 0;
    let slopeY = 0;
    let slopeXX = 0;
    let slopeXY = 0;
    for (let bin = 0; bin < BIN_COUNT; bin++) {
      total += binCounts[bin];
      const center = (bin + 0.5) * binKm;
      if (center <= 100) within100 += binCounts[bin];
      if (binCounts[bin] < 3) continue;
      centers.push(center);
      mean.push(-means[bin]);
      low.push(-percentiles[bin * PERCENTILE_COUNT + 2]);
      high.push(-percentiles[bin * PERCENTILE_COUNT]);
      median.push(-percentiles[bin * PERCENTILE_COUNT + 1]);
      if (binCounts[bin] >= 8 && center <= 500) {
        slopeN++;
        slopeX += center;
        slopeY += percentiles[bin * PERCENTILE_COUNT + 1];
        slopeXX += center * center;
        slopeXY += center * percentiles[bin * PERCENTILE_COUNT + 1];
      }
    }
    ctx.setReadout(
      'eventsAnalysed',
      `${formatCount(total)} events M${options.minimumMagnitude.toFixed(1)}+ with a distance`
    );
    ctx.setReadout(
      'nearShare',
      total > 0 ? `${((within100 / total) * 100).toFixed(0)}% within 100 km` : 'n/a'
    );
    const slope =
      slopeN >= 3
        ? (slopeN * slopeXY - slopeX * slopeY) / (slopeN * slopeXX - slopeX * slopeX)
        : Number.NaN;
    ctx.setReadout(
      'apparentDip',
      Number.isFinite(slope)
        ? `${formatNumber(slope, 2)} km deeper per km (${formatNumber((Math.atan(slope) * 180) / Math.PI, 0)} degrees)`
        : 'too few events'
    );
    let deviationSum = 0;
    let deviationCount = 0;
    for (let bin = 0; bin < BIN_COUNT; bin++) {
      if (binCounts[bin] >= 3 && Number.isFinite(deviations[bin])) {
        deviationSum += deviations[bin] * binCounts[bin];
        deviationCount += binCounts[bin];
      }
    }
    ctx.setReadout(
      'depthSpread',
      deviationCount > 0
        ? `${formatNumber(deviationSum / deviationCount, 0)} km (${options.variance} deviation, weighted over bins)`
        : 'n/a'
    );
    ctx.setChart('depthByDistance', {
      kind: 'line',
      xLabel: 'distance to the nearest plate boundary (km)',
      yLabel: 'depth (km, down is deeper)',
      series: [
        {label: 'median depth', x: centers, y: median, color: 0},
        {label: 'mean depth', x: centers, y: mean, color: 3, dashed: true}
      ],
      band: {
        x: centers,
        low,
        high,
        label: `${Math.round(options.lowerFraction * 100)} to ${Math.round(options.upperFraction * 100)}% of events`
      },
      formatY: value => `${Math.abs(Math.round(value))}`,
      description:
        'Median and mean depth of earthquakes against distance to the nearest plate boundary, with a percentile band.'
    });

    // Where each depth class lives: share of the class per distance bin.
    const classTotals = [0, 0, 0];
    for (let depthClass = 0; depthClass < CLASS_COUNT; depthClass++) {
      for (let bin = 0; bin < BIN_COUNT; bin++) {
        classTotals[depthClass] += classCounts[depthClass * BIN_COUNT + bin];
      }
    }
    const binCentersAll = Array.from({length: BIN_COUNT}, (_, bin) => (bin + 0.5) * binKm);
    ctx.setChart('depthClasses', {
      kind: 'line',
      xLabel: 'distance to the nearest plate boundary (km)',
      yLabel: 'share of the depth class (% per bin)',
      series: QUAKE_DEPTH_CLASS_NAMES.map((name, depthClass) => ({
        label: `${name} (${formatCount(classTotals[depthClass])})`,
        x: binCentersAll,
        y: Array.from({length: BIN_COUNT}, (_, bin) =>
          classTotals[depthClass] > 0
            ? (classCounts[depthClass * BIN_COUNT + bin] / classTotals[depthClass]) * 100
            : 0
        ),
        color: depthClass + 1,
        area: depthClass === 0
      })),
      description: 'Histogram of distance to the nearest plate boundary for each depth class.'
    });
    const medians = QUAKE_DEPTH_CLASS_NAMES.map((name, depthClass) => {
      const row = Array.from(
        classCounts.subarray(depthClass * BIN_COUNT, (depthClass + 1) * BIN_COUNT)
      );
      const value = getHistogramMedian(row, binKm);
      return `${name}: ${Number.isFinite(value) ? `${Math.round(value)} km` : 'none'}`;
    });
    ctx.setReadout('classMedians', medians.join('\n'));
    showGutenbergRichter();
  }

  function showGutenbergRichter(): void {
    if (!magnitudeCounts) return;
    const options = ctx.options;
    const counts = magnitudeCounts;
    const fit = fitGutenbergRichter(counts, MAGNITUDE_START, 0.1, options.completeness);
    const magnitudes: number[] = [];
    const logCumulative: number[] = [];
    let cumulative = 0;
    const cumulativeCounts = new Array<number>(counts.length).fill(0);
    for (let bin = counts.length - 1; bin >= 0; bin--) {
      cumulative += counts[bin];
      cumulativeCounts[bin] = cumulative;
    }
    for (let bin = 0; bin < counts.length; bin++) {
      if (cumulativeCounts[bin] > 0) {
        magnitudes.push(MAGNITUDE_START + bin * 0.1);
        logCumulative.push(Math.log10(cumulativeCounts[bin]));
      }
    }
    const subset =
      options.grSubset === 'all'
        ? 'all events'
        : options.grSubset === 'near'
          ? `events within ${options.grDistance} km of a boundary`
          : `events farther than ${options.grDistance} km`;
    ctx.setReadout(
      'bValue',
      Number.isFinite(fit.b)
        ? `b = ${fit.b.toFixed(2)} ± ${fit.standardError.toFixed(2)}`
        : 'too few events'
    );
    ctx.setReadout(
      'grEvents',
      `${formatCount(fit.count)} of ${formatCount(cumulativeCounts[0] ?? 0)} (${subset}) at or above M${options.completeness.toFixed(1)}`
    );
    const maximumMagnitude =
      magnitudes.length > 0 ? magnitudes[magnitudes.length - 1] : MAGNITUDE_START;
    const series: ChartSeries[] = [
      {label: 'observed log10 N(M or more)', x: magnitudes, y: logCumulative}
    ];
    if (Number.isFinite(fit.b)) {
      series.push({
        label: `fit, b = ${fit.b.toFixed(2)}`,
        x: [options.completeness, maximumMagnitude],
        y: [fit.a - fit.b * options.completeness, fit.a - fit.b * maximumMagnitude],
        dashed: true,
        color: 1
      });
    }
    ctx.setChart('gutenberg', {
      kind: 'line',
      xLabel: 'magnitude M',
      yLabel: 'log10 of the number of events of magnitude M or more',
      series,
      markers: [{x: options.completeness, label: 'Mc'}],
      description: 'Cumulative magnitude-frequency distribution with the Gutenberg-Richter fit.'
    });
  }

  /** Distance graph of the current region and algorithm, compiled when the choice changes. */
  let currentDistance: DistanceVariant | null = null;

  function adopt(): void {
    current = getRegion();
    currentDistance = getDistanceVariant(current);
    magnitudeCounts = null;
    writeSeeds();
    writeParameters();
    ctx.setReadout(
      'events',
      `${formatCount(current.events.count)} M4+ earthquakes in ${QUAKE_REGIONS[ctx.options.region].label}`
    );
    ctx.requestLayers();
  }

  adopt();

  return {
    getCompiledGraphs: () =>
      current && currentDistance
        ? ([currentDistance.compiled, current.statistics] as CompiledGPUCommandGraph<never>[])
        : [],

    setOption(id) {
      if (id === 'region' || id === 'variance') {
        adopt();
        if (id === 'region') {
          const [west, south, east, north] = QUAKE_REGIONS[ctx.options.region].bbox;
          const view = ctx.getViewState();
          const inside =
            view.longitude >= west &&
            view.longitude <= east &&
            view.latitude >= south &&
            view.latitude <= north;
          if (!inside) ctx.flyTo(QUAKE_REGIONS[ctx.options.region].view, {transitionMs: 1000});
        }
      } else if (id === 'boundaryClasses') {
        writeSeeds();
        writeParameters();
      } else if (id === 'completeness') {
        showGutenbergRichter();
      } else if (id === 'algorithm') {
        if (current) currentDistance = getDistanceVariant(current);
        dirty = true;
      } else {
        writeParameters();
      }
      ctx.requestLayers();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      if (!current || !currentDistance) return;
      if (dirty || frame.frameIndex < 2) {
        currentDistance.compiled.encode(commandEncoder, {parameters: undefined});
        current.statistics.encode(commandEncoder, {parameters: undefined});
        dirty = false;
        current.reader.request(commandEncoder);
      }
      current.reader.flush(commandEncoder);
    },

    getLayers() {
      if (!current) return [];
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const origin = current.events.origin;
      const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
      const {grid} = current;
      const layers: Layer[] = [];
      if (options.mapShows !== 'events') {
        const common = {
          coordinateOrigin,
          gridSize: [GRID, GRID] as const,
          bounds: [
            grid.origin[0],
            grid.origin[1],
            grid.origin[0] + GRID * grid.cellSize,
            grid.origin[1] + GRID * grid.cellSize
          ] as const,
          tessellation: 48,
          opacity: options.mapOpacity,
          color: [255, 255, 255, 255] as const
        };
        if (options.mapShows === 'distance') {
          const rangeKm = Math.min(options.maximumDistance, 1000);
          layers.push(
            new SpatialAnalysisRasterLayer({
              ...common,
              id: 'quake-distance',
              values: current.distances,
              valueFormat: 'float32',
              colormap: 'cividis',
              // Negative scale: a boundary cell is the bright end of the ramp.
              valueScale: -0.001,
              valueRange: [-rangeKm, 0]
            })
          );
        } else {
          layers.push(
            new SpatialAnalysisRasterLayer({
              ...common,
              id: 'quake-allocation',
              values: current.allocation,
              valueFormat: 'uint32',
              colormap: 'category',
              palette: BOUNDARY_FAMILY_COLORS.map(color => [...color, 230] as const)
            })
          );
        }
      }
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'quake-steps-all',
          coordinateOrigin,
          segments: current.stepSegments,
          instanceCount: current.steps.count,
          widthPixels: 1,
          colormap: 'uniform',
          color: dark ? [230, 235, 245, 70] : [40, 50, 70, 80]
        })
      );
      if (current.selectedStepCount > 0) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'quake-steps-selected',
            coordinateOrigin,
            segments: current.stepSegments,
            ids: current.selectedSteps,
            values: current.stepFamily,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: BOUNDARY_FAMILY_COLORS.map(color => [...color, 255] as const),
            instanceCount: current.selectedStepCount,
            widthPixels: 2.4
          })
        );
      }
      if (options.showLinks) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'quake-links',
            coordinateOrigin,
            segments: current.linkSegments,
            weights: current.linkFade,
            instanceCount: current.events.count,
            widthPixels: 0.9,
            colormap: 'uniform',
            color: dark ? [255, 255, 255, 110] : [20, 25, 40, 130]
          })
        );
      }
      layers.push(
        new QuakeEventLayer({
          id: 'quake-plate-events',
          coordinateOrigin,
          positions: current.positions,
          times: current.depth,
          magnitudes: current.magnitude,
          depths: current.depth,
          instanceCount: current.events.count,
          staticMode: true,
          minimumMagnitude: options.minimumMagnitude - 0.0001,
          sizePixels: 2.1 * options.sizeScale,
          sizeGrowth: 1.8,
          colorMode: options.eventColor,
          ramp: 'magma',
          depthMax: 300,
          palette: QUAKE_DEPTH_CLASS_COLORS[dark ? 'dark' : 'light'],
          classLimits: [options.shallowLimit, options.deepLimit],
          opacity: 0.9,
          outlineColor: dark ? [255, 255, 255, 80] : [20, 25, 40, 190]
        })
      );
      return layers;
    },

    destroy() {
      destroyed = true;
      for (const variant of regionVariants.values()) variant.reader.stop();
      resources.destroy();
    }
  };
}
