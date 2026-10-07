// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {
  getGPULineDensityParameterValues,
  GPU_LINE_DENSITY_PARAMETER_LENGTH,
  GPULineDensity,
  GPULineLengthPerPolygon
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisRasterLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {
  formatCount,
  getViewportMetricBounds,
  SpatialAnalysisResources
} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {formatCompiledGraphTiming, measureCompiledGraph} from '../../engine/vector-timing';
import type {SceneContext, SceneInstance} from '../scene';
import {LocalMetricProjection} from '../../engine/projection';
import {
  getFeatureRowAt,
  rasterizeFeatureRows,
  readPolygonSet,
  type FeatureRowGrid,
  type StreetPolygonSet
} from './b1-street-polygons';

/** Option state of the street-density scene. */
export type StreetDensityOptions = {
  view: 'density' | 'polygons';
  roadClass: 'all' | 'arterial' | 'local';
  system: 'planar' | 'spherical';
  cellSize: number;
  cellValue: 'density' | 'length';
  polygonSet: 'areas' | 'tracts';
  metric: 'length' | 'perArea' | 'perResident' | 'meanAttribute' | 'segments';
  weight: 'speed' | 'oneway' | 'major';
  ramp: 'inferno' | 'magma' | 'viridis' | 'cividis';
  showStreets: boolean;
  showOutlines: boolean;
};

const COLUMNS = 192;
const ROWS = 120;
const CELL_COUNT = COLUMNS * ROWS;
const CHOROPLETH_COLUMNS = 420;
const SETTLE_MILLISECONDS = 300;
const NO_DATA_VALUE = 0xffffffff;

type RoadSet = {
  id: StreetDensityOptions['roadClass'];
  pathCount: number;
  vertexCount: number;
  lengthMeters: number;
  positionsMeters: Float32Array;
  positionsDegrees: Float32Array;
  pathOffsets: Uint32Array;
  weights: Record<StreetDensityOptions['weight'], Float32Array>;
  segments: Float32Array;
};

type Buf = ReturnType<SpatialAnalysisResources['createBuffer']>;

type RoadBuffers = {
  segments: Buf;
  segmentCount: number;
  positionsMeters: Buf;
  positionsDegrees: Buf;
  pathOffsets: Buf;
  pathWeights: Buf;
};

type DensityVariant = {
  compiled: CompiledGPUCommandGraph<void>;
};

type PolygonVariant = {
  compiled: CompiledGPUCommandGraph<void>;
};

type PolygonState = {
  set: StreetPolygonSet;
  grid: FeatureRowGrid;
  featureCount: number;
  outline: Buf;
  cellRows: Buf;
  areas: Buf;
  population: Buf;
  lengths: Buf;
  weighted: Buf;
  counts: Buf;
  perArea: Buf;
  perResident: Buf;
  meanAttribute: Buf;
  overflow: Buf;
  polygonPositionsMeters: Buf;
  polygonPositionsDegrees: Buf;
  featureOffsets: Buf;
  polygonOffsets: Buf;
  ringOffsets: Buf;
  reader: SummaryReader;
  values: Record<'length' | 'perArea' | 'perResident' | 'meanAttribute', Float32Array>;
  countValues: Uint32Array;
  ranges: Record<StreetDensityOptions['metric'], number>;
  inside: number;
  overflowFlag: boolean;
  ready: boolean;
};

/**
 * Street density of Chicago. `GPULineDensity` clips every street to a camera-following grid and sums
 * the length per cell (planar meters or longitude/latitude with great-circle pieces);
 * `GPULineLengthPerPolygon` clips the same streets into community areas or census tracts and
 * returns length, an attribute-weighted length and a segment count per polygon. The road class
 * subset, the coordinate system and the polygon set are compile-time and compile lazily on first
 * use (cached); the cell size, weight attribute, view and metric are buffer writes.
 */
export async function createStreetDensity(
  ctx: SceneContext<StreetDensityOptions>
): Promise<SceneInstance<StreetDensityOptions>> {
  const {device} = ctx;
  const roads = ctx.datasets.get('chicago-roads');
  const areasData = ctx.datasets.get('chicago-community-areas');
  const tractsData = ctx.datasets.get('chicago-tracts');
  const origin = roads.defaultOrigin;
  const projection = new LocalMetricProjection(origin);
  const resources = new SpatialAnalysisResources(device, 'street-density');
  const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
  const metersPerDegreeLongitude = projection.project(origin[0] + 1, origin[1])[0];
  const metersPerDegreeLatitude = projection.project(origin[0], origin[1] + 1)[1];

  // ---- Road subsets (each street once: two-way streets keep one direction) ----
  const edgeVertices = roads.column<Float32Array>('edgeVertices');
  const edgePathOffsets = roads.column<Uint32Array>('edgePathOffsets');
  const edgeClass = roads.column<Uint8Array>('edgeClass');
  const edgeSpeed = roads.column<Uint8Array>('edgeSpeed');
  const edgeOneway = roads.column<Uint8Array>('edgeOneway');
  const edgeReverse = roads.column<Uint32Array>('edgeReverse');
  const edgeCount = edgeClass.length;
  const roadSets = new Map<StreetDensityOptions['roadClass'], RoadSet>();
  const roadBuffers = new Map<StreetDensityOptions['roadClass'], RoadBuffers>();

  const inClass = (roadClass: StreetDensityOptions['roadClass'], value: number) =>
    roadClass === 'all' || (roadClass === 'arterial' ? value <= 3 : value >= 4);

  function getRoadSet(roadClass: StreetDensityOptions['roadClass']): RoadSet {
    let set = roadSets.get(roadClass);
    if (set) return set;
    const kept: number[] = [];
    let vertexTotal = 0;
    for (let edge = 0; edge < edgeCount; edge++) {
      const unique = edgeReverse[edge] === NO_DATA_VALUE || edge < edgeReverse[edge];
      if (unique && inClass(roadClass, edgeClass[edge])) {
        kept.push(edge);
        vertexTotal += edgePathOffsets[edge + 1] - edgePathOffsets[edge];
      }
    }
    const positionsMeters = new Float32Array(vertexTotal * 2);
    const positionsDegrees = new Float32Array(vertexTotal * 2);
    const pathOffsets = new Uint32Array(kept.length + 1);
    const speed = new Float32Array(kept.length);
    const oneway = new Float32Array(kept.length);
    const major = new Float32Array(kept.length);
    const segments: number[] = [];
    let row = 0;
    let lengthMeters = 0;
    kept.forEach((edge, path) => {
      pathOffsets[path] = row;
      speed[path] = edgeSpeed[edge];
      oneway[path] = edgeOneway[edge] ? 1 : 0;
      major[path] = edgeClass[edge] <= 3 ? 1 : 0;
      let previous: [number, number] | null = null;
      for (let vertex = edgePathOffsets[edge]; vertex < edgePathOffsets[edge + 1]; vertex++) {
        const longitude = edgeVertices[vertex * 2];
        const latitude = edgeVertices[vertex * 2 + 1];
        const point = projection.project(longitude, latitude);
        positionsDegrees.set([longitude, latitude], row * 2);
        positionsMeters.set(point, row * 2);
        if (previous) {
          segments.push(previous[0], previous[1], point[0], point[1]);
          lengthMeters += Math.hypot(point[0] - previous[0], point[1] - previous[1]);
        }
        previous = point;
        row++;
      }
    });
    pathOffsets[kept.length] = row;
    set = {
      id: roadClass,
      pathCount: kept.length,
      vertexCount: vertexTotal,
      lengthMeters,
      positionsMeters,
      positionsDegrees,
      pathOffsets,
      weights: {speed, oneway, major},
      segments: Float32Array.from(segments)
    };
    roadSets.set(roadClass, set);
    return set;
  }

  function getRoadBuffers(roadClass: StreetDensityOptions['roadClass']): RoadBuffers {
    let buffers = roadBuffers.get(roadClass);
    if (buffers) return buffers;
    const set = getRoadSet(roadClass);
    buffers = {
      segments: resources.createBuffer(`${roadClass}-segments`, set.segments),
      segmentCount: set.segments.length / 4,
      positionsMeters: resources.createBuffer(`${roadClass}-positions-m`, set.positionsMeters),
      positionsDegrees: resources.createBuffer(`${roadClass}-positions-deg`, set.positionsDegrees),
      pathOffsets: resources.createBuffer(`${roadClass}-path-offsets`, set.pathOffsets),
      pathWeights: resources.createBuffer(
        `${roadClass}-path-weights`,
        set.weights[ctx.options.weight]
      )
    };
    roadBuffers.set(roadClass, buffers);
    return buffers;
  }

  // ---- Density graphs ----
  const densityParameters = resources.createParameterBuffer(
    'density-parameters',
    'float32',
    GPU_LINE_DENSITY_PARAMETER_LENGTH
  );
  const gridBounds = resources.createParameterBuffer('grid-bounds', 'float32', 4);
  const lengths = resources.createBuffer('cell-lengths', CELL_COUNT * 4);
  const densities = resources.createBuffer('cell-densities', CELL_COUNT * 4);
  const densityOverflow = resources.createBuffer('density-overflow', 4);
  const totalRecords = resources.createBuffer('total-records', 4);
  const densityVariants = new Map<string, DensityVariant>();

  function getDensityVariant(options: StreetDensityOptions): DensityVariant {
    const key = `${options.roadClass}|${options.system}`;
    let variant = densityVariants.get(key);
    if (variant) return variant;
    const road = getRoadSet(options.roadClass);
    const buffers = getRoadBuffers(options.roadClass);
    const spherical = options.system === 'spherical';
    const graph = new GPUCommandGraph<void>(device, {id: `line-density-${key}`});
    graph.add(
      new GPULineDensity({
        id: 'line-density',
        positions: importGraphBuffer(
          graph,
          'positions',
          spherical ? buffers.positionsDegrees : buffers.positionsMeters,
          'float32x2',
          road.vertexCount
        ),
        pathOffsets: importGraphBuffer(
          graph,
          'path-offsets',
          buffers.pathOffsets,
          'uint32',
          road.pathCount + 1
        ),
        columns: COLUMNS,
        rows: ROWS,
        coordinateSystem: options.system,
        maximumRecords: Math.max(1024, 4 * road.vertexCount),
        parameters: densityParameters.importToGraph(graph),
        output: {
          lengths: importGraphBuffer(graph, 'lengths', lengths, 'float32', CELL_COUNT),
          densities: importGraphBuffer(graph, 'densities', densities, 'float32', CELL_COUNT),
          overflow: importGraphBuffer(graph, 'overflow', densityOverflow, 'uint32', 1),
          totalRecords: importGraphBuffer(graph, 'total-records', totalRecords, 'uint32', 1)
        }
      })
    );
    variant = {compiled: resources.track(graph.compile())};
    densityVariants.set(key, variant);
    return variant;
  }

  // ---- Polygon sets ----
  const areaNames = ((areasData.manifest as unknown as {names?: string[]}).names ?? []).slice();
  const tractIds = ((tractsData.manifest as unknown as {geoid?: string[]}).geoid ?? []).slice();
  const tractPopulation = tractsData.column<Float32Array>('population');
  const tractCommunity = tractsData.column<Uint8Array>('communityArea');
  const areaPopulation = new Float32Array(areaNames.length);
  for (let tract = 0; tract < tractPopulation.length; tract++) {
    const area = tractCommunity[tract];
    if (area > 0 && Number.isFinite(tractPopulation[tract]))
      areaPopulation[area - 1] += tractPopulation[tract];
  }
  const polygonSets: Record<StreetDensityOptions['polygonSet'], StreetPolygonSet> = {
    areas: readPolygonSet(areasData, projection, {
      id: 'areas',
      label: 'Community areas (77)',
      names: areaNames,
      population: areaPopulation
    }),
    tracts: readPolygonSet(tractsData, projection, {
      id: 'tracts',
      label: 'Census tracts (791)',
      names: tractIds.map(id => `Tract ${id}`),
      population: Float32Array.from(tractPopulation, value => (Number.isFinite(value) ? value : 0))
    })
  };
  const polygonStates = new Map<StreetDensityOptions['polygonSet'], PolygonState>();
  const polygonVariants = new Map<string, PolygonVariant>();

  function getPolygonState(id: StreetDensityOptions['polygonSet']): PolygonState {
    let state = polygonStates.get(id);
    if (state) return state;
    const set = polygonSets[id];
    const featureCount = set.featureOffsets.length - 1;
    const grid = rasterizeFeatureRows(set, CHOROPLETH_COLUMNS);
    const noData = new Float32Array(featureCount + 1).fill(Number.NaN);
    const noDataCounts = new Uint32Array(featureCount + 1).fill(NO_DATA_VALUE);
    const next: PolygonState = {
      set,
      grid,
      featureCount,
      outline: resources.createBuffer(`${id}-outline`, set.outlineSegments),
      cellRows: resources.createBuffer(`${id}-cell-rows`, grid.featureRows),
      areas: resources.createBuffer(`${id}-areas`, set.areas),
      population: resources.createBuffer(`${id}-population`, set.population),
      lengths: resources.createBuffer(`${id}-lengths`, noData),
      weighted: resources.createBuffer(`${id}-weighted`, noData),
      counts: resources.createBuffer(`${id}-counts`, noDataCounts),
      perArea: resources.createBuffer(`${id}-per-area`, noData),
      perResident: resources.createBuffer(`${id}-per-resident`, noData),
      meanAttribute: resources.createBuffer(`${id}-mean-attribute`, noData),
      overflow: resources.createBuffer(`${id}-overflow`, 4),
      polygonPositionsMeters: resources.createBuffer(`${id}-positions-m`, set.positions),
      polygonPositionsDegrees: resources.createBuffer(`${id}-positions-deg`, set.degrees),
      featureOffsets: resources.createBuffer(`${id}-feature-offsets`, set.featureOffsets),
      polygonOffsets: resources.createBuffer(`${id}-polygon-offsets`, set.polygonOffsets),
      ringOffsets: resources.createBuffer(`${id}-ring-offsets`, set.ringOffsets),
      reader: undefined as unknown as SummaryReader,
      values: {
        length: new Float32Array(featureCount),
        perArea: new Float32Array(featureCount),
        perResident: new Float32Array(featureCount),
        meanAttribute: new Float32Array(featureCount)
      },
      countValues: new Uint32Array(featureCount),
      ranges: {length: 1, perArea: 1, perResident: 1, meanAttribute: 1, segments: 1},
      inside: 0,
      overflowFlag: false,
      ready: false
    };
    next.reader = new SummaryReader(
      resources,
      `polygons-${id}`,
      [
        {buffer: next.lengths, size: featureCount * 4},
        {buffer: next.perArea, size: featureCount * 4},
        {buffer: next.perResident, size: featureCount * 4},
        {buffer: next.meanAttribute, size: featureCount * 4},
        {buffer: next.counts, size: featureCount * 4},
        {buffer: next.overflow, size: 4}
      ],
      bytes => handlePolygonSummary(next, bytes)
    );
    polygonStates.set(id, next);
    return next;
  }

  function getPolygonVariant(options: StreetDensityOptions): PolygonVariant {
    const key = `${options.polygonSet}|${options.roadClass}|${options.system}`;
    let variant = polygonVariants.get(key);
    if (variant) return variant;
    const state = getPolygonState(options.polygonSet);
    const road = getRoadSet(options.roadClass);
    const buffers = getRoadBuffers(options.roadClass);
    const spherical = options.system === 'spherical';
    const {featureCount, set} = state;
    const graph = new GPUCommandGraph<void>(device, {id: `line-length-${key}`});
    const view = <F extends 'float32' | 'uint32' | 'float32x2'>(
      name: string,
      buffer: Buf,
      format: F,
      length: number
    ) => importGraphBuffer(graph, name, buffer, format, length);
    const lengthView = view('lengths', state.lengths, 'float32', featureCount);
    const weightedView = view('weighted', state.weighted, 'float32', featureCount);
    graph.add(
      new GPULineLengthPerPolygon({
        id: `line-length-${key}`,
        positions: view(
          'positions',
          spherical ? buffers.positionsDegrees : buffers.positionsMeters,
          'float32x2',
          road.vertexCount
        ),
        pathOffsets: view('path-offsets', buffers.pathOffsets, 'uint32', road.pathCount + 1),
        pathWeights: view('path-weights', buffers.pathWeights, 'float32', road.pathCount),
        coordinateSystem: options.system,
        polygons: {
          kind: 'polygons',
          positions: view(
            'polygon-positions',
            spherical ? state.polygonPositionsDegrees : state.polygonPositionsMeters,
            'float32x2',
            set.positions.length / 2
          ),
          featureOffsets: view(
            'feature-offsets',
            state.featureOffsets,
            'uint32',
            set.featureOffsets.length
          ),
          polygonOffsets: view(
            'polygon-offsets',
            state.polygonOffsets,
            'uint32',
            set.polygonOffsets.length
          ),
          ringOffsets: view('ring-offsets', state.ringOffsets, 'uint32', set.ringOffsets.length)
        },
        maximumCandidatePairs: Math.max(1024, 3 * road.vertexCount),
        output: {
          lengths: lengthView,
          weightedLengths: weightedView,
          segmentCounts: view('counts', state.counts, 'uint32', featureCount),
          overflow: view('overflow', state.overflow, 'uint32', 1)
        }
      })
    );
    // Derived metrics, written for every polygon so the metric option only picks a buffer.
    addKernelPass(graph, {
      id: `${key}-metrics`,
      invocationCount: featureCount,
      bindings: [
        {name: 'lengths', view: lengthView, type: 'f32', access: 'read'},
        {name: 'weighted', view: weightedView, type: 'f32', access: 'read'},
        {
          name: 'areas',
          view: view('areas', state.areas, 'float32', featureCount),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'population',
          view: view('population', state.population, 'float32', featureCount),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'perArea',
          view: view('per-area', state.perArea, 'float32', featureCount),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'perResident',
          view: view('per-resident', state.perResident, 'float32', featureCount),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'meanAttribute',
          view: view('mean-attribute', state.meanAttribute, 'float32', featureCount),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: /* wgsl */ `
  let length = lengths[lengthsOffset + index];
  // kilometers of street per square kilometer
  perArea[perAreaOffset + index] = length / max(areas[areasOffset + index], 1.0) * 1000.0;
  let residents = population[populationOffset + index];
  let noData = bitcast<f32>(0x7fc00000u);
  perResident[perResidentOffset + index] = select(noData, length / residents, residents > 0.0 && length > 0.0);
  meanAttribute[meanAttributeOffset + index] = select(noData, weighted[weightedOffset + index] / length, length > 0.0);`
    });
    variant = {compiled: resources.track(graph.compile())};
    polygonVariants.set(key, variant);
    return variant;
  }

  function handlePolygonSummary(state: PolygonState, bytes: ArrayBuffer): void {
    if (destroyed) return;
    const n = state.featureCount;
    state.values.length = new Float32Array(bytes.slice(0, n * 4));
    state.values.perArea = new Float32Array(bytes.slice(n * 4, n * 8));
    state.values.perResident = new Float32Array(bytes.slice(n * 8, n * 12));
    state.values.meanAttribute = new Float32Array(bytes.slice(n * 12, n * 16));
    state.countValues = new Uint32Array(bytes.slice(n * 16, n * 20));
    state.overflowFlag = new Uint32Array(bytes, n * 20, 1)[0] !== 0;
    state.inside = 0;
    for (let row = 0; row < n; row++) state.inside += state.values.length[row];
    const percentile = (values: ArrayLike<number>) => {
      const sorted = Array.from(values)
        .filter(value => Number.isFinite(value) && value > 0)
        .sort((a, b) => a - b);
      return sorted.length
        ? sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.98))]
        : 1;
    };
    state.ranges.length = percentile(state.values.length);
    state.ranges.perArea = percentile(state.values.perArea);
    state.ranges.perResident = percentile(state.values.perResident);
    state.ranges.meanAttribute = percentile(state.values.meanAttribute);
    state.ranges.segments = percentile(state.countValues);
    state.ready = true;
    updatePolygonReadouts();
    pushLegendExtent();
    ctx.requestLayers();
  }

  function pushLegendExtent(): void {
    const options = ctx.options;
    if (options.view !== 'polygons') return;
    const state = polygonStates.get(options.polygonSet);
    if (!state?.ready) return;
    ctx.setLegendExtent('metric', [0, state.ranges[options.metric]]);
  }

  function updatePolygonReadouts(): void {
    const options = ctx.options;
    const state = polygonStates.get(options.polygonSet);
    const road = getRoadSet(options.roadClass);
    ctx.setReadout(
      'polygons',
      `${formatCount(polygonSets[options.polygonSet].featureOffsets.length - 1)} ${options.polygonSet === 'areas' ? 'community areas' : 'census tracts'}`
    );
    if (!state?.ready) {
      for (const id of ['inside', 'conservation', 'polygonMaximum', 'polygonOverflow'])
        ctx.setReadout(id, 'switch to the polygon view');
      return;
    }
    const percent = (100 * state.inside) / road.lengthMeters;
    ctx.setReadout(
      'inside',
      `${(state.inside / 1000).toFixed(0)} km of ${(road.lengthMeters / 1000).toFixed(0)} km (${percent.toFixed(2)}%)`
    );
    ctx.setReadout(
      'conservation',
      Math.abs(percent - 100) < 1
        ? `conserved (${(percent - 100).toFixed(2)}%)`
        : `${(100 - percent).toFixed(1)}% of street length is outside the polygons or lost in simplification`
    );
    let maxRow = 0;
    for (let row = 1; row < state.featureCount; row++)
      if (state.values.length[row] > state.values.length[maxRow]) maxRow = row;
    ctx.setReadout(
      'polygonMaximum',
      `${state.set.names[maxRow]}: ${(state.values.length[maxRow] / 1000).toFixed(0)} km`
    );
    ctx.setReadout('polygonOverflow', state.overflowFlag ? 'yes (lengths may be low)' : 'no');
  }

  // ---- State ----
  let destroyed = false;
  let measuring = false;
  let densityMaximum = 0.01;
  let lengthMaximum = 100;
  let lastSignature = '';
  let dirty = true;
  let polygonDirty = true;
  let lastChangeTime = performance.now();
  let displayedDensity: DensityVariant | null = null;
  let displayedPolygons: PolygonVariant | null = null;
  let viewWidth = 1;

  const densityReader = new SummaryReader(
    resources,
    'density',
    [
      {buffer: lengths, size: CELL_COUNT * 4},
      {buffer: densities, size: CELL_COUNT * 4},
      {buffer: densityOverflow, size: 4},
      {buffer: totalRecords, size: 4}
    ],
    bytes => {
      if (destroyed) return;
      const floats = new Float32Array(bytes, 0, CELL_COUNT * 2);
      const words = new Uint32Array(bytes, CELL_COUNT * 8, 2);
      let total = 0;
      const positiveDensity: number[] = [];
      const positiveLength: number[] = [];
      for (let cell = 0; cell < CELL_COUNT; cell++) {
        total += floats[cell];
        if (floats[CELL_COUNT + cell] > 0) positiveDensity.push(floats[CELL_COUNT + cell]);
        if (floats[cell] > 0) positiveLength.push(floats[cell]);
      }
      const percentile = (values: number[], fallback: number) => {
        values.sort((a, b) => a - b);
        return values.length
          ? values[Math.min(values.length - 1, Math.floor(values.length * 0.99))]
          : fallback;
      };
      const nextDensity = percentile(positiveDensity, 0.001);
      const nextLength = percentile(positiveLength, 1);
      const road = getRoadSet(ctx.options.roadClass);
      ctx.setReadout(
        'lengthInGrid',
        `${(total / 1000).toFixed(0)} km of ${(road.lengthMeters / 1000).toFixed(0)} km (${((100 * total) / road.lengthMeters).toFixed(0)}%)`
      );
      ctx.setReadout(
        'densityPeak',
        `${(nextDensity * 1000).toFixed(1)} km per km² (99th percentile cell)`
      );
      ctx.setReadout(
        'records',
        `${formatCount(words[1])} pieces of ${formatCount(Math.max(1024, 4 * road.vertexCount))}`
      );
      ctx.setReadout('overflow', words[0] ? 'yes (lengths low)' : 'no');
      const changed =
        Math.abs(nextDensity - densityMaximum) > densityMaximum * 0.1 ||
        Math.abs(nextLength - lengthMaximum) > lengthMaximum * 0.1;
      densityMaximum = nextDensity;
      lengthMaximum = nextLength;
      ctx.setLegendExtent('cells', [
        0,
        ctx.options.cellValue === 'length' ? lengthMaximum : densityMaximum
      ]);
      if (changed) ctx.requestLayers();
    }
  );

  function markDirty(): void {
    dirty = true;
    polygonDirty = true;
    lastChangeTime = performance.now();
  }

  function ensureVariants(): void {
    const options = ctx.options;
    displayedDensity = getDensityVariant(options);
    if (options.view === 'polygons') displayedPolygons = getPolygonVariant(options);
    else
      displayedPolygons =
        polygonVariants.get(`${options.polygonSet}|${options.roadClass}|${options.system}`) ?? null;
    ctx.setReadout(
      'segments',
      `${formatCount(getRoadSet(options.roadClass).pathCount)} streets, ${formatCount(getRoadBuffers(options.roadClass).segmentCount)} segments`
    );
    markDirty();
  }

  ctx.setReadout('density', `${COLUMNS} x ${ROWS} = ${formatCount(CELL_COUNT)} cells`);
  ensureVariants();
  updatePolygonReadouts();

  async function measureGraphs(): Promise<void> {
    if (measuring || destroyed) return;
    measuring = true;
    ctx.setReadout('timing', 'measuring…');
    try {
      const options = ctx.options;
      const planar = getDensityVariant({...options, system: 'planar'});
      const spherical = getDensityVariant({...options, system: 'spherical'});
      const measureOptions = {
        parameters: undefined,
        completionBuffer: densityOverflow,
        signal: ctx.signal,
        warmUpRuns: 3
      };
      // Time each coordinate system with its own parameters.
      const bounds = lastBounds ?? [0, 0, 1000, 1000];
      writeDensityParameters(options, 'planar', bounds);
      const planarTiming = await measureCompiledGraph(device, planar.compiled, measureOptions);
      writeDensityParameters(options, 'spherical', bounds);
      const sphericalTiming = await measureCompiledGraph(
        device,
        spherical.compiled,
        measureOptions
      );
      writeDensityParameters(options, options.system, bounds);
      if (!destroyed) {
        ctx.setReadout(
          'timing',
          `planar ${formatCompiledGraphTiming(planarTiming).split(' · ')[0]}, spherical ${formatCompiledGraphTiming(sphericalTiming).split(' · ')[0]}`
        );
      }
    } catch {
      // Interrupted.
    } finally {
      measuring = false;
      markDirty();
    }
  }

  let lastBounds: [number, number, number, number] | null = null;

  /** Writes the grid origin and cell size in the units of the chosen coordinate system. */
  function writeDensityParameters(
    options: StreetDensityOptions,
    system: 'planar' | 'spherical',
    bounds: readonly [number, number, number, number]
  ): void {
    const size = options.cellSize;
    if (system === 'planar') {
      const minimumX = Math.floor(bounds[0] / size) * size;
      const minimumY = Math.floor(bounds[1] / size) * size;
      densityParameters.write(
        getGPULineDensityParameterValues({
          minX: minimumX,
          minY: minimumY,
          cellWidth: size,
          cellHeight: size
        })
      );
      gridBounds.write(
        Float32Array.of(minimumX, minimumY, minimumX + COLUMNS * size, minimumY + ROWS * size)
      );
    } else {
      const cellWidth = size / metersPerDegreeLongitude;
      const cellHeight = size / metersPerDegreeLatitude;
      const [west, south] = projection.unproject(bounds[0], bounds[1]);
      const minimumLongitude = Math.floor((west - origin[0]) / cellWidth) * cellWidth + origin[0];
      const minimumLatitude = Math.floor((south - origin[1]) / cellHeight) * cellHeight + origin[1];
      densityParameters.write(
        getGPULineDensityParameterValues({
          minX: minimumLongitude,
          minY: minimumLatitude,
          cellWidth,
          cellHeight
        })
      );
      const [x0, y0] = projection.project(minimumLongitude, minimumLatitude);
      const [x1, y1] = projection.project(
        minimumLongitude + COLUMNS * cellWidth,
        minimumLatitude + ROWS * cellHeight
      );
      gridBounds.write(Float32Array.of(x0, y0, x1, y1));
    }
  }

  return {
    getCompiledGraphs: () =>
      [displayedDensity?.compiled, displayedPolygons?.compiled].filter(
        Boolean
      ) as unknown as CompiledGPUCommandGraph<never>[],

    setOption(id, _value, state) {
      if (id === 'roadClass' || id === 'system' || id === 'polygonSet' || id === 'view') {
        ensureVariants();
        updatePolygonReadouts();
        pushLegendExtent();
        lastSignature = '';
        ctx.requestLayers();
      } else if (id === 'weight') {
        for (const [roadClass, buffers] of roadBuffers)
          buffers.pathWeights.write(getRoadSet(roadClass).weights[state.weight]);
        markDirty();
        ctx.requestLayers();
      } else if (id === 'cellSize') {
        lastSignature = '';
      } else if (id === 'cellValue') {
        ctx.setLegendExtent('cells', [
          0,
          state.cellValue === 'length' ? lengthMaximum : densityMaximum
        ]);
        ctx.requestLayers();
      } else if (id === 'metric') {
        pushLegendExtent();
        ctx.requestLayers();
      } else {
        ctx.requestLayers();
      }
    },

    onAction(id) {
      if (id === 'measure') void measureGraphs();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      const options = ctx.options;
      if (!displayedDensity) return;
      const bounds = getViewportMetricBounds(frame.viewport, projection);
      lastBounds = bounds;
      viewWidth = bounds[2] - bounds[0];
      const signature = [
        options.cellSize,
        options.system,
        options.roadClass,
        ...bounds.map(value => Math.round(value / 5))
      ].join(',');
      if (signature !== lastSignature) {
        lastSignature = signature;
        writeDensityParameters(options, options.system, bounds);
        dirty = true;
        lastChangeTime = performance.now();
      }
      const settled = performance.now() - lastChangeTime > SETTLE_MILLISECONDS;
      if (dirty && (settled || frame.frameIndex < 3 || options.view === 'density')) {
        displayedDensity.compiled.encode(commandEncoder, {parameters: undefined});
        densityReader.request(commandEncoder);
        dirty = false;
        ctx.setReadout(
          'grid',
          `${COLUMNS} x ${ROWS} cells of ${options.cellSize} m (${((COLUMNS * options.cellSize) / 1000).toFixed(1)} x ${((ROWS * options.cellSize) / 1000).toFixed(1)} km)`
        );
        ctx.setReadout(
          'coverage',
          `${Math.round((100 * COLUMNS * options.cellSize) / viewWidth)}% of the view width`
        );
      }
      densityReader.flush(commandEncoder);
      for (const state of polygonStates.values()) state.reader.flush(commandEncoder);
      if (options.view === 'polygons' && displayedPolygons && polygonDirty) {
        const state = getPolygonState(options.polygonSet);
        displayedPolygons.compiled.encode(commandEncoder, {parameters: undefined});
        state.reader.request(commandEncoder);
        polygonDirty = false;
      }
    },

    getLayers() {
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const buffers = getRoadBuffers(options.roadClass);
      const layers: Layer[] = [];
      if (options.view === 'polygons') {
        const state = getPolygonState(options.polygonSet);
        const metricBuffer =
          options.metric === 'length'
            ? state.lengths
            : options.metric === 'perArea'
              ? state.perArea
              : options.metric === 'perResident'
                ? state.perResident
                : options.metric === 'segments'
                  ? state.counts
                  : state.meanAttribute;
        // Lengths are in meters: scale to kilometers on the GPU color path.
        const scale = options.metric === 'length' ? 0.001 : 1;
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: `street-choropleth-${options.polygonSet}`,
            coordinateOrigin,
            gridSize: [state.grid.columns, state.grid.rows],
            bounds: state.grid.bounds,
            rowOrigin: 'south',
            values: metricBuffer,
            valueFormat: options.metric === 'segments' ? 'uint32' : 'float32',
            valueIndices: state.cellRows,
            colormap: options.ramp,
            valueRange: [0, state.ranges[options.metric] * scale],
            valueScale: scale,
            noDataColor: [0, 0, 0, 0],
            color: [255, 255, 255, 225]
          })
        );
        if (options.showStreets) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'street-density-streets',
              coordinateOrigin,
              segments: buffers.segments,
              instanceCount: buffers.segmentCount,
              widthPixels: 0.5,
              color: dark ? [255, 255, 255, 40] : [20, 20, 40, 45]
            })
          );
        }
        if (options.showOutlines) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: `street-outline-${options.polygonSet}`,
              coordinateOrigin,
              segments: state.outline,
              instanceCount: state.set.outlineSegments.length / 4,
              widthPixels: options.polygonSet === 'areas' ? 1.6 : 0.8,
              color: dark ? [255, 225, 160, 220] : [255, 255, 255, 230]
            })
          );
        }
        return layers;
      }
      layers.push(
        new SpatialAnalysisRasterLayer({
          id: 'street-density-cells',
          coordinateOrigin,
          gridSize: [COLUMNS, ROWS],
          bounds: gridBounds.buffer,
          values: options.cellValue === 'length' ? lengths : densities,
          valueFormat: 'float32',
          colormap: options.ramp,
          valueRange: [0, options.cellValue === 'length' ? lengthMaximum : densityMaximum],
          sqrtScale: true,
          discardAtOrBelow: 0,
          color: [255, 255, 255, 225]
        })
      );
      if (options.showStreets) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'street-density-streets',
            coordinateOrigin,
            segments: buffers.segments,
            instanceCount: buffers.segmentCount,
            widthPixels: 0.6,
            color: dark ? [150, 190, 255, 60] : [30, 60, 120, 60]
          })
        );
      }
      return layers;
    },

    getTooltip(event) {
      const options = ctx.options;
      if (options.view !== 'polygons' || !event.coordinate) return null;
      const state = polygonStates.get(options.polygonSet);
      if (!state?.ready) return null;
      const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
      const row = getFeatureRowAt(state.grid, state.featureCount, x, y);
      if (row < 0) return null;
      const area = state.set.areas[row] / 1e6;
      const attribute = state.values.meanAttribute[row];
      const population = state.set.population[row];
      const text = `${state.set.names[row]}: ${(state.values.length[row] / 1000).toFixed(1)} km of street in ${formatCount(state.countValues[row])} segments, ${state.values.perArea[row].toFixed(1)} km per km² (${area.toFixed(2)} km²)${population > 0 ? `, ${Number.isFinite(state.values.perResident[row]) ? state.values.perResident[row].toFixed(1) : '-'} m per resident (${formatCount(population)} residents)` : ''}, mean ${options.weight === 'speed' ? `${attribute.toFixed(0)} km/h` : `${(attribute * 100).toFixed(0)}% ${options.weight === 'oneway' ? 'one-way' : 'major roads'}`}`;
      ctx.setReadout('hovered', text);
      return text;
    },

    destroy() {
      destroyed = true;
      densityReader.stop();
      for (const state of polygonStates.values()) state.reader.stop();
      resources.destroy();
    }
  };
}
