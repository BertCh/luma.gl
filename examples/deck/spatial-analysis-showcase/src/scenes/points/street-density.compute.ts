// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {CommandEncoder} from '@luma.gl/core';
import {
  getGPULineDensityParameterValues,
  GPU_LINE_DENSITY_PARAMETER_LENGTH,
  GPULineDensity,
  GPULineLengthPerPolygon,
  type GPUParameterBuffer
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getClassBreaks, getClassCounts, getGoodnessOfVarianceFit} from '../../cartography/breaks';
import {
  getClassIndexOf,
  getClassLabel,
  getClassTableLayerProps,
  makeClassTable
} from '../../cartography/class-table';
import {CHICAGO, nearestPlace, nearestPlaceLabel} from '../../cartography/gazetteer';
import {
  formatArea,
  formatCount,
  formatDistance,
  formatOrdinal,
  formatPercent
} from '../../cartography/live-text';
import {
  createFeatureLocator,
  getGeometryPolygons,
  type FeatureLocator
} from '../../cartography/picking';
import {buildPolygonMesh, type PolygonMesh} from '../../cartography/polygon-mesh';
import {squareMileFrame} from '../../cartography/reference-geometry';
import type {ClassTable, LngLat, MapAnnotation} from '../../cartography/types';
import type {GeoJsonCollection} from '../../data/loaders';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {
  SpatialAnalysisPolygonLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer
} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {LocalMetricProjection} from '../../engine/projection';
import {createPolygonMeshBuffers, type PolygonMeshBuffers} from '../../engine/polygon-buffers';
import {getViewportMetricBounds, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {formatCompiledGraphTiming, measureCompiledGraph} from '../../engine/vector-timing';
import type {SceneContext, SceneInstance, TooltipContent, TooltipRow} from '../scene';
import {loadCityGeometry, type CityGeometry} from './b1-city-geometry';
import {readPolygonSet, type StreetPolygonSet} from './b1-street-polygons';
import {
  countAtOrBelow,
  findVoids,
  getArterialProfile,
  getBearingStatistics,
  getFieldStatistics,
  getRankFlipRows,
  getRanks,
  rasterizeRingsMask,
  type ArterialProfile,
  type FieldStatistics,
  type StreetVertices
} from './street-density-analysis';
import {
  CELL_SWEEP_SIZES,
  DENSITY_BREAKS,
  getDensityColors,
  getFieldOpacity,
  getOutsideColor,
  getPolygonLineColor,
  getStreetStyle,
  LAKE_PAPER_COLOR,
  LENGTH_BREAKS,
  makeDensityTable,
  MILE_CORRIDOR,
  MILE_METERS,
  MILE_VIEW,
  ROSE_BIN_COUNT,
  SUPPRESSION_RESIDENTS,
  VOID_DENSITY,
  type StreetColor,
  type StreetGround
} from './street-density-look';

/** Option state of the street-density scene. */
export type StreetDensityOptions = {
  /** Street length per grid cell, or street length clipped into polygons. */
  view: 'density' | 'polygons';
  /** What the density view draws: the street lines, the classed field, or the field over the lines. */
  layers: 'streets' | 'field' | 'both';
  /** Street lines: every street alike, or arterials bright and local streets pushed back. */
  streetEmphasis: 'all' | 'arterials';
  roadClass: 'all' | 'arterial' | 'local';
  system: 'planar' | 'spherical';
  cellSize: number;
  cellValue: 'density' | 'length';
  polygonSet: 'areas' | 'tracts';
  metric: 'length' | 'perArea' | 'perResident' | 'meanAttribute' | 'segments';
  weight: 'speed' | 'oneway' | 'major';
  showOutlines: boolean;
};

const COLUMNS = 192;
const ROWS = 120;
const CELL_COUNT = COLUMNS * ROWS;
const NO_DATA_VALUE = 0xffffffff;
/** Notes about gaps and the busiest cell are only drawn for cells at least this large (metres). */
const NOTE_MINIMUM_CELL_METERS = 200;
/** A gap must cover at least this area to be named (square metres). */
const VOID_MINIMUM_AREA = 1.5e6;
/** Gaps named on the map, at most. */
const VOID_NOTE_COUNT = 2;
/** The upper end of the density histogram, km of street per km² (higher cells fall in the last bin). */
const HISTOGRAM_TOP = 30;
/** Rows of the rank-flip chart come from the top of either list. */
const RANK_FLIP_TOP = 10;
/** Natural-break classes of the polygon maps. */
const POLYGON_CLASS_COUNT = 5;
/** Accent and neutral of the bearing rose (YlOrBr class 5 and the context grey). */
const ROSE_ACCENT: StreetColor = [217, 95, 14, 255];
const ROSE_NEUTRAL: StreetColor = [138, 148, 163, 255];
/** Places a gap note may name (not stations or stadiums). */
const VOID_PLACE_KINDS = ['airport', 'park', 'water', 'site', 'landform', 'landmark'] as const;

type RoadClass = StreetDensityOptions['roadClass'];
type Buf = ReturnType<SpatialAnalysisResources['createBuffer']>;

type RoadSet = {
  id: RoadClass;
  pathCount: number;
  vertexCount: number;
  lengthMeters: number;
  positionsMeters: Float32Array;
  positionsDegrees: Float32Array;
  pathOffsets: Uint32Array;
  pathClass: Uint8Array;
  weights: Record<StreetDensityOptions['weight'], Float32Array>;
  /** Segment rows `x0, y0, x1, y1` of motorways to secondary roads, and of everything else. */
  arterialSegments: Float32Array;
  localSegments: Float32Array;
};

type RoadBuffers = {
  arterial: Buf;
  arterialCount: number;
  local: Buf;
  localCount: number;
  positionsMeters: Buf;
  positionsDegrees: Buf;
  pathOffsets: Buf;
  pathWeights: Buf;
};

type DensityVariant = {compiled: CompiledGPUCommandGraph<void>};
type PolygonVariant = {compiled: CompiledGPUCommandGraph<void>};

/** The density grid in metres: where it sits and how big its cells are. */
type Grid = {minX: number; minY: number; maxX: number; maxY: number; cellMeters: number};

/** One density readback with what was derived from it. */
type FieldSnapshot = {
  grid: Grid;
  lengths: Float32Array;
  densities: Float32Array;
  cityMask: Uint8Array;
  stats: FieldStatistics;
  /** Non-empty city cells per class of the length table. */
  lengthClassCounts: number[];
  /** Most street in one city cell, metres. */
  maximumLength: number;
};

/** The frozen classification of one polygon set and metric. */
type PolygonClassing = {breaks: number[]; goodness: number; counts: number[]};

type PolygonState = {
  set: StreetPolygonSet;
  mesh: PolygonMesh;
  meshBuffers: PolygonMeshBuffers;
  locator: FeatureLocator;
  geojson: GeoJsonCollection;
  featureCount: number;
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
  /** Display values per metric: km, km per km², m per resident (NaN when suppressed), mean, segments. */
  values: Record<StreetDensityOptions['metric'], Float32Array>;
  ready: boolean;
  inside: number;
  overflowFlag: boolean;
};

/** What the legend needs about the displayed polygon map. */
export type PolygonLegendData = {
  key: string;
  table: ClassTable;
  counts: number[];
  title: string;
  note: string;
  suppressed: number;
};

/** Titles, units and number formats of the polygon metrics. */
function getMetricInfo(
  metric: StreetDensityOptions['metric'],
  weight: StreetDensityOptions['weight']
): {title: string; unit: string; digits: number; format: (value: number) => string} {
  switch (metric) {
    case 'length':
      return {
        title: 'Street length',
        unit: 'km of street',
        digits: 0,
        format: value => `${formatCount(value)} km`
      };
    case 'perArea':
      return {
        title: 'Street length per area',
        unit: 'km per km²',
        digits: 1,
        format: value => `${value.toFixed(1)} km per km²`
      };
    case 'perResident':
      return {
        title: 'Street length per resident',
        unit: 'm per resident',
        digits: 1,
        format: value => `${value.toFixed(1)} m per resident`
      };
    case 'segments':
      return {
        title: 'Street segments',
        unit: 'segments',
        digits: 0,
        format: value => `${formatCount(value)} segments`
      };
    default:
      return weight === 'speed'
        ? {
            title: 'Mean speed limit',
            unit: 'km/h',
            digits: 0,
            format: value => `${value.toFixed(0)} km/h`
          }
        : {
            title: weight === 'oneway' ? 'Share of one-way streets' : 'Share of major roads',
            unit: 'share of street length',
            digits: 2,
            format: value => formatPercent(value, 0)
          };
  }
}

/** Picks `count` colours spread over a class table (the ends and the middle), low class first. */
function pickColors(colors: readonly StreetColor[], count: number): StreetColor[] {
  if (count >= colors.length) return [...colors];
  if (count <= 1) return [colors[colors.length - 1]];
  return Array.from(
    {length: count},
    (_, index) => colors[Math.round((index * (colors.length - 1)) / (count - 1))]
  );
}

/**
 * Street density of Chicago, from lines. `GPULineDensity` clips every street to a camera-following
 * grid and sums the length per cell (planar metres or longitude/latitude with great-circle
 * pieces); `GPULineLengthPerPolygon` clips the same streets into community areas or census tracts
 * and returns length, an attribute-weighted length and a segment count per polygon. The road class
 * subset, the coordinate system and the polygon set are compile-time and compile lazily on first
 * use (cached); the cell size, weight attribute, view and metric are buffer writes. The scale sweep
 * of the cell-size step runs a second compiled copy of the density graph on its own buffers, six
 * parameter writes, so the map never flickers.
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
  const city: CityGeometry = await loadCityGeometry(ctx, origin);
  const unproject = (x: number, y: number) => projection.unproject(x, y) as LngLat;
  // The edge: the world outside the city (lake included) and the lake itself, uploaded once.
  const outsideBuffers = createPolygonMeshBuffers(resources, city.outsideMesh, 'outside');
  const lakeBuffers = city.lakeMesh
    ? createPolygonMeshBuffers(resources, city.lakeMesh, 'lake')
    : null;

  // ---- Road subsets (each street once: two-way streets keep one direction) ----
  const edgeVertices = roads.column<Float32Array>('edgeVertices');
  const edgePathOffsets = roads.column<Uint32Array>('edgePathOffsets');
  const edgeClass = roads.column<Uint8Array>('edgeClass');
  const edgeSpeed = roads.column<Uint8Array>('edgeSpeed');
  const edgeOneway = roads.column<Uint8Array>('edgeOneway');
  const edgeReverse = roads.column<Uint32Array>('edgeReverse');
  const edgeCount = edgeClass.length;
  const roadsWest = roads.manifest.bbox[0];
  const roadSets = new Map<RoadClass, RoadSet>();
  const roadBuffers = new Map<RoadClass, RoadBuffers>();

  const inClass = (roadClass: RoadClass, value: number) =>
    roadClass === 'all' || (roadClass === 'arterial' ? value <= 3 : value >= 4);

  function getRoadSet(roadClass: RoadClass): RoadSet {
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
    const pathClass = new Uint8Array(kept.length);
    const speed = new Float32Array(kept.length);
    const oneway = new Float32Array(kept.length);
    const major = new Float32Array(kept.length);
    const arterialSegments: number[] = [];
    const localSegments: number[] = [];
    let row = 0;
    let lengthMeters = 0;
    kept.forEach((edge, path) => {
      pathOffsets[path] = row;
      pathClass[path] = edgeClass[edge];
      speed[path] = edgeSpeed[edge];
      oneway[path] = edgeOneway[edge] ? 1 : 0;
      major[path] = edgeClass[edge] <= 3 ? 1 : 0;
      const target = edgeClass[edge] <= 3 ? arterialSegments : localSegments;
      let previous: [number, number] | null = null;
      for (let vertex = edgePathOffsets[edge]; vertex < edgePathOffsets[edge + 1]; vertex++) {
        const longitude = edgeVertices[vertex * 2];
        const latitude = edgeVertices[vertex * 2 + 1];
        const point = projection.project(longitude, latitude);
        positionsDegrees.set([longitude, latitude], row * 2);
        positionsMeters.set(point, row * 2);
        if (previous) {
          target.push(previous[0], previous[1], point[0], point[1]);
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
      pathClass,
      weights: {speed, oneway, major},
      arterialSegments: Float32Array.from(arterialSegments),
      localSegments: Float32Array.from(localSegments)
    };
    roadSets.set(roadClass, set);
    return set;
  }

  function getRoadBuffers(roadClass: RoadClass): RoadBuffers {
    let buffers = roadBuffers.get(roadClass);
    if (buffers) return buffers;
    const set = getRoadSet(roadClass);
    buffers = {
      arterial: resources.createBuffer(`${roadClass}-arterial-segments`, set.arterialSegments),
      arterialCount: set.arterialSegments.length / 4,
      local: resources.createBuffer(`${roadClass}-local-segments`, set.localSegments),
      localCount: set.localSegments.length / 4,
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

  const allStreets = getRoadSet('all');
  const streetVertices: StreetVertices = {
    positions: allStreets.positionsMeters,
    pathOffsets: allStreets.pathOffsets,
    pathClass: allStreets.pathClass
  };

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

  function getMaximumRecords(road: RoadSet): number {
    return Math.max(1024, 4 * road.vertexCount);
  }

  function buildDensityGraph(
    id: string,
    roadClass: RoadClass,
    system: StreetDensityOptions['system'],
    parameters: GPUParameterBuffer<'float32'>,
    outputs: {lengths: Buf; densities: Buf; overflow: Buf; totalRecords: Buf}
  ): CompiledGPUCommandGraph<void> {
    const road = getRoadSet(roadClass);
    const buffers = getRoadBuffers(roadClass);
    const spherical = system === 'spherical';
    const graph = new GPUCommandGraph<void>(device, {id: `line-density-${id}`});
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
        spatialContext:
          system === 'spherical'
            ? {coordinateSpace: 'longitude-latitude', metric: 'great-circle', units: 'meters'}
            : {coordinateSpace: 'planar', metric: 'native', units: 'native'},
        maximumRecords: getMaximumRecords(road),
        parameters: parameters.importToGraph(graph),
        output: {
          lengths: importGraphBuffer(graph, 'lengths', outputs.lengths, 'float32', CELL_COUNT),
          densities: importGraphBuffer(
            graph,
            'densities',
            outputs.densities,
            'float32',
            CELL_COUNT
          ),
          overflow: importGraphBuffer(graph, 'overflow', outputs.overflow, 'uint32', 1),
          totalRecords: importGraphBuffer(graph, 'total-records', outputs.totalRecords, 'uint32', 1)
        }
      })
    );
    return resources.track(graph.compile());
  }

  function getDensityVariant(options: StreetDensityOptions): DensityVariant {
    const key = `${options.roadClass}|${options.system}`;
    let variant = densityVariants.get(key);
    if (variant) return variant;
    variant = {
      compiled: buildDensityGraph(key, options.roadClass, options.system, densityParameters, {
        lengths,
        densities,
        overflow: densityOverflow,
        totalRecords
      })
    };
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
  const polygonGeojson: Record<StreetDensityOptions['polygonSet'], GeoJsonCollection> = {
    areas: areasData.geojson ?? {type: 'FeatureCollection', features: []},
    tracts: tractsData.geojson ?? {type: 'FeatureCollection', features: []}
  };
  const polygonStates = new Map<StreetDensityOptions['polygonSet'], PolygonState>();
  const polygonVariants = new Map<string, PolygonVariant>();
  /** Natural breaks per (set, road class, metric, weight): computed once from the readback, then frozen. */
  const frozenClassings = new Map<string, PolygonClassing>();

  function getPolygonState(id: StreetDensityOptions['polygonSet']): PolygonState {
    let state = polygonStates.get(id);
    if (state) return state;
    const set = polygonSets[id];
    const featureCount = set.featureOffsets.length - 1;
    const geojson = polygonGeojson[id];
    const mesh = buildPolygonMesh(geojson, (longitude, latitude) =>
      projection.project(longitude, latitude)
    );
    const noData = new Float32Array(featureCount + 1).fill(Number.NaN);
    const noDataCounts = new Uint32Array(featureCount + 1).fill(NO_DATA_VALUE);
    const next: PolygonState = {
      set,
      mesh,
      meshBuffers: createPolygonMeshBuffers(resources, mesh, `${id}-mesh`),
      locator: createFeatureLocator(geojson),
      geojson,
      featureCount,
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
        meanAttribute: new Float32Array(featureCount),
        segments: new Float32Array(featureCount)
      },
      ready: false,
      inside: 0,
      overflowFlag: false
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
      bytes => handlePolygonSummary(id, next, bytes)
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
    // Derived metrics, written for every polygon so the metric option only picks a buffer. A rate
    // per resident is not defined (NaN, drawn as "no data") where too few people live.
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
  let noData = bitcast<f32>(0x7fc00000u | (index & 0u));
  perResident[perResidentOffset + index] = select(noData, length / residents, residents >= ${SUPPRESSION_RESIDENTS}.0 && length > 0.0);
  meanAttribute[meanAttributeOffset + index] = select(noData, weighted[weightedOffset + index] / length, length > 0.0);`
    });
    variant = {compiled: resources.track(graph.compile())};
    polygonVariants.set(key, variant);
    return variant;
  }

  // ---- State ----
  let destroyed = false;
  let measuring = false;
  let dirty = true;
  let needsRead = false;
  let polygonDirty = true;
  let lastSignature = '';
  let displayedDensity: DensityVariant | null = null;
  let displayedPolygons: PolygonVariant | null = null;
  let viewWidth = 1;
  let lastViewBounds: [number, number, number, number] | null = null;
  /** The grid parameters last written, the grid of the last encode, and the grid a pending readback holds. */
  let currentGrid: Grid | null = null;
  let encodedGrid: Grid | null = null;
  let readGrid: Grid | null = null;
  let snapshot: FieldSnapshot | null = null;
  let hoveredCell = -1;
  let legendHighlight: readonly number[] | null = null;
  let polygonLegendHighlight: readonly number[] | null = null;
  let lastGround: StreetGround = ctx.ground();

  const densityReader = new SummaryReader(
    resources,
    'density',
    [
      {buffer: lengths, size: CELL_COUNT * 4},
      {buffer: densities, size: CELL_COUNT * 4},
      {buffer: densityOverflow, size: 4},
      {buffer: totalRecords, size: 4}
    ],
    bytes => handleDensity(bytes)
  );

  // ---- Standing facts from the street vertices (CPU, once) ----
  const bearings = getBearingStatistics(streetVertices);
  const corridorCorners = [
    projection.project(MILE_CORRIDOR[0], MILE_CORRIDOR[1]),
    projection.project(MILE_CORRIDOR[2], MILE_CORRIDOR[3])
  ];
  const profile: ArterialProfile = getArterialProfile(streetVertices, [
    Math.min(corridorCorners[0][0], corridorCorners[1][0]),
    Math.min(corridorCorners[0][1], corridorCorners[1][1]),
    Math.max(corridorCorners[0][0], corridorCorners[1][0]),
    Math.max(corridorCorners[0][1], corridorCorners[1][1])
  ]);
  const squareMile = squareMileFrame([MILE_VIEW.longitude, MILE_VIEW.latitude]);
  /** Rings of the city limit, largest first, as GeoJSON-style rings for the dashed "data ends" frame. */
  const cityRings: LngLat[][] = city.cityPolygons
    .map(polygon => polygon[0].map(vertex => [vertex[0], vertex[1]] as LngLat))
    .sort((a, b) => b.length - a.length);

  ctx.setReadout('streetKm', `${formatCount(bearings.totalMeters / 1000)} km`);
  ctx.setReadout('gridShare', formatPercent(bearings.gridShare, 0));
  ctx.setReadout(
    'mileShare',
    `${formatPercent(profile.mileShare, 0)} (chance ${formatPercent(profile.mileChance, 0)})`
  );
  ctx.setReadout(
    'halfMileShare',
    `${formatPercent(profile.halfMileShare, 0)} (chance ${formatPercent(profile.halfMileChance, 0)})`
  );
  ctx.setReadout('density', `${COLUMNS} x ${ROWS} = ${formatCount(CELL_COUNT)} cells`);
  ctx.setLegendData('ground', lastGround);
  publishBearingChart();
  publishMileChart();

  /** The bearing rose: street length by direction, the two grid axes in the accent. */
  function publishBearingChart(): void {
    const half = ROSE_BIN_COUNT / 2;
    const quarter = ROSE_BIN_COUNT / 4;
    ctx.setChart('bearingRose', {
      kind: 'rose',
      title: 'Street length by bearing',
      values: bearings.rose,
      labels: Array.from(
        {length: ROSE_BIN_COUNT},
        (_, index) => ['N', 'E', 'S', 'W'][index / quarter] ?? ''
      ),
      colors: bearings.rose.map((_, index) =>
        index % quarter === 0 && (index % half === 0 || index % half === quarter)
          ? ROSE_ACCENT
          : ROSE_NEUTRAL
      ),
      table: false,
      description:
        'Rose diagram of street length by bearing in 10 degree bins: almost all of it points north-south or east-west.'
    });
  }

  /** The east-west arterial profile: length by position from south to north, with the fitted mile lattice. */
  function publishMileChart(): void {
    const spanKm = (profile.bins.length * profile.binMeters) / 1000;
    const markers: {x: number; label?: string}[] = [];
    for (let at = profile.milePhaseMeters; at < spanKm * 1000; at += MILE_METERS) {
      markers.push(markers.length === 0 ? {x: at / 1000, label: 'mile lattice'} : {x: at / 1000});
    }
    ctx.setChart('mileGrid', {
      kind: 'histogram',
      title: 'East-west arterials by position',
      values: profile.bins,
      xDomain: [0, spanKm],
      markers,
      xLabel: 'Distance north (km)',
      yLabel: 'Arterial km per 100 m',
      formatX: value => value.toFixed(0),
      table: false,
      description:
        'East-west primary and secondary roads west of the Loop, counted by north-south position in 100 metre bins. Peaks fall on a one-mile lattice on the South Side and on a half-mile lattice in the north.'
    });
  }

  // ---- Grid parameters ----
  /** Writes the grid origin and cell size in the units of the chosen coordinate system; returns the grid in metres. */
  function writeDensityParameters(
    options: StreetDensityOptions,
    system: 'planar' | 'spherical',
    bounds: readonly [number, number, number, number]
  ): Grid {
    const size = options.cellSize;
    const centerX = (bounds[0] + bounds[2]) / 2;
    const centerY = (bounds[1] + bounds[3]) / 2;
    if (system === 'planar') {
      const minX = Math.floor((centerX - (COLUMNS * size) / 2) / size) * size;
      const minY = Math.floor((centerY - (ROWS * size) / 2) / size) * size;
      densityParameters.write(
        getGPULineDensityParameterValues({minX, minY, cellWidth: size, cellHeight: size})
      );
      const grid = {
        minX,
        minY,
        maxX: minX + COLUMNS * size,
        maxY: minY + ROWS * size,
        cellMeters: size
      };
      gridBounds.write(Float32Array.of(grid.minX, grid.minY, grid.maxX, grid.maxY));
      return grid;
    }
    const cellWidth = size / metersPerDegreeLongitude;
    const cellHeight = size / metersPerDegreeLatitude;
    const [centerLongitude, centerLatitude] = projection.unproject(centerX, centerY);
    const minimumLongitude =
      Math.floor((centerLongitude - origin[0] - (COLUMNS * cellWidth) / 2) / cellWidth) *
        cellWidth +
      origin[0];
    const minimumLatitude =
      Math.floor((centerLatitude - origin[1] - (ROWS * cellHeight) / 2) / cellHeight) * cellHeight +
      origin[1];
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
    return {minX: x0, minY: y0, maxX: x1, maxY: y1, cellMeters: size};
  }

  // ---- The density readback ----
  function getTable(mode: 'density' | 'length' = ctx.options.cellValue): ClassTable {
    return makeDensityTable(lastGround, mode);
  }

  function handleDensity(bytes: ArrayBuffer): void {
    if (destroyed || !readGrid) return;
    const grid = readGrid;
    const floats = new Float32Array(bytes, 0, CELL_COUNT * 2);
    const words = new Uint32Array(bytes, CELL_COUNT * 8, 2);
    const cellLengths = floats.slice(0, CELL_COUNT);
    const cellDensities = floats.slice(CELL_COUNT);
    const cityMask = rasterizeRingsMask(
      city.cityRingsMeters,
      [grid.minX, grid.minY, grid.maxX, grid.maxY],
      COLUMNS,
      ROWS
    );
    const stats = getFieldStatistics(cellDensities, cityMask, DENSITY_BREAKS, HISTOGRAM_TOP);
    const cityLengths: number[] = [];
    let maximumLength = 0;
    for (let index = 0; index < CELL_COUNT; index++) {
      if (!cityMask[index] || !(cellLengths[index] > 0)) continue;
      cityLengths.push(cellLengths[index]);
      maximumLength = Math.max(maximumLength, cellLengths[index]);
    }
    snapshot = {
      grid,
      lengths: cellLengths,
      densities: cellDensities,
      cityMask,
      stats,
      lengthClassCounts: getClassCounts(cityLengths, LENGTH_BREAKS),
      maximumLength
    };
    const road = getRoadSet(ctx.options.roadClass);
    ctx.setReadout('densityPeak', stats.peak > 0 ? `${stats.peak.toFixed(1)} km per km²` : null);
    ctx.setReadout('cellEdge', formatDistance(grid.cellMeters));
    ctx.setReadout('cellLength', formatDistance(maximumLength));
    ctx.setReadout(
      'records',
      `${formatCount(words[1])} pieces of ${formatCount(getMaximumRecords(road))}`
    );
    ctx.setReadout('overflow', words[0] ? 'yes (lengths low)' : 'no');
    ctx.setLegendData('field', {
      counts: stats.classCounts,
      lengthCounts: snapshot.lengthClassCounts
    });
    publishHistogram();
    updateFieldAnnotations();
    ctx.requestLayers();
  }

  /** The distribution of city cells over density, in the colours and breaks of the map. */
  function publishHistogram(): void {
    if (!snapshot || snapshot.stats.sorted.length === 0) {
      ctx.setChart('densityHistogram', null);
      return;
    }
    const table = getTable('density');
    ctx.setChart('densityHistogram', {
      kind: 'histogram',
      title: 'City cells by street density',
      values: snapshot.stats.histogram,
      xDomain: [0, HISTOGRAM_TOP],
      breaks: table.breaks,
      classColors: table.colors,
      now:
        hoveredCell >= 0
          ? Math.min(snapshot.densities[hoveredCell] * 1000, HISTOGRAM_TOP)
          : undefined,
      nowLabel: hoveredCell >= 0 ? 'This cell' : undefined,
      xLabel: 'km of street per km²',
      yLabel: 'Cells',
      table: false,
      description:
        'Histogram of the city cells that hold any street, coloured by the map classes: most fall in the middle classes, with a thin tail of very dense cells.'
    });
  }

  // ---- Notes on the map ----
  /** The gap and peak notes of the field steps, from the last readback. */
  function updateFieldAnnotations(): void {
    const options = ctx.options;
    const current =
      snapshot && currentGrid && sameGrid(snapshot.grid, currentGrid) ? snapshot : null;
    const showField = options.view === 'density' && options.layers !== 'streets';
    if (
      !current ||
      !showField ||
      options.cellValue !== 'density' ||
      options.cellSize < NOTE_MINIMUM_CELL_METERS
    ) {
      ctx.setAnnotations('field-notes', null);
      return;
    }
    const {grid, stats} = current;
    const notes: MapAnnotation[] = [];
    if (stats.peakIndex >= 0) {
      const column = stats.peakIndex % COLUMNS;
      const row = Math.floor(stats.peakIndex / COLUMNS);
      const coordinate = unproject(
        grid.minX + (column + 0.5) * grid.cellMeters,
        grid.minY + (row + 0.5) * grid.cellMeters
      );
      notes.push({
        kind: 'note',
        id: 'densest-cell',
        coordinate,
        title: `${stats.peak.toFixed(1)} km of street per km²`,
        text: `Densest cell, ${nearestPlaceLabel(CHICAGO, coordinate, {maxDistanceMeters: 8000}) ?? 'in the city'}`,
        tone: 'accent',
        priority: 5
      });
    }
    const minimumCells = Math.max(2, Math.ceil(VOID_MINIMUM_AREA / grid.cellMeters ** 2));
    const voids = findVoids(
      current.densities,
      current.cityMask,
      COLUMNS,
      ROWS,
      [grid.minX, grid.minY, grid.maxX, grid.maxY],
      VOID_DENSITY,
      minimumCells
    ).slice(0, VOID_NOTE_COUNT);
    voids.forEach((region, rank) => {
      const coordinate = unproject(region.x, region.y);
      const place = nearestPlace(CHICAGO, coordinate, {
        maxDistanceMeters: 4000,
        kinds: VOID_PLACE_KINDS
      });
      const outsideExtract = coordinate[0] < roadsWest + 0.01;
      notes.push({
        kind: 'note',
        id: `gap-${rank}`,
        coordinate,
        title: place
          ? `${place.name}: ${outsideExtract ? 'outside the street extract' : 'almost no streets'}`
          : 'A gap in the street grid',
        text: outsideExtract
          ? 'The street data stops here: no data, not zero'
          : `${formatArea(region.cells * grid.cellMeters ** 2)} under ${VOID_DENSITY} km of street per km²`,
        tone: 'ink',
        priority: 4 - rank
      });
    });
    ctx.setAnnotations('field-notes', notes.length ? notes : null);
  }

  /** The dashed city limit with its "no data outside" statement, drawn with the field. */
  function updateEdgeAnnotations(): void {
    const options = ctx.options;
    if (options.view !== 'density' || options.layers === 'streets') {
      ctx.setAnnotations('city-limit', null);
      return;
    }
    ctx.setAnnotations(
      'city-limit',
      cityRings.map((ring, index) => ({
        kind: 'frame' as const,
        id: `city-limit-${index}`,
        ring,
        text:
          index === 0 ? 'Streets end at the city limit: outside is no data, not zero' : undefined
      }))
    );
  }

  /** One square mile at the centre of the mile-grid step: the size of a Public Land Survey section. */
  function updateSquareMile(): void {
    const options = ctx.options;
    if (options.view !== 'density' || options.streetEmphasis !== 'arterials') {
      ctx.setAnnotations('square-mile', null);
      return;
    }
    ctx.setAnnotations('square-mile', [
      {kind: 'outline', id: 'square-mile', rings: [squareMile], text: '1 square mile'}
    ]);
  }

  // ---- Polygon summaries ----
  function getClassingKey(options: StreetDensityOptions): string {
    return `${options.polygonSet}|${options.roadClass}|${options.metric}|${
      options.metric === 'meanAttribute' ? options.weight : '-'
    }`;
  }

  /** Display values that enter a classification: finite numbers only (suppressed tracts are left out). */
  function getFiniteValues(state: PolygonState, metric: StreetDensityOptions['metric']): number[] {
    return Array.from(state.values[metric]).filter(value => Number.isFinite(value));
  }

  /** Natural breaks for the current set and metric, computed once from the readback and then frozen. */
  function getClassing(state: PolygonState, options: StreetDensityOptions): PolygonClassing {
    const key = getClassingKey(options);
    const frozen = frozenClassings.get(key);
    if (frozen) return frozen;
    const values = getFiniteValues(state, options.metric);
    const {digits} = getMetricInfo(options.metric, options.weight);
    const precision = 10 ** digits;
    const breaks = [
      ...new Set(
        getClassBreaks(values, POLYGON_CLASS_COUNT, 'natural-breaks').map(
          value => Math.round(value * precision) / precision
        )
      )
    ].sort((a, b) => a - b);
    const classing = {
      breaks,
      goodness: getGoodnessOfVarianceFit(values, breaks),
      counts: getClassCounts(values, breaks)
    };
    frozenClassings.set(key, classing);
    return classing;
  }

  function getPolygonTable(state: PolygonState, options: StreetDensityOptions): ClassTable {
    const classing = getClassing(state, options);
    const info = getMetricInfo(options.metric, options.weight);
    const suppressed = options.metric === 'perResident';
    return makeClassTable({
      breaks: classing.breaks,
      colors: pickColors(getDensityColors(lastGround), classing.breaks.length + 1),
      unit: info.unit,
      method: `Natural breaks (Jenks), GVF ${classing.goodness.toFixed(2)}`,
      format: value => (info.digits === 0 ? formatCount(value) : value.toFixed(info.digits)),
      noData: {
        label: suppressed
          ? `Fewer than ${SUPPRESSION_RESIDENTS} residents: rate unstable`
          : 'No street data',
        hatched: suppressed
      }
    });
  }

  function countSuppressed(state: PolygonState): number {
    let count = 0;
    for (let feature = 0; feature < state.featureCount; feature++) {
      if (Number.isNaN(state.values.perResident[feature]) && state.values.length[feature] > 0)
        count++;
    }
    return count;
  }

  function handlePolygonSummary(
    id: StreetDensityOptions['polygonSet'],
    state: PolygonState,
    bytes: ArrayBuffer
  ): void {
    if (destroyed) return;
    const n = state.featureCount;
    state.values.length = Float32Array.from(new Float32Array(bytes.slice(0, n * 4)), v => v / 1000);
    state.values.perArea = new Float32Array(bytes.slice(n * 4, n * 8));
    state.values.perResident = new Float32Array(bytes.slice(n * 8, n * 12));
    state.values.meanAttribute = new Float32Array(bytes.slice(n * 12, n * 16));
    state.values.segments = Float32Array.from(new Uint32Array(bytes.slice(n * 16, n * 20)), v =>
      v === NO_DATA_VALUE ? Number.NaN : v
    );
    state.overflowFlag = new Uint32Array(bytes, n * 20, 1)[0] !== 0;
    state.inside = 0;
    for (let row = 0; row < n; row++) state.inside += state.values.length[row] * 1000;
    state.ready = true;
    if (ctx.options.polygonSet === id) publishPolygons();
  }

  /** Readouts, legend data, rank-flip chart and notes of the displayed polygon map. */
  function publishPolygons(): void {
    const options = ctx.options;
    const road = getRoadSet(options.roadClass);
    ctx.setReadout(
      'polygons',
      `${formatCount(polygonSets[options.polygonSet].featureOffsets.length - 1)} ${options.polygonSet === 'areas' ? 'community areas' : 'census tracts'}`
    );
    const state = polygonStates.get(options.polygonSet);
    if (!state?.ready || options.view !== 'polygons') {
      for (const readout of ['inside', 'conservation', 'polygonOverflow'])
        ctx.setReadout(
          readout,
          options.view === 'polygons' ? 'computing…' : 'switch to the polygon view'
        );
      return;
    }
    const percent = (100 * state.inside) / road.lengthMeters;
    ctx.setReadout(
      'inside',
      `${formatCount(state.inside / 1000)} km of ${formatCount(road.lengthMeters / 1000)} km (${percent.toFixed(2)}%)`
    );
    ctx.setReadout(
      'conservation',
      Math.abs(percent - 100) < 1
        ? `conserved (${(percent - 100).toFixed(2)}%)`
        : `${(100 - percent).toFixed(1)}% of street length is outside the polygons or lost in simplification`
    );
    ctx.setReadout('polygonOverflow', state.overflowFlag ? 'yes (lengths may be low)' : 'no');

    // Legend data: the frozen classification as a table on the current ground.
    const classing = getClassing(state, options);
    const table = getPolygonTable(state, options);
    const info = getMetricInfo(options.metric, options.weight);
    const suppressed = countSuppressed(state);
    const legend: PolygonLegendData = {
      key: `${options.polygonSet}|${options.roadClass}|${options.metric}|${options.weight}`,
      table: {
        ...table,
        noData: {...table.noData, count: options.metric === 'perResident' ? suppressed : undefined}
      },
      counts: classing.counts,
      title: info.title,
      note: `${table.method}. Breaks are fixed once for this map.`,
      suppressed
    };
    ctx.setLegendData('polygon', legend);

    // The two leaders and the chart that shows the ranking flip (community areas).
    const lengthKm = state.values.length;
    const perArea = state.values.perArea;
    const topBy = (values: Float32Array) => {
      let best = -1;
      for (let feature = 0; feature < values.length; feature++) {
        if (Number.isFinite(values[feature]) && (best < 0 || values[feature] > values[best]))
          best = feature;
      }
      return best;
    };
    const nameOf = (feature: number) =>
      options.polygonSet === 'areas'
        ? state.set.names[feature]
        : (nearestPlaceLabel(CHICAGO, state.mesh.labelPoints[feature] as LngLat, {
            maxDistanceMeters: 8000
          }) ?? state.set.names[feature]);
    const topLength = topBy(lengthKm);
    const topDensity = topBy(perArea);
    ctx.setReadout(
      'topLength',
      topLength >= 0 ? `${nameOf(topLength)} (${formatCount(lengthKm[topLength])} km)` : null
    );
    ctx.setReadout(
      'topDensity',
      topDensity >= 0
        ? `${nameOf(topDensity)} (${perArea[topDensity].toFixed(1)} km per km²)`
        : null
    );
    ctx.setReadout(
      'suppressed',
      options.polygonSet === 'tracts'
        ? `${formatCount(suppressed)} tracts with fewer than ${SUPPRESSION_RESIDENTS} residents hatched`
        : null
    );
    const rates = getFiniteValues(state, 'perResident').sort((a, b) => a - b);
    ctx.setReadout(
      'residentMedian',
      rates.length ? `${rates[Math.floor(rates.length / 2)].toFixed(1)} m per resident` : null
    );
    publishRankFlip(state, options);
    publishPolygonNotes(state, options, topLength, topDensity, nameOf);
  }

  /** Slope chart of the community areas' rank by length against rank by length per area. */
  function publishRankFlip(state: PolygonState, options: StreetDensityOptions): void {
    if (options.polygonSet !== 'areas') {
      ctx.setChart('rankFlip', null);
      return;
    }
    const count = state.featureCount;
    const {rows, movers} = getRankFlipRows(
      state.values.length,
      state.values.perArea,
      RANK_FLIP_TOP,
      3
    );
    const maximumRank = Math.max(...rows.flatMap(row => [row.rankByLength, row.rankByDensity]));
    ctx.setChart('rankFlip', {
      kind: 'slope',
      title: 'Community areas: rank flips with the denominator',
      rows: rows.map(row => ({
        label: state.set.names[row.feature],
        a: count + 1 - row.rankByLength,
        b: count + 1 - row.rankByDensity,
        highlight: movers.has(row.feature)
      })),
      aLabel: 'By length',
      bLabel: 'By length per km²',
      yDomain: [count + 1 - maximumRank - 1, count],
      formatY: value => formatOrdinal(count + 1 - value),
      table: false,
      description:
        'Slopegraph of the community areas in the top ten by street length or by street length per square kilometre, joined by line between the two ranks; the biggest movers are highlighted.'
    });
  }

  /** The leader of each ranking, and the leader of the displayed metric, as notes at their label points. */
  function publishPolygonNotes(
    state: PolygonState,
    options: StreetDensityOptions,
    topLength: number,
    topDensity: number,
    nameOf: (feature: number) => string
  ): void {
    if (options.view !== 'polygons') {
      ctx.setAnnotations('polygon-notes', null);
      return;
    }
    const notes: MapAnnotation[] = [];
    const add = (feature: number, title: string, text: string, tone: 'accent' | 'ink') => {
      const point = state.mesh.labelPoints[feature];
      if (feature < 0 || !point || !Number.isFinite(point[0])) return;
      notes.push({
        kind: 'note',
        id: `leader-${notes.length}`,
        coordinate: [point[0], point[1]],
        title,
        text,
        tone,
        priority: 5 - notes.length
      });
    };
    const info = getMetricInfo(options.metric, options.weight);
    if (
      options.polygonSet === 'areas' &&
      (options.metric === 'length' || options.metric === 'perArea')
    ) {
      add(
        topLength,
        `${nameOf(topLength)}: most street`,
        `${formatCount(state.values.length[topLength])} km in all`,
        options.metric === 'length' ? 'accent' : 'ink'
      );
      add(
        topDensity,
        `${nameOf(topDensity)}: densest`,
        `${state.values.perArea[topDensity].toFixed(1)} km per km²`,
        options.metric === 'perArea' ? 'accent' : 'ink'
      );
    } else {
      let leader = -1;
      const values = state.values[options.metric];
      for (let feature = 0; feature < values.length; feature++) {
        if (Number.isFinite(values[feature]) && (leader < 0 || values[feature] > values[leader]))
          leader = feature;
      }
      if (leader >= 0) {
        add(leader, `${nameOf(leader)}: highest`, info.format(values[leader]), 'accent');
      }
    }
    ctx.setAnnotations('polygon-notes', notes.length ? notes : null);
  }

  // ---- Variants ----
  function ensureVariants(): void {
    const options = ctx.options;
    displayedDensity = getDensityVariant(options);
    if (options.view === 'polygons') displayedPolygons = getPolygonVariant(options);
    else
      displayedPolygons =
        polygonVariants.get(`${options.polygonSet}|${options.roadClass}|${options.system}`) ?? null;
    const road = getRoadSet(options.roadClass);
    const buffers = getRoadBuffers(options.roadClass);
    ctx.setReadout(
      'segments',
      `${formatCount(road.pathCount)} streets, ${formatCount(buffers.arterialCount + buffers.localCount)} segments`
    );
    ctx.setCost({
      records: buffers.arterialCount + buffers.localCount,
      passes: (options.view === 'polygons' ? displayedPolygons : displayedDensity)?.compiled.stats
        .nodeOrder.length
    });
    markDirty();
  }

  function markDirty(): void {
    dirty = true;
    polygonDirty = true;
  }

  ensureVariants();
  publishPolygons();
  updateEdgeAnnotations();
  updateSquareMile();

  // ---- The cell-size sweep (MAUP scale) ----
  /** Where the sweep looks: the grid is centred on the Loop and the window is the 60 m grid's extent. */
  const loopCenter = projection.project(
    CHICAGO.places.loop.lngLat[0],
    CHICAGO.places.loop.lngLat[1]
  );
  const sweepHalfWidth = (COLUMNS * CELL_SWEEP_SIZES[0]) / 2;
  const sweepHalfHeight = (ROWS * CELL_SWEEP_SIZES[0]) / 2;
  type Sweep = {
    graph: CompiledGPUCommandGraph<void>;
    parameters: GPUParameterBuffer<'float32'>;
    reader: SummaryReader;
  };
  let sweep: Sweep | null = null;
  let sweepState: 'idle' | 'compiling' | 'running' | 'done' = 'idle';
  let sweepIndex = 0;
  let sweepWaiting = false;
  let sweepGrid: Grid | null = null;
  const sweepBusiest: number[] = [];
  const sweepTypical: number[] = [];

  /** Compiles the second density graph (own outputs) once, after the first change of the cell size. */
  function startSweep(): void {
    if (sweepState !== 'idle' || destroyed) return;
    sweepState = 'compiling';
    setTimeout(() => {
      if (destroyed) return;
      const parameters = resources.createParameterBuffer(
        'sweep-parameters',
        'float32',
        GPU_LINE_DENSITY_PARAMETER_LENGTH
      );
      const sweepLengths = resources.createBuffer('sweep-lengths', CELL_COUNT * 4);
      const sweepDensities = resources.createBuffer('sweep-densities', CELL_COUNT * 4);
      const sweepOverflow = resources.createBuffer('sweep-overflow', 4);
      const sweepRecords = resources.createBuffer('sweep-records', 4);
      const graph = buildDensityGraph('sweep', 'all', 'planar', parameters, {
        lengths: sweepLengths,
        densities: sweepDensities,
        overflow: sweepOverflow,
        totalRecords: sweepRecords
      });
      const reader = new SummaryReader(
        resources,
        'sweep',
        [{buffer: sweepDensities, size: CELL_COUNT * 4}],
        bytes => handleSweep(bytes)
      );
      sweep = {graph, parameters, reader};
      sweepIndex = 0;
      sweepBusiest.length = 0;
      sweepTypical.length = 0;
      sweepState = 'running';
      ctx.setReadout('sweepCost', 'compiled once, 6 runs');
      ctx.requestLayers();
    }, 0);
  }

  /** Writes the grid of the next sweep size (centred on the Loop) and returns it. */
  function writeSweepParameters(size: number): Grid {
    const minX = Math.floor((loopCenter[0] - (COLUMNS * size) / 2) / size) * size;
    const minY = Math.floor((loopCenter[1] - (ROWS * size) / 2) / size) * size;
    sweep?.parameters.write(
      getGPULineDensityParameterValues({minX, minY, cellWidth: size, cellHeight: size})
    );
    return {minX, minY, maxX: minX + COLUMNS * size, maxY: minY + ROWS * size, cellMeters: size};
  }

  /** Busiest and median density of the city cells inside the Loop window at this cell size. */
  function handleSweep(bytes: ArrayBuffer): void {
    if (destroyed || !sweepGrid) return;
    const grid = sweepGrid;
    const cellDensities = new Float32Array(bytes);
    const values: number[] = [];
    for (let row = 0; row < ROWS; row++) {
      const y = grid.minY + (row + 0.5) * grid.cellMeters;
      if (Math.abs(y - loopCenter[1]) > sweepHalfHeight) continue;
      for (let column = 0; column < COLUMNS; column++) {
        const x = grid.minX + (column + 0.5) * grid.cellMeters;
        if (Math.abs(x - loopCenter[0]) > sweepHalfWidth) continue;
        const density = cellDensities[row * COLUMNS + column] * 1000;
        if (density > 0 && city.containsMeters(x, y)) values.push(density);
      }
    }
    values.sort((a, b) => a - b);
    sweepBusiest.push(values.length ? values[values.length - 1] : 0);
    sweepTypical.push(values.length ? values[Math.floor(values.length / 2)] : 0);
    sweepIndex++;
    sweepWaiting = false;
    if (sweepIndex >= CELL_SWEEP_SIZES.length) {
      sweepState = 'done';
      publishSweepChart();
    }
  }

  function publishSweepChart(): void {
    const sizes = [...CELL_SWEEP_SIZES];
    ctx.setChart('cellSizeCurve', {
      kind: 'line',
      title: 'Peaks fall as cells grow',
      xLabel: 'Cell size (m)',
      yLabel: 'km of street per km²',
      xDomain: [50, sizes[sizes.length - 1]],
      series: [
        {label: 'Busiest cell', x: sizes, y: sweepBusiest, points: true},
        {label: 'Typical cell', x: sizes, y: sweepTypical, points: true, dashed: true}
      ],
      link: {option: 'cellSize', label: value => `${value} m`},
      table: false,
      description:
        'Street density of the busiest and of the typical city cell around the Loop for six cell sizes: the peak falls steeply as the cell grows while the typical cell hardly moves.'
    });
  }

  /** Encodes one sweep run per frame while the sweep is running. */
  function encodeSweep(commandEncoder: CommandEncoder): void {
    if (!sweep || sweepState !== 'running') return;
    if (sweepWaiting) {
      sweep.reader.request(commandEncoder);
      return;
    }
    sweepGrid = writeSweepParameters(CELL_SWEEP_SIZES[sweepIndex]);
    sweep.graph.encode(commandEncoder, {parameters: undefined});
    sweepWaiting = true;
    sweep.reader.request(commandEncoder);
  }

  // ---- Furniture ----
  const runtimeFurniture: {
    title: {sample: string};
    scaleBar: {units: 'metric'; ticks?: number[]};
  } = {
    title: {
      sample: `${formatCount(allStreets.lengthMeters / 1000)} km of drivable streets, OpenStreetMap`
    },
    scaleBar: {units: 'metric'}
  };
  ctx.setFurniture(runtimeFurniture);
  let lastTick: number | null | undefined;
  updateScaleTick();

  /** The scale bar is ticked at the cell size while the field is on the map. */
  function updateScaleTick(): void {
    const options = ctx.options;
    const tick =
      options.view === 'density' && options.layers !== 'streets' ? options.cellSize : null;
    if (tick === lastTick) return;
    lastTick = tick;
    runtimeFurniture.scaleBar = {units: 'metric', ticks: tick ? [tick] : undefined};
    ctx.setFurniture(runtimeFurniture);
  }

  function sameGrid(a: Grid, b: Grid): boolean {
    return a.minX === b.minX && a.minY === b.minY && a.cellMeters === b.cellMeters;
  }

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
      const bounds = lastViewBounds ?? [0, 0, 1000, 1000];
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
      lastSignature = '';
      markDirty();
    }
  }

  // ---- Tooltips ----
  function getDensityTooltip(coordinate: readonly [number, number]): TooltipContent | null {
    const options = ctx.options;
    const [x, y] = projection.project(coordinate[0], coordinate[1]);
    const clear = () => {
      if (hoveredCell !== -1) {
        hoveredCell = -1;
        ctx.setLegendData('marker', null);
        publishHistogram();
      }
      return null;
    };
    if (!snapshot || !currentGrid || !sameGrid(snapshot.grid, currentGrid)) return clear();
    const {grid} = snapshot;
    const column = Math.floor((x - grid.minX) / grid.cellMeters);
    const row = Math.floor((y - grid.minY) / grid.cellMeters);
    if (column < 0 || row < 0 || column >= COLUMNS || row >= ROWS) return clear();
    const index = row * COLUMNS + column;
    if (!snapshot.cityMask[index]) {
      clear();
      return {
        title: 'Outside the city',
        note: 'No data: the street extract ends at the city limit, so this is not a zero.'
      };
    }
    const lengthMeters = snapshot.lengths[index];
    const density = snapshot.densities[index] * 1000;
    if (!(lengthMeters > 0)) {
      clear();
      return {
        title: `${formatDistance(grid.cellMeters)} cell`,
        note: 'No street in this cell (left clear on the map).'
      };
    }
    const mode = options.cellValue;
    const table = getTable(mode);
    const value = mode === 'density' ? density : lengthMeters;
    const classIndex = getClassIndexOf(table, value);
    if (index !== hoveredCell) {
      hoveredCell = index;
      ctx.setLegendData('marker', value);
      publishHistogram();
    }
    const swatch = table.colors[classIndex];
    const percentile =
      countAtOrBelow(snapshot.stats.sorted, density) / snapshot.stats.sorted.length;
    const west = grid.minX + column * grid.cellMeters;
    const south = grid.minY + row * grid.cellMeters;
    const [west0, south0] = unproject(west, south);
    const [east1, north1] = unproject(west + grid.cellMeters, south + grid.cellMeters);
    const rows: TooltipRow[] = [
      mode === 'density'
        ? {
            label: 'Street density',
            value: density.toFixed(1),
            unit: 'km per km²',
            swatch,
            emphasis: true
          }
        : {
            label: 'Street in the cell',
            value: formatDistance(lengthMeters),
            swatch,
            emphasis: true
          },
      mode === 'density'
        ? {label: 'Street in the cell', value: formatDistance(lengthMeters)}
        : {label: 'Street density', value: density.toFixed(1), unit: 'km per km²'},
      {label: 'Class', value: getClassLabel(table, classIndex)},
      {label: 'Rank', value: `${formatOrdinal(percentile * 100)} percentile of cells with street`}
    ];
    return {
      title: `${formatDistance(grid.cellMeters)} cell`,
      subtitle:
        nearestPlaceLabel(
          CHICAGO,
          unproject(west + grid.cellMeters / 2, south + grid.cellMeters / 2),
          {
            maxDistanceMeters: 8000
          }
        ) ?? undefined,
      rows,
      highlight: {kind: 'box', bounds: [west0, south0, east1, north1]}
    };
  }

  function getPolygonTooltip(coordinate: readonly [number, number]): TooltipContent | null {
    const options = ctx.options;
    const state = polygonStates.get(options.polygonSet);
    if (!state?.ready) return null;
    const found = state.locator.find([coordinate[0], coordinate[1]]);
    if (!found) return null;
    const feature = found.index;
    const table = getPolygonTable(state, options);
    const info = getMetricInfo(options.metric, options.weight);
    const value = state.values[options.metric][feature];
    const ranks = getRanks(
      Array.from(state.values[options.metric], v => (Number.isFinite(v) ? v : -Infinity))
    );
    const suppressed = Number.isNaN(value);
    const area = state.set.areas[feature] / 1e6;
    const population = state.set.population[feature];
    const rows: TooltipRow[] = [
      {
        label: info.title,
        value: suppressed ? 'not shown' : info.format(value),
        swatch: suppressed ? undefined : table.colors[getClassIndexOf(table, value)],
        emphasis: true
      }
    ];
    if (!suppressed) {
      rows.push({
        label: 'Rank',
        value: `${formatOrdinal(ranks[feature])} of ${formatCount(state.featureCount)}`
      });
    }
    rows.push(
      {label: 'Street length', value: `${formatCount(state.values.length[feature])} km`},
      {label: 'Per area', value: `${state.values.perArea[feature].toFixed(1)} km per km²`},
      {label: 'Area', value: `${area.toFixed(2)} km²`},
      {
        label: 'Residents',
        value: population > 0 ? formatCount(population) : 'none counted'
      }
    );
    const geometry = state.geojson.features[feature]?.geometry ?? null;
    const rings = getGeometryPolygons(geometry).flat();
    const point = state.mesh.labelPoints[feature];
    return {
      title: state.set.names[feature],
      subtitle: options.polygonSet === 'areas' ? 'Community area' : 'Census tract',
      rows,
      note:
        options.metric === 'perResident' && suppressed
          ? `Fewer than ${SUPPRESSION_RESIDENTS} residents: rate unstable`
          : undefined,
      anchor: point && Number.isFinite(point[0]) ? [point[0], point[1]] : undefined,
      highlight: {kind: 'polygon', rings: rings as unknown as LngLat[][]}
    };
  }

  // ---- The instance ----
  return {
    getCompiledGraphs: () =>
      [displayedDensity?.compiled, displayedPolygons?.compiled].filter(
        Boolean
      ) as unknown as CompiledGPUCommandGraph<never>[],

    setOption(id, _value, state) {
      if (id === 'roadClass' || id === 'system' || id === 'polygonSet' || id === 'view') {
        ensureVariants();
        publishPolygons();
        lastSignature = '';
        if (id === 'view') {
          snapshot = null;
          ctx.setAnnotations('field-notes', null);
        }
      } else if (id === 'weight') {
        for (const [roadClass, buffers] of roadBuffers)
          buffers.pathWeights.write(getRoadSet(roadClass).weights[state.weight]);
        markDirty();
        publishPolygons();
      } else if (id === 'cellSize') {
        lastSignature = '';
        startSweep();
      } else if (id === 'metric') {
        publishPolygons();
      }
      updateEdgeAnnotations();
      updateSquareMile();
      updateFieldAnnotations();
      updateScaleTick();
      ctx.requestLayers();
    },

    onAction(id) {
      if (id === 'measure') void measureGraphs();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    // The class tables are authored per ground, so a ground flip rebuilds them.
    onGroundChange(ground) {
      lastGround = ground;
      ctx.setLegendData('ground', ground);
      publishHistogram();
      publishPolygons();
      ctx.requestLayers();
    },

    onLegendFilter(id, classes) {
      if (id === 'density-classes') legendHighlight = classes;
      else polygonLegendHighlight = classes;
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      const options = ctx.options;
      if (!displayedDensity) return;
      const bounds = getViewportMetricBounds(frame.viewport, projection);
      lastViewBounds = bounds;
      viewWidth = bounds[2] - bounds[0];
      const signature = [
        options.cellSize,
        options.system,
        options.roadClass,
        ...bounds.map(value => Math.round(value / 5))
      ].join(',');
      if (signature !== lastSignature) {
        lastSignature = signature;
        currentGrid = writeDensityParameters(options, options.system, bounds);
        dirty = true;
        if (hoveredCell !== -1) hoveredCell = -1;
        updateScaleTick();
      }
      if (options.view === 'density' && currentGrid) {
        if (dirty) {
          displayedDensity.compiled.encode(commandEncoder, {parameters: undefined});
          encodedGrid = currentGrid;
          needsRead = true;
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
        if (needsRead && !densityReader.isPending) {
          densityReader.request(commandEncoder);
          // `pending` means the copy was recorded in this encoder: it holds the grid just encoded.
          if (densityReader.isPending) {
            readGrid = encodedGrid;
            needsRead = false;
          }
        }
      }
      for (const state of polygonStates.values()) state.reader.flush(commandEncoder);
      if (options.view === 'polygons' && displayedPolygons && polygonDirty) {
        const state = getPolygonState(options.polygonSet);
        displayedPolygons.compiled.encode(commandEncoder, {parameters: undefined});
        state.reader.request(commandEncoder);
        polygonDirty = false;
      }
      if (sweepState === 'idle' && options.cellSize !== 350) startSweep();
      encodeSweep(commandEncoder);
    },

    getLayers() {
      const options = ctx.options;
      const ground = ctx.ground();
      const layers: Layer[] = [];
      if (options.view === 'polygons') {
        const state = getPolygonState(options.polygonSet);
        const lineColor = getPolygonLineColor(ground);
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
        const suppressed = options.metric === 'perResident';
        // The classes are frozen from the first readback, so nothing is filled before it arrives.
        if (state.ready) {
          const table = getPolygonTable(state, options);
          layers.push(
            new SpatialAnalysisPolygonLayer({
              id: `street-polygons-${options.polygonSet}`,
              coordinateOrigin,
              triangles: state.meshBuffers.triangles,
              features: state.meshBuffers.triangleFeatures,
              vertexCount: state.meshBuffers.vertexCount,
              values: metricBuffer,
              valueFormat: options.metric === 'segments' ? 'uint32' : 'float32',
              // Lengths are written in metres; the classes are in kilometres.
              valueScale: options.metric === 'length' ? 0.001 : 1,
              colormap: 'greys',
              ...getClassTableLayerProps(table),
              noDataColor: [0, 0, 0, 0],
              hatchNoData: suppressed,
              hatchColor: ground === 'dark' ? [232, 237, 242, 110] : [31, 41, 51, 110],
              highlightClasses: polygonLegendHighlight,
              opacity: ground === 'dark' ? 0.9 : 0.88
            })
          );
        }
        if (options.showOutlines) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: `street-polygon-outline-${options.polygonSet}`,
              coordinateOrigin,
              segments: state.meshBuffers.outline,
              instanceCount: state.meshBuffers.outlineCount,
              widthPixels: options.polygonSet === 'areas' ? 0.9 : 0.4,
              color: lineColor
            })
          );
        }
        return layers;
      }

      const buffers = getRoadBuffers(options.roadClass);
      const showStreets = options.layers !== 'field';
      const showField = options.layers !== 'streets';
      if (showStreets) {
        // Under a field the lines are context hairlines; alone they are the figure.
        const style = getStreetStyle(ground, options.streetEmphasis, showField);
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'street-density-local',
            coordinateOrigin,
            segments: buffers.local,
            instanceCount: buffers.localCount,
            widthPixels: style.local.widthPixels,
            color: style.local.color
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'street-density-arterial',
            coordinateOrigin,
            segments: buffers.arterial,
            instanceCount: buffers.arterialCount,
            widthPixels: style.arterial.widthPixels,
            color: style.arterial.color
          })
        );
      }
      if (showField) {
        const length = options.cellValue === 'length';
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: 'street-density-cells',
            coordinateOrigin,
            gridSize: [COLUMNS, ROWS],
            bounds: gridBounds.buffer,
            values: length ? lengths : densities,
            valueFormat: 'float32',
            // Densities are metres per square metre; the classes are kilometres per square kilometre.
            valueScale: length ? 1 : 1000,
            colormap: 'greys',
            ...getClassTableLayerProps(getTable()),
            highlightClasses: legendHighlight,
            // Cells with no street are left clear: zero is not a class.
            discardAtOrBelow: 0,
            opacity: getFieldOpacity(ground, options.layers === 'both' ? 'both' : 'field')
          })
        );
        // The edge: outside the city is no data, in the ground colour above the field.
        layers.push(
          new SpatialAnalysisPolygonLayer({
            id: 'street-density-outside',
            coordinateOrigin,
            triangles: outsideBuffers.triangles,
            features: outsideBuffers.triangleFeatures,
            vertexCount: outsideBuffers.vertexCount,
            colormap: 'uniform',
            color: getOutsideColor(ground)
          })
        );
        if (lakeBuffers && ground === 'light') {
          layers.push(
            new SpatialAnalysisPolygonLayer({
              id: 'street-density-lake',
              coordinateOrigin,
              triangles: lakeBuffers.triangles,
              features: lakeBuffers.triangleFeatures,
              vertexCount: lakeBuffers.vertexCount,
              colormap: 'uniform',
              color: LAKE_PAPER_COLOR
            })
          );
        }
      }
      return layers;
    },

    getTooltip(event) {
      const options = ctx.options;
      if (!event.coordinate) return null;
      if (options.view === 'polygons') return getPolygonTooltip(event.coordinate);
      if (options.layers === 'streets') return null;
      return getDensityTooltip(event.coordinate);
    },

    destroy() {
      destroyed = true;
      densityReader.stop();
      sweep?.reader.stop();
      for (const state of polygonStates.values()) state.reader.stop();
      resources.destroy();
    }
  };
}
