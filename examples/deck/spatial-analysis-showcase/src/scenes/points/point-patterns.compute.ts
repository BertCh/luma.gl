// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUPointPatternIndicesParameterValues,
  getGPURipleyDistanceParameterValues,
  getGPURipleyParameterValues,
  GPU_CLARK_EVANS_LENGTH,
  GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH,
  GPU_QUADRAT_STATISTICS_LENGTH,
  GPU_RIPLEY_DISTANCE_PARAMETER_LENGTH,
  GPU_RIPLEY_PARAMETER_LENGTH,
  GPUPointPatternIndices,
  GPURipley,
  GPURipleyDistanceFunctions,
  type GPURipleyDistanceEdgeCorrection,
  type GPURipleyEdgeCorrection
} from '@luma.gl/experimental/gpu-dataframe';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  getClassIndexOf,
  getClassLabel,
  getClassTableLayerProps
} from '../../cartography/class-table';
import {CHICAGO, findPlace, nearestPlaceLabel} from '../../cartography/gazetteer';
import {
  formatArea,
  formatCount,
  formatDistance,
  formatPercent,
  formatRate,
  formatSigned
} from '../../cartography/live-text';
import {createNearestIndex, type NearestIndex} from '../../cartography/picking';
import {buildPolygonMesh, type PolygonMesh} from '../../cartography/polygon-mesh';
import type {ClassTable, LngLat, MapAnnotation} from '../../cartography/types';
import {DENSE_POINT_RADIUS_STOPS} from '../../cartography/zoom';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisPolygonLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer
} from '../../engine/layers';
import {createPolygonMeshBuffers} from '../../engine/polygon-buffers';
import {createSeededRandom} from '../../engine/projection';
import {getViewportMetricBounds, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {formatCompiledGraphTiming, measureCompiledGraph} from '../../engine/vector-timing';
import type {SceneContext, SceneInstance, TooltipContent, TooltipRow} from '../scene';
import {loadCityGeometry} from './b1-city-geometry';
import {formatCategory} from './b1-nature-data';
import {
  getGhostColor,
  getParameterInk,
  getSecondColor,
  getSubjectColor,
  NATURE_SAMPLE,
  type PointsGround
} from './b1-points-look';
import {
  buildClarkEvansGauge,
  buildDistanceChart,
  buildLCurveChart,
  buildQuadratChart,
  formatRadius,
  OBSERVED_SLOT,
  SECOND_SLOT,
  type PreviousCurve
} from './point-patterns-charts';
import {makeNearestTable, makeQuadratTable, type QuadratClasses} from './point-patterns-classes';
import {
  createLandSampler,
  ENVELOPE_RUNS,
  EnvelopeAccumulator,
  fillRandomPositions,
  type LandSampler,
  type PatternBounds
} from './point-patterns-envelope';
import {
  countSharedCoordinates,
  findDensestPoint,
  getNearestDistance,
  type DensestPoint
} from './point-patterns-neighbours';

/** Option state of the point-patterns scene. */
export type PointPatternOptions = {
  subject: 'observations' | 'places';
  groupCategory: string;
  placeCategory: string;
  maximumDistance: number;
  window: 'city' | 'view';
  nullModel: 'window' | 'city';
  radius: number;
  ripleyCorrection: GPURipleyEdgeCorrection;
  distanceCorrection: GPURipleyDistanceEdgeCorrection;
  jitter: number;
  radiusCount: string;
  quadratGrid: string;
  referenceGrid: string;
  colorPoints: 'selection' | 'nearest';
  showQuadrats: boolean;
  showOthers: boolean;
  showRandom: boolean;
  showRadius: boolean;
  emphasizeEdge: boolean;
  compareCorrections: boolean;
};

const SEARCH_GRID: readonly [number, number] = [96, 96];
const SETTLE_MILLISECONDS = 350;
/** Seed of the random pattern drawn on the map; simulation `i` uses `RANDOM_SEED + i`. */
const RANDOM_SEED = 4242;
const CORRECTIONS: readonly GPURipleyEdgeCorrection[] = ['none', 'border', 'isotropic'];
/** Dot radius on a paper ground, where dots are ink and do not add up. */
const PAPER_POINT_RADIUS_STOPS: readonly (readonly [number, number])[] = [
  [10, 1],
  [12, 1.6],
  [14, 2.6]
];

/** Human labels for place categories. */
export const PLACE_LABELS: Record<string, string> = {
  restaurant_cafe: 'Restaurants and cafes',
  bar_nightlife: 'Bars and nightlife',
  grocery: 'Grocery and convenience',
  health: 'Health care',
  school_education: 'Schools and education',
  park_recreation: 'Parks and recreation',
  transit: 'Transit stops',
  retail: 'Retail',
  finance_business: 'Finance and business',
  personal_services: 'Personal services',
  arts_culture: 'Arts and culture',
  worship_community: 'Worship and community',
  lodging: 'Lodging',
  other: 'Other'
};

type SubjectData = {
  id: 'observations' | 'places';
  label: string;
  count: number;
  /** Loaded positions in local metres. */
  base: Float32Array;
  /** Positions currently uploaded (the base, or the jittered copy). */
  current: Float32Array;
  /** Raw `lng, lat` pairs, for the hover index. */
  lngLat: Float32Array;
  category: Uint8Array;
  categoryNames: readonly string[];
  positionsBuffer: Buffer;
  maskBuffer: Buffer;
  /** One complete-spatial-randomness realisation of the same size, drawn when "random" is shown. */
  randomBuffer: Buffer;
  randomScratch: Float32Array;
  /** CPU copy of the 0/1 mask. */
  selected: Uint32Array;
  jitterApplied: number;
  /** Hover index over the selected points (rebuilt after the mask changes). */
  index: NearestIndex | null;
  indexRows: Uint32Array | null;
};

type PatternSet = {
  key: string;
  subject: SubjectData;
  radiusCount: number;
  quadratGrid: readonly [number, number];
  compiled: CompiledGPUCommandGraph<void>;
  nearestDistances: Buffer;
  quadratCounts: Buffer;
  reader: SummaryReader;
};

/** The same pair statistics compiled over a second positions buffer, re-run for every random pattern. */
type SimulationSet = {
  key: string;
  subject: SubjectData;
  radiusCount: number;
  compiled: CompiledGPUCommandGraph<void>;
  positionsBuffer: Buffer;
  maskBuffer: Buffer;
  /** Rows currently set to 1 in the mask buffer. */
  maskCount: number;
  scratch: Float32Array;
  reader: SummaryReader;
};

type Job =
  | {kind: 'observed'; correction: GPURipleyEdgeCorrection; primary: boolean; generation: number}
  | {kind: 'simulation'; index: number; generation: number};

type ObservedResult = {
  radii: Float32Array;
  lMinusR: Float32Array;
  g: Float32Array;
  f: Float32Array;
  j: Float32Array;
  clark: Float32Array;
  quadrat: Float32Array;
  counts: Uint32Array;
};

type OutsideHatch = {
  mesh: ReturnType<typeof createPolygonMeshBuffers>;
};

/**
 * Point-pattern analysis of one Chicago observation group or place type: `GPURipley` (K, L, L minus r),
 * `GPURipleyDistanceFunctions` (G, F, J) and `GPUPointPatternIndices` (nearest-neighbour distances,
 * Clark-Evans, quadrat counts) run over the same points, mask and window.
 *
 * The Monte Carlo envelope reuses the compiled Ripley and distance-function graphs on a second graph
 * over a simulation positions buffer: each of 39 random patterns is a positions-buffer write plus one
 * encode of the same compiled graph, never a recompile. The null model is complete spatial randomness
 * in the window rectangle or on city land (rejection sampling with the city polygon).
 *
 * The category, window, radius, edge corrections and jitter are buffer writes; the number of radii,
 * the quadrat grid and the F reference lattice are compile-time and select cached graphs.
 */
export async function createPointPatterns(
  ctx: SceneContext<PointPatternOptions>
): Promise<SceneInstance<PointPatternOptions>> {
  const {device} = ctx;
  const observations = ctx.datasets.get('chicago-nature');
  const places = ctx.datasets.get('chicago-places');
  const origin = observations.defaultOrigin;
  const projection = observations.getProjection(origin);
  const city = await loadCityGeometry(ctx, origin);
  const resources = new SpatialAnalysisResources(device, 'point-patterns');

  const makeSubject = (
    id: 'observations' | 'places',
    label: string,
    dataset: typeof observations
  ): SubjectData => {
    const base = dataset.projectColumn('position', origin);
    const count = base.length / 2;
    return {
      id,
      label,
      count,
      base,
      current: base,
      lngLat: dataset.column<Float32Array>('position'),
      category: dataset.column<Uint8Array>('category'),
      categoryNames: dataset.categories('category'),
      positionsBuffer: resources.createBuffer(`${id}-positions`, base.slice()),
      maskBuffer: resources.createBuffer(`${id}-mask`, new Uint32Array(count).fill(1)),
      randomBuffer: resources.createBuffer(`${id}-random`, count * 8),
      randomScratch: new Float32Array(count * 2),
      selected: new Uint32Array(count).fill(1),
      jitterApplied: 0,
      index: null,
      indexRows: null
    };
  };
  const subjects = {
    observations: makeSubject('observations', 'Nature observations', observations),
    places: makeSubject('places', 'Places (Overture)', places)
  };

  // The study window of "the city": the observation bounding box widened to hold the whole city
  // limit, so the hatched outside mask can be the window rectangle minus the city.
  const cityWindow = ((): PatternBounds => {
    const [cityMinX, cityMinY, cityMaxX, cityMaxY] = city.cityBoundsMeters;
    let minX = cityMinX;
    let minY = cityMinY;
    let maxX = cityMaxX;
    let maxY = cityMaxY;
    const {base, count} = subjects.observations;
    for (let index = 0; index < count; index++) {
      minX = Math.min(minX, base[index * 2]);
      maxX = Math.max(maxX, base[index * 2]);
      minY = Math.min(minY, base[index * 2 + 1]);
      maxY = Math.max(maxY, base[index * 2 + 1]);
    }
    return [minX - 5, minY - 5, maxX + 5, maxY + 5];
  })();

  /** The window rectangle minus the city limit, triangulated for the hatch. */
  const buildCityWindowMesh = (): PolygonMesh => {
    const [minX, minY, maxX, maxY] = cityWindow;
    const corners = [
      projection.unproject(minX, minY),
      projection.unproject(maxX, minY),
      projection.unproject(maxX, maxY),
      projection.unproject(minX, maxY)
    ];
    return buildPolygonMesh(
      [
        {
          type: 'Feature',
          properties: null,
          geometry: {
            type: 'Polygon',
            coordinates: [
              [...corners, corners[0]],
              ...city.cityPolygons.map(polygon => polygon[0] as number[][])
            ]
          }
        }
      ],
      (longitude, latitude) => projection.project(longitude, latitude)
    );
  };
  const hatches: Record<'city' | 'view', OutsideHatch> = {
    city: {mesh: createPolygonMeshBuffers(resources, buildCityWindowMesh(), 'outside-window')},
    view: {mesh: createPolygonMeshBuffers(resources, city.outsideMesh, 'outside-city')}
  };
  const hatchValues = resources.createBuffer('hatch-values', new Uint32Array([0]));

  const ripleyParameters = resources.createParameterBuffer(
    'ripley',
    'float32',
    GPU_RIPLEY_PARAMETER_LENGTH
  );
  const distanceParameters = resources.createParameterBuffer(
    'distance',
    'float32',
    GPU_RIPLEY_DISTANCE_PARAMETER_LENGTH
  );
  const indicesParameters = resources.createParameterBuffer(
    'indices',
    'float32',
    GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH
  );
  const windowBuffer = resources.createParameterBuffer('window', 'float32', 4);
  const windowSegments = resources.createBuffer('window-segments', 4 * 16);

  // ---- state -------------------------------------------------------------------------------
  let destroyed = false;
  let measuring = false;
  let lastChangeTime = performance.now();
  let bounds: PatternBounds = [...cityWindow];
  let lastViewSignature: string | null = null;
  let active: PatternSet | null = null;
  let activeSimulation: SimulationSet | null = null;
  const sets = new Map<string, PatternSet>();
  const simulationSets = new Map<string, SimulationSet>();

  // Work to do once the camera and options settle.
  let derivedDirty = true;
  let landDirty = true;
  let observedDirty = true;
  let envelopeDirty = true;
  let observedGeneration = 0;
  let envelopeGeneration = 0;
  let observedJobs: Job[] = [];
  let inFlight: Job | null = null;
  let inFlightSince = 0;
  // The r-ring follows the radius slider live; the densest point is only recomputed after a pause.
  let densestDirty = false;
  let densestChangeTime = 0;

  // Replaced by the first `derive()`; the raster costs a few polygon tests per cell.
  let land: LandSampler = {landShare: 0, contains: () => false};
  let selectedInside = 0;
  let densest: (DensestPoint & {x: number; y: number}) | null = null;
  let observed: ObservedResult | null = null;
  const edgeCurves: Partial<Record<GPURipleyEdgeCorrection, Float32Array>> = {};
  let currentCurve: PreviousCurve | null = null;
  let previousCurve: PreviousCurve | null = null;
  const envelope = new EnvelopeAccumulator();
  let envelopeRunning = false;
  let simulationsStarted = 0;
  let randomCount = 0;
  let nearestTable: ClassTable | null = null;
  let nearestExpected = Number.NaN;
  let quadratClasses: QuadratClasses | null = null;
  let quadratKey = '';
  let lastTickKey = '';

  const subjectOf = (options: PointPatternOptions) => subjects[options.subject];
  const categoryOf = (options: PointPatternOptions): string =>
    options.subject === 'observations' ? options.groupCategory : options.placeCategory;
  const categoryLabel = (options: PointPatternOptions): string => {
    const category = categoryOf(options);
    if (category === 'all')
      return options.subject === 'observations' ? 'All observations' : 'All places';
    return options.subject === 'observations'
      ? formatCategory(category)
      : (PLACE_LABELS[category] ?? category);
  };
  const getNullName = (model: PointPatternOptions['nullModel']) =>
    model === 'city' ? 'random on city land' : 'random in the rectangle';
  const getSetKey = (options: PointPatternOptions) =>
    `${options.subject}|${options.radiusCount}|${options.quadratGrid}|${options.referenceGrid}`;
  const getSimulationKey = (options: PointPatternOptions) =>
    `${options.subject}|${options.radiusCount}|${options.referenceGrid}`;
  const parseGrid = (value: string): [number, number] => [Number(value), Number(value)];
  const windowArea = () => (bounds[2] - bounds[0]) * (bounds[3] - bounds[1]);
  const toLngLat = (x: number, y: number): LngLat => projection.unproject(x, y) as LngLat;

  // ---- graphs ------------------------------------------------------------------------------

  function buildSet(options: PointPatternOptions): PatternSet {
    const key = getSetKey(options);
    const subject = subjectOf(options);
    const radiusCount = Number(options.radiusCount);
    const quadratGrid = parseGrid(options.quadratGrid);
    const referenceGrid = parseGrid(options.referenceGrid);
    const id = key.replace(/\|/g, '-');
    const quadratCells = quadratGrid[0] * quadratGrid[1];
    const buffer = (name: string, floats: number) =>
      resources.createBuffer(`${id}-${name}`, floats * 4);
    const outputs = {
      k: buffer('k', radiusCount),
      l: buffer('l', radiusCount),
      lMinusR: buffer('l-minus-r', radiusCount),
      radii: buffer('radii', radiusCount),
      g: buffer('g', radiusCount),
      f: buffer('f', radiusCount),
      j: buffer('j', radiusCount),
      nearest: buffer('nearest', subject.count),
      clarkEvans: buffer('clark-evans', GPU_CLARK_EVANS_LENGTH),
      quadratCounts: buffer('quadrat-counts', quadratCells),
      quadratStatistics: buffer('quadrat-statistics', GPU_QUADRAT_STATISTICS_LENGTH)
    };
    const graph = new GPUCommandGraph<void>(device, {id: `point-patterns-${id}`});
    const positions = importGraphBuffer(
      graph,
      'positions',
      subject.positionsBuffer,
      'float32x2',
      subject.count
    );
    const mask = importGraphBuffer(graph, 'mask', subject.maskBuffer, 'uint32', subject.count);
    const view = <F extends 'float32' | 'uint32'>(
      name: keyof typeof outputs,
      format: F,
      length: number
    ) => importGraphBuffer(graph, name, outputs[name], format, length);
    graph.add(
      new GPURipley({
        id: 'ripley',
        positions,
        mask,
        parameters: ripleyParameters.importToGraph(graph),
        gridSize: SEARCH_GRID,
        radiusCount,
        k: view('k', 'float32', radiusCount),
        l: view('l', 'float32', radiusCount),
        lMinusR: view('lMinusR', 'float32', radiusCount),
        radii: view('radii', 'float32', radiusCount)
      })
    );
    graph.add(
      new GPURipleyDistanceFunctions({
        id: 'ripley-distance',
        positions,
        mask,
        parameters: distanceParameters.importToGraph(graph),
        gridSize: SEARCH_GRID,
        referenceGrid,
        radiusCount,
        g: view('g', 'float32', radiusCount),
        f: view('f', 'float32', radiusCount),
        j: view('j', 'float32', radiusCount)
      })
    );
    graph.add(
      new GPUPointPatternIndices({
        id: 'indices',
        positions,
        mask,
        parameters: indicesParameters.importToGraph(graph),
        gridSize: SEARCH_GRID,
        quadratGrid,
        nearestNeighborDistances: view('nearest', 'float32', subject.count),
        clarkEvans: view('clarkEvans', 'float32', GPU_CLARK_EVANS_LENGTH),
        quadratCounts: view('quadratCounts', 'uint32', quadratCells),
        quadratStatistics: view('quadratStatistics', 'float32', GPU_QUADRAT_STATISTICS_LENGTH)
      })
    );
    const compiled = resources.track(graph.compile());
    const set: PatternSet = {
      key,
      subject,
      radiusCount,
      quadratGrid,
      compiled,
      nearestDistances: outputs.nearest,
      quadratCounts: outputs.quadratCounts,
      reader: new SummaryReader(
        resources,
        `summary-${id}`,
        [
          {buffer: outputs.lMinusR, size: radiusCount * 4},
          {buffer: outputs.g, size: radiusCount * 4},
          {buffer: outputs.f, size: radiusCount * 4},
          {buffer: outputs.j, size: radiusCount * 4},
          {buffer: outputs.radii, size: radiusCount * 4},
          {buffer: outputs.clarkEvans, size: GPU_CLARK_EVANS_LENGTH * 4},
          {buffer: outputs.quadratStatistics, size: GPU_QUADRAT_STATISTICS_LENGTH * 4},
          {buffer: outputs.quadratCounts, size: quadratCells * 4}
        ],
        bytes => handleObservedSummary(set, bytes)
      )
    };
    return set;
  }

  /**
   * The envelope's graph: Ripley and the distance functions again, compiled once over their own
   * positions and mask buffers (the random pattern), sharing the observed graph's parameter buffers.
   */
  function buildSimulation(options: PointPatternOptions): SimulationSet {
    const key = getSimulationKey(options);
    const subject = subjectOf(options);
    const radiusCount = Number(options.radiusCount);
    const referenceGrid = parseGrid(options.referenceGrid);
    const id = `${key.replace(/\|/g, '-')}-random`;
    const buffer = (name: string, floats: number) =>
      resources.createBuffer(`${id}-${name}`, floats * 4);
    const positionsBuffer = resources.createBuffer(`${id}-positions`, subject.count * 8);
    const maskBuffer = resources.createBuffer(`${id}-mask`, new Uint32Array(subject.count));
    const outputs = {
      k: buffer('k', radiusCount),
      l: buffer('l', radiusCount),
      lMinusR: buffer('l-minus-r', radiusCount),
      radii: buffer('radii', radiusCount),
      g: buffer('g', radiusCount),
      f: buffer('f', radiusCount),
      j: buffer('j', radiusCount)
    };
    const graph = new GPUCommandGraph<void>(device, {id: `point-patterns-${id}`});
    const positions = importGraphBuffer(
      graph,
      'positions',
      positionsBuffer,
      'float32x2',
      subject.count
    );
    const mask = importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', subject.count);
    const view = (name: keyof typeof outputs) =>
      importGraphBuffer(graph, name, outputs[name], 'float32', radiusCount);
    graph.add(
      new GPURipley({
        id: 'ripley',
        positions,
        mask,
        parameters: ripleyParameters.importToGraph(graph),
        gridSize: SEARCH_GRID,
        radiusCount,
        k: view('k'),
        l: view('l'),
        lMinusR: view('lMinusR'),
        radii: view('radii')
      })
    );
    graph.add(
      new GPURipleyDistanceFunctions({
        id: 'ripley-distance',
        positions,
        mask,
        parameters: distanceParameters.importToGraph(graph),
        gridSize: SEARCH_GRID,
        referenceGrid,
        radiusCount,
        g: view('g'),
        f: view('f'),
        j: view('j')
      })
    );
    const compiled = resources.track(graph.compile());
    const simulation: SimulationSet = {
      key,
      subject,
      radiusCount,
      compiled,
      positionsBuffer,
      maskBuffer,
      maskCount: 0,
      scratch: new Float32Array(subject.count * 2),
      reader: new SummaryReader(
        resources,
        `summary-${id}`,
        [
          {buffer: outputs.lMinusR, size: radiusCount * 4},
          {buffer: outputs.g, size: radiusCount * 4},
          {buffer: outputs.f, size: radiusCount * 4},
          {buffer: outputs.j, size: radiusCount * 4}
        ],
        bytes => handleSimulationSummary(simulation, bytes)
      )
    };
    return simulation;
  }

  function selectSet(): void {
    const options = ctx.options;
    const key = getSetKey(options);
    if (!active || active.key !== key) {
      let next = sets.get(key);
      if (!next) {
        next = buildSet(options);
        sets.set(key, next);
      }
      active = next;
    }
    const simulationKey = getSimulationKey(options);
    if (!activeSimulation || activeSimulation.key !== simulationKey) {
      let next = simulationSets.get(simulationKey);
      if (!next) {
        next = buildSimulation(options);
        simulationSets.set(simulationKey, next);
      }
      activeSimulation = next;
    }
  }

  // ---- change tracking ---------------------------------------------------------------------

  /** The observed pattern changed (jitter, grids, corrections): re-run the observed graph. */
  function markObservedChanged(): void {
    observedGeneration++;
    observedJobs = [];
    observedDirty = true;
    derivedDirty = true;
    lastChangeTime = performance.now();
  }

  /** The null changed (selection, window, null model, radii, corrections): restart the envelope. */
  function markEnvelopeChanged(): void {
    envelopeGeneration++;
    envelopeRunning = false;
    envelopeDirty = true;
    derivedDirty = true;
    envelope.reset(Number(ctx.options.radiusCount));
    lastChangeTime = performance.now();
    ctx.setReadout('simulations', `0 of ${ENVELOPE_RUNS} random patterns`);
  }

  // ---- CPU derived quantities --------------------------------------------------------------

  function recountSelection(): void {
    const subject = subjectOf(ctx.options);
    let count = 0;
    for (let row = 0; row < subject.count; row++) {
      if (!subject.selected[row]) continue;
      const x = subject.current[row * 2];
      const y = subject.current[row * 2 + 1];
      if (x >= bounds[0] && x <= bounds[2] && y >= bounds[1] && y <= bounds[3]) count++;
    }
    selectedInside = count;
  }

  /** Draws the random pattern shown on the map: the first of the envelope's simulations. */
  function refreshRandom(): void {
    const subject = subjectOf(ctx.options);
    const sampler = ctx.options.nullModel === 'city' ? land : null;
    fillRandomPositions(
      subject.randomScratch,
      selectedInside,
      bounds,
      sampler,
      createSeededRandom(RANDOM_SEED)
    );
    if (selectedInside > 0) {
      subject.randomBuffer.write(subject.randomScratch.subarray(0, selectedInside * 2));
    }
    randomCount = selectedInside;
  }

  function startEnvelope(): void {
    envelopeDirty = false;
    simulationsStarted = 0;
    envelopeRunning = selectedInside >= 3;
    refreshRandom();
    publishCurveChart();
    publishDistanceChart();
    if (ctx.options.showRandom) ctx.requestLayers();
  }

  function refreshDensest(): void {
    densestDirty = false;
    const options = ctx.options;
    const subject = subjectOf(options);
    const found = findDensestPoint(subject.current, subject.selected, bounds, options.radius);
    densest = found
      ? {
          ...found,
          x: subject.current[found.row * 2],
          y: subject.current[found.row * 2 + 1]
        }
      : null;
    publishRadius();
  }

  function refreshShared(): void {
    const subject = subjectOf(ctx.options);
    const {share} = countSharedCoordinates(subject.base, subject.selected);
    ctx.setReadout('duplicateShare', Number.isFinite(share) ? formatPercent(share, 1) : null);
  }

  function derive(): void {
    derivedDirty = false;
    if (landDirty) {
      landDirty = false;
      land = createLandSampler(city.containsMeters, bounds);
    }
    recountSelection();
    publishWindow();
    if (envelopeDirty) startEnvelope();
    refreshDensest();
    refreshShared();
    ctx.setCost({
      records: selectedInside,
      passes: active ? active.compiled.stats.nodeOrder.length : undefined
    });
  }

  // ---- publishing --------------------------------------------------------------------------

  function publishWindow(): void {
    const options = ctx.options;
    const area = windowArea();
    ctx.setReadout('windowArea', formatArea(area));
    ctx.setReadout('windowLandShare', formatPercent(land.landShare));
    ctx.setReadout('outsideShare', formatPercent(1 - land.landShare));
    ctx.setReadout('nullName', getNullName(options.nullModel));
    ctx.setReadout('intensity', `${formatRate(selectedInside / (area / 1e6), 'km²')} of window`);
    const [west, south] = toLngLat(bounds[0], bounds[1]);
    const [east, north] = toLngLat(bounds[2], bounds[3]);
    const annotations: MapAnnotation[] =
      options.window === 'city'
        ? [
            {
              kind: 'frame',
              id: 'window-frame',
              bounds: [west, south, east, north],
              text: `Study window, ${formatArea(area)}`
            }
          ]
        : [];
    ctx.setAnnotations('window', annotations.length ? annotations : null);
    const lake = findPlace(CHICAGO, 'lake-michigan');
    ctx.setAnnotations(
      'outside',
      options.emphasizeEdge && lake
        ? [
            {
              kind: 'note',
              id: 'outside-note',
              coordinate: lake.lngLat as LngLat,
              title: `${formatPercent(1 - land.landShare)} of the window`,
              text: 'Lake and suburbs: no record can fall here'
            }
          ]
        : null
    );
    publishFurniture();
  }

  function publishFurniture(): void {
    const options = ctx.options;
    const subject = subjectOf(options);
    const sample =
      subject.id === 'observations'
        ? `${formatCount(subject.count)} ${NATURE_SAMPLE}`
        : `${formatCount(subject.count)} Overture places, Chicago`;
    const tick = options.showRadius ? options.radius : 0;
    const key = `${sample}|${tick}`;
    if (key === lastTickKey) return;
    lastTickKey = key;
    ctx.setFurniture({
      title: {sample},
      scaleBar: {units: 'metric', ticks: tick ? [tick] : undefined}
    });
  }

  /** The r-ring and its note at the densest selected point (cheap: runs on every radius change). */
  function publishRadius(): void {
    const options = ctx.options;
    publishFurniture();
    if (!options.showRadius || !densest) {
      ctx.setAnnotations('radius', null);
      ctx.setReadout('withinRadius', null);
      return;
    }
    const coordinate = toLngLat(densest.x, densest.y);
    const noun = options.subject === 'observations' ? 'sightings' : 'places';
    ctx.setReadout(
      'withinRadius',
      `${formatCount(densest.neighbours)} ${noun} within ${formatDistance(options.radius)}`
    );
    ctx.setAnnotations('radius', [
      {
        kind: 'ring',
        id: 'radius-ring',
        coordinate,
        radiusMeters: options.radius,
        text: `r = ${formatDistance(options.radius)}`,
        dashed: true
      },
      {
        kind: 'note',
        id: 'radius-note',
        coordinate,
        title: `${formatCount(densest.neighbours)} ${noun} within r`,
        text: nearestPlaceLabel(CHICAGO, coordinate, {maxDistanceMeters: 8000}) ?? undefined,
        tone: 'accent'
      }
    ]);
  }

  function getSlot(): number {
    return ctx.options.subject === 'observations' ? OBSERVED_SLOT : SECOND_SLOT;
  }

  function publishCurveChart(): void {
    if (!observed) return;
    const options = ctx.options;
    const complete =
      CORRECTIONS.every(name => edgeCurves[name]) && options.compareCorrections
        ? {
            none: edgeCurves.none as Float32Array,
            border: edgeCurves.border as Float32Array,
            isotropic: edgeCurves.isotropic as Float32Array
          }
        : null;
    ctx.setChart(
      'lCurve',
      buildLCurveChart({
        label: categoryLabel(options),
        radii: observed.radii,
        lMinusR: observed.lMinusR,
        slot: getSlot(),
        previous: previousCurve,
        envelope: envelope.getEnvelopes()?.lMinusR ?? null,
        simulationsDone: envelope.count,
        nullName: getNullName(options.nullModel),
        corrections: complete
      })
    );
  }

  function publishDistanceChart(): void {
    if (!observed) return;
    const options = ctx.options;
    const envelopes = envelope.getEnvelopes();
    ctx.setChart(
      'gfjCurves',
      buildDistanceChart({
        radii: observed.radii,
        g: observed.g,
        f: observed.f,
        j: observed.j,
        intensity: observed.clark[0] / windowArea(),
        slot: getSlot(),
        envelopes: envelopes ? {g: envelopes.g, f: envelopes.f, j: envelopes.j} : null,
        nullName: getNullName(options.nullModel)
      })
    );
  }

  function publishSimulationProgress(): void {
    ctx.setReadout('simulations', `${envelope.count} of ${ENVELOPE_RUNS} random patterns`);
    if (envelope.isComplete) {
      ctx.setReadout(
        'graphs',
        `2 graphs compiled once, ${formatCount(envelope.count)} random runs of one of them`
      );
    }
    publishCurveChart();
    if (envelope.isComplete) publishDistanceChart();
  }

  // ---- observed results --------------------------------------------------------------------

  /** `L(r) - r` at `distance` by linear interpolation along the radii. */
  function interpolateAt(radii: Float32Array, values: Float32Array, distance: number): number {
    for (let index = 0; index < radii.length; index++) {
      if (radii[index] >= distance) {
        if (index === 0) return values[0];
        const span = radii[index] - radii[index - 1];
        const t = span > 0 ? (distance - radii[index - 1]) / span : 1;
        return values[index - 1] + t * (values[index] - values[index - 1]);
      }
    }
    return Number.NaN;
  }

  function handleObservedSummary(set: PatternSet, bytes: ArrayBuffer): void {
    const job = inFlight;
    inFlight = null;
    if (destroyed || !job || job.kind !== 'observed') return;
    if (job.generation !== observedGeneration || set !== active) return;
    const floats = new Float32Array(bytes);
    const words = new Uint32Array(bytes);
    const count = set.radiusCount;
    const lMinusR = floats.slice(0, count);
    edgeCurves[job.correction] = lMinusR;
    if (!job.primary) {
      publishCurveChart();
      return;
    }
    const g = floats.slice(count, count * 2);
    const f = floats.slice(count * 2, count * 3);
    const j = floats.slice(count * 3, count * 4);
    const radii = floats.slice(count * 4, count * 5);
    const clarkOffset = count * 5;
    const quadratOffset = clarkOffset + GPU_CLARK_EVANS_LENGTH;
    const countsOffset = quadratOffset + GPU_QUADRAT_STATISTICS_LENGTH;
    const clark = floats.slice(clarkOffset, quadratOffset);
    const quadrat = floats.slice(quadratOffset, countsOffset);
    const quadratCells = set.quadratGrid[0] * set.quadratGrid[1];
    const counts = words.slice(countsOffset, countsOffset + quadratCells);
    observed = {radii, lMinusR, g, f, j, clark, quadrat, counts};
    const options = ctx.options;

    ctx.setReadout('included', formatCount(clark[0]));
    // L(r) - r
    let peakIndex = -1;
    let peakValue = -Infinity;
    for (let index = 0; index < count; index++) {
      if (Number.isFinite(lMinusR[index]) && lMinusR[index] > peakValue) {
        peakValue = lMinusR[index];
        peakIndex = index;
      }
    }
    ctx.setReadout(
      'lPeak',
      peakIndex >= 0
        ? `${formatSigned(peakValue, 0)} m at r = ${formatDistance(radii[peakIndex])}`
        : 'undefined'
    );
    ctx.setReadout('peakR', peakIndex >= 0 ? formatDistance(radii[peakIndex]) : null);
    const atKilometer = interpolateAt(radii, lMinusR, 1000);
    ctx.setReadout(
      'lAt1km',
      Number.isFinite(atKilometer) ? `${formatSigned(atKilometer, 0)} m` : null
    );
    currentCurve = {
      label: categoryLabel(options),
      radii: Array.from(radii),
      lMinusR: Array.from(lMinusR)
    };

    // Clark-Evans: [n, observed, expected, ratio, standardError, z]
    const ratio = clark[3];
    ctx.setReadout('clarkEvansR', Number.isFinite(ratio) ? ratio.toFixed(2) : null);
    ctx.setReadout(
      'clarkEvans',
      Number.isFinite(ratio)
        ? `R = ${ratio.toFixed(3)} (${ratio < 1 ? 'clustered' : 'dispersed'}), z = ${formatSigned(clark[5], 1)}`
        : 'undefined'
    );
    ctx.setReadout('expectedNN', formatDistance(clark[2]));
    ctx.setReadout('observedNN', formatDistance(clark[1]));
    ctx.setChart(
      'clarkEvansGauge',
      buildClarkEvansGauge(ratio, clark[5], (1.96 * clark[4]) / clark[2])
    );

    // The nearest-neighbour classes follow the expected distance (rebuilt when it moves by 5 %).
    const ground = ctx.ground();
    if (
      Number.isFinite(clark[2]) &&
      (!nearestTable || Math.abs(clark[2] - nearestExpected) > nearestExpected * 0.05)
    ) {
      nearestExpected = clark[2];
      nearestTable = makeNearestTable(nearestExpected, ground);
      ctx.setLegendData('nearestTable', nearestTable);
      ctx.setLegendData('nearestExpected', nearestExpected);
      ctx.requestLayers();
    }

    // Quadrat: [count, mean, variance, VMR, chiSquare, df]. The classes are computed once per
    // selection, window and grid and then frozen, so toggling anything else keeps the same breaks.
    ctx.setReadout('vmr', Number.isFinite(quadrat[3]) ? quadrat[3].toFixed(1) : null);
    ctx.setReadout(
      'chiSquare',
      Number.isFinite(quadrat[3])
        ? `chi-square ${formatCount(quadrat[4])} on ${formatCount(quadrat[5])} degrees of freedom`
        : null
    );
    const nextQuadratKey = `${set.key}|${categoryOf(options)}|${options.window}`;
    if (!quadratClasses || nextQuadratKey !== quadratKey) {
      quadratKey = nextQuadratKey;
      quadratClasses = makeQuadratTable(counts, ground);
      ctx.setLegendData('quadrat', quadratClasses);
      ctx.requestLayers();
    }
    ctx.setChart(
      'quadratHistogram',
      buildQuadratChart(counts, quadrat[1], quadrat[3], quadratClasses.table)
    );

    publishWindow();
    publishCurveChart();
    publishDistanceChart();
  }

  function handleSimulationSummary(simulation: SimulationSet, bytes: ArrayBuffer): void {
    const job = inFlight;
    inFlight = null;
    if (destroyed || !job || job.kind !== 'simulation') return;
    if (job.generation !== envelopeGeneration || simulation !== activeSimulation) return;
    const floats = new Float32Array(bytes);
    const count = simulation.radiusCount;
    envelope.add({
      lMinusR: floats.subarray(0, count),
      g: floats.subarray(count, count * 2),
      f: floats.subarray(count * 2, count * 3),
      j: floats.subarray(count * 3, count * 4)
    });
    if (envelope.isComplete) envelopeRunning = false;
    publishSimulationProgress();
  }

  // ---- parameters and buffers --------------------------------------------------------------

  /** Writes the shared parameter buffers; one job per frame, so the values are the job's. */
  function writeParameters(ripleyCorrection: GPURipleyEdgeCorrection): void {
    const options = ctx.options;
    const maximumDistance = options.maximumDistance;
    ripleyParameters.write(
      getGPURipleyParameterValues({bounds, maximumDistance, edgeCorrection: ripleyCorrection})
    );
    distanceParameters.write(
      getGPURipleyDistanceParameterValues({
        bounds,
        maximumDistance,
        edgeCorrection: options.distanceCorrection
      })
    );
    indicesParameters.write(
      getGPUPointPatternIndicesParameterValues({
        bounds,
        maximumDistance: Math.min(maximumDistance, 400)
      })
    );
  }

  /** The window as a parameter buffer (the quadrat raster reads it) and a frame of four segments. */
  function writeWindowGeometry(): void {
    windowBuffer.write(Float32Array.from(bounds));
    const [x0, y0, x1, y1] = bounds;
    windowSegments.write(
      Float32Array.of(x0, y0, x1, y0, x1, y0, x1, y1, x1, y1, x0, y1, x0, y1, x0, y0)
    );
  }

  function writeMask(): void {
    const options = ctx.options;
    const subject = subjectOf(options);
    const category = categoryOf(options);
    const wanted = category === 'all' ? -1 : subject.categoryNames.indexOf(category);
    const mask = new Uint32Array(subject.count);
    for (let index = 0; index < subject.count; index++) {
      mask[index] = wanted < 0 || subject.category[index] === wanted ? 1 : 0;
    }
    subject.selected = mask;
    subject.index = null;
    subject.indexRows = null;
    subject.maskBuffer.write(mask);
  }

  function writeJitter(subject: SubjectData, radius: number): void {
    if (subject.jitterApplied === radius) return;
    subject.jitterApplied = radius;
    if (radius === 0) {
      subject.current = subject.base;
    } else {
      const random = createSeededRandom(subject.id === 'observations' ? 11 : 23);
      const jittered = new Float32Array(subject.base.length);
      for (let index = 0; index < subject.count; index++) {
        const angle = random() * Math.PI * 2;
        const distance = Math.sqrt(random()) * radius;
        jittered[index * 2] = subject.base[index * 2] + Math.cos(angle) * distance;
        jittered[index * 2 + 1] = subject.base[index * 2 + 1] + Math.sin(angle) * distance;
      }
      subject.current = jittered;
    }
    subject.positionsBuffer.write(subject.current);
  }

  /** Remembers the curve being replaced so the next selection can be compared with it. */
  function rememberCurve(): void {
    if (currentCurve) previousCurve = currentCurve;
  }

  function ensureIndex(subject: SubjectData): NearestIndex {
    if (subject.index && subject.indexRows) return subject.index;
    const rows: number[] = [];
    for (let row = 0; row < subject.count; row++) if (subject.selected[row]) rows.push(row);
    const pairs = new Float32Array(rows.length * 2);
    rows.forEach((row, slot) => {
      pairs[slot * 2] = subject.lngLat[row * 2];
      pairs[slot * 2 + 1] = subject.lngLat[row * 2 + 1];
    });
    subject.indexRows = Uint32Array.from(rows);
    subject.index = createNearestIndex(pairs);
    return subject.index;
  }

  // ---- first state -------------------------------------------------------------------------

  writeMask();
  writeJitter(
    subjects.observations,
    ctx.options.subject === 'observations' ? ctx.options.jitter : 0
  );
  writeJitter(subjects.places, ctx.options.subject === 'places' ? ctx.options.jitter : 0);
  writeWindowGeometry();
  selectSet();
  envelope.reset(Number(ctx.options.radiusCount));
  landDirty = true;
  derive();
  lastChangeTime = performance.now();

  async function measurePattern(): Promise<void> {
    const set = active;
    if (!set || measuring || destroyed) return;
    measuring = true;
    ctx.setReadout('timing', 'measuring…');
    try {
      const timing = await measureCompiledGraph(device, set.compiled, {
        parameters: undefined,
        completionBuffer: set.quadratCounts,
        signal: ctx.signal
      });
      if (!destroyed) {
        ctx.setReadout(
          'timing',
          `${formatCompiledGraphTiming(timing).split(' · ')[0]} for ${formatCount(set.subject.count)} points, r up to ${formatRadius(ctx.options.maximumDistance)}`
        );
      }
    } catch {
      // Interrupted.
    } finally {
      measuring = false;
    }
  }

  ctx.setReadout(
    'isotropicCap',
    'Isotropic edge weights are capped at 100, so pairs very near the window edge at large r are under-corrected. Kaplan-Meier and Hanisch weights exist for G, F and J only. Positions are float32 metres.'
  );

  // ---- jobs --------------------------------------------------------------------------------

  function scheduleObserved(): void {
    const options = ctx.options;
    const current = options.ripleyCorrection;
    const order = options.compareCorrections
      ? [...CORRECTIONS.filter(name => name !== current), current]
      : [current];
    observedJobs = order.map((correction, index) => ({
      kind: 'observed' as const,
      correction,
      primary: index === order.length - 1,
      generation: observedGeneration
    }));
    observedDirty = false;
  }

  function runObserved(
    job: Extract<Job, {kind: 'observed'}>,
    set: PatternSet,
    commandEncoder: Parameters<SceneInstance<PointPatternOptions>['encode']>[0]
  ): void {
    writeParameters(job.correction);
    set.compiled.encode(commandEncoder, {parameters: undefined});
    set.reader.request(commandEncoder);
  }

  function runSimulation(
    job: Extract<Job, {kind: 'simulation'}>,
    simulation: SimulationSet,
    commandEncoder: Parameters<SceneInstance<PointPatternOptions>['encode']>[0]
  ): void {
    const count = selectedInside;
    if (simulation.maskCount !== count) {
      const mask = new Uint32Array(simulation.subject.count);
      mask.fill(1, 0, count);
      simulation.maskBuffer.write(mask);
      simulation.maskCount = count;
    }
    fillRandomPositions(
      simulation.scratch,
      count,
      bounds,
      ctx.options.nullModel === 'city' ? land : null,
      createSeededRandom(RANDOM_SEED + job.index)
    );
    if (count > 0) simulation.positionsBuffer.write(simulation.scratch.subarray(0, count * 2));
    writeParameters(ctx.options.ripleyCorrection);
    simulation.compiled.encode(commandEncoder, {parameters: undefined});
    simulation.reader.request(commandEncoder);
  }

  // ---- layers ------------------------------------------------------------------------------

  /** Amber alpha per dot: more points saturate sooner, so the alpha falls with the count. */
  function getDotAlpha(): number {
    const count = Math.max(selectedInside, 1000);
    return Math.round(Math.min(70, Math.max(14, 40 * Math.sqrt(11000 / count))));
  }

  return {
    getCompiledGraphs: () =>
      [active?.compiled, activeSimulation?.compiled].filter(
        Boolean
      ) as unknown as CompiledGPUCommandGraph<never>[],

    setOption(id, _value, state) {
      switch (id) {
        case 'subject':
        case 'groupCategory':
        case 'placeCategory':
          rememberCurve();
          selectSet();
          writeJitter(subjectOf(state), state.jitter);
          writeMask();
          observed = null;
          for (const name of CORRECTIONS) delete edgeCurves[name];
          markObservedChanged();
          markEnvelopeChanged();
          lastTickKey = '';
          break;
        case 'radiusCount':
        case 'quadratGrid':
        case 'referenceGrid':
          selectSet();
          observed = null;
          for (const name of CORRECTIONS) delete edgeCurves[name];
          markObservedChanged();
          markEnvelopeChanged();
          break;
        case 'jitter':
          writeJitter(subjectOf(state), state.jitter);
          markObservedChanged();
          break;
        case 'window':
          if (state.window === 'city') bounds = [...cityWindow];
          lastViewSignature = null;
          writeWindowGeometry();
          landDirty = true;
          markObservedChanged();
          markEnvelopeChanged();
          break;
        case 'maximumDistance':
        case 'ripleyCorrection':
        case 'distanceCorrection':
          for (const name of CORRECTIONS) delete edgeCurves[name];
          markObservedChanged();
          markEnvelopeChanged();
          break;
        case 'nullModel':
          markEnvelopeChanged();
          break;
        case 'compareCorrections':
          for (const name of CORRECTIONS) delete edgeCurves[name];
          markObservedChanged();
          break;
        case 'radius':
        case 'showRadius':
          densestDirty = true;
          densestChangeTime = performance.now();
          publishRadius();
          break;
        case 'emphasizeEdge':
          publishWindow();
          break;
        default:
          break;
      }
      ctx.requestLayers();
    },

    onAction(id) {
      if (id === 'measure') void measurePattern();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    onGroundChange(ground) {
      if (Number.isFinite(nearestExpected)) {
        nearestTable = makeNearestTable(nearestExpected, ground);
        ctx.setLegendData('nearestTable', nearestTable);
      }
      if (observed) {
        quadratClasses = makeQuadratTable(observed.counts, ground);
        ctx.setLegendData('quadrat', quadratClasses);
        publishCurveChart();
      }
      ctx.setLegendData('ground', ground);
      ctx.requestLayers();
    },

    getTooltip(event): TooltipContent | null {
      const options = ctx.options;
      const subject = subjectOf(options);
      if (!event.coordinate || !observed) return null;
      const index = ensureIndex(subject);
      const hit = index.nearest(event.coordinate, 30);
      const rows = subject.indexRows;
      if (!hit || !rows) return null;
      const row = rows[hit.index];
      const lngLat: LngLat = [subject.lngLat[row * 2], subject.lngLat[row * 2 + 1]];
      const neighbours = index.within(lngLat, options.radius).length - 1;
      let nearest = Number.POSITIVE_INFINITY;
      for (const searchRadius of [25, 100, 400, 1600, 6400]) {
        const near = index.within(lngLat, searchRadius).map(slot => rows[slot]);
        nearest = getNearestDistance(subject.current, row, near);
        if (Number.isFinite(nearest)) break;
      }
      const expected = observed.clark[2];
      const classIndex = nearestTable ? getClassIndexOf(nearestTable, nearest) : -1;
      const tooltipRows: TooltipRow[] = [
        {
          label: 'Nearest neighbour',
          value: Number.isFinite(nearest) ? formatDistance(nearest) : 'none',
          swatch: nearestTable && classIndex >= 0 ? nearestTable.colors[classIndex] : undefined,
          emphasis: true
        }
      ];
      if (nearestTable && classIndex >= 0) {
        tooltipRows.push({label: 'Class', value: getClassLabel(nearestTable, classIndex)});
      }
      if (Number.isFinite(nearest) && expected > 0) {
        tooltipRows.push({
          label: 'Compared with random',
          value: `${(nearest / expected).toFixed(2)} ×`,
          unit: 'expected'
        });
      }
      tooltipRows.push({
        label: `Within ${formatDistance(options.radius)}`,
        value: formatCount(neighbours),
        unit: options.subject === 'observations' ? 'other sightings' : 'other places'
      });
      return {
        title: categoryLabel(options),
        subtitle: nearestPlaceLabel(CHICAGO, lngLat, {maxDistanceMeters: 8000}) ?? undefined,
        rows: tooltipRows,
        highlight: {kind: 'point', coordinate: lngLat}
      };
    },

    encode(commandEncoder, frame) {
      const set = active;
      const simulation = activeSimulation;
      if (!set || !simulation) return;
      if (ctx.options.window === 'view') {
        const viewBounds = getViewportMetricBounds(frame.viewport, projection);
        const signature = viewBounds.map(value => Math.round(value / 5)).join(',');
        if (signature !== lastViewSignature) {
          lastViewSignature = signature;
          bounds = [...viewBounds];
          writeWindowGeometry();
          landDirty = true;
          markObservedChanged();
          markEnvelopeChanged();
        }
      }
      // Deliver finished readbacks and retry a copy that found no free ticket.
      set.reader.flush(commandEncoder);
      simulation.reader.flush(commandEncoder);
      const now = performance.now();
      if (densestDirty && now - densestChangeTime > 200) refreshDensest();
      if (inFlight) {
        // A readback that ended without a result (device or ring torn down) must not block the queue.
        if (now - inFlightSince > 4000 && !set.reader.isPending && !simulation.reader.isPending) {
          inFlight = null;
        }
        return;
      }
      if (now - lastChangeTime <= SETTLE_MILLISECONDS) return;
      if (derivedDirty) derive();
      if (observedDirty && observedJobs.length === 0) scheduleObserved();
      // One job per frame: they share the parameter buffers, and the observed graph comes first.
      const job: Job | undefined =
        observedJobs.shift() ??
        (envelopeRunning
          ? {kind: 'simulation', index: simulationsStarted++, generation: envelopeGeneration}
          : undefined);
      if (!job) return;
      inFlight = job;
      inFlightSince = now;
      if (job.kind === 'observed') runObserved(job, set, commandEncoder);
      else runSimulation(job, simulation, commandEncoder);
    },

    getLayers() {
      const set = active;
      if (!set) return [];
      const options = ctx.options;
      const subject = set.subject;
      const ground: PointsGround = ctx.ground();
      const night = ground === 'dark';
      const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
      const layers: Layer[] = [];

      // Quadrat fill: one frozen class table, empty quadrats transparent (the lake stays empty).
      if (options.showQuadrats && quadratClasses) {
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: `pattern-quadrats-${set.key}`,
            coordinateOrigin,
            gridSize: set.quadratGrid,
            bounds: windowBuffer.buffer,
            values: set.quadratCounts,
            valueFormat: 'uint32',
            colormap: 'uniform',
            ...getClassTableLayerProps(quadratClasses.table),
            outlineClasses: {color: getParameterInk(ground, 90), widthPixels: 0.7},
            color: [255, 255, 255, night ? 214 : 224]
          })
        );
      }

      // The window: everything inside it that is not city land is hatched ("no record can fall here").
      const hatch = hatches[options.window];
      layers.push(
        new SpatialAnalysisPolygonLayer({
          id: `pattern-outside-${options.window}`,
          coordinateOrigin,
          triangles: hatch.mesh.triangles,
          features: hatch.mesh.triangleFeatures,
          vertexCount: hatch.mesh.vertexCount,
          values: hatchValues,
          valueFormat: 'uint32',
          colormap: 'uniform',
          classBreaks: [1],
          classColors: [
            [0, 0, 0, 0],
            [0, 0, 0, 0]
          ],
          hatchClasses: [0],
          hatchColor: getParameterInk(ground, options.emphasizeEdge ? 102 : 56),
          hatchSpacingPixels: 5,
          hatchWidthPixels: 1
        })
      );

      if (options.showOthers && options.colorPoints !== 'nearest') {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: `pattern-others-${subject.id}`,
            coordinateOrigin,
            positions: subject.positionsBuffer,
            instanceCount: subject.count,
            radiusPixels: 0.8,
            color: getGhostColor(ground)
          })
        );
      }

      const radiusStops = night ? DENSE_POINT_RADIUS_STOPS : PAPER_POINT_RADIUS_STOPS;
      if (options.colorPoints === 'nearest' && nearestTable) {
        const layerColors = nearestTable.colors.map((color, index) =>
          index === 0
            ? ([0, 0, 0, 0] as const)
            : ([color[0], color[1], color[2], color[3] ?? 255] as const)
        );
        layers.push(
          new SpatialAnalysisPointLayer({
            id: `pattern-nearest-${subject.id}`,
            coordinateOrigin,
            positions: subject.positionsBuffer,
            instanceCount: subject.count,
            values: set.nearestDistances,
            valueFormat: 'float32',
            colormap: 'uniform',
            classBreaks: [...nearestTable.breaks],
            classColors: layerColors,
            noDataColor: [0, 0, 0, 0],
            radiusPixels: radiusStops.map(([zoom, radius]) => [zoom, radius + 0.6] as const)
          }),
          // Records on exactly the same coordinate: a ring, because a dot cannot show a stack.
          new SpatialAnalysisPointLayer({
            id: `pattern-duplicates-${subject.id}`,
            coordinateOrigin,
            positions: subject.positionsBuffer,
            instanceCount: subject.count,
            values: set.nearestDistances,
            valueFormat: 'float32',
            colormap: 'uniform',
            classBreaks: [...nearestTable.breaks.slice(0, 1)],
            classColors: [nearestTable.colors[0], [0, 0, 0, 0]],
            noDataColor: [0, 0, 0, 0],
            shape: 'ring',
            outlineWidthPixels: 1.2,
            radiusPixels: radiusStops.map(([zoom, radius]) => [zoom, radius + 3] as const)
          })
        );
      } else if (options.showRandom && randomCount > 0) {
        const alpha = night ? getDotAlpha() : 200;
        layers.push(
          new SpatialAnalysisPointLayer({
            id: `pattern-random-${subject.id}`,
            coordinateOrigin,
            positions: subject.randomBuffer,
            instanceCount: randomCount,
            radiusPixels: radiusStops,
            blending: night ? 'additive' : 'normal',
            color:
              subject.id === 'observations'
                ? getSecondColor(ground, alpha)
                : ([233, 236, 240, alpha] as const)
          })
        );
      } else {
        const alpha = night ? getDotAlpha() : 210;
        layers.push(
          new SpatialAnalysisPointLayer({
            id: `pattern-selected-${subject.id}`,
            coordinateOrigin,
            positions: subject.positionsBuffer,
            instanceCount: subject.count,
            values: subject.maskBuffer,
            valueFormat: 'uint32',
            colormap: 'mask',
            color:
              subject.id === 'observations'
                ? getSubjectColor(ground, alpha)
                : getSecondColor(ground, alpha),
            noDataColor: [0, 0, 0, 0],
            radiusPixels: radiusStops,
            blending: night ? 'additive' : 'normal'
          })
        );
      }

      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'pattern-window',
          coordinateOrigin,
          segments: windowSegments,
          instanceCount: 4,
          widthPixels: 1.5,
          color: getParameterInk(ground, 235)
        })
      );
      return layers;
    },

    destroy() {
      destroyed = true;
      for (const set of sets.values()) set.reader.stop();
      for (const simulation of simulationSets.values()) simulation.reader.stop();
      resources.destroy();
    }
  };
}
