// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUPointPatternIndicesParameterValues,
  getGPURipleyDistanceFunctionsParameterValues,
  getGPURipleyParameterValues,
  GPU_CLARK_EVANS_LENGTH,
  GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH,
  GPU_QUADRAT_STATISTICS_LENGTH,
  GPU_RIPLEY_DISTANCE_FUNCTIONS_PARAMETER_LENGTH,
  GPU_RIPLEY_PARAMETER_LENGTH,
  GPUPointPatternIndices,
  GPURipley,
  GPURipleyDistanceFunctions,
  type GPURipleyDistanceEdgeCorrection,
  type GPURipleyEdgeCorrection
} from '@luma.gl/experimental/gpu-dataframe';
import {
  GPU_STUDY_WINDOW_SUMMARY_LENGTH,
  type GPUPolygonGeometryFields,
  GPUStudyWindowPreparation,
  GPURectangleClip,
  GPURandomPointsInPolygon,
  GPU_DOT_DENSITY_PARAMETER_LENGTH,
  getGPUDotDensityParameterValues,
  getGPUNeighborSearchParameterValues,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  createTransientView,
  GPUCommandGraph,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
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
import {
  createAnalysisFrameProjection,
  getViewportFrameBounds,
  projectLngLatToFrame
} from '../../engine/analysis-frame';
import {addKernelPass} from '../../engine/mode-kernels';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {GraphSwap} from '../../engine/graph-swap';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisPolygonLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer
} from '../../engine/layers';
import {createPolygonMeshBuffers} from '../../engine/polygon-buffers';
import {SpatialAnalysisResources} from '../../engine/resources';
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
import {ENVELOPE_RUNS, EnvelopeAccumulator, type PatternBounds} from './point-patterns-envelope';
import {
  addDensestPoint,
  addSharedCoordinates,
  type DensestPoint
} from './point-patterns-neighbours';
import {createPatternJitter} from './point-patterns-jitter';

/** A p-value for a readout: three decimals below 0.1, two above, `p < 0.001` below that. */
function formatPValue(p: number): string {
  if (!Number.isFinite(p)) return 'p undefined';
  return p < 0.001 ? 'p < 0.001' : `p = ${p.toFixed(p < 0.1 ? 3 : 2)}`;
}

/** Option state of the point-patterns scene. */
export type PointPatternOptions = {
  subject: 'observations' | 'places';
  groupCategory: string;
  placeCategory: string;
  maximumDistance: number;
  window: 'city' | 'rectangle' | 'view';
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
  /** Immutable unjittered positions, used by GPU jitter and duplicate counts. */
  baseBuffer: Buffer;
  jitter: Awaited<ReturnType<typeof createPatternJitter>> | null;
  jitterDirty: boolean;
  /** Raw `lng, lat` pairs, for the hover index. */
  lngLat: Float32Array;
  category: Uint8Array;
  categoryNames: readonly string[];
  positionsBuffer: Buffer;
  maskBuffer: Buffer;
  /** One complete-spatial-randomness realisation of the same size, drawn when "random" is shown. */
  randomBuffer: Buffer;
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
  quadratRatios: Buffer;
  windowOutline: Buffer;
  windowOutlineWeights: Buffer;
  windowEdgeCapacity: number;
  reader: SummaryReader;
};

/** The same pair statistics compiled over a second positions buffer, re-run for every random pattern. */
type SimulationSet = {
  key: string;
  subject: SubjectData;
  radiusCount: number;
  compiled: CompiledGPUCommandGraph<void>;
  positionsBuffer: Buffer;
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
  areas: Float32Array;
  ratios: Float32Array;
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
 * over a simulation positions buffer: each of 39 random patterns is a GPU sampling pass plus one
 * encode of the same compiled graph, never a recompile. The null model is complete spatial randomness
 * in the window rectangle or exactly on city land, sampled by GPURandomPointsInPolygon.
 *
 * The category, radius, edge corrections and jitter are buffer writes; window mode, null model,
 * radius count, quadrat grid and the F reference lattice select compiled graph variants.
 */
export async function createPointPatterns(
  ctx: SceneContext<PointPatternOptions>
): Promise<SceneInstance<PointPatternOptions>> {
  const {device} = ctx;
  const observations = ctx.datasets.get('chicago-nature');
  const places = ctx.datasets.get('chicago-places');
  const origin = observations.defaultOrigin;
  const projection = createAnalysisFrameProjection({kind: 'lambert-equal-area', origin});
  const city = await loadCityGeometry(ctx, origin, projection.project);
  const resources = new SpatialAnalysisResources(device, 'point-patterns');

  const makeSubject = (
    id: 'observations' | 'places',
    label: string,
    dataset: typeof observations
  ): SubjectData => {
    const base = projectLngLatToFrame(projection, dataset.column<Float32Array>('position'));
    const count = base.length / 2;
    return {
      id,
      label,
      count,
      baseBuffer: resources.createBuffer(`${id}-base`, base),
      jitter: null,
      jitterDirty: false,
      lngLat: dataset.column<Float32Array>('position'),
      category: dataset.column<Uint8Array>('category'),
      categoryNames: dataset.categories('category'),
      positionsBuffer: resources.createBuffer(`${id}-positions`, base.slice()),
      maskBuffer: resources.createBuffer(`${id}-mask`, new Uint32Array(count).fill(1)),
      randomBuffer: resources.createBuffer(`${id}-random`, count * 8),
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

  for (const subject of Object.values(subjects)) {
    subject.jitter = await createPatternJitter(
      resources,
      subject,
      projection.spatialContext,
      subject.id === 'observations' ? 11 : 23
    );
  }

  const cityWindow: PatternBounds = [...city.cityBoundsMeters];

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
        },
        ...city.cityPolygons.flatMap(polygon =>
          polygon.slice(1).map(ring => ({
            type: 'Feature' as const,
            properties: null,
            geometry: {type: 'Polygon' as const, coordinates: [ring as number[][]]}
          }))
        )
      ],
      (longitude, latitude) => projection.project(longitude, latitude)
    );
  };
  const cityHatch = {
    mesh: createPolygonMeshBuffers(resources, buildCityWindowMesh(), 'outside-window')
  };
  const hatches: Record<PointPatternOptions['window'], OutsideHatch> = {
    city: cityHatch,
    rectangle: cityHatch,
    view: cityHatch
  };
  const hatchValues = resources.createBuffer(
    'hatch-values',
    new Uint32Array(1 + city.cityPolygons.reduce((count, polygon) => count + polygon.length - 1, 0))
  );

  const ripleyParameters = resources.createParameterBuffer(
    'ripley',
    'float32',
    GPU_RIPLEY_PARAMETER_LENGTH
  );
  const distanceParameters = resources.createParameterBuffer(
    'distance',
    'float32',
    GPU_RIPLEY_DISTANCE_FUNCTIONS_PARAMETER_LENGTH
  );
  const indicesParameters = resources.createParameterBuffer(
    'indices',
    'float32',
    GPU_POINT_PATTERN_INDICES_PARAMETER_LENGTH
  );
  const neighborParameters = resources.createParameterBuffer(
    'neighbors',
    'float32',
    GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH
  );
  const windowBuffer = resources.createParameterBuffer('window', 'float32', 4);
  // Each handle is added to one graph and tracked before that graph, so disposal is reversed.
  // Only preparations reading rectangle positions or clip output need geometry invalidation.
  const mutablePreparations: GPUStudyWindowPreparation[] = [];
  const clipParameters = resources.createParameterBuffer('clip-window', 'float32', 4);
  const randomParameters = resources.createParameterBuffer(
    'random',
    'uint32',
    GPU_DOT_DENSITY_PARAMETER_LENGTH
  );
  const randomCounts = resources.createBuffer('random-counts', Uint32Array.of(0));
  const rectanglePositions = resources.createBuffer('rectangle-positions', 8 * 4);
  const cityColumns = city.cityWindowColumns;
  const cityBuffers = {
    positions: resources.createBuffer('city-positions', cityColumns.positions),
    featureOffsets: resources.createBuffer('city-features', cityColumns.featureOffsets),
    polygonOffsets: resources.createBuffer('city-polygons', cityColumns.polygonOffsets),
    ringOffsets: resources.createBuffer('city-rings', cityColumns.ringOffsets)
  };
  const rectangleFeatureOffsets = resources.createBuffer(
    'rectangle-features',
    Uint32Array.of(0, 1)
  );
  const rectanglePolygonOffsets = resources.createBuffer(
    'rectangle-polygons',
    Uint32Array.of(0, 1)
  );
  const rectangleRingOffsets = resources.createBuffer('rectangle-rings', Uint32Array.of(0, 4));

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
  let observedDirty = true;
  let envelopeDirty = true;
  let observedGeneration = 0;
  let envelopeGeneration = 0;
  let observedJobs: Job[] = [];
  let inFlight: Job | null = null;
  let inFlightSince = 0;
  // Hover reads are lazy and versioned against the exact observed encoding.
  let nearestRun = 0;
  let nearestReadyRun = -1;
  let nearestRequestedRun = -1;
  let nearestReadback: Float32Array | null = null;

  let measuredWindowArea = Number.NaN;
  let measuredLandShare = Number.NaN;
  let lastGuardDistance = Number.NaN;
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
    model === 'city'
      ? ctx.options.window === 'view'
        ? 'random on city land within the view'
        : 'random on city land'
      : ctx.options.window === 'view'
        ? 'random in the viewport rectangle'
        : 'random in the rectangle';
  const getSetKey = (options: PointPatternOptions) =>
    `${options.subject}|${options.radiusCount}|${options.quadratGrid}|${options.referenceGrid}|${options.window}`;
  const getSimulationKey = (options: PointPatternOptions) =>
    `${options.subject}|${options.radiusCount}|${options.referenceGrid}|${options.window}|${options.nullModel}`;
  const parseGrid = (value: string): [number, number] => [Number(value), Number(value)];
  const windowArea = () => measuredWindowArea;
  const toLngLat = (x: number, y: number): LngLat => projection.unproject(x, y) as LngLat;

  // ---- graphs ------------------------------------------------------------------------------

  /** Imports the one-feature city and rectangle layouts; all positions share the analysis frame. */
  function importWindowGeometry(graph: GPUCommandGraph<void>) {
    const cityGeometry: GPUPolygonGeometryFields = {
      positions: importGraphBuffer(graph, 'city-positions', cityBuffers.positions, 'float32x2'),
      featureOffsets: importGraphBuffer(
        graph,
        'city-features',
        cityBuffers.featureOffsets,
        'uint32'
      ),
      polygonOffsets: importGraphBuffer(
        graph,
        'city-polygons',
        cityBuffers.polygonOffsets,
        'uint32'
      ),
      ringOffsets: importGraphBuffer(graph, 'city-rings', cityBuffers.ringOffsets, 'uint32')
    };
    const rectangleGeometry: GPUPolygonGeometryFields = {
      positions: importGraphBuffer(graph, 'rectangle-positions', rectanglePositions, 'float32x2'),
      featureOffsets: importGraphBuffer(
        graph,
        'rectangle-features',
        rectangleFeatureOffsets,
        'uint32'
      ),
      polygonOffsets: importGraphBuffer(
        graph,
        'rectangle-polygons',
        rectanglePolygonOffsets,
        'uint32'
      ),
      ringOffsets: importGraphBuffer(graph, 'rectangle-rings', rectangleRingOffsets, 'uint32')
    };
    return {cityGeometry, rectangleGeometry};
  }

  /** Prepares the GPU window; clipped city rings also supply the viewport land-area readout. */
  function addStudyWindow(graph: GPUCommandGraph<void>, options: PointPatternOptions) {
    const {cityGeometry, rectangleGeometry} = importWindowGeometry(graph);
    let landGeometry = cityGeometry;
    if (options.window === 'view') {
      // Four Sutherland-Hodgman stages can each double the vertex count. Ring slots stay fixed.
      const positions = createTransientView(
        graph,
        'clipped-positions',
        'float32x2',
        cityGeometry.positions.length * 16
      );
      const ringOffsets = createTransientView(
        graph,
        'clipped-rings',
        'uint32',
        cityGeometry.ringOffsets.length
      );
      graph.add(
        new GPURectangleClip({
          id: 'clip-city',
          geometryType: 'polygons',
          ...cityGeometry,
          spatialContext: projection.spatialContext,
          parameters: clipParameters.importToGraph(graph),
          output: {
            positions,
            lineOffsets: ringOffsets,
            status: {
              count: createTransientView(graph, 'clip-count', 'uint32', 1),
              overflow: createTransientView(graph, 'clip-overflow', 'uint32', 1)
            }
          }
        })
      );
      landGeometry = {...cityGeometry, positions, ringOffsets};
    }
    const geometry = options.window === 'rectangle' ? rectangleGeometry : landGeometry;
    const summaryBuffer = resources.createBuffer(
      `${graph.id}-window-summary`,
      GPU_STUDY_WINDOW_SUMMARY_LENGTH * 4
    );
    const summary = importGraphBuffer(graph, 'window-summary', summaryBuffer, 'float32');
    const preparation = resources.track(
      new GPUStudyWindowPreparation({
        id: 'study-window',
        ...geometry,
        spatialContext: projection.spatialContext,
        output: {summary}
      })
    );
    graph.add(preparation);
    if (options.window !== 'city') mutablePreparations.push(preparation);
    let citySummaryBuffer = summaryBuffer;
    if (options.window === 'rectangle') {
      citySummaryBuffer = resources.createBuffer(
        `${graph.id}-city-summary`,
        GPU_STUDY_WINDOW_SUMMARY_LENGTH * 4
      );
      const cityPreparation = resources.track(
        new GPUStudyWindowPreparation({
          id: 'city-land-window',
          ...cityGeometry,
          spatialContext: projection.spatialContext,
          output: {
            summary: importGraphBuffer(graph, 'city-summary', citySummaryBuffer, 'float32')
          }
        })
      );
      graph.add(cityPreparation);
    }
    return {
      preparation,
      summary,
      summaryBuffer,
      landSummaryBuffer: options.window === 'rectangle' ? citySummaryBuffer : summaryBuffer,
      studyWindow: options.window === 'rectangle' ? ('rectangle' as const) : preparation,
      // Under `view`, the clipped land: the sampler skips the clip's empty and degenerate rings.
      randomGeometry: options.nullModel === 'city' ? landGeometry : rectangleGeometry
    };
  }

  async function buildSet(options: PointPatternOptions): Promise<PatternSet> {
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
      quadratAreas: buffer('quadrat-areas', quadratCells),
      quadratRatios: buffer('quadrat-ratios', quadratCells),
      quadratStatistics: buffer('quadrat-statistics', GPU_QUADRAT_STATISTICS_LENGTH),
      densest: buffer('densest', 2),
      duplicateShare: buffer('duplicate-share', 1)
    };
    const graph = new GPUCommandGraph<void>(device, {id: `point-patterns-${id}`});
    const window = addStudyWindow(graph, options);
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
    const quadratCounts = view('quadratCounts', 'uint32', quadratCells);
    const quadratAreas = view('quadratAreas', 'float32', quadratCells);
    const quadratRatios = view('quadratRatios', 'float32', quadratCells);
    const clarkEvans = view('clarkEvans', 'float32', GPU_CLARK_EVANS_LENGTH);
    const nearest = view('nearest', 'float32', subject.count);
    graph.add(
      new GPURipley({
        spatialContext: projection.spatialContext,
        studyWindow: window.studyWindow,
        id: 'ripley',
        positions,
        mask,
        parameters: ripleyParameters.importToGraph(graph),
        gridSize: SEARCH_GRID,
        radiusCount,
        output: {
          k: view('k', 'float32', radiusCount),
          l: view('l', 'float32', radiusCount),
          lMinusR: view('lMinusR', 'float32', radiusCount),
          radii: view('radii', 'float32', radiusCount)
        }
      })
    );
    graph.add(
      new GPURipleyDistanceFunctions({
        spatialContext: projection.spatialContext,
        studyWindow: window.studyWindow,
        id: 'ripley-distance',
        positions,
        mask,
        parameters: distanceParameters.importToGraph(graph),
        gridSize: SEARCH_GRID,
        referenceGrid,
        radiusCount,
        output: {
          g: view('g', 'float32', radiusCount),
          f: view('f', 'float32', radiusCount),
          j: view('j', 'float32', radiusCount)
        }
      })
    );
    graph.add(
      new GPUPointPatternIndices({
        spatialContext: projection.spatialContext,
        studyWindow: window.studyWindow,
        id: 'indices',
        positions,
        mask,
        parameters: indicesParameters.importToGraph(graph),
        gridSize: SEARCH_GRID,
        quadratGrid,
        output: {
          nearestNeighborDistances: nearest,
          clarkEvans: clarkEvans,
          quadratCounts: quadratCounts,
          quadratAreas: quadratAreas,
          quadratStatistics: view('quadratStatistics', 'float32', GPU_QUADRAT_STATISTICS_LENGTH)
        }
      })
    );
    addDensestPoint(graph, {
      positions,
      nearest,
      spatialContext: projection.spatialContext,
      parameters: neighborParameters.importToGraph(graph),
      output: view('densest', 'uint32', 2)
    });
    addSharedCoordinates(
      graph,
      importGraphBuffer(graph, 'base', subject.baseBuffer, 'float32x2'),
      mask,
      view('duplicateShare', 'float32', 1)
    );
    addKernelPass(graph, {
      id: 'quadrat-relative-counts',
      invocationCount: quadratCells,
      bindings: [
        {name: 'counts', view: quadratCounts, type: 'u32', access: 'read'},
        {name: 'areas', view: quadratAreas, type: 'f32', access: 'read'},
        {name: 'clark', view: clarkEvans, type: 'f32', access: 'read'},
        {name: 'summary', view: window.summary, type: 'f32', access: 'read'},
        {name: 'ratios', view: quadratRatios, type: 'f32', access: 'read_write'}
      ],
      body: `var nanBits = 0x7fc00000u;
        let expected = clark[clarkOffset] * areas[areasOffset + index] / summary[summaryOffset + 4u];
        ratios[ratiosOffset + index] = select(bitcast<f32>(nanBits), f32(counts[countsOffset + index]) / expected, expected > 0.0);`
    });
    const edges = window.preparation.getViews(graph).edges;
    const windowOutline = resources.createBuffer(`${id}-outline`, edges.length * 4);
    const windowOutlineWeights = resources.createBuffer(`${id}-outline-weights`, edges.length);
    addKernelPass(graph, {
      id: 'window-outline',
      invocationCount: edges.length / 4,
      bindings: [
        {name: 'edges', view: edges, type: 'f32', access: 'read'},
        {
          name: 'weights',
          view: importGraphBuffer(graph, 'outline-weights', windowOutlineWeights, 'float32'),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'outline',
          view: importGraphBuffer(graph, 'outline', windowOutline, 'float32'),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: `var valid = true;
        for (var component = 0u; component < 4u; component++) {
          valid = valid && (bitcast<u32>(edges[edgesOffset + 4u * index + component]) & 0x7fffffffu) < 0x7f800000u;
        }
        weights[weightsOffset + index] = select(0.0, 1.0, valid);
        for (var component = 0u; component < 4u; component++) {
          outline[outlineOffset + 4u * index + component] = select(0.0, edges[edgesOffset + 4u * index + component], valid);
        }`
    });
    const compiled = resources.track(await graph.compileAsync());
    const set: PatternSet = {
      key,
      subject,
      radiusCount,
      quadratGrid,
      compiled,
      nearestDistances: outputs.nearest,
      quadratCounts: outputs.quadratCounts,
      quadratRatios: outputs.quadratRatios,
      windowOutline,
      windowOutlineWeights,
      windowEdgeCapacity: edges.length / 4,
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
          {buffer: outputs.quadratCounts, size: quadratCells * 4},
          {buffer: outputs.quadratAreas, size: quadratCells * 4},
          {buffer: outputs.quadratRatios, size: quadratCells * 4},
          {buffer: window.summaryBuffer, size: GPU_STUDY_WINDOW_SUMMARY_LENGTH * 4},
          {buffer: window.landSummaryBuffer, size: GPU_STUDY_WINDOW_SUMMARY_LENGTH * 4},
          {buffer: outputs.densest, size: 8},
          {buffer: outputs.duplicateShare, size: 4}
        ],
        bytes => handleObservedSummary(set, bytes)
      )
    };
    return set;
  }

  /**
   * The envelope's graph: GPU sampling followed by Ripley and the distance functions over the
   * sampled positions. Unused rows are NaN; statistics share the observed graph's parameters.
   */
  async function buildSimulation(options: PointPatternOptions): Promise<SimulationSet> {
    const key = getSimulationKey(options);
    const subject = subjectOf(options);
    const radiusCount = Number(options.radiusCount);
    const referenceGrid = parseGrid(options.referenceGrid);
    const id = `${key.replace(/\|/g, '-')}-random`;
    const buffer = (name: string, floats: number) =>
      resources.createBuffer(`${id}-${name}`, floats * 4);
    const positionsBuffer = resources.createBuffer(`${id}-positions`, subject.count * 8);
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
    const window = addStudyWindow(graph, options);
    const positions = importGraphBuffer(
      graph,
      'positions',
      positionsBuffer,
      'float32x2',
      subject.count
    );
    const failedBuffer = resources.createBuffer(`${id}-failed`, 4);
    const parameters = randomParameters.importToGraph(graph);
    const requestedCount = importGraphBuffer(graph, 'random-counts', randomCounts, 'uint32');
    const failedCount = importGraphBuffer(graph, 'random-failed', failedBuffer, 'uint32');
    graph.add(
      new GPURandomPointsInPolygon({
        id: 'random-points',
        ...window.randomGeometry,
        spatialContext: projection.spatialContext,
        counts: requestedCount,
        parameters,
        output: {
          positions,
          dots: {
            ids: createTransientView(graph, 'random-ids', 'uint32', subject.count),
            status: {
              count: createTransientView(graph, 'random-count', 'uint32', 1),
              overflow: createTransientView(graph, 'random-overflow', 'uint32', 1)
            }
          },
          status: {failedCount}
        }
      })
    );
    addKernelPass(graph, {
      id: 'retain-first-random-pattern',
      invocationCount: subject.count * 2,
      bindings: [
        {name: 'positions', view: positions, type: 'f32', access: 'read'},
        {name: 'parameters', view: parameters, type: 'u32', access: 'read'},
        {
          name: 'displayPositions',
          view: importGraphBuffer(graph, 'random-display', subject.randomBuffer, 'float32x2'),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: `if (parameters[parametersOffset] == ${RANDOM_SEED}u) {
        displayPositions[displayPositionsOffset + index] = positions[positionsOffset + index];
      }`
    });
    const view = (name: keyof typeof outputs) =>
      importGraphBuffer(graph, name, outputs[name], 'float32', radiusCount);
    graph.add(
      new GPURipley({
        spatialContext: projection.spatialContext,
        studyWindow: window.studyWindow,
        id: 'ripley',
        positions,
        parameters: ripleyParameters.importToGraph(graph),
        gridSize: SEARCH_GRID,
        radiusCount,
        output: {
          k: view('k'),
          l: view('l'),
          lMinusR: view('lMinusR'),
          radii: view('radii')
        }
      })
    );
    graph.add(
      new GPURipleyDistanceFunctions({
        spatialContext: projection.spatialContext,
        studyWindow: window.studyWindow,
        id: 'ripley-distance',
        positions,
        parameters: distanceParameters.importToGraph(graph),
        gridSize: SEARCH_GRID,
        referenceGrid,
        radiusCount,
        output: {
          g: view('g'),
          f: view('f'),
          j: view('j')
        }
      })
    );
    const compiled = resources.track(await graph.compileAsync());
    const simulation: SimulationSet = {
      key,
      subject,
      radiusCount,
      compiled,
      positionsBuffer,
      reader: new SummaryReader(
        resources,
        `summary-${id}`,
        [
          {buffer: outputs.lMinusR, size: radiusCount * 4},
          {buffer: outputs.g, size: radiusCount * 4},
          {buffer: outputs.f, size: radiusCount * 4},
          {buffer: outputs.j, size: radiusCount * 4},
          {buffer: failedBuffer, size: 4},
          {buffer: window.summaryBuffer, size: GPU_STUDY_WINDOW_SUMMARY_LENGTH * 4}
        ],
        bytes => handleSimulationSummary(simulation, bytes)
      )
    };
    return simulation;
  }

  const setSwap = resources.track(new GraphSwap());

  /**
   * Selects the pattern and simulation sets for the current options once they are compiled
   * (cached in `sets` and `simulationSets`, which `resources` owns). The active sets keep
   * encoding until then; `afterSelect` runs in the commit, after the swap.
   */
  function requestSet(afterSelect?: () => void): Promise<boolean> {
    const options = {...ctx.options};
    markObservedChanged();
    markEnvelopeChanged();
    return setSwap.request({
      build: async () => {
        const key = getSetKey(options);
        let nextSet = sets.get(key);
        if (!nextSet) {
          nextSet = await buildSet(options);
          sets.set(key, nextSet);
        }
        const simulationKey = getSimulationKey(options);
        let nextSimulation = simulationSets.get(simulationKey);
        if (!nextSimulation) {
          nextSimulation = await buildSimulation(options);
          simulationSets.set(simulationKey, nextSimulation);
        }
        return {set: nextSet, simulation: nextSimulation};
      },
      commit: next => {
        active = next.set;
        activeSimulation = next.simulation;
        afterSelect?.();
        ctx.requestLayers();
      }
    });
  }

  /** Resets everything derived from the selected pattern set after a set swap. */
  function applySelection(): void {
    writeJitter(subjectOf(ctx.options), ctx.options.jitter);
    writeMask();
    observed = null;
    quadratClasses = null;
    for (const name of CORRECTIONS) delete edgeCurves[name];
    markObservedChanged();
    markEnvelopeChanged();
    lastTickKey = '';
  }

  // ---- change tracking ---------------------------------------------------------------------

  /** The observed pattern changed (jitter, grids, corrections): re-run the observed graph. */
  function markObservedChanged(): void {
    observedGeneration++;
    nearestRun++;
    nearestReadyRun = -1;
    nearestReadback = null;
    observedJobs = [];
    observedDirty = true;
    derivedDirty = true;
    lastChangeTime = performance.now();
  }

  /** The null changed (selection, window, null model, radii, corrections): restart the envelope. */
  function markEnvelopeChanged(): void {
    envelopeGeneration++;
    envelopeRunning = false;
    randomCount = 0;
    envelopeDirty = true;
    derivedDirty = true;
    // The live simulation's count: a pending radiusCount swap applies its own reset at commit.
    envelope.reset(activeSimulation?.radiusCount ?? Number(ctx.options.radiusCount));
    lastChangeTime = performance.now();
    ctx.setReadout('simulations', `0 of ${ENVELOPE_RUNS} random patterns`);
  }

  // ---- display derived from GPU summaries --------------------------------------------------

  function startEnvelope(): void {
    envelopeDirty = false;
    simulationsStarted = 0;
    envelopeRunning = selectedInside >= 3;
    randomCount = 0;
    publishCurveChart();
    publishDistanceChart();
    if (ctx.options.showRandom) ctx.requestLayers();
  }

  function derive(): void {
    derivedDirty = false;
    publishWindow();
    publishRadius();
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
    ctx.setReadout('windowLandShare', formatPercent(measuredLandShare));
    ctx.setReadout('outsideShare', formatPercent(1 - measuredLandShare));
    ctx.setReadout('nullName', getNullName(options.nullModel));
    ctx.setReadout('intensity', `${formatRate(selectedInside / (area / 1e6), 'km²')} of window`);
    const [west, south] = toLngLat(bounds[0], bounds[1]);
    const [east, north] = toLngLat(bounds[2], bounds[3]);
    const annotations: MapAnnotation[] =
      options.window === 'rectangle'
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
              title: `${formatPercent(1 - measuredLandShare)} outside city land in the bounding box`,
              text: 'Hatching marks areas outside the city boundary'
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

  /** Places the r-ring at the source coordinate of the GPU-selected row (display only). */
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
        `2 analysis graphs and 2 jitter graphs compiled once; ${formatCount(envelope.count)} random runs`
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
    const areasOffset = countsOffset + quadratCells;
    const areas = floats.slice(areasOffset, areasOffset + quadratCells);
    const ratios = floats.slice(areasOffset + quadratCells, areasOffset + quadratCells * 2);
    const summaryOffset = areasOffset + quadratCells * 2;
    const summary = floats.subarray(summaryOffset, summaryOffset + GPU_STUDY_WINDOW_SUMMARY_LENGTH);
    const landArea = floats[summaryOffset + GPU_STUDY_WINDOW_SUMMARY_LENGTH + 4];
    measuredWindowArea = summary[4];
    measuredLandShare = landArea / ((bounds[2] - bounds[0]) * (bounds[3] - bounds[1]));
    const helperOffset = summaryOffset + GPU_STUDY_WINDOW_SUMMARY_LENGTH * 2;
    const densestRow = words[helperOffset];
    // CPU lon/lat is used only to place a display annotation for the GPU-selected row.
    const coordinate =
      densestRow < set.subject.count
        ? projection.project(
            set.subject.lngLat[densestRow * 2],
            set.subject.lngLat[densestRow * 2 + 1]
          )
        : null;
    densest = coordinate
      ? {row: densestRow, neighbours: words[helperOffset + 1], x: coordinate[0], y: coordinate[1]}
      : null;
    const duplicateShare = floats[helperOffset + 2];
    ctx.setReadout(
      'duplicateShare',
      Number.isFinite(duplicateShare) ? formatPercent(duplicateShare, 1) : null
    );
    publishRadius();
    nearestReadyRun = nearestRun;
    selectedInside = clark[0];
    windowBuffer.write(summary.subarray(0, 4));
    observed = {radii, lMinusR, g, f, j, clark, quadrat, counts, areas, ratios};
    // Bootstrap the guard from GPU n/A, then rerun once if the last readback changed it.
    if (lastGuardDistance !== getGuardDistance()) {
      observedDirty = true;
      publishWindow();
      return;
    }
    if (envelopeDirty) startEnvelope();
    ctx.setCost({records: selectedInside, passes: set.compiled.stats.nodeOrder.length});
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

    // Clark-Evans: [n, observed, expected, ratio, standardError, z, p]. The expectation carries
    // Guard correction averages interior focus points; n includes all selected points in the window.
    const ratio = clark[3];
    ctx.setReadout('clarkEvansR', Number.isFinite(ratio) ? ratio.toFixed(2) : null);
    ctx.setReadout(
      'clarkEvans',
      Number.isFinite(ratio)
        ? `R = ${ratio.toFixed(3)} (${ratio < 1 ? 'clustered' : 'dispersed'}), z = ${formatSigned(clark[5], 1)}, ${formatPValue(clark[6])}`
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

    // Quadrat: [count, mean, residual variance, reduced chi-square, chi-square, df, p].
    // The fixed ratio breaks compare each cell with its own area-weighted expectation.
    ctx.setReadout('vmr', Number.isFinite(quadrat[3]) ? quadrat[3].toFixed(1) : null);
    ctx.setReadout(
      'chiSquare',
      Number.isFinite(quadrat[3])
        ? `chi-square ${formatCount(quadrat[4])} on ${formatCount(quadrat[5])} degrees of freedom, ${formatPValue(quadrat[6])}`
        : null
    );
    const nextQuadratKey = `${set.key}|${categoryOf(options)}|${bounds.join()}|${options.jitter}`;
    if (!quadratClasses || nextQuadratKey !== quadratKey) {
      quadratKey = nextQuadratKey;
      quadratClasses = makeQuadratTable(ratios, ground);
      ctx.setLegendData('quadrat', quadratClasses);
      ctx.requestLayers();
    }
    ctx.setChart(
      'quadratHistogram',
      buildQuadratChart(ratios, 1, quadrat[3], quadratClasses.table)
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
    const failed = new Uint32Array(bytes)[count * 4];
    if (failed > 0) {
      envelopeRunning = false;
      randomCount = 0;
      ctx.setReadout(
        'simulations',
        `GPU sampler could not place ${formatCount(failed)} points; envelope stopped`
      );
      ctx.requestLayers();
      return;
    }
    if (job.index === 0) {
      randomCount = selectedInside;
      ctx.requestLayers();
    }
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

  /** Guard width from the last GPU count/area, never smaller than one radius step. */
  function getGuardDistance(): number {
    const radiusStep = ctx.options.maximumDistance / Number(ctx.options.radiusCount);
    return selectedInside > 0 && measuredWindowArea > 0
      ? Math.max(2 * Math.sqrt(measuredWindowArea / selectedInside), radiusStep)
      : radiusStep;
  }

  /** Writes the shared parameter buffers; one job per frame, so the values are the job's. */
  function writeParameters(ripleyCorrection: GPURipleyEdgeCorrection): void {
    const options = ctx.options;
    const maximumDistance = options.maximumDistance;
    const parameterBounds = options.window === 'rectangle' ? bounds : 'study-window';
    neighborParameters.write(getGPUNeighborSearchParameterValues({bounds, radius: options.radius}));
    lastGuardDistance = getGuardDistance();
    ripleyParameters.write(
      getGPURipleyParameterValues({
        bounds: parameterBounds,
        maximumDistance,
        edgeCorrection: ripleyCorrection
      })
    );
    distanceParameters.write(
      getGPURipleyDistanceFunctionsParameterValues({
        bounds: parameterBounds,
        maximumDistance,
        edgeCorrection: options.distanceCorrection
      })
    );
    indicesParameters.write(
      getGPUPointPatternIndicesParameterValues({
        bounds: parameterBounds,
        clarkEvansCorrection: 'guard',
        guardDistance: lastGuardDistance,
        maximumDistance: Math.min(maximumDistance, 400)
      })
    );
  }

  /** Writes rectangle inputs; the GPU summary later supplies the quadrat raster bounds. */
  function writeWindowGeometry(): void {
    windowBuffer.write(Float32Array.from(bounds));
    const [x0, y0, x1, y1] = bounds;
    clipParameters.write(Float32Array.from(bounds));
    rectanglePositions.write(Float32Array.of(x0, y0, x1, y0, x1, y1, x0, y1));
    for (const preparation of mutablePreparations) preparation.invalidate();
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

  /** Updates only the radius; the subject's precompiled GPU graph writes the positions. */
  function writeJitter(subject: SubjectData, radius: number): void {
    if (subject.jitterApplied === radius) return;
    subject.jitterApplied = radius;
    subject.jitter?.radius.write(Float32Array.of(radius));
    subject.jitterDirty = true;
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
  const initialViewport = ctx.getViewport();
  if (ctx.options.window === 'view' && initialViewport) {
    bounds = getViewportFrameBounds(initialViewport, projection);
  }
  writeWindowGeometry();
  await requestSet();
  envelope.reset(Number(ctx.options.radiusCount));
  derive();
  lastChangeTime = performance.now();

  async function measurePattern(): Promise<void> {
    const set = active;
    if (!set || measuring || destroyed) return;
    measuring = true;
    ctx.setReadout('timing', 'measuring…');
    writeParameters(ctx.options.ripleyCorrection);
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
    'Isotropic weights are capped at 100. Polygon Hanisch and quadrat areas use lattices. Positions are Lambert equal-area float32 metres; city-scale distance distortion is below 0.01%.'
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
    nearestRun++;
    nearestReadyRun = -1;
    nearestReadback = null;
    set.compiled.encode(commandEncoder, {parameters: undefined});
    set.reader.request(commandEncoder);
  }

  function runSimulation(
    job: Extract<Job, {kind: 'simulation'}>,
    simulation: SimulationSet,
    commandEncoder: Parameters<SceneInstance<PointPatternOptions>['encode']>[0]
  ): void {
    randomCounts.write(Uint32Array.of(selectedInside));
    randomParameters.write(getGPUDotDensityParameterValues({seed: RANDOM_SEED + job.index}));
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
      [
        active?.compiled,
        activeSimulation?.compiled,
        ...Object.values(subjects).map(subject => subject.jitter?.compiled)
      ].filter(Boolean) as unknown as CompiledGPUCommandGraph<never>[],

    setOption(id, _value, state) {
      switch (id) {
        case 'subject':
        case 'groupCategory':
        case 'placeCategory':
          rememberCurve();
          void requestSet(applySelection);
          break;
        case 'radiusCount':
        case 'quadratGrid':
        case 'referenceGrid':
          void requestSet(applySelection);
          break;
        case 'jitter':
          writeJitter(subjectOf(state), state.jitter);
          markObservedChanged();
          markEnvelopeChanged();
          break;
        case 'window':
          bounds = [...cityWindow];
          lastViewSignature = null;
          writeWindowGeometry();
          void requestSet(applySelection);
          break;
        case 'maximumDistance':
        case 'ripleyCorrection':
        case 'distanceCorrection':
          for (const name of CORRECTIONS) delete edgeCurves[name];
          markObservedChanged();
          markEnvelopeChanged();
          break;
        case 'nullModel':
          void requestSet(applySelection);
          break;
        case 'compareCorrections':
          for (const name of CORRECTIONS) delete edgeCurves[name];
          markObservedChanged();
          break;
        case 'radius':
          markObservedChanged();
          publishRadius();
          break;
        case 'showRadius':
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
        quadratClasses = makeQuadratTable(observed.ratios, ground);
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
      if (active && nearestReadyRun === nearestRun && nearestRequestedRun !== nearestRun) {
        const run = nearestRun;
        const set = active;
        nearestRequestedRun = run;
        void set.nearestDistances
          .readAsync()
          .then(bytes => {
            if (!destroyed && active === set && run === nearestRun) {
              nearestReadback = new Float32Array(bytes.slice().buffer);
              ctx.requestLayers();
            }
          })
          .catch(() => {
            // A disposed scene or lost device cannot supply a tooltip value.
          });
      }
      const nearest = nearestReadback?.[row] ?? Number.NaN;
      const expected = observed.clark[2];
      const classIndex = nearestTable ? getClassIndexOf(nearestTable, nearest) : -1;
      const tooltipRows: TooltipRow[] = [
        {
          label: 'Nearest neighbour',
          value: !nearestReadback
            ? '…'
            : Number.isFinite(nearest)
              ? formatDistance(nearest)
              : 'none',
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
      return {
        title: categoryLabel(options),
        subtitle: nearestPlaceLabel(CHICAGO, lngLat, {maxDistanceMeters: 8000}) ?? undefined,
        rows: tooltipRows,
        highlight: {kind: 'point', coordinate: lngLat}
      };
    },

    encode(commandEncoder, frame) {
      for (const subject of Object.values(subjects)) {
        if (subject.jitterDirty && subject.jitter) {
          subject.jitter.compiled.encode(commandEncoder, {parameters: undefined});
          subject.jitterDirty = false;
        }
      }
      const set = active;
      const simulation = activeSimulation;
      if (!set || !simulation) return;
      if (ctx.options.window === 'view') {
        const viewBounds = getViewportFrameBounds(frame.viewport, projection);
        const signature = viewBounds.map(value => Math.round(value / 5)).join(',');
        if (signature !== lastViewSignature) {
          lastViewSignature = signature;
          bounds = [...viewBounds];
          writeWindowGeometry();
          markObservedChanged();
          markEnvelopeChanged();
        }
      }
      // Deliver finished readbacks and retry a copy that found no free ticket.
      set.reader.flush(commandEncoder);
      simulation.reader.flush(commandEncoder);
      const now = performance.now();
      if (setSwap.pending || measuring) return;
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

      // Counts relative to area-weighted expectation; zero-area quadrats carry NaN.
      if (options.showQuadrats && quadratClasses) {
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: `pattern-quadrats-${set.key}`,
            coordinateOrigin,
            analysisFrame: projection.frame,
            gridSize: set.quadratGrid,
            bounds: windowBuffer.buffer,
            values: set.quadratRatios,
            valueFormat: 'float32',
            noDataColor: [0, 0, 0, 0],
            colormap: 'uniform',
            ...getClassTableLayerProps(quadratClasses.table),
            outlineClasses: {color: getParameterInk(ground, 90), widthPixels: 0.7},
            color: [255, 255, 255, night ? 214 : 224]
          })
        );
      }

      // Outside-city hatching includes holes and stays in the same analysis frame.
      const hatch = hatches[options.window];
      layers.push(
        new SpatialAnalysisPolygonLayer({
          id: `pattern-outside-${options.window}`,
          coordinateOrigin,
          analysisFrame: projection.frame,
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
            analysisFrame: projection.frame,
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
            analysisFrame: projection.frame,
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
            analysisFrame: projection.frame,
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
            analysisFrame: projection.frame,
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
            analysisFrame: projection.frame,
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
          analysisFrame: projection.frame,
          segments: set.windowOutline,
          weights: set.windowOutlineWeights,
          instanceCount: set.windowEdgeCapacity,
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
