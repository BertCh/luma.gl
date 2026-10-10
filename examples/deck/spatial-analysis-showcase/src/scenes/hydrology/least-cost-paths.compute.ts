// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer, CommandEncoder} from '@luma.gl/core';
import {
  getGPUCostDistanceParameterValues,
  getGPUDistanceFieldParameterValues,
  GPU_COST_DISTANCE_NONE,
  GPU_COST_DISTANCE_PARAMETER_LENGTH,
  GPU_DISTANCE_FIELD_PARAMETER_LENGTH,
  GPUCostDistance,
  GPUCostDistancePath,
  GPUDistanceField
} from '@luma.gl/experimental/gpu-raster';
import {
  getGPUTerrainDerivativesParameterValues,
  GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH,
  GPUTerrainDerivatives
} from '@luma.gl/experimental/gpu-terrain';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer
} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {measureCompiledGraph} from '../../engine/vector-timing';
import type {SceneContext, SceneInstance} from '../scene';
import {
  addPathSegmentsPass,
  createCanyonGrid,
  createDixieGrid,
  formatDuration,
  formatInteger,
  getCellAt,
  getCellAtClamped,
  getCellCenter,
  getGridSettings,
  NAN_DECLARATIONS,
  type TerrainGrid
} from './b15-common';
import {
  BAND_COUNT,
  BAND_PALETTE,
  LAND_COVER_PALETTE,
  NO_COST_LIMIT_MINUTES,
  NO_DISTANCE_CAP_KILOMETERS,
  START_PALETTE
} from './least-cost-paths-style';

/** Option state of the least-cost-paths scene. */
export type LeastCostOptions = {
  display: 'cost' | 'bands' | 'friction' | 'distance' | 'allocation' | 'detour' | 'landcover';
  speed: number;
  slopePenalty: number;
  maxSlope: number;
  costLimit: number;
  rangeHours: number;
  bandMinutes: number;
  forest: number;
  openGround: number;
  builtUp: number;
  wetland: number;
  waterBarrier: boolean;
  placement: 'destination' | 'start' | 'add-start';
  secondStart: boolean;
  extraDelay: number;
  distanceAlgorithm: 'exact' | 'jump-flood-0' | 'jump-flood-1' | 'jump-flood-2';
  distanceCap: number;
  distanceSeeds: 'starts' | 'water' | 'built-up';
  showPath: boolean;
  showStraightLine: boolean;
  showBarriers: boolean;
  overlayOpacity: number;
};

const MAXIMUM_ITERATIONS = 384;
const PATH_CAPACITY = 4096;
const SOURCE_CAPACITY = 8;
/** Walking minutes per meter at the reference pace of 6 km/h on flat ground. */
const FLAT_MINUTES_PER_METER = 0.01;
const REFERENCE_SPEED = 6;
/** Tobler's hiking function magnitude: time grows as exp(3.5 * tan(slope)). */
const TOBLER_EXPONENT = 3.5;
const SUN_AZIMUTH_DEGREES = 315;
const SUN_ALTITUDE_DEGREES = 40;
const CLASS_TABLE_SIZE = 11;
const FRICTION_PARAMETER_LENGTH = 16;
const NO_SEED_CLASS = 255;

type PlaceDefinition = {
  name: string;
  lngLat?: readonly [number, number];
  rowColumn?: readonly [number, number];
};

type LandscapeConfig = {
  key: 'canyon' | 'dixie';
  label: string;
  starts: PlaceDefinition[];
  destination: PlaceDefinition;
  second: PlaceDefinition;
};

/** Default endpoints of both landscapes. The second start is added with the extra-start click mode. */
export const CANYON_PLACES = {
  start: {name: 'Grand Canyon Village', lngLat: [-112.1401, 36.0544]},
  destination: {name: 'Phantom Ranch', lngLat: [-112.0953, 36.107]},
  second: {name: 'North Rim, Bright Angel Point', lngLat: [-112.0525, 36.192]}
} as const;
export const DIXIE_PLACES = {
  start: {name: 'Greenville', rowColumn: [538, 587]},
  destination: {name: 'Lake Almanor east shore', rowColumn: [300, 50]},
  second: {name: 'Round Valley, south', rowColumn: [688, 470]}
} as const;

type Landscape = {
  config: LandscapeConfig;
  grid: TerrainGrid;
  resources: SpatialAnalysisResources;
  cellCount: number;
  flippedExtent: [number, number, number, number];
  /** Starts placed by clicks (the story start first). */
  starts: number[];
  /** Starts actually used: `starts` plus the optional second start. */
  activeStarts: number[];
  destination: number;
  costDirty: boolean;
  pathDirty: boolean;
  distanceDirty: boolean;
  displayDirty: boolean;
  setup: CompiledGPUCommandGraph<void>;
  cost: CompiledGPUCommandGraph<void>;
  path: CompiledGPUCommandGraph<void>;
  display: CompiledGPUCommandGraph<void>;
  distanceGraphs: Map<string, CompiledGPUCommandGraph<void>>;
  summary: SummaryReader;
  buffers: Record<string, Buffer>;
  parameters: {
    derivatives: ReturnType<SpatialAnalysisResources['createParameterBuffer']>;
    friction: ReturnType<SpatialAnalysisResources['createParameterBuffer']>;
    pace: ReturnType<SpatialAnalysisResources['createParameterBuffer']>;
    cost: ReturnType<SpatialAnalysisResources['createParameterBuffer']>;
    sources: ReturnType<SpatialAnalysisResources['createParameterBuffer']>;
    sourceCosts: ReturnType<SpatialAnalysisResources['createParameterBuffer']>;
    sourceCount: ReturnType<SpatialAnalysisResources['createParameterBuffer']>;
    target: ReturnType<SpatialAnalysisResources['createParameterBuffer']>;
    bands: ReturnType<SpatialAnalysisResources['createParameterBuffer']>;
    distance: ReturnType<SpatialAnalysisResources['createParameterBuffer']>;
    seed: ReturnType<SpatialAnalysisResources['createParameterBuffer']>;
    seedCount: ReturnType<SpatialAnalysisResources['createParameterBuffer']>;
  };
  drawCommands: DrawCommandBuffer;
  distanceGraphBuilder: (mode: string) => CompiledGPUCommandGraph<void>;
  frictionScale: ReturnType<SpatialAnalysisResources['createParameterBuffer']>;
  /** Ground distance from the nearest start to the destination, meters. */
  straightLineMeters: number;
};

/**
 * Least-cost travel and straight-line distance over two real landscapes at once: the Grand Canyon
 * (a slope-and-cliff friction) and the Dixie fire area (slope plus ESA WorldCover friction). Each
 * landscape owns four graphs: a one-off slope graph, a cost graph (`GPUCostDistance` over a friction
 * kernel), a path graph (`GPUCostDistancePath`) and a distance-field graph (`GPUDistanceField`), plus
 * a display graph. Every control is a parameter-buffer write except the distance algorithm, which
 * switches between compiled variants.
 */
export async function createLeastCostPaths(
  ctx: SceneContext<LeastCostOptions>
): Promise<SceneInstance<LeastCostOptions>> {
  const {device} = ctx;
  let destroyed = false;
  ctx.setStatus('Preparing the Grand Canyon and Dixie fire rasters...');
  const canyonGrid = createCanyonGrid(ctx.datasets.get('grand-canyon-dem'), 2);
  const dixieGrid = await createDixieGrid(ctx.datasets.get('dixie-fire'), ctx.signal);
  ctx.signal.throwIfAborted();

  const placeCell = (grid: TerrainGrid, place: PlaceDefinition): number =>
    place.rowColumn
      ? place.rowColumn[0] * grid.width + place.rowColumn[1]
      : getCellAtClamped(grid, place.lngLat![0], place.lngLat![1]);

  const landscapes: Landscape[] = [
    createLandscape(
      {
        key: 'canyon',
        label: 'Grand Canyon',
        starts: [CANYON_PLACES.start],
        destination: CANYON_PLACES.destination,
        second: CANYON_PLACES.second
      },
      canyonGrid
    ),
    createLandscape(
      {
        key: 'dixie',
        label: 'Dixie fire area',
        starts: [DIXIE_PLACES.start],
        destination: DIXIE_PLACES.destination,
        second: DIXIE_PLACES.second
      },
      dixieGrid
    )
  ];
  ctx.setStatus('');
  let measuring = false;

  function createLandscape(config: LandscapeConfig, grid: TerrainGrid): Landscape {
    const resources = new SpatialAnalysisResources(device, `least-cost-${config.key}`);
    const {width, height, cellCount} = grid;
    const flippedExtent: [number, number, number, number] = [
      grid.bounds[0],
      -grid.bounds[3],
      grid.bounds[2],
      -grid.bounds[1]
    ];
    const cells = (name: string) => resources.createBuffer(name, cellCount * 4);
    const buffers: Record<string, Buffer> = {
      elevation: resources.createBuffer('elevation', grid.elevation),
      classes: resources.createBuffer(
        'classes',
        Uint32Array.from(grid.landCover ?? new Uint8Array(cellCount))
      ),
      hillshade: cells('hillshade'),
      slope: cells('slope'),
      friction: cells('friction'),
      costs: cells('costs'),
      backLinks: cells('back-links'),
      bands: cells('bands'),
      bandCounts: resources.createBuffer('band-counts', BAND_COUNT * 4),
      converged: resources.createBuffer('converged', 4),
      iterations: resources.createBuffer('iteration-count', 4),
      pathIds: resources.createBuffer('path-ids', PATH_CAPACITY * 4),
      pathCount: resources.createBuffer('path-count', 4),
      pathOverflow: resources.createBuffer('path-overflow', 4),
      pathTotal: resources.createBuffer('path-total', 4),
      pathSegments: resources.createBuffer('path-segments', PATH_CAPACITY * 16),
      destinationCost: resources.createBuffer('destination-cost', 4),
      distances: cells('distances'),
      allocation: cells('allocation'),
      seedMask: cells('seed-mask'),
      costDisplay: cells('cost-display'),
      frictionDisplay: cells('friction-display'),
      barriers: cells('barriers'),
      detour: cells('detour'),
      landCoverIndex: cells('land-cover-index'),
      startPositions: resources.createBuffer('start-positions', SOURCE_CAPACITY * 8),
      startIds: resources.createBuffer(
        'start-ids',
        Uint32Array.from({length: SOURCE_CAPACITY}, (_, index) => index)
      ),
      destinationPosition: resources.createBuffer('destination-position', 8),
      straightLine: resources.createBuffer('straight-line', 16),
      startMarkers: resources.createBuffer('start-markers', SOURCE_CAPACITY * 8),
      destinationMarker: resources.createBuffer('destination-marker', 8)
    };
    const parameters = {
      derivatives: resources.createParameterBuffer(
        'derivatives-settings',
        'float32',
        GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH
      ),
      friction: resources.createParameterBuffer(
        'friction-parameters',
        'float32',
        FRICTION_PARAMETER_LENGTH
      ),
      pace: resources.createParameterBuffer('pace', 'float32', 4),
      cost: resources.createParameterBuffer(
        'cost-settings',
        'float32',
        GPU_COST_DISTANCE_PARAMETER_LENGTH
      ),
      sources: resources.createParameterBuffer('sources', 'uint32', SOURCE_CAPACITY),
      sourceCosts: resources.createParameterBuffer('source-costs', 'float32', SOURCE_CAPACITY),
      sourceCount: resources.createParameterBuffer('source-count', 'uint32', 1),
      target: resources.createParameterBuffer('target', 'uint32', 1),
      bands: resources.createParameterBuffer('band-thresholds', 'float32', BAND_COUNT),
      distance: resources.createParameterBuffer(
        'distance-settings',
        'float32',
        GPU_DISTANCE_FIELD_PARAMETER_LENGTH
      ),
      seed: resources.createParameterBuffer('seed-class', 'float32', 4),
      seedCount: resources.createParameterBuffer('seed-count', 'uint32', 1)
    };
    const frictionScale = resources.createParameterBuffer('friction-scale', 'float32', 2);
    const drawCommands = resources.track(
      new DrawCommandBuffer(device, {
        id: `least-cost-${config.key}-path-draw`,
        type: 'draw',
        commands: [{vertexCount: 6, instanceCount: 0}]
      })
    );

    // --- Setup graph: slope and hillshade, once ------------------------------------------------
    const setupGraph = new GPUCommandGraph<void>(device, {id: `least-cost-${config.key}-setup`});
    setupGraph.add(
      new GPUTerrainDerivatives({
        id: 'derivatives',
        width,
        height,
        elevation: {
          id: 'elevation',
          format: 'float32',
          storage: {
            kind: 'buffer',
            values: importGraphBuffer(
              setupGraph,
              'elevation',
              buffers.elevation,
              'float32',
              cellCount
            )
          }
        },
        settings: parameters.derivatives.importToGraph(setupGraph),
        slope: importGraphBuffer(setupGraph, 'slope', buffers.slope, 'float32', cellCount),
        hillshade: importGraphBuffer(
          setupGraph,
          'hillshade',
          buffers.hillshade,
          'float32',
          cellCount
        ),
        cellSizeMode: grid.cellSizeMode,
        rowDirection: 'south'
      })
    );
    const setup = resources.track(setupGraph.compile());

    // --- Cost graph: friction kernel, then GPUCostDistance ---------------------------------------
    const costGraph = new GPUCommandGraph<void>(device, {id: `least-cost-${config.key}-cost`});
    const frictionView = importGraphBuffer(
      costGraph,
      'friction',
      buffers.friction,
      'float32',
      cellCount
    );
    addFrictionPass(costGraph, {
      cellCount,
      slope: importGraphBuffer(costGraph, 'slope', buffers.slope, 'float32', cellCount),
      classes: importGraphBuffer(costGraph, 'classes', buffers.classes, 'uint32', cellCount),
      parameters: parameters.friction.importToGraph(costGraph),
      output: frictionView
    });
    costGraph.add(
      new GPUCostDistance({
        id: 'cost',
        width,
        height,
        friction: {
          id: 'friction',
          format: 'float32',
          storage: {kind: 'buffer', values: frictionView}
        },
        // friction = value * scale + offset: scale is 6 / walking speed, so one slider re-prices the map.
        frictionParameters: frictionScale.importToGraph(costGraph),
        settings: parameters.cost.importToGraph(costGraph),
        cellSizeMode: grid.cellSizeMode,
        sources: parameters.sources.importToGraph(costGraph),
        sourceCosts: parameters.sourceCosts.importToGraph(costGraph),
        sourceCount: parameters.sourceCount.importToGraph(costGraph),
        maxIterations: MAXIMUM_ITERATIONS,
        costs: importGraphBuffer(costGraph, 'costs', buffers.costs, 'float32', cellCount),
        backLinks: importGraphBuffer(
          costGraph,
          'back-links',
          buffers.backLinks,
          'uint32',
          cellCount
        ),
        bandThresholds: parameters.bands.importToGraph(costGraph),
        bands: importGraphBuffer(costGraph, 'bands', buffers.bands, 'uint32', cellCount),
        bandCounts: importGraphBuffer(
          costGraph,
          'band-counts',
          buffers.bandCounts,
          'uint32',
          BAND_COUNT
        ),
        converged: importGraphBuffer(costGraph, 'converged', buffers.converged, 'uint32', 1),
        iterationCount: importGraphBuffer(
          costGraph,
          'iteration-count',
          buffers.iterations,
          'uint32',
          1
        )
      })
    );
    const cost = resources.track(costGraph.compile());

    // --- Path graph: back-links to a cell list to segments ---------------------------------------
    const pathGraph = new GPUCommandGraph<void>(device, {id: `least-cost-${config.key}-path`});
    const pathIds = importGraphBuffer(
      pathGraph,
      'path-ids',
      buffers.pathIds,
      'uint32',
      PATH_CAPACITY
    );
    const pathCount = importGraphBuffer(pathGraph, 'path-count', buffers.pathCount, 'uint32', 1);
    const targetView = parameters.target.importToGraph(pathGraph);
    pathGraph.add(
      new GPUCostDistancePath({
        id: 'path',
        width,
        height,
        backLinks: importGraphBuffer(
          pathGraph,
          'back-links',
          buffers.backLinks,
          'uint32',
          cellCount
        ),
        target: targetView,
        output: {
          ids: pathIds,
          count: pathCount,
          overflow: importGraphBuffer(
            pathGraph,
            'path-overflow',
            buffers.pathOverflow,
            'uint32',
            1
          ),
          requiredCount: importGraphBuffer(
            pathGraph,
            'path-total-output',
            buffers.pathTotal,
            'uint32',
            1
          )
        }
      })
    );
    addPathSegmentsPass(pathGraph, {
      id: 'path-segments',
      grid,
      capacity: PATH_CAPACITY,
      ids: pathIds,
      count: pathCount,
      segments: importGraphBuffer(
        pathGraph,
        'path-segments',
        buffers.pathSegments,
        'float32',
        PATH_CAPACITY * 4
      )
    });
    addGatherCostPass(pathGraph, {
      costs: importGraphBuffer(pathGraph, 'costs', buffers.costs, 'float32', cellCount),
      target: targetView,
      output: importGraphBuffer(
        pathGraph,
        'destination-cost',
        buffers.destinationCost,
        'float32',
        1
      )
    });
    const path = resources.track(pathGraph.compile());

    // --- Display graph: drawable rasters from the cost and distance outputs --------------------
    const displayGraph = new GPUCommandGraph<void>(device, {
      id: `least-cost-${config.key}-display`
    });
    const displayCosts = importGraphBuffer(
      displayGraph,
      'costs',
      buffers.costs,
      'float32',
      cellCount
    );
    addKernelPass(displayGraph, {
      id: 'cost-display',
      bindings: [
        {name: 'costs', view: displayCosts, type: 'f32', access: 'read'},
        {
          name: 'display',
          view: importGraphBuffer(
            displayGraph,
            'cost-display',
            buffers.costDisplay,
            'float32',
            cellCount
          ),
          type: 'f32',
          access: 'read_write'
        }
      ],
      invocationCount: cellCount,
      declarations: NAN_DECLARATIONS,
      body: `let value = costs[costsOffset + index];
  display[displayOffset + index] = select(value, -1.0, isNaNValue(value) || isInfiniteValue(value));`
    });
    const paceView = parameters.pace.importToGraph(displayGraph);
    addKernelPass(displayGraph, {
      id: 'friction-display',
      bindings: [
        {
          name: 'friction',
          view: importGraphBuffer(displayGraph, 'friction', buffers.friction, 'float32', cellCount),
          type: 'f32',
          access: 'read'
        },
        {name: 'pace', view: paceView, type: 'f32', access: 'read'},
        {
          name: 'display',
          view: importGraphBuffer(
            displayGraph,
            'friction-display',
            buffers.frictionDisplay,
            'float32',
            cellCount
          ),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'barriers',
          view: importGraphBuffer(displayGraph, 'barriers', buffers.barriers, 'uint32', cellCount),
          type: 'u32',
          access: 'read_write'
        }
      ],
      invocationCount: cellCount,
      declarations: NAN_DECLARATIONS,
      body: `let value = friction[frictionOffset + index];
  let barrier = isNaNValue(value) || isInfiniteValue(value) || value < 0.0;
  // Minutes per kilometre at the chosen pace.
  display[displayOffset + index] = select(value * pace[paceOffset] * 1000.0, -1.0, barrier);
  barriers[barriersOffset + index] = select(0u, 1u, barrier);`
    });
    addKernelPass(displayGraph, {
      id: 'detour-display',
      bindings: [
        {name: 'costs', view: displayCosts, type: 'f32', access: 'read'},
        {
          name: 'distances',
          view: importGraphBuffer(
            displayGraph,
            'distances',
            buffers.distances,
            'float32',
            cellCount
          ),
          type: 'f32',
          access: 'read'
        },
        {name: 'pace', view: paceView, type: 'f32', access: 'read'},
        {
          name: 'detour',
          view: importGraphBuffer(displayGraph, 'detour', buffers.detour, 'float32', cellCount),
          type: 'f32',
          access: 'read_write'
        }
      ],
      invocationCount: cellCount,
      declarations: NAN_DECLARATIONS,
      body: `let travel = costs[costsOffset + index];
  let straight = distances[distancesOffset + index];
  let valid = !isNaNValue(travel) && !isInfiniteValue(travel) && !isInfiniteValue(straight) && straight > 1.0;
  // Travel time over the time to walk the straight line on flat ground at the same pace.
  let flatMinutes = straight * ${FLAT_MINUTES_PER_METER} * pace[paceOffset];
  detour[detourOffset + index] = select(-1.0, travel / max(flatMinutes, 0.000001), valid);`
    });
    addKernelPass(displayGraph, {
      id: 'land-cover-index',
      bindings: [
        {
          name: 'classes',
          view: importGraphBuffer(displayGraph, 'classes', buffers.classes, 'uint32', cellCount),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'indices',
          view: importGraphBuffer(
            displayGraph,
            'land-cover-index',
            buffers.landCoverIndex,
            'uint32',
            cellCount
          ),
          type: 'u32',
          access: 'read_write'
        }
      ],
      invocationCount: cellCount,
      body: `let code = classes[classesOffset + index];
  var result = 0xffffffffu;
  if (code == 10u) { result = 0u; }
  if (code == 20u) { result = 1u; }
  if (code == 30u) { result = 2u; }
  if (code == 40u) { result = 3u; }
  if (code == 50u) { result = 4u; }
  if (code == 60u) { result = 5u; }
  if (code == 80u) { result = 6u; }
  if (code == 90u || code == 95u) { result = 7u; }
  indices[indicesOffset + index] = result;`
    });
    const display = resources.track(displayGraph.compile());

    // --- Distance field variants (compiled on first use) ----------------------------------------
    const distanceGraphs = new Map<string, CompiledGPUCommandGraph<void>>();
    const distanceGraphBuilder = (mode: string): CompiledGPUCommandGraph<void> => {
      const graph = new GPUCommandGraph<void>(device, {
        id: `least-cost-${config.key}-distance-${mode}`
      });
      const maskView = importGraphBuffer(graph, 'seed-mask', buffers.seedMask, 'uint32', cellCount);
      addKernelPass(graph, {
        id: 'seed-mask',
        bindings: [
          {
            name: 'classes',
            view: importGraphBuffer(graph, 'classes', buffers.classes, 'uint32', cellCount),
            type: 'u32',
            access: 'read'
          },
          {name: 'seed', view: parameters.seed.importToGraph(graph), type: 'f32', access: 'read'},
          {name: 'mask', view: maskView, type: 'u32', access: 'read_write'}
        ],
        invocationCount: cellCount,
        // A cell is a seed when its WorldCover code equals the selected class (water 80, built-up 50).
        body: `let wanted = u32(seed[seedOffset] + 0.5);
  mask[maskOffset + index] = select(0u, 1u, classes[classesOffset + index] == wanted);`
      });
      graph.add(
        new GPUDistanceField({
          id: 'distance',
          width,
          height,
          mode: mode === 'exact' ? 'exact' : 'jump-flood',
          ...(mode === 'exact'
            ? {}
            : {jumpFloodRefinementPasses: Number(mode.slice(-1)) as 0 | 1 | 2}),
          settings: parameters.distance.importToGraph(graph),
          seedPositions: importGraphBuffer(
            graph,
            'start-seeds',
            buffers.startPositions,
            'float32x2',
            SOURCE_CAPACITY
          ),
          seedCount: parameters.seedCount.importToGraph(graph),
          seedMask: maskView,
          output: {
            distances: importGraphBuffer(
              graph,
              'distances',
              buffers.distances,
              'float32',
              cellCount
            ),
            allocation: importGraphBuffer(
              graph,
              'allocation',
              buffers.allocation,
              'uint32',
              cellCount
            )
          }
        })
      );
      return resources.track(graph.compile());
    };

    const landscape: Landscape = {
      config,
      grid,
      resources,
      cellCount,
      flippedExtent,
      starts: config.starts.map(place => placeCell(grid, place)),
      activeStarts: [],
      destination: placeCell(grid, config.destination),
      costDirty: true,
      pathDirty: true,
      distanceDirty: true,
      displayDirty: true,
      setup,
      cost,
      path,
      display,
      distanceGraphs,
      summary: null as unknown as SummaryReader,
      buffers,
      parameters: {...parameters},
      drawCommands,
      distanceGraphBuilder,
      frictionScale,
      straightLineMeters: 0
    };
    landscape.summary = new SummaryReader(
      resources,
      `${config.key}`,
      [
        {buffer: buffers.destinationCost, size: 4},
        {buffer: buffers.pathCount, size: 4},
        {buffer: buffers.pathOverflow, size: 4},
        {buffer: buffers.pathTotal, size: 4},
        {buffer: buffers.converged, size: 4},
        {buffer: buffers.iterations, size: 4},
        {buffer: buffers.bandCounts, size: BAND_COUNT * 4},
        {buffer: buffers.pathIds, size: PATH_CAPACITY * 4}
      ],
      bytes => {
        if (!destroyed) readSummary(landscape, bytes);
      }
    );
    parameters.derivatives.write(
      getGPUTerrainDerivativesParameterValues({
        ...getGridSettings(grid),
        azimuthDegrees: SUN_AZIMUTH_DEGREES,
        altitudeDegrees: SUN_ALTITUDE_DEGREES
      })
    );
    return landscape;
  }

  // --- Option-driven parameter writes --------------------------------------------------------------
  const getFrictionScale = () => REFERENCE_SPEED / ctx.options.speed;
  const getCostLimit = () =>
    ctx.options.costLimit >= NO_COST_LIMIT_MINUTES ? Infinity : ctx.options.costLimit;

  /** Multiplier of each WorldCover code index (code / 10); a negative value is a barrier. */
  function getClassTable(): Float32Array {
    const {forest, openGround, builtUp, wetland, waterBarrier} = ctx.options;
    const table = new Float32Array(CLASS_TABLE_SIZE).fill(1);
    table[1] = forest;
    table[2] = openGround;
    table[3] = openGround;
    table[4] = openGround;
    table[5] = builtUp;
    table[6] = openGround;
    table[7] = openGround;
    table[8] = waterBarrier ? -1 : 4;
    table[9] = wetland;
    table[10] = openGround;
    return table;
  }

  function writeLandscapeParameters(landscape: Landscape): void {
    const {grid, parameters} = landscape;
    const options = ctx.options;
    const friction = new Float32Array(FRICTION_PARAMETER_LENGTH);
    friction[0] = options.slopePenalty;
    friction[1] = options.maxSlope;
    if (landscape.config.key === 'dixie') friction.set(getClassTable(), 4);
    else friction.fill(1, 4, 4 + CLASS_TABLE_SIZE);
    parameters.friction.write(friction);
    landscape.frictionScale.write(Float32Array.of(getFrictionScale(), 0));
    parameters.pace.write(Float32Array.of(getFrictionScale(), 0, 0, 0));
    parameters.cost.write(
      getGPUCostDistanceParameterValues({...getGridSettings(grid), costLimit: getCostLimit()})
    );
    parameters.bands.write(
      Float32Array.from({length: BAND_COUNT}, (_, band) => (band + 1) * options.bandMinutes)
    );
    const seedClass =
      landscape.config.key === 'dixie' && options.distanceSeeds !== 'starts'
        ? options.distanceSeeds === 'water'
          ? 80
          : 50
        : NO_SEED_CLASS;
    parameters.seed.write(Float32Array.of(seedClass, 0, 0, 0));
    writeSeedCount(landscape);
    parameters.distance.write(
      getGPUDistanceFieldParameterValues({
        bounds: landscape.flippedExtent,
        gridSize: [grid.width, grid.height],
        maxDistance:
          options.distanceCap >= NO_DISTANCE_CAP_KILOMETERS ? Infinity : options.distanceCap * 1000
      })
    );
    landscape.costDirty = true;
    landscape.pathDirty = true;
    landscape.distanceDirty = true;
    landscape.displayDirty = true;
  }

  /** Point seeds of the distance field: the starts, unless a land cover class supplies the seeds. */
  function writeSeedCount(landscape: Landscape): void {
    const rasterSeeds = landscape.config.key === 'dixie' && ctx.options.distanceSeeds !== 'starts';
    landscape.parameters.seedCount.write(
      Uint32Array.of(rasterSeeds ? 0 : landscape.activeStarts.length)
    );
  }

  function writeEndpoints(landscape: Landscape): void {
    const {grid, parameters, buffers} = landscape;
    const sources = new Uint32Array(SOURCE_CAPACITY);
    const costs = new Float32Array(SOURCE_CAPACITY);
    const positions = new Float32Array(SOURCE_CAPACITY * 2);
    landscape.activeStarts = getActiveStarts(landscape);
    landscape.activeStarts.forEach((cell, index) => {
      sources[index] = cell;
      costs[index] = index === 0 ? 0 : ctx.options.extraDelay;
      const [x, y] = getCellCenter(grid, cell);
      // The distance field wants row 0 at the minimum y: negate y, as in `flippedExtent`.
      positions[2 * index] = x;
      positions[2 * index + 1] = -y;
    });
    parameters.sources.write(sources);
    parameters.sourceCosts.write(costs);
    parameters.sourceCount.write(Uint32Array.of(landscape.activeStarts.length));
    writeSeedCount(landscape);
    parameters.target.write(Uint32Array.of(landscape.destination));
    buffers.startPositions.write(positions);
    const display = new Float32Array(SOURCE_CAPACITY * 2);
    landscape.activeStarts.forEach((cell, index) => {
      const [x, y] = getCellCenter(grid, cell);
      display[2 * index] = x;
      display[2 * index + 1] = y;
    });
    // Marker positions (y up) share the buffer layout of the seeds but not their sign.
    buffers.startMarkers.write(display);
    const destination = getCellCenter(grid, landscape.destination);
    buffers.destinationMarker.write(Float32Array.from(destination));
    // Straight line from the nearest start (by ground distance) to the destination.
    updateStraightLine(landscape);
    landscape.costDirty = true;
    landscape.pathDirty = true;
    landscape.distanceDirty = true;
    landscape.displayDirty = true;
  }

  function getActiveStarts(landscape: Landscape): number[] {
    const starts = [...landscape.starts];
    if (ctx.options.secondStart) starts.push(placeCell(landscape.grid, landscape.config.second));
    return starts.slice(-SOURCE_CAPACITY);
  }

  /** Straight line from the nearest start (by ground distance) to the destination. */
  function updateStraightLine(landscape: Landscape): void {
    const {grid, buffers} = landscape;
    const destination = getCellCenter(grid, landscape.destination);
    let nearest = landscape.activeStarts[0];
    let nearestDistance = Infinity;
    for (const cell of landscape.activeStarts) {
      const [x, y] = getCellCenter(grid, cell);
      const distance = Math.hypot(x - destination[0], y - destination[1]);
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearest = cell;
      }
    }
    const [startX, startY] = getCellCenter(grid, nearest);
    buffers.straightLine.write(Float32Array.of(startX, startY, destination[0], destination[1]));
    landscape.straightLineMeters = nearestDistance;
  }

  // --- Summary readback --------------------------------------------------------------------------
  function readSummary(landscape: Landscape, bytes: ArrayBuffer): void {
    const {grid, config} = landscape;
    const words = new Uint32Array(bytes);
    const floats = new Float32Array(bytes);
    const prefix = config.key;
    const minutes = floats[0];
    const pathCount = words[1];
    const overflow = words[2];
    const total = words[3];
    const converged = words[4];
    const iterations = words[5];
    ctx.setReadout(
      `${prefix}Converged`,
      `${converged ? 'yes' : 'no (iteration limit)'} after ${iterations} of ${MAXIMUM_ITERATIONS}`
    );
    if (total === 0 || !Number.isFinite(minutes)) {
      ctx.setReadout(`${prefix}Time`, 'destination unreached');
      ctx.setReadout(`${prefix}Path`, 'no route: the destination is walled in or beyond the limit');
      ctx.setReadout(`${prefix}Climb`, '-');
      ctx.setReadout(`${prefix}Detour`, '-');
    } else {
      const ids = words.subarray(
        6 + BAND_COUNT,
        6 + BAND_COUNT + Math.min(pathCount, PATH_CAPACITY)
      );
      let length = 0;
      let ascent = 0;
      let descent = 0;
      for (let step = 1; step < ids.length; step++) {
        const [x0, y0] = getCellCenter(grid, ids[step - 1]);
        const [x1, y1] = getCellCenter(grid, ids[step]);
        length += Math.hypot(x1 - x0, y1 - y0);
        // Ids run destination to start: ascending toward the start means descending for the walker.
        const rise = grid.elevation[ids[step - 1]] - grid.elevation[ids[step]];
        if (rise > 0) descent += rise;
        else ascent -= rise;
      }
      ctx.setReadout(`${prefix}Time`, formatDuration(minutes));
      ctx.setReadout(
        `${prefix}Path`,
        `${(length / 1000).toFixed(1)} km on ${formatInteger(total)} cells${overflow ? ' (capacity reached)' : ''}`
      );
      // The walker travels start to destination, so the sums above (destination-first) are mirrored.
      ctx.setReadout(
        `${prefix}Climb`,
        `${formatInteger(descent)} m up, ${formatInteger(ascent)} m down`
      );
      const straightMinutes =
        (landscape.straightLineMeters * FLAT_MINUTES_PER_METER * REFERENCE_SPEED) /
        ctx.options.speed;
      ctx.setReadout(
        `${prefix}Detour`,
        `${(minutes / Math.max(straightMinutes, 1e-6)).toFixed(2)}× the straight line over flat ground (${(landscape.straightLineMeters / 1000).toFixed(1)} km)`
      );
    }
    ctx.setReadout(
      `${prefix}Bands`,
      Array.from(words.subarray(6, 6 + BAND_COUNT), count => formatInteger(count)).join(' / ')
    );
  }

  /** The distance graph compiled for `mode`, built on first use. */
  function getDistanceGraph(landscape: Landscape, mode: string): CompiledGPUCommandGraph<void> {
    let graph = landscape.distanceGraphs.get(mode);
    if (!graph) {
      graph = landscape.distanceGraphBuilder(mode);
      landscape.distanceGraphs.set(mode, graph);
    }
    return graph;
  }

  for (const landscape of landscapes) {
    writeLandscapeParameters(landscape);
    writeEndpoints(landscape);
    getDistanceGraph(landscape, ctx.options.distanceAlgorithm);
  }

  async function timeDistanceAlgorithms(): Promise<void> {
    if (measuring || destroyed) return;
    measuring = true;
    ctx.setReadout('distanceTiming', 'measuring...');
    try {
      const canyon = landscapes[0];
      const parts: string[] = [];
      for (const mode of ['exact', 'jump-flood-0', 'jump-flood-1', 'jump-flood-2']) {
        const timing = await measureCompiledGraph(device, getDistanceGraph(canyon, mode), {
          parameters: undefined,
          completionBuffer: canyon.buffers.converged,
          signal: ctx.signal
        });
        parts.push(
          `${mode === 'exact' ? 'exact' : `JFA+${mode.slice(-1)}`} ${timing.milliseconds.toFixed(2)} ms`
        );
        if (destroyed) return;
      }
      ctx.setReadout(
        'distanceTiming',
        `${parts.join(' · ')} (${canyon.grid.width}×${canyon.grid.height})`
      );
    } catch {
      // Aborted or destroyed while measuring.
    } finally {
      measuring = false;
    }
  }

  // --- Interaction -------------------------------------------------------------------------------
  let dragging: Landscape | null = null;
  const findLandscape = (coordinate: readonly [number, number]) => {
    for (const landscape of landscapes) {
      if (getCellAt(landscape.grid, coordinate[0], coordinate[1]) >= 0) return landscape;
    }
    return null;
  };

  return {
    getCompiledGraphs: () =>
      landscapes.flatMap(landscape => [
        landscape.setup,
        landscape.cost,
        landscape.path,
        landscape.display,
        getDistanceGraph(landscape, ctx.options.distanceAlgorithm)
      ]),

    setOption(id, _value, state) {
      if (id === 'distanceAlgorithm') {
        for (const landscape of landscapes) {
          getDistanceGraph(landscape, state.distanceAlgorithm);
          landscape.distanceDirty = true;
          landscape.displayDirty = true;
        }
      } else if (id === 'extraDelay' || id === 'secondStart') {
        for (const landscape of landscapes) writeEndpoints(landscape);
      } else if (
        id === 'display' ||
        id === 'overlayOpacity' ||
        id === 'showPath' ||
        id === 'showStraightLine' ||
        id === 'showBarriers' ||
        id === 'placement'
      ) {
        // Display-only options: no parameter writes.
      } else {
        for (const landscape of landscapes) writeLandscapeParameters(landscape);
      }
      ctx.requestLayers();
    },

    onAction(id) {
      if (id === 'timeDistance') void timeDistanceAlgorithms();
      if (id === 'resetEndpoints') {
        for (const landscape of landscapes) {
          landscape.starts = [placeCell(landscape.grid, landscape.config.starts[0])];
          landscape.destination = placeCell(landscape.grid, landscape.config.destination);
          writeEndpoints(landscape);
        }
        ctx.requestLayers();
      }
    },

    encode(commandEncoder: CommandEncoder, frame) {
      const mode = ctx.options.distanceAlgorithm;
      for (const landscape of landscapes) {
        if (frame.frameIndex < 2) landscape.setup.encode(commandEncoder, {parameters: undefined});
        let changed = false;
        if (landscape.costDirty || frame.frameIndex < 2) {
          landscape.cost.encode(commandEncoder, {parameters: undefined});
          landscape.costDirty = false;
          landscape.pathDirty = true;
          landscape.displayDirty = true;
          changed = true;
        }
        if (landscape.pathDirty) {
          landscape.path.encode(commandEncoder, {parameters: undefined});
          // Instance-count word of the one 16-byte draw record: one segment per path cell.
          commandEncoder.copyBufferToBuffer({
            sourceBuffer: landscape.buffers.pathCount,
            sourceOffset: 0,
            destinationBuffer: landscape.drawCommands.buffer,
            destinationOffset: 4,
            size: 4
          });
          landscape.pathDirty = false;
          changed = true;
        }
        if (landscape.distanceDirty) {
          getDistanceGraph(landscape, mode).encode(commandEncoder, {parameters: undefined});
          landscape.distanceDirty = false;
          landscape.displayDirty = true;
        }
        if (landscape.displayDirty) {
          landscape.display.encode(commandEncoder, {parameters: undefined});
          landscape.displayDirty = false;
        }
        if (changed) landscape.summary.request(commandEncoder);
        landscape.summary.flush(commandEncoder);
      }
    },

    getLayers() {
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      const alpha = Math.round(options.overlayOpacity * 255);
      for (const landscape of landscapes) {
        const {grid, buffers} = landscape;
        const origin: [number, number, number] = [grid.origin[0], grid.origin[1], 0];
        const key = landscape.config.key;
        const rasterProps = {
          coordinateOrigin: origin,
          gridSize: [grid.width, grid.height] as const,
          bounds: grid.bounds,
          rowOrigin: 'north' as const
        };
        const rangeMinutes = options.rangeHours * 60;
        layers.push(
          new SpatialAnalysisRasterLayer({
            ...rasterProps,
            id: `least-cost-${key}-hillshade`,
            values: buffers.hillshade,
            valueFormat: 'float32',
            colormap: 'grayscale',
            valueRange: [0, 1],
            color: [255, 255, 255, dark ? 190 : 170]
          })
        );
        switch (options.display) {
          case 'cost':
            layers.push(
              new SpatialAnalysisRasterLayer({
                ...rasterProps,
                id: `least-cost-${key}-cost`,
                values: buffers.costDisplay,
                valueFormat: 'float32',
                colormap: 'lajolla',
                valueRange: [0, rangeMinutes],
                discardAtOrBelow: -0.5,
                color: [255, 255, 255, alpha]
              })
            );
            break;
          case 'bands':
            layers.push(
              new SpatialAnalysisRasterLayer({
                ...rasterProps,
                id: `least-cost-${key}-bands`,
                values: buffers.bands,
                valueFormat: 'uint32',
                colormap: 'category',
                palette: BAND_PALETTE,
                noDataValue: GPU_COST_DISTANCE_NONE,
                noDataColor: [0, 0, 0, 0],
                color: [255, 255, 255, alpha]
              })
            );
            break;
          case 'friction':
            layers.push(
              new SpatialAnalysisRasterLayer({
                ...rasterProps,
                id: `least-cost-${key}-friction`,
                values: buffers.frictionDisplay,
                valueFormat: 'float32',
                colormap: 'inferno',
                valueRange: [5, 60],
                discardAtOrBelow: -0.5,
                color: [255, 255, 255, alpha]
              })
            );
            break;
          case 'distance':
            layers.push(
              new SpatialAnalysisRasterLayer({
                ...rasterProps,
                id: `least-cost-${key}-distance`,
                values: buffers.distances,
                valueFormat: 'float32',
                colormap: 'cividis',
                valueScale: 0.001,
                valueRange: [0, 20],
                noDataColor: [0, 0, 0, 0],
                color: [255, 255, 255, alpha]
              })
            );
            break;
          case 'allocation':
            layers.push(
              new SpatialAnalysisRasterLayer({
                ...rasterProps,
                id: `least-cost-${key}-allocation`,
                values: buffers.allocation,
                valueFormat: 'uint32',
                colormap: 'category',
                palette: START_PALETTE,
                noDataColor: [0, 0, 0, 0],
                color: [255, 255, 255, alpha]
              })
            );
            break;
          case 'detour':
            layers.push(
              new SpatialAnalysisRasterLayer({
                ...rasterProps,
                id: `least-cost-${key}-detour`,
                values: buffers.detour,
                valueFormat: 'float32',
                colormap: 'magma',
                valueRange: [1, 4],
                discardAtOrBelow: -0.5,
                color: [255, 255, 255, alpha]
              })
            );
            break;
          case 'landcover':
            if (key === 'dixie') {
              layers.push(
                new SpatialAnalysisRasterLayer({
                  ...rasterProps,
                  id: `least-cost-${key}-landcover`,
                  values: buffers.landCoverIndex,
                  valueFormat: 'uint32',
                  colormap: 'category',
                  palette: LAND_COVER_PALETTE,
                  noDataColor: [0, 0, 0, 0],
                  color: [255, 255, 255, alpha]
                })
              );
            }
            break;
        }
        if (options.showBarriers) {
          layers.push(
            new SpatialAnalysisRasterLayer({
              ...rasterProps,
              id: `least-cost-${key}-barriers`,
              values: buffers.barriers,
              valueFormat: 'uint32',
              colormap: 'mask',
              color: dark ? [255, 90, 90, 190] : [200, 30, 30, 170],
              noDataColor: [0, 0, 0, 0]
            })
          );
        }
        if (options.showStraightLine) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: `least-cost-${key}-straight`,
              coordinateOrigin: origin,
              segments: buffers.straightLine,
              instanceCount: 1,
              widthPixels: 1.5,
              color: dark ? [255, 255, 255, 190] : [30, 30, 30, 190]
            })
          );
        }
        if (options.showPath) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: `least-cost-${key}-path-halo`,
              coordinateOrigin: origin,
              segments: buffers.pathSegments,
              drawCommands: landscape.drawCommands,
              drawCommandIndex: 0,
              widthPixels: 6.5,
              color: [0, 0, 0, 200]
            }),
            new SpatialAnalysisSegmentLayer({
              id: `least-cost-${key}-path`,
              coordinateOrigin: origin,
              segments: buffers.pathSegments,
              drawCommands: landscape.drawCommands,
              drawCommandIndex: 0,
              widthPixels: 3.5,
              color: [255, 80, 70, 255]
            })
          );
        }
        layers.push(
          new SpatialAnalysisPointLayer({
            id: `least-cost-${key}-starts-halo`,
            coordinateOrigin: origin,
            positions: buffers.startMarkers,
            instanceCount: landscape.activeStarts.length,
            radiusPixels: 10,
            color: [20, 20, 20, 255]
          }),
          new SpatialAnalysisPointLayer({
            id: `least-cost-${key}-starts`,
            coordinateOrigin: origin,
            positions: buffers.startMarkers,
            instanceCount: landscape.activeStarts.length,
            radiusPixels: 7,
            values: buffers.startIds,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: START_PALETTE.map(([r, g, b]) => [r, g, b, 255] as const)
          }),
          new SpatialAnalysisPointLayer({
            id: `least-cost-${key}-destination-halo`,
            coordinateOrigin: origin,
            positions: buffers.destinationMarker,
            instanceCount: 1,
            radiusPixels: 11,
            color: [255, 255, 255, 255]
          }),
          new SpatialAnalysisPointLayer({
            id: `least-cost-${key}-destination`,
            coordinateOrigin: origin,
            positions: buffers.destinationMarker,
            instanceCount: 1,
            radiusPixels: 7.5,
            color: [235, 40, 60, 255]
          })
        );
      }
      return layers;
    },

    onClick(event) {
      if (!event.coordinate) return false;
      const landscape = findLandscape(event.coordinate);
      if (!landscape) return false;
      const cell = getCellAt(landscape.grid, event.coordinate[0], event.coordinate[1]);
      const {placement} = ctx.options;
      if (placement === 'destination') landscape.destination = cell;
      else if (placement === 'start') landscape.starts = [cell];
      else {
        if (landscape.starts.length >= SOURCE_CAPACITY) landscape.starts.shift();
        landscape.starts.push(cell);
      }
      writeEndpoints(landscape);
      ctx.requestLayers();
      return true;
    },

    onDragStart(event) {
      const viewport = ctx.getViewport();
      if (!viewport || !event.coordinate) return false;
      for (const landscape of landscapes) {
        const [longitude, latitude] = landscape.grid.projection.unproject(
          ...getCellCenter(landscape.grid, landscape.destination)
        );
        const [x, y] = viewport.project([longitude, latitude]);
        if (Math.hypot(x - event.pixel[0], y - event.pixel[1]) < 26) {
          dragging = landscape;
          return true;
        }
      }
      return false;
    },

    onDrag(event) {
      if (!dragging || !event.coordinate) return;
      const cell = getCellAt(dragging.grid, event.coordinate[0], event.coordinate[1]);
      if (cell < 0 || cell === dragging.destination) return;
      dragging.destination = cell;
      dragging.pathDirty = true;
      writeDestination(dragging);
    },

    onDragEnd() {
      dragging = null;
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      const landscape = findLandscape(event.coordinate);
      if (!landscape) return null;
      const cell = getCellAt(landscape.grid, event.coordinate[0], event.coordinate[1]);
      const {grid} = landscape;
      let text = `Elevation ${formatInteger(grid.elevation[cell])} m`;
      if (grid.landCover) {
        const code = grid.landCover[cell];
        const names: Record<number, string> = {
          10: 'Tree cover',
          20: 'Shrubland',
          30: 'Grassland',
          40: 'Cropland',
          50: 'Built-up',
          60: 'Bare / sparse',
          80: 'Water',
          90: 'Wetland'
        };
        text += `\n${names[code] ?? `Class ${code}`}`;
      }
      return text;
    },

    destroy() {
      destroyed = true;
      for (const landscape of landscapes) {
        landscape.summary.stop();
        landscape.resources.destroy();
      }
    }
  };

  function writeDestination(landscape: Landscape): void {
    landscape.parameters.target.write(Uint32Array.of(landscape.destination));
    const destination = getCellCenter(landscape.grid, landscape.destination);
    landscape.buffers.destinationMarker.write(Float32Array.from(destination));
    updateStraightLine(landscape);
  }
}

/**
 * `friction = base * exp(3.5 * strength * tan(slope)) * landCoverMultiplier` in walking minutes per
 * meter at 6 km/h (Tobler's hiking function magnitude, symmetric), NaN (a barrier) above the cliff
 * limit or where the multiplier is negative. `parameters` is
 * `[strength, maxSlopeDegrees, 0, 0, multiplier by WorldCover code / 10 ...]`.
 */
function addFrictionPass(
  graph: GPUCommandGraph<void>,
  props: {
    cellCount: number;
    slope: ReturnType<typeof importGraphBuffer<'float32', void>>;
    classes: ReturnType<typeof importGraphBuffer<'uint32', void>>;
    parameters: ReturnType<typeof importGraphBuffer<'float32', void>>;
    output: ReturnType<typeof importGraphBuffer<'float32', void>>;
  }
): void {
  addKernelPass(graph, {
    id: 'friction',
    bindings: [
      {name: 'slope', view: props.slope, type: 'f32', access: 'read'},
      {name: 'classes', view: props.classes, type: 'u32', access: 'read'},
      {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'friction', view: props.output, type: 'f32', access: 'read_write'}
    ],
    invocationCount: props.cellCount,
    declarations: NAN_DECLARATIONS,
    body: `let slopeDegrees = slope[slopeOffset + index];
  let code = min(classes[classesOffset + index] / 10u, ${CLASS_TABLE_SIZE - 1}u);
  let multiplier = parameters[parametersOffset + 4u + code];
  let steepness = tan(radians(min(slopeDegrees, 89.0)));
  let walking = ${FLAT_MINUTES_PER_METER} * exp(${TOBLER_EXPONENT} * parameters[parametersOffset] * steepness) * multiplier;
  let barrier = isNaNValue(slopeDegrees) || slopeDegrees > parameters[parametersOffset + 1u] || multiplier < 0.0;
  friction[frictionOffset + index] = select(walking, getQuietNaN(), barrier);`
  });
}

/** `output = costs[target]`: the cost at the destination, for the readout. */
function addGatherCostPass(
  graph: GPUCommandGraph<void>,
  props: {
    costs: ReturnType<typeof importGraphBuffer<'float32', void>>;
    target: ReturnType<typeof importGraphBuffer<'uint32', void>>;
    output: ReturnType<typeof importGraphBuffer<'float32', void>>;
  }
): void {
  addKernelPass(graph, {
    id: 'destination-cost',
    bindings: [
      {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
      {name: 'targetCell', view: props.target, type: 'u32', access: 'read'},
      {name: 'result', view: props.output, type: 'f32', access: 'read_write'}
    ],
    invocationCount: 1,
    body: `result[resultOffset] = costs[costsOffset + targetCell[targetCellOffset]];`
  });
}
