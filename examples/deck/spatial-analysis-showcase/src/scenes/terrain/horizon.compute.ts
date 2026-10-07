// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer, CommandEncoder} from '@luma.gl/core';
import {
  GPUCommandGraph,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPURasterExtremaPyramid,
  getGPURasterExtremaPyramidLayout
} from '@luma.gl/experimental/gpu-raster';
import {
  GPUPointHorizonProfile,
  GPUPointHorizonVisibility,
  GPUProfilePeaks,
  GPUTerrainHorizon,
  GPU_POINT_HORIZON_PARAMETER_LENGTH,
  GPU_POINT_HORIZON_VISIBILITY_PARAMETER_LENGTH,
  GPU_PROFILE_PEAKS_PARAMETER_LENGTH,
  GPU_TERRAIN_HORIZON_PARAMETER_LENGTH,
  GPU_TERRAIN_VISIBILITY,
  getGPUPointHorizonParameterValues,
  getGPUPointHorizonVisibilityParameterValues,
  getGPUProfilePeaksParameterValues,
  getGPUTerrainCurvatureCoefficient,
  getGPUTerrainHorizonParameterValues
} from '@luma.gl/experimental/gpu-terrain';
import {getClassTableLayerProps} from '../../cartography/class-table';
import {hexToRgba, MAP_INK} from '../../cartography/hue-registry';
import type {MapAnnotation} from '../../cartography/types';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer
} from '../../engine/layers';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {formatCompiledGraphTiming, measureCompiledGraph} from '../../engine/vector-timing';
import type {SceneContext, SceneInstance, TooltipContent, TooltipRow} from '../scene';
import {
  createElevationBand,
  formatDistance,
  getAzimuthDegrees,
  prepareAlpsTerrain,
  toFloat32,
  toUint32
} from './b14b-terrain';
import {createDemProbe, createTerrainDemFromTerrain} from './cpu-dem';
import {
  buildPanoramaChart,
  buildRayProfileChart,
  buildSkylinePeaksChart,
  type PanoramaPeak
} from './horizon-chart';
import {buildPeakCatalogue, getLabelOrder} from './horizon-catalogue';
import {castSightRay, getRayEnd, getRefractionValue, type SightRay} from './horizon-ray';
import {
  FAR_PEAK_METERS,
  formatBearing,
  getCompassLabel,
  getHorizonSubtitle,
  makeHorizonTables,
  MINIMUM_PEAK_ELEVATION_METERS,
  SKY_VIEW_RAMP_RANGE,
  SKY_VIEW_RANGE,
  type HorizonOptions,
  type HorizonView
} from './horizon-style';
import {BARE_EARTH_CHIP, demSampleLine} from './terrain-furniture';
import {createTerrainGround, rasterizeGlacierMask} from './terrain-ground';
import {
  getObserverColors,
  getVisibilitySymbolColors,
  type TerrainGroundTone
} from './terrain-palettes';
import {formatElevationMeters, loadAlpsContext, terrainLabel} from './terrain-places';

/** Stride of the grid-wide `GPUTerrainHorizon`: the 2,331 x 2,181 DEM becomes 777 x 727, 40 m cells. */
const GRID_STRIDE = 3;
/** Compile-time ray length of the skyline, longer than the farthest corner of the window. */
const HORIZON_MAXIMUM_DISTANCE = 20000;
/** Rays drawn as the fan of the `rays` view. */
const FAN_RAY_COUNT = 36;
const FAN_CAPACITY = 64;
const PEAK_MARKER_CAPACITY = 64;
const DRAG_RADIUS_PIXELS = 22;
const PICK_RADIUS_PIXELS = 14;
/** Ground step of the CPU sight ray. */
const RAY_STEP_METERS = 20;
/** A bearing selects a catalogue peak when it is this close to the peak's own, degrees. */
const SELECT_TOLERANCE_DEGREES = 0.75;
/** Charts, labels and readouts are republished at most this often while the eye is dragged. */
const PUBLISH_DELAY_MILLISECONDS = 60;
const MAXIMUM_PEAK_LABELS = 5;
const MAXIMUM_PLAIN_LABELS = 7;
const MAXIMUM_SKYLINE_ROWS = 8;
const MAXIMUM_SKYLINE_MARKERS = 6;
/** The eye stands at the Gornergrat when it is within this of the starting cell, metres. */
const HOME_RADIUS_METERS = 300;

const VISIBILITY = GPU_TERRAIN_VISIBILITY;
const VISIBILITY_NAMES: Record<number, string> = {
  [VISIBILITY.hidden]: 'Hidden',
  [VISIBILITY.visible]: 'Visible',
  [VISIBILITY.marginal]: 'Marginal',
  [VISIBILITY.outOfRange]: 'Out of range',
  [VISIBILITY.noData]: 'No data'
};

/** Skyline discontinuity: the ring is not drawn between two rays whose hits are this far apart. */
const isSkylineJump = (near: number, far: number): boolean =>
  Math.abs(near - far) > 0.15 * Math.max(near, far) + 120;

const isValidAngle = (angle: number): boolean => Number.isFinite(angle) && angle > -89.5;

const wrapDegrees = (degrees: number): number => ((degrees % 360) + 360) % 360;

/** Smallest angle between two bearings, degrees. */
const getBearingDifference = (left: number, right: number): number => {
  const difference = Math.abs(wrapDegrees(left) - wrapDegrees(right));
  return Math.min(difference, 360 - difference);
};

/** The `bearing` option runs from 180 to 540 so it lines up with the panorama axis. */
const toBearingOption = (azimuthDegrees: number): number => {
  const wrapped = wrapDegrees(azimuthDegrees);
  return wrapped < 180 ? wrapped + 360 : wrapped;
};

/** Everything that depends on the compile-time skyline options; rebuilt as a unit. */
type Skyline = {
  key: string;
  azimuthCount: number;
  compiled: CompiledGPUCommandGraph<void>;
  angle: Buffer;
  distance: Buffer;
  samples: Buffer;
  prominence: Buffer;
  refinedIndex: Buffer;
  refinedValue: Buffer;
  peakMask: Buffer;
  converged: Buffer;
  ring: Buffer;
  weights: Buffer;
  fan: Buffer;
  peakMarkers: Buffer;
  reader: SummaryReader;
};

type VisibilityGraph = {
  key: string;
  compiled: CompiledGPUCommandGraph<void>;
  reader: SummaryReader;
};

type GridGraph = {
  key: string;
  directions: number;
  compiled: CompiledGPUCommandGraph<void>;
  horizon: Buffer;
};

type SkylinePeak = {
  azimuth: number;
  angle: number;
  prominence: number;
  distance: number;
  meters: [number, number];
};

/** The ray the reader cast: what the map draws and what the profile chart and the verdict say. */
type CastRay = {
  ray: SightRay;
  targetIndex: number;
  verdict: string;
  ridgeAngle: number | null;
};

/**
 * "Which peaks can I see from Gornergrat?" The skyline (`GPUPointHorizonProfile`) is the highest
 * elevation angle at every bearing from one eye. `GPUProfilePeaks` finds the maxima of that
 * circular profile, `GPUPointHorizonVisibility` compares each catalogue peak's own angle with the
 * highest ground in front of it, and `GPUTerrainHorizon` repeats the horizon for every cell of a
 * coarser grid (sky-view factor). The ground is the chapter relief; the CPU casts the one ray the
 * reader clicks (`horizon-ray.ts`) so the profile chart can show what the GPU does for every peak.
 */
export async function createHorizon(
  ctx: SceneContext<HorizonOptions>
): Promise<SceneInstance<HorizonOptions>> {
  const {device} = ctx;
  ctx.setStatus('Loading the wide terrain and the OpenStreetMap peaks');
  const context = await loadAlpsContext(ctx.datasets.get('alps-context'), ctx.signal);
  ctx.signal.throwIfAborted();
  const dataset = ctx.datasets.get('alps-dem-wide');
  const terrain = prepareAlpsTerrain(dataset, 1);
  const coarse = prepareAlpsTerrain(dataset, GRID_STRIDE);
  const dem = createTerrainDemFromTerrain(terrain);
  const probe = createDemProbe(dem);
  const {width, height, pixelCount} = terrain;
  const origin: [number, number, number] = [terrain.origin[0], terrain.origin[1], 0];

  const ground = createTerrainGround({
    dem,
    device,
    ground: ctx.ground(),
    glacierMask: rasterizeGlacierMask(context.glacierGeoJson, dem)
  });
  ctx.setStatus('Building the shaded relief');
  await ground.prepare();
  ctx.signal.throwIfAborted();

  const resources = new SpatialAnalysisResources(device, 'horizon');
  let destroyed = false;
  let tables = makeHorizonTables(ctx.ground());
  ctx.setLegendData('tables', tables);

  // --- Buffers ----------------------------------------------------------------------------------
  const elevationBuffer = resources.createBuffer('elevation', terrain.elevation);
  const validityBuffer = resources.createBuffer('validity', terrain.validity);
  const coarseElevation = resources.createBuffer('coarse-elevation', coarse.elevation);
  const coarseValidity = resources.createBuffer('coarse-validity', coarse.validity);
  const skyViewBuffer = resources.createBuffer('sky-view', coarse.pixelCount * 4);
  const horizonSettings = resources.createParameterBuffer(
    'horizon-settings',
    'float32',
    GPU_POINT_HORIZON_PARAMETER_LENGTH
  );
  const visibilitySettings = resources.createParameterBuffer(
    'visibility-settings',
    'float32',
    GPU_POINT_HORIZON_VISIBILITY_PARAMETER_LENGTH
  );
  const peakSettings = resources.createParameterBuffer(
    'peak-settings',
    'float32',
    GPU_PROFILE_PEAKS_PARAMETER_LENGTH
  );
  const gridSettings = resources.createParameterBuffer(
    'grid-settings',
    'float32',
    GPU_TERRAIN_HORIZON_PARAMETER_LENGTH
  );
  const observerRow = resources.createBuffer('observer', new Float32Array(4));
  const observerMarker = resources.createBuffer('observer-marker', new Float32Array(2));
  const rayBefore = resources.createBuffer('ray-before', new Float32Array(4));
  const rayAfter = resources.createBuffer('ray-after', new Float32Array(4));

  // --- The shared extrema pyramid (built once) --------------------------------------------------
  const setupGraph = new GPUCommandGraph<void>(device, {id: 'horizon-setup'});
  const setupElevation = createElevationBand(
    setupGraph,
    'setup',
    elevationBuffer,
    validityBuffer,
    pixelCount
  );
  const pyramidLayout = getGPURasterExtremaPyramidLayout(width, height, {
    firstBlockSize: 4,
    footprint: 'bilinear'
  });
  const pyramidCombined = resources.createBuffer('pyramid', pyramidLayout.length * 8);
  setupGraph.add(
    new GPURasterExtremaPyramid({
      id: 'extrema-pyramid',
      width,
      height,
      input: setupElevation,
      firstBlockSize: 4,
      footprint: 'bilinear',
      combined: importGraphBuffer(
        setupGraph,
        'pyramid',
        pyramidCombined,
        'float32',
        2 * pyramidLayout.length
      )
    })
  );
  const setupCompiled = resources.track(setupGraph.compile());

  // --- The eye and the catalogue ----------------------------------------------------------------
  const home = terrain.observers.find(place => place.name === 'Gornergrat');
  let observer: [number, number] = home
    ? terrain.getPixel(home.longitude, home.latitude)
    : [width / 2, height / 2];
  const homeMeters = terrain.getMeters(observer[0], observer[1]);
  const catalogue = buildPeakCatalogue(context, dem, terrain.getMeters, homeMeters);
  const targetCount = catalogue.length;
  const labelOrder = getLabelOrder(catalogue);
  const targetRows = new Float32Array(targetCount * 4);
  catalogue.forEach((peak, index) => {
    targetRows.set([peak.column, peak.row, 0, 0], index * 4);
  });
  const targetBuffer = resources.createBuffer('targets', targetRows);
  const targetMarkers = resources.createBuffer(
    'target-markers',
    Float32Array.from(catalogue.flatMap(peak => peak.meters))
  );
  const targetVisibility = resources.createBuffer('target-visibility', targetCount * 4);
  const targetDetails = resources.createBuffer('target-details', targetCount * 16);
  const targetOnSkyline = resources.createBuffer('target-on-skyline', targetCount * 4);

  // --- State ------------------------------------------------------------------------------------
  let skylineDirty = true;
  let visibilityDirty = true;
  let skyline: Skyline | null = null;
  let visibilityGraph: VisibilityGraph | null = null;
  let gridGraph: GridGraph | null = null;
  let gridDirty = true;
  let gridReady = false;
  let gridReadPending = false;
  let gridReadNeeded = true;
  let gridValues = new Float32Array(0);
  let gridDirectionCount = 16;
  const gridRing = resources.track(
    new GPUReadbackRing(device, {id: 'horizon-grid-read', byteLength: 192})
  );
  let setupDone = false;
  let dragging = false;
  let angles = new Float32Array(0);
  let distances = new Float32Array(0);
  let hits = new Float32Array(0);
  let visibilityReady = false;
  let visibilityCodes = new Uint32Array(targetCount);
  let details = new Float32Array(targetCount * 4);
  let targetDistance = new Float64Array(targetCount);
  let targetAzimuth = new Float64Array(targetCount);
  let skylinePeaks: SkylinePeak[] = [];
  let peakMarkerCount = 0;
  let highestSkyline: {angle: number; azimuth: number; meters: [number, number]} | null = null;
  let selectedIndex = -1;
  let selectedSkylinePeak = -1;
  let castRay: CastRay | null = null;
  let publishTimer: ReturnType<typeof setTimeout> | null = null;
  let lastFurnitureKey = '';
  let highlightOwned = false;

  const getCurvature = (): number =>
    getGPUTerrainCurvatureCoefficient(getRefractionValue(ctx.options.refraction));

  const getEyeMeters = (): [number, number] => terrain.getMeters(observer[0], observer[1]);
  const getEyeLngLat = (): [number, number] =>
    terrain.getLongitudeLatitude(observer[0], observer[1]);
  const getMaximumMeters = (): number => ctx.options.maxDistance * 1000;
  const unproject = (x: number, y: number): [number, number] => terrain.projection.unproject(x, y);

  function refreshTargetGeometry(): void {
    const [eyeX, eyeY] = getEyeMeters();
    catalogue.forEach((peak, index) => {
      const dx = peak.meters[0] - eyeX;
      const dy = peak.meters[1] - eyeY;
      targetDistance[index] = Math.hypot(dx, dy);
      targetAzimuth[index] = getAzimuthDegrees(dx, dy);
    });
  }

  const isInRange = (index: number): boolean => targetDistance[index] <= getMaximumMeters();

  // --- Graph builders ---------------------------------------------------------------------------
  function getSharedPyramid(graph: GPUCommandGraph<void>) {
    return {
      layout: pyramidLayout,
      combined: importGraphBuffer(
        graph,
        'pyramid',
        pyramidCombined,
        'float32',
        2 * pyramidLayout.length
      )
    };
  }

  function getSkylineKey(): string {
    const options = ctx.options;
    return [options.projection, options.traversal, options.azimuthCount, options.peakWindow].join(
      '|'
    );
  }

  function destroySkyline(): void {
    if (!skyline) return;
    skyline.reader.stop();
    resources.release(skyline.compiled);
    for (const buffer of [
      skyline.angle,
      skyline.distance,
      skyline.samples,
      skyline.prominence,
      skyline.refinedIndex,
      skyline.refinedValue,
      skyline.peakMask,
      skyline.converged,
      skyline.ring,
      skyline.weights,
      skyline.fan,
      skyline.peakMarkers
    ]) {
      resources.release(buffer);
    }
    skyline = null;
  }

  function buildSkyline(): Skyline {
    const options = ctx.options;
    const azimuthCount = Number(options.azimuthCount);
    const window = Number(options.peakWindow);
    const angle = resources.createBuffer('skyline-angle', azimuthCount * 4);
    const distance = resources.createBuffer('skyline-distance', azimuthCount * 4);
    const samples = resources.createBuffer('skyline-samples', azimuthCount * 4);
    const prominence = resources.createBuffer('peak-prominence', azimuthCount * 4);
    const refinedIndex = resources.createBuffer('peak-index', azimuthCount * 4);
    const refinedValue = resources.createBuffer('peak-value', azimuthCount * 4);
    const peakMask = resources.createBuffer('peak-mask', azimuthCount * 4);
    const converged = resources.createBuffer('peak-converged', 4);
    const ring = resources.createBuffer('skyline-ring', azimuthCount * 16);
    const weights = resources.createBuffer('skyline-weights', azimuthCount * 4);
    const fan = resources.createBuffer('skyline-fan', FAN_CAPACITY * 16);
    const peakMarkers = resources.createBuffer('peak-markers', PEAK_MARKER_CAPACITY * 8);
    const offsets = resources.createBuffer('skyline-offsets', Uint32Array.of(0, azimuthCount));
    const graph = new GPUCommandGraph<void>(device, {id: 'horizon-skyline'});
    const angleView = importGraphBuffer(graph, 'angle', angle, 'float32', azimuthCount);
    graph.add(
      new GPUPointHorizonProfile({
        id: 'skyline',
        width,
        height,
        elevation: createElevationBand(
          graph,
          'skyline',
          elevationBuffer,
          validityBuffer,
          pixelCount
        ),
        projection: options.projection,
        traversal: options.traversal,
        pyramid: options.traversal === 'pyramid' ? getSharedPyramid(graph) : undefined,
        azimuthCount,
        maximumDistance: HORIZON_MAXIMUM_DISTANCE,
        cellSize: terrain.groundCellSize,
        observers: importGraphBuffer(graph, 'observer', observerRow, 'float32x4', 1),
        settings: horizonSettings.importToGraph(graph),
        skylineAngle: angleView,
        distance: importGraphBuffer(graph, 'distance', distance, 'float32', azimuthCount),
        samples: importGraphBuffer(graph, 'samples', samples, 'uint32', azimuthCount)
      })
    );
    graph.add(
      new GPUProfilePeaks({
        id: 'skyline-peaks',
        values: angleView,
        offsets: importGraphBuffer(graph, 'offsets', offsets, 'uint32', 2),
        settings: peakSettings.importToGraph(graph),
        // The skyline is a full circle: its ends wrap.
        wrap: true,
        window,
        prominence: importGraphBuffer(graph, 'prominence', prominence, 'float32', azimuthCount),
        refinedIndex: importGraphBuffer(
          graph,
          'refined-index',
          refinedIndex,
          'float32',
          azimuthCount
        ),
        refinedValue: importGraphBuffer(
          graph,
          'refined-value',
          refinedValue,
          'float32',
          azimuthCount
        ),
        peakMask: importGraphBuffer(graph, 'peak-mask', peakMask, 'uint32', azimuthCount),
        converged: importGraphBuffer(graph, 'converged', converged, 'uint32', 1)
      })
    );
    const compiled = resources.track(graph.compile());
    const reader = new SummaryReader(
      resources,
      'skyline',
      [
        {buffer: angle, size: azimuthCount * 4},
        {buffer: distance, size: azimuthCount * 4},
        {buffer: samples, size: azimuthCount * 4},
        {buffer: prominence, size: azimuthCount * 4},
        {buffer: refinedIndex, size: azimuthCount * 4},
        {buffer: converged, size: 4}
      ],
      bytes => applySkyline(bytes)
    );
    return {
      key: getSkylineKey(),
      azimuthCount,
      compiled,
      angle,
      distance,
      samples,
      prominence,
      refinedIndex,
      refinedValue,
      peakMask,
      converged,
      ring,
      weights,
      fan,
      peakMarkers,
      reader
    };
  }

  function getVisibilityKey(): string {
    const options = ctx.options;
    return [options.projection, options.traversal].join('|');
  }

  function buildVisibility(): VisibilityGraph {
    const options = ctx.options;
    const graph = new GPUCommandGraph<void>(device, {id: 'horizon-visibility'});
    graph.add(
      new GPUPointHorizonVisibility({
        id: 'visibility',
        width,
        height,
        elevation: createElevationBand(
          graph,
          'visibility',
          elevationBuffer,
          validityBuffer,
          pixelCount
        ),
        projection: options.projection,
        traversal: options.traversal,
        pyramid: options.traversal === 'pyramid' ? getSharedPyramid(graph) : undefined,
        maximumDistance: HORIZON_MAXIMUM_DISTANCE,
        cellSize: terrain.groundCellSize,
        observers: importGraphBuffer(graph, 'observer', observerRow, 'float32x4', 1),
        targets: importGraphBuffer(graph, 'targets', targetBuffer, 'float32x4', targetCount),
        settings: visibilitySettings.importToGraph(graph),
        visibility: importGraphBuffer(graph, 'visibility', targetVisibility, 'uint32', targetCount),
        details: importGraphBuffer(graph, 'details', targetDetails, 'float32x4', targetCount)
      })
    );
    const compiled = resources.track(graph.compile());
    const reader = new SummaryReader(
      resources,
      'visibility',
      [
        {buffer: targetVisibility, size: targetCount * 4},
        {buffer: targetDetails, size: targetCount * 16}
      ],
      bytes => applyVisibility(bytes)
    );
    return {key: getVisibilityKey(), compiled, reader};
  }

  function getGridKey(): string {
    const options = ctx.options;
    return [options.gridDirections, options.gridRadius, options.gridAlgorithm].join('|');
  }

  function buildGrid(): GridGraph {
    const options = ctx.options;
    const directions = Number(options.gridDirections);
    const graph = new GPUCommandGraph<void>(device, {id: 'horizon-grid'});
    const horizon = resources.createBuffer(
      `grid-horizon-${options.gridDirections}`,
      coarse.pixelCount * directions * 4
    );
    graph.add(
      new GPUTerrainHorizon({
        id: 'terrain-horizon',
        width: coarse.width,
        height: coarse.height,
        elevation: createElevationBand(
          graph,
          'grid',
          coarseElevation,
          coarseValidity,
          coarse.pixelCount
        ),
        settings: gridSettings.importToGraph(graph),
        directionCount: directions,
        maximumRadius: Number(options.gridRadius),
        algorithm: options.gridAlgorithm,
        cellSizeMode: 'web-mercator',
        rowDirection: 'south',
        horizon: importGraphBuffer(
          graph,
          'horizon',
          horizon,
          'float32',
          coarse.pixelCount * directions
        ),
        skyViewFactor: importGraphBuffer(
          graph,
          'sky-view',
          skyViewBuffer,
          'float32',
          coarse.pixelCount
        )
      })
    );
    const compiled = resources.track(graph.compile());
    return {key: getGridKey(), directions, compiled, horizon};
  }

  function ensureGraphs(): void {
    const key = getSkylineKey();
    if (!skyline || skyline.key !== key) {
      destroySkyline();
      skyline = buildSkyline();
      skylineDirty = true;
      peakMarkerCount = 0;
      angles = new Float32Array(0);
    }
    const visibilityKey = getVisibilityKey();
    if (!visibilityGraph || visibilityGraph.key !== visibilityKey) {
      if (visibilityGraph) {
        visibilityGraph.reader.stop();
        resources.release(visibilityGraph.compiled);
      }
      visibilityGraph = buildVisibility();
      visibilityDirty = true;
    }
  }

  /** The grid horizon is compiled and run only while the sky-view is the view. */
  function ensureGrid(): void {
    if (ctx.options.view !== 'sky-view') return;
    const key = getGridKey();
    if (!gridGraph || gridGraph.key !== key) {
      if (gridGraph) {
        resources.release(gridGraph.compiled);
        resources.release(gridGraph.horizon);
      }
      gridGraph = buildGrid();
      gridDirty = true;
      gridReady = false;
      gridValues = new Float32Array(0);
    }
  }

  // --- Parameters -------------------------------------------------------------------------------
  function getModelSettings() {
    const options = ctx.options;
    const common = {
      curvatureCoefficient: getCurvature(),
      maximumDistance: options.maxDistance * 1000
    };
    return options.projection === 'web-mercator'
      ? {...common, worldPixelSize: terrain.worldPixelSize, originY: terrain.originY}
      : {...common, cellSize: [terrain.groundCellSize, terrain.groundCellSize] as const};
  }

  function writeObserver(): void {
    observerRow.write(Float32Array.of(observer[0], observer[1], ctx.options.observerHeight, 0));
    observerMarker.write(Float32Array.from(getEyeMeters()));
    refreshTargetGeometry();
    skylineDirty = true;
    visibilityDirty = true;
    gridReadNeeded = true;
  }

  function writeParameters(): void {
    const options = ctx.options;
    horizonSettings.write(getGPUPointHorizonParameterValues(getModelSettings()));
    visibilitySettings.write(
      getGPUPointHorizonVisibilityParameterValues({
        ...getModelSettings(),
        toleranceDegrees: options.toleranceDegrees,
        sigmaZ: options.sigmaZ,
        skylineToleranceDegrees: options.skylineToleranceDegrees,
        targetIgnoreDistance: options.targetIgnoreDistance,
        targetIgnoreFraction: options.targetIgnoreFraction
      })
    );
    peakSettings.write(getGPUProfilePeaksParameterValues({minProminence: options.minProminence}));
    gridSettings.write(
      getGPUTerrainHorizonParameterValues({
        cellSize: coarse.mercatorCellSettings.cellSize,
        northEdge: coarse.northEdge,
        southEdge: coarse.southEdge,
        curvatureCoefficient: getCurvature()
      })
    );
    ctx.setReadout(
      'drop',
      `${(getCurvature() * (options.maxDistance * 1000) ** 2).toFixed(1)} m at ${options.maxDistance} km`
    );
    writeObserver();
  }

  // --- Selection --------------------------------------------------------------------------------
  function syncSelectionFromBearing(): void {
    const bearing = wrapDegrees(ctx.options.bearing);
    if (
      selectedIndex >= 0 &&
      isInRange(selectedIndex) &&
      getBearingDifference(targetAzimuth[selectedIndex], bearing) <= SELECT_TOLERANCE_DEGREES
    ) {
      return;
    }
    let best = -1;
    let bestDifference = SELECT_TOLERANCE_DEGREES;
    catalogue.forEach((_, index) => {
      if (!isInRange(index)) return;
      const difference = getBearingDifference(targetAzimuth[index], bearing);
      if (difference < bestDifference) {
        best = index;
        bestDifference = difference;
      }
    });
    selectedIndex = best;
  }

  function setBearing(azimuth: number): void {
    ctx.setOptions({bearing: toBearingOption(azimuth)});
  }

  // --- Readbacks --------------------------------------------------------------------------------
  function applySkyline(bytes: ArrayBuffer): void {
    const active = skyline;
    if (!active) return;
    const count = active.azimuthCount;
    angles = toFloat32(bytes, count);
    distances = toFloat32(bytes.slice(count * 4), count);
    const samples = toUint32(bytes.slice(count * 8), count);
    const prominence = toFloat32(bytes.slice(count * 12), count);
    const refinedIndex = toFloat32(bytes.slice(count * 16), count);
    const converged = toUint32(bytes.slice(count * 20), 1)[0];
    const [eyeX, eyeY] = getEyeMeters();
    hits = new Float32Array(count * 2);
    const valid = new Uint8Array(count);
    let highest = Number.NEGATIVE_INFINITY;
    let highestRay = -1;
    let totalSamples = 0;
    for (let ray = 0; ray < count; ray++) {
      const radians = (ray * 2 * Math.PI) / count;
      const ok = isValidAngle(angles[ray]) && distances[ray] > 0;
      valid[ray] = ok ? 1 : 0;
      const distance = ok ? distances[ray] : 0;
      hits[ray * 2] = eyeX + distance * Math.sin(radians);
      hits[ray * 2 + 1] = eyeY + distance * Math.cos(radians);
      totalSamples += samples[ray];
      if (ok && angles[ray] > highest) {
        highest = angles[ray];
        highestRay = ray;
      }
    }
    // The skyline ring: one segment per ray to the next ray's hit, broken where the skyline jumps
    // from a near ridge to a far one (a chord there would draw a line through the air).
    const ring = new Float32Array(count * 4);
    const weights = new Float32Array(count);
    for (let ray = 0; ray < count; ray++) {
      const next = (ray + 1) % count;
      ring.set([hits[ray * 2], hits[ray * 2 + 1], hits[next * 2], hits[next * 2 + 1]], ray * 4);
      weights[ray] =
        valid[ray] && valid[next] && !isSkylineJump(distances[ray], distances[next]) ? 1 : 0;
    }
    active.ring.write(ring);
    active.weights.write(weights);
    // The fan: every (count / 36)th ray from the eye to its hit.
    const fan = new Float32Array(FAN_CAPACITY * 4);
    for (let step = 0; step < FAN_RAY_COUNT; step++) {
      const ray = Math.round((step * count) / FAN_RAY_COUNT) % count;
      if (!valid[ray]) continue;
      fan.set([eyeX, eyeY, hits[ray * 2], hits[ray * 2 + 1]], step * 4);
    }
    active.fan.write(fan);
    highestSkyline =
      highestRay >= 0
        ? {
            angle: highest,
            azimuth: (highestRay * 360) / count,
            meters: [hits[highestRay * 2], hits[highestRay * 2 + 1]]
          }
        : null;

    // Peaks of the circular skyline.
    skylinePeaks = [];
    for (let ray = 0; ray < count; ray++) {
      if (!Number.isFinite(prominence[ray])) continue;
      const index = refinedIndex[ray];
      const azimuth = wrapDegrees((index * 360) / count);
      const nearest = ((Math.round(index) % count) + count) % count;
      const distance = distances[nearest];
      const radians = (azimuth * Math.PI) / 180;
      skylinePeaks.push({
        azimuth,
        angle: angles[nearest],
        prominence: prominence[ray],
        distance,
        meters: [eyeX + distance * Math.sin(radians), eyeY + distance * Math.cos(radians)]
      });
    }
    skylinePeaks.sort((left, right) => right.prominence - left.prominence);
    peakMarkerCount = Math.min(skylinePeaks.length, PEAK_MARKER_CAPACITY);
    active.peakMarkers.write(
      Float32Array.from(skylinePeaks.slice(0, peakMarkerCount).flatMap(peak => peak.meters))
    );
    ctx.setReadout('skylinePeakCount', skylinePeaks.length);
    ctx.setReadout(
      'peakSuppression',
      converged ? 'converged' : 'not converged (the suppression rounds ran out)'
    );
    ctx.setReadout(
      'highestAngle',
      highestSkyline
        ? `${highestSkyline.angle.toFixed(1)}° ${getCompassLabel(highestSkyline.azimuth)}`
        : null
    );
    ctx.setReadout(
      'samples',
      `${formatCount(totalSamples)} over ${count} rays (${(totalSamples / count).toFixed(0)} per ray)`
    );
    schedulePublish();
    ctx.requestLayers();
    updateGridComparison();
  }

  function applyVisibility(bytes: ArrayBuffer): void {
    visibilityCodes = toUint32(bytes, targetCount);
    details = toFloat32(bytes.slice(targetCount * 4), targetCount * 4);
    const onSkyline = new Uint32Array(targetCount);
    for (let index = 0; index < targetCount; index++) {
      onSkyline[index] = details[index * 4 + 3] > 0.5 ? 1 : 0;
    }
    targetOnSkyline.write(onSkyline);
    visibilityReady = true;
    schedulePublish();
    ctx.requestLayers();
  }

  function updateGridComparison(): void {
    // Compare the grid-wide horizon at the eye's coarse cell with the point skyline.
    if (!skyline || gridValues.length === 0 || angles.length !== skyline.azimuthCount) return;
    const count = skyline.azimuthCount;
    let sum = 0;
    let largest = 0;
    let used = 0;
    for (let sector = 0; sector < gridDirectionCount; sector++) {
      const ray = Math.round((sector / gridDirectionCount) * count) % count;
      const pointAngle = angles[ray];
      const gridAngle = gridValues[sector];
      if (!isValidAngle(pointAngle) || !Number.isFinite(gridAngle)) continue;
      const difference = Math.abs(pointAngle - gridAngle);
      sum += difference;
      largest = Math.max(largest, difference);
      used++;
    }
    ctx.setReadout(
      'gridCompare',
      used
        ? `mean ${(sum / used).toFixed(2)}°, largest ${largest.toFixed(2)}° over ${used} sectors`
        : null
    );
  }

  function requestGridHorizon(commandEncoder: CommandEncoder): void {
    if (!gridGraph || gridReadPending) return;
    const ticket = gridRing.tryAcquire();
    if (!ticket) return;
    const [column, row] = coarse.getPixel(...getEyeLngLat());
    const pixel =
      Math.min(Math.max(Math.round(row), 0), coarse.height - 1) * coarse.width +
      Math.min(Math.max(Math.round(column), 0), coarse.width - 1);
    const directions = gridGraph.directions;
    commandEncoder.copyBufferToBuffer({
      sourceBuffer: gridGraph.horizon,
      sourceOffset: pixel * directions * 4,
      destinationBuffer: ticket.buffer,
      destinationOffset: 0,
      size: directions * 4
    });
    commandEncoder.copyBufferToBuffer({
      sourceBuffer: skyViewBuffer,
      sourceOffset: pixel * 4,
      destinationBuffer: ticket.buffer,
      destinationOffset: 128,
      size: 4
    });
    ticket.markEncoded({byteOffset: 0, byteLength: 132});
    gridReadPending = true;
    gridReadNeeded = false;
    void ticket
      .read()
      .then(bytes => {
        if (destroyed) return;
        gridDirectionCount = directions;
        gridValues = new Float32Array(bytes.slice(0, directions * 4).buffer);
        const skyView = new Float32Array(bytes.slice(128, 132).buffer)[0];
        ctx.setReadout('skyViewHere', Number.isFinite(skyView) ? skyView.toFixed(2) : null);
        updateGridComparison();
        ctx.requestLayers();
      })
      .catch(() => {})
      .finally(() => {
        gridReadPending = false;
      });
  }

  // --- Measuring --------------------------------------------------------------------------------
  async function measure(): Promise<void> {
    if (!skyline) return;
    try {
      ctx.setReadout('timing', 'measuring...');
      const timing = await measureCompiledGraph(device, skyline.compiled, {
        parameters: undefined,
        completionBuffer: skyline.converged,
        signal: ctx.signal,
        runs: 5
      });
      ctx.setReadout(
        'timing',
        `${skyline.compiled.stats.nodeOrder.length} nodes · ${formatCompiledGraphTiming(timing)}`
      );
    } catch (error) {
      if (!destroyed) ctx.setReadout('timing', `failed: ${(error as Error).message}`);
    }
  }

  // --- The cast ray -----------------------------------------------------------------------------
  function updateRay(): void {
    castRay = null;
    const state = ctx.options;
    if (state.view !== 'visibility' || !skyline || angles.length !== skyline.azimuthCount) return;
    const eyeMeters = getEyeMeters();
    const eyeLngLat = getEyeLngLat();
    const refraction = getRefractionValue(state.refraction);
    if (selectedIndex >= 0 && isInRange(selectedIndex)) {
      const peak = catalogue[selectedIndex];
      const distance = targetDistance[selectedIndex];
      const ray = castSightRay(probe, {
        from: eyeLngLat,
        to: peak.lngLat,
        stepMeters: RAY_STEP_METERS,
        eyeHeightMeters: state.observerHeight,
        refraction,
        cutoffMeters:
          distance - Math.max(state.targetIgnoreDistance, state.targetIgnoreFraction * distance)
      });
      const code = visibilityReady ? visibilityCodes[selectedIndex] : VISIBILITY.visible;
      // The GPU's own angles when it has classified the peak, else the CPU's.
      const peakAngle = visibilityReady ? details[selectedIndex * 4] : ray.endAngleDegrees;
      const ridgeAngle = visibilityReady
        ? details[selectedIndex * 4 + 1]
        : (ray.ridge?.angleDegrees ?? Number.NaN);
      const ridge = ray.ridge;
      const ridgeText = ridge
        ? `${Math.round(ridge.heightMeters).toLocaleString('en-US')} m ridge ${(ridge.distanceMeters / 1000).toFixed(1)} km from the eye`
        : 'no ridge in front';
      let verdict: string;
      if (code === VISIBILITY.hidden) {
        verdict = `${peak.displayName} is hidden by a ${ridgeText}: ${ridgeAngle.toFixed(1)}° against ${peakAngle.toFixed(1)}°`;
      } else if (code === VISIBILITY.marginal) {
        verdict = `${peak.displayName} is marginal: ${peakAngle.toFixed(1)}° is within the tolerance of the ${ridgeText} (${ridgeAngle.toFixed(1)}°)`;
      } else {
        verdict = `${peak.displayName} is visible: ${peakAngle.toFixed(1)}° clears the highest ground in front (${ridgeAngle.toFixed(1)}°)`;
      }
      castRay = {ray, targetIndex: selectedIndex, verdict, ridgeAngle};
      const peakMeters = peak.meters;
      if (code === VISIBILITY.hidden && ridge) {
        const [ridgeX, ridgeY] = terrain.projection.project(ridge.lngLat[0], ridge.lngLat[1]);
        rayBefore.write(Float32Array.of(eyeMeters[0], eyeMeters[1], ridgeX, ridgeY));
        rayAfter.write(Float32Array.of(ridgeX, ridgeY, peakMeters[0], peakMeters[1]));
      } else {
        rayBefore.write(Float32Array.of(eyeMeters[0], eyeMeters[1], peakMeters[0], peakMeters[1]));
        rayAfter.write(new Float32Array(4));
      }
      ctx.setChart(
        'rayProfile',
        buildRayProfileChart({ray, targetName: peak.displayName, targetAngle: peakAngle})
      );
      ctx.setReadout('rayVerdict', verdict);
      return;
    }
    // A free ray: along the bearing to the end of the data or the ray length.
    const bearing = wrapDegrees(state.bearing);
    const end = getRayEnd(
      dem,
      eyeMeters,
      bearing,
      Math.min(getMaximumMeters(), HORIZON_MAXIMUM_DISTANCE),
      unproject
    );
    const ray = castSightRay(probe, {
      from: eyeLngLat,
      to: end.lngLat,
      stepMeters: RAY_STEP_METERS,
      eyeHeightMeters: state.observerHeight,
      refraction
    });
    const rayIndex = Math.round((bearing / 360) * skyline.azimuthCount) % skyline.azimuthCount;
    const skylineAngle = angles[rayIndex];
    const ridge = ray.ridge;
    const verdict = ridge
      ? `Towards ${formatBearing(bearing)} the skyline is ${skylineAngle.toFixed(1)}° high, set by ground ${(ridge.distanceMeters / 1000).toFixed(1)} km away`
      : `Towards ${formatBearing(bearing)} no ground rises above the eye`;
    castRay = {ray, targetIndex: -1, verdict, ridgeAngle: ridge?.angleDegrees ?? null};
    if (ridge) {
      const [ridgeX, ridgeY] = terrain.projection.project(ridge.lngLat[0], ridge.lngLat[1]);
      rayBefore.write(Float32Array.of(eyeMeters[0], eyeMeters[1], ridgeX, ridgeY));
      rayAfter.write(Float32Array.of(ridgeX, ridgeY, end.meters[0], end.meters[1]));
    } else {
      rayBefore.write(Float32Array.of(eyeMeters[0], eyeMeters[1], end.meters[0], end.meters[1]));
      rayAfter.write(new Float32Array(4));
    }
    ctx.setChart('rayProfile', buildRayProfileChart({ray, targetName: null, targetAngle: null}));
    ctx.setReadout('rayVerdict', verdict);
  }

  // --- Labels ------------------------------------------------------------------------------------
  function getLabelledIndices(view: HorizonView): number[] {
    if (view === 'sky-view') return [];
    const order = labelOrder.filter(isInRange);
    if (view === 'plain') return order.slice(0, MAXIMUM_PLAIN_LABELS);
    if (view !== 'visibility') return order.slice(0, MAXIMUM_PEAK_LABELS);
    const picks: number[] = [];
    if (selectedIndex >= 0 && isInRange(selectedIndex)) picks.push(selectedIndex);
    if (visibilityReady) {
      // Two visible and two hidden peaks beside the selected one: the contrast the step is about.
      for (const code of [VISIBILITY.visible, VISIBILITY.hidden]) {
        let taken = 0;
        for (const index of order) {
          if (taken >= 2) break;
          if (visibilityCodes[index] === code && !picks.includes(index)) {
            picks.push(index);
            taken++;
          }
        }
      }
    }
    for (const index of order) {
      if (picks.length >= MAXIMUM_PEAK_LABELS) break;
      if (!picks.includes(index)) picks.push(index);
    }
    return picks.slice(0, MAXIMUM_PEAK_LABELS);
  }

  function getEyeAnnotation(): MapAnnotation {
    const eyeMeters = getEyeMeters();
    const atHome =
      Math.hypot(eyeMeters[0] - homeMeters[0], eyeMeters[1] - homeMeters[1]) < HOME_RADIUS_METERS;
    const coordinate = getEyeLngLat();
    if (atHome) {
      return terrainLabel('gornergrat-station', {
        id: 'horizon-eye',
        coordinate,
        marker: 'none',
        rank: 'subject',
        minZoom: 0,
        priority: 100
      } as Partial<MapAnnotation>);
    }
    return {
      kind: 'point',
      id: 'horizon-eye',
      coordinate,
      text: 'The eye',
      detail: formatElevationMeters(terrain.sampleElevation(observer[0], observer[1])),
      marker: 'none',
      rank: 'subject',
      minZoom: 0,
      priority: 100
    };
  }

  function publishAnnotations(): void {
    const state = ctx.options;
    const list: MapAnnotation[] = [getEyeAnnotation()];
    const labelled = getLabelledIndices(state.view);
    labelled.forEach((index, position) => {
      const peak = catalogue[index];
      list.push({
        kind: 'landform',
        id: `horizon-peak-${peak.id}`,
        coordinate: peak.lngLat,
        text: peak.displayName,
        elevationMeters: peak.publishedElevationMeters,
        marker: state.view === 'visibility' ? 'none' : 'peak',
        tone: targetDistance[index] > FAR_PEAK_METERS ? 'muted' : 'ink',
        priority: 50 - position
      });
    });
    if ((state.view === 'rays' || state.view === 'skyline') && highestSkyline) {
      list.push({
        kind: 'note',
        id: 'horizon-steepest',
        coordinate: unproject(...highestSkyline.meters),
        title: `${highestSkyline.angle.toFixed(1)}° at ${getCompassLabel(highestSkyline.azimuth)}`,
        text: 'steepest skyline angle',
        priority: 80
      });
    }
    if (state.view === 'visibility' && castRay?.ray.ridge) {
      const {ridge} = castRay.ray;
      list.push({
        kind: 'note',
        id: 'horizon-ridge',
        coordinate: ridge.lngLat,
        title: `${castRay.ridgeAngle?.toFixed(1) ?? ridge.angleDegrees.toFixed(1)}° ridge`,
        text: `${(ridge.distanceMeters / 1000).toFixed(1)} km from the eye`,
        priority: 80
      });
    }
    if (state.view === 'rays' || state.view === 'sky-view') {
      list.push({
        kind: 'frame',
        id: 'horizon-edge',
        bounds: terrain.lngLatBounds,
        text: 'Data ends here'
      });
    }
    ctx.setAnnotations('horizon', list);
  }

  function publishHighlight(): void {
    const state = ctx.options;
    if (state.view === 'visibility') {
      if (selectedIndex >= 0 && isInRange(selectedIndex)) {
        ctx.setHighlight({
          kind: 'point',
          coordinate: catalogue[selectedIndex].lngLat,
          radiusPixels: 13
        });
      } else {
        ctx.setHighlight(null);
      }
      highlightOwned = true;
    } else if (state.view === 'skyline-peaks' && selectedSkylinePeak >= 0) {
      const peak = skylinePeaks[selectedSkylinePeak];
      if (peak) {
        ctx.setHighlight({
          kind: 'point',
          coordinate: unproject(...peak.meters),
          radiusPixels: 11,
          tone: 'signal',
          pulse: true
        });
        highlightOwned = true;
      }
    } else if (highlightOwned) {
      ctx.setHighlight(null);
      highlightOwned = false;
    }
  }

  // --- Charts, readouts and furniture -------------------------------------------------------------
  /** Name of the catalogue peak at a skyline peak, or its bearing when it is an unnamed notch. */
  function getSkylinePeakName(peak: SkylinePeak): string {
    let best = -1;
    let bestDifference = 1.2;
    catalogue.forEach((_, index) => {
      if (!isInRange(index)) return;
      const difference = getBearingDifference(targetAzimuth[index], peak.azimuth);
      const angleGap = visibilityReady ? Math.abs(details[index * 4] - peak.angle) : 0;
      if (difference < bestDifference && angleGap < 0.6) {
        best = index;
        bestDifference = difference;
      }
    });
    return best >= 0 ? catalogue[best].displayName : formatBearing(peak.azimuth);
  }

  function publishCharts(): void {
    const state = ctx.options;
    if (!skyline || angles.length !== skyline.azimuthCount) return;
    const labelled = getLabelledIndices(state.view);
    const peaks: PanoramaPeak[] = [];
    if (state.view === 'visibility' && visibilityReady) {
      catalogue.forEach((_, index) => {
        const code = visibilityCodes[index];
        const angle = details[index * 4];
        if (!isInRange(index) || !Number.isFinite(angle)) return;
        if (code === VISIBILITY.visible) {
          peaks.push({azimuth: targetAzimuth[index], angle, code: 'visible'});
        } else if (code === VISIBILITY.marginal) {
          peaks.push({azimuth: targetAzimuth[index], angle, code: 'marginal'});
        } else if (code === VISIBILITY.hidden) {
          peaks.push({azimuth: targetAzimuth[index], angle, code: 'hidden'});
        }
      });
    }
    const topSkylinePeaks = skylinePeaks.slice(0, MAXIMUM_SKYLINE_MARKERS);
    const markers =
      state.view === 'skyline-peaks'
        ? topSkylinePeaks.map(peak => ({
            azimuth: peak.azimuth,
            label: getSkylinePeakName(peak)
          }))
        : labelled
            .filter(index => index !== selectedIndex || state.view !== 'visibility')
            .map(index => ({
              azimuth: targetAzimuth[index],
              label: catalogue[index].displayName
            }));
    ctx.setChart(
      'panorama',
      buildPanoramaChart({
        view: state.view,
        azimuthCount: skyline.azimuthCount,
        skylineAngles: angles,
        peaks,
        markers,
        skylinePeaks: topSkylinePeaks,
        ...(state.view === 'visibility'
          ? {
              linkLabel: (value: number) =>
                selectedIndex >= 0 ? catalogue[selectedIndex].displayName : formatBearing(value)
            }
          : {})
      })
    );
    ctx.setChart(
      'skylinePeaks',
      skylinePeaks.length
        ? buildSkylinePeaksChart(
            skylinePeaks.slice(0, MAXIMUM_SKYLINE_ROWS).map(peak => ({
              label: getSkylinePeakName(peak),
              prominence: peak.prominence
            })),
            index => {
              selectedSkylinePeak = index;
              publishHighlight();
            }
          )
        : null
    );
  }

  function publishReadouts(): void {
    const state = ctx.options;
    const inRange = catalogue.map((_, index) => index).filter(isInRange);
    ctx.setReadout('peaksInRange', inRange.length);
    let highest = -1;
    for (const index of inRange) {
      if (
        highest < 0 ||
        catalogue[index].publishedElevationMeters > catalogue[highest].publishedElevationMeters
      ) {
        highest = index;
      }
    }
    ctx.setReadout(
      'highestPeak',
      highest >= 0
        ? `${catalogue[highest].displayName} ${formatElevationMeters(catalogue[highest].publishedElevationMeters)}`
        : null
    );
    if (visibilityReady) {
      let visible = 0;
      let marginal = 0;
      let hidden = 0;
      let onSkyline = 0;
      for (const index of inRange) {
        const code = visibilityCodes[index];
        if (code === VISIBILITY.visible) visible++;
        else if (code === VISIBILITY.marginal) marginal++;
        else if (code === VISIBILITY.hidden) hidden++;
        if (details[index * 4 + 3] > 0.5) onSkyline++;
      }
      ctx.setReadout('visibleCount', visible);
      ctx.setReadout('marginalCount', marginal);
      ctx.setReadout('hiddenCount', hidden);
      ctx.setReadout('onSkylineCount', onSkyline);
    }
    ctx.setReadout('rayCount', `${state.azimuthCount} rays`);
    ctx.setReadout('rayStep', `${(360 / Number(state.azimuthCount)).toFixed(2)}°`);
    ctx.setReadout('eyeHeight', `${state.observerHeight} m`);
    ctx.setReadout('gridDirectionsReadout', `${state.gridDirections} sectors`);
    ctx.setReadout('gridRadiusReadout', `${(getGridRadiusMeters() / 1000).toFixed(1)} km`);
  }

  const getGridRadiusMeters = (): number => Number(ctx.options.gridRadius) * coarse.groundCellSize;

  function getScaleTicks(): number[] {
    const state = ctx.options;
    if (state.view === 'sky-view') return [getGridRadiusMeters()];
    if (state.view === 'plain') return [];
    if (state.view === 'visibility') {
      return selectedIndex >= 0 ? [targetDistance[selectedIndex]] : [];
    }
    return [getMaximumMeters()];
  }

  function publishFurniture(): void {
    const state = ctx.options;
    const ticks = getScaleTicks();
    const subtitle = getHorizonSubtitle(state, {
      rayCount: Number(state.azimuthCount),
      gridRadiusKilometers: getGridRadiusMeters() / 1000
    });
    const key = `${subtitle}|${ticks.map(Math.round).join(',')}`;
    if (key === lastFurnitureKey) return;
    lastFurnitureKey = key;
    ctx.setFurniture({
      title: {subtitle, sample: demSampleLine(terrain), chips: [BARE_EARTH_CHIP]},
      scaleBar: {units: 'metric', ticks}
    });
  }

  function publishCost(): void {
    const state = ctx.options;
    if (state.view === 'sky-view') {
      ctx.setCost({
        records: coarse.pixelCount,
        note: `${state.gridDirections} sectors, ${state.gridRadius} steps each`
      });
    } else {
      ctx.setCost({
        records: Number(state.azimuthCount),
        note: `rays, and ${targetCount} peaks tested`
      });
    }
  }

  function publish(): void {
    publishTimer = null;
    if (destroyed) return;
    updateRay();
    publishReadouts();
    publishFurniture();
    publishCharts();
    publishAnnotations();
    publishHighlight();
    publishCost();
    ctx.requestLayers();
  }

  function schedulePublish(): void {
    if (publishTimer !== null || destroyed) return;
    publishTimer = setTimeout(publish, PUBLISH_DELAY_MILLISECONDS);
  }

  // --- Pointer interaction ----------------------------------------------------------------------
  function getScreenDistance(
    meters: readonly [number, number],
    event: {pixel: readonly [number, number]}
  ): number {
    const viewport = ctx.getViewport();
    if (!viewport) return Number.POSITIVE_INFINITY;
    const [longitude, latitude] = unproject(meters[0], meters[1]);
    const [x, y] = viewport.project([longitude, latitude]);
    return Math.hypot(x - event.pixel[0], y - event.pixel[1]);
  }

  /** Index of the catalogue peak under the pointer, or -1. */
  function findPeakAt(event: {pixel: readonly [number, number]}, onlyLabelled: boolean): number {
    const candidates = onlyLabelled
      ? getLabelledIndices(ctx.options.view)
      : catalogue.map((_, i) => i);
    let best = -1;
    let bestDistance = PICK_RADIUS_PIXELS;
    for (const index of candidates) {
      if (!isInRange(index)) continue;
      const distance = getScreenDistance(catalogue[index].meters, event);
      if (distance < bestDistance) {
        best = index;
        bestDistance = distance;
      }
    }
    return best;
  }

  function moveObserver(event: {coordinate: readonly [number, number] | null}): void {
    if (!event.coordinate) return;
    const [x, y] = terrain.projection.project(event.coordinate[0], event.coordinate[1]);
    observer = terrain.getPixelFromMeters(x, y);
    writeObserver();
    // A selected peak stays selected: the ray follows it as the eye moves.
    if (selectedIndex >= 0) setBearing(targetAzimuth[selectedIndex]);
    schedulePublish();
  }

  // --- Tooltips ---------------------------------------------------------------------------------
  function getPeakAngleFromEye(index: number): number {
    const distance = targetDistance[index];
    const eye = terrain.sampleElevation(observer[0], observer[1]) + ctx.options.observerHeight;
    return (
      (Math.atan2(
        catalogue[index].demElevationMeters - getCurvature() * distance ** 2 - eye,
        distance
      ) *
        180) /
      Math.PI
    );
  }

  function getPeakTooltip(index: number): TooltipContent {
    const state = ctx.options;
    const peak = catalogue[index];
    const symbols = getVisibilitySymbolColors(ctx.ground());
    const classified = state.view === 'visibility' && visibilityReady;
    const code = visibilityCodes[index];
    const peakAngle = visibilityReady ? details[index * 4] : getPeakAngleFromEye(index);
    const rows: TooltipRow[] = [];
    let note: string | undefined;
    if (classified) {
      rows.push({
        label: 'Visibility',
        value: VISIBILITY_NAMES[code] ?? `code ${code}`,
        swatch:
          code === VISIBILITY.visible
            ? symbols.visible
            : code === VISIBILITY.marginal
              ? symbols.marginal
              : symbols.hidden,
        emphasis: true
      });
    }
    rows.push({
      label: 'Angle above the horizontal',
      value: peakAngle.toFixed(1),
      unit: '°',
      emphasis: !classified
    });
    if (classified && Number.isFinite(details[index * 4 + 1])) {
      rows.push({
        label: 'Highest ground in front',
        value: details[index * 4 + 1].toFixed(1),
        unit: '°'
      });
    }
    rows.push(
      {
        label: 'Distance',
        value: `${formatDistance(targetDistance[index])} ${getCompassLabel(targetAzimuth[index])}`
      },
      {
        label: 'Elevation',
        value: `${Math.round(peak.publishedElevationMeters).toLocaleString('en-US')} m`,
        unit: `(DEM cell ${Math.round(peak.demElevationMeters).toLocaleString('en-US')} m)`
      }
    );
    if (classified && code === VISIBILITY.hidden) {
      const distance = targetDistance[index];
      const ray = castSightRay(probe, {
        from: getEyeLngLat(),
        to: peak.lngLat,
        stepMeters: RAY_STEP_METERS,
        eyeHeightMeters: state.observerHeight,
        refraction: getRefractionValue(state.refraction),
        cutoffMeters:
          distance - Math.max(state.targetIgnoreDistance, state.targetIgnoreFraction * distance)
      });
      if (ray.ridge) {
        note = `Hidden by the ridge at ${formatDistance(ray.ridge.distanceMeters)} (${details[index * 4 + 1].toFixed(1)} against ${peakAngle.toFixed(1)}°)`;
      }
    } else if (classified && details[index * 4 + 3] > 0.5) {
      note = 'On the skyline';
    }
    return {
      title: peak.displayName,
      subtitle: 'Summit',
      rows,
      ...(note ? {note} : {}),
      highlight: {kind: 'point', coordinate: peak.lngLat}
    };
  }

  function getGroundTooltip(coordinate: readonly [number, number]): TooltipContent {
    const [x, y] = terrain.projection.project(coordinate[0], coordinate[1]);
    const [column, row] = terrain.getPixelFromMeters(x, y);
    const eyeMeters = getEyeMeters();
    const dx = x - eyeMeters[0];
    const dy = y - eyeMeters[1];
    const distance = Math.hypot(dx, dy);
    const elevation = terrain.sampleElevation(column, row);
    const eyeElevation =
      terrain.sampleElevation(observer[0], observer[1]) + ctx.options.observerHeight;
    const rows: TooltipRow[] = [
      {
        label: 'Elevation',
        value: Math.round(elevation).toLocaleString('en-US'),
        unit: 'm',
        emphasis: true
      }
    ];
    if (distance > 100) {
      const angle =
        (Math.atan2(elevation - getCurvature() * distance ** 2 - eyeElevation, distance) * 180) /
        Math.PI;
      rows.push(
        {label: 'Angle from the eye', value: angle.toFixed(1), unit: '°'},
        {
          label: 'Distance',
          value: `${formatDistance(distance)} ${getCompassLabel(getAzimuthDegrees(dx, dy))}`
        }
      );
    }
    return {title: 'Ground', rows};
  }

  // --- Initialise -------------------------------------------------------------------------------
  ctx.setReadout(
    'catalogue',
    `${targetCount} named summits above ${formatCount(MINIMUM_PEAK_ELEVATION_METERS)} m, one per massif`
  );
  ctx.setReadout('peakFloor', `${formatCount(MINIMUM_PEAK_ELEVATION_METERS)} m`);
  ctx.setReadout(
    'grid',
    `${width} × ${height} cells at ${terrain.groundCellSize.toFixed(1)} m (Web Mercator ${terrain.mercatorCellSize.toFixed(1)} m); grid horizon ${coarse.width} × ${coarse.height} at ${coarse.groundCellSize.toFixed(0)} m`
  );
  writeParameters();
  syncSelectionFromBearing();

  // The setup graph (the extrema pyramid) runs once, before the first frame.
  {
    const encoder = device.createCommandEncoder({id: 'horizon-setup-encoder'});
    setupCompiled.encode(encoder, {parameters: undefined});
    device.submit(encoder.finish());
    setupDone = true;
  }
  ensureGraphs();
  ensureGrid();
  schedulePublish();

  const COMPILE_OPTIONS = new Set([
    'projection',
    'traversal',
    'azimuthCount',
    'peakWindow',
    'gridDirections',
    'gridRadius',
    'gridAlgorithm'
  ]);

  return {
    getCompiledGraphs: () => {
      const graphs = [setupCompiled];
      if (skyline) graphs.push(skyline.compiled);
      if (visibilityGraph) graphs.push(visibilityGraph.compiled);
      if (gridGraph) graphs.push(gridGraph.compiled);
      return graphs;
    },

    setOption(id) {
      if (id === 'view') {
        ensureGrid();
        gridReadNeeded = true;
        schedulePublish();
        ctx.requestLayers();
        return;
      }
      if (id === 'bearing') {
        syncSelectionFromBearing();
        schedulePublish();
        ctx.requestLayers();
        return;
      }
      if (COMPILE_OPTIONS.has(id)) {
        writeParameters();
        ensureGraphs();
        ensureGrid();
        schedulePublish();
        ctx.requestLayers();
        return;
      }
      // The grid horizon depends on the curvature only; everything else is the point contributors.
      if (id === 'refraction') gridDirty = true;
      writeParameters();
      syncSelectionFromBearing();
      schedulePublish();
    },

    onAction(id) {
      if (id === 'measure') void measure();
    },

    onGroundChange(next: TerrainGroundTone) {
      ground.setGround(next);
      tables = makeHorizonTables(next);
      ctx.setLegendData('tables', tables);
      lastFurnitureKey = '';
      ctx.requestLayers();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder) {
      if (!setupDone || !skyline || !visibilityGraph) return;
      if (ctx.options.view === 'sky-view' && gridGraph) {
        if (gridDirty) {
          gridGraph.compiled.encode(commandEncoder, {parameters: undefined});
          gridDirty = false;
          gridReady = true;
          gridReadNeeded = true;
        }
        if (gridReadNeeded) requestGridHorizon(commandEncoder);
      }
      if (skylineDirty) {
        skyline.compiled.encode(commandEncoder, {parameters: undefined});
        skyline.reader.request(commandEncoder);
        skylineDirty = false;
      }
      if (visibilityDirty) {
        visibilityGraph.compiled.encode(commandEncoder, {parameters: undefined});
        visibilityGraph.reader.request(commandEncoder);
        visibilityDirty = false;
      }
      skyline.reader.flush(commandEncoder);
      visibilityGraph.reader.flush(commandEncoder);
    },

    getLayers() {
      const state = ctx.options;
      const tone = ctx.ground();
      const ink = MAP_INK[tone];
      const inkColor = hexToRgba(ink.ink);
      const haloColor = hexToRgba(ink.halo, Math.round(ink.haloAlpha * 255));
      const eye = getObserverColors(tone);
      const symbols = getVisibilitySymbolColors(tone);
      const transparent = [0, 0, 0, 0] as const;
      const layers: Layer[] = [ground.getLayer()];
      const active = skyline;
      const ready = active !== null && angles.length === active.azimuthCount;

      if (state.view === 'sky-view' && gridReady) {
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: 'horizon-sky-view',
            coordinateOrigin: origin,
            gridSize: [coarse.width, coarse.height] as const,
            bounds: coarse.bounds,
            rowOrigin: 'north',
            values: skyViewBuffer,
            valueFormat: 'float32',
            colormap: 'grayscale',
            valueRange: SKY_VIEW_RANGE,
            rampRange: SKY_VIEW_RAMP_RANGE,
            blending: 'multiply'
          })
        );
      }
      if (ready && active) {
        const ringIds = `horizon-skyline-${active.key}`;
        if (state.view === 'visibility') {
          // Context: the same ring as one quiet ink line.
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: `${ringIds}-context`,
              coordinateOrigin: origin,
              segments: active.ring,
              instanceCount: active.azimuthCount,
              weights: active.weights,
              color: [inkColor[0], inkColor[1], inkColor[2], 120],
              widthPixels: 1.4,
              outlineColor: haloColor,
              outlineWidthPixels: 0.8
            })
          );
        } else if (state.view !== 'plain' && state.view !== 'sky-view') {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: ringIds,
              coordinateOrigin: origin,
              segments: active.ring,
              instanceCount: active.azimuthCount,
              weights: active.weights,
              values: active.angle,
              valueFormat: 'float32',
              colormap: 'grayscale',
              ...getClassTableLayerProps(tables.skyline),
              widthPixels: 2.5,
              outlineColor: haloColor,
              outlineWidthPixels: 1
            })
          );
        }
        if (state.view === 'rays') {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: `horizon-fan-${active.key}`,
              coordinateOrigin: origin,
              segments: active.fan,
              instanceCount: FAN_RAY_COUNT,
              color: [inkColor[0], inkColor[1], inkColor[2], 77],
              widthPixels: 0.8
            })
          );
        }
        if (state.view === 'skyline-peaks' && peakMarkerCount > 0) {
          layers.push(
            new SpatialAnalysisPointLayer({
              id: `horizon-skyline-ticks-${active.key}`,
              coordinateOrigin: origin,
              positions: active.peakMarkers,
              instanceCount: peakMarkerCount,
              shape: 'diamond',
              radiusPixels: 5,
              color: inkColor,
              outlineColor: haloColor,
              outlineWidthPixels: 1.5
            })
          );
        }
      }
      if (state.view === 'visibility') {
        if (castRay) {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'horizon-ray-after',
              coordinateOrigin: origin,
              segments: rayAfter,
              instanceCount: 1,
              color: [inkColor[0], inkColor[1], inkColor[2], 170],
              widthPixels: 1.6,
              dashArray: [5, 4],
              cap: 'butt'
            }),
            new SpatialAnalysisSegmentLayer({
              id: 'horizon-ray-before',
              coordinateOrigin: origin,
              segments: rayBefore,
              instanceCount: 1,
              color: inkColor,
              widthPixels: 2.2,
              outlineColor: haloColor,
              outlineWidthPixels: 1.2,
              cap: 'round'
            })
          );
        }
        if (visibilityReady && targetCount > 0) {
          const common = {
            coordinateOrigin: origin,
            positions: targetMarkers,
            instanceCount: targetCount,
            values: targetVisibility,
            valueFormat: 'uint32' as const,
            colormap: 'category' as const
          };
          // GPU_TERRAIN_VISIBILITY: hidden, visible, out of range, no data, marginal.
          const only = (code: number, color: readonly [number, number, number, number]) =>
            [0, 1, 2, 3, 4].map(entry => (entry === code ? color : transparent));
          layers.push(
            new SpatialAnalysisPointLayer({
              ...common,
              id: 'horizon-hidden-fill',
              palette: only(VISIBILITY.hidden, [haloColor[0], haloColor[1], haloColor[2], 150]),
              radiusPixels: 6.5
            }),
            new SpatialAnalysisPointLayer({
              ...common,
              id: 'horizon-hidden-ring',
              palette: only(VISIBILITY.hidden, symbols.hidden),
              shape: 'ring',
              radiusPixels: 7,
              outlineWidthPixels: 2
            }),
            new SpatialAnalysisPointLayer({
              id: 'horizon-on-skyline',
              coordinateOrigin: origin,
              positions: targetMarkers,
              instanceCount: targetCount,
              values: targetOnSkyline,
              valueFormat: 'uint32',
              colormap: 'category',
              palette: [transparent, inkColor],
              shape: 'ring',
              radiusPixels: 10.5,
              outlineWidthPixels: 1.5
            }),
            new SpatialAnalysisPointLayer({
              ...common,
              id: 'horizon-marginal-ring',
              palette: only(VISIBILITY.marginal, symbols.marginal),
              shape: 'ring',
              radiusPixels: 7,
              outlineWidthPixels: 2
            }),
            new SpatialAnalysisPointLayer({
              ...common,
              id: 'horizon-marginal-dot',
              palette: only(VISIBILITY.marginal, symbols.marginal),
              radiusPixels: 3,
              outlineColor: haloColor,
              outlineWidthPixels: 1
            }),
            new SpatialAnalysisPointLayer({
              ...common,
              id: 'horizon-visible',
              palette: only(VISIBILITY.visible, symbols.visible),
              radiusPixels: 6,
              outlineColor: haloColor,
              outlineWidthPixels: 1.5
            })
          );
        }
      }
      // The eye: a gold disc, outlined in navy, on a paper halo.
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'horizon-observer-halo',
          coordinateOrigin: origin,
          positions: observerMarker,
          instanceCount: 1,
          radiusPixels: 12.5,
          color: eye.halo,
          outlineColor: eye.halo,
          outlineWidthPixels: 1
        }),
        new SpatialAnalysisPointLayer({
          id: 'horizon-observer',
          coordinateOrigin: origin,
          positions: observerMarker,
          instanceCount: 1,
          radiusPixels: 8,
          color: eye.eye,
          outlineColor: eye.outline,
          outlineWidthPixels: 2
        })
      );
      return layers;
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      const viewport = ctx.getViewport();
      if (
        viewport &&
        getScreenDistance(getEyeMeters(), event) <= PICK_RADIUS_PIXELS &&
        ctx.options.view !== 'sky-view'
      ) {
        const eyeGround = terrain.sampleElevation(observer[0], observer[1]);
        return {
          title: 'The eye',
          subtitle: 'Drag it to move the viewpoint',
          rows: [
            {
              label: 'Eye elevation',
              value: Math.round(eyeGround + ctx.options.observerHeight).toLocaleString('en-US'),
              unit: 'm',
              emphasis: true
            },
            {label: 'Ground', value: Math.round(eyeGround).toLocaleString('en-US'), unit: 'm'},
            {label: 'Eye height', value: ctx.options.observerHeight, unit: 'm'}
          ]
        };
      }
      if (ctx.options.view !== 'sky-view' && ctx.options.view !== 'plain') {
        const hit = findPeakAt(event, ctx.options.view !== 'visibility');
        if (hit >= 0) return getPeakTooltip(hit);
      } else if (ctx.options.view === 'plain') {
        const hit = findPeakAt(event, true);
        if (hit >= 0) return getPeakTooltip(hit);
      }
      return getGroundTooltip(event.coordinate);
    },

    onClick(event) {
      if (ctx.options.view !== 'visibility' || !event.coordinate) return false;
      const hit = findPeakAt(event, false);
      if (hit >= 0) {
        selectedIndex = hit;
        setBearing(targetAzimuth[hit]);
      } else {
        const [x, y] = terrain.projection.project(event.coordinate[0], event.coordinate[1]);
        const [eyeX, eyeY] = getEyeMeters();
        selectedIndex = -1;
        setBearing(getAzimuthDegrees(x - eyeX, y - eyeY));
        syncSelectionFromBearing();
      }
      schedulePublish();
      ctx.requestLayers();
      return true;
    },

    onDragStart(event) {
      if (getScreenDistance(getEyeMeters(), event) > DRAG_RADIUS_PIXELS) return false;
      dragging = true;
      return true;
    },

    onDrag(event) {
      if (dragging) moveObserver(event);
    },

    onDragEnd(event) {
      if (!dragging) return;
      moveObserver(event);
      dragging = false;
    },

    destroy() {
      destroyed = true;
      if (publishTimer !== null) clearTimeout(publishTimer);
      skyline?.reader.stop();
      visibilityGraph?.reader.stop();
      ground.destroy();
      resources.destroy();
    }
  };
}
