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
  GPUTerrainDerivatives,
  GPUTerrainHorizon,
  GPU_POINT_HORIZON_PARAMETER_LENGTH,
  GPU_POINT_HORIZON_VISIBILITY_PARAMETER_LENGTH,
  GPU_PROFILE_PEAKS_PARAMETER_LENGTH,
  GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH,
  GPU_TERRAIN_HORIZON_PARAMETER_LENGTH,
  GPU_TERRAIN_VISIBILITY,
  getGPUPointHorizonParameterValues,
  getGPUPointHorizonVisibilityParameterValues,
  getGPUProfilePeaksParameterValues,
  getGPUTerrainCurvatureCoefficient,
  getGPUTerrainDerivativesParameterValues,
  getGPUTerrainHorizonParameterValues,
  type GPUTerrainHorizonAlgorithm
} from '@luma.gl/experimental/gpu-terrain';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisRasterLayer} from '../../engine/layers';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {formatCompiledGraphTiming, measureCompiledGraph} from '../../engine/vector-timing';
import type {SceneContext, SceneInstance} from '../scene';
import {
  ALPS_PEAK_CATALOGUE,
  createElevationBand,
  formatDistance,
  getAzimuthDegrees,
  getCompassName,
  prepareAlpsTerrain,
  toFloat32,
  toUint32,
  type AlpsTerrain
} from './b14b-terrain';

/** Option state of the horizon scene. */
export type HorizonOptions = {
  observerHeight: number;
  maxDistance: number;
  refraction: 'none' | 'mt-image' | 'gdal';
  projection: 'web-mercator' | 'planar';
  traversal: 'pyramid' | 'march';
  azimuthCount: '180' | '360' | '720' | '1440';
  sector: 'full' | 'west';
  heightReference: 'ground' | 'absolute';
  peakWindow: '8' | '16' | '32' | '64';
  minProminence: number;
  toleranceDegrees: number;
  sigmaZ: number;
  skylineToleranceDegrees: number;
  targetIgnoreDistance: number;
  targetIgnoreFraction: number;
  gridDirections: '8' | '16' | '32' | '64';
  gridRadius: '64' | '128' | '256';
  gridAlgorithm: GPUTerrainHorizonAlgorithm;
  base: 'hillshade' | 'sky-view' | 'openness';
  showSkyline: boolean;
  showPeaks: boolean;
};

/** Stride of the grid-wide `GPUTerrainHorizon` (512 x 512, 26.6 m ground cells). */
const GRID_STRIDE = 4;
/** Compile-time ray length of the skyline, longer than the diagonal of the window. */
const HORIZON_MAXIMUM_DISTANCE = 20000;
const UNNAMED_PEAK_COUNT = 14;
const PEAK_MARKER_CAPACITY = 64;
const DRAG_RADIUS_PIXELS = 20;
const REFRACTION_COEFFICIENT: Record<HorizonOptions['refraction'], number | null> = {
  none: null,
  'mt-image': 0.13,
  gdal: 1 / 7
};
const BARS = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
const VISIBILITY_NAMES: Record<number, string> = {
  [GPU_TERRAIN_VISIBILITY.hidden]: 'behind a ridge',
  [GPU_TERRAIN_VISIBILITY.visible]: 'visible',
  [GPU_TERRAIN_VISIBILITY.outOfRange]: 'out of range',
  [GPU_TERRAIN_VISIBILITY.noData]: 'no data',
  [GPU_TERRAIN_VISIBILITY.marginal]: 'marginal'
};

type Target = {
  name: string;
  catalogueElevation: number | null;
  column: number;
  row: number;
  meters: [number, number];
};

/** Everything that depends on the compile-time skyline options; rebuilt as a unit. */
type Skyline = {
  key: string;
  azimuthCount: number;
  firstAzimuth: number;
  span: number;
  compiled: CompiledGPUCommandGraph<void>;
  angle: Buffer;
  distance: Buffer;
  samples: Buffer;
  prominence: Buffer;
  refinedIndex: Buffer;
  refinedValue: Buffer;
  converged: Buffer;
  ring: Buffer;
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
};

/**
 * "Which peaks can I see from Gornergrat?" The skyline (`GPUPointHorizonProfile`) is the 360 degree
 * elevation angle of the highest ground in every direction, from the Web Mercator great-circle
 * model. `GPUProfilePeaks` finds the peaks of that circular profile, `GPUPointHorizonVisibility`
 * classifies catalogue peaks against it (visible, behind a ridge, marginal, on the skyline), and
 * `GPUTerrainHorizon` computes the same kind of horizon for every cell of a coarser grid.
 */
export async function createHorizon(
  ctx: SceneContext<HorizonOptions>
): Promise<SceneInstance<HorizonOptions>> {
  const dataset = ctx.datasets.get('alps-dem');
  const terrain = prepareAlpsTerrain(dataset, 1);
  const coarse = prepareAlpsTerrain(dataset, GRID_STRIDE);
  const {device} = ctx;
  const {width, height, pixelCount} = terrain;
  const origin: [number, number, number] = [terrain.origin[0], terrain.origin[1], 0];
  const resources = new SpatialAnalysisResources(device, 'horizon');
  let destroyed = false;

  // --- Buffers ----------------------------------------------------------------------------------
  const elevationBuffer = resources.createBuffer('elevation', terrain.elevation);
  const validityBuffer = resources.createBuffer('validity', terrain.validity);
  const hillshadeBuffer = resources.createBuffer('hillshade', pixelCount * 4);
  const coarseElevation = resources.createBuffer('coarse-elevation', coarse.elevation);
  const coarseValidity = resources.createBuffer('coarse-validity', coarse.validity);
  const skyViewBuffer = resources.createBuffer('sky-view', coarse.pixelCount * 4);
  const opennessBuffer = resources.createBuffer('openness', coarse.pixelCount * 4);
  const derivativesSettings = resources.createParameterBuffer(
    'derivatives-settings',
    'float32',
    GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH
  );
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

  // --- Hillshade at full resolution and the shared extrema pyramid (both built once) ------------
  const setupGraph = new GPUCommandGraph<void>(device, {id: 'horizon-setup'});
  const setupElevation = createElevationBand(
    setupGraph,
    'setup',
    elevationBuffer,
    validityBuffer,
    pixelCount
  );
  setupGraph.add(
    new GPUTerrainDerivatives({
      id: 'derivatives',
      width,
      height,
      elevation: setupElevation,
      settings: derivativesSettings.importToGraph(setupGraph),
      hillshade: importGraphBuffer(setupGraph, 'hillshade', hillshadeBuffer, 'float32', pixelCount),
      cellSizeMode: 'web-mercator',
      rowDirection: 'south'
    })
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
  derivativesSettings.write(
    getGPUTerrainDerivativesParameterValues({
      ...terrain.mercatorCellSettings,
      azimuthDegrees: 315,
      altitudeDegrees: 40
    })
  );

  // --- Targets: the named catalogue snapped to the DEM, plus the highest unnamed summits --------
  const targets = buildTargets(terrain);
  const targetCount = targets.length;
  const targetRows = new Float32Array(targetCount * 4);
  targets.forEach((target, index) => {
    targetRows.set([target.column, target.row, 0, 0], index * 4);
  });
  const targetBuffer = resources.createBuffer('targets', targetRows);
  const targetMarkers = resources.createBuffer(
    'target-markers',
    Float32Array.from(targets.flatMap(target => target.meters))
  );
  const targetVisibility = resources.createBuffer('target-visibility', targetCount * 4);
  const targetDetails = resources.createBuffer('target-details', targetCount * 16);
  const targetOnSkyline = resources.createBuffer('target-on-skyline', targetCount * 4);

  // --- State ------------------------------------------------------------------------------------
  const gornergrat = terrain.observers.find(place => place.name === 'Gornergrat');
  let observer: [number, number] = gornergrat
    ? terrain.getPixel(gornergrat.longitude, gornergrat.latitude)
    : [width / 2, height / 2];
  let dragging = false;
  let skylineDirty = true;
  let visibilityDirty = true;
  let skyline: Skyline | null = null;
  let visibilityGraph: VisibilityGraph | null = null;
  let gridGraph: GridGraph | null = null;
  let angles = new Float32Array(0);
  let distances = new Float32Array(0);
  let visibilityCodes = new Uint32Array(targetCount);
  let details = new Float32Array(targetCount * 4);
  let skylinePeaks: {azimuth: number; angle: number; prominence: number; distance: number}[] = [];
  let skylineRange: [number, number] = [-3, 10];
  let gridDirty = true;
  let gridReadPending = false;
  const gridRing = resources.track(
    new GPUReadbackRing(device, {id: 'horizon-grid-read', byteLength: 64 * 4})
  );
  let gridReadNeeded = true;
  let gridValues = new Float32Array(0);
  let gridDirectionCount = 16;
  let setupDone = false;

  const getCurvature = (): number => {
    const coefficient = REFRACTION_COEFFICIENT[ctx.options.refraction];
    return coefficient === null ? 0 : getGPUTerrainCurvatureCoefficient(coefficient);
  };

  function getSectorOptions(): {azimuthCount: number; firstAzimuth: number; span: number} {
    const azimuthCount = Number(ctx.options.azimuthCount);
    return ctx.options.sector === 'full'
      ? {azimuthCount, firstAzimuth: 0, span: azimuthCount}
      : {azimuthCount, firstAzimuth: azimuthCount / 2, span: azimuthCount / 2};
  }

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
    return [
      options.projection,
      options.traversal,
      options.azimuthCount,
      options.sector,
      options.heightReference,
      options.peakWindow
    ].join('|');
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
      skyline.converged,
      skyline.ring,
      skyline.peakMarkers
    ]) {
      resources.release(buffer);
    }
    skyline = null;
  }

  function buildSkyline(): Skyline {
    const options = ctx.options;
    const {azimuthCount, firstAzimuth, span} = getSectorOptions();
    const window = Number(options.peakWindow);
    const angle = resources.createBuffer('skyline-angle', span * 4);
    const distance = resources.createBuffer('skyline-distance', span * 4);
    const samples = resources.createBuffer('skyline-samples', span * 4);
    const prominence = resources.createBuffer('peak-prominence', span * 4);
    const refinedIndex = resources.createBuffer('peak-index', span * 4);
    const refinedValue = resources.createBuffer('peak-value', span * 4);
    const peakMask = resources.createBuffer('peak-mask', span * 4);
    const converged = resources.createBuffer('peak-converged', 4);
    const ring = resources.createBuffer('skyline-ring', span * 8);
    const peakMarkers = resources.createBuffer('peak-markers', PEAK_MARKER_CAPACITY * 8);
    const offsets = resources.createBuffer('skyline-offsets', Uint32Array.of(0, span));
    const graph = new GPUCommandGraph<void>(device, {id: 'horizon-skyline'});
    const angleView = importGraphBuffer(graph, 'angle', angle, 'float32', span);
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
        heightReference: options.heightReference,
        traversal: options.traversal,
        pyramid: options.traversal === 'pyramid' ? getSharedPyramid(graph) : undefined,
        azimuthCount,
        firstAzimuth,
        azimuthSpan: span,
        maximumDistance: HORIZON_MAXIMUM_DISTANCE,
        cellSize: terrain.groundCellSize,
        observers: importGraphBuffer(graph, 'observer', observerRow, 'float32x4', 1),
        settings: horizonSettings.importToGraph(graph),
        skylineAngle: angleView,
        distance: importGraphBuffer(graph, 'distance', distance, 'float32', span),
        samples: importGraphBuffer(graph, 'samples', samples, 'uint32', span)
      })
    );
    graph.add(
      new GPUProfilePeaks({
        id: 'skyline-peaks',
        values: angleView,
        offsets: importGraphBuffer(graph, 'offsets', offsets, 'uint32', 2),
        settings: peakSettings.importToGraph(graph),
        // A full circle wraps; the west half-circle is an ordinary profile.
        wrap: options.sector === 'full',
        window,
        prominence: importGraphBuffer(graph, 'prominence', prominence, 'float32', span),
        refinedIndex: importGraphBuffer(graph, 'refined-index', refinedIndex, 'float32', span),
        refinedValue: importGraphBuffer(graph, 'refined-value', refinedValue, 'float32', span),
        peakMask: importGraphBuffer(graph, 'peak-mask', peakMask, 'uint32', span),
        converged: importGraphBuffer(graph, 'converged', converged, 'uint32', 1)
      })
    );
    const compiled = resources.track(graph.compile());
    const reader = new SummaryReader(
      resources,
      'skyline',
      [
        {buffer: angle, size: span * 4},
        {buffer: distance, size: span * 4},
        {buffer: samples, size: span * 4},
        {buffer: prominence, size: span * 4},
        {buffer: refinedIndex, size: span * 4},
        {buffer: refinedValue, size: span * 4},
        {buffer: converged, size: 4}
      ],
      bytes => applySkyline(bytes)
    );
    return {
      key: getSkylineKey(),
      azimuthCount,
      firstAzimuth,
      span,
      compiled,
      angle,
      distance,
      samples,
      prominence,
      refinedIndex,
      refinedValue,
      converged,
      ring,
      peakMarkers,
      reader
    };
  }

  function getVisibilityKey(): string {
    const options = ctx.options;
    return [options.projection, options.traversal, options.heightReference].join('|');
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
        heightReference: options.heightReference,
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
    const horizonBuffer = resources.createBuffer(
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
          horizonBuffer,
          'float32',
          coarse.pixelCount * directions
        ),
        skyViewFactor: importGraphBuffer(
          graph,
          'sky-view',
          skyViewBuffer,
          'float32',
          coarse.pixelCount
        ),
        positiveOpenness: importGraphBuffer(
          graph,
          'openness',
          opennessBuffer,
          'float32',
          coarse.pixelCount
        )
      })
    );
    const compiled = resources.track(graph.compile());
    gridHorizonBuffer = horizonBuffer;
    return {key: getGridKey(), directions, compiled};
  }
  let gridHorizonBuffer: Buffer | null = null;

  // --- Parameters -------------------------------------------------------------------------------
  function getEyeHeight(): number {
    const options = ctx.options;
    return options.heightReference === 'absolute'
      ? terrain.sampleElevation(observer[0], observer[1]) + options.observerHeight
      : options.observerHeight;
  }

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
    observerRow.write(Float32Array.of(observer[0], observer[1], getEyeHeight(), 0));
    observerMarker.write(Float32Array.from(terrain.getMeters(observer[0], observer[1])));
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
      `${(getCurvature() * Math.pow(options.maxDistance * 1000, 2)).toFixed(2)} m at ${options.maxDistance} km`
    );
    writeObserver();
  }

  // --- Readbacks --------------------------------------------------------------------------------
  function applySkyline(bytes: ArrayBuffer): void {
    const active = skyline;
    if (!active) return;
    const {span} = active;
    angles = toFloat32(bytes, span);
    distances = toFloat32(bytes.slice(span * 4), span);
    const samples = toUint32(bytes.slice(span * 8), span);
    const prominence = toFloat32(bytes.slice(span * 12), span);
    const refinedIndex = toFloat32(bytes.slice(span * 16), span);
    const converged = toUint32(bytes.slice(span * 24), 1)[0];
    const observerMeters = terrain.getMeters(observer[0], observer[1]);
    const ring = new Float32Array(span * 2);
    let minimum = Infinity;
    let maximum = -Infinity;
    let highest = -Infinity;
    let highestAzimuth = 0;
    for (let ray = 0; ray < span; ray++) {
      const azimuth = ((active.firstAzimuth + ray) * 360) / active.azimuthCount;
      const radians = (azimuth * Math.PI) / 180;
      const angle = angles[ray];
      const valid = Number.isFinite(angle) && angle > -89.5 && distances[ray] > 0;
      ring[ray * 2] = observerMeters[0] + (valid ? distances[ray] : 0) * Math.sin(radians);
      ring[ray * 2 + 1] = observerMeters[1] + (valid ? distances[ray] : 0) * Math.cos(radians);
      if (valid) {
        minimum = Math.min(minimum, angle);
        maximum = Math.max(maximum, angle);
        if (angle > highest) {
          highest = angle;
          highestAzimuth = azimuth;
        }
      }
    }
    active.ring.write(ring);
    // Peaks of the (circular) skyline.
    skylinePeaks = [];
    const markers: number[] = [];
    for (let ray = 0; ray < span; ray++) {
      if (!Number.isFinite(prominence[ray])) continue;
      const index = refinedIndex[ray];
      const azimuth = ((active.firstAzimuth + index) * 360) / active.azimuthCount;
      const nearest = Math.min(Math.max(Math.round(index), 0), span - 1);
      const distance = distances[nearest];
      const angle = angles[nearest];
      skylinePeaks.push({azimuth: azimuth % 360, angle, prominence: prominence[ray], distance});
      const radians = (azimuth * Math.PI) / 180;
      markers.push(
        observerMeters[0] + distance * Math.sin(radians),
        observerMeters[1] + distance * Math.cos(radians)
      );
    }
    skylinePeaks.sort((left, right) => right.prominence - left.prominence);
    const capped = Math.min(markers.length / 2, PEAK_MARKER_CAPACITY);
    peakMarkerCount = capped;
    active.peakMarkers.write(Float32Array.from(markers.slice(0, capped * 2)));

    let totalSamples = 0;
    for (let ray = 0; ray < span; ray++) totalSamples += samples[ray];
    const nextRange: [number, number] = Number.isFinite(minimum)
      ? [Math.floor(minimum), Math.ceil(maximum)]
      : [-3, 10];
    if (nextRange[0] !== skylineRange[0] || nextRange[1] !== skylineRange[1]) {
      skylineRange = nextRange;
      ctx.setLegendExtent('skyline', skylineRange);
      ctx.requestLayers();
    }
    ctx.setReadout(
      'highest',
      Number.isFinite(highest)
        ? `${highest.toFixed(2)}° towards ${highestAzimuth.toFixed(1)}° (${getCompassName(highestAzimuth)})`
        : 'no data (observer outside the grid)'
    );
    ctx.setReadout('panorama', getPanorama(active, minimum, maximum));
    ctx.setReadout(
      'skylinePeaks',
      skylinePeaks.length
        ? `${skylinePeaks.length}${converged ? '' : ' (suppression not converged)'}: ${skylinePeaks
            .slice(0, 4)
            .map(
              peak =>
                `${peak.azimuth.toFixed(0)}° ${peak.angle >= 0 ? '+' : ''}${peak.angle.toFixed(1)}°`
            )
            .join(', ')}${skylinePeaks.length > 4 ? ', ...' : ''}`
        : 'none above the prominence limit'
    );
    ctx.setReadout(
      'samples',
      `${formatCount(totalSamples)} evaluated over ${span} rays (${(totalSamples / span).toFixed(0)} per ray)`
    );
    ctx.requestLayers();
    updateGridComparison();
  }

  let peakMarkerCount = 0;

  function getPanorama(active: Skyline, minimum: number, maximum: number): string {
    if (!Number.isFinite(minimum)) return '-';
    const bins = 36;
    const perBin = active.span / bins;
    let text = '';
    for (let bin = 0; bin < bins; bin++) {
      let best = -Infinity;
      for (let ray = Math.floor(bin * perBin); ray < Math.floor((bin + 1) * perBin); ray++) {
        const value = angles[ray];
        if (Number.isFinite(value) && value > -89.5 && value > best) best = value;
      }
      const level = Number.isFinite(best)
        ? Math.round(((best - minimum) / Math.max(maximum - minimum, 0.5)) * 7)
        : 0;
      text += BARS[Math.min(Math.max(level, 0), 7)];
    }
    const first = (active.firstAzimuth * 360) / active.azimuthCount;
    const last = first + (active.span * 360) / active.azimuthCount;
    return `${first.toFixed(0)}° ${text} ${(last % 360 === 0 ? 360 : last % 360).toFixed(0)}°`;
  }

  function applyVisibility(bytes: ArrayBuffer): void {
    visibilityCodes = toUint32(bytes, targetCount);
    details = toFloat32(bytes.slice(targetCount * 4), targetCount * 4);
    const onSkyline = new Uint32Array(targetCount);
    const groups: Record<'visible' | 'marginal' | 'hidden', string[]> = {
      visible: [],
      marginal: [],
      hidden: []
    };
    let skylineCount = 0;
    targets.forEach((target, index) => {
      const code = visibilityCodes[index];
      const flag = details[index * 4 + 3] > 0.5;
      onSkyline[index] = flag ? 1 : 0;
      if (flag) skylineCount++;
      const label = target.catalogueElevation === null ? '' : target.name;
      const bucket =
        code === GPU_TERRAIN_VISIBILITY.visible
          ? groups.visible
          : code === GPU_TERRAIN_VISIBILITY.marginal
            ? groups.marginal
            : groups.hidden;
      if (label) bucket.push(label);
    });
    targetOnSkyline.write(onSkyline);
    const unnamed = (code: number) =>
      targets.filter(
        (target, index) => target.catalogueElevation === null && visibilityCodes[index] === code
      ).length;
    const describe = (names: string[], code: number) =>
      names.length || unnamed(code)
        ? `${names.join(', ')}${unnamed(code) ? `${names.length ? ' + ' : ''}${unnamed(code)} unnamed` : ''}`
        : 'none';
    ctx.setReadout('peaksVisible', describe(groups.visible, GPU_TERRAIN_VISIBILITY.visible));
    ctx.setReadout('peaksMarginal', describe(groups.marginal, GPU_TERRAIN_VISIBILITY.marginal));
    ctx.setReadout('peaksHidden', describe(groups.hidden, GPU_TERRAIN_VISIBILITY.hidden));
    ctx.setReadout('peaksOnSkyline', `${skylineCount} of ${targetCount} peaks touch the skyline`);
    ctx.requestLayers();
  }

  function updateGridComparison(): void {
    // Compare the grid-wide horizon at the observer's coarse cell with the point skyline.
    if (!skyline || gridValues.length === 0 || angles.length === 0) return;
    const active = skyline;
    let sum = 0;
    let largest = 0;
    let used = 0;
    for (let sector = 0; sector < gridDirectionCount; sector++) {
      const azimuth = (sector * 360) / gridDirectionCount;
      const ray = Math.round((azimuth / 360) * active.azimuthCount) - active.firstAzimuth;
      const wrapped = ((ray % active.azimuthCount) + active.azimuthCount) % active.azimuthCount;
      if (wrapped >= active.span) continue;
      const pointAngle = angles[wrapped];
      const gridAngle = gridValues[sector];
      if (!Number.isFinite(pointAngle) || !Number.isFinite(gridAngle) || pointAngle < -89) continue;
      const difference = Math.abs(pointAngle - gridAngle);
      sum += difference;
      largest = Math.max(largest, difference);
      used++;
    }
    ctx.setReadout(
      'gridCompare',
      used
        ? `${used} sectors: mean ${(sum / used).toFixed(2)}°, largest ${largest.toFixed(2)}° (grid horizon vs point skyline)`
        : '-'
    );
  }

  function requestGridHorizon(commandEncoder: CommandEncoder): void {
    if (!gridHorizonBuffer || gridReadPending || !gridGraph) return;
    const ticket = gridRing.tryAcquire();
    if (!ticket) return;
    const [column, row] = coarse.getPixel(
      ...terrain.getLongitudeLatitude(observer[0], observer[1])
    );
    const pixel =
      Math.min(Math.max(Math.round(row), 0), coarse.height - 1) * coarse.width +
      Math.min(Math.max(Math.round(column), 0), coarse.width - 1);
    const directions = gridGraph.directions;
    commandEncoder.copyBufferToBuffer({
      sourceBuffer: gridHorizonBuffer,
      sourceOffset: pixel * directions * 4,
      destinationBuffer: ticket.buffer,
      destinationOffset: 0,
      size: directions * 4
    });
    ticket.markEncoded({byteOffset: 0, byteLength: directions * 4});
    gridReadPending = true;
    gridReadNeeded = false;
    void ticket
      .read()
      .then(bytes => {
        if (destroyed) return;
        gridDirectionCount = directions;
        gridValues = new Float32Array(bytes.slice(0, directions * 4).buffer);
        updateGridComparison();
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

  // --- Pointer interaction ----------------------------------------------------------------------
  function getScreenDistance(
    meters: readonly [number, number],
    event: {pixel: readonly [number, number]}
  ): number {
    const viewport = ctx.getViewport();
    if (!viewport) return Infinity;
    const [longitude, latitude] = terrain.projection.unproject(meters[0], meters[1]);
    const [x, y] = viewport.project([longitude, latitude]);
    return Math.hypot(x - event.pixel[0], y - event.pixel[1]);
  }

  function moveObserver(event: {coordinate: readonly [number, number] | null}): void {
    if (!event.coordinate) return;
    const [x, y] = terrain.projection.project(event.coordinate[0], event.coordinate[1]);
    observer = terrain.getPixelFromMeters(x, y);
    writeObserver();
  }

  // --- Initialise -------------------------------------------------------------------------------
  ctx.setReadout(
    'catalogue',
    `${targets.filter(target => target.catalogueElevation !== null).length} named peaks and ${UNNAMED_PEAK_COUNT} unnamed summits`
  );
  ctx.setReadout(
    'grid',
    `${width} × ${height} cells at ${terrain.groundCellSize.toFixed(1)} m (Web Mercator ${terrain.mercatorCellSize.toFixed(1)} m)`
  );
  writeParameters();

  function ensureGraphs(): void {
    const key = getSkylineKey();
    if (!skyline || skyline.key !== key) {
      destroySkyline();
      skyline = buildSkyline();
      skylineDirty = true;
      peakMarkerCount = 0;
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
    const gridKey = getGridKey();
    if (!gridGraph || gridGraph.key !== gridKey) {
      if (gridGraph) resources.release(gridGraph.compiled);
      gridGraph = buildGrid();
      gridDirty = true;
    }
  }

  // The setup graph (hillshade, pyramid) runs once, before the first frame.
  {
    const encoder = device.createCommandEncoder({id: 'horizon-setup-encoder'});
    setupCompiled.encode(encoder, {parameters: undefined});
    device.submit(encoder.finish());
    setupDone = true;
  }
  ensureGraphs();

  return {
    getCompiledGraphs: () => {
      const graphs = [setupCompiled];
      if (skyline) graphs.push(skyline.compiled);
      if (visibilityGraph) graphs.push(visibilityGraph.compiled);
      if (gridGraph) graphs.push(gridGraph.compiled);
      return graphs;
    },

    setOption(id) {
      const compileOptions = [
        'projection',
        'traversal',
        'azimuthCount',
        'sector',
        'heightReference',
        'peakWindow',
        'gridDirections',
        'gridRadius',
        'gridAlgorithm'
      ];
      if (compileOptions.includes(id)) {
        writeParameters();
        ensureGraphs();
        ctx.requestLayers();
        return;
      }
      if (id === 'base' || id === 'showSkyline' || id === 'showPeaks') {
        ctx.requestLayers();
        return;
      }
      // The grid horizon depends on the curvature only; everything else is the point contributors.
      if (id === 'refraction') gridDirty = true;
      writeParameters();
    },

    onAction(id) {
      if (id === 'measure') void measure();
    },

    encode(commandEncoder) {
      if (!setupDone || !skyline || !visibilityGraph || !gridGraph) return;
      if (gridDirty) {
        gridGraph.compiled.encode(commandEncoder, {parameters: undefined});
        gridDirty = false;
        gridReadNeeded = true;
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
      if (gridReadNeeded) requestGridHorizon(commandEncoder);
      skyline.reader.flush(commandEncoder);
      visibilityGraph.reader.flush(commandEncoder);
    },

    getLayers() {
      const options = ctx.options;
      const layers: Layer[] = [];
      const fine = {
        coordinateOrigin: origin,
        gridSize: [width, height] as const,
        bounds: terrain.bounds,
        rowOrigin: 'north' as const
      };
      const coarseProps = {
        coordinateOrigin: origin,
        gridSize: [coarse.width, coarse.height] as const,
        bounds: coarse.bounds,
        rowOrigin: 'north' as const
      };
      if (options.base === 'hillshade') {
        layers.push(
          new SpatialAnalysisRasterLayer({
            ...fine,
            id: 'horizon-hillshade',
            values: hillshadeBuffer,
            valueFormat: 'float32',
            colormap: 'grayscale',
            valueRange: [0, 1],
            color: [255, 255, 255, 225]
          })
        );
      } else if (options.base === 'sky-view') {
        layers.push(
          new SpatialAnalysisRasterLayer({
            ...coarseProps,
            id: 'horizon-sky-view',
            values: skyViewBuffer,
            valueFormat: 'float32',
            colormap: 'cividis',
            valueRange: [0.5, 1],
            color: [255, 255, 255, 225]
          })
        );
      } else {
        layers.push(
          new SpatialAnalysisRasterLayer({
            ...coarseProps,
            id: 'horizon-openness',
            values: opennessBuffer,
            valueFormat: 'float32',
            colormap: 'magma',
            valueRange: [60, 90],
            color: [255, 255, 255, 225]
          })
        );
      }
      if (options.showSkyline && skyline && angles.length > 0) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: `horizon-skyline-${skyline.key}`,
            coordinateOrigin: origin,
            positions: skyline.ring,
            instanceCount: skyline.span,
            values: skyline.angle,
            valueFormat: 'float32',
            colormap: 'inferno',
            valueRange: skylineRange,
            noDataColor: [0, 0, 0, 0],
            radiusPixels: 3
          })
        );
        if (peakMarkerCount > 0) {
          layers.push(
            new SpatialAnalysisPointLayer({
              id: `horizon-skyline-peaks-${skyline.key}`,
              coordinateOrigin: origin,
              positions: skyline.peakMarkers,
              instanceCount: peakMarkerCount,
              radiusPixels: 6,
              color: [255, 255, 255, 255]
            }),
            new SpatialAnalysisPointLayer({
              id: `horizon-skyline-peaks-inner-${skyline.key}`,
              coordinateOrigin: origin,
              positions: skyline.peakMarkers,
              instanceCount: peakMarkerCount,
              radiusPixels: 3.5,
              color: [20, 20, 30, 255]
            })
          );
        }
      }
      if (options.showPeaks) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'horizon-target-skyline-halo',
            coordinateOrigin: origin,
            positions: targetMarkers,
            instanceCount: targetCount,
            values: targetOnSkyline,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: [
              [0, 0, 0, 0],
              [255, 255, 255, 255]
            ],
            radiusPixels: 10.5
          }),
          new SpatialAnalysisPointLayer({
            id: 'horizon-targets',
            coordinateOrigin: origin,
            positions: targetMarkers,
            instanceCount: targetCount,
            values: targetVisibility,
            valueFormat: 'uint32',
            colormap: 'category',
            // GPU_TERRAIN_VISIBILITY: hidden, visible, outOfRange, noData, marginal.
            palette: [
              [150, 70, 90, 255],
              [40, 205, 105, 255],
              [120, 120, 120, 255],
              [120, 120, 120, 255],
              [255, 175, 45, 255]
            ],
            radiusPixels: 7
          })
        );
      }
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'horizon-observer-halo',
          coordinateOrigin: origin,
          positions: observerMarker,
          instanceCount: 1,
          radiusPixels: 11,
          color: [255, 255, 255, 235]
        }),
        new SpatialAnalysisPointLayer({
          id: 'horizon-observer',
          coordinateOrigin: origin,
          positions: observerMarker,
          instanceCount: 1,
          radiusPixels: 7,
          color: [225, 40, 60, 255]
        })
      );
      return layers;
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      if (ctx.options.showPeaks) {
        let best = -1;
        let bestDistance = 14;
        targets.forEach((target, index) => {
          const distance = getScreenDistance(target.meters, event);
          if (distance < bestDistance) {
            best = index;
            bestDistance = distance;
          }
        });
        if (best >= 0) {
          const target = targets[best];
          const dx = (target.column - observer[0]) * terrain.groundCellSize;
          const dy = (observer[1] - target.row) * terrain.groundCellSize;
          const elevation = terrain.sampleElevation(target.column, target.row);
          const code = visibilityCodes[best];
          const row = details.subarray(best * 4, best * 4 + 4);
          return (
            `${target.name} · ${elevation.toFixed(0)} m` +
            (target.catalogueElevation === null
              ? ''
              : ` (catalogue ${target.catalogueElevation} m)`) +
            ` · ${formatDistance(Math.hypot(dx, dy))} ${getCompassName(getAzimuthDegrees(dx, dy))}` +
            ` · ${VISIBILITY_NAMES[code] ?? `code ${code}`}` +
            (Number.isFinite(row[0]) ? ` · ${row[0].toFixed(1)}° above the horizon plane` : '') +
            (row[3] > 0.5 ? ' · on the skyline' : '')
          );
        }
      }
      const [x, y] = terrain.projection.project(event.coordinate[0], event.coordinate[1]);
      const [column, row] = terrain.getPixelFromMeters(x, y);
      const dx = (column - observer[0]) * terrain.groundCellSize;
      const dy = (observer[1] - row) * terrain.groundCellSize;
      return `${terrain.sampleElevation(column, row).toFixed(0)} m · ${formatDistance(Math.hypot(dx, dy))} ${getCompassName(getAzimuthDegrees(dx, dy))} of the observer`;
    },

    onClick(event) {
      if (!event.coordinate) return false;
      moveObserver(event);
      return true;
    },

    onDragStart(event) {
      if (
        getScreenDistance(terrain.getMeters(observer[0], observer[1]), event) > DRAG_RADIUS_PIXELS
      ) {
        return false;
      }
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
      skyline?.reader.stop();
      visibilityGraph?.reader.stop();
      resources.destroy();
    }
  };
}

/** Named catalogue peaks snapped to the DEM summit, plus the highest unnamed summits. */
function buildTargets(terrain: AlpsTerrain): Target[] {
  const {width, height, elevation} = terrain;
  const result: Target[] = [];
  const snap = (column: number, row: number, radius: number): [number, number] => {
    let best: [number, number] = [Math.round(column), Math.round(row)];
    let bestElevation = -Infinity;
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        if (dx * dx + dy * dy > radius * radius) continue;
        const c = Math.min(Math.max(Math.round(column) + dx, 1), width - 2);
        const r = Math.min(Math.max(Math.round(row) + dy, 1), height - 2);
        const value = elevation[r * width + c];
        if (value > bestElevation) {
          bestElevation = value;
          best = [c, r];
        }
      }
    }
    return best;
  };
  for (const peak of ALPS_PEAK_CATALOGUE) {
    const [column, row] = terrain.getPixel(peak.longitude, peak.latitude);
    const [snappedColumn, snappedRow] = snap(column, row, 24);
    result.push({
      name: peak.name,
      catalogueElevation: peak.elevationMeters,
      column: snappedColumn,
      row: snappedRow,
      meters: terrain.getMeters(snappedColumn, snappedRow)
    });
  }
  // Highest unnamed summits: maxima of a 61 pixel (about 400 m) window, away from named peaks.
  const radius = 30;
  const candidates: {column: number; row: number; value: number}[] = [];
  for (let row = radius; row < height - radius; row += 2) {
    for (let column = radius; column < width - radius; column += 2) {
      const value = elevation[row * width + column];
      if (value < 2900) continue;
      let isMaximum = true;
      for (let dy = -radius; dy <= radius && isMaximum; dy += 3) {
        for (let dx = -radius; dx <= radius; dx += 3) {
          if (elevation[(row + dy) * width + column + dx] > value) {
            isMaximum = false;
            break;
          }
        }
      }
      if (isMaximum) candidates.push({column, row, value});
    }
  }
  candidates.sort((left, right) => right.value - left.value);
  const taken = [...result];
  let added = 0;
  for (const candidate of candidates) {
    if (added >= UNNAMED_PEAK_COUNT) break;
    if (
      taken.some(
        target => Math.hypot(target.column - candidate.column, target.row - candidate.row) < 70
      )
    ) {
      continue;
    }
    const [column, row] = snap(candidate.column, candidate.row, 3);
    const entry: Target = {
      name: `Summit ${Math.round(elevation[row * width + column]).toLocaleString('en-US')} m`,
      catalogueElevation: null,
      column,
      row,
      meters: terrain.getMeters(column, row)
    };
    taken.push(entry);
    result.push(entry);
    added++;
  }
  return result;
}
