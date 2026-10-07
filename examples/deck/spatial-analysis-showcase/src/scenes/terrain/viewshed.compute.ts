// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, GPUHistogram, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPURasterExtremaPyramid,
  GPURasterProfile,
  GPU_RASTER_PROFILE_PARAMETER_LENGTH,
  getGPURasterExtremaPyramidLayout,
  getGPURasterProfileParameterValues,
  type GPURasterExtremaPyramidLayout
} from '@luma.gl/experimental/gpu-raster';
import {
  GPUTerrainCumulativeViewshed,
  GPUTerrainDerivatives,
  GPUTerrainLineOfSight,
  GPUTerrainViewshed,
  GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH,
  GPU_TERRAIN_SIGHT_LINE_PARAMETER_LENGTH,
  GPU_TERRAIN_VIEWSHED_PARAMETER_LENGTH,
  GPU_TERRAIN_VISIBILITY,
  GPU_TERRAIN_VISIBILITY_TOLERANCE_PARAMETER_LENGTH,
  getGPUTerrainCurvatureCoefficient,
  getGPUTerrainDerivativesParameterValues,
  getGPUTerrainSightLineParameterValues,
  getGPUTerrainViewshedParameterValues,
  getGPUTerrainVisibilityToleranceParameterValues,
  type GPUTerrainViewshedProps
} from '@luma.gl/experimental/gpu-terrain';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer
} from '../../engine/layers';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {
  formatCompiledGraphTiming,
  formatSpeedup,
  measureCompiledGraph
} from '../../engine/vector-timing';
import type {SceneContext, SceneInstance} from '../scene';
import {
  createElevationBand,
  formatDistance,
  getAzimuthDegrees,
  getCompassName,
  prepareAlpsTerrain,
  toFloat32,
  toUint32,
  ZERMATT_VIEWPOINTS
} from './b14b-terrain';

/** Option state of the viewshed scene. */
export type ViewshedOptions = {
  observerHeight: number;
  targetHeight: number;
  maxDistance: number;
  refraction: 'none' | 'mt-image' | 'gdal';
  toleranceMeters: number;
  tolerancePerKilometer: number;
  targetIgnoreDistance: number;
  targetIgnoreFraction: number;
  traversal: 'march' | 'pyramid';
  pyramidBlockSize: '4' | '8' | '16';
  display: 'viewshed' | 'cumulative' | 'pyramid';
  cumulativeObservers: number;
  cumulativeMetric: 'visible' | 'marginal';
  pyramidLevel: number;
  base: 'hillshade' | 'elevation';
  showProfile: boolean;
  profileSpacing: number;
  profileMethod: 'nearest' | 'bilinear' | 'bicubic';
};

/**
 * Analysis grid stride. The viewshed costs O(pixels x range), so the 2048 x 2048 DEM is averaged
 * to 1024 x 1024 (13.3 m ground cells) to keep an observer drag interactive.
 */
const ANALYSIS_STRIDE = 2;
const CUMULATIVE_SLOTS = 6;
const PARKED_OBSERVER = -1000;
const PROFILE_CAPACITY = 8192;
const VISIBILITY_CODE_COUNT = 5;
const DRAG_RADIUS_PIXELS = 20;
const VISIBILITY_NAMES: Record<number, string> = {
  [GPU_TERRAIN_VISIBILITY.hidden]: 'hidden',
  [GPU_TERRAIN_VISIBILITY.visible]: 'visible',
  [GPU_TERRAIN_VISIBILITY.outOfRange]: 'out of range',
  [GPU_TERRAIN_VISIBILITY.noData]: 'no data',
  [GPU_TERRAIN_VISIBILITY.marginal]: 'marginal'
};
const REFRACTION_COEFFICIENT: Record<ViewshedOptions['refraction'], number | null> = {
  none: null,
  'mt-image': 0.13,
  gdal: 1 / 7
};

type Traversal = ViewshedOptions['traversal'];

/** Outputs of one traversal; the two are compared bit for bit by the verify button. */
type TraversalOutputs = {
  visibility: Buffer;
  counts: Buffer;
  losCode: Buffer;
  losClearance: Buffer;
  cumulativeVisible: Buffer;
  cumulativeMarginal: Buffer;
};

type PyramidState = {
  blockSize: number;
  layout: GPURasterExtremaPyramidLayout;
  combined: Buffer;
  levelBuffers: Buffer[];
};

/**
 * Viewshed, line of sight, cumulative viewshed and profile over the Matterhorn DEM.
 *
 * Both traversals produce bit-identical output, so each is compiled lazily into its own graph with
 * its own output buffers: choosing one changes which graph is encoded and which buffer is drawn, and
 * the verify button runs both and compares every word on the GPU output. The min-max pyramid is
 * built once by `GPURasterExtremaPyramid` and shared by every `'pyramid'` consumer through their
 * `pyramid` prop.
 */
export async function createViewshed(
  ctx: SceneContext<ViewshedOptions>
): Promise<SceneInstance<ViewshedOptions>> {
  const dataset = ctx.datasets.get('alps-dem');
  const terrain = prepareAlpsTerrain(dataset, ANALYSIS_STRIDE);
  const {device} = ctx;
  const {width, height, pixelCount} = terrain;
  const origin: [number, number, number] = [terrain.origin[0], terrain.origin[1], 0];
  const resources = new SpatialAnalysisResources(device, 'viewshed');
  let destroyed = false;

  // --- Terrain buffers and parameter buffers ----------------------------------------------------
  const elevationBuffer = resources.createBuffer('elevation', terrain.elevation);
  const validityBuffer = resources.createBuffer('validity', terrain.validity);
  const hillshadeBuffer = resources.createBuffer('hillshade', pixelCount * 4);
  const derivativesSettings = resources.createParameterBuffer(
    'derivatives-settings',
    'float32',
    GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH
  );
  const viewshedSettings = resources.createParameterBuffer(
    'viewshed-settings',
    'float32',
    GPU_TERRAIN_VIEWSHED_PARAMETER_LENGTH
  );
  const toleranceSettings = resources.createParameterBuffer(
    'tolerance-settings',
    'float32',
    GPU_TERRAIN_VISIBILITY_TOLERANCE_PARAMETER_LENGTH
  );
  const sightSettings = resources.createParameterBuffer(
    'sight-settings',
    'float32',
    GPU_TERRAIN_SIGHT_LINE_PARAMETER_LENGTH
  );
  const profileSettings = resources.createParameterBuffer(
    'profile-settings',
    'float32',
    GPU_RASTER_PROFILE_PARAMETER_LENGTH
  );
  const sightPair = resources.createBuffer('sight-pair', new Float32Array(4));
  const cumulativeObservers = resources.createBuffer(
    'cumulative-observers',
    new Float32Array(CUMULATIVE_SLOTS * 2)
  );
  const pathPositions = resources.createBuffer('path-positions', new Float32Array(4));
  const pathOffsets = resources.createBuffer('path-offsets', Uint32Array.of(0, 2));

  // Marker and overlay buffers in layer meters.
  const observerMarker = resources.createBuffer('observer-marker', new Float32Array(2));
  const targetMarker = resources.createBuffer('target-marker', new Float32Array(2));
  const extraMarkers = resources.createBuffer(
    'extra-markers',
    new Float32Array((CUMULATIVE_SLOTS - 1) * 2)
  );
  const sightSegment = resources.createBuffer('sight-segment', new Float32Array(4));
  const profilePoints = resources.createBuffer('profile-points', PROFILE_CAPACITY * 8);
  const profileExcess = resources.createBuffer('profile-excess', PROFILE_CAPACITY * 4);

  // --- Per-traversal outputs --------------------------------------------------------------------
  const createOutputs = (name: string): TraversalOutputs => ({
    visibility: resources.createBuffer(`${name}-visibility`, pixelCount * 4),
    counts: resources.createBuffer(`${name}-counts`, VISIBILITY_CODE_COUNT * 4),
    losCode: resources.createBuffer(`${name}-los-code`, 4),
    losClearance: resources.createBuffer(`${name}-los-clearance`, 4),
    cumulativeVisible: resources.createBuffer(`${name}-cumulative-visible`, pixelCount * 4),
    cumulativeMarginal: resources.createBuffer(`${name}-cumulative-marginal`, pixelCount * 4)
  });
  const outputs: Record<Traversal, TraversalOutputs> = {
    march: createOutputs('march'),
    pyramid: createOutputs('pyramid')
  };

  // --- Hillshade (once) -------------------------------------------------------------------------
  const baseGraph = new GPUCommandGraph<void>(device, {id: 'viewshed-base'});
  baseGraph.add(
    new GPUTerrainDerivatives({
      id: 'derivatives',
      width,
      height,
      elevation: createElevationBand(
        baseGraph,
        'base',
        elevationBuffer,
        validityBuffer,
        pixelCount
      ),
      settings: derivativesSettings.importToGraph(baseGraph),
      hillshade: importGraphBuffer(baseGraph, 'hillshade', hillshadeBuffer, 'float32', pixelCount),
      cellSizeMode: 'web-mercator',
      rowDirection: 'south'
    })
  );
  const baseCompiled = resources.track(baseGraph.compile());
  derivativesSettings.write(
    getGPUTerrainDerivativesParameterValues({
      ...terrain.mercatorCellSettings,
      azimuthDegrees: 315,
      altitudeDegrees: 40
    })
  );

  // --- Extrema pyramid (built on first use, rebuilt when its block size changes) ---------------
  let pyramid: PyramidState | null = null;
  let pyramidCompiled: CompiledGPUCommandGraph<void> | null = null;
  /** Compiled traversal graphs by `main-<traversal>` or `cumulative-<traversal>`. */
  const graphCache = new Map<string, CompiledGPUCommandGraph<void>>();

  function getPyramid(): PyramidState {
    const blockSize = Number(ctx.options.pyramidBlockSize);
    if (pyramid && pyramid.blockSize === blockSize) return pyramid;
    if (pyramid) {
      for (const key of ['main-pyramid', 'cumulative-pyramid']) {
        const compiled = graphCache.get(key);
        if (compiled) resources.release(compiled);
        graphCache.delete(key);
      }
      if (pyramidCompiled) resources.release(pyramidCompiled);
      for (const buffer of [pyramid.combined, ...pyramid.levelBuffers]) resources.release(buffer);
    }
    const layout = getGPURasterExtremaPyramidLayout(width, height, {
      firstBlockSize: blockSize,
      footprint: 'bilinear'
    });
    const combined = resources.createBuffer(`pyramid-${blockSize}`, layout.length * 8);
    const levelBuffers = layout.levels.map(level =>
      resources.createBuffer(
        `pyramid-${blockSize}-level-${level.level}`,
        level.width * level.height * 4
      )
    );
    const graph = new GPUCommandGraph<void>(device, {id: `viewshed-pyramid-${blockSize}`});
    graph.add(
      new GPURasterExtremaPyramid({
        id: 'extrema-pyramid',
        width,
        height,
        input: createElevationBand(graph, 'pyramid', elevationBuffer, validityBuffer, pixelCount),
        firstBlockSize: blockSize,
        footprint: 'bilinear',
        combined: importGraphBuffer(graph, 'combined', combined, 'float32', 2 * layout.length)
      })
    );
    pyramidCompiled = resources.track(graph.compile());
    pyramid = {blockSize, layout, combined, levelBuffers};
    // Build the pyramid once and copy each maximum level out for display.
    const encoder = device.createCommandEncoder({id: 'viewshed-pyramid-encoder'});
    pyramidCompiled.encode(encoder, {parameters: undefined});
    layout.levels.forEach((level, index) => {
      encoder.copyBufferToBuffer({
        sourceBuffer: combined,
        sourceOffset: level.offset * 4,
        destinationBuffer: levelBuffers[index],
        size: level.width * level.height * 4
      });
    });
    device.submit(encoder.finish());
    void readPyramidRoot(pyramid);
    return pyramid;
  }

  async function readPyramidRoot(state: PyramidState): Promise<void> {
    try {
      const root = state.layout.levels[state.layout.levels.length - 1];
      const [maximumBytes, minimumBytes] = await Promise.all([
        state.combined.readAsync(root.offset * 4, 4),
        state.combined.readAsync((state.layout.length + root.offset) * 4, 4)
      ]);
      if (destroyed || pyramid !== state) return;
      const maximum = toFloat32(maximumBytes, 1)[0];
      const minimum = toFloat32(minimumBytes, 1)[0];
      ctx.setReadout(
        'pyramid',
        `${state.layout.levels.length} levels, ${formatCount(state.layout.length)} cells, root ${minimum.toFixed(0)} to ${maximum.toFixed(0)} m`
      );
    } catch {
      // Destroyed while reading.
    }
  }

  // --- Lazily compiled traversal graphs ---------------------------------------------------------
  const sharedPyramidViews = new WeakMap<
    GPUCommandGraph<void>,
    GPUTerrainViewshedProps['pyramid']
  >();
  function getSharedPyramid(graph: GPUCommandGraph<void>): GPUTerrainViewshedProps['pyramid'] {
    // One import per graph: every consumer of the graph reads the same pyramid view.
    const known = sharedPyramidViews.get(graph);
    if (known) return known;
    const state = getPyramid();
    const view = {
      layout: state.layout,
      combined: importGraphBuffer(
        graph,
        'pyramid-combined',
        state.combined,
        'float32',
        2 * state.layout.length
      )
    };
    sharedPyramidViews.set(graph, view);
    return view;
  }

  function getMainGraph(traversal: Traversal): CompiledGPUCommandGraph<void> {
    const cached = graphCache.get(`main-${traversal}`);
    if (cached) return cached;
    const output = outputs[traversal];
    const graph = new GPUCommandGraph<void>(device, {id: `viewshed-${traversal}`});
    const elevation = createElevationBand(
      graph,
      'main',
      elevationBuffer,
      validityBuffer,
      pixelCount
    );
    const shared = traversal === 'pyramid' ? getSharedPyramid(graph) : undefined;
    const visibility = importGraphBuffer(
      graph,
      'visibility',
      output.visibility,
      'uint32',
      pixelCount
    );
    graph.add(
      new GPUTerrainViewshed({
        id: 'viewshed',
        width,
        height,
        elevation,
        traversal,
        pyramid: shared,
        settings: viewshedSettings.importToGraph(graph),
        tolerance: toleranceSettings.importToGraph(graph),
        visibility
      })
    );
    graph.add(
      new GPUTerrainLineOfSight({
        id: 'line-of-sight',
        width,
        height,
        elevation,
        traversal,
        pyramid: traversal === 'pyramid' ? getSharedPyramid(graph) : undefined,
        pairs: importGraphBuffer(graph, 'sight-pair', sightPair, 'float32x4', 1),
        settings: sightSettings.importToGraph(graph),
        visibility: importGraphBuffer(graph, 'los-code', output.losCode, 'uint32', 1),
        clearance: importGraphBuffer(graph, 'los-clearance', output.losClearance, 'float32', 1)
      })
    );
    graph.add(
      new GPUHistogram({
        id: 'visibility-counts',
        input: visibility,
        output: importGraphBuffer(graph, 'counts', output.counts, 'uint32', VISIBILITY_CODE_COUNT),
        edges: [0, 1, 2, 3, 4, 5]
      })
    );
    const compiled = resources.track(graph.compile());
    graphCache.set(`main-${traversal}`, compiled);
    return compiled;
  }

  function getCumulativeGraph(traversal: Traversal): CompiledGPUCommandGraph<void> {
    const key = `cumulative-${traversal}`;
    const cached = graphCache.get(key);
    if (cached) return cached;
    const output = outputs[traversal];
    const graph = new GPUCommandGraph<void>(device, {id: `viewshed-${key}`});
    graph.add(
      new GPUTerrainCumulativeViewshed({
        id: 'cumulative',
        width,
        height,
        elevation: createElevationBand(
          graph,
          'cumulative',
          elevationBuffer,
          validityBuffer,
          pixelCount
        ),
        traversal,
        pyramid: traversal === 'pyramid' ? getSharedPyramid(graph) : undefined,
        observers: importGraphBuffer(
          graph,
          'observers',
          cumulativeObservers,
          'float32x2',
          CUMULATIVE_SLOTS
        ),
        settings: sightSettings.importToGraph(graph),
        visibleCount: importGraphBuffer(
          graph,
          'visible-count',
          output.cumulativeVisible,
          'uint32',
          pixelCount
        ),
        marginalCount: importGraphBuffer(
          graph,
          'marginal-count',
          output.cumulativeMarginal,
          'uint32',
          pixelCount
        )
      })
    );
    const compiled = resources.track(graph.compile());
    graphCache.set(key, compiled);
    return compiled;
  }

  // --- Elevation profile along the sight line (GPURasterProfile) -------------------------------
  const profileValues = resources.createBuffer('profile-values', PROFILE_CAPACITY * 4);
  const profileCount = resources.createBuffer('profile-count', 4);
  const profileOverflow = resources.createBuffer('profile-overflow', 4);
  const profileLength = resources.createBuffer('profile-length', 4);
  const profileGain = resources.createBuffer('profile-gain', 4);
  const profileLoss = resources.createBuffer('profile-loss', 4);
  const profileMinimum = resources.createBuffer('profile-minimum', 4);
  const profileGraph = new GPUCommandGraph<void>(device, {id: 'viewshed-profile'});
  profileGraph.add(
    new GPURasterProfile({
      id: 'profile',
      width,
      height,
      values: importGraphBuffer(profileGraph, 'dem', elevationBuffer, 'float32', pixelCount),
      validity: importGraphBuffer(
        profileGraph,
        'dem-validity',
        validityBuffer,
        'uint32',
        pixelCount
      ),
      pathPositions: importGraphBuffer(
        profileGraph,
        'path-positions',
        pathPositions,
        'float32x2',
        2
      ),
      pathOffsets: importGraphBuffer(profileGraph, 'path-offsets', pathOffsets, 'uint32', 2),
      sampleCapacity: PROFILE_CAPACITY,
      parameters: profileSettings.importToGraph(profileGraph),
      output: {
        count: importGraphBuffer(profileGraph, 'count', profileCount, 'uint32', 1),
        overflow: importGraphBuffer(profileGraph, 'overflow', profileOverflow, 'uint32', 1),
        sampleValues: importGraphBuffer(
          profileGraph,
          'values',
          profileValues,
          'float32',
          PROFILE_CAPACITY
        ),
        pathLength: importGraphBuffer(profileGraph, 'length', profileLength, 'float32', 1),
        pathGain: importGraphBuffer(profileGraph, 'gain', profileGain, 'float32', 1),
        pathLoss: importGraphBuffer(profileGraph, 'loss', profileLoss, 'float32', 1),
        pathMinimum: importGraphBuffer(profileGraph, 'minimum', profileMinimum, 'float32', 1)
      }
    })
  );
  const profileCompiled = resources.track(profileGraph.compile());

  // --- State ------------------------------------------------------------------------------------
  const viewpoints = ZERMATT_VIEWPOINTS.map(place =>
    terrain.getPixel(place.longitude, place.latitude)
  );
  const gornergrat = terrain.observers.find(place => place.name === 'Gornergrat');
  let observer: [number, number] = gornergrat
    ? terrain.getPixel(gornergrat.longitude, gornergrat.latitude)
    : viewpoints[0];
  const extraObservers: [number, number][] = [1, 2, 3, 4, 5].map(index => viewpoints[index]);
  let target: [number, number] = snapToSummit(
    terrain.getPixel(ZERMATT_VIEWPOINTS[5].longitude, ZERMATT_VIEWPOINTS[5].latitude)
  );
  let dragging: 'observer' | 'target' | null = null;
  let baseDirty = true;
  let mainDirty = true;
  let cumulativeDirty = true;
  let profileDirty = true;
  let losCode = -1;
  let profileCountValue = 0;
  let verifying = false;

  function snapToSummit(point: [number, number]): [number, number] {
    let best: [number, number] = [Math.round(point[0]), Math.round(point[1])];
    let bestElevation = -Infinity;
    for (let dy = -4; dy <= 4; dy++) {
      for (let dx = -4; dx <= 4; dx++) {
        const column = Math.min(Math.max(Math.round(point[0]) + dx, 0), width - 1);
        const row = Math.min(Math.max(Math.round(point[1]) + dy, 0), height - 1);
        const value = terrain.elevation[row * width + column];
        if (value > bestElevation) {
          bestElevation = value;
          best = [column, row];
        }
      }
    }
    return best;
  }

  const getCurvature = (): number => {
    const coefficient = REFRACTION_COEFFICIENT[ctx.options.refraction];
    return coefficient === null ? 0 : getGPUTerrainCurvatureCoefficient(coefficient);
  };

  const getCellSize = (): [number, number] => [terrain.groundCellSize, terrain.groundCellSize];

  function writeParameters(): void {
    const options = ctx.options;
    const curvatureCoefficient = getCurvature();
    const maxDistance = options.maxDistance * 1000;
    viewshedSettings.write(
      getGPUTerrainViewshedParameterValues({
        observer,
        observerHeight: options.observerHeight,
        targetHeight: options.targetHeight,
        maxDistance,
        cellSize: getCellSize(),
        curvatureCoefficient
      })
    );
    toleranceSettings.write(
      getGPUTerrainVisibilityToleranceParameterValues({
        toleranceMeters: options.toleranceMeters,
        tolerancePerKilometer: options.tolerancePerKilometer,
        targetIgnoreDistance: options.targetIgnoreDistance,
        targetIgnoreFraction: options.targetIgnoreFraction
      })
    );
    sightSettings.write(
      getGPUTerrainSightLineParameterValues({
        observerHeight: options.observerHeight,
        targetHeight: options.targetHeight,
        maxDistance,
        cellSize: getCellSize(),
        curvatureCoefficient,
        toleranceMeters: options.toleranceMeters,
        tolerancePerKilometer: options.tolerancePerKilometer,
        targetIgnoreDistance: options.targetIgnoreDistance,
        targetIgnoreFraction: options.targetIgnoreFraction
      })
    );
    sightPair.write(Float32Array.of(observer[0], observer[1], target[0], target[1]));
    const slots = Math.min(Math.max(Math.round(options.cumulativeObservers), 1), CUMULATIVE_SLOTS);
    const observerRows = new Float32Array(CUMULATIVE_SLOTS * 2).fill(PARKED_OBSERVER);
    observerRows[0] = observer[0];
    observerRows[1] = observer[1];
    for (let slot = 1; slot < slots; slot++) {
      observerRows[slot * 2] = extraObservers[slot - 1][0];
      observerRows[slot * 2 + 1] = extraObservers[slot - 1][1];
    }
    cumulativeObservers.write(observerRows);
    const observerMeters = terrain.getMeters(observer[0], observer[1]);
    const targetMeters = terrain.getMeters(target[0], target[1]);
    observerMarker.write(Float32Array.of(...observerMeters));
    targetMarker.write(Float32Array.of(...targetMeters));
    sightSegment.write(Float32Array.of(...observerMeters, ...targetMeters));
    extraMarkers.write(
      Float32Array.from(
        extraObservers
          .slice(0, slots - 1)
          .concat(
            Array.from(
              {length: CUMULATIVE_SLOTS - slots},
              () => [PARKED_OBSERVER, PARKED_OBSERVER] as [number, number]
            )
          )
          .flatMap(([column, row]) =>
            column === PARKED_OBSERVER ? [1e7, 1e7] : terrain.getMeters(column, row)
          )
      )
    );
    // Profile path in pixel-center coordinates of an extent [0, 0, width, height].
    pathPositions.write(
      Float32Array.of(observer[0] + 0.5, observer[1] + 0.5, target[0] + 0.5, target[1] + 0.5)
    );
    profileSettings.write(
      getGPURasterProfileParameterValues({
        width,
        height,
        extent: [0, 0, width, height],
        method: options.profileMethod,
        spacing: Math.max(options.profileSpacing / terrain.groundCellSize, 0.25)
      })
    );
    ctx.setReadout(
      'drop',
      `${(getCurvature() * maxDistance * maxDistance).toFixed(2)} m at ${options.maxDistance} km`
    );
    mainDirty = true;
    cumulativeDirty = true;
    profileDirty = true;
    verifiedLabelStale();
  }

  function verifiedLabelStale(): void {
    ctx.setReadout('identical', 'press Verify');
  }

  ctx.setReadout(
    'grid',
    `${width} × ${height} cells · ${terrain.groundCellSize.toFixed(1)} m ground (${ANALYSIS_STRIDE}× box average)`
  );
  writeParameters();

  // --- Readers ----------------------------------------------------------------------------------
  const readers: Partial<Record<Traversal, SummaryReader>> = {};
  const getReader = (traversal: Traversal): SummaryReader => {
    let reader = readers[traversal];
    if (!reader) {
      const output = outputs[traversal];
      reader = new SummaryReader(
        resources,
        `viewshed-${traversal}`,
        [
          {buffer: output.counts, size: VISIBILITY_CODE_COUNT * 4},
          {buffer: output.losCode, size: 4},
          {buffer: output.losClearance, size: 4}
        ],
        bytes => applySummary(traversal, bytes)
      );
      readers[traversal] = reader;
    }
    return reader;
  };

  const profileReader = new SummaryReader(
    resources,
    'viewshed-profile',
    [
      {buffer: profileCount, size: 4},
      {buffer: profileOverflow, size: 4},
      {buffer: profileLength, size: 4},
      {buffer: profileGain, size: 4},
      {buffer: profileLoss, size: 4},
      {buffer: profileMinimum, size: 4},
      {buffer: profileValues, size: PROFILE_CAPACITY * 4}
    ],
    bytes => applyProfile(bytes)
  );

  function applySummary(traversal: Traversal, bytes: ArrayBuffer): void {
    if (traversal !== ctx.options.traversal) return;
    const counts = toUint32(bytes, VISIBILITY_CODE_COUNT);
    const code = toUint32(bytes.slice(VISIBILITY_CODE_COUNT * 4), 1)[0];
    const clearance = toFloat32(bytes.slice((VISIBILITY_CODE_COUNT + 1) * 4), 1)[0];
    const hidden = counts[GPU_TERRAIN_VISIBILITY.hidden];
    const visible = counts[GPU_TERRAIN_VISIBILITY.visible];
    const marginal = counts[GPU_TERRAIN_VISIBILITY.marginal];
    const outOfRange = counts[GPU_TERRAIN_VISIBILITY.outOfRange];
    const inRange = hidden + visible + marginal;
    const cellArea = terrain.groundCellSize * terrain.groundCellSize;
    ctx.setReadout(
      'visible',
      inRange > 0
        ? `${((100 * visible) / inRange).toFixed(1)}% · ${((visible * cellArea) / 1e6).toFixed(2)} km²`
        : 'no valid observer'
    );
    ctx.setReadout(
      'marginal',
      inRange > 0
        ? `${((100 * marginal) / inRange).toFixed(1)}% · ${((marginal * cellArea) / 1e6).toFixed(2)} km²`
        : '-'
    );
    ctx.setReadout(
      'hidden',
      inRange > 0 ? `${((100 * hidden) / inRange).toFixed(1)}% · ${formatCount(hidden)} cells` : '-'
    );
    ctx.setReadout(
      'range',
      `${formatCount(inRange)} cells in range, ${formatCount(outOfRange)} beyond`
    );
    const distance = Math.hypot(
      (target[0] - observer[0]) * terrain.groundCellSize,
      (target[1] - observer[1]) * terrain.groundCellSize
    );
    const showClearance =
      code === GPU_TERRAIN_VISIBILITY.visible ||
      code === GPU_TERRAIN_VISIBILITY.hidden ||
      code === GPU_TERRAIN_VISIBILITY.marginal;
    ctx.setReadout(
      'lineOfSight',
      `${VISIBILITY_NAMES[code] ?? `code ${code}`} · ${formatDistance(distance)}` +
        (showClearance && Number.isFinite(clearance) && Math.abs(clearance) < 1e30
          ? ` · clearance ${clearance >= 0 ? '+' : ''}${clearance.toFixed(1)} m`
          : '')
    );
    if (code !== losCode) {
      losCode = code;
      ctx.requestLayers();
    }
  }

  function applyProfile(bytes: ArrayBuffer): void {
    const words = toUint32(bytes, 2);
    const stats = toFloat32(bytes.slice(8), 4);
    const count = Math.min(words[0], PROFILE_CAPACITY);
    const values = toFloat32(bytes.slice(24), PROFILE_CAPACITY);
    ctx.setReadout(
      'profile',
      `${formatCount(count)} samples · gain ${stats[1].toFixed(0)} m · loss ${stats[2].toFixed(0)} m · lowest ${stats[3].toFixed(0)} m` +
        (words[1] ? ' (capacity exceeded)' : '')
    );
    // Terrain above (+) or below (-) the straight sight line, drawn along the path.
    const options = ctx.options;
    const eye = terrain.sampleElevation(observer[0], observer[1]) + options.observerHeight;
    const goal = terrain.sampleElevation(target[0], target[1]) + options.targetHeight;
    const pixelLength = stats[0];
    const distanceMeters = pixelLength * terrain.groundCellSize;
    const curvature = getCurvature();
    const positions = new Float32Array(count * 2);
    const excess = new Float32Array(count);
    for (let index = 0; index < count; index++) {
      const fraction = count > 1 ? Math.min((index * stepPixels()) / pixelLength, 1) : 0;
      const along = fraction * distanceMeters;
      const column = observer[0] + (target[0] - observer[0]) * fraction;
      const row = observer[1] + (target[1] - observer[1]) * fraction;
      const [x, y] = terrain.getMeters(column, row);
      positions[index * 2] = x;
      positions[index * 2 + 1] = y;
      const line = eye + (goal - curvature * distanceMeters * distanceMeters - eye) * fraction;
      excess[index] = values[index] - curvature * along * along - line;
    }
    profilePoints.write(positions);
    profileExcess.write(excess);
    if (count !== profileCountValue) {
      profileCountValue = count;
    }
    ctx.requestLayers();
  }

  const stepPixels = (): number =>
    Math.max(ctx.options.profileSpacing / terrain.groundCellSize, 0.25);

  // --- Verification and timing ------------------------------------------------------------------
  async function verify(): Promise<void> {
    if (verifying) return;
    verifying = true;
    try {
      ctx.setReadout('identical', 'running both traversals...');
      const march = getMainGraph('march');
      const pyramidMain = getMainGraph('pyramid');
      const encoder = device.createCommandEncoder({id: 'viewshed-verify'});
      march.encode(encoder, {parameters: undefined});
      pyramidMain.encode(encoder, {parameters: undefined});
      device.submit(encoder.finish());
      const readWords = async (buffer: Buffer, words: number) =>
        toUint32(await buffer.readAsync(0, words * 4), words);
      const [marchCells, pyramidCells, marchLos, pyramidLos] = await Promise.all([
        readWords(outputs.march.visibility, pixelCount),
        readWords(outputs.pyramid.visibility, pixelCount),
        Promise.all([
          readWords(outputs.march.losCode, 1),
          readWords(outputs.march.losClearance, 1)
        ]),
        Promise.all([
          readWords(outputs.pyramid.losCode, 1),
          readWords(outputs.pyramid.losClearance, 1)
        ])
      ]);
      if (destroyed) return;
      let differing = 0;
      for (let index = 0; index < pixelCount; index++) {
        if (marchCells[index] !== pyramidCells[index]) differing++;
      }
      const losDiffering =
        Number(marchLos[0][0] !== pyramidLos[0][0]) + Number(marchLos[1][0] !== pyramidLos[1][0]);
      ctx.setReadout(
        'identical',
        differing + losDiffering === 0
          ? `identical: ${formatCount(pixelCount)} cells and the sight line match bit for bit`
          : `MISMATCH: ${formatCount(differing)} cells, ${losDiffering} sight-line words`
      );
    } catch (error) {
      if (!destroyed) ctx.setReadout('identical', `failed: ${(error as Error).message}`);
    } finally {
      verifying = false;
    }
  }

  async function measure(): Promise<void> {
    try {
      ctx.setReadout('marchTime', 'measuring...');
      ctx.setReadout('pyramidTime', 'measuring...');
      const run = (compiled: CompiledGPUCommandGraph<void>, completionBuffer: Buffer) =>
        measureCompiledGraph(device, compiled, {
          parameters: undefined,
          completionBuffer,
          signal: ctx.signal,
          runs: 3,
          warmUpRuns: 1,
          repetitions: 2
        });
      const marchTiming = await run(getMainGraph('march'), outputs.march.counts);
      ctx.setReadout('marchTime', formatCompiledGraphTiming(marchTiming));
      const pyramidTiming = await run(getMainGraph('pyramid'), outputs.pyramid.counts);
      ctx.setReadout(
        'pyramidTime',
        `${formatCompiledGraphTiming(pyramidTiming)} · ${formatSpeedup(marchTiming.milliseconds, pyramidTiming.milliseconds)}`
      );
    } catch (error) {
      if (!destroyed) ctx.setReadout('marchTime', `failed: ${(error as Error).message}`);
    }
  }

  // --- Pointer interaction ----------------------------------------------------------------------
  function movePoint(
    which: 'observer' | 'target',
    event: {coordinate: readonly [number, number] | null}
  ): void {
    if (!event.coordinate) return;
    const [x, y] = terrain.projection.project(event.coordinate[0], event.coordinate[1]);
    const point = terrain.getPixelFromMeters(x, y);
    if (which === 'observer') observer = point;
    else target = point;
    writeParameters();
  }

  function getPointerDistance(
    point: readonly [number, number],
    event: {pixel: readonly [number, number]}
  ): number {
    const viewport = ctx.getViewport();
    if (!viewport) return Infinity;
    const [x, y] = terrain.getMeters(point[0], point[1]);
    const [longitude, latitude] = terrain.projection.unproject(x, y);
    const [screenX, screenY] = viewport.project([longitude, latitude]);
    return Math.hypot(screenX - event.pixel[0], screenY - event.pixel[1]);
  }

  // --- Instance ---------------------------------------------------------------------------------
  const getActiveGraphs = (): CompiledGPUCommandGraph<void>[] => {
    const options = ctx.options;
    const graphs: CompiledGPUCommandGraph<void>[] = [baseCompiled, profileCompiled];
    if (pyramidCompiled) graphs.push(pyramidCompiled);
    graphs.push(getMainGraph(options.traversal));
    if (options.display === 'cumulative') graphs.push(getCumulativeGraph(options.traversal));
    return graphs;
  };

  return {
    getCompiledGraphs: () => {
      const graphs: CompiledGPUCommandGraph<void>[] = [baseCompiled, profileCompiled];
      if (pyramidCompiled) graphs.push(pyramidCompiled);
      for (const compiled of graphCache.values()) graphs.push(compiled);
      return graphs;
    },

    setOption(id, value) {
      if (id === 'traversal') {
        // Compile (or reuse) the chosen traversal's graphs and rerun them.
        getActiveGraphs();
        mainDirty = true;
        cumulativeDirty = true;
        ctx.requestLayers();
        return;
      }
      if (id === 'pyramidBlockSize') {
        if (pyramid) {
          getPyramid();
          mainDirty = true;
          cumulativeDirty = true;
        }
        ctx.requestLayers();
        return;
      }
      if (id === 'display') {
        if (value === 'pyramid') getPyramid();
        if (value === 'cumulative') getActiveGraphs();
        cumulativeDirty = true;
        ctx.requestLayers();
        return;
      }
      if (id === 'pyramidLevel' || id === 'base' || id === 'cumulativeMetric') {
        ctx.requestLayers();
        return;
      }
      if (id === 'showProfile') {
        ctx.requestLayers();
        return;
      }
      writeParameters();
      ctx.requestLayers();
    },

    onAction(id) {
      if (id === 'verify') void verify();
      if (id === 'measure') void measure();
    },

    encode(commandEncoder) {
      const options = ctx.options;
      const traversal = options.traversal;
      if (baseDirty) {
        baseCompiled.encode(commandEncoder, {parameters: undefined});
        baseDirty = false;
      }
      if (mainDirty) {
        getMainGraph(traversal).encode(commandEncoder, {parameters: undefined});
        getReader(traversal).request(commandEncoder);
        mainDirty = false;
      }
      if (options.display === 'cumulative' && cumulativeDirty) {
        getCumulativeGraph(traversal).encode(commandEncoder, {parameters: undefined});
        cumulativeDirty = false;
      }
      if (profileDirty) {
        profileCompiled.encode(commandEncoder, {parameters: undefined});
        profileReader.request(commandEncoder);
        profileDirty = false;
      }
      getReader(traversal).flush(commandEncoder);
      profileReader.flush(commandEncoder);
    },

    getLayers() {
      const options = ctx.options;
      const output = outputs[options.traversal];
      const layers: Layer[] = [];
      const rasterProps = {
        coordinateOrigin: origin,
        gridSize: [width, height] as const,
        bounds: terrain.bounds,
        rowOrigin: 'north' as const
      };
      if (options.base === 'hillshade') {
        layers.push(
          new SpatialAnalysisRasterLayer({
            ...rasterProps,
            id: 'viewshed-hillshade',
            values: hillshadeBuffer,
            valueFormat: 'float32',
            colormap: 'grayscale',
            valueRange: [0, 1],
            color: [255, 255, 255, 215]
          })
        );
      } else {
        layers.push(
          new SpatialAnalysisRasterLayer({
            ...rasterProps,
            id: 'viewshed-elevation',
            values: elevationBuffer,
            valueFormat: 'float32',
            colormap: 'cividis',
            valueRange: [terrain.elevationRange[0], terrain.elevationRange[1]],
            color: [255, 255, 255, 215]
          })
        );
      }
      if (options.display === 'pyramid' && pyramid) {
        const levelIndex = Math.min(
          Math.max(Math.round(options.pyramidLevel), 0),
          pyramid.layout.levels.length - 1
        );
        const level = pyramid.layout.levels[levelIndex];
        const blockX = level.blockSize * terrain.layerCellSize[0];
        const blockY = level.blockSize * terrain.layerCellSize[1];
        layers.push(
          new SpatialAnalysisRasterLayer({
            coordinateOrigin: origin,
            id: `viewshed-pyramid-${level.level}`,
            gridSize: [level.width, level.height],
            // Cells start at the raster's north-west corner and may overhang the south-east edge.
            bounds: [
              terrain.bounds[0],
              terrain.bounds[3] - level.height * blockY,
              terrain.bounds[0] + level.width * blockX,
              terrain.bounds[3]
            ],
            rowOrigin: 'north',
            values: pyramid.levelBuffers[levelIndex],
            valueFormat: 'float32',
            colormap: 'viridis',
            valueRange: [terrain.elevationRange[0], terrain.elevationRange[1]],
            discardAtOrBelow: -1e30,
            color: [255, 255, 255, 215]
          })
        );
      } else if (options.display === 'cumulative') {
        const slots = Math.min(
          Math.max(Math.round(options.cumulativeObservers), 1),
          CUMULATIVE_SLOTS
        );
        layers.push(
          new SpatialAnalysisRasterLayer({
            ...rasterProps,
            id: `viewshed-cumulative-${options.cumulativeMetric}`,
            values:
              options.cumulativeMetric === 'visible'
                ? output.cumulativeVisible
                : output.cumulativeMarginal,
            valueFormat: 'uint32',
            colormap: 'viridis',
            valueRange: [1, Math.max(slots, 2)],
            discardAtOrBelow: 0.5,
            color: [255, 255, 255, 215]
          })
        );
      } else {
        layers.push(
          new SpatialAnalysisRasterLayer({
            ...rasterProps,
            id: 'viewshed-codes',
            values: output.visibility,
            valueFormat: 'uint32',
            colormap: 'category',
            // GPU_TERRAIN_VISIBILITY: hidden, visible, outOfRange, noData, marginal.
            palette: [
              [12, 14, 40, 110],
              [60, 230, 110, 120],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [255, 170, 40, 175]
            ]
          })
        );
      }
      const sightColor: [number, number, number, number] =
        losCode === GPU_TERRAIN_VISIBILITY.visible
          ? [40, 200, 100, 255]
          : losCode === GPU_TERRAIN_VISIBILITY.marginal
            ? [255, 170, 40, 255]
            : [235, 60, 70, 255];
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'viewshed-sight-halo',
          coordinateOrigin: origin,
          segments: sightSegment,
          instanceCount: 1,
          widthPixels: 5,
          color: [10, 16, 30, 190]
        }),
        new SpatialAnalysisSegmentLayer({
          id: 'viewshed-sight-line',
          coordinateOrigin: origin,
          segments: sightSegment,
          instanceCount: 1,
          widthPixels: 2.2,
          color: sightColor
        })
      );
      if (options.showProfile && profileCountValue > 0) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'viewshed-profile',
            coordinateOrigin: origin,
            positions: profilePoints,
            instanceCount: profileCountValue,
            values: profileExcess,
            valueFormat: 'float32',
            colormap: 'diverging',
            valueRange: [-80, 80],
            radiusPixels: 2.6
          })
        );
      }
      if (options.display === 'cumulative') {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'viewshed-extra-observers',
            coordinateOrigin: origin,
            positions: extraMarkers,
            instanceCount: CUMULATIVE_SLOTS - 1,
            radiusPixels: 5.5,
            color: [255, 150, 40, 255]
          })
        );
      }
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'viewshed-target-halo',
          coordinateOrigin: origin,
          positions: targetMarker,
          instanceCount: 1,
          radiusPixels: 10,
          color: [255, 255, 255, 235]
        }),
        new SpatialAnalysisPointLayer({
          id: 'viewshed-target',
          coordinateOrigin: origin,
          positions: targetMarker,
          instanceCount: 1,
          radiusPixels: 6,
          color: [40, 190, 245, 255]
        }),
        new SpatialAnalysisPointLayer({
          id: 'viewshed-observer-halo',
          coordinateOrigin: origin,
          positions: observerMarker,
          instanceCount: 1,
          radiusPixels: 11,
          color: [255, 255, 255, 235]
        }),
        new SpatialAnalysisPointLayer({
          id: 'viewshed-observer',
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
      const [x, y] = terrain.projection.project(event.coordinate[0], event.coordinate[1]);
      const [column, row] = terrain.getPixelFromMeters(x, y);
      const elevation = terrain.sampleElevation(column, row);
      const dx = (column - observer[0]) * terrain.groundCellSize;
      const dy = (observer[1] - row) * terrain.groundCellSize;
      const distance = Math.hypot(dx, dy);
      return `${elevation.toFixed(0)} m · ${formatDistance(distance)} ${getCompassName(getAzimuthDegrees(dx, dy))} of the observer`;
    },

    onClick(event) {
      if (!event.coordinate) return false;
      movePoint('target', event);
      return true;
    },

    onDragStart(event) {
      const observerDistance = getPointerDistance(observer, event);
      const targetDistance = getPointerDistance(target, event);
      if (Math.min(observerDistance, targetDistance) > DRAG_RADIUS_PIXELS) return false;
      dragging = observerDistance <= targetDistance ? 'observer' : 'target';
      return true;
    },

    onDrag(event) {
      if (dragging) movePoint(dragging, event);
    },

    onDragEnd(event) {
      if (!dragging) return;
      movePoint(dragging, event);
      dragging = null;
    },

    destroy() {
      destroyed = true;
      for (const reader of Object.values(readers)) reader?.stop();
      profileReader.stop();
      resources.destroy();
    }
  };
}
