// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, GPUHistogram, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import type {GPUParameterBuffer} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  GPURasterExtremaPyramid,
  getGPURasterExtremaPyramidLayout,
  type GPURasterExtremaPyramidLayout
} from '@luma.gl/experimental/gpu-raster';
import {
  GPUTerrainCumulativeViewshed,
  GPUTerrainLineOfSight,
  GPUTerrainViewshed,
  GPU_TERRAIN_SIGHT_LINE_PARAMETER_LENGTH,
  GPU_TERRAIN_VIEWSHED_PARAMETER_LENGTH,
  GPU_TERRAIN_VISIBILITY,
  GPU_TERRAIN_VISIBILITY_TOLERANCE_PARAMETER_LENGTH,
  getGPUTerrainCurvatureCoefficient,
  getGPUTerrainSightLineParameterValues,
  getGPUTerrainViewshedParameterValues,
  getGPUTerrainVisibilityToleranceParameterValues,
  type GPUTerrainViewshedProps
} from '@luma.gl/experimental/gpu-terrain';
import {
  formatArea,
  formatCount,
  formatDistance,
  formatPercent,
  formatSigned
} from '../../cartography/live-text';
import type {ClassTable, MapAnnotation} from '../../cartography/types';
import {getClassTableLayerProps, hexToRgba} from '../../cartography/class-table';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer
} from '../../engine/layers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {
  formatCompiledGraphTiming,
  formatSpeedup,
  measureCompiledGraph
} from '../../engine/vector-timing';
import type {SceneContext, SceneInstance, TooltipContent} from '../scene';
import {
  type AlpsTerrain,
  createElevationBand,
  getAzimuthDegrees,
  getCompassName,
  prepareAlpsTerrain,
  toFloat32,
  toUint32
} from './b14b-terrain';
import {createDemProbe, createTerrainDemFromTerrain, earthDrop, type DemProbe} from './cpu-dem';
import {createTerrainGround, rasterizeGlacierMask} from './terrain-ground';
import {BARE_EARTH_CHIP, demSampleLine} from './terrain-furniture';
import {type AlpsContext, glacierLabels, loadAlpsContext, placeLabels} from './terrain-places';
import {
  getObserverColors,
  getTerrainInk,
  getVisibilityLayerProps,
  makeCumulativeTable,
  makeMarkVisibleTable,
  makeVisibilityTable
} from './terrain-palettes';
import {
  addClassifyPass,
  addCoarseUnseenPass,
  addCumulativeClassPass,
  addDilatePass,
  addFlipPass,
  addGatherPass
} from './viewshed-kernels';
import {
  getLargestUnseenGap,
  getPeakVisibility,
  resolveLookouts,
  resolveSummits,
  type Lookout,
  type Summit
} from './viewshed-places';
import {
  BEYOND_REACH_CLASS,
  COARSE_STRIDE,
  getCoarseSize,
  getCumulativeLayerTable,
  LOOKOUT_SLOTS,
  makePyramidTable,
  NO_DATA_CLASS
} from './viewshed-style';
import {
  describeVerdict,
  getProfileChart,
  getRayVerdict,
  type RayTolerance,
  type RayVerdict
} from './viewshed-ray';

/** Option state of the viewshed scene. */
export type ViewshedOptions = {
  /** Eye height above the ground at the observer, metres. */
  observerHeight: number;
  /** Height added to every target cell, metres. */
  targetHeight: number;
  /** Maximum distance, kilometres. */
  maxDistance: number;
  /** Refraction coefficient of the curved-earth model. */
  refraction: 'mt-image' | 'gdal';
  /** Curved earth, flat earth, or the curved map with the cells that flip against flat marked. */
  earthModel: 'curved' | 'flat' | 'changes';
  toleranceMeters: number;
  tolerancePerKilometer: number;
  targetIgnoreDistance: number;
  targetIgnoreFraction: number;
  /** The wrong-way comparison: paint what is seen instead of veiling what is hidden. */
  markVisible: boolean;
  /** Draw the sight line to the teal target. */
  showSightLine: boolean;
  /** The sight-line target: a named summit or the point the reader dragged. */
  rayTarget: 'matterhorn' | 'zumsteinspitze' | 'custom';
  /** The scan position along the sight line, percent from the eye. */
  rayPosition: number;
  display: 'viewshed' | 'cumulative' | 'pyramid';
  /** Lookouts counted by the cumulative viewshed, the observer first. */
  cumulativeObservers: number;
  pyramidLevel: number;
  /** Analysis cell: the source grid averaged by this stride (1, 2 or 4). */
  cellSize: '1' | '2' | '4';
  traversal: 'march' | 'pyramid';
  pyramidBlockSize: '4' | '8' | '16';
};

type Traversal = ViewshedOptions['traversal'];

/** Slots of the lookout buffers beyond the observer. */
const PARKED_LOOKOUT = -1000;
const VISIBILITY_CODE_COUNT = 5;
const CUMULATIVE_BIN_COUNT = 8;
const DRAG_RADIUS_PIXELS = 20;
const SCAN_DOT_RADIUS_PIXELS = 4.5;
const STATION_NEAR_METERS = 200;
const VISIBILITY_NAMES: Record<number, string> = {
  [GPU_TERRAIN_VISIBILITY.hidden]: 'hidden',
  [GPU_TERRAIN_VISIBILITY.visible]: 'visible',
  [GPU_TERRAIN_VISIBILITY.outOfRange]: 'out of range',
  [GPU_TERRAIN_VISIBILITY.noData]: 'no data',
  [GPU_TERRAIN_VISIBILITY.marginal]: 'marginal'
};
const REFRACTION_COEFFICIENT: Record<ViewshedOptions['refraction'], number> = {
  'mt-image': 0.13,
  gdal: 1 / 7
};
/** Ring distances in metres that are drawn besides the maximum distance. */
const REFERENCE_RINGS = [2000, 5000];
/** The camera zoom of the sight-line step, kept when the target changes. */
const SIGHT_LINE_ZOOM = 12.3;
/** Sight-line target presets: the summit named in the gazetteer or the context dataset. */
const TARGET_SUMMITS = {matterhorn: 'Matterhorn', zumsteinspitze: 'Zumsteinspitze'} as const;

/** Outputs of one traversal; the two are compared bit for bit by the verify button. */
type TraversalOutputs = {
  visibility: Buffer;
  classes: Buffer;
  counts: Buffer;
  losCode: Buffer;
  losClearance: Buffer;
  peakCodes: Buffer;
};

type CumulativeOutputs = {
  counts: Buffer;
  classes: Buffer;
  histogram: Buffer;
  coarseMask: Buffer;
};

type PyramidState = {
  blockSize: number;
  layout: GPURasterExtremaPyramidLayout;
  combined: Buffer;
  levelBuffers: Buffer[];
};

/** Everything that depends on the analysis cell: the grid, its buffers, graphs and readers. */
type Analysis = {
  stride: number;
  terrain: AlpsTerrain;
  probe: DemProbe;
  resources: SpatialAnalysisResources;
  summits: Summit[];
  summitCellsBuffer: Buffer;
  elevationBuffer: Buffer;
  validityBuffer: Buffer;
  viewshedSettings: GPUParameterBuffer<'float32'>;
  oppositeSettings: GPUParameterBuffer<'float32'>;
  toleranceSettings: GPUParameterBuffer<'float32'>;
  sightSettings: GPUParameterBuffer<'float32'>;
  reachSettings: GPUParameterBuffer<'float32'>;
  sightPair: Buffer;
  lookoutBuffer: Buffer;
  observerMarker: Buffer;
  targetMarker: Buffer;
  extraMarkers: Buffer;
  sightSegment: Buffer;
  scanSegment: Buffer;
  scanMarker: Buffer;
  outputs: Partial<Record<Traversal, TraversalOutputs>>;
  cumulative: Partial<Record<Traversal, CumulativeOutputs>>;
  graphs: Map<string, CompiledGPUCommandGraph<void>>;
  readers: Map<string, SummaryReader>;
  /** Pyramid state, built on first use. */
  pyramid: PyramidState | null;
  pyramidCompiled: CompiledGPUCommandGraph<void> | null;
  /** Opposite-model viewshed, flips and drawn flips (the earth-model comparison). */
  compare: {altVisibility: Buffer; flips: Buffer; flipsDrawn: Buffer; flipCounts: Buffer} | null;
  destroy: () => void;
};

const clamp = (value: number, low: number, high: number) => Math.min(Math.max(value, low), high);

/**
 * Viewshed, line of sight and cumulative viewshed over the wide Gornergrat DEM.
 *
 * The analysis runs on one grid at a time (the source DEM averaged by a stride of 1, 2 or 4: the
 * cell-size control rebuilds it) while the relief ground is always built from the full-resolution
 * DEM. Both traversals produce bit-identical output, so each is compiled lazily into its own graph
 * with its own output buffers: choosing one changes which graph is encoded and which buffer is
 * drawn. The min-max pyramid is built once by `GPURasterExtremaPyramid` and shared by every
 * `'pyramid'` consumer.
 *
 * The scene kernels in `viewshed-kernels.ts` turn the visibility codes into class values for the
 * chapter class table, flag the cells that flip between two earth models, classify the lookout
 * counts and gather the codes at the named summits, so only a few words are ever read back.
 */
export async function createViewshed(
  ctx: SceneContext<ViewshedOptions>
): Promise<SceneInstance<ViewshedOptions>> {
  const {device} = ctx;
  const dataset = ctx.datasets.get('alps-dem-wide');
  ctx.setStatus('Reading the context');
  const alpsContext: AlpsContext = await loadAlpsContext(
    ctx.datasets.get('alps-context'),
    ctx.signal
  );
  const fullTerrain = prepareAlpsTerrain(dataset, 1);
  const fullDem = createTerrainDemFromTerrain(fullTerrain);
  const origin: [number, number, number] = [fullTerrain.origin[0], fullTerrain.origin[1], 0];
  const [west, south, east, north] = fullTerrain.lngLatBounds;

  // The relief ground, built once from the full-resolution DEM, glaciers as pale ice.
  ctx.setStatus('Drawing the relief');
  const ground = createTerrainGround({
    dem: fullDem,
    device,
    ground: ctx.ground(),
    glacierMask: rasterizeGlacierMask(alpsContext.glacierGeoJson, fullDem)
  });
  await ground.prepare();
  if (ctx.signal.aborted) {
    ground.destroy();
    throw new DOMException('Aborted', 'AbortError');
  }
  ctx.setLegendData('ground', ctx.ground());

  const lookouts: Lookout[] = resolveLookouts(alpsContext, {
    name: 'Gornergrat',
    lngLat: [
      fullTerrain.observers[0]?.longitude ?? west,
      fullTerrain.observers[0]?.latitude ?? south
    ]
  });
  ctx.setLegendData(
    'lookouts',
    lookouts.map(lookout => lookout.name)
  );
  const station = lookouts[0];

  let destroyed = false;
  const retired: Analysis[] = [];
  let observerLngLat: [number, number] = [station.lngLat[0], station.lngLat[1]];
  let targetLngLat: [number, number] = [station.lngLat[0], station.lngLat[1]];
  let dragging: 'observer' | 'target' | null = null;
  let mainDirty = true;
  let cumulativeDirty = true;
  let compareDirty = true;
  let losCode = -1;
  let verifying = false;
  let latestPeakCodes: Uint32Array | null = null;
  const shareByStride = new Map<number, number>();
  let shareSignature = '';
  let lastFurnitureKey = '';

  // --- The analysis grid -------------------------------------------------------------------
  function buildAnalysis(stride: number): Analysis {
    const terrain = stride === 1 ? fullTerrain : prepareAlpsTerrain(dataset, stride);
    const dem = createTerrainDemFromTerrain(terrain);
    const resources = new SpatialAnalysisResources(device, `viewshed-${stride}`);
    const summits = resolveSummits(alpsContext, dem);
    const cells = summits.length
      ? Uint32Array.from(summits.map(summit => summit.cellIndex))
      : Uint32Array.of(NO_DATA_CLASS);
    const analysis: Analysis = {
      stride,
      terrain,
      probe: createDemProbe(dem),
      resources,
      summits,
      summitCellsBuffer: resources.createBuffer('summit-cells', cells),
      elevationBuffer: resources.createBuffer('elevation', terrain.elevation),
      validityBuffer: resources.createBuffer('validity', terrain.validity),
      viewshedSettings: resources.createParameterBuffer(
        'viewshed-settings',
        'float32',
        GPU_TERRAIN_VIEWSHED_PARAMETER_LENGTH
      ),
      oppositeSettings: resources.createParameterBuffer(
        'opposite-settings',
        'float32',
        GPU_TERRAIN_VIEWSHED_PARAMETER_LENGTH
      ),
      toleranceSettings: resources.createParameterBuffer(
        'tolerance-settings',
        'float32',
        GPU_TERRAIN_VISIBILITY_TOLERANCE_PARAMETER_LENGTH
      ),
      sightSettings: resources.createParameterBuffer(
        'sight-settings',
        'float32',
        GPU_TERRAIN_SIGHT_LINE_PARAMETER_LENGTH
      ),
      reachSettings: resources.createParameterBuffer('reach-settings', 'float32', 4),
      sightPair: resources.createBuffer('sight-pair', new Float32Array(4)),
      lookoutBuffer: resources.createBuffer('lookouts', new Float32Array(LOOKOUT_SLOTS * 2)),
      observerMarker: resources.createBuffer('observer-marker', new Float32Array(2)),
      targetMarker: resources.createBuffer('target-marker', new Float32Array(2)),
      extraMarkers: resources.createBuffer(
        'extra-markers',
        new Float32Array((LOOKOUT_SLOTS - 1) * 2)
      ),
      sightSegment: resources.createBuffer('sight-segment', new Float32Array(4)),
      scanSegment: resources.createBuffer('scan-segment', new Float32Array(4)),
      scanMarker: resources.createBuffer('scan-marker', new Float32Array(2)),
      outputs: {},
      cumulative: {},
      graphs: new Map(),
      readers: new Map(),
      pyramid: null,
      pyramidCompiled: null,
      compare: null,
      destroy: () => {
        for (const reader of analysis.readers.values()) reader.stop();
        resources.destroy();
      }
    };
    return analysis;
  }

  let analysis: Analysis = buildAnalysis(Number(ctx.options.cellSize));

  // --- Per-traversal outputs ---------------------------------------------------------------
  function getOutputs(traversal: Traversal): TraversalOutputs {
    const existing = analysis.outputs[traversal];
    if (existing) return existing;
    const {resources, terrain, summits} = analysis;
    const created: TraversalOutputs = {
      visibility: resources.createBuffer(`${traversal}-visibility`, terrain.pixelCount * 4),
      classes: resources.createBuffer(`${traversal}-classes`, terrain.pixelCount * 4),
      counts: resources.createBuffer(`${traversal}-counts`, VISIBILITY_CODE_COUNT * 4),
      losCode: resources.createBuffer(`${traversal}-los-code`, 4),
      losClearance: resources.createBuffer(`${traversal}-los-clearance`, 4),
      peakCodes: resources.createBuffer(`${traversal}-peak-codes`, Math.max(1, summits.length) * 4)
    };
    analysis.outputs[traversal] = created;
    return created;
  }

  function getCumulativeOutputs(traversal: Traversal): CumulativeOutputs {
    const existing = analysis.cumulative[traversal];
    if (existing) return existing;
    const {resources, terrain} = analysis;
    const coarse = getCoarseSize(terrain.width, terrain.height);
    const created: CumulativeOutputs = {
      counts: resources.createBuffer(`${traversal}-cumulative-counts`, terrain.pixelCount * 4),
      classes: resources.createBuffer(`${traversal}-cumulative-classes`, terrain.pixelCount * 4),
      histogram: resources.createBuffer(`${traversal}-cumulative-bins`, CUMULATIVE_BIN_COUNT * 4),
      coarseMask: resources.createBuffer(
        `${traversal}-coarse-unseen`,
        coarse.width * coarse.height * 4
      )
    };
    analysis.cumulative[traversal] = created;
    return created;
  }

  // --- Extrema pyramid (built on first use, rebuilt when its block size changes) -----------
  function getPyramid(): PyramidState {
    const blockSize = Number(ctx.options.pyramidBlockSize);
    const current = analysis.pyramid;
    if (current && current.blockSize === blockSize) return current;
    const {resources, terrain, elevationBuffer, validityBuffer} = analysis;
    const {width, height, pixelCount} = terrain;
    if (current) {
      for (const key of [...analysis.graphs.keys()]) {
        if (key.endsWith('-pyramid') && !key.startsWith('compare')) {
          const compiled = analysis.graphs.get(key);
          if (compiled) resources.release(compiled);
          analysis.graphs.delete(key);
        }
      }
      if (analysis.pyramidCompiled) resources.release(analysis.pyramidCompiled);
      for (const buffer of [current.combined, ...current.levelBuffers]) resources.release(buffer);
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
    const compiled = resources.track(graph.compile());
    analysis.pyramidCompiled = compiled;
    const state: PyramidState = {blockSize, layout, combined, levelBuffers};
    analysis.pyramid = state;
    // Build the pyramid once and copy each maximum level out for display.
    const encoder = device.createCommandEncoder({id: 'viewshed-pyramid-encoder'});
    compiled.encode(encoder, {parameters: undefined});
    layout.levels.forEach((level, index) => {
      encoder.copyBufferToBuffer({
        sourceBuffer: combined,
        sourceOffset: level.offset * 4,
        destinationBuffer: levelBuffers[index],
        size: level.width * level.height * 4
      });
    });
    device.submit(encoder.finish());
    void readPyramidRoot(analysis, state);
    return state;
  }

  async function readPyramidRoot(owner: Analysis, state: PyramidState): Promise<void> {
    try {
      const root = state.layout.levels[state.layout.levels.length - 1];
      const [maximumBytes, minimumBytes] = await Promise.all([
        state.combined.readAsync(root.offset * 4, 4),
        state.combined.readAsync((state.layout.length + root.offset) * 4, 4)
      ]);
      if (destroyed || analysis !== owner || owner.pyramid !== state) return;
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

  // --- Lazily compiled graphs --------------------------------------------------------------
  function getMainGraph(traversal: Traversal): CompiledGPUCommandGraph<void> {
    const key = `main-${traversal}`;
    const cached = analysis.graphs.get(key);
    if (cached) return cached;
    const {resources, terrain, summits} = analysis;
    const {width, height, pixelCount} = terrain;
    const output = getOutputs(traversal);
    const graph = new GPUCommandGraph<void>(device, {id: `viewshed-${traversal}`});
    const elevation = createElevationBand(
      graph,
      'main',
      analysis.elevationBuffer,
      analysis.validityBuffer,
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
        settings: analysis.viewshedSettings.importToGraph(graph),
        tolerance: analysis.toleranceSettings.importToGraph(graph),
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
        pairs: importGraphBuffer(graph, 'sight-pair', analysis.sightPair, 'float32x4', 1),
        settings: analysis.sightSettings.importToGraph(graph),
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
    addClassifyPass(graph, {
      id: 'classify-visibility',
      codes: visibility,
      classes: importGraphBuffer(graph, 'classes', output.classes, 'uint32', pixelCount),
      count: pixelCount
    });
    addGatherPass(graph, {
      id: 'gather-summit-codes',
      source: visibility,
      cells: importGraphBuffer(
        graph,
        'summit-cells',
        analysis.summitCellsBuffer,
        'uint32',
        Math.max(1, summits.length)
      ),
      values: importGraphBuffer(
        graph,
        'peak-codes',
        output.peakCodes,
        'uint32',
        Math.max(1, summits.length)
      ),
      count: Math.max(1, summits.length)
    });
    const compiled = resources.track(graph.compile());
    analysis.graphs.set(key, compiled);
    return compiled;
  }

  function getCumulativeGraph(traversal: Traversal): CompiledGPUCommandGraph<void> {
    const key = `cumulative-${traversal}`;
    const cached = analysis.graphs.get(key);
    if (cached) return cached;
    const {resources, terrain} = analysis;
    const {width, height, pixelCount} = terrain;
    const output = getCumulativeOutputs(traversal);
    const graph = new GPUCommandGraph<void>(device, {id: `viewshed-${key}`});
    const counts = importGraphBuffer(graph, 'visible-count', output.counts, 'uint32', pixelCount);
    graph.add(
      new GPUTerrainCumulativeViewshed({
        id: 'cumulative',
        width,
        height,
        elevation: createElevationBand(
          graph,
          'cumulative',
          analysis.elevationBuffer,
          analysis.validityBuffer,
          pixelCount
        ),
        traversal,
        pyramid: traversal === 'pyramid' ? getSharedPyramid(graph) : undefined,
        observers: importGraphBuffer(
          graph,
          'observers',
          analysis.lookoutBuffer,
          'float32x2',
          LOOKOUT_SLOTS
        ),
        settings: analysis.sightSettings.importToGraph(graph),
        visibleCount: counts
      })
    );
    const classes = importGraphBuffer(graph, 'classes', output.classes, 'uint32', pixelCount);
    addCumulativeClassPass(graph, {
      id: 'classify-cumulative',
      counts,
      lookouts: importGraphBuffer(
        graph,
        'lookouts',
        analysis.lookoutBuffer,
        'float32',
        LOOKOUT_SLOTS * 2
      ),
      reach: analysis.reachSettings.importToGraph(graph),
      classes,
      width,
      height
    });
    graph.add(
      new GPUHistogram({
        id: 'cumulative-bins',
        input: classes,
        output: importGraphBuffer(graph, 'bins', output.histogram, 'uint32', CUMULATIVE_BIN_COUNT),
        edges: Array.from({length: CUMULATIVE_BIN_COUNT + 1}, (_, index) => index)
      })
    );
    const coarse = getCoarseSize(width, height);
    addCoarseUnseenPass(graph, {
      id: 'coarse-unseen',
      classes,
      mask: importGraphBuffer(
        graph,
        'coarse-mask',
        output.coarseMask,
        'uint32',
        coarse.width * coarse.height
      ),
      width,
      height
    });
    const compiled = resources.track(graph.compile());
    analysis.graphs.set(key, compiled);
    return compiled;
  }

  function getCompareGraph(traversal: Traversal): CompiledGPUCommandGraph<void> {
    const key = `compare-${traversal}`;
    const cached = analysis.graphs.get(key);
    if (cached) return cached;
    const {resources, terrain} = analysis;
    const {width, height, pixelCount} = terrain;
    if (!analysis.compare) {
      analysis.compare = {
        altVisibility: resources.createBuffer('alt-visibility', pixelCount * 4),
        flips: resources.createBuffer('flips', pixelCount * 4),
        flipsDrawn: resources.createBuffer('flips-drawn', pixelCount * 4),
        flipCounts: resources.createBuffer('flip-counts', 2 * 4)
      };
    }
    const compare = analysis.compare;
    const graph = new GPUCommandGraph<void>(device, {id: `viewshed-${key}`});
    const alt = importGraphBuffer(
      graph,
      'alt-visibility',
      compare.altVisibility,
      'uint32',
      pixelCount
    );
    graph.add(
      new GPUTerrainViewshed({
        id: 'viewshed-opposite',
        width,
        height,
        elevation: createElevationBand(
          graph,
          'compare',
          analysis.elevationBuffer,
          analysis.validityBuffer,
          pixelCount
        ),
        traversal: 'march',
        settings: analysis.oppositeSettings.importToGraph(graph),
        tolerance: analysis.toleranceSettings.importToGraph(graph),
        visibility: alt
      })
    );
    const flips = importGraphBuffer(graph, 'flips', compare.flips, 'uint32', pixelCount);
    addFlipPass(graph, {
      id: 'flip-cells',
      first: importGraphBuffer(
        graph,
        'main-visibility',
        getOutputs(traversal).visibility,
        'uint32',
        pixelCount
      ),
      second: alt,
      flips,
      count: pixelCount
    });
    addDilatePass(graph, {
      id: 'dilate-flips',
      source: flips,
      grown: importGraphBuffer(graph, 'flips-drawn', compare.flipsDrawn, 'uint32', pixelCount),
      width,
      height
    });
    graph.add(
      new GPUHistogram({
        id: 'flip-counts',
        input: flips,
        output: importGraphBuffer(graph, 'flip-counts', compare.flipCounts, 'uint32', 2),
        edges: [0, 1, 2]
      })
    );
    const compiled = resources.track(graph.compile());
    analysis.graphs.set(key, compiled);
    return compiled;
  }

  // --- Parameters and the observer state ---------------------------------------------------
  const getRefraction = (): number => REFRACTION_COEFFICIENT[ctx.options.refraction];
  /** Refraction coefficient of the displayed model; 1 makes `c = 0` (flat earth). */
  const getModelRefraction = (): number =>
    ctx.options.earthModel === 'flat' ? 1 : getRefraction();
  const getCurvature = (refraction: number): number =>
    refraction >= 1 ? 0 : getGPUTerrainCurvatureCoefficient(refraction);

  const getPixel = (lngLat: readonly [number, number]): [number, number] => {
    const terrain = analysis.terrain;
    const [column, row] = terrain.getPixel(lngLat[0], lngLat[1]);
    return [clamp(column, 0, terrain.width - 1), clamp(row, 0, terrain.height - 1)];
  };
  const getCellSize = (): [number, number] => [
    analysis.terrain.groundCellSize,
    analysis.terrain.groundCellSize
  ];
  const getTolerance = (): RayTolerance => ({
    toleranceMeters: ctx.options.toleranceMeters,
    tolerancePerKilometer: ctx.options.tolerancePerKilometer,
    targetIgnoreDistance: ctx.options.targetIgnoreDistance,
    targetIgnoreFraction: ctx.options.targetIgnoreFraction
  });

  const getSummit = (name: string): Summit | undefined =>
    analysis.summits.find(summit => summit.name === name);

  function resolveTarget(): void {
    const preset = ctx.options.rayTarget;
    if (preset === 'custom') return;
    const summit = getSummit(TARGET_SUMMITS[preset]);
    if (summit) targetLngLat = [summit.lngLat[0], summit.lngLat[1]];
  }

  let latestRay: {verdict: RayVerdict; chartKey: string} | null = null;
  let latestHorizon: {lngLat: [number, number]; distance: number; height: number} | null = null;

  /** The CPU sight line from the observer to the target: profile chart, scan dot and the note. */
  function refreshRay(): void {
    const options = ctx.options;
    const {terrain, probe} = analysis;
    const refraction = getModelRefraction();
    const ray = probe.sampleRay(observerLngLat, targetLngLat, terrain.groundCellSize, {
      refraction,
      eyeHeight: options.observerHeight,
      targetHeight: options.targetHeight
    });
    const verdict = getRayVerdict(ray, getTolerance());
    const scanFraction = clamp(options.rayPosition / 100, 0, 1);
    const scanIndex = Math.min(ray.count - 1, Math.round(scanFraction * (ray.count - 1)));
    // The scan dot and the stretch scanned so far, in layer metres.
    const observerMeters = terrain.projection.project(observerLngLat[0], observerLngLat[1]);
    const scanMeters = terrain.projection.project(
      ray.lngLat[scanIndex * 2],
      ray.lngLat[scanIndex * 2 + 1]
    );
    analysis.scanMarker.write(Float32Array.of(...scanMeters));
    analysis.scanSegment.write(Float32Array.of(...observerMeters, ...scanMeters));
    const horizonLngLat: [number, number] | null =
      verdict.horizonIndex >= 0
        ? [ray.lngLat[verdict.horizonIndex * 2], ray.lngLat[verdict.horizonIndex * 2 + 1]]
        : null;
    latestHorizon = horizonLngLat
      ? {lngLat: horizonLngLat, distance: verdict.horizonDistance, height: verdict.horizonHeight}
      : null;
    latestRay = {verdict, chartKey: ''};
    if (options.showSightLine) {
      const targetName =
        options.rayTarget === 'custom' ? 'the target' : TARGET_SUMMITS[options.rayTarget];
      ctx.setChart(
        'profile',
        getProfileChart(ray, verdict, {
          scanFraction,
          showCurve: refraction < 1,
          observerName: station && isNearStation() ? 'Gornergrat station' : 'the observer',
          targetName
        })
      );
      ctx.setReadout(
        'rayLength',
        `${formatDistance(ray.totalDistance)}, steepest ground ${formatDistance(verdict.horizonDistance)} from the eye (${verdict.code})`
      );
    }
    ctx.requestLayers();
  }

  function isNearStation(): boolean {
    const [x, y] = analysis.terrain.projection.project(observerLngLat[0], observerLngLat[1]);
    const [stationX, stationY] = analysis.terrain.projection.project(
      station.lngLat[0],
      station.lngLat[1]
    );
    return Math.hypot(x - stationX, y - stationY) < STATION_NEAR_METERS;
  }

  function writeParameters(): void {
    const options = ctx.options;
    const {terrain} = analysis;
    const observer = getPixel(observerLngLat);
    const target = getPixel(targetLngLat);
    const maxDistance = options.maxDistance * 1000;
    const shownRefraction = getModelRefraction();
    const oppositeRefraction = options.earthModel === 'flat' ? getRefraction() : 1;
    const common = {
      observer,
      observerHeight: options.observerHeight,
      targetHeight: options.targetHeight,
      maxDistance,
      cellSize: getCellSize()
    };
    analysis.viewshedSettings.write(
      getGPUTerrainViewshedParameterValues({
        ...common,
        curvatureCoefficient: getCurvature(shownRefraction)
      })
    );
    analysis.oppositeSettings.write(
      getGPUTerrainViewshedParameterValues({
        ...common,
        curvatureCoefficient: getCurvature(oppositeRefraction)
      })
    );
    analysis.toleranceSettings.write(
      getGPUTerrainVisibilityToleranceParameterValues({
        toleranceMeters: options.toleranceMeters,
        tolerancePerKilometer: options.tolerancePerKilometer,
        targetIgnoreDistance: options.targetIgnoreDistance,
        targetIgnoreFraction: options.targetIgnoreFraction
      })
    );
    analysis.sightSettings.write(
      getGPUTerrainSightLineParameterValues({
        observerHeight: options.observerHeight,
        targetHeight: options.targetHeight,
        maxDistance,
        cellSize: getCellSize(),
        curvatureCoefficient: getCurvature(shownRefraction),
        toleranceMeters: options.toleranceMeters,
        tolerancePerKilometer: options.tolerancePerKilometer,
        targetIgnoreDistance: options.targetIgnoreDistance,
        targetIgnoreFraction: options.targetIgnoreFraction
      })
    );
    analysis.reachSettings.write(Float32Array.of(maxDistance, terrain.groundCellSize, 0, 0));
    analysis.sightPair.write(Float32Array.of(observer[0], observer[1], target[0], target[1]));
    const slots = clamp(Math.round(options.cumulativeObservers), 1, lookouts.length);
    const lookoutRows = new Float32Array(LOOKOUT_SLOTS * 2).fill(PARKED_LOOKOUT);
    const extraMeters = new Float32Array((LOOKOUT_SLOTS - 1) * 2).fill(1e7);
    lookoutRows[0] = observer[0];
    lookoutRows[1] = observer[1];
    for (let slot = 1; slot < slots; slot++) {
      const pixel = getPixel(lookouts[slot].lngLat);
      lookoutRows[slot * 2] = pixel[0];
      lookoutRows[slot * 2 + 1] = pixel[1];
      const meters = terrain.projection.project(lookouts[slot].lngLat[0], lookouts[slot].lngLat[1]);
      extraMeters[(slot - 1) * 2] = meters[0];
      extraMeters[(slot - 1) * 2 + 1] = meters[1];
    }
    analysis.lookoutBuffer.write(lookoutRows);
    analysis.extraMarkers.write(extraMeters);
    const observerMeters = terrain.projection.project(observerLngLat[0], observerLngLat[1]);
    const targetMeters = terrain.projection.project(targetLngLat[0], targetLngLat[1]);
    analysis.observerMarker.write(Float32Array.of(...observerMeters));
    analysis.targetMarker.write(Float32Array.of(...targetMeters));
    analysis.sightSegment.write(Float32Array.of(...observerMeters, ...targetMeters));
    ctx.setReadout(
      'dropAtReach',
      `${earthDrop(maxDistance, getRefraction()).toFixed(1)} m at ${formatDistance(maxDistance)}`
    );
    mainDirty = true;
    cumulativeDirty = true;
    compareDirty = true;
    ctx.setReadout('identical', 'press Time and compare');
    refreshRay();
  }

  // --- Furniture ---------------------------------------------------------------------------
  function updateFurniture(): void {
    const options = ctx.options;
    const {terrain} = analysis;
    const earth =
      options.earthModel === 'flat'
        ? 'flat earth'
        : options.refraction === 'gdal'
          ? 'k = 1/7'
          : 'k = 0.13';
    const subtitle =
      options.display === 'cumulative'
        ? `${Math.min(options.cumulativeObservers, lookouts.length)} lookouts · eye ${options.observerHeight} m · ${options.maxDistance} km · ${earth}`
        : options.display === 'pyramid'
          ? 'Highest ground per pyramid block, metres'
          : `Eye ${options.observerHeight} m · target ${options.targetHeight} m · ${options.maxDistance} km · ${earth}`;
    const key = `${subtitle}|${terrain.width}|${options.maxDistance}`;
    if (key === lastFurnitureKey) return;
    lastFurnitureKey = key;
    ctx.setFurniture({
      title: {subtitle, sample: demSampleLine(terrain), chips: [BARE_EARTH_CHIP]},
      scaleBar: {units: 'metric', ticks: [options.maxDistance * 1000]}
    });
  }

  // --- Annotations -------------------------------------------------------------------------
  function publishObserverAnnotations(): void {
    const options = ctx.options;
    const ground = analysis.probe.sampleAt(observerLngLat);
    const labels: MapAnnotation[] = [
      {
        kind: 'point',
        id: 'observer',
        coordinate: observerLngLat,
        text: isNearStation() ? 'Gornergrat station' : 'Observer',
        ...(Number.isFinite(ground)
          ? {detail: `${Math.round(ground).toLocaleString('en-US')} m`}
          : {}),
        marker: 'none',
        rank: 'subject',
        priority: 40
      }
    ];
    if (options.display === 'viewshed') {
      const reach = options.maxDistance * 1000;
      const drop = earthDrop(reach, getRefraction());
      for (const meters of REFERENCE_RINGS) {
        if (meters < reach - 600) {
          labels.push({
            kind: 'ring',
            id: `ring-${meters}`,
            coordinate: observerLngLat,
            radiusMeters: meters,
            text: formatDistance(meters),
            dashed: true,
            tone: 'muted'
          });
        }
      }
      labels.push({
        kind: 'ring',
        id: 'ring-reach',
        coordinate: observerLngLat,
        radiusMeters: reach,
        text:
          options.earthModel === 'flat'
            ? `${formatDistance(reach)} · flat earth`
            : `${formatDistance(reach)} · earth drops ${drop.toFixed(1)} m`,
        dashed: true,
        tone: 'ink'
      });
    }
    ctx.setAnnotations('observer', labels);
  }

  function publishPeakAnnotations(): void {
    const options = ctx.options;
    const summits = analysis.summits;
    if (!summits.length) {
      ctx.setAnnotations('peaks', null);
      return;
    }
    const codes =
      latestPeakCodes ?? Uint32Array.from(summits, () => GPU_TERRAIN_VISIBILITY.visible);
    const useVisibility = options.display === 'viewshed' && latestPeakCodes !== null;
    const extraNamed =
      options.showSightLine && options.rayTarget !== 'custom'
        ? [TARGET_SUMMITS[options.rayTarget]]
        : [];
    const selection = getPeakVisibility(summits, codes, {
      maxLabels: useVisibility ? 4 + extraNamed.length : 2 + extraNamed.length,
      extraNamed
    });
    ctx.setAnnotations('peaks', selection.labels);
    if (options.display === 'viewshed' && latestPeakCodes) {
      const listed = selection.hidden.slice(0, 5).join(', ');
      const more = selection.hidden.length > 5 ? ` and ${selection.hidden.length - 5} more` : '';
      ctx.setReadout(
        'hiddenPeaks',
        selection.hidden.length ? `${listed}${more}` : 'no named summit in range'
      );
    }
  }

  function publishContextAnnotations(): void {
    const window = analysis.terrain.lngLatBounds;
    const labels: MapAnnotation[] = [
      ...placeLabels(alpsContext, {
        kinds: ['settlement'],
        names: ['Zermatt'],
        window,
        max: 1,
        rank: 'context'
      }),
      ...glacierLabels(alpsContext, {window, names: ['Gornergletscher'], max: 1, minZoom: 12.4})
    ].map(label => ({...label, priority: label.kind === 'water' ? 1 : 12}) as MapAnnotation);
    ctx.setAnnotations('context', labels);
    ctx.setAnnotations('frame', [
      {
        kind: 'frame',
        id: 'data-edge',
        bounds: [west, south, east, north],
        text: 'Data ends here',
        maxZoom: 11.9
      }
    ]);
  }

  function publishMarkerAnnotations(): void {
    const options = ctx.options;
    if (options.display !== 'cumulative') {
      ctx.setAnnotations('markers', null);
      return;
    }
    const slots = clamp(Math.round(options.cumulativeObservers), 1, lookouts.length);
    ctx.setAnnotations(
      'markers',
      lookouts.slice(1, slots).map((lookout, index) => ({
        kind: 'marker',
        id: `lookout-${index + 2}`,
        coordinate: lookout.lngLat as [number, number],
        number: index + 2,
        text: lookout.name
      }))
    );
  }

  function publishNoteAnnotations(): void {
    const options = ctx.options;
    const notes: MapAnnotation[] = [];
    if (options.showSightLine && latestHorizon && latestRay) {
      notes.push({
        kind: 'note',
        id: 'steepest-ground',
        coordinate: latestHorizon.lngLat,
        title: 'Steepest ground',
        text: `${formatDistance(latestHorizon.distance)} from the eye, ${Math.round(latestHorizon.height).toLocaleString('en-US')} m`,
        priority: 5
      });
    }
    if (options.display === 'cumulative' && unseenGap) {
      notes.push({
        kind: 'note',
        id: 'unseen-gap',
        coordinate: unseenGap.lngLat,
        title: 'No station sees this slope',
        text: `The largest unseen circle is ${formatDistance(unseenGap.radiusMeters * 2)} across.`,
        priority: 5
      });
      notes.push({
        kind: 'ring',
        id: 'unseen-ring',
        coordinate: unseenGap.lngLat,
        radiusMeters: unseenGap.radiusMeters,
        dashed: false,
        tone: 'accent'
      });
    }
    ctx.setAnnotations('notes', notes);
  }

  let unseenGap: {lngLat: [number, number]; radiusMeters: number} | null = null;

  function refreshAnnotations(): void {
    publishObserverAnnotations();
    publishPeakAnnotations();
    publishMarkerAnnotations();
    publishNoteAnnotations();
  }

  // --- Summaries read back from the GPU ----------------------------------------------------
  function recordShare(visibleShare: number): void {
    const options = ctx.options;
    const signature = JSON.stringify([
      observerLngLat.map(value => Math.round(value * 1e5)),
      options.observerHeight,
      options.targetHeight,
      options.maxDistance,
      options.earthModel === 'flat',
      options.refraction,
      options.toleranceMeters,
      options.tolerancePerKilometer,
      options.targetIgnoreDistance,
      options.targetIgnoreFraction
    ]);
    if (signature !== shareSignature) {
      shareSignature = signature;
      shareByStride.clear();
    }
    shareByStride.set(analysis.stride, visibleShare);
    publishResolutionChart();
  }

  function publishResolutionChart(): void {
    const labels = [1, 2, 4].map(stride => formatDistance(fullTerrain.groundCellSize * stride));
    const values = [1, 2, 4].map(stride => (shareByStride.get(stride) ?? Number.NaN) * 100);
    const highest = Math.max(10, ...values.filter(Number.isFinite));
    ctx.setChart('visibleByCell', {
      kind: 'bars',
      height: 120,
      values,
      labels,
      highlight: [[1, 2, 4].indexOf(analysis.stride)],
      yLabel: 'Visible ground (%)',
      yDomain: [0, Math.ceil((highest * 1.25) / 5) * 5],
      formatY: value => `${value.toFixed(0)}`,
      description:
        'Share of the ground in reach that is visible from the observer, by analysis cell size; sizes not tried yet have no bar.'
    });
  }

  function applyMainSummary(owner: Analysis, traversal: Traversal, bytes: ArrayBuffer): void {
    if (owner !== analysis || traversal !== ctx.options.traversal) return;
    const {summits} = analysis;
    const counts = toUint32(bytes, VISIBILITY_CODE_COUNT);
    const code = toUint32(bytes.slice(VISIBILITY_CODE_COUNT * 4), 1)[0];
    const clearance = toFloat32(bytes.slice((VISIBILITY_CODE_COUNT + 1) * 4), 1)[0];
    const peakCodes = toUint32(
      bytes.slice((VISIBILITY_CODE_COUNT + 2) * 4),
      Math.max(1, summits.length)
    );
    const hidden = counts[GPU_TERRAIN_VISIBILITY.hidden];
    const visible = counts[GPU_TERRAIN_VISIBILITY.visible];
    const marginal = counts[GPU_TERRAIN_VISIBILITY.marginal];
    const outOfRange = counts[GPU_TERRAIN_VISIBILITY.outOfRange];
    const inRange = hidden + visible + marginal;
    const cellArea = analysis.terrain.groundCellSize ** 2;
    ctx.setReadout('visibleShare', inRange > 0 ? formatPercent(visible / inRange, 1) : null);
    ctx.setReadout('visibleArea', inRange > 0 ? formatArea(visible * cellArea) : null);
    ctx.setReadout('marginalShare', inRange > 0 ? formatPercent(marginal / inRange, 1) : null);
    ctx.setReadout(
      'range',
      `${formatCount(inRange)} cells in range, ${formatCount(outOfRange)} beyond`
    );
    ctx.setCost({records: inRange, passes: 4});
    if (inRange > 0) recordShare(visible / inRange);
    const showClearance =
      code === GPU_TERRAIN_VISIBILITY.visible ||
      code === GPU_TERRAIN_VISIBILITY.hidden ||
      code === GPU_TERRAIN_VISIBILITY.marginal;
    ctx.setReadout(
      'clearance',
      showClearance && Number.isFinite(clearance) && Math.abs(clearance) < 1e30
        ? `${formatSigned(clearance)} m`
        : (VISIBILITY_NAMES[code] ?? null)
    );
    latestPeakCodes = peakCodes;
    publishPeakAnnotations();
    if (code !== losCode) {
      losCode = code;
      ctx.requestLayers();
    }
  }

  function applyCumulativeSummary(owner: Analysis, bytes: ArrayBuffer): void {
    if (owner !== analysis || ctx.options.display !== 'cumulative') return;
    const terrain = analysis.terrain;
    const bins = toUint32(bytes, CUMULATIVE_BIN_COUNT);
    const coarse = getCoarseSize(terrain.width, terrain.height);
    const mask = toUint32(bytes.slice(CUMULATIVE_BIN_COUNT * 4), coarse.width * coarse.height);
    let reached = 0;
    for (let count = 0; count < BEYOND_REACH_CLASS; count++) reached += bins[count];
    const active = clamp(Math.round(ctx.options.cumulativeObservers), 1, lookouts.length);
    ctx.setReadout('seenByNone', reached > 0 ? formatPercent(bins[0] / reached, 0) : null);
    ctx.setReadout('seenByAll', reached > 0 ? formatPercent(bins[active] / reached, 1) : null);
    const gap = getLargestUnseenGap(mask, coarse.width, coarse.height);
    if (gap) {
      const column = Math.min(
        gap.coarseColumn * COARSE_STRIDE + COARSE_STRIDE / 2,
        terrain.width - 1
      );
      const row = Math.min(gap.coarseRow * COARSE_STRIDE + COARSE_STRIDE / 2, terrain.height - 1);
      const lngLat = terrain.getLongitudeLatitude(column, row);
      unseenGap = {lngLat, radiusMeters: gap.radiusCells * COARSE_STRIDE * terrain.groundCellSize};
      ctx.setReadout(
        'unseenGap',
        `a circle ${formatDistance(unseenGap.radiusMeters * 2)} across holds no seen cell`
      );
    } else {
      unseenGap = null;
      ctx.setReadout('unseenGap', 'none');
    }
    ctx.setCost({records: reached, passes: active});
    publishNoteAnnotations();
  }

  function applyCompareSummary(owner: Analysis, bytes: ArrayBuffer): void {
    if (owner !== analysis || ctx.options.earthModel === 'curved') return;
    const counts = toUint32(bytes, 2);
    ctx.setReadout(
      'flippedCells',
      `${formatCount(counts[1])} cells · ${formatArea(counts[1] * analysis.terrain.groundCellSize ** 2)}`
    );
  }

  function getReader(key: string, make: () => SummaryReader): SummaryReader {
    let reader = analysis.readers.get(key);
    if (!reader) {
      reader = make();
      analysis.readers.set(key, reader);
    }
    return reader;
  }

  const getMainReader = (traversal: Traversal): SummaryReader =>
    getReader(`main-${traversal}`, () => {
      const owner = analysis;
      const output = getOutputs(traversal);
      return new SummaryReader(
        owner.resources,
        `viewshed-${traversal}`,
        [
          {buffer: output.counts, size: VISIBILITY_CODE_COUNT * 4},
          {buffer: output.losCode, size: 4},
          {buffer: output.losClearance, size: 4},
          {buffer: output.peakCodes, size: Math.max(1, owner.summits.length) * 4}
        ],
        bytes => applyMainSummary(owner, traversal, bytes)
      );
    });

  const getCumulativeReader = (traversal: Traversal): SummaryReader =>
    getReader(`cumulative-${traversal}`, () => {
      const owner = analysis;
      const output = getCumulativeOutputs(traversal);
      const coarse = getCoarseSize(owner.terrain.width, owner.terrain.height);
      return new SummaryReader(
        owner.resources,
        `cumulative-${traversal}`,
        [
          {buffer: output.histogram, size: CUMULATIVE_BIN_COUNT * 4},
          {buffer: output.coarseMask, size: coarse.width * coarse.height * 4}
        ],
        bytes => applyCumulativeSummary(owner, bytes)
      );
    });

  const getCompareReader = (): SummaryReader =>
    getReader('compare', () => {
      const owner = analysis;
      getCompareGraph(ctx.options.traversal);
      return new SummaryReader(
        owner.resources,
        'earth-compare',
        [{buffer: owner.compare!.flipCounts, size: 2 * 4}],
        bytes => applyCompareSummary(owner, bytes)
      );
    });

  // --- Verification and timing -------------------------------------------------------------
  async function verify(): Promise<void> {
    if (verifying) return;
    verifying = true;
    const owner = analysis;
    try {
      ctx.setReadout('identical', 'running both traversals...');
      const march = getMainGraph('march');
      const pyramidMain = getMainGraph('pyramid');
      const encoder = device.createCommandEncoder({id: 'viewshed-verify'});
      march.encode(encoder, {parameters: undefined});
      pyramidMain.encode(encoder, {parameters: undefined});
      device.submit(encoder.finish());
      const {pixelCount} = owner.terrain;
      const readWords = async (buffer: Buffer, words: number) =>
        toUint32(await buffer.readAsync(0, words * 4), words);
      const marchOutput = getOutputs('march');
      const pyramidOutput = getOutputs('pyramid');
      const [marchCells, pyramidCells, marchLos, pyramidLos] = await Promise.all([
        readWords(marchOutput.visibility, pixelCount),
        readWords(pyramidOutput.visibility, pixelCount),
        Promise.all([readWords(marchOutput.losCode, 1), readWords(marchOutput.losClearance, 1)]),
        Promise.all([readWords(pyramidOutput.losCode, 1), readWords(pyramidOutput.losClearance, 1)])
      ]);
      if (destroyed || owner !== analysis) return;
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
    const owner = analysis;
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
      const marchTiming = await run(getMainGraph('march'), getOutputs('march').counts);
      if (owner !== analysis) return;
      ctx.setReadout('marchTime', formatCompiledGraphTiming(marchTiming));
      const pyramidTiming = await run(getMainGraph('pyramid'), getOutputs('pyramid').counts);
      if (owner !== analysis) return;
      ctx.setReadout(
        'pyramidTime',
        `${formatCompiledGraphTiming(pyramidTiming)} · ${formatSpeedup(marchTiming.milliseconds, pyramidTiming.milliseconds)}`
      );
      await verify();
    } catch (error) {
      if (!destroyed) ctx.setReadout('marchTime', `failed: ${(error as Error).message}`);
    }
  }

  // --- Pointer interaction -----------------------------------------------------------------
  function movePoint(
    which: 'observer' | 'target',
    event: {coordinate: readonly [number, number] | null}
  ): void {
    if (!event.coordinate) return;
    const lngLat: [number, number] = [
      clamp(event.coordinate[0], west, east),
      clamp(event.coordinate[1], south, north)
    ];
    if (which === 'observer') {
      observerLngLat = lngLat;
      if (ctx.options.rayTarget !== 'custom') resolveTarget();
    } else {
      targetLngLat = lngLat;
      if (ctx.options.rayTarget !== 'custom') ctx.setOptions({rayTarget: 'custom'});
    }
    writeParameters();
    publishObserverAnnotations();
  }

  function getPointerDistance(
    lngLat: readonly [number, number],
    event: {pixel: readonly [number, number]}
  ): number {
    const viewport = ctx.getViewport();
    if (!viewport) return Number.POSITIVE_INFINITY;
    const [screenX, screenY] = viewport.project([lngLat[0], lngLat[1]]);
    return Math.hypot(screenX - event.pixel[0], screenY - event.pixel[1]);
  }

  // --- Tooltip -----------------------------------------------------------------------------
  function getCellTooltip(coordinate: readonly [number, number]): TooltipContent | null {
    const options = ctx.options;
    const {terrain, probe} = analysis;
    if (
      coordinate[0] < west ||
      coordinate[0] > east ||
      coordinate[1] < south ||
      coordinate[1] > north
    ) {
      return null;
    }
    const elevation = probe.sampleAt([coordinate[0], coordinate[1]]);
    if (!Number.isFinite(elevation)) return null;
    const [x, y] = terrain.projection.project(coordinate[0], coordinate[1]);
    const [observerX, observerY] = terrain.projection.project(observerLngLat[0], observerLngLat[1]);
    const dx = x - observerX;
    const dy = y - observerY;
    const distance = Math.hypot(dx, dy);
    const ground = ctx.ground();
    const rows: {
      label: string;
      value: string | number;
      unit?: string;
      swatch?: readonly [number, number, number, number?];
      emphasis?: boolean;
    }[] = [];
    const reach = options.maxDistance * 1000;
    if (options.display === 'viewshed') {
      const table = options.markVisible
        ? makeMarkVisibleTable(ground)
        : makeVisibilityTable(ground);
      const refraction = getModelRefraction();
      if (distance > reach) {
        rows.push({label: 'Visibility', value: 'Out of range', emphasis: true});
      } else if (distance < terrain.groundCellSize) {
        rows.push({label: 'Visibility', value: 'The observer', emphasis: true});
      } else {
        const ray = probe.sampleRay(
          observerLngLat,
          [coordinate[0], coordinate[1]],
          terrain.groundCellSize,
          {
            refraction,
            eyeHeight: options.observerHeight,
            targetHeight: options.targetHeight
          }
        );
        const verdict = getRayVerdict(ray, getTolerance());
        const classIndex = verdict.code === 'visible' ? 0 : verdict.code === 'marginal' ? 1 : 2;
        const color = table.colors[classIndex];
        rows.push({
          label: 'Visibility',
          value: describeVerdict(verdict).replace(/ \(.*/, ''),
          swatch: color[3] === 0 ? undefined : color,
          emphasis: true
        });
        if (verdict.clearanceMeters !== null) {
          rows.push({
            label: 'Clearance',
            value: formatSigned(verdict.clearanceMeters),
            unit: 'm'
          });
        }
      }
    } else if (options.display === 'cumulative') {
      const slots = clamp(Math.round(options.cumulativeObservers), 1, lookouts.length);
      const names: string[] = [];
      let within = 0;
      for (let slot = 0; slot < slots; slot++) {
        const from = slot === 0 ? observerLngLat : lookouts[slot].lngLat;
        const [fromX, fromY] = terrain.projection.project(from[0], from[1]);
        if (Math.hypot(x - fromX, y - fromY) > reach) continue;
        within++;
        const ray = probe.sampleRay(from, [coordinate[0], coordinate[1]], terrain.groundCellSize, {
          refraction: getRefraction(),
          eyeHeight: options.observerHeight,
          targetHeight: options.targetHeight
        });
        if (getRayVerdict(ray, getTolerance()).code === 'visible') {
          names.push(slot === 0 ? 'Gornergrat' : lookouts[slot].name);
        }
      }
      const table = makeCumulativeTable(ground, slots);
      const color = table.colors[Math.min(names.length, table.colors.length - 1)];
      rows.push({
        label: 'Seen from',
        value: within ? `${names.length} of ${slots} stations` : 'no station in reach',
        swatch: within ? color : undefined,
        emphasis: true
      });
      if (names.length) rows.push({label: 'Stations', value: names.join(', ')});
    }
    rows.push({label: 'Ground', value: Math.round(elevation).toLocaleString('en-US'), unit: 'm'});
    rows.push({
      label: 'From the observer',
      value: `${formatDistance(distance)} ${getCompassName(getAzimuthDegrees(dx, dy))}`
    });
    return {
      title: `${Math.round(elevation).toLocaleString('en-US')} m ground`,
      subtitle: `${coordinate[1].toFixed(4)} N, ${coordinate[0].toFixed(4)} E`,
      rows
    };
  }

  // --- Layers ------------------------------------------------------------------------------
  const rasterProps = () => ({
    coordinateOrigin: origin,
    gridSize: [analysis.terrain.width, analysis.terrain.height] as const,
    bounds: analysis.terrain.bounds,
    rowOrigin: 'north' as const
  });

  function getAnalysisLayers(): Layer[] {
    const options = ctx.options;
    const tone = ctx.ground();
    const {terrain} = analysis;
    const output = getOutputs(options.traversal);
    if (options.display === 'pyramid' && analysis.pyramid) {
      const pyramid = analysis.pyramid;
      const levelIndex = clamp(
        Math.round(options.pyramidLevel),
        0,
        pyramid.layout.levels.length - 1
      );
      const level = pyramid.layout.levels[levelIndex];
      const blockX = level.blockSize * terrain.layerCellSize[0];
      const blockY = level.blockSize * terrain.layerCellSize[1];
      const table = getPyramidTable(tone);
      return [
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
          ...getClassTableLayerProps(table),
          discardAtOrBelow: -1e30
        })
      ];
    }
    if (options.display === 'cumulative') {
      const slots = clamp(Math.round(options.cumulativeObservers), 1, lookouts.length);
      const table = getCumulativeLayerTable(tone, slots);
      const cumulative = getCumulativeOutputs(options.traversal);
      return [
        new SpatialAnalysisRasterLayer({
          ...rasterProps(),
          id: 'viewshed-cumulative',
          values: cumulative.classes,
          valueFormat: 'uint32',
          classBreaks: table.breaks,
          classColors: table.colors,
          outlineClasses: {color: [14, 21, 48, 90], widthPixels: 0.75}
        })
      ];
    }
    const table = options.markVisible ? makeMarkVisibleTable(tone) : makeVisibilityTable(tone);
    const layers: Layer[] = [
      new SpatialAnalysisRasterLayer({
        ...rasterProps(),
        id: options.markVisible ? 'viewshed-painted' : 'viewshed-classes',
        values: output.classes,
        valueFormat: 'uint32',
        ...getVisibilityLayerProps(table),
        noDataColor: table.noData?.color ?? [217, 217, 217, 180],
        hatchNoData: true,
        ...(options.markVisible ? {outlineClasses: undefined} : {})
      })
    ];
    if (options.earthModel === 'changes' && analysis.compare) {
      const colors = getObserverColors(tone);
      layers.push(
        new SpatialAnalysisRasterLayer({
          ...rasterProps(),
          id: 'viewshed-flips',
          values: analysis.compare.flipsDrawn,
          valueFormat: 'uint32',
          colormap: 'mask',
          color: [colors.changed[0], colors.changed[1], colors.changed[2], 255],
          noDataColor: [0, 0, 0, 0]
        })
      );
    }
    return layers;
  }

  function getMarkerLayers(): Layer[] {
    const options = ctx.options;
    const tone = ctx.ground();
    const colors = getObserverColors(tone);
    const ink = getTerrainInk(tone);
    const layers: Layer[] = [];
    const inkColor = hexToRgba(ink.ink);
    const haloColor: [number, number, number, number] = [
      colors.halo[0],
      colors.halo[1],
      colors.halo[2],
      Math.round(ink.haloAlpha * 255)
    ];
    if (options.showSightLine && options.display === 'viewshed') {
      // Solid when visible, dashed when marginal, dotted (round caps on a near-zero dash) when hidden.
      const dash: [number, number] | undefined =
        losCode === GPU_TERRAIN_VISIBILITY.visible
          ? undefined
          : losCode === GPU_TERRAIN_VISIBILITY.marginal
            ? [8, 5]
            : [0.5, 5.5];
      const scanning = options.rayPosition < 100;
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'viewshed-sight-casing',
          coordinateOrigin: origin,
          segments: analysis.sightSegment,
          instanceCount: 1,
          widthPixels: 5.5,
          color: haloColor,
          cap: 'butt'
        })
      );
      if (scanning) {
        // The stretch scanned so far, in the observer's gold, under the ink line.
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'viewshed-scan-segment',
            coordinateOrigin: origin,
            segments: analysis.scanSegment,
            instanceCount: 1,
            widthPixels: 4.4,
            color: [colors.eye[0], colors.eye[1], colors.eye[2], 235],
            cap: 'butt'
          })
        );
      }
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: `viewshed-sight-line-${losCode}`,
          coordinateOrigin: origin,
          segments: analysis.sightSegment,
          instanceCount: 1,
          widthPixels: 2.2,
          color: [inkColor[0], inkColor[1], inkColor[2], 255],
          cap: losCode === GPU_TERRAIN_VISIBILITY.hidden || losCode < 0 ? 'round' : 'butt',
          ...(dash ? {dashArray: dash} : {})
        })
      );
      if (scanning) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'viewshed-scan-dot',
            coordinateOrigin: origin,
            positions: analysis.scanMarker,
            instanceCount: 1,
            radiusPixels: SCAN_DOT_RADIUS_PIXELS,
            shape: 'ring',
            outlineWidthPixels: 2,
            color: [inkColor[0], inkColor[1], inkColor[2], 255]
          })
        );
      }
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'viewshed-target-halo',
          coordinateOrigin: origin,
          positions: analysis.targetMarker,
          instanceCount: 1,
          radiusPixels: 10,
          color: haloColor
        }),
        new SpatialAnalysisPointLayer({
          id: 'viewshed-target',
          coordinateOrigin: origin,
          positions: analysis.targetMarker,
          instanceCount: 1,
          radiusPixels: 6,
          color: colors.target
        })
      );
    }
    if (options.display === 'cumulative') {
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'viewshed-extra-halo',
          coordinateOrigin: origin,
          positions: analysis.extraMarkers,
          instanceCount: LOOKOUT_SLOTS - 1,
          radiusPixels: 9,
          color: haloColor
        }),
        new SpatialAnalysisPointLayer({
          id: 'viewshed-extra-lookouts',
          coordinateOrigin: origin,
          positions: analysis.extraMarkers,
          instanceCount: LOOKOUT_SLOTS - 1,
          radiusPixels: 5.5,
          color: colors.lookout,
          outlineColor: colors.outline,
          outlineWidthPixels: 1.5
        })
      );
    }
    layers.push(
      new SpatialAnalysisPointLayer({
        id: 'viewshed-observer-halo',
        coordinateOrigin: origin,
        positions: analysis.observerMarker,
        instanceCount: 1,
        radiusPixels: 12,
        color: haloColor
      }),
      new SpatialAnalysisPointLayer({
        id: 'viewshed-observer',
        coordinateOrigin: origin,
        positions: analysis.observerMarker,
        instanceCount: 1,
        radiusPixels: 7,
        color: colors.eye,
        outlineColor: colors.outline,
        outlineWidthPixels: 2
      })
    );
    return layers;
  }

  const pyramidTables = new Map<string, ClassTable>();
  function getPyramidTable(tone: 'light' | 'dark'): ClassTable {
    let table = pyramidTables.get(tone);
    if (!table) {
      table = makePyramidTable(tone);
      pyramidTables.set(tone, table);
    }
    return table;
  }

  // --- Setup -------------------------------------------------------------------------------
  ctx.setReadout('hiddenPeaks', null);
  resolveTarget();
  publishContextAnnotations();
  publishResolutionChart();

  function describeGrid(): void {
    const {terrain} = analysis;
    ctx.setReadout(
      'grid',
      `${formatCount(terrain.width)} × ${formatCount(terrain.height)} cells · ${terrain.groundCellSize.toFixed(1)} m ground${terrain.stride > 1 ? ` (${terrain.stride} × ${terrain.stride} box average)` : ''}`
    );
  }
  describeGrid();
  ensureGraphs();
  writeParameters();
  updateFurniture();
  refreshAnnotations();
  ctx.setStatus('');

  /** Compiles the graphs the current options encode, so no frame compiles anything. */
  function ensureGraphs(): void {
    const options = ctx.options;
    getMainGraph(options.traversal);
    if (options.display === 'cumulative') getCumulativeGraph(options.traversal);
    if (options.display === 'pyramid') getPyramid();
    if (options.earthModel !== 'curved') getCompareGraph(options.traversal);
  }

  function rebuildAnalysis(stride: number): void {
    if (stride === analysis.stride) return;
    const previous = analysis;
    for (const reader of previous.readers.values()) reader.stop();
    analysis = buildAnalysis(stride);
    retired.push(previous);
    latestPeakCodes = null;
    unseenGap = null;
    resolveTarget();
    describeGrid();
    ensureGraphs();
    lastFurnitureKey = '';
    writeParameters();
    updateFurniture();
    refreshAnnotations();
    ctx.requestLayers();
    // The previous grid's buffers may still be bound by the layers of the frame being drawn.
    setTimeout(() => {
      const index = retired.indexOf(previous);
      if (index >= 0) {
        retired.splice(index, 1);
        previous.destroy();
      }
    }, 800);
  }

  const getActiveGraphs = (): CompiledGPUCommandGraph<void>[] => [...analysis.graphs.values()];

  return {
    getCompiledGraphs: () => {
      const graphs = getActiveGraphs();
      if (analysis.pyramidCompiled) graphs.push(analysis.pyramidCompiled);
      return graphs;
    },

    setOption(id, value) {
      const options = ctx.options;
      if (id === 'cellSize') {
        rebuildAnalysis(Number(value));
        return;
      }
      if (id === 'traversal') {
        // Compile (or reuse) the chosen traversal's graphs and rerun them.
        getMainGraph(options.traversal);
        if (options.display === 'cumulative') getCumulativeGraph(options.traversal);
        if (options.earthModel !== 'curved') getCompareGraph(options.traversal);
        mainDirty = true;
        cumulativeDirty = true;
        compareDirty = true;
        ctx.requestLayers();
        return;
      }
      if (id === 'pyramidBlockSize') {
        if (analysis.pyramid) {
          getPyramid();
          mainDirty = true;
          cumulativeDirty = true;
        }
        ctx.requestLayers();
        return;
      }
      if (id === 'display') {
        if (value === 'pyramid') getPyramid();
        if (value === 'cumulative') getCumulativeGraph(options.traversal);
        cumulativeDirty = true;
        lastFurnitureKey = '';
        updateFurniture();
        refreshAnnotations();
        ctx.requestLayers();
        return;
      }
      if (id === 'earthModel') {
        if (value !== 'curved') getCompareGraph(options.traversal);
        if (value === 'curved') {
          ctx.setReadout('flippedCells', null);
        }
      }
      if (id === 'rayTarget') {
        resolveTarget();
        if (value !== 'custom') {
          ctx.flyTo(
            {
              longitude: (observerLngLat[0] + targetLngLat[0]) / 2,
              latitude: (observerLngLat[1] + targetLngLat[1]) / 2,
              zoom: SIGHT_LINE_ZOOM
            },
            {transitionMs: 1000}
          );
        }
        writeParameters();
        refreshAnnotations();
        return;
      }
      if (id === 'rayPosition') {
        refreshRay();
        return;
      }
      if (id === 'pyramidLevel' || id === 'markVisible' || id === 'showSightLine') {
        if (id === 'showSightLine') {
          if (value) refreshRay();
          else {
            ctx.setChart('profile', null);
            ctx.setReadout('rayLength', null);
          }
          refreshAnnotations();
        }
        ctx.requestLayers();
        return;
      }
      writeParameters();
      lastFurnitureKey = '';
      updateFurniture();
      refreshAnnotations();
      ctx.requestLayers();
    },

    onAction(id) {
      if (id === 'verify') void verify();
      if (id === 'measure') void measure();
    },

    onGroundChange(tone) {
      ground.setGround(tone);
      ctx.setLegendData('ground', tone);
      ctx.requestLayers();
    },

    onThemeChange() {
      ground.setGround(ctx.ground());
      ctx.setLegendData('ground', ctx.ground());
      ctx.requestLayers();
    },

    encode(commandEncoder) {
      const options = ctx.options;
      const traversal = options.traversal;
      if (mainDirty) {
        getMainGraph(traversal).encode(commandEncoder, {parameters: undefined});
        getMainReader(traversal).request(commandEncoder);
        mainDirty = false;
      }
      if (options.display === 'cumulative' && cumulativeDirty) {
        getCumulativeGraph(traversal).encode(commandEncoder, {parameters: undefined});
        getCumulativeReader(traversal).request(commandEncoder);
        cumulativeDirty = false;
      }
      if (options.earthModel !== 'curved' && compareDirty && analysis.compare) {
        getCompareGraph(traversal).encode(commandEncoder, {parameters: undefined});
        getCompareReader().request(commandEncoder);
        compareDirty = false;
      }
      getMainReader(traversal).flush(commandEncoder);
      if (options.display === 'cumulative') getCumulativeReader(traversal).flush(commandEncoder);
      if (options.earthModel !== 'curved' && analysis.compare) {
        getCompareReader().flush(commandEncoder);
      }
    },

    getLayers() {
      return [ground.getLayer(), ...getAnalysisLayers(), ...getMarkerLayers()];
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      return getCellTooltip(event.coordinate);
    },

    onClick(event) {
      if (!event.coordinate || !ctx.options.showSightLine) return false;
      movePoint('target', event);
      return true;
    },

    onDragStart(event) {
      const observerDistance = getPointerDistance(observerLngLat, event);
      const targetDistance = ctx.options.showSightLine
        ? getPointerDistance(targetLngLat, event)
        : Number.POSITIVE_INFINITY;
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
      ctx.setAnnotations('observer', null);
      ctx.setAnnotations('peaks', null);
      ctx.setAnnotations('context', null);
      ctx.setAnnotations('frame', null);
      ctx.setAnnotations('markers', null);
      ctx.setAnnotations('notes', null);
      analysis.destroy();
      for (const previous of retired) previous.destroy();
      retired.length = 0;
      ground.destroy();
    }
  };
}
