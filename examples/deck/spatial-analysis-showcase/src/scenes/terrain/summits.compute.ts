// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import type {CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPU_TERRAIN_CRITICAL_POINT,
  GPU_TERRAIN_CRITICAL_POINT_CLASS_COUNT,
  GPU_TERRAIN_PEAK_SNAP_PARAMETER_LENGTH,
  GPU_TERRAIN_SUMMITS_PARAMETER_LENGTH,
  getGPUTerrainPeakSnapParameterValues,
  getGPUTerrainSummitsParameterValues
} from '@luma.gl/experimental/gpu-terrain';
import {formatCount, liveText} from '../../cartography/live-text';
import type {LngLat, MapAnnotation} from '../../cartography/types';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisPolygonLayer,
  SpatialAnalysisSegmentLayer
} from '../../engine/layers';
import {createSeededRandom} from '../../engine/projection';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {formatCompiledGraphTiming, measureCompiledGraph} from '../../engine/vector-timing';
import type {SceneContext, SceneInstance} from '../scene';
import {prepareAlpsTerrain, toFloat32, toUint32} from './b14b-terrain';
import {createDemProbe, createTerrainDemFromTerrain} from './cpu-dem';
import {
  applyDropThreshold,
  buildCatalogue,
  getDropClass,
  getDropHistogram,
  matchNamedSurvivors,
  parseSummitList,
  summarizeSnap,
  type CataloguePeak,
  type SummitCandidate,
  type SummitList
} from './summits-analysis';
import {
  CANDIDATE_CAPACITY,
  CONTOUR_LEVEL_COUNT,
  CONTOUR_SEGMENT_CAPACITY,
  compileContourGraph,
  compileCriticalGraph,
  compileSnapGraph,
  compileSummitGraph,
  createContourBuffers,
  createSnapBuffers,
  createSummitBuffers,
  type SummitBuffers
} from './summits-graphs';
import {
  CRITICAL_FADE_ZOOM,
  CRITICAL_MINIMUM_ZOOM,
  RADIUS_LADDER_METERS,
  SUMMIT_LIST_CAPACITY,
  formatMeters,
  getContourClass,
  getSnapPalette,
  getSnapStatusId,
  getSummitColors,
  type SummitsOptions
} from './summits-style';
import {
  getElevationTooltip,
  getProbeTooltip,
  getRejectedTooltip,
  getSnapTooltip,
  getSummitTooltip
} from './summits-tooltips';
import {demSampleLine} from './terrain-furniture';
import {createTerrainGround, rasterizeGlacierMask} from './terrain-ground';
import {getSnapStatusColor, SUMMIT_DROP_CLASSES} from './terrain-palettes';
import {loadAlpsContext, snapLngLatToHighestCell} from './terrain-places';

export type {SummitsOptions} from './summits-style';

/** The analysis grid of summits, snapping and critical points: the DEM averaged 4 x 4. */
const COARSE_STRIDE = 4;
/** Critical points drawn per class (the tile has far fewer). */
const CRITICAL_CAPACITY = 65536;
/** Bin width and count of the drop histogram, metres. */
const HISTOGRAM_BIN_METERS = 25;
const HISTOGRAM_BIN_COUNT = 20;
/** Names on the map per step: the largest drops that match a catalogue peak. */
const MAXIMUM_NAMED_SUMMITS = 5;
/** Labels of the snap step: the problem cases first, then the highest peaks. */
const MAXIMUM_SNAP_LABELS = 6;
/** Pointer reach for summits and catalogue points, CSS pixels. */
const HOVER_REACH_PIXELS = 12;
/** Radius of the rejected-candidate and critical-point glyphs, CSS pixels. */
const GHOST_RADIUS_PIXELS = 3.5;

/** One compiled summit run: a bound and a window-edge mode, with its own outputs. */
type SummitVariant = {
  key: string;
  mode: 'reject' | 'ignore';
  compiled: CompiledGPUCommandGraph<void>;
  buffers: SummitBuffers;
  reader: SummaryReader;
  dirty: boolean;
  list: SummitList | null;
};

/**
 * Summits, peak snapping, critical points and contours of the Matterhorn DEM.
 *
 * Summit-scale analyses run on the DEM averaged 4 x 4 (26.6 m ground cells); the contours and the
 * relief ground use the full 2048 x 2048 grid. The summit kernel always runs with a minimum drop
 * of 0: the GPU computes every disc maximum and its drop, and the scene applies the drop
 * threshold on the read-back list, so a moving drop slider is a CPU filter and the rejected
 * candidates are known.
 */
export async function createSummits(
  ctx: SceneContext<SummitsOptions>
): Promise<SceneInstance<SummitsOptions>> {
  const {device} = ctx;
  ctx.setStatus('Loading the DEM and its OpenStreetMap context');
  const dataset = ctx.datasets.get('alps-dem');
  const context = await loadAlpsContext(ctx.datasets.get('alps-context'), ctx.signal);
  ctx.signal.throwIfAborted();
  const full = prepareAlpsTerrain(dataset, 1);
  const coarse = prepareAlpsTerrain(dataset, COARSE_STRIDE);
  const fullDem = createTerrainDemFromTerrain(full);
  const probe = createDemProbe(createTerrainDemFromTerrain(coarse));
  const catalogue = buildCatalogue(context, coarse);
  const glacierMask = rasterizeGlacierMask(context.glacierGeoJson, fullDem);
  const ground = createTerrainGround({dem: fullDem, device, ground: ctx.ground(), glacierMask});
  ctx.setStatus('Building the shaded relief');
  await ground.prepare();
  if (ctx.signal.aborted) {
    ground.destroy();
    ctx.signal.throwIfAborted();
  }

  const {width, height, pixelCount} = coarse;
  const origin: [number, number, number] = [full.origin[0], full.origin[1], 0];
  const cellMeters = coarse.groundCellSize;
  const resources = new SpatialAnalysisResources(device, 'summits');
  let destroyed = false;

  // --- Buffers ----------------------------------------------------------------------------------
  const fullElevation = resources.createBuffer('full-elevation', full.elevation);
  const fullValidity = resources.createBuffer('full-validity', full.validity);
  const coarseElevation = resources.createBuffer('coarse-elevation', coarse.elevation);
  const coarseValidity = resources.createBuffer('coarse-validity', coarse.validity);
  const summitSettings = resources.createParameterBuffer(
    'summit-settings',
    'float32',
    GPU_TERRAIN_SUMMITS_PARAMETER_LENGTH
  );
  const sweepSettings = resources.createParameterBuffer(
    'sweep-settings',
    'float32',
    GPU_TERRAIN_SUMMITS_PARAMETER_LENGTH
  );
  const snapSettings = resources.createParameterBuffer(
    'snap-settings',
    'float32',
    GPU_TERRAIN_PEAK_SNAP_PARAMETER_LENGTH
  );
  const sweepBuffers = createSummitBuffers(resources, 'sweep', 16);
  const snapBuffers = createSnapBuffers(resources);
  const contourBuffers = createContourBuffers(resources);

  // Glyph positions, planar metres, written from the read-back lists.
  const summitClassBuffers = Array.from(
    {length: SUMMIT_DROP_CLASSES.sizesPixels.length},
    (_, level) => resources.createBuffer(`summit-class-${level}`, SUMMIT_LIST_CAPACITY * 8)
  );
  const ghostBuffer = resources.createBuffer('rejected', SUMMIT_LIST_CAPACITY * 8);
  const originalMarkers = resources.createBuffer('original-markers', CANDIDATE_CAPACITY * 8);
  const snappedMarkers = resources.createBuffer('snapped-markers', CANDIDATE_CAPACITY * 8);
  const snapSegments = resources.createBuffer('snap-segments', CANDIDATE_CAPACITY * 16);
  const criticalClasses = resources.createBuffer('critical-classes', pixelCount * 4);
  const criticalSigns = resources.createBuffer('critical-signs', pixelCount * 4);
  const criticalCounts = resources.createBuffer(
    'critical-counts',
    GPU_TERRAIN_CRITICAL_POINT_CLASS_COUNT * 4
  );
  const criticalPeakBuffer = resources.createBuffer('critical-peaks', CRITICAL_CAPACITY * 8);
  const criticalSaddleBuffer = resources.createBuffer('critical-saddles', CRITICAL_CAPACITY * 8);
  const criticalPitBuffer = resources.createBuffer('critical-pits', CRITICAL_CAPACITY * 8);
  // The incomplete-disc band: four rectangles (24 vertices), one hatched class.
  const bandTriangles = resources.createBuffer('edge-band-triangles', 24 * 8);
  const bandFeatures = resources.createBuffer('edge-band-features', new Uint32Array(24));
  const bandValues = resources.createBuffer('edge-band-values', new Uint32Array(1));

  // --- Compiled graphs --------------------------------------------------------------------------
  const getBound = (): number => Number(ctx.options.summitMaximumRadius);
  const variants = new Map<string, SummitVariant>();
  const sweepGraphs = new Map<string, CompiledGPUCommandGraph<void>>();
  let lastList: SummitList | null = null;

  function getVariant(mode: 'reject' | 'ignore'): SummitVariant {
    const bound = getBound();
    const key = `${bound}|${mode}`;
    const existing = variants.get(key);
    if (existing) return existing;
    const buffers = createSummitBuffers(resources, `variant-${key}`, SUMMIT_LIST_CAPACITY);
    const compiled = compileSummitGraph(resources, {
      name: `summits-${mode}-${bound}`,
      width,
      height,
      pixelCount,
      elevation: coarseElevation,
      validity: coarseValidity,
      maximumRadiusPixels: bound,
      incompleteNeighborhood: mode,
      settings: summitSettings,
      buffers,
      capacity: SUMMIT_LIST_CAPACITY
    });
    const variant: SummitVariant = {
      key,
      mode,
      compiled,
      buffers,
      dirty: true,
      list: null,
      reader: new SummaryReader(
        resources,
        `summits-${key}`,
        [
          {buffer: buffers.count, size: 4},
          {buffer: buffers.total, size: 4},
          {buffer: buffers.overflow, size: 4},
          {buffer: buffers.clamped, size: 4},
          {buffer: buffers.ids, size: SUMMIT_LIST_CAPACITY * 4},
          {buffer: buffers.drops, size: SUMMIT_LIST_CAPACITY * 4}
        ],
        bytes => applySummits(variant, bytes)
      )
    };
    variants.set(key, variant);
    return variant;
  }

  /** The variants a frame runs: the chosen one, plus `ignore` when `reject` needs a comparison. */
  function getRunningVariants(): SummitVariant[] {
    const chosen = getVariant(ctx.options.incompleteNeighborhood);
    return chosen.mode === 'reject' ? [chosen, getVariant('ignore')] : [chosen];
  }

  function getSweepGraph(): CompiledGPUCommandGraph<void> {
    const mode = ctx.options.incompleteNeighborhood;
    const key = `${getBound()}|${mode}`;
    let graph = sweepGraphs.get(key);
    if (!graph) {
      graph = compileSummitGraph(resources, {
        name: `summits-sweep-${key}`,
        width,
        height,
        pixelCount,
        elevation: coarseElevation,
        validity: coarseValidity,
        maximumRadiusPixels: getBound(),
        incompleteNeighborhood: mode,
        settings: sweepSettings,
        buffers: sweepBuffers,
        capacity: 16
      });
      sweepGraphs.set(key, graph);
    }
    return graph;
  }

  const snapCompiled = compileSnapGraph(resources, {
    width,
    height,
    pixelCount,
    elevation: coarseElevation,
    validity: coarseValidity,
    settings: snapSettings,
    buffers: snapBuffers
  });
  let criticalGraph: {key: string; compiled: CompiledGPUCommandGraph<void>} | null = null;
  const contourCompiled = compileContourGraph(resources, {
    width: full.width,
    height: full.height,
    pixelCount: full.pixelCount,
    elevation: fullElevation,
    validity: fullValidity,
    buffers: contourBuffers
  });

  // --- State ------------------------------------------------------------------------------------
  let snapDirty = true;
  let criticalDirty = true;
  let contourDirty = true;
  let contourReady = false;
  let contourCountsRead = new Uint32Array(CONTOUR_LEVEL_COUNT);
  let survivors: SummitCandidate[] = [];
  let rejected: SummitCandidate[] = [];
  let survivorNames = new Map<number, CataloguePeak>();
  let rejectedNames = new Map<number, CataloguePeak>();
  let classCounts: number[] = new Array(SUMMIT_DROP_CLASSES.sizesPixels.length).fill(0);
  let snapRows: {status: number; distance: number; height: number}[] = [];
  let snapCountsLast: ReturnType<typeof summarizeSnap> | null = null;
  const criticalCountsRead = {peaks: 0, saddles: 0, pits: 0};
  let sweepVersion = 0;
  let sweepTimer: ReturnType<typeof setTimeout> | null = null;

  // Catalogue points: the real OSM nodes plus a seeded direction of simulated error each.
  const random = createSeededRandom(7);
  const errorAngles = catalogue.map(() => random() * Math.PI * 2);
  const observer = coarse.observers.find(place => place.name === 'Gornergrat');
  const observerPixel: [number, number] = observer
    ? coarse.getPixel(observer.longitude, observer.latitude)
    : [width / 2, height / 2];

  const getCandidatePixel = (index: number): [number, number] => {
    const errorPixels = ctx.options.catalogueError / cellMeters;
    const peak = catalogue[index];
    return [
      peak.column + Math.cos(errorAngles[index]) * errorPixels,
      peak.row + Math.sin(errorAngles[index]) * errorPixels
    ];
  };

  // --- Parameter writes -------------------------------------------------------------------------
  function writeSummitSettings(): void {
    // Minimum drop 0: the drop test is applied on the read-back list (see the module comment).
    summitSettings.write(
      getGPUTerrainSummitsParameterValues({
        radius: ctx.options.summitRadius,
        minimumDrop: 0,
        ...coarse.mercatorCellSettings
      })
    );
    for (const variant of variants.values()) variant.dirty = true;
    writeEdgeBand();
  }

  function writeEdgeBand(): void {
    const [minimumX, minimumY, maximumX, maximumY] = full.bounds;
    const inset = Math.min(
      ctx.options.summitRadius,
      (maximumX - minimumX) / 2,
      (maximumY - minimumY) / 2
    );
    const rectangles: [number, number, number, number][] = [
      [minimumX, minimumY, minimumX + inset, maximumY],
      [maximumX - inset, minimumY, maximumX, maximumY],
      [minimumX + inset, minimumY, maximumX - inset, minimumY + inset],
      [minimumX + inset, maximumY - inset, maximumX - inset, maximumY]
    ];
    const triangles = new Float32Array(rectangles.length * 12);
    rectangles.forEach(([x0, y0, x1, y1], index) => {
      triangles.set([x0, y0, x1, y0, x1, y1, x0, y0, x1, y1, x0, y1], index * 12);
    });
    bandTriangles.write(triangles);
  }

  function writeCandidates(): void {
    const options = ctx.options;
    const positions = new Float32Array(CANDIDATE_CAPACITY * 2).fill(-5);
    const heights = new Float32Array(CANDIDATE_CAPACITY).fill(Number.NaN);
    const radii = new Float32Array(CANDIDATE_CAPACITY).fill(Number.NaN);
    const original = new Float32Array(CANDIDATE_CAPACITY * 2).fill(1e7);
    catalogue.forEach((peak, index) => {
      const [column, row] = getCandidatePixel(index);
      positions[index * 2] = column;
      positions[index * 2 + 1] = row;
      if (options.snapCatalogueHeights && peak.elevationMeters !== null) {
        heights[index] = peak.elevationMeters;
      }
      if (options.snapDistanceRule) {
        // mt-image: min(250 m, 60 m + 0.4 % of the distance to the viewer).
        const distance = Math.hypot(column - observerPixel[0], row - observerPixel[1]) * cellMeters;
        radii[index] = Math.min(250, 60 + 0.004 * distance);
      }
      const [x, y] = coarse.getMeters(column, row);
      original[index * 2] = x;
      original[index * 2 + 1] = y;
    });
    snapBuffers.candidates.write(positions);
    snapBuffers.candidateHeights.write(heights);
    snapBuffers.candidateRadii.write(radii);
    originalMarkers.write(original);
    snapDirty = true;
  }

  function writeSnapSettings(): void {
    const options = ctx.options;
    snapSettings.write(
      getGPUTerrainPeakSnapParameterValues({
        radius: options.snapRadius,
        maximumMove: options.snapMaximumMove,
        maximumHeightChange: options.snapMaximumHeightChange,
        interior: options.snapInterior,
        ...coarse.mercatorCellSettings
      })
    );
    snapDirty = true;
  }

  function getContourLevels(): number[] {
    const interval = ctx.options.contourInterval;
    // Start at the DEM minimum rounded down to the interval: every slot from there up is a level.
    const first = Math.floor(full.elevationRange[0] / interval) * interval;
    return Array.from({length: CONTOUR_LEVEL_COUNT}, (_, level) => first + level * interval);
  }

  function writeContourLevels(): void {
    contourBuffers.levels.write(Float32Array.from(getContourLevels()));
    contourDirty = true;
  }

  // --- Furniture --------------------------------------------------------------------------------
  function publishFurniture(): void {
    const options = ctx.options;
    let subtitle = `Highest within ${options.summitRadius} m, at least ${options.summitMinimumDrop} m above its ring · ${cellMeters.toFixed(1)} m analysis grid`;
    let ticks = [options.summitRadius];
    if (options.showSnap) {
      subtitle = `Catalogue peaks snapped within ${options.snapRadius} m · ${cellMeters.toFixed(1)} m analysis grid`;
      ticks = [options.snapRadius];
    } else if (options.showContours || options.showCritical) {
      subtitle = `Contours every ${options.contourInterval} m · critical points on the ${cellMeters.toFixed(1)} m grid`;
      ticks = [];
    }
    ctx.setFurniture({
      title: {subtitle, sample: demSampleLine(full)},
      scaleBar: {units: 'metric', ticks}
    });
    ctx.setReadout('radiusNow', options.summitRadius);
    ctx.setReadout('dropNow', options.summitMinimumDrop);
  }

  // --- Summit lists -----------------------------------------------------------------------------
  function applySummits(variant: SummitVariant, bytes: ArrayBuffer): void {
    variant.list = parseSummitList(bytes, SUMMIT_LIST_CAPACITY, width, coarse.elevation);
    const running = getRunningVariants();
    if (running.includes(variant)) refreshSummits();
  }

  /** Applies the drop threshold to the chosen run's list and rewrites everything drawn from it. */
  function refreshSummits(): void {
    const options = ctx.options;
    const chosen = getVariant(options.incompleteNeighborhood);
    const list = chosen.list ?? lastList;
    if (!list) return;
    lastList = list;
    const split = applyDropThreshold(list.candidates, options.summitMinimumDrop);
    survivors = split.survivors;
    rejected = split.rejected;
    survivorNames = matchNamedSurvivors(survivors, catalogue, cellMeters);
    rejectedNames = matchNamedSurvivors(rejected, catalogue, cellMeters);

    // Triangles by drop class, one point layer each (one size per layer).
    const byClass: number[][] = summitClassBuffers.map(() => []);
    classCounts = classCounts.map(() => 0);
    for (const candidate of survivors) {
      const level = getDropClass(candidate.drop);
      const [x, y] = coarse.getMeters(candidate.column, candidate.row);
      byClass[level].push(x, y);
      classCounts[level]++;
    }
    byClass.forEach((positions, level) => {
      if (positions.length) summitClassBuffers[level].write(Float32Array.from(positions));
    });
    const ghostPositions: number[] = [];
    for (const candidate of rejected) {
      const [x, y] = coarse.getMeters(candidate.column, candidate.row);
      ghostPositions.push(x, y);
    }
    if (ghostPositions.length) ghostBuffer.write(Float32Array.from(ghostPositions));

    ctx.setReadout('summitCount', survivors.length);
    ctx.setReadout('rejectedCount', rejected.length);
    ctx.setReadout(
      'listUse',
      list.listOverflow
        ? `capped: ${formatCount(SUMMIT_LIST_CAPACITY)} of ${formatCount(list.total)} candidates listed`
        : `${formatCount(list.total)} of ${formatCount(SUMMIT_LIST_CAPACITY)} places`
    );
    ctx.setReadout(
      'radiusClamp',
      list.radiusClamped
        ? `clamped to ${formatMeters(getBound() * cellMeters)} m by the search bound`
        : 'as requested'
    );
    ctx.setReadout('edgeLost', getEdgeLoss());
    ctx.setChart('dropHistogram', {
      kind: 'histogram',
      values: getDropHistogram(list.candidates, HISTOGRAM_BIN_METERS, HISTOGRAM_BIN_COUNT),
      xDomain: [0, HISTOGRAM_BIN_METERS * HISTOGRAM_BIN_COUNT],
      xLabel: 'Drop to the ring (m); the last bin holds all larger drops',
      yLabel: 'Disc maxima',
      height: 130,
      link: {option: 'summitMinimumDrop', label: value => `at least ${value} m`},
      description:
        'Histogram of the drop of every disc maximum the GPU found, with the minimum drop marked: candidates left of the mark fail the test.'
    });
    publishAnnotations();
    ctx.requestLayers();
  }

  /** Survivors that a window-edge Reject would remove, read off the data. */
  function getEdgeLoss(): number | null {
    const options = ctx.options;
    const ignoreList = getVariant('ignore').list;
    const minimumDrop = options.summitMinimumDrop;
    if (options.incompleteNeighborhood === 'reject') {
      const rejectList = getVariant('reject').list;
      if (!ignoreList || !rejectList) return null;
      return (
        applyDropThreshold(ignoreList.candidates, minimumDrop).survivors.length -
        applyDropThreshold(rejectList.candidates, minimumDrop).survivors.length
      );
    }
    if (!ignoreList) return null;
    const reach = Math.ceil(options.summitRadius / (cellMeters * 0.98)) + 1;
    let lost = 0;
    for (const candidate of applyDropThreshold(ignoreList.candidates, minimumDrop).survivors) {
      const nearEdge =
        candidate.column <= reach ||
        candidate.row <= reach ||
        candidate.column >= width - 1 - reach ||
        candidate.row >= height - 1 - reach;
      if (
        nearEdge &&
        probe.discAndRing(candidate.column, candidate.row, options.summitRadius, 'ignore')
          .incomplete
      ) {
        lost++;
      }
    }
    return lost;
  }

  // --- The count-by-radius curve ----------------------------------------------------------------
  function scheduleSweep(): void {
    if (sweepTimer) clearTimeout(sweepTimer);
    sweepTimer = setTimeout(() => void runSweep(), 200);
  }

  /**
   * Runs the summit kernel at each radius of the ladder as a parameter write, outside the frame,
   * with its own settings and outputs so the frame's results are never overwritten.
   */
  async function runSweep(): Promise<void> {
    const version = ++sweepVersion;
    const options = ctx.options;
    const graph = getSweepGraph();
    const counts: number[] = [];
    try {
      for (const radius of RADIUS_LADDER_METERS) {
        sweepSettings.write(
          getGPUTerrainSummitsParameterValues({
            radius,
            minimumDrop: options.summitMinimumDrop,
            ...coarse.mercatorCellSettings
          })
        );
        const commandEncoder = device.createCommandEncoder({id: 'summits-sweep-encoder'});
        graph.encode(commandEncoder, {parameters: undefined});
        device.submit(commandEncoder.finish());
        const [totalBytes, clampedBytes] = await Promise.all([
          sweepBuffers.total.readAsync(0, 4),
          sweepBuffers.clamped.readAsync(0, 4)
        ]);
        if (destroyed || version !== sweepVersion) return;
        counts.push(toUint32(clampedBytes, 1)[0] ? Number.NaN : toUint32(totalBytes, 1)[0]);
      }
    } catch {
      return;
    }
    ctx.setChart('countByRadius', {
      kind: 'line',
      series: [{label: 'Summits', x: RADIUS_LADDER_METERS, y: counts, points: true}],
      xLabel: 'Radius (m)',
      yLabel: 'Summits',
      xDomain: [RADIUS_LADDER_METERS[0], RADIUS_LADDER_METERS[RADIUS_LADDER_METERS.length - 1]],
      height: 130,
      link: {option: 'summitRadius', label: value => `r = ${value} m`},
      description:
        'Number of summits found at six radii with the current minimum drop: the count falls steeply as the radius grows. A radius beyond the search bound is left out.'
    });
  }

  // --- Peak snap --------------------------------------------------------------------------------
  const snapReader = new SummaryReader(
    resources,
    'snap',
    [
      {buffer: snapBuffers.positions, size: CANDIDATE_CAPACITY * 8},
      {buffer: snapBuffers.heights, size: CANDIDATE_CAPACITY * 4},
      {buffer: snapBuffers.status, size: CANDIDATE_CAPACITY * 4},
      {buffer: snapBuffers.distance, size: CANDIDATE_CAPACITY * 4},
      {buffer: snapBuffers.overflow, size: 4}
    ],
    bytes => applySnap(bytes)
  );
  let snappedPixels: [number, number][] = [];

  function applySnap(bytes: ArrayBuffer): void {
    const positions = toFloat32(bytes, CANDIDATE_CAPACITY * 2);
    const heights = toFloat32(bytes.slice(CANDIDATE_CAPACITY * 8), CANDIDATE_CAPACITY);
    const status = toUint32(bytes.slice(CANDIDATE_CAPACITY * 12), CANDIDATE_CAPACITY);
    const distance = toFloat32(bytes.slice(CANDIDATE_CAPACITY * 16), CANDIDATE_CAPACITY);
    snapRows = [];
    snappedPixels = [];
    const snapped = new Float32Array(CANDIDATE_CAPACITY * 2).fill(1e7);
    const segments = new Float32Array(CANDIDATE_CAPACITY * 4).fill(1e7);
    catalogue.forEach((_, index) => {
      snapRows.push({status: status[index], distance: distance[index], height: heights[index]});
      const column = positions[index * 2];
      const row = positions[index * 2 + 1];
      snappedPixels.push([column, row]);
      const [x, y] = coarse.getMeters(column, row);
      snapped[index * 2] = x;
      snapped[index * 2 + 1] = y;
      const [originalColumn, originalRow] = getCandidatePixel(index);
      const [originalX, originalY] = coarse.getMeters(originalColumn, originalRow);
      segments.set([originalX, originalY, x, y], index * 4);
    });
    snappedMarkers.write(snapped);
    snapSegments.write(segments);
    snapCountsLast = summarizeSnap(status, distance, catalogue.length);
    ctx.setLegendData('snapCounts', snapCountsLast.counts);
    ctx.setReadout('medianMove', snapCountsLast.medianMoveMeters);
    ctx.setReadout('catalogueSize', catalogue.length);
    publishSnapChart();
    publishAnnotations();
    ctx.requestLayers();
  }

  function publishSnapChart(): void {
    if (!snapCountsLast) return;
    const tone = ctx.ground();
    const segmentsList = (
      [
        ['snapped', 'Snapped'],
        ['unchanged', 'Already on the summit'],
        ['flank', 'On a flank'],
        ['too-far', 'Too far'],
        ['height-mismatch', 'Height mismatch']
      ] as const
    )
      .filter(([id]) => snapCountsLast!.counts[id] > 0)
      .map(([id, label]) => ({
        label,
        value: snapCountsLast!.counts[id],
        color: getSnapStatusColor(id, tone)
      }));
    ctx.setChart('snapOutcomes', {
      kind: 'stacked',
      format: 'value',
      segments: segmentsList,
      description: 'How many catalogue peaks ended in each snap outcome, in the map colours.'
    });
  }

  // --- Critical points --------------------------------------------------------------------------
  const criticalReader = new SummaryReader(
    resources,
    'critical',
    [
      {buffer: criticalClasses, size: pixelCount * 4},
      {buffer: criticalSigns, size: pixelCount * 4}
    ],
    bytes => applyCritical(bytes)
  );

  function applyCritical(bytes: ArrayBuffer): void {
    const classes = toUint32(bytes, pixelCount);
    const signs = toUint32(bytes.slice(pixelCount * 4), pixelCount);
    const peakPositions: number[] = [];
    const saddlePositions: number[] = [];
    const pitPositions: number[] = [];
    let multiplicity = 0;
    let boundary = 0;
    for (let index = 0; index < pixelCount; index++) {
      const code = classes[index];
      if (code === 0 || code === GPU_TERRAIN_CRITICAL_POINT.regular) continue;
      const [x, y] = coarse.getMeters(index % width, Math.floor(index / width));
      if (code === GPU_TERRAIN_CRITICAL_POINT.peak) peakPositions.push(x, y);
      else if (code === GPU_TERRAIN_CRITICAL_POINT.pit) pitPositions.push(x, y);
      else if (code === GPU_TERRAIN_CRITICAL_POINT.saddle) {
        saddlePositions.push(x, y);
        multiplicity += signs[index] / 2 - 1;
      } else if (code === GPU_TERRAIN_CRITICAL_POINT.boundary) boundary++;
    }
    const clip = (positions: number[]) =>
      Float32Array.from(positions.slice(0, CRITICAL_CAPACITY * 2));
    if (peakPositions.length) criticalPeakBuffer.write(clip(peakPositions));
    if (saddlePositions.length) criticalSaddleBuffer.write(clip(saddlePositions));
    if (pitPositions.length) criticalPitBuffer.write(clip(pitPositions));
    criticalCountsRead.peaks = peakPositions.length / 2;
    criticalCountsRead.saddles = saddlePositions.length / 2;
    criticalCountsRead.pits = pitPositions.length / 2;
    ctx.setReadout(
      'criticalCounts',
      `${formatCount(criticalCountsRead.peaks)} peaks, ${formatCount(criticalCountsRead.saddles)} saddles, ${formatCount(criticalCountsRead.pits)} pits`
    );
    ctx.setReadout(
      'euler',
      `peaks - saddles + pits = ${formatCount(criticalCountsRead.peaks - multiplicity + criticalCountsRead.pits)} (saddles counted with multiplicity; ${formatCount(boundary)} boundary cells unclassified)`
    );
    ctx.requestLayers();
  }

  function ensureCriticalGraph(): void {
    const connectivity = ctx.options.connectivity;
    if (criticalGraph?.key === connectivity) return;
    if (criticalGraph) resources.release(criticalGraph.compiled);
    criticalGraph = {
      key: connectivity,
      compiled: compileCriticalGraph(resources, {
        width,
        height,
        pixelCount,
        elevation: coarseElevation,
        validity: coarseValidity,
        connectivity: connectivity === '8' ? 8 : 6,
        classes: criticalClasses,
        signChanges: criticalSigns,
        counts: criticalCounts
      })
    };
    criticalDirty = true;
  }

  // --- Contours ---------------------------------------------------------------------------------
  const contourReader = new SummaryReader(
    resources,
    'contours',
    [
      {buffer: contourBuffers.overflow, size: 4},
      ...contourBuffers.counts.map(buffer => ({buffer, size: 4}))
    ],
    bytes => applyContours(bytes)
  );

  function applyContours(bytes: ArrayBuffer): void {
    const overflow = toUint32(bytes, 1)[0];
    contourCountsRead = toUint32(bytes.slice(4), CONTOUR_LEVEL_COUNT);
    let segments = 0;
    let levels = 0;
    for (const count of contourCountsRead) {
      segments += count;
      if (count > 0) levels++;
    }
    contourReady = true;
    ctx.setReadout(
      'contourSummary',
      `${levels} levels with data, ${formatCount(segments)} segments` +
        (overflow ? `; a level exceeded ${formatCount(CONTOUR_SEGMENT_CAPACITY)} segments` : '')
    );
    ctx.requestLayers();
  }

  // --- Annotations ------------------------------------------------------------------------------
  function getSummitLngLat(candidate: SummitCandidate): LngLat {
    return coarse.getLongitudeLatitude(candidate.column, candidate.row);
  }

  function publishAnnotations(): void {
    const options = ctx.options;
    const list: MapAnnotation[] = [];
    if (options.showSummits) {
      // The largest drops that carry a catalogue name, with the published elevation.
      const named = [...survivorNames]
        .map(([index, peak]) => ({candidate: survivors[index], peak}))
        .sort((a, b) => b.candidate.drop - a.candidate.drop)
        .slice(0, MAXIMUM_NAMED_SUMMITS);
      named.forEach(({candidate, peak}, rank) => {
        list.push({
          kind: 'landform',
          id: `summits:name:${peak.osmId}`,
          coordinate: getSummitLngLat(candidate),
          text: peak.name,
          marker: 'none',
          ...(peak.elevationMeters !== null ? {elevationMeters: peak.elevationMeters} : {}),
          priority: MAXIMUM_NAMED_SUMMITS - rank
        });
      });
    }
    if (options.showRejected && options.summitMinimumDrop > SUMMIT_DROP_CLASSES.minimumDropMeters) {
      // A named peak that passes the default 100 m test and fails this one: the closest miss.
      const misses = [...rejectedNames]
        .map(([index, peak]) => ({candidate: rejected[index], peak}))
        .filter(({candidate}) => candidate.drop >= SUMMIT_DROP_CLASSES.minimumDropMeters)
        .sort((a, b) => b.candidate.drop - a.candidate.drop);
      const miss = misses[0];
      if (miss) {
        list.push({
          kind: 'note',
          id: 'summits:miss',
          coordinate: getSummitLngLat(miss.candidate),
          title: liveText('{name}: {drop:integer} m drop', {
            name: miss.peak.name,
            drop: miss.candidate.drop
          }),
          text: liveText('Highest in its disc, but {needed:integer} m are needed', {
            needed: options.summitMinimumDrop
          }),
          anchor: 'se',
          distance: 50,
          priority: 4
        });
      }
    }
    if (options.showEdge) {
      list.push({
        kind: 'frame',
        id: 'summits:edge',
        bounds: full.lngLatBounds,
        text: 'Data ends here'
      });
    }
    if (options.showSnap && snapRows.length) {
      list.push(...getSnapLabels());
    }
    ctx.setAnnotations('summits', list);
  }

  /** Two problem cases first (a flank, a rejection), then the highest peaks, six labels in all. */
  function getSnapLabels(): MapAnnotation[] {
    const indices = catalogue.map((_, index) => index);
    const isProblem = (index: number) => {
      const id = getSnapStatusId(snapRows[index].status);
      return id === 'flank' || id === 'too-far' || id === 'height-mismatch';
    };
    const chosen = [
      ...indices.filter(isProblem).slice(0, 2),
      ...indices.filter(index => !isProblem(index))
    ].slice(0, MAXIMUM_SNAP_LABELS);
    return chosen.map((index, rank) => {
      const peak = catalogue[index];
      const [column, row] = snappedPixels[index] ?? getCandidatePixel(index);
      return {
        kind: 'landform' as const,
        id: `summits:snap:${peak.osmId}`,
        coordinate: coarse.getLongitudeLatitude(column, row),
        text: peak.name,
        marker: 'none' as const,
        ...(peak.elevationMeters !== null ? {elevationMeters: peak.elevationMeters} : {}),
        priority: MAXIMUM_SNAP_LABELS - rank
      };
    });
  }

  // --- Timing -----------------------------------------------------------------------------------
  async function measure(): Promise<void> {
    try {
      ctx.setReadout('timing', 'measuring...');
      const run = async (
        compiled: CompiledGPUCommandGraph<void>,
        buffer: Buffer,
        label: string
      ) => {
        const timing = await measureCompiledGraph(device, compiled, {
          parameters: undefined,
          completionBuffer: buffer,
          signal: ctx.signal,
          runs: 3,
          warmUpRuns: 1,
          repetitions: 2
        });
        return `${label} ${formatCompiledGraphTiming(timing).replace(/ · .*/, '')}`;
      };
      ensureCriticalGraph();
      const chosen = getVariant(ctx.options.incompleteNeighborhood);
      const parts = [
        await run(chosen.compiled, chosen.buffers.count, 'summits'),
        await run(criticalGraph!.compiled, criticalCounts, 'critical points'),
        await run(snapCompiled, snapBuffers.status, 'snap'),
        await run(contourCompiled, contourBuffers.overflow, `${CONTOUR_LEVEL_COUNT} contour levels`)
      ];
      ctx.setReadout('timing', parts.join(' · '));
    } catch (error) {
      if (!destroyed) ctx.setReadout('timing', `failed: ${(error as Error).message}`);
    }
  }

  // --- Pointer ----------------------------------------------------------------------------------
  /** The analysis-grid position under the pointer, or null off the DEM. */
  function getPointerPixel(event: {
    coordinate: readonly [number, number] | null;
  }): [number, number] | null {
    if (!event.coordinate) return null;
    const [x, y] = full.projection.project(event.coordinate[0], event.coordinate[1]);
    const [minimumX, minimumY, maximumX, maximumY] = full.bounds;
    if (x < minimumX || x > maximumX || y < minimumY || y > maximumY) return null;
    return coarse.getPixelFromMeters(x, y);
  }

  function findNearest(
    candidates: readonly SummitCandidate[],
    pixel: readonly [number, number]
  ): number {
    const reach = (HOVER_REACH_PIXELS * ctx.getMetersPerPixel()) / cellMeters;
    let best = -1;
    let bestDistance = reach;
    candidates.forEach((candidate, index) => {
      const distance = Math.hypot(candidate.column - pixel[0], candidate.row - pixel[1]);
      if (distance < bestDistance) {
        best = index;
        bestDistance = distance;
      }
    });
    return best;
  }

  // --- Setup ------------------------------------------------------------------------------------
  ctx.setLegendData('ground', ctx.ground());
  ctx.setReadout(
    'grid',
    `${width} × ${height} cells at ${cellMeters.toFixed(1)} m for summits; ${full.width} × ${full.height} at ${full.groundCellSize.toFixed(1)} m for contours and the relief`
  );
  writeSummitSettings();
  writeCandidates();
  writeSnapSettings();
  writeContourLevels();
  getRunningVariants();
  publishFurniture();
  scheduleSweep();

  return {
    getCompiledGraphs: () => {
      const graphs = [snapCompiled, contourCompiled, ...getRunningVariants().map(v => v.compiled)];
      if (criticalGraph) graphs.push(criticalGraph.compiled);
      return graphs;
    },

    setOption(id) {
      switch (id) {
        case 'summitRadius':
          writeSummitSettings();
          break;
        case 'summitMinimumDrop':
          refreshSummits();
          scheduleSweep();
          break;
        case 'summitMaximumRadius':
        case 'incompleteNeighborhood':
          getRunningVariants();
          writeSummitSettings();
          scheduleSweep();
          break;
        case 'catalogueError':
        case 'snapCatalogueHeights':
        case 'snapDistanceRule':
          writeCandidates();
          break;
        case 'snapRadius':
        case 'snapMaximumMove':
        case 'snapMaximumHeightChange':
        case 'snapInterior':
          writeSnapSettings();
          break;
        case 'connectivity':
          criticalDirty = true;
          break;
        case 'contourInterval':
          writeContourLevels();
          break;
        default:
          break;
      }
      publishFurniture();
      publishAnnotations();
      ctx.requestLayers();
    },

    onAction(id) {
      if (id === 'measure') void measure();
    },

    encode(commandEncoder) {
      const options = ctx.options;
      for (const variant of getRunningVariants()) {
        if (variant.dirty) {
          variant.compiled.encode(commandEncoder, {parameters: undefined});
          variant.reader.request(commandEncoder);
          variant.dirty = false;
        }
        variant.reader.flush(commandEncoder);
      }
      if (snapDirty && options.showSnap) {
        snapCompiled.encode(commandEncoder, {parameters: undefined});
        snapReader.request(commandEncoder);
        snapDirty = false;
      }
      snapReader.flush(commandEncoder);
      if (options.showCritical) {
        ensureCriticalGraph();
        if (criticalDirty && criticalGraph) {
          criticalGraph.compiled.encode(commandEncoder, {parameters: undefined});
          criticalReader.request(commandEncoder);
          criticalDirty = false;
        }
      }
      criticalReader.flush(commandEncoder);
      if (contourDirty && options.showContours) {
        contourCompiled.encode(commandEncoder, {parameters: undefined});
        contourReader.request(commandEncoder);
        contourDirty = false;
      }
      contourReader.flush(commandEncoder);
    },

    getLayers() {
      const options = ctx.options;
      const tone = ctx.ground();
      const colors = getSummitColors(tone);
      const layers: Layer[] = [ground.getLayer()];

      if (options.showContours && contourReady) {
        const levels = getContourLevels();
        for (let level = 0; level < CONTOUR_LEVEL_COUNT; level++) {
          if (contourCountsRead[level] === 0) continue;
          const isIndex =
            Math.round(levels[level] / options.contourInterval) % options.contourIndexEvery === 0;
          const style = getContourClass(tone, isIndex);
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: `summits-contour-${level}`,
              coordinateOrigin: origin,
              segments: contourBuffers.vertices[level],
              drawCommands: contourBuffers.drawCommands,
              drawCommandIndex: level,
              // Vertices are in pixel-edge grid units with row 0 at the north edge.
              positionScale: [full.layerCellSize[0], -full.layerCellSize[1]],
              positionOffset: [full.bounds[0], full.bounds[3]],
              widthPixels: style.widthPixels,
              color: style.color
            })
          );
        }
      }

      if (options.showEdge) {
        layers.push(
          new SpatialAnalysisPolygonLayer({
            id: 'summits-edge-band',
            coordinateOrigin: origin,
            triangles: bandTriangles,
            features: bandFeatures,
            vertexCount: 24,
            values: bandValues,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: [[0, 0, 0, 0]],
            hatchClasses: [0],
            hatchColor: colors.hatch,
            hatchSpacingPixels: 6,
            hatchWidthPixels: 1.2
          })
        );
      }

      if (options.showCritical) {
        const fade = [
          [CRITICAL_MINIMUM_ZOOM - CRITICAL_FADE_ZOOM, 0],
          [CRITICAL_MINIMUM_ZOOM + CRITICAL_FADE_ZOOM, 1]
        ] as const;
        const common = {coordinateOrigin: origin, opacityStops: fade};
        if (criticalCountsRead.pits > 0) {
          layers.push(
            new SpatialAnalysisPointLayer({
              ...common,
              id: 'summits-critical-pits',
              positions: criticalPitBuffer,
              instanceCount: Math.min(criticalCountsRead.pits, CRITICAL_CAPACITY),
              shape: 'ring',
              radiusPixels: 4,
              outlineWidthPixels: 1.5,
              color: colors.pit
            })
          );
        }
        if (criticalCountsRead.saddles > 0) {
          layers.push(
            new SpatialAnalysisPointLayer({
              ...common,
              id: 'summits-critical-saddles',
              positions: criticalSaddleBuffer,
              instanceCount: Math.min(criticalCountsRead.saddles, CRITICAL_CAPACITY),
              shape: 'diamond',
              radiusPixels: 4,
              outlineColor: colors.summit,
              outlineWidthPixels: 1,
              color: colors.saddle
            })
          );
        }
        if (criticalCountsRead.peaks > 0) {
          layers.push(
            new SpatialAnalysisPointLayer({
              ...common,
              id: 'summits-critical-peaks',
              positions: criticalPeakBuffer,
              instanceCount: Math.min(criticalCountsRead.peaks, CRITICAL_CAPACITY),
              shape: 'triangle',
              radiusPixels: 4,
              outlineColor: colors.paper,
              outlineWidthPixels: 1,
              color: colors.criticalPeak
            })
          );
        }
      }

      if (options.showRejected && rejected.length > 0) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'summits-rejected',
            coordinateOrigin: origin,
            positions: ghostBuffer,
            instanceCount: rejected.length,
            shape: 'triangle',
            radiusPixels: GHOST_RADIUS_PIXELS,
            fillOpacity: 0,
            outlineColor: colors.rejected,
            outlineWidthPixels: 1,
            color: colors.rejected
          })
        );
      }

      if (options.showSummits) {
        SUMMIT_DROP_CLASSES.sizesPixels.forEach((size, level) => {
          if (classCounts[level] === 0) return;
          layers.push(
            new SpatialAnalysisPointLayer({
              id: `summits-class-${level}`,
              coordinateOrigin: origin,
              positions: summitClassBuffers[level],
              instanceCount: classCounts[level],
              shape: 'triangle',
              // One visual variable: size by drop class; the ink never changes.
              radiusPixels: size / 2,
              outlineColor: colors.paper,
              outlineWidthPixels: 1.5,
              color: colors.summit
            })
          );
        });
      }

      if (options.showSnap && snapRows.length > 0) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'summits-snap-segments',
            coordinateOrigin: origin,
            segments: snapSegments,
            instanceCount: catalogue.length,
            widthPixels: 1.5,
            color: colors.connector
          }),
          new SpatialAnalysisPointLayer({
            id: 'summits-snap-original-edge',
            coordinateOrigin: origin,
            positions: originalMarkers,
            instanceCount: catalogue.length,
            shape: 'ring',
            radiusPixels: 5.25,
            outlineWidthPixels: 3,
            color: colors.summit
          }),
          new SpatialAnalysisPointLayer({
            id: 'summits-snap-original',
            coordinateOrigin: origin,
            positions: originalMarkers,
            instanceCount: catalogue.length,
            shape: 'ring',
            radiusPixels: 4.5,
            outlineWidthPixels: 1.5,
            color: colors.paper
          }),
          new SpatialAnalysisPointLayer({
            id: 'summits-snap-result',
            coordinateOrigin: origin,
            positions: snappedMarkers,
            instanceCount: catalogue.length,
            values: snapBuffers.status,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: getSnapPalette(tone),
            shape: 'circle',
            radiusPixels: 4
          })
        );
      }
      return layers;
    },

    getTooltip(event) {
      const pixel = getPointerPixel(event);
      if (!pixel) return null;
      const options = ctx.options;
      const tone = ctx.ground();
      const colors = getSummitColors(tone);

      if (options.showSnap && snapRows.length) {
        const reach = (HOVER_REACH_PIXELS * ctx.getMetersPerPixel()) / cellMeters;
        let best = -1;
        let bestDistance = reach;
        catalogue.forEach((_, index) => {
          const [column, row] = getCandidatePixel(index);
          const snapped = snappedPixels[index] ?? [column, row];
          const distance = Math.min(
            Math.hypot(column - pixel[0], row - pixel[1]),
            Math.hypot(snapped[0] - pixel[0], snapped[1] - pixel[1])
          );
          if (distance < bestDistance) {
            best = index;
            bestDistance = distance;
          }
        });
        const status = best >= 0 ? getSnapStatusId(snapRows[best].status) : null;
        if (best >= 0 && status) {
          const snapped = snappedPixels[best];
          const lngLat = coarse.getLongitudeLatitude(snapped[0], snapped[1]);
          const topCell = snapLngLatToHighestCell(fullDem, lngLat, cellMeters * 1.5);
          return getSnapTooltip({
            peak: catalogue[best],
            status,
            moveMeters: snapRows[best].distance,
            errorMeters: options.catalogueError,
            analysisCellHeight: snapRows[best].height,
            fullCellHeight: topCell ? topCell.elevation : null,
            analysisCellMeters: cellMeters,
            fullCellMeters: full.groundCellSize,
            ground: tone
          });
        }
      }

      if (options.showSummits) {
        const index = findNearest(survivors, pixel);
        if (index >= 0) {
          return getSummitTooltip({
            candidate: survivors[index],
            peak: survivorNames.get(index) ?? null,
            radiusMeters: options.summitRadius,
            colors
          });
        }
      }
      if (options.showRejected) {
        const index = findNearest(rejected, pixel);
        if (index >= 0) {
          return getRejectedTooltip({
            candidate: rejected[index],
            minimumDrop: options.summitMinimumDrop,
            peak: rejectedNames.get(index) ?? null,
            colors
          });
        }
      }

      const column = Math.round(pixel[0]);
      const row = Math.round(pixel[1]);
      const elevation = coarse.elevation[row * width + column];
      if (options.showProbe) {
        const test = probe.discAndRing(
          column,
          row,
          options.summitRadius,
          options.incompleteNeighborhood
        );
        const distanceToHigher = test.beatenBy
          ? Math.hypot(test.beatenBy[0] - column, test.beatenBy[1] - row) *
            probe.getGroundCellSize(row)
          : null;
        return getProbeTooltip({
          test,
          elevation,
          radiusMeters: options.summitRadius,
          minimumDrop: options.summitMinimumDrop,
          center: coarse.getLongitudeLatitude(column, row),
          distanceToHigherMeters: distanceToHigher,
          colors
        });
      }
      return getElevationTooltip({elevation, cellMeters, colors});
    },

    onGroundChange(next) {
      ground.setGround(next);
      ctx.setLegendData('ground', next);
      publishSnapChart();
      ctx.requestLayers();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    destroy() {
      destroyed = true;
      if (sweepTimer) clearTimeout(sweepTimer);
      for (const variant of variants.values()) variant.reader.stop();
      snapReader.stop();
      criticalReader.stop();
      contourReader.stop();
      ground.destroy();
      resources.destroy();
    }
  };
}
