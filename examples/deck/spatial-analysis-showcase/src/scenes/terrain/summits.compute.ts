// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUTerrainContours,
  GPUTerrainCriticalPoints,
  GPUTerrainDerivatives,
  GPUTerrainPeakSnap,
  GPUTerrainSummits,
  GPU_TERRAIN_CRITICAL_POINT,
  GPU_TERRAIN_CRITICAL_POINT_CLASS_COUNT,
  GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH,
  GPU_TERRAIN_PEAK_SNAP_PARAMETER_LENGTH,
  GPU_TERRAIN_PEAK_SNAP_STATUS,
  GPU_TERRAIN_SUMMITS_PARAMETER_LENGTH,
  getGPUTerrainDerivativesParameterValues,
  getGPUTerrainPeakSnapParameterValues,
  getGPUTerrainSummitsParameterValues
} from '@luma.gl/experimental/gpu-terrain';
import {createWGSLKernelNode} from '../../../../../../modules/experimental/src/utils/wgsl-kernel-nodes';
import {
  addTerrainSummedAreaTableNodes,
  TERRAIN_SUMMED_AREA_BOX_WGSL,
  TERRAIN_SUMMED_AREA_WGSL_HELPERS
} from '../../../../../../modules/experimental/src/gpu-terrain/topographic-position/terrain-summed-area-table';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisRasterLayer,
  SpatialAnalysisSegmentLayer
} from '../../engine/layers';
import {createSeededRandom} from '../../engine/projection';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {formatCompiledGraphTiming, measureCompiledGraph} from '../../engine/vector-timing';
import type {SceneContext, SceneInstance} from '../scene';
import {
  ALPS_PEAK_CATALOGUE,
  createElevationBand,
  getAzimuthDegrees,
  getCompassName,
  formatDistance,
  prepareAlpsTerrain,
  toFloat32,
  toUint32
} from './b14b-terrain';

/** Option state of the summits scene. */
export type SummitsOptions = {
  summitRadius: number;
  summitMinimumDrop: number;
  summitMaximumRadius: '16' | '24' | '32';
  incompleteNeighborhood: 'reject' | 'ignore';
  catalogueError: number;
  snapRadius: number;
  snapMaximumMove: number;
  snapMaximumHeightChange: number;
  snapInterior: boolean;
  snapCatalogueHeights: boolean;
  snapDistanceRule: boolean;
  connectivity: '8' | '6';
  devRadius: number;
  contourInterval: number;
  contourIndexEvery: number;
  base: 'hillshade' | 'relative-height' | 'elevation';
  showSummits: boolean;
  showSnap: boolean;
  showCritical: boolean;
  showContours: boolean;
};

/** The analysis grid of summits, snapping and critical points: 512 x 512, 26.6 m ground cells. */
const COARSE_STRIDE = 4;
const SUMMIT_CAPACITY = 1024;
const CANDIDATE_CAPACITY = 24;
const CONTOUR_LEVEL_COUNT = 40;
const CONTOUR_SEGMENT_CAPACITY = 30000;
const SUMMIT_BUCKET_COUNT = 4;
const SUMMIT_BUCKET_DROPS = [0, 200, 400, 700];
const SUMMIT_BUCKET_RADIUS = [3.5, 5.5, 8, 11];
const SUMMIT_BUCKET_COLOR: readonly (readonly [number, number, number, number])[] = [
  [255, 226, 110, 235],
  [255, 176, 70, 240],
  [255, 118, 60, 245],
  [235, 50, 70, 250]
];
const CRITICAL_PALETTE = [
  [0, 0, 0, 0],
  [235, 60, 70, 235],
  [70, 140, 255, 235],
  [255, 214, 70, 235],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0]
] as const;
const SNAP_STATUS_PALETTE = [
  [190, 200, 215, 255],
  [70, 235, 130, 255],
  [255, 160, 60, 255],
  [235, 60, 70, 255],
  [220, 90, 230, 255],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0]
] as const;
const SNAP_STATUS_NAMES = [
  'unchanged',
  'snapped',
  'on the ring (flank)',
  'move too far',
  'height change too large',
  'no data',
  'outside'
];
const DEV_QUANTUM = 1 / 256;

type Candidate = {name: string; catalogueElevation: number | null; column: number; row: number};

type SummitGraph = {key: string; compiled: CompiledGPUCommandGraph<void>};

/**
 * Summits, peak snapping, critical points, a summed-area-table relative-height map and contours
 * of the Matterhorn DEM. Summit-scale analyses run on the DEM averaged 4 x 4 (26.6 m ground); the
 * contours and the hillshade use the full 2048 x 2048 grid.
 */
export async function createSummits(
  ctx: SceneContext<SummitsOptions>
): Promise<SceneInstance<SummitsOptions>> {
  const dataset = ctx.datasets.get('alps-dem');
  const full = prepareAlpsTerrain(dataset, 1);
  const coarse = prepareAlpsTerrain(dataset, COARSE_STRIDE);
  const {device} = ctx;
  const origin: [number, number, number] = [full.origin[0], full.origin[1], 0];
  const resources = new SpatialAnalysisResources(device, 'summits');
  const {width, height, pixelCount} = coarse;
  let destroyed = false;

  // --- Buffers ----------------------------------------------------------------------------------
  const fullElevation = resources.createBuffer('full-elevation', full.elevation);
  const fullValidity = resources.createBuffer('full-validity', full.validity);
  const coarseElevation = resources.createBuffer('coarse-elevation', coarse.elevation);
  const coarseValidity = resources.createBuffer('coarse-validity', coarse.validity);
  const hillshadeBuffer = resources.createBuffer('hillshade', full.pixelCount * 4);
  const derivativesSettings = resources.createParameterBuffer(
    'derivatives-settings',
    'float32',
    GPU_TERRAIN_DERIVATIVES_PARAMETER_LENGTH
  );
  const summitSettings = resources.createParameterBuffer(
    'summit-settings',
    'float32',
    GPU_TERRAIN_SUMMITS_PARAMETER_LENGTH
  );
  const snapSettings = resources.createParameterBuffer(
    'snap-settings',
    'float32',
    GPU_TERRAIN_PEAK_SNAP_PARAMETER_LENGTH
  );
  const devSettings = resources.createParameterBuffer('dev-settings', 'float32', 4);
  const levelValues = resources.createParameterBuffer('levels', 'float32', CONTOUR_LEVEL_COUNT);

  const summitIds = resources.createBuffer('summit-ids', SUMMIT_CAPACITY * 4);
  const summitDrops = resources.createBuffer('summit-drops', SUMMIT_CAPACITY * 4);
  const summitCount = resources.createBuffer('summit-count', 4);
  const summitTotal = resources.createBuffer('summit-total', 4);
  const summitOverflow = resources.createBuffer('summit-overflow', 4);
  const summitClamped = resources.createBuffer('summit-clamped', 4);
  const summitBuckets = Array.from({length: SUMMIT_BUCKET_COUNT}, (_, bucket) =>
    resources.createBuffer(`summit-bucket-${bucket}`, SUMMIT_CAPACITY * 8)
  );
  const criticalClasses = resources.createBuffer('critical-classes', pixelCount * 4);
  const criticalSigns = resources.createBuffer('critical-signs', pixelCount * 4);
  const criticalCounts = resources.createBuffer(
    'critical-counts',
    GPU_TERRAIN_CRITICAL_POINT_CLASS_COUNT * 4
  );
  const devBuffer = resources.createBuffer('relative-height', pixelCount * 4);

  const candidateBuffer = resources.createBuffer('candidates', CANDIDATE_CAPACITY * 8);
  const candidateHeights = resources.createBuffer('candidate-heights', CANDIDATE_CAPACITY * 4);
  const candidateRadii = resources.createBuffer('candidate-radii', CANDIDATE_CAPACITY * 4);
  const snapPositions = resources.createBuffer('snap-positions', CANDIDATE_CAPACITY * 8);
  const snapHeights = resources.createBuffer('snap-heights', CANDIDATE_CAPACITY * 4);
  const snapStatus = resources.createBuffer('snap-status', CANDIDATE_CAPACITY * 4);
  const snapDistance = resources.createBuffer('snap-distance', CANDIDATE_CAPACITY * 4);
  const snapOverflow = resources.createBuffer('snap-overflow', 4);
  const originalMarkers = resources.createBuffer('original-markers', CANDIDATE_CAPACITY * 8);
  const snappedMarkers = resources.createBuffer('snapped-markers', CANDIDATE_CAPACITY * 8);
  const snapSegments = resources.createBuffer('snap-segments', CANDIDATE_CAPACITY * 16);

  const contourOverflow = resources.createBuffer('contour-overflow', 4);
  const contourVertices: Buffer[] = [];
  const contourCounts: Buffer[] = [];
  for (let level = 0; level < CONTOUR_LEVEL_COUNT; level++) {
    contourVertices.push(
      resources.createBuffer(`contour-vertices-${level}`, CONTOUR_SEGMENT_CAPACITY * 16)
    );
    contourCounts.push(resources.createBuffer(`contour-count-${level}`, 4));
  }
  const drawCommands = resources.track(
    new DrawCommandBuffer(device, {
      id: 'summits-contour-draw',
      type: 'draw',
      commands: Array.from({length: CONTOUR_LEVEL_COUNT}, () => ({
        vertexCount: 6,
        instanceCount: 0
      }))
    })
  );

  // --- Hillshade (full resolution, once) --------------------------------------------------------
  const hillshadeGraph = new GPUCommandGraph<void>(device, {id: 'summits-hillshade'});
  hillshadeGraph.add(
    new GPUTerrainDerivatives({
      id: 'derivatives',
      width: full.width,
      height: full.height,
      elevation: createElevationBand(
        hillshadeGraph,
        'hillshade',
        fullElevation,
        fullValidity,
        full.pixelCount
      ),
      settings: derivativesSettings.importToGraph(hillshadeGraph),
      hillshade: importGraphBuffer(
        hillshadeGraph,
        'hillshade',
        hillshadeBuffer,
        'float32',
        full.pixelCount
      ),
      cellSizeMode: 'web-mercator',
      rowDirection: 'south'
    })
  );
  const hillshadeCompiled = resources.track(hillshadeGraph.compile());
  derivativesSettings.write(
    getGPUTerrainDerivativesParameterValues({
      ...full.mercatorCellSettings,
      azimuthDegrees: 315,
      altitudeDegrees: 40
    })
  );

  // --- Summits ----------------------------------------------------------------------------------
  let summitGraph: SummitGraph | null = null;
  const getSummitKey = (): string =>
    `${ctx.options.summitMaximumRadius}|${ctx.options.incompleteNeighborhood}`;
  function buildSummitGraph(): SummitGraph {
    const options = ctx.options;
    const graph = new GPUCommandGraph<void>(device, {id: 'summits-summits'});
    graph.add(
      new GPUTerrainSummits({
        id: 'summits',
        width,
        height,
        elevation: createElevationBand(
          graph,
          'summits',
          coarseElevation,
          coarseValidity,
          pixelCount
        ),
        cellSizeMode: 'web-mercator',
        maximumRadiusPixels: Number(options.summitMaximumRadius),
        incompleteNeighborhood: options.incompleteNeighborhood,
        settings: summitSettings.importToGraph(graph),
        output: {
          ids: importGraphBuffer(graph, 'ids', summitIds, 'uint32', SUMMIT_CAPACITY),
          count: importGraphBuffer(graph, 'count', summitCount, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'overflow', summitOverflow, 'uint32', 1),
          totalCount: importGraphBuffer(graph, 'total', summitTotal, 'uint32', 1)
        },
        outputDrop: importGraphBuffer(graph, 'drops', summitDrops, 'float32', SUMMIT_CAPACITY),
        overflow: importGraphBuffer(graph, 'clamped', summitClamped, 'uint32', 1)
      })
    );
    return {key: getSummitKey(), compiled: resources.track(graph.compile())};
  }

  // --- Critical points --------------------------------------------------------------------------
  let criticalGraph: {key: string; compiled: CompiledGPUCommandGraph<void>} | null = null;
  function buildCriticalGraph() {
    const connectivity = ctx.options.connectivity;
    const graph = new GPUCommandGraph<void>(device, {id: `summits-critical-${connectivity}`});
    graph.add(
      new GPUTerrainCriticalPoints({
        id: 'critical',
        width,
        height,
        elevation: createElevationBand(
          graph,
          'critical',
          coarseElevation,
          coarseValidity,
          pixelCount
        ),
        connectivity: connectivity === '8' ? 8 : 6,
        classes: importGraphBuffer(graph, 'classes', criticalClasses, 'uint32', pixelCount),
        signChanges: importGraphBuffer(graph, 'signs', criticalSigns, 'uint32', pixelCount),
        counts: importGraphBuffer(
          graph,
          'counts',
          criticalCounts,
          'uint32',
          GPU_TERRAIN_CRITICAL_POINT_CLASS_COUNT
        )
      })
    );
    return {key: connectivity, compiled: resources.track(graph.compile())};
  }

  // --- Peak snap --------------------------------------------------------------------------------
  const snapGraph = new GPUCommandGraph<void>(device, {id: 'summits-snap'});
  snapGraph.add(
    new GPUTerrainPeakSnap({
      id: 'snap',
      width,
      height,
      elevation: createElevationBand(
        snapGraph,
        'snap',
        coarseElevation,
        coarseValidity,
        pixelCount
      ),
      cellSizeMode: 'web-mercator',
      maximumRadiusPixels: 16,
      candidates: importGraphBuffer(
        snapGraph,
        'candidates',
        candidateBuffer,
        'float32x2',
        CANDIDATE_CAPACITY
      ),
      candidateHeights: importGraphBuffer(
        snapGraph,
        'candidate-heights',
        candidateHeights,
        'float32',
        CANDIDATE_CAPACITY
      ),
      candidateRadii: importGraphBuffer(
        snapGraph,
        'candidate-radii',
        candidateRadii,
        'float32',
        CANDIDATE_CAPACITY
      ),
      settings: snapSettings.importToGraph(snapGraph),
      positions: importGraphBuffer(
        snapGraph,
        'positions',
        snapPositions,
        'float32x2',
        CANDIDATE_CAPACITY
      ),
      heights: importGraphBuffer(snapGraph, 'heights', snapHeights, 'float32', CANDIDATE_CAPACITY),
      status: importGraphBuffer(snapGraph, 'status', snapStatus, 'uint32', CANDIDATE_CAPACITY),
      snapDistance: importGraphBuffer(
        snapGraph,
        'distance',
        snapDistance,
        'float32',
        CANDIDATE_CAPACITY
      ),
      overflow: importGraphBuffer(snapGraph, 'overflow', snapOverflow, 'uint32', 1)
    })
  );
  const snapCompiled = resources.track(snapGraph.compile());

  // --- Relative height from a summed-area table -------------------------------------------------
  // `addTerrainSummedAreaTableNodes` quantizes the elevation, builds an exact modular summed-area
  // table of heights, squares and counts (two scans and a transpose), and the kernel below reads
  // any box mean and variance from it in O(1). The radius is a per-frame parameter.
  const devGraph = new GPUCommandGraph<void>(device, {id: 'summits-relative-height'});
  const devElevation = createElevationBand(
    devGraph,
    'dev',
    coarseElevation,
    coarseValidity,
    pixelCount
  );
  const devSettingsView = devSettings.importToGraph(devGraph);
  const devOutput = importGraphBuffer(devGraph, 'dev', devBuffer, 'float32', pixelCount);
  const devProducer: GPUCommandNodeProducer = {
    getCommandNodes<Parameters>(
      graph: GPUCommandGraph<Parameters>
    ): readonly GPUCommandNode<Parameters>[] {
      const sat = addTerrainSummedAreaTableNodes(graph, {
        id: 'relative-height',
        width,
        height,
        elevation: devElevation,
        quantum: DEV_QUANTUM
      });
      const evaluate = createWGSLKernelNode<Parameters>(graph, {
        id: 'relative-height-evaluate',
        operation: 'GPUTerrainTopographicPosition',
        variant: 'relative-height',
        bindings: [
          {name: 'lowTable', view: sat.lowTable, type: 'u32', access: 'read'},
          {name: 'highTable', view: sat.highTable, type: 'u32', access: 'read'},
          {name: 'settings', view: devSettingsView, type: 'f32', access: 'read'},
          {name: 'devValues', view: devOutput, type: 'f32', access: 'read_write'}
        ],
        invocationCount: pixelCount,
        declarations: `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const PIXEL_COUNT: u32 = ${pixelCount}u;
const INVERSE_QUANTUM: f32 = ${(1 / DEV_QUANTUM).toFixed(1)};
${TERRAIN_SUMMED_AREA_WGSL_HELPERS}
${TERRAIN_SUMMED_AREA_BOX_WGSL}
fn readClippedBox(column: i32, row: i32, radius: i32) -> TerrainBoxSums {
  return readBoxSums(
    max(column - radius, 0i),
    min(column + radius, i32(WIDTH) - 1i),
    max(row - radius, 0i),
    min(row + radius, i32(HEIGHT) - 1i)
  );
}`,
        body: `let column = i32(index % WIDTH);
  let row = i32(index / WIDTH);
  let centre = readBoxSums(column, column, row, row);
  let isValid = centre.count == 1u;
  let invalidValue = bitcast<f32>(0x7fc00000u | (index & 0u));
  let quantized = bitcast<i32>(centre.sum.x);
  let magnitude = u32(abs(quantized));
  let isNegative = quantized < 0i;
  let centreSquare = multiplyWide32(magnitude, magnitude);
  let radius = max(i32(round(settings[settingsOffset])), 1i);
  let outer = readClippedBox(column, row, radius);
  // Population statistics of the window relative to the centre cell:
  // d1 = sum(q - qc), d2 = sum((q - qc)^2); deviation = (qc - mean) / sd = -d1 / n / sd.
  var centreTotal = multiply64By32(vec2<u32>(outer.count, 0u), magnitude);
  if (isNegative) {
    centreTotal = negate64(centreTotal);
  }
  let centredSum = subtract64(outer.sum, centreTotal);
  var twiceCentreSum = multiply64By32(outer.sum, magnitude);
  twiceCentreSum = add64(twiceCentreSum, twiceCentreSum);
  if (isNegative) {
    twiceCentreSum = negate64(twiceCentreSum);
  }
  let centredSquare = add64(
    subtract64(outer.square, twiceCentreSum),
    multiply64By32(centreSquare, outer.count)
  );
  let count = f32(outer.count);
  let meanOffset = signed64ToFloat(centredSum) / count;
  let variance = unsigned64ToFloat(centredSquare) / count - meanOffset * meanOffset;
  var deviation = 0.0;
  if (variance > 0.0) {
    deviation = -meanOffset / sqrt(variance);
  }
  if (!isValid) {
    deviation = invalidValue;
  }
  devValues[devValuesOffset + index] = deviation;`
      });
      return [...sat.nodes, evaluate];
    }
  };
  devGraph.add(devProducer);
  const devCompiled = resources.track(devGraph.compile());

  // --- Contours (full resolution; 40 level slots with GPU-written draw records) ----------------
  const contourGraph = new GPUCommandGraph<void>(device, {id: 'summits-contours'});
  const contourElevation = createElevationBand(
    contourGraph,
    'contours',
    fullElevation,
    fullValidity,
    full.pixelCount
  );
  const levelsView = levelValues.importToGraph(contourGraph);
  const drawView = drawCommands.importToGraph(contourGraph);
  contourGraph.add(
    new GPUTerrainContours({
      id: 'contours',
      width: full.width,
      height: full.height,
      elevation: contourElevation,
      overflow: importGraphBuffer(contourGraph, 'overflow', contourOverflow, 'uint32', 1),
      levels: Array.from({length: CONTOUR_LEVEL_COUNT}, (_, level) => ({
        level: contourGraph.createDataView(levelsView.buffer, {
          format: 'float32',
          length: 1,
          byteOffset: level * 4
        }),
        vertices: importGraphBuffer(
          contourGraph,
          `vertices-${level}`,
          contourVertices[level],
          'float32x2',
          CONTOUR_SEGMENT_CAPACITY * 2
        ),
        segmentCount: importGraphBuffer(
          contourGraph,
          `count-${level}`,
          contourCounts[level],
          'uint32',
          1
        ),
        // The contributor rewrites [verticesPerInstance, segmentCount, 0, 0] on the GPU, so the
        // segment layer draws exactly the segments found, with no readback.
        draw: drawView,
        drawCommandIndex: level,
        drawLayout: 'instanced' as const,
        verticesPerInstance: 6,
        capacity: CONTOUR_SEGMENT_CAPACITY
      }))
    })
  );
  const contourCompiled = resources.track(contourGraph.compile());

  // --- State ------------------------------------------------------------------------------------
  const random = createSeededRandom(7);
  const errorAngles = ALPS_PEAK_CATALOGUE.map(() => random() * Math.PI * 2);
  const candidates: Candidate[] = ALPS_PEAK_CATALOGUE.map(peak => {
    const [column, row] = coarse.getPixel(peak.longitude, peak.latitude);
    return {name: peak.name, catalogueElevation: peak.elevationMeters, column, row};
  });
  const observer = coarse.observers.find(place => place.name === 'Gornergrat');
  const observerPixel: [number, number] = observer
    ? coarse.getPixel(observer.longitude, observer.latitude)
    : [width / 2, height / 2];

  let summitDirty = true;
  let criticalDirty = true;
  let snapDirty = true;
  let devDirty = true;
  let contourDirty = true;
  let hillshadeDirty = true;
  let summitPositions: {column: number; row: number; drop: number; elevation: number}[] = [];
  let summitBucketCounts = new Array<number>(SUMMIT_BUCKET_COUNT).fill(0);
  let snapRows: {status: number; distance: number; height: number}[] = [];
  let snappedPixels: [number, number][] = [];
  let contourCountsRead = new Uint32Array(CONTOUR_LEVEL_COUNT);
  let contourReady = false;
  let clickCount = 0;

  const getCoarseMeters = (column: number, row: number): [number, number] =>
    coarse.getMeters(column, row);

  function writeSummitSettings(): void {
    summitSettings.write(
      getGPUTerrainSummitsParameterValues({
        radius: ctx.options.summitRadius,
        minimumDrop: ctx.options.summitMinimumDrop,
        ...coarse.mercatorCellSettings
      })
    );
    summitDirty = true;
  }

  function writeCandidates(): void {
    const options = ctx.options;
    const rows = new Float32Array(CANDIDATE_CAPACITY * 2).fill(-5);
    const heights = new Float32Array(CANDIDATE_CAPACITY).fill(NaN);
    const radii = new Float32Array(CANDIDATE_CAPACITY).fill(NaN);
    const original = new Float32Array(CANDIDATE_CAPACITY * 2).fill(1e7);
    const errorPixels = options.catalogueError / coarse.groundCellSize;
    candidates.forEach((candidate, index) => {
      const isCatalogue = index < ALPS_PEAK_CATALOGUE.length;
      const angle = isCatalogue ? errorAngles[index] : 0;
      const column = candidate.column + (isCatalogue ? Math.cos(angle) * errorPixels : 0);
      const row = candidate.row + (isCatalogue ? Math.sin(angle) * errorPixels : 0);
      rows[index * 2] = column;
      rows[index * 2 + 1] = row;
      if (options.snapCatalogueHeights && candidate.catalogueElevation !== null) {
        heights[index] = candidate.catalogueElevation;
      }
      if (options.snapDistanceRule) {
        // mt-image: min(250 m, 60 m + 0.4 % of the distance to the viewer).
        const distance =
          Math.hypot(column - observerPixel[0], row - observerPixel[1]) * coarse.groundCellSize;
        radii[index] = Math.min(250, 60 + 0.004 * distance);
      }
      const [x, y] = getCoarseMeters(column, row);
      original[index * 2] = x;
      original[index * 2 + 1] = y;
    });
    candidateBuffer.write(rows);
    candidateHeights.write(heights);
    candidateRadii.write(radii);
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

  function writeDevSettings(): void {
    devSettings.write(Float32Array.of(ctx.options.devRadius, 0, 0, 0));
    devDirty = true;
  }

  function getContourLevels(): number[] {
    const interval = ctx.options.contourInterval;
    const first = Math.ceil(full.elevationRange[0] / interval) * interval;
    return Array.from({length: CONTOUR_LEVEL_COUNT}, (_, level) => first + level * interval);
  }

  function writeContourLevels(): void {
    levelValues.write(Float32Array.from(getContourLevels()));
    contourDirty = true;
  }

  // --- Readers ----------------------------------------------------------------------------------
  const summitReader = new SummaryReader(
    resources,
    'summits',
    [
      {buffer: summitCount, size: 4},
      {buffer: summitTotal, size: 4},
      {buffer: summitOverflow, size: 4},
      {buffer: summitClamped, size: 4},
      {buffer: summitIds, size: SUMMIT_CAPACITY * 4},
      {buffer: summitDrops, size: SUMMIT_CAPACITY * 4}
    ],
    bytes => applySummits(bytes)
  );
  const snapReader = new SummaryReader(
    resources,
    'snap',
    [
      {buffer: snapPositions, size: CANDIDATE_CAPACITY * 8},
      {buffer: snapHeights, size: CANDIDATE_CAPACITY * 4},
      {buffer: snapStatus, size: CANDIDATE_CAPACITY * 4},
      {buffer: snapDistance, size: CANDIDATE_CAPACITY * 4},
      {buffer: snapOverflow, size: 4}
    ],
    bytes => applySnap(bytes)
  );
  const contourReader = new SummaryReader(
    resources,
    'contours',
    [{buffer: contourOverflow, size: 4}, ...contourCounts.map(buffer => ({buffer, size: 4}))],
    bytes => applyContours(bytes)
  );

  function applySummits(bytes: ArrayBuffer): void {
    const words = toUint32(bytes, 4);
    const count = Math.min(words[0], SUMMIT_CAPACITY);
    const ids = toUint32(bytes.slice(16), SUMMIT_CAPACITY);
    const drops = toFloat32(bytes.slice(16 + SUMMIT_CAPACITY * 4), SUMMIT_CAPACITY);
    summitPositions = [];
    const buckets: number[][] = Array.from({length: SUMMIT_BUCKET_COUNT}, () => []);
    for (let index = 0; index < count; index++) {
      const id = ids[index];
      const column = id % width;
      const row = Math.floor(id / width);
      const drop = drops[index];
      const bucket = SUMMIT_BUCKET_DROPS.reduce(
        (found, limit, level) => (drop >= limit ? level : found),
        0
      );
      const [x, y] = getCoarseMeters(column, row);
      buckets[bucket].push(x, y);
      summitPositions.push({column, row, drop, elevation: coarse.elevation[id]});
    }
    summitBucketCounts = buckets.map(bucket => bucket.length / 2);
    buckets.forEach((bucket, index) => summitBuckets[index].write(Float32Array.from(bucket)));
    const highest = summitPositions.reduce(
      (best, summit) => (summit.elevation > best.elevation ? summit : best),
      {column: 0, row: 0, drop: 0, elevation: -Infinity}
    );
    ctx.setReadout(
      'summits',
      `${formatCount(words[1])} summits` +
        (words[2] ? ` (list capped at ${formatCount(SUMMIT_CAPACITY)})` : '') +
        (count
          ? `, highest ${highest.elevation.toFixed(0)} m, largest drop ${Math.max(...summitPositions.map(summit => summit.drop)).toFixed(0)} m`
          : '')
    );
    ctx.setReadout(
      'summitRadius',
      words[3]
        ? `clamped: the radius exceeds ${ctx.options.summitMaximumRadius} pixels (${(Number(ctx.options.summitMaximumRadius) * coarse.groundCellSize).toFixed(0)} m)`
        : 'as requested'
    );
    ctx.requestLayers();
  }

  function applySnap(bytes: ArrayBuffer): void {
    const positions = toFloat32(bytes, CANDIDATE_CAPACITY * 2);
    const heights = toFloat32(bytes.slice(CANDIDATE_CAPACITY * 8), CANDIDATE_CAPACITY);
    const status = toUint32(bytes.slice(CANDIDATE_CAPACITY * 12), CANDIDATE_CAPACITY);
    const distance = toFloat32(bytes.slice(CANDIDATE_CAPACITY * 16), CANDIDATE_CAPACITY);
    const overflow = toUint32(bytes.slice(CANDIDATE_CAPACITY * 20), 1)[0];
    snapRows = [];
    snappedPixels = [];
    const snapped = new Float32Array(CANDIDATE_CAPACITY * 2).fill(1e7);
    const segments = new Float32Array(CANDIDATE_CAPACITY * 4).fill(1e7);
    const tally = new Array<number>(7).fill(0);
    candidates.forEach((candidate, index) => {
      snapRows.push({status: status[index], distance: distance[index], height: heights[index]});
      const column = positions[index * 2];
      const row = positions[index * 2 + 1];
      snappedPixels.push([column, row]);
      tally[status[index]] = (tally[status[index]] ?? 0) + 1;
      const [x, y] = getCoarseMeters(column, row);
      snapped[index * 2] = x;
      snapped[index * 2 + 1] = y;
      const original = candidateOriginalMeters(index);
      segments.set([original[0], original[1], x, y], index * 4);
    });
    snappedMarkers.write(snapped);
    snapSegments.write(segments);
    ctx.setReadout(
      'snap',
      `${tally[GPU_TERRAIN_PEAK_SNAP_STATUS.snapped]} snapped, ${tally[GPU_TERRAIN_PEAK_SNAP_STATUS.unchanged]} unchanged, ` +
        `${tally[GPU_TERRAIN_PEAK_SNAP_STATUS.onRing]} on ring, ${tally[GPU_TERRAIN_PEAK_SNAP_STATUS.rejectedMove]} move too far, ` +
        `${tally[GPU_TERRAIN_PEAK_SNAP_STATUS.rejectedHeight]} height change` +
        (overflow ? ' (radius clamped)' : '')
    );
    const named = candidates
      .map((candidate, index) => ({candidate, index}))
      .filter(({candidate}) => candidate.catalogueElevation !== null);
    const moves = named
      .filter(({index}) => status[index] === GPU_TERRAIN_PEAK_SNAP_STATUS.snapped)
      .map(({index}) => distance[index]);
    ctx.setReadout(
      'snapMove',
      moves.length
        ? `mean ${(moves.reduce((sum, value) => sum + value, 0) / moves.length).toFixed(0)} m, largest ${Math.max(...moves).toFixed(0)} m`
        : 'no catalogue peak moved'
    );
    ctx.requestLayers();
  }

  function candidateOriginalMeters(index: number): [number, number] {
    const options = ctx.options;
    const candidate = candidates[index];
    const isCatalogue = index < ALPS_PEAK_CATALOGUE.length;
    const errorPixels = options.catalogueError / coarse.groundCellSize;
    const angle = isCatalogue ? errorAngles[index] : 0;
    return getCoarseMeters(
      candidate.column + (isCatalogue ? Math.cos(angle) * errorPixels : 0),
      candidate.row + (isCatalogue ? Math.sin(angle) * errorPixels : 0)
    );
  }

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
      'contours',
      `${levels} levels with data, ${formatCount(segments)} segments` +
        (overflow
          ? ` · overflow: a level exceeded ${formatCount(CONTOUR_SEGMENT_CAPACITY)} segments`
          : '')
    );
    ctx.requestLayers();
  }

  async function readCritical(): Promise<void> {
    try {
      const [classBytes, signBytes] = await Promise.all([
        criticalClasses.readAsync(0, pixelCount * 4),
        criticalSigns.readAsync(0, pixelCount * 4)
      ]);
      if (destroyed) return;
      const classes = toUint32(classBytes, pixelCount);
      const signs = toUint32(signBytes, pixelCount);
      let peaks = 0;
      let pits = 0;
      let saddles = 0;
      let multiplicity = 0;
      let boundary = 0;
      for (let index = 0; index < pixelCount; index++) {
        const code = classes[index];
        if (code === GPU_TERRAIN_CRITICAL_POINT.peak) peaks++;
        else if (code === GPU_TERRAIN_CRITICAL_POINT.pit) pits++;
        else if (code === GPU_TERRAIN_CRITICAL_POINT.saddle) {
          saddles++;
          multiplicity += signs[index] / 2 - 1;
        } else if (code === GPU_TERRAIN_CRITICAL_POINT.boundary) boundary++;
      }
      ctx.setReadout(
        'critical',
        `${formatCount(peaks)} peaks, ${formatCount(saddles)} saddles (${formatCount(multiplicity)} with multiplicity), ${formatCount(pits)} pits`
      );
      ctx.setReadout(
        'euler',
        `peaks - saddles + pits = ${formatCount(peaks - multiplicity + pits)} (${formatCount(boundary)} boundary cells unclassified)`
      );
    } catch {
      // Destroyed while reading.
    }
  }

  // --- Setup and per-frame work -----------------------------------------------------------------
  function ensureGraphs(): void {
    if (!summitGraph || summitGraph.key !== getSummitKey()) {
      if (summitGraph) resources.release(summitGraph.compiled);
      summitGraph = buildSummitGraph();
      summitDirty = true;
    }
    if (!criticalGraph || criticalGraph.key !== ctx.options.connectivity) {
      if (criticalGraph) resources.release(criticalGraph.compiled);
      criticalGraph = buildCriticalGraph();
      criticalDirty = true;
    }
  }

  ctx.setReadout(
    'grid',
    `${width} × ${height} cells at ${coarse.groundCellSize.toFixed(1)} m for summits; ${full.width} × ${full.height} at ${full.groundCellSize.toFixed(1)} m for contours`
  );
  writeSummitSettings();
  writeCandidates();
  writeSnapSettings();
  writeDevSettings();
  writeContourLevels();
  ensureGraphs();

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
      const parts = [
        await run(summitGraph!.compiled, summitCount, 'summits'),
        await run(criticalGraph!.compiled, criticalCounts, 'critical'),
        await run(snapCompiled, snapStatus, 'snap'),
        await run(devCompiled, devBuffer, 'SAT + relative height'),
        await run(contourCompiled, contourOverflow, `${CONTOUR_LEVEL_COUNT} contour levels`)
      ];
      ctx.setReadout('timing', parts.join(' · '));
    } catch (error) {
      if (!destroyed) ctx.setReadout('timing', `failed: ${(error as Error).message}`);
    }
  }

  // --- Pointer interaction ----------------------------------------------------------------------
  function getPixelFromEvent(event: {
    coordinate: readonly [number, number] | null;
  }): [number, number] | null {
    if (!event.coordinate) return null;
    const [x, y] = full.projection.project(event.coordinate[0], event.coordinate[1]);
    const [column, row] = coarse.getPixelFromMeters(x, y);
    return [column, row];
  }

  function getScreenDistance(
    meters: readonly [number, number],
    event: {pixel: readonly [number, number]}
  ): number {
    const viewport = ctx.getViewport();
    if (!viewport) return Infinity;
    const [longitude, latitude] = full.projection.unproject(meters[0], meters[1]);
    const [x, y] = viewport.project([longitude, latitude]);
    return Math.hypot(x - event.pixel[0], y - event.pixel[1]);
  }

  return {
    getCompiledGraphs: () => {
      const graphs = [hillshadeCompiled, snapCompiled, devCompiled, contourCompiled];
      if (summitGraph) graphs.push(summitGraph.compiled);
      if (criticalGraph) graphs.push(criticalGraph.compiled);
      return graphs;
    },

    setOption(id) {
      switch (id) {
        case 'summitRadius':
        case 'summitMinimumDrop':
          writeSummitSettings();
          break;
        case 'summitMaximumRadius':
        case 'incompleteNeighborhood':
        case 'connectivity':
          ensureGraphs();
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
        case 'devRadius':
          writeDevSettings();
          break;
        case 'contourInterval':
          writeContourLevels();
          break;
        default:
          break;
      }
      ctx.requestLayers();
    },

    onAction(id) {
      if (id === 'clear') {
        candidates.length = ALPS_PEAK_CATALOGUE.length;
        clickCount = 0;
        writeCandidates();
      }
      if (id === 'measure') void measure();
    },

    encode(commandEncoder) {
      if (!summitGraph || !criticalGraph) return;
      if (hillshadeDirty) {
        hillshadeCompiled.encode(commandEncoder, {parameters: undefined});
        hillshadeDirty = false;
      }
      if (summitDirty) {
        summitGraph.compiled.encode(commandEncoder, {parameters: undefined});
        summitReader.request(commandEncoder);
        summitDirty = false;
      }
      if (criticalDirty) {
        criticalGraph.compiled.encode(commandEncoder, {parameters: undefined});
        criticalDirty = false;
        // Two 1 MB reads, once per ring change, outside the frame's critical path.
        setTimeout(() => void readCritical(), 100);
      }
      if (snapDirty) {
        snapCompiled.encode(commandEncoder, {parameters: undefined});
        snapReader.request(commandEncoder);
        snapDirty = false;
      }
      if (devDirty && ctx.options.base === 'relative-height') {
        devCompiled.encode(commandEncoder, {parameters: undefined});
        devDirty = false;
      }
      if (contourDirty && ctx.options.showContours) {
        contourCompiled.encode(commandEncoder, {parameters: undefined});
        contourReader.request(commandEncoder);
        contourDirty = false;
      }
      summitReader.flush(commandEncoder);
      snapReader.flush(commandEncoder);
      contourReader.flush(commandEncoder);
    },

    getLayers() {
      const options = ctx.options;
      const layers: Layer[] = [];
      const fine = {
        coordinateOrigin: origin,
        gridSize: [full.width, full.height] as const,
        bounds: full.bounds,
        rowOrigin: 'north' as const,
        valueFormat: 'float32' as const
      };
      const coarseProps = {
        coordinateOrigin: origin,
        gridSize: [width, height] as const,
        bounds: coarse.bounds,
        rowOrigin: 'north' as const
      };
      layers.push(
        new SpatialAnalysisRasterLayer({
          ...fine,
          id: 'summits-hillshade',
          values: hillshadeBuffer,
          colormap: 'grayscale',
          valueRange: [0, 1],
          color: [255, 255, 255, options.base === 'hillshade' ? 225 : 150]
        })
      );
      if (options.base === 'elevation') {
        layers.push(
          new SpatialAnalysisRasterLayer({
            ...fine,
            id: 'summits-elevation',
            values: fullElevation,
            colormap: 'cividis',
            valueRange: [full.elevationRange[0], full.elevationRange[1]],
            color: [255, 255, 255, 160]
          })
        );
      } else if (options.base === 'relative-height') {
        layers.push(
          new SpatialAnalysisRasterLayer({
            ...coarseProps,
            id: 'summits-relative-height',
            values: devBuffer,
            valueFormat: 'float32',
            colormap: 'diverging',
            valueRange: [-3, 3],
            color: [255, 255, 255, 190]
          })
        );
      }
      if (options.showContours && contourReady) {
        const levels = getContourLevels();
        for (let level = 0; level < CONTOUR_LEVEL_COUNT; level++) {
          const isIndex =
            Math.round(levels[level] / options.contourInterval) % options.contourIndexEvery === 0;
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: `summits-contour-${level}`,
              coordinateOrigin: origin,
              segments: contourVertices[level],
              drawCommands,
              drawCommandIndex: level,
              // Vertices are in pixel-edge grid units with row 0 at the north edge.
              positionScale: [full.layerCellSize[0], -full.layerCellSize[1]],
              positionOffset: [full.bounds[0], full.bounds[3]],
              widthPixels: isIndex ? 1.7 : 0.8,
              color: isIndex ? [150, 80, 20, 255] : [190, 120, 50, 190]
            })
          );
        }
      }
      if (options.showCritical) {
        layers.push(
          new SpatialAnalysisRasterLayer({
            ...coarseProps,
            id: `summits-critical-${options.connectivity}`,
            values: criticalClasses,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: CRITICAL_PALETTE
          })
        );
      }
      if (options.showSummits) {
        summitBuckets.forEach((buffer, bucket) => {
          if (summitBucketCounts[bucket] === 0) return;
          layers.push(
            new SpatialAnalysisPointLayer({
              id: `summits-bucket-halo-${bucket}`,
              coordinateOrigin: origin,
              positions: buffer,
              instanceCount: summitBucketCounts[bucket],
              radiusPixels: SUMMIT_BUCKET_RADIUS[bucket] + 1.6,
              color: [20, 24, 36, 220]
            }),
            new SpatialAnalysisPointLayer({
              id: `summits-bucket-${bucket}`,
              coordinateOrigin: origin,
              positions: buffer,
              instanceCount: summitBucketCounts[bucket],
              radiusPixels: SUMMIT_BUCKET_RADIUS[bucket],
              color: SUMMIT_BUCKET_COLOR[bucket]
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
            instanceCount: candidates.length,
            widthPixels: 1.8,
            color: [20, 24, 36, 230]
          }),
          new SpatialAnalysisPointLayer({
            id: 'summits-snap-original',
            coordinateOrigin: origin,
            positions: originalMarkers,
            instanceCount: candidates.length,
            radiusPixels: 6.5,
            color: [255, 255, 255, 255]
          }),
          new SpatialAnalysisPointLayer({
            id: 'summits-snap-original-hole',
            coordinateOrigin: origin,
            positions: originalMarkers,
            instanceCount: candidates.length,
            radiusPixels: 3.6,
            color: [20, 24, 36, 255]
          }),
          new SpatialAnalysisPointLayer({
            id: 'summits-snap-result',
            coordinateOrigin: origin,
            positions: snappedMarkers,
            instanceCount: candidates.length,
            values: snapStatus,
            valueFormat: 'uint32',
            colormap: 'category',
            palette: SNAP_STATUS_PALETTE,
            radiusPixels: 5
          })
        );
      }
      return layers;
    },

    getTooltip(event) {
      if (!event.coordinate) return null;
      const options = ctx.options;
      if (options.showSnap) {
        let best = -1;
        let bestDistance = 14;
        candidates.forEach((_, index) => {
          const distance = getScreenDistance(candidateOriginalMeters(index), event);
          if (distance < bestDistance) {
            best = index;
            bestDistance = distance;
          }
        });
        if (best >= 0 && snapRows[best]) {
          const row = snapRows[best];
          const candidate = candidates[best];
          return (
            `${candidate.name}: ${SNAP_STATUS_NAMES[row.status] ?? row.status}` +
            (row.status === GPU_TERRAIN_PEAK_SNAP_STATUS.snapped
              ? ` by ${row.distance.toFixed(0)} m`
              : '') +
            ` · DEM ${Number.isFinite(row.height) ? row.height.toFixed(0) : '?'} m` +
            (candidate.catalogueElevation !== null
              ? ` · catalogue ${candidate.catalogueElevation} m`
              : '')
          );
        }
      }
      if (options.showSummits) {
        let best: (typeof summitPositions)[number] | null = null;
        let bestDistance = 12;
        for (const summit of summitPositions) {
          const distance = getScreenDistance(getCoarseMeters(summit.column, summit.row), event);
          if (distance < bestDistance) {
            best = summit;
            bestDistance = distance;
          }
        }
        if (best) {
          const named = candidates.slice(0, ALPS_PEAK_CATALOGUE.length).find((candidate, index) => {
            const snapped = snappedPixels[index];
            return snapped && Math.hypot(snapped[0] - best!.column, snapped[1] - best!.row) < 1.5;
          });
          return `${named ? `${named.name}, ` : 'Summit, '}${best.elevation.toFixed(0)} m · drop at least ${best.drop.toFixed(0)} m within the ring`;
        }
      }
      const [column, row] = getPixelFromEvent(event) ?? [0, 0];
      const dx = (column - observerPixel[0]) * coarse.groundCellSize;
      const dy = (observerPixel[1] - row) * coarse.groundCellSize;
      return `${coarse.sampleElevation(column, row).toFixed(0)} m · ${formatDistance(Math.hypot(dx, dy))} ${getCompassName(getAzimuthDegrees(dx, dy))} of Gornergrat`;
    },

    onClick(event) {
      const pixel = getPixelFromEvent(event);
      if (!pixel || !ctx.options.showSnap) return false;
      if (candidates.length >= CANDIDATE_CAPACITY) return false;
      clickCount++;
      candidates.push({
        name: `Clicked point ${clickCount}`,
        catalogueElevation: null,
        column: pixel[0],
        row: pixel[1]
      });
      writeCandidates();
      return true;
    },

    destroy() {
      destroyed = true;
      summitReader.stop();
      snapReader.stop();
      contourReader.stop();
      resources.destroy();
    }
  };
}
