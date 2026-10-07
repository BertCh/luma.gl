// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {WebMercatorViewport, type Layer, type Viewport} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPURegionStatisticsSummaryLength,
  GPUPickRegionMask,
  GPURegionMask,
  GPURegionStatistics,
  GPURegionStatisticsReadback,
  type GPURegionSelection,
  type GPURegionStatisticsResult
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  GPUGridIndex,
  type CompiledGPUCommandGraph,
  type GPUGridIndexView,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {CHICAGO, nearestPlaceLabel} from '../../cartography/gazetteer';
import {getPolygonLabelPoint} from '../../cartography/anchors';
import {hexToRgba, MAP_INK} from '../../cartography/hue-registry';
import {
  formatArea,
  formatCount,
  formatDistance,
  formatPercent,
  formatRate,
  liveText
} from '../../cartography/live-text';
import {createFeatureLocator, getGeometryPolygons} from '../../cartography/picking';
import type {LngLat, MapAnnotation} from '../../cartography/types';
import {importGraphBuffer, submitGraph} from '../../engine/graph-buffers';
import {
  SpatialAnalysisPointLayer,
  SpatialAnalysisPolygonLayer,
  SpatialAnalysisSegmentLayer
} from '../../engine/layers';
import {LocalMetricProjection} from '../../engine/projection';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {
  formatCompiledGraphTiming,
  formatSpeedup,
  measureCompiledGraph,
  type CompiledGraphTiming
} from '../../engine/vector-timing';
import type {SceneContext, SceneInstance, ScenePointerEvent, TooltipContent} from '../scene';
import {readNatureColumns} from './b1-nature-data';
import {
  addRackPickPasses,
  createRackPickBuffers,
  getPickMatrix,
  getPickWindow,
  PICK_RESULT_PAIRS,
  PICK_TARGET_SIZE,
  PICK_WINDOW_SIZE
} from './b1-lasso-pick';
import {
  getGhostColor,
  getParameterHalo,
  getParameterInk,
  getSubjectColor,
  type PointsColor
} from './b1-points-look';
import {
  CIRCLE_SEGMENTS,
  createCircleRing,
  FILL_VERTEX_CAPACITY,
  getBoundsSegments,
  getCentroid,
  getEqualAreaRadius,
  getGridCellSegments,
  getGridSegmentCapacity,
  getPaddedBounds,
  getRingBounds,
  getRingSignature,
  simplifyRing,
  summariseInsideCircle,
  summariseInsidePolygon,
  summariseInsideRectangle,
  triangulateRing,
  VERTEX_CAPACITY,
  type InsideSummary,
  type Point
} from './lasso-explorer-geometry';
import {
  buildClockChart,
  buildSelectionChart,
  countBins,
  getBinIndex,
  MONTH_STARTS_2023,
  summarisePeak,
  toShares,
  VALUE_KINDS,
  type Normalisation,
  type SelectionChartInput,
  type ValueKind
} from './lasso-explorer-stats';

/** Option state of the lasso-explorer scene. */
export type LassoOptions = {
  shape: 'polygon' | 'radius' | 'rectangle';
  path: 'direct' | 'mask' | 'pick';
  space: 'world' | 'screen';
  area: string;
  radius: number;
  equalArea: boolean;
  circleAt: 'area' | 'montrose-point' | 'custom';
  value: ValueKind;
  domain: 'fixed' | 'selection';
  normalise: Normalisation;
  clockView: boolean;
  /** `'none'`, `'previous'` (the area lassoed before this one) or an area index. */
  compareWith: string;
  gridIndex: boolean;
  candidateCapacity: string;
  withMask: boolean;
  selectedIds: 'mask' | 'ids';
  idCapacity: string;
  showAreas: boolean;
  showBounds: boolean;
  ghostOpacity: number;
};

type GraphKey =
  | 'polygon'
  | 'radius'
  | 'rectangle'
  | 'polygon-screen'
  | 'rectangle-screen'
  | 'mask-polygon'
  | 'mask-rectangle'
  | 'mask-polygon-screen'
  | 'mask-rectangle-screen'
  | 'pick';

/** The area the story starts on (Uptown, community area 3); the first "previous" area of the ghost series. */
const DEFAULT_AREA_INDEX = 2;
const OUTLINE_CAPACITY = VERTEX_CAPACITY;
const READBACK_INTERVAL_FRAMES = 8;
const MINIMUM_VERTEX_SPACING_PIXELS = 5;
const GRID_SIZE = [256, 256] as const;
const CPU_CHECK_IMMEDIATE_LIMIT = 100_000;
const CPU_CHECK_DEBOUNCE_MILLISECONDS = 300;
const AUTO_MEASURE_FRAME = 60;
const ID_CAPACITIES = {small: 1_000, medium: 10_000, all: 65_536} as const;
/** Ghost dots by zoom band, in pixels: a quiet context tier that grows a little when zoomed in. */
const GHOST_RADIUS_STOPS: readonly (readonly [number, number])[] = [
  [10, 0.7],
  [12, 0.9],
  [14, 1.3],
  [16, 1.8]
];
/** The selection reads as a solid figure: the ghost radius plus 0.6 px. */
const SELECTED_RADIUS_STOPS = GHOST_RADIUS_STOPS.map(
  ([zoom, radius]) => [zoom, radius + 0.6] as const
);
/** Radius of a picked point, so single points read. */
const PICK_RADIUS_PIXELS = 3.4;
/** The dark outline of a selected dot appears from this zoom. */
const SELECTED_OUTLINE_STOPS: readonly (readonly [number, number])[] = [
  [12.9, 0],
  [13, 0.6]
];
const CIRCLE_ANCHORS: Record<string, LngLat> = {
  'montrose-point': CHICAGO.places['montrose-point'].lngLat
};

/** One community area: the lasso presets, the context outlines and the tooltip. */
type AreaInfo = {
  id: number;
  name: string;
  areaKm2: number;
  records: number;
  weekendRecords: number;
};

type Variant = {
  key: string;
  compiled: CompiledGPUCommandGraph<void>;
  summary: Buffer;
  readback: GPURegionStatisticsReadback;
  binCount: number;
  valueKind: ValueKind;
};

/**
 * Region statistics over Chicago nature observations. The selection (a lasso polygon, circle,
 * rectangle or pick window) lives in parameter buffers rewritten every frame; the shape KIND,
 * selection path, coordinate space, histogram setting, grid index, mask and id outputs are
 * compile-time, so each combination is a separate compiled graph, built the first time it is needed
 * and cached. The GPU histogram of the selection is read back through a ring every few frames and
 * drawn as a linked chart next to a citywide baseline counted once on the CPU.
 */
export async function createLassoExplorer(
  ctx: SceneContext<LassoOptions>
): Promise<SceneInstance<LassoOptions>> {
  const {device} = ctx;
  const observations = ctx.datasets.get('chicago-nature');
  const areas = ctx.datasets.get('chicago-community-areas');
  const origin = observations.defaultOrigin;
  const projection = new LocalMetricProjection(origin);
  const columns = readNatureColumns(observations, origin);
  const positions = columns.positions;
  const timestamps = observations.column<Uint32Array>('timestamp');
  const pointCount = columns.count;
  const resources = new SpatialAnalysisResources(device, 'lasso');

  // Value columns, one per kind, uploaded into one buffer whose contents change with the option.
  const valueColumns: Record<ValueKind, Float32Array> = {
    hour: new Float32Array(pointCount),
    weekday: new Float32Array(pointCount),
    month: new Float32Array(pointCount),
    category: Float32Array.from(columns.category),
    researchGrade: columns.researchGrade
  };
  const isWeekend = new Uint8Array(pointCount);
  let weekendTotal = 0;
  for (let index = 0; index < pointCount; index++) {
    const seconds = timestamps[index];
    const day = Math.floor(seconds / 86400);
    const hours = (seconds % 86400) / 3600;
    valueColumns.hour[index] = hours;
    valueColumns.weekday[index] = (day % 7) + hours / 24;
    let month = 0;
    while (month < 11 && day >= MONTH_STARTS_2023[month + 1]) month++;
    valueColumns.month[index] =
      month +
      (day - MONTH_STARTS_2023[month]) / (MONTH_STARTS_2023[month + 1] - MONTH_STARTS_2023[month]);
    const weekday = columns.weekday[index];
    isWeekend[index] = weekday === 0 || weekday === 6 ? 1 : 0;
    weekendTotal += isWeekend[index];
  }

  // The citywide shape of every value kind: the denominator of "more than the city".
  const citywideShares = Object.fromEntries(
    (Object.keys(VALUE_KINDS) as ValueKind[]).map(kind => [
      kind,
      toShares(countBins(valueColumns[kind], kind))
    ])
  ) as Record<ValueKind, Float64Array>;
  ctx.setReadout('cityWeekendShare', formatPercent(weekendTotal / pointCount, 0));
  const sampleLine = `${formatCount(pointCount)} iNaturalist records, Chicago, ${new Date(
    timestamps[0] * 1000
  ).getUTCFullYear()}`;

  const positionsBuffer = resources.createBuffer('positions', positions);
  const valuesBuffer = resources.createBuffer('values', valueColumns[ctx.options.value]);
  const maskBuffer = resources.createBuffer('mask', pointCount * 4);
  const regionMaskBuffer = resources.createBuffer('region-mask', pointCount * 4);
  const regionMaskOverflow = resources.createBuffer('region-mask-overflow', 4);
  const idsBuffer = resources.createBuffer('selected-ids', ID_CAPACITIES.all * 4);
  const idsCount = resources.createBuffer('selected-ids-count', 4);
  const idsOverflow = resources.createBuffer('selected-ids-overflow', 4);
  const idsTotal = resources.createBuffer('selected-ids-total', 4);
  const drawCommands = resources.track(
    new DrawCommandBuffer(device, {
      id: 'lasso-selected-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const pick = createRackPickBuffers(resources, pointCount);
  const outlineBuffer = resources.createBuffer('outline', OUTLINE_CAPACITY * 16);
  const flatOutlineBuffer = resources.createBuffer('flat-outline', OUTLINE_CAPACITY * 16);
  const boundsBuffer = resources.createBuffer('bounds-outline', 4 * 16);
  const gridSegmentCapacity = getGridSegmentCapacity(GRID_SIZE);
  const gridLinesBuffer = resources.createBuffer('grid-lines', gridSegmentCapacity * 16);
  const fillTriangles = resources.createBuffer('fill-triangles', FILL_VERTEX_CAPACITY * 8);
  const fillFeatures = resources.createBuffer('fill-features', FILL_VERTEX_CAPACITY * 4);
  const vertexBuffer = resources.createBuffer('vertices', VERTEX_CAPACITY * 8);
  const vertexCount = resources.createParameterBuffer('vertex-count', 'uint32', 1);
  const circleParameters = resources.createParameterBuffer('circle', 'float32', 3);
  const rectangleParameters = resources.createParameterBuffer('rectangle', 'float32', 4);
  const screenTransform = resources.createParameterBuffer('screen-transform', 'float32', 20);

  // Padded bounds and the grid index over the (static) positions, built once.
  const bounds = getPaddedBounds(positions);
  const cellCount = GRID_SIZE[0] * GRID_SIZE[1];
  const cellOffsets = resources.createBuffer('cell-offsets', (cellCount + 1) * 4);
  const objectIds = resources.createBuffer('object-ids', pointCount * 4);
  const indexCount = resources.createBuffer('index-count', 4);
  const indexOverflow = resources.createBuffer('index-overflow', 4);
  {
    const indexGraph = new GPUCommandGraph<void>(device, {id: 'lasso-index'});
    const view = importGridIndexView(indexGraph);
    indexGraph.add(
      new GPUGridIndex({
        id: 'lasso-grid-index',
        positions: importGraphBuffer(
          indexGraph,
          'positions',
          positionsBuffer,
          'float32x2',
          pointCount
        ),
        gridSize: GRID_SIZE,
        bounds,
        cellOffsets: view.cellOffsets,
        objectIds: view.objectIds,
        count: view.count,
        overflow: view.overflow
      })
    );
    submitGraph(device, resources.track(indexGraph.compile()));
  }

  function importGridIndexView(graph: GPUCommandGraph<void>): GPUGridIndexView {
    return {
      gridSize: GRID_SIZE,
      bounds,
      cellOffsets: importGraphBuffer(graph, 'cell-offsets', cellOffsets, 'uint32', cellCount + 1),
      objectIds: importGraphBuffer(graph, 'object-ids', objectIds, 'uint32', pointCount),
      count: importGraphBuffer(graph, 'index-count', indexCount, 'uint32', 1),
      overflow: importGraphBuffer(graph, 'index-overflow', indexOverflow, 'uint32', 1)
    } as GPUGridIndexView;
  }

  // Community-area lassos, context outlines and the per-area facts of the tooltip.
  const areaPolygons = new Map<string, Point[]>();
  const areaInfos: AreaInfo[] = [];
  const areaSegments: number[] = [];
  const communityArea = observations.column<Uint8Array>('communityArea');
  const recordsByAreaId = new Float64Array(78);
  const weekendByAreaId = new Float64Array(78);
  for (let index = 0; index < pointCount; index++) {
    recordsByAreaId[communityArea[index]]++;
    weekendByAreaId[communityArea[index]] += isWeekend[index];
  }
  for (const feature of areas.geojson?.features ?? []) {
    const properties = feature.properties ?? {};
    const id = Number(properties.id ?? areaInfos.length + 1);
    areaInfos.push({
      id,
      name: String(properties.name ?? ''),
      areaKm2: Number(properties.areaKm2 ?? 0),
      records: recordsByAreaId[id] ?? 0,
      weekendRecords: weekendByAreaId[id] ?? 0
    });
    const geometry = feature.geometry;
    if (!geometry) continue;
    const polygons = getGeometryPolygons(geometry) as number[][][][];
    let largest: number[][] = [];
    for (const polygon of polygons) {
      if (polygon[0].length > largest.length) largest = polygon[0];
      const ring = polygon[0];
      for (let index = 0; index < ring.length - 1; index++) {
        const a = projection.project(ring[index][0], ring[index][1]);
        const b = projection.project(ring[index + 1][0], ring[index + 1][1]);
        areaSegments.push(a[0], a[1], b[0], b[1]);
      }
    }
    const simplified = simplifyRing(
      largest.map(([longitude, latitude]) => projection.project(longitude, latitude) as Point),
      VERTEX_CAPACITY - 2
    );
    areaPolygons.set(String(areaInfos.length - 1), simplified);
  }
  const areaSegmentBuffer = resources.createBuffer(
    'area-segments',
    Float32Array.from(areaSegments)
  );
  const areaSegmentCount = areaSegments.length / 4;
  const areaLocator = areas.geojson ? createFeatureLocator(areas.geojson) : null;

  // Interaction state.
  let currentAreaIndex = Number(ctx.options.area);
  // At load the previous area is the default one, so a deep link to the comparison step has its ghost.
  let previousAreaIndex = DEFAULT_AREA_INDEX;
  let lassoIsArea = true;
  let vertices: Point[] = [...(areaPolygons.get(ctx.options.area) ?? [])];
  let circleCenter: Point = getCentroid(vertices);
  let circleCleared = false;
  let rectangleCorners: [Point, Point] | null = null;
  let pickCoordinate: readonly [number, number] | null = null;
  let drawingArmed = false;
  let drawing = false;
  let lastVertexPixel: Point | null = null;
  let destroyed = false;
  let cpuSummary: InsideSummary = {count: 0, weekend: 0};
  let cpuCount = 0;
  let cpuSignature = '';
  let cpuTimer: ReturnType<typeof setTimeout> | undefined;
  let latestResult: GPURegionStatisticsResult | null = null;
  let extraResult = {regionMaskOverflow: 0, pickOverflow: 0, pickCount: 0, pickResultOverflow: 0};
  let idsResult = {count: 0, overflow: 0, total: 0};
  let framesSinceBuild = 0;
  let autoMeasurePending = true;
  let measuring = false;
  let nextTicketFrame = 0;
  let displayed: Variant | null = null;
  let lastScreenSpace = false;
  let screenConversionStableFrames = 0;
  let screenConversionSignature = '';
  let screenConversionThreshold = 25;
  // Derived display state.
  let overlaySignature = '';
  let labelAnchor: LngLat | null = null;
  let noteSignature = '';
  let chartSignature = '';
  let furnitureSignature = '';
  let radiusNoteSignature = '';
  let reportedSelected = -1;
  let pinned: {name: string; shares: Float64Array} | null = null;
  let pinnedKey = '';

  const variants = new Map<string, Variant>();
  const readbacks = new Map<number, GPURegionStatisticsReadback>();

  function getGraphKey(options: LassoOptions): GraphKey {
    if (options.path === 'pick') return 'pick';
    const screen = lastScreenSpace && options.shape !== 'radius';
    const shape = options.shape === 'radius' && options.path === 'mask' ? 'polygon' : options.shape;
    const suffix = screen && shape !== 'radius' ? '-screen' : '';
    if (options.path === 'mask') return `mask-${shape}${suffix}` as GraphKey;
    return `${shape}${suffix}` as GraphKey;
  }

  function getVariantKey(options: LassoOptions): string {
    const graphKey = getGraphKey(options);
    const direct = ['polygon', 'radius', 'rectangle'].includes(graphKey);
    return [
      graphKey,
      options.value,
      options.domain,
      direct && options.gridIndex ? `grid-${options.candidateCapacity}` : 'brute',
      options.withMask && options.selectedIds !== 'ids',
      options.selectedIds === 'ids' ? `ids-${options.idCapacity}` : 'noids'
    ].join('|');
  }

  function compileVariant(options: LassoOptions): Variant {
    const key = getVariantKey(options);
    const graphKey = getGraphKey(options);
    const kind = VALUE_KINDS[options.value];
    const direct = ['polygon', 'radius', 'rectangle'].includes(graphKey);
    const withGrid = direct && options.gridIndex;
    const id = `lasso-${key.replace(/\|/g, '-')}`;
    const summaryLength = getGPURegionStatisticsSummaryLength(kind.binCount);
    const summary = resources.createBuffer(`${id}-summary`, summaryLength * 4);
    const graph = new GPUCommandGraph<void>(device, {id});
    const positionsView = importGraphBuffer(
      graph,
      'positions',
      positionsBuffer,
      'float32x2',
      pointCount
    );
    const screen = graphKey.endsWith('-screen');
    const screenView = screen ? screenTransform.importToGraph(graph) : undefined;
    const polygonShape = () => ({
      kind: 'polygon' as const,
      vertices: importGraphBuffer(graph, 'vertices', vertexBuffer, 'float32x2', VERTEX_CAPACITY),
      vertexCount: vertexCount.importToGraph(graph),
      ...(screenView ? {screenTransform: screenView} : {})
    });
    const rectangleShape = () => ({
      kind: 'rectangle' as const,
      bounds: rectangleParameters.importToGraph(graph),
      ...(screenView ? {screenTransform: screenView} : {})
    });
    let selection: GPURegionSelection;
    if (graphKey === 'polygon' || graphKey === 'polygon-screen') {
      selection = polygonShape();
    } else if (graphKey === 'radius') {
      selection = {kind: 'radius', circle: circleParameters.importToGraph(graph)};
    } else if (graphKey === 'rectangle' || graphKey === 'rectangle-screen') {
      selection = rectangleShape();
    } else if (graphKey === 'pick') {
      const {result} = addRackPickPasses(graph, device, {
        id: 'lasso-pick',
        positions: positionsBuffer,
        count: pointCount,
        buffers: pick
      });
      const mask = importGraphBuffer(graph, 'pick-mask', pick.mask, 'uint32', pointCount);
      graph.add(
        new GPUPickRegionMask({
          id: 'lasso-pick-region-mask',
          result,
          outputMask: mask,
          overflow: importGraphBuffer(graph, 'pick-overflow', pick.overflow, 'uint32', 1)
        })
      );
      selection = {kind: 'mask', mask};
    } else {
      const mask: GraphDataView<'uint32'> = importGraphBuffer(
        graph,
        'region-mask',
        regionMaskBuffer,
        'uint32',
        pointCount
      );
      graph.add(
        new GPURegionMask({
          id: 'lasso-region-mask',
          positions: positionsView,
          region: graphKey.startsWith('mask-polygon') ? polygonShape() : rectangleShape(),
          outputMask: mask,
          overflow: importGraphBuffer(
            graph,
            'region-mask-overflow',
            regionMaskOverflow,
            'uint32',
            1
          )
        })
      );
      selection = {kind: 'mask', mask};
    }
    const wantsIds = options.selectedIds === 'ids';
    const idCapacity =
      ID_CAPACITIES[options.idCapacity as keyof typeof ID_CAPACITIES] ?? ID_CAPACITIES.medium;
    graph.add(
      new GPURegionStatistics({
        id,
        selection,
        ...(selection.kind === 'mask' ? {} : {positions: positionsView}),
        values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', pointCount),
        histogram: {
          binCount: kind.binCount,
          domain: options.domain === 'selection' ? 'selection' : kind.domain
        },
        outputMask:
          options.withMask && !wantsIds && selection.kind !== 'mask'
            ? importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', pointCount)
            : undefined,
        ...(wantsIds
          ? {
              output: {
                ids: importGraphBuffer(graph, 'selected-ids', idsBuffer, 'uint32', idCapacity),
                count: importGraphBuffer(graph, 'selected-ids-count', idsCount, 'uint32', 1),
                overflow: importGraphBuffer(
                  graph,
                  'selected-ids-overflow',
                  idsOverflow,
                  'uint32',
                  1
                ),
                totalCount: importGraphBuffer(graph, 'selected-ids-total', idsTotal, 'uint32', 1)
              },
              drawInstanceCount: graph.importGPUData(
                'draw-instance-count',
                drawCommands.getInstanceCountData(0)
              )
            }
          : {}),
        summary: importGraphBuffer(graph, 'summary', summary, 'uint32', summaryLength),
        spatialIndex: withGrid
          ? {
              kind: 'grid',
              index: importGridIndexView(graph),
              candidateCapacity: Math.max(
                1,
                Math.ceil(pointCount * Number(options.candidateCapacity))
              )
            }
          : undefined
      })
    );
    let readback = readbacks.get(kind.binCount);
    if (!readback) {
      readback = resources.track(
        new GPURegionStatisticsReadback(device, {
          id: `lasso-readback-${kind.binCount}`,
          binCount: kind.binCount
        })
      );
      readbacks.set(kind.binCount, readback);
    }
    return {
      key,
      compiled: resources.track(graph.compile()),
      summary,
      readback,
      binCount: kind.binCount,
      valueKind: options.value
    };
  }

  function ensureVariant(): void {
    const key = getVariantKey(ctx.options);
    let variant = variants.get(key);
    if (!variant) {
      try {
        variant = compileVariant(ctx.options);
      } catch (error) {
        if (!displayed) throw error;
        ctx.setStatus(`That combination cannot be compiled: ${String(error).slice(0, 120)}`);
        return;
      }
      variants.set(key, variant);
    }
    if (displayed?.key !== key) {
      displayed = variant;
      latestResult = null;
      cpuSignature = '';
      chartSignature = '';
      showExtraReadout();
      updateOptionReadouts();
      framesSinceBuild = 0;
      autoMeasurePending = true;
      ctx.setCost({
        records: pointCount,
        passes: ctx.options.path === 'mask' ? 2 : ctx.options.path === 'pick' ? 3 : 1
      });
    }
  }

  const extraReader = new SummaryReader(
    resources,
    'lasso-extra',
    [
      {buffer: regionMaskOverflow, size: 4},
      {buffer: pick.overflow, size: 4},
      {buffer: pick.result, size: 8},
      {buffer: idsCount, size: 4},
      {buffer: idsOverflow, size: 4},
      {buffer: idsTotal, size: 4}
    ],
    bytes => {
      const words = new Uint32Array(bytes);
      extraResult = {
        regionMaskOverflow: words[0],
        pickOverflow: words[1],
        pickCount: words[2],
        pickResultOverflow: words[3]
      };
      idsResult = {count: words[4], overflow: words[5], total: words[6]};
      showExtraReadout();
    }
  );

  // ---------------------------------------------------------------------------------------------
  // Selection names, the baseline and the linked charts
  // ---------------------------------------------------------------------------------------------

  const theme = () => ctx.theme();
  const ground = () => ctx.ground();

  function getSelectionName(): string {
    const options = ctx.options;
    if (options.path === 'pick') return 'Picked points';
    if (options.shape === 'radius') return 'Circle';
    if (options.shape === 'rectangle') return 'Rectangle';
    return lassoIsArea ? (areaInfos[currentAreaIndex]?.name ?? 'Lasso') : 'Your lasso';
  }

  /** The area index shown as the dashed ghost series, or -1. */
  function getPinnedIndex(): number {
    const mode = ctx.options.compareWith;
    if (mode === 'none') return -1;
    const index = mode === 'previous' ? previousAreaIndex : Number(mode);
    const sameAsSelection =
      ctx.options.shape === 'polygon' &&
      ctx.options.path !== 'pick' &&
      lassoIsArea &&
      index === currentAreaIndex;
    return sameAsSelection || !areaInfos[index] ? -1 : index;
  }

  /** Shares of the records whose `communityArea` is the area, per bin of the current value kind. */
  function refreshPinned(): void {
    const index = getPinnedIndex();
    const key = `${index}:${ctx.options.value}`;
    if (key === pinnedKey) return;
    pinnedKey = key;
    if (index < 0) {
      pinned = null;
      return;
    }
    const kind = ctx.options.value;
    const counts = new Float64Array(VALUE_KINDS[kind].binCount);
    const areaId = areaInfos[index].id;
    for (let row = 0; row < pointCount; row++) {
      if (communityArea[row] === areaId) counts[getBinIndex(kind, valueColumns[kind][row])]++;
    }
    pinned = {name: areaInfos[index].name, shares: toShares(counts)};
  }

  function getChartInput(): SelectionChartInput | null {
    if (!latestResult || ctx.options.path === 'pick') return null;
    const options = ctx.options;
    return {
      kind: options.value,
      counts: latestResult.histogram,
      citywide: citywideShares[options.value],
      pinned: options.normalise === 'counts' ? null : pinned,
      normalise: options.normalise,
      selectionName: getSelectionName(),
      domainMode: options.domain,
      categoryNames: columns.categoryNames,
      theme: theme()
    };
  }

  /** Publishes the linked chart and the clock when the histogram or a display option changed. */
  function publishCharts(): void {
    const options = ctx.options;
    const input = getChartInput();
    const signature = input
      ? [
          options.value,
          options.normalise,
          options.clockView,
          options.domain,
          input.selectionName,
          pinnedKey,
          input.theme,
          Array.from(input.counts).join(',')
        ].join('|')
      : 'empty';
    if (signature === chartSignature) return;
    chartSignature = signature;
    ctx.setChart('selectionChart', input ? buildSelectionChart(input) : null);
    ctx.setChart('clockChart', input && options.clockView ? buildClockChart(input) : null);
  }

  function getSubtitle(): string {
    const {value, normalise, domain} = ctx.options;
    const noun = VALUE_KINDS[value].noun;
    if (domain === 'selection') return `Records inside the shape, by ${noun}, bins stretched`;
    if (normalise === 'counts') return `Records inside the shape, by ${noun}`;
    if (normalise === 'share') return `Share of the selection by ${noun}, against the city`;
    return `Selection minus citywide, by ${noun}, in points`;
  }

  /** Cartouche subtitle, sample line and the scale-bar tick at the circle radius (a parameter). */
  function updateFurniture(): void {
    const options = ctx.options;
    const tick =
      options.shape === 'radius' && options.path !== 'pick' && !circleCleared
        ? Math.round(getRadius())
        : null;
    const subtitle = getSubtitle();
    const signature = `${subtitle}|${tick}`;
    if (signature === furnitureSignature) return;
    furnitureSignature = signature;
    ctx.setFurniture({
      title: {subtitle, sample: sampleLine},
      scaleBar: {units: 'metric', ticks: tick ? [tick] : undefined}
    });
  }

  /** The finding note at the lasso: the live count, from the readback. */
  function updateSelectionNote(): void {
    const options = ctx.options;
    const count = latestResult?.selectedCount ?? 0;
    if (!latestResult || !labelAnchor || count <= 0 || options.path === 'pick') {
      if (noteSignature !== 'none') {
        noteSignature = 'none';
        ctx.setAnnotations('selection', null);
      }
      return;
    }
    const signature = `${count}|${labelAnchor[0].toFixed(5)},${labelAnchor[1].toFixed(5)}|${getSelectionName()}`;
    if (signature === noteSignature) return;
    noteSignature = signature;
    const place =
      options.shape === 'polygon' && lassoIsArea
        ? undefined
        : (nearestPlaceLabel(CHICAGO, labelAnchor, {maxDistanceMeters: 4000}) ?? undefined);
    const note: MapAnnotation = {
      kind: 'note',
      id: 'selection-note',
      coordinate: labelAnchor,
      title: liveText('{n:integer} records', {n: count}),
      text: lassoIsArea && options.shape === 'polygon' ? getSelectionName() : place,
      distance: 30
    };
    ctx.setAnnotations('selection', [note]);
  }

  /** The radius written on the circle: a dimension line from the centre to the east edge. */
  function updateRadiusAnnotation(): void {
    const options = ctx.options;
    const visible = options.shape === 'radius' && options.path !== 'pick' && !circleCleared;
    const radius = getRadius();
    const signature = visible ? `${circleCenter.join(',')}|${Math.round(radius)}` : 'none';
    if (signature === radiusNoteSignature) return;
    radiusNoteSignature = signature;
    if (!visible) {
      ctx.setAnnotations('radius', null);
      return;
    }
    const from = projection.unproject(circleCenter[0], circleCenter[1]) as LngLat;
    const to = projection.unproject(circleCenter[0] + radius, circleCenter[1]) as LngLat;
    ctx.setAnnotations('radius', [
      {kind: 'dimension', id: 'radius-dimension', from, to, text: `r = ${formatDistance(radius)}`}
    ]);
  }

  // ---------------------------------------------------------------------------------------------
  // Readouts
  // ---------------------------------------------------------------------------------------------

  function showResult(result: GPURegionStatisticsResult): void {
    latestResult = result;
    const options = ctx.options;
    const kind = VALUE_KINDS[options.value];
    const matches = result.selectedCount === cpuCount;
    ctx.setReadout('selected', formatCount(result.selectedCount));
    ctx.setReadout(
      'cpuCheck',
      cpuCount === -2
        ? 'n/a (picking texture)'
        : cpuCount === -3
          ? 'n/a (screen-space shape)'
          : cpuCount < 0
            ? 'recounting…'
            : `${formatCount(cpuCount)} ${matches ? '(match)' : '(in flight or differs)'}`
    );
    ctx.setReadout(
      'weekendShare',
      cpuCount > 0 && cpuSummary.count === cpuCount
        ? formatPercent(cpuSummary.weekend / cpuCount, 0)
        : '–'
    );
    ctx.setReadout('valueCount', formatCount(result.valueCount));
    ctx.setReadout(
      'mean',
      options.value === 'researchGrade' && result.valueCount > 0
        ? formatPercent(result.mean, 1)
        : '–'
    );
    ctx.setReadout(
      'range',
      result.valueCount > 0 ? `${result.minimum.toFixed(2)} to ${result.maximum.toFixed(2)}` : '-'
    );
    const peak =
      result.valueCount > 0 && options.domain === 'fixed'
        ? summarisePeak(result.histogram, options.value, columns.categoryNames)
        : null;
    ctx.setReadout(
      'peak',
      peak ? `${peak.peakName} (${formatPercent(peak.peakShare, 0)} of the selection)` : '–'
    );
    ctx.setReadout(
      'peakWindowShare',
      peak?.window ? `${formatPercent(peak.window.share, 0)} (${peak.window.text})` : '–'
    );
    ctx.setReadout(
      'histogramLabel',
      `${kind.label}${options.domain === 'selection' ? ' (stretched)' : ''}`
    );
    ctx.setReadout('outside', formatCount(result.histogramOutsideCount));
    const flags = [
      result.regionTruncated ? 'region truncated' : '',
      result.selectionTruncated ? 'id list truncated' : '',
      result.candidatesTruncated ? 'grid candidates truncated' : ''
    ].filter(Boolean);
    ctx.setReadout('flags', flags.length > 0 ? flags.join(', ') : 'none');
    showExtraReadout();
    if (result.selectedCount !== reportedSelected) {
      reportedSelected = result.selectedCount;
      ctx.setLegendData('selected', result.selectedCount);
    }
    publishCharts();
    updateSelectionNote();
  }

  function showExtraReadout(): void {
    const options = ctx.options;
    if (options.path === 'direct') {
      ctx.setReadout('maskContributor', 'direct path: statistics take the shape');
    } else if (options.path === 'mask') {
      ctx.setReadout(
        'maskContributor',
        `GPURegionMask ${options.shape === 'rectangle' ? 'rectangle' : 'polygon'}${options.space === 'screen' ? ' in screen space' : ''}, overflow ${extraResult.regionMaskOverflow ? 'YES' : 'no'}`
      );
    } else {
      ctx.setReadout(
        'maskContributor',
        pickCoordinate
          ? `${formatCount(extraResult.pickCount)} picked pixels in a ${PICK_WINDOW_SIZE}px window; overflow ${extraResult.pickResultOverflow || extraResult.pickOverflow ? 'YES' : 'no'} (capacity ${formatCount(PICK_RESULT_PAIRS)})`
          : 'click or drag over the map'
      );
    }
    ctx.setReadout(
      'ids',
      options.selectedIds === 'ids'
        ? `${formatCount(idsResult.count)} listed of ${formatCount(idsResult.total)}${idsResult.overflow ? ' (list full)' : ''}`
        : 'mask output'
    );
  }

  function updateOptionReadouts(): void {
    const options = ctx.options;
    const parts: string[] = [];
    if (options.path === 'direct') {
      parts.push(
        options.gridIndex
          ? `grid ${GRID_SIZE[0]}x${GRID_SIZE[1]}, candidates ${formatCount(Math.ceil(pointCount * Number(options.candidateCapacity)))}`
          : 'brute force over every observation'
      );
    } else {
      parts.push(options.path === 'mask' ? 'mask first, then statistics' : 'index-picking texture');
    }
    parts.push(
      options.selectedIds === 'ids'
        ? 'compact id list'
        : options.withMask
          ? 'selection mask on'
          : 'statistics only'
    );
    ctx.setReadout('region', parts.join(', '));
  }

  /** Circle and polygon areas, the numbers of the "which boundary" comparison. */
  function updateAreaReadouts(): void {
    const info = areaInfos[currentAreaIndex];
    const radius = getRadius();
    ctx.setReadout('circleArea', formatArea(Math.PI * radius * radius));
    ctx.setReadout('polygonArea', info && lassoIsArea ? formatArea(info.areaKm2 * 1e6) : '–');
    ctx.setReadout('polygonRecords', info && lassoIsArea ? formatCount(info.records) : '–');
  }

  function updateCpuCount(signature: string, count: () => InsideSummary): void {
    if (signature === cpuSignature) return;
    cpuSignature = signature;
    clearTimeout(cpuTimer);
    if (pointCount <= CPU_CHECK_IMMEDIATE_LIMIT) {
      cpuSummary = count();
      cpuCount = cpuSummary.count;
      return;
    }
    cpuCount = -1;
    cpuTimer = setTimeout(() => {
      if (destroyed || signature !== cpuSignature) return;
      cpuSummary = count();
      cpuCount = cpuSummary.count;
      if (latestResult) showResult(latestResult);
    }, CPU_CHECK_DEBOUNCE_MILLISECONDS);
  }

  // ---------------------------------------------------------------------------------------------
  // The shape: parameter buffers, outline, interior tint, bounding box and grid cells
  // ---------------------------------------------------------------------------------------------

  /** Radius of the circle: the slider, or the circle of the same area as the lassoed area. */
  function getRadius(): number {
    const info = areaInfos[currentAreaIndex];
    return ctx.options.equalArea && info ? getEqualAreaRadius(info.areaKm2) : ctx.options.radius;
  }

  /** Writes the radius slider to the equal-area radius (no recompile: a parameter write). */
  function syncEqualAreaRadius(): void {
    if (!ctx.options.equalArea) return;
    const info = areaInfos[currentAreaIndex];
    if (info) ctx.setOptions({radius: Math.round(getEqualAreaRadius(info.areaKm2) / 10) * 10});
  }

  function applyCircleAnchor(): void {
    const {circleAt} = ctx.options;
    if (circleAt === 'area') {
      circleCenter = getCentroid(areaPolygons.get(String(currentAreaIndex)) ?? vertices);
    } else if (circleAt !== 'custom') {
      const anchor = CIRCLE_ANCHORS[circleAt];
      if (anchor) circleCenter = projection.project(anchor[0], anchor[1]);
    }
    circleCleared = false;
  }

  /** Meters point from a screen pixel through the viewport. */
  const pixelToMeters = (viewport: Viewport, pixel: Point): Point => {
    const [longitude, latitude] = viewport.unproject([pixel[0], pixel[1]]);
    return projection.project(longitude, latitude);
  };

  const usesScreen = () =>
    lastScreenSpace && ctx.options.shape !== 'radius' && ctx.options.path !== 'pick';

  /** The same screen pixels unprojected through a flat (pitch 0) copy of the camera. */
  function getFlatFootprint(viewport: Viewport, pixels: readonly Point[]): Point[] | null {
    const camera = viewport as unknown as {
      longitude: number;
      latitude: number;
      zoom: number;
      bearing: number;
      pitch: number;
      width: number;
      height: number;
    };
    if (!(camera.pitch > 1)) return null;
    const flat = new WebMercatorViewport({
      width: camera.width,
      height: camera.height,
      longitude: camera.longitude,
      latitude: camera.latitude,
      zoom: camera.zoom,
      bearing: camera.bearing,
      pitch: 0
    });
    return pixels.map(pixel => {
      const [longitude, latitude] = flat.unproject([pixel[0], pixel[1]]);
      return projection.project(longitude, latitude);
    });
  }

  function packSegments(ring: readonly Point[], capacity: number): Float32Array {
    const segments = new Float32Array(capacity * 4).fill(Number.NaN);
    const count = Math.min(ring.length, capacity);
    if (count >= 2) {
      for (let index = 0; index < count; index++) {
        const a = ring[index];
        const b = ring[(index + 1) % count];
        segments.set([a[0], a[1], b[0], b[1]], index * 4);
      }
    }
    return segments;
  }

  /** Anchor of the count note: the pole of the lasso, or the centre of a circle or rectangle. */
  function getAnchor(ring: readonly Point[]): LngLat | null {
    if (ring.length < 2) return null;
    const options = ctx.options;
    if (options.shape === 'radius' && options.path !== 'pick') {
      return projection.unproject(circleCenter[0], circleCenter[1]) as LngLat;
    }
    const lngLats = ring.map(point => projection.unproject(point[0], point[1]) as [number, number]);
    const pole = lngLats.length >= 3 ? getPolygonLabelPoint([lngLats]) : null;
    if (pole) return pole;
    const center = getCentroid(ring);
    return projection.unproject(center[0], center[1]) as LngLat;
  }

  /** Rewrites the tint, the bounding box and the grid cells when the shape changed. */
  function updateOverlays(ring: Point[], flatRing: Point[] | null): void {
    const options = ctx.options;
    const fill = options.path !== 'pick' && ring.length >= 3;
    const showBox = options.showBounds && options.path !== 'pick' && ring.length >= 2;
    const signature = [
      getRingSignature(ring),
      flatRing ? getRingSignature(flatRing) : '',
      fill,
      showBox,
      options.gridIndex
    ].join('|');
    if (signature === overlaySignature) return;
    overlaySignature = signature;
    labelAnchor = getAnchor(ring);
    noteSignature = '';

    const triangles = fill ? triangulateRing(ring) : [];
    const packed = new Float32Array(FILL_VERTEX_CAPACITY * 2);
    packed.set(triangles.slice(0, FILL_VERTEX_CAPACITY * 2));
    fillTriangles.write(packed);

    const shapeBounds = getRingBounds(ring);
    const boxRows = new Float32Array(16).fill(Number.NaN);
    const gridRows = new Float32Array(gridSegmentCapacity * 4).fill(Number.NaN);
    if (showBox && shapeBounds) {
      boxRows.set(getBoundsSegments(shapeBounds));
      if (options.gridIndex && options.path === 'direct') {
        const cells = getGridCellSegments(shapeBounds, bounds, GRID_SIZE);
        gridRows.set(cells.slice(0, gridRows.length));
      }
    }
    boundsBuffer.write(boxRows);
    gridLinesBuffer.write(gridRows);
    flatOutlineBuffer.write(packSegments(flatRing ?? [], OUTLINE_CAPACITY));
  }

  function writeSelection(viewport: Viewport): void {
    const options = ctx.options;
    const screen = usesScreen();
    let ring: Point[] = [];
    let flatRing: Point[] | null = null;
    if (screen) {
      const matrix = getPickMatrix(viewport, origin);
      matrix[16] = viewport.width;
      matrix[17] = viewport.height;
      screenTransform.write(matrix);
    }
    const toRingPoint = (vertex: Point): Point =>
      screen ? pixelToMeters(viewport, vertex) : vertex;
    const writePolygon = (polygon: readonly Point[], signature: string, countable: boolean) => {
      const count = Math.min(polygon.length, VERTEX_CAPACITY);
      const clipped = polygon.slice(0, count);
      const packed = new Float32Array(VERTEX_CAPACITY * 2);
      clipped.forEach((vertex, index) => packed.set(vertex, index * 2));
      vertexBuffer.write(packed);
      vertexCount.write(Uint32Array.of(count));
      ring = clipped.map(toRingPoint);
      if (screen && count >= 3) flatRing = getFlatFootprint(viewport, clipped);
      if (countable) {
        updateCpuCount(`${signature}:${clipped.join(',')}`, () =>
          summariseInsidePolygon(positions, isWeekend, clipped)
        );
      } else {
        cpuSignature = 'screen';
        cpuCount = -3;
      }
    };
    if (options.path === 'pick') {
      let window: Uint32Array<ArrayBuffer> = Uint32Array.of(0, 0, 0, 0);
      if (pickCoordinate) {
        const pixel = viewport.project([pickCoordinate[0], pickCoordinate[1]]);
        window = getPickWindow(viewport, [pixel[0], pixel[1]]);
        const scaleX = viewport.width / PICK_TARGET_SIZE;
        const scaleY = viewport.height / PICK_TARGET_SIZE;
        ring = (
          [
            [window[0], window[1]],
            [window[0] + window[2], window[1]],
            [window[0] + window[2], window[1] + window[3]],
            [window[0], window[1] + window[3]]
          ] as Point[]
        ).map(([x, y]) => pixelToMeters(viewport, [x * scaleX, y * scaleY]));
      }
      pick.region.write(window);
      pick.matrix.write(getPickMatrix(viewport, origin));
      cpuSignature = 'pick';
      cpuCount = -2;
    } else if (options.shape === 'polygon') {
      writePolygon(vertices, 'polygon', !screen);
    } else if (options.shape === 'rectangle') {
      const corners = rectangleCorners;
      const rectangle = corners
        ? [
            Math.min(corners[0][0], corners[1][0]),
            Math.min(corners[0][1], corners[1][1]),
            Math.max(corners[0][0], corners[1][0]),
            Math.max(corners[0][1], corners[1][1])
          ]
        : [Number.NaN, Number.NaN, Number.NaN, Number.NaN];
      rectangleParameters.write(Float32Array.from(rectangle));
      if (corners) {
        const [x0, y0, x1, y1] = rectangle;
        const corner: Point[] = [
          [x0, y0],
          [x1, y0],
          [x1, y1],
          [x0, y1]
        ];
        ring = corner.map(toRingPoint);
        if (screen) flatRing = getFlatFootprint(viewport, corner);
      }
      if (screen) {
        cpuSignature = 'screen';
        cpuCount = -3;
      } else {
        updateCpuCount(`rectangle:${rectangle.join(',')}`, () =>
          corners
            ? summariseInsideRectangle(positions, isWeekend, rectangle)
            : {count: 0, weekend: 0}
        );
      }
    } else if (options.path === 'mask') {
      const polygon = circleCleared ? [] : createCircleRing(circleCenter, getRadius());
      writePolygon(polygon, 'circle-polygon', true);
    } else {
      const radius = circleCleared ? Number.NaN : getRadius();
      circleParameters.write(Float32Array.of(circleCenter[0], circleCenter[1], radius));
      if (!circleCleared) ring = createCircleRing(circleCenter, radius, CIRCLE_SEGMENTS);
      const center = circleCenter;
      updateCpuCount(`radius:${circleCleared ? 'none' : `${center},${radius}`}`, () =>
        circleCleared
          ? {count: 0, weekend: 0}
          : summariseInsideCircle(positions, isWeekend, center, radius)
      );
    }
    outlineBuffer.write(packSegments(ring, OUTLINE_CAPACITY));
    updateOverlays(ring, flatRing);
    if (latestResult) showResult(latestResult);
  }

  // ---------------------------------------------------------------------------------------------
  // Pointer interaction
  // ---------------------------------------------------------------------------------------------

  const toMeters = (event: ScenePointerEvent): Point | null =>
    event.coordinate ? projection.project(event.coordinate[0], event.coordinate[1]) : null;
  const toShapePoint = (event: ScenePointerEvent): Point | null =>
    usesScreen() ? ([event.pixel[0], event.pixel[1]] as Point) : toMeters(event);

  function setCircleCenter(event: ScenePointerEvent): boolean {
    const center = toMeters(event);
    if (!center) return false;
    circleCenter = center;
    circleCleared = false;
    if (ctx.options.circleAt !== 'custom') ctx.setOptions({circleAt: 'custom'});
    return true;
  }

  function appendVertex(event: ScenePointerEvent, force = false): void {
    const point = toShapePoint(event);
    if (!point || vertices.length >= VERTEX_CAPACITY) return;
    if (!force && lastVertexPixel) {
      const distance = Math.hypot(
        event.pixel[0] - lastVertexPixel[0],
        event.pixel[1] - lastVertexPixel[1]
      );
      if (distance < MINIMUM_VERTEX_SPACING_PIXELS) return;
    }
    vertices.push(point);
    lastVertexPixel = event.pixel;
  }

  function updateStatus(): void {
    const options = ctx.options;
    if (drawingArmed) {
      ctx.setStatus(
        drawing ? 'Drawing lasso… release to close' : 'Drag on the map to draw a lasso'
      );
    } else if (options.path === 'pick') {
      ctx.setStatus('Click or drag over the points to pick the ones you can see');
    } else if (options.shape === 'radius') {
      ctx.setStatus('Click or drag on the map to move the circle');
    } else if (options.shape === 'rectangle') {
      ctx.setStatus('Drag on the map to draw a rectangle');
    } else {
      ctx.setStatus('');
    }
  }

  function finishDrawing(): void {
    drawing = false;
    drawingArmed = false;
    lastVertexPixel = null;
    ctx.setMapDragEnabled(true);
    updateStatus();
  }

  function writeValues(): void {
    valuesBuffer.write(valueColumns[ctx.options.value]);
    latestResult = null;
  }

  function convertSpace(viewport: Viewport): void {
    const toScreen = ctx.options.space === 'screen';
    if (toScreen === lastScreenSpace) return;
    lastScreenSpace = toScreen;
    const convert = (point: Point): Point => {
      if (toScreen) {
        const [longitude, latitude] = projection.unproject(point[0], point[1]);
        const pixel = viewport.project([longitude, latitude]);
        return [pixel[0], pixel[1]];
      }
      return pixelToMeters(viewport, point);
    };
    vertices = vertices.map(convert);
    if (rectangleCorners) {
      rectangleCorners = rectangleCorners.map(convert) as [Point, Point];
    }
    cpuSignature = '';
    overlaySignature = '';
    ensureVariant();
  }

  function applyArea(): void {
    const polygon = areaPolygons.get(ctx.options.area);
    if (!polygon) return;
    vertices = [...polygon];
    lassoIsArea = true;
    if (ctx.options.circleAt === 'area') circleCenter = getCentroid(vertices);
    circleCleared = false;
    if (lastScreenSpace) {
      lastScreenSpace = false;
      screenConversionStableFrames = 0;
      screenConversionThreshold = 1;
      ensureVariant();
    }
    cpuSignature = '';
    overlaySignature = '';
    latestResult = null;
  }

  async function measureGridIndex(): Promise<void> {
    const options = ctx.options;
    if (measuring || destroyed) return;
    if (options.path !== 'direct') {
      ctx.setReadout('timingStatus', 'times the direct path; set Selection path to direct');
      return;
    }
    measuring = true;
    ctx.setReadout('timingStatus', 'measuring…');
    const target = displayed;
    const gridWasOn = options.gridIndex;
    try {
      const otherOptions = {...options, gridIndex: !gridWasOn};
      const otherKey = getVariantKey(otherOptions);
      let other = variants.get(otherKey);
      if (!other) {
        other = compileVariant(otherOptions);
        variants.set(otherKey, other);
      }
      if (!target) return;
      const measureOptions = {
        parameters: undefined,
        completionBuffer: target.summary,
        signal: ctx.signal
      };
      const warm = {...measureOptions, warmUpRuns: 4};
      const first = await measureCompiledGraph(device, target.compiled, warm);
      const firstOther = await measureCompiledGraph(device, other.compiled, {
        ...warm,
        completionBuffer: other.summary
      });
      const second = await measureCompiledGraph(device, target.compiled, measureOptions);
      const secondOther = await measureCompiledGraph(device, other.compiled, {
        ...measureOptions,
        completionBuffer: other.summary
      });
      const currentTiming = getFaster(first, second);
      const otherTiming = getFaster(firstOther, secondOther);
      if (destroyed) return;
      const gridTiming = gridWasOn ? currentTiming : otherTiming;
      const bruteTiming = gridWasOn ? otherTiming : currentTiming;
      ctx.setReadout('gridOn', formatCompiledGraphTiming(gridTiming));
      ctx.setReadout('gridOff', formatCompiledGraphTiming(bruteTiming));
      ctx.setReadout(
        'speedup',
        `${formatSpeedup(bruteTiming.milliseconds, gridTiming.milliseconds)} (${formatCount(pointCount)} points)`
      );
      ctx.setReadout('timingStatus', 'done');
    } catch (error) {
      if (!destroyed) ctx.setReadout('timingStatus', `failed: ${String(error).slice(0, 80)}`);
    } finally {
      measuring = false;
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Start-up
  // ---------------------------------------------------------------------------------------------

  ctx.setReadout('points', formatCount(pointCount));
  applyCircleAnchor();
  syncEqualAreaRadius();
  refreshPinned();
  ensureVariant();
  updateStatus();
  updateOptionReadouts();
  updateAreaReadouts();
  updateFurniture();
  ctx.setLegendData('look', {ground: ground(), theme: theme()});

  /** Colour the context tier of the area outlines: ground ink at 0.25. */
  const getAreaColor = (): PointsColor => {
    const [r, g, b] = hexToRgba(MAP_INK[ground()].context);
    return [r, g, b, 64];
  };

  return {
    getCompiledGraphs: () =>
      (displayed ? [displayed.compiled] : []) as unknown as CompiledGraphList,

    setOption(id, _value, state) {
      if (id === 'area') {
        previousAreaIndex = currentAreaIndex;
        currentAreaIndex = Number(state.area);
        applyArea();
        syncEqualAreaRadius();
        refreshPinned();
        updateAreaReadouts();
        chartSignature = '';
        drawingArmed = false;
        ctx.setMapDragEnabled(true);
        ctx.requestLayers();
      } else if (id === 'value') {
        writeValues();
        refreshPinned();
        ensureVariant();
      } else if (id === 'space') {
        screenConversionStableFrames = 0;
        screenConversionThreshold = 1;
      } else if (
        [
          'shape',
          'path',
          'domain',
          'gridIndex',
          'candidateCapacity',
          'withMask',
          'selectedIds',
          'idCapacity'
        ].includes(id)
      ) {
        if (id === 'shape' || id === 'path') {
          drawingArmed = false;
          ctx.setMapDragEnabled(true);
          circleCleared = false;
          cpuSignature = '';
          if (id === 'path') latestResult = null;
          refreshPinned();
        }
        overlaySignature = '';
        chartSignature = '';
        ensureVariant();
        updateStatus();
        updateOptionReadouts();
        ctx.requestLayers();
      } else if (id === 'radius') {
        circleCleared = false;
        if (ctx.options.equalArea) ctx.setOptions({equalArea: false});
        updateAreaReadouts();
      } else if (id === 'equalArea') {
        syncEqualAreaRadius();
        updateAreaReadouts();
      } else if (id === 'circleAt') {
        applyCircleAnchor();
      } else if (id === 'compareWith') {
        refreshPinned();
        chartSignature = '';
      } else if (id === 'showBounds') {
        overlaySignature = '';
        ctx.requestLayers();
      } else {
        ctx.requestLayers();
      }
      if (latestResult) {
        publishCharts();
        updateSelectionNote();
      }
      updateFurniture();
      updateRadiusAnnotation();
    },

    onAction(id) {
      if (id === 'drawLasso') {
        if (ctx.options.shape !== 'polygon' || ctx.options.path === 'pick') {
          ctx.setStatus('Set Shape to Lasso polygon and Selection path to Direct or Mask to draw');
          return;
        }
        drawingArmed = true;
        ctx.setMapDragEnabled(false);
        updateStatus();
        ctx.requestLayers();
      } else if (id === 'clear') {
        vertices = [];
        lassoIsArea = false;
        circleCleared = true;
        rectangleCorners = null;
        pickCoordinate = null;
        cpuSignature = '';
        overlaySignature = '';
        updateStatus();
        updateAreaReadouts();
      } else if (id === 'measure') {
        void measureGridIndex();
      }
    },

    onThemeChange() {
      ctx.setLegendData('look', {ground: ground(), theme: theme()});
      chartSignature = '';
      publishCharts();
      ctx.requestLayers();
    },

    onGroundChange() {
      ctx.setLegendData('look', {ground: ground(), theme: theme()});
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      if (!displayed) return;
      if (autoMeasurePending && ++framesSinceBuild >= AUTO_MEASURE_FRAME) {
        autoMeasurePending = false;
        // The timing story needs the grid index; elsewhere the second graph would be wasted work.
        if (ctx.options.path === 'direct' && ctx.options.gridIndex) void measureGridIndex();
      }
      if ((ctx.options.space === 'screen') !== lastScreenSpace) {
        const viewport = frame.viewport;
        const signature = Array.from(viewport.viewProjectionMatrix as ArrayLike<number>)
          .map(value => value.toFixed(6))
          .join(',');
        screenConversionStableFrames =
          signature === screenConversionSignature ? screenConversionStableFrames + 1 : 0;
        screenConversionSignature = signature;
        if (screenConversionStableFrames >= screenConversionThreshold) {
          convertSpace(viewport);
          screenConversionThreshold = 25;
        }
      }
      const variant = displayed;
      writeSelection(frame.viewport);
      updateFurniture();
      updateRadiusAnnotation();
      variant.compiled.encode(commandEncoder, {parameters: undefined});
      if (frame.frameIndex % READBACK_INTERVAL_FRAMES === 0) extraReader.markStale();
      extraReader.flush(commandEncoder);
      if (frame.frameIndex >= nextTicketFrame) {
        const ticket = variant.readback.encodeRead(commandEncoder, variant.summary);
        if (ticket) {
          nextTicketFrame = frame.frameIndex + READBACK_INTERVAL_FRAMES;
          variant.readback
            .read(ticket)
            .then(result => {
              if (!destroyed && displayed === variant) showResult(result);
            })
            .catch(() => {
              // Destroyed while in flight.
            });
        }
      }
    },

    getLayers() {
      const options = ctx.options;
      const currentGround = ground();
      const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
      const layers: Layer[] = [];
      const subject = getSubjectColor(currentGround, 235);
      const selectedOutline: PointsColor =
        currentGround === 'dark' ? [19, 23, 28, 210] : [255, 255, 255, 235];
      // Context tier: community outlines, then every observation as a quiet ghost.
      if (options.showAreas) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'lasso-areas',
            coordinateOrigin,
            segments: areaSegmentBuffer,
            instanceCount: areaSegmentCount,
            widthPixels: 0.6,
            color: getAreaColor()
          })
        );
      }
      if (options.ghostOpacity > 0) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'lasso-all-points',
            coordinateOrigin,
            positions: positionsBuffer,
            instanceCount: pointCount,
            radiusPixels: GHOST_RADIUS_STOPS,
            color: getGhostColor(currentGround, Math.round(options.ghostOpacity * 255))
          })
        );
      }
      // The translucent interior of the shape, under the selected dots.
      layers.push(
        new SpatialAnalysisPolygonLayer({
          id: 'lasso-fill',
          coordinateOrigin,
          triangles: fillTriangles,
          features: fillFeatures,
          vertexCount: FILL_VERTEX_CAPACITY,
          color: getSubjectColor(currentGround, currentGround === 'dark' ? 18 : 15)
        })
      );
      // The selection is one amber figure whatever path produced it.
      if (options.path === 'direct' && options.selectedIds === 'ids') {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'lasso-selected-ids',
            coordinateOrigin,
            positions: positionsBuffer,
            drawCommands,
            ids: idsBuffer,
            radiusPixels: SELECTED_RADIUS_STOPS,
            color: subject,
            outlineColor: selectedOutline,
            outlineWidthPixels: SELECTED_OUTLINE_STOPS
          })
        );
      } else if (options.path !== 'direct' || options.withMask) {
        const mask =
          options.path === 'mask'
            ? regionMaskBuffer
            : options.path === 'pick'
              ? pick.mask
              : maskBuffer;
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'lasso-selected-points',
            coordinateOrigin,
            positions: positionsBuffer,
            instanceCount: pointCount,
            radiusPixels: options.path === 'pick' ? PICK_RADIUS_PIXELS : SELECTED_RADIUS_STOPS,
            values: mask,
            valueFormat: 'uint32',
            colormap: 'mask',
            color: subject,
            noDataColor: [0, 0, 0, 0],
            outlineColor: selectedOutline,
            outlineWidthPixels: SELECTED_OUTLINE_STOPS
          })
        );
      }
      // Under the hood: the grid cells and the bounding box the index starts from.
      if (options.showBounds) {
        if (options.gridIndex && options.path === 'direct') {
          layers.push(
            new SpatialAnalysisSegmentLayer({
              id: 'lasso-grid-cells',
              coordinateOrigin,
              segments: gridLinesBuffer,
              instanceCount: gridSegmentCapacity,
              widthPixels: 0.8,
              color: getParameterInk(currentGround, 70)
            })
          );
        }
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'lasso-bounds',
            coordinateOrigin,
            segments: boundsBuffer,
            instanceCount: 4,
            widthPixels: 1.2,
            dashArray: [6, 4],
            color: getParameterInk(currentGround, 190)
          })
        );
      }
      // A screen lasso on a tilted map: the same pixels on a flat map, dashed and faint.
      if (options.space === 'screen' && options.shape !== 'radius' && options.path !== 'pick') {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'lasso-flat-footprint',
            coordinateOrigin,
            segments: flatOutlineBuffer,
            instanceCount: OUTLINE_CAPACITY,
            widthPixels: 1.2,
            dashArray: [5, 4],
            color: getParameterInk(currentGround, 140)
          })
        );
      }
      // The selection outline: achromatic ink on a halo, solid because it is the selection.
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'lasso-outline-halo',
          coordinateOrigin,
          segments: outlineBuffer,
          instanceCount: OUTLINE_CAPACITY,
          widthPixels: 4,
          color: getParameterHalo(currentGround)
        }),
        new SpatialAnalysisSegmentLayer({
          id: 'lasso-outline',
          coordinateOrigin,
          segments: outlineBuffer,
          instanceCount: OUTLINE_CAPACITY,
          widthPixels: 2,
          color: getParameterInk(currentGround)
        })
      );
      return layers;
    },

    getTooltip(event): TooltipContent | null {
      if (drawingArmed || drawing || !event.coordinate || !areaLocator) return null;
      const hit = areaLocator.find(event.coordinate);
      if (!hit) return null;
      const info = areaInfos[hit.index];
      if (!info) return null;
      const options = ctx.options;
      const clickable = options.shape === 'polygon' && options.path !== 'pick';
      const rings = getGeometryPolygons(hit.feature.geometry).map(
        polygon => polygon[0] as unknown as LngLat[]
      );
      return {
        title: info.name,
        subtitle: 'Community area',
        rows: [
          {label: 'Records', value: formatCount(info.records), emphasis: true},
          {
            label: 'Density',
            value: info.areaKm2 > 0 ? formatRate(info.records / info.areaKm2, 'km²') : '–'
          },
          {
            label: 'On weekends',
            value: info.records > 0 ? formatPercent(info.weekendRecords / info.records, 0) : '–'
          }
        ],
        note: clickable ? 'Click to lasso this area' : undefined,
        highlight: {kind: 'polygon', rings}
      };
    },

    onClick(event) {
      const options = ctx.options;
      if (options.path === 'pick') return setPickCoordinate(event);
      if (options.shape === 'radius') return setCircleCenter(event);
      if (options.shape === 'polygon' && !drawingArmed && event.coordinate && areaLocator) {
        const hit = areaLocator.find(event.coordinate);
        if (!hit) return false;
        ctx.setOptions({area: String(hit.index)}, {notify: true});
        return true;
      }
      return false;
    },
    onDragStart(event) {
      const options = ctx.options;
      if (options.path === 'pick') return setPickCoordinate(event);
      if (options.shape === 'radius') return setCircleCenter(event);
      if (options.shape === 'rectangle') {
        const corner = toShapePoint(event);
        if (!corner) return false;
        rectangleCorners = [corner, corner];
        return true;
      }
      if (!drawingArmed) return false;
      drawing = true;
      lassoIsArea = false;
      vertices = [];
      lastVertexPixel = null;
      appendVertex(event, true);
      updateStatus();
      return true;
    },
    onDrag(event) {
      const options = ctx.options;
      if (options.path === 'pick') {
        setPickCoordinate(event);
      } else if (options.shape === 'radius') {
        setCircleCenter(event);
      } else if (options.shape === 'rectangle') {
        const corner = toShapePoint(event);
        if (corner && rectangleCorners) rectangleCorners = [rectangleCorners[0], corner];
      } else if (drawing) {
        appendVertex(event);
        if (vertices.length >= VERTEX_CAPACITY) finishDrawing();
      }
    },
    onDragEnd(event) {
      if (ctx.options.path !== 'pick' && ctx.options.shape === 'polygon' && drawing) {
        appendVertex(event);
        finishDrawing();
        updateAreaReadouts();
      }
    },

    destroy() {
      destroyed = true;
      clearTimeout(cpuTimer);
      extraReader.stop();
      ctx.setMapDragEnabled(true);
      ctx.setAnnotations('selection', null);
      ctx.setAnnotations('radius', null);
      resources.destroy();
    }
  };

  function setPickCoordinate(event: ScenePointerEvent): boolean {
    if (!event.coordinate) return false;
    pickCoordinate = event.coordinate;
    return true;
  }
}

type CompiledGraphList = readonly CompiledGPUCommandGraph<never>[];

function getFaster(first: CompiledGraphTiming, second: CompiledGraphTiming): CompiledGraphTiming {
  return second.milliseconds < first.milliseconds ? second : first;
}
