// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer, Viewport} from '@deck.gl/core';
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
import {importGraphBuffer, submitGraph} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {LocalMetricProjection} from '../../engine/projection';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {
  formatCompiledGraphTiming,
  formatSpeedup,
  measureCompiledGraph,
  type CompiledGraphTiming
} from '../../engine/vector-timing';
import type {SceneContext, SceneInstance, ScenePointerEvent} from '../scene';
import {formatHourWindow} from './b1-nature-data';
import {
  addRackPickPasses,
  createRackPickBuffers,
  getPickMatrix,
  getPickWindow,
  PICK_RESULT_PAIRS,
  PICK_TARGET_SIZE,
  PICK_WINDOW_SIZE
} from './b1-lasso-pick';

/** Option state of the lasso-explorer scene. */
export type LassoOptions = {
  shape: 'polygon' | 'radius' | 'rectangle';
  path: 'direct' | 'mask' | 'pick';
  space: 'world' | 'screen';
  area: string;
  radius: number;
  value: 'hour' | 'weekday' | 'month' | 'researchGrade';
  domain: 'fixed' | 'selection';
  gridIndex: boolean;
  candidateCapacity: string;
  withMask: boolean;
  selectedIds: 'mask' | 'ids';
  idCapacity: string;
  showAreas: boolean;
  showPoints: boolean;
};

type Point = readonly [number, number];
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

const VERTEX_CAPACITY = 256;
const CIRCLE_SEGMENTS = 64;
const OUTLINE_CAPACITY = VERTEX_CAPACITY;
const READBACK_INTERVAL_FRAMES = 8;
const MINIMUM_VERTEX_SPACING_PIXELS = 5;
const GRID_SIZE = [256, 256] as const;
const CPU_CHECK_IMMEDIATE_LIMIT = 100_000;
const CPU_CHECK_DEBOUNCE_MILLISECONDS = 300;
const SPARK_LEVELS = '▁▂▃▄▅▆▇█';
const AUTO_MEASURE_FRAME = 60;
const ID_CAPACITIES = {small: 1_000, medium: 10_000, all: 65_536} as const;

/** Value kinds: the quantity histogrammed over the selected observations. */
const VALUE_KINDS = {
  hour: {binCount: 24, domain: [0, 24] as const, label: 'Hour of day'},
  weekday: {binCount: 7, domain: [0, 7] as const, label: 'Day of week'},
  month: {binCount: 12, domain: [0, 12] as const, label: 'Month of year'},
  researchGrade: {binCount: 2, domain: [0, 1] as const, label: 'Research grade (0 or 1)'}
};
const WEEKDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const MONTH_STARTS_2023 = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365];

type Variant = {
  key: string;
  compiled: CompiledGPUCommandGraph<void>;
  summary: Buffer;
  readback: GPURegionStatisticsReadback;
  binCount: number;
  valueKind: LassoOptions['value'];
};

/**
 * Region statistics over Chicago nature observations. The selection (a lasso polygon, circle, rectangle
 * or pick window) lives in parameter buffers rewritten every frame; the shape KIND, selection path,
 * coordinate space, histogram setting, grid index, mask and id outputs are compile-time, so each
 * combination is a separate compiled graph, built the first time it is needed and cached.
 */
export async function createLassoExplorer(
  ctx: SceneContext<LassoOptions>
): Promise<SceneInstance<LassoOptions>> {
  const {device} = ctx;
  const observations = ctx.datasets.get('chicago-nature');
  const areas = ctx.datasets.get('chicago-community-areas');
  const origin = observations.defaultOrigin;
  const projection = new LocalMetricProjection(origin);
  const positions = observations.projectColumn('position', origin);
  const timestamps = observations.column<Uint32Array>('timestamp');
  const researchGradeRaw = observations.column<Uint8Array>('researchGrade');
  const pointCount = timestamps.length;
  const resources = new SpatialAnalysisResources(device, 'lasso');

  // Value columns, one per kind, uploaded into one buffer whose contents change with the option.
  const valueColumns: Record<LassoOptions['value'], Float32Array> = {
    hour: new Float32Array(pointCount),
    weekday: new Float32Array(pointCount),
    month: new Float32Array(pointCount),
    researchGrade: new Float32Array(pointCount)
  };
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
    valueColumns.researchGrade[index] = researchGradeRaw[index] ? 1 : 0;
  }

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

  // Community-area lassos and context outlines.
  const areaPolygons = new Map<string, Point[]>();
  const areaNames: string[] = [];
  const areaSegments: number[] = [];
  for (const feature of areas.geojson?.features ?? []) {
    const name = String(feature.properties?.name ?? '');
    areaNames.push(name);
    const geometry = feature.geometry;
    if (!geometry) continue;
    const polygons = (
      geometry.type === 'MultiPolygon' ? geometry.coordinates : [geometry.coordinates]
    ) as number[][][][];
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
    areaPolygons.set(String(areaNames.length - 1), simplified);
  }
  const areaSegmentBuffer = resources.createBuffer(
    'area-segments',
    Float32Array.from(areaSegments)
  );
  const areaSegmentCount = areaSegments.length / 4;

  // Interaction state.
  let vertices: Point[] = [...(areaPolygons.get(ctx.options.area) ?? [])];
  let circleCenter: Point = getCentroid(vertices);
  let circleCleared = false;
  let rectangleCorners: [Point, Point] | null = null;
  let pickCoordinate: readonly [number, number] | null = null;
  let drawingArmed = false;
  let drawing = false;
  let lastVertexPixel: Point | null = null;
  let destroyed = false;
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
      showExtraReadout();
      updateOptionReadouts();
      framesSinceBuild = 0;
      autoMeasurePending = true;
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

  function formatHistogram(histogram: Uint32Array): string {
    const maximum = Math.max(...histogram, 1);
    return Array.from(histogram, count =>
      count === 0 ? '·' : SPARK_LEVELS[Math.min(7, Math.floor((count / maximum) * 8))]
    ).join('');
  }

  function describePeak(histogram: Uint32Array, valueKind: LassoOptions['value']): string {
    let peak = 0;
    for (let bin = 1; bin < histogram.length; bin++)
      if (histogram[bin] > histogram[peak]) peak = bin;
    const total = histogram.reduce((sum, count) => sum + count, 0) || 1;
    const share = `${((histogram[peak] / total) * 100).toFixed(0)}%`;
    if (valueKind === 'hour')
      return `${formatHourWindow([peak, peak + 1], false)} (${share} of observations)`;
    if (valueKind === 'weekday') return `${WEEKDAYS[peak]} (${share})`;
    if (valueKind === 'month') return `${MONTHS[peak]} (${share})`;
    return `${peak === 1 ? 'research grade' : 'not yet confirmed'} (${share})`;
  }

  function showResult(result: GPURegionStatisticsResult): void {
    latestResult = result;
    const options = ctx.options;
    const kind = VALUE_KINDS[options.value];
    const matches = result.selectedCount === cpuCount;
    ctx.setReadout(
      'selected',
      `${formatCount(result.selectedCount)} of ${formatCount(pointCount)}`
    );
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
    ctx.setReadout('valueCount', formatCount(result.valueCount));
    let mean = result.mean.toFixed(2);
    if (options.value === 'hour') mean = `${result.mean.toFixed(1)} h (linear mean)`;
    if (options.value === 'researchGrade')
      mean = `${(result.mean * 100).toFixed(1)}% research grade`;
    ctx.setReadout('mean', result.valueCount > 0 ? mean : '-');
    ctx.setReadout(
      'range',
      result.valueCount > 0 ? `${result.minimum.toFixed(2)} to ${result.maximum.toFixed(2)}` : '-'
    );
    ctx.setReadout(
      'histogramLabel',
      `${kind.label}${options.domain === 'selection' ? ' (range = selection)' : ''}`
    );
    ctx.setReadout('histogram', result.valueCount > 0 ? formatHistogram(result.histogram) : '-');
    ctx.setReadout(
      'peak',
      result.valueCount > 0 ? describePeak(result.histogram, options.value) : '-'
    );
    ctx.setReadout('outside', formatCount(result.histogramOutsideCount));
    const flags = [
      result.regionTruncated ? 'region truncated' : '',
      result.selectionTruncated ? 'id list truncated' : '',
      result.candidatesTruncated ? 'grid candidates truncated' : ''
    ].filter(Boolean);
    ctx.setReadout('flags', flags.length > 0 ? flags.join(', ') : 'none');
    showExtraReadout();
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

  function updateCpuCount(signature: string, count: () => number): void {
    if (signature === cpuSignature) return;
    cpuSignature = signature;
    clearTimeout(cpuTimer);
    if (pointCount <= CPU_CHECK_IMMEDIATE_LIMIT) {
      cpuCount = count();
      return;
    }
    cpuCount = -1;
    cpuTimer = setTimeout(() => {
      if (destroyed || signature !== cpuSignature) return;
      cpuCount = count();
      if (latestResult) showResult(latestResult);
    }, CPU_CHECK_DEBOUNCE_MILLISECONDS);
  }

  /** Meters point from a screen pixel through the viewport. */
  const pixelToMeters = (viewport: Viewport, pixel: Point): Point => {
    const [longitude, latitude] = viewport.unproject([pixel[0], pixel[1]]);
    return projection.project(longitude, latitude);
  };

  function writeSelection(viewport: Viewport): void {
    const options = ctx.options;
    const screen =
      options.space === 'screen' && options.shape !== 'radius' && options.path !== 'pick';
    const outline = new Float32Array(OUTLINE_CAPACITY * 4).fill(Number.NaN);
    if (screen) {
      const matrix = getPickMatrix(viewport, origin);
      matrix[16] = viewport.width;
      matrix[17] = viewport.height;
      screenTransform.write(matrix);
    }
    const toOutlinePoint = (vertex: Point): Point =>
      screen ? pixelToMeters(viewport, vertex) : vertex;
    const writePolygon = (polygon: readonly Point[], signature: string, countable: boolean) => {
      const count = Math.min(polygon.length, VERTEX_CAPACITY);
      const packed = new Float32Array(VERTEX_CAPACITY * 2);
      polygon.slice(0, count).forEach((vertex, index) => packed.set(vertex, index * 2));
      vertexBuffer.write(packed);
      vertexCount.write(Uint32Array.of(count));
      if (count >= 2) {
        for (let index = 0; index < count; index++) {
          outline.set(
            [...toOutlinePoint(polygon[index]), ...toOutlinePoint(polygon[(index + 1) % count])],
            index * 4
          );
        }
      }
      if (countable) {
        const clipped = polygon.slice(0, count);
        updateCpuCount(`${signature}:${clipped.join(',')}`, () =>
          countInsidePolygon(positions, clipped)
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
        const corners = (
          [
            [window[0], window[1]],
            [window[0] + window[2], window[1]],
            [window[0] + window[2], window[1] + window[3]],
            [window[0], window[1] + window[3]]
          ] as Point[]
        ).map(([x, y]) => pixelToMeters(viewport, [x * scaleX, y * scaleY]));
        for (let index = 0; index < 4; index++) {
          outline.set([...corners[index], ...corners[(index + 1) % 4]], index * 4);
        }
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
        const ring: Point[] = [
          [x0, y0],
          [x1, y0],
          [x1, y1],
          [x0, y1]
        ];
        for (let index = 0; index < 4; index++) {
          outline.set(
            [...toOutlinePoint(ring[index]), ...toOutlinePoint(ring[(index + 1) % 4])],
            index * 4
          );
        }
      }
      if (screen) {
        cpuSignature = 'screen';
        cpuCount = -3;
      } else {
        updateCpuCount(`rectangle:${rectangle.join(',')}`, () =>
          corners ? countInsideRectangle(positions, rectangle) : 0
        );
      }
    } else if (options.path === 'mask') {
      const ring: Point[] = circleCleared
        ? []
        : Array.from({length: CIRCLE_SEGMENTS}, (_, index) => {
            const angle = (index / CIRCLE_SEGMENTS) * Math.PI * 2;
            return [
              circleCenter[0] + Math.cos(angle) * options.radius,
              circleCenter[1] + Math.sin(angle) * options.radius
            ] as Point;
          });
      writePolygon(ring, 'circle-polygon', true);
    } else {
      const radius = circleCleared ? Number.NaN : options.radius;
      circleParameters.write(Float32Array.of(circleCenter[0], circleCenter[1], radius));
      if (!circleCleared) {
        for (let index = 0; index < CIRCLE_SEGMENTS; index++) {
          const angle0 = (index / CIRCLE_SEGMENTS) * Math.PI * 2;
          const angle1 = ((index + 1) / CIRCLE_SEGMENTS) * Math.PI * 2;
          outline.set(
            [
              circleCenter[0] + Math.cos(angle0) * options.radius,
              circleCenter[1] + Math.sin(angle0) * options.radius,
              circleCenter[0] + Math.cos(angle1) * options.radius,
              circleCenter[1] + Math.sin(angle1) * options.radius
            ],
            index * 4
          );
        }
      }
      const center = circleCenter;
      updateCpuCount(`radius:${circleCleared ? 'none' : `${center},${options.radius}`}`, () =>
        circleCleared ? 0 : countInsideCircle(positions, center, options.radius)
      );
    }
    outlineBuffer.write(outline);
    if (latestResult) showResult(latestResult);
  }

  const toMeters = (event: ScenePointerEvent): Point | null =>
    event.coordinate ? projection.project(event.coordinate[0], event.coordinate[1]) : null;
  const usesScreen = () =>
    lastScreenSpace && ctx.options.shape !== 'radius' && ctx.options.path !== 'pick';
  const toShapePoint = (event: ScenePointerEvent): Point | null =>
    usesScreen() ? ([event.pixel[0], event.pixel[1]] as Point) : toMeters(event);

  function setCircleCenter(event: ScenePointerEvent): boolean {
    const center = toMeters(event);
    if (!center) return false;
    circleCenter = center;
    circleCleared = false;
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
    ensureVariant();
  }

  function applyArea(): void {
    const polygon = areaPolygons.get(ctx.options.area);
    if (!polygon) return;
    vertices = [...polygon];
    circleCenter = getCentroid(vertices);
    circleCleared = false;
    if (lastScreenSpace) {
      lastScreenSpace = false;
      screenConversionStableFrames = 0;
      screenConversionThreshold = 1;
      ensureVariant();
    }
    cpuSignature = '';
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

  ctx.setReadout('points', formatCount(pointCount));
  ensureVariant();
  updateStatus();
  updateOptionReadouts();

  return {
    getCompiledGraphs: () =>
      (displayed ? [displayed.compiled] : []) as unknown as CompiledGraphList,

    setOption(id, _value, state) {
      if (id === 'area') {
        applyArea();
        drawingArmed = false;
        ctx.setMapDragEnabled(true);
        ctx.requestLayers();
      } else if (id === 'value') {
        writeValues();
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
        }
        ensureVariant();
        updateStatus();
        updateOptionReadouts();
        ctx.requestLayers();
      } else if (id === 'radius') {
        circleCleared = false;
      } else {
        ctx.requestLayers();
      }
      void state;
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
        circleCleared = true;
        rectangleCorners = null;
        pickCoordinate = null;
        cpuSignature = '';
        updateStatus();
      } else if (id === 'measure') {
        void measureGridIndex();
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      if (!displayed) return;
      if (autoMeasurePending && ++framesSinceBuild >= AUTO_MEASURE_FRAME) {
        autoMeasurePending = false;
        if (ctx.options.path === 'direct') void measureGridIndex();
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
      const dark = ctx.theme() === 'dark';
      const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
      const layers: Layer[] = [];
      if (options.showAreas) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'lasso-areas',
            coordinateOrigin,
            segments: areaSegmentBuffer,
            instanceCount: areaSegmentCount,
            widthPixels: 1,
            color: dark ? [180, 190, 210, 90] : [60, 70, 90, 90]
          })
        );
      }
      if (options.showPoints) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'lasso-all-points',
            coordinateOrigin,
            positions: positionsBuffer,
            instanceCount: pointCount,
            radiusPixels: 1,
            color: dark ? [130, 160, 210, 45] : [70, 100, 160, 55]
          })
        );
      }
      const selectedColor: [number, number, number, number] =
        options.path === 'direct' ? [255, 190, 60, 255] : [255, 110, 200, 255];
      if (options.path === 'direct' && options.selectedIds === 'ids') {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'lasso-selected-ids',
            coordinateOrigin,
            positions: positionsBuffer,
            drawCommands,
            ids: idsBuffer,
            radiusPixels: 1.6,
            color: [selectedColor[0], selectedColor[1], selectedColor[2], 200]
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
            radiusPixels: options.path === 'pick' ? 3 : 1.6,
            values: mask,
            valueFormat: 'uint32',
            colormap: 'mask',
            color: [selectedColor[0], selectedColor[1], selectedColor[2], 200],
            noDataColor: [0, 0, 0, 0]
          })
        );
      }
      layers.push(
        new SpatialAnalysisSegmentLayer({
          id: 'lasso-outline',
          coordinateOrigin,
          segments: outlineBuffer,
          instanceCount: OUTLINE_CAPACITY,
          widthPixels: 2.5,
          color: dark ? [90, 240, 220, 255] : [0, 140, 130, 255]
        })
      );
      return layers;
    },

    onClick(event) {
      if (ctx.options.path === 'pick') return setPickCoordinate(event);
      if (ctx.options.shape === 'radius') return setCircleCenter(event);
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
      }
    },

    destroy() {
      destroyed = true;
      clearTimeout(cpuTimer);
      extraReader.stop();
      ctx.setMapDragEnabled(true);
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

function getCentroid(polygon: readonly Point[]): Point {
  const sum = polygon.reduce((total, point) => [total[0] + point[0], total[1] + point[1]], [0, 0]);
  return [sum[0] / Math.max(polygon.length, 1), sum[1] / Math.max(polygon.length, 1)];
}

/** Reduces a ring to at most `limit` vertices by Douglas-Peucker with a growing tolerance. */
function simplifyRing(ring: Point[], limit: number): Point[] {
  const points =
    ring.length > 1 &&
    ring[0][0] === ring[ring.length - 1][0] &&
    ring[0][1] === ring[ring.length - 1][1]
      ? ring.slice(0, -1)
      : ring;
  if (points.length <= limit) return points;
  let tolerance = 5;
  let result = points;
  while (result.length > limit) {
    result = douglasPeucker(points, tolerance);
    tolerance *= 1.5;
  }
  return result;
}

function douglasPeucker(points: Point[], tolerance: number): Point[] {
  const keep = new Uint8Array(points.length);
  keep[0] = 1;
  keep[points.length - 1] = 1;
  const stack: [number, number][] = [[0, points.length - 1]];
  while (stack.length > 0) {
    const [start, end] = stack.pop()!;
    let maximum = 0;
    let index = -1;
    const [ax, ay] = points[start];
    const [bx, by] = points[end];
    const length = Math.hypot(bx - ax, by - ay) || 1;
    for (let candidate = start + 1; candidate < end; candidate++) {
      const distance =
        Math.abs(
          (bx - ax) * (ay - points[candidate][1]) - (ax - points[candidate][0]) * (by - ay)
        ) / length;
      if (distance > maximum) {
        maximum = distance;
        index = candidate;
      }
    }
    if (index >= 0 && maximum > tolerance) {
      keep[index] = 1;
      stack.push([start, index], [index, end]);
    }
  }
  return points.filter((_, index) => keep[index]);
}

function getPaddedBounds(positions: Float32Array): [number, number, number, number] {
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  for (let index = 0; index < positions.length; index += 2) {
    minimumX = Math.min(minimumX, positions[index]);
    maximumX = Math.max(maximumX, positions[index]);
    minimumY = Math.min(minimumY, positions[index + 1]);
    maximumY = Math.max(maximumY, positions[index + 1]);
  }
  const padding = 0.02 * Math.max(maximumX - minimumX, maximumY - minimumY, 1);
  return [minimumX - padding, minimumY - padding, maximumX + padding, maximumY + padding];
}

/** CPU even-odd point-in-polygon count, used only to cross-check the GPU summary. */
function countInsidePolygon(positions: Float32Array, polygon: readonly Point[]): number {
  if (polygon.length < 3) return 0;
  let count = 0;
  for (let row = 0; row < positions.length / 2; row++) {
    const x = positions[row * 2];
    const y = positions[row * 2 + 1];
    let inside = false;
    for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index++) {
      const [xi, yi] = polygon[index];
      const [xj, yj] = polygon[previous];
      if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
    }
    if (inside) count++;
  }
  return count;
}

function countInsideRectangle(positions: Float32Array, bounds: readonly number[]): number {
  let count = 0;
  for (let row = 0; row < positions.length / 2; row++) {
    const x = positions[row * 2];
    const y = positions[row * 2 + 1];
    if (x >= bounds[0] && x <= bounds[2] && y >= bounds[1] && y <= bounds[3]) count++;
  }
  return count;
}

function countInsideCircle(positions: Float32Array, center: Point, radius: number): number {
  let count = 0;
  for (let row = 0; row < positions.length / 2; row++) {
    if (Math.hypot(positions[row * 2] - center[0], positions[row * 2 + 1] - center[1]) <= radius)
      count++;
  }
  return count;
}
