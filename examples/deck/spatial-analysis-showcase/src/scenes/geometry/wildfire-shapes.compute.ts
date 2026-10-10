// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUShapeDescriptorsParameterValues,
  GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH,
  GPUGeometryMeasures,
  GPUShapeDescriptors
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {
  buildRingEdges,
  createGraphImporter,
  createStaticPaths,
  findFeatureAt,
  formatNumber,
  triangulatePolygons
} from './b3-common';
import {FeatureTriangleLayer, PathOutputLayer} from './b3-layers';
import {
  ACRES_PER_SQUARE_METER,
  describeFire,
  formatCorrelation,
  fromMercator,
  getFiniteQuantile,
  getSpearmanCorrelation,
  loadWildfires,
  SIZE_CLASS_NAMES,
  toMercator,
  YEAR_COLORS,
  type WildfireData
} from './wildfire-data';

/** What colors the fires. */
export type ShapeMetric =
  | 'year'
  | 'area'
  | 'perimeter'
  | 'areaDistortion'
  | 'vertices'
  | 'polsbyPopper'
  | 'schwartzberg'
  | 'elongation'
  | 'convexity'
  | 'sliver';

/** Option state of the wildfire-shapes scene. */
export type WildfireShapesOptions = {
  metric: ShapeMetric;
  areaSystem: 'planar' | 'spherical' | 'wgs84' | 'geodesic';
  holeRule: 'winding' | 'first-ring-exterior';
  largeRings: 'cooperative' | 'serial';
  convexityMethod: 'auto' | 'gift-wrapping' | 'monotone-chain';
  sliverThreshold: number;
  year: 'all' | '2020' | '2021' | '2022' | '2023';
  minAcres: number;
  opacity: number;
  showOutlines: boolean;
  showMarkers: boolean;
  showAxes: boolean;
  shapeView: 'map' | 'gallery' | 'detail';
};

type MetricSpec = {
  mapping: 'ramp' | 'category' | 'flag';
  /** Natural bounds the color range is clamped to. */
  natural?: readonly [number, number];
  sqrt?: boolean;
  /** Histogram the base-10 logarithm. */
  log?: boolean;
  label: string;
};

export const METRIC_SPECS: Record<ShapeMetric, MetricSpec> = {
  year: {mapping: 'category', label: 'Perimeter year'},
  area: {mapping: 'ramp', sqrt: true, log: true, label: 'Area (acres)'},
  perimeter: {mapping: 'ramp', log: true, label: 'Perimeter (km)'},
  areaDistortion: {mapping: 'ramp', label: 'Planar area / WGS84 area'},
  vertices: {mapping: 'ramp', sqrt: true, log: true, label: 'Vertices'},
  polsbyPopper: {mapping: 'ramp', natural: [0, 1], label: 'Polsby-Popper compactness'},
  schwartzberg: {mapping: 'ramp', natural: [0, 1], label: 'Schwartzberg compactness'},
  elongation: {mapping: 'ramp', natural: [0, 1], label: 'Elongation'},
  convexity: {mapping: 'ramp', natural: [0, 1], label: 'Convexity'},
  sliver: {mapping: 'flag', label: 'Sliver flag'}
};

const SYSTEMS = ['planar', 'spherical', 'wgs84', 'geodesic'] as const;
const NOT_SLIVER_COLOR = [150, 160, 175, 200] as const;
const SLIVER_COLOR = [226, 96, 80, 255] as const;
const YEAR_PALETTE = YEAR_COLORS.map(([r, g, b]) => [r, g, b, 255] as const);
const SIZE_CLASS_SHORT = ['<10k', '11-33k', '50-97k', '>300k'];

/** Per-fire columns read back once the controls settle. */
type FireTable = {
  area: Record<(typeof SYSTEMS)[number], Float32Array>;
  length: Record<(typeof SYSTEMS)[number], Float32Array>;
  vertices: Float32Array;
  polsbyPopper: Float32Array;
  schwartzberg: Float32Array;
  elongation: Float32Array;
  convexity: Float32Array;
  sliver: Float32Array;
  orientation: Float32Array;
  centroid: Float32Array;
};

/**
 * Wildfire shapes: `GPUGeometryMeasures` (area, perimeter, centroid, vertex count in four
 * coordinate systems) and `GPUShapeDescriptors` (compactness, elongation, convexity, orientation,
 * sliver flag) over 90 western fires. The per-fire columns are bound straight into the fill,
 * outline and marker layers; a small table is read back for the charts and readouts.
 */
export async function createWildfireShapes(
  ctx: SceneContext<WildfireShapesOptions>
): Promise<SceneInstance<WildfireShapesOptions>> {
  const {device} = ctx;
  const data: WildfireData = loadWildfires(ctx.datasets.get('poopdeck-wildfires'));
  const {layout, count} = data;
  const {ringCount, vertexCount} = layout;
  const resources = new SpatialAnalysisResources(device, 'wildfire-shapes');
  let destroyed = false;

  let galleryRows: number[] = [];
  const galleryCentres: [number, number][] = [];
  let galleryPaths: ReturnType<typeof createStaticPaths> | null = null;
  const makeGalleryPaths = (rows: readonly number[]) => {
    const positions: number[] = [];
    const offsets = [0];
    galleryCentres.length = 0;
    for (let galleryIndex = 0; galleryIndex < rows.length; galleryIndex++) {
      const fire = rows[galleryIndex];
      const [west, south, east, north] = layout.featureBounds.subarray(fire * 4, fire * 4 + 4);
      const allBounds = layout.featureBounds;
      let allWest = Infinity,
        allSouth = Infinity,
        allEast = -Infinity,
        allNorth = -Infinity;
      for (let row = 0; row < count; row++) {
        allWest = Math.min(allWest, allBounds[row * 4]);
        allSouth = Math.min(allSouth, allBounds[row * 4 + 1]);
        allEast = Math.max(allEast, allBounds[row * 4 + 2]);
        allNorth = Math.max(allNorth, allBounds[row * 4 + 3]);
      }
      const centreLongitude = allWest + ((galleryIndex % 4) + 0.5) * ((allEast - allWest) / 4);
      const centreLatitude =
        allNorth - (Math.floor(galleryIndex / 4) + 0.5) * ((allNorth - allSouth) / 3);
      galleryCentres.push([centreLongitude, centreLatitude]);
      const scale = 0.7 / Math.max(east - west, north - south, 1e-6);
      const sourceLongitude = (west + east) / 2;
      const sourceLatitude = (south + north) / 2;
      for (
        let ring = layout.featureRingOffsets[fire];
        ring < layout.featureRingOffsets[fire + 1];
        ring++
      ) {
        for (
          let vertex = layout.ringOffsets[ring];
          vertex < layout.ringOffsets[ring + 1];
          vertex++
        ) {
          positions.push(
            centreLongitude + (layout.lngLat[vertex * 2] - sourceLongitude) * scale,
            centreLatitude + (layout.lngLat[vertex * 2 + 1] - sourceLatitude) * scale
          );
        }
        offsets.push(positions.length / 2);
      }
    }
    return createStaticPaths(
      resources,
      'equal-size-gallery',
      Float32Array.from(positions),
      Uint32Array.from(offsets)
    );
  };
  let detailRing = 0;
  for (let ring = 1; ring < ringCount; ring++) {
    if (
      layout.ringOffsets[ring + 1] - layout.ringOffsets[ring] >
      layout.ringOffsets[detailRing + 1] - layout.ringOffsets[detailRing]
    ) {
      detailRing = ring;
    }
  }
  const detailStart = layout.ringOffsets[detailRing];
  const detailEnd = layout.ringOffsets[detailRing + 1];
  const detailOriginal = createStaticPaths(
    resources,
    'detail-original',
    layout.lngLat.slice(detailStart * 2, detailEnd * 2),
    Uint32Array.of(0, detailEnd - detailStart)
  );
  // This archive has multipart rings and no prepared topology-safe simplification hierarchy.
  // Detail is therefore the honest vertices-per-kilometre fallback, not display decimation.
  ctx.setReadout('detailOriginalVertices', formatCount(detailEnd - detailStart));

  // ---- Static inputs and drawing data ------------------------------------------------------------
  const lngLatBuffer = resources.createBuffer('lnglat', layout.lngLat);
  const planarBuffer = resources.createBuffer('planar', data.mercator);
  const ringOffsetsBuffer = resources.createBuffer('ring-offsets', layout.ringOffsets);
  const featureRingsBuffer = resources.createBuffer('feature-rings', layout.featureRingOffsets);
  const triangulated = triangulatePolygons(layout);
  const fill = {
    corners: resources.createBuffer('fill-corners', triangulated.corners),
    featureRows: resources.createBuffer('fill-feature-rows', triangulated.featureRows),
    triangleCount: triangulated.triangleCount
  };
  const edgeColumns = buildRingEdges(layout);
  const edgeSegments = new Float32Array(edgeColumns.edgeCount * 4);
  for (let edge = 0; edge < edgeColumns.edgeCount; edge++) {
    edgeSegments.set(edgeColumns.starts.subarray(edge * 2, edge * 2 + 2), edge * 4);
    edgeSegments.set(edgeColumns.ends.subarray(edge * 2, edge * 2 + 2), edge * 4 + 2);
  }
  const edges = {
    segments: resources.createBuffer('edge-segments', edgeSegments),
    featureRows: resources.createBuffer('edge-feature-rows', edgeColumns.featureRows),
    edgeCount: edgeColumns.edgeCount
  };
  let maximumFireEdges = 1;
  const edgesPerFire = new Uint32Array(count);
  for (const fire of edgeColumns.featureRows) edgesPerFire[fire]++;
  for (let fire = 0; fire < count; fire++)
    maximumFireEdges = Math.max(maximumFireEdges, edgesPerFire[fire]);
  const selectionSegments = resources.createBuffer('selection-segments', maximumFireEdges * 16);
  const axisSegments = resources.createBuffer('axis-segments', count * 16);
  const yearFloat = resources.createBuffer('year-f', Float32Array.from(data.yearIndex));
  const visibleBuffer = resources.createBuffer('visible', count * 4);

  // ---- Per-fire output columns ------------------------------------------------------------------
  const makeColumn = (name: string, length = count) =>
    resources.createBuffer(name, Math.max(length, 1) * 4);
  const area = Object.fromEntries(
    SYSTEMS.map(system => [system, makeColumn(`area-${system}`)])
  ) as Record<(typeof SYSTEMS)[number], Buffer>;
  const length = Object.fromEntries(
    SYSTEMS.map(system => [system, makeColumn(`length-${system}`)])
  ) as Record<(typeof SYSTEMS)[number], Buffer>;
  const centroids = makeColumn('centroids', count * 2);
  const vertexCounts = makeColumn('vertex-counts');
  const columns = {
    ratio: makeColumn('ratio'),
    vertexF: makeColumn('vertex-f'),
    polsbyPopper: makeColumn('polsby-popper'),
    schwartzberg: makeColumn('schwartzberg'),
    elongation: makeColumn('elongation'),
    convexity: makeColumn('convexity'),
    sliverF: makeColumn('sliver-f'),
    orientation: makeColumn('orientation')
  };
  const clockwise = makeColumn('clockwise');
  const sliver = makeColumn('sliver');
  const display = makeColumn('display');
  const displayCategory = makeColumn('display-category');
  // One value per authored PP class. These representatives make the YlOrRd ramp discrete while
  // retaining its fixed low-PP-is-strongest ordering in the fill and legend.
  const compactnessClass = makeColumn('compactness-class');
  const shapeParameters = resources.createParameterBuffer(
    'shape-parameters',
    'float32',
    GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH,
    getGPUShapeDescriptorsParameterValues({sliverThreshold: ctx.options.sliverThreshold})
  );

  // ---- Graphs -----------------------------------------------------------------------------------
  const buildMeasures = (options: WildfireShapesOptions): CompiledGPUCommandGraph<void> => {
    const graph = new GPUCommandGraph<void>(device, {id: 'wildfire-measures'});
    const imp = createGraphImporter(graph);
    // Spread objects: the props exist in the contributor source; spreading keeps tsc quiet when the
    // package's generated declarations lag behind.
    const largeRingProps = options.largeRings === 'serial' ? {cooperativeRingRows: 0} : {};
    const lngLat = imp('lnglat', lngLatBuffer, 'float32x2', vertexCount);
    const planar = imp('planar', planarBuffer, 'float32x2', vertexCount);
    const ringOffsets = imp('ring-offsets', ringOffsetsBuffer, 'uint32', ringCount + 1);
    const featureRings = imp('feature-rings', featureRingsBuffer, 'uint32', count + 1);
    const view = (name: string, buffer: Buffer, rows = count) => imp(name, buffer, 'float32', rows);
    for (const system of SYSTEMS) {
      graph.add(
        new GPUGeometryMeasures({
          spatialContext: {
            coordinateSpace: system === 'planar' ? 'planar' : 'longitude-latitude',
            metric:
              system === 'planar'
                ? 'native'
                : system === 'spherical'
                  ? 'great-circle'
                  : 'ellipsoidal',
            units: system === 'planar' ? 'native' : 'meters'
          },
          ellipsoidalEdgeModel: system === 'wgs84' ? 'coordinate-linear' : undefined,
          id: `measures-${system}`,
          positions: system === 'planar' ? planar : lngLat,
          ringOffsets,
          featureRingOffsets: featureRings,
          geometryType: 'polygons',

          holeRule: options.holeRule,
          ...largeRingProps,
          output: {
            areas: view(`area-${system}`, area[system]),
            lengths: view(`length-${system}`, length[system]),
            ...(system === 'wgs84'
              ? {
                  centroids: imp('centroids', centroids, 'float32x2', count),
                  vertexCounts: imp('vertex-counts', vertexCounts, 'uint32', count)
                }
              : {})
          }
        })
      );
    }
    addKernelPass(graph, {
      id: 'area-distortion',
      invocationCount: count,
      bindings: [
        {name: 'planar', view: view('area-planar', area.planar), type: 'f32', access: 'read'},
        {name: 'ellipsoid', view: view('area-wgs84', area.wgs84), type: 'f32', access: 'read'},
        {name: 'ratio', view: view('ratio', columns.ratio), type: 'f32', access: 'read_write'}
      ],
      body: /* wgsl */ `ratio[ratioOffset + index] = planar[planarOffset + index] / max(ellipsoid[ellipsoidOffset + index], 1e-9);`
    });
    addKernelPass(graph, {
      id: 'vertex-count-to-float',
      invocationCount: count,
      bindings: [
        {
          name: 'counts',
          view: imp('vertex-counts', vertexCounts, 'uint32', count),
          type: 'u32',
          access: 'read'
        },
        {name: 'values', view: view('vertex-f', columns.vertexF), type: 'f32', access: 'read_write'}
      ],
      body: /* wgsl */ `values[valuesOffset + index] = f32(counts[countsOffset + index]);`
    });
    return resources.track(graph.compile());
  };

  const buildShape = (options: WildfireShapesOptions): CompiledGPUCommandGraph<void> => {
    const graph = new GPUCommandGraph<void>(device, {id: 'wildfire-shape'});
    const imp = createGraphImporter(graph);
    const convexityProps = {convexityMethod: options.convexityMethod};
    const view = (name: string, buffer: Buffer, rows = count) => imp(name, buffer, 'float32', rows);
    graph.add(
      new GPUShapeDescriptors({
        id: 'shape',
        positions: imp('planar', planarBuffer, 'float32x2', vertexCount),
        ringOffsets: imp('ring-offsets', ringOffsetsBuffer, 'uint32', ringCount + 1),
        featureRingOffsets: imp('feature-rings', featureRingsBuffer, 'uint32', count + 1),
        holeRule: options.holeRule,
        ...convexityProps,
        parameters: shapeParameters.importToGraph(graph),
        output: {
          polsbyPopper: view('polsby-popper', columns.polsbyPopper),
          schwartzberg: view('schwartzberg', columns.schwartzberg),
          elongation: view('elongation', columns.elongation),
          orientation: view('orientation', columns.orientation),
          convexity: view('convexity', columns.convexity),
          clockwise: imp('clockwise', clockwise, 'uint32', count),
          sliver: imp('sliver', sliver, 'uint32', count)
        }
      })
    );
    addKernelPass(graph, {
      id: 'sliver-to-float',
      invocationCount: count,
      bindings: [
        {name: 'flags', view: imp('sliver', sliver, 'uint32', count), type: 'u32', access: 'read'},
        {name: 'values', view: view('sliver-f', columns.sliverF), type: 'f32', access: 'read_write'}
      ],
      body: /* wgsl */ `values[valuesOffset + index] = f32(flags[flagsOffset + index]);`
    });
    return resources.track(graph.compile());
  };

  // Hides filtered fires: NaN for the float display column, the no-data word for the category column.
  const maskGraphSource = new GPUCommandGraph<void>(device, {id: 'wildfire-display-mask'});
  {
    const imp = createGraphImporter(maskGraphSource);
    addKernelPass(maskGraphSource, {
      id: 'mask-display',
      invocationCount: count,
      bindings: [
        {
          name: 'display',
          view: imp('display', display, 'float32', count),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'categories',
          view: imp('display-category', displayCategory, 'uint32', count),
          type: 'u32',
          access: 'read_write'
        },
        {
          name: 'polsbyPopper',
          view: imp('polsby-popper', columns.polsbyPopper, 'float32', count),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'compactnessClass',
          view: imp('compactness-class', compactnessClass, 'float32', count),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'visible',
          view: imp('visible', visibleBuffer, 'float32', count),
          type: 'f32',
          access: 'read'
        }
      ],
      body: /* wgsl */ `
  let value = display[displayOffset + index];
  var nanBits = 0x7fc00000u;
  if (visible[visibleOffset + index] < 0.5) {
    display[displayOffset + index] = bitcast<f32>(nanBits);
    categories[categoriesOffset + index] = 0xffffffffu;
    compactnessClass[compactnessClassOffset + index] = bitcast<f32>(nanBits);
  } else {
    categories[categoriesOffset + index] = u32(max(value, 0.0));
    let pp = polsbyPopper[polsbyPopperOffset + index];
    compactnessClass[compactnessClassOffset + index] = select(
      select(select(select(0.025, 0.10, pp >= 0.05), 0.225, pp >= 0.15), 0.40, pp >= 0.30),
      0.60, pp >= 0.50
    );
  }`
    });
  }
  const maskGraph = resources.track(maskGraphSource.compile());
  let measures = buildMeasures(ctx.options);
  let shape = buildShape(ctx.options);

  // ---- Readback table ---------------------------------------------------------------------------
  const tableSources: {buffer: Buffer; size: number}[] = [
    ...SYSTEMS.map(system => ({buffer: area[system], size: count * 4})),
    ...SYSTEMS.map(system => ({buffer: length[system], size: count * 4})),
    {buffer: columns.vertexF, size: count * 4},
    {buffer: columns.polsbyPopper, size: count * 4},
    {buffer: columns.schwartzberg, size: count * 4},
    {buffer: columns.elongation, size: count * 4},
    {buffer: columns.convexity, size: count * 4},
    {buffer: columns.sliverF, size: count * 4},
    {buffer: columns.orientation, size: count * 4},
    {buffer: centroids, size: count * 8}
  ];
  let table: FireTable | null = null;
  let currentRange: [number, number] = [0, 1];
  let selected = -1;
  let selectedEdgeCount = 0;
  let axisCount = 0;
  const visible = new Float32Array(count);
  const dirty = {measures: true, shape: true, display: true, table: true};
  let lastChange = performance.now();

  const readTable = (bytes: ArrayBuffer): FireTable => {
    let offset = 0;
    const next = (rows: number) => {
      const out = new Float32Array(bytes, offset, rows);
      offset += rows * 4;
      return out;
    };
    const areaColumns = {} as FireTable['area'];
    const lengthColumns = {} as FireTable['length'];
    for (const system of SYSTEMS) areaColumns[system] = next(count);
    for (const system of SYSTEMS) lengthColumns[system] = next(count);
    return {
      area: areaColumns,
      length: lengthColumns,
      vertices: next(count),
      polsbyPopper: next(count),
      schwartzberg: next(count),
      elongation: next(count),
      convexity: next(count),
      sliver: next(count),
      orientation: next(count),
      centroid: next(count * 2)
    };
  };

  const updateVisibility = () => {
    const {year, minAcres} = ctx.options;
    for (let fire = 0; fire < count; fire++) {
      const matchesYear = year === 'all' || String(data.year[fire]) === year;
      visible[fire] = matchesYear && data.acres[fire] >= minAcres ? 1 : 0;
    }
    visibleBuffer.write(visible);
  };

  const getMetricBuffer = (options: WildfireShapesOptions): Buffer => {
    switch (options.metric) {
      case 'year':
        return yearFloat;
      case 'area':
        return area[options.areaSystem];
      case 'perimeter':
        return length[options.areaSystem];
      case 'areaDistortion':
        return columns.ratio;
      case 'vertices':
        return columns.vertexF;
      case 'polsbyPopper':
        return columns.polsbyPopper;
      case 'schwartzberg':
        return columns.schwartzberg;
      case 'elongation':
        return columns.elongation;
      case 'convexity':
        return columns.convexity;
      case 'sliver':
        return columns.sliverF;
    }
  };

  const getMetricValues = (t: FireTable, options: WildfireShapesOptions): ArrayLike<number> => {
    switch (options.metric) {
      case 'year':
        return Float32Array.from(data.yearIndex);
      case 'area':
        return t.area[options.areaSystem];
      case 'perimeter':
        return t.length[options.areaSystem];
      case 'areaDistortion':
        return Float32Array.from(
          t.area.planar,
          (value, fire) => value / Math.max(t.area.wgs84[fire], 1e-9)
        );
      case 'vertices':
        return t.vertices;
      case 'polsbyPopper':
        return t.polsbyPopper;
      case 'schwartzberg':
        return t.schwartzberg;
      case 'elongation':
        return t.elongation;
      case 'convexity':
        return t.convexity;
      case 'sliver':
        return t.sliver;
    }
  };

  /** Median-spaced groups of visible fires by area, for the trend chart. */
  const buildCharts = () => {
    if (!table) return;
    const options = ctx.options;
    const spec = METRIC_SPECS[options.metric];
    const values = getMetricValues(table, options);
    const shown: number[] = [];
    for (let fire = 0; fire < count; fire++) if (visible[fire] > 0.5) shown.push(fire);
    const shownValues = shown.map(fire => values[fire]);
    // Legend range: natural bounds, else a robust 2 to 98 percent range of what is visible.
    if (spec.mapping === 'ramp') {
      const scale = options.metric === 'perimeter' ? 1 / 1000 : 1;
      const acreScale = options.metric === 'area' ? ACRES_PER_SQUARE_METER : 1;
      const low = getFiniteQuantile(shownValues, 0.02) * scale * acreScale;
      const high = getFiniteQuantile(shownValues, 0.98) * scale * acreScale;
      currentRange = spec.natural
        ? [
            Math.max(spec.natural[0], Math.min(low, high)),
            Math.min(spec.natural[1], Math.max(low, high))
          ]
        : [Math.min(spec.sqrt ? 0 : low, high), Math.max(high, low + 1e-6)];
      if (!(currentRange[1] > currentRange[0]))
        currentRange = spec.natural ? [...spec.natural] : [0, 1];
      ctx.setLegendExtent('value', currentRange);
    }
    // Chart 1: distribution of the shown metric.
    if (options.metric === 'year') {
      const counts = [0, 0, 0, 0];
      for (const fire of shown) counts[data.yearIndex[fire]]++;
      ctx.setChart('metricChart', {
        kind: 'bars',
        values: counts,
        labels: ['2020', '2021', '2022', '2023'],
        yLabel: 'fires',
        description: 'Number of shown fires per perimeter year'
      });
    } else if (options.metric === 'sliver') {
      const slivers = shown.filter(fire => table!.sliver[fire] > 0.5).length;
      ctx.setChart('metricChart', {
        kind: 'bars',
        values: [shown.length - slivers, slivers],
        labels: ['not sliver', 'sliver'],
        highlight: [1],
        yLabel: 'fires',
        description: 'Fires flagged as slivers by Polsby-Popper below the threshold'
      });
    } else {
      const scale =
        options.metric === 'perimeter'
          ? 1 / 1000
          : options.metric === 'area'
            ? ACRES_PER_SQUARE_METER
            : 1;
      const transformed = shownValues
        .map(value => value * scale)
        .filter(value => Number.isFinite(value) && (!spec.log || value > 0))
        .map(value => (spec.log ? Math.log10(value) : value));
      if (transformed.length) {
        const low = spec.natural?.[0] ?? Math.min(...transformed);
        const high = spec.natural?.[1] ?? Math.max(...transformed);
        const bins = new Array(16).fill(0);
        for (const value of transformed) {
          const bin = Math.min(
            15,
            Math.max(0, Math.floor(((value - low) / Math.max(high - low, 1e-9)) * 16))
          );
          bins[bin]++;
        }
        ctx.setChart('metricChart', {
          kind: 'histogram',
          values: bins,
          xDomain: [low, high],
          xLabel: spec.log ? `${spec.label}, log10` : spec.label,
          yLabel: 'fires',
          description: `Distribution of ${spec.label} across the shown fires`
        });
      } else {
        ctx.setChart('metricChart', null);
      }
    }
    // Chart 2: compactness against size, in equal-count groups of fires ordered by area.
    const acresOf = (fire: number) => table!.area.wgs84[fire] * ACRES_PER_SQUARE_METER;
    const bySize = shown
      .filter(fire => acresOf(fire) > 0 && Number.isFinite(table!.polsbyPopper[fire]))
      .sort((a, b) => acresOf(a) - acresOf(b));
    const groupCount = Math.min(6, Math.floor(bySize.length / 3));
    if (groupCount >= 2) {
      const x: number[] = [];
      const median: number[] = [];
      const low: number[] = [];
      const high: number[] = [];
      for (let group = 0; group < groupCount; group++) {
        const members = bySize.slice(
          Math.floor((group * bySize.length) / groupCount),
          Math.floor(((group + 1) * bySize.length) / groupCount)
        );
        const sizes = members.map(fire => Math.log10(acresOf(fire)));
        const compactness = members.map(fire => table!.polsbyPopper[fire]);
        x.push(getFiniteQuantile(sizes, 0.5));
        median.push(getFiniteQuantile(compactness, 0.5));
        low.push(getFiniteQuantile(compactness, 0.25));
        high.push(getFiniteQuantile(compactness, 0.75));
      }
      ctx.setChart('trendChart', {
        kind: 'line',
        series: [{label: 'median compactness', x, y: median}],
        band: {x, low, high, label: 'middle half of fires'},
        xLabel: 'fire size (acres)',
        yLabel: 'Polsby-Popper',
        formatX: value => formatNumber(10 ** value),
        yDomain: [0, Math.max(0.2, ...high) * 1.1],
        description:
          'Median Polsby-Popper compactness against fire size, in equal-count size groups'
      });
    } else {
      ctx.setChart('trendChart', null);
    }
    // Chart 3: median compactness by acreage class.
    const classValues = SIZE_CLASS_NAMES.map((_, sizeClass) =>
      getFiniteQuantile(
        shown
          .filter(fire => data.sizeClass[fire] === sizeClass)
          .map(fire => table!.polsbyPopper[fire]),
        0.5
      )
    );
    ctx.setChart('classChart', {
      kind: 'bars',
      values: classValues.map(value => (Number.isFinite(value) ? value : 0)),
      labels: SIZE_CLASS_SHORT,
      highlight: classValues.some(Number.isFinite)
        ? [classValues.indexOf(Math.min(...classValues.filter(Number.isFinite)))]
        : [],
      yLabel: 'median PP',
      description: 'Median Polsby-Popper compactness per acreage class (poopdeck severity column)'
    });
    // Readouts.
    const shownAcres = shown.reduce((sum, fire) => sum + acresOf(fire), 0);
    const nifcAcres = shown.reduce((sum, fire) => sum + data.acres[fire], 0);
    const ratios = shown.map(fire => acresOf(fire) / data.acres[fire]);
    ctx.setReadout('fireCount', shown.length);
    ctx.setReadout('gpuAcres', shownAcres);
    ctx.setReadout('nifcAcres', nifcAcres);
    ctx.setReadout(
      'medianRatio',
      Number.isFinite(getFiniteQuantile(ratios, 0.5))
        ? getFiniteQuantile(ratios, 0.5).toFixed(2)
        : 'n/a'
    );
    let worstFire = -1;
    let worstDeviation = 0;
    shown.forEach((fire, index) => {
      const deviation = Math.abs(ratios[index] - 1);
      if (deviation > worstDeviation) {
        worstDeviation = deviation;
        worstFire = fire;
      }
    });
    ctx.setReadout(
      'worstError',
      worstFire < 0
        ? 'none'
        : `${data.names[worstFire]}, ${ratios[shown.indexOf(worstFire)] >= 1 ? '+' : '-'}${(worstDeviation * 100).toFixed(worstDeviation < 0.1 ? 1 : 0)}%`
    );
    ctx.setReadout(
      'medianCompactness',
      getFiniteQuantile(
        shown.map(fire => table!.polsbyPopper[fire]),
        0.5
      ).toFixed(2)
    );
    const areas = shown.map(acresOf);
    ctx.setReadout(
      'areaCompactness',
      formatCorrelation(
        getSpearmanCorrelation(
          areas,
          shown.map(fire => table!.polsbyPopper[fire])
        )
      )
    );
    ctx.setReadout(
      'areaVertices',
      formatCorrelation(
        getSpearmanCorrelation(
          areas,
          shown.map(fire => table!.vertices[fire])
        )
      )
    );
    const detailDensities = shown
      .map(fire => table!.vertices[fire] / Math.max(table!.length.wgs84[fire] / 1000, 1e-6))
      .filter(Number.isFinite);
    ctx.setReadout(
      'detailDensity',
      detailDensities.length
        ? `${getFiniteQuantile(detailDensities, 0.5).toFixed(1)} median vertices/km`
        : 'n/a'
    );
    ctx.setReadout(
      'areaElongation',
      formatCorrelation(
        getSpearmanCorrelation(
          areas,
          shown.map(fire => table!.elongation[fire])
        )
      )
    );
    ctx.requestLayers();
  };

  const writeAxes = () => {
    if (!table) return;
    const segments = new Float32Array(count * 4).fill(Number.NaN);
    axisCount = count;
    for (let fire = 0; fire < count; fire++) {
      const longitude = table.centroid[fire * 2];
      const latitude = table.centroid[fire * 2 + 1];
      const planarArea = table.area.planar[fire];
      if (!Number.isFinite(longitude) || !(planarArea > 0)) continue;
      const [x, y] = toMercator(longitude, latitude);
      // Half-length: the radius of a circle of the fire's planar area, stretched by elongation.
      const half = Math.sqrt(planarArea / Math.PI) * (1 + 1.5 * table.elongation[fire]);
      const angle = table.orientation[fire];
      const a = fromMercator(x - Math.cos(angle) * half, y - Math.sin(angle) * half);
      const b = fromMercator(x + Math.cos(angle) * half, y + Math.sin(angle) * half);
      segments.set([a[0], a[1], b[0], b[1]], fire * 4);
    }
    axisSegments.write(segments);
  };

  const selectFire = (fire: number) => {
    selected = fire;
    selectedEdgeCount = 0;
    if (fire >= 0) {
      const segments: number[] = [];
      for (let edge = 0; edge < edges.edgeCount; edge++) {
        if (edgeColumns.featureRows[edge] === fire) {
          segments.push(
            edgeColumns.starts[edge * 2],
            edgeColumns.starts[edge * 2 + 1],
            edgeColumns.ends[edge * 2],
            edgeColumns.ends[edge * 2 + 1]
          );
        }
      }
      selectionSegments.write(Float32Array.from(segments));
      selectedEdgeCount = segments.length / 4;
    }
    if (fire < 0 || !table) {
      ctx.setReadout('selected', fire < 0 ? 'Click a fire' : describeFire(data, fire));
    } else {
      const acres = table.area.wgs84[fire] * ACRES_PER_SQUARE_METER;
      ctx.setReadout(
        'selected',
        [
          describeFire(data, fire),
          `parts ${data.partCount[fire]}, vertices ${formatCount(table.vertices[fire])}`,
          `GPU area ${formatNumber(acres)} acres, perimeter ${(table.length[ctx.options.areaSystem][fire] / 1000).toFixed(1)} km`,
          `compactness ${table.polsbyPopper[fire].toFixed(3)}, elongation ${table.elongation[fire].toFixed(2)}`,
          `convexity ${table.convexity[fire].toFixed(2)}, axis ${((table.orientation[fire] * 180) / Math.PI).toFixed(0)} deg`
        ].join('\n')
      );
    }
    ctx.requestLayers();
  };

  const tableReader = new SummaryReader(resources, 'wildfire-table', tableSources, bytes => {
    if (destroyed) return;
    table = readTable(bytes);
    const shown = Array.from({length: count}, (_, fire) => fire).filter(
      fire => visible[fire] > 0.5
    );
    if (shown.length) {
      const ordered = shown.sort(
        (left, right) => table!.polsbyPopper[left] - table!.polsbyPopper[right]
      );
      galleryRows = [...ordered.slice(0, 6), ...ordered.slice(-6).reverse()];
      galleryPaths = makeGalleryPaths(galleryRows);
    }
    writeAxes();
    buildCharts();
    if (selected >= 0) selectFire(selected);
  });

  updateVisibility();
  ctx.setReadout('selected', 'Click a fire');

  return {
    getCompiledGraphs: () => [measures, shape, maskGraph],

    setOption(id, _value, options) {
      lastChange = performance.now();
      if (id === 'holeRule' || id === 'largeRings') {
        resources.release(measures);
        measures = buildMeasures(options);
        dirty.measures = true;
        if (id === 'holeRule') {
          resources.release(shape);
          shape = buildShape(options);
          dirty.shape = true;
        }
      } else if (id === 'convexityMethod') {
        resources.release(shape);
        shape = buildShape(options);
        dirty.shape = true;
      } else if (id === 'sliverThreshold') {
        shapeParameters.write(
          getGPUShapeDescriptorsParameterValues({sliverThreshold: options.sliverThreshold})
        );
        dirty.shape = true;
      } else if (id === 'year' || id === 'minAcres') {
        updateVisibility();
        galleryPaths = null;
      }
      dirty.display = true;
      dirty.table = true;
      ctx.requestLayers();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip(event) {
      if (ctx.options.shapeView === 'gallery' && event.coordinate && table && galleryPaths) {
        let galleryIndex = -1;
        let closest = Infinity;
        for (let index = 0; index < galleryCentres.length; index++) {
          const distance = Math.hypot(
            event.coordinate[0] - galleryCentres[index][0],
            event.coordinate[1] - galleryCentres[index][1]
          );
          if (distance < closest) {
            closest = distance;
            galleryIndex = index;
          }
        }
        if (galleryIndex >= 0 && closest < 0.9) {
          const fire = galleryRows[galleryIndex];
          return `${data.names[fire]} · ${formatNumber(data.acres[fire])} acres · ${data.partCount[fire]} parts · PP ${table.polsbyPopper[fire].toFixed(3)}`;
        }
        return null;
      }
      if (!event.coordinate) return null;
      const fire = findFeatureAt(layout, event.coordinate[0], event.coordinate[1]);
      if (fire < 0 || visible[fire] < 0.5) return null;
      if (!table) return describeFire(data, fire);
      return `${data.names[fire]} (${data.year[fire]}) · ${formatNumber(table.area.wgs84[fire] * ACRES_PER_SQUARE_METER)} acres · compactness ${table.polsbyPopper[fire].toFixed(2)}`;
    },

    onClick(event) {
      if (!event.coordinate) return false;
      if (ctx.options.shapeView === 'gallery' && galleryPaths) {
        let galleryIndex = -1;
        let closest = Infinity;
        for (let index = 0; index < galleryCentres.length; index++) {
          const distance = Math.hypot(
            event.coordinate[0] - galleryCentres[index][0],
            event.coordinate[1] - galleryCentres[index][1]
          );
          if (distance < closest) {
            closest = distance;
            galleryIndex = index;
          }
        }
        if (galleryIndex >= 0 && closest < 0.9) {
          const fire = galleryRows[galleryIndex];
          selectFire(fire === selected ? -1 : fire);
          return true;
        }
        return false;
      }
      const fire = findFeatureAt(layout, event.coordinate[0], event.coordinate[1]);
      selectFire(fire === selected || fire < 0 || visible[fire] < 0.5 ? -1 : fire);
      return true;
    },

    encode(commandEncoder) {
      if (dirty.measures) {
        measures.encode(commandEncoder, {parameters: undefined});
        dirty.measures = false;
      }
      if (dirty.shape) {
        shape.encode(commandEncoder, {parameters: undefined});
        dirty.shape = false;
      }
      if (dirty.display) {
        commandEncoder.copyBufferToBuffer({
          sourceBuffer: getMetricBuffer(ctx.options),
          destinationBuffer: display,
          size: count * 4
        });
        maskGraph.encode(commandEncoder, {parameters: undefined});
        dirty.display = false;
      }
      if (dirty.table && performance.now() - lastChange > 120) {
        tableReader.request(commandEncoder);
        dirty.table = false;
      } else {
        tableReader.flush(commandEncoder);
      }
    },

    getLayers() {
      const options = ctx.options;
      const spec = METRIC_SPECS[options.metric];
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      const opacity = Math.round(options.opacity * 255);
      if (options.shapeView === 'gallery') {
        if (!galleryPaths) return [];
        return [
          new PathOutputLayer({
            id: 'equal-size-silhouettes',
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            positions: galleryPaths.positions,
            pathOffsets: galleryPaths.offsets,
            pathOffsetCount: galleryPaths.offsetCount,
            vertexCount: galleryPaths.vertexCount,
            drawCommands: galleryPaths.drawCommands,
            color: dark ? [229, 94, 0, 255] : [154, 52, 18, 255],
            widthPixels: 1.6
          })
        ];
      }
      if (options.shapeView === 'detail') {
        return [
          new PathOutputLayer({
            id: 'detail-original',
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            positions: detailOriginal.positions,
            pathOffsets: detailOriginal.offsets,
            pathOffsetCount: detailOriginal.offsetCount,
            vertexCount: detailOriginal.vertexCount,
            drawCommands: detailOriginal.drawCommands,
            color: dark ? [154, 165, 177, 180] : [90, 100, 120, 180],
            widthPixels: 1.2
          })
        ];
      }
      layers.push(
        new FeatureTriangleLayer({
          id: 'fire-fill',
          coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
          corners: fill.corners,
          featureRows: fill.featureRows,
          instanceCount: fill.triangleCount,
          values: options.metric === 'polsbyPopper' ? compactnessClass : display,
          valueMapping: spec.mapping,
          colormap: 'ylorrd',
          valueRange: options.metric === 'polsbyPopper' ? [0.6, 0] : currentRange,
          sqrtScale: Boolean(spec.sqrt),
          color:
            spec.mapping === 'flag'
              ? [SLIVER_COLOR[0], SLIVER_COLOR[1], SLIVER_COLOR[2], opacity]
              : [255, 255, 255, opacity],
          secondaryColor: [
            NOT_SLIVER_COLOR[0],
            NOT_SLIVER_COLOR[1],
            NOT_SLIVER_COLOR[2],
            Math.round(opacity * 0.6)
          ]
        })
      );
      if (options.showOutlines) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'fire-outline',
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            segments: edges.segments,
            instanceCount: edges.edgeCount,
            valueIndices: edges.featureRows,
            values: visibleBuffer,
            valueFormat: 'float32',
            colormap: 'mask',
            color: dark ? [235, 240, 248, 150] : [40, 52, 70, 150],
            noDataColor: [0, 0, 0, 0],
            widthPixels: 1
          })
        );
      }
      if (options.showAxes && axisCount > 0) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'fire-axes',
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            segments: axisSegments,
            instanceCount: axisCount,
            color: [78, 168, 222, 255],
            widthPixels: 2
          })
        );
      }
      if (options.showMarkers) {
        const categorical = spec.mapping !== 'ramp';
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'fire-markers',
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            positions: centroids,
            instanceCount: count,
            radiusPixels: 4.5,
            values: categorical ? displayCategory : display,
            valueFormat: categorical ? 'uint32' : 'float32',
            colormap: categorical ? 'category' : 'ylorrd',
            palette: options.metric === 'sliver' ? [NOT_SLIVER_COLOR, SLIVER_COLOR] : YEAR_PALETTE,
            valueRange: currentRange,
            sqrtScale: Boolean(spec.sqrt),
            noDataColor: [0, 0, 0, 0]
          })
        );
      }
      if (selected >= 0 && selectedEdgeCount > 0) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'fire-selection',
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            segments: selectionSegments,
            instanceCount: selectedEdgeCount,
            color: [255, 214, 90, 255],
            widthPixels: 3
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      tableReader.stop();
      resources.destroy();
    }
  };
}
