// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import type {Buffer, CommandEncoder} from '@luma.gl/core';
import {
  getGPUGridGeneratorParameterValues,
  getGPUGridVerticesPerCell,
  getGPUOutlineGeometryParameterValues,
  getGPUOutlineGeometryVerticesPerInput,
  getGPURectangleClipParameterValues,
  getGPUShapeGeneratorParameterValues,
  getGPUShapeMinimumSegments,
  getGPUShapeVertexCount,
  GPUGridGenerator,
  GPUHilbertKeys,
  GPUOutlineGeometry,
  GPURectangleClip,
  GPUShapeGenerator,
  GPU_GRID_GENERATOR_PARAMETER_LENGTH,
  GPU_OUTLINE_GEOMETRY_PARAMETER_LENGTH,
  GPU_RECTANGLE_CLIP_PARAMETER_LENGTH,
  GPU_SHAPE_GENERATOR_PARAMETER_LENGTH,
  type GPUGridType,
  type GPUShapeType
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {SpatialAnalysisPointLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneFrame, SceneInstance, ScenePointerEvent} from '../scene';
import {
  B3_PALETTE,
  FeatureTriangleLayer,
  PairSegmentLayer,
  PathOutputLayer,
  RingEdgeLayer,
  TriangleListLayer
} from './b3-layers';
import {
  copyCountToDrawRecord,
  createGraphImporter,
  createPathOutputBuffers,
  createStaticPaths,
  formatDistance,
  formatNumber,
  importPathOutput
} from './b3-common';
import {
  CHICAGO_LOOP,
  loadCityData,
  loadStreetPaths,
  type CityData,
  type PathSet
} from './b3-city-data';

/** Option state of the buffers-and-shapes scene. */
export type BuffersAndShapesOptions = {
  view: 'buffers' | 'shapes' | 'grid' | 'clip';
  bufferSource: 'stations' | 'lines' | 'rings';
  bufferDistance: number;
  bufferSystem: 'planar' | 'spherical';
  joinSegments: number;
  showSource: boolean;
  shapeKind: 'circle' | 'sector' | 'ellipse';
  shapeSystem: 'planar' | 'geodesic';
  maxSegments: string;
  segmentCount: number;
  shapeRadius: number;
  sectorSweep: number;
  ellipseRatio: number;
  ellipseSpacing: 'parameter' | 'arc-length';
  showShapeFill: boolean;
  gridType: 'square' | 'hex' | 'triangle' | 'point';
  cellSize: number;
  hilbertOrder: number;
  hilbertTarget: 'grid' | 'stops';
  showCurve: boolean;
  clipGeometry: 'roads' | 'areas';
  clipSize: number;
  showOriginal: boolean;
};

type Options = BuffersAndShapesOptions;

/** What the scene needs from one of its four views. */
type View = {
  getCompiledGraphs: () => CompiledGPUCommandGraph<never>[];
  encode: (commandEncoder: CommandEncoder, frame: SceneFrame) => void;
  getLayers: () => Layer[];
  setOption: (id: keyof Options, options: Options) => void;
  onClick?: (event: ScenePointerEvent) => boolean;
  onDragStart?: (event: ScenePointerEvent) => boolean;
  onDrag?: (event: ScenePointerEvent) => void;
  onDragEnd?: (event: ScenePointerEvent) => void;
  getTooltip?: (event: ScenePointerEvent) => string | null;
  destroy: () => void;
};

type ViewEnvironment = {
  ctx: SceneContext<Options>;
  city: CityData;
  coordinateOrigin: [number, number, number];
  loadStreets: () => PathSet;
};

const LOCAL = COORDINATE_SYSTEM.METER_OFFSETS;
const LNGLAT = COORDINATE_SYSTEM.LNGLAT;

const flat = (color: readonly number[], alpha: number): [number, number, number, number] => [
  color[0],
  color[1],
  color[2],
  alpha
];

/**
 * Buffers, generated shapes, grids, Hilbert keys and clipping on Chicago layers. Four views, each
 * with its own compiled graphs and resources, are created the first time they are shown; switching
 * to one is a compile-time change (the panel shows the rebuild badge), every slider inside a view
 * is a parameter write.
 */
export async function createBuffersAndShapes(
  ctx: SceneContext<Options>
): Promise<SceneInstance<Options>> {
  const city = loadCityData(
    ctx.datasets.get('cta-transit'),
    ctx.datasets.get('chicago-community-areas')
  );
  let streets: PathSet | null = null;
  const env: ViewEnvironment = {
    ctx,
    city,
    coordinateOrigin: [city.origin[0], city.origin[1], 0],
    loadStreets: () => {
      streets ??= loadStreetPaths(ctx.datasets.get('chicago-roads'), city.projection);
      return streets;
    }
  };
  const views = new Map<Options['view'], View>();
  const factories: Record<Options['view'], (env: ViewEnvironment) => View> = {
    buffers: createBuffersView,
    shapes: createShapesView,
    grid: createGridView,
    clip: createClipView
  };
  let destroyed = false;
  const getView = (): View => {
    const name = ctx.options.view;
    let view = views.get(name);
    if (!view) {
      view = factories[name](env);
      views.set(name, view);
    }
    return view;
  };
  getView();

  return {
    getCompiledGraphs: () => getView().getCompiledGraphs(),
    setOption(id, _value, options) {
      if (id === 'view') {
        getView();
        ctx.requestLayers();
        return;
      }
      getView().setOption(id, options);
      ctx.requestLayers();
    },
    onThemeChange: () => ctx.requestLayers(),
    getTooltip: event => getView().getTooltip?.(event) ?? null,
    onClick: event => getView().onClick?.(event) ?? false,
    onDragStart: event => getView().onDragStart?.(event) ?? false,
    onDrag: event => getView().onDrag?.(event),
    onDragEnd: event => getView().onDragEnd?.(event),
    encode(commandEncoder, frame) {
      if (!destroyed) getView().encode(commandEncoder, frame);
    },
    getLayers: () => getView().getLayers(),
    destroy() {
      destroyed = true;
      for (const view of views.values()) view.destroy();
      views.clear();
    }
  };
}

// ---------------------------------------------------------------------------------------------
// View: buffers (GPUOutlineGeometry)
// ---------------------------------------------------------------------------------------------

function createBuffersView(env: ViewEnvironment): View {
  const {ctx, city} = env;
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'buffers');
  const dark = () => ctx.theme() === 'dark';

  // Inputs per source, in local meters and in degrees.
  const sources = {
    stations: {
      type: 'points' as const,
      local: resources.createBuffer('stations-local', city.railStations.local),
      lngLat: resources.createBuffer('stations-lnglat', city.railStations.lngLat),
      count: city.railStations.count,
      offsets: null as Buffer | null,
      pathCount: 0
    },
    lines: {
      type: 'lines' as const,
      local: resources.createBuffer('lines-local', city.railLines.local),
      lngLat: resources.createBuffer('lines-lnglat', city.railLines.lngLat),
      count: city.railLines.vertexCount,
      offsets: resources.createBuffer('lines-offsets', city.railLines.offsets),
      pathCount: city.railLines.pathCount
    },
    rings: {
      type: 'rings' as const,
      local: resources.createBuffer('rings-local', city.areas.local),
      lngLat: resources.createBuffer('rings-lnglat', city.areas.lngLat),
      count: city.areas.vertexCount,
      offsets: resources.createBuffer('rings-offsets', city.areas.ringOffsets),
      pathCount: city.areas.ringCount
    }
  };
  const parameters = resources.createParameterBuffer(
    'outline-parameters',
    'float32',
    GPU_OUTLINE_GEOMETRY_PARAMETER_LENGTH
  );
  const staticLines = createStaticPaths(
    resources,
    'buffers-lines',
    city.railLines.local,
    city.railLines.offsets
  );
  const staticRings = createStaticPaths(
    resources,
    'buffers-rings',
    city.areas.local,
    city.areas.ringOffsets
  );

  let built: {
    key: string;
    compiled: CompiledGPUCommandGraph<void>;
    triangles: Buffer;
    triangleCount: number;
  } | null = null;
  let dirty = true;

  const build = (options: Options) => {
    const key = `${options.bufferSource}|${options.bufferSystem}|${options.joinSegments}`;
    if (built?.key === key) return;
    if (built) resources.release(built.compiled);
    const source = sources[options.bufferSource];
    const spherical = options.bufferSystem === 'spherical';
    const verticesPerInput = getGPUOutlineGeometryVerticesPerInput(options.joinSegments);
    const outputCount = source.count * verticesPerInput;
    const triangles = resources.createBuffer(`triangles-${key}`, outputCount * 8);
    const graph = new GPUCommandGraph<void>(device, {id: `outline-${key}`});
    const imp = createGraphImporter(graph);
    graph.add(
      new GPUOutlineGeometry({
        id: 'outline',
        positions: imp(
          'positions',
          spherical ? source.lngLat : source.local,
          'float32x2',
          source.count
        ),
        geometryType: source.type,
        pathOffsets: source.offsets
          ? imp('offsets', source.offsets, 'uint32', source.pathCount + 1)
          : undefined,
        coordinateSystem: spherical ? 'spherical' : 'planar',
        joinSegments: options.joinSegments,
        parameters: parameters.importToGraph(graph),
        output: {positions: imp('triangles', triangles, 'float32x2', outputCount)}
      })
    );
    built = {
      key,
      compiled: resources.track(graph.compile()),
      triangles,
      triangleCount: outputCount / 3
    };
    dirty = true;
    ctx.setReadout('bufferInputs', `${formatCount(source.count)} vertices`);
    ctx.setReadout(
      'bufferTriangles',
      `${formatCount(outputCount / 3)} (${verticesPerInput} vertices each input)`
    );
  };
  const writeDistance = (options: Options) => {
    parameters.write(getGPUOutlineGeometryParameterValues({distance: options.bufferDistance}));
    // Inscribed round joins are up to distance * (1 - cos(pi / n)) narrower than the true offset.
    const narrowing = options.bufferDistance * (1 - Math.cos(Math.PI / options.joinSegments));
    ctx.setReadout(
      'bufferNarrowing',
      `${formatDistance(narrowing)} (${(100 * (1 - Math.cos(Math.PI / options.joinSegments))).toFixed(1)}%)`
    );
    dirty = true;
  };
  build(ctx.options);
  writeDistance(ctx.options);

  return {
    getCompiledGraphs: () => (built ? [built.compiled as CompiledGPUCommandGraph<never>] : []),
    setOption(id, options) {
      if (id === 'bufferSource' || id === 'bufferSystem' || id === 'joinSegments') build(options);
      if (id === 'bufferDistance' || id === 'joinSegments') writeDistance(options);
    },
    encode(commandEncoder) {
      if (built && dirty) {
        built.compiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
      }
    },
    getLayers() {
      if (!built) return [];
      const options = ctx.options;
      const spherical = options.bufferSystem === 'spherical';
      const layers: Layer[] = [
        new TriangleListLayer({
          id: 'buffer-triangles',
          coordinateSystem: spherical ? LNGLAT : LOCAL,
          coordinateOrigin: env.coordinateOrigin,
          positions: built.triangles,
          triangleCount: built.triangleCount,
          color: flat(dark() ? B3_PALETTE[0] : B3_PALETTE[0], 235),
          opacity: 0.55
        })
      ];
      if (options.showSource) {
        const color: [number, number, number, number] = dark()
          ? [255, 255, 255, 235]
          : [20, 30, 50, 235];
        if (options.bufferSource === 'stations') {
          layers.push(
            new SpatialAnalysisPointLayer({
              id: 'buffer-source-points',
              coordinateSystem: LOCAL,
              coordinateOrigin: env.coordinateOrigin,
              positions: sources.stations.local,
              instanceCount: sources.stations.count,
              radiusPixels: 2.6,
              color
            })
          );
        } else {
          const paths = options.bufferSource === 'lines' ? staticLines : staticRings;
          layers.push(
            new PathOutputLayer({
              id: 'buffer-source-paths',
              coordinateSystem: LOCAL,
              coordinateOrigin: env.coordinateOrigin,
              positions: paths.positions,
              pathOffsets: paths.offsets,
              pathOffsetCount: paths.offsetCount,
              vertexCount: paths.vertexCount,
              drawCommands: paths.drawCommands,
              color,
              widthPixels: 1.4
            })
          );
        }
      }
      return layers;
    },
    destroy: () => resources.destroy()
  };
}

// ---------------------------------------------------------------------------------------------
// View: shapes (GPUShapeGenerator)
// ---------------------------------------------------------------------------------------------

function createShapesView(env: ViewEnvironment): View {
  const {ctx, city} = env;
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'shapes');
  const stations = city.railStations;
  const count = stations.count;
  const loopLocal = city.projection.project(CHICAGO_LOOP[0], CHICAGO_LOOP[1]);

  // Per-station data: bearing toward the Loop, distance to it, a size factor from route count.
  const bearings = new Float32Array(count);
  const distances = new Float32Array(count);
  const factors = new Float32Array(count);
  for (let row = 0; row < count; row++) {
    const dx = loopLocal[0] - stations.local[row * 2];
    const dy = loopLocal[1] - stations.local[row * 2 + 1];
    bearings[row] = ((Math.atan2(dx, dy) * 180) / Math.PI + 360) % 360;
    distances[row] = Math.hypot(dx, dy) / 1000;
    factors[row] = 0.65 + 0.2 * Math.min(stations.routeCount[row], 5);
  }
  let meanFactor = 0;
  for (const factor of factors) meanFactor += factor;
  meanFactor /= count;
  for (let row = 0; row < count; row++) factors[row] /= meanFactor;

  const centersLocal = resources.createBuffer('centers-local', stations.local);
  const centersLngLat = resources.createBuffer('centers-lnglat', stations.lngLat);
  const circleRadii = resources.createBuffer('circle-radii', factors);
  const ellipseRadii = resources.createBuffer('ellipse-radii', count * 8);
  const rotations = resources.createBuffer('rotations', bearings);
  const sectorBearings = resources.createBuffer('sector-bearings', count * 8);
  const distanceValues = resources.createBuffer('distance-values', distances);
  const parameters = resources.createParameterBuffer(
    'shape-parameters',
    'float32',
    GPU_SHAPE_GENERATOR_PARAMETER_LENGTH
  );

  type Built = {
    key: string;
    compiled: CompiledGPUCommandGraph<void>;
    positions: Buffer;
    corners: Buffer;
    featureRows: Buffer;
    slotCount: number;
    maximumSegments: number;
  };
  const cache = new Map<string, Built>();
  let active: Built | null = null;
  let dirty = true;

  const writeBearings = (options: Options) => {
    const data = new Float32Array(count * 2);
    for (let row = 0; row < count; row++) {
      data[row * 2] = bearings[row] - options.sectorSweep / 2;
      data[row * 2 + 1] = bearings[row] + options.sectorSweep / 2;
    }
    sectorBearings.write(data);
  };
  const writeEllipse = (options: Options) => {
    const data = new Float32Array(count * 2);
    for (let row = 0; row < count; row++) {
      data[row * 2] = factors[row];
      data[row * 2 + 1] = factors[row] * options.ellipseRatio;
    }
    ellipseRadii.write(data);
  };
  const clampSegments = (options: Options, kind: GPUShapeType, maximum: number) =>
    Math.min(Math.max(options.segmentCount, getGPUShapeMinimumSegments(kind)), maximum);
  const writeParameters = (options: Options) => {
    parameters.write(
      getGPUShapeGeneratorParameterValues({
        segmentCount: options.segmentCount,
        radiusScale: options.shapeRadius
      })
    );
    dirty = true;
  };

  const build = (options: Options) => {
    const maximumSegments = Number(options.maxSegments);
    const key = `${options.shapeKind}|${options.shapeSystem}|${options.ellipseSpacing}|${maximumSegments}`;
    let built = cache.get(key);
    if (!built) {
      const kind = options.shapeKind;
      const geodesic = options.shapeSystem === 'geodesic';
      const slotCount = count * getGPUShapeVertexCount(kind, maximumSegments);
      const positions = resources.createBuffer(`positions-${key}`, slotCount * 8);
      const offsets = resources.createBuffer(`offsets-${key}`, (count + 1) * 4);
      const total = resources.createBuffer(`total-${key}`, 4);
      const corners = resources.createBuffer(`corners-${key}`, slotCount * 3 * 8);
      const featureRows = resources.createBuffer(`rows-${key}`, slotCount * 4);
      const graph = new GPUCommandGraph<void>(device, {id: `shape-${key}`});
      const imp = createGraphImporter(graph);
      const positionsView = imp('positions', positions, 'float32x2', slotCount);
      const offsetsView = imp('offsets', offsets, 'uint32', count + 1);
      const totalView = imp('total', total, 'uint32', 1);
      graph.add(
        new GPUShapeGenerator({
          id: `shape-${kind}`,
          shape: kind,
          coordinateSystem: geodesic ? 'geodesic' : 'planar',
          ellipseSpacing: kind === 'ellipse' ? options.ellipseSpacing : undefined,
          maximumSegments,
          centers: imp('centers', geodesic ? centersLngLat : centersLocal, 'float32x2', count),
          radii:
            kind === 'ellipse'
              ? imp('radii', ellipseRadii, 'float32x2', count)
              : imp('radii', circleRadii, 'float32', count),
          bearings:
            kind === 'sector' ? imp('bearings', sectorBearings, 'float32x2', count) : undefined,
          rotations: kind === 'ellipse' ? imp('rotations', rotations, 'float32', count) : undefined,
          parameters: parameters.importToGraph(graph),
          output: {positions: positionsView, offsets: offsetsView, vertexCount: totalView}
        })
      );
      // Fan triangles (center, vertex i, vertex i + 1) fill each shape; slots past the written
      // vertices and the closing slot of each ring are NaN and hidden.
      addKernelPass(graph, {
        id: 'shape-fans',
        invocationCount: slotCount,
        bindings: [
          {name: 'positions', view: positionsView, type: 'f32', access: 'read'},
          {name: 'offsets', view: offsetsView, type: 'u32', access: 'read'},
          {name: 'total', view: totalView, type: 'u32', access: 'read'},
          {
            name: 'centers',
            view: imp('centers', geodesic ? centersLngLat : centersLocal, 'float32x2', count),
            type: 'f32',
            access: 'read'
          },
          {
            name: 'corners',
            view: imp('corners', corners, 'float32x2', slotCount * 3),
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'rows',
            view: imp('rows', featureRows, 'uint32', slotCount),
            type: 'u32',
            access: 'read_write'
          }
        ],
        declarations: `const FEATURES: u32 = ${count}u;`,
        body: /* wgsl */ `
  let stride = max(offsets[offsetsOffset + 1u], 1u);
  let feature = min(index / stride, FEATURES - 1u);
  let center = vec2<f32>(centers[centersOffset + feature * 2u], centers[centersOffset + feature * 2u + 1u]);
  let zero = f32(index) * 0.0;
  var a = vec2<f32>(zero / zero);
  var b = a;
  var c = a;
  if (index < total[totalOffset] && (index + 1u) % stride != 0u) {
    a = center;
    b = vec2<f32>(positions[positionsOffset + index * 2u], positions[positionsOffset + index * 2u + 1u]);
    c = vec2<f32>(positions[positionsOffset + index * 2u + 2u], positions[positionsOffset + index * 2u + 3u]);
  }
  corners[cornersOffset + index * 6u] = a.x;
  corners[cornersOffset + index * 6u + 1u] = a.y;
  corners[cornersOffset + index * 6u + 2u] = b.x;
  corners[cornersOffset + index * 6u + 3u] = b.y;
  corners[cornersOffset + index * 6u + 4u] = c.x;
  corners[cornersOffset + index * 6u + 5u] = c.y;
  rows[rowsOffset + index] = feature;`
      });
      built = {
        key,
        compiled: resources.track(graph.compile()),
        positions,
        corners,
        featureRows,
        slotCount,
        maximumSegments
      };
      cache.set(key, built);
    }
    active = built;
    dirty = true;
  };

  writeBearings(ctx.options);
  writeEllipse(ctx.options);
  writeParameters(ctx.options);
  build(ctx.options);
  ctx.setReadout('shapeCount', count);

  const updateReadouts = (options: Options) => {
    const kind = options.shapeKind;
    if (!active) return;
    const segments = clampSegments(options, kind, active.maximumSegments);
    ctx.setReadout(
      'shapeVertices',
      `${getGPUShapeVertexCount(kind, segments)} per shape (${segments} segments${segments !== options.segmentCount ? `, clamped to ${active.maximumSegments} by maximumSegments` : ''})`
    );
    const radiusKm = options.shapeRadius / 1000;
    const area =
      kind === 'circle'
        ? Math.PI * radiusKm ** 2
        : kind === 'sector'
          ? (Math.PI * radiusKm ** 2 * options.sectorSweep) / 360
          : Math.PI * radiusKm ** 2 * options.ellipseRatio;
    ctx.setReadout('shapeArea', `${formatNumber(area, 2)} km² per average station`);
  };
  updateReadouts(ctx.options);

  const dark = () => ctx.theme() === 'dark';
  return {
    getCompiledGraphs: () => (active ? [active.compiled as CompiledGPUCommandGraph<never>] : []),
    setOption(id, options) {
      if (id === 'sectorSweep') {
        writeBearings(options);
        dirty = true;
      }
      if (id === 'ellipseRatio') {
        writeEllipse(options);
        dirty = true;
      }
      if (id === 'segmentCount' || id === 'shapeRadius') writeParameters(options);
      if (
        id === 'shapeKind' ||
        id === 'shapeSystem' ||
        id === 'ellipseSpacing' ||
        id === 'maxSegments'
      ) {
        build(options);
      }
      updateReadouts(options);
    },
    encode(commandEncoder) {
      if (active && dirty) {
        active.compiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
      }
    },
    getLayers() {
      if (!active) return [];
      const options = ctx.options;
      const geodesic = options.shapeSystem === 'geodesic';
      const system = geodesic ? LNGLAT : LOCAL;
      const segments = clampSegments(options, options.shapeKind, active.maximumSegments);
      const stride = getGPUShapeVertexCount(options.shapeKind, segments);
      const layers: Layer[] = [];
      if (options.showShapeFill) {
        layers.push(
          new FeatureTriangleLayer({
            id: 'shape-fill',
            coordinateSystem: system,
            coordinateOrigin: env.coordinateOrigin,
            corners: active.corners,
            featureRows: active.featureRows,
            instanceCount: active.slotCount,
            values: distanceValues,
            colormap: 'ylgnbu',
            valueRange: [0, 20],
            color: [255, 255, 255, 255],
            opacity: 0.42
          })
        );
      }
      layers.push(
        new RingEdgeLayer({
          id: 'shape-rings',
          coordinateSystem: system,
          coordinateOrigin: env.coordinateOrigin,
          positions: active.positions,
          stride,
          ringCount: count,
          values: distanceValues,
          colorSource: 'slot-value',
          colormap: 'ylgnbu',
          valueRange: [0, 20],
          color: [255, 255, 255, 255],
          widthPixels: 1.4
        }),
        new SpatialAnalysisPointLayer({
          id: 'shape-centers',
          coordinateSystem: LOCAL,
          coordinateOrigin: env.coordinateOrigin,
          positions: centersLocal,
          instanceCount: count,
          radiusPixels: 2.6,
          color: dark() ? [255, 255, 255, 255] : [20, 30, 50, 255]
        })
      );
      return layers;
    },
    destroy: () => resources.destroy()
  };
}

// ---------------------------------------------------------------------------------------------
// View: grid + Hilbert (GPUGridGenerator, GPUHilbertKeys)
// ---------------------------------------------------------------------------------------------

const GRID_COLUMNS = 56;
const GRID_ROWS = 72;

function createGridView(env: ViewEnvironment): View {
  const {ctx, city} = env;
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'grid');
  const stopCount = city.allStops.count;

  const gridParameters = resources.createParameterBuffer(
    'grid-parameters',
    'float32',
    GPU_GRID_GENERATOR_PARAMETER_LENGTH
  );
  const gridHilbertBounds = resources.createParameterBuffer('grid-hilbert-bounds', 'float32', 4);
  const stopsHilbertBounds = resources.createParameterBuffer('stops-hilbert-bounds', 'float32', 4);
  const stopPositions = resources.createBuffer('stop-positions', city.allStops.local);

  type GridBuild = {
    key: string;
    compiled: CompiledGPUCommandGraph<void>;
    cellCount: number;
    verticesPerCell: number;
    positions: Buffer | null;
    centers: Buffer;
    values: Buffer;
    fanCorners: Buffer | null;
    fanRows: Buffer | null;
    fanCount: number;
    curveStarts: Buffer;
    curveEnds: Buffer;
    keys: Buffer;
    rows: Buffer;
  };
  type StopsBuild = {
    key: string;
    compiled: CompiledGPUCommandGraph<void>;
    values: Buffer;
    curveStarts: Buffer;
    curveEnds: Buffer;
    rows: Buffer;
  };
  let grid: GridBuild | null = null;
  let stops: StopsBuild | null = null;
  let dirty = true;
  let stopsDirty = true;
  let stopsReader: SummaryReader | null = null;

  // Bounds of the transit stops in local meters, padded a little.
  let stopMinX = Infinity;
  let stopMinY = Infinity;
  let stopMaxX = -Infinity;
  let stopMaxY = -Infinity;
  for (let row = 0; row < stopCount; row++) {
    stopMinX = Math.min(stopMinX, city.allStops.local[row * 2]);
    stopMaxX = Math.max(stopMaxX, city.allStops.local[row * 2]);
    stopMinY = Math.min(stopMinY, city.allStops.local[row * 2 + 1]);
    stopMaxY = Math.max(stopMaxY, city.allStops.local[row * 2 + 1]);
  }
  stopsHilbertBounds.write(
    Float32Array.of(stopMinX - 100, stopMinY - 100, stopMaxX + 100, stopMaxY + 100)
  );

  /** Curve segments through the sorted items, plus normalized key values, in one kernel. */
  const addCurvePass = (
    graph: GPUCommandGraph<void>,
    imp: ReturnType<typeof createGraphImporter>,
    count: number,
    centersView: ReturnType<ReturnType<typeof createGraphImporter>>,
    keysView: ReturnType<ReturnType<typeof createGraphImporter>>,
    rowsView: ReturnType<ReturnType<typeof createGraphImporter>>,
    buffers: {starts: Buffer; ends: Buffer; values: Buffer},
    order: number
  ) => {
    addKernelPass(graph, {
      id: 'hilbert-curve',
      invocationCount: count,
      bindings: [
        {name: 'rows', view: rowsView, type: 'u32', access: 'read'},
        {name: 'keys', view: keysView, type: 'u32', access: 'read'},
        {name: 'centers', view: centersView, type: 'f32', access: 'read'},
        {
          name: 'starts',
          view: imp('curve-starts', buffers.starts, 'float32', count * 2),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'ends',
          view: imp('curve-ends', buffers.ends, 'float32', count * 2),
          type: 'f32',
          access: 'read_write'
        },
        {
          name: 'values',
          view: imp('values', buffers.values, 'float32', count),
          type: 'f32',
          access: 'read_write'
        }
      ],
      declarations: `const COUNT: u32 = ${count}u;\nconst CELLS: f32 = ${(4 ** order).toFixed(1)};`,
      body: /* wgsl */ `
  let zero = f32(index) * 0.0;
  var start = vec2<f32>(zero / zero);
  var end = start;
  if (index + 1u < COUNT) {
    let first = rows[rowsOffset + index];
    let second = rows[rowsOffset + index + 1u];
    start = vec2<f32>(centers[centersOffset + first * 2u], centers[centersOffset + first * 2u + 1u]);
    end = vec2<f32>(centers[centersOffset + second * 2u], centers[centersOffset + second * 2u + 1u]);
  }
  starts[startsOffset + index * 2u] = start.x;
  starts[startsOffset + index * 2u + 1u] = start.y;
  ends[endsOffset + index * 2u] = end.x;
  ends[endsOffset + index * 2u + 1u] = end.y;
  values[valuesOffset + index] = f32(keys[keysOffset + rows[rowsOffset + index]]) / CELLS;`
    });
  };

  const buildGrid = (options: Options) => {
    const key = `${options.gridType}|${options.hilbertOrder}`;
    if (grid?.key === key) return;
    if (grid) resources.release(grid.compiled);
    const type = options.gridType as GPUGridType;
    const verticesPerCell = getGPUGridVerticesPerCell(type);
    const cellCount = GRID_COLUMNS * GRID_ROWS * (type === 'triangle' ? 2 : 1);
    const positions =
      verticesPerCell > 0
        ? resources.createBuffer(`grid-positions-${key}`, cellCount * verticesPerCell * 8)
        : null;
    const centers = resources.createBuffer(`grid-centers-${key}`, cellCount * 8);
    const keys = resources.createBuffer(`grid-keys-${key}`, cellCount * 4);
    const rows = resources.createBuffer(`grid-rows-${key}`, cellCount * 4);
    const values = resources.createBuffer(`grid-curve-values-${key}`, cellCount * 4);
    const curveStarts = resources.createBuffer(`grid-curve-starts-${key}`, cellCount * 8);
    const curveEnds = resources.createBuffer(`grid-curve-ends-${key}`, cellCount * 8);
    const cellValues = resources.createBuffer(`grid-cell-values-${key}`, cellCount * 4);
    const fanCount = verticesPerCell > 0 ? cellCount * verticesPerCell : 0;
    const fanCorners = fanCount
      ? resources.createBuffer(`grid-fan-${key}`, fanCount * 3 * 8)
      : null;
    const fanRowData = new Uint32Array(Math.max(fanCount, 1));
    for (let triangle = 0; triangle < fanCount; triangle++)
      fanRowData[triangle] = Math.floor(triangle / Math.max(verticesPerCell, 1));
    const fanRows = fanCount ? resources.createBuffer(`grid-fan-rows-${key}`, fanRowData) : null;

    const graph = new GPUCommandGraph<void>(device, {id: `grid-${key}`});
    const imp = createGraphImporter(graph);
    const centersView = imp('centers', centers, 'float32x2', cellCount);
    graph.add(
      new GPUGridGenerator({
        id: 'grid',
        gridType: type,
        columns: GRID_COLUMNS,
        rows: GRID_ROWS,
        parameters: gridParameters.importToGraph(graph),
        output: {
          ...(positions
            ? {positions: imp('positions', positions, 'float32x2', cellCount * verticesPerCell)}
            : {}),
          centers: centersView
        }
      })
    );
    const keysView = imp('keys', keys, 'uint32', cellCount);
    const rowsView = imp('rows', rows, 'uint32', cellCount);
    graph.add(
      new GPUHilbertKeys({
        id: 'hilbert',
        order: options.hilbertOrder,
        points: centersView,
        bounds: gridHilbertBounds.importToGraph(graph),
        output: {keys: keysView, sortedRows: rowsView}
      })
    );
    addCurvePass(
      graph,
      imp,
      cellCount,
      centersView,
      keysView,
      rowsView,
      {starts: curveStarts, ends: curveEnds, values},
      options.hilbertOrder
    );
    // Per-cell value: key as a fraction of the curve.
    addKernelPass(graph, {
      id: 'cell-values',
      invocationCount: cellCount,
      bindings: [
        {name: 'keys', view: keysView, type: 'u32', access: 'read'},
        {
          name: 'cellValues',
          view: imp('cell-values', cellValues, 'float32', cellCount),
          type: 'f32',
          access: 'read_write'
        }
      ],
      declarations: `const CELLS: f32 = ${(4 ** options.hilbertOrder).toFixed(1)};`,
      body: /* wgsl */ `cellValues[cellValuesOffset + index] = f32(keys[keysOffset + index]) / CELLS;`
    });
    if (fanCorners && positions) {
      addKernelPass(graph, {
        id: 'cell-fans',
        invocationCount: fanCount,
        bindings: [
          {
            name: 'positions',
            view: imp('positions', positions, 'float32x2', cellCount * verticesPerCell),
            type: 'f32',
            access: 'read'
          },
          {name: 'centers', view: centersView, type: 'f32', access: 'read'},
          {
            name: 'corners',
            view: imp('fan', fanCorners, 'float32x2', fanCount * 3),
            type: 'f32',
            access: 'read_write'
          }
        ],
        declarations: `const K: u32 = ${verticesPerCell}u;`,
        body: /* wgsl */ `
  let cell = index / K;
  let step = index % K;
  let next = (step + 1u) % K;
  let center = vec2<f32>(centers[centersOffset + cell * 2u], centers[centersOffset + cell * 2u + 1u]);
  let a = vec2<f32>(positions[positionsOffset + (cell * K + step) * 2u], positions[positionsOffset + (cell * K + step) * 2u + 1u]);
  let b = vec2<f32>(positions[positionsOffset + (cell * K + next) * 2u], positions[positionsOffset + (cell * K + next) * 2u + 1u]);
  corners[cornersOffset + index * 6u] = center.x;
  corners[cornersOffset + index * 6u + 1u] = center.y;
  corners[cornersOffset + index * 6u + 2u] = a.x;
  corners[cornersOffset + index * 6u + 3u] = a.y;
  corners[cornersOffset + index * 6u + 4u] = b.x;
  corners[cornersOffset + index * 6u + 5u] = b.y;`
      });
    }
    grid = {
      key,
      compiled: resources.track(graph.compile()),
      cellCount,
      verticesPerCell,
      positions,
      centers,
      values: cellValues,
      fanCorners,
      fanRows,
      fanCount,
      curveStarts,
      curveEnds,
      keys,
      rows
    };
    // Curve color uses the sorted-order values (key fraction by curve position).
    (grid as GridBuild & {curveValues: Buffer}).curveValues = values;
    dirty = true;
    ctx.setReadout(
      'gridCells',
      `${formatCount(cellCount)} ${type === 'point' ? 'points' : type + ' cells'}`
    );
    ctx.setReadout(
      'hilbertCells',
      `${formatCount(4 ** options.hilbertOrder)} curve cells (order ${options.hilbertOrder})`
    );
  };

  const buildStops = (options: Options) => {
    const key = `${options.hilbertOrder}`;
    if (stops?.key === key) return;
    if (stops) resources.release(stops.compiled);
    stopsReader?.stop();
    const keys = resources.createBuffer(`stops-keys-${key}`, stopCount * 4);
    const rows = resources.createBuffer(`stops-rows-${key}`, stopCount * 4);
    const values = resources.createBuffer(`stops-values-${key}`, stopCount * 4);
    const pointValues = resources.createBuffer(`stops-point-values-${key}`, stopCount * 4);
    const curveStarts = resources.createBuffer(`stops-curve-starts-${key}`, stopCount * 8);
    const curveEnds = resources.createBuffer(`stops-curve-ends-${key}`, stopCount * 8);
    const graph = new GPUCommandGraph<void>(device, {id: `stops-${key}`});
    const imp = createGraphImporter(graph);
    const points = imp('points', stopPositions, 'float32x2', stopCount);
    const keysView = imp('keys', keys, 'uint32', stopCount);
    const rowsView = imp('rows', rows, 'uint32', stopCount);
    graph.add(
      new GPUHilbertKeys({
        id: 'stops-hilbert',
        order: options.hilbertOrder,
        points,
        bounds: stopsHilbertBounds.importToGraph(graph),
        output: {keys: keysView, sortedRows: rowsView}
      })
    );
    addCurvePass(
      graph,
      imp,
      stopCount,
      points,
      keysView,
      rowsView,
      {starts: curveStarts, ends: curveEnds, values},
      options.hilbertOrder
    );
    addKernelPass(graph, {
      id: 'stop-values',
      invocationCount: stopCount,
      bindings: [
        {name: 'keys', view: keysView, type: 'u32', access: 'read'},
        {
          name: 'out',
          view: imp('point-values', pointValues, 'float32', stopCount),
          type: 'f32',
          access: 'read_write'
        }
      ],
      declarations: `const CELLS: f32 = ${(4 ** options.hilbertOrder).toFixed(1)};`,
      body: /* wgsl */ `out[outOffset + index] = f32(keys[keysOffset + index]) / CELLS;`
    });
    stops = {
      key,
      compiled: resources.track(graph.compile()),
      values: pointValues,
      curveStarts,
      curveEnds,
      rows
    };
    stopsReader = new SummaryReader(
      resources,
      `stops-rows-${key}`,
      [{buffer: rows, size: stopCount * 4}],
      bytes => {
        const order = new Uint32Array(bytes);
        let hop = 0;
        let inputHop = 0;
        for (let index = 0; index + 1 < stopCount; index++) {
          const a = order[index];
          const b = order[index + 1];
          hop += Math.hypot(
            city.allStops.local[a * 2] - city.allStops.local[b * 2],
            city.allStops.local[a * 2 + 1] - city.allStops.local[b * 2 + 1]
          );
          inputHop += Math.hypot(
            city.allStops.local[index * 2] - city.allStops.local[index * 2 + 2],
            city.allStops.local[index * 2 + 1] - city.allStops.local[index * 2 + 3]
          );
        }
        ctx.setReadout(
          'hop',
          `${formatDistance(hop / (stopCount - 1))} along the curve vs ${formatDistance(inputHop / (stopCount - 1))} in file order`
        );
      }
    );
    stopsDirty = true;
  };

  const writeGrid = (options: Options) => {
    const type = options.gridType as GPUGridType;
    const width = options.cellSize;
    const height = type === 'triangle' ? (width * Math.sqrt(3)) / 2 : width;
    // Hexagon rows are 1.5 circumradii apart (circumradius = width / sqrt(3)).
    const rowPitch = type === 'hex' ? 1.5 * (width / Math.sqrt(3)) : height;
    const extentX = GRID_COLUMNS * width + (type === 'hex' || type === 'triangle' ? width / 2 : 0);
    const extentY = GRID_ROWS * rowPitch;
    const minX = -extentX / 2;
    const minY = -extentY / 2;
    gridParameters.write(
      getGPUGridGeneratorParameterValues({minX, minY, cellWidth: width, cellHeight: height})
    );
    gridHilbertBounds.write(Float32Array.of(minX, minY, minX + extentX, minY + extentY));
    ctx.setReadout(
      'gridExtent',
      `${(extentX / 1000).toFixed(1)} × ${(extentY / 1000).toFixed(1)} km`
    );
    dirty = true;
  };

  buildGrid(ctx.options);
  buildStops(ctx.options);
  writeGrid(ctx.options);

  return {
    getCompiledGraphs() {
      const graphs: CompiledGPUCommandGraph<never>[] = [];
      if (grid) graphs.push(grid.compiled as CompiledGPUCommandGraph<never>);
      if (stops) graphs.push(stops.compiled as CompiledGPUCommandGraph<never>);
      return graphs;
    },
    setOption(id, options) {
      if (id === 'gridType' || id === 'hilbertOrder') {
        buildGrid(options);
        writeGrid(options);
      }
      if (id === 'hilbertOrder') buildStops(options);
      if (id === 'cellSize') writeGrid(options);
    },
    encode(commandEncoder) {
      if (grid && dirty) {
        grid.compiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
      }
      if (stops && stopsDirty) {
        stops.compiled.encode(commandEncoder, {parameters: undefined});
        stopsReader?.request(commandEncoder);
        stopsDirty = false;
      } else {
        stopsReader?.flush(commandEncoder);
      }
    },
    getLayers() {
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      const edge: [number, number, number, number] = dark
        ? [255, 255, 255, 120]
        : [20, 30, 50, 120];
      if (options.hilbertTarget === 'grid' && grid) {
        if (grid.fanCorners && grid.fanRows) {
          layers.push(
            new FeatureTriangleLayer({
              id: 'grid-fill',
              coordinateSystem: LOCAL,
              coordinateOrigin: env.coordinateOrigin,
              corners: grid.fanCorners,
              featureRows: grid.fanRows,
              instanceCount: grid.fanCount,
              values: grid.values,
              colormap: 'ylgnbu',
              valueRange: [0, 1],
              color: [255, 255, 255, 255],
              opacity: 0.7
            })
          );
          layers.push(
            new RingEdgeLayer({
              id: 'grid-edges',
              coordinateSystem: LOCAL,
              coordinateOrigin: env.coordinateOrigin,
              positions: grid.positions!,
              stride: grid.verticesPerCell,
              ringCount: grid.cellCount,
              color: edge,
              widthPixels: 0.8
            })
          );
        } else {
          layers.push(
            new SpatialAnalysisPointLayer({
              id: 'grid-points',
              coordinateSystem: LOCAL,
              coordinateOrigin: env.coordinateOrigin,
              positions: grid.centers,
              instanceCount: grid.cellCount,
              radiusPixels: 4,
              values: grid.values,
              colormap: 'ylgnbu',
              valueRange: [0, 1]
            })
          );
        }
        if (options.showCurve) {
          layers.push(
            new PairSegmentLayer({
              id: 'grid-curve',
              coordinateSystem: LOCAL,
              coordinateOrigin: env.coordinateOrigin,
              starts: grid.curveStarts,
              ends: grid.curveEnds,
              instanceCount: grid.cellCount - 1,
              values: (grid as GridBuild & {curveValues: Buffer}).curveValues,
              colormap: 'ylorbr',
              valueRange: [0, 1],
              color: [255, 255, 255, 255],
              widthPixels: 1.6
            })
          );
        }
      }
      if (options.hilbertTarget === 'stops' && stops) {
        if (options.showCurve) {
          layers.push(
            new PairSegmentLayer({
              id: 'stops-curve',
              coordinateSystem: LOCAL,
              coordinateOrigin: env.coordinateOrigin,
              starts: stops.curveStarts,
              ends: stops.curveEnds,
              instanceCount: stopCount - 1,
              values: stops.values,
              colormap: 'ylgnbu',
              valueRange: [0, 1],
              color: [255, 255, 255, 255],
              widthPixels: 0.8,
              opacity: 0.55
            })
          );
        }
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'stops-points',
            coordinateSystem: LOCAL,
            coordinateOrigin: env.coordinateOrigin,
            positions: stopPositions,
            instanceCount: stopCount,
            radiusPixels: 2,
            values: stops.values,
            colormap: 'ylgnbu',
            valueRange: [0, 1]
          })
        );
      }
      return layers;
    },
    destroy() {
      stopsReader?.stop();
      resources.destroy();
    }
  };
}

// ---------------------------------------------------------------------------------------------
// View: clip (GPURectangleClip)
// ---------------------------------------------------------------------------------------------

function createClipView(env: ViewEnvironment): View {
  const {ctx, city} = env;
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'clip');
  const streets = env.loadStreets();
  const parameters = resources.createParameterBuffer(
    'clip-parameters',
    'float32',
    GPU_RECTANGLE_CLIP_PARAMETER_LENGTH
  );
  const rectangleStarts = resources.createBuffer('rectangle-starts', 4 * 8);
  const rectangleEnds = resources.createBuffer('rectangle-ends', 4 * 8);
  const loop = city.projection.project(CHICAGO_LOOP[0], CHICAGO_LOOP[1]);
  let center: [number, number] = [loop[0] - 2500, loop[1] + 1500];
  let dragging = false;
  let dragOffset: [number, number] = [0, 0];
  let dirty = true;

  const staticStreets = createStaticPaths(
    resources,
    'clip-streets',
    streets.local,
    streets.offsets
  );
  const staticAreas = createStaticPaths(
    resources,
    'clip-areas',
    city.areas.local,
    city.areas.ringOffsets
  );
  const streetPositions = resources.createBuffer('street-positions', streets.local);
  const streetOffsets = resources.createBuffer('street-offsets', streets.offsets);
  const areaPositions = resources.createBuffer('area-positions', city.areas.local);
  const areaOffsets = resources.createBuffer('area-offsets', city.areas.ringOffsets);
  const streetClasses = resources.createBuffer(
    'street-classes',
    Float32Array.from(streets.classes ?? [])
  );

  type ClipBuild = {
    compiled: CompiledGPUCommandGraph<void>;
    output: ReturnType<typeof createPathOutputBuffers>;
    pathValues: Buffer | null;
    reader: SummaryReader;
  };
  const builds = new Map<Options['clipGeometry'], ClipBuild>();

  const build = (geometry: Options['clipGeometry']): ClipBuild => {
    let existing = builds.get(geometry);
    if (existing) return existing;
    const lines = geometry === 'roads';
    const vertexCount = lines ? streets.vertexCount : city.areas.vertexCount;
    const pathCount = lines ? streets.pathCount : city.areas.ringCount;
    const vertexCapacity = vertexCount * 2 + 64;
    const pathCapacity = lines ? pathCount * 2 : pathCount;
    const output = createPathOutputBuffers(
      resources,
      `clip-${geometry}`,
      vertexCapacity,
      pathCapacity
    );
    const pathValues = lines ? resources.createBuffer('clip-path-values', pathCapacity * 4) : null;
    const graph = new GPUCommandGraph<void>(device, {id: `clip-${geometry}`});
    const imp = createGraphImporter(graph);
    const outputViews = importPathOutput(graph, output, {sourcePaths: lines, pathCount: lines});
    graph.add(
      new GPURectangleClip({
        id: 'clip',
        positions: imp(
          'positions',
          lines ? streetPositions : areaPositions,
          'float32x2',
          vertexCount
        ),
        geometryType: lines ? 'lines' : 'polygons',
        pathOffsets: imp('offsets', lines ? streetOffsets : areaOffsets, 'uint32', pathCount + 1),
        parameters: parameters.importToGraph(graph),
        output: outputViews
      })
    );
    if (lines && pathValues) {
      // Color each output piece by the class of the street it was cut from.
      addKernelPass(graph, {
        id: 'source-classes',
        invocationCount: pathCapacity,
        bindings: [
          {name: 'sources', view: outputViews.sourcePaths!, type: 'u32', access: 'read'},
          {
            name: 'classes',
            view: imp('classes', streetClasses, 'float32', pathCount),
            type: 'f32',
            access: 'read'
          },
          {
            name: 'values',
            view: imp('path-values', pathValues, 'float32', pathCapacity),
            type: 'f32',
            access: 'read_write'
          }
        ],
        declarations: `const PATHS: u32 = ${pathCount}u;`,
        body: /* wgsl */ `
  let source = sources[sourcesOffset + index];
  let zero = f32(index) * 0.0;
  var value = zero / zero;
  if (source < PATHS) {
    value = classes[classesOffset + source];
  }
  values[valuesOffset + index] = value;`
      });
    }
    const reader = new SummaryReader(
      resources,
      `clip-${geometry}`,
      [
        {buffer: output.count, size: 4},
        {buffer: output.totalCount, size: 4},
        {buffer: output.overflow, size: 4},
        {buffer: output.pathCount, size: 4}
      ],
      bytes => {
        const words = new Uint32Array(bytes);
        ctx.setReadout(
          'clipVertices',
          `${formatCount(words[0])} of ${formatCount(vertexCount)} input vertices`
        );
        ctx.setReadout('clipTotal', formatCount(words[1]));
        ctx.setReadout('clipOverflow', words[2] ? 'yes: output truncated' : 'no');
        ctx.setReadout(
          'clipPaths',
          lines
            ? `${formatCount(words[3])} pieces from ${formatCount(pathCount)} streets`
            : `${formatCount(pathCount)} rings (one output ring per input ring)`
        );
      }
    );
    existing = {compiled: resources.track(graph.compile()), output, pathValues, reader};
    builds.set(geometry, existing);
    return existing;
  };

  const writeRectangle = (options: Options) => {
    const half = options.clipSize * 1000;
    const [x, y] = center;
    parameters.write(
      getGPURectangleClipParameterValues({
        minX: x - half,
        minY: y - half,
        maxX: x + half,
        maxY: y + half
      })
    );
    rectangleStarts.write(
      Float32Array.of(
        x - half,
        y - half,
        x + half,
        y - half,
        x + half,
        y + half,
        x - half,
        y + half
      )
    );
    rectangleEnds.write(
      Float32Array.of(
        x + half,
        y - half,
        x + half,
        y + half,
        x - half,
        y + half,
        x - half,
        y - half
      )
    );
    dirty = true;
  };
  build(ctx.options.clipGeometry);
  writeRectangle(ctx.options);

  const toLocal = (event: ScenePointerEvent): [number, number] | null =>
    event.coordinate ? city.projection.project(event.coordinate[0], event.coordinate[1]) : null;
  const inside = (point: [number, number]) => {
    const half = ctx.options.clipSize * 1000;
    return Math.abs(point[0] - center[0]) <= half && Math.abs(point[1] - center[1]) <= half;
  };

  return {
    getCompiledGraphs: () =>
      [...builds.values()].map(entry => entry.compiled as CompiledGPUCommandGraph<never>),
    setOption(id, options) {
      if (id === 'clipGeometry') {
        build(options.clipGeometry);
        dirty = true;
      }
      if (id === 'clipSize') writeRectangle(options);
    },
    onClick(event) {
      const point = toLocal(event);
      if (!point) return false;
      center = point;
      writeRectangle(ctx.options);
      return true;
    },
    onDragStart(event) {
      const point = toLocal(event);
      if (!point || !inside(point)) return false;
      dragging = true;
      dragOffset = [center[0] - point[0], center[1] - point[1]];
      ctx.setMapDragEnabled(false);
      return true;
    },
    onDrag(event) {
      const point = toLocal(event);
      if (!dragging || !point) return;
      center = [point[0] + dragOffset[0], point[1] + dragOffset[1]];
      writeRectangle(ctx.options);
    },
    onDragEnd() {
      dragging = false;
      ctx.setMapDragEnabled(true);
    },
    encode(commandEncoder) {
      const entry = builds.get(ctx.options.clipGeometry);
      if (!entry) return;
      if (dirty) {
        entry.compiled.encode(commandEncoder, {parameters: undefined});
        copyCountToDrawRecord(commandEncoder, entry.output.count, entry.output.drawCommands);
        entry.reader.request(commandEncoder);
        dirty = false;
      } else {
        entry.reader.flush(commandEncoder);
      }
    },
    getLayers() {
      const options = ctx.options;
      const entry = builds.get(options.clipGeometry);
      if (!entry) return [];
      const dark = ctx.theme() === 'dark';
      const lines = options.clipGeometry === 'roads';
      const layers: Layer[] = [];
      if (options.showOriginal) {
        const paths = lines ? staticStreets : staticAreas;
        layers.push(
          new PathOutputLayer({
            id: 'clip-original',
            coordinateSystem: LOCAL,
            coordinateOrigin: env.coordinateOrigin,
            positions: paths.positions,
            pathOffsets: paths.offsets,
            pathOffsetCount: paths.offsetCount,
            vertexCount: paths.vertexCount,
            drawCommands: paths.drawCommands,
            color: dark ? [150, 160, 180, 70] : [90, 100, 120, 70],
            widthPixels: 0.8
          })
        );
      }
      layers.push(
        new PathOutputLayer({
          id: 'clip-output',
          coordinateSystem: LOCAL,
          coordinateOrigin: env.coordinateOrigin,
          positions: entry.output.positions,
          pathOffsets: entry.output.offsets,
          pathOffsetCount: entry.output.offsetCount,
          vertexCount: entry.output.count,
          drawCommands: entry.output.drawCommands,
          closed: !lines,
          values: lines ? entry.pathValues : null,
          colorSource: lines ? 'path-value' : 'uniform',
          valueMapping: 'category',
          color: lines ? [255, 255, 255, 255] : [78, 168, 222, 255],
          widthPixels: lines ? 1.8 : 2.4
        }),
        new PairSegmentLayer({
          id: 'clip-rectangle',
          coordinateSystem: LOCAL,
          coordinateOrigin: env.coordinateOrigin,
          starts: rectangleStarts,
          ends: rectangleEnds,
          instanceCount: 4,
          color: dark ? [255, 214, 120, 255] : [200, 90, 10, 255],
          widthPixels: 2.5
        })
      );
      return layers;
    },
    destroy() {
      for (const entry of builds.values()) entry.reader.stop();
      resources.destroy();
    }
  };
}
