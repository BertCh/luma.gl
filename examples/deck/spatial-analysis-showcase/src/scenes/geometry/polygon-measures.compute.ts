// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUShapeDescriptorsParameterValues,
  GPU_GEOMETRY_VALIDITY_BIT,
  GPU_GEOMETRY_VALIDITY_STRUCTURAL_MASK,
  GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH,
  GPUContiguityWeights,
  GPUGeometryMeasures,
  GPUGeometryValidity,
  GPULabelPoint,
  GPUMapColoring,
  GPUShapeDescriptors
} from '@luma.gl/experimental/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {SpatialAnalysisPointLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import type {RampName} from '../../engine/ramps';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {B3_PALETTE, FeatureTriangleLayer, PairSegmentLayer, type B3ValueMapping} from './b3-layers';
import {
  buildRingEdges,
  createGraphImporter,
  findFeatureAt,
  isPointInFeature,
  formatNumber,
  getQuantile,
  loadPolygonLayout,
  projectWebMercator,
  triangulatePolygons,
  type PolygonLayout
} from './b3-common';

/** Datasets of the scene. */
export type PolygonDatasetId = 'us-counties' | 'chicago-community-areas' | 'chicago-tracts';

/** What colors the polygons. */
export type PolygonMetric =
  | 'area'
  | 'areaDistortion'
  | 'perimeter'
  | 'vertices'
  | 'groupArea'
  | 'compactness'
  | 'schwartzberg'
  | 'elongation'
  | 'convexity'
  | 'sliver'
  | 'validity'
  | 'mapColor';

/** Option state of the polygon-measures scene. */
export type PolygonMeasuresOptions = {
  dataset: PolygonDatasetId;
  metric: PolygonMetric;
  areaSystem: 'planar' | 'spherical' | 'wgs84';
  holeRule: 'winding' | 'first-ring-exterior';
  ramp: Extract<RampName, 'viridis' | 'magma' | 'inferno' | 'cividis'>;
  opacity: number;
  showOutlines: boolean;
  showLabels: boolean;
  showInscribed: boolean;
  showCentroids: boolean;
  showAxes: boolean;
  showBounds: boolean;
  sliverThreshold: number;
  initialGridSize: number;
  refinementCandidates: number;
  orientation: 'counter-clockwise-shell' | 'clockwise-shell' | 'ignore';
  ringClosure: 'implicit' | 'explicit';
  injectDefects: boolean;
  contiguity: 'rook' | 'queen';
  colorSeed: number;
  maxRounds: number;
};

/** Number of segments of an inscribed circle. */
const CIRCLE_SEGMENTS = 32;
/** Mercator sphere radius. */
const MERCATOR_RADIUS = 6378137;
/** Validity category colors: valid, orientation only, structural defect. */
export const VALIDITY_PALETTE_INDEXES = [3, 5, 7] as const;
export const VALIDITY_COLORS = VALIDITY_PALETTE_INDEXES.map(index => B3_PALETTE[index]);

type DatasetSpec = {
  space: 'local' | 'mercator';
  groupColumn: string | null;
  describe: (properties: Record<string, unknown>, row: number) => string;
  groupName: (row: number) => string;
};

const DATASET_SPECS: Record<PolygonDatasetId, DatasetSpec> = {
  'us-counties': {
    space: 'mercator',
    groupColumn: 'stateFips',
    describe: properties => `${properties.name} County, ${properties.state}`,
    groupName: row => `state ${row}`
  },
  'chicago-community-areas': {
    space: 'local',
    groupColumn: null,
    describe: properties => String(properties.name),
    groupName: row => `area ${row}`
  },
  'chicago-tracts': {
    space: 'local',
    groupColumn: 'communityArea',
    describe: properties =>
      `Tract ${properties.GEOID} (community area ${properties.communityArea})`,
    groupName: row => `community area ${row}`
  }
};

/** Per-feature columns read back for tooltips, legends and readouts. */
type FeatureTable = {
  areaPlanar: Float32Array;
  areaSphere: Float32Array;
  areaWgs84: Float32Array;
  lengthPlanar: Float32Array;
  lengthSphere: Float32Array;
  lengthWgs84: Float32Array;
  vertexCount: Uint32Array;
  polsbyPopper: Float32Array;
  schwartzberg: Float32Array;
  elongation: Float32Array;
  convexity: Float32Array;
  sliver: Uint32Array;
  clockwise: Uint32Array;
  labelPoint: Float32Array;
  radius: Float32Array;
  centroid: Float32Array;
  mask: Uint32Array;
  groupAreas: Float32Array;
};

type ColoringTable = {
  colorCount: number;
  conflictCount: number;
  converged: number;
  roundCount: number;
  overflow: number;
  adjacencies: number;
};

/** Everything built for one dataset; destroyed when the dataset option changes. */
type DatasetState = {
  id: PolygonDatasetId;
  layout: PolygonLayout;
  names: string[];
  groupIds: Uint32Array;
  groupCount: number;
  resources: SpatialAnalysisResources;
  fill: {corners: Buffer; featureRows: Buffer; partRows: Buffer; triangleCount: number};
  edges: {starts: Buffer; ends: Buffer; edgeCount: number};
  /** Float32 per-feature display columns. */
  columns: Record<
    | 'areaPlanar'
    | 'areaSphere'
    | 'areaWgs84'
    | 'lengthPlanar'
    | 'lengthSphere'
    | 'lengthWgs84'
    | 'ratio'
    | 'vertexF'
    | 'groupAreaF'
    | 'polsbyPopper'
    | 'schwartzberg'
    | 'elongation'
    | 'convexity'
    | 'sliverF'
    | 'validityF'
    | 'colorF',
    Buffer
  >;
  overlays: {
    labelPoints: Buffer;
    circleStarts: Buffer;
    circleEnds: Buffer;
    axisStarts: Buffer;
    axisEnds: Buffer;
    centroids: Buffer;
    boundsStarts: Buffer;
    boundsEnds: Buffer;
    selectedStarts: Buffer;
    selectedEnds: Buffer;
  };
  selectedEdgeCapacity: number;
  builders: {
    measures: (options: PolygonMeasuresOptions) => CompiledGPUCommandGraph<void>;
    shape: (options: PolygonMeasuresOptions) => CompiledGPUCommandGraph<void>;
    validity: (options: PolygonMeasuresOptions) => CompiledGPUCommandGraph<void>;
    coloring: (options: PolygonMeasuresOptions) => CompiledGPUCommandGraph<void>;
  };
  measures: CompiledGPUCommandGraph<void>;
  shape: CompiledGPUCommandGraph<void> | null;
  validity: CompiledGPUCommandGraph<void> | null;
  coloring: CompiledGPUCommandGraph<void> | null;
  shapeParameters: ReturnType<SpatialAnalysisResources['createParameterBuffer']>;
  validityPositions: Buffer;
  tableReader: SummaryReader | null;
  coloringReader: SummaryReader | null;
  table: FeatureTable | null;
  coloringTable: ColoringTable | null;
  dirty: {measures: boolean; shape: boolean; validity: boolean; coloring: boolean; table: boolean};
  selected: number;
  selectedCount: number;
};

/**
 * Polygon measures, shape descriptors, label points, validity and map coloring on real polygon
 * layers (US counties, Chicago community areas and tracts). Every contributor runs on the GPU
 * over the whole layer; the per-feature outputs are bound straight into the fill layer, and a
 * small summary table is read back once the controls settle for tooltips and legends.
 */
export async function createPolygonMeasures(
  ctx: SceneContext<PolygonMeasuresOptions>
): Promise<SceneInstance<PolygonMeasuresOptions>> {
  const {device} = ctx;
  let destroyed = false;
  let state: DatasetState | null = null;
  const layouts = new Map<PolygonDatasetId, PolygonLayout>();

  const getLayout = (id: PolygonDatasetId): PolygonLayout => {
    let layout = layouts.get(id);
    if (!layout) {
      layout = loadPolygonLayout(ctx.datasets.get(id));
      layouts.set(id, layout);
    }
    return layout;
  };

  // -------------------------------------------------------------------------------------------
  // Building
  // -------------------------------------------------------------------------------------------

  function buildDataset(id: PolygonDatasetId): DatasetState {
    const dataset = ctx.datasets.get(id);
    const spec = DATASET_SPECS[id];
    const layout = getLayout(id);
    const {featureCount, ringCount, vertexCount, partCount} = layout;
    const resources = new SpatialAnalysisResources(device, `polygon-measures-${id}`);
    const origin = dataset.defaultOrigin;
    const projection = dataset.getProjection(origin);

    // Planar space: Web Mercator for the continental layer, local meters for city layers.
    const planar =
      spec.space === 'mercator'
        ? projectWebMercator(layout.lngLat)
        : dataset.projectColumn('vertices', origin);

    // Group ids: dense indexes of the group column (state or community area).
    let groupIds = new Uint32Array(featureCount);
    let groupCount = 1;
    if (spec.groupColumn) {
      const column = dataset.column<Uint8Array>(spec.groupColumn);
      const dense = new Map<number, number>();
      for (let feature = 0; feature < featureCount; feature++) {
        const key = column[feature];
        if (!dense.has(key)) dense.set(key, dense.size);
        groupIds[feature] = dense.get(key)!;
      }
      groupCount = dense.size;
    }

    const names: string[] = [];
    const features = dataset.geojson?.features ?? [];
    for (let feature = 0; feature < featureCount; feature++) {
      names.push(
        spec.describe((features[feature]?.properties ?? {}) as Record<string, unknown>, feature)
      );
    }

    // Static inputs.
    const lngLatBuffer = resources.createBuffer('lnglat', layout.lngLat);
    const planarBuffer = resources.createBuffer('planar', planar);
    const ringOffsetsBuffer = resources.createBuffer('ring-offsets', layout.ringOffsets);
    const featureRingsBuffer = resources.createBuffer('feature-rings', layout.featureRingOffsets);
    const polygonOffsetsBuffer = resources.createBuffer('polygon-offsets', layout.polygonOffsets);
    const featureOffsetsBuffer = resources.createBuffer('feature-offsets', layout.featureOffsets);
    const groupIdsBuffer = resources.createBuffer('group-ids', groupIds);
    const validityPositions = resources.createBuffer('validity-positions', layout.lngLat);

    // Drawing data.
    const triangulated = triangulatePolygons(layout);
    const edgeColumns = buildRingEdges(layout);
    const fill = {
      corners: resources.createBuffer('fill-corners', triangulated.corners),
      featureRows: resources.createBuffer('fill-feature-rows', triangulated.featureRows),
      partRows: resources.createBuffer('fill-part-rows', triangulated.partRows),
      triangleCount: triangulated.triangleCount
    };
    const edges = {
      starts: resources.createBuffer('edge-starts', edgeColumns.starts),
      ends: resources.createBuffer('edge-ends', edgeColumns.ends),
      edgeCount: edgeColumns.edgeCount
    };
    let selectedEdgeCapacity = 1;
    for (let feature = 0; feature < featureCount; feature++) {
      const first = layout.ringOffsets[layout.featureRingOffsets[feature]];
      const last = layout.ringOffsets[layout.featureRingOffsets[feature + 1]];
      selectedEdgeCapacity = Math.max(selectedEdgeCapacity, last - first);
    }

    // Per-feature columns.
    const makeColumn = (name: string, length = featureCount) =>
      resources.createBuffer(name, Math.max(length, 1) * 4);
    const columns = {
      areaPlanar: makeColumn('area-planar'),
      areaSphere: makeColumn('area-sphere'),
      areaWgs84: makeColumn('area-wgs84'),
      lengthPlanar: makeColumn('length-planar'),
      lengthSphere: makeColumn('length-sphere'),
      lengthWgs84: makeColumn('length-wgs84'),
      ratio: makeColumn('ratio'),
      vertexF: makeColumn('vertex-f'),
      groupAreaF: makeColumn('group-area-f'),
      polsbyPopper: makeColumn('polsby-popper'),
      schwartzberg: makeColumn('schwartzberg'),
      elongation: makeColumn('elongation'),
      convexity: makeColumn('convexity'),
      sliverF: makeColumn('sliver-f'),
      validityF: makeColumn('validity-f'),
      colorF: makeColumn('color-f', partCount)
    };
    const centroidsPlanar = makeColumn('centroids-planar', featureCount * 2);
    const centroidsWgs84 = makeColumn('centroids-wgs84', featureCount * 2);
    const boundsWgs84 = makeColumn('bounds-wgs84', featureCount * 4);
    const vertexCounts = makeColumn('vertex-counts');
    const groupAreas = makeColumn('group-areas', groupCount);
    const groupCentroids = makeColumn('group-centroids', groupCount * 2);
    const groupFeatureCounts = makeColumn('group-feature-counts', groupCount);

    const labelPoints = makeColumn('label-points', featureCount * 2);
    const labelRadii = makeColumn('label-radii');
    const labelDegenerate = makeColumn('label-degenerate');
    const clockwise = makeColumn('clockwise');
    const sliver = makeColumn('sliver');
    const orientation = makeColumn('orientation');
    const circleStarts = makeColumn('circle-starts', featureCount * CIRCLE_SEGMENTS * 2);
    const circleEnds = makeColumn('circle-ends', featureCount * CIRCLE_SEGMENTS * 2);
    const axisStarts = makeColumn('axis-starts', featureCount * 2);
    const axisEnds = makeColumn('axis-ends', featureCount * 2);
    const labelLngLat = makeColumn('label-lnglat', featureCount * 2);
    const boundsStarts = makeColumn('bounds-starts', featureCount * 4 * 2);
    const boundsEnds = makeColumn('bounds-ends', featureCount * 4 * 2);
    const selectedStarts = makeColumn('selected-starts', selectedEdgeCapacity * 2);
    const selectedEnds = makeColumn('selected-ends', selectedEdgeCapacity * 2);
    const mask = makeColumn('validity-mask');

    const shapeParameters = resources.createParameterBuffer(
      'shape-parameters',
      'float32',
      GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH,
      getGPUShapeDescriptorsParameterValues({sliverThreshold: ctx.options.sliverThreshold})
    );

    // Coordinate conversion shared by the drawing kernels.
    const toLngLat = (() => {
      if (spec.space === 'mercator') {
        return /* wgsl */ `
fn toLngLat(p: vec2<f32>) -> vec2<f32> {
  let r = ${MERCATOR_RADIUS.toFixed(1)};
  return vec2<f32>(p.x / r * 57.29578, (2.0 * atan(exp(p.y / r)) - 1.5707963) * 57.29578);
}`;
      }
      const [metersPerDegreeX] = projection.project(origin[0] + 1, origin[1]);
      const [, metersPerDegreeY] = projection.project(origin[0], origin[1] + 1);
      return /* wgsl */ `
fn toLngLat(p: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(${origin[0]}, ${origin[1]}) + p / vec2<f32>(${metersPerDegreeX.toFixed(3)}, ${metersPerDegreeY.toFixed(3)});
}`;
    })();

    // ---- Measures graph: three coordinate systems, groups, derived columns ----
    const buildMeasures = ({holeRule}: PolygonMeasuresOptions) => {
      const graph = new GPUCommandGraph<void>(device, {id: `measures-${id}`});
      const imp = createGraphImporter(graph);
      const lngLat = imp('lnglat', lngLatBuffer, 'float32x2', vertexCount);
      const planarView = imp('planar', planarBuffer, 'float32x2', vertexCount);
      const ringOffsets = imp('ring-offsets', ringOffsetsBuffer, 'uint32', ringCount + 1);
      const featureRings = imp('feature-rings', featureRingsBuffer, 'uint32', featureCount + 1);
      const view = (name: string, buffer: Buffer, length = featureCount) =>
        imp(name, buffer, 'float32', length);
      const areaPlanar = view('area-planar', columns.areaPlanar);
      const areaWgs84 = view('area-wgs84', columns.areaWgs84);
      const common = {
        geometryType: 'polygons' as const,
        holeRule,
        featureRingOffsets: featureRings
      };
      graph.add(
        new GPUGeometryMeasures({
          id: 'measures-planar',
          positions: planarView,
          ringOffsets,
          ...common,
          coordinateSystem: 'planar',
          output: {
            areas: areaPlanar,
            lengths: view('length-planar', columns.lengthPlanar),
            centroids: imp('centroids-planar', centroidsPlanar, 'float32x2', featureCount)
          }
        })
      );
      graph.add(
        new GPUGeometryMeasures({
          id: 'measures-sphere',
          positions: lngLat,
          ringOffsets,
          ...common,
          coordinateSystem: 'spherical',
          output: {
            areas: view('area-sphere', columns.areaSphere),
            lengths: view('length-sphere', columns.lengthSphere)
          }
        })
      );
      const groupArrays = spec.groupColumn
        ? {
            groupIds: imp('group-ids', groupIdsBuffer, 'uint32', featureCount),
            groupCount,
            groupOutput: {
              areas: view('group-areas', groupAreas, groupCount),
              centroids: imp('group-centroids', groupCentroids, 'float32x2', groupCount),
              featureCounts: imp('group-feature-counts', groupFeatureCounts, 'uint32', groupCount)
            }
          }
        : {};
      graph.add(
        new GPUGeometryMeasures({
          id: 'measures-wgs84',
          positions: lngLat,
          ringOffsets,
          ...common,
          coordinateSystem: 'wgs84',
          output: {
            areas: areaWgs84,
            lengths: view('length-wgs84', columns.lengthWgs84),
            centroids: imp('centroids-wgs84', centroidsWgs84, 'float32x2', featureCount),
            bounds: imp('bounds-wgs84', boundsWgs84, 'float32x4', featureCount),
            vertexCounts: imp('vertex-counts', vertexCounts, 'uint32', featureCount)
          },
          ...groupArrays
        })
      );
      const ratioView = view('ratio', columns.ratio);
      addKernelPass(graph, {
        id: 'area-distortion',
        invocationCount: featureCount,
        bindings: [
          {name: 'planar', view: areaPlanar, type: 'f32', access: 'read'},
          {name: 'ellipsoid', view: areaWgs84, type: 'f32', access: 'read'},
          {name: 'ratio', view: ratioView, type: 'f32', access: 'read_write'}
        ],
        body: /* wgsl */ `ratio[ratioOffset + index] = planar[planarOffset + index] / max(ellipsoid[ellipsoidOffset + index], 1e-9);`
      });
      addKernelPass(graph, {
        id: 'vertex-count-to-float',
        invocationCount: featureCount,
        bindings: [
          {
            name: 'counts',
            view: imp('vertex-counts', vertexCounts, 'uint32', featureCount),
            type: 'u32',
            access: 'read'
          },
          {
            name: 'values',
            view: view('vertex-f', columns.vertexF),
            type: 'f32',
            access: 'read_write'
          }
        ],
        body: /* wgsl */ `values[valuesOffset + index] = f32(counts[countsOffset + index]);`
      });
      addKernelPass(graph, {
        id: 'gather-group-area',
        invocationCount: featureCount,
        bindings: [
          {
            name: 'groups',
            view: imp('group-ids', groupIdsBuffer, 'uint32', featureCount),
            type: 'u32',
            access: 'read'
          },
          {
            name: 'totals',
            view: view('group-areas', groupAreas, groupCount),
            type: 'f32',
            access: 'read'
          },
          {
            name: 'values',
            view: view('group-area-f', columns.groupAreaF),
            type: 'f32',
            access: 'read_write'
          }
        ],
        declarations: `const HAS_GROUPS: bool = ${spec.groupColumn !== null};`,
        body: /* wgsl */ `
  var value = f32(index) * 0.0;
  value = value / value;
  if (HAS_GROUPS) {
    value = totals[totalsOffset + groups[groupsOffset + index]];
  }
  values[valuesOffset + index] = value;`
      });
      addKernelPass(graph, {
        id: 'bounds-boxes',
        invocationCount: featureCount,
        bindings: [
          {
            name: 'bounds',
            view: imp('bounds-wgs84', boundsWgs84, 'float32x4', featureCount),
            type: 'f32',
            access: 'read'
          },
          {
            name: 'starts',
            view: imp('bounds-starts', boundsStarts, 'float32', featureCount * 8),
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'ends',
            view: imp('bounds-ends', boundsEnds, 'float32', featureCount * 8),
            type: 'f32',
            access: 'read_write'
          }
        ],
        body: /* wgsl */ `
  let b = vec4<f32>(
    bounds[boundsOffset + index * 4u], bounds[boundsOffset + index * 4u + 1u],
    bounds[boundsOffset + index * 4u + 2u], bounds[boundsOffset + index * 4u + 3u]
  );
  var corners = array<vec2<f32>, 4>(
    vec2<f32>(b.x, b.y), vec2<f32>(b.z, b.y), vec2<f32>(b.z, b.w), vec2<f32>(b.x, b.w)
  );
  for (var side = 0u; side < 4u; side++) {
    let a = corners[side];
    let c = corners[(side + 1u) % 4u];
    let slot = (index * 4u + side) * 2u;
    starts[startsOffset + slot] = a.x;
    starts[startsOffset + slot + 1u] = a.y;
    ends[endsOffset + slot] = c.x;
    ends[endsOffset + slot + 1u] = c.y;
  }`
      });
      return resources.track(graph.compile());
    };

    // ---- Shape graph: label points, descriptors, drawing kernels ----
    const buildShape = (options: PolygonMeasuresOptions) => {
      const graph = new GPUCommandGraph<void>(device, {id: `shape-${id}`});
      const imp = createGraphImporter(graph);
      const planarView = imp('planar', planarBuffer, 'float32x2', vertexCount);
      const ringOffsets = imp('ring-offsets', ringOffsetsBuffer, 'uint32', ringCount + 1);
      const featureRings = imp('feature-rings', featureRingsBuffer, 'uint32', featureCount + 1);
      const points = imp('label-points', labelPoints, 'float32x2', featureCount);
      const radii = imp('label-radii', labelRadii, 'float32', featureCount);
      const degenerate = imp('label-degenerate', labelDegenerate, 'uint32', featureCount);
      graph.add(
        new GPULabelPoint({
          id: 'label-point',
          positions: planarView,
          ringOffsets,
          featureRingOffsets: featureRings,
          initialGridSize: options.initialGridSize,
          refinementCandidates: options.refinementCandidates,
          output: {points, distances: radii, degenerate}
        })
      );
      const view = (name: string, buffer: Buffer) => imp(name, buffer, 'float32', featureCount);
      const elongationView = view('elongation', columns.elongation);
      const orientationView = view('orientation', orientation);
      graph.add(
        new GPUShapeDescriptors({
          id: 'shape-descriptors',
          positions: planarView,
          ringOffsets,
          featureRingOffsets: featureRings,
          holeRule: options.holeRule,
          parameters: shapeParameters.importToGraph(graph),
          output: {
            polsbyPopper: view('polsby-popper', columns.polsbyPopper),
            schwartzberg: view('schwartzberg', columns.schwartzberg),
            elongation: elongationView,
            orientation: orientationView,
            convexity: view('convexity', columns.convexity),
            clockwise: imp('clockwise', clockwise, 'uint32', featureCount),
            sliver: imp('sliver', sliver, 'uint32', featureCount)
          }
        })
      );
      addKernelPass(graph, {
        id: 'sliver-to-float',
        invocationCount: featureCount,
        bindings: [
          {
            name: 'flags',
            view: imp('sliver', sliver, 'uint32', featureCount),
            type: 'u32',
            access: 'read'
          },
          {
            name: 'values',
            view: view('sliver-f', columns.sliverF),
            type: 'f32',
            access: 'read_write'
          }
        ],
        body: /* wgsl */ `values[valuesOffset + index] = f32(flags[flagsOffset + index]);`
      });
      addKernelPass(graph, {
        id: 'label-drawing',
        invocationCount: featureCount * CIRCLE_SEGMENTS,
        bindings: [
          {name: 'points', view: points, type: 'f32', access: 'read'},
          {name: 'radii', view: radii, type: 'f32', access: 'read'},
          {name: 'degenerate', view: degenerate, type: 'u32', access: 'read'},
          {
            name: 'lngLat',
            view: imp('label-lnglat', labelLngLat, 'float32', featureCount * 2),
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'starts',
            view: imp('circle-starts', circleStarts, 'float32', featureCount * CIRCLE_SEGMENTS * 2),
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'ends',
            view: imp('circle-ends', circleEnds, 'float32', featureCount * CIRCLE_SEGMENTS * 2),
            type: 'f32',
            access: 'read_write'
          }
        ],
        declarations: `const SEGMENTS: u32 = ${CIRCLE_SEGMENTS}u;\n${toLngLat}`,
        body: /* wgsl */ `
  let feature = index / SEGMENTS;
  let step = index % SEGMENTS;
  let center = vec2<f32>(points[pointsOffset + feature * 2u], points[pointsOffset + feature * 2u + 1u]);
  let radius = radii[radiiOffset + feature];
  if (step == 0u) {
    let anchor = toLngLat(center);
    lngLat[lngLatOffset + feature * 2u] = anchor.x;
    lngLat[lngLatOffset + feature * 2u + 1u] = anchor.y;
  }
  let zero = f32(index) * 0.0;
  var start = vec2<f32>(zero / zero);
  var end = start;
  if (degenerate[degenerateOffset + feature] == 0u && radius == radius) {
    let angle0 = 6.2831853 * f32(step) / f32(SEGMENTS);
    let angle1 = 6.2831853 * f32(step + 1u) / f32(SEGMENTS);
    start = toLngLat(center + radius * vec2<f32>(cos(angle0), sin(angle0)));
    end = toLngLat(center + radius * vec2<f32>(cos(angle1), sin(angle1)));
  }
  starts[startsOffset + index * 2u] = start.x;
  starts[startsOffset + index * 2u + 1u] = start.y;
  ends[endsOffset + index * 2u] = end.x;
  ends[endsOffset + index * 2u + 1u] = end.y;`
      });
      addKernelPass(graph, {
        id: 'axis-glyphs',
        invocationCount: featureCount,
        bindings: [
          {name: 'points', view: points, type: 'f32', access: 'read'},
          {name: 'radii', view: radii, type: 'f32', access: 'read'},
          {name: 'orientation', view: orientationView, type: 'f32', access: 'read'},
          {name: 'elongation', view: elongationView, type: 'f32', access: 'read'},
          {
            name: 'starts',
            view: imp('axis-starts', axisStarts, 'float32', featureCount * 2),
            type: 'f32',
            access: 'read_write'
          },
          {
            name: 'ends',
            view: imp('axis-ends', axisEnds, 'float32', featureCount * 2),
            type: 'f32',
            access: 'read_write'
          }
        ],
        declarations: toLngLat,
        body: /* wgsl */ `
  let center = vec2<f32>(points[pointsOffset + index * 2u], points[pointsOffset + index * 2u + 1u]);
  let angle = orientation[orientationOffset + index];
  let halfLength = max(radii[radiiOffset + index], 0.0) * (1.0 + 3.0 * elongation[elongationOffset + index]);
  let direction = vec2<f32>(cos(angle), sin(angle)) * halfLength;
  let start = toLngLat(center - direction);
  let end = toLngLat(center + direction);
  starts[startsOffset + index * 2u] = start.x;
  starts[startsOffset + index * 2u + 1u] = start.y;
  ends[endsOffset + index * 2u] = end.x;
  ends[endsOffset + index * 2u + 1u] = end.y;`
      });
      return resources.track(graph.compile());
    };

    // ---- Validity graph ----
    const buildValidity = (options: PolygonMeasuresOptions) => {
      const graph = new GPUCommandGraph<void>(device, {id: `validity-${id}`});
      const imp = createGraphImporter(graph);
      const maskView = imp('mask', mask, 'uint32', featureCount);
      const overflow = resources.createBuffer('validity-overflow', 4);
      graph.add(
        new GPUGeometryValidity({
          id: 'validity',
          polygons: {
            kind: 'polygons',
            positions: imp('validity-positions', validityPositions, 'float32x2', vertexCount),
            featureOffsets: imp(
              'feature-offsets',
              featureOffsetsBuffer,
              'uint32',
              featureCount + 1
            ),
            polygonOffsets: imp('polygon-offsets', polygonOffsetsBuffer, 'uint32', partCount + 1),
            ringOffsets: imp('ring-offsets', ringOffsetsBuffer, 'uint32', ringCount + 1)
          },
          mask: maskView,
          overflow: imp('validity-overflow', overflow, 'uint32', 1),
          intersectionCapacity: Math.max(4096, featureCount * 4),
          ringClosure: options.ringClosure,
          orientation: options.orientation
        })
      );
      addKernelPass(graph, {
        id: 'validity-category',
        invocationCount: featureCount,
        bindings: [
          {name: 'mask', view: maskView, type: 'u32', access: 'read'},
          {
            name: 'category',
            view: imp('validity-f', columns.validityF, 'float32', featureCount),
            type: 'f32',
            access: 'read_write'
          }
        ],
        declarations: `const STRUCTURAL: u32 = ${GPU_GEOMETRY_VALIDITY_STRUCTURAL_MASK}u;`,
        body: /* wgsl */ `
  let bits = mask[maskOffset + index];
  var value = ${VALIDITY_PALETTE_INDEXES[0]}.0;
  if ((bits & STRUCTURAL) != 0u) {
    value = ${VALIDITY_PALETTE_INDEXES[2]}.0;
  } else if (bits != 0u) {
    value = ${VALIDITY_PALETTE_INDEXES[1]}.0;
  }
  category[categoryOffset + index] = value;`
      });
      return resources.track(graph.compile());
    };

    // ---- Coloring graph ----
    const neighborCapacity = Math.max(1024, partCount * 24);
    const weightOffsets = resources.createBuffer('weight-offsets', (partCount + 1) * 4);
    const weightNeighbors = resources.createBuffer('weight-neighbors', neighborCapacity * 4);
    const weightValues = resources.createBuffer('weight-values', neighborCapacity * 4);
    const weightOverflow = resources.createBuffer('weight-overflow', 4);
    const colors = resources.createBuffer('colors', partCount * 4);
    const colorSummary = resources.createBuffer('color-summary', 4);
    const colorSummaryParts = {
      conflict: resources.createBuffer('conflict-count', 4),
      converged: resources.createBuffer('coloring-converged', 4),
      rounds: resources.createBuffer('coloring-rounds', 4)
    };
    const buildColoring = (options: PolygonMeasuresOptions) => {
      const graph = new GPUCommandGraph<void>(device, {id: `coloring-${id}`});
      const imp = createGraphImporter(graph);
      const weights = {
        offsets: imp('weight-offsets', weightOffsets, 'uint32', partCount + 1),
        neighbors: imp('weight-neighbors', weightNeighbors, 'uint32', neighborCapacity),
        weights: imp('weight-values', weightValues, 'float32', neighborCapacity)
      };
      graph.add(
        new GPUContiguityWeights({
          id: 'contiguity',
          criterion: options.contiguity,
          positions: imp('lnglat', lngLatBuffer, 'float32x2', vertexCount),
          ringOffsets: imp('ring-offsets', ringOffsetsBuffer, 'uint32', ringCount + 1),
          polygonOffsets: imp('polygon-offsets', polygonOffsetsBuffer, 'uint32', partCount + 1),
          weights,
          // Shared boundaries emit one directed pair per shared vertex or edge before deduplication.
          pairCapacity: Math.max(65536, vertexCount * 16),
          overflow: imp('weight-overflow', weightOverflow, 'uint32', 1)
        })
      );
      const colorsView = imp('colors', colors, 'uint32', partCount);
      graph.add(
        new GPUMapColoring({
          id: 'coloring',
          weights,
          colors: colorsView,
          colorCount: imp('color-count', colorSummary, 'uint32', 1),
          seed: options.colorSeed,
          maximumRounds: options.maxRounds,
          conflictCount: imp('conflict-count', colorSummaryParts.conflict, 'uint32', 1),
          converged: imp('coloring-converged', colorSummaryParts.converged, 'uint32', 1),
          roundCount: imp('coloring-rounds', colorSummaryParts.rounds, 'uint32', 1)
        })
      );
      addKernelPass(graph, {
        id: 'colors-to-float',
        invocationCount: partCount,
        bindings: [
          {name: 'colors', view: colorsView, type: 'u32', access: 'read'},
          {
            name: 'values',
            view: imp('color-f', columns.colorF, 'float32', partCount),
            type: 'f32',
            access: 'read_write'
          }
        ],
        body: /* wgsl */ `values[valuesOffset + index] = f32(colors[colorsOffset + index]);`
      });
      return resources.track(graph.compile());
    };
    const next: DatasetState = {
      id,
      layout,
      names,
      groupIds,
      groupCount,
      resources,
      fill,
      edges,
      columns,
      overlays: {
        labelPoints: labelLngLat,
        circleStarts,
        circleEnds,
        axisStarts,
        axisEnds,
        centroids: centroidsWgs84,
        boundsStarts,
        boundsEnds,
        selectedStarts,
        selectedEnds
      },
      selectedEdgeCapacity,
      builders: {
        measures: buildMeasures,
        shape: buildShape,
        validity: buildValidity,
        coloring: buildColoring
      },
      measures: buildMeasures(ctx.options),
      shape: buildShape(ctx.options),
      validity: buildValidity(ctx.options),
      coloring: buildColoring(ctx.options),
      shapeParameters,
      validityPositions,
      tableReader: null,
      coloringReader: null,
      table: null,
      coloringTable: null,
      dirty: {measures: true, shape: true, validity: true, coloring: true, table: true},
      selected: -1,
      selectedCount: 0
    };

    // Readers: one table of per-feature columns, one coloring summary.
    const float = (buffer: Buffer, length: number) => ({buffer, size: length * 4});
    const tableSources = [
      float(columns.areaPlanar, featureCount),
      float(columns.areaSphere, featureCount),
      float(columns.areaWgs84, featureCount),
      float(columns.lengthPlanar, featureCount),
      float(columns.lengthSphere, featureCount),
      float(columns.lengthWgs84, featureCount),
      float(vertexCounts, featureCount),
      float(columns.polsbyPopper, featureCount),
      float(columns.schwartzberg, featureCount),
      float(columns.elongation, featureCount),
      float(columns.convexity, featureCount),
      float(sliver, featureCount),
      float(clockwise, featureCount),
      float(labelPoints, featureCount * 2),
      float(labelRadii, featureCount),
      float(centroidsWgs84, featureCount * 2),
      float(mask, featureCount),
      float(groupAreas, groupCount)
    ];
    next.tableReader = new SummaryReader(resources, `table-${id}`, tableSources, bytes => {
      if (destroyed || state !== next) return;
      let offset = 0;
      const f32 = (length: number) => {
        const view = new Float32Array(bytes, offset, length);
        offset += length * 4;
        return view;
      };
      const u32 = (length: number) => {
        const view = new Uint32Array(bytes, offset, length);
        offset += length * 4;
        return view;
      };
      next.table = {
        areaPlanar: f32(featureCount),
        areaSphere: f32(featureCount),
        areaWgs84: f32(featureCount),
        lengthPlanar: f32(featureCount),
        lengthSphere: f32(featureCount),
        lengthWgs84: f32(featureCount),
        vertexCount: u32(featureCount),
        polsbyPopper: f32(featureCount),
        schwartzberg: f32(featureCount),
        elongation: f32(featureCount),
        convexity: f32(featureCount),
        sliver: u32(featureCount),
        clockwise: u32(featureCount),
        labelPoint: f32(featureCount * 2),
        radius: f32(featureCount),
        centroid: f32(featureCount * 2),
        mask: u32(featureCount),
        groupAreas: f32(groupCount)
      };
      publishTable(next);
    });
    next.coloringReader = new SummaryReader(
      resources,
      `coloring-${id}`,
      [
        {buffer: colorSummary, size: 4},
        {buffer: colorSummaryParts.conflict, size: 4},
        {buffer: colorSummaryParts.converged, size: 4},
        {buffer: colorSummaryParts.rounds, size: 4},
        {buffer: weightOverflow, size: 4},
        {buffer: weightOffsets, size: (partCount + 1) * 4}
      ],
      bytes => {
        if (destroyed || state !== next) return;
        const words = new Uint32Array(bytes);
        next.coloringTable = {
          colorCount: words[0],
          conflictCount: words[1],
          converged: words[2],
          roundCount: words[3],
          overflow: words[4],
          adjacencies: words[5 + partCount]
        };
        publishColoring(next);
      }
    );
    return next;
  }

  // -------------------------------------------------------------------------------------------
  // Readback handling
  // -------------------------------------------------------------------------------------------

  function getMetricColumn(
    table: FeatureTable,
    options: PolygonMeasuresOptions
  ): Float32Array | null {
    const system = options.areaSystem;
    switch (options.metric) {
      case 'area':
        return system === 'planar'
          ? table.areaPlanar
          : system === 'spherical'
            ? table.areaSphere
            : table.areaWgs84;
      case 'perimeter':
        return system === 'planar'
          ? table.lengthPlanar
          : system === 'spherical'
            ? table.lengthSphere
            : table.lengthWgs84;
      case 'areaDistortion': {
        const ratio = new Float32Array(table.areaPlanar.length);
        for (let index = 0; index < ratio.length; index++) {
          ratio[index] = table.areaPlanar[index] / Math.max(table.areaWgs84[index], 1e-9);
        }
        return ratio;
      }
      case 'vertices':
        return Float32Array.from(table.vertexCount);
      case 'groupArea': {
        const active = state;
        if (!active) return null;
        const values = new Float32Array(active.groupIds.length);
        for (let index = 0; index < values.length; index++) {
          values[index] = table.groupAreas[active.groupIds[index]];
        }
        return values;
      }
      case 'compactness':
        return table.polsbyPopper;
      case 'schwartzberg':
        return table.schwartzberg;
      case 'elongation':
        return table.elongation;
      case 'convexity':
        return table.convexity;
      default:
        return null;
    }
  }

  /** Value range of a continuous metric: robust percentiles so outliers do not wash out the ramp. */
  function getMetricRange(metric: PolygonMetric, values: Float32Array): [number, number] {
    const low = getQuantile(values, metric === 'areaDistortion' ? 0 : 0.02);
    const high = getQuantile(values, 0.98);
    if (metric === 'compactness' || metric === 'schwartzberg' || metric === 'convexity') {
      return [0, Math.max(high, 0.05)];
    }
    if (metric === 'elongation') return [0, Math.max(high, 0.05)];
    if (metric === 'areaDistortion') return [Math.min(low, 1), Math.max(high, 1.01)];
    return [Math.min(low, high), Math.max(high, low + 1e-6)];
  }

  let currentRange: [number, number] = [0, 1];

  function publishTable(active: DatasetState): void {
    const table = active.table;
    if (!table) return;
    const options = ctx.options;
    const featureCount = active.layout.featureCount;
    const values = getMetricColumn(table, options);
    if (values) {
      currentRange = getMetricRange(options.metric, values);
      ctx.setLegendExtent('value', currentRange);
      ctx.requestLayers();
    }
    ctx.setReadout('features', featureCount);
    ctx.setReadout('vertices', active.layout.vertexCount);
    let planarTotal = 0;
    let sphereTotal = 0;
    let wgsTotal = 0;
    let maximumDistortion = 0;
    let maximumDistortionRow = 0;
    let slivers = 0;
    let compactnessSum = 0;
    for (let feature = 0; feature < featureCount; feature++) {
      planarTotal += table.areaPlanar[feature];
      sphereTotal += table.areaSphere[feature];
      wgsTotal += table.areaWgs84[feature];
      const ratio = table.areaPlanar[feature] / Math.max(table.areaWgs84[feature], 1e-9);
      if (ratio > maximumDistortion) {
        maximumDistortion = ratio;
        maximumDistortionRow = feature;
      }
      slivers += table.sliver[feature];
      compactnessSum += table.polsbyPopper[feature];
    }
    ctx.setReadout('areaPlanar', `${formatNumber(planarTotal / 1e6)} km²`);
    ctx.setReadout('areaSphere', `${formatNumber(sphereTotal / 1e6)} km²`);
    ctx.setReadout('areaWgs84', `${formatNumber(wgsTotal / 1e6)} km²`);
    ctx.setReadout(
      'distortion',
      `${maximumDistortion.toFixed(2)}× (${active.names[maximumDistortionRow]})`
    );
    ctx.setReadout('meanCompactness', (compactnessSum / featureCount).toFixed(3));
    ctx.setReadout(
      'slivers',
      `${formatCount(slivers)} (${((100 * slivers) / featureCount).toFixed(1)}%)`
    );
    // Centroids that fall outside their own polygon (concave shapes).
    let outside = 0;
    for (let feature = 0; feature < featureCount; feature++) {
      const x = table.centroid[feature * 2];
      const y = table.centroid[feature * 2 + 1];
      if (Number.isFinite(x) && !isPointInFeature(active.layout, feature, x, y)) outside++;
    }
    ctx.setReadout('centroidOutside', `${formatCount(outside)} of ${formatCount(featureCount)}`);
    // Validity bits.
    const counts: Record<string, number> = {};
    for (const [bitName, bit] of Object.entries(GPU_GEOMETRY_VALIDITY_BIT)) {
      let count = 0;
      for (let feature = 0; feature < featureCount; feature++)
        if (table.mask[feature] & bit) count++;
      if (count > 0) counts[bitName] = count;
    }
    const entries = Object.entries(counts);
    ctx.setReadout(
      'validity',
      entries.length
        ? entries.map(([name, count]) => `${name} ${formatCount(count)}`).join(', ')
        : 'all valid'
    );
    if (active.selected >= 0) updateSelectedReadout(active);
  }

  function publishColoring(active: DatasetState): void {
    const summary = active.coloringTable;
    if (!summary) return;
    ctx.setReadout(
      'coloring',
      `${summary.colorCount} colors, ${summary.conflictCount} conflicts, ${summary.roundCount} rounds${summary.converged ? '' : ' (not converged)'}`
    );
    ctx.setReadout('adjacencies', summary.adjacencies);
    if (summary.overflow) ctx.setStatus('Contiguity capacity overflowed.');
  }

  function updateSelectedReadout(active: DatasetState): void {
    const table = active.table;
    const feature = active.selected;
    if (!table || feature < 0) {
      ctx.setReadout('selected', null);
      return;
    }
    ctx.setReadout(
      'selected',
      `${active.names[feature]}: ${formatNumber(table.areaWgs84[feature] / 1e6, 1)} km² (planar ${formatNumber(table.areaPlanar[feature] / 1e6, 1)}), compactness ${table.polsbyPopper[feature].toFixed(2)}`
    );
  }

  // -------------------------------------------------------------------------------------------
  // Defects (test data for the validity checks)
  // -------------------------------------------------------------------------------------------

  /** Writes the validity positions: the clean layer, or a copy with six injected defects. */
  function writeValidityPositions(active: DatasetState, inject: boolean): void {
    const layout = active.layout;
    const positions = Float32Array.from(layout.lngLat);
    if (inject) {
      const f = layout.featureCount;
      const used = new Set<number>();
      // First ring of a feature near k / 7 of the way through the layer with at least 12 vertices.
      const pickRing = (k: number): number => {
        let feature = Math.floor(((k + 1) * f) / 7);
        for (let attempt = 0; attempt < f; attempt++, feature = (feature + 1) % f) {
          const ring = layout.featureRingOffsets[feature];
          if (!used.has(ring) && layout.ringOffsets[ring + 1] - layout.ringOffsets[ring] >= 12) {
            used.add(ring);
            return ring;
          }
        }
        return layout.featureRingOffsets[Math.floor(((k + 1) * f) / 7)];
      };
      const rings = pickRing;
      const ringRange = (ring: number): [number, number] => [
        layout.ringOffsets[ring],
        layout.ringOffsets[ring + 1]
      ];
      const set = (vertex: number, x: number, y: number) => {
        positions[vertex * 2] = x;
        positions[vertex * 2 + 1] = y;
      };
      const get = (vertex: number): [number, number] => [
        positions[vertex * 2],
        positions[vertex * 2 + 1]
      ];
      // 0: bow-tie (swap two vertices from opposite sides of the ring).
      {
        const [a, b] = ringRange(rings(0));
        const other = a + Math.floor((b - a) / 2);
        const p = get(a + 1);
        const q = get(other);
        set(a + 1, q[0], q[1]);
        set(other, p[0], p[1]);
      }
      // 1: NaN coordinate.
      {
        const [a] = ringRange(rings(1));
        set(a + 1, Number.NaN, Number.NaN);
      }
      // 2: repeated vertex.
      {
        const [a] = ringRange(rings(2));
        const p = get(a + 1);
        set(a + 2, p[0], p[1]);
      }
      // 3: unclosed ring (only reported with ringClosure 'explicit').
      {
        const [, b] = ringRange(rings(3));
        const p = get(b - 1);
        set(b - 1, p[0] + 0.01, p[1] + 0.006);
      }
      // 4: spike (one vertex dragged to the far side of the ring).
      {
        const [a, b] = ringRange(rings(4));
        const p = get(a + Math.floor((b - a) / 2) + 1);
        set(a + 2, p[0], p[1]);
      }
      // 5: reversed winding.
      {
        const [a, b] = ringRange(rings(5));
        for (let low = a, high = b - 1; low < high; low++, high--) {
          const p = get(low);
          const q = get(high);
          set(low, q[0], q[1]);
          set(high, p[0], p[1]);
        }
      }
    }
    // Buffer writes copy; the typed array is a fresh copy so the dataset memo is not touched.
    (active.validityPositions as Buffer).write(positions);
  }

  // -------------------------------------------------------------------------------------------
  // Instance
  // -------------------------------------------------------------------------------------------

  const activate = (id: PolygonDatasetId) => {
    if (state) {
      state.tableReader?.stop();
      state.coloringReader?.stop();
      state.resources.destroy();
    }
    state = buildDataset(id);
    writeValidityPositions(state, ctx.options.injectDefects);
    ctx.setStatus('');
  };

  activate(ctx.options.dataset);

  const settle = {lastChange: performance.now(), pendingTable: true, pendingColoring: true};
  const touch = () => {
    settle.lastChange = performance.now();
  };

  const selectFeature = (feature: number) => {
    const active = state;
    if (!active) return;
    active.selected = feature;
    if (feature >= 0) {
      const layout = active.layout;
      const starts: number[] = [];
      const ends: number[] = [];
      for (
        let ring = layout.featureRingOffsets[feature];
        ring < layout.featureRingOffsets[feature + 1];
        ring++
      ) {
        const first = layout.ringOffsets[ring];
        const last = layout.ringOffsets[ring + 1];
        for (let vertex = first; vertex < last; vertex++) {
          const following = vertex + 1 < last ? vertex + 1 : first;
          starts.push(layout.lngLat[vertex * 2], layout.lngLat[vertex * 2 + 1]);
          ends.push(layout.lngLat[following * 2], layout.lngLat[following * 2 + 1]);
        }
      }
      active.selectedCount = starts.length / 2;
      (active.overlays.selectedStarts as Buffer).write(Float32Array.from(starts));
      (active.overlays.selectedEnds as Buffer).write(Float32Array.from(ends));
    } else {
      active.selectedCount = 0;
    }
    updateSelectedReadout(active);
    ctx.requestLayers();
  };

  return {
    getCompiledGraphs() {
      if (!state) return [];
      return [state.measures, state.shape, state.validity, state.coloring].filter(
        Boolean
      ) as CompiledGPUCommandGraph<never>[];
    },

    setOption(id, _value, options) {
      const active = state;
      if (!active) return;
      if (id === 'dataset') {
        activate(options.dataset);
        ctx.requestLayers();
        return;
      }
      if (id === 'holeRule') {
        active.resources.release(active.measures);
        active.measures = active.builders.measures(options);
        if (active.shape) active.resources.release(active.shape);
        active.shape = active.builders.shape(options);
        active.dirty.measures = active.dirty.shape = active.dirty.table = true;
      } else if (id === 'sliverThreshold') {
        active.shapeParameters.write(
          getGPUShapeDescriptorsParameterValues({sliverThreshold: options.sliverThreshold})
        );
        active.dirty.shape = active.dirty.table = true;
      } else if (id === 'initialGridSize' || id === 'refinementCandidates') {
        if (active.shape) active.resources.release(active.shape);
        active.shape = active.builders.shape(options);
        active.dirty.shape = active.dirty.table = true;
      } else if (id === 'orientation' || id === 'ringClosure') {
        if (active.validity) active.resources.release(active.validity);
        active.validity = active.builders.validity(options);
        active.dirty.validity = active.dirty.table = true;
      } else if (id === 'injectDefects') {
        writeValidityPositions(active, options.injectDefects);
        active.dirty.validity = active.dirty.table = true;
      } else if (id === 'contiguity' || id === 'colorSeed' || id === 'maxRounds') {
        if (active.coloring) active.resources.release(active.coloring);
        active.coloring = active.builders.coloring(options);
        active.dirty.coloring = true;
      } else if (id === 'areaSystem' || id === 'metric') {
        active.dirty.table = true;
        ctx.setLegendExtent('value', currentRange);
      }
      touch();
      ctx.requestLayers();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    getTooltip(event) {
      const active = state;
      if (!active || !event.coordinate) return null;
      const feature = findFeatureAt(active.layout, event.coordinate[0], event.coordinate[1]);
      if (feature < 0) return null;
      const table = active.table;
      if (!table) return active.names[feature];
      return `${active.names[feature]} · ${formatNumber(table.areaWgs84[feature] / 1e6, 1)} km² · compactness ${table.polsbyPopper[feature].toFixed(2)}`;
    },

    onClick(event) {
      const active = state;
      if (!active || !event.coordinate) return false;
      const feature = findFeatureAt(active.layout, event.coordinate[0], event.coordinate[1]);
      selectFeature(feature === active.selected ? -1 : feature);
      return true;
    },

    encode(commandEncoder) {
      const active = state;
      if (!active) return;
      if (active.dirty.measures) {
        active.measures.encode(commandEncoder, {parameters: undefined});
        active.dirty.measures = false;
      }
      if (active.dirty.shape && active.shape) {
        active.shape.encode(commandEncoder, {parameters: undefined});
        active.dirty.shape = false;
      }
      if (active.dirty.validity && active.validity) {
        active.validity.encode(commandEncoder, {parameters: undefined});
        active.dirty.validity = false;
      }
      if (active.dirty.coloring && active.coloring) {
        active.coloring.encode(commandEncoder, {parameters: undefined});
        active.dirty.coloring = false;
        active.coloringReader?.request(commandEncoder);
      }
      if (active.dirty.table && performance.now() - settle.lastChange > 120) {
        active.tableReader?.request(commandEncoder);
        active.dirty.table = false;
      } else {
        active.tableReader?.flush(commandEncoder);
        active.coloringReader?.flush(commandEncoder);
      }
    },

    getLayers() {
      const active = state;
      if (!active) return [];
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      const metric = options.metric;
      const columns = active.columns;
      let values: Buffer | null = null;
      let mapping: B3ValueMapping = 'ramp';
      let rows = active.fill.featureRows;
      let range: [number, number] = currentRange;
      let sqrtScale = false;
      let secondary: readonly [number, number, number, number] | undefined;
      const system = options.areaSystem;
      switch (metric) {
        case 'area':
          values =
            system === 'planar'
              ? columns.areaPlanar
              : system === 'spherical'
                ? columns.areaSphere
                : columns.areaWgs84;
          sqrtScale = true;
          break;
        case 'perimeter':
          values =
            system === 'planar'
              ? columns.lengthPlanar
              : system === 'spherical'
                ? columns.lengthSphere
                : columns.lengthWgs84;
          break;
        case 'areaDistortion':
          values = columns.ratio;
          break;
        case 'vertices':
          values = columns.vertexF;
          sqrtScale = true;
          break;
        case 'groupArea':
          values = columns.groupAreaF;
          break;
        case 'compactness':
          values = columns.polsbyPopper;
          break;
        case 'schwartzberg':
          values = columns.schwartzberg;
          break;
        case 'elongation':
          values = columns.elongation;
          break;
        case 'convexity':
          values = columns.convexity;
          break;
        case 'sliver':
          values = columns.sliverF;
          mapping = 'flag';
          secondary = dark ? [70, 80, 96, 150] : [196, 204, 214, 150];
          break;
        case 'validity':
          values = columns.validityF;
          mapping = 'category';
          break;
        case 'mapColor':
          values = columns.colorF;
          rows = active.fill.partRows;
          mapping = 'category';
          break;
      }
      layers.push(
        new FeatureTriangleLayer({
          id: 'polygon-fill',
          coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
          corners: active.fill.corners,
          featureRows: rows,
          instanceCount: active.fill.triangleCount,
          values,
          valueMapping: mapping,
          colormap: options.ramp,
          valueRange: range,
          sqrtScale,
          color:
            metric === 'sliver'
              ? [226, 96, 80, Math.round(options.opacity * 255)]
              : [255, 255, 255, Math.round(options.opacity * 255)],
          secondaryColor: secondary,
          updateTriggers: {values: [metric, system]}
        })
      );
      const lineColor: [number, number, number, number] = dark
        ? [235, 240, 248, 150]
        : [40, 52, 70, 150];
      if (options.showOutlines) {
        layers.push(
          new PairSegmentLayer({
            id: 'polygon-outline',
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            starts: active.edges.starts,
            ends: active.edges.ends,
            instanceCount: active.edges.edgeCount,
            color: lineColor,
            widthPixels: active.id === 'us-counties' ? 0.8 : 1.1
          })
        );
      }
      if (options.showBounds) {
        layers.push(
          new PairSegmentLayer({
            id: 'polygon-bounds',
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            starts: active.overlays.boundsStarts,
            ends: active.overlays.boundsEnds,
            instanceCount: active.layout.featureCount * 4,
            color: [78, 168, 222, 180],
            widthPixels: 1
          })
        );
      }
      if (options.showInscribed) {
        layers.push(
          new PairSegmentLayer({
            id: 'inscribed-circles',
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            starts: active.overlays.circleStarts,
            ends: active.overlays.circleEnds,
            instanceCount: active.layout.featureCount * CIRCLE_SEGMENTS,
            color: dark ? [255, 214, 120, 230] : [180, 90, 10, 230],
            widthPixels: 1.4
          })
        );
      }
      if (options.showAxes) {
        layers.push(
          new PairSegmentLayer({
            id: 'axis-glyphs',
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            starts: active.overlays.axisStarts,
            ends: active.overlays.axisEnds,
            instanceCount: active.layout.featureCount,
            color: dark ? [150, 230, 255, 235] : [0, 90, 140, 235],
            widthPixels: 2
          })
        );
      }
      if (active.selectedCount > 0) {
        layers.push(
          new PairSegmentLayer({
            id: 'selected-outline',
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            starts: active.overlays.selectedStarts,
            ends: active.overlays.selectedEnds,
            instanceCount: active.selectedCount,
            color: dark ? [255, 255, 255, 255] : [0, 0, 0, 255],
            widthPixels: 3
          })
        );
      }
      if (options.showCentroids) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'centroids',
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            positions: active.overlays.centroids,
            instanceCount: active.layout.featureCount,
            radiusPixels: active.id === 'us-counties' ? 1.6 : 3,
            color: [232, 90, 140, 255]
          })
        );
      }
      if (options.showLabels) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'label-points',
            coordinateSystem: COORDINATE_SYSTEM.LNGLAT,
            positions: active.overlays.labelPoints,
            instanceCount: active.layout.featureCount,
            radiusPixels: active.id === 'us-counties' ? 1.6 : 3,
            color: dark ? [255, 214, 120, 255] : [180, 90, 10, 255]
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      if (state) {
        state.tableReader?.stop();
        state.coloringReader?.stop();
        state.resources.destroy();
        state = null;
      }
    }
  };
}
