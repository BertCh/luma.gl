// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import earcut from 'earcut';
import type {Buffer, CommandEncoder} from '@luma.gl/core';
import type {Layer} from '@deck.gl/core';
import {
  DrawCommandBuffer,
  type CompiledGPUCommandGraph,
  type GPUCommandGraph
} from '@luma.gl/gpgpu/gpu-core';
import type {LoadedDataset} from '../../data/catalog';
import {importGraphBuffer} from '../../engine/graph-buffers';
import type {SpatialAnalysisResources} from '../../engine/resources';
import type {SceneFrame, ScenePointerEvent} from '../scene';

/**
 * Helpers shared by the scenes of the Geometry chapter: GeoArrow polygon layouts from the bundled
 * datasets, polygon triangulation and edge lists for drawing, Web Mercator projection, and the
 * path-output buffer plumbing of the line contributors.
 */

// ---------------------------------------------------------------------------------------------
// Polygon layout
// ---------------------------------------------------------------------------------------------

/** A polygon dataset in the GeoArrow layout the contributors take, plus its longitude/latitude. */
export type PolygonLayout = {
  featureCount: number;
  partCount: number;
  ringCount: number;
  vertexCount: number;
  /** Interleaved longitude, latitude degrees, one row per ring vertex. */
  lngLat: Float32Array;
  /** `ringCount + 1` vertex offsets. */
  ringOffsets: Uint32Array;
  /** `partCount + 1` ring offsets (polygon part to rings). */
  polygonOffsets: Uint32Array;
  /** `featureCount + 1` polygon-part offsets (feature to parts). */
  featureOffsets: Uint32Array;
  /** `featureCount + 1` ring offsets (feature to rings), for `featureRingOffsets` inputs. */
  featureRingOffsets: Uint32Array;
  /** Feature row of every ring. */
  ringFeature: Uint32Array;
  /** Per-feature `[west, south, east, north]`, interleaved. */
  featureBounds: Float32Array;
};

/** Reads the binary GeoArrow columns of a polygon dataset. */
export function loadPolygonLayout(dataset: LoadedDataset): PolygonLayout {
  const lngLat = dataset.column<Float32Array>('vertices');
  const ringOffsets = dataset.column<Uint32Array>('ringOffsets');
  const polygonOffsets = dataset.column<Uint32Array>('polygonRingOffsets');
  const partCount = polygonOffsets.length - 1;
  let featureOffsets: Uint32Array;
  if (dataset.hasColumn('countyPolygonOffsets')) {
    featureOffsets = dataset.column<Uint32Array>('countyPolygonOffsets');
  } else {
    const partFeature = dataset.column<Uint32Array>('partFeature');
    const featureCount = partFeature.length ? partFeature[partFeature.length - 1] + 1 : 0;
    featureOffsets = new Uint32Array(featureCount + 1);
    for (let part = 0; part < partFeature.length; part++) featureOffsets[partFeature[part] + 1]++;
    for (let feature = 0; feature < featureCount; feature++) {
      featureOffsets[feature + 1] += featureOffsets[feature];
    }
  }
  const featureCount = featureOffsets.length - 1;
  const ringCount = ringOffsets.length - 1;
  const featureRingOffsets = new Uint32Array(featureCount + 1);
  const ringFeature = new Uint32Array(ringCount);
  const featureBounds = new Float32Array(featureCount * 4);
  for (let feature = 0; feature <= featureCount; feature++) {
    featureRingOffsets[feature] = polygonOffsets[featureOffsets[feature]];
  }
  for (let feature = 0; feature < featureCount; feature++) {
    let west = Infinity;
    let south = Infinity;
    let east = -Infinity;
    let north = -Infinity;
    for (let ring = featureRingOffsets[feature]; ring < featureRingOffsets[feature + 1]; ring++) {
      ringFeature[ring] = feature;
      for (let vertex = ringOffsets[ring]; vertex < ringOffsets[ring + 1]; vertex++) {
        const x = lngLat[vertex * 2];
        const y = lngLat[vertex * 2 + 1];
        west = Math.min(west, x);
        east = Math.max(east, x);
        south = Math.min(south, y);
        north = Math.max(north, y);
      }
    }
    featureBounds.set([west, south, east, north], feature * 4);
  }
  return {
    featureCount,
    partCount,
    ringCount,
    vertexCount: lngLat.length / 2,
    lngLat,
    ringOffsets,
    polygonOffsets,
    featureOffsets,
    featureRingOffsets,
    ringFeature,
    featureBounds
  };
}

/** Web Mercator (EPSG:3857) meters of interleaved longitude/latitude degrees. */
export function projectWebMercator(lngLat: ArrayLike<number>): Float32Array {
  const radius = 6378137;
  const count = Math.floor(lngLat.length / 2);
  const meters = new Float32Array(count * 2);
  for (let index = 0; index < count; index++) {
    const longitude = (lngLat[index * 2] * Math.PI) / 180;
    const latitude = (Math.min(Math.max(lngLat[index * 2 + 1], -85), 85) * Math.PI) / 180;
    meters[index * 2] = radius * longitude;
    meters[index * 2 + 1] = radius * Math.log(Math.tan(Math.PI / 4 + latitude / 2));
  }
  return meters;
}

/** Triangulated polygon fill: corners (longitude/latitude) and the feature row of each triangle. */
export type PolygonFill = {
  corners: Float32Array;
  featureRows: Uint32Array;
  /** Polygon-part row of each triangle (parts are the rows of contiguity weights). */
  partRows: Uint32Array;
  triangleCount: number;
};

/** Triangulates every polygon part (shell plus holes) with earcut. */
export function triangulatePolygons(layout: PolygonLayout): PolygonFill {
  const corners: number[] = [];
  const rows: number[] = [];
  const partRows: number[] = [];
  const {lngLat, ringOffsets, polygonOffsets, featureOffsets} = layout;
  for (let feature = 0; feature < layout.featureCount; feature++) {
    for (let part = featureOffsets[feature]; part < featureOffsets[feature + 1]; part++) {
      const coordinates: number[] = [];
      const holes: number[] = [];
      for (let ring = polygonOffsets[part]; ring < polygonOffsets[part + 1]; ring++) {
        if (ring > polygonOffsets[part]) holes.push(coordinates.length / 2);
        // Drop the repeated closing vertex.
        let end = ringOffsets[ring + 1];
        const start = ringOffsets[ring];
        if (
          end - start > 1 &&
          lngLat[start * 2] === lngLat[(end - 1) * 2] &&
          lngLat[start * 2 + 1] === lngLat[(end - 1) * 2 + 1]
        ) {
          end--;
        }
        for (let vertex = start; vertex < end; vertex++) {
          coordinates.push(lngLat[vertex * 2], lngLat[vertex * 2 + 1]);
        }
      }
      const triangles = earcut(coordinates, holes, 2);
      for (const index of triangles)
        corners.push(coordinates[index * 2], coordinates[index * 2 + 1]);
      for (let triangle = 0; triangle < triangles.length / 3; triangle++) {
        rows.push(feature);
        partRows.push(part);
      }
    }
  }
  return {
    corners: Float32Array.from(corners),
    featureRows: Uint32Array.from(rows),
    partRows: Uint32Array.from(partRows),
    triangleCount: rows.length
  };
}

/** Ring edges as start and end columns (longitude/latitude), with the feature row of each edge. */
export type EdgeColumns = {
  starts: Float32Array;
  ends: Float32Array;
  featureRows: Uint32Array;
  edgeCount: number;
};

/** One edge per consecutive vertex pair of every ring (a repeated closing vertex adds no edge). */
export function buildRingEdges(layout: PolygonLayout): EdgeColumns {
  const starts: number[] = [];
  const ends: number[] = [];
  const rows: number[] = [];
  const {lngLat, ringOffsets} = layout;
  for (let ring = 0; ring < layout.ringCount; ring++) {
    const first = ringOffsets[ring];
    const last = ringOffsets[ring + 1];
    for (let vertex = first; vertex < last; vertex++) {
      const next = vertex + 1 < last ? vertex + 1 : first;
      if (
        lngLat[vertex * 2] === lngLat[next * 2] &&
        lngLat[vertex * 2 + 1] === lngLat[next * 2 + 1]
      )
        continue;
      starts.push(lngLat[vertex * 2], lngLat[vertex * 2 + 1]);
      ends.push(lngLat[next * 2], lngLat[next * 2 + 1]);
      rows.push(layout.ringFeature[ring]);
    }
  }
  return {
    starts: Float32Array.from(starts),
    ends: Float32Array.from(ends),
    featureRows: Uint32Array.from(rows),
    edgeCount: rows.length
  };
}

/** Even-odd point-in-feature test (all rings of one feature) in longitude/latitude. */
export function isPointInFeature(
  layout: PolygonLayout,
  feature: number,
  longitude: number,
  latitude: number
): boolean {
  const {lngLat, ringOffsets, featureRingOffsets} = layout;
  let inside = false;
  for (let ring = featureRingOffsets[feature]; ring < featureRingOffsets[feature + 1]; ring++) {
    const first = ringOffsets[ring];
    const last = ringOffsets[ring + 1];
    for (let vertex = first, previous = last - 1; vertex < last; previous = vertex++) {
      const xi = lngLat[vertex * 2];
      const yi = lngLat[vertex * 2 + 1];
      const xj = lngLat[previous * 2];
      const yj = lngLat[previous * 2 + 1];
      if (
        yi > latitude !== yj > latitude &&
        longitude < ((xj - xi) * (latitude - yi)) / (yj - yi) + xi
      ) {
        inside = !inside;
      }
    }
  }
  return inside;
}

/** Index of the feature containing a point (longitude/latitude), or -1. For tooltips and clicks. */
export function findFeatureAt(layout: PolygonLayout, longitude: number, latitude: number): number {
  const {lngLat, ringOffsets, featureRingOffsets, featureBounds} = layout;
  for (let feature = 0; feature < layout.featureCount; feature++) {
    const b = feature * 4;
    if (
      longitude < featureBounds[b] ||
      longitude > featureBounds[b + 2] ||
      latitude < featureBounds[b + 1] ||
      latitude > featureBounds[b + 3]
    ) {
      continue;
    }
    let inside = false;
    for (let ring = featureRingOffsets[feature]; ring < featureRingOffsets[feature + 1]; ring++) {
      const first = ringOffsets[ring];
      const last = ringOffsets[ring + 1];
      for (let vertex = first, previous = last - 1; vertex < last; previous = vertex++) {
        const xi = lngLat[vertex * 2];
        const yi = lngLat[vertex * 2 + 1];
        const xj = lngLat[previous * 2];
        const yj = lngLat[previous * 2 + 1];
        if (
          yi > latitude !== yj > latitude &&
          longitude < ((xj - xi) * (latitude - yi)) / (yj - yi) + xi
        ) {
          inside = !inside;
        }
      }
    }
    if (inside) return feature;
  }
  return -1;
}

// ---------------------------------------------------------------------------------------------
// Graph imports
// ---------------------------------------------------------------------------------------------

/**
 * Returns an importer for one graph that imports each named buffer once and hands out the same
 * view afterwards (a graph rejects two imports with one id).
 */
export function createGraphImporter<Parameters>(graph: GPUCommandGraph<Parameters>) {
  const views = new Map<string, unknown>();
  return <Format extends Parameters2<typeof importGraphBuffer>[3]>(
    name: string,
    buffer: Buffer,
    format: Format,
    length?: number
  ) => {
    let view = views.get(name);
    if (!view) {
      view = importGraphBuffer(graph, name, buffer, format, length);
      views.set(name, view);
    }
    return view as ReturnType<typeof importGraphBuffer<Format, Parameters>>;
  };
}

type Parameters2<T extends (...args: never[]) => unknown> = T extends (...args: infer P) => unknown
  ? P
  : never;

// ---------------------------------------------------------------------------------------------
// Path outputs
// ---------------------------------------------------------------------------------------------

/** A path layout drawn without a contributor: static buffers plus an indirect record. */
export type StaticPaths = {
  positions: Buffer;
  offsets: Buffer;
  vertexCount: Buffer;
  offsetCount: number;
  drawCommands: DrawCommandBuffer;
};

/** Uploads a static flat path layout (vertices plus offsets) in the shape {@link PathOutputLayer} draws. */
export function createStaticPaths(
  resources: SpatialAnalysisResources,
  name: string,
  positions: Float32Array,
  offsets: Uint32Array
): StaticPaths {
  const vertexCount = positions.length / 2;
  return {
    positions: resources.createBuffer(`${name}-positions`, positions),
    offsets: resources.createBuffer(`${name}-offsets`, offsets),
    vertexCount: resources.createBuffer(`${name}-count`, Uint32Array.of(vertexCount)),
    offsetCount: offsets.length,
    drawCommands: resources.track(
      new DrawCommandBuffer(resources.device, {
        id: `${name}-draw`,
        type: 'draw',
        commands: [{vertexCount: 6, instanceCount: Math.max(vertexCount - 1, 0)}]
      })
    )
  };
}

/** A contributor path output: caller-owned buffers plus the indirect record fed from `count`. */
export type PathOutputBuffers = {
  positions: Buffer;
  offsets: Buffer;
  count: Buffer;
  overflow: Buffer;
  totalCount: Buffer;
  pathCount: Buffer;
  measures: Buffer;
  sourcePaths: Buffer;
  vertexCapacity: number;
  pathCapacity: number;
  offsetCount: number;
  drawCommands: DrawCommandBuffer;
};

/** Allocates the buffers of a `GPULinePathOutput` of the given capacities. */
export function createPathOutputBuffers(
  resources: SpatialAnalysisResources,
  name: string,
  vertexCapacity: number,
  pathCapacity: number
): PathOutputBuffers {
  return {
    positions: resources.createBuffer(`${name}-positions`, vertexCapacity * 8),
    offsets: resources.createBuffer(`${name}-offsets`, (pathCapacity + 1) * 4),
    count: resources.createBuffer(`${name}-count`, 4),
    overflow: resources.createBuffer(`${name}-overflow`, 4),
    totalCount: resources.createBuffer(`${name}-total`, 4),
    pathCount: resources.createBuffer(`${name}-path-count`, 4),
    measures: resources.createBuffer(`${name}-measures`, vertexCapacity * 4),
    sourcePaths: resources.createBuffer(`${name}-source-paths`, pathCapacity * 4),
    vertexCapacity,
    pathCapacity,
    offsetCount: pathCapacity + 1,
    drawCommands: resources.track(
      new DrawCommandBuffer(resources.device, {
        id: `${name}-draw`,
        type: 'draw',
        commands: [{vertexCount: 6, instanceCount: 0}]
      })
    )
  };
}

/** Copies the GPU-written vertex count into word 1 (instance count) of an indirect record. */
export function copyCountToDrawRecord(
  commandEncoder: CommandEncoder,
  count: Buffer,
  drawCommands: DrawCommandBuffer
): void {
  commandEncoder.copyBufferToBuffer({
    sourceBuffer: count,
    sourceOffset: 0,
    destinationBuffer: drawCommands.buffer,
    destinationOffset: 4,
    size: 4
  });
}

/** Binds a path output's views into a graph and returns the contributor `output` object. */
export function importPathOutput(
  graph: GPUCommandGraph<void>,
  output: PathOutputBuffers,
  options: {measures?: boolean; pathCount?: boolean; sourcePaths?: boolean} = {}
) {
  const {vertexCapacity, pathCapacity} = output;
  return {
    positions: importGraphBuffer(
      graph,
      'out-positions',
      output.positions,
      'float32x2',
      vertexCapacity
    ),
    pathOffsets: importGraphBuffer(
      graph,
      'out-offsets',
      output.offsets,
      'uint32',
      pathCapacity + 1
    ),
    count: importGraphBuffer(graph, 'out-count', output.count, 'uint32', 1),
    overflow: importGraphBuffer(graph, 'out-overflow', output.overflow, 'uint32', 1),
    totalCount: importGraphBuffer(graph, 'out-total', output.totalCount, 'uint32', 1),
    ...(options.pathCount
      ? {pathCount: importGraphBuffer(graph, 'out-path-count', output.pathCount, 'uint32', 1)}
      : {}),
    ...(options.sourcePaths
      ? {
          sourcePaths: importGraphBuffer(
            graph,
            'out-source-paths',
            output.sourcePaths,
            'uint32',
            pathCapacity
          )
        }
      : {}),
    ...(options.measures
      ? {
          measures: importGraphBuffer(
            graph,
            'out-measures',
            output.measures,
            'float32',
            vertexCapacity
          )
        }
      : {})
  };
}

// ---------------------------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------------------------

/** `12,345 km2` style number with the given fraction digits. */
export function formatNumber(value: number, fractionDigits = 0): string {
  if (!Number.isFinite(value)) return 'n/a';
  return value.toLocaleString('en-US', {
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits
  });
}

/** Meters as `640 m` or `12.3 km`. */
export function formatDistance(meters: number): string {
  if (!Number.isFinite(meters)) return 'n/a';
  if (meters < 1000) return `${Math.round(meters)} m`;
  return `${(meters / 1000).toLocaleString('en-US', {maximumFractionDigits: meters < 100_000 ? 1 : 0})} km`;
}

/** Statistics of the finite values of a column. */
export function summarizeColumn(values: ArrayLike<number>): {
  minimum: number;
  maximum: number;
  mean: number;
  count: number;
} {
  let minimum = Infinity;
  let maximum = -Infinity;
  let sum = 0;
  let count = 0;
  for (let index = 0; index < values.length; index++) {
    const value = values[index];
    if (!Number.isFinite(value)) continue;
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
    sum += value;
    count++;
  }
  return {minimum, maximum, mean: count ? sum / count : NaN, count};
}

/** Value at a quantile of the finite values (for robust legend extents). */
export function getQuantile(values: ArrayLike<number>, quantile: number): number {
  const finite = Array.from(values as ArrayLike<number>).filter(value => Number.isFinite(value));
  if (!finite.length) return NaN;
  finite.sort((a, b) => a - b);
  return finite[
    Math.min(finite.length - 1, Math.max(0, Math.round(quantile * (finite.length - 1))))
  ];
}

// ---------------------------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------------------------

/**
 * One tool of a multi-tool scene: its own compiled graphs and resources, created the first time it
 * is shown and destroyed with the scene.
 */
export type GeometryView<Options> = {
  getCompiledGraphs: () => CompiledGPUCommandGraph<never>[];
  encode: (commandEncoder: CommandEncoder, frame: SceneFrame) => void;
  getLayers: () => Layer[];
  setOption: (id: keyof Options & string, options: Options) => void;
  onClick?: (event: ScenePointerEvent) => boolean;
  onDragStart?: (event: ScenePointerEvent) => boolean;
  onDrag?: (event: ScenePointerEvent) => void;
  onDragEnd?: (event: ScenePointerEvent) => void;
  getTooltip?: (event: ScenePointerEvent) => string | null;
  destroy: () => void;
};

/** Meters covered by one CSS pixel at a latitude and Web Mercator zoom (512 px world tiles). */
export function getMetersPerPixel(zoom: number, latitude: number): number {
  return (40075016.686 * Math.cos((latitude * Math.PI) / 180)) / (512 * 2 ** zoom);
}
