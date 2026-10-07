// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * GPU helpers shared by the "joins" scenes: uploading polygon and line sets, importing them into a
 * graph in the layout the joins read, and building the zone raster that fills polygons on the map.
 */

import type {Buffer} from '@luma.gl/core';
import {
  getGPUPolygonRasterizationExtentValues,
  GPUPolygonRasterization
} from '@luma.gl/experimental/gpu-raster';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {importGraphBuffer, submitGraph} from '../../engine/graph-buffers';
import type {SpatialAnalysisResources} from '../../engine/resources';
import type {LineSet, PolygonSet} from './b2-geometry';

/** A polygon set uploaded once and shared by every graph of a scene. */
export type PolygonBuffers = {
  set: PolygonSet;
  positions: Buffer;
  featureOffsets: Buffer;
  polygonOffsets: Buffer;
  ringOffsets: Buffer;
  outline: Buffer;
  outlineRows: Buffer;
};

/** A line set uploaded once. */
export type LineBuffers = {
  set: LineSet;
  positions: Buffer;
  lineOffsets: Buffer;
  segments: Buffer;
  segmentRows: Buffer;
};

/** Geometry views of a polygon set inside one graph, as `GPUSpatialJoinPolygons` expects. */
export type PolygonViews = {
  kind: 'polygons';
  positions: GraphDataView<'float32x2'>;
  featureOffsets: GraphDataView<'uint32'>;
  polygonOffsets: GraphDataView<'uint32'>;
  ringOffsets: GraphDataView<'uint32'>;
};

/** Geometry views of a line set inside one graph, as `GPUSpatialJoinLines` expects. */
export type LineViews = {
  kind: 'lines';
  positions: GraphDataView<'float32x2'>;
  lineOffsets: GraphDataView<'uint32'>;
};

/** Uploads a polygon set as storage buffers owned by `resources`. */
export function uploadPolygons(
  resources: SpatialAnalysisResources,
  name: string,
  set: PolygonSet
): PolygonBuffers {
  return {
    set,
    positions: resources.createBuffer(`${name}-positions`, set.positions),
    featureOffsets: resources.createBuffer(`${name}-feature-offsets`, set.featureOffsets),
    polygonOffsets: resources.createBuffer(`${name}-polygon-offsets`, set.polygonOffsets),
    ringOffsets: resources.createBuffer(`${name}-ring-offsets`, set.ringOffsets),
    outline: resources.createBuffer(`${name}-outline`, set.outline),
    outlineRows: resources.createBuffer(`${name}-outline-rows`, set.outlineRows)
  };
}

/** Uploads a line set as storage buffers owned by `resources`. */
export function uploadLines(
  resources: SpatialAnalysisResources,
  name: string,
  set: LineSet
): LineBuffers {
  return {
    set,
    positions: resources.createBuffer(`${name}-positions`, set.positions),
    lineOffsets: resources.createBuffer(`${name}-line-offsets`, set.lineOffsets),
    segments: resources.createBuffer(`${name}-segments`, set.segments),
    segmentRows: resources.createBuffer(`${name}-segment-rows`, set.segmentRows)
  };
}

/** Imports uploaded polygons into `graph`. */
export function importPolygons<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  name: string,
  buffers: PolygonBuffers
): PolygonViews {
  const {set} = buffers;
  return {
    kind: 'polygons',
    positions: importGraphBuffer(
      graph,
      `${name}-positions`,
      buffers.positions,
      'float32x2',
      set.positions.length / 2
    ),
    featureOffsets: importGraphBuffer(
      graph,
      `${name}-feature-offsets`,
      buffers.featureOffsets,
      'uint32',
      set.featureOffsets.length
    ),
    polygonOffsets: importGraphBuffer(
      graph,
      `${name}-polygon-offsets`,
      buffers.polygonOffsets,
      'uint32',
      set.polygonOffsets.length
    ),
    ringOffsets: importGraphBuffer(
      graph,
      `${name}-ring-offsets`,
      buffers.ringOffsets,
      'uint32',
      set.ringOffsets.length
    )
  };
}

/** Imports uploaded lines into `graph`. */
export function importLines<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  name: string,
  buffers: LineBuffers
): LineViews {
  const {set} = buffers;
  return {
    kind: 'lines',
    positions: importGraphBuffer(
      graph,
      `${name}-positions`,
      buffers.positions,
      'float32x2',
      set.positions.length / 2
    ),
    lineOffsets: importGraphBuffer(
      graph,
      `${name}-line-offsets`,
      buffers.lineOffsets,
      'uint32',
      set.lineOffsets.length
    )
  };
}

/**
 * Estimates the (edge, raster row) crossings of `GPUPolygonRasterization` for a raster whose row
 * height is `cellHeight` and that starts at `originY` with `height` rows. Used to size
 * `crossingCapacity` exactly instead of guessing.
 */
export function estimateCrossings(
  set: PolygonSet,
  originY: number,
  cellHeight: number,
  height: number
): number {
  let crossings = 0;
  for (let edge = 0; edge < set.outline.length; edge += 4) {
    const y0 = (set.outline[edge + 1] - originY) / cellHeight - 0.5;
    const y1 = (set.outline[edge + 3] - originY) / cellHeight - 0.5;
    const low = Math.max(0, Math.ceil(Math.min(y0, y1)));
    const high = Math.min(height - 1, Math.ceil(Math.max(y0, y1)) - 1);
    if (high >= low) crossings += high - low + 1;
  }
  return crossings;
}

/** A zone raster: the feature row of the polygon under every cell center, or no zone. */
export type ZoneRaster = {
  width: number;
  height: number;
  /** `[minX, minY, maxX, maxY]` meters of the raster's outer cell edges. */
  bounds: [number, number, number, number];
  /** `width * height` uint32 feature rows (`0xffffffff` = no zone). */
  zones: Buffer;
  /** `width * height` uint32 flags: 1 where a polygon edge touches the cell. */
  boundary: Buffer;
  /** Crossings the rasterization needed (the capacity it was given was sufficient when not overflowed). */
  crossingCount: number;
  overflowed: boolean;
};

/**
 * Rasterizes `polygons` once with `GPUPolygonRasterization` into a zone raster and boundary flags,
 * outside the frame. The scenes use the raster to paint polygon fills and to hit-test cells.
 *
 * @param bounds `[minX, minY, maxX, maxY]` meters to cover.
 * @param longSide Cells along the longer side of `bounds`.
 */
export async function createZoneRaster(
  resources: SpatialAnalysisResources,
  name: string,
  polygons: PolygonBuffers,
  bounds: readonly [number, number, number, number],
  longSide: number
): Promise<ZoneRaster> {
  const [minX, minY, maxX, maxY] = bounds;
  const widthMeters = maxX - minX;
  const heightMeters = maxY - minY;
  const cellSize = Math.max(widthMeters, heightMeters) / longSide;
  const width = Math.ceil(widthMeters / cellSize);
  const height = Math.ceil(heightMeters / cellSize);
  const cellCount = width * height;
  const crossingEstimate = estimateCrossings(polygons.set, minY, cellSize, height);
  const crossingCapacity = Math.max(1024, Math.ceil(crossingEstimate * 1.25));

  const zones = resources.createBuffer(`${name}-zones`, cellCount * 4);
  const boundary = resources.createBuffer(`${name}-boundary`, cellCount * 4);
  const overflow = resources.createBuffer(`${name}-overflow`, 4);
  const crossingCount = resources.createBuffer(`${name}-crossings`, 4);
  const extent = resources.createParameterBuffer(
    `${name}-extent`,
    'float32',
    4,
    getGPUPolygonRasterizationExtentValues(minX, minY, cellSize, cellSize)
  );
  const graph = new GPUCommandGraph<undefined>(resources.device, {id: `${name}-rasterize`});
  const views = importPolygons(graph, `${name}-polygons`, polygons);
  graph.add(
    new GPUPolygonRasterization({
      id: `${name}-rasterization`,
      width,
      height,
      extent: extent.importToGraph(graph),
      polygonPositions: views.positions,
      featureOffsets: views.featureOffsets,
      polygonOffsets: views.polygonOffsets,
      ringOffsets: views.ringOffsets,
      crossingCapacity,
      zones: importGraphBuffer(graph, `${name}-zones`, zones, 'uint32', cellCount),
      boundary: importGraphBuffer(graph, `${name}-boundary`, boundary, 'uint32', cellCount),
      overflow: importGraphBuffer(graph, `${name}-overflow`, overflow, 'uint32', 1),
      crossingCount: importGraphBuffer(graph, `${name}-crossings`, crossingCount, 'uint32', 1)
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(resources.device, compiled);
    const flags = new Uint32Array((await overflow.readAsync()).slice().buffer);
    const counts = new Uint32Array((await crossingCount.readAsync()).slice().buffer);
    return {
      width,
      height,
      bounds: [minX, minY, minX + width * cellSize, minY + height * cellSize],
      zones,
      boundary,
      crossingCount: counts[0],
      overflowed: flags[0] !== 0
    };
  } finally {
    compiled.destroy();
  }
}

/** Reads a whole buffer as `uint32` (outside the frame). */
export async function readUint32(buffer: Buffer, byteLength?: number): Promise<Uint32Array> {
  return new Uint32Array((await buffer.readAsync(0, byteLength)).slice().buffer);
}

/** Reads a whole buffer as `float32` (outside the frame). */
export async function readFloat32(buffer: Buffer, byteLength?: number): Promise<Float32Array> {
  return new Float32Array((await buffer.readAsync(0, byteLength)).slice().buffer);
}

/** Packs 0-255 channels into the `rgba8` layout of the classification contributors. */
export function packColor(r: number, g: number, b: number, a = 255): number {
  return ((r & 255) | ((g & 255) << 8) | ((b & 255) << 16) | ((a & 255) << 24)) >>> 0;
}

/** Piecewise-linear interpolation through `[r, g, b]` stops; `t` is clamped to `[0, 1]`. */
export function sampleStops(
  stops: readonly (readonly number[])[],
  t: number
): [number, number, number] {
  const scaled = Math.min(Math.max(t, 0), 1) * (stops.length - 1);
  const index = Math.min(Math.floor(scaled), stops.length - 2);
  const fraction = scaled - index;
  const from = stops[index];
  const to = stops[index + 1];
  return [
    from[0] + (to[0] - from[0]) * fraction,
    from[1] + (to[1] - from[1]) * fraction,
    from[2] + (to[2] - from[2]) * fraction
  ];
}

/** Returns `count` packed colors sampled evenly along `stops`. */
export function getStopsPalette(stops: readonly (readonly number[])[], count: number): Uint32Array {
  const palette = new Uint32Array(count);
  for (let index = 0; index < count; index++) {
    const [r, g, b] = sampleStops(stops, count === 1 ? 0.5 : index / (count - 1));
    palette[index] = packColor(Math.round(r), Math.round(g), Math.round(b));
  }
  return palette;
}
