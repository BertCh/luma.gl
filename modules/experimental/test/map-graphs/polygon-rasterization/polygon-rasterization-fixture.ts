// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {GPUMapGraphParameterBuffer, importGraphBuffer} from '../../../src/map-graphs';
import {
  getGPUPolygonRasterizationExtentValues,
  GPUPolygonRasterization
} from '../../../src/map-graphs/polygon-rasterization';
import {createInputBuffer, createOutputBuffer, readUint32} from '../map-graph-test-utils';
import type {OraclePolygons} from './polygon-rasterization-oracle';

/** Rasterization graph plus the buffers a test reads or rewrites. */
export type RasterizationFixture = {
  graph: GPUCommandGraph;
  extent: GPUMapGraphParameterBuffer<'float32'>;
  extentView: GraphDataView<'float32'>;
  zonesBuffer: Buffer;
  boundaryBuffer: Buffer;
  overflowBuffer: Buffer;
  crossingCountBuffer: Buffer;
  zonesView: GraphDataView<'uint32'>;
  boundaryView: GraphDataView<'uint32'>;
  polygonBuffers: Buffer[];
  owned: Buffer[];
  width: number;
  height: number;
  recipe: GPUPolygonRasterization;
};

/** Adds a `GPUPolygonRasterization` with every output to a new graph (not compiled). */
export function createRasterizationFixture(
  device: Device,
  props: {
    polygons: OraclePolygons;
    width: number;
    height: number;
    extent: readonly [number, number, number, number];
    crossingCapacity: number;
  }
): RasterizationFixture {
  const {polygons, width, height} = props;
  const graph = new GPUCommandGraph(device, {id: 'polygon-rasterization-test'});
  const extent = new GPUMapGraphParameterBuffer(device, {
    id: 'extent',
    format: 'float32',
    length: 4,
    values: getGPUPolygonRasterizationExtentValues(...props.extent)
  });
  const polygonBuffers = [
    createInputBuffer(device, polygons.positions),
    createInputBuffer(device, polygons.featureOffsets),
    createInputBuffer(device, polygons.polygonOffsets),
    createInputBuffer(device, polygons.ringOffsets)
  ];
  const cellCount = width * height;
  const zonesBuffer = createOutputBuffer(device, cellCount);
  const boundaryBuffer = createOutputBuffer(device, cellCount);
  const overflowBuffer = createOutputBuffer(device, 1);
  const crossingCountBuffer = createOutputBuffer(device, 1);
  const extentView = extent.importToGraph(graph);
  const zonesView = importGraphBuffer(graph, 'zones', zonesBuffer, 'uint32', cellCount);
  const boundaryView = importGraphBuffer(graph, 'boundary', boundaryBuffer, 'uint32', cellCount);
  const recipe = new GPUPolygonRasterization({
    width,
    height,
    extent: extentView,
    polygonPositions: importGraphBuffer(
      graph,
      'positions',
      polygonBuffers[0],
      'float32x2',
      polygons.positions.length / 2
    ),
    featureOffsets: importGraphBuffer(
      graph,
      'feature-offsets',
      polygonBuffers[1],
      'uint32',
      polygons.featureOffsets.length
    ),
    polygonOffsets: importGraphBuffer(
      graph,
      'polygon-offsets',
      polygonBuffers[2],
      'uint32',
      polygons.polygonOffsets.length
    ),
    ringOffsets: importGraphBuffer(
      graph,
      'ring-offsets',
      polygonBuffers[3],
      'uint32',
      polygons.ringOffsets.length
    ),
    crossingCapacity: props.crossingCapacity,
    zones: zonesView,
    boundary: boundaryView,
    overflow: importGraphBuffer(graph, 'overflow', overflowBuffer, 'uint32', 1),
    crossingCount: importGraphBuffer(graph, 'crossing-count', crossingCountBuffer, 'uint32', 1)
  });
  graph.add(recipe);
  return {
    graph,
    extent,
    extentView,
    zonesBuffer,
    boundaryBuffer,
    overflowBuffer,
    crossingCountBuffer,
    zonesView,
    boundaryView,
    polygonBuffers,
    owned: [...polygonBuffers, zonesBuffer, boundaryBuffer, overflowBuffer, crossingCountBuffer],
    width,
    height,
    recipe
  };
}

/** Reads zones, boundary flags, overflow, and the crossing count. */
export async function readRasterization(fixture: RasterizationFixture): Promise<{
  zones: number[];
  boundary: number[];
  overflow: number;
  crossingCount: number;
}> {
  const cellCount = fixture.width * fixture.height;
  return {
    zones: await readUint32(fixture.zonesBuffer, cellCount),
    boundary: await readUint32(fixture.boundaryBuffer, cellCount),
    overflow: (await readUint32(fixture.overflowBuffer, 1))[0],
    crossingCount: (await readUint32(fixture.crossingCountBuffer, 1))[0]
  };
}

/** Destroys every buffer the fixture created. */
export function destroyRasterizationFixture(fixture: RasterizationFixture): void {
  for (const buffer of fixture.owned) {
    buffer.destroy();
  }
  fixture.extent.destroy();
}
