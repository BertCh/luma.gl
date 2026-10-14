// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {GPUPolygonOverlayOutput} from '../../../src/gpu-spatial-analysis';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';

/** Graph-imported buffers that a spec destroys together. */
export type BufferGraph = {
  buffers: Buffer[];
  input<Format extends 'float32' | 'float32x2' | 'uint32'>(
    name: string,
    values: Float32Array | Uint32Array,
    format: Format,
    length: number
  ): GraphDataView<Format>;
  output<Format extends 'float32x2' | 'float32x4' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ): {buffer: Buffer; view: GraphDataView<Format>};
};

export function createBufferGraph(device: Device, graph: GPUCommandGraph): BufferGraph {
  const buffers: Buffer[] = [];
  return {
    buffers,
    input(name, values, format, length) {
      const buffer = createInputBuffer(device, values);
      buffers.push(buffer);
      return importGraphBuffer(graph, name, buffer, format, length);
    },
    output(name, format, length) {
      const components = format === 'float32x4' ? 4 : format === 'float32x2' ? 2 : 1;
      const buffer = createOutputBuffer(device, length * components);
      buffers.push(buffer);
      return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
    }
  };
}

export type BufferResult = {
  output: GPUPolygonOverlayOutput;
  positions: Buffer;
  featureOffsets: Buffer;
  polygonOffsets: Buffer;
  ringOffsets: Buffer;
  count: Buffer;
  requiredCount: Buffer;
  overflow: Buffer;
};

export function createBufferResult(
  graphBuffers: BufferGraph,
  prefix: string,
  vertexCapacity = 512,
  ringCapacity = 32,
  boundaryCapacity = 512
): BufferResult {
  const positions = graphBuffers.output(`${prefix}-positions`, 'float32x2', vertexCapacity);
  const featureOffsets = graphBuffers.output(`${prefix}-feature`, 'uint32', ringCapacity + 1);
  const polygonOffsets = graphBuffers.output(`${prefix}-polygon`, 'uint32', ringCapacity + 1);
  const ringOffsets = graphBuffers.output(`${prefix}-ring`, 'uint32', ringCapacity + 1);
  const sourceIds = graphBuffers.output(`${prefix}-source-ids`, 'uint32', ringCapacity);
  const scalar = (name: string) => graphBuffers.output(`${prefix}-${name}`, 'uint32', 1);
  const count = scalar('count');
  const requiredCount = scalar('required-count');
  const overflow = scalar('overflow');
  const candidateOverflow = scalar('candidate-overflow');
  const output: GPUPolygonOverlayOutput = {
    geometry: {
      kind: 'polygons',
      positions: positions.view,
      featureOffsets: featureOffsets.view,
      polygonOffsets: polygonOffsets.view,
      ringOffsets: ringOffsets.view,
      sourceIds: sourceIds.view
    },
    sourceIds: sourceIds.view,
    status: {
      count: count.view,
      requiredCount: requiredCount.view,
      overflow: overflow.view,
      candidateOverflow: candidateOverflow.view
    },
    boundary: {
      endpoints: graphBuffers.output(`${prefix}-boundary`, 'float32x4', boundaryCapacity).view,
      operandIds: graphBuffers.output(`${prefix}-boundary-operands`, 'uint32', boundaryCapacity)
        .view,
      sourceFeatureIds: graphBuffers.output(
        `${prefix}-boundary-features`,
        'uint32',
        boundaryCapacity
      ).view,
      status: {
        count: scalar('boundary-count').view,
        requiredCount: scalar('boundary-required-count').view,
        overflow: scalar('boundary-overflow').view,
        candidateOverflow: scalar('boundary-candidate-overflow').view
      }
    }
  };
  return {
    output,
    positions: positions.buffer,
    featureOffsets: featureOffsets.buffer,
    polygonOffsets: polygonOffsets.buffer,
    ringOffsets: ringOffsets.buffer,
    count: count.buffer,
    requiredCount: requiredCount.buffer,
    overflow: overflow.buffer
  };
}

/** Rings of the result as flat coordinate arrays, grouped by polygon. */
export async function readResultPolygons(result: BufferResult): Promise<number[][][]> {
  const [featureCount] = await readUint32(result.count, 1);
  if (featureCount === 0) {
    return [];
  }
  const features = await readUint32(result.featureOffsets, featureCount + 1);
  const polygonCount = features[featureCount];
  const polygons = await readUint32(result.polygonOffsets, polygonCount + 1);
  const ringCount = polygons[polygonCount];
  const rings = await readUint32(result.ringOffsets, ringCount + 1);
  const coordinates = await readFloat32(result.positions, rings[ringCount] * 2);
  const output: number[][][] = [];
  for (let polygon = 0; polygon < polygonCount; polygon++) {
    const polygonRings: number[][] = [];
    for (let ring = polygons[polygon]; ring < polygons[polygon + 1]; ring++) {
      polygonRings.push(coordinates.slice(rings[ring] * 2, rings[ring + 1] * 2));
    }
    output.push(polygonRings);
  }
  return output;
}

function getRingArea(ring: readonly number[]): number {
  let doubled = 0;
  const count = ring.length / 2;
  for (let row = 0; row < count; row++) {
    const next = (row + 1) % count;
    doubled += ring[row * 2] * ring[next * 2 + 1] - ring[next * 2] * ring[row * 2 + 1];
  }
  return doubled / 2;
}

/** Area of shells minus holes (the first ring of a polygon is its shell). */
export function getPolygonsArea(polygons: readonly (readonly (readonly number[])[])[]): number {
  let area = 0;
  for (const rings of polygons) {
    rings.forEach((ring, index) => {
      area += (index === 0 ? 1 : -1) * Math.abs(getRingArea(ring));
    });
  }
  return area;
}

function isInsideRing(x: number, y: number, ring: readonly number[]): boolean {
  let inside = false;
  const count = ring.length / 2;
  for (let row = 0, previous = count - 1; row < count; previous = row++) {
    const ax = ring[previous * 2];
    const ay = ring[previous * 2 + 1];
    const bx = ring[row * 2];
    const by = ring[row * 2 + 1];
    if (ay > y !== by > y && x < ((bx - ax) * (y - ay)) / (by - ay) + ax) {
      inside = !inside;
    }
  }
  return inside;
}

/** True when the point is inside a shell and outside that polygon's holes. */
export function isInsidePolygons(
  x: number,
  y: number,
  polygons: readonly (readonly (readonly number[])[])[]
): boolean {
  return polygons.some(
    rings => isInsideRing(x, y, rings[0]) && !rings.slice(1).some(hole => isInsideRing(x, y, hole))
  );
}
