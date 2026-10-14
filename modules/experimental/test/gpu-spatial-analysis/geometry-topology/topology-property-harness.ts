// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPULineNoding,
  GPUMakeValid
} from '../../../src/gpu-spatial-analysis/geometry-topology/index';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

/** Harness for the property specs of `GPULineNoding` and `GPUMakeValid` (fixtures from shapely). */

export type Point = readonly [number, number];

/** One noded segment as written by `GPULineNoding`. */
export type NodedSegment = {endpoints: [number, number, number, number]; sourceEdge: number};

export type LineNodingResult = {
  count: number;
  requiredCount: number;
  overflow: number;
  candidateOverflow: number;
  segments: NodedSegment[];
};

function flatten(rings: readonly (readonly Point[])[]) {
  const positions: number[] = [];
  const offsets = [0];
  for (const ring of rings) {
    for (const [x, y] of ring) {
      positions.push(x, y);
    }
    offsets.push(positions.length / 2);
  }
  return {positions: Float32Array.from(positions), offsets: Uint32Array.from(offsets)};
}

class GraphBuffers {
  readonly buffers: Buffer[] = [];
  constructor(
    readonly device: Device,
    readonly graph: GPUCommandGraph
  ) {}

  input<Format extends 'float32x2' | 'uint32'>(
    name: string,
    values: Float32Array | Uint32Array,
    format: Format,
    length: number
  ) {
    const buffer = createInputBuffer(this.device, values);
    this.buffers.push(buffer);
    return importGraphBuffer(this.graph, name, buffer, format, length);
  }

  output<Format extends 'uint32' | 'float32' | 'float32x2' | 'float32x4'>(
    name: string,
    format: Format,
    length: number
  ) {
    const components = {uint32: 1, float32: 1, float32x2: 2, float32x4: 4}[format];
    const buffer = createOutputBuffer(this.device, length * components);
    this.buffers.push(buffer);
    return {buffer, view: importGraphBuffer(this.graph, name, buffer, format, length)};
  }

  destroy(): void {
    for (const buffer of this.buffers) {
      buffer.destroy();
    }
  }
}

/** Runs `GPULineNoding` on `lines` and reads the written segments back. */
export async function runLineNoding(
  device: Device,
  lines: readonly (readonly Point[])[],
  segmentCapacity: number,
  intersectionCapacity: number
): Promise<LineNodingResult> {
  const graph = new GPUCommandGraph(device, {id: 'line-noding-property'});
  const io = new GraphBuffers(device, graph);
  const arrays = flatten(lines);
  const endpoints = io.output('endpoints', 'float32x4', segmentCapacity);
  const features = io.output('features', 'uint32', segmentCapacity);
  const rings = io.output('rings', 'uint32', segmentCapacity);
  const edges = io.output('edges', 'uint32', segmentCapacity);
  const starts = io.output('start-parameters', 'float32', segmentCapacity);
  const ends = io.output('end-parameters', 'float32', segmentCapacity);
  const count = io.output('count', 'uint32', 1);
  const required = io.output('required', 'uint32', 1);
  const overflow = io.output('overflow', 'uint32', 1);
  const candidateOverflow = io.output('candidate-overflow', 'uint32', 1);
  graph.add(
    new GPULineNoding({
      lines: {
        kind: 'lines',
        positions: io.input(
          'positions',
          arrays.positions,
          'float32x2',
          arrays.positions.length / 2
        ),
        lineOffsets: io.input('line-offsets', arrays.offsets, 'uint32', arrays.offsets.length)
      },
      intersectionCapacity,
      output: {
        endpoints: endpoints.view,
        sourceFeatureIds: features.view,
        sourceRingIds: rings.view,
        sourceEdgeIds: edges.view,
        sourceStartParameters: starts.view,
        sourceEndParameters: ends.view,
        status: {
          count: count.view,
          requiredCount: required.view,
          overflow: overflow.view,
          candidateOverflow: candidateOverflow.view
        }
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const written = (await readUint32(count.buffer, 1))[0];
  const coordinates = await readFloat32(endpoints.buffer, 4 * segmentCapacity);
  const sourceEdges = await readUint32(edges.buffer, segmentCapacity);
  const result: LineNodingResult = {
    count: written,
    requiredCount: (await readUint32(required.buffer, 1))[0],
    overflow: (await readUint32(overflow.buffer, 1))[0],
    candidateOverflow: (await readUint32(candidateOverflow.buffer, 1))[0],
    segments: Array.from({length: Math.min(written, segmentCapacity)}, (_, row) => ({
      endpoints: [
        coordinates[4 * row],
        coordinates[4 * row + 1],
        coordinates[4 * row + 2],
        coordinates[4 * row + 3]
      ],
      sourceEdge: sourceEdges[row]
    }))
  };
  compiled.destroy();
  io.destroy();
  return result;
}

export type MakeValidResult = {
  /** Polygons as rings of `[x, y]` vertices; the first ring is the shell. */
  polygons: number[][][][];
  count: number;
  requiredCount: number;
  overflow: number;
  candidateOverflow: number;
};

/** Runs `GPUMakeValid` on one polygon feature given as closed rings, and reads the faces back. */
export async function runMakeValid(
  device: Device,
  rings: readonly (readonly Point[])[],
  capacity: {segments: number; intersections: number; rings: number; vertices: number}
): Promise<MakeValidResult> {
  const graph = new GPUCommandGraph(device, {id: 'make-valid-property'});
  const io = new GraphBuffers(device, graph);
  const arrays = flatten(rings);
  const directed = 2 * capacity.segments;
  const nodedEndpoints = io.output('noded-endpoints', 'float32x4', capacity.segments);
  const nodedFeatures = io.output('noded-features', 'uint32', capacity.segments);
  const nodedRings = io.output('noded-rings', 'uint32', capacity.segments);
  const nodedEdges = io.output('noded-edges', 'uint32', capacity.segments);
  const nodedStarts = io.output('noded-starts', 'float32', capacity.segments);
  const nodedEnds = io.output('noded-ends', 'float32', capacity.segments);
  const directedEndpoints = io.output('directed-endpoints', 'float32x4', directed);
  const directedSegments = io.output('directed-segments', 'uint32', directed);
  const directedFeatures = io.output('directed-features', 'uint32', directed);
  const directedTwins = io.output('directed-twins', 'uint32', directed);
  const directedRings = io.output('directed-rings', 'uint32', directed);
  const directedFlags = io.output('directed-flags', 'uint32', directed);
  const positions = io.output('positions', 'float32x2', capacity.vertices);
  const featureOffsets = io.output('feature-offsets', 'uint32', capacity.rings + 1);
  const polygonOffsets = io.output('polygon-offsets', 'uint32', capacity.rings + 1);
  const ringOffsets = io.output('ring-offsets', 'uint32', capacity.rings + 1);
  const sourceIds = io.output('source-ids', 'uint32', capacity.rings);
  const edgeClasses = io.output('edge-classes', 'uint32', capacity.segments);
  const scalar = (name: string) => io.output(name, 'uint32', 1);
  const nodedStatus = {
    count: scalar('noded-count'),
    requiredCount: scalar('noded-required'),
    overflow: scalar('noded-overflow'),
    candidateOverflow: scalar('noded-candidate-overflow')
  };
  const directedStatus = {
    count: scalar('directed-count'),
    requiredCount: scalar('directed-required'),
    overflow: scalar('directed-overflow'),
    candidateOverflow: scalar('directed-candidate-overflow')
  };
  const status = {
    count: scalar('count'),
    requiredCount: scalar('required'),
    overflow: scalar('overflow'),
    candidateOverflow: scalar('candidate-overflow')
  };
  const diagnostics = [
    scalar('closed-rings'),
    scalar('open-chains'),
    scalar('cut-edges'),
    scalar('dangles'),
    scalar('unused-edges')
  ];
  const views = (ports: Record<string, {view: unknown}>) =>
    Object.fromEntries(Object.entries(ports).map(([name, port]) => [name, port.view])) as never;
  graph.add(
    new GPUMakeValid({
      intersectionCapacity: capacity.intersections,
      precision: {vertexTolerance: 1e-5, predicates: 'exact-or-uncertain', storage: 'float32'},
      polygons: {
        kind: 'polygons',
        positions: io.input(
          'input-positions',
          arrays.positions,
          'float32x2',
          arrays.positions.length / 2
        ),
        ringOffsets: io.input(
          'input-ring-offsets',
          arrays.offsets,
          'uint32',
          arrays.offsets.length
        ),
        polygonOffsets: io.input(
          'input-polygon-offsets',
          Uint32Array.of(0, rings.length),
          'uint32',
          2
        ),
        featureOffsets: io.input('input-feature-offsets', Uint32Array.of(0, 1), 'uint32', 2)
      },
      output: {
        nodedSegments: {
          endpoints: nodedEndpoints.view,
          sourceFeatureIds: nodedFeatures.view,
          sourceRingIds: nodedRings.view,
          sourceEdgeIds: nodedEdges.view,
          sourceStartParameters: nodedStarts.view,
          sourceEndParameters: nodedEnds.view,
          status: views(nodedStatus)
        },
        directedEdges: {
          endpoints: directedEndpoints.view,
          sourceSegmentIds: directedSegments.view,
          sourceFeatureIds: directedFeatures.view,
          twinIds: directedTwins.view,
          ringIds: directedRings.view,
          flags: directedFlags.view,
          status: views(directedStatus)
        },
        polygons: {
          kind: 'polygons',
          positions: positions.view,
          featureOffsets: featureOffsets.view,
          polygonOffsets: polygonOffsets.view,
          ringOffsets: ringOffsets.view,
          sourceIds: sourceIds.view
        },
        diagnostics: {
          edgeClasses: edgeClasses.view,
          closedRingCount: diagnostics[0].view,
          openChainCount: diagnostics[1].view,
          cutEdgeCount: diagnostics[2].view,
          dangleCount: diagnostics[3].view,
          unusedEdgeCount: diagnostics[4].view
        },
        status: views(status)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const count = (await readUint32(status.count.buffer, 1))[0];
  const polygonRows = await readUint32(polygonOffsets.buffer, capacity.rings + 1);
  const ringCount = polygonRows[count];
  const ringRows = await readUint32(ringOffsets.buffer, capacity.rings + 1);
  const vertexCount = ringRows[ringCount];
  const coordinates = await readFloat32(positions.buffer, 2 * capacity.vertices);
  const polygons: number[][][][] = [];
  for (let polygon = 0; polygon < count; polygon++) {
    const polygonRings: number[][][] = [];
    for (let ring = polygonRows[polygon]; ring < polygonRows[polygon + 1]; ring++) {
      const vertices: number[][] = [];
      for (let vertex = ringRows[ring]; vertex < ringRows[ring + 1]; vertex++) {
        vertices.push([coordinates[2 * vertex], coordinates[2 * vertex + 1]]);
      }
      polygonRings.push(vertices);
    }
    polygons.push(polygonRings);
  }
  const result: MakeValidResult = {
    polygons,
    count,
    requiredCount: (await readUint32(status.requiredCount.buffer, 1))[0],
    overflow: (await readUint32(status.overflow.buffer, 1))[0],
    candidateOverflow: (await readUint32(status.candidateOverflow.buffer, 1))[0]
  };
  if (vertexCount > capacity.vertices) {
    result.overflow = Math.max(result.overflow, 1);
  }
  compiled.destroy();
  io.destroy();
  return result;
}
