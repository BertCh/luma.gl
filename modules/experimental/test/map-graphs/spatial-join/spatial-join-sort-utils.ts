// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUMapGraphParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {GPUNearestFeatureJoin, GPUPointInPolygonJoin} from '../../../src/map-graphs/spatial-join';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../map-graph-test-utils';
import {buildPolygonFeatureArrays, type OraclePolygonFeature} from './spatial-join-oracle';

type Point = [number, number];

/** Results read back from one spatial join encoding. */
export type SpatialJoinRunResult = {
  featureRows: number[];
  distances?: number[];
  counts: number[];
  candidateCount: number;
  overflow: number;
};

/** A compiled join that can be encoded repeatedly, read back, and destroyed. */
export type SpatialJoinRun = {
  /** Submits one encoding without waiting. */
  encode: () => void;
  /** Submits one encoding and waits for completion through a one-word readback. */
  encodeAndWait: () => Promise<void>;
  /** Encodes once and reads every output. */
  readResult: () => Promise<SpatialJoinRunResult>;
  destroy: () => void;
};

/** Options shared by both join runners. */
export type SpatialJoinRunOptions = {
  spatialSort: boolean;
  candidateCapacity: number;
  leafCapacity?: number;
};

/** Compiles a point-in-polygon join over `features` and `points`. */
export function createPolygonJoinRun(
  device: Device,
  features: OraclePolygonFeature[],
  points: Point[],
  options: SpatialJoinRunOptions
): SpatialJoinRun {
  const graph = new GPUCommandGraph(device, {id: 'pip-sort-join'});
  const arrays = buildPolygonFeatureArrays(features);
  const featureCount = features.length;
  const pointCount = points.length;
  const buffers: Buffer[] = [
    createInputBuffer(device, Float32Array.from(points.flat())),
    createInputBuffer(device, arrays.polygonPositions),
    createInputBuffer(device, arrays.featureOffsets),
    createInputBuffer(device, arrays.polygonOffsets),
    createInputBuffer(device, arrays.ringOffsets)
  ];
  const [pointsBuffer, positionsBuffer, featureOffsetsBuffer, polygonOffsetsBuffer, ringBuffer] =
    buffers;
  const ids = createOutputBuffer(device, pointCount);
  const counts = createOutputBuffer(device, featureCount);
  const overflow = createOutputBuffer(device, 1);
  const candidateCount = createOutputBuffer(device, 1);
  buffers.push(ids, counts, overflow, candidateCount);
  graph.add(
    new GPUPointInPolygonJoin({
      id: 'pip-sort-join',
      points: importGraphBuffer(graph, 'points', pointsBuffer, 'float32x2', pointCount),
      polygonPositions: importGraphBuffer(
        graph,
        'polygon-positions',
        positionsBuffer,
        'float32x2',
        arrays.polygonPositions.length / 2
      ),
      featureOffsets: importGraphBuffer(
        graph,
        'feature-offsets',
        featureOffsetsBuffer,
        'uint32',
        arrays.featureOffsets.length
      ),
      polygonOffsets: importGraphBuffer(
        graph,
        'polygon-offsets',
        polygonOffsetsBuffer,
        'uint32',
        arrays.polygonOffsets.length
      ),
      ringOffsets: importGraphBuffer(
        graph,
        'ring-offsets',
        ringBuffer,
        'uint32',
        arrays.ringOffsets.length
      ),
      candidateCapacity: options.candidateCapacity,
      leafCapacity: options.leafCapacity,
      spatialSort: options.spatialSort,
      pointFeatureIds: importGraphBuffer(graph, 'ids', ids, 'uint32', pointCount),
      featureCounts: importGraphBuffer(graph, 'counts', counts, 'uint32', featureCount),
      overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1),
      candidateCount: importGraphBuffer(graph, 'candidate-count', candidateCount, 'uint32', 1)
    })
  );
  const compiled = graph.compile();
  return createRun(
    device,
    compiled,
    buffers,
    [],
    async () => ({
      featureRows: await readUint32(ids, pointCount),
      counts: await readUint32(counts, featureCount),
      candidateCount: (await readUint32(candidateCount, 1))[0],
      overflow: (await readUint32(overflow, 1))[0]
    }),
    overflow
  );
}

/** Compiles a nearest-feature join over segment (or point, when `ends` is omitted) features. */
export function createNearestJoinRun(
  device: Device,
  features: {starts: Point[]; ends?: Point[]},
  points: Point[],
  radiusValue: number,
  options: SpatialJoinRunOptions
): SpatialJoinRun {
  const graph = new GPUCommandGraph(device, {id: 'nearest-sort-join'});
  const featureCount = features.starts.length;
  const pointCount = points.length;
  const pointsBuffer = createInputBuffer(device, Float32Array.from(points.flat()));
  const startsBuffer = createInputBuffer(device, Float32Array.from(features.starts.flat()));
  const endsBuffer = features.ends
    ? createInputBuffer(device, Float32Array.from(features.ends.flat()))
    : undefined;
  const ids = createOutputBuffer(device, pointCount);
  const distances = createOutputBuffer(device, pointCount);
  const counts = createOutputBuffer(device, featureCount);
  const overflow = createOutputBuffer(device, 1);
  const candidateCount = createOutputBuffer(device, 1);
  const radius = new GPUMapGraphParameterBuffer(device, {
    id: 'radius',
    format: 'float32',
    length: 1,
    values: Float32Array.of(radiusValue)
  });
  const buffers = [pointsBuffer, startsBuffer, ids, distances, counts, overflow, candidateCount];
  if (endsBuffer) {
    buffers.push(endsBuffer);
  }
  graph.add(
    new GPUNearestFeatureJoin({
      id: 'nearest-sort-join',
      points: importGraphBuffer(graph, 'points', pointsBuffer, 'float32x2', pointCount),
      features: endsBuffer
        ? {
            kind: 'segments',
            starts: importGraphBuffer(graph, 'starts', startsBuffer, 'float32x2', featureCount),
            ends: importGraphBuffer(graph, 'ends', endsBuffer, 'float32x2', featureCount)
          }
        : {
            kind: 'points',
            positions: importGraphBuffer(graph, 'starts', startsBuffer, 'float32x2', featureCount)
          },
      radius: radius.importToGraph(graph),
      candidateCapacity: options.candidateCapacity,
      leafCapacity: options.leafCapacity,
      spatialSort: options.spatialSort,
      nearestFeatureIds: importGraphBuffer(graph, 'ids', ids, 'uint32', pointCount),
      nearestDistances: importGraphBuffer(graph, 'distances', distances, 'float32', pointCount),
      featureCounts: importGraphBuffer(graph, 'counts', counts, 'uint32', featureCount),
      overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1),
      candidateCount: importGraphBuffer(graph, 'candidate-count', candidateCount, 'uint32', 1)
    })
  );
  const compiled = graph.compile();
  return createRun(
    device,
    compiled,
    buffers,
    [radius],
    async () => ({
      featureRows: await readUint32(ids, pointCount),
      distances: await readFloat32(distances, pointCount),
      counts: await readUint32(counts, featureCount),
      candidateCount: (await readUint32(candidateCount, 1))[0],
      overflow: (await readUint32(overflow, 1))[0]
    }),
    overflow
  );
}

function createRun(
  device: Device,
  compiled: CompiledGPUCommandGraph<void>,
  buffers: Buffer[],
  owned: {destroy: () => void}[],
  read: () => Promise<SpatialJoinRunResult>,
  syncBuffer: Buffer
): SpatialJoinRun {
  const encode = () => {
    submitGraph(device, compiled, undefined);
  };
  return {
    encode,
    encodeAndWait: async () => {
      encode();
      await readUint32(syncBuffer, 1);
    },
    readResult: async () => {
      encode();
      return read();
    },
    destroy: () => {
      compiled.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
      for (const resource of owned) {
        resource.destroy();
      }
    }
  };
}

/** Returns a copy of `values` in a deterministic random order, plus the permutation used. */
export function shuffleDeterministic<Value>(
  values: readonly Value[],
  random: () => number
): {shuffled: Value[]; order: number[]} {
  const order = values.map((_, index) => index);
  for (let index = order.length - 1; index > 0; index--) {
    const swap = Math.floor(random() * (index + 1));
    [order[index], order[swap]] = [order[swap], order[index]];
  }
  return {shuffled: order.map(row => values[row]), order};
}

/** Axis-aligned square polygon feature. */
export function createSquareFeature(
  minX: number,
  minY: number,
  size: number
): OraclePolygonFeature {
  return [
    [
      [
        [minX, minY],
        [minX + size, minY],
        [minX + size, minY + size],
        [minX, minY + size]
      ]
    ]
  ];
}
