// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPU_POINT_IN_POLYGON_CLASSIFICATION,
  GPUPairwisePointInPolygon
} from '../../../src/geospatial';
import {importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {
  GPU_SPATIAL_JOIN_NO_FEATURE,
  GPUPointInPolygonJoin
} from '../../../src/map-graphs/spatial-join';
import {createInputBuffer, createOutputBuffer, readUint32} from '../map-graph-test-utils';
import {UNCERTAINTY_FEATURES, UNCERTAINTY_POINTS} from './uncertainty-scene';
import {
  buildPolygonFeatureArrays,
  classifyPointInFeature,
  joinPointsInPolygons,
  type OraclePolygonFeature
} from './spatial-join-oracle';

type Point = [number, number];

function toFloat32Points(points: Point[]): Float32Array {
  return Float32Array.from(points.flat());
}

it('GPUPairwisePointInPolygon proves containment despite near-collinear far edges', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const rows: {point: Point; feature: OraclePolygonFeature}[] = [];
  for (const point of UNCERTAINTY_POINTS) {
    for (const feature of UNCERTAINTY_FEATURES) {
      rows.push({point, feature});
    }
  }
  const positions: number[] = [];
  const geometryOffsets = [0];
  const polygonOffsets = [0];
  const ringOffsets = [0];
  let polygonCount = 0;
  let ringCount = 0;
  for (const {feature} of rows) {
    for (const polygon of feature) {
      for (const ring of polygon) {
        for (const vertex of ring) {
          positions.push(...vertex);
        }
        ringCount++;
        ringOffsets.push(positions.length / 2);
      }
      polygonCount++;
      polygonOffsets.push(ringCount);
    }
    geometryOffsets.push(polygonCount);
  }

  const graph = new GPUCommandGraph(device, {id: 'pip-uncertainty'});
  const buffers: Buffer[] = [];
  const input = (data: Float32Array | Uint32Array): Buffer => {
    const buffer = createInputBuffer(device, data);
    buffers.push(buffer);
    return buffer;
  };
  const output = createOutputBuffer(device, rows.length);
  buffers.push(output);
  new GPUPairwisePointInPolygon({
    points: importGraphBuffer(
      graph,
      'points',
      input(toFloat32Points(rows.map(row => row.point))),
      'float32x2',
      rows.length
    ),
    polygonPositions: importGraphBuffer(
      graph,
      'positions',
      input(Float32Array.from(positions)),
      'float32x2',
      positions.length / 2
    ),
    geometryOffsets: importGraphBuffer(
      graph,
      'geometry-offsets',
      input(Uint32Array.from(geometryOffsets)),
      'uint32',
      geometryOffsets.length
    ),
    polygonOffsets: importGraphBuffer(
      graph,
      'polygon-offsets',
      input(Uint32Array.from(polygonOffsets)),
      'uint32',
      polygonOffsets.length
    ),
    ringOffsets: importGraphBuffer(
      graph,
      'ring-offsets',
      input(Uint32Array.from(ringOffsets)),
      'uint32',
      ringOffsets.length
    ),
    output: importGraphBuffer(graph, 'output', output, 'uint32', rows.length)
  }).addToGraph(graph);
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const classifications = await readUint32(output, rows.length);
  const {inside, outside} = GPU_POINT_IN_POLYGON_CLASSIFICATION;
  for (const [index, {point, feature}] of rows.entries()) {
    const expected = classifyPointInFeature(point, feature);
    expect(expected, `oracle row ${index} is never boundary`).not.toBe('boundary');
    expect(classifications[index], `point ${point} row ${index}`).toBe(
      expected === 'inside' ? inside : outside
    );
  }
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
});

it('GPUPointInPolygonJoin assigns near-collinear points and reports no uncertainty', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const arrays = buildPolygonFeatureArrays(UNCERTAINTY_FEATURES);
  const pointCount = UNCERTAINTY_POINTS.length;
  const featureCount = UNCERTAINTY_FEATURES.length;
  const graph = new GPUCommandGraph(device, {id: 'pip-join-uncertainty'});
  const buffers = {
    points: createInputBuffer(device, toFloat32Points(UNCERTAINTY_POINTS)),
    polygonPositions: createInputBuffer(device, arrays.polygonPositions),
    featureOffsets: createInputBuffer(device, arrays.featureOffsets),
    polygonOffsets: createInputBuffer(device, arrays.polygonOffsets),
    ringOffsets: createInputBuffer(device, arrays.ringOffsets),
    pointFeatureIds: createOutputBuffer(device, pointCount),
    featureCounts: createOutputBuffer(device, featureCount),
    overflow: createOutputBuffer(device, 1),
    uncertainCount: createOutputBuffer(device, 1)
  };
  graph.add(
    new GPUPointInPolygonJoin({
      points: importGraphBuffer(graph, 'points', buffers.points, 'float32x2', pointCount),
      polygonPositions: importGraphBuffer(
        graph,
        'polygon-positions',
        buffers.polygonPositions,
        'float32x2',
        arrays.polygonPositions.length / 2
      ),
      featureOffsets: importGraphBuffer(
        graph,
        'feature-offsets',
        buffers.featureOffsets,
        'uint32',
        arrays.featureOffsets.length
      ),
      polygonOffsets: importGraphBuffer(
        graph,
        'polygon-offsets',
        buffers.polygonOffsets,
        'uint32',
        arrays.polygonOffsets.length
      ),
      ringOffsets: importGraphBuffer(
        graph,
        'ring-offsets',
        buffers.ringOffsets,
        'uint32',
        arrays.ringOffsets.length
      ),
      candidateCapacity: pointCount * featureCount,
      pointFeatureIds: importGraphBuffer(
        graph,
        'point-feature-ids',
        buffers.pointFeatureIds,
        'uint32',
        pointCount
      ),
      featureCounts: importGraphBuffer(
        graph,
        'feature-counts',
        buffers.featureCounts,
        'uint32',
        featureCount
      ),
      overflow: importGraphBuffer(graph, 'overflow', buffers.overflow, 'uint32', 1),
      uncertainCount: importGraphBuffer(
        graph,
        'uncertain-count',
        buffers.uncertainCount,
        'uint32',
        1
      )
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const oracle = joinPointsInPolygons(UNCERTAINTY_POINTS, UNCERTAINTY_FEATURES, false);
  // Sanity: the scene exercises assignment, not just rejection.
  expect(oracle.featureRows.filter(row => row !== GPU_SPATIAL_JOIN_NO_FEATURE).length).toBe(11);
  expect(await readUint32(buffers.pointFeatureIds, pointCount)).toEqual(oracle.featureRows);
  expect(await readUint32(buffers.featureCounts, featureCount)).toEqual(oracle.counts);
  expect(await readUint32(buffers.uncertainCount, 1)).toEqual([0]);
  expect(await readUint32(buffers.overflow, 1)).toEqual([0]);
  compiled.destroy();
  for (const buffer of Object.values(buffers)) {
    buffer.destroy();
  }
});
