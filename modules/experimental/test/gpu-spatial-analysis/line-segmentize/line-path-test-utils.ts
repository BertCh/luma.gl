// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {expect, vi} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import type {GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import type {GPULinePathOutput} from '../../../src/gpu-spatial-analysis/line-segmentize/line-segmentize-types';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {getHaversineDistance} from '../geometry-measures/geodesic-oracle';
import type {FlatPathResult} from './line-segmentize-oracle';

/** Read-back of a `GPULinePathOutput`. */
export type LinePathReadback = {
  positions: number[][];
  pathOffsets: number[];
  count: number;
  overflow: number;
  requiredCount: number;
  pathCount: number;
  sourcePaths: number[];
  sourceRows: number[];
  measures: number[];
};

/** A compiled single-contributor graph with typed inputs, a parameter buffer and a path output. */
export type LinePathFixture = {
  /** Writes parameters, encodes once, and reads the output. */
  run(parameters: Float32Array): Promise<LinePathReadback>;
  /** Number of `compile()` calls after setup. */
  getCompileCount(): number;
  destroy(): void;
};

/**
 * Builds a fixture around one contributor. `inputs` are imported by name; `createContributor` receives the
 * imported views, the parameter view and the output views.
 */
export function createLinePathFixture(
  device: Device,
  options: {
    inputs: Record<
      string,
      {
        values: Float32Array | Uint32Array;
        format: 'float32x2' | 'uint32' | 'float32';
      }
    >;
    capacity: number;
    /** Whether to request the per-vertex `sourceRows` and `measures` outputs. Default true. */
    withVertexColumns?: boolean;
    pathCapacity: number;
    createContributor: (
      inputs: Record<string, GraphDataView>,
      parameters: GraphDataView<'float32'>,
      output: GPULinePathOutput
    ) => GPUCommandNodeProducer;
  }
): LinePathFixture {
  const buffers: Buffer[] = [];
  const graph = new GPUCommandGraph(device, {id: 'line-path-fixture'});
  const views: Record<string, GraphDataView> = {};
  for (const [name, input] of Object.entries(options.inputs)) {
    const buffer = createInputBuffer(device, input.values);
    buffers.push(buffer);
    const components = input.format === 'float32x2' ? 2 : 1;
    views[name] = importGraphBuffer(
      graph,
      name,
      buffer,
      input.format,
      input.values.length / components
    );
  }
  const {capacity, pathCapacity} = options;
  const output = (name: string, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return buffer;
  };
  const outputs = {
    positions: output('positions', 2 * capacity),
    pathOffsets: output('path-offsets', pathCapacity + 1),
    count: output('count', 1),
    overflow: output('overflow', 1),
    requiredCount: output('total', 1),
    pathCount: output('path-count', 1),
    sourcePaths: output('source-paths', pathCapacity),
    sourceRows: output('source-rows', capacity),
    measures: output('measures', capacity)
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'parameters',
    format: 'float32',
    length: 4
  });
  graph.add(
    options.createContributor(views, parameterBuffer.importToGraph(graph), {
      positions: importGraphBuffer(graph, 'o-positions', outputs.positions, 'float32x2', capacity),
      pathOffsets: importGraphBuffer(
        graph,
        'o-path-offsets',
        outputs.pathOffsets,
        'uint32',
        pathCapacity + 1
      ),
      count: importGraphBuffer(graph, 'o-count', outputs.count, 'uint32', 1),
      overflow: importGraphBuffer(graph, 'o-overflow', outputs.overflow, 'uint32', 1),
      requiredCount: importGraphBuffer(graph, 'o-total', outputs.requiredCount, 'uint32', 1),
      pathCount: importGraphBuffer(graph, 'o-path-count', outputs.pathCount, 'uint32', 1),
      sourcePaths: importGraphBuffer(
        graph,
        'o-source-paths',
        outputs.sourcePaths,
        'uint32',
        pathCapacity
      ),
      ...(options.withVertexColumns === false
        ? {}
        : {
            sourceRows: importGraphBuffer(
              graph,
              'o-source-rows',
              outputs.sourceRows,
              'uint32',
              capacity
            ),
            measures: importGraphBuffer(graph, 'o-measures', outputs.measures, 'float32', capacity)
          })
    })
  );
  const compiled = graph.compile();
  const compileSpy = vi.spyOn(graph, 'compile');
  return {
    async run(parameters) {
      parameterBuffer.write(parameters);
      submitGraph(device, compiled, undefined);
      const [count] = await readUint32(outputs.count, 1);
      const flatPositions = await readFloat32(outputs.positions, 2 * capacity);
      return {
        positions: Array.from({length: count}, (_, row) => [
          flatPositions[2 * row],
          flatPositions[2 * row + 1]
        ]),
        pathOffsets: await readUint32(outputs.pathOffsets, pathCapacity + 1),
        count,
        overflow: (await readUint32(outputs.overflow, 1))[0],
        requiredCount: (await readUint32(outputs.requiredCount, 1))[0],
        pathCount: (await readUint32(outputs.pathCount, 1))[0],
        sourcePaths: await readUint32(outputs.sourcePaths, pathCapacity),
        sourceRows: (await readUint32(outputs.sourceRows, capacity)).slice(0, count),
        measures: (await readFloat32(outputs.measures, capacity)).slice(0, count)
      };
    },
    getCompileCount() {
      return compileSpy.mock.calls.length;
    },
    destroy() {
      compiled.destroy();
      parameterBuffer.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

/**
 * Asserts that a read-back path output equals an f64 oracle result truncated to `capacity`.
 *
 * @param positionTolerance Absolute tolerance per coordinate.
 * @param measureTolerance Relative tolerance on measures (scaled by `max(1, |expected|)`).
 * @param getPositionError Error metric, by default the largest absolute coordinate difference.
 * @param expectedOverflow Expected overflow flag when another capacity (paths) can overflow.
 * @returns The largest position error.
 */
export function expectLinePathParity(
  actual: LinePathReadback,
  expected: FlatPathResult,
  capacity: number,
  positionTolerance: number,
  measureTolerance: number,
  getPositionError: (actual: number[], expected: number[]) => number = getMaximumCoordinateError,
  expectedOverflow?: number
): number {
  const total = expected.positions.length;
  expect(actual.requiredCount).toBe(total);
  expect(actual.count).toBe(Math.min(total, capacity));
  expect(actual.overflow).toBe(expectedOverflow ?? (total > capacity ? 1 : 0));
  expect(actual.pathOffsets).toEqual(
    expected.pathOffsets.map(offset => Math.min(offset, capacity))
  );
  if (expected.sourceRows.length > 0) {
    expect(actual.sourceRows).toEqual(expected.sourceRows.slice(0, capacity));
  }
  let maximumError = 0;
  actual.positions.forEach((vertex, row) => {
    const reference = expected.positions[row];
    const error = getPositionError(vertex, reference);
    maximumError = Math.max(maximumError, error);
    expect(error, `vertex ${row}: ${vertex} vs ${reference}`).toBeLessThanOrEqual(
      positionTolerance
    );
  });
  (expected.measures.length > 0 ? actual.measures : []).forEach((measure, row) => {
    const reference = expected.measures[row];
    expect(Math.abs(measure - reference), `measure ${row}`).toBeLessThanOrEqual(
      measureTolerance * Math.max(1, Math.abs(reference))
    );
  });
  return maximumError;
}

/** Largest absolute coordinate difference. */
export function getMaximumCoordinateError(actual: number[], expected: number[]): number {
  return Math.max(Math.abs(actual[0] - expected[0]), Math.abs(actual[1] - expected[1]));
}

/** Great-circle distance in meters between two longitude/latitude positions. */
export function getSphericalErrorMeters(actual: number[], expected: number[]): number {
  return getHaversineDistance(actual, expected, 6371008.8);
}
