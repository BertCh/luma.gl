// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it, vi} from 'vitest';
import {
  getGPULineLocateParameterValues,
  GPULinearReferencing,
  GPULineLocate
} from '../../../src/gpu-spatial-analysis/linear-referencing';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  findNearestSegment,
  getVertexMeasures,
  locateAlong,
  projectOntoSegment,
  type ReferenceLines
} from './linear-referencing-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

const NO_LINE = 0xffffffff;

function createRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function createLines(lines: number[][][]): ReferenceLines {
  const values: number[] = [];
  const lineOffsets = [0];
  for (const line of lines) {
    for (const vertex of line) {
      values.push(vertex[0], vertex[1]);
    }
    lineOffsets.push(values.length / 2);
  }
  return {positions: new Float32Array(values), lineOffsets: new Uint32Array(lineOffsets)};
}

type ReferencingReadback = {
  lineIndices: number[];
  segmentIndices: number[];
  fractions: number[];
  footPoints: number[][];
  distances: number[];
  measures: number[];
  sides: number[];
  signedOffsets: number[];
  vertexMeasures: number[];
  overflow: number;
};

function createReferencingFixture(
  device: Device,
  lines: ReferenceLines,
  points: Float32Array,
  candidateCapacity: number
) {
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const pointCount = points.length / 2;
  const rowCount = lines.positions.length / 2;
  const graph = new GPUCommandGraph(device, {id: 'linear-referencing-fixture'});
  const radius = new GPUParameterBuffer(device, {
    id: 'radius',
    format: 'float32',
    length: 1
  });
  const outputs = {
    lineIndices: track(createOutputBuffer(device, pointCount)),
    segmentIndices: track(createOutputBuffer(device, pointCount)),
    fractions: track(createOutputBuffer(device, pointCount)),
    footPoints: track(createOutputBuffer(device, 2 * pointCount)),
    distances: track(createOutputBuffer(device, pointCount)),
    measures: track(createOutputBuffer(device, pointCount)),
    sides: track(createOutputBuffer(device, pointCount)),
    signedOffsets: track(createOutputBuffer(device, pointCount)),
    vertexMeasures: track(createOutputBuffer(device, rowCount)),
    overflow: track(createOutputBuffer(device, 1))
  };
  graph.add(
    new GPULinearReferencing({
      queryPositions: importGraphBuffer(
        graph,
        'points',
        track(createInputBuffer(device, points)),
        'float32x2',
        pointCount
      ),
      positions: importGraphBuffer(
        graph,
        'positions',
        track(createInputBuffer(device, lines.positions)),
        'float32x2',
        rowCount
      ),
      lineOffsets: importGraphBuffer(
        graph,
        'line-offsets',
        track(createInputBuffer(device, lines.lineOffsets)),
        'uint32',
        lines.lineOffsets.length
      ),
      radius: radius.importToGraph(graph),
      candidateCapacity,
      spatialSort: true,
      output: {
        lineIndices: importGraphBuffer(graph, 'o-line', outputs.lineIndices, 'uint32', pointCount),
        segmentIndices: importGraphBuffer(
          graph,
          'o-segment',
          outputs.segmentIndices,
          'uint32',
          pointCount
        ),
        fractions: importGraphBuffer(graph, 'o-fraction', outputs.fractions, 'float32', pointCount),
        footPoints: importGraphBuffer(graph, 'o-foot', outputs.footPoints, 'float32x2', pointCount),
        distances: importGraphBuffer(graph, 'o-distance', outputs.distances, 'float32', pointCount),
        measures: importGraphBuffer(graph, 'o-measure', outputs.measures, 'float32', pointCount),
        sides: importGraphBuffer(graph, 'o-side', outputs.sides, 'sint32', pointCount),
        signedOffsets: importGraphBuffer(
          graph,
          'o-signed',
          outputs.signedOffsets,
          'float32',
          pointCount
        ),
        vertexMeasures: importGraphBuffer(
          graph,
          'o-vertex-measures',
          outputs.vertexMeasures,
          'float32',
          rowCount
        ),
        status: {overflow: importGraphBuffer(graph, 'o-overflow', outputs.overflow, 'uint32', 1)}
      }
    })
  );
  const compiled = graph.compile();
  const compileSpy = vi.spyOn(graph, 'compile');
  return {
    async run(searchRadius: number): Promise<ReferencingReadback> {
      radius.write(new Float32Array([searchRadius]));
      submitGraph(device, compiled, undefined);
      const foot = await readFloat32(outputs.footPoints, 2 * pointCount);
      const sideBytes = await outputs.sides.readAsync();
      return {
        lineIndices: await readUint32(outputs.lineIndices, pointCount),
        segmentIndices: await readUint32(outputs.segmentIndices, pointCount),
        fractions: await readFloat32(outputs.fractions, pointCount),
        footPoints: Array.from({length: pointCount}, (_, row) => foot.slice(2 * row, 2 * row + 2)),
        distances: await readFloat32(outputs.distances, pointCount),
        measures: await readFloat32(outputs.measures, pointCount),
        sides: Array.from(new Int32Array(sideBytes.buffer, sideBytes.byteOffset, pointCount)),
        signedOffsets: await readFloat32(outputs.signedOffsets, pointCount),
        vertexMeasures: await readFloat32(outputs.vertexMeasures, rowCount),
        overflow: (await readUint32(outputs.overflow, 1))[0]
      };
    },
    getCompileCount: () => compileSpy.mock.calls.length,
    destroy() {
      compiled.destroy();
      radius.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

function expectReferencingParity(
  actual: ReferencingReadback,
  lines: ReferenceLines,
  points: Float32Array,
  radius: number
): number {
  const measures = getVertexMeasures(lines);
  let matched = 0;
  for (let point = 0; point < points.length / 2; point++) {
    const query = [points[2 * point], points[2 * point + 1]];
    const expected = findNearestSegment(lines, measures, query, radius);
    const label = `point ${point}`;
    if (!expected) {
      expect(actual.lineIndices[point], label).toBe(NO_LINE);
      expect(actual.distances[point], label).toBe(-1);
      expect(actual.sides[point], label).toBe(0);
      expect(actual.fractions[point], label).toBeNaN();
      expect(actual.signedOffsets[point], label).toBeNaN();
      continue;
    }
    matched++;
    expect(actual.lineIndices[point], label).not.toBe(NO_LINE);
    const segmentRow = lines.lineOffsets[actual.lineIndices[point]] + actual.segmentIndices[point];
    // f32 distances may order near-ties differently from f64: accept any segment as close.
    const chosen = projectOntoSegment(lines, measures, query, segmentRow);
    const scale = Math.max(1, Math.abs(query[0]), Math.abs(query[1]));
    expect(chosen.distance - expected.distance, label).toBeLessThanOrEqual(1e-5 * scale);
    if (chosen.segmentRow === expected.segmentRow) {
      expect(actual.segmentIndices[point], label).toBe(expected.segmentIndex);
    }
    expect(Math.abs(actual.fractions[point] - chosen.fraction), label).toBeLessThanOrEqual(1e-4);
    expect(Math.abs(actual.footPoints[point][0] - chosen.foot[0]), label).toBeLessThanOrEqual(
      1e-5 * scale
    );
    expect(Math.abs(actual.footPoints[point][1] - chosen.foot[1]), label).toBeLessThanOrEqual(
      1e-5 * scale
    );
    expect(Math.abs(actual.distances[point] - chosen.distance), label).toBeLessThanOrEqual(
      1e-5 * scale
    );
    expect(
      Math.abs(actual.measures[point] - chosen.measure),
      `${label} measure`
    ).toBeLessThanOrEqual(1e-5 * Math.max(1, chosen.measure));
    if (chosen.distance > 1e-4 * scale) {
      expect(actual.sides[point], label).toBe(chosen.side);
      expect(Math.abs(actual.signedOffsets[point] - chosen.side * chosen.distance)).toBeLessThan(
        1e-5 * scale
      );
    }
  }
  return matched;
}

it('GPULinearReferencing matches brute force on random polylines and per-frame radii', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createRandom(3);
  const lineList = Array.from({length: 120}, (_, index) => {
    if (index % 31 === 4) {
      return [];
    }
    if (index % 29 === 6) {
      return [[random() * 100, random() * 100]];
    }
    let x = random() * 100;
    let y = random() * 100;
    return Array.from({length: 2 + Math.floor(random() * 10)}, (__, vertex) => {
      if (vertex > 0 && random() > 0.08) {
        x += (random() - 0.5) * 12;
        y += (random() - 0.5) * 12;
      }
      return [x, y];
    });
  });
  const lines = createLines(lineList);
  const pointCount = 2000;
  const points = new Float32Array(pointCount * 2);
  for (let point = 0; point < pointCount; point++) {
    points[2 * point] = random() * 110 - 5;
    points[2 * point + 1] = random() * 110 - 5;
  }
  // Exactly on vertices and on segment midpoints.
  points.set([lines.positions[0], lines.positions[1], lines.positions[2], lines.positions[3]], 0);
  const fixture = createReferencingFixture(device, lines, points, 400000);
  for (const radius of [1.5, 5, 0.25, 12]) {
    const actual = await fixture.run(radius);
    expect(actual.overflow).toBe(0);
    const matched = expectReferencingParity(actual, lines, points, radius);
    expect(matched).toBeGreaterThan(0);
  }
  const measures = getVertexMeasures(lines);
  const actual = await fixture.run(1);
  actual.vertexMeasures.forEach((measure, row) => {
    if (!Number.isNaN(measures[row])) {
      expect(Math.abs(measure - measures[row])).toBeLessThanOrEqual(
        1e-5 * Math.max(1, measures[row])
      );
    }
  });
  expect(fixture.getCompileCount()).toBe(0);
  fixture.destroy();
});

it('GPULinearReferencing breaks ties on the smallest segment and reports sides', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lines = createLines([
    // An L: (0,0) -> (10,0) -> (10,10)
    [
      [0, 0],
      [10, 0],
      [10, 10]
    ],
    // A parallel line at y = 4 running west: ties with line 0 at y = 2.
    [
      [10, 4],
      [0, 4]
    ],
    // Zero-length segment followed by a real one.
    [
      [20, 0],
      [20, 0],
      [30, 0]
    ]
  ]);
  const points = new Float32Array([
    5,
    2, // equidistant to line 0 segment 0 and line 1: smallest segment row wins
    5,
    -3, // right of line 0
    12,
    5, // right of the vertical leg (direction north, point east)
    8,
    5, // left of the vertical leg, also 1 from line 1
    10,
    0, // on the corner vertex: segment 0 at fraction 1
    20,
    1, // nearest the zero-length segment's point and the next segment's start
    25,
    -2
  ]);
  const fixture = createReferencingFixture(device, lines, points, 1000);
  const actual = await fixture.run(100);
  expectReferencingParity(actual, lines, points, 100);
  expect(actual.lineIndices.slice(0, 5)).toEqual([0, 0, 0, 1, 0]);
  expect(actual.segmentIndices[0]).toBe(0);
  // Point 3 snaps to the westbound line 1, so it lies on that line's right.
  expect(actual.sides.slice(0, 5)).toEqual([1, -1, -1, -1, 0]);
  expect(actual.measures[2]).toBe(15);
  expect(actual.distances[4]).toBe(0);
  expect(actual.fractions[4]).toBe(1);
  // Zero-length segment (row 5) wins the tie with segment row 6 at distance 1.
  expect([actual.lineIndices[5], actual.segmentIndices[5], actual.fractions[5]]).toEqual([2, 0, 0]);
  expect(actual.measures[6]).toBe(5);
  expect(actual.signedOffsets[6]).toBe(-2);
  fixture.destroy();
});

it('GPULinearReferencing reports candidate overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lines = createLines([Array.from({length: 50}, (_, index) => [index, index % 2])]);
  const points = new Float32Array(Array.from({length: 100}, (_, index) => index / 2));
  const fixture = createReferencingFixture(device, lines, points, 8);
  const actual = await fixture.run(10);
  expect(actual.overflow).toBe(1);
  fixture.destroy();
});

it('GPULineLocate places events by distance and fraction, with offsets and per-frame animation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createRandom(5);
  const lineList = Array.from({length: 40}, (_, index) => {
    if (index === 7) {
      return [];
    }
    if (index === 8) {
      return [[3, 4]];
    }
    let x = random() * 100;
    let y = random() * 100;
    return Array.from({length: 2 + Math.floor(random() * 8)}, (__, vertex) => {
      if (vertex > 0 && random() > 0.1) {
        x += (random() - 0.5) * 12;
        y += (random() - 0.5) * 12;
      }
      return [x, y];
    });
  });
  const lines = createLines(lineList);
  const measures = getVertexMeasures(lines);
  const eventCount = 600;
  const eventLineIndices = new Uint32Array(eventCount);
  const eventMeasures = new Float32Array(eventCount);
  const eventOffsets = new Float32Array(eventCount);
  for (let event = 0; event < eventCount; event++) {
    eventLineIndices[event] = event === 3 ? 999 : Math.floor(random() * 40);
    eventMeasures[event] = random() * 60 - 5;
    eventOffsets[event] = event % 3 === 0 ? random() * 2 - 1 : 0;
  }
  // Exactly on a vertex measure.
  eventLineIndices[0] = 0;
  eventMeasures[0] = Math.fround(measures[lines.lineOffsets[0] + 1]);
  eventOffsets[0] = 0;

  for (const measureMode of ['distance', 'fraction'] as const) {
    const buffers: Buffer[] = [];
    const track = (buffer: Buffer) => {
      buffers.push(buffer);
      return buffer;
    };
    const graph = new GPUCommandGraph(device, {id: `line-locate-${measureMode}`});
    const parameters = new GPUParameterBuffer(device, {
      id: 'locate-parameters',
      format: 'float32',
      length: 4
    });
    const outputs = {
      positions: track(createOutputBuffer(device, 2 * eventCount)),
      segmentIndices: track(createOutputBuffer(device, eventCount)),
      tangents: track(createOutputBuffer(device, 2 * eventCount)),
      angles: track(createOutputBuffer(device, eventCount)),
      statuses: track(createOutputBuffer(device, eventCount))
    };
    const input = <Format extends 'float32x2' | 'uint32' | 'float32'>(
      name: string,
      values: Float32Array | Uint32Array,
      format: Format
    ): GraphDataView<Format> =>
      importGraphBuffer(
        graph,
        name,
        track(createInputBuffer(device, values)),
        format,
        format === 'float32x2' ? values.length / 2 : values.length
      );
    graph.add(
      new GPULineLocate({
        positions: input('positions', lines.positions, 'float32x2'),
        lineOffsets: input('line-offsets', lines.lineOffsets, 'uint32'),
        eventLineIndices: input('event-lines', eventLineIndices, 'uint32'),
        eventMeasures: input('event-measures', eventMeasures, 'float32'),
        eventOffsets: input('event-offsets', eventOffsets, 'float32'),
        measureMode,
        parameters: parameters.importToGraph(graph),
        output: {
          positions: importGraphBuffer(
            graph,
            'o-positions',
            outputs.positions,
            'float32x2',
            eventCount
          ),
          segmentIndices: importGraphBuffer(
            graph,
            'o-segments',
            outputs.segmentIndices,
            'uint32',
            eventCount
          ),
          tangents: importGraphBuffer(
            graph,
            'o-tangents',
            outputs.tangents,
            'float32x2',
            eventCount
          ),
          angles: importGraphBuffer(graph, 'o-angles', outputs.angles, 'float32', eventCount),
          statuses: importGraphBuffer(graph, 'o-statuses', outputs.statuses, 'uint32', eventCount)
        }
      })
    );
    const compiled = graph.compile();
    const compileSpy = vi.spyOn(graph, 'compile');
    const frames =
      measureMode === 'distance'
        ? [
            {measureScale: 1, measureOffset: 0},
            {measureScale: 1, measureOffset: 7.5}
          ]
        : [
            {measureScale: 1 / 50, measureOffset: 0},
            {measureScale: 1 / 50, measureOffset: 0.25}
          ];
    for (const frame of frames) {
      parameters.write(getGPULineLocateParameterValues(frame));
      submitGraph(device, compiled, undefined);
      const positions = await readFloat32(outputs.positions, 2 * eventCount);
      const segmentIndices = await readUint32(outputs.segmentIndices, eventCount);
      const tangents = await readFloat32(outputs.tangents, 2 * eventCount);
      const angles = await readFloat32(outputs.angles, eventCount);
      const statuses = await readUint32(outputs.statuses, eventCount);
      for (let event = 0; event < eventCount; event++) {
        const measure =
          Math.fround(eventMeasures[event] * Math.fround(frame.measureScale)) +
          Math.fround(frame.measureOffset);
        const expected = locateAlong(lines, measures, eventLineIndices[event], measure, {
          fraction: measureMode === 'fraction',
          offset: eventOffsets[event]
        });
        const label = `${measureMode} event ${event}`;
        // Measures within f32 rounding of a line end may clamp on one side only.
        const lineEnd = lines.lineOffsets[Math.min(eventLineIndices[event], 39) + 1] - 1;
        const total = measures[lineEnd] ?? 0;
        const target = measureMode === 'fraction' ? measure * total : measure;
        const nearEnd = Math.abs(target) < 1e-4 || Math.abs(target - total) < 1e-4;
        if (!nearEnd) {
          expect(statuses[event], label).toBe(expected.status);
        }
        if (expected.status === 2) {
          expect(positions[2 * event], label).toBeNaN();
          continue;
        }
        // Measures on a vertex may land on either neighbor segment in f32; positions agree anyway.
        const lineStart = lines.lineOffsets[Math.min(eventLineIndices[event], 39)];
        const nearVertex = measures
          .slice(lineStart, lineEnd + 1)
          .some(vertexMeasure => Math.abs(vertexMeasure - target) < 1e-4);
        if (!nearVertex) {
          expect(segmentIndices[event], label).toBe(expected.segmentIndex);
        }
        expect(Math.abs(positions[2 * event] - expected.position[0]), label).toBeLessThan(2e-4);
        expect(Math.abs(positions[2 * event + 1] - expected.position[1]), label).toBeLessThan(2e-4);
        if (segmentIndices[event] === expected.segmentIndex) {
          expect(Math.abs(tangents[2 * event] - expected.tangent[0]), label).toBeLessThan(1e-5);
          expect(
            Math.abs(
              angles[event] - (Math.atan2(expected.tangent[1], expected.tangent[0]) * 180) / Math.PI
            ),
            label
          ).toBeLessThan(1e-3);
        }
      }
    }
    expect(compileSpy.mock.calls.length).toBe(0);
    compiled.destroy();
    parameters.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
});
