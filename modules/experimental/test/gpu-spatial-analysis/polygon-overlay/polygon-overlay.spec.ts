// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPUBufferSurface,
  GPUPolygonOverlay,
  type GPUPolygonOverlayOperation,
  type GPUPolygonOverlayOutput
} from '../../../src/gpu-spatial-analysis';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

type GraphBuffers = {
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

function createGraphBuffers(device: Device, graph: GPUCommandGraph): GraphBuffers {
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

function createPolygonInput(
  graphBuffers: GraphBuffers,
  prefix: string,
  coordinates: readonly number[]
) {
  return {
    kind: 'polygons' as const,
    positions: graphBuffers.input(
      `${prefix}-positions`,
      Float32Array.from(coordinates),
      'float32x2',
      coordinates.length / 2
    ),
    featureOffsets: graphBuffers.input(
      `${prefix}-feature-offsets`,
      Uint32Array.of(0, 1),
      'uint32',
      2
    ),
    polygonOffsets: graphBuffers.input(
      `${prefix}-polygon-offsets`,
      Uint32Array.of(0, 1),
      'uint32',
      2
    ),
    ringOffsets: graphBuffers.input(
      `${prefix}-ring-offsets`,
      Uint32Array.of(0, coordinates.length / 2),
      'uint32',
      2
    )
  };
}

function createOverlayOutput(
  graphBuffers: GraphBuffers,
  prefix: string,
  vertexCapacity = 128,
  ringCapacity = 32,
  boundaryCapacity = 128
): {
  output: GPUPolygonOverlayOutput;
  positions: Buffer;
  featureOffsets: Buffer;
  polygonOffsets: Buffer;
  ringOffsets: Buffer;
  count: Buffer;
  requiredCount: Buffer;
  overflow: Buffer;
  candidateOverflow: Buffer;
} {
  const positions = graphBuffers.output(`${prefix}-positions`, 'float32x2', vertexCapacity);
  const featureOffsets = graphBuffers.output(
    `${prefix}-feature-offsets`,
    'uint32',
    ringCapacity + 1
  );
  const polygonOffsets = graphBuffers.output(
    `${prefix}-polygon-offsets`,
    'uint32',
    ringCapacity + 1
  );
  const ringOffsets = graphBuffers.output(`${prefix}-ring-offsets`, 'uint32', ringCapacity + 1);
  const sourceIds = graphBuffers.output(`${prefix}-source-ids`, 'uint32', ringCapacity);
  const scalar = (name: string) => graphBuffers.output(`${prefix}-${name}`, 'uint32', 1);
  const count = scalar('count');
  const requiredCount = scalar('required-count');
  const overflow = scalar('overflow');
  const candidateOverflow = scalar('candidate-overflow');
  const boundaryCount = scalar('boundary-count');
  const boundaryRequiredCount = scalar('boundary-required-count');
  const boundaryOverflow = scalar('boundary-overflow');
  const boundaryCandidateOverflow = scalar('boundary-candidate-overflow');
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
        count: boundaryCount.view,
        requiredCount: boundaryRequiredCount.view,
        overflow: boundaryOverflow.view,
        candidateOverflow: boundaryCandidateOverflow.view
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
    overflow: overflow.buffer,
    candidateOverflow: candidateOverflow.buffer
  };
}

async function readPolygonArea(result: ReturnType<typeof createOverlayOutput>): Promise<number> {
  const [featureCount] = await readUint32(result.count, 1);
  if (featureCount === 0) {
    return 0;
  }
  const features = await readUint32(result.featureOffsets, featureCount + 1);
  const polygonCount = features[featureCount];
  const polygons = await readUint32(result.polygonOffsets, polygonCount + 1);
  const ringCount = polygons[polygonCount];
  const rings = await readUint32(result.ringOffsets, ringCount + 1);
  const coordinates = await readFloat32(result.positions, rings[ringCount] * 2);
  let area = 0;
  for (let ring = 0; ring < ringCount; ring++) {
    let doubled = 0;
    const begin = rings[ring];
    const end = rings[ring + 1];
    for (let row = begin; row < end; row++) {
      const next = row + 1 === end ? begin : row + 1;
      doubled +=
        coordinates[row * 2] * coordinates[next * 2 + 1] -
        coordinates[next * 2] * coordinates[row * 2 + 1];
    }
    area += doubled / 2;
  }
  return Math.abs(area);
}

async function runRectangleOverlay(
  device: Device,
  operation: GPUPolygonOverlayOperation,
  capacity = {intersections: 128, nodedSegments: 128}
) {
  const graph = new GPUCommandGraph(device, {id: `overlay-${operation}`});
  const graphBuffers = createGraphBuffers(device, graph);
  const left = createPolygonInput(graphBuffers, 'left', [0, 0, 2, 0, 2, 2, 0, 2]);
  const right = createPolygonInput(graphBuffers, 'right', [1, 0, 3, 0, 3, 2, 1, 2]);
  const result = createOverlayOutput(graphBuffers, 'result');
  graph.add(
    new GPUPolygonOverlay({
      left,
      right,
      operation,
      capacity,
      vertexTolerance: 1e-5,
      output: result.output
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [count, requiredCount, overflow, candidateOverflow, area] = await Promise.all([
    readUint32(result.count, 1).then(values => values[0]),
    readUint32(result.requiredCount, 1).then(values => values[0]),
    readUint32(result.overflow, 1).then(values => values[0]),
    readUint32(result.candidateOverflow, 1).then(values => values[0]),
    readPolygonArea(result)
  ]);
  compiled.destroy();
  for (const buffer of graphBuffers.buffers) {
    buffer.destroy();
  }
  return {count, requiredCount, overflow, candidateOverflow, area};
}

it('GPUPolygonOverlay matches rectangle set-operation areas and complete status', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const [operation, expectedArea, expectedFeatureCount] of [
    ['intersection', 2, 1],
    ['union', 6, 1],
    ['difference', 2, 1],
    ['symmetric-difference', 4, 2]
  ] as const) {
    const result = await runRectangleOverlay(device, operation);
    expect(result.count, operation).toBe(expectedFeatureCount);
    expect(result.overflow, operation).toBe(0);
    expect(result.candidateOverflow, operation).toBe(0);
    expect(result.area, operation).toBeCloseTo(expectedArea, 4);
  }
});

it('GPUPolygonOverlay reports incomplete candidate generation when arrangement capacity is bounded', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const result = await runRectangleOverlay(device, 'intersection', {
    intersections: 1,
    nodedSegments: 1
  });
  expect(result.overflow).toBe(1);
  expect(result.requiredCount).toBeGreaterThanOrEqual(result.count);
});

it('GPUPolygonOverlay dissolves overlapping features into their union surface', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const graph = new GPUCommandGraph(device, {id: 'overlay-dissolve'});
  const graphBuffers = createGraphBuffers(device, graph);
  const result = createOverlayOutput(graphBuffers, 'dissolve-result');
  graph.add(
    new GPUPolygonOverlay({
      left: {
        kind: 'polygons',
        positions: graphBuffers.input(
          'dissolve-positions',
          Float32Array.of(0, 0, 2, 0, 2, 2, 0, 2, 1, 0, 3, 0, 3, 2, 1, 2),
          'float32x2',
          8
        ),
        featureOffsets: graphBuffers.input(
          'dissolve-feature-offsets',
          Uint32Array.of(0, 1, 2),
          'uint32',
          3
        ),
        polygonOffsets: graphBuffers.input(
          'dissolve-polygon-offsets',
          Uint32Array.of(0, 1, 2),
          'uint32',
          3
        ),
        ringOffsets: graphBuffers.input(
          'dissolve-ring-offsets',
          Uint32Array.of(0, 4, 8),
          'uint32',
          3
        )
      },
      operation: 'dissolve',
      capacity: {intersections: 128, nodedSegments: 128},
      vertexTolerance: 1e-5,
      output: result.output
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [count, overflow, candidateOverflow, area] = await Promise.all([
    readUint32(result.count, 1).then(values => values[0]),
    readUint32(result.overflow, 1).then(values => values[0]),
    readUint32(result.candidateOverflow, 1).then(values => values[0]),
    readPolygonArea(result)
  ]);
  expect(count).toBe(1);
  expect(overflow).toBe(0);
  expect(candidateOverflow).toBe(0);
  expect(area).toBeCloseTo(6, 4);
  compiled.destroy();
  for (const buffer of graphBuffers.buffers) {
    buffer.destroy();
  }
});

it('GPUBufferSurface assembles a square-capped line into a repaired polygon', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const graph = new GPUCommandGraph(device, {id: 'buffer-line'});
  const graphBuffers = createGraphBuffers(device, graph);
  const result = createOverlayOutput(graphBuffers, 'buffer-result', 128, 32, 128);
  graph.add(
    new GPUBufferSurface({
      geometry: {
        kind: 'lines',
        positions: graphBuffers.input(
          'line-positions',
          Float32Array.of(0, 0, 2, 0),
          'float32x2',
          2
        ),
        lineOffsets: graphBuffers.input('line-offsets', Uint32Array.of(0, 2), 'uint32', 2)
      },
      parameters: graphBuffers.input(
        'buffer-parameters',
        Float32Array.of(1, 5, 0, 0),
        'float32',
        4
      ),
      joinStyle: 'bevel',
      capStyle: 'square',
      quadSegments: 2,
      vertexTolerance: 1e-5,
      capacity: {intersections: 128, nodedSegments: 128},
      output: result.output
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [count, overflow, area] = await Promise.all([
    readUint32(result.count, 1).then(values => values[0]),
    readUint32(result.overflow, 1).then(values => values[0]),
    readPolygonArea(result)
  ]);
  expect(count).toBe(1);
  expect(overflow).toBe(0);
  expect(area).toBeCloseTo(8, 4);
  compiled.destroy();
  for (const buffer of graphBuffers.buffers) {
    buffer.destroy();
  }
});

it('GPUBufferSurface grows and shrinks polygon surfaces with signed distance', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const [distance, expectedArea] of [
    [1, 16],
    [-0.5, 1]
  ] as const) {
    const graph = new GPUCommandGraph(device, {id: `buffer-polygon-${distance}`});
    const graphBuffers = createGraphBuffers(device, graph);
    const result = createOverlayOutput(graphBuffers, `polygon-buffer-result-${distance}`);
    graph.add(
      new GPUBufferSurface({
        geometry: createPolygonInput(
          graphBuffers,
          `polygon-buffer-${distance}`,
          [0, 0, 2, 0, 2, 2, 0, 2]
        ),
        parameters: graphBuffers.input(
          `polygon-buffer-parameters-${distance}`,
          Float32Array.of(distance, 5, 0, 0),
          'float32',
          4
        ),
        joinStyle: 'mitre',
        vertexTolerance: 1e-5,
        capacity: {intersections: 128, nodedSegments: 128},
        output: result.output
      })
    );
    const compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    const [count, overflow, area] = await Promise.all([
      readUint32(result.count, 1).then(values => values[0]),
      readUint32(result.overflow, 1).then(values => values[0]),
      readPolygonArea(result)
    ]);
    expect(count, String(distance)).toBe(1);
    expect(overflow, String(distance)).toBe(0);
    expect(area, String(distance)).toBeCloseTo(expectedArea, 4);
    compiled.destroy();
    for (const buffer of graphBuffers.buffers) {
      buffer.destroy();
    }
  }
});
