// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUNetworkSnapping,
  GPU_NETWORK_SNAPPING_NONE,
  type GPUNetworkSnappingProps,
  type GPUNetworkSnappingSeedDirection
} from '../../../src/gpu-network/network-accessibility';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {buildCSR, createGridFixture, NONE, snapOracle} from './network-accessibility-oracle';

type SnappingResult = {
  edges: number[];
  fractions: number[];
  distances: number[];
  sourceCosts: number[];
  targetCosts: number[];
  positions: number[];
  seedNodes: number[];
  seedCosts: number[];
  overflow?: number;
};

type SnappingOptions = {
  edgeInput: 'coo' | 'csr';
  edgeCosts?: boolean;
  maxSnapDistance?: number;
  candidateCapacity?: number;
  seedDirection?: GPUNetworkSnappingSeedDirection;
};

/** Grid fixture plus a CSR; `edges` keeps both directions of every link. */
const WIDTH = 8;
const NODE_COUNT = WIDTH * WIDTH;
const GRID = createGridFixture(21, WIDTH, WIDTH);
const CSR = buildCSR(NODE_COUNT, GRID.edges);

/**
 * Points: on an edge interior, exactly on a node (tie among incident edges), on a node at the grid
 * corner, near an edge endpoint, outside the grid, and far away (beyond a 1.5 snap distance).
 */
const POINTS = Float32Array.from([
  0.25, 0.1, 3, 3, 0, 0, 6.98, 5, 7.6, 2.5, 20, 20, 4.5, 6.5, 2.2, 0.9, 1, 7, 5.5, 5.5
]);
const POINT_COUNT = POINTS.length / 2;

async function runSnapping(device: Device, options: SnappingOptions): Promise<SnappingResult> {
  const graph = new GPUCommandGraph(device, {id: 'snapping'});
  const buffers: Buffer[] = [];
  const input = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    id: string,
    values: Float32Array | Uint32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      id,
      buffer,
      format,
      format === 'float32x2' ? values.length / 2 : values.length
    );
  };
  const outputs: Record<string, Buffer> = {};
  const output = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    id: string,
    format: Format,
    length: number
  ) => {
    const buffer = createOutputBuffer(device, format === 'float32x2' ? length * 2 : length);
    buffers.push(buffer);
    outputs[id] = buffer;
    return importGraphBuffer(graph, id, buffer, format, length);
  };
  const maxSnapDistance =
    options.maxSnapDistance !== undefined
      ? new GPUParameterBuffer(device, {
          id: 'max-snap-distance',
          format: 'float32',
          length: 1,
          values: Float32Array.from([options.maxSnapDistance])
        })
      : undefined;
  const props: GPUNetworkSnappingProps = {
    id: 'snap',
    points: input('points', POINTS, 'float32x2'),
    nodePositions: input('positions', GRID.positions, 'float32x2'),
    edgeTargets: input('targets', CSR.neighbors, 'uint32'),
    ...(options.edgeInput === 'csr'
      ? {offsets: input('offsets', CSR.offsets, 'uint32')}
      : {edgeSources: input('sources', CSR.sources, 'uint32')}),
    edgeCosts: options.edgeCosts ? input('costs', CSR.weights, 'float32') : undefined,
    maxSnapDistance: maxSnapDistance?.importToGraph(graph),
    candidateCapacity: options.candidateCapacity,
    seedDirection: options.seedDirection,
    snappedEdges: output('edges', 'uint32', POINT_COUNT),
    snapFractions: output('fractions', 'float32', POINT_COUNT),
    snapDistances: output('distances', 'float32', POINT_COUNT),
    sourceCosts: output('sourceCosts', 'float32', POINT_COUNT),
    targetCosts: output('targetCosts', 'float32', POINT_COUNT),
    snappedPositions: output('positions-out', 'float32x2', POINT_COUNT),
    seedNodes: output('seedNodes', 'uint32', POINT_COUNT * 2),
    seedCosts: output('seedCosts', 'float32', POINT_COUNT * 2),
    overflow: options.candidateCapacity ? output('overflow', 'uint32', 1) : undefined
  };
  graph.add(new GPUNetworkSnapping(props));
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result: SnappingResult = {
    edges: await readUint32(outputs['edges'], POINT_COUNT),
    fractions: await readFloat32(outputs['fractions'], POINT_COUNT),
    distances: await readFloat32(outputs['distances'], POINT_COUNT),
    sourceCosts: await readFloat32(outputs['sourceCosts'], POINT_COUNT),
    targetCosts: await readFloat32(outputs['targetCosts'], POINT_COUNT),
    positions: await readFloat32(outputs['positions-out'], POINT_COUNT * 2),
    seedNodes: await readUint32(outputs['seedNodes'], POINT_COUNT * 2),
    seedCosts: await readFloat32(outputs['seedCosts'], POINT_COUNT * 2),
    overflow: outputs['overflow'] ? (await readUint32(outputs['overflow'], 1))[0] : undefined
  };
  compiled.destroy();
  maxSnapDistance?.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

function expectSnapsMatchOracle(
  result: SnappingResult,
  options: {
    edgeCosts?: boolean;
    maxSnapDistance?: number;
    seedDirection?: string;
  }
): void {
  const expected = snapOracle(POINTS, GRID.positions, CSR.sources, CSR.neighbors, {
    edgeCosts: options.edgeCosts ? CSR.weights : undefined,
    maxSnapDistance: options.maxSnapDistance
  });
  expect(result.edges).toEqual(expected.map(snap => snap.edge));
  const direction = options.seedDirection ?? 'both';
  expected.forEach((snap, point) => {
    for (const [actual, value] of [
      [result.fractions[point], snap.fraction],
      [result.distances[point], snap.distance],
      [result.sourceCosts[point], snap.sourceCost],
      [result.targetCosts[point], snap.targetCost]
    ]) {
      expect(actual).toBeCloseTo(value, 4);
    }
    const isSnapped = snap.edge !== NONE;
    const useSource = isSnapped && direction !== 'forward';
    const useTarget = isSnapped && direction !== 'reverse';
    expect(result.seedNodes[point * 2]).toBe(useSource ? CSR.sources[snap.edge] : NONE);
    expect(result.seedNodes[point * 2 + 1]).toBe(useTarget ? CSR.neighbors[snap.edge] : NONE);
    expect(result.seedCosts[point * 2]).toBeCloseTo(useSource ? snap.sourceCost : -1, 4);
    expect(result.seedCosts[point * 2 + 1]).toBeCloseTo(useTarget ? snap.targetCost : -1, 4);
    if (isSnapped) {
      const source = CSR.sources[snap.edge];
      const target = CSR.neighbors[snap.edge];
      for (const axis of [0, 1]) {
        const start = GRID.positions[source * 2 + axis];
        const end = GRID.positions[target * 2 + axis];
        expect(result.positions[point * 2 + axis]).toBeCloseTo(
          start + snap.fraction * (end - start),
          4
        );
      }
    } else {
      expect(result.positions[point * 2]).toBeCloseTo(POINTS[point * 2], 5);
    }
  });
}

it('GPUNetworkSnapping exhaustive scan matches the oracle, including endpoint ties', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const result = await runSnapping(device, {edgeInput: 'coo'});
  expectSnapsMatchOracle(result, {});
  // (3, 3) is node 27: every incident edge is at distance 0 and the smallest CSR row wins. That is
  // the in-edge 19 -> 27, so the point sits at fraction 1 with a zero cost to the edge target.
  const incident = Array.from(CSR.neighbors.keys()).filter(
    edge => CSR.sources[edge] === 27 || CSR.neighbors[edge] === 27
  );
  expect(result.edges[1]).toBe(Math.min(...incident));
  expect(CSR.sources[result.edges[1]]).toBe(19);
  expect(result.fractions[1]).toBe(1);
  expect(result.distances[1]).toBe(0);
  expect(result.targetCosts[1]).toBe(0);
  // The corner node (0, 0) snaps onto its first out-edge at fraction 0.
  expect(result.edges[2]).toBe(0);
  // Unlimited: even the far point snaps.
  expect(result.edges[5]).not.toBe(GPU_NETWORK_SNAPPING_NONE);
});

it('GPUNetworkSnapping supports CSR input, edge costs, a snap limit, and seed directions', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const seedDirection of ['both', 'forward', 'reverse'] as const) {
    const options = {
      edgeInput: 'csr' as const,
      edgeCosts: true,
      maxSnapDistance: 1.5,
      seedDirection
    };
    const result = await runSnapping(device, options);
    expectSnapsMatchOracle(result, options);
    expect(result.edges[5]).toBe(GPU_NETWORK_SNAPPING_NONE);
    expect(result.distances[5]).toBe(-1);
  }
});

it('GPUNetworkSnapping BVH candidate path matches the exhaustive scan', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const options = {
    edgeInput: 'coo' as const,
    edgeCosts: true,
    maxSnapDistance: 1.5,
    candidateCapacity: 1024
  };
  const joined = await runSnapping(device, options);
  expectSnapsMatchOracle(joined, options);
  expect(joined.overflow).toBe(0);
  const scanned = await runSnapping(device, {
    ...options,
    candidateCapacity: undefined
  });
  expect(joined.edges).toEqual(scanned.edges);
  expect(joined.fractions).toEqual(scanned.fractions);
  expect(joined.distances).toEqual(scanned.distances);

  const overflowing = await runSnapping(device, {
    ...options,
    candidateCapacity: 4
  });
  expect(overflowing.overflow).toBe(1);
});
