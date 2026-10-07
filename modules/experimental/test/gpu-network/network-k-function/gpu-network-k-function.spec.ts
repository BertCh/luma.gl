// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUNetworkKFunctionParameterValues,
  GPU_NETWORK_K_FUNCTION_PARAMETER_LENGTH,
  GPUNetworkKFunction
} from '../../../src/gpu-network/network-k-function/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  buildCSR,
  createGridFixture,
  type AccessibilityCSR
} from '../network-accessibility/network-accessibility-oracle';
import {
  getKValues,
  getPairCounts,
  simulateEvents,
  snapEvents,
  type KFunctionEvent
} from './network-k-function-oracle';
import {SPAGHETTI_FIXTURE} from './spaghetti-fixture';

type KFunctionOptions = {
  csr: AccessibilityCSR;
  positions: Float32Array;
  points: Float32Array;
  networkLength: number;
  maxDistance: number;
  bandCount: number;
  simulationCount?: number;
  activeSimulations?: number;
  seed?: number;
  rowsPerBlock?: number;
  laneCount?: number;
  maxSnapDistance?: number;
  maxIterations?: number;
};

type KFunctionResult = {
  kValues: number[];
  pairCounts: number[];
  envelope?: number[];
  snappedEventCount: number;
  converged: number;
};

async function runKFunction(device: Device, options: KFunctionOptions): Promise<KFunctionResult> {
  const {csr, positions, points, bandCount} = options;
  const simulationCount = options.simulationCount ?? 0;
  const patternCount = 1 + simulationCount;
  const graph = new GPUCommandGraph(device, {id: 'k-function'});
  const buffers: Buffer[] = [];
  const parameterBuffers: GPUParameterBuffer[] = [];
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
  const output = <Format extends 'uint32' | 'float32'>(
    id: string,
    format: Format,
    length: number
  ) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    outputs[id] = buffer;
    return importGraphBuffer(graph, id, buffer, format, length);
  };
  const scalar = (id: string, value: number) => {
    const buffer = new GPUParameterBuffer(device, {
      id,
      format: 'float32',
      length: 1,
      values: Float32Array.from([value])
    });
    parameterBuffers.push(buffer);
    return buffer.importToGraph(graph);
  };
  const parameters = new GPUParameterBuffer(device, {
    id: 'k-parameters',
    format: 'uint32',
    length: GPU_NETWORK_K_FUNCTION_PARAMETER_LENGTH,
    values: getGPUNetworkKFunctionParameterValues({
      seed: options.seed ?? 12345,
      activeSimulations: options.activeSimulations ?? simulationCount
    })
  });
  parameterBuffers.push(parameters);
  graph.add(
    new GPUNetworkKFunction({
      id: 'k',
      points: input('points', points, 'float32x2'),
      nodePositions: input('positions', positions, 'float32x2'),
      offsets: input('offsets', csr.offsets, 'uint32'),
      neighbors: input('neighbors', csr.neighbors, 'uint32'),
      weights: input('weights', csr.weights, 'float32'),
      maxSnapDistance:
        options.maxSnapDistance !== undefined
          ? scalar('max-snap-distance', options.maxSnapDistance)
          : undefined,
      maxDistance: scalar('max-distance', options.maxDistance),
      networkLength: scalar('network-length', options.networkLength),
      parameters: parameters.importToGraph(graph),
      bandCount,
      simulationCount,
      rowsPerBlock: options.rowsPerBlock,
      laneCount: options.laneCount,
      maxIterations: options.maxIterations ?? 32,
      kValues: output('k-values', 'float32', patternCount * bandCount),
      pairCounts: output('pair-counts', 'uint32', patternCount * bandCount),
      envelope: simulationCount > 0 ? output('envelope', 'float32', 3 * bandCount) : undefined,
      snappedEventCount: output('snapped', 'uint32', 1),
      converged: output('converged', 'uint32', 1)
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result: KFunctionResult = {
    kValues: await readFloat32(outputs['k-values'], patternCount * bandCount),
    pairCounts: await readUint32(outputs['pair-counts'], patternCount * bandCount),
    envelope: outputs['envelope']
      ? await readFloat32(outputs['envelope'], 3 * bandCount)
      : undefined,
    snappedEventCount: (await readUint32(outputs['snapped'], 1))[0],
    converged: (await readUint32(outputs['converged'], 1))[0]
  };
  compiled.destroy();
  for (const buffer of parameterBuffers) {
    buffer.destroy();
  }
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

it('GPUNetworkKFunction matches spaghetti GlobalAutoK on a jittered grid', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const fixture = SPAGHETTI_FIXTURE;
  const nodePositions = Float32Array.from(fixture.nodes.flat());
  const edges: [number, number, number][] = [];
  for (const [from, to] of fixture.edges) {
    const length = Math.fround(
      Math.hypot(
        nodePositions[to * 2] - nodePositions[from * 2],
        nodePositions[to * 2 + 1] - nodePositions[from * 2 + 1]
      )
    );
    edges.push([from, to, length], [to, from, length]);
  }
  const csr = buildCSR(fixture.nodes.length, edges);
  const result = await runKFunction(device, {
    csr,
    positions: nodePositions,
    points: Float32Array.from(fixture.points.flat()),
    networkLength: fixture.networkLength,
    maxDistance: fixture.maxDistance,
    bandCount: fixture.bandCount,
    rowsPerBlock: 16
  });
  expect(result.converged).toBe(1);
  expect(result.snappedEventCount).toBe(fixture.points.length);
  expect(result.pairCounts).toEqual(fixture.counts);
  result.kValues.forEach((value, band) => {
    expect(value).toBeCloseTo(fixture.k[band], 3);
  });
});

const GRID = createGridFixture(5, 6, 5);
const GRID_NODE_COUNT = 30;
const GRID_CSR = buildCSR(GRID_NODE_COUNT, GRID.edges);
const GRID_NETWORK_LENGTH = GRID_CSR.weights.reduce((sum, weight) => sum + weight, 0) / 2;
const GRID_POINTS = Float32Array.from(
  Array.from({length: 24}, (_, index) => {
    // Deterministic scatter over [-0.3, 5.3] x [-0.3, 4.3], with a few far outside the grid.
    const x = (((index * 37) % 19) / 19) * 5.6 - 0.3 + (index === 5 ? 9 : 0);
    const y = (((index * 53) % 17) / 17) * 4.6 - 0.3 + (index === 11 ? 9 : 0);
    return [x, y];
  }).flat()
);

it('GPUNetworkKFunction matches the CPU oracle for observed and simulated patterns', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const bandCount = 9;
  const maxDistance = 14;
  const simulationCount = 4;
  const activeSimulations = 3;
  const seed = 987654;
  const maxSnapDistance = 2;
  const result = await runKFunction(device, {
    csr: GRID_CSR,
    positions: GRID.positions,
    points: GRID_POINTS,
    networkLength: GRID_NETWORK_LENGTH,
    maxDistance,
    bandCount,
    simulationCount,
    activeSimulations,
    seed,
    maxSnapDistance,
    rowsPerBlock: 17,
    maxIterations: 48
  });
  const observed = snapEvents(GRID_POINTS, GRID.positions, GRID_CSR, maxSnapDistance);
  const validCount = observed.filter(event => event.first < GRID_NODE_COUNT).length;
  expect(validCount).toBeLessThan(GRID_POINTS.length / 2);
  expect(result.snappedEventCount).toBe(validCount);
  expect(result.converged).toBe(1);

  const expectedCounts: number[][] = [];
  for (let pattern = 0; pattern <= simulationCount; pattern++) {
    let events: KFunctionEvent[] = [];
    if (pattern === 0) {
      events = observed;
    } else if (pattern <= activeSimulations) {
      events = simulateEvents(
        GRID_CSR,
        GRID_NODE_COUNT,
        GRID_NETWORK_LENGTH,
        seed,
        pattern,
        GRID_POINTS.length / 2,
        validCount
      );
    }
    expectedCounts.push(
      pattern <= activeSimulations
        ? getPairCounts(GRID_CSR, GRID_NODE_COUNT, events, maxDistance, bandCount)
        : new Array<number>(bandCount).fill(0)
    );
  }
  const expectedK = expectedCounts.map((counts, pattern) =>
    pattern <= activeSimulations ? getKValues(counts, validCount, GRID_NETWORK_LENGTH) : counts
  );
  expect(result.pairCounts).toEqual(expectedCounts.flat());
  result.kValues.forEach((value, index) => {
    expect(value).toBeCloseTo(expectedK.flat()[index], 3);
  });
  // The last threshold counts more pairs than the first; simulations differ from the observed.
  expect(expectedCounts[0][bandCount - 1]).toBeGreaterThan(0);
  expect(expectedCounts[1]).not.toEqual(expectedCounts[0]);
  // Envelope: min, mean and max over the active simulations per band.
  const envelope = result.envelope!;
  for (let band = 0; band < bandCount; band++) {
    const values = [1, 2, 3].map(pattern => expectedK[pattern][band]);
    expect(envelope[band]).toBeCloseTo(Math.min(...values), 3);
    expect(envelope[bandCount + band]).toBeCloseTo(
      values.reduce((sum, value) => sum + value, 0) / values.length,
      3
    );
    expect(envelope[2 * bandCount + band]).toBeCloseTo(Math.max(...values), 3);
  }
});

it('GPUNetworkKFunction results do not depend on the block size', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const base = {
    csr: GRID_CSR,
    positions: GRID.positions,
    points: GRID_POINTS,
    networkLength: GRID_NETWORK_LENGTH,
    maxDistance: 10,
    bandCount: 6,
    simulationCount: 2,
    maxIterations: 48
  };
  const small = await runKFunction(device, {...base, rowsPerBlock: 5});
  const large = await runKFunction(device, {...base, rowsPerBlock: 72});
  const lanes = await runKFunction(device, {...base, rowsPerBlock: 11, laneCount: 4});
  expect(small.pairCounts).toEqual(large.pairCounts);
  expect(small.kValues).toEqual(large.kValues);
  expect(lanes.pairCounts).toEqual(large.pairCounts);
  expect(lanes.kValues).toEqual(large.kValues);
});
