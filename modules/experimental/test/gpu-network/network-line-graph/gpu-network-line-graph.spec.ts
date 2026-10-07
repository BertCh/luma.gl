// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUNetworkReachability} from '../../../src/gpu-network/network-reachability/index';
import {
  getGPUNetworkLineGraphParameterValues,
  GPU_NETWORK_LINE_GRAPH_PARAMETER_LENGTH,
  GPUNetworkLineGraph,
  type GPUNetworkLineGraphSettings
} from '../../../src/gpu-network/network-line-graph/index';
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
  dijkstra,
  type AccessibilityEdge
} from '../network-accessibility/network-accessibility-oracle';
import {lineGraphOracle} from './network-line-graph-oracle';

const WIDTH = 5;
const HEIGHT = 4;
const NODE_COUNT = WIDTH * HEIGHT;
const GRID = createGridFixture(7, WIDTH, HEIGHT);
// Make a few roads one-way by dropping one direction, and close one edge with a negative weight.
const EDGES: AccessibilityEdge[] = GRID.edges
  .filter((_, index) => index % 9 !== 4)
  .map((edge, index) => (index === 11 ? ([edge[0], edge[1], -1] as const) : edge));
const BASE = buildCSR(NODE_COUNT, EDGES);
const EDGE_COUNT = BASE.neighbors.length;
const BANNED: [number, number][] = [
  [BASE.offsets[6], BASE.offsets[7]],
  [3, BASE.offsets[BASE.neighbors[3]]],
  [10, BASE.offsets[BASE.neighbors[10]] + 1]
];

type LineGraphResult = {
  offsets: number[];
  neighbors: number[];
  weights: number[];
  arcCount: number;
  overflow: number;
};

async function runLineGraph(
  device: Device,
  settings: GPUNetworkLineGraphSettings,
  arcCapacity: number,
  useBans: boolean
): Promise<LineGraphResult> {
  const graph = new GPUCommandGraph(device, {id: 'line-graph'});
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
  const parameters = new GPUParameterBuffer(device, {
    id: 'turn-parameters',
    format: 'float32',
    length: GPU_NETWORK_LINE_GRAPH_PARAMETER_LENGTH,
    values: getGPUNetworkLineGraphParameterValues({
      ...settings,
      bannedTurnCount: useBans ? BANNED.length : 0
    })
  });
  graph.add(
    new GPUNetworkLineGraph({
      id: 'line',
      offsets: input('offsets', BASE.offsets, 'uint32'),
      neighbors: input('neighbors', BASE.neighbors, 'uint32'),
      weights: input('weights', BASE.weights, 'float32'),
      nodePositions: input('positions', GRID.positions, 'float32x2'),
      parameters: parameters.importToGraph(graph),
      bannedTurns: input('banned', Uint32Array.from(BANNED.flat()), 'uint32'),
      lineOffsets: output('line-offsets', 'uint32', EDGE_COUNT + 1),
      lineNeighbors: output('line-neighbors', 'uint32', arcCapacity),
      lineWeights: output('line-weights', 'float32', arcCapacity),
      arcCount: output('arc-count', 'uint32', 1),
      overflow: output('overflow', 'uint32', 1)
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    offsets: await readUint32(outputs['line-offsets'], EDGE_COUNT + 1),
    neighbors: await readUint32(outputs['line-neighbors'], arcCapacity),
    weights: await readFloat32(outputs['line-weights'], arcCapacity),
    arcCount: (await readUint32(outputs['arc-count'], 1))[0],
    overflow: (await readUint32(outputs['overflow'], 1))[0]
  };
  compiled.destroy();
  parameters.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

function expectMatchesOracle(
  result: LineGraphResult,
  settings: GPUNetworkLineGraphSettings,
  arcCapacity: number,
  useBans: boolean
): ReturnType<typeof lineGraphOracle> {
  const oracle = lineGraphOracle(
    BASE,
    GRID.positions,
    useBans ? BANNED : [],
    settings,
    arcCapacity
  );
  expect(result.offsets).toEqual(Array.from(oracle.csr.offsets));
  const used = result.offsets[EDGE_COUNT];
  expect(result.neighbors.slice(0, used)).toEqual(Array.from(oracle.csr.neighbors.slice(0, used)));
  for (let arc = 0; arc < used; arc++) {
    expect(result.weights[arc]).toBeCloseTo(oracle.csr.weights[arc], 4);
  }
  expect(result.arcCount).toBe(Math.min(oracle.totalArcs, arcCapacity));
  expect(result.overflow).toBe(oracle.totalArcs > arcCapacity ? 1 : 0);
  return oracle;
}

const TURN_SETTINGS: GPUNetworkLineGraphSettings = {
  angleCost: 2,
  leftTurnCost: 1.5,
  rightTurnCost: 0.5,
  uTurnCost: -1
};

it('GPUNetworkLineGraph matches the CPU line graph with turn costs, banned U-turns and bans', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const capacity = EDGE_COUNT * 4;
  const banned = await runLineGraph(device, TURN_SETTINGS, capacity, true);
  const oracle = expectMatchesOracle(banned, TURN_SETTINGS, capacity, true);
  const unbanned = await runLineGraph(device, TURN_SETTINGS, capacity, false);
  expectMatchesOracle(unbanned, TURN_SETTINGS, capacity, false);
  expect(unbanned.arcCount).toBeGreaterThan(banned.arcCount);
  expect(oracle.totalArcs).toBeGreaterThan(EDGE_COUNT);
  const withUTurns: GPUNetworkLineGraphSettings = {...TURN_SETTINGS, uTurnCost: 5};
  const allowed = await runLineGraph(device, withUTurns, capacity, false);
  expectMatchesOracle(allowed, withUTurns, capacity, false);
  expect(allowed.arcCount).toBeGreaterThan(unbanned.arcCount);
});

it('GPUNetworkLineGraph clamps rows and flags overflow past the arc capacity', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const capacity = 40;
  const result = await runLineGraph(device, TURN_SETTINGS, capacity, true);
  expect(result.overflow).toBe(1);
  expect(result.arcCount).toBe(capacity);
  expect(result.offsets[EDGE_COUNT]).toBe(capacity);
  expectMatchesOracle(result, TURN_SETTINGS, capacity, true);
});

it('GPUNetworkReachability over the line graph prices turns like the CPU Dijkstra', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const capacity = EDGE_COUNT * 4;
  const result = await runLineGraph(device, TURN_SETTINGS, capacity, true);
  const oracle = lineGraphOracle(BASE, GRID.positions, BANNED, TURN_SETTINGS, capacity);
  const origin = 0;
  // Seed the out-edges of the origin at their own cost; the cost of line node e is the cost of
  // arriving at head(e) along e with every turn paid.
  const seedEdges = Array.from(
    {length: BASE.offsets[origin + 1] - BASE.offsets[origin]},
    (_, index) => BASE.offsets[origin] + index
  ).filter(edge => BASE.weights[edge] >= 0);
  const seeds = seedEdges.map(edge => ({node: edge, cost: BASE.weights[edge]}));
  const expected = dijkstra({...oracle.csr, sources: new Uint32Array(0)}, EDGE_COUNT, seeds);

  const graph = new GPUCommandGraph(device, {id: 'line-reachability'});
  const buffers: Buffer[] = [];
  const input = <Format extends 'uint32' | 'float32'>(
    id: string,
    values: Float32Array | Uint32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(graph, id, buffer, format, values.length);
  };
  const costsBuffer = createOutputBuffer(device, EDGE_COUNT);
  buffers.push(costsBuffer);
  graph.add(
    new GPUNetworkReachability({
      id: 'line-reach',
      offsets: input('line-offsets', Uint32Array.from(result.offsets), 'uint32'),
      neighbors: input('line-neighbors', Uint32Array.from(result.neighbors), 'uint32'),
      weights: input('line-weights', Float32Array.from(result.weights), 'float32'),
      sources: input('sources', Uint32Array.from(seeds.map(seed => seed.node)), 'uint32'),
      sourceCosts: input(
        'source-costs',
        Float32Array.from(seeds.map(seed => seed.cost)),
        'float32'
      ),
      maxIterations: 64,
      costs: importGraphBuffer(graph, 'costs', costsBuffer, 'float32', EDGE_COUNT)
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const costs = await readFloat32(costsBuffer, EDGE_COUNT);
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  let reached = 0;
  for (let edge = 0; edge < EDGE_COUNT; edge++) {
    if (Number.isFinite(expected[edge])) {
      reached++;
      expect(costs[edge]).toBeCloseTo(expected[edge], 3);
    } else {
      expect(costs[edge]).toBe(Infinity);
    }
  }
  expect(reached).toBeGreaterThan(EDGE_COUNT / 2);
});
