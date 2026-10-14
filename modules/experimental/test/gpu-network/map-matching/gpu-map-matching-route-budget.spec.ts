// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUMapMatching,
  encodeGPUMapMatchingParameters,
  type GPUMapMatchingParameters
} from '../../../src/gpu-network/map-matching/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {buildCSR, createGridFixture} from '../network-accessibility/network-accessibility-oracle';
import {matchOracle} from './map-matching-oracle';

// Review B-02 on the real kernel: a 41 x 41 Manhattan grid with 100 m blocks, one track whose
// samples are 6 blocks east and 8 north apart (1 km straight, 1.4 km by road), one candidate per
// point. The route-search mirror (route-search-mirror.ts) misses all three routes at
// routeNodeBudget 64 after dropping 35, 38 and 38 nodes, and finds them exactly at 128 with no drop.
const WIDTH = 41;
const BLOCK = 100;
const GRID = createGridFixture(3, WIDTH, WIDTH);
const POSITIONS = GRID.positions.map(value => value * BLOCK);
const CSR = buildCSR(WIDTH * WIDTH, GRID.edges);
const POINTS = Float32Array.from(
  [0, 1, 2, 3].flatMap(step => [(2 + 6 * step) * BLOCK + 3, (2 + 8 * step) * BLOCK - 4])
);
const TRACK_OFFSETS = [0, 4];
const PARAMETERS: GPUMapMatchingParameters = {
  sigma: 10,
  beta: 50,
  searchRadius: 55,
  routeFactor: 3,
  routeSlack: 50
};

type BudgetResult = {breaks: number[]; overflow: number; routeOverflow: number; edges: number[]};

async function runBudget(device: Device, routeNodeBudget: number): Promise<BudgetResult> {
  const graph = new GPUCommandGraph(device, {id: 'map-matching-budget'});
  const buffers: Buffer[] = [];
  const input = <Format extends 'uint32' | 'float32x2'>(
    id: string,
    values: Float32Array | Uint32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    const length = format === 'float32x2' ? values.length / 2 : values.length;
    return importGraphBuffer(graph, id, buffer, format, length);
  };
  const output = (id: string, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, id, buffer, 'uint32', length)};
  };
  const parameters = new GPUParameterBuffer(device, {
    id: 'map-matching-budget-parameters',
    format: 'float32',
    length: 8,
    values: encodeGPUMapMatchingParameters(PARAMETERS)
  });
  const edges = output('edges', 4);
  const breaks = output('breaks', 4);
  const overflow = output('overflow', 1);
  const routeOverflow = output('route-overflow', 1);
  graph.add(
    new GPUMapMatching({
      id: 'budget',
      points: input('points', POINTS, 'float32x2'),
      trackOffsets: input('track-offsets', Uint32Array.from(TRACK_OFFSETS), 'uint32'),
      nodePositions: input('nodes', POSITIONS, 'float32x2'),
      offsets: input('offsets', CSR.offsets, 'uint32'),
      edgeTargets: input('targets', CSR.neighbors, 'uint32'),
      parameters: parameters.importToGraph(graph),
      candidateCount: 1,
      cellSize: 60,
      bounds: {minimum: [-100, -100], maximum: [4100, 4100]},
      routeNodeBudget,
      output: {
        matchedEdges: edges.view,
        breaks: breaks.view,
        overflow: overflow.view,
        routeOverflow: routeOverflow.view
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    edges: await readUint32(edges.buffer, 4),
    breaks: await readUint32(breaks.buffer, 4),
    overflow: (await readUint32(overflow.buffer, 1))[0],
    routeOverflow: (await readUint32(routeOverflow.buffer, 1))[0]
  };
  compiled.destroy();
  parameters.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

it('GPUMapMatching flags budget breaks and counts dropped route nodes', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const expected = matchOracle(
    POINTS,
    TRACK_OFFSETS,
    POSITIONS,
    CSR,
    {...PARAMETERS, routeFactor: 3, routeSlack: 50},
    1,
    60
  );
  // Exact Dijkstra finds every 1.4 km route: no break in the model.
  expect(expected.breaks).toEqual([0, 0, 0, 0]);

  // Budget 64: the routes are missed, and the breaks say so instead of posing as HMM breaks.
  const tight = await runBudget(device, 64);
  expect(tight.breaks).toEqual([0, 2, 2, 2]);
  expect(tight.overflow).toBe(1);
  expect(tight.routeOverflow).toBeGreaterThan(0);

  // Budget 128 covers the spacing: the oracle's answer, with nothing dropped.
  const ample = await runBudget(device, 128);
  expect(ample.breaks).toEqual(expected.breaks);
  expect(ample.edges).toEqual(expected.edges);
  expect(ample.overflow).toBe(0);
  expect(ample.routeOverflow).toBe(0);
});

it('GPUMapMatching ignores CSR rows past offsets[nodeCount]', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Review B-04: a CSR shorter than edgeTargets. Before the fix the per-node kernel never wrote the
  // tail rows' edge-table words, so they kept zeros (a zero-length edge at node 0) or an aliased
  // transient's words (phantom edges). Now they are invalid and never become candidates.
  const width = 3;
  const grid = createGridFixture(1, width, width);
  const positions = grid.positions.map(value => value * BLOCK);
  const csr = buildCSR(width * width, grid.edges);
  const tailTargets = [8, 5, 7, 4];
  const targets = Uint32Array.from([...csr.neighbors, ...tailTargets]);
  // Two block centers, 50 m from every real road (beyond the 40 m radius) but on the line from
  // node 0 to the tail target 8, then a point 1 m from node 0 where a zero-length tail edge would
  // compete with the real roads.
  const points = Float32Array.of(50, 50, 150, 150, 1, 0.5);
  const graph = new GPUCommandGraph(device, {id: 'map-matching-tail'});
  const buffers: Buffer[] = [];
  const input = <Format extends 'uint32' | 'float32x2'>(
    id: string,
    values: Float32Array | Uint32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    const length = format === 'float32x2' ? values.length / 2 : values.length;
    return importGraphBuffer(graph, id, buffer, format, length);
  };
  const edgesBuffer = createOutputBuffer(device, 3);
  buffers.push(edgesBuffer);
  const parameters = new GPUParameterBuffer(device, {
    id: 'map-matching-tail-parameters',
    format: 'float32',
    length: 8,
    values: encodeGPUMapMatchingParameters({...PARAMETERS, searchRadius: 40})
  });
  graph.add(
    new GPUMapMatching({
      id: 'tail',
      points: input('points', points, 'float32x2'),
      trackOffsets: input('track-offsets', Uint32Array.of(0, 3), 'uint32'),
      nodePositions: input('nodes', positions, 'float32x2'),
      offsets: input('offsets', csr.offsets, 'uint32'),
      edgeTargets: input('targets', targets, 'uint32'),
      parameters: parameters.importToGraph(graph),
      cellSize: 50,
      bounds: {minimum: [-50, -50], maximum: [250, 250]},
      candidateCount: 8,
      output: {matchedEdges: importGraphBuffer(graph, 'edges', edgesBuffer, 'uint32', 3)}
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const matched = await readUint32(edgesBuffer, 3);
  compiled.destroy();
  parameters.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  expect(matched.slice(0, 2)).toEqual([0xffffffff, 0xffffffff]);
  // The point by node 0 matches the road along y = 0 (0.5 m away), a live row.
  expect(matched[2]).toBeLessThan(csr.neighbors.length);
  expect(
    positions[csr.neighbors[matched[2]] * 2 + 1] + positions[csr.sources[matched[2]] * 2 + 1]
  ).toBe(0);
});
