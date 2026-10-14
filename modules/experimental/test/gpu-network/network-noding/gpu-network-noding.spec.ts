// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUNetworkNoding} from '../../../src/gpu-network/network-noding/index';
import {GPUNetworkReachability} from '../../../src/gpu-network/network-reachability/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {flattenLines, type Point} from '../../gpu-spatial-analysis/line-split/line-split-harness';

type NodingResult = {
  nodeCount: number;
  totalNodeCount: number;
  overflow: number;
  edgeCount: number;
  nodes: Point[];
  from: number[];
  to: number[];
  lengths: number[];
  offsets: number[];
  neighbors: number[];
  weights: number[];
  edgeIds: number[];
  costs?: number[];
};

async function runNoding(
  device: Device,
  lines: readonly (readonly Point[])[],
  options: {tolerance?: number; nodeCapacity?: number; source?: number} = {}
): Promise<NodingResult> {
  const graph = new GPUCommandGraph(device, {id: 'network-noding-test'});
  const buffers: Buffer[] = [];
  const arrays = flattenLines(lines);
  const input = (
    name: string,
    data: Float32Array | Uint32Array,
    format: 'float32x2' | 'uint32'
  ) => {
    const buffer = createInputBuffer(device, data);
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      name,
      buffer,
      format,
      format === 'uint32' ? data.length : data.length / 2
    ) as never;
  };
  const output = (
    name: string,
    length: number,
    format: 'uint32' | 'float32' | 'float32x2' = 'uint32'
  ) => {
    const buffer = createOutputBuffer(device, format === 'float32x2' ? length * 2 : length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length) as never};
  };
  const pieceCapacity = 64;
  const nodeCapacity = options.nodeCapacity ?? 64;
  const tolerance = output('tolerance', 1, 'float32');
  tolerance.buffer.write(new Float32Array([options.tolerance ?? 0]));
  const pieceLines = output('piece-lines', pieceCapacity);
  const pieceOffsets = output('piece-offsets', pieceCapacity + 1);
  const piecePositions = output('piece-positions', 512, 'float32x2');
  const pieceCount = output('piece-count', 1);
  const pieceOverflow = output('piece-overflow', 1);
  const nodePositions = output('node-positions', nodeCapacity, 'float32x2');
  const nodeCount = output('node-count', 1);
  const totalNodeCount = output('total-node-count', 1);
  const from = output('from', pieceCapacity);
  const to = output('to', pieceCapacity);
  const lengths = output('lengths', pieceCapacity, 'float32');
  const offsets = output('offsets', nodeCapacity + 1);
  const neighbors = output('neighbors', pieceCapacity * 2);
  const weights = output('weights', pieceCapacity * 2, 'float32');
  const edgeIds = output('edge-ids', pieceCapacity * 2);
  const overflow = output('overflow', 1);
  graph.add(
    new GPUNetworkNoding({
      lines: {
        kind: 'lines',
        positions: input('positions', arrays.positions, 'float32x2'),
        lineOffsets: input('line-offsets', arrays.lineOffsets, 'uint32')
      },
      intersectionCapacity: 256,
      tolerance: tolerance.view,
      pieces: {
        geometry: {
          kind: 'lines',
          positions: piecePositions.view,
          lineOffsets: pieceOffsets.view
        },
        sourceIds: pieceLines.view,
        status: {count: pieceCount.view, overflow: pieceOverflow.view}
      },
      nodes: {
        positions: nodePositions.view,
        count: nodeCount.view,
        requiredCount: totalNodeCount.view
      },
      edges: {fromNodes: from.view, toNodes: to.view, lengths: lengths.view},
      csr: {
        offsets: offsets.view,
        neighbors: neighbors.view,
        weights: weights.view,
        edgeIds: edgeIds.view
      },
      overflow: overflow.view
    })
  );
  let costs: ReturnType<typeof output> | undefined;
  if (options.source !== undefined) {
    // Prove the CSR routes: shortest path costs from one node.
    costs = output('costs', nodeCapacity, 'float32');
    const sources = output('sources', 1);
    sources.buffer.write(new Uint32Array([options.source]));
    graph.add(
      new GPUNetworkReachability({
        offsets: offsets.view,
        neighbors: neighbors.view,
        weights: weights.view,
        sources: sources.view,
        costs: costs.view
      })
    );
  }
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const edgeCount = (await readUint32(pieceCount.buffer, 1))[0];
  const nodesTotal = (await readUint32(nodeCount.buffer, 1))[0];
  const positions = await readFloat32(nodePositions.buffer, nodeCapacity * 2);
  const result: NodingResult = {
    nodeCount: nodesTotal,
    totalNodeCount: (await readUint32(totalNodeCount.buffer, 1))[0],
    overflow: (await readUint32(overflow.buffer, 1))[0],
    edgeCount,
    nodes: Array.from(
      {length: nodesTotal},
      (_, node): Point => [positions[node * 2], positions[node * 2 + 1]]
    ),
    from: (await readUint32(from.buffer, edgeCount)).slice(0, edgeCount),
    to: (await readUint32(to.buffer, edgeCount)).slice(0, edgeCount),
    lengths: (await readFloat32(lengths.buffer, edgeCount)).slice(0, edgeCount),
    offsets: await readUint32(offsets.buffer, nodeCapacity + 1),
    neighbors: await readUint32(neighbors.buffer, edgeCount * 2),
    weights: await readFloat32(weights.buffer, edgeCount * 2),
    edgeIds: await readUint32(edgeIds.buffer, edgeCount * 2)
  };
  if (costs) {
    result.costs = await readFloat32(costs.buffer, nodeCapacity);
  }
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

const GRID: Point[][] = [
  // Two horizontal and two vertical roads crossing in a 2 x 2 grid, plus a dangling spur.
  [
    [0, 0],
    [10, 0]
  ],
  [
    [0, 10],
    [10, 10]
  ],
  [
    [0, -5],
    [0, 15]
  ],
  [
    [10, -5],
    [10, 15]
  ],
  [
    [10, 10],
    [20, 10]
  ]
];

it('GPUNetworkNoding#nodes a grid: node and edge counts and CSR symmetry', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const result = await runNoding(device, GRID);
  // Nodes: 4 grid crossings, 4 road ends (0,-5) (0,15) (10,-5) (10,15), spur end (20,10).
  expect(result.nodeCount).toBe(9);
  expect(result.totalNodeCount).toBe(9);
  // Edges: each horizontal 1, each vertical 3, spur 1.
  expect(result.edgeCount).toBe(1 + 1 + 3 + 3 + 1);
  expect(result.overflow).toBe(0);
  expect(result.offsets[result.nodeCount]).toBe(result.edgeCount * 2);
  expect(result.offsets[63]).toBe(result.edgeCount * 2);
  // Degrees: crossings of (0,0) and (10,0): 3 edges at (10,0) (left, up, down), 3 at (0,0).
  const degree = (node: number) => result.offsets[node + 1] - result.offsets[node];
  const degrees = Array.from({length: result.nodeCount}, (_, node) => degree(node)).sort();
  expect(degrees).toEqual([1, 1, 1, 1, 1, 3, 3, 3, 4]);
  // Every entry has its mirror and the same weight.
  for (let node = 0; node < result.nodeCount; node++) {
    for (let entry = result.offsets[node]; entry < result.offsets[node + 1]; entry++) {
      const neighbor = result.neighbors[entry];
      const edge = result.edgeIds[entry];
      expect([result.from[edge], result.to[edge]].sort()).toEqual([node, neighbor].sort());
      expect(result.weights[entry]).toBeCloseTo(result.lengths[edge], 5);
    }
  }
  const total = result.lengths.reduce((sum, length) => sum + length, 0);
  // Input length: 10 + 10 + 20 + 20 + 10 = 70.
  expect(total).toBeCloseTo(70, 4);
});

it('GPUNetworkNoding#snaps nearby end points within the tolerance and routes over the result', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  // A path of three roads whose joints are 0.04 apart.
  const lines: Point[][] = [
    [
      [0, 0],
      [10, 0]
    ],
    [
      [10.04, 0.01],
      [20, 0]
    ],
    [
      [20.03, 0.0],
      [30, 0]
    ]
  ];
  const exact = await runNoding(device, lines, {tolerance: 0});
  expect(exact.nodeCount).toBe(6);
  const snapped = await runNoding(device, lines, {tolerance: 1, source: 0});
  expect(snapped.nodeCount).toBe(4);
  expect(snapped.edgeCount).toBe(3);
  const costs = (snapped.costs as number[]).slice(0, 4).sort((a, b) => a - b);
  expect(costs[0]).toBe(0);
  expect(costs[3]).toBeGreaterThan(29.9);
  expect(costs[3]).toBeLessThan(30.1);
  expect(costs.every(Number.isFinite)).toBe(true);
});

it('GPUNetworkNoding#reports node overflow and drops edges at nodes beyond the capacity', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const result = await runNoding(device, GRID, {nodeCapacity: 5});
  expect(result.totalNodeCount).toBe(9);
  expect(result.nodeCount).toBe(5);
  expect(result.overflow).toBe(1);
  expect(result.offsets[5]).toBeLessThan(result.edgeCount * 2);
});

/**
 * Three disjoint roads in UTM coordinates (easting about 500 000 m, northing about 4.6e6 m). The
 * end of A and the start of B are 0.25 m apart; the start of A and the start of C share an easting
 * and are 10 m apart in northing. All values are exact in f32 (ulp 0.03125 m at the eastings,
 * 0.5 m at the northings).
 */
const UTM_LINES: Point[][] = [
  [
    [500000, 4600000],
    [500100, 4600000]
  ],
  [
    [500100.25, 4600000],
    [500100.25, 4600100]
  ],
  [
    [500000, 4600010],
    [499900, 4600010]
  ]
];

it('GPUNetworkNoding#snaps UTM-scale end points in exact cells relative to the end point origin', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  // 0.5 m: origin (499900, 4600000); A's end and B's start both fall in x cell floor(200 / 0.5)
  // = floor(200.25 / 0.5) = 400 and y cell 0, so they share one node: 6 end points, 5 nodes.
  const snapped = await runNoding(device, UTM_LINES, {tolerance: 0.5});
  expect(snapped.edgeCount).toBe(3);
  expect(snapped.nodeCount).toBe(5);
  expect(snapped.overflow).toBe(0);
  // The shared node keeps one of the two end points as its position (the first in sort order).
  const gapEnds = snapped.nodes.filter(
    ([x, y]) => (x === 500100 || x === 500100.25) && y === 4600000
  );
  expect(gapEnds).toHaveLength(1);
  // Some node is used by two edges: the merged end of A and start of B.
  const nodeUses = [...snapped.from, ...snapped.to];
  expect(nodeUses.some(node => nodeUses.filter(other => other === node).length === 2)).toBe(true);
  // 1 mm: the 0.25 m gap and the 10 m northing gap stay separate. The former absolute f32 key
  // clamped 4.6e6 / 0.001 cells to 1e9, so every northing shared one key and A's start merged
  // with C's start.
  const exact = await runNoding(device, UTM_LINES, {tolerance: 0.001});
  expect(exact.nodeCount).toBe(6);
  expect(exact.overflow).toBe(0);
  // 0.01 mm: 200 m of eastings is 2e7 cells, beyond the 2^24 exact cells, so overflow is raised.
  const tooFine = await runNoding(device, UTM_LINES, {tolerance: 0.00001});
  expect(tooFine.overflow).toBe(1);
});
