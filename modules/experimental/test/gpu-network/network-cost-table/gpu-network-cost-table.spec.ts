// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUNetworkCostTable} from '../../../src/gpu-network/network-cost-table/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  buildCSR,
  createGridNetwork,
  createRandomNetwork,
  dijkstra,
  type NetworkEdge
} from '../network-reachability/network-reachability-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

type TableRun = {
  offsets: number[];
  targets: number[];
  costs: number[];
  overflow: number;
  requiredCount: number;
  converged: number;
};

/** Builds, encodes once and reads back one cost table. */
async function runCostTable(
  device: Device,
  edges: readonly NetworkEdge[],
  nodeCount: number,
  options: {costLimit: number; capacity: number; laneCount?: number}
): Promise<TableRun> {
  const csr = buildCSR(nodeCount, edges);
  const graph = new GPUCommandGraph(device, {id: 'cost-table'});
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
  const outputBuffers: Record<string, Buffer> = {};
  const output = <Format extends 'uint32' | 'float32'>(
    id: string,
    format: Format,
    length: number
  ) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    outputBuffers[id] = buffer;
    return importGraphBuffer(graph, id, buffer, format, length);
  };
  graph.add(
    new GPUNetworkCostTable({
      id: 'table',
      topology: {
        offsets: input('offsets', csr.offsets, 'uint32'),
        neighbors: input('neighbors', csr.neighbors, 'uint32'),
        weights: input('weights', csr.weights, 'float32')
      },
      costLimit: input('cost-limit', Float32Array.of(options.costLimit), 'float32'),
      laneCount: options.laneCount,
      output: {
        offsets: output('table-offsets', 'uint32', nodeCount + 1),
        targets: output('table-targets', 'uint32', options.capacity),
        costs: output('table-costs', 'float32', options.capacity),
        status: {
          overflow: output('overflow', 'uint32', 1),
          requiredCount: output('required-count', 'uint32', 1),
          converged: output('converged', 'uint32', 1)
        }
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const run: TableRun = {
    offsets: await readUint32(outputBuffers['table-offsets'], nodeCount + 1),
    targets: await readUint32(outputBuffers['table-targets'], options.capacity),
    costs: await readFloat32(outputBuffers['table-costs'], options.capacity),
    overflow: (await readUint32(outputBuffers['overflow'], 1))[0],
    requiredCount: (await readUint32(outputBuffers['required-count'], 1))[0],
    converged: (await readUint32(outputBuffers['converged'], 1))[0]
  };
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return run;
}

/** CPU Dijkstra with a cutoff from every source: the rows of the complete table. */
function getExpectedRows(edges: readonly NetworkEdge[], nodeCount: number, costLimit: number) {
  const csr = buildCSR(nodeCount, edges);
  return Array.from({length: nodeCount}, (_, source) => {
    const distances = dijkstra(csr, nodeCount, [{node: source, cost: 0}], costLimit);
    const targets: number[] = [];
    const costs: number[] = [];
    distances.forEach((cost, node) => {
      if (Number.isFinite(cost)) {
        targets.push(node);
        costs.push(cost);
      }
    });
    return {targets, costs};
  });
}

function expectCompleteTable(
  run: TableRun,
  edges: readonly NetworkEdge[],
  nodeCount: number,
  costLimit: number
): number {
  const rows = getExpectedRows(edges, nodeCount, costLimit);
  const total = rows.reduce((sum, row) => sum + row.targets.length, 0);
  expect(run.overflow).toBe(0);
  expect(run.converged).toBe(1);
  expect(run.requiredCount).toBe(total);
  expect(run.offsets[nodeCount]).toBe(total);
  rows.forEach((row, source) => {
    const start = run.offsets[source];
    const end = run.offsets[source + 1];
    // Integer weights make every f32 sum exact, so costs match bit for bit.
    expect(run.targets.slice(start, end), `row ${source} targets`).toEqual(row.targets);
    expect(run.costs.slice(start, end), `row ${source} costs`).toEqual(row.costs);
  });
  return total;
}

const GRID_SIZE = 12;
const GRID = createGridNetwork(7, GRID_SIZE, GRID_SIZE);
const RANDOM_NODES = 150;
const RANDOM = createRandomNetwork(3, RANDOM_NODES, 600);

it('GPUNetworkCostTable matches CPU Dijkstra with a cutoff on a grid', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const nodeCount = GRID_SIZE * GRID_SIZE;
  // 144 nodes in lanes of 20: seven full batches and a short one.
  for (const laneCount of [20, undefined]) {
    const run = await runCostTable(device, GRID, nodeCount, {
      costLimit: 30,
      capacity: nodeCount * nodeCount,
      laneCount
    });
    expectCompleteTable(run, GRID, nodeCount, 30);
  }
});

it('GPUNetworkCostTable matches CPU Dijkstra with a cutoff on a random directed network', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const costLimit of [4, 12, 1000]) {
    const run = await runCostTable(device, RANDOM, RANDOM_NODES, {
      costLimit,
      capacity: RANDOM_NODES * RANDOM_NODES,
      laneCount: 64
    });
    expectCompleteTable(run, RANDOM, RANDOM_NODES, costLimit);
  }
});

it('GPUNetworkCostTable truncates consistently and reports the required count on overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const nodeCount = GRID_SIZE * GRID_SIZE;
  const rows = getExpectedRows(GRID, nodeCount, 30);
  const total = rows.reduce((sum, row) => sum + row.targets.length, 0);
  const capacity = Math.floor(total / 2);
  const run = await runCostTable(device, GRID, nodeCount, {costLimit: 30, capacity, laneCount: 20});
  expect(run.overflow).toBe(1);
  expect(run.requiredCount).toBe(total);
  // The offsets stay a consistent CSR clamped to the capacity.
  expect(run.offsets[nodeCount]).toBe(capacity);
  for (let node = 0; node < nodeCount; node++) {
    expect(run.offsets[node + 1]).toBeGreaterThanOrEqual(run.offsets[node]);
    expect(run.offsets[node + 1]).toBeLessThanOrEqual(capacity);
  }
  // Every stored entry is the same as the complete table's entry at that position.
  const expectedTargets = rows.flatMap(row => row.targets);
  const expectedCosts = rows.flatMap(row => row.costs);
  expect(run.targets).toEqual(expectedTargets.slice(0, capacity));
  expect(run.costs).toEqual(expectedCosts.slice(0, capacity));
});
