// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUNetworkNeighborhood} from '../../../src/gpu-network/network-analysis/gpu-network-neighborhood';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  buildCSR,
  createRandomNetwork,
  F1_EDGES,
  NONE,
  type NetworkEdge
} from '../network-reachability/network-reachability-oracle';
import {neighborhoodOracle} from './network-neighborhood-oracle';

type FixtureOptions = {
  edges: readonly NetworkEdge[];
  nodeCount: number;
  seeds: number[];
  seedCount?: number;
  hops: number;
  maxHops?: number;
  nodeIds?: number[];
  edgeIds?: number[];
  nodeCapacity?: number;
  edgeCapacity?: number;
};

type Fixture = {
  compiled: CompiledGPUCommandGraph;
  seeds: GPUParameterBuffer<'uint32'>;
  seedCount?: GPUParameterBuffer<'uint32'>;
  hops: GPUParameterBuffer<'uint32'>;
  nodeIds?: GPUParameterBuffer<'uint32'>;
  edgeIds?: GPUParameterBuffer<'uint32'>;
  buffers: {
    hopDistances: Buffer;
    nodeMask: Buffer;
    edgeMask: Buffer;
    nodeIds: Buffer;
    nodeCount: Buffer;
    nodeOverflow: Buffer;
    edgeIds: Buffer;
    edgeCount: Buffer;
    edgeOverflow: Buffer;
  };
  owned: Buffer[];
  parameters: GPUParameterBuffer[];
  destroy(): void;
};

function createFixture(device: Device, options: FixtureOptions, id: string): Fixture {
  const {nodeCount} = options;
  const csr = buildCSR(nodeCount, options.edges);
  const edgeCount = csr.neighbors.length;
  const nodeCapacity = options.nodeCapacity ?? nodeCount;
  const edgeCapacity = options.edgeCapacity ?? Math.max(edgeCount, 1);
  const graph = new GPUCommandGraph(device, {id});
  const owned: Buffer[] = [];
  const parameters: GPUParameterBuffer[] = [];
  const input = (name: string, values: Uint32Array, length: number) => {
    const buffer = createInputBuffer(device, values.length ? values : new Uint32Array(1));
    owned.push(buffer);
    return importGraphBuffer(graph, `${id}-${name}`, buffer, 'uint32', length);
  };
  const output = (name: string, length: number) => {
    const buffer = createOutputBuffer(device, length);
    owned.push(buffer);
    return {buffer, view: importGraphBuffer(graph, `${id}-${name}`, buffer, 'uint32', length)};
  };
  const parameter = (name: string, values: number[]) => {
    const buffer = new GPUParameterBuffer(device, {
      id: `${id}-${name}`,
      format: 'uint32',
      length: values.length,
      values: Uint32Array.from(values)
    });
    parameters.push(buffer);
    return buffer;
  };
  const seeds = parameter('seeds', options.seeds);
  const seedCount =
    options.seedCount !== undefined ? parameter('seed-count', [options.seedCount]) : undefined;
  const hops = parameter('hops', [options.hops]);
  const nodeIds = options.nodeIds ? parameter('node-ids', options.nodeIds) : undefined;
  const edgeIds = options.edgeIds ? parameter('edge-ids', options.edgeIds) : undefined;
  const hopDistances = output('hop-distances', nodeCount);
  const nodeMask = output('node-mask', nodeCount);
  const edgeMask = output('edge-mask', edgeCount);
  const nodeResultIds = output('node-result-ids', nodeCapacity);
  const nodeResultCount = output('node-result-count', 1);
  const nodeResultOverflow = output('node-result-overflow', 1);
  const edgeResultIds = output('edge-result-ids', edgeCapacity);
  const edgeResultCount = output('edge-result-count', 1);
  const edgeResultOverflow = output('edge-result-overflow', 1);
  graph.add(
    new GPUNetworkNeighborhood({
      id,
      offsets: input('offsets', csr.offsets, nodeCount + 1),
      neighbors: input('neighbors', csr.neighbors, edgeCount),
      seeds: seeds.importToGraph(graph),
      seedCount: seedCount?.importToGraph(graph),
      hops: hops.importToGraph(graph),
      maxHops: options.maxHops,
      nodeIds: nodeIds?.importToGraph(graph),
      edgeIds: edgeIds?.importToGraph(graph),
      hopDistances: hopDistances.view,
      nodeMask: nodeMask.view,
      edgeMask: edgeMask.view,
      nodes: {
        ids: nodeResultIds.view,
        count: nodeResultCount.view,
        overflow: nodeResultOverflow.view
      },
      edges: {
        ids: edgeResultIds.view,
        count: edgeResultCount.view,
        overflow: edgeResultOverflow.view
      }
    })
  );
  const compiled = graph.compile();
  return {
    compiled,
    seeds,
    seedCount,
    hops,
    nodeIds,
    edgeIds,
    buffers: {
      hopDistances: hopDistances.buffer,
      nodeMask: nodeMask.buffer,
      edgeMask: edgeMask.buffer,
      nodeIds: nodeResultIds.buffer,
      nodeCount: nodeResultCount.buffer,
      nodeOverflow: nodeResultOverflow.buffer,
      edgeIds: edgeResultIds.buffer,
      edgeCount: edgeResultCount.buffer,
      edgeOverflow: edgeResultOverflow.buffer
    },
    owned,
    parameters,
    destroy() {
      compiled.destroy();
      for (const buffer of owned) {
        buffer.destroy();
      }
      for (const buffer of parameters) {
        buffer.destroy();
      }
    }
  };
}

/** Encodes the compiled graph and compares every output against the oracle. */
async function expectMatchesOracle(
  device: Device,
  fixture: Fixture,
  options: FixtureOptions
): Promise<void> {
  const csr = buildCSR(options.nodeCount, options.edges);
  const expected = neighborhoodOracle({
    csr,
    nodeCount: options.nodeCount,
    seeds: options.seeds,
    seedCount: options.seedCount,
    hops: options.hops,
    maxHops: options.maxHops ?? 8,
    nodeIds: options.nodeIds,
    edgeIds: options.edgeIds,
    nodeCapacity: options.nodeCapacity ?? options.nodeCount,
    edgeCapacity: options.edgeCapacity ?? Math.max(csr.neighbors.length, 1)
  });
  submitGraph(device, fixture.compiled, undefined);
  const {buffers} = fixture;
  expect(await readUint32(buffers.hopDistances, options.nodeCount)).toEqual(expected.hopDistances);
  expect(await readUint32(buffers.nodeMask, options.nodeCount)).toEqual(expected.nodeMask);
  expect(await readUint32(buffers.edgeMask, csr.neighbors.length)).toEqual(expected.edgeMask);
  const [nodeCount] = await readUint32(buffers.nodeCount, 1);
  expect(nodeCount).toBe(expected.nodeCount);
  expect(await readUint32(buffers.nodeIds, nodeCount)).toEqual(expected.nodeIds);
  expect(await readUint32(buffers.nodeOverflow, 1)).toEqual([expected.nodeOverflow]);
  const [edgeCount] = await readUint32(buffers.edgeCount, 1);
  expect(edgeCount).toBe(expected.edgeCount);
  expect(await readUint32(buffers.edgeIds, edgeCount)).toEqual(expected.edgeIds);
  expect(await readUint32(buffers.edgeOverflow, 1)).toEqual([expected.edgeOverflow]);
}

it('GPUNetworkNeighborhood matches the oracle on the F1 fixture across hop radii', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const options: FixtureOptions = {edges: F1_EDGES, nodeCount: 8, seeds: [0], hops: 0};
  const fixture = createFixture(device, options, 'hood-f1');
  // Re-encode the SAME compiled graph with different per-frame radii; 10 clamps to maxHops 8.
  for (const hops of [0, 1, 2, 10, 1]) {
    fixture.hops.write(Uint32Array.of(hops));
    await expectMatchesOracle(device, fixture, {...options, hops});
  }
  expect(await readUint32(fixture.buffers.hopDistances, 8)).toEqual(
    neighborhoodOracle({
      csr: buildCSR(8, F1_EDGES),
      nodeCount: 8,
      seeds: [0],
      hops: 1,
      maxHops: 8
    }).hopDistances
  );
  fixture.destroy();
});

it('GPUNetworkNeighborhood matches the oracle on a random network with 3 seeds', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const nodeCount = 2000;
  const edges = createRandomNetwork(7, nodeCount, 5000);
  const options: FixtureOptions = {
    edges,
    nodeCount,
    seeds: [3, 977, 1500],
    hops: 3,
    nodeCapacity: nodeCount,
    edgeCapacity: edges.length
  };
  const fixture = createFixture(device, options, 'hood-random');
  await expectMatchesOracle(device, fixture, options);
  expect(
    neighborhoodOracle({
      csr: buildCSR(nodeCount, edges),
      nodeCount,
      seeds: options.seeds,
      hops: 3,
      maxHops: 8
    }).nodeTotal
  ).toBeGreaterThan(3);
  fixture.destroy();
});

it('GPUNetworkNeighborhood handles seedCount changes and seed moves without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const options: FixtureOptions = {
    edges: F1_EDGES,
    nodeCount: 8,
    seeds: [0, 5, 99],
    seedCount: 3,
    hops: 2
  };
  const fixture = createFixture(device, options, 'hood-seeds');
  await expectMatchesOracle(device, fixture, options);
  for (const seedCount of [1, 2, 0, 3]) {
    fixture.seedCount!.write(Uint32Array.of(seedCount));
    await expectMatchesOracle(device, fixture, {...options, seedCount});
  }
  // Move the seeds: out-of-range entries are ignored.
  fixture.seeds.write(Uint32Array.of(4, 100, 6));
  await expectMatchesOracle(device, fixture, {...options, seeds: [4, 100, 6], seedCount: 3});
  fixture.destroy();
});

it('GPUNetworkNeighborhood clamps counts and flags overflow when capacity is too small', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const options: FixtureOptions = {
    edges: F1_EDGES,
    nodeCount: 8,
    seeds: [0],
    hops: 8,
    nodeCapacity: 2,
    edgeCapacity: 3
  };
  const fixture = createFixture(device, options, 'hood-overflow');
  await expectMatchesOracle(device, fixture, options);
  expect(await readUint32(fixture.buffers.nodeCount, 1)).toEqual([2]);
  expect(await readUint32(fixture.buffers.nodeOverflow, 1)).toEqual([1]);
  expect(await readUint32(fixture.buffers.edgeCount, 1)).toEqual([3]);
  expect(await readUint32(fixture.buffers.edgeOverflow, 1)).toEqual([1]);
  fixture.destroy();
});

it('GPUNetworkNeighborhood clamps hops above maxHops', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Path 0 -> 1 -> 2 -> 3 -> 4 -> 5.
  const edges: NetworkEdge[] = [0, 1, 2, 3, 4].map(node => [node, node + 1, 1]);
  const options: FixtureOptions = {edges, nodeCount: 6, seeds: [0], hops: 100, maxHops: 2};
  const fixture = createFixture(device, options, 'hood-clamp');
  await expectMatchesOracle(device, fixture, options);
  expect(await readUint32(fixture.buffers.hopDistances, 6)).toEqual([0, 1, 2, NONE, NONE, NONE]);
  fixture.destroy();
});

it('GPUNetworkNeighborhood returns an empty ego network for zero active seeds', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const options: FixtureOptions = {
    edges: F1_EDGES,
    nodeCount: 8,
    seeds: [0, 1],
    seedCount: 0,
    hops: 3
  };
  const fixture = createFixture(device, options, 'hood-empty');
  await expectMatchesOracle(device, fixture, options);
  expect(await readUint32(fixture.buffers.hopDistances, 8)).toEqual(new Array(8).fill(NONE));
  expect(await readUint32(fixture.buffers.nodeCount, 1)).toEqual([0]);
  expect(await readUint32(fixture.buffers.edgeCount, 1)).toEqual([0]);
  fixture.destroy();
});

it('GPUNetworkNeighborhood handles a single node and edgeless networks', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const options: FixtureOptions = {edges: [], nodeCount: 1, seeds: [0], hops: 2};
  const fixture = createFixture(device, options, 'hood-single');
  await expectMatchesOracle(device, fixture, options);
  expect(await readUint32(fixture.buffers.nodeIds, 1)).toEqual([0]);
  fixture.destroy();
});

it('GPUNetworkNeighborhood emits stable node and edge IDs', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const csr = buildCSR(8, F1_EDGES);
  const options: FixtureOptions = {
    edges: F1_EDGES,
    nodeCount: 8,
    seeds: [2],
    hops: 2,
    nodeIds: Array.from({length: 8}, (_, node) => 1000 + node),
    edgeIds: Array.from({length: csr.neighbors.length}, (_, edge) => 5000 + edge),
    nodeCapacity: 3
  };
  const fixture = createFixture(device, options, 'hood-ids');
  await expectMatchesOracle(device, fixture, options);
  const count = (await readUint32(fixture.buffers.nodeCount, 1))[0];
  expect((await readUint32(fixture.buffers.nodeIds, count)).every(id => id >= 1000)).toBe(true);
  fixture.destroy();
});
