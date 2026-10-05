// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {GPUNetworkNeighborhood} from '../../../src/gpu-network/network-analysis/gpu-network-neighborhood';
import {createInputBuffer, createOutputBuffer, readUint32} from '../../utils/gpu-contributor-test-utils';
import {buildCSR, type NetworkEdge} from '../network-reachability/network-reachability-oracle';
import {createGridEdges, createSymmetricEdges} from './network-analytics-oracle';
import {neighborhoodOracle} from './network-neighborhood-oracle';

/**
 * GPUNetworkNeighborhood composes GPUNetworkReachability with the default `localIterations` (16),
 * so one relaxation round can chain many hops past `k`. Costs never drop below the true hop
 * distance, so the final `cost <= k` filter must keep k-hop results exact even when `k` is far
 * smaller than the hops a single round covers.
 */
async function expectExactNeighborhood(
  device: Device,
  name: string,
  nodeCount: number,
  edges: readonly NetworkEdge[],
  seeds: number[],
  maxHops: number,
  hopsPerFrame: readonly number[]
): Promise<void> {
  const csr = buildCSR(nodeCount, edges);
  const edgeCount = csr.neighbors.length;
  const graph = new GPUCommandGraph(device, {id: name});
  const owned: Buffer[] = [];
  const input = (suffix: string, values: Uint32Array, length: number) => {
    const buffer = createInputBuffer(device, values);
    owned.push(buffer);
    return importGraphBuffer(graph, `${name}-${suffix}`, buffer, 'uint32', length);
  };
  const output = (suffix: string, length: number) => {
    const buffer = createOutputBuffer(device, length);
    owned.push(buffer);
    return {buffer, view: importGraphBuffer(graph, `${name}-${suffix}`, buffer, 'uint32', length)};
  };
  const seedBuffer = new GPUParameterBuffer(device, {
    id: `${name}-seeds`,
    format: 'uint32',
    length: seeds.length,
    values: Uint32Array.from(seeds)
  });
  const hops = new GPUParameterBuffer(device, {
    id: `${name}-hops`,
    format: 'uint32',
    length: 1,
    values: Uint32Array.of(hopsPerFrame[0])
  });
  const hopDistances = output('hop-distances', nodeCount);
  const nodeMask = output('node-mask', nodeCount);
  const edgeMask = output('edge-mask', edgeCount);
  graph.add(
    new GPUNetworkNeighborhood({
      id: name,
      offsets: input('offsets', csr.offsets, nodeCount + 1),
      neighbors: input('neighbors', csr.neighbors, edgeCount),
      seeds: seedBuffer.importToGraph(graph),
      hops: hops.importToGraph(graph),
      maxHops,
      hopDistances: hopDistances.view,
      nodeMask: nodeMask.view,
      edgeMask: edgeMask.view
    })
  );
  const compiled = graph.compile();
  for (const k of hopsPerFrame) {
    // Per-frame k without recompiling.
    hops.write(Uint32Array.of(k));
    submitGraph(device, compiled, undefined);
    const expected = neighborhoodOracle({csr, nodeCount, seeds, hops: k, maxHops});
    expect(await readUint32(hopDistances.buffer, nodeCount), `${name} k=${k}`).toEqual(
      expected.hopDistances
    );
    expect(await readUint32(nodeMask.buffer, nodeCount), `${name} k=${k}`).toEqual(
      expected.nodeMask
    );
    expect(await readUint32(edgeMask.buffer, edgeCount), `${name} k=${k}`).toEqual(
      expected.edgeMask
    );
    // A long chain or grid reaches far beyond k, so exactness is not trivial.
    expect(expected.nodeMask.filter(Boolean).length).toBeLessThan(nodeCount);
  }
  compiled.destroy();
  seedBuffer.destroy();
  hops.destroy();
  for (const buffer of owned) buffer.destroy();
}

it('GPUNetworkNeighborhood keeps k-hop results exact when k is far below localIterations', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const chainLength = 200;
  const chain = createSymmetricEdges(
    Array.from({length: chainLength - 1}, (_, node) => [node, node + 1, 1] as NetworkEdge)
  );
  // maxHops equal to k: the reachability round budget is k, yet a round chains 16 hops.
  await expectExactNeighborhood(device, 'chain-k1', chainLength, chain, [100], 1, [1]);
  await expectExactNeighborhood(device, 'chain-k2', chainLength, chain, [100], 2, [2]);
  // maxHops above k with per-frame k of 1 and 2.
  await expectExactNeighborhood(device, 'chain-k12', chainLength, chain, [10, 150], 8, [1, 2]);

  const grid = createGridEdges(20, 20);
  await expectExactNeighborhood(device, 'grid-k1', 400, grid, [210], 1, [1]);
  await expectExactNeighborhood(device, 'grid-k2', 400, grid, [210], 2, [2]);
  await expectExactNeighborhood(device, 'grid-k12', 400, grid, [0, 210], 8, [1, 2]);
});
