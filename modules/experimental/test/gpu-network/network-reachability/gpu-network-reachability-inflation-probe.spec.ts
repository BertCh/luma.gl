// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUNetworkReachabilityInflation,
  GPUNetworkReachability
} from '../../../src/gpu-network/network-reachability';
import {
  createInputBuffer,
  createOutputBuffer,
  isSoftwareDevice,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  buildCSR,
  createGridNetwork,
  createRandomNetwork,
  type NetworkEdge
} from './network-reachability-oracle';

/**
 * Inflation probe for the algorithms roadmap (P1): measures relaxations per reached edge of the
 * chaotic Bellman-Ford on representative networks. The roadmap rejects delta-stepping and near-far
 * unless the inflation exceeds 3 (see `getGPUNetworkReachabilityInflation`). Prints a table with
 * console.warn; asserts only `inflation >= 1`. Skips on software devices.
 */

type ProbeScene = {
  name: string;
  nodeCount: number;
  edges: readonly NetworkEdge[];
  /** Start rows, as expanded node indices when `laneCount > 1`. */
  sources: number[];
  laneCount?: number;
};

type ProbeRow = {
  scene: string;
  localIterations: number;
  rounds: number;
  converged: number;
  relaxations: number;
  reachedEdges: number;
  inflation: number;
};

/** A grid plus a few long, fast "highway" links between far-apart cells (skewed weights). */
function createHighwayGridNetwork(side: number): NetworkEdge[] {
  const edges = createGridNetwork(11, side, side);
  let state = 12345;
  const next = (limit: number) => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return Math.floor((state / 2 ** 32) * limit);
  };
  for (let highway = 0; highway < 24; highway++) {
    const from = next(side * side);
    const to = next(side * side);
    if (from !== to) {
      // Cheaper than a few grid hops although the cells are up to ~side hops apart.
      edges.push([from, to, 4], [to, from, 4]);
    }
  }
  return edges;
}

function createScenes(): ProbeScene[] {
  const gridSide = 224;
  const laneSide = 64;
  const laneCount = 32;
  const laneNodeCount = laneSide * laneSide;
  return [
    {
      name: 'grid 224x224 random weights',
      nodeCount: gridSide * gridSide,
      edges: createGridNetwork(7, gridSide, gridSide),
      sources: [0]
    },
    {
      name: 'random 50k nodes 200k edges',
      nodeCount: gridSide * gridSide,
      edges: createRandomNetwork(5, gridSide * gridSide, gridSide * gridSide * 4),
      sources: [0, 20000]
    },
    {
      name: 'grid 224x224 + 24 highways',
      nodeCount: gridSide * gridSide,
      edges: createHighwayGridNetwork(gridSide),
      sources: [0]
    },
    {
      name: `grid 64x64 x ${laneCount} lanes`,
      nodeCount: laneNodeCount,
      edges: createGridNetwork(3, laneSide, laneSide),
      laneCount,
      // One source per lane at a different cell.
      sources: Array.from(
        {length: laneCount},
        (_, lane) => lane * laneNodeCount + ((lane * 131) % laneNodeCount)
      )
    }
  ];
}

async function probeScene(
  device: Device,
  scene: ProbeScene,
  localIterations: number
): Promise<ProbeRow> {
  const laneCount = scene.laneCount ?? 1;
  const csr = buildCSR(scene.nodeCount, scene.edges);
  const expandedNodeCount = scene.nodeCount * laneCount;
  const buffers = {
    offsets: createInputBuffer(device, csr.offsets),
    neighbors: createInputBuffer(device, csr.neighbors),
    weights: createInputBuffer(device, csr.weights),
    sources: createInputBuffer(device, Uint32Array.from(scene.sources)),
    costs: createOutputBuffer(device, expandedNodeCount),
    converged: createOutputBuffer(device, 1),
    iterationCount: createOutputBuffer(device, 1),
    relaxationCount: createOutputBuffer(device, 2),
    reachedEdgeCount: createOutputBuffer(device, 2)
  };
  const graph = new GPUCommandGraph(device, {id: `inflation-probe-${scene.name}`});
  // One lane chain covers about 0.85 * localIterations hops; a grid side of 224 needs ~450 hops.
  const maximumIterations = Math.min(1024, Math.ceil(900 / localIterations) + 8);
  graph.add(
    new GPUNetworkReachability({
      id: 'probe',
      topology: {
        offsets: importGraphBuffer(
          graph,
          'offsets',
          buffers.offsets,
          'uint32',
          scene.nodeCount + 1
        ),
        neighbors: importGraphBuffer(
          graph,
          'neighbors',
          buffers.neighbors,
          'uint32',
          csr.neighbors.length
        ),
        weights: importGraphBuffer(graph, 'weights', buffers.weights, 'float32', csr.weights.length)
      },
      startNodes: importGraphBuffer(
        graph,
        'sources',
        buffers.sources,
        'uint32',
        scene.sources.length
      ),
      laneCount,
      maximumIterations,
      localIterations,
      output: {
        costs: importGraphBuffer(graph, 'costs', buffers.costs, 'float32', expandedNodeCount),
        status: {
          converged: importGraphBuffer(graph, 'converged', buffers.converged, 'uint32', 1),
          iterationCount: importGraphBuffer(graph, 'rounds', buffers.iterationCount, 'uint32', 1),
          relaxationCount: importGraphBuffer(
            graph,
            'relaxation-count',
            buffers.relaxationCount,
            'uint32',
            2
          ),
          reachedEdgeCount: importGraphBuffer(
            graph,
            'reached-edge-count',
            buffers.reachedEdgeCount,
            'uint32',
            2
          )
        }
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const relaxationWords = await readUint32(buffers.relaxationCount, 2);
  const reachedWords = await readUint32(buffers.reachedEdgeCount, 2);
  const row: ProbeRow = {
    scene: scene.name,
    localIterations,
    rounds: (await readUint32(buffers.iterationCount, 1))[0],
    converged: (await readUint32(buffers.converged, 1))[0],
    relaxations: relaxationWords[0] + relaxationWords[1] * 2 ** 32,
    reachedEdges: reachedWords[0] + reachedWords[1] * 2 ** 32,
    inflation: getGPUNetworkReachabilityInflation(relaxationWords, reachedWords)
  };
  compiled.destroy();
  for (const buffer of Object.values(buffers)) {
    buffer.destroy();
  }
  return row;
}

function formatProbe(rows: ProbeRow[]): string {
  const lines = rows.map(
    row =>
      `${row.scene.padEnd(30)} li ${String(row.localIterations).padStart(2)}  rounds ${String(row.rounds).padStart(4)}  converged ${row.converged}  relaxations ${String(row.relaxations).padStart(10)}  reached ${String(row.reachedEdges).padStart(10)}  inflation ${row.inflation.toFixed(3)}`
  );
  return `network-reachability inflation probe (delta-stepping only worth it above 3)\n${lines.join('\n')}`;
}

it('GPUNetworkReachability inflation probe: relaxations per reached edge', async () => {
  const device = await getWebGPUTestDevice();
  if (!device || isSoftwareDevice(device)) {
    return;
  }
  const rows: ProbeRow[] = [];
  for (const scene of createScenes()) {
    for (const localIterations of [1, 16, 32]) {
      rows.push(await probeScene(device, scene, localIterations));
    }
  }
  // eslint-disable-next-line no-console
  console.warn(formatProbe(rows));
  for (const row of rows) {
    expect(row.inflation, `${row.scene} li ${row.localIterations}`).toBeGreaterThanOrEqual(1);
  }
}, 900000);
