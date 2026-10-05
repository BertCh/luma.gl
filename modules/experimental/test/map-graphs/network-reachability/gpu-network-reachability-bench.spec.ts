// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUNetworkReachability} from '../../../src/map-graphs/network-reachability';
import {isSoftwareDevice} from '../map-graph-test-utils';
import {
  formatReachabilityBench,
  measureReachability,
  type ReachabilityBenchResult
} from './network-reachability-bench-utils';
import {createGridNetwork, createRandomNetwork} from './network-reachability-oracle';

const GRID_SIDE = 224;
const NODE_COUNT = GRID_SIDE * GRID_SIDE;

/** Hops one round chain covers is `localIterations`; rounds are sized for ~700 grid hops. */
function getRoundBudget(localIterations: number): number {
  return Math.min(1024, Math.ceil(700 / localIterations) + 8);
}

it('GPUNetworkReachability bench: graph nodes, CPU encode, and rounds on a 50k-node grid', async () => {
  const device = await getWebGPUTestDevice();
  if (!device || isSoftwareDevice(device)) {
    return;
  }
  const edges = createGridNetwork(7, GRID_SIDE, GRID_SIDE);
  const results: ReachabilityBenchResult[] = [];
  const configurations: {label: string; maxIterations: number; localIterations: number}[] = [
    {label: 'hop-per-round, 640 rounds', maxIterations: 640, localIterations: 1}
  ];
  for (const localIterations of [4, 16, 64]) {
    configurations.push({
      label: `localIterations ${localIterations}`,
      maxIterations: getRoundBudget(localIterations),
      localIterations
    });
  }
  for (const configuration of configurations) {
    results.push(
      await measureReachability(device, GPUNetworkReachability, {
        ...configuration,
        nodeCount: NODE_COUNT,
        edges,
        sources: [0]
      })
    );
  }
  // eslint-disable-next-line no-console
  console.warn(
    formatReachabilityBench(`network-reachability bench ${GRID_SIDE}x${GRID_SIDE} grid`, results)
  );
  const randomResults: ReachabilityBenchResult[] = [];
  const randomEdges = createRandomNetwork(5, NODE_COUNT, NODE_COUNT * 4);
  for (const localIterations of [1, 16, 64]) {
    randomResults.push(
      await measureReachability(device, GPUNetworkReachability, {
        label: `random localIterations ${localIterations}`,
        maxIterations: 64,
        localIterations,
        nodeCount: NODE_COUNT,
        edges: randomEdges,
        sources: [0, 20000]
      })
    );
  }
  console.warn(
    formatReachabilityBench(
      'network-reachability bench random 50k nodes, 200k edges',
      randomResults
    )
  );
  // Predecessors on versus off: the tie-level phase adds maxTieIterations + 4 graph nodes.
  const predecessorResults: ReachabilityBenchResult[] = [];
  for (const predecessors of [false, true]) {
    predecessorResults.push(
      await measureReachability(device, GPUNetworkReachability, {
        label: `grid li 16 predecessors ${predecessors ? 'on' : 'off'}`,
        maxIterations: getRoundBudget(16),
        localIterations: 16,
        nodeCount: NODE_COUNT,
        edges,
        sources: [0],
        predecessors
      }),
      await measureReachability(device, GPUNetworkReachability, {
        label: `random li 16 predecessors ${predecessors ? 'on' : 'off'}`,
        maxIterations: 64,
        localIterations: 16,
        nodeCount: NODE_COUNT,
        edges: randomEdges,
        sources: [0, 20000],
        predecessors
      })
    );
  }
  console.warn(
    formatReachabilityBench('network-reachability bench predecessors', predecessorResults)
  );
  for (const result of results.slice(1)) {
    expect(result.converged).toBe(1);
    expect(result.graphNodes).toBeLessThan(results[0].graphNodes);
  }
}, 900000);
