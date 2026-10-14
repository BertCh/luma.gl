// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {buildCSR} from '../network-reachability/network-reachability-oracle';
import {computeNetworkStatisticsOracle} from './network-statistics-oracle';
import {
  compareNetworkStatistics,
  getNetworkStatisticsReferenceEdges,
  NETWORK_STATISTICS_REFERENCE
} from './network-statistics-reference';

// Library rule 22 (reviews B-15, B-21): the TypeScript statistics oracle used by the GPU specs is
// checked against networkx degrees, components, isolates and modularity, self-loops included.

it('the statistics fixture records its generator, versions and tolerances', () => {
  expect(NETWORK_STATISTICS_REFERENCE.generator).toBe('test/oracles/network-statistics.py');
  expect(NETWORK_STATISTICS_REFERENCE.versions['networkx']).toMatch(/^\d+\.\d+/);
  for (const scene of Object.values(NETWORK_STATISTICS_REFERENCE.scenes)) {
    // (nodeCount + 4) 2^-24, derived in the generator.
    expect(scene.tolerance.modularityAbsolute).toBeCloseTo((scene.nodeCount + 4) * 2 ** -24, 20);
    expect(scene.selfLoopCount).toBeGreaterThan(0);
  }
});

it('the statistics oracle matches networkx, self-loops counted twice when undirected', () => {
  for (const [name, scene] of Object.entries(NETWORK_STATISTICS_REFERENCE.scenes)) {
    const words = computeNetworkStatisticsOracle({
      nodeCount: scene.nodeCount,
      csr: buildCSR(scene.nodeCount, getNetworkStatisticsReferenceEdges(scene)),
      directed: scene.directed,
      communities: scene.communities,
      resolution: scene.resolution
    });
    expect(compareNetworkStatistics(scene, words), name).toBeUndefined();
  }
});
