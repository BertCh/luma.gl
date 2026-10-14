// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {dijkstra} from './network-reachability-oracle';
import {
  compareShortestPathCosts,
  getShortestPathReferenceCSR,
  SHORTEST_PATH_REFERENCE
} from './network-reachability-reference';

// Library rule 22 (review B-21): the TypeScript Dijkstra oracle that the GPU specs compare against
// is itself checked against networkx, so a convention shared by the kernel and its oracle cannot
// hide a wrong answer.

it('the shortest-path fixture records its generator, versions and tolerances', () => {
  expect(SHORTEST_PATH_REFERENCE.generator).toBe('test/oracles/network-shortest-paths.py');
  expect(SHORTEST_PATH_REFERENCE.versions['networkx']).toMatch(/^\d+\.\d+/);
  const names = Object.keys(SHORTEST_PATH_REFERENCE.scenes);
  expect(names).toEqual(['hand', 'random-1', 'random-2', 'random-3', 'grid']);
  for (const scene of Object.values(SHORTEST_PATH_REFERENCE.scenes)) {
    // (2 h + 1) 2^-24, derived in the generator.
    expect(scene.tolerance.costRelative).toBeCloseTo((2 * scene.maximumHops + 1) * 2 ** -24, 20);
    expect(scene.costs).toHaveLength(scene.nodeCount);
  }
});

it('the float32 Dijkstra oracle matches networkx on every shortest-path scene', () => {
  for (const [name, scene] of Object.entries(SHORTEST_PATH_REFERENCE.scenes)) {
    const costs = dijkstra(
      getShortestPathReferenceCSR(scene),
      scene.nodeCount,
      scene.sources.map((node, row) => ({node, cost: scene.sourceCosts[row]})),
      scene.costLimit ?? Infinity
    );
    expect(compareShortestPathCosts(scene, costs), name).toBeUndefined();
  }
});

it('the hand scene pins the zero-weight plateau, impassable edge and cost limit', () => {
  const {hand} = SHORTEST_PATH_REFERENCE.scenes;
  // Source 5 starts at 2.5 and reaches 6 and 3 across zero-weight edges; node 7 is only behind
  // the impassable edge or past the limit of 9; node 8 has no edges.
  expect(hand.costs).toEqual([0, 3, 1, 2.5, 5.5, 2.5, 2.5, null, null]);
});
