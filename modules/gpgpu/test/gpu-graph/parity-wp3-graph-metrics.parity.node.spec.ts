// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {compareParity, EXACT} from '../../../experimental/test/gpu-network/parity-wp3-comparison';
import {evaluateRankResidual, graphMetricsFixtures} from './parity-wp3-graph-metrics-fixture';

for (const fixture of graphMetricsFixtures) {
  it(`graph metric fixture integrity only: ${fixture.case}`, () => {
    expect(fixture.versions['networkx']).toBeTruthy();
    expect(fixture.cores).toHaveLength(fixture.nodeCount);
    const weakNeighbors = Array.from({length: fixture.nodeCount}, () => new Set<number>());
    for (const [source, target] of fixture.edges) {
      expect(source).toBeLessThan(fixture.nodeCount);
      expect(target).toBeLessThan(fixture.nodeCount);
      if (source !== target) {
        weakNeighbors[source].add(target);
        weakNeighbors[target].add(source);
      }
    }
    // Certificate for each NetworkX k-core: every member has >=k neighbors in that core.
    // This is fixture integrity, not another core-number solver or GPU evidence.
    compareParity(
      'GPUGraphCoreNumber fixture integrity',
      fixture,
      'all core memberships feasible',
      fixture.cores.map((core, node) =>
        Number(
          !Number.isInteger(core) ||
            core < 0 ||
            [...weakNeighbors[node]].filter(neighbor => fixture.cores[neighbor] >= core).length <
              core
        )
      ),
      new Array(fixture.nodeCount).fill(0),
      EXACT
    );
    expect(Math.max(...fixture.cores)).toBe(fixture.degeneracy);
    const occurrences = fixture.directed
      ? fixture.edges.length
      : fixture.edges.reduce((sum, [source, target]) => sum + (source === target ? 1 : 2), 0);
    for (const direction of ['incoming', 'outgoing'] as const) {
      compareParity(
        'GPUGraphDegree fixture integrity',
        fixture,
        `${direction} edge conservation`,
        [fixture[direction].reduce((sum, value) => sum + value, 0)],
        [occurrences],
        EXACT
      );
    }
    for (const run of fixture.rankCases) {
      expect(run.ranks).toHaveLength(fixture.nodeCount);
      expect(run.ranks.every(value => Number.isFinite(value) && value >= 0)).toBe(true);
      compareParity(
        'GPUGraphPageRank fixture integrity',
        fixture,
        `all NetworkX residuals alpha=${run.damping}`,
        evaluateRankResidual(fixture, run.ranks, run.damping),
        new Array(fixture.nodeCount).fill(0),
        {absolute: 2e-14, relative: 0}
      );
      compareParity(
        'GPUGraphPageRank fixture integrity',
        fixture,
        'probability sum',
        [run.ranks.reduce((sum, value) => sum + value, 0)],
        [1],
        {absolute: 2e-13, relative: 0}
      );
    }
    if (fixture.case.startsWith('openflights')) {
      expect(fixture.nodeCount).toBe(3257);
      expect(fixture.edges).toHaveLength(18930);
    }
  });
}
