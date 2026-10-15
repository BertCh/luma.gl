// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {mapEdgeId, mapNodeId, routingFixtures} from './parity-wp3-routing-fixture';

for (const {input, cases} of routingFixtures) {
  for (const fixture of cases) {
    it(`reference integrity, not GPU parity: routing ${fixture.case}`, () => {
      expect(fixture.versions['networkx']).toBeTruthy();
      expect(fixture.costs).toHaveLength(input.nodeCount);
      expect(
        new Set(Array.from({length: input.nodeCount}, (_, row) => mapNodeId(input, row))).size
      ).toBe(input.nodeCount);
      expect(new Set(input.neighbors.map((_, row) => mapEdgeId(input, row))).size).toBe(
        input.neighbors.length
      );
      const roots = new Set(
        fixture.startNodes.filter(
          (node, index) =>
            node < input.nodeCount && fixture.costs[node] === fixture.startCosts[index]
        )
      );
      const certified = new Set(roots);
      for (let node = 0; node < input.nodeCount; node++) {
        const cost = fixture.costs[node];
        if (!Number.isFinite(cost)) {
          expect(fixture.predecessors[node]).toBe(0xffffffff);
          continue;
        }
        expect(Math.fround(cost)).toBe(cost);
        expect(cost).toBeLessThanOrEqual(fixture.limit);
        for (let edge = input.offsets[node]; edge < input.offsets[node + 1]; edge++) {
          const candidate = cost + input.weights[edge];
          if (candidate <= fixture.limit)
            expect(fixture.costs[input.neighbors[edge]]).toBeLessThanOrEqual(candidate);
        }
        const chain = new Set<number>();
        let current = node;
        while (!certified.has(current)) {
          expect(chain.has(current), 'acyclic source-rooted predecessor certificate').toBe(false);
          chain.add(current);
          const parent = fixture.predecessors[current];
          expect(parent).toBeLessThan(input.nodeCount);
          let tight = false;
          for (let edge = input.offsets[parent]; edge < input.offsets[parent + 1]; edge++) {
            tight ||=
              input.neighbors[edge] === current &&
              fixture.costs[parent] + input.weights[edge] === fixture.costs[current];
          }
          expect(tight).toBe(true);
          current = parent;
        }
        for (const visited of chain) certified.add(visited);
      }
      for (const [row, path] of fixture.paths.entries()) {
        const edges = fixture.edgePaths[row];
        expect(edges.length).toBe(Math.max(0, path.length - 1));
        if (!path.length) {
          expect(fixture.pathCosts[row]).toBe(Infinity);
          continue;
        }
        expect(roots.has(path[0])).toBe(true);
        expect(path[path.length - 1]).toBe(fixture.targets[row]);
        expect(new Set(path).size).toBe(path.length);
        for (const [index, edge] of edges.entries()) {
          expect(edge).toBeGreaterThanOrEqual(input.offsets[path[index]]);
          expect(edge).toBeLessThan(input.offsets[path[index] + 1]);
          expect(input.neighbors[edge]).toBe(path[index + 1]);
        }
        expect(
          edges.reduce((cost, edge) => cost + input.weights[edge], fixture.costs[path[0]])
        ).toBe(fixture.pathCosts[row]);
      }
      expect(fixture.bandCounts.reduce((sum, count) => sum + count, 0)).toBe(
        fixture.bands.filter(value => value !== 0xffffffff).length
      );
    });
  }
}
