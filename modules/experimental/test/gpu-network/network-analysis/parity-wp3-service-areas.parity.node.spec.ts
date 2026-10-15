// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {compareParity, EXACT} from '../parity-wp3-comparison';
import {getPredecessorViolations, serviceFixtures} from './parity-wp3-service-areas-fixture';

for (const {input, cases} of serviceFixtures) {
  for (const fixture of cases) {
    it(`WP3 service-area fixture integrity only, no GPU: ${fixture.case}`, () => {
      expect(fixture.versions['networkx']).toBeTruthy();
      const className = 'GPUNetworkServiceAreas/CPU-fixture-integrity-only';
      const counts = new Array<number>(input.facilities.length).fill(0);
      const sums = new Array<number>(input.facilities.length).fill(0);
      const children = Array.from({length: input.nodeCount}, () => [] as number[]);
      for (let node = 0; node < input.nodeCount; node++) {
        const assignment = fixture.assignments[node];
        if (assignment !== 0xffffffff) {
          counts[assignment]++;
          sums[assignment] += fixture.costs[node];
        }
        for (const predecessor of fixture.allowedPredecessors[node])
          children[predecessor].push(node);
      }
      compareParity(
        className,
        fixture,
        'counts cover all assigned vertices',
        counts,
        fixture.counts,
        EXACT
      );
      compareParity(className, fixture, 'all facility sums', sums, fixture.sums, EXACT);
      compareParity(
        className,
        fixture,
        'exact unassigned/infinite masks',
        fixture.costs.map(value => Number(!Number.isFinite(value))),
        fixture.assignments.map(value => Number(value === 0xffffffff)),
        EXACT
      );
      const predecessors = new Array<number>(input.nodeCount).fill(0xffffffff);
      const visited = new Set<number>();
      const queue: number[] = [];
      input.facilities.forEach((node, row) => {
        if (fixture.assignments[node] === row) {
          queue.push(node);
          visited.add(node);
        }
      });
      for (let index = 0; index < queue.length; index++) {
        const parent = queue[index];
        for (const node of children[parent]) {
          if (!visited.has(node) && fixture.assignments[node] === fixture.assignments[parent]) {
            visited.add(node);
            queue.push(node);
            predecessors[node] = parent;
          }
        }
      }
      compareParity(
        className,
        fixture,
        'every independent allowed predecessor set admits a rooted path',
        getPredecessorViolations(input, fixture, predecessors),
        new Array(input.nodeCount).fill(0),
        EXACT
      );
      if (fixture.case === 'synthetic-boundary') {
        const corrupted = [...predecessors];
        corrupted[3] = 4;
        corrupted[4] = 3;
        const violations = getPredecessorViolations(input, fixture, corrupted);
        compareParity(
          className,
          fixture,
          'zero-cost predecessor cycle detected',
          [violations[3], violations[4]],
          [1, 1],
          EXACT
        );
      }
    });
  }
}
