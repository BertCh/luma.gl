// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {lineGraphFixtures} from './parity-wp3-line-graph-fixture';

for (const {input, cases} of lineGraphFixtures) {
  it(`reference integrity, not GPU parity: line graph ${input.case}`, () => {
    expect(input.versions['networkx']).toBeTruthy();
    const origins = new Array<number>(input.neighbors.length);
    for (let node = 0; node < input.nodeCount; node++) {
      for (let edge = input.offsets[node]; edge < input.offsets[node + 1]; edge++)
        origins[edge] = node;
    }
    const all = cases.find(fixture => fixture.case.endsWith('-all'))!;
    // Independently certify the full directed incidence product, not just an arc count.
    for (let edge = 0; edge < input.neighbors.length; edge++) {
      const pivot = input.neighbors[edge];
      const expected = Array.from(
        {length: input.offsets[pivot + 1] - input.offsets[pivot]},
        (_, index) => input.offsets[pivot] + index
      );
      expect(all.neighbors.slice(all.offsets[edge], all.offsets[edge + 1])).toEqual(expected);
    }
    for (const fixture of cases) {
      expect(fixture.offsets).toHaveLength(input.neighbors.length + 1);
      expect(fixture.neighbors).toHaveLength(fixture.count);
      expect(fixture.weights).toHaveLength(fixture.count);
      expect(fixture.offsets[input.neighbors.length]).toBe(fixture.count);
      const banned = new Set(
        Array.from(
          {length: fixture.bans.length / 2},
          (_, row) => `${fixture.bans[2 * row]},${fixture.bans[2 * row + 1]}`
        )
      );
      for (let first = 0; first < input.neighbors.length; first++) {
        let previous = -1;
        for (let entry = fixture.offsets[first]; entry < fixture.offsets[first + 1]; entry++) {
          const second = fixture.neighbors[entry];
          expect(origins[second]).toBe(input.neighbors[first]);
          expect(second).toBeGreaterThan(previous);
          expect(banned.has(`${first},${second}`)).toBe(false);
          if (fixture.exactCosts)
            expect(fixture.weights[entry]).toBe(
              input.weights[second] >= 0 ? input.weights[second] : -1
            );
          previous = second;
        }
      }
      if (fixture.case.endsWith('-banned'))
        expect(fixture.count).toBe(all.count - fixture.bans.length / 2);
    }
  });
}
