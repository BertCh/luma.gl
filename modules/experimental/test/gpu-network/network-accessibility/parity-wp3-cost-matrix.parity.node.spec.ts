// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {compareParity, EXACT} from '../parity-wp3-comparison';
import {costMatrixFixtures} from './parity-wp3-cost-matrix-fixture';

for (const {input, cases} of costMatrixFixtures) {
  it(`cost-matrix fixture integrity only: ${input.case}`, () => {
    expect(input.versions['networkx']).toBeTruthy();
    expect(input.offsets.length).toBe(input.nodeCount + 1);
    expect(input.offsets.at(-1)).toBe(input.neighbors.length);
    expect(input.weights.length).toBe(input.neighbors.length);
    for (const fixture of cases) {
      expect(fixture.costs.length).toBe(input.rowCount * input.nodeCount);
      const gathered = Array.from({length: input.rowCount}, (_, row) =>
        input.targets.map(target =>
          target < input.nodeCount ? fixture.costs[row * input.nodeCount + target] : Infinity
        )
      ).flat();
      compareParity(
        'GPUNetworkCostMatrix fixture integrity',
        fixture,
        'full compact gather, invalid and duplicate targets',
        gathered,
        fixture.compact,
        EXACT
      );
      const lastRow = fixture.costs.slice((input.rowCount - 1) * input.nodeCount);
      compareParity(
        'GPUNetworkCostMatrix fixture integrity',
        fixture,
        'empty final row stays infinite',
        lastRow,
        new Array(input.nodeCount).fill(Infinity),
        EXACT
      );
      compareParity(
        'GPUNetworkCostMatrix fixture integrity',
        fixture,
        'all finite distances exactly representable and inside inclusive cutoff',
        fixture.costs.map(value =>
          Number(Number.isFinite(value) && (Math.fround(value) !== value || value > fixture.budget))
        ),
        new Array(fixture.costs.length).fill(0),
        EXACT
      );
    }
  });
}
