// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {ACCESSIBILITY_FIXTURES} from './parity-wp3-network-accessibility-fixture';
import {compareParity, EXACT} from '../parity-wp3-comparison';

// Provenance and model-alignment checks only; no GPU numerical coverage is claimed here.
for (const {input, cases} of ACCESSIBILITY_FIXTURES) {
  it(`accessibility reference integrity: ${input.case}`, () => {
    const facilityCount = input.supply.length;
    expect(input.costs).toHaveLength(input.originCount * facilityCount);
    expect(input.demand).toHaveLength(input.originCount);
    expect(input.versions['access']).toBeTruthy();
    expect(input.versions['networkx']).toBeTruthy();
    const selfCosts = input.facilityNodes.map(
      (node, facility) => input.costs[facility * input.originCount + node]
    );
    compareParity(
      'GPUNetworkAccessibility',
      input,
      'oracle integrity: facility self-costs (CPU only)',
      selfCosts,
      selfCosts.map(() => 0),
      EXACT
    );
    for (const reference of cases) {
      for (const metric of ['cumulative', 'gravity', 'catchment'] as const) {
        expect(reference.expected[metric]).toHaveLength(input.originCount);
        expect(reference.expected[metric].every(Number.isFinite)).toBe(true);
        expect(reference.expected[metric].every(value => value >= 0)).toBe(true);
      }
      expect(reference.expected.ratios).toHaveLength(facilityCount);
      if (!reference.settings.decay || reference.settings.decay === 'none') {
        compareParity(
          'GPUNetworkAccessibility',
          reference,
          'oracle integrity: no-decay weighted/unweighted catchment (CPU only)',
          reference.expected.gravity,
          reference.expected.cumulative,
          EXACT
        );
      }
    }
    if (input.case === 'chicago') {
      expect(input.originCount).toBe(29557);
      expect(input.source.fullNodeCount).toBe(input.originCount);
      expect(input.source.fullEdgeCount).toBe(77140);
      expect(
        Object.values(input.source.sha256 ?? {}).every(hash => /^[0-9a-f]{64}$/.test(hash))
      ).toBe(true);
      const unlimited = cases.find(reference => reference.case.endsWith('unlimited'))!;
      compareParity(
        'GPUNetworkAccessibility',
        unlimited,
        'oracle integrity: all origins reach both facilities (CPU only)',
        unlimited.expected.cumulative,
        input.demand.map(() => input.supply.reduce((total, supply) => total + supply, 0)),
        EXACT
      );
    } else {
      // 0->1 is free; 1->0 costs 3. 2->1 takes 2->3->1 = 5, not the reverse arc cost 2.
      compareParity(
        'GPUNetworkAccessibility',
        input,
        'oracle integrity: forward travel direction and unreachable masks (CPU only)',
        input.costs.slice(0, 8),
        [0, 0, 5, 1, 3, Infinity, Infinity, Infinity],
        EXACT
      );
      for (const reference of cases) {
        compareParity(
          'GPUNetworkAccessibility',
          reference,
          'oracle integrity: disconnected origins (CPU only)',
          reference.expected.catchment.slice(5),
          [0, 0, 0],
          EXACT
        );
      }
      const boundary = cases.find(reference => reference.case.endsWith('boundary'))!;
      compareParity(
        'GPUNetworkAccessibility',
        boundary,
        'oracle integrity: inclusive exact budget (CPU only)',
        boundary.expected.cumulative,
        [10, 10, 0, 10, 19, 0, 0, 0],
        EXACT
      );
    }
  });
}
