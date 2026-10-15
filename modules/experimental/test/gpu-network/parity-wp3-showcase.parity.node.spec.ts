// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {showcaseMatchFixtures} from './map-matching/parity-wp3-map-matching-fixture';
import {showcaseAccessibility} from './network-accessibility/parity-wp3-network-accessibility-fixture';
import {showcaseNeighborhoods} from './network-analysis/parity-wp3-network-analysis-fixture';
import {showcaseGraphs} from '../../../gpgpu/test/gpu-graph/parity-wp3-graph-showcase-fixture';

it('reference integrity only: all map compile combinations and slider probes retained', () => {
  expect(showcaseMatchFixtures).toHaveLength(74);
  expect(new Set(showcaseMatchFixtures.map(fixture => fixture.case)).size).toBe(74);
  const defaults = showcaseMatchFixtures.filter(fixture => fixture.case.includes('/defaults/'));
  expect(
    new Set(
      defaults.map(
        fixture => `${fixture.candidateCount}/${fixture.routeNodeBudget}/${fixture.cellSize}`
      )
    ).size
  ).toBe(64);
  for (const fixture of showcaseMatchFixtures) {
    expect(fixture.versions['networkx']).toBeTruthy();
    expect(fixture.versions['scipy']).toBeTruthy();
    expect(fixture.settings.searchRadius).toBeLessThanOrEqual(2 * fixture.cellSize);
    for (const track of fixture.candidates)
      for (const choices of track)
        expect(choices.length).toBeLessThanOrEqual(fixture.candidateCount!);
    for (const expected of fixture.expected) {
      expect(Number.isFinite(expected.logLikelihood)).toBe(true);
      expect(expected.alternatives.length).toBeGreaterThan(0);
    }
  }
});

it('reference integrity only: PySAL 2SFCA/3SFCA conserve all supplied opportunities', () => {
  expect(showcaseAccessibility.cases).toHaveLength(18);
  const supply = showcaseAccessibility.input.supply.reduce((sum, value) => sum + value, 0);
  for (const fixture of showcaseAccessibility.cases) {
    expect(fixture.reference).toContain(
      fixture.method === '3sfca' ? 'three_stage_fca' : 'two_stage_fca'
    );
    const consumed = fixture.expected.catchment.reduce(
      (sum, value, index) => sum + value * showcaseAccessibility.input.demand[index],
      0
    );
    expect(consumed).toBeCloseTo(supply, 8);
    expect(fixture.expected.catchment.at(-1)).toBe(0);
    expect(fixture.expected.ratios.every(value => Number.isFinite(value) && value > 0)).toBe(true);
  }
});

it('reference integrity only: every reachable routing hop count has a complete reference', () => {
  expect(showcaseNeighborhoods.map(fixture => fixture.hops)).toEqual(
    Array.from({length: 24}, (_, index) => index + 1)
  );
  for (const fixture of showcaseNeighborhoods) {
    expect(fixture.maximumHops).toBe(24);
    expect(fixture.nodeIds).toHaveLength(fixture.nodeCount);
    expect(fixture.edgeIds).toHaveLength(fixture.edgeCount);
    expect(fixture.hopDistances[0]).toBe(0);
  }
});

for (const fixture of showcaseGraphs) {
  it(`reference integrity only: Google matrix finite iterates ${fixture.case}`, () => {
    expect(fixture.versions['networkx']).toBeTruthy();
    expect(fixture.runs).toHaveLength(36);
    expect(fixture.runs.some(run => run.residual > 1e-4)).toBe(true);
    for (const run of fixture.runs) {
      expect(run.ranks).toHaveLength(fixture.nodeCount);
      expect(run.ranks.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 11);
      expect(run.ranks.every(value => value >= 0)).toBe(true);
      expect(
        run.ranks.reduce((sum, value, index) => sum + Math.abs(value - run.previous[index]), 0)
      ).toBeCloseTo(run.residual, 14);
    }
  });
}
