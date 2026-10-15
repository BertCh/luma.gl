// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {compareParity, EXACT} from '../parity-wp3-comparison';
import {snappingFixtures} from './parity-wp3-snapping-fixture';

for (const fixture of snappingFixtures) {
  it(`snapping fixture integrity only: ${fixture.case}`, () => {
    expect(fixture.versions['shapely']).toBeTruthy();
    // Exact-copy assertions are justified only when upload introduces no rounding.
    const inputCoordinates = fixture.queries.flatMap(query => query.point);
    compareParity(
      'GPUNetworkSnapping fixture integrity',
      fixture,
      'raw JSON preserves every f32 upload coordinate exactly',
      inputCoordinates,
      inputCoordinates.map(Math.fround),
      EXACT
    );
    if (fixture.case === 'chicago') {
      expect(
        Math.max(
          ...fixture.positions.map(Math.abs),
          ...fixture.queries.flatMap(query => query.point.map(Math.abs))
        )
      ).toBeLessThan(32768);
      expect(fixture.coordinateBound).toBe(16 * 2 ** -9);
    }
    expect(fixture.positions.length).toBe(2 * fixture.nodeCount);
    expect(fixture.neighbors.length).toBe(fixture.weights.length);
    expect(fixture.offsets.at(-1)).toBe(fixture.neighbors.length);
    expect(fixture.originalPointIds.length).toBe(fixture.queries.length);
    expect(fixture.candidateCapacity).toBeGreaterThan(fixture.boundingBoxCandidateCount);
    compareParity(
      'GPUNetworkSnapping fixture integrity',
      fixture,
      'all original source-edge IDs preserved',
      [...fixture.originalEdgeIds].sort((first, second) => first - second),
      Array.from({length: fixture.neighbors.length}, (_, index) => index),
      EXACT
    );
    const violations = fixture.queries.map(query =>
      Number(
        query.candidates.length > 0 !== query.nearestDistance <= fixture.radius ||
          query.candidates.some(
            candidate =>
              candidate.distance > query.nearestDistance + 2 * fixture.coordinateBound ||
              candidate.fraction < 0 ||
              candidate.fraction > 1 ||
              candidate.edge < fixture.offsets[candidate.source] ||
              candidate.edge >= fixture.offsets[candidate.source + 1] ||
              fixture.neighbors[candidate.edge] !== candidate.target ||
              fixture.weights[candidate.edge] !== candidate.weight
          ) ||
          (fixture.exactTieBreak &&
            query.candidates.length > 0 &&
            query.candidates[0].edge !== Math.min(...query.nearestEdges))
      )
    );
    compareParity(
      'GPUNetworkSnapping fixture integrity',
      fixture,
      'all tie intervals, fractions, endpoint IDs and costs are consistent',
      violations,
      new Array(fixture.queries.length).fill(0),
      EXACT
    );
    if (fixture.case === 'chicago') {
      expect(fixture.nodeCount).toBe(29557);
      expect(fixture.neighbors.length).toBe(77140);
      expect(fixture.queries.length).toBe(2048);
    }
  });
}

it('Chicago projection roundoff fits the existing G bound without widening', () => {
  const fixture = snappingFixtures.find(value => value.case === 'chicago')!;
  const unitRoundoff = 2 ** -24;
  const gamma5 = (5 * unitRoundoff) / (1 - 5 * unitRoundoff);
  const gamma3 = (3 * unitRoundoff) / (1 - 3 * unitRoundoff);
  const positionBounds: number[] = [];
  for (const query of fixture.queries) {
    for (const candidate of query.candidates) {
      if (candidate.length === 0) continue;
      const start = fixture.positions.slice(candidate.source * 2, candidate.source * 2 + 2);
      const end = fixture.positions.slice(candidate.target * 2, candidate.target * 2 + 2);
      const delta = end.map((value, axis) => value - start[axis]);
      const offset = query.point.map((value, axis) => value - start[axis]);
      // For rounded differences and 2D dot products, gamma5 bounds relative error
      // in v.v and absolute error gamma5*|w|.|v| in w.v. Cauchy-Schwarz gives
      // |delta t| <= (2*gamma5 + 5*u*(1+gamma5))/(1-gamma5) * ||w||/||v||.
      // The 5*u term allows 2.5-ulp WGSL division accuracy, not just IEEE 0.5 ulp.
      // Clamp is nonexpansive. For a+t*v, gamma3 covers subtraction/multiply/add:
      // |delta x_j| <= |v_j|*|delta t| + gamma3*(|a_j|+|v_j|*(1+|delta t|)).
      // Inputs are exact f32 normal metre coordinates; no underflow/overflow here.
      const fractionError =
        (((2 * gamma5 + 5 * unitRoundoff * (1 + gamma5)) / (1 - gamma5)) * Math.hypot(...offset)) /
        candidate.length;
      for (let axis = 0; axis < 2; axis++) {
        positionBounds.push(
          Math.abs(delta[axis]) * fractionError +
            gamma3 * (Math.abs(start[axis]) + Math.abs(delta[axis]) * (1 + fractionError))
        );
      }
    }
  }
  console.log(
    'PARITY',
    JSON.stringify({
      class: 'GPUNetworkSnapping bound derivation',
      case: fixture.case,
      reference: 'standard gamma_n roundoff bound on f32 inputs; 2.5-ulp WGSL division allowance',
      n: positionBounds.length,
      metric: 'all selected-edge interpolation error bounds (metres)',
      maxAbs: Math.max(...positionBounds),
      maxRel: 0,
      mismatches: positionBounds.filter(bound => bound > fixture.coordinateBound).length,
      tolerance: {absolute: fixture.coordinateBound, relative: 0}
    })
  );
  expect(Math.max(...positionBounds)).toBeLessThan(fixture.coordinateBound);
});
