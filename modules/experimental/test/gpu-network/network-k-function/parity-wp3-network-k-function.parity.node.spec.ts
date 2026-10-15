// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {compareParity, EXACT} from '../parity-wp3-comparison';
import {getDistributionDistance, kFixtures} from './parity-wp3-network-k-function-fixture';

for (const fixture of kFixtures) {
  it(`WP3 K fixture integrity only, no GPU execution: ${fixture.case}`, () => {
    expect(fixture.versions['spaghetti']).toBeTruthy();
    const className = 'GPUNetworkKFunction/CPU-fixture-integrity-only';
    const length = fixture.weights.reduce((sum, weight) => sum + weight, 0) / 2;
    compareParity(className, fixture, 'network length', [length], [fixture.exactNetworkLength], {
      absolute: 1e-8,
      relative: 1e-12
    });
    compareParity(
      className,
      fixture,
      'event intensity',
      [fixture.points.length / length],
      [fixture.intensity],
      {absolute: 1e-12, relative: 1e-12}
    );
    compareParity(
      className,
      fixture,
      'Spaghetti observed K normalization',
      fixture.counts.map(count => (2 * count * length) / fixture.points.length ** 2),
      fixture.observed,
      {absolute: 1e-10, relative: 1e-12}
    );
    compareParity(
      className,
      fixture,
      'all CSR edges present',
      [fixture.offsets.at(-1)!],
      [fixture.neighbors.length],
      EXACT
    );
    compareParity(
      className,
      fixture,
      'boundary ambiguity fully enumerated',
      fixture.upperCounts.map((value, index) => value - fixture.lowerCounts[index]),
      fixture.ambiguousPairs.map(pairs => pairs.length),
      EXACT
    );
    compareParity(
      className,
      fixture,
      'counts inside reference intervals',
      fixture.counts.map((value, index) =>
        Number(value < fixture.lowerCounts[index] || value > fixture.upperCounts[index])
      ),
      new Array(fixture.radii.length).fill(0),
      EXACT
    );
  });
}
it('WP3 empirical CDF checks ties on both sides of jumps (CPU integrity only)', () => {
  compareParity(
    'GPUNetworkKFunction/CPU-fixture-integrity-only',
    {case: 'CDF helper', reference: 'exact empirical CDF definitions'},
    'CDF distances',
    [
      getDistributionDistance([0, 0, 1], [0, 1, 1]),
      getDistributionDistance([0, 0], [1, 1]),
      getDistributionDistance([1, 1], [1, 1])
    ],
    [1 / 3, 1, 0],
    {absolute: 1e-15, relative: 0}
  );
});
