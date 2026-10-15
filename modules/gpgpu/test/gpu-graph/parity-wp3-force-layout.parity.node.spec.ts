// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {compareParity} from '../../../experimental/test/gpu-network/parity-wp3-comparison';
import {evaluateLayout, layoutFixtures} from './parity-wp3-force-layout-fixture';

for (const fixture of layoutFixtures) {
  it(`layout energy evaluator integrity only: ${fixture.case}`, () => {
    expect(fixture.versions['scipy']).toBeTruthy();
    expect(fixture.versions['networkx']).toBeTruthy();
    for (const run of fixture.runs) {
      expect(run.initial.positions.every(value => Math.fround(value) === value)).toBe(true);
      for (const method of ['initial', 'optimized', 'networkx'] as const) {
        const reference = run[method];
        expect(reference.positions).toHaveLength(2 * fixture.nodeCount);
        expect(reference.edgeLengths).toHaveLength(fixture.edges.length);
        const actual = evaluateLayout(fixture, reference.positions);
        for (const metric of ['gradient', 'edgeLengths', 'components'] as const) {
          compareParity(
            'GPUGraphForceLayout evaluator integrity',
            {...fixture, case: `${fixture.case}/${run.seed}/${method}`},
            `every ${metric} versus SciPy`,
            actual[metric],
            reference[metric],
            {absolute: 1e-8, relative: 1e-10}
          );
        }
        compareParity(
          'GPUGraphForceLayout evaluator integrity',
          fixture,
          'energy and equilibrium summary versus SciPy',
          [actual.energy, actual.rootMeanSquareGradient],
          [reference.energy, reference.rootMeanSquareGradient],
          {absolute: 1e-8, relative: 1e-10}
        );
      }
    }
  });
}

it('composite objective gradient matches finite differences, including softened collisions', () => {
  const fixture = layoutFixtures.find(value => value.case === 'synthetic')!;
  const positions = [...fixture.runs[0].initial.positions];
  positions[0] = positions[2] + 0.003;
  positions[1] = positions[3] - 0.002;
  const gradient = evaluateLayout(fixture, positions).gradient;
  const increment = 1e-6;
  const numerical = positions.map((_, index) => {
    const before = [...positions];
    const after = [...positions];
    before[index] -= increment;
    after[index] += increment;
    return (
      (evaluateLayout(fixture, after).energy - evaluateLayout(fixture, before).energy) /
      (2 * increment)
    );
  });
  compareParity(
    'GPUGraphForceLayout evaluator integrity',
    fixture,
    'full energy gradient finite differences',
    gradient,
    numerical,
    {absolute: 1e-7, relative: 1e-7}
  );
});
