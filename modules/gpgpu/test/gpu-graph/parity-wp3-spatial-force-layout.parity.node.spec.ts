// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {compareParity, EXACT} from '../../../experimental/test/gpu-network/parity-wp3-comparison';
import {evaluateLayout} from './parity-wp3-force-layout-fixture';
import {canonicalizeGridIds, spatialFixtures} from './parity-wp3-spatial-force-layout-fixture';

it('grid comparison ignores within-cell order but preserves membership and multiplicity', () => {
  const offsets = [0, 3, 3, 5];
  const expected = canonicalizeGridIds([0, 2, 4, 1, 3], offsets);
  expect(canonicalizeGridIds([4, 0, 2, 3, 1], offsets)).toEqual(expected);
  expect(canonicalizeGridIds([4, 0, 3, 2, 1], offsets)).not.toEqual(expected);
  expect(canonicalizeGridIds([4, 0, 0, 3, 1], offsets)).not.toEqual(expected);
});

for (const fixture of spatialFixtures) {
  it(`pinned objective and grid reference integrity only: ${fixture.case}`, () => {
    expect(fixture.versions['scipy']).toBeTruthy();
    expect(fixture.versions['networkx']).toBeTruthy();
    for (const run of fixture.runs) {
      expect(run.initial.positions.every(value => Math.fround(value) === value)).toBe(true);
      const anchorIndices = run.initial.positions
        .map((_, index) => index)
        .filter(index => fixture.pinned[Math.floor(index / 2)]);
      for (const method of ['initial', 'optimized', 'networkx'] as const) {
        const reference = run[method];
        const actual = evaluateLayout(fixture, reference.positions);
        for (const metric of ['gradient', 'edgeLengths', 'components'] as const) {
          compareParity(
            'GPUGraphSpatialForceLayout evaluator integrity',
            {...fixture, case: `${fixture.case}/${run.seed}/${method}`},
            metric,
            actual[metric],
            reference[metric],
            {absolute: 1e-8, relative: 1e-10}
          );
        }
        compareParity(
          'GPUGraphSpatialForceLayout evaluator integrity',
          fixture,
          `${method}: every fixed anchor`,
          anchorIndices.map(index => reference.positions[index]),
          anchorIndices.map(index => run.initial.positions[index]),
          EXACT
        );
      }
      const displacement =
        fixture.frames *
        fixture.iterationsPerFrame *
        fixture.parameters.timeStep *
        fixture.parameters.maxVelocity;
      expect(Math.max(...run.initial.positions.map(Math.abs)) + displacement).toBeLessThan(
        fixture.bounds[2]
      );
      const expectedGradient = run.initial.gradient.map((value, index) =>
        fixture.pinned[Math.floor(index / 2)] ? 0 : value
      );
      expect(Math.max(...expectedGradient.map(Math.abs)) * Math.fround(0.001)).toBeLessThan(
        1e6 / Math.SQRT2
      );
      const grid = run.gridReference;
      expect(grid.offsets).toHaveLength(fixture.gridSize[0] * fixture.gridSize[1] + 1);
      expect(grid.offsets[grid.offsets.length - 1]).toBe(fixture.nodeCount);
      expect([...grid.vertexIds].sort((first, second) => first - second)).toEqual(
        Array.from({length: fixture.nodeCount}, (_, index) => index)
      );
      let guaranteedFarPopulations = 0;
      const cellWidth = (fixture.bounds[2] - fixture.bounds[0]) / fixture.gridSize[0];
      const cellHeight = (fixture.bounds[3] - fixture.bounds[1]) / fixture.gridSize[1];
      const diameter = Math.hypot(cellWidth, cellHeight);
      for (let cell = 0; cell < grid.offsets.length - 1; cell++) {
        const members = grid.vertexIds.slice(grid.offsets[cell], grid.offsets[cell + 1]);
        expect(members).toEqual([...members].sort((first, second) => first - second));
        if (members.length > 1) {
          const minimumX = fixture.bounds[0] + (cell % fixture.gridSize[0]) * cellWidth;
          const minimumY = fixture.bounds[1] + Math.floor(cell / fixture.gridSize[0]) * cellHeight;
          for (let node = 0; node < fixture.nodeCount; node++) {
            // Conservative geometric coverage certificate: distance to the entire cell
            // exceeds diameter/theta, so its population must enter the far-field branch.
            // This checks test coverage, not reference forces or GPU stepping.
            const distanceX = Math.max(
              minimumX - run.initial.positions[2 * node],
              run.initial.positions[2 * node] - minimumX - cellWidth,
              0
            );
            const distanceY = Math.max(
              minimumY - run.initial.positions[2 * node + 1],
              run.initial.positions[2 * node + 1] - minimumY - cellHeight,
              0
            );
            if (Math.hypot(distanceX, distanceY) > diameter / 0.6) guaranteedFarPopulations++;
          }
        }
        for (const axis of [0, 1]) {
          const mean =
            members.reduce((sum, node) => sum + run.initial.positions[node * 2 + axis], 0) /
            (members.length || 1);
          expect(grid.centers[cell * 2 + axis]).toBeCloseTo(mean, 12);
        }
      }
      if (fixture.case.includes('openflights')) expect(guaranteedFarPopulations).toBeGreaterThan(0);
    }
  });
}
