// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {findDisconnectedLabels} from './large-lattice-invariants';
import {MAX_P_FIXTURE} from './max-p-reference';
import {getRegionSums, runMaxPRegionsMirror} from './max-p-construction-mirror';

/**
 * Oracle: spopt `MaxPHeuristic` (test/oracles/statistics-max-p.py) on rook lattices and Delaunay
 * graphs. The GPU construction is a different heuristic, so the region count is compared by a
 * fraction, not equality. spopt's own region count varies by up to 3 regions (about 8 percent on
 * the small instances) across its five seeds, and the GPU construction takes the best of 16
 * starts, so the required fraction of spopt's best count is 0.9: below the spread of a single
 * heuristic with room for one extra pocket, far above what a poor packing (the sequential
 * construction's 70 to 80 percent when seeds crowd) reaches. The measured ratio to spopt's best is 0.975 to 1.04.
 *
 * The CPU mirror (max-p-construction-mirror.ts) executes the GPU rules; the GPU spec
 * (gpu-max-p-regions.spec.ts) pins the GPU labels to the mirror on exact inputs.
 */
const REQUIRED_FRACTION = 0.9;

function getWithinSumOfSquares(values: number[], labels: ArrayLike<number>): number {
  const groups = new Map<number, number[]>();
  Array.from(labels).forEach((label, row) => {
    groups.set(label, [...(groups.get(label) ?? []), values[row]]);
  });
  let total = 0;
  for (const group of groups.values()) {
    const mean = group.reduce((sum, value) => sum + value, 0) / group.length;
    total += group.reduce((sum, value) => sum + (value - mean) ** 2, 0);
  }
  return total;
}

it('the max-p construction reaches at least 90 percent of the spopt region count on every instance', () => {
  expect(MAX_P_FIXTURE.instances.length).toBeGreaterThanOrEqual(8);
  for (const instance of MAX_P_FIXTURE.instances) {
    const result = runMaxPRegionsMirror({
      offsets: instance.offsets,
      neighbors: instance.neighbors,
      values: instance.values,
      extensive: instance.extensive,
      threshold: instance.threshold,
      // The fixture values are already standardized.
      standardize: false,
      startCount: 16
    });
    const name = instance.name;
    // Feasible: every region reaches the threshold.
    expect(result.infeasibleRegionCount, name).toBe(0);
    for (const sum of getRegionSums(result.labels, instance.extensive).values()) {
      expect(sum, name).toBeGreaterThanOrEqual(instance.threshold);
    }
    // Contiguous.
    expect(
      findDisconnectedLabels(instance.offsets, instance.neighbors, result.labels),
      name
    ).toEqual([]);
    // Region count against spopt and the arithmetic upper bound.
    expect(result.regionCount, name).toBeLessThanOrEqual(instance.upperBound);
    expect(result.regionCount, name).toBeGreaterThanOrEqual(
      Math.ceil(REQUIRED_FRACTION * instance.spoptBestRegionCount)
    );
    // Objective: the construction has no annealing and grows compact regions before similar
    // ones (fewest unassigned neighbors first), so before the optional AZP improvement its
    // within-region sum of squares is above spopt's annealed result: the largest measured ratio is
    // 1.55 (delaunay-400, 20 rows per region), and the bound is 2.
    const objective = getWithinSumOfSquares(instance.values, result.labels);
    expect(objective, name).toBeLessThan(2 * instance.spoptBestObjective);
  }
});
