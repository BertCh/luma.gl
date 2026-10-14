// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeVariogramOnCPU} from './variogram-oracle';
import {getVariogramReferenceMismatches, VARIOGRAM_REFERENCE} from './variogram-reference';

// The f64 CPU mirror of GPUVariogram's inclusion and binning rules against scikit-gstat 1.0.24
// (Q4, B-21), from the committed generator test/oracles/dataframe-variogram.py.

it('variogram fixture records its reference packages and a safe lag margin', () => {
  const {versions, generator, nearestEdgeDistance, maximumDistance} = VARIOGRAM_REFERENCE;
  expect(generator).toBe('test/oracles/dataframe-variogram.py');
  expect(versions['scikit-gstat']).toBe('1.0.24');
  // f32 distances (relative 2^-23) cannot move a pair across an edge this far away.
  expect(nearestEdgeDistance).toBeGreaterThan(maximumDistance * 2 ** -16);
});

it('variogram oracle matches scikit-gstat Matheron and Cressie-Hawkins bins', () => {
  const {positions, bounds, maximumDistance, lagCount, scenes} = VARIOGRAM_REFERENCE;
  const scenePositions = Float32Array.from(positions);
  for (const [name, scene] of Object.entries(scenes)) {
    const result = computeVariogramOnCPU(
      {positions: scenePositions, values: Float32Array.from(scene.values)},
      {bounds, maximumDistance},
      lagCount
    );
    expect(
      result.edgePairCounts.every(count => count === 0),
      name
    ).toBe(true);
    expect(getVariogramReferenceMismatches(name, result, scene)).toEqual([]);
  }
});
