// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {createGeometryProjection} from '../../../src/computation-views/views/geometry-projection';

describe('native geometry projection', () => {
  test('preserves a 2:1 aspect, centres and reverses y', () => {
    const projection = createGeometryProjection({
      coordinateModel: 'projected',
      points: [
        [0, 0],
        [2, 1]
      ],
      width: 100,
      height: 100,
      padding: 0,
      unit: 'm'
    });
    expect(projection.project(0, 0)).toEqual([0, 75]);
    expect(projection.project(2, 1)).toEqual([100, 25]);
    expect(projection.statement).toContain('(m)');
  });
  test('states geographic scaling and limitations', () => {
    const projection = createGeometryProjection({
      coordinateModel: 'geographic',
      points: [
        [0, 59],
        [2, 61]
      ],
      width: 200,
      height: 200,
      padding: 0
    });
    expect(projection.statement).toContain('cos 60°');
    expect(projection.statement).toContain('not equal-area');
    expect(projection.project(2, 61)[0] - projection.project(0, 59)[0]).toBeCloseTo(100);
  });
  test('states unknown coordinates explicitly', () => {
    expect(
      createGeometryProjection({coordinateModel: 'unknown', points: [], width: 100, height: 100})
        .statement
    ).toBe('Coordinate model unknown: drawn as planar x, y.');
  });
  test('expands degenerate extents and ignores non-finite coordinates', () => {
    const projection = createGeometryProjection({
      coordinateModel: 'projected',
      points: [
        [5, 8],
        [NaN, Infinity]
      ],
      width: 100,
      height: 80
    });
    expect(projection.extent).toEqual([
      [4, 7],
      [6, 9]
    ]);
    expect(projection.project(5, 8)).toEqual([50, 40]);
    const empty = createGeometryProjection({
      coordinateModel: 'unknown',
      points: [],
      width: 100,
      height: 100
    });
    expect(empty.extent).toEqual([
      [-1, -1],
      [1, 1]
    ]);
  });
});
