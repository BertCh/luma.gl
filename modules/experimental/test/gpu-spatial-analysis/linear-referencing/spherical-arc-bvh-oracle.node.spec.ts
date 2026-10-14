// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createSphericalBVHScene, getNearestArcCandidates} from './spherical-arc-bvh-oracle';

it('distinguishes shared-endpoint ties from ties between different lines', () => {
  const result = getNearestArcCandidates(
    [0, -1],
    [
      [-1, 1],
      [0, 0],
      [1, 1],
      [10, 0],
      [11, 0]
    ],
    [0, 3, 5]
  );
  expect(result.secondDistance).toBe(result.best.distance);
  expect(result.competingLineDistance - result.best.distance).toBeGreaterThan(15);
  expect(result.best.line).toBe(0);
});

it('the spherical BVH scene has mostly unambiguous lines despite adjacent segment ties', () => {
  const {vertices, lineOffsets, points} = createSphericalBVHScene();
  let segmentTies = 0;
  let lineTies = 0;
  for (const point of points) {
    const result = getNearestArcCandidates(point, vertices, lineOffsets);
    if (result.secondDistance - result.best.distance <= 15) segmentTies++;
    if (result.competingLineDistance - result.best.distance <= 15) lineTies++;
  }
  expect(segmentTies).toBe(1005);
  expect(lineTies).toBeLessThan(points.length * 0.2);
}, 120_000);
