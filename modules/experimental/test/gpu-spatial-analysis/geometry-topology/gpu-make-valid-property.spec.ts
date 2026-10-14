// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {runMakeValid, type Point} from './topology-property-harness';
import fixtureJson from './fixtures/make-valid.json';
import {requireTestDevice} from '../../utils/require-test-device';

// Oracle: fixtures/make-valid.json, written by test/oracles/geometry-properties.py with
// shapely.make_valid(Polygon(shell, holes), method='linework'). Six shuffled-vertex polygons and four
// polygons whose hole ring crosses the shell. Area and polygon count come from shapely; the
// symmetric difference is checked through membership of sample points that are farther than the
// fixture margin from the repaired boundary (see the fixture's tolerance notes).

type Fixture = {
  tolerances: {margin: number; delta: number};
  cases: {
    kind: string;
    rings: Point[][];
    area: number;
    polygonCount: number;
    holeCount: number;
    boundaryLength: number;
    areaTolerance: number;
    samples: [number, number, number][];
  }[];
};

const fixture: Fixture = fixtureJson as unknown as Fixture;

function getRingArea(ring: readonly number[][]): number {
  let doubled = 0;
  for (let index = 0; index < ring.length; index++) {
    const [x0, y0] = ring[index];
    const [x1, y1] = ring[(index + 1) % ring.length];
    doubled += x0 * y1 - x1 * y0;
  }
  return Math.abs(doubled) / 2;
}

/** Number of vertices an (open or closed) ring visits more than once. */
function countShellPinches(ring: readonly number[][]): number {
  const open =
    ring.length > 1 &&
    ring[0][0] === ring[ring.length - 1][0] &&
    ring[0][1] === ring[ring.length - 1][1]
      ? ring.slice(0, -1)
      : ring;
  const visits = new Map<string, number>();
  for (const [x, y] of open) {
    const key = `${x},${y}`;
    visits.set(key, (visits.get(key) ?? 0) + 1);
  }
  let pinches = 0;
  for (const count of visits.values()) {
    pinches += count - 1;
  }
  return pinches;
}

function isInsideRing(x: number, y: number, ring: readonly number[][]): boolean {
  let inside = false;
  for (let index = 0, previous = ring.length - 1; index < ring.length; previous = index++) {
    const [ax, ay] = ring[previous];
    const [bx, by] = ring[index];
    if (ay > y !== by > y && x < ((bx - ax) * (y - ay)) / (by - ay) + ax) {
      inside = !inside;
    }
  }
  return inside;
}

it('GPUMakeValid matches shapely.make_valid linework on random invalid polygons', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [index, testCase] of fixture.cases.entries()) {
    const label = `${testCase.kind} ${index}`;
    const result = await runMakeValid(device, testCase.rings, {
      segments: 512,
      intersections: 2048,
      rings: 128,
      vertices: 2048
    });
    expect(result.overflow, `${label} overflow`).toBe(0);
    expect(result.candidateOverflow, `${label} candidate overflow`).toBe(0);
    expect(result.polygons.length, `${label} polygon count`).toBe(testCase.polygonCount);
    // GEOS writes a hole that touches its shell at one vertex as a hole; the GPU ring assembly may
    // instead walk it into the shell as a pinch (the shell visits that vertex twice). Both describe
    // the same point set (the area and membership checks below compare it), so each pinch counts as
    // one hole.
    const holeCount = result.polygons.reduce(
      (total, rings) => total + rings.length - 1 + countShellPinches(rings[0]),
      0
    );
    expect(holeCount, `${label} hole count`).toBe(testCase.holeCount);
    const area = result.polygons.reduce(
      (total, rings) =>
        total +
        rings.reduce(
          (sum, ring, ringIndex) => sum + (ringIndex === 0 ? 1 : -1) * getRingArea(ring),
          0
        ),
      0
    );
    expect(
      Math.abs(area - testCase.area),
      `${label} area ${area} vs ${testCase.area}`
    ).toBeLessThan(testCase.areaTolerance);
    // Symmetric difference: every sample outside the margin of the repaired boundary must agree.
    const mismatches = testCase.samples.filter(([x, y, expected]) => {
      const inside = result.polygons.some(
        rings =>
          isInsideRing(x, y, rings[0]) && !rings.slice(1).some(hole => isInsideRing(x, y, hole))
      );
      return inside !== (expected === 1);
    });
    expect(mismatches, `${label} membership mismatches`).toEqual([]);
  }
  device.destroy?.();
});
