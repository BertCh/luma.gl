// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {runLineNoding, type Point} from './topology-property-harness';
import fixtureJson from './fixtures/line-noding.json';
import {requireTestDevice} from '../../utils/require-test-device';

// Oracle: fixtures/line-noding.json, written by test/oracles/geometry-properties.py. The expected
// segments are the consecutive vertex pairs of shapely.node(MultiLineString); the fixture states the
// count identity (input segments + 2 * crossings) and the endpoint tolerance with its derivation.

type Fixture = {
  tolerances: {endpointDistance: number};
  cases: {
    lines: Point[][];
    inputSegmentCount: number;
    crossingCount: number;
    nodedSegmentCount: number;
    segments: [Point, Point][];
  }[];
};

const fixture: Fixture = fixtureJson as unknown as Fixture;

function distance(first: readonly number[], second: readonly number[]): number {
  return Math.max(Math.abs(first[0] - second[0]), Math.abs(first[1] - second[1]));
}

it('GPULineNoding matches the shapely.node segment multiset on random linework', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const tolerance = fixture.tolerances.endpointDistance;
  for (const [index, testCase] of fixture.cases.entries()) {
    const label = `case ${index}`;
    expect(testCase.nodedSegmentCount, `${label} fixture identity`).toBe(
      testCase.inputSegmentCount + 2 * testCase.crossingCount
    );
    const result = await runLineNoding(device, testCase.lines, 512, 2048);
    expect(result.overflow, `${label} overflow`).toBe(0);
    expect(result.candidateOverflow, `${label} candidate overflow`).toBe(0);
    // Counts are exact.
    expect(result.requiredCount, `${label} required`).toBe(testCase.nodedSegmentCount);
    expect(result.count, `${label} count`).toBe(testCase.nodedSegmentCount);
    // Multiset match of undirected segments: noded vertices are at least 0.05 apart, far above the
    // tolerance, so each expected segment has exactly one candidate.
    const unmatched = new Set(result.segments.keys());
    for (const [start, end] of testCase.segments) {
      let found = -1;
      for (const row of unmatched) {
        const [x0, y0, x1, y1] = result.segments[row].endpoints;
        const forward = Math.max(distance([x0, y0], start), distance([x1, y1], end));
        const backward = Math.max(distance([x0, y0], end), distance([x1, y1], start));
        if (Math.min(forward, backward) <= tolerance) {
          found = row;
          break;
        }
      }
      expect(found, `${label} segment ${start} -> ${end}`).toBeGreaterThanOrEqual(0);
      unmatched.delete(found);
    }
    expect(unmatched.size, `${label} unexpected segments`).toBe(0);
  }
  device.destroy?.();
});
