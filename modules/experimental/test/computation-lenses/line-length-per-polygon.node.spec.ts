// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {statSync} from 'node:fs';
import {describe, expect, test} from 'vitest';
import {loadLineLengthPerPolygonInputs} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/line-length-per-polygon.lens';
import {loadS3ChicagoRoads} from '../../src/computation-lenses/scenarios/gpu-spatial-analysis/s3-lens-data';
import type {Point} from '../gpu-spatial-analysis/line-density/line-density-oracle';
import {computeLineLengthPerPolygon} from '../gpu-spatial-analysis/line-density/line-length-per-polygon-oracle';

function getPaths(positions: Float32Array, offsets: Uint32Array): Point[][] {
  return Array.from({length: offsets.length - 1}, (_, path) =>
    Array.from({length: offsets[path + 1] - offsets[path]}, (_, vertex): Point => {
      const row = offsets[path] + vertex;
      return [positions[2 * row], positions[2 * row + 1]];
    })
  );
}

describe('line length per polygon lens data', () => {
  test('the original zero lengths reflect the street extent, not frame or winding', async () => {
    const originalRoads = await loadS3ChicagoRoads();
    const {roads, selectedPositions, selectedOffsets} = await loadLineLengthPerPolygonInputs();
    expect(originalRoads.projected).toEqual(roads.projected);
    const lines = getPaths(originalRoads.positionsProjected, originalRoads.lineOffsets);
    const rings = getPaths(selectedPositions, selectedOffsets);
    for (const reverse of [false, true]) {
      const polygons = rings.map(ring => [[reverse ? [...ring].reverse() : ring]]);
      const result = computeLineLengthPerPolygon(lines, undefined, polygons);
      expect(Array.from(result.lengths.slice(0, 2))).toEqual([0, 0]);
      expect(result.lengths[2]).toBeCloseTo(27206.7, 0);
      expect(Array.from(result.segmentCounts)).toEqual([0, 0, 290]);
    }
  });

  test('the selected streets contribute to all three areas and cross their boundaries', async () => {
    const {roads, selectedNames, selectedPositions, selectedOffsets} =
      await loadLineLengthPerPolygonInputs();
    expect(selectedNames).toEqual(['area 7', 'area 22', 'area 24']);
    expect(roads.projected).toMatchObject({
      crs: 'EPSG:32616',
      unit: 'm',
      falseOrigin: [440000, 4630000]
    });
    const lines = getPaths(roads.positionsProjected, roads.lineOffsets);
    const polygons = getPaths(selectedPositions, selectedOffsets).map(ring => [[ring]]);
    const result = computeLineLengthPerPolygon(lines, undefined, polygons);
    expect(lines).toHaveLength(448);
    for (const [index, expected] of [7845.95, 27341.81, 22321.82].entries()) {
      expect(result.lengths[index]).toBeCloseTo(expected, 0);
    }
    expect(Array.from(result.segmentCounts)).toEqual([77, 329, 270]);

    // Clipped lengths cannot exceed the input; segments crossing an area boundary count twice.
    let totalLength = 0;
    let segmentCount = 0;
    for (const line of lines) {
      for (let vertex = 1; vertex < line.length; vertex++) {
        totalLength += Math.hypot(
          line[vertex][0] - line[vertex - 1][0],
          line[vertex][1] - line[vertex - 1][1]
        );
        segmentCount++;
      }
    }
    expect(result.lengths.reduce((sum, length) => sum + length, 0)).toBeLessThanOrEqual(
      totalLength
    );
    expect(result.segmentCounts.reduce((sum, count) => sum + count, 0)).toBeGreaterThan(
      segmentCount
    );
    // Even every segment paired with every area fits the example's candidate capacity.
    expect(segmentCount * polygons.length).toBeLessThan(4096);
  });

  test('the dedicated extract fits the docs data budget', () => {
    const path = new URL(
      '../../src/computation-lenses/data/s3-chicago-area-roads.json',
      import.meta.url
    );
    expect(statSync(path).size).toBeLessThanOrEqual(150 * 1024);
  });
});
