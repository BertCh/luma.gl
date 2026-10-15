// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {lonLatFixtures} from '../../oracles/parity-wp2-round5-lonLat-fixtures';
import {outlineFixtures} from '../../oracles/parity-wp2-round5-outline-fixtures';
import {rectangleFixtures} from '../../oracles/parity-wp2-round5-rectangle-fixtures';
import {compareParity} from '../../oracles/parity-wp2-assert';
import {prepareLonLatPolygons} from './lon-lat-polygon-oracle';
import {buildOutlineTriangles, type Point} from '../outline-geometry/outline-geometry-oracle';
import {clipLines} from '../rectangle-clip/rectangle-clip-oracle';
import seamFixture from '../../oracles/fixtures/parity-wp2-great-circle-arcs-seam-regression.json';

// Convention bridges only: GPU specs consume GEOS/PROJ fixtures directly.
for (const fixture of lonLatFixtures)
  for (const entry of fixture.cases) {
    it(`GEOS preparation convention bridge ${entry.name}`, () => {
      const actual = prepareLonLatPolygons(
        {
          positions: Float64Array.from(entry.positions),
          featureOffsets: Uint32Array.from(entry.featureOffsets),
          polygonOffsets: Uint32Array.from(entry.polygonOffsets),
          ringOffsets: Uint32Array.from(entry.ringOffsets)
        },
        {antimeridian: entry.reject ? 'reject' : 'split', maximumSegmentDegrees: 0}
      );
      for (const field of ['featureOffsets', 'polygonOffsets', 'ringOffsets'] as const)
        compareParity(
          'GPULonLatPolygonPreparation',
          entry.name,
          fixture.reference,
          `node convention ${field}`,
          actual[field],
          entry.expected[field]
        );
      compareParity(
        'GPULonLatPolygonPreparation',
        entry.name,
        fixture.reference,
        'node flags',
        actual.ringFlags,
        entry.ringFlags
      );
      compareParity(
        'GPULonLatPolygonPreparation',
        entry.name,
        fixture.reference,
        'node all status fields',
        [
          actual.status.count,
          actual.status.requiredCount,
          actual.status.overflow,
          actual.status.invalidCount,
          actual.status.approximation
        ],
        [
          entry.expected.positions.length / 2,
          entry.expected.positions.length / 2,
          0,
          entry.invalidCount,
          0
        ]
      );
      // Independent ring start rotation; preserve winding and all vertices.
      const aligned: number[] = [];
      for (let ring = 0; ring + 1 < entry.expected.ringOffsets.length; ring++) {
        const start = entry.expected.ringOffsets[ring];
        const end = entry.expected.ringOffsets[ring + 1];
        let closest = start;
        const distance = (index: number) =>
          Math.hypot(
            actual.positions[2 * start] - entry.expected.positions[2 * index],
            actual.positions[2 * start + 1] - entry.expected.positions[2 * index + 1]
          );
        for (let row = start; row < end; row++)
          if (distance(row) < distance(closest)) closest = row;
        for (let row = start; row < end; row++) {
          const index = start + ((closest - start + row - start) % (end - start));
          aligned.push(
            entry.expected.positions[2 * index],
            entry.expected.positions[2 * index + 1]
          );
        }
      }
      compareParity(
        'GPULonLatPolygonPreparation',
        entry.name,
        fixture.reference,
        'node cyclic coordinates',
        actual.positions,
        aligned,
        1e-9
      );
    });
  }
for (const fixture of outlineFixtures)
  for (const entry of fixture.cases) {
    it(`GEOS outline convention bridge ${entry.name}`, () => {
      const vertices: Point[] = Array.from({length: entry.positions.length / 2}, (_, index) => [
        entry.positions[2 * index],
        entry.positions[2 * index + 1]
      ]);
      const geometryType = entry.geometryType;
      if (geometryType !== 'points' && geometryType !== 'lines' && geometryType !== 'polygons')
        throw new Error('Unexpected fixture geometry');
      for (const frame of entry.frames)
        compareParity(
          'GPUOutlineGeometry',
          entry.name,
          fixture.reference,
          'node all coordinates',
          buildOutlineTriangles(
            vertices,
            entry.lineOffsets,
            geometryType,
            frame.distance,
            16
          ).flat(),
          frame.positions,
          1e-9
        );
    });
  }
for (const fixture of rectangleFixtures)
  for (const entry of fixture.cases) {
    it(`GEOS rectangle convention bridge ${entry.name}`, () => {
      const vertices: Point[] = Array.from({length: entry.positions.length / 2}, (_, index) => [
        entry.positions[2 * index],
        entry.positions[2 * index + 1]
      ]);
      const lines = entry.lineOffsets
        .slice(1)
        .map((end, index) => vertices.slice(entry.lineOffsets[index], end));
      for (const frame of entry.frames) {
        const [minX, minY, maxX, maxY] = frame.rectangle;
        const actual = clipLines(lines, {minX, minY, maxX, maxY});
        compareParity(
          'GPURectangleClip',
          entry.name,
          fixture.reference,
          'node all coordinates',
          actual.lines.flat(2),
          frame.positions,
          1e-9
        );
        compareParity(
          'GPURectangleClip',
          entry.name,
          fixture.reference,
          'node source lines',
          actual.sourceLineIndices,
          frame.sourceLineIndices
        );
      }
    });
  }
it('short seam separation loses bits before wrapping, independently of vertex count', () => {
  const entry = seamFixture.cases[0];
  for (const row of [0, 1]) {
    const source = Math.fround(entry.sources[2 * row]);
    const target = Math.fround(entry.targets[2 * row]);
    const difference = target - source;
    const expected = difference + 360;
    const old = Math.fround(Math.fround(difference) + 360);
    const stable = Math.fround(Math.fround(target + 180) - Math.fround(source - 180));
    expect(Math.abs(old - expected)).toBeGreaterThan(1e-5);
    expect(stable).toBe(expected);
    const sparse = entry.frames[0];
    const dense = entry.frames[1];
    expect(sparse.measures[sparse.lineOffsets[row + 1] - 1]).toBe(
      dense.measures[dense.lineOffsets[row + 1] - 1]
    );
  }
});
