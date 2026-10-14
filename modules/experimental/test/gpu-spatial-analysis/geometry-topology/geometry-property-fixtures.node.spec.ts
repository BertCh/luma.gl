// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {expect, it} from 'vitest';
import {checkTriangulation} from '../delaunay-tessellation/delaunay-reference';
import {rayCrossesEdge, segmentCrossesEdge} from '../trajectory-zones/zone-crossing-rule';

// Node-side checks of the property fixtures written by test/oracles/geometry-properties.py: every
// fixture records versions, seed and tolerances, and the structural invariants the GPU specs rely on
// hold. The zone fixture is also replayed with the TypeScript rule (zone-crossing-rule.ts), which
// cross-validates it against the python fractions oracle without a GPU.

const ROOT = new URL('../', import.meta.url);

function load(path: string) {
  return JSON.parse(readFileSync(new URL(path, ROOT), 'utf8'));
}

const FIXTURES = {
  rollup: 'cell-aggregation/fixtures/cell-rollup.json',
  noding: 'geometry-topology/fixtures/line-noding.json',
  makeValid: 'geometry-topology/fixtures/make-valid.json',
  delaunay: 'delaunay-tessellation/fixtures/delaunay-properties.json',
  buffer: 'polygon-buffer/fixtures/buffer-properties.json',
  cover: 'cell-cover/fixtures/cell-cover-h3.json',
  zones: 'trajectory-zones/fixtures/zone-events-grid.json'
} as const;

it('every property fixture records generator, seed, package versions and tolerances', () => {
  for (const [name, path] of Object.entries(FIXTURES)) {
    const fixture = load(path);
    expect(fixture.generator, name).toBe(
      'modules/experimental/test/oracles/geometry-properties.py'
    );
    expect(typeof fixture.seed, name).toBe('number');
    for (const key of ['python', 'numpy', 'scipy', 'shapely', 'geos', 'h3']) {
      expect(typeof fixture.versions[key], `${name} versions.${key}`).toBe('string');
    }
    expect(Object.keys(fixture.tolerances).length, `${name} tolerances`).toBeGreaterThan(0);
    expect(typeof fixture.oracle, `${name} oracle`).toBe('string');
  }
});

it('line noding fixtures satisfy the count identity and keep noded vertices apart', () => {
  const fixture = load(FIXTURES.noding);
  expect(fixture.cases.length).toBe(4);
  for (const testCase of fixture.cases) {
    expect(testCase.nodedSegmentCount).toBe(
      testCase.inputSegmentCount + 2 * testCase.crossingCount
    );
    expect(testCase.segments.length).toBe(testCase.nodedSegmentCount);
    expect(testCase.minimumVertexSpacing).toBeGreaterThan(10 * fixture.tolerances.endpointDistance);
  }
});

it('delaunay fixtures satisfy Euler counts and scipy triangles pass the exact empty-circle check', () => {
  const fixture = load(FIXTURES.delaunay);
  expect(fixture.sets.length).toBe(3);
  for (const set of fixture.sets) {
    expect(set.triangleCount, set.name).toBe(2 * set.siteCount - 2 - set.hullCount);
    expect(set.coordinates.length, set.name).toBe(2 * set.siteCount);
    if (set.triangles) {
      expect(set.triangles.length, set.name).toBe(set.triangleCount);
      // scipy lists vertex ids sorted, so orientation is unknown here: orient each triple CCW first.
      const coordinates = Float32Array.from(set.coordinates);
      const oriented = set.triangles.map(([a, b, c]: number[]) => {
        const cross =
          (coordinates[2 * b] - coordinates[2 * a]) *
            (coordinates[2 * c + 1] - coordinates[2 * a + 1]) -
          (coordinates[2 * b + 1] - coordinates[2 * a + 1]) *
            (coordinates[2 * c] - coordinates[2 * a]);
        return cross > 0 ? [a, b, c] : [a, c, b];
      });
      const check = checkTriangulation(coordinates, oriented);
      expect(check, set.name).toEqual({
        degenerate: 0,
        circleViolations: 0,
        duplicateEdges: 0,
        areaDifference: 0n
      });
    }
  }
});

it('cell cover and rollup fixtures are sorted, unique and respect the center margin', () => {
  const cover = load(FIXTURES.cover);
  expect(cover.cases.length).toBe(8);
  expect(
    cover.cases.filter((testCase: {center: number[]}) => testCase.center[1] >= 60).length
  ).toBeGreaterThanOrEqual(3);
  for (const testCase of cover.cases) {
    expect(new Set(testCase.cells).size).toBe(testCase.cells.length);
    expect(testCase.resolution).toBeGreaterThanOrEqual(3);
    expect(testCase.resolution).toBeLessThanOrEqual(5);
    expect(testCase.minimumCenterDistance).toBeGreaterThanOrEqual(cover.tolerances.centerMargin);
  }
  const rollup = load(FIXTURES.rollup);
  const keys = rollup.source.map((row: {cell: string}) => BigInt(`0x${row.cell}`));
  expect(keys.every((key: bigint, index: number) => index === 0 || keys[index - 1] < key)).toBe(
    true
  );
  for (const testCase of rollup.cases) {
    const total = testCase.parents.reduce(
      (sum: number, row: {count: number}) => sum + row.count,
      0
    );
    expect(total).toBe(
      rollup.source.reduce((sum: number, row: {count: number}) => sum + row.count, 0)
    );
  }
});

it('zone fixture agrees with the TypeScript replay of the boundary rule', () => {
  const fixture = load(FIXTURES.zones);
  type Point = [number, number];
  expect(fixture.tracks.length).toBe(20);
  expect(fixture.zones.length).toBe(3);
  for (const track of fixture.tracks) {
    for (const [zoneIndex, expected] of track.zones.entries()) {
      const ring: Point[] = fixture.zones[zoneIndex];
      const edges = ring.map((start, index): [Point, Point] => [
        start,
        ring[(index + 1) % ring.length]
      ]);
      const positions: Point[] = track.positions;
      const inside = (point: Point) =>
        edges.filter(([a, b]) => rayCrossesEdge(point, a, b)).length % 2 === 1;
      expect(inside(positions[0])).toBe(expected.initialInside);
      let state = inside(positions[0]);
      let enters = 0;
      let exits = 0;
      for (let row = 1; row < positions.length; row++) {
        const crossings = edges.filter(([a, b]) =>
          segmentCrossesEdge(positions[row - 1], positions[row], a, b)
        ).length;
        for (let toggle = 0; toggle < crossings; toggle++) {
          state = !state;
          if (state) enters++;
          else exits++;
        }
      }
      expect({enters, exits, state}).toEqual({
        enters: expected.enters,
        exits: expected.exits,
        state: expected.finalInside
      });
    }
  }
});

it('make-valid and buffer fixtures carry stable structure and positive tolerances', () => {
  const makeValid = load(FIXTURES.makeValid);
  expect(makeValid.cases.length).toBe(10);
  for (const testCase of makeValid.cases) {
    expect(testCase.polygonCount).toBeGreaterThan(0);
    expect(testCase.areaTolerance).toBeGreaterThan(0);
    expect(testCase.samples.length).toBeGreaterThan(100);
  }
  const buffer = load(FIXTURES.buffer);
  expect(buffer.cases.filter((testCase: {kind: string}) => testCase.kind === 'line').length).toBe(
    6
  );
  expect(
    buffer.cases.filter((testCase: {kind: string}) => testCase.kind === 'polygon').length
  ).toBe(4);
  for (const testCase of buffer.cases) {
    expect(testCase.areaTolerance).toBeGreaterThan(0);
    expect(testCase.areaTolerance).toBeLessThan(0.02 * testCase.area);
  }
});
