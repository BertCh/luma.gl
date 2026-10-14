// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import largeFixture from '../../oracles/fixtures/geometry-triangulation-large.json';
import {
  checkTriangulation,
  createLargePolygonRings,
  getRingsChecksum,
  type LargePolygonDescriptor
} from './polygon-triangulation-monotone-cases';
import {triangulateMonotoneModel} from './polygon-triangulation-monotone-model';

// The large polygons of `test/oracles/geometry-triangulation-large.py` are rebuilt from integer
// descriptors (the fixture holds no coordinates). This spec proves that the TypeScript twins of
// the generators reproduce the polygons Shapely measured (vertex counts and coordinate
// checksums), and that the CPU model of the monotone algorithm reproduces the Shapely and
// constrained Delaunay areas and counts on the cases small enough for a node spec. The GPU spec
// runs the same polygons through the kernels.

type FixtureCase = LargePolygonDescriptor & {
  vertexCount: number;
  holeCount: number;
  triangleCount: number;
  checksum: number;
  area: number;
  valid: boolean;
  cdtArea: number;
  cdtTriangleCount: number;
};

const cases = largeFixture.cases as FixtureCase[];

it('the oracle fixture records Shapely-valid polygons with consistent constrained Delaunay results', () => {
  expect(largeFixture.generator).toBe(
    'modules/experimental/test/oracles/geometry-triangulation-large.py'
  );
  expect(largeFixture.versions.shapely).toMatch(/^2\./);
  expect(cases.length).toBeGreaterThanOrEqual(6);
  const sizes = cases.map(entry => entry.vertexCount);
  expect(Math.min(...sizes)).toBeGreaterThanOrEqual(5000);
  expect(Math.max(...sizes)).toBeGreaterThanOrEqual(200_000);
  for (const entry of cases) {
    expect(entry.valid, entry.name).toBe(true);
    expect(entry.triangleCount, entry.name).toBe(entry.vertexCount + 2 * entry.holeCount - 2);
    expect(entry.cdtTriangleCount, entry.name).toBe(entry.triangleCount);
    expect(Math.abs(entry.cdtArea - entry.area) / entry.area, entry.name).toBeLessThan(1e-9);
  }
});

it('the TypeScript polygon builders reproduce the polygons of the oracle generator', () => {
  for (const entry of cases) {
    const rings = createLargePolygonRings(entry);
    expect(rings.flat().length, `${entry.name} vertices`).toBe(entry.vertexCount);
    expect(rings.length - 1, `${entry.name} holes`).toBe(entry.holeCount);
    expect(getRingsChecksum(rings), `${entry.name} checksum`).toBe(entry.checksum);
  }
});

it('the CPU model matches Shapely and the constrained Delaunay area on the fixture polygons', () => {
  for (const entry of cases.filter(candidate => candidate.vertexCount <= 20_000)) {
    const rings = createLargePolygonRings(entry);
    const result = triangulateMonotoneModel(rings);
    expect(result.failure, entry.name).toBeUndefined();
    expect(result.triangles.length, entry.name).toBe(entry.triangleCount);
    const check = checkTriangulation(rings, result.triangles);
    expect(check.minimumSignedArea, entry.name).toBeGreaterThanOrEqual(0);
    expect(Math.abs(check.area - entry.area) / entry.area, `${entry.name} Shapely`).toBeLessThan(
      1e-9
    );
    expect(Math.abs(check.area - entry.cdtArea) / entry.cdtArea, `${entry.name} CDT`).toBeLessThan(
      1e-9
    );
    expect(check.boundaryMatches, entry.name).toBe(true);
  }
}, 300_000);
