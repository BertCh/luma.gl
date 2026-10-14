// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  checkTriangulation,
  createSymbolicPredicates,
  createUniformFixture,
  digestTriangleSet,
  incircleExact,
  mulberry32,
  orientExact,
  replaySymbolicDelaunay,
  SCIPY_FIXTURES
} from './delaunay-reference';

describe('symbolic super vertices against a finite exact configuration', () => {
  // A real configuration realizing the symbolic geometry: tilt e = 1/1024 (directions scaled by
  // 1024) and radii that satisfy R_0 e >> scale, R_(k+1) >> R_k^2 / e.
  const DIRECTIONS = [
    [1024n, -1n],
    [1n, 1024n],
    [-1024n, -1n]
  ];
  const RADII = [2n ** 30n, 2n ** 90n, 2n ** 200n];

  function compareWithFinite(coordinates: Float32Array) {
    const pointCount = coordinates.length / 2;
    const rows: bigint[][] = [];
    for (let row = 0; row < pointCount; row++) {
      rows.push([BigInt(coordinates[2 * row]), BigInt(coordinates[2 * row + 1])]);
    }
    for (let k = 0; k < 3; k++) {
      rows.push([RADII[k] * DIRECTIONS[k][0], RADII[k] * DIRECTIONS[k][1]]);
    }
    const {orientSymbolic, isInsideCircumcircle} = createSymbolicPredicates(coordinates);
    let incircleChecks = 0;
    let orientChecks = 0;
    for (let a = 0; a < rows.length; a++) {
      for (let b = 0; b < rows.length; b++) {
        if (a === b) {
          continue;
        }
        for (let p = 0; p < pointCount; p++) {
          if (p === a || p === b) {
            continue;
          }
          expect(orientSymbolic(a, b, p)).toBe(orientExact(rows[a], rows[b], rows[p]));
          orientChecks++;
        }
        for (let c = 0; c < rows.length; c++) {
          if (c === a || c === b || orientExact(rows[a], rows[b], rows[c]) <= 0) {
            continue;
          }
          const hasSuper = Math.max(a, b, c) >= pointCount;
          for (let p = 0; p < pointCount; p++) {
            if (p === a || p === b || p === c) {
              continue;
            }
            const finite = incircleExact(rows[a], rows[b], rows[c], rows[p]);
            if (hasSuper) {
              // A symbolic site is never exactly on a circle through a super vertex.
              expect(finite).not.toBe(0);
            }
            expect(isInsideCircumcircle(a, b, c, p)).toBe(hasSuper ? finite > 0 : finite >= 0);
            incircleChecks++;
          }
        }
      }
    }
    return {incircleChecks, orientChecks};
  }

  it('matches exact orientation and in-circle on random integer sites with collinear and cocircular ties', () => {
    const random = mulberry32(7);
    for (let trial = 0; trial < 6; trial++) {
      const count = 8;
      const coordinates = new Float32Array(2 * count);
      const seen = new Set<string>();
      for (let row = 0; row < count; ) {
        const x = Math.floor(random() * 7) - 3;
        const y = Math.floor(random() * 7) - 3;
        if (!seen.has(`${x},${y}`)) {
          seen.add(`${x},${y}`);
          coordinates[2 * row] = x;
          coordinates[2 * row + 1] = y;
          row++;
        }
      }
      const {incircleChecks} = compareWithFinite(coordinates);
      expect(incircleChecks).toBeGreaterThan(500);
    }
  });

  it('matches on a full lattice, where every hull edge is axis-parallel and collinear', () => {
    const coordinates = Float32Array.from([-1, 0, 1].flatMap(y => [-1, 0, 1].flatMap(x => [x, y])));
    compareWithFinite(coordinates);
  });
});

describe('symbolic Delaunay replay against scipy', () => {
  for (const [name, fixture] of Object.entries(SCIPY_FIXTURES)) {
    it(`reproduces the scipy triangle set on ${name} and keeps every hull triangle`, () => {
      const coordinates = createUniformFixture(
        fixture.seed,
        fixture.count,
        fixture.width,
        fixture.height
      );
      const triangles = replaySymbolicDelaunay(coordinates);
      // T = 2n - 2 - h with h hull vertices (generic position: no hull collinearity).
      expect(triangles.length).toBe(2 * fixture.count - 2 - fixture.hull);
      expect(triangles.length).toBe(fixture.triangles);
      expect(digestTriangleSet(triangles)).toBe(fixture.digest);
      expect(checkTriangulation(coordinates, triangles)).toEqual({
        degenerate: 0,
        circleViolations: 0,
        duplicateEdges: 0,
        areaDifference: 0n
      });
    });
  }
});

describe('symbolic Delaunay replay on degenerate input', () => {
  function shuffled<T>(items: readonly T[], seed: number): T[] {
    const random = mulberry32(seed);
    const result = [...items];
    for (let index = result.length - 1; index > 0; index--) {
      const other = Math.floor(random() * (index + 1));
      [result[index], result[other]] = [result[other], result[index]];
    }
    return result;
  }
  const flatten = (points: readonly (readonly number[])[]) => Float32Array.from(points.flat());

  it('triangulates 12 cocircular sites into n - 2 triangles in any order', () => {
    // Derived: every site is a hull vertex, so T = 2n - 2 - n = 10 whatever diagonals the ties pick.
    const circle = [
      [5, 0],
      [4, 3],
      [3, 4],
      [0, 5],
      [-3, 4],
      [-4, 3],
      [-5, 0],
      [-4, -3],
      [-3, -4],
      [0, -5],
      [3, -4],
      [4, -3]
    ];
    for (const seed of [1, 2, 3, 4]) {
      const coordinates = flatten(seed === 1 ? circle : shuffled(circle, seed));
      const triangles = replaySymbolicDelaunay(coordinates);
      expect(triangles.length).toBe(10);
      expect(checkTriangulation(coordinates, triangles)).toEqual({
        degenerate: 0,
        circleViolations: 0,
        duplicateEdges: 0,
        areaDifference: 0n
      });
    }
  });

  it('keeps sites on hull edges: a square with edge midpoints and a centre gives 2n - 2 - 8 = 8 triangles', () => {
    // Boundary points b = 8 (4 corners + 4 midpoints), n = 9, T = 2n - 2 - b = 8 (scipy agrees).
    const square = [
      [0, 0],
      [2, 0],
      [2, 2],
      [0, 2],
      [1, 1],
      [1, 0],
      [2, 1],
      [1, 2],
      [0, 1]
    ];
    const midpointsFirst = [
      [1, 0],
      [2, 1],
      [1, 2],
      [0, 1],
      [0, 0],
      [2, 0],
      [2, 2],
      [0, 2],
      [1, 1]
    ];
    for (const points of [square, midpointsFirst, shuffled(square, 11), shuffled(square, 12)]) {
      const coordinates = flatten(points);
      const triangles = replaySymbolicDelaunay(coordinates);
      expect(triangles.length).toBe(8);
      expect(checkTriangulation(coordinates, triangles)).toEqual({
        degenerate: 0,
        circleViolations: 0,
        duplicateEdges: 0,
        areaDifference: 0n
      });
    }
  });

  it('triangulates a 5x5 lattice into 32 triangles (scipy count; boundary points 16)', () => {
    const lattice = Array.from({length: 25}, (_, index) => [index % 5, Math.floor(index / 5)]);
    for (const points of [lattice, shuffled(lattice, 21)]) {
      const coordinates = flatten(points);
      const triangles = replaySymbolicDelaunay(coordinates);
      expect(triangles.length).toBe(32);
      expect(checkTriangulation(coordinates, triangles)).toEqual({
        degenerate: 0,
        circleViolations: 0,
        duplicateEdges: 0,
        areaDifference: 0n
      });
    }
  });

  it('emits no triangle for collinear sites and ignores duplicate rows', () => {
    expect(replaySymbolicDelaunay(Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0))).toEqual([]);
    expect(replaySymbolicDelaunay(Float32Array.of(3, 3, 1, 1, 2, 2, 0, 0, 5, 5))).toEqual([]);
    const withDuplicate = replaySymbolicDelaunay(
      Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1, 0.5, 0.5, 0, 0)
    );
    expect(withDuplicate.length).toBe(4);
  });
});
