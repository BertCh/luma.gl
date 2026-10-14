// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {getDelaunayStepCount} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation';
import {
  checkLocallyDelaunay,
  checkTriangulation,
  countHullBoundarySites,
  createCircleFixture,
  createClusteredFixture,
  createLatticeFixture,
  createLineFixture,
  createSortedFixture,
  createUniformFixture,
  digestTriangleSet,
  SCIPY_FIXTURES,
  toExactIntegers
} from './delaunay-reference';
import {
  createFlipPredicates,
  getFlipPriority,
  getInsertionPriority,
  triangulateParallelReference,
  type WinnerOrder
} from './delaunay-parallel-reference';
import fixtureJson from './fixtures/delaunay-properties.json';

// CPU mirror of the GPU insertion-and-flipping algorithm (delaunay-parallel-reference.ts). These
// specs validate the GPU logic: counts, scipy digests, the exact empty circle, the Simulation of
// Simplicity tie-break, and the step budget `getDelaunayStepCount`.

const CLEAN = {degenerate: 0, circleViolations: 0, duplicateEdges: 0, areaDifference: 0n};

function canonicalSet(rows: readonly (readonly number[])[]): string[] {
  return rows
    .map(row => [...row].sort((a, b) => a - b).join(','))
    .sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
}

describe('parallel Delaunay mirror: scipy fixtures', () => {
  for (const [name, fixture] of Object.entries(SCIPY_FIXTURES)) {
    it(`matches scipy on ${name} (T = 2n - 2 - h, digest, empty circle)`, () => {
      const coordinates = createUniformFixture(
        fixture.seed,
        fixture.count,
        fixture.width,
        fixture.height
      );
      const result = triangulateParallelReference(coordinates, {validate: true});
      expect(result.failed).toBe(false);
      expect(result.invalidCount).toBe(0);
      expect(result.requiredCount).toBe(fixture.triangles);
      expect(result.requiredCount).toBe(2 * fixture.count - 2 - fixture.hull);
      expect(digestTriangleSet(result.triangles)).toBe(fixture.digest);
      expect(checkTriangulation(coordinates, result.triangles)).toEqual(CLEAN);
      expect(result.activeSteps).toBeLessThanOrEqual(getDelaunayStepCount(fixture.count));
    });
  }

  type SiteSet = {
    name: string;
    coordinates: number[];
    siteCount: number;
    hullCount: number;
    triangleCount: number;
    generalPosition: boolean;
    triangles: number[][] | null;
  };
  for (const set of (fixtureJson as unknown as {sets: SiteSet[]}).sets) {
    it(`matches scipy on fixtures/delaunay-properties.json ${set.name}`, () => {
      const coordinates = Float32Array.from(set.coordinates);
      const result = triangulateParallelReference(coordinates, {validate: true});
      expect(result.failed).toBe(false);
      expect(result.requiredCount).toBe(set.triangleCount);
      expect(result.requiredCount).toBe(2 * set.siteCount - 2 - set.hullCount);
      expect(checkTriangulation(coordinates, result.triangles)).toEqual(CLEAN);
      if (set.generalPosition && set.triangles) {
        expect(canonicalSet(result.triangles)).toEqual(canonicalSet(set.triangles));
      }
    });
  }
});

describe('parallel Delaunay mirror: degenerate cases', () => {
  it('collinear sites produce zero triangles without failing', () => {
    const result = triangulateParallelReference(Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0), {
      validate: true
    });
    expect(result).toMatchObject({requiredCount: 0, failed: false, invalidCount: 0});
  });

  it('bootstraps a lone apex regardless of row order, line direction or duplicate extremes', () => {
    for (const transform of [
      (along: number, away: number) => [along, away],
      (along: number, away: number) => [away, along],
      (along: number, away: number) => [along + away, along - away]
    ]) {
      const sites = Array.from({length: 2000}, (_, row) => transform(row, 0));
      sites.splice(973, 0, transform(1000, 1));
      sites.push(sites[0], sites[1999]);
      for (const rows of [sites, [...sites].reverse()]) {
        const coordinates = Float32Array.from(rows.flat());
        const result = triangulateParallelReference(coordinates);
        expect(result).toMatchObject({
          failed: false,
          invalidCount: 0,
          requiredCount: 1999,
          duplicateCount: 2
        });
        expect(result.activeSteps).toBeLessThan(100);
        expect(checkLocallyDelaunay(coordinates, result.triangles)).toEqual(CLEAN);
      }
    }
  });

  it('one, two and three sites', () => {
    for (const [coordinates, expected] of [
      [Float32Array.of(1, 1), 0],
      [Float32Array.of(1, 1, 2, 2), 0],
      [Float32Array.of(0, 0, 1, 0, 0, 1), 1],
      [Float32Array.of(0, 0, 1, 0, 0, 1, 1, 1, 0.5, 0.5), 4]
    ] as const) {
      const result = triangulateParallelReference(coordinates, {validate: true});
      expect(result.requiredCount).toBe(expected);
      expect(result.failed).toBe(false);
    }
  });

  it('a four-site square is two counter-clockwise triangles (SoS picks the diagonal)', () => {
    const result = triangulateParallelReference(Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1), {
      validate: true
    });
    expect(result.requiredCount).toBe(2);
    expect(canonicalSet(result.triangles)).toEqual(
      constructPerturbedDelaunay(Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1))
    );
    // Row 0 has the largest lift perturbation, which pushes it out of every circle through it:
    // the diagonal avoids row 0, and the independent construction below agrees.
    expect(result.triangles).toEqual([
      [0, 1, 3],
      [1, 2, 3]
    ]);
  });

  it('counts duplicates (-0 equals +0, first row owns) and cancels on a non-finite row', () => {
    const duplicates = triangulateParallelReference(
      Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1, 0.5, 0.5, 0, 0, -0, 0),
      {validate: true}
    );
    expect(duplicates).toMatchObject({duplicateCount: 2, requiredCount: 4, invalidCount: 0});
    const invalid = triangulateParallelReference(
      Float32Array.of(0, 0, 1, 0, 1, 1, Number.NaN, 0, 0, 1)
    );
    expect(invalid).toMatchObject({invalidCount: 1, requiredCount: 0, activeSteps: 0});
  });

  const cocircular12 = [
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
  const cases: {name: string; points: number[][]; triangles: number}[] = [
    {name: 'twelve cocircular sites', points: cocircular12, triangles: 10},
    {
      name: 'square with edge midpoints, corners first',
      points: [
        [0, 0],
        [2, 0],
        [2, 2],
        [0, 2],
        [1, 1],
        [1, 0],
        [2, 1],
        [1, 2],
        [0, 1]
      ],
      triangles: 8
    },
    {
      name: 'square with edge midpoints, midpoints first',
      points: [
        [1, 0],
        [2, 1],
        [1, 2],
        [0, 1],
        [0, 0],
        [2, 0],
        [2, 2],
        [0, 2],
        [1, 1]
      ],
      triangles: 8
    },
    {
      name: '5x5 lattice',
      points: Array.from({length: 25}, (_, index) => [index % 5, Math.floor(index / 5)]),
      triangles: 32
    }
  ];
  for (const {name, points, triangles} of cases) {
    it(`${name}: ${triangles} triangles and a clean exact check`, () => {
      const coordinates = Float32Array.from(points.flat());
      const result = triangulateParallelReference(coordinates, {validate: true});
      expect(result.requiredCount).toBe(triangles);
      expect(checkTriangulation(coordinates, result.triangles)).toEqual(CLEAN);
    });
  }

  it('a row permutation of the lattice is still a valid triangulation', () => {
    const lattice = createLatticeFixture(36, 6);
    for (const seed of [1, 2, 3]) {
      const order = Array.from({length: 36}, (_, index) => index);
      // Fisher-Yates with a fixed LCG, so the permutation is reproducible.
      let state = seed;
      for (let index = order.length - 1; index > 0; index--) {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        const other = state % (index + 1);
        [order[index], order[other]] = [order[other], order[index]];
      }
      const permuted = new Float32Array(72);
      order.forEach((row, index) => {
        permuted[2 * index] = lattice[2 * row];
        permuted[2 * index + 1] = lattice[2 * row + 1];
      });
      const result = triangulateParallelReference(permuted, {validate: true});
      expect(result.requiredCount).toBe(2 * 36 - 2 - countHullBoundarySites(permuted));
      expect(checkTriangulation(permuted, result.triangles)).toEqual(CLEAN);
    }
  });

  it('running out of steps is a reported failure with zero triangles, never a partial result', () => {
    const coordinates = createUniformFixture(7, 200, 100, 100);
    const result = triangulateParallelReference(coordinates, {stepCount: 2});
    expect(result).toMatchObject({failed: true, requiredCount: 0, invalidCount: 1});
    expect(result.triangles).toEqual([]);
  });

  it('flip priorities are distinct over the edge ids in use (a bijection)', () => {
    const seen = new Set<number>();
    for (let id = 0; id < 600000; id++) {
      seen.add(getFlipPriority(id));
    }
    expect(seen.size).toBe(600000);
    expect([...seen].every(value => value < 2 ** 31)).toBe(true);
  });

  it('fmix32 priorities are distinct (a bijection)', () => {
    const seen = new Set<number>();
    for (let row = 0; row < 100000; row++) {
      seen.add(getInsertionPriority(row));
    }
    expect(seen.size).toBe(100000);
  });
});

// ---------------------------------------------------------------------------------------------
// Simulation of Simplicity
// ---------------------------------------------------------------------------------------------

type Point = readonly [number, number];

function det3(rows: readonly (readonly bigint[])[]): bigint {
  const [[a, b, c], [d, e, f], [g, h, i]] = rows;
  return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

/**
 * Determinant of the rows (x, y, x^2 + y^2 + eps^(row + 1), 1) as an exact sign, with the
 * perturbation parameter eps = 2^-128 made explicit. The determinant is linear in the lift
 * column, so it equals D + sum_k eps^(row_k + 1) C_k with C_k the cofactor of entry (k, lift)
 * (Laplace expansion along the lift column, not the (-1)^j orient shortcut of the shader).
 * Scaled by (2^128)^(maxRow + 1), every term is an integer.
 */
function getPerturbedDeterminantSign(
  points: readonly [Point, Point, Point, Point],
  rows: readonly [number, number, number, number]
): number {
  const integers = toExactIntegers(points.flat() as number[]);
  const P = [0, 1, 2, 3].map(index => [integers[2 * index], integers[2 * index + 1]]);
  const lift = P.map(([x, y]) => x * x + y * y);
  // The exact integers carry a common scale 2^s, so x^2 + y^2 carries 2^2s; D is homogeneous,
  // and the perturbation size is irrelevant to the sign as long as it is smaller than any
  // nonzero determinant, which 2^-128 per level guarantees for these coordinates.
  const cofactor = (k: number): bigint => {
    const others = [0, 1, 2, 3].filter(index => index !== k).map(index => [...P[index], 1n]);
    return ((k + 2) % 2 === 0 ? 1n : -1n) * det3(others);
  };
  const base = [0, 1, 2, 3].reduce((sum, k) => sum + lift[k] * cofactor(k), 0n);
  const levels = Math.max(...rows) + 1;
  const scale = 1n << 128n;
  let total = base * scale ** BigInt(levels);
  for (let k = 0; k < 4; k++) {
    total += cofactor(k) * scale ** BigInt(levels - (rows[k] + 1));
  }
  return total > 0n ? 1 : total < 0n ? -1 : 0;
}

const CIRCLE_65: Point[] = (() => {
  const points: Point[] = [];
  for (let x = -65; x <= 65; x++) {
    for (let y = -65; y <= 65; y++) {
      if (x * x + y * y === 4225) {
        points.push([x, y]);
      }
    }
  }
  points.sort((a, b) => Math.atan2(a[1], a[0]) - Math.atan2(b[1], b[0]));
  return points;
})();

/** 16 of the 44 integer points of x^2 + y^2 = 65^2, in angular order. */
function getCocircular16(): Float32Array {
  const picked = Array.from(
    {length: 16},
    (_, index) => CIRCLE_65[Math.floor((index * CIRCLE_65.length) / 16)]
  );
  return Float32Array.from(picked.flat());
}

/** The triangles (as sorted row triples) of the lift-perturbed Delaunay triangulation, by brute force. */
function constructPerturbedDelaunay(coordinates: Float32Array): string[] {
  const count = coordinates.length / 2;
  const point = (row: number): Point => [coordinates[2 * row], coordinates[2 * row + 1]];
  const orient = (a: number, b: number, c: number) => {
    const [ax, ay] = point(a);
    const [bx, by] = point(b);
    const [cx, cy] = point(c);
    const value = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    return value > 0 ? 1 : value < 0 ? -1 : 0;
  };
  // D(a, b, c, p) with plain doubles: integer coordinates here, so exact.
  const incircleDeterminant = (a: number, b: number, c: number, p: number) => {
    const [px, py] = point(p);
    const terms = [a, b, c].map(row => {
      const [x, y] = point(row);
      return [x - px, y - py, (x - px) ** 2 + (y - py) ** 2];
    });
    const [[ax, ay, al], [bx, by, bl], [cx, cy, cl]] = terms;
    return al * (bx * cy - cx * by) - bl * (ax * cy - cx * ay) + cl * (ax * by - bx * ay);
  };
  const result: string[] = [];
  for (let i = 0; i < count; i++) {
    for (let j = i + 1; j < count; j++) {
      for (let k = j + 1; k < count; k++) {
        const o = orient(i, j, k);
        if (o === 0) {
          continue;
        }
        const [a, b, c] = o > 0 ? [i, j, k] : [i, k, j];
        let empty = true;
        for (let p = 0; p < count && empty; p++) {
          if (p === i || p === j || p === k) {
            continue;
          }
          const plain = incircleDeterminant(a, b, c, p);
          let sign = plain > 0 ? 1 : plain < 0 ? -1 : 0;
          if (sign === 0) {
            sign = getPerturbedDeterminantSign(
              [point(a), point(b), point(c), point(p)],
              [a, b, c, p]
            );
          }
          if (sign > 0) {
            empty = false;
          }
        }
        if (empty) {
          result.push([i, j, k].join(','));
        }
      }
    }
  }
  return result.sort();
}

describe('parallel Delaunay mirror: Simulation of Simplicity', () => {
  it('the D sign convention: positive iff p3 is strictly inside the ccw circle of p0 p1 p2', () => {
    // 4-site coordinates: (0,0), (4,0), (0,4) is ccw; (1,1) is inside, (5,5) outside.
    const inside = Float32Array.of(0, 0, 4, 0, 0, 4, 1, 1);
    const outside = Float32Array.of(0, 0, 4, 0, 0, 4, 5, 5);
    // Real rows only: rows 0..3 are all < pointCount = 4.
    expect(createFlipPredicates(inside).getInsideSign(0, 1, 2, 3)).toBe(1);
    expect(createFlipPredicates(outside).getInsideSign(0, 1, 2, 3)).toBe(-1);
    // The plain 4x4 determinant of the rows (x, y, x^2 + y^2, 1) of p0..p3 has the same sign.
    const plain = (points: number[][]) => {
      const rows = points.map(([x, y]) => [x, y, x * x + y * y, 1]);
      // Laplace expansion along the last column: cofactor sign (-1)^(k + 3).
      let total = 0;
      for (let k = 0; k < 4; k++) {
        const minor = rows.filter((_, index) => index !== k).map(row => row.slice(0, 3));
        const determinant = det3(minor.map(row => row.map(value => BigInt(value))));
        total += (k % 2 === 0 ? -1 : 1) * Number(determinant);
      }
      return Math.sign(total);
    };
    expect(
      plain([
        [0, 0],
        [4, 0],
        [0, 4],
        [1, 1]
      ])
    ).toBe(1);
    expect(
      plain([
        [0, 0],
        [4, 0],
        [0, 4],
        [5, 5]
      ])
    ).toBe(-1);
  });

  it('getInsideSign equals the explicit-epsilon BigInt sign on every cocircular quadruple', () => {
    const coordinates = getCocircular16();
    const predicates = createFlipPredicates(coordinates);
    const point = (row: number): Point => [coordinates[2 * row], coordinates[2 * row + 1]];
    let checked = 0;
    for (let a = 0; a < 16; a++) {
      for (let b = 0; b < 16; b++) {
        for (let c = 0; c < 16; c++) {
          if (a === b || b === c || a === c || predicates.orientSymbolic(a, b, c) <= 0) {
            continue;
          }
          for (let p = 0; p < 16; p += 3) {
            if (p === a || p === b || p === c) {
              continue;
            }
            const expected = getPerturbedDeterminantSign(
              [point(a), point(b), point(c), point(p)],
              [a, b, c, p]
            );
            expect(expected).not.toBe(0);
            expect(predicates.getInsideSign(a, b, c, p)).toBe(expected);
            checked++;
          }
        }
      }
    }
    expect(checked).toBeGreaterThan(1000);
  });

  const winnerOrders: WinnerOrder[] = [
    'ascending',
    'descending',
    {shuffleSeed: 1},
    {shuffleSeed: 2},
    {shuffleSeed: 3}
  ];
  const sosInputs: [string, Float32Array][] = [
    ['7x7 lattice', createLatticeFixture(49, 7)],
    ['16 cocircular sites', getCocircular16()]
  ];
  for (const [name, coordinates] of sosInputs) {
    it(`${name}: five slot allocation orders give one answer equal to the perturbed construction`, () => {
      const n = coordinates.length / 2;
      const results = winnerOrders.map(winnerOrder =>
        triangulateParallelReference(coordinates, {winnerOrder, validate: true})
      );
      for (const result of results) {
        expect(result.failed).toBe(false);
        expect(result.triangles).toEqual(results[0].triangles);
        expect(result.requiredCount).toBe(
          2 * (n - result.duplicateCount) - 2 - countHullBoundarySites(coordinates)
        );
      }
      // The schedules really differ (different slot ids), yet the output does not.
      expect(checkTriangulation(coordinates, results[0].triangles)).toEqual(CLEAN);
      const constructed = constructPerturbedDelaunay(coordinates);
      expect(canonicalSet(results[0].triangles)).toEqual(constructed);
    });
  }
});

// ---------------------------------------------------------------------------------------------
// Step-count study
// ---------------------------------------------------------------------------------------------

type StudyRow = {
  input: string;
  n: number;
  activeSteps: number;
  insertionSteps: number;
  flips: number;
  budget: number;
};

// Seed-simplex insertion removes the old late-apex chain. The extended indirect-dispatch
// budget remains a bounded safety net for arbitrary flip dependencies.
const getRecommendedStepCount = getDelaunayStepCount;

const STUDY_INPUTS: [string, (n: number) => Float32Array][] = [
  ['uniform', n => createUniformFixture(11, n, 100, 100)],
  ['clustered', n => createClusteredFixture(12, n)],
  ['lattice', n => createLatticeFixture(n, Math.ceil(Math.sqrt(n)))],
  ['circle', n => createCircleFixture(13, n)],
  ['line+1', n => createLineFixture(n)],
  ['x-sorted', n => createSortedFixture(14, n)]
];

describe('parallel Delaunay mirror: step-count study', () => {
  const sizes = process.env['DELAUNAY_STUDY_1E5'] === '1' ? [1e3, 1e4, 1e5] : [1e3, 1e4];
  const rows: StudyRow[] = [];
  for (const n of sizes) {
    for (const [input, create] of STUDY_INPUTS) {
      it(`${input} n=${n}: converges within the recommended budget`, () => {
        const coordinates = create(n);
        const budget = getRecommendedStepCount(n);
        const result = triangulateParallelReference(coordinates, {stepCount: budget});
        rows.push({
          input,
          n,
          activeSteps: result.activeSteps,
          insertionSteps: result.insertionSteps,
          flips: result.flips,
          budget
        });
        expect(result.failed, `${input} ${n} failed`).toBe(false);
        expect(result.invalidCount).toBe(0);
        expect(result.activeSteps, `${input} ${n} steps`).toBeLessThanOrEqual(budget);
        expect(result.requiredCount).toBe(
          2 * (n - result.duplicateCount) - 2 - countHullBoundarySites(coordinates)
        );
        if (n <= 1e4) {
          expect(checkLocallyDelaunay(coordinates, result.triangles)).toEqual(CLEAN);
        }
      }, 900_000);
    }
  }
  it('prints the measured table', () => {
    // eslint-disable-next-line no-console
    console.log(
      ['input n activeSteps insertionSteps flips budget']
        .concat(
          rows.map(
            row =>
              `${row.input} ${row.n} ${row.activeSteps} ${row.insertionSteps} ${row.flips} ${row.budget}`
          )
        )
        .join('\n')
    );
  });
});
