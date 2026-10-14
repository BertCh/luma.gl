// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  checkTriangulation,
  createMonotoneCases,
  createRandomStarPolygon,
  createTwoOptPolygon,
  getPolygonArea,
  trianglesOverlap
} from './polygon-triangulation-monotone-cases';
import {triangulateMonotoneModel} from './polygon-triangulation-monotone-model';

describe('monotone triangulation CPU model', () => {
  for (const entry of createMonotoneCases()) {
    it(`triangulates ${entry.name}`, () => {
      const result = triangulateMonotoneModel(entry.rings);
      expect(result.failure).toBeUndefined();
      expect(result.valid).toBe(true);
      const check = checkTriangulation(entry.rings, result.triangles);
      expect(check.minimumSignedArea).toBeGreaterThanOrEqual(0);
      expect(check.area).toBe(getPolygonArea(entry.rings));
      expect(check.boundaryMatches).toBe(true);
    });
  }
});

describe('monotone triangulation CPU model, randomized', () => {
  it('triangulates 2-opt polygons with sparse and dense coordinates', () => {
    let tested = 0;
    for (let seed = 1; seed <= 120; seed++) {
      const dense = seed % 2 === 0;
      const polygon = createTwoOptPolygon(20 + (seed % 60), seed, dense ? 30 : 5000);
      if (!polygon) continue;
      tested++;
      const result = triangulateMonotoneModel([polygon]);
      expect(result.failure, `seed ${seed}`).toBeUndefined();
      const check = checkTriangulation([polygon], result.triangles);
      expect(check.minimumSignedArea, `seed ${seed}`).toBeGreaterThanOrEqual(0);
      expect(check.area, `seed ${seed}`).toBe(getPolygonArea([polygon]));
      expect(check.boundaryMatches, `seed ${seed}`).toBe(true);
      const brute = triangulateMonotoneModel([polygon], {rayShooting: 'brute'});
      expect(brute.triangles, `seed ${seed} brute`).toEqual(result.triangles);
    }
    expect(tested).toBeGreaterThan(100);
  });

  it('triangulates a 20000-vertex random star polygon', () => {
    const polygon = createRandomStarPolygon(20_000, 20261014);
    const result = triangulateMonotoneModel([polygon]);
    expect(result.failure).toBeUndefined();
    expect(result.triangles.length).toBe(19_998);
    const check = checkTriangulation([polygon], result.triangles);
    expect(check.minimumSignedArea).toBeGreaterThanOrEqual(0);
    const expected = getPolygonArea([polygon]);
    expect(Math.abs(check.area - expected) / expected).toBeLessThan(1e-9);
    expect(check.boundaryMatches).toBe(true);
  }, 120_000);

  it('does not overlap triangles pairwise on the small cases', () => {
    for (const entry of createMonotoneCases().filter(c => c.rings.flat().length < 120)) {
      const result = triangulateMonotoneModel(entry.rings);
      const positions = entry.rings.flat();
      const triangles = result.triangles.map(
        ([a, b, c]) => [positions[a], positions[b], positions[c]] as const
      );
      for (let i = 0; i < triangles.length; i++) {
        for (let j = i + 1; j < triangles.length; j++) {
          expect(trianglesOverlap(triangles[i], triangles[j]), `${entry.name} ${i}/${j}`).toBe(
            false
          );
        }
      }
    }
  });

  it('flags invalid input instead of producing triangles', () => {
    const bowtie = triangulateMonotoneModel([
      [
        [0, 0],
        [10, 10],
        [10, 0],
        [0, 10]
      ]
    ]);
    expect(bowtie.valid).toBe(false);
    const twoVertexShell = triangulateMonotoneModel([
      [
        [0, 0],
        [5, 5]
      ]
    ]);
    expect(twoVertexShell.valid).toBe(false);
  });
});

it('checks the geometric boundary when a hole and shell share distinct rows at one vertex', () => {
  const rings: [number, number][][] = [
    [
      [0, 0],
      [100, 0],
      [100, 100],
      [0, 100]
    ],
    [
      [0, 0],
      [30, 20],
      [20, 50]
    ]
  ];
  // Earcut's triangulation uses both input rows 0 and 4 for the same point (0, 0).
  const triangles: [number, number, number][] = [
    [3, 4, 6],
    [5, 0, 1],
    [1, 2, 3],
    [6, 5, 1],
    [1, 3, 6]
  ];
  const check = checkTriangulation(rings, triangles);
  expect(check.boundaryMatches).toBe(true);
  expect(check.area).toBe(9450);
  expect(check.minimumSignedArea).toBeGreaterThan(0);
  expect(checkTriangulation(rings, triangles.slice(1)).boundaryMatches).toBe(false);
});
