// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  checkTriangulation,
  createMonotoneCases,
  createTwoOptPolygon,
  getPolygonArea,
  type MonotoneCase
} from './polygon-triangulation-monotone-cases';
import {runMonotoneKernels, type HarnessResult} from './polygon-triangulation-monotone-harness';
import {triangulateMonotoneModel, type ModelPoint} from './polygon-triangulation-monotone-model';

// The real WGSL kernels of the monotone path run in the wgsl_reflect interpreter (scans and sorts
// replaced by CPU equivalents, see the harness) and are compared with the CPU model, which the
// model spec validates on exact area and boundary checks. This is the closest check of the
// kernels that does not need a GPU.

const SMALL_VERTEX_LIMIT = 250;

function getSlice(result: HarnessResult, polygon: number): [number, number, number][] {
  const base = result.polygonRows[polygon];
  const slots = (result.sliceStarts[polygon + 1] - result.sliceStarts[polygon]) / 3;
  const slice: [number, number, number][] = [];
  for (let k = 0; k < slots; k++) {
    const at = result.sliceStarts[polygon] + 3 * k;
    slice.push([
      result.indices[at] - base,
      result.indices[at + 1] - base,
      result.indices[at + 2] - base
    ]);
  }
  return slice;
}

function canonical(triangle: readonly number[]): string {
  const smallest = triangle.indexOf(Math.min(...triangle));
  return [0, 1, 2].map(k => triangle[(smallest + k) % 3]).join(',');
}

/** Checks the slice of `polygon` against the CPU model and the exact area and boundary checks. */
function expectMatchesModel(result: HarnessResult, polygon: number, entry: MonotoneCase) {
  const model = triangulateMonotoneModel(entry.rings);
  expect(result.failures[polygon], `${entry.name} failure bits`).toBe(0);
  expect(result.valid[polygon], `${entry.name} valid`).toBe(1);
  expect(result.triangleCount[polygon], `${entry.name} count`).toBe(model.triangles.length);
  const slice = getSlice(result, polygon);
  const real = slice.slice(0, model.triangles.length);
  expect(real.map(canonical).sort(), entry.name).toEqual(model.triangles.map(canonical).sort());
  for (const padding of slice.slice(model.triangles.length)) {
    expect(padding, `${entry.name} padding`).toEqual([0, 0, 0]);
  }
  const check = checkTriangulation(entry.rings, real);
  expect(check.minimumSignedArea, `${entry.name} winding`).toBeGreaterThanOrEqual(0);
  expect(check.area, `${entry.name} area`).toBe(getPolygonArea(entry.rings));
  expect(check.boundaryMatches, `${entry.name} boundary`).toBe(true);
}

const cases = createMonotoneCases().filter(
  entry => entry.rings.flat().length <= SMALL_VERTEX_LIMIT
);

describe('monotone kernels in the WGSL interpreter', () => {
  it('matches the CPU model on every small adversarial polygon, one run each, with edge slabs', () => {
    let slabbed = 0;
    for (const entry of cases) {
      const result = runMonotoneKernels([entry.rings]);
      slabbed += result.slabbedPolygonCount;
      expectMatchesModel(result, 0, entry);
    }
    // The ray shooting ran through slab queries, not only through full walks.
    expect(slabbed).toBeGreaterThan(8);
  }, 600_000);

  it('matches the CPU model when all polygons share one run (walked edge slabs)', () => {
    const result = runMonotoneKernels(
      cases.map(entry => entry.rings),
      {slabs: 'walk'}
    );
    cases.forEach((entry, polygon) => expectMatchesModel(result, polygon, entry));
    expect(result.routing.every(route => route === 1)).toBe(true);
    // The slices of neighbors are disjoint. The capacity helper also reserves spare tail slots.
    expect(result.indices.slice(0, result.sliceStarts.at(-1)).includes(0xdeadbeef)).toBe(false);
  }, 600_000);

  it('matches the CPU model on random polygons with many equal coordinates, shared run', () => {
    const polygons: MonotoneCase[] = [];
    for (let seed = 1; polygons.length < 16; seed++) {
      const ring = createTwoOptPolygon(18 + (seed % 20), 100 + seed, seed % 2 === 0 ? 24 : 4000);
      if (ring) polygons.push({name: `two-opt-${seed}`, rings: [ring]});
    }
    const result = runMonotoneKernels(polygons.map(entry => entry.rings));
    polygons.forEach((entry, polygon) => expectMatchesModel(result, polygon, entry));
  }, 600_000);

  it('only touches routed polygons: small ones keep their slices, valid and counts', () => {
    const small = cases.filter(entry => entry.rings.flat().length < 20);
    const large = cases.filter(entry => entry.rings.flat().length >= 20).slice(0, 4);
    const ordered = [small[0], large[0], small[1], large[1], large[2], small[2]];
    const result = runMonotoneKernels(
      ordered.map(entry => entry.rings),
      {minimumVertices: 20}
    );
    ordered.forEach((entry, polygon) => {
      const routed = entry.rings.flat().length >= 20;
      expect(result.routing[polygon], entry.name).toBe(routed ? 1 : 0);
      if (routed) {
        expectMatchesModel(result, polygon, entry);
      } else {
        // Earcut owns these polygons: the monotone path leaves their outputs alone.
        expect(result.valid[polygon], entry.name).toBe(0);
        expect(
          getSlice(result, polygon)
            .flat()
            .every(value => value + result.polygonRows[polygon] === 0xdeadbeef)
        ).toBe(true);
      }
    });
  }, 600_000);

  it('retries polygons that Earcut failed (route 2) with the same result as size routing', () => {
    const picked = cases.slice(0, 6);
    const retried = runMonotoneKernels(
      picked.map(entry => entry.rings),
      {minimumVertices: 1_000_000, retryPolygons: [0, 2, 3, 5]}
    );
    picked.forEach((entry, polygon) => {
      if ([0, 2, 3, 5].includes(polygon)) {
        expect(retried.routing[polygon]).toBe(2);
        expectMatchesModel(retried, polygon, entry);
      } else {
        expect(retried.routing[polygon], entry.name).toBe(0);
        expect(retried.valid[polygon], entry.name).toBe(0);
      }
    });
  }, 600_000);

  it('flags polygons that exceed the capacity and keeps their neighbors intact', () => {
    const picked = [cases[0], cases[2], cases[4], cases[0]];
    const vertexCounts = picked.map(entry => entry.rings.flat().length);
    // Room for the first two polygons only: the third overflows, and so does everything after it.
    const capacity = vertexCounts[0] + vertexCounts[1];
    const result = runMonotoneKernels(
      picked.map(entry => entry.rings),
      {capacity}
    );
    expectMatchesModel(result, 0, picked[0]);
    expectMatchesModel(result, 1, picked[1]);
    for (const polygon of [2, 3]) {
      // Overflowed polygons are handed to the Earcut fallback pass (route 4).
      expect(result.routing[polygon]).toBe(4);
      expect(result.valid[polygon]).toBe(0);
      expect(result.triangleCount[polygon]).toBe(0);
      const shellFirstRow = 0;
      void shellFirstRow;
      const slice = getSlice(result, polygon);
      expect(slice.length).toBeGreaterThan(0);
      for (const triangle of slice) expect(triangle).toEqual([0, 0, 0]);
    }
  }, 600_000);

  it('flags invalid polygons, pads their slices and keeps their neighbors intact', () => {
    const square = cases[0].rings;
    const bowtie: ModelPoint[][] = [
      [
        [0, 0],
        [10, 10],
        [10, 0],
        [0, 10]
      ]
    ];
    const twoVertexShell: ModelPoint[][] = [
      [
        [0, 0],
        [5, 5],
        [5, 5]
      ]
    ];
    const withNonFinite: ModelPoint[][] = [
      [
        [0, 0],
        [10, 0],
        [Infinity, 10],
        [0, 10]
      ]
    ];
    const overlappingHole: ModelPoint[][] = [
      [
        [0, 0],
        [40, 0],
        [40, 40],
        [0, 40]
      ],
      [
        [10, 10],
        [10, 30],
        [30, 30],
        [30, 10]
      ],
      [
        [20, 5],
        [20, 20],
        [35, 20],
        [35, 5]
      ]
    ];
    const polygons = [square, bowtie, twoVertexShell, withNonFinite, overlappingHole, square];
    const result = runMonotoneKernels(polygons);
    expectMatchesModel(result, 0, {name: 'square before', rings: square});
    expectMatchesModel(result, 5, {name: 'square after', rings: square});
    for (const polygon of [1, 2, 3]) {
      expect(result.valid[polygon], `polygon ${polygon}`).toBe(0);
      expect(result.triangleCount[polygon]).toBe(0);
      for (const triangle of getSlice(result, polygon)) {
        // Degenerate (first, first, first) of the polygon's shell.
        expect(triangle[0]).toBe(triangle[1]);
        expect(triangle[1]).toBe(triangle[2]);
      }
    }
    // Overlapping holes are invalid input; whatever the path reports, the slice must stay inside
    // the polygon's own slots (the neighbors above are bit-checked) and the counts must agree.
    const slots = getSlice(result, 4).length;
    expect(result.triangleCount[4]).toBeLessThanOrEqual(slots);
    if (result.valid[4] === 0) expect(result.triangleCount[4]).toBe(0);
  }, 600_000);

  it('ignores holes with fewer than three distinct vertices, like Earcut', () => {
    const rings: ModelPoint[][] = [
      [
        [0, 0],
        [100, 0],
        [100, 100],
        [0, 100]
      ],
      [
        [10, 10],
        [10, 10],
        [20, 20]
      ],
      [
        [50, 50],
        [50, 70],
        [70, 70],
        [70, 50]
      ]
    ];
    const result = runMonotoneKernels([rings]);
    // Two live rings: 8 vertices and one hole give 8 + 2 - 2 = 8 triangles; the slice has room for
    // 11 vertices and two holes, 13 slots in all.
    expect(result.valid).toEqual([1]);
    expect(result.triangleCount).toEqual([8]);
    expect(getSlice(result, 0).length).toBe(11 + 4 - 2);
    const model = triangulateMonotoneModel(rings);
    expect(model.triangles.length).toBe(8);
    const real = getSlice(result, 0).slice(0, 8);
    expect(real.map(canonical).sort()).toEqual(model.triangles.map(canonical).sort());
    for (const padding of getSlice(result, 0).slice(8)) expect(padding).toEqual([0, 0, 0]);
  }, 600_000);

  it('flags a hole that touches the shell at a vertex and hands the polygon to Earcut (route 4)', () => {
    const touching: ModelPoint[][] = [
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
    const separate: ModelPoint[][] = [
      touching[0],
      [
        [10, 10],
        [30, 20],
        [20, 50]
      ]
    ];
    const result = runMonotoneKernels([touching, separate]);
    // Coincident vertices need a zero-length diagonal, which the decomposition does not support.
    expect(result.valid[0]).toBe(0);
    expect(result.routing[0]).toBe(4);
    expect(result.failures[0]).not.toBe(0);
    for (const triangle of getSlice(result, 0)) expect(triangle).toEqual([0, 0, 0]);
    expectMatchesModel(result, 1, {name: 'separate hole', rings: separate});
    // Holes that touch each other at a vertex need no such diagonal.
    const holesTouch: ModelPoint[][] = [
      touching[0],
      [
        [10, 10],
        [30, 10],
        [30, 30],
        [10, 30]
      ],
      [
        [30, 30],
        [50, 30],
        [50, 50],
        [30, 50]
      ]
    ];
    const touchingHoles = runMonotoneKernels([holesTouch]);
    expectMatchesModel(touchingHoles, 0, {name: 'holes touch', rings: holesTouch});
  }, 600_000);
});
