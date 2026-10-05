// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createPolygons, type OraclePolygons} from './polygon-rasterization-oracle';

/** Deterministic pseudo-random generator so failures reproduce. */
export function createRandom(seed: number): () => number {
  let state = seed;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function rectangle(x0: number, y0: number, x1: number, y1: number): number[][] {
  return [
    [x0, y0],
    [x1, y0],
    [x1, y1],
    [x0, y1]
  ];
}

/**
 * Hand-built scene on a 16 x 12 raster with unit cells at the origin, in exactly representable
 * coordinates:
 * - 0: square with a hole (hole wound like the shell), explicitly closed shell ring
 * - 1: rectangle overlapping feature 0 (feature 0 wins the overlap)
 * - 2: multipolygon of a sub-cell triangle covering no center and a sub-cell square covering one
 * - 3: rectangle whose edges lie exactly on cell centers
 * - 4: rectangle sharing feature 3's right edge, also on centers
 * - 5: triangle with a diagonal through centers, partly above the raster
 * - 6: duplicate of feature 1 (always loses to feature 1)
 * - 7: rectangle extending past the left edge of the raster
 */
export function createExactScene(): OraclePolygons {
  const shell = rectangle(1, 1, 7, 7);
  return createPolygons([
    [[[...shell, shell[0]], rectangle(3, 3, 5, 5)]],
    [[rectangle(5, 2, 10, 6)]],
    [
      [
        [
          [12.1, 1.1],
          [12.4, 1.1],
          [12.4, 1.4]
        ]
      ],
      [rectangle(12.375, 3.375, 12.625, 3.625)]
    ],
    [[rectangle(0.5, 8.5, 3.5, 10.5)]],
    [[rectangle(3.5, 8.5, 5.5, 10.5)]],
    [
      [
        [
          [6.5, 8.5],
          [11.5, 8.5],
          [6.5, 13.5]
        ]
      ]
    ],
    [[rectangle(5, 2, 10, 6)]],
    [[rectangle(-3, 7.25, 0.75, 7.75)]]
  ]);
}

/**
 * Random star-shaped polygons, some with a scaled inner hole and some as two-polygon features, in
 * `[0, extentWidth] x [0, extentHeight]`.
 */
export function createRandomScene(
  seed: number,
  featureCount: number,
  extentWidth: number,
  extentHeight: number,
  maximumRadius: number
): OraclePolygons {
  const random = createRandom(seed);
  const createStar = (centerX: number, centerY: number, radius: number, vertexCount: number) => {
    const ring: number[][] = [];
    for (let vertex = 0; vertex < vertexCount; vertex++) {
      const angle = (vertex / vertexCount) * Math.PI * 2;
      const distance = radius * (0.45 + 0.55 * random());
      ring.push([centerX + Math.cos(angle) * distance, centerY + Math.sin(angle) * distance]);
    }
    return ring;
  };
  const features: number[][][][][] = [];
  for (let feature = 0; feature < featureCount; feature++) {
    const polygons: number[][][][] = [];
    const polygonCount = random() < 0.2 ? 2 : 1;
    for (let polygon = 0; polygon < polygonCount; polygon++) {
      const centerX = random() * extentWidth;
      const centerY = random() * extentHeight;
      const radius = maximumRadius * (0.05 + 0.95 * random());
      const vertexCount = 3 + Math.floor(random() * 14);
      const rings = [createStar(centerX, centerY, radius, vertexCount)];
      if (random() < 0.35) {
        rings.push(createStar(centerX, centerY, radius * 0.3, 3 + Math.floor(random() * 6)));
      }
      polygons.push(rings);
    }
    features.push(polygons);
  }
  return createPolygons(features);
}
