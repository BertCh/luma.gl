// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createSeededRandom, type OraclePolygons} from '../spatial-weights/spatial-weights-oracle';

type Point = [number, number];

/**
 * Horizontal strips whose widths vary between 0.03 and 0.13, so the shared boundaries run closer
 * than the simplification tolerances used in the specs. Boundary `i + 1` is boundary `i` plus the
 * strip width (always positive), so the input boundaries never touch, but simplifying each shared
 * boundary on its own can make neighbours cross. Boundaries share their end points, so every strip
 * is a valid polygon and the coverage is gap-free. Coordinates are rounded to f32 so shared
 * vertices are bit-identical.
 */
export function createWavyStrips(stripCount: number, seed: number): OraclePolygons {
  const random = createSeededRandom(seed);
  const width = 10;
  const steps = 160;
  const heights = new Float64Array(steps + 1);
  const boundaries: Point[][] = [];
  for (let boundary = 0; boundary <= stripCount; boundary++) {
    const frequency = 1.5 + random() * 3;
    const phase = random() * 6;
    for (let step = 0; step <= steps; step++) {
      const x = (width * step) / steps;
      const edge = step === 0 || step === steps;
      if (boundary === 0) {
        heights[step] = edge ? 0 : 0.3 * Math.sin(x * 1.1) + (random() - 0.5) * 0.04;
      } else {
        const stripWidth = 0.03 + 0.1 * (0.5 + 0.5 * Math.sin(x * frequency + phase));
        heights[step] += edge ? 0.08 : stripWidth;
      }
    }
    boundaries.push(
      Array.from(
        {length: steps + 1},
        (_, step): Point => [Math.fround((width * step) / steps), Math.fround(heights[step])]
      )
    );
  }
  const polygons: OraclePolygons = [];
  for (let strip = 0; strip < stripCount; strip++) {
    // Bottom boundary left to right, then the top boundary right to left.
    polygons.push([[...boundaries[strip], ...[...boundaries[strip + 1]].reverse()]]);
  }
  return polygons;
}
