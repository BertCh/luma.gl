// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {OracleFeature, OraclePoint} from './spatial-predicate-oracle';

/** Realistic polygon scenes (zip-code-like rings) shared by the relate benchmark and scale spec. */

/** Star-shaped (hence simple) polygon with a wobbly radius, `count` vertices. */
export function createWobblyPolygon(
  random: () => number,
  centerX: number,
  centerY: number,
  radius: number,
  count: number
): OracleFeature {
  const phaseA = random() * 6.28;
  const phaseB = random() * 6.28;
  const ring: OraclePoint[] = [];
  for (let index = 0; index < count; index++) {
    const angle = (index / count) * Math.PI * 2;
    const scale =
      1 +
      0.25 * Math.sin(3 * angle + phaseA) +
      0.1 * Math.sin(7 * angle + phaseB) +
      0.04 * (random() - 0.5);
    ring.push([
      Math.fround(centerX + Math.cos(angle) * radius * scale),
      Math.fround(centerY + Math.sin(angle) * radius * scale)
    ]);
  }
  return {kind: 'polygons', polygons: [[ring]]};
}

export function createRandomWalkLine(
  random: () => number,
  startX: number,
  startY: number,
  count: number,
  step: number
): OracleFeature {
  const vertices: OraclePoint[] = [];
  let angle = random() * 6.28;
  let x = startX;
  let y = startY;
  for (let index = 0; index < count; index++) {
    vertices.push([Math.fround(x), Math.fround(y)]);
    angle += (random() - 0.5) * 0.8;
    x += Math.cos(angle) * step;
    y += Math.sin(angle) * step;
  }
  return {kind: 'lines', vertices};
}

/**
 * Lattice of `grid x grid` cells whose shared edges are wobbly polylines of `segments` segments,
 * stored once so neighbours share vertices and long runs of boundary exactly (as zip codes do).
 * Returns the cells and `blockSize x blockSize` blocks of cells (their rings follow the same
 * polylines, so every block edge coincides with cell edges).
 */
export function createSharedBoundaryLattice(
  random: () => number,
  grid: number,
  segments: number,
  blockSize: number
): {cells: OracleFeature[]; blocks: OracleFeature[]} {
  const corners: OraclePoint[][] = [];
  for (let row = 0; row <= grid; row++) {
    corners.push([]);
    for (let column = 0; column <= grid; column++) {
      corners[row].push([
        Math.fround(column + (random() - 0.5) * 0.3),
        Math.fround(row + (random() - 0.5) * 0.3)
      ]);
    }
  }
  const wobble = (from: OraclePoint, to: OraclePoint): OraclePoint[] => {
    const phase = random() * 6.28;
    const amplitude = 0.1 + random() * 0.05;
    const points: OraclePoint[] = [];
    for (let index = 0; index <= segments; index++) {
      const t = index / segments;
      const offset = Math.sin(t * Math.PI) * amplitude * Math.sin(5 * t * Math.PI + phase);
      points.push([
        Math.fround(from[0] + (to[0] - from[0]) * t - (to[1] - from[1]) * offset),
        Math.fround(from[1] + (to[1] - from[1]) * t + (to[0] - from[0]) * offset)
      ]);
    }
    points[0] = from;
    points[segments] = to;
    return points;
  };
  const horizontal: OraclePoint[][][] = []; // [row][column]: corner (row, column) to (row, column + 1)
  const vertical: OraclePoint[][][] = []; // [row][column]: corner (row, column) to (row + 1, column)
  for (let row = 0; row <= grid; row++) {
    horizontal.push([]);
    vertical.push([]);
    for (let column = 0; column <= grid; column++) {
      if (column < grid) {
        horizontal[row].push(wobble(corners[row][column], corners[row][column + 1]));
      }
      if (row < grid) {
        vertical[row].push(wobble(corners[row][column], corners[row + 1][column]));
      }
    }
  }
  const reversed = (points: OraclePoint[]) => points.slice().reverse();
  /** Counter-clockwise ring of the cells [row, row + height) x [column, column + width). */
  const createRing = (row: number, column: number, height: number, width: number) => {
    const ring: OraclePoint[] = [];
    const append = (points: OraclePoint[]) => ring.push(...points.slice(0, -1));
    for (let c = column; c < column + width; c++) append(horizontal[row][c]);
    for (let r = row; r < row + height; r++) append(vertical[r][column + width]);
    for (let c = column + width - 1; c >= column; c--)
      append(reversed(horizontal[row + height][c]));
    for (let r = row + height - 1; r >= row; r--) append(reversed(vertical[r][column]));
    return ring;
  };
  const cells: OracleFeature[] = [];
  for (let row = 0; row < grid; row++) {
    for (let column = 0; column < grid; column++) {
      cells.push({kind: 'polygons', polygons: [[createRing(row, column, 1, 1)]]});
    }
  }
  const blocks: OracleFeature[] = [];
  for (const [row, column] of [
    [0, 0],
    [1, 1],
    [grid - blockSize, grid - blockSize],
    [0, grid - blockSize]
  ]) {
    blocks.push({kind: 'polygons', polygons: [[createRing(row, column, blockSize, blockSize)]]});
  }
  return {cells, blocks};
}

/** Hashes the f32 coordinate bits of polygon and line features (FNV-1a), to pin generated scenes. */
export function hashFeatureCoordinates(features: OracleFeature[]): string {
  let hash = 2166136261;
  for (const feature of features) {
    const coordinates =
      feature.kind === 'lines'
        ? feature.vertices.flat()
        : feature.kind === 'polygons'
          ? feature.polygons.flat(2).flat()
          : feature.vertex;
    for (const bits of new Uint32Array(Float32Array.from(coordinates).buffer)) {
      hash = Math.imul(hash ^ bits, 16777619) >>> 0;
    }
  }
  return hash.toString(16);
}

/**
 * The scene pinned by `shapely-scale-fixtures.ts`: 9 zip-like polygons of 120 to 239 vertices, 3
 * query polygons of 60 to 139 vertices, 3 random-walk lines of 60 vertices, and a 3 x 3 lattice of
 * cells with shared boundaries against four 2 x 2 blocks.
 */
export function createRealisticRelateScene(
  createRandom: (seed: number) => () => number
): Record<'zips' | 'queries' | 'lines' | 'cells' | 'blocks', OracleFeature[]> {
  const random = createRandom(515);
  const zips: OracleFeature[] = [];
  for (let row = 0; row < 3; row++) {
    for (let column = 0; column < 3; column++) {
      zips.push(
        createWobblyPolygon(
          random,
          column * 2 + random(),
          row * 2 + random(),
          1.1,
          120 + Math.floor(random() * 120)
        )
      );
    }
  }
  const queries = [0, 1, 2].map(() =>
    createWobblyPolygon(random, random() * 6, random() * 6, 2.2, 60 + Math.floor(random() * 80))
  );
  const lines = [0, 1, 2].map(() =>
    createRandomWalkLine(random, random() * 6, random() * 6, 60, 0.3)
  );
  const lattice = createSharedBoundaryLattice(random, 3, 14, 2);
  return {zips, queries, lines, cells: lattice.cells, blocks: lattice.blocks};
}

/**
 * Small features that share boundaries: a 3 x 3 lattice of 12-vertex cells (and 2 x 2 blocks of 24
 * vertices) whose shared edges are 3-segment polylines. Pinned by `shapely-scale-fixtures.ts`.
 */
export function createSmallSharedBoundaryScene(createRandom: (seed: number) => () => number): {
  cells: OracleFeature[];
  blocks: OracleFeature[];
} {
  return createSharedBoundaryLattice(createRandom(616), 3, 3, 2);
}
