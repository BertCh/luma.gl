// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Adversarial polygons and exact verification helpers for the monotone triangulation path. Shared
 * by the CPU model spec and the GPU specs. All coordinates are integers below 2^24, so every
 * f32 position, cross product and area sum is exact in f64.
 */

import type {ModelPoint} from './polygon-triangulation-monotone-model';

export type MonotoneCase = {
  name: string;
  /** Shell first, then holes. Any winding. */
  rings: ModelPoint[][];
};

/** Deterministic pseudo-random generator (mulberry32). */
export function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

/** Comb with `teeth` teeth pointing up (y up). Many start, merge and regular vertices. */
export function createComb(teeth: number, flip: boolean = false): ModelPoint[] {
  const points: ModelPoint[] = [
    [0, 0],
    [4 * teeth, 0],
    [4 * teeth, 3]
  ];
  for (let k = teeth - 1; k >= 0; k--) {
    points.push([4 * k + 3, 3], [4 * k + 3, 10], [4 * k + 1, 10], [4 * k + 1, 3]);
  }
  points.push([0, 3]);
  return flip ? points.map(([x, y]) => [x, -y] as ModelPoint).reverse() : points;
}

/** Teeth up on top and down on the bottom of a thick bar: splits and merges everywhere. */
export function createDoubleComb(teeth: number): ModelPoint[] {
  const points: ModelPoint[] = [];
  // Bottom edge with teeth pointing down, left to right.
  points.push([0, 7]);
  for (let k = 0; k < teeth; k++) {
    points.push([4 * k + 1, 7], [4 * k + 1, 0], [4 * k + 3, 0], [4 * k + 3, 7]);
  }
  points.push([4 * teeth, 7], [4 * teeth, 13]);
  // Top edge with teeth pointing up, right to left.
  for (let k = teeth - 1; k >= 0; k--) {
    points.push([4 * k + 3, 13], [4 * k + 3, 20], [4 * k + 1, 20], [4 * k + 1, 13]);
  }
  points.push([0, 13]);
  return points;
}

/** Rectilinear spiral corridor of half width 1, with `loops` turns of 4 segments. */
export function createSpiral(loops: number): ModelPoint[] {
  const path: ModelPoint[] = [[0, 0]];
  const directions: ModelPoint[] = [
    [1, 0],
    [0, 1],
    [-1, 0],
    [0, -1]
  ];
  let length = 8 * loops + 8;
  let [x, y] = path[0];
  for (let segment = 0; segment < 4 * loops; segment++) {
    const [dx, dy] = directions[segment % 4];
    // Loops shrink by 4 every two segments after the first.
    if (segment >= 2 && segment % 2 === 0) length -= 4;
    x += dx * length;
    y += dy * length;
    path.push([x, y]);
  }
  // Offset each segment by +-1 and intersect consecutive offset lines.
  const left: ModelPoint[] = [];
  const right: ModelPoint[] = [];
  const segmentCount = path.length - 1;
  const normal = (index: number): ModelPoint => {
    const [dx, dy] = directions[index % 4];
    return [-dy, dx];
  };
  const offsetPoint = (point: ModelPoint, segmentA: number, segmentB: number, sign: number) => {
    // Both segments are axis aligned; an inner corner moves by +-1 along each normal and a
    // path end (segmentB < 0) by +-1 along the one normal.
    const [nax, nay] = normal(segmentA);
    const [nbx, nby] = segmentB < 0 ? [0, 0] : normal(segmentB);
    return [point[0] + sign * (nax + nbx), point[1] + sign * (nay + nby)] as ModelPoint;
  };
  left.push(offsetPoint(path[0], 0, -1, 1));
  right.push(offsetPoint(path[0], 0, -1, -1));
  for (let index = 1; index < segmentCount; index++) {
    left.push(offsetPoint(path[index], index - 1, index, 1));
    right.push(offsetPoint(path[index], index - 1, index, -1));
  }
  left.push(offsetPoint(path[segmentCount], segmentCount - 1, -1, 1));
  right.push(offsetPoint(path[segmentCount], segmentCount - 1, -1, -1));
  return left.concat(right.reverse());
}

/** Regular star with `points` spikes, rounded to integers. */
export function createStar(
  points: number,
  outer: number = 1000,
  inner: number = 400
): ModelPoint[] {
  const ring: ModelPoint[] = [];
  for (let k = 0; k < 2 * points; k++) {
    const angle = (Math.PI * k) / points;
    const radius = k % 2 === 0 ? outer : inner;
    ring.push([Math.round(Math.cos(angle) * radius), Math.round(Math.sin(angle) * radius)]);
  }
  return ring;
}

/** Square shell with a grid of square holes; holes of a row share their y values. */
export function createHoleGrid(
  columns: number,
  rows: number,
  diamond: boolean = false
): ModelPoint[][] {
  const pitch = 100;
  const size = 20;
  const shell: ModelPoint[] = [
    [0, 0],
    [pitch * (columns + 1), 0],
    [pitch * (columns + 1), pitch * (rows + 1)],
    [0, pitch * (rows + 1)]
  ];
  const rings: ModelPoint[][] = [shell];
  for (let row = 0; row < rows; row++) {
    for (let column = 0; column < columns; column++) {
      const cx = pitch * (column + 1);
      const cy = pitch * (row + 1);
      rings.push(
        diamond
          ? [
              [cx - size, cy],
              [cx, cy - size],
              [cx + size, cy],
              [cx, cy + size]
            ]
          : [
              [cx - size, cy - size],
              [cx + size, cy - size],
              [cx + size, cy + size],
              [cx - size, cy + size]
            ]
      );
    }
  }
  return rings;
}

/** Star-shaped polygon with random radii, simple by construction (strictly increasing angle). */
export function createRandomStarPolygon(
  vertexCount: number,
  seed: number,
  radius: number = 500_000
): ModelPoint[] {
  const random = createRandom(seed);
  const ring: ModelPoint[] = [];
  for (let k = 0; k < vertexCount; k++) {
    const angle = (2 * Math.PI * (k + 0.25 + 0.5 * random())) / vertexCount;
    const length = radius * (0.2 + 0.8 * random());
    ring.push([Math.round(Math.cos(angle) * length), Math.round(Math.sin(angle) * length)]);
  }
  return ring;
}

/**
 * Random simple polygon from `count` random integer points: a random tour made simple by 2-opt
 * moves (reverse the stretch between two crossing or touching edges). Small coordinate ranges
 * give many equal coordinates and collinear triples. Returns `undefined` when the tour did not
 * become simple within the move budget.
 */
export function createTwoOptPolygon(
  count: number,
  seed: number,
  range: number
): ModelPoint[] | undefined {
  const random = createRandom(seed);
  const seen = new Set<number>();
  let tour: ModelPoint[] = [];
  let attempts = 0;
  while (tour.length < count && attempts++ < 100 * count) {
    const x = Math.floor(random() * range);
    const y = Math.floor(random() * range);
    if (!seen.has(x * 1048576 + y)) {
      seen.add(x * 1048576 + y);
      tour.push([x, y]);
    }
  }
  if (tour.length < count) return undefined;
  const orient = (a: ModelPoint, b: ModelPoint, c: ModelPoint) =>
    Math.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]));
  const onSegment = (a: ModelPoint, b: ModelPoint, p: ModelPoint) =>
    p[0] >= Math.min(a[0], b[0]) &&
    p[0] <= Math.max(a[0], b[0]) &&
    p[1] >= Math.min(a[1], b[1]) &&
    p[1] <= Math.max(a[1], b[1]);
  const touches = (a: ModelPoint, b: ModelPoint, c: ModelPoint, d: ModelPoint) => {
    const o1 = orient(a, b, c);
    const o2 = orient(a, b, d);
    const o3 = orient(c, d, a);
    const o4 = orient(c, d, b);
    if (o1 * o2 < 0 && o3 * o4 < 0) return true;
    return (
      (o1 === 0 && onSegment(a, b, c)) ||
      (o2 === 0 && onSegment(a, b, d)) ||
      (o3 === 0 && onSegment(c, d, a)) ||
      (o4 === 0 && onSegment(c, d, b))
    );
  };
  for (let move = 0; move < 20 * count * count; move++) {
    let found = false;
    for (let i = 0; i < count && !found; i++) {
      for (let j = i + 2; j < count && !found; j++) {
        if (i === 0 && j === count - 1) continue;
        if (touches(tour[i], tour[i + 1], tour[j], tour[(j + 1) % count])) {
          tour = tour.slice(0, i + 1).concat(tour.slice(i + 1, j + 1).reverse(), tour.slice(j + 1));
          found = true;
        }
      }
    }
    if (!found) return tour;
  }
  return undefined;
}

/** Rectilinear staircase: many horizontal edges and many vertices with equal y. */
export function createStaircase(steps: number): ModelPoint[] {
  const ring: ModelPoint[] = [[0, 0]];
  for (let k = 0; k < steps; k++) {
    ring.push([3 * (k + 1), 2 * k], [3 * (k + 1), 2 * (k + 1)]);
  }
  ring.push([0, 2 * steps]);
  return ring;
}

/** Deterministic 32-bit hash stream (splitmix32), the twin of `splitmix32` in the oracle generator. */
export function createSplitMix32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x9e3779b9) >>> 0;
    let z = state;
    z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0;
    z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0;
    return (z ^ (z >>> 16)) >>> 0;
  };
}

/**
 * Bar-graph outline of `steps` columns with `holes` 2x2 square holes, the twin of `skyline` in
 * `test/oracles/geometry-triangulation-large.py`. Heights repeat often but differ between
 * neighbors, widths are 1 to 3.
 */
export function createSkylineRings(steps: number, seed: number, holes: number): ModelPoint[][] {
  const next = createSplitMix32(seed);
  const heights: number[] = [];
  const columns = [0];
  let previous = 0;
  for (let step = 0; step < steps; step++) {
    let height = 16 + (next() % 40);
    if (height === previous) height = 16 + ((height - 16 + 1) % 40);
    previous = height;
    heights.push(height);
    columns.push(columns[columns.length - 1] + 1 + (next() % 3));
  }
  const total = columns[steps];
  const shell: ModelPoint[] = [
    [0, 0],
    [total, 0]
  ];
  for (let index = steps - 1; index >= 0; index--) {
    shell.push([columns[index + 1], heights[index]], [columns[index], heights[index]]);
  }
  const rings = [shell];
  for (let hole = 0; hole < holes; hole++) {
    const x = 3 * hole + 2;
    if (x + 2 >= total) throw new Error('too many holes for the skyline width');
    rings.push([
      [x, 4],
      [x, 6],
      [x + 2, 6],
      [x + 2, 4]
    ]);
  }
  return rings;
}

/** Sum of `x + 3 y + row` over all rows modulo 2^32, the checksum of the oracle fixture. */
export function getRingsChecksum(rings: readonly (readonly ModelPoint[])[]): number {
  let total = 0;
  let row = 0;
  for (const ring of rings) {
    for (const [x, y] of ring) {
      total += x + 3 * y + row;
      row += 1;
    }
  }
  return total % 4294967296;
}

/** Descriptor of one case of `fixtures/geometry-triangulation-large.json`. */
export type LargePolygonDescriptor = {
  name: string;
  family: 'skyline' | 'spiral' | 'doubleComb';
  params: {steps?: number; seed?: number; holes?: number; loops?: number; teeth?: number};
};

/** Rebuilds the rings of a fixture case from its descriptor. */
export function createLargePolygonRings(descriptor: LargePolygonDescriptor): ModelPoint[][] {
  const {family, params} = descriptor;
  if (family === 'skyline') {
    return createSkylineRings(params.steps!, params.seed!, params.holes!);
  }
  if (family === 'spiral') return [createSpiral(params.loops!)];
  return [createDoubleComb(params.teeth!)];
}

/** The adversarial set used by the model and GPU specs. */
export function createMonotoneCases(): MonotoneCase[] {
  const square: ModelPoint[] = [
    [0, 0],
    [100, 0],
    [100, 100],
    [0, 100]
  ];
  const collinearHole: ModelPoint[] = [
    [20, 20],
    [20, 30],
    [20, 40],
    [20, 50],
    [30, 50],
    [40, 50],
    [50, 50],
    [50, 40],
    [50, 30],
    [50, 20],
    [40, 20],
    [30, 20]
  ];
  const collinearShell: ModelPoint[] = [];
  for (let x = 0; x <= 100; x += 10) collinearShell.push([x, 0]);
  for (let y = 10; y <= 100; y += 10) collinearShell.push([100, y]);
  for (let x = 90; x >= 0; x -= 10) collinearShell.push([x, 100]);
  for (let y = 90; y >= 10; y -= 10) collinearShell.push([0, y]);
  const withDuplicates: ModelPoint[] = [
    [0, 0],
    [0, 0],
    [100, 0],
    [100, 0],
    [100, 0],
    [100, 100],
    [0, 100],
    [0, 100],
    [0, 0]
  ];
  const duplicateHole: ModelPoint[] = [
    [30, 30],
    [30, 30],
    [60, 30],
    [60, 60],
    [60, 60],
    [30, 60],
    [30, 30]
  ];
  const equalHeights: ModelPoint[] = [
    // Two lobes joined by a thin bridge, many vertices on the same y.
    [0, 0],
    [40, 0],
    [40, 10],
    [60, 10],
    [60, 0],
    [100, 0],
    [100, 40],
    [60, 40],
    [60, 30],
    [40, 30],
    [40, 40],
    [0, 40]
  ];
  return [
    {name: 'square', rings: [square]},
    {name: 'square-clockwise', rings: [square.slice().reverse()]},
    {name: 'comb-8', rings: [createComb(8)]},
    {name: 'comb-down-8', rings: [createComb(8, true)]},
    {name: 'double-comb-6', rings: [createDoubleComb(6)]},
    {name: 'double-comb-25', rings: [createDoubleComb(25)]},
    {name: 'spiral-3', rings: [createSpiral(3)]},
    {name: 'spiral-10', rings: [createSpiral(10)]},
    {name: 'star-12', rings: [createStar(12)]},
    {name: 'star-200', rings: [createStar(200)]},
    {name: 'holes-50', rings: createHoleGrid(10, 5)},
    {name: 'holes-50-diamonds', rings: createHoleGrid(10, 5, true)},
    {name: 'collinear-runs', rings: [collinearShell, collinearHole]},
    {name: 'equal-heights', rings: [equalHeights]},
    {name: 'staircase-30', rings: [createStaircase(30)]},
    {name: 'duplicate-vertices', rings: [withDuplicates, duplicateHole]},
    {name: 'random-star-600', rings: [createRandomStarPolygon(600, 7)]},
    {
      name: 'clockwise-shell-with-hole',
      rings: [
        [
          [0, 0],
          [0, 100],
          [100, 100],
          [100, 0]
        ],
        [
          [20, 20],
          [20, 80],
          [80, 80],
          [80, 20]
        ]
      ]
    }
  ];
}

/** Twice the signed area of a ring (positive when counter-clockwise). */
export function getRingSignedArea(ring: readonly ModelPoint[]): number {
  let sum = 0;
  for (let k = 0; k < ring.length; k++) {
    const a = ring[k];
    const b = ring[(k + 1) % ring.length];
    sum += a[0] * b[1] - b[0] * a[1];
  }
  return sum / 2;
}

/** Polygon area: shell minus holes (rings are assumed disjoint, as in valid input). */
export function getPolygonArea(rings: readonly (readonly ModelPoint[])[]): number {
  return rings.reduce(
    (sum, ring, index) => sum + (index === 0 ? 1 : -1) * Math.abs(getRingSignedArea(ring)),
    0
  );
}

export type TriangleCheck = {
  /** Sum of triangle areas. */
  area: number;
  /** Smallest signed triangle area (negative means a clockwise triangle). */
  minimumSignedArea: number;
  /** Whether the directed edges cancel to exactly the oriented polygon boundary. */
  boundaryMatches: boolean;
  /** Number of zero-area triangles. */
  degenerateCount: number;
};

/**
 * Exact checks of a triangulation by row index. With every triangle counter-clockwise, the
 * triangles' directed edges cancel pairwise to exactly the oriented polygon boundary if and only
 * if the 2-chain they form has the polygon boundary as its boundary; its coverage is then the
 * winding number of that boundary, 1 inside the polygon and 0 elsewhere, so the triangles are
 * non-overlapping and cover the polygon exactly.
 */
export function checkTriangulation(
  rings: readonly (readonly ModelPoint[])[],
  triangles: readonly (readonly [number, number, number])[]
): TriangleCheck {
  const positions = rings.flat();
  let area = 0;
  let minimumSignedArea = Infinity;
  let degenerateCount = 0;
  const edgeCounts = new Map<string, number>();
  const canonicalRows = new Map<string, number>();
  const canonical = positions.map(([x, y], row) => {
    const key = `${x},${y}`;
    if (!canonicalRows.has(key)) canonicalRows.set(key, row);
    return canonicalRows.get(key)!;
  });
  const bump = (from: number, to: number, delta: number) => {
    from = canonical[from];
    to = canonical[to];
    if (from === to) return;
    const key = `${from}>${to}`;
    const reverse = `${to}>${from}`;
    const reverseCount = edgeCounts.get(reverse) ?? 0;
    if (delta > 0 && reverseCount > 0) {
      edgeCounts.set(reverse, reverseCount - 1);
      return;
    }
    edgeCounts.set(key, (edgeCounts.get(key) ?? 0) + delta);
  };
  for (const [a, b, c] of triangles) {
    const pa = positions[a];
    const pb = positions[b];
    const pc = positions[c];
    const signed = ((pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])) / 2;
    area += Math.abs(signed);
    minimumSignedArea = Math.min(minimumSignedArea, signed);
    if (signed === 0) degenerateCount++;
    bump(a, b, 1);
    bump(b, c, 1);
    bump(c, a, 1);
  }
  // Oriented boundary on cleaned rows: shell counter-clockwise, holes clockwise.
  const boundary = new Map<string, number>();
  let rowBase = 0;
  rings.forEach((ring, ringIndex) => {
    const rows: number[] = [];
    ring.forEach((point, k) => {
      const previous = ring[(k + ring.length - 1) % ring.length];
      if (point[0] !== previous[0] || point[1] !== previous[1]) rows.push(rowBase + k);
    });
    const counterClockwise = getRingSignedArea(ring) > 0;
    const ordered = counterClockwise === (ringIndex === 0) ? rows : rows.slice().reverse();
    if (rows.length >= 3) {
      ordered.forEach((row, k) => {
        const key = `${canonical[row]}>${canonical[ordered[(k + 1) % ordered.length]]}`;
        boundary.set(key, (boundary.get(key) ?? 0) + 1);
      });
    }
    rowBase += ring.length;
  });
  const remaining = [...edgeCounts.entries()].filter(([, count]) => count !== 0);
  const boundaryMatches =
    remaining.length === boundary.size &&
    remaining.every(([key, count]) => boundary.get(key) === count);
  return {area, minimumSignedArea, boundaryMatches, degenerateCount};
}

type Point = ModelPoint;

/** True when the open interiors of two triangles share area (separating axes, exact). */
export function trianglesOverlap(
  first: readonly [Point, Point, Point],
  second: readonly [Point, Point, Point]
): boolean {
  for (const triangle of [first, second]) {
    for (let edge = 0; edge < 3; edge++) {
      const a = triangle[edge];
      const b = triangle[(edge + 1) % 3];
      const axis: Point = [b[1] - a[1], a[0] - b[0]];
      const project = (point: Point) => axis[0] * point[0] + axis[1] * point[1];
      const firstValues = first.map(project);
      const secondValues = second.map(project);
      if (
        Math.max(...firstValues) <= Math.min(...secondValues) ||
        Math.max(...secondValues) <= Math.min(...firstValues)
      ) {
        return false;
      }
    }
  }
  return true;
}
