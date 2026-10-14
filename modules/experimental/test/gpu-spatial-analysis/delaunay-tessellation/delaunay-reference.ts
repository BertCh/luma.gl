// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * CPU references for the Delaunay specs: seeded fixtures, exact rational predicates on f32 inputs,
 * a triangulation validity check, and a replay of the shader's symbolic super-vertex rules.
 * The fixtures and the digest are mirrored by the oracle script
 * `scratchpad/fix/work/W2A/delaunay/oracle.py` (scipy.spatial 1.18.1), which produced the frozen
 * constants in the specs.
 */

// Constants frozen from scipy.spatial 1.18.1 (Delaunay, ConvexHull) by
// scratchpad/fix/work/W2A/delaunay/oracle.py. `triangles` = len(Delaunay.simplices),
// `hull` = len(ConvexHull.vertices), and 2n - 2 - hull = triangles for these generic inputs.
// `digest` is FNV-1a over the sorted canonical triangle set.
export const SCIPY_FIXTURES = {
  uniform600: {
    seed: 20260613,
    count: 600,
    width: 100,
    height: 100,
    triangles: 1183,
    hull: 15,
    digest: 3455654972
  },
  band100to1: {
    seed: 20260614,
    count: 600,
    width: 100,
    height: 1,
    triangles: 1175,
    hull: 23,
    digest: 706783573
  },
  band1em2: {
    seed: 20260615,
    count: 200,
    width: 1,
    height: 0.01,
    triangles: 388,
    hull: 10,
    digest: 2999109385
  }
} as const;

/** Seeded generator, bit-identical to the oracle's `mulberry32`. Returns values in [0, 1). */
export function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Interleaved f32 coordinates uniformly filling [0, width] x [0, height]. */
export function createUniformFixture(
  seed: number,
  count: number,
  width: number,
  height: number
): Float32Array {
  const random = mulberry32(seed);
  const coordinates = new Float32Array(2 * count);
  for (let row = 0; row < count; row++) {
    coordinates[2 * row] = random() * width;
    coordinates[2 * row + 1] = random() * height;
  }
  return coordinates;
}

/** FNV-1a over the sorted canonical triangle set; identical to the oracle's `digest`. */
export function digestTriangleSet(triangles: ArrayLike<ArrayLike<number>>): number {
  const keys: string[] = [];
  for (let index = 0; index < triangles.length; index++) {
    keys.push(
      Array.from(triangles[index])
        .sort((a, b) => a - b)
        .join(',')
    );
  }
  keys.sort();
  let hash = 0x811c9dc5;
  const text = keys.join(';');
  for (let index = 0; index < text.length; index++) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

/** Splits flat uint32 triples into arrays of three. */
export function toTriangleRows(indices: ArrayLike<number>): number[][] {
  const rows: number[][] = [];
  for (let base = 0; base + 2 < indices.length; base += 3) {
    rows.push([indices[base], indices[base + 1], indices[base + 2]]);
  }
  return rows;
}

/** Exact integer image of f32 values, all scaled by the same power of two. */
export function toExactIntegers(values: ArrayLike<number>): bigint[] {
  const view = new DataView(new ArrayBuffer(4));
  const parts: {negative: boolean; mantissa: number; exponent: number}[] = [];
  let minimumExponent = Number.POSITIVE_INFINITY;
  for (let index = 0; index < values.length; index++) {
    view.setFloat32(0, values[index]);
    const bits = view.getUint32(0);
    const biased = (bits >>> 23) & 0xff;
    const fraction = bits & 0x7fffff;
    const mantissa = biased === 0 ? fraction : fraction | 0x800000;
    const exponent = biased === 0 ? -149 : biased - 150;
    parts.push({negative: bits >>> 31 === 1, mantissa, exponent});
    if (mantissa !== 0) {
      minimumExponent = Math.min(minimumExponent, exponent);
    }
  }
  return parts.map(({negative, mantissa, exponent}) => {
    if (mantissa === 0) {
      return 0n;
    }
    const magnitude = BigInt(mantissa) << BigInt(exponent - minimumExponent);
    return negative ? -magnitude : magnitude;
  });
}

/** Exact sign of cross(b - a, c - a). */
export function orientExact(
  [ax, ay]: readonly bigint[],
  [bx, by]: readonly bigint[],
  [cx, cy]: readonly bigint[]
): number {
  const determinant = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
  return determinant > 0n ? 1 : determinant < 0n ? -1 : 0;
}

/** Exact sign of the in-circle determinant: positive when p is inside the CCW circle abc. */
export function incircleExact(
  a: readonly bigint[],
  b: readonly bigint[],
  c: readonly bigint[],
  p: readonly bigint[]
): number {
  const ax = a[0] - p[0];
  const ay = a[1] - p[1];
  const bx = b[0] - p[0];
  const by = b[1] - p[1];
  const cx = c[0] - p[0];
  const cy = c[1] - p[1];
  const determinant =
    (ax * ax + ay * ay) * (bx * cy - cx * by) -
    (bx * bx + by * by) * (ax * cy - cx * ay) +
    (cx * cx + cy * cy) * (ax * by - bx * ay);
  return determinant > 0n ? 1 : determinant < 0n ? -1 : 0;
}

/**
 * Exact in-circle for f32 rows with a double-precision filter. Row coordinates are exact in
 * double, so the filter bound 1e-9 of the permanent is far above its rounding error; ties and
 * near ties go to the exact BigInt determinant.
 */
function createInCircleTester(coordinates: ArrayLike<number>) {
  let exact: bigint[] | undefined;
  return (a: number, b: number, c: number, p: number): number => {
    const ax = coordinates[2 * a] - coordinates[2 * p];
    const ay = coordinates[2 * a + 1] - coordinates[2 * p + 1];
    const bx = coordinates[2 * b] - coordinates[2 * p];
    const by = coordinates[2 * b + 1] - coordinates[2 * p + 1];
    const cx = coordinates[2 * c] - coordinates[2 * p];
    const cy = coordinates[2 * c + 1] - coordinates[2 * p + 1];
    const aLift = ax * ax + ay * ay;
    const bLift = bx * bx + by * by;
    const cLift = cx * cx + cy * cy;
    const determinant =
      aLift * (bx * cy - cx * by) - bLift * (ax * cy - cx * ay) + cLift * (ax * by - bx * ay);
    const permanent =
      aLift * (Math.abs(bx * cy) + Math.abs(cx * by)) +
      bLift * (Math.abs(ax * cy) + Math.abs(cx * ay)) +
      cLift * (Math.abs(ax * by) + Math.abs(bx * ay));
    if (Math.abs(determinant) > 1e-9 * permanent) {
      return determinant > 0 ? 1 : -1;
    }
    exact ??= toExactIntegers(coordinates);
    const row = (index: number) => [exact![2 * index], exact![2 * index + 1]];
    return incircleExact(row(a), row(b), row(c), row(p));
  };
}

export type TriangulationCheck = {
  /** Triangles that are not strictly counter-clockwise. */
  degenerate: number;
  /** (triangle, site) pairs with the site strictly inside the circumcircle. */
  circleViolations: number;
  /** Directed edges used by more than one triangle. */
  duplicateEdges: number;
  /** Doubled area of the triangles minus doubled area of the convex hull; zero when they tile it. */
  areaDifference: bigint;
};

/** Exact validity report for a triangulation of f32 sites (exact rational arithmetic). */
export function checkTriangulation(
  coordinates: Float32Array,
  triangles: readonly (readonly number[])[]
): TriangulationCheck {
  const exact = toExactIntegers(coordinates);
  const row = (index: number) => [exact[2 * index], exact[2 * index + 1]];
  const inCircle = createInCircleTester(coordinates);
  const count = coordinates.length / 2;
  const directedEdges = new Set<string>();
  let degenerate = 0;
  let circleViolations = 0;
  let duplicateEdges = 0;
  let doubledArea = 0n;
  for (const [a, b, c] of triangles) {
    if (orientExact(row(a), row(b), row(c)) <= 0) {
      degenerate++;
      continue;
    }
    const [ax, ay] = row(a);
    const [bx, by] = row(b);
    const [cx, cy] = row(c);
    doubledArea += (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    for (const key of [`${a}>${b}`, `${b}>${c}`, `${c}>${a}`]) {
      if (directedEdges.has(key)) {
        duplicateEdges++;
      }
      directedEdges.add(key);
    }
    for (let site = 0; site < count; site++) {
      if (site !== a && site !== b && site !== c && inCircle(a, b, c, site) > 0) {
        circleViolations++;
      }
    }
  }
  // Doubled area of the convex hull by exact monotone chain (collinear points dropped).
  const order = Array.from({length: count}, (_, index) => index).sort((first, second) => {
    const [fx, fy] = row(first);
    const [sx, sy] = row(second);
    return fx === sx ? (fy < sy ? -1 : fy > sy ? 1 : 0) : fx < sx ? -1 : 1;
  });
  const chain = (rows: number[]) => {
    const stack: number[] = [];
    for (const index of rows) {
      while (
        stack.length >= 2 &&
        orientExact(row(stack[stack.length - 2]), row(stack[stack.length - 1]), row(index)) <= 0
      ) {
        stack.pop();
      }
      stack.push(index);
    }
    return stack;
  };
  const lower = chain(order);
  const upper = chain([...order].reverse());
  const hull = [...lower.slice(0, -1), ...upper.slice(0, -1)];
  let hullArea = 0n;
  for (let index = 0; index < hull.length; index++) {
    const [x0, y0] = row(hull[index]);
    const [x1, y1] = row(hull[(index + 1) % hull.length]);
    hullArea += x0 * y1 - x1 * y0;
  }
  return {degenerate, circleViolations, duplicateEdges, areaDifference: doubledArea - hullArea};
}

// ---------------------------------------------------------------------------------------------
// Replay of the shader's symbolic super-vertex predicates (see SYMBOLIC_SUPER_WGSL).
// Rows >= pointCount are super vertices S0 (+x), S1 (+y), S2 (-x).
// ---------------------------------------------------------------------------------------------

function compare(a: number, b: number): number {
  return a < b ? -1 : a > b ? 1 : 0;
}

/** Sign of orient(r, S_k, p) with the tilted direction d_k: axis comparison, then the tilt. */
export function orientRealSuper(
  r: readonly [number, number],
  superIndex: number,
  p: readonly [number, number]
): number {
  const dx = compare(p[0], r[0]);
  const dy = compare(p[1], r[1]);
  if (superIndex === 0) {
    return dy !== 0 ? dy : dx;
  }
  if (superIndex === 1) {
    return dx !== 0 ? -dx : dy;
  }
  return dy !== 0 ? -dy : dx;
}

/** Predicates over real rows plus the three symbolic super rows. */
export function createSymbolicPredicates(coordinates: Float32Array) {
  const pointCount = coordinates.length / 2;
  const exact = toExactIntegers(coordinates);
  const inCircleReal = createInCircleTester(coordinates);
  const point = (row: number): [number, number] => [coordinates[2 * row], coordinates[2 * row + 1]];
  const exactPoint = (row: number) => [exact[2 * row], exact[2 * row + 1]];

  /** Sign of orient(a, b, p) for a real site p; a and b may be super rows. */
  function orientSymbolic(a: number, b: number, p: number): number {
    if (a < pointCount && b < pointCount) {
      return orientExact(exactPoint(a), exactPoint(b), exactPoint(p));
    }
    if (a < pointCount) {
      return orientRealSuper(point(a), b - pointCount, point(p));
    }
    if (b < pointCount) {
      return -orientRealSuper(point(b), a - pointCount, point(p));
    }
    return b - pointCount === (a - pointCount + 1) % 3 ? 1 : -1;
  }

  /** True when site p is inside or on the circumcircle of the counter-clockwise triangle. */
  function isInsideCircumcircle(v0: number, v1: number, v2: number, p: number): boolean {
    const rows = [v0, v1, v2];
    let farthestSlot = 3;
    for (let slot = 0; slot < 3; slot++) {
      if (rows[slot] >= pointCount && (farthestSlot === 3 || rows[slot] > rows[farthestSlot])) {
        farthestSlot = slot;
      }
    }
    if (farthestSlot === 3) {
      return inCircleReal(v0, v1, v2, p) >= 0;
    }
    const x = rows[(farthestSlot + 1) % 3];
    const y = rows[(farthestSlot + 2) % 3];
    const side = orientSymbolic(x, y, p);
    if (side !== 0) {
      return side > 0;
    }
    const [ax, ay] = point(x);
    const [bx, by] = point(y);
    const [px, py] = point(p);
    return ax !== bx
      ? px > Math.min(ax, bx) && px < Math.max(ax, bx)
      : py > Math.min(ay, by) && py < Math.max(ay, by);
  }

  return {pointCount, orientSymbolic, isInsideCircumcircle};
}

/**
 * CPU replay of the kernel's Bowyer-Watson loop with the same symbolic predicates, cavity rule
 * (every triangle whose circumcircle holds the site, shared edges removed, edges collinear with
 * the site dropped) and first-row duplicate ownership. Returns the all-real triangles.
 */
export function replaySymbolicDelaunay(coordinates: Float32Array): number[][] {
  const pointCount = coordinates.length / 2;
  if (pointCount < 3) {
    return [];
  }
  const {orientSymbolic, isInsideCircumcircle} = createSymbolicPredicates(coordinates);
  const owners = new Set<string>();
  let triangles: number[][] = [[pointCount, pointCount + 1, pointCount + 2]];
  for (let row = 0; row < pointCount; row++) {
    const key = `${coordinates[2 * row] + 0},${coordinates[2 * row + 1] + 0}`;
    if (owners.has(key)) {
      continue;
    }
    owners.add(key);
    const bad = triangles.map(([a, b, c]) => isInsideCircumcircle(a, b, c, row));
    const survivors = triangles.filter((_, index) => !bad[index]);
    const created: number[][] = [];
    triangles.forEach((triangle, index) => {
      if (!bad[index]) {
        return;
      }
      for (let edge = 0; edge < 3; edge++) {
        const a = triangle[edge];
        const b = triangle[(edge + 1) % 3];
        const shared = triangles.some(
          (other, otherIndex) =>
            otherIndex !== index &&
            bad[otherIndex] &&
            other.some((vertex, k) => {
              const next = other[(k + 1) % 3];
              return (vertex === a && next === b) || (vertex === b && next === a);
            })
        );
        if (!shared && orientSymbolic(a, b, row) !== 0) {
          created.push(orientSymbolic(a, b, row) < 0 ? [b, a, row] : [a, b, row]);
        }
      }
    });
    triangles = [...survivors, ...created];
  }
  return triangles.filter(triangle => triangle.every(vertex => vertex < pointCount));
}

// ---------------------------------------------------------------------------------------------
// Additions for the parallel insertion-and-flipping study (DEL-REF): input generators and
// checks that stay affordable at 1e4 and 1e5 sites.
// ---------------------------------------------------------------------------------------------

/** Gaussian clusters (Box-Muller on `mulberry32`): `clusterCount` centers, sigma 1% of the span. */
export function createClusteredFixture(
  seed: number,
  count: number,
  clusterCount = 8
): Float32Array {
  const random = mulberry32(seed);
  const centers = Array.from({length: clusterCount}, () => [random() * 100, random() * 100]);
  const coordinates = new Float32Array(2 * count);
  for (let row = 0; row < count; row++) {
    const center = centers[Math.floor(random() * clusterCount)];
    const radius = Math.sqrt(-2 * Math.log(1 - random()));
    const angle = 2 * Math.PI * random();
    coordinates[2 * row] = center[0] + radius * Math.cos(angle);
    coordinates[2 * row + 1] = center[1] + radius * Math.sin(angle);
  }
  return coordinates;
}

/** Integer lattice, row-major, `columns` per row, `count` points: maximal cocircular degeneracy. */
export function createLatticeFixture(count: number, columns: number): Float32Array {
  const coordinates = new Float32Array(2 * count);
  for (let row = 0; row < count; row++) {
    coordinates[2 * row] = row % columns;
    coordinates[2 * row + 1] = Math.floor(row / columns);
  }
  return coordinates;
}

/** Points on a circle of radius 100, rounded to f32 (nearly, not exactly, cocircular). */
export function createCircleFixture(seed: number, count: number): Float32Array {
  const random = mulberry32(seed);
  const coordinates = new Float32Array(2 * count);
  for (let row = 0; row < count; row++) {
    const angle = 2 * Math.PI * random();
    coordinates[2 * row] = 100 * Math.cos(angle);
    coordinates[2 * row + 1] = 100 * Math.sin(angle);
  }
  return coordinates;
}

/** `count - 1` collinear sites (0, 0) .. (count - 2, 0) in row order plus one site off the line. */
export function createLineFixture(count: number): Float32Array {
  const coordinates = new Float32Array(2 * count);
  for (let row = 0; row < count - 1; row++) {
    coordinates[2 * row] = row;
    coordinates[2 * row + 1] = 0;
  }
  coordinates[2 * (count - 1)] = (count - 1) / 2 + 0.25;
  coordinates[2 * (count - 1) + 1] = 1;
  return coordinates;
}

/** Uniform sites with rows sorted by x (the worst case for any rule that inserts in row order). */
export function createSortedFixture(seed: number, count: number): Float32Array {
  const source = createUniformFixture(seed, count, 100, 100);
  const order = Array.from({length: count}, (_, row) => row).sort(
    (first, second) => source[2 * first] - source[2 * second]
  );
  const coordinates = new Float32Array(2 * count);
  order.forEach((row, index) => {
    coordinates[2 * index] = source[2 * row];
    coordinates[2 * index + 1] = source[2 * row + 1];
  });
  return coordinates;
}

/**
 * Number of sites on the boundary of the convex hull, collinear boundary sites included (exact).
 * For distinct sites that are not all collinear, a triangulation has `2n - 2 - b` triangles.
 */
export function countHullBoundarySites(coordinates: Float32Array): number {
  const count = coordinates.length / 2;
  const exact = toExactIntegers(coordinates);
  const row = (index: number) => [exact[2 * index], exact[2 * index + 1]];
  const unique = new Map<string, number>();
  for (let site = 0; site < count; site++) {
    const key = `${coordinates[2 * site] + 0},${coordinates[2 * site + 1] + 0}`;
    if (!unique.has(key)) {
      unique.set(key, site);
    }
  }
  const order = [...unique.values()].sort((first, second) => {
    const [fx, fy] = row(first);
    const [sx, sy] = row(second);
    return fx === sx ? (fy < sy ? -1 : fy > sy ? 1 : 0) : fx < sx ? -1 : 1;
  });
  if (order.length < 3) {
    return order.length;
  }
  const chain = (rows: number[]) => {
    const stack: number[] = [];
    for (const index of rows) {
      while (
        stack.length >= 2 &&
        orientExact(row(stack[stack.length - 2]), row(stack[stack.length - 1]), row(index)) < 0
      ) {
        stack.pop();
      }
      stack.push(index);
    }
    return stack;
  };
  const lower = chain(order);
  const upper = chain([...order].reverse());
  return new Set([...lower, ...upper]).size;
}

/**
 * Linear-time validity report: strictly counter-clockwise triangles, each directed edge used
 * once, doubled area equal to the hull's, and every interior edge locally Delaunay (the opposite
 * vertex of the twin triangle is not strictly inside, exact). Local Delaunay everywhere is
 * equivalent to the empty-circle property, so this replaces {@link checkTriangulation}, which
 * is O(triangles * sites), at 1e4 sites and beyond.
 */
export function checkLocallyDelaunay(
  coordinates: Float32Array,
  triangles: readonly (readonly number[])[]
): TriangulationCheck {
  const exact = toExactIntegers(coordinates);
  const row = (index: number) => [exact[2 * index], exact[2 * index + 1]];
  const inCircle = createInCircleTester(coordinates);
  const opposite = new Map<string, number>();
  let degenerate = 0;
  let duplicateEdges = 0;
  let circleViolations = 0;
  let doubledArea = 0n;
  for (const [a, b, c] of triangles) {
    if (orientExact(row(a), row(b), row(c)) <= 0) {
      degenerate++;
      continue;
    }
    const [ax, ay] = row(a);
    const [bx, by] = row(b);
    const [cx, cy] = row(c);
    doubledArea += (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    for (const [from, to, other] of [
      [a, b, c],
      [b, c, a],
      [c, a, b]
    ]) {
      const key = `${from}>${to}`;
      if (opposite.has(key)) {
        duplicateEdges++;
      }
      opposite.set(key, other);
    }
  }
  for (const [a, b, c] of triangles) {
    for (const [from, to, other] of [
      [a, b, c],
      [b, c, a],
      [c, a, b]
    ]) {
      const twin = opposite.get(`${to}>${from}`);
      if (twin !== undefined && from < to && inCircle(from, to, other, twin) > 0) {
        circleViolations++;
      }
    }
  }
  const hullDoubledArea = getHullDoubledArea(coordinates);
  return {
    degenerate,
    circleViolations,
    duplicateEdges,
    areaDifference: doubledArea - hullDoubledArea
  };
}

function getHullDoubledArea(coordinates: Float32Array): bigint {
  const count = coordinates.length / 2;
  const exact = toExactIntegers(coordinates);
  const row = (index: number) => [exact[2 * index], exact[2 * index + 1]];
  const order = Array.from({length: count}, (_, index) => index).sort((first, second) => {
    const [fx, fy] = row(first);
    const [sx, sy] = row(second);
    return fx === sx ? (fy < sy ? -1 : fy > sy ? 1 : 0) : fx < sx ? -1 : 1;
  });
  const chain = (rows: number[]) => {
    const stack: number[] = [];
    for (const index of rows) {
      while (
        stack.length >= 2 &&
        orientExact(row(stack[stack.length - 2]), row(stack[stack.length - 1]), row(index)) <= 0
      ) {
        stack.pop();
      }
      stack.push(index);
    }
    return stack;
  };
  const hull = [...chain(order).slice(0, -1), ...chain([...order].reverse()).slice(0, -1)];
  let area = 0n;
  for (let index = 0; index < hull.length; index++) {
    const [x0, y0] = row(hull[index]);
    const [x1, y1] = row(hull[(index + 1) % hull.length]);
    area += x0 * y1 - x1 * y0;
  }
  return area;
}
