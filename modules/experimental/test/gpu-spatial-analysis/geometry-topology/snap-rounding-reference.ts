// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Exact BigInt reference of snap-rounded noding on the integer lattice (Hobby; Guibas and
// Marimont; GEOS SnapRoundingNoder with a fixed precision model), written independently of the
// WGSL: crossings and half-open pixel incidence use rational arithmetic, the order along a segment
// comes from the parameter of the middle of each pixel's overlap interval (not from the
// lexicographic shortcut of the kernel), and the world-to-lattice frame uses exact dyadic
// arithmetic. It is O(segments^2 x pixels), for small inputs only.

/** A lattice point (integer coordinates). */
export type LatticePoint = readonly [number, number];

/** One output piece: the source segment (start-vertex row) and its lattice endpoints. */
export type ReferencePiece = {segment: number; from: LatticePoint; to: LatticePoint};

const signOf = (value: bigint): number => (value > 0n ? 1 : value < 0n ? -1 : 0);

/** Floor of n / d for d > 0. */
export function floorDivide(numerator: bigint, denominator: bigint): bigint {
  const quotient = numerator / denominator;
  return numerator % denominator !== 0n && numerator < 0n !== denominator < 0n
    ? quotient - 1n
    : quotient;
}

function orientation(a: LatticePoint, b: LatticePoint, c: LatticePoint): number {
  const value =
    BigInt(b[0] - a[0]) * BigInt(c[1] - a[1]) - BigInt(b[1] - a[1]) * BigInt(c[0] - a[0]);
  return signOf(value);
}

/** Whether the interiors of ab and cd cross at one point (all four orientations are non-zero). */
export function isProperCrossing(
  a: LatticePoint,
  b: LatticePoint,
  c: LatticePoint,
  d: LatticePoint
): boolean {
  const o1 = orientation(a, b, c);
  const o2 = orientation(a, b, d);
  const o3 = orientation(c, d, a);
  const o4 = orientation(c, d, b);
  return o1 * o2 < 0 && o3 * o4 < 0;
}

/**
 * The proper crossing of ab and cd rounded half up to the lattice, computed by exact division, or
 * `undefined` when the segments do not cross properly.
 */
export function referenceRoundedCrossing(
  a: LatticePoint,
  b: LatticePoint,
  c: LatticePoint,
  d: LatticePoint
): LatticePoint | undefined {
  if (!isProperCrossing(a, b, c, d)) {
    return undefined;
  }
  let denominator =
    BigInt(b[0] - a[0]) * BigInt(d[1] - c[1]) - BigInt(b[1] - a[1]) * BigInt(d[0] - c[0]);
  let numerator =
    BigInt(c[0] - a[0]) * BigInt(d[1] - c[1]) - BigInt(c[1] - a[1]) * BigInt(d[0] - c[0]);
  if (denominator < 0n) {
    denominator = -denominator;
    numerator = -numerator;
  }
  // x = a + N (b - a) / D; floor(x + 1/2) = a + floor((2 N (b - a) + D) / (2 D)).
  const round = (origin: number, target: number): number =>
    origin +
    Number(floorDivide(2n * numerator * BigInt(target - origin) + denominator, 2n * denominator));
  return [round(a[0], b[0]), round(a[1], b[1])];
}

type Fraction = {n: bigint; d: bigint};
const compareFractions = (left: Fraction, right: Fraction): number =>
  signOf(left.n * right.d - right.n * left.d);
const fraction = (n: bigint, d: bigint): Fraction => (d < 0n ? {n: -n, d: -d} : {n, d});

type Bound = {value: Fraction; closed: boolean};

/**
 * The parameter interval of segment ab inside the half-open pixel
 * `[c - 1/2, c + 1/2) x [c - 1/2, c + 1/2)` (the closed square when `closedSquare`) as its midpoint
 * parameter, or `undefined` when the segment misses the pixel. Exact rational arithmetic on doubled coordinates.
 */
export function referencePixelParameter(
  a: LatticePoint,
  b: LatticePoint,
  center: LatticePoint,
  closedSquare = false
): Fraction | undefined {
  let lower: Bound = {value: {n: 0n, d: 1n}, closed: true};
  let upper: Bound = {value: {n: 1n, d: 1n}, closed: true};
  const tighten = (candidate: Bound, isLower: boolean) => {
    const current = isLower ? lower : upper;
    const order = compareFractions(candidate.value, current.value);
    const replaces = isLower ? order > 0 : order < 0;
    const merges = order === 0 && !candidate.closed;
    if (replaces || merges) {
      if (isLower) {
        lower = candidate;
      } else {
        upper = candidate;
      }
    }
  };
  for (const axis of [0, 1]) {
    // Doubled coordinates: pixel [2c - 1, 2c + 1), segment 2a + t (2b - 2a).
    const start = BigInt(2 * a[axis]);
    const delta = BigInt(2 * (b[axis] - a[axis]));
    const low = BigInt(2 * center[axis] - 1);
    const high = BigInt(2 * center[axis] + 1);
    if (delta === 0n) {
      if (!(low <= start && (closedSquare ? start <= high : start < high))) {
        return undefined;
      }
      continue;
    }
    // low <= start + t delta < high
    const atLow = fraction(low - start, delta);
    const atHigh = fraction(high - start, delta);
    if (delta > 0n) {
      tighten({value: atLow, closed: true}, true);
      tighten({value: atHigh, closed: closedSquare}, false);
    } else {
      tighten({value: atHigh, closed: closedSquare}, true);
      tighten({value: atLow, closed: true}, false);
    }
  }
  const order = compareFractions(lower.value, upper.value);
  if (order > 0 || (order === 0 && !(lower.closed && upper.closed))) {
    return undefined;
  }
  return fraction(
    lower.value.n * upper.value.d + upper.value.n * lower.value.d,
    2n * lower.value.d * upper.value.d
  );
}

/** Whether segment ab meets the half-open pixel around `center` (or its closed square). */
export function referenceMeetsPixel(
  a: LatticePoint,
  b: LatticePoint,
  center: LatticePoint,
  closedSquare = false
): boolean {
  return referencePixelParameter(a, b, center, closedSquare) !== undefined;
}

/** A line of lattice vertices; `null` marks a non-finite vertex. */
export type ReferenceLine = readonly (LatticePoint | null)[];

/** Result of {@link referenceSnapRound}. */
export type ReferenceNoding = {
  /** Pieces ordered by segment and position along the segment. */
  pieces: ReferencePiece[];
  /** Hot pixel centres, ordered by (x, y). */
  hotPixels: LatticePoint[];
  /** Number of usable (finite, non-collapsed) segments. */
  usableSegments: number;
  /** Rounded proper crossings, as (left segment, right segment, point). */
  crossings: {left: number; right: number; point: LatticePoint}[];
};

/** Snap rounding of lattice lines (vertex rows are numbered line by line). */
export function referenceSnapRound(lines: readonly ReferenceLine[]): ReferenceNoding {
  const vertices: (LatticePoint | null)[] = [];
  const segments: {id: number; a: LatticePoint; b: LatticePoint}[] = [];
  const hotKeys = new Set<string>();
  const hot: LatticePoint[] = [];
  const addHot = (point: LatticePoint) => {
    const key = `${point[0]},${point[1]}`;
    if (!hotKeys.has(key)) {
      hotKeys.add(key);
      hot.push(point);
    }
  };
  for (const line of lines) {
    const base = vertices.length;
    vertices.push(...line);
    const usable = (row: number): boolean => {
      const start = line[row];
      const end = line[row + 1];
      return (
        row >= 0 &&
        row + 1 < line.length &&
        start !== null &&
        end !== null &&
        (start[0] !== end[0] || start[1] !== end[1])
      );
    };
    for (let row = 0; row < line.length; row++) {
      if (usable(row)) {
        segments.push({id: base + row, a: line[row]!, b: line[row + 1]!});
      }
      if (usable(row) || usable(row - 1)) {
        addHot(line[row]!);
      }
    }
  }
  const crossings: ReferenceNoding['crossings'] = [];
  for (let left = 0; left < segments.length; left++) {
    for (let right = left + 1; right < segments.length; right++) {
      const point = referenceRoundedCrossing(
        segments[left].a,
        segments[left].b,
        segments[right].a,
        segments[right].b
      );
      if (point) {
        crossings.push({left: segments[left].id, right: segments[right].id, point});
        addHot(point);
      }
    }
  }
  hot.sort((p, q) => p[0] - q[0] || p[1] - q[1]);
  const pieces: ReferencePiece[] = [];
  for (const segment of segments) {
    const visited = hot
      .map(center => ({center, parameter: referencePixelParameter(segment.a, segment.b, center)}))
      .filter(
        (entry): entry is {center: LatticePoint; parameter: Fraction} =>
          entry.parameter !== undefined
      )
      .sort((p, q) => compareFractions(p.parameter, q.parameter));
    for (let row = 0; row + 1 < visited.length; row++) {
      pieces.push({segment: segment.id, from: visited[row].center, to: visited[row + 1].center});
    }
  }
  return {pieces, hotPixels: hot, usableSegments: segments.length, crossings};
}

/**
 * The snap-rounding guarantee for `pieces` against the hot pixels: every endpoint is a hot pixel
 * centre, and any two pieces are identical (either direction) or meet only at a shared endpoint.
 * Returns the list of violations (empty when the guarantee holds).
 */
export function checkSnapRoundingGuarantee(
  pieces: readonly ReferencePiece[],
  hotPixels: readonly LatticePoint[]
): string[] {
  const problems: string[] = [];
  const hotKeys = new Set(hotPixels.map(point => `${point[0]},${point[1]}`));
  const isEndpoint = (point: LatticePoint, piece: ReferencePiece) =>
    (point[0] === piece.from[0] && point[1] === piece.from[1]) ||
    (point[0] === piece.to[0] && point[1] === piece.to[1]);
  for (const piece of pieces) {
    for (const end of [piece.from, piece.to]) {
      if (!hotKeys.has(`${end[0]},${end[1]}`)) {
        problems.push(`piece of segment ${piece.segment} ends at non-hot ${end}`);
      }
    }
  }
  for (let first = 0; first < pieces.length; first++) {
    for (let second = first + 1; second < pieces.length; second++) {
      const p = pieces[first];
      const q = pieces[second];
      const same =
        (p.from[0] === q.from[0] &&
          p.from[1] === q.from[1] &&
          p.to[0] === q.to[0] &&
          p.to[1] === q.to[1]) ||
        (p.from[0] === q.to[0] &&
          p.from[1] === q.to[1] &&
          p.to[0] === q.from[0] &&
          p.to[1] === q.from[1]);
      if (same) {
        continue;
      }
      const o1 = orientation(p.from, p.to, q.from);
      const o2 = orientation(p.from, p.to, q.to);
      const o3 = orientation(q.from, q.to, p.from);
      const o4 = orientation(q.from, q.to, p.to);
      if (o1 * o2 < 0 && o3 * o4 < 0) {
        problems.push(`pieces ${first} and ${second} cross`);
        continue;
      }
      const touching: LatticePoint[] = [];
      // A vertex of one on the other (closed): T-junction or shared endpoint or overlap.
      const within = (point: LatticePoint, a: LatticePoint, b: LatticePoint) =>
        orientation(a, b, point) === 0 &&
        Math.min(a[0], b[0]) <= point[0] &&
        point[0] <= Math.max(a[0], b[0]) &&
        Math.min(a[1], b[1]) <= point[1] &&
        point[1] <= Math.max(a[1], b[1]);
      for (const point of [p.from, p.to]) {
        if (within(point, q.from, q.to)) {
          touching.push(point);
        }
      }
      for (const point of [q.from, q.to]) {
        if (within(point, p.from, p.to)) {
          touching.push(point);
        }
      }
      for (const point of touching) {
        if (!isEndpoint(point, p) || !isEndpoint(point, q)) {
          problems.push(
            `pieces ${first} and ${second} touch at ${point} away from a shared endpoint`
          );
        }
      }
      const distinct = new Set(touching.map(point => `${point[0]},${point[1]}`));
      if (distinct.size > 1) {
        problems.push(`pieces ${first} and ${second} overlap along a sub-segment`);
      }
    }
  }
  return problems;
}

// ---- World to lattice frame (exact dyadic arithmetic) ----------------------------------------

const FRACTION_BITS = 300n;

/** The value of a finite double as an integer multiple of 2^-300. */
export function toScaledInteger(value: number): bigint {
  if (value === 0) {
    return 0n;
  }
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, value);
  const bits = view.getBigUint64(0);
  const negative = bits >> 63n === 1n;
  const exponent = Number((bits >> 52n) & 0x7ffn);
  let mantissa = bits & 0xfffffffffffffn;
  let power = exponent - 1075;
  if (exponent === 0) {
    power = -1074;
  } else {
    mantissa |= 1n << 52n;
  }
  const scaled = mantissa << (FRACTION_BITS + BigInt(power));
  return negative ? -scaled : scaled;
}

/** Smallest power of two (as an exponent) that is at least `value` > 0, for finite doubles. */
function ceilLog2(value: number): number {
  let exponent = Math.ceil(Math.log2(value));
  while (2 ** exponent < value) {
    exponent++;
  }
  while (2 ** (exponent - 1) >= value) {
    exponent--;
  }
  return exponent;
}

/** The lattice frame and snapped vertices that `GPUSnapRoundedNoding` computes. */
export type ReferenceFrame = {
  origin: [number, number];
  cell: number;
  /** Snapped vertices; `null` for non-finite inputs. */
  points: (LatticePoint | null)[];
};

/**
 * Frame and rounding of float32 `positions` (x, y pairs): `cell` is the smallest power of two at
 * least `max(gridSize, E 2^-22)` with E the smallest power of two at least the float32 extent, the
 * origin is `floor(min / cell) cell`, and vertices round half up.
 */
export function referenceLatticeFrame(
  positions: ArrayLike<number>,
  gridSize: number
): ReferenceFrame {
  const count = positions.length / 2;
  const finite = (row: number) =>
    Number.isFinite(positions[2 * row]) && Number.isFinite(positions[2 * row + 1]);
  let minimum = [Infinity, Infinity];
  let maximum = [-Infinity, -Infinity];
  for (let row = 0; row < count; row++) {
    if (finite(row)) {
      for (const axis of [0, 1]) {
        minimum[axis] = Math.min(minimum[axis], positions[2 * row + axis]);
        maximum[axis] = Math.max(maximum[axis], positions[2 * row + axis]);
      }
    }
  }
  if (minimum[0] === Infinity) {
    minimum = [0, 0];
    maximum = [0, 0];
  }
  const extent = Math.max(
    Math.fround(maximum[0] - minimum[0]),
    Math.fround(maximum[1] - minimum[1])
  );
  const extentExponent = extent > 2 ** -126 ? ceilLog2(extent) : -126;
  const gridExponent = Math.min(127, Math.max(-126, ceilLog2(gridSize)));
  const cellExponent = Math.max(gridExponent, extentExponent - 22);
  const cell = 2 ** cellExponent;
  const scaledCell = 1n << (FRACTION_BITS + BigInt(cellExponent));
  const origin: [number, number] = [0, 0];
  const originScaled: bigint[] = [0n, 0n];
  for (const axis of [0, 1]) {
    originScaled[axis] = floorDivide(toScaledInteger(minimum[axis]), scaledCell) * scaledCell;
    origin[axis] = Number(originScaled[axis]) / 2 ** Number(FRACTION_BITS);
  }
  const points = Array.from({length: count}, (_, row): LatticePoint | null => {
    if (!finite(row)) {
      return null;
    }
    const lattice = [0, 1].map(axis =>
      Number(
        floorDivide(
          2n * (toScaledInteger(positions[2 * row + axis]) - originScaled[axis]) + scaledCell,
          2n * scaledCell
        )
      )
    );
    return [lattice[0], lattice[1]];
  });
  return {origin, cell, points};
}
