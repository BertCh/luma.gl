// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Deterministic xorshift in [0, 1). */
export function createRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 4294967296;
  };
}

/** Returns the group key of a row, or `groupCount` when the row is excluded. */
export function getOracleGroupKeys(
  positions: ArrayLike<number>,
  labels: ArrayLike<number>,
  groupCount: number,
  noiseLabel?: number
): number[] {
  const keys: number[] = [];
  for (let row = 0; row < labels.length; row++) {
    const valid =
      Number.isFinite(positions[row * 2]) &&
      Number.isFinite(positions[row * 2 + 1]) &&
      labels[row] < groupCount &&
      labels[row] !== noiseLabel;
    keys.push(valid ? labels[row] : groupCount);
  }
  return keys;
}

/** Per-group bounds, counts and medoids in float64, lowest row index on ties. */
export function computeGroupGeometryOracle(
  positions: ArrayLike<number>,
  keys: readonly number[],
  groupCount: number
) {
  const counts = new Array(groupCount).fill(0);
  const bounds = new Array(groupCount * 4).fill(NaN);
  const medoids = new Array(groupCount).fill(0xffffffff);
  for (let group = 0; group < groupCount; group++) {
    const rows: number[] = [];
    for (let row = 0; row < keys.length; row++) {
      if (keys[row] === group) {
        rows.push(row);
      }
    }
    counts[group] = rows.length;
    if (rows.length === 0) {
      continue;
    }
    let best = Infinity;
    for (const row of rows) {
      let cost = 0;
      for (const other of rows) {
        cost += Math.hypot(
          positions[other * 2] - positions[row * 2],
          positions[other * 2 + 1] - positions[row * 2 + 1]
        );
      }
      if (cost < best) {
        best = cost;
        medoids[group] = row;
      }
    }
    const xs = rows.map(row => positions[row * 2]);
    const ys = rows.map(row => positions[row * 2 + 1]);
    bounds.splice(group * 4, 4, Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys));
  }
  return {counts, bounds, medoids};
}

/** Result of {@link computeGroupConvexHullOracle}. */
export type ConvexHullOracleResult = {
  /** Original row indices of the hull per group, counter-clockwise from the smallest vertex. */
  hulls: number[][];
};

/** Lattice quantization of the contributor: global power-of-two scale, f32 arithmetic. */
export function quantizeToLattice(
  positions: ArrayLike<number>,
  valid: readonly boolean[]
): {x: number[]; y: number[]} {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let row = 0; row < valid.length; row++) {
    if (valid[row]) {
      minX = Math.min(minX, positions[row * 2]);
      maxX = Math.max(maxX, positions[row * 2]);
      minY = Math.min(minY, positions[row * 2 + 1]);
      maxY = Math.max(maxY, positions[row * 2 + 1]);
    }
  }
  const extent = Math.max(Math.fround(maxX - minX), Math.fround(maxY - minY));
  let scale = 1;
  if (extent > 0) {
    const exponent = Math.max(Math.floor(Math.log2(extent)), -99);
    scale = 2 ** (28 - exponent);
  }
  const x: number[] = [];
  const y: number[] = [];
  for (let row = 0; row < valid.length; row++) {
    if (!valid[row]) {
      x.push(0);
      y.push(0);
      continue;
    }
    x.push(Math.floor(Math.fround(positions[row * 2] - minX) * scale + 0.5));
    y.push(Math.floor(Math.fround(positions[row * 2 + 1] - minY) * scale + 0.5));
  }
  return {x, y};
}

function cross(
  a: readonly [number, number],
  b: readonly [number, number],
  c: readonly [number, number]
): bigint {
  return BigInt(b[0] - a[0]) * BigInt(c[1] - a[1]) - BigInt(b[1] - a[1]) * BigInt(c[0] - a[0]);
}

/**
 * Monotone chain on the same lattice as the contributor with exact BigInt orientation. Equal
 * lattice points collapse to the lowest row index. Collinear points are dropped.
 */
export function computeGroupConvexHullOracle(
  positions: ArrayLike<number>,
  keys: readonly number[],
  groupCount: number
): ConvexHullOracleResult {
  const lattice = quantizeToLattice(
    positions,
    keys.map(key => key < groupCount)
  );
  const hulls: number[][] = [];
  for (let group = 0; group < groupCount; group++) {
    const rows: number[] = [];
    for (let row = 0; row < keys.length; row++) {
      if (keys[row] === group) {
        rows.push(row);
      }
    }
    rows.sort((a, b) => lattice.x[a] - lattice.x[b] || lattice.y[a] - lattice.y[b] || a - b);
    const unique = rows.filter(
      (row, index) =>
        index === 0 ||
        lattice.x[row] !== lattice.x[rows[index - 1]] ||
        lattice.y[row] !== lattice.y[rows[index - 1]]
    );
    const point = (row: number): [number, number] => [lattice.x[row], lattice.y[row]];
    if (unique.length <= 1) {
      hulls.push(unique);
      continue;
    }
    const stack: number[] = [];
    for (const row of unique) {
      while (
        stack.length >= 2 &&
        cross(point(stack[stack.length - 2]), point(stack[stack.length - 1]), point(row)) <= 0n
      ) {
        stack.pop();
      }
      stack.push(row);
    }
    const lowerLength = stack.length + 1;
    for (let index = unique.length - 2; index >= 0; index--) {
      const row = unique[index];
      while (
        stack.length >= lowerLength &&
        cross(point(stack[stack.length - 2]), point(stack[stack.length - 1]), point(row)) <= 0n
      ) {
        stack.pop();
      }
      stack.push(row);
    }
    stack.pop();
    hulls.push(stack);
  }
  return {hulls};
}
