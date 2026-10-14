// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {OracleCSR} from '../spatial-weights/spatial-weights-oracle';

/** Mirrors `priorityHash` of `GPUMapColoring`. */
export function getPriorityHash(value: number, seed: number): number {
  let h = (value ^ seed) >>> 0;
  h = (h ^ (h >>> 16)) >>> 0;
  h = Math.imul(h, 0x7feb352d) >>> 0;
  h = (h ^ (h >>> 15)) >>> 0;
  h = Math.imul(h, 0x846ca68b) >>> 0;
  h = (h ^ (h >>> 16)) >>> 0;
  return h;
}

/** Mirrors the `priority` prop of `GPUMapColoring`. */
export type OraclePriority = 'largest-degree-first' | 'random';

/** Returns the rows in claim order: (degree desc,) hash desc, ID asc. Degree is the CSR row length. */
export function getMapColoringOrder(
  csr: OracleCSR,
  seed: number,
  priority: OraclePriority = 'largest-degree-first'
): number[] {
  const rows = csr.offsets.length - 1;
  const degree = (row: number) =>
    priority === 'largest-degree-first' ? csr.offsets[row + 1] - csr.offsets[row] : 0;
  return Array.from({length: rows}, (_, row) => row).sort((a, b) => {
    if (degree(a) !== degree(b)) return degree(b) - degree(a);
    const hashA = getPriorityHash(a, seed);
    const hashB = getPriorityHash(b, seed);
    return hashA !== hashB ? hashB - hashA : a - b;
  });
}

/**
 * Sequential greedy coloring in claim order (see {@link getMapColoringOrder}), which is exactly
 * what the parallel Jones-Plassmann rounds converge to.
 */
export function computeMapColoringOracle(
  csr: OracleCSR,
  seed: number,
  priority: OraclePriority = 'largest-degree-first'
): number[] {
  const rows = csr.offsets.length - 1;
  const colors = new Array<number>(rows).fill(-1);
  for (const row of getMapColoringOrder(csr, seed, priority)) {
    const used = new Set<number>();
    for (let slot = csr.offsets[row]; slot < csr.offsets[row + 1]; slot++) {
      const color = colors[csr.neighbors[slot]];
      if (color >= 0) used.add(color);
    }
    let color = 0;
    while (used.has(color)) color++;
    colors[row] = color;
  }
  return colors;
}

/**
 * TS port of the kernel's rounds with synchronous (two-phase) scheduling: a row colors in a round
 * when every uncolored neighbor is outranked by it. Returns colors and the rounds that colored at
 * least one row. The fused GPU pass needs the same or fewer rounds and yields the same colors.
 */
export function simulateMapColoringRounds(
  csr: OracleCSR,
  seed: number,
  priority: OraclePriority = 'largest-degree-first'
): {colors: number[]; rounds: number} {
  const rows = csr.offsets.length - 1;
  const degree = (row: number) =>
    priority === 'largest-degree-first' ? csr.offsets[row + 1] - csr.offsets[row] : 0;
  const outranks = (other: number, row: number) => {
    if (degree(other) !== degree(row)) return degree(other) > degree(row);
    const otherHash = getPriorityHash(other, seed);
    const rowHash = getPriorityHash(row, seed);
    return otherHash > rowHash || (otherHash === rowHash && other < row);
  };
  const colors = new Array<number>(rows).fill(-1);
  let rounds = 0;
  for (;;) {
    const ready: number[] = [];
    for (let row = 0; row < rows; row++) {
      if (colors[row] >= 0) continue;
      let blocked = false;
      for (let slot = csr.offsets[row]; slot < csr.offsets[row + 1]; slot++) {
        const other = csr.neighbors[slot];
        if (other !== row && colors[other] < 0 && outranks(other, row)) {
          blocked = true;
          break;
        }
      }
      if (!blocked) ready.push(row);
    }
    if (ready.length === 0) break;
    rounds++;
    for (const row of ready) {
      const used = new Set<number>();
      for (let slot = csr.offsets[row]; slot < csr.offsets[row + 1]; slot++) {
        const other = csr.neighbors[slot];
        if (other !== row && colors[other] >= 0) used.add(colors[other]);
      }
      let color = 0;
      while (used.has(color)) color++;
      colors[row] = color;
    }
  }
  return {colors, rounds};
}

/** Builds a symmetric CSR from an undirected edge list (sorted rows, no duplicates, no self loops). */
export function createOracleCSRFromEdges(rows: number, edges: [number, number][]): OracleCSR {
  const sets = Array.from({length: rows}, () => new Set<number>());
  for (const [a, b] of edges) {
    if (a === b) continue;
    sets[a].add(b);
    sets[b].add(a);
  }
  const csr: OracleCSR = {offsets: [0], neighbors: [], weights: [], distances: []};
  for (const set of sets) {
    for (const neighbor of [...set].sort((x, y) => x - y)) {
      csr.neighbors.push(neighbor);
      csr.weights.push(1);
    }
    csr.offsets.push(csr.neighbors.length);
  }
  return csr;
}

/** Rook lattice graph of `columns x rows` cells. */
export function createRookLatticeCSR(columns: number, rows: number): OracleCSR {
  const edges: [number, number][] = [];
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < columns; x++) {
      if (x + 1 < columns) edges.push([y * columns + x, y * columns + x + 1]);
      if (y + 1 < rows) edges.push([y * columns + x, (y + 1) * columns + x]);
    }
  }
  return createOracleCSRFromEdges(columns * rows, edges);
}

/**
 * Random planar map graph: a lattice with one randomly chosen diagonal per cell and a random
 * fraction of lattice edges dropped. Planar by construction (no crossing edges).
 */
export function createRandomPlanarCSR(
  columns: number,
  rows: number,
  random: () => number
): OracleCSR {
  const edges: [number, number][] = [];
  const id = (x: number, y: number) => y * columns + x;
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < columns; x++) {
      if (x + 1 < columns && random() < 0.9) edges.push([id(x, y), id(x + 1, y)]);
      if (y + 1 < rows && random() < 0.9) edges.push([id(x, y), id(x, y + 1)]);
      if (x + 1 < columns && y + 1 < rows) {
        edges.push(random() < 0.5 ? [id(x, y), id(x + 1, y + 1)] : [id(x + 1, y), id(x, y + 1)]);
      }
    }
  }
  return createOracleCSRFromEdges(columns * rows, edges);
}

/** Star graph: row 0 is the hub of `leaves` leaves. */
export function createStarCSR(leaves: number): OracleCSR {
  return createOracleCSRFromEdges(
    leaves + 1,
    Array.from({length: leaves}, (_, leaf): [number, number] => [0, leaf + 1])
  );
}

/** Complete graph on `size` rows. */
export function createCliqueCSR(size: number): OracleCSR {
  const edges: [number, number][] = [];
  for (let a = 0; a < size; a++) for (let b = a + 1; b < size; b++) edges.push([a, b]);
  return createOracleCSRFromEdges(size, edges);
}

/** Counts neighbor pairs `(i, j)`, `j > i`, that share a color. */
export function countColoringConflicts(csr: OracleCSR, colors: readonly number[]): number {
  let conflicts = 0;
  for (let row = 0; row < colors.length; row++) {
    for (let slot = csr.offsets[row]; slot < csr.offsets[row + 1]; slot++) {
      if (csr.neighbors[slot] > row && colors[csr.neighbors[slot]] === colors[row]) conflicts++;
    }
  }
  return conflicts;
}
