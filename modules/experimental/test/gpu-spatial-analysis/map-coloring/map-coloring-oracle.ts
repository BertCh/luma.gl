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

/**
 * Sequential greedy coloring in descending priority order (larger hash first, then lower ID),
 * which is exactly what the parallel Jones-Plassmann rounds converge to.
 */
export function computeMapColoringOracle(csr: OracleCSR, seed: number): number[] {
  const rows = csr.offsets.length - 1;
  const order = Array.from({length: rows}, (_, row) => row).sort((a, b) => {
    const hashA = getPriorityHash(a, seed);
    const hashB = getPriorityHash(b, seed);
    return hashA !== hashB ? hashB - hashA : a - b;
  });
  const colors = new Array<number>(rows).fill(-1);
  for (const row of order) {
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
