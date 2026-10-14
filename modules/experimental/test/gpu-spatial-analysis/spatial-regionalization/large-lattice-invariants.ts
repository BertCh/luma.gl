// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createSeededRandom} from '../spatial-weights/spatial-weights-oracle';

/** Rook-contiguity CSR of a width x height lattice, neighbors ascending per row, unit weights. */
export function createRookLattice(width: number, height: number) {
  const offsets = [0];
  const neighbors: number[] = [];
  for (let row = 0; row < width * height; row++) {
    const x = row % width;
    for (const candidate of [
      row - width,
      x > 0 ? row - 1 : -1,
      x < width - 1 ? row + 1 : -1,
      row + width
    ]) {
      if (candidate >= 0 && candidate < width * height) neighbors.push(candidate);
    }
    offsets.push(neighbors.length);
  }
  return {
    offsets: Uint32Array.from(offsets),
    neighbors: Uint32Array.from(neighbors),
    weights: Float32Array.from({length: neighbors.length}, () => 1)
  };
}

/** Deterministic pseudo-random float32 attribute in [0, scale). */
export function createRandomValues(count: number, seed: number, scale = 10): Float32Array {
  const random = createSeededRandom(seed);
  return Float32Array.from({length: count}, () => Math.fround(random() * scale));
}

/** Asserts-by-return: the labels whose region is not connected through the weights (BFS). */
export function findDisconnectedLabels(
  offsets: ArrayLike<number>,
  neighbors: ArrayLike<number>,
  labels: ArrayLike<number>
): number[] {
  const rows = labels.length;
  const visited = new Uint8Array(rows);
  const seenLabels = new Set<number>();
  const disconnected: number[] = [];
  for (let start = 0; start < rows; start++) {
    if (visited[start]) continue;
    const label = labels[start];
    if (seenLabels.has(label)) {
      // A second connected component with the same label.
      disconnected.push(label);
    }
    seenLabels.add(label);
    const queue = [start];
    visited[start] = 1;
    for (let head = 0; head < queue.length; head++) {
      const row = queue[head];
      for (let slot = offsets[row]; slot < offsets[row + 1]; slot++) {
        const neighbor = neighbors[slot];
        if (!visited[neighbor] && labels[neighbor] === label) {
          visited[neighbor] = 1;
          queue.push(neighbor);
        }
      }
    }
  }
  return disconnected;
}
