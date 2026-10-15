// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Reconstruct compact fixture INPUT distances; no correlation/permutation implementation. */
export function makeMantelPairs(positions: readonly number[]): {
  offsets: Uint32Array;
  neighbors: Uint32Array;
  distances: Float32Array;
} {
  const count = positions.length / 2;
  const pairCount = (count * (count - 1)) / 2;
  const offsets = new Uint32Array(count + 1);
  const neighbors = new Uint32Array(pairCount);
  const distances = new Float32Array(pairCount);
  let entry = 0;
  for (let row = 0; row < count; row++) {
    offsets[row] = entry;
    for (let neighbor = row + 1; neighbor < count; neighbor++) {
      neighbors[entry] = neighbor;
      distances[entry] = Math.hypot(
        positions[2 * row] - positions[2 * neighbor],
        positions[2 * row + 1] - positions[2 * neighbor + 1]
      );
      entry++;
    }
  }
  offsets[count] = entry;
  return {offsets, neighbors, distances};
}
