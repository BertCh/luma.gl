// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Shared CPU helpers for the pair-statistics oracles. Not part of the public barrel. */

/** Planar point scene shared by the pair-statistics oracles and tests. */
export type PairStatisticsScene = {
  positions: Float32Array;
  values?: Float32Array;
  mask?: Uint32Array;
};

/** Per-frame window and distance shared by the pair-statistics oracles. */
export type PairStatisticsFrame = {
  bounds: readonly [number, number, number, number];
  maximumDistance: number;
};

const fround = Math.fround;

/** Deterministic xorshift in `[0, 1)`. */
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

/** Returns the included rows, mirroring the GPU inclusion rule. */
export function getIncludedRows(scene: PairStatisticsScene, frame: PairStatisticsFrame): number[] {
  const [minX, minY, maxX, maxY] = frame.bounds.map(fround);
  const rows = scene.positions.length / 2;
  const included: number[] = [];
  for (let row = 0; row < rows; row++) {
    const x = scene.positions[row * 2];
    const y = scene.positions[row * 2 + 1];
    if (scene.mask && scene.mask[row] === 0) {
      continue;
    }
    if (scene.values && !Number.isFinite(scene.values[row])) {
      continue;
    }
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      continue;
    }
    if (x < minX || x > maxX || y < minY || y > maxY) {
      continue;
    }
    included.push(row);
  }
  return included;
}

/** One visited pair with the f32 offsets and distance the GPU kernel computes. */
export type PairVisit = {
  focus: number;
  neighbor: number;
  deltaX: number;
  deltaY: number;
  distance: number;
};

/**
 * Calls `visit` for every pair of included rows within `maximumDistance`, evaluating the offsets,
 * squared distance and threshold in f32 like the GPU. `'unordered'` visits `i < j` once,
 * `'ordered'` both directions.
 */
export function forEachPair(
  scene: PairStatisticsScene,
  frame: PairStatisticsFrame,
  pairOrder: 'unordered' | 'ordered',
  visit: (pair: PairVisit) => void
): void {
  const included = getIncludedRows(scene, frame);
  const radius = fround(frame.maximumDistance);
  const radiusSquared = fround(radius * radius);
  for (const focus of included) {
    const x = scene.positions[focus * 2];
    const y = scene.positions[focus * 2 + 1];
    for (const neighbor of included) {
      if (pairOrder === 'unordered' ? neighbor <= focus : neighbor === focus) {
        continue;
      }
      const deltaX = fround(scene.positions[neighbor * 2] - x);
      const deltaY = fround(scene.positions[neighbor * 2 + 1] - y);
      const distanceSquared = fround(fround(deltaX * deltaX) + fround(deltaY * deltaY));
      if (distanceSquared <= radiusSquared) {
        visit({focus, neighbor, deltaX, deltaY, distance: fround(Math.sqrt(distanceSquared))});
      }
    }
  }
}

/** Returns the f32 bin `min(floor(distance / maximumDistance * binCount), binCount - 1)`. */
export function getDistanceBin(
  distance: number,
  maximumDistance: number,
  binCount: number
): number {
  const scaled = fround(fround(distance / fround(maximumDistance)) * binCount);
  return Math.min(Math.floor(scaled), binCount - 1);
}

/**
 * True when `distance` lies so close to a bin edge (relative `1e-5` of a bin) that GPU division
 * rounding may put the pair in the neighboring bin.
 */
export function isNearBinEdge(
  distance: number,
  maximumDistance: number,
  binCount: number
): boolean {
  const scaled = (distance / maximumDistance) * binCount;
  return (
    Math.abs(scaled - Math.round(scaled)) < 1e-5 &&
    Math.round(scaled) > 0 &&
    Math.round(scaled) < binCount
  );
}

/** Writes `[minX, minY, maxX, maxY, maximumDistance]` followed by `extra` into a parameter row. */
export function getSharedParameterValues(frame: PairStatisticsFrame): number[] {
  return [...frame.bounds, frame.maximumDistance];
}
