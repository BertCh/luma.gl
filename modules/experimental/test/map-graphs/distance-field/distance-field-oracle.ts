// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_DISTANCE_FIELD_NONE} from '../../../src/map-graphs/distance-field';

/** CPU description of one distance-field run. */
export type DistanceFieldScene = {
  width: number;
  height: number;
  /** Packed settings, see `getGPUDistanceFieldParameterValues`. */
  settings: Float32Array;
  /** Interleaved seed positions; the length / 2 is the seed capacity. */
  positions?: Float32Array;
  /** Optional seed ID per point row, default the row index. */
  ids?: Uint32Array;
  /** Optional active point count. */
  count?: number;
  /** Optional per-cell seed raster, value `v` is seed ID `v - 1`. */
  mask?: Uint32Array;
};

/** Brute-force reference result. */
export type DistanceFieldOracleResult = {
  /** f64 Euclidean ground distance per cell, `Infinity` when unreached. */
  distances: Float64Array;
  /** Smallest-ID nearest seed per cell. */
  allocation: Uint32Array;
  /** Row-major cell of the nearest seed. */
  nearestCells: Uint32Array;
  /** Seed ID per cell after snapping (smallest ID per cell), `NONE` when the cell holds no seed. */
  cellSeeds: Uint32Array;
};

const NONE = GPU_DISTANCE_FIELD_NONE;

/** Snaps seeds to cells exactly like the GPU scatter kernel (f32 subtract, f32 multiply, floor). */
export function getCellSeeds(scene: DistanceFieldScene): Uint32Array {
  const {width, height, settings} = scene;
  const cellSeeds = new Uint32Array(width * height).fill(NONE);
  if (scene.mask) {
    for (let cell = 0; cell < cellSeeds.length; cell++) {
      cellSeeds[cell] = (scene.mask[cell] - 1) >>> 0;
    }
  }
  const positions = scene.positions ?? new Float32Array(0);
  const capacity = positions.length / 2;
  const count = Math.min(scene.count ?? capacity, capacity);
  for (let row = 0; row < count; row++) {
    const id = scene.ids ? scene.ids[row] : row;
    if (id === NONE) {
      continue;
    }
    const column = Math.floor(
      Math.fround(Math.fround(positions[row * 2] - settings[0]) * settings[4])
    );
    const gridRow = Math.floor(
      Math.fround(Math.fround(positions[row * 2 + 1] - settings[1]) * settings[5])
    );
    if (!(column >= 0 && column < width && gridRow >= 0 && gridRow < height)) {
      continue;
    }
    const cell = gridRow * width + column;
    cellSeeds[cell] = Math.min(cellSeeds[cell], id);
  }
  return cellSeeds;
}

/**
 * Brute-force nearest seed per cell: every cell against every seed cell.
 *
 * Isotropic grids compare exact integer squared offsets; anisotropic grids compare f64 squared
 * ground distances. Equal keys take the smaller seed ID.
 */
export function computeDistanceFieldOnCPU(scene: DistanceFieldScene): DistanceFieldOracleResult {
  const {width, height, settings} = scene;
  const cellSizeX = settings[2];
  const cellSizeY = settings[3];
  const maxDistance = settings[6];
  const isotropic = cellSizeX === cellSizeY;
  const cellSeeds = getCellSeeds(scene);
  const seedCells: number[] = [];
  for (let cell = 0; cell < cellSeeds.length; cell++) {
    if (cellSeeds[cell] !== NONE) {
      seedCells.push(cell);
    }
  }
  const cellCount = width * height;
  const distances = new Float64Array(cellCount).fill(Infinity);
  const allocation = new Uint32Array(cellCount).fill(NONE);
  const nearestCells = new Uint32Array(cellCount).fill(NONE);
  for (let cell = 0; cell < cellCount; cell++) {
    const x = cell % width;
    const y = Math.floor(cell / width);
    let bestKey = Infinity;
    let bestId = NONE;
    let bestCell = NONE;
    for (const seedCell of seedCells) {
      const dx = Math.abs(x - (seedCell % width));
      const dy = Math.abs(y - Math.floor(seedCell / width));
      const key = isotropic
        ? dx * dx + dy * dy
        : cellSizeX * dx * (cellSizeX * dx) + cellSizeY * dy * (cellSizeY * dy);
      const id = cellSeeds[seedCell];
      if (key < bestKey || (key === bestKey && id < bestId)) {
        bestKey = key;
        bestId = id;
        bestCell = seedCell;
      }
    }
    if (bestCell === NONE) {
      continue;
    }
    const distance = isotropic ? cellSizeX * Math.sqrt(bestKey) : Math.sqrt(bestKey);
    if (Math.fround(distance) > maxDistance) {
      continue;
    }
    distances[cell] = distance;
    allocation[cell] = bestId;
    nearestCells[cell] = bestCell;
  }
  return {distances, allocation, nearestCells, cellSeeds};
}

/** Ground distance between two cell centers, f64. */
export function getCellDistance(
  width: number,
  settings: Float32Array,
  cell: number,
  seedCell: number
): number {
  const dx = Math.abs((cell % width) - (seedCell % width));
  const dy = Math.abs(Math.floor(cell / width) - Math.floor(seedCell / width));
  return Math.hypot(settings[2] * dx, settings[3] * dy);
}

/** Distance in f32 units in the last place between an f32 GPU value and an f64 reference. */
export function getUlpDistance(actual: number, expected: number): number {
  if (actual === expected) {
    return 0;
  }
  if (!Number.isFinite(actual) || !Number.isFinite(expected)) {
    return Infinity;
  }
  const bits = new Uint32Array(new Float32Array([actual, Math.fround(expected)]).buffer);
  const roundedDifference = Math.abs(bits[0] - bits[1]);
  return roundedDifference;
}

/** Deterministic xorshift random numbers in `[0, 1)`. */
export function createRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 0x100000000;
  };
}

/** Random seed points inside `[origin, origin + size)` plus a few rejected rows. */
export function createRandomSeedPositions(
  random: () => number,
  count: number,
  origin: readonly [number, number],
  size: readonly [number, number]
): Float32Array {
  const positions = new Float32Array(count * 2);
  for (let row = 0; row < count; row++) {
    positions[row * 2] = origin[0] + random() * size[0];
    positions[row * 2 + 1] = origin[1] + random() * size[1];
  }
  return positions;
}
