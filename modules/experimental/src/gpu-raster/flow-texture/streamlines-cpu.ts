// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {sampleFieldOnCPU, type FieldRaster} from './flow-texture-field';
import {getPhilox4x32, getPhiloxUnitFloat} from './flow-texture-random';

/** Philox key word 1 of the seed jitter and priority stream. */
export const STREAMLINES_SEED_PURPOSE = 2;

/** Compile-time shape shared by `GPUStreamlines` and its CPU oracle. */
export type StreamlinesCPUConfig = {
  /** Seed lattice columns. */
  seedColumns: number;
  /** Seed lattice rows. */
  seedRows: number;
  /** Integration steps per direction `L`. */
  stepsPerDirection: number;
  /** Occupancy grid width in cells. */
  gridWidth: number;
  /** Occupancy grid height in cells. */
  gridHeight: number;
  /** Published line capacity. */
  lineCapacity: number;
  /** Published point capacity. */
  pointCapacity: number;
};

/** Traced candidate streamlines before pruning, laid out like the GPU transients. */
export type StreamlineCandidates = {
  /**
   * `seedCount * (2L + 1)` points: slot `s * (2L + 1) + L + k` is step `k` (negative backward).
   * The seed point (`k = 0`) is NaN for a seed outside the field or in a too-slow cell.
   */
  points: Float32Array;
  /** `seedCount * 2` traced step counts `[backward, forward]`. */
  spans: Uint32Array;
};

/** Result of {@link generateStreamlinesOnCPU}. */
export type StreamlinesCPUResult = {
  /** Candidates that were pruned. */
  candidates: StreamlineCandidates;
  /** Priority key per seed. */
  keys: Uint32Array;
  /** Accepted seed indices in ascending order (all of them, before capacity). */
  acceptedSeeds: number[];
  /** `seedCount * 2` trimmed step counts `[backward, forward]` of accepted seeds. */
  trimmed: Uint32Array;
  /** Published seed IDs. */
  ids: number[];
  /** Published CSR offsets, `ids.length + 1` entries. */
  pathOffsets: number[];
  /** Published points, `pathOffsets[ids.length] * 2` floats. */
  points: Float32Array;
  /** Number of accepted lines before capacity. */
  totalCount: number;
  /** 1 when lines were dropped for capacity. */
  overflow: number;
};

const fround = Math.fround;

/** Priority key of seed `index`: 16 random high bits, then `0xffff - index` so ties favour low IDs. */
export function getStreamlineKey(index: number, seed: number): number {
  const word = getPhilox4x32([index, 0, 0, 0], [seed, STREAMLINES_SEED_PURPOSE])[2];
  return (((word >>> 16) << 16) | (0xffff - index)) >>> 0;
}

/** Grid cell of a point, or -1 outside the grid. Mirrors the WGSL `getGridCell`. */
export function getStreamlineGridCell(
  x: number,
  y: number,
  parameters: ArrayLike<number>,
  config: Pick<StreamlinesCPUConfig, 'gridWidth' | 'gridHeight'>
): number {
  const localX = fround(fround(x - parameters[4]) * fround(parameters[8]));
  const localY = fround(fround(y - parameters[5]) * fround(parameters[9]));
  if (!(localX >= 0 && localY >= 0 && localX < config.gridWidth && localY < config.gridHeight)) {
    return -1;
  }
  return Math.floor(localY) * config.gridWidth + Math.floor(localX);
}

/** Traces every seed of the jittered lattice forward and backward (CPU mirror of the GPU trace). */
export function traceStreamlineCandidatesOnCPU(
  field: FieldRaster,
  config: StreamlinesCPUConfig,
  parameters: ArrayLike<number>,
  words: ArrayLike<number>
): StreamlineCandidates {
  const seedCount = config.seedColumns * config.seedRows;
  const length = config.stepsPerDirection;
  const stride = 2 * length + 1;
  const points = new Float32Array(seedCount * stride * 2).fill(NaN);
  const spans = new Uint32Array(seedCount * 2);
  const fieldExtent = [parameters[0], parameters[1], parameters[2], parameters[3]].map(fround);
  const gridOriginX = fround(parameters[4]);
  const gridOriginY = fround(parameters[5]);
  const seedCellX = fround(fround(fround(parameters[6]) * config.gridWidth) / config.seedColumns);
  const seedCellY = fround(fround(fround(parameters[7]) * config.gridHeight) / config.seedRows);
  const stepLength = fround(parameters[10]);
  const minimumSpeed = fround(parameters[11]);
  const seed = words[0] >>> 0;
  const getDirection = (x: number, y: number, sign: number): [number, number] | undefined => {
    const sample = sampleFieldOnCPU(field, x, y, fieldExtent);
    if (!sample) {
      return undefined;
    }
    const speed = Math.hypot(sample[0], sample[1]);
    if (!(speed > 0) || speed < minimumSpeed) {
      return undefined;
    }
    return [(sign * sample[0]) / speed, (sign * sample[1]) / speed];
  };
  for (let index = 0; index < seedCount; index++) {
    const random = getPhilox4x32([index, 0, 0, 0], [seed, STREAMLINES_SEED_PURPOSE]);
    const column = index % config.seedColumns;
    const row = Math.floor(index / config.seedColumns);
    const startX = fround(
      gridOriginX + fround(fround(column + getPhiloxUnitFloat(random[0])) * seedCellX)
    );
    const startY = fround(
      gridOriginY + fround(fround(row + getPhiloxUnitFloat(random[1])) * seedCellY)
    );
    const base = index * stride + length;
    if (
      !getDirection(startX, startY, 1) ||
      getStreamlineGridCell(startX, startY, parameters, config) < 0
    ) {
      continue;
    }
    points[2 * base] = startX;
    points[2 * base + 1] = startY;
    for (const sign of [-1, 1]) {
      let x = startX;
      let y = startY;
      let steps = 0;
      for (let step = 1; step <= length; step++) {
        const first = getDirection(x, y, sign);
        if (!first) {
          break;
        }
        const middle = getDirection(
          fround(x + 0.5 * stepLength * first[0]),
          fround(y + 0.5 * stepLength * first[1]),
          sign
        );
        if (!middle) {
          break;
        }
        const nextX = fround(x + stepLength * middle[0]);
        const nextY = fround(y + stepLength * middle[1]);
        if (getStreamlineGridCell(nextX, nextY, parameters, config) < 0) {
          break;
        }
        x = nextX;
        y = nextY;
        steps = step;
        points[2 * (base + sign * step)] = x;
        points[2 * (base + sign * step) + 1] = y;
      }
      spans[2 * index + (sign < 0 ? 0 : 1)] = steps;
    }
  }
  return {points, spans};
}

/**
 * CPU oracle of `GPUStreamlines`: traces (or reuses given) candidates, then prunes them greedily in
 * descending priority key. A line is rejected when its seed cell is occupied by an accepted line;
 * otherwise each direction is cut before its first point in an occupied cell, the line is accepted
 * when it keeps at least `minimumPoints` points, and its cells become occupied.
 *
 * @param field Packed vector field.
 * @param config Compile-time shape.
 * @param parameters Float parameters from `getGPUStreamlinesParameterValues`.
 * @param words Word parameters from `getGPUStreamlinesWordParameterValues`.
 * @param candidates Optional candidates to prune instead of tracing, for example read back from
 * the GPU so pruning can be compared exactly.
 */
export function generateStreamlinesOnCPU(
  field: FieldRaster,
  config: StreamlinesCPUConfig,
  parameters: ArrayLike<number>,
  words: ArrayLike<number>,
  candidates: StreamlineCandidates = traceStreamlineCandidatesOnCPU(
    field,
    config,
    parameters,
    words
  )
): StreamlinesCPUResult {
  const seedCount = config.seedColumns * config.seedRows;
  const length = config.stepsPerDirection;
  const stride = 2 * length + 1;
  const seed = words[0] >>> 0;
  const minimumPoints = words[1] >>> 0;
  const keys = new Uint32Array(seedCount);
  for (let index = 0; index < seedCount; index++) {
    keys[index] = getStreamlineKey(index, seed);
  }
  const order = Array.from({length: seedCount}, (_, index) => index).sort(
    (left, right) => keys[right] - keys[left]
  );
  const occupied = new Uint8Array(config.gridWidth * config.gridHeight);
  const trimmed = new Uint32Array(seedCount * 2);
  const accepted = new Uint8Array(seedCount);
  const getCell = (index: number, step: number) => {
    const slot = index * stride + length + step;
    return getStreamlineGridCell(
      candidates.points[2 * slot],
      candidates.points[2 * slot + 1],
      parameters,
      config
    );
  };
  for (const index of order) {
    if (Number.isNaN(candidates.points[2 * (index * stride + length)])) {
      continue;
    }
    if (occupied[getCell(index, 0)]) {
      continue;
    }
    const kept = [0, 0];
    for (const [direction, sign] of [
      [0, -1],
      [1, 1]
    ]) {
      const steps = candidates.spans[2 * index + direction];
      while (kept[direction] < steps && !occupied[getCell(index, sign * (kept[direction] + 1))]) {
        kept[direction]++;
      }
    }
    if (1 + kept[0] + kept[1] < minimumPoints) {
      continue;
    }
    accepted[index] = 1;
    trimmed[2 * index] = kept[0];
    trimmed[2 * index + 1] = kept[1];
    for (let step = -kept[0]; step <= kept[1]; step++) {
      occupied[getCell(index, step)] = 1;
    }
  }
  const acceptedSeeds: number[] = [];
  for (let index = 0; index < seedCount; index++) {
    if (accepted[index]) {
      acceptedSeeds.push(index);
    }
  }
  const ids: number[] = [];
  const pathOffsets = [0];
  const points: number[] = [];
  for (const index of acceptedSeeds) {
    const count = 1 + trimmed[2 * index] + trimmed[2 * index + 1];
    if (ids.length >= config.lineCapacity || points.length / 2 + count > config.pointCapacity) {
      break;
    }
    ids.push(index);
    for (let step = -trimmed[2 * index]; step <= trimmed[2 * index + 1]; step++) {
      const slot = index * stride + length + step;
      points.push(candidates.points[2 * slot], candidates.points[2 * slot + 1]);
    }
    pathOffsets.push(points.length / 2);
  }
  return {
    candidates,
    keys,
    acceptedSeeds,
    trimmed,
    ids,
    pathOffsets,
    points: new Float32Array(points),
    totalCount: acceptedSeeds.length,
    overflow: ids.length < acceptedSeeds.length ? 1 : 0
  };
}
