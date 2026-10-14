// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_COST_DISTANCE_NONE, GPU_COST_DISTANCE_UNRESOLVED} from '../../../src/gpu-raster';

const TILE_SIZE = 16;
const INNER_ITERATIONS = 2 * TILE_SIZE;
const COLUMN_OFFSETS = [1, 1, 0, -1, -1, -1, 0, 1];
const ROW_OFFSETS = [0, 1, 1, 1, 0, -1, -1, -1];

/** Inputs of {@link emulateTiledRelaxation}. */
export type TiledRelaxationEmulatorInput = {
  width: number;
  height: number;
  /** Initial values; updated in place. */
  values: Float64Array;
  /** Per-cell participation: false is a NaN auxiliary value (the cell is excluded). */
  participates: (cell: number) => boolean;
  /** Edge cost from the neighbor in `direction` (D8 index) of `cell` into `cell`. */
  getEdgeCost: (cell: number, direction: number) => number;
  /** Unrolled gated iterations. */
  maxIterations: number;
  /** Iteration at which directional tile sweeps start; undefined keeps plain tiled passes. */
  sweepAfterIteration?: number;
  /** Every tile starts active (the tie phase) or none (seed kernels mark tiles). */
  activateAllTiles: boolean;
};

/**
 * CPU emulation of the schedule of `createRasterTiledRelaxationNodes` (additive edge costs).
 *
 * It follows the WGSL: 16x16 tiles with a one-cell halo and up to 32 Jacobi inner iterations, tile
 * activity from the 3x3 stamps (`>= iteration + 1`, a changed tile stamps `iteration + 2`), the
 * plain tiled pass, the cycle tiled, right, left, down, up from `sweepAfterIteration` on, and a
 * gate that stops at the first unchanged iteration. Where the GPU order is unspecified it takes
 * the slowest legal order: tiles of one pass and the lines of one sweep read the values and stamps
 * of the start of the pass, and only a sweep line sees its own earlier tiles. So an emulated
 * convergence is a guarantee for the GPU schedule, not a lucky interleaving.
 */
export function emulateTiledRelaxation(input: TiledRelaxationEmulatorInput): {
  converged: boolean;
  iterationCount: number;
} {
  const {width, height, values} = input;
  const tilesX = Math.ceil(width / TILE_SIZE);
  const tilesY = Math.ceil(height / TILE_SIZE);
  const stamps = new Uint32Array(tilesX * tilesY).fill(input.activateAllTiles ? 1 : 0);
  const edgeCosts = new Float64Array(width * height * 8);
  for (let cell = 0; cell < width * height; cell++) {
    for (let direction = 0; direction < 8; direction++) {
      const neighbor = getNeighbor(width, height, cell, direction);
      edgeCosts[cell * 8 + direction] =
        neighbor >= 0 && input.participates(neighbor)
          ? input.getEdgeCost(cell, direction)
          : Infinity;
    }
  }

  // Relaxes one tile from `read` into `write`; returns true when a cell improved.
  const relaxTile = (
    tile: number,
    iteration: number,
    read: Float64Array,
    write: Float64Array,
    stampsRead: Uint32Array,
    stampsWrite: Uint32Array[]
  ): boolean => {
    const tileColumn = tile % tilesX;
    const tileRow = Math.floor(tile / tilesX);
    let active = false;
    for (let rowDelta = -1; rowDelta <= 1; rowDelta++) {
      for (let columnDelta = -1; columnDelta <= 1; columnDelta++) {
        const column = tileColumn + columnDelta;
        const row = tileRow + rowDelta;
        if (column < 0 || row < 0 || column >= tilesX || row >= tilesY) continue;
        if (stampsRead[row * tilesX + column] >= iteration + 1) active = true;
      }
    }
    if (!active) return false;
    const cells: number[] = [];
    for (let localRow = 0; localRow < TILE_SIZE; localRow++) {
      for (let localColumn = 0; localColumn < TILE_SIZE; localColumn++) {
        const column = tileColumn * TILE_SIZE + localColumn;
        const row = tileRow * TILE_SIZE + localRow;
        if (column < width && row < height && input.participates(row * width + column)) {
          cells.push(row * width + column);
        }
      }
    }
    // The halo is the tile's own copy: cells outside the tile keep their loaded values.
    const local = new Map<number, number>();
    const get = (cell: number) => local.get(cell) ?? read[cell];
    let changed = false;
    for (let inner = 0; inner < INNER_ITERATIONS; inner++) {
      const next: [number, number][] = [];
      for (const cell of cells) {
        let best = get(cell);
        for (let direction = 0; direction < 8; direction++) {
          const neighbor = getNeighbor(width, height, cell, direction);
          if (neighbor < 0) continue;
          const candidate = get(neighbor) + edgeCosts[cell * 8 + direction];
          if (candidate < best) best = candidate;
        }
        if (best < get(cell)) next.push([cell, best]);
      }
      if (next.length === 0) break;
      for (const [cell, value] of next) local.set(cell, value);
    }
    for (const [cell, value] of local) {
      if (value < read[cell]) {
        write[cell] = Math.min(write[cell], value);
        changed = true;
      }
    }
    if (changed) {
      for (const target of stampsWrite) target[tile] = Math.max(target[tile], iteration + 2);
    }
    return changed;
  };

  for (let iteration = 0; iteration < input.maxIterations; iteration++) {
    const cycle =
      input.sweepAfterIteration === undefined || iteration < input.sweepAfterIteration
        ? 0
        : (iteration - input.sweepAfterIteration) % 5;
    const snapshot = Float64Array.from(values);
    const stampSnapshot = Uint32Array.from(stamps);
    let changed = false;
    if (cycle === 0) {
      for (let tile = 0; tile < tilesX * tilesY; tile++) {
        changed = relaxTile(tile, iteration, snapshot, values, stampSnapshot, [stamps]) || changed;
      }
    } else {
      const vertical = cycle >= 3;
      const reverse = cycle === 2 || cycle === 4;
      const lineCount = vertical ? tilesX : tilesY;
      const stepCount = vertical ? tilesY : tilesX;
      for (let line = 0; line < lineCount; line++) {
        const lineValues = Float64Array.from(snapshot);
        const lineStamps = Uint32Array.from(stampSnapshot);
        for (let step = 0; step < stepCount; step++) {
          const position = reverse ? stepCount - 1 - step : step;
          const tile = vertical ? position * tilesX + line : line * tilesX + position;
          // The line sees its own earlier tiles; its improvements merge into the global values.
          if (
            relaxTile(tile, iteration, lineValues, lineValues, lineStamps, [lineStamps, stamps])
          ) {
            changed = true;
          }
        }
        for (let cell = 0; cell < values.length; cell++) {
          values[cell] = Math.min(values[cell], lineValues[cell]);
        }
      }
    }
    if (!changed) return {converged: true, iterationCount: iteration + 1};
  }
  return {converged: false, iterationCount: input.maxIterations};
}

/** Returns the D8 neighbor of `cell` in `direction`, or -1 outside the grid. */
export function getNeighbor(
  width: number,
  height: number,
  cell: number,
  direction: number
): number {
  const column = (cell % width) + COLUMN_OFFSETS[direction];
  const row = Math.floor(cell / width) + ROW_OFFSETS[direction];
  return column < 0 || row < 0 || column >= width || row >= height ? -1 : row * width + column;
}

/** Inputs of the back-link and path twins. */
export type BackLinkTwinInput = {
  width: number;
  height: number;
  costs: ArrayLike<number>;
  passable: (cell: number) => boolean;
  /** True when a strictly cheaper tight neighbor exists, with its D8 direction. */
  strictDirection: (cell: number) => number;
  /** Tie mask: bit d set when the neighbor in direction d is an equal-cost zero-edge neighbor. */
  masks: ArrayLike<number>;
  levels: ArrayLike<number>;
  costConverged: boolean;
};

/**
 * Twin of the back-link WGSL in `GPUCostDistance`. `rule: 'previous'` is the code before review
 * finding C-02 (an unreached tie level kept link 0, "reached as a source").
 */
export function computeBackLinksTwin(
  input: BackLinkTwinInput,
  rule: 'current' | 'previous' = 'current'
): Uint32Array {
  const {width, height, costs, levels} = input;
  const links = new Uint32Array(width * height);
  for (let cell = 0; cell < width * height; cell++) {
    let link = GPU_COST_DISTANCE_NONE;
    if (rule === 'current' && input.passable(cell) && !input.costConverged) {
      link = GPU_COST_DISTANCE_UNRESOLVED;
    } else if (input.passable(cell) && Number.isFinite(costs[cell])) {
      const strict = input.strictDirection(cell);
      if (strict >= 0) {
        link = 1 << strict;
      } else {
        const level = levels[cell];
        const lowerLink = () => {
          for (let direction = 0; direction < 8; direction++) {
            if (((input.masks[cell] >> direction) & 1) === 0) continue;
            const neighbor = getNeighbor(width, height, cell, direction);
            if (neighbor >= 0 && levels[neighbor] === level - 1) return 1 << direction;
          }
          return undefined;
        };
        if (rule === 'previous') {
          link = 0;
          if (Number.isFinite(level) && level > 0.5) link = lowerLink() ?? 0;
        } else if (Number.isFinite(level) && level < 0.5) {
          link = 0;
        } else {
          link = (Number.isFinite(level) ? lowerLink() : undefined) ?? GPU_COST_DISTANCE_UNRESOLVED;
        }
      }
    }
    links[cell] = link;
  }
  return links;
}

/** Twin of the `GPUCostDistancePath` walk: the path (target first) and the unresolved flag. */
export function walkBackLinksTwin(
  width: number,
  height: number,
  backLinks: ArrayLike<number>,
  target: number
): {path: number[]; iterationLimitReached: boolean} {
  const cellCount = width * height;
  const path: number[] = [];
  if (target >= cellCount || backLinks[target] === GPU_COST_DISTANCE_NONE) {
    return {path, iterationLimitReached: false};
  }
  let cell = target;
  for (let walk = 0; walk < cellCount; walk++) {
    const code = backLinks[cell];
    if (code === GPU_COST_DISTANCE_UNRESOLVED) return {path: [], iterationLimitReached: true};
    path.push(cell);
    if (code === 0) return {path, iterationLimitReached: false};
    const direction = Math.log2(code);
    if (!Number.isInteger(direction) || direction > 7) break;
    const next = getNeighbor(width, height, cell, direction);
    if (next < 0 || backLinks[next] === GPU_COST_DISTANCE_NONE) break;
    cell = next;
  }
  return {path: [], iterationLimitReached: false};
}

/**
 * Twin of the previous `GPUCostDistancePath` walk, which stopped at the first code 0 and reported
 * whatever it had walked.
 */
export function walkBackLinksPreviousTwin(
  width: number,
  height: number,
  backLinks: ArrayLike<number>,
  target: number
): number[] {
  const path: number[] = [];
  if (target >= width * height || backLinks[target] === GPU_COST_DISTANCE_NONE) return path;
  let cell = target;
  for (let walk = 0; walk < width * height; walk++) {
    path.push(cell);
    const code = backLinks[cell];
    if (code === 0 || code > 128) break;
    const next = getNeighbor(width, height, cell, Math.log2(code));
    if (next < 0 || backLinks[next] === GPU_COST_DISTANCE_NONE) break;
    cell = next;
  }
  return path;
}
