// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUTerrainCellSizeMode} from '../../../src/map-graphs/terrain-analysis';
import {
  GPU_RASTER_D8_DIRECTIONS,
  getRasterD8Distance,
  type RasterGridSettings
} from '../../../src/map-graphs/cost-distance/raster-grid-utils';

export const NONE = 0xffffffff;

/** One oracle source: cell index and initial cost. */
export type CostDistanceOracleSource = {cell: number; cost: number};

/** Inputs of {@link computeCostDistance}. */
export type CostDistanceOracleOptions = {
  width: number;
  height: number;
  /** Friction after calibration; NaN, infinite, or negative cells are impassable. */
  friction: ArrayLike<number>;
  cellSizeMode?: GPUTerrainCellSizeMode;
  settings: RasterGridSettings;
  sources: readonly CostDistanceOracleSource[];
  costLimit?: number;
};

/** Returns whether a friction sample is passable. */
export function isPassable(value: number): boolean {
  return Number.isFinite(value) && value >= 0;
}

/** Returns the float64 cost of one D8 move from `cell` in `direction`. */
export function getEdgeCost(
  options: CostDistanceOracleOptions,
  cell: number,
  direction: number
): number {
  const {width, height} = options;
  const row = Math.floor(cell / width);
  const neighbor = getNeighbor(width, height, cell, direction);
  return (
    getRasterD8Distance(
      options.cellSizeMode ?? 'uniform',
      direction,
      row,
      height,
      options.settings
    ) *
    (0.5 * (options.friction[cell] + options.friction[neighbor]))
  );
}

/** Returns the neighbor cell in a D8 direction index, or -1 outside the grid. */
export function getNeighbor(
  width: number,
  height: number,
  cell: number,
  direction: number
): number {
  const {columnOffset, rowOffset} = GPU_RASTER_D8_DIRECTIONS[direction];
  const column = (cell % width) + columnOffset;
  const row = Math.floor(cell / width) + rowOffset;
  return column < 0 || row < 0 || column >= width || row >= height ? -1 : row * width + column;
}

/** Grid Dijkstra in float64 with the recipe's impassable, limit, and seed rules. */
export function computeCostDistance(options: CostDistanceOracleOptions): Float64Array {
  const {width, height, friction} = options;
  const costLimit = options.costLimit ?? Infinity;
  const cellCount = width * height;
  const costs = new Float64Array(cellCount).fill(Infinity);
  const heap: [number, number][] = [];
  const push = (cost: number, cell: number) => {
    heap.push([cost, cell]);
    let index = heap.length - 1;
    while (index > 0) {
      const parent = (index - 1) >> 1;
      if (heap[parent][0] <= heap[index][0]) break;
      [heap[parent], heap[index]] = [heap[index], heap[parent]];
      index = parent;
    }
  };
  const pop = (): [number, number] => {
    const top = heap[0];
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      let index = 0;
      for (;;) {
        const left = 2 * index + 1;
        const right = left + 1;
        let smallest = index;
        if (left < heap.length && heap[left][0] < heap[smallest][0]) smallest = left;
        if (right < heap.length && heap[right][0] < heap[smallest][0]) smallest = right;
        if (smallest === index) break;
        [heap[smallest], heap[index]] = [heap[index], heap[smallest]];
        index = smallest;
      }
    }
    return top;
  };
  for (const {cell, cost} of options.sources) {
    if (
      cell >= 0 &&
      cell < cellCount &&
      isPassable(friction[cell]) &&
      Number.isFinite(cost) &&
      cost >= 0 &&
      cost <= costLimit &&
      cost < costs[cell]
    ) {
      costs[cell] = cost;
      push(cost, cell);
    }
  }
  while (heap.length > 0) {
    const [cost, cell] = pop();
    if (cost > costs[cell]) continue;
    for (let direction = 0; direction < 8; direction++) {
      const neighbor = getNeighbor(width, height, cell, direction);
      if (neighbor < 0 || !isPassable(friction[neighbor])) continue;
      const candidate = cost + getEdgeCost(options, cell, direction);
      if (candidate <= costLimit && candidate < costs[neighbor]) {
        costs[neighbor] = candidate;
        push(candidate, neighbor);
      }
    }
  }
  return costs;
}

/** Number of thresholds below each cost, or `NONE` (matches the recipe's band rule). */
export function computeBands(costs: ArrayLike<number>, thresholds: readonly number[]): Uint32Array {
  return Uint32Array.from(costs as ArrayLike<number>, cost => {
    if (!Number.isFinite(cost)) return NONE;
    const count = thresholds.filter(threshold => Math.fround(threshold) < cost).length;
    return count < thresholds.length ? count : NONE;
  });
}

/** Counts cells per band, ignoring `NONE`. */
export function computeBandCounts(bands: ArrayLike<number>, bandCount: number): number[] {
  const counts = new Array<number>(bandCount).fill(0);
  for (let index = 0; index < bands.length; index++) {
    if (bands[index] !== NONE) counts[bands[index]]++;
  }
  return counts;
}

/**
 * Checks back-link validity against GPU costs and returns the first problem, or `undefined`.
 *
 * Every reached cell must hold a code that moves to a neighbor of strictly lower cost with
 * `cost_n + edge ~= cost_c`; unreached cells must hold `NONE`; following links must end at a code-0
 * cell within `cellCount` steps. Exact codes are not checked because equal-cost ties may differ.
 *
 * @param seedCosts Optional cell to initial cost map; when given, terminal cells must be seeds.
 */
export function checkBackLinks(
  options: CostDistanceOracleOptions,
  costs: ArrayLike<number>,
  backLinks: ArrayLike<number>,
  seedCosts?: ReadonlyMap<number, number>
): string | undefined {
  const {width, height} = options;
  const cellCount = width * height;
  for (let cell = 0; cell < cellCount; cell++) {
    const reached = Number.isFinite(costs[cell]);
    if (!reached) {
      if (backLinks[cell] !== NONE)
        return `cell ${cell} is unreached but has back-link ${backLinks[cell]}`;
      continue;
    }
    const code = backLinks[cell];
    if (code === 0) continue;
    const direction = GPU_RASTER_D8_DIRECTIONS.findIndex(entry => entry.code === code);
    if (direction < 0) return `cell ${cell} has invalid back-link ${code}`;
    const neighbor = getNeighbor(width, height, cell, direction);
    if (neighbor < 0) return `cell ${cell} links outside the grid`;
    if (!(costs[neighbor] < costs[cell])) return `cell ${cell} links to a non-cheaper cell`;
    const edge = getEdgeCost(options, cell, direction);
    const expected = costs[neighbor] + edge;
    if (Math.abs(expected - costs[cell]) > 1e-4 * Math.max(1, costs[cell])) {
      return `cell ${cell} link cost ${expected} differs from ${costs[cell]}`;
    }
  }
  for (let start = 0; start < cellCount; start++) {
    if (!Number.isFinite(costs[start])) continue;
    let cell = start;
    let steps = 0;
    while (backLinks[cell] !== 0) {
      const direction = GPU_RASTER_D8_DIRECTIONS.findIndex(entry => entry.code === backLinks[cell]);
      cell = getNeighbor(width, height, cell, direction);
      if (++steps > cellCount) return `cell ${start} back-links do not terminate`;
    }
    if (seedCosts) {
      const seed = seedCosts.get(cell);
      if (seed === undefined || Math.abs(seed - costs[cell]) > 1e-4 * Math.max(1, seed)) {
        return `cell ${start} terminates at non-source cell ${cell}`;
      }
    }
  }
  return undefined;
}

/** Tie-level oracle result of {@link computeTieBackLinks}. */
export type TieBackLinkOracle = {
  /** True when a cell has a strictly cheaper tight in-neighbor (that cell links strictly). */
  hasStrict: Uint8Array;
  /** BFS level over zero-cost equal-cost edges from strict entries and root sources; -1 if unreachable. */
  level: Int32Array;
  /** Expected D8 direction for cells that link across a tie (smallest direction one level lower), else -1. */
  tieDirection: Int32Array;
};

/**
 * Computes the cycle-safe tie rule on settled costs: a cell links strictly when a strictly cheaper
 * neighbor is tight; otherwise it links to the smallest-direction equal-cost zero-edge neighbor exactly
 * one level lower, where level 0 is a strict entry or a root source (seed cost equals the final cost).
 * Tie edges are exact zero-cost edges, so the oracle does not depend on float rounding.
 */
export function computeTieBackLinks(
  options: CostDistanceOracleOptions,
  costs: ArrayLike<number>,
  seedCosts: ReadonlyMap<number, number>
): TieBackLinkOracle {
  const {width, height} = options;
  const cellCount = width * height;
  const hasStrict = new Uint8Array(cellCount);
  const level = new Int32Array(cellCount).fill(-1);
  const isTie = (cell: number, direction: number): boolean => {
    const neighbor = getNeighbor(width, height, cell, direction);
    return (
      neighbor >= 0 &&
      Number.isFinite(costs[cell]) &&
      costs[neighbor] === costs[cell] &&
      isPassable(options.friction[neighbor]) &&
      getEdgeCost(options, cell, direction) === 0
    );
  };
  let frontier: number[] = [];
  for (let cell = 0; cell < cellCount; cell++) {
    if (!Number.isFinite(costs[cell])) continue;
    for (let direction = 0; direction < 8; direction++) {
      const neighbor = getNeighbor(width, height, cell, direction);
      if (neighbor < 0 || !isPassable(options.friction[neighbor])) continue;
      if (!(costs[neighbor] < costs[cell])) continue;
      const expected = costs[neighbor] + getEdgeCost(options, cell, direction);
      if (Math.abs(expected - costs[cell]) <= 1e-4 * Math.max(1, costs[cell])) {
        hasStrict[cell] = 1;
      }
    }
    const seed = seedCosts.get(cell);
    if (
      hasStrict[cell] ||
      (seed !== undefined && Math.abs(seed - costs[cell]) <= 1e-4 * Math.max(1, seed))
    ) {
      level[cell] = 0;
      frontier.push(cell);
    }
  }
  for (let depth = 1; frontier.length > 0; depth++) {
    const next: number[] = [];
    for (const cell of frontier) {
      for (let direction = 0; direction < 8; direction++) {
        const neighbor = getNeighbor(width, height, cell, direction);
        // The edge neighbor -> cell is the tie edge; zero edges are symmetric.
        if (neighbor >= 0 && level[neighbor] < 0 && isTie(cell, direction)) {
          level[neighbor] = depth;
          next.push(neighbor);
        }
      }
    }
    frontier = next;
  }
  const tieDirection = new Int32Array(cellCount).fill(-1);
  for (let cell = 0; cell < cellCount; cell++) {
    if (hasStrict[cell] || level[cell] <= 0) continue;
    for (let direction = 0; direction < 8; direction++) {
      const neighbor = getNeighbor(width, height, cell, direction);
      if (neighbor >= 0 && isTie(cell, direction) && level[neighbor] === level[cell] - 1) {
        tieDirection[cell] = direction;
        break;
      }
    }
  }
  return {hasStrict, level, tieDirection};
}

/**
 * Checks back-links on graphs that may contain zero-cost plateaus and returns the first problem.
 *
 * Every reached cell must link to a neighbor of no greater cost across a tight edge (equal costs only
 * across zero-cost edges); walking links must reach a seed whose cost equals the cell's final cost
 * within `cellCount` steps without revisiting a cell; unreached cells must be `NONE`. Also compares
 * every tie link exactly with {@link computeTieBackLinks}.
 */
export function checkBackLinkWalks(
  options: CostDistanceOracleOptions,
  costs: ArrayLike<number>,
  backLinks: ArrayLike<number>,
  seedCosts: ReadonlyMap<number, number>
): string | undefined {
  const {width, height} = options;
  const cellCount = width * height;
  const tolerance = (value: number) => 1e-4 * Math.max(1, value);
  const oracle = computeTieBackLinks(options, costs, seedCosts);
  for (let cell = 0; cell < cellCount; cell++) {
    if (!Number.isFinite(costs[cell])) {
      if (backLinks[cell] !== NONE) return `cell ${cell} is unreached but has back-link`;
      continue;
    }
    const code = backLinks[cell];
    if (code === NONE) return `cell ${cell} is reached but has no back-link`;
    if (code === 0) {
      const seed = seedCosts.get(cell);
      if (seed === undefined || Math.abs(seed - costs[cell]) > tolerance(seed)) {
        return `cell ${cell} has code 0 but is not a source at its final cost`;
      }
      continue;
    }
    const direction = GPU_RASTER_D8_DIRECTIONS.findIndex(entry => entry.code === code);
    if (direction < 0) return `cell ${cell} has invalid back-link ${code}`;
    const neighbor = getNeighbor(width, height, cell, direction);
    if (neighbor < 0) return `cell ${cell} links outside the grid`;
    if (costs[neighbor] > costs[cell]) return `cell ${cell} links to a costlier cell`;
    const edge = getEdgeCost(options, cell, direction);
    if (costs[neighbor] === costs[cell] && edge !== 0) {
      return `cell ${cell} links across an equal-cost edge that is not zero-cost`;
    }
    if (Math.abs(costs[neighbor] + edge - costs[cell]) > tolerance(costs[cell])) {
      return `cell ${cell} link is not tight`;
    }
    if (
      !oracle.hasStrict[cell] &&
      oracle.level[cell] > 0 &&
      direction !== oracle.tieDirection[cell]
    ) {
      return `cell ${cell} tie link ${direction} differs from oracle ${oracle.tieDirection[cell]}`;
    }
  }
  for (let start = 0; start < cellCount; start++) {
    if (!Number.isFinite(costs[start])) continue;
    const visited = new Set<number>();
    let cell = start;
    while (backLinks[cell] !== 0) {
      if (visited.has(cell)) return `cell ${start} back-links cycle at ${cell}`;
      visited.add(cell);
      const direction = GPU_RASTER_D8_DIRECTIONS.findIndex(entry => entry.code === backLinks[cell]);
      cell = getNeighbor(width, height, cell, direction);
    }
    const seed = seedCosts.get(cell);
    if (seed === undefined || Math.abs(seed - costs[cell]) > tolerance(seed)) {
      return `cell ${start} terminates at non-source cell ${cell}`;
    }
  }
  return undefined;
}

/** Deterministic PRNG in `[0, 1)`. */
export function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
}

/** Uniform friction. */
export function createUniformFriction(width: number, height: number, value: number): Float32Array {
  return new Float32Array(width * height).fill(value);
}

/** Random integer friction in `[1, 9]`. */
export function createRandomFriction(width: number, height: number, seed: number): Float32Array {
  const random = createRandom(seed);
  return Float32Array.from({length: width * height}, () => 1 + Math.floor(random() * 9));
}

/** Friction 1 with a vertical barrier wall (`NaN`) at `wallColumn` except for rows in the gap. */
export function createWallFriction(
  width: number,
  height: number,
  wallColumn: number,
  gapRows: readonly number[]
): Float32Array {
  const friction = createUniformFriction(width, height, 1);
  for (let row = 0; row < height; row++) {
    if (!gapRows.includes(row)) friction[row * width + wallColumn] = NaN;
  }
  return friction;
}

/** Random integer friction with random holes set to `noDataValue`. */
export function createHoleFriction(
  width: number,
  height: number,
  seed: number,
  noDataValue: number,
  holeFraction: number
): Float32Array {
  const random = createRandom(seed + 1);
  const friction = createRandomFriction(width, height, seed);
  for (let index = 0; index < friction.length; index++) {
    if (random() < holeFraction) friction[index] = noDataValue;
  }
  return friction;
}
