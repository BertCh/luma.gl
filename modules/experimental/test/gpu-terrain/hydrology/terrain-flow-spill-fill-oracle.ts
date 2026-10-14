// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createSeededRandom} from '../terrain-oracle-utils';

/**
 * Literal Priority-Flood (Barnes, Lehman and Mulla 2014, Algorithm 1, epsilon 0) with the same
 * boundary rule as GPUTerrainFlow: a valid cell on the grid edge or next to an invalid cell is an
 * outlet and keeps its elevation. 8-connected. NaN elevation marks an invalid cell, which stays NaN.
 */
export function fillDepressionsPriorityFlood(
  elevation: Float32Array,
  width: number,
  height: number
): Float32Array {
  const cellCount = width * height;
  const filled = new Float32Array(cellCount).fill(NaN);
  const closed = new Uint8Array(cellCount);
  const columnOffsets = [1, 1, 0, -1, -1, -1, 0, 1];
  const rowOffsets = [0, 1, 1, 1, 0, -1, -1, -1];
  const getNeighbors = (cell: number) => {
    const neighbors: number[] = [];
    for (let direction = 0; direction < 8; direction++) {
      const column = (cell % width) + columnOffsets[direction];
      const row = Math.floor(cell / width) + rowOffsets[direction];
      neighbors.push(
        column < 0 || row < 0 || column >= width || row >= height ? -1 : row * width + column
      );
    }
    return neighbors;
  };
  // Binary min-heap of [value, cell].
  const heap: [number, number][] = [];
  const push = (entry: [number, number]) => {
    heap.push(entry);
    let child = heap.length - 1;
    while (child > 0) {
      const parent = (child - 1) >> 1;
      if (heap[parent][0] <= heap[child][0]) break;
      [heap[parent], heap[child]] = [heap[child], heap[parent]];
      child = parent;
    }
  };
  const pop = (): [number, number] => {
    const top = heap[0];
    const last = heap.pop() as [number, number];
    if (heap.length > 0) {
      heap[0] = last;
      let parent = 0;
      for (;;) {
        const left = 2 * parent + 1;
        const right = left + 1;
        let smallest = parent;
        if (left < heap.length && heap[left][0] < heap[smallest][0]) smallest = left;
        if (right < heap.length && heap[right][0] < heap[smallest][0]) smallest = right;
        if (smallest === parent) break;
        [heap[parent], heap[smallest]] = [heap[smallest], heap[parent]];
        parent = smallest;
      }
    }
    return top;
  };
  for (let cell = 0; cell < cellCount; cell++) {
    if (!Number.isFinite(elevation[cell])) continue;
    const isOutlet = getNeighbors(cell).some(
      neighbor => neighbor < 0 || !Number.isFinite(elevation[neighbor])
    );
    if (isOutlet) {
      closed[cell] = 1;
      filled[cell] = elevation[cell];
      push([elevation[cell], cell]);
    }
  }
  while (heap.length > 0) {
    const [level, cell] = pop();
    for (const neighbor of getNeighbors(cell)) {
      if (neighbor < 0 || closed[neighbor] || !Number.isFinite(elevation[neighbor])) continue;
      closed[neighbor] = 1;
      filled[neighbor] = Math.max(elevation[neighbor], level);
      push([filled[neighbor], neighbor]);
    }
  }
  return filled;
}

/** One adversarial DEM of the spill fill specs. */
export type SpillFillCase = {name: string; width: number; height: number; elevation: Float32Array};

const make = (
  name: string,
  width: number,
  height: number,
  getElevation: (column: number, row: number, cell: number) => number
): SpillFillCase => ({
  name,
  width,
  height,
  elevation: Float32Array.from({length: width * height}, (_, cell) =>
    getElevation(cell % width, Math.floor(cell / width), cell)
  )
});

/**
 * Concentric rings of corridors of width `corridorWidth` separated by one-cell walls, each wall with
 * one gap, the gaps alternating between two sides, so the corridor path winds inward like a spiral.
 * Corridor floors descend by one per step of breadth-first distance from the outermost ring (which
 * touches the grid edge), walls are high, so every ring is a depression that spills at its gap.
 */
export function makeSpiralCase(size: number, corridorWidth: number): SpillFillCase {
  const period = corridorWidth + 1;
  const isWall = (column: number, row: number) => {
    const depth = Math.min(column, row, size - 1 - column, size - 1 - row);
    if (depth % period !== period - 1) return false;
    const ring = Math.floor(depth / period);
    // One gap per wall ring, on the top edge of the ring for even rings and the bottom for odd.
    const gapColumn = ring % 2 === 0 ? size - 1 - depth - 1 : depth + 1;
    const onGapSide = ring % 2 === 0 ? row === depth : row === size - 1 - depth;
    return !(onGapSide && column === gapColumn);
  };
  const distance = new Int32Array(size * size).fill(-1);
  const queue: number[] = [];
  for (let cell = 0; cell < size * size; cell++) {
    const column = cell % size;
    const row = Math.floor(cell / size);
    if (Math.min(column, row, size - 1 - column, size - 1 - row) === 0 && !isWall(column, row)) {
      distance[cell] = 0;
      queue.push(cell);
    }
  }
  for (let head = 0; head < queue.length; head++) {
    const cell = queue[head];
    for (const [dx, dy] of [
      [1, 0],
      [-1, 0],
      [0, 1],
      [0, -1]
    ]) {
      const column = (cell % size) + dx;
      const row = Math.floor(cell / size) + dy;
      if (column < 0 || row < 0 || column >= size || row >= size) continue;
      const neighbor = row * size + column;
      if (distance[neighbor] >= 0 || isWall(column, row)) continue;
      distance[neighbor] = distance[cell] + 1;
      queue.push(neighbor);
    }
  }
  return make(`spiral-${size}-w${corridorWidth}`, size, size, (column, row, cell) =>
    isWall(column, row) || distance[cell] < 0
      ? 100000 + ((column * 7 + row * 13) % 5)
      : 50000 - distance[cell]
  );
}

/** Named adversarial DEMs: plateaus, ties, nested bowls, lakes over many tiles, nodata and 1 x N. */
export function makeSpillFillCases(): SpillFillCase[] {
  const cases: SpillFillCase[] = [];
  const random = createSeededRandom(31);
  cases.push(makeSpiralCase(23, 1), makeSpiralCase(37, 3), makeSpiralCase(61, 1));
  // A bowl 150 x 137 deep over many 32-cell tiles, spilling through one notch in the rim.
  cases.push(
    make('lake-many-tiles', 150, 137, (column, row) => {
      const radius = Math.hypot(column - 75, row - 68);
      if (column === 149 && row === 40) return 7;
      return Math.min(60, Math.floor(radius)) + 10;
    })
  );
  // Nested bowls: a deep pit inside a shallow basin inside a ridge.
  cases.push(
    make('nested-bowls', 90, 70, (column, row) => {
      const radius = Math.hypot(column - 45, row - 35);
      const ridge = radius > 30 ? 10 : 60 - radius;
      return radius < 8 ? 1 + radius : ridge + (column === 5 && row === 5 ? -50 : 0);
    })
  );
  cases.push(make('plateau-ties', 97, 65, () => Math.floor(random() * 3)));
  cases.push(make('flat-zero', 64, 64, () => 0));
  cases.push(make('noise-small', 41, 33, () => Math.floor(random() * 20)));
  cases.push(make('noise-large', 130, 70, () => Math.floor(random() * 500) / 4));
  cases.push(make('negative-values', 34, 34, (column, row) => ((column * row) % 11) - 5.5));
  for (const [width, height] of [
    [1, 1],
    [1, 37],
    [41, 1],
    [2, 50],
    [3, 3],
    [33, 33],
    [31, 65]
  ] as const) {
    cases.push(make(`strip-${width}x${height}`, width, height, () => Math.floor(random() * 9)));
  }
  // Valid islands enclosed by nodata, a nodata hole inside a lake, and a nodata border column.
  cases.push(
    make('nodata-islands', 75, 60, (column, row) => {
      const ringDistance = Math.max(Math.abs(column - 20), Math.abs(row - 20));
      if (ringDistance === 7) return NaN;
      if (ringDistance < 7) return 3 + ((column + row) % 4);
      if (Math.hypot(column - 55, row - 35) < 3) return NaN;
      if (column === 0) return NaN;
      const radius = Math.hypot(column - 52, row - 35);
      return radius < 15 ? 2 + (radius > 12 ? 20 : 0) : 25 + ((column * 3 + row) % 7);
    })
  );
  cases.push(make('all-nodata', 12, 9, () => NaN));
  cases.push(make('single-valid', 12, 9, (_, __, cell) => (cell === 50 ? 3 : NaN)));
  cases.push(
    make('nodata-checker', 40, 40, (column, row) =>
      (column + row) % 5 === 0 ? NaN : 20 - Math.min(column, row, 39 - column, 39 - row)
    )
  );
  // A corridor of width 1 that winds across tile borders (a serpentine road in a basin).
  cases.push(
    make('serpentine-wall', 70, 70, (column, row) => {
      if (row % 4 === 2) return column === (Math.floor(row / 4) % 2 === 0 ? 66 : 3) ? 1 : 90;
      return 5 + ((column + row) % 3);
    })
  );
  return cases;
}
