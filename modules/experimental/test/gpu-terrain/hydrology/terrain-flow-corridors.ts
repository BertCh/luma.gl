// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Test DEMs and a fast reference for depression filling on very long basins.
 *
 * The Gauss-Seidel oracle in `terrain-flow-oracle.ts` needs minutes on a serpentine corridor, so
 * these tests compare against a priority-flood fill (Barnes, Lehman and Mulla 2014) instead.
 */

/**
 * A square DEM with a 3-cell-wide serpentine corridor walled by 100 m ridges. The floor descends
 * toward the corridor's far end ('pit') or toward its entrance ('drain'). Filling it requires a
 * front to travel the whole corridor (about `n * n / 2` cells).
 */
export function createSerpentineDEM(size: number, mode: 'pit' | 'drain'): Float32Array {
  const elevation = new Float32Array(size * size).fill(100);
  for (let band = 0; band * 4 + 3 < size - 1; band++) {
    for (let rowInBand = 1; rowInBand <= 3; rowInBand++) {
      const row = band * 4 + rowInBand;
      for (let column = 1; column <= size - 2; column++) {
        const along = band % 2 === 0 ? column : size - 1 - column;
        const pathIndex = band * size + along;
        elevation[row * size + column] =
          mode === 'pit' ? 50 - pathIndex * 1e-4 : 10 + pathIndex * 1e-4;
      }
    }
    // Gap in the wall that leads to the next band.
    const wallRow = band * 4 + 4;
    if (wallRow < size - 1) {
      const gapColumn = band % 2 === 0 ? size - 2 : 1;
      elevation[wallRow * size + gapColumn] = elevation[(wallRow - 1) * size + gapColumn];
    }
  }
  // Entrance on the west boundary.
  elevation[size] = elevation[size + 1];
  return elevation;
}

/** Priority-flood depression fill of a square DEM with an 8-neighborhood. Boundary cells keep their elevation. */
export function fillDepressionsPriorityFlood(elevation: Float32Array, size: number): Float32Array {
  const filled = new Float32Array(size * size).fill(NaN);
  const heapKeys: number[] = [];
  const heapCells: number[] = [];
  const push = (key: number, cell: number) => {
    let index = heapKeys.length;
    heapKeys.push(key);
    heapCells.push(cell);
    while (index > 0) {
      const parent = (index - 1) >> 1;
      if (heapKeys[parent] <= key) {
        break;
      }
      heapKeys[index] = heapKeys[parent];
      heapCells[index] = heapCells[parent];
      index = parent;
    }
    heapKeys[index] = key;
    heapCells[index] = cell;
  };
  const pop = (): [number, number] => {
    const rootKey = heapKeys[0];
    const rootCell = heapCells[0];
    const lastKey = heapKeys.pop()!;
    const lastCell = heapCells.pop()!;
    const length = heapKeys.length;
    if (length > 0) {
      let index = 0;
      for (;;) {
        let child = 2 * index + 1;
        if (child >= length) {
          break;
        }
        if (child + 1 < length && heapKeys[child + 1] < heapKeys[child]) {
          child++;
        }
        if (heapKeys[child] >= lastKey) {
          break;
        }
        heapKeys[index] = heapKeys[child];
        heapCells[index] = heapCells[child];
        index = child;
      }
      heapKeys[index] = lastKey;
      heapCells[index] = lastCell;
    }
    return [rootKey, rootCell];
  };
  const seen = new Uint8Array(size * size);
  for (let row = 0; row < size; row++) {
    for (let column = 0; column < size; column++) {
      if (row === 0 || column === 0 || row === size - 1 || column === size - 1) {
        const cell = row * size + column;
        seen[cell] = 1;
        filled[cell] = elevation[cell];
        push(elevation[cell], cell);
      }
    }
  }
  while (heapKeys.length > 0) {
    const [level, cell] = pop();
    const cellColumn = cell % size;
    const cellRow = (cell / size) | 0;
    for (let rowDelta = -1; rowDelta <= 1; rowDelta++) {
      for (let columnDelta = -1; columnDelta <= 1; columnDelta++) {
        const column = cellColumn + columnDelta;
        const row = cellRow + rowDelta;
        if (column < 0 || row < 0 || column >= size || row >= size) {
          continue;
        }
        const neighbor = row * size + column;
        if (seen[neighbor]) {
          continue;
        }
        seen[neighbor] = 1;
        filled[neighbor] = Math.max(elevation[neighbor], level);
        push(filled[neighbor], neighbor);
      }
    }
  }
  return filled;
}
