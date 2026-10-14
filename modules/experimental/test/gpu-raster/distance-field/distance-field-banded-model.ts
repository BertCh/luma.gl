// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// CPU models of the exact distance-field passes: the sequential column scans and
// Felzenszwalb-Huttenlocher row envelopes that the kernels used before banding, and the banded
// (Parallel Banding Algorithm) passes the kernels use now, step for step with the same tie rules.

import {GPU_DISTANCE_FIELD_NONE} from '../../../src/gpu-raster/distance-field';

const NONE = GPU_DISTANCE_FIELD_NONE;

/** Nearest seed row and ID per cell from the column pass, row-major. */
export type DistanceFieldColumnCandidates = {columnRows: Uint32Array; columnIds: Uint32Array};

/** Nearest seed cell and ID per cell from the row pass, row-major. */
export type DistanceFieldNearest = {nearestCells: Uint32Array; nearestIds: Uint32Array};

/** Returns the WGSL `isCloser` order: exact integer keys on isotropic grids, else f32 keys. */
export function createDistanceFieldOrder(
  cellSizeX: number,
  cellSizeY: number
): (dxA: number, dyA: number, idA: number, dxB: number, dyB: number, idB: number) => boolean {
  const sizeX = Math.fround(cellSizeX);
  const sizeY = Math.fround(cellSizeY);
  const isotropic = sizeX === sizeY;
  const getKey = (dx: number, dy: number): number => {
    if (isotropic) {
      return dx * dx + dy * dy;
    }
    const groundX = Math.fround(sizeX * dx);
    const groundY = Math.fround(sizeY * dy);
    return Math.fround(Math.fround(groundX * groundX) + Math.fround(groundY * groundY));
  };
  return (dxA, dyA, idA, dxB, dyB, idB) => {
    const keyA = getKey(dxA, dyA);
    const keyB = getKey(dxB, dyB);
    return keyA !== keyB ? keyA < keyB : idA < idB;
  };
}

/** The pre-banding column pass: one forward and one backward scan per column. */
export function computeColumnsSequential(
  width: number,
  height: number,
  cellSeeds: Uint32Array
): DistanceFieldColumnCandidates {
  const columnRows = new Uint32Array(width * height);
  const columnIds = new Uint32Array(width * height);
  for (let column = 0; column < width; column++) {
    scanColumnBand(width, column, 0, height, [NONE, NONE], [NONE, NONE], cellSeeds, {
      columnRows,
      columnIds
    });
  }
  return {columnRows, columnIds};
}

/** The banded column pass: band summaries, per-column carries, then per-band scans. */
export function computeColumnsBanded(
  width: number,
  height: number,
  cellSeeds: Uint32Array,
  bandSize: number
): DistanceFieldColumnCandidates {
  const bandCount = Math.ceil(height / bandSize);
  const columnRows = new Uint32Array(width * height);
  const columnIds = new Uint32Array(width * height);
  for (let column = 0; column < width; column++) {
    const tops: [number, number][] = [];
    const bottoms: [number, number][] = [];
    for (let band = 0; band < bandCount; band++) {
      let top: [number, number] = [NONE, NONE];
      let bottom: [number, number] = [NONE, NONE];
      for (let row = band * bandSize; row < Math.min((band + 1) * bandSize, height); row++) {
        const seedId = cellSeeds[row * width + column];
        if (seedId !== NONE) {
          if (top[0] === NONE) {
            top = [row, seedId];
          }
          bottom = [row, seedId];
        }
      }
      tops.push(top);
      bottoms.push(bottom);
    }
    const above: [number, number][] = [];
    let last: [number, number] = [NONE, NONE];
    for (let band = 0; band < bandCount; band++) {
      above.push(last);
      if (bottoms[band][0] !== NONE) {
        last = bottoms[band];
      }
    }
    const below: [number, number][] = new Array(bandCount);
    last = [NONE, NONE];
    for (let band = bandCount - 1; band >= 0; band--) {
      below[band] = last;
      if (tops[band][0] !== NONE) {
        last = tops[band];
      }
    }
    for (let band = 0; band < bandCount; band++) {
      scanColumnBand(
        width,
        column,
        band * bandSize,
        Math.min((band + 1) * bandSize, height),
        above[band],
        below[band],
        cellSeeds,
        {columnRows, columnIds}
      );
    }
  }
  return {columnRows, columnIds};
}

function scanColumnBand(
  width: number,
  column: number,
  rowStart: number,
  rowEnd: number,
  above: [number, number],
  below: [number, number],
  cellSeeds: Uint32Array,
  target: DistanceFieldColumnCandidates
): void {
  let [lastRow, lastId] = above;
  for (let row = rowStart; row < rowEnd; row++) {
    const cell = row * width + column;
    if (cellSeeds[cell] !== NONE) {
      lastRow = row;
      lastId = cellSeeds[cell];
    }
    target.columnRows[cell] = lastRow;
    target.columnIds[cell] = lastId;
  }
  [lastRow, lastId] = below;
  for (let row = rowEnd - 1; row >= rowStart; row--) {
    const cell = row * width + column;
    if (cellSeeds[cell] !== NONE) {
      lastRow = row;
      lastId = cellSeeds[cell];
    }
    if (lastRow === NONE) {
      continue;
    }
    const forwardRow = target.columnRows[cell];
    let useBackward = forwardRow === NONE;
    if (!useBackward) {
      const forwardDistance = row - forwardRow;
      const backwardDistance = lastRow - row;
      useBackward =
        backwardDistance < forwardDistance ||
        (backwardDistance === forwardDistance && lastId < target.columnIds[cell]);
    }
    if (useBackward) {
      target.columnRows[cell] = lastRow;
      target.columnIds[cell] = lastId;
    }
  }
}

type RowContext = {
  width: number;
  row: number;
  candidates: DistanceFieldColumnCandidates;
  isCloser: ReturnType<typeof createDistanceFieldOrder>;
  starts: Uint32Array;
  next: Uint32Array;
  previous: Uint32Array;
};

/** `beatsAt` of the WGSL: q's candidate beats v's at cell x of the row. */
function beatsAt(context: RowContext, q: number, v: number, x: number): boolean {
  const {width, row, candidates} = context;
  const base = row * width;
  return context.isCloser(
    Math.abs(x - q),
    Math.abs(candidates.columnRows[base + q] - row),
    candidates.columnIds[base + q],
    Math.abs(x - v),
    Math.abs(candidates.columnRows[base + v] - row),
    candidates.columnIds[base + v]
  );
}

/** `insertEntry` of the WGSL: pops beaten tail entries, returns [new tail, start of q]. */
function insertEntry(context: RowContext, q: number, tailIn: number): [number, number] {
  let tail = tailIn;
  for (;;) {
    if (tail === NONE) {
      return [NONE, 0];
    }
    const vStart = context.starts[tail];
    if (beatsAt(context, q, tail, vStart)) {
      context.starts[tail] = NONE;
      tail = context.previous[tail];
      continue;
    }
    let low = vStart + 1;
    let high = context.width;
    while (low < high) {
      const middle = Math.floor((low + high) / 2);
      if (beatsAt(context, q, tail, middle)) {
        high = middle;
      } else {
        low = middle + 1;
      }
    }
    return [tail, low];
  }
}

/** The pre-banding row pass: one Felzenszwalb-Huttenlocher stack per row. */
export function computeRowsSequential(
  width: number,
  height: number,
  candidates: DistanceFieldColumnCandidates,
  isCloser: ReturnType<typeof createDistanceFieldOrder>
): DistanceFieldNearest {
  const nearestCells = new Uint32Array(width * height);
  const nearestIds = new Uint32Array(width * height);
  for (let row = 0; row < height; row++) {
    const base = row * width;
    const stackColumns: number[] = [];
    const stackStarts: number[] = [];
    for (let q = 0; q < width; q++) {
      if (candidates.columnRows[base + q] === NONE) {
        continue;
      }
      const context = {width, row, candidates, isCloser} as RowContext;
      let start = 0;
      for (;;) {
        if (stackColumns.length === 0) {
          start = 0;
          break;
        }
        const v = stackColumns[stackColumns.length - 1];
        const vStart = stackStarts[stackStarts.length - 1];
        if (beatsAt(context, q, v, vStart)) {
          stackColumns.pop();
          stackStarts.pop();
          continue;
        }
        let low = vStart + 1;
        let high = width;
        while (low < high) {
          const middle = Math.floor((low + high) / 2);
          if (beatsAt(context, q, v, middle)) {
            high = middle;
          } else {
            low = middle + 1;
          }
        }
        start = low;
        break;
      }
      if (start < width) {
        stackColumns.push(q);
        stackStarts.push(start);
      }
    }
    let entry = 0;
    for (let x = 0; x < width; x++) {
      if (stackColumns.length === 0) {
        nearestCells[base + x] = NONE;
        nearestIds[base + x] = NONE;
        continue;
      }
      while (entry + 1 < stackColumns.length && stackStarts[entry + 1] <= x) {
        entry++;
      }
      const column = stackColumns[entry];
      nearestCells[base + x] = candidates.columnRows[base + column] * width + column;
      nearestIds[base + x] = candidates.columnIds[base + column];
    }
  }
  return {nearestCells, nearestIds};
}

/** Statistics of one banded row pass, for specs that check the merges did work. */
export type DistanceFieldBandedRowStatistics = {mergeRounds: number; mergeInsertions: number};

/** The banded row pass: band envelopes, pairwise merges with the early stop, marker coloring. */
export function computeRowsBanded(
  width: number,
  height: number,
  candidates: DistanceFieldColumnCandidates,
  isCloser: ReturnType<typeof createDistanceFieldOrder>,
  bandSize: number,
  statistics: DistanceFieldBandedRowStatistics = {mergeRounds: 0, mergeInsertions: 0}
): DistanceFieldNearest {
  const bandCount = Math.ceil(width / bandSize);
  const mergeRoundCount = bandCount > 1 ? Math.ceil(Math.log2(bandCount)) : 0;
  statistics.mergeRounds = mergeRoundCount;
  const nearestCells = new Uint32Array(width * height);
  const nearestIds = new Uint32Array(width * height);
  for (let row = 0; row < height; row++) {
    const base = row * width;
    const context: RowContext = {
      width,
      row,
      candidates,
      isCloser,
      starts: new Uint32Array(width).fill(NONE),
      next: new Uint32Array(width).fill(NONE),
      previous: new Uint32Array(width).fill(NONE)
    };
    const heads = new Uint32Array(bandCount).fill(NONE);
    const tails = new Uint32Array(bandCount).fill(NONE);
    for (let band = 0; band < bandCount; band++) {
      let head = NONE;
      let tail = NONE;
      for (let q = band * bandSize; q < Math.min((band + 1) * bandSize, width); q++) {
        context.starts[q] = NONE;
        if (candidates.columnRows[base + q] === NONE) {
          continue;
        }
        const [newTail, start] = insertEntry(context, q, tail);
        tail = newTail;
        if (start < width) {
          context.starts[q] = start;
          context.previous[q] = tail;
          context.next[q] = NONE;
          if (tail === NONE) {
            head = q;
          } else {
            context.next[tail] = q;
          }
          tail = q;
        }
      }
      heads[band] = head;
      tails[band] = tail;
    }
    for (let round = 0; round < mergeRoundCount; round++) {
      const blockBands = 2 ** round;
      for (let leftBand = 0; leftBand + blockBands < bandCount; leftBand += 2 * blockBands) {
        const rightBand = leftBand + blockBands;
        const leftHead = heads[leftBand];
        const rightTail = tails[rightBand];
        let tail = tails[leftBand];
        let current = heads[rightBand];
        let mergedHead = NONE;
        let reachedLocalState = false;
        while (current !== NONE) {
          statistics.mergeInsertions++;
          const localPrevious = context.previous[current];
          const localStart = context.starts[current];
          const localNext = context.next[current];
          const [newTail, start] = insertEntry(context, current, tail);
          tail = newTail;
          if (start >= width) {
            context.starts[current] = NONE;
            current = localNext;
            continue;
          }
          context.starts[current] = start;
          context.previous[current] = tail;
          if (tail === NONE) {
            mergedHead = current;
          } else {
            context.next[tail] = current;
          }
          if (tail === localPrevious && start === localStart) {
            reachedLocalState = true;
            break;
          }
          tail = current;
          current = localNext;
        }
        if (leftHead !== NONE && context.starts[leftHead] !== NONE) {
          mergedHead = leftHead;
        }
        heads[leftBand] = mergedHead;
        tails[leftBand] = reachedLocalState ? rightTail : tail;
      }
    }
    const markers = new Uint32Array(width);
    for (let column = 0; column < width; column++) {
      if (context.starts[column] !== NONE) {
        markers[context.starts[column]] = column + 1;
      }
    }
    // Banded running maximum, as the marker-band, carry and color kernels compute it.
    const bandMaxima = new Uint32Array(bandCount);
    for (let band = 0; band < bandCount; band++) {
      for (let x = band * bandSize; x < Math.min((band + 1) * bandSize, width); x++) {
        bandMaxima[band] = Math.max(bandMaxima[band], markers[x]);
      }
    }
    let carry = 0;
    for (let band = 0; band < bandCount; band++) {
      let running = carry;
      carry = Math.max(carry, bandMaxima[band]);
      for (let x = band * bandSize; x < Math.min((band + 1) * bandSize, width); x++) {
        running = Math.max(running, markers[x]);
        if (running === 0) {
          nearestCells[base + x] = NONE;
          nearestIds[base + x] = NONE;
          continue;
        }
        const column = running - 1;
        nearestCells[base + x] = candidates.columnRows[base + column] * width + column;
        nearestIds[base + x] = candidates.columnIds[base + column];
      }
    }
  }
  return {nearestCells, nearestIds};
}
