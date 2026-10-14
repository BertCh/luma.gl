// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * CPU models (test oracles only) of the two keep-flag passes of `GPUGeometryCleanup`: the
 * sequential per-ring pass and the tolerance-0 vertex-parallel passes (flag, resolve, apply).
 * They port the WGSL line by line, with `Math.fround` for f32 arithmetic.
 */

/** Flat ring layout: `positions` holds `x, y` pairs, `ringOffsets` the vertex offset of each ring. */
export type CleanupModelInput = {
  positions: Float32Array;
  ringOffsets: number[];
  isPolygon: boolean;
  gridSize: number;
  tolerance: number;
  removeRepeatedPoints: boolean;
};

/** Keep flags per vertex and the number of collapsed polygon rings. */
export type CleanupModelResult = {keepFlags: number[]; collapsedCount: number};

const f32 = Math.fround;

/** Mirrors `CLEANUP_REPEAT_CHUNK` of the flag pass. */
const CLEANUP_MODEL_CHUNK = 32;

function snapValue(value: number, gridSize: number): number {
  return gridSize > 0 ? f32(f32(Math.floor(f32(f32(value / gridSize) + 0.5))) * gridSize) : value;
}

function readSnappedPoint(input: CleanupModelInput, row: number): [number, number] {
  return [
    snapValue(input.positions[2 * row], input.gridSize),
    snapValue(input.positions[2 * row + 1], input.gridSize)
  ];
}

/** Mirror of WGSL `isWithinTolerance(a - b, tolerance)`. */
function isWithinTolerance(a: [number, number], b: [number, number], tolerance: number): boolean {
  const dx = f32(a[0] - b[0]);
  const dy = f32(a[1] - b[1]);
  if (tolerance > 0) {
    return f32(f32(dx * dx) + f32(dy * dy)) <= f32(tolerance * tolerance);
  }
  return dx === 0 && dy === 0;
}

function getRingRange(input: CleanupModelInput, ring: number): [number, number] {
  const vertexCount = input.positions.length / 2;
  return [input.ringOffsets[ring], Math.min(input.ringOffsets[ring + 1], vertexCount)];
}

/** Port of `createCleanupFlagNode`: one sequential walk per ring. */
export function runSequentialCleanupModel(input: CleanupModelInput): CleanupModelResult {
  const keepFlags = new Array<number>(input.positions.length / 2).fill(0);
  let collapsedCount = 0;
  const tolerance = Math.max(input.tolerance, 0);
  for (let ring = 0; ring < input.ringOffsets.length - 1; ring++) {
    const [start, end] = getRingRange(input, ring);
    if (end <= start) continue;
    if (!input.removeRepeatedPoints || end - start < 2) {
      for (let row = start; row < end; row++) keepFlags[row] = 1;
      continue;
    }
    let lastKept = start;
    let keptCount = 1;
    keepFlags[start] = 1;
    let lastKeptPoint = readSnappedPoint(input, start);
    for (let row = start + 1; row + 1 < end; row++) {
      const point = readSnappedPoint(input, row);
      if (isWithinTolerance(point, lastKeptPoint, tolerance)) {
        keepFlags[row] = 0;
      } else {
        keepFlags[row] = 1;
        lastKept = row;
        lastKeptPoint = point;
        keptCount++;
      }
    }
    const finalRow = end - 1;
    keepFlags[finalRow] = 1;
    keptCount++;
    if (
      lastKept !== start &&
      isWithinTolerance(readSnappedPoint(input, finalRow), lastKeptPoint, tolerance)
    ) {
      keepFlags[lastKept] = 0;
      keptCount--;
    }
    if (input.isPolygon && keptCount < 3) {
      for (let row = start; row < end; row++) keepFlags[row] = 0;
      collapsedCount++;
    }
  }
  return {keepFlags, collapsedCount};
}

/**
 * Port of the tolerance-0 passes `createCleanupRepeatFlagNode`, `createCleanupRepeatResolveNode`
 * and `createCleanupRepeatApplyNode`. Each pass runs over all invocations before the next, as the
 * GPU dispatches do; the per-ring words model the atomic add and max. Ignores `input.tolerance`.
 */
export function runParallelCleanupModel(input: CleanupModelInput): CleanupModelResult {
  const vertexCount = input.positions.length / 2;
  const ringCount = input.ringOffsets.length - 1;
  const keepFlags = new Array<number>(vertexCount).fill(0);
  const ringState = Array.from({length: ringCount}, () => [0, 0, 0]);
  let collapsedCount = 0;
  const findRing = (row: number) => {
    let low = 0;
    let high = ringCount;
    while (high - low > 1) {
      const middle = Math.floor((low + high) / 2);
      if (input.ringOffsets[middle] <= row) low = middle;
      else high = middle;
    }
    return low;
  };
  const locate = (row: number): {ring: number; start: number; end: number} | undefined => {
    if (row < input.ringOffsets[0] || row >= Math.min(input.ringOffsets[ringCount], vertexCount)) {
      return undefined;
    }
    const ring = findRing(row);
    const [start, end] = getRingRange(input, ring);
    return row < start || row >= end ? undefined : {ring, start, end};
  };
  // Pass 1: chunked flags; one count/max update per ring segment of a chunk.
  const chunkCount = Math.ceil(vertexCount / CLEANUP_MODEL_CHUNK);
  for (let chunk = 0; chunk < chunkCount; chunk++) {
    const chunkEnd = Math.min(
      (chunk + 1) * CLEANUP_MODEL_CHUNK,
      Math.min(input.ringOffsets[ringCount], vertexCount)
    );
    let row = Math.max(chunk * CLEANUP_MODEL_CHUNK, input.ringOffsets[0]);
    while (row < chunkEnd) {
      const ring = findRing(row);
      const [start, end] = getRingRange(input, ring);
      if (row < start || row >= end) {
        row++;
        continue;
      }
      const segmentEnd = Math.min(end, chunkEnd);
      const plain = !input.removeRepeatedPoints || end - start < 2;
      let count = 0;
      let last = 0;
      for (; row < segmentEnd; row++) {
        if (plain || row === start || row + 1 === end) {
          keepFlags[row] = 1;
        } else if (
          isWithinTolerance(readSnappedPoint(input, row), readSnappedPoint(input, row - 1), 0)
        ) {
          keepFlags[row] = 0;
        } else {
          keepFlags[row] = 1;
          count++;
          last = row - start;
        }
      }
      if (count > 0) {
        ringState[ring][0] += count;
        ringState[ring][1] = Math.max(ringState[ring][1], last);
      }
    }
  }
  // Pass 2: one invocation per ring.
  const COLLAPSED = 0xffffffff;
  for (let ring = 0; ring < ringCount; ring++) {
    const [start, end] = getRingRange(input, ring);
    if (end <= start || end - start < 2 || !input.removeRepeatedPoints) continue;
    let keptCount = 2 + ringState[ring][0];
    const lastKept = ringState[ring][1];
    let replaced = 0;
    if (
      lastKept !== 0 &&
      isWithinTolerance(readSnappedPoint(input, end - 1), readSnappedPoint(input, end - 2), 0)
    ) {
      replaced = lastKept;
      keptCount--;
    }
    if (input.isPolygon && keptCount < 3) {
      ringState[ring][2] = COLLAPSED;
      collapsedCount++;
    } else {
      ringState[ring][2] = replaced;
    }
  }
  // Pass 3: vertex-parallel apply.
  for (let row = 0; row < vertexCount; row++) {
    const found = locate(row);
    if (!found) continue;
    const resolution = ringState[found.ring][2];
    if (resolution === COLLAPSED || (resolution !== 0 && row - found.start === resolution)) {
      keepFlags[row] = 0;
    }
  }
  return {keepFlags, collapsedCount};
}

/** Builds a model input from rings given as `[x, y]` lists (any trailing positions are outside every ring). */
export function createCleanupModelInput(
  rings: number[][][],
  options: Partial<Omit<CleanupModelInput, 'positions' | 'ringOffsets'>> = {},
  trailingVertices: number[][] = []
): CleanupModelInput {
  const values: number[] = [];
  const ringOffsets = [0];
  for (const ring of rings) {
    for (const [x, y] of ring) values.push(x, y);
    ringOffsets.push(values.length / 2);
  }
  for (const [x, y] of trailingVertices) values.push(x, y);
  return {
    positions: Float32Array.from(values),
    ringOffsets,
    isPolygon: false,
    gridSize: 0,
    tolerance: 0,
    removeRepeatedPoints: true,
    ...options
  };
}
