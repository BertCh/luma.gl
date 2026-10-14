// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * CPU model of the `searchStructure: 'bvh'` path of `GPUNeighborSearch`: the same complete binary
 * tree (children `2n + 1` and `2n + 2`, empty boxes for invalid and padding leaves), the same
 * best-first traversal with the same pruning slack and `(d^2, id)` top-k list as the WGSL in
 * `neighbor-search-kernels.ts`, in f32 arithmetic. It checks the algorithm against the brute-force
 * oracle without a GPU, and counts the work done so specs can show clustered data stays sublinear.
 */

import type {NeighborSearchOracleInput} from './neighbor-search-oracle';

const f32 = Math.fround;
const BOUND_SLACK = f32(1.000001);

/** Table of the model BVH. */
export type NeighborSearchBVHModel = {
  targetRows: number;
  leafCapacity: number;
  internalNodeCount: number;
  levelCount: number;
  /** Per node: `[minimumX, minimumY, maximumX, maximumY]`; empty when minimum exceeds maximum. */
  boxes: Float32Array;
  /** Per leaf slot: the target row, or `0xffffffff` for padding. */
  leafRows: Uint32Array;
};

/** Work counters of one model query. */
export type NeighborSearchBVHWork = {nodeBoundTests: number; leafTests: number};

function getHilbertIndex(cellX: number, cellY: number): number {
  let x = cellX;
  let y = cellY;
  let key = 0;
  for (let s = 32768; s > 0; s >>= 1) {
    const rx = (x & s) !== 0 ? 1 : 0;
    const ry = (y & s) !== 0 ? 1 : 0;
    key += s * s * ((3 * rx) ^ ry);
    if (ry === 0) {
      if (rx === 1) {
        x = 65535 - x;
        y = 65535 - y;
      }
      [x, y] = [y, x];
    }
  }
  return key;
}

/**
 * Builds the model tree over the valid targets (`validTargets[row]`), leaves in Hilbert order of
 * the points, invalid targets and padding last, boxes refit bottom-up.
 */
export function buildNeighborSearchBVHModel(
  positions: Float32Array,
  validTargets: readonly boolean[]
): NeighborSearchBVHModel {
  const targetRows = positions.length / 2;
  let leafCapacity = 2;
  while (leafCapacity < targetRows) {
    leafCapacity *= 2;
  }
  const valid = Array.from({length: targetRows}, (_, row) => row).filter(row => validTargets[row]);
  const bounds = [Infinity, Infinity, -Infinity, -Infinity];
  for (const row of valid) {
    bounds[0] = Math.min(bounds[0], positions[row * 2]);
    bounds[1] = Math.min(bounds[1], positions[row * 2 + 1]);
    bounds[2] = Math.max(bounds[2], positions[row * 2]);
    bounds[3] = Math.max(bounds[3], positions[row * 2 + 1]);
  }
  const quantize = (value: number, minimum: number, maximum: number) =>
    maximum > minimum
      ? Math.min(Math.floor(((value - minimum) / (maximum - minimum)) * 65535 + 0.5), 65535)
      : 0;
  const keyed = valid.map(row => ({
    row,
    key: getHilbertIndex(
      quantize(positions[row * 2], bounds[0], bounds[2]),
      quantize(positions[row * 2 + 1], bounds[1], bounds[3])
    )
  }));
  keyed.sort((a, b) => a.key - b.key || a.row - b.row);
  const order = [
    ...keyed.map(entry => entry.row),
    ...Array.from({length: targetRows}, (_, row) => row).filter(row => !validTargets[row])
  ];
  const nodeCount = 2 * leafCapacity - 1;
  const internalNodeCount = leafCapacity - 1;
  const boxes = new Float32Array(nodeCount * 4);
  const leafRows = new Uint32Array(leafCapacity).fill(0xffffffff);
  const maximum = f32(3.402823466e38);
  for (let node = 0; node < nodeCount; node++) {
    boxes.set([maximum, maximum, -maximum, -maximum], node * 4);
  }
  for (let slot = 0; slot < targetRows; slot++) {
    const row = order[slot];
    leafRows[slot] = row;
    if (validTargets[row]) {
      const base = (internalNodeCount + slot) * 4;
      boxes.set(
        [positions[row * 2], positions[row * 2 + 1], positions[row * 2], positions[row * 2 + 1]],
        base
      );
    }
  }
  for (let node = internalNodeCount - 1; node >= 0; node--) {
    const first = (node * 2 + 1) * 4;
    const second = (node * 2 + 2) * 4;
    boxes[node * 4] = Math.min(boxes[first], boxes[second]);
    boxes[node * 4 + 1] = Math.min(boxes[first + 1], boxes[second + 1]);
    boxes[node * 4 + 2] = Math.max(boxes[first + 2], boxes[second + 2]);
    boxes[node * 4 + 3] = Math.max(boxes[first + 3], boxes[second + 3]);
  }
  return {
    targetRows,
    leafCapacity,
    internalNodeCount,
    levelCount: Math.log2(leafCapacity) + 1,
    boxes,
    leafRows
  };
}

function getBoxBound(model: NeighborSearchBVHModel, node: number, x: number, y: number): number {
  const base = node * 4;
  const {boxes} = model;
  if (boxes[base] > boxes[base + 2] || boxes[base + 1] > boxes[base + 3]) {
    return -1;
  }
  const gapX = Math.max(Math.max(f32(boxes[base] - x), f32(x - boxes[base + 2])), 0);
  const gapY = Math.max(Math.max(f32(boxes[base + 1] - y), f32(y - boxes[base + 3])), 0);
  return f32(f32(gapX * gapX) + f32(gapY * gapY));
}

/**
 * One query of the model. `k` set: the k nearest as `{id, distanceSquared}` by `(d^2, id)`; `k`
 * unset: every target within `radiusSquared`. `skipRow` is the query's own row in a self join.
 * Rows come back unsorted by ID, as from the kernel.
 */
export function searchNeighborSearchBVHModel(
  model: NeighborSearchBVHModel,
  query: {x: number; y: number; skipRow: number},
  options: {k?: number; radius: number},
  work: NeighborSearchBVHWork = {nodeBoundTests: 0, leafTests: 0}
): {id: number; distanceSquared: number}[] {
  const {x, y, skipRow} = query;
  const radiusSquared = f32(f32(options.radius) * f32(options.radius));
  const bounded =
    options.k === undefined || (Number.isFinite(options.radius) && options.radius > 0);
  const found: {id: number; distanceSquared: number}[] = [];
  const isBefore = (a: {id: number; distanceSquared: number}, b: typeof a) =>
    a.distanceSquared < b.distanceSquared ||
    (a.distanceSquared === b.distanceSquared && a.id < b.id);
  const stack: {node: number; bound: number}[] = [];
  work.nodeBoundTests++;
  const rootBound = getBoxBound(model, 0, x, y);
  if (rootBound >= 0) {
    stack.push({node: 0, bound: rootBound});
  }
  while (stack.length > 0) {
    const {node, bound} = stack.pop()!;
    let limited = bounded;
    let limit = f32(radiusSquared * BOUND_SLACK);
    if (options.k !== undefined && found.length === options.k) {
      const kth = f32(found[options.k - 1].distanceSquared * BOUND_SLACK);
      limit = limited ? Math.min(limit, kth) : kth;
      limited = true;
    }
    if (limited && bound > limit) {
      continue;
    }
    if (node < model.internalNodeCount) {
      const first = node * 2 + 1;
      const second = first + 1;
      work.nodeBoundTests += 2;
      const firstBound = getBoxBound(model, first, x, y);
      const secondBound = getBoxBound(model, second, x, y);
      const firstUsable = firstBound >= 0 && !(limited && firstBound > limit);
      const secondUsable = secondBound >= 0 && !(limited && secondBound > limit);
      if (options.k === undefined) {
        // Radius traversal: order is irrelevant.
        if (firstUsable) stack.push({node: first, bound: firstBound});
        if (secondUsable) stack.push({node: second, bound: secondBound});
      } else if (firstUsable && secondUsable) {
        if (firstBound <= secondBound) {
          stack.push({node: second, bound: secondBound}, {node: first, bound: firstBound});
        } else {
          stack.push({node: first, bound: firstBound}, {node: second, bound: secondBound});
        }
      } else if (firstUsable) {
        stack.push({node: first, bound: firstBound});
      } else if (secondUsable) {
        stack.push({node: second, bound: secondBound});
      }
      continue;
    }
    const slot = node - model.internalNodeCount;
    const id = model.leafRows[slot];
    if (id >= model.targetRows || id === skipRow) {
      continue;
    }
    work.leafTests++;
    const deltaX = f32(model.boxes[node * 4] - x);
    const deltaY = f32(model.boxes[node * 4 + 1] - y);
    const distanceSquared = f32(f32(deltaX * deltaX) + f32(deltaY * deltaY));
    if (options.k !== undefined && bounded && !(distanceSquared <= radiusSquared)) {
      continue;
    }
    if (options.k === undefined) {
      if (distanceSquared <= radiusSquared) {
        found.push({id, distanceSquared});
      }
      continue;
    }
    const candidate = {id, distanceSquared};
    if (found.length === options.k && !isBefore(candidate, found[options.k - 1])) {
      continue;
    }
    let position = found.length;
    while (position > 0 && !isBefore(found[position - 1], candidate)) {
      position--;
    }
    found.splice(position, 0, candidate);
    if (found.length > options.k) {
      found.pop();
    }
  }
  return found;
}

/** Self- or cross-join CSR from the model, in the oracle's result shape (rows ascending by ID). */
export function computeNeighborSearchBVHModel(
  input: NeighborSearchOracleInput,
  work: NeighborSearchBVHWork = {nodeBoundTests: 0, leafTests: 0}
): {offsets: number[]; neighbors: number[]; distancesSquared: number[]} {
  const {positions, parameters} = input;
  const crossJoin = Boolean(input.queryPositions);
  const queries = input.queryPositions ?? positions;
  const queryMask = input.queryMask ?? (crossJoin ? undefined : input.mask);
  const bounds = parameters.bounds.map(f32);
  const radius = f32(parameters.radius ?? Infinity);
  const inside = (x: number, y: number) =>
    Number.isFinite(x) &&
    Number.isFinite(y) &&
    x >= bounds[0] &&
    x <= bounds[2] &&
    y >= bounds[1] &&
    y <= bounds[3];
  // As the lattice: a radius mode without a finite positive radius has no valid point at all.
  const radiusSquared = f32(radius * radius);
  const latticeValid =
    bounds.every(Number.isFinite) &&
    bounds[2] >= bounds[0] &&
    bounds[3] >= bounds[1] &&
    (input.mode === 'knn' ||
      (Number.isFinite(radius) && Number.isFinite(radiusSquared) && radius > 0));
  const validTargets = Array.from(
    {length: positions.length / 2},
    (_, row) =>
      latticeValid &&
      (!input.mask || input.mask[row] !== 0) &&
      inside(positions[row * 2], positions[row * 2 + 1])
  );
  const model = buildNeighborSearchBVHModel(positions, validTargets);
  const result = {offsets: [0], neighbors: [] as number[], distancesSquared: [] as number[]};
  for (let row = 0; row < queries.length / 2; row++) {
    const x = queries[row * 2];
    const y = queries[row * 2 + 1];
    if (latticeValid && (!queryMask || queryMask[row] !== 0) && inside(x, y)) {
      const rowResult = searchNeighborSearchBVHModel(
        model,
        {x, y, skipRow: crossJoin ? -1 : row},
        {k: input.mode === 'knn' ? (input.k ?? 1) : undefined, radius},
        work
      ).sort((a, b) => a.id - b.id);
      for (const entry of rowResult) {
        result.neighbors.push(entry.id);
        result.distancesSquared.push(entry.distanceSquared);
      }
    }
    result.offsets.push(result.neighbors.length);
  }
  return result;
}

/**
 * Query-weighted cell occupancy `sum(count^2) / total` of the lattice the `'auto'` gate reduces:
 * `gridSize` cells over `bounds`, valid targets only.
 */
export function computeNeighborSearchOccupancy(
  positions: Float32Array,
  validTargets: readonly boolean[],
  bounds: readonly number[],
  gridSize: readonly [number, number]
): number {
  const counts = new Map<number, number>();
  let total = 0;
  const cellWidth = (bounds[2] - bounds[0]) / gridSize[0] || 1;
  const cellHeight = (bounds[3] - bounds[1]) / gridSize[1] || 1;
  for (let row = 0; row < positions.length / 2; row++) {
    if (!validTargets[row]) continue;
    const column = Math.min(
      Math.floor((positions[row * 2] - bounds[0]) / cellWidth),
      gridSize[0] - 1
    );
    const cellRow = Math.min(
      Math.floor((positions[row * 2 + 1] - bounds[1]) / cellHeight),
      gridSize[1] - 1
    );
    const cell = cellRow * gridSize[0] + column;
    counts.set(cell, (counts.get(cell) ?? 0) + 1);
    total++;
  }
  let sum = 0;
  for (const count of counts.values()) sum += count * count;
  return total > 0 ? sum / total : 0;
}
