// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getSpillRoundCount,
  getSpillBitLength
} from '../../../src/gpu-terrain/hydrology/terrain-flow-spill-utils';
import {getTerrainFlowSpillBoruvkaRoundCount} from '../../../src/gpu-terrain/hydrology/terrain-flow-spill-fill';
import {createSeededRandom} from '../terrain-oracle-utils';

/*
 * CPU model of the spill graph depression fill (terrain-flow-spill-fill.ts and
 * terrain-flow-spill-tree.ts). It repeats the GPU algorithm stage by stage with the same keys,
 * edge slots, tie-breaks, hook rules, gate groups and pointer-jumping rounds, so node specs can
 * check the algorithm against literal oracles while the GPU specs wait. Sequential loops stand in
 * for atomics; the GPU result must not depend on their order, which `order` lets specs vary.
 */

export const SPILL_NONE = 0xffffffff;
export const SPILL_BOUNDARY_FLAG = 0x80000000;
export const SPILL_ID_MASK = 0x7fffffff;
export const SPILL_INFINITE_KEY = 0xffffffff;

const scratchFloat = new Float32Array(1);
const scratchBits = new Uint32Array(scratchFloat.buffer);

/** Order-preserving u32 key of a float32 (the WGSL `getSpillKey`). */
export function getSpillKey(value: number): number {
  scratchFloat[0] = value;
  const bits = scratchBits[0];
  return bits >>> 31 === 1 ? ~bits >>> 0 : (bits | 0x80000000) >>> 0;
}

/** Float32 of a key (the WGSL `getSpillValueBits`, as a number). */
export function getSpillValue(key: number): number {
  scratchBits[0] = key >>> 31 === 1 ? key & 0x7fffffff : ~key >>> 0;
  return scratchFloat[0];
}

const COLUMN_OFFSETS = [1, 1, 0, -1, -1, -1, 0, 1];
const ROW_OFFSETS = [0, 1, 1, 1, 0, -1, -1, -1];

function getNeighbor(width: number, height: number, cell: number, direction: number): number {
  const column = (cell % width) + COLUMN_OFFSETS[direction];
  const row = Math.floor(cell / width) + ROW_OFFSETS[direction];
  return column < 0 || row < 0 || column >= width || row >= height
    ? SPILL_NONE
    : row * width + column;
}

const isFinite32 = (value: number) => Number.isFinite(value);

/** Result of {@link modelTileFill}. */
export type SpillTileResult = {
  /** Tile-local fill L, NaN for invalid cells. */
  localFill: Float32Array;
  /** Label per cell (outlet cell index, plus the boundary flag on boundary outlets), or NONE. */
  labels: Uint32Array;
  /** Cell category: 0 none, 1 interior, 2 perimeter outlet, 3 boundary outlet. */
  category: Uint8Array;
};

/** Stage 1: tile-local fill and labels. */
export function modelTileFill(
  elevation: Float32Array,
  width: number,
  height: number,
  tileSize: number
): SpillTileResult {
  const cellCount = width * height;
  const localFill = new Float32Array(cellCount).fill(NaN);
  const labels = new Uint32Array(cellCount).fill(SPILL_NONE);
  const category = new Uint8Array(cellCount);
  const tilesX = Math.ceil(width / tileSize);
  const tilesY = Math.ceil(height / tileSize);
  const tileCells = tileSize * tileSize;
  for (let tile = 0; tile < tilesX * tilesY; tile++) {
    const tileColumn = tile % tilesX;
    const tileRow = Math.floor(tile / tilesX);
    const keys = new Uint32Array(tileCells).fill(SPILL_INFINITE_KEY);
    const tileLabels = new Uint32Array(tileCells).fill(SPILL_NONE);
    const categories = new Uint8Array(tileCells);
    const getCell = (local: number) => {
      const column = tileColumn * tileSize + (local % tileSize);
      const row = tileRow * tileSize + Math.floor(local / tileSize);
      return column < width && row < height ? row * width + column : -1;
    };
    for (let local = 0; local < tileCells; local++) {
      const cell = getCell(local);
      if (cell < 0 || !isFinite32(elevation[cell])) continue;
      let boundary = false;
      for (let direction = 0; direction < 8; direction++) {
        const neighbor = getNeighbor(width, height, cell, direction);
        if (neighbor === SPILL_NONE || !isFinite32(elevation[neighbor])) {
          boundary = true;
          break;
        }
      }
      const localColumn = local % tileSize;
      const localRow = Math.floor(local / tileSize);
      const perimeter =
        localColumn === 0 ||
        localRow === 0 ||
        localColumn === tileSize - 1 ||
        localRow === tileSize - 1;
      if (boundary) {
        categories[local] = 3;
        keys[local] = getSpillKey(elevation[cell]);
        tileLabels[local] = (cell | SPILL_BOUNDARY_FLAG) >>> 0;
      } else if (perimeter) {
        categories[local] = 2;
        keys[local] = getSpillKey(elevation[cell]);
        tileLabels[local] = cell;
      } else {
        categories[local] = 1;
      }
    }
    const neighborLocal = (local: number, direction: number) =>
      local + ROW_OFFSETS[direction] * tileSize + COLUMN_OFFSETS[direction];
    // Fill fixpoint (in-place, chaotic: the fixpoint is unique).
    for (let pass = 0; pass <= tileCells; pass++) {
      let changed = false;
      for (let local = 0; local < tileCells; local++) {
        if (categories[local] !== 1) continue;
        const ownKey = getSpillKey(elevation[getCell(local)]);
        let lowest = SPILL_INFINITE_KEY;
        for (let direction = 0; direction < 8; direction++) {
          lowest = Math.min(lowest, keys[neighborLocal(local, direction)]);
        }
        const candidate = Math.max(ownKey, lowest);
        if (candidate < keys[local]) {
          keys[local] = candidate;
          changed = true;
        }
      }
      if (!changed) break;
    }
    // Label fixpoint.
    for (let pass = 0; pass <= tileCells; pass++) {
      let changed = false;
      for (let local = 0; local < tileCells; local++) {
        if (categories[local] !== 1) continue;
        const ownKey = keys[local];
        let best = SPILL_NONE;
        for (let direction = 0; direction < 8; direction++) {
          const neighbor = neighborLocal(local, direction);
          if (tileLabels[neighbor] !== SPILL_NONE && keys[neighbor] <= ownKey) {
            best = Math.min(best, tileLabels[neighbor] & SPILL_ID_MASK);
          }
        }
        if (best < tileLabels[local]) {
          tileLabels[local] = best;
          changed = true;
        }
      }
      if (!changed) break;
    }
    for (let local = 0; local < tileCells; local++) {
      const cell = getCell(local);
      if (cell < 0) continue;
      category[cell] = categories[local];
      if (categories[local] === 0) continue;
      localFill[cell] = getSpillValue(keys[local]);
      labels[cell] = tileLabels[local];
    }
  }
  return {localFill, labels, category};
}

/** One implicit spill edge. */
export type SpillEdge = {valid: boolean; first: number; second: number; key: number};

/** Stage 2: the implicit edge of `slot` (the WGSL `getSpillEdge`). */
export function getSpillEdge(
  slot: number,
  width: number,
  height: number,
  tile: Pick<SpillTileResult, 'labels' | 'localFill'>
): SpillEdge {
  const cellCount = width * height;
  const edge = {valid: false, first: 0, second: 0, key: 0};
  if (slot < 4 * cellCount) {
    const cell = slot >> 2;
    const neighbor = getNeighbor(width, height, cell, slot & 3);
    if (neighbor === SPILL_NONE) return edge;
    const firstWord = tile.labels[cell];
    const secondWord = tile.labels[neighbor];
    if (firstWord === SPILL_NONE || secondWord === SPILL_NONE) return edge;
    edge.first = firstWord & SPILL_ID_MASK;
    edge.second = secondWord & SPILL_ID_MASK;
    if (edge.first === edge.second) return edge;
    edge.key = Math.max(getSpillKey(tile.localFill[cell]), getSpillKey(tile.localFill[neighbor]));
    edge.valid = true;
    return edge;
  }
  const cell = slot - 4 * cellCount;
  const word = tile.labels[cell];
  if (word === SPILL_NONE || (word & SPILL_BOUNDARY_FLAG) >>> 0 === 0) return edge;
  edge.first = cell;
  edge.second = cellCount;
  edge.key = getSpillKey(tile.localFill[cell]);
  edge.valid = true;
  return edge;
}

/** Gate group bookkeeping shared by the gated loops (groups of four rounds). */
function runGatedGroups(
  maxIterations: number,
  groupSize: number,
  round: (iteration: number) => boolean
): {iterationCount: number; converged: boolean} {
  let iterationCount = 0;
  for (let start = 0; start < maxIterations; start += groupSize) {
    const size = Math.min(groupSize, maxIterations - start);
    const changed: boolean[] = [];
    for (let offset = 0; offset < size; offset++) changed.push(round(start + offset));
    const firstUnchanged = changed.findIndex(value => !value);
    if (firstUnchanged >= 0) {
      return {iterationCount: iterationCount + firstUnchanged + 1, converged: true};
    }
    iterationCount += size;
  }
  return {iterationCount, converged: false};
}

/** Pointer jumping `pointer[i] = pointer[pointer[i]]` to roots; ping-pongs and ends in `primary`. */
function jumpPointers(primary: Uint32Array, maxIterations: number): number {
  const scratch = new Uint32Array(primary.length);
  const result = runGatedGroups(maxIterations, 4, iteration => {
    const from = iteration % 2 === 0 ? primary : scratch;
    const to = iteration % 2 === 0 ? scratch : primary;
    let changed = false;
    for (let index = 0; index < from.length; index++) {
      const next = from[index];
      const jumped = from[next];
      to[index] = jumped;
      if (jumped !== next) changed = true;
    }
    return changed;
  });
  return result.iterationCount;
}

/** Options that vary the order of the atomic operations, which must not change the result. */
export type SpillModelOptions = {
  tileSize?: number;
  /** Seed that shuffles the order of tree edge recording and of the half-edge lists. */
  shuffleSeed?: number;
};

/** Result of {@link modelSpillFill}. */
export type SpillModelResult = {
  filled: Float32Array;
  converged: boolean;
  /** Boruvka rounds executed, including the confirming round. */
  boruvkaRounds: number;
  /** Largest number of pointer-jumping rounds executed in one Boruvka round. */
  maximumJumpRounds: number;
  treeEdgeCount: number;
  nodeCount: number;
  tile: SpillTileResult;
  /** Bottleneck key to the ocean per label node (0 for nodes outside the tree). */
  pathKey: Uint32Array;
};

function shuffleOrder(count: number, seed: number | undefined): number[] {
  const order = Array.from({length: count}, (_, index) => index);
  if (seed === undefined) return order;
  const random = createSeededRandom(seed);
  for (let index = count - 1; index > 0; index--) {
    const other = Math.floor(random() * (index + 1));
    [order[index], order[other]] = [order[other], order[index]];
  }
  return order;
}

/** Stages 1 to 4 of the spill fill for epsilon 0. */
export function modelSpillFill(
  elevation: Float32Array,
  width: number,
  height: number,
  options: SpillModelOptions = {}
): SpillModelResult {
  const cellCount = width * height;
  const nodeCount = cellCount + 1;
  const ocean = cellCount;
  const tile = modelTileFill(elevation, width, height, options.tileSize ?? 32);
  const edgeSlotCount = 5 * cellCount;

  // Stage 3a: Boruvka.
  const comp = Uint32Array.from({length: nodeCount}, (_, index) => index);
  const mstFirst = new Uint32Array(cellCount);
  const mstSecond = new Uint32Array(cellCount);
  const mstKey = new Uint32Array(cellCount);
  let mstCount = 0;
  const rounds = getTerrainFlowSpillBoruvkaRoundCount(cellCount);
  const jumpRounds = getSpillRoundCount(nodeCount);
  let enabled = true;
  let converged = false;
  let boruvkaRounds = 0;
  let maximumJumpRounds = 0;
  for (let round = 0; round < rounds && enabled; round++) {
    const bestKey = new Uint32Array(nodeCount).fill(SPILL_NONE);
    const bestEdge = new Uint32Array(nodeCount).fill(SPILL_NONE);
    for (let slot = 0; slot < edgeSlotCount; slot++) {
      const edge = getSpillEdge(slot, width, height, tile);
      if (!edge.valid) continue;
      const firstComponent = comp[edge.first];
      const secondComponent = comp[edge.second];
      if (firstComponent === secondComponent) continue;
      bestKey[firstComponent] = Math.min(bestKey[firstComponent], edge.key);
      bestKey[secondComponent] = Math.min(bestKey[secondComponent], edge.key);
    }
    for (let slot = 0; slot < edgeSlotCount; slot++) {
      const edge = getSpillEdge(slot, width, height, tile);
      if (!edge.valid) continue;
      const firstComponent = comp[edge.first];
      const secondComponent = comp[edge.second];
      if (firstComponent === secondComponent) continue;
      if (edge.key === bestKey[firstComponent]) {
        bestEdge[firstComponent] = Math.min(bestEdge[firstComponent], slot);
      }
      if (edge.key === bestKey[secondComponent]) {
        bestEdge[secondComponent] = Math.min(bestEdge[secondComponent], slot);
      }
    }
    const hookPointer = new Uint32Array(nodeCount);
    let hooked = false;
    for (let index = 0; index < nodeCount; index++) {
      let hookTarget = index;
      if (comp[index] === index && index !== ocean) {
        const slot = bestEdge[index];
        if (slot !== SPILL_NONE) {
          const edge = getSpillEdge(slot, width, height, tile);
          const firstComponent = comp[edge.first];
          const secondComponent = comp[edge.second];
          const other = firstComponent === index ? secondComponent : firstComponent;
          const mutual = bestEdge[other] === slot;
          if (!(mutual && index > other)) {
            hookTarget = other;
            hooked = true;
          }
        }
      }
      hookPointer[index] = hookTarget;
    }
    // Record in a (possibly shuffled) order: the GPU uses an atomic counter.
    for (const index of shuffleOrder(nodeCount, options.shuffleSeed)) {
      if (hookPointer[index] === index) continue;
      const edge = getSpillEdge(bestEdge[index], width, height, tile);
      if (mstCount < cellCount) {
        mstFirst[mstCount] = edge.first;
        mstSecond[mstCount] = edge.second;
        mstKey[mstCount] = edge.key;
      }
      mstCount++;
    }
    if (hooked) {
      maximumJumpRounds = Math.max(maximumJumpRounds, jumpPointers(hookPointer, jumpRounds));
    }
    for (let index = 0; index < nodeCount; index++) comp[index] = hookPointer[comp[index]];
    boruvkaRounds++;
    if (!hooked) {
      enabled = false;
      converged = true;
    } else if (boruvkaRounds >= rounds) {
      enabled = false;
    }
  }

  // Stage 3b: Euler tour of the tree, ranked, rooted, bottleneck keys.
  const halfEdgeCount = 2 * cellCount;
  const head = new Uint32Array(nodeCount).fill(SPILL_NONE);
  const listNext = new Uint32Array(halfEdgeCount);
  for (const half of shuffleOrder(
    2 * mstCount,
    options.shuffleSeed === undefined ? undefined : options.shuffleSeed + 1
  )) {
    const edge = half >> 1;
    const origin = (half & 1) === 1 ? mstSecond[edge] : mstFirst[edge];
    listNext[half] = head[origin];
    head[origin] = half;
  }
  const successor = new Uint32Array(halfEdgeCount);
  const distance = new Uint32Array(halfEdgeCount);
  const rootStart = head[ocean];
  for (let half = 0; half < halfEdgeCount; half++) {
    if (half >= 2 * mstCount) {
      successor[half] = half;
      continue;
    }
    const edge = half >> 1;
    const arrival = (half & 1) === 1 ? mstFirst[edge] : mstSecond[edge];
    let following = listNext[half ^ 1];
    if (following === SPILL_NONE) following = head[arrival];
    if (following === rootStart) {
      successor[half] = half;
      distance[half] = 0;
    } else {
      successor[half] = following;
      distance[half] = 1;
    }
  }
  const scratchSuccessor = new Uint32Array(halfEdgeCount);
  const scratchDistance = new Uint32Array(halfEdgeCount);
  runGatedGroups(getSpillRoundCount(halfEdgeCount), 4, iteration => {
    const [succIn, distIn, succOut, distOut] =
      iteration % 2 === 0
        ? [successor, distance, scratchSuccessor, scratchDistance]
        : [scratchSuccessor, scratchDistance, successor, distance];
    let changed = false;
    for (let index = 0; index < halfEdgeCount; index++) {
      const next = succIn[index];
      const jumped = succIn[next];
      let ranked = distIn[index];
      if (next !== jumped) {
        ranked += distIn[next];
        changed = true;
      }
      succOut[index] = jumped;
      distOut[index] = ranked;
    }
    return changed;
  });
  const parent = Uint32Array.from({length: nodeCount}, (_, index) => index);
  const parentKey = new Uint32Array(nodeCount);
  for (let edge = 0; edge < mstCount; edge++) {
    const forward = distance[2 * edge] > distance[2 * edge + 1];
    const child = forward ? mstSecond[edge] : mstFirst[edge];
    const owner = forward ? mstFirst[edge] : mstSecond[edge];
    parent[child] = owner;
    parentKey[child] = mstKey[edge];
  }
  const scratchParent = new Uint32Array(nodeCount);
  const scratchParentKey = new Uint32Array(nodeCount);
  runGatedGroups(getSpillRoundCount(nodeCount), 4, iteration => {
    const [parentIn, keyIn, parentOut, keyOut] =
      iteration % 2 === 0
        ? [parent, parentKey, scratchParent, scratchParentKey]
        : [scratchParent, scratchParentKey, parent, parentKey];
    let changed = false;
    for (let index = 0; index < nodeCount; index++) {
      const ancestor = parentIn[index];
      const jumped = parentIn[ancestor];
      keyOut[index] = Math.max(keyIn[index], keyIn[ancestor]);
      parentOut[index] = jumped;
      if (jumped !== ancestor) changed = true;
    }
    return changed;
  });

  // Stage 4.
  const filled = new Float32Array(cellCount);
  let resolved = converged;
  for (let cell = 0; cell < cellCount; cell++) {
    if (!isFinite32(elevation[cell])) {
      filled[cell] = NaN;
      continue;
    }
    const word = tile.labels[cell];
    if (word === SPILL_NONE || parent[word & SPILL_ID_MASK] !== ocean) {
      filled[cell] = NaN;
      resolved = false;
      continue;
    }
    const key = Math.max(getSpillKey(tile.localFill[cell]), parentKey[word & SPILL_ID_MASK]);
    filled[cell] = getSpillValue(key);
  }
  return {
    filled,
    converged: resolved,
    boruvkaRounds,
    maximumJumpRounds,
    treeEdgeCount: mstCount,
    nodeCount,
    tile,
    pathKey: parentKey
  };
}

/** Number of unrolled rounds the GPU graph allocates, for graph-size reasoning in specs. */
export function getSpillUnrolledRoundCounts(cellCount: number) {
  return {
    boruvka: getTerrainFlowSpillBoruvkaRoundCount(cellCount),
    jump: getSpillRoundCount(cellCount + 1),
    rank: getSpillRoundCount(2 * cellCount),
    bitLength: getSpillBitLength(cellCount)
  };
}
