// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createSeededRandom} from '../terrain-oracle-utils';
import {getTerrainFlowTourRoundCount} from '../../../src/gpu-terrain/hydrology/terrain-flow-euler-tour';

/**
 * CPU model of the Euler-tour algorithms of `terrain-flow-euler-tour.ts`,
 * `createTerrainFlowAccumulationNodes` and `GPUTerrainStreamOrder`. It performs the same steps in
 * the same order as the kernels: tour links from a per-cell child enumeration in fixed direction
 * order, double-buffered Wyllie rounds in gate groups of four, exact `u32` arithmetic, exclusive
 * scans over tour positions, and the level loop with the same indicator definitions.
 */

/** Marker for "no cell" and "no edge" (`0xffffffff`). */
export const TOUR_NONE = 0xffffffff;
/** `GPU_TERRAIN_FLOW_UNRESOLVED` direction code. */
const UNRESOLVED_DIRECTION = 0xfffffffe;
const COLUMN_OFFSETS = [1, 1, 0, -1, -1, -1, 0, 1];
const ROW_OFFSETS = [0, 1, 1, 1, 0, -1, -1, -1];

/** A forest on a raster: `parents[cell]` is the parent cell or {@link TOUR_NONE}. */
export type TourForest = {width: number; height: number; parents: Uint32Array};

/** Returns the neighbor of `cell` in `direction` (0..7, east first, clockwise) or {@link TOUR_NONE}. */
export function getTourNeighbor(width: number, height: number, cell: number, direction: number) {
  const column = (cell % width) + COLUMN_OFFSETS[direction];
  const row = Math.floor(cell / width) + ROW_OFFSETS[direction];
  return column < 0 || row < 0 || column >= width || row >= height
    ? TOUR_NONE
    : row * width + column;
}

/** Ranked tour: the terminal of every edge and its distance to it. */
export type RankedTour = {
  successor: Uint32Array;
  distance: Uint32Array;
  /** Rounds executed, counted like the GPU gate (up to and including the first unchanged round). */
  roundsExecuted: number;
  /** 1 when a round changed nothing before the round limit. */
  converged: number;
};

/** Link kernel: initial successor and distance of the enter edge `2 v` and exit edge `2 v + 1`. */
export function linkTour(forest: TourForest): {successor: Uint32Array; distance: Uint32Array} {
  const {width, height, parents} = forest;
  const cellCount = width * height;
  const successor = new Uint32Array(2 * cellCount);
  const distance = new Uint32Array(2 * cellCount);
  for (let cell = 0; cell < cellCount; cell++) {
    const parent = parents[cell];
    let firstChild = TOUR_NONE;
    for (let direction = 0; direction < 8 && firstChild === TOUR_NONE; direction++) {
      const neighbor = getTourNeighbor(width, height, cell, direction);
      if (neighbor !== TOUR_NONE && parents[neighbor] === cell) firstChild = neighbor;
    }
    const exitEdge = 2 * cell + 1;
    successor[2 * cell] = firstChild !== TOUR_NONE ? 2 * firstChild : exitEdge;
    distance[2 * cell] = 1;
    successor[exitEdge] = exitEdge;
    distance[exitEdge] = 0;
    if (parent !== TOUR_NONE) {
      let sibling = TOUR_NONE;
      let passed = false;
      for (let direction = 0; direction < 8 && sibling === TOUR_NONE; direction++) {
        const neighbor = getTourNeighbor(width, height, parent, direction);
        if (neighbor === TOUR_NONE) continue;
        if (neighbor === cell) {
          passed = true;
        } else if (passed && parents[neighbor] === parent) {
          sibling = neighbor;
        }
      }
      successor[exitEdge] = sibling !== TOUR_NONE ? 2 * sibling : 2 * parent + 1;
      distance[exitEdge] = 1;
    }
  }
  return {successor, distance};
}

/** Wyllie rounds with ping-pong buffers and the gate-group stop rule of the GPU loop. */
export function rankTour(
  linked: {successor: Uint32Array; distance: Uint32Array},
  cellCount: number
): RankedTour {
  let successor = linked.successor.slice();
  let distance = linked.distance.slice();
  const edgeCount = 2 * cellCount;
  const maximumRounds = getTerrainFlowTourRoundCount(cellCount);
  const groupSize = 4;
  let roundsExecuted = 0;
  let converged = 0;
  for (let start = 0; start < maximumRounds && !converged; start += groupSize) {
    let executed = groupSize;
    for (let round = 0; round < groupSize; round++) {
      const nextSuccessor = new Uint32Array(edgeCount);
      const nextDistance = new Uint32Array(edgeCount);
      let changed = false;
      for (let edge = 0; edge < edgeCount; edge++) {
        const next = successor[edge];
        const target = successor[next];
        let value = distance[edge];
        if (next !== target) {
          value = (value + distance[next]) >>> 0;
          changed = true;
        }
        nextSuccessor[edge] = target;
        nextDistance[edge] = value;
      }
      successor = nextSuccessor;
      distance = nextDistance;
      // The kernels of a group keep running after a converged round; they only copy.
      if (!changed && !converged) {
        executed = round + 1;
        converged = 1;
      }
    }
    roundsExecuted += executed;
  }
  return {successor, distance, roundsExecuted, converged};
}

/** `isTourEdgeResolved`: the list of `edge` ends at the exit edge of a root. */
export function isEdgeResolved(forest: TourForest, ranked: RankedTour, edge: number): boolean {
  const terminal = ranked.successor[edge];
  return (terminal & 1) === 1 && forest.parents[terminal >>> 1] === TOUR_NONE;
}

/** Subtree sizes (cells including the cell) from ranks; 0 marks an unresolved cell. */
export function getTourSubtreeSizes(forest: TourForest, ranked: RankedTour): Uint32Array {
  const cellCount = forest.width * forest.height;
  const sizes = new Uint32Array(cellCount);
  for (let cell = 0; cell < cellCount; cell++) {
    if (isEdgeResolved(forest, ranked, 2 * cell)) {
      sizes[cell] = ((ranked.distance[2 * cell] - ranked.distance[2 * cell + 1] + 1) >>> 0) / 2;
    }
  }
  return sizes;
}

/**
 * Model of D8 'cells' accumulation: `receivers` as written by the flow-direction pass
 * (`TOUR_NONE` for roots and invalid cells), `valid` per cell. Returns float32 values, NaN for
 * invalid and unresolved cells, and the converged flag.
 */
export function modelTourAccumulation(
  width: number,
  height: number,
  receivers: Uint32Array,
  valid: ArrayLike<boolean>
): {accumulation: Float32Array; converged: number; roundsExecuted: number} {
  const forest: TourForest = {width, height, parents: receivers};
  const cellCount = width * height;
  const ranked = rankTour(linkTour(forest), cellCount);
  const sizes = getTourSubtreeSizes(forest, ranked);
  const accumulation = new Float32Array(cellCount);
  let converged = ranked.converged;
  for (let cell = 0; cell < cellCount; cell++) {
    if (!valid[cell]) {
      accumulation[cell] = NaN;
    } else if (!isEdgeResolved(forest, ranked, 2 * cell)) {
      accumulation[cell] = NaN;
      converged = 0;
    } else {
      accumulation[cell] = sizes[cell];
    }
  }
  return {accumulation, converged, roundsExecuted: ranked.roundsExecuted};
}

/** Direction code to receiver cell, mirroring `getFlowReceiver` of `terrain-flow-tracing.ts`. */
export function getFlowReceiverOfCode(
  width: number,
  height: number,
  cell: number,
  code: number
): number {
  if (code === 0 || code > 128 || (code & (code - 1)) !== 0) return TOUR_NONE;
  return getTourNeighbor(width, height, cell, 31 - Math.clz32(code));
}

/** Exclusive prefix sum modulo 2^32, like `GPUScan`. */
function scanExclusive(values: Uint32Array): Uint32Array {
  const result = new Uint32Array(values.length);
  let sum = 0;
  for (let index = 0; index < values.length; index++) {
    result[index] = sum;
    sum = (sum + values[index]) >>> 0;
  }
  return result;
}

/** Result of {@link modelStrahlerOrder}. */
export type StrahlerModelResult = {
  order: Uint32Array;
  converged: number;
  /** Levels executed before the loop ended (the gate counts every level it closes). */
  levelsExecuted: number;
};

/**
 * Model of `GPUTerrainStreamOrder`: stream forest, Euler tour, positions in the concatenated
 * tours, levels `A(k+1) = cells whose subtree has a cell with two children in A(k)`, and the
 * unknown-order pass.
 */
export function modelStrahlerOrder(props: {
  width: number;
  height: number;
  flowDirections: Uint32Array;
  streams: Uint32Array;
  /** `GPU_TERRAIN_FLOW_STREAM_UNRESOLVED`. */
  unresolvedStream: number;
}): StrahlerModelResult {
  const {width, height, flowDirections, streams} = props;
  const cellCount = width * height;
  const isStream = (cell: number) => streams[cell] !== 0 && flowDirections[cell] !== TOUR_NONE;
  const parents = new Uint32Array(cellCount).fill(TOUR_NONE);
  for (let cell = 0; cell < cellCount; cell++) {
    if (!isStream(cell)) continue;
    const receiver = getFlowReceiverOfCode(width, height, cell, flowDirections[cell]);
    if (receiver !== TOUR_NONE && isStream(receiver)) parents[cell] = receiver;
  }
  const forest: TourForest = {width, height, parents};
  const ranked = rankTour(linkTour(forest), cellCount);
  const edgeCount = 2 * cellCount;

  const rootLength = new Uint32Array(cellCount);
  for (let cell = 0; cell < cellCount; cell++) {
    const isRoot = isEdgeResolved(forest, ranked, 2 * cell) && parents[cell] === TOUR_NONE;
    rootLength[cell] = isRoot ? ranked.distance[2 * cell] + 1 : 0;
  }
  const rootOffset = scanExclusive(rootLength);
  const position = new Uint32Array(edgeCount);
  for (let edge = 0; edge < edgeCount; edge++) {
    if (!isEdgeResolved(forest, ranked, edge)) {
      position[edge] = TOUR_NONE;
      continue;
    }
    const root = ranked.successor[edge] >>> 1;
    position[edge] = rootOffset[root] + ranked.distance[2 * root] - ranked.distance[edge];
  }

  const order = new Uint32Array(cellCount);
  const member = new Uint8Array(cellCount);
  for (let cell = 0; cell < cellCount; cell++) {
    order[cell] = isStream(cell) ? 1 : 0;
    member[cell] = isStream(cell) ? 1 : 0;
  }
  // Stale slots (positions no resolved edge covers) hold garbage on the GPU; the model poisons them.
  const indicator = new Uint32Array(edgeCount).fill(0xdeadbeef);
  const getSubtreeCount = (cell: number, prefix: Uint32Array) =>
    (prefix[position[2 * cell + 1]] - prefix[position[2 * cell]]) >>> 0;

  const levelCount = Math.min(32, Math.max(1, 31 - Math.clz32(Math.max(cellCount, 1))));
  let enabled = true;
  let levelsExecuted = 0;
  for (let level = 0; level < levelCount; level++) {
    // Gated kernels do nothing once the gate disarmed; the scan always runs.
    if (enabled) {
      for (let edge = 0; edge < edgeCount; edge++) {
        const place = position[edge];
        if (place === TOUR_NONE) continue;
        let junction = 0;
        if ((edge & 1) === 0) {
          const cell = edge >>> 1;
          let childCount = 0;
          for (let direction = 0; direction < 8; direction++) {
            const neighbor = getTourNeighbor(width, height, cell, direction);
            if (neighbor !== TOUR_NONE && parents[neighbor] === cell && member[neighbor]) {
              childCount++;
            }
          }
          junction = childCount >= 2 ? 1 : 0;
        }
        indicator[place] = junction;
      }
    }
    const prefix = scanExclusive(indicator);
    if (enabled) {
      let joined = 0;
      for (let cell = 0; cell < cellCount; cell++) {
        if (!member[cell] || position[2 * cell] === TOUR_NONE) continue;
        if (getSubtreeCount(cell, prefix) > 0) {
          order[cell]++;
          joined++;
        } else {
          member[cell] = 0;
        }
      }
      levelsExecuted++;
      if (joined === 0) enabled = false;
    }
  }

  // Unknown orders.
  for (let edge = 0; edge < edgeCount; edge++) {
    const place = position[edge];
    if (place === TOUR_NONE) continue;
    let seed = 0;
    const cell = edge >>> 1;
    if ((edge & 1) === 0 && isStream(cell)) {
      let isSeed =
        streams[cell] === props.unresolvedStream || flowDirections[cell] === UNRESOLVED_DIRECTION;
      for (let direction = 0; direction < 8; direction++) {
        const neighbor = getTourNeighbor(width, height, cell, direction);
        if (
          neighbor !== TOUR_NONE &&
          isStream(neighbor) &&
          flowDirections[neighbor] === UNRESOLVED_DIRECTION
        ) {
          isSeed = true;
        }
      }
      seed = isSeed ? 1 : 0;
    }
    indicator[place] = seed;
  }
  const prefix = scanExclusive(indicator);
  let converged = ranked.converged;
  for (let cell = 0; cell < cellCount; cell++) {
    if (!isStream(cell)) continue;
    if (position[2 * cell] === TOUR_NONE) {
      order[cell] = TOUR_NONE;
      converged = 0;
    } else if (getSubtreeCount(cell, prefix) > 0) {
      order[cell] = TOUR_NONE;
    }
  }
  return {order, converged, levelsExecuted};
}

/** Direct recursive oracle: subtree sizes by explicit post-order, independent of the tour. */
export function computeSubtreeSizesRecursive(forest: TourForest): Uint32Array {
  const {width, height, parents} = forest;
  const cellCount = width * height;
  const children: number[][] = Array.from({length: cellCount}, () => []);
  for (let cell = 0; cell < cellCount; cell++) {
    if (parents[cell] !== TOUR_NONE) children[parents[cell]].push(cell);
  }
  const sizes = new Uint32Array(cellCount);
  const visited = new Uint8Array(cellCount);
  for (let root = 0; root < cellCount; root++) {
    if (parents[root] !== TOUR_NONE) continue;
    const stack: [number, number][] = [[root, 0]];
    visited[root] = 1;
    sizes[root] = 1;
    while (stack.length) {
      const top = stack[stack.length - 1];
      const [cell, next] = top;
      if (next < children[cell].length) {
        top[1]++;
        const child = children[cell][next];
        visited[child] = 1;
        sizes[child] = 1;
        stack.push([child, 0]);
      } else {
        stack.pop();
        if (stack.length) sizes[stack[stack.length - 1][0]] += sizes[cell];
      }
    }
  }
  return sizes;
}

/**
 * Direct recursive Strahler oracle with the pull contract: donors are stream neighbors whose code
 * points back; unresolved streams and unknown-direction neighbors poison the cell and everything
 * downstream. Receiver cycles are not supported.
 */
export function computeStrahlerOrderRecursive(props: {
  width: number;
  height: number;
  flowDirections: Uint32Array;
  streams: Uint32Array;
  unresolvedStream: number;
}): Uint32Array {
  const {width, height, flowDirections, streams} = props;
  const cellCount = width * height;
  const isStream = (cell: number) => streams[cell] !== 0 && flowDirections[cell] !== TOUR_NONE;
  const memo = new Map<number, number>();
  const getOrder = (cell: number): number => {
    const known = memo.get(cell);
    if (known !== undefined) return known;
    let result: number;
    if (streams[cell] === props.unresolvedStream || flowDirections[cell] === UNRESOLVED_DIRECTION) {
      result = TOUR_NONE;
    } else {
      let maximumOrder = 0;
      let maximumCount = 0;
      let unknown = false;
      for (let direction = 0; direction < 8; direction++) {
        const donor = getTourNeighbor(width, height, cell, direction);
        if (donor === TOUR_NONE || streams[donor] === 0) continue;
        if (flowDirections[donor] === UNRESOLVED_DIRECTION) {
          unknown = true;
          continue;
        }
        if (getFlowReceiverOfCode(width, height, donor, flowDirections[donor]) !== cell) continue;
        const donorOrder = getOrder(donor);
        if (donorOrder === TOUR_NONE) {
          unknown = true;
        } else if (donorOrder > maximumOrder) {
          maximumOrder = donorOrder;
          maximumCount = 1;
        } else if (donorOrder === maximumOrder) {
          maximumCount++;
        }
      }
      result = unknown
        ? TOUR_NONE
        : maximumCount === 0
          ? 1
          : maximumOrder + (maximumCount >= 2 ? 1 : 0);
    }
    memo.set(cell, result);
    return result;
  };
  const order = new Uint32Array(cellCount);
  for (let cell = 0; cell < cellCount; cell++) {
    // Recursion depth is the stream length: raise the cell order bottom-up in index order is not
    // needed here because callers keep recursive cases short (stems use the iterative oracle).
    order[cell] = isStream(cell) ? getOrder(cell) : 0;
  }
  return order;
}

// Forest builders shared by the node and GPU specs.

/** Direction code that sends `cell` to its neighbor `parent`. */
export function getCode(width: number, height: number, cell: number, parent: number): number {
  for (let direction = 0; direction < 8; direction++) {
    if (getTourNeighbor(width, height, cell, direction) === parent) return 1 << direction;
  }
  throw new Error('parent is not a neighbor');
}

/** Codes for a parent array: 0 for roots. */
export function getCodes(forest: TourForest): Uint32Array {
  const codes = new Uint32Array(forest.parents.length);
  for (let cell = 0; cell < codes.length; cell++) {
    const parent = forest.parents[cell];
    codes[cell] = parent === TOUR_NONE ? 0 : getCode(forest.width, forest.height, cell, parent);
  }
  return codes;
}

export function makeStem(length: number): TourForest {
  const parents = new Uint32Array(length).fill(TOUR_NONE);
  for (let cell = 0; cell + 1 < length; cell++) parents[cell] = cell + 1;
  return {width: length, height: 1, parents};
}

/** Spine along the middle row of three rows, one leaf above and one below every spine cell. */
export function makeCaterpillar(length: number): TourForest {
  const parents = new Uint32Array(3 * length).fill(TOUR_NONE);
  for (let column = 0; column < length; column++) {
    parents[length + column] = column + 1 < length ? length + column + 1 : TOUR_NONE;
    parents[column] = length + column;
    parents[2 * length + column] = length + column;
  }
  return {width: length, height: 3, parents};
}

/**
 * Complete binary tree of Strahler order `order`: every junction has two children of order
 * `order - 1` entering from opposite sides, joined by straight stems that double in length per level.
 */
export function makeStrahlerTree(order: number): TourForest & {root: number} {
  const cells = new Map<string, [number, number] | null>();
  const key = (x: number, y: number) => `${x},${y}`;
  const build = (
    x: number,
    y: number,
    dx: number,
    dy: number,
    level: number,
    parent: [number, number] | null
  ) => {
    const stemLength = 2 ** level + 1;
    let previous = parent;
    for (let step = 0; step < stemLength; step++) {
      const cx = x - step * dx;
      const cy = y - step * dy;
      if (cells.has(key(cx, cy))) throw new Error('layout collision');
      cells.set(key(cx, cy), previous);
      previous = [cx, cy];
    }
    if (level > 1) {
      const jx = x - (stemLength - 1) * dx;
      const jy = y - (stemLength - 1) * dy;
      // Children enter the junction from both sides, flowing toward it.
      build(jx - dy, jy + dx, dy, -dx, level - 1, [jx, jy]);
      build(jx + dy, jy - dx, -dy, dx, level - 1, [jx, jy]);
    }
  };
  build(0, 0, 1, 0, order, null);
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const name of cells.keys()) {
    const [cx, cy] = name.split(',').map(Number);
    minX = Math.min(minX, cx);
    maxX = Math.max(maxX, cx);
    minY = Math.min(minY, cy);
    maxY = Math.max(maxY, cy);
  }
  const width = maxX - minX + 1;
  const height = maxY - minY + 1;
  const parents = new Uint32Array(width * height).fill(TOUR_NONE);
  for (const [name, parent] of cells) {
    const [cx, cy] = name.split(',').map(Number);
    if (parent)
      parents[(cy - minY) * width + cx - minX] = (parent[1] - minY) * width + parent[0] - minX;
  }
  return {width, height, parents, root: -minY * width - minX};
}

/** Random acyclic forest: a random potential and a random lower neighbor; some cells are holes. */
export function makeRandomForest(width: number, height: number, seed: number, holeRate: number) {
  const random = createSeededRandom(seed);
  const cellCount = width * height;
  const potential = Float64Array.from({length: cellCount}, () => random());
  const valid = Array.from({length: cellCount}, () => random() >= holeRate);
  const parents = new Uint32Array(cellCount).fill(TOUR_NONE);
  for (let cell = 0; cell < cellCount; cell++) {
    if (!valid[cell] || random() < 0.1) continue;
    const candidates: number[] = [];
    for (let direction = 0; direction < 8; direction++) {
      const neighbor = getTourNeighbor(width, height, cell, direction);
      if (neighbor !== TOUR_NONE && valid[neighbor] && potential[neighbor] < potential[cell]) {
        candidates.push(neighbor);
      }
    }
    if (candidates.length) parents[cell] = candidates[Math.floor(random() * candidates.length)];
  }
  return {forest: {width, height, parents} as TourForest, valid};
}
