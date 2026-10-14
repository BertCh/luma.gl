// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * CPU mirror of `GPUDelaunayTessellation` (parallel insertion and flipping). It transcribes the
 * WGSL kernels of `gpu-delaunay-tessellation.ts` one for one: the same topology arrays, the same
 * vote keys and `fmix32` priorities, the same split and flip formulas, FRESH/replacement
 * adjacency repair, children-based relocation, the same step gate and finalize logic, and the
 * same canonical output sort. Each kernel is run as a sequential loop over its threads, which is
 * faithful because the kernels are race free by construction (votes are read from the pre-phase
 * state, winners touch disjoint triangles, repair and relocation write only their own thread's
 * words). The only schedule dependent choice is the slot allocation order of the winners (the
 * `atomicAdd` order); `winnerOrder` varies it.
 *
 * Predicates are exact: a double-precision filter (the f32 inputs are exact doubles) in front of
 * the BigInt determinants of `delaunay-reference.ts`.
 *
 * Test only. The WGSL is the source of truth; where this file and the design note disagree, this
 * file follows the WGSL.
 */

import {getDelaunayStepCount} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation';
import {incircleExact, mulberry32, orientExact, toExactIntegers} from './delaunay-reference';

const NONE = 0xffffffff;
const FRESH = 0x80000000;

/** Order in which simultaneous winners allocate their triangle slots. */
export type WinnerOrder = 'ascending' | 'descending' | {shuffleSeed: number};

export type ParallelDelaunayOptions = {
  /** Slot allocation order of the winners of one phase. Defaults to ascending site row. */
  winnerOrder?: WinnerOrder;
  /** Number of unrolled steps. Defaults to `getDelaunayStepCount(pointCount)`. */
  stepCount?: number;
  /** Check adjacency symmetry and counter-clockwise orientation after every phase (slow). */
  validate?: boolean;
};

export type ParallelDelaunayResult = {
  /** All-real triangles, counter-clockwise, smallest row first, sorted by (v0, v1). */
  triangles: number[][];
  /** Steps in which the gate was active (insertion and flip pass); the count the cap must cover. */
  activeSteps: number;
  /** Steps that ran an insertion phase. */
  insertionSteps: number;
  /** Total flips applied. */
  flips: number;
  /** Total triangle slots allocated. */
  slotCount: number;
  /** `invalidCount` as published: invalid rows plus one on internal failure. */
  invalidCount: number;
  duplicateCount: number;
  /** `status.requiredCount`. */
  requiredCount: number;
  /** True when the steps ran out or scratch capacity was exceeded. */
  failed: boolean;
  /** Unrolled step count used. */
  stepCount: number;
  /** Illegal edges found by the flip evaluation of each active step. */
  illegalHistory: number[];
};

/** murmur3 `fmix32`, as `getInsertionPriority`. */
export function getInsertionPriority(row: number): number {
  let hash = row >>> 0;
  hash = (hash ^ (hash >>> 16)) >>> 0;
  hash = Math.imul(hash, 0x85ebca6b) >>> 0;
  hash = (hash ^ (hash >>> 13)) >>> 0;
  hash = Math.imul(hash, 0xc2b2ae35) >>> 0;
  hash = (hash ^ (hash >>> 16)) >>> 0;
  return hash;
}

/** `getFlipPriority`: a bijection on [0, 2^31) that scrambles the slot-ordered edge ids. */
export function getFlipPriority(edgeId: number): number {
  let hash = edgeId & 0x7fffffff;
  hash ^= hash >>> 15;
  hash = Math.imul(hash, 0x2c1b3c6d) & 0x7fffffff;
  hash ^= hash >>> 12;
  hash = Math.imul(hash, 0x297a2d39) & 0x7fffffff;
  hash ^= hash >>> 15;
  return hash >>> 0;
}

/** Exact orientation and in-circle predicates over f32 rows, double filter in front of BigInt. */
export function createFilteredPredicates(coordinates: Float32Array) {
  const X = (row: number) => coordinates[2 * row];
  const Y = (row: number) => coordinates[2 * row + 1];
  // Small integer coordinates make every double product exact, so the filter is the answer.
  let integerExact = true;
  for (let index = 0; index < coordinates.length; index++) {
    const value = coordinates[index];
    if (!Number.isInteger(value) || Math.abs(value) > 1024) {
      integerExact = false;
      break;
    }
  }
  let exact: bigint[] | undefined;
  const exactRow = (row: number) => {
    exact ??= toExactIntegers(coordinates);
    return [exact[2 * row], exact[2 * row + 1]];
  };
  const sign = (value: number) => (value > 0 ? 1 : value < 0 ? -1 : 0);

  /** Sign of orient(a, b, p), real rows. */
  function orientReal(a: number, b: number, p: number): number {
    const left = (X(b) - X(a)) * (Y(p) - Y(a));
    const right = (Y(b) - Y(a)) * (X(p) - X(a));
    const determinant = left - right;
    if (integerExact || Math.abs(determinant) > 1e-13 * (Math.abs(left) + Math.abs(right))) {
      return sign(determinant);
    }
    return orientExact(exactRow(a), exactRow(b), exactRow(p));
  }

  /** Sign of the in-circle determinant: 1 when p is strictly inside the ccw circle abc. */
  function incircleReal(a: number, b: number, c: number, p: number): number {
    const ax = X(a) - X(p);
    const ay = Y(a) - Y(p);
    const bx = X(b) - X(p);
    const by = Y(b) - Y(p);
    const cx = X(c) - X(p);
    const cy = Y(c) - Y(p);
    const aLift = ax * ax + ay * ay;
    const bLift = bx * bx + by * by;
    const cLift = cx * cx + cy * cy;
    const determinant =
      aLift * (bx * cy - cx * by) - bLift * (ax * cy - cx * ay) + cLift * (ax * by - bx * ay);
    if (integerExact) {
      return sign(determinant);
    }
    const permanent =
      aLift * (Math.abs(bx * cy) + Math.abs(cx * by)) +
      bLift * (Math.abs(ax * cy) + Math.abs(cx * ay)) +
      cLift * (Math.abs(ax * by) + Math.abs(bx * ay));
    if (Math.abs(determinant) > 1e-9 * permanent) {
      return sign(determinant);
    }
    return incircleExact(exactRow(a), exactRow(b), exactRow(c), exactRow(p));
  }
  return {orientReal, incircleReal};
}

/**
 * The WGSL predicate layer (`orientSymbolic`, `orientMostlySuper`, `getInsideSign`,
 * `isFlipRequired`) over rows `0 .. pointCount - 1` (real) and `pointCount .. + 2` (symbolic
 * super vertices).
 */
export function createFlipPredicates(coordinates: Float32Array) {
  const pointCount = coordinates.length / 2;
  const {orientReal, incircleReal} = createFilteredPredicates(coordinates);
  const compareCoordinates = (a: number, b: number) => (a < b ? -1 : a > b ? 1 : 0);

  function orientRealSuper(r: number, superIndex: number, p: number): number {
    const dx = compareCoordinates(coordinates[2 * p], coordinates[2 * r]);
    const dy = compareCoordinates(coordinates[2 * p + 1], coordinates[2 * r + 1]);
    if (superIndex === 0) {
      return dy !== 0 ? dy : dx;
    }
    if (superIndex === 1) {
      return dx !== 0 ? -dx : dy;
    }
    return dy !== 0 ? -dy : dx;
  }

  /** Sign of orient(a, b, p) for a real site p. */
  function orientSymbolic(a: number, b: number, p: number): number {
    if (a < pointCount && b < pointCount) {
      return orientReal(a, b, p);
    }
    if (a < pointCount) {
      return orientRealSuper(a, b - pointCount, p);
    }
    if (b < pointCount) {
      return -orientRealSuper(b, a - pointCount, p);
    }
    return b - pointCount === (a - pointCount + 1) % 3 ? 1 : -1;
  }

  function isInsideOpenSegment(a: number, b: number, p: number): boolean {
    const ax = coordinates[2 * a];
    const bx = coordinates[2 * b];
    if (ax !== bx) {
      return coordinates[2 * p] > Math.min(ax, bx) && coordinates[2 * p] < Math.max(ax, bx);
    }
    const ay = coordinates[2 * a + 1];
    const by = coordinates[2 * b + 1];
    return coordinates[2 * p + 1] > Math.min(ay, by) && coordinates[2 * p + 1] < Math.max(ay, by);
  }

  function orientMostlySuper(x: number, y: number, z: number): number {
    const rows = [x, y, z];
    let realSlot = 3;
    for (let slot = 0; slot < 3; slot++) {
      if (rows[slot] < pointCount) {
        realSlot = slot;
      }
    }
    if (realSlot === 3) {
      return rows[1] - pointCount === (rows[0] - pointCount + 1) % 3 ? 1 : -1;
    }
    const first = rows[(realSlot + 1) % 3] - pointCount;
    const second = rows[(realSlot + 2) % 3] - pointCount;
    return second === (first + 1) % 3 ? 1 : -1;
  }

  function getInsideSign(x: number, y: number, z: number, p: number): number {
    const rows = [x, y, z, p];
    let farthestSlot = 3;
    for (let slot = 0; slot < 3; slot++) {
      if (rows[slot] >= pointCount && (farthestSlot === 3 || rows[slot] > rows[farthestSlot])) {
        farthestSlot = slot;
      }
    }
    if (farthestSlot === 3) {
      const incircle = incircleReal(x, y, z, p);
      if (incircle !== 0) {
        return incircle;
      }
    }
    let first = 0;
    let second = 0;
    let third = p;
    let cofactorSign = 1;
    if (farthestSlot !== 3) {
      first = rows[(farthestSlot + 1) % 3];
      second = rows[(farthestSlot + 2) % 3];
    } else {
      let dominant = 0;
      for (let slot = 1; slot < 4; slot++) {
        if (rows[slot] < rows[dominant]) {
          dominant = slot;
        }
      }
      const others: number[] = [];
      for (let slot = 0; slot < 4; slot++) {
        if (slot !== dominant) {
          others.push(rows[slot]);
        }
      }
      [first, second, third] = others;
      cofactorSign = (dominant & 1) === 1 ? -1 : 1;
    }
    const side = orientSymbolic(first, second, third);
    if (farthestSlot !== 3) {
      if (side !== 0) {
        return side;
      }
      return isInsideOpenSegment(first, second, p) ? 1 : -1;
    }
    return cofactorSign * side;
  }

  function isFlipRequired(a: number, b: number, c: number, d: number): boolean {
    let x = a;
    let y = b;
    let z = c;
    let p = d;
    let parity = 1;
    let knownCounterClockwise = true;
    if (d >= pointCount) {
      if (c < pointCount) {
        x = b;
        y = a;
        z = d;
        p = c;
      } else if (a < pointCount) {
        x = b;
        y = c;
        z = d;
        p = a;
        parity = -1;
        knownCounterClockwise = false;
      } else {
        x = a;
        y = c;
        z = d;
        p = b;
        knownCounterClockwise = false;
      }
    }
    if (!knownCounterClockwise && orientMostlySuper(x, y, z) < 0) {
      const swap = x;
      x = y;
      y = swap;
      parity = -parity;
    }
    return parity * getInsideSign(x, y, z, p) > 0;
  }

  return {
    pointCount,
    orientSymbolic,
    orientMostlySuper,
    getInsideSign,
    isFlipRequired,
    incircleReal,
    orientReal
  };
}

function shuffleInPlace(values: number[], seed: number): void {
  const random = mulberry32(seed);
  for (let index = values.length - 1; index > 0; index--) {
    const other = Math.floor(random() * (index + 1));
    [values[index], values[other]] = [values[other], values[index]];
  }
}

/** Runs the mirror. See the file comment. */
export function triangulateParallelReference(
  coordinates: Float32Array,
  options: ParallelDelaunayOptions = {}
): ParallelDelaunayResult {
  const pointCount = coordinates.length / 2;
  const stepCount = options.stepCount ?? getDelaunayStepCount(pointCount);
  const result: ParallelDelaunayResult = {
    triangles: [],
    activeSteps: 0,
    insertionSteps: 0,
    flips: 0,
    slotCount: 0,
    invalidCount: 0,
    duplicateCount: 0,
    requiredCount: 0,
    failed: false,
    stepCount,
    illegalHistory: []
  };
  if (pointCount === 0) {
    return result;
  }

  // --- classify-sites, site-sort, flag-duplicate-sites ---
  let invalid = 0;
  const duplicateFlags = new Uint8Array(pointCount);
  const owners = new Set<string>();
  for (let row = 0; row < pointCount; row++) {
    const x = coordinates[2 * row];
    const y = coordinates[2 * row + 1];
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      invalid++;
      continue;
    }
    const key = `${x + 0},${y + 0}`;
    if (owners.has(key)) {
      duplicateFlags[row] = 1;
      result.duplicateCount++;
    }
    owners.add(key);
  }
  result.invalidCount = invalid;

  const capacity = Math.max(2 * pointCount + 8, 8);
  const triangleVertices = new Uint32Array(4 * capacity);
  const triangleAdjacency = new Uint32Array(4 * capacity);
  const replacement = new Uint32Array(3 * capacity);
  const children = new Uint32Array(4 * capacity);
  const votes = new Uint32Array(capacity).fill(NONE);
  const pointLocations = new Uint32Array(pointCount);
  // [triangleCount, failed, remaining, phase, illegalCount, insertedCount, stepCount, unconverged]
  const state = new Uint32Array(8);

  // --- initialize-triangles / initialize-sites ---
  const triangulate = invalid === 0;
  for (let index = 0; index < capacity; index++) {
    triangleVertices[4 * index] = index === 0 && triangulate ? pointCount : NONE;
    triangleVertices[4 * index + 1] = index === 0 && triangulate ? pointCount + 1 : NONE;
    triangleVertices[4 * index + 2] = index === 0 && triangulate ? pointCount + 2 : NONE;
    triangleVertices[4 * index + 3] = 0;
    for (let word = 0; word < 3; word++) {
      triangleAdjacency[4 * index + word] = NONE;
    }
  }
  if (triangulate) {
    state[0] = 1;
    state[4] = 1;
  }
  for (let row = 0; row < pointCount; row++) {
    const located = triangulate && duplicateFlags[row] === 0;
    pointLocations[row] = located ? 3 : NONE;
    if (located) {
      state[2]++;
    }
  }

  const predicates = createFlipPredicates(coordinates);
  const {orientSymbolic, isFlipRequired} = predicates;

  const getVertex = (triangle: number, vertex: number) => triangleVertices[4 * triangle + vertex];
  const getStamp = (triangle: number) => triangleVertices[4 * triangle + 3];
  const getAdjacency = (triangle: number, edge: number) =>
    triangleAdjacency[4 * triangle + (edge % 3)];
  const getReference = (triangle: number, edge: number) => 3 * triangle + edge;
  const writeTriangle = (triangle: number, a: number, b: number, c: number, phase: number) => {
    triangleVertices[4 * triangle] = a;
    triangleVertices[4 * triangle + 1] = b;
    triangleVertices[4 * triangle + 2] = c;
    triangleVertices[4 * triangle + 3] = phase;
  };
  const writeAdjacency = (triangle: number, edge0: number, edge1: number, edge2: number) => {
    triangleAdjacency[4 * triangle] = edge0;
    triangleAdjacency[4 * triangle + 1] = edge1;
    triangleAdjacency[4 * triangle + 2] = edge2;
    triangleAdjacency[4 * triangle + 3] = 0;
  };
  const writeReplacement = (triangle: number, edge: number, reference: number) => {
    replacement[3 * triangle + (edge % 3)] = reference;
  };
  const writeChildren = (triangle: number, first: number, second: number, third: number) => {
    children[4 * triangle] = first;
    children[4 * triangle + 1] = second;
    children[4 * triangle + 2] = third;
  };
  const liveCount = () => Math.min(state[0], capacity);

  function validateTopology(label: string): void {
    const count = liveCount();
    for (let triangle = 0; triangle < count; triangle++) {
      const rows = [getVertex(triangle, 0), getVertex(triangle, 1), getVertex(triangle, 2)];
      // Counter-clockwise: rotate so the last row is real when there is one.
      let slot = rows.findIndex(row => row < pointCount);
      let orientation = 1;
      if (slot >= 0) {
        orientation = orientSymbolic(rows[(slot + 1) % 3], rows[(slot + 2) % 3], rows[slot]);
      } else {
        orientation = predicates.orientMostlySuper(rows[0], rows[1], rows[2]);
      }
      if (orientation <= 0) {
        throw new Error(`${label}: triangle ${triangle} (${rows}) is not counter-clockwise`);
      }
      slot = 0;
      for (let edge = 0; edge < 3; edge++) {
        const reference = getAdjacency(triangle, edge);
        if (reference === NONE) {
          continue;
        }
        if ((reference & FRESH) !== 0) {
          throw new Error(`${label}: FRESH left on ${triangle}/${edge}`);
        }
        const other = Math.floor(reference / 3);
        const otherEdge = reference % 3;
        if (other >= count || getAdjacency(other, otherEdge) !== 3 * triangle + edge) {
          throw new Error(`${label}: asymmetric adjacency ${triangle}/${edge} -> ${reference}`);
        }
        if (
          getVertex(other, otherEdge) !== getVertex(triangle, (edge + 1) % 3) ||
          getVertex(other, (otherEdge + 1) % 3) !== getVertex(triangle, edge)
        ) {
          throw new Error(`${label}: edge ${triangle}/${edge} does not match its twin`);
        }
      }
    }
  }

  function orderWinners(rows: number[]): number[] {
    const order = options.winnerOrder ?? 'ascending';
    if (order === 'descending') {
      rows.reverse();
    } else if (order !== 'ascending') {
      shuffleInPlace(rows, order.shuffleSeed);
    }
    return rows;
  }

  // --- adjacency repair + relocation, shared by both phases (variant: phase offset) ---
  function repairAdjacency(phaseOffset: number): void {
    const phase = state[3] + phaseOffset;
    for (let index = 0; index < liveCount(); index++) {
      votes[index] = NONE;
      for (let edge = 0; edge < 3; edge++) {
        const reference = triangleAdjacency[4 * index + edge];
        if (reference === NONE) {
          continue;
        }
        let resolved = reference;
        if ((reference & FRESH) !== 0) {
          resolved = (reference & ~FRESH) >>> 0;
        } else if (getStamp(Math.floor(reference / 3)) === phase) {
          resolved = replacement[reference];
        }
        triangleAdjacency[4 * index + edge] = resolved;
      }
    }
  }

  function relocate(phaseOffset: number): void {
    const phase = state[3] + phaseOffset;
    for (let site = 0; site < pointCount; site++) {
      const location = pointLocations[site];
      if (location === NONE) {
        continue;
      }
      const triangle = location >>> 2;
      if (getStamp(triangle) !== phase) {
        continue;
      }
      let found = NONE;
      let foundCode = 3;
      for (let slot = 0; slot < 3; slot++) {
        const candidate = children[4 * triangle + slot];
        if (candidate === NONE || found !== NONE) {
          continue;
        }
        let outside = false;
        let zeroEdge = 3;
        for (let edge = 0; edge < 3; edge++) {
          const side = orientSymbolic(
            getVertex(candidate, edge),
            getVertex(candidate, (edge + 1) % 3),
            site
          );
          if (side < 0) {
            outside = true;
          }
          if (side === 0) {
            zeroEdge = edge;
          }
        }
        if (!outside) {
          found = candidate;
          foundCode = zeroEdge;
        }
      }
      if (found === NONE) {
        state[1] = 1;
        continue;
      }
      pointLocations[site] = ((found << 2) | foundCode) >>> 0;
    }
  }

  // The same unsigned coordinate-bit ordering used by the GPU duplicate sort.
  const coordinateBits = new Uint32Array(
    coordinates.buffer,
    coordinates.byteOffset,
    coordinates.length
  );
  const coordinateKey = (coordinate: number) =>
    coordinates[coordinate] === 0 ? 0 : coordinateBits[coordinate];
  const sortedSites = Array.from({length: pointCount}, (_, row) => row)
    .filter(row => duplicateFlags[row] === 0)
    .sort(
      (left, right) =>
        coordinateKey(2 * left) - coordinateKey(2 * right) ||
        coordinateKey(2 * left + 1) - coordinateKey(2 * right + 1) ||
        left - right
    );
  const firstSeed = sortedSites[0];
  const secondSeed = sortedSites[sortedSites.length - 1];
  let thirdSeed = NONE;
  if (invalid === 0) {
    for (let site = 0; site < pointCount; site++) {
      if (predicates.orientReal(firstSeed, secondSeed, site) !== 0) {
        thirdSeed = site;
        break;
      }
    }
  }
  const seeds = [firstSeed, secondSeed, thirdSeed];

  // --- the unrolled steps ---
  for (let step = 0; step < stepCount; step++) {
    // step-gate
    state[7] = state[2] > 0 || state[4] > 0 ? 1 : 0;
    const active = state[1] === 0 && state[7] !== 0;
    const inserting = active && state[2] > 0;
    state[3] += 2;
    if (active) {
      state[4] = 0;
      state[5] = 0;
      state[6]++;
      result.activeSteps++;
    }
    if (inserting) {
      result.insertionSteps++;
      // insertion-vote
      for (let site = 0; site < pointCount; site++) {
        const location = pointLocations[site];
        if (location === NONE || (state[6] <= 3 && site !== seeds[state[6] - 1])) {
          continue;
        }
        const triangle = location >>> 2;
        const code = location & 3;
        const priority = getInsertionPriority(site);
        votes[triangle] = Math.min(votes[triangle], priority);
        if (code < 3) {
          const reference = triangleAdjacency[4 * triangle + code];
          const neighbor = Math.floor(reference / 3);
          votes[neighbor] = Math.min(votes[neighbor], priority);
        }
      }
      // insertion-split, winners in allocation order
      const candidates: number[] = [];
      for (let site = 0; site < pointCount; site++) {
        if (pointLocations[site] !== NONE && (state[6] > 3 || site === seeds[state[6] - 1])) {
          candidates.push(site);
        }
      }
      for (const site of orderWinners(candidates)) {
        const location = pointLocations[site];
        const triangle = location >>> 2;
        const code = location & 3;
        const priority = getInsertionPriority(site);
        if (votes[triangle] !== priority) {
          continue;
        }
        let neighbor = NONE;
        let neighborEdge = 0;
        if (code < 3) {
          const reference = getAdjacency(triangle, code);
          neighbor = Math.floor(reference / 3);
          neighborEdge = reference % 3;
          if (votes[neighbor] !== priority) {
            continue;
          }
        }
        const first = state[0];
        state[0] += 2;
        if (first + 2 > capacity) {
          state[1] = 1;
          continue;
        }
        const second = first + 1;
        const phase = state[3];
        if (code === 3) {
          const a = getVertex(triangle, 0);
          const b = getVertex(triangle, 1);
          const c = getVertex(triangle, 2);
          const old0 = getAdjacency(triangle, 0);
          const old1 = getAdjacency(triangle, 1);
          const old2 = getAdjacency(triangle, 2);
          writeTriangle(triangle, a, b, site, phase);
          writeTriangle(first, b, c, site, phase);
          writeTriangle(second, c, a, site, phase);
          writeAdjacency(
            triangle,
            old0,
            FRESH | getReference(first, 2),
            FRESH | getReference(second, 1)
          );
          writeAdjacency(
            first,
            old1,
            FRESH | getReference(second, 2),
            FRESH | getReference(triangle, 1)
          );
          writeAdjacency(
            second,
            old2,
            FRESH | getReference(triangle, 2),
            FRESH | getReference(first, 1)
          );
          writeReplacement(triangle, 0, getReference(triangle, 0));
          writeReplacement(triangle, 1, getReference(first, 0));
          writeReplacement(triangle, 2, getReference(second, 0));
          writeChildren(triangle, triangle, first, second);
        } else {
          const a = getVertex(triangle, code);
          const b = getVertex(triangle, (code + 1) % 3);
          const c = getVertex(triangle, (code + 2) % 3);
          const d = getVertex(neighbor, (neighborEdge + 2) % 3);
          const nextOld = getAdjacency(triangle, code + 1);
          const previousOld = getAdjacency(triangle, code + 2);
          const neighborNextOld = getAdjacency(neighbor, neighborEdge + 1);
          const neighborPreviousOld = getAdjacency(neighbor, neighborEdge + 2);
          writeTriangle(triangle, a, site, c, phase);
          writeTriangle(first, site, b, c, phase);
          writeTriangle(neighbor, b, site, d, phase);
          writeTriangle(second, site, a, d, phase);
          writeAdjacency(
            triangle,
            FRESH | getReference(second, 0),
            FRESH | getReference(first, 2),
            previousOld
          );
          writeAdjacency(
            first,
            FRESH | getReference(neighbor, 0),
            nextOld,
            FRESH | getReference(triangle, 1)
          );
          writeAdjacency(
            neighbor,
            FRESH | getReference(first, 0),
            FRESH | getReference(second, 2),
            neighborPreviousOld
          );
          writeAdjacency(
            second,
            FRESH | getReference(triangle, 0),
            neighborNextOld,
            FRESH | getReference(neighbor, 1)
          );
          writeReplacement(triangle, code + 1, getReference(first, 1));
          writeReplacement(triangle, code + 2, getReference(triangle, 2));
          writeReplacement(neighbor, neighborEdge + 1, getReference(second, 1));
          writeReplacement(neighbor, neighborEdge + 2, getReference(neighbor, 2));
          writeChildren(triangle, triangle, first, NONE);
          writeChildren(neighbor, neighbor, second, NONE);
        }
        pointLocations[site] = NONE;
        state[2]--;
        state[5]++;
      }
      repairAdjacency(0);
      relocate(0);
      if (options.validate) {
        validateTopology(`step ${step} insertion`);
      }
    }
    if (active) {
      // flip-evaluate
      for (let index = 0; index < liveCount(); index++) {
        for (let edge = 0; edge < 3; edge++) {
          const reference = triangleAdjacency[4 * index + edge];
          if (reference === NONE || Math.floor(reference / 3) <= index) {
            continue;
          }
          const neighbor = Math.floor(reference / 3);
          const a = getVertex(index, edge);
          const b = getVertex(index, (edge + 1) % 3);
          const c = getVertex(index, (edge + 2) % 3);
          const d = getVertex(neighbor, ((reference % 3) + 2) % 3);
          if (isFlipRequired(a, b, c, d)) {
            const key = getFlipPriority(3 * index + edge);
            votes[index] = Math.min(votes[index], key);
            votes[neighbor] = Math.min(votes[neighbor], key);
            state[4]++;
          }
        }
      }
      result.illegalHistory.push(state[4]);
      // flip-apply
      const indices: number[] = [];
      for (let index = 0; index < liveCount(); index++) {
        indices.push(index);
      }
      for (const index of orderWinners(indices)) {
        const key = votes[index];
        let edge = 3;
        for (let candidate = 0; candidate < 3; candidate++) {
          if (getFlipPriority(3 * index + candidate) === key) {
            edge = candidate;
          }
        }
        if (edge === 3) {
          continue;
        }
        const reference = getAdjacency(index, edge);
        const neighbor = Math.floor(reference / 3);
        const neighborEdge = reference % 3;
        if (votes[neighbor] !== key) {
          continue;
        }
        const phase = state[3] + 1;
        const a = getVertex(index, edge);
        const b = getVertex(index, (edge + 1) % 3);
        const c = getVertex(index, (edge + 2) % 3);
        const d = getVertex(neighbor, (neighborEdge + 2) % 3);
        const nextOld = getAdjacency(index, edge + 1);
        const previousOld = getAdjacency(index, edge + 2);
        const neighborNextOld = getAdjacency(neighbor, neighborEdge + 1);
        const neighborPreviousOld = getAdjacency(neighbor, neighborEdge + 2);
        writeTriangle(index, c, a, d, phase);
        writeTriangle(neighbor, d, b, c, phase);
        writeAdjacency(index, previousOld, neighborNextOld, FRESH | getReference(neighbor, 2));
        writeAdjacency(neighbor, neighborPreviousOld, nextOld, FRESH | getReference(index, 2));
        writeReplacement(index, edge + 2, getReference(index, 0));
        writeReplacement(neighbor, neighborEdge + 1, getReference(index, 1));
        writeReplacement(neighbor, neighborEdge + 2, getReference(neighbor, 0));
        writeReplacement(index, edge + 1, getReference(neighbor, 1));
        writeChildren(index, index, neighbor, NONE);
        writeChildren(neighbor, index, neighbor, NONE);
        result.flips++;
      }
      repairAdjacency(1);
      relocate(1);
      if (options.validate) {
        validateTopology(`step ${step} flip`);
      }
    }
  }

  // --- finalize ---
  if (state[1] === 0 && (state[2] > 0 || state[4] > 0)) {
    state[1] = 1;
  }
  result.invalidCount += state[1];
  result.failed = state[1] !== 0;
  result.slotCount = liveCount();

  // --- classify-output + stable lexicographic sort ---
  const accepted: {first: number; second: number; triangle: number[]}[] = [];
  if (state[1] === 0) {
    for (let index = 0; index < liveCount(); index++) {
      const a = getVertex(index, 0);
      const b = getVertex(index, 1);
      const c = getVertex(index, 2);
      let canonical = [a, b, c];
      if (b < a && b < c) {
        canonical = [b, c, a];
      } else if (c < a && c < b) {
        canonical = [c, a, b];
      }
      if (canonical.every(row => row < pointCount)) {
        accepted.push({first: canonical[0], second: canonical[1], triangle: canonical});
      }
    }
  }
  accepted.sort((left, right) => left.first - right.first || left.second - right.second);
  result.requiredCount = accepted.length;
  result.triangles = accepted.map(entry => entry.triangle);
  return result;
}
