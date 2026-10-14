// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * CPU model of the monotone path of `GPUPolygonTriangulation` (test only, never a fallback).
 *
 * It runs steps 1 to 5 of the GPU design as the same data-parallel passes over flat arrays, so a
 * defect in the algorithm (as opposed to a WGSL slip) shows up here on adversarial polygons:
 *
 * 1. clean the rings and rank the vertices in sweep order (y descending, x ascending, row last),
 * 2. classify every vertex (start, end, split, merge, regular) with the exact orientation,
 * 3. find the edge directly left of every split, merge and right-boundary vertex,
 * 4. derive the helper diagonals from the sorted (edge, rank) membership list,
 * 5. build the next-pointer of every half-edge from the angular order at each vertex, label the
 *    pieces by pointer jumping and triangulate each y-monotone piece with the stack algorithm.
 *
 * Positions are doubles here (the tests use f32-representable values); all orientation signs are
 * exact (a floating-point filter, then BigInt arithmetic on the exact binary expansions).
 */

export type ModelPoint = readonly [number, number];

/** Vertex classes of the sweep. */
export const MODEL_TYPE = {
  start: 0,
  end: 1,
  split: 2,
  merge: 3,
  regularLeft: 4,
  regularRight: 5
} as const;

const NONE = -1;
const DIAGONAL_SLOTS = 4;

const float64 = new Float64Array(1);
const uint64 = new BigUint64Array(float64.buffer);

/** Decomposes a finite double into `[mantissa, exponent]` with `value = mantissa * 2^exponent`. */
function decompose(value: number): [bigint, number] {
  if (value === 0) {
    return [0n, 0];
  }
  float64[0] = value;
  const bits = uint64[0];
  const negative = bits >> 63n === 1n;
  const exponentBits = Number((bits >> 52n) & 0x7ffn);
  const fraction = bits & 0xfffffffffffffn;
  const mantissa = exponentBits === 0 ? fraction : fraction | (1n << 52n);
  const exponent = (exponentBits === 0 ? 1 : exponentBits) - 1075;
  return [negative ? -mantissa : mantissa, exponent];
}

/** Sign of `cross(b - a, c - a)`, exact for finite doubles. */
export function orientExactSign(a: ModelPoint, b: ModelPoint, c: ModelPoint): number {
  const left = (b[0] - a[0]) * (c[1] - a[1]);
  const right = (b[1] - a[1]) * (c[0] - a[0]);
  const determinant = left - right;
  const bound = 3.3306690738754716e-16 * (Math.abs(left) + Math.abs(right));
  if (determinant > bound) return 1;
  if (-determinant > bound) return -1;
  const values = [a[0], a[1], b[0], b[1], c[0], c[1]].map(decompose);
  const minimumExponent = Math.min(...values.map(([, exponent]) => exponent));
  const scaled = values.map(
    ([mantissa, exponent]) => mantissa << BigInt(exponent - minimumExponent)
  );
  const [ax, ay, bx, by, cx, cy] = scaled;
  const exact = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
  return exact > 0n ? 1 : exact < 0n ? -1 : 0;
}

export type MonotoneModelDiagnostics = {
  /** Cleaned vertex count of the alive rings. */
  cleanVertexCount: number;
  /** Number of helper diagonals. */
  diagonalCount: number;
  /** Number of monotone pieces. */
  pieceCount: number;
  /** Vertex class counts, indexed by {@link MODEL_TYPE}. */
  typeCounts: number[];
  /** Pointer-jumping rounds that were run. */
  jumpRounds: number;
  /** Largest number of diagonals at one vertex. */
  maximumDiagonalDegree: number;
};

export type MonotoneModelResult = {
  /** Whether the monotone path would report `valid = 1`. */
  valid: boolean;
  /** Reason of the first failure, when invalid. */
  failure?: string;
  /** Triangles as row indices into the flattened rings (ring order), counter-clockwise. */
  triangles: [number, number, number][];
  /** Expected triangle count `cleanVertices + 2 * aliveHoles - 2`. */
  expectedTriangleCount: number;
  diagnostics: MonotoneModelDiagnostics;
};

type Chain = 'left' | 'right';

/**
 * Triangulates one polygon (`rings[0]` is the shell, any winding) with the monotone method.
 *
 * Invalid input (self-intersection, overlapping holes, zero-length diagonals) returns
 * `valid: false` when the cheap consistency checks of the GPU path catch it; as on the GPU, some
 * invalid inputs are not detected.
 */
export function triangulateMonotoneModel(
  rings: readonly (readonly ModelPoint[])[],
  options: {rayShooting?: 'slabs' | 'brute'} = {}
): MonotoneModelResult {
  const diagnostics: MonotoneModelDiagnostics = {
    cleanVertexCount: 0,
    diagonalCount: 0,
    pieceCount: 0,
    typeCounts: [0, 0, 0, 0, 0, 0],
    jumpRounds: 0,
    maximumDiagonalDegree: 0
  };
  const fail = (failure: string): MonotoneModelResult => ({
    valid: false,
    failure,
    triangles: [],
    expectedTriangleCount: 0,
    diagnostics
  });

  // ---- Step 1a: rows, dedupe flags, compaction (kernels index / ring / compact). ----
  const rowPositions: ModelPoint[] = [];
  const rowRing: number[] = [];
  const ringBegins: number[] = [];
  for (const [ringIndex, ring] of rings.entries()) {
    ringBegins.push(rowPositions.length);
    for (const point of ring) {
      rowPositions.push(point);
      rowRing.push(ringIndex);
    }
  }
  ringBegins.push(rowPositions.length);
  const rowCount = rowPositions.length;
  const samePoint = (first: ModelPoint, second: ModelPoint) =>
    first[0] === second[0] && first[1] === second[1];
  const keep: number[] = [];
  for (let row = 0; row < rowCount; row++) {
    const ring = rowRing[row];
    const begin = ringBegins[ring];
    const end = ringBegins[ring + 1];
    const previous = row === begin ? end - 1 : row - 1;
    keep.push(samePoint(rowPositions[row], rowPositions[previous]) ? 0 : 1);
  }
  const scan: number[] = [0];
  for (let row = 0; row < rowCount; row++) {
    scan.push(scan[row] + keep[row]);
  }
  // Ring state: clean range, alive (3 or more clean vertices, and a live shell).
  const ringCleanBegin = rings.map((_, ring) => scan[ringBegins[ring]]);
  const ringCleanCount = rings.map(
    (_, ring) => scan[ringBegins[ring + 1]] - scan[ringBegins[ring]]
  );
  if (rings.length === 0 || ringCleanCount[0] < 3) {
    return fail('shell has fewer than three distinct vertices');
  }
  const ringAlive = ringCleanCount.map(count => count >= 3);
  const vertexCount = scan[rowCount];
  const cleanRow: number[] = new Array(vertexCount).fill(NONE);
  const cleanRing: number[] = new Array(vertexCount).fill(NONE);
  for (let row = 0; row < rowCount; row++) {
    if (keep[row] === 1) {
      cleanRow[scan[row]] = row;
      cleanRing[scan[row]] = rowRing[row];
    }
  }
  const alive = (vertex: number) => ringAlive[cleanRing[vertex]];
  const position = (vertex: number): ModelPoint => rowPositions[cleanRow[vertex]];
  let aliveCount = 0;
  let aliveHoles = 0;
  for (let ring = 0; ring < rings.length; ring++) {
    if (ringAlive[ring]) {
      aliveCount += ringCleanCount[ring];
      aliveHoles += ring > 0 ? 1 : 0;
    }
  }
  diagnostics.cleanVertexCount = aliveCount;
  const expectedTriangleCount = aliveCount + 2 * aliveHoles - 2;
  const rawNext = (vertex: number) => {
    const ring = cleanRing[vertex];
    const begin = ringCleanBegin[ring];
    const count = ringCleanCount[ring];
    return begin + ((vertex - begin + 1) % count);
  };
  const rawPrevious = (vertex: number) => {
    const ring = cleanRing[vertex];
    const begin = ringCleanBegin[ring];
    const count = ringCleanCount[ring];
    return begin + ((vertex - begin + count - 1) % count);
  };

  // ---- Step 1b: sweep order. y descending, x ascending, compact index last. ----
  const order: number[] = [];
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    if (alive(vertex)) order.push(vertex);
  }
  order.sort((first, second) => {
    const a = position(first);
    const b = position(second);
    if (a[1] !== b[1]) return b[1] - a[1];
    if (a[0] !== b[0]) return a[0] - b[0];
    return first - second;
  });
  const rank: number[] = new Array(vertexCount).fill(NONE);
  order.forEach((vertex, index) => {
    rank[vertex] = index;
  });

  // ---- Ring orientation at the top vertex of each ring (kernels ring-top, ring-flip). ----
  const ringTop: number[] = rings.map(() => Number.MAX_SAFE_INTEGER);
  for (const vertex of order) {
    ringTop[cleanRing[vertex]] = Math.min(ringTop[cleanRing[vertex]], rank[vertex]);
  }
  const ringFlip: boolean[] = rings.map(() => false);
  for (let ring = 0; ring < rings.length; ring++) {
    if (!ringAlive[ring]) continue;
    const top = order[ringTop[ring]];
    const sign = orientExactSign(position(rawPrevious(top)), position(top), position(rawNext(top)));
    if (sign === 0) return fail('zero-area ring');
    // Shell counter-clockwise, holes clockwise: the interior is on the left of travel.
    ringFlip[ring] = ring === 0 ? sign < 0 : sign > 0;
  }
  const next = (vertex: number) =>
    ringFlip[cleanRing[vertex]] ? rawPrevious(vertex) : rawNext(vertex);
  const previous = (vertex: number) =>
    ringFlip[cleanRing[vertex]] ? rawNext(vertex) : rawPrevious(vertex);

  // ---- Step 1c: classification. ----
  const type: number[] = new Array(vertexCount).fill(NONE);
  for (const vertex of order) {
    const before = previous(vertex);
    const after = next(vertex);
    const previousBelow = rank[before] > rank[vertex];
    const nextBelow = rank[after] > rank[vertex];
    if (previousBelow === nextBelow) {
      const sign = orientExactSign(position(before), position(vertex), position(after));
      if (sign === 0) return fail('degenerate spike');
      if (previousBelow) {
        type[vertex] = sign > 0 ? MODEL_TYPE.start : MODEL_TYPE.split;
      } else {
        type[vertex] = sign > 0 ? MODEL_TYPE.end : MODEL_TYPE.merge;
      }
    } else {
      type[vertex] = previousBelow ? MODEL_TYPE.regularRight : MODEL_TYPE.regularLeft;
    }
    diagnostics.typeCounts[type[vertex]]++;
  }

  // ---- Step 2: edge directly left of every split, merge and right-boundary vertex. ----
  // The edge is named by its logical start vertex (the upper end). On the GPU the candidates come
  // from the polygon edge slabs; here every down edge is a candidate, which is the superset the
  // slab walk is specified to produce.
  const downEdges: number[] = [];
  for (const vertex of order) {
    if (rank[next(vertex)] > rank[vertex]) downEdges.push(vertex);
  }
  /** True when edge `second` lies to the right of edge `first` at the sweep position of the query. */
  const isRightOf = (second: number, first: number): number => {
    const upper2 = second;
    const lower2 = next(second);
    const upper1 = first;
    const lower1 = next(first);
    let side: number;
    if (upper2 === upper1) {
      side = orientExactSign(position(upper1), position(lower1), position(lower2));
    } else if (rank[upper2] > rank[upper1]) {
      side = orientExactSign(position(upper1), position(lower1), position(upper2));
      if (side === 0) side = orientExactSign(position(upper1), position(lower1), position(lower2));
    } else {
      side = -orientExactSign(position(upper2), position(lower2), position(upper1));
      if (side === 0) side = -orientExactSign(position(upper2), position(lower2), position(lower1));
    }
    if (side !== 0) return side;
    // Touching edges: the larger edge name wins, a deterministic tie-break.
    return second > first ? 1 : -1;
  };
  // Candidate edges come from a y-slab index like the shared GPU edge-slab index: about one slab
  // per four edges, an edge listed in every slab between its two ends, a query visiting the slab
  // of its height (`brute` visits every edge instead).
  let minimumY = Infinity;
  let maximumY = -Infinity;
  for (const vertex of order) {
    minimumY = Math.min(minimumY, position(vertex)[1]);
    maximumY = Math.max(maximumY, position(vertex)[1]);
  }
  const slabCount =
    options.rayShooting === 'brute' || order.length < 8 || !(maximumY > minimumY)
      ? 1
      : Math.min(Math.floor(order.length / 4), 4096);
  const slabScale = slabCount / (maximumY - minimumY);
  const slabOf = (y: number) =>
    slabCount <= 1
      ? 0
      : Math.min(Math.max(Math.floor((y - minimumY) * slabScale), 0), slabCount - 1);
  const slabs: number[][] = Array.from({length: slabCount}, () => []);
  for (const edge of downEdges) {
    const first = slabOf(position(next(edge))[1]);
    const last = slabOf(position(edge)[1]);
    for (let slab = Math.min(first, last); slab <= Math.max(first, last); slab++)
      slabs[slab].push(edge);
  }
  const leftEdge: number[] = new Array(vertexCount).fill(NONE);
  for (const vertex of order) {
    const kind = type[vertex];
    if (
      kind !== MODEL_TYPE.split &&
      kind !== MODEL_TYPE.merge &&
      kind !== MODEL_TYPE.regularRight
    ) {
      continue;
    }
    let best = NONE;
    for (const edge of slabs[slabOf(position(vertex)[1])]) {
      const lower = next(edge);
      if (!(rank[edge] < rank[vertex] && rank[vertex] < rank[lower])) continue;
      if (orientExactSign(position(edge), position(lower), position(vertex)) <= 0) continue;
      if (best === NONE || isRightOf(edge, best) > 0) best = edge;
    }
    if (best === NONE) return fail('no edge to the left');
    leftEdge[vertex] = best;
  }

  // ---- Step 3: members sorted by (edge, rank), consecutive pairs give the diagonals. ----
  // Three member slots per vertex in rank order: 0 left-edge member, 1 lower end of the incoming
  // down edge, 2 upper end of the outgoing down edge.
  type Member = {group: number; vertex: number; slot: number; key: number};
  const members: Member[] = [];
  order.forEach((vertex, vertexRank) => {
    const kind = type[vertex];
    const hasLeft =
      kind === MODEL_TYPE.split || kind === MODEL_TYPE.merge || kind === MODEL_TYPE.regularRight;
    const incoming = previous(vertex);
    members.push({
      group: hasLeft ? leftEdge[vertex] : NONE,
      vertex,
      slot: 0,
      key: 3 * vertexRank
    });
    members.push({
      group: rank[incoming] < rank[vertex] ? incoming : NONE,
      vertex,
      slot: 1,
      key: 3 * vertexRank + 1
    });
    members.push({
      group: rank[next(vertex)] > rank[vertex] ? vertex : NONE,
      vertex,
      slot: 2,
      key: 3 * vertexRank + 2
    });
  });
  const sortedMembers = members
    .filter(member => member.group !== NONE)
    .sort((first, second) => first.group - second.group || first.key - second.key);

  // vertexDiagonal[4 * v + s] = (other << 2 | reverse slot) or NONE.
  const vertexDiagonal: number[] = new Array(DIAGONAL_SLOTS * vertexCount).fill(NONE);
  for (let index = 1; index < sortedMembers.length; index++) {
    const later = sortedMembers[index];
    const earlier = sortedMembers[index - 1];
    if (later.group !== earlier.group) continue;
    const laterVertex = later.vertex;
    const earlierVertex = earlier.vertex;
    if (!(type[laterVertex] === MODEL_TYPE.split || type[earlierVertex] === MODEL_TYPE.merge)) {
      continue;
    }
    const laterSlot = later.slot === 0 ? 0 : 1;
    if (earlier.slot === 1) return fail('lower end has a successor');
    const earlierSlot = earlier.slot === 0 ? 2 : 3;
    if (later.slot === 2) return fail('upper end has a predecessor');
    if (
      vertexDiagonal[DIAGONAL_SLOTS * laterVertex + laterSlot] !== NONE ||
      vertexDiagonal[DIAGONAL_SLOTS * earlierVertex + earlierSlot] !== NONE
    ) {
      return fail('diagonal slot taken');
    }
    vertexDiagonal[DIAGONAL_SLOTS * laterVertex + laterSlot] = (earlierVertex << 2) | earlierSlot;
    vertexDiagonal[DIAGONAL_SLOTS * earlierVertex + earlierSlot] = (laterVertex << 2) | laterSlot;
    diagnostics.diagonalCount++;
  }

  // ---- Step 4: next pointers from the angular order at each vertex. ----
  // Half-edge ids: ring half-edge of v (v -> next(v)) is v, diagonal slot s of v is
  // vertexCount + 4 * v + s.
  const halfEdgeCount = vertexCount + DIAGONAL_SLOTS * vertexCount;
  const nextHalfEdge: number[] = new Array(halfEdgeCount).fill(NONE);
  const isValidHalfEdge = (id: number) =>
    id < vertexCount ? alive(id) : vertexDiagonal[id - vertexCount] !== NONE;
  /** Angular category of `target` seen from `center`, counted counter-clockwise from `base`. */
  const sameDirection = (center: ModelPoint, first: ModelPoint, second: ModelPoint) =>
    Math.sign(first[0] - center[0]) === Math.sign(second[0] - center[0]) &&
    Math.sign(first[1] - center[1]) === Math.sign(second[1] - center[1]);
  const category = (center: ModelPoint, base: ModelPoint, target: ModelPoint): number => {
    const sign = orientExactSign(center, base, target);
    if (sign > 0) return 0;
    if (sign < 0) return 2;
    return sameDirection(center, base, target) ? -1 : 1;
  };
  for (const vertex of order) {
    const center = position(vertex);
    const base = position(next(vertex));
    const back = position(previous(vertex));
    const entries: {slot: number; point: ModelPoint}[] = [];
    for (let slot = 0; slot < DIAGONAL_SLOTS; slot++) {
      const value = vertexDiagonal[DIAGONAL_SLOTS * vertex + slot];
      if (value === NONE) continue;
      entries.push({slot, point: position(value >> 2)});
    }
    diagnostics.maximumDiagonalDegree = Math.max(diagnostics.maximumDiagonalDegree, entries.length);
    let invalid = false;
    const lessThan = (first: ModelPoint, second: ModelPoint): boolean => {
      const firstCategory = category(center, base, first);
      const secondCategory = category(center, base, second);
      if (firstCategory !== secondCategory) return firstCategory < secondCategory;
      if (firstCategory === 1 || firstCategory === -1) {
        invalid = true;
        return false;
      }
      return orientExactSign(center, first, second) > 0;
    };
    // Insertion sort, the same bounded private-array sort as the kernel.
    for (let index = 1; index < entries.length; index++) {
      const held = entries[index];
      let hole = index;
      while (hole > 0 && lessThan(held.point, entries[hole - 1].point)) {
        entries[hole] = entries[hole - 1];
        hole--;
      }
      entries[hole] = held;
    }
    // Every diagonal lies strictly inside the wedge from the ring-out edge to the ring-back edge.
    for (const entry of entries) {
      if (samePoint(entry.point, center)) invalid = true;
      if (category(center, base, entry.point) === -1) invalid = true;
      if (!lessThan(entry.point, back)) invalid = true;
    }
    for (let index = 1; index < entries.length; index++) {
      if (!lessThan(entries[index - 1].point, entries[index].point)) invalid = true;
    }
    if (invalid) return fail('diagonal order');
    // L = [ring out, d_1 .. d_k]; the half-edge before reverse(h) in L is next(h).
    const list = [
      vertex,
      ...entries.map(entry => vertexCount + DIAGONAL_SLOTS * vertex + entry.slot)
    ];
    nextHalfEdge[previous(vertex)] = list[list.length - 1];
    entries.forEach((entry, index) => {
      const value = vertexDiagonal[DIAGONAL_SLOTS * vertex + entry.slot];
      const other = value >> 2;
      const otherSlot = value & 3;
      nextHalfEdge[vertexCount + DIAGONAL_SLOTS * other + otherSlot] = list[index];
    });
  }

  // ---- Step 5: label the cycles by double-buffered pointer jumping. ----
  const aliveHalfEdges: number[] = [];
  for (let id = 0; id < halfEdgeCount; id++) {
    if (isValidHalfEdge(id)) aliveHalfEdges.push(id);
  }
  for (const id of aliveHalfEdges) {
    if (nextHalfEdge[id] === NONE || !isValidHalfEdge(nextHalfEdge[id])) {
      return fail('dangling next pointer');
    }
  }
  let pointer = nextHalfEdge.slice();
  let label = Array.from({length: halfEdgeCount}, (_, id) => id);
  const polygonBound = 3 * aliveCount + 3;
  const rounds = 2 * Math.ceil(Math.ceil(Math.log2(Math.max(polygonBound, 2))) / 2);
  diagnostics.jumpRounds = rounds;
  for (let round = 0; round < rounds; round++) {
    const nextPointer = pointer.slice();
    const nextLabel = label.slice();
    for (const id of aliveHalfEdges) {
      const target = pointer[id];
      nextLabel[id] = Math.min(label[id], label[target]);
      nextPointer[id] = pointer[target];
    }
    pointer = nextPointer;
    label = nextLabel;
  }
  const representatives = aliveHalfEdges.filter(id => label[id] === id);
  diagnostics.pieceCount = representatives.length;

  // ---- Pieces: measure, then triangulate each monotone piece. ----
  const tail = (id: number) =>
    id < vertexCount ? id : Math.floor((id - vertexCount) / DIAGONAL_SLOTS);
  const pieces: {head: number; sequence: number[]; topIndex: number; bottomIndex: number}[] = [];
  for (const head of representatives) {
    const sequenceFromHead: number[] = [];
    let id = head;
    do {
      sequenceFromHead.push(tail(id));
      id = nextHalfEdge[id];
      if (sequenceFromHead.length > polygonBound) return fail('cycle does not close');
    } while (id !== head);
    if (sequenceFromHead.length < 3) return fail('piece with fewer than three vertices');
    let topIndex = 0;
    let bottomIndex = 0;
    sequenceFromHead.forEach((vertex, index) => {
      if (rank[vertex] < rank[sequenceFromHead[topIndex]]) topIndex = index;
      if (rank[vertex] > rank[sequenceFromHead[bottomIndex]]) bottomIndex = index;
    });
    // Rotate so the sequence starts at the top vertex.
    const sequence = sequenceFromHead.slice(topIndex).concat(sequenceFromHead.slice(0, topIndex));
    const bottomRotated = (bottomIndex - topIndex + sequence.length) % sequence.length;
    pieces.push({head, sequence, topIndex: 0, bottomIndex: bottomRotated});
  }
  const triangles: [number, number, number][] = [];
  for (const piece of pieces) {
    const {sequence, bottomIndex} = piece;
    const count = sequence.length;
    // Both chains must be strictly monotone in rank.
    for (let index = 0; index < bottomIndex; index++) {
      if (!(rank[sequence[index]] < rank[sequence[index + 1]]))
        return fail('left chain not monotone');
    }
    let previousRank = rank[sequence[0]];
    for (let index = count - 1; index >= bottomIndex; index--) {
      if (!(rank[sequence[index]] > previousRank)) return fail('right chain not monotone');
      previousRank = rank[sequence[index]];
    }
    // Merge the chains by rank.
    const merged: {vertex: number; chain: Chain}[] = [{vertex: sequence[0], chain: 'left'}];
    let leftIndex = 1;
    let rightIndex = count - 1;
    while (leftIndex <= bottomIndex || rightIndex > bottomIndex) {
      const leftAvailable = leftIndex <= bottomIndex;
      const rightAvailable = rightIndex > bottomIndex;
      if (
        leftAvailable &&
        (!rightAvailable || rank[sequence[leftIndex]] < rank[sequence[rightIndex]])
      ) {
        merged.push({vertex: sequence[leftIndex++], chain: 'left'});
      } else {
        merged.push({vertex: sequence[rightIndex--], chain: 'right'});
      }
    }
    const stack: {vertex: number; chain: Chain}[] = [merged[0], merged[1]];
    const emit = (a: number, b: number, c: number) =>
      triangles.push([cleanRow[a], cleanRow[b], cleanRow[c]]);
    for (let index = 2; index < merged.length - 1; index++) {
      const current = merged[index];
      const top = stack[stack.length - 1];
      if (current.chain !== top.chain) {
        // Different chains: fan from the current vertex over the whole stack.
        for (let item = 0; item + 1 < stack.length; item++) {
          const upper = stack[item].vertex;
          const lower = stack[item + 1].vertex;
          if (top.chain === 'left') emit(upper, lower, current.vertex);
          else emit(lower, upper, current.vertex);
        }
        const previousTop = stack[stack.length - 1];
        stack.length = 0;
        stack.push(previousTop, current);
      } else {
        let last = stack.pop()!;
        while (stack.length > 0) {
          const upper = stack[stack.length - 1];
          const sign =
            current.chain === 'left'
              ? orientExactSign(
                  position(upper.vertex),
                  position(last.vertex),
                  position(current.vertex)
                )
              : orientExactSign(
                  position(current.vertex),
                  position(last.vertex),
                  position(upper.vertex)
                );
          if (sign < 0) break;
          if (current.chain === 'left') emit(upper.vertex, last.vertex, current.vertex);
          else emit(current.vertex, last.vertex, upper.vertex);
          last = stack.pop()!;
        }
        stack.push(last, current);
      }
    }
    const bottom = merged[merged.length - 1];
    const stackChain = stack[stack.length - 1].chain;
    for (let item = 0; item + 1 < stack.length; item++) {
      const upper = stack[item].vertex;
      const lower = stack[item + 1].vertex;
      if (stackChain === 'left') emit(upper, lower, bottom.vertex);
      else emit(lower, upper, bottom.vertex);
    }
  }
  if (triangles.length !== expectedTriangleCount) {
    return {
      valid: false,
      failure: `triangle count ${triangles.length} differs from ${expectedTriangleCount}`,
      triangles,
      expectedTriangleCount,
      diagnostics
    };
  }
  return {valid: true, triangles, expectedTriangleCount, diagnostics};
}
