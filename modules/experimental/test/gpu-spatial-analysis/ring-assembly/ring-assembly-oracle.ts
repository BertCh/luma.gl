// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** One directed segment `[x0, y0, x1, y1]`. */
export type OracleSegment = [number, number, number, number];

/** Options of {@link assembleRingsOnCPU}. */
export type OracleRingOptions = {
  tolerance: number;
  interiorSide: 'left' | 'right';
  normalizeWinding?: boolean;
  geographic?: boolean;
  groups?: number[];
  /** Number of valid leading segments. Defaults to all. */
  count?: number;
  /** Mirrors `splitTouchingRings` (default true). */
  splitTouchingRings?: boolean;
};

/** One assembled ring. */
export type OracleRing = {
  /** Closed vertex list. */
  vertices: [number, number][];
  /** Input segment index per vertex except the closing vertex, in output order. */
  segments: number[];
  area: number;
  isHole: boolean;
  /** Index of the innermost enclosing shell (a shell's own index), or null. */
  shell: number | null;
  group: number;
};

/** Result of {@link assembleRingsOnCPU}. */
export type OracleRingResult = {
  rings: OracleRing[];
  /** Ring index per segment, or -1 when open. */
  segmentRing: number[];
  openSegments: number;
  touchingSegments: number;
};

/** Brute-force CPU reference for `GPUSegmentRingAssembly` (same matching, turn and tie rules). */
export function assembleRingsOnCPU(
  segmentList: readonly OracleSegment[],
  options: OracleRingOptions
): OracleRingResult {
  const valid = Math.min(options.count ?? segmentList.length, segmentList.length);
  const side = options.interiorSide === 'left' ? 1 : -1;
  const geographic = options.geographic ?? true;
  const groupOf = (row: number) => options.groups?.[row] ?? 0;
  const next: number[] = new Array(segmentList.length).fill(-1);
  const alternative: number[] = new Array(segmentList.length).fill(-1);
  let touchingSegments = 0;
  for (let index = 0; index < valid; index++) {
    const [ox, oy, vx, vy] = segmentList[index];
    const scale = geographic ? Math.max(Math.cos((vy * Math.PI) / 180), 0.01) : 1;
    const incomingX = (vx - ox) * scale;
    const incomingY = vy - oy;
    let best = -1;
    let bestTurn = 0;
    let matches = 0;
    const found: number[] = [];
    for (let candidate = 0; candidate < valid; candidate++) {
      if (candidate === index || groupOf(candidate) !== groupOf(index)) {
        continue;
      }
      const [cx, cy, ex, ey] = segmentList[candidate];
      if (Math.abs(cx - vx) > options.tolerance || Math.abs(cy - vy) > options.tolerance) {
        continue;
      }
      matches++;
      found.push(candidate);
      const outgoingX = (ex - cx) * scale;
      const outgoingY = ey - cy;
      const turn =
        side *
        getTurnKey(
          incomingX * outgoingY - incomingY * outgoingX,
          incomingX * outgoingX + incomingY * outgoingY
        );
      if (best < 0 || turn > bestTurn || (turn === bestTurn && candidate < best)) {
        best = candidate;
        bestTurn = turn;
      }
    }
    next[index] = best;
    if (matches === 2) {
      alternative[index] = found.find(candidate => candidate !== best) ?? -1;
    }
    if (matches > 1) {
      touchingSegments++;
    }
  }
  // Injective: lowest-index claimant keeps the continuation.
  const claim = new Map<number, number>();
  for (let index = 0; index < valid; index++) {
    if (next[index] >= 0 && !claim.has(next[index])) {
      claim.set(next[index], index);
    }
  }
  for (let index = 0; index < valid; index++) {
    if (next[index] >= 0 && claim.get(next[index]) !== index) {
      next[index] = -1;
    }
  }
  const flip = Boolean(options.normalizeWinding) && options.interiorSide === 'right';
  const shellSign = options.normalizeWinding || options.interiorSide === 'left' ? 1 : -1;
  const trace = (successor: number[]) => {
    const visited = new Array(segmentList.length).fill(false);
    const chains: number[][] = [];
    for (let start = 0; start < valid; start++) {
      if (visited[start]) {
        continue;
      }
      const chain = [start];
      visited[start] = true;
      let current = successor[start];
      let closed = false;
      while (current >= 0) {
        if (current === start) {
          closed = true;
          break;
        }
        if (visited[current]) {
          break;
        }
        visited[current] = true;
        chain.push(current);
        current = successor[current];
      }
      if (closed) {
        chains.push(chain);
      }
    }
    return chains;
  };
  let chains = trace(next);
  if (options.splitTouchingRings !== false) {
    const ringOf = new Map<number, number>();
    chains.forEach((chain, ring) => chain.forEach(segment => ringOf.set(segment, ring)));
    const swapped = [...next];
    for (let index = 0; index < valid; index++) {
      const chosen = next[index];
      const other = alternative[index];
      if (chosen < 0 || other < 0) {
        continue;
      }
      const partner = claim.get(other);
      if (
        partner !== undefined &&
        partner !== index &&
        alternative[partner] === chosen &&
        next[partner] === other &&
        ringOf.has(index) &&
        ringOf.get(index) === ringOf.get(partner)
      ) {
        swapped[index] = other;
      }
    }
    chains = trace(swapped);
  }
  const segmentRing: number[] = new Array(segmentList.length).fill(-1);
  const rings: OracleRing[] = [];
  for (const chain of chains) {
    const ordered = flip ? [chain[0], ...chain.slice(1).reverse()] : chain;
    const vertices: [number, number][] = ordered.map(
      segment => [segmentList[segment][0], segmentList[segment][1]] as [number, number]
    );
    vertices.push(vertices[0]);
    ordered.forEach(segment => {
      segmentRing[segment] = rings.length;
    });
    rings.push({
      vertices,
      segments: ordered,
      area: getSignedArea(vertices),
      isHole: false,
      shell: null,
      group: groupOf(chain[0])
    });
  }
  rings.forEach(ring => {
    ring.isHole = ring.area * shellSign < 0;
  });
  rings.forEach((ring, index) => {
    if (!ring.isHole) {
      ring.shell = index;
      return;
    }
    const probe: [number, number] = [
      0.5 * (ring.vertices[0][0] + ring.vertices[1][0]),
      0.5 * (ring.vertices[0][1] + ring.vertices[1][1])
    ];
    let best = -1;
    for (let candidateIndex = 0; candidateIndex < rings.length; candidateIndex++) {
      const candidate = rings[candidateIndex];
      if (candidate.isHole || candidate.group !== ring.group) {
        continue;
      }
      if (best >= 0 && Math.abs(candidate.area) >= Math.abs(rings[best].area)) {
        continue;
      }
      if (containsPoint(candidate.vertices, probe)) {
        best = candidateIndex;
      }
    }
    ring.shell = best >= 0 ? best : null;
  });
  const closedSegments = segmentRing.filter(ring => ring >= 0).length;
  return {rings, segmentRing, openSegments: valid - closedSegments, touchingSegments};
}

/** Shoelace area relative to the first vertex (counter-clockwise positive). */
export function getSignedArea(vertices: readonly [number, number][]): number {
  let sum = 0;
  const [ox, oy] = vertices[0];
  for (let index = 0; index + 1 < vertices.length; index++) {
    const ax = vertices[index][0] - ox;
    const ay = vertices[index][1] - oy;
    const bx = vertices[index + 1][0] - ox;
    const by = vertices[index + 1][1] - oy;
    sum += ax * by - bx * ay;
  }
  return 0.5 * sum;
}

/** Even-odd ray casting with the half-open crossing rule the GPU kernel uses. */
export function containsPoint(vertices: readonly [number, number][], point: [number, number]) {
  let inside = false;
  for (let index = 0; index + 1 < vertices.length; index++) {
    const [px, py] = vertices[index];
    const [qx, qy] = vertices[index + 1];
    if (py > point[1] !== qy > point[1]) {
      const crossing = px + ((point[1] - py) * (qx - px)) / (qy - py);
      if (crossing > point[0]) {
        inside = !inside;
      }
    }
  }
  return inside;
}

/** Monotone pseudo angle of a turn (counter-clockwise positive, u-turn at +2). */
export function getTurnKey(cross: number, dot: number): number {
  const denominator = Math.abs(cross) + Math.abs(dot);
  if (denominator === 0) {
    return 0;
  }
  const ratio = cross / denominator;
  if (dot >= 0) {
    return ratio;
  }
  return (cross >= 0 ? 2 : -2) - ratio;
}
