// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** One finite planar segment `[x0, y0, x1, y1]`. */
export type PolygonizationSegment = readonly [number, number, number, number];

/** Input of {@link getSegmentPolygonizationDiagnostics}. */
export type SegmentPolygonizationDiagnosticsInput = {
  /** Noded planar linework. Intersections must occur at shared segment endpoints. */
  segments: readonly PolygonizationSegment[];
  /** Number of valid leading segment rows. Defaults to every row. */
  count?: number;
  /** Optional group per segment. Vertices and rings never connect across groups. */
  groups?: readonly number[];
  /** Chebyshev distance used to identify equal endpoints. Defaults to exact equality. */
  vertexTolerance?: number;
};

/** One bounded face recovered from noded linework. */
export type SegmentPolygonizationRing = {
  /** Segment indices in traversal order. */
  segmentIndices: number[];
  /** Closed counter-clockwise vertex sequence. */
  positions: [number, number][];
  /** Positive signed planar area. */
  signedArea: number;
};

/** Why a valid input segment was or was not consumed by a polygon ring. */
export type SegmentPolygonizationClassification = 'closed-ring' | 'cut-edge' | 'dangle' | 'unused';

/** Full diagnostics for a polygonization pass over already-noded planar segments. */
export type SegmentPolygonizationDiagnostics = {
  /** Counter-clockwise bounded faces. The unbounded exterior face is omitted. */
  closedRings: SegmentPolygonizationRing[];
  /** Maximal connected chains of non-degenerate segments that belong to no ring. */
  openChains: number[][];
  /** Ring segment indices, sorted in input order. */
  closedRingSegmentIndices: number[];
  /** Bridges left after recursively removing dangles, sorted in input order. */
  cutEdgeSegmentIndices: number[];
  /** Segments recursively removed from degree-one vertices, sorted in input order. */
  dangleSegmentIndices: number[];
  /** Every segment not consumed by a bounded face, sorted in input order. */
  unusedSegmentIndices: number[];
  /** One classification for every valid leading segment row. */
  segmentClassifications: SegmentPolygonizationClassification[];
};

type Vertex = {
  x: number;
  y: number;
  group: number;
  incidentSegments: number[];
};

type SegmentRecord = {
  startVertex: number;
  endVertex: number;
};

type HalfEdge = {
  segmentIndex: number;
  startVertex: number;
  endVertex: number;
  twinIndex: number;
};

/**
 * Classifies already-noded finite planar linework using the polygonizer graph decomposition.
 *
 * Degree-one edges are peeled recursively as dangles. Tarjan bridges in the remaining graph are
 * cut edges. The remaining half-edge graph is traversed clockwise from each incoming twin so its
 * positive-area walks are the bounded polygon faces. `unusedSegmentIndices` is the complete union
 * of dangles, cut edges and degenerate segments, while `openChains` preserves their connectivity.
 */
export function getSegmentPolygonizationDiagnostics(
  input: SegmentPolygonizationDiagnosticsInput
): SegmentPolygonizationDiagnostics {
  const count = input.count ?? input.segments.length;
  if (!Number.isInteger(count) || count < 0 || count > input.segments.length) {
    throw new Error('count must select valid leading segment rows');
  }
  if (input.groups && input.groups.length < count) {
    throw new Error('groups must cover every valid segment row');
  }
  const vertexTolerance = input.vertexTolerance ?? 0;
  if (!Number.isFinite(vertexTolerance) || vertexTolerance < 0) {
    throw new Error('vertexTolerance must be finite and non-negative');
  }

  const vertices: Vertex[] = [];
  const segmentRecords: SegmentRecord[] = [];
  const degenerateSegments = new Set<number>();
  const findOrInsertVertex = (x: number, y: number, group: number): number => {
    for (let vertexIndex = 0; vertexIndex < vertices.length; vertexIndex++) {
      const vertex = vertices[vertexIndex];
      if (
        vertex.group === group &&
        Math.abs(vertex.x - x) <= vertexTolerance &&
        Math.abs(vertex.y - y) <= vertexTolerance
      ) {
        return vertexIndex;
      }
    }
    vertices.push({x, y, group, incidentSegments: []});
    return vertices.length - 1;
  };

  for (let segmentIndex = 0; segmentIndex < count; segmentIndex++) {
    const [startX, startY, endX, endY] = input.segments[segmentIndex];
    if (![startX, startY, endX, endY].every(Number.isFinite)) {
      throw new Error('segments must contain finite coordinates');
    }
    const group = input.groups?.[segmentIndex] ?? 0;
    const startVertex = findOrInsertVertex(startX, startY, group);
    const endVertex = findOrInsertVertex(endX, endY, group);
    segmentRecords.push({startVertex, endVertex});
    if (startVertex === endVertex) {
      degenerateSegments.add(segmentIndex);
      continue;
    }
    vertices[startVertex].incidentSegments.push(segmentIndex);
    vertices[endVertex].incidentSegments.push(segmentIndex);
  }

  const activeSegments = new Array(count).fill(true);
  const vertexDegrees = vertices.map(vertex => vertex.incidentSegments.length);
  for (const segmentIndex of degenerateSegments) {
    activeSegments[segmentIndex] = false;
  }
  const dangleSegments = new Set<number>();
  const queuedVertices: number[] = [];
  for (let vertexIndex = 0; vertexIndex < vertices.length; vertexIndex++) {
    if (vertexDegrees[vertexIndex] <= 1) {
      queuedVertices.push(vertexIndex);
    }
  }
  for (let queueIndex = 0; queueIndex < queuedVertices.length; queueIndex++) {
    const vertexIndex = queuedVertices[queueIndex];
    for (const segmentIndex of vertices[vertexIndex].incidentSegments) {
      if (!activeSegments[segmentIndex]) {
        continue;
      }
      activeSegments[segmentIndex] = false;
      dangleSegments.add(segmentIndex);
      const {startVertex, endVertex} = segmentRecords[segmentIndex];
      vertexDegrees[startVertex]--;
      vertexDegrees[endVertex]--;
      const otherVertex = startVertex === vertexIndex ? endVertex : startVertex;
      if (vertexDegrees[otherVertex] === 1) {
        queuedVertices.push(otherVertex);
      }
    }
  }

  const discoveryIndices = new Array(vertices.length).fill(-1);
  const lowLinks = new Array(vertices.length).fill(-1);
  const cutEdgeSegments = new Set<number>();
  let nextDiscoveryIndex = 0;
  const visitVertex = (vertexIndex: number, incomingSegment: number): void => {
    discoveryIndices[vertexIndex] = nextDiscoveryIndex;
    lowLinks[vertexIndex] = nextDiscoveryIndex;
    nextDiscoveryIndex++;
    for (const segmentIndex of vertices[vertexIndex].incidentSegments) {
      if (!activeSegments[segmentIndex] || segmentIndex === incomingSegment) {
        continue;
      }
      const record = segmentRecords[segmentIndex];
      const nextVertex = record.startVertex === vertexIndex ? record.endVertex : record.startVertex;
      if (discoveryIndices[nextVertex] < 0) {
        visitVertex(nextVertex, segmentIndex);
        lowLinks[vertexIndex] = Math.min(lowLinks[vertexIndex], lowLinks[nextVertex]);
        if (lowLinks[nextVertex] > discoveryIndices[vertexIndex]) {
          cutEdgeSegments.add(segmentIndex);
        }
      } else {
        lowLinks[vertexIndex] = Math.min(lowLinks[vertexIndex], discoveryIndices[nextVertex]);
      }
    }
  };
  for (let vertexIndex = 0; vertexIndex < vertices.length; vertexIndex++) {
    if (discoveryIndices[vertexIndex] < 0) {
      visitVertex(vertexIndex, -1);
    }
  }

  const closedSegmentSet = new Set<number>();
  for (let segmentIndex = 0; segmentIndex < count; segmentIndex++) {
    if (activeSegments[segmentIndex] && !cutEdgeSegments.has(segmentIndex)) {
      closedSegmentSet.add(segmentIndex);
    }
  }
  const closedRings = getClosedRings(vertices, segmentRecords, closedSegmentSet);
  const closedRingSegmentIndices = [...closedSegmentSet].sort(compareNumbers);
  const cutEdgeSegmentIndices = [...cutEdgeSegments].sort(compareNumbers);
  const dangleSegmentIndices = [...dangleSegments].sort(compareNumbers);
  const unusedSegmentIndices: number[] = [];
  const segmentClassifications: SegmentPolygonizationClassification[] = [];
  for (let segmentIndex = 0; segmentIndex < count; segmentIndex++) {
    if (closedSegmentSet.has(segmentIndex)) {
      segmentClassifications.push('closed-ring');
    } else if (cutEdgeSegments.has(segmentIndex)) {
      segmentClassifications.push('cut-edge');
      unusedSegmentIndices.push(segmentIndex);
    } else if (dangleSegments.has(segmentIndex)) {
      segmentClassifications.push('dangle');
      unusedSegmentIndices.push(segmentIndex);
    } else {
      segmentClassifications.push('unused');
      unusedSegmentIndices.push(segmentIndex);
    }
  }

  return {
    closedRings,
    openChains: getOpenChains(vertices, segmentRecords, unusedSegmentIndices, degenerateSegments),
    closedRingSegmentIndices,
    cutEdgeSegmentIndices,
    dangleSegmentIndices,
    unusedSegmentIndices,
    segmentClassifications
  };
}

function getClosedRings(
  vertices: readonly Vertex[],
  segmentRecords: readonly SegmentRecord[],
  closedSegments: ReadonlySet<number>
): SegmentPolygonizationRing[] {
  const halfEdges: HalfEdge[] = [];
  const outgoingHalfEdges = vertices.map(() => [] as number[]);
  for (const segmentIndex of closedSegments) {
    const {startVertex, endVertex} = segmentRecords[segmentIndex];
    const forwardIndex = halfEdges.length;
    const reverseIndex = forwardIndex + 1;
    halfEdges.push({segmentIndex, startVertex, endVertex, twinIndex: reverseIndex});
    halfEdges.push({
      segmentIndex,
      startVertex: endVertex,
      endVertex: startVertex,
      twinIndex: forwardIndex
    });
    outgoingHalfEdges[startVertex].push(forwardIndex);
    outgoingHalfEdges[endVertex].push(reverseIndex);
  }
  for (const outgoing of outgoingHalfEdges) {
    outgoing.sort((leftIndex, rightIndex) => {
      const left = halfEdges[leftIndex];
      const right = halfEdges[rightIndex];
      const leftStart = vertices[left.startVertex];
      const leftEnd = vertices[left.endVertex];
      const rightStart = vertices[right.startVertex];
      const rightEnd = vertices[right.endVertex];
      return (
        Math.atan2(leftEnd.y - leftStart.y, leftEnd.x - leftStart.x) -
          Math.atan2(rightEnd.y - rightStart.y, rightEnd.x - rightStart.x) || leftIndex - rightIndex
      );
    });
  }
  const nextHalfEdges = new Array(halfEdges.length).fill(-1);
  for (let halfEdgeIndex = 0; halfEdgeIndex < halfEdges.length; halfEdgeIndex++) {
    const halfEdge = halfEdges[halfEdgeIndex];
    const outgoing = outgoingHalfEdges[halfEdge.endVertex];
    const twinPosition = outgoing.indexOf(halfEdge.twinIndex);
    nextHalfEdges[halfEdgeIndex] = outgoing[(twinPosition + outgoing.length - 1) % outgoing.length];
  }

  const visited = new Array(halfEdges.length).fill(false);
  const rings: SegmentPolygonizationRing[] = [];
  for (let startHalfEdge = 0; startHalfEdge < halfEdges.length; startHalfEdge++) {
    if (visited[startHalfEdge]) {
      continue;
    }
    const segmentIndices: number[] = [];
    const positions: [number, number][] = [];
    let halfEdgeIndex = startHalfEdge;
    while (!visited[halfEdgeIndex]) {
      visited[halfEdgeIndex] = true;
      const halfEdge = halfEdges[halfEdgeIndex];
      const start = vertices[halfEdge.startVertex];
      positions.push([start.x, start.y]);
      segmentIndices.push(halfEdge.segmentIndex);
      halfEdgeIndex = nextHalfEdges[halfEdgeIndex];
    }
    if (halfEdgeIndex !== startHalfEdge || positions.length < 3) {
      continue;
    }
    positions.push(positions[0]);
    const signedArea = getSignedArea(positions);
    if (signedArea > 0) {
      rings.push({segmentIndices, positions, signedArea});
    }
  }
  return rings;
}

function getOpenChains(
  vertices: readonly Vertex[],
  segmentRecords: readonly SegmentRecord[],
  unusedSegmentIndices: readonly number[],
  degenerateSegments: ReadonlySet<number>
): number[][] {
  const unusedSegments = new Set(
    unusedSegmentIndices.filter(segmentIndex => !degenerateSegments.has(segmentIndex))
  );
  const incidentUnusedSegments = vertices.map(vertex =>
    vertex.incidentSegments.filter(segmentIndex => unusedSegments.has(segmentIndex))
  );
  const visitedSegments = new Set<number>();
  const chains: number[][] = [];
  const walkChain = (startVertex: number, firstSegment: number): number[] => {
    const chain: number[] = [];
    let vertexIndex = startVertex;
    let segmentIndex = firstSegment;
    while (!visitedSegments.has(segmentIndex)) {
      visitedSegments.add(segmentIndex);
      chain.push(segmentIndex);
      const record = segmentRecords[segmentIndex];
      vertexIndex = record.startVertex === vertexIndex ? record.endVertex : record.startVertex;
      if (incidentUnusedSegments[vertexIndex].length !== 2) {
        break;
      }
      const nextSegment = incidentUnusedSegments[vertexIndex].find(
        candidate => !visitedSegments.has(candidate)
      );
      if (nextSegment === undefined) {
        break;
      }
      segmentIndex = nextSegment;
    }
    return chain;
  };

  for (let vertexIndex = 0; vertexIndex < vertices.length; vertexIndex++) {
    if (incidentUnusedSegments[vertexIndex].length === 2) {
      continue;
    }
    for (const segmentIndex of incidentUnusedSegments[vertexIndex]) {
      if (!visitedSegments.has(segmentIndex)) {
        chains.push(walkChain(vertexIndex, segmentIndex));
      }
    }
  }
  for (const segmentIndex of unusedSegments) {
    if (!visitedSegments.has(segmentIndex)) {
      chains.push(walkChain(segmentRecords[segmentIndex].startVertex, segmentIndex));
    }
  }
  return chains;
}

function getSignedArea(positions: readonly [number, number][]): number {
  let doubledArea = 0;
  for (let index = 0; index + 1 < positions.length; index++) {
    doubledArea +=
      positions[index][0] * positions[index + 1][1] - positions[index + 1][0] * positions[index][1];
  }
  return doubledArea * 0.5;
}

function compareNumbers(left: number, right: number): number {
  return left - right;
}
