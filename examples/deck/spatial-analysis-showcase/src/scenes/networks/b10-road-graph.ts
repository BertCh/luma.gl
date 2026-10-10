// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Shared road-graph preparation for the three B10 scenes. Everything here is CPU work that runs
 * once while a scene is created: it turns the `chicago-roads` dataset (directed edges with
 * polyline geometry) into the CSR layouts and draw buffers the GPU contributors consume.
 *
 * - {@link buildRoadGraph}: a symmetric CSR (both directions of every street, equal lengths) for
 *   the undirected contributors (network K function, analytics columns, statistics, subgraph
 *   filter), plus one drawn polyline per street as segment rows tagged with their CSR slot.
 * - {@link buildDenseDirectedGraph}: the directed graph with every polyline vertex as a node, so
 *   a map matcher that treats edges as straight lines sees the real street geometry. The CSR row
 *   of that graph is the matcher's edge id; `rowEdge` maps it back to the dataset edge.
 */

import type {LoadedDataset} from '../../data/catalog';

/** Dataset edge index meaning "no reverse edge" / "no edge". */
export const NO_EDGE = 0xffffffff;

/** Road class names of `chicago-roads.edgeClass`. */
export const ROAD_CLASS_NAMES = [
  'motorway',
  'trunk',
  'primary',
  'secondary',
  'tertiary',
  'residential',
  'service/other'
] as const;

/** Street graph prepared for the undirected contributors and for drawing. */
export type RoadGraph = {
  origin: readonly [number, number];
  nodeCount: number;
  /** `x, y` meters per node. */
  nodePositions: Float32Array;
  /** Dataset directed edge count. */
  edgeCount: number;
  edgeSource: Uint32Array;
  edgeTarget: Uint32Array;
  edgeLength: Float32Array;
  edgeClass: Uint8Array;
  edgeSpeed: Uint8Array;
  edgeReverse: Uint32Array;
  /** Symmetric CSR slot count (directed slots, both directions of every street). */
  slotCount: number;
  offsets: Uint32Array;
  neighbors: Uint32Array;
  /** Street length in meters per slot (equal for the two directions). */
  weights: Float32Array;
  /** Source node of every slot. */
  slotSource: Uint32Array;
  /** Dataset edge each slot was copied from. */
  slotEdge: Uint32Array;
  /** Road class index per slot, as float. */
  slotClass: Float32Array;
  /** Speed limit in km/h per slot, as float. */
  slotSpeed: Float32Array;
  /** Slot of every dataset edge (its own direction). */
  slotOfEdge: Uint32Array;
  /** Half the sum of all slot weights: the network length `L` of the K function, in meters. */
  networkLength: number;
  /** Drawn polyline segments as `x0, y0, x1, y1` rows (one polyline per street). */
  segments: Float32Array;
  segmentCount: number;
  /** Slot that colors each drawn segment. */
  segmentSlots: Uint32Array;
  /** Dataset edge of each drawn segment. */
  segmentEdge: Uint32Array;
  /** Planar bounds of all nodes, `[minX, minY, maxX, maxY]`. */
  bounds: [number, number, number, number];
};

/** Reads the road columns and builds the symmetric CSR and the draw segments. */
export function buildRoadGraph(roads: LoadedDataset, origin: readonly [number, number]): RoadGraph {
  const nodePositions = roads.projectColumn('nodes', origin);
  const nodeCount = nodePositions.length / 2;
  const edgeSource = roads.column<Uint32Array>('edgeSource');
  const edgeTarget = roads.column<Uint32Array>('edgeTarget');
  const edgeLength = roads.column<Float32Array>('edgeLength');
  const edgeClass = roads.column<Uint8Array>('edgeClass');
  const edgeSpeed = roads.column<Uint8Array>('edgeSpeed');
  const edgeReverse = roads.column<Uint32Array>('edgeReverse');
  const pathOffsets = roads.column<Uint32Array>('edgePathOffsets');
  const pathVertices = roads.projectColumn('edgeVertices', origin);
  const edgeCount = edgeSource.length;

  // Symmetric slots: every dataset edge, plus a reverse copy for one-way streets.
  let slotCount = edgeCount;
  for (let edge = 0; edge < edgeCount; edge++) if (edgeReverse[edge] === NO_EDGE) slotCount++;
  const sourceOfSlot = new Uint32Array(slotCount);
  const targetOfSlot = new Uint32Array(slotCount);
  const edgeOfSlot = new Uint32Array(slotCount);
  const primary = new Uint8Array(slotCount);
  let cursor = 0;
  for (let edge = 0; edge < edgeCount; edge++) {
    sourceOfSlot[cursor] = edgeSource[edge];
    targetOfSlot[cursor] = edgeTarget[edge];
    edgeOfSlot[cursor] = edge;
    primary[cursor++] = 1;
    if (edgeReverse[edge] === NO_EDGE) {
      sourceOfSlot[cursor] = edgeTarget[edge];
      targetOfSlot[cursor] = edgeSource[edge];
      edgeOfSlot[cursor++] = edge;
    }
  }
  const offsets = new Uint32Array(nodeCount + 1);
  for (let slot = 0; slot < slotCount; slot++) offsets[sourceOfSlot[slot] + 1]++;
  for (let node = 0; node < nodeCount; node++) offsets[node + 1] += offsets[node];
  const fill = offsets.slice(0, nodeCount);
  const neighbors = new Uint32Array(slotCount);
  const weights = new Float32Array(slotCount);
  const slotSource = new Uint32Array(slotCount);
  const slotEdge = new Uint32Array(slotCount);
  const slotClass = new Float32Array(slotCount);
  const slotSpeed = new Float32Array(slotCount);
  const slotOfEdge = new Uint32Array(edgeCount).fill(NO_EDGE);
  let totalWeight = 0;
  for (let input = 0; input < slotCount; input++) {
    const slot = fill[sourceOfSlot[input]]++;
    const edge = edgeOfSlot[input];
    neighbors[slot] = targetOfSlot[input];
    // Strictly positive weights keep reachability predecessors well defined.
    weights[slot] = Math.max(edgeLength[edge], 0.5);
    slotSource[slot] = sourceOfSlot[input];
    slotEdge[slot] = edge;
    slotClass[slot] = edgeClass[edge];
    slotSpeed[slot] = edgeSpeed[edge];
    totalWeight += weights[slot];
    if (primary[input]) slotOfEdge[edge] = slot;
  }

  // One drawn polyline per street: skip the edge whose reverse has a smaller index.
  let segmentCount = 0;
  for (let edge = 0; edge < edgeCount; edge++) {
    if (edgeReverse[edge] === NO_EDGE || edge < edgeReverse[edge]) {
      segmentCount += pathOffsets[edge + 1] - pathOffsets[edge] - 1;
    }
  }
  const segments = new Float32Array(segmentCount * 4);
  const segmentSlots = new Uint32Array(segmentCount);
  const segmentEdge = new Uint32Array(segmentCount);
  let row = 0;
  for (let edge = 0; edge < edgeCount; edge++) {
    if (!(edgeReverse[edge] === NO_EDGE || edge < edgeReverse[edge])) continue;
    const first = pathOffsets[edge];
    const last = pathOffsets[edge + 1] - 1;
    for (let vertex = first; vertex < last; vertex++) {
      // End vertices use the node coordinates so chains meet exactly (GPULineMerge compares
      // float32 endpoints for equality).
      const start = vertex === first ? nodePositions.subarray(edgeSource[edge] * 2) : null;
      const end = vertex + 1 === last ? nodePositions.subarray(edgeTarget[edge] * 2) : null;
      segments[row * 4] = start ? start[0] : pathVertices[vertex * 2];
      segments[row * 4 + 1] = start ? start[1] : pathVertices[vertex * 2 + 1];
      segments[row * 4 + 2] = end ? end[0] : pathVertices[vertex * 2 + 2];
      segments[row * 4 + 3] = end ? end[1] : pathVertices[vertex * 2 + 3];
      segmentSlots[row] = slotOfEdge[edge];
      segmentEdge[row] = edge;
      row++;
    }
  }

  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  for (let node = 0; node < nodeCount; node++) {
    minimumX = Math.min(minimumX, nodePositions[node * 2]);
    maximumX = Math.max(maximumX, nodePositions[node * 2]);
    minimumY = Math.min(minimumY, nodePositions[node * 2 + 1]);
    maximumY = Math.max(maximumY, nodePositions[node * 2 + 1]);
  }

  return {
    origin,
    nodeCount,
    nodePositions,
    edgeCount,
    edgeSource,
    edgeTarget,
    edgeLength,
    edgeClass,
    edgeSpeed,
    edgeReverse,
    slotCount,
    offsets,
    neighbors,
    weights,
    slotSource,
    slotEdge,
    slotClass,
    slotSpeed,
    slotOfEdge,
    networkLength: totalWeight / 2,
    segments,
    segmentCount,
    segmentSlots,
    segmentEdge,
    bounds: [minimumX, minimumY, maximumX, maximumY]
  };
}

/** Directed graph with one node per polyline vertex, for map matching. */
export type DenseDirectedGraph = {
  nodeCount: number;
  nodePositions: Float32Array;
  /** CSR offsets; the CSR row is the matcher's edge id. */
  offsets: Uint32Array;
  targets: Uint32Array;
  /** Dataset edge of every CSR row. */
  rowEdge: Uint32Array;
  rowCount: number;
};

/** Longest dense-graph edge in meters; the matcher's edge grid drops edges spanning too many cells. */
const MAXIMUM_DENSE_EDGE_LENGTH = 300;

/**
 * Splits every directed dataset edge into one CSR row per polyline segment (and long segments into
 * pieces of at most 300 m, which the matcher's edge grid needs). Interior vertices become nodes
 * private to their edge, so a street stays two parallel one-way lanes (as the dataset models it)
 * and a route can only leave a street at its end nodes.
 */
export function buildDenseDirectedGraph(
  roads: LoadedDataset,
  graph: RoadGraph
): DenseDirectedGraph {
  const pathOffsets = roads.column<Uint32Array>('edgePathOffsets');
  const pathVertices = roads.projectColumn('edgeVertices', graph.origin);
  const {edgeCount, edgeSource, edgeTarget, nodePositions} = graph;
  const extraPositions: number[] = [];
  const sources: number[] = [];
  const targets: number[] = [];
  const edges: number[] = [];
  let nextNode = graph.nodeCount;
  const positionOf = (node: number): [number, number] =>
    node < graph.nodeCount
      ? [nodePositions[node * 2], nodePositions[node * 2 + 1]]
      : [
          extraPositions[(node - graph.nodeCount) * 2],
          extraPositions[(node - graph.nodeCount) * 2 + 1]
        ];
  const addNode = (x: number, y: number): number => {
    extraPositions.push(x, y);
    return nextNode++;
  };
  for (let edge = 0; edge < edgeCount; edge++) {
    const first = pathOffsets[edge];
    const vertexCount = pathOffsets[edge + 1] - first;
    // Chain of nodes along the polyline, end nodes shared with the junctions.
    let previous = edgeSource[edge];
    if (vertexCount < 2) {
      sources.push(previous);
      targets.push(edgeTarget[edge]);
      edges.push(edge);
      continue;
    }
    for (let vertex = 1; vertex < vertexCount; vertex++) {
      const last = vertex === vertexCount - 1;
      const [x0, y0] = positionOf(previous);
      const x1 = last ? nodePositions[edgeTarget[edge] * 2] : pathVertices[(first + vertex) * 2];
      const y1 = last
        ? nodePositions[edgeTarget[edge] * 2 + 1]
        : pathVertices[(first + vertex) * 2 + 1];
      const pieces = Math.max(
        1,
        Math.ceil(Math.hypot(x1 - x0, y1 - y0) / MAXIMUM_DENSE_EDGE_LENGTH)
      );
      for (let piece = 1; piece <= pieces; piece++) {
        const node =
          piece === pieces && last
            ? edgeTarget[edge]
            : piece === pieces
              ? addNode(x1, y1)
              : addNode(x0 + ((x1 - x0) * piece) / pieces, y0 + ((y1 - y0) * piece) / pieces);
        sources.push(previous);
        targets.push(node);
        edges.push(edge);
        previous = node;
      }
    }
  }
  const nodeCount = nextNode;
  const positions = new Float32Array(nodeCount * 2);
  positions.set(nodePositions);
  positions.set(extraPositions, graph.nodeCount * 2);
  const rowCount = sources.length;
  const offsets = new Uint32Array(nodeCount + 1);
  for (let index = 0; index < rowCount; index++) offsets[sources[index] + 1]++;
  for (let node = 0; node < nodeCount; node++) offsets[node + 1] += offsets[node];
  const fill = offsets.slice(0, nodeCount);
  const sortedTargets = new Uint32Array(rowCount);
  const rowEdge = new Uint32Array(rowCount);
  for (let index = 0; index < rowCount; index++) {
    const slot = fill[sources[index]]++;
    sortedTargets[slot] = targets[index];
    rowEdge[slot] = edges[index];
  }
  return {nodeCount, nodePositions: positions, offsets, targets: sortedTargets, rowEdge, rowCount};
}

/** Uniform grid over drawn segments for hover lookups. */
export class SegmentIndex {
  private readonly cellSize: number;
  private readonly columns: number;
  private readonly rows: number;
  private readonly minimumX: number;
  private readonly minimumY: number;
  private readonly cellStart: Uint32Array;
  private readonly cellItems: Uint32Array;
  private readonly segments: Float32Array;

  constructor(
    segments: Float32Array,
    bounds: readonly [number, number, number, number],
    cellSize = 150
  ) {
    this.segments = segments;
    this.cellSize = cellSize;
    this.minimumX = bounds[0] - cellSize;
    this.minimumY = bounds[1] - cellSize;
    this.columns = Math.ceil((bounds[2] - bounds[0]) / cellSize) + 3;
    this.rows = Math.ceil((bounds[3] - bounds[1]) / cellSize) + 3;
    const cellCount = this.columns * this.rows;
    const counts = new Uint32Array(cellCount + 1);
    const segmentCount = segments.length / 4;
    const visit = (segment: number, callback: (cell: number) => void) => {
      const x0 = segments[segment * 4];
      const y0 = segments[segment * 4 + 1];
      const x1 = segments[segment * 4 + 2];
      const y1 = segments[segment * 4 + 3];
      const steps = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / (cellSize * 0.5)));
      let last = -1;
      for (let step = 0; step <= steps; step++) {
        const t = step / steps;
        const column = Math.floor((x0 + (x1 - x0) * t - this.minimumX) / cellSize);
        const row = Math.floor((y0 + (y1 - y0) * t - this.minimumY) / cellSize);
        const cell = row * this.columns + column;
        if (cell !== last && cell >= 0 && cell < cellCount) callback(cell);
        last = cell;
      }
    };
    for (let segment = 0; segment < segmentCount; segment++) {
      visit(segment, cell => counts[cell + 1]++);
    }
    for (let cell = 0; cell < cellCount; cell++) counts[cell + 1] += counts[cell];
    this.cellStart = counts;
    const fill = counts.slice(0, cellCount);
    this.cellItems = new Uint32Array(counts[cellCount]);
    for (let segment = 0; segment < segmentCount; segment++) {
      visit(segment, cell => {
        this.cellItems[fill[cell]++] = segment;
      });
    }
  }

  /** Nearest segment within `maximumDistance` meters of `(x, y)`, or -1. */
  nearest(x: number, y: number, maximumDistance: number): number {
    const column = Math.floor((x - this.minimumX) / this.cellSize);
    const row = Math.floor((y - this.minimumY) / this.cellSize);
    const reach = Math.ceil(maximumDistance / this.cellSize);
    let best = -1;
    let bestDistance = maximumDistance * maximumDistance;
    for (let ring = 0; ring <= reach; ring++) {
      for (let rowOffset = -ring; rowOffset <= ring; rowOffset++) {
        for (let columnOffset = -ring; columnOffset <= ring; columnOffset++) {
          if (ring && Math.abs(rowOffset) !== ring && Math.abs(columnOffset) !== ring) continue;
          const c = column + columnOffset;
          const r = row + rowOffset;
          if (c < 0 || r < 0 || c >= this.columns || r >= this.rows) continue;
          const cell = r * this.columns + c;
          for (let item = this.cellStart[cell]; item < this.cellStart[cell + 1]; item++) {
            const segment = this.cellItems[item];
            const distance = this.distanceSquared(segment, x, y);
            if (distance < bestDistance) {
              bestDistance = distance;
              best = segment;
            }
          }
        }
      }
      // Segments are sampled into the grid at half-cell intervals. Two extra rings cover the
      // cell containing the closest sampled point before the distance bound can terminate search.
      const searchedDistance = Math.max(0, ring - 2) * this.cellSize;
      if (best >= 0 && bestDistance <= searchedDistance * searchedDistance) break;
    }
    return best;
  }

  private distanceSquared(segment: number, x: number, y: number): number {
    const s = this.segments;
    const x0 = s[segment * 4];
    const y0 = s[segment * 4 + 1];
    const dx = s[segment * 4 + 2] - x0;
    const dy = s[segment * 4 + 3] - y0;
    const lengthSquared = dx * dx + dy * dy;
    const t =
      lengthSquared > 0
        ? Math.min(1, Math.max(0, ((x - x0) * dx + (y - y0) * dy) / lengthSquared))
        : 0;
    const px = x0 + dx * t - x;
    const py = y0 + dy * t - y;
    return px * px + py * py;
  }
}

/** Formats a length in meters as `850 m` or `3.2 km`. */
export function formatLength(meters: number): string {
  return meters >= 1000
    ? `${(meters / 1000).toFixed(meters >= 10000 ? 0 : 1)} km`
    : `${Math.round(meters)} m`;
}

/** Eight-level unicode bar for `value` in `[0, 1]`. */
export function sparkBar(value: number): string {
  const bars = '▁▂▃▄▅▆▇█';
  return bars[Math.max(0, Math.min(7, Math.round(value * 7)))];
}
