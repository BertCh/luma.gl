// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Road-network helpers shared by the reachability and network modes. */

import type {SpatialAnalysisRoadNetwork} from '../spatial-analysis-data';

/** Walking speed in m/s. */
export const WALK_SPEED = 1.4;
/** Travel time assigned to edges vehicles may not use; larger than any cost limit. */
export const CLOSED_EDGE_SECONDS = 1e9;
/** Drive speed in m/s by road class: motorway, primary/secondary, tertiary, other. */
export const DRIVE_SPEEDS = [20, 12, 9, 7] as const;

/** How edge travel times are derived from lengths. */
export type Transport = 'walk' | 'drive';

/** Directed edges sorted by source node (CSR row order) with a road class per edge. */
export type SortedEdges = {
  sources: Uint32Array;
  targets: Uint32Array;
  lengths: Float32Array;
  classes: Uint8Array;
  /** 1 when vehicles may use the edge (not against a one-way street). */
  drivable: Uint8Array;
};

/** Linear scan for the node closest to `[x, y]` meters. */
export function findNearestNode(
  nodePositions: Float32Array,
  [x, y]: readonly [number, number]
): number {
  let nearest = 0;
  let nearestDistance = Infinity;
  for (let node = 0; node < nodePositions.length / 2; node++) {
    const distance = (nodePositions[node * 2] - x) ** 2 + (nodePositions[node * 2 + 1] - y) ** 2;
    if (distance < nearestDistance) {
      nearestDistance = distance;
      nearest = node;
    }
  }
  return nearest;
}

/**
 * Counting-sorts directed edges by source (the row order `GPUCOOToCSR` requires) and recovers a
 * road class per edge. The loader emits edges per drawable segment in segment order, so each edge
 * is matched to the next segment whose endpoints it connects.
 */
export function sortEdgesBySource(roads: SpatialAnalysisRoadNetwork): SortedEdges {
  const {edgeSources, edgeTargets, edgeLengths, nodePositions, segments, segmentClasses} = roads;
  const edgeCount = edgeSources.length;
  const nodeCount = nodePositions.length / 2;
  const edgeClasses = new Uint8Array(edgeCount).fill(3);
  let segment = 0;
  const connects = (edge: number, candidate: number) => {
    const sourceX = nodePositions[edgeSources[edge] * 2];
    const sourceY = nodePositions[edgeSources[edge] * 2 + 1];
    const targetX = nodePositions[edgeTargets[edge] * 2];
    const targetY = nodePositions[edgeTargets[edge] * 2 + 1];
    const [x0, y0, x1, y1] = segments.subarray(candidate * 4, candidate * 4 + 4);
    return (
      (sourceX === x0 && sourceY === y0 && targetX === x1 && targetY === y1) ||
      (sourceX === x1 && sourceY === y1 && targetX === x0 && targetY === y0)
    );
  };
  for (let edge = 0; edge < edgeCount; edge++) {
    while (segment < segmentClasses.length && !connects(edge, segment)) segment++;
    if (segment >= segmentClasses.length) break;
    edgeClasses[edge] = segmentClasses[segment];
  }
  const starts = new Uint32Array(nodeCount + 1);
  for (let edge = 0; edge < edgeCount; edge++) starts[edgeSources[edge] + 1]++;
  for (let node = 0; node < nodeCount; node++) starts[node + 1] += starts[node];
  const sorted: SortedEdges = {
    sources: new Uint32Array(edgeCount),
    targets: new Uint32Array(edgeCount),
    lengths: new Float32Array(edgeCount),
    classes: new Uint8Array(edgeCount),
    drivable: new Uint8Array(edgeCount)
  };
  for (let edge = 0; edge < edgeCount; edge++) {
    const slot = starts[edgeSources[edge]]++;
    sorted.sources[slot] = edgeSources[edge];
    sorted.targets[slot] = edgeTargets[edge];
    sorted.lengths[slot] = edgeLengths[edge];
    sorted.classes[slot] = edgeClasses[edge];
    sorted.drivable[slot] = roads.edgeDrivable[edge];
  }
  return sorted;
}

/**
 * Writes travel time in seconds per sorted edge. Driving keeps edges against a one-way street in
 * the topology but at {@link CLOSED_EDGE_SECONDS}, so walk and drive share one CSR and differ only
 * in the weight buffer contents.
 */
export function writeEdgeTravelSeconds(
  edges: SortedEdges,
  transport: Transport,
  target: Float32Array
): void {
  for (let edge = 0; edge < target.length; edge++) {
    if (transport === 'walk') {
      target[edge] = edges.lengths[edge] / WALK_SPEED;
    } else {
      target[edge] = edges.drivable[edge]
        ? edges.lengths[edge] / DRIVE_SPEEDS[edges.classes[edge]]
        : CLOSED_EDGE_SECONDS;
    }
  }
}
