// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';
import type {LocalMetricProjection} from '../../engine/projection';

/** Road classes of `chicago-roads` `edgeClass`, in category order. */
export const ROAD_CLASS_NAMES = [
  'motorway',
  'trunk',
  'primary',
  'secondary',
  'tertiary',
  'residential',
  'service/other'
] as const;

/** Sentinel for "no edge" in `edgeReverse`. */
export const NO_EDGE = 0xffffffff;

/**
 * The Chicago drive network of the `chicago-roads` dataset in planar meters, ready for the GPU:
 * a directed CSR (the dataset is already sorted by source), per-edge attributes, and drawable
 * polyline segments that remember which edge and which node each segment belongs to.
 */
export type RoadNetwork = {
  /** `[longitude, latitude]` origin of the planar meter frame. */
  origin: [number, number];
  projection: LocalMetricProjection;
  nodeCount: number;
  edgeCount: number;
  /** `x, y` meters per node. */
  nodePositions: Float32Array;
  /** `longitude, latitude` degrees per node. */
  nodeLngLat: Float32Array;
  /** CSR row offsets, `nodeCount + 1`. Edge `e` of node `u` is `offsets[u] <= e < offsets[u + 1]`. */
  offsets: Uint32Array;
  sources: Uint32Array;
  targets: Uint32Array;
  /** Edge length in meters. */
  length: Float32Array;
  /** Posted or class-default speed in km/h. */
  speedKmh: Uint8Array;
  /** Road class index into {@link ROAD_CLASS_NAMES}. */
  roadClass: Uint8Array;
  /** Free-flow travel time in seconds. */
  travelTime: Float32Array;
  /** Edge row of the opposite direction, or {@link NO_EDGE} for one-way streets. */
  reverse: Uint32Array;
  /** Drawable segments `x0, y0, x1, y1` in meters (the edge polylines, split per vertex pair). */
  segments: Float32Array;
  /** Edge row of each drawable segment. */
  segmentEdges: Uint32Array;
  /** Target node of the edge of each drawable segment. */
  segmentTargetNodes: Uint32Array;
  /** Source node of the edge of each drawable segment. */
  segmentSourceNodes: Uint32Array;
  /** Segments of primary and faster roads, for a heavier base layer. */
  majorSegments: Float32Array;
  /** Nearest node to a point in meters. */
  findNearestNode: (x: number, y: number) => number;
  /** Nearest edge to a point in meters, with the fraction along the edge's straight chord. */
  findNearestEdge: (x: number, y: number) => {edge: number; distance: number};
  /** Approximate total road length in km (each two-way street counted once). */
  streetKilometers: number;
};

const GRID_CELL_METERS = 250;

/** Builds the planar-meter road network of `chicago-roads`. */
export function buildRoadNetwork(roads: LoadedDataset): RoadNetwork {
  const origin = roads.defaultOrigin;
  const projection = roads.getProjection(origin);
  const nodeLngLat = roads.column<Float32Array>('nodes');
  const nodePositions = roads.projectColumn('nodes', origin);
  const nodeCount = nodePositions.length / 2;
  const sources = roads.column<Uint32Array>('edgeSource');
  const targets = roads.column<Uint32Array>('edgeTarget');
  const length = roads.column<Float32Array>('edgeLength');
  const roadClass = roads.column<Uint8Array>('edgeClass');
  const speedKmh = roads.column<Uint8Array>('edgeSpeed');
  const travelTime = roads.column<Float32Array>('edgeTravelTime');
  const reverse = roads.column<Uint32Array>('edgeReverse');
  const offsets = roads.column<Uint32Array>('nodeEdgeOffsets');
  const pathOffsets = roads.column<Uint32Array>('edgePathOffsets');
  const vertices = roads.projectColumn('edgeVertices', origin);
  const edgeCount = sources.length;

  let segmentCount = 0;
  for (let edge = 0; edge < edgeCount; edge++) {
    segmentCount += Math.max(0, pathOffsets[edge + 1] - pathOffsets[edge] - 1);
  }
  const segments = new Float32Array(segmentCount * 4);
  const segmentEdges = new Uint32Array(segmentCount);
  const segmentTargetNodes = new Uint32Array(segmentCount);
  const segmentSourceNodes = new Uint32Array(segmentCount);
  const major: number[] = [];
  let cursor = 0;
  for (let edge = 0; edge < edgeCount; edge++) {
    const first = pathOffsets[edge];
    const last = pathOffsets[edge + 1] - 1;
    for (let vertex = first; vertex < last; vertex++) {
      segments[cursor * 4] = vertices[vertex * 2];
      segments[cursor * 4 + 1] = vertices[vertex * 2 + 1];
      segments[cursor * 4 + 2] = vertices[(vertex + 1) * 2];
      segments[cursor * 4 + 3] = vertices[(vertex + 1) * 2 + 1];
      segmentEdges[cursor] = edge;
      segmentTargetNodes[cursor] = targets[edge];
      segmentSourceNodes[cursor] = sources[edge];
      if (roadClass[edge] <= 3) {
        major.push(
          segments[cursor * 4],
          segments[cursor * 4 + 1],
          segments[cursor * 4 + 2],
          segments[cursor * 4 + 3]
        );
      }
      cursor++;
    }
  }

  // Node grid for click picking.
  const key = (cellX: number, cellY: number) => cellX * 100003 + cellY;
  let minX = Infinity;
  let minY = Infinity;
  for (let node = 0; node < nodeCount; node++) {
    minX = Math.min(minX, nodePositions[node * 2]);
    minY = Math.min(minY, nodePositions[node * 2 + 1]);
  }
  const nodeGrid = new Map<number, number[]>();
  for (let node = 0; node < nodeCount; node++) {
    const cellKey = key(
      Math.floor((nodePositions[node * 2] - minX) / GRID_CELL_METERS),
      Math.floor((nodePositions[node * 2 + 1] - minY) / GRID_CELL_METERS)
    );
    const bucket = nodeGrid.get(cellKey);
    if (bucket) bucket.push(node);
    else nodeGrid.set(cellKey, [node]);
  }
  const findNearestNode = (x: number, y: number): number => {
    const cellX = Math.floor((x - minX) / GRID_CELL_METERS);
    const cellY = Math.floor((y - minY) / GRID_CELL_METERS);
    let best = -1;
    let bestDistance = Infinity;
    for (
      let ring = 0;
      ring < 60 && (best < 0 || ring * GRID_CELL_METERS < Math.sqrt(bestDistance));
      ring++
    ) {
      for (let dx = -ring; dx <= ring; dx++) {
        for (let dy = -ring; dy <= ring; dy++) {
          if (Math.max(Math.abs(dx), Math.abs(dy)) !== ring) continue;
          const bucket = nodeGrid.get(key(cellX + dx, cellY + dy));
          if (!bucket) continue;
          for (const node of bucket) {
            const distance =
              (nodePositions[node * 2] - x) ** 2 + (nodePositions[node * 2 + 1] - y) ** 2;
            if (distance < bestDistance) {
              bestDistance = distance;
              best = node;
            }
          }
        }
      }
    }
    return Math.max(best, 0);
  };

  // Segment grid for edge picking: every segment is registered in the cells along it.
  const segmentGrid = new Map<number, number[]>();
  for (let segment = 0; segment < segmentCount; segment++) {
    const x0 = segments[segment * 4];
    const y0 = segments[segment * 4 + 1];
    const x1 = segments[segment * 4 + 2];
    const y1 = segments[segment * 4 + 3];
    const steps = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / (GRID_CELL_METERS / 2)));
    let lastKey = -1;
    for (let step = 0; step <= steps; step++) {
      const t = step / steps;
      const cellKey = key(
        Math.floor((x0 + (x1 - x0) * t - minX) / GRID_CELL_METERS),
        Math.floor((y0 + (y1 - y0) * t - minY) / GRID_CELL_METERS)
      );
      if (cellKey === lastKey) continue;
      lastKey = cellKey;
      const bucket = segmentGrid.get(cellKey);
      if (bucket) bucket.push(segment);
      else segmentGrid.set(cellKey, [segment]);
    }
  }
  const findNearestEdge = (x: number, y: number) => {
    const cellX = Math.floor((x - minX) / GRID_CELL_METERS);
    const cellY = Math.floor((y - minY) / GRID_CELL_METERS);
    let best = 0;
    let bestDistance = Infinity;
    for (
      let ring = 0;
      ring < 40 && (bestDistance === Infinity || ring * GRID_CELL_METERS < bestDistance);
      ring++
    ) {
      for (let dx = -ring; dx <= ring; dx++) {
        for (let dy = -ring; dy <= ring; dy++) {
          if (Math.max(Math.abs(dx), Math.abs(dy)) !== ring) continue;
          const bucket = segmentGrid.get(key(cellX + dx, cellY + dy));
          if (!bucket) continue;
          for (const segment of bucket) {
            const distance = pointSegmentDistance(
              x,
              y,
              segments[segment * 4],
              segments[segment * 4 + 1],
              segments[segment * 4 + 2],
              segments[segment * 4 + 3]
            );
            if (distance < bestDistance) {
              bestDistance = distance;
              best = segmentEdges[segment];
            }
          }
        }
      }
    }
    return {edge: best, distance: bestDistance};
  };

  let directedMeters = 0;
  for (let edge = 0; edge < edgeCount; edge++) directedMeters += length[edge];

  return {
    origin,
    projection,
    nodeCount,
    edgeCount,
    nodePositions,
    nodeLngLat,
    offsets,
    sources,
    targets,
    length,
    speedKmh,
    roadClass,
    travelTime,
    reverse,
    segments,
    segmentEdges,
    segmentTargetNodes,
    segmentSourceNodes,
    majorSegments: Float32Array.from(major),
    findNearestNode,
    findNearestEdge,
    streetKilometers: directedMeters / 2000
  };
}

/** Distance from a point to a segment, in the same units as the inputs. */
export function pointSegmentDistance(
  px: number,
  py: number,
  x0: number,
  y0: number,
  x1: number,
  y1: number
): number {
  const dx = x1 - x0;
  const dy = y1 - y0;
  const lengthSquared = dx * dx + dy * dy;
  const t =
    lengthSquared > 0
      ? Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / lengthSquared))
      : 0;
  return Math.hypot(px - (x0 + dx * t), py - (y0 + dy * t));
}

/** Options of {@link writeDriveCosts}. */
export type DriveCostSettings = {
  /** Multiplier on motorway and trunk travel time (1 = free flow, 3 = heavy congestion). */
  expresswaySlowdown: number;
  /** Seconds added for crossing each intersection (every edge ends at one). */
  intersectionDelay: number;
  /** Closes motorway and trunk edges (negative cost = impassable). */
  closeExpressways: boolean;
  /** Multiplier on the travel time of every road (1 = free flow). Defaults to 1. */
  allRoadsSlowdown?: number;
};

/**
 * Writes the per-edge drive cost in seconds into `target` (CSR order): free-flow travel time,
 * a slowdown on expressways, a fixed delay per intersection, and closures as -1. Costs are
 * floored above zero so equal-cost plateaus stay rare.
 */
export function writeDriveCosts(
  network: RoadNetwork,
  settings: DriveCostSettings,
  target: Float32Array
): void {
  for (let edge = 0; edge < network.edgeCount; edge++) {
    const expressway = network.roadClass[edge] <= 1;
    if (expressway && settings.closeExpressways) {
      target[edge] = -1;
      continue;
    }
    const time =
      network.travelTime[edge] *
      (expressway ? settings.expresswaySlowdown : 1) *
      (settings.allRoadsSlowdown ?? 1);
    target[edge] = Math.max(0.1, time + settings.intersectionDelay);
  }
}

/** Signed turn angle in radians from edge `from` to edge `to`, positive to the left. */
export function getTurnAngle(network: RoadNetwork, from: number, to: number): number {
  const ax =
    network.nodePositions[network.targets[from] * 2] -
    network.nodePositions[network.sources[from] * 2];
  const ay =
    network.nodePositions[network.targets[from] * 2 + 1] -
    network.nodePositions[network.sources[from] * 2 + 1];
  const bx =
    network.nodePositions[network.targets[to] * 2] - network.nodePositions[network.sources[to] * 2];
  const by =
    network.nodePositions[network.targets[to] * 2 + 1] -
    network.nodePositions[network.sources[to] * 2 + 1];
  return Math.atan2(ax * by - ay * bx, ax * bx + ay * by);
}
