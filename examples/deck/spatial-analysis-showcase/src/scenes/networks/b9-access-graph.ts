// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {LoadedDataset} from '../../data/catalog';
import {NO_EDGE, type RoadNetwork} from './b9-road-network';

/** Edge kinds of the multimodal access graph. */
export const EDGE_WALK = 0;
export const EDGE_CONNECTOR = 1;
export const EDGE_BOARD = 2;
export const EDGE_ALIGHT = 3;
export const EDGE_HOP = 4;

/** Seconds of service the GTFS weekday trip counts are spread over. */
const SERVICE_SPAN_SECONDS = 18 * 3600;

/**
 * Walk plus transit network of Chicago in one CSR, stored reversed (an edge `u -> v` of the trip
 * is stored as `v -> u`) so a search from an opportunity gives the cost from every node to it.
 *
 * Node layout: road nodes `0..N-1`, then one *walk* node per transit stop (where you stand), then
 * one *vehicle* node per stop (where you ride). Boarding goes walk to vehicle and costs the
 * expected wait; riding goes vehicle to vehicle along the GTFS hop graph; alighting is free.
 */
export type AccessGraph = {
  nodeCount: number;
  edgeCount: number;
  roadNodeCount: number;
  stopCount: number;
  /** Reversed CSR offsets. */
  offsets: Uint32Array;
  /** Reversed CSR targets. */
  neighbors: Uint32Array;
  kind: Uint8Array;
  /** Meters for walking edges and connectors, seconds for hops, stop id for board edges. */
  base: Float32Array;
  /** Stop of each edge (board edges), else 0. */
  edgeStop: Uint32Array;
  /** Mode of each hop edge: 0 bus, 1 rail. */
  edgeMode: Uint8Array;
  /** Road walking edges as COO for snapping (both directions), planar nodes of the road graph. */
  roadSources: Uint32Array;
  roadTargets: Uint32Array;
  roadMeters: Float32Array;
  /** Stop positions in meters and per-stop attributes. */
  stopPositions: Float32Array;
  stopMode: Uint8Array;
  stopWeekdayTrips: Uint16Array;
  /** Expected waiting time at each stop in seconds at a wait factor of 1. */
  stopWait: Float32Array;
  /** Nearest road node of each stop. */
  stopRoadNode: Uint32Array;
  /** Meters walked between each stop and its road node. */
  stopAccessMeters: Float32Array;
};

/** Settings that turn the structure of the access graph into edge weights in seconds. */
export type AccessCostSettings = {
  walkSpeed: number;
  transit: boolean;
  waitFactor: number;
};

/** Builds the reversed multimodal CSR from the road network and the CTA stop and hop tables. */
export function buildAccessGraph(network: RoadNetwork, transit: LoadedDataset): AccessGraph {
  const roadNodeCount = network.nodeCount;
  const stopPositions = transit.projectColumn('stopPosition', network.origin);
  const stopCount = stopPositions.length / 2;
  const stopMode = transit.column<Uint8Array>('stopMode');
  const stopWeekdayTrips = transit.column<Uint16Array>('stopWeekdayTrips');
  const hopSource = transit.column<Uint32Array>('hopSource');
  const hopTarget = transit.column<Uint32Array>('hopTarget');
  const hopSeconds = transit.column<Float32Array>('hopSeconds');
  const hopMode = transit.column<Uint8Array>('hopMode');
  const nodeCount = roadNodeCount + 2 * stopCount;

  const sources: number[] = [];
  const targets: number[] = [];
  const kinds: number[] = [];
  const bases: number[] = [];
  const stops: number[] = [];
  const modes: number[] = [];
  const roadSources: number[] = [];
  const roadTargets: number[] = [];
  const roadMeters: number[] = [];
  const add = (from: number, to: number, kind: number, base: number, stop = 0, mode = 0) => {
    // Stored reversed.
    sources.push(to);
    targets.push(from);
    kinds.push(kind);
    bases.push(base);
    stops.push(stop);
    modes.push(mode);
  };

  // Walking on every street except expressways; one-way streets are walkable both ways.
  for (let edge = 0; edge < network.edgeCount; edge++) {
    if (network.roadClass[edge] <= 1) continue;
    const from = network.sources[edge];
    const to = network.targets[edge];
    add(from, to, EDGE_WALK, network.length[edge]);
    roadSources.push(from);
    roadTargets.push(to);
    roadMeters.push(network.length[edge]);
    if (network.reverse[edge] === NO_EDGE) {
      add(to, from, EDGE_WALK, network.length[edge]);
      roadSources.push(to);
      roadTargets.push(from);
      roadMeters.push(network.length[edge]);
    }
  }

  const stopWait = new Float32Array(stopCount);
  const stopRoadNode = new Uint32Array(stopCount);
  const stopAccessMeters = new Float32Array(stopCount);
  for (let stop = 0; stop < stopCount; stop++) {
    const x = stopPositions[stop * 2];
    const y = stopPositions[stop * 2 + 1];
    const node = network.findNearestNode(x, y);
    const meters =
      Math.hypot(network.nodePositions[node * 2] - x, network.nodePositions[node * 2 + 1] - y) + 10;
    stopRoadNode[stop] = node;
    stopAccessMeters[stop] = meters;
    const walkNode = roadNodeCount + stop;
    const vehicleNode = roadNodeCount + stopCount + stop;
    add(node, walkNode, EDGE_CONNECTOR, meters);
    add(walkNode, node, EDGE_CONNECTOR, meters);
    // Expected wait: half the average headway of all departures from the stop.
    const trips = Math.max(stopWeekdayTrips[stop], 1);
    stopWait[stop] = Math.min(900, Math.max(45, (0.5 * SERVICE_SPAN_SECONDS) / trips));
    add(walkNode, vehicleNode, EDGE_BOARD, 0, stop);
    add(vehicleNode, walkNode, EDGE_ALIGHT, 0, stop);
  }
  for (let hop = 0; hop < hopSource.length; hop++) {
    add(
      roadNodeCount + stopCount + hopSource[hop],
      roadNodeCount + stopCount + hopTarget[hop],
      EDGE_HOP,
      Math.max(hopSeconds[hop], 5),
      0,
      hopMode[hop]
    );
  }

  // Counting sort by (reversed) source into CSR order.
  const edgeCount = sources.length;
  const offsets = new Uint32Array(nodeCount + 1);
  for (let edge = 0; edge < edgeCount; edge++) offsets[sources[edge] + 1]++;
  for (let node = 0; node < nodeCount; node++) offsets[node + 1] += offsets[node];
  const cursor = offsets.slice(0, nodeCount);
  const neighbors = new Uint32Array(edgeCount);
  const kind = new Uint8Array(edgeCount);
  const base = new Float32Array(edgeCount);
  const edgeStop = new Uint32Array(edgeCount);
  const edgeMode = new Uint8Array(edgeCount);
  for (let edge = 0; edge < edgeCount; edge++) {
    const slot = cursor[sources[edge]]++;
    neighbors[slot] = targets[edge];
    kind[slot] = kinds[edge];
    base[slot] = bases[edge];
    edgeStop[slot] = stops[edge];
    edgeMode[slot] = modes[edge];
  }
  return {
    nodeCount,
    edgeCount,
    roadNodeCount,
    stopCount,
    offsets,
    neighbors,
    kind,
    base,
    edgeStop,
    edgeMode,
    roadSources: Uint32Array.from(roadSources),
    roadTargets: Uint32Array.from(roadTargets),
    roadMeters: Float32Array.from(roadMeters),
    stopPositions,
    stopMode,
    stopWeekdayTrips,
    stopWait,
    stopRoadNode,
    stopAccessMeters
  };
}

/** Writes edge weights in seconds: walking at `walkSpeed` m/s, expected waits, and in-vehicle time. */
export function writeAccessWeights(
  graph: AccessGraph,
  settings: AccessCostSettings,
  target: Float32Array
): void {
  for (let edge = 0; edge < graph.edgeCount; edge++) {
    switch (graph.kind[edge]) {
      case EDGE_WALK:
      case EDGE_CONNECTOR:
        target[edge] = Math.max(0.5, graph.base[edge] / settings.walkSpeed);
        break;
      case EDGE_BOARD:
        target[edge] = settings.transit
          ? Math.max(1, graph.stopWait[graph.edgeStop[edge]] * settings.waitFactor)
          : -1;
        break;
      case EDGE_ALIGHT:
        target[edge] = settings.transit ? 5 : -1;
        break;
      default:
        target[edge] = settings.transit ? graph.base[edge] : -1;
    }
  }
}

/** Walking edge costs in seconds for the snapping edge set (road edges, both directions). */
export function writeRoadWalkCosts(
  graph: AccessGraph,
  walkSpeed: number,
  target: Float32Array
): void {
  for (let edge = 0; edge < graph.roadMeters.length; edge++) {
    target[edge] = Math.max(0.5, graph.roadMeters[edge] / walkSpeed);
  }
}
