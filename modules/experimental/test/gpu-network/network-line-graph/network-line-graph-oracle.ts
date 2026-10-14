// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUNetworkLineGraphSettings} from '../../../src/gpu-network/network-line-graph/index';
import {getHaversineDistance, getInitialBearing} from '../network-metric-oracle';

/** Directed CSR of a base network or of its line graph. */
export type LineGraphCSR = {
  offsets: Uint32Array;
  neighbors: Uint32Array;
  weights: Float32Array;
};

/** Signed turn angle from arrival direction `directionA` to departure `directionB`, left positive. */
export function getLineGraphTurnAngle(
  directionA: readonly [number, number],
  directionB: readonly [number, number]
): number {
  const dotA = directionA[0] ** 2 + directionA[1] ** 2;
  const dotB = directionB[0] ** 2 + directionB[1] ** 2;
  return dotA < 0.5 || dotB < 0.5
    ? 0
    : Math.atan2(
        directionA[0] * directionB[1] - directionA[1] * directionB[0],
        directionA[0] * directionB[0] + directionA[1] * directionB[1]
      );
}

/**
 * Unit departure (at the tail) and arrival (at the head) direction of every edge, `(east, north)`.
 * Planar edges use one f32-rounded direction for both; great-circle edges use the initial bearing
 * from the tail and the reversed initial bearing from the head back to the tail.
 */
export function getLineGraphEdgeDirections(
  base: LineGraphCSR,
  positions: ArrayLike<number>,
  isGreatCircle = false
): {departures: [number, number][]; arrivals: [number, number][]} {
  const nodeCount = base.offsets.length - 1;
  const edgeCount = base.neighbors.length;
  const departures: [number, number][] = new Array(edgeCount);
  const arrivals: [number, number][] = new Array(edgeCount);
  for (let node = 0; node < nodeCount; node++) {
    for (let edge = base.offsets[node]; edge < base.offsets[node + 1]; edge++) {
      const head = base.neighbors[edge];
      if (isGreatCircle) {
        const tail = [positions[node * 2], positions[node * 2 + 1]];
        const headPosition = [positions[head * 2], positions[head * 2 + 1]];
        if (head < nodeCount && getHaversineDistance(tail, headPosition) > 0) {
          const departure = getInitialBearing(tail, headPosition);
          const back = getInitialBearing(headPosition, tail);
          departures[edge] = [Math.sin(departure), Math.cos(departure)];
          arrivals[edge] = [-Math.sin(back), -Math.cos(back)];
        } else {
          departures[edge] = [0, 0];
          arrivals[edge] = [0, 0];
        }
        continue;
      }
      const dx = Math.fround(positions[head * 2] - positions[node * 2]);
      const dy = Math.fround(positions[head * 2 + 1] - positions[node * 2 + 1]);
      const length = Math.hypot(dx, dy);
      departures[edge] = length > 0 ? [dx / length, dy / length] : [0, 0];
      arrivals[edge] = departures[edge];
    }
  }
  return {departures, arrivals};
}

/** Turn cost of `from -> to` from the arrival and departure directions, or -1 for a banned U-turn. */
function getTurnCost(
  directionA: readonly [number, number],
  directionB: readonly [number, number],
  settings: Required<GPUNetworkLineGraphSettings>
): number {
  const angle = getLineGraphTurnAngle(directionA, directionB);
  const magnitude = Math.abs(angle);
  if (magnitude >= settings.uTurnAngle) {
    return settings.uTurnCost;
  }
  let cost = settings.angleCost * magnitude;
  if (magnitude > settings.straightAngle) {
    cost += angle > 0 ? settings.leftTurnCost : settings.rightTurnCost;
  }
  return cost;
}

/**
 * CPU line graph: line node per base edge row, arcs to every following edge of the head in
 * ascending row order, minus banned turns and banned U-turns, with arc cost `weight + turn cost`.
 * Rows past `arcCapacity` are truncated like the GPU contributor. Returns the CSR before
 * truncation clamps and the total arc count. `isGreatCircle` takes longitude/latitude positions.
 */
export function lineGraphOracle(
  base: LineGraphCSR,
  positions: ArrayLike<number>,
  bannedTurns: readonly (readonly [number, number])[],
  settings: GPUNetworkLineGraphSettings,
  arcCapacity: number,
  isGreatCircle = false
): {csr: LineGraphCSR; totalArcs: number} {
  const resolved: Required<GPUNetworkLineGraphSettings> = {
    angleCost: settings.angleCost ?? 0,
    leftTurnCost: settings.leftTurnCost ?? 0,
    rightTurnCost: settings.rightTurnCost ?? 0,
    uTurnCost: settings.uTurnCost ?? -1,
    straightAngle: settings.straightAngle ?? 0.5,
    uTurnAngle: settings.uTurnAngle ?? 2.9,
    bannedTurnCount: settings.bannedTurnCount ?? bannedTurns.length
  };
  const nodeCount = base.offsets.length - 1;
  const edgeCount = base.neighbors.length;
  const {departures, arrivals} = getLineGraphEdgeDirections(base, positions, isGreatCircle);
  const active = bannedTurns.slice(0, resolved.bannedTurnCount);
  const rows: number[][] = [];
  const weights: number[][] = [];
  for (let edge = 0; edge < edgeCount; edge++) {
    const row: number[] = [];
    const rowWeights: number[] = [];
    const head = base.neighbors[edge];
    if (head < nodeCount) {
      for (let next = base.offsets[head]; next < base.offsets[head + 1]; next++) {
        const banned = active.some(([from, to]) => from === edge && to === next);
        const turnCost = getTurnCost(arrivals[edge], departures[next], resolved);
        if (!banned && turnCost >= 0) {
          row.push(next);
          const baseCost = base.weights[next];
          rowWeights.push(baseCost >= 0 ? baseCost + Math.max(turnCost, 0) : -1);
        }
      }
    }
    rows.push(row);
    weights.push(rowWeights);
  }
  const offsets = new Uint32Array(edgeCount + 1);
  for (let edge = 0; edge < edgeCount; edge++) {
    offsets[edge + 1] = offsets[edge] + rows[edge].length;
  }
  const totalArcs = offsets[edgeCount];
  const neighbors = new Uint32Array(arcCapacity);
  const arcWeights = new Float32Array(arcCapacity);
  let slot = 0;
  for (let edge = 0; edge < edgeCount; edge++) {
    rows[edge].forEach((next, index) => {
      if (slot < arcCapacity) {
        neighbors[slot] = next;
        arcWeights[slot] = weights[edge][index];
      }
      slot++;
    });
  }
  return {
    csr: {
      offsets: offsets.map(value => Math.min(value, arcCapacity)),
      neighbors,
      weights: arcWeights
    },
    totalArcs
  };
}
