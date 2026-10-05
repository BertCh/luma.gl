// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {findNearestHexagon} from '../point-density/point-density-oracle';

type Bounds = [number, number, number, number];

/** Zone description mirrored by {@link computeFlowAggregation}. */
export type FlowOracleZones =
  | {kind: 'grid'; bounds: Bounds; gridSize: [number, number]}
  | {
      kind: 'hexagon';
      bounds: Bounds;
      gridSize: [number, number];
      radius: number;
    }
  | {kind: 'ids'; zoneCount: number};

/** Inputs of the CPU flow-aggregation reference. */
export type FlowOracleProps = {
  zones: FlowOracleZones;
  /** Flat `[x0, y0, x1, y1, ...]` origins (grid and hexagon). */
  origins?: number[];
  destinations?: number[];
  originZoneIds?: number[];
  destinationZoneIds?: number[];
  weights?: number[];
  mask?: number[];
  timeWindow?: {timestamps: number[]; start: number; end: number};
  /** Exact Int64 word time gate: accepted when `start <= timestamp <= end`, compared with BigInt. */
  wordTimeWindow?: {timestamps: bigint[]; start: bigint; end: bigint};
  /**
   * `'atomic'` (default) sums in source order in float64; `'tree'` mirrors `sumOrder: 'sorted'`:
   * float32 weights summed per key by a 256-lane strided partial sum plus a fixed binary tree.
   */
  sumMode?: 'atomic' | 'tree';
  excludeSelfFlows?: boolean;
};

/** One distinct (origin, destination) pair. */
export type FlowOracleFlow = {
  pairKey: number;
  originZone: number;
  destinationZone: number;
  count: number;
  weight: number;
};

/** CPU reference results. `flows` lists every distinct pair, sorted. */
export type FlowOracleResult = {
  zoneCount: number;
  flows: FlowOracleFlow[];
  zoneOutCounts: number[];
  zoneInCounts: number[];
  zoneOutWeights: number[];
  zoneInWeights: number[];
};

/** Number of zones of a zone description. */
export function getFlowOracleZoneCount(zones: FlowOracleZones): number {
  return zones.kind === 'ids' ? zones.zoneCount : zones.gridSize[0] * zones.gridSize[1];
}

/** Zone of one position, or `undefined` when rejected. Grid mirrors `GPUGridBinning`. */
export function findFlowZone(zones: FlowOracleZones, x: number, y: number): number | undefined {
  if (zones.kind === 'ids') {
    return undefined;
  }
  const [minX, minY, maxX, maxY] = zones.bounds;
  const [columns, rows] = zones.gridSize;
  if (!Number.isFinite(x) || !Number.isFinite(y) || x < minX || x > maxX || y < minY || y > maxY) {
    return undefined;
  }
  if (zones.kind === 'grid') {
    const column = Math.min(Math.floor(((x - minX) / (maxX - minX || 1)) * columns), columns - 1);
    const row = Math.min(Math.floor(((y - minY) / (maxY - minY || 1)) * rows), rows - 1);
    return row * columns + column;
  }
  const [column, row] = findNearestHexagon(x, y, minX, minY, zones.radius, columns, rows).cell;
  return column < 0 || row < 0 || column >= columns || row >= rows
    ? undefined
    : row * columns + column;
}

/** CPU reference for `GPUFlowAggregation`, independent of the GPU code paths. */
export function computeFlowAggregation(props: FlowOracleProps): FlowOracleResult {
  const {zones} = props;
  const zoneCount = getFlowOracleZoneCount(zones);
  const rowCount =
    zones.kind === 'ids' ? (props.originZoneIds?.length ?? 0) : (props.origins?.length ?? 0) / 2;
  const zoneOutCounts = new Array<number>(zoneCount).fill(0);
  const zoneInCounts = new Array<number>(zoneCount).fill(0);
  const zoneOutWeights = new Array<number>(zoneCount).fill(0);
  const zoneInWeights = new Array<number>(zoneCount).fill(0);
  const pairs = new Map<number, FlowOracleFlow>();
  const treeRows =
    props.sumMode === 'tree'
      ? {
          out: Array.from({length: zoneCount}, () => [] as number[]),
          in: Array.from({length: zoneCount}, () => [] as number[]),
          pair: new Map<number, number[]>()
        }
      : undefined;

  for (let row = 0; row < rowCount; row++) {
    let origin: number | undefined;
    let destination: number | undefined;
    if (zones.kind === 'ids') {
      origin = props.originZoneIds?.[row];
      destination = props.destinationZoneIds?.[row];
      if (origin === undefined || origin >= zoneCount) origin = undefined;
      if (destination === undefined || destination >= zoneCount) destination = undefined;
    } else {
      origin = findFlowZone(zones, props.origins![row * 2], props.origins![row * 2 + 1]);
      destination = findFlowZone(
        zones,
        props.destinations![row * 2],
        props.destinations![row * 2 + 1]
      );
    }
    if (origin === undefined || destination === undefined) continue;
    if (props.excludeSelfFlows && origin === destination) continue;
    if (props.mask && props.mask[row] === 0) continue;
    if (props.timeWindow) {
      const time = props.timeWindow.timestamps[row];
      if (!(time >= props.timeWindow.start && time <= props.timeWindow.end)) continue;
    }
    if (props.wordTimeWindow) {
      const time = props.wordTimeWindow.timestamps[row];
      if (!(time >= props.wordTimeWindow.start && time <= props.wordTimeWindow.end)) continue;
    }
    const weight = props.weights?.[row];
    const finiteWeight = weight !== undefined && Number.isFinite(weight) ? weight : 0;
    zoneOutCounts[origin]++;
    zoneInCounts[destination]++;
    zoneOutWeights[origin] += finiteWeight;
    zoneInWeights[destination] += finiteWeight;
    const pairKey = origin * zoneCount + destination;
    let flow = pairs.get(pairKey);
    if (!flow) {
      flow = {
        pairKey,
        originZone: origin,
        destinationZone: destination,
        count: 0,
        weight: 0
      };
      pairs.set(pairKey, flow);
    }
    flow.count++;
    flow.weight += finiteWeight;
    if (treeRows) {
      treeRows.out[origin].push(finiteWeight);
      treeRows.in[destination].push(finiteWeight);
      treeRows.pair.get(pairKey)?.push(finiteWeight) ?? treeRows.pair.set(pairKey, [finiteWeight]);
    }
  }
  if (treeRows) {
    for (let zone = 0; zone < zoneCount; zone++) {
      zoneOutWeights[zone] = sumFixedTree(treeRows.out[zone]);
      zoneInWeights[zone] = sumFixedTree(treeRows.in[zone]);
    }
    for (const flow of pairs.values()) {
      flow.weight = sumFixedTree(treeRows.pair.get(flow.pairKey)!);
    }
  }

  const useWeights = Boolean(props.weights);
  const flows = [...pairs.values()].sort((left, right) => {
    const difference = useWeights ? right.weight - left.weight : right.count - left.count;
    return difference || left.pairKey - right.pairKey;
  });
  if (!useWeights) {
    for (const flow of flows) flow.weight = flow.count;
  }
  return {
    zoneCount,
    flows,
    zoneOutCounts,
    zoneInCounts,
    zoneOutWeights,
    zoneInWeights
  };
}

/** Deterministic LCG helper returning floats in `[0, 1)`. */
export function createSeededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
}

/**
 * Float32 fixed-tree sum mirroring the GPU segment sum: 256 lanes each add every 256th value in
 * order, then lanes combine by a binary tree with strides 128, 64, ..., 1. Every add is rounded to
 * float32, which is exact IEEE behavior on the GPU.
 */
export function sumFixedTree(values: readonly number[]): number {
  const lanes = new Float32Array(256);
  for (let index = 0; index < values.length; index++) {
    lanes[index % 256] = Math.fround(lanes[index % 256] + Math.fround(values[index]));
  }
  for (let stride = 128; stride > 0; stride >>= 1) {
    for (let lane = 0; lane < stride; lane++) {
      lanes[lane] = Math.fround(lanes[lane] + lanes[lane + stride]);
    }
  }
  return lanes[0];
}
