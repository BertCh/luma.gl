// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {splitTimeWords, joinTimeWords} from '../../../src/gpu-dataframe/time-window-filter/time-words';

type Range = readonly [number, number] | null | undefined;

export type SubgraphOracleOptions = {
  nodeCount: number;
  offsets: Uint32Array;
  neighbors: Uint32Array;
  directed?: boolean;
  vertexMask?: ArrayLike<number>;
  edgeMask?: ArrayLike<number>;
  vertexColumns?: readonly Float32Array[];
  vertexRanges?: readonly Range[];
  edgeColumns?: readonly Float32Array[];
  edgeRanges?: readonly Range[];
  edgeTimes?: Float32Array;
  edgeTimeWindow?: Range;
  /** Int64 slot times. */
  edgeTimeWords?: BigInt64Array;
  timeWordWindow?: {start: number | bigint; end: number | bigint};
  /** Undirected only; default true. See `GPUNetworkSubgraphFilterProps.pairUndirectedSlots`. */
  pairUndirectedSlots?: boolean;
  dropIsolated?: boolean;
};

export type SubgraphOracleResult = {
  vertexMask: Uint32Array;
  edgeMask: Uint32Array;
  /** `[liveVertexCount, liveSlotCount, selfLoopSlotCount, liveEdgeCount]`. */
  counts: number[];
  liveVertexIds: number[];
  liveSlotIds: number[];
  inducedOffsets: Uint32Array;
  inducedNeighbors: number[];
  inducedSlots: number[];
};

/** Half-open attribute rule `min <= value < max`; NaN fails. Bounds are rounded to f32. */
function passes(value: number, range: Range): boolean {
  if (!range) return true;
  return value >= Math.fround(range[0]) && value < Math.fround(range[1]);
}

/** CPU reference of {@link GPUNetworkSubgraphFilter}. */
export function computeSubgraphOracle(options: SubgraphOracleOptions): SubgraphOracleResult {
  const {nodeCount, offsets, neighbors} = options;
  const slotCount = neighbors.length;
  const vertexMask = new Uint32Array(nodeCount);
  for (let v = 0; v < nodeCount; v++) {
    let live = options.vertexMask ? options.vertexMask[v] !== 0 : true;
    for (const [i, column] of (options.vertexColumns ?? []).entries()) {
      live = live && passes(column[v], options.vertexRanges?.[i]);
    }
    vertexMask[v] = live ? 1 : 0;
  }
  const own = new Uint32Array(slotCount);
  for (let u = 0; u < nodeCount; u++) {
    for (let slot = offsets[u]; slot < offsets[u + 1]; slot++) {
      let live = options.edgeMask ? options.edgeMask[slot] !== 0 : true;
      for (const [i, column] of (options.edgeColumns ?? []).entries()) {
        live = live && passes(column[slot], options.edgeRanges?.[i]);
      }
      if (options.edgeTimes && options.edgeTimeWindow) {
        const time = options.edgeTimes[slot];
        const [start, end] = options.edgeTimeWindow;
        live = live && time >= Math.fround(start) && time <= Math.fround(end);
      }
      if (options.edgeTimeWords && options.timeWordWindow) {
        const start = splitTimeWords(options.timeWordWindow.start);
        const end = splitTimeWords(options.timeWordWindow.end);
        const startInteger = joinTimeWords(start.low, start.high);
        const endInteger = joinTimeWords(end.low, end.high);
        const time = options.edgeTimeWords[slot];
        live =
          live &&
          (time > startInteger || (time === startInteger && start.fraction === 0)) &&
          time <= endInteger;
      }
      own[slot] = live ? 1 : 0;
    }
  }
  const pairSlots = !options.directed && options.pairUndirectedSlots !== false;
  const edgeMask = new Uint32Array(slotCount);
  for (let u = 0; u < nodeCount; u++) {
    for (let slot = offsets[u]; slot < offsets[u + 1]; slot++) {
      const v = neighbors[slot];
      let live = own[slot] !== 0 && vertexMask[u] !== 0 && v < nodeCount;
      if (live && pairSlots && v !== u) {
        // The j-th slot u -> v pairs with the j-th slot v -> u; a missing reverse keeps own.
        let ordinal = 0;
        for (let earlier = offsets[u]; earlier < slot; earlier++) {
          if (neighbors[earlier] === v) ordinal++;
        }
        let seen = 0;
        for (let reverse = offsets[v]; reverse < offsets[v + 1]; reverse++) {
          if (neighbors[reverse] === u) {
            if (seen === ordinal) {
              live = own[reverse] !== 0;
              break;
            }
            seen++;
          }
        }
      }
      live = live && v < nodeCount && vertexMask[v] !== 0;
      edgeMask[slot] = live ? 1 : 0;
    }
  }
  if (options.dropIsolated) {
    const incident = new Uint8Array(nodeCount);
    for (let u = 0; u < nodeCount; u++) {
      for (let slot = offsets[u]; slot < offsets[u + 1]; slot++) {
        if (edgeMask[slot]) {
          incident[u] = 1;
          incident[neighbors[slot]] = 1;
        }
      }
    }
    for (let v = 0; v < nodeCount; v++) {
      if (!incident[v]) vertexMask[v] = 0;
    }
  }
  let liveSlots = 0;
  let selfLoops = 0;
  const inducedOffsets = new Uint32Array(nodeCount + 1);
  const inducedNeighbors: number[] = [];
  const inducedSlots: number[] = [];
  for (let u = 0; u < nodeCount; u++) {
    for (let slot = offsets[u]; slot < offsets[u + 1]; slot++) {
      if (edgeMask[slot]) {
        liveSlots++;
        if (neighbors[slot] === u) selfLoops++;
        inducedNeighbors.push(neighbors[slot]);
        inducedSlots.push(slot);
      }
    }
    inducedOffsets[u + 1] = inducedNeighbors.length;
  }
  const liveVertexIds = [...vertexMask.keys()].filter(v => vertexMask[v]);
  return {
    vertexMask,
    edgeMask,
    counts: [
      liveVertexIds.length,
      liveSlots,
      selfLoops,
      options.directed ? liveSlots : (liveSlots - selfLoops) / 2 + selfLoops
    ],
    liveVertexIds,
    liveSlotIds: [...edgeMask.keys()].filter(slot => edgeMask[slot]),
    inducedOffsets,
    inducedNeighbors,
    inducedSlots
  };
}
