// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {NetworkCSR} from '../network-reachability/network-reachability-oracle';

const HEADER_LENGTH = 16;
const WORKGROUP_SIZE = 256;

export type NetworkStatisticsOracleOptions = {
  nodeCount: number;
  csr: Pick<NetworkCSR, 'offsets' | 'neighbors'>;
  directed?: boolean;
  countSelfLoopsTwice?: boolean;
  vertexMask?: ArrayLike<number>;
  edgeMask?: ArrayLike<number>;
  communities?: ArrayLike<number>;
  resolution?: number;
  binWidth?: number;
  binCount?: number;
  binning?: 'linear' | 'log2';
};

/** Degree bin of one degree under the contributor's two rules. */
export function getDegreeBin(
  degree: number,
  binCount: number,
  binning: 'linear' | 'log2',
  binWidth: number
): number {
  if (binning === 'log2') {
    return degree === 0 ? 0 : Math.min(1 + Math.floor(Math.log2(degree)), binCount - 1);
  }
  return Math.min(Math.floor(degree / Math.max(1, binWidth)), binCount - 1);
}

/** CPU mirror of the GPU summary, including the f32 structure of the modularity finish. */
export function computeNetworkStatisticsOracle(
  options: NetworkStatisticsOracleOptions
): Uint32Array {
  const {nodeCount, csr, vertexMask, edgeMask, communities} = options;
  const directed = Boolean(options.directed);
  const countTwice = Boolean(options.countSelfLoopsTwice) && !directed;
  const binCount = options.binCount ?? 32;
  const binning = options.binning ?? 'linear';
  const binWidth = options.binWidth ?? 1;
  const words = new Uint32Array(HEADER_LENGTH + 3 * binCount);
  const isLive = (vertex: number) => !vertexMask || vertexMask[vertex] !== 0;

  const outDegree = new Uint32Array(nodeCount);
  const inDegree = new Uint32Array(nodeCount);
  const parent = Array.from({length: nodeCount}, (_, vertex) => vertex);
  const find = (vertex: number): number => {
    while (parent[vertex] !== vertex) {
      parent[vertex] = parent[parent[vertex]];
      vertex = parent[vertex];
    }
    return vertex;
  };
  let liveSlots = 0;
  let selfLoops = 0;
  let intra = 0;
  for (let vertex = 0; vertex < nodeCount; vertex++) {
    for (let slot = csr.offsets[vertex]; slot < csr.offsets[vertex + 1]; slot++) {
      const target = csr.neighbors[slot];
      const live =
        isLive(vertex) &&
        target < nodeCount &&
        isLive(target) &&
        (!edgeMask || edgeMask[slot] !== 0);
      if (!live) {
        continue;
      }
      liveSlots++;
      outDegree[vertex]++;
      inDegree[target]++;
      if (target === vertex) {
        selfLoops++;
        if (countTwice) {
          outDegree[vertex]++;
        }
      }
      if (communities && communities[vertex] === communities[target]) {
        intra++;
      }
      parent[find(vertex)] = find(target);
    }
  }

  let liveVertices = 0;
  let isolated = 0;
  let maxOut = 0;
  let maxIn = 0;
  let maxTotal = 0;
  const componentSizes = new Map<number, number>();
  let invalidLabels = 0;
  const communityOut = new Map<number, number>();
  const communityIn = new Map<number, number>();
  for (let vertex = 0; vertex < nodeCount; vertex++) {
    if (!isLive(vertex)) {
      continue;
    }
    liveVertices++;
    const outValue = outDegree[vertex];
    const inValue = directed ? inDegree[vertex] : outValue;
    const totalValue = directed ? outValue + inValue : outValue;
    if (totalValue === 0) {
      isolated++;
    }
    maxOut = Math.max(maxOut, outValue);
    maxIn = Math.max(maxIn, inValue);
    maxTotal = Math.max(maxTotal, totalValue);
    words[HEADER_LENGTH + getDegreeBin(outValue, binCount, binning, binWidth)]++;
    words[HEADER_LENGTH + binCount + getDegreeBin(inValue, binCount, binning, binWidth)]++;
    words[HEADER_LENGTH + 2 * binCount + getDegreeBin(totalValue, binCount, binning, binWidth)]++;
    const root = find(vertex);
    componentSizes.set(root, (componentSizes.get(root) ?? 0) + 1);
    if (communities) {
      const label = communities[vertex];
      if (label >= nodeCount) {
        invalidLabels++;
      } else {
        communityOut.set(label, (communityOut.get(label) ?? 0) + outValue);
        communityIn.set(label, (communityIn.get(label) ?? 0) + inValue);
      }
    }
  }

  let modularity = 0;
  const valid = Boolean(communities) && invalidLabels === 0 && liveSlots > 0;
  if (valid) {
    // Mirror the single-workgroup finish: strided per-thread f32 sums, then a binary tree.
    const slotCount = Math.fround(liveSlots + (countTwice ? selfLoops : 0));
    const partial = new Float32Array(WORKGROUP_SIZE);
    for (let label = 0; label < nodeCount; label++) {
      const thread = label % WORKGROUP_SIZE;
      const outTerm = Math.fround((communityOut.get(label) ?? 0) / slotCount);
      const inTerm = Math.fround(
        (directed ? (communityIn.get(label) ?? 0) : (communityOut.get(label) ?? 0)) / slotCount
      );
      partial[thread] = Math.fround(partial[thread] + Math.fround(outTerm * inTerm));
    }
    for (let stride = WORKGROUP_SIZE / 2; stride > 0; stride >>= 1) {
      for (let thread = 0; thread < stride; thread++) {
        partial[thread] = Math.fround(partial[thread] + partial[thread + stride]);
      }
    }
    const resolution = Math.fround(options.resolution ?? 1);
    modularity = Math.fround(
      Math.fround((intra + (countTwice ? selfLoops : 0)) / slotCount) -
        Math.fround(resolution * partial[0])
    );
  }

  words[0] = liveVertices;
  words[1] = directed ? liveSlots : (liveSlots - selfLoops) / 2 + selfLoops;
  words[2] = liveSlots;
  words[3] = componentSizes.size;
  words[4] = Math.max(0, ...componentSizes.values());
  words[5] = 1;
  words[6] = isolated;
  words[7] = maxOut;
  words[8] = maxIn;
  words[9] = maxTotal;
  new Float32Array(words.buffer, 40, 1)[0] = modularity;
  words[11] = valid ? 1 : 0;
  words[12] = intra;
  words[13] = selfLoops;
  return words;
}
