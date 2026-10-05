// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** One directed CSR slot source: `[from, to, weight]`. */
export type CoarseningEdge = readonly [from: number, to: number, weight: number];

/** Forward CSR with per-slot weights. */
export type CoarseningCSR = {
  offsets: Uint32Array;
  neighbors: Uint32Array;
  weights: Float32Array;
};

/** Builds a forward CSR; slots of one row keep input order. */
export function buildCoarseningCSR(
  nodeCount: number,
  edges: readonly CoarseningEdge[]
): CoarseningCSR {
  const rows: CoarseningEdge[][] = Array.from({length: nodeCount}, () => []);
  for (const edge of edges) rows[edge[0]].push(edge);
  const offsets = new Uint32Array(nodeCount + 1);
  for (let row = 0; row < nodeCount; row++) offsets[row + 1] = offsets[row] + rows[row].length;
  const slots = rows.flat();
  return {
    offsets,
    neighbors: Uint32Array.from(slots, edge => edge[1]),
    weights: Float32Array.from(slots, edge => edge[2])
  };
}

/** Seeded linear congruential generator in `[0, 1)`. */
export function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
}

/** Random multigraph edges; undirected graphs list both directions (self-loops once). */
export function createRandomCoarseningEdges(
  seed: number,
  nodeCount: number,
  edgeCount: number,
  directed: boolean
): CoarseningEdge[] {
  const random = createRandom(seed);
  const edges: CoarseningEdge[] = [];
  for (let index = 0; index < edgeCount; index++) {
    const from = Math.floor(random() * nodeCount);
    const to = random() < 0.05 ? from : Math.floor(random() * nodeCount);
    const weight = Math.round((random() * 8 - 2) * 4) / 4;
    edges.push([from, to, weight]);
    if (!directed && from !== to) edges.push([to, from, weight]);
  }
  return edges;
}

export type CoarseningOracleOptions = {
  nodeCount: number;
  csr: CoarseningCSR;
  labels: Uint32Array;
  groupCapacity: number;
  directed?: boolean;
  vertexMask?: Uint32Array;
  edgeMask?: Uint32Array;
  /** Interleaved x, y. */
  positions?: Float32Array;
  vertexValues?: Float32Array;
  /** Use unit weights. */
  unweighted?: boolean;
  fixedPointScale?: number;
  edgeCapacity: number;
};

export type CoarseningOracleResult = {
  groupVertexCount: Uint32Array;
  groupIntraEdgeCount: Uint32Array;
  /** Exact fixed-point sums converted with the GPU's float32 steps. */
  groupIntraWeight: Float32Array;
  groupCentroid: Float32Array;
  groupBounds: Float32Array;
  groupValueSum: Float32Array;
  /** All superedges sorted by key, unclamped. */
  edges: {source: number; target: number; count: number; weight: number}[];
  summary: number[];
  overflow: boolean;
};

const TWO_64 = 1n << 64n;
const fround = Math.fround;

function roundHalfEven(value: number): number {
  const floor = Math.floor(value);
  const difference = value - floor;
  if (difference < 0.5) return floor;
  if (difference > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

/** Mirrors the WGSL `quantize`: sign-magnitude float32 rounding to a signed integer. */
export function quantize(value: number, scale: number): bigint {
  const magnitude = Math.min(roundHalfEven(fround(Math.abs(value) * scale)), fround(4.0e18));
  const quantized = BigInt(magnitude);
  return value < 0 ? -quantized : quantized;
}

/** Mirrors the WGSL `dequantize` of a wrapped 64-bit sum, including its float32 rounding steps. */
export function dequantize(sum: bigint, scale: number): number {
  const wrapped = BigInt.asUintN(64, sum);
  const negative = wrapped >> 63n === 1n;
  const magnitude = negative ? (TWO_64 - wrapped) % TWO_64 : wrapped;
  const high = Number(magnitude >> 32n);
  const low = Number(magnitude & 0xffffffffn);
  const result = fround(fround(fround(high) * 4294967296) + fround(low));
  return fround((negative ? -result : result) / scale);
}

/** CPU reference with the same liveness, grouping, ordering and fixed-point rules as the recipe. */
export function computeCoarseningOracle(options: CoarseningOracleOptions): CoarseningOracleResult {
  const {nodeCount, csr, labels, groupCapacity, directed, edgeCapacity} = options;
  const scale = options.fixedPointScale ?? 65536;
  const DEAD = -1;
  const OVER = -2;
  const effective = Array.from({length: nodeCount}, (_, vertex) => {
    if (options.vertexMask && options.vertexMask[vertex] === 0) return DEAD;
    return labels[vertex] >= groupCapacity ? OVER : labels[vertex];
  });
  const count = new Uint32Array(groupCapacity);
  const sums = Array.from({length: groupCapacity}, () => [0n, 0n, 0n]);
  const bounds = Array.from({length: groupCapacity}, () => [
    Infinity,
    Infinity,
    -Infinity,
    -Infinity
  ]);
  const summary = new Array(8).fill(0);
  for (let vertex = 0; vertex < nodeCount; vertex++) {
    const label = effective[vertex];
    if (label === DEAD) continue;
    summary[0]++;
    if (label === OVER) {
      summary[1]++;
      continue;
    }
    if (count[label]++ === 0) summary[6]++;
    if (options.positions) {
      const x = options.positions[2 * vertex];
      const y = options.positions[2 * vertex + 1];
      sums[label][0] += quantize(x, scale);
      sums[label][1] += quantize(y, scale);
      const box = bounds[label];
      box[0] = Math.min(box[0], x);
      box[1] = Math.min(box[1], y);
      box[2] = Math.max(box[2], x);
      box[3] = Math.max(box[3], y);
    }
    if (options.vertexValues) sums[label][2] += quantize(options.vertexValues[vertex], scale);
  }
  const intraCount = new Uint32Array(groupCapacity);
  const intraSum = new Array<bigint>(groupCapacity).fill(0n);
  const pairs = new Map<number, {count: number; sum: bigint}>();
  for (let source = 0; source < nodeCount; source++) {
    for (let slot = csr.offsets[source]; slot < csr.offsets[source + 1]; slot++) {
      const target = csr.neighbors[slot];
      if (effective[source] === DEAD || target >= nodeCount || effective[target] === DEAD) continue;
      if (options.edgeMask && options.edgeMask[slot] === 0) continue;
      if (!directed && source > target) continue;
      summary[2]++;
      const a = effective[source];
      const b = effective[target];
      if (a === OVER || b === OVER) {
        summary[3]++;
        continue;
      }
      const weight = quantize(options.unweighted ? 1 : csr.weights[slot], scale);
      if (a === b) {
        summary[4]++;
        intraCount[a]++;
        intraSum[a] += weight;
      } else {
        summary[5]++;
        const key = directed
          ? a * groupCapacity + b
          : Math.min(a, b) * groupCapacity + Math.max(a, b);
        const entry = pairs.get(key) ?? {count: 0, sum: 0n};
        entry.count++;
        entry.sum += weight;
        pairs.set(key, entry);
      }
    }
  }
  const edges = [...pairs.entries()]
    .sort((left, right) => left[0] - right[0])
    .map(([key, entry]) => ({
      source: Math.floor(key / groupCapacity),
      target: key % groupCapacity,
      count: entry.count,
      weight: dequantize(entry.sum, scale)
    }));
  summary[7] = edges.length;
  const centroid = new Float32Array(2 * groupCapacity);
  const boundsOut = new Float32Array(4 * groupCapacity);
  const valueSum = new Float32Array(groupCapacity);
  const intraWeight = new Float32Array(groupCapacity);
  for (let group = 0; group < groupCapacity; group++) {
    if (count[group] > 0) {
      centroid[2 * group] = fround(dequantize(sums[group][0], scale) / count[group]);
      centroid[2 * group + 1] = fround(dequantize(sums[group][1], scale) / count[group]);
      if (options.positions) boundsOut.set(bounds[group], 4 * group);
    }
    valueSum[group] = dequantize(sums[group][2], scale);
    intraWeight[group] = dequantize(intraSum[group], scale);
  }
  return {
    groupVertexCount: count,
    groupIntraEdgeCount: intraCount,
    groupIntraWeight: intraWeight,
    groupCentroid: centroid,
    groupBounds: boundsOut,
    groupValueSum: valueSum,
    edges,
    summary,
    overflow: summary[1] > 0 || summary[3] > 0 || edges.length > edgeCapacity
  };
}
