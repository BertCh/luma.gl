// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/gpgpu/gpu-data';
import type {GPUGraphAdjacency} from '@luma.gl/gpgpu/gpu-graph';
import type {BixiFlows} from './bixi-data';

const SCALAR_BYTES = 4;

/**
 * Creates the caller-owned `GPUVector`s the graph contributors read and write, and destroys them
 * (vectors first, then their buffers). The contributors never allocate their outputs.
 */
export class BixiGraphVectors {
  private readonly buffers: Buffer[] = [];
  private readonly vectors: GPUVector[] = [];

  constructor(
    private readonly device: Device,
    private readonly prefix: string
  ) {}

  /** A packed scalar column, zero filled or initialised from `values`. */
  scalar<Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    length: number,
    values?: Uint32Array | Float32Array
  ): GPUVector<Format> {
    const buffer = this.device.createBuffer({
      id: `${this.prefix}-${name}`,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST,
      ...(values && values.length > 0
        ? {data: values}
        : {byteLength: Math.max(length, 1) * SCALAR_BYTES})
    });
    this.buffers.push(buffer);
    const vector = new GPUVector<Format>({
      type: 'buffer',
      name,
      format,
      buffer,
      length,
      ownsBuffer: false
    });
    this.vectors.push(vector);
    return vector;
  }

  /** A single-chunk edge column the graph borrows; rewrite it with `getBuffer(...).write(...)`. */
  edgeColumn<Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    values: Uint32Array | Float32Array
  ): GPUVector<Format> {
    const buffer = this.device.createBuffer({
      id: `${this.prefix}-${name}`,
      data: values,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    this.buffers.push(buffer);
    const data = new GPUData<Format>({buffer, format, length: values.length, ownsBuffer: false});
    const vector = new GPUVector<Format>({
      type: 'data',
      name,
      format,
      data: [data],
      ownsData: false
    });
    this.vectors.push(vector);
    return vector;
  }

  /** Offsets, neighbors, edge ids, weights and the count and overflow rows of one CSR direction. */
  adjacency(name: string, vertexCount: number, capacity: number): GPUGraphAdjacency {
    return {
      offsets: this.scalar(`${name}-offsets`, 'uint32', vertexCount + 1),
      neighbors: this.scalar(`${name}-neighbors`, 'uint32', capacity),
      edgeIds: this.scalar(`${name}-edge-ids`, 'uint32', capacity),
      edgeWeights: this.scalar(`${name}-edge-weights`, 'float32', capacity),
      count: this.scalar(`${name}-count`, 'uint32', 1),
      overflow: this.scalar(`${name}-overflow`, 'uint32', 1)
    };
  }

  /** The physical buffer behind a single-chunk vector. */
  getBuffer(vector: GPUVector): Buffer {
    return (vector.data[0] as GPUData).buffer as Buffer;
  }

  destroy(): void {
    for (const vector of this.vectors) vector.destroy();
    for (const buffer of this.buffers) buffer.destroy();
    this.vectors.length = 0;
    this.buffers.length = 0;
  }
}

/** Station pairs merged to undirected edges (rides in both directions summed), heaviest first. */
export type UndirectedEdges = {
  count: number;
  a: Uint32Array;
  b: Uint32Array;
  rides: Float32Array;
  /** `[longitude, latitude]` of both ends, four numbers per edge. */
  segments: Float32Array;
  totalRides: number;
};

/** Merges `A -> B` and `B -> A` into one weighted undirected edge. */
export function buildUndirectedEdges(flows: BixiFlows): UndirectedEdges {
  const {pairs, stationCount, lngLat} = flows;
  const merged = new Map<number, number>();
  for (let pair = 0; pair < pairs.count.length; pair++) {
    const from = pairs.origin[pair];
    const to = pairs.destination[pair];
    if (from === to) continue;
    const key = from < to ? from * stationCount + to : to * stationCount + from;
    merged.set(key, (merged.get(key) ?? 0) + pairs.count[pair]);
  }
  const keys = Array.from(merged.keys()).sort((x, y) => merged.get(y)! - merged.get(x)!);
  const count = keys.length;
  const a = new Uint32Array(count);
  const b = new Uint32Array(count);
  const rides = new Float32Array(count);
  const segments = new Float32Array(count * 4);
  let totalRides = 0;
  for (let edge = 0; edge < count; edge++) {
    const key = keys[edge];
    a[edge] = Math.floor(key / stationCount);
    b[edge] = key % stationCount;
    rides[edge] = merged.get(key)!;
    totalRides += rides[edge];
    segments[edge * 4] = lngLat[a[edge] * 2];
    segments[edge * 4 + 1] = lngLat[a[edge] * 2 + 1];
    segments[edge * 4 + 2] = lngLat[b[edge] * 2];
    segments[edge * 4 + 3] = lngLat[b[edge] * 2 + 1];
  }
  return {count, a, b, rides, segments, totalRides};
}

/** Normalised mutual information of two labelings, 0 (independent) to 1 (identical partitions). */
export function normalizedMutualInformation(
  first: ArrayLike<number>,
  second: ArrayLike<number>
): number {
  const n = first.length;
  const joint = new Map<string, number>();
  const firstCounts = new Map<number, number>();
  const secondCounts = new Map<number, number>();
  for (let index = 0; index < n; index++) {
    const x = first[index];
    const y = second[index];
    joint.set(`${x}:${y}`, (joint.get(`${x}:${y}`) ?? 0) + 1);
    firstCounts.set(x, (firstCounts.get(x) ?? 0) + 1);
    secondCounts.set(y, (secondCounts.get(y) ?? 0) + 1);
  }
  const entropy = (counts: Map<number, number>) => {
    let h = 0;
    for (const c of counts.values()) h -= (c / n) * Math.log(c / n);
    return h;
  };
  let mutual = 0;
  for (const [key, c] of joint) {
    const [x, y] = key.split(':').map(Number);
    mutual += (c / n) * Math.log((c * n) / (firstCounts.get(x)! * secondCounts.get(y)!));
  }
  const denominator = (entropy(firstCounts) + entropy(secondCounts)) / 2;
  return denominator > 0 ? mutual / denominator : 1;
}
