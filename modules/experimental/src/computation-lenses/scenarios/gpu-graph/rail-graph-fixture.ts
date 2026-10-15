// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUVector} from '@luma.gl/gpgpu/gpu-data';
import {GPUGraph, GPUGraphTopology, type GPUGraphAdjacency} from '@luma.gl/gpgpu/gpu-graph';
import railData from '../../data/g-nl-rail-graph.json';

/** Input preparation and explicitly retained observation resources for a rail scenario. */
export type RailGraphFixtureOptions = {
  id: string;
  directed: boolean;
  weights?: boolean;
  reverse?: boolean;
  retain?: readonly string[];
};

/** Caller-owned rail inputs and outputs; building the fixture never submits commands. */
export type RailGraphFixture = {
  device: Device;
  commandGraph: GPUCommandGraph<void>;
  stations: {
    vertexCount: number;
    names: readonly string[];
    lonLat: Float32Array;
    positions: GPUVector<'float32x2'>;
  };
  edgeCount: number;
  graph: GPUGraph;
  railTopology: GPUGraphTopology;
  destroy(): void;
  track<T extends {destroy(): void}>(resource: T): T;
};

/** Allocates packed rail columns and declares a GPU CSR rebuild, including on NullDevice. */
export function createRailGraphFixture(
  device: Device,
  options: RailGraphFixtureOptions
): RailGraphFixture {
  const resources = new Set<{destroy(): void}>();
  const track = <Resource extends {destroy(): void}>(resource: Resource): Resource => {
    resources.add(resource);
    return resource;
  };
  const destroy = (): void => {
    for (const resource of [...resources].reverse()) resource.destroy();
    resources.clear();
  };
  const createVector = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    name: string,
    format: Format,
    length: number,
    values?: Uint32Array | Float32Array
  ): GPUVector<Format> => {
    const buffer = track(
      device.createBuffer({
        id: `${options.id}-${name}`,
        byteLength: Math.max(length, 1) * (format === 'float32x2' ? 8 : 4),
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      })
    );
    if (values?.length) buffer.write(values);
    return track(new GPUVector({type: 'buffer', name, format, buffer, length, ownsBuffer: false}));
  };
  try {
    const sources: number[] = [];
    const targets: number[] = [];
    const weights: number[] = [];
    const pairs = new Set<string>();
    railData.edges.source.forEach((source, index) => {
      let target = railData.edges.target[index];
      if (!options.directed) {
        if (source === target) return;
        [source, target] = [Math.min(source, target), Math.max(source, target)];
        const pair = `${source}:${target}`;
        if (pairs.has(pair)) return;
        pairs.add(pair);
      }
      sources.push(source);
      targets.push(target);
      // Deduplication keeps the first source edge's weight, never an implicit aggregation.
      weights.push(railData.edges.travelTimeSeconds[index]);
    });
    const vertexCount = railData.stationNames.length;
    const edgeCount = sources.length;
    const capacity = edgeCount * (options.directed ? 1 : 2);
    const createAdjacency = (name: string): GPUGraphAdjacency => ({
      offsets: createVector(`${name}-offsets`, 'uint32', vertexCount + 1),
      neighbors: createVector(`${name}-neighbors`, 'uint32', capacity),
      edgeIds: createVector(`${name}-edge-ids`, 'uint32', capacity),
      ...(options.weights && {edgeWeights: createVector(`${name}-weights`, 'float32', capacity)}),
      count: createVector(`${name}-count`, 'uint32', 1),
      overflow: createVector(`${name}-overflow`, 'uint32', 1)
    });
    const graph = new GPUGraph({
      vertexCount,
      directed: options.directed,
      sourceVertices: createVector('edge-source', 'uint32', edgeCount, new Uint32Array(sources)),
      targetVertices: createVector('edge-target', 'uint32', edgeCount, new Uint32Array(targets)),
      ...(options.weights && {
        edgeWeights: createVector('edge-weights', 'float32', edgeCount, new Float32Array(weights))
      })
    });
    const railTopology = new GPUGraphTopology({
      id: 'rail-topology',
      graph,
      forward: createAdjacency('forward'),
      ...(options.directed && options.reverse && {reverse: createAdjacency('reverse')}),
      invalidEdgeCount: createVector('invalid-edge-count', 'uint32', 1)
    });
    const lonLat = new Float32Array(railData.lonLat);
    const positions = createVector('station-positions', 'float32x2', vertexCount, lonLat);
    const commandGraph = new GPUCommandGraph<void>(device, {
      id: options.id,
      recording: {retain: options.retain}
    });
    // Reserve stable capture ids before contributors import these same physical allocations.
    commandGraph.importGPUVector('csr-offsets', railTopology.forward.offsets);
    commandGraph.importGPUVector('csr-neighbors', railTopology.forward.neighbors);
    commandGraph.importGPUVector('station-positions', positions);
    railTopology.addToGraph(commandGraph);
    return {
      device,
      commandGraph,
      stations: {vertexCount, names: railData.stationNames, lonLat, positions},
      edgeCount,
      graph,
      railTopology,
      destroy,
      track
    };
  } catch (error) {
    destroy();
    throw error;
  }
}
