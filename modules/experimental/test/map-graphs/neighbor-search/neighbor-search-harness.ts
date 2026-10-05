// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUMapGraphParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {
  getGPUNeighborSearchParameterValues,
  GPUNeighborSearch,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  type GPUNeighborSearchParameters
} from '../../../src/map-graphs/neighbor-search';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../map-graph-test-utils';

const GARBAGE = 0x7f7f7f7f;

/** Options of {@link createNeighborSearchHarness}. */
export type NeighborSearchHarnessOptions = {
  mode: 'knn' | 'radius';
  positions: Float32Array;
  queryPositions?: Float32Array;
  mask?: Uint32Array;
  queryMask?: Uint32Array;
  k?: number;
  gridSize?: readonly [number, number];
  capacity: number;
  parameters: GPUNeighborSearchParameters;
};

/** One encoding read back. Neighbor columns are trimmed to `offsets[rows]`. */
export type NeighborSearchReadback = {
  offsets: number[];
  neighbors: number[];
  distances: number[];
  weights: number[];
  overflow: number;
  totalNeighbors: number;
  neighborCounts: number[];
  /** Raw weight bits for bitwise determinism checks. */
  weightBits: number[];
};

/** A compiled neighbor-search graph resubmitted per frame. */
export type NeighborSearchHarness = {
  readonly buildCount: number;
  run(parameters?: GPUNeighborSearchParameters): Promise<NeighborSearchReadback>;
  writePositions(positions: Float32Array): void;
  writeMask(mask: Uint32Array): void;
  destroy(): void;
};

/** Builds and compiles one `GPUNeighborSearch` graph with every optional output. */
export function createNeighborSearchHarness(
  device: Device,
  options: NeighborSearchHarnessOptions
): NeighborSearchHarness {
  const targetRows = options.positions.length / 2;
  const queryRows = (options.queryPositions ?? options.positions).length / 2;
  const {capacity} = options;
  const positionsBuffer = createInputBuffer(device, options.positions);
  const queryBuffer = options.queryPositions && createInputBuffer(device, options.queryPositions);
  const maskBuffer = options.mask && createInputBuffer(device, options.mask);
  const queryMaskBuffer = options.queryMask && createInputBuffer(device, options.queryMask);
  const parameterBuffer = new GPUMapGraphParameterBuffer(device, {
    id: 'neighbor-search-parameters',
    format: 'float32',
    length: GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
    values: getGPUNeighborSearchParameterValues(options.parameters)
  });
  const outputs: Record<
    'offsets' | 'neighbors' | 'distances' | 'weights' | 'overflow' | 'total' | 'counts',
    Buffer
  > = {
    offsets: createOutputBuffer(device, queryRows + 1),
    neighbors: createOutputBuffer(device, capacity),
    distances: createOutputBuffer(device, capacity),
    weights: createOutputBuffer(device, capacity),
    overflow: createOutputBuffer(device, 1),
    total: createOutputBuffer(device, 1),
    counts: createOutputBuffer(device, queryRows)
  };
  const graph = new GPUCommandGraph(device, {id: 'neighbor-search-test'});
  const recipe = new GPUNeighborSearch({
    mode: options.mode,
    k: options.k,
    gridSize: options.gridSize ?? [16, 16],
    positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', targetRows),
    queryPositions:
      queryBuffer &&
      importGraphBuffer(graph, 'query-positions', queryBuffer, 'float32x2', queryRows),
    mask: maskBuffer && importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', targetRows),
    queryMask:
      queryMaskBuffer &&
      importGraphBuffer(graph, 'query-mask', queryMaskBuffer, 'uint32', queryRows),
    parameters: parameterBuffer.importToGraph(graph),
    weights: {
      offsets: importGraphBuffer(graph, 'offsets', outputs.offsets, 'uint32', queryRows + 1),
      neighbors: importGraphBuffer(graph, 'neighbors', outputs.neighbors, 'uint32', capacity),
      weights: importGraphBuffer(graph, 'weights', outputs.weights, 'float32', capacity),
      distances: importGraphBuffer(graph, 'distances', outputs.distances, 'float32', capacity)
    },
    overflow: importGraphBuffer(graph, 'overflow', outputs.overflow, 'uint32', 1),
    totalNeighbors: importGraphBuffer(graph, 'total', outputs.total, 'uint32', 1),
    neighborCounts: importGraphBuffer(graph, 'counts', outputs.counts, 'uint32', queryRows)
  });
  let buildCount = 0;
  const getCommandNodes = recipe.getCommandNodes.bind(recipe);
  recipe.getCommandNodes = (target => {
    buildCount++;
    return getCommandNodes(target);
  }) as typeof recipe.getCommandNodes;
  graph.add(recipe);
  const compiled = graph.compile();

  return {
    get buildCount() {
      return buildCount;
    },
    async run(parameters) {
      if (parameters) {
        parameterBuffer.write(getGPUNeighborSearchParameterValues(parameters));
      }
      for (const buffer of Object.values(outputs)) {
        buffer.write(new Uint32Array(buffer.byteLength / 4).fill(GARBAGE));
      }
      submitGraph(device, compiled, undefined);
      const offsets = await readUint32(outputs.offsets, queryRows + 1);
      const used = offsets[queryRows];
      return {
        offsets,
        neighbors: await readUint32(outputs.neighbors, used),
        distances: await readFloat32(outputs.distances, used),
        weights: await readFloat32(outputs.weights, used),
        weightBits: await readUint32(outputs.weights, used),
        overflow: (await readUint32(outputs.overflow, 1))[0],
        totalNeighbors: (await readUint32(outputs.total, 1))[0],
        neighborCounts: await readUint32(outputs.counts, queryRows)
      };
    },
    writePositions(positions) {
      positionsBuffer.write(positions);
    },
    writeMask(mask) {
      if (!maskBuffer) {
        throw new Error('harness was created without a mask');
      }
      maskBuffer.write(mask);
    },
    destroy() {
      compiled.destroy();
      positionsBuffer.destroy();
      queryBuffer?.destroy();
      maskBuffer?.destroy();
      queryMaskBuffer?.destroy();
      parameterBuffer.destroy();
      for (const buffer of Object.values(outputs)) {
        buffer.destroy();
      }
    }
  };
}
