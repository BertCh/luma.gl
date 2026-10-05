// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUNeighborSearchParameterValues,
  GPUNeighborSearch,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  type GPUNeighborSearchParameters
} from '../../../src/gpu-spatial-analysis/neighbor-search';
import type {GPUSpatialWeights} from '../../../src/gpu-spatial-analysis/spatial-weights';
import {
  getGPUSpatialAutocorrelationParameterValues,
  GPUHotSpotAnalysis,
  GPULocalMoran,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  type GPUSpatialAutocorrelationParameters
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

const GARBAGE = 0x7f7f7f7f;

/** Scene uploaded once per harness. */
export type SpatialAutocorrelationScene = {
  values: Float32Array;
  mask?: Uint32Array;
};

/**
 * Where the weights come from: a hand-built CSR uploaded once, or a `GPUNeighborSearch` radius
 * search in the same graph over `positions`.
 */
export type SpatialAutocorrelationWeightsSource =
  | {kind: 'csr'; offsets: Uint32Array; neighbors: Uint32Array; weights: Float32Array}
  | {
      kind: 'neighbor-search';
      positions: Float32Array;
      parameters: GPUNeighborSearchParameters;
      /** Slot capacity of the CSR the search writes. */
      capacity: number;
    };

/** Every output of one encoding, read back. */
export type SpatialAutocorrelationReadback = {
  zScores: number[];
  pValues: number[];
  neighborCounts: number[];
  globalStatistics: number[];
  /** Gi* only. */
  bins: number[];
  /** Local Moran only. */
  localI: number[];
  spatialLag: number[];
  quadrants: number[];
  /** Raw z-score bits, for bitwise determinism checks. */
  zScoreBits: number[];
  /** The CSR the contributor read, trimmed to `offsets[rows]` slots. */
  csr: {offsets: number[]; neighbors: number[]; weights: number[]};
};

/** One compiled graph with every optional output, resubmitted per frame. */
export type SpatialAutocorrelationHarness = {
  /** Number of `getCommandNodes` calls, i.e. graph builds. Stays 1 across frames. */
  readonly buildCount: number;
  run(
    parameters?: GPUSpatialAutocorrelationParameters | Float32Array,
    neighborParameters?: GPUNeighborSearchParameters
  ): Promise<SpatialAutocorrelationReadback>;
  writeValues(values: Float32Array): void;
  writeMask(mask: Uint32Array): void;
  destroy(): void;
};

/** Builds a harness around `GPUHotSpotAnalysis` or `GPULocalMoran`. */
export function createSpatialAutocorrelationHarness(
  device: Device,
  options: {
    contributor: 'hot-spot' | 'local-moran';
    scene: SpatialAutocorrelationScene;
    weights: SpatialAutocorrelationWeightsSource;
    parameters?: GPUSpatialAutocorrelationParameters;
    gridSize?: readonly [number, number];
    falseDiscoveryRate?: boolean;
    /** Gi* only. */
    selfWeight?: number;
  }
): SpatialAutocorrelationHarness {
  const {scene} = options;
  const rows = scene.values.length;
  const valuesBuffer = createInputBuffer(device, scene.values);
  const maskBuffer = scene.mask && createInputBuffer(device, scene.mask);
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'spatial-autocorrelation-parameters',
    format: 'float32',
    length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
    values: getGPUSpatialAutocorrelationParameterValues(options.parameters)
  });
  const source = options.weights;
  const capacity = source.kind === 'csr' ? source.neighbors.length : source.capacity;
  const buffers: Buffer[] = [];
  const createBuffer = (values: Float32Array | Uint32Array): Buffer => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  const outputs: Record<
    | 'zScores'
    | 'pValues'
    | 'neighborCounts'
    | 'globalStatistics'
    | 'bins'
    | 'localI'
    | 'spatialLag'
    | 'quadrants'
    | 'offsets'
    | 'neighbors'
    | 'weights'
    | 'overflow',
    Buffer
  > = {
    zScores: createOutputBuffer(device, rows),
    pValues: createOutputBuffer(device, rows),
    neighborCounts: createOutputBuffer(device, rows),
    globalStatistics: createOutputBuffer(device, 4),
    bins: createOutputBuffer(device, rows),
    localI: createOutputBuffer(device, rows),
    spatialLag: createOutputBuffer(device, rows),
    quadrants: createOutputBuffer(device, rows),
    offsets: createOutputBuffer(device, rows + 1),
    neighbors: createOutputBuffer(device, capacity),
    weights: createOutputBuffer(device, capacity),
    overflow: createOutputBuffer(device, 1)
  };
  // Hand-built CSR inputs are uploaded into the same buffers the search would write.
  if (source.kind === 'csr') {
    outputs.offsets.write(source.offsets);
    outputs.neighbors.write(source.neighbors);
    outputs.weights.write(source.weights);
  }
  const graph = new GPUCommandGraph(device, {id: `${options.contributor}-test`});
  const weights: GPUSpatialWeights = {
    offsets: importGraphBuffer(graph, 'offsets', outputs.offsets, 'uint32', rows + 1),
    neighbors: importGraphBuffer(graph, 'neighbors', outputs.neighbors, 'uint32', capacity),
    weights: importGraphBuffer(graph, 'weights', outputs.weights, 'float32', capacity)
  };
  const positionsBuffer = source.kind === 'neighbor-search' ? createBuffer(source.positions) : null;
  const neighborParameterBuffer =
    source.kind === 'neighbor-search'
      ? new GPUParameterBuffer(device, {
          id: 'neighbor-search-parameters',
          format: 'float32',
          length: GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
          values: getGPUNeighborSearchParameterValues(source.parameters)
        })
      : null;
  if (source.kind === 'neighbor-search' && positionsBuffer && neighborParameterBuffer) {
    graph.add(
      new GPUNeighborSearch({
        mode: 'radius',
        gridSize: options.gridSize ?? [32, 32],
        positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', rows),
        parameters: neighborParameterBuffer.importToGraph(graph),
        weights,
        overflow: importGraphBuffer(graph, 'overflow', outputs.overflow, 'uint32', 1)
      })
    );
  }
  const common = {
    weights,
    values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', rows),
    parameters: parameterBuffer.importToGraph(graph),
    mask: maskBuffer && importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', rows),
    falseDiscoveryRate: options.falseDiscoveryRate,
    zScores: importGraphBuffer(graph, 'z-scores', outputs.zScores, 'float32', rows),
    pValues: importGraphBuffer(graph, 'p-values', outputs.pValues, 'float32', rows),
    neighborCounts: importGraphBuffer(
      graph,
      'neighbor-counts',
      outputs.neighborCounts,
      'uint32',
      rows
    ),
    globalStatistics: importGraphBuffer(
      graph,
      'global-statistics',
      outputs.globalStatistics,
      'float32',
      4
    )
  };
  const contributor =
    options.contributor === 'hot-spot'
      ? new GPUHotSpotAnalysis({
          ...common,
          selfWeight: options.selfWeight,
          bins: importGraphBuffer(graph, 'bins', outputs.bins, 'sint32', rows)
        })
      : new GPULocalMoran({
          ...common,
          localI: importGraphBuffer(graph, 'local-i', outputs.localI, 'float32', rows),
          spatialLag: importGraphBuffer(graph, 'spatial-lag', outputs.spatialLag, 'float32', rows),
          quadrants: importGraphBuffer(graph, 'quadrants', outputs.quadrants, 'uint32', rows)
        });
  let buildCount = 0;
  const getCommandNodes = contributor.getCommandNodes.bind(contributor);
  contributor.getCommandNodes = (target => {
    buildCount++;
    return getCommandNodes(target);
  }) as typeof contributor.getCommandNodes;
  graph.add(contributor);
  const compiled = graph.compile();

  return {
    get buildCount() {
      return buildCount;
    },
    async run(parameters = {}, neighborParameters) {
      parameterBuffer.write(
        parameters instanceof Float32Array
          ? parameters
          : getGPUSpatialAutocorrelationParameterValues(parameters)
      );
      if (neighborParameters && neighborParameterBuffer) {
        neighborParameterBuffer.write(getGPUNeighborSearchParameterValues(neighborParameters));
      }
      for (const [name, buffer] of Object.entries(outputs)) {
        // A hand-built CSR is an input and must survive between frames.
        if (source.kind === 'csr' && ['offsets', 'neighbors', 'weights'].includes(name)) {
          continue;
        }
        buffer.write(new Uint32Array(buffer.byteLength / 4).fill(GARBAGE));
      }
      submitGraph(device, compiled, undefined);
      const zScoreBits = await readUint32(outputs.zScores, rows);
      const binBits = await readUint32(outputs.bins, rows);
      const offsets = await readUint32(outputs.offsets, rows + 1);
      const used = offsets[rows];
      return {
        zScores: Array.from(new Float32Array(Uint32Array.from(zScoreBits).buffer)),
        zScoreBits,
        pValues: await readFloat32(outputs.pValues, rows),
        neighborCounts: await readUint32(outputs.neighborCounts, rows),
        globalStatistics: await readFloat32(outputs.globalStatistics, 4),
        bins: Array.from(new Int32Array(Uint32Array.from(binBits).buffer)),
        localI: await readFloat32(outputs.localI, rows),
        spatialLag: await readFloat32(outputs.spatialLag, rows),
        quadrants: await readUint32(outputs.quadrants, rows),
        csr: {
          offsets,
          neighbors: await readUint32(outputs.neighbors, used),
          weights: await readFloat32(outputs.weights, used)
        }
      };
    },
    writeValues(values) {
      valuesBuffer.write(values);
    },
    writeMask(mask) {
      if (!maskBuffer) {
        throw new Error('harness was created without a mask');
      }
      maskBuffer.write(mask);
    },
    destroy() {
      compiled.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
      neighborParameterBuffer?.destroy();
      valuesBuffer.destroy();
      maskBuffer?.destroy();
      parameterBuffer.destroy();
      for (const buffer of Object.values(outputs)) {
        buffer.destroy();
      }
    }
  };
}

/** Absolute-plus-relative closeness used for every float comparison against the oracle. */
export function isClose(
  actual: number,
  expected: number,
  absolute: number,
  relative: number
): boolean {
  if (Number.isNaN(expected)) {
    return Number.isNaN(actual);
  }
  return Math.abs(actual - expected) <= absolute + relative * Math.abs(expected);
}
