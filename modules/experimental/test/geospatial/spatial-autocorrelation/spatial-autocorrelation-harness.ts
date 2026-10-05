// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {GPUParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUSpatialAutocorrelationParameterValues,
  GPUHotSpotAnalysis,
  GPULocalMoran,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  type GPUSpatialAutocorrelationParameters
} from '../../../src/geospatial/spatial-autocorrelation';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';

const GARBAGE = 0x7f7f7f7f;

/** Scene uploaded once per harness. */
export type SpatialAutocorrelationScene = {
  positions: Float32Array;
  values: Float32Array;
  mask?: Uint32Array;
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
};

/** One compiled graph with every optional output, resubmitted per frame. */
export type SpatialAutocorrelationHarness = {
  /** Number of `getCommandNodes` calls, i.e. graph builds. Stays 1 across frames. */
  readonly buildCount: number;
  run(
    parameters: GPUSpatialAutocorrelationParameters | Float32Array
  ): Promise<SpatialAutocorrelationReadback>;
  writeValues(values: Float32Array): void;
  writeMask(mask: Uint32Array): void;
  destroy(): void;
};

/** Builds a harness around `GPUHotSpotAnalysis` or `GPULocalMoran`. */
export function createSpatialAutocorrelationHarness(
  device: Device,
  options: {
    recipe: 'hot-spot' | 'local-moran';
    scene: SpatialAutocorrelationScene;
    parameters: GPUSpatialAutocorrelationParameters;
    gridSize?: readonly [number, number];
    falseDiscoveryRate?: boolean;
  }
): SpatialAutocorrelationHarness {
  const {scene} = options;
  const rows = scene.values.length;
  const positionsBuffer = createInputBuffer(device, scene.positions);
  const valuesBuffer = createInputBuffer(device, scene.values);
  const maskBuffer = scene.mask && createInputBuffer(device, scene.mask);
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'spatial-autocorrelation-parameters',
    format: 'float32',
    length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
    values: getGPUSpatialAutocorrelationParameterValues(options.parameters)
  });
  const outputs: Record<
    | 'zScores'
    | 'pValues'
    | 'neighborCounts'
    | 'globalStatistics'
    | 'bins'
    | 'localI'
    | 'spatialLag'
    | 'quadrants',
    Buffer
  > = {
    zScores: createOutputBuffer(device, rows),
    pValues: createOutputBuffer(device, rows),
    neighborCounts: createOutputBuffer(device, rows),
    globalStatistics: createOutputBuffer(device, 4),
    bins: createOutputBuffer(device, rows),
    localI: createOutputBuffer(device, rows),
    spatialLag: createOutputBuffer(device, rows),
    quadrants: createOutputBuffer(device, rows)
  };
  const graph = new GPUCommandGraph(device, {id: `${options.recipe}-test`});
  const common = {
    positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', rows),
    values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', rows),
    parameters: parameterBuffer.importToGraph(graph),
    mask: maskBuffer && importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', rows),
    gridSize: options.gridSize ?? ([32, 32] as const),
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
  const recipe =
    options.recipe === 'hot-spot'
      ? new GPUHotSpotAnalysis({
          ...common,
          bins: importGraphBuffer(graph, 'bins', outputs.bins, 'sint32', rows)
        })
      : new GPULocalMoran({
          ...common,
          localI: importGraphBuffer(graph, 'local-i', outputs.localI, 'float32', rows),
          spatialLag: importGraphBuffer(graph, 'spatial-lag', outputs.spatialLag, 'float32', rows),
          quadrants: importGraphBuffer(graph, 'quadrants', outputs.quadrants, 'uint32', rows)
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
      parameterBuffer.write(
        parameters instanceof Float32Array
          ? parameters
          : getGPUSpatialAutocorrelationParameterValues(parameters)
      );
      for (const buffer of Object.values(outputs)) {
        buffer.write(new Uint32Array(buffer.byteLength / 4).fill(GARBAGE));
      }
      submitGraph(device, compiled, undefined);
      const zScoreBits = await readUint32(outputs.zScores, rows);
      const binBits = await readUint32(outputs.bins, rows);
      return {
        zScores: Array.from(new Float32Array(Uint32Array.from(zScoreBits).buffer)),
        zScoreBits,
        pValues: await readFloat32(outputs.pValues, rows),
        neighborCounts: await readUint32(outputs.neighborCounts, rows),
        globalStatistics: await readFloat32(outputs.globalStatistics, 4),
        bins: Array.from(new Int32Array(Uint32Array.from(binBits).buffer)),
        localI: await readFloat32(outputs.localI, rows),
        spatialLag: await readFloat32(outputs.spatialLag, rows),
        quadrants: await readUint32(outputs.quadrants, rows)
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
      positionsBuffer.destroy();
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
