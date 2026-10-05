// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {GPUMapGraphParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {
  getGPUPermutationParameterValues,
  GPUGlobalPermutationTest,
  GPULocalPermutationTest,
  GPU_GLOBAL_PERMUTATION_RESULT,
  GPU_PERMUTATION_PARAMETER_LENGTH,
  type GPUGlobalPermutationStatistic,
  type GPULocalPermutationStatistic,
  type GPUPermutationParameters
} from '../../../src/map-graphs/permutation-inference';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../map-graph-test-utils';
import type {CPUSpatialWeights} from './permutation-oracle';

const GARBAGE = 0x7f7f7f7f;

/** Inputs uploaded once and rewritable between submissions. */
type Scene = {
  weights: CPUSpatialWeights;
  values: Float32Array;
  secondValues?: Float32Array;
  mask?: Uint32Array;
};

type SceneBuffers = {
  offsets: Buffer;
  neighbors: Buffer;
  weights: Buffer;
  values: Buffer;
  secondValues?: Buffer;
  mask?: Buffer;
  parameters: GPUMapGraphParameterBuffer<'uint32'>;
};

function createSceneBuffers(
  device: Device,
  scene: Scene,
  parameters: GPUPermutationParameters
): SceneBuffers {
  return {
    offsets: createInputBuffer(device, scene.weights.offsets),
    neighbors: createInputBuffer(device, scene.weights.neighbors),
    weights: createInputBuffer(device, scene.weights.weights),
    values: createInputBuffer(device, scene.values),
    secondValues: scene.secondValues && createInputBuffer(device, scene.secondValues),
    mask: scene.mask && createInputBuffer(device, scene.mask),
    parameters: new GPUMapGraphParameterBuffer(device, {
      id: 'permutation-parameters',
      format: 'uint32',
      length: GPU_PERMUTATION_PARAMETER_LENGTH,
      values: getGPUPermutationParameterValues(parameters)
    })
  };
}

function importScene(graph: GPUCommandGraph, scene: Scene, buffers: SceneBuffers) {
  const rows = scene.values.length;
  const capacity = scene.weights.neighbors.length;
  return {
    weights: {
      offsets: importGraphBuffer(graph, 'offsets', buffers.offsets, 'uint32', rows + 1),
      neighbors: importGraphBuffer(graph, 'neighbors', buffers.neighbors, 'uint32', capacity),
      weights: importGraphBuffer(graph, 'weights', buffers.weights, 'float32', capacity)
    },
    values: importGraphBuffer(graph, 'values', buffers.values, 'float32', rows),
    secondValues:
      buffers.secondValues &&
      importGraphBuffer(graph, 'second', buffers.secondValues, 'float32', rows),
    mask: buffers.mask && importGraphBuffer(graph, 'mask', buffers.mask, 'uint32', rows),
    parameters: buffers.parameters.importToGraph(graph)
  };
}

function countBuilds(recipe: GPUCommandNodeProducer<unknown>): () => number {
  let buildCount = 0;
  const getCommandNodes = recipe.getCommandNodes.bind(recipe);
  recipe.getCommandNodes = (target => {
    buildCount++;
    return getCommandNodes(target);
  }) as typeof recipe.getCommandNodes;
  return () => buildCount;
}

/** Local permutation readback. */
export type LocalPermutationReadback = {
  exceedances: number[];
  pseudoPValues: number[];
  observed: number[];
  significant: number[];
  overflow: number;
};

/** Builds a compiled `GPULocalPermutationTest` graph with every optional output. */
export function createLocalPermutationHarness(
  device: Device,
  options: {
    scene: Scene;
    statistic: GPULocalPermutationStatistic;
    parameters: GPUPermutationParameters;
    maximumPermutations?: number;
    maximumNeighbors?: number;
    falseDiscoveryRate?: boolean;
  }
) {
  const {scene} = options;
  const rows = scene.values.length;
  const buffers = createSceneBuffers(device, scene, options.parameters);
  const outputs = {
    exceedances: createOutputBuffer(device, rows),
    pseudoPValues: createOutputBuffer(device, rows),
    observed: createOutputBuffer(device, rows),
    significant: createOutputBuffer(device, rows),
    overflow: createOutputBuffer(device, 1)
  };
  const graph = new GPUCommandGraph(device, {id: 'local-permutation-test'});
  const recipe = new GPULocalPermutationTest({
    ...importScene(graph, scene, buffers),
    statistic: options.statistic,
    maximumPermutations: options.maximumPermutations ?? 999,
    maximumNeighbors: options.maximumNeighbors,
    falseDiscoveryRate: options.falseDiscoveryRate,
    exceedances: importGraphBuffer(graph, 'exceedances', outputs.exceedances, 'uint32', rows),
    pseudoPValues: importGraphBuffer(graph, 'pseudo-p', outputs.pseudoPValues, 'float32', rows),
    observed: importGraphBuffer(graph, 'observed', outputs.observed, 'float32', rows),
    significant: importGraphBuffer(graph, 'significant', outputs.significant, 'uint32', rows),
    overflow: importGraphBuffer(graph, 'overflow', outputs.overflow, 'uint32', 1)
  });
  const getBuildCount = countBuilds(recipe);
  graph.add(recipe);
  const compiled = graph.compile();
  return {
    get buildCount() {
      return getBuildCount();
    },
    async run(
      parameters?: GPUPermutationParameters,
      update?: Partial<Scene>
    ): Promise<LocalPermutationReadback> {
      if (parameters) {
        buffers.parameters.write(getGPUPermutationParameterValues(parameters));
      }
      if (update?.values) {
        buffers.values.write(update.values);
      }
      if (update?.mask && buffers.mask) {
        buffers.mask.write(update.mask);
      }
      for (const buffer of Object.values(outputs)) {
        buffer.write(new Uint32Array(buffer.byteLength / 4).fill(GARBAGE));
      }
      submitGraph(device, compiled, undefined);
      return {
        exceedances: await readUint32(outputs.exceedances, rows),
        pseudoPValues: await readFloat32(outputs.pseudoPValues, rows),
        observed: await readFloat32(outputs.observed, rows),
        significant: await readUint32(outputs.significant, rows),
        overflow: (await readUint32(outputs.overflow, 1))[0]
      };
    },
    destroy() {
      compiled.destroy();
      for (const buffer of [...Object.values(buffers), ...Object.values(outputs)]) {
        buffer?.destroy();
      }
    }
  };
}

/** Global permutation readback. */
export type GlobalPermutationReadback = {
  results: number[];
  resultBits: number[];
  referenceDistribution: number[];
  histogram: number[];
};

/** Builds a compiled `GPUGlobalPermutationTest` graph with every optional output. */
export function createGlobalPermutationHarness(
  device: Device,
  options: {
    scene: Scene;
    statistic: GPUGlobalPermutationStatistic;
    parameters: GPUPermutationParameters;
    maximumPermutations: number;
    histogramBins?: number;
  }
) {
  const {scene, maximumPermutations} = options;
  const bins = options.histogramBins ?? 16;
  const buffers = createSceneBuffers(device, scene, options.parameters);
  const outputs = {
    results: createOutputBuffer(device, GPU_GLOBAL_PERMUTATION_RESULT.length),
    referenceDistribution: createOutputBuffer(device, maximumPermutations),
    histogram: createOutputBuffer(device, bins)
  };
  const graph = new GPUCommandGraph(device, {id: 'global-permutation-test'});
  const recipe = new GPUGlobalPermutationTest({
    ...importScene(graph, scene, buffers),
    statistic: options.statistic,
    maximumPermutations,
    results: importGraphBuffer(
      graph,
      'results',
      outputs.results,
      'float32',
      GPU_GLOBAL_PERMUTATION_RESULT.length
    ),
    referenceDistribution: importGraphBuffer(
      graph,
      'reference',
      outputs.referenceDistribution,
      'float32',
      maximumPermutations
    ),
    histogram: importGraphBuffer(graph, 'histogram', outputs.histogram, 'uint32', bins)
  });
  const getBuildCount = countBuilds(recipe);
  graph.add(recipe);
  const compiled = graph.compile();
  return {
    get buildCount() {
      return getBuildCount();
    },
    async run(
      parameters?: GPUPermutationParameters,
      update?: Partial<Scene>
    ): Promise<GlobalPermutationReadback> {
      if (parameters) {
        buffers.parameters.write(getGPUPermutationParameterValues(parameters));
      }
      if (update?.values) {
        buffers.values.write(update.values);
      }
      for (const buffer of Object.values(outputs)) {
        buffer.write(new Uint32Array(buffer.byteLength / 4).fill(GARBAGE));
      }
      submitGraph(device, compiled, undefined);
      return {
        results: await readFloat32(outputs.results, GPU_GLOBAL_PERMUTATION_RESULT.length),
        resultBits: await readUint32(outputs.results, GPU_GLOBAL_PERMUTATION_RESULT.length),
        referenceDistribution: await readFloat32(
          outputs.referenceDistribution,
          maximumPermutations
        ),
        histogram: await readUint32(outputs.histogram, bins)
      };
    },
    destroy() {
      compiled.destroy();
      for (const buffer of [...Object.values(buffers), ...Object.values(outputs)]) {
        buffer?.destroy();
      }
    }
  };
}
