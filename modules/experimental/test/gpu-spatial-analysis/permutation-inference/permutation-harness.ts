// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GPUCommandNodeProducer} from '@luma.gl/gpgpu/gpu-core';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUPermutationParameterValues,
  GPUGlobalPermutationTest,
  GPULocalPermutationTest,
  GPU_GLOBAL_PERMUTATION_SUMMARY,
  GPU_PERMUTATION_PARAMETER_LENGTH,
  type GPUGlobalPermutationStatistic,
  type GPULocalJoinCountCase,
  type GPULocalPermutationStatistic,
  type GPUPermutationSettings
} from '../../../src/gpu-spatial-analysis/permutation-inference';
import type {
  GPUInferenceAlternative,
  GPUMultipleTestingMethod
} from '../../../src/analysis-core/contracts';
import type {GPUIslandPolicy} from '../../../src/gpu-spatial-analysis/spatial-weights';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import type {CPUSpatialWeights} from './permutation-oracle';

const GARBAGE = 0x7f7f7f7f;

/** Inputs uploaded once and rewritable between submissions. */
type Scene = {
  weights: CPUSpatialWeights;
  /** Mark the weights as row standardized (`information.transform === 'row'`). */
  rowStandardized?: boolean;
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
  parameters: GPUParameterBuffer<'uint32'>;
};

function createSceneBuffers(
  device: Device,
  scene: Scene,
  parameters: GPUPermutationSettings
): SceneBuffers {
  return {
    offsets: createInputBuffer(device, scene.weights.offsets),
    neighbors: createInputBuffer(device, scene.weights.neighbors),
    weights: createInputBuffer(device, scene.weights.weights),
    values: createInputBuffer(device, scene.values),
    secondValues: scene.secondValues && createInputBuffer(device, scene.secondValues),
    mask: scene.mask && createInputBuffer(device, scene.mask),
    parameters: new GPUParameterBuffer(device, {
      id: 'permutation-parameters',
      format: 'uint32',
      length: GPU_PERMUTATION_PARAMETER_LENGTH,
      values: getGPUPermutationParameterValues(parameters)
    })
  };
}

/** Parameters as the object API, or raw packed words (to exercise invalid counts). */
type ParametersInput = GPUPermutationSettings | Uint32Array;

function packParameters(parameters: ParametersInput): Uint32Array {
  return parameters instanceof Uint32Array
    ? parameters
    : getGPUPermutationParameterValues(parameters);
}

function importScene(graph: GPUCommandGraph, scene: Scene, buffers: SceneBuffers) {
  const rows = scene.values.length;
  const capacity = scene.weights.neighbors.length;
  return {
    weights: {
      information: scene.rowStandardized
        ? {rowCount: rows, transform: 'row' as const, symmetric: 'unknown' as const}
        : undefined,
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

function countBuilds(contributor: GPUCommandNodeProducer<unknown>): () => number {
  let buildCount = 0;
  const getCommandNodes = contributor.getCommandNodes.bind(contributor);
  contributor.getCommandNodes = (target => {
    buildCount++;
    return getCommandNodes(target);
  }) as typeof contributor.getCommandNodes;
  return () => buildCount;
}

/** Local permutation readback. */
export type LocalPermutationReadback = {
  exceedances: number[];
  /** `permutationCounts` output (only with `stopAfterExceedances`; empty otherwise). */
  permutationCounts: number[];
  pseudoPValues: number[];
  adjustedPValues: number[];
  observed: number[];
  significant: number[];
  /** `classes` output (local Geary only; empty otherwise). */
  classes: number[];
  /** `quadrants` output (local bivariate Moran only; empty otherwise). */
  quadrants: number[];
  overflow: number;
  islandCount: number;
};

/** Builds a compiled `GPULocalPermutationTest` graph with every optional output. */
export function createLocalPermutationHarness(
  device: Device,
  options: {
    scene: Scene;
    statistic: GPULocalPermutationStatistic;
    parameters: GPUPermutationSettings;
    maximumPermutations?: number;
    maximumNeighbors?: number;
    multipleTesting?: GPUMultipleTestingMethod;
    islands?: GPUIslandPolicy;
    alternative?: GPUInferenceAlternative;
    joinCase?: GPULocalJoinCountCase;
    /** Besag-Clifford target; also requests the `permutationCounts` output. */
    stopAfterExceedances?: number;
  }
) {
  const {scene} = options;
  const withCounts = options.stopAfterExceedances !== undefined;
  const rows = scene.values.length;
  const withClasses = options.statistic === 'localGeary';
  const withQuadrants = options.statistic === 'localBivariateMoran';
  const buffers = createSceneBuffers(device, scene, options.parameters);
  const outputs = {
    exceedances: createOutputBuffer(device, rows),
    permutationCounts: createOutputBuffer(device, rows),
    pseudoPValues: createOutputBuffer(device, rows),
    adjustedPValues: createOutputBuffer(device, rows),
    observed: createOutputBuffer(device, rows),
    significant: createOutputBuffer(device, rows),
    classes: createOutputBuffer(device, rows),
    quadrants: createOutputBuffer(device, rows),
    overflow: createOutputBuffer(device, 1),
    islandCount: createOutputBuffer(device, 1)
  };
  const graph = new GPUCommandGraph(device, {id: 'local-permutation-test'});
  const contributor = new GPULocalPermutationTest({
    ...importScene(graph, scene, buffers),
    statistic: options.statistic,
    maximumPermutations: options.maximumPermutations ?? 999,
    maximumNeighbors: options.maximumNeighbors,
    multipleTesting: options.multipleTesting,
    islands: options.islands,
    alternative: options.alternative,
    joinCase: options.joinCase,
    stopAfterExceedances: options.stopAfterExceedances,
    output: {
      permutationCounts: withCounts
        ? importGraphBuffer(graph, 'permutation-counts', outputs.permutationCounts, 'uint32', rows)
        : undefined,
      classes: withClasses
        ? importGraphBuffer(graph, 'classes', outputs.classes, 'uint32', rows)
        : undefined,
      quadrants: withQuadrants
        ? importGraphBuffer(graph, 'quadrants', outputs.quadrants, 'uint32', rows)
        : undefined,
      exceedances: importGraphBuffer(graph, 'exceedances', outputs.exceedances, 'uint32', rows),
      pseudoPValues: importGraphBuffer(graph, 'pseudo-p', outputs.pseudoPValues, 'float32', rows),
      adjustedPValues: importGraphBuffer(
        graph,
        'adjusted-p',
        outputs.adjustedPValues,
        'float32',
        rows
      ),
      islandCount: importGraphBuffer(graph, 'island-count', outputs.islandCount, 'uint32', 1),
      observed: importGraphBuffer(graph, 'observed', outputs.observed, 'float32', rows),
      significant: importGraphBuffer(graph, 'significant', outputs.significant, 'uint32', rows),
      status: {overflow: importGraphBuffer(graph, 'overflow', outputs.overflow, 'uint32', 1)}
    }
  });
  const getBuildCount = countBuilds(contributor);
  graph.add(contributor);
  const compiled = graph.compile();
  return {
    get buildCount() {
      return getBuildCount();
    },
    async run(
      parameters?: ParametersInput,
      update?: Partial<Scene>
    ): Promise<LocalPermutationReadback> {
      if (parameters) {
        buffers.parameters.write(packParameters(parameters));
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
        permutationCounts: withCounts ? await readUint32(outputs.permutationCounts, rows) : [],
        pseudoPValues: await readFloat32(outputs.pseudoPValues, rows),
        adjustedPValues: await readFloat32(outputs.adjustedPValues, rows),
        observed: await readFloat32(outputs.observed, rows),
        significant: await readUint32(outputs.significant, rows),
        classes: withClasses ? await readUint32(outputs.classes, rows) : [],
        quadrants: withQuadrants ? await readUint32(outputs.quadrants, rows) : [],
        overflow: (await readUint32(outputs.overflow, 1))[0],
        islandCount: (await readUint32(outputs.islandCount, 1))[0]
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
    parameters: GPUPermutationSettings;
    maximumPermutations: number;
    histogramBins?: number;
    islands?: GPUIslandPolicy;
    alternative?: GPUInferenceAlternative;
    /** Besag-Clifford target (sequential batches). */
    stopAfterExceedances?: number;
  }
) {
  const {scene, maximumPermutations} = options;
  const bins = options.histogramBins ?? 16;
  const buffers = createSceneBuffers(device, scene, options.parameters);
  const outputs = {
    results: createOutputBuffer(device, GPU_GLOBAL_PERMUTATION_SUMMARY.length),
    referenceDistribution: createOutputBuffer(device, maximumPermutations),
    histogram: createOutputBuffer(device, bins)
  };
  const graph = new GPUCommandGraph(device, {id: 'global-permutation-test'});
  const contributor = new GPUGlobalPermutationTest({
    ...importScene(graph, scene, buffers),
    statistic: options.statistic,
    alternative: options.alternative,
    islands: options.islands,
    maximumPermutations,
    stopAfterExceedances: options.stopAfterExceedances,
    output: {
      summary: importGraphBuffer(
        graph,
        'results',
        outputs.results,
        'float32',
        GPU_GLOBAL_PERMUTATION_SUMMARY.length
      ),
      referenceDistribution: importGraphBuffer(
        graph,
        'reference',
        outputs.referenceDistribution,
        'float32',
        maximumPermutations
      ),
      histogram: importGraphBuffer(graph, 'histogram', outputs.histogram, 'uint32', bins)
    }
  });
  const getBuildCount = countBuilds(contributor);
  graph.add(contributor);
  const compiled = graph.compile();
  return {
    get buildCount() {
      return getBuildCount();
    },
    async run(
      parameters?: ParametersInput,
      update?: Partial<Scene>
    ): Promise<GlobalPermutationReadback> {
      if (parameters) {
        buffers.parameters.write(packParameters(parameters));
      }
      if (update?.values) {
        buffers.values.write(update.values);
      }
      for (const buffer of Object.values(outputs)) {
        buffer.write(new Uint32Array(buffer.byteLength / 4).fill(GARBAGE));
      }
      submitGraph(device, compiled, undefined);
      return {
        results: await readFloat32(outputs.results, GPU_GLOBAL_PERMUTATION_SUMMARY.length),
        resultBits: await readUint32(outputs.results, GPU_GLOBAL_PERMUTATION_SUMMARY.length),
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
