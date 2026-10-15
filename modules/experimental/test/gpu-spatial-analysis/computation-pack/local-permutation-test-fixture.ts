// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {
  GPULocalPermutationTest,
  getLocalStatisticSpec,
  type GPULocalPermutationTestProps
} from '../../../src/gpu-spatial-analysis/permutation-inference/gpu-local-permutation-test';
import {GPU_PERMUTATION_PARAMETER_LENGTH} from '../../../src/gpu-spatial-analysis/permutation-inference/permutation-parameters';

/** Builds the real contributor without a GPU; stable view offsets keep complete WGSL comparable. */
export function createReplayTest(
  graph: GPUCommandGraph,
  configuration: Partial<GPULocalPermutationTestProps> = {}
) {
  const view = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  const statistic = configuration.statistic ?? 'localMoran';
  return new GPULocalPermutationTest({
    id: 'golden',
    weights: {
      offsets: view('offsets', 'uint32', 13),
      neighbors: view('neighbors', 'uint32', 24),
      weights: view('weights', 'float32', 24)
    },
    values: view('values', 'float32', 12),
    secondValues: getLocalStatisticSpec(statistic).bivariate
      ? view('second-values', 'float32', 12)
      : undefined,
    parameters: view('parameters', 'uint32', GPU_PERMUTATION_PARAMETER_LENGTH),
    maximumPermutations: 99,
    maximumNeighbors: 4,
    statistic,
    output: {
      exceedances: view('exceedances', 'uint32', 12),
      pseudoPValues: view('pseudo-p-values', 'float32', 12),
      status: {overflow: view('overflow', 'uint32', 1)}
    },
    ...configuration
  });
}
