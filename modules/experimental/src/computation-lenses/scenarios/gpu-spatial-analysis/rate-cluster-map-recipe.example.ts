// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {packGPUColor} from '@luma.gl/experimental/gpu-dataframe';
import {
  addRateClusterMapRecipe,
  GPU_PERMUTATION_PARAMETER_LENGTH,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  GPUParameterBuffer,
  type GPURateClusterMapRecipeProps
} from '@luma.gl/experimental/gpu-spatial-analysis';

/** Inputs the lens scenario uploads from the Chicago North Side tracts. */
export type RateClusterMapRecipeExampleInput = {
  /** Tract polygons as four-level views. */
  tracts: GPURateClusterMapRecipeProps['polygons'];
  /** Bird observations per tract. */
  birdObservations: GraphDataView<'uint32'>;
  /** All observations per tract, the population at risk of being a bird. */
  natureObservations: GraphDataView<'uint32'>;
};

/** Clusters of high and low bird shares among observations, from smoothed rates and queen weights. */
export async function runExample(
  graph: GPUCommandGraph<undefined>,
  {tracts, birdObservations, natureObservations}: RateClusterMapRecipeExampleInput
) {
  const count = birdObservations.length;
  const usage = Buffer.STORAGE | Buffer.COPY_SRC;
  const output = <Format extends 'float32' | 'uint32'>(
    id: string,
    format: Format,
    length: number
  ) => createTransientView(graph, id, format, length, usage);
  const parameters = new GPUParameterBuffer(graph.device, {
    id: 'autocorrelation-parameters',
    format: 'float32',
    length: GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
  });
  const permutationParameters = new GPUParameterBuffer(graph.device, {
    id: 'permutation-parameters',
    format: 'uint32',
    length: GPU_PERMUTATION_PARAMETER_LENGTH
  });
  const palette = new GPUParameterBuffer(graph.device, {
    id: 'quadrant-palette',
    format: 'uint32',
    length: 5,
    values: Uint32Array.of(
      packGPUColor(200, 200, 200),
      packGPUColor(215, 25, 28),
      packGPUColor(171, 217, 233),
      packGPUColor(44, 123, 182),
      packGPUColor(253, 174, 97)
    )
  });
  const standardizedRates = output('standardizedRates', 'float32', count);
  const zScores = output('zScores', 'float32', count);
  const quadrants = output('quadrants', 'uint32', count);
  const colors = output('colors', 'uint32', count);
  const permutation = {
    pseudoPValues: output('pseudoPValues', 'float32', count),
    adjustedPValues: output('adjustedPValues', 'float32', count),
    significant: output('significant', 'uint32', count)
  };
  // #region example
  const clusters = addRateClusterMapRecipe(graph, {
    events: birdObservations,
    populations: natureObservations,
    polygons: tracts,
    criterion: 'queen',
    neighborCapacity: 4096,
    parameters,
    settings: {significanceLevel: 0.05, seed: 7, permutationCount: 199},
    permutation: {parameters: permutationParameters, maximumPermutations: 199},
    palette: palette.importToGraph(graph),
    outputs: {standardizedRates, zScores, quadrants, colors, permutation}
  });
  // quadrants colours each tract: 0 not significant, then high-high, low-high, low-low, high-low
  // #endregion example
  return {
    clusters,
    standardizedRates,
    zScores,
    quadrants,
    colors,
    permutation,
    parameters: [parameters, permutationParameters, palette]
  };
}
