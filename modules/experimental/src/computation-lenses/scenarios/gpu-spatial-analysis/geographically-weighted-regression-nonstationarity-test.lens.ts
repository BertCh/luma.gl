// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {registerRegressionLens} from './s1-regression-lens';

/** Permutations of the scenario; mirrors the example's constant. */
const PERMUTATIONS = 99;

registerRegressionLens({
  slug: 'geographically-weighted-regression-nonstationarity-test',
  element: 'GPUGeographicallyWeightedRegressionNonstationarityTest',
  title: 'Which GWR coefficients vary across Chicago tracts more than chance?',
  rungs: ['result', 'formula', 'kernel', 'graph'],
  withMap: false,
  choices: {
    permutations: PERMUTATIONS,
    seed: 1,
    bandwidthPolicy: 'fixed at the bandwidth selected on the observed data'
  },
  metadata: {
    coordinateModel: 'projected',
    distanceFrame: 'azimuthal equidistant, origin -87.65 41.84, metres',
    // Four coefficients are tested separately; no correction across them is applied.
    pCorrection: 'none'
  },
  async loadUploads() {
    const spatial = await import('@luma.gl/experimental/gpu-spatial-analysis');
    return {
      parameters: spatial.getGPUGeographicallyWeightedRegressionParameterValues(
        {kernel: 'bisquare', bandwidthMode: 'adaptive', bandwidths: [0, 0]},
        2
      ),
      permutationParameters: spatial.getGPUPermutationParameterValues(
        {seed: 1, permutationCount: PERMUTATIONS},
        {maximumPermutations: PERMUTATIONS}
      )
    };
  },
  loadExample: async () => {
    const example = await import(
      './geographically-weighted-regression-nonstationarity-test.example'
    );
    return {
      runExample(device, input) {
        assert(input.parameters && input.permutationParameters);
        return example.runExample(device, {
          ...input,
          parameters: input.parameters,
          permutationParameters: input.permutationParameters
        });
      }
    };
  },
  prepare(execution, input, model) {
    const table = model.extractTable(input.graph, {
      id: 'nonstationarityColumns',
      source: execution.output.table,
      stride: 6,
      rowCount: 4,
      columns: {observedStandardDeviation: 0, pseudoPValue: 1, simulatedMean: 3, zSimulated: 5},
      keySpace: {id: 'coefficient', label: 'Intercept and predictors', size: 4, indexSpace: '1d'},
      meaning: 'intercept, poverty150Pct, noVehiclePct, uninsuredPct'
    });
    return {
      views: {
        ...table.views,
        summary: execution.output.summary,
        standardDeviations: execution.output.standardDeviations
      },
      tags: table.tags,
      keySpaces: table.keySpaces
    };
  }
});
