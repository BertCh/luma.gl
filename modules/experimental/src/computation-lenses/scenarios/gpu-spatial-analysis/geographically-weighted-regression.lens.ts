// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {registerRegressionLens} from './s1-regression-lens';

registerRegressionLens({
  slug: 'geographically-weighted-regression',
  element: 'GPUGeographicallyWeightedRegression',
  title: 'Local effect of poverty on diabetes prevalence across Chicago tracts',
  // No iterations rung: the golden-section nodes carry no loop stamps, so the search is not a
  // recorded loop (the AICc candidate table is captured as plain data).
  rungs: ['result', 'datum', 'formula', 'kernel', 'graph'],
  withMap: true,
  choices: {
    kernel: 'bisquare',
    bandwidthMode: 'adaptive',
    bandwidthUnit: 'nearest tracts k (the tract itself counts)',
    bandwidthSearch: 'golden-section',
    significanceLevel: 0.05
  },
  metadata: {
    coordinateModel: 'projected',
    distanceFrame: 'azimuthal equidistant, origin -87.65 41.84, metres',
    pCorrection: 'adjusted-alpha',
    mappedVariableExtent: 'intensive'
  },
  async loadUploads() {
    const spatial = await import('@luma.gl/experimental/gpu-spatial-analysis');
    return {
      parameters: spatial.getGPUGeographicallyWeightedRegressionParameterValues(
        {
          kernel: 'bisquare',
          bandwidthMode: 'adaptive',
          // 0 takes the default search bound: [40 + 2 (predictors + 1), min(n, 128)] for adaptive k
          bandwidths: [0, 0],
          significanceLevel: 0.05
        },
        2
      )
    };
  },
  loadExample: async () => {
    const example = await import('./geographically-weighted-regression.example');
    return {
      runExample(device, input) {
        assert(input.parameters);
        return example.runExample(device, {...input, parameters: input.parameters});
      }
    };
  },
  prepare(execution, input, model) {
    const {graph} = input;
    const count = input.response.length;
    const terms = ['intercept', 'poverty150Pct', 'noVehiclePct', 'uninsuredPct'];
    const coefficientViews = model.extractColumns(graph, {
      id: 'coefficientColumns',
      source: execution.output.coefficients,
      stride: 4,
      rowCount: count,
      columns: Object.fromEntries(terms.map((term, index) => [`${term}Coefficient`, index]))
    });
    const povertyTValue = model.extractColumns(graph, {
      id: 'povertyTValueColumn',
      source: execution.output.tValues,
      stride: 4,
      rowCount: count,
      columns: {poverty150PctTValue: 1}
    });
    return {
      views: {
        response: input.response,
        localR2: execution.output.localR2,
        selectedBandwidth: execution.output.selectedBandwidth,
        candidateTable: execution.output.candidateTable,
        summary: execution.output.summary,
        ...coefficientViews,
        ...povertyTValue
      },
      tags: {
        response: {
          role: 'input',
          polygons: true,
          meaning: 'CDC PLACES diabetes prevalence (%)'
        },
        poverty150PctCoefficient: {
          role: 'output',
          polygons: true,
          meaning:
            'Local effect of poverty150Pct on diabetes: percentage points per point of poverty; diverging scale with midpoint 0'
        },
        selectedBandwidth: {
          role: 'intermediate',
          keySpace: 'bandwidth',
          meaning:
            'Selected adaptive bandwidth: [candidate table row, k nearest tracts including the tract itself]'
        }
      },
      keySpaces: [
        {id: 'bandwidth', label: 'Selected candidate and bandwidth', size: 2, indexSpace: '1d'}
      ]
    };
  }
});
