// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {compareParity, EXACT} from '../../../experimental/test/gpu-network/parity-wp3-comparison';
import {
  maximumMoveGain,
  optimizationFixtures,
  scorePartition
} from './parity-wp3-modularity-optimization-fixture';

for (const fixture of optimizationFixtures) {
  it(`optimizer reference/evaluator integrity only: ${fixture.case}`, () => {
    expect(fixture.versions['networkx']).toBeTruthy();
    for (const run of fixture.runs) {
      const comparison = {
        ...fixture,
        case: `${fixture.case}/${run.resolution}/${run.seed}/${run.start}`
      };
      compareParity(
        'GPUGraphModularityOptimization fixture integrity',
        comparison,
        'all independent reference objectives',
        [
          scorePartition(fixture, run.initial, run.resolution),
          scorePartition(fixture, run.labels, run.resolution),
          scorePartition(fixture, run.greedyLabels, run.resolution)
        ],
        [run.initialScore, run.score, run.greedyScore],
        {absolute: 1e-12, relative: 1e-12}
      );
      compareParity(
        'GPUGraphModularityOptimization fixture integrity',
        comparison,
        'full vertex permutation',
        [...run.permutation].sort((first, second) => first - second),
        Array.from({length: fixture.nodeCount}, (_, index) => index),
        EXACT
      );
      if (fixture.nodeCount < 20) {
        compareParity(
          'GPUGraphModularityOptimization fixture integrity',
          comparison,
          'exhaustive gain certificate versus NetworkX modularity',
          [maximumMoveGain(fixture, run.initial, run.resolution)],
          [run.initialMaximumGain],
          {absolute: 1e-12, relative: 1e-12}
        );
      }
    }
  });
}
