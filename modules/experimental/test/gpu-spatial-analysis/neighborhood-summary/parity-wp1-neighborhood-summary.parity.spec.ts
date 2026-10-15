// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {GPUNeighborhoodSummary} from '../../../src/gpu-spatial-analysis/neighborhood-summary';
import fixture from '../../oracles/fixtures/parity-wp1-round5-neighborhood.json';
import {decodeRound4Fixture} from '../../oracles/parity-wp1-round4-fixture';
import type {NeighborhoodCase} from '../../oracles/parity-wp1-round5-fixture';
import {createRound4Harness} from '../../oracles/parity-wp1-round4-harness';
import {compareParity} from '../../oracles/parity-wp1-compare';
import {requireTestDevice} from '../../utils/require-test-device';

const cases = decodeRound4Fixture<{cases: NeighborhoodCase[]}>(fixture);
for (let caseIndex = 0; caseIndex < 12; caseIndex++) {
  it(`GPUNeighborhoodSummary independent full-output parity case ${caseIndex}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const testCase = (await cases).cases[caseIndex];
    const harness = createRound4Harness(device);
    try {
      harness.graph.add(
        new GPUNeighborhoodSummary({
          weights: {
            offsets: harness.input('offsets', testCase.offsets, 'uint32'),
            neighbors: harness.input('neighbors', testCase.neighbors, 'uint32'),
            weights: harness.input('weights', testCase.weights, 'float32'),
            information: {
              rowCount: testCase.n,
              transform: testCase.normalize ? 'row' : 'raw',
              symmetric: 'unknown'
            },
            status: testCase.truncated
              ? {overflow: harness.input('weights-overflow', [1], 'uint32')}
              : undefined
          },
          values: harness.input(
            'values',
            testCase.values,
            testCase.integerValues ? 'uint32' : 'float32'
          ),
          categories: harness.input('categories', testCase.categories, 'uint32'),
          mask: harness.input('mask', testCase.mask, 'uint32'),
          includeFocal: testCase.includeFocal,
          focalWeight: testCase.focalWeight,
          maximumNeighbors: testCase.maximumNeighbors,
          statistics: testCase.statistics,
          output: {
            table: harness.output('table', testCase.n * testCase.statistics.length, 'float32'),
            modes: harness.output('modes', testCase.n, 'uint32'),
            entropy: harness.output('entropy', testCase.n, 'float32'),
            status: {overflow: harness.output('overflow', 1, 'uint32')}
          }
        })
      );
      harness.run();
      const table = await harness.read('table');
      // Compare columns separately so a large sum cannot inflate the tolerance
      // for counts, means, medians or deviations. S is unchanged; count is exact.
      testCase.statistics.forEach((statistic, column) => {
        compareParity(
          'GPUNeighborhoodSummary',
          testCase,
          statistic,
          table.filter((_, index) => index % testCase.statistics.length === column),
          testCase.expected.table.filter(
            (_, index) => index % testCase.statistics.length === column
          ),
          statistic === 'count' ? 'E' : 'S'
        );
      });
      for (const metric of ['modes', 'entropy', 'overflow'] as const) {
        compareParity(
          'GPUNeighborhoodSummary',
          testCase,
          metric,
          await harness.read(metric),
          testCase.expected[metric],
          metric === 'entropy' ? 'S' : 'E'
        );
      }
    } finally {
      harness.destroy();
    }
  });
}
