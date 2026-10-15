// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {GPURegionPartitionEvaluation} from '../../../src/gpu-spatial-analysis/spatial-regionalization';
import fixture from '../../oracles/fixtures/parity-wp1-round5-partition.json';
import {decodeRound4Fixture} from '../../oracles/parity-wp1-round4-fixture';
import type {PartitionCase} from '../../oracles/parity-wp1-round5-fixture';
import {createRound4Harness} from '../../oracles/parity-wp1-round4-harness';
import {compareParity} from '../../oracles/parity-wp1-compare';
import {requireTestDevice} from '../../utils/require-test-device';

const cases = decodeRound4Fixture<{cases: PartitionCase[]}>(fixture);
const SUMMARY_METRICS = [
  'regionCount',
  'withinSumOfSquares',
  'betweenSumOfSquares',
  'totalSumOfSquares',
  'minimumSize',
  'maximumSize',
  'crossLinkFraction',
  'ignoredRows',
  'pseudoF'
];
for (let caseIndex = 0; caseIndex < 8; caseIndex++) {
  it(`GPURegionPartitionEvaluation pandas/sklearn parity case ${caseIndex}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const testCase = (await cases).cases[caseIndex];
    const harness = createRound4Harness(device);
    try {
      harness.graph.add(
        new GPURegionPartitionEvaluation({
          values: harness.input('values', testCase.values, 'float32'),
          labels: harness.input('labels', testCase.labels, 'uint32'),
          columnCount: testCase.columns,
          labelCapacity: testCase.labelCapacity,
          weights: testCase.graph
            ? {
                offsets: harness.input('offsets', testCase.graph.offsets, 'uint32'),
                neighbors: harness.input('neighbors', testCase.graph.neighbors, 'uint32'),
                weights: harness.input('weights', testCase.graph.weights, 'float32')
              }
            : undefined,
          output: {
            summary: harness.output('summary', 9, 'float32'),
            regionSizes: harness.output('regionSizes', testCase.labelCapacity, 'uint32'),
            regionWithinSumOfSquares: harness.output(
              'regionWithinSumOfSquares',
              testCase.labelCapacity,
              'float32'
            )
          }
        })
      );
      harness.run();
      const summary = await harness.read('summary');
      SUMMARY_METRICS.forEach((metric, index) => {
        compareParity(
          'GPURegionPartitionEvaluation',
          testCase,
          metric,
          [summary[index]],
          [testCase.expected.summary[index]],
          [0, 4, 5, 7].includes(index) ? 'E' : 'S'
        );
      });
      for (const metric of ['regionSizes', 'regionWithinSumOfSquares'] as const) {
        compareParity(
          'GPURegionPartitionEvaluation',
          testCase,
          metric,
          await harness.read(metric),
          testCase.expected[metric],
          metric === 'regionSizes' ? 'E' : 'S'
        );
      }
    } finally {
      harness.destroy();
    }
  });
}
