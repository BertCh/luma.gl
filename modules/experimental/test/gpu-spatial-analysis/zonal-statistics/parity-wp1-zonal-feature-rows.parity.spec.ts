// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {
  GPUZonalStatistics,
  type GPUZonalStatisticsOutput
} from '../../../src/gpu-spatial-analysis/zonal-statistics';
import realFixture from '../../oracles/fixtures/parity-wp1-zonal-real.json';
import smallFixture from '../../oracles/fixtures/parity-wp1-zonal-small.json';
import {
  decodeRound4Fixture,
  ZONAL_FLOAT_OUTPUTS,
  type ZonalCase
} from '../../oracles/parity-wp1-round4-fixture';
import {createRound4Harness} from '../../oracles/parity-wp1-round4-harness';
import {compareParity} from '../../oracles/parity-wp1-compare';
import {requireTestDevice} from '../../utils/require-test-device';

for (const sumOrder of ['atomic', 'sorted'] as const) {
  for (let caseIndex = -1; caseIndex < 4; caseIndex++) {
    it(`GPUZonalStatistics feature-rows Shapely/pandas parity ${sumOrder} case ${caseIndex}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const testCase =
        caseIndex === -1
          ? await decodeRound4Fixture<ZonalCase>(realFixture)
          : (await decodeRound4Fixture<{cases: ZonalCase[]}>(smallFixture)).cases[caseIndex];
      const harness = createRound4Harness(device);
      try {
        const output: GPUZonalStatisticsOutput = {extentStatistic: 'mean'};
        for (const key of ZONAL_FLOAT_OUTPUTS.filter(key => key !== 'featureAreas')) {
          output[key] = harness.output(
            key,
            key === 'extent' ? 2 : testCase.featureCount,
            'float32'
          );
        }
        output.counts = harness.output('counts', testCase.featureCount, 'uint32');
        output.valueCounts = harness.output('valueCounts', testCase.featureCount, 'uint32');
        output.status = {
          overflow: harness.output('overflow', 1, 'uint32'),
          uncertainCount: harness.output('uncertainCount', 1, 'uint32')
        };
        if (testCase.weights.length) {
          output.weightSums = harness.output('weightSums', testCase.featureCount, 'float32');
          output.weightedDensities = harness.output(
            'weightedDensities',
            testCase.featureCount,
            'float32'
          );
        }
        harness.graph.add(
          new GPUZonalStatistics({
            features: {
              kind: 'feature-rows',
              pointFeatureRows: harness.input(
                'membership',
                testCase.expected.pointFeatureRows,
                'uint32'
              ),
              featureCount: testCase.featureCount
            },
            areas: harness.input('areas', testCase.expected.featureAreas, 'float32'),
            values: harness.input('values', testCase.values, 'float32'),
            weights: testCase.weights.length
              ? harness.input('weights', testCase.weights, 'float32')
              : undefined,
            sumOrder,
            output
          })
        );
        harness.run();
        const context = {...testCase, name: `${testCase.name}-${sumOrder}`};
        for (const key of ['counts', 'valueCounts', 'overflow', 'uncertainCount'] as const) {
          compareParity(
            'GPUZonalStatistics',
            context,
            key,
            await harness.read(key),
            testCase.expected[key],
            'E'
          );
        }
        for (const key of ZONAL_FLOAT_OUTPUTS.filter(key => key !== 'featureAreas')) {
          compareParity(
            'GPUZonalStatistics',
            context,
            key,
            await harness.read(key),
            testCase.expected[key]
          );
        }
        if (testCase.weights.length) {
          for (const key of ['weightSums', 'weightedDensities'] as const)
            compareParity(
              'GPUZonalStatistics',
              context,
              key,
              await harness.read(key),
              testCase.expected[key]
            );
        }
      } finally {
        harness.destroy();
      }
    }, 120000);
  }
}
