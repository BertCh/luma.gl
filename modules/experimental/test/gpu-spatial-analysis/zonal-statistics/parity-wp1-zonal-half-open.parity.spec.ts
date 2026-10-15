// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {
  GPUZonalStatistics,
  type GPUZonalStatisticsOutput
} from '../../../src/gpu-spatial-analysis/zonal-statistics';
import fixture from '../../oracles/fixtures/parity-wp1-round8-zonal-half-open.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {
  ZONAL_FLOAT_OUTPUTS,
  type ZonalCase as BaseZonalCase
} from '../../oracles/parity-wp1-round4-fixture';
import {createRound4Harness} from '../../oracles/parity-wp1-round4-harness';
import {compareParity} from '../../oracles/parity-wp1-compare';
import {requireTestDevice} from '../../utils/require-test-device';
import {buildPolygonFeatureArrays} from '../spatial-join/spatial-join-oracle';

type ZonalCase = Omit<BaseZonalCase, 'boundaryRule'> & {boundaryRule: 'half-open'};
for (const sumOrder of ['atomic', 'sorted'] as const) {
  for (let caseIndex = 0; caseIndex < 2; caseIndex++) {
    it(`GPUZonalStatistics half-open Shapely/pandas parity ${sumOrder} case ${caseIndex}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const testCase = decodeOracleFixture<{cases: ZonalCase[]}>(fixture).cases[caseIndex];
      const harness = createRound4Harness(device);
      try {
        const geometry = buildPolygonFeatureArrays(testCase.features);
        const output: GPUZonalStatisticsOutput = {extentStatistic: 'mean'};
        for (const key of ZONAL_FLOAT_OUTPUTS) {
          output[key] = harness.output(
            key,
            key === 'extent' ? 2 : testCase.featureCount,
            'float32'
          );
        }
        output.counts = harness.output('counts', testCase.featureCount, 'uint32');
        output.valueCounts = harness.output('valueCounts', testCase.featureCount, 'uint32');
        output.pointFeatureRows = harness.output('pointFeatureRows', testCase.n, 'uint32');
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
              kind: 'polygons',
              positions: harness.input(
                'polygon-positions',
                Array.from(geometry.polygonPositions),
                'float32x2'
              ),
              featureOffsets: harness.input(
                'feature-offsets',
                Array.from(geometry.featureOffsets),
                'uint32'
              ),
              polygonOffsets: harness.input(
                'polygon-offsets',
                Array.from(geometry.polygonOffsets),
                'uint32'
              ),
              ringOffsets: harness.input(
                'ring-offsets',
                Array.from(geometry.ringOffsets),
                'uint32'
              ),
              // The real tract bounding boxes are sparse. 32 candidates per point is
              // conservative; overflow is still an exact asserted output, never ignored.
              candidateCapacity: testCase.n * Math.min(32, testCase.featureCount),
              boundaryRule: testCase.boundaryRule
            },
            positions: harness.input('positions', testCase.positions, 'float32x2'),
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
        for (const key of [
          'counts',
          'valueCounts',
          'pointFeatureRows',
          'overflow',
          'uncertainCount'
        ] as const) {
          compareParity(
            'GPUZonalStatistics',
            context,
            key,
            await harness.read(key),
            testCase.expected[key],
            'E'
          );
        }
        for (const key of ZONAL_FLOAT_OUTPUTS) {
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
