// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {
  GPUZonalStatistics,
  type GPUZonalStatisticsExtentStatistic,
  type GPUZonalStatisticsOutput
} from '../../../src/gpu-spatial-analysis/zonal-statistics';
import fixture from '../../oracles/fixtures/parity-wp1-round8-zonal-showcase.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {type ZonalCase as BaseZonalCase} from '../../oracles/parity-wp1-round4-fixture';
import {createRound4Harness} from '../../oracles/parity-wp1-round4-harness';
import {compareParity} from '../../oracles/parity-wp1-compare';
import {requireTestDevice} from '../../utils/require-test-device';
import {buildPolygonFeatureArrays} from '../spatial-join/spatial-join-oracle';

type ZonalCase = Omit<BaseZonalCase, 'boundaryRule'> & {
  boundaryRule: 'open' | 'closed' | 'half-open';
  statistic:
    | 'count'
    | 'sum'
    | 'mean'
    | 'minimum'
    | 'maximum'
    | 'density-area'
    | 'density-population';
  sumOrder: 'atomic' | 'sorted';
  areas: number[];
};
for (let caseIndex = 0; caseIndex < 42; caseIndex++) {
  it(`GPUZonalStatistics showcase output Shapely/pandas parity case ${caseIndex}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const testCase = decodeOracleFixture<{cases: ZonalCase[]}>(fixture).cases[caseIndex];
    const sumOrder = testCase.sumOrder;
    const statistic = testCase.statistic;
    const key = {
      count: 'counts',
      sum: 'sums',
      mean: 'means',
      minimum: 'minima',
      maximum: 'maxima',
      'density-area': 'densities',
      'density-population': 'densities'
    }[statistic] as 'counts' | 'sums' | 'means' | 'minima' | 'maxima' | 'densities';
    const usesValues = statistic !== 'count' && !statistic.startsWith('density');
    const harness = createRound4Harness(device);
    try {
      const geometry = buildPolygonFeatureArrays(testCase.features);
      const output: GPUZonalStatisticsOutput = {
        counts: harness.output('counts', testCase.featureCount, 'uint32'),
        extent: harness.output('extent', 2, 'float32'),
        extentStatistic: (statistic.startsWith('density')
          ? 'density'
          : statistic) as GPUZonalStatisticsExtentStatistic,
        status: {
          overflow: harness.output('overflow', 1, 'uint32'),
          uncertainCount: harness.output('uncertainCount', 1, 'uint32')
        }
      };
      if (key !== 'counts') output[key] = harness.output(key, testCase.featureCount, 'float32');
      if (statistic === 'density-area')
        output.featureAreas = harness.output('featureAreas', testCase.featureCount, 'float32');
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
            ringOffsets: harness.input('ring-offsets', Array.from(geometry.ringOffsets), 'uint32'),
            // The real tract bounding boxes are sparse. 32 candidates per point is
            // conservative; overflow is still an exact asserted output, never ignored.
            candidateCapacity: testCase.n * Math.min(32, testCase.featureCount),
            boundaryRule: testCase.boundaryRule
          },
          positions: harness.input('positions', testCase.positions, 'float32x2'),
          values: usesValues ? harness.input('values', testCase.values, 'float32') : undefined,
          areas: testCase.areas.length
            ? harness.input('areas', testCase.areas, 'float32')
            : undefined,
          sumOrder,
          output
        })
      );
      harness.run();
      const context = {...testCase, name: `${testCase.name}-${sumOrder}`};
      for (const outputKey of ['counts', 'overflow', 'uncertainCount'] as const)
        compareParity(
          'GPUZonalStatistics',
          context,
          outputKey,
          await harness.read(outputKey),
          testCase.expected[outputKey],
          'E'
        );
      if (key !== 'counts')
        compareParity(
          'GPUZonalStatistics',
          context,
          key,
          await harness.read(key),
          testCase.expected[key]
        );
      compareParity(
        'GPUZonalStatistics',
        context,
        'extent',
        await harness.read('extent'),
        testCase.expected.extent
      );
      if (statistic === 'density-area')
        compareParity(
          'GPUZonalStatistics',
          context,
          'featureAreas',
          await harness.read('featureAreas'),
          testCase.expected.featureAreas
        );
    } finally {
      harness.destroy();
    }
  }, 120000);
}
