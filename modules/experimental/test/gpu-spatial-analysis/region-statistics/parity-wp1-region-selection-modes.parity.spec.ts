// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {
  GPURegionStatistics,
  decodeGPURegionStatistics,
  getGPURegionStatisticsSummaryLength,
  type GPURegionStatisticsResult,
  type GPURegionSelection
} from '../../../src/gpu-spatial-analysis/region-statistics';
import realFixture from '../../oracles/fixtures/parity-wp1-round5-region-real.json';
import smallFixture from '../../oracles/fixtures/parity-wp1-round5-region-small.json';
import {decodeRound4Fixture} from '../../oracles/parity-wp1-round4-fixture';
import type {RegionCase, RegionQuery} from '../../oracles/parity-wp1-round5-fixture';
import {createRound4Harness} from '../../oracles/parity-wp1-round4-harness';
import {compareParity} from '../../oracles/parity-wp1-compare';
import {requireTestDevice} from '../../utils/require-test-device';

const fixtures = {
  real: decodeRound4Fixture<RegionCase>(realFixture),
  small: decodeRound4Fixture<RegionCase>(smallFixture)
};
function compareSummary(
  testCase: RegionCase,
  query: RegionQuery,
  result: GPURegionStatisticsResult,
  capacity: number,
  suffix: string
) {
  const comparison = {...testCase, name: `${testCase.name}/${query.name}/${suffix}`};
  compareParity(
    'GPURegionStatistics',
    comparison,
    'counts',
    [result.selectedCount, result.valueCount, result.histogramOutsideCount],
    query.expected.counts,
    'E'
  );
  const statistics = [result.sum, result.mean, result.minimum, result.maximum];
  for (const [index, metric] of ['sum', 'mean', 'minimum', 'maximum'].entries()) {
    compareParity(
      'GPURegionStatistics',
      comparison,
      metric,
      [statistics[index]],
      [query.expected.statistics[index]]
    );
  }
  compareParity(
    'GPURegionStatistics',
    comparison,
    'histogram',
    Array.from(result.histogram),
    query.expected.histogram,
    'E'
  );
  compareParity(
    'GPURegionStatistics',
    comparison,
    'flags',
    [
      Number(result.selectionTruncated),
      Number(result.regionTruncated),
      Number(result.candidatesTruncated)
    ],
    [Number(query.expected.ids.length > capacity), 0, 0],
    'E'
  );
  return comparison;
}

for (const selectionMode of ['mask', 'pick-region'] as const) {
  for (const fixtureName of ['real', 'small'] as const) {
    for (let queryIndex = 0; queryIndex < (fixtureName === 'real' ? 4 : 6); queryIndex++) {
      for (const bounded of [false, true]) {
        it(`GPURegionStatistics Shapely/pandas ${selectionMode} ${fixtureName} query ${queryIndex} bounded ${bounded}`, async () => {
          const device = await getWebGPUTestDevice();
          requireTestDevice(device);
          const testCase = await fixtures[fixtureName];
          const query = testCase.queries[queryIndex];
          const capacity = bounded ? 2 : testCase.n;
          const harness = createRound4Harness(device);
          try {
            const selectedIndices = query.expected.mask.flatMap((selected, row) =>
              selected ? [row, 0] : []
            );
            const region: GPURegionSelection =
              selectionMode === 'mask'
                ? {
                    kind: 'mask',
                    mask: harness.input('selection-mask', query.expected.mask, 'uint32')
                  }
                : {
                    kind: 'pick-region',
                    picks: harness.input(
                      'picks',
                      [
                        selectedIndices.length / 2,
                        0,
                        ...(selectedIndices.length ? selectedIndices : [0, 0])
                      ],
                      'uint32'
                    )
                  };
            harness.graph.add(
              new GPURegionStatistics({
                region,
                values: harness.input('values', testCase.values, 'float32'),
                sourceIds: harness.input('source-ids', testCase.sourceIds, 'uint32'),
                histogram: query.binCount
                  ? {binCount: query.binCount, domain: query.domain ?? 'selection'}
                  : undefined,
                output: {
                  summary: harness.output(
                    'summary',
                    getGPURegionStatisticsSummaryLength(query.binCount),
                    'uint32'
                  ),
                  ...(selectionMode === 'pick-region'
                    ? {mask: harness.output('mask', testCase.n, 'uint32')}
                    : {}),
                  selectedRows: {
                    ids: harness.output('ids', capacity, 'uint32'),
                    status: {
                      count: harness.output('count', 1, 'uint32'),
                      overflow: harness.output('overflow', 1, 'uint32'),
                      requiredCount: harness.output('requiredCount', 1, 'uint32')
                    }
                  },
                  drawInstanceCount: harness.output('draw-count', 1, 'uint32')
                }
              })
            );
            harness.run();
            const summary = decodeGPURegionStatistics(
              Uint32Array.from(await harness.read('summary'))
            );
            const comparison = compareSummary(
              testCase,
              query,
              summary,
              capacity,
              `${selectionMode}-capacity-${capacity}`
            );
            if (selectionMode === 'pick-region')
              compareParity(
                'GPURegionStatistics',
                comparison,
                'mask',
                await harness.read('mask'),
                query.expected.mask,
                'E'
              );
            const expectedCount = Math.min(capacity, query.expected.ids.length);
            compareParity(
              'GPURegionStatistics',
              comparison,
              'selectedIds',
              (await harness.read('ids')).slice(0, expectedCount),
              query.expected.ids.slice(0, capacity),
              'E'
            );
            for (const [metric, expected] of Object.entries({
              count: expectedCount,
              overflow: Number(query.expected.ids.length > capacity),
              requiredCount: query.expected.ids.length,
              'draw-count': expectedCount
            })) {
              compareParity(
                'GPURegionStatistics',
                comparison,
                metric,
                await harness.read(metric),
                [expected],
                'E'
              );
            }
          } finally {
            harness.destroy();
          }
        });
      }
    }
  }
}
