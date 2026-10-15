// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  GPUGroupStatistics,
  type GPUGroupStatistic,
  type GPUGroupStatisticsColumnOutput
} from '../../../src/gpu-dataframe/group-statistics';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {compareParity} from '../../oracles/parity-wp6-compare';
import {ParityReductionHarness} from '../../oracles/parity-wp6-reduction-harness';
import fixtureJson from '../../oracles/fixtures/parity-wp6-group-statistics.json';

import round8Json from '../../oracles/fixtures/parity-wp6-round8-groups.json';

type Scene = {
  name: string;
  omitMask?: boolean;
  statistics?: GPUGroupStatistic[][];
  variances?: ('sample' | 'population')[];
  keys: number[];
  keyBits: 32 | 64;
  columns: number[][];
  mask: number[];
  capacity: number;
  keyCount: number;
  sumScale: number;
  fractions: number[];
  expected: {
    keys: number[];
    counts: number[];
    status: number[];
    columns: Record<
      'sample' | 'population',
      Record<keyof GPUGroupStatisticsColumnOutput, number[]>[]
    >;
  };
};
const fixture = decodeOracleFixture<{reference: string; cases: Scene[]}>(fixtureJson);
const statistics: GPUGroupStatistic[] = [
  'count',
  'sum',
  'mean',
  'minimum',
  'maximum',
  'variance',
  'standardDeviation',
  'skewness',
  'kurtosis',
  'median',
  'percentiles',
  'mode',
  'uniqueCount',
  'zScore'
];

for (const scene of [
  ...fixture.cases,
  ...decodeOracleFixture<{cases: Scene[]}>(round8Json).cases
]) {
  for (const variance of scene.variances ?? (['sample', 'population'] as const)) {
    it(`GPUGroupStatistics pandas/SciPy parity: ${scene.name}/${variance}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const harness = new ParityReductionHarness(device);
      const rows = scene.mask.length;
      const keyFormat = scene.keyBits === 64 ? 'uint32x2' : 'uint32';
      const expectedColumns = scene.expected.columns[variance];
      const columns = scene.columns.map((values, columnIndex) => {
        const output: GPUGroupStatisticsColumnOutput = {};
        const expected = expectedColumns[columnIndex];
        if (expected.counts)
          output.counts = harness.output(`${columnIndex}/counts`, scene.capacity, 'uint32');
        if (expected.sums)
          output.sums = harness.output(`${columnIndex}/sums`, scene.capacity, 'uint32x2');
        if (expected.uniqueCounts)
          output.uniqueCounts = harness.output(
            `${columnIndex}/uniqueCounts`,
            scene.capacity,
            'uint32'
          );
        for (const field of [
          'sumValues',
          'means',
          'minimums',
          'maximums',
          'variances',
          'standardDeviations',
          'skewness',
          'kurtosis',
          'medians',
          'percentiles',
          'modes',
          'zScores'
        ] as const) {
          if (!expected[field]) continue;
          const length =
            field === 'zScores'
              ? rows
              : field === 'percentiles'
                ? scene.capacity * scene.fractions.length
                : scene.capacity;
          output[field] = harness.output(`${columnIndex}/${field}`, length, 'float32');
        }
        return {
          values: harness.input(`values-${columnIndex}`, Float32Array.from(values), 'float32'),
          statistics: scene.statistics?.[columnIndex] ?? statistics,
          output
        };
      });
      harness.graph.add(
        new GPUGroupStatistics({
          keys: harness.input('keys', Uint32Array.from(scene.keys), keyFormat),
          mask: scene.omitMask
            ? undefined
            : harness.input('mask', Uint32Array.from(scene.mask), 'uint32'),
          columns,
          variance,
          sumScale: scene.sumScale,
          keyCount: scene.keyCount || undefined,
          percentiles: columns.some(column => column.statistics.includes('percentiles'))
            ? harness.input('fractions', Float32Array.from(scene.fractions), 'float32')
            : undefined,
          output: {
            keys: harness.output('keys', scene.capacity, keyFormat),
            counts: harness.output('counts', scene.capacity, 'uint32'),
            status: {
              count: harness.output('count', 1, 'uint32'),
              overflow: harness.output('overflow', 1, 'uint32'),
              requiredCount: harness.output('requiredCount', 1, 'uint32')
            }
          }
        })
      );
      let compiled: ReturnType<typeof harness.graph.compile> | undefined;
      try {
        compiled = harness.graph.compile();
        submitGraph(device, compiled, undefined);
        const actual = await harness.read();
        const compare = (metric: string, observed: number[], expected: number[], exact = false) =>
          compareParity(
            'GPUGroupStatistics',
            `${scene.name}/${variance}`,
            fixture.reference,
            rows,
            metric,
            observed,
            expected,
            exact
          );
        compare('keys', actual['keys'], scene.expected.keys, true);
        compare('counts', actual['counts'], scene.expected.counts, true);
        compare(
          'status',
          [...actual['count'], ...actual['overflow'], ...actual['requiredCount']],
          scene.expected.status,
          true
        );
        for (const [columnIndex, expected] of expectedColumns.entries()) {
          for (const [field, values] of Object.entries(expected)) {
            const metric = `${columnIndex}/${field}`;
            compare(
              metric,
              actual[metric],
              values,
              ['counts', 'sums', 'uniqueCounts', 'minimums', 'maximums', 'modes'].includes(field)
            );
          }
        }
      } finally {
        compiled?.destroy();
        harness.destroy();
      }
    }, 120000);
  }
}
