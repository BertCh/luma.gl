// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  GPUColumnProfile,
  GPU_COLUMN_PROFILE_STATISTIC,
  GPU_COLUMN_PROFILE_STATISTIC_COUNT,
  getGPUColumnProfileParameterValues,
  type GPUColumnProfileColumn
} from '../../../src/gpu-dataframe/column-profile';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {compareParity} from '../../oracles/parity-wp6-compare';
import {ParityReductionHarness} from '../../oracles/parity-wp6-reduction-harness';
import fixtureJson from '../../oracles/fixtures/parity-wp6-column-profile.json';
import offsetFixtureJson from '../../oracles/fixtures/parity-wp6-offset-profiles.json';

import round8Json from '../../oracles/fixtures/parity-wp6-round8-profiles.json';

type Scene = {
  name: string;
  statisticsOnly?: boolean;
  parameterRuns?: {name: string; domains: [number, number][]; expected: Record<string, number[]>}[];
  columns: {
    values: number[];
    kind?: 'category';
    categoryCount?: number;
    domain?: [number, number];
  }[];
  mask: number[];
  bins: number;
  precision: number;
  topCount: number;
  expected: Record<string, number[]>;
};
const fixture = decodeOracleFixture<{reference: string; cases: Scene[]}>(fixtureJson);

const offsetFixture = decodeOracleFixture<{cases: Scene[]}>(offsetFixtureJson);

for (const scene of [
  ...fixture.cases,
  ...offsetFixture.cases,
  ...decodeOracleFixture<{cases: Scene[]}>(round8Json).cases
]) {
  it(`GPUColumnProfile independent pandas/HLL parity: ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = new ParityReductionHarness(device);
    const columnCount = scene.columns.length;
    const columns = scene.columns.map(
      (column, index): GPUColumnProfileColumn =>
        column.kind === 'category'
          ? {
              kind: 'category',
              categoryCount: column.categoryCount!,
              values: harness.input(`column-${index}`, Uint32Array.from(column.values), 'uint32')
            }
          : {values: harness.input(`column-${index}`, Float32Array.from(column.values), 'float32')}
    );
    const parameterView = scene.statisticsOnly
      ? undefined
      : harness.input(
          'parameters',
          getGPUColumnProfileParameterValues(scene.columns.map(column => column.domain)),
          'float32'
        );
    const parameterBuffer = parameterView ? harness.buffers[harness.buffers.length - 1] : undefined;
    harness.graph.add(
      new GPUColumnProfile({
        columns,
        mask: scene.statisticsOnly
          ? undefined
          : harness.input('mask', Uint32Array.from(scene.mask), 'uint32'),
        parameters: parameterView,
        histogramBinCount: scene.bins,
        hyperLogLogPrecision: scene.precision,
        topCategoryCount: scene.topCount,
        output: {
          statistics: harness.output(
            'statistics',
            columnCount * GPU_COLUMN_PROFILE_STATISTIC_COUNT,
            'float32'
          ),
          ...(scene.statisticsOnly
            ? {}
            : {
                counts: harness.output('counts', columnCount * 2, 'uint32'),
                histograms: harness.output('histograms', columnCount * scene.bins, 'uint32'),
                hyperLogLogRegisters: harness.output(
                  'registers',
                  columnCount * 2 ** scene.precision,
                  'uint32'
                ),
                topCategories: harness.output(
                  'topCategories',
                  columnCount * scene.topCount,
                  'uint32'
                ),
                topCategoryCounts: harness.output(
                  'topCategoryCounts',
                  columnCount * scene.topCount,
                  'uint32'
                )
              })
        }
      })
    );
    let compiled: ReturnType<typeof harness.graph.compile> | undefined;
    try {
      compiled = harness.graph.compile();
      const initial = {
        name: 'initial',
        domains: scene.columns.map(column => column.domain),
        expected: scene.expected
      };
      const runs = [initial, ...(scene.parameterRuns ?? []), initial];
      for (const run of runs) {
        parameterBuffer?.write(getGPUColumnProfileParameterValues(run.domains));
        submitGraph(device, compiled, undefined);
        const actual = await harness.read();
        const compare = (metric: string, observed: number[], expected: number[], exact = false) =>
          compareParity(
            'GPUColumnProfile',
            `${scene.name}/domain=${run.name}`,
            fixture.reference,
            scene.mask.length,
            metric,
            observed,
            expected,
            exact
          );
        for (const field of [
          'counts',
          'histograms',
          'registers',
          'topCategories',
          'topCategoryCounts'
        ]) {
          if (scene.statisticsOnly) continue;
          compare(field, actual[field], run.expected[field], true);
        }
        for (const [field, offset] of Object.entries(GPU_COLUMN_PROFILE_STATISTIC)) {
          const select = (values: number[]) =>
            scene.columns.map(
              (_, index) => values[index * GPU_COLUMN_PROFILE_STATISTIC_COUNT + offset]
            );
          const exact = ['count', 'nullCount', 'minimum', 'maximum', 'overflowCount'].includes(
            field
          );
          const expected = select(run.expected['statistics']);
          compare(
            `statistics.${field}`,
            select(actual['statistics']),
            exact ? expected.map(Math.fround) : expected,
            exact
          );
        }
        // Numerical equality alone cannot distinguish the documented -0 minimum/+0 maximum.
        const zeroSigns = (values: number[]) => values.map(value => (Object.is(value, -0) ? 1 : 0));
        const extrema = (values: number[]) =>
          scene.columns.flatMap((_, index) =>
            values.slice(
              index * GPU_COLUMN_PROFILE_STATISTIC_COUNT + 2,
              index * GPU_COLUMN_PROFILE_STATISTIC_COUNT + 4
            )
          );
        compare(
          'extrema.negativeZero',
          zeroSigns(extrema(actual['statistics'])),
          zeroSigns(extrema(run.expected['statistics'])),
          true
        );
      }
    } finally {
      compiled?.destroy();
      harness.destroy();
    }
  }, 120000);
}
