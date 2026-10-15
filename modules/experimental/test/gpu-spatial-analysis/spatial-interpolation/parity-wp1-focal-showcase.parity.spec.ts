// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {
  GPUFocalStatistics,
  getGPUFocalStatisticsParameterValues,
  type GPUFocalStatisticsOutput
} from '../../../src/gpu-spatial-analysis/spatial-interpolation';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import fixture from '../../oracles/fixtures/parity-wp1-round8-focal.json';
import {decodeRound4Fixture, type FocalCase} from '../../oracles/parity-wp1-round4-fixture';
import {createRound4Harness} from '../../oracles/parity-wp1-round4-harness';
import {compareParity} from '../../oracles/parity-wp1-compare';
import {requireTestDevice} from '../../utils/require-test-device';

const showcaseOutputs = ['mean', 'min', 'max', 'range', 'standardDeviation'] as const;
const decoded = decodeRound4Fixture<{cases: FocalCase[]}>(fixture);
for (let caseIndex = 0; caseIndex < 720; caseIndex++) {
  it(`GPUFocalStatistics showcase-parameter SciPy parity case ${caseIndex}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const testCase = (await decoded).cases[caseIndex];
    const harness = createRound4Harness(device);
    try {
      const output: GPUFocalStatisticsOutput = {};
      for (const key of showcaseOutputs) output[key] = harness.output(key, testCase.n, 'float32');
      harness.graph.add(
        new GPUFocalStatistics({
          // The showcase passes nodata as NaN, with no separate validity/sentinel.
          values: harness.input(
            'values',
            testCase.values.map((value, cell) =>
              testCase.validity[cell] && value !== testCase.noDataValue ? value : Number.NaN
            ),
            'float32'
          ),
          grid: makeGPURasterMetadata({width: testCase.width, height: testCase.height}),
          maximumRadius: testCase.maximumRadius,
          parameters: harness.input(
            'parameters',
            Array.from(getGPUFocalStatisticsParameterValues(testCase.settings)),
            'float32'
          ),
          output
        })
      );
      harness.run();
      for (const key of showcaseOutputs) {
        // Profile R uses the valid input range, never the large common offset.
        compareParity(
          'GPUFocalStatistics',
          testCase,
          key,
          await harness.read(key),
          testCase.expected[key],
          'R',
          1e-5 * testCase.rangeScale,
          1e-4
        );
      }
    } finally {
      harness.destroy();
    }
  }, 120000);
}
