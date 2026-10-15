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
import squareFixture0 from '../../oracles/fixtures/parity-wp1-focal-square-0.json';
import squareFixture1 from '../../oracles/fixtures/parity-wp1-focal-square-1.json';
import squareFixture2 from '../../oracles/fixtures/parity-wp1-focal-square-2.json';
import squareFixture3 from '../../oracles/fixtures/parity-wp1-focal-square-3.json';
import squareFixture4 from '../../oracles/fixtures/parity-wp1-focal-square-4.json';
import squareFixture5 from '../../oracles/fixtures/parity-wp1-focal-square-5.json';
import squareFixture6 from '../../oracles/fixtures/parity-wp1-focal-square-6.json';
import squareFixture7 from '../../oracles/fixtures/parity-wp1-focal-square-7.json';
import circleFixture0 from '../../oracles/fixtures/parity-wp1-focal-circle-0.json';
import circleFixture1 from '../../oracles/fixtures/parity-wp1-focal-circle-1.json';
import circleFixture2 from '../../oracles/fixtures/parity-wp1-focal-circle-2.json';
import circleFixture3 from '../../oracles/fixtures/parity-wp1-focal-circle-3.json';
import circleFixture4 from '../../oracles/fixtures/parity-wp1-focal-circle-4.json';
import circleFixture5 from '../../oracles/fixtures/parity-wp1-focal-circle-5.json';
import circleFixture6 from '../../oracles/fixtures/parity-wp1-focal-circle-6.json';
import circleFixture7 from '../../oracles/fixtures/parity-wp1-focal-circle-7.json';
import smallFixture from '../../oracles/fixtures/parity-wp1-focal-small.json';
import {
  decodeRound4Fixture,
  FOCAL_FLOAT_OUTPUTS,
  type FocalCase
} from '../../oracles/parity-wp1-round4-fixture';
import {createRound4Harness} from '../../oracles/parity-wp1-round4-harness';
import {compareParity} from '../../oracles/parity-wp1-compare';
import {requireTestDevice} from '../../utils/require-test-device';

const realFixtures = [
  squareFixture0,
  squareFixture1,
  squareFixture2,
  squareFixture3,
  squareFixture4,
  squareFixture5,
  squareFixture6,
  squareFixture7,
  circleFixture0,
  circleFixture1,
  circleFixture2,
  circleFixture3,
  circleFixture4,
  circleFixture5,
  circleFixture6,
  circleFixture7
];
for (let caseIndex = 0; caseIndex < realFixtures.length + 7; caseIndex++) {
  it(`GPUFocalStatistics independent SciPy parity case ${caseIndex}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const testCase =
      caseIndex < realFixtures.length
        ? await decodeRound4Fixture<FocalCase>(realFixtures[caseIndex])
        : (await decodeRound4Fixture<{cases: FocalCase[]}>(smallFixture)).cases[
            caseIndex - realFixtures.length
          ];
    const harness = createRound4Harness(device);
    try {
      const output: GPUFocalStatisticsOutput = {
        counts: harness.output('counts', testCase.n, 'uint32')
      };
      for (const key of FOCAL_FLOAT_OUTPUTS)
        output[key] = harness.output(key, testCase.n, 'float32');
      harness.graph.add(
        new GPUFocalStatistics({
          values: harness.input('values', testCase.values, 'float32'),
          validity: harness.input('validity', testCase.validity, 'uint32'),
          noDataValue: testCase.noDataValue,
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
      compareParity(
        'GPUFocalStatistics',
        testCase,
        'counts',
        await harness.read('counts'),
        testCase.expected.counts,
        'E'
      );
      for (const key of FOCAL_FLOAT_OUTPUTS) {
        // Profile R uses the valid input range, never the large common offset.
        // Full local arrays include halo rows; core hashes independently certify
        // that the eight stripes cover every unpartitioned 512x512 reference cell.
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
