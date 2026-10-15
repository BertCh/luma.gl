// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUSpatialLag} from '../../../src/gpu-spatial-analysis/spatial-weights';
import fixtureJson from '../../oracles/fixtures/parity-wp1-lag-modes.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
const fixture = decodeOracleFixture<{
  cases: (Omit<(typeof fixtureJson.cases)[number], 'values' | 'expected'> & {
    values: number[];
    expected: number[];
  })[];
}>(fixtureJson);
import {compareParity} from '../../oracles/parity-wp1-compare';
import {requireTestDevice} from '../../utils/require-test-device';
import {AnalysisRig} from '../catchment-accessibility/rig';

for (const testCase of fixture.cases) {
  it(`GPUSpatialLag rectangular scipy parity: ${testCase.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    try {
      const output = rig.output('float32', testCase.n * testCase.columns);
      rig.run(
        new GPUSpatialLag({
          weights: rig.weights(testCase),
          values: testCase.integer
            ? rig.input(Uint32Array.from(testCase.values), 'uint32')
            : rig.input(Float32Array.from(testCase.values), 'float32'),
          columnCount: testCase.columns,
          normalize: testCase.normalize,
          output
        })
      );
      const actual = await rig.readFloat(output);
      // Separate columns keep population scale from hiding errors in rates or counts.
      for (let column = 0; column < testCase.columns; column++) {
        compareParity(
          'GPUSpatialLag',
          testCase,
          `column-${column}`,
          actual.filter((_, index) => index % testCase.columns === column),
          testCase.expected.filter((_, index) => index % testCase.columns === column)
        );
      }
    } finally {
      rig.destroy();
    }
  });
}
