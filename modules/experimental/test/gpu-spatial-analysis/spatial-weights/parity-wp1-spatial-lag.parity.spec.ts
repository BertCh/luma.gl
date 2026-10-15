// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUSpatialLag} from '../../../src/gpu-spatial-analysis/spatial-weights';
import fixture from '../../oracles/fixtures/parity-wp1-skater-lag.json';
import {compareParity} from '../../oracles/parity-wp1-compare';
import {requireTestDevice} from '../../utils/require-test-device';
import {AnalysisRig} from '../catchment-accessibility/rig';

for (const testCase of fixture.lag) {
  it(`GPUSpatialLag libpysal parity: ${testCase.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new AnalysisRig(device);
    try {
      const output = rig.output('float32', testCase.n * testCase.columns);
      rig.run(
        new GPUSpatialLag({
          weights: rig.weights(testCase),
          values: rig.input(Float32Array.from(testCase.values), 'float32'),
          columnCount: testCase.columns,
          mask: rig.input(Uint32Array.from(testCase.mask), 'uint32'),
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
