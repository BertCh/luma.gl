// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  GPUColumnQuantiles,
  getGPUColumnQuantilesParameterValues,
  getGPUColumnQuantilesParameterLength,
  type GPUColumnQuantileInterpolation
} from '../../../src/gpu-dataframe/column-classification';
import {GPUParameterBuffer} from '../../../src/utils/gpu-contributor-utils';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {compareParity} from '../../oracles/parity-wp6-compare';
import {ParityReductionHarness} from '../../oracles/parity-wp6-reduction-harness';
import fixtureJson from '../../oracles/fixtures/parity-wp6-column-quantiles.json';

type Scene = {
  name: string;
  values: number[];
  mask: number[];
  probabilities: number[];
  filterRange: [number, number];
  expected: {
    quantiles: Record<GPUColumnQuantileInterpolation, number[]>;
    validCount: number[];
    filterBounds: number[];
    outputMask: number[];
  };
};
const fixture = decodeOracleFixture<{reference: string; cases: Scene[]}>(fixtureJson);
const methods: GPUColumnQuantileInterpolation[] = [
  'lower',
  'higher',
  'nearest',
  'linear',
  'midpoint'
];

for (const scene of fixture.cases) {
  it(`GPUColumnQuantiles NumPy parity: ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = new ParityReductionHarness(device);
    const parameters = new GPUParameterBuffer(device, {
      id: 'parity-quantile-parameters',
      format: 'float32',
      length: getGPUColumnQuantilesParameterLength(scene.probabilities.length)
    });
    harness.graph.add(
      new GPUColumnQuantiles({
        values: harness.input('values', Float32Array.from(scene.values), 'float32'),
        mask: harness.input('mask', Uint32Array.from(scene.mask), 'uint32'),
        parameters: parameters.importToGraph(harness.graph),
        quantileCount: scene.probabilities.length,
        output: {
          quantiles: harness.output('quantiles', scene.probabilities.length, 'float32'),
          validCount: harness.output('validCount', 1, 'uint32'),
          filterBounds: harness.output('filterBounds', 2, 'float32'),
          outputMask: harness.output('outputMask', scene.values.length, 'uint32')
        }
      })
    );
    let compiled: ReturnType<typeof harness.graph.compile> | undefined;
    try {
      compiled = harness.graph.compile();
      // Exercise per-encoding interpolation changes on one compiled graph.
      for (const interpolation of methods) {
        parameters.write(
          getGPUColumnQuantilesParameterValues({
            quantiles: scene.probabilities,
            interpolation,
            filterRange: scene.filterRange
          })
        );
        submitGraph(device, compiled, undefined);
        const actual = await harness.read();
        const compare = (metric: string, observed: number[], expected: number[], exact = false) =>
          compareParity(
            'GPUColumnQuantiles',
            `${scene.name}/${interpolation}`,
            fixture.reference,
            scene.values.length,
            metric,
            observed,
            expected,
            exact
          );
        const exact = ['lower', 'higher', 'nearest'].includes(interpolation);
        const quantiles = scene.expected.quantiles[interpolation];
        compare(
          'quantiles',
          actual['quantiles'],
          exact ? quantiles.map(Math.fround) : quantiles,
          exact
        );
        compare('validCount', actual['validCount'], scene.expected.validCount, true);
        compare(
          'filterBounds',
          actual['filterBounds'],
          scene.expected.filterBounds.map(Math.fround),
          true
        );
        compare('outputMask', actual['outputMask'], scene.expected.outputMask, true);
      }
    } finally {
      compiled?.destroy();
      parameters.destroy();
      harness.destroy();
    }
  }, 120000);
}
