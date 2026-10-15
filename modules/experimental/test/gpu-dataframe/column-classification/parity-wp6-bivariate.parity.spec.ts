// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUBivariateClassification} from '../../../src/gpu-dataframe/column-classification/gpu-bivariate-classification';
import {getGPUBivariateClassificationParameterValues} from '../../../src/gpu-dataframe/column-classification/bivariate-classification-parameters';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {compareParity} from '../../oracles/parity-wp6-compare';
import {ParityReductionHarness} from '../../oracles/parity-wp6-reduction-harness';
import fixtureJson from '../../oracles/fixtures/parity-wp6-bivariate.json';

type Scene = {
  name: string;
  valuesX: number[];
  valuesY: number[];
  mask: number[];
  alphaValues: number[];
  palette: number[];
  frames: {
    breaksX: number[];
    breaksY: number[];
    alpha: boolean;
    expected: Record<string, number[]>;
  }[];
};
const fixture = decodeOracleFixture<{reference: string; cases: Scene[]}>(fixtureJson);
for (const scene of fixture.cases) {
  it(`GPUBivariateClassification NumPy parity: ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = new ParityReductionHarness(device);
    const breaksX = harness.input('breaks-x', new Float32Array(5), 'float32');
    const breaksXBuffer = harness.buffers.at(-1)!;
    const breaksY = harness.input('breaks-y', new Float32Array(5), 'float32');
    const breaksYBuffer = harness.buffers.at(-1)!;
    const parameters = harness.input('parameters', new Float32Array(8), 'float32');
    const parameterBuffer = harness.buffers.at(-1)!;
    harness.graph.add(
      new GPUBivariateClassification({
        valuesX: harness.input('x', Float32Array.from(scene.valuesX), 'float32'),
        valuesY: harness.input('y', Float32Array.from(scene.valuesY), 'float32'),
        mask: harness.input('mask', Uint32Array.from(scene.mask), 'uint32'),
        alphaValues: harness.input('alpha', Float32Array.from(scene.alphaValues), 'float32'),
        palette: harness.input('palette', Uint32Array.from(scene.palette), 'uint32'),
        breaksX,
        breaksY,
        parameters,
        maximumClassCount: 4,
        output: {
          classIds: harness.output('classIds', scene.mask.length, 'uint32'),
          colors: harness.output('colors', scene.mask.length, 'uint32'),
          classCounts: harness.output('classCounts', 16, 'uint32')
        }
      })
    );
    let compiled: ReturnType<typeof harness.graph.compile> | undefined;
    try {
      compiled = harness.graph.compile();
      for (let encoding = 0; encoding < 2; encoding++) {
        for (const [frameIndex, frame] of scene.frames.entries()) {
          const padded = (values: number[]) => {
            const result = new Float32Array(5);
            result.set(values);
            return result;
          };
          breaksXBuffer.write(padded(frame.breaksX));
          breaksYBuffer.write(padded(frame.breaksY));
          parameterBuffer.write(
            getGPUBivariateClassificationParameterValues({
              classCountX: frame.breaksX.length - 1,
              classCountY: frame.breaksY.length - 1,
              noDataColor: 0x0a0b0c0d,
              ...(frame.alpha ? {valueByAlpha: {domain: [0, 1] as const, minimumAlpha: 0.25}} : {})
            })
          );
          submitGraph(device, compiled, undefined);
          const actual = await harness.read();
          for (const metric of ['classIds', 'colors', 'classCounts']) {
            // All alpha factors and products are dyadic exact integers in this fixture;
            // E is stronger than the audit's R bound and checks every packed color byte.
            compareParity(
              'GPUBivariateClassification',
              `${scene.name}/frame=${frameIndex}/encoding=${encoding}`,
              fixture.reference,
              scene.mask.length,
              metric,
              actual[metric],
              frame.expected[metric],
              true
            );
          }
        }
      }
    } finally {
      compiled?.destroy();
      harness.destroy();
    }
  }, 120000);
}
