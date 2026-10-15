// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUSpatialWeightsTranspose} from '../../../src/gpu-spatial-analysis/spatial-weights';
import fixture from '../../oracles/fixtures/parity-wp1-round6-transpose.json';
import {decodeRound4Fixture} from '../../oracles/parity-wp1-round4-fixture';
import {sumSparseRows, type TransposeCase} from '../../oracles/parity-wp1-round6-fixture';
import {compareParity} from '../../oracles/parity-wp1-compare';
import {readUint32} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {WeightsRig} from './spatial-weights-harness';

const decoded = decodeRound4Fixture<{cases: TransposeCase[]}>(fixture);
for (let caseIndex = 0; caseIndex < 5; caseIndex++) {
  it(`GPUSpatialWeightsTranspose full SciPy output case ${caseIndex}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const testCase = (await decoded).cases[caseIndex];
    const rig = new WeightsRig(device);
    try {
      const input = rig.uploadWeights(testCase.input, 5);
      const output = rig.weightsOutput(
        testCase.columnCount,
        input.neighbors.length,
        testCase.input.distances.length > 0
      );
      // The shared fixture decoder maps JSON null (not applicable) to NaN.
      // Symmetry is defined only for square matrices; rectangular transpose is supported.
      const asymmetric = Number.isFinite(testCase.asymmetricSlots)
        ? rig.output('uint32', 1)
        : undefined;
      rig.run(
        new GPUSpatialWeightsTranspose({
          weights: input,
          columnCount: testCase.columnCount,
          output: {...output.spatialWeights, asymmetricSlots: asymmetric?.view}
        })
      );
      const actual = await output.read();
      for (const metric of ['offsets', 'neighbors', 'weights', 'distances'] as const) {
        // Transpose only copies f32 bytes: no arithmetic error is needed, even
        // for weights/distances. Duplicate slots remain separate and ordered.
        compareParity(
          'GPUSpatialWeightsTranspose',
          testCase,
          metric,
          actual[metric],
          testCase.expected[metric],
          'E'
        );
      }
      compareParity(
        'GPUSpatialWeightsTranspose',
        testCase,
        'rowSums',
        sumSparseRows(actual),
        testCase.rowSums
      );
      if (asymmetric)
        compareParity(
          'GPUSpatialWeightsTranspose',
          testCase,
          'asymmetricSlots',
          await readUint32(asymmetric.buffer, 1),
          [testCase.asymmetricSlots],
          'E'
        );
    } finally {
      rig.destroy();
    }
  }, 120000);
}
