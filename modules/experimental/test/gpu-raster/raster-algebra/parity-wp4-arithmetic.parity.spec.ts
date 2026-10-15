// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  GPURasterArithmetic,
  getGPURasterArithmeticParameterValues
} from '../../../src/gpu-raster/raster-algebra';
import {arithmeticCases} from '../../oracles/parity-wp4-round4-fixtures';
import {decodePackedNumbers} from '../../oracles/parity-wp4-round3-fixtures';
import {ParityRasterGraph} from '../../oracles/parity-wp4-round3-graph';
import {compareRoundFour} from '../../oracles/parity-wp4-round4-compare';
import {readFloat32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const scene of arithmeticCases) {
  it(`GPURasterArithmetic NumPy parity: ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const [first, second, expected] = await Promise.all([
      decodePackedNumbers(scene.first),
      scene.second ? decodePackedNumbers(scene.second) : Promise.resolve(null),
      decodePackedNumbers(scene.expected)
    ]);
    const owner = new ParityRasterGraph(device, scene.name);
    const output = owner.output('output', 'float32', scene.length);
    owner.graph.add(
      new GPURasterArithmetic({
        cellCount: scene.length,
        a: owner.input('first', 'float32', new Float32Array(first)),
        b: second ? owner.input('second', 'float32', new Float32Array(second)) : undefined,
        noDataValue: -9999,
        parameters: owner.input(
          'parameters',
          'float32',
          getGPURasterArithmeticParameterValues(scene.settings)
        ),
        output: {values: output.view}
      })
    );
    const compiled = owner.graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      compareRoundFour(
        'GPURasterArithmetic',
        scene.name,
        scene.reference,
        'every-value-and-exceptional-mask',
        await readFloat32(output.buffer, scene.length),
        expected,
        'R'
      );
    } finally {
      compiled.destroy();
      owner.destroy();
    }
  });
}
