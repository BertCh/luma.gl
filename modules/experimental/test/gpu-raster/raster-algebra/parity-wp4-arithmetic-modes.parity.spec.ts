// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  GPURasterArithmetic,
  getGPURasterArithmeticParameterValues
} from '../../../src/gpu-raster/raster-algebra';
import {GPUParameterBuffer} from '../../../src/utils/gpu-contributor-utils';
import {arithmeticCases} from '../../oracles/parity-wp4-round4-fixtures';
import {decodePackedNumbers} from '../../oracles/parity-wp4-round3-fixtures';
import {ParityRasterGraph} from '../../oracles/parity-wp4-round3-graph';
import {compareRoundFour} from '../../oracles/parity-wp4-round4-compare';
import {readFloat32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

const groups = new Map<string, typeof arithmeticCases>();
for (const scene of arithmeticCases) {
  const key = `${scene.length}/${Boolean(scene.second)}`;
  const cases = groups.get(key) ?? [];
  cases.push(scene);
  groups.set(key, cases);
}
for (const [name, cases] of groups) {
  it(`GPURasterArithmetic independent references through one changing graph: ${name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const owner = new ParityRasterGraph(device, name);
    const firstCase = cases[0];
    const first = new GPUParameterBuffer(device, {
      id: 'first',
      format: 'float32',
      length: firstCase.length
    });
    const second = new GPUParameterBuffer(device, {
      id: 'second',
      format: 'float32',
      length: firstCase.length
    });
    const parameters = new GPUParameterBuffer(device, {
      id: 'parameters',
      format: 'float32',
      length: getGPURasterArithmeticParameterValues(firstCase.settings).length
    });
    const output = owner.output('output', 'float32', firstCase.length);
    owner.graph.add(
      new GPURasterArithmetic({
        cellCount: firstCase.length,
        noDataValue: -9999,
        a: first.importToGraph(owner.graph),
        b: firstCase.second ? second.importToGraph(owner.graph) : undefined,
        parameters: parameters.importToGraph(owner.graph),
        output: {values: output.view}
      })
    );
    const compiled = owner.graph.compile();
    try {
      for (const scene of [...cases, cases[0]]) {
        first.write(new Float32Array(await decodePackedNumbers(scene.first)));
        if (scene.second) second.write(new Float32Array(await decodePackedNumbers(scene.second)));
        parameters.write(getGPURasterArithmeticParameterValues(scene.settings));
        submitGraph(device, compiled, undefined);
        compareRoundFour(
          'GPURasterArithmetic',
          `${scene.name}/same-graph`,
          scene.reference,
          'all-values',
          await readFloat32(output.buffer, scene.length),
          await decodePackedNumbers(scene.expected),
          'R'
        );
      }
    } finally {
      compiled.destroy();
      first.destroy();
      second.destroy();
      parameters.destroy();
      owner.destroy();
    }
  });
}
