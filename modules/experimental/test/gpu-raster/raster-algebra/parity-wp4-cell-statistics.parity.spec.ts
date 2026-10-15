// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  GPURasterCellStatistics,
  getGPURasterCellStatisticsParameterValues
} from '../../../src/gpu-raster/raster-algebra';
import {GPUParameterBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  statisticsCases,
  floatStatistics,
  integerStatistics
} from '../../oracles/parity-wp4-round5-fixtures';
import {decodePackedNumbers} from '../../oracles/parity-wp4-round3-fixtures';
import {ParityRasterGraph} from '../../oracles/parity-wp4-round3-graph';
import {compareRoundFour} from '../../oracles/parity-wp4-round4-compare';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const scene of statisticsCases) {
  it(`GPURasterCellStatistics NumPy parity: ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const owner = new ParityRasterGraph(device, scene.name);
    const parameters = new GPUParameterBuffer(device, {
      id: 'parameters',
      format: 'float32',
      length: 4
    });
    const floats = floatStatistics.map(name => ({
      name,
      ...owner.output(name, 'float32', scene.cellCount)
    }));
    const integers = integerStatistics.map(name => ({
      name,
      ...owner.output(name, 'uint32', scene.cellCount)
    }));
    owner.graph.add(
      new GPURasterCellStatistics({
        cellCount: scene.cellCount,
        layerCount: scene.layerCount,
        noDataValue: -9999,
        stack: owner.input(
          'stack',
          'float32',
          new Float32Array(await decodePackedNumbers(scene.stack))
        ),
        parameters: parameters.importToGraph(owner.graph),
        output: Object.fromEntries(
          [...floats, ...integers].map(output => [output.name, output.view])
        )
      })
    );
    const compiled = owner.graph.compile();
    try {
      // All policies run on one compiled graph to check per-frame parameter updates.
      for (const [frameIndex, frame] of scene.frames.entries()) {
        parameters.write(getGPURasterCellStatisticsParameterValues(frame.settings));
        submitGraph(device, compiled, undefined);
        for (const output of floats) {
          compareRoundFour(
            'GPURasterCellStatistics',
            `${scene.name}/frame-${frameIndex}`,
            scene.reference,
            output.name,
            await readFloat32(output.buffer, scene.cellCount),
            await decodePackedNumbers(frame.expected[output.name]),
            ['minimum', 'maximum', 'majority', 'minority'].includes(output.name) ? 'E' : 'R'
          );
        }
        for (const output of integers) {
          compareRoundFour(
            'GPURasterCellStatistics',
            `${scene.name}/frame-${frameIndex}`,
            scene.reference,
            output.name,
            await readUint32(output.buffer, scene.cellCount),
            await decodePackedNumbers(frame.expected[output.name]),
            'E'
          );
        }
      }
    } finally {
      compiled.destroy();
      parameters.destroy();
      owner.destroy();
    }
  });
}
