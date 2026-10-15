// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {GPURasterStatistics} from '../../src/gpu-raster';
import {GPUParameterBuffer} from '../../src/utils/gpu-contributor-utils';
import {requireTestDevice} from '../utils/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../utils/gpu-contributor-test-utils';
import {ParityRasterGraph} from '../oracles/parity-wp4-round3-graph';
import {decodePackedNumbers} from '../oracles/parity-wp4-round3-fixtures';
import {compareRasterParity} from '../oracles/parity-wp4-compare';
import {statisticsCases, statisticsMetrics} from '../oracles/parity-wp4-round7-fixtures';

for (const scene of statisticsCases) {
  it(`GPURasterStatistics independent NumPy parity: ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new ParityRasterGraph(device, scene.name);
    const length = scene.width * scene.height;
    const validity = new GPUParameterBuffer(device, {id: 'validity', format: 'uint32', length});
    const originalValidity = new Uint32Array(await decodePackedNumbers(scene.validity));
    const outputs = {
      count: rig.output('count', 'uint32', 1),
      sum: rig.output('sum', 'float32', 1),
      mean: rig.output('mean', 'float32', 1),
      extent: rig.output('extent', 'float32', 2)
    };
    new GPURasterStatistics({
      width: scene.width,
      height: scene.height,
      input: {
        id: 'input',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: rig.input(
            'values',
            'float32',
            new Float32Array(await decodePackedNumbers(scene.values))
          )
        },
        validity: validity.importToGraph(rig.graph),
        noDataValue: -9999,
        scale: scene.scale,
        offset: scene.offset
      },
      count: outputs.count.view,
      sum: outputs.sum.view,
      mean: outputs.mean.view,
      extent: outputs.extent.view
    }).addToGraph(rig.graph);
    const compiled = rig.graph.compile();
    try {
      for (const [frameIndex, frame] of scene.frames.entries()) {
        validity.write(frame.masked ? new Uint32Array(length) : originalValidity);
        submitGraph(device, compiled, undefined);
        for (const metric of statisticsMetrics) {
          const expected = frame.expected[metric];
          compareRasterParity(
            'GPURasterStatistics',
            `${scene.name}/frame-${frameIndex}`,
            scene.reference,
            metric,
            await (metric === 'count' ? readUint32 : readFloat32)(
              outputs[metric].buffer,
              expected.length
            ),
            expected,
            metric === 'count' ? 'E' : 'S'
          );
        }
      }
    } finally {
      compiled.destroy();
      validity.destroy();
      rig.destroy();
    }
  });
}
