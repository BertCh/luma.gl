// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {
  GPURasterSieve,
  getGPURasterSieveParameterValues
} from '../../../src/gpu-raster/raster-patches';
import {GPUParameterBuffer} from '../../../src/utils/gpu-contributor-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {ParityRasterGraph} from '../../oracles/parity-wp4-round3-graph';
import {decodePackedNumbers} from '../../oracles/parity-wp4-round3-fixtures';
import {compareRasterParity} from '../../oracles/parity-wp4-compare';
import {sieveCases, sieveMetrics} from '../../oracles/parity-wp4-round7-fixtures';

for (const scene of sieveCases) {
  it(`GPURasterSieve independent parity: ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new ParityRasterGraph(device, scene.name);
    const parameters = new GPUParameterBuffer(device, {
      id: 'parameters',
      format: 'uint32',
      length: 4
    });
    const outputs = {
      labels: rig.output('output-labels', 'uint32', scene.width * scene.height),
      patchTargets: rig.output('targets', 'uint32', scene.capacity),
      sievedCount: rig.output('count', 'uint32', 1)
    };
    rig.graph.add(
      new GPURasterSieve({
        grid: {
          width: scene.width,
          height: scene.height,
          affine: [1, 0, 0, 0, 1, 0],
          pixelInterpretation: 'area'
        },
        labels: rig.input(
          'labels',
          'uint32',
          new Uint32Array(await decodePackedNumbers(scene.labels))
        ),
        patchCapacity: scene.capacity,
        mode: scene.mode,
        connectivity: scene.connectivity,
        parameters: parameters.importToGraph(rig.graph),
        output: {
          labels: outputs.labels.view,
          patchTargets: outputs.patchTargets.view,
          sievedCount: outputs.sievedCount.view
        }
      })
    );
    const compiled = rig.graph.compile();
    try {
      for (const [frameIndex, frame] of scene.frames.entries()) {
        parameters.write(getGPURasterSieveParameterValues(frame));
        submitGraph(device, compiled, undefined);
        for (const metric of sieveMetrics) {
          const expected = await decodePackedNumbers(frame.expected[metric]);
          compareRasterParity(
            'GPURasterSieve',
            `${scene.name}/frame-${frameIndex}`,
            scene.reference,
            metric,
            await readUint32(outputs[metric].buffer, expected.length),
            expected,
            'E'
          );
        }
      }
    } finally {
      compiled.destroy();
      parameters.destroy();
      rig.destroy();
    }
  });
}
