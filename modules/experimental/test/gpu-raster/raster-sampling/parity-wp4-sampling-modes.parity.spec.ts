// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {
  GPURasterSampling,
  getGPURasterSamplingParameterValues
} from '../../../src/gpu-raster/raster-sampling';
import {GPUParameterBuffer} from '../../../src/utils/gpu-contributor-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {readFloat32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {ParityRasterGraph} from '../../oracles/parity-wp4-round3-graph';
import {decodePackedNumbers} from '../../oracles/parity-wp4-round3-fixtures';
import {compareRoundFour} from '../../oracles/parity-wp4-round4-compare';
import {samplingCases} from '../../oracles/parity-wp4-round6-fixtures';

for (const scene of samplingCases) {
  for (const omitCount of [false, true]) {
    for (const omitMask of [false, true]) {
      it(`GPURasterSampling independent SciPy parity: ${scene.name}`, async () => {
        const device = await getWebGPUTestDevice();
        requireTestDevice(device);
        const [values, validity, points] = await Promise.all(
          [scene.values, scene.validity, scene.points].map(decodePackedNumbers)
        );
        const rig = new ParityRasterGraph(device, scene.name);
        const parameters = new GPUParameterBuffer(device, {
          id: 'sampling-parameters',
          format: 'float32',
          length: 12
        });
        const activeCount = new GPUParameterBuffer(device, {
          id: 'active-count',
          format: 'uint32',
          length: 1
        });
        const outputValues = rig.output('output-values', 'float32', scene.pointCount);
        rig.graph.add(
          new GPURasterSampling({
            grid: scene.grid,
            // The independent reference already excludes these cells. Replacing their values
            // by the declared sentinel preserves that same missing-data domain.
            values: rig.input(
              'values',
              'float32',
              Float32Array.from(values, (value, index) =>
                omitMask && !validity[index] ? -9999 : value
              )
            ),
            validity: omitMask
              ? undefined
              : rig.input('validity', 'uint32', new Uint32Array(validity)),
            noDataValue: -9999,
            positions: rig.input('points', 'float32x2', new Float32Array(points)),
            pointCount: omitCount ? undefined : activeCount.importToGraph(rig.graph),
            parameters: parameters.importToGraph(rig.graph),
            output: {values: outputValues.view}
          })
        );
        const compiled = rig.graph.compile();
        try {
          for (const [frameIndex, frame] of scene.frames.entries()) {
            if (omitCount && frame.activeCount !== scene.pointCount) continue;
            parameters.write(
              getGPURasterSamplingParameterValues(
                {...frame, localOrigin: scene.localOrigin},
                scene.grid
              )
            );
            activeCount.write(new Uint32Array([frame.activeCount]));
            submitGraph(device, compiled, undefined);
            const caseName = `${scene.name}/omit-count-${omitCount}/omit-mask-${omitMask}/${frame.method}/${frame.noDataPolicy}/frame-${frameIndex}`;
            compareRoundFour(
              'GPURasterSampling',
              caseName,
              scene.reference,
              'values',
              await readFloat32(outputValues.buffer, scene.pointCount),
              await decodePackedNumbers(frame.expected.values),
              frame.method === 'nearest' ? 'E' : 'R'
            );
          }
        } finally {
          compiled.destroy();
          parameters.destroy();
          activeCount.destroy();
          rig.destroy();
        }
      });
    }
  }
}
