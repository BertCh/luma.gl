// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {
  GPUWeightedOverlay,
  getGPUWeightedOverlayParameterLength,
  getGPUWeightedOverlayParameterValues
} from '../../../src/gpu-raster/raster-algebra';
import {GPUParameterBuffer} from '../../../src/utils/gpu-contributor-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {ParityRasterGraph} from '../../oracles/parity-wp4-round3-graph';
import {decodePackedNumbers} from '../../oracles/parity-wp4-round3-fixtures';
import {compareRoundFour} from '../../oracles/parity-wp4-round4-compare';
import {overlayCases} from '../../oracles/parity-wp4-round7-fixtures';

for (const scene of overlayCases) {
  for (const tableFree of [false, true]) {
    for (const withValidity of [false, true]) {
      for (const withRange of [false, true]) {
        const frames = tableFree
          ? scene.frames.filter(frame =>
              frame.settings.layers.every(layer => layer.mode !== 'table')
            )
          : scene.frames;
        if (!frames.length) continue;
        it(`GPUWeightedOverlay independent NumPy parity: ${scene.name}`, async () => {
          const device = await getWebGPUTestDevice();
          requireTestDevice(device);
          const rig = new ParityRasterGraph(device, scene.name);
          const parameters = new GPUParameterBuffer(device, {
            id: 'parameters',
            format: 'float32',
            length: getGPUWeightedOverlayParameterLength(scene.layerCount)
          });
          const [stack, remapBreaks, remapValues] = await Promise.all(
            [scene.stack, scene.remapBreaks, scene.remapValues].map(decodePackedNumbers)
          );
          const outputs = {
            score: rig.output('score', 'float32', scene.cellCount),
            outputValidity: rig.output('validity', 'uint32', scene.cellCount),
            scoreRange: rig.output('range', 'float32', 2)
          };
          rig.graph.add(
            new GPUWeightedOverlay({
              stack: rig.input('stack', 'float32', new Float32Array(stack)),
              layerCount: scene.layerCount,
              cellCount: scene.cellCount,
              noDataValue: -9999,
              parameters: parameters.importToGraph(rig.graph),
              remapBreaks: tableFree
                ? undefined
                : rig.input('breaks', 'float32', new Float32Array(remapBreaks)),
              remapValues: tableFree
                ? undefined
                : rig.input('remap-values', 'float32', new Float32Array(remapValues)),
              maximumBreakCount: tableFree ? undefined : 4,
              output: {
                score: outputs.score.view,
                scoreRange: withRange ? outputs.scoreRange.view : undefined,
                outputValidity: withValidity ? outputs.outputValidity.view : undefined
              }
            })
          );
          const compiled = rig.graph.compile();
          try {
            for (const [frameIndex, frame] of frames.entries()) {
              parameters.write(getGPUWeightedOverlayParameterValues(frame.settings));
              submitGraph(device, compiled, undefined);
              for (const metric of ['score', 'scoreRange', 'outputValidity'] as const) {
                if (
                  (metric === 'scoreRange' && !withRange) ||
                  (metric === 'outputValidity' && !withValidity)
                )
                  continue;
                const expected = await decodePackedNumbers(frame.expected[metric]);
                compareRoundFour(
                  'GPUWeightedOverlay',
                  `${scene.name}/table-free-${tableFree}/range-${withRange}/validity-${withValidity}/frame-${frameIndex}`,
                  scene.reference,
                  metric,
                  await (metric === 'outputValidity' ? readUint32 : readFloat32)(
                    outputs[metric].buffer,
                    expected.length
                  ),
                  expected,
                  metric === 'outputValidity' ? 'E' : 'R'
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
    }
  }
}
