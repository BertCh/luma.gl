// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {
  GPUReliefBlend,
  getGPUReliefBlendParameterValues
} from '../../../src/gpu-terrain/relief-visualization/gpu-relief-blend';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  BLEND_CASES,
  BLEND_REFERENCE,
  EXACT_TOLERANCE,
  compareRound5Values
} from '../parity-wp5-round5-fixtures';

for (const fixture of BLEND_CASES) {
  it(`GPUReliefBlend RVT kernel parity: ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const count = fixture.width * fixture.height;
    const graph = new GPUCommandGraph(device, {id: 'parity-wp5-blend'});
    const inputs = fixture.layers.map(layer => createInputBuffer(device, Float32Array.from(layer)));
    const outputs = {
      blend: createOutputBuffer(device, count),
      color: createOutputBuffer(device, count),
      validity: createOutputBuffer(device, count)
    };
    const parameters = new GPUParameterBuffer(device, {
      id: 'settings',
      format: 'float32',
      length: fixture.settings.length * 5,
      values: getGPUReliefBlendParameterValues(fixture.settings)
    });
    try {
      graph.add(
        new GPUReliefBlend({
          grid: makeGPURasterMetadata({width: fixture.width, height: fixture.height}),
          layers: inputs.map((buffer, index) =>
            importGraphBuffer(graph, `layer-${index}`, buffer, 'float32', count)
          ),
          parameters: parameters.importToGraph(graph),
          output: {
            blend: importGraphBuffer(graph, 'blend', outputs.blend, 'float32', count),
            color: importGraphBuffer(graph, 'color', outputs.color, 'uint32', count),
            outputValidity: importGraphBuffer(graph, 'validity', outputs.validity, 'uint32', count)
          }
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        compareRound5Values(
          'GPUReliefBlend',
          fixture.name,
          BLEND_REFERENCE,
          'blend',
          await readFloat32(outputs.blend, count),
          fixture.expected,
          fixture.tolerance
        );
        compareRound5Values(
          'GPUReliefBlend',
          fixture.name,
          BLEND_REFERENCE,
          'validity',
          await readUint32(outputs.validity, count),
          fixture.expectedValidity,
          EXACT_TOLERANCE
        );
        const colors = await readUint32(outputs.color, count);
        // R permits one quantization level per RGB channel. Packed-word error
        // would hide channel/alpha corruption, so compare all channels separately.
        for (const [channel, shift] of [
          ['red', 0],
          ['green', 8],
          ['blue', 16],
          ['alpha', 24]
        ] as const) {
          compareRound5Values(
            'GPUReliefBlend',
            fixture.name,
            BLEND_REFERENCE,
            channel,
            colors.map(value => (value >>> shift) & 255),
            fixture.expectedColor.map(value => (value >>> shift) & 255),
            channel === 'alpha'
              ? EXACT_TOLERANCE
              : {profile: 'R encoding', absolute: 1, relative: 0}
          );
        }
        compareRound5Values(
          'GPUReliefBlend',
          fixture.name,
          BLEND_REFERENCE,
          'invalid-transparent-black',
          colors.filter((_, index) => !fixture.expectedValidity[index]),
          fixture.expectedColor.filter((_, index) => !fixture.expectedValidity[index]),
          EXACT_TOLERANCE
        );
      } finally {
        compiled.destroy();
      }
    } finally {
      parameters.destroy();
      for (const buffer of [...inputs, ...Object.values(outputs)]) buffer.destroy();
    }
  });
}
