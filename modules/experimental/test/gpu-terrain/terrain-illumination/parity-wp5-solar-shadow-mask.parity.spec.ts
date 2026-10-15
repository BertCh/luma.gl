// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {
  GPUSolarShadowMask,
  getGPUSolarShadowMaskParameterValues
} from '../../../src/gpu-terrain/terrain-illumination/gpu-solar-shadow-mask';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {compareSolarValues, SHADOW_CASES} from './parity-wp5-solar-fixtures';

for (const fixture of SHADOW_CASES) {
  it(`GPUSolarShadowMask independent disk/irradiance parity: ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const pixelCount = fixture.width * fixture.height;
    const graph = new GPUCommandGraph(device, {id: `parity-wp5-${fixture.name}`});
    const inputs = [fixture.horizon, fixture.slope, fixture.aspect, fixture.skyViewFactor].map(
      values => createInputBuffer(device, Float32Array.from(values))
    );
    const outputs = [
      createOutputBuffer(device, pixelCount),
      createOutputBuffer(device, pixelCount)
    ];
    const parameters = new GPUParameterBuffer(device, {
      id: 'parameters',
      format: 'float32',
      length: 8,
      values: getGPUSolarShadowMaskParameterValues(fixture.settings)
    });
    try {
      const importFloat = (id: string, buffer: (typeof inputs)[number], length = pixelCount) =>
        importGraphBuffer(graph, id, buffer, 'float32', length);
      graph.add(
        new GPUSolarShadowMask({
          grid: makeGPURasterMetadata({width: fixture.width, height: fixture.height}),
          directionCount: fixture.directionCount,
          horizon: importFloat('horizon', inputs[0], fixture.horizon.length),
          slope: importFloat('slope', inputs[1]),
          aspect: importFloat('aspect', inputs[2]),
          skyViewFactor: importFloat('sky-view', inputs[3]),
          parameters: parameters.importToGraph(graph),
          output: {
            sunVisibility: importFloat('visibility', outputs[0]),
            illumination: importFloat('illumination', outputs[1])
          }
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        for (const [index, metric] of (['sunVisibility', 'illumination'] as const).entries()) {
          compareSolarValues(
            'GPUSolarShadowMask',
            fixture.name,
            fixture.reference,
            metric,
            await readFloat32(outputs[index], pixelCount),
            fixture.expected[metric],
            fixture.tolerances[metric]
          );
        }
      } finally {
        compiled.destroy();
      }
    } finally {
      parameters.destroy();
      for (const buffer of [...inputs, ...outputs]) buffer.destroy();
    }
  });
}
