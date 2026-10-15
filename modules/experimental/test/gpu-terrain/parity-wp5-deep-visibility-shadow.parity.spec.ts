// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {makeGPURasterMetadata} from '../../src/gpu-raster';
import {
  GPUTerrainCastShadow,
  getGPUTerrainCastShadowParameterValues
} from '../../src/gpu-terrain/terrain-illumination/gpu-terrain-cast-shadow';
import {GPUParameterBuffer, importGraphBuffer} from '../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';
import {measureComparison, assertComparisons} from './parity-wp5-round8-fixtures';
import fixture from '../oracles/fixtures/parity-wp5-deep-saddle-visibility.json';

for (const settings of fixture.shadows) {
  it(`GPUTerrainCastShadow minimal GRASS saddle/${settings.azimuthDegrees}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const {width, height, cellSize, elevation} = fixture.input;
    const count = width * height;
    const grid = makeGPURasterMetadata({width, height, cellSize});
    const graph = new GPUCommandGraph(device, {id: 'parity-wp5-deep-shadow'});
    const input = createInputBuffer(device, Float32Array.from(elevation));
    const output = createOutputBuffer(device, count);
    const parameters = new GPUParameterBuffer(device, {
      id: 'settings',
      format: 'float32',
      length: 16,
      values: getGPUTerrainCastShadowParameterValues({...settings, angularRadiusDegrees: 0}, grid)
    });
    graph.add(
      new GPUTerrainCastShadow({
        grid,
        elevation: {
          id: 'elevation',
          format: 'float32',
          storage: {
            kind: 'buffer',
            values: importGraphBuffer(graph, 'elevation', input, 'float32', count)
          }
        },
        parameters: parameters.importToGraph(graph),
        output: {sunVisibility: importGraphBuffer(graph, 'output', output, 'float32', count)}
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      assertComparisons([
        measureComparison(
          'GPUTerrainCastShadow',
          `saddle/${settings.azimuthDegrees}`,
          'GRASS 8.5 r.sunmask -z',
          'hardSunVisibility',
          await readFloat32(output, count),
          settings.sunVisibility
        )
      ]);
    } finally {
      compiled.destroy();
      parameters.destroy();
      input.destroy();
      output.destroy();
    }
  });
}
