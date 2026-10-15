// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {
  GPUSimpleLocalRelief,
  getGPUSimpleLocalReliefParameterValues
} from '../../../src/gpu-terrain/relief-visualization/gpu-simple-local-relief';
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
  RELIEF_CASES,
  RELIEF_REFERENCE,
  EXACT_TOLERANCE,
  compareRound5Values
} from '../parity-wp5-round5-fixtures';

for (const fixture of RELIEF_CASES) {
  it(`GPUSimpleLocalRelief independent reference: ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const count = fixture.width * fixture.height;
    const graph = new GPUCommandGraph(device, {id: 'parity-wp5-slrm'});
    const inputs = [
      createInputBuffer(device, Float32Array.from(fixture.elevation)),
      createInputBuffer(device, Uint32Array.from(fixture.validity))
    ];
    const outputs = {
      relief: createOutputBuffer(device, count),
      validity: createOutputBuffer(device, count)
    };
    const parameters = new GPUParameterBuffer(device, {
      id: 'settings',
      format: 'float32',
      length: 4,
      values: getGPUSimpleLocalReliefParameterValues({
        verticalExaggeration: fixture.verticalExaggeration,
        radius: fixture.defaultRadius ? undefined : fixture.radius
      })
    });
    try {
      graph.add(
        new GPUSimpleLocalRelief({
          grid: makeGPURasterMetadata({width: fixture.width, height: fixture.height}),
          maximumRadius: fixture.maximumRadius,
          elevation: {
            id: 'elevation',
            format: 'float32',
            noDataValue: -9999,
            validity: importGraphBuffer(graph, 'input-validity', inputs[1], 'uint32', count),
            storage: {
              kind: 'buffer',
              values: importGraphBuffer(graph, 'elevation', inputs[0], 'float32', count)
            }
          },
          parameters: parameters.importToGraph(graph),
          output: {
            relief: importGraphBuffer(graph, 'relief', outputs.relief, 'float32', count),
            outputValidity: importGraphBuffer(graph, 'validity', outputs.validity, 'uint32', count)
          }
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        compareRound5Values(
          'GPUSimpleLocalRelief',
          fixture.name,
          RELIEF_REFERENCE,
          'relief',
          await readFloat32(outputs.relief, count),
          fixture.expected,
          fixture.tolerance
        );
        compareRound5Values(
          'GPUSimpleLocalRelief',
          fixture.name,
          RELIEF_REFERENCE,
          'validity',
          await readUint32(outputs.validity, count),
          fixture.expectedValidity,
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
