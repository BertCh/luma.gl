// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUSolarPosition} from '../../../src/gpu-terrain/terrain-illumination/gpu-solar-position';
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
  compareSolarValues,
  reportSolarModelDifference,
  EXACT_SOLAR_TOLERANCE,
  POSITION_FIXTURE
} from './parity-wp5-solar-fixtures';

// Compare the published NOAA algorithm; report SPA model error separately.
// Per-row angle bounds derive from orbital-angle f32 ulps and zenith conditioning.
for (const fixture of POSITION_FIXTURE.cases) {
  it(`GPUSolarPosition published NOAA parity: ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rowCount = fixture.positions.length / 2;
    const graph = new GPUCommandGraph(device, {id: 'parity-wp5-solar-position'});
    const positions = createInputBuffer(device, Float32Array.from(fixture.positions));
    const outputs = Array.from({length: 3}, () => createOutputBuffer(device, rowCount));
    const parameters = new GPUParameterBuffer(device, {
      id: 'parameters',
      format: 'float32',
      length: 4,
      values: Float32Array.from(fixture.parameters)
    });
    try {
      graph.add(
        new GPUSolarPosition({
          refraction: false,
          positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', rowCount),
          parameters: parameters.importToGraph(graph),
          output: {
            azimuth: importGraphBuffer(graph, 'azimuth', outputs[0], 'float32', rowCount),
            altitude: importGraphBuffer(graph, 'altitude', outputs[1], 'float32', rowCount),
            daylight: importGraphBuffer(graph, 'daylight', outputs[2], 'uint32', rowCount)
          }
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        const azimuth = await readFloat32(outputs[0], rowCount);
        const altitude = await readFloat32(outputs[1], rowCount);
        const daylight = await readUint32(outputs[2], rowCount);
        for (const [metric, values] of [
          ['azimuth', azimuth],
          ['altitude', altitude],
          ['zenith', altitude.map(value => 90 - value)]
        ] as const) {
          compareSolarValues(
            'GPUSolarPosition',
            fixture.name,
            fixture.reference,
            metric,
            values,
            fixture.expected[metric],
            fixture.tolerance,
            fixture.angleBounds
          );
        }
        reportSolarModelDifference('GPUSolarPosition', fixture, azimuth, altitude, daylight);
        compareSolarValues(
          'GPUSolarPosition',
          fixture.name,
          fixture.reference,
          'daylight',
          daylight,
          fixture.expected.daylight,
          EXACT_SOLAR_TOLERANCE
        );
      } finally {
        compiled.destroy();
      }
    } finally {
      parameters.destroy();
      positions.destroy();
      for (const buffer of outputs) buffer.destroy();
    }
  });
}
