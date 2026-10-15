// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {
  GPUSolarIrradiance,
  getGPUSolarIrradianceParameterValues
} from '../../../src/gpu-terrain/terrain-illumination/gpu-solar-irradiance';
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
  EXACT_SOLAR_TOLERANCE,
  IRRADIANCE_CASES
} from './parity-wp5-solar-fixtures';

for (const fixture of IRRADIANCE_CASES) {
  for (const variant of [
    {component: 'total', chunked: false},
    {component: 'total', chunked: true},
    {component: 'direct', chunked: true},
    {component: 'diffuse', chunked: true}
  ] as const) {
    const caseName = `${fixture.name}-${variant.component}-${variant.chunked ? 'chunked' : 'single'}`;
    it(`GPUSolarIrradiance pvlib parity: ${caseName}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const pixelCount = fixture.width * fixture.height;
      const sampleCapacity = fixture.sunTable.length / 4 + 3;
      const table = new Float32Array(sampleCapacity * 4);
      table.set(fixture.sunTable);
      // Deliberately nonzero spare rows verify sampleCount independently of capacity.
      for (let sample = fixture.sunTable.length / 4; sample < sampleCapacity; sample++) {
        table.set([180, 80, 100, 5000], sample * 4);
      }
      if (variant.component === 'diffuse') {
        for (let sample = 0; sample < sampleCapacity; sample++) {
          table[sample * 4 + 3] = 0;
        }
      }
      const graph = new GPUCommandGraph(device, {id: `parity-wp5-${caseName}`});
      const inputs = [
        fixture.horizon,
        table,
        fixture.slope,
        fixture.aspect,
        fixture.skyViewFactor
      ].map(values => createInputBuffer(device, Float32Array.from(values)));
      const outputs = Array.from({length: 3}, () => createOutputBuffer(device, pixelCount));
      const parameters = new GPUParameterBuffer(device, {
        id: 'parameters',
        format: 'float32',
        length: 8,
        values: getGPUSolarIrradianceParameterValues({
          sampleCount: fixture.sampleCount,
          angularRadiusDegrees: fixture.angularRadiusDegrees,
          diffuseIrradiance: variant.component === 'direct' ? 0 : fixture.diffuseIrradiance
        })
      });
      try {
        const importFloat = (id: string, buffer: (typeof inputs)[number], length = pixelCount) =>
          importGraphBuffer(graph, id, buffer, 'float32', length);
        graph.add(
          new GPUSolarIrradiance({
            grid: makeGPURasterMetadata({width: fixture.width, height: fixture.height}),
            directionCount: fixture.directionCount,
            horizon: importFloat('horizon', inputs[0], fixture.horizon.length),
            sunTable: importFloat('table', inputs[1], table.length),
            slope: importFloat('slope', inputs[2]),
            aspect: importFloat('aspect', inputs[3]),
            skyViewFactor: importFloat('sky-view', inputs[4]),
            sampleCapacity,
            samplesPerNode: variant.chunked
              ? Math.min(13, Math.max(1, Math.floor(sampleCapacity / 2)))
              : sampleCapacity,
            parameters: parameters.importToGraph(graph),
            output: {
              sunHours: importFloat('sun-hours', outputs[0]),
              insolation: importFloat('insolation', outputs[1]),
              outputValidity: importGraphBuffer(graph, 'validity', outputs[2], 'uint32', pixelCount)
            }
          })
        );
        const compiled = graph.compile();
        try {
          submitGraph(device, compiled, undefined);
          compareSolarValues(
            'GPUSolarIrradiance',
            caseName,
            fixture.reference,
            'sunHours',
            await readFloat32(outputs[0], pixelCount),
            fixture.expected.sunHours,
            fixture.tolerances.sunHours
          );
          compareSolarValues(
            'GPUSolarIrradiance',
            caseName,
            fixture.reference,
            `${variant.component}EnergyWhM2`,
            await readFloat32(outputs[1], pixelCount),
            fixture.expected[variant.component],
            fixture.tolerances[variant.component]
          );
          compareSolarValues(
            'GPUSolarIrradiance',
            caseName,
            'finite input mask',
            'validity',
            await readUint32(outputs[2], pixelCount),
            fixture.expectedValidity,
            EXACT_SOLAR_TOLERANCE
          );
        } finally {
          compiled.destroy();
        }
      } finally {
        parameters.destroy();
        for (const buffer of [...inputs, ...outputs]) buffer.destroy();
      }
    });
  }
}
