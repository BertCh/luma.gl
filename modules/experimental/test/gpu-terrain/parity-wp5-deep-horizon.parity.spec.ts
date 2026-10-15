// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import fixture from '../oracles/fixtures/parity-wp5-deep-horizon.json';
import {makeGPURasterMetadata} from '../../src/gpu-raster';
import {GPUParameterBuffer, importGraphBuffer} from '../../src/utils/gpu-contributor-utils';
import {
  GPUTerrainHorizon,
  getGPUTerrainHorizonParameterValues
} from '../../src/gpu-terrain/terrain-illumination';
import {
  GPUPointHorizonProfile,
  getGPUPointHorizonProfileParameterValues
} from '../../src/gpu-terrain/point-horizon';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';
import {measureComparison, assertComparisons} from './parity-wp5-round8-fixtures';

for (const settings of fixture.cases) {
  for (const algorithm of ['march', 'sweep'] as const) {
    it(`GPUTerrainHorizon minimal GRASS ${settings.name}/${algorithm}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const {width, height, cellSize} = settings;
      const count = width * height;
      const grid = makeGPURasterMetadata({width, height, cellSize});
      const graph = new GPUCommandGraph(device, {id: 'parity-wp5-deep-horizon'});
      const input = createInputBuffer(device, Float32Array.from(settings.elevation));
      const output = createOutputBuffer(device, count * 4);
      const parameters = new GPUParameterBuffer(device, {
        id: 'settings',
        format: 'float32',
        length: 8,
        values: getGPUTerrainHorizonParameterValues(settings, grid)
      });
      graph.add(
        new GPUTerrainHorizon({
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
          directionCount: 4,
          maximumRadius: settings.maximumRadius,
          algorithm,
          output: {horizon: importGraphBuffer(graph, 'horizon', output, 'float32', count * 4)}
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        assertComparisons([
          measureComparison(
            'GPUTerrainHorizon',
            `${settings.name}/${algorithm}`,
            'GRASS r.horizon matched cap and nodata region',
            'nonnegativeHorizonDegrees',
            (await readFloat32(output, count * 4)).map(value => Math.max(0, value)),
            settings.horizon,
            'D'
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
  if (!settings.pointHorizon) continue;
  for (const traversal of ['march', 'pyramid'] as const) {
    it(`GPUPointHorizonProfile minimal ${settings.name}/${traversal}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const {width, height, cellSize} = settings;
      const count = width * height;
      const grid = makeGPURasterMetadata({width, height, cellSize});
      const graph = new GPUCommandGraph(device, {id: 'parity-wp5-deep-point'});
      const input = createInputBuffer(device, Float32Array.from(settings.elevation));
      const angles = createOutputBuffer(device, 4);
      const distances = createOutputBuffer(device, 4);
      const observers = createInputBuffer(device, Float32Array.of(4, 4, 0, 0));
      const parameters = new GPUParameterBuffer(device, {
        id: 'settings',
        format: 'float32',
        length: 8,
        values: getGPUPointHorizonProfileParameterValues(settings, grid)
      });
      graph.add(
        new GPUPointHorizonProfile({
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
          observers: importGraphBuffer(graph, 'observers', observers, 'float32x4', 1),
          azimuthCount: 4,
          minimumDistance: settings.minimumDistance!,
          maximumDistance: settings.maximumDistance!,
          cellSize,
          ...settings.latticeOptions,
          traversal,
          output: {
            skylineAngle: importGraphBuffer(graph, 'angles', angles, 'float32', 4),
            distance: importGraphBuffer(graph, 'distance', distances, 'float32', 4)
          }
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        const actualAngles = (await readFloat32(angles, 4)).map(value => Math.max(0, value));
        const actualDistances = (await readFloat32(distances, 4)).map((value, index) =>
          actualAngles[index] > 0 ? value : 0
        );
        const reference =
          settings.name === 'point-matched'
            ? 'GRASS r.horizon matched cell-centre lattice'
            : 'composite SciPy interp1d dyadic lattice';
        const expectedAngles =
          settings.name === 'point-matched'
            ? settings.horizon.slice((4 * width + 4) * 4, (4 * width + 4) * 4 + 4)
            : settings.pointHorizon!;
        assertComparisons([
          measureComparison(
            'GPUPointHorizonProfile',
            `${settings.name}/${traversal}`,
            reference,
            'nonnegativeHorizonDegrees',
            actualAngles,
            expectedAngles,
            'D'
          ),
          // Same D budget as Round 10; these exact dyadic distances need no enlarged tolerance.
          measureComparison(
            'GPUPointHorizonProfile',
            `${settings.name}/${traversal}`,
            'SciPy argmax on declared lattice',
            'positiveHorizonDistanceMeters',
            actualDistances,
            settings.pointDistance!,
            'D',
            {absolute: 1e-3, relative: 2e-6}
          )
        ]);
      } finally {
        compiled.destroy();
        parameters.destroy();
        for (const buffer of [input, angles, distances, observers]) buffer.destroy();
      }
    });
  }
}
