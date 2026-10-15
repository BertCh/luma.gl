// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {makeGPURasterMetadata} from '../../src/gpu-raster';
import {
  GPUTerrainCurvature,
  getGPUTerrainCurvatureParameterValues,
  type GPUTerrainCurvatureKind
} from '../../src/gpu-terrain/terrain-curvature';
import {GPUParameterBuffer, importGraphBuffer} from '../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';
import {CURVATURE_FIXTURES} from './parity-wp5-round8-curvature-fixtures';
import {measureComparison, assertComparisons} from './parity-wp5-round8-fixtures';
for (const fixture of CURVATURE_FIXTURES)
  for (const settings of fixture.cases)
    it(`GPUTerrainCurvature row-metric tensor parity ${settings.name}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const {width, height, cellSize, north} = fixture.input;
      const count = width * height;
      const grid = makeGPURasterMetadata({width, height, cellSize, origin: [0, north]});
      const graph = new GPUCommandGraph(device, {id: 'parity-wp5-r8-curvature'});
      const input = createInputBuffer(device, Float32Array.from(fixture.input.elevation));
      const outputValidity = createOutputBuffer(device, count);
      const results = Object.entries(settings.expected).map(([kind, expected]) => ({
        kind: kind as GPUTerrainCurvatureKind,
        expected,
        buffer: createOutputBuffer(device, count)
      }));
      const curvatures: Partial<Record<GPUTerrainCurvatureKind, GraphDataView<'float32'>>> = {};
      for (const {kind, buffer} of results)
        curvatures[kind] = importGraphBuffer(graph, kind, buffer, 'float32', count);
      const parameters = new GPUParameterBuffer(device, {
        id: 'settings',
        format: 'float32',
        length: 12,
        values: getGPUTerrainCurvatureParameterValues({}, grid)
      });
      graph.add(
        new GPUTerrainCurvature({
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
          method: settings.method,
          borderMode: settings.borderMode,
          cellSizeMode: 'web-mercator',
          output: {
            curvatures,
            outputValidity: importGraphBuffer(graph, 'validity', outputValidity, 'uint32', count)
          }
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        const errors = [];
        for (const {kind, expected, buffer} of results)
          errors.push(
            measureComparison(
              'GPUTerrainCurvature',
              settings.name,
              'composite scipy.linalg.eigh(II,I), polynomial fits; Minar 2020',
              kind,
              await readFloat32(buffer, count),
              expected,
              'R'
            )
          );
        errors.push(
          measureComparison(
            'GPUTerrainCurvature',
            settings.name,
            'scipy.ndimage.minimum_filter',
            'validity',
            await readUint32(outputValidity, count),
            settings.validity
          )
        );
        assertComparisons(errors);
      } finally {
        compiled.destroy();
        parameters.destroy();
        input.destroy();
        outputValidity.destroy();
        for (const {buffer} of results) buffer.destroy();
      }
    });
