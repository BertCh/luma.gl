// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {type Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {makeGPURasterMetadata} from '../../src/gpu-raster';
import {GPUParameterBuffer, importGraphBuffer} from '../../src/utils/gpu-contributor-utils';
import {
  GPUTerrainTopographicPosition,
  GPUTerrainVectorRuggedness,
  GPUTerrainWeissLandforms,
  getGPUTerrainVectorRuggednessParameterValues,
  getGPUTerrainWeissLandformsParameterValues
} from '../../src/gpu-terrain/topographic-position';
import {
  GPUTextureShading,
  getGPUTextureShadingParameterValues
} from '../../src/gpu-terrain/terrain-illumination';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';
import {
  TPI_FIXTURES,
  VRM_FIXTURES,
  WEISS_FIXTURES,
  TEXTURE_FIXTURES
} from './parity-wp5-round8-composite-fixtures';
import {measureComparison, assertComparisons} from './parity-wp5-round8-fixtures';

for (const [className, fixtures] of [
  ['GPUTerrainTopographicPosition', TPI_FIXTURES],
  ['GPUTerrainVectorRuggedness', VRM_FIXTURES],
  ['GPUTerrainWeissLandforms', WEISS_FIXTURES],
  ['GPUTextureShading', TEXTURE_FIXTURES]
] as const) {
  for (const fixture of fixtures)
    for (const settings of fixture.cases) {
      // Both valid-data optimization paths are independent comparisons to the same definition.
      const nodataModes =
        className === 'GPUTextureShading' && fixture.input.validity.every(Boolean)
          ? [false, true]
          : [true];
      for (const hasNodata of nodataModes)
        it(`${className} published composite ${settings.name}/nodata-${hasNodata}`, async () => {
          const device = await getWebGPUTestDevice();
          requireTestDevice(device);
          const {width, height, cellSize, north} = fixture.input;
          const count = width * height;
          const grid = makeGPURasterMetadata({width, height, cellSize, origin: [0, north]});
          const graph = new GPUCommandGraph(device, {id: 'parity-wp5-r8-composite'});
          const input = createInputBuffer(device, Float32Array.from(fixture.input.elevation));
          const validity = createInputBuffer(device, Uint32Array.from(fixture.input.validity));
          const buffers: Buffer[] = [input, validity];
          const floats: Record<string, GraphDataView<'float32'>> = {};
          const words: Record<string, GraphDataView<'uint32'>> = {};
          const results: {metric: string; buffer: Buffer; integer: boolean; length: number}[] = [];
          for (const [metric, expected] of Object.entries(settings.expected)) {
            const buffer = createOutputBuffer(device, expected.length);
            buffers.push(buffer);
            const integer = ['outputValidity', 'maximumDeviationRadius', 'landforms'].includes(
              metric
            );
            if (integer)
              words[metric] = importGraphBuffer(graph, metric, buffer, 'uint32', expected.length);
            else
              floats[metric] = importGraphBuffer(graph, metric, buffer, 'float32', expected.length);
            results.push({metric, buffer, integer, length: expected.length});
          }
          const common = {
            grid,
            elevation: {
              id: 'elevation',
              format: 'float32' as const,
              storage: {
                kind: 'buffer' as const,
                values: importGraphBuffer(graph, 'elevation', input, 'float32', count)
              },
              validity: importGraphBuffer(graph, 'validity', validity, 'uint32', count)
            }
          };
          const values =
            className === 'GPUTerrainVectorRuggedness'
              ? getGPUTerrainVectorRuggednessParameterValues({radius: settings.radius}, grid)
              : className === 'GPUTerrainWeissLandforms'
                ? getGPUTerrainWeissLandformsParameterValues({}, grid)
                : getGPUTextureShadingParameterValues({
                    detail: settings.detail,
                    gain: settings.gain
                  });
          const parameters = new GPUParameterBuffer(device, {
            id: 'settings',
            format: 'float32',
            length: values.length,
            values
          });
          if (className === 'GPUTerrainTopographicPosition')
            graph.add(
              new GPUTerrainTopographicPosition({
                ...common,
                scales: settings.scales!,
                quantum: settings.quantum,
                output: {
                  topographicPositionIndex: floats['topographicPositionIndex'],
                  deviationFromMean: floats['deviationFromMean'],
                  maximumDeviation: floats['maximumDeviation'],
                  maximumDeviationRadius: words['maximumDeviationRadius'],
                  outputValidity: words['outputValidity']
                }
              })
            );
          else if (className === 'GPUTerrainVectorRuggedness')
            graph.add(
              new GPUTerrainVectorRuggedness({
                ...common,
                parameters: parameters.importToGraph(graph),
                maximumRadius: settings.radius,
                cellSizeMode: settings.cellSizeMode,
                borderMode: settings.borderMode,
                output: {
                  vectorRuggedness: floats['vectorRuggedness'],
                  outputValidity: words['outputValidity']
                }
              })
            );
          else if (className === 'GPUTerrainWeissLandforms')
            graph.add(
              new GPUTerrainWeissLandforms({
                ...common,
                parameters: parameters.importToGraph(graph),
                standardization: settings.standardization,
                smallScale: settings.smallScale,
                largeScale: settings.largeScale,
                cellSizeMode: settings.cellSizeMode,
                output: {
                  landforms: words['landforms'],
                  standardizedPosition: floats['standardizedPosition'],
                  outputValidity: words['outputValidity']
                }
              })
            );
          else
            graph.add(
              new GPUTextureShading({
                ...common,
                parameters: parameters.importToGraph(graph),
                levelCount: settings.levelCount,
                baseSigma: settings.baseSigma,
                hasNodata,
                downsampleLevels: false,
                output: {
                  textureShade: floats['textureShade'],
                  outputValidity: words['outputValidity']
                }
              })
            );
          const compiled = graph.compile();
          try {
            submitGraph(device, compiled, undefined);
            const errors = [];
            for (const {metric, buffer, integer, length} of results) {
              const actual = await (integer
                ? readUint32(buffer, length)
                : readFloat32(buffer, length));
              errors.push(
                measureComparison(
                  className,
                  `${settings.name}/nodata-${hasNodata}`,
                  'composite SciPy/NumPy; citations in fixture',
                  metric,
                  actual,
                  settings.expected[metric],
                  integer ? 'E' : className === 'GPUTextureShading' ? 'L' : 'R'
                )
              );
            }
            assertComparisons(errors);
          } finally {
            compiled.destroy();
            parameters.destroy();
            for (const buffer of buffers) buffer.destroy();
          }
        });
    }
}
