// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {
  GPUTerrainCurvature,
  getGPUTerrainCurvatureParameterValues
} from '../../../src/gpu-terrain/terrain-curvature';
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
  compareCurvatureValues,
  CURVATURE_CASES,
  CURVATURE_KINDS
} from './parity-wp5-curvature-fixtures';

for (const fixture of CURVATURE_CASES) {
  it(`GPUTerrainCurvature external parity: ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const pixelCount = fixture.width * fixture.height;
    const grid = makeGPURasterMetadata({
      width: fixture.width,
      height: fixture.height,
      cellSize: fixture.cellSize
    });
    const graph = new GPUCommandGraph(device, {id: `parity-wp5-${fixture.name}`});
    const elevation = createInputBuffer(device, Float32Array.from(fixture.elevation));
    const inputValidity = createInputBuffer(device, Uint32Array.from(fixture.inputValidity));
    const outputValidity = createOutputBuffer(device, pixelCount);
    const outputBuffers = CURVATURE_KINDS.map(kind => ({
      kind,
      buffer: createOutputBuffer(device, pixelCount)
    }));
    const parameters = new GPUParameterBuffer(device, {
      id: 'curvature-parameters',
      format: 'float32',
      length: 12,
      values: getGPUTerrainCurvatureParameterValues({zFactor: fixture.zFactor}, grid)
    });
    try {
      graph.add(
        new GPUTerrainCurvature({
          grid,
          method: fixture.method,
          borderMode: fixture.borderMode,
          rowDirection: fixture.rowDirection,
          cellSizeMode: 'uniform',
          elevation: {
            id: 'elevation',
            format: 'float32',
            storage: {
              kind: 'buffer',
              values: importGraphBuffer(graph, 'elevation', elevation, 'float32', pixelCount)
            },
            validity: importGraphBuffer(
              graph,
              'input-validity',
              inputValidity,
              'uint32',
              pixelCount
            )
          },
          parameters: parameters.importToGraph(graph),
          output: {
            curvatures: Object.fromEntries(
              outputBuffers.map(({kind, buffer}) => [
                kind,
                importGraphBuffer(graph, kind, buffer, 'float32', pixelCount)
              ])
            ),
            outputValidity: importGraphBuffer(
              graph,
              'output-validity',
              outputValidity,
              'uint32',
              pixelCount
            )
          }
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        // Compare every output cell, including halo edges; no summary-only assertions.
        // These adjoining tiles were checked against a united-window reference by the generator.
        for (const {kind, buffer} of outputBuffers) {
          compareCurvatureValues(
            fixture.name,
            fixture.references[kind],
            kind,
            await readFloat32(buffer, pixelCount),
            fixture.expected[kind],
            fixture.tolerances[kind]
          );
        }
        compareCurvatureValues(
          fixture.name,
          'scipy.ndimage.minimum_filter (full fitting-window validity)',
          'validity',
          await readUint32(outputValidity, pixelCount),
          fixture.expectedValidity,
          {profile: 'E', absolute: 0, relative: 0}
        );
      } finally {
        compiled.destroy();
      }
    } finally {
      parameters.destroy();
      elevation.destroy();
      inputValidity.destroy();
      outputValidity.destroy();
      for (const {buffer} of outputBuffers) {
        buffer.destroy();
      }
    }
  });
}
