// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUTerrainFlowField,
  getGPUTerrainFlowFieldParameterValues
} from '../../../src/gpu-terrain/terrain-flow-field';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {FLOW_CASES, FLOW_REFERENCE, compareValues} from '../parity-wp5-round6-fixtures';

for (const fixture of FLOW_CASES) {
  it(`GPUTerrainFlowField independent projection parity: ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const count = fixture.width * fixture.height;
    const grid = makeGPURasterMetadata({
      width: fixture.width,
      height: fixture.height,
      cellSize: fixture.cellSize
    });
    const graph = new GPUCommandGraph(device, {id: 'parity-wp5-flow'});
    const elevation = createInputBuffer(device, Float32Array.from(fixture.elevation));
    const validity = createInputBuffer(device, Uint32Array.from(fixture.validity));
    const velocities = createOutputBuffer(device, count * 2);
    const parameters = new GPUParameterBuffer(device, {
      id: 'parameters',
      format: 'float32',
      length: 8,
      values: getGPUTerrainFlowFieldParameterValues(
        {wind: fixture.winds[0], verticalExaggeration: fixture.exaggeration},
        grid
      )
    });
    graph.add(
      new GPUTerrainFlowField({
        grid,
        parameters: parameters.importToGraph(graph),
        elevation: {
          id: 'elevation',
          format: 'float32',
          noDataValue: -9999,
          scale: fixture.scale,
          offset: fixture.offset,
          storage: {
            kind: 'buffer',
            values: importGraphBuffer(graph, 'elevation', elevation, 'float32', count)
          },
          validity: importGraphBuffer(graph, 'validity', validity, 'uint32', count)
        },
        output: {velocities: importGraphBuffer(graph, 'velocities', velocities, 'float32x2', count)}
      })
    );
    const compiled = graph.compile();
    try {
      for (const [windIndex, wind] of fixture.winds.entries()) {
        parameters.write(
          getGPUTerrainFlowFieldParameterValues(
            {wind, verticalExaggeration: fixture.exaggeration},
            grid
          )
        );
        submitGraph(device, compiled, undefined);
        const actual = await readFloat32(velocities, count * 2);
        for (const component of [0, 1])
          compareValues(
            'GPUTerrainFlowField',
            fixture.name,
            FLOW_REFERENCE,
            `wind-${windIndex}/component-${component}`,
            actual.filter((_, index) => index % 2 === component),
            fixture.expected[windIndex].filter((_, index) => index % 2 === component),
            fixture.tolerances[windIndex][component]
          );
      }
    } finally {
      compiled.destroy();
      parameters.destroy();
      for (const buffer of [elevation, validity, velocities]) buffer.destroy();
    }
  });
}
