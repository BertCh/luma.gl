// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {makeGPURasterBand, makeGPURasterMetadata} from '../../../src/gpu-raster';
import {GPUTerrainPeakProminence} from '../../../src/gpu-terrain';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixture from '../../oracles/terrain-peak-prominence.json';
import {getSerialProminence} from './terrain-peak-prominence-model';

it('matches NumPy prominence and serial parentPeak ties on plateaus, barriers and cyclic spill graphs on every execution', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const testCase of fixture.cases) {
    const grid = makeGPURasterMetadata(testCase);
    const count = testCase.width * testCase.height;
    const elevation = createInputBuffer(device, Float32Array.from(testCase.elevation.map(Number)));
    const prominence = createOutputBuffer(device, count);
    const saddle = createOutputBuffer(device, count);
    const parent = createOutputBuffer(device, count);
    const graph = new GPUCommandGraph(device);
    graph.add(
      new GPUTerrainPeakProminence({
        grid,
        elevation: makeGPURasterBand(
          importGraphBuffer(graph, 'elevation', elevation, 'float32', count),
          grid
        ),
        output: {
          prominence: importGraphBuffer(graph, 'prominence', prominence, 'float32', count),
          keySaddle: importGraphBuffer(graph, 'saddle', saddle, 'float32', count),
          parentPeak: importGraphBuffer(graph, 'parent', parent, 'uint32', count)
        }
      })
    );
    const compiled = graph.compile();
    try {
      for (let execution = 0; execution < 2; execution++) {
        submitGraph(device, compiled, undefined);
        const actual = await readFloat32(prominence, count);
        expect(actual, testCase.name).toEqual(testCase.prominence.map(Number));
        expect(await readFloat32(saddle, count), testCase.name).toEqual(
          testCase.keySaddle.map(Number)
        );
        expect(await readUint32(parent, count), testCase.name).toEqual(
          getSerialProminence(testCase).parentPeaks
        );
      }
    } finally {
      compiled.destroy();
      for (const buffer of [elevation, prominence, saddle, parent]) buffer.destroy();
    }
  }
});
