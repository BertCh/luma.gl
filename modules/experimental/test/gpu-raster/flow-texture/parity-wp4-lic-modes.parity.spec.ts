// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture, type Buffer} from '@luma.gl/core';
import {GPURasterTextureToBuffer} from '../../../src/gpu-raster/gpu-raster-texture-to-buffer';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {
  GPULineIntegralConvolution,
  getGPULineIntegralConvolutionParameterValues,
  getGPULineIntegralConvolutionUint32ParameterValues
} from '../../../src/gpu-raster/flow-texture';
import {makeGPURasterMetadata} from '../../../src/gpu-raster/raster-grid';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {compareLICMode, type LICModeCase} from '../../oracles/parity-wp4-round8-compare';
import first from '../../oracles/fixtures/parity-wp4-lic-modes-16.json';
import second from '../../oracles/fixtures/parity-wp4-lic-modes-32.json';
import third from '../../oracles/fixtures/parity-wp4-lic-modes-64.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
const licCases = [first, second, third].flatMap(
  fixture => decodeOracleFixture<{cases: LICModeCase[]}>(fixture).cases
);
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const scene of licCases) {
  for (const withTexture of [false, true]) {
    it(`GPULineIntegralConvolution independent parity: ${scene.name}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const graph = new GPUCommandGraph(device, {id: `parity-wp4-${scene.name}`});
      const buffers: Buffer[] = [];
      const trackBuffer = (buffer: Buffer) => {
        buffers.push(buffer);
        return buffer;
      };
      const length = scene.width * scene.height;
      const valuesBuffer = trackBuffer(createOutputBuffer(device, length));
      const parameters = getGPULineIntegralConvolutionParameterValues(scene.settings);
      const words = getGPULineIntegralConvolutionUint32ParameterValues(scene.settings);
      const texture = withTexture
        ? device.createTexture({
            format: 'r32float',
            width: scene.width,
            height: scene.height,
            usage: Texture.STORAGE | Texture.SAMPLE | Texture.COPY_SRC | Texture.COPY_DST
          })
        : undefined;
      const textureView = texture
        ? graph.createTextureView(
            graph.importTexture(
              {
                id: 'lic-texture',
                format: 'r32float',
                width: scene.width,
                height: scene.height,
                usage: texture.props.usage
              },
              texture
            )
          )
        : undefined;
      const readback = trackBuffer(createOutputBuffer(device, length));
      const readbackValidity = trackBuffer(createOutputBuffer(device, length));
      const fieldBuffer = trackBuffer(createInputBuffer(device, new Float32Array(scene.field)));
      graph.add(
        new GPULineIntegralConvolution({
          velocities: importGraphBuffer(
            graph,
            'field',
            fieldBuffer,
            'float32x2',
            scene.fieldWidth * scene.fieldHeight
          ),
          fieldWidth: scene.fieldWidth,
          fieldHeight: scene.fieldHeight,
          grid: makeGPURasterMetadata({width: scene.width, height: scene.height}),
          stepCount: scene.stepCount,
          parameters: importGraphBuffer(
            graph,
            'parameters',
            trackBuffer(createInputBuffer(device, parameters)),
            'float32',
            parameters.length
          ),
          uint32Parameters: importGraphBuffer(
            graph,
            'words',
            trackBuffer(createInputBuffer(device, words)),
            'uint32',
            words.length
          ),
          output: {
            values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', length),
            texture: textureView
          }
        })
      );
      if (textureView)
        new GPURasterTextureToBuffer({
          input: {id: 'texture', format: 'float32', storage: {kind: 'texture', view: textureView}},
          output: importGraphBuffer(graph, 'texture-values', readback, 'float32', length),
          outputValidity: importGraphBuffer(
            graph,
            'texture-validity',
            readbackValidity,
            'uint32',
            length
          )
        }).addToGraph(graph);
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        compareLICMode(scene, await readFloat32(valuesBuffer, length), `texture-${withTexture}/`);
        if (withTexture)
          compareLICMode(scene, await readFloat32(readback, length), 'texture-readback/');
      } finally {
        compiled.destroy();
        texture?.destroy();
        for (const buffer of buffers) buffer.destroy();
      }
    });
  }
}
