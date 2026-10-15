// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import type {GPUDataFrameQueryParameters} from '@luma.gl/experimental/gpu-dataframe';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {requireTestDevice} from '../utils/require-test-device';
import {compareParity} from '../oracles/parity-wp6-compare';
import {
  dataframeFixture,
  getDataFrameBatches,
  createParityDataFrame,
  createParityDataFrameQuery,
  readDataFrameVector,
  compareDataFrameSelection
} from '../oracles/parity-wp6-dataframe-harness';

for (const scene of dataframeFixture.cases) {
  for (const irregular of [false, true]) {
    it(`GPUDataFrameHistogramQuery NumPy parity: ${scene.name}/irregular=${irregular}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const frame = createParityDataFrame(device, scene);
      const graph = new GPUCommandGraph<GPUDataFrameQueryParameters>(device);
      const compiled = createParityDataFrameQuery(frame)
        .histogram(
          'score',
          irregular ? {edges: scene.edges} : {bins: scene.binCount, domain: scene.domain}
        )
        .compile(graph);
      const rowCount = getDataFrameBatches(scene).reduce((total, batch) => total + batch.length, 0);
      try {
        for (let encoding = 0; encoding < 2; encoding++) {
          for (const [frameIndex, parameters] of scene.frames.entries()) {
            const commandEncoder = device.createCommandEncoder();
            compiled.encode(commandEncoder, {maximumCategory: parameters.bound});
            device.submit(commandEncoder.finish());
            await compareDataFrameSelection(
              'GPUDataFrameHistogramQuery',
              scene,
              frameIndex,
              encoding,
              compiled
            );
            const expected = irregular ? parameters.histogram : parameters.equalHistogram;
            const label = `${scene.name}/irregular=${irregular}/frame=${frameIndex}/encoding=${encoding}`;
            compareParity(
              'GPUDataFrameHistogramQuery',
              label,
              dataframeFixture.reference,
              rowCount,
              'binIds',
              (await readDataFrameVector(compiled.table.gpuVectors['bin'])).flat(),
              expected.map((_, index) => index),
              true
            );
            compareParity(
              'GPUDataFrameHistogramQuery',
              label,
              dataframeFixture.reference,
              rowCount,
              'counts',
              (await readDataFrameVector(compiled.table.gpuVectors['count'])).flat(),
              expected,
              true
            );
          }
        }
      } finally {
        compiled.destroy();
        frame.destroy();
      }
    }, 180000);
  }
}
