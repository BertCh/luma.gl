// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import type {GPUDataFrameQueryParameters} from '@luma.gl/experimental/gpu-dataframe';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../utils/require-test-device';
import {compareParity} from '../oracles/parity-wp6-compare';
import {
  dataframeFixture,
  getDataFrameBatches,
  decodeDataFramePayload,
  createParityDataFrame,
  createParityDataFrameQuery,
  readDataFrameVector,
  compareDataFrameSelection
} from '../oracles/parity-wp6-dataframe-harness';

for (const scene of dataframeFixture.cases) {
  it(`GPUDataFrame/GPUDataFrameQuery pandas parity: ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const frame = createParityDataFrame(device, scene);
    const graph = new GPUCommandGraph<GPUDataFrameQueryParameters>(device);
    const compiled = createParityDataFrameQuery(frame).select(['category', 'score']).compile(graph);
    const batches = getDataFrameBatches(scene);
    try {
      expect(compiled.table.batches.map(batch => batch.numRows)).toEqual(
        batches.map(batch => batch.length)
      );
      // GPUDataFrame's projections preserve every source value and explicit validity sidecar.
      for (const [name, field] of [
        ['category', 'category'],
        ['value', 'values']
      ] as const) {
        const actual = await readDataFrameVector(frame.table.gpuVectors[name]);
        const validity = await readDataFrameVector(frame.validity[name]!);
        for (const [batchIndex, batch] of batches.entries()) {
          const label = `${scene.name}/batch=${batchIndex}`;
          compareParity(
            'GPUDataFrame',
            label,
            dataframeFixture.reference,
            batch.length,
            name,
            actual[batchIndex],
            decodeDataFramePayload(batch[field]),
            true
          );
          compareParity(
            'GPUDataFrame',
            label,
            dataframeFixture.reference,
            batch.length,
            `${name}.validity`,
            validity[batchIndex],
            decodeDataFramePayload(
              name === 'category' ? batch.categoryValidity : batch.valueValidity
            ),
            true
          );
        }
      }
      for (let encoding = 0; encoding < 2; encoding++) {
        for (const [frameIndex, parameters] of scene.frames.entries()) {
          const commandEncoder = device.createCommandEncoder();
          compiled.encode(commandEncoder, {maximumCategory: parameters.bound});
          device.submit(commandEncoder.finish());
          await compareDataFrameSelection(
            'GPUDataFrameQuery',
            scene,
            frameIndex,
            encoding,
            compiled
          );
          const scores = await readDataFrameVector(compiled.table.gpuVectors['score']);
          const validity = await readDataFrameVector(compiled.validity.score!);
          for (const [batchIndex, batch] of batches.entries()) {
            const label = `${scene.name}/frame=${frameIndex}/encoding=${encoding}/batch=${batchIndex}`;
            // All fixture products/additions are dyadic and representable, so E is stronger than S.
            compareParity(
              'GPUDataFrameQuery',
              label,
              dataframeFixture.reference,
              batch.length,
              'score',
              scores[batchIndex],
              decodeDataFramePayload(batch.score),
              true
            );
            compareParity(
              'GPUDataFrameQuery',
              label,
              dataframeFixture.reference,
              batch.length,
              'score.validity',
              validity[batchIndex],
              decodeDataFramePayload(batch.valueValidity).map(value => Number(value !== 0)),
              true
            );
          }
        }
      }
    } finally {
      compiled.destroy();
      frame.destroy();
    }
  }, 180000);
}
