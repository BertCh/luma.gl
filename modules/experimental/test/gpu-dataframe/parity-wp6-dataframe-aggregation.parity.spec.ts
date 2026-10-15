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
  for (const grouped of [false, true]) {
    const className = grouped
      ? 'GPUDataFrameGroupedAggregationQuery'
      : 'GPUDataFrameAggregationQuery';
    it(`${className} pandas parity: ${scene.name}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const frame = createParityDataFrame(device, scene);
      const graph = new GPUCommandGraph<GPUDataFrameQueryParameters>(device);
      const query = createParityDataFrameQuery(frame);
      const definitions = {
        count: 'count',
        sum: {sum: 'score'},
        minimum: {min: 'score'},
        maximum: {max: 'score'},
        mean: {mean: 'score'}
      } as const;
      const compiled = (
        grouped ? query.groupBy('category').aggregate(definitions) : query.aggregate(definitions)
      ).compile(graph);
      const rowCount = getDataFrameBatches(scene).reduce((total, batch) => total + batch.length, 0);
      try {
        for (let encoding = 0; encoding < 2; encoding++) {
          for (const [frameIndex, parameters] of scene.frames.entries()) {
            const commandEncoder = device.createCommandEncoder();
            compiled.encode(commandEncoder, {maximumCategory: parameters.bound});
            device.submit(commandEncoder.finish());
            await compareDataFrameSelection(className, scene, frameIndex, encoding, compiled);
            const expected = grouped ? parameters.groups : parameters.global;
            const label = `${scene.name}/frame=${frameIndex}/encoding=${encoding}`;
            for (const [metric, vector] of Object.entries(compiled.table.gpuVectors)) {
              const actual = (await readDataFrameVector(vector)).flat();
              const exact = ['count', 'category', 'minimum', 'maximum'].includes(metric);
              compareParity(
                className,
                label,
                dataframeFixture.reference,
                rowCount,
                metric,
                actual,
                exact ? expected[metric].map(Math.fround) : expected[metric],
                exact
              );
            }
            for (const [metric, vector] of Object.entries(compiled.validity)) {
              if (!vector) continue;
              compareParity(
                className,
                label,
                dataframeFixture.reference,
                rowCount,
                `${metric}.validity`,
                (await readDataFrameVector(vector)).flat(),
                expected['validity'],
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
}
