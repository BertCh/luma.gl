// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {
  getGPULocalDominanceParameterValues,
  GPULocalDominance
} from '../../../src/gpu-terrain/relief-visualization/gpu-local-dominance';
import {
  getGPUMultiScaleReliefParameterValues,
  GPUMultiScaleRelief
} from '../../../src/gpu-terrain/relief-visualization/gpu-multi-scale-relief';
import {
  getGPUReliefBlendParameterValues,
  GPU_RELIEF_BLEND_VAT_ARCHAEOLOGICAL,
  GPUReliefBlend
} from '../../../src/gpu-terrain/relief-visualization/gpu-relief-blend';
import {
  getGPUSimpleLocalReliefParameterValues,
  GPUSimpleLocalRelief
} from '../../../src/gpu-terrain/relief-visualization/gpu-simple-local-relief';
import {createInputBuffer, createOutputBuffer} from '../../utils/gpu-contributor-test-utils';
import {createSmoothTerrain} from '../terrain-illumination/terrain-horizon-oracle';

// Opt in with `VITE_RELIEF_VISUALIZATION_BENCHMARK=true npx vitest run --project headless <this file>`.
const RUN_BENCHMARK =
  (import.meta as unknown as {env?: Record<string, string | undefined>}).env?.[
    'VITE_RELIEF_VISUALIZATION_BENCHMARK'
  ] === 'true';

const SIZE = 1024;
const PIXEL_COUNT = SIZE * SIZE;
const SAMPLE_COUNT = 5;

async function waitForQueue(device: Device): Promise<void> {
  await (device as unknown as {handle: GPUDevice}).handle.queue.onSubmittedWorkDone();
}

async function measureCase(
  device: Device,
  label: string,
  build: (graph: GPUCommandGraph, buffers: Buffer[]) => void
): Promise<string> {
  const buffers: Buffer[] = [];
  const graph = new GPUCommandGraph(device, {id: `bench-${label}`});
  build(graph, buffers);
  const compileStart = performance.now();
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  await waitForQueue(device);
  const firstMilliseconds = performance.now() - compileStart;
  const samples: number[] = [];
  for (let sample = 0; sample < SAMPLE_COUNT; sample++) {
    const start = performance.now();
    submitGraph(device, compiled, undefined);
    await waitForQueue(device);
    samples.push(performance.now() - start);
  }
  samples.sort((left, right) => left - right);
  compiled.destroy();
  for (const buffer of buffers) buffer.destroy();
  return `${label}: median ${samples[Math.floor(samples.length / 2)].toFixed(1)} ms, first ${firstMilliseconds.toFixed(0)} ms`;
}

it.skipIf(!RUN_BENCHMARK)('relief visualization 1024^2 timings', {timeout: 600000}, async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const elevation = createSmoothTerrain(SIZE, SIZE, 1).map(value => value * 4);
  const settingsBuffers: GPUParameterBuffer<'float32'>[] = [];
  const createSettings = (values: Float32Array) => {
    const buffer = new GPUParameterBuffer(device, {
      id: `settings-${settingsBuffers.length}`,
      format: 'float32',
      length: values.length,
      values
    });
    settingsBuffers.push(buffer);
    return buffer;
  };
  const band = (graph: GPUCommandGraph, buffers: Buffer[]) => {
    const buffer = createInputBuffer(device, elevation);
    buffers.push(buffer);
    return {
      id: 'elevation',
      format: 'float32' as const,
      storage: {
        kind: 'buffer' as const,
        values: importGraphBuffer(graph, 'elevation', buffer, 'float32', PIXEL_COUNT)
      }
    };
  };
  const output = (graph: GPUCommandGraph, buffers: Buffer[], name: string) => {
    const buffer = createOutputBuffer(device, PIXEL_COUNT);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, 'float32', PIXEL_COUNT);
  };
  const slrmSettings = createSettings(getGPUSimpleLocalReliefParameterValues());
  const msrmSettings = createSettings(getGPUMultiScaleReliefParameterValues());
  const dominanceSettings = createSettings(getGPULocalDominanceParameterValues());
  const blendSettings = createSettings(
    getGPUReliefBlendParameterValues(GPU_RELIEF_BLEND_VAT_ARCHAEOLOGICAL)
  );
  const lines = [
    await measureCase(device, 'SLRM radius 20', (graph, buffers) => {
      graph.add(
        new GPUSimpleLocalRelief({
          width: SIZE,
          height: SIZE,
          radius: 20,
          elevation: band(graph, buffers),
          settings: slrmSettings.importToGraph(graph),
          relief: output(graph, buffers, 'relief')
        })
      );
    }),
    await measureCase(device, 'MSRM res 1, 1..20 m, s 1 (radii 0..10)', (graph, buffers) => {
      graph.add(
        new GPUMultiScaleRelief({
          width: SIZE,
          height: SIZE,
          resolution: 1,
          featureMinimum: 1,
          featureMaximum: 20,
          scalingFactor: 1,
          elevation: band(graph, buffers),
          settings: msrmSettings.importToGraph(graph),
          relief: output(graph, buffers, 'relief')
        })
      );
    }),
    await measureCase(device, 'MSRM res 1, 3..200 m, s 2 (radii 1..100)', (graph, buffers) => {
      graph.add(
        new GPUMultiScaleRelief({
          width: SIZE,
          height: SIZE,
          resolution: 1,
          featureMinimum: 3,
          featureMaximum: 200,
          scalingFactor: 2,
          elevation: band(graph, buffers),
          settings: msrmSettings.importToGraph(graph),
          relief: output(graph, buffers, 'relief')
        })
      );
    }),
    await measureCase(device, 'local dominance defaults (264 samples)', (graph, buffers) => {
      graph.add(
        new GPULocalDominance({
          width: SIZE,
          height: SIZE,
          elevation: band(graph, buffers),
          settings: dominanceSettings.importToGraph(graph),
          dominance: output(graph, buffers, 'dominance')
        })
      );
    }),
    await measureCase(device, 'blend, 4 VAT layers + color', (graph, buffers) => {
      const colorBuffer = createOutputBuffer(device, PIXEL_COUNT);
      buffers.push(colorBuffer);
      graph.add(
        new GPUReliefBlend({
          width: SIZE,
          height: SIZE,
          layers: [0, 1, 2, 3].map(layer => {
            const buffer = createInputBuffer(device, elevation);
            buffers.push(buffer);
            return importGraphBuffer(graph, `layer-${layer}`, buffer, 'float32', PIXEL_COUNT);
          }),
          settings: blendSettings.importToGraph(graph),
          blend: output(graph, buffers, 'blend'),
          color: importGraphBuffer(graph, 'color', colorBuffer, 'uint32', PIXEL_COUNT)
        })
      );
    })
  ];
  console.log(`relief visualization ${SIZE}^2\n${lines.join('\n')}`);
  for (const buffer of settingsBuffers) buffer.destroy();
});
