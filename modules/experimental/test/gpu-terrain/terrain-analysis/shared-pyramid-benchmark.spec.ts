// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  createSharedPyramidScene,
  type SharedPyramidConsumer,
  type SharedPyramidMode
} from './shared-pyramid-fixture';

// Opt in with `VITE_SHARED_PYRAMID_BENCHMARK=true npx vitest run --project headless <this file>`.
const RUN_BENCHMARK =
  (import.meta as unknown as {env?: Record<string, string | undefined>}).env?.[
    'VITE_SHARED_PYRAMID_BENCHMARK'
  ] === 'true';

const SAMPLE_COUNT = 15;
const MODES: SharedPyramidMode[] = ['march', 'self', 'shared', 'prebuilt'];

async function waitForQueue(device: Device): Promise<void> {
  await (device as unknown as {handle: GPUDevice}).handle.queue.onSubmittedWorkDone();
}

async function measureMode(
  device: Device,
  size: number,
  mode: SharedPyramidMode,
  only?: readonly SharedPyramidConsumer[]
): Promise<string> {
  const scene = createSharedPyramidScene(device, {
    width: size,
    height: size,
    mode,
    only,
    pairCount: 4096,
    observerCount: 16,
    azimuthCount: 720,
    cumulativeObserverCount: size > 1024 ? 1 : 2,
    terrainAmplitude: 40,
    observerHeight: 25,
    curvatureCoefficient: 6.8e-8
  });
  // Warm-up: pipeline compilation.
  scene.run();
  await waitForQueue(device);
  const wall: number[] = [];
  const gpu: number[] = [];
  let nodeCount = 0;
  for (let sample = 0; sample < SAMPLE_COUNT; sample++) {
    const start = performance.now();
    const encoding = scene.run();
    await waitForQueue(device);
    wall.push(performance.now() - start);
    nodeCount = encoding.stats.nodeCount;
    if (encoding.canReadGPUTimings) {
      const report = await encoding.readTimings();
      gpu.push(
        report.nodes.reduce(
          (sum, node) => sum + ((node as {gpuTimeMilliseconds?: number}).gpuTimeMilliseconds ?? 0),
          0
        )
      );
    }
  }
  const outputs = await scene.read();
  const hidden = outputs.viewshed.filter(code => code === 0).length;
  const visible = outputs.viewshed.filter(code => code === 1).length;
  scene.destroy();
  const median = (values: number[]) => [...values].sort((a, b) => a - b)[values.length >> 1];
  return `${size}^2 ${(only?.[0] ?? 'all').padEnd(17)} ${mode.padEnd(8)} nodes ${String(nodeCount).padStart(3)} (hidden ${hidden}, visible ${visible})  wall median ${median(wall).toFixed(1)} ms${
    gpu.length ? `  GPU timestamps median ${median(gpu).toFixed(1)} ms` : ''
  }`;
}

it.skipIf(!RUN_BENCHMARK)(
  'shared pyramid benchmark: march vs self-built vs shared vs prebuilt',
  async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const consumers: (readonly SharedPyramidConsumer[] | undefined)[] = [
      undefined,
      ['viewshed'],
      ['lineOfSight'],
      ['cumulative'],
      ['profile'],
      ['horizonVisibility']
    ];
    for (const size of [512, 2048]) {
      for (const only of consumers) {
        for (const mode of MODES) {
          console.log(await measureMode(device, size, mode, only));
        }
      }
    }
    expect(true).toBe(true);
  },
  600_000
);
