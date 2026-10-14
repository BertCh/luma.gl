// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {createSeededRandom} from '../terrain-oracle-utils';
import {createFillRunner, type FillImplementation} from './terrain-flow-spill-fill-gpu';
import {makeSpiralCase} from './terrain-flow-spill-fill-oracle';

// Same-run A/B of the spill graph fill against the tiled relaxation (roadmap P3 #8). Opt in with
// `VITE_TERRAIN_FLOW_SPILL_FILL_BENCHMARK=true npx vitest run --project headless <this file>`.
const RUN_BENCHMARK =
  (import.meta as unknown as {env?: Record<string, string | undefined>}).env?.[
    'VITE_TERRAIN_FLOW_SPILL_FILL_BENCHMARK'
  ] === 'true';

const SAMPLE_COUNT = 5;

async function waitForQueue(device: Device): Promise<void> {
  await (device as unknown as {handle: GPUDevice}).handle.queue.onSubmittedWorkDone();
}

async function measureFill(
  device: Device,
  implementation: FillImplementation,
  name: string,
  elevation: Float32Array,
  size: number
): Promise<string> {
  const compileStart = performance.now();
  const runner = createFillRunner(device, implementation, elevation, size, size);
  runner.submit();
  await waitForQueue(device);
  const firstMilliseconds = performance.now() - compileStart;
  const samples: number[] = [];
  for (let sample = 0; sample < SAMPLE_COUNT; sample++) {
    const start = performance.now();
    runner.submit();
    await waitForQueue(device);
    samples.push(performance.now() - start);
  }
  samples.sort((left, right) => left - right);
  const result = await runner.run();
  runner.destroy();
  return `${name} ${implementation}: median ${samples[Math.floor(samples.length / 2)].toFixed(1)} ms, first ${firstMilliseconds.toFixed(0)} ms, converged ${result.converged}, ${result.iterations} iterations`;
}

it.skipIf(!RUN_BENCHMARK)(
  'depression fill 1024^2 spiral and 2048^2 pits timings',
  {timeout: 900000},
  async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const random = createSeededRandom(9);
    const pits2048 = Float32Array.from({length: 2048 * 2048}, () => Math.floor(random() * 400) / 4);
    const dems: [string, Float32Array, number][] = [
      ['spiral-1024-w1', makeSpiralCase(1024, 1).elevation, 1024],
      ['spiral-1024-w3', makeSpiralCase(1024, 3).elevation, 1024],
      ['pits-2048', pits2048, 2048]
    ];
    const lines: string[] = [];
    for (const [name, elevation, size] of dems) {
      for (const implementation of ['spill', 'relaxation'] as const) {
        lines.push(await measureFill(device, implementation, name, elevation, size));
      }
      // Reverse order, so warm-up and thermal drift do not favor one implementation.
      for (const implementation of ['relaxation', 'spill'] as const) {
        lines.push(await measureFill(device, implementation, name, elevation, size));
      }
    }
    console.log(`depression fill A/B\n${lines.join('\n')}`);
  }
);
