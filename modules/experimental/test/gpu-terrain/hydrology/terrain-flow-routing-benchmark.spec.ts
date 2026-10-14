// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {makeGPURasterBand, makeGPURasterMetadata} from '../../../src/gpu-raster';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUTerrainFlow,
  getGPUTerrainFlowParameterValues,
  type GPUTerrainFlowRouting
} from '../../../src/gpu-terrain/hydrology';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createSmoothTerrain} from '../terrain-illumination/terrain-horizon-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

// Same-run A/B of the accumulation routings (review C §5: D-infinity/MFD cost against D8).
// Opt in with `VITE_TERRAIN_FLOW_ROUTING_BENCHMARK=true npx vitest run --project headless <this file>`.
const RUN_BENCHMARK =
  (import.meta as unknown as {env?: Record<string, string | undefined>}).env?.[
    'VITE_TERRAIN_FLOW_ROUTING_BENCHMARK'
  ] === 'true';

const SIZE = 2048;
const CELL_COUNT = SIZE * SIZE;
const SAMPLE_COUNT = 5;
const ROUTINGS: GPUTerrainFlowRouting[] = ['d8', 'd-infinity', 'mfd-freeman', 'mfd-quinn'];

async function waitForQueue(device: Device): Promise<void> {
  await (device as unknown as {handle: GPUDevice}).handle.queue.onSubmittedWorkDone();
}

async function measureRouting(
  device: Device,
  elevation: Float32Array,
  routing: GPUTerrainFlowRouting,
  conditioned: boolean
): Promise<string> {
  const grid = makeGPURasterMetadata({width: SIZE, height: SIZE, cellSize: 10});
  const elevationBuffer = createInputBuffer(device, elevation);
  const accumulation = createOutputBuffer(device, CELL_COUNT);
  const converged = createOutputBuffer(device, 1);
  const rounds = createOutputBuffer(device, 1);
  const settings = new GPUParameterBuffer(device, {
    id: 'flow-settings',
    format: 'float32',
    length: 8,
    values: getGPUTerrainFlowParameterValues({}, grid)
  });
  const graph = new GPUCommandGraph(device, {id: `flow-bench-${routing}`});
  graph.add(
    new GPUTerrainFlow({
      grid,
      elevation: makeGPURasterBand(
        importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', CELL_COUNT),
        grid,
        {id: 'elevation'}
      ),
      parameters: settings.importToGraph(graph),
      fillDepressions: conditioned,
      resolveFlats: conditioned,
      flowRouting: routing,
      maximumAccumulationIterations: 1024,
      output: {
        accumulation: importGraphBuffer(graph, 'accumulation', accumulation, 'float32', CELL_COUNT),
        status: {
          accumulationConverged: importGraphBuffer(graph, 'converged', converged, 'uint32', 1),
          accumulationIterationCount: importGraphBuffer(graph, 'rounds', rounds, 'uint32', 1)
        }
      }
    })
  );
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
  const [isConverged] = await readUint32(converged, 1);
  const [roundCount] = await readUint32(rounds, 1);
  compiled.destroy();
  settings.destroy();
  elevationBuffer.destroy();
  accumulation.destroy();
  converged.destroy();
  rounds.destroy();
  const median = samples[Math.floor(samples.length / 2)];
  return `${routing}${conditioned ? ' (fill + flats)' : ''}: median ${median.toFixed(1)} ms, first ${firstMilliseconds.toFixed(0)} ms, converged ${isConverged}, ${roundCount} accumulation rounds`;
}

it.skipIf(!RUN_BENCHMARK)('terrain flow routing 2048^2 timings', {timeout: 600000}, async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const elevation = createSmoothTerrain(SIZE, SIZE, 1).map(value => value * 4);
  const lines: string[] = [];
  for (const conditioned of [true, false]) {
    for (const routing of ROUTINGS) {
      lines.push(await measureRouting(device, elevation, routing, conditioned));
    }
    // Second pass in reverse order, so warm-up and thermal drift do not favor one routing.
    for (const routing of [...ROUTINGS].reverse()) {
      lines.push(await measureRouting(device, elevation, routing, conditioned));
    }
  }
  console.log(`terrain flow routing ${SIZE}^2\n${lines.join('\n')}`);
});
