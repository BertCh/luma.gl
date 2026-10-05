// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUMapGraphParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {
  getGPUReliefShadingParameterValues,
  GPUReliefShading
} from '../../../src/map-graphs/terrain-illumination/gpu-relief-shading';
import {
  getGPUSolarPositionParameterValues,
  GPUSolarPosition
} from '../../../src/map-graphs/terrain-illumination/gpu-solar-position';
import {
  getGPUSolarShadowMaskParameterValues,
  GPUSolarShadowMask
} from '../../../src/map-graphs/terrain-illumination/gpu-solar-shadow-mask';
import {
  getGPUTerrainHorizonParameterValues,
  GPUTerrainHorizon
} from '../../../src/map-graphs/terrain-illumination/gpu-terrain-horizon';
import {
  getGPUTextureShadingParameterValues,
  GPUTextureShading
} from '../../../src/map-graphs/terrain-illumination/gpu-texture-shading';
import {createInputBuffer, createOutputBuffer} from '../map-graph-test-utils';
import {createSmoothTerrain} from './terrain-horizon-oracle';

// Opt in with `VITE_TERRAIN_ILLUMINATION_BENCHMARK=true npx vitest run --project headless <this file>`.
const RUN_BENCHMARK =
  (import.meta as unknown as {env?: Record<string, string | undefined>}).env?.[
    'VITE_TERRAIN_ILLUMINATION_BENCHMARK'
  ] === 'true';

const SIZE = 2048;
const PIXEL_COUNT = SIZE * SIZE;
const SAMPLE_COUNT = 5;

type BenchmarkCase = {
  label: string;
  build: (graph: GPUCommandGraph, buffers: Buffer[]) => void;
};

async function waitForQueue(device: Device): Promise<void> {
  await (device as unknown as {handle: GPUDevice}).handle.queue.onSubmittedWorkDone();
}

async function measureCase(device: Device, benchmarkCase: BenchmarkCase): Promise<string> {
  const buffers: Buffer[] = [];
  const graph = new GPUCommandGraph(device, {
    id: `bench-${benchmarkCase.label}`
  });
  benchmarkCase.build(graph, buffers);
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
  const median = samples[Math.floor(samples.length / 2)];
  compiled.destroy();
  for (const buffer of buffers) buffer.destroy();
  return `${benchmarkCase.label}: median ${median.toFixed(1)} ms (${(PIXEL_COUNT / median / 1000).toFixed(1)} Mpx/s), first ${firstMilliseconds.toFixed(0)} ms`;
}

it.skipIf(!RUN_BENCHMARK)('terrain illumination 2048^2 throughput', {timeout: 600000}, async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const elevation = createSmoothTerrain(SIZE, SIZE, 1).map(value => value * 4);
  const parameters: GPUMapGraphParameterBuffer<'float32'>[] = [];
  const createParameters = (values: Float32Array) => {
    const parameterBuffer = new GPUMapGraphParameterBuffer(device, {
      id: `parameters-${parameters.length}`,
      format: 'float32',
      length: values.length,
      values
    });
    parameters.push(parameterBuffer);
    return parameterBuffer;
  };
  const horizonSettings = createParameters(
    getGPUTerrainHorizonParameterValues({cellSize: [30, 30]})
  );
  const shadowSettings = createParameters(
    getGPUSolarShadowMaskParameterValues({
      azimuthDegrees: 250,
      altitudeDegrees: 12
    })
  );
  const reliefSettings = createParameters(getGPUReliefShadingParameterValues({cellSize: [30, 30]}));
  const textureSettings = createParameters(getGPUTextureShadingParameterValues({detail: 0.5}));
  const solarSettings = createParameters(
    getGPUSolarPositionParameterValues({
      timestamp: Date.UTC(2026, 9, 4, 12)
    })
  );
  const elevationBand = (graph: GPUCommandGraph, buffers: Buffer[]) => {
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
  const output = (
    graph: GPUCommandGraph,
    buffers: Buffer[],
    id: string,
    length = PIXEL_COUNT,
    format: 'float32' | 'uint32' = 'float32'
  ) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return importGraphBuffer(graph, id, buffer, format, length) as never;
  };
  const maxBinding = device.limits.maxStorageBufferBindingSize;
  const horizonCases: BenchmarkCase[] = (
    [
      [16, 64, 1],
      [16, 256, 1.1],
      [32, 256, 1.1]
    ] as const
  ).flatMap(([directionCount, maximumRadius, stepGrowth]) => [
    {
      label: `GPUTerrainHorizon sky-view D=${directionCount} R=${maximumRadius} growth=${stepGrowth}`,
      build: (graph: GPUCommandGraph, buffers: Buffer[]) =>
        graph.add(
          new GPUTerrainHorizon({
            width: SIZE,
            height: SIZE,
            elevation: elevationBand(graph, buffers),
            settings: horizonSettings.importToGraph(graph),
            directionCount,
            maximumRadius,
            stepGrowth,
            skyViewFactor: output(graph, buffers, 'svf')
          })
        )
    }
  ]);
  const horizonBytes = PIXEL_COUNT * 16 * 4;
  const cases: BenchmarkCase[] = [
    ...horizonCases,
    ...(horizonBytes <= maxBinding
      ? [
          {
            label: 'GPUTerrainHorizon full horizon map D=16 R=256 growth=1.1',
            build: (graph: GPUCommandGraph, buffers: Buffer[]) =>
              graph.add(
                new GPUTerrainHorizon({
                  width: SIZE,
                  height: SIZE,
                  elevation: elevationBand(graph, buffers),
                  settings: horizonSettings.importToGraph(graph),
                  directionCount: 16,
                  maximumRadius: 256,
                  stepGrowth: 1.1,
                  horizon: output(graph, buffers, 'horizon', PIXEL_COUNT * 16)
                })
              )
          },
          {
            label: 'GPUSolarShadowMask per frame D=16 (sun visibility)',
            build: (graph: GPUCommandGraph, buffers: Buffer[]) =>
              graph.add(
                new GPUSolarShadowMask({
                  width: SIZE,
                  height: SIZE,
                  directionCount: 16,
                  horizon: output(graph, buffers, 'horizon', PIXEL_COUNT * 16),
                  settings: shadowSettings.importToGraph(graph),
                  sunVisibility: output(graph, buffers, 'visibility')
                })
              )
          }
        ]
      : []),
    {
      label: 'GPUReliefShading MDOW hillshade + Imhof color',
      build: (graph, buffers) =>
        graph.add(
          new GPUReliefShading({
            width: SIZE,
            height: SIZE,
            elevation: elevationBand(graph, buffers),
            settings: reliefSettings.importToGraph(graph),
            relief: output(graph, buffers, 'relief'),
            color: output(graph, buffers, 'color', PIXEL_COUNT, 'uint32')
          })
        )
    },
    ...[4, 6].map(levelCount => ({
      label: `GPUTextureShading L=${levelCount}`,
      build: (graph: GPUCommandGraph, buffers: Buffer[]) =>
        graph.add(
          new GPUTextureShading({
            width: SIZE,
            height: SIZE,
            elevation: elevationBand(graph, buffers),
            settings: textureSettings.importToGraph(graph),
            levelCount,
            textureShade: output(graph, buffers, 'texture-shade')
          })
        )
    })),
    {
      label: 'GPUSolarPosition 4M rows',
      build: (graph, buffers) => {
        const positions = createInputBuffer(
          device,
          Float32Array.from({length: PIXEL_COUNT * 2}, (_, index) =>
            index % 2 === 0 ? -180 + ((index / 2) % 360) : -60 + ((index / 720) % 120)
          )
        );
        buffers.push(positions);
        graph.add(
          new GPUSolarPosition({
            positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', PIXEL_COUNT),
            settings: solarSettings.importToGraph(graph),
            altitude: output(graph, buffers, 'altitude')
          })
        );
      }
    }
  ];
  const lines = [`maxStorageBufferBindingSize ${maxBinding}`];
  for (const benchmarkCase of cases) {
    lines.push(await measureCase(device, benchmarkCase));
  }
  for (const parameterBuffer of parameters) parameterBuffer.destroy();
  // eslint-disable-next-line no-console
  console.log(`terrain illumination ${SIZE}^2\n${lines.join('\n')}`);
  expect(lines.length).toBeGreaterThan(1);
});
