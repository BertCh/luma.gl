// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/gpu-raster/raster-grid/index';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUIsobands} from '../../../src/gpu-raster/isolines/gpu-isobands';
import {getGPUIsobandsParameterValues} from '../../../src/gpu-raster/isolines/isobands-parameters';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// Same-run A/B of the isoband triangle and boundary-edge paths (review C §5: isBorderShared cost).
// Opt in with `VITE_ISOBANDS_BENCHMARK=true npx vitest run --project headless <this file>`.
const RUN_BENCHMARK =
  (import.meta as unknown as {env?: Record<string, string | undefined>}).env?.[
    'VITE_ISOBANDS_BENCHMARK'
  ] === 'true';

const SIZE = 4096;
const SAMPLE_COUNT = 5;
const BREAKS = [55, 70, 85, 100, 115, 130, 145, 160];
/** Triangle rows at the 128 MiB binding limit (3 float32x2 vertices = 24 bytes per triangle). */
const TRIANGLE_CAPACITY = Math.floor(134217728 / 24);
/** Edge rows at the 128 MiB binding limit (one float32x4 = 16 bytes per edge). */
const EDGE_CAPACITY = Math.floor(134217728 / 16);

type Path = 'classes' | 'triangles' | 'edges';

async function waitForQueue(device: Device): Promise<void> {
  await (device as unknown as {handle: GPUDevice}).handle.queue.onSubmittedWorkDone();
}

async function measurePath(device: Device, values: Float32Array, path: Path): Promise<string> {
  const sampleCount = SIZE * SIZE;
  const grid = makeGPURasterMetadata({width: SIZE, height: SIZE});
  const owned: Buffer[] = [];
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    owned.push(buffer);
    return buffer;
  };
  const valuesBuffer = createInputBuffer(device, values);
  const breaksBuffer = createInputBuffer(device, Float32Array.from(BREAKS));
  owned.push(valuesBuffer, breaksBuffer);
  const parameters = new GPUParameterBuffer(device, {
    id: 'isoband-settings',
    format: 'float32',
    length: 12,
    values: getGPUIsobandsParameterValues(
      {breakCount: BREAKS.length, extent: [0, 0, SIZE, SIZE]},
      grid
    )
  });
  const graph = new GPUCommandGraph(device, {id: `isobands-bench-${path}`});
  const view = <Format extends 'uint32' | 'float32x2' | 'float32x4'>(
    name: string,
    buffer: Buffer,
    format: Format,
    length: number
  ) => importGraphBuffer(graph, name, buffer, format, length);
  const count = output(1);
  graph.add(
    new GPUIsobands({
      id: 'isobands',
      grid,
      values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', sampleCount),
      breaks: importGraphBuffer(graph, 'breaks', breaksBuffer, 'float32', BREAKS.length),
      parameters: parameters.importToGraph(graph),
      output:
        path === 'classes'
          ? {bandClasses: view('classes', output(sampleCount), 'uint32', sampleCount)}
          : path === 'triangles'
            ? {
                triangles: view(
                  'triangles',
                  output(TRIANGLE_CAPACITY * 6),
                  'float32x2',
                  TRIANGLE_CAPACITY * 3
                ),
                triangleBands: view(
                  'bands',
                  output(TRIANGLE_CAPACITY),
                  'uint32',
                  TRIANGLE_CAPACITY
                ),
                status: {
                  count: view('count', count, 'uint32', 1),
                  overflow: view('overflow', output(1), 'uint32', 1)
                }
              }
            : {
                edges: view('edges', output(EDGE_CAPACITY * 4), 'float32x4', EDGE_CAPACITY),
                edgeBands: view('edge-bands', output(EDGE_CAPACITY), 'uint32', EDGE_CAPACITY),
                status: {
                  edgeCount: view('edge-count', count, 'uint32', 1),
                  edgeOverflow: view('edge-overflow', output(1), 'uint32', 1)
                }
              }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  await waitForQueue(device);
  const samples: number[] = [];
  for (let sample = 0; sample < SAMPLE_COUNT; sample++) {
    const start = performance.now();
    submitGraph(device, compiled, undefined);
    await waitForQueue(device);
    samples.push(performance.now() - start);
  }
  samples.sort((left, right) => left - right);
  const [rows] = path === 'classes' ? [0] : await readUint32(count, 1);
  compiled.destroy();
  parameters.destroy();
  for (const buffer of owned) buffer.destroy();
  return `${path}: median ${samples[Math.floor(samples.length / 2)].toFixed(1)} ms, ${rows} rows`;
}

it.skipIf(!RUN_BENCHMARK)('isobands 4096^2 triangles vs edges', {timeout: 600000}, async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Low-frequency waves: every band boundary is a long smooth contour, and the outputs fit.
  const values = Float32Array.from({length: SIZE * SIZE}, (_, index) => {
    const x = index % SIZE;
    const y = Math.floor(index / SIZE);
    return 100 + 40 * Math.sin(x * 0.011 + 0.3) + 30 * Math.sin(y * 0.007 + x * 0.003);
  });
  const paths: Path[] = ['classes', 'triangles', 'edges'];
  const lines: string[] = [];
  for (const path of [...paths, ...[...paths].reverse()]) {
    lines.push(await measurePath(device, values, path));
  }
  console.log(`isobands ${SIZE}^2, ${BREAKS.length} breaks\n${lines.join('\n')}`);
});
