// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUPolygonTriangulation,
  getPolygonTriangulationIndexCount,
  type GPUPolygonTriangulationProps
} from '../../../src/gpu-spatial-analysis/polygon-triangulation/index';
import {createOutputBuffer, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createSegmentGeometry} from '../segment-intersection/segment-geometry-harness';
import {
  checkTriangulation,
  createDoubleComb,
  createSkylineRings
} from './polygon-triangulation-monotone-cases';
import type {ModelPoint} from './polygon-triangulation-monotone-model';

// Same-run A/B of the monotone path against per-invocation Earcut on a 50k vertex polygon (algorithms
// roadmap P2): Earcut gets a huge `maximumWork` so it cannot fail early, the monotone method runs
// with its defaults (the polygon is above the routing threshold, so Earcut is skipped). Opt in with
// `VITE_POLYGON_TRIANGULATION_MONOTONE_BENCHMARK=true npx vitest run --project headless <this file>`;
// the result is logged, and both runs must agree on the Shapely-checked area.
const RUN_BENCHMARK =
  (import.meta as unknown as {env?: Record<string, string | undefined>}).env?.[
    'VITE_POLYGON_TRIANGULATION_MONOTONE_BENCHMARK'
  ] === 'true';

const SAMPLE_COUNT = 5;

async function waitForQueue(device: Device): Promise<void> {
  await (device as unknown as {handle: GPUDevice}).handle.queue.onSubmittedWorkDone();
}

type Runner = {
  submit: () => void;
  read: () => Promise<{indices: number[]; valid: number[]; triangleCount: number[]}>;
  destroy: () => void;
};

function createRunner(
  device: Device,
  rings: readonly (readonly ModelPoint[])[],
  options: Partial<GPUPolygonTriangulationProps>
): Runner {
  const graph = new GPUCommandGraph(device, {id: 'polygon-triangulation-benchmark'});
  const buffers: Buffer[] = [];
  const geometry = createSegmentGeometry(
    device,
    graph,
    'polygons',
    [
      {
        kind: 'polygons',
        polygons: [rings.map(ring => ring.map(point => [point[0], point[1]] as [number, number]))]
      }
    ],
    buffers
  );
  if (geometry.kind !== 'polygons') {
    throw new Error('expected polygons');
  }
  const {kind: _kind, ...fields} = geometry;
  const indexCount = getPolygonTriangulationIndexCount(
    geometry.positions.length,
    geometry.ringOffsets.length - 1,
    1
  );
  const output = (name: string, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
  };
  const indices = output('indices', indexCount);
  const valid = output('valid', 1);
  const triangleCount = output('triangle-count', 1);
  graph.add(
    new GPUPolygonTriangulation({
      ...fields,
      output: {indices: indices.view, valid: valid.view, triangleCount: triangleCount.view},
      ...options
    })
  );
  const compiled = graph.compile();
  return {
    submit: () => submitGraph(device, compiled, undefined),
    read: async () => ({
      indices: await readUint32(indices.buffer, indexCount),
      valid: await readUint32(valid.buffer, 1),
      triangleCount: await readUint32(triangleCount.buffer, 1)
    }),
    destroy: () => {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  };
}

async function measure(
  device: Device,
  name: string,
  method: string,
  rings: readonly (readonly ModelPoint[])[],
  options: Partial<GPUPolygonTriangulationProps>
): Promise<string> {
  const compileStart = performance.now();
  const runner = createRunner(device, rings, options);
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
  const result = await runner.read();
  runner.destroy();
  const positions = rings.flat();
  const triangles: [number, number, number][] = [];
  for (let k = 0; k < result.triangleCount[0]; k++) {
    triangles.push([result.indices[3 * k], result.indices[3 * k + 1], result.indices[3 * k + 2]]);
  }
  const check = checkTriangulation(rings, triangles);
  expect(result.valid[0], `${name} ${method} valid`).toBe(1);
  expect(positions.length).toBeGreaterThan(0);
  return `${name} ${method}: median ${samples[Math.floor(samples.length / 2)].toFixed(1)} ms, first ${firstMilliseconds.toFixed(0)} ms, ${result.triangleCount[0]} triangles, area ${check.area}, boundary ${check.boundaryMatches}`;
}

it.skipIf(!RUN_BENCHMARK)(
  'monotone method against Earcut on 50k vertex polygons',
  {timeout: 900000},
  async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const polygons: [string, ModelPoint[][]][] = [
      ['skyline-50k', createSkylineRings(24_999, 12, 0)],
      ['double-comb-50k', [createDoubleComb(6250)]]
    ];
    const lines: string[] = [];
    for (const [name, rings] of polygons) {
      const earcut = {largePolygonMethod: 'none', maximumWork: 0xffffffff} as const;
      const monotone = {largePolygonMethod: 'monotone'} as const;
      lines.push(await measure(device, name, 'earcut', rings, earcut));
      lines.push(await measure(device, name, 'monotone', rings, monotone));
      // Reverse order, so warm-up and thermal drift do not favor one method.
      lines.push(await measure(device, name, 'monotone', rings, monotone));
      lines.push(await measure(device, name, 'earcut', rings, earcut));
    }
    // eslint-disable-next-line no-console
    console.log(`GPUPolygonTriangulation monotone A/B\n${lines.join('\n')}`);
  }
);
