// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUDelaunayTessellation} from '../../../../src/gpu-spatial-analysis/delaunay-tessellation/index';
import {importGraphBuffer} from '../../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../../utils/require-test-device';
import {createUniformFixture, toTriangleRows} from '../delaunay-reference';
import {LegacyGPUDelaunayTessellation} from './legacy-delaunay-tessellation';

// Same-run A/B of the legacy per-site Bowyer-Watson against parallel insertion and flipping.
// Run with A5_AB=1. Compile time is excluded; each side reports the best of three submits, each
// timed to the readback of the published count. 1e5 runs the new path only (legacy would need
// 4e5 graph nodes). Prints one table line per size.

const ENABLED = process.env['A5_AB'] === '1';

type Implementation = 'legacy' | 'parallel';

async function prepare(device: Device, implementation: Implementation, coordinates: Float32Array) {
  const count = coordinates.length / 2;
  const capacity = 2 * count;
  const graph = new GPUCommandGraph(device, {id: `delaunay-ab-${implementation}-${count}`});
  const buffers: Buffer[] = [];
  const positionsBuffer = createInputBuffer(device, coordinates);
  const triangleBuffer = createOutputBuffer(device, 3 * capacity);
  buffers.push(positionsBuffer, triangleBuffer);
  const scalar = (name: string) => {
    const buffer = createOutputBuffer(device, 1);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', 1)};
  };
  const written = scalar('count');
  const requiredCount = scalar('required-count');
  const overflow = scalar('overflow');
  const invalidCount = scalar('invalid-count');
  const duplicateCount = scalar('duplicate-count');
  const props = {
    positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', count),
    output: {
      triangles: importGraphBuffer(graph, 'triangles', triangleBuffer, 'uint32x3', capacity),
      status: {
        count: written.view,
        requiredCount: requiredCount.view,
        overflow: overflow.view,
        invalidCount: invalidCount.view,
        duplicateCount: duplicateCount.view
      }
    }
  };
  graph.add(
    implementation === 'legacy'
      ? new LegacyGPUDelaunayTessellation(props)
      : new GPUDelaunayTessellation(props)
  );
  const compileStart = performance.now();
  const compiled = graph.compile();
  const compileMilliseconds = performance.now() - compileStart;
  return {
    compileMilliseconds,
    async submitOnce() {
      const start = performance.now();
      submitGraph(device, compiled, undefined);
      const rows = (await readUint32(written.buffer, 1))[0];
      return {milliseconds: performance.now() - start, rows};
    },
    async readTriangles() {
      const rows = (await readUint32(written.buffer, 1))[0];
      const invalid = (await readUint32(invalidCount.buffer, 1))[0];
      return {rows, invalid, indices: await readUint32(triangleBuffer, 3 * rows)};
    },
    destroy() {
      compiled.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

async function measure(device: Device, implementation: Implementation, coordinates: Float32Array) {
  const run = await prepare(device, implementation, coordinates);
  // Warm-up submit (pipeline creation, buffer allocation) is not timed.
  await run.submitOnce();
  let best = Number.POSITIVE_INFINITY;
  for (let attempt = 0; attempt < 3; attempt++) {
    best = Math.min(best, (await run.submitOnce()).milliseconds);
  }
  const output = await run.readTriangles();
  const summary = {
    best,
    compileMilliseconds: run.compileMilliseconds,
    output
  };
  run.destroy();
  return summary;
}

const canonical = (rows: number[][]) =>
  rows
    .map(row => [...row].sort((a, b) => a - b).join(','))
    .sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));

for (const count of [1000, 10000, 100000]) {
  it.skipIf(!ENABLED)(
    `Delaunay A/B at ${count} uniform sites`,
    async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const coordinates = createUniformFixture(20261016, count, 100, 100);
      const parallel = await measure(device, 'parallel', coordinates);
      expect(parallel.output.invalid).toBe(0);
      let legacyText = 'legacy infeasible (4 nodes per site, each O(n))';
      let ratioText = 'n/a';
      if (count <= 10000) {
        const legacy = await measure(device, 'legacy', coordinates);
        expect(legacy.output.invalid).toBe(0);
        // Same triangle set (general position), though the order differs: legacy is not canonical.
        expect(canonical(toTriangleRows(parallel.output.indices))).toEqual(
          canonical(toTriangleRows(legacy.output.indices))
        );
        legacyText = `legacy ${legacy.best.toFixed(1)} ms`;
        ratioText = `${(legacy.best / parallel.best).toFixed(1)}x`;
      }
      // eslint-disable-next-line no-console
      console.log(
        `delaunay-ab n=${count} ${legacyText} | parallel ${parallel.best.toFixed(1)} ms | speedup ${ratioText}`
      );
    },
    1_800_000
  );
}
