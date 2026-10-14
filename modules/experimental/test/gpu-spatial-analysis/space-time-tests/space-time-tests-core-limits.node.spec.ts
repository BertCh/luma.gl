// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, it} from 'vitest';
import {GPUKnoxTest, GPUMantelTest} from '../../../src/gpu-spatial-analysis/space-time-tests';
import {
  GPU_SPACE_TIME_PARAMETER_LENGTH,
  GPU_SPACE_TIME_SUMMARY_LENGTH
} from '../../../src/gpu-spatial-analysis/space-time-tests/space-time-parameters';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  CORE_PRIVATE_MEMORY_LIMIT_BYTES,
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES,
  getPrivateMemoryBytes,
  getStorageBindingCount,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';

/** Documented maximum of `maximumPermutations`. */
const MAXIMUM_PERMUTATIONS = 65536;
let serial = 0;

type Resource = {buffer: {byteLength?: number; length?: number; rowByteLength?: number}};

/** Bytes of every buffer or view bound by the nodes. */
function getBoundBytes(nodes: readonly unknown[]): {id: string; bytes: number}[] {
  const result: {id: string; bytes: number}[] = [];
  for (const node of nodes as {
    id: string;
    resources?: Resource[];
    workload?: {resources?: Resource[]};
  }[]) {
    for (const resource of node.resources ?? node.workload?.resources ?? []) {
      const buffer = resource.buffer;
      const bytes =
        buffer.byteLength ??
        (buffer.length ?? 0) * (buffer.rowByteLength ?? Uint32Array.BYTES_PER_ELEMENT);
      result.push({id: node.id, bytes});
    }
  }
  return result;
}

function expectCoreLimits(nodes: readonly unknown[], device = createNullWebGPUDevice()): void {
  const sources = getGPUCommandNodeKernelSources(nodes);
  expect(sources.length).toBeGreaterThan(0);
  for (const {id, source} of sources) {
    expect(getStorageBindingCount(source), `${id} storage bindings`).toBeLessThanOrEqual(8);
    expect(getPrivateMemoryBytes(source), `${id} private bytes`).toBeLessThanOrEqual(
      CORE_PRIVATE_MEMORY_LIMIT_BYTES
    );
    expect(getWorkgroupMemoryBytes(source), `${id} workgroup bytes`).toBeLessThanOrEqual(
      CORE_WORKGROUP_STORAGE_LIMIT_BYTES
    );
  }
  const bound = getBoundBytes(nodes);
  expect(bound.length).toBeGreaterThan(0);
  for (const {id, bytes} of bound) {
    expect(bytes, `${id} bound bytes`).toBeLessThanOrEqual(
      device.limits.maxStorageBufferBindingSize
    );
  }
}

function build(kind: 'knox' | 'mantel', rows: number, maximumPermutations: number, slots: number) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: `space-time-limits-${serial++}`});
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `space-time-limits-${serial++}`, format, length);
  const common = {
    id: kind,
    pairs: {
      offsets: view('uint32', rows + 1),
      neighbors: view('uint32', slots * rows),
      weights: view('float32', slots * rows),
      distances: view('float32', slots * rows)
    },
    times: view('float32', rows),
    parameters: view('uint32', GPU_SPACE_TIME_PARAMETER_LENGTH),
    maximumPermutations
  };
  const summary = view('float32', GPU_SPACE_TIME_SUMMARY_LENGTH);
  const contributor =
    kind === 'knox'
      ? new GPUKnoxTest({
          ...common,
          output: {statistics: view('uint32', maximumPermutations + 1), summary}
        })
      : new GPUMantelTest({
          ...common,
          output: {statistics: view('float32', maximumPermutations + 1), summary}
        });
  return {nodes: contributor.getCommandNodes(graph), device};
}

describe('space-time tests at core limits', () => {
  for (const kind of ['knox', 'mantel'] as const) {
    for (const [rows, permutations] of [
      [300, 99],
      [300, MAXIMUM_PERMUTATIONS],
      [100000, 999],
      [100000, MAXIMUM_PERMUTATIONS]
    ] as const) {
      // The Mantel partials table was 201,331,712 B (3 words x 256 blocks x 65537 slots) at the
      // documented maximum of 65536 permutations, above the 128 MiB binding limit; the block count
      // now shrinks with the permutation count so the table fits.
      it(`${kind} with ${rows} rows and ${permutations} permutations`, () => {
        const {nodes, device} = build(kind, rows, permutations, 4);
        expectCoreLimits(nodes, device);
      });
    }
  }
});
