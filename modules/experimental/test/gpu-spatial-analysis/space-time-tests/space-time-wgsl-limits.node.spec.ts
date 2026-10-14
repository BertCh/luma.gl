// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUKnoxTest} from '../../../src/gpu-spatial-analysis/space-time-tests/gpu-knox-test';
import {GPUMantelTest} from '../../../src/gpu-spatial-analysis/space-time-tests/gpu-mantel-test';
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

const ROWS = 300;
const MAXIMUM_PERMUTATIONS = 99;
let serial = 0;

function getSources(kind: 'knox' | 'mantel'): {id: string; source: string}[] {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `space-time-${serial++}`, format, length);
  const pairs = {
    offsets: view('uint32', ROWS + 1),
    neighbors: view('uint32', 4 * ROWS),
    weights: view('float32', 4 * ROWS),
    distances: view('float32', 4 * ROWS)
  };
  const common = {
    pairs,
    times: view('float32', ROWS),
    parameters: view('uint32', GPU_SPACE_TIME_PARAMETER_LENGTH),
    maximumPermutations: MAXIMUM_PERMUTATIONS
  };
  const summary = view('float32', GPU_SPACE_TIME_SUMMARY_LENGTH);
  const test =
    kind === 'knox'
      ? new GPUKnoxTest({
          ...common,
          output: {statistics: view('uint32', MAXIMUM_PERMUTATIONS + 1), summary}
        })
      : new GPUMantelTest({
          ...common,
          output: {statistics: view('float32', MAXIMUM_PERMUTATIONS + 1), summary}
        });
  const nodes = test.getCommandNodes(graph);
  const sources = getGPUCommandNodeKernelSources(nodes);
  device.destroy();
  return sources;
}

for (const kind of ['knox', 'mantel'] as const) {
  it(`${kind} kernels fit the core binding, private and workgroup limits`, () => {
    const sources = getSources(kind);
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
  });

  it(`${kind} permutes through the swap-or-not shuffle with the barrier in uniform flow`, () => {
    const partials = getSources(kind).find(({id}) => id.endsWith('-partials'))!;
    // The Feistel helpers of the shared random module stay in the source, but are not called.
    expect(partials.source).not.toContain('getFeistelPermutationIndex(row');
    expect(partials.source).not.toContain('getFeistelRoundKeys(readSeedKey()');
    expect(partials.source).toContain('loadSwapOrNotRounds(');
    expect(partials.source).toContain('getSwapOrNotPermutationIndex(');
    // The workgroup holds the rounds of one permutation: 2 arrays of 150 words.
    expect(getWorkgroupMemoryBytes(partials.source)).toBe(2 * 150 * 4);
    // The barrier follows the conditional load at function scope, never inside a branch.
    const body = partials.source.slice(partials.source.indexOf('fn main('));
    expect(body).toMatch(/\n {2}workgroupBarrier\(\);/);
    const prologue = body.slice(
      body.indexOf('let workgroup = '),
      body.indexOf('workgroupBarrier()')
    );
    expect(prologue).not.toContain('return');
    // No kernel but the partials one (and the sort) pays for the workgroup memory.
    for (const {id, source} of getSources(kind)) {
      if (!id.endsWith('-partials') && !id.includes('-sort-')) {
        // Knox count-close keeps a workgroup (low word, carry count) pair of two atomics.
        expect(getWorkgroupMemoryBytes(source), id).toBe(id.endsWith('-count-close') ? 8 : 0);
      }
    }
  });
}

/** Returns the text of the first `if (<condition>) { ... }` block, matching braces. */
function getBlockAfter(source: string, condition: string): string {
  const start = source.indexOf(condition);
  expect(start, condition).toBeGreaterThanOrEqual(0);
  const open = source.indexOf('{', start);
  let depth = 0;
  for (let position = open; position < source.length; position++) {
    if (source[position] === '{') depth++;
    if (source[position] === '}' && --depth === 0) return source.slice(open, position + 1);
  }
  throw new Error('unbalanced block');
}

it('knox count-close issues its global atomics once per workgroup, from lane 0', () => {
  const kernel = getSources('knox').find(({id}) => id.endsWith('-count-close'))!;
  const body = kernel.source.slice(kernel.source.indexOf('fn main('));
  // Per-row atomics only touch workgroup counters.
  const globalAtomics = body.match(/atomicAdd\(&timeClose/g) ?? [];
  expect(globalAtomics).toHaveLength(2);
  const laneZero = getBlockAfter(body, 'if (localInvocationIndex == 0u)');
  expect(laneZero.match(/atomicAdd\(&timeClose/g)).toHaveLength(2);
  expect(body.match(/atomicAdd\(&group(Low|Carry)/g)).toHaveLength(2);
  // The barrier is at function scope, after the row work and before the global adds.
  expect(body).toMatch(/\n {2}workgroupBarrier\(\);\n {2}if \(localInvocationIndex == 0u\)/);
  expect(body.indexOf('workgroupBarrier()')).toBeLessThan(body.indexOf('atomicAdd(&timeClose'));
  // The row work is guarded by the row count, not an early return.
  expect(body).toContain('if (index < ROWS) {');
  // The generated prologue's only return is the workgroup-uniform dispatch overflow check.
  expect(
    body.slice(body.indexOf('let index = '), body.indexOf('workgroupBarrier()'))
  ).not.toContain('return');
  expect(getWorkgroupMemoryBytes(kernel.source)).toBe(2 * 4);
});
