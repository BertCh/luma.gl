// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUTransitionDiagnostics,
  GPU_TRANSITION_HOMOGENEITY_LENGTH,
  GPU_TRANSITION_STATE_SPARSITY_LENGTH
} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-transition-diagnostics';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {
  CORE_PRIVATE_MEMORY_LIMIT_BYTES,
  getPrivateMemoryBytes,
  getStorageBindingCount
} from '../space-time-tests/wgsl-memory';

let serial = 0;

function getSources(classCount: number, conditionCount: number) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `diagnostics-${serial++}`, format, length);
  const chainCount = conditionCount > 1 ? conditionCount + 1 : 1;
  const diagnostics = new GPUTransitionDiagnostics({
    counts: view('uint32', conditionCount * classCount * classCount),
    classCount,
    conditionCount,
    output: {
      homogeneity:
        conditionCount > 1 ? view('float32', GPU_TRANSITION_HOMOGENEITY_LENGTH) : undefined,
      steadyStates: view('float32', chainCount * classCount),
      stateSparsity: view('uint32', GPU_TRANSITION_STATE_SPARSITY_LENGTH),
      status: {converged: view('uint32', 1)}
    }
  });
  const sources = getGPUCommandNodeKernelSources(diagnostics.getCommandNodes(graph));
  device.destroy();
  return sources;
}

it('the private-memory counter sees the 16.9 KB the old steady-state kernel needed', () => {
  // 64 * 64 f32 matrix + 64 u32 + 64 f32 (review finding E-10).
  const oldDeclarations = `var<private> work: array<f32, 4096>;
var<private> usedStates: array<u32, 64>;
var<private> solution: array<f32, 64>;`;
  expect(getPrivateMemoryBytes(oldDeclarations)).toBe(16896);
  expect(getPrivateMemoryBytes(oldDeclarations)).toBeGreaterThan(CORE_PRIVATE_MEMORY_LIMIT_BYTES);
});

for (const [classCount, conditionCount] of [
  [64, 1],
  [64, 4],
  [10, 3]
] as const) {
  it(`constructs at ${classCount} classes x ${conditionCount} conditions within core limits`, () => {
    const sources = getSources(classCount, conditionCount);
    expect(sources.some(({id}) => id.endsWith('-steady-states'))).toBe(true);
    for (const {id, source} of sources) {
      expect(getPrivateMemoryBytes(source), `${id} private bytes`).toBeLessThanOrEqual(
        CORE_PRIVATE_MEMORY_LIMIT_BYTES
      );
      expect(getStorageBindingCount(source), `${id} storage bindings`).toBeLessThanOrEqual(8);
    }
    // Only 2 * K words stay private in the solver (used-state list and solution).
    const solver = sources.find(({id}) => id.endsWith('-steady-states'))!;
    expect(getPrivateMemoryBytes(solver.source)).toBe(2 * classCount * 4);
  });
}
