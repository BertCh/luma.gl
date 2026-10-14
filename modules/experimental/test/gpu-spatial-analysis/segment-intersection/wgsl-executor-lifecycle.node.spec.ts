// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUBVH, GPUCompaction, GPUScan, GPUSort} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, it, vi} from 'vitest';
import {TypedData, WgslExec} from 'wgsl_reflect';
import * as boundsReduction from '../../../src/gpu-spatial-analysis/hilbert-keys/bounds-reduction';
import {useCPUExecutorTestHooks} from './cpu-executor-test-hooks';
import {createCPUGraphMemory, executeCommandNodesOnCPU} from './wgsl-graph-executor';

const primitives = [GPUSort, GPUScan, GPUCompaction, GPUBVH];
const originalMethods = primitives.map(primitive => primitive.prototype.getCommandNodes);
const originalBounds = boundsReduction.getBoundsReductionNodes;

describe('CPU primitive scope', () => {
  useCPUExecutorTestHooks();
  it('patches the actual cached constructors during the case', () => {
    primitives.forEach((primitive, index) =>
      expect(primitive.prototype.getCommandNodes).not.toBe(originalMethods[index])
    );
    expect(boundsReduction.getBoundsReductionNodes).not.toBe(originalBounds);
  });
});

it('restores the real primitives before the next suite', () => {
  primitives.forEach((primitive, index) =>
    expect(primitive.prototype.getCommandNodes).toBe(originalMethods[index])
  );
  expect(boundsReduction.getBoundsReductionNodes).toBe(originalBounds);
});

for (const fails of [false, true]) {
  it(`restores interpreter methods and destroys compiled nodes after ${fails ? 'failure' : 'success'}`, () => {
    const interpreter = Object.getOwnPropertyDescriptors(WgslExec.prototype);
    const typedData = Object.getOwnPropertyDescriptors(TypedData.prototype);
    const destroy = vi.fn();
    const execute = () =>
      executeCommandNodesOnCPU(
        [
          {
            id: 'lifecycle',
            compile: () => ({
              encode: () => {
                expect(Object.getOwnPropertyDescriptor(WgslExec.prototype, '_call')).not.toEqual(
                  interpreter._call
                );
                if (fails) throw new Error('intentional encode failure');
              },
              destroy
            })
          }
        ],
        createCPUGraphMemory()
      );
    if (fails) expect(execute).toThrow('intentional encode failure');
    else execute();
    expect(destroy).toHaveBeenCalledOnce();
    expect(Object.getOwnPropertyDescriptors(WgslExec.prototype)).toEqual(interpreter);
    expect(Object.getOwnPropertyDescriptors(TypedData.prototype)).toEqual(typedData);
  });
}
