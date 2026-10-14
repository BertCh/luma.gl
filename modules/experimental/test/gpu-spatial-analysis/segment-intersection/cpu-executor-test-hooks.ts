// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUBVH, GPUCompaction, GPUScan, GPUSort} from '@luma.gl/gpgpu/gpu-core';
import {afterEach, beforeEach, vi} from 'vitest';
import * as boundsReduction from '../../../src/gpu-spatial-analysis/hilbert-keys/bounds-reduction';
import {getBoundsReductionNodes} from './cpu-bounds-reduction-emulation';
import {CPUGPUBVH, CPUGPUCompaction, CPUGPUScan, CPUGPUSort} from './cpu-gpu-core-emulations';

/** Installs CPU primitives for each case, including contributors already loaded by another spec. */
export function useCPUExecutorTestHooks(): void {
  const restorers: (() => void)[] = [];
  beforeEach(() => {
    // Module mocks cannot replace references in cached contributors with isolate:false. Patch
    // the actual methods, then restore just these spies after the case, even when it throws.
    const replace = (original: {prototype: object}, replacement: {prototype: object}) => {
      type Primitive = {getCommandNodes: () => unknown[]};
      const prototype = original.prototype as Primitive;
      const emulate = (replacement.prototype as Primitive).getCommandNodes;
      const spy = vi.spyOn(prototype, 'getCommandNodes').mockImplementation(function (
        this: object
      ) {
        // gpu-core stores the validated props on the instance; the emulators accept that shape.
        return emulate.call(
          Object.assign(Object.create(replacement.prototype), this, {props: this})
        );
      });
      restorers.push(() => spy.mockRestore());
    };
    replace(GPUSort, CPUGPUSort);
    replace(GPUScan, CPUGPUScan);
    replace(GPUCompaction, CPUGPUCompaction);
    replace(GPUBVH, CPUGPUBVH);
    const bounds = vi
      .spyOn(boundsReduction, 'getBoundsReductionNodes')
      .mockImplementation(getBoundsReductionNodes);
    restorers.push(() => bounds.mockRestore());
  });
  afterEach(() => {
    for (const restore of restorers.splice(0).reverse()) restore();
  });
}
