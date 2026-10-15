// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, GPUHistogram, createTransientView} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../../experimental/test/utils/gpu-contributor-test-utils';

it('WP6 corrected histogram WGSL parses for local/global atomics and literal/GPU/auto domains', () => {
  for (const binCount of [9, 512]) {
    for (const domainMode of ['literal', 'gpu', 'auto'] as const) {
      const device = createNullWebGPUDevice();
      try {
        const graph = new GPUCommandGraph(device);
        const contributor = new GPUHistogram({
          input: createTransientView(graph, 'input', 'float32', 513),
          mask: createTransientView(graph, 'mask', 'uint32', 513),
          output: createTransientView(graph, 'output', 'uint32', binCount),
          domain:
            domainMode === 'literal'
              ? [-1, 8]
              : domainMode === 'auto'
                ? 'auto'
                : createTransientView(graph, 'domain', 'float32', 2)
        });
        const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
        expect(kernels.length).toBeGreaterThan(0);
        for (const kernel of kernels)
          expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
      } finally {
        device.destroy();
      }
    }
  }
});
