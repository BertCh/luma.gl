// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {GPURipleyDistanceFunctions} from '../../../src/gpu-dataframe/pair-statistics/gpu-ripley-distance-functions';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {createRipleyTestWindow} from './ripley-envelope-fixture';

for (const polygon of [false, true]) {
  for (const allOutputs of [false, true]) {
    it(`WP6 survival kernels parse with polygon=${polygon}, allOutputs=${allOutputs}`, () => {
      const device = createNullWebGPUDevice();
      try {
        const graph = new GPUCommandGraph(device);
        const contributor = new GPURipleyDistanceFunctions({
          positions: createTransientView(graph, 'positions', 'float32x2', 256),
          parameters: createTransientView(graph, 'parameters', 'float32', 8),
          studyWindow: polygon ? createRipleyTestWindow(graph) : undefined,
          gridSize: [4, 4],
          referenceGrid: [24, 24],
          radiusCount: 30,
          output: {
            j: createTransientView(graph, 'j', 'float32', 30),
            ...(allOutputs
              ? {
                  g: createTransientView(graph, 'g', 'float32', 30),
                  f: createTransientView(graph, 'f', 'float32', 30),
                  radii: createTransientView(graph, 'radii', 'float32', 30)
                }
              : {})
          }
        });
        const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
        expect(sources.some(({id}) => id.endsWith('-finish'))).toBe(true);
        for (const {id, source} of sources) {
          expect(() => new WgslReflect(source), id).not.toThrow();
          expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
        }
      } finally {
        device.destroy();
      }
    });
  }
}
