// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {GPUClassBreaks} from '../../../src/gpu-dataframe/column-classification/gpu-class-breaks';
import {getGPUClassBreaksParameterLength} from '../../../src/gpu-dataframe/column-classification/class-breaks-parameters';
import {
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES,
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';

for (const rowCount of [1, 4103, 1 << 24]) {
  for (const masked of [false, true]) {
    it(`WP6 moment kernels parse and fit core limits: rows=${rowCount}, mask=${masked}`, () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      const contributor = new GPUClassBreaks({
        values: createTransientView(graph, 'values', 'float32', rowCount),
        mask: masked ? createTransientView(graph, 'mask', 'uint32', rowCount) : undefined,
        parameters: createTransientView(
          graph,
          'parameters',
          'float32',
          getGPUClassBreaksParameterLength(4)
        ),
        maximumClassCount: 4,
        methods: ['standard-deviation', 'head-tail'],
        output: {
          breaks: createTransientView(graph, 'breaks', 'float32', 5),
          classCount: createTransientView(graph, 'class-count', 'uint32', 1),
          classCounts: createTransientView(graph, 'class-counts', 'uint32', 4)
        }
      });
      const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
      expect(sources.length).toBeGreaterThan(0);
      for (const {id, source} of sources) {
        expect(() => new WgslReflect(source), id).not.toThrow();
        expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
        expect(getWorkgroupMemoryBytes(source), id).toBeLessThanOrEqual(
          CORE_WORKGROUP_STORAGE_LIMIT_BYTES
        );
      }
      device.destroy();
    });
  }
}
