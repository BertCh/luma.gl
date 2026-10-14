// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {GPUShapeDescriptors} from '../../../src/gpu-spatial-analysis/shape-descriptors';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('review D-19: orientation guards atan2(0, 0) and keeps the kernel parseable', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'shape-descriptors'});
  const contributor = new GPUShapeDescriptors({
    positions: createTransientView(graph, 'positions', 'float32x2', 8),
    ringOffsets: createTransientView(graph, 'ring-offsets', 'uint32', 3),
    polygonOffsets: createTransientView(graph, 'polygon-offsets', 'uint32', 3),
    featureOffsets: createTransientView(graph, 'feature-offsets', 'uint32', 3),
    parameters: createTransientView(graph, 'parameters', 'float32', 4),
    output: {orientation: createTransientView(graph, 'orientation', 'float32', 2)}
  });
  const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
  const source = sources
    .map(entry => entry.source)
    .find(text => text.includes('orientation[orientationOffset'));
  expect(source).toBeDefined();
  expect(() => new WgslReflect(source!)).not.toThrow();
  expect(source).toContain(
    'select(0.0, 0.5 * atan2(2.0 * xy, xx - yy), (xy != 0.0) || (xx != yy))'
  );
});
