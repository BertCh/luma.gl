// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import type {GPUSpatialContext} from '../../../src/analysis-core/contracts';
import {GPUSegmentRingAssembly} from '../../../src/analysis-core/ring-assembly';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it.each([
  'linear',
  'great-circle',
  undefined
] as const)('GPUSegmentRingAssembly preserves the longitude/latitude edge model %s', edgeModel => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const spatialContext: GPUSpatialContext = {
      coordinateSpace: 'longitude-latitude',
      metric: 'none',
      ...(edgeModel === undefined ? {} : {edgeModel})
    };
    const assembly = new GPUSegmentRingAssembly({
      spatialContext,
      endpoints: createTransientView(graph, 'endpoints', 'float32x4', 4),
      output: {
        positions: createTransientView(graph, 'positions', 'float32x2', 5),
        ringOffsets: createTransientView(graph, 'ring-offsets', 'uint32', 2),
        status: {
          count: createTransientView(graph, 'count', 'uint32', 1),
          overflow: createTransientView(graph, 'overflow', 'uint32', 1)
        }
      }
    });
    expect(assembly.spatialContext).toEqual(spatialContext);
    expect('edgeModel' in assembly.spatialContext).toBe(edgeModel !== undefined);
    expect(assembly.spatialContext.edgeModel).toBe(edgeModel);
    expect(
      () =>
        new GPUSegmentRingAssembly({
          ...assembly.props,
          spatialContext: {coordinateSpace: 'planar', metric: 'native', edgeModel: 'linear'}
        })
    ).toThrow('segment-ring-assembly edgeModel needs a longitude-latitude coordinate space');
  } finally {
    device.destroy();
  }
});
