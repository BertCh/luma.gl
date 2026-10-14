// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  createLonLatPolygonCheckNode,
  getLonLatRingWGSL,
  hasLonLatPolygonStatus,
  LON_LAT_RING_WGSL
} from '../../../src/analysis-core/lon-lat-ring-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('exports one binding-free ring walk source with every shared function and tolerance', () => {
  expect(getLonLatRingWGSL()).toBe(LON_LAT_RING_WGSL);
  for (const name of [
    'lonLatWrapLongitudeStep',
    'lonLatIsPoleEdge',
    'lonLatRingStep',
    'lonLatIsCrossingEdge',
    'lonLatIsAmbiguousLinearEdge',
    'lonLatIsAmbiguousGreatCircleEdge',
    'lonLatIsPoleWinding'
  ]) {
    expect(LON_LAT_RING_WGSL).toContain(`fn ${name}(`);
  }
  expect(LON_LAT_RING_WGSL).toContain('rawStep - 360.0 * round(rawStep / 360.0)');
  expect(LON_LAT_RING_WGSL).toContain('LON_LAT_POLE_LATITUDE_TOLERANCE: f32 = 1e-6');
  expect(LON_LAT_RING_WGSL).toContain('LON_LAT_AMBIGUOUS_STEP_TOLERANCE: f32 = 1e-4');
  expect(LON_LAT_RING_WGSL).toContain('LON_LAT_POLE_WINDING_LIMIT: f32 = 180.0');
  expect(LON_LAT_RING_WGSL).not.toContain('@binding');
  expect(() => new WgslReflect(LON_LAT_RING_WGSL)).not.toThrow();
});

it('binds only requested status fields, including a harmless empty-status node', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const mask of [0, 1, 2, 3]) {
      const graph = new GPUCommandGraph(device);
      const status = {
        ...(mask & 1
          ? {antimeridianUncertainCount: createTransientView(graph, 'crossing', 'uint32', 1)}
          : {}),
        ...(mask & 2 ? {poleUncertainCount: createTransientView(graph, 'pole', 'uint32', 1)} : {})
      };
      expect(hasLonLatPolygonStatus(status)).toBe(mask !== 0);
      const node = createLonLatPolygonCheckNode(graph, {
        id: 'check',
        operation: 'test-consumer',
        positions: createTransientView(graph, 'positions', 'float32x2', 8),
        ringOffsets: createTransientView(graph, 'rings', 'uint32', 3),
        status
      });
      const [kernel] = getGPUCommandNodeKernelSources([node]);
      expect(() => new WgslReflect(kernel.source)).not.toThrow();
      expect(kernel.operation).toBe('test-consumer');
      expect(kernel.source.includes('atomicAdd(&antimeridianUncertainCount')).toBe(
        Boolean(mask & 1)
      );
      expect(kernel.source.includes('atomicAdd(&poleUncertainCount')).toBe(Boolean(mask & 2));
      expect(new WgslReflect(kernel.source).storage.length).toBe(
        2 + Number(Boolean(mask & 1)) + Number(Boolean(mask & 2))
      );
    }
    expect(hasLonLatPolygonStatus()).toBe(false);
  } finally {
    device.destroy();
  }
});

it('validates declared status scalar lengths using the consumer operation name', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    expect(() =>
      createLonLatPolygonCheckNode(graph, {
        id: 'check',
        operation: 'Consumer',
        positions: createTransientView(graph, 'positions', 'float32x2', 0),
        ringOffsets: createTransientView(graph, 'rings', 'uint32', 1),
        status: {poleUncertainCount: createTransientView(graph, 'pole', 'uint32', 0)}
      })
    ).toThrow('Consumer poleUncertainCount must contain one uint32 row');
  } finally {
    device.destroy();
  }
});
