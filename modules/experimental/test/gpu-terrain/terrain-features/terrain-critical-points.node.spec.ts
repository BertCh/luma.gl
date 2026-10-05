// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPU_TERRAIN_CRITICAL_POINT,
  GPUTerrainCriticalPoints
} from '../../../src/gpu-terrain/terrain-features/gpu-terrain-critical-points';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  CRITICAL_POINT,
  classifyCriticalPoints,
  createEggCrate,
  getEulerSum
} from './terrain-critical-points-oracle';

it('exports stable class codes shared with the oracle', () => {
  expect(GPU_TERRAIN_CRITICAL_POINT).toEqual({
    regular: 0,
    peak: 1,
    pit: 2,
    saddle: 3,
    boundary: 4,
    noData: 5
  });
  expect(CRITICAL_POINT).toEqual(GPU_TERRAIN_CRITICAL_POINT);
});

it('oracle classifies hand-made neighborhoods', () => {
  // 3x3 grids classify only the center pixel (everything else is boundary).
  const classify = (values: number[], connectivity: 8 | 6 = 8) =>
    classifyCriticalPoints(Float32Array.from(values), undefined, 3, 3, connectivity).classes[4];
  expect(classify([0, 0, 0, 0, 9, 0, 0, 0, 0])).toBe(CRITICAL_POINT.peak);
  expect(classify([9, 9, 9, 9, 0, 9, 9, 9, 9])).toBe(CRITICAL_POINT.pit);
  expect(classify([0, 1, 2, 0, 1, 2, 0, 1, 2])).toBe(CRITICAL_POINT.regular);
  // Saddle: high north and south, low west and east.
  expect(classify([5, 9, 5, 0, 5, 0, 5, 9, 5])).toBe(CRITICAL_POINT.saddle);
  // A flat plateau resolves by row-major index: the center has higher-index neighbors only on one
  // side, so it is regular, never a peak or a pit.
  expect(classify(new Array(9).fill(3))).toBe(CRITICAL_POINT.regular);
  // Invalid neighbor -> boundary, invalid center -> nodata.
  const validity = Uint32Array.from([1, 1, 1, 1, 1, 1, 1, 0, 1]);
  const withHole = classifyCriticalPoints(new Float32Array(9).fill(1), validity, 3, 3, 8);
  expect(withHole.classes[4]).toBe(CRITICAL_POINT.boundary);
  expect(withHole.classes[7]).toBe(CRITICAL_POINT.noData);
  // The 6-ring ignores the NE and SW diagonals: they may be invalid without changing the result.
  const sixValidity = Uint32Array.from([1, 1, 0, 1, 1, 1, 0, 1, 1]);
  expect(
    classifyCriticalPoints(Float32Array.from([0, 0, 0, 0, 9, 0, 0, 0, 0]), sixValidity, 3, 3, 6)
      .classes[4]
  ).toBe(CRITICAL_POINT.peak);
});

it('oracle egg-crate counts follow the torus Euler relation for 6-ring only', () => {
  const period = 8;
  const size = period * 2 + 2;
  const elevation = createEggCrate(size, size, period, 0.3);
  const window = {columnStart: 1, columnEnd: 1 + period * 2, rowStart: 1, rowEnd: 1 + period * 2};
  expect(
    getEulerSum(classifyCriticalPoints(elevation, undefined, size, size, 6), size, window).sum
  ).toBe(0);
  expect(
    getEulerSum(classifyCriticalPoints(elevation, undefined, size, size, 8), size, window).sum
  ).toBeLessThan(0);
});

it('GPUTerrainCriticalPoints validates props and schedules its nodes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Record<string, unknown> = {}) => {
    instance++;
    return new GPUTerrainCriticalPoints({
      width: 6,
      height: 5,
      elevation: {
        id: `elevation-${instance}`,
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: createTransientView(graph, `e-${instance}`, 'float32', 30)
        }
      },
      classes: createTransientView(graph, `classes-${instance}`, 'uint32', 30),
      ...overrides
    });
  };
  const recipe = create();
  expect(recipe.recipe).toBe('terrain-critical-points');
  expect(recipe.getCommandNodes(graph).map(node => node.id)).toEqual([
    'terrain-critical-points-elevation',
    'terrain-critical-points-classify'
  ]);
  const full = create({
    id: 'peaks',
    connectivity: 6,
    signChanges: createTransientView(graph, 'sign-changes', 'uint32', 30),
    counts: createTransientView(graph, 'counts', 'uint32', 6)
  });
  expect(full.getCommandNodes(graph).map(node => node.id)).toEqual([
    'peaks-elevation',
    'peaks-clear-counts',
    'peaks-classify'
  ]);

  expect(() => create({width: 0})).toThrow(/dimensions/);
  expect(() => create({connectivity: 4})).toThrow(/connectivity/);
  expect(() => create({classes: createTransientView(graph, 'short', 'uint32', 29)})).toThrow(
    /one value per pixel/
  );
  expect(() =>
    create({signChanges: createTransientView(graph, 'short-sign-changes', 'uint32', 3)})
  ).toThrow(/signChanges/);
  expect(() => create({counts: createTransientView(graph, 'short-counts', 'uint32', 5)})).toThrow(
    /one value per class/
  );
  const shared = createTransientView(graph, 'shared', 'uint32', 30);
  expect(() => create({classes: shared, signChanges: shared})).toThrow(/share buffers/);
  const otherGraph = new GPUCommandGraph(device);
  expect(() =>
    new GPUTerrainCriticalPoints({
      width: 2,
      height: 2,
      elevation: {
        id: 'foreign',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: createTransientView(otherGraph, 'foreign-e', 'float32', 4)
        }
      },
      classes: createTransientView(otherGraph, 'foreign-classes', 'uint32', 4)
    }).getCommandNodes(graph)
  ).toThrow(/belong to the target graph/);
  device.destroy();
});
