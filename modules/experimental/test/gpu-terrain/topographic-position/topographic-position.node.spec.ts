// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPU_TERRAIN_WEISS_LANDFORMS,
  GPUTerrainTopographicPosition,
  GPUTerrainWeissLandforms,
  getGPUTerrainWeissLandformsParameterValues,
  type GPUTerrainTopographicPositionProps,
  type GPUTerrainWeissLandformsProps
} from '../../../src/gpu-terrain/topographic-position';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createBand} from '../terrain-test-utils';

it('GPUTerrainTopographicPosition builds one exact summed-area table and validates props', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Partial<GPUTerrainTopographicPositionProps> = {}) => {
    instance++;
    return new GPUTerrainTopographicPosition({
      width: 6,
      height: 5,
      elevation: createBand(graph, `elevation-${instance}`, 30),
      scales: [{radius: 1}, {radius: 4, innerRadius: 2}],
      deviationFromMean: createTransientView(graph, `dev-${instance}`, 'float32', 60),
      ...overrides
    });
  };
  const contributor = create();
  expect(contributor.requiredHalo).toBe(4);
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  expect(ids[0]).toBe('terrain-topographic-position-elevation');
  expect(ids[1]).toBe('terrain-topographic-position-summed-area-quantize');
  expect(ids.at(-1)).toBe('terrain-topographic-position-evaluate');
  expect(ids.filter(id => id.includes('-transpose-'))).toHaveLength(5);
  expect(ids.every(id => id.startsWith('terrain-topographic-position-'))).toBe(true);

  expect(() => create({deviationFromMean: undefined})).toThrow(/at least one output/);
  expect(() => create({scales: []})).toThrow(/scales must contain/);
  expect(() => create({scales: [{radius: 0}]})).toThrow(/integer radius/);
  expect(() => create({scales: [{radius: 2, innerRadius: 2}]})).toThrow(/innerRadius < radius/);
  expect(() => create({quantum: 0.003})).toThrow(/power of two/);
  expect(() =>
    create({deviationFromMean: createTransientView(graph, 'short-dev', 'float32', 30)})
  ).toThrow(/must contain 60 values/);
  const shared = createTransientView(graph, 'shared', 'float32', 60);
  expect(() => create({deviationFromMean: shared, topographicPositionIndex: shared})).toThrow(
    /must not share buffers/
  );

  // The table needs 3 * pixelCount words per binding.
  const large = new GPUTerrainTopographicPosition({
    id: 'large',
    width: 8192,
    height: 2048,
    elevation: createBand(graph, 'large-elevation', 8192 * 2048),
    scales: [{radius: 1}],
    maximumDeviation: createTransientView(graph, 'large-dev', 'float32', 8192 * 2048)
  });
  expect(() => large.getCommandNodes(graph)).toThrow(/maxStorageBufferBindingSize/);
});

it('GPUTerrainWeissLandforms composes position, slope, statistics, and classification', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Partial<GPUTerrainWeissLandformsProps> = {}) => {
    instance++;
    return new GPUTerrainWeissLandforms({
      id: `weiss-${instance}`,
      width: 6,
      height: 5,
      elevation: createBand(graph, `elevation-${instance}`, 30),
      settings: createTransientView(graph, `settings-${instance}`, 'float32', 8),
      landforms: createTransientView(graph, `landforms-${instance}`, 'uint32', 30),
      ...overrides
    });
  };
  const global = create();
  expect(global.requiredHalo).toBe(15);
  const globalIds = global.getCommandNodes(graph).map(node => node.id);
  expect(globalIds).toContain('weiss-1-slope');
  expect(globalIds).toContain('weiss-1-squares');
  expect(globalIds.at(-1)).toBe('weiss-1-classify');
  const local = create({
    standardization: 'local',
    smallScale: {radius: 2},
    largeScale: {radius: 6}
  });
  expect(local.requiredHalo).toBe(6);
  const localIds = local.getCommandNodes(graph).map(node => node.id);
  expect(localIds.some(id => id === 'weiss-2-squares' || id.startsWith('weiss-2-sum-'))).toBe(
    false
  );
  expect(globalIds.some(id => id.startsWith('weiss-1-sum-0'))).toBe(true);

  expect(Array.from(getGPUTerrainWeissLandformsParameterValues({cellSize: [2, 3]}))).toEqual([
    2, 3, 1, 0, 0, 1, 5, 0
  ]);
  expect(GPU_TERRAIN_WEISS_LANDFORMS.mountainTop).toBe(10);
  expect(() => create({landforms: undefined})).toThrow(/at least one output/);
  expect(() => create({standardization: 'median' as never})).toThrow(/global or local/);
  expect(() => create({smallScale: {radius: 1.5}})).toThrow(/integer radius/);
  expect(() =>
    create({settings: createTransientView(graph, 'short-settings', 'float32', 7)})
  ).toThrow(/at least 8/);
  expect(() =>
    create({standardizedPosition: createTransientView(graph, 'short-std', 'float32', 30)})
  ).toThrow(/two values per pixel/);
});
