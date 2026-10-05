// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUTerrainCastShadowParameterValues,
  GPU_TERRAIN_CAST_SHADOW_PARAMETER_LENGTH,
  GPUTerrainCastShadow,
  type GPUTerrainCastShadowProps
} from '../../../src/gpu-terrain/terrain-illumination/gpu-terrain-cast-shadow';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('getGPUTerrainCastShadowParameterValues packs the sun line geometry', () => {
  expect(GPU_TERRAIN_CAST_SHADOW_PARAMETER_LENGTH).toBe(16);
  const pack = (azimuthDegrees: number, rowDirection?: 'south' | 'north') =>
    Array.from(
      getGPUTerrainCastShadowParameterValues(
        {cellSize: [10, 12], azimuthDegrees, altitudeDegrees: 20, zFactor: 2},
        rowDirection
      )
    );
  // East: x-major, travel +1, no slope.
  expect(pack(90)).toEqual([10, 12, 2, 0, 0, 0, 0, 90, 20, expect.any(Number), 1, 0, 1, 1, 0, 0]);
  // North on a south-increasing grid goes toward row 0: y-major, travel -1.
  expect(pack(0).slice(10, 14)).toEqual([0, 0, -1, 1]);
  // North on a north-increasing grid goes toward higher rows.
  expect(pack(0, 'north').slice(10, 14)).toEqual([0, 0, 1, 1]);
  // North-east is the exact diagonal: slope 1, step sqrt(2).
  const diagonal = pack(45);
  expect(Math.abs(diagonal[11])).toBe(65536);
  expect(diagonal[13]).toBe(Math.fround(Math.SQRT2));
  // A 22.5 degree sun is steep in one axis only.
  const steep = pack(22.5);
  expect(steep[10]).toBe(0); // y-major
  expect(Math.abs(steep[11])).toBeLessThan(65536);
  expect(Math.abs(steep[11])).toBeGreaterThan(0);
  expect(
    Array.from(
      getGPUTerrainCastShadowParameterValues({
        cellSize: [1, 1],
        azimuthDegrees: 5,
        altitudeDegrees: 5
      })
    )[9]
  ).toBeCloseTo(0.2666, 6);
});

it('getGPUTerrainCastShadowParameterValues validates its input', () => {
  const valid = {cellSize: [1, 1] as [number, number], azimuthDegrees: 1, altitudeDegrees: 1};
  expect(() => getGPUTerrainCastShadowParameterValues({...valid, cellSize: [0, 1]})).toThrow(
    /cell size/
  );
  expect(() =>
    getGPUTerrainCastShadowParameterValues({...valid, angularRadiusDegrees: -1})
  ).toThrow(/angular radius/);
  expect(() =>
    getGPUTerrainCastShadowParameterValues({...valid, azimuthDegrees: Number.NaN})
  ).toThrow(/finite/);
  expect(() =>
    getGPUTerrainCastShadowParameterValues(valid, 'south', new Float32Array(15))
  ).toThrow(/16 values/);
});

it('GPUTerrainCastShadow validates props and schedules one sweep node', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Partial<GPUTerrainCastShadowProps> = {}) => {
    instance++;
    const length = 30;
    return new GPUTerrainCastShadow({
      width: 6,
      height: 5,
      elevation: {
        id: `elevation-${instance}`,
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: createTransientView(graph, `elevation-${instance}`, 'float32', length)
        }
      },
      settings: createTransientView(graph, `settings-${instance}`, 'float32', 16),
      sunVisibility: createTransientView(graph, `visibility-${instance}`, 'float32', length),
      ...overrides
    });
  };
  const contributor = create();
  expect(contributor.id).toBe('terrain-cast-shadow');
  // Whole tile by default: the diagonal covers every azimuth.
  expect(contributor.maximumRadius).toBe(Math.ceil(Math.hypot(6, 5)));
  expect(contributor.requiredHalo).toBe(contributor.maximumRadius);
  expect(create({maximumRadius: 4}).requiredHalo).toBe(4);
  const nodes = contributor.getCommandNodes(graph);
  expect(nodes.some(node => node.id === 'terrain-cast-shadow-sweep')).toBe(true);
  expect(nodes.filter(node => node.id.includes('sweep'))).toHaveLength(1);

  expect(() => create({sunVisibility: undefined})).toThrow(/at least one output/);
  expect(() => create({maximumRadius: 0})).toThrow(/maximumRadius/);
  expect(() => create({maximumRadius: 2.5})).toThrow(/maximumRadius/);
  expect(() => create({width: 40000})).toThrow(/32767/);
  expect(() =>
    create({settings: createTransientView(graph, 'short-settings', 'float32', 8)})
  ).toThrow(/at least 16/);
  expect(() =>
    create({sunVisibility: createTransientView(graph, 'short-visibility', 'float32', 3)})
  ).toThrow(/30 float32/);
  expect(() => create({cellSizeMode: 'bogus' as 'uniform'})).toThrow(/cellSizeMode/);
  expect(() => create({rowDirection: 'up' as 'south'})).toThrow(/rowDirection/);
  const other = new GPUCommandGraph(device);
  expect(() => create().getCommandNodes(other)).toThrow(/belong/);
});
