// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPU_TERRAIN_HYDROLOGIC_INDICES_PARAMETER_LENGTH,
  GPUTerrainHydrologicIndices,
  getGPUTerrainHydrologicIndicesParameterValues,
  type GPUTerrainHydrologicIndicesProps
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-hydrologic-indices';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Partial<GPUTerrainHydrologicIndicesProps> = {}) => {
    instance++;
    return new GPUTerrainHydrologicIndices({
      width: 6,
      height: 5,
      elevation: {
        id: `elevation-${instance}`,
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: createTransientView(graph, `elevation-${instance}`, 'float32', 30)
        }
      },
      accumulation: createTransientView(graph, `accumulation-${instance}`, 'float32', 30),
      settings: createTransientView(graph, `settings-${instance}`, 'float32', 8),
      wetnessIndex: createTransientView(graph, `wetness-${instance}`, 'float32', 30),
      ...overrides
    });
  };
  return {device, graph, create};
}

it('getGPUTerrainHydrologicIndicesParameterValues packs settings', () => {
  expect(GPU_TERRAIN_HYDROLOGIC_INDICES_PARAMETER_LENGTH).toBe(8);
  expect(
    Array.from(
      getGPUTerrainHydrologicIndicesParameterValues({
        cellSize: [2, 3],
        northEdge: 0.25,
        southEdge: 0.5,
        minimumSlope: 0.125
      })
    )
  ).toEqual([2, 3, 0.25, 0.5, 0.125, 0, 0, 0]);
  expect(getGPUTerrainHydrologicIndicesParameterValues({cellSize: [1, 1]})[4]).toBeCloseTo(
    0.001,
    9
  );
  expect(() =>
    getGPUTerrainHydrologicIndicesParameterValues({cellSize: [1, 1]}, new Float32Array(7))
  ).toThrow(/8 values/);
});

it('GPUTerrainHydrologicIndices prefixes node ids', () => {
  const {device, graph, create} = createFixture();
  const contributor = create({id: 'twi'});
  expect(contributor.getCommandNodes(graph).map(node => node.id)).toEqual([
    'twi-elevation',
    'twi-indices'
  ]);
  device.destroy();
});

it('GPUTerrainHydrologicIndices validates its props', () => {
  const {device, graph, create} = createFixture();
  const view = (name: string, format: 'float32' | 'uint32', length: number) =>
    createTransientView(graph, name, format, length);
  expect(() => create({width: 0})).toThrow(/dimensions/);
  expect(() => create({wetnessIndex: undefined})).toThrow(/at least one output/);
  expect(() => create({settings: view('s7', 'float32', 7)})).toThrow(/settings/);
  expect(() => create({cellSizeMode: 'polar' as never})).toThrow(/cellSizeMode/);
  expect(() => create({accumulation: view('short', 'float32', 29)})).toThrow(/one value per cell/);
  expect(() => create({slope: view('uint', 'uint32', 30) as never})).toThrow(/float32/);
  const shared = view('shared', 'float32', 30);
  expect(() => create({wetnessIndex: shared, slope: shared})).toThrow(/share buffers/);
  const accumulation = view('input', 'float32', 30);
  expect(() => create({accumulation, wetnessIndex: accumulation})).toThrow(/share buffers/);
  const otherGraph = new GPUCommandGraph(device);
  expect(() =>
    create({
      wetnessIndex: createTransientView(otherGraph, 'foreign', 'float32', 30)
    }).getCommandNodes(graph)
  ).toThrow(/belong to the target graph/);
  device.destroy();
});
