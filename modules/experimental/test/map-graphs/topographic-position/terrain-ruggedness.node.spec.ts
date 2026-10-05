// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUTerrainRuggedness,
  type GPUTerrainRuggednessProps
} from '../../../src/map-graphs/topographic-position/gpu-terrain-ruggedness';
import {
  GPU_TERRAIN_VECTOR_RUGGEDNESS_PARAMETER_LENGTH,
  getGPUTerrainVectorRuggednessParameterValues,
  GPUTerrainVectorRuggedness,
  type GPUTerrainVectorRuggednessProps
} from '../../../src/map-graphs/topographic-position/gpu-terrain-vector-ruggedness';
import {createNullWebGPUDevice} from '../map-graph-test-utils';

function createBand(graph: GPUCommandGraph, id: string, length: number) {
  return {
    id,
    format: 'float32' as const,
    storage: {kind: 'buffer' as const, values: createTransientView(graph, id, 'float32', length)}
  };
}

it('GPUTerrainRuggedness schedules one window kernel and validates props', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Partial<GPUTerrainRuggednessProps> = {}) => {
    instance++;
    return new GPUTerrainRuggedness({
      width: 6,
      height: 5,
      elevation: createBand(graph, `elevation-${instance}`, 30),
      roughness: createTransientView(graph, `roughness-${instance}`, 'float32', 30),
      ...overrides
    });
  };
  const recipe = create();
  expect(recipe.requiredHalo).toBe(1);
  expect(recipe.recipe).toBe('terrain-ruggedness');
  expect(recipe.getCommandNodes(graph).map(node => node.id)).toEqual([
    'terrain-ruggedness-elevation',
    'terrain-ruggedness-window'
  ]);
  expect(
    create({id: 'tri', edgeMode: 'extrapolate', terrainRuggednessAlgorithm: 'wilson'})
      .getCommandNodes(graph)
      .every(node => node.id.startsWith('tri-'))
  ).toBe(true);

  expect(() => create({roughness: undefined})).toThrow(/at least one output/);
  expect(() => create({roughness: createTransientView(graph, 'short', 'float32', 29)})).toThrow(
    /one value per pixel/
  );
  expect(() => create({terrainRuggednessAlgorithm: 'other' as never})).toThrow(
    /terrainRuggednessAlgorithm/
  );
  expect(() => create({edgeMode: 'wrap' as never})).toThrow(/edgeMode/);
  expect(() => create({edgeMode: 'extrapolate', width: 1, height: 30})).toThrow(/at least 2/);
  expect(() => create({width: 0})).toThrow(/dimensions/);
  const shared = createBand(graph, 'shared', 30);
  expect(() => create({elevation: shared, roughness: shared.storage.values})).toThrow(
    /share buffers/
  );
  const otherGraph = new GPUCommandGraph(device);
  expect(() =>
    new GPUTerrainRuggedness({
      width: 2,
      height: 2,
      elevation: createBand(otherGraph, 'foreign', 4),
      roughness: createTransientView(otherGraph, 'foreign-roughness', 'float32', 4)
    }).getCommandNodes(graph)
  ).toThrow(/belong to the target graph/);
  device.destroy();
});

it('GPUTerrainVectorRuggedness schedules normals and window kernels', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Partial<GPUTerrainVectorRuggednessProps> = {}) => {
    instance++;
    return new GPUTerrainVectorRuggedness({
      width: 6,
      height: 5,
      elevation: createBand(graph, `elevation-${instance}`, 30),
      settings: createTransientView(graph, `settings-${instance}`, 'float32', 8),
      vectorRuggedness: createTransientView(graph, `vrm-${instance}`, 'float32', 30),
      ...overrides
    });
  };
  const recipe = create();
  expect(recipe.recipe).toBe('terrain-vector-ruggedness');
  expect(recipe.requiredHalo).toBe(2);
  expect(create({radius: 4}).requiredHalo).toBe(5);
  expect(recipe.getCommandNodes(graph).map(node => node.id)).toEqual([
    'terrain-vector-ruggedness-elevation',
    'terrain-vector-ruggedness-normals',
    'terrain-vector-ruggedness-window'
  ]);
  expect(GPU_TERRAIN_VECTOR_RUGGEDNESS_PARAMETER_LENGTH).toBe(8);
  expect(Array.from(getGPUTerrainVectorRuggednessParameterValues({cellSize: [2, 3]}))).toEqual([
    2, 3, 1, 0, 0, 0, 0, 0
  ]);
  expect(
    Array.from(
      getGPUTerrainVectorRuggednessParameterValues({
        cellSize: [2, 3],
        zFactor: 4,
        northEdge: 60,
        southEdge: 59
      })
    )
  ).toEqual([2, 3, 4, 60, 59, 0, 0, 0]);
  const reused = new Float32Array(8).fill(9);
  expect(getGPUTerrainVectorRuggednessParameterValues({cellSize: [1, 1]}, reused)).toBe(reused);
  expect(Array.from(reused)).toEqual([1, 1, 1, 0, 0, 0, 0, 0]);
  expect(() =>
    getGPUTerrainVectorRuggednessParameterValues({cellSize: [1, 1]}, new Float32Array(7))
  ).toThrow(/8 values/);

  expect(() => create({vectorRuggedness: undefined})).toThrow(/at least one output/);
  expect(() => create({radius: 0})).toThrow(/radius/);
  expect(() => create({radius: 1.5})).toThrow(/radius/);
  expect(() => create({settings: createTransientView(graph, 'settings7', 'float32', 7)})).toThrow(
    /settings/
  );
  expect(() => create({cellSizeMode: 'polar' as never})).toThrow(/cellSizeMode/);
  expect(() => create({rowDirection: 'east' as never})).toThrow(/rowDirection/);
  expect(() => create({borderMode: 'wrap' as never})).toThrow(/borderMode/);
  expect(() => create({width: 0})).toThrow(/dimensions/);
  device.destroy();
});
