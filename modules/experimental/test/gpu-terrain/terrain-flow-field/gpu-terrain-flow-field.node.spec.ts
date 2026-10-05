// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPU_TERRAIN_FLOW_FIELD_PARAMETER_LENGTH,
  GPUTerrainFlowField,
  getGPUTerrainFlowFieldParameterValues,
  type GPUTerrainFlowFieldProps
} from '../../../src/gpu-terrain/terrain-flow-field';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Partial<GPUTerrainFlowFieldProps> = {}) => {
    instance++;
    return new GPUTerrainFlowField({
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
      settings: createTransientView(graph, `settings-${instance}`, 'float32', 8),
      velocities: createTransientView(graph, `velocities-${instance}`, 'float32x2', 30),
      ...overrides
    });
  };
  return {device, graph, create};
}

it('getGPUTerrainFlowFieldParameterValues packs settings', () => {
  expect(GPU_TERRAIN_FLOW_FIELD_PARAMETER_LENGTH).toBe(8);
  expect(
    Array.from(
      getGPUTerrainFlowFieldParameterValues({
        cellSize: [2, 3],
        northEdge: 0.25,
        southEdge: 0.5,
        wind: [4, -5],
        verticalExaggeration: 1.5
      })
    )
  ).toEqual([2, 3, 0.25, 0.5, 4, -5, 1.5, 0]);
  expect(getGPUTerrainFlowFieldParameterValues({cellSize: [1, 1], wind: [0, 0]})[6]).toBe(1);
  expect(() =>
    getGPUTerrainFlowFieldParameterValues({cellSize: [1, 1], wind: [0, 0]}, new Float32Array(7))
  ).toThrow(/8 values/);
});

it('GPUTerrainFlowField prefixes node ids', () => {
  const {device, graph, create} = createFixture();
  const contributor = create({id: 'wind'});
  expect(contributor.getCommandNodes(graph).map(node => node.id)).toEqual([
    'wind-elevation',
    'wind-deflect'
  ]);
  device.destroy();
});

it('GPUTerrainFlowField validates its props', () => {
  const {device, graph, create} = createFixture();
  expect(() => create({width: 0})).toThrow(/dimensions/);
  expect(() => create({settings: createTransientView(graph, 's7', 'float32', 7)})).toThrow(
    /settings/
  );
  expect(() => create({cellSizeMode: 'polar' as never})).toThrow(/cellSizeMode/);
  expect(() => create({velocities: createTransientView(graph, 'short', 'float32x2', 29)})).toThrow(
    /one row per cell/
  );
  expect(() =>
    create({velocities: createTransientView(graph, 'scalar', 'float32', 30) as never})
  ).toThrow(/float32x2/);
  const inputs = graph.createTransientBuffer({id: 'inputs', byteLength: 4096, usage: 128});
  expect(() =>
    create({
      settings: graph.createDataView(inputs, {format: 'float32', length: 8}),
      velocities: graph.createDataView(inputs, {format: 'float32x2', length: 30, byteOffset: 1024})
    })
  ).toThrow(/share buffers/);
  const otherGraph = new GPUCommandGraph(device);
  expect(() =>
    create({
      velocities: createTransientView(otherGraph, 'foreign', 'float32x2', 30)
    }).getCommandNodes(graph)
  ).toThrow(/belong to the target graph/);
  device.destroy();
});
