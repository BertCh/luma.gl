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
import {makeGPURasterBand, makeGPURasterMetadata} from '../../../src/gpu-raster';
import {getRasterGridGeoreference} from '../../../src/analysis-core/raster-grid';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const GRID = makeGPURasterMetadata({width: 6, height: 5});

function createFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Partial<GPUTerrainFlowFieldProps> = {}) => {
    instance++;
    return new GPUTerrainFlowField({
      grid: GRID,
      elevation: makeGPURasterBand(
        createTransientView(graph, `elevation-${instance}`, 'float32', 30),
        GRID,
        {id: `elevation-${instance}`}
      ),
      parameters: createTransientView(graph, `settings-${instance}`, 'float32', 8),
      output: {
        velocities: createTransientView(graph, `velocities-${instance}`, 'float32x2', 30)
      },
      ...overrides
    });
  };
  return {device, graph, create};
}

it('getGPUTerrainFlowFieldParameterValues packs settings', () => {
  expect(GPU_TERRAIN_FLOW_FIELD_PARAMETER_LENGTH).toBe(8);
  // Cell size and row edges come from the grid.
  const grid = makeGPURasterMetadata({width: 6, height: 5, cellSize: [2, 3], origin: [0, 20]});
  const {cellSize, northEdge, southEdge} = getRasterGridGeoreference(grid);
  expect([cellSize, northEdge, southEdge]).toEqual([[2, 3], 20, 5]);
  expect(
    Array.from(
      getGPUTerrainFlowFieldParameterValues({wind: [4, -5], verticalExaggeration: 1.5}, grid)
    )
  ).toEqual([2, 3, 20, 5, 4, -5, 1.5, 0]);
  expect(getGPUTerrainFlowFieldParameterValues({wind: [0, 0]}, GRID)[6]).toBe(1);
  expect(() =>
    getGPUTerrainFlowFieldParameterValues({wind: [0, 0]}, GRID, new Float32Array(7))
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
  expect(() => create({grid: {...GRID, width: 0}})).toThrow(/dimensions/);
  expect(() => create({parameters: createTransientView(graph, 's7', 'float32', 7)})).toThrow(
    /parameters/
  );
  expect(() => create({cellSizeMode: 'polar' as never})).toThrow(/cellSizeMode/);
  expect(() =>
    create({
      cellSizeMode: 'geographic',
      grid: {...GRID, affine: [1, 0.5, 0, 0, -1, 0]}
    })
  ).toThrow(/axis-aligned/);
  expect(() =>
    create({output: {velocities: createTransientView(graph, 'short', 'float32x2', 29)}})
  ).toThrow(/one row per cell/);
  expect(() =>
    create({output: {velocities: createTransientView(graph, 'scalar', 'float32', 30) as never}})
  ).toThrow(/float32x2/);
  const inputs = graph.createTransientBuffer({id: 'inputs', byteLength: 4096, usage: 128});
  expect(() =>
    create({
      parameters: graph.createDataView(inputs, {format: 'float32', length: 8}),
      output: {
        velocities: graph.createDataView(inputs, {
          format: 'float32x2',
          length: 30,
          byteOffset: 1024
        })
      }
    })
  ).toThrow(/share buffers/);
  const otherGraph = new GPUCommandGraph(device);
  expect(() =>
    create({
      output: {velocities: createTransientView(otherGraph, 'foreign', 'float32x2', 30)}
    }).getCommandNodes(graph)
  ).toThrow(/belong to the target graph/);
  device.destroy();
});
