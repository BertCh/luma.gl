// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {getRasterGridGeoreference} from '../../../src/analysis-core/raster-grid';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {
  GPUTerrainRuggedness,
  type GPUTerrainRuggednessProps
} from '../../../src/gpu-terrain/topographic-position/gpu-terrain-ruggedness';
import {
  GPU_TERRAIN_VECTOR_RUGGEDNESS_PARAMETER_LENGTH,
  getGPUTerrainVectorRuggednessParameterValues,
  GPUTerrainVectorRuggedness,
  type GPUTerrainVectorRuggednessProps
} from '../../../src/gpu-terrain/topographic-position/gpu-terrain-vector-ruggedness';
import {
  createNullWebGPUDevice,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {createBand} from '../terrain-test-utils';

const GRID = makeGPURasterMetadata({width: 6, height: 5, cellSize: [2, 3], origin: [0, 60]});

it('GPUTerrainRuggedness schedules one window kernel and validates props', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Partial<GPUTerrainRuggednessProps> = {}) => {
    instance++;
    return new GPUTerrainRuggedness({
      grid: GRID,
      elevation: createBand(graph, `elevation-${instance}`, 30),
      output: {roughness: createTransientView(graph, `roughness-${instance}`, 'float32', 30)},
      ...overrides
    });
  };
  const contributor = create();
  expect(contributor.requiredHalo).toBe(1);
  expect(contributor.getCommandNodes(graph).map(node => node.id)).toEqual([
    'terrain-ruggedness-elevation',
    'terrain-ruggedness-window'
  ]);
  expect(
    create({id: 'tri', edgeMode: 'extrapolate', terrainRuggednessAlgorithm: 'wilson'})
      .getCommandNodes(graph)
      .every(node => node.id.startsWith('tri-'))
  ).toBe(true);

  expect(() => create({output: {}})).toThrow(/at least one output/);
  expect(() =>
    create({output: {roughness: createTransientView(graph, 'short', 'float32', 29)}})
  ).toThrow(/one value per pixel/);
  expect(() => create({terrainRuggednessAlgorithm: 'other' as never})).toThrow(
    /terrainRuggednessAlgorithm/
  );
  expect(() => create({edgeMode: 'wrap' as never})).toThrow(/edgeMode/);
  expect(() =>
    create({edgeMode: 'extrapolate', grid: makeGPURasterMetadata({width: 1, height: 30})})
  ).toThrow(/at least 2/);
  expect(() => create({grid: {...GRID, width: 0}})).toThrow(/dimensions/);
  const shared = createBand(graph, 'shared', 30);
  expect(() => create({elevation: shared, output: {roughness: shared.storage.values}})).toThrow(
    /share buffers/
  );
  const otherGraph = new GPUCommandGraph(device);
  expect(() =>
    new GPUTerrainRuggedness({
      grid: makeGPURasterMetadata({width: 2, height: 2}),
      elevation: createBand(otherGraph, 'foreign', 4),
      output: {roughness: createTransientView(otherGraph, 'foreign-roughness', 'float32', 4)}
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
      grid: GRID,
      elevation: createBand(graph, `elevation-${instance}`, 30),
      parameters: createTransientView(graph, `settings-${instance}`, 'float32', 8),
      output: {vectorRuggedness: createTransientView(graph, `vrm-${instance}`, 'float32', 30)},
      ...overrides
    });
  };
  const contributor = create();
  expect(contributor.requiredHalo).toBe(2);
  expect(create({maximumRadius: 4}).requiredHalo).toBe(5);
  expect(contributor.getCommandNodes(graph).map(node => node.id)).toEqual([
    'terrain-vector-ruggedness-elevation',
    'terrain-vector-ruggedness-normals',
    'terrain-vector-ruggedness-window'
  ]);
  expect(
    create({maximumRadius: 4, id: 'wide-vrm'})
      .getCommandNodes(graph)
      .map(node => node.id)
  ).toEqual(['wide-vrm-elevation', 'wide-vrm-normals', 'wide-vrm-window-rows', 'wide-vrm-window']);
  expect(GPU_TERRAIN_VECTOR_RUGGEDNESS_PARAMETER_LENGTH).toBe(8);
  // Rows of 3 units advance south from y = 60 over 5 rows: edges 60 and 45. The radius word is 0
  // (the kernel reads it as maximumRadius) unless settings.radius is set.
  expect(Array.from(getGPUTerrainVectorRuggednessParameterValues({}, GRID))).toEqual([
    2, 3, 1, 60, 45, 0, 0, 0
  ]);
  const georeference = getRasterGridGeoreference(GRID);
  expect(Array.from(getGPUTerrainVectorRuggednessParameterValues({}, GRID).subarray(0, 5))).toEqual(
    [...georeference.cellSize, 1, georeference.northEdge, georeference.southEdge]
  );
  expect(
    Array.from(getGPUTerrainVectorRuggednessParameterValues({zFactor: 4, radius: 3}, GRID))
  ).toEqual([2, 3, 4, 60, 45, 3, 0, 0]);
  const reused = new Float32Array(8).fill(9);
  const unit = makeGPURasterMetadata({width: 4, height: 4});
  expect(getGPUTerrainVectorRuggednessParameterValues({}, unit, reused)).toBe(reused);
  expect(Array.from(reused)).toEqual([1, 1, 1, 0, -4, 0, 0, 0]);
  expect(() => getGPUTerrainVectorRuggednessParameterValues({}, unit, new Float32Array(7))).toThrow(
    /getGPUTerrainVectorRuggednessParameterValues/
  );
  expect(() => getGPUTerrainVectorRuggednessParameterValues({radius: 0}, unit)).toThrow(/radius/);
  expect(() => getGPUTerrainVectorRuggednessParameterValues({radius: 1.5}, unit)).toThrow(/radius/);

  expect(() => create({output: {}})).toThrow(/at least one output/);
  expect(() => create({maximumRadius: 0})).toThrow(/maximumRadius/);
  expect(() => create({maximumRadius: 1.5})).toThrow(/maximumRadius/);
  expect(() => create({parameters: createTransientView(graph, 'settings7', 'float32', 7)})).toThrow(
    /parameters/
  );
  expect(() => create({cellSizeMode: 'polar' as never})).toThrow(/cellSizeMode/);
  expect(() => create({rowDirection: 'east' as never})).toThrow(/rowDirection/);
  expect(() => create({borderMode: 'wrap' as never})).toThrow(/borderMode/);
  expect(() => create({grid: {...GRID, width: 0}})).toThrow(/dimensions/);
  device.destroy();
});

it('GPUTerrainVectorRuggedness serves every per-frame radius with the same pipelines', () => {
  const device = createNullWebGPUDevice();
  let instance = 0;
  const build = (maximumRadius: number) => {
    instance++;
    const graph = new GPUCommandGraph(device);
    return new GPUTerrainVectorRuggedness({
      grid: GRID,
      elevation: createBand(graph, `elevation-${instance}`, 30),
      parameters: createTransientView(graph, `settings-${instance}`, 'float32', 8),
      output: {vectorRuggedness: createTransientView(graph, `vrm-${instance}`, 'float32', 30)},
      maximumRadius
    }).getCommandNodes(graph);
  };
  // The radius is a value of the parameters buffer, not of the kernels: a second graph build with
  // other buffers (and so any radius written into them) compiles to the same pipelines, and the
  // number of pipelines does not grow with the radius.
  for (const maximumRadius of [1, 2, 4, 9]) {
    const first = getGPUCommandNodePipelineKeys(build(maximumRadius));
    const second = getGPUCommandNodePipelineKeys(build(maximumRadius));
    expect([...second]).toEqual([...first]);
    // Canonicalization, normals and the window sum (direct, or rows and columns from radius 3).
    expect(first.size).toBe(maximumRadius < 3 ? 3 : 4);
  }
  // The maximum is compile time: it changes the kernels, the per-frame radius does not.
  expect(getGPUCommandNodePipelineKeys(build(4))).not.toEqual(
    getGPUCommandNodePipelineKeys(build(5))
  );
  device.destroy();
});
