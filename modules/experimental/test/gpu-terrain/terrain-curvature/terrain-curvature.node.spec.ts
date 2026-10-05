// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPU_TERRAIN_CURVATURE_PARAMETER_LENGTH,
  GPUTerrainCurvature,
  getGPUTerrainCurvatureParameterValues,
  type GPUTerrainCurvatureKind,
  type GPUTerrainCurvatureProps
} from '../../../src/gpu-terrain/terrain-curvature';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {ORACLE_CURVATURE_KINDS} from './terrain-curvature-oracle';
import {createBand} from '../terrain-test-utils';

it('GPUTerrainCurvature packs settings and groups outputs into kernels', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Partial<GPUTerrainCurvatureProps> = {}) => {
    instance++;
    return new GPUTerrainCurvature({
      width: 6,
      height: 5,
      elevation: createBand(graph, `elevation-${instance}`, 30),
      settings: createTransientView(graph, `settings-${instance}`, 'float32', 12),
      curvatures: {
        mean: createTransientView(graph, `mean-${instance}`, 'float32', 30)
      },
      ...overrides
    });
  };
  const allCurvatures = (): Partial<Record<GPUTerrainCurvatureKind, never>> =>
    Object.fromEntries(
      ORACLE_CURVATURE_KINDS.map(kind => [
        kind,
        createTransientView(graph, `${kind}-all-${instance}`, 'float32', 30)
      ])
    );

  expect(GPU_TERRAIN_CURVATURE_PARAMETER_LENGTH).toBe(12);
  expect(Array.from(getGPUTerrainCurvatureParameterValues({cellSize: [2, 3]}))).toEqual([
    2,
    3,
    1,
    0,
    0,
    Math.fround(1e-6),
    Math.fround(0.55),
    Math.fround(0.45),
    0,
    0,
    0,
    0
  ]);
  expect(
    Array.from(
      getGPUTerrainCurvatureParameterValues({
        cellSize: [2, 3],
        zFactor: 4,
        northEdge: 5,
        southEdge: 6,
        flatGradient: 0.5,
        ringGains: [1, 2, 3, 4]
      })
    )
  ).toEqual([2, 3, 4, 5, 6, 0.5, 1, 2, 3, 4, 0, 0]);
  expect(() =>
    getGPUTerrainCurvatureParameterValues({cellSize: [1, 1], ringGains: [1, 1, 1, 1, 1]})
  ).toThrow(/ringGains/);

  const single = create();
  const singleIds = single.getCommandNodes(graph).map(node => node.id);
  expect(singleIds).toEqual(['terrain-curvature-elevation', 'terrain-curvature-curvature-0']);

  const full = create({
    id: 'full',
    curvatures: allCurvatures(),
    ringCurvature: createTransientView(graph, 'ring-full', 'float32', 30),
    validity: createTransientView(graph, 'validity-full', 'uint32', 30)
  });
  const fullIds = full.getCommandNodes(graph).map(node => node.id);
  expect(fullIds.filter(id => id.startsWith('full-curvature-'))).toEqual([
    'full-curvature-0',
    'full-curvature-1',
    'full-curvature-2',
    'full-curvature-3'
  ]);
  expect(fullIds.every(id => id.startsWith('full-'))).toBe(true);

  expect(create({method: 'evans-young'}).requiredHalo).toBe(1);
  expect(create({method: 'zevenbergen-thorne'}).requiredHalo).toBe(1);
  expect(create({method: 'florinsky'}).requiredHalo).toBe(2);
  const ringView = () => createTransientView(graph, `ring-${++instance}`, 'float32', 30);
  expect(create({ringCurvature: ringView()}).requiredHalo).toBe(8);
  expect(create({ringCurvature: ringView(), ringRadii: [1, 2]}).requiredHalo).toBe(2);
  expect(
    create({method: 'florinsky', ringCurvature: ringView(), ringRadii: [1]}).requiredHalo
  ).toBe(2);
  expect(create({method: 'florinsky', ringRadii: [9]}).requiredHalo).toBe(2);
  device.destroy();
});

it('GPUTerrainCurvature validates its properties', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const view = (length: number = 30) =>
    createTransientView(graph, `view-${++instance}`, 'float32', length);
  const create = (overrides: Partial<GPUTerrainCurvatureProps> = {}) =>
    new GPUTerrainCurvature({
      width: 6,
      height: 5,
      elevation: createBand(graph, `elevation-${++instance}`, 30),
      settings: view(12),
      curvatures: {mean: view()},
      ...overrides
    });

  expect(() => create({curvatures: {}})).toThrow(/at least one output/);
  expect(() => create({curvatures: undefined})).toThrow(/at least one output/);
  expect(() => create({curvatures: {mean: view(29)}})).toThrow(/one value per pixel/);
  expect(() => create({curvatures: {bogus: view()} as never})).toThrow(/unknown curvature kind/);
  expect(() => create({ringCurvature: view(29)})).toThrow(/one value per pixel/);
  expect(() => create({settings: view(11)})).toThrow(/settings/);
  expect(() => create({method: 'horn' as never})).toThrow(/method/);
  expect(() => create({cellSizeMode: 'polar' as never})).toThrow(/cellSizeMode/);
  expect(() => create({rowDirection: 'up' as never})).toThrow(/rowDirection/);
  expect(() => create({borderMode: 'reflect' as never})).toThrow(/borderMode/);
  expect(() => create({ringSquash: 'tanh' as never})).toThrow(/ringSquash/);
  expect(() => create({ringRadii: []})).toThrow(/ringRadii/);
  expect(() => create({ringRadii: [1, 2, 3, 4, 5]})).toThrow(/ringRadii/);
  expect(() => create({ringRadii: [4, 2]})).toThrow(/ringRadii/);
  expect(() => create({ringRadii: [2, 2]})).toThrow(/ringRadii/);
  expect(() => create({ringRadii: [1.5]})).toThrow(/ringRadii/);
  expect(() => create({ringRadii: [0]})).toThrow(/ringRadii/);
  expect(() => create({width: 0})).toThrow(/dimensions/);

  const shared = createBand(graph, 'shared-elevation', 30);
  expect(() => create({elevation: shared, curvatures: {mean: shared.storage.values}})).toThrow(
    /share buffers/
  );
  const duplicate = view();
  expect(() => create({curvatures: {mean: duplicate, plan: duplicate}})).toThrow(/share buffers/);

  const otherGraph = new GPUCommandGraph(device);
  expect(() =>
    new GPUTerrainCurvature({
      width: 2,
      height: 2,
      elevation: createBand(otherGraph, 'foreign', 4),
      settings: createTransientView(otherGraph, 'foreign-settings', 'float32', 12),
      curvatures: {mean: createTransientView(otherGraph, 'foreign-mean', 'float32', 4)}
    }).getCommandNodes(graph)
  ).toThrow(/belong to the target graph/);
  device.destroy();
});

it('GPUTerrainCurvature reports kernels that exceed the storage binding limit', () => {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxStorageBuffersPerShaderStage: 6}
  });
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUTerrainCurvature({
    width: 6,
    height: 5,
    elevation: createBand(graph, 'elevation', 30),
    settings: createTransientView(graph, 'settings', 'float32', 12),
    curvatures: Object.fromEntries(
      ORACLE_CURVATURE_KINDS.slice(0, 6).map(kind => [
        kind,
        createTransientView(graph, kind, 'float32', 30)
      ])
    )
  });
  expect(() => contributor.getCommandNodes(graph)).toThrow(/storage buffers/);
  device.destroy();
});
