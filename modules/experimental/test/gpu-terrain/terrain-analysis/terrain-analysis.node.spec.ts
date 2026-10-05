// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, DrawCommandBuffer, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUTerrainCurvatureCoefficient,
  GPUTerrainContours,
  GPUTerrainDerivatives,
  GPUTerrainViewshed,
  getGPUTerrainDerivativesParameterValues,
  getGPUTerrainViewshedParameterValues,
  type GPUTerrainDerivativesProps
} from '../../../src/gpu-terrain/terrain-analysis';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createBand(graph: GPUCommandGraph, id: string, length: number) {
  return {
    id,
    format: 'float32' as const,
    storage: {kind: 'buffer' as const, values: createTransientView(graph, id, 'float32', length)}
  };
}

it('GPUTerrainDerivatives schedules gradients and the shade kernel', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Partial<GPUTerrainDerivativesProps> = {}) => {
    instance++;
    return new GPUTerrainDerivatives({
      width: 6,
      height: 5,
      elevation: createBand(graph, `elevation-${instance}`, 30),
      settings: createTransientView(graph, `settings-${instance}`, 'float32', 8),
      slope: createTransientView(graph, `slope-${instance}`, 'float32', 30),
      ...overrides
    });
  };
  const recipe = create();
  expect(recipe.requiredHalo).toBe(1);
  expect(recipe.recipe).toBe('terrain-derivatives');
  expect(recipe.getCommandNodes(graph).map(node => node.id)).toEqual([
    'terrain-derivatives-gradient-x',
    'terrain-derivatives-gradient-y',
    'terrain-derivatives-shade'
  ]);
  expect(
    create({id: 'hill'})
      .getCommandNodes(graph)
      .every(node => node.id.startsWith('hill-'))
  ).toBe(true);
  expect(Array.from(getGPUTerrainDerivativesParameterValues({cellSize: [2, 3]}))).toEqual([
    2, 3, 1, 315, 45, 0, 0, 0
  ]);

  expect(() => create({slope: undefined})).toThrow(/at least one output/);
  expect(() => create({slope: createTransientView(graph, 'short-slope', 'float32', 29)})).toThrow(
    /one value per pixel/
  );
  expect(() => create({settings: createTransientView(graph, 'settings7', 'float32', 7)})).toThrow(
    /settings/
  );
  const shared = createBand(graph, 'shared-elevation', 30);
  expect(() => create({elevation: shared, slope: shared.storage.values})).toThrow(/share buffers/);
  expect(() => create({cellSizeMode: 'polar' as never})).toThrow(/cellSizeMode/);
  expect(() => create({width: 0})).toThrow(/dimensions/);
  const otherGraph = new GPUCommandGraph(device);
  expect(() =>
    new GPUTerrainDerivatives({
      width: 2,
      height: 2,
      elevation: createBand(otherGraph, 'foreign', 4),
      settings: createTransientView(otherGraph, 'foreign-settings', 'float32', 8),
      slope: createTransientView(otherGraph, 'foreign-slope', 'float32', 4)
    }).getCommandNodes(graph)
  ).toThrow(/belong to the target graph/);
  device.destroy();
});

it('GPUTerrainContours schedules one contour pipeline and overflow OR per level', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const level = (name: string) => ({
    level: 3.5,
    vertices: createTransientView(graph, `${name}-vertices`, 'float32x2', 64),
    segmentCount: createTransientView(graph, `${name}-count`, 'uint32', 1)
  });
  const levels = [level('a'), level('b')];
  const overflow = createTransientView(graph, 'overflow', 'uint32', 1);
  const recipe = new GPUTerrainContours({
    width: 5,
    height: 5,
    elevation: createBand(graph, 'elevation', 25),
    levels,
    overflow
  });
  expect(recipe.requiredHalo).toBe(1);
  const ids = recipe.getCommandNodes(graph).map(node => node.id);
  expect(ids[0]).toBe('terrain-contours-level-0-classify');
  for (const id of [
    'terrain-contours-level-0-scatter',
    'terrain-contours-level-0-publish',
    'terrain-contours-overflow-0',
    'terrain-contours-level-1-classify',
    'terrain-contours-overflow-1'
  ]) {
    expect(ids).toContain(id);
  }
  expect(ids.indexOf('terrain-contours-overflow-0')).toBeLessThan(
    ids.indexOf('terrain-contours-level-1-classify')
  );
  expect(
    () =>
      new GPUTerrainContours({
        width: 5,
        height: 5,
        elevation: createBand(graph, 'e2', 25),
        levels: [],
        overflow
      })
  ).toThrow(/at least one contour level/);
  expect(
    () =>
      new GPUTerrainContours({
        width: 5,
        height: 5,
        elevation: createBand(graph, 'e3', 25),
        levels: [levels[0]],
        overflow: levels[0].segmentCount
      })
  ).toThrow(/share buffers/);
  expect(
    () =>
      new GPUTerrainContours({
        width: 0,
        height: 5,
        elevation: createBand(graph, 'e4', 25),
        levels,
        overflow
      })
  ).toThrow(/dimensions/);
  device.destroy();
});

it('GPUTerrainViewshed schedules canonical elevation and the line-of-sight kernel', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = {
    width: 9,
    height: 7,
    elevation: createBand(graph, 'elevation', 63),
    settings: createTransientView(graph, 'settings', 'float32', 8),
    visibility: createTransientView(graph, 'visibility', 'uint32', 63)
  };
  expect(new GPUTerrainViewshed(props).getCommandNodes(graph).map(node => node.id)).toEqual([
    'terrain-viewshed-elevation',
    'terrain-viewshed-viewshed'
  ]);
  expect(
    Array.from(getGPUTerrainViewshedParameterValues({observer: [1, 2], cellSize: [30, 30]}))
  ).toEqual(Array.from(Float32Array.from([1, 2, 1.7, 0, 0, 30, 30, 0])));
  expect(getGPUTerrainCurvatureCoefficient()).toBeCloseTo(6.8277e-8, 11);
  expect(() => new GPUTerrainViewshed({...props, visibility: undefined})).toThrow(
    /at least one output/
  );
  expect(
    () =>
      new GPUTerrainViewshed({
        ...props,
        visibility: createTransientView(graph, 'v62', 'uint32', 62)
      })
  ).toThrow(/one value per pixel/);
  expect(
    () =>
      new GPUTerrainViewshed({...props, settings: createTransientView(graph, 's7', 'float32', 7)})
  ).toThrow(/settings/);
  expect(() => getGPUTerrainViewshedParameterValues({observer: [0, 0], cellSize: [0, 1]})).toThrow(
    /cell size/
  );
  device.destroy();
});

it('GPUTerrainContours validates and schedules indirect draw layouts', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const commands = new DrawCommandBuffer(device, {id: 'draw', type: 'draw', capacity: 2});
  const draw = commands.importToGraph(graph);
  let instance = 0;
  const create = (levelOverrides: Record<string, unknown> = {}) => {
    instance++;
    return new GPUTerrainContours({
      id: `contours-${instance}`,
      width: 5,
      height: 5,
      elevation: createBand(graph, `elevation-${instance}`, 25),
      levels: [
        {
          level: 3.5,
          vertices: createTransientView(graph, `vertices-${instance}`, 'float32x2', 64),
          segmentCount: createTransientView(graph, `count-${instance}`, 'uint32', 1),
          draw,
          ...levelOverrides
        }
      ],
      overflow: createTransientView(graph, `overflow-${instance}`, 'uint32', 1)
    });
  };
  const drawNodes = (recipe: GPUTerrainContours) =>
    recipe
      .getCommandNodes(graph)
      .map(node => node.id)
      .filter(id => id.endsWith('-draw'));

  // Default two-vertex instanced record is written by the raster publish pass.
  expect(drawNodes(create())).toEqual([]);
  expect(drawNodes(create({verticesPerInstance: 2}))).toEqual([]);
  // Other shapes add one recipe node after the level pipeline.
  expect(drawNodes(create({verticesPerInstance: 6, drawCommandIndex: 1}))).toHaveLength(1);
  expect(drawNodes(create({drawLayout: 'line-list'}))).toHaveLength(1);

  expect(() => create({drawLayout: 'line-list', verticesPerInstance: 2})).toThrow(
    /verticesPerInstance requires drawLayout/
  );
  expect(() => create({verticesPerInstance: 0})).toThrow(/positive integer/);
  expect(() => create({verticesPerInstance: 2.5})).toThrow(/positive integer/);
  expect(() => create({drawLayout: 'triangles'})).toThrow(/drawLayout must be/);
  expect(() => create({drawCommandIndex: 2})).toThrow(/exceeds the indirect draw capacity/);
  expect(() => create({drawCommandIndex: -1})).toThrow(/non-negative integer/);
  expect(() => create({draw: undefined, drawLayout: 'line-list'})).toThrow(/require draw/);
  expect(() => create({draw: undefined, verticesPerInstance: 4})).toThrow(/require draw/);
  expect(() => create({draw: {...draw, type: 'draw-indexed'}})).toThrow(/non-indexed/);
  expect(() => create({draw: {...draw, recordByteLength: 20}})).toThrow(/non-indexed/);
  expect(() => create({draw: {...draw, words: draw.instanceCounts}})).toThrow(/draw words/);
  commands.destroy();
  device.destroy();
});
