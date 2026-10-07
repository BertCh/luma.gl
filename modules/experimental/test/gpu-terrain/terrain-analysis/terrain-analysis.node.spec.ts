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
  GPUTerrainCumulativeViewshed,
  GPUTerrainLineOfSight,
  GPU_TERRAIN_VISIBILITY,
  getGPUTerrainSightLineParameterValues,
  getGPUTerrainVisibilityToleranceParameterValues,
  getGPUTerrainDerivativesParameterValues,
  getGPUTerrainViewshedParameterValues,
  type GPUTerrainDerivativesProps
} from '../../../src/gpu-terrain/terrain-analysis';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createBand} from '../terrain-test-utils';

/** Collapses the consecutive pyramid builder node ids into one '<pyramid>' marker. */
function withoutPyramidNodes(ids: string[]): string[] {
  return ids.reduce<string[]>((result, id) => {
    const isPyramid = /-pyramid-level/.test(id);
    if (!isPyramid || result.at(-1) !== '<pyramid>') {
      result.push(isPyramid ? '<pyramid>' : id);
    }
    return result;
  }, []);
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
  const contributor = create();
  expect(contributor.requiredHalo).toBe(1);
  expect(contributor.id).toBe('terrain-derivatives');
  expect(contributor.getCommandNodes(graph).map(node => node.id)).toEqual([
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
  const contributor = new GPUTerrainContours({
    width: 5,
    height: 5,
    elevation: createBand(graph, 'elevation', 25),
    levels,
    overflow
  });
  expect(contributor.requiredHalo).toBe(1);
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
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
  const drawNodes = (contributor: GPUTerrainContours) =>
    contributor
      .getCommandNodes(graph)
      .map(node => node.id)
      .filter(id => id.endsWith('-draw'));

  // Default two-vertex instanced record is written by the raster publish pass.
  expect(drawNodes(create())).toEqual([]);
  expect(drawNodes(create({verticesPerInstance: 2}))).toEqual([]);
  // Other shapes add one contributor node after the level pipeline.
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

it('GPUTerrainViewshed pyramid and tolerance variants schedule nodes within 8 bindings', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Record<string, unknown> = {}) => {
    const prefix = `v${instance++}`;
    return new GPUTerrainViewshed({
      id: prefix,
      width: 9,
      height: 7,
      elevation: createBand(graph, `${prefix}-elevation`, 63),
      settings: createTransientView(graph, `${prefix}-settings`, 'float32', 8),
      visibility: createTransientView(graph, `${prefix}-visibility`, 'uint32', 63),
      ...overrides
    });
  };
  expect(GPU_TERRAIN_VISIBILITY.marginal).toBe(4);
  const pyramid = create({traversal: 'pyramid'});
  const pyramidNodes = pyramid.getCommandNodes(graph);
  // The pyramid builder may fuse its coarse levels into fewer nodes; only its presence matters.
  expect(withoutPyramidNodes(pyramidNodes.map(node => node.id))).toEqual([
    `${pyramid.id}-elevation`,
    '<pyramid>',
    `${pyramid.id}-viewshed`
  ]);
  // elevation values, validity, pyramid, settings, visibility
  expect(pyramidNodes.at(-1)!.resources).toHaveLength(5);
  const tolerant = create({
    traversal: 'pyramid',
    tolerance: createTransientView(graph, 'tolerance-pyramid', 'float32', 4)
  });
  const tolerantNodes = tolerant.getCommandNodes(graph);
  expect(tolerantNodes.every(node => (node.resources ?? []).length <= 8)).toBe(true);
  expect(tolerantNodes.at(-1)!.resources).toHaveLength(6);
  const marchTolerant = create({
    tolerance: createTransientView(graph, 'tolerance-march', 'float32', 4)
  });
  expect(marchTolerant.getCommandNodes(graph).map(node => node.id)).toEqual([
    `${marchTolerant.id}-elevation`,
    `${marchTolerant.id}-viewshed`
  ]);
  expect(() => create({traversal: 'octree'})).toThrow(/traversal/);
  expect(() =>
    create({tolerance: createTransientView(graph, 'tolerance-short', 'float32', 3)})
  ).toThrow(/settings/);
  expect(
    Array.from(
      getGPUTerrainVisibilityToleranceParameterValues({
        toleranceMeters: 2,
        tolerancePerKilometer: 1,
        targetIgnoreDistance: 150,
        targetIgnoreFraction: 0.02
      })
    )
  ).toEqual(Array.from(Float32Array.from([2, 1, 150, 0.02])));
  expect(() => getGPUTerrainVisibilityToleranceParameterValues({toleranceMeters: -1})).toThrow(
    /non-negative/
  );
  device.destroy();
});

it('GPUTerrainLineOfSight validates and schedules within 8 bindings', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Record<string, unknown> = {}) => {
    const prefix = `los${instance++}`;
    return new GPUTerrainLineOfSight({
      id: prefix,
      width: 9,
      height: 7,
      elevation: createBand(graph, `${prefix}-elevation`, 63),
      pairs: createTransientView(graph, `${prefix}-pairs`, 'float32x4', 5),
      settings: createTransientView(graph, `${prefix}-settings`, 'float32', 12),
      visibility: createTransientView(graph, `${prefix}-visibility`, 'uint32', 5),
      ...overrides
    });
  };
  const march = create();
  expect(march.getCommandNodes(graph).map(node => node.id)).toEqual([
    `${march.id}-elevation`,
    `${march.id}-line-of-sight`
  ]);
  const full = create({
    traversal: 'pyramid',
    pairHeights: createTransientView(graph, 'los-heights', 'float32x2', 5),
    clearance: createTransientView(graph, 'los-clearance', 'float32', 5)
  });
  const fullNodes = full.getCommandNodes(graph);
  expect(withoutPyramidNodes(fullNodes.map(node => node.id))).toEqual([
    `${full.id}-elevation`,
    '<pyramid>',
    `${full.id}-line-of-sight`
  ]);
  // values, validity, pyramid, settings, pairs, pairHeights, visibility, clearance
  expect(fullNodes.at(-1)!.resources).toHaveLength(8);
  expect(() => create({visibility: createTransientView(graph, 'los-v4', 'uint32', 4)})).toThrow(
    /one value per pair/
  );
  expect(() => create({pairHeights: createTransientView(graph, 'los-h4', 'float32x2', 4)})).toThrow(
    /one row per pair/
  );
  expect(() => create({clearance: createTransientView(graph, 'los-c4', 'float32', 4)})).toThrow(
    /one value per pair/
  );
  expect(() => create({settings: createTransientView(graph, 'los-s11', 'float32', 11)})).toThrow(
    /at least 12/
  );
  expect(() => create({pairs: createTransientView(graph, 'los-p2', 'float32x2', 5)})).toThrow(
    /pairs/
  );
  expect(() => create({traversal: 'bvh'})).toThrow(/traversal/);
  expect(
    Array.from(
      getGPUTerrainSightLineParameterValues({
        cellSize: [30, 20],
        maxDistance: 500,
        curvatureCoefficient: 1e-7,
        toleranceMeters: 2,
        tolerancePerKilometer: 1,
        targetIgnoreDistance: 150,
        targetIgnoreFraction: 0.02
      })
    )
  ).toEqual(Array.from(Float32Array.from([1.7, 0, 500, 30, 20, 1e-7, 2, 1, 150, 0.02, 0, 0])));
  expect(() => getGPUTerrainSightLineParameterValues({cellSize: [1, 0]})).toThrow(/cell size/);
  expect(() =>
    getGPUTerrainSightLineParameterValues({cellSize: [1, 1], toleranceMeters: -1})
  ).toThrow(/non-negative/);
  device.destroy();
});

it('GPUTerrainCumulativeViewshed batches observers within 8 bindings', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Record<string, unknown> = {}) => {
    const prefix = `cum${instance++}`;
    return new GPUTerrainCumulativeViewshed({
      id: prefix,
      width: 9,
      height: 7,
      elevation: createBand(graph, `${prefix}-elevation`, 63),
      observers: createTransientView(graph, `${prefix}-observers`, 'float32x2', 5),
      settings: createTransientView(graph, `${prefix}-settings`, 'float32', 12),
      visibleCount: createTransientView(graph, `${prefix}-visible`, 'uint32', 63),
      ...overrides
    });
  };
  const batched = create({observersPerDispatch: 2});
  expect(batched.getCommandNodes(graph).map(node => node.id)).toEqual([
    `${batched.id}-elevation`,
    `${batched.id}-visible-count-zero`,
    `${batched.id}-batch-0`,
    `${batched.id}-batch-1`,
    `${batched.id}-batch-2`
  ]);
  // The default packs every observer of a small grid into one dispatch.
  expect(create().observersPerDispatch).toBe((2 ** 22 / 63) | 0);
  const full = create({
    traversal: 'pyramid',
    observerHeights: createTransientView(graph, 'cum-heights', 'float32', 5),
    marginalCount: createTransientView(graph, 'cum-marginal', 'uint32', 63)
  });
  const fullNodes = full.getCommandNodes(graph);
  expect(fullNodes.map(node => node.id)).toContain(`${full.id}-marginal-count-zero`);
  // values, validity, pyramid, settings, observers, observerHeights, visibleCount, marginalCount
  expect(fullNodes.at(-1)!.resources).toHaveLength(8);
  expect(() => create({visibleCount: createTransientView(graph, 'cum-v62', 'uint32', 62)})).toThrow(
    /one value per pixel/
  );
  expect(() =>
    create({observerHeights: createTransientView(graph, 'cum-h4', 'float32', 4)})
  ).toThrow(/one value per observer/);
  expect(() => create({observersPerDispatch: 0})).toThrow(/positive integer/);
  expect(() => create({traversal: 'bvh'})).toThrow(/traversal/);
  device.destroy();
});
