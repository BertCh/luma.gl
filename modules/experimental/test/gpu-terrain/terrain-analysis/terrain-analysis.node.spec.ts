// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, DrawCommandBuffer, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {
  getGPUTerrainCurvatureCoefficient,
  GPUTerrainContours,
  GPUTerrainDerivatives,
  GPUTerrainViewshed,
  GPUTerrainCumulativeViewshed,
  GPUTerrainLineOfSight,
  GPU_TERRAIN_VISIBILITY,
  getGPUTerrainLineOfSightParameterValues,
  getGPUTerrainViewshedToleranceParameterValues,
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
      grid: makeGPURasterMetadata({width: 6, height: 5}),
      elevation: createBand(graph, `elevation-${instance}`, 30),
      parameters: createTransientView(graph, `settings-${instance}`, 'float32', 8),
      output: {slope: createTransientView(graph, `slope-${instance}`, 'float32', 30)},
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
  expect(
    Array.from(
      getGPUTerrainDerivativesParameterValues(
        {},
        makeGPURasterMetadata({width: 6, height: 5, cellSize: [2, 3]})
      )
    )
  ).toEqual([2, 3, 1, 315, 45, 0, -15, 0]);

  expect(() => create({output: {}})).toThrow(/at least one output/);
  expect(() =>
    create({output: {slope: createTransientView(graph, 'short-slope', 'float32', 29)}})
  ).toThrow(/one value per pixel/);
  expect(() => create({parameters: createTransientView(graph, 'settings7', 'float32', 7)})).toThrow(
    /parameters/
  );
  const shared = createBand(graph, 'shared-elevation', 30);
  expect(() => create({elevation: shared, output: {slope: shared.storage.values}})).toThrow(
    /share buffers/
  );
  expect(() => create({cellSizeMode: 'polar' as never})).toThrow(/cellSizeMode/);
  expect(() => create({grid: makeGPURasterMetadata({width: 0, height: 5})})).toThrow(/dimensions/);
  const otherGraph = new GPUCommandGraph(device);
  expect(() =>
    new GPUTerrainDerivatives({
      grid: makeGPURasterMetadata({width: 2, height: 2}),
      elevation: createBand(otherGraph, 'foreign', 4),
      parameters: createTransientView(otherGraph, 'foreign-settings', 'float32', 8),
      output: {slope: createTransientView(otherGraph, 'foreign-slope', 'float32', 4)}
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
    status: {count: createTransientView(graph, `${name}-count`, 'uint32', 1)}
  });
  const levels = [level('a'), level('b')];
  const overflow = createTransientView(graph, 'overflow', 'uint32', 1);
  const contributor = new GPUTerrainContours({
    grid: makeGPURasterMetadata({width: 5, height: 5}),
    elevation: createBand(graph, 'elevation', 25),
    levels,
    output: {status: {overflow}}
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
        grid: makeGPURasterMetadata({width: 5, height: 5}),
        elevation: createBand(graph, 'e2', 25),
        levels: [],
        output: {status: {overflow}}
      })
  ).toThrow(/at least one contour level/);
  expect(
    () =>
      new GPUTerrainContours({
        grid: makeGPURasterMetadata({width: 5, height: 5}),
        elevation: createBand(graph, 'e3', 25),
        levels: [levels[0]],
        output: {status: {overflow: levels[0].status.count}}
      })
  ).toThrow(/share buffers/);
  expect(
    () =>
      new GPUTerrainContours({
        grid: makeGPURasterMetadata({width: 0, height: 5}),
        elevation: createBand(graph, 'e4', 25),
        levels,
        output: {status: {overflow}}
      })
  ).toThrow(/dimensions/);
  device.destroy();
});

it('GPUTerrainViewshed schedules canonical elevation and the line-of-sight kernel', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const grid = makeGPURasterMetadata({width: 9, height: 7, cellSize: 30});
  const props = {
    grid,
    elevation: createBand(graph, 'elevation', 63),
    parameters: createTransientView(graph, 'settings', 'float32', 10),
    output: {visibility: createTransientView(graph, 'visibility', 'uint32', 63)}
  };
  expect(new GPUTerrainViewshed(props).getCommandNodes(graph).map(node => node.id)).toEqual([
    'terrain-viewshed-elevation',
    'terrain-viewshed-viewshed'
  ]);
  expect(Array.from(getGPUTerrainViewshedParameterValues({observer: [1, 2]}, grid))).toEqual(
    Array.from(Float32Array.from([1, 2, 1.7, 0, 0, 30, 30, 0, 0, -210]))
  );
  expect(getGPUTerrainCurvatureCoefficient()).toBeCloseTo(6.8277e-8, 11);
  expect(() => new GPUTerrainViewshed({...props, output: {}})).toThrow(/at least one output/);
  expect(
    () =>
      new GPUTerrainViewshed({
        ...props,
        output: {visibility: createTransientView(graph, 'v62', 'uint32', 62)}
      })
  ).toThrow(/one value per pixel/);
  expect(
    () =>
      new GPUTerrainViewshed({
        ...props,
        parameters: createTransientView(graph, 's7', 'float32', 7)
      })
  ).toThrow(/parameters/);
  expect(() =>
    getGPUTerrainViewshedParameterValues(
      {observer: [0, 0]},
      makeGPURasterMetadata({width: 9, height: 7, cellSize: [0, 1]})
    )
  ).toThrow(/cell size/);
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
      grid: makeGPURasterMetadata({width: 5, height: 5}),
      elevation: createBand(graph, `elevation-${instance}`, 25),
      levels: [
        {
          level: 3.5,
          vertices: createTransientView(graph, `vertices-${instance}`, 'float32x2', 64),
          status: {count: createTransientView(graph, `count-${instance}`, 'uint32', 1)},
          draw,
          ...levelOverrides
        }
      ],
      output: {
        status: {overflow: createTransientView(graph, `overflow-${instance}`, 'uint32', 1)}
      }
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
      grid: makeGPURasterMetadata({width: 9, height: 7}),
      elevation: createBand(graph, `${prefix}-elevation`, 63),
      parameters: createTransientView(graph, `${prefix}-settings`, 'float32', 10),
      output: {visibility: createTransientView(graph, `${prefix}-visibility`, 'uint32', 63)},
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
    toleranceParameters: createTransientView(graph, 'tolerance-pyramid', 'float32', 4)
  });
  const tolerantNodes = tolerant.getCommandNodes(graph);
  expect(tolerantNodes.every(node => (node.resources ?? []).length <= 8)).toBe(true);
  expect(tolerantNodes.at(-1)!.resources).toHaveLength(6);
  const marchTolerant = create({
    toleranceParameters: createTransientView(graph, 'tolerance-march', 'float32', 4)
  });
  expect(marchTolerant.getCommandNodes(graph).map(node => node.id)).toEqual([
    `${marchTolerant.id}-elevation`,
    `${marchTolerant.id}-viewshed`
  ]);
  expect(() => create({traversal: 'octree'})).toThrow(/traversal/);
  expect(() =>
    create({toleranceParameters: createTransientView(graph, 'tolerance-short', 'float32', 3)})
  ).toThrow(/toleranceParameters/);
  expect(
    Array.from(
      getGPUTerrainViewshedToleranceParameterValues({
        toleranceMeters: 2,
        tolerancePerKilometer: 1,
        targetIgnoreDistance: 150,
        targetIgnoreFraction: 0.02
      })
    )
  ).toEqual(Array.from(Float32Array.from([2, 1, 150, 0.02])));
  expect(() => getGPUTerrainViewshedToleranceParameterValues({toleranceMeters: -1})).toThrow(
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
      grid: makeGPURasterMetadata({width: 9, height: 7}),
      elevation: createBand(graph, `${prefix}-elevation`, 63),
      pairs: createTransientView(graph, `${prefix}-pairs`, 'float32x4', 5),
      parameters: createTransientView(graph, `${prefix}-settings`, 'float32', 12),
      output: {visibility: createTransientView(graph, `${prefix}-visibility`, 'uint32', 5)},
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
    output: {
      visibility: createTransientView(graph, 'los-full-visibility', 'uint32', 5),
      clearance: createTransientView(graph, 'los-clearance', 'float32', 5)
    }
  });
  const fullNodes = full.getCommandNodes(graph);
  expect(withoutPyramidNodes(fullNodes.map(node => node.id))).toEqual([
    `${full.id}-elevation`,
    '<pyramid>',
    `${full.id}-line-of-sight`
  ]);
  // values, validity, pyramid, settings, pairs, pairHeights, visibility, clearance
  expect(fullNodes.at(-1)!.resources).toHaveLength(8);
  expect(() =>
    create({output: {visibility: createTransientView(graph, 'los-v4', 'uint32', 4)}})
  ).toThrow(/one value per pair/);
  expect(() => create({pairHeights: createTransientView(graph, 'los-h4', 'float32x2', 4)})).toThrow(
    /one row per pair/
  );
  expect(() =>
    create({
      output: {
        visibility: createTransientView(graph, 'los-cv5', 'uint32', 5),
        clearance: createTransientView(graph, 'los-c4', 'float32', 4)
      }
    })
  ).toThrow(/one value per pair/);
  expect(() => create({parameters: createTransientView(graph, 'los-s11', 'float32', 11)})).toThrow(
    /at least 12/
  );
  expect(() => create({pairs: createTransientView(graph, 'los-p2', 'float32x2', 5)})).toThrow(
    /pairs/
  );
  expect(() => create({traversal: 'bvh'})).toThrow(/traversal/);
  const grid = makeGPURasterMetadata({width: 9, height: 7, cellSize: [30, 20]});
  expect(
    Array.from(
      getGPUTerrainLineOfSightParameterValues(
        {
          maximumDistance: 500,
          curvatureCoefficient: 1e-7,
          toleranceMeters: 2,
          tolerancePerKilometer: 1,
          targetIgnoreDistance: 150,
          targetIgnoreFraction: 0.02
        },
        grid
      )
    )
  ).toEqual(Array.from(Float32Array.from([1.7, 0, 500, 30, 20, 1e-7, 2, 1, 150, 0.02, 0, -140])));
  expect(() =>
    getGPUTerrainLineOfSightParameterValues(
      {},
      makeGPURasterMetadata({width: 9, height: 7, cellSize: [1, 0]})
    )
  ).toThrow(/cell size/);
  expect(() =>
    getGPUTerrainLineOfSightParameterValues(
      {toleranceMeters: -1},
      makeGPURasterMetadata({width: 9, height: 7})
    )
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
      grid: makeGPURasterMetadata({width: 9, height: 7}),
      elevation: createBand(graph, `${prefix}-elevation`, 63),
      observers: createTransientView(graph, `${prefix}-observers`, 'float32x2', 5),
      parameters: createTransientView(graph, `${prefix}-settings`, 'float32', 12),
      output: {visibleCount: createTransientView(graph, `${prefix}-visible`, 'uint32', 63)},
      ...overrides
    });
  };
  const batched = create({observersPerDispatch: 2});
  expect(batched.getCommandNodes(graph).map(node => node.id)).toEqual([
    `${batched.id}-elevation`,
    `${batched.id}-visible-count-zero`,
    `${batched.id}-observer-records`,
    `${batched.id}-batch-table`,
    `${batched.id}-batch-0`,
    `${batched.id}-batch-1`,
    `${batched.id}-batch-2`
  ]);
  // The default packs every observer of a small grid into one dispatch: 2^28 / 9 samples per
  // dispatch is more than the 5 * 63 (observer, pixel) pairs.
  expect(create().invocationsPerDispatch).toBe(5 * 63);
  expect(create().observersPerDispatch).toBe(5);
  const full = create({
    traversal: 'pyramid',
    observerHeights: createTransientView(graph, 'cum-heights', 'float32', 5),
    output: {
      visibleCount: createTransientView(graph, 'cum-full-visible', 'uint32', 63),
      marginalCount: createTransientView(graph, 'cum-marginal', 'uint32', 63)
    }
  });
  const fullNodes = full.getCommandNodes(graph);
  expect(fullNodes.map(node => node.id)).toContain(`${full.id}-marginal-count-zero`);
  // values, validity, pyramid, settings, observer records, visibleCount, marginalCount, batch word
  expect(fullNodes.at(-1)!.resources).toHaveLength(8);
  expect(() =>
    create({output: {visibleCount: createTransientView(graph, 'cum-v62', 'uint32', 62)}})
  ).toThrow(/one value per pixel/);
  expect(() =>
    create({observerHeights: createTransientView(graph, 'cum-h4', 'float32', 4)})
  ).toThrow(/one value per observer/);
  expect(() => create({observersPerDispatch: 0})).toThrow(/positive integer/);
  expect(() => create({traversal: 'bvh'})).toThrow(/traversal/);
  device.destroy();
});
