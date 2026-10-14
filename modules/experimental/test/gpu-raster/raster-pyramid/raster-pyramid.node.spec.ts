// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {
  getGPURasterExtremaPyramidLayout,
  GPURasterExtremaPyramid,
  type GPURasterExtremaPyramidProps
} from '../../../src/gpu-raster/raster-pyramid';
import {
  createRasterExtremaPyramidNodes,
  getRasterExtremaPyramidWGSL
} from '../../../src/gpu-raster/raster-pyramid/gpu-raster-extrema-pyramid';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  computeRasterExtremaPyramid,
  computeRasterExtremaPyramidDirect
} from './raster-pyramid-oracle';

it('getGPURasterExtremaPyramidLayout computes levels, widths, offsets and length', () => {
  const layout = getGPURasterExtremaPyramidLayout(makeGPURasterMetadata({width: 37, height: 23}));
  expect(
    layout.levels.map(level => [level.blockSize, level.width, level.height, level.offset])
  ).toEqual([
    [4, 10, 6, 0],
    [8, 5, 3, 60],
    [16, 3, 2, 75],
    [32, 2, 1, 81],
    [64, 1, 1, 83]
  ]);
  expect(layout.length).toBe(84);
  expect(layout.footprint).toBe('bilinear');

  const small = getGPURasterExtremaPyramidLayout(makeGPURasterMetadata({width: 1, height: 1}), {
    firstBlockSize: 1
  });
  expect(small.levels).toEqual([{level: 0, blockSize: 1, width: 1, height: 1, offset: 0}]);
  expect(small.length).toBe(1);

  const capped = getGPURasterExtremaPyramidLayout(makeGPURasterMetadata({width: 64, height: 64}), {
    firstBlockSize: 1,
    levelCount: 3
  });
  expect(capped.levels.map(level => level.width)).toEqual([64, 32, 16]);
  expect(capped.length).toBe(64 * 64 + 32 * 32 + 16 * 16);
  expect(
    getGPURasterExtremaPyramidLayout(makeGPURasterMetadata({width: 8, height: 8}), {levelCount: 99})
      .levels.length
  ).toBe(2);
  expect(
    getGPURasterExtremaPyramidLayout(makeGPURasterMetadata({width: 64, height: 64}), {
      firstBlockSize: 1
    }).levels.length
  ).toBe(7);

  expect(() =>
    getGPURasterExtremaPyramidLayout(makeGPURasterMetadata({width: 0, height: 4}))
  ).toThrow(/dimensions/);
  for (const firstBlockSize of [0, 3, 512, 1.5]) {
    expect(() =>
      getGPURasterExtremaPyramidLayout(makeGPURasterMetadata({width: 8, height: 8}), {
        firstBlockSize
      })
    ).toThrow(/firstBlockSize/);
  }
  for (const levelCount of [0, 1.5, -1, Number.NaN]) {
    expect(() =>
      getGPURasterExtremaPyramidLayout(makeGPURasterMetadata({width: 8, height: 8}), {levelCount})
    ).toThrow(/levelCount/);
  }
  expect(() =>
    getGPURasterExtremaPyramidLayout(makeGPURasterMetadata({width: 8, height: 8}), {
      footprint: 'x' as never
    })
  ).toThrow(/footprint/);
});

it('GPURasterExtremaPyramid schedules level 0 plus one fused node per four levels with at most 8 bindings', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const layout = getGPURasterExtremaPyramidLayout(makeGPURasterMetadata({width: 37, height: 23}));
  let instance = 0;
  const create = (overrides: Partial<GPURasterExtremaPyramidProps> = {}) => {
    instance++;
    return new GPURasterExtremaPyramid({
      grid: makeGPURasterMetadata({width: 37, height: 23}),
      input: {
        id: `input-${instance}`,
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: createTransientView(graph, `in-${instance}`, 'float32', 851)
        }
      },
      output: {
        maximum: createTransientView(graph, `max-${instance}`, 'float32', layout.length),
        minimum: createTransientView(graph, `min-${instance}`, 'float32', layout.length)
      },
      ...overrides
    });
  };
  const contributor = create();
  expect(contributor.layout.length).toBe(84);
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  // Five levels: level 0 scans pixels, levels 1 to 4 share one workgroup-tile dispatch.
  expect(ids.filter(id => id.startsWith('raster-extrema-pyramid-level'))).toEqual([
    'raster-extrema-pyramid-level-0',
    'raster-extrema-pyramid-levels-1-4'
  ]);
  expect(ids.some(id => id.startsWith('raster-extrema-pyramid-elevation'))).toBe(true);
  expect(
    create({
      id: 'p',
      firstBlockSize: 8,
      output: {
        maximum: createTransientView(
          graph,
          'p-max',
          'float32',
          getGPURasterExtremaPyramidLayout(makeGPURasterMetadata({width: 37, height: 23}), {
            firstBlockSize: 8
          }).length
        )
      }
    })
      .getCommandNodes(graph)
      .filter(node => node.id.startsWith('p-level')).length
  ).toBe(2);

  expect(() => create({output: {}})).toThrow(/at least one output/);
  expect(() =>
    create({output: {maximum: createTransientView(graph, 'short', 'float32', layout.length - 1)}})
  ).toThrow(/must contain 84 values/);
  expect(() => create({firstBlockSize: 3})).toThrow(/firstBlockSize/);
  expect(() => create({levelCount: 0})).toThrow(/levelCount/);
  const shared = createTransientView(graph, 'shared', 'float32', layout.length);
  expect(() => create({output: {maximum: shared, minimum: shared}})).toThrow(/share buffers/);

  // combined: one binding for both extrema; every variant is within the 8 binding limit.
  const values = createTransientView(graph, 'values', 'float32', 851);
  const validity = createTransientView(graph, 'validity', 'uint32', 851);
  const combined = createTransientView(graph, 'combined', 'float32', 2 * layout.length);
  const combinedNodes = createRasterExtremaPyramidNodes(graph, {
    id: 'c',
    layout,
    values,
    validity,
    combined
  });
  expect(combinedNodes.map(node => node.id)).toEqual(['c-level-0', 'c-levels-1-4']);
  expect(combinedNodes.every(node => (node.resources ?? []).length <= 3)).toBe(true);
  expect(() =>
    createRasterExtremaPyramidNodes(graph, {
      id: 'c2',
      layout,
      values,
      validity,
      combined: createTransientView(graph, 'combined-short', 'float32', layout.length)
    })
  ).toThrow(/combined must contain 168 values/);
  expect(() =>
    createRasterExtremaPyramidNodes(graph, {
      id: 'c3',
      layout,
      values,
      validity,
      combined,
      maximum: shared
    })
  ).toThrow(/combined/);
  const both = createRasterExtremaPyramidNodes(graph, {
    id: 'b',
    layout,
    values,
    validity,
    maximum: shared,
    minimum: createTransientView(graph, 'b-min', 'float32', layout.length)
  });
  expect(Math.max(...both.map(node => (node.resources ?? []).length))).toBeLessThanOrEqual(8);
  device.destroy();
});

it('getRasterExtremaPyramidWGSL bakes per-level constants and prefixed names', () => {
  const layout = getGPURasterExtremaPyramidLayout(makeGPURasterMetadata({width: 37, height: 23}));
  const wgsl = getRasterExtremaPyramidWGSL(layout, 'pointHorizon');
  expect(wgsl).toContain('const POINT_HORIZON_LEVEL_COUNT: u32 = 5u;');
  expect(wgsl).toContain('const POINT_HORIZON_MINIMUM_OFFSET: u32 = 84u;');
  expect(wgsl).toContain('array<u32, 5>(4u, 8u, 16u, 32u, 64u)');
  expect(wgsl).toContain('array<u32, 5>(0u, 60u, 75u, 81u, 83u)');
  expect(wgsl).toContain('fn pointHorizonLevelIndex(level: u32, column: u32, row: u32) -> u32');
  expect(wgsl).toContain('fn pointHorizonLevelBlockSize(level: u32) -> u32');
  expect(getRasterExtremaPyramidWGSL(layout)).toContain(
    'const PYRAMID_EMPTY_MAXIMUM: f32 = -3.4028234663852886e+38;'
  );
});

it('the hierarchical oracle equals the direct definition', () => {
  let seed = 12345;
  const random = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 0x100000000;
  };
  for (const [width, height] of [
    [37, 23],
    [64, 64],
    [1, 1],
    [5, 130]
  ]) {
    const values = Float32Array.from({length: width * height}, () => (random() - 0.5) * 1000);
    const validity = Uint32Array.from({length: width * height}, () => (random() < 0.3 ? 0 : 1));
    for (const firstBlockSize of [1, 2, 4]) {
      for (const footprint of ['cell', 'bilinear'] as const) {
        const layout = getGPURasterExtremaPyramidLayout(
          makeGPURasterMetadata({width: width, height: height}),
          {firstBlockSize, footprint}
        );
        const hierarchical = computeRasterExtremaPyramid(values, validity, layout);
        const direct = computeRasterExtremaPyramidDirect(values, validity, layout);
        for (let i = 0; i < layout.length; i++) {
          expect(hierarchical.maximum[i] === direct.maximum[i]).toBe(true);
          expect(hierarchical.minimum[i] === direct.minimum[i]).toBe(true);
        }
      }
    }
  }
});

it('GPURasterExtremaPyramid fuses levels in groups of four', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  // 64x64 at block 1: levels 0..6 -> level 0, levels 1-4, levels 5-6.
  const layout = getGPURasterExtremaPyramidLayout(makeGPURasterMetadata({width: 64, height: 64}), {
    firstBlockSize: 1
  });
  expect(layout.levels.length).toBe(7);
  const nodes = createRasterExtremaPyramidNodes(graph, {
    id: 'f',
    layout,
    values: createTransientView(graph, 'f-values', 'float32', 4096),
    validity: createTransientView(graph, 'f-validity', 'uint32', 4096),
    combined: createTransientView(graph, 'f-combined', 'float32', 2 * layout.length)
  });
  expect(nodes.map(node => node.id)).toEqual(['f-level-0', 'f-levels-1-4', 'f-levels-5-6']);
});

it('GPURasterExtremaPyramid skips the identity copy for canonical float32 buffer bands', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const layout = getGPURasterExtremaPyramidLayout(makeGPURasterMetadata({width: 37, height: 23}));
  const create = (id: string, withValidity: boolean, scale?: number) =>
    new GPURasterExtremaPyramid({
      id,
      grid: makeGPURasterMetadata({width: 37, height: 23}),
      input: {
        id: `${id}-input`,
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: createTransientView(graph, `${id}-in`, 'float32', 851)
        },
        ...(withValidity
          ? {validity: createTransientView(graph, `${id}-valid`, 'uint32', 851)}
          : {}),
        ...(scale === undefined ? {} : {scale})
      },
      output: {maximum: createTransientView(graph, `${id}-max`, 'float32', layout.length)}
    })
      .getCommandNodes(graph)
      .map(node => node.id);
  const countCopies = (ids: string[]) => ids.filter(id => id.includes('-elevation')).length;
  const canonical = create('a', true);
  expect(countCopies(canonical)).toBe(0);
  expect(canonical).toEqual(['a-level-0', 'a-levels-1-4']);
  expect(countCopies(create('b', false))).toBe(1);
  expect(countCopies(create('c', true, 2))).toBe(1);
  device.destroy();
});
