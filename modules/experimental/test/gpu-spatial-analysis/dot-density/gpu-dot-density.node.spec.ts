// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  getGPUDotDensityParameterValues,
  GPUDotDensity,
  GPURandomPointsInPolygon,
  type GPUDotDensityProps
} from '../../../src/gpu-spatial-analysis/dot-density';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {generateDotCountsOnCPU} from './dot-density-cpu';
import {createPolygonColumns, SCENE_FEATURES} from './dot-density-scene';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUDotDensityProps> = {}
): GPUDotDensityProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  return {
    positions: view('float32x2', 20),
    featureOffsets: view('uint32', 4),
    polygonOffsets: view('uint32', 5),
    ringOffsets: view('uint32', 6),
    values: view('float32', 6),
    categoryCount: 2,
    parameters: view('uint32', 8),
    output: {
      positions: view('float32x2', 100),
      dots: {
        ids: view('uint32', 100),
        status: {
          count: view('uint32', 1),
          overflow: view('uint32', 1),
          requiredCount: view('uint32', 1)
        }
      },
      categories: view('uint32', 100),
      status: {failedCount: view('uint32', 1)},
      slotCounts: view('uint32', 6),
      slotOffsets: view('uint32', 6)
    },
    ...overrides
  };
}

it('GPUDotDensity validates props and builds deterministic nodes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {
    id: 'dot-validation'
  });
  const contributor = new GPUDotDensity(createProps(graph, {id: 'dots'}));
  expect(contributor.featureCount).toBe(3);
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  expect(ids[0]).toBe('dots-triangulation-earcut');
  expect(ids).toContain('dots-feature-triangles');
  expect(ids).toContain('dots-sample');
  expect(ids[ids.length - 1]).toBe('dots-publish');
  expect(() => new GPUDotDensity(createProps(graph, {categoryCount: 3}))).toThrow(
    /featureCount \* categoryCount/
  );
  expect(() => new GPUDotDensity(createProps(graph, {maximumAttempts: 0}))).toThrow(
    /maximumAttempts/
  );
  const props = createProps(graph);
  expect(
    () =>
      new GPUDotDensity({
        ...props,
        output: {
          ...props.output,
          categories: createTransientView(graph, 'c-short', 'uint32', 9)
        }
      })
  ).toThrow(/categories/);
  expect(
    () =>
      new GPUDotDensity({
        ...props,
        mask: {
          weights: createTransientView(graph, 'mask', 'float32', 5),
          width: 2,
          height: 2
        }
      })
  ).toThrow(/mask/);
  expect(
    () =>
      new GPUDotDensity({
        ...props,
        output: {...props.output, positions: props.values as never}
      })
  ).toThrow();
  const points = new GPURandomPointsInPolygon({
    ...props,
    output: {...props.output, slotCounts: undefined, slotOffsets: undefined},
    counts: createTransientView(graph, 'counts', 'uint32', 3)
  });
  expect(points.id).toBe('random-points-in-polygon');
  expect(
    () =>
      new GPURandomPointsInPolygon({
        ...props,
        counts: createTransientView(graph, 'counts-short', 'uint32', 2)
      })
  ).toThrow(/one row per feature/);
});

it('getGPUDotDensityParameterValues packs float bits next to the seed', () => {
  const words = getGPUDotDensityParameterValues({
    seed: 42,
    dotsPerUnit: 0.25,
    maskExtent: [1, 2, 3, 4]
  });
  const floats = new Float32Array(words.buffer);
  expect(words[0]).toBe(42);
  expect(Array.from(floats.slice(1, 6))).toEqual([0.25, 1, 2, 3, 4]);
  expect(() => getGPUDotDensityParameterValues({seed: 1.5})).toThrow(/seed/);
});

it('generateDotCountsOnCPU counts are monotone in dotsPerUnit', () => {
  const polygons = createPolygonColumns(SCENE_FEATURES);
  const values = new Float32Array([37.3, 12.6, 8.2, 0, Number.NaN, -4]);
  const run = (dotsPerUnit: number) =>
    generateDotCountsOnCPU({
      ...polygons,
      values,
      categoryCount: 2,
      parameters: getGPUDotDensityParameterValues({seed: 9, dotsPerUnit}),
      capacity: 10000
    });
  const coarse = run(1);
  const fine = run(4);
  for (let slot = 0; slot < 6; slot++) {
    expect(fine.slotCounts[slot]).toBeGreaterThanOrEqual(coarse.slotCounts[slot]);
  }
  // floor(x) plus a Bernoulli remainder; slots without a positive value draw nothing.
  expect(Math.abs(coarse.slotCounts[0] - 37.3)).toBeLessThan(1);
  expect(Array.from(coarse.slotCounts.slice(3))).toEqual([0, 0, 0]);
  // The expected count over many seeds is the fractional value.
  let total = 0;
  for (let seed = 0; seed < 400; seed++) {
    total += generateDotCountsOnCPU({
      ...polygons,
      values: new Float32Array([0.3, 0, 0]),
      parameters: getGPUDotDensityParameterValues({seed}),
      capacity: 10
    }).requiredCount;
  }
  expect(total / 400).toBeGreaterThan(0.24);
  expect(total / 400).toBeLessThan(0.36);
});

it('generateDotCountsOnCPU reports clamping', () => {
  const polygons = createPolygonColumns(SCENE_FEATURES);
  const clamped = generateDotCountsOnCPU({
    ...polygons,
    counts: new Uint32Array([5, 3, 2]),
    parameters: getGPUDotDensityParameterValues({seed: 3}),
    capacity: 4
  });
  expect(clamped.count).toBe(4);
  expect(clamped.overflow).toBe(1);
  expect(Array.from(clamped.slotCounts)).toEqual([5, 3, 2]);
  const huge = generateDotCountsOnCPU({
    ...polygons,
    values: new Float32Array([1e30, 1, 1]),
    parameters: getGPUDotDensityParameterValues({seed: 3}),
    capacity: 4
  });
  expect(huge.slotCounts[0]).toBe(5);
  expect(huge.overflow).toBe(1);
});

it('dot sampling kernels parse, stay within 8 storage bindings and bake no loop index', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'dot-kernels'});
  const mask = {
    weights: createTransientView(graph, 'mask-weights', 'float32', 4),
    width: 2,
    height: 2
  };
  const buildSources = (maximumAttempts: number, masked: boolean, categories: boolean) => {
    const props = createProps(graph, {id: `dots-${serial++}-${masked}-${categories}`});
    const contributor = new GPUDotDensity({
      ...props,
      maximumAttempts,
      mask: masked ? mask : undefined,
      output: {...props.output, categories: categories ? props.output.categories : undefined}
    });
    return getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
  };
  const sources = [
    ...buildSources(32, false, true),
    ...buildSources(32, true, true),
    ...buildSources(32, true, false)
  ];
  const variants = new Set(
    sources.map(({id}) => id.replace(/^dots-\d+-(true|false)-(true|false)-/, ''))
  );
  for (const name of [
    'triangle-areas',
    'feature-triangles',
    'cumulative-areas',
    'counts',
    'total',
    'locate',
    'sample'
  ]) {
    expect(variants.has(name), name).toBe(true);
  }
  for (const {id, source} of sources) {
    const reflection = new WgslReflect(source);
    expect(reflection.storage.length, id).toBeLessThanOrEqual(8);
    // No local NaN helper copy (rule 8): the shared isNanBits is used.
    expect(source, id).not.toMatch(/fn isNanFloat/);
  }
  const sample = sources.filter(({id}) => id.endsWith('-sample'));
  expect(sample.length).toBe(3);
  // Rule 12: the attempt limit depends on the configuration, never on a loop index, so changing
  // only maximumAttempts changes only the masked sampler and rebuilding is deterministic.
  const sourcesAt64 = buildSources(64, true, true).filter(({id}) => id.endsWith('-sample'));
  const sourcesAt32 = buildSources(32, true, true).filter(({id}) => id.endsWith('-sample'));
  expect(sourcesAt64[0].source).not.toBe(sourcesAt32[0].source);
  const unmaskedAt64 = buildSources(64, false, true).filter(({id}) => id.endsWith('-sample'));
  expect(unmaskedAt64[0].source).toBe(sample[0].source);
});

it('GPUDotDensity and GPURandomPointsInPolygon accept planar contexts and reject longitude-latitude', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const geographic = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;
  expect(new GPUDotDensity(createProps(graph)).spatialContext.coordinateSpace).toBe('planar');
  expect(() => new GPUDotDensity(createProps(graph, {spatialContext: geographic}))).toThrow(
    /project positions to a planar frame/
  );
  device.destroy();
});
