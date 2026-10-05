// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  generateDotsOnCPU,
  getGPUDotDensityParameterValues,
  GPUDotDensity,
  GPURandomPointsInPolygon,
  type GPUDotDensityProps
} from '../../../src/map-graphs/dot-density';
import {createNullWebGPUDevice} from '../map-graph-test-utils';
import {createPolygonColumns, isInSceneHole, SCENE_FEATURES} from './dot-density-scene';

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
    polygonPositions: view('float32x2', 20),
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
        count: view('uint32', 1),
        overflow: view('uint32', 1),
        totalCount: view('uint32', 1)
      },
      categories: view('uint32', 100),
      failedCount: view('uint32', 1),
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
  const recipe = new GPUDotDensity(createProps(graph, {id: 'dots'}));
  expect(recipe.recipe).toBe('dot-density');
  expect(recipe.featureCount).toBe(3);
  const ids = recipe.getCommandNodes(graph).map(node => node.id);
  expect(ids[0]).toBe('dots-features');
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
  expect(points.recipe).toBe('random-points-in-polygon');
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

it('generateDotsOnCPU counts are monotone in dotsPerUnit and positions form a stable prefix', () => {
  const polygons = createPolygonColumns(SCENE_FEATURES);
  const values = new Float32Array([37.3, 12.6, 8.2, 0, Number.NaN, -4]);
  const run = (dotsPerUnit: number) =>
    generateDotsOnCPU({
      ...polygons,
      values,
      categoryCount: 2,
      parameters: getGPUDotDensityParameterValues({seed: 9, dotsPerUnit}),
      capacity: 10000,
      maximumAttempts: 32
    });
  const coarse = run(1);
  const fine = run(4);
  for (let slot = 0; slot < 6; slot++) {
    expect(fine.slotCounts[slot]).toBeGreaterThanOrEqual(coarse.slotCounts[slot]);
    for (let rank = 0; rank < coarse.slotCounts[slot]; rank++) {
      const coarseDot = coarse.slotOffsets[slot] + rank;
      const fineDot = fine.slotOffsets[slot] + rank;
      expect(fine.positions[2 * fineDot]).toBe(coarse.positions[2 * coarseDot]);
      expect(fine.positions[2 * fineDot + 1]).toBe(coarse.positions[2 * coarseDot + 1]);
    }
  }
  // floor(x) plus a Bernoulli remainder; slots without a positive value draw nothing.
  expect(Math.abs(coarse.slotCounts[0] - 37.3)).toBeLessThan(1);
  expect(Array.from(coarse.slotCounts.slice(3))).toEqual([0, 0, 0]);
  // Holes are excluded and every dot is in its feature's bounding box.
  for (let dot = 0; dot < fine.count; dot++) {
    const x = fine.positions[2 * dot];
    const y = fine.positions[2 * dot + 1];
    if (Number.isNaN(x)) {
      continue;
    }
    expect(isInSceneHole(x, y)).toBe(false);
    if (fine.featureIds[dot] === 1) {
      expect(x).toBeGreaterThanOrEqual(20);
    }
  }
  // The expected count over many seeds is the fractional value.
  let total = 0;
  for (let seed = 0; seed < 400; seed++) {
    total += generateDotsOnCPU({
      ...polygons,
      values: new Float32Array([0.3, 0, 0]),
      parameters: getGPUDotDensityParameterValues({seed}),
      capacity: 10,
      maximumAttempts: 1
    }).totalCount;
  }
  expect(total / 400).toBeGreaterThan(0.24);
  expect(total / 400).toBeLessThan(0.36);
});

it('generateDotsOnCPU honours the mask and reports failures and clamping', () => {
  const polygons = createPolygonColumns(SCENE_FEATURES);
  // 10 x 10 mask over [0, 10]^2: weight 1 for x >= 5 only.
  const weights = new Float32Array(100);
  for (let row = 0; row < 10; row++) {
    weights.fill(1, row * 10 + 5, row * 10 + 10);
  }
  const masked = generateDotsOnCPU({
    ...polygons,
    counts: new Uint32Array([200, 0, 0]),
    parameters: getGPUDotDensityParameterValues({
      seed: 3,
      maskExtent: [0, 0, 1, 1]
    }),
    capacity: 1000,
    maximumAttempts: 64,
    mask: {weights, width: 10, height: 10}
  });
  expect(masked.failedCount).toBe(0);
  for (let dot = 0; dot < masked.count; dot++) {
    expect(masked.positions[2 * dot]).toBeGreaterThanOrEqual(5);
  }
  const sliver = generateDotsOnCPU({
    ...polygons,
    counts: new Uint32Array([0, 0, 50]),
    parameters: getGPUDotDensityParameterValues({seed: 3}),
    capacity: 1000,
    maximumAttempts: 2
  });
  expect(sliver.failedCount).toBeGreaterThan(40);
  const clamped = generateDotsOnCPU({
    ...polygons,
    counts: new Uint32Array([5, 3, 2]),
    parameters: getGPUDotDensityParameterValues({seed: 3}),
    capacity: 4,
    maximumAttempts: 2
  });
  expect(clamped.count).toBe(4);
  expect(clamped.overflow).toBe(1);
  expect(Array.from(clamped.slotCounts)).toEqual([5, 3, 2]);
  const huge = generateDotsOnCPU({
    ...polygons,
    values: new Float32Array([1e30, 1, 1]),
    parameters: getGPUDotDensityParameterValues({seed: 3}),
    capacity: 4,
    maximumAttempts: 1
  });
  expect(huge.slotCounts[0]).toBe(5);
  expect(huge.overflow).toBe(1);
});
