// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUEmergingHotSpotParameterValues,
  GPUEmergingHotSpots,
  GPU_EMERGING_HOT_SPOT_CATEGORIES,
  GPU_EMERGING_HOT_SPOT_PARAMETER_LENGTH,
  type GPUEmergingHotSpotsProps
} from '../../../src/gpu-spatial-analysis/emerging-hot-spots';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  classifyEmergingHotSpotSeries,
  computeEmergingHotSpotCells,
  computeEmergingHotSpotMoments,
  computeMannKendall,
  computeSpaceTimeGiStar,
  createDesignedCube,
  createDesignedSeries
} from './emerging-hot-spots-oracle';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUEmergingHotSpotsProps> = {}
): GPUEmergingHotSpotsProps {
  const view = <Format extends 'uint32' | 'float32' | 'sint32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${serial++}`, format, length);
  // 4 x 3 lattice, 5 slices.
  return {
    values: view('float32', 60),
    gridWidth: 4,
    gridHeight: 3,
    sliceCount: 5,
    parameters: view('float32', GPU_EMERGING_HOT_SPOT_PARAMETER_LENGTH),
    giZScores: view('float32', 60),
    trendZ: view('float32', 12),
    trendP: view('float32', 12),
    trendS: view('sint32', 12),
    category: view('uint32', 12),
    hotSliceCount: view('uint32', 12),
    coldSliceCount: view('uint32', 12),
    ...overrides
  };
}

function expectThrows(
  overrides: (graph: GPUCommandGraph) => Partial<GPUEmergingHotSpotsProps>,
  message: RegExp
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUEmergingHotSpots(createProps(graph, overrides(graph)))).toThrow(message);
  device.destroy();
}

it('getGPUEmergingHotSpotParameterValues packs and validates the layout', () => {
  expect(Array.from(getGPUEmergingHotSpotParameterValues({radius: 2, temporalWindow: 3}))).toEqual([
    2,
    3,
    Math.fround(1.6448536269514722),
    Math.fround(0.1),
    Math.fround(0.9),
    0,
    0,
    0
  ]);
  const strict = getGPUEmergingHotSpotParameterValues({
    radius: 1.5,
    temporalWindow: 0,
    confidenceLevel: 0.99,
    persistentFraction: 0.8
  });
  expect(strict[2]).toBeCloseTo(2.5758293, 6);
  expect(strict[3]).toBeCloseTo(0.01, 6);
  expect(strict[4]).toBeCloseTo(0.8, 6);
  expect(
    getGPUEmergingHotSpotParameterValues({
      radius: 0,
      temporalWindow: 1,
      criticalZ: 3
    })[2]
  ).toBe(3);
  const bad = (parameters: Parameters<typeof getGPUEmergingHotSpotParameterValues>[0]) =>
    expect(() => getGPUEmergingHotSpotParameterValues(parameters));
  bad({radius: -1, temporalWindow: 0}).toThrow(/radius/);
  bad({radius: NaN, temporalWindow: 0}).toThrow(/radius/);
  bad({radius: 1, temporalWindow: 1.5}).toThrow(/temporalWindow/);
  bad({radius: 1, temporalWindow: -1}).toThrow(/temporalWindow/);
  bad({radius: 1, temporalWindow: 0, criticalZ: 0}).toThrow(/criticalZ/);
  bad({radius: 1, temporalWindow: 0, trendSignificanceLevel: 1}).toThrow(/trendSignificanceLevel/);
  bad({radius: 1, temporalWindow: 0, persistentFraction: 0.5}).toThrow(/persistentFraction/);
  bad({radius: 1, temporalWindow: 0, confidenceLevel: 0.8 as 0.9}).toThrow(/confidenceLevel/);
  expect(() =>
    getGPUEmergingHotSpotParameterValues({radius: 1, temporalWindow: 0}, new Float32Array(2))
  ).toThrow(/target/);
});

it('GPUEmergingHotSpots validates its inputs', () => {
  expectThrows(() => ({gridWidth: 0}), /gridWidth/);
  expectThrows(() => ({gridHeight: undefined}), /gridHeight/);
  expectThrows(() => ({gridWidth: undefined, gridHeight: undefined}), /exactly one/);
  expectThrows(
    graph => ({
      weights: {
        offsets: createTransientView(graph, 'o', 'uint32', 13),
        neighbors: createTransientView(graph, 'n', 'uint32', 20),
        weights: createTransientView(graph, 'w', 'float32', 20)
      }
    }),
    /exactly one/
  );
  expectThrows(
    graph => ({
      gridWidth: undefined,
      gridHeight: undefined,
      selfWeight: -1,
      weights: {
        offsets: createTransientView(graph, 'o', 'uint32', 13),
        neighbors: createTransientView(graph, 'n', 'uint32', 20),
        weights: createTransientView(graph, 'w', 'float32', 20)
      }
    }),
    /selfWeight/
  );
  expectThrows(
    graph => ({
      gridWidth: undefined,
      gridHeight: undefined,
      weights: {
        offsets: createTransientView(graph, 'o', 'uint32', 11),
        neighbors: createTransientView(graph, 'n', 'uint32', 20),
        weights: createTransientView(graph, 'w', 'float32', 20)
      }
    }),
    /values length/
  );
  expectThrows(() => ({sliceCount: 1.5}), /sliceCount/);
  expectThrows(() => ({sliceCount: 257}), /sliceCount must be at most 256/);
  expectThrows(() => ({maximumRadius: 33}), /maximumRadius/);
  expectThrows(() => ({maximumRadius: 1.5}), /maximumRadius/);
  expectThrows(
    graph => ({values: createTransientView(graph, 'short', 'float32', 59)}),
    /values length/
  );
  expectThrows(
    graph => ({values: createTransientView(graph, 'bad', 'float32x2', 60)}) as never,
    /values/
  );
  expectThrows(
    graph => ({parameters: createTransientView(graph, 'p', 'float32', 7)}),
    /parameters/
  );
  expectThrows(graph => ({mask: createTransientView(graph, 'm', 'uint32', 11)}), /mask length/);
  expectThrows(
    graph => ({giZScores: createTransientView(graph, 'z', 'float32', 12)}),
    /giZScores length/
  );
  expectThrows(
    graph => ({category: createTransientView(graph, 'c', 'uint32', 60)}),
    /category length/
  );
  expectThrows(
    graph => ({
      globalStatistics: createTransientView(graph, 'g', 'float32', 3)
    }),
    /globalStatistics/
  );
  expectThrows(graph => {
    const values = createTransientView(graph, 'alias', 'float32', 60);
    return {values, giZScores: values};
  }, /must not share buffers/);
});

it('GPUEmergingHotSpots creates deterministic nodes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const mask = createTransientView(graph, 'mask', 'uint32', 12);
  const contributor = new GPUEmergingHotSpots(createProps(graph, {id: 'eh', mask}));
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  expect(ids).toEqual([
    'eh-block-sums',
    'eh-mean',
    'eh-block-squares',
    'eh-deviation',
    'eh-gi-star',
    // Mann-Kendall and the category classification share one pass over each cell's series.
    'eh-mann-kendall'
  ]);
  const secondGraph = new GPUCommandGraph(device);
  const again = new GPUEmergingHotSpots(createProps(secondGraph, {id: 'eh'}));
  expect(again.getCommandNodes(secondGraph).map(node => node.id)).toEqual(ids);
  // Views from another graph are rejected.
  const foreign = new GPUEmergingHotSpots(createProps(secondGraph, {id: 'foreign'}));
  expect(() => foreign.getCommandNodes(graph)).toThrow(/must belong to the target graph/);
  // uint32 counts are accepted.
  const countsGraph = new GPUCommandGraph(device);
  const counts = new GPUEmergingHotSpots(
    createProps(countsGraph, {
      values: createTransientView(countsGraph, 'counts', 'uint32', 60)
    })
  );
  expect(counts.getCommandNodes(countsGraph)).toHaveLength(6);
  device.destroy();
});

it('the Mann-Kendall oracle matches hand-computed statistics', () => {
  // Monotone series of 10: S = 45, Var = 10 * 9 * 25 / 18 = 125.
  const monotone = computeMannKendall(Array.from({length: 10}, (_, index) => index));
  expect(monotone.statistic).toBe(45);
  expect(monotone.variance).toBe(125);
  expect(monotone.z).toBeCloseTo(44 / Math.sqrt(125), 12);
  expect(monotone.p).toBeCloseTo(8.1e-5, 5);
  // Ties: [1, 2, 2, 3] has S = 5 and one tie group of size 2: (4 * 3 * 13 - 18) / 18 = 138 / 18.
  const tied = computeMannKendall([1, 2, NaN, 2, 3]);
  expect(tied.count).toBe(4);
  expect(tied.statistic).toBe(5);
  expect(tied.tieTerm).toBe(18);
  expect(tied.variance).toBeCloseTo(138 / 18, 12);
  expect(tied.z).toBeCloseTo(4 / Math.sqrt(138 / 18), 12);
  // Descending, constant and degenerate series.
  expect(computeMannKendall([5, 4, 3]).statistic).toBe(-3);
  expect(computeMannKendall([5, 4, 3]).z).toBeLessThan(0);
  const constant = computeMannKendall([2, 2, 2, 2]);
  expect([constant.statistic, constant.z, constant.p]).toEqual([0, 0, 1]);
  expect(computeMannKendall([NaN, NaN])).toMatchObject({
    count: 0,
    statistic: 0,
    z: 0,
    p: 1
  });
});

it('the classifier oracle reaches all 17 categories on designed z series', () => {
  const packed = getGPUEmergingHotSpotParameterValues({
    radius: 0,
    temporalWindow: 0
  });
  const hot = 3;
  const cold = -3;
  const series = (...segments: number[][]) => segments.flat();
  const ramp = (from: number, to: number, length = 20) =>
    Array.from({length}, (_, index) => from + ((to - from) * index) / (length - 1));
  const rep = (count: number, value: number) => new Array<number>(count).fill(value);
  const classify = (values: number[]) =>
    classifyEmergingHotSpotSeries(values, computeMannKendall(values), packed);
  const categories = GPU_EMERGING_HOT_SPOT_CATEGORIES;
  const cases: [string, number[], number][] = [
    ['new hot', series(rep(9, 0), [hot]), categories.NEW_HOT],
    ['consecutive hot', series(rep(7, 0), rep(3, hot)), categories.CONSECUTIVE_HOT],
    ['intensifying hot', ramp(2, 4), categories.INTENSIFYING_HOT],
    ['persistent hot', rep(20, hot), categories.PERSISTENT_HOT],
    ['diminishing hot', ramp(4, 2), categories.DIMINISHING_HOT],
    ['sporadic hot', series([hot], rep(3, 0), [hot], rep(5, 0), [hot, 0]), categories.SPORADIC_HOT],
    ['sporadic hot final hot', series([hot, 0, hot, 0, hot]), categories.SPORADIC_HOT],
    ['oscillating hot', series([cold, 0, hot, 0, hot, hot]), categories.OSCILLATING_HOT],
    ['historical hot', series(rep(19, hot), [0]), categories.HISTORICAL_HOT],
    ['new cold', series(rep(9, 0), [cold]), categories.NEW_COLD],
    ['consecutive cold', series(rep(7, 0), rep(3, cold)), categories.CONSECUTIVE_COLD],
    ['intensifying cold', ramp(-2, -4), categories.INTENSIFYING_COLD],
    ['persistent cold', rep(20, cold), categories.PERSISTENT_COLD],
    ['diminishing cold', ramp(-4, -2), categories.DIMINISHING_COLD],
    ['sporadic cold', series([cold, 0, 0, cold, 0]), categories.SPORADIC_COLD],
    ['oscillating cold', series([hot, 0, cold, 0, cold, cold]), categories.OSCILLATING_COLD],
    ['historical cold', series(rep(19, cold), [0]), categories.HISTORICAL_COLD],
    ['no pattern', rep(10, 0), categories.NO_PATTERN],
    ['mixed history, quiet end', series([hot, 0, cold, 0]), categories.NO_PATTERN],
    ['all missing', [NaN, NaN, NaN], categories.NO_PATTERN]
  ];
  const reached = new Set<number>();
  for (const [name, values, expected] of cases) {
    expect(classify(values), name).toBe(expected);
    reached.add(expected);
  }
  expect(reached.size).toBe(17);
  expect(Object.values(categories).sort((a, b) => a - b)).toEqual(
    Array.from({length: 17}, (_, i) => i)
  );
  // Missing slices are skipped: the final finite slice decides.
  expect(classify([0, 0, hot, NaN])).toBe(categories.NEW_HOT);
  // Hot 90% boundary: 9 of 10 is persistent, 8 of 10 is not.
  expect(classify(series(rep(1, 0), rep(9, hot)))).toBe(categories.PERSISTENT_HOT);
  expect(classify(series(rep(2, 0), rep(8, hot)))).toBe(categories.CONSECUTIVE_HOT);
});

it('the designed cube is classified as designed by the double precision oracle', () => {
  const {cube, designed, expectedCategories} = createDesignedCube();
  expect(designed.map(entry => entry.expected).sort((a, b) => a - b)).toEqual(
    expect.arrayContaining(Array.from({length: 17}, (_, index) => index))
  );
  const packed = getGPUEmergingHotSpotParameterValues({
    radius: 0,
    temporalWindow: 0
  });
  const zScores = computeSpaceTimeGiStar(cube, packed);
  const cells = computeEmergingHotSpotCells(zScores, cube, packed);
  for (const [cell, expected] of expectedCategories.entries()) {
    expect(cells.category[cell], `cell ${cell} ${designed[cell]?.name}`).toBe(expected);
  }
  // Radius 0 and window 0: z is the plain standard score of each valid bin.
  const moments = computeEmergingHotSpotMoments(cube);
  expect(zScores[0]).toBeCloseTo((cube.values[0] - moments.mean) / moments.standardDeviation, 12);
  // Masked and all-missing cells are excluded.
  expect(Number.isNaN(zScores[designedLength(designed) * 20])).toBe(true);
  expect(cells.category[designedLength(designed)]).toBe(0);
  expect(cells.category[designedLength(designed) + 1]).toBe(0);
  expect(createDesignedSeries()).toHaveLength(designed.length);
});

function designedLength(designed: unknown[]): number {
  return designed.length;
}

it('GPUEmergingHotSpots weights mode creates the same pipeline over weights bindings', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `weights-mode-${serial++}`, format, length);
  const base = createProps(graph, {gridWidth: undefined, gridHeight: undefined});
  const contributor = new GPUEmergingHotSpots({
    ...base,
    weights: {
      offsets: view('uint32', 13),
      neighbors: view('uint32', 30),
      weights: view('float32', 30)
    }
  });
  const ids = contributor.getCommandNodes(graph).map(node => node.id);
  expect(ids).toContain('emerging-hot-spots-gi-star');
  expect(new Set(ids).size).toBe(ids.length);
  device.destroy();
});
