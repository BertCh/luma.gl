// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  getGPUDistanceFieldParameterValues,
  GPUDistanceField,
  GPU_DISTANCE_FIELD_NONE
} from '../../../src/gpu-raster/distance-field';
import {
  DISTANCE_FIELD_BAND_SIZE,
  getDistanceFieldBandLayout
} from '../../../src/gpu-raster/distance-field/distance-field-kernels';
import {makeGPURasterMetadata} from '../../../src/gpu-raster/raster-grid';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {getRasterPipelines} from '../raster-pipeline-count';
import {computeDistanceFieldOnCPU, createRandom, getCellSeeds} from './distance-field-oracle';
import {
  computeColumnsBanded,
  computeColumnsSequential,
  computeRowsBanded,
  computeRowsSequential,
  createDistanceFieldOrder,
  type DistanceFieldBandedRowStatistics
} from './distance-field-banded-model';

const NONE = GPU_DISTANCE_FIELD_NONE;

type Case = {
  name: string;
  width: number;
  height: number;
  cellSize: [number, number];
  seeds: Uint32Array;
};

/** Seed raster with `count` random seeds (IDs drawn from a small range, so ties are common). */
function createRandomSeeds(
  width: number,
  height: number,
  count: number,
  random: () => number,
  idRange: number
): Uint32Array {
  const seeds = new Uint32Array(width * height).fill(NONE);
  for (let seed = 0; seed < count; seed++) {
    const cell = Math.floor(random() * width * height);
    seeds[cell] = Math.min(seeds[cell], Math.floor(random() * idRange));
  }
  return seeds;
}

function createCases(): Case[] {
  const random = createRandom(20261014);
  const cases: Case[] = [];
  const isotropic: [number, number] = [1, 1];
  const fromCells = (width: number, height: number, cells: [number, number][]) => {
    const seeds = new Uint32Array(width * height).fill(NONE);
    cells.forEach(([column, row], id) => {
      seeds[row * width + column] = id;
    });
    return seeds;
  };
  cases.push({
    name: 'empty',
    width: 37,
    height: 11,
    cellSize: isotropic,
    seeds: new Uint32Array(37 * 11).fill(NONE)
  });
  cases.push({
    name: 'single seed',
    width: 97,
    height: 40,
    cellSize: isotropic,
    seeds: fromCells(97, 40, [[3, 5]])
  });
  cases.push({
    name: 'seeds only in the last band',
    width: 70,
    height: 70,
    cellSize: isotropic,
    seeds: fromCells(70, 70, [
      [68, 69],
      [65, 66],
      [69, 64]
    ])
  });
  // Mirror-symmetric seeds: every cell on the axis is equidistant and breaks on the smaller ID.
  const symmetric = new Uint32Array(41 * 9).fill(NONE);
  for (let row = 0; row < 9; row++) {
    symmetric[row * 41 + 4] = 9 - row;
    symmetric[row * 41 + 36] = row;
  }
  cases.push({
    name: 'equidistant ties',
    width: 41,
    height: 9,
    cellSize: isotropic,
    seeds: symmetric
  });
  cases.push({
    name: '1xN',
    width: 1,
    height: 77,
    cellSize: isotropic,
    seeds: createRandomSeeds(1, 77, 3, random, 4)
  });
  cases.push({
    name: 'Nx1',
    width: 131,
    height: 1,
    cellSize: isotropic,
    seeds: createRandomSeeds(131, 1, 4, random, 4)
  });
  for (let index = 0; index < 24; index++) {
    const width = 1 + Math.floor(random() * 150);
    const height = 1 + Math.floor(random() * 90);
    const density = [0.001, 0.01, 0.05, 0.3][index % 4];
    const count = Math.max(1, Math.round(width * height * density));
    const cellSize: [number, number] =
      index % 3 === 2 ? [1, 2.5] : index % 3 === 1 ? [0.3, 0.7] : [2, 2];
    cases.push({
      name: `random ${index} (${width}x${height}, ${cellSize.join('x')})`,
      width,
      height,
      cellSize,
      seeds: createRandomSeeds(width, height, count, random, index % 2 === 0 ? 3 : 1 << 20)
    });
  }
  return cases;
}

it('banded column scans equal the sequential column scans for every band size', () => {
  for (const scene of createCases()) {
    const expected = computeColumnsSequential(scene.width, scene.height, scene.seeds);
    for (const bandSize of [1, 2, 3, 7, DISTANCE_FIELD_BAND_SIZE]) {
      const actual = computeColumnsBanded(scene.width, scene.height, scene.seeds, bandSize);
      expect(Array.from(actual.columnRows), `${scene.name} band ${bandSize}`).toEqual(
        Array.from(expected.columnRows)
      );
      expect(Array.from(actual.columnIds), `${scene.name} band ${bandSize}`).toEqual(
        Array.from(expected.columnIds)
      );
    }
  }
});

it('banded envelopes with pairwise merges equal one sequential envelope per row', () => {
  let insertions = 0;
  let rounds = 0;
  for (const scene of createCases()) {
    const isCloser = createDistanceFieldOrder(...scene.cellSize);
    const candidates = computeColumnsSequential(scene.width, scene.height, scene.seeds);
    const expected = computeRowsSequential(scene.width, scene.height, candidates, isCloser);
    for (const bandSize of [1, 2, 3, 5, 8, DISTANCE_FIELD_BAND_SIZE]) {
      const statistics: DistanceFieldBandedRowStatistics = {mergeRounds: 0, mergeInsertions: 0};
      const actual = computeRowsBanded(
        scene.width,
        scene.height,
        candidates,
        isCloser,
        bandSize,
        statistics
      );
      insertions += statistics.mergeInsertions;
      rounds += statistics.mergeRounds;
      expect(Array.from(actual.nearestCells), `${scene.name} band ${bandSize}`).toEqual(
        Array.from(expected.nearestCells)
      );
      expect(Array.from(actual.nearestIds), `${scene.name} band ${bandSize}`).toEqual(
        Array.from(expected.nearestIds)
      );
    }
  }
  // The merges really ran (the cases are not all single-band).
  expect(rounds).toBeGreaterThan(100);
  expect(insertions).toBeGreaterThan(1000);
});

it('banded exact transform matches the brute-force oracle on isotropic grids', () => {
  for (const scene of createCases().filter(item => item.cellSize[0] === item.cellSize[1])) {
    const settings = getGPUDistanceFieldParameterValues(
      {},
      makeGPURasterMetadata({width: scene.width, height: scene.height, cellSize: scene.cellSize})
    );
    const mask = scene.seeds.map(seed => (seed === NONE ? 0 : seed + 1));
    const oracle = computeDistanceFieldOnCPU({
      width: scene.width,
      height: scene.height,
      settings,
      mask
    });
    const cellSeeds = getCellSeeds({width: scene.width, height: scene.height, settings, mask});
    const isCloser = createDistanceFieldOrder(...scene.cellSize);
    const candidates = computeColumnsBanded(scene.width, scene.height, cellSeeds, 4);
    const actual = computeRowsBanded(scene.width, scene.height, candidates, isCloser, 4);
    expect(Array.from(actual.nearestIds), scene.name).toEqual(Array.from(oracle.allocation));
    // Seeds sharing an ID at the same distance are interchangeable: compare the squared offsets.
    const getSquaredOffset = (cell: number, seedCell: number) =>
      seedCell === NONE
        ? -1
        : (Math.floor(cell / scene.width) - Math.floor(seedCell / scene.width)) ** 2 +
          ((cell % scene.width) - (seedCell % scene.width)) ** 2;
    expect(
      Array.from(actual.nearestCells, (seedCell, cell) => getSquaredOffset(cell, seedCell)),
      scene.name
    ).toEqual(
      Array.from(oracle.nearestCells, (seedCell, cell) => getSquaredOffset(cell, seedCell))
    );
  }
});

it('GPUDistanceField schedules the banded passes with one merge pipeline (rule 12)', () => {
  const device = createNullWebGPUDevice();
  const build = (width: number, height: number) => {
    const graph = new GPUCommandGraph(device);
    const nodes = new GPUDistanceField({
      id: 'df',
      grid: makeGPURasterMetadata({width, height}),
      parameters: createTransientView(graph, 'parameters', 'float32', 8),
      seedMask: createTransientView(graph, 'mask', 'uint32', width * height),
      output: {distances: createTransientView(graph, 'distances', 'float32', width * height)}
    }).getCommandNodes(graph);
    return nodes;
  };
  expect(build(20, 10).map(node => node.id)).toEqual([
    'df-seed-clear',
    'df-column-bands',
    'df-column-carry',
    'df-column-fill',
    'df-row-bands',
    'df-row-marker-clear',
    'df-row-start-markers',
    'df-row-marker-bands',
    'df-row-marker-carry',
    'df-row-color',
    'df-finalize'
  ]);
  // 300 columns: 10 bands, 4 merge rounds; 1100 columns: 35 bands, 6 merge rounds.
  expect(getDistanceFieldBandLayout({width: 300, height: 7}).mergeRoundCount).toBe(4);
  expect(getDistanceFieldBandLayout({width: 1100, height: 7}).mergeRoundCount).toBe(6);
  const mergeKeys = (width: number) => {
    const merges = getRasterPipelines(build(width, 7)).filter(pipeline =>
      /-row-merge-\d+$/.test(pipeline.id)
    );
    return {count: merges.length, keys: new Set(merges.map(pipeline => pipeline.key))};
  };
  const small = mergeKeys(300);
  const large = mergeKeys(1100);
  expect(small.count).toBe(4);
  expect(large.count).toBe(6);
  expect(small.keys.size).toBe(1);
  expect(large.keys.size).toBe(1);
  // The per-pass pipelines do not depend on the round count.
  const passKeys = (width: number) =>
    new Set(getRasterPipelines(build(width, 7)).map(pipeline => pipeline.key)).size;
  expect(passKeys(300)).toBe(passKeys(1100));
  device.destroy();
});

it('every banded distance-field kernel parses as WGSL', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const width = 300;
  const height = 40;
  const nodes = new GPUDistanceField({
    grid: makeGPURasterMetadata({width, height, cellSize: [1, 2]}),
    parameters: createTransientView(graph, 'parameters', 'float32', 8),
    seedPositions: createTransientView(graph, 'positions', 'float32x2', 16),
    output: {distances: createTransientView(graph, 'distances', 'float32', width * height)}
  }).getCommandNodes(graph);
  const sources = getGPUCommandNodeKernelSources(nodes);
  expect(sources.length).toBeGreaterThan(8);
  for (const kernel of sources) {
    expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
  }
  device.destroy();
});
