// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUNeighborhoodSummary,
  GPU_NEIGHBORHOOD_SUMMARY_NO_MODE,
  type GPUNeighborhoodSummaryStatistic
} from '../../../src/gpu-spatial-analysis/neighborhood-summary';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  computeNeighborhoodSummaryOracle,
  createWeights,
  type CPUWeights
} from './neighborhood-summary-oracle';

const GARBAGE = 0x7f7f7f7f;
const ALL: GPUNeighborhoodSummaryStatistic[] = [
  'count',
  'weightSum',
  'sum',
  'mean',
  'min',
  'max',
  'standardDeviation',
  'median'
];

function createScene(rows: number) {
  // Deterministic pseudo-random lists with varied degree (0 to 40), some islands.
  let state = 12345;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
  const lists = Array.from({length: rows}, (_, row) => {
    const degree = row % 17 === 0 ? 0 : row % 29 === 0 ? 40 : Math.floor(random() * 9);
    const chosen = new Map<number, number>();
    while (chosen.size < degree) {
      const neighbor = Math.floor(random() * rows);
      if (neighbor !== row) {
        chosen.set(neighbor, 0.25 + Math.floor(random() * 4) * 0.25);
      }
    }
    return [...chosen.entries()];
  });
  const values = Float32Array.from({length: rows}, () => Math.fround(random() * 20 - 5));
  const categories = Uint32Array.from({length: rows}, () => Math.floor(random() * 5) + 3);
  return {weights: createWeights(lists), values, categories};
}

async function runSummary(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  scene: {weights: CPUWeights; values: Float32Array; categories: Uint32Array; mask?: Uint32Array},
  options: {
    includeFocal?: boolean;
    focalWeight?: number;
    statistics: GPUNeighborhoodSummaryStatistic[];
    maximumNeighbors?: number;
    categorical?: boolean;
  }
) {
  const rows = scene.values.length;
  const capacity = scene.weights.neighbors.length;
  const graph = new GPUCommandGraph(device, {id: 'neighborhood-summary'});
  const buffers = {
    offsets: createInputBuffer(device, scene.weights.offsets),
    neighbors: createInputBuffer(device, scene.weights.neighbors),
    weights: createInputBuffer(device, scene.weights.weights),
    values: createInputBuffer(device, scene.values),
    categories: createInputBuffer(device, scene.categories),
    mask: scene.mask && createInputBuffer(device, scene.mask)
  };
  const outputs = {
    output: createOutputBuffer(device, rows * Math.max(options.statistics.length, 1)),
    overflow: createOutputBuffer(device, 1),
    modes: createOutputBuffer(device, rows),
    entropy: createOutputBuffer(device, rows)
  };
  for (const buffer of Object.values(outputs)) {
    buffer.write(new Uint32Array(buffer.byteLength / 4).fill(GARBAGE));
  }
  const statisticCount = options.statistics.length;
  const contributor = new GPUNeighborhoodSummary({
    weights: {
      offsets: importGraphBuffer(graph, 'offsets', buffers.offsets, 'uint32', rows + 1),
      neighbors: importGraphBuffer(graph, 'neighbors', buffers.neighbors, 'uint32', capacity),
      weights: importGraphBuffer(graph, 'weights', buffers.weights, 'float32', capacity)
    },
    values: importGraphBuffer(graph, 'values', buffers.values, 'float32', rows),
    categories: importGraphBuffer(graph, 'categories', buffers.categories, 'uint32', rows),
    mask: buffers.mask && importGraphBuffer(graph, 'mask', buffers.mask, 'uint32', rows),
    includeFocal: options.includeFocal,
    focalWeight: options.focalWeight,
    statistics: options.statistics,
    maximumNeighbors: options.maximumNeighbors,
    output: statisticCount
      ? importGraphBuffer(graph, 'output', outputs.output, 'float32', rows * statisticCount)
      : undefined,
    overflow: importGraphBuffer(graph, 'overflow', outputs.overflow, 'uint32', 1),
    modes: options.categorical
      ? importGraphBuffer(graph, 'modes', outputs.modes, 'uint32', rows)
      : undefined,
    entropy: options.categorical
      ? importGraphBuffer(graph, 'entropy', outputs.entropy, 'float32', rows)
      : undefined
  });
  graph.add(contributor);
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    table: statisticCount ? await readFloat32(outputs.output, rows * statisticCount) : [],
    overflow: (await readUint32(outputs.overflow, 1))[0],
    modes: options.categorical ? await readUint32(outputs.modes, rows) : [],
    entropy: options.categorical ? await readFloat32(outputs.entropy, rows) : []
  };
  compiled.destroy();
  for (const buffer of [...Object.values(buffers), ...Object.values(outputs)]) {
    buffer?.destroy();
  }
  return result;
}

function expectClose(actual: number, expected: number, label: string): void {
  if (Number.isNaN(expected)) {
    expect(actual, label).toBeNaN();
    return;
  }
  expect(Math.abs(actual - expected), `${label}: ${actual} vs ${expected}`).toBeLessThanOrEqual(
    1e-4 + 1e-4 * Math.abs(expected)
  );
}

for (const includeFocal of [false, true]) {
  it(`GPUNeighborhoodSummary matches the CPU oracle (includeFocal ${includeFocal})`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const scene = createScene(600);
    const mask = Uint32Array.from({length: 600}, (_, row) => (row % 13 === 5 ? 0 : 1));
    scene.values[7] = NaN;
    const options = {includeFocal, focalWeight: 2, maximumNeighbors: 32};
    const result = await runSummary(
      device,
      {...scene, mask},
      {...options, statistics: ALL, categorical: true}
    );
    const oracle = computeNeighborhoodSummaryOracle({...scene, mask, ...options});
    let medianOverflowRows = 0;
    let multiCategoryRows = 0;
    for (const [row, expected] of oracle.entries()) {
      const label = `row ${row}`;
      for (const [column, statistic] of ALL.entries()) {
        expectClose(
          result.table[row * ALL.length + column],
          expected[statistic],
          `${label} ${statistic}`
        );
      }
      expect(result.modes[row], `${label} mode`).toBe(expected.mode);
      expectClose(result.entropy[row], expected.entropy, `${label} entropy`);
      medianOverflowRows += Number.isNaN(expected.median) && expected.count > 32 ? 1 : 0;
      multiCategoryRows += expected.entropy > 0.1 ? 1 : 0;
    }
    // Degree-40 rows exceed 32 members; others are plentiful.
    expect(medianOverflowRows).toBeGreaterThan(5);
    expect(result.overflow).toBe(1);
    expect(multiCategoryRows).toBeGreaterThan(200);
    expect(
      result.modes.filter(mode => mode !== GPU_NEIGHBORHOOD_SUMMARY_NO_MODE).length
    ).toBeGreaterThan(300);
    // Masked rows output count 0 and NaN.
    expect(result.table[5 * ALL.length]).toBe(0);
    expect(result.table[5 * ALL.length + 3]).toBeNaN();
  });
}

it('GPUNeighborhoodSummary mode breaks ties to the lowest category and entropy is exact on a small case', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Row 0: neighbors 1..4 with categories 9, 4, 4, 9, equal weights: a tie, so the lowest (4).
  // Row 1: neighbors 0 (category 7, weight 1) and 4 (category 9, weight 3): 9 wins by weight.
  // Row 2: one neighbor. Row 3: island. Row 4: neighbors 1 and 2 (categories 9 and 4): a tie.
  const weights = createWeights([
    [
      [1, 1],
      [2, 1],
      [3, 1],
      [4, 1]
    ],
    [
      [0, 1],
      [4, 3]
    ],
    [[0, 2]],
    [],
    [
      [1, 1],
      [2, 1]
    ]
  ]);
  const values = Float32Array.from([1, 2, 3, 4, 5]);
  const categories = Uint32Array.from([7, 9, 4, 4, 9]);
  const result = await runSummary(
    device,
    {weights, values, categories},
    {
      statistics: ['count', 'median', 'min', 'max'],
      categorical: true
    }
  );
  expect(result.modes).toEqual([4, 9, 7, GPU_NEIGHBORHOOD_SUMMARY_NO_MODE, 4]);
  expect(result.entropy[0]).toBeCloseTo(Math.log(2), 6);
  expect(result.entropy[1]).toBeCloseTo(-(0.25 * Math.log(0.25) + 0.75 * Math.log(0.75)), 6);
  expect(result.entropy[2]).toBe(0);
  expect(result.entropy[3]).toBeNaN();
  expect(result.table.slice(0, 4)).toEqual([4, 3.5, 2, 5]);
});
