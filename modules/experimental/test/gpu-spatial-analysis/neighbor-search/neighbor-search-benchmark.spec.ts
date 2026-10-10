// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Neighbor-search benchmark: the same uniform points in random and Morton order, including a
 * deliberately coarse radius grid that exercises cell-level candidate pruning. Queries run in
 * grid-cell order, so the two input orders must cost about the same.
 *
 * Run with `LUMA_TEST_BROWSER_BENCHMARKS=true npx vitest run --project headless --no-file-parallelism --silent=false <this file>`.
 */

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUNeighborSearchParameterValues,
  GPUNeighborSearch,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/neighbor-search';
import {
  createInputBuffer,
  createOutputBuffer,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

const DOMAIN_SIZE = 1000;
const WARMUP_COUNT = 3;
const SAMPLE_COUNT = 7;

function createRandomPoints(count: number): Float32Array {
  let state = 12345;
  const random = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000;
  };
  const points = new Float32Array(count * 2);
  for (let index = 0; index < points.length; index++) points[index] = random() * DOMAIN_SIZE;
  return points;
}

function sortMorton(points: Float32Array): Float32Array {
  const count = points.length / 2;
  const keys = new Uint32Array(count);
  for (let row = 0; row < count; row++) {
    const x = Math.min(1023, ((points[2 * row] / DOMAIN_SIZE) * 1024) | 0);
    const y = Math.min(1023, ((points[2 * row + 1] / DOMAIN_SIZE) * 1024) | 0);
    let key = 0;
    for (let bit = 0; bit < 10; bit++) {
      key |= ((x >> bit) & 1) << (2 * bit);
      key |= ((y >> bit) & 1) << (2 * bit + 1);
    }
    keys[row] = key;
  }
  const order = Uint32Array.from({length: count}, (_, row) => row).sort(
    (a, b) => keys[a] - keys[b]
  );
  const sorted = new Float32Array(points.length);
  for (let row = 0; row < count; row++) {
    sorted[2 * row] = points[2 * order[row]];
    sorted[2 * row + 1] = points[2 * order[row] + 1];
  }
  return sorted;
}

// Logs random-order cost relative to Morton order; timings only, no assertion, since the GPU may be shared.
it('GPUNeighborSearch random order versus Morton order benchmark', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const NeighborSearchClass = GPUNeighborSearch;

  async function measure(
    points: Float32Array,
    mode: 'knn' | 'radius',
    k: number,
    radius: number,
    gridSize: [number, number],
    capacity: number
  ): Promise<number> {
    const count = points.length / 2;
    const graph = new GPUCommandGraph(device!, {id: 'neighbor-search-benchmark'});
    const positionsBuffer = createInputBuffer(device!, points);
    const parameters = new GPUParameterBuffer(device!, {
      id: 'parameters',
      format: 'float32',
      length: GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
      values: getGPUNeighborSearchParameterValues({
        bounds: [0, 0, DOMAIN_SIZE, DOMAIN_SIZE],
        radius: mode === 'radius' ? radius : undefined
      })
    });
    const offsets = createOutputBuffer(device!, count + 1);
    const neighbors = createOutputBuffer(device!, capacity);
    const weights = createOutputBuffer(device!, capacity);
    const overflow = createOutputBuffer(device!, 1);
    const total = createOutputBuffer(device!, 1);
    const buffers = [positionsBuffer, offsets, neighbors, weights, overflow, total];
    let compiled: ReturnType<GPUCommandGraph['compile']> | undefined;
    try {
      graph.add(
        new NeighborSearchClass({
          mode,
          k,
          gridSize,
          positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', count),
          parameters: parameters.importToGraph(graph),
          weights: {
            offsets: importGraphBuffer(graph, 'offsets', offsets, 'uint32', count + 1),
            neighbors: importGraphBuffer(graph, 'neighbors', neighbors, 'uint32', capacity),
            weights: importGraphBuffer(graph, 'weights', weights, 'float32', capacity)
          },
          overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1),
          totalNeighbors: importGraphBuffer(graph, 'total', total, 'uint32', 1)
        })
      );
      compiled = graph.compile();
      const samples: number[] = [];
      for (let run = 0; run < WARMUP_COUNT + SAMPLE_COUNT; run++) {
        const start = performance.now();
        submitGraph(device!, compiled, undefined);
        await total.readAsync(0, 4);
        if (run >= WARMUP_COUNT) samples.push(performance.now() - start);
      }
      samples.sort((a, b) => a - b);
      return samples[Math.floor(samples.length / 2)];
    } finally {
      compiled?.destroy();
      parameters.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  }

  for (const count of [100_000, 1_000_000]) {
    const random = createRandomPoints(count);
    const morton = sortMorton(random);
    const radius = Math.sqrt((10 * DOMAIN_SIZE * DOMAIN_SIZE) / (Math.PI * count));
    const radiusGrid = Math.floor(DOMAIN_SIZE / radius);
    const knnGrid = Math.ceil(Math.sqrt(count / 2));
    const cases = [
      {name: 'knn', mode: 'knn' as const, k: 8, radius: 0, grid: knnGrid, capacity: count * 8},
      {
        name: 'radius',
        mode: 'radius' as const,
        k: 0,
        radius,
        grid: radiusGrid,
        capacity: Math.ceil(count * 13)
      },
      ...(count === 100_000
        ? [
            {
              name: 'radius-coarse-grid',
              mode: 'radius' as const,
              k: 0,
              radius,
              grid: 8,
              capacity: Math.ceil(count * 13)
            }
          ]
        : [])
    ];
    for (const {name, mode, k, radius: caseRadius, grid, capacity} of cases) {
      const args = [mode, k, caseRadius, [grid, grid] as [number, number], capacity] as const;
      const randomMilliseconds = await measure(random, ...args);
      const mortonMilliseconds = await measure(morton, ...args);
      const ratio = randomMilliseconds / mortonMilliseconds;
      console.log(
        `neighbor-search n=${count} ${name}: random ${randomMilliseconds.toFixed(2)} ms, morton ${mortonMilliseconds.toFixed(2)} ms, ratio ${ratio.toFixed(2)}`
      );
    }
  }
}, 600_000);
