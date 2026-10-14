// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Same-run A/B of `GPUNeighborSearch` `searchStructure` ('grid' vs 'bvh' vs 'auto') on 2e5 clustered
 * points (90% inside 1% of the area) and 2e5 uniform points, kNN (k = 8) and radius (about ten
 * neighbors per query in the dense region), logging the median encode-to-readback time of each.
 * 'auto' pays the grid index and the BVH build every encoding, so it should land near the faster
 * path plus the build; if it does not beat the grid on uniform data by being close to it, or does
 * not beat it on clustered data, change the default or the gate threshold
 * (`GPU_NEIGHBOR_SEARCH_BVH_OCCUPANCY_THRESHOLD`).
 *
 * Run with `LUMA_TEST_BROWSER_BENCHMARKS=true npx vitest run --project headless --no-file-parallelism --silent=false <this file>`.
 */

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUNeighborSearchParameterValues,
  GPUNeighborSearch,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  type GPUNeighborSearchStructure
} from '../../../src/gpu-spatial-analysis/neighbor-search';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createSeededRandom} from './neighbor-search-oracle';

const DOMAIN_SIZE = 1000;
const POINT_COUNT = 200_000;
const WARMUP_COUNT = 3;
const SAMPLE_COUNT = 7;
const STRUCTURES: GPUNeighborSearchStructure[] = ['grid', 'bvh', 'auto'];

function createPoints(clustered: boolean): Float32Array {
  const random = createSeededRandom(12345);
  const points = new Float32Array(POINT_COUNT * 2);
  for (let row = 0; row < POINT_COUNT; row++) {
    // 90% inside a 100 x 100 box (1% of the domain).
    const dense = clustered && row % 10 !== 0;
    points[row * 2] = Math.fround(dense ? 450 + random() * 100 : random() * DOMAIN_SIZE);
    points[row * 2 + 1] = Math.fround(dense ? 450 + random() * 100 : random() * DOMAIN_SIZE);
  }
  return points;
}

it('GPUNeighborSearch grid versus bvh versus auto on clustered and uniform points', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);

  async function measure(
    points: Float32Array,
    structure: GPUNeighborSearchStructure,
    mode: 'knn' | 'radius',
    radius: number
  ): Promise<{milliseconds: number; total: number}> {
    const count = points.length / 2;
    const capacity = count * 16;
    const gridSize: [number, number] =
      mode === 'knn'
        ? [Math.ceil(Math.sqrt(count / 2)), Math.ceil(Math.sqrt(count / 2))]
        : [
            Math.min(1024, Math.floor(DOMAIN_SIZE / radius)),
            Math.min(1024, Math.floor(DOMAIN_SIZE / radius))
          ];
    const graph = new GPUCommandGraph(device!, {id: 'neighbor-search-structure-ab'});
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
        new GPUNeighborSearch({
          mode,
          k: mode === 'knn' ? 8 : undefined,
          gridSize,
          searchStructure: structure,
          positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', count),
          parameters: parameters.importToGraph(graph),
          output: {
            offsets: importGraphBuffer(graph, 'offsets', offsets, 'uint32', count + 1),
            neighbors: importGraphBuffer(graph, 'neighbors', neighbors, 'uint32', capacity),
            weights: importGraphBuffer(graph, 'weights', weights, 'float32', capacity),
            status: {
              overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1),
              requiredCount: importGraphBuffer(graph, 'total', total, 'uint32', 1)
            }
          }
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
      return {
        milliseconds: samples[Math.floor(samples.length / 2)],
        total: (await readUint32(total, 1))[0]
      };
    } finally {
      compiled?.destroy();
      parameters.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  }

  for (const clustered of [true, false]) {
    const points = createPoints(clustered);
    // About ten neighbors per query where the points are densest.
    const denseArea = clustered ? DOMAIN_SIZE * DOMAIN_SIZE * 0.01 : DOMAIN_SIZE * DOMAIN_SIZE;
    const denseCount = clustered ? POINT_COUNT * 0.9 : POINT_COUNT;
    const radius = Math.sqrt((10 * denseArea) / (Math.PI * denseCount));
    for (const mode of ['knn', 'radius'] as const) {
      const results = new Map<GPUNeighborSearchStructure, {milliseconds: number; total: number}>();
      for (const structure of STRUCTURES) {
        results.set(structure, await measure(points, structure, mode, radius));
      }
      const name = `${clustered ? 'clustered' : 'uniform'} ${mode}`;
      console.log(
        `neighbor-search-structure n=${POINT_COUNT} ${name}: ` +
          STRUCTURES.map(
            structure => `${structure} ${results.get(structure)!.milliseconds.toFixed(2)} ms`
          ).join(', ') +
          `, bvh/grid ${(results.get('bvh')!.milliseconds / results.get('grid')!.milliseconds).toFixed(2)}`
      );
      // Equal work: every structure finds the same number of neighbors.
      expect(results.get('bvh')!.total, name).toBe(results.get('grid')!.total);
      expect(results.get('auto')!.total, name).toBe(results.get('grid')!.total);
    }
  }
}, 900_000);
