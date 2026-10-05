// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {createRandom, GridStatisticsHarness} from './gpu-region-statistics-grid-harness';

const ROW_COUNT = 1_000_000;
const DOMAIN_SIZE = 1000;
const GRID_SIZE = [256, 256] as const;
const WARMUP_COUNT = 3;
const SAMPLE_COUNT = 10;
const PIPELINE_DEPTH = 20;

type BenchmarkCase = {
  label: string;
  kind: 'rectangle' | 'radius' | 'polygon';
  /** Region area as a fraction of the domain. */
  fraction: number;
  apply: (harness: GridStatisticsHarness) => void;
};

const CELL_SIZE = DOMAIN_SIZE / GRID_SIZE[0];

function createRectangleCase(fraction: number): BenchmarkCase {
  const side = Math.sqrt(fraction) * DOMAIN_SIZE;
  const minimum = 400;
  return {
    label: `rectangle ${fraction * 100}%`,
    kind: 'rectangle',
    fraction,
    apply: harness =>
      harness.setRectangle([minimum, minimum + 50, minimum + side, minimum + 50 + side])
  };
}

function createLassoCase(fraction: number): BenchmarkCase {
  const radius = Math.sqrt((fraction * DOMAIN_SIZE * DOMAIN_SIZE) / Math.PI);
  const vertices: number[] = [];
  for (let vertex = 0; vertex < 12; vertex++) {
    const angle = (vertex / 12) * Math.PI * 2;
    vertices.push(500 + Math.cos(angle) * radius, 500 + Math.sin(angle) * radius);
  }
  return {
    label: `lasso ${fraction * 100}%`,
    kind: 'polygon',
    fraction,
    apply: harness => harness.setPolygon(vertices)
  };
}

const CASES = [
  createRectangleCase(0.0001),
  createRectangleCase(0.001),
  createRectangleCase(0.01),
  createRectangleCase(0.1),
  createLassoCase(0.01)
];

function getMedian(samples: number[]): number {
  const sorted = [...samples].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)];
}

async function measureEncoding(
  harness: GridStatisticsHarness,
  which: 'brute' | 'grid'
): Promise<number> {
  const samples: number[] = [];
  for (let sample = 0; sample < WARMUP_COUNT + SAMPLE_COUNT; sample++) {
    const start = performance.now();
    if (which === 'brute') harness.encodeBrute();
    else harness.encodeGrid();
    await harness.sync(which);
    if (sample >= WARMUP_COUNT) samples.push(performance.now() - start);
  }
  return getMedian(samples);
}

/** Median per-encoding time when many encodings are queued before one readback. */
async function measurePipelinedEncoding(
  harness: GridStatisticsHarness,
  which: 'brute' | 'grid'
): Promise<number> {
  const samples: number[] = [];
  for (let batch = 0; batch < 5; batch++) {
    const start = performance.now();
    for (let encoding = 0; encoding < PIPELINE_DEPTH; encoding++) {
      if (which === 'brute') harness.encodeBrute();
      else harness.encodeGrid();
    }
    await harness.sync(which);
    samples.push((performance.now() - start) / PIPELINE_DEPTH);
  }
  return getMedian(samples);
}

it('GPURegionStatistics grid index benchmark', {timeout: 120000}, async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const random = createRandom(2024);
  const positions = new Float32Array(ROW_COUNT * 2);
  const values = new Float32Array(ROW_COUNT);
  for (let row = 0; row < ROW_COUNT; row++) {
    positions[row * 2] = random() * DOMAIN_SIZE;
    positions[row * 2 + 1] = random() * DOMAIN_SIZE;
    values[row] = random() * 100;
  }
  const lines: string[] = [];
  let indexBuildMilliseconds = 0;
  for (const benchmarkCase of CASES) {
    for (const withOutput of [false, true]) {
      const side = Math.sqrt(benchmarkCase.fraction) * DOMAIN_SIZE + 4 * CELL_SIZE;
      const candidateCapacity = Math.ceil(((side * side) / DOMAIN_SIZE ** 2) * ROW_COUNT * 1.25);
      const harness = new GridStatisticsHarness({
        device,
        id: `bench-${benchmarkCase.label}-${withOutput}`.replace(/[^a-z0-9-]/g, '-'),
        selectionKind: benchmarkCase.kind,
        positions,
        values,
        candidateCapacity,
        binCount: 16,
        gridSize: GRID_SIZE,
        domain: [0, 0, DOMAIN_SIZE, DOMAIN_SIZE],
        withOutput,
        indexMode: 'separate-graph'
      });
      const buildSamples: number[] = [];
      for (let build = 0; build < 3; build++) {
        const start = performance.now();
        harness.buildIndex();
        await harness.readIndexCount();
        buildSamples.push(performance.now() - start);
      }
      if (indexBuildMilliseconds === 0) indexBuildMilliseconds = getMedian(buildSamples);
      benchmarkCase.apply(harness);
      const bruteMilliseconds = await measureEncoding(harness, 'brute');
      const gridMilliseconds = await measureEncoding(harness, 'grid');
      const brutePipelined = await measurePipelinedEncoding(harness, 'brute');
      const gridPipelined = await measurePipelinedEncoding(harness, 'grid');
      const brute = await harness.read('brute');
      const grid = await harness.read('grid');
      expect(grid.result.candidatesTruncated).toBe(false);
      expect(grid.result.selectedCount).toBe(brute.result.selectedCount);
      expect(grid.result.minimum).toBe(brute.result.minimum);
      expect(grid.result.maximum).toBe(brute.result.maximum);
      expect(grid.result.selectedCount).toBeGreaterThan(0);
      if (withOutput) expect(grid.ids).toEqual(brute.ids);
      lines.push(
        `${benchmarkCase.label.padEnd(18)} ${withOutput ? 'ids ' : 'none'} ${String(brute.result.selectedCount).padStart(8)} ` +
          `${bruteMilliseconds.toFixed(2).padStart(9)} ${gridMilliseconds.toFixed(2).padStart(9)} ` +
          `${(bruteMilliseconds / gridMilliseconds).toFixed(1).padStart(7)}x ` +
          `${brutePipelined.toFixed(2).padStart(10)} ${gridPipelined.toFixed(2).padStart(9)} ` +
          `${(brutePipelined / gridPipelined).toFixed(1).padStart(7)}x  K=${candidateCapacity}`
      );
      harness.destroy();
    }
  }
  // eslint-disable-next-line no-console
  console.log(
    [
      `GPURegionStatistics grid index, ${ROW_COUNT} points, ${GRID_SIZE.join('x')} grid, median of ${SAMPLE_COUNT} synced encodings, then ${PIPELINE_DEPTH} queued encodings per readback`,
      `index build (separate graph, once): ${indexBuildMilliseconds.toFixed(2)} ms`,
      'selection          out   selected  brute ms   grid ms  speedup  piped brute  piped grid  piped speedup',
      ...lines
    ].join('\n')
  );
});
