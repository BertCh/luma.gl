// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPU_CELL_GRID_DISTANCE_UNDEFINED,
  GPUCellGridPath
} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-cell-grid-path';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {h3ToBigInt, joinCellKey, splitCellKey} from '../cell-aggregation/cell-aggregation-oracle';
import {H3_GRID_PATH_FIXTURES} from './h3-grid-path-fixtures';

/** Pairs H3 rejects or the kernel must reject: resolution mismatch, zero cell, masked row. */
const EXTRA_PAIRS: [string, string, 'undefined' | 'masked'][] = [
  ['85283473fffffff', '8528342bfffffff', 'masked'],
  ['85283473fffffff', '8a283082a677fff', 'undefined'],
  ['0', '85283473fffffff', 'undefined'],
  ['85283473fffffff', 'ffffffffffffffff', 'undefined']
];

type Run = {
  distances: number[];
  cells: number[];
  counts: number[];
};

/** Builds and runs one graph over `pairs`, with `maximumPathLength = 0` meaning distance only. */
async function runGridPath(
  device: Device,
  pairs: {origin: bigint; destination: bigint; isMasked?: boolean}[],
  maximumPathLength: number,
  wordOrder: 'little-endian' | 'high-low' = 'little-endian'
): Promise<Run> {
  const rows = pairs.length;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const graph = new GPUCommandGraph(device, {id: 'cell-grid-path-graph'});
  const words = (select: 'origin' | 'destination') => {
    const array = new Uint32Array(2 * rows);
    pairs.forEach((pair, row) => {
      const [low, high] = splitCellKey(pair[select]);
      array.set(wordOrder === 'high-low' ? [high, low] : [low, high], 2 * row);
    });
    return array;
  };
  const originsBuffer = track(createInputBuffer(device, words('origin')));
  const destinationsBuffer = track(createInputBuffer(device, words('destination')));
  const maskBuffer = track(
    createInputBuffer(
      device,
      Uint32Array.from(pairs, pair => (pair.isMasked ? 0 : 1))
    )
  );
  const distancesBuffer = track(createOutputBuffer(device, rows));
  const cellsBuffer = track(createOutputBuffer(device, 2 * rows * Math.max(maximumPathLength, 1)));
  const countsBuffer = track(createOutputBuffer(device, rows));
  graph.add(
    new GPUCellGridPath({
      family: 'h3',
      origins: importGraphBuffer(graph, 'origins', originsBuffer, 'uint32x2', rows),
      destinations: importGraphBuffer(graph, 'destinations', destinationsBuffer, 'uint32x2', rows),
      wordOrder,
      mask: importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', rows),
      maximumPathLength: maximumPathLength || undefined,
      output: {
        distances: importGraphBuffer(graph, 'distances', distancesBuffer, 'uint32', rows),
        cells: maximumPathLength
          ? importGraphBuffer(graph, 'cells', cellsBuffer, 'uint32x2', rows * maximumPathLength)
          : undefined,
        counts: maximumPathLength
          ? importGraphBuffer(graph, 'counts', countsBuffer, 'uint32', rows)
          : undefined
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    distances: await readUint32(distancesBuffer, rows),
    cells: await readUint32(cellsBuffer, 2 * rows * Math.max(maximumPathLength, 1)),
    counts: await readUint32(countsBuffer, rows)
  };
  compiled.destroy();
  buffers.forEach(buffer => buffer.destroy());
  return result;
}

function getPathRow(run: Run, row: number, stride: number, wordOrder = 'little-endian'): string[] {
  const path: string[] = [];
  for (let slot = 0; slot < stride; slot++) {
    const index = 2 * (row * stride + slot);
    const low = wordOrder === 'high-low' ? run.cells[index + 1] : run.cells[index];
    const high = wordOrder === 'high-low' ? run.cells[index] : run.cells[index + 1];
    const cell = joinCellKey(low, high);
    path.push(cell === 0n ? '' : cell.toString(16));
  }
  return path;
}

it('GPUCellGridPath distances and paths match python h3 4.5.0 gridDistance and gridPathCells', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const stride = 64;
  const pairs = [
    ...H3_GRID_PATH_FIXTURES.map(([origin, destination]) => ({
      origin: h3ToBigInt(origin),
      destination: h3ToBigInt(destination)
    })),
    ...EXTRA_PAIRS.map(([origin, destination, kind]) => ({
      origin: h3ToBigInt(origin),
      destination: h3ToBigInt(destination),
      isMasked: kind === 'masked'
    }))
  ];
  const startTime = performance.now();
  const run = await runGridPath(device, pairs, stride);
  const elapsed = performance.now() - startTime;

  let sameCount = 0;
  let tieCount = 0;
  let failedCount = 0;
  H3_GRID_PATH_FIXTURES.forEach(([origin, destination, distance, relation, expectedPath], row) => {
    const label = `${origin} -> ${destination}`;
    expect(run.distances[row], `${label} distance`).toBe(
      distance < 0 ? GPU_CELL_GRID_DISTANCE_UNDEFINED : distance
    );
    if (distance < 0) {
      failedCount++;
      expect(run.counts[row], `${label} count`).toBe(0);
      expect(
        getPathRow(run, row, stride).every(cell => cell === ''),
        `${label} empty`
      ).toBe(true);
      return;
    }
    expect(run.counts[row], `${label} count`).toBe(distance + 1);
    if (!expectedPath) {
      // Longer than the pinned paths: the first `stride` cells start at the origin.
      expect(getPathRow(run, row, stride)[0], `${label} origin`).toBe(origin);
      return;
    }
    const actualPath = getPathRow(run, row, stride).slice(0, expectedPath.length);
    if (relation === 'same') {
      sameCount++;
      expect(actualPath, `${label} path`).toEqual(expectedPath);
    } else {
      tieCount++;
      expect(actualPath[0]).toBe(expectedPath[0]);
      expect(actualPath.at(-1)).toBe(expectedPath.at(-1));
      const differing = actualPath.filter((cell, slot) => cell !== expectedPath[slot]).length;
      expect(differing, `${label} cells differing at a cube-rounding tie`).toBeLessThanOrEqual(2);
    }
    expect(
      getPathRow(run, row, stride)
        .slice(expectedPath.length)
        .every(cell => cell === '')
    ).toBe(true);
  });
  EXTRA_PAIRS.forEach(([, , kind], extra) => {
    const row = H3_GRID_PATH_FIXTURES.length + extra;
    expect(run.distances[row], `extra ${kind} ${extra} distance`).toBe(
      GPU_CELL_GRID_DISTANCE_UNDEFINED
    );
    expect(run.counts[row]).toBe(0);
  });
  console.log(
    `GPUCellGridPath ${pairs.length} pairs in ${elapsed.toFixed(0)} ms: ${sameCount} identical paths, ${tieCount} tie deviations, ${failedCount} undefined`
  );
  expect(sameCount).toBeGreaterThan(200);
}, 120_000);

it('GPUCellGridPath distance-only mode, overflow counts and high-low word order', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const pairs = H3_GRID_PATH_FIXTURES.map(([origin, destination]) => ({
    origin: h3ToBigInt(origin),
    destination: h3ToBigInt(destination)
  }));
  const distanceOnly = await runGridPath(device, pairs, 0);
  H3_GRID_PATH_FIXTURES.forEach(([origin, destination, distance], row) => {
    expect(distanceOnly.distances[row], `${origin} -> ${destination}`).toBe(
      distance < 0 ? GPU_CELL_GRID_DISTANCE_UNDEFINED : distance
    );
  });

  // A short stride truncates every pinned path to its first cells and reports the true length.
  const shortStride = 8;
  const truncated = await runGridPath(device, pairs, shortStride, 'high-low');
  let overflowRows = 0;
  H3_GRID_PATH_FIXTURES.forEach(([origin, destination, distance, relation, expectedPath], row) => {
    if (distance < 0) {
      return;
    }
    expect(truncated.counts[row]).toBe(distance + 1);
    if (distance + 1 > shortStride) {
      overflowRows++;
    }
    if (relation === 'same' && expectedPath) {
      expect(
        getPathRow(truncated, row, shortStride, 'high-low'),
        `${origin} -> ${destination}`
      ).toEqual(Array.from({length: shortStride}, (_, slot) => expectedPath[slot] ?? ''));
    }
  });
  expect(overflowRows).toBeGreaterThan(50);
}, 120_000);

it('GPUCellGridPath validates its properties', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const graph = new GPUCommandGraph(device, {id: 'cell-grid-path-validation'});
  const buffer = createInputBuffer(device, new Uint32Array(4));
  const outBuffer = createOutputBuffer(device, 2);
  const origins = importGraphBuffer(graph, 'origins', buffer, 'uint32x2', 2);
  const distances = importGraphBuffer(graph, 'distances', outBuffer, 'uint32', 2);
  const base = {family: 'h3' as const, origins, destinations: origins, output: {distances}};
  expect(() => new GPUCellGridPath({...base, output: {}})).toThrow(/needs output/);
  expect(
    () =>
      new GPUCellGridPath({
        ...base,
        output: {cells: importGraphBuffer(graph, 'cells', buffer, 'uint32x2', 2)}
      })
  ).toThrow(/maximumPathLength/);
  expect(() => new GPUCellGridPath({...base, family: 'quadbin' as unknown as 'h3'})).toThrow(
    /family/
  );
  buffer.destroy();
  outBuffer.destroy();
});
