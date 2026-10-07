// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUContiguityWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {
  GPUMapColoring,
  GPU_MAP_COLORING_UNCOLORED
} from '../../../src/gpu-spatial-analysis/map-coloring/index';
import {readUint32} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {
  computeContiguityOracle,
  createSeededRandom,
  flattenPolygons,
  type OraclePolygons
} from '../spatial-weights/spatial-weights-oracle';
import {computeMapColoringOracle, countColoringConflicts} from './map-coloring-oracle';

type Point = [number, number];

function createSquareGrid(columns: number, rows: number): OraclePolygons {
  const polygons: OraclePolygons = [];
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < columns; x++) {
      const ring: Point[] = [
        [x, y],
        [x + 1, y],
        [x + 1, y + 1],
        [x, y + 1]
      ];
      polygons.push([ring]);
    }
  }
  return polygons;
}

function createRandomRectangles(count: number, seed: number): OraclePolygons {
  const random = createSeededRandom(seed);
  const polygons: OraclePolygons = [];
  for (let index = 0; index < count; index++) {
    const x = Math.floor(random() * 14);
    const y = Math.floor(random() * 14);
    const width = 1 + Math.floor(random() * 3);
    const height = 1 + Math.floor(random() * 3);
    polygons.push([
      [
        [x, y],
        [x + width, y],
        [x + width, y + height],
        [x, y + height]
      ]
    ]);
  }
  return polygons;
}

async function runColoring(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  polygons: OraclePolygons,
  options: {seed?: number; maximumRounds?: number; upload?: boolean}
) {
  const rig = new WeightsRig(device);
  const rows = polygons.length;
  const colors = rig.output('uint32', rows);
  const colorCount = rig.output('uint32', 1);
  const conflictCount = rig.output('uint32', 1);
  const converged = rig.output('uint32', 1);
  const roundCount = rig.output('uint32', 1);
  const csr = computeContiguityOracle(polygons, 'queen');
  const producers = [];
  let weights;
  if (options.upload) {
    weights = rig.uploadWeights(csr);
  } else {
    // Chain: GPUContiguityWeights feeds GPUMapColoring inside one graph.
    const layout = flattenPolygons(polygons);
    const output = rig.weightsOutput(rows, csr.neighbors.length + 4);
    const overflow = rig.output('uint32', 1);
    weights = output.spatialWeights;
    producers.push(
      new GPUContiguityWeights({
        criterion: 'queen',
        positions: rig.input(layout.positions, 'float32x2', layout.positions.length / 2),
        ringOffsets: rig.input(layout.ringOffsets, 'uint32', layout.ringOffsets.length),
        polygonOffsets: rig.input(layout.polygonOffsets, 'uint32', layout.polygonOffsets.length),
        weights,
        overflow: overflow.view
      })
    );
  }
  producers.push(
    new GPUMapColoring({
      weights,
      colors: colors.view,
      colorCount: colorCount.view,
      conflictCount: conflictCount.view,
      converged: converged.view,
      roundCount: roundCount.view,
      seed: options.seed,
      maximumRounds: options.maximumRounds
    })
  );
  rig.run(...producers);
  const result = {
    csr,
    colors: await readUint32(colors.buffer, rows),
    colorCount: (await readUint32(colorCount.buffer, 1))[0],
    conflicts: (await readUint32(conflictCount.buffer, 1))[0],
    converged: (await readUint32(converged.buffer, 1))[0],
    roundCount: (await readUint32(roundCount.buffer, 1))[0]
  };
  rig.destroy();
  return result;
}

it('GPUMapColoring matches the sequential priority-order oracle and is proper', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const scenes: [string, OraclePolygons][] = [
    ['grid', createSquareGrid(12, 9)],
    ['rectangles', createRandomRectangles(120, 4)]
  ];
  for (const [name, polygons] of scenes) {
    for (const seed of [0, 7]) {
      const result = await runColoring(device, polygons, {seed, upload: true});
      const expected = computeMapColoringOracle(result.csr, seed);
      expect(result.colors, `${name} seed ${seed}`).toEqual(expected);
      expect(result.converged).toBe(1);
      expect(result.conflicts).toBe(0);
      expect(countColoringConflicts(result.csr, result.colors)).toBe(0);
      const maximum = Math.max(...expected) + 1;
      expect(result.colorCount).toBe(maximum);
      // Queen adjacency of a grid contains 4-cliques, so at least 4 colors are needed.
      expect(maximum).toBeGreaterThanOrEqual(name === 'grid' ? 4 : 2);
      expect(result.roundCount).toBeGreaterThan(1);
    }
  }
});

it('GPUMapColoring chains behind GPUContiguityWeights in one graph', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const polygons = createSquareGrid(8, 6);
  const result = await runColoring(device, polygons, {seed: 3});
  expect(result.colors).toEqual(computeMapColoringOracle(result.csr, 3));
  expect(result.conflicts).toBe(0);
  expect(result.colors.every(color => color !== GPU_MAP_COLORING_UNCOLORED)).toBe(true);
});

it('GPUMapColoring reports non-convergence when the round cap is too small', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const result = await runColoring(device, createSquareGrid(12, 9), {
    upload: true,
    maximumRounds: 1
  });
  expect(result.converged).toBe(0);
  expect(result.roundCount).toBe(1);
  expect(result.colors.some(color => color === GPU_MAP_COLORING_UNCOLORED)).toBe(true);
  expect(result.colors.some(color => color !== GPU_MAP_COLORING_UNCOLORED)).toBe(true);
  expect(result.conflicts).toBe(0);
});

it('GPUMapColoring fused rounds still equal sequential greedy on a large grid', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  // Many rows per workgroup so rows color against neighbors written in the same round.
  for (const seed of [1, 5]) {
    const result = await runColoring(device, createSquareGrid(60, 50), {seed, upload: true});
    expect(result.colors).toEqual(computeMapColoringOracle(result.csr, seed));
    expect(result.converged).toBe(1);
    expect(result.conflicts).toBe(0);
  }
});
