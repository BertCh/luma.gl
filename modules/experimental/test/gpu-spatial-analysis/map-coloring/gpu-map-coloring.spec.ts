// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUContiguityWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {
  getGPUMapColoringParameterValues,
  GPUMapColoring,
  GPU_MAP_COLORING_PARAMETER_LENGTH,
  GPU_MAP_COLORING_UNCOLORED
} from '../../../src/gpu-spatial-analysis/map-coloring/index';
import {readUint32} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {
  computeContiguityOracle,
  createSeededRandom,
  flattenPolygons,
  type OracleCSR,
  type OraclePolygons
} from '../spatial-weights/spatial-weights-oracle';
import {
  computeMapColoringOracle,
  countColoringConflicts,
  createCliqueCSR,
  createOracleCSRFromEdges,
  createRandomPlanarCSR,
  createRookLatticeCSR,
  createStarCSR
} from './map-coloring-oracle';
import fixture from '../../oracles/fixtures/geometry-map-coloring.json';
import {requireTestDevice} from '../../utils/require-test-device';

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
  options: {
    seed?: number;
    maximumIterations?: number;
    upload?: boolean;
    priority?: 'largest-degree-first' | 'random';
    /** Uploaded graph used instead of the contiguity of `polygons`. */
    graph?: OracleCSR;
  }
) {
  const rig = new WeightsRig(device);
  const rows = options.graph ? options.graph.offsets.length - 1 : polygons.length;
  const colors = rig.output('uint32', rows);
  const colorCount = rig.output('uint32', 1);
  const conflictCount = rig.output('uint32', 1);
  const converged = rig.output('uint32', 1);
  const iterationCount = rig.output('uint32', 1);
  const csr = options.graph ?? computeContiguityOracle(polygons, 'queen');
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
        polygons: rig.polygons(layout),
        output: {...output.spatialWeights, status: {overflow: overflow.view}}
      })
    );
  }
  producers.push(
    new GPUMapColoring({
      weights,
      parameters: rig.input(
        getGPUMapColoringParameterValues({seed: options.seed}),
        'uint32',
        GPU_MAP_COLORING_PARAMETER_LENGTH
      ),
      output: {
        colors: colors.view,
        colorCount: colorCount.view,
        conflictCount: conflictCount.view,
        status: {converged: converged.view, iterationCount: iterationCount.view}
      },
      maximumIterations: options.maximumIterations,
      priority: options.priority
    })
  );
  rig.run(...producers);
  const result = {
    csr,
    colors: await readUint32(colors.buffer, rows),
    colorCount: (await readUint32(colorCount.buffer, 1))[0],
    conflicts: (await readUint32(conflictCount.buffer, 1))[0],
    converged: (await readUint32(converged.buffer, 1))[0],
    iterationCount: (await readUint32(iterationCount.buffer, 1))[0]
  };
  rig.destroy();
  return result;
}

it('GPUMapColoring matches the sequential priority-order oracle and is proper', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
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
      expect(result.iterationCount).toBeGreaterThan(1);
    }
  }
});

it('GPUMapColoring chains behind GPUContiguityWeights in one graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const polygons = createSquareGrid(8, 6);
  const result = await runColoring(device, polygons, {seed: 3});
  expect(result.colors).toEqual(computeMapColoringOracle(result.csr, 3));
  expect(result.conflicts).toBe(0);
  expect(result.colors.every(color => color !== GPU_MAP_COLORING_UNCOLORED)).toBe(true);
});

it('GPUMapColoring reports non-convergence when the round cap is too small', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const result = await runColoring(device, createSquareGrid(12, 9), {
    upload: true,
    maximumIterations: 1
  });
  expect(result.converged).toBe(0);
  expect(result.iterationCount).toBe(1);
  expect(result.colors.some(color => color === GPU_MAP_COLORING_UNCOLORED)).toBe(true);
  expect(result.colors.some(color => color !== GPU_MAP_COLORING_UNCOLORED)).toBe(true);
  expect(result.conflicts).toBe(0);
});

it('GPUMapColoring fused rounds still equal sequential greedy on a large grid', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Many rows per workgroup so rows color against neighbors written in the same round.
  for (const seed of [1, 5]) {
    const result = await runColoring(device, createSquareGrid(60, 50), {seed, upload: true});
    expect(result.colors).toEqual(computeMapColoringOracle(result.csr, seed));
    expect(result.converged).toBe(1);
    expect(result.conflicts).toBe(0);
  }
});

it('GPUMapColoring equals sequential greedy for both priorities on lattice, planar, star, clique, empty and isolated graphs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graphs: [string, OracleCSR][] = [
    ['rook lattice', createRookLatticeCSR(40, 30)],
    ['random planar', createRandomPlanarCSR(50, 50, createSeededRandom(21))],
    ['star', createStarCSR(200)],
    ['clique', createCliqueCSR(12)],
    ['isolated rows', createOracleCSRFromEdges(9, [[2, 3]])]
  ];
  let degreeFirstColors = 0;
  let randomColors = 0;
  for (const [name, graph] of graphs) {
    for (const priority of ['largest-degree-first', 'random'] as const) {
      for (const seed of [0, 9]) {
        const result = await runColoring(device, [], {
          graph,
          upload: true,
          seed,
          priority,
          maximumIterations: 128
        });
        const label = `${name} ${priority} seed ${seed}`;
        expect(result.colors, label).toEqual(computeMapColoringOracle(graph, seed, priority));
        expect(result.converged, label).toBe(1);
        expect(result.conflicts, label).toBe(0);
        if (name === 'random planar') {
          if (priority === 'random') randomColors += result.colorCount;
          else degreeFirstColors += result.colorCount;
        }
      }
    }
  }
  // Quality: largest-degree-first does not use more colors than random order on the planar map.
  expect(degreeFirstColors).toBeLessThanOrEqual(randomColors);
  // A star needs exactly two colors, and the hub goes first under largest-degree-first.
  const star = await runColoring(device, [], {graph: createStarCSR(200), upload: true});
  expect(star.colorCount).toBe(2);
  expect(star.iterationCount).toBeLessThanOrEqual(2);
});

it('GPUMapColoring A/B: color counts and rounds for both priorities on a ~1e5-polygon contiguity', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // 316 x 316 = 99856 rectangles; queen contiguity of a grid (needs at least 4 colors).
  const polygons = createSquareGrid(316, 316);
  for (const priority of ['random', 'largest-degree-first'] as const) {
    const start = performance.now();
    const result = await runColoring(device, polygons, {
      upload: true,
      priority,
      maximumIterations: 256
    });
    const milliseconds = performance.now() - start;
    // eslint-disable-next-line no-console
    console.log(
      `map-coloring A/B ${priority}: colors=${result.colorCount} rounds=${result.iterationCount} converged=${result.converged} ms=${milliseconds.toFixed(1)}`
    );
    expect(result.converged).toBe(1);
    expect(result.conflicts).toBe(0);
  }
}, 120_000);

it('GPUMapColoring largest-degree-first matches networkx largest_first quality on Delaunay maps', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const graph of fixture.graphs) {
    const csr: OracleCSR = {
      offsets: graph.offsets,
      neighbors: graph.neighbors,
      weights: [],
      distances: []
    };
    const counts: Record<string, number> = {};
    for (const priority of ['largest-degree-first', 'random'] as const) {
      const result = await runColoring(device, [], {graph: csr, upload: true, priority});
      expect(result.colors, `${graph.name} ${priority}`).toEqual(
        computeMapColoringOracle(csr, 0, priority)
      );
      expect(result.converged).toBe(1);
      expect(result.conflicts).toBe(0);
      counts[priority] = result.colorCount;
    }
    // Same heuristic as networkx largest_first with a different tie-break: one color of slack.
    expect(counts['largest-degree-first'], graph.name).toBeLessThanOrEqual(
      graph.networkxColorCounts.largest_first + 1
    );
    expect(counts['largest-degree-first'], graph.name).toBeLessThanOrEqual(counts['random']);
  }
});
