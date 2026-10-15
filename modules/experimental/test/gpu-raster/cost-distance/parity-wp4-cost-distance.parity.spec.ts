// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {it} from 'vitest';
import {makeGPURasterBand, makeGPURasterMetadata} from '../../../src/gpu-raster';
import {
  GPUCostDistance,
  GPUCostDistancePath,
  getGPUCostDistanceParameterValues,
  GPU_COST_DISTANCE_NONE
} from '../../../src/gpu-raster/cost-distance';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import fixtureJSON from '../../oracles/fixtures/parity-wp4-cost-distance.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {compareRasterParity} from '../../oracles/parity-wp4-compare';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

type ReferencePath = {target: number; indices: number[]; cost: number; unique: boolean};
type CostCase = {
  name: string;
  width: number;
  height: number;
  spacing: [number, number];
  friction: number[];
  sources: number[];
  costs: number[];
  unreachable: number[];
  predecessorMasks: number[];
  paths: ReferencePath[];
};
const fixture = decodeOracleFixture<{reference: string; cases: CostCase[]}>(fixtureJSON);
const OFFSETS = [
  [0, 1],
  [1, 1],
  [1, 0],
  [1, -1],
  [0, -1],
  [-1, -1],
  [-1, 0],
  [-1, 1]
];

/** Decode the public D8 representation, without reproducing the GPU solver. */
function getNeighbor(scene: CostCase, cell: number, code: number): number {
  const direction = Math.log2(code);
  if (!Number.isInteger(direction) || direction < 0 || direction >= 8) return -1;
  const [rowOffset, columnOffset] = OFFSETS[direction];
  const row = Math.floor(cell / scene.width) + rowOffset;
  const column = (cell % scene.width) + columnOffset;
  return row >= 0 && row < scene.height && column >= 0 && column < scene.width
    ? row * scene.width + column
    : -1;
}

function getEdgeCost(scene: CostCase, cell: number, neighbor: number): number {
  const rowDistance =
    (Math.floor(cell / scene.width) - Math.floor(neighbor / scene.width)) * scene.spacing[1];
  const columnDistance = ((cell % scene.width) - (neighbor % scene.width)) * scene.spacing[0];
  return (
    (Math.hypot(rowDistance, columnDistance) * (scene.friction[cell] + scene.friction[neighbor])) /
    2
  );
}

function getPath(scene: CostCase, links: number[], target: number): number[] {
  const path: number[] = [];
  const visited = new Set<number>();
  let cell = target;
  while (cell >= 0 && cell < links.length && !visited.has(cell)) {
    visited.add(cell);
    path.push(cell);
    if (links[cell] === 0) return scene.sources.includes(cell) ? path : [];
    cell = getNeighbor(scene, cell, links[cell]);
  }
  return [];
}

for (const scene of fixture.cases) {
  it(`GPUCostDistance and GPUCostDistancePath skimage parity: ${scene.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const grid = makeGPURasterMetadata({
      width: scene.width,
      height: scene.height,
      cellSize: scene.spacing
    });
    const cellCount = scene.width * scene.height;
    const buffers: Buffer[] = [];
    const input = (values: Float32Array | Uint32Array) => {
      const buffer = createInputBuffer(device, values);
      buffers.push(buffer);
      return buffer;
    };
    const output = (length: number) => {
      const buffer = createOutputBuffer(device, length);
      buffers.push(buffer);
      return buffer;
    };
    const costs = output(cellCount);
    const links = output(cellCount);
    const converged = output(1);
    const backLinks = importGraphBuffer(graph, 'links', links, 'uint32', cellCount);
    const parameters = new GPUParameterBuffer(device, {
      id: 'parameters',
      format: 'float32',
      length: 8
    });
    parameters.write(getGPUCostDistanceParameterValues({}, grid));
    graph.add(
      new GPUCostDistance({
        grid,
        friction: makeGPURasterBand(
          importGraphBuffer(graph, 'friction', input(new Float32Array(scene.friction)), 'float32'),
          grid
        ),
        parameters: parameters.importToGraph(graph),
        sources: importGraphBuffer(
          graph,
          'sources',
          input(new Uint32Array(scene.sources)),
          'uint32'
        ),
        maximumIterations: 256,
        maximumTieIterations: 256,
        output: {
          costs: importGraphBuffer(graph, 'costs', costs, 'float32'),
          backLinks,
          status: {converged: importGraphBuffer(graph, 'converged', converged, 'uint32')}
        }
      })
    );
    const pathRuns = [
      ...scene.paths.map(path => ({path, capacity: cellCount})),
      {path: scene.paths[0], capacity: 3}
    ].map(({path, capacity}, index) => {
      const prefix = `path-${index}`;
      const ids = output(capacity);
      const count = output(1);
      const overflow = output(1);
      const requiredCount = output(1);
      const iterationLimitReached = output(1);
      graph.add(
        new GPUCostDistancePath({
          id: prefix,
          grid,
          backLinks,
          target: importGraphBuffer(
            graph,
            `${prefix}-target`,
            input(new Uint32Array([path.target])),
            'uint32'
          ),
          output: {
            path: {
              ids: importGraphBuffer(graph, `${prefix}-ids`, ids, 'uint32'),
              status: {
                count: importGraphBuffer(graph, `${prefix}-count`, count, 'uint32'),
                overflow: importGraphBuffer(graph, `${prefix}-overflow`, overflow, 'uint32'),
                requiredCount: importGraphBuffer(
                  graph,
                  `${prefix}-required`,
                  requiredCount,
                  'uint32'
                )
              }
            },
            status: {
              iterationLimitReached: importGraphBuffer(
                graph,
                `${prefix}-limit`,
                iterationLimitReached,
                'uint32'
              )
            }
          }
        })
      );
      return {path, capacity, ids, count, overflow, requiredCount, iterationLimitReached};
    });
    const compiled = graph.compile();
    const compare = (
      className: string,
      metric: string,
      actual: ArrayLike<number>,
      expected: ArrayLike<number>,
      exact = false
    ) =>
      compareRasterParity(
        className,
        scene.name,
        fixture.reference,
        metric,
        actual,
        expected,
        exact ? 'E' : 'L'
      );
    try {
      submitGraph(device, compiled, undefined);
      const actualCosts = await readFloat32(costs, cellCount);
      const actualLinks = await readUint32(links, cellCount);
      compare('GPUCostDistance', 'converged', await readUint32(converged, 1), [1], true);
      compare('GPUCostDistance', 'costs', actualCosts, scene.costs);
      compare(
        'GPUCostDistance',
        'unreachableMask',
        actualCosts.map(value => (value === Infinity ? 1 : 0)),
        scene.unreachable,
        true
      );
      const validLinks: number[] = [];
      const walkCosts: number[] = [];
      const predecessorCosts: number[] = [];
      for (let cell = 0; cell < cellCount; cell++) {
        const code = actualLinks[cell];
        if (scene.unreachable[cell]) {
          validLinks.push(code === GPU_COST_DISTANCE_NONE ? 1 : 0);
          walkCosts.push(Infinity);
          predecessorCosts.push(Infinity);
        } else if (scene.sources.includes(cell)) {
          validLinks.push(code === 0 ? 1 : 0);
          walkCosts.push(code === 0 ? 0 : NaN);
          predecessorCosts.push(code === 0 ? 0 : NaN);
        } else {
          const neighbor = getNeighbor(scene, cell, code);
          validLinks.push(neighbor >= 0 && (scene.predecessorMasks[cell] & code) === code ? 1 : 0);
          predecessorCosts.push(
            neighbor < 0 ? NaN : scene.costs[neighbor] + getEdgeCost(scene, cell, neighbor)
          );
          const path = getPath(scene, actualLinks, cell);
          walkCosts.push(
            path.length === 0
              ? NaN
              : path
                  .slice(1)
                  .reduce((sum, next, index) => sum + getEdgeCost(scene, path[index], next), 0)
          );
        }
      }
      compare(
        'GPUCostDistance',
        'allBacklinksInIndependentOptimalSet',
        validLinks,
        new Uint32Array(cellCount).fill(1),
        true
      );
      compare('GPUCostDistance', 'allBacklinkBellmanCosts', predecessorCosts, scene.costs);
      compare('GPUCostDistance', 'allCompleteWalkCosts', walkCosts, scene.costs);
      for (const run of pathRuns) {
        const target = run.path.target;
        const expectedPath = scene.unreachable[target] ? [] : getPath(scene, actualLinks, target);
        const label = `target=${target}/capacity=${run.capacity}`;
        const actualCount = (await readUint32(run.count, 1))[0];
        compare(
          'GPUCostDistancePath',
          `${label}/count`,
          [actualCount],
          [Math.min(expectedPath.length, run.capacity)],
          true
        );
        compare(
          'GPUCostDistancePath',
          `${label}/requiredCount`,
          await readUint32(run.requiredCount, 1),
          [expectedPath.length],
          true
        );
        compare(
          'GPUCostDistancePath',
          `${label}/overflow`,
          await readUint32(run.overflow, 1),
          [Number(expectedPath.length > run.capacity)],
          true
        );
        compare(
          'GPUCostDistancePath',
          `${label}/iterationLimitReached`,
          await readUint32(run.iterationLimitReached, 1),
          [0],
          true
        );
        const actualPath = await readUint32(run.ids, actualCount);
        compare(
          'GPUCostDistancePath',
          `${label}/everyPathId`,
          actualPath,
          expectedPath.slice(0, run.capacity),
          true
        );
        if (run.path.unique)
          compare(
            'GPUCostDistancePath',
            `${label}/uniqueSkimagePath`,
            actualPath,
            run.path.indices.slice(0, run.capacity),
            true
          );
        if (actualCount > 0 && expectedPath.length <= run.capacity) {
          const pathCost = actualPath
            .slice(1)
            .reduce((sum, next, index) => sum + getEdgeCost(scene, actualPath[index], next), 0);
          compare(
            'GPUCostDistancePath',
            `${label}/independentOptimalCost`,
            [pathCost],
            [run.path.cost]
          );
        }
      }
    } finally {
      compiled.destroy();
      parameters.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  }, 120_000);
}
