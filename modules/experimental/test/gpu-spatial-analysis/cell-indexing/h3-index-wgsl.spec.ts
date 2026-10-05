// luma.gl
// SPDX-License-Identifier: MIT AND Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileCopyrightText: Copyright 2016-2024 Uber Technologies, Inc.

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {dggs} from '@luma.gl/shadertools';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {
  getHexagonEdgeLengthAvg,
  gridDistance,
  latLngToCell,
  cellToLatLng,
  greatCircleDistance
} from 'h3-js';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/utils/wgsl-kernel-nodes';
import {H3_INDEX_WGSL} from '../../../src/gpu-spatial-analysis/cell-indexing/h3-index-wgsl';
import {
  createInputBuffer,
  createOutputBuffer,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  createCellEdgePoints,
  createPentagonNeighborhoodPoints,
  createSpecialH3Points,
  createUniformSpherePoints
} from './h3-index-oracle';

const CAPACITY = 32768;

type H3ForwardRunner = {
  /** Runs `cellIndexH3FromLngLat` for every lng/lat pair and returns the 64-bit keys as bigint. */
  run(points: Float32Array, resolution: number): Promise<bigint[]>;
  destroy(): void;
};

/** Compiles one reusable kernel that indexes up to CAPACITY points at a runtime resolution. */
function createRunner(device: Device): H3ForwardRunner {
  const positions = createInputBuffer(device, new Float32Array(2 * CAPACITY));
  const meta = createInputBuffer(device, new Uint32Array([0, 0]));
  const output = createOutputBuffer(device, 2 * CAPACITY);
  const graph = new GPUCommandGraph(device, {id: 'h3-forward-graph'});
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'h3-forward-kernel',
      operation: 'H3Forward',
      bindings: [
        {
          name: 'positions',
          view: importGraphBuffer(graph, 'positions', positions, 'float32x2', CAPACITY),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'runParams',
          view: importGraphBuffer(graph, 'meta', meta, 'uint32', 2),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'cells',
          view: importGraphBuffer(graph, 'cells', output, 'uint32x2', CAPACITY),
          type: 'u32',
          access: 'read_write'
        }
      ],
      invocationCount: CAPACITY,
      declarations: `${dggs.source}\n${H3_INDEX_WGSL}`,
      body: `var key = vec2u(0u);
  if (index < runParams[runParamsOffset + 1u]) {
    key = cellIndexH3FromLngLat(
      vec2f(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]),
      runParams[runParamsOffset]
    );
  }
  cells[cellsOffset + 2u * index] = key.y;
  cells[cellsOffset + 2u * index + 1u] = key.x;`
    })
  );
  const compiled = graph.compile();
  return {
    async run(points, resolution) {
      const count = points.length / 2;
      expect(count).toBeLessThanOrEqual(CAPACITY);
      positions.write(points);
      meta.write(new Uint32Array([resolution, count]));
      submitGraph(device, compiled, undefined);
      const bytes = await output.readAsync();
      const words = new Uint32Array(bytes.buffer, bytes.byteOffset, 2 * count);
      const keys: bigint[] = [];
      for (let row = 0; row < count; row++) {
        keys.push((BigInt(words[2 * row + 1]) << 32n) | BigInt(words[2 * row]));
      }
      return keys;
    },
    destroy() {
      compiled.destroy();
      positions.destroy();
      meta.destroy();
      output.destroy();
    }
  };
}

type Comparison = {rows: number; mismatches: number; neighbors: number};

/** Compares GPU keys with h3-js on the same float32 inputs; mismatches are classified. */
function compareWithH3(points: Float32Array, keys: bigint[], resolution: number): Comparison {
  let mismatches = 0;
  let neighbors = 0;
  const spacing = (getHexagonEdgeLengthAvg(resolution, 'm') * Math.sqrt(3)) / 1000;
  for (let row = 0; row < keys.length; row++) {
    const expected = latLngToCell(points[2 * row + 1], points[2 * row], resolution);
    const actual = keys[row].toString(16);
    if (actual === expected) {
      continue;
    }
    mismatches++;
    let isNeighbor = false;
    try {
      isNeighbor = gridDistance(expected, actual) <= 1;
    } catch {
      // Pentagon distortion or a null key: fall back to center spacing below.
      if (keys[row] !== 0n) {
        const [expectedLat, expectedLng] = cellToLatLng(expected);
        const [actualLat, actualLng] = cellToLatLng(actual);
        isNeighbor =
          greatCircleDistance([expectedLat, expectedLng], [actualLat, actualLng], 'km') <
          1.2 * spacing;
      }
    }
    if (isNeighbor) {
      neighbors++;
    }
  }
  return {rows: keys.length, mismatches, neighbors};
}

/**
 * Maximum allowed mismatch rate on area-uniform points, by resolution. Measured on the headless
 * WebGPU device (20000 points per resolution) and set to roughly 2 to 3 times the observation.
 * Resolutions 0..4 are exact.
 */
const UNIFORM_MISMATCH_BOUND: readonly number[] = [
  0, 0, 0, 0, 0, 1e-3, 1e-3, 2e-3, 2e-3, 4e-3, 8e-3, 2.5e-2, 5e-2, 0.1, 0.2, 0.4
];
/**
 * Maximum allowed mismatch rate on near-edge points (jittered by 2 percent of an edge length), by
 * resolution (6000 points per resolution). Resolutions 0..2 are exact.
 */
const EDGE_MISMATCH_BOUND: readonly number[] = [
  0, 0, 0, 2e-3, 4e-3, 1e-2, 2e-2, 3e-2, 6e-2, 0.12, 0.2, 0.3, 0.4, 0.5, 0.5, 0.55
];
/** Maximum allowed mismatch rate within two edge lengths of a pentagon center, by resolution. */
const PENTAGON_MISMATCH_BOUND: readonly number[] = [
  0, 0, 0, 0, 0, 0, 0, 0, 0, 1e-2, 3e-2, 5e-2, 6e-2, 0.15, 0.3, 0.7
];

it('cellIndexH3FromLngLat matches h3-js on the same f32 inputs, with measured per-resolution limits', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const runner = createRunner(device);
  const table: string[] = [
    'res | uniform n  mism  rate     nbr | edge n  mism  rate     nbr | pent n mism nbr'
  ];
  for (let resolution = 0; resolution <= 15; resolution++) {
    const uniformPoints = createUniformSpherePoints(20000, 1000 + resolution);
    const uniform = compareWithH3(
      uniformPoints,
      await runner.run(uniformPoints, resolution),
      resolution
    );
    const edgePoints = createCellEdgePoints(resolution, 6000, 2000 + resolution);
    const edge = compareWithH3(edgePoints, await runner.run(edgePoints, resolution), resolution);
    const pentagonPoints = createPentagonNeighborhoodPoints(resolution, 100, 3000 + resolution);
    const pentagon = compareWithH3(
      pentagonPoints,
      await runner.run(pentagonPoints, resolution),
      resolution
    );
    const rate = (c: Comparison) => (c.mismatches / c.rows).toExponential(2);
    table.push(
      `${String(resolution).padStart(3)} | ${uniform.rows} ${String(uniform.mismatches).padStart(5)} ${rate(uniform)} ${String(uniform.neighbors).padStart(5)} | ` +
        `${edge.rows} ${String(edge.mismatches).padStart(5)} ${rate(edge)} ${String(edge.neighbors).padStart(5)} | ` +
        `${pentagon.rows} ${String(pentagon.mismatches).padStart(4)} ${String(pentagon.neighbors).padStart(4)}`
    );
    expect(uniform.mismatches / uniform.rows, `uniform res ${resolution}`).toBeLessThanOrEqual(
      UNIFORM_MISMATCH_BOUND[resolution]
    );
    expect(edge.mismatches / edge.rows, `edge res ${resolution}`).toBeLessThanOrEqual(
      EDGE_MISMATCH_BOUND[resolution]
    );
    expect(pentagon.mismatches / pentagon.rows, `pentagon res ${resolution}`).toBeLessThanOrEqual(
      PENTAGON_MISMATCH_BOUND[resolution]
    );
    // Float32 error moves a point across one cell edge, so a wrong answer is a grid neighbor.
    // Resolution 15 is excluded: the float32 input is coarser than the cell there.
    for (const comparison of resolution < 15 ? [uniform, edge, pentagon] : []) {
      expect(comparison.neighbors, `neighbors res ${resolution}`).toBeGreaterThanOrEqual(
        Math.floor(0.98 * comparison.mismatches)
      );
    }
  }
  console.error(`H3 forward GPU vs h3-js (float32 inputs)\n${table.join('\n')}`);
  runner.destroy();
});

it('cellIndexH3FromLngLat handles poles, antimeridian, equator, NaN and invalid resolution', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const runner = createRunner(device);
  const special = createSpecialH3Points();
  for (const resolution of [0, 1, 2, 3, 4]) {
    const comparison = compareWithH3(special, await runner.run(special, resolution), resolution);
    // Poles and +-180 sit on edges or vertices at coarse resolutions; allow only grid neighbors.
    expect(comparison.mismatches - comparison.neighbors, `special res ${resolution}`).toBe(0);
    expect(comparison.mismatches / comparison.rows).toBeLessThan(0.1);
  }
  const bad = new Float32Array([NaN, 10, 10, NaN, Infinity, 0, 0, -Infinity, 20, 30]);
  const keys = await runner.run(bad, 5);
  expect(keys.slice(0, 4)).toEqual([0n, 0n, 0n, 0n]);
  expect(keys[4]).toBe(BigInt(`0x${latLngToCell(30, 20, 5)}`));
  expect(await runner.run(new Float32Array([10, 10]), 16)).toEqual([0n]);
  runner.destroy();
});
