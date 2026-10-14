// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUNetworkCostMatrix} from '../../../src/gpu-network/network-accessibility';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  buildCSR,
  costMatrixOracle,
  createGridFixture,
  NONE,
  ZERO_WEIGHT_EDGES,
  type AccessibilityCSR,
  type AccessibilitySeed
} from './network-accessibility-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

/** Flattens seed rows into seed views with explicit rows. */
function flattenRows(rows: readonly (readonly AccessibilitySeed[])[]) {
  const nodes: number[] = [];
  const costs: number[] = [];
  const rowIds: number[] = [];
  rows.forEach((seeds, row) => {
    for (const seed of seeds) {
      nodes.push(seed.node);
      costs.push(seed.cost);
      rowIds.push(row);
    }
  });
  return {nodes, costs, rowIds};
}

/**
 * Builds, encodes once, and reads one cost matrix. Returns the matrix and convergence flag.
 */
async function runCostMatrix(
  device: Device,
  csr: AccessibilityCSR,
  nodeCount: number,
  rows: readonly (readonly AccessibilitySeed[])[],
  options: {
    laneCount?: number;
    maximumIterations?: number;
    costLimit?: number;
    targets?: readonly number[];
  } = {}
): Promise<{matrix: Float32Array; converged: number}> {
  const graph = new GPUCommandGraph(device, {id: 'cost-matrix'});
  const {nodes, costs, rowIds} = flattenRows(rows);
  const buffers: Buffer[] = [];
  const input = (values: Uint32Array | Float32Array) => {
    const buffer = createInputBuffer(device, values.length ? values : new Uint32Array(1));
    buffers.push(buffer);
    return buffer;
  };
  const columnCount = options.targets?.length ?? nodeCount;
  const costsBuffer = createOutputBuffer(device, rows.length * columnCount);
  const convergedBuffer = createOutputBuffer(device, 1);
  const edgeCount = csr.neighbors.length;
  buffers.push(costsBuffer, convergedBuffer);
  graph.add(
    new GPUNetworkCostMatrix({
      id: 'matrix',
      topology: {
        offsets: importGraphBuffer(graph, 'offsets', input(csr.offsets), 'uint32', nodeCount + 1),
        neighbors: importGraphBuffer(graph, 'neighbors', input(csr.neighbors), 'uint32', edgeCount),
        weights: importGraphBuffer(graph, 'weights', input(csr.weights), 'float32', edgeCount)
      },
      startNodes: importGraphBuffer(
        graph,
        'start-nodes',
        input(Uint32Array.from(nodes)),
        'uint32',
        nodes.length
      ),
      startCosts: importGraphBuffer(
        graph,
        'start-costs',
        input(Float32Array.from(costs)),
        'float32',
        costs.length
      ),
      startRows: importGraphBuffer(
        graph,
        'start-rows',
        input(Uint32Array.from(rowIds)),
        'uint32',
        rowIds.length
      ),
      costLimit:
        options.costLimit !== undefined
          ? importGraphBuffer(
              graph,
              'cost-limit',
              input(Float32Array.from([options.costLimit])),
              'float32',
              1
            )
          : undefined,
      targetNodes: options.targets
        ? importGraphBuffer(
            graph,
            'target-nodes',
            input(Uint32Array.from(options.targets)),
            'uint32',
            options.targets.length
          )
        : undefined,
      laneCount: options.laneCount,
      maximumIterations: options.maximumIterations ?? 16,
      output: {
        costs: importGraphBuffer(graph, 'costs', costsBuffer, 'float32', rows.length * columnCount),
        status: {
          converged: importGraphBuffer(graph, 'converged', convergedBuffer, 'uint32', 1)
        }
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const matrix = Float32Array.from(await readFloat32(costsBuffer, rows.length * columnCount));
  const [converged] = await readUint32(convergedBuffer, 1);
  compiled.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return {matrix, converged};
}

const ZERO_WEIGHT_ROWS: AccessibilitySeed[][] = [
  [{node: 0, cost: 0}],
  [{node: 2, cost: 0}],
  // Two seeds with initial costs, as a snapped point between nodes 4 and 5.
  [
    {node: 4, cost: 0.25},
    {node: 5, cost: 1.25}
  ],
  [{node: 6, cost: 0}],
  // A row whose only seed is rejected (negative cost) stays unreached.
  [{node: 1, cost: -1}],
  [{node: 8, cost: 0}],
  [
    {node: 3, cost: 0},
    {node: NONE, cost: 0}
  ]
];

it('GPUNetworkCostMatrix matches Dijkstra with zero-weight edges and disconnected components', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const csr = buildCSR(9, ZERO_WEIGHT_EDGES);
  const expected = costMatrixOracle(csr, 9, ZERO_WEIGHT_ROWS);
  // Spot checks of the fixture semantics.
  expect(Array.from(expected.subarray(0, 9))).toEqual([
    0,
    2,
    5,
    5,
    5,
    6.5,
    Infinity,
    Infinity,
    Infinity
  ]);
  expect(Array.from(expected.subarray(4 * 9, 5 * 9)).every(cost => cost === Infinity)).toBe(true);
  const results: Float32Array[] = [];
  for (const laneCount of [1, 3, ZERO_WEIGHT_ROWS.length]) {
    const {matrix, converged} = await runCostMatrix(device, csr, 9, ZERO_WEIGHT_ROWS, {laneCount});
    expect(Array.from(matrix)).toEqual(Array.from(expected));
    expect(converged).toBe(1);
    results.push(matrix);
  }
  // Bit-identical for every lane count.
  for (const matrix of results) {
    expect(new Uint32Array(matrix.buffer)).toEqual(new Uint32Array(results[0].buffer));
  }
});

it('GPUNetworkCostMatrix honors a cost limit and partial last batches on a grid', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 20;
  const nodeCount = width * width;
  const {edges} = createGridFixture(3, width, width);
  const csr = buildCSR(nodeCount, edges);
  const rows: AccessibilitySeed[][] = Array.from({length: 37}, (_, row) => [
    {node: (row * 97) % nodeCount, cost: row % 3}
  ]);
  const unbounded = await runCostMatrix(device, csr, nodeCount, rows, {
    laneCount: 16,
    maximumIterations: 32
  });
  expect(unbounded.converged).toBe(1);
  expect(Array.from(unbounded.matrix)).toEqual(Array.from(costMatrixOracle(csr, nodeCount, rows)));

  const bounded = await runCostMatrix(device, csr, nodeCount, rows, {
    laneCount: 8,
    maximumIterations: 32,
    costLimit: 20
  });
  expect(Array.from(bounded.matrix)).toEqual(
    Array.from(costMatrixOracle(csr, nodeCount, rows, 20))
  );

  // Too few rounds: the flag reports it.
  const truncated = await runCostMatrix(device, csr, nodeCount, rows, {
    laneCount: 37,
    maximumIterations: 1
  });
  expect(truncated.converged).toBe(0);
});

it('GPUNetworkCostMatrix default laneCount is bit-identical to laneCount 1', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 12;
  const nodeCount = width * width;
  const {edges} = createGridFixture(5, width, width);
  const csr = buildCSR(nodeCount, edges);
  const rows: AccessibilitySeed[][] = Array.from({length: nodeCount}, (_, row) => [
    {node: row, cost: 0}
  ]);
  const options = {maximumIterations: 32};
  const defaults = await runCostMatrix(device, csr, nodeCount, rows, options);
  const single = await runCostMatrix(device, csr, nodeCount, rows, {...options, laneCount: 1});
  expect(defaults.converged).toBe(1);
  expect(new Uint32Array(defaults.matrix.buffer)).toEqual(new Uint32Array(single.matrix.buffer));
});

/** Picks the target columns out of a dense matrix; targets `>= nodeCount` are `+Infinity`. */
function gatherDenseColumns(
  dense: Float32Array,
  nodeCount: number,
  rowCount: number,
  targets: readonly number[]
): Float32Array {
  const compact = new Float32Array(rowCount * targets.length);
  for (let row = 0; row < rowCount; row++) {
    targets.forEach((target, column) => {
      compact[row * targets.length + column] =
        target < nodeCount ? dense[row * nodeCount + target] : Infinity;
    });
  }
  return compact;
}

it('GPUNetworkCostMatrix targetNodes equals the dense matrix columns bit for bit', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 20;
  const nodeCount = width * width;
  const {edges} = createGridFixture(3, width, width);
  const csr = buildCSR(nodeCount, edges);
  const rows: AccessibilitySeed[][] = Array.from({length: 37}, (_, row) => [
    {node: (row * 97) % nodeCount, cost: row % 3}
  ]);
  // Duplicates, an out-of-range target, and the NONE sentinel.
  const targets = [5, 399, 5, 120, nodeCount + 7, 0, 120, NONE];
  const base = {maximumIterations: 32};
  const layouts = [
    {name: 'one batch', laneCount: rows.length},
    {name: 'multi batch with a short last batch', laneCount: 16},
    {name: 'one lane per batch', laneCount: 1}
  ];
  for (const costLimit of [undefined, 20]) {
    const dense = await runCostMatrix(device, csr, nodeCount, rows, {
      ...base,
      laneCount: 16,
      costLimit
    });
    const oracle = costMatrixOracle(csr, nodeCount, rows, costLimit);
    expect(Array.from(dense.matrix)).toEqual(Array.from(oracle));
    const expected = gatherDenseColumns(dense.matrix, nodeCount, rows.length, targets);
    for (const {name, laneCount} of layouts) {
      const compact = await runCostMatrix(device, csr, nodeCount, rows, {
        ...base,
        laneCount,
        costLimit,
        targets
      });
      expect(compact.matrix.length, name).toBe(rows.length * targets.length);
      expect(new Uint32Array(compact.matrix.buffer), name).toEqual(
        new Uint32Array(expected.buffer)
      );
      expect(compact.converged, name).toBe(1);
    }
  }
});

it('GPUNetworkCostMatrix targetNodes keeps unreached targets at infinity', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const csr = buildCSR(9, ZERO_WEIGHT_EDGES);
  const targets = [8, 0, 6, 0];
  const dense = await runCostMatrix(device, csr, 9, ZERO_WEIGHT_ROWS, {laneCount: 3});
  const compact = await runCostMatrix(device, csr, 9, ZERO_WEIGHT_ROWS, {laneCount: 3, targets});
  expect(new Uint32Array(compact.matrix.buffer)).toEqual(
    new Uint32Array(gatherDenseColumns(dense.matrix, 9, ZERO_WEIGHT_ROWS.length, targets).buffer)
  );
  expect(compact.matrix[0 * targets.length + 2]).toBe(Infinity);
});
