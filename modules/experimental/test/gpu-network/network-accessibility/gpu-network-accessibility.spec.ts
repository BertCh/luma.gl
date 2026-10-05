// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUNetworkAccessibility,
  GPUNetworkCostMatrix,
  GPUNetworkSnapping,
  GPU_NETWORK_ACCESSIBILITY_PARAMETER_LENGTH,
  encodeGPUNetworkAccessibilityParameters,
  type GPUNetworkAccessibilityParameters
} from '../../../src/gpu-network/network-accessibility';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  accessibilityOracle,
  buildCSR,
  catchmentOracle,
  costMatrixOracle,
  createGridFixture,
  dijkstra,
  snapOracle,
  ZERO_WEIGHT_EDGES,
  type AccessibilitySeed,
  type OracleParameters
} from './network-accessibility-oracle';

function toOracleParameters(parameters: GPUNetworkAccessibilityParameters): OracleParameters {
  return {
    threshold: parameters.threshold,
    decay: parameters.decay ?? 'none',
    beta: parameters.beta ?? 0,
    minimumCost: parameters.minimumCost ?? 1
  };
}

function expectClose(actual: readonly number[], expected: readonly number[]): void {
  expect(actual.length).toBe(expected.length);
  actual.forEach((value, index) => {
    const tolerance = 2e-5 * Math.max(1, Math.abs(expected[index]));
    if (Math.abs(value - expected[index]) > tolerance) {
      throw new Error(`row ${index}: ${value} != ${expected[index]}`);
    }
  });
}

type OutputName = 'cumulative' | 'gravity' | 'catchment' | 'ratios';
type ScoringResult = Record<OutputName, number[]>;

/** Scoring graph over a caller-owned matrix buffer, compiled once and re-encoded per frame. */
class ScoringFixture {
  readonly graph: GPUCommandGraph;
  readonly parameters: GPUParameterBuffer<'float32'>;
  readonly buffers: Buffer[] = [];
  readonly outputs: Partial<Record<OutputName, {buffer: Buffer; length: number}>> = {};
  compiled?: ReturnType<GPUCommandGraph['compile']>;
  compileCount = 0;

  constructor(
    readonly device: Device,
    matrix: Float32Array,
    props: {
      orientation: 'opportunity-rows' | 'origin-rows';
      weights: number[];
      originCount: number;
      demand?: number[];
    }
  ) {
    this.graph = new GPUCommandGraph(device, {id: 'scoring'});
    this.parameters = new GPUParameterBuffer(device, {
      id: 'accessibility-parameters',
      format: 'float32',
      length: GPU_NETWORK_ACCESSIBILITY_PARAMETER_LENGTH
    });
    const input = (id: string, values: Float32Array) => {
      const buffer = createInputBuffer(device, values);
      this.buffers.push(buffer);
      return importGraphBuffer(this.graph, id, buffer, 'float32', values.length);
    };
    const output = (id: OutputName, length: number): GraphDataView<'float32'> => {
      const buffer = createOutputBuffer(device, length);
      this.buffers.push(buffer);
      this.outputs[id] = {buffer, length};
      return importGraphBuffer(this.graph, id, buffer, 'float32', length);
    };
    const rowCount = props.orientation === 'opportunity-rows' ? props.weights.length : 0;
    this.graph.add(
      new GPUNetworkAccessibility({
        id: 'accessibility',
        costs: input('costs', matrix),
        orientation: props.orientation,
        opportunityWeights: input('weights', Float32Array.from(props.weights)),
        parameters: this.parameters.importToGraph(this.graph),
        cumulative: output('cumulative', props.originCount),
        gravity: output('gravity', props.originCount),
        catchment: props.demand
          ? {
              demand: input('demand', Float32Array.from(props.demand)),
              output: output('catchment', props.originCount),
              ratios: output('ratios', rowCount)
            }
          : undefined
      })
    );
  }

  /** Writes parameters and encodes the same compiled graph. */
  async run(parameters: GPUNetworkAccessibilityParameters): Promise<ScoringResult> {
    if (!this.compiled) {
      this.compiled = this.graph.compile();
      this.compileCount++;
    }
    this.parameters.write(encodeGPUNetworkAccessibilityParameters(parameters));
    submitGraph(this.device, this.compiled, undefined);
    const result: ScoringResult = {
      cumulative: [],
      gravity: [],
      catchment: [],
      ratios: []
    };
    for (const [id, output] of Object.entries(this.outputs)) {
      result[id as OutputName] = await readFloat32(output.buffer, output.length);
    }
    return result;
  }

  destroy(): void {
    this.compiled?.destroy();
    this.parameters.destroy();
    for (const buffer of this.buffers) {
      buffer.destroy();
    }
  }
}

const PARAMETER_SWEEP: GPUNetworkAccessibilityParameters[] = [
  {threshold: 10},
  {threshold: 25, decay: 'exponential', beta: 0.15},
  {threshold: Infinity, decay: 'exponential', beta: 0.05},
  {threshold: 18, decay: 'power', beta: 1.5, minimumCost: 1},
  {threshold: 0},
  {threshold: 40, decay: 'power', beta: 2, minimumCost: 2}
];

it('GPUNetworkAccessibility scores opportunity rows; threshold and decay change without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 16;
  const nodeCount = width * width;
  const {edges} = createGridFixture(11, width, width);
  const csr = buildCSR(nodeCount, edges);
  const opportunityRows: AccessibilitySeed[][] = Array.from({length: 24}, (_, row) => [
    {node: (row * 53 + 7) % nodeCount, cost: 0}
  ]);
  const weights = opportunityRows.map((_, row) => 1 + (row % 5));
  const matrix = costMatrixOracle(csr, nodeCount, opportunityRows, 40);
  const fixture = new ScoringFixture(device, matrix, {
    orientation: 'opportunity-rows',
    weights,
    originCount: nodeCount
  });
  const first: ScoringResult[] = [];
  for (const parameters of PARAMETER_SWEEP) {
    const result = await fixture.run(parameters);
    first.push(result);
    const expected = accessibilityOracle(
      matrix,
      weights.length,
      nodeCount,
      'opportunity-rows',
      weights,
      toOracleParameters(parameters)
    );
    // Integer weights: the fixed-order f32 cumulative sum is exact.
    expect(result.cumulative).toEqual(expected.cumulative);
    expectClose(result.gravity, expected.gravity);
  }
  // Re-encoding with the same parameters is bitwise identical (deterministic gathers).
  for (const [index, parameters] of PARAMETER_SWEEP.entries()) {
    const again = await fixture.run(parameters);
    expect(new Uint32Array(Float32Array.from(again.gravity).buffer)).toEqual(
      new Uint32Array(Float32Array.from(first[index].gravity).buffer)
    );
  }
  expect(fixture.compileCount).toBe(1);
  fixture.destroy();
});

it('GPUNetworkAccessibility scores origin rows with a deterministic per-row tree', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 24;
  const nodeCount = width * width;
  const {edges} = createGridFixture(5, width, width);
  const csr = buildCSR(nodeCount, edges);
  const originRows: AccessibilitySeed[][] = Array.from({length: 9}, (_, row) => [
    {node: (row * 61) % nodeCount, cost: 0}
  ]);
  // Opportunities on every node, zero on most.
  const weights = Array.from({length: nodeCount}, (_, node) => (node % 7 === 0 ? node % 4 : 0));
  const matrix = costMatrixOracle(csr, nodeCount, originRows);
  const fixture = new ScoringFixture(device, matrix, {
    orientation: 'origin-rows',
    weights,
    originCount: originRows.length
  });
  for (const parameters of PARAMETER_SWEEP) {
    const result = await fixture.run(parameters);
    const expected = accessibilityOracle(
      matrix,
      originRows.length,
      nodeCount,
      'origin-rows',
      weights,
      toOracleParameters(parameters)
    );
    expect(result.cumulative).toEqual(expected.cumulative);
    expectClose(result.gravity, expected.gravity);
  }
  expect(fixture.compileCount).toBe(1);
  fixture.destroy();
});

it('GPUNetworkAccessibility computes the two-step floating catchment area', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Facilities on the zero-weight fixture: one in the disconnected component with no demand.
  const nodeCount = 9;
  const csr = buildCSR(nodeCount, ZERO_WEIGHT_EDGES);
  const facilityRows: AccessibilitySeed[][] = [
    [{node: 0, cost: 0}],
    [{node: 4, cost: 0}],
    [{node: 8, cost: 0}],
    [{node: 3, cost: 0}]
  ];
  const supply = [10, 4, 7, 2];
  const demand = [100, 50, 20, 0, 0, 30, 0, 0, 0];
  const matrix = costMatrixOracle(csr, nodeCount, facilityRows);
  const fixture = new ScoringFixture(device, matrix, {
    orientation: 'opportunity-rows',
    weights: supply,
    originCount: nodeCount,
    demand
  });
  for (const parameters of PARAMETER_SWEEP) {
    const result = await fixture.run(parameters);
    const expected = catchmentOracle(
      matrix,
      facilityRows.length,
      nodeCount,
      supply,
      demand,
      toOracleParameters(parameters)
    );
    expectClose(result.ratios, expected.ratios);
    expectClose(result.catchment, expected.accessibility);
  }
  // The facility in the demand-free component has ratio 0, not NaN.
  const result = await fixture.run({threshold: Infinity});
  expect(result.ratios[2]).toBe(0);
  fixture.destroy();
});

it('GPUNetworkAccessibility end to end: snapped opportunities, lane-batched matrix, scores', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 12;
  const nodeCount = width * width;
  const {positions, edges} = createGridFixture(9, width, width);
  const csr = buildCSR(nodeCount, edges);
  // Opportunity points between grid nodes and one exactly on a node.
  const points = Float32Array.from([
    0.25, 0, 3.5, 2.2, 7, 7, 10.75, 4, 5.1, 9.6, 1, 10.5, 8.4, 0.2, 11, 11, 2.5, 6.25, 6.6, 3
  ]);
  const pointCount = points.length / 2;
  const opportunityWeights = Array.from({length: pointCount}, (_, row) => 1 + (row % 3));

  const graph = new GPUCommandGraph(device, {id: 'end-to-end'});
  const buffers: Buffer[] = [];
  const input = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    id: string,
    values: Float32Array | Uint32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    const length = format === 'float32x2' ? values.length / 2 : values.length;
    return importGraphBuffer(graph, id, buffer, format, length);
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return buffer;
  };
  const costLimit = 30;
  const matrixBuffer = output(pointCount * nodeCount);
  const gravityBuffer = output(nodeCount);
  const matrixView = importGraphBuffer(
    graph,
    'matrix',
    matrixBuffer,
    'float32',
    pointCount * nodeCount
  );
  const seedNodes = createTransientView(graph, 'seed-nodes', 'uint32', pointCount * 2);
  const seedCosts = createTransientView(graph, 'seed-costs', 'float32', pointCount * 2);
  const offsets = input('offsets', csr.offsets, 'uint32');
  const neighbors = input('neighbors', csr.neighbors, 'uint32');
  const weights = input('weights', csr.weights, 'float32');
  const snappedEdgesBuffer = output(pointCount);
  graph.add(
    new GPUNetworkSnapping({
      id: 'snap',
      points: input('points', points, 'float32x2'),
      nodePositions: input('positions', positions, 'float32x2'),
      offsets,
      edgeTargets: neighbors,
      edgeCosts: weights,
      snappedEdges: importGraphBuffer(
        graph,
        'snapped-edges',
        snappedEdgesBuffer,
        'uint32',
        pointCount
      ),
      seedNodes,
      seedCosts
    })
  );
  graph.add(
    new GPUNetworkCostMatrix({
      id: 'matrix',
      offsets,
      neighbors,
      weights,
      seedNodes,
      seedCosts,
      seedsPerRow: 2,
      laneCount: 4,
      maxIterations: 16,
      costLimit: input('cost-limit', Float32Array.from([costLimit]), 'float32'),
      costs: matrixView
    })
  );
  const parameters = new GPUParameterBuffer(device, {
    id: 'parameters',
    format: 'float32',
    length: GPU_NETWORK_ACCESSIBILITY_PARAMETER_LENGTH,
    values: encodeGPUNetworkAccessibilityParameters({
      threshold: 20,
      decay: 'exponential',
      beta: 0.1
    })
  });
  graph.add(
    new GPUNetworkAccessibility({
      id: 'accessibility',
      costs: matrixView,
      opportunityWeights: input(
        'opportunity-weights',
        Float32Array.from(opportunityWeights),
        'float32'
      ),
      parameters: parameters.importToGraph(graph),
      gravity: importGraphBuffer(graph, 'gravity', gravityBuffer, 'float32', nodeCount)
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);

  // Oracle: snap in f64, seed Dijkstra with both endpoint costs (undirected grid).
  const snaps = snapOracle(points, positions, csr.sources, csr.neighbors, {
    edgeCosts: csr.weights
  });
  expect(await readUint32(snappedEdgesBuffer, pointCount)).toEqual(snaps.map(snap => snap.edge));
  const rows = snaps.map(snap => [
    {node: csr.sources[snap.edge], cost: snap.sourceCost},
    {node: csr.neighbors[snap.edge], cost: snap.targetCost}
  ]);
  const expectedMatrix = costMatrixOracle(csr, nodeCount, rows, costLimit);
  const matrix = await readFloat32(matrixBuffer, pointCount * nodeCount);
  expectClose(
    matrix.map(value => (Number.isFinite(value) ? value : -1)),
    Array.from(expectedMatrix, value => (Number.isFinite(value) ? value : -1))
  );
  const expected = accessibilityOracle(
    expectedMatrix,
    pointCount,
    nodeCount,
    'opportunity-rows',
    opportunityWeights,
    {threshold: 20, decay: 'exponential', beta: 0.1, minimumCost: 1}
  );
  expectClose(await readFloat32(gravityBuffer, nodeCount), expected.gravity);
  // The first point sits a quarter along edge 0 -> 1, the first CSR slot.
  expect(snaps[0].edge).toBe(0);
  expect(dijkstra(csr, nodeCount, rows[0])[0]).toBeCloseTo(0.25 * csr.weights[0], 5);

  compiled.destroy();
  parameters.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
});
