// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUNetworkServiceAreas,
  type GPUNetworkServiceAreasProps
} from '../../../src/gpu-network/network-analysis/gpu-network-service-areas';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  buildCSR,
  createRandomNetwork,
  dijkstra,
  NONE,
  type NetworkCSR,
  type NetworkEdge
} from '../network-reachability/network-reachability-oracle';
import {serviceAreasOracle, type ServiceAreasFacility} from './network-service-areas-oracle';

type ServiceOptions = {
  /** Facility nodes; row = index. Also the facility capacity. */
  facilities: number[];
  facilityCosts?: number[];
  facilityCount?: number;
  costLimit?: number;
  maxIterations?: number;
  activeIterations?: number;
};

type ServiceFixture = {
  graph: GPUCommandGraph;
  nodeCount: number;
  capacity: number;
  parameters: GPUParameterBuffer[];
  buffers: Buffer[];
  weightsBuffer: Buffer;
  facilities: GPUParameterBuffer<'uint32'>;
  facilityCosts?: GPUParameterBuffer<'float32'>;
  facilityCount?: GPUParameterBuffer<'uint32'>;
  costLimit?: GPUParameterBuffer<'float32'>;
  activeIterations?: GPUParameterBuffer<'uint32'>;
  assignmentsBuffer: Buffer;
  costsBuffer: Buffer;
  nodeCountsBuffer: Buffer;
  costSumsBuffer: Buffer;
  convergedBuffer: Buffer;
};

function createServiceFixture(
  device: Device,
  csr: NetworkCSR,
  nodeCount: number,
  options: ServiceOptions
): ServiceFixture {
  const graph = new GPUCommandGraph(device, {id: 'service-areas'});
  const capacity = options.facilities.length;
  const edgeCount = csr.neighbors.length;
  const offsetsBuffer = createInputBuffer(device, csr.offsets);
  const neighborsBuffer = createInputBuffer(device, edgeCount ? csr.neighbors : new Uint32Array(1));
  const weightsBuffer = createInputBuffer(device, edgeCount ? csr.weights : new Float32Array(1));
  const assignmentsBuffer = createOutputBuffer(device, nodeCount);
  const costsBuffer = createOutputBuffer(device, nodeCount);
  const nodeCountsBuffer = createOutputBuffer(device, capacity);
  const costSumsBuffer = createOutputBuffer(device, capacity);
  const convergedBuffer = createOutputBuffer(device, 1);
  const parameters: GPUParameterBuffer[] = [];
  const createParameter = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    values: number[]
  ) => {
    const parameter = new GPUParameterBuffer(device, {
      id: `service-${name}`,
      format,
      length: values.length,
      values: format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values)
    });
    parameters.push(parameter);
    return parameter;
  };
  const facilities = createParameter('facilities', 'uint32', options.facilities);
  const facilityCosts = options.facilityCosts
    ? createParameter('facility-costs', 'float32', options.facilityCosts)
    : undefined;
  const facilityCount =
    options.facilityCount !== undefined
      ? createParameter('facility-count', 'uint32', [options.facilityCount])
      : undefined;
  const costLimit =
    options.costLimit !== undefined
      ? createParameter('cost-limit', 'float32', [options.costLimit])
      : undefined;
  const activeIterations =
    options.activeIterations !== undefined
      ? createParameter('active-iterations', 'uint32', [options.activeIterations])
      : undefined;
  const props: GPUNetworkServiceAreasProps = {
    id: 'service',
    offsets: importGraphBuffer(graph, 'service-offsets', offsetsBuffer, 'uint32', nodeCount + 1),
    neighbors: importGraphBuffer(graph, 'service-neighbors', neighborsBuffer, 'uint32', edgeCount),
    weights: importGraphBuffer(graph, 'service-weights', weightsBuffer, 'float32', edgeCount),
    facilities: facilities.importToGraph(graph),
    facilityCosts: facilityCosts?.importToGraph(graph),
    facilityCount: facilityCount?.importToGraph(graph),
    costLimit: costLimit?.importToGraph(graph),
    maxIterations: options.maxIterations ?? 32,
    activeIterations: activeIterations?.importToGraph(graph),
    assignments: importGraphBuffer(
      graph,
      'service-assignments',
      assignmentsBuffer,
      'uint32',
      nodeCount
    ),
    costs: importGraphBuffer(graph, 'service-costs', costsBuffer, 'float32', nodeCount),
    facilityNodeCounts: importGraphBuffer(
      graph,
      'service-node-counts',
      nodeCountsBuffer,
      'uint32',
      capacity
    ),
    facilityCostSums: importGraphBuffer(
      graph,
      'service-cost-sums',
      costSumsBuffer,
      'float32',
      capacity
    ),
    converged: importGraphBuffer(graph, 'service-converged', convergedBuffer, 'uint32', 1)
  };
  graph.add(new GPUNetworkServiceAreas(props));
  return {
    graph,
    nodeCount,
    capacity,
    parameters,
    buffers: [
      offsetsBuffer,
      neighborsBuffer,
      weightsBuffer,
      assignmentsBuffer,
      costsBuffer,
      nodeCountsBuffer,
      costSumsBuffer,
      convergedBuffer
    ],
    weightsBuffer,
    facilities,
    facilityCosts,
    facilityCount,
    costLimit,
    activeIterations,
    assignmentsBuffer,
    costsBuffer,
    nodeCountsBuffer,
    costSumsBuffer,
    convergedBuffer
  };
}

function destroyFixture(fixture: ServiceFixture): void {
  for (const parameter of fixture.parameters) parameter.destroy();
  for (const buffer of fixture.buffers) buffer.destroy();
}

/** Reads every output and compares it with the oracle evaluated on the GPU costs. */
async function expectMatchesOracle(
  fixture: ServiceFixture,
  csr: NetworkCSR,
  facilities: readonly ServiceAreasFacility[],
  facilityCount: number,
  costLimit: number,
  exactCosts: boolean
): Promise<{costs: Float32Array; assignments: number[]}> {
  const {nodeCount, capacity} = fixture;
  const costs = Float32Array.from(await readFloat32(fixture.costsBuffer, nodeCount));
  const reference = dijkstra(csr, nodeCount, facilities.slice(0, facilityCount), costLimit);
  for (let node = 0; node < nodeCount; node++) {
    if (exactCosts) {
      expect(costs[node]).toBe(reference[node]);
    } else {
      expect(Number.isFinite(costs[node])).toBe(Number.isFinite(reference[node]));
      if (Number.isFinite(reference[node])) {
        expect(Math.abs(costs[node] - reference[node])).toBeLessThanOrEqual(
          1e-4 * Math.max(1, reference[node])
        );
      }
    }
  }
  const expected = serviceAreasOracle(csr, nodeCount, facilities, facilityCount, costLimit, costs);
  const assignments = await readUint32(fixture.assignmentsBuffer, nodeCount);
  expect(assignments).toEqual(Array.from(expected.assignments));
  const rowCount = Math.min(capacity, facilities.length);
  expect(await readUint32(fixture.nodeCountsBuffer, rowCount)).toEqual(
    Array.from(expected.nodeCounts.slice(0, rowCount))
  );
  const sums = await readFloat32(fixture.costSumsBuffer, rowCount);
  for (let row = 0; row < rowCount; row++) {
    expect(Math.abs(sums[row] - expected.costSums[row])).toBeLessThanOrEqual(
      1e-3 * Math.max(1, expected.costSums[row])
    );
  }
  return {costs, assignments};
}

const HAND_EDGES: NetworkEdge[] = [
  [5, 4, 2],
  [0, 4, 2],
  [4, 3, 0], // zero-weight connector keeps the facility of node 4
  [4, 6, 1],
  [8, 7, 1],
  [7, 6, 2], // node 6 is equidistant (3) from facility rows 0/1 and row 2
  [1, 2, 1],
  [0, 2, 7],
  [9, 10, 1],
  [10, 9, 1]
];
const HAND = buildCSR(11, HAND_EDGES);
// Row 0 sits on the largest node index among the tied facilities; row 3 starts at cost 5.
const HAND_FACILITIES: ServiceAreasFacility[] = [
  {node: 5, cost: 0},
  {node: 0, cost: 0},
  {node: 8, cost: 0},
  {node: 1, cost: 5}
];

it('GPUNetworkServiceAreas breaks ties by facility row on a hand fixture', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const fixture = createServiceFixture(device, HAND, 11, {
    facilities: HAND_FACILITIES.map(facility => facility.node),
    facilityCosts: HAND_FACILITIES.map(facility => facility.cost)
  });
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  const {costs, assignments} = await expectMatchesOracle(
    fixture,
    HAND,
    HAND_FACILITIES,
    4,
    Infinity,
    true
  );
  expect(Array.from(costs)).toEqual([0, 5, 6, 2, 2, 0, 3, 1, 0, Infinity, Infinity]);
  expect(assignments).toEqual([1, 3, 3, 0, 0, 0, 0, 2, 2, NONE, NONE]);
  expect(await readUint32(fixture.nodeCountsBuffer, 4)).toEqual([4, 1, 2, 2]);
  expect(await readUint32(fixture.convergedBuffer, 1)).toEqual([1]);
  compiled.destroy();
  destroyFixture(fixture);
});

async function runRandomNetwork(
  device: Device,
  csr: NetworkCSR,
  facilityNodes: number[],
  exactCosts: boolean
): Promise<void> {
  const nodeCount = 2000;
  const facilities = facilityNodes.map(node => ({node, cost: 0}));
  const fixture = createServiceFixture(device, csr, nodeCount, {
    facilities: facilityNodes,
    maxIterations: 256
  });
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  const {assignments} = await expectMatchesOracle(
    fixture,
    csr,
    facilities,
    facilities.length,
    Infinity,
    exactCosts
  );
  expect(assignments.some(row => row !== NONE)).toBe(true);
  expect(await readUint32(fixture.convergedBuffer, 1)).toEqual([1]);
  compiled.destroy();
  destroyFixture(fixture);
}

it('GPUNetworkServiceAreas matches the oracle on random networks', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const facilityNodes = [10, 250, 600, 777, 1000, 1300, 1600, 1900];
  const edges = createRandomNetwork(11, 2000, 8000);
  await runRandomNetwork(device, buildCSR(2000, edges), facilityNodes, true);
  const fractional = edges.map(
    ([from, to, weight]): NetworkEdge => [from, to, Math.fround(weight * 0.37)]
  );
  await runRandomNetwork(device, buildCSR(2000, fractional), facilityNodes, false);
});

it('GPUNetworkServiceAreas re-encodes per-frame facility, count, and limit changes', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const nodeCount = 2000;
  const csr = buildCSR(nodeCount, createRandomNetwork(5, nodeCount, 8000));
  const nodes = [3, 400, 900, 1500];
  const fixture = createServiceFixture(device, csr, nodeCount, {
    facilities: nodes,
    facilityCosts: [0, 0, 0, 0],
    facilityCount: 4,
    costLimit: Infinity,
    maxIterations: 256
  });
  const compiled = fixture.graph.compile();
  const facilities = nodes.map(node => ({node, cost: 0}));
  submitGraph(device, compiled, undefined);
  await expectMatchesOracle(fixture, csr, facilities, 4, Infinity, true);

  // Move a facility.
  facilities[1] = {node: 1234, cost: 0};
  fixture.facilities.write(Uint32Array.from(facilities.map(facility => facility.node)));
  submitGraph(device, compiled, undefined);
  await expectMatchesOracle(fixture, csr, facilities, 4, Infinity, true);

  // Change facility starting costs and the active count.
  facilities[0] = {node: 3, cost: 2};
  fixture.facilityCosts!.write(Float32Array.from([2, 0, 0, 0]));
  fixture.facilityCount!.write(Uint32Array.from([2]));
  submitGraph(device, compiled, undefined);
  await expectMatchesOracle(fixture, csr, facilities, 2, Infinity, true);

  // Change the cost limit.
  fixture.facilityCount!.write(Uint32Array.from([4]));
  fixture.costLimit!.write(Float32Array.from([6]));
  submitGraph(device, compiled, undefined);
  const limited = await expectMatchesOracle(fixture, csr, facilities, 4, 6, true);
  expect(limited.assignments.some(row => row === NONE)).toBe(true);

  // A count larger than the capacity is clamped to the capacity.
  fixture.costLimit!.write(Float32Array.from([Infinity]));
  fixture.facilityCount!.write(Uint32Array.from([100]));
  submitGraph(device, compiled, undefined);
  await expectMatchesOracle(fixture, csr, facilities, 4, Infinity, true);

  // Change weights in place.
  const heavier = buildCSR(
    nodeCount,
    createRandomNetwork(5, nodeCount, 8000).map(
      ([from, to, weight]): NetworkEdge => [from, to, weight + 1]
    )
  );
  fixture.weightsBuffer.write(heavier.weights);
  submitGraph(device, compiled, undefined);
  await expectMatchesOracle(fixture, heavier, facilities, 4, Infinity, true);
  compiled.destroy();
  destroyFixture(fixture);
});

it('GPUNetworkServiceAreas handles no facilities, duplicates, and invalid facilities', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const fixture = createServiceFixture(device, HAND, 11, {
    facilities: [5, 0, 5, 99999],
    facilityCount: 0
  });
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readUint32(fixture.assignmentsBuffer, 11)).toEqual(new Array(11).fill(NONE));
  expect(await readFloat32(fixture.costsBuffer, 11)).toEqual(new Array(11).fill(Infinity));
  expect(await readUint32(fixture.nodeCountsBuffer, 4)).toEqual([0, 0, 0, 0]);
  expect(await readFloat32(fixture.costSumsBuffer, 4)).toEqual([0, 0, 0, 0]);
  expect(await readUint32(fixture.convergedBuffer, 1)).toEqual([1]);

  // Rows 0 and 2 share node 5: the smaller row wins. Row 3 is out of range.
  fixture.facilityCount!.write(Uint32Array.from([4]));
  submitGraph(device, compiled, undefined);
  const facilities = [5, 0, 5, 99999].map(node => ({node, cost: 0}));
  const {assignments} = await expectMatchesOracle(fixture, HAND, facilities, 4, Infinity, true);
  expect(assignments[5]).toBe(0);
  expect(assignments[0]).toBe(1);
  expect(await readUint32(fixture.nodeCountsBuffer, 4)).toEqual([4, 2, 0, 0]);
  compiled.destroy();
  destroyFixture(fixture);

  // Single node with one facility.
  const single = createServiceFixture(device, buildCSR(1, []), 1, {facilities: [0]});
  const singleCompiled = single.graph.compile();
  submitGraph(device, singleCompiled, undefined);
  expect(await readUint32(single.assignmentsBuffer, 1)).toEqual([0]);
  expect(await readUint32(single.nodeCountsBuffer, 1)).toEqual([1]);
  expect(await readUint32(single.convergedBuffer, 1)).toEqual([1]);
  singleCompiled.destroy();
  destroyFixture(single);
});

it('GPUNetworkServiceAreas reports convergence against the iteration limit', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const pathEdges: NetworkEdge[] = Array.from({length: 19}, (_, node) => [node, node + 1, 1]);
  const path = buildCSR(20, pathEdges);
  const fixture = createServiceFixture(device, path, 20, {
    facilities: [0, 19],
    maxIterations: 32,
    activeIterations: 1
  });
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  expect(await readUint32(fixture.convergedBuffer, 1)).toEqual([0]);

  fixture.activeIterations!.write(Uint32Array.from([32]));
  submitGraph(device, compiled, undefined);
  expect(await readUint32(fixture.convergedBuffer, 1)).toEqual([1]);
  const facilities = [
    {node: 0, cost: 0},
    {node: 19, cost: 0}
  ];
  const {assignments} = await expectMatchesOracle(fixture, path, facilities, 2, Infinity, true);
  expect(assignments.slice(0, 19).every(row => row === 0)).toBe(true);
  expect(assignments[19]).toBe(1);

  fixture.activeIterations!.write(Uint32Array.from([0]));
  submitGraph(device, compiled, undefined);
  expect(await readUint32(fixture.convergedBuffer, 1)).toEqual([0]);
  compiled.destroy();
  destroyFixture(fixture);
});
