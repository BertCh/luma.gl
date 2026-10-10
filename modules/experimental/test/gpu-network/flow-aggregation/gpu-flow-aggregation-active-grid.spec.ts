// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {GPUFlowAggregation} from '../../../src/gpu-network/flow-aggregation';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createSeededPoints} from '../../gpu-spatial-analysis/point-density/point-density-oracle';
import {computeFlowAggregation} from './flow-aggregation-oracle';

const BOUNDS: [number, number, number, number] = [0, 0, 12, 9];
const CAPACITY: [number, number] = [12, 9];
const ROW_COUNT = 1500;
const TOP_COUNT = 400;

/** Extent of the nonzero entries of a count column; zeros when none. */
function getNonzeroExtent(counts: readonly number[]): [number, number] {
  const nonzero = counts.filter(count => count > 0);
  return nonzero.length ? [Math.min(...nonzero), Math.max(...nonzero)] : [0, 0];
}

it('GPUFlowAggregation applies a per-frame active grid size and reports the zone count extent', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const origins = createSeededPoints(31, ROW_COUNT, BOUNDS);
  const destinations = createSeededPoints(47, ROW_COUNT, BOUNDS);
  const inputs = [
    createInputBuffer(device, Float32Array.from(origins)),
    createInputBuffer(device, Float32Array.from(destinations))
  ];
  const activeSize = new GPUParameterBuffer(device, {
    id: 'flow-active-size',
    format: 'uint32',
    length: 2,
    values: Uint32Array.from(CAPACITY)
  });
  const zoneCapacity = CAPACITY[0] * CAPACITY[1];
  const outputs = {
    ids: createOutputBuffer(device, TOP_COUNT),
    count: createOutputBuffer(device, 1),
    overflow: createOutputBuffer(device, 1),
    total: createOutputBuffer(device, 1),
    origins: createOutputBuffer(device, TOP_COUNT),
    destinations: createOutputBuffer(device, TOP_COUNT),
    flowCounts: createOutputBuffer(device, TOP_COUNT),
    zoneOut: createOutputBuffer(device, zoneCapacity),
    zoneIn: createOutputBuffer(device, zoneCapacity),
    outExtent: createOutputBuffer(device, 2),
    inExtent: createOutputBuffer(device, 2)
  };
  const graph = new GPUCommandGraph<void>(device, {id: 'flow-active-grid'});
  const view = <Format extends 'uint32' | 'float32'>(
    name: keyof typeof outputs,
    format: Format,
    length: number
  ) => importGraphBuffer(graph, `flow-active-${name}`, outputs[name], format, length);
  graph.add(
    new GPUFlowAggregation({
      id: 'flow-active-grid',
      zones: {
        kind: 'grid',
        bounds: BOUNDS,
        gridSize: CAPACITY,
        activeGridSize: activeSize.importToGraph(graph)
      },
      origins: importGraphBuffer(graph, 'origins', inputs[0], 'float32x2', ROW_COUNT),
      destinations: importGraphBuffer(graph, 'destinations', inputs[1], 'float32x2', ROW_COUNT),
      pairCapacity: 4096,
      output: {
        ids: view('ids', 'uint32', TOP_COUNT),
        count: view('count', 'uint32', 1),
        overflow: view('overflow', 'uint32', 1),
        requiredCount: view('total', 'uint32', 1)
      },
      flowOriginZoneIds: view('origins', 'uint32', TOP_COUNT),
      flowDestinationZoneIds: view('destinations', 'uint32', TOP_COUNT),
      flowCounts: view('flowCounts', 'uint32', TOP_COUNT),
      zoneOutCounts: view('zoneOut', 'uint32', zoneCapacity),
      zoneInCounts: view('zoneIn', 'uint32', zoneCapacity),
      zoneOutCountExtent: view('outExtent', 'float32', 2),
      zoneInCountExtent: view('inExtent', 'float32', 2)
    })
  );
  const compiled = graph.compile();

  for (const size of [
    [12, 9],
    [6, 5],
    [3, 3],
    [20, 20],
    [1, 1]
  ] as const) {
    activeSize.write(Uint32Array.from(size));
    submitGraph(device, compiled, undefined);
    // Sizes above the capacity clamp to it.
    const active: [number, number] = [
      Math.min(size[0], CAPACITY[0]),
      Math.min(size[1], CAPACITY[1])
    ];
    const oracle = computeFlowAggregation({
      zones: {kind: 'grid', bounds: BOUNDS, gridSize: active},
      origins,
      destinations
    });
    const zoneCount = active[0] * active[1];
    const zoneOut = await readUint32(outputs.zoneOut, zoneCapacity);
    const zoneIn = await readUint32(outputs.zoneIn, zoneCapacity);
    expect(zoneOut.slice(0, zoneCount)).toEqual(oracle.zoneOutCounts);
    expect(zoneIn.slice(0, zoneCount)).toEqual(oracle.zoneInCounts);
    expect(zoneOut.slice(zoneCount).every(count => count === 0)).toBe(true);
    expect(zoneIn.slice(zoneCount).every(count => count === 0)).toBe(true);
    expect(Array.from(await readFloat32(outputs.outExtent, 2))).toEqual(
      getNonzeroExtent(oracle.zoneOutCounts)
    );
    expect(Array.from(await readFloat32(outputs.inExtent, 2))).toEqual(
      getNonzeroExtent(oracle.zoneInCounts)
    );
    const count = (await readUint32(outputs.count, 1))[0];
    expect(count).toBe(Math.min(oracle.flows.length, TOP_COUNT));
    const flowOrigins = await readUint32(outputs.origins, TOP_COUNT);
    const flowDestinations = await readUint32(outputs.destinations, TOP_COUNT);
    const flowCounts = await readUint32(outputs.flowCounts, TOP_COUNT);
    for (let rank = 0; rank < count; rank++) {
      expect([flowOrigins[rank], flowDestinations[rank], flowCounts[rank]]).toEqual([
        oracle.flows[rank].originZone,
        oracle.flows[rank].destinationZone,
        oracle.flows[rank].count
      ]);
    }
  }

  compiled.destroy();
  activeSize.destroy();
  for (const buffer of [...inputs, ...Object.values(outputs)]) buffer.destroy();
});

it('GPUFlowAggregation rejects an activeGridSize view that is too short', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const graph = new GPUCommandGraph<void>(device, {id: 'flow-active-invalid'});
  const buffer = createOutputBuffer(device, 4);
  const one = importGraphBuffer(graph, 'one', buffer, 'uint32', 1);
  const ids = importGraphBuffer(graph, 'ids', buffer, 'uint32', 2);
  expect(
    () =>
      new GPUFlowAggregation({
        zones: {kind: 'grid', bounds: BOUNDS, gridSize: CAPACITY, activeGridSize: one},
        origins: importGraphBuffer(graph, 'o', buffer, 'float32x2', 1),
        destinations: importGraphBuffer(graph, 'd', buffer, 'float32x2', 1),
        pairCapacity: 8,
        output: {ids, count: one, overflow: one}
      })
  ).toThrow(/activeGridSize/);
  buffer.destroy();
});
