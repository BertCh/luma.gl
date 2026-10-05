// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUFlowAggregation,
  GPU_FLOW_AGGREGATION_NO_ZONE,
  type GPUFlowAggregationProps
} from '../../../src/gpu-network/flow-aggregation';
import {getGPUPointDensityHexagonGridSize} from '../../../src/geospatial/point-density';
import {
  getGPUTimeWindowParameterValues,
  getGPUTimeWindowWordParameterValues,
  getInt64TimeWords,
  GPU_TIME_WINDOW_PARAMETER_LENGTH,
  GPU_TIME_WORD_WINDOW_PARAMETER_LENGTH
} from '../../../src/gpu-dataframe/time-window-filter';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {findNearestHexagon, createSeededPoints} from '../../geospatial/point-density/point-density-oracle';
import {
  computeFlowAggregation,
  createSeededRandom,
  type FlowOracleResult
} from './flow-aggregation-oracle';

const NO_ZONE = GPU_FLOW_AGGREGATION_NO_ZONE;
const GARBAGE = 0xdeadbeef;

type FlowResult = {
  ids: number[];
  count: number;
  totalCount: number;
  overflow: number;
  pairOverflow: number;
  drawInstanceCount: number;
  originZones: number[];
  destinationZones: number[];
  counts: number[];
  weights: number[];
  zoneOutCounts: number[];
  zoneInCounts: number[];
  zoneOutWeights: number[];
  zoneInWeights: number[];
};

type Harness = {
  compiled: CompiledGPUCommandGraph<void>;
  submit(): void;
  read(): Promise<FlowResult>;
  destroy(): void;
};

type HarnessOptions = {
  id: string;
  zoneCount: number;
  topCount: number;
  pairCapacity: number;
  hasWeights: boolean;
  maxProbeCount?: number;
  createProps: (
    graph: GPUCommandGraph<void>
  ) => Pick<GPUFlowAggregationProps, 'zones'> & Partial<GPUFlowAggregationProps>;
};

/** Builds one flow-aggregation graph with every output imported, scrambled before each submit. */
function createHarness(device: Device, options: HarnessOptions): Harness {
  const {topCount, zoneCount, hasWeights} = options;
  const buffers: Buffer[] = [];
  const outputs: {buffer: Buffer; length: number}[] = [];
  const graph = new GPUCommandGraph<void>(device, {id: options.id});
  const createOutput = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    length: number
  ) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    outputs.push({buffer, length: Math.max(length, 1)});
    return {
      buffer,
      view: importGraphBuffer(graph, `${options.id}-${name}`, buffer, format, length)
    };
  };
  const ids = createOutput('ids', 'uint32', topCount);
  const count = createOutput('count', 'uint32', 1);
  const overflow = createOutput('overflow', 'uint32', 1);
  const totalCount = createOutput('total', 'uint32', 1);
  const pairOverflow = createOutput('pair-overflow', 'uint32', 1);
  const drawInstanceCount = createOutput('draw', 'uint32', 1);
  const originZones = createOutput('origin-zones', 'uint32', topCount);
  const destinationZones = createOutput('destination-zones', 'uint32', topCount);
  const counts = createOutput('counts', 'uint32', topCount);
  const weights = createOutput('weights', 'float32', topCount);
  const zoneOutCounts = createOutput('zone-out-counts', 'uint32', zoneCount);
  const zoneInCounts = createOutput('zone-in-counts', 'uint32', zoneCount);
  const zoneOutWeights = createOutput('zone-out-weights', 'float32', zoneCount);
  const zoneInWeights = createOutput('zone-in-weights', 'float32', zoneCount);

  const recipe = new GPUFlowAggregation({
    id: options.id,
    pairCapacity: options.pairCapacity,
    maxProbeCount: options.maxProbeCount,
    ...options.createProps(graph),
    output: {
      ids: ids.view,
      count: count.view,
      overflow: overflow.view,
      totalCount: totalCount.view
    },
    pairOverflow: pairOverflow.view,
    drawInstanceCount: drawInstanceCount.view,
    flowOriginZoneIds: originZones.view,
    flowDestinationZoneIds: destinationZones.view,
    flowCounts: counts.view,
    flowWeights: weights.view,
    zoneOutCounts: zoneOutCounts.view,
    zoneInCounts: zoneInCounts.view,
    ...(hasWeights
      ? {
          zoneOutWeights: zoneOutWeights.view,
          zoneInWeights: zoneInWeights.view
        }
      : {})
  });
  graph.add(recipe);
  const compiled = graph.compile();
  return {
    compiled,
    submit: () => {
      for (const output of outputs) {
        output.buffer.write(new Uint32Array(output.length).fill(GARBAGE));
      }
      submitGraph(device, compiled, undefined);
    },
    read: async () => ({
      ids: await readUint32(ids.buffer, topCount),
      count: (await readUint32(count.buffer, 1))[0],
      totalCount: (await readUint32(totalCount.buffer, 1))[0],
      overflow: (await readUint32(overflow.buffer, 1))[0],
      pairOverflow: (await readUint32(pairOverflow.buffer, 1))[0],
      drawInstanceCount: (await readUint32(drawInstanceCount.buffer, 1))[0],
      originZones: await readUint32(originZones.buffer, topCount),
      destinationZones: await readUint32(destinationZones.buffer, topCount),
      counts: await readUint32(counts.buffer, topCount),
      weights: await readFloat32(weights.buffer, topCount),
      zoneOutCounts: await readUint32(zoneOutCounts.buffer, zoneCount),
      zoneInCounts: await readUint32(zoneInCounts.buffer, zoneCount),
      zoneOutWeights: await readFloat32(zoneOutWeights.buffer, zoneCount),
      zoneInWeights: await readFloat32(zoneInWeights.buffer, zoneCount)
    }),
    destroy: () => {
      compiled.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

/** Compares every output against the oracle for a table that did not overflow. */
function expectMatchesOracle(
  result: FlowResult,
  oracle: FlowOracleResult,
  topCount: number,
  hasWeights: boolean
): void {
  const distinct = oracle.flows.length;
  const count = Math.min(distinct, topCount);
  expect(result.count).toBe(count);
  expect(result.drawInstanceCount).toBe(count);
  expect(result.totalCount).toBe(distinct);
  expect(result.pairOverflow).toBe(0);
  expect(result.overflow).toBe(distinct > topCount ? 1 : 0);
  for (let rank = 0; rank < topCount; rank++) {
    const flow = oracle.flows[rank];
    if (rank < count) {
      expect([
        result.ids[rank],
        result.originZones[rank],
        result.destinationZones[rank],
        result.counts[rank],
        result.weights[rank]
      ]).toEqual([flow.pairKey, flow.originZone, flow.destinationZone, flow.count, flow.weight]);
    } else {
      expect([
        result.ids[rank],
        result.originZones[rank],
        result.destinationZones[rank],
        result.counts[rank],
        result.weights[rank]
      ]).toEqual([NO_ZONE, NO_ZONE, NO_ZONE, 0, 0]);
    }
  }
  expectZoneTotals(result, oracle, hasWeights);
}

function expectZoneTotals(result: FlowResult, oracle: FlowOracleResult, hasWeights: boolean): void {
  expect(result.zoneOutCounts).toEqual(oracle.zoneOutCounts);
  expect(result.zoneInCounts).toEqual(oracle.zoneInCounts);
  if (hasWeights) {
    expect(result.zoneOutWeights).toEqual(oracle.zoneOutWeights);
    expect(result.zoneInWeights).toEqual(oracle.zoneInWeights);
  }
}

/** Seeded random OD zone IDs. */
function createZoneIds(seed: number, rowCount: number, zoneCount: number, outOfRange = 0) {
  const next = createSeededRandom(seed);
  const origins: number[] = [];
  const destinations: number[] = [];
  for (let row = 0; row < rowCount; row++) {
    origins.push(Math.floor(next() * (zoneCount + outOfRange)));
    destinations.push(Math.floor(next() * (zoneCount + outOfRange)));
  }
  return {origins, destinations};
}

/** Integer weights 1..5 with a NaN every 17th row. */
function createWeights(seed: number, rowCount: number): number[] {
  const next = createSeededRandom(seed);
  return Array.from({length: rowCount}, (_, row) =>
    row % 17 === 5 ? Number.NaN : 1 + Math.floor(next() * 5)
  );
}

it('GPUFlowAggregation aggregates grid zones against the oracle (weighted and unweighted)', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const bounds: [number, number, number, number] = [0, 0, 6, 5];
  const gridSize: [number, number] = [6, 5];
  const rowCount = 2000;
  const origins = createSeededPoints(11, rowCount, bounds);
  const destinations = createSeededPoints(23, rowCount, bounds);
  // Rejected rows: out of bounds, NaN, and exact maximum edges (which are accepted).
  origins.splice(0, 8, 7, 1, 1, -1, Number.NaN, 2, 3, Number.NaN);
  destinations.splice(10, 4, 6, 5, 0, 0);
  const weightValues = createWeights(5, rowCount);

  for (const useWeights of [true, false]) {
    for (const topCount of [50, 1024]) {
      const buffers = [
        createInputBuffer(device, Float32Array.from(origins)),
        createInputBuffer(device, Float32Array.from(destinations)),
        createInputBuffer(device, Float32Array.from(weightValues))
      ];
      const harness = createHarness(device, {
        id: 'flow-grid',
        zoneCount: 30,
        topCount,
        pairCapacity: 1024,
        hasWeights: useWeights,
        createProps: graph => ({
          zones: {kind: 'grid', bounds, gridSize},
          origins: importGraphBuffer(graph, 'origins', buffers[0], 'float32x2', rowCount),
          destinations: importGraphBuffer(graph, 'destinations', buffers[1], 'float32x2', rowCount),
          weights: useWeights
            ? importGraphBuffer(graph, 'weights', buffers[2], 'float32', rowCount)
            : undefined
        })
      });
      harness.submit();
      const oracle = computeFlowAggregation({
        zones: {kind: 'grid', bounds, gridSize},
        origins,
        destinations,
        weights: useWeights ? weightValues : undefined
      });
      expect(oracle.flows.length).toBeGreaterThan(50);
      const result = await harness.read();
      expectMatchesOracle(result, oracle, topCount, useWeights);
      if (topCount === 50) {
        expect(result.overflow).toBe(1);
        expect(result.count).toBe(50);
        expect(result.totalCount).toBe(oracle.flows.length);
      }
      harness.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  }
});

it('GPUFlowAggregation orders equal weights by pair key', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Pairs (origin, destination) with two rows each, inserted in scrambled order.
  const pairs = [
    [3, 1],
    [0, 2],
    [2, 2],
    [1, 3],
    [0, 2],
    [3, 1],
    [2, 2],
    [1, 3],
    [0, 0]
  ];
  const originIds = pairs.map(pair => pair[0]);
  const destinationIds = pairs.map(pair => pair[1]);
  const buffers = [
    createInputBuffer(device, Uint32Array.from(originIds)),
    createInputBuffer(device, Uint32Array.from(destinationIds))
  ];
  const harness = createHarness(device, {
    id: 'flow-ties',
    zoneCount: 4,
    topCount: 8,
    pairCapacity: 8,
    hasWeights: false,
    createProps: graph => ({
      zones: {kind: 'ids', zoneCount: 4},
      originZoneIds: importGraphBuffer(graph, 'origin-ids', buffers[0], 'uint32', pairs.length),
      destinationZoneIds: importGraphBuffer(
        graph,
        'destination-ids',
        buffers[1],
        'uint32',
        pairs.length
      )
    })
  });
  harness.submit();
  const result = await harness.read();
  const oracle = computeFlowAggregation({
    zones: {kind: 'ids', zoneCount: 4},
    originZoneIds: originIds,
    destinationZoneIds: destinationIds
  });
  expectMatchesOracle(result, oracle, 8, false);
  // Four pairs of count two ordered by key (0,2)=2, (1,3)=7, (2,2)=10, (3,1)=13, then (0,0).
  expect(result.ids.slice(0, 5)).toEqual([2, 7, 10, 13, 0]);
  expect(result.counts.slice(0, 5)).toEqual([2, 2, 2, 2, 1]);
  harness.destroy();
  for (const buffer of buffers) buffer.destroy();
});

function createHexagonPoints(radii: number[]): number[] {
  const candidates = createSeededPoints(7, 200, [0, 0, 6, 6]);
  const points: number[] = [];
  for (let index = 0; index < candidates.length / 2 && points.length < 128; index++) {
    const [x, y] = [candidates[index * 2], candidates[index * 2 + 1]];
    const isClear = radii.every(radius => {
      const nearest = findNearestHexagon(x, y, 0, 0, radius, 10, 10);
      return nearest.secondNearest - nearest.nearest >= 1e-3 * radius;
    });
    if (isClear) {
      points.push(x, y);
    }
  }
  return [...points, Number.NaN, Number.NaN, 7, 7];
}

it('GPUFlowAggregation bins hexagon zones with per-frame radius and bounds', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const origins = createHexagonPoints([1, 1.4, 0.6]);
  const rowCount = origins.length / 2;
  const destinations: number[] = [];
  for (let row = rowCount - 1; row >= 0; row--) {
    destinations.push(origins[row * 2], origins[row * 2 + 1]);
  }
  const weightValues = Array.from({length: rowCount}, (_, row) => 1 + (row % 4));
  const gridSize = getGPUPointDensityHexagonGridSize([0, 0, 6, 6], 1);
  const zoneCount = gridSize[0] * gridSize[1];
  const radius = new GPUParameterBuffer(device, {
    id: 'flow-radius',
    format: 'float32',
    length: 1,
    values: Float32Array.from([1])
  });
  const bounds = new GPUParameterBuffer(device, {
    id: 'flow-bounds',
    format: 'float32',
    length: 4,
    values: Float32Array.from([0, 0, 6, 6])
  });
  const buffers = [
    createInputBuffer(device, Float32Array.from(origins)),
    createInputBuffer(device, Float32Array.from(destinations)),
    createInputBuffer(device, Float32Array.from(weightValues))
  ];
  const harness = createHarness(device, {
    id: 'flow-hexagon',
    zoneCount,
    topCount: 64,
    pairCapacity: 512,
    hasWeights: true,
    createProps: graph => ({
      zones: {
        kind: 'hexagon',
        bounds: bounds.importToGraph(graph),
        gridSize,
        radius: radius.importToGraph(graph)
      },
      origins: importGraphBuffer(graph, 'origins', buffers[0], 'float32x2', rowCount),
      destinations: importGraphBuffer(graph, 'destinations', buffers[1], 'float32x2', rowCount),
      weights: importGraphBuffer(graph, 'weights', buffers[2], 'float32', rowCount)
    })
  });
  for (const [radiusValue, boundsValue] of [
    [1, [0, 0, 6, 6]],
    [1.4, [0, 0, 6, 6]],
    [0.6, [0, 0, 6, 6]],
    [1, [0, 0, 4, 4]]
  ] as const) {
    radius.write(Float32Array.from([radiusValue]));
    bounds.write(Float32Array.from(boundsValue));
    harness.submit();
    const oracle = computeFlowAggregation({
      zones: {
        kind: 'hexagon',
        bounds: [...boundsValue] as [number, number, number, number],
        gridSize,
        radius: radiusValue
      },
      origins,
      destinations,
      weights: weightValues
    });
    expectMatchesOracle(await harness.read(), oracle, 64, true);
  }
  harness.destroy();
  radius.destroy();
  bounds.destroy();
  for (const buffer of buffers) buffer.destroy();
});

it('GPUFlowAggregation applies mask, time window, and self-flow exclusion per frame', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const rowCount = 500;
  const zoneCount = 8;
  const {origins, destinations} = createZoneIds(3, rowCount, zoneCount, 2);
  const timestamps = Array.from({length: rowCount}, (_, row) => (row * 7) % 100);
  const weightValues = createWeights(9, rowCount);
  const maskValues = Array.from({length: rowCount}, (_, row) => (row % 3 === 0 ? 0 : 1));
  const window = new GPUParameterBuffer(device, {
    id: 'flow-window',
    format: 'float32',
    length: GPU_TIME_WINDOW_PARAMETER_LENGTH,
    values: getGPUTimeWindowParameterValues({start: 10, end: 60})
  });
  const buffers = [
    createInputBuffer(device, Uint32Array.from(origins)),
    createInputBuffer(device, Uint32Array.from(destinations)),
    createInputBuffer(device, Float32Array.from(weightValues)),
    createInputBuffer(device, Uint32Array.from(maskValues)),
    createInputBuffer(device, Float32Array.from(timestamps))
  ];
  const harness = createHarness(device, {
    id: 'flow-ids',
    zoneCount,
    topCount: 64,
    pairCapacity: 64,
    hasWeights: true,
    createProps: graph => ({
      zones: {kind: 'ids', zoneCount},
      originZoneIds: importGraphBuffer(graph, 'origin-ids', buffers[0], 'uint32', rowCount),
      destinationZoneIds: importGraphBuffer(
        graph,
        'destination-ids',
        buffers[1],
        'uint32',
        rowCount
      ),
      weights: importGraphBuffer(graph, 'weights', buffers[2], 'float32', rowCount),
      mask: importGraphBuffer(graph, 'mask', buffers[3], 'uint32', rowCount),
      timeWindow: {
        timestamps: importGraphBuffer(graph, 'timestamps', buffers[4], 'float32', rowCount),
        window: window.importToGraph(graph)
      },
      excludeSelfFlows: true
    })
  });
  const verify = async (start: number, end: number, mask: number[]) => {
    harness.submit();
    const oracle = computeFlowAggregation({
      zones: {kind: 'ids', zoneCount},
      originZoneIds: origins,
      destinationZoneIds: destinations,
      weights: weightValues,
      mask,
      timeWindow: {timestamps, start, end},
      excludeSelfFlows: true
    });
    expectMatchesOracle(await harness.read(), oracle, 64, true);
    return oracle;
  };
  const first = await verify(10, 60, maskValues);
  expect(first.flows.length).toBeGreaterThan(0);
  window.write(getGPUTimeWindowParameterValues({start: 40, end: 90}));
  const second = await verify(40, 90, maskValues);
  expect(second.flows).not.toEqual(first.flows);
  const newMask = maskValues.map(value => 1 - value);
  buffers[3].write(Uint32Array.from(newMask));
  await verify(40, 90, newMask);
  window.write(getGPUTimeWindowParameterValues({start: 200, end: 100}));
  const empty = await verify(200, 100, newMask);
  expect(empty.flows.length).toBe(0);

  harness.destroy();
  window.destroy();
  for (const buffer of buffers) buffer.destroy();
});

it('GPUFlowAggregation reports pair table overflow and keeps zone totals exact', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const rowCount = 300;
  const zoneCount = 16;
  const {origins, destinations} = createZoneIds(41, rowCount, zoneCount);
  const weightValues = createWeights(2, rowCount);
  const buffers = [
    createInputBuffer(device, Uint32Array.from(origins)),
    createInputBuffer(device, Uint32Array.from(destinations)),
    createInputBuffer(device, Float32Array.from(weightValues))
  ];
  const topCount = 6;
  const harness = createHarness(device, {
    id: 'flow-overflow',
    zoneCount,
    topCount,
    pairCapacity: 4,
    hasWeights: true,
    createProps: graph => ({
      zones: {kind: 'ids', zoneCount},
      originZoneIds: importGraphBuffer(graph, 'origin-ids', buffers[0], 'uint32', rowCount),
      destinationZoneIds: importGraphBuffer(
        graph,
        'destination-ids',
        buffers[1],
        'uint32',
        rowCount
      ),
      weights: importGraphBuffer(graph, 'weights', buffers[2], 'float32', rowCount)
    })
  });
  harness.submit();
  const result = await harness.read();
  const oracle = computeFlowAggregation({
    zones: {kind: 'ids', zoneCount},
    originZoneIds: origins,
    destinationZoneIds: destinations,
    weights: weightValues
  });
  expect(oracle.flows.length).toBeGreaterThan(4);
  expect(result.pairOverflow).toBe(1);
  expect(result.overflow).toBe(1);
  expect(result.totalCount).toBe(4);
  expect(result.count).toBe(4);
  expect(result.drawInstanceCount).toBe(4);
  for (let rank = 0; rank < topCount; rank++) {
    if (rank >= 4) {
      expect([result.ids[rank], result.counts[rank], result.weights[rank]]).toEqual([
        NO_ZONE,
        0,
        0
      ]);
      continue;
    }
    const flow = oracle.flows.find(candidate => candidate.pairKey === result.ids[rank]);
    expect(flow).toBeDefined();
    expect([result.counts[rank], result.weights[rank]]).toEqual([flow!.count, flow!.weight]);
    expect([result.originZones[rank], result.destinationZones[rank]]).toEqual([
      flow!.originZone,
      flow!.destinationZone
    ]);
    if (rank > 0) {
      const earlier = [result.weights[rank - 1], result.ids[rank - 1]];
      expect(
        earlier[0] > result.weights[rank] ||
          (earlier[0] === result.weights[rank] && earlier[1] < result.ids[rank])
      ).toBe(true);
    }
  }
  expectZoneTotals(result, oracle, true);
  harness.destroy();
  for (const buffer of buffers) buffer.destroy();
});

it('GPUFlowAggregation handles empty input', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const buffers = [
    createInputBuffer(device, Uint32Array.from([0])),
    createInputBuffer(device, Uint32Array.from([0])),
    createInputBuffer(device, Float32Array.from([0]))
  ];
  const harness = createHarness(device, {
    id: 'flow-empty',
    zoneCount: 5,
    topCount: 4,
    pairCapacity: 8,
    hasWeights: true,
    createProps: graph => ({
      zones: {kind: 'ids', zoneCount: 5},
      originZoneIds: importGraphBuffer(graph, 'origin-ids', buffers[0], 'uint32', 0),
      destinationZoneIds: importGraphBuffer(graph, 'destination-ids', buffers[1], 'uint32', 0),
      weights: importGraphBuffer(graph, 'weights', buffers[2], 'float32', 0)
    })
  });
  harness.submit();
  const result = await harness.read();
  expectMatchesOracle(
    result,
    computeFlowAggregation({
      zones: {kind: 'ids', zoneCount: 5},
      originZoneIds: [],
      destinationZoneIds: []
    }),
    4,
    true
  );
  expect(result.count).toBe(0);
  expect(result.overflow).toBe(0);
  harness.destroy();
  for (const buffer of buffers) buffer.destroy();
});

it('GPUFlowAggregation handles empty grid input', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const buffers = [
    createInputBuffer(device, Float32Array.from([0, 0])),
    createInputBuffer(device, Float32Array.from([0, 0]))
  ];
  const harness = createHarness(device, {
    id: 'flow-empty-grid',
    zoneCount: 4,
    topCount: 2,
    pairCapacity: 2,
    hasWeights: false,
    createProps: graph => ({
      zones: {kind: 'grid', bounds: [0, 0, 1, 1], gridSize: [2, 2]},
      origins: importGraphBuffer(graph, 'origins', buffers[0], 'float32x2', 0),
      destinations: importGraphBuffer(graph, 'destinations', buffers[1], 'float32x2', 0)
    })
  });
  harness.submit();
  const result = await harness.read();
  expect(result.count).toBe(0);
  expect(result.overflow).toBe(0);
  expect(result.zoneOutCounts).toEqual([0, 0, 0, 0]);
  expect(result.ids).toEqual([NO_ZONE, NO_ZONE]);
  harness.destroy();
  for (const buffer of buffers) buffer.destroy();
});

it('GPUFlowAggregation handles a single row', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const buffers = [
    createInputBuffer(device, Uint32Array.from([2])),
    createInputBuffer(device, Uint32Array.from([1])),
    createInputBuffer(device, Float32Array.from([7]))
  ];
  const harness = createHarness(device, {
    id: 'flow-single',
    zoneCount: 3,
    topCount: 4,
    pairCapacity: 1,
    hasWeights: true,
    createProps: graph => ({
      zones: {kind: 'ids', zoneCount: 3},
      originZoneIds: importGraphBuffer(graph, 'origin-ids', buffers[0], 'uint32', 1),
      destinationZoneIds: importGraphBuffer(graph, 'destination-ids', buffers[1], 'uint32', 1),
      weights: importGraphBuffer(graph, 'weights', buffers[2], 'float32', 1)
    })
  });
  harness.submit();
  const result = await harness.read();
  expectMatchesOracle(
    result,
    computeFlowAggregation({
      zones: {kind: 'ids', zoneCount: 3},
      originZoneIds: [2],
      destinationZoneIds: [1],
      weights: [7]
    }),
    4,
    true
  );
  expect(result.ids).toEqual([7, NO_ZONE, NO_ZONE, NO_ZONE]);
  expect(result.weights[0]).toBe(7);
  harness.destroy();
  for (const buffer of buffers) buffer.destroy();
});

it('GPUFlowAggregation gates rows exactly with Int64 word timestamps', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const rowCount = 900;
  const zoneCount = 6;
  const {origins, destinations} = createZoneIds(31, rowCount, zoneCount, 1);
  const weightValues = createWeights(13, rowCount);
  const TWO_32 = 2n ** 32n;
  // Windows across a low-word wrap (k * 2^32), at a real epoch (~1.7e12 ms), and high word zero.
  const windows = [
    {start: 3n * TWO_32 - 4n, end: 3n * TWO_32 + 6n},
    {start: 1_700_000_000_000n, end: 1_700_000_000_000n + 86_400_000n},
    {start: 395n * TWO_32 - 1000n, end: 396n * TWO_32 + 1000n},
    {start: 100n, end: 100n},
    {start: 50n, end: 10n}
  ];
  const timestamps: bigint[] = [];
  for (const {start, end} of windows) {
    for (const edge of [start, end]) {
      timestamps.push(edge - 1n, edge, edge + 1n);
    }
  }
  const next = createSeededRandom(77);
  while (timestamps.length < rowCount) {
    const {start, end} = windows[Math.floor(next() * windows.length)];
    timestamps.push(start - 2000n + BigInt(Math.floor(next() * Number(end - start + 4000n))));
  }
  const timestampWords = getInt64TimeWords(BigInt64Array.from(timestamps));
  const window = new GPUParameterBuffer(device, {
    id: 'flow-word-window',
    format: 'uint32',
    length: GPU_TIME_WORD_WINDOW_PARAMETER_LENGTH,
    values: getGPUTimeWindowWordParameterValues(windows[0])
  });
  const buffers = [
    createInputBuffer(device, Uint32Array.from(origins)),
    createInputBuffer(device, Uint32Array.from(destinations)),
    createInputBuffer(device, Float32Array.from(weightValues)),
    createInputBuffer(device, timestampWords)
  ];
  const harness = createHarness(device, {
    id: 'flow-word-time',
    zoneCount,
    topCount: 64,
    pairCapacity: 64,
    hasWeights: true,
    createProps: graph => ({
      zones: {kind: 'ids', zoneCount},
      originZoneIds: importGraphBuffer(graph, 'origin-ids', buffers[0], 'uint32', rowCount),
      destinationZoneIds: importGraphBuffer(
        graph,
        'destination-ids',
        buffers[1],
        'uint32',
        rowCount
      ),
      weights: importGraphBuffer(graph, 'weights', buffers[2], 'float32', rowCount),
      timeWindow: {
        timestamps: importGraphBuffer(graph, 'timestamps', buffers[3], 'uint32x2', rowCount),
        window: window.importToGraph(graph)
      }
    })
  });
  let previousTotal = -1;
  for (const {start, end} of windows) {
    window.write(getGPUTimeWindowWordParameterValues({start, end}));
    harness.submit();
    const oracle = computeFlowAggregation({
      zones: {kind: 'ids', zoneCount},
      originZoneIds: origins,
      destinationZoneIds: destinations,
      weights: weightValues,
      wordTimeWindow: {timestamps, start, end}
    });
    const result = await harness.read();
    expectMatchesOracle(result, oracle, 64, true);
    const accepted = result.zoneOutCounts.reduce((sum, value) => sum + value, 0);
    if (end >= start) {
      expect(accepted).toBeGreaterThan(0);
      expect(accepted).toBeLessThan(rowCount);
    } else {
      expect(accepted).toBe(0);
    }
    expect(accepted).not.toBe(previousTotal);
    previousTotal = accepted;
  }
  harness.destroy();
  window.destroy();
  for (const buffer of buffers) buffer.destroy();
});

/** Weights that expose summation order: 1e8, 1, fractions, and non-finite values. */
function createOrderSensitiveWeights(seed: number, rowCount: number): Float32Array {
  const next = createSeededRandom(seed);
  return Float32Array.from({length: rowCount}, (_, row) => {
    if (row % 53 === 7) return Number.NaN;
    const draw = next();
    if (draw < 0.3) return 1e8;
    if (draw < 0.6) return 1;
    return Math.fround(next() * 3.3);
  });
}

function getFloat32Bits(values: readonly number[]): number[] {
  return Array.from(new Uint32Array(Float32Array.from(values).buffer));
}

it('GPUFlowAggregation sorted sums are bitwise stable and match the fixed-tree oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const rowCount = 4000;
  const zoneCount = 3;
  const topCount = 16;
  const {origins, destinations} = createZoneIds(41, rowCount, zoneCount, 1);
  const weightValues = createOrderSensitiveWeights(19, rowCount);
  const weightArray = Array.from(weightValues);
  const buffers = [
    createInputBuffer(device, Uint32Array.from(origins)),
    createInputBuffer(device, Uint32Array.from(destinations)),
    createInputBuffer(device, weightValues)
  ];
  const createFlowHarness = (sumOrder: 'atomic' | 'sorted') =>
    createHarness(device, {
      id: `flow-${sumOrder}-sums`,
      zoneCount,
      topCount,
      pairCapacity: 16,
      hasWeights: true,
      createProps: graph => ({
        sumOrder,
        zones: {kind: 'ids', zoneCount},
        originZoneIds: importGraphBuffer(graph, 'origin-ids', buffers[0], 'uint32', rowCount),
        destinationZoneIds: importGraphBuffer(
          graph,
          'destination-ids',
          buffers[1],
          'uint32',
          rowCount
        ),
        weights: importGraphBuffer(graph, 'weights', buffers[2], 'float32', rowCount)
      })
    });
  const oracle = computeFlowAggregation({
    zones: {kind: 'ids', zoneCount},
    originZoneIds: origins,
    destinationZoneIds: destinations,
    weights: weightArray,
    sumMode: 'tree'
  });
  const sequentialOracle = computeFlowAggregation({
    zones: {kind: 'ids', zoneCount},
    originZoneIds: origins,
    destinationZoneIds: destinations,
    weights: weightArray
  });

  const sorted = createFlowHarness('sorted');
  let firstBits: number[] | undefined;
  for (let run = 0; run < 5; run++) {
    sorted.submit();
    const result = await sorted.read();
    expectMatchesOracle(result, oracle, topCount, true);
    // Tight agreement with the float64 sequential sum.
    for (let rank = 0; rank < result.count; rank++) {
      const exact = sequentialOracle.flows.find(flow => flow.pairKey === result.ids[rank])!;
      expect(Math.abs(result.weights[rank] - exact.weight)).toBeLessThanOrEqual(
        Math.abs(exact.weight) * 1e-3 + 64
      );
    }
    const bits = getFloat32Bits([
      ...result.weights,
      ...result.zoneOutWeights,
      ...result.zoneInWeights
    ]);
    firstBits ??= bits;
    expect(bits).toEqual(firstBits);
  }
  sorted.destroy();

  // Informational: how far the atomic sums are from the fixed-order sums for this input.
  const atomic = createFlowHarness('atomic');
  let differing = 0;
  for (let run = 0; run < 3; run++) {
    atomic.submit();
    const result = await atomic.read();
    differing += getFloat32Bits(result.zoneOutWeights).filter(
      (bits, zone) => bits !== getFloat32Bits(oracle.zoneOutWeights)[zone]
    ).length;
    expect(result.zoneOutCounts).toEqual(oracle.zoneOutCounts);
  }
  console.log(
    `flow sums: atomic zoneOutWeights differing from fixed-tree bits over 3 runs: ${differing}`
  );
  atomic.destroy();
  for (const buffer of buffers) buffer.destroy();
});

it('GPUFlowAggregation sorted sums keep zone totals exact when the pair table overflows', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const rowCount = 3000;
  const zoneCount = 40;
  const {origins, destinations} = createZoneIds(5, rowCount, zoneCount, 2);
  const weightValues = createOrderSensitiveWeights(23, rowCount);
  const oracle = computeFlowAggregation({
    zones: {kind: 'ids', zoneCount},
    originZoneIds: origins,
    destinationZoneIds: destinations,
    weights: Array.from(weightValues),
    sumMode: 'tree'
  });
  const buffers = [
    createInputBuffer(device, Uint32Array.from(origins)),
    createInputBuffer(device, Uint32Array.from(destinations)),
    createInputBuffer(device, weightValues)
  ];
  const harness = createHarness(device, {
    id: 'flow-sorted-overflow',
    zoneCount,
    topCount: 8,
    pairCapacity: 64,
    maxProbeCount: 8,
    hasWeights: true,
    createProps: graph => ({
      sumOrder: 'sorted',
      zones: {kind: 'ids', zoneCount},
      originZoneIds: importGraphBuffer(graph, 'origin-ids', buffers[0], 'uint32', rowCount),
      destinationZoneIds: importGraphBuffer(
        graph,
        'destination-ids',
        buffers[1],
        'uint32',
        rowCount
      ),
      weights: importGraphBuffer(graph, 'weights', buffers[2], 'float32', rowCount)
    })
  });
  harness.submit();
  const result = await harness.read();
  expect(result.pairOverflow).toBe(1);
  expect(result.overflow).toBe(1);
  expectZoneTotals(result, oracle, true);
  // Reported flow sums are each pair's fixed-tree sum, in descending order.
  for (let rank = 0; rank < result.count; rank++) {
    const flow = oracle.flows.find(candidate => candidate.pairKey === result.ids[rank])!;
    expect(result.weights[rank]).toBe(flow.weight);
    if (rank > 0) expect(result.weights[rank]).toBeLessThanOrEqual(result.weights[rank - 1]);
  }
  harness.destroy();
  for (const buffer of buffers) buffer.destroy();
});

it('GPUFlowAggregation sorted sums with grid zones and a mask match the oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const bounds: [number, number, number, number] = [0, 0, 3, 2];
  const gridSize: [number, number] = [3, 2];
  const rowCount = 1500;
  const origins = createSeededPoints(3, rowCount, bounds);
  const destinations = createSeededPoints(4, rowCount, bounds);
  origins.splice(0, 4, 9, 9, Number.NaN, 1);
  const weightValues = createOrderSensitiveWeights(2, rowCount);
  const maskValues = Array.from({length: rowCount}, (_, row) => (row % 5 === 0 ? 0 : 1));
  const buffers = [
    createInputBuffer(device, Float32Array.from(origins)),
    createInputBuffer(device, Float32Array.from(destinations)),
    createInputBuffer(device, weightValues),
    createInputBuffer(device, Uint32Array.from(maskValues))
  ];
  const harness = createHarness(device, {
    id: 'flow-sorted-grid',
    zoneCount: 6,
    topCount: 36,
    pairCapacity: 64,
    hasWeights: true,
    createProps: graph => ({
      sumOrder: 'sorted',
      zones: {kind: 'grid', bounds, gridSize},
      origins: importGraphBuffer(graph, 'origins', buffers[0], 'float32x2', rowCount),
      destinations: importGraphBuffer(graph, 'destinations', buffers[1], 'float32x2', rowCount),
      weights: importGraphBuffer(graph, 'weights', buffers[2], 'float32', rowCount),
      mask: importGraphBuffer(graph, 'mask', buffers[3], 'uint32', rowCount)
    })
  });
  harness.submit();
  expectMatchesOracle(
    await harness.read(),
    computeFlowAggregation({
      zones: {kind: 'grid', bounds, gridSize},
      origins,
      destinations,
      weights: Array.from(weightValues),
      mask: maskValues,
      sumMode: 'tree'
    }),
    36,
    true
  );
  harness.destroy();
  for (const buffer of buffers) buffer.destroy();
});

it('GPUFlowAggregation measures atomic versus sorted sum cost on 1M rows', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const rowCount = 1 << 20;
  const zoneCount = 64;
  const {origins, destinations} = createZoneIds(8, rowCount, zoneCount);
  const weightValues = createOrderSensitiveWeights(6, rowCount);
  const buffers = [
    createInputBuffer(device, Uint32Array.from(origins)),
    createInputBuffer(device, Uint32Array.from(destinations)),
    createInputBuffer(device, weightValues)
  ];
  for (const sumOrder of ['atomic', 'sorted'] as const) {
    const harness = createHarness(device, {
      id: `flow-cost-${sumOrder}`,
      zoneCount,
      topCount: 64,
      pairCapacity: 8192,
      maxProbeCount: 64,
      hasWeights: true,
      createProps: graph => ({
        sumOrder,
        zones: {kind: 'ids', zoneCount},
        originZoneIds: importGraphBuffer(graph, 'origin-ids', buffers[0], 'uint32', rowCount),
        destinationZoneIds: importGraphBuffer(
          graph,
          'destination-ids',
          buffers[1],
          'uint32',
          rowCount
        ),
        weights: importGraphBuffer(graph, 'weights', buffers[2], 'float32', rowCount)
      })
    });
    harness.submit();
    await harness.read();
    const runCount = 3;
    const startTime = performance.now();
    for (let run = 0; run < runCount; run++) {
      harness.submit();
      await harness.read();
    }
    const milliseconds = (performance.now() - startTime) / runCount;
    const result = await harness.read();
    expect(result.zoneOutCounts.reduce((sum, value) => sum + value, 0)).toBe(rowCount);
    console.log(`flow cost 1M rows, ${sumOrder}: ${milliseconds.toFixed(1)} ms per submit+read`);
    harness.destroy();
  }
  for (const buffer of buffers) buffer.destroy();
});
