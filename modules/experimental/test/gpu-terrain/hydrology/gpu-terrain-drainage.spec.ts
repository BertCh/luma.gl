// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {GPUTerrainHeightAboveDrainage} from '../../../src/gpu-terrain/hydrology/gpu-terrain-height-above-drainage';
import {GPUTerrainStreamOrder} from '../../../src/gpu-terrain/hydrology/gpu-terrain-stream-order';
import {
  GPU_TERRAIN_WATERSHED_NONE,
  GPUTerrainWatersheds
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-watersheds';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {
  computeHeightAboveDrainageOnCPU,
  computeStreamOrderOnCPU,
  computeWatershedsOnCPU
} from './terrain-drainage-oracle';
import {TERRAIN_FLOW_DEMS as DEMS, computeTerrainFlow} from './terrain-flow-oracle';

type DrainageInput = {
  width: number;
  height: number;
  elevation: Float32Array;
  directions: Uint32Array;
  streams: Uint32Array;
  pourPoints?: Uint32Array;
  maxIterations?: number;
  maxOrderIterations?: number;
};

async function runDrainage(device: Device, input: DrainageInput) {
  const {width, height} = input;
  const cellCount = width * height;
  const inputBuffers = [
    createInputBuffer(device, input.elevation),
    createInputBuffer(device, input.directions),
    createInputBuffer(device, input.streams),
    ...(input.pourPoints ? [createInputBuffer(device, input.pourPoints)] : [])
  ];
  const outputs = {
    hand: createOutputBuffer(device, cellCount),
    drainage: createOutputBuffer(device, cellCount),
    handConverged: createOutputBuffer(device, 1),
    basins: createOutputBuffer(device, cellCount),
    basinsConverged: createOutputBuffer(device, 1),
    pour: createOutputBuffer(device, cellCount),
    pourConverged: createOutputBuffer(device, 1),
    order: createOutputBuffer(device, cellCount),
    orderConverged: createOutputBuffer(device, 1)
  };
  const graph = new GPUCommandGraph(device, {id: 'terrain-drainage-test'});
  const elevation = importGraphBuffer(graph, 'elevation', inputBuffers[0], 'float32', cellCount);
  const directions = importGraphBuffer(graph, 'directions', inputBuffers[1], 'uint32', cellCount);
  const streams = importGraphBuffer(graph, 'streams', inputBuffers[2], 'uint32', cellCount);
  const view = (name: keyof typeof outputs, format: 'float32' | 'uint32', length = cellCount) =>
    importGraphBuffer(graph, name, outputs[name], format, length);
  graph.add(
    new GPUTerrainHeightAboveDrainage({
      width,
      height,
      elevation: {id: 'elevation', format: 'float32', storage: {kind: 'buffer', values: elevation}},
      flowDirections: directions,
      streams,
      heightAboveDrainage: view('hand', 'float32'),
      drainageCells: view('drainage', 'uint32'),
      converged: view('handConverged', 'uint32', 1),
      maxIterations: input.maxIterations
    })
  );
  graph.add(
    new GPUTerrainWatersheds({
      id: 'basins',
      width,
      height,
      flowDirections: directions,
      labels: view('basins', 'uint32'),
      converged: view('basinsConverged', 'uint32', 1),
      maxIterations: input.maxIterations
    })
  );
  if (input.pourPoints) {
    graph.add(
      new GPUTerrainWatersheds({
        id: 'pour',
        width,
        height,
        flowDirections: directions,
        pourPoints: importGraphBuffer(
          graph,
          'pour-points',
          inputBuffers[3],
          'uint32',
          input.pourPoints.length
        ),
        labels: view('pour', 'uint32'),
        converged: view('pourConverged', 'uint32', 1),
        maxIterations: input.maxIterations
      })
    );
  }
  graph.add(
    new GPUTerrainStreamOrder({
      width,
      height,
      flowDirections: directions,
      streams,
      streamOrder: view('order', 'uint32'),
      converged: view('orderConverged', 'uint32', 1),
      maxIterations: input.maxOrderIterations
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    hand: Float32Array.from(await readFloat32(outputs.hand, cellCount)),
    drainage: Uint32Array.from(await readUint32(outputs.drainage, cellCount)),
    handConverged: (await readUint32(outputs.handConverged, 1))[0],
    basins: Uint32Array.from(await readUint32(outputs.basins, cellCount)),
    basinsConverged: (await readUint32(outputs.basinsConverged, 1))[0],
    pour: Uint32Array.from(await readUint32(outputs.pour, cellCount)),
    pourConverged: (await readUint32(outputs.pourConverged, 1))[0],
    order: Uint32Array.from(await readUint32(outputs.order, cellCount)),
    orderConverged: (await readUint32(outputs.orderConverged, 1))[0]
  };
  compiled.destroy();
  for (const buffer of [...inputBuffers, ...Object.values(outputs)]) {
    buffer.destroy();
  }
  return result;
}

/** Bit pattern comparison that treats NaN as equal to NaN. */
function toBits(values: Float32Array): number[] {
  const bits = new Uint32Array(values.length);
  const floats = new Float32Array(bits.buffer);
  for (const [index, value] of values.entries()) {
    floats[index] = Number.isNaN(value) ? NaN : value;
  }
  return Array.from(bits);
}

const DEM_CASES: [string, (width: number, height: number) => Float32Array, boolean][] = [
  ['cone', DEMS.cone, false],
  ['valley', DEMS.valley, true],
  ['noise', (width, height) => DEMS.noise(width, height), true],
  ['noise raw', (width, height) => DEMS.noise(width, height), false],
  ['nodata holes', (width, height) => DEMS.noDataHoles(width, height), true]
];

for (const [width, height] of [
  [37, 29],
  [70, 45]
]) {
  it(`drainage recipes match the oracles on synthetic DEMs ${width}x${height}`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    let sawHeight = false;
    let sawOrderTwo = false;
    let sawManyLabels = false;
    let sawManyPourLabels = false;
    for (const [name, createDem, fill] of DEM_CASES) {
      const raw = createDem(width, height);
      const flow = computeTerrainFlow({
        elevation: raw,
        width,
        height,
        fillDepressions: fill,
        settings: {cellSize: [1, 1], fillEpsilon: fill ? 0.25 : 0, streamThreshold: 5}
      });
      const elevation = fill ? flow.filled : raw;
      const cellCount = width * height;
      // Point 0 and 1 share a cell, one point is out of range, others are scattered cells.
      const random = (index: number) => (index * 2654435761) % cellCount;
      const pourPoints = Uint32Array.from([
        random(3),
        random(3),
        cellCount + 5,
        random(7),
        random(11),
        random(19),
        0xffffffff,
        random(23)
      ]);
      const result = await runDrainage(device, {
        width,
        height,
        elevation,
        directions: flow.directions,
        streams: flow.streams,
        pourPoints
      });
      const hand = computeHeightAboveDrainageOnCPU(
        elevation,
        flow.directions,
        flow.streams,
        width,
        height
      );
      expect(toBits(result.hand), `${name} hand`).toEqual(toBits(hand.heightAboveDrainage));
      expect(Array.from(result.drainage), `${name} drainage`).toEqual(
        Array.from(hand.drainageCells)
      );
      expect(Array.from(result.basins), `${name} basins`).toEqual(
        Array.from(computeWatershedsOnCPU(flow.directions, width, height))
      );
      expect(Array.from(result.pour), `${name} pour`).toEqual(
        Array.from(computeWatershedsOnCPU(flow.directions, width, height, pourPoints))
      );
      expect(Array.from(result.order), `${name} order`).toEqual(
        Array.from(computeStreamOrderOnCPU(flow.directions, flow.streams, width, height))
      );
      expect([result.handConverged, result.basinsConverged, result.pourConverged]).toEqual([
        1, 1, 1
      ]);
      expect(result.orderConverged).toBe(1);
      for (const [cell, value] of flow.streams.entries()) {
        if (value !== 0 && flow.directions[cell] !== 0xffffffff) {
          expect(result.hand[cell], `${name} hand on stream ${cell}`).toBe(0);
          expect(result.drainage[cell]).toBe(cell);
        }
      }
      sawHeight ||= result.hand.some(value => value > 0);
      sawOrderTwo ||= result.order.some(value => value >= 2);
      sawManyLabels ||= new Set(result.basins).size > 2;
      sawManyPourLabels ||= new Set(result.pour).size > 2;
    }
    expect(sawHeight).toBe(true);
    expect(sawOrderTwo).toBe(true);
    expect(sawManyLabels).toBe(true);
    expect(sawManyPourLabels).toBe(true);
    device.destroy();
  });
}

it('nested pour points give nested watersheds and the lowest duplicate index wins', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // 8x1 strip flowing east: cell i drains to i + 1, cell 7 is terminal.
  const directions = Uint32Array.from([1, 1, 1, 1, 1, 1, 1, 0]);
  const result = await runDrainage(device, {
    width: 8,
    height: 1,
    elevation: Float32Array.from([8, 7, 6, 5, 4, 3, 2, 1]),
    directions,
    streams: new Uint32Array(8),
    // Point 0 is an outlet-side point at cell 5, point 1 nests above it at cell 2, points 2 and 3
    // share cell 2 with point 1 (lower index wins), point 4 is out of range.
    pourPoints: Uint32Array.from([5, 2, 2, 2, 99])
  });
  const none = GPU_TERRAIN_WATERSHED_NONE;
  expect(Array.from(result.pour)).toEqual([1, 1, 1, 0, 0, 0, none, none]);
  expect(Array.from(result.basins)).toEqual(new Array(8).fill(7));
  device.destroy();
});

it('a 1x3000 strip needs about 12 doubling rounds', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const length = 3000;
  const directions = new Uint32Array(length).fill(4);
  directions[length - 1] = 0;
  const streams = new Uint32Array(length);
  streams.fill(1, length - 10);
  const elevation = Float32Array.from({length}, (_, row) => length - row);
  const input = {width: 1, height: length, elevation, directions, streams};
  const result = await runDrainage(device, input);
  expect(result.handConverged).toBe(1);
  expect(result.basinsConverged).toBe(1);
  expect(result.drainage[0]).toBe(length - 10);
  expect(result.hand[0]).toBe(length - 10);
  expect(result.basins[0]).toBe(length - 1);
  expect(result.hand[length - 5]).toBe(0);
  const oracle = computeHeightAboveDrainageOnCPU(elevation, directions, streams, 1, length);
  expect(toBits(result.hand)).toEqual(toBits(oracle.heightAboveDrainage));
  expect(Array.from(result.order)).toEqual(
    Array.from(computeStreamOrderOnCPU(directions, streams, 1, length))
  );
  expect(result.orderConverged).toBe(1);

  const limited = await runDrainage(device, {...input, maxIterations: 2});
  expect(limited.handConverged).toBe(0);
  expect(limited.basinsConverged).toBe(0);
  device.destroy();
});

it('GPUTerrainStreamOrder follows Strahler on a hand-built network', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 7;
  const height = 5;
  const directions = new Uint32Array(width * height);
  const streams = new Uint32Array(width * height);
  const at = (column: number, row: number) => row * width + column;
  const connect = (from: [number, number], code: number, isStream = true) => {
    directions[at(...from)] = code;
    streams[at(...from)] = isStream ? 1 : 0;
  };
  // Trunk along row 2 toward the east edge.
  connect([3, 2], 1);
  connect([4, 2], 1);
  connect([5, 2], 1);
  streams[at(6, 2)] = 1; // terminal outlet
  // Two leaves join at (3,2): 1,1 -> 2.
  connect([2, 1], 2);
  connect([2, 3], 128);
  // (4,2) joins order 2 with a leaf: 2,1 -> 2.
  connect([4, 1], 4);
  // (5,3) has two leaves, then joins the trunk at (5,2): 2,2 -> 3.
  connect([5, 3], 64);
  connect([4, 4], 128);
  connect([6, 4], 32);
  // A non-stream cell draining into the trunk does not count as a donor.
  connect([4, 3], 128, false);
  const result = await runDrainage(device, {
    width,
    height,
    elevation: new Float32Array(width * height),
    directions,
    streams
  });
  const expected = new Uint32Array(width * height);
  for (const [cell, order] of [
    [at(2, 1), 1],
    [at(2, 3), 1],
    [at(3, 2), 2],
    [at(4, 1), 1],
    [at(4, 2), 2],
    [at(4, 4), 1],
    [at(6, 4), 1],
    [at(5, 3), 2],
    [at(5, 2), 3],
    [at(6, 2), 3]
  ]) {
    expected[cell] = order;
  }
  expect(Array.from(result.order)).toEqual(Array.from(expected));
  expect(Array.from(computeStreamOrderOnCPU(directions, streams, width, height))).toEqual(
    Array.from(expected)
  );
  device.destroy();
});

it('invalid, multi-bit, and out-of-grid direction codes are terminal', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Cell 0 points west out of the grid, cell 2 has the invalid code 3, cell 4 is multi-bit.
  const directions = Uint32Array.from([16, 1, 3, 1, 5, 0xffffffff]);
  const streams = Uint32Array.from([0, 0, 1, 0, 0, 1]);
  const result = await runDrainage(device, {
    width: 6,
    height: 1,
    elevation: Float32Array.from([5, 4, 3, 2, 1, 0]),
    directions,
    streams
  });
  expect(Array.from(result.basins)).toEqual([0, 2, 2, 4, 4, GPU_TERRAIN_WATERSHED_NONE]);
  expect(Array.from(result.drainage)).toEqual([
    0xffffffff, 2, 2, 0xffffffff, 0xffffffff, 0xffffffff
  ]);
  expect(toBits(result.hand)).toEqual(toBits(Float32Array.from([NaN, 1, 0, NaN, NaN, NaN])));
  // The invalid-code cell 5 is not a stream cell even though its mask is set.
  expect(Array.from(result.order)).toEqual([0, 0, 1, 0, 0, 0]);
  device.destroy();
});
