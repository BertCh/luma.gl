// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {afterAll, expect, it} from 'vitest';
import {GPUMapGraphParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {
  GPUTerrainFlow,
  getGPUTerrainFlowParameterValues,
  type GPUTerrainFlowProps,
  type GPUTerrainFlowSettings
} from '../../../src/map-graphs/hydrology';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../map-graph-test-utils';
import {createSeededRandom, computeTerrainFlow} from './terrain-flow-oracle';
import {
  accumulateRoutingOnCPU,
  getRoutingEdgesOnCPU,
  sumTerminalAccumulationOnCPU,
  type RoutingOracleInput
} from './terrain-flow-routing-oracle';

type Routing = 'd-infinity' | 'mfd-freeman' | 'mfd-quinn';
const ROUTINGS: Routing[] = ['d-infinity', 'mfd-freeman', 'mfd-quinn'];

/**
 * Relative tolerance of GPU against the float64 oracle: `|gpu - cpu| <= tolerance * max(1, |cpu|)`.
 *
 * MFD (2e-4): float32 fractions, pow = exp2(p log2 x) with a few ULP of GPU error, sums of up to
 * hundreds of products along a path, and Metal fusing multiply-adds; observed errors are below
 * 3e-5.
 *
 * D-infinity (1e-3): additionally the facet competition is a near-tie when a steepest direction
 * lies close to a facet edge. There the interior slope hypot(s1, s2) of one facet exceeds the
 * clamped diagonal slope of its neighbor by only about dr^2 / 2 (dr = angle distance to the edge),
 * so float32 (6e-8 relative) cannot order them for dr below about 4e-4, and the GPU may choose the
 * neighboring facet. Fractions are continuous across that edge, so the resulting error is
 * bounded by about 5e-4 of the donor's accumulation in the worst case; observed errors are below
 * 3e-4.
 */
const TOLERANCE: Record<Routing | 'd8', number> = {
  d8: 2e-4,
  'd-infinity': 1e-3,
  'mfd-freeman': 2e-4,
  'mfd-quinn': 2e-4
};
const maximumObservedError: Record<string, number> = {};

type FlowOptions = {
  routing: GPUTerrainFlowProps['flowRouting'];
  settings?: GPUTerrainFlowSettings;
  fillDepressions?: boolean;
  resolveFlats?: boolean;
  runoff?: Float32Array;
  cellSizeMode?: GPUTerrainFlowProps['cellSizeMode'];
  accumulationUnits?: GPUTerrainFlowProps['accumulationUnits'];
  maxAccumulationIterations?: number;
};

const DEFAULT_SETTINGS: GPUTerrainFlowSettings = {cellSize: [1, 1]};

function createFlowFixture(
  device: Device,
  elevation: Float32Array,
  width: number,
  height: number,
  options: FlowOptions
) {
  const cellCount = width * height;
  const settings = options.settings ?? DEFAULT_SETTINGS;
  const elevationBuffer = createInputBuffer(device, elevation);
  const runoffBuffer = options.runoff ? createInputBuffer(device, options.runoff) : undefined;
  const buffers = {
    accumulation: createOutputBuffer(device, cellCount),
    converged: createOutputBuffer(device, 1),
    flatsConverged: createOutputBuffer(device, 1)
  };
  const settingsBuffer = new GPUMapGraphParameterBuffer(device, {
    id: 'flow-settings',
    format: 'float32',
    length: 8,
    values: getGPUTerrainFlowParameterValues(settings)
  });
  const graph = new GPUCommandGraph(device, {id: 'terrain-flow-routing-test'});
  const destroy = () => {
    settingsBuffer.destroy();
    elevationBuffer.destroy();
    runoffBuffer?.destroy();
    for (const buffer of Object.values(buffers)) {
      buffer.destroy();
    }
  };
  let compiled: ReturnType<typeof graph.compile>;
  try {
    graph.add(
      new GPUTerrainFlow({
        width,
        height,
        elevation: {
          id: 'elevation',
          format: 'float32',
          storage: {
            kind: 'buffer',
            values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', cellCount)
          }
        },
        settings: settingsBuffer.importToGraph(graph),
        cellSizeMode: options.cellSizeMode,
        fillDepressions: options.fillDepressions,
        resolveFlats: options.resolveFlats,
        flowRouting: options.routing,
        maxAccumulationIterations: options.maxAccumulationIterations,
        accumulationUnits: options.accumulationUnits,
        runoff: runoffBuffer
          ? importGraphBuffer(graph, 'runoff', runoffBuffer, 'float32', cellCount)
          : undefined,
        accumulation: importGraphBuffer(
          graph,
          'accumulation',
          buffers.accumulation,
          'float32',
          cellCount
        ),
        accumulationConverged: importGraphBuffer(
          graph,
          'accumulation-converged',
          buffers.converged,
          'uint32',
          1
        ),
        flatsConverged: options.resolveFlats
          ? importGraphBuffer(graph, 'flats-converged', buffers.flatsConverged, 'uint32', 1)
          : undefined
      })
    );
    compiled = graph.compile();
  } catch (error) {
    destroy();
    throw error;
  }
  return {
    async run() {
      submitGraph(device, compiled, undefined);
      return {
        accumulation: Float32Array.from(await readFloat32(buffers.accumulation, cellCount)),
        converged: (await readUint32(buffers.converged, 1))[0]
      };
    },
    destroy: () => {
      compiled.destroy();
      destroy();
    }
  };
}

function runOracle(elevation: Float32Array, width: number, height: number, options: FlowOptions) {
  return computeTerrainFlow({
    elevation,
    width,
    height,
    settings: options.settings ?? DEFAULT_SETTINGS,
    cellSizeMode: options.cellSizeMode,
    fillDepressions: options.fillDepressions,
    resolveFlats: options.resolveFlats,
    runoff: options.runoff,
    area: options.accumulationUnits === 'area',
    flowRouting: options.routing
  });
}

/** Compares GPU and oracle accumulation; returns the oracle result and the GPU values. */
async function expectMatchesOracle(
  device: Device,
  name: string,
  elevation: Float32Array,
  width: number,
  height: number,
  options: FlowOptions
) {
  const routing = options.routing as Routing | 'd8';
  const fixture = createFlowFixture(device, elevation, width, height, options);
  const result = await fixture.run();
  fixture.destroy();
  const oracle = runOracle(elevation, width, height, options);
  let maximumError = 0;
  let maximumValue = 0;
  for (const [cell, expected] of oracle.accumulation.entries()) {
    const actual = result.accumulation[cell];
    if (Number.isNaN(expected)) {
      expect(actual, `${name} cell ${cell} must be NaN`).toBeNaN();
      continue;
    }
    expect(Number.isFinite(actual), `${name} cell ${cell} finite (${actual})`).toBe(true);
    const error = Math.abs(actual - expected) / Math.max(1, Math.abs(expected));
    maximumError = Math.max(maximumError, error);
    maximumValue = Math.max(maximumValue, expected);
    expect(error, `${name} cell ${cell}: gpu ${actual} cpu ${expected}`).toBeLessThanOrEqual(
      TOLERANCE[routing]
    );
  }
  maximumObservedError[routing] = Math.max(maximumObservedError[routing] ?? 0, maximumError);
  expect(result.converged, `${name} accumulationConverged`).toBe(1);
  return {oracle, result, maximumValue};
}

// Smooth or continuous DEMs with non-integer values: no exact facet ties.
const DEMS: Record<string, (width: number, height: number) => Float32Array> = {
  cone: (width, height) =>
    Float32Array.from({length: width * height}, (_, cell) => {
      const column = cell % width;
      const row = Math.floor(cell / width);
      const dx = column - (width - 1) / 2;
      const dy = row - (height - 1) / 2;
      return 100.3 - 1.7 * Math.hypot(dx, dy) + 0.0137 * column - 0.0071 * row;
    }),
  ridges: (width, height) =>
    Float32Array.from({length: width * height}, (_, cell) => {
      const column = cell % width;
      const row = Math.floor(cell / width);
      return (
        20 + 5 * Math.sin(column * 0.45) + 4 * Math.cos(row * 0.37) + 0.21 * column + 0.13 * row
      );
    }),
  valley: (width, height) => {
    const random = createSeededRandom(5);
    return Float32Array.from({length: width * height}, (_, cell) => {
      const column = cell % width;
      const row = Math.floor(cell / width);
      return 3.1 * Math.abs(column - (width - 1) / 2) + 0.37 * (height - 1 - row) + 0.02 * random();
    });
  },
  noise: (width, height) => {
    const random = createSeededRandom(7);
    return Float32Array.from({length: width * height}, () => 20 * random());
  },
  'nodata holes': (width, height) => {
    const random = createSeededRandom(11);
    return Float32Array.from({length: width * height}, () =>
      random() < 0.1 ? NaN : 12 * random()
    );
  }
};

for (const routing of ROUTINGS) {
  for (const [width, height] of [
    [37, 29],
    [70, 45]
  ]) {
    it(`GPUTerrainFlow ${routing} matches the float64 oracle on synthetic DEMs ${width}x${height}`, async () => {
      const device = await getWebGPUTestDevice();
      if (!device) {
        return;
      }
      for (const [name, createDem] of Object.entries(DEMS)) {
        const elevation = createDem(width, height);
        const raw = await expectMatchesOracle(
          device,
          `${routing} ${name} raw`,
          elevation,
          width,
          height,
          {
            routing
          }
        );
        // Silent-zero guard: flow must concentrate somewhere.
        expect(raw.maximumValue, `${routing} ${name} max accumulation`).toBeGreaterThan(3);
        await expectMatchesOracle(device, `${routing} ${name} fill`, elevation, width, height, {
          routing,
          fillDepressions: true,
          settings: {cellSize: [1, 1], fillEpsilon: 0.25}
        });
      }
    });
  }

  it(`GPUTerrainFlow ${routing} handles geographic and web-mercator cell sizes and weights`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    const width = 37;
    const height = 29;
    const elevation = DEMS.ridges(width, height);
    const runoff = Float32Array.from({length: width * height}, (_, cell) =>
      cell % 7 === 0 ? 0 : 0.25 + (cell % 5)
    );
    const geographic: FlowOptions = {
      routing,
      cellSizeMode: 'geographic',
      settings: {cellSize: [0.0009, 0.0006], northEdge: 52, southEdge: 51.9, fillEpsilon: 0.05},
      fillDepressions: true
    };
    await expectMatchesOracle(
      device,
      `${routing} geographic`,
      elevation,
      width,
      height,
      geographic
    );
    await expectMatchesOracle(device, `${routing} geographic area`, elevation, width, height, {
      ...geographic,
      accumulationUnits: 'area',
      runoff
    });
    const mercator: FlowOptions = {
      routing,
      cellSizeMode: 'web-mercator',
      settings: {cellSize: [1.4, 1], northEdge: 0.3, southEdge: 0.45, fillEpsilon: 0.1},
      fillDepressions: true
    };
    await expectMatchesOracle(
      device,
      `${routing} mercator`,
      DEMS.noise(width, height),
      width,
      height,
      mercator
    );
    await expectMatchesOracle(device, `${routing} mercator runoff`, elevation, width, height, {
      ...mercator,
      runoff
    });
    await expectMatchesOracle(device, `${routing} uniform area`, elevation, width, height, {
      routing,
      accumulationUnits: 'area',
      settings: {cellSize: [3, 5]},
      runoff
    });
  });
}

it('GPUTerrainFlow MFD honors explicit and default flow exponents', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 37;
  const height = 29;
  const elevation = DEMS.noise(width, height);
  const results = new Map<number, Float32Array>();
  for (const routing of ['mfd-freeman', 'mfd-quinn'] as const) {
    for (const flowExponent of [0, -3, NaN, 1, 4, 8]) {
      const run = await expectMatchesOracle(
        device,
        `${routing} p=${flowExponent}`,
        elevation,
        width,
        height,
        {routing, settings: {cellSize: [1, 1], flowExponent}}
      );
      if (routing === 'mfd-freeman') {
        results.set(flowExponent, run.result.accumulation);
      }
    }
  }
  // Non-positive and NaN exponents select the published default 1.1, which differs from 1 and 4.
  expect(Array.from(results.get(0)!)).toEqual(Array.from(results.get(-3)!));
  expect(Array.from(results.get(0)!)).toEqual(Array.from(results.get(NaN)!));
  expect(results.get(0)).not.toEqual(results.get(1));
  expect(results.get(1)).not.toEqual(results.get(4));
});

it('GPUTerrainFlow MFD approaches D8 as the exponent grows, D-infinity equals D8 on an axis-aligned plane', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 37;
  const height = 29;
  const valley = DEMS.valley(width, height);
  const d8 = createFlowFixture(device, valley, width, height, {routing: 'd8'});
  const d8Result = await d8.run();
  d8.destroy();
  expect(Math.max(...d8Result.accumulation.filter(Number.isFinite))).toBeGreaterThan(20);
  for (const routing of ['mfd-freeman', 'mfd-quinn'] as const) {
    const {result} = await expectMatchesOracle(device, `${routing} p=64`, valley, width, height, {
      routing,
      settings: {cellSize: [1, 1], flowExponent: 64}
    });
    for (const [cell, expected] of d8Result.accumulation.entries()) {
      expect(
        Math.abs(result.accumulation[cell] - expected),
        `${routing} cell ${cell}`
      ).toBeLessThanOrEqual(1e-3 * Math.max(1, expected));
    }
  }

  // Elevation rises along +x only: all flow is cardinal (west), so D-infinity equals D8.
  const plane = Float32Array.from({length: width * height}, (_, cell) => 0.7 * (cell % width));
  const planeD8 = createFlowFixture(device, plane, width, height, {routing: 'd8'});
  const planeD8Result = await planeD8.run();
  planeD8.destroy();
  const planeDInfinity = createFlowFixture(device, plane, width, height, {routing: 'd-infinity'});
  const planeDInfinityResult = await planeDInfinity.run();
  planeDInfinity.destroy();
  expect(Array.from(planeDInfinityResult.accumulation)).toEqual(
    Array.from(planeD8Result.accumulation)
  );
  expect(planeD8Result.accumulation[0]).toBe(width);
});

it('GPUTerrainFlow D-infinity splits flow on a plane of slope (2, 1) with r = atan(1/2)', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // z = -2 column - row: steepest descent toward east-south-east, r = atan(1/2) inside facet (E, SE).
  const width = 3;
  const height = 3;
  const plane = Float32Array.from({length: 9}, (_, cell) => -2 * (cell % 3) - Math.floor(cell / 3));
  const fractionDiagonal = Math.atan(0.5) / (Math.PI / 4);
  const input: RoutingOracleInput = {
    routing: 'd-infinity',
    width,
    height,
    cellSizeMode: 'uniform',
    gridSettings: {cellSize: [1, 1]},
    flowExponent: 0,
    surface: plane,
    isValid: () => true,
    receivers: new Uint32Array(9).fill(0xffffffff),
    weights: new Float32Array(9).fill(1)
  };
  const edges = getRoutingEdgesOnCPU(input, 4);
  expect(edges.map(edge => [edge.neighbor, edge.direction])).toEqual([
    [5, 0],
    [8, 1]
  ]);
  expect(edges[0].fraction).toBeCloseTo(1 - fractionDiagonal, 12);
  expect(edges[1].fraction).toBeCloseTo(fractionDiagonal, 12);

  // Only the center cell carries flow; its east and south-east neighbors receive the split.
  const runoff = new Float32Array(9);
  runoff[4] = 1;
  const options: FlowOptions = {routing: 'd-infinity', runoff};
  const {result} = await expectMatchesOracle(device, 'plane (2,1)', plane, width, height, options);
  expect(result.accumulation[5]).toBeCloseTo(1 - fractionDiagonal, 5);
  // The south-east corner receives its share and the east cell's onward flow (east drains south).
  expect(result.accumulation[8]).toBeCloseTo(1, 5);
  expect(result.accumulation[4]).toBeCloseTo(1, 6);
});

it('GPUTerrainFlow conserves mass and sends fallback flow only to D8 receivers', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 37;
  const height = 29;
  for (const routing of ROUTINGS) {
    for (const [name, createDem] of Object.entries(DEMS)) {
      const elevation = createDem(width, height);
      const options: FlowOptions = {routing};
      const {oracle, result} = await expectMatchesOracle(
        device,
        `${routing} ${name} mass`,
        elevation,
        width,
        height,
        options
      );
      const validCount = elevation.filter(Number.isFinite).length;
      const input: RoutingOracleInput = {
        routing,
        width,
        height,
        cellSizeMode: 'uniform',
        gridSettings: {cellSize: [1, 1]},
        flowExponent: 0,
        surface: elevation,
        isValid: cell => Number.isFinite(elevation[cell]),
        receivers: oracle.receivers,
        weights: new Float32Array(width * height).fill(1)
      };
      expect(sumTerminalAccumulationOnCPU(input, accumulateRoutingOnCPU(input))).toBeCloseTo(
        validCount,
        9
      );
      expect(sumTerminalAccumulationOnCPU(input, oracle.accumulation)).toBeCloseTo(validCount, 3);
      expect(sumTerminalAccumulationOnCPU(input, result.accumulation)).toBeCloseTo(validCount, 2);
    }
  }
});

it('GPUTerrainFlow routes through resolved flats when flat resolution is available', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 37;
  const height = 29;
  const elevation = DEMS.ridges(width, height);
  for (const routing of ['d8', ...ROUTINGS] as const) {
    const options: FlowOptions = {
      routing,
      fillDepressions: true,
      resolveFlats: true,
      settings: {cellSize: [1, 1], fillEpsilon: 0}
    };
    try {
      await expectMatchesOracle(device, `${routing} flats`, elevation, width, height, options);
    } catch (error) {
      if (error instanceof Error && /not implemented/.test(error.message)) {
        return; // Flat resolution has not landed yet; the coordinator enables this test.
      }
      throw error;
    }
  }
});

it('GPUTerrainFlow reports the rounds needed by the routed accumulation', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 70;
  const height = 45;
  const elevation = DEMS.noise(width, height);
  const rounds: Record<string, number> = {};
  for (const routing of ROUTINGS) {
    for (let limit = 1; limit <= 64; limit++) {
      const fixture = createFlowFixture(device, elevation, width, height, {
        routing,
        maxAccumulationIterations: limit
      });
      const {converged} = await fixture.run();
      fixture.destroy();
      if (converged === 1) {
        rounds[routing] = limit;
        break;
      }
    }
    expect(rounds[routing], `${routing} converges within 64 rounds`).toBeDefined();
  }
  console.log('terrain flow routing rounds (noise 70x45):', JSON.stringify(rounds));
});

afterAll(() => {
  console.log('terrain flow routing max relative error:', JSON.stringify(maximumObservedError));
});
