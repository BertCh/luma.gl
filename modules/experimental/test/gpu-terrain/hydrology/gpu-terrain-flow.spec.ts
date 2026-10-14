// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import hydrologyReference from '../../oracles/fixtures/terrain-hydrology.json';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  makeGPURasterBand,
  makeGPURasterMetadata,
  type GPURasterMetadata
} from '../../../src/gpu-raster';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPU_TERRAIN_FLOW_CELL_CLASS as CELL_CLASS,
  GPU_TERRAIN_FLOW_STREAM_UNRESOLVED,
  GPUTerrainFlow,
  getGPUTerrainFlowParameterValues,
  type GPUTerrainFlowProps,
  type GPUTerrainFlowSettings
} from '../../../src/gpu-terrain/hydrology';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  TERRAIN_FLOW_DEMS as DEMS,
  computeTerrainFlow,
  createSeededRandom,
  type TerrainFlowOracleResult
} from './terrain-flow-oracle';
import {toBits} from '../terrain-test-utils';
import {getOracleWebMercatorY} from '../terrain-oracle-utils';
import {createSerpentineDEM, fillDepressionsPriorityFlood} from './terrain-flow-corridors';
import {requireTestDevice} from '../../utils/require-test-device';

type FlowOptions = {
  /** Packed settings; the cell size and latitude band come from `cellSize` and `origin`. */
  settings?: GPUTerrainFlowSettings;
  /** Ground cell size of the grid. Defaults to 1. */
  cellSize?: number | [number, number];
  /** World position of the top-left corner of the grid. Defaults to `[0, 0]`. */
  origin?: [number, number];
  fillDepressions?: boolean;
  runoff?: Float32Array;
  cellSizeMode?: GPUTerrainFlowProps['cellSizeMode'];
  accumulationUnits?: GPUTerrainFlowProps['accumulationUnits'];
  maximumFillIterations?: number;
  maximumAccumulationIterations?: number;
  /** Publish both convergence flags as 1-row views at byte offsets 0 and 4 of one 8-byte buffer. */
  sharedSummaryBuffer?: boolean;
};

type FlowResult = TerrainFlowOracleResult & {fillConverged: number; accumulationConverged: number};

const DEFAULT_SETTINGS: GPUTerrainFlowSettings = {};

/** Returns the raster grid that a spec places its DEM on. */
function getFlowGrid(width: number, height: number, options: FlowOptions = {}): GPURasterMetadata {
  return makeGPURasterMetadata({width, height, cellSize: options.cellSize, origin: options.origin});
}

function createFlowFixture(
  device: Device,
  elevation: Float32Array,
  width: number,
  height: number,
  options: FlowOptions = {}
) {
  const cellCount = width * height;
  const grid = getFlowGrid(width, height, options);
  const settings = options.settings ?? DEFAULT_SETTINGS;
  const elevationBuffer = createInputBuffer(device, elevation);
  const runoffBuffer = options.runoff ? createInputBuffer(device, options.runoff) : undefined;
  const buffers = {
    filled: createOutputBuffer(device, cellCount),
    directions: createOutputBuffer(device, cellCount),
    classes: createOutputBuffer(device, cellCount),
    accumulation: createOutputBuffer(device, cellCount),
    streams: createOutputBuffer(device, cellCount),
    fillConverged: createOutputBuffer(device, options.sharedSummaryBuffer ? 2 : 1),
    accumulationConverged: createOutputBuffer(device, 1)
  };
  const settingsBuffer = new GPUParameterBuffer(device, {
    id: 'flow-settings',
    format: 'float32',
    length: 8,
    values: getGPUTerrainFlowParameterValues(settings, grid)
  });
  const graph = new GPUCommandGraph(device, {id: 'terrain-flow-test'});
  const fillDepressions = options.fillDepressions ?? false;
  const summary = options.sharedSummaryBuffer
    ? importGraphBuffer(graph, 'summary', buffers.fillConverged, 'uint32', 2)
    : undefined;
  const flagAt = (byteOffset: number) =>
    graph.createDataView(summary!.buffer, {format: 'uint32', length: 1, byteOffset});
  graph.add(
    new GPUTerrainFlow({
      grid,
      elevation: makeGPURasterBand(
        importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', cellCount),
        grid,
        {id: 'elevation'}
      ),
      parameters: settingsBuffer.importToGraph(graph),
      cellSizeMode: options.cellSizeMode,
      fillDepressions,
      maximumFillIterations: options.maximumFillIterations,
      maximumAccumulationIterations: options.maximumAccumulationIterations,
      accumulationUnits: options.accumulationUnits,
      runoff: runoffBuffer
        ? importGraphBuffer(graph, 'runoff', runoffBuffer, 'float32', cellCount)
        : undefined,
      output: {
        filledElevation: fillDepressions
          ? importGraphBuffer(graph, 'filled', buffers.filled, 'float32', cellCount)
          : undefined,
        flowDirections: importGraphBuffer(
          graph,
          'directions',
          buffers.directions,
          'uint32',
          cellCount
        ),
        cellClasses: importGraphBuffer(graph, 'classes', buffers.classes, 'uint32', cellCount),
        accumulation: importGraphBuffer(
          graph,
          'accumulation',
          buffers.accumulation,
          'float32',
          cellCount
        ),
        streams: importGraphBuffer(graph, 'streams', buffers.streams, 'uint32', cellCount),
        status: {
          fillConverged: fillDepressions
            ? summary
              ? flagAt(0)
              : importGraphBuffer(graph, 'fill-converged', buffers.fillConverged, 'uint32', 1)
            : undefined,
          accumulationConverged: summary
            ? flagAt(4)
            : importGraphBuffer(
                graph,
                'accumulation-converged',
                buffers.accumulationConverged,
                'uint32',
                1
              )
        }
      }
    })
  );
  const compiled = graph.compile();
  return {
    compiled,
    settingsBuffer,
    elevationBuffer,
    async run(nextSettings?: GPUTerrainFlowSettings): Promise<FlowResult> {
      if (nextSettings) {
        settingsBuffer.write(getGPUTerrainFlowParameterValues(nextSettings, grid));
      }
      submitGraph(device, compiled, undefined);
      return {
        filled: Float32Array.from(await readFloat32(buffers.filled, cellCount)),
        directions: Uint32Array.from(await readUint32(buffers.directions, cellCount)),
        classes: Uint32Array.from(await readUint32(buffers.classes, cellCount)),
        receivers: new Uint32Array(0),
        accumulation: Float32Array.from(await readFloat32(buffers.accumulation, cellCount)),
        streams: Uint32Array.from(await readUint32(buffers.streams, cellCount)),
        fillConverged: (await readUint32(buffers.fillConverged, 1))[0],
        accumulationConverged: options.sharedSummaryBuffer
          ? (await readUint32(buffers.fillConverged, 2))[1]
          : (await readUint32(buffers.accumulationConverged, 1))[0]
      };
    },
    destroy: () => {
      compiled.destroy();
      settingsBuffer.destroy();
      elevationBuffer.destroy();
      runoffBuffer?.destroy();
      for (const buffer of Object.values(buffers)) {
        buffer.destroy();
      }
    }
  };
}

/** Float32 unit roundoff `u = 2^-24`. */
const FLOAT32_UNIT_ROUNDOFF = 2 ** -24;

/**
 * Per-cell bound on |GPU - oracle| for D8 accumulation whose per-cell weights differ by at most
 * `weightRelativeError` while both sides sum in the same float32 order.
 *
 * Both sides add a cell's own weight and then its donors in direction order, rounding after every
 * addition. A term entering the sum at cell `c` is rounded `R(c)` times at most, where for donors
 * `d_1..d_m` of `c`, `R(c) = max(m, max_j (m - j + 1 + R(d_j)))`. With non-negative terms the
 * standard recursive-summation bound gives `|fl(sum) - sum| <= gamma_R * sum`,
 * `gamma_R = R u / (1 - R u)`, on each side; the weights add `weightRelativeError * sum`. So
 * `|GPU - oracle| <= A (1 + gamma_R) (2 gamma_R + weightRelativeError)` with `A` the oracle value.
 */
function getAccumulationErrorBounds(
  oracle: TerrainFlowOracleResult,
  width: number,
  height: number,
  weightRelativeError: number
): Float64Array {
  const cellCount = width * height;
  const columnOffsets = [1, 1, 0, -1, -1, -1, 0, 1];
  const rowOffsets = [0, 1, 1, 1, 0, -1, -1, -1];
  const roundings = new Int32Array(cellCount).fill(-1);
  const getRoundings = (cell: number): number => {
    if (roundings[cell] >= 0) {
      return roundings[cell];
    }
    const donors: number[] = [];
    for (let direction = 0; direction < 8; direction++) {
      const column = (cell % width) + columnOffsets[direction];
      const row = Math.floor(cell / width) + rowOffsets[direction];
      if (column < 0 || row < 0 || column >= width || row >= height) continue;
      const neighbor = row * width + column;
      if (oracle.receivers[neighbor] === cell) donors.push(neighbor);
    }
    let result = donors.length;
    for (const [index, donor] of donors.entries()) {
      result = Math.max(result, donors.length - index + getRoundings(donor));
    }
    roundings[cell] = result;
    return result;
  };
  return Float64Array.from(oracle.accumulation, (value, cell) => {
    if (!Number.isFinite(value)) return NaN;
    const roundingCount = getRoundings(cell) * FLOAT32_UNIT_ROUNDOFF;
    const gamma = roundingCount / (1 - roundingCount);
    return Math.abs(value) * (1 + gamma) * (2 * gamma + weightRelativeError);
  });
}

async function expectMatchesOracle(
  device: Device,
  name: string,
  elevation: Float32Array,
  width: number,
  height: number,
  options: FlowOptions = {},
  /** Relative error budget of each GPU weight; omit for bit-exact accumulation. */
  weightRelativeError?: number
) {
  const settings = options.settings ?? DEFAULT_SETTINGS;
  const fixture = createFlowFixture(device, elevation, width, height, options);
  const result = await fixture.run();
  const oracle = computeTerrainFlow({
    elevation,
    grid: getFlowGrid(width, height, options),
    settings,
    cellSizeMode: options.cellSizeMode,
    fillDepressions: options.fillDepressions,
    runoff: options.runoff,
    area: options.accumulationUnits === 'area'
  });
  expect(Array.from(result.directions), `${name} directions`).toEqual(
    Array.from(oracle.directions)
  );
  expect(Array.from(result.classes), `${name} classes`).toEqual(Array.from(oracle.classes));
  expect(Array.from(result.streams), `${name} streams`).toEqual(Array.from(oracle.streams));
  if (options.fillDepressions) {
    expect(toBits(result.filled), `${name} filled`).toEqual(toBits(oracle.filled));
    expect(result.fillConverged, `${name} fillConverged`).toBe(1);
  }
  if (weightRelativeError === undefined) {
    expect(toBits(result.accumulation), `${name} accumulation`).toEqual(
      toBits(oracle.accumulation)
    );
  } else {
    const bounds = getAccumulationErrorBounds(oracle, width, height, weightRelativeError);
    for (const [cell, value] of oracle.accumulation.entries()) {
      if (Number.isNaN(value)) {
        expect(result.accumulation[cell]).toBeNaN();
      } else {
        expect(
          Math.abs(result.accumulation[cell] - value),
          `${name} accumulation ${cell}`
        ).toBeLessThanOrEqual(bounds[cell]);
      }
    }
  }
  expect(result.accumulationConverged, `${name} accumulationConverged`).toBe(1);
  fixture.destroy();
  return oracle;
}

const DEM_CASES: [string, (width: number, height: number) => Float32Array][] = [
  ['cone', DEMS.cone],
  ['inverted cone', DEMS.invertedCone],
  ['valley', DEMS.valley],
  ['tilted plane', DEMS.tiltedPlane],
  ['noise', (width, height) => DEMS.noise(width, height)],
  ['plateau', DEMS.plateau],
  ['nodata holes', (width, height) => DEMS.noDataHoles(width, height)]
];

for (const [width, height] of [
  [37, 29],
  [70, 45]
]) {
  it(`GPUTerrainFlow matches the oracle on synthetic DEMs ${width}x${height}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const [name, createDem] of DEM_CASES) {
      const elevation = createDem(width, height);
      await expectMatchesOracle(device, `${name} raw`, elevation, width, height);
      await expectMatchesOracle(device, `${name} fill 0`, elevation, width, height, {
        fillDepressions: true
      });
      await expectMatchesOracle(device, `${name} fill 0.25`, elevation, width, height, {
        fillDepressions: true,
        settings: {fillEpsilon: 0.25, streamThreshold: 5}
      });
    }
  });
}

it('GPUTerrainFlow handles a 1x1 grid and an all-invalid grid', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const oracle = await expectMatchesOracle(device, 'single', Float32Array.of(3), 1, 1);
  expect(Array.from(oracle.classes)).toEqual([CELL_CLASS.outlet]);
  await expectMatchesOracle(device, 'single fill', Float32Array.of(3), 1, 1, {
    fillDepressions: true
  });
  await expectMatchesOracle(device, 'invalid', Float32Array.of(NaN, NaN, NaN, NaN), 2, 2);
});

it('GPUTerrainFlow reports pits without fill and drains them with fill and epsilon', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 9;
  const height = 9;
  const elevation = DEMS.invertedCone(width, height);
  const center = 4 * width + 4;
  let fixture = createFlowFixture(device, elevation, width, height);
  let result = await fixture.run();
  expect(result.classes[center]).toBe(CELL_CLASS.pit);
  expect(result.directions[center]).toBe(0);
  fixture.destroy();

  // A rim of value 4 around the pit forces a nontrivial fill level.
  const options: FlowOptions = {
    fillDepressions: true,
    settings: {fillEpsilon: 0.25}
  } as const;
  fixture = createFlowFixture(device, elevation, width, height, options);
  result = await fixture.run();
  expect(result.classes[center]).toBe(CELL_CLASS.draining);
  expect(result.fillConverged).toBe(1);
  // Every interior cell now drains: the only terminal cells are boundary outlets.
  for (const [cell, cellClass] of result.classes.entries()) {
    const column = cell % width;
    const row = Math.floor(cell / width);
    if (column > 0 && row > 0 && column < width - 1 && row < height - 1) {
      expect(cellClass).toBe(CELL_CLASS.draining);
    }
  }
  // All water leaves the grid: outlet accumulation equals the cell count.
  let outletTotal = 0;
  for (const [cell, cellClass] of result.classes.entries()) {
    if (cellClass === CELL_CLASS.outlet) {
      outletTotal += result.accumulation[cell];
    }
  }
  expect(outletTotal).toBe(width * height);
  fixture.destroy();
});

it('GPUTerrainFlow treats nodata holes as outlets', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // 5x5 bowl whose center is nodata: its neighbors are boundary cells.
  const elevation = Float32Array.from({length: 25}, (_, cell) => (cell === 12 ? NaN : 10));
  elevation[7] = 12;
  const fixture = createFlowFixture(device, elevation, 5, 5);
  const result = await fixture.run();
  expect(result.classes[12]).toBe(CELL_CLASS.invalid);
  expect(result.accumulation[12]).toBeNaN();
  expect(result.directions[12]).toBe(0xffffffff);
  // Cell 7 (above the hole) drops toward lower neighbors; cell 11 beside the hole is an outlet or flat.
  expect(result.classes[11]).not.toBe(CELL_CLASS.pit);
  fixture.destroy();
});

it('GPUTerrainFlow honors the cell size mode', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Center 10; east 8 (drop 2) and south 8.5 (drop 1.5); everything else is high.
  const elevation = new Float32Array(9).fill(100);
  elevation[4] = 10;
  elevation[5] = 8;
  elevation[7] = 8.5;
  const uniform = await expectMatchesOracle(device, 'uniform', elevation, 3, 3);
  expect(uniform.directions[4]).toBe(1);
  const geographic: FlowOptions = {
    cellSizeMode: 'geographic',
    cellSize: [1, 0.25],
    origin: [0, 10]
  };
  const geographicOracle = await expectMatchesOracle(
    device,
    'geographic',
    elevation,
    3,
    3,
    geographic
  );
  expect(geographicOracle.directions[4]).toBe(4);
  await expectMatchesOracle(device, 'mercator', DEMS.noise(23, 17, 3), 23, 17, {
    cellSizeMode: 'web-mercator',
    cellSize: 100000,
    origin: [0, getOracleWebMercatorY(0.3)]
  });
});

it('GPUTerrainFlow weights accumulation by runoff and area', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 31;
  const height = 23;
  const elevation = DEMS.valley(width, height);
  const random = createSeededRandom(5);
  const runoff = Float32Array.from({length: width * height}, () => Math.floor(random() * 4) * 0.5);
  runoff[3] = -1;
  runoff[9] = NaN;
  await expectMatchesOracle(device, 'runoff', elevation, width, height, {runoff});
  await expectMatchesOracle(
    device,
    'area',
    elevation,
    width,
    height,
    {
      accumulationUnits: 'area',
      cellSizeMode: 'geographic',
      cellSize: [0.001, 0.1 / 23],
      origin: [0, 50],
      runoff
    },
    // Area weights: the GPU evaluates the WGS84 row area in float32 (mix, sin, cos, sqrt, two
    // divisions, five products; WGSL allows 2.5 ulp for division and inverse sqrt) while the oracle
    // rounds a float64 area once. A python3 -I emulation with correctly rounded float32 ops
    // (scratchpad q0ter/py/flow_area.py) gives per-row areas 0 to 3 ulps from the oracle and
    // reproduces the GPU's 2865064 at cell 480 exactly (oracle 2865063.5), with the same summation
    // order. The budget is 8 ulps of relative size 2^-23, i.e. 2^-20, and the summation part of
    // the bound comes from getAccumulationErrorBounds.
    2 ** -20
  );
});

it('GPUTerrainFlow follows per-frame changes without recompiling and is deterministic', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 37;
  const height = 29;
  const first = DEMS.noise(width, height, 1);
  const second = DEMS.noise(width, height, 2);
  const options: FlowOptions = {
    fillDepressions: true,
    settings: {streamThreshold: 3}
  };
  const fixture = createFlowFixture(device, first, width, height, options);
  const run1 = await fixture.run();
  const run2 = await fixture.run();
  expect(toBits(run2.accumulation)).toEqual(toBits(run1.accumulation));

  fixture.elevationBuffer.write(second);
  const settings = {fillEpsilon: 0.5, streamThreshold: 7} as const;
  const changed = await fixture.run(settings);
  const oracle = computeTerrainFlow({
    elevation: second,
    grid: getFlowGrid(width, height),
    settings,
    fillDepressions: true
  });
  expect(Array.from(changed.directions)).toEqual(Array.from(oracle.directions));
  expect(toBits(changed.accumulation)).toEqual(toBits(oracle.accumulation));
  expect(Array.from(changed.streams)).toEqual(Array.from(oracle.streams));
  expect(changed.fillConverged).toBe(1);
  fixture.destroy();
});

it('GPUTerrainFlow reports fill and accumulation non-convergence', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Positive epsilon selects the capped relaxation. Interior cells change from +infinity in
  // its first round, so it needs another round to certify that no values changed.
  const width = 200;
  const height = 3;
  const bowl = Float32Array.from({length: width * height}, (_, cell) => {
    const column = cell % width;
    return column === 0 ? 0 : 50;
  });
  const bowlFixture = createFlowFixture(device, bowl, width, height, {
    fillDepressions: true,
    maximumFillIterations: 1,
    settings: {fillEpsilon: 0.125}
  });
  const bowlResult = await bowlFixture.run();
  expect(bowlResult.fillConverged).toBe(0);
  // Epsilon zero selects the exact spill graph, which ignores the relaxation iteration cap.
  const spillResult = await bowlFixture.run({fillEpsilon: 0});
  expect(spillResult.fillConverged).toBe(1);
  expect(toBits(spillResult.filled)).toEqual(toBits(bowl));
  bowlFixture.destroy();

  const length = 3000;
  const strip = Float32Array.from({length}, (_, row) => length - row);
  // Unit runoff selects the pull rounds: D8 'cells' accumulation without runoff runs on the Euler
  // tour, which ignores the limit (see terrain-flow-euler-tour.spec.ts).
  const shortFixture = createFlowFixture(device, strip, 1, length, {
    maximumAccumulationIterations: 1,
    runoff: new Float32Array(length).fill(1)
  });
  const shortResult = await shortFixture.run();
  if (shortResult.accumulationConverged === 1) {
    expect(shortResult.accumulation.every(value => Number.isFinite(value))).toBe(true);
  } else {
    expect(shortResult.accumulation.some(value => Number.isNaN(value))).toBe(true);
  }
  // Review C-03: an unresolved (NaN) cell is never reported as "not a stream".
  for (const [cell, value] of shortResult.accumulation.entries()) {
    expect(shortResult.streams[cell], `streams ${cell}`).toBe(
      Number.isNaN(value) ? GPU_TERRAIN_FLOW_STREAM_UNRESOLVED : 0
    );
  }
  shortFixture.destroy();

  await expectMatchesOracle(device, 'long strip', strip, 1, length, {
    maximumAccumulationIterations: 64
  });
});

it('GPUTerrainFlow publishes both convergence flags into one summary buffer', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const width = 37;
  const height = 29;
  const elevation = DEMS.noise(width, height);
  const options: FlowOptions = {
    fillDepressions: true,
    settings: {fillEpsilon: 0.25}
  } as const;
  const separate = createFlowFixture(device, elevation, width, height, options);
  const expected = await separate.run();
  separate.destroy();

  const gpuDevice = device.handle as GPUDevice;
  gpuDevice.pushErrorScope('validation');
  const shared = createFlowFixture(device, elevation, width, height, {
    ...options,
    sharedSummaryBuffer: true
  });
  const actual = await shared.run();
  await device.handle.queue.onSubmittedWorkDone();
  const validationError = await gpuDevice.popErrorScope();
  expect(validationError?.message).toBeUndefined();
  expect(actual.fillConverged).toBe(1);
  expect(actual.accumulationConverged).toBe(1);
  expect(expected.fillConverged).toBe(1);
  expect(Array.from(actual.directions)).toEqual(Array.from(expected.directions));
  expect(Array.from(actual.classes)).toEqual(Array.from(expected.classes));
  expect(toBits(actual.filled)).toEqual(toBits(expected.filled));
  expect(toBits(actual.accumulation)).toEqual(toBits(expected.accumulation));
  shared.destroy();
});

it('GPUTerrainFlow fills a 256x256 serpentine corridor at default iterations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Before directional sweeps a fill front advanced about one tile per iteration and this corridor
  // (about 32k cells long) never converged within the default iteration budget.
  const size = 256;
  for (const mode of ['pit', 'drain'] as const) {
    const elevation = createSerpentineDEM(size, mode);
    const fixture = createFlowFixture(device, elevation, size, size, {fillDepressions: true});
    const actual = await fixture.run();
    expect(actual.fillConverged).toBe(1);
    expect(toBits(actual.filled)).toEqual(toBits(fillDepressionsPriorityFlood(elevation, size)));
    fixture.destroy();
  }
});

it('GPUTerrainFlow D8 directions and accumulation match pysheds', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Fixture: test/oracles/terrain-hydrology.py (pysheds Grid.flowdir and accumulation; pysheds'
  // terminal -1/-2 mapped to 0). Integer cell counts are exact in float32, so both checks are exact.
  for (const testCase of hydrologyReference.cases) {
    const fixture = createFlowFixture(
      device,
      Float32Array.from(testCase.elevation),
      testCase.width,
      testCase.height,
      {cellSize: testCase.cellSize as [number, number], maximumAccumulationIterations: 256}
    );
    const result = await fixture.run();
    expect(Array.from(result.directions), testCase.name).toEqual(testCase.directions);
    expect(Array.from(result.accumulation), testCase.name).toEqual(testCase.accumulation);
    expect(result.accumulationConverged, testCase.name).toBe(1);
    fixture.destroy();
  }
});
