// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPU_TERRAIN_FLOW_CELL_CLASS as CELL_CLASS,
  GPUTerrainFlow,
  getGPUTerrainFlowParameterValues,
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
  type TerrainFlowOracleResult
} from './terrain-flow-oracle';
import {toBits} from '../terrain-test-utils';

type FlatsOptions = {
  fillDepressions?: boolean;
  resolveFlats?: boolean;
  maxFlatIterations?: number;
  settings?: GPUTerrainFlowSettings;
};

type FlatsResult = Omit<TerrainFlowOracleResult, 'receivers' | 'filled'> & {flatsConverged: number};

const DEFAULT_SETTINGS: GPUTerrainFlowSettings = {cellSize: [1, 1], streamThreshold: 4};

/** Compiles and runs one GPUTerrainFlow graph with flat resolution outputs. */
async function runFlow(
  device: Device,
  elevation: Float32Array,
  width: number,
  height: number,
  options: FlatsOptions
): Promise<FlatsResult> {
  const cellCount = width * height;
  const settings = options.settings ?? DEFAULT_SETTINGS;
  const elevationBuffer = createInputBuffer(device, elevation);
  const buffers = {
    directions: createOutputBuffer(device, cellCount),
    classes: createOutputBuffer(device, cellCount),
    accumulation: createOutputBuffer(device, cellCount),
    streams: createOutputBuffer(device, cellCount),
    flatsConverged: createOutputBuffer(device, 1)
  };
  const settingsBuffer = new GPUParameterBuffer(device, {
    id: 'flats-settings',
    format: 'float32',
    length: 8,
    values: getGPUTerrainFlowParameterValues(settings)
  });
  const graph = new GPUCommandGraph(device, {id: 'terrain-flats-test'});
  const resolveFlats = options.resolveFlats ?? true;
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
      fillDepressions: options.fillDepressions ?? false,
      resolveFlats,
      maxFlatIterations: options.maxFlatIterations,
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
      flatsConverged: resolveFlats
        ? importGraphBuffer(graph, 'flats-converged', buffers.flatsConverged, 'uint32', 1)
        : undefined
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result: FlatsResult = {
    directions: Uint32Array.from(await readUint32(buffers.directions, cellCount)),
    classes: Uint32Array.from(await readUint32(buffers.classes, cellCount)),
    accumulation: Float32Array.from(await readFloat32(buffers.accumulation, cellCount)),
    streams: Uint32Array.from(await readUint32(buffers.streams, cellCount)),
    flatsConverged: (await readUint32(buffers.flatsConverged, 1))[0]
  };
  compiled.destroy();
  settingsBuffer.destroy();
  elevationBuffer.destroy();
  for (const buffer of Object.values(buffers)) {
    buffer.destroy();
  }
  return result;
}

function countClass(classes: Uint32Array, cellClass: number): number {
  let count = 0;
  for (const value of classes) {
    if (value === cellClass) {
      count++;
    }
  }
  return count;
}

/**
 * Asserts the Barnes guarantee on a result: no remaining flat cell touches an equal-surface
 * cell that has flow, otherwise it would have been routed.
 */
function expectNoRoutableFlatLeft(
  name: string,
  surface: Float32Array,
  classes: Uint32Array,
  width: number,
  height: number
) {
  for (let cell = 0; cell < classes.length; cell++) {
    if (classes[cell] !== CELL_CLASS.flat) {
      continue;
    }
    for (let dRow = -1; dRow <= 1; dRow++) {
      for (let dColumn = -1; dColumn <= 1; dColumn++) {
        const column = (cell % width) + dColumn;
        const row = Math.floor(cell / width) + dRow;
        if (column < 0 || row < 0 || column >= width || row >= height) {
          continue;
        }
        const neighbor = row * width + column;
        if (neighbor === cell || surface[neighbor] !== surface[cell]) {
          continue;
        }
        const neighborClass = classes[neighbor];
        expect(
          neighborClass === CELL_CLASS.draining || neighborClass === CELL_CLASS.outlet,
          `${name} flat ${cell} touches equal has-flow cell ${neighbor}`
        ).toBe(false);
      }
    }
  }
}

/** Runs the GPU and the oracle, compares them bit-exactly, and returns both plus the baseline. */
async function expectMatchesOracle(
  device: Device,
  name: string,
  elevation: Float32Array,
  width: number,
  height: number,
  options: FlatsOptions = {}
) {
  const result = await runFlow(device, elevation, width, height, options);
  const settings = options.settings ?? DEFAULT_SETTINGS;
  const common = {
    elevation,
    width,
    height,
    settings,
    fillDepressions: options.fillDepressions
  };
  const oracle = computeTerrainFlow({...common, resolveFlats: true});
  const baseline = computeTerrainFlow({...common, resolveFlats: false});
  expect(Array.from(result.directions), `${name} directions`).toEqual(
    Array.from(oracle.directions)
  );
  expect(Array.from(result.classes), `${name} classes`).toEqual(Array.from(oracle.classes));
  expect(Array.from(result.streams), `${name} streams`).toEqual(Array.from(oracle.streams));
  expect(toBits(result.accumulation), `${name} accumulation`).toEqual(toBits(oracle.accumulation));
  expect(result.flatsConverged, `${name} flatsConverged`).toBe(1);
  const surface = options.fillDepressions ? oracle.filled : elevation;
  expectNoRoutableFlatLeft(name, surface, oracle.classes, width, height);
  const resolvedCount =
    countClass(baseline.classes, CELL_CLASS.flat) - countClass(oracle.classes, CELL_CLASS.flat);
  return {result, oracle, baseline, resolvedCount};
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
  it(`GPUTerrainFlow resolveFlats matches the oracle on synthetic DEMs ${width}x${height}`, async () => {
    const device = await getWebGPUTestDevice();
    if (!device) {
      return;
    }
    let totalResolved = 0;
    for (const [name, createDem] of DEM_CASES) {
      const elevation = createDem(width, height);
      const raw = await expectMatchesOracle(device, `${name} raw`, elevation, width, height);
      const filled = await expectMatchesOracle(device, `${name} fill 0`, elevation, width, height, {
        fillDepressions: true
      });
      await expectMatchesOracle(device, `${name} fill 0.25`, elevation, width, height, {
        fillDepressions: true,
        settings: {cellSize: [1, 1], fillEpsilon: 0.25, streamThreshold: 5}
      });
      if (name === 'plateau' || name === 'noise') {
        expect(raw.resolvedCount, `${name} raw resolved flats`).toBeGreaterThan(0);
        expect(filled.resolvedCount, `${name} fill 0 resolved flats`).toBeGreaterThan(0);
      }
      totalResolved += raw.resolvedCount + filled.resolvedCount;
    }
    expect(totalResolved).toBeGreaterThan(20);
  });
}

it('GPUTerrainFlow resolveFlats drains a plateau through one boundary gap', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 14;
  const height = 12;
  const elevation = Float32Array.from({length: width * height}, (_, cell) => {
    const column = cell % width;
    const row = Math.floor(cell / width);
    const onRing = column === 0 || row === 0 || column === width - 1 || row === height - 1;
    return onRing ? 20 : 10;
  });
  const gap = 5 * width;
  elevation[gap] = 5;
  const {result, resolvedCount} = await expectMatchesOracle(
    device,
    'plateau gap',
    elevation,
    width,
    height
  );
  expect(resolvedCount).toBeGreaterThan(40);
  expect(result.classes[gap]).toBe(CELL_CLASS.outlet);
  expect(countClass(result.classes, CELL_CLASS.flat)).toBe(0);
  // The single outlet receives every cell.
  expect(result.accumulation[gap]).toBe(width * height);
});

it('GPUTerrainFlow resolveFlats routes away from the higher side of a flat', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Walls (30) on three sides, a plateau (10), and a low outlet column (5) on the right.
  const width = 24;
  const height = 7;
  const elevation = Float32Array.from({length: width * height}, (_, cell) => {
    const column = cell % width;
    const row = Math.floor(cell / width);
    if (column === width - 1) {
      return 5;
    }
    return column === 0 || row === 0 || row === height - 1 ? 30 : 10;
  });
  const {result, resolvedCount} = await expectMatchesOracle(
    device,
    'high side',
    elevation,
    width,
    height
  );
  expect(resolvedCount).toBeGreaterThan(50);
  // Cell (5, 1) touches the top wall. Without the away-from-higher term east and south-east tie
  // and east wins; with it the cell steps toward the middle of the flat (south-east, code 2).
  expect(result.directions[1 * width + 5]).toBe(2);
  // The middle row heads straight east.
  expect(result.directions[3 * width + 5]).toBe(1);
});

it('GPUTerrainFlow resolveFlats leaves a closed flat without an outlet', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const size = 9;
  const elevation = Float32Array.from({length: size * size}, (_, cell) => {
    const column = cell % size;
    const row = Math.floor(cell / size);
    return column === 0 || row === 0 || column === size - 1 || row === size - 1 ? 20 : 10;
  });
  const {result, baseline} = await expectMatchesOracle(device, 'closed', elevation, size, size);
  expect(countClass(result.classes, CELL_CLASS.flat)).toBe(49);
  expect(Array.from(result.classes)).toEqual(Array.from(baseline.classes));
  expect(result.flatsConverged).toBe(1);
});

it('GPUTerrainFlow resolveFlats leaves no interior flat after filling with epsilon 0', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const [name, elevation, width, height] of [
    ['noise', DEMS.noise(37, 29), 37, 29],
    ['noise 70x45', DEMS.noise(70, 45, 3), 70, 45],
    ['inverted cone', DEMS.invertedCone(31, 31), 31, 31]
  ] as const) {
    const {result, resolvedCount} = await expectMatchesOracle(
      device,
      `${name} fill 0`,
      elevation,
      width,
      height,
      {fillDepressions: true}
    );
    expect(resolvedCount, `${name} resolved`).toBeGreaterThan(0);
    for (let cell = 0; cell < width * height; cell++) {
      const column = cell % width;
      const row = Math.floor(cell / width);
      if (column > 0 && row > 0 && column < width - 1 && row < height - 1) {
        expect(result.classes[cell], `${name} interior ${cell}`).toBe(CELL_CLASS.draining);
      }
    }
    let outletTotal = 0;
    for (const [cell, cellClass] of result.classes.entries()) {
      if (cellClass === CELL_CLASS.outlet) {
        outletTotal += result.accumulation[cell];
      }
    }
    expect(outletTotal, `${name} outlet total`).toBe(width * height);
  }
});

it('GPUTerrainFlow reports flat resolution non-convergence', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // A 200x5 plateau enclosed by walls drains west through column 1 into column 0.
  const width = 200;
  const height = 5;
  const strip = Float32Array.from({length: width * height}, (_, cell) => {
    const column = cell % width;
    const row = Math.floor(cell / width);
    if (column === 0) {
      return 0;
    }
    return row === 0 || row === height - 1 || column === width - 1 ? 20 : 10;
  });
  const limited = await runFlow(device, strip, width, height, {maxFlatIterations: 1});
  expect(limited.flatsConverged).toBe(0);
  const full = await expectMatchesOracle(device, 'long strip', strip, width, height);
  expect(full.resolvedCount).toBeGreaterThan(400);
  expect(full.result.flatsConverged).toBe(1);
});

it('GPUTerrainFlow resolveFlats handles nodata holes inside a plateau', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 40;
  const height = 33;
  const elevation = DEMS.plateau(width, height);
  for (const cell of [500, 501, 541, 700, 701, 702, 742]) {
    elevation[cell] = NaN;
  }
  const raw = await expectMatchesOracle(device, 'plateau holes', elevation, width, height);
  expect(raw.resolvedCount).toBeGreaterThan(0);
  await expectMatchesOracle(device, 'plateau holes fill', elevation, width, height, {
    fillDepressions: true
  });
});
