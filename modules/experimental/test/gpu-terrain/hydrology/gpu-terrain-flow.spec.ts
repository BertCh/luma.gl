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

type FlowOptions = {
  settings?: GPUTerrainFlowSettings;
  fillDepressions?: boolean;
  runoff?: Float32Array;
  cellSizeMode?: GPUTerrainFlowProps['cellSizeMode'];
  accumulationUnits?: GPUTerrainFlowProps['accumulationUnits'];
  maxFillIterations?: number;
  maxAccumulationIterations?: number;
  /** Publish both convergence flags as 1-row views at byte offsets 0 and 4 of one 8-byte buffer. */
  sharedSummaryBuffer?: boolean;
};

type FlowResult = TerrainFlowOracleResult & {fillConverged: number; accumulationConverged: number};

const DEFAULT_SETTINGS: GPUTerrainFlowSettings = {cellSize: [1, 1]};

function createFlowFixture(
  device: Device,
  elevation: Float32Array,
  width: number,
  height: number,
  options: FlowOptions = {}
) {
  const cellCount = width * height;
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
    values: getGPUTerrainFlowParameterValues(settings)
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
      fillDepressions,
      maxFillIterations: options.maxFillIterations,
      maxAccumulationIterations: options.maxAccumulationIterations,
      accumulationUnits: options.accumulationUnits,
      runoff: runoffBuffer
        ? importGraphBuffer(graph, 'runoff', runoffBuffer, 'float32', cellCount)
        : undefined,
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
    })
  );
  const compiled = graph.compile();
  return {
    compiled,
    settingsBuffer,
    elevationBuffer,
    async run(nextSettings?: GPUTerrainFlowSettings): Promise<FlowResult> {
      if (nextSettings) {
        settingsBuffer.write(getGPUTerrainFlowParameterValues(nextSettings));
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

/** Bit pattern comparison that treats NaN as equal to NaN. */
function toBits(values: Float32Array): number[] {
  const bits = new Uint32Array(values.length);
  const floats = new Float32Array(bits.buffer);
  for (const [index, value] of values.entries()) {
    floats[index] = Number.isNaN(value) ? NaN : value;
  }
  return Array.from(bits);
}

async function expectMatchesOracle(
  device: Device,
  name: string,
  elevation: Float32Array,
  width: number,
  height: number,
  options: FlowOptions = {},
  tolerance?: number
) {
  const settings = options.settings ?? DEFAULT_SETTINGS;
  const fixture = createFlowFixture(device, elevation, width, height, options);
  const result = await fixture.run();
  const oracle = computeTerrainFlow({
    elevation,
    width,
    height,
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
  if (tolerance === undefined) {
    expect(toBits(result.accumulation), `${name} accumulation`).toEqual(
      toBits(oracle.accumulation)
    );
  } else {
    for (const [cell, value] of oracle.accumulation.entries()) {
      if (Number.isNaN(value)) {
        expect(result.accumulation[cell]).toBeNaN();
      } else {
        expect(result.accumulation[cell], `${name} accumulation ${cell}`).toBeCloseTo(
          value,
          Math.max(0, -Math.ceil(Math.log10(Math.abs(value) * tolerance + 1e-30)))
        );
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
    if (!device) {
      return;
    }
    for (const [name, createDem] of DEM_CASES) {
      const elevation = createDem(width, height);
      await expectMatchesOracle(device, `${name} raw`, elevation, width, height);
      await expectMatchesOracle(device, `${name} fill 0`, elevation, width, height, {
        fillDepressions: true
      });
      await expectMatchesOracle(device, `${name} fill 0.25`, elevation, width, height, {
        fillDepressions: true,
        settings: {cellSize: [1, 1], fillEpsilon: 0.25, streamThreshold: 5}
      });
    }
  });
}

it('GPUTerrainFlow handles a 1x1 grid and an all-invalid grid', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const oracle = await expectMatchesOracle(device, 'single', Float32Array.of(3), 1, 1);
  expect(Array.from(oracle.classes)).toEqual([CELL_CLASS.outlet]);
  await expectMatchesOracle(device, 'single fill', Float32Array.of(3), 1, 1, {
    fillDepressions: true
  });
  await expectMatchesOracle(device, 'invalid', Float32Array.of(NaN, NaN, NaN, NaN), 2, 2);
});

it('GPUTerrainFlow reports pits without fill and drains them with fill and epsilon', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
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
  const options = {fillDepressions: true, settings: {cellSize: [1, 1], fillEpsilon: 0.25}} as const;
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
  if (!device) {
    return;
  }
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
  if (!device) {
    return;
  }
  // Center 10; east 8 (drop 2) and south 8.5 (drop 1.5); everything else is high.
  const elevation = new Float32Array(9).fill(100);
  elevation[4] = 10;
  elevation[5] = 8;
  elevation[7] = 8.5;
  const uniform = await expectMatchesOracle(device, 'uniform', elevation, 3, 3);
  expect(uniform.directions[4]).toBe(1);
  const geographic: FlowOptions = {
    cellSizeMode: 'geographic',
    settings: {cellSize: [1, 0.25], northEdge: 10, southEdge: 0}
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
    settings: {cellSize: [1, 1.5], northEdge: 0.3, southEdge: 0.45}
  });
});

it('GPUTerrainFlow weights accumulation by runoff and area', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
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
      settings: {cellSize: [0.001, 0.001], northEdge: 50, southEdge: 49.9},
      runoff
    },
    1e-5
  );
});

it('GPUTerrainFlow follows per-frame changes without recompiling and is deterministic', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 37;
  const height = 29;
  const first = DEMS.noise(width, height, 1);
  const second = DEMS.noise(width, height, 2);
  const options = {fillDepressions: true, settings: {cellSize: [1, 1], streamThreshold: 3}};
  const fixture = createFlowFixture(device, first, width, height, options);
  const run1 = await fixture.run();
  const run2 = await fixture.run();
  expect(toBits(run2.accumulation)).toEqual(toBits(run1.accumulation));

  fixture.elevationBuffer.write(second);
  const settings = {cellSize: [1, 1], fillEpsilon: 0.5, streamThreshold: 7} as const;
  const changed = await fixture.run(settings);
  const oracle = computeTerrainFlow({
    elevation: second,
    width,
    height,
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
  if (!device) {
    return;
  }
  // A 200x3 channel drains from east to west; one fill iteration cannot lower +infinity far.
  const width = 200;
  const height = 3;
  const bowl = Float32Array.from({length: width * height}, (_, cell) => {
    const column = cell % width;
    return column === 0 ? 0 : 50;
  });
  const bowlFixture = createFlowFixture(device, bowl, width, height, {
    fillDepressions: true,
    maxFillIterations: 1
  });
  const bowlResult = await bowlFixture.run();
  expect(bowlResult.fillConverged).toBe(0);
  bowlFixture.destroy();

  const length = 3000;
  const strip = Float32Array.from({length}, (_, row) => length - row);
  const shortFixture = createFlowFixture(device, strip, 1, length, {maxAccumulationIterations: 1});
  const shortResult = await shortFixture.run();
  if (shortResult.accumulationConverged === 1) {
    expect(shortResult.accumulation.every(value => Number.isFinite(value))).toBe(true);
  } else {
    expect(shortResult.accumulation.some(value => Number.isNaN(value))).toBe(true);
  }
  shortFixture.destroy();

  await expectMatchesOracle(device, 'long strip', strip, 1, length, {
    maxAccumulationIterations: 64
  });
});

it('GPUTerrainFlow publishes both convergence flags into one summary buffer', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 37;
  const height = 29;
  const elevation = DEMS.noise(width, height);
  const options = {fillDepressions: true, settings: {cellSize: [1, 1], fillEpsilon: 0.25}} as const;
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
