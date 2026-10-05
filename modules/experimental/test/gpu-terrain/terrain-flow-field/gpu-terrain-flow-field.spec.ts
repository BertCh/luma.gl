// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  advanceParticlesOnCPU,
  createParticleAdvectionCPUState,
  getGPUParticleAdvectionParameterValues,
  getGPUParticleAdvectionWordParameterValues,
  GPUParticleAdvection
} from '../../../src/gpu-raster/particle-advection';
import type {GPUTerrainCellSizeMode} from '../../../src/gpu-terrain/terrain-analysis';
import {
  GPUTerrainFlowField,
  getGPUTerrainFlowFieldParameterValues,
  type GPUTerrainFlowFieldSettings
} from '../../../src/gpu-terrain/terrain-flow-field';
import {createSeededRandom} from '../hydrology/terrain-flow-oracle';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {computeTerrainFlowField} from './terrain-flow-field-oracle';

/**
 * Error bound against the float64 oracle, as a fraction of the wind speed. The kernel is a handful
 * of f32 operations (Metal may fuse FMAs); measured maxima on Dawn/Metal are 1.4e-7 (uniform) and
 * 1.6e-7 (web-mercator and geographic, which add the GPU `cosh`/`cos` of the ground cell width).
 */
const TOLERANCE: Record<GPUTerrainCellSizeMode, number> = {
  uniform: 1e-6,
  'web-mercator': 2e-6,
  geographic: 2e-6
};

const DEMS = {
  cone: (width: number, height: number) =>
    Float32Array.from({length: width * height}, (_, cell) => {
      const dx = (cell % width) - (width - 1) / 2;
      const dy = Math.floor(cell / width) - (height - 1) / 2;
      return 400 - 37.3 * Math.hypot(dx, dy);
    }),
  ridges: (width: number, height: number) =>
    Float32Array.from({length: width * height}, (_, cell) => {
      const column = cell % width;
      const row = Math.floor(cell / width);
      return 1200 + 150 * Math.sin(column * 0.31) * Math.cos(row * 0.17) + 3.7 * row;
    }),
  noise: (width: number, height: number) => {
    const random = createSeededRandom(13);
    return Float32Array.from({length: width * height}, () => 500 + 80 * random());
  },
  holes: (width: number, height: number) => {
    const random = createSeededRandom(29);
    return Float32Array.from({length: width * height}, () =>
      random() < 0.08 ? NaN : 900 + 120 * random()
    );
  }
} as const;

type FieldOptions = {
  settings: GPUTerrainFlowFieldSettings;
  cellSizeMode?: GPUTerrainCellSizeMode;
};

function createFieldFixture(
  device: Device,
  elevation: Float32Array,
  width: number,
  height: number,
  options: FieldOptions
) {
  const cellCount = width * height;
  const elevationBuffer = createInputBuffer(device, elevation);
  const output = createOutputBuffer(device, cellCount * 2);
  const settingsBuffer = new GPUParameterBuffer(device, {
    id: 'flow-field-settings',
    format: 'float32',
    length: 8,
    values: getGPUTerrainFlowFieldParameterValues(options.settings)
  });
  const graph = new GPUCommandGraph(device, {id: 'terrain-flow-field-test'});
  graph.add(
    new GPUTerrainFlowField({
      width,
      height,
      cellSizeMode: options.cellSizeMode,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', cellCount)
        }
      },
      settings: settingsBuffer.importToGraph(graph),
      velocities: importGraphBuffer(graph, 'velocities', output, 'float32x2', cellCount)
    })
  );
  const compiled = graph.compile();
  return {
    elevationBuffer,
    async run(nextSettings?: GPUTerrainFlowFieldSettings): Promise<Float32Array> {
      if (nextSettings) {
        settingsBuffer.write(getGPUTerrainFlowFieldParameterValues(nextSettings));
      }
      submitGraph(device, compiled, undefined);
      return Float32Array.from(await readFloat32(output, cellCount * 2));
    },
    destroy() {
      compiled.destroy();
      settingsBuffer.destroy();
      elevationBuffer.destroy();
      output.destroy();
    }
  };
}

function expectFieldClose(
  name: string,
  actual: Float32Array,
  expected: Float64Array,
  settings: GPUTerrainFlowFieldSettings,
  cellSizeMode: GPUTerrainCellSizeMode = 'uniform'
): number {
  const speed = Math.hypot(settings.wind[0], settings.wind[1]);
  const tolerance = TOLERANCE[cellSizeMode] * Math.max(speed, 1e-30);
  let maximumError = 0;
  for (let index = 0; index < expected.length; index++) {
    if (Number.isNaN(expected[index])) {
      expect(actual[index], `${name} ${index} NaN`).toBeNaN();
      continue;
    }
    const error = Math.abs(actual[index] - expected[index]);
    maximumError = Math.max(maximumError, error / Math.max(speed, 1e-30));
    expect(error, `${name} component ${index}`).toBeLessThanOrEqual(tolerance);
  }
  return maximumError;
}

it('GPUTerrainFlowField matches the float64 oracle on synthetic DEMs', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const cases: [string, FieldOptions][] = [
    ['uniform', {settings: {cellSize: [30, 20], wind: [6, -2.5]}}],
    ['exaggerated', {settings: {cellSize: [25, 25], wind: [-3, 4], verticalExaggeration: 3}}],
    [
      'geographic',
      {
        cellSizeMode: 'geographic',
        settings: {cellSize: [0.0003, 0.0002], northEdge: 46.8, southEdge: 46.79, wind: [5, 5]}
      }
    ],
    [
      'web-mercator',
      {
        cellSizeMode: 'web-mercator',
        settings: {cellSize: [40, 40], northEdge: 0.35, southEdge: 0.351, wind: [0, 10]}
      }
    ]
  ];
  let largestDeflection = 0;
  for (const [width, height] of [
    [37, 29],
    [70, 45]
  ]) {
    for (const [demName, createDem] of Object.entries(DEMS)) {
      const elevation = createDem(width, height);
      for (const [caseName, options] of cases) {
        const name = `${demName} ${caseName} ${width}x${height}`;
        const fixture = createFieldFixture(device, elevation, width, height, options);
        const actual = await fixture.run();
        const expected = computeTerrainFlowField({
          elevation,
          width,
          height,
          settings: options.settings,
          cellSizeMode: options.cellSizeMode
        });
        expectFieldClose(name, actual, expected, options.settings, options.cellSizeMode);
        const [windX, windY] = options.settings.wind;
        for (let cell = 0; cell < width * height; cell++) {
          const deflection = Math.hypot(actual[2 * cell] - windX, actual[2 * cell + 1] - windY);
          if (Number.isFinite(deflection)) {
            largestDeflection = Math.max(largestDeflection, deflection);
          }
        }
        fixture.destroy();
      }
    }
  }
  // The terrain really bends the wind (a shader that silently wrote the wind would give 0).
  expect(largestDeflection).toBeGreaterThan(1);
});

it('GPUTerrainFlowField keeps wind on flat ground and across slopes, and slows it up slopes', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 9;
  const height = 7;
  const flat = new Float32Array(width * height).fill(250);
  let fixture = createFieldFixture(device, flat, width, height, {
    settings: {cellSize: [10, 10], wind: [3.25, -1.5]}
  });
  let field = await fixture.run();
  for (let cell = 0; cell < width * height; cell++) {
    expect([field[2 * cell], field[2 * cell + 1]]).toEqual([3.25, -1.5]);
  }
  fixture.destroy();

  // A plane rising along columns with gradient 2: wind along rows is untouched, wind along
  // columns keeps 1 / (1 + 4) of its speed.
  const plane = Float32Array.from({length: width * height}, (_, cell) => 20 * (cell % width));
  fixture = createFieldFixture(device, plane, width, height, {
    settings: {cellSize: [10, 10], wind: [0, 4]}
  });
  field = await fixture.run();
  for (let cell = 0; cell < width * height; cell++) {
    expect([field[2 * cell], field[2 * cell + 1]]).toEqual([0, 4]);
  }
  field = await fixture.run({cellSize: [10, 10], wind: [5, 0]});
  for (let cell = 0; cell < width * height; cell++) {
    expect(field[2 * cell]).toBeCloseTo(1, 6);
    expect(field[2 * cell + 1]).toBe(0);
  }
  fixture.destroy();
});

it('GPUTerrainFlowField marks cells next to nodata and handles single-row grids', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 7;
  const height = 5;
  const elevation = Float32Array.from({length: width * height}, (_, cell) => cell * 0.5);
  const hole = 2 * width + 3;
  elevation[hole] = NaN;
  const settings: GPUTerrainFlowFieldSettings = {cellSize: [1, 1], wind: [1, 1]};
  const fixture = createFieldFixture(device, elevation, width, height, {settings});
  const field = await fixture.run();
  const invalid = [hole, hole - 1, hole + 1, hole - width, hole + width];
  for (let cell = 0; cell < width * height; cell++) {
    const isInvalid = invalid.includes(cell);
    expect(Number.isNaN(field[2 * cell]), `cell ${cell}`).toBe(isInvalid);
    expect(Number.isNaN(field[2 * cell + 1]), `cell ${cell}`).toBe(isInvalid);
  }
  expectFieldClose(
    'hole',
    field,
    computeTerrainFlowField({elevation, width, height, settings}),
    settings
  );
  fixture.destroy();

  for (const [rowWidth, rowHeight] of [
    [12, 1],
    [1, 12]
  ]) {
    const line = Float32Array.from({length: 12}, (_, index) => index * index * 0.75);
    const lineSettings: GPUTerrainFlowFieldSettings = {cellSize: [2, 3], wind: [1.5, -2]};
    const lineFixture = createFieldFixture(device, line, rowWidth, rowHeight, {
      settings: lineSettings
    });
    expectFieldClose(
      `line ${rowWidth}x${rowHeight}`,
      await lineFixture.run(),
      computeTerrainFlowField({
        elevation: line,
        width: rowWidth,
        height: rowHeight,
        settings: lineSettings
      }),
      lineSettings
    );
    lineFixture.destroy();
  }
});

it('GPUTerrainFlowField follows per-frame elevation and wind changes without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 33;
  const height = 21;
  const first = DEMS.ridges(width, height);
  const second = DEMS.cone(width, height);
  const settings: GPUTerrainFlowFieldSettings = {cellSize: [15, 15], wind: [2, 7]};
  const fixture = createFieldFixture(device, first, width, height, {settings});
  const run1 = await fixture.run();
  const run2 = await fixture.run();
  expect(Array.from(run2)).toEqual(Array.from(run1));
  fixture.elevationBuffer.write(second);
  const nextSettings: GPUTerrainFlowFieldSettings = {
    cellSize: [12, 18],
    wind: [-9, 1],
    verticalExaggeration: 2
  };
  expectFieldClose(
    'changed',
    await fixture.run(nextSettings),
    computeTerrainFlowField({elevation: second, width, height, settings: nextSettings}),
    nextSettings
  );
  fixture.destroy();
});

it('GPUTerrainFlowField feeds GPUParticleAdvection in one graph', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 48;
  const height = 40;
  const cellCount = width * height;
  const particleCount = 512;
  const elevation = DEMS.ridges(width, height);
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const elevationBuffer = track(createInputBuffer(device, elevation));
  const fieldBuffer = track(createOutputBuffer(device, cellCount * 2));
  const positions = track(createOutputBuffer(device, particleCount * 2));
  const ages = track(createOutputBuffer(device, particleCount));
  const generations = track(createOutputBuffer(device, particleCount));
  const fieldSettings = new GPUParameterBuffer(device, {
    id: 'field-settings',
    format: 'float32',
    length: 8,
    values: getGPUTerrainFlowFieldParameterValues({cellSize: [20, 20], wind: [3, 1]})
  });
  const particleParameters = new GPUParameterBuffer(device, {
    id: 'particle-parameters',
    format: 'float32',
    length: 12
  });
  const particleWords = new GPUParameterBuffer(device, {
    id: 'particle-words',
    format: 'uint32',
    length: 4
  });
  const graph = new GPUCommandGraph(device, {id: 'terrain-wind-particles'});
  const velocities = importGraphBuffer(graph, 'wind-field', fieldBuffer, 'float32x2', cellCount);
  graph.add(
    new GPUTerrainFlowField({
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
      settings: fieldSettings.importToGraph(graph),
      velocities
    })
  );
  graph.add(
    new GPUParticleAdvection({
      velocities,
      fieldWidth: width,
      fieldHeight: height,
      parameters: particleParameters.importToGraph(graph),
      wordParameters: particleWords.importToGraph(graph),
      state: {
        positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', particleCount),
        ages: importGraphBuffer(graph, 'ages', ages, 'uint32', particleCount),
        generations: importGraphBuffer(graph, 'generations', generations, 'uint32', particleCount)
      }
    })
  );
  const compiled = graph.compile();
  // Particles live in raster-local cell coordinates, the frame the field is written in.
  const parameterValues = getGPUParticleAdvectionParameterValues(
    {fieldExtent: [0, 0, 1, 1], timeStep: 0.5},
    [width, height]
  );
  particleParameters.write(parameterValues);
  const cpuState = createParticleAdvectionCPUState(particleCount);
  let field: Float32Array | undefined;
  let moved = 0;
  for (let frame = 0; frame < 4; frame++) {
    const words = getGPUParticleAdvectionWordParameterValues({
      seed: 5,
      frame,
      maximumAge: 0,
      reset: frame === 0
    });
    particleWords.write(words);
    submitGraph(device, compiled, undefined);
    // The field is static, so the CPU replay uses the GPU-written field read back once.
    field ??= Float32Array.from(await readFloat32(fieldBuffer, cellCount * 2));
    const before = cpuState.positions.slice();
    advanceParticlesOnCPU(cpuState, {velocities: field, width, height}, parameterValues, words);
    if (frame > 0) {
      for (let index = 0; index < particleCount * 2; index++) {
        moved = Math.max(moved, Math.abs(cpuState.positions[index] - before[index]));
      }
    }
    const gpuPositions = await readFloat32(positions, particleCount * 2);
    for (let index = 0; index < particleCount * 2; index++) {
      expect(Math.abs(gpuPositions[index] - cpuState.positions[index])).toBeLessThan(2e-4);
    }
  }
  expect(Array.from(await readUint32(generations, particleCount)).length).toBe(particleCount);
  expect(moved).toBeGreaterThan(0.5);
  compiled.destroy();
  fieldSettings.destroy();
  particleParameters.destroy();
  particleWords.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
});
