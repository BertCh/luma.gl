// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPUTerrainHydrologicIndices,
  getGPUTerrainHydrologicIndicesParameterValues,
  type GPUTerrainHydrologicIndicesSettings
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-hydrologic-indices';
import type {GPUTerrainCellSizeMode} from '../../../src/gpu-terrain/terrain-analysis';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  computeHydrologicIndices,
  type HydrologicIndicesOracleResult
} from './terrain-hydrologic-indices-oracle';
import {TERRAIN_FLOW_DEMS as DEMS, computeTerrainFlow} from './terrain-flow-oracle';

const OUTPUTS = ['slope', 'specificCatchmentArea', 'wetnessIndex', 'streamPowerIndex'] as const;

/**
 * Relative bound for slope, catchment and stream power (a few f32 operations plus the GPU
 * `sqrt`, and `cos`/`cosh` of the ground cell width on curved grids) and absolute bound for the
 * wetness index (`log` of those).
 */
const RELATIVE_TOLERANCE = 2e-6;
const WETNESS_TOLERANCE = 4e-6;

type IndicesOptions = {
  settings: GPUTerrainHydrologicIndicesSettings;
  cellSizeMode?: GPUTerrainCellSizeMode;
};

async function runIndices(
  device: Device,
  elevation: Float32Array,
  accumulation: Float32Array,
  width: number,
  height: number,
  options: IndicesOptions
): Promise<Record<(typeof OUTPUTS)[number], Float32Array>> {
  const cellCount = width * height;
  const elevationBuffer = createInputBuffer(device, elevation);
  const accumulationBuffer = createInputBuffer(device, accumulation);
  const outputs = Object.fromEntries(
    OUTPUTS.map(name => [name, createOutputBuffer(device, cellCount)])
  ) as Record<(typeof OUTPUTS)[number], ReturnType<typeof createOutputBuffer>>;
  const settingsBuffer = new GPUParameterBuffer(device, {
    id: 'indices-settings',
    format: 'float32',
    length: 8,
    values: getGPUTerrainHydrologicIndicesParameterValues(options.settings)
  });
  const graph = new GPUCommandGraph(device, {id: 'hydrologic-indices-test'});
  graph.add(
    new GPUTerrainHydrologicIndices({
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
      accumulation: importGraphBuffer(
        graph,
        'accumulation',
        accumulationBuffer,
        'float32',
        cellCount
      ),
      settings: settingsBuffer.importToGraph(graph),
      ...Object.fromEntries(
        OUTPUTS.map(name => [
          name,
          importGraphBuffer(graph, name, outputs[name], 'float32', cellCount)
        ])
      )
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {} as Record<(typeof OUTPUTS)[number], Float32Array>;
  for (const name of OUTPUTS) {
    result[name] = Float32Array.from(await readFloat32(outputs[name], cellCount));
    outputs[name].destroy();
  }
  compiled.destroy();
  settingsBuffer.destroy();
  elevationBuffer.destroy();
  accumulationBuffer.destroy();
  return result;
}

function expectIndicesClose(
  name: string,
  actual: Record<(typeof OUTPUTS)[number], Float32Array>,
  expected: HydrologicIndicesOracleResult
): void {
  for (const output of OUTPUTS) {
    for (const [cell, value] of expected[output].entries()) {
      const label = `${name} ${output} ${cell}`;
      if (!Number.isFinite(value)) {
        expect(actual[output][cell], label).toBe(value);
        continue;
      }
      const tolerance =
        output === 'wetnessIndex' ? WETNESS_TOLERANCE : RELATIVE_TOLERANCE * Math.abs(value);
      expect(Math.abs(actual[output][cell] - value), label).toBeLessThanOrEqual(tolerance);
    }
  }
}

it('GPUTerrainHydrologicIndices matches the float64 oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const cases: [string, IndicesOptions, GPUTerrainCellSizeMode | undefined][] = [
    ['uniform', {settings: {cellSize: [30, 30]}}, undefined],
    ['rectangular', {settings: {cellSize: [25, 40], minimumSlope: 0.01}}, undefined],
    [
      'geographic',
      {
        cellSizeMode: 'geographic',
        settings: {cellSize: [0.0003, 0.0002], northEdge: 46.8, southEdge: 46.79}
      },
      'geographic'
    ],
    [
      'web-mercator',
      {
        cellSizeMode: 'web-mercator',
        settings: {cellSize: [40, 40], northEdge: 0.35, southEdge: 0.351}
      },
      'web-mercator'
    ]
  ];
  let largestWetness = 0;
  for (const [width, height] of [
    [37, 29],
    [70, 45]
  ]) {
    for (const [demName, createDem] of Object.entries(DEMS)) {
      const elevation = createDem(width, height);
      for (const [caseName, options] of cases) {
        // Contributing area from the CPU flow oracle (the GPU flow contributor has its own tests).
        const flow = computeTerrainFlow({
          elevation,
          width,
          height,
          settings: {...options.settings, fillEpsilon: 0.25},
          cellSizeMode: options.cellSizeMode,
          fillDepressions: true,
          area: true
        });
        const name = `${demName} ${caseName} ${width}x${height}`;
        const actual = await runIndices(
          device,
          flow.filled,
          flow.accumulation,
          width,
          height,
          options
        );
        const expected = computeHydrologicIndices({
          elevation: flow.filled,
          accumulation: flow.accumulation,
          width,
          height,
          settings: options.settings,
          cellSizeMode: options.cellSizeMode
        });
        expectIndicesClose(name, actual, expected);
        for (const value of actual.wetnessIndex) {
          if (Number.isFinite(value)) {
            largestWetness = Math.max(largestWetness, value);
          }
        }
      }
    }
  }
  // Wetness grows along drainage lines; a silently failed shader would leave zeros.
  expect(largestWetness).toBeGreaterThan(5);
});

it('GPUTerrainHydrologicIndices handles flats, zero area, nodata, and the analytic plane', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 6;
  const height = 4;
  const cellCount = width * height;
  // A plane falling 3 m per 10 m column: tan(beta) = 0.3 everywhere except the lowest column,
  // which has no lower neighbor and takes the minimum slope.
  const plane = Float32Array.from({length: cellCount}, (_, cell) => 30 - 3 * (cell % width));
  const area = Float32Array.from({length: cellCount}, (_, cell) => 100 * (1 + (cell % width)));
  area[7] = 0;
  area[8] = NaN;
  plane[15] = NaN;
  const settings: GPUTerrainHydrologicIndicesSettings = {cellSize: [10, 10], minimumSlope: 0};
  const actual = await runIndices(device, plane, area, width, height, {settings});
  expectIndicesClose(
    'plane',
    actual,
    computeHydrologicIndices({elevation: plane, accumulation: area, width, height, settings})
  );
  expect(actual.slope[0]).toBeCloseTo(0.3, 6);
  expect(actual.specificCatchmentArea[0]).toBeCloseTo(10, 5);
  expect(actual.wetnessIndex[0]).toBeCloseTo(Math.log(10 / 0.3), 5);
  expect(actual.streamPowerIndex[0]).toBeCloseTo(3, 5);
  expect(actual.wetnessIndex[7]).toBe(-Infinity);
  expect(actual.wetnessIndex[8]).toBeNaN();
  expect(actual.slope[15]).toBeNaN();
  // The lowest column (index 5) has no lower neighbor: zero slope, infinite wetness, no power.
  expect(actual.slope[5]).toBe(0);
  expect(actual.wetnessIndex[5]).toBe(Infinity);
  expect(actual.streamPowerIndex[5]).toBe(0);

  const clamped = await runIndices(device, plane, area, width, height, {
    settings: {cellSize: [10, 10]}
  });
  expect(clamped.slope[5]).toBeCloseTo(0.001, 8);
  expect(clamped.wetnessIndex[5]).toBeCloseTo(Math.log(60 / 0.001), 4);
});
