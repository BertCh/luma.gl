// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPUCostDistance,
  getGPUCostDistanceParameterValues
} from '../../../src/gpu-raster/cost-distance';
import {makeGPURasterBand, makeGPURasterMetadata} from '../../../src/gpu-raster';
import {GPUTerrainFlow, getGPUTerrainFlowParameterValues} from '../../../src/gpu-terrain/hydrology';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

function createFixture(device: Device, recording: boolean) {
  const graph = new GPUCommandGraph(device, {
    id: 'recording-equivalence',
    ...(recording ? {recording: {}} : {})
  });
  const buffers: Buffer[] = [];
  const createInput = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    values: Float32Array | Uint32Array
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, format, values.length);
  };
  const createOutput = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  return {
    graph,
    createInput,
    createOutput,
    destroy: () => buffers.forEach(buffer => buffer.destroy())
  };
}

it('cost recording and settledAt preserve cost bits and convergence on a partial-tile raster', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const width = 40;
  const height = 37;
  const cellCount = width * height;
  const grid = makeGPURasterMetadata({width, height});
  const friction = new Float32Array(cellCount).fill(1);
  // Both sources are left of a full-height wall; cells to its right remain unreachable.
  const wallColumn = 30;
  for (let row = 0; row < height; row++) friction[row * width + wallColumn] = -1;
  const sources = [3 * width + 3, 31 * width + 22];
  const run = async (recording: boolean) => {
    const fixture = createFixture(device, recording);
    const {graph, createInput, createOutput} = fixture;
    const costs = createOutput('costs', 'float32', cellCount);
    const converged = createOutput('converged', 'uint32', 1);
    const iterationCount = createOutput('iterations', 'uint32', 1);
    const settledAt = recording ? createOutput('settled-at', 'uint32', cellCount) : undefined;
    // Prove the reset clears old diagnostics, including sources, barriers and unreachable cells.
    settledAt?.buffer.write(new Uint32Array(cellCount).fill(0xffffffff));
    graph.add(
      new GPUCostDistance({
        grid,
        friction: makeGPURasterBand(createInput('friction', 'float32', friction), grid, {
          id: 'friction'
        }),
        parameters: createInput('settings', 'float32', getGPUCostDistanceParameterValues({}, grid)),
        sources: createInput('sources', 'uint32', Uint32Array.from(sources)),
        maximumIterations: 64,
        output: {
          costs: costs.view,
          ...(settledAt ? {settledAt: settledAt.view} : {}),
          status: {converged: converged.view, iterationCount: iterationCount.view}
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const [costBits, convergence, iterations, stamps] = await Promise.all([
        readUint32(costs.buffer, cellCount),
        readUint32(converged.buffer, 1),
        readUint32(iterationCount.buffer, 1),
        settledAt ? readUint32(settledAt.buffer, cellCount) : Promise.resolve([])
      ]);
      return {costBits, convergence, iterations, stamps};
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  };
  const plain = await run(false);
  const recorded = await run(true);
  expect(recorded.costBits).toEqual(plain.costBits);
  expect(recorded.convergence).toEqual(plain.convergence);
  expect(recorded.iterations).toEqual(plain.iterations);
  expect(recorded.convergence).toEqual([1]);
  const costs = new Float32Array(Uint32Array.from(recorded.costBits).buffer);
  for (let cell = 0; cell < cellCount; cell++) {
    const stamp = recorded.stamps[cell];
    if (sources.includes(cell) || !Number.isFinite(costs[cell])) {
      expect(stamp, `cell ${cell}`).toBe(0);
    } else {
      expect(stamp, `cell ${cell}`).toBeGreaterThanOrEqual(1);
    }
    if (cell % width >= wallColumn) {
      expect(costs[cell]).toBe(Infinity);
      expect(stamp).toBe(0);
    } else {
      expect(Number.isFinite(costs[cell])).toBe(true);
    }
  }
  expect(Math.max(...recorded.stamps)).toBeLessThanOrEqual(recorded.iterations[0]);
});

it('terrain D8 fill, flats and accumulation keep the same fixpoint and status with recording', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const width = 9;
  const height = 9;
  const cellCount = width * height;
  const grid = makeGPURasterMetadata({width, height});
  // A low central pit, a broad flat interior, a raised rim and one low outlet.
  const elevation = new Float32Array(cellCount).fill(4);
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      if (row === 0 || column === 0 || row === height - 1 || column === width - 1) {
        elevation[row * width + column] = 5;
      }
    }
  }
  elevation[4 * width + 4] = 0;
  elevation[8 * width + 4] = 3;
  const run = async (recording: boolean) => {
    const fixture = createFixture(device, recording);
    const {graph, createInput, createOutput} = fixture;
    const filledElevation = createOutput('filled', 'float32', cellCount);
    const flowDirections = createOutput('directions', 'uint32', cellCount);
    const accumulation = createOutput('accumulation', 'float32', cellCount);
    const fillConverged = createOutput('fill-converged', 'uint32', 1);
    const flatsConverged = createOutput('flats-converged', 'uint32', 1);
    const accumulationConverged = createOutput('accumulation-converged', 'uint32', 1);
    const fillIterationCount = createOutput('fill-iterations', 'uint32', 1);
    const flatsIterationCount = createOutput('flats-iterations', 'uint32', 1);
    const accumulationIterationCount = createOutput('accumulation-iterations', 'uint32', 1);
    graph.add(
      new GPUTerrainFlow({
        grid,
        elevation: makeGPURasterBand(createInput('elevation', 'float32', elevation), grid, {
          id: 'elevation'
        }),
        parameters: createInput(
          'settings',
          'float32',
          getGPUTerrainFlowParameterValues({fillEpsilon: 0.125}, grid)
        ),
        flowRouting: 'd8',
        fillDepressions: true,
        resolveFlats: true,
        // Explicit runoff selects the gated pull accumulation path.
        runoff: createInput('runoff', 'float32', new Float32Array(cellCount).fill(1)),
        maximumFillIterations: 64,
        maximumFlatIterations: 64,
        maximumAccumulationIterations: 128,
        output: {
          filledElevation: filledElevation.view,
          flowDirections: flowDirections.view,
          accumulation: accumulation.view,
          status: {
            fillConverged: fillConverged.view,
            flatsConverged: flatsConverged.view,
            accumulationConverged: accumulationConverged.view,
            fillIterationCount: fillIterationCount.view,
            flatsIterationCount: flatsIterationCount.view,
            accumulationIterationCount: accumulationIterationCount.view
          }
        }
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const [filledBits, directions, accumulationBits, convergence, iterations] = await Promise.all(
        [
          readUint32(filledElevation.buffer, cellCount),
          readUint32(flowDirections.buffer, cellCount),
          readUint32(accumulation.buffer, cellCount),
          Promise.all(
            [fillConverged, flatsConverged, accumulationConverged].map(output =>
              readUint32(output.buffer, 1)
            )
          ),
          Promise.all(
            [fillIterationCount, flatsIterationCount, accumulationIterationCount].map(output =>
              readUint32(output.buffer, 1)
            )
          )
        ]
      );
      return {
        filledBits,
        directions,
        accumulationBits,
        convergence: convergence.flat(),
        iterations: iterations.flat()
      };
    } finally {
      compiled.destroy();
      fixture.destroy();
    }
  };
  const plain = await run(false);
  const recorded = await run(true);
  expect(recorded.filledBits).toEqual(plain.filledBits);
  expect(recorded.directions).toEqual(plain.directions);
  expect(recorded.accumulationBits).toEqual(plain.accumulationBits);
  expect(recorded.iterations).toEqual(plain.iterations);
  expect(recorded.convergence).toEqual(plain.convergence);
  expect(recorded.convergence).toEqual([1, 1, 1]);
  expect(recorded.iterations.every(iteration => iteration > 0)).toBe(true);
  expect(
    new Float32Array(Uint32Array.from(recorded.filledBits).buffer)[4 * width + 4]
  ).toBeGreaterThan(0);
});
