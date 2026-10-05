// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUMapGraphParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {
  generateStreamlinesOnCPU,
  getGPUStreamlinesParameterValues,
  getGPUStreamlinesWordParameterValues,
  GPUStreamlines,
  traceStreamlineCandidatesOnCPU,
  type GPUStreamlinesSettings,
  type StreamlineCandidates,
  type StreamlinesCPUConfig
} from '../../../src/map-graphs/flow-texture';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../map-graph-test-utils';
import {createUniformField, createVortexField, type TestField} from './flow-texture-scenes';

type Result = {
  ids: number[];
  pathOffsets: number[];
  points: Float32Array;
  count: number;
  overflow: number;
  totalCount: number;
  pointCount: number;
  unconverged: number;
  candidates: StreamlineCandidates;
  keptSpans: Uint32Array;
};

type Fixture = {
  run(settings: GPUStreamlinesSettings): Promise<Result>;
  destroy(): void;
};

function createFixture(
  device: Device,
  field: TestField,
  config: StreamlinesCPUConfig,
  roundCount: number
): Fixture {
  const seedCount = config.seedColumns * config.seedRows;
  const stride = 2 * config.stepsPerDirection + 1;
  const graph = new GPUCommandGraph(device, {id: 'streamlines-graph'});
  const buffers: Buffer[] = [];
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return buffer;
  };
  const parameterBuffer = new GPUMapGraphParameterBuffer(device, {
    id: 'streamline-parameters',
    format: 'float32',
    length: 12
  });
  const wordBuffer = new GPUMapGraphParameterBuffer(device, {
    id: 'streamline-words',
    format: 'uint32',
    length: 4
  });
  const fieldBuffer = createInputBuffer(device, field.velocities);
  buffers.push(fieldBuffer);
  const outputs = {
    ids: output(config.lineCapacity),
    count: output(1),
    overflow: output(1),
    total: output(1),
    pathOffsets: output(config.lineCapacity + 1),
    points: output(config.pointCapacity * 2),
    pointCount: output(1),
    unconverged: output(1),
    candidates: output(seedCount * stride * 2),
    spans: output(seedCount * 4)
  };
  graph.add(
    new GPUStreamlines({
      id: 'streamlines',
      velocities: importGraphBuffer(
        graph,
        'field',
        fieldBuffer,
        'float32x2',
        field.width * field.height
      ),
      fieldWidth: field.width,
      fieldHeight: field.height,
      gridWidth: config.gridWidth,
      gridHeight: config.gridHeight,
      seedColumns: config.seedColumns,
      seedRows: config.seedRows,
      stepsPerDirection: config.stepsPerDirection,
      roundCount,
      parameters: parameterBuffer.importToGraph(graph),
      wordParameters: wordBuffer.importToGraph(graph),
      output: {
        lines: {
          ids: importGraphBuffer(graph, 'ids', outputs.ids, 'uint32', config.lineCapacity),
          count: importGraphBuffer(graph, 'count', outputs.count, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'overflow', outputs.overflow, 'uint32', 1),
          totalCount: importGraphBuffer(graph, 'total', outputs.total, 'uint32', 1)
        },
        pathOffsets: importGraphBuffer(
          graph,
          'path-offsets',
          outputs.pathOffsets,
          'uint32',
          config.lineCapacity + 1
        ),
        points: importGraphBuffer(
          graph,
          'points',
          outputs.points,
          'float32x2',
          config.pointCapacity
        ),
        pointCount: importGraphBuffer(graph, 'point-count', outputs.pointCount, 'uint32', 1),
        unconverged: importGraphBuffer(graph, 'unconverged', outputs.unconverged, 'uint32', 1),
        candidates: {
          points: importGraphBuffer(
            graph,
            'candidates',
            outputs.candidates,
            'float32x2',
            seedCount * stride
          ),
          spans: importGraphBuffer(graph, 'spans', outputs.spans, 'uint32', seedCount * 4)
        }
      }
    })
  );
  const compiled = graph.compile();
  return {
    async run(settings) {
      parameterBuffer.write(getGPUStreamlinesParameterValues(settings));
      wordBuffer.write(getGPUStreamlinesWordParameterValues(settings));
      submitGraph(device, compiled, undefined);
      const [count] = await readUint32(outputs.count, 1);
      const [pointCount] = await readUint32(outputs.pointCount, 1);
      const spans = await readUint32(outputs.spans, seedCount * 4);
      const candidateSpans = new Uint32Array(seedCount * 2);
      const keptSpans = new Uint32Array(seedCount * 2);
      for (let seed = 0; seed < seedCount; seed++) {
        candidateSpans.set(spans.slice(4 * seed, 4 * seed + 2), 2 * seed);
        keptSpans.set(spans.slice(4 * seed + 2, 4 * seed + 4), 2 * seed);
      }
      return {
        count,
        pointCount,
        overflow: (await readUint32(outputs.overflow, 1))[0],
        totalCount: (await readUint32(outputs.total, 1))[0],
        unconverged: (await readUint32(outputs.unconverged, 1))[0],
        ids: await readUint32(outputs.ids, count),
        pathOffsets: await readUint32(outputs.pathOffsets, count + 1),
        points: Float32Array.from(await readFloat32(outputs.points, pointCount * 2)),
        candidates: {
          points: Float32Array.from(await readFloat32(outputs.candidates, seedCount * stride * 2)),
          spans: candidateSpans
        },
        keptSpans
      };
    },
    destroy() {
      compiled.destroy();
      parameterBuffer.destroy();
      wordBuffer.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

function expectPruningMatchesOracle(
  actual: Result,
  field: TestField,
  config: StreamlinesCPUConfig,
  settings: GPUStreamlinesSettings
) {
  // Prune the GPU's own traced candidates on the CPU: the result must be identical.
  const expected = generateStreamlinesOnCPU(
    field,
    config,
    getGPUStreamlinesParameterValues(settings),
    getGPUStreamlinesWordParameterValues(settings),
    actual.candidates
  );
  expect(actual.unconverged).toBe(0);
  expect(actual.ids).toEqual(expected.ids);
  expect(actual.pathOffsets).toEqual(expected.pathOffsets);
  expect(new Uint32Array(actual.points.buffer)).toEqual(new Uint32Array(expected.points.buffer));
  expect(actual.totalCount).toBe(expected.totalCount);
  expect(actual.overflow).toBe(expected.overflow);
  expect(actual.pointCount).toBe(expected.pathOffsets[expected.ids.length]);
  return expected;
}

const VORTEX_CONFIG: StreamlinesCPUConfig = {
  seedColumns: 24,
  seedRows: 24,
  stepsPerDirection: 40,
  gridWidth: 48,
  gridHeight: 48,
  lineCapacity: 576,
  pointCapacity: 576 * 81
};

const VORTEX_SETTINGS: GPUStreamlinesSettings = {
  fieldExtent: [0, 0, 1, 1],
  gridExtent: [0, 0, 32 / 48, 32 / 48],
  stepLength: 0.3,
  minimumSpeed: 0.05,
  seed: 5,
  minimumPoints: 6
};

it('GPUStreamlines traces like the CPU oracle and prunes exactly like the greedy pass', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const field = createVortexField(32, 0.1);
  const fixture = createFixture(device, field, VORTEX_CONFIG, 64);
  try {
    const actual = await fixture.run(VORTEX_SETTINGS);
    // Trace: same step counts for nearly every seed, positions within a small drift.
    const traced = traceStreamlineCandidatesOnCPU(
      field,
      VORTEX_CONFIG,
      getGPUStreamlinesParameterValues(VORTEX_SETTINGS),
      getGPUStreamlinesWordParameterValues(VORTEX_SETTINGS)
    );
    let spanMismatches = 0;
    let maximumDrift = 0;
    const seedCount = 24 * 24;
    const stride = 81;
    for (let seed = 0; seed < seedCount; seed++) {
      const same =
        traced.spans[2 * seed] === actual.candidates.spans[2 * seed] &&
        traced.spans[2 * seed + 1] === actual.candidates.spans[2 * seed + 1];
      if (!same) {
        spanMismatches++;
        continue;
      }
      for (let slot = seed * stride; slot < (seed + 1) * stride; slot++) {
        const expectedX = traced.points[2 * slot];
        if (Number.isNaN(expectedX)) {
          expect(Number.isNaN(actual.candidates.points[2 * slot])).toBe(true);
          continue;
        }
        maximumDrift = Math.max(
          maximumDrift,
          Math.abs(actual.candidates.points[2 * slot] - expectedX),
          Math.abs(actual.candidates.points[2 * slot + 1] - traced.points[2 * slot + 1])
        );
      }
    }
    expect(spanMismatches).toBeLessThanOrEqual(seedCount / 100);
    expect(maximumDrift).toBeLessThan(1e-3);
    // Prune: exact.
    const expected = expectPruningMatchesOracle(actual, field, VORTEX_CONFIG, VORTEX_SETTINGS);
    expect(expected.ids.length).toBeGreaterThan(10);
    expect(expected.overflow).toBe(0);
    // Kept spans of accepted lines match the oracle.
    for (const seed of expected.acceptedSeeds) {
      expect(actual.keptSpans[2 * seed]).toBe(expected.trimmed[2 * seed]);
      expect(actual.keptSpans[2 * seed + 1]).toBe(expected.trimmed[2 * seed + 1]);
    }
  } finally {
    fixture.destroy();
  }
});

it('GPUStreamlines spaces straight lines on a uniform field and never shares a grid cell', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const field = createUniformField(16, 16, 1, 0);
  const config: StreamlinesCPUConfig = {
    seedColumns: 16,
    seedRows: 16,
    stepsPerDirection: 64,
    gridWidth: 32,
    gridHeight: 32,
    lineCapacity: 256,
    pointCapacity: 256 * 129
  };
  const settings: GPUStreamlinesSettings = {
    fieldExtent: [0, 0, 1, 1],
    gridExtent: [0, 0, 0.5, 0.5],
    stepLength: 0.25,
    seed: 1,
    minimumPoints: 2
  };
  const fixture = createFixture(device, field, config, 32);
  try {
    const actual = await fixture.run(settings);
    expectPruningMatchesOracle(actual, field, config, settings);
    // The axis-aligned trace is exact, so the full CPU pipeline agrees too.
    const endToEnd = generateStreamlinesOnCPU(
      field,
      config,
      getGPUStreamlinesParameterValues(settings),
      getGPUStreamlinesWordParameterValues(settings)
    );
    expect(actual.ids).toEqual(endToEnd.ids);
    const owner = new Map<number, number>();
    for (let line = 0; line < actual.count; line++) {
      const y = actual.points[2 * actual.pathOffsets[line] + 1];
      for (let point = actual.pathOffsets[line]; point < actual.pathOffsets[line + 1]; point++) {
        // Horizontal lines keep their y.
        expect(actual.points[2 * point + 1]).toBe(y);
        const cell =
          Math.floor(actual.points[2 * point + 1] * 2) * 32 +
          Math.floor(actual.points[2 * point] * 2);
        expect(owner.get(cell) ?? line).toBe(line);
        owner.set(cell, line);
      }
    }
    // At most one line per grid row, and most rows are used.
    expect(actual.count).toBeLessThanOrEqual(32);
    expect(actual.count).toBeGreaterThan(16);
  } finally {
    fixture.destroy();
  }
});

it('GPUStreamlines reports capacity overflow, non-convergence, replays and reseeds without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const field = createVortexField(32, 0.1);
  const smallConfig = {...VORTEX_CONFIG, lineCapacity: 8, pointCapacity: 200};
  const small = createFixture(device, field, smallConfig, 64);
  try {
    const actual = await small.run(VORTEX_SETTINGS);
    expect(actual.overflow).toBe(1);
    expect(actual.count).toBeLessThanOrEqual(8);
    expect(actual.pointCount).toBeLessThanOrEqual(200);
    expect(actual.totalCount).toBeGreaterThan(actual.count);
    expectPruningMatchesOracle(actual, field, smallConfig, VORTEX_SETTINGS);

    // Same graph, new seed: different lines; old seed again: bitwise replay.
    const reseeded = await small.run({...VORTEX_SETTINGS, seed: 6});
    const replay = await small.run(VORTEX_SETTINGS);
    expect(reseeded.ids).not.toEqual(actual.ids);
    expect(replay.ids).toEqual(actual.ids);
    expect(new Uint32Array(replay.points.buffer)).toEqual(new Uint32Array(actual.points.buffer));
  } finally {
    small.destroy();
  }

  const oneRound = createFixture(device, field, VORTEX_CONFIG, 1);
  try {
    const actual = await oneRound.run(VORTEX_SETTINGS);
    expect(actual.unconverged).toBe(1);
    // Lines decided in the only round are a subset of the converged answer.
    const full = generateStreamlinesOnCPU(
      field,
      VORTEX_CONFIG,
      getGPUStreamlinesParameterValues(VORTEX_SETTINGS),
      getGPUStreamlinesWordParameterValues(VORTEX_SETTINGS),
      actual.candidates
    );
    expect(actual.count).toBeGreaterThan(0);
    expect(actual.ids.every(seed => full.acceptedSeeds.includes(seed))).toBe(true);
  } finally {
    oneRound.destroy();
  }
});
