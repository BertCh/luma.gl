// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {
  generateDotsOnCPU,
  getGPUDotDensityParameterValues,
  GPUDotDensity,
  GPURandomPointsInPolygon,
  type DotDensityCPUResult,
  type GPUDotDensitySettings
} from '../../../src/geospatial/dot-density';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {createPolygonColumns, isInSceneHole, SCENE_FEATURES} from './dot-density-scene';

/** Candidate coordinates are below 50; a fused multiply-add changes them by at most one f32 ulp. */
const POSITION_TOLERANCE = 1e-5;

type Mask = {weights: Float32Array; width: number; height: number};

type GPUResult = DotDensityCPUResult & {failedCount: number};

type Fixture = {
  run(settings: GPUDotDensitySettings): Promise<GPUResult>;
  destroy(): void;
};

type FixtureProps = {
  values?: Float32Array;
  counts?: Uint32Array;
  categoryCount?: number;
  capacity: number;
  maximumAttempts?: number;
  mask?: Mask;
};

function createFixture(device: Device, props: FixtureProps): Fixture {
  const polygons = createPolygonColumns(SCENE_FEATURES);
  const featureCount = polygons.featureOffsets.length - 1;
  const categoryCount = props.categoryCount ?? 1;
  const slotCount = featureCount * categoryCount;
  const {capacity} = props;
  const graph = new GPUCommandGraph(device, {id: 'dot-density-graph'});
  const buffers: Buffer[] = [];
  const input = (values: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return buffer;
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'dot-parameters',
    format: 'uint32',
    length: 8
  });
  const outputs = {
    positions: output(capacity * 2),
    ids: output(capacity),
    count: output(1),
    overflow: output(1),
    total: output(1),
    categories: output(capacity),
    failed: output(1),
    slotCounts: output(slotCount),
    slotOffsets: output(slotCount)
  };
  const polygonViews = {
    polygonPositions: importGraphBuffer(
      graph,
      'polygon-positions',
      input(polygons.polygonPositions),
      'float32x2',
      polygons.polygonPositions.length / 2
    ),
    featureOffsets: importGraphBuffer(
      graph,
      'feature-offsets',
      input(polygons.featureOffsets),
      'uint32',
      polygons.featureOffsets.length
    ),
    polygonOffsets: importGraphBuffer(
      graph,
      'polygon-offsets',
      input(polygons.polygonOffsets),
      'uint32',
      polygons.polygonOffsets.length
    ),
    ringOffsets: importGraphBuffer(
      graph,
      'ring-offsets',
      input(polygons.ringOffsets),
      'uint32',
      polygons.ringOffsets.length
    )
  };
  const mask = props.mask
    ? {
        weights: importGraphBuffer(
          graph,
          'mask',
          input(props.mask.weights),
          'float32',
          props.mask.weights.length
        ),
        width: props.mask.width,
        height: props.mask.height
      }
    : undefined;
  const sharedOutput = {
    positions: importGraphBuffer(graph, 'o-positions', outputs.positions, 'float32x2', capacity),
    dots: {
      ids: importGraphBuffer(graph, 'o-ids', outputs.ids, 'uint32', capacity),
      count: importGraphBuffer(graph, 'o-count', outputs.count, 'uint32', 1),
      overflow: importGraphBuffer(graph, 'o-overflow', outputs.overflow, 'uint32', 1),
      totalCount: importGraphBuffer(graph, 'o-total', outputs.total, 'uint32', 1)
    },
    failedCount: importGraphBuffer(graph, 'o-failed', outputs.failed, 'uint32', 1),
    slotCounts: importGraphBuffer(graph, 'o-slot-counts', outputs.slotCounts, 'uint32', slotCount),
    slotOffsets: importGraphBuffer(
      graph,
      'o-slot-offsets',
      outputs.slotOffsets,
      'uint32',
      slotCount
    )
  };
  const parameters = parameterBuffer.importToGraph(graph);
  if (props.values) {
    graph.add(
      new GPUDotDensity({
        id: 'dots',
        ...polygonViews,
        values: importGraphBuffer(graph, 'values', input(props.values), 'float32', slotCount),
        categoryCount,
        parameters,
        maximumAttempts: props.maximumAttempts,
        mask,
        output: {
          ...sharedOutput,
          categories: importGraphBuffer(
            graph,
            'o-categories',
            outputs.categories,
            'uint32',
            capacity
          )
        }
      })
    );
  } else {
    graph.add(
      new GPURandomPointsInPolygon({
        id: 'points',
        ...polygonViews,
        counts: importGraphBuffer(graph, 'counts', input(props.counts!), 'uint32', featureCount),
        parameters,
        maximumAttempts: props.maximumAttempts,
        mask,
        output: sharedOutput
      })
    );
  }
  const compiled = graph.compile();
  return {
    async run(settings) {
      parameterBuffer.write(getGPUDotDensityParameterValues(settings));
      submitGraph(device, compiled, undefined);
      const [count] = await readUint32(outputs.count, 1);
      return {
        count,
        overflow: (await readUint32(outputs.overflow, 1))[0],
        totalCount: (await readUint32(outputs.total, 1))[0],
        failedCount: (await readUint32(outputs.failed, 1))[0],
        slotCounts: Uint32Array.from(await readUint32(outputs.slotCounts, slotCount)),
        slotOffsets: Uint32Array.from(await readUint32(outputs.slotOffsets, slotCount)),
        positions: Float32Array.from(await readFloat32(outputs.positions, capacity * 2)),
        featureIds: Uint32Array.from(await readUint32(outputs.ids, capacity)),
        categories: Uint32Array.from(await readUint32(outputs.categories, capacity))
      };
    },
    destroy() {
      compiled.destroy();
      parameterBuffer.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

function expectMatchesOracle(
  actual: GPUResult,
  expected: DotDensityCPUResult,
  checkCategories: boolean
): void {
  expect(Array.from(actual.slotCounts)).toEqual(Array.from(expected.slotCounts));
  expect(Array.from(actual.slotOffsets)).toEqual(Array.from(expected.slotOffsets));
  expect(actual.totalCount).toBe(expected.totalCount);
  expect(actual.count).toBe(expected.count);
  expect(actual.overflow).toBe(expected.overflow);
  expect(actual.failedCount).toBe(expected.failedCount);
  expect(Array.from(actual.featureIds.slice(0, actual.count))).toEqual(
    Array.from(expected.featureIds)
  );
  if (checkCategories) {
    expect(Array.from(actual.categories.slice(0, actual.count))).toEqual(
      Array.from(expected.categories)
    );
  }
  for (let index = 0; index < expected.count * 2; index++) {
    const value = actual.positions[index];
    const reference = expected.positions[index];
    if (Number.isNaN(reference)) {
      expect(Number.isNaN(value)).toBe(true);
    } else {
      expect(Math.abs(value - reference)).toBeLessThan(POSITION_TOLERANCE);
    }
  }
  // Rows past the count are cleared.
  for (let dot = actual.count; dot < actual.featureIds.length; dot++) {
    expect(actual.featureIds[dot]).toBe(0xffffffff);
    expect(Number.isNaN(actual.positions[2 * dot])).toBe(true);
  }
}

const VALUES = new Float32Array([370.3, 120.6, 80.2, 45.5, 0, Number.NaN]);

it('GPUDotDensity matches the CPU oracle with two categories', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const polygons = createPolygonColumns(SCENE_FEATURES);
  const fixture = createFixture(device, {
    values: VALUES,
    categoryCount: 2,
    capacity: 2048
  });
  try {
    for (const settings of [
      {seed: 5, dotsPerUnit: 1},
      {seed: 6, dotsPerUnit: 2.5}
    ]) {
      const actual = await fixture.run(settings);
      const expected = generateDotsOnCPU({
        ...polygons,
        values: VALUES,
        categoryCount: 2,
        parameters: getGPUDotDensityParameterValues(settings),
        capacity: 2048,
        maximumAttempts: 32
      });
      expectMatchesOracle(actual, expected, true);
      expect(actual.count).toBeGreaterThan(500);
      for (let dot = 0; dot < actual.count; dot++) {
        expect(isInSceneHole(actual.positions[2 * dot], actual.positions[2 * dot + 1])).toBe(false);
      }
    }
  } finally {
    fixture.destroy();
  }
});

it('GPUDotDensity keeps dot positions stable when the dot value changes', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const fixture = createFixture(device, {
    values: VALUES,
    categoryCount: 2,
    capacity: 4096
  });
  try {
    const coarse = await fixture.run({seed: 11, dotsPerUnit: 0.5});
    const fine = await fixture.run({seed: 11, dotsPerUnit: 3});
    let comparedDots = 0;
    for (let slot = 0; slot < 6; slot++) {
      expect(fine.slotCounts[slot]).toBeGreaterThanOrEqual(coarse.slotCounts[slot]);
      for (let rank = 0; rank < coarse.slotCounts[slot]; rank++) {
        const coarseDot = coarse.slotOffsets[slot] + rank;
        const fineDot = fine.slotOffsets[slot] + rank;
        // Bitwise equal: same Philox candidates on the same GPU.
        expect(fine.positions[2 * fineDot]).toBe(coarse.positions[2 * coarseDot]);
        expect(fine.positions[2 * fineDot + 1]).toBe(coarse.positions[2 * coarseDot + 1]);
        comparedDots++;
      }
    }
    expect(comparedDots).toBeGreaterThan(200);
    // Replay of the first frame on the same compiled graph is bitwise identical.
    const replay = await fixture.run({seed: 11, dotsPerUnit: 0.5});
    expect(new Uint32Array(replay.positions.buffer)).toEqual(
      new Uint32Array(coarse.positions.buffer)
    );
  } finally {
    fixture.destroy();
  }
});

it('GPUDotDensity clamps to capacity, applies a mask and counts failures', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const polygons = createPolygonColumns(SCENE_FEATURES);
  const small = createFixture(device, {
    values: VALUES,
    categoryCount: 2,
    capacity: 100
  });
  try {
    const settings = {seed: 2, dotsPerUnit: 1};
    const actual = await small.run(settings);
    const expected = generateDotsOnCPU({
      ...polygons,
      values: VALUES,
      categoryCount: 2,
      parameters: getGPUDotDensityParameterValues(settings),
      capacity: 100,
      maximumAttempts: 32
    });
    expect(actual.count).toBe(100);
    expect(actual.overflow).toBe(1);
    expect(actual.totalCount).toBeGreaterThan(100);
    expectMatchesOracle(actual, expected, true);
  } finally {
    small.destroy();
  }

  const weights = new Float32Array(100);
  for (let row = 0; row < 10; row++) {
    weights.fill(0.75, row * 10 + 5, row * 10 + 10);
  }
  const mask = {weights, width: 10, height: 10};
  const masked = createFixture(device, {
    values: new Float32Array([300, 50, 40]),
    capacity: 1024,
    maximumAttempts: 8,
    mask
  });
  try {
    const settings = {
      seed: 4,
      dotsPerUnit: 1,
      maskExtent: [0, 0, 1, 1] as const
    };
    const actual = await masked.run(settings);
    const expected = generateDotsOnCPU({
      ...polygons,
      values: new Float32Array([300, 50, 40]),
      parameters: getGPUDotDensityParameterValues(settings),
      capacity: 1024,
      maximumAttempts: 8,
      mask
    });
    expectMatchesOracle(actual, expected, false);
    // Features 1 and 2 lie outside the mask: every one of their dots fails.
    expect(actual.failedCount).toBeGreaterThanOrEqual(90);
    for (let dot = 0; dot < actual.count; dot++) {
      const x = actual.positions[2 * dot];
      if (!Number.isNaN(x)) {
        expect(actual.featureIds[dot]).toBe(0);
        expect(x).toBeGreaterThanOrEqual(5);
      }
    }
  } finally {
    masked.destroy();
  }
});

it('GPURandomPointsInPolygon matches the oracle, excludes holes and reports sliver failures', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const polygons = createPolygonColumns(SCENE_FEATURES);
  const counts = new Uint32Array([500, 300, 64]);
  const fixture = createFixture(device, {
    counts,
    capacity: 1024,
    maximumAttempts: 3
  });
  try {
    const settings = {seed: 77};
    const actual = await fixture.run(settings);
    const expected = generateDotsOnCPU({
      ...polygons,
      counts,
      parameters: getGPUDotDensityParameterValues(settings),
      capacity: 1024,
      maximumAttempts: 3
    });
    expectMatchesOracle(actual, expected, false);
    expect(actual.count).toBe(864);
    // The sliver covers 0.1% of its box, so 3 attempts almost always fail.
    expect(actual.failedCount).toBeGreaterThan(55);
    for (let dot = 0; dot < 500; dot++) {
      const x = actual.positions[2 * dot];
      const y = actual.positions[2 * dot + 1];
      if (!Number.isNaN(x)) {
        expect(isInSceneHole(x, y)).toBe(false);
        expect(x >= 0 && x <= 10 && y >= 0 && y <= 10).toBe(true);
      }
    }
  } finally {
    fixture.destroy();
  }
});
