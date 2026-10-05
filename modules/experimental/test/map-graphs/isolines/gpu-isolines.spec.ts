// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUMapGraphParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {GPUIsolines} from '../../../src/map-graphs/isolines/gpu-isolines';
import {getGPUIsolinesParameterValues} from '../../../src/map-graphs/isolines/isolines-parameters';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../map-graph-test-utils';
import {
  computeIsolinesOnCPU,
  getPolylineVertices,
  stitchIsolinesOnCPU,
  type IsolinesScene
} from './isolines-oracle';

type HarnessOptions = {
  width: number;
  height: number;
  maximumLevelCount: number;
  segmentCapacity: number;
  vertexCapacity?: number;
  polylines?: boolean;
  noDataValue?: number;
  withValidity?: boolean;
};

/** Builds one compiled graph that is reused across frames. */
function createHarness(device: Device, options: HarnessOptions) {
  const {width, height, maximumLevelCount, segmentCapacity: capacity} = options;
  const vertexCapacity = options.vertexCapacity ?? 2 * capacity;
  const rasterBuffer = createInputBuffer(device, new Float32Array(width * height));
  const validityBuffer = options.withValidity
    ? createInputBuffer(device, new Uint32Array(width * height).fill(1))
    : undefined;
  const levelsBuffer = createInputBuffer(device, new Float32Array(maximumLevelCount));
  const parameterBuffer = new GPUMapGraphParameterBuffer(device, {
    id: 'isolines-parameters',
    format: 'float32',
    length: 8
  });
  const buffers = {
    segments: createOutputBuffer(device, 4 * capacity),
    segmentLevels: createOutputBuffer(device, capacity),
    segmentEdges: createOutputBuffer(device, 2 * capacity),
    count: createOutputBuffer(device, 1),
    overflow: createOutputBuffer(device, 1),
    totalCount: createOutputBuffer(device, 1),
    vertices: createOutputBuffer(device, 2 * vertexCapacity),
    polylineOffsets: createOutputBuffer(device, capacity + 1),
    polylineLevels: createOutputBuffer(device, capacity),
    polylineClosed: createOutputBuffer(device, capacity),
    polylineCount: createOutputBuffer(device, 1),
    vertexCount: createOutputBuffer(device, 1),
    polylineOverflow: createOutputBuffer(device, 1)
  };
  const graph = new GPUCommandGraph(device, {id: 'isolines-graph'});
  const importView = <Format extends 'float32' | 'uint32' | 'float32x2' | 'float32x4' | 'uint32x2'>(
    name: string,
    buffer: ReturnType<typeof createOutputBuffer>,
    format: Format,
    length: number
  ) => importGraphBuffer(graph, name, buffer, format, length);
  graph.add(
    new GPUIsolines({
      width,
      height,
      values: importView('raster', rasterBuffer, 'float32', width * height),
      validity: validityBuffer
        ? importView('validity', validityBuffer, 'uint32', width * height)
        : undefined,
      noDataValue: options.noDataValue,
      levels: importView('levels', levelsBuffer, 'float32', maximumLevelCount),
      parameters: parameterBuffer.importToGraph(graph),
      output: {
        segments: importView('segments', buffers.segments, 'float32x4', capacity),
        segmentLevels: importView('segment-levels', buffers.segmentLevels, 'uint32', capacity),
        segmentEdges: importView('segment-edges', buffers.segmentEdges, 'uint32x2', capacity),
        count: importView('count', buffers.count, 'uint32', 1),
        overflow: importView('overflow', buffers.overflow, 'uint32', 1),
        totalCount: importView('total-count', buffers.totalCount, 'uint32', 1)
      },
      polylines: options.polylines
        ? {
            vertices: importView('vertices', buffers.vertices, 'float32x2', vertexCapacity),
            polylineOffsets: importView(
              'polyline-offsets',
              buffers.polylineOffsets,
              'uint32',
              capacity + 1
            ),
            polylineLevels: importView(
              'polyline-levels',
              buffers.polylineLevels,
              'uint32',
              capacity
            ),
            polylineClosed: importView(
              'polyline-closed',
              buffers.polylineClosed,
              'uint32',
              capacity
            ),
            polylineCount: importView('polyline-count', buffers.polylineCount, 'uint32', 1),
            vertexCount: importView('vertex-count', buffers.vertexCount, 'uint32', 1),
            overflow: importView('polyline-overflow', buffers.polylineOverflow, 'uint32', 1)
          }
        : undefined
    })
  );
  const compiled = graph.compile();
  let compileCount = 1;
  return {
    get rebuildCount() {
      return compileCount - 1;
    },
    async run(scene: IsolinesScene, levelCount: number = scene.levels.length) {
      rasterBuffer.write(scene.values);
      if (validityBuffer && scene.validity) {
        validityBuffer.write(scene.validity);
      }
      const levels = new Float32Array(maximumLevelCount);
      levels.set(scene.levels.slice(0, maximumLevelCount));
      levelsBuffer.write(levels);
      parameterBuffer.write(
        getGPUIsolinesParameterValues({width, height, levelCount, extent: scene.extent})
      );
      submitGraph(device, compiled, undefined);
      const [count] = await readUint32(buffers.count, 1);
      const [overflow] = await readUint32(buffers.overflow, 1);
      const [totalCount] = await readUint32(buffers.totalCount, 1);
      const floats = await readFloat32(buffers.segments, 4 * count);
      const segmentLevels = await readUint32(buffers.segmentLevels, count);
      const edgeWords = await readUint32(buffers.segmentEdges, 2 * count);
      const result = {
        count,
        overflow,
        totalCount,
        segments: Array.from({length: count}, (_, index) => ({
          p0: [floats[4 * index], floats[4 * index + 1]] as [number, number],
          p1: [floats[4 * index + 2], floats[4 * index + 3]] as [number, number],
          level: segmentLevels[index],
          startEdge: edgeWords[2 * index],
          endEdge: edgeWords[2 * index + 1]
        })),
        polylineCount: 0,
        vertexCount: 0,
        polylineOverflow: 0,
        offsets: [] as number[],
        polylineLevels: [] as number[],
        polylineClosed: [] as number[],
        vertices: [] as number[]
      };
      if (options.polylines) {
        result.polylineCount = (await readUint32(buffers.polylineCount, 1))[0];
        result.vertexCount = (await readUint32(buffers.vertexCount, 1))[0];
        result.polylineOverflow = (await readUint32(buffers.polylineOverflow, 1))[0];
        const polylineCount = result.polylineCount;
        result.offsets = await readUint32(buffers.polylineOffsets, polylineCount + 1);
        result.polylineLevels = await readUint32(buffers.polylineLevels, polylineCount);
        result.polylineClosed = await readUint32(buffers.polylineClosed, polylineCount);
        result.vertices = await readFloat32(buffers.vertices, 2 * result.vertexCount);
      }
      return result;
    }
  };
}

/** Coordinates may differ by FMA contraction and division rounding: a few ULP of the extent. */
function getCoordinateTolerance(scene: IsolinesScene): number {
  const magnitude = Math.max(...scene.extent.map(Math.abs));
  return 8 * magnitude * 2 ** -23;
}

function expectSegmentsMatch(
  actual: ReturnType<typeof createHarness> extends {run: (...args: never[]) => Promise<infer R>}
    ? R
    : never,
  scene: IsolinesScene,
  levelCount: number,
  label: string
) {
  const expected = computeIsolinesOnCPU(scene, levelCount);
  const tolerance = getCoordinateTolerance(scene);
  expect(actual.totalCount, `${label} total`).toBe(expected.length);
  const comparable = Math.min(expected.length, actual.segments.length);
  expect(actual.count, `${label} count`).toBe(comparable);
  for (let index = 0; index < comparable; index++) {
    const gpu = actual.segments[index];
    const cpu = expected[index];
    expect(gpu.level, `${label} level ${index}`).toBe(cpu.level);
    expect([gpu.startEdge, gpu.endEdge], `${label} edges ${index}`).toEqual([
      cpu.startEdge,
      cpu.endEdge
    ]);
    for (const [name, left, right] of [
      ['p0', gpu.p0, cpu.p0],
      ['p1', gpu.p1, cpu.p1]
    ] as const) {
      expect(Math.abs(left[0] - right[0]), `${label} ${name}.x ${index}`).toBeLessThanOrEqual(
        tolerance
      );
      expect(Math.abs(left[1] - right[1]), `${label} ${name}.y ${index}`).toBeLessThanOrEqual(
        tolerance
      );
    }
  }
  return expected;
}

/** Polylines must match the oracle structure exactly and reuse GPU segment endpoints bitwise. */
function expectPolylinesMatch(
  actual: Awaited<ReturnType<ReturnType<typeof createHarness>['run']>>,
  expectedSegments: ReturnType<typeof computeIsolinesOnCPU>,
  label: string
) {
  const {polylines} = stitchIsolinesOnCPU(expectedSegments);
  expect(actual.polylineCount, `${label} polyline count`).toBe(polylines.length);
  expect(actual.polylineClosed, `${label} closed`).toEqual(
    polylines.map(polyline => (polyline.closed ? 1 : 0))
  );
  expect(actual.polylineLevels, `${label} levels`).toEqual(polylines.map(p => p.level));
  const flattened = getPolylineVertices(polylines, actual.segments);
  expect(actual.offsets, `${label} offsets`).toEqual(flattened.offsets);
  expect(actual.vertexCount).toBe(flattened.vertices.length / 2);
  const toBits = (values: readonly number[]) =>
    Array.from(new Uint32Array(Float32Array.from(values).buffer));
  expect(toBits(actual.vertices), `${label} vertices bit-for-bit`).toEqual(
    toBits(flattened.vertices)
  );
  expect(actual.polylineOverflow).toBe(0);
  return polylines;
}

function createRandom(seed: number): () => number {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 4294967296;
  };
}

function createRandomScene(seed: number, width: number, height: number): IsolinesScene {
  const random = createRandom(seed);
  const values = new Float32Array(width * height);
  for (let index = 0; index < values.length; index++) {
    const roll = random();
    values[index] = roll < 0.04 ? NaN : roll < 0.7 ? Math.floor(random() * 10) : random() * 10;
  }
  return {
    width,
    height,
    values,
    levels: [2, 4.5, 5, 7.25],
    extent: [100, -50, 100 + width * 2, -50 + height * 3]
  };
}

it('GPUIsolines matches the CPU oracle on random fields as levels and counts change', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 37;
  const height = 23;
  const harness = createHarness(device, {
    width,
    height,
    maximumLevelCount: 6,
    segmentCapacity: 4000,
    polylines: true
  });
  const scene = createRandomScene(11, width, height);
  // Level 2 and 5 equal many samples; 4.5 and 7.25 do not.
  const frames: {levels: number[]; levelCount: number}[] = [
    {levels: [2, 4.5, 5, 7.25, 0, 0], levelCount: 4},
    {levels: [5, 1, 9.5, 3, 0, 0], levelCount: 1},
    {levels: [2, 4.5, 5, 7.25, 3, 6], levelCount: 6},
    {levels: [2, 4.5, 5, 7.25, 3, 6], levelCount: 0},
    {levels: [3, 3, 8, 0, 0, 0], levelCount: 3},
    {levels: [2, 4.5, 5, 7.25, 3, 6], levelCount: 99}
  ];
  for (const [index, frame] of frames.entries()) {
    const frameScene = {...scene, levels: frame.levels};
    const levelCount = Math.min(frame.levelCount, 6);
    const actual = await harness.run(frameScene, frame.levelCount);
    const expected = expectSegmentsMatch(actual, frameScene, levelCount, `frame ${index}`);
    expect(actual.overflow).toBe(0);
    if (expected.length > 0) {
      expectPolylinesMatch(actual, expected, `frame ${index}`);
    } else {
      expect(actual.polylineCount).toBe(0);
    }
  }
  expect(harness.rebuildCount).toBe(0);
});

it('GPUIsolines resolves both saddle types and keeps a plateau empty', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const harness = createHarness(device, {
    width: 4,
    height: 3,
    maximumLevelCount: 2,
    segmentCapacity: 64,
    polylines: true
  });
  // Alternating checkerboard cells: every cell is a saddle; the corner value sets centre vs level.
  const make = (high: number, low: number): IsolinesScene => ({
    width: 4,
    height: 3,
    values: Float32Array.from(
      Array.from({length: 12}, (_, index) =>
        ((index % 4) + Math.floor(index / 4)) % 2 === 0 ? high : low
      )
    ),
    levels: [1],
    extent: [0, 0, 4, 3]
  });
  for (const [high, low, label] of [
    [3, 0, 'centre above'],
    [2, -1, 'centre below']
  ] as const) {
    const scene = make(high, low);
    const actual = await harness.run(scene);
    const expected = expectSegmentsMatch(actual, scene, 1, label);
    expect(expected).toHaveLength(12);
    expectPolylinesMatch(actual, expected, label);
  }
  // Plateau: constant field equal to the level emits nothing.
  const plateau = {...make(1, 1), levels: [1]};
  const empty = await harness.run(plateau);
  expect(empty.count).toBe(0);
  expect(empty.totalCount).toBe(0);
  expect(empty.polylineCount).toBe(0);
  expect(empty.overflow).toBe(0);
  expect(empty.polylineOverflow).toBe(0);
});

it('GPUIsolines closes a ring around a peak and chains an open ramp contour', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 21;
  const height = 21;
  const harness = createHarness(device, {
    width,
    height,
    maximumLevelCount: 2,
    segmentCapacity: 500,
    polylines: true
  });
  const peak = new Float32Array(width * height);
  const ramp = new Float32Array(width * height);
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      peak[row * width + column] = 10 - Math.hypot(column - 10, row - 10);
      ramp[row * width + column] = column;
    }
  }
  const extent = [0, 0, 21, 21] as const;
  const ringScene: IsolinesScene = {width, height, values: peak, levels: [5.3], extent};
  const ring = await harness.run(ringScene);
  const ringExpected = expectSegmentsMatch(ring, ringScene, 1, 'ring');
  expectPolylinesMatch(ring, ringExpected, 'ring');
  expect(ring.polylineCount).toBe(1);
  expect(ring.polylineClosed).toEqual([1]);
  expect(ring.vertexCount).toBe(ring.count + 1);
  expect(ring.vertices.slice(0, 2)).toEqual(ring.vertices.slice(-2));

  const rampScene: IsolinesScene = {width, height, values: ramp, levels: [3.5, 9], extent};
  const open = await harness.run(rampScene);
  const rampExpected = expectSegmentsMatch(open, rampScene, 2, 'ramp');
  expectPolylinesMatch(open, rampExpected, 'ramp');
  expect(open.polylineCount).toBe(2);
  expect(open.polylineClosed).toEqual([0, 0]);
  expect(open.polylineLevels).toEqual([0, 1]);
  // Each chain runs the full height, touching both borders.
  expect(open.offsets).toEqual([0, height, 2 * height]);
});

it('GPUIsolines honours nodata sentinels, validity and NaN holes', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 19;
  const height = 15;
  const harness = createHarness(device, {
    width,
    height,
    maximumLevelCount: 3,
    segmentCapacity: 1500,
    polylines: true,
    noDataValue: -9999,
    withValidity: true
  });
  const scene = createRandomScene(5, width, height);
  const random = createRandom(99);
  const validity = new Uint32Array(width * height);
  for (let index = 0; index < validity.length; index++) {
    validity[index] = random() < 0.05 ? 0 : 1;
    if (random() < 0.04) {
      scene.values[index] = -9999;
    }
  }
  const sentinelScene = {...scene, validity, noDataValue: -9999, levels: [3, 5, 7]};
  const actual = await harness.run(sentinelScene);
  const expected = expectSegmentsMatch(actual, sentinelScene, 3, 'nodata');
  expect(expected.length).toBeGreaterThan(0);
  expectPolylinesMatch(actual, expected, 'nodata');
});

it('GPUIsolines clamps to capacity, reports overflow, and drops polylines on overflow', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 31;
  const height = 21;
  const scene = createRandomScene(21, width, height);
  const full = computeIsolinesOnCPU(scene, 4);
  expect(full.length).toBeGreaterThan(40);
  const small = createHarness(device, {
    width,
    height,
    maximumLevelCount: 4,
    segmentCapacity: 37,
    polylines: true
  });
  const actual = await small.run(scene);
  expectSegmentsMatch(actual, scene, 4, 'overflow');
  expect(actual.count).toBe(37);
  expect(actual.totalCount).toBe(full.length);
  expect(actual.overflow).toBe(1);
  expect(actual.polylineCount).toBe(0);
  expect(actual.vertexCount).toBe(0);
  expect(actual.polylineOverflow).toBe(1);

  // Exactly at capacity is not an overflow.
  const exact = createHarness(device, {
    width,
    height,
    maximumLevelCount: 4,
    segmentCapacity: full.length,
    polylines: true
  });
  const fits = await exact.run(scene);
  expect(fits.overflow).toBe(0);
  expect(fits.count).toBe(full.length);
  expectPolylinesMatch(fits, full.length ? computeIsolinesOnCPU(scene, 4) : [], 'exact');

  // Vertex capacity below the polyline needs: complete polylines only, overflow set.
  const tight = createHarness(device, {
    width,
    height,
    maximumLevelCount: 4,
    segmentCapacity: full.length,
    vertexCapacity: Math.floor(full.length / 2),
    polylines: true
  });
  const clipped = await tight.run(scene);
  const {polylines} = stitchIsolinesOnCPU(full);
  const flattened = getPolylineVertices(polylines, clipped.segments);
  let keep = 0;
  while (keep < polylines.length && flattened.offsets[keep + 1] <= Math.floor(full.length / 2)) {
    keep++;
  }
  expect(keep).toBeLessThan(polylines.length);
  expect(clipped.polylineCount).toBe(keep);
  expect(clipped.vertexCount).toBe(flattened.offsets[keep]);
  expect(clipped.polylineOverflow).toBe(1);
  expect(clipped.offsets).toEqual(flattened.offsets.slice(0, keep + 1));
});

it('GPUIsolines stitches a large smooth field with many long chains and rings', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 129;
  const height = 97;
  const values = new Float32Array(width * height);
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      values[row * width + column] =
        5 +
        3 * Math.sin(column * 0.21) * Math.cos(row * 0.17) +
        2 * Math.sin((column + row) * 0.05);
    }
  }
  values[40 * width + 50] = Number.NaN;
  const scene: IsolinesScene = {
    width,
    height,
    values,
    levels: [2.5, 4, 5, 6.5],
    extent: [-1000, -500, 1000, 500]
  };
  const harness = createHarness(device, {
    width,
    height,
    maximumLevelCount: 4,
    segmentCapacity: 8192,
    polylines: true
  });
  const actual = await harness.run(scene);
  const expected = expectSegmentsMatch(actual, scene, 4, 'large');
  const polylines = expectPolylinesMatch(actual, expected, 'large');
  expect(polylines.some(polyline => polyline.closed)).toBe(true);
  expect(polylines.some(polyline => !polyline.closed)).toBe(true);
  expect(Math.max(...polylines.map(polyline => polyline.segmentIndices.length))).toBeGreaterThan(
    100
  );
});
