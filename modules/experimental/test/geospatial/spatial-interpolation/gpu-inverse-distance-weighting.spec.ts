// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUInverseDistanceWeightingParameterValues,
  GPUInverseDistanceWeighting,
  type GPUInverseDistanceWeightingSettings
} from '../../../src/geospatial/spatial-interpolation';
import {GPUTerrainContours} from '../../../src/gpu-terrain/terrain-analysis';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {countContourSegments} from '../../gpu-terrain/terrain-analysis/terrain-analysis-oracle';
import {interpolateInverseDistanceWeightingOnCPU} from './spatial-interpolation-oracle';

/**
 * GPU versus float64 oracle tolerance: `|gpu - cpu| <= 2e-5 * valueScale`, where `valueScale` is
 * the largest absolute sample value. Covers f32 log2/exp2 weights and f32 accumulation over up to
 * a few hundred contributors.
 */
const RELATIVE_TOLERANCE = 2e-5;

type Scene = {
  positions: Float32Array;
  values: Float32Array;
  mask?: Uint32Array;
  width: number;
  height: number;
  indexGridSize: [number, number];
  indexBounds: [number, number, number, number];
  maximumNeighborCount: number;
};

type Result = {values: Float32Array; counts: Uint32Array};

type Fixture = {
  run(settings: GPUInverseDistanceWeightingSettings): Promise<Result>;
  /** Graph compilations after the first; per-frame changes must keep it at zero. */
  rebuildCount: number;
  destroy(): void;
};

/** Deterministic xorshift in [0, 1). */
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

function createFixture(device: Device, scene: Scene): Fixture {
  const sampleCount = scene.values.length;
  const cellCount = scene.width * scene.height;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'idw-parameters',
    format: 'float32',
    length: 8
  });
  const valuesOut = track(createOutputBuffer(device, cellCount));
  const countsOut = track(createOutputBuffer(device, cellCount));
  const graph = new GPUCommandGraph(device, {id: 'idw-graph'});
  graph.add(
    new GPUInverseDistanceWeighting({
      id: 'idw',
      positions: importGraphBuffer(
        graph,
        'positions',
        track(createInputBuffer(device, scene.positions)),
        'float32x2',
        sampleCount
      ),
      values: importGraphBuffer(
        graph,
        'values',
        track(createInputBuffer(device, scene.values)),
        'float32',
        sampleCount
      ),
      mask: scene.mask
        ? importGraphBuffer(
            graph,
            'mask',
            track(createInputBuffer(device, scene.mask)),
            'uint32',
            sampleCount
          )
        : undefined,
      parameters: parameterBuffer.importToGraph(graph),
      width: scene.width,
      height: scene.height,
      indexGridSize: scene.indexGridSize,
      indexBounds: scene.indexBounds,
      maximumNeighborCount: scene.maximumNeighborCount,
      output: {
        values: importGraphBuffer(graph, 'values-out', valuesOut, 'float32', cellCount),
        counts: importGraphBuffer(graph, 'counts-out', countsOut, 'uint32', cellCount)
      }
    })
  );
  let compileCount = 0;
  const compiled = graph.compile();
  compileCount++;
  return {
    async run(settings) {
      parameterBuffer.write(getGPUInverseDistanceWeightingParameterValues(settings));
      submitGraph(device, compiled, undefined);
      return {
        values: Float32Array.from(await readFloat32(valuesOut, cellCount)),
        counts: Uint32Array.from(await readUint32(countsOut, cellCount))
      };
    },
    get rebuildCount() {
      return compileCount - 1;
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

function expectParity(
  actual: Result,
  scene: Scene,
  settings: GPUInverseDistanceWeightingSettings
): Result {
  const expected = interpolateInverseDistanceWeightingOnCPU({
    ...scene,
    extent: settings.extent,
    searchRadius: settings.searchRadius,
    power: settings.power ?? 2,
    neighborCount: Math.min(settings.neighborCount ?? 0, scene.maximumNeighborCount),
    minimumNeighborCount: settings.minimumNeighborCount ?? 1
  });
  expect(Array.from(actual.counts)).toEqual(Array.from(expected.counts));
  let valueScale = 1;
  for (const value of scene.values) {
    if (Number.isFinite(value)) {
      valueScale = Math.max(valueScale, Math.abs(value));
    }
  }
  const tolerance = RELATIVE_TOLERANCE * valueScale;
  let worst = 0;
  for (let cell = 0; cell < expected.values.length; cell++) {
    const gpu = actual.values[cell];
    const cpu = expected.values[cell];
    if (Number.isNaN(cpu) || Number.isNaN(gpu)) {
      expect(Number.isNaN(gpu), `cell ${cell} nodata`).toBe(Number.isNaN(cpu));
      continue;
    }
    worst = Math.max(worst, Math.abs(gpu - cpu));
  }
  expect(worst).toBeLessThanOrEqual(tolerance);
  return expected;
}

function createRandomScene(seed: number, sampleCount: number): Scene {
  const random = createRandom(seed);
  const positions = new Float32Array(sampleCount * 2);
  const values = new Float32Array(sampleCount);
  const mask = new Uint32Array(sampleCount);
  for (let row = 0; row < sampleCount; row++) {
    positions[2 * row] = random() * 100;
    positions[2 * row + 1] = random() * 60;
    // A smooth field plus noise so contours exist.
    values[row] =
      Math.sin(positions[2 * row] / 15) * 40 + positions[2 * row + 1] + (random() - 0.5) * 5;
    mask[row] = random() < 0.1 ? 0 : 1;
  }
  // A few NaN values, non-finite positions, and samples outside the index domain are skipped.
  values[3] = NaN;
  positions[2 * 7] = NaN;
  positions[2 * 11] = 150;
  positions[2 * 13 + 1] = -5;
  return {
    positions,
    values,
    mask,
    width: 40,
    height: 24,
    indexGridSize: [16, 10],
    indexBounds: [0, 0, 100, 60],
    maximumNeighborCount: 12
  };
}

it('GPUInverseDistanceWeighting matches the CPU oracle and changes parameters without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createRandomScene(1, 900);
  const fixture = createFixture(device, scene);
  const extent = [0, 0, 100, 60] as const;
  const frames: GPUInverseDistanceWeightingSettings[] = [
    {extent, searchRadius: 10, power: 2},
    {extent, searchRadius: 10, power: 0},
    {extent, searchRadius: 10, power: 1},
    {extent, searchRadius: 10, power: 3.5},
    {extent, searchRadius: 4, power: 2},
    {extent, searchRadius: 25, power: 2, minimumNeighborCount: 3},
    {extent, searchRadius: 20, power: 2, neighborCount: 1},
    {extent, searchRadius: 20, power: 2, neighborCount: 5},
    // k above the compile-time capacity is clamped to 12.
    {extent, searchRadius: 20, power: 2, neighborCount: 40},
    {extent, searchRadius: 3, power: 2, minimumNeighborCount: 4},
    // Panned and zoomed extents, partly outside the index domain.
    {extent: [-20, 10, 60, 50], searchRadius: 8, power: 2},
    {
      extent: [30.5, 20.25, 40.5, 26.25],
      searchRadius: 6,
      power: 2.5,
      neighborCount: 8
    },
    {extent, searchRadius: Infinity, power: 2, neighborCount: 6},
    {extent, searchRadius: 10, power: 2}
  ];
  let first: Result | undefined;
  for (const settings of frames) {
    const actual = await fixture.run(settings);
    const expected = expectParity(actual, scene, settings);
    first ??= actual;
    // Every frame exercises interpolated (non-NaN) cells.
    expect(expected.values.some(value => !Number.isNaN(value))).toBe(true);
  }
  // Same parameters give a bitwise-identical raster (in-cell IDs are re-sorted every encoding).
  const repeat = await fixture.run(frames[0]);
  expect(Array.from(new Uint32Array(repeat.values.buffer))).toEqual(
    Array.from(new Uint32Array(first!.values.buffer))
  );
  // Radius 4 with minimumNeighborCount 4 leaves nodata cells.
  const sparse = await fixture.run(frames[9]);
  expect(Array.from(sparse.values).some(Number.isNaN)).toBe(true);
  expect(fixture.rebuildCount).toBe(0);
  fixture.destroy();
});

it('GPUInverseDistanceWeighting resolves exact hits and nearest-k ties by the smallest row', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Extent [0, 8] x [0, 8] at 8 x 8 puts cell centers on half-integers, exactly representable.
  const positions = Float32Array.from([
    // Rows 0-1: duplicates on center (2.5, 3.5) with different values; row 0 must win.
    2.5, 3.5, 2.5, 3.5,
    // Rows 2-5: four samples at distance 1 around center (5.5, 5.5); row 2 is the tie winner.
    6.5, 5.5, 4.5, 5.5, 5.5, 6.5, 5.5, 4.5,
    // Row 6: masked exact hit on center (0.5, 0.5); row 7: NaN-valued exact hit on (7.5, 7.5).
    0.5, 0.5, 7.5, 7.5
  ]);
  const scene: Scene = {
    positions,
    values: Float32Array.from([11, 99, 1, 2, 3, 4, 50, NaN]),
    mask: Uint32Array.from([1, 1, 1, 1, 1, 1, 0, 1]),
    width: 8,
    height: 8,
    indexGridSize: [4, 4],
    indexBounds: [0, 0, 8, 8],
    maximumNeighborCount: 4
  };
  const fixture = createFixture(device, scene);
  const extent = [0, 0, 8, 8] as const;
  const at = (x: number, y: number) => y * 8 + x;

  let actual = await fixture.run({extent, searchRadius: 1.5, power: 2});
  expectParity(actual, scene, {extent, searchRadius: 1.5, power: 2});
  expect(actual.values[at(2, 3)]).toBe(11);
  expect(actual.counts[at(2, 3)]).toBe(2);
  // Equidistant samples give their mean.
  expect(actual.values[at(5, 5)]).toBeCloseTo(2.5, 5);
  expect(actual.counts[at(5, 5)]).toBe(4);
  // The masked and NaN exact hits are ignored: no other sample is in range.
  expect(Number.isNaN(actual.values[at(0, 0)])).toBe(true);
  expect(Number.isNaN(actual.values[at(7, 7)])).toBe(true);
  expect(actual.counts[at(0, 0)]).toBe(0);

  // k = 1 among four equal distances keeps row 2 (value 1).
  const nearestOne = {extent, searchRadius: 1.5, power: 2, neighborCount: 1};
  actual = await fixture.run(nearestOne);
  expectParity(actual, scene, nearestOne);
  expect(actual.values[at(5, 5)]).toBe(1);
  expect(actual.counts[at(5, 5)]).toBe(1);
  // k = 2 keeps rows 2 and 3.
  const nearestTwo = {extent, searchRadius: 1.5, power: 2, neighborCount: 2};
  actual = await fixture.run(nearestTwo);
  expect(actual.values[at(5, 5)]).toBeCloseTo(1.5, 5);
  // An exact hit wins regardless of minimumNeighborCount.
  actual = await fixture.run({
    extent,
    searchRadius: 0.5,
    power: 2,
    minimumNeighborCount: 5
  });
  expect(actual.values[at(2, 3)]).toBe(11);
  expect(Number.isNaN(actual.values[at(5, 5)])).toBe(true);
  // Invalid radius or power gives an all-nodata raster with zero counts.
  for (const invalid of [
    {extent, searchRadius: -1, power: 2},
    {extent, searchRadius: NaN, power: 2},
    {extent, searchRadius: 2, power: -1}
  ]) {
    actual = await fixture.run(invalid);
    expect(Array.from(actual.values).every(Number.isNaN)).toBe(true);
    expect(Array.from(actual.counts).every(count => count === 0)).toBe(true);
  }
  expect(fixture.rebuildCount).toBe(0);
  fixture.destroy();
});

it('GPUInverseDistanceWeighting stays finite for tiny distances and large powers', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // Samples about 1.2e-7 and 1e-3 from the single cell center; d^-12 overflows f32 without the
  // log-space accumulator.
  const scene: Scene = {
    positions: Float32Array.from([0.5 + 1e-7, 0.5, 0.501, 0.5]),
    values: Float32Array.from([10, 20]),
    width: 1,
    height: 1,
    indexGridSize: [1, 1],
    indexBounds: [0, 0, 1, 1],
    maximumNeighborCount: 0
  };
  const fixture = createFixture(device, scene);
  const settings = {
    extent: [0, 0, 1, 1] as const,
    searchRadius: 1,
    power: 12
  };
  const actual = await fixture.run(settings);
  expectParity(actual, scene, settings);
  expect(actual.values[0]).toBeCloseTo(10, 4);
  fixture.destroy();
});

it('GPUInverseDistanceWeighting output feeds GPUTerrainContours in the same graph', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const scene = createRandomScene(4, 600);
  const {width, height} = scene;
  const cellCount = width * height;
  const sampleCount = scene.values.length;
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'idw-contour-parameters',
    format: 'float32',
    length: 8,
    values: getGPUInverseDistanceWeightingParameterValues({
      extent: [0, 0, 100, 60],
      searchRadius: 4,
      power: 2,
      minimumNeighborCount: 2
    })
  });
  const buffers = {
    positions: createInputBuffer(device, scene.positions),
    values: createInputBuffer(device, scene.values),
    surface: createOutputBuffer(device, cellCount),
    vertices: createOutputBuffer(device, 2 * 2 * 4096),
    segmentCount: createOutputBuffer(device, 1),
    overflow: createOutputBuffer(device, 1)
  };
  const graph = new GPUCommandGraph(device, {id: 'idw-contours'});
  const surface = importGraphBuffer(graph, 'surface', buffers.surface, 'float32', cellCount);
  graph.add(
    new GPUInverseDistanceWeighting({
      id: 'idw',
      positions: importGraphBuffer(graph, 'positions', buffers.positions, 'float32x2', sampleCount),
      values: importGraphBuffer(graph, 'values', buffers.values, 'float32', sampleCount),
      parameters: parameterBuffer.importToGraph(graph),
      width,
      height,
      indexGridSize: scene.indexGridSize,
      indexBounds: scene.indexBounds,
      output: {values: surface}
    })
  );
  const level = 30;
  graph.add(
    new GPUTerrainContours({
      width,
      height,
      elevation: {
        id: 'surface',
        format: 'float32',
        storage: {kind: 'buffer', values: surface}
      },
      levels: [
        {
          level,
          vertices: importGraphBuffer(graph, 'vertices', buffers.vertices, 'float32x2', 2 * 4096),
          segmentCount: importGraphBuffer(graph, 'segment-count', buffers.segmentCount, 'uint32', 1)
        }
      ],
      overflow: importGraphBuffer(graph, 'overflow', buffers.overflow, 'uint32', 1)
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const raster = await readFloat32(buffers.surface, cellCount);
  const [segmentCount] = await readUint32(buffers.segmentCount, 1);
  const [overflow] = await readUint32(buffers.overflow, 1);
  const valid = raster.map(value => (Number.isNaN(value) ? 0 : 1));
  expect(valid.some(flag => flag === 0)).toBe(true);
  expect(segmentCount).toBeGreaterThan(10);
  expect(segmentCount).toBe(countContourSegments(raster, width, height, level, valid));
  expect(overflow).toBe(0);
  compiled.destroy();
  parameterBuffer.destroy();
  for (const buffer of Object.values(buffers)) {
    buffer.destroy();
  }
});
