// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPURasterProfileParameterValues,
  GPURasterProfile,
  GPU_RASTER_PROFILE_NO_PATH_ID,
  type GPURasterProfileSettings
} from '../../../src/gpu-raster/raster-sampling';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createRandom, expectFloatArraysClose} from '../raster-algebra/raster-algebra-test-utils';
import {
  profileRasterOnCPU,
  type OracleRaster,
  type RasterProfileOracle
} from './raster-sampling-oracle';

type Scene = {
  raster: OracleRaster;
  pathPositions: Float32Array;
  pathOffsets: Uint32Array;
};

const SAMPLE_COLUMNS = {
  samplePositions: 'float32x2',
  sampleDistances: 'float32',
  sampleValues: 'float32',
  samplePathIds: 'uint32',
  sampleCumulativeGain: 'float32',
  sampleCumulativeLoss: 'float32'
} as const;
const PATH_COLUMNS = {
  pathLength: 'float32',
  pathGain: 'float32',
  pathLoss: 'float32',
  pathMinimum: 'float32',
  pathMaximum: 'float32'
} as const;
type SampleColumn = keyof typeof SAMPLE_COLUMNS;
type PathColumn = keyof typeof PATH_COLUMNS;
type ColumnName = SampleColumn | PathColumn | 'pathSampleOffsets';

async function createHarness(
  scene: Scene,
  capacity: number,
  columns: {samples: readonly SampleColumn[]; paths: readonly PathColumn[]; offsets: boolean}
) {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return null;
  }
  const {raster, pathPositions, pathOffsets} = scene;
  const pathCount = pathOffsets.length - 1;
  const inputs = {
    raster: createInputBuffer(device, raster.values),
    validity: raster.validity ? createInputBuffer(device, raster.validity) : undefined,
    pathPositions: createInputBuffer(device, pathPositions),
    pathOffsets: createInputBuffer(device, pathOffsets)
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'profile-parameters',
    format: 'float32',
    length: 12
  });
  const graph = new GPUCommandGraph(device, {id: 'profile-graph'});
  const scalars = {
    count: createOutputBuffer(device, 1),
    overflow: createOutputBuffer(device, 1),
    totalCount: createOutputBuffer(device, 1)
  };
  const output: Record<string, unknown> = {
    count: importGraphBuffer(graph, 'count', scalars.count, 'uint32', 1),
    overflow: importGraphBuffer(graph, 'overflow', scalars.overflow, 'uint32', 1),
    totalCount: importGraphBuffer(graph, 'total-count', scalars.totalCount, 'uint32', 1)
  };
  const outputBuffers: Record<string, ReturnType<typeof createOutputBuffer>> = {};
  const lengths: Record<string, number> = {};
  const formats: Record<string, string> = {};
  const addOutput = (name: string, format: string, length: number, wordsPerRow = 1) => {
    outputBuffers[name] = createOutputBuffer(device, length * wordsPerRow);
    lengths[name] = length * wordsPerRow;
    formats[name] = format;
    output[name] = importGraphBuffer(graph, name, outputBuffers[name], format as 'float32', length);
  };
  for (const name of columns.samples) {
    addOutput(name, SAMPLE_COLUMNS[name], capacity, name === 'samplePositions' ? 2 : 1);
  }
  for (const name of columns.paths) {
    addOutput(name, PATH_COLUMNS[name], pathCount);
  }
  if (columns.offsets) {
    addOutput('pathSampleOffsets', 'uint32', pathCount + 1);
  }
  graph.add(
    new GPURasterProfile({
      id: 'profile',
      width: raster.width,
      height: raster.height,
      values: importGraphBuffer(graph, 'raster', inputs.raster, 'float32', raster.values.length),
      validity: inputs.validity
        ? importGraphBuffer(graph, 'validity', inputs.validity, 'uint32', raster.values.length)
        : undefined,
      noDataValue: raster.noDataValue,
      pathPositions: importGraphBuffer(
        graph,
        'path-positions',
        inputs.pathPositions,
        'float32x2',
        pathPositions.length / 2
      ),
      pathOffsets: importGraphBuffer(
        graph,
        'path-offsets',
        inputs.pathOffsets,
        'uint32',
        pathCount + 1
      ),
      sampleCapacity: capacity,
      parameters: parameterBuffer.importToGraph(graph),
      output: output as ConstructorParameters<typeof GPURasterProfile>[0]['output']
    })
  );
  let compileCount = 0;
  const compiled = graph.compile();
  compileCount++;
  return {
    async run(settings: GPURasterProfileSettings) {
      parameterBuffer.write(getGPURasterProfileParameterValues(settings));
      submitGraph(device, compiled, undefined);
      const read: Partial<Record<ColumnName, number[]>> = {};
      for (const name of Object.keys(outputBuffers) as ColumnName[]) {
        read[name] =
          formats[name] === 'uint32'
            ? await readUint32(outputBuffers[name], lengths[name])
            : await readFloat32(outputBuffers[name], lengths[name]);
      }
      return {
        columns: read,
        count: (await readUint32(scalars.count, 1))[0],
        overflow: (await readUint32(scalars.overflow, 1))[0],
        totalCount: (await readUint32(scalars.totalCount, 1))[0]
      };
    },
    get rebuildCount() {
      return compileCount - 1;
    },
    destroy() {
      compiled.destroy();
      parameterBuffer.destroy();
      for (const buffer of [
        ...Object.values(inputs),
        ...Object.values(scalars),
        ...Object.values(outputBuffers)
      ]) {
        buffer?.destroy();
      }
    }
  };
}

/** Compares a harness result with the oracle; returns the number of compared sample rows. */
function expectMatchesOracle(
  result: Awaited<ReturnType<NonNullable<Awaited<ReturnType<typeof createHarness>>>['run']>>,
  expected: RasterProfileOracle,
  capacity: number,
  maximumUlp: number,
  label: string
): void {
  const count = Math.min(expected.totalCount, capacity);
  expect(result.totalCount, `${label} totalCount`).toBe(expected.totalCount);
  expect(result.count, `${label} count`).toBe(count);
  expect(result.overflow, `${label} overflow`).toBe(
    expected.totalCount > capacity || expected.truncated ? 1 : 0
  );
  const pad = (rows: ArrayLike<number>, wordsPerRow: number, fill: number) => {
    const padded = new Array<number>(capacity * wordsPerRow).fill(fill);
    for (let index = 0; index < count * wordsPerRow; index++) {
      padded[index] = rows[index];
    }
    return padded;
  };
  const columns = result.columns;
  if (columns.samplePositions) {
    expectFloatArraysClose(
      columns.samplePositions,
      pad(expected.samplePositions, 2, NaN),
      maximumUlp,
      `${label} samplePositions`
    );
  }
  for (const name of ['sampleDistances', 'sampleValues'] as const) {
    if (columns[name]) {
      expectFloatArraysClose(
        columns[name],
        pad(expected[name], 1, NaN),
        maximumUlp,
        `${label} ${name}`
      );
    }
  }
  if (columns.samplePathIds) {
    expect(columns.samplePathIds, `${label} samplePathIds`).toEqual(
      pad(expected.samplePathIds, 1, GPU_RASTER_PROFILE_NO_PATH_ID)
    );
  }
  // Cumulative columns are only written for emitted rows.
  for (const name of ['sampleCumulativeGain', 'sampleCumulativeLoss'] as const) {
    if (columns[name]) {
      expectFloatArraysClose(
        columns[name].slice(0, count),
        Array.from(expected[name]).slice(0, count),
        maximumUlp,
        `${label} ${name}`
      );
    }
  }
  if (columns.pathSampleOffsets) {
    expect(columns.pathSampleOffsets, `${label} pathSampleOffsets`).toEqual(
      Array.from(expected.sampleOffsets, offset => Math.min(offset, capacity))
    );
  }
  if (expected.totalCount <= capacity) {
    for (const name of ['pathGain', 'pathLoss', 'pathMinimum', 'pathMaximum'] as const) {
      if (columns[name]) {
        expectFloatArraysClose(columns[name], expected[name], maximumUlp, `${label} ${name}`);
      }
    }
  }
  if (columns.pathLength) {
    expectFloatArraysClose(
      columns.pathLength,
      expected.pathLength,
      maximumUlp,
      `${label} pathLength`
    );
  }
}

const EXTENT = [0, 0, 8, 8] as const;
const ALL_SAMPLES = Object.keys(SAMPLE_COLUMNS) as SampleColumn[];
const ALL_PATHS = Object.keys(PATH_COLUMNS) as PathColumn[];

function createRamp(): OracleRaster {
  const values = new Float32Array(64);
  for (let cell = 0; cell < 64; cell++) {
    values[cell] = (cell % 8) * 10;
  }
  return {width: 8, height: 8, values};
}

it('GPURasterProfile profiles a ramp with known gain and loss as spacing changes without rebuilding', async () => {
  // Path 0 rises, path 1 is zero length, path 2 is empty, path 3 descends.
  const scene: Scene = {
    raster: createRamp(),
    pathPositions: new Float32Array([0.5, 4, 6.5, 4, 3, 3, 6.5, 4, 0.5, 4]),
    pathOffsets: Uint32Array.from([0, 2, 3, 3, 5])
  };
  const harness = await createHarness(scene, 64, {
    samples: ALL_SAMPLES,
    paths: ALL_PATHS,
    offsets: true
  });
  if (!harness) {
    return;
  }
  const first = await harness.run({width: 8, height: 8, extent: EXTENT, spacing: 2});
  expect(first.count).toBe(9);
  expect(first.columns.pathSampleOffsets).toEqual([0, 4, 5, 5, 9]);
  expect(first.columns.sampleDistances?.slice(0, 4)).toEqual([0, 2, 4, 6]);
  expect(first.columns.sampleValues?.slice(0, 4)).toEqual([0, 20, 40, 60]);
  expect(first.columns.pathGain).toEqual([60, 0, 0, 0]);
  expect(first.columns.pathLoss).toEqual([0, 0, 0, 60]);
  expect(first.columns.pathLength).toEqual([6, 0, 0, 6]);
  expect(first.columns.samplePathIds?.slice(0, 9)).toEqual([0, 0, 0, 0, 1, 3, 3, 3, 3]);
  for (const [spacing, method] of [
    [2, 'bilinear'],
    [4, 'bilinear'],
    [0.5, 'bilinear'],
    [1, 'bicubic'],
    [0.25, 'bicubic'],
    [10, 'bilinear'],
    [2, 'bilinear']
  ] as const) {
    const settings: GPURasterProfileSettings = {
      width: 8,
      height: 8,
      extent: EXTENT,
      method,
      spacing
    };
    const result = await harness.run(settings);
    expectMatchesOracle(
      result,
      profileRasterOnCPU(
        scene.raster,
        getGPURasterProfileParameterValues(settings),
        scene.pathPositions,
        scene.pathOffsets
      ),
      64,
      // Segment length 6 makes the interpolation parameter a non-dyadic division, which WebGPU
      // only bounds; counts, offsets and path ids are still exact.
      64,
      `ramp ${method} spacing ${spacing}`
    );
  }
  expect(harness.rebuildCount).toBe(0);
  harness.destroy();
});

/** Random axis-aligned polylines with power-of-two segment lengths: all f32 arithmetic is exact. */
function createExactScene(seed: number): Scene {
  const random = createRandom(seed);
  const width = 64;
  const height = 64;
  const values = new Float32Array(width * height);
  for (let cell = 0; cell < values.length; cell++) {
    const roll = random();
    values[cell] = roll < 0.05 ? NaN : Math.floor(random() * 200) - 50;
  }
  const lengths = [2, 4, 8];
  const positions: number[] = [];
  const offsets = [0];
  for (let path = 0; path < 14; path++) {
    // Path 5 has a single vertex (zero length); path 9 is empty; path 11 has coincident vertices.
    let x = Math.floor(random() * 40) + 8;
    let y = Math.floor(random() * 40) + 8;
    const vertexCount = path === 5 ? 1 : path === 9 ? 0 : 2 + Math.floor(random() * 5);
    for (let vertex = 0; vertex < vertexCount; vertex++) {
      positions.push(x, y);
      if (path === 11) {
        continue;
      }
      const length = lengths[Math.floor(random() * 3)] * (random() < 0.5 ? -1 : 1);
      if (random() < 0.5) {
        x = Math.min(Math.max(x + length, -2), 66);
      } else {
        y = Math.min(Math.max(y + length, -2), 66);
      }
    }
    offsets.push(positions.length / 2);
  }
  return {
    raster: {width, height, values},
    pathPositions: new Float32Array(positions),
    pathOffsets: Uint32Array.from(offsets)
  };
}

it('GPURasterProfile matches the oracle bit-exactly on exact multi-path scenes for every method and policy', async () => {
  const scene = createExactScene(3);
  const harness = await createHarness(scene, 4096, {
    samples: ALL_SAMPLES,
    paths: ALL_PATHS,
    offsets: true
  });
  if (!harness) {
    return;
  }
  const extent = [0, 0, 64, 64] as const;
  let sawNaNValue = false;
  for (const [method, noDataPolicy, spacing] of [
    ['nearest', 'strict', 0.75],
    ['bilinear', 'strict', 1.5],
    ['bilinear', 'renormalize', 0.5],
    ['bicubic', 'strict', 2],
    ['bicubic', 'renormalize', 1]
  ] as const) {
    const settings: GPURasterProfileSettings = {
      width: 64,
      height: 64,
      extent,
      method,
      noDataPolicy,
      spacing
    };
    const parameters = getGPURasterProfileParameterValues(settings);
    const expected = profileRasterOnCPU(
      scene.raster,
      parameters,
      scene.pathPositions,
      scene.pathOffsets
    );
    expect(expected.totalCount).toBeGreaterThan(50);
    sawNaNValue ||= expected.sampleValues.some(Number.isNaN);
    const result = await harness.run(settings);
    // renormalize divides, which WebGPU only bounds; everything else is exact.
    expectMatchesOracle(
      result,
      expected,
      4096,
      noDataPolicy === 'renormalize' ? 8 : 0,
      `${method}/${noDataPolicy}`
    );
  }
  expect(sawNaNValue).toBe(true);
  expect(harness.rebuildCount).toBe(0);
  harness.destroy();
});

it('GPURasterProfile clamps to the sample capacity and reports overflow and the total', async () => {
  const scene = createExactScene(8);
  const extent = [0, 0, 64, 64] as const;
  const settings: GPURasterProfileSettings = {
    width: 64,
    height: 64,
    extent,
    method: 'bilinear',
    spacing: 1
  };
  const expected = profileRasterOnCPU(
    scene.raster,
    getGPURasterProfileParameterValues(settings),
    scene.pathPositions,
    scene.pathOffsets
  );
  expect(expected.totalCount).toBeGreaterThan(40);
  for (const capacity of [40, expected.totalCount, expected.totalCount + 25]) {
    const harness = await createHarness(scene, capacity, {
      samples: ALL_SAMPLES,
      paths: ALL_PATHS,
      offsets: true
    });
    if (!harness) {
      return;
    }
    const result = await harness.run(settings);
    expectMatchesOracle(result, expected, capacity, 0, `capacity ${capacity}`);
    expect(result.overflow).toBe(capacity < expected.totalCount ? 1 : 0);
    harness.destroy();
  }
});

it('GPURasterProfile supports subsets of its optional outputs', async () => {
  const scene = createExactScene(5);
  const settings: GPURasterProfileSettings = {
    width: 64,
    height: 64,
    extent: [0, 0, 64, 64],
    method: 'bilinear',
    spacing: 1.5
  };
  const expected = profileRasterOnCPU(
    scene.raster,
    getGPURasterProfileParameterValues(settings),
    scene.pathPositions,
    scene.pathOffsets
  );
  const subsets: {samples: SampleColumn[]; paths: PathColumn[]; offsets: boolean}[] = [
    {samples: [], paths: [], offsets: false},
    {samples: ['sampleValues'], paths: ['pathGain', 'pathLoss'], offsets: false},
    {samples: ['samplePathIds', 'sampleDistances'], paths: [], offsets: true},
    {samples: ['sampleCumulativeLoss'], paths: ['pathMinimum', 'pathMaximum'], offsets: false},
    {samples: ['samplePositions'], paths: ['pathLength'], offsets: true}
  ];
  for (const subset of subsets) {
    const harness = await createHarness(scene, 1024, subset);
    if (!harness) {
      return;
    }
    expectMatchesOracle(await harness.run(settings), expected, 1024, 0, JSON.stringify(subset));
    harness.destroy();
  }
});

it('GPURasterProfile honours nodata sentinels and validity masks', async () => {
  const scene = createExactScene(13);
  scene.raster.noDataValue = 7;
  scene.raster.validity = new Uint32Array(64 * 64).fill(1);
  for (let cell = 0; cell < 4096; cell += 11) {
    scene.raster.validity[cell] = 0;
  }
  const harness = await createHarness(scene, 2048, {
    samples: ['sampleValues'],
    paths: ['pathGain', 'pathLoss', 'pathMinimum', 'pathMaximum'],
    offsets: false
  });
  if (!harness) {
    return;
  }
  const settings: GPURasterProfileSettings = {
    width: 64,
    height: 64,
    extent: [0, 0, 64, 64],
    method: 'nearest',
    spacing: 1
  };
  const expected = profileRasterOnCPU(
    scene.raster,
    getGPURasterProfileParameterValues(settings),
    scene.pathPositions,
    scene.pathOffsets
  );
  expectMatchesOracle(await harness.run(settings), expected, 2048, 0, 'nodata');
  harness.destroy();
});

it('GPURasterProfile limits samples per path and flags the truncation', async () => {
  // Four paths of length 6; a 1e-7 spacing asks for ~6e7 samples each, above the 2^24 limit.
  const scene: Scene = {
    raster: createRamp(),
    pathPositions: new Float32Array([0.5, 4, 6.5, 4, 0.5, 2, 6.5, 2, 0.5, 6, 6.5, 6, 3, 3]),
    pathOffsets: Uint32Array.from([0, 2, 4, 6, 7])
  };
  const harness = await createHarness(scene, 16, {
    samples: ['sampleDistances', 'sampleValues'],
    paths: [],
    offsets: true
  });
  if (!harness) {
    return;
  }
  const result = await harness.run({
    width: 8,
    height: 8,
    extent: EXTENT,
    method: 'nearest',
    spacing: 1e-7
  });
  const perPath = 1 << 24;
  expect(result.totalCount).toBe(3 * perPath + 1);
  expect(result.count).toBe(16);
  expect(result.overflow).toBe(1);
  expect(result.columns.pathSampleOffsets).toEqual([0, 16, 16, 16, 16]);
  expect(result.columns.sampleDistances?.slice(0, 3)).toEqual([
    0,
    Math.fround(1e-7),
    Math.fround(2 * Math.fround(1e-7))
  ]);
  harness.destroy();
});
