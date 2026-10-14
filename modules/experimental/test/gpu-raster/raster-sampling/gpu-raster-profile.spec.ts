// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {makeGPURasterMetadata} from '../../../src/gpu-raster/raster-grid';
import {
  GPURasterProfile,
  GPU_RASTER_PROFILE_NO_LINE_ID
} from '../../../src/gpu-raster/raster-sampling';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {getSegmentedScanTreeHeight} from '../../../src/gpu-raster/raster-sampling/gpu-raster-profile-scan';
import {createRandom, expectFloatArraysClose} from '../raster-algebra/raster-algebra-test-utils';
import {
  packProfileCase,
  profileRasterOnCPU,
  type OracleRaster,
  type ProfileCase,
  type RasterProfileOracle
} from './raster-sampling-oracle';
import {requireTestDevice} from '../../utils/require-test-device';

type Scene = {
  raster: OracleRaster;
  linePositions: Float32Array;
  lineOffsets: Uint32Array;
};

const SAMPLE_COLUMNS = {
  samplePositions: 'float32x2',
  sampleDistances: 'float32',
  sampleValues: 'float32',
  sampleLineIds: 'uint32',
  sampleCumulativeGain: 'float32',
  sampleCumulativeLoss: 'float32'
} as const;
const LINE_COLUMNS = {
  lineLengths: 'float32',
  lineGains: 'float32',
  lineLosses: 'float32',
  lineMinimums: 'float32',
  lineMaximums: 'float32'
} as const;
type SampleColumn = keyof typeof SAMPLE_COLUMNS;
type LineColumn = keyof typeof LINE_COLUMNS;
type ColumnName = SampleColumn | LineColumn | 'lineSampleOffsets';

async function createHarness(
  scene: Scene,
  capacity: number,
  columns: {samples: readonly SampleColumn[]; lines: readonly LineColumn[]; offsets: boolean}
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {raster, linePositions, lineOffsets} = scene;
  const lineCount = lineOffsets.length - 1;
  const inputs = {
    raster: createInputBuffer(device, raster.values),
    validity: raster.validity ? createInputBuffer(device, raster.validity) : undefined,
    linePositions: createInputBuffer(device, linePositions),
    lineOffsets: createInputBuffer(device, lineOffsets)
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
    requiredCount: createOutputBuffer(device, 1)
  };
  const output: Record<string, unknown> = {
    status: {
      count: importGraphBuffer(graph, 'count', scalars.count, 'uint32', 1),
      overflow: importGraphBuffer(graph, 'overflow', scalars.overflow, 'uint32', 1),
      requiredCount: importGraphBuffer(graph, 'total-count', scalars.requiredCount, 'uint32', 1)
    }
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
  for (const name of columns.lines) {
    addOutput(name, LINE_COLUMNS[name], lineCount);
  }
  if (columns.offsets) {
    addOutput('lineSampleOffsets', 'uint32', lineCount + 1);
  }
  graph.add(
    new GPURasterProfile({
      id: 'profile',
      grid: makeGPURasterMetadata({width: raster.width, height: raster.height}),
      values: importGraphBuffer(graph, 'raster', inputs.raster, 'float32', raster.values.length),
      validity: inputs.validity
        ? importGraphBuffer(graph, 'validity', inputs.validity, 'uint32', raster.values.length)
        : undefined,
      noDataValue: raster.noDataValue,
      positions: importGraphBuffer(
        graph,
        'line-positions',
        inputs.linePositions,
        'float32x2',
        linePositions.length / 2
      ),
      lineOffsets: importGraphBuffer(
        graph,
        'line-offsets',
        inputs.lineOffsets,
        'uint32',
        lineCount + 1
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
    async run(settings: ProfileCase) {
      parameterBuffer.write(packProfileCase(settings));
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
        requiredCount: (await readUint32(scalars.requiredCount, 1))[0]
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
  const count = Math.min(expected.requiredCount, capacity);
  expect(result.requiredCount, `${label} requiredCount`).toBe(expected.requiredCount);
  expect(result.count, `${label} count`).toBe(count);
  expect(result.overflow, `${label} overflow`).toBe(
    expected.requiredCount > capacity || expected.truncated ? 1 : 0
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
  if (columns.sampleLineIds) {
    expect(columns.sampleLineIds, `${label} sampleLineIds`).toEqual(
      pad(expected.sampleLineIds, 1, GPU_RASTER_PROFILE_NO_LINE_ID)
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
  if (columns.lineSampleOffsets) {
    expect(columns.lineSampleOffsets, `${label} lineSampleOffsets`).toEqual(
      Array.from(expected.sampleOffsets, offset => Math.min(offset, capacity))
    );
  }
  if (expected.requiredCount <= capacity) {
    for (const name of ['lineGains', 'lineLosses', 'lineMinimums', 'lineMaximums'] as const) {
      if (columns[name]) {
        expectFloatArraysClose(columns[name], expected[name], maximumUlp, `${label} ${name}`);
      }
    }
  }
  if (columns.lineLengths) {
    expectFloatArraysClose(
      columns.lineLengths,
      expected.lineLengths,
      maximumUlp,
      `${label} lineLengths`
    );
  }
}

const EXTENT = [0, 0, 8, 8] as const;
const ALL_SAMPLES = Object.keys(SAMPLE_COLUMNS) as SampleColumn[];
const ALL_LINES = Object.keys(LINE_COLUMNS) as LineColumn[];

function createRamp(): OracleRaster {
  const values = new Float32Array(64);
  for (let cell = 0; cell < 64; cell++) {
    values[cell] = (cell % 8) * 10;
  }
  return {width: 8, height: 8, values};
}

it('GPURasterProfile profiles a ramp with known gain and loss as spacing changes without rebuilding', async () => {
  // Line 0 rises, line 1 is zero length, line 2 is empty, line 3 descends.
  const scene: Scene = {
    raster: createRamp(),
    linePositions: new Float32Array([0.5, 4, 6.5, 4, 3, 3, 6.5, 4, 0.5, 4]),
    lineOffsets: Uint32Array.from([0, 2, 3, 3, 5])
  };
  const harness = await createHarness(scene, 64, {
    samples: ALL_SAMPLES,
    lines: ALL_LINES,
    offsets: true
  });
  if (!harness) {
    return;
  }
  const first = await harness.run({width: 8, height: 8, extent: EXTENT, spacing: 2});
  expect(first.count).toBe(9);
  expect(first.columns.lineSampleOffsets).toEqual([0, 4, 5, 5, 9]);
  expect(first.columns.sampleDistances?.slice(0, 4)).toEqual([0, 2, 4, 6]);
  expect(first.columns.sampleValues?.slice(0, 4)).toEqual([0, 20, 40, 60]);
  expect(first.columns.lineGains).toEqual([60, 0, 0, 0]);
  expect(first.columns.lineLosses).toEqual([0, 0, 0, 60]);
  expect(first.columns.lineLengths).toEqual([6, 0, 0, 6]);
  expect(first.columns.sampleLineIds?.slice(0, 9)).toEqual([0, 0, 0, 0, 1, 3, 3, 3, 3]);
  for (const [spacing, method] of [
    [2, 'bilinear'],
    [4, 'bilinear'],
    [0.5, 'bilinear'],
    [1, 'bicubic'],
    [0.25, 'bicubic'],
    [10, 'bilinear'],
    [2, 'bilinear']
  ] as const) {
    const settings: ProfileCase = {
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
        packProfileCase(settings),
        scene.linePositions,
        scene.lineOffsets
      ),
      64,
      // Segment length 6 makes the interpolation parameter a non-dyadic division, which WebGPU
      // only bounds; counts, offsets and line ids are still exact.
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
  for (let line = 0; line < 14; line++) {
    // Line 5 has a single vertex (zero length); line 9 is empty; line 11 has coincident vertices.
    let x = Math.floor(random() * 40) + 8;
    let y = Math.floor(random() * 40) + 8;
    const vertexCount = line === 5 ? 1 : line === 9 ? 0 : 2 + Math.floor(random() * 5);
    for (let vertex = 0; vertex < vertexCount; vertex++) {
      positions.push(x, y);
      if (line === 11) {
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
    linePositions: new Float32Array(positions),
    lineOffsets: Uint32Array.from(offsets)
  };
}

it('GPURasterProfile matches the oracle bit-exactly on exact multi-line scenes for every method and policy', async () => {
  const scene = createExactScene(3);
  const harness = await createHarness(scene, 4096, {
    samples: ALL_SAMPLES,
    lines: ALL_LINES,
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
    const settings: ProfileCase = {
      width: 64,
      height: 64,
      extent,
      method,
      noDataPolicy,
      spacing
    };
    const parameters = packProfileCase(settings);
    const expected = profileRasterOnCPU(
      scene.raster,
      parameters,
      scene.linePositions,
      scene.lineOffsets
    );
    expect(expected.requiredCount).toBeGreaterThan(50);
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
  const settings: ProfileCase = {
    width: 64,
    height: 64,
    extent,
    method: 'bilinear',
    spacing: 1
  };
  const expected = profileRasterOnCPU(
    scene.raster,
    packProfileCase(settings),
    scene.linePositions,
    scene.lineOffsets
  );
  expect(expected.requiredCount).toBeGreaterThan(40);
  for (const capacity of [40, expected.requiredCount, expected.requiredCount + 25]) {
    const harness = await createHarness(scene, capacity, {
      samples: ALL_SAMPLES,
      lines: ALL_LINES,
      offsets: true
    });
    if (!harness) {
      return;
    }
    const result = await harness.run(settings);
    expectMatchesOracle(result, expected, capacity, 0, `capacity ${capacity}`);
    expect(result.overflow).toBe(capacity < expected.requiredCount ? 1 : 0);
    harness.destroy();
  }
});

it('GPURasterProfile supports subsets of its optional outputs', async () => {
  const scene = createExactScene(5);
  const settings: ProfileCase = {
    width: 64,
    height: 64,
    extent: [0, 0, 64, 64],
    method: 'bilinear',
    spacing: 1.5
  };
  const expected = profileRasterOnCPU(
    scene.raster,
    packProfileCase(settings),
    scene.linePositions,
    scene.lineOffsets
  );
  const subsets: {samples: SampleColumn[]; lines: LineColumn[]; offsets: boolean}[] = [
    {samples: [], lines: [], offsets: false},
    {samples: ['sampleValues'], lines: ['lineGains', 'lineLosses'], offsets: false},
    {samples: ['sampleLineIds', 'sampleDistances'], lines: [], offsets: true},
    {samples: ['sampleCumulativeLoss'], lines: ['lineMinimums', 'lineMaximums'], offsets: false},
    {samples: ['samplePositions'], lines: ['lineLengths'], offsets: true}
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
    lines: ['lineGains', 'lineLosses', 'lineMinimums', 'lineMaximums'],
    offsets: false
  });
  if (!harness) {
    return;
  }
  const settings: ProfileCase = {
    width: 64,
    height: 64,
    extent: [0, 0, 64, 64],
    method: 'nearest',
    spacing: 1
  };
  const expected = profileRasterOnCPU(
    scene.raster,
    packProfileCase(settings),
    scene.linePositions,
    scene.lineOffsets
  );
  expectMatchesOracle(await harness.run(settings), expected, 2048, 0, 'nodata');
  harness.destroy();
});

it('GPURasterProfile limits samples per line and flags the truncation', async () => {
  // Four lines of length 6; a 1e-7 spacing asks for ~6e7 samples each, above the 2^24 limit.
  const scene: Scene = {
    raster: createRamp(),
    linePositions: new Float32Array([0.5, 4, 6.5, 4, 0.5, 2, 6.5, 2, 0.5, 6, 6.5, 6, 3, 3]),
    lineOffsets: Uint32Array.from([0, 2, 4, 6, 7])
  };
  const harness = await createHarness(scene, 16, {
    samples: ['sampleDistances', 'sampleValues'],
    lines: [],
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
  const perLine = 1 << 24;
  expect(result.requiredCount).toBe(3 * perLine + 1);
  expect(result.count).toBe(16);
  expect(result.overflow).toBe(1);
  expect(result.columns.lineSampleOffsets).toEqual([0, 16, 16, 16, 16]);
  expect(result.columns.sampleDistances?.slice(0, 3)).toEqual([
    0,
    Math.fround(1e-7),
    Math.fround(2 * Math.fround(1e-7))
  ]);
  harness.destroy();
});

/**
 * Serpentine transect over a 512 x 512 raster: 512 rows of 511 plus 511 unit connectors, one line of
 * 262144 samples at spacing 1 (the final vertex lands on an integer distance), then an empty line
 * and a short second line. Cell values come from `makeValue(random)`; about 2% are NaN.
 */
function createLongTransectScene(makeValue: (random: () => number) => number): Scene {
  const size = 512;
  const random = createRandom(29);
  const values = new Float32Array(size * size);
  for (let cell = 0; cell < values.length; cell++) {
    values[cell] = random() < 0.02 ? NaN : makeValue(random);
  }
  const positions: number[] = [];
  for (let row = 0; row < size; row++) {
    const left = 0.5;
    const right = size - 0.5;
    positions.push(row % 2 === 0 ? left : right, row + 0.5);
    positions.push(row % 2 === 0 ? right : left, row + 0.5);
  }
  const longVertexCount = positions.length / 2;
  // Second line (after an empty one): east 30, then south 6.
  positions.push(10.5, 3.5, 40.5, 3.5, 40.5, 9.5);
  return {
    raster: {width: size, height: size, values},
    linePositions: new Float32Array(positions),
    lineOffsets: Uint32Array.from([0, longVertexCount, longVertexCount, longVertexCount + 3])
  };
}

const LONG_TRANSECT_SETTINGS: ProfileCase = {
  width: 512,
  height: 512,
  extent: [0, 0, 512, 512],
  method: 'nearest',
  spacing: 1
};

it('GPURasterProfile scans a 2.6e5-sample transect bit-exactly on integer data', async () => {
  // Cell values are integers in 0..9, so every delta is an integer of magnitude <= 9 and gain plus
  // loss over the 262144 samples is at most 9 * 262143 = 2.4e6 < 2^24. Every partial sum, in any
  // association order, is an integer below 2^24 and exact in float32: the segmented scan must
  // equal the serial float32 oracle bit for bit, including through the NaN cells.
  const scene = createLongTransectScene(random => Math.floor(random() * 10));
  const capacity = 262144 + 64;
  const harness = await createHarness(scene, capacity, {
    samples: ['sampleValues', 'sampleCumulativeGain', 'sampleCumulativeLoss'],
    lines: ALL_LINES,
    offsets: true
  });
  if (!harness) {
    return;
  }
  const expected = profileRasterOnCPU(
    scene.raster,
    packProfileCase(LONG_TRANSECT_SETTINGS),
    scene.linePositions,
    scene.lineOffsets
  );
  expect(expected.requiredCount).toBe(262144 + 37);
  expectMatchesOracle(await harness.run(LONG_TRANSECT_SETTINGS), expected, capacity, 0, 'long');
  harness.destroy();
});

it('GPURasterProfile scans a 2.6e5-sample transect within the derived tree tolerance on fractional data', async () => {
  // Fractional float32 cell values: gain and loss are no longer exact. The scan sums
  // non-negative deltas in a fixed tree whose height is at most H = 2 * ceil(ceil(n / 256) / 256)
  // + 15 = 23 combine levels for n = 262144 (see getSegmentedScanTreeHeight); gain and loss add up
  // to two terms per level, so the addition tree has height h <= 2H = 46, and each delta is itself
  // rounded once. A non-negative sum then has relative error at most
  // (h + 1) * 2^-24 / (1 - (h + 1) * 2^-24) = 2.8e-6 against the exact float64 sum of the float32
  // deltas (the serial float32 walk it replaces only guarantees (n - 1) * 2^-24 = 1.6e-2). The
  // reference uses the GPU's own sample values, which nearest sampling returns exactly.
  const scene = createLongTransectScene(random => Math.fround(random() * 10));
  const capacity = 262144 + 64;
  const harness = await createHarness(scene, capacity, {
    samples: ['sampleValues', 'sampleCumulativeGain', 'sampleCumulativeLoss'],
    lines: ALL_LINES,
    offsets: true
  });
  if (!harness) {
    return;
  }
  const result = await harness.run(LONG_TRANSECT_SETTINGS);
  const sampleValues = result.columns.sampleValues!;
  const lineOffsets = result.columns.lineSampleOffsets!;
  const height = 2 * getSegmentedScanTreeHeight(262144);
  const factor = (height + 1) * 2 ** -24;
  const relativeBound = factor / (1 - factor);
  for (let line = 0; line + 1 < lineOffsets.length; line++) {
    let gain = 0;
    let loss = 0;
    let previous = 0;
    let hasPrevious = false;
    let minimum = NaN;
    let maximum = NaN;
    for (let sample = lineOffsets[line]; sample < lineOffsets[line + 1]; sample++) {
      const value = sampleValues[sample];
      if (Number.isFinite(value)) {
        if (hasPrevious) {
          const delta = value - previous;
          gain += Math.max(delta, 0);
          loss += Math.max(-delta, 0);
        }
        minimum = hasPrevious ? Math.min(minimum, value) : value;
        maximum = hasPrevious ? Math.max(maximum, value) : value;
        previous = value;
        hasPrevious = true;
      }
      const cumulativeGain = result.columns.sampleCumulativeGain![sample];
      const cumulativeLoss = result.columns.sampleCumulativeLoss![sample];
      expect(Math.abs(cumulativeGain - gain), `gain[${sample}]`).toBeLessThanOrEqual(
        relativeBound * gain
      );
      expect(Math.abs(cumulativeLoss - loss), `loss[${sample}]`).toBeLessThanOrEqual(
        relativeBound * loss
      );
    }
    const empty = lineOffsets[line + 1] === lineOffsets[line];
    expect(
      Math.abs(result.columns.lineGains![line] - gain),
      `lineGains ${line}`
    ).toBeLessThanOrEqual(relativeBound * gain);
    expect(
      Math.abs(result.columns.lineLosses![line] - loss),
      `lineLosses ${line}`
    ).toBeLessThanOrEqual(relativeBound * loss);
    // Extrema involve no rounding: exact; an empty line reports NaN.
    expect(result.columns.lineMinimums![line], `lineMinimums ${line}`).toBe(empty ? NaN : minimum);
    expect(result.columns.lineMaximums![line], `lineMaximums ${line}`).toBe(empty ? NaN : maximum);
  }
  harness.destroy();
});
