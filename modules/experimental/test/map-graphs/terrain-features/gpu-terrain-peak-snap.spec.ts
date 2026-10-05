// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUMapGraphParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {
  getGPUTerrainPeakSnapParameterValues,
  GPU_TERRAIN_PEAK_SNAP_PARAMETER_LENGTH,
  GPU_TERRAIN_PEAK_SNAP_STATUS as S,
  GPUTerrainPeakSnap,
  type GPUTerrainPeakSnapProps,
  type GPUTerrainPeakSnapSettings
} from '../../../src/map-graphs/terrain-features/gpu-terrain-peak-snap';
import {
  createInputBuffer,
  createOutputBuffer,
  isSoftwareDevice,
  readFloat32,
  readUint32
} from '../map-graph-test-utils';
import {computeTerrainPeakSnap} from './terrain-summits-oracle';

type FixtureOptions = {
  values: Float32Array;
  validity?: Uint32Array;
  width: number;
  height: number;
  candidates: number[];
  candidateHeights?: number[];
  candidateRadii?: number[];
  settings: GPUTerrainPeakSnapSettings;
  overflow?: boolean;
  recipe?: Partial<GPUTerrainPeakSnapProps>;
};

function createPeakSnapFixture(device: Device, options: FixtureOptions) {
  const {width, height} = options;
  const pixelCount = width * height;
  const candidateCount = options.candidates.length / 2;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const elevation = track(createInputBuffer(device, options.values));
  const validity = options.validity
    ? track(createInputBuffer(device, options.validity))
    : undefined;
  const candidates = track(createInputBuffer(device, Float32Array.from(options.candidates)));
  const candidateHeights = options.candidateHeights
    ? track(createInputBuffer(device, Float32Array.from(options.candidateHeights)))
    : undefined;
  const candidateRadii = options.candidateRadii
    ? track(createInputBuffer(device, Float32Array.from(options.candidateRadii)))
    : undefined;
  const positions = track(createOutputBuffer(device, candidateCount * 2));
  const heights = track(createOutputBuffer(device, candidateCount));
  const status = track(createOutputBuffer(device, candidateCount));
  const snapDistance = track(createOutputBuffer(device, candidateCount));
  const overflow = track(createOutputBuffer(device, 1));
  const settingsBuffer = new GPUMapGraphParameterBuffer(device, {
    id: 'snap-settings',
    format: 'float32',
    length: GPU_TERRAIN_PEAK_SNAP_PARAMETER_LENGTH,
    values: getGPUTerrainPeakSnapParameterValues(options.settings)
  });
  const graph = new GPUCommandGraph(device, {id: 'terrain-peak-snap-test'});
  graph.add(
    new GPUTerrainPeakSnap({
      width,
      height,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', elevation, 'float32', pixelCount)
        },
        validity: validity
          ? importGraphBuffer(graph, 'validity', validity, 'uint32', pixelCount)
          : undefined
      },
      settings: settingsBuffer.importToGraph(graph),
      candidates: importGraphBuffer(graph, 'candidates', candidates, 'float32x2', candidateCount),
      candidateHeights: candidateHeights
        ? importGraphBuffer(graph, 'candidate-heights', candidateHeights, 'float32', candidateCount)
        : undefined,
      candidateRadii: candidateRadii
        ? importGraphBuffer(graph, 'candidate-radii', candidateRadii, 'float32', candidateCount)
        : undefined,
      positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', candidateCount),
      heights: importGraphBuffer(graph, 'heights', heights, 'float32', candidateCount),
      status: importGraphBuffer(graph, 'status', status, 'uint32', candidateCount),
      snapDistance: importGraphBuffer(graph, 'distance', snapDistance, 'float32', candidateCount),
      overflow: options.overflow
        ? importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1)
        : undefined,
      ...options.recipe
    })
  );
  const compiled = graph.compile();
  return {
    async run(settings: GPUTerrainPeakSnapSettings = options.settings) {
      settingsBuffer.write(getGPUTerrainPeakSnapParameterValues(settings));
      submitGraph(device, compiled, undefined);
      return {
        positions: await readFloat32(positions, candidateCount * 2),
        heights: await readFloat32(heights, candidateCount),
        status: await readUint32(status, candidateCount),
        snapDistance: await readFloat32(snapDistance, candidateCount),
        overflow: (await readUint32(overflow, 1))[0]
      };
    },
    destroy() {
      compiled.destroy();
      settingsBuffer.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

function expectMatchesOracle(
  actual: Awaited<ReturnType<ReturnType<typeof createPeakSnapFixture>['run']>>,
  expected: ReturnType<typeof computeTerrainPeakSnap>,
  exactHeights: boolean = true
) {
  expect(actual.status).toEqual(expected.status);
  expect(actual.positions).toEqual(expected.positions);
  if (exactHeights) {
    expect(actual.heights).toEqual(expected.heights);
  }
  actual.snapDistance.forEach((distance, index) => {
    // sqrt is not correctly rounded in WGSL: allow a few ULP.
    expect(Math.abs(distance - expected.snapDistance[index])).toBeLessThanOrEqual(
      4e-7 * Math.max(1, expected.snapDistance[index])
    );
  });
}

function createNoise(width: number, height: number, levels: number, seed: number): Float32Array {
  let state = seed >>> 0;
  return Float32Array.from({length: width * height}, () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return (state >>> 8) % levels;
  });
}

function createHills(width: number, height: number): Float32Array {
  const hills = [
    {x: 12, y: 10, amplitude: 100, sigma: 4},
    {x: 24, y: 16, amplitude: 60, sigma: 3}
  ];
  return Float32Array.from({length: width * height}, (_, index) => {
    const column = index % width;
    const row = Math.floor(index / width);
    return hills.reduce(
      (sum, hill) =>
        sum +
        hill.amplitude *
          Math.exp(-((column - hill.x) ** 2 + (row - hill.y) ** 2) / (2 * hill.sigma ** 2)),
      0
    );
  });
}

const WIDTH = 32;
const HEIGHT = 24;
const SETTINGS: GPUTerrainPeakSnapSettings = {
  radius: 4,
  maximumMove: 100,
  maximumHeightChange: 1000,
  cellSize: [1, 1]
};

it('GPUTerrainPeakSnap snaps to hill summits, keeps flanks, and reports status', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const values = createHills(WIDTH, HEIGHT);
  const candidates = [
    10.25,
    8.5, // near the first summit: snapped
    12,
    10, // exactly on the summit: unchanged
    2,
    10, // far west flank: best pixel on the ring
    24.5,
    15.75, // near the second summit
    -1,
    3, // outside
    Number.NaN,
    4 // non-finite
  ];
  const parameters = getGPUTerrainPeakSnapParameterValues(SETTINGS);
  const fixture = createPeakSnapFixture(device, {
    values,
    width: WIDTH,
    height: HEIGHT,
    candidates,
    settings: SETTINGS,
    overflow: true
  });
  const result = await fixture.run();
  const expected = computeTerrainPeakSnap(
    values,
    undefined,
    WIDTH,
    HEIGHT,
    candidates,
    undefined,
    undefined,
    parameters
  );
  expectMatchesOracle(result, expected);
  expect(result.status).toEqual([
    S.snapped,
    S.unchanged,
    S.onRing,
    S.snapped,
    S.outside,
    S.outside
  ]);
  expect(result.positions.slice(0, 4)).toEqual([12, 10, 12, 10]);
  expect(result.positions.slice(4, 6)).toEqual([2, 10]);
  expect(result.heights[0]).toBe(values[10 * WIDTH + 12]);
  expect(result.snapDistance[0]).toBeCloseTo(Math.hypot(1.75, 1.5), 5);
  expect(result.overflow).toBe(0);

  // With the interior rule off the flank candidate climbs to the ring maximum.
  const noInterior = {...SETTINGS, interior: false};
  const flank = await fixture.run(noInterior);
  expect(flank.status[2]).toBe(S.snapped);
  expectMatchesOracle(
    flank,
    computeTerrainPeakSnap(
      values,
      undefined,
      WIDTH,
      HEIGHT,
      candidates,
      undefined,
      undefined,
      getGPUTerrainPeakSnapParameterValues(noInterior)
    )
  );
  fixture.destroy();
});

it('GPUTerrainPeakSnap breaks plateau ties by lowest index', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const values = new Float32Array(WIDTH * HEIGHT);
  for (let row = 5; row < 12; row++) {
    for (let column = 8; column < 15; column++) {
      values[row * WIDTH + column] = 10;
    }
  }
  const candidates = [11, 8.25];
  const settings = {...SETTINGS, radius: 2, interior: false};
  const fixture = createPeakSnapFixture(device, {
    values,
    width: WIDTH,
    height: HEIGHT,
    candidates,
    settings
  });
  const result = await fixture.run();
  expectMatchesOracle(
    result,
    computeTerrainPeakSnap(
      values,
      undefined,
      WIDTH,
      HEIGHT,
      candidates,
      undefined,
      undefined,
      getGPUTerrainPeakSnapParameterValues(settings)
    )
  );
  // The disc around (11, 8.25) reaches rows 7..10 (row 6 is 2.25 away); the top row's
  // lowest column wins.
  expect(result.status[0]).toBe(S.snapped);
  expect(result.positions[1]).toBe(7);
  const interior = await fixture.run({...settings, interior: true});
  expect(interior.status[0]).toBe(S.onRing);
  expect(interior.positions).toEqual([11, 8.25]);
  fixture.destroy();
});

it('GPUTerrainPeakSnap never lets nodata bleed and reports noData', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const values = Float32Array.from({length: WIDTH * HEIGHT}, (_, index) => {
    const column = index % WIDTH;
    const row = Math.floor(index / WIDTH);
    return 10 - 0.37 * Math.abs(column - 8) - 0.51 * Math.abs(row - 8);
  });
  const validity = new Uint32Array(WIDTH * HEIGHT).fill(1);
  for (const [column, row, sentinel] of [
    [9, 8, 1e30],
    [8, 9, -32768]
  ] as const) {
    values[row * WIDTH + column] = sentinel;
    validity[row * WIDTH + column] = 0;
  }
  // A block of invalid pixels with a huge sentinel in every value slot.
  for (let row = 16; row < 23; row++) {
    for (let column = 20; column < 27; column++) {
      values[row * WIDTH + column] = 1e30;
      validity[row * WIDTH + column] = 0;
    }
  }
  const settings = {...SETTINGS, radius: 3, interior: false};
  const candidates = [9, 9, 9.5, 8.5, 23, 19, 7.5, 7.5];
  const fixture = createPeakSnapFixture(device, {
    values,
    validity,
    width: WIDTH,
    height: HEIGHT,
    candidates,
    settings,
    recipe: {}
  });
  const result = await fixture.run();
  const expected = computeTerrainPeakSnap(
    values,
    validity,
    WIDTH,
    HEIGHT,
    candidates,
    undefined,
    undefined,
    getGPUTerrainPeakSnapParameterValues(settings)
  );
  expectMatchesOracle(result, expected);
  expect(result.status[2]).toBe(S.noData);
  expect(Number.isNaN(result.heights[2])).toBe(true);
  expect(result.status[0]).toBe(S.snapped);
  expect(result.positions.slice(0, 2)).toEqual([8, 8]);
  expect(result.heights[0]).toBe(10);
  expect(result.heights.every(value => !(Math.abs(value) > 1e6))).toBe(true);
  // A rejected move reports the bilinear DEM at the candidate, unknown (NaN) because the
  // corner (9, 8) is invalid; the sentinel in its value slot never appears.
  const rejected = await fixture.run({...settings, maximumMove: 0.5});
  expect(rejected.status[1]).toBe(S.rejectedMove);
  expect(Number.isNaN(rejected.heights[1])).toBe(true);
  expect(rejected.positions.slice(2, 4)).toEqual([9.5, 8.5]);
  fixture.destroy();
});

it('GPUTerrainPeakSnap rejects long moves and implausible height changes', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const values = new Float32Array(WIDTH * HEIGHT);
  values[10 * WIDTH + 12] = 100;
  values[9 * WIDTH + 12] = 40;
  values[10 * WIDTH + 11] = 40;
  values[10 * WIDTH + 13] = 40;
  values[11 * WIDTH + 12] = 40;
  const candidates = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11.5, 10];
  const candidateHeights = [Number.NaN, 95, 30, Number.POSITIVE_INFINITY, Number.NaN, 100];
  const base = {
    radius: 5,
    maximumMove: 10,
    maximumHeightChange: 20,
    cellSize: [1, 1],
    interior: false
  } as const;
  const fixture = createPeakSnapFixture(device, {
    values,
    width: WIDTH,
    height: HEIGHT,
    candidates,
    candidateHeights,
    settings: base
  });
  const result = await fixture.run();
  expectMatchesOracle(
    result,
    computeTerrainPeakSnap(
      values,
      undefined,
      WIDTH,
      HEIGHT,
      candidates,
      candidateHeights,
      undefined,
      getGPUTerrainPeakSnapParameterValues(base)
    )
  );
  // Candidate 0: DEM at (10, 10) is 0, summit 100 differs by 100 > 20.
  expect(result.status[0]).toBe(S.rejectedHeight);
  // Candidate 1: catalogue height 95 is close to 100.
  expect(result.status[1]).toBe(S.snapped);
  // Candidate 2: catalogue height 30 is 70 below the summit.
  expect(result.status[2]).toBe(S.rejectedHeight);
  // Candidate 3: infinite height is unknown, falls back to the DEM (0), still rejected.
  expect(result.status[3]).toBe(S.rejectedHeight);
  // Candidate 5 sits next to the summit: its nearest pixel is the best pixel.
  expect(result.status[5]).toBe(S.unchanged);
  // Move limit.
  const tight = {...base, maximumMove: 1.5};
  const moved = await fixture.run(tight);
  expect(moved.status[1]).toBe(S.rejectedMove);
  expect(moved.positions.slice(2, 4)).toEqual([10, 10]);
  expect(moved.heights[1]).toBe(0);
  expectMatchesOracle(
    moved,
    computeTerrainPeakSnap(
      values,
      undefined,
      WIDTH,
      HEIGHT,
      candidates,
      candidateHeights,
      undefined,
      getGPUTerrainPeakSnapParameterValues(tight)
    )
  );
  fixture.destroy();
});

it('GPUTerrainPeakSnap matches the oracle on random candidates and per-candidate radii', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const values = createNoise(WIDTH, HEIGHT, 100, 9);
  let state = 12345;
  const random = (count: number) => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return (state >>> 8) % count;
  };
  const candidateCount = 120;
  const candidates: number[] = [];
  const candidateHeights: number[] = [];
  const candidateRadii: number[] = [];
  for (let index = 0; index < candidateCount; index++) {
    candidates.push(
      random(4 * (WIDTH - 1) + 5) / 4 - 0.25,
      random(4 * (HEIGHT - 1) + 5) / 4 - 0.25
    );
    candidateHeights.push(random(5) === 0 ? Number.NaN : random(100));
    candidateRadii.push([0, Number.NaN, 1.5, 2.5, 3, 4][random(6)]);
  }
  const settings = {
    radius: 3,
    maximumMove: 2.5,
    maximumHeightChange: 40,
    cellSize: [1, 1]
  } as const;
  for (const interior of [true, false]) {
    const fixture = createPeakSnapFixture(device, {
      values,
      width: WIDTH,
      height: HEIGHT,
      candidates,
      candidateHeights,
      candidateRadii,
      settings: {...settings, interior}
    });
    const result = await fixture.run();
    const expected = computeTerrainPeakSnap(
      values,
      undefined,
      WIDTH,
      HEIGHT,
      candidates,
      candidateHeights,
      candidateRadii,
      getGPUTerrainPeakSnapParameterValues({...settings, interior})
    );
    expectMatchesOracle(result, expected);
    // Every outcome occurs, so the comparison is not vacuous.
    for (const code of [S.unchanged, S.snapped, S.rejectedMove, S.rejectedHeight, S.outside]) {
      expect(expected.status).toContain(code);
    }
    if (interior) {
      expect(expected.status).toContain(S.onRing);
    }
    fixture.destroy();
  }
});

it('GPUTerrainPeakSnap clamps the radius to maximumRadiusPixels and raises overflow', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const values = createNoise(WIDTH, HEIGHT, 1000, 4);
  const candidates = [10, 10, 20.5, 12.25];
  const options = {maximumRadiusPixels: 4};
  const settings = {...SETTINGS, radius: 3, interior: false};
  const fixture = createPeakSnapFixture(device, {
    values,
    width: WIDTH,
    height: HEIGHT,
    candidates,
    settings,
    overflow: true,
    recipe: options
  });
  expect((await fixture.run()).overflow).toBe(0);
  const wide = {...settings, radius: 9};
  const result = await fixture.run(wide);
  expect(result.overflow).toBe(1);
  const clamped = computeTerrainPeakSnap(
    values,
    undefined,
    WIDTH,
    HEIGHT,
    candidates,
    undefined,
    undefined,
    getGPUTerrainPeakSnapParameterValues({...settings, radius: 4}),
    options
  );
  expectMatchesOracle(result, clamped);
  expect(clamped.status).toContain(S.snapped);
  fixture.destroy();
});

it('GPUTerrainPeakSnap follows per-row cell sizes in web-mercator and geographic modes', async () => {
  const device = await getWebGPUTestDevice();
  if (!device || isSoftwareDevice(device)) {
    return;
  }
  const values = createNoise(WIDTH, HEIGHT, 100000, 31);
  const candidates: number[] = [];
  for (let row = 2; row < HEIGHT - 2; row += 2) {
    candidates.push(10.5 + (row % 3) * 0.25, row);
  }
  const cases = [
    {
      cellSizeMode: 'web-mercator' as const,
      settings: {
        radius: 70,
        maximumMove: 1000,
        maximumHeightChange: 1e9,
        cellSize: [20, 20],
        northEdge: 0.3,
        southEdge: 0.45,
        interior: false
      } as const
    },
    {
      cellSizeMode: 'geographic' as const,
      settings: {
        radius: 90,
        maximumMove: 1000,
        maximumHeightChange: 1e9,
        cellSize: [0.0003, 0.0003],
        northEdge: 70,
        southEdge: 50,
        interior: false
      } as const
    }
  ];
  for (const {cellSizeMode, settings} of cases) {
    const compute = (radiusScale: number, mode: 'uniform' | typeof cellSizeMode) =>
      computeTerrainPeakSnap(
        values,
        undefined,
        WIDTH,
        HEIGHT,
        candidates,
        undefined,
        undefined,
        getGPUTerrainPeakSnapParameterValues({...settings, radius: settings.radius * radiusScale}),
        {cellSizeMode: mode}
      );
    const expected = compute(1, cellSizeMode);
    // Away from disc boundaries: a 0.01 % radius change leaves every outcome unchanged.
    expect(compute(1.0001, cellSizeMode).positions).toEqual(expected.positions);
    expect(compute(0.9999, cellSizeMode).positions).toEqual(expected.positions);
    expect(compute(1, 'uniform').positions).not.toEqual(expected.positions);
    const fixture = createPeakSnapFixture(device, {
      values,
      width: WIDTH,
      height: HEIGHT,
      candidates,
      settings,
      recipe: {cellSizeMode}
    });
    const result = await fixture.run();
    expect(result.status).toEqual(expected.status);
    expect(result.positions).toEqual(expected.positions);
    expect(expected.status).toContain(S.snapped);
    fixture.destroy();
  }
});

it('getGPUTerrainPeakSnapParameterValues packs defaults and validates', () => {
  expect(Array.from(getGPUTerrainPeakSnapParameterValues({cellSize: [2, 3]}))).toEqual([
    150, 300, 80, 2, 3, 0, 0, 1
  ]);
  expect(
    Array.from(
      getGPUTerrainPeakSnapParameterValues({
        radius: 10,
        maximumMove: 20,
        maximumHeightChange: 0,
        cellSize: [1, 1],
        interior: false,
        northEdge: 1,
        southEdge: 2
      })
    )
  ).toEqual([10, 20, 0, 1, 1, 1, 2, 0]);
  expect(() => getGPUTerrainPeakSnapParameterValues({radius: 0, cellSize: [1, 1]})).toThrow(
    /radius/
  );
  expect(() => getGPUTerrainPeakSnapParameterValues({maximumMove: -1, cellSize: [1, 1]})).toThrow(
    /maximumMove/
  );
  expect(() => getGPUTerrainPeakSnapParameterValues({cellSize: [Number.NaN, 1]})).toThrow(
    /cell size/
  );
});
