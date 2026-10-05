// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUTerrainSummitsParameterValues,
  GPU_TERRAIN_SUMMITS_PARAMETER_LENGTH,
  GPUTerrainSummits,
  type GPUTerrainSummitsProps,
  type GPUTerrainSummitsSettings
} from '../../../src/gpu-terrain/terrain-features/gpu-terrain-summits';
import {
  createInputBuffer,
  createOutputBuffer,
  isSoftwareDevice,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {computeTerrainSummits} from './terrain-summits-oracle';

type FixtureOptions = {
  values: Float32Array;
  validity?: Uint32Array;
  width: number;
  height: number;
  settings: GPUTerrainSummitsSettings;
  capacity?: number;
  overflow?: boolean;
  contributor?: Partial<GPUTerrainSummitsProps>;
};

function createSummitsFixture(device: Device, options: FixtureOptions) {
  const {width, height} = options;
  const pixelCount = width * height;
  const capacity = options.capacity ?? pixelCount;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const elevation = track(createInputBuffer(device, options.values));
  const validity = options.validity
    ? track(createInputBuffer(device, options.validity))
    : undefined;
  const mask = track(createOutputBuffer(device, pixelCount));
  const drop = track(createOutputBuffer(device, pixelCount));
  const ids = track(createOutputBuffer(device, capacity));
  const count = track(createOutputBuffer(device, 1));
  const listOverflow = track(createOutputBuffer(device, 1));
  const totalCount = track(createOutputBuffer(device, 1));
  const outputDrop = track(createOutputBuffer(device, capacity));
  const radiusOverflow = track(createOutputBuffer(device, 1));
  const settingsBuffer = new GPUParameterBuffer(device, {
    id: 'summit-settings',
    format: 'float32',
    length: GPU_TERRAIN_SUMMITS_PARAMETER_LENGTH,
    values: getGPUTerrainSummitsParameterValues(options.settings)
  });
  const graph = new GPUCommandGraph(device, {id: 'terrain-summits-test'});
  graph.add(
    new GPUTerrainSummits({
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
      summitMask: importGraphBuffer(graph, 'mask', mask, 'uint32', pixelCount),
      drop: importGraphBuffer(graph, 'drop', drop, 'float32', pixelCount),
      output: {
        ids: importGraphBuffer(graph, 'ids', ids, 'uint32', capacity),
        count: importGraphBuffer(graph, 'count', count, 'uint32', 1),
        overflow: importGraphBuffer(graph, 'list-overflow', listOverflow, 'uint32', 1),
        totalCount: importGraphBuffer(graph, 'total', totalCount, 'uint32', 1)
      },
      outputDrop: importGraphBuffer(graph, 'output-drop', outputDrop, 'float32', capacity),
      overflow: options.overflow
        ? importGraphBuffer(graph, 'radius-overflow', radiusOverflow, 'uint32', 1)
        : undefined,
      ...options.contributor
    })
  );
  const compiled = graph.compile();
  return {
    async run(settings: GPUTerrainSummitsSettings = options.settings) {
      settingsBuffer.write(getGPUTerrainSummitsParameterValues(settings));
      submitGraph(device, compiled, undefined);
      const [listCount] = await readUint32(count, 1);
      return {
        mask: await readUint32(mask, pixelCount),
        drop: await readFloat32(drop, pixelCount),
        ids: (await readUint32(ids, capacity)).slice(0, listCount),
        count: listCount,
        listOverflow: (await readUint32(listOverflow, 1))[0],
        total: (await readUint32(totalCount, 1))[0],
        outputDrop: await readFloat32(outputDrop, capacity),
        radiusOverflow: (await readUint32(radiusOverflow, 1))[0]
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

/** Deterministic integer noise in `[0, levels)` so equal heights (ties) are common. */
function createNoise(width: number, height: number, levels: number, seed: number): Float32Array {
  let state = seed >>> 0;
  return Float32Array.from({length: width * height}, () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return (state >>> 8) % levels;
  });
}

function createHills(width: number, height: number): Float32Array {
  const hills = [
    {x: 12, y: 10, amplitude: 100, sigma: 6},
    {x: 30, y: 21, amplitude: 80, sigma: 5}
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

function expectMatchesOracle(
  actual: {mask: number[]; drop: number[]; ids: number[]; outputDrop: number[]},
  expected: ReturnType<typeof computeTerrainSummits>
) {
  expect(actual.mask).toEqual(expected.mask);
  expect(actual.drop).toEqual(expected.drop);
  expect(actual.ids).toEqual(expected.ids);
  expect(actual.outputDrop.slice(0, expected.ids.length)).toEqual(
    expected.ids.map(id => expected.drop[id])
  );
}

const BASE: GPUTerrainSummitsSettings = {radius: 3, cellSize: [1, 1]};

it('GPUTerrainSummits finds Gaussian hill summits and matches the oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 44;
  const height = 32;
  const values = createHills(width, height);
  const settings = {radius: 4, cellSize: [1, 1]} as const;
  const fixture = createSummitsFixture(device, {values, width, height, settings});
  const result = await fixture.run();
  const expected = computeTerrainSummits(
    values,
    undefined,
    width,
    height,
    getGPUTerrainSummitsParameterValues(settings)
  );
  expectMatchesOracle(result, expected);
  // Non-trivial: both hills yield their centre, and nothing else of note.
  expect(result.ids).toContain(10 * width + 12);
  expect(result.ids).toContain(21 * width + 30);
  expect(result.count).toBe(result.total);
  expect(result.listOverflow).toBe(0);
  expect(result.mask.filter(Boolean).length).toBe(result.count);
  expect(result.drop[10 * width + 12]).toBeGreaterThan(0);
  fixture.destroy();
});

it('GPUTerrainSummits keeps the lowest index of a tied plateau and filters by minimum drop', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 16;
  const height = 12;
  const values = new Float32Array(width * height);
  for (let row = 3; row < 8; row++) {
    for (let column = 4; column < 9; column++) {
      values[row * width + column] = 10;
    }
  }
  // A small bump with a drop of 2 away from the plateau.
  values[9 * width + 13] = 2;
  const settings = {radius: 2, cellSize: [1, 1]} as const;
  const fixture = createSummitsFixture(device, {values, width, height, settings});
  const result = await fixture.run();
  expectMatchesOracle(
    result,
    computeTerrainSummits(
      values,
      undefined,
      width,
      height,
      getGPUTerrainSummitsParameterValues(settings)
    )
  );
  const first = 3 * width + 4;
  expect(result.mask[first]).toBe(1);
  expect(result.mask[first + 1]).toBe(0);
  expect(result.mask[4 * width + 4]).toBe(0);
  // Plateau neighbours at ring distance have equal height, so the drop is zero.
  expect(result.drop[first]).toBe(0);
  expect(result.drop[9 * width + 13]).toBe(2);

  const strict = {...settings, minimumDrop: 3};
  const filtered = await fixture.run(strict);
  expect(filtered.mask[first]).toBe(0);
  expect(filtered.mask[9 * width + 13]).toBe(0);
  expectMatchesOracle(
    filtered,
    computeTerrainSummits(
      values,
      undefined,
      width,
      height,
      getGPUTerrainSummitsParameterValues(strict)
    )
  );
  fixture.destroy();
});

it('GPUTerrainSummits matches the oracle on tie-heavy noise', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 37;
  const height = 29;
  for (const [seed, levels, radius] of [
    [1, 6, 2],
    [2, 1000, 3],
    [3, 25, 2.5]
  ] as const) {
    const values = createNoise(width, height, levels, seed);
    const settings = {radius, cellSize: [1, 1]} as const;
    for (const incompleteNeighborhood of ['reject', 'ignore'] as const) {
      const fixture = createSummitsFixture(device, {
        values,
        width,
        height,
        settings,
        contributor: {incompleteNeighborhood}
      });
      const result = await fixture.run();
      const expected = computeTerrainSummits(
        values,
        undefined,
        width,
        height,
        getGPUTerrainSummitsParameterValues(settings),
        {incompleteNeighborhood}
      );
      expect(expected.ids.length).toBeGreaterThan(3);
      expectMatchesOracle(result, expected);
      fixture.destroy();
    }
  }
});

it('GPUTerrainSummits rejects or ignores grid edges', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 20;
  const height = 16;
  const values = createNoise(width, height, 50, 7);
  // Raise the corner and an edge pixel so they win their discs.
  values[0] = 1000;
  values[5 * width + width - 1] = 900;
  const parameters = getGPUTerrainSummitsParameterValues(BASE);
  const reject = createSummitsFixture(device, {values, width, height, settings: BASE});
  const rejected = await reject.run();
  expect(rejected.mask[0]).toBe(0);
  expect(rejected.mask[5 * width + width - 1]).toBe(0);
  expectMatchesOracle(
    rejected,
    computeTerrainSummits(values, undefined, width, height, parameters)
  );
  reject.destroy();

  const ignore = createSummitsFixture(device, {
    values,
    width,
    height,
    settings: BASE,
    contributor: {incompleteNeighborhood: 'ignore'}
  });
  const ignored = await ignore.run();
  expect(ignored.mask[0]).toBe(1);
  expect(ignored.mask[5 * width + width - 1]).toBe(1);
  expectMatchesOracle(
    ignored,
    computeTerrainSummits(values, undefined, width, height, parameters, {
      incompleteNeighborhood: 'ignore'
    })
  );
  ignore.destroy();
});

it('GPUTerrainSummits never lets nodata bleed into a summit', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 24;
  const height = 20;
  const values = Float32Array.from({length: width * height}, (_, index) => {
    const column = index % width;
    const row = Math.floor(index / width);
    return 10 - 0.37 * Math.abs(column - 8) - 0.51 * Math.abs(row - 8);
  });
  const validity = new Uint32Array(width * height).fill(1);
  // Invalid cells with enormous sentinels beside the summit and elsewhere.
  for (const [column, row, sentinel] of [
    [9, 8, 1e30],
    [8, 9, -32768],
    [18, 14, 1e30]
  ] as const) {
    values[row * width + column] = sentinel;
    validity[row * width + column] = 0;
  }
  const parameters = getGPUTerrainSummitsParameterValues(BASE);
  const summit = 8 * width + 8;

  const reject = createSummitsFixture(device, {values, validity, width, height, settings: BASE});
  const rejected = await reject.run();
  expect(rejected.mask[summit]).toBe(0);
  expect(rejected.mask[9 * width + 8]).toBe(0);
  expectMatchesOracle(rejected, computeTerrainSummits(values, validity, width, height, parameters));
  reject.destroy();

  const ignore = createSummitsFixture(device, {
    values,
    validity,
    width,
    height,
    settings: BASE,
    contributor: {incompleteNeighborhood: 'ignore'}
  });
  const ignored = await ignore.run();
  expect(ignored.mask[summit]).toBe(1);
  expect(ignored.mask[9 * width + 8]).toBe(0);
  expect(ignored.drop.every(value => !(value > 1000 || value < -1000))).toBe(true);
  expectMatchesOracle(
    ignored,
    computeTerrainSummits(values, validity, width, height, parameters, {
      incompleteNeighborhood: 'ignore'
    })
  );
  ignore.destroy();
});

it('GPUTerrainSummits fills compact output in ascending order and reports capacity overflow', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 37;
  const height = 29;
  const values = createNoise(width, height, 1000, 11);
  const expected = computeTerrainSummits(
    values,
    undefined,
    width,
    height,
    getGPUTerrainSummitsParameterValues(BASE)
  );
  expect(expected.ids.length).toBeGreaterThan(6);
  expect([...expected.ids].sort((a, b) => a - b)).toEqual(expected.ids);

  const small = createSummitsFixture(device, {values, width, height, settings: BASE, capacity: 4});
  const clipped = await small.run();
  expect(clipped.count).toBe(4);
  expect(clipped.listOverflow).toBe(1);
  expect(clipped.total).toBe(expected.ids.length);
  expect(clipped.ids).toEqual(expected.ids.slice(0, 4));
  expect(clipped.outputDrop.slice(0, 4)).toEqual(
    expected.ids.slice(0, 4).map(id => expected.drop[id])
  );
  expect(clipped.mask).toEqual(expected.mask);
  small.destroy();

  const large = createSummitsFixture(device, {
    values,
    width,
    height,
    settings: BASE,
    capacity: expected.ids.length + 5
  });
  const full = await large.run();
  expect(full.listOverflow).toBe(0);
  expect(full.ids).toEqual(expected.ids);
  expect(full.outputDrop.slice(expected.ids.length).every(Number.isNaN)).toBe(true);
  large.destroy();
});

it('GPUTerrainSummits clamps the radius to maximumRadiusPixels and raises overflow', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 40;
  const height = 30;
  const values = createNoise(width, height, 1000, 5);
  const fixture = createSummitsFixture(device, {
    values,
    width,
    height,
    settings: {radius: 3, cellSize: [1, 1]},
    overflow: true,
    contributor: {maximumRadiusPixels: 4}
  });
  const within = await fixture.run();
  expect(within.radiusOverflow).toBe(0);
  const options = {maximumRadiusPixels: 4};
  expectMatchesOracle(
    within,
    computeTerrainSummits(
      values,
      undefined,
      width,
      height,
      getGPUTerrainSummitsParameterValues({radius: 3, cellSize: [1, 1]}),
      options
    )
  );
  const clamped = await fixture.run({radius: 7, cellSize: [1, 1]});
  expect(clamped.radiusOverflow).toBe(1);
  // Clamped to radius 4: identical to an in-range request of 4.
  const reference = computeTerrainSummits(
    values,
    undefined,
    width,
    height,
    getGPUTerrainSummitsParameterValues({radius: 4, cellSize: [1, 1]}),
    options
  );
  expectMatchesOracle(clamped, {...reference, overflow: 1});
  expect(reference.ids.length).toBeGreaterThan(0);
  const back = await fixture.run({radius: 4, cellSize: [1, 1]});
  expect(back.radiusOverflow).toBe(0);
  fixture.destroy();
});

it('GPUTerrainSummits follows per-row cell sizes in web-mercator and geographic modes', async () => {
  const device = await getWebGPUTestDevice();
  if (!device || isSoftwareDevice(device)) {
    return;
  }
  const width = 40;
  const height = 40;
  const values = createNoise(width, height, 100000, 21);
  const cases = [
    {
      cellSizeMode: 'web-mercator' as const,
      settings: {radius: 70, cellSize: [20, 20], northEdge: 0.3, southEdge: 0.45} as const
    },
    {
      cellSizeMode: 'geographic' as const,
      settings: {radius: 90, cellSize: [0.0003, 0.0003], northEdge: 70, southEdge: 50} as const
    }
  ];
  for (const {cellSizeMode, settings} of cases) {
    const parameters = getGPUTerrainSummitsParameterValues(settings);
    const run = (override: Partial<GPUTerrainSummitsSettings> = {}) =>
      computeTerrainSummits(
        values,
        undefined,
        width,
        height,
        getGPUTerrainSummitsParameterValues({...settings, ...override}),
        {cellSizeMode, incompleteNeighborhood: 'ignore'}
      );
    const expected = run();
    // Away from disc boundaries: perturbing the radius by 0.01 % changes nothing.
    expect(run({radius: settings.radius * 1.0001}).ids).toEqual(expected.ids);
    expect(run({radius: settings.radius * 0.9999}).ids).toEqual(expected.ids);
    const uniform = computeTerrainSummits(values, undefined, width, height, parameters, {
      incompleteNeighborhood: 'ignore'
    });
    expect(expected.ids.length).toBeGreaterThan(5);
    expect(expected.ids).not.toEqual(uniform.ids);
    const fixture = createSummitsFixture(device, {
      values,
      width,
      height,
      settings,
      contributor: {cellSizeMode, incompleteNeighborhood: 'ignore'}
    });
    const result = await fixture.run();
    expect(result.mask).toEqual(expected.mask);
    expect(result.ids).toEqual(expected.ids);
    // Drops are float32 differences of exact integers.
    expect(result.drop).toEqual(expected.drop);
    fixture.destroy();
  }
});

it('getGPUTerrainSummitsParameterValues packs and validates settings', () => {
  expect(
    Array.from(
      getGPUTerrainSummitsParameterValues({
        radius: 4,
        minimumDrop: 2,
        cellSize: [3, 5],
        northEdge: 0.25,
        southEdge: 0.5
      })
    )
  ).toEqual([4, 2, 3, 5, 0.25, 0.5, 0, 0]);
  expect(() => getGPUTerrainSummitsParameterValues({radius: 0, cellSize: [1, 1]})).toThrow(
    /radius/
  );
  expect(() =>
    getGPUTerrainSummitsParameterValues({radius: 1, minimumDrop: -1, cellSize: [1, 1]})
  ).toThrow(/minimumDrop/);
  expect(() => getGPUTerrainSummitsParameterValues({radius: 1, cellSize: [1, 0]})).toThrow(
    /cell size/
  );
});
