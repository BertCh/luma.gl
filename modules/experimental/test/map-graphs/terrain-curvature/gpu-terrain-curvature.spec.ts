// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUMapGraphParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {
  GPUTerrainCurvature,
  getGPUTerrainCurvatureParameterValues,
  type GPUTerrainCurvatureKind,
  type GPUTerrainCurvatureMethod,
  type GPUTerrainCurvatureProps,
  type GPUTerrainCurvatureSettings
} from '../../../src/map-graphs/terrain-curvature';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../map-graph-test-utils';
import {
  computeCurvaturesFromPartials,
  computeTerrainCurvature,
  ORACLE_CURVATURE_KINDS,
  type TerrainCurvatureOptions
} from './terrain-curvature-oracle';

const WIDTH = 20;
const HEIGHT = 18;
const PIXEL_COUNT = WIDTH * HEIGHT;
const METHODS: GPUTerrainCurvatureMethod[] = ['evans-young', 'zevenbergen-thorne', 'florinsky'];

/** Largest error relative to the largest oracle magnitude per output, accumulated across tests. */
const measuredErrors = {relativeToScale: 0, relativeToValue: 0};

/** Deterministic smooth surface with a little noise, 500 m base height. */
function createSurface(): Float32Array {
  let seed = 12345;
  const noise = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 0x100000000 - 0.5;
  };
  return Float32Array.from({length: PIXEL_COUNT}, (_, index) => {
    const column = index % WIDTH;
    const row = Math.floor(index / WIDTH);
    return (
      500 +
      12 * Math.sin(column * 0.45) +
      9 * Math.cos(row * 0.38 + 0.4) +
      0.35 * column * row * 0.1 +
      0.6 * noise()
    );
  });
}

function expectCurvatureClose(actual: number[], expected: number[], label: string): void {
  expect(actual.length, label).toBe(expected.length);
  const scale = Math.max(1e-12, ...expected.filter(Number.isFinite).map(Math.abs));
  for (const [index, value] of expected.entries()) {
    if (Number.isNaN(value)) {
      expect(Number.isNaN(actual[index]), `${label}[${index}] should be NaN`).toBe(true);
      continue;
    }
    const difference = Math.abs(actual[index] - value);
    measuredErrors.relativeToScale = Math.max(measuredErrors.relativeToScale, difference / scale);
    if (Math.abs(value) > 1e-3 * scale) {
      measuredErrors.relativeToValue = Math.max(
        measuredErrors.relativeToValue,
        difference / Math.abs(value)
      );
    }
    expect(difference, `${label}[${index}] actual ${actual[index]} expected ${value}`).toBeLessThan(
      1e-4 * scale + 1e-4 * Math.abs(value)
    );
  }
}

type Fixture = {
  graph: GPUCommandGraph;
  settings: GPUMapGraphParameterBuffer<'float32'>;
  curvatures: Record<GPUTerrainCurvatureKind, Buffer>;
  ring: Buffer;
  validity: Buffer;
  owned: Buffer[];
};

function createFixture(
  device: Device,
  elevation: Float32Array,
  settingsValues: Float32Array,
  overrides: Partial<GPUTerrainCurvatureProps> = {},
  mask?: Uint32Array
): Fixture {
  const graph = new GPUCommandGraph(device, {id: 'terrain-curvature-test'});
  const elevationBuffer = createInputBuffer(device, elevation);
  const maskBuffer = mask ? createInputBuffer(device, mask) : undefined;
  const curvatures = Object.fromEntries(
    ORACLE_CURVATURE_KINDS.map(kind => [kind, createOutputBuffer(device, PIXEL_COUNT)])
  ) as Record<GPUTerrainCurvatureKind, Buffer>;
  const ring = createOutputBuffer(device, PIXEL_COUNT);
  const validity = createOutputBuffer(device, PIXEL_COUNT);
  const settings = new GPUMapGraphParameterBuffer(device, {
    id: 'curvature-settings',
    format: 'float32',
    length: 12,
    values: settingsValues
  });
  graph.add(
    new GPUTerrainCurvature({
      width: WIDTH,
      height: HEIGHT,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', PIXEL_COUNT)
        },
        validity: maskBuffer
          ? importGraphBuffer(graph, 'mask', maskBuffer, 'uint32', PIXEL_COUNT)
          : undefined
      },
      settings: settings.importToGraph(graph),
      curvatures: Object.fromEntries(
        ORACLE_CURVATURE_KINDS.map(kind => [
          kind,
          importGraphBuffer(graph, `curvature-${kind}`, curvatures[kind], 'float32', PIXEL_COUNT)
        ])
      ),
      ringCurvature: importGraphBuffer(graph, 'ring-output', ring, 'float32', PIXEL_COUNT),
      validity: importGraphBuffer(graph, 'validity', validity, 'uint32', PIXEL_COUNT),
      ...overrides
    })
  );
  return {
    graph,
    settings,
    curvatures,
    ring,
    validity,
    owned: [
      elevationBuffer,
      ...(maskBuffer ? [maskBuffer] : []),
      ...Object.values(curvatures),
      ring,
      validity
    ]
  };
}

function destroyFixture(fixture: Fixture): void {
  fixture.settings.destroy();
  for (const buffer of fixture.owned) buffer.destroy();
}

async function expectMatchesOracle(
  fixture: Fixture,
  elevation: Float32Array,
  settingsValues: Float32Array,
  options: TerrainCurvatureOptions,
  mask?: Uint32Array,
  label: string = ''
): Promise<void> {
  const oracle = computeTerrainCurvature(elevation, mask, WIDTH, HEIGHT, settingsValues, options);
  for (const kind of ORACLE_CURVATURE_KINDS) {
    const actual = await readFloat32(fixture.curvatures[kind], PIXEL_COUNT);
    expectCurvatureClose(actual, oracle.curvatures[kind], `${label} ${kind}`);
    // A failed WGSL compile reads back zeros: some outputs must be non-zero.
    if (!['accumulation'].includes(kind)) {
      expect(
        actual.some(value => Number.isFinite(value) && value !== 0),
        `${label} ${kind} is all zero`
      ).toBe(true);
    }
  }
  expectCurvatureClose(await readFloat32(fixture.ring, PIXEL_COUNT), oracle.ring, `${label} ring`);
  const expectedValidity = oracle.windowValidity.map(
    (value, index) => value & oracle.ringValidity[index]
  );
  const validity = await readUint32(fixture.validity, PIXEL_COUNT);
  expect(validity).toEqual(expectedValidity);
  expect(validity.includes(0) || !mask).toBe(true);
  expect(validity.includes(1), `${label} has valid pixels`).toBe(true);
}

it('GPUTerrainCurvature matches the float64 oracle for every kind, method, and border mode', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const elevation = createSurface();
  const settingsValues = getGPUTerrainCurvatureParameterValues({
    cellSize: [10, 12],
    zFactor: 1.5,
    ringGains: [0.55, 0.45, 0.2, 0.1]
  });
  for (const method of METHODS) {
    for (const borderMode of ['clamp', 'nodata'] as const) {
      const options: TerrainCurvatureOptions = {
        method,
        borderMode,
        ringRadii: [2, 5, 7],
        ringSquash: borderMode === 'clamp' ? 'none' : 'pade-tanh'
      };
      const fixture = createFixture(device, elevation, settingsValues, {
        method,
        borderMode,
        ringRadii: [2, 5, 7],
        ringSquash: options.ringSquash
      });
      const compiled = fixture.graph.compile();
      submitGraph(device, compiled, undefined);
      await expectMatchesOracle(
        fixture,
        elevation,
        settingsValues,
        options,
        undefined,
        `${method}/${borderMode}`
      );
      const validity = await readUint32(fixture.validity, PIXEL_COUNT);
      // Rings of radius 7 always leave the raster near the border.
      expect(validity[0]).toBe(0);
      expect(validity[9 * WIDTH + 10]).toBe(1);
      compiled.destroy();
      destroyFixture(fixture);
    }
  }
});

it('GPUTerrainCurvature reproduces exact quadratics and has convex hills positive', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // z = 1000 + a x^2 + b y^2 + c x y + d x + e y at cell centres in metres, x east, y north.
  const cell = 8;
  const a = -1 / 400;
  const b = -1 / 300;
  const c = 1 / 1000;
  const d = 0.05;
  const e = -0.03;
  const centreColumn = 9;
  const centreRow = 8;
  const elevation = Float32Array.from({length: PIXEL_COUNT}, (_, index) => {
    const x = ((index % WIDTH) - centreColumn) * cell;
    const y = (centreRow - Math.floor(index / WIDTH)) * cell;
    return 1000 + a * x * x + b * y * y + c * x * y + d * x + e * y;
  });
  const settingsValues = getGPUTerrainCurvatureParameterValues({cellSize: [cell, cell]});
  for (const method of METHODS) {
    const fixture = createFixture(device, elevation, settingsValues, {method});
    const compiled = fixture.graph.compile();
    submitGraph(device, compiled, undefined);
    for (const index of [8 * WIDTH + 9, 5 * WIDTH + 12, 11 * WIDTH + 6]) {
      const x = ((index % WIDTH) - centreColumn) * cell;
      const y = (centreRow - Math.floor(index / WIDTH)) * cell;
      const expected = computeCurvaturesFromPartials({
        p: 2 * a * x + c * y + d,
        q: 2 * b * y + c * x + e,
        r: 2 * a,
        s: c,
        t: 2 * b
      });
      for (const kind of ORACLE_CURVATURE_KINDS) {
        const actual = (await readFloat32(fixture.curvatures[kind], PIXEL_COUNT))[index];
        // Elevation is rounded to float32 at 1000 m (6e-5), so allow ~1e-6 absolute.
        expect(
          Math.abs(actual - expected[kind]),
          `${method} ${kind} at ${index}: ${actual} vs ${expected[kind]}`
        ).toBeLessThan(3e-6 + 2e-3 * Math.abs(expected[kind]));
      }
    }
    // Convex hill: positive profile, plan, tangential, mean, maximal-minimal ordering.
    const profile = (await readFloat32(fixture.curvatures.profile, PIXEL_COUNT))[5 * WIDTH + 12];
    const plan = (await readFloat32(fixture.curvatures.plan, PIXEL_COUNT))[5 * WIDTH + 12];
    const tangential = (await readFloat32(fixture.curvatures.tangential, PIXEL_COUNT))[
      5 * WIDTH + 12
    ];
    const mean = (await readFloat32(fixture.curvatures.mean, PIXEL_COUNT))[5 * WIDTH + 12];
    expect(profile).toBeGreaterThan(0);
    expect(plan).toBeGreaterThan(0);
    expect(tangential).toBeGreaterThan(0);
    expect(mean).toBeGreaterThan(0);
    const gaussian = (await readFloat32(fixture.curvatures.gaussian, PIXEL_COUNT))[5 * WIDTH + 12];
    expect(gaussian).toBeGreaterThan(0);
    compiled.destroy();
    destroyFixture(fixture);
  }
});

it('GPUTerrainCurvature gives zero curvature on planes and guards flat gradients', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const settingsValues = getGPUTerrainCurvatureParameterValues({cellSize: [10, 10]});
  const plane = Float32Array.from(
    {length: PIXEL_COUNT},
    (_, index) => 2000 + 2 * (index % WIDTH) + 3 * Math.floor(index / WIDTH)
  );
  for (const method of METHODS) {
    const fixture = createFixture(device, plane, settingsValues, {method});
    const compiled = fixture.graph.compile();
    submitGraph(device, compiled, undefined);
    const interior = 8 * WIDTH + 9;
    for (const kind of ORACLE_CURVATURE_KINDS) {
      const value = (await readFloat32(fixture.curvatures[kind], PIXEL_COUNT))[interior];
      expect(Math.abs(value), `${method} plane ${kind}`).toBeLessThan(1e-6);
    }
    compiled.destroy();
    destroyFixture(fixture);
  }

  const flat = new Float32Array(PIXEL_COUNT).fill(100);
  // A paraboloid apex: gradient is exactly 0 so direction-dependent curvatures are 0 but mean is not.
  const bowl = Float32Array.from({length: PIXEL_COUNT}, (_, index) => {
    const x = (index % WIDTH) - 9;
    const y = Math.floor(index / WIDTH) - 8;
    return 100 - 0.5 * (x * x + y * y);
  });
  const fixtures = [
    createFixture(device, flat, settingsValues),
    createFixture(device, bowl, settingsValues)
  ];
  const compiledFixtures = fixtures.map(fixture => fixture.graph.compile());
  compiledFixtures.forEach(compiled => submitGraph(device, compiled, undefined));
  const apex = 8 * WIDTH + 9;
  const direction = ORACLE_CURVATURE_KINDS.filter(
    kind => !['mean', 'gaussian', 'minimal', 'maximal', 'unsphericity', 'laplacian'].includes(kind)
  );
  for (const fixture of fixtures) {
    for (const kind of direction) {
      expect((await readFloat32(fixture.curvatures[kind], PIXEL_COUNT))[apex]).toBe(0);
    }
    expect((await readUint32(fixture.validity, PIXEL_COUNT))[apex]).toBe(1);
  }
  const bowlMean = (await readFloat32(fixtures[1].curvatures.mean, PIXEL_COUNT))[apex];
  const bowlLaplacian = (await readFloat32(fixtures[1].curvatures.laplacian, PIXEL_COUNT))[apex];
  // z = 100 - (x^2 + y^2) / 2: r = t = -0.01 per m^2 (10 m cells): mean = +0.01 (convex), laplacian = -0.02.
  expect(bowlMean).toBeCloseTo(0.01, 5);
  expect(bowlLaplacian).toBeCloseTo(-0.02, 5);
  expect(
    (await readFloat32(fixtures[0].curvatures.mean, PIXEL_COUNT)).every(value => value === 0)
  ).toBe(true);
  compiledFixtures.forEach(compiled => compiled.destroy());
  fixtures.forEach(destroyFixture);
});

it('GPUTerrainCurvature handles nodata, row direction, and cell size models', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const elevation = createSurface();
  const mask = new Uint32Array(PIXEL_COUNT).fill(1);
  mask[8 * WIDTH + 9] = 0;
  mask[3 * WIDTH + 14] = 0;
  const cases: {
    settings: GPUTerrainCurvatureSettings;
    options: TerrainCurvatureOptions;
    mask?: Uint32Array;
  }[] = [
    {settings: {cellSize: [10, 10]}, options: {method: 'florinsky'}, mask},
    {settings: {cellSize: [10, 10]}, options: {method: 'evans-young', borderMode: 'nodata'}, mask},
    {
      settings: {cellSize: [10, 10], zFactor: 0.5},
      options: {method: 'zevenbergen-thorne', rowDirection: 'north'}
    },
    {
      settings: {cellSize: [20, 20], northEdge: 0.25, southEdge: 0.375},
      options: {method: 'evans-young', cellSizeMode: 'web-mercator', rowDirection: 'north'}
    },
    {
      settings: {cellSize: [0.0002, 0.0002], northEdge: 60, southEdge: 59.9},
      options: {method: 'florinsky', cellSizeMode: 'geographic'}
    }
  ];
  for (const {settings, options, mask: caseMask} of cases) {
    const settingsValues = getGPUTerrainCurvatureParameterValues(settings);
    const fixture = createFixture(
      device,
      elevation,
      settingsValues,
      {
        method: options.method,
        borderMode: options.borderMode,
        rowDirection: options.rowDirection,
        cellSizeMode: options.cellSizeMode,
        ringRadii: [2, 5]
      },
      caseMask
    );
    const compiled = fixture.graph.compile();
    submitGraph(device, compiled, undefined);
    await expectMatchesOracle(
      fixture,
      elevation,
      settingsValues,
      {...options, ringRadii: [2, 5]},
      caseMask,
      JSON.stringify(options)
    );
    if (caseMask) {
      const validity = await readUint32(fixture.validity, PIXEL_COUNT);
      expect(validity[8 * WIDTH + 9]).toBe(0);
      expect(validity[9 * WIDTH + 10]).toBe(0);
      const mean = await readFloat32(fixture.curvatures.mean, PIXEL_COUNT);
      expect(Number.isNaN(mean[8 * WIDTH + 9])).toBe(true);
    }
    compiled.destroy();
    destroyFixture(fixture);
  }
});

it('GPUTerrainCurvature rewrites settings without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const elevation = createSurface();
  const first = getGPUTerrainCurvatureParameterValues({cellSize: [10, 10]});
  const fixture = createFixture(device, elevation, first, {method: 'florinsky'});
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  await expectMatchesOracle(fixture, elevation, first, {method: 'florinsky'}, undefined, 'first');
  const before = await readFloat32(fixture.curvatures.mean, PIXEL_COUNT);

  const second = getGPUTerrainCurvatureParameterValues({
    cellSize: [25, 15],
    zFactor: 2,
    flatGradient: 0.01,
    ringGains: [0.2, 0.9]
  });
  fixture.settings.write(second);
  submitGraph(device, compiled, undefined);
  await expectMatchesOracle(fixture, elevation, second, {method: 'florinsky'}, undefined, 'second');
  const after = await readFloat32(fixture.curvatures.mean, PIXEL_COUNT);
  expect(after[9 * WIDTH + 9]).not.toBe(before[9 * WIDTH + 9]);
  compiled.destroy();
  destroyFixture(fixture);
  // Measured on Metal: about 4e-6 of the per-output scale.
  expect(measuredErrors.relativeToScale).toBeLessThan(1e-4);
});
