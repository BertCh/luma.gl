// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  getGPUCostDistanceParameterValues,
  GPUCostDistance,
  GPUCostDistancePath,
  GPU_COST_DISTANCE_NONE,
  type GPUCostDistanceSettings
} from '../../../src/gpu-raster/cost-distance';
import type {GPUTerrainCellSizeMode} from '../../../src/gpu-terrain/terrain-analysis';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  checkBackLinks,
  checkBackLinkWalks,
  computeBandCounts,
  computeBands,
  computeCostDistance,
  createHoleFriction,
  createRandom,
  createRandomFriction,
  createUniformFriction,
  createWallFriction,
  type CostDistanceOracleOptions
} from './cost-distance-oracle';

const NONE = GPU_COST_DISTANCE_NONE;

type BufferName =
  | 'friction'
  | 'costs'
  | 'converged'
  | 'iterationCount'
  | 'mask'
  | 'backLinks'
  | 'bands'
  | 'bandCounts'
  | 'ids'
  | 'count'
  | 'overflow'
  | 'totalCount';

type FixtureOptions = {
  width: number;
  height: number;
  friction: Float32Array;
  noDataValue?: number;
  settings?: GPUCostDistanceSettings;
  cellSizeMode?: GPUTerrainCellSizeMode;
  sources?: number[];
  sourceCosts?: number[];
  sourceCount?: number;
  mask?: Uint32Array;
  thresholds?: number[];
  maxIterations?: number;
  maxTieIterations?: number;
  backLinks?: boolean;
  path?: {target: number; capacity: number};
};

type Fixture = {
  graph: GPUCommandGraph;
  options: FixtureOptions;
  cellCount: number;
  frictionBuffer: Buffer;
  settings: GPUParameterBuffer<'float32'>;
  sources?: GPUParameterBuffer<'uint32'>;
  sourceCosts?: GPUParameterBuffer<'float32'>;
  sourceCount?: GPUParameterBuffer<'uint32'>;
  target?: GPUParameterBuffer<'uint32'>;
  thresholds?: GPUParameterBuffer<'float32'>;
  /** Buffers that the options did not request are absent at runtime. */
  buffers: Record<BufferName, Buffer>;
  parameters: GPUParameterBuffer[];
};

function createFixture(device: Device, options: FixtureOptions): Fixture {
  const {width, height} = options;
  const cellCount = width * height;
  const graph = new GPUCommandGraph(device, {id: 'cost-distance-test'});
  const parameters: GPUParameterBuffer[] = [];
  const createParameter = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    values: number[]
  ) => {
    const parameter = new GPUParameterBuffer(device, {
      id: `cd-${name}`,
      format,
      length: values.length,
      values: format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values)
    });
    parameters.push(parameter);
    return parameter;
  };
  const frictionBuffer = createInputBuffer(device, options.friction);
  const buffers = {
    costs: createOutputBuffer(device, cellCount),
    converged: createOutputBuffer(device, 1),
    iterationCount: createOutputBuffer(device, 1)
  } as Record<BufferName, Buffer>;
  const settings = createParameter(
    'settings',
    'float32',
    Array.from(
      getGPUCostDistanceParameterValues({
        cellSize: [1, 1],
        ...options.settings
      } as never)
    )
  ) as GPUParameterBuffer<'float32'>;
  const sources = options.sources
    ? createParameter('sources', 'uint32', options.sources)
    : undefined;
  const sourceCosts = options.sourceCosts
    ? createParameter('source-costs', 'float32', options.sourceCosts)
    : undefined;
  const sourceCount =
    options.sourceCount !== undefined
      ? createParameter('source-count', 'uint32', [options.sourceCount])
      : undefined;
  const thresholds = options.thresholds
    ? createParameter('thresholds', 'float32', options.thresholds)
    : undefined;
  const target = options.path
    ? createParameter('target', 'uint32', [options.path.target])
    : undefined;
  let maskBuffer: Buffer | undefined;
  if (options.mask) {
    maskBuffer = createInputBuffer(device, options.mask);
    buffers.mask = maskBuffer;
  }
  if (options.backLinks || options.path) buffers.backLinks = createOutputBuffer(device, cellCount);
  if (thresholds) {
    buffers.bands = createOutputBuffer(device, cellCount);
    buffers.bandCounts = createOutputBuffer(device, options.thresholds!.length);
  }
  const views = new Map<string, unknown>();
  const view = <Format extends 'uint32' | 'float32'>(
    name: BufferName,
    format: Format,
    length: number
  ) => {
    if (!views.has(name)) {
      views.set(name, importGraphBuffer(graph, `cd-${name}`, buffers[name], format, length));
    }
    return views.get(name) as ReturnType<typeof importGraphBuffer<Format, void>>;
  };
  graph.add(
    new GPUCostDistance({
      id: 'cd',
      width,
      height,
      friction: {
        id: 'friction',
        format: 'float32',
        noDataValue: options.noDataValue,
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'cd-friction', frictionBuffer, 'float32', cellCount)
        }
      },
      settings: settings.importToGraph(graph),
      cellSizeMode: options.cellSizeMode,
      sources: sources?.importToGraph(graph),
      sourceCosts: sourceCosts?.importToGraph(graph),
      sourceCount: sourceCount?.importToGraph(graph),
      sourceMask: maskBuffer ? view('mask', 'uint32', cellCount) : undefined,
      maxIterations: options.maxIterations ?? 64,
      maxTieIterations: options.maxTieIterations,
      costs: view('costs', 'float32', cellCount),
      backLinks: buffers.backLinks ? view('backLinks', 'uint32', cellCount) : undefined,
      bandThresholds: thresholds?.importToGraph(graph),
      bands: thresholds ? view('bands', 'uint32', cellCount) : undefined,
      bandCounts: thresholds ? view('bandCounts', 'uint32', options.thresholds!.length) : undefined,
      converged: view('converged', 'uint32', 1),
      iterationCount: view('iterationCount', 'uint32', 1)
    })
  );
  if (options.path) {
    buffers.ids = createOutputBuffer(device, options.path.capacity);
    buffers.count = createOutputBuffer(device, 1);
    buffers.overflow = createOutputBuffer(device, 1);
    buffers.totalCount = createOutputBuffer(device, 1);
    graph.add(
      new GPUCostDistancePath({
        id: 'cdp',
        width,
        height,
        backLinks: view('backLinks', 'uint32', cellCount),
        target: target!.importToGraph(graph),
        output: {
          ids: view('ids', 'uint32', options.path.capacity),
          count: view('count', 'uint32', 1),
          overflow: view('overflow', 'uint32', 1),
          totalCount: view('totalCount', 'uint32', 1)
        }
      })
    );
  }
  return {
    graph,
    options,
    cellCount,
    frictionBuffer,
    settings,
    sources,
    sourceCosts,
    sourceCount,
    target,
    thresholds,
    buffers: {...buffers, friction: frictionBuffer},
    parameters
  };
}

function destroyFixture(fixture: Fixture): void {
  for (const parameter of fixture.parameters) parameter.destroy();
  for (const buffer of Object.values(fixture.buffers)) buffer.destroy();
}

/** Returns the oracle options for the fixture's current CPU-side description. */
function getOracleOptions(
  fixture: Fixture,
  sources: {cell: number; cost: number}[],
  settings: GPUCostDistanceSettings = {
    cellSize: [1, 1],
    ...fixture.options.settings
  } as never,
  friction: Float32Array = fixture.options.friction
): CostDistanceOracleOptions {
  const {noDataValue} = fixture.options;
  return {
    width: fixture.options.width,
    height: fixture.options.height,
    friction: Float64Array.from(friction, value => (value === noDataValue ? NaN : value)),
    cellSizeMode: fixture.options.cellSizeMode,
    settings,
    sources,
    costLimit: settings.costLimit
  };
}

function expectCostsClose(actual: number[], expected: Float64Array, costLimit?: number): void {
  expect(actual.length).toBe(expected.length);
  for (const [index, value] of expected.entries()) {
    if (!Number.isFinite(value) || !Number.isFinite(actual[index])) {
      if (Number.isFinite(value) !== Number.isFinite(actual[index]) && costLimit !== undefined) {
        // A cell within float rounding of the limit may land on either side.
        const finite = Number.isFinite(value) ? value : actual[index];
        expect(Math.abs(finite - costLimit)).toBeLessThan(1e-3 * costLimit);
        continue;
      }
      expect(actual[index]).toBe(value);
      continue;
    }
    expect(Math.abs(actual[index] - value)).toBeLessThanOrEqual(Math.max(1e-4 * value, 1e-5));
  }
}

async function run(device: Device, fixture: Fixture) {
  const compiled = fixture.graph.compile();
  submitGraph(device, compiled, undefined);
  return compiled;
}

async function readCosts(fixture: Fixture): Promise<number[]> {
  return readFloat32(fixture.buffers.costs, fixture.cellCount);
}

async function readIterations(fixture: Fixture) {
  const [[converged], [iterationCount]] = await Promise.all([
    readUint32(fixture.buffers.converged, 1),
    readUint32(fixture.buffers.iterationCount, 1)
  ]);
  return {converged, iterationCount};
}

it('GPUCostDistance spreads octile distances symmetrically from a center source', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const size = 31;
  const center = 15 * size + 15;
  const fixture = createFixture(device, {
    width: size,
    height: size,
    friction: createUniformFriction(size, size, 1),
    settings: {cellSize: [2, 2]},
    sources: [center],
    backLinks: true
  });
  const compiled = await run(device, fixture);
  const costs = await readCosts(fixture);
  const oracle = computeCostDistance(getOracleOptions(fixture, [{cell: center, cost: 0}]));
  expectCostsClose(costs, oracle);
  for (const [dx, dy] of [
    [3, 5],
    [7, 0],
    [12, 12]
  ]) {
    const values = [
      [dx, dy],
      [-dx, dy],
      [dx, -dy],
      [-dx, -dy],
      [dy, dx],
      [-dy, dx],
      [dy, -dx],
      [-dy, -dx]
    ].map(([x, y]) => costs[(15 + y) * size + 15 + x]);
    for (const value of values) expect(Math.abs(value - values[0])).toBeLessThan(1e-4 * values[0]);
  }
  expect(costs[center]).toBe(0);
  expect(costs[center + 1]).toBeCloseTo(2, 5);
  expect(costs[center + size + 1]).toBeCloseTo(2 * Math.SQRT2, 5);
  const backLinks = await readUint32(fixture.buffers.backLinks, fixture.cellCount);
  expect(checkBackLinks(getOracleOptions(fixture, []), costs, backLinks)).toBeUndefined();
  expect(backLinks[center]).toBe(0);
  const {converged, iterationCount} = await readIterations(fixture);
  expect(converged).toBe(1);
  console.log(`cost-distance uniform 31x31 iterations: ${iterationCount}`);
  compiled.destroy();
  destroyFixture(fixture);
});

it('GPUCostDistance matches Dijkstra on random friction with listed sources, costs, and count', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const width = 97;
  const height = 61;
  const friction = createRandomFriction(width, height, 11);
  friction[5] = NaN;
  const sources = [10 * width + 20, 50 * width + 90, 30 * width + 3, 99999, 5, 0];
  const sourceCosts = [0, 3.5, 1, 0, 0, 2];
  const thresholds = [10, 40, 80];
  const fixture = createFixture(device, {
    width,
    height,
    friction,
    settings: {cellSize: [1, 1]},
    sources,
    sourceCosts,
    sourceCount: 5,
    thresholds,
    backLinks: true
  });
  const compiled = await run(device, fixture);
  const seeds = sources.slice(0, 5).map((cell, index) => ({cell, cost: sourceCosts[index]}));
  const oracleOptions = getOracleOptions(fixture, seeds);
  const oracle = computeCostDistance(oracleOptions);
  const costs = await readCosts(fixture);
  expectCostsClose(costs, oracle);
  expect(costs[5]).toBe(Infinity);
  const seedMap = new Map(
    seeds
      .filter(seed => seed.cell < width * height && seed.cell !== 5)
      .map(seed => [seed.cell, seed.cost])
  );
  const backLinks = await readUint32(fixture.buffers.backLinks, width * height);
  expect(checkBackLinks(oracleOptions, costs, backLinks, seedMap)).toBeUndefined();
  expect(backLinks[5]).toBe(NONE);
  const bands = await readUint32(fixture.buffers.bands, width * height);
  expect(bands).toEqual(Array.from(computeBands(Float32Array.from(costs), thresholds)));
  expect(await readUint32(fixture.buffers.bandCounts, 3)).toEqual(
    computeBandCounts(bands, thresholds.length)
  );
  const {converged, iterationCount} = await readIterations(fixture);
  expect(converged).toBe(1);
  console.log(`cost-distance random 97x61 iterations: ${iterationCount}`);

  // Per-frame source changes without recompiling.
  fixture.sources!.write(Uint32Array.from([0, 1, 2, 3, 4, 5].map(i => i * width + 7)));
  fixture.sourceCosts!.write(Float32Array.from([0, 0, 0, 1, 2, 0]));
  fixture.sourceCount!.write(Uint32Array.from([4]));
  submitGraph(device, compiled, undefined);
  const nextSeeds = [0, 1, 2, 3].map((i, index) => ({
    cell: i * width + 7,
    cost: [0, 0, 0, 1][index]
  }));
  expectCostsClose(
    await readCosts(fixture),
    computeCostDistance(getOracleOptions(fixture, nextSeeds))
  );
  fixture.sourceCount!.write(Uint32Array.from([0]));
  submitGraph(device, compiled, undefined);
  expect((await readCosts(fixture)).every(value => value === Infinity)).toBe(true);
  expect(await readIterations(fixture)).toMatchObject({converged: 1});
  compiled.destroy();
  destroyFixture(fixture);
});

it('GPUCostDistance seeds from a source mask', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const width = 50;
  const height = 40;
  const friction = createRandomFriction(width, height, 5);
  const random = createRandom(9);
  const mask = Uint32Array.from({length: width * height}, () => (random() < 0.005 ? 1 : 0));
  mask[0] = 1;
  const fixture = createFixture(device, {
    width,
    height,
    friction,
    mask,
    backLinks: true
  });
  const compiled = await run(device, fixture);
  const seeds = Array.from(mask, (value, cell) => ({cell, cost: 0, value}))
    .filter(entry => entry.value)
    .map(({cell, cost}) => ({cell, cost}));
  const options = getOracleOptions(fixture, seeds);
  const costs = await readCosts(fixture);
  expectCostsClose(costs, computeCostDistance(options));
  const backLinks = await readUint32(fixture.buffers.backLinks, width * height);
  expect(
    checkBackLinks(options, costs, backLinks, new Map(seeds.map(seed => [seed.cell, 0])))
  ).toBeUndefined();
  compiled.destroy();
  destroyFixture(fixture);
});

it('GPUCostDistance routes around a barrier wall through its gap', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const width = 41;
  const height = 21;
  const friction = createWallFriction(width, height, 20, [18]);
  const source = 2 * width + 2;
  const fixture = createFixture(device, {
    width,
    height,
    friction,
    sources: [source],
    backLinks: true
  });
  const compiled = await run(device, fixture);
  const options = getOracleOptions(fixture, [{cell: source, cost: 0}]);
  const costs = await readCosts(fixture);
  expectCostsClose(costs, computeCostDistance(options));
  const across = costs[2 * width + 38];
  expect(across).toBeGreaterThan(40);
  expect(costs[2 * width + 20]).toBe(Infinity);
  const backLinks = await readUint32(fixture.buffers.backLinks, width * height);
  expect(checkBackLinks(options, costs, backLinks)).toBeUndefined();
  expect(backLinks[2 * width + 20]).toBe(NONE);
  compiled.destroy();
  destroyFixture(fixture);
});

it('GPUCostDistance treats nodata holes as barriers', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const width = 64;
  const height = 48;
  const friction = createHoleFriction(width, height, 21, -9999, 0.2);
  friction[0] = 3;
  const fixture = createFixture(device, {
    width,
    height,
    friction,
    noDataValue: -9999,
    sources: [0],
    backLinks: true
  });
  const compiled = await run(device, fixture);
  const options = getOracleOptions(fixture, [{cell: 0, cost: 0}]);
  const costs = await readCosts(fixture);
  expectCostsClose(costs, computeCostDistance(options));
  expect(costs.some(value => value === Infinity)).toBe(true);
  const backLinks = await readUint32(fixture.buffers.backLinks, width * height);
  expect(checkBackLinks(options, costs, backLinks)).toBeUndefined();
  compiled.destroy();
  destroyFixture(fixture);
});

it('GPUCostDistance uses row-dependent distances in web-mercator and geographic modes', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const cases: {
    mode: GPUTerrainCellSizeMode;
    settings: GPUCostDistanceSettings;
  }[] = [
    {
      mode: 'web-mercator',
      settings: {cellSize: [100, 100], northEdge: 0.3, southEdge: 0.36}
    },
    {
      mode: 'geographic',
      settings: {cellSize: [0.001, 0.001], northEdge: 60, southEdge: 59.9}
    }
  ];
  for (const {mode, settings} of cases) {
    const width = 40;
    const height = 40;
    const friction = createRandomFriction(width, height, 3);
    const source = 5 * width + 5;
    const fixture = createFixture(device, {
      width,
      height,
      friction,
      settings,
      cellSizeMode: mode,
      sources: [source],
      backLinks: true
    });
    const compiled = await run(device, fixture);
    const options = getOracleOptions(fixture, [{cell: source, cost: 0}], settings);
    const costs = await readCosts(fixture);
    expectCostsClose(costs, computeCostDistance(options));
    const backLinks = await readUint32(fixture.buffers.backLinks, width * height);
    expect(checkBackLinks(options, costs, backLinks)).toBeUndefined();
    compiled.destroy();
    destroyFixture(fixture);
  }
});

it('GPUCostDistance changes cost limit and friction per frame without recompiling', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const width = 60;
  const height = 45;
  const friction = createRandomFriction(width, height, 17);
  const source = 20 * width + 30;
  const fixture = createFixture(device, {
    width,
    height,
    friction,
    sources: [source]
  });
  const compiled = await run(device, fixture);
  const seeds = [{cell: source, cost: 0}];
  expectCostsClose(await readCosts(fixture), computeCostDistance(getOracleOptions(fixture, seeds)));

  const limited: GPUCostDistanceSettings = {
    cellSize: [1, 1],
    costLimit: 23.37
  };
  fixture.settings.write(getGPUCostDistanceParameterValues(limited));
  submitGraph(device, compiled, undefined);
  const limitedCosts = await readCosts(fixture);
  expectCostsClose(
    limitedCosts,
    computeCostDistance(getOracleOptions(fixture, seeds, limited)),
    23.37
  );
  expect(limitedCosts.some(value => value === Infinity)).toBe(true);

  const doubled: GPUCostDistanceSettings = {cellSize: [2, 2]};
  fixture.settings.write(getGPUCostDistanceParameterValues(doubled));
  const changed = createRandomFriction(width, height, 18);
  fixture.frictionBuffer.write(changed);
  submitGraph(device, compiled, undefined);
  expectCostsClose(
    await readCosts(fixture),
    computeCostDistance(getOracleOptions(fixture, seeds, doubled, changed))
  );
  compiled.destroy();
  destroyFixture(fixture);
});

it('GPUCostDistance reports non-convergence when maxIterations is too small', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const width = 200;
  const height = 10;
  const fixture = createFixture(device, {
    width,
    height,
    friction: createUniformFriction(width, height, 1),
    sources: [5 * width],
    maxIterations: 1
  });
  const compiled = await run(device, fixture);
  expect(await readIterations(fixture)).toEqual({
    converged: 0,
    iterationCount: 1
  });
  const costs = await readCosts(fixture);
  expect(costs[5 * width + 1]).toBe(1);
  expect(costs[5 * width + 199]).toBe(Infinity);
  compiled.destroy();

  // Enough iterations on the same grid converge to the oracle.
  const second = createFixture(device, {
    width,
    height,
    friction: createUniformFriction(width, height, 1),
    sources: [5 * width],
    maxIterations: 40
  });
  const secondCompiled = await run(device, second);
  expect((await readIterations(second)).converged).toBe(1);
  expectCostsClose(
    await readCosts(second),
    computeCostDistance(getOracleOptions(second, [{cell: 5 * width, cost: 0}]))
  );
  console.log(`cost-distance 200x10 iterations: ${(await readIterations(second)).iterationCount}`);
  secondCompiled.destroy();
  destroyFixture(fixture);
  destroyFixture(second);
});

it('GPUCostDistance handles a single-cell grid and an empty source count', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const single = createFixture(device, {
    width: 1,
    height: 1,
    friction: Float32Array.of(2),
    sources: [0],
    sourceCosts: [1.5],
    backLinks: true
  });
  const singleCompiled = await run(device, single);
  expect(await readCosts(single)).toEqual([1.5]);
  expect(await readUint32(single.buffers.backLinks, 1)).toEqual([0]);
  expect(await readIterations(single)).toMatchObject({converged: 1});
  singleCompiled.destroy();
  destroyFixture(single);

  const empty = createFixture(device, {
    width: 20,
    height: 20,
    friction: createUniformFriction(20, 20, 1),
    sources: [0, 1],
    sourceCount: 0,
    backLinks: true
  });
  const emptyCompiled = await run(device, empty);
  expect((await readCosts(empty)).every(value => value === Infinity)).toBe(true);
  expect((await readUint32(empty.buffers.backLinks, 400)).every(value => value === NONE)).toBe(
    true
  );
  expect(await readIterations(empty)).toMatchObject({converged: 1});
  emptyCompiled.destroy();
  destroyFixture(empty);
});

it('GPUCostDistance ignores impassable sources and negative source costs', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const friction = createUniformFriction(8, 8, 1);
  friction[0] = NaN;
  const fixture = createFixture(device, {
    width: 8,
    height: 8,
    friction,
    sources: [0, 9, 63],
    sourceCosts: [0, -1, 4]
  });
  const compiled = await run(device, fixture);
  expectCostsClose(
    await readCosts(fixture),
    computeCostDistance(getOracleOptions(fixture, [{cell: 63, cost: 4}]))
  );
  compiled.destroy();
  destroyFixture(fixture);
});

it('GPUCostDistance emits exact back-links and GPUCostDistancePath extracts exact paths', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  // Row 0: . . .   Row 1: . X .   Row 2: . X .   Source at cell 0, path 8 -> 5 -> 1 -> 0.
  const friction = Float32Array.of(1, 1, 1, 1, NaN, 1, 1, NaN, 1);
  const fixture = createFixture(device, {
    width: 3,
    height: 3,
    friction,
    sources: [0],
    path: {target: 8, capacity: 8}
  });
  const compiled = await run(device, fixture);
  expect(await readUint32(fixture.buffers.backLinks, 9)).toEqual([
    0,
    16,
    16,
    64,
    NONE,
    32,
    64,
    NONE,
    64
  ]);
  expect((await readCosts(fixture)).map(value => Number(value.toFixed(4)))).toEqual([
    0,
    1,
    2,
    1,
    Infinity,
    2.4142,
    2,
    Infinity,
    3.4142
  ]);
  const readPath = async () => {
    const [count] = await readUint32(fixture.buffers.count, 1);
    return {
      ids: await readUint32(fixture.buffers.ids, count),
      count,
      overflow: (await readUint32(fixture.buffers.overflow, 1))[0],
      total: (await readUint32(fixture.buffers.totalCount, 1))[0]
    };
  };
  expect(await readPath()).toEqual({
    ids: [8, 5, 1, 0],
    count: 4,
    overflow: 0,
    total: 4
  });

  fixture.target!.write(Uint32Array.of(2));
  submitGraph(device, compiled, undefined);
  expect(await readPath()).toEqual({
    ids: [2, 1, 0],
    count: 3,
    overflow: 0,
    total: 3
  });

  fixture.target!.write(Uint32Array.of(0));
  submitGraph(device, compiled, undefined);
  expect(await readPath()).toEqual({
    ids: [0],
    count: 1,
    overflow: 0,
    total: 1
  });

  fixture.target!.write(Uint32Array.of(4));
  submitGraph(device, compiled, undefined);
  expect(await readPath()).toEqual({
    ids: [],
    count: 0,
    overflow: 0,
    total: 0
  });

  fixture.target!.write(Uint32Array.of(100));
  submitGraph(device, compiled, undefined);
  expect(await readPath()).toEqual({
    ids: [],
    count: 0,
    overflow: 0,
    total: 0
  });
  compiled.destroy();
  destroyFixture(fixture);
});

it('GPUCostDistancePath reports overflow with the exact total when capacity is short', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const friction = Float32Array.of(1, 1, 1, 1, NaN, 1, 1, NaN, 1);
  const fixture = createFixture(device, {
    width: 3,
    height: 3,
    friction,
    sources: [0],
    path: {target: 8, capacity: 2}
  });
  const compiled = await run(device, fixture);
  expect(await readUint32(fixture.buffers.ids, 2)).toEqual([8, 5]);
  expect(await readUint32(fixture.buffers.count, 1)).toEqual([2]);
  expect(await readUint32(fixture.buffers.overflow, 1)).toEqual([1]);
  expect(await readUint32(fixture.buffers.totalCount, 1)).toEqual([4]);
  compiled.destroy();
  destroyFixture(fixture);
});

it('GPUCostDistancePath walks a long path on a random grid and ends at the source', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const width = 97;
  const height = 61;
  const source = 3 * width + 3;
  const target = 57 * width + 93;
  const fixture = createFixture(device, {
    width,
    height,
    friction: createRandomFriction(width, height, 29),
    sources: [source],
    path: {target, capacity: width * height}
  });
  const compiled = await run(device, fixture);
  const [count] = await readUint32(fixture.buffers.count, 1);
  const ids = await readUint32(fixture.buffers.ids, count);
  expect(ids[0]).toBe(target);
  expect(ids[count - 1]).toBe(source);
  expect(count).toBeGreaterThanOrEqual(90);
  expect(new Set(ids).size).toBe(count);
  expect(await readUint32(fixture.buffers.overflow, 1)).toEqual([0]);
  compiled.destroy();
  destroyFixture(fixture);
});

/** Runs cost distance with back-links on a friction raster and checks costs and back-link walks. */
async function expectTieSafeBackLinks(
  device: Device,
  options: {
    width: number;
    height: number;
    friction: Float32Array;
    sources: number[];
    sourceCosts?: number[];
    maxIterations?: number;
    maxTieIterations?: number;
    path?: {target: number; capacity: number};
  }
) {
  const {width, height} = options;
  const fixture = createFixture(device, {
    ...options,
    maxIterations: options.maxIterations ?? 16,
    backLinks: true
  });
  const compiled = await run(device, fixture);
  const seeds = options.sources.map((cell, index) => ({
    cell,
    cost: options.sourceCosts?.[index] ?? 0
  }));
  const oracleOptions = getOracleOptions(fixture, seeds);
  const costs = await readCosts(fixture);
  expectCostsClose(costs, computeCostDistance(oracleOptions));
  const backLinks = await readUint32(fixture.buffers.backLinks, width * height);
  const seedMap = new Map<number, number>();
  for (const seed of seeds) {
    seedMap.set(seed.cell, Math.min(seed.cost, seedMap.get(seed.cell) ?? Infinity));
  }
  expect(checkBackLinkWalks(oracleOptions, costs, backLinks, seedMap)).toBeUndefined();
  const iterations = await readIterations(fixture);
  const result = {costs, backLinks, converged: iterations.converged};
  let path: number[] | undefined;
  if (options.path) {
    const [count] = await readUint32(fixture.buffers.count, 1);
    const [total] = await readUint32(fixture.buffers.totalCount, 1);
    path = await readUint32(fixture.buffers.ids, count);
    expect(total).toBe(count);
  }
  compiled.destroy();
  destroyFixture(fixture);
  return {...result, path};
}

it('GPUCostDistance back-links cross zero-friction corridors and plateaus to a source', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const width = 40;
  const height = 24;
  // Friction 3 everywhere, a zero-friction snake corridor, and a zero-friction block.
  const friction = createUniformFriction(width, height, 3);
  for (let column = 2; column < 36; column++) friction[2 * width + column] = 0;
  for (let row = 2; row < 20; row++) friction[row * width + 35] = 0;
  for (let column = 10; column < 35; column++) friction[19 * width + column] = 0;
  for (let row = 8; row < 14; row++) {
    for (let column = 12; column < 22; column++) friction[row * width + column] = 0;
  }
  const source = 2 * width + 2;
  const target = 19 * width + 10;
  const result = await expectTieSafeBackLinks(device, {
    width,
    height,
    friction,
    sources: [source],
    maxIterations: 64,
    path: {target, capacity: width * height}
  });
  expect(result.converged).toBe(1);
  // The whole corridor has cost 0, so the target path crosses plateau cells and reaches the source.
  expect(result.path![0]).toBe(target);
  expect(result.path![result.path!.length - 1]).toBe(source);
  expect(new Set(result.path).size).toBe(result.path!.length);
  // A cell deep in the zero-cost corridor must not claim to be a source.
  expect(result.backLinks[2 * width + 30]).not.toBe(0);
  expect(result.backLinks[source]).toBe(0);
});

it('GPUCostDistance back-links reach a source through equal-cost plateaus and tied sources', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const width = 36;
  const height = 30;
  const friction = createUniformFriction(width, height, 1);
  // Two zero-friction blocks entered across positive friction, reached at equal cost from two sources.
  for (const [row0, column0] of [
    [4, 4],
    [18, 20]
  ]) {
    for (let row = row0; row < row0 + 7; row++) {
      for (let column = column0; column < column0 + 9; column++) {
        friction[row * width + column] = 0;
      }
    }
  }
  const result = await expectTieSafeBackLinks(device, {
    width,
    height,
    friction,
    sources: [0, width * height - 1, 12 * width + 12],
    sourceCosts: [0, 0, 5],
    maxIterations: 64
  });
  expect(result.converged).toBe(1);
});

it('GPUCostDistance back-links stay valid on random zero-heavy friction', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  for (const seed of [3, 11, 23]) {
    const width = 48;
    const height = 36;
    const random = createRandom(seed);
    const friction = Float32Array.from({length: width * height}, () => {
      const draw = random();
      return draw < 0.45 ? 0 : 1 + Math.floor(random() * 4);
    });
    const sources = [5 * width + 5, 30 * width + 40, 17 * width + 17];
    const result = await expectTieSafeBackLinks(device, {
      width,
      height,
      friction,
      sources,
      sourceCosts: [0, 0, 2],
      maxIterations: 64,
      maxTieIterations: 64
    });
    expect(result.converged).toBe(1);
  }
});

it('GPUCostDistance reports non-convergence when maxTieIterations cannot cover a plateau', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) return;
  const width = 96;
  const height = 4;
  const friction = createUniformFriction(width, height, 3);
  for (let column = 0; column < width; column++) friction[column] = 0;
  const fixture = createFixture(device, {
    width,
    height,
    friction,
    sources: [0],
    maxIterations: 64,
    maxTieIterations: 1,
    backLinks: true
  });
  const compiled = await run(device, fixture);
  expect((await readIterations(fixture)).converged).toBe(0);
  compiled.destroy();
  destroyFixture(fixture);
  const covered = await expectTieSafeBackLinks(device, {
    width,
    height,
    friction,
    sources: [0],
    maxIterations: 64,
    maxTieIterations: 8
  });
  expect(covered.converged).toBe(1);
});
