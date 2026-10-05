// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Texture, type Buffer, type Device} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it, vi} from 'vitest';
import {GPURasterTextureToBuffer} from '../../../src/gpu-raster';
import {GPUMapGraphParameterBuffer, importGraphBuffer, submitGraph} from '../../../src/map-graphs';
import {
  getGPUDistanceFieldParameterValues,
  GPUDistanceField,
  GPU_DISTANCE_FIELD_NONE,
  type GPUDistanceFieldMode,
  type GPUDistanceFieldSettings
} from '../../../src/map-graphs/distance-field';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../map-graph-test-utils';
import {
  computeDistanceFieldOnCPU,
  createRandom,
  createRandomSeedPositions,
  getCellDistance,
  getUlpDistance,
  type DistanceFieldOracleResult,
  type DistanceFieldScene
} from './distance-field-oracle';

const NONE = GPU_DISTANCE_FIELD_NONE;

type FixtureOptions = {
  width: number;
  height: number;
  settings: GPUDistanceFieldSettings;
  positions?: Float32Array;
  ids?: Uint32Array;
  count?: number;
  mask?: Uint32Array;
  mode?: GPUDistanceFieldMode;
  jumpFloodRefinementPasses?: 0 | 1 | 2;
  texture?: boolean;
};

type FixtureResult = {
  distances: number[];
  allocation: number[];
  nearestCells: number[];
  withinDistance: number[];
  texture?: number[];
};

/** One compiled distance-field graph with rewritable seeds, count, mask, and settings. */
class Fixture {
  readonly options: FixtureOptions;
  readonly cellCount: number;
  readonly recipe: GPUDistanceField;
  readonly compiled: CompiledGPUCommandGraph;
  readonly settings: GPUMapGraphParameterBuffer<'float32'>;
  readonly positions?: GPUMapGraphParameterBuffer<'float32'>;
  readonly ids?: GPUMapGraphParameterBuffer<'uint32'>;
  readonly count?: GPUMapGraphParameterBuffer<'uint32'>;
  readonly mask?: Buffer;
  readonly outputs: Record<'distances' | 'allocation' | 'nearestCells' | 'withinDistance', Buffer>;
  readonly textureBuffer?: Buffer;
  readonly resources: {destroy(): void}[] = [];
  readonly getCommandNodes: ReturnType<typeof vi.spyOn>;
  scene: DistanceFieldScene;

  constructor(
    readonly device: Device,
    options: FixtureOptions
  ) {
    this.options = options;
    const {width, height} = options;
    this.cellCount = width * height;
    const graph = new GPUCommandGraph(device, {id: 'distance-field-test'});
    const settingsValues = getGPUDistanceFieldParameterValues(options.settings);
    this.settings = this.track(
      new GPUMapGraphParameterBuffer(device, {
        id: 'df-settings',
        format: 'float32',
        length: 8,
        values: settingsValues
      })
    );
    if (options.positions) {
      this.positions = this.track(
        new GPUMapGraphParameterBuffer(device, {
          id: 'df-positions',
          format: 'float32',
          length: options.positions.length,
          values: options.positions
        })
      );
    }
    if (options.ids) {
      this.ids = this.track(
        new GPUMapGraphParameterBuffer(device, {
          id: 'df-ids',
          format: 'uint32',
          length: options.ids.length,
          values: options.ids
        })
      );
    }
    if (options.count !== undefined) {
      this.count = this.track(
        new GPUMapGraphParameterBuffer(device, {
          id: 'df-count',
          format: 'uint32',
          length: 1,
          values: Uint32Array.of(options.count)
        })
      );
    }
    if (options.mask) {
      this.mask = this.track(createInputBuffer(device, options.mask));
    }
    this.outputs = {
      distances: this.track(createOutputBuffer(device, this.cellCount)),
      allocation: this.track(createOutputBuffer(device, this.cellCount)),
      nearestCells: this.track(createOutputBuffer(device, this.cellCount)),
      withinDistance: this.track(createOutputBuffer(device, this.cellCount))
    };
    const hasTexture = options.texture && device.getTextureFormatCapabilities('r32float').store;
    const texture = hasTexture
      ? this.track(
          device.createTexture({
            format: 'r32float',
            width,
            height,
            usage: Texture.STORAGE | Texture.SAMPLE | Texture.COPY_SRC | Texture.COPY_DST
          })
        )
      : undefined;
    const textureView = texture
      ? graph.createTextureView(
          graph.importTexture(
            {
              id: 'df-texture',
              format: 'r32float',
              width,
              height,
              usage: texture.props.usage
            },
            texture
          ),
          {mipLevelCount: 1}
        )
      : undefined;
    const view = (name: keyof Fixture['outputs'], format: 'float32' | 'uint32') =>
      importGraphBuffer(graph, `df-${name}`, this.outputs[name], format, this.cellCount);
    this.recipe = new GPUDistanceField({
      id: 'df',
      width,
      height,
      settings: this.settings.importToGraph(graph),
      seedPositions: this.positions
        ? importGraphBuffer(
            graph,
            'df-positions',
            this.positions.buffer,
            'float32x2',
            options.positions!.length / 2
          )
        : undefined,
      seedIds: this.ids?.importToGraph(graph),
      seedCount: this.count?.importToGraph(graph),
      seedMask: this.mask
        ? importGraphBuffer(graph, 'df-mask', this.mask, 'uint32', this.cellCount)
        : undefined,
      mode: options.mode,
      jumpFloodRefinementPasses: options.jumpFloodRefinementPasses,
      output: {
        distances: view('distances', 'float32') as never,
        allocation: view('allocation', 'uint32') as never,
        nearestCells: view('nearestCells', 'uint32') as never,
        withinDistance: view('withinDistance', 'uint32') as never,
        texture: textureView as never
      }
    });
    this.getCommandNodes = vi.spyOn(this.recipe, 'getCommandNodes');
    graph.add(this.recipe);
    if (textureView) {
      this.textureBuffer = this.track(createOutputBuffer(device, this.cellCount));
      const validity = this.track(createOutputBuffer(device, this.cellCount));
      new GPURasterTextureToBuffer({
        id: 'df-texture-readback',
        input: {
          id: 'df-texture-band',
          format: 'float32',
          storage: {kind: 'texture', view: textureView}
        },
        output: importGraphBuffer(
          graph,
          'df-texture-readback',
          this.textureBuffer,
          'float32',
          this.cellCount
        ),
        outputValidity: importGraphBuffer(
          graph,
          'df-texture-validity',
          validity,
          'uint32',
          this.cellCount
        )
      }).addToGraph(graph);
    }
    this.compiled = graph.compile();
    this.scene = {
      width,
      height,
      settings: settingsValues,
      positions: options.positions,
      ids: options.ids,
      count: options.count,
      mask: options.mask
    };
  }

  /** Rewrites per-frame inputs without recompiling. */
  update(changes: {
    settings?: GPUDistanceFieldSettings;
    positions?: Float32Array;
    ids?: Uint32Array;
    count?: number;
    mask?: Uint32Array;
  }): void {
    if (changes.settings) {
      const values = getGPUDistanceFieldParameterValues(changes.settings);
      this.settings.write(values);
      this.scene = {...this.scene, settings: values};
    }
    if (changes.positions) {
      this.positions!.write(changes.positions);
      this.scene = {...this.scene, positions: changes.positions};
    }
    if (changes.ids) {
      this.ids!.write(changes.ids);
      this.scene = {...this.scene, ids: changes.ids};
    }
    if (changes.count !== undefined) {
      this.count!.write(Uint32Array.of(changes.count));
      this.scene = {...this.scene, count: changes.count};
    }
    if (changes.mask) {
      this.mask!.write(changes.mask);
      this.scene = {...this.scene, mask: changes.mask};
    }
  }

  async run(): Promise<FixtureResult> {
    submitGraph(this.device, this.compiled, undefined);
    const [distances, allocation, nearestCells, withinDistance, texture] = await Promise.all([
      readFloat32(this.outputs.distances, this.cellCount),
      readUint32(this.outputs.allocation, this.cellCount),
      readUint32(this.outputs.nearestCells, this.cellCount),
      readUint32(this.outputs.withinDistance, this.cellCount),
      this.textureBuffer ? readFloat32(this.textureBuffer, this.cellCount) : undefined
    ]);
    return {distances, allocation, nearestCells, withinDistance, texture};
  }

  destroy(): void {
    this.compiled.destroy();
    for (const resource of this.resources) {
      resource.destroy();
    }
  }

  private track<T extends {destroy(): void}>(resource: T): T {
    this.resources.push(resource);
    return resource;
  }
}

/** The texture copy holds every finite distance; texture readback reports `+Infinity` texels as invalid (NaN). */
function expectTextureMatches(result: FixtureResult): void {
  expect(
    result.texture!.map((value, cell) =>
      Number.isFinite(result.distances[cell]) ? value : Number.isFinite(value) ? value : 'unreached'
    )
  ).toEqual(result.distances.map(value => (Number.isFinite(value) ? value : 'unreached')));
}

/**
 * Asserts exact allocation and nearest cells and distances within `maximumUlps` of the f64
 * oracle. Returns the largest ULP difference seen.
 */
function expectExactParity(
  actual: FixtureResult,
  expected: DistanceFieldOracleResult,
  maximumUlps = 1
): number {
  expect(actual.allocation).toEqual(Array.from(expected.allocation));
  expect(actual.nearestCells).toEqual(Array.from(expected.nearestCells));
  expect(actual.withinDistance).toEqual(
    Array.from(expected.allocation, id => (id === NONE ? 0 : 1))
  );
  let worstUlps = 0;
  for (const [cell, distance] of expected.distances.entries()) {
    const ulps = getUlpDistance(actual.distances[cell], distance);
    worstUlps = Math.max(worstUlps, ulps);
  }
  expect(worstUlps).toBeLessThanOrEqual(maximumUlps);
  return worstUlps;
}

it('GPUDistanceField exact mode matches the brute-force oracle and follows per-frame seeds', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 193;
  const height = 129;
  const random = createRandom(17);
  const capacity = 160;
  const positions = createRandomSeedPositions(random, capacity, [-10, 5], [width * 2, height * 2]);
  // Rejected rows: NaN, outside the grid on each side, and an infinite coordinate.
  positions.set([Number.NaN, 10, -11, 10, 10, 4.5, 10000, 10, 10, Infinity], 0);
  // Several seeds in one cell: the smallest ID wins.
  positions.set([20.5, 20.5, 20.9, 20.1, 21.9, 21.9], 20);
  const ids = Uint32Array.from({length: capacity}, (_, row) => (row * 7919) % 1000);
  ids[30] = NONE;
  const fixture = new Fixture(device, {
    width,
    height,
    settings: {origin: [-10, 5], cellSize: [2, 2]},
    positions,
    ids,
    count: capacity,
    texture: true
  });
  const first = await fixture.run();
  const worstUlps = expectExactParity(first, computeDistanceFieldOnCPU(fixture.scene));
  if (first.texture) {
    expectTextureMatches(first);
  }

  // Move every seed and shrink the active count with the same compiled graph.
  for (const [seed, count] of [
    [5, 100],
    [6, 1],
    [7, capacity]
  ]) {
    fixture.update({
      positions: createRandomSeedPositions(
        createRandom(seed),
        capacity,
        [-10, 5],
        [width * 2, height * 2]
      ),
      count
    });
    expectExactParity(await fixture.run(), computeDistanceFieldOnCPU(fixture.scene));
  }
  expect(fixture.getCommandNodes).toHaveBeenCalledTimes(1);
  console.log(`distance-field exact isotropic worst ULPs: ${worstUlps}`);
  fixture.destroy();
});

it('GPUDistanceField breaks exact ties on the smallest seed ID, including three-way ties', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 21;
  const height = 21;
  // Cell (10, 10) is exactly 5 cells from every seed: three 3-4-5 / 0-5 offsets above, and
  // straight offsets below, left, and right.
  const seedCells = [
    [7, 6],
    [10, 5],
    [13, 6],
    [10, 15],
    [5, 10],
    [15, 10]
  ];
  const positions = Float32Array.from(seedCells.flatMap(([x, y]) => [x + 0.5, y + 0.5]));
  const permutations = [
    [5, 0, 9, 7, 8, 6],
    [9, 3, 1, 4, 2, 8],
    [2, 8, 0, 6, 7, 5],
    [8, 7, 6, 2, 9, 1]
  ];
  for (const mode of ['exact', 'jump-flood'] as const) {
    const fixture = new Fixture(device, {
      width,
      height,
      settings: {cellSize: [1, 1]},
      positions,
      ids: Uint32Array.from(permutations[0]),
      mode
    });
    for (const permutation of permutations) {
      fixture.update({ids: Uint32Array.from(permutation)});
      const actual = await fixture.run();
      const expected = computeDistanceFieldOnCPU(fixture.scene);
      // All six seeds are 5 cells from the center: the smallest ID wins.
      expect(actual.allocation[10 * width + 10]).toBe(Math.min(...permutation));
      expect(actual.distances[10 * width + 10]).toBe(5);
      if (mode === 'exact') {
        expectExactParity(actual, expected, 0);
      }
    }
    fixture.destroy();
  }
});

it('GPUDistanceField supports anisotropic cells within one ULP of the oracle', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 150;
  const height = 97;
  const random = createRandom(23);
  const capacity = 90;
  const positions = createRandomSeedPositions(random, capacity, [0, 0], [width, height]);
  const fixture = new Fixture(device, {
    width,
    height,
    settings: {cellSize: [1, 2.5]},
    positions
  });
  const results: string[] = [];
  for (const settings of [
    {cellSize: [1, 2.5]},
    {cellSize: [0.3, 0.7]},
    {bounds: [0, 0, 1000, 300], gridSize: [width, height]},
    {cellSize: [3, 3]}
  ] as GPUDistanceFieldSettings[]) {
    // Seeds stay at the same cells: rescale positions with the cell size.
    const values = getGPUDistanceFieldParameterValues(settings);
    fixture.update({
      settings,
      positions: positions.map(
        (value, index) => value * values[2 + (index % 2)] + values[index % 2]
      )
    });
    const actual = await fixture.run();
    const expected = computeDistanceFieldOnCPU(fixture.scene);
    let mismatches = 0;
    let worstUlps = 0;
    for (let cell = 0; cell < fixture.cellCount; cell++) {
      worstUlps = Math.max(
        worstUlps,
        getUlpDistance(actual.distances[cell], expected.distances[cell])
      );
      if (actual.allocation[cell] !== expected.allocation[cell]) {
        // Only f32 near-ties may resolve differently: the chosen seed is as close as the best.
        mismatches++;
        const chosen = getCellDistance(
          width,
          fixture.scene.settings,
          cell,
          actual.nearestCells[cell]
        );
        expect(Math.abs(chosen - expected.distances[cell])).toBeLessThanOrEqual(
          2 ** -22 * expected.distances[cell]
        );
      }
    }
    expect(worstUlps).toBeLessThanOrEqual(1);
    results.push(`${values[2]}x${values[3]}: ${worstUlps} ulp, ${mismatches} near-tie mismatches`);
  }
  expect(fixture.getCommandNodes).toHaveBeenCalledTimes(1);
  console.log(`distance-field anisotropic: ${results.join('; ')}`);
  fixture.destroy();
});

it('GPUDistanceField combines a labeled seed mask with points and applies maxDistance', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 64;
  const height = 48;
  const random = createRandom(31);
  const mask = new Uint32Array(width * height);
  for (let cell = 0; cell < mask.length; cell++) {
    if (random() < 0.004) {
      mask[cell] = 1 + Math.floor(random() * 5) * 10;
    }
  }
  // A horizontal "road" with label 3 (ID 2).
  for (let x = 5; x < 40; x++) {
    mask[30 * width + x] = 3;
  }
  const positions = createRandomSeedPositions(random, 12, [0, 0], [width * 10, height * 10]);
  const ids = Uint32Array.from({length: 12}, (_, row) => 100 + row);
  for (const mode of ['exact', 'jump-flood'] as const) {
    const fixture = new Fixture(device, {
      width,
      height,
      settings: {cellSize: [10, 10], maxDistance: 75},
      positions,
      ids,
      mask,
      mode,
      texture: true
    });
    const actual = await fixture.run();
    const expected = computeDistanceFieldOnCPU(fixture.scene);
    if (mode === 'exact') {
      expectExactParity(actual, expected);
    }
    expect(actual.distances.filter(value => value === Infinity).length).toBeGreaterThan(0);
    for (let cell = 0; cell < fixture.cellCount; cell++) {
      expect(actual.distances[cell] <= 75 || actual.distances[cell] === Infinity).toBe(true);
    }
    if (actual.texture) {
      expectTextureMatches(actual);
    }
    // Clearing the mask and raising the limit leaves only the points.
    fixture.update({
      mask: new Uint32Array(width * height),
      settings: {cellSize: [10, 10]}
    });
    const pointsOnly = await fixture.run();
    if (mode === 'exact') {
      expectExactParity(pointsOnly, computeDistanceFieldOnCPU(fixture.scene));
    }
    expect(new Set(pointsOnly.allocation).size).toBeLessThanOrEqual(12);
    fixture.destroy();
  }
});

it('GPUDistanceField writes empty fields when no seed is active', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  for (const mode of ['exact', 'jump-flood'] as const) {
    const fixture = new Fixture(device, {
      width: 9,
      height: 5,
      settings: {cellSize: [1, 1]},
      positions: Float32Array.of(1, 1, 2, 2),
      count: 0,
      mode
    });
    const empty = await fixture.run();
    expect(empty.distances.every(value => value === Infinity)).toBe(true);
    expect(empty.allocation.every(value => value === NONE)).toBe(true);
    expect(empty.nearestCells.every(value => value === NONE)).toBe(true);
    expect(empty.withinDistance.every(value => value === 0)).toBe(true);
    fixture.update({count: 1});
    const one = await fixture.run();
    expect(one.allocation.every(value => value === 0)).toBe(true);
    expect(one.distances[1 * 9 + 1]).toBe(0);
    expect(one.distances[4 * 9 + 8]).toBe(Math.fround(Math.hypot(7, 3)));
    fixture.update({count: 0});
    expect((await fixture.run()).distances.every(value => value === Infinity)).toBe(true);
    fixture.destroy();
  }
  // A 1x1 grid with a mask seed.
  const single = new Fixture(device, {
    width: 1,
    height: 1,
    settings: {cellSize: [1, 1]},
    mask: Uint32Array.of(8),
    mode: 'jump-flood'
  });
  const result = await single.run();
  expect(result.distances).toEqual([0]);
  expect(result.allocation).toEqual([7]);
  single.destroy();
});

it('GPUDistanceField jump flooding stays within a recorded error of the exact field', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 256;
  const height = 256;
  const report: string[] = [];
  for (const seedCount of [16, 256, 2048]) {
    const positions = createRandomSeedPositions(
      createRandom(seedCount),
      seedCount,
      [0, 0],
      [width, height]
    );
    for (const refinementPasses of [0, 1, 2] as const) {
      const fixture = new Fixture(device, {
        width,
        height,
        settings: {cellSize: [1, 1]},
        positions,
        mode: 'jump-flood',
        jumpFloodRefinementPasses: refinementPasses
      });
      const actual = await fixture.run();
      const expected = computeDistanceFieldOnCPU(fixture.scene);
      let wrongCells = 0;
      let worstError = 0;
      for (let cell = 0; cell < fixture.cellCount; cell++) {
        const error = actual.distances[cell] - expected.distances[cell];
        // JFA always reports the distance to a real seed, so it never undershoots.
        expect(error).toBeGreaterThanOrEqual(-1e-4);
        if (actual.allocation[cell] !== expected.allocation[cell]) {
          wrongCells++;
        }
        worstError = Math.max(worstError, error);
      }
      const wrongFraction = wrongCells / fixture.cellCount;
      // Bounds measured on this scene with margin; JFA errors are rare and sub-cell to few-cell.
      expect(wrongFraction).toBeLessThan(refinementPasses === 0 ? 0.01 : 0.005);
      expect(worstError).toBeLessThan(4);
      report.push(
        `seeds ${seedCount} JFA+${refinementPasses}: ${wrongCells} wrong cells (${(wrongFraction * 100).toFixed(3)}%), max error ${worstError.toFixed(3)} cells`
      );
      fixture.destroy();
    }
  }
  console.log(`distance-field jump-flood error 256x256:\n${report.join('\n')}`);
});

it('GPUDistanceField reports rough wall-clock timings at 1024x1024', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 1024;
  const height = 1024;
  const positions = createRandomSeedPositions(createRandom(3), 1000, [0, 0], [width, height]);
  const report: string[] = [];
  for (const mode of ['exact', 'jump-flood'] as const) {
    const fixture = new Fixture(device, {
      width,
      height,
      settings: {cellSize: [1, 1]},
      positions,
      mode
    });
    await fixture.run();
    const start = performance.now();
    const repeats = 5;
    for (let repeat = 0; repeat < repeats; repeat++) {
      submitGraph(device, fixture.compiled, undefined);
    }
    await readUint32(fixture.outputs.allocation, 1);
    report.push(`${mode} ${((performance.now() - start) / repeats).toFixed(2)} ms`);
    fixture.destroy();
  }
  console.log(
    `distance-field 1024x1024, 1000 seeds, submit+drain per encoding: ${report.join(', ')}`
  );
});
