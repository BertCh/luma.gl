// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {
  GPUTerrainSummits,
  getTerrainSummitsStackBytes,
  type GPUTerrainSummitsProps
} from '../../../src/gpu-terrain/terrain-features/gpu-terrain-summits';
import {
  CORE_PRIVATE_MEMORY_LIMIT_BYTES,
  createNullWebGPUDevice,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources,
  getPrivateMemoryBytes,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {createBand} from '../terrain-test-utils';
import {computeTerrainSummits, isInOracleDisc, isOnOracleRing} from './terrain-summits-oracle';
import {
  computeSummitsPreviousKernel,
  computeSummitsRangeKernel,
  getModelDiscColumnHalfHeight,
  getModelDiscRowHalfWidth,
  type SummitsModelInput,
  type SummitsModelStatistics
} from './terrain-summits-range-model';

const f32 = Math.fround;
const FLOAT_MAXIMUM = 3.4028234663852886e38;

function createRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function getSettings(radius: number, cellX: number, cellY: number, minimumDrop = 0) {
  return [radius, minimumDrop, cellX, cellY, 5000000, 4900000, 0, 0];
}

function expectSameResult(input: SummitsModelInput, label: string) {
  const previous = computeSummitsPreviousKernel(input);
  const range = computeSummitsRangeKernel(input);
  expect(range.mask, `${label} mask`).toEqual(previous.mask);
  expect(range.radiusClamped, `${label} clamp`).toBe(previous.radiusClamped);
  for (let pixel = 0; pixel < previous.drop.length; pixel++) {
    // Object.is: the sign of a zero drop and NaN positions must match too.
    expect(Object.is(range.drop[pixel], previous.drop[pixel]), `${label} drop ${pixel}`).toBe(true);
  }
  return previous;
}

type Dem = {values: Float32Array; validity: Uint32Array | undefined; label: string};

function createDems(width: number, height: number, seed: number): Dem[] {
  const random = createRandom(seed);
  const size = width * height;
  const make = (label: string, fill: (x: number, y: number) => number, invalidRate = 0): Dem => {
    const values = new Float32Array(size);
    const validity = new Uint32Array(size).fill(1);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        values[y * width + x] = fill(x, y);
        if (random() < invalidRate) {
          validity[y * width + x] = 0;
          values[y * width + x] = random() < 0.5 ? Number.NaN : 1e9;
        }
      }
    }
    return {label, values, validity: invalidRate > 0 ? validity : undefined};
  };
  return [
    make('plateau', () => 5),
    make('ties', () => Math.floor(random() * 3)),
    make('signed zeros', () => (random() < 0.5 ? -0 : 0)),
    make('noise', () => random() * 100),
    make('hills', (x, y) => Math.sin(x / 3.3) * Math.cos(y / 2.7) * 40 + random()),
    make('nodata', () => random() * 10, 0.12),
    make('ties nodata', () => Math.floor(random() * 4), 0.08),
    make('extremes', () => [FLOAT_MAXIMUM, -FLOAT_MAXIMUM, 0, 1][Math.floor(random() * 4)]),
    make('extremes nodata', () => (random() < 0.5 ? FLOAT_MAXIMUM : -FLOAT_MAXIMUM), 0.1)
  ];
}

it('range kernel equals the previous O(R^2) kernel on adversarial DEMs', () => {
  const grids: [number, number][] = [
    [1, 1],
    [1, 9],
    [9, 1],
    [7, 5],
    [19, 13],
    [27, 23]
  ];
  const cells: [number, number][] = [
    [1, 1],
    [1, 3],
    [10, 1],
    [f32(0.1), f32(1 / 3)]
  ];
  let summitTotal = 0;
  let caseCount = 0;
  for (const [width, height] of grids) {
    for (const dem of createDems(width, height, width * 31 + height)) {
      for (const [cellX, cellY] of cells) {
        for (const incompleteNeighborhood of ['reject', 'ignore'] as const) {
          for (const [radiusPixels, maximumRadiusPixels] of [
            [2.5, 16],
            [6.2, 16],
            [11, 12],
            [30, 5] // clamped
          ]) {
            const radius = f32(radiusPixels * Math.min(cellX, cellY));
            const input: SummitsModelInput = {
              values: dem.values,
              validity: dem.validity,
              width,
              height,
              settings: getSettings(radius, cellX, cellY, caseCount % 3 === 0 ? 0.5 : 0),
              maximumRadiusPixels,
              incompleteNeighborhood
            };
            const result = expectSameResult(
              input,
              `${dem.label} ${width}x${height} ${cellX}x${cellY} ${incompleteNeighborhood} r${radiusPixels}`
            );
            summitTotal += result.mask.reduce((sum, value) => sum + value, 0);
            caseCount++;
          }
        }
      }
    }
  }
  // The comparison is not vacuous.
  expect(summitTotal).toBeGreaterThan(200);
  expect(caseCount).toBeGreaterThan(1000);
});

it('range kernel equals the previous kernel in web-mercator mode', () => {
  for (const dem of createDems(21, 17, 7)) {
    for (const incompleteNeighborhood of ['reject', 'ignore'] as const) {
      expectSameResult(
        {
          values: dem.values,
          validity: dem.validity,
          width: 21,
          height: 17,
          settings: getSettings(4500, 1000, 1000),
          cellSizeMode: 'web-mercator',
          incompleteNeighborhood
        },
        `${dem.label} mercator`
      );
    }
  }
});

it('range kernel agrees with the definitional oracle', () => {
  for (const dem of createDems(25, 21, 99).filter(entry => !/extremes/.test(entry.label))) {
    for (const incompleteNeighborhood of ['reject', 'ignore'] as const) {
      const settings = getSettings(f32(4.4 * 2), 2, 3);
      settings[3] = 3;
      const input: SummitsModelInput = {
        values: dem.values,
        validity: dem.validity,
        width: 25,
        height: 21,
        settings,
        incompleteNeighborhood
      };
      const range = computeSummitsRangeKernel(input);
      const oracle = computeTerrainSummits(dem.values, dem.validity, 25, 21, settings, {
        incompleteNeighborhood
      });
      expect(range.mask, dem.label).toEqual(oracle.mask);
      // The oracle takes Math.max of the ring, which does not preserve the first of +0 / -0; the
      // previous-kernel comparison above checks the sign of zero bit for bit.
      expect(
        range.drop.map(value => (Number.isNaN(value) ? 'nan' : value + 0)),
        dem.label
      ).toEqual(oracle.drop.map(value => (Number.isNaN(value) ? 'nan' : value + 0)));
    }
  }
});

const CELL_SIZES: [number, number][] = [
  [1, 1],
  [1, 3],
  [10, 1],
  [f32(0.1), f32(1 / 3)],
  [f32(7.77), f32(0.013)],
  [f32(1e-3), f32(2.5e-3)]
];

function* getDiscCases() {
  const random = createRandom(5);
  for (const [cellX, cellY] of CELL_SIZES) {
    for (let sample = 0; sample < 40; sample++) {
      const maximumRadiusPixels = 1 + Math.floor(random() * 20);
      const requested = f32(random() * 30 * Math.max(cellX, cellY));
      if (requested <= 0) {
        continue;
      }
      const maximum = Math.min(f32(maximumRadiusPixels * cellX), f32(maximumRadiusPixels * cellY));
      const radius = Math.min(requested, maximum);
      yield {cellX, cellY, maximumRadiusPixels, radius, radiusSquared: f32(radius * radius)};
    }
  }
}

it('disc rows, columns and the extent box match the per-pixel predicate', () => {
  for (const {cellX, cellY, maximumRadiusPixels, radius, radiusSquared} of getDiscCases()) {
    const inDisc = (dx: number, dy: number) =>
      isInOracleDisc(dx, dy, 0, 0, cellX, cellY, radiusSquared);
    const extentX = Math.min(Math.floor(f32(f32(radius / cellX) + 0.5)) + 1, maximumRadiusPixels);
    const extentY = Math.min(Math.floor(f32(f32(radius / cellY) + 0.5)) + 1, maximumRadiusPixels);
    for (let dy = -80; dy <= 80; dy++) {
      let brute = -1;
      for (let dx = 0; dx <= 80; dx++) {
        if (inDisc(dx, dy)) {
          brute = dx;
        }
      }
      // Monotone in |dx|: the in-disc offsets of a row are exactly [-halfWidth, halfWidth].
      for (let dx = -80; dx <= 80; dx++) {
        expect(inDisc(dx, dy)).toBe(Math.abs(dx) <= brute);
      }
      expect(getModelDiscRowHalfWidth(dy, cellX, cellY, radiusSquared, maximumRadiusPixels)).toBe(
        brute
      );
      // The disc is inside the box the previous kernel scanned.
      if (brute >= 0) {
        expect(Math.abs(dy)).toBeLessThanOrEqual(extentY);
        expect(brute).toBeLessThanOrEqual(extentX);
      }
    }
    let tallest = 0;
    for (let dy = 0; dy <= 80; dy++) {
      if (inDisc(0, dy)) {
        tallest = dy;
      }
    }
    expect(getModelDiscColumnHalfHeight(cellX, cellY, radiusSquared, maximumRadiusPixels)).toBe(
      tallest
    );
  }
});

it('ring runs equal isOnDiscRing for every disc pixel', () => {
  for (const {cellX, cellY, maximumRadiusPixels, radiusSquared} of getDiscCases()) {
    const halfWidth = (dy: number) =>
      getModelDiscRowHalfWidth(dy, cellX, cellY, radiusSquared, maximumRadiusPixels);
    for (let dy = -30; dy <= 30; dy++) {
      const interior = Math.min(halfWidth(dy - 1), halfWidth(dy), halfWidth(dy + 1));
      for (let dx = -30; dx <= 30; dx++) {
        const inDisc = isInOracleDisc(dx, dy, 0, 0, cellX, cellY, radiusSquared);
        const onRing = isOnOracleRing(dx, dy, 0, 0, cellX, cellY, radiusSquared);
        const isRunPixel = Math.abs(dx) <= halfWidth(dy) && !(Math.abs(dx) <= interior - 1);
        expect(inDisc && onRing, `${dx},${dy}`).toBe(isRunPixel);
      }
    }
  }
});

it('block containment and disjointness tests are exact under float32 rounding', () => {
  const random = createRandom(11);
  for (const {cellX, cellY, radiusSquared} of getDiscCases()) {
    const inDisc = (dx: number, dy: number) =>
      isInOracleDisc(dx, dy, 0, 0, cellX, cellY, radiusSquared);
    for (let sample = 0; sample < 60; sample++) {
      const dxLow = Math.floor(random() * 50) - 25;
      const dyLow = Math.floor(random() * 50) - 25;
      const dxHigh = dxLow + Math.floor(random() * 8);
      const dyHigh = dyLow + Math.floor(random() * 8);
      let all = true;
      let none = true;
      let corners = true;
      for (let dy = dyLow; dy <= dyHigh; dy++) {
        for (let dx = dxLow; dx <= dxHigh; dx++) {
          all = all && inDisc(dx, dy);
          none = none && !inDisc(dx, dy);
        }
      }
      for (const dx of [dxLow, dxHigh]) {
        for (const dy of [dyLow, dyHigh]) {
          corners = corners && inDisc(dx, dy);
        }
      }
      const nearX = dxLow > 0 ? dxLow : dxHigh < 0 ? dxHigh : 0;
      const nearY = dyLow > 0 ? dyLow : dyHigh < 0 ? dyHigh : 0;
      const farX = Math.max(Math.abs(dxLow), Math.abs(dxHigh));
      const farY = Math.max(Math.abs(dyLow), Math.abs(dyHigh));
      expect(inDisc(farX, farY)).toBe(all);
      expect(corners).toBe(all);
      expect(!inDisc(nearX, nearY)).toBe(none);
    }
  }
});

it('the descent visits O(R) blocks per pixel', () => {
  const size = 241;
  const random = createRandom(3);
  const values = new Float32Array(size * size);
  for (let pixel = 0; pixel < values.length; pixel++) {
    values[pixel] = random();
  }
  // A single strict maximum in the middle forces the whole descent for it.
  values[120 * size + 120] = 10;
  const visits = (radiusPixels: number) => {
    const statistics: SummitsModelStatistics = {blocksVisited: 0, maximumDepth: 0};
    const result = computeSummitsRangeKernel(
      {
        values,
        width: size,
        height: size,
        settings: getSettings(radiusPixels, 1, 1),
        maximumRadiusPixels: 64,
        incompleteNeighborhood: 'ignore'
      },
      statistics
    );
    expect(result.mask[120 * size + 120]).toBe(1);
    return statistics;
  };
  const small = visits(8);
  const large = visits(48);
  // Linear growth is 6x; quadratic would be 36x.
  expect(large.blocksVisited / small.blocksVisited).toBeLessThan(12);
  expect(large.maximumDepth).toBeLessThanOrEqual(3 * 8 + 4);
});

function createSummits(
  width: number,
  height: number,
  overrides: Partial<GPUTerrainSummitsProps> = {}
) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const pixelCount = width * height;
  const summits = new GPUTerrainSummits({
    grid: makeGPURasterMetadata({width, height}),
    elevation: createBand(graph, 'elevation', pixelCount),
    parameters: createTransientView(graph, 'settings', 'float32', 8),
    output: {
      summitMask: createTransientView(graph, 'mask', 'uint32', pixelCount),
      drop: createTransientView(graph, 'drop', 'float32', pixelCount),
      status: {radiusClamped: createTransientView(graph, 'clamp', 'uint32', 1)}
    },
    ...overrides
  });
  return {summits, nodes: summits.getCommandNodes(graph)};
}

it('summit kernels parse, fit private memory and bindings, and keep one pipeline', () => {
  for (const incompleteNeighborhood of ['reject', 'ignore'] as const) {
    for (const cellSizeMode of ['uniform', 'geographic'] as const) {
      const keyCounts = [8, 31, 64, 129, 513].map(size => {
        const {summits, nodes} = createSummits(size, size + 3, {
          incompleteNeighborhood,
          cellSizeMode,
          maximumRadiusPixels: 64
        });
        const sources = getParsedKernelSources(nodes);
        const kernel = sources.get('terrain-summits-summits')!;
        expect(kernel).toBeDefined();
        expect(getPrivateMemoryBytes(kernel)).toBeLessThan(CORE_PRIVATE_MEMORY_LIMIT_BYTES);
        expect(getStorageBindingCount(kernel)).toBeLessThanOrEqual(8);
        expect(getTerrainSummitsStackBytes(summits.pyramidLayout)).toBeLessThan(
          CORE_PRIVATE_MEMORY_LIMIT_BYTES - 256
        );
        // Radius is a per-frame parameter: the source carries no radius or loop-count constants.
        expect(kernel).not.toMatch(/const RADIUS/);
        return getGPUCommandNodePipelineKeys(nodes, {idPattern: /terrain-summits-summits$/}).size;
      });
      expect(keyCounts).toEqual([1, 1, 1, 1, 1]);
    }
  }
});

it('the reject mode prepares an invalid-as-maximum pyramid input, ignore does not', () => {
  const reject = createSummits(16, 12, {incompleteNeighborhood: 'reject'});
  expect(reject.nodes.map(node => node.id)).toContain('terrain-summits-pyramid-input');
  const ignore = createSummits(16, 12, {incompleteNeighborhood: 'ignore'});
  expect(ignore.nodes.map(node => node.id)).not.toContain('terrain-summits-pyramid-input');
  expect(ignore.nodes.map(node => node.id)).toContain('terrain-summits-pyramid-level-0');
  expect(ignore.summits.pyramidLayout.footprint).toBe('cell');
  expect(ignore.summits.pyramidLayout.firstBlockSize).toBe(2);
});

it('a prebuilt pyramid is validated', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const layout = createSummits(16, 12).summits.pyramidLayout;
  const make = (overrides: Record<string, unknown>, pyramidOverrides = {}) =>
    new GPUTerrainSummits({
      grid: makeGPURasterMetadata({width: 16, height: 12}),
      elevation: createBand(graph, `elevation-${Math.random()}`, 192),
      parameters: createTransientView(graph, `settings-${Math.random()}`, 'float32', 8),
      output: {summitMask: createTransientView(graph, `mask-${Math.random()}`, 'uint32', 192)},
      incompleteNeighborhood: 'ignore',
      pyramid: {
        layout,
        combined: createTransientView(
          graph,
          `pyramid-${Math.random()}`,
          'float32',
          2 * layout.length
        ),
        ...pyramidOverrides
      },
      ...overrides
    });
  expect(() => make({})).not.toThrow();
  expect(() => make({incompleteNeighborhood: 'reject'})).toThrow(/requires incompleteNeighborhood/);
  expect(() => make({incompleteNeighborhood: undefined})).toThrow(
    /requires incompleteNeighborhood/
  );
  expect(() => make({}, {layout: {...layout, footprint: 'bilinear'}})).toThrow(/'cell' footprint/);
  expect(() => make({}, {layout: {...layout, width: 17}})).toThrow(/covers/);
  expect(() =>
    make({}, {combined: createTransientView(graph, 'short', 'float32', layout.length)})
  ).toThrow(/must contain/);
});
