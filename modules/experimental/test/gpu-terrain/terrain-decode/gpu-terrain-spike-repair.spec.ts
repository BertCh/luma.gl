// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  GPU_TERRAIN_SPIKE_REPAIR_STATISTICS as S,
  GPUTerrainSpikeRepair,
  type GPUTerrainSpikeRepairProps
} from '../../../src/gpu-terrain/terrain-decode/gpu-terrain-spike-repair';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {
  computeTerrainSpikeRepair,
  type TerrainSpikeRepairOracleOptions
} from './terrain-spike-repair-oracle';

const WIDTH = 64;
const HEIGHT = 48;
const SENTINEL = -32768;

type Options = TerrainSpikeRepairOracleOptions & {componentIterations?: number};

/** Smooth Terrarium-like terrain; every value is a multiple of 1/256 and gradients stay below 12 m. */
function createTerrain(width: number, height: number): Float32Array {
  return Float32Array.from({length: width * height}, (_, index) => {
    const x = index % width;
    const y = Math.floor(index / width);
    const meters = 800 + 3 * x + 2 * y + 40 * Math.sin(x / 6) * Math.cos(y / 7);
    return Math.round(meters * 256) / 256;
  });
}

function addBlock(
  values: Float32Array,
  width: number,
  left: number,
  top: number,
  blockWidth: number,
  blockHeight: number,
  offset: number
): void {
  for (let y = top; y < top + blockHeight; y++) {
    for (let x = left; x < left + blockWidth; x++) {
      values[y * width + x] += offset;
    }
  }
}

async function runRepair(
  device: Device,
  values: Float32Array,
  validity: Uint32Array | undefined,
  width: number,
  height: number,
  options: Options = {}
) {
  const pixelCount = width * height;
  const inputValues = createInputBuffer(device, values);
  const inputValidity = validity ? createInputBuffer(device, validity) : undefined;
  const outputs = {
    values: createOutputBuffer(device, pixelCount),
    validity: createOutputBuffer(device, pixelCount),
    labels: createOutputBuffer(device, pixelCount),
    statistics: createOutputBuffer(device, 5)
  };
  const graph = new GPUCommandGraph(device, {id: 'terrain-spike-repair-test'});
  const props: GPUTerrainSpikeRepairProps = {
    ...options,
    width,
    height,
    elevation: {
      id: 'elevation',
      format: 'float32',
      storage: {
        kind: 'buffer',
        values: importGraphBuffer(graph, 'elevation', inputValues, 'float32', pixelCount)
      },
      validity: inputValidity
        ? importGraphBuffer(graph, 'elevation-validity', inputValidity, 'uint32', pixelCount)
        : undefined
    },
    values: importGraphBuffer(graph, 'values', outputs.values, 'float32', pixelCount),
    validity: importGraphBuffer(graph, 'validity', outputs.validity, 'uint32', pixelCount),
    labels: importGraphBuffer(graph, 'labels', outputs.labels, 'uint32', pixelCount),
    statistics: importGraphBuffer(graph, 'statistics', outputs.statistics, 'uint32', 5)
  };
  const contributor = new GPUTerrainSpikeRepair(props);
  graph.add(contributor);
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    values: await readFloat32(outputs.values, pixelCount),
    validity: await readUint32(outputs.validity, pixelCount),
    labels: await readUint32(outputs.labels, pixelCount),
    statistics: await readUint32(outputs.statistics, 5)
  };
  compiled.destroy();
  contributor.destroy();
  for (const buffer of [inputValues, inputValidity, ...Object.values(outputs)]) {
    buffer?.destroy();
  }
  return result;
}

/** Runs the GPU and the oracle and requires bit-exact equality of every output. */
async function expectMatchesOracle(
  device: Device,
  values: Float32Array,
  validity: Uint32Array | undefined,
  width: number,
  height: number,
  options: Options = {}
) {
  const gpu = await runRepair(device, values, validity, width, height, options);
  const oracle = computeTerrainSpikeRepair(values, validity, width, height, options);
  expect(gpu.values).toEqual(Array.from(oracle.values));
  expect(gpu.validity).toEqual(Array.from(oracle.validity));
  expect(gpu.labels).toEqual(Array.from(oracle.labels));
  expect(gpu.statistics).toEqual(oracle.statistics);
  return {gpu, oracle};
}

it('GPUTerrainSpikeRepair removes single pixels and blocks of +/-256 and +512', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const clean = createTerrain(WIDTH, HEIGHT);
  const noisy = Float32Array.from(clean);
  addBlock(noisy, WIDTH, 20, 20, 1, 1, 256);
  addBlock(noisy, WIDTH, 30, 10, 5, 4, -256);
  addBlock(noisy, WIDTH, 10, 30, 4, 4, 512);
  addBlock(noisy, WIDTH, 0, 0, 6, 6, 256);
  addBlock(noisy, WIDTH, 58, 38, 6, 10, -256);
  const {gpu} = await expectMatchesOracle(device, noisy, undefined, WIDTH, HEIGHT);
  expect(gpu.values).toEqual(Array.from(clean));
  expect(gpu.statistics[S.shiftedComponentCount]).toBe(5);
  expect(gpu.statistics[S.repairedPixelCount]).toBe(1 + 20 + 16 + 36 + 60);
  expect(gpu.statistics[S.jumpCount]).toBeGreaterThan(0);
  expect(gpu.statistics[S.remainingJumpCount]).toBe(0);
  expect(gpu.statistics[S.converged]).toBe(1);
  expect(gpu.validity.every(flag => flag === 1)).toBe(true);
  device.destroy?.();
});

it('GPUTerrainSpikeRepair leaves real cliffs, large regions and the main component alone', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // A 350 m cliff is not a multiple of 256 within 40 m.
  const cliff = createTerrain(WIDTH, HEIGHT);
  addBlock(cliff, WIDTH, 20, 10, 8, 8, 350);
  let result = await expectMatchesOracle(device, cliff, undefined, WIDTH, HEIGHT);
  expect(result.gpu.values).toEqual(Array.from(cliff));
  expect(result.gpu.statistics[S.repairedPixelCount]).toBe(0);
  expect(result.gpu.statistics[S.jumpCount]).toBe(result.gpu.statistics[S.remainingJumpCount]);
  expect(result.gpu.statistics[S.jumpCount]).toBeGreaterThan(0);

  // A +256 plateau covering about 31 % of the tile exceeds the 25 % limit.
  const plateau = createTerrain(WIDTH, HEIGHT);
  addBlock(plateau, WIDTH, 44, 0, 20, HEIGHT, 256);
  result = await expectMatchesOracle(device, plateau, undefined, WIDTH, HEIGHT);
  expect(result.gpu.values).toEqual(Array.from(plateau));
  // Allowing larger components repairs the same plateau.
  result = await expectMatchesOracle(device, plateau, undefined, WIDTH, HEIGHT, {
    maximumComponentFraction: 0.5
  });
  expect(result.gpu.statistics[S.shiftedComponentCount]).toBe(1);
  expect(result.gpu.statistics[S.remainingJumpCount]).toBe(0);
  device.destroy?.();
});

it('GPUTerrainSpikeRepair matches the oracle on 216-296 m butte walls', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const terrain = createTerrain(WIDTH, HEIGHT);
  // Real enclosed buttes with walls strictly inside step +/- tolerance (216 and 296 m are the
  // exclusive limits) look like +256 m errors, so the opt-in repair shifts them; 215 m is left.
  addBlock(terrain, WIDTH, 8, 8, 5, 5, 220);
  addBlock(terrain, WIDTH, 30, 20, 5, 5, 290);
  addBlock(terrain, WIDTH, 40, 30, 5, 5, 215);
  const {gpu} = await expectMatchesOracle(device, terrain, undefined, WIDTH, HEIGHT);
  expect(gpu.statistics[S.shiftedComponentCount]).toBe(2);
  expect(gpu.statistics[S.repairedPixelCount]).toBe(50);
  device.destroy?.();
});

it('GPUTerrainSpikeRepair never lets nodata bleed or get repaired', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const noisy = createTerrain(WIDTH, HEIGHT);
  addBlock(noisy, WIDTH, 30, 10, 4, 4, 256);
  const validity = new Uint32Array(WIDTH * HEIGHT).fill(1);
  // A hole beside the spike and a hole splitting the terrain, with nodata sentinels in the value slot.
  const holes = [
    [34, 10],
    [34, 11],
    [29, 12],
    [5, 5],
    [5, 6],
    [20, 40]
  ];
  for (const [x, y] of holes) {
    validity[y * WIDTH + x] = 0;
    noisy[y * WIDTH + x] = SENTINEL;
  }
  noisy[40 * WIDTH + 20] = 1e30;
  const {gpu} = await expectMatchesOracle(device, noisy, validity, WIDTH, HEIGHT);
  for (const [x, y] of holes) {
    expect(gpu.validity[y * WIDTH + x]).toBe(0);
    expect(gpu.values[y * WIDTH + x]).toBeNaN();
    expect(gpu.labels[y * WIDTH + x]).toBe(y * WIDTH + x);
  }
  expect(gpu.statistics[S.repairedPixelCount]).toBeGreaterThan(0);
  // Changing the sentinel never changes any valid output.
  const other = Float32Array.from(noisy);
  for (const [x, y] of holes) {
    other[y * WIDTH + x] = 12345.5;
  }
  const otherResult = await runRepair(device, other, validity, WIDTH, HEIGHT);
  expect(otherResult).toEqual(gpu);
  // A spike whose only neighbours are nodata stays isolated and untouched.
  const lone = new Float32Array(9).fill(SENTINEL);
  const loneValidity = new Uint32Array(9);
  lone[4] = 5000;
  loneValidity[4] = 1;
  const loneResult = await runRepair(device, lone, loneValidity, 3, 3);
  expect(loneResult.values[4]).toBe(5000);
  expect(loneResult.statistics.slice(0, 4)).toEqual([0, 0, 0, 0]);
  device.destroy?.();
});

it('GPUTerrainSpikeRepair breaks equal-size ties toward the lowest label', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const values = new Float32Array(8 * 4).fill(100);
  addBlock(values, 8, 4, 0, 4, 4, 256);
  const options = {maximumComponentFraction: 0.5};
  const {gpu} = await expectMatchesOracle(device, values, undefined, 8, 4, options);
  // The left half (label 0) is main; the right half moves down to 100.
  expect(gpu.values).toEqual(new Array(32).fill(100));
  expect(gpu.statistics[S.shiftedComponentCount]).toBe(1);
  // Mirror image: the left half is raised and stays main, so the right half moves up.
  const mirrored = new Float32Array(8 * 4).fill(100);
  addBlock(mirrored, 8, 0, 0, 4, 4, 256);
  const mirror = await expectMatchesOracle(device, mirrored, undefined, 8, 4, options);
  expect(mirror.gpu.values).toEqual(new Array(32).fill(356));
  device.destroy?.();
});

/** Serpentine corridor of valid pixels separated by nodata walls with alternating gaps. */
function createSerpentine(size: number) {
  const values = createTerrain(size, size);
  const validity = new Uint32Array(size * size).fill(1);
  for (let y = 1; y < size; y += 2) {
    const gap = (y >> 1) % 2 === 0 ? size - 1 : 0;
    for (let x = 0; x < size; x++) {
      if (x !== gap) {
        validity[y * size + x] = 0;
        values[y * size + x] = SENTINEL;
      }
    }
  }
  return {values, validity};
}

it('GPUTerrainSpikeRepair converges on a serpentine component and fails closed otherwise', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // One 131 k pixel corridor in a 513 x 513 tile; measured: 14 iterations converge it (the
  // fixed-point check needs one confirming round), 13 do not. The default of 32 leaves margin.
  const size = 513;
  const {values, validity} = createSerpentine(size);
  addBlock(values, size, 200, 256, 3, 1, 256);
  const {gpu} = await expectMatchesOracle(device, values, validity, size, size);
  expect(gpu.statistics[S.converged]).toBe(1);
  expect(gpu.statistics[S.repairedPixelCount]).toBe(3);

  const failed = await runRepair(device, values, validity, size, size, {componentIterations: 1});
  expect(failed.statistics[S.converged]).toBe(0);
  expect(failed.statistics[S.repairedPixelCount]).toBe(0);
  expect(failed.statistics[S.shiftedComponentCount]).toBe(0);
  expect(failed.statistics[S.remainingJumpCount]).toBe(failed.statistics[S.jumpCount]);
  expect(failed.values.filter(Number.isFinite)).toEqual(
    Array.from(values).filter((_, index) => validity[index] !== 0)
  );
  device.destroy?.();
});

it('GPUTerrainSpikeRepair supports a non-default step', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const terrain = createTerrain(WIDTH, HEIGHT);
  addBlock(terrain, WIDTH, 20, 20, 3, 3, 512);
  const {gpu} = await expectMatchesOracle(device, terrain, undefined, WIDTH, HEIGHT, {
    step: 512,
    tolerance: 60,
    jump: 300
  });
  expect(gpu.statistics[S.repairedPixelCount]).toBe(9);
  device.destroy?.();
});

it('GPUTerrainSpikeRepair counts label runs correctly across strips, rows and nodata', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  // 37 x 29 = 1073 pixels is not a multiple of the 16-pixel counting strips, and blocks start
  // mid-strip, wrap across row ends, interleave labels (one-pixel-wide column) and contain nodata.
  const width = 37;
  const height = 29;
  const terrain = createTerrain(width, height);
  addBlock(terrain, width, 3, 4, 1, 12, 256);
  addBlock(terrain, width, 10, 2, 7, 3, -256);
  addBlock(terrain, width, 20, 14, 17, 2, 512);
  addBlock(terrain, width, 5, 22, 9, 7, 256);
  const validity = new Uint32Array(width * height).fill(1);
  for (const index of [23 * width + 7, 24 * width + 8, 15 * width + 30, 0, width * height - 1]) {
    validity[index] = 0;
  }
  const {gpu} = await expectMatchesOracle(device, terrain, validity, width, height);
  expect(gpu.statistics[S.shiftedComponentCount]).toBeGreaterThan(0);
  expect(gpu.statistics[S.repairedPixelCount]).toBeGreaterThan(0);
  device.destroy?.();
});
