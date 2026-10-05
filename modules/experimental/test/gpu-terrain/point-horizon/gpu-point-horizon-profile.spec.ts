// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {
  getGPUPointHorizonDistanceLattice,
  getGPUPointHorizonParameterValues,
  getGPUPointHorizonSegments,
  GPUPointHorizonProfile,
  type GPUPointHorizonModelOptions,
  type GPUPointHorizonSettings
} from '../../../src/gpu-terrain/point-horizon';
import {
  createInputBuffer,
  createOutputBuffer,
  isSoftwareDevice,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {
  computePointHorizonProfile,
  createFractalTerrain,
  type PointHorizonOracleModel
} from './point-horizon-oracle';

type ProfileConfig = {
  width: number;
  height: number;
  values: Float32Array;
  validity?: Uint32Array;
  observers: readonly (readonly [number, number, number])[];
  settings: GPUPointHorizonSettings;
  model: GPUPointHorizonModelOptions;
  /** Replace the skyline-angle output with the evaluated-sample debug counter. */
  countSamples?: boolean;
};

type ProfileResult = {
  tangent: Float32Array;
  skylineAngle: Float32Array;
  distance: Float32Array;
  /** Evaluated samples per ray, only when `countSamples` replaced the skyline-angle output. */
  samples?: Uint32Array;
};

async function runProfile(device: Device, config: ProfileConfig): Promise<ProfileResult> {
  const {width, height, values, validity, observers, model} = config;
  const azimuthSpan = model.azimuthSpan ?? model.azimuthCount ?? 720;
  const rayCount = observers.length * azimuthSpan;
  const terrain = createInputBuffer(device, values);
  const validityBuffer = validity ? createInputBuffer(device, validity) : undefined;
  const observerBuffer = createInputBuffer(
    device,
    Float32Array.from(observers.flatMap(([column, row, eyeHeight]) => [column, row, eyeHeight, 0]))
  );
  const settings = new GPUParameterBuffer(device, {
    id: 'horizon-settings',
    format: 'float32',
    length: 8,
    values: getGPUPointHorizonParameterValues(config.settings)
  });
  const outputs = [0, 1, 2].map(() => createOutputBuffer(device, rayCount));
  const graph = new GPUCommandGraph(device, {id: 'point-horizon-test'});
  graph.add(
    new GPUPointHorizonProfile({
      ...model,
      width,
      height,
      elevation: {
        id: 'terrain',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'terrain', terrain, 'float32', width * height)
        },
        validity: validityBuffer
          ? importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', width * height)
          : undefined
      },
      observers: importGraphBuffer(
        graph,
        'observers',
        observerBuffer,
        'float32x4',
        observers.length
      ),
      settings: settings.importToGraph(graph),
      tangent: importGraphBuffer(graph, 'tangent', outputs[0], 'float32', rayCount),
      skylineAngle: config.countSamples
        ? undefined
        : importGraphBuffer(graph, 'skyline-angle', outputs[1], 'float32', rayCount),
      samples: config.countSamples
        ? importGraphBuffer(graph, 'samples', outputs[1], 'uint32', rayCount)
        : undefined,
      distance: importGraphBuffer(graph, 'distance', outputs[2], 'float32', rayCount)
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [tangent, skylineAngle, distance] = await Promise.all(
    outputs.map(async buffer => Float32Array.from(await readFloat32(buffer, rayCount)))
  );
  const samples = config.countSamples
    ? Uint32Array.from(await readUint32(outputs[1], rayCount))
    : undefined;
  compiled.destroy();
  settings.destroy();
  for (const buffer of [
    terrain,
    observerBuffer,
    ...outputs,
    ...(validityBuffer ? [validityBuffer] : [])
  ]) {
    buffer.destroy();
  }
  return {tangent, skylineAngle, distance, samples};
}

function toBits(values: Float32Array): Uint32Array {
  return new Uint32Array(values.buffer, values.byteOffset, values.length);
}

function expectBitIdentical(left: ProfileResult, right: ProfileResult) {
  expect(Array.from(toBits(left.tangent))).toEqual(Array.from(toBits(right.tangent)));
  expect(Array.from(toBits(left.distance))).toEqual(Array.from(toBits(right.distance)));
}

function getOracleModel(config: ProfileConfig): PointHorizonOracleModel {
  const {model} = config;
  const lattice = getGPUPointHorizonDistanceLattice(model);
  const settings = config.settings;
  const azimuthCount = model.azimuthCount ?? 720;
  return {
    width: config.width,
    height: config.height,
    values: config.values,
    validity: config.validity,
    projection: model.projection ?? 'planar',
    rowDirection: model.rowDirection,
    heightReference: model.heightReference,
    azimuthCount,
    firstAzimuth: model.firstAzimuth ?? 0,
    azimuthSpan: model.azimuthSpan ?? azimuthCount,
    lattice,
    segments:
      model.projection === 'web-mercator' ? getGPUPointHorizonSegments(lattice, model) : undefined,
    curvature: settings.curvatureCoefficient ?? 0,
    maximumDistance: settings.maximumDistance,
    cellSize: 'cellSize' in settings ? settings.cellSize : undefined,
    worldPixelSize: 'worldPixelSize' in settings ? settings.worldPixelSize : undefined,
    originY: 'originY' in settings ? settings.originY : undefined
  };
}

function maximumSkylineAngleError(gpu: Float32Array, oracle: Float64Array): number {
  let maximum = 0;
  for (let index = 0; index < gpu.length; index++) {
    if (Number.isNaN(oracle[index])) {
      expect(gpu[index]).toBeNaN();
      continue;
    }
    const delta = Math.abs(gpu[index] - oracle[index]);
    maximum = Math.max(maximum, delta);
  }
  return maximum;
}

const PLANAR_MODEL: GPUPointHorizonModelOptions = {
  projection: 'planar',
  azimuthCount: 72,
  maximumDistance: 600,
  cellSize: 10
};

it('GPUPointHorizonProfile finds the curvature dip of a flat plane', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const size = 129;
  const curvature = 1e-4;
  const height = 10;
  for (const traversal of ['march', 'pyramid'] as const) {
    const result = await runProfile(device, {
      width: size,
      height: size,
      values: new Float32Array(size * size),
      observers: [[64, 64, height]],
      settings: {cellSize: [10, 10], curvatureCoefficient: curvature},
      model: {...PLANAR_MODEL, traversal}
    });
    // max over d of (-h / d - c d) = -2 sqrt(h c) at d = sqrt(h / c) = 316 m
    const expected = (Math.atan(-2 * Math.sqrt(height * curvature)) * 180) / Math.PI;
    expect(result.skylineAngle.length).toBe(72);
    for (const value of result.skylineAngle) {
      expect(Math.abs(value - expected)).toBeLessThan(1e-3);
    }
    expect(Math.abs(result.distance[0] - Math.sqrt(height / curvature))).toBeLessThan(25);
  }
});

it('GPUPointHorizonProfile sees a wall at its angle and a higher eye lowers it', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const size = 129;
  const wallHeight = 40;
  const wallColumn = 84; // 200 m east of the observer
  const values = new Float32Array(size * size);
  for (let row = 0; row < size; row++) {
    values[row * size + wallColumn] = wallHeight;
  }
  const curvature = 1e-6;
  const run = (eyeHeight: number, curvatureCoefficient: number) =>
    runProfile(device, {
      width: size,
      height: size,
      values,
      observers: [[64, 64, eyeHeight]],
      settings: {cellSize: [10, 10], curvatureCoefficient},
      model: {...PLANAR_MODEL, azimuthCount: 4, maximumDistance: 600}
    });
  const low = await run(0, curvature);
  const east = 1; // azimuth index 1 of 4 is 90 degrees
  const expected = (Math.atan(wallHeight / 200 - 200 * curvature) * 180) / Math.PI;
  expect(Math.abs(low.skylineAngle[east] - expected)).toBeLessThan(1e-3);
  expect(Math.abs(low.distance[east] - 200)).toBeLessThan(1e-3);
  const high = await run(20, curvature);
  expect(high.skylineAngle[east]).toBeLessThan(low.skylineAngle[east]);
  const flat = await run(0, 0);
  const curved = await run(0, 4e-4);
  expect(curved.skylineAngle[east]).toBeLessThan(flat.skylineAngle[east]);
  // A smaller drop coefficient (larger refraction k) raises the skyline.
  expect(flat.skylineAngle[east]).toBeGreaterThan(low.skylineAngle[east]);
});

it('GPUPointHorizonProfile pyramid equals march bit for bit on fractal terrain (planar)', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 200;
  const height = 176;
  const values = createFractalTerrain(width, height, 7, 500, 40);
  const validity = new Uint32Array(width * height).fill(1);
  for (let row = 100; row < 108; row++) {
    for (let column = 120; column < 130; column++) {
      validity[row * width + column] = 0;
    }
  }
  for (const heightReference of ['ground', 'absolute'] as const) {
    const config: ProfileConfig = {
      width,
      height,
      values,
      validity,
      observers: [
        [100.13, 88.31, heightReference === 'ground' ? 2 : 700],
        [60.37, 40.81, heightReference === 'ground' ? 30 : 900],
        [150.5, 140.25, heightReference === 'ground' ? 1.7 : 400],
        [-3, 10, 2]
      ],
      settings: {cellSize: [12, 15], curvatureCoefficient: 3e-5},
      model: {
        projection: 'planar',
        heightReference,
        azimuthCount: 180,
        maximumDistance: 2200,
        cellSize: 13
      }
    };
    const march = await runProfile(device, {
      ...config,
      model: {...config.model, traversal: 'march'}
    });
    const pyramid = await runProfile(device, {
      ...config,
      model: {...config.model, traversal: 'pyramid'}
    });
    expectBitIdentical(march, pyramid);
    // Non-trivial: varied skyline, hidden terrain, and an invalid (outside) observer.
    const row = march.skylineAngle.slice(0, 180);
    expect(new Set(row).size).toBeGreaterThan(60);
    expect(Math.max(...row)).toBeGreaterThan(Math.min(...row) + 1);
    expect(march.skylineAngle[3 * 180]).toBeNaN();
    expect(march.tangent[3 * 180 + 5]).toBeNaN();
    expect(march.distance[3 * 180 + 5]).toBeNaN();
    if (!isSoftwareDevice(device)) {
      const oracle = computePointHorizonProfile(getOracleModel(config), config.observers);
      const error = maximumSkylineAngleError(march.skylineAngle, oracle.skylineAngle);
      expect(error).toBeLessThan(2e-3);
    }
  }
});

it('GPUPointHorizonProfile pyramid equals march bit for bit on fractal terrain (web mercator)', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 192;
  const height = 192;
  const values = createFractalTerrain(width, height, 11, 600, 36);
  const worldPixelSize = 512 * 2 ** 12;
  const latitude = (46 * Math.PI) / 180;
  const centerRow = 96;
  const originY =
    worldPixelSize * (0.5 - Math.asinh(Math.tan(latitude)) / (2 * Math.PI)) - centerRow - 0.5;
  const config: ProfileConfig = {
    width,
    height,
    values,
    observers: [
      [96, 96, 2],
      [50.25, 130.5, 20],
      [140, 30.75, 1.7]
    ],
    settings: {
      worldPixelSize,
      originY,
      curvatureCoefficient: 2e-5
    },
    model: {
      projection: 'web-mercator',
      azimuthCount: 144,
      maximumDistance: 2000,
      cellSize: 13.5,
      maximumLatitude: 47
    }
  };
  const march = await runProfile(device, {...config, model: {...config.model, traversal: 'march'}});
  const pyramid = await runProfile(device, {
    ...config,
    model: {...config.model, traversal: 'pyramid'}
  });
  expectBitIdentical(march, pyramid);
  expect(new Set(march.skylineAngle).size).toBeGreaterThan(60);
  if (!isSoftwareDevice(device)) {
    const oracle = computePointHorizonProfile(getOracleModel(config), config.observers);
    const error = maximumSkylineAngleError(march.skylineAngle, oracle.skylineAngle);
    expect(error).toBeLessThan(2e-3);
  }
});

it('GPUPointHorizonProfile sector run equals the slice of the full run', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 160;
  const height = 160;
  const values = createFractalTerrain(width, height, 3, 300, 32);
  const config: ProfileConfig = {
    width,
    height,
    values,
    observers: [[80, 80, 3]],
    settings: {cellSize: [10, 10]},
    model: {azimuthCount: 120, maximumDistance: 700, cellSize: 10}
  };
  const full = await runProfile(device, config);
  const sector = await runProfile(device, {
    ...config,
    model: {...config.model, firstAzimuth: 30, azimuthSpan: 25}
  });
  expect(Array.from(toBits(sector.tangent))).toEqual(
    Array.from(toBits(full.tangent.slice(30, 55)))
  );
  expect(Array.from(toBits(sector.distance))).toEqual(
    Array.from(toBits(full.distance.slice(30, 55)))
  );
  expect(new Set(full.skylineAngle).size).toBeGreaterThan(20);
});

it('GPUPointHorizonProfile pyramid omits samples without changing the result', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 256;
  const height = 256;
  const config: ProfileConfig = {
    width,
    height,
    values: createFractalTerrain(width, height, 21, 800, 64),
    observers: [
      [128.4, 128.7, 5],
      [40.25, 200.5, 5]
    ],
    settings: {cellSize: [20, 20], curvatureCoefficient: 6.8e-8},
    model: {azimuthCount: 90, maximumDistance: 2500, cellSize: 2},
    countSamples: true
  };
  const march = await runProfile(device, {...config, model: {...config.model, traversal: 'march'}});
  const pyramid = await runProfile(device, {
    ...config,
    model: {...config.model, traversal: 'pyramid'}
  });
  expectBitIdentical(march, pyramid);
  const total = (counts?: Uint32Array) => {
    let sum = 0;
    for (const count of counts ?? []) {
      sum += count;
    }
    return sum;
  };
  expect(total(march.samples)).toBeGreaterThan(90 * 2 * 500);
  expect(total(pyramid.samples)).toBeLessThan(0.7 * total(march.samples));
  expect(total(pyramid.samples)).toBeGreaterThan(0);
});
