// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer, importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {
  decodeGPUTerrainHorizonUnorm16,
  encodeGPUTerrainHorizonUnorm16,
  getGPUTerrainHorizonDirection,
  getGPUTerrainHorizonParameterValues,
  getGPUTerrainHorizonStepDistances,
  GPUTerrainHorizon,
  GPU_TERRAIN_HORIZON_UNORM16_STEP_DEGREES,
  unpackGPUTerrainHorizonUnorm16,
  type GPUTerrainHorizonFormat,
  type GPUTerrainHorizonProps,
  type GPUTerrainHorizonSettings
} from '../../../src/gpu-terrain/terrain-illumination/gpu-terrain-horizon';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../../utils/gpu-contributor-test-utils';
import {
  getAnisotropicWeightOracle,
  computeTerrainHorizon,
  createSmoothTerrain,
  decodeHorizonUnorm16Oracle,
  encodeHorizonUnorm16Oracle,
  type HorizonOracleResult
} from './terrain-horizon-oracle';
import {computeTerrainHorizonSweepBruteForce} from './terrain-horizon-sweep-oracle';
import {expectClose} from '../terrain-test-utils';

type HorizonFixture = {
  run(settings: GPUTerrainHorizonSettings): Promise<HorizonOracleResult>;
  compileCount: number;
  destroy(): void;
};

type FixtureOptions = Omit<
  GPUTerrainHorizonProps,
  'elevation' | 'settings' | 'horizon' | 'skyViewFactor' | 'positiveOpenness' | 'validity'
> & {withHorizon?: boolean};

function createHorizonFixture(
  device: Device,
  elevation: Float32Array,
  options: FixtureOptions
): HorizonFixture {
  const {width, height} = options;
  const pixelCount = width * height;
  const directionCount = options.directionCount ?? 16;
  const withHorizon = options.withHorizon ?? true;
  const graph = new GPUCommandGraph(device, {id: 'terrain-horizon-test'});
  const elevationBuffer = createInputBuffer(device, elevation);
  const buffers = {
    horizon: withHorizon ? createOutputBuffer(device, pixelCount * directionCount) : undefined,
    skyViewFactor: createOutputBuffer(device, pixelCount),
    positiveOpenness: createOutputBuffer(device, pixelCount),
    validity: createOutputBuffer(device, pixelCount)
  };
  const settings = new GPUParameterBuffer(device, {
    id: 'horizon-settings',
    format: 'float32',
    length: 8,
    values: getGPUTerrainHorizonParameterValues({cellSize: [10, 10]})
  });
  const {withHorizon: _, ...contributorOptions} = options;
  graph.add(
    new GPUTerrainHorizon({
      ...contributorOptions,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', pixelCount)
        }
      },
      settings: settings.importToGraph(graph),
      horizon: buffers.horizon
        ? importGraphBuffer(
            graph,
            'horizon',
            buffers.horizon,
            'float32',
            pixelCount * directionCount
          )
        : undefined,
      skyViewFactor: importGraphBuffer(graph, 'svf', buffers.skyViewFactor, 'float32', pixelCount),
      positiveOpenness: importGraphBuffer(
        graph,
        'openness',
        buffers.positiveOpenness,
        'float32',
        pixelCount
      ),
      validity: importGraphBuffer(graph, 'validity', buffers.validity, 'uint32', pixelCount)
    })
  );
  const compiled = graph.compile();
  const fixture: HorizonFixture = {
    compileCount: 1,
    async run(values) {
      settings.write(getGPUTerrainHorizonParameterValues(values));
      submitGraph(device, compiled, undefined);
      return {
        horizon: buffers.horizon
          ? await readFloat32(buffers.horizon, pixelCount * directionCount)
          : [],
        skyViewFactor: await readFloat32(buffers.skyViewFactor, pixelCount),
        positiveOpenness: await readFloat32(buffers.positiveOpenness, pixelCount),
        validity: await readUint32(buffers.validity, pixelCount)
      };
    },
    destroy() {
      compiled.destroy();
      settings.destroy();
      elevationBuffer.destroy();
      for (const buffer of Object.values(buffers)) (buffer as Buffer | undefined)?.destroy();
    }
  };
  return fixture;
}

function expectOracle(actual: HorizonOracleResult, expected: HorizonOracleResult): void {
  if (actual.horizon.length > 0) {
    expectClose(actual.horizon, expected.horizon, 2e-3);
  }
  expectClose(actual.skyViewFactor, expected.skyViewFactor, 1e-5);
  expectClose(actual.positiveOpenness, expected.positiveOpenness, 2e-3);
  expect(actual.validity).toEqual(expected.validity);
}

it('GPUTerrainHorizon sees a flat sky on a plane and a raised horizon in a pit', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 9;
  const height = 9;
  const flat = new Float32Array(width * height).fill(50);
  const flatFixture = createHorizonFixture(device, flat, {
    width,
    height,
    directionCount: 8,
    maximumRadius: 6
  });
  const flatResult = await flatFixture.run({cellSize: [10, 10]});
  expect(flatResult.horizon.every(value => value === 0)).toBe(true);
  expect(flatResult.skyViewFactor.every(value => value === 1)).toBe(true);
  expect(flatResult.positiveOpenness.every(value => value === 90)).toBe(true);
  flatFixture.destroy();

  // A one-pixel pit 10 m deep at the center. Axis rays sample a neighbour 10 m above at 10 m: 45
  // deg. Diagonal rays sample at (0.707, 0.707) pixels where the bilinear pit weight is
  // (1 - 0.707)^2, so the rise is 10 * (1 - (1 - 0.707)^2) over 10 m.
  const pit = new Float32Array(width * height).fill(50);
  const center = 4 * width + 4;
  pit[center] = 40;
  const pitFixture = createHorizonFixture(device, pit, {
    width,
    height,
    directionCount: 8,
    maximumRadius: 1
  });
  const pitResult = await pitFixture.run({cellSize: [10, 10]});
  const diagonal = (Math.atan(1 - (1 - Math.SQRT1_2) ** 2) * 180) / Math.PI;
  const expected = [45, diagonal, 45, diagonal, 45, diagonal, 45, diagonal];
  for (let sector = 0; sector < 8; sector++) {
    expect(pitResult.horizon[center * 8 + sector]).toBeCloseTo(expected[sector], 3);
  }
  const svf = 1 - expected.reduce((sum, angle) => sum + Math.sin((angle * Math.PI) / 180), 0) / 8;
  expect(pitResult.skyViewFactor[center]).toBeCloseTo(svf, 5);
  pitFixture.destroy();
});

it('GPUTerrainHorizon matches the oracle on random terrain with nodata', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 48;
  const height = 40;
  const elevation = createSmoothTerrain(width, height, 7);
  for (const hole of [5 * width + 7, 5 * width + 8, 20 * width + 30, 39 * width + 47]) {
    elevation[hole] = NaN;
  }
  for (const [directionCount, stepGrowth, maximumRadius] of [
    [8, 1, 12],
    [16, 1.25, 30],
    [7, 1.5, 20]
  ] as const) {
    const fixture = createHorizonFixture(device, elevation, {
      width,
      height,
      directionCount,
      maximumRadius,
      stepGrowth
    });
    const stepDistances = getGPUTerrainHorizonStepDistances(maximumRadius, stepGrowth);
    const settingsList: GPUTerrainHorizonSettings[] = [
      {cellSize: [10, 10]},
      {cellSize: [10, 12], zFactor: 2},
      {cellSize: [10, 10], curvatureCoefficient: 1e-3, maximumDistance: 95}
    ];
    for (const settings of settingsList) {
      const actual = await fixture.run(settings);
      expectOracle(
        actual,
        computeTerrainHorizon({
          width,
          height,
          elevation,
          directionCount,
          stepDistances,
          cellSize: [settings.cellSize[0], settings.cellSize[1]],
          zFactor: settings.zFactor,
          curvatureCoefficient: settings.curvatureCoefficient,
          maximumDistance: settings.maximumDistance
        })
      );
    }
    // Settings changes reuse the compiled graph.
    expect(fixture.compileCount).toBe(1);
    fixture.destroy();
  }
});

it('GPUTerrainHorizon supports north rows, Web Mercator cells, and sky-view-only output', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 32;
  const height = 24;
  const elevation = createSmoothTerrain(width, height, 3);
  const settings: GPUTerrainHorizonSettings = {
    cellSize: [20, 20],
    northEdge: 0.3,
    southEdge: 0.32
  };
  const options = {
    width,
    height,
    directionCount: 12,
    maximumRadius: 10,
    rowDirection: 'north' as const,
    cellSizeMode: 'web-mercator' as const
  };
  const full = createHorizonFixture(device, elevation, options);
  const fullResult = await full.run(settings);
  expectOracle(
    fullResult,
    computeTerrainHorizon({
      ...options,
      elevation,
      stepDistances: getGPUTerrainHorizonStepDistances(10),
      cellSize: [20, 20],
      northEdge: 0.3,
      southEdge: 0.32
    })
  );
  full.destroy();
  const skyOnly = createHorizonFixture(device, elevation, {
    ...options,
    withHorizon: false
  });
  const skyResult = await skyOnly.run(settings);
  expect(skyResult.skyViewFactor).toEqual(fullResult.skyViewFactor);
  expect(skyResult.positiveOpenness).toEqual(fullResult.positiveOpenness);
  skyOnly.destroy();
});

it('GPUTerrainHorizon gives identical results however the sectors are split over dispatches', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 37;
  const height = 29;
  const elevation = createSmoothTerrain(width, height, 11);
  elevation[7 * width + 9] = NaN;
  elevation[20 * width + 30] = NaN;
  const settings: GPUTerrainHorizonSettings = {cellSize: [10, 12], zFactor: 1.5};
  for (const directionCount of [8, 7]) {
    const options = {width, height, directionCount, maximumRadius: 14, stepGrowth: 1.2};
    const fused = createHorizonFixture(device, elevation, options);
    // One sector per dispatch: sums continue through the buffers between chunks.
    const chunked = createHorizonFixture(device, elevation, {
      ...options,
      maximumStepsPerDispatch: 1
    });
    const fusedResult = await fused.run(settings);
    const chunkedResult = await chunked.run(settings);
    // The two kernels are compiled separately, so fused multiply-add choices may differ by an ulp.
    expectClose(chunkedResult.horizon, fusedResult.horizon, 1e-4);
    expectClose(chunkedResult.skyViewFactor, fusedResult.skyViewFactor, 1e-6);
    expectClose(chunkedResult.positiveOpenness, fusedResult.positiveOpenness, 1e-4);
    expect(chunkedResult.validity).toEqual(fusedResult.validity);
    fused.destroy();
    chunked.destroy();
  }
});

// Fast-math checks: Metal may fuse or reassociate `zFactor * (z - z0)` and `rise / distance`, which
// would turn a flat plane at a large elevation into spurious nonzero horizons.
it('GPUTerrainHorizon keeps a high-elevation flat plane exactly flat for any z factor', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 24;
  const height = 24;
  const flat = new Float32Array(width * height).fill(4321.123);
  const fixture = createHorizonFixture(device, flat, {
    width,
    height,
    directionCount: 16,
    maximumRadius: 20
  });
  for (const zFactor of [1.37, 0.3, 2.9]) {
    const result = await fixture.run({cellSize: [10, 10], zFactor});
    const worstHorizon = result.horizon.reduce(
      (worst, value) => Math.max(worst, Math.abs(value)),
      0
    );
    const worstSvf = result.skyViewFactor.reduce(
      (worst, value) => Math.max(worst, Math.abs(value - 1)),
      0
    );
    // Prove the kernel ran: every pixel valid, and openness is the flat-sky 90 degrees.
    expect(result.validity.every(value => value === 1)).toBe(true);
    expect(result.positiveOpenness.every(value => value === 90)).toBe(true);
    expect(result.horizon.length).toBe(width * height * 16);
    expect(worstHorizon, `zFactor ${zFactor}`).toBe(0);
    expect(worstSvf, `zFactor ${zFactor}`).toBe(0);
  }
  fixture.destroy();
});

it('GPUTerrainHorizon matches analytic and float64 horizons on a tilted high-elevation plane', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 48;
  const height = 48;
  const directionCount = 16;
  // 3.75 m per 10 m cell (exact in float32): slope 0.375, z = 4000 + 0.375 * x.
  const plane = Float32Array.from(
    {length: width * height},
    (_, index) => 4000 + 3.75 * (index % width)
  );
  const fixture = createHorizonFixture(device, plane, {
    width,
    height,
    directionCount,
    maximumRadius: 6
  });
  const result = await fixture.run({cellSize: [10, 10]});
  let worstAxis = 0;
  let worstOther = 0;
  let nonTrivial = 0;
  for (let row = 7; row < height - 7; row++) {
    for (let column = 7; column < width - 7; column++) {
      for (let sector = 0; sector < directionCount; sector++) {
        const [dx] = getGPUTerrainHorizonDirection(sector, directionCount);
        const expected = (Math.atan(0.375 * dx) * 180) / Math.PI;
        const error = Math.abs(
          result.horizon[(row * width + column) * directionCount + sector] - expected
        );
        if (sector % 4 === 0) {
          worstAxis = Math.max(worstAxis, error);
        } else {
          worstOther = Math.max(worstOther, error);
        }
        if (Math.abs(expected) > 5) {
          nonTrivial++;
        }
      }
    }
  }
  expect(nonTrivial).toBeGreaterThan(1000);
  expect(worstAxis).toBeLessThan(2e-5);
  expect(worstOther).toBeLessThan(2e-4);
  fixture.destroy();
});

it('GPUTerrainHorizon matches float64 on a tilted plane with curvature and a long radius', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 64;
  const height = 64;
  const directionCount = 8;
  const plane = Float32Array.from(
    {length: width * height},
    (_, index) => 4000 + 3.75 * (index % width) + 1.5 * Math.floor(index / width)
  );
  const fixture = createHorizonFixture(device, plane, {
    width,
    height,
    directionCount,
    maximumRadius: 60
  });
  const settings = {cellSize: [10, 10] as [number, number], curvatureCoefficient: 6.8e-8};
  const actual = await fixture.run(settings);
  const expected = computeTerrainHorizon({
    width,
    height,
    elevation: plane,
    directionCount,
    stepDistances: getGPUTerrainHorizonStepDistances(60),
    cellSize: [10, 10],
    curvatureCoefficient: 6.8e-8
  });
  // Axis sectors sample exact pixels (no interpolation). Off-axis samples interpolate
  // centre-relative differences, so the 4.9e-4 m float32 spacing near 4000 m no longer bounds the
  // agreement (it did before the march subtracted the centre ahead of interpolation: 2e-3 deg).
  let worstAxis = 0;
  let worstOther = 0;
  let steep = 0;
  for (const [index, value] of expected.horizon.entries()) {
    const error = Math.abs(actual.horizon[index] - value);
    if ((index % directionCount) % 2 === 0) {
      worstAxis = Math.max(worstAxis, error);
    } else {
      worstOther = Math.max(worstOther, error);
    }
    steep += Math.abs(value) > 5 ? 1 : 0;
  }
  expect(steep).toBeGreaterThan(1000);
  expect(worstAxis).toBeLessThan(2e-5);
  expect(worstOther).toBeLessThan(2e-4);
  fixture.destroy();
});

type ExtendedOutputs = {
  horizonWords: number[];
  horizon: number[];
  skyViewFactor: number[];
  positiveOpenness: number[];
  negativeOpenness: number[];
  anisotropicSkyViewFactor: number[];
  validity: number[];
};

/** Runs one GPUTerrainHorizon with every output bound, in the given horizon format. */
async function runExtendedHorizon(
  device: Device,
  elevation: Float32Array,
  options: {
    width: number;
    height: number;
    directionCount: number;
    maximumRadius: number;
    horizonFormat: GPUTerrainHorizonFormat;
    algorithm?: 'march' | 'sweep';
  },
  settings: GPUTerrainHorizonSettings
): Promise<ExtendedOutputs> {
  const {width, height, directionCount, horizonFormat} = options;
  const pixelCount = width * height;
  const elementCount = pixelCount * directionCount;
  const horizonLength = horizonFormat === 'unorm16' ? Math.ceil(elementCount / 2) : elementCount;
  const graph = new GPUCommandGraph(device, {id: 'terrain-horizon-extended-test'});
  const elevationBuffer = createInputBuffer(device, elevation);
  const outputs = {
    horizon: createOutputBuffer(device, horizonLength),
    svf: createOutputBuffer(device, pixelCount),
    positive: createOutputBuffer(device, pixelCount),
    negative: createOutputBuffer(device, pixelCount),
    anisotropic: createOutputBuffer(device, pixelCount),
    validity: createOutputBuffer(device, pixelCount)
  };
  const settingsBuffer = new GPUParameterBuffer(device, {
    id: 'horizon-extended-settings',
    format: 'float32',
    length: 12,
    values: getGPUTerrainHorizonParameterValues(settings, new Float32Array(12))
  });
  graph.add(
    new GPUTerrainHorizon({
      width,
      height,
      directionCount,
      maximumRadius: options.maximumRadius,
      horizonFormat,
      algorithm: options.algorithm,
      elevation: {
        id: 'elevation',
        format: 'float32',
        storage: {
          kind: 'buffer',
          values: importGraphBuffer(graph, 'elevation', elevationBuffer, 'float32', pixelCount)
        }
      },
      settings: settingsBuffer.importToGraph(graph),
      horizon:
        horizonFormat === 'unorm16'
          ? importGraphBuffer(graph, 'horizon', outputs.horizon, 'uint32', horizonLength)
          : importGraphBuffer(graph, 'horizon', outputs.horizon, 'float32', horizonLength),
      skyViewFactor: importGraphBuffer(graph, 'svf', outputs.svf, 'float32', pixelCount),
      positiveOpenness: importGraphBuffer(
        graph,
        'positive',
        outputs.positive,
        'float32',
        pixelCount
      ),
      negativeOpenness: importGraphBuffer(
        graph,
        'negative',
        outputs.negative,
        'float32',
        pixelCount
      ),
      anisotropicSkyViewFactor: importGraphBuffer(
        graph,
        'anisotropic',
        outputs.anisotropic,
        'float32',
        pixelCount
      ),
      validity: importGraphBuffer(graph, 'validity', outputs.validity, 'uint32', pixelCount)
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const horizonWords = await readUint32(outputs.horizon, horizonLength);
  const horizonBytes = await outputs.horizon.readAsync();
  const result: ExtendedOutputs = {
    horizonWords,
    horizon:
      horizonFormat === 'unorm16'
        ? Array.from(unpackGPUTerrainHorizonUnorm16(horizonWords, elementCount))
        : Array.from(new Float32Array(horizonBytes.buffer, horizonBytes.byteOffset, elementCount)),
    skyViewFactor: await readFloat32(outputs.svf, pixelCount),
    positiveOpenness: await readFloat32(outputs.positive, pixelCount),
    negativeOpenness: await readFloat32(outputs.negative, pixelCount),
    anisotropicSkyViewFactor: await readFloat32(outputs.anisotropic, pixelCount),
    validity: await readUint32(outputs.validity, pixelCount)
  };
  compiled.destroy();
  settingsBuffer.destroy();
  elevationBuffer.destroy();
  for (const buffer of Object.values(outputs)) buffer.destroy();
  return result;
}

it('GPUTerrainHorizon writes negative openness and anisotropic sky-view factor matching float64', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 36;
  const height = 30;
  const elevation = createSmoothTerrain(width, height, 13);
  for (const hole of [4 * width + 6, 4 * width + 7, 20 * width + 20, 29 * width + 35]) {
    elevation[hole] = NaN;
  }
  const options = {
    width,
    height,
    directionCount: 16,
    maximumRadius: 14,
    horizonFormat: 'float32'
  } as const;
  for (const anisotropy of [
    {azimuthDegrees: 315, level: 4, minimumWeight: 0.4},
    {azimuthDegrees: 45, level: 8, minimumWeight: 0.1},
    {azimuthDegrees: 100, level: 0, minimumWeight: 0}
  ]) {
    const settings = {
      cellSize: [10, 10] as [number, number],
      zFactor: 1.5,
      anisotropyAzimuthDegrees: anisotropy.azimuthDegrees,
      anisotropyLevel: anisotropy.level,
      anisotropyMinimumWeight: anisotropy.minimumWeight
    };
    const actual = await runExtendedHorizon(device, elevation, options, settings);
    const expected = computeTerrainHorizon({
      width,
      height,
      elevation,
      directionCount: 16,
      stepDistances: getGPUTerrainHorizonStepDistances(14),
      cellSize: [10, 10],
      zFactor: 1.5,
      computeNadir: true,
      anisotropy
    });
    expectClose(actual.negativeOpenness, expected.negativeOpenness!, 2e-3);
    expectClose(actual.anisotropicSkyViewFactor, expected.anisotropicSkyViewFactor!, 1e-5);
    expectClose(actual.skyViewFactor, expected.skyViewFactor, 1e-5);
    expectClose(actual.positiveOpenness, expected.positiveOpenness, 2e-3);
    // Non-trivial structure: valid pixels vary and negative openness differs from positive.
    const valid = actual.validity.filter(value => value === 1).length;
    expect(valid).toBe(width * height - 4);
    const spread = (values: number[]) => {
      const finite = values.filter(Number.isFinite);
      return Math.max(...finite) - Math.min(...finite);
    };
    expect(spread(actual.negativeOpenness)).toBeGreaterThan(1);
    expect(spread(actual.anisotropicSkyViewFactor)).toBeGreaterThan(0.01);
    expect(actual.negativeOpenness.filter(Number.isNaN).length).toBe(4);
    expect(actual.anisotropicSkyViewFactor.filter(Number.isNaN).length).toBe(4);
    if (anisotropy.level === 0) {
      // Equal weights reduce the anisotropic factor to the plain sky-view factor.
      expectClose(actual.anisotropicSkyViewFactor, actual.skyViewFactor, 1e-5);
    }
  }
});

it('GPUTerrainHorizon negative openness sees a peak as a pit', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 9;
  const height = 9;
  const center = 4 * width + 4;
  const options = {
    width,
    height,
    directionCount: 8,
    maximumRadius: 1,
    horizonFormat: 'float32'
  } as const;
  const peak = new Float32Array(width * height).fill(50);
  peak[center] = 60;
  const peakResult = await runExtendedHorizon(device, peak, options, {cellSize: [10, 10]});
  const pit = new Float32Array(width * height).fill(50);
  pit[center] = 40;
  const pitResult = await runExtendedHorizon(device, pit, options, {cellSize: [10, 10]});
  // The inverted peak is the pit: nadir openness of the peak equals positive openness of the pit.
  expect(peakResult.negativeOpenness[center]).toBeCloseTo(pitResult.positiveOpenness[center], 4);
  expect(peakResult.negativeOpenness[center]).toBeLessThan(80);
  // A peak sees a flat sky above, and the pit's negative openness is the flat 90.
  expect(peakResult.positiveOpenness[center]).toBeGreaterThan(90);
  expect(pitResult.negativeOpenness[center]).toBeGreaterThan(90);
  expect(peakResult.skyViewFactor[center]).toBe(1);
});

it('GPUTerrainHorizon unorm16 horizon matches the quantised float32 horizon', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 17;
  const height = 13;
  const elevation = createSmoothTerrain(width, height, 5);
  elevation[3 * width + 3] = NaN;
  elevation[9 * width + 12] = NaN;
  // 7 is odd: width * height * 7 = 1547 elements, so the last word is half used.
  for (const directionCount of [8, 7]) {
    const base = {width, height, directionCount, maximumRadius: 10};
    const settings = {cellSize: [10, 10] as [number, number]};
    const reference = await runExtendedHorizon(
      device,
      elevation,
      {...base, horizonFormat: 'float32'},
      settings
    );
    const packed = await runExtendedHorizon(
      device,
      elevation,
      {...base, horizonFormat: 'unorm16'},
      settings
    );
    const elementCount = width * height * directionCount;
    expect(packed.horizonWords.length).toBe(Math.ceil(elementCount / 2));
    let worstCodeDifference = 0;
    let exactMatches = 0;
    let nanCount = 0;
    for (let element = 0; element < elementCount; element++) {
      const word = packed.horizonWords[element >> 1];
      const code = (word >>> ((element & 1) * 16)) & 0xffff;
      const expectedCode = encodeHorizonUnorm16Oracle(reference.horizon[element]);
      if (Number.isNaN(reference.horizon[element])) {
        expect(code, `element ${element}`).toBe(0);
        nanCount++;
        continue;
      }
      worstCodeDifference = Math.max(worstCodeDifference, Math.abs(code - expectedCode));
      exactMatches += code === expectedCode ? 1 : 0;
      expect(Math.abs(packed.horizon[element] - reference.horizon[element])).toBeLessThanOrEqual(
        GPU_TERRAIN_HORIZON_UNORM16_STEP_DEGREES
      );
    }
    expect(nanCount).toBe(2 * directionCount);
    expect(worstCodeDifference).toBeLessThanOrEqual(1);
    expect(exactMatches).toBeGreaterThan((elementCount - nanCount) * 0.99);
    // Sums use the unquantised angle: equal to the float32 run up to the fused multiply-add choices
    // of the two separately compiled kernels (an ulp).
    expectClose(packed.skyViewFactor, reference.skyViewFactor, 1e-6);
    expectClose(packed.positiveOpenness, reference.positiveOpenness, 1e-4);
    expectClose(packed.negativeOpenness, reference.negativeOpenness, 1e-4);
    // Structure: codes are spread over a real range, not all one value.
    expect(new Set(packed.horizonWords).size).toBeGreaterThan(50);
  }
});

it('unorm16 horizon codec round-trips within half a step and flags invalid as 0', () => {
  expect(encodeGPUTerrainHorizonUnorm16(NaN)).toBe(0);
  expect(encodeGPUTerrainHorizonUnorm16(-90)).toBe(1);
  expect(encodeGPUTerrainHorizonUnorm16(90)).toBe(65535);
  expect(encodeGPUTerrainHorizonUnorm16(0)).toBe(32768);
  expect(decodeGPUTerrainHorizonUnorm16(0)).toBeNaN();
  expect(decodeGPUTerrainHorizonUnorm16(32768)).toBe(0);
  expect(decodeGPUTerrainHorizonUnorm16(65535)).toBeCloseTo(90, 9);
  for (let angle = -91; angle <= 91; angle += 0.37) {
    const clamped = Math.min(Math.max(angle, -90), 90);
    const decoded = decodeGPUTerrainHorizonUnorm16(encodeGPUTerrainHorizonUnorm16(angle));
    expect(Math.abs(decoded - clamped)).toBeLessThanOrEqual(
      GPU_TERRAIN_HORIZON_UNORM16_STEP_DEGREES * 0.5 + 1e-6
    );
    expect(decoded).toBeCloseTo(decodeHorizonUnorm16Oracle(encodeHorizonUnorm16Oracle(angle)), 9);
  }
});

it('GPUTerrainHorizon sweep algorithm matches the float64 digital-line oracle for every output', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const width = 40;
  const height = 33;
  const directionCount = 16;
  const elevation = createSmoothTerrain(width, height, 21);
  for (const hole of [3 * width + 5, 3 * width + 6, 17 * width + 22, 32 * width + 39]) {
    elevation[hole] = NaN;
  }
  const anisotropy = {azimuthDegrees: 315, level: 4, minimumWeight: 0.4};
  for (const [maximumRadius, horizonFormat] of [
    [12, 'float32'],
    [60, 'unorm16']
  ] as const) {
    const settings = {
      cellSize: [10, 10] as [number, number],
      zFactor: 1.3,
      curvatureCoefficient: 2e-4,
      anisotropyAzimuthDegrees: anisotropy.azimuthDegrees,
      anisotropyLevel: anisotropy.level,
      anisotropyMinimumWeight: anisotropy.minimumWeight
    };
    const actual = await runExtendedHorizon(
      device,
      elevation,
      {width, height, directionCount, maximumRadius, horizonFormat, algorithm: 'sweep'},
      settings
    );
    const expected = computeTerrainHorizonSweepBruteForce({
      width,
      height,
      elevation,
      directionCount,
      maximumRadius,
      cellSize: [10, 10],
      zFactor: 1.3,
      curvatureCoefficient: 2e-4,
      nadir: true
    });
    const horizonTolerance =
      horizonFormat === 'unorm16' ? GPU_TERRAIN_HORIZON_UNORM16_STEP_DEGREES : 2e-3;
    expectClose(actual.horizon, expected.horizon, horizonTolerance);
    expectClose(actual.skyViewFactor, expected.skyViewFactor, 1e-5);
    expectClose(actual.positiveOpenness, expected.positiveOpenness, 2e-3);
    expectClose(actual.negativeOpenness, expected.negativeOpenness, 2e-3);
    const pixelCount = width * height;
    const expectedAnisotropic = Array.from({length: pixelCount}, (_, pixel) => {
      if (expected.validity[pixel] !== 1) {
        return NaN;
      }
      let weighted = 0;
      let weights = 0;
      for (let sector = 0; sector < directionCount; sector++) {
        const weight = getAnisotropicWeightOracle(sector, directionCount, anisotropy);
        const angle = expected.horizon[pixel * directionCount + sector];
        weighted += weight * Math.sin((Math.max(angle, 0) * Math.PI) / 180);
        weights += weight;
      }
      return 1 - weighted / weights;
    });
    expectClose(actual.anisotropicSkyViewFactor, expectedAnisotropic, 1e-5);
    expect(actual.validity).toEqual(expected.validity);
    // Non-trivial structure: real horizons, both signs, and a spread of sky-view factors.
    const finite = actual.horizon.filter(Number.isFinite);
    expect(Math.max(...finite)).toBeGreaterThan(5);
    expect(Math.min(...finite)).toBeLessThan(-5);
    const svf = actual.skyViewFactor.filter(Number.isFinite);
    expect(Math.max(...svf) - Math.min(...svf)).toBeGreaterThan(0.05);
  }
});
