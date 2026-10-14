// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUReliefShadingParameterValues,
  GPU_RELIEF_SHADING_PARAMETER_LENGTH,
  GPUReliefShading,
  type GPUReliefShadingOutput,
  type GPUReliefShadingProps
} from '../../../src/gpu-terrain/terrain-illumination/gpu-relief-shading';
import {
  getGPUSolarShadowMaskParameterValues,
  GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES,
  GPUSolarShadowMask,
  type GPUSolarShadowMaskOutput,
  type GPUSolarShadowMaskProps
} from '../../../src/gpu-terrain/terrain-illumination/gpu-solar-shadow-mask';
import {
  getGPUTerrainHorizonDirection,
  getGPUTerrainHorizonParameterValues,
  getGPUTerrainHorizonStepDistances,
  getTerrainHorizonSectorOutput,
  GPU_TERRAIN_HORIZON_ANISOTROPIC_PARAMETER_LENGTH,
  GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_AZIMUTH_DEGREES,
  GPU_TERRAIN_HORIZON_PARAMETER_LENGTH,
  GPUTerrainHorizon,
  type GPUTerrainHorizonOutput,
  type GPUTerrainHorizonProps
} from '../../../src/gpu-terrain/terrain-illumination/gpu-terrain-horizon';
import {
  getGPUTextureShadingCascadeSigmas,
  getGPUTextureShadingKernel,
  getGPUTextureShadingParameterValues,
  GPUTextureShading,
  type GPUTextureShadingOutput,
  type GPUTextureShadingProps
} from '../../../src/gpu-terrain/terrain-illumination/gpu-texture-shading';
import {
  getRasterGridGeoreference,
  getRasterGroundCellSize
} from '../../../src/analysis-core/raster-grid';
import {makeGPURasterMetadata} from '../../../src/gpu-raster';
import {getTerrainHorizonReadWGSL} from '../../../src/gpu-terrain/terrain-illumination/terrain-illumination-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  getAnisotropicWeightOracle,
  getVisibleDiskFraction,
  integrateVisibleDiskFraction
} from './terrain-horizon-oracle';
import {computeTextureShading} from './texture-shading-oracle';
import {createBand} from '../terrain-test-utils';

it('GPUTerrainHorizon validates props and schedules one node per sector', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (
    overrides: Partial<Omit<GPUTerrainHorizonProps, 'output'>> = {},
    outputOverrides?: GPUTerrainHorizonOutput
  ) => {
    instance++;
    return new GPUTerrainHorizon({
      grid: makeGPURasterMetadata({width: 6, height: 5}),
      maximumRadius: 4,
      directionCount: 8,
      elevation: createBand(graph, `elevation-${instance}`, 30),
      parameters: createTransientView(graph, `parameters-${instance}`, 'float32', 8),
      output: outputOverrides ?? {
        skyViewFactor: createTransientView(graph, `svf-${instance}`, 'float32', 30)
      },
      ...overrides
    });
  };
  const contributor = create();
  expect(contributor.id).toBe('terrain-horizon');
  expect(contributor.requiredHalo).toBe(4);
  const nodeIds = contributor.getCommandNodes(graph).map(node => node.id);
  // A tiny grid fits every sector into one fused march chunk.
  expect(nodeIds.filter(id => /-horizon-march-\d$/.test(id))).toHaveLength(1);
  expect(nodeIds.filter(id => /-horizon-\d$/.test(id))).toHaveLength(0);
  expect(nodeIds.at(-1)).toBe('terrain-horizon-sky-view');
  expect(() => create({}, {})).toThrow(/at least one output/);
  // A large tile exceeds the dispatch step budget, so the sectors split into several march chunks.
  const wide = create(
    {
      id: 'wide-horizon',
      grid: makeGPURasterMetadata({width: 4096, height: 4096}),
      elevation: createBand(graph, 'wide-elevation', 4096 * 4096),
      maximumRadius: 64
    },
    {skyViewFactor: createTransientView(graph, 'wide-svf', 'float32', 4096 * 4096)}
  );
  expect(
    wide.getCommandNodes(graph).filter(node => /-horizon-march-\d$/.test(node.id))
  ).toHaveLength(8);
  expect(() => create({maximumStepsPerDispatch: 0})).toThrow(/maximumStepsPerDispatch/);
  expect(() => create({directionCount: 3})).toThrow(/directionCount/);
  expect(() => create({directionCount: 65})).toThrow(/directionCount/);
  expect(() => create({maximumRadius: 0})).toThrow(/maximumRadius/);
  expect(() => create({stepGrowth: 0.5})).toThrow(/stepGrowth/);
  expect(() =>
    create({}, {horizon: createTransientView(graph, 'short-horizon', 'float32', 30)})
  ).toThrow(/240 float32/);
  expect(() =>
    create({
      parameters: createTransientView(graph, 'parameters-7', 'float32', 7)
    })
  ).toThrow(/parameters/);
  expect(() => create({cellSizeMode: 'polar' as never})).toThrow(/cellSizeMode/);
  expect(() => create({rowDirection: 'east' as never})).toThrow(/rowDirection/);
  expect(contributor.algorithm).toBe('march');
  expect(() => create({algorithm: 'pyramid' as never})).toThrow(/algorithm/);
  expect(() => create({algorithm: 'sweep', stepGrowth: 1.2})).toThrow(/march algorithm only/);
  // Geographic sweeps derive their line family per frame from the reference row's cell size.
  const geographicSweep = create({
    id: 'geographic-sweep',
    algorithm: 'sweep',
    cellSizeMode: 'geographic'
  });
  expect(
    geographicSweep.getCommandNodes(graph).filter(node => /-horizon-\d$/.test(node.id))
  ).toHaveLength(8);
  const sweep = create(
    {id: 'sweep-horizon', algorithm: 'sweep'},
    {
      skyViewFactor: createTransientView(graph, 'sweep-svf', 'float32', 30),
      negativeOpenness: createTransientView(graph, 'sweep-negative', 'float32', 30)
    }
  );
  const sweepIds = sweep.getCommandNodes(graph).map(node => node.id);
  // One zenith and one nadir sweep node per sector.
  expect(sweepIds.filter(id => /-horizon-\d$/.test(id))).toHaveLength(8);
  expect(sweepIds.filter(id => /-horizon-\d-nadir$/.test(id))).toHaveLength(8);
  const shared = createBand(graph, 'shared', 30);
  expect(() => create({elevation: shared}, {skyViewFactor: shared.storage.values})).toThrow(
    /share buffers/
  );

  expect(Array.from(getGPUTerrainHorizonStepDistances(5))).toEqual([1, 2, 3, 4, 5]);
  expect(Array.from(getGPUTerrainHorizonStepDistances(10, 1.5))).toEqual([
    1,
    2,
    3,
    4.5,
    Math.fround(6.75),
    10
  ]);
  expect(getGPUTerrainHorizonDirection(0, 8)).toEqual([0, -1]);
  expect(getGPUTerrainHorizonDirection(2, 8)).toEqual([1, 0]);
  expect(getGPUTerrainHorizonDirection(4, 8, 'north')).toEqual([0, -1]);
  expect(
    Array.from(
      getGPUTerrainHorizonParameterValues(
        {},
        makeGPURasterMetadata({width: 6, height: 5, cellSize: [2, 3]})
      )
    )
  ).toEqual([2, 3, 1, 0, 0, -15, 0, 0]);
  expect(() =>
    getGPUTerrainHorizonParameterValues(
      {},
      makeGPURasterMetadata({width: 6, height: 5}),
      new Float32Array(7)
    )
  ).toThrow(/8 values/);
  device.destroy();
});

it('GPUSolarShadowMask validates props and uses the circular-segment disk fraction', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (
    overrides: Partial<Omit<GPUSolarShadowMaskProps, 'output'>> = {},
    outputOverrides?: GPUSolarShadowMaskOutput
  ) => {
    instance++;
    return new GPUSolarShadowMask({
      grid: makeGPURasterMetadata({width: 4, height: 3}),
      directionCount: 8,
      horizon: createTransientView(graph, `horizon-${instance}`, 'float32', 96),
      parameters: createTransientView(graph, `parameters-${instance}`, 'float32', 8),
      output: outputOverrides ?? {
        sunVisibility: createTransientView(graph, `visibility-${instance}`, 'float32', 12)
      },
      ...overrides
    });
  };
  expect(
    create()
      .getCommandNodes(graph)
      .map(node => node.id)
  ).toEqual(['solar-shadow-mask-shadow']);
  expect(() => create({}, {})).toThrow(/at least one output/);
  expect(() => create({directionCount: 2})).toThrow(/directionCount/);
  expect(() => create({horizon: createTransientView(graph, 'h-short', 'float32', 95)})).toThrow(
    /96 float32/
  );
  expect(() => create({slope: createTransientView(graph, 'slope-only', 'float32', 12)})).toThrow(
    /together/
  );
  expect(() =>
    create({}, {illumination: createTransientView(graph, 'illumination-only', 'float32', 12)})
  ).toThrow(/requires slope/);
  const horizon = createTransientView(graph, 'aliased-horizon', 'float32', 96);
  expect(() => create({horizon}, {sunVisibility: horizon as never})).toThrow();
  expect(
    Array.from(
      getGPUSolarShadowMaskParameterValues({
        azimuthDegrees: 90,
        altitudeDegrees: 10
      })
    )
  ).toEqual([90, 10, Math.fround(GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES), 1, 0, 0, 0, 0]);
  expect(() =>
    getGPUSolarShadowMaskParameterValues({
      azimuthDegrees: 0,
      altitudeDegrees: 0,
      angularRadiusDegrees: -1
    })
  ).toThrow(/angular radius/);
  for (const offset of [-1.2, -0.9, -0.5, -0.1, 0, 0.3, 0.75, 0.99, 1.5]) {
    expect(getVisibleDiskFraction(10 + offset * 0.27, 10, 0.27)).toBeCloseTo(
      integrateVisibleDiskFraction(10 + offset * 0.27, 10, 0.27),
      3
    );
  }
  device.destroy();
});

it('GPUReliefShading validates props and packs MDOW and custom lights', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (
    overrides: Partial<Omit<GPUReliefShadingProps, 'output'>> = {},
    outputOverrides?: GPUReliefShadingOutput
  ) => {
    instance++;
    return new GPUReliefShading({
      grid: makeGPURasterMetadata({width: 6, height: 5}),
      elevation: createBand(graph, `elevation-${instance}`, 30),
      parameters: createTransientView(graph, `parameters-${instance}`, 'float32', 80),
      output: outputOverrides ?? {
        hillshade: createTransientView(graph, `hillshade-${instance}`, 'float32', 30)
      },
      ...overrides
    });
  };
  const contributor = create();
  expect(contributor.requiredHalo).toBe(1);
  expect(contributor.getCommandNodes(graph).map(node => node.id)).toEqual([
    'relief-shading-gradient-x',
    'relief-shading-gradient-y',
    'relief-shading-hillshade'
  ]);
  expect(
    create(
      {id: 'colored'},
      {
        hillshade: createTransientView(graph, 'colored-hillshade', 'float32', 30),
        color: createTransientView(graph, 'color', 'uint32', 30)
      }
    )
      .getCommandNodes(graph)
      .map(node => node.id)
      .at(-1)
  ).toBe('colored-compose');
  expect(() => create({}, {})).toThrow(/at least one output/);
  expect(() =>
    create({}, {relief: createTransientView(graph, 'short-relief', 'float32', 29)})
  ).toThrow(/30 float32/);
  expect(() =>
    create({
      parameters: createTransientView(graph, 'parameters-79', 'float32', 79)
    })
  ).toThrow(/parameters/);
  const svf = createTransientView(graph, 'svf-alias', 'float32', 30);
  expect(() => create({skyViewFactor: svf}, {hillshade: svf})).toThrow(/share buffers/);

  const grid = makeGPURasterMetadata({width: 6, height: 5, cellSize: [10, 12]});
  const mdow = getGPUReliefShadingParameterValues({}, grid);
  // Cell size and row edges come from the grid.
  const georeference = getRasterGridGeoreference(grid);
  expect(Array.from(mdow.subarray(0, 2))).toEqual([...georeference.cellSize]);
  expect(Array.from(mdow.subarray(3, 5))).toEqual([georeference.northEdge, georeference.southEdge]);
  expect(mdow.length).toBe(GPU_RELIEF_SHADING_PARAMETER_LENGTH);
  expect(mdow[5]).toBe(4);
  expect(mdow[6]).toBe(1);
  expect(Array.from(mdow.subarray(20, 32))).toEqual([
    225, 30, 1, 270, 30, 1, 315, 30, 1, 360, 30, 1
  ]);
  const custom = getGPUReliefShadingParameterValues(
    {
      lights: [{azimuthDegrees: 100, weight: 2}],
      elevationStops: [
        {elevation: 0, color: [0, 0.5, 0]},
        {elevation: 10, color: [1, 1, 1]}
      ]
    },
    grid
  );
  expect(custom[5]).toBe(1);
  expect(custom[6]).toBe(0);
  expect(Array.from(custom.subarray(20, 23))).toEqual([100, 45, 2]);
  expect(custom[15]).toBe(2);
  expect(Array.from(custom.subarray(44, 52))).toEqual([0, 0, 0.5, 0, 10, 1, 1, 1]);
  const tooMany = Array.from({length: 9}, (_, index) => ({
    azimuthDegrees: index * 40
  }));
  expect(() => getGPUReliefShadingParameterValues({lights: tooMany}, grid)).toThrow(
    /1 to 8 lights/
  );
  expect(() =>
    getGPUReliefShadingParameterValues(
      {
        elevationStops: [
          {elevation: 5, color: [0, 0, 0]},
          {elevation: 5, color: [1, 1, 1]}
        ]
      },
      grid
    )
  ).toThrow(/ascending/);
  device.destroy();
});

it('GPUTextureShading validates props and approximates a fractional Laplacian', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (
    overrides: Partial<Omit<GPUTextureShadingProps, 'output'>> = {},
    outputOverrides?: GPUTextureShadingOutput
  ) => {
    instance++;
    return new GPUTextureShading({
      grid: makeGPURasterMetadata({width: 6, height: 5}),
      levelCount: 3,
      elevation: createBand(graph, `elevation-${instance}`, 30),
      parameters: createTransientView(graph, `parameters-${instance}`, 'float32', 12),
      output: outputOverrides ?? {
        textureShade: createTransientView(graph, `shade-${instance}`, 'float32', 30)
      },
      ...overrides
    });
  };
  const contributor = create();
  const radii = getGPUTextureShadingCascadeSigmas(3, 1).map(sigma => Math.ceil(3 * sigma));
  expect(contributor.requiredHalo).toBe(radii.reduce((sum, radius) => sum + radius, 0));
  expect(
    contributor
      .getCommandNodes(graph)
      .map(node => node.id)
      .at(-1)
  ).toBe('texture-shading-finalize');
  expect(() => create({}, {})).toThrow(/at least one output/);
  expect(() => create({levelCount: 9})).toThrow(/levelCount/);
  expect(() => create({baseSigma: 0})).toThrow(/baseSigma/);
  expect(() =>
    create({
      parameters: createTransientView(graph, 'parameters-11', 'float32', 11)
    })
  ).toThrow(/parameters/);
  const kernel = getGPUTextureShadingKernel(2);
  expect(kernel.length).toBe(13);
  expect(kernel.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 6);
  expect(getGPUTextureShadingCascadeSigmas(3, 1).map(sigma => sigma ** 2)).toEqual([
    1,
    expect.closeTo(3, 10),
    expect.closeTo(12, 10)
  ]);
  const packed = getGPUTextureShadingParameterValues({detail: 1, gain: 2});
  expect(packed[0]).toBe(2);
  expect(Array.from(packed.subarray(1, 4))).toEqual([1, 0.5, 0.25]);
  expect(() => getGPUTextureShadingParameterValues({levelWeights: new Array(9).fill(1)})).toThrow(
    /at most 8/
  );

  // Response to sinusoids grows like |f|^alpha across the pyramid's mid band.
  const width = 1024;
  const getGain = (frequency: number, detail: number) => {
    const signal = Float32Array.from({length: width}, (_, x) =>
      Math.sin(2 * Math.PI * frequency * x)
    );
    const response = computeTextureShading({
      width,
      height: 1,
      elevation: signal,
      levelCount: 6,
      baseSigma: 1,
      gain: 1,
      weights: Array.from(getGPUTextureShadingParameterValues({detail}).subarray(1, 7))
    });
    return Math.max(...response.slice(width / 4, (3 * width) / 4).map(Math.abs));
  };
  for (const detail of [0.5, 1]) {
    for (const [low, high] of [
      [1 / 48, 1 / 24],
      [1 / 24, 1 / 12]
    ]) {
      const slope = Math.log(getGain(high, detail) / getGain(low, detail)) / Math.log(high / low);
      expect(Math.abs(slope - detail)).toBeLessThan(0.2);
    }
  }
  device.destroy();
});

it('getRasterGroundCellSize scales Web Mercator and geographic rows', () => {
  expect(getRasterGroundCellSize('uniform', {cellSize: [3, 4]}, 2.5, 10)).toEqual([3, 4]);
  // WGS84 radii of curvature at the row latitude (python3 -I, closed form, a = 6378137,
  // f = 1 / 298.257223563, e2 = f (2 - f), w = 1 - e2 sin^2(lat), N = a / sqrt(w),
  // M = a (1 - e2) / w^1.5): at 60 degrees N cos(lat) PI / 180 = 55800.001572436144 m and
  // M PI / 180 = 111412.28745777943 m per degree (the equatorial sphere gave 55659.75 / 111319.49).
  const [x, y] = getRasterGroundCellSize(
    'geographic',
    {cellSize: [0.001, 0.001], northEdge: 60, southEdge: 60},
    0.5,
    1
  );
  expect(x).toBeCloseTo(55.800001572436144, 9);
  expect(y).toBeCloseTo(111.41228745777943, 9);
  // EPSG:3857 y = 0 is the equator: x = N(0) / a = 1, y = M(0) / a =
  // 1 - e2 = 0.9933056200098586 (python3 -I, same formulas).
  const mercator = getRasterGroundCellSize(
    'web-mercator',
    {cellSize: [10, 10], northEdge: 0, southEdge: 0},
    0.5,
    1
  );
  expect(mercator[0]).toBeCloseTo(10, 6);
  expect(mercator[1]).toBeCloseTo(9.933056200098586, 9);
});

it('GPUTerrainHorizon validates unorm16, anisotropic, and negative-openness options', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const grid = makeGPURasterMetadata({width: 6, height: 5, cellSize: [2, 3]});
  let instance = 0;
  const create = (
    overrides: Partial<Omit<GPUTerrainHorizonProps, 'output'>> = {},
    outputOverrides?: GPUTerrainHorizonOutput
  ) => {
    instance++;
    return new GPUTerrainHorizon({
      grid,
      maximumRadius: 4,
      directionCount: 7,
      elevation: createBand(graph, `elevation-${instance}`, 30),
      parameters: createTransientView(graph, `parameters-${instance}`, 'float32', 12),
      output: outputOverrides ?? {
        skyViewFactor: createTransientView(graph, `svf-${instance}`, 'float32', 30)
      },
      ...overrides
    });
  };
  // Defaults are unchanged: float32 horizon storage and 8 parameters.
  expect(create().horizonFormat).toBe('float32');
  expect(GPU_TERRAIN_HORIZON_PARAMETER_LENGTH).toBe(8);
  expect(GPU_TERRAIN_HORIZON_ANISOTROPIC_PARAMETER_LENGTH).toBe(12);
  expect(() => create({horizonFormat: 'float16' as never})).toThrow(/horizonFormat/);
  // 30 pixels * 7 sectors = 210 elements = 105 packed words.
  const packed = createTransientView(graph, 'packed', 'uint32', 105);
  const contributor = create({horizonFormat: 'unorm16'}, {horizon: packed});
  expect(contributor.horizonFormat).toBe('unorm16');
  expect(() =>
    create(
      {horizonFormat: 'unorm16'},
      {horizon: createTransientView(graph, 'p-short', 'uint32', 104)}
    )
  ).toThrow(/105 uint32/);
  expect(() =>
    create(
      {horizonFormat: 'unorm16'},
      {horizon: createTransientView(graph, 'p-float', 'float32', 105)}
    )
  ).toThrow(/output.horizon/);
  expect(() => create({}, {horizon: createTransientView(graph, 'f-words', 'uint32', 210)})).toThrow(
    /output.horizon/
  );
  expect(() =>
    create(
      {horizonFormat: 'unorm16'},
      {horizon: createTransientView(graph, 'p-full', 'uint32', 210)}
    )
  ).toThrow(/105 uint32/);

  const anisotropic = createTransientView(graph, 'anisotropic', 'float32', 30);
  expect(() =>
    create(
      {parameters: createTransientView(graph, 'short-parameters', 'float32', 8)},
      {anisotropicSkyViewFactor: anisotropic}
    )
  ).toThrow(/at least 12 values/);
  expect(() =>
    create(
      {},
      {anisotropicSkyViewFactor: createTransientView(graph, 'anisotropic-short', 'float32', 29)}
    )
  ).toThrow(/30 float32/);
  expect(() => create({}, {})).toThrow(/at least one output/);
  // A negative-openness-only contributor is a valid output set.
  const everything = create(
    {horizonFormat: 'unorm16'},
    {
      skyViewFactor: createTransientView(graph, 'all-svf', 'float32', 30),
      positiveOpenness: createTransientView(graph, 'all-positive', 'float32', 30),
      negativeOpenness: createTransientView(graph, 'all-negative', 'float32', 30),
      anisotropicSkyViewFactor: createTransientView(graph, 'all-anisotropic', 'float32', 30),
      outputValidity: createTransientView(graph, 'all-validity', 'uint32', 30),
      horizon: createTransientView(graph, 'all-horizon', 'uint32', 105)
    }
  );
  const nodeIds = everything.getCommandNodes(graph).map(node => node.id);
  expect(nodeIds.filter(id => /-horizon-\d$/.test(id))).toHaveLength(7);
  expect(nodeIds).toContain('terrain-horizon-sky-view');
  expect(nodeIds).toContain('terrain-horizon-extended');
  // The same buffer cannot back two outputs.
  const shared = createTransientView(graph, 'shared-output', 'float32', 30);
  expect(() => create({}, {skyViewFactor: shared, negativeOpenness: shared})).toThrow(
    /share buffers/
  );

  // Parameter packing: 8 values unless anisotropy is given or the target holds 12; the cell size
  // and row edges come from the grid (3 x 5 rows south of y = 0).
  expect(getGPUTerrainHorizonParameterValues({}, grid)).toHaveLength(8);
  const withAnisotropy = getGPUTerrainHorizonParameterValues(
    {anisotropyAzimuthDegrees: 45, anisotropyLevel: 8, anisotropyMinimumWeight: 0.1},
    grid
  );
  expect(Array.from(withAnisotropy)).toEqual([
    2,
    3,
    1,
    0,
    0,
    -15,
    0,
    0,
    45,
    8,
    Math.fround(0.1),
    0
  ]);
  expect(Array.from(getGPUTerrainHorizonParameterValues({}, grid, new Float32Array(12)))).toEqual([
    2,
    3,
    1,
    0,
    0,
    -15,
    0,
    0,
    315,
    4,
    Math.fround(0.4),
    0
  ]);
  const georeference = getRasterGridGeoreference(grid);
  expect(Array.from(withAnisotropy.subarray(0, 2))).toEqual([...georeference.cellSize]);
  expect(Array.from(withAnisotropy.subarray(4, 6))).toEqual([
    georeference.northEdge,
    georeference.southEdge
  ]);
  expect(() =>
    getGPUTerrainHorizonParameterValues({anisotropyLevel: 4}, grid, new Float32Array(8))
  ).toThrow(/12 values/);
  expect(() => getGPUTerrainHorizonParameterValues({}, grid, new Float32Array(7))).toThrow(
    /must hold 8 values/
  );
  expect(() => getGPUTerrainHorizonParameterValues({anisotropyMinimumWeight: 1.5}, grid)).toThrow(
    /minimum weight/
  );
  expect(() => getGPUTerrainHorizonParameterValues({anisotropyLevel: -1}, grid)).toThrow(/level/);
  device.destroy();
});

it('anisotropic sky-view weights use RVT asvf directions, default and signed integer levels', () => {
  // python3 -I, rvt/vis.py sky_view_factor_compute: RVT direction i has angle a_i = i * 360 / n and
  // np.roll(height, (round(cos a * r), round(sin a * r)), axis=(0, 1)) samples north = cos a,
  // east = -sin a, i.e. compass (360 - a) mod 360 (counterclockwise from north). Weights
  // w_i = (1 - wm) * cos((a_i - A) / 2) ** level + wm (signed), listed per compass sector d = RVT
  // direction (n - d) % n.
  const expected: [number, number, number, number, number[]][] = [
    [
      16,
      315,
      4,
      0.4,
      [
        0.8371320343559643, 0.9551968683423773, 1.0, 0.9551968683423773, 0.8371320343559643,
        0.6867720211205357, 0.55, 0.45716196170148204, 0.4128679656440358, 0.40086914883560504, 0.4,
        0.40086914883560504, 0.4128679656440358, 0.45716196170148193, 0.5499999999999999,
        0.686772021120536
      ]
    ],
    [
      16,
      45,
      8,
      0.1,
      [
        0.5777110386504597, 0.30559548024389227, 0.15624999999999997, 0.10816872466390426,
        0.10041396134954021, 0.10000188854924609, 0.1, 0.10000188854924609, 0.10041396134954021,
        0.10816872466390426, 0.15625000000000003, 0.3055954802438927, 0.5777110386504597,
        0.8706089065429569, 1.0, 0.8706089065429569
      ]
    ],
    // Odd level: RVT's signed power drops sector 1 below the minimum weight 0.2.
    [
      16,
      100,
      3,
      0.2,
      [
        0.41246748505503594, 0.11097852329608822, 0.17824718835568318, 0.19896776814945358,
        0.20052963662809076, 0.2175294469227998, 0.2787601956662583, 0.3961795350038978,
        0.5596266658713869, 0.7391166792705333, 0.8939824562506804, 0.9858198288324129,
        0.990901984112861, 0.907899075651571, 0.7583117383509435, 0.5794729518119288
      ]
    ],
    [16, 100, 0, 0, new Array(16).fill(1)]
  ];
  for (const [directionCount, azimuthDegrees, level, minimumWeight, weights] of expected) {
    for (let sector = 0; sector < directionCount; sector++) {
      expect(
        getAnisotropicWeightOracle(sector, directionCount, {azimuthDegrees, level, minimumWeight})
      ).toBeCloseTo(weights[sector], 12);
    }
  }
  // RVT's default 315 is compass north-east: the maximum weight is sector 2 of 16 (45 degrees).
  // Before the RVT convention the same number put it at sector 14 (compass 315, north-west).
  expect(GPU_TERRAIN_HORIZON_DEFAULT_ANISOTROPY_AZIMUTH_DEGREES).toBe(315);
  expect(expected[0][4].indexOf(Math.max(...expected[0][4]))).toBe(2);

  // The kernel maps compass sectors to RVT directions and keeps the sign of odd levels.
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const output = getTerrainHorizonSectorOutput({
    sector: 0,
    directionCount: 16,
    anisotropicSum: createTransientView(graph, 'w', 'float32', 16),
    mode: 'zenith'
  });
  expect(output.declarations).toContain('f32((16u - sector) % 16u) * 22.5');
  expect(output.declarations).toContain('cos((rvtDirection - mainDirection) * 0.5');
  expect(output.declarations).toContain('fract(level * 0.5) != 0.0');
  expect(output.declarations).not.toContain('abs(cos(');

  // Levels are RVT's integer a_poly_level; the packed layout is unchanged.
  const grid = makeGPURasterMetadata({width: 6, height: 5, cellSize: [2, 3]});
  for (const anisotropyLevel of [1.5, -1, Number.NaN]) {
    expect(() => getGPUTerrainHorizonParameterValues({anisotropyLevel}, grid)).toThrow(
      /non-negative integer/
    );
  }
  expect(Array.from(getGPUTerrainHorizonParameterValues({anisotropyLevel: 3}, grid))).toEqual([
    2,
    3,
    1,
    0,
    0,
    -15,
    0,
    0,
    315,
    3,
    Math.fround(0.4),
    0
  ]);
  device.destroy();
});

it('getTerrainHorizonSectorOutput emits zenith and nadir accumulations and checks bindings', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = (id: string) => createTransientView(graph, id, 'float32', 16);
  const wordView = (id: string) => createTransientView(graph, id, 'uint32', 16);
  const first = getTerrainHorizonSectorOutput({
    sector: 0,
    directionCount: 8,
    horizon: view('h'),
    sineSum: view('s'),
    angleSum: view('a'),
    anisotropicSum: view('w'),
    mode: 'zenith'
  });
  expect(first.bindings.map(binding => binding.name)).toEqual([
    'horizon',
    'sineSum',
    'angleSum',
    'anisotropicSum'
  ]);
  expect(first.wgsl).toContain('sineSum[sineSumOffset + pixel] = sin(');
  expect(first.wgsl).toContain('getAnisotropicWeight(0u)');
  expect(first.declarations).toContain('fn getAnisotropicWeight');
  const later = getTerrainHorizonSectorOutput({
    sector: 3,
    directionCount: 8,
    sineSum: view('s2'),
    mode: 'zenith'
  });
  expect(later.wgsl).toContain('sineSum[sineSumOffset + pixel] += sin(');
  const packed = getTerrainHorizonSectorOutput({
    sector: 3,
    directionCount: 7,
    horizonFormat: 'unorm16',
    horizon: wordView('hp'),
    mode: 'zenith'
  });
  expect(packed.bindings[0].type).toBe('u32');
  expect(packed.declarations).toContain('fn encodeHorizonUnorm16');
  expect(packed.wgsl).toContain('pixel * 7u + 3u');
  const nadir = getTerrainHorizonSectorOutput({
    sector: 2,
    directionCount: 8,
    nadirSum: view('n'),
    mode: 'nadir'
  });
  expect(nadir.bindings.map(binding => binding.name)).toEqual(['nadirSum']);
  expect(nadir.wgsl).toContain('nadirSum[nadirSumOffset + pixel] += horizonAngle');
  expect(() =>
    getTerrainHorizonSectorOutput({sector: 0, directionCount: 8, sineSum: view('x'), mode: 'nadir'})
  ).toThrow(/only accumulates nadirSum/);
  expect(() =>
    getTerrainHorizonSectorOutput({
      sector: 0,
      directionCount: 8,
      nadirSum: view('y'),
      mode: 'zenith'
    })
  ).toThrow(/cannot write nadirSum/);
  expect(() =>
    getTerrainHorizonSectorOutput({sector: 0, directionCount: 8, mode: 'zenith'})
  ).toThrow(/at least one output/);
  expect(() =>
    getTerrainHorizonSectorOutput({
      sector: 0,
      directionCount: 8,
      horizon: view('h9'),
      sineSum: view('s9'),
      angleSum: view('a9'),
      anisotropicSum: view('w9'),
      mode: 'zenith',
      reservedBindingCount: 5
    })
  ).toThrow(/needs 9 storage bindings/);
  expect(getTerrainHorizonReadWGSL('float32', 'horizon')).toContain(
    'horizon[horizonOffset + element]'
  );
  expect(getTerrainHorizonReadWGSL('unorm16', 'horizon')).toContain('code == 0u');
  device.destroy();
});

it('review C-14: texture shading blurs elevation relative to a per-grid anchor', () => {
  for (const hasNodata of [true, false]) {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    const nodes = new GPUTextureShading({
      grid: makeGPURasterMetadata({width: 6, height: 5}),
      levelCount: 3,
      hasNodata,
      elevation: createBand(graph, 'elevation', 30),
      parameters: createTransientView(graph, 'parameters', 'float32', 12),
      output: {textureShade: createTransientView(graph, 'shade', 'float32', 30)}
    }).getCommandNodes(graph);
    const ids = nodes.map(node => node.id);
    expect(ids.indexOf('texture-shading-anchor')).toBe(
      ids.indexOf('texture-shading-premultiply') - 1
    );
  }
  // f32 twin of one band difference at 4000 m: blur(z) - z with a 5-tap normalized kernel.
  const f = Math.fround;
  const kernel = [1, 4, 6, 4, 1].map(weight => f(weight / 16));
  const relief = [0.013, -0.021, 0.008, 0.017, -0.004];
  const getBand = (offset: number) => {
    let blurred = f(0);
    for (const [tap, weight] of kernel.entries()) {
      blurred = f(blurred + f(weight * f(4000 + relief[tap] - offset)));
    }
    return f(blurred - f(4000 + relief[2] - offset));
  };
  const exact =
    kernel.reduce((sum, weight, tap) => sum + (weight * relief[tap]) / 1, 0) - relief[2];
  // Absolute: about one ulp(4000) = 2.4e-4 m of noise; anchored on 4000 m: below 1e-8 m.
  expect(Math.abs(getBand(0) - exact)).toBeGreaterThan(1e-5);
  expect(Math.abs(getBand(4000) - exact)).toBeLessThan(1e-8);
});
