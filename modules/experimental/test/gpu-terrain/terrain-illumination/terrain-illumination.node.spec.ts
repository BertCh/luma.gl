// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUReliefShadingParameterValues,
  GPU_RELIEF_SHADING_PARAMETER_LENGTH,
  GPUReliefShading,
  type GPUReliefShadingProps
} from '../../../src/gpu-terrain/terrain-illumination/gpu-relief-shading';
import {
  getGPUSolarShadowMaskParameterValues,
  GPU_SOLAR_DISK_ANGULAR_RADIUS_DEGREES,
  GPUSolarShadowMask,
  type GPUSolarShadowMaskProps
} from '../../../src/gpu-terrain/terrain-illumination/gpu-solar-shadow-mask';
import {
  getGPUTerrainHorizonDirection,
  getGPUTerrainHorizonParameterValues,
  getGPUTerrainHorizonStepDistances,
  getTerrainHorizonSectorOutput,
  GPU_TERRAIN_HORIZON_ANISOTROPIC_PARAMETER_LENGTH,
  GPU_TERRAIN_HORIZON_PARAMETER_LENGTH,
  GPUTerrainHorizon,
  type GPUTerrainHorizonProps
} from '../../../src/gpu-terrain/terrain-illumination/gpu-terrain-horizon';
import {
  getGPUTextureShadingCascadeSigmas,
  getGPUTextureShadingKernel,
  getGPUTextureShadingParameterValues,
  GPUTextureShading,
  type GPUTextureShadingProps
} from '../../../src/gpu-terrain/terrain-illumination/gpu-texture-shading';
import {
  getTerrainHorizonReadWGSL,
  getTerrainIlluminationGroundCellSize
} from '../../../src/gpu-terrain/terrain-illumination/terrain-illumination-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {getVisibleDiskFraction, integrateVisibleDiskFraction} from './terrain-horizon-oracle';
import {computeTextureShading} from './texture-shading-oracle';

function createBand(graph: GPUCommandGraph, id: string, length: number) {
  return {
    id,
    format: 'float32' as const,
    storage: {
      kind: 'buffer' as const,
      values: createTransientView(graph, id, 'float32', length)
    }
  };
}

it('GPUTerrainHorizon validates props and schedules one node per sector', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Partial<GPUTerrainHorizonProps> = {}) => {
    instance++;
    return new GPUTerrainHorizon({
      width: 6,
      height: 5,
      maximumRadius: 4,
      directionCount: 8,
      elevation: createBand(graph, `elevation-${instance}`, 30),
      settings: createTransientView(graph, `settings-${instance}`, 'float32', 8),
      skyViewFactor: createTransientView(graph, `svf-${instance}`, 'float32', 30),
      ...overrides
    });
  };
  const recipe = create();
  expect(recipe.recipe).toBe('terrain-horizon');
  expect(recipe.requiredHalo).toBe(4);
  const nodeIds = recipe.getCommandNodes(graph).map(node => node.id);
  expect(nodeIds.filter(id => /-horizon-\d$/.test(id))).toHaveLength(8);
  expect(nodeIds.at(-1)).toBe('terrain-horizon-sky-view');
  expect(() => create({skyViewFactor: undefined})).toThrow(/at least one output/);
  expect(() => create({directionCount: 3})).toThrow(/directionCount/);
  expect(() => create({directionCount: 65})).toThrow(/directionCount/);
  expect(() => create({maximumRadius: 0})).toThrow(/maximumRadius/);
  expect(() => create({stepGrowth: 0.5})).toThrow(/stepGrowth/);
  expect(() =>
    create({
      horizon: createTransientView(graph, 'short-horizon', 'float32', 30)
    })
  ).toThrow(/240 float32/);
  expect(() =>
    create({
      settings: createTransientView(graph, 'settings-7', 'float32', 7)
    })
  ).toThrow(/settings/);
  expect(() => create({cellSizeMode: 'polar' as never})).toThrow(/cellSizeMode/);
  expect(() => create({rowDirection: 'east' as never})).toThrow(/rowDirection/);
  const shared = createBand(graph, 'shared', 30);
  expect(() => create({elevation: shared, skyViewFactor: shared.storage.values})).toThrow(
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
  expect(Array.from(getGPUTerrainHorizonParameterValues({cellSize: [2, 3]}))).toEqual([
    2, 3, 1, 0, 0, 0, 0, 0
  ]);
  expect(() => getGPUTerrainHorizonParameterValues({cellSize: [0, 1]})).toThrow(/cell size/);
  device.destroy();
});

it('GPUSolarShadowMask validates props and uses the circular-segment disk fraction', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Partial<GPUSolarShadowMaskProps> = {}) => {
    instance++;
    return new GPUSolarShadowMask({
      width: 4,
      height: 3,
      directionCount: 8,
      horizon: createTransientView(graph, `horizon-${instance}`, 'float32', 96),
      settings: createTransientView(graph, `settings-${instance}`, 'float32', 8),
      sunVisibility: createTransientView(graph, `visibility-${instance}`, 'float32', 12),
      ...overrides
    });
  };
  expect(
    create()
      .getCommandNodes(graph)
      .map(node => node.id)
  ).toEqual(['solar-shadow-mask-shadow']);
  expect(() => create({sunVisibility: undefined})).toThrow(/at least one output/);
  expect(() => create({directionCount: 2})).toThrow(/directionCount/);
  expect(() => create({horizon: createTransientView(graph, 'h-short', 'float32', 95)})).toThrow(
    /96 float32/
  );
  expect(() => create({slope: createTransientView(graph, 'slope-only', 'float32', 12)})).toThrow(
    /together/
  );
  expect(() =>
    create({
      illumination: createTransientView(graph, 'illumination-only', 'float32', 12)
    })
  ).toThrow(/requires slope/);
  const horizon = createTransientView(graph, 'aliased-horizon', 'float32', 96);
  expect(() => create({horizon, sunVisibility: horizon as never})).toThrow();
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
  const create = (overrides: Partial<GPUReliefShadingProps> = {}) => {
    instance++;
    return new GPUReliefShading({
      width: 6,
      height: 5,
      elevation: createBand(graph, `elevation-${instance}`, 30),
      settings: createTransientView(graph, `settings-${instance}`, 'float32', 80),
      hillshade: createTransientView(graph, `hillshade-${instance}`, 'float32', 30),
      ...overrides
    });
  };
  const recipe = create();
  expect(recipe.requiredHalo).toBe(1);
  expect(recipe.getCommandNodes(graph).map(node => node.id)).toEqual([
    'relief-shading-gradient-x',
    'relief-shading-gradient-y',
    'relief-shading-hillshade'
  ]);
  expect(
    create({
      id: 'colored',
      color: createTransientView(graph, 'color', 'uint32', 30)
    })
      .getCommandNodes(graph)
      .map(node => node.id)
      .at(-1)
  ).toBe('colored-compose');
  expect(() => create({hillshade: undefined})).toThrow(/at least one output/);
  expect(() =>
    create({
      relief: createTransientView(graph, 'short-relief', 'float32', 29)
    })
  ).toThrow(/30 float32/);
  expect(() =>
    create({
      settings: createTransientView(graph, 'settings-79', 'float32', 79)
    })
  ).toThrow(/settings/);
  const svf = createTransientView(graph, 'svf-alias', 'float32', 30);
  expect(() => create({skyViewFactor: svf, hillshade: svf})).toThrow(/share buffers/);

  const mdow = getGPUReliefShadingParameterValues({cellSize: [10, 10]});
  expect(mdow.length).toBe(GPU_RELIEF_SHADING_PARAMETER_LENGTH);
  expect(mdow[5]).toBe(4);
  expect(mdow[6]).toBe(1);
  expect(Array.from(mdow.subarray(20, 32))).toEqual([
    225, 30, 1, 270, 30, 1, 315, 30, 1, 360, 30, 1
  ]);
  const custom = getGPUReliefShadingParameterValues({
    cellSize: [10, 10],
    lights: [{azimuthDegrees: 100, weight: 2}],
    elevationStops: [
      {elevation: 0, color: [0, 0.5, 0]},
      {elevation: 10, color: [1, 1, 1]}
    ]
  });
  expect(custom[5]).toBe(1);
  expect(custom[6]).toBe(0);
  expect(Array.from(custom.subarray(20, 23))).toEqual([100, 45, 2]);
  expect(custom[15]).toBe(2);
  expect(Array.from(custom.subarray(44, 52))).toEqual([0, 0, 0.5, 0, 10, 1, 1, 1]);
  const tooMany = Array.from({length: 9}, (_, index) => ({
    azimuthDegrees: index * 40
  }));
  expect(() => getGPUReliefShadingParameterValues({cellSize: [1, 1], lights: tooMany})).toThrow(
    /1 to 8 lights/
  );
  expect(() =>
    getGPUReliefShadingParameterValues({
      cellSize: [1, 1],
      elevationStops: [
        {elevation: 5, color: [0, 0, 0]},
        {elevation: 5, color: [1, 1, 1]}
      ]
    })
  ).toThrow(/ascending/);
  device.destroy();
});

it('GPUTextureShading validates props and approximates a fractional Laplacian', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Partial<GPUTextureShadingProps> = {}) => {
    instance++;
    return new GPUTextureShading({
      width: 6,
      height: 5,
      levelCount: 3,
      elevation: createBand(graph, `elevation-${instance}`, 30),
      settings: createTransientView(graph, `settings-${instance}`, 'float32', 12),
      textureShade: createTransientView(graph, `shade-${instance}`, 'float32', 30),
      ...overrides
    });
  };
  const recipe = create();
  const radii = getGPUTextureShadingCascadeSigmas(3, 1).map(sigma => Math.ceil(3 * sigma));
  expect(recipe.requiredHalo).toBe(radii.reduce((sum, radius) => sum + radius, 0));
  expect(
    recipe
      .getCommandNodes(graph)
      .map(node => node.id)
      .at(-1)
  ).toBe('texture-shading-finalize');
  expect(() => create({textureShade: undefined})).toThrow(/at least one output/);
  expect(() => create({levelCount: 9})).toThrow(/levelCount/);
  expect(() => create({baseSigma: 0})).toThrow(/baseSigma/);
  expect(() =>
    create({
      settings: createTransientView(graph, 'settings-11', 'float32', 11)
    })
  ).toThrow(/settings/);
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

it('getTerrainIlluminationGroundCellSize scales Web Mercator and geographic rows', () => {
  expect(getTerrainIlluminationGroundCellSize('uniform', [3, 4], 0, 0, 2, 10)).toEqual([3, 4]);
  const [x, y] = getTerrainIlluminationGroundCellSize('geographic', [0.001, 0.001], 60, 60, 0, 1);
  expect(x).toBeCloseTo(111.31949 * 0.5, 3);
  expect(y).toBeCloseTo(111.31949, 3);
  const mercator = getTerrainIlluminationGroundCellSize('web-mercator', [10, 10], 0.5, 0.5, 0, 1);
  expect(mercator[0]).toBeCloseTo(10, 6);
});

it('GPUTerrainHorizon validates unorm16, anisotropic, and negative-openness options', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Partial<GPUTerrainHorizonProps> = {}) => {
    instance++;
    return new GPUTerrainHorizon({
      width: 6,
      height: 5,
      maximumRadius: 4,
      directionCount: 7,
      elevation: createBand(graph, `elevation-${instance}`, 30),
      settings: createTransientView(graph, `settings-${instance}`, 'float32', 12),
      skyViewFactor: createTransientView(graph, `svf-${instance}`, 'float32', 30),
      ...overrides
    });
  };
  // Defaults are unchanged: float32 horizon storage and 8 settings.
  expect(create().horizonFormat).toBe('float32');
  expect(GPU_TERRAIN_HORIZON_PARAMETER_LENGTH).toBe(8);
  expect(GPU_TERRAIN_HORIZON_ANISOTROPIC_PARAMETER_LENGTH).toBe(12);
  expect(() => create({horizonFormat: 'float16' as never})).toThrow(/horizonFormat/);
  // 30 pixels * 7 sectors = 210 elements = 105 packed words.
  const packed = createTransientView(graph, 'packed', 'uint32', 105);
  const recipe = create({horizonFormat: 'unorm16', horizon: packed});
  expect(recipe.horizonFormat).toBe('unorm16');
  expect(() =>
    create({
      horizonFormat: 'unorm16',
      horizon: createTransientView(graph, 'p-short', 'uint32', 104)
    })
  ).toThrow(/105 uint32/);
  expect(() =>
    create({
      horizonFormat: 'unorm16',
      horizon: createTransientView(graph, 'p-float', 'float32', 105)
    })
  ).toThrow(/horizon/);
  expect(() => create({horizon: createTransientView(graph, 'f-words', 'uint32', 210)})).toThrow(
    /horizon/
  );
  expect(() =>
    create({horizonFormat: 'unorm16', horizon: createTransientView(graph, 'p-full', 'uint32', 210)})
  ).toThrow(/105 uint32/);

  const anisotropic = createTransientView(graph, 'anisotropic', 'float32', 30);
  expect(() =>
    create({
      anisotropicSkyViewFactor: anisotropic,
      settings: createTransientView(graph, 'short-settings', 'float32', 8)
    })
  ).toThrow(/at least 12 float32/);
  expect(() =>
    create({
      skyViewFactor: undefined,
      anisotropicSkyViewFactor: createTransientView(graph, 'anisotropic-short', 'float32', 29)
    })
  ).toThrow(/30 float32/);
  expect(() => create({skyViewFactor: undefined})).toThrow(/at least one output/);
  // A negative-openness-only recipe is a valid output set.
  const everything = create({
    skyViewFactor: createTransientView(graph, 'all-svf', 'float32', 30),
    positiveOpenness: createTransientView(graph, 'all-positive', 'float32', 30),
    negativeOpenness: createTransientView(graph, 'all-negative', 'float32', 30),
    anisotropicSkyViewFactor: createTransientView(graph, 'all-anisotropic', 'float32', 30),
    validity: createTransientView(graph, 'all-validity', 'uint32', 30),
    horizonFormat: 'unorm16',
    horizon: createTransientView(graph, 'all-horizon', 'uint32', 105)
  });
  const nodeIds = everything.getCommandNodes(graph).map(node => node.id);
  expect(nodeIds.filter(id => /-horizon-\d$/.test(id))).toHaveLength(7);
  expect(nodeIds).toContain('terrain-horizon-sky-view');
  expect(nodeIds).toContain('terrain-horizon-extended');
  // The same buffer cannot back two outputs.
  const shared = createTransientView(graph, 'shared-output', 'float32', 30);
  expect(() => create({skyViewFactor: shared, negativeOpenness: shared})).toThrow(/share buffers/);

  // Settings packing: 8 values unless anisotropy is given or the target holds 12.
  expect(getGPUTerrainHorizonParameterValues({cellSize: [2, 3]})).toHaveLength(8);
  const withAnisotropy = getGPUTerrainHorizonParameterValues({
    cellSize: [2, 3],
    anisotropyAzimuthDegrees: 45,
    anisotropyLevel: 8,
    anisotropyMinimumWeight: 0.1
  });
  expect(Array.from(withAnisotropy)).toEqual([2, 3, 1, 0, 0, 0, 0, 0, 45, 8, Math.fround(0.1), 0]);
  expect(
    Array.from(getGPUTerrainHorizonParameterValues({cellSize: [2, 3]}, new Float32Array(12)))
  ).toEqual([2, 3, 1, 0, 0, 0, 0, 0, 315, 4, Math.fround(0.4), 0]);
  expect(() =>
    getGPUTerrainHorizonParameterValues({cellSize: [2, 3], anisotropyLevel: 4}, new Float32Array(8))
  ).toThrow(/12 values/);
  expect(() =>
    getGPUTerrainHorizonParameterValues({cellSize: [2, 3]}, new Float32Array(7))
  ).toThrow(/must hold 8 values/);
  expect(() =>
    getGPUTerrainHorizonParameterValues({cellSize: [2, 3], anisotropyMinimumWeight: 1.5})
  ).toThrow(/minimum weight/);
  expect(() =>
    getGPUTerrainHorizonParameterValues({cellSize: [2, 3], anisotropyLevel: -1})
  ).toThrow(/level/);
  device.destroy();
});

it('getTerrainHorizonSectorOutput emits zenith and nadir accumulations and checks bindings', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = (id: string, format: 'float32' | 'uint32' = 'float32') =>
    createTransientView(graph, id, format, 16);
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
    horizon: view('hp', 'uint32'),
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
