// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUSolarDirectNormalIrradiance,
  getGPUSolarIrradianceParameterValues,
  getGPUSolarIrradianceSunTable,
  GPU_SOLAR_IRRADIANCE_PARAMETER_LENGTH,
  GPU_SOLAR_IRRADIANCE_SUN_TABLE_STRIDE,
  GPUSolarIrradiance,
  type GPUSolarIrradianceProps
} from '../../../src/gpu-terrain/terrain-illumination/gpu-solar-irradiance';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('getGPUSolarIrradianceSunTable samples midpoints with real durations', () => {
  expect(GPU_SOLAR_IRRADIANCE_SUN_TABLE_STRIDE).toBe(4);
  const start = Date.UTC(2024, 5, 20);
  const day = getGPUSolarIrradianceSunTable({
    longitude: 8,
    latitude: 47,
    start,
    end: start + 86400000
  });
  expect(day.sampleCount).toBe(288);
  expect(day.values.length).toBe(288 * 4);
  let hours = 0;
  let up = 0;
  for (let row = 0; row < day.sampleCount; row++) {
    hours += day.values[row * 4 + 2];
    if (day.values[row * 4 + 1] > 0) {
      up++;
      expect(day.values[row * 4 + 3]).toBeGreaterThan(0);
    } else {
      expect(day.values[row * 4 + 3]).toBe(0);
    }
  }
  expect(hours).toBeCloseTo(24, 4);
  // Midsummer at 47 N: about 16 hours of sun.
  expect(up / 12).toBeGreaterThan(15);
  expect(up / 12).toBeLessThan(17);
  // A partial last step is shortened: 12 minutes in steps of 5 gives 5, 5, 2 minutes.
  const partial = getGPUSolarIrradianceSunTable({
    longitude: 0,
    latitude: 0,
    start,
    end: start + 12 * 60000,
    directNormalIrradiance: 900
  });
  expect(partial.sampleCount).toBe(3);
  expect(partial.values[2]).toBeCloseTo(5 / 60, 6);
  expect(partial.values[10]).toBeCloseTo(2 / 60, 6);
  // Constant DNI applies only while the sun is up (midnight at lon 0).
  expect(partial.values[3]).toBe(0);
  // Meinel: 1353 * 0.7^(AM^0.678); the sun at the zenith has AM ~ 1.
  expect(getGPUSolarDirectNormalIrradiance(90)).toBeGreaterThan(900);
  expect(getGPUSolarDirectNormalIrradiance(90)).toBeLessThan(1100);
  expect(getGPUSolarDirectNormalIrradiance(10)).toBeLessThan(getGPUSolarDirectNormalIrradiance(60));
  expect(getGPUSolarDirectNormalIrradiance(0)).toBe(0);
  expect(() =>
    getGPUSolarIrradianceSunTable({longitude: 0, latitude: 0, start, end: start})
  ).toThrow(/after start/);
  expect(() =>
    getGPUSolarIrradianceSunTable({
      longitude: 0,
      latitude: 0,
      start,
      end: start + 1e6,
      stepMinutes: 0
    })
  ).toThrow(/stepMinutes/);
  expect(() =>
    getGPUSolarIrradianceSunTable({
      longitude: 0,
      latitude: 0,
      start,
      end: start + 86400000,
      target: new Float32Array(8)
    })
  ).toThrow(/target/);
});

it('getGPUSolarIrradianceParameterValues packs and validates settings', () => {
  expect(GPU_SOLAR_IRRADIANCE_PARAMETER_LENGTH).toBe(8);
  const values = getGPUSolarIrradianceParameterValues({sampleCount: 288, diffuseIrradiance: 70});
  expect(Array.from(values)).toEqual([288, Math.fround(0.2666), 70, 0, 0, 0, 0, 0]);
  expect(() => getGPUSolarIrradianceParameterValues({sampleCount: 1.5})).toThrow(/sampleCount/);
  expect(() =>
    getGPUSolarIrradianceParameterValues({sampleCount: 1, angularRadiusDegrees: -1})
  ).toThrow(/radius/);
  expect(() =>
    getGPUSolarIrradianceParameterValues({sampleCount: 1, diffuseIrradiance: -1})
  ).toThrow(/diffuse/);
  expect(() => getGPUSolarIrradianceParameterValues({sampleCount: 1}, new Float32Array(4))).toThrow(
    /8 values/
  );
});

it('GPUSolarIrradiance validates props and schedules chunked nodes', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const width = 4;
  const height = 3;
  const create = (overrides: Partial<GPUSolarIrradianceProps> = {}) => {
    instance++;
    const view = (name: string, length: number) =>
      createTransientView(graph, `${name}-${instance}`, 'float32', length);
    return new GPUSolarIrradiance({
      id: `irradiance-${instance}`,
      width,
      height,
      directionCount: 8,
      horizon: view('horizon', width * height * 8),
      sunTable: view('table', 10 * 4),
      sampleCapacity: 10,
      settings: view('settings', 8),
      sunHours: view('hours', width * height),
      ...overrides
    });
  };
  const float = (name: string) => createTransientView(graph, name, 'float32', width * height);
  const single = create();
  expect(single.getCommandNodes(graph).length).toBe(1);
  expect(create({samplesPerNode: 4}).getCommandNodes(graph).length).toBe(3);
  const all = create({
    insolation: float('all-insolation'),
    validity: createTransientView(graph, 'all-validity', 'uint32', width * height),
    slope: float('all-slope'),
    aspect: float('all-aspect'),
    skyViewFactor: float('all-svf'),
    samplesPerNode: 5
  });
  // Two accumulation nodes (8 storage bindings each) and one validity node.
  expect(all.getCommandNodes(graph).length).toBe(3);
  const unorm = create({
    horizonFormat: 'unorm16',
    horizon: createTransientView(
      graph,
      'unorm-horizon',
      'uint32',
      Math.ceil((width * height * 8) / 2)
    )
  });
  expect(unorm.horizonFormat).toBe('unorm16');
  expect(() => create({horizonFormat: 'unorm16'})).toThrow(/uint32/);
  expect(() => create({directionCount: 3})).toThrow(/directionCount/);
  expect(() => create({sampleCapacity: 0})).toThrow(/sampleCapacity/);
  expect(() => create({samplesPerNode: 0})).toThrow(/samplesPerNode/);
  expect(() => create({sampleCapacity: 11})).toThrow(/sunTable/);
  expect(() => create({sunHours: undefined})).toThrow(/at least one output/);
  expect(() => create({slope: float('lonely-slope')})).toThrow(/together/);
  const shared = float('shared');
  expect(() => create({sunHours: shared, insolation: shared})).toThrow(/share buffers/);
  expect(() => create({settings: createTransientView(graph, 'short', 'float32', 4)})).toThrow(
    /at least 8/
  );
});
