// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  getGPUSolarPositionParameterValues,
  GPU_SOLAR_SUNRISE_ALTITUDE_DEGREES,
  GPUSolarPosition,
  type GPUSolarPositionProps
} from '../../../src/gpu-terrain/terrain-illumination/gpu-solar-position';
import {
  getSolarPosition,
  getSolarRefractionDegrees,
  getSolarTimeParameter
} from '../../../src/gpu-terrain/terrain-illumination/solar-position';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createRandom} from './terrain-horizon-oracle';
import {getAngleDifference, getAlmanacSolarPosition} from './solar-position-oracle';

it('getSolarPosition matches equinox, solstice, and equation-of-time references', () => {
  // Astronomical events from the USNO / IMCCE tables.
  expect(getSolarPosition(Date.UTC(2024, 2, 20, 3, 6), 0, 0).declinationDegrees).toBeCloseTo(0, 2);
  expect(getSolarPosition(Date.UTC(2024, 5, 20, 20, 51), 0, 0).declinationDegrees).toBeCloseTo(
    23.4362,
    2
  );
  expect(getSolarPosition(Date.UTC(2024, 11, 21, 9, 21), 0, 0).declinationDegrees).toBeCloseTo(
    -23.4362,
    2
  );
  // Equation of time extremes: about +16.4 min near Nov 3 and -14.2 min near Feb 11.
  expect(
    Math.abs(getSolarPosition(Date.UTC(2024, 10, 3, 12), 0, 0).equationOfTimeMinutes - 16.4)
  ).toBeLessThan(0.2);
  expect(
    Math.abs(getSolarPosition(Date.UTC(2024, 1, 11, 12), 0, 0).equationOfTimeMinutes + 14.2)
  ).toBeLessThan(0.2);

  // At local solar noon the sun is due south (north of the tropics) at 90 - latitude + declination.
  const longitude = 8.54;
  const latitude = 47.37;
  const day = Date.UTC(2025, 6, 15);
  const approximate = getSolarPosition(day + 12 * 3600000, longitude, latitude);
  const noon = day + (720 - 4 * longitude - approximate.equationOfTimeMinutes) * 60000;
  const atNoon = getSolarPosition(noon, longitude, latitude, {
    refraction: false
  });
  expect(Math.abs(atNoon.hourAngleDegrees)).toBeLessThan(0.01);
  expect(atNoon.azimuthDegrees).toBeCloseTo(180, 1);
  expect(atNoon.altitudeDegrees).toBeCloseTo(90 - latitude + atNoon.declinationDegrees, 2);
  // Southern hemisphere noon faces north.
  const south = getSolarPosition(noon - 4 * (151.2 - longitude) * 60000, 151.2, -33.87);
  expect(getAngleDifference(south.azimuthDegrees, 0)).toBeLessThan(0.5);
});

it('getSolarPosition agrees with the independent Astronomical Almanac formulas', () => {
  const random = createRandom(5);
  let checked = 0;
  for (let sample = 0; sample < 2000; sample++) {
    const timestamp = Date.UTC(2000, 0, 1) + random() * 40 * 365.25 * 86400000;
    const longitude = -180 + random() * 360;
    const latitude = -66 + random() * 132;
    const reference = getAlmanacSolarPosition(timestamp, longitude, latitude);
    const actual = getSolarPosition(timestamp, longitude, latitude, {
      refraction: false
    });
    expect(Math.abs(actual.declinationDegrees - reference.declinationDegrees)).toBeLessThan(0.02);
    expect(Math.abs(actual.altitudeDegrees - reference.altitudeDegrees)).toBeLessThan(0.03);
    if (reference.altitudeDegrees > -10 && reference.altitudeDegrees < 80) {
      expect(getAngleDifference(actual.azimuthDegrees, reference.azimuthDegrees)).toBeLessThan(0.1);
      checked++;
    }
  }
  expect(checked).toBeGreaterThan(500);
});

it('getSolarPosition applies NOAA refraction and splits time for float32', () => {
  expect(getSolarRefractionDegrees(90)).toBe(0);
  expect(getSolarRefractionDegrees(0)).toBeCloseTo(1735 / 3600, 6);
  expect(getSolarRefractionDegrees(10)).toBeCloseTo(0.0886, 3);
  const timestamp = Date.UTC(2026, 9, 4, 17, 30, 15, 250);
  const refracted = getSolarPosition(timestamp, -105, 40);
  const geometric = getSolarPosition(timestamp, -105, 40, {
    refraction: false
  });
  expect(refracted.altitudeDegrees - geometric.altitudeDegrees).toBeCloseTo(
    getSolarRefractionDegrees(geometric.altitudeDegrees),
    9
  );
  expect(refracted.geometricAltitudeDegrees).toBe(geometric.altitudeDegrees);

  const {dayNumber, dayFraction} = getSolarTimeParameter(timestamp);
  expect(Number.isInteger(dayNumber)).toBe(true);
  expect(dayFraction).toBeGreaterThanOrEqual(0);
  expect(dayFraction).toBeLessThan(1);
  expect((dayNumber + dayFraction) * 86400000 + Date.UTC(2000, 0, 1, 12)).toBeCloseTo(timestamp, 0);
  expect(getSolarTimeParameter(new Date(Date.UTC(2000, 0, 1, 12)))).toEqual({
    dayNumber: 0,
    dayFraction: 0
  });
  expect(() => getSolarTimeParameter(NaN)).toThrow(/finite/);
  const packed = getGPUSolarPositionParameterValues({timestamp});
  expect(packed[0]).toBe(dayNumber);
  expect(packed[1]).toBeCloseTo(dayFraction, 6);
  expect(packed[2]).toBeCloseTo(GPU_SOLAR_SUNRISE_ALTITUDE_DEGREES, 6);
  expect(() => getGPUSolarPositionParameterValues({timestamp}, new Float32Array(3))).toThrow(
    /4 values/
  );
});

it('GPUSolarPosition validates views and schedules one node', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = (overrides: Partial<GPUSolarPositionProps> = {}) => {
    instance++;
    return new GPUSolarPosition({
      positions: createTransientView(graph, `positions-${instance}`, 'float32x2', 10),
      settings: createTransientView(graph, `settings-${instance}`, 'float32', 4),
      altitude: createTransientView(graph, `altitude-${instance}`, 'float32', 10),
      ...overrides
    });
  };
  const recipe = create();
  expect(recipe.recipe).toBe('solar-position');
  expect(recipe.getCommandNodes(graph).map(node => node.id)).toEqual(['solar-position-sun']);
  expect(() => create({altitude: undefined})).toThrow(/at least one output/);
  expect(() => create({azimuth: createTransientView(graph, 'short', 'float32', 9)})).toThrow(
    /10 float32/
  );
  expect(() =>
    create({
      daylight: createTransientView(graph, 'daylight-f', 'float32', 10) as never
    })
  ).toThrow();
  expect(() =>
    create({
      settings: createTransientView(graph, 'settings-3', 'float32', 3)
    })
  ).toThrow(/settings/);
  const shared = createTransientView(graph, 'shared', 'float32', 10);
  expect(() => create({altitude: shared, azimuth: shared})).toThrow(/share buffers/);
  device.destroy();
});
