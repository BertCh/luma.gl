// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {it} from 'vitest';
import {getSolarPosition} from '../../../src/gpu-terrain/terrain-illumination/solar-position';
import {
  compareSolarValues,
  reportSolarModelDifference,
  EXACT_SOLAR_TOLERANCE,
  POSITION_FIXTURE
} from './parity-wp5-solar-fixtures';

// The independently published NOAA equations validate the CPU helper at f64 precision.
// This is reference integrity, not evidence of GPU execution.
for (const fixture of POSITION_FIXTURE.cases) {
  it(`NOAA CPU versus published NOAA equations: ${fixture.name}`, () => {
    const azimuth: number[] = [];
    const altitude: number[] = [];
    const daylight: number[] = [];
    // Reconstruct the exact f32-represented time; Date parsing would drop sub-ms precision.
    const timestamp = 946728000000 + (fixture.parameters[0] + fixture.parameters[1]) * 86400000;
    for (let index = 0; index < fixture.positions.length; index += 2) {
      const longitude = fixture.positions[index];
      const latitude = fixture.positions[index + 1];
      if (!Number.isFinite(longitude) || !Number.isFinite(latitude) || Math.abs(latitude) > 90) {
        // Mirror the public GPU validity contract; the CPU helper has no invalid-row guard.
        azimuth.push(NaN);
        altitude.push(NaN);
        daylight.push(0);
        continue;
      }
      const position = getSolarPosition(timestamp, longitude, latitude, {refraction: false});
      azimuth.push(position.azimuthDegrees);
      altitude.push(position.altitudeDegrees);
      daylight.push(position.geometricAltitudeDegrees > fixture.parameters[2] ? 1 : 0);
    }
    for (const [metric, values] of [
      ['azimuth', azimuth],
      ['altitude', altitude],
      ['zenith', altitude.map(value => 90 - value)]
    ] as const) {
      compareSolarValues(
        'getSolarPosition CPU (NOAA; not GPU)',
        fixture.name,
        fixture.reference,
        metric,
        values,
        fixture.expected[metric],
        {profile: 'float64 reference integrity', absolute: 1e-8, relative: 0}
      );
    }
    reportSolarModelDifference(
      'getSolarPosition CPU (not GPU)',
      fixture,
      azimuth,
      altitude,
      daylight
    );
    compareSolarValues(
      'getSolarPosition CPU (NOAA; not GPU)',
      fixture.name,
      fixture.reference,
      'daylight',
      daylight,
      fixture.expected.daylight,
      EXACT_SOLAR_TOLERANCE
    );
  });
}

for (const event of POSITION_FIXTURE.events) {
  it(`NOAA versus SPA published sunrise/set timing accuracy: ${event.name}`, () => {
    compareSolarValues(
      'NOAA reference (not GPU)',
      event.name,
      'pvlib SPA + scipy.optimize.brentq at geometric -0.833 degrees',
      'crossing seconds UTC',
      [event.noaaSeconds],
      [event.spaSeconds],
      {
        profile: 'NOAA published sunrise/set timing accuracy',
        absolute: event.boundSeconds,
        relative: 0
      }
    );
  });
}

it('split solar time retains millisecond precision across the fixture dates', () => {
  const representedSeconds = POSITION_FIXTURE.cases.map(
    fixture => (fixture.parameters[0] + fixture.parameters[1]) * 86400
  );
  const requestedSeconds = POSITION_FIXTURE.cases.map(
    fixture => (fixture.timestamp - 946728000000) / 1000
  );
  // A fraction in [0, 1) has f32 ulp <= 2^-24 day. Half an ulp is
  // 2.574921 ms; 2.6 ms also covers double-precision epoch subtraction.
  compareSolarValues(
    'Solar time transport (CPU packing, not GPU)',
    '2000-2041 split day/fraction',
    'UTC timestamps represented by float32 day/fraction',
    'seconds since J2000',
    representedSeconds,
    requestedSeconds,
    {profile: 'half-ulp32 of fractional day', absolute: 0.0026, relative: 0}
  );
});
