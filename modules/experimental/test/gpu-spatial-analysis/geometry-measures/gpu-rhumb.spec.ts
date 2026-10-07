// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {GPUGeodesicDestination} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geodesic-destination';
import {GPUGeodesicPairs} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geodesic-pairs';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

const EARTH_RADIUS = 6371008.8;
const D2R = Math.PI / 180;
const R2D = 180 / Math.PI;

/** f64 closed forms of the sphere rhumb line (the formulas of turf `rhumb*` and geo `Rhumb`). */
function getRhumbPair(a: number[], b: number[]) {
  const phi1 = a[1] * D2R;
  const phi2 = b[1] * D2R;
  let deltaLambda = (b[0] - a[0]) * D2R;
  if (Math.abs(deltaLambda) > Math.PI) {
    deltaLambda += deltaLambda > 0 ? -2 * Math.PI : 2 * Math.PI;
  }
  const deltaPsi = Math.log(Math.tan(Math.PI / 4 + phi2 / 2) / Math.tan(Math.PI / 4 + phi1 / 2));
  const deltaPhi = phi2 - phi1;
  const q = Math.abs(deltaPsi) > 1e-12 ? deltaPhi / deltaPsi : Math.cos(phi1);
  return {
    distance: Math.hypot(deltaPhi, q * deltaLambda) * EARTH_RADIUS,
    bearing: Math.atan2(deltaLambda, deltaPsi) * R2D,
    midpoint: [a[0] + 0.5 * (deltaLambda * R2D), 0.5 * (a[1] + b[1])]
  };
}

function getRhumbDestination(origin: number[], bearing: number, distance: number): number[] {
  const delta = distance / EARTH_RADIUS;
  const phi1 = origin[1] * D2R;
  const theta = bearing * D2R;
  const deltaPhi = delta * Math.cos(theta);
  let phi2 = phi1 + deltaPhi;
  if (Math.abs(phi2) > Math.PI / 2) {
    phi2 = phi2 > 0 ? Math.PI - phi2 : -Math.PI - phi2;
  }
  const deltaPsi = Math.log(Math.tan(phi2 / 2 + Math.PI / 4) / Math.tan(phi1 / 2 + Math.PI / 4));
  const q = Math.abs(deltaPsi) > 1e-11 ? deltaPhi / deltaPsi : Math.cos(phi1);
  return [origin[0] + ((delta * Math.sin(theta)) / q) * R2D, phi2 * R2D];
}

/**
 * Pinned from turf (`@turf/rhumb-distance`, `rhumb-bearing`, `rhumb-destination` 7.x, f64, meters,
 * radius 6371008.8); generator in scratchpad build/J/gen.mjs.
 */
const TURF_PAIRS: [string, number[], number[], number, number][] = [
  ['1 m', [2.35, 48.85], [2.350013, 48.850001], 0.9576870745754993, 83.33246619922852],
  ['1 km', [-73.98, 40.75], [-73.97, 40.752], 871.2242351558756, 75.21090863073431],
  ['100 km', [139.7, 35.7], [140.5, 36.4], 105976.75200217181, 42.737699719311934],
  ['1,000 km', [-0.13, 51.5], [12.5, 41.9], 1435306.4704436501, 138.04967075079367],
  ['transpacific', [151.2, -33.9], [-118.2, 34], 12108362.161459818, 51.42430062401968],
  ['antimeridian', [179.5, 10], [-179.5, 12], 247730.22859042592, 26.141194384294977],
  ['east-west', [10, 60], [30, 60], 1111950.8023353291, 90],
  ['north-south', [10, -20], [10, 50], 7783655.616347303, 0],
  ['polar', [0, 89.9], [179, 89.8], 51336.28927988376, 102.50949020230757]
];

const TURF_DESTINATIONS: [string, number[], number, number, number[]][] = [
  ['100 km NE', [2, 48], 45, 100000, [2.9562865493012396, 48.63591552764876]],
  ['1,000 km E', [-30, 50], 90, 1000000, [-16.00905882172958, 50]],
  ['500 km S', [100, -10], 180, 500000, [100, -14.49660181862269]],
  ['3,000 km W', [170, 20], 270, 3000000, [141.28889775768187, 19.999999999999993]],
  ['1 km NNW', [-73.98, 40.75], 340, 1000, [-73.98406044675147, 40.75845084709515]],
  ['zero', [10, 10], 77, 0, [10, 10]]
];

function getBearingError(actual: number, expected: number): number {
  const difference = Math.abs(actual - expected) % 360;
  return Math.min(difference, 360 - difference);
}

it('rhumb f64 closed form reproduces the pinned turf values', () => {
  for (const [name, a, b, distance, bearing] of TURF_PAIRS) {
    const actual = getRhumbPair(a, b);
    expect(Math.abs(actual.distance - distance), name).toBeLessThan(1e-6 + 1e-9 * distance);
    expect(getBearingError(actual.bearing, bearing), name).toBeLessThan(1e-9);
  }
  for (const [name, origin, bearing, distance, expected] of TURF_DESTINATIONS) {
    const actual = getRhumbDestination(origin, bearing, distance);
    expect(Math.abs(actual[0] - expected[0]), name).toBeLessThan(1e-9);
    expect(Math.abs(actual[1] - expected[1]), name).toBeLessThan(1e-9);
  }
});

it('GPUGeodesicPairs rhumb matches the f64 closed form (pinned to turf)', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const count = TURF_PAIRS.length;
  const origins = new Float32Array(TURF_PAIRS.flatMap(([, a]) => a));
  const targets = new Float32Array(TURF_PAIRS.flatMap(([, , b]) => b));
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const graph = new GPUCommandGraph(device, {id: 'rhumb-pairs'});
  const outputs = {
    distances: track(createOutputBuffer(device, count)),
    initialBearings: track(createOutputBuffer(device, count)),
    finalBearings: track(createOutputBuffer(device, count)),
    midpoints: track(createOutputBuffer(device, 2 * count))
  };
  graph.add(
    new GPUGeodesicPairs({
      origins: importGraphBuffer(
        graph,
        'origins',
        track(createInputBuffer(device, origins)),
        'float32x2',
        count
      ),
      targets: importGraphBuffer(
        graph,
        'targets',
        track(createInputBuffer(device, targets)),
        'float32x2',
        count
      ),
      model: 'rhumb',
      output: {
        distances: importGraphBuffer(graph, 'o-d', outputs.distances, 'float32', count),
        initialBearings: importGraphBuffer(graph, 'o-i', outputs.initialBearings, 'float32', count),
        finalBearings: importGraphBuffer(graph, 'o-f', outputs.finalBearings, 'float32', count),
        midpoints: importGraphBuffer(graph, 'o-m', outputs.midpoints, 'float32x2', count)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const distances = await readFloat32(outputs.distances, count);
  const initial = await readFloat32(outputs.initialBearings, count);
  const final = await readFloat32(outputs.finalBearings, count);
  const midpoints = await readFloat32(outputs.midpoints, 2 * count);
  const report: string[] = [];
  TURF_PAIRS.forEach(([name], row) => {
    // The f64 reference sees the same f32-rounded inputs the GPU does.
    const a = [origins[2 * row], origins[2 * row + 1]];
    const b = [targets[2 * row], targets[2 * row + 1]];
    const expected = getRhumbPair(a, b);
    expect(final[row], `${name} final`).toBe(initial[row]);
    const error = Math.abs(distances[row] - expected.distance);
    report.push(`${name}: ${error.toExponential(2)} m`);
    expect(error, name).toBeLessThanOrEqual(1e-3 + 3e-7 * expected.distance);
    expect(getBearingError(initial[row], expected.bearing), `${name} bearing`).toBeLessThan(
      name === '1 m' ? 1e-2 : 1e-4
    );
    expect(Math.abs(midpoints[2 * row] - expected.midpoint[0]), `${name} mid x`).toBeLessThan(1e-4);
    expect(Math.abs(midpoints[2 * row + 1] - expected.midpoint[1]), `${name} mid y`).toBeLessThan(
      1e-4
    );
  });
  // Same-latitude and same-meridian edges hit the east-west limit and the zero-longitude case.
  expect(Math.abs(distances[6] - 1111950.8023353291)).toBeLessThan(5);
  expect(initial[6]).toBeCloseTo(90, 4);
  expect(initial[7]).toBeCloseTo(0, 4);
  console.log(`GPUGeodesicPairs rhumb distance errors vs f64 closed form: ${report.join(', ')}`);
  compiled.destroy();
  buffers.forEach(buffer => buffer.destroy());
});

it('GPUGeodesicDestination rhumb matches the f64 closed form (pinned to turf)', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const rows = [
    ...TURF_DESTINATIONS.map(([, origin, bearing, distance]) => ({origin, bearing, distance})),
    // Overshoots the north pole: reflected back across it.
    {origin: [20, 85], bearing: 0, distance: 1000000}
  ];
  const count = rows.length;
  const origins = new Float32Array(rows.flatMap(row => row.origin));
  const bearings = new Float32Array(rows.map(row => row.bearing));
  const distances = new Float32Array(rows.map(row => row.distance));
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const graph = new GPUCommandGraph(device, {id: 'rhumb-destination'});
  const destinations = track(createOutputBuffer(device, 2 * count));
  const finalBearings = track(createOutputBuffer(device, count));
  graph.add(
    new GPUGeodesicDestination({
      origins: importGraphBuffer(
        graph,
        'origins',
        track(createInputBuffer(device, origins)),
        'float32x2',
        count
      ),
      bearings: importGraphBuffer(
        graph,
        'bearings',
        track(createInputBuffer(device, bearings)),
        'float32',
        count
      ),
      distances: importGraphBuffer(
        graph,
        'distances',
        track(createInputBuffer(device, distances)),
        'float32',
        count
      ),
      model: 'rhumb',
      output: {
        destinations: importGraphBuffer(graph, 'o-dest', destinations, 'float32x2', count),
        finalBearings: importGraphBuffer(graph, 'o-final', finalBearings, 'float32', count)
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const actual = await readFloat32(destinations, 2 * count);
  const actualFinal = await readFloat32(finalBearings, count);
  const report: string[] = [];
  rows.forEach((row, index) => {
    const origin = [origins[2 * index], origins[2 * index + 1]];
    const expected = getRhumbDestination(origin, row.bearing, row.distance);
    const errorLng = Math.abs(actual[2 * index] - expected[0]);
    const errorLat = Math.abs(actual[2 * index + 1] - expected[1]);
    report.push(`#${index}: ${errorLng.toExponential(1)}/${errorLat.toExponential(1)} deg`);
    // Near the pole the longitude is ill-conditioned (q -> 0), so allow a looser bound there.
    expect(errorLng, `row ${index} lng`).toBeLessThan(index === count - 1 ? 0.05 : 2e-4);
    expect(errorLat, `row ${index} lat`).toBeLessThan(2e-4);
    if (index < count - 1) {
      expect(getBearingError(actualFinal[index], row.bearing)).toBeLessThan(1e-4);
    }
  });
  // North-pole overshoot flips the final bearing to southbound.
  expect(getBearingError(actualFinal[count - 1], 180)).toBeLessThan(1e-4);
  console.log(`GPUGeodesicDestination rhumb errors vs f64 closed form: ${report.join(', ')}`);
  compiled.destroy();
  buffers.forEach(buffer => buffer.destroy());
});
