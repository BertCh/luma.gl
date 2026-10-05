// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it, vi} from 'vitest';
import {GPUGeodesicDestination} from '../../../src/map-graphs/geometry-measures/gpu-geodesic-destination';
import {GPUGeodesicPairs} from '../../../src/map-graphs/geometry-measures/gpu-geodesic-pairs';
import {importGraphBuffer, submitGraph} from '../../../src/map-graphs/map-graph-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32
} from '../map-graph-test-utils';
import {
  getDestination,
  getHaversineDistance,
  getInitialBearingDegrees,
  interpolateGreatCircle
} from './geodesic-oracle';
import {getVincentyDirect, getVincentyInverse} from './vincenty-oracle';

const EARTH_RADIUS = 6371008.8;

/** Named pairs spanning 1 m to near-antipodal. */
const PAIRS: [string, number[], number[]][] = [
  ['1 m', [2.35, 48.85], [2.350013, 48.850001]],
  ['1 km', [-73.98, 40.75], [-73.97, 40.752]],
  ['100 km', [139.7, 35.7], [140.5, 36.4]],
  ['1,000 km', [-0.13, 51.5], [12.5, 41.9]],
  ['10,000 km', [151.2, -33.9], [-118.2, 34]],
  ['polar', [0, 89.9], [179, 89.8]],
  ['equator', [0, 0], [90, 0]],
  ['zero', [10, 10], [10, 10]],
  ['near-antipodal', [0, 0], [179.7, 0.2]]
];

function getBearingError(actual: number, expected: number): number {
  const difference = Math.abs(actual - expected) % 360;
  return Math.min(difference, 360 - difference);
}

type PairsReadback = {
  distances: number[];
  initialBearings: number[];
  finalBearings: number[];
  midpoints: number[][];
  converged: number[];
};

function createPairsFixture(device: Device, model: 'sphere' | 'wgs84', pairCount: number) {
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const graph = new GPUCommandGraph(device, {id: `geodesic-pairs-${model}`});
  const origins = track(createOutputBuffer(device, 2 * pairCount));
  const targets = track(createOutputBuffer(device, 2 * pairCount));
  const outputs = {
    distances: track(createOutputBuffer(device, pairCount)),
    initialBearings: track(createOutputBuffer(device, pairCount)),
    finalBearings: track(createOutputBuffer(device, pairCount)),
    midpoints: track(createOutputBuffer(device, 2 * pairCount)),
    converged: track(createOutputBuffer(device, pairCount))
  };
  graph.add(
    new GPUGeodesicPairs({
      origins: importGraphBuffer(graph, 'origins', origins, 'float32x2', pairCount),
      targets: importGraphBuffer(graph, 'targets', targets, 'float32x2', pairCount),
      model,
      output: {
        distances: importGraphBuffer(graph, 'o-distances', outputs.distances, 'float32', pairCount),
        initialBearings: importGraphBuffer(
          graph,
          'o-initial',
          outputs.initialBearings,
          'float32',
          pairCount
        ),
        finalBearings: importGraphBuffer(
          graph,
          'o-final',
          outputs.finalBearings,
          'float32',
          pairCount
        ),
        midpoints: importGraphBuffer(
          graph,
          'o-midpoints',
          outputs.midpoints,
          'float32x2',
          pairCount
        ),
        ...(model === 'wgs84'
          ? {
              converged: importGraphBuffer(
                graph,
                'o-converged',
                outputs.converged,
                'uint32',
                pairCount
              )
            }
          : {})
      }
    })
  );
  const compiled = graph.compile();
  const compileSpy = vi.spyOn(graph, 'compile');
  return {
    async run(originValues: Float32Array, targetValues: Float32Array): Promise<PairsReadback> {
      origins.write(originValues);
      targets.write(targetValues);
      submitGraph(device, compiled, undefined);
      const midpoints = await readFloat32(outputs.midpoints, 2 * pairCount);
      return {
        distances: await readFloat32(outputs.distances, pairCount),
        initialBearings: await readFloat32(outputs.initialBearings, pairCount),
        finalBearings: await readFloat32(outputs.finalBearings, pairCount),
        midpoints: Array.from({length: pairCount}, (_, row) =>
          midpoints.slice(2 * row, 2 * row + 2)
        ),
        converged: await readUint32(outputs.converged, pairCount)
      };
    },
    getCompileCount: () => compileSpy.mock.calls.length,
    destroy() {
      compiled.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

function getPairValues(pairs: typeof PAIRS): {origins: Float32Array; targets: Float32Array} {
  return {
    origins: new Float32Array(pairs.flatMap(([, origin]) => origin)),
    targets: new Float32Array(pairs.flatMap(([, , target]) => target))
  };
}

const rounded = (values: Float32Array, row: number) => [values[2 * row], values[2 * row + 1]];

it('GPUGeodesicPairs sphere matches f64 haversine, bearings and midpoints', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {origins, targets} = getPairValues(PAIRS);
  const fixture = createPairsFixture(device, 'sphere', PAIRS.length);
  const actual = await fixture.run(origins, targets);
  const report: string[] = [];
  PAIRS.forEach(([name], row) => {
    const a = rounded(origins, row);
    const b = rounded(targets, row);
    const distance = getHaversineDistance(a, b, EARTH_RADIUS);
    const error = Math.abs(actual.distances[row] - distance);
    report.push(`${name}: ${error.toExponential(2)} m`);
    expect(error, name).toBeLessThanOrEqual(1e-3 + 2e-7 * distance);
    if (distance > 0.5 && name !== 'near-antipodal') {
      const bearingError = getBearingError(
        actual.initialBearings[row],
        getInitialBearingDegrees(a, b)
      );
      expect(bearingError, `${name} bearing`).toBeLessThan(name === '1 m' ? 1e-2 : 1e-4);
      const finalBearing = getInitialBearingDegrees(b, a) + 180;
      expect(
        getBearingError(actual.finalBearings[row], finalBearing),
        `${name} final`
      ).toBeLessThan(name === '1 m' ? 1e-2 : 1e-4);
      const midpoint = interpolateGreatCircle(a, b, 0.5);
      expect(getHaversineDistance(actual.midpoints[row], midpoint, EARTH_RADIUS)).toBeLessThan(2);
    }
  });
  console.log(`GPUGeodesicPairs sphere distance errors: ${report.join(', ')}`);
  // New pairs, same graph.
  const swapped = await fixture.run(targets, origins);
  expect(Math.abs(swapped.distances[3] - actual.distances[3])).toBeLessThan(1);
  expect(fixture.getCompileCount()).toBe(0);
  fixture.destroy();
});

it('GPUGeodesicPairs WGS84 Vincenty matches f64 Vincenty and flags near-antipodal pairs', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const {origins, targets} = getPairValues(PAIRS);
  const fixture = createPairsFixture(device, 'wgs84', PAIRS.length);
  const actual = await fixture.run(origins, targets);
  const report: string[] = [];
  PAIRS.forEach(([name], row) => {
    const a = rounded(origins, row);
    const b = rounded(targets, row);
    const reference = getVincentyInverse(a, b);
    if (name === 'near-antipodal') {
      expect(actual.converged[row], name).toBe(0);
      // Sphere fallback (f32 near-antipodal sphere distances are good to a few meters).
      expect(
        Math.abs(actual.distances[row] - getHaversineDistance(a, b, EARTH_RADIUS))
      ).toBeLessThan(5);
      return;
    }
    expect(actual.converged[row], name).toBe(1);
    const error = Math.abs(actual.distances[row] - reference.distance);
    report.push(`${name}: ${error.toExponential(2)} m`);
    expect(error, name).toBeLessThanOrEqual(1e-3 + 1e-6 * reference.distance);
    if (reference.distance > 0.5) {
      expect(
        getBearingError(actual.initialBearings[row], reference.initialBearing),
        `${name} bearing`
      ).toBeLessThan(name === '1 m' ? 1e-2 : 1e-3);
      expect(
        getBearingError(actual.finalBearings[row], reference.finalBearing),
        `${name} final bearing`
      ).toBeLessThan(name === '1 m' ? 1e-2 : 1e-3);
    }
  });
  console.log(`GPUGeodesicPairs wgs84 distance errors vs f64 Vincenty: ${report.join(', ')}`);
  fixture.destroy();
});

it('GPUGeodesicDestination matches f64 sphere and Vincenty direct solutions', async () => {
  const device = await getWebGPUTestDevice();
  if (!device) {
    return;
  }
  const rows = [
    {origin: [2.35, 48.85], bearing: 45, distance: 1},
    {origin: [-73.98, 40.75], bearing: -120, distance: 1500},
    {origin: [139.7, 35.7], bearing: 10, distance: 250000},
    {origin: [170, -20], bearing: 80, distance: 2500000},
    {origin: [0, 0], bearing: 90, distance: 10000000},
    {origin: [20, 85], bearing: 0, distance: 1000000}
  ];
  const origins = new Float32Array(rows.flatMap(row => row.origin));
  const bearings = new Float32Array(rows.map(row => row.bearing));
  const distances = new Float32Array(rows.map(row => row.distance));
  for (const model of ['sphere', 'wgs84'] as const) {
    const buffers: Buffer[] = [];
    const track = (buffer: Buffer) => {
      buffers.push(buffer);
      return buffer;
    };
    const graph = new GPUCommandGraph(device, {id: `geodesic-destination-${model}`});
    const destinations = track(createOutputBuffer(device, 2 * rows.length));
    const finalBearings = track(createOutputBuffer(device, rows.length));
    const converged = track(createOutputBuffer(device, rows.length));
    graph.add(
      new GPUGeodesicDestination({
        origins: importGraphBuffer(
          graph,
          'origins',
          track(createInputBuffer(device, origins)),
          'float32x2',
          rows.length
        ),
        bearings: importGraphBuffer(
          graph,
          'bearings',
          track(createInputBuffer(device, bearings)),
          'float32',
          rows.length
        ),
        distances: importGraphBuffer(
          graph,
          'distances',
          track(createInputBuffer(device, distances)),
          'float32',
          rows.length
        ),
        model,
        output: {
          destinations: importGraphBuffer(
            graph,
            'o-destinations',
            destinations,
            'float32x2',
            rows.length
          ),
          finalBearings: importGraphBuffer(graph, 'o-final', finalBearings, 'float32', rows.length),
          ...(model === 'wgs84'
            ? {converged: importGraphBuffer(graph, 'o-converged', converged, 'uint32', rows.length)}
            : {})
        }
      })
    );
    const compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    const actual = await readFloat32(destinations, 2 * rows.length);
    const actualFinal = await readFloat32(finalBearings, rows.length);
    const actualConverged = await readUint32(converged, rows.length);
    const report: string[] = [];
    rows.forEach((row, index) => {
      const origin = rounded(origins, index);
      const expected =
        model === 'sphere'
          ? {
              destination: getDestination(origin, bearings[index], distances[index] / EARTH_RADIUS),
              finalBearing: Number.NaN
            }
          : getVincentyDirect(origin, bearings[index], distances[index]);
      const error = getHaversineDistance(
        [actual[2 * index], actual[2 * index + 1]],
        expected.destination,
        EARTH_RADIUS
      );
      report.push(`${row.distance} m: ${error.toExponential(2)} m`);
      // Output latitudes are f32 degrees: one ulp is about 0.4 m at mid latitudes.
      expect(error, `${model} row ${index}`).toBeLessThan(1 + 2e-7 * row.distance);
      if (model === 'wgs84') {
        expect(actualConverged[index]).toBe(1);
        expect(getBearingError(actualFinal[index], expected.finalBearing)).toBeLessThan(1e-3);
      }
    });
    console.log(`GPUGeodesicDestination ${model} errors: ${report.join(', ')}`);
    compiled.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
});
