// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import geodesics from '../../oracles/fixtures/parity-wp2-round8-geodesics.json';
import weights from '../../oracles/fixtures/parity-wp2-round8-weights.json';
import arcs from '../../oracles/fixtures/parity-wp2-round8-arcs-0.json';
import joins from '../../oracles/fixtures/parity-wp2-round8-joins.json';
import encounters from '../../oracles/fixtures/parity-wp2-round8-encounters.json';
import density from '../../oracles/fixtures/parity-wp2-round8-spherical-density.json';
import polygons from '../../oracles/fixtures/parity-wp2-round8-spherical-polygons.json';

it('round 8 references span every discrete k/power/radius and arc control value', () => {
  expect(weights.cases.map(entry => entry.k)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
  for (const entry of weights.cases) {
    expect(
      entry.rules.filter(rule => rule.weightType === 'inverse-distance').map(rule => rule.power)
    ).toEqual([0.5, 1, 1.5, 2, 2.5, 3]);
    expect(entry.expected.offsets.at(-1)).toBe(entry.expected.neighbors.length);
    expect(entry.expected.distances).toHaveLength(entry.expected.neighbors.length);
  }
  const settings = arcs.cases[0].settings;
  expect(settings).toHaveLength(41 * 64);
  expect(
    new Set(settings.map(value => `${value.maximumSegmentLength}/${value.minimumSegments}`)).size
  ).toBe(41 * 64);
  expect(
    settings.every(value =>
      value.counts.every(count => count >= value.minimumSegments && count <= 64)
    )
  ).toBe(true);
  for (const entry of joins.cases) {
    const radii = new Set(entry.frames.map(frame => frame.radius));
    for (let radius = 0; radius <= 2000; radius += 25) expect(radii.has(radius)).toBe(true);
    for (let radius = 100; radius <= 8000; radius += 100) expect(radii.has(radius)).toBe(true);
  }
});

it('meter/kilometer references preserve the same physical distances', () => {
  for (const metric of ['rhumb', 'great-circle', 'ellipsoidal']) {
    const meters = geodesics.cases.find(entry => entry.name === `${metric}-meters`)!;
    const kilometers = geodesics.cases.find(entry => entry.name === `${metric}-kilometers`)!;
    meters.distances.forEach((value, index) =>
      expect(Math.abs(value - 1000 * kilometers.distances[index])).toBeLessThan(1e-7)
    );
  }
  const meters = density.cases[0].frames[0];
  const kilometers = density.cases[1].frames[0];
  meters.lengths.forEach((value, index) =>
    expect(Math.abs(value - 1000 * kilometers.lengths[index])).toBeLessThan(1e-7)
  );
  // Density is length / area, so converting meters to kilometers multiplies by 1000.
  meters.densities.forEach((value, index) =>
    expect(Math.abs(value * 1000 - kilometers.densities[index])).toBeLessThan(1e-10)
  );
});

it('independent spherical fixtures retain complete output arrays and strict empty rows', () => {
  for (const entry of encounters.cases) {
    expect(entry.samples).toHaveLength(entry.trackCount * entry.bucketCount * 2);
    expect(entry.expected.counts.reduce((total, value) => total + value, 0)).toBe(entry.hitCount);
    expect(entry.hitCount).toBeGreaterThan(1);
  }
  for (const entry of polygons.cases) {
    expect(entry.expected.lengths).toHaveLength(entry.polygons.featureOffsets.length - 1);
    expect(entry.polygons.ringOffsets.at(-1)).toBe(entry.polygons.positions.length / 2);
    if (!entry.useWeights) expect(entry.expected.weightedLengths).toEqual(entry.expected.lengths);
  }
  for (const entry of density.cases)
    for (const frame of entry.frames) {
      expect(frame.lengths).toHaveLength(entry.columns * entry.rows);
      frame.lengths.forEach((length, index) => {
        if (length === 0) {
          expect(frame.lengthBounds[index]).toBe(0);
          expect(frame.densityBounds[index]).toBe(0);
        }
      });
    }
});
