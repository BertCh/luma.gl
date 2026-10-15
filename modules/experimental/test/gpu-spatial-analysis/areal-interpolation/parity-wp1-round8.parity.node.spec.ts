// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {readFileSync} from 'node:fs';
import {inflateSync} from 'node:zlib';
import {createHash} from 'node:crypto';
import pycno from '../../oracles/fixtures/parity-wp1-pycnophylactic.json';
import windows from '../../oracles/fixtures/parity-wp1-study-window.json';
import scan from '../../oracles/fixtures/parity-wp1-scan.json';
import clustering from '../../oracles/fixtures/parity-wp1-clustering-modes.json';
import areal from '../../oracles/fixtures/parity-wp1-areal-modes.json';

function readFixture(name: string) {
  const wrapper = JSON.parse(
    readFileSync(new URL(`../../oracles/fixtures/parity-wp1-${name}.json`, import.meta.url), 'utf8')
  );
  expect(wrapper.command).toContain('/scratchpad/eval/venv/bin/python -I ');
  expect(Object.keys(wrapper.versions).length).toBeGreaterThan(0);
  if (!wrapper.data) return wrapper;
  const data = inflateSync(Buffer.from(wrapper.data, 'base64'));
  expect(createHash('sha256').update(data).digest('hex')).toBe(wrapper.sha256);
  return JSON.parse(data.toString('utf8'));
}

it('Round 8 packed fixtures preserve provenance and the complete focal parameter product', () => {
  const expectedCounts = {
    pycnophylactic: 15,
    'study-window': 5,
    scan: 30,
    'scan-showcase': 6,
    'clustering-modes': 2,
    'areal-modes': 2,
    'round8-zonal-half-open': 2,
    'round8-zonal-showcase': 42,
    'round8-geographic-modes': 4,
    'mantel-band': 3,
    'lag-modes': 6,
    'round8-transpose': 3,
    'round8-algebra': 57,
    'round8-neighborhood': 9,
    'round8-focal': 720,
    'idw-modes': 21,
    'emerging-lengths': 9,
    'emerging-weighted': 54
  };
  for (const [name, count] of Object.entries(expectedCounts)) {
    expect(readFixture(name).cases).toHaveLength(count);
  }
  const focal = readFixture('round8-focal') as {
    cases: {
      settings: {
        radius: number;
        shape: string;
        minimumCount: number;
        propagateCenterNoData: boolean;
      };
    }[];
  };
  const keys = new Set(focal.cases.map(scene => JSON.stringify(scene.settings)));
  expect(keys.size).toBe(720);
  for (let radius = 0; radius <= 8; radius++)
    for (const shape of ['circle', 'square'])
      for (let minimumCount = 1; minimumCount <= 20; minimumCount++)
        for (const propagateCenterNoData of [false, true])
          expect(
            focal.cases.some(
              scene =>
                scene.settings.radius === radius &&
                scene.settings.shape === shape &&
                scene.settings.minimumCount === minimumCount &&
                scene.settings.propagateCenterNoData === propagateCenterNoData
            )
          ).toBe(true);
});

it('Round 8 actual showcase scan horizon and emerging cube lengths are retained', () => {
  const showcase = readFixture('scan-showcase') as {
    cases: {
      baselineKind: string;
      shape: string;
      timeBuckets: number;
      maximumTimeBuckets: number;
      windows: unknown[];
      expected: {candidateLikelihoods: number[][]; maxima: number[]; pValue: number};
    }[];
  };
  expect(new Set(showcase.cases.map(scene => `${scene.baselineKind}/${scene.shape}`)).size).toBe(6);
  for (const scene of showcase.cases) {
    expect(scene.timeBuckets).toBe(26);
    expect(scene.maximumTimeBuckets).toBe(6);
    scene.expected.candidateLikelihoods.forEach((values, index) => {
      expect(values).toHaveLength(scene.windows.length);
      expect(Math.max(...values)).toBe(scene.expected.maxima[index]);
    });
    expect(scene.expected.pValue).toBe(
      (1 +
        scene.expected.maxima.slice(1).filter(value => value >= scene.expected.maxima[0]).length) /
        20
    );
  }
  for (const name of ['emerging-lengths', 'emerging-weighted']) {
    const fixture = readFixture(name) as {cases: {sliceCount: number}[]};
    expect([...new Set(fixture.cases.map(scene => scene.sliceCount))]).toEqual([12, 52, 73]);
  }
});

it('Round 8 pycno adapter preserves all zone totals and covers the showcase ladder', () => {
  expect(pycno.versions.pycno).toBe('1.4.1');
  for (const kernel of ['rook', 'box']) {
    expect(
      pycno.cases
        .filter(scene => scene.kernel === kernel && scene.tolerance < 0)
        .map(scene => scene.iterations)
    ).toEqual(kernel === 'rook' ? [0, 2, 8, 32, 128, 512, 32] : [0, 2, 8, 32, 128, 512]);
    const converged = pycno.cases.find(scene => scene.name === `${kernel}-convergence`)!;
    expect(converged.expected.converged).toBe(1);
    expect(converged.expected.iterations).toBeLessThan(converged.iterations);
  }
  for (const scene of pycno.cases) {
    expect(scene.expected.mass).toHaveLength(scene.width * scene.height);
    expect(scene.expected.mass.every(value => Number.isFinite(value) && value >= 0)).toBe(true);
    for (let zone = 0; zone < scene.totals.length; zone++) {
      const total = scene.expected.mass.reduce(
        (sum, value, cell) => sum + (scene.zones[cell] === zone ? value : 0),
        0
      );
      expect(total).toBeCloseTo(scene.totals[zone], 8);
    }
  }
});

it('Round 8 Shapely windows preserve directed-edge area and sample order', () => {
  for (const scene of windows.cases) {
    expect(scene.edges).toHaveLength(scene.positions.length * 2);
    const area = scene.edges.reduce(
      (sum, _, index, edges) =>
        index % 4 === 0
          ? sum + (edges[index] * edges[index + 3] - edges[index + 2] * edges[index + 1]) / 2
          : sum,
      0
    );
    expect(area).toBeCloseTo(scene.summary[4], 5);
    expect(scene.sampleCells.map(value => value & 0x7fffffff)).toEqual(
      scene.sampleCells.map(value => value & 0x7fffffff).sort((left, right) => left - right)
    );
    expect(scene.ringOffsets.at(-1)).toBe(scene.positions.length / 2);
  }
});

it('Round 8 SpatialEpi likelihood and supplied-null contracts', () => {
  expect(scan.versions.SpatialEpi).toBe('1.2.8');
  expect(scan.cases).toHaveLength(30);
  for (const scene of scan.cases) {
    const total = scene.cases.reduce((sum, value) => sum + value, 0);
    expect(scene.expected.candidateLikelihoods).toHaveLength(20);
    scene.expected.candidateLikelihoods.forEach((values, index) => {
      expect(values).toHaveLength(scene.windows.length);
      expect(Math.max(...values)).toBe(scene.expected.maxima[index]);
    });
    expect(scene.expected.pValue).toBe(
      (1 +
        scene.expected.maxima.slice(1).filter(value => value >= scene.expected.maxima[0]).length) /
        20
    );
    for (const replicate of scene.replicates) {
      expect(replicate.reduce((sum, value) => sum + value, 0)).toBe(total);
      if (scene.model === 'bernoulli')
        expect(
          replicate.every((value, index) => value <= scene.cases[index] + scene.controls[index])
        ).toBe(true);
      if (scene.model === 'space-time-permutation') {
        for (let zone = 0; zone < 7; zone++) {
          const start = zone * scene.timeBuckets;
          expect(
            replicate.slice(start, start + scene.timeBuckets).reduce((sum, value) => sum + value, 0)
          ).toBe(
            scene.cases
              .slice(start, start + scene.timeBuckets)
              .reduce((sum, value) => sum + value, 0)
          );
        }
        for (let bucket = 0; bucket < scene.timeBuckets; bucket++)
          expect(
            replicate
              .filter((_, index) => index % scene.timeBuckets === bucket)
              .reduce((sum, value) => sum + value, 0)
          ).toBe(
            scene.cases
              .filter((_, index) => index % scene.timeBuckets === bucket)
              .reduce((sum, value) => sum + value, 0)
          );
      }
    }
  }
});

it('Round 8 clustering and weighted category fixtures retain independent contracts', () => {
  for (const scene of clustering.cases) {
    expect(scene.thresholdMargin).toBeGreaterThan(0.1);
    scene.roots.forEach((root, label) => {
      expect(scene.coreMask[root]).toBe(1);
      expect(scene.labels[root]).toBe(label);
      expect(scene.sizes[label]).toBe(scene.labels.filter(value => value === label).length);
    });
  }
  for (const scene of areal.cases) {
    expect(scene.offsets.at(-1)).toBe(scene.neighbors.length);
    for (let target = 0; target < scene.targetCount; target++)
      expect(scene.categoryShares[2 * target] + scene.categoryShares[2 * target + 1]).toBeCloseTo(
        1,
        12
      );
  }
});
