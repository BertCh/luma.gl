// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync, statSync} from 'node:fs';
import {resolve} from 'node:path';
import {expect, it} from 'vitest';

const directory = resolve('modules/experimental/test/oracles/fixtures');
function readFixture(topic: string) {
  const path = resolve(directory, `parity-wp6-round8-${topic}.json`);
  expect(statSync(path).size).toBeLessThanOrEqual(1500000);
  return JSON.parse(readFileSync(path, 'utf8'));
}

it('WP6 corrected references cover the full window/correction/resolution product', () => {
  const ripley = readFixture('ripley');
  expect(ripley.versions['spatstat.explore']).toBe('3.8.3');
  expect(ripley.cases).toHaveLength(27);
  const names = new Set(ripley.cases.map((scene: {name: string}) => scene.name));
  for (const window of ['rectangle', 'city', 'view']) {
    for (const count of [20, 30, 60]) {
      for (const correction of ['none', 'border', 'isotropic']) {
        expect(names.has(`${window}/${count}/${correction}`)).toBe(true);
      }
    }
    const distance = readFixture(`distance-${window}`);
    expect(distance.cases).toHaveLength(36);
    const distanceNames = new Set(distance.cases.map((scene: {name: string}) => scene.name));
    for (const count of [20, 30, 60]) {
      for (const grid of [24, 47, 94]) {
        for (const correction of ['none', 'border', 'kaplan-meier', 'hanisch']) {
          expect(distanceNames.has(`${window}/${count}/${grid}/${correction}`)).toBe(true);
        }
      }
    }
  }
  for (const scene of readFixture('indices').cases) {
    expect(scene.expected.quadratStatistics[0]).toBe(
      scene.expected.quadratAreas.filter((area: number) => area > 0).length
    );
    expect(
      scene.expected.quadratCounts.reduce((sum: number, value: number) => sum + value, 0)
    ).toBe(scene.expected.clarkEvans[0]);
    expect(scene.expected.nearestNeighborIds).toHaveLength(scene.positions.length / 2);
  }
});

it('WP6 dataframe showcase fixtures retain every register and every option combination', () => {
  const profiles = readFixture('profiles').cases;
  expect(
    profiles.filter((scene: {name: string}) => scene.name.startsWith('showcase-'))
  ).toHaveLength(20);
  for (const scene of profiles) {
    expect(scene.expected.registers).toHaveLength(scene.columns.length * 2 ** scene.precision);
    expect(scene.expected.histograms).toHaveLength(scene.columns.length * scene.bins);
  }
  const correlogram = readFixture('correlogram').cases.filter((scene: {name: string}) =>
    scene.name.startsWith('showcase-')
  );
  expect(correlogram.map((scene: {bandCount: number}) => scene.bandCount)).toEqual(
    Array.from({length: 15}, (_, index) => 4 + 2 * index)
  );
  for (const scene of correlogram) expect(scene.runs).toHaveLength(4);
  for (const scaler of ['min-max', 'z-score', 'rank']) {
    for (const scene of readFixture(`composite-${scaler}`).cases) {
      expect(scene.indicatorCount).toBe(10);
      expect(scene.mask).toHaveLength(791);
      expect(scene.expected.scaled).toHaveLength(7910);
    }
  }
});

it('WP6 NumPy rank adapter independently pins an observable f32 tie transition', () => {
  const scene = readFixture('quantiles').cases.find(
    (entry: {name: string}) => entry.name === 'rank-rounding'
  );
  const index = scene.probabilities.findIndex(
    (value: number) => Math.fround(value) === Math.fround(0.1)
  );
  expect(scene.expected.quantiles.nearest[index]).toBe(0);
  expect(scene.expected.quantiles.linear[index]).toBe(5);
});
