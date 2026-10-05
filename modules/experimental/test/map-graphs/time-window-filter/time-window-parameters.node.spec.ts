// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getGPUTimeWindowParameterValues,
  splitTimestamps
} from '../../../src/map-graphs/time-window-filter';

it('getGPUTimeWindowParameterValues packs high and low window parts', () => {
  expect(Array.from(getGPUTimeWindowParameterValues({start: 10, end: 20}))).toEqual([
    10, 0, 20, 0, 0, 0, 0, 0
  ]);
  const start = 1_700_000_015;
  const values = getGPUTimeWindowParameterValues({
    start,
    end: start + 1,
    startFadeDuration: 2
  });
  expect(values[0]).toBe(Math.fround(start));
  expect(values[0] + values[1]).toBe(start);
  expect(values[4]).toBe(2);

  const target = new Float32Array(8);
  expect(getGPUTimeWindowParameterValues({start: 1, end: 2}, target)).toBe(target);
  expect(() => getGPUTimeWindowParameterValues({start: 1, end: 2}, new Float32Array(4))).toThrow();
  expect(() => getGPUTimeWindowParameterValues({start: 1, end: 2, endFadeDuration: -1})).toThrow();
  expect(() => getGPUTimeWindowParameterValues({start: Number.NaN, end: 2})).toThrow();
});

it('splitTimestamps splits values into exact float32 pairs', () => {
  const values = [1_700_000_010, 0.5];
  const {high, low} = splitTimestamps(values);
  for (let index = 0; index < values.length; index++) {
    expect(high[index] + low[index]).toBe(values[index]);
    expect(Math.fround(high[index])).toBe(high[index]);
  }
  expect(low[1]).toBe(0);
});
