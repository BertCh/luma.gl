// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getGPUTimeWindowFilterParameterValues,
  splitTimestamps
} from '../../../src/gpu-dataframe/time-window-filter';

it('getGPUTimeWindowFilterParameterValues packs high and low window parts', () => {
  expect(Array.from(getGPUTimeWindowFilterParameterValues({start: 10, end: 20}))).toEqual([
    10, 0, 20, 0, 0, 0, 0, 0
  ]);
  const start = 1_700_000_015;
  const values = getGPUTimeWindowFilterParameterValues({
    start,
    end: start + 1,
    startFadeDuration: 2
  });
  expect(values[0]).toBe(Math.fround(start));
  expect(values[0] + values[1]).toBe(start);
  expect(values[4]).toBe(2);

  const target = new Float32Array(8);
  expect(getGPUTimeWindowFilterParameterValues({start: 1, end: 2}, target)).toBe(target);
  expect(() =>
    getGPUTimeWindowFilterParameterValues({start: 1, end: 2}, new Float32Array(4))
  ).toThrow();
  expect(() =>
    getGPUTimeWindowFilterParameterValues({start: 1, end: 2, endFadeDuration: -1})
  ).toThrow();
  expect(() => getGPUTimeWindowFilterParameterValues({start: Number.NaN, end: 2})).toThrow();
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
