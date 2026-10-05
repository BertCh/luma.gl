// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  getGPUPermutationParameterValues,
  GPU_PERMUTATION_PARAMETER_LENGTH
} from '../../../src/map-graphs/permutation-inference';
import {
  getFeistelHalfBits,
  getFeistelPermutationIndex,
  getFeistelRoundKeys,
  getPhilox4x32,
  PhiloxStream
} from '../../../src/map-graphs/permutation-inference/permutation-random';

const hex = (words: number[]) => words.map(word => word.toString(16).padStart(8, '0')).join(' ');

it('Philox 4x32-10 matches the Random123 known-answer vectors', () => {
  expect(hex(getPhilox4x32([0, 0, 0, 0], [0, 0]))).toBe('6627e8d5 e169c58d bc57ac4c 9b00dbd8');
  const ones = 0xffffffff;
  expect(hex(getPhilox4x32([ones, ones, ones, ones], [ones, ones]))).toBe(
    '408f276d 41c83b0e a20bc7c6 6d5451fd'
  );
  expect(
    hex(getPhilox4x32([0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344], [0xa4093822, 0x299f31d0]))
  ).toBe('d16cfe09 94fdcceb 5001e420 24126ea1');
});

it('PhiloxStream walks blocks in order and nextBelow is unbiased and in range', () => {
  const stream = new PhiloxStream([5, 6], 7, 8, 9);
  const first = [
    stream.nextUint32(),
    stream.nextUint32(),
    stream.nextUint32(),
    stream.nextUint32()
  ];
  expect(first).toEqual(getPhilox4x32([0, 7, 8, 9], [5, 6]));
  expect(stream.nextUint32()).toBe(getPhilox4x32([1, 7, 8, 9], [5, 6])[0]);
  for (const bound of [1, 2, 3, 7, 1000, 0x80000001, 0xffffffff]) {
    const draws = new PhiloxStream([1, 2], bound, 0, 0);
    for (let index = 0; index < 2000; index++) {
      const value = draws.nextBelow(bound);
      expect(Number.isInteger(value) && value >= 0 && value < bound).toBe(true);
    }
  }
  const counts = new Array(6).fill(0);
  const dice = new PhiloxStream([3, 4], 0, 0, 0);
  for (let index = 0; index < 60000; index++) {
    counts[dice.nextBelow(6)]++;
  }
  for (const count of counts) {
    // Six standard deviations of a binomial(60000, 1/6).
    expect(Math.abs(count - 10000)).toBeLessThan(6 * Math.sqrt(60000 * (1 / 6) * (5 / 6)));
  }
});

it('the keyed Feistel permutation is a bijection for every domain size and differs per key', () => {
  for (const count of [1, 2, 3, 4, 5, 17, 64, 65, 1000, 4097]) {
    const keys = getFeistelRoundKeys([11, 0], 3);
    const image = new Set<number>();
    for (let index = 0; index < count; index++) {
      image.add(getFeistelPermutationIndex(index, count, keys));
    }
    expect(image.size).toBe(count);
    expect(Math.max(...image)).toBe(count - 1);
  }
  expect(getFeistelHalfBits(1)).toBe(1);
  expect(getFeistelHalfBits(4)).toBe(1);
  expect(getFeistelHalfBits(5)).toBe(2);
  expect(getFeistelHalfBits(2 ** 20)).toBe(10);
  expect(getFeistelHalfBits(2 ** 20 + 1)).toBe(11);
  const a = Array.from({length: 50}, (_, index) =>
    getFeistelPermutationIndex(index, 50, getFeistelRoundKeys([1, 0], 1))
  );
  const b = Array.from({length: 50}, (_, index) =>
    getFeistelPermutationIndex(index, 50, getFeistelRoundKeys([1, 0], 2))
  );
  expect(a).not.toEqual(b);
});

it('getGPUPermutationParameterValues packs the seed words, count and level bits', () => {
  const values = getGPUPermutationParameterValues({
    seed: 2 ** 40 + 5,
    permutations: 999,
    significanceLevel: 0.01
  });
  expect(values.length).toBe(GPU_PERMUTATION_PARAMETER_LENGTH);
  expect(values[0]).toBe(5);
  expect(values[1]).toBe(256);
  expect(values[2]).toBe(999);
  expect(new Float32Array(values.buffer)[3]).toBe(Math.fround(0.01));
  expect(() => getGPUPermutationParameterValues({seed: -1, permutations: 9})).toThrow(/seed/);
  expect(() => getGPUPermutationParameterValues({seed: 1, permutations: 0})).toThrow(
    /permutations/
  );
});
