// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  getGPUSpaceTimeParameterValues,
  getGPUSpaceTimeRelativeTimes
} from '../../../src/gpu-spatial-analysis/space-time-tests/index';
import {getGPUSpaceTimeInferenceMetadata} from '../../../src/gpu-spatial-analysis/space-time-tests/space-time-parameters';

describe('getGPUSpaceTimeRelativeTimes', () => {
  const epoch = 1_700_000_000;

  it('keeps one-second resolution that raw float32 epoch seconds lose', () => {
    const times = [epoch, epoch + 1, epoch + 2, epoch + 3600];
    // Raw float32 epoch seconds collapse neighbors (spacing 128 s).
    const raw = Float32Array.from(times);
    expect(raw[0]).toBe(raw[1]);
    const relative = getGPUSpaceTimeRelativeTimes(times);
    expect(relative.origin).toBe(epoch);
    expect(Array.from(relative.times)).toEqual([0, 1, 2, 3600]);
    // Spacing at a 3600 s span: 2^(floor(log2 3600) - 23) = 2^-12.
    expect(relative.resolution).toBe(2 ** -12);
  });

  it('uses an explicit origin and handles a constant series', () => {
    const relative = getGPUSpaceTimeRelativeTimes([epoch + 5, epoch + 7], epoch);
    expect(Array.from(relative.times)).toEqual([5, 7]);
    expect(getGPUSpaceTimeRelativeTimes([epoch, epoch]).resolution).toBe(0);
  });

  it('rejects non-finite times and spans coarser than the required resolution', () => {
    expect(() => getGPUSpaceTimeRelativeTimes([0, Number.NaN])).toThrow();
    expect(() => getGPUSpaceTimeRelativeTimes([0, Infinity])).toThrow();
    // 4.6 months of seconds (1.2e7 s) resolve to 1 s, not 1 ms.
    expect(() => getGPUSpaceTimeRelativeTimes([0, 1.2e7], 0, 1e-3)).toThrow(/resolution/);
    expect(() => getGPUSpaceTimeRelativeTimes([0, 1.2e7], 0, 1)).not.toThrow();
  });
});

describe('getGPUSpaceTimeInferenceMetadata', () => {
  it('states the one-sided Monte Carlo alternative from the packed settings', () => {
    expect(getGPUSpaceTimeInferenceMetadata('knox', {seed: 7, permutationCount: 99}, 999)).toEqual({
      alternative: 'greater',
      permutationCount: 99,
      seed: 7,
      multipleTesting: 'none',
      testedRowsDefinition:
        'all unordered event pairs listed in the spatial pair list, counted once'
    });
  });

  it('clamps the count as the kernels do and validates as the packer does', () => {
    const settings = {seed: 1, permutationCount: 500};
    expect(getGPUSpaceTimeInferenceMetadata('mantel', settings, 64).permutationCount).toBe(64);
    expect(getGPUSpaceTimeParameterValues(settings)[2]).toBe(500);
    for (const invalid of [
      {seed: -1, permutationCount: 9},
      {seed: 1, permutationCount: 0},
      {seed: 1, permutationCount: 9, timeThreshold: -1}
    ]) {
      expect(() => getGPUSpaceTimeParameterValues(invalid)).toThrow();
      expect(() => getGPUSpaceTimeInferenceMetadata('knox', invalid, 64)).toThrow();
    }
  });
});
