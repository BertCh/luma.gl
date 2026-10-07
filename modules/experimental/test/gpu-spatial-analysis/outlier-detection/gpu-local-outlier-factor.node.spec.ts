// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  getGPULocalOutlierFactorParameterValues,
  GPU_LOCAL_OUTLIER_FACTOR_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/outlier-detection/index';

describe('getGPULocalOutlierFactorParameterValues', () => {
  it('packs threshold and density floor with defaults', () => {
    const values = getGPULocalOutlierFactorParameterValues();
    expect(values.length).toBe(GPU_LOCAL_OUTLIER_FACTOR_PARAMETER_LENGTH);
    expect(values[0]).toBe(1.5);
    expect(values[1]).toBe(Math.fround(1e-10));
    expect(
      Array.from(getGPULocalOutlierFactorParameterValues({threshold: 2, densityFloor: 0.5}))
    ).toEqual([2, 0.5, 0, 0]);
  });

  it('rejects a non-positive density floor and a NaN threshold', () => {
    expect(() => getGPULocalOutlierFactorParameterValues({densityFloor: 0})).toThrow();
    expect(() => getGPULocalOutlierFactorParameterValues({densityFloor: 1e-50})).toThrow();
    expect(() => getGPULocalOutlierFactorParameterValues({threshold: NaN})).toThrow();
  });
});
