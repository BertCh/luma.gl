// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, expectTypeOf, it} from 'vitest';
import {
  validateGPUStatusPort,
  type GPULonLatPolygonStatusPort,
  type GPUStatusScalar
} from '../../../src/analysis-core/contracts/status';

function createStatusScalar(format: string): GPUStatusScalar {
  return {format, length: 1} as GPUStatusScalar;
}

describe('longitude-latitude polygon status contract', () => {
  it('exposes exactly the antimeridian and pole uncertainty keys', () => {
    expectTypeOf<keyof GPULonLatPolygonStatusPort>().toEqualTypeOf<
      'antimeridianUncertainCount' | 'poleUncertainCount'
    >();
    const keys: Array<keyof GPULonLatPolygonStatusPort> = [
      'antimeridianUncertainCount',
      'poleUncertainCount'
    ];
    expect(keys).toEqual(['antimeridianUncertainCount', 'poleUncertainCount']);
  });

  it('accepts uint32 scalar uncertainty counts', () => {
    const status: GPULonLatPolygonStatusPort = {
      antimeridianUncertainCount: createStatusScalar('uint32'),
      poleUncertainCount: createStatusScalar('uint32')
    };
    expect(() => validateGPUStatusPort('X', status)).not.toThrow();
  });

  it('rejects a float32 pole uncertainty count with the exact diagnostic', () => {
    expect(() =>
      validateGPUStatusPort('X', {
        antimeridianUncertainCount: createStatusScalar('uint32'),
        poleUncertainCount: createStatusScalar('float32')
      })
    ).toThrowError(new Error('X poleUncertainCount must contain one uint32 row'));
  });
});
