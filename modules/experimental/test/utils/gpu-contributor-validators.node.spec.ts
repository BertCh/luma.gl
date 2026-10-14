// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, GraphVectorView} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, it} from 'vitest';
import {
  validateCapacity,
  validateExtent,
  validatePackedView,
  validatePowerOfTwo
} from '../../src/utils/gpu-contributor-utils';
import {createNullWebGPUDevice} from './gpu-contributor-test-utils';

// Rule 17 (A-16): user-facing prop errors go through one set of short shared validators.

describe('shared contributor validators', () => {
  it('validatePackedView accepts one packed view and rejects vectors and wrong formats', () => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'validators'});
    const view = createTransientView(graph, 'rows', 'uint32', 4);
    expect(() => validatePackedView(view, ['uint32'], 'test rows')).not.toThrow();
    expect(() => validatePackedView(view, ['float32'], 'test rows')).toThrow(/test rows/);
    const vector = new GraphVectorView({
      id: 'rows-vector',
      name: 'rows-vector',
      format: 'uint32',
      length: 4,
      valueLength: 4,
      stride: 1,
      byteStride: 4,
      rowByteLength: 4,
      data: [view]
    });
    expect(() => validatePackedView(vector, ['uint32'], 'test rows')).toThrow(
      'test rows must be one packed view'
    );
  });

  it('validateCapacity bounds integers', () => {
    expect(() => validateCapacity('test', 'pairCapacity', 1)).not.toThrow();
    expect(() => validateCapacity('test', 'pairCapacity', 0xffffffff)).not.toThrow();
    for (const value of [0, -1, 1.5, Number.NaN, 2 ** 32]) {
      expect(() => validateCapacity('test', 'pairCapacity', value)).toThrow(
        'test pairCapacity must be an integer in [1, 4294967295]'
      );
    }
    expect(() => validateCapacity('test', 'rounds', 0, 0, 8)).not.toThrow();
    expect(() => validateCapacity('test', 'rounds', 9, 0, 8)).toThrow('[0, 8]');
  });

  it('validateExtent needs four finite values with max > min', () => {
    expect(() => validateExtent('test', [0, 0, 1, 1])).not.toThrow();
    for (const extent of [
      [0, 0, 0, 1],
      [0, 1, 1, 0],
      [0, 0, Infinity, 1],
      [0, 0, 1]
    ]) {
      expect(() => validateExtent('test', extent)).toThrow('test extent must be finite');
    }
    expect(() => validateExtent('test', [1, 0, 0, 1], 'bounds')).toThrow('test bounds');
  });

  it('validatePowerOfTwo bounds powers of two', () => {
    for (const value of [1, 2, 64, 2 ** 31]) {
      expect(() => validatePowerOfTwo('test', 'leafCapacity', value)).not.toThrow();
    }
    for (const value of [0, 3, 6, 2.5, 2 ** 32]) {
      expect(() => validatePowerOfTwo('test', 'leafCapacity', value)).toThrow(
        'test leafCapacity must be a power of two'
      );
    }
    expect(() => validatePowerOfTwo('test', 'blockSize', 512, 1, 256)).toThrow('[1, 256]');
  });
});
