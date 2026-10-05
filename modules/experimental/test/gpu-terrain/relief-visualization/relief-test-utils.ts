// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect} from 'vitest';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {createInputBuffer} from '../../utils/gpu-contributor-test-utils';

/** Imports a float32 elevation buffer into `graph` as a raster band. */
export function createElevationBand(
  graph: GPUCommandGraph,
  device: Device,
  elevation: Float32Array,
  buffers: Buffer[]
) {
  const buffer = createInputBuffer(device, elevation);
  buffers.push(buffer);
  return {
    id: 'elevation',
    format: 'float32' as const,
    storage: {
      kind: 'buffer' as const,
      values: importGraphBuffer(graph, 'elevation', buffer, 'float32', elevation.length)
    }
  };
}

/** Asserts `actual` equals `expected` within `tolerance`, with identical NaN patterns. Returns the worst error. */
export function expectClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  tolerance: number,
  label = ''
): number {
  expect(actual.length).toBe(expected.length);
  let worst = 0;
  for (let index = 0; index < expected.length; index++) {
    if (Number.isNaN(expected[index])) {
      expect(Number.isNaN(actual[index]), `${label} index ${index} should be NaN`).toBe(true);
      continue;
    }
    expect(Number.isNaN(actual[index]), `${label} index ${index} expected ${expected[index]}`).toBe(
      false
    );
    worst = Math.max(worst, Math.abs(actual[index] - expected[index]));
  }
  expect(worst, `${label} worst error`).toBeLessThan(tolerance);
  return worst;
}

/** Largest absolute finite value, to assert that an output is not all zeros. */
export function getMaximumMagnitude(values: ArrayLike<number>): number {
  let maximum = 0;
  for (let index = 0; index < values.length; index++) {
    if (Number.isFinite(values[index])) {
      maximum = Math.max(maximum, Math.abs(values[index]));
    }
  }
  return maximum;
}

/** Writes NaN into the listed pixels. */
export function punchHoles(
  elevation: Float32Array,
  width: number,
  holes: [number, number][]
): void {
  for (const [column, row] of holes) {
    elevation[row * width + column] = NaN;
  }
}
