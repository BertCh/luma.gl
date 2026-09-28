// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGPUSortRadixPlan} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';

const WORKGROUP_SIZE = 256;

it('getGPUSortRadixPlan tiles eight keys per thread by default', () => {
  const plan = getGPUSortRadixPlan({
    keys: {length: 1_700_000},
    keyBits: 16,
    digitBits: 4,
    elementsPerThread: 8
  });

  expect(plan.tileSize, 'one workgroup covers 256 threads x 8 keys').toBe(2048);
  expect(plan.workgroupCount, '1.7M keys dispatch 831 workgroups').toBe(831);
  expect(plan.histogramLength, 'the scanned histogram holds 16 buckets x 831 tiles').toBe(13_296);
  expect(plan.passCount, 'a 16-bit key needs four four-bit passes').toBe(4);
});

it('getGPUSortRadixPlan reports the single-element tiling it replaces', () => {
  const plan = getGPUSortRadixPlan({
    keys: {length: 1_700_000},
    keyBits: 16,
    digitBits: 4,
    elementsPerThread: 1
  });

  expect(plan.tileSize, 'one key per thread covers only the workgroup').toBe(WORKGROUP_SIZE);
  expect(plan.workgroupCount, 'the same rows need 6641 workgroups').toBe(6_641);
  expect(plan.histogramLength, 'and a 106256-entry histogram to scan').toBe(106_256);
});

it('getGPUSortRadixPlan halves the pass count for eight-bit digits', () => {
  const wide = getGPUSortRadixPlan({
    keys: {length: 1_000_000},
    keyBits: 32,
    digitBits: 8,
    elementsPerThread: 8
  });
  const narrow = getGPUSortRadixPlan({
    keys: {length: 1_000_000},
    keyBits: 32,
    digitBits: 4,
    elementsPerThread: 8
  });

  expect(wide.passCount, 'four eight-bit passes cover a full 32-bit key').toBe(4);
  expect(narrow.passCount, 'the same key needs eight four-bit passes').toBe(8);
  expect(wide.bucketCount, 'eight-bit digits scan 256 buckets').toBe(256);
  expect(
    wide.workgroupStorageBytes,
    'the 256-bucket ballot mask and cursors fit the guaranteed 16KB of workgroup storage'
  ).toBeLessThanOrEqual(16_384);
});

it('getGPUSortRadixPlan keeps at least one tile for an empty range', () => {
  const plan = getGPUSortRadixPlan({
    keys: {length: 0},
    keyBits: 16,
    digitBits: 4,
    elementsPerThread: 8
  });

  expect(plan.workgroupCount, 'an empty range still reports a single tile').toBe(1);
});
