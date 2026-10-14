// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';

const hasWebGPU = typeof navigator !== 'undefined' && 'gpu' in navigator;

// Contributor GPU specs return early when no WebGPU device exists, so a run without one reports
// green with no assertions. This spec makes that state visible: it is skipped (and listed as
// skipped) when the browser has no WebGPU, and fails when WebGPU exists but yields no device.
it.skipIf(!hasWebGPU)('WebGPU test devices are available for contributor GPU specs', async () => {
  const maxDevice = await getWebGPUTestDevice();
  const coreDevice = await getWebGPUTestDevice('core');
  expect(maxDevice, "featureLevel 'max' WebGPU test device").toBeTruthy();
  expect(coreDevice, "featureLevel 'core' WebGPU test device").toBeTruthy();
  expect(coreDevice?.limits.maxStorageBuffersPerShaderStage).toBeGreaterThanOrEqual(8);
});
