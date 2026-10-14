// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from './require-test-device';

// Contributor GPU specs call `requireTestDevice`, which fails them in browser-backed projects when
// no WebGPU device exists. This canary states the provisioning requirement once: it fails under
// the headless and browser projects when the browser has no WebGPU or WebGPU yields no device.
it('WebGPU test devices are available for contributor GPU specs', async () => {
  expect(typeof navigator !== 'undefined' && 'gpu' in navigator, 'navigator.gpu').toBe(true);
  const maxDevice = await getWebGPUTestDevice();
  const coreDevice = await getWebGPUTestDevice('core');
  expect(maxDevice, "featureLevel 'max' WebGPU test device").toBeTruthy();
  expect(coreDevice, "featureLevel 'core' WebGPU test device").toBeTruthy();
  requireTestDevice(coreDevice);
  expect(coreDevice.limits.maxStorageBuffersPerShaderStage).toBeGreaterThanOrEqual(8);
});

it('requireTestDevice fails, rather than skips, in browser-backed projects', () => {
  expect(() => requireTestDevice(null)).toThrow('No WebGPU test device');
});
