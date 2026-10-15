// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test, vi} from 'vitest';
import {
  getAnalysisDevice,
  getAnalysisDeviceLimits,
  resetAnalysisDevice,
  WEBGPU_ANALYSIS_DEVICE_PROFILE
} from '../../../src/computation-views/hosts/device-profile';

const limits = {
  maxStorageBuffersPerShaderStage: 32,
  maxStorageBufferBindingSize: 2 ** 28,
  maxBufferSize: 2 ** 30
};

describe('analysis device profile', () => {
  test('caps bindings at 16 and preserves adapter buffer sizes', () => {
    expect(getAnalysisDeviceLimits(limits)).toEqual({
      ...limits,
      maxStorageBuffersPerShaderStage: 16
    });
    expect(getAnalysisDeviceLimits({...limits, maxStorageBuffersPerShaderStage: 8})).toEqual({
      ...limits,
      maxStorageBuffersPerShaderStage: 8
    });
    expect(WEBGPU_ANALYSIS_DEVICE_PROFILE.optionalFeatures).toEqual(['timestamp-query']);
  });
  test('memoizes concurrent creation, passing timestamp and raised limits once', async () => {
    resetAnalysisDevice();
    const device = {id: 'device'};
    const createDevice = vi.fn(async () => device);
    const first = getAnalysisDevice(createDevice, {adapterLimits: limits});
    const second = getAnalysisDevice(createDevice);
    expect(first).toBe(second);
    expect(await first).toBe(device);
    expect(createDevice).toHaveBeenCalledTimes(1);
    expect(createDevice).toHaveBeenCalledWith({
      type: 'webgpu',
      optionalFeatures: ['timestamp-query'],
      requiredLimits: {...limits, maxStorageBuffersPerShaderStage: 16}
    });
    resetAnalysisDevice();
    await getAnalysisDevice(createDevice);
    expect(createDevice).toHaveBeenLastCalledWith({
      type: 'webgpu',
      optionalFeatures: ['timestamp-query']
    });
    resetAnalysisDevice();
  });
  test('retries failed creation, including synchronous factory throws', async () => {
    resetAnalysisDevice();
    await expect(
      getAnalysisDevice(() => {
        throw Error('No adapter');
      })
    ).rejects.toThrow('No adapter');
    const createDevice = vi.fn(async () => ({id: 'retry'}));
    await expect(getAnalysisDevice(createDevice)).resolves.toEqual({id: 'retry'});
    expect(createDevice).toHaveBeenCalledTimes(1);
    resetAnalysisDevice();
  });
});
