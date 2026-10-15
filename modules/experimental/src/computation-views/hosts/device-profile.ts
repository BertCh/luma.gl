// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** One analysis device per page, with optional timestamps and raised storage limits. */
export const WEBGPU_ANALYSIS_DEVICE_PROFILE = {
  id: 'webgpu-analysis',
  type: 'webgpu',
  optionalFeatures: ['timestamp-query'],
  requestedStorageBuffersPerShaderStage: 16
} as const;

/** Adapter limits needed by the analysis device profile. */
export type AnalysisAdapterLimits = {
  maxStorageBuffersPerShaderStage: number;
  maxStorageBufferBindingSize: number;
  maxBufferSize: number;
};

/** Uses the showcase's storage policy: up to 16 bindings and the adapter's maximum buffer sizes. */
export function getAnalysisDeviceLimits(
  adapterLimits: AnalysisAdapterLimits
): Record<string, number> {
  return {
    maxStorageBuffersPerShaderStage: Math.min(
      adapterLimits.maxStorageBuffersPerShaderStage,
      WEBGPU_ANALYSIS_DEVICE_PROFILE.requestedStorageBuffersPerShaderStage
    ),
    maxStorageBufferBindingSize: adapterLimits.maxStorageBufferBindingSize,
    maxBufferSize: adapterLimits.maxBufferSize
  };
}

let analysisDevice: Promise<unknown> | undefined;

/** Memoizes creation (including concurrent requests); failed creation can be retried. Callers share a device type. */
export function getAnalysisDevice<TDevice>(
  createDevice: (props: {
    type: 'webgpu';
    optionalFeatures: string[];
    requiredLimits?: Record<string, number>;
  }) => Promise<TDevice>,
  options: {adapterLimits?: AnalysisAdapterLimits} = {}
): Promise<TDevice> {
  if (!analysisDevice) {
    const pending = Promise.resolve().then(() =>
      createDevice({
        type: 'webgpu',
        optionalFeatures: [...WEBGPU_ANALYSIS_DEVICE_PROFILE.optionalFeatures],
        ...(options.adapterLimits
          ? {requiredLimits: getAnalysisDeviceLimits(options.adapterLimits)}
          : {})
      })
    );
    analysisDevice = pending;
    void pending.catch(() => {
      if (analysisDevice === pending) analysisDevice = undefined;
    });
  }
  // Every host in one realm must use the same concrete device type.
  return analysisDevice as Promise<TDevice>;
}

/** Clears the memoized handle for tests; ownership and device destruction belong to the host. */
export function resetAnalysisDevice(): void {
  analysisDevice = undefined;
}
