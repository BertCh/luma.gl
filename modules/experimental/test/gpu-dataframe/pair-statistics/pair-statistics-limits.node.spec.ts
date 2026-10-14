// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {GPURipleyDistanceFunctions} from '../../../src/gpu-dataframe/pair-statistics/gpu-ripley-distance-functions';
import {GPUVariogram} from '../../../src/gpu-dataframe/pair-statistics/gpu-variogram';
import {GPU_RIPLEY_DISTANCE_FUNCTIONS_PARAMETER_LENGTH} from '../../../src/gpu-dataframe/pair-statistics/ripley-distance-parameters';
import {
  GPU_VARIOGRAM_PARAMETER_LENGTH,
  GPU_VARIOGRAM_STATISTICS_LENGTH
} from '../../../src/gpu-dataframe/pair-statistics/variogram-parameters';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  CORE_WORKGROUP_STORAGE_LIMIT_BYTES,
  getStorageBindingCount,
  getWorkgroupMemoryBytes
} from '../../utils/gpu-contributor-test-utils';

// Rule 16 at the largest shapes the constructors accept (report B section 5: both sit at exactly
// 16384 bytes of workgroup memory, so any further `var<workgroup>` would fail on a core device).

let serial = 0;

function expectWithinCoreLimits(nodes: readonly unknown[], label: string): number {
  let largest = 0;
  for (const {id, source} of getGPUCommandNodeKernelSources(nodes)) {
    expect(() => new WgslReflect(source), id).not.toThrow();
    const bytes = getWorkgroupMemoryBytes(source);
    largest = Math.max(largest, bytes);
    expect(bytes, `${label} ${id} workgroup bytes`).toBeLessThanOrEqual(
      CORE_WORKGROUP_STORAGE_LIMIT_BYTES
    );
    expect(getStorageBindingCount(source), `${label} ${id} storage bindings`).toBeLessThanOrEqual(
      8
    );
  }
  return largest;
}

it('GPUVariogram at 512 bins (409 with the robust channel) fits core limits', () => {
  for (const [lagCount, directionCount, robust] of [
    [128, 4, false],
    [409, 1, true]
  ] as const) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `limits-${serial++}`, format, length);
    const binCount = lagCount * directionCount;
    const variogram = new GPUVariogram({
      positions: view('float32x2', 64),
      values: view('float32', 64),
      parameters: view('float32', GPU_VARIOGRAM_PARAMETER_LENGTH),
      gridSize: [16, 16],
      lagCount,
      directionCount,
      output: {
        semivariances: view('float32', binCount),
        pairCounts: view('uint32', binCount),
        meanDistances: view('float32', binCount),
        ...(robust ? {robustSemivariances: view('float32', binCount)} : {}),
        statistics: view('float32', GPU_VARIOGRAM_STATISTICS_LENGTH),
        status: {overflow: view('uint32', 1)}
      }
    });
    const largest = expectWithinCoreLimits(
      variogram.getCommandNodes(graph),
      `variogram ${binCount}`
    );
    expect(largest).toBeGreaterThan(0);
    device.destroy();
  }
});

it('GPURipleyDistanceFunctions at 256 radii fits core limits', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `limits-${serial++}`, format, length);
  const functions = new GPURipleyDistanceFunctions({
    positions: view('float32x2', 64),
    parameters: view('float32', GPU_RIPLEY_DISTANCE_FUNCTIONS_PARAMETER_LENGTH),
    gridSize: [16, 16],
    referenceGrid: [8, 8],
    radiusCount: 256,
    output: {
      g: view('float32', 256),
      f: view('float32', 256),
      j: view('float32', 256)
    }
  });
  expectWithinCoreLimits(functions.getCommandNodes(graph), 'ripley distance 256');
  device.destroy();
});
