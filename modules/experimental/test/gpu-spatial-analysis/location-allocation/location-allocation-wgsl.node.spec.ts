// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {
  GPULocationAllocation,
  GPU_LOCATION_ALLOCATION_STATUS,
  type GPULocationAllocationOperation
} from '../../../src/gpu-spatial-analysis/location-allocation/gpu-location-allocation';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

const DEMAND_COUNT = 12;
const FACILITY_COUNT = 5;

function getKernelSources(
  operation: GPULocationAllocationOperation,
  maximumSwapIterations?: number
): {id: string; source: string}[] {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `wgsl-${serial++}`, format, length);
  const allocation = new GPULocationAllocation({
    costs: view('float32', DEMAND_COUNT * FACILITY_COUNT),
    facilityCount: FACILITY_COUNT,
    demandWeights: view('float32', DEMAND_COUNT),
    operation,
    maximumFacilityCount: 2,
    maximumSwapIterations,
    coverageDistance: operation === 'p-median' ? undefined : view('float32', 1),
    output: {
      selectedFacilities: view('uint32', FACILITY_COUNT),
      assignments: view('uint32', DEMAND_COUNT),
      assignedCosts: view('float32', DEMAND_COUNT),
      objective: view('float32', 1),
      status: view('uint32', GPU_LOCATION_ALLOCATION_STATUS.length)
    }
  });
  const nodes = allocation.getCommandNodes(graph);
  const sources = getGPUCommandNodeKernelSources(nodes);
  device.destroy();
  return sources;
}

it('GPULocationAllocation kernels parse as WGSL (syntax only; no type checking)', () => {
  for (const [operation, swaps] of [
    ['p-median', 2],
    ['p-median', 0],
    ['maximum-coverage', 2],
    ['set-covering', undefined]
  ] as const) {
    const sources = getKernelSources(operation, swaps);
    expect(sources.length, operation).toBeGreaterThan(0);
    for (const {id, source} of sources) {
      expect(() => new WgslReflect(source), `${operation} ${id}`).not.toThrow();
    }
  }
});
