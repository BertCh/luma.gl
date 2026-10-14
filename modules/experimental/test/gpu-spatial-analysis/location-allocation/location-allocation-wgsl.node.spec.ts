// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {
  GPULocationAllocation,
  GPU_LOCATION_ALLOCATION_STATUS,
  type GPULocationAllocationOperation
} from '../../../src/gpu-spatial-analysis/location-allocation/gpu-location-allocation';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

const DEMAND_COUNT = 12;
const FACILITY_COUNT = 5;
const ENTRY_COUNT = 40;

/** The cost input: the dense matrix, or CSR sparse costs of {@link ENTRY_COUNT} entries. */
function getCostInput(
  sparse: boolean,
  view: <Format extends 'uint32' | 'float32'>(
    format: Format,
    length: number
  ) => GraphDataView<Format>
) {
  return sparse
    ? {
        sparseCosts: {
          offsets: view('uint32', DEMAND_COUNT + 1),
          facilities: view('uint32', ENTRY_COUNT),
          costs: view('float32', ENTRY_COUNT)
        }
      }
    : {costs: view('float32', DEMAND_COUNT * FACILITY_COUNT)};
}

function getKernelSources(
  operation: GPULocationAllocationOperation,
  maximumSwapIterations?: number,
  sparse = false
): {id: string; source: string}[] {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `wgsl-${serial++}`, format, length);
  const allocation = new GPULocationAllocation({
    ...getCostInput(sparse, view),
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
  for (const sparse of [false, true]) {
    for (const [operation, swaps] of [
      ['p-median', 2],
      ['p-median', 0],
      ['maximum-coverage', 2],
      ['set-covering', undefined]
    ] as const) {
      const sources = getKernelSources(operation, swaps, sparse);
      expect(sources.length, operation).toBeGreaterThan(0);
      for (const {id, source} of sources) {
        expect(() => new WgslReflect(source), `${operation} sparse ${sparse} ${id}`).not.toThrow();
        // The generated kernels never exceed the core limit of eight storage buffers.
        expect(
          getStorageBindingCount(source),
          `${operation} sparse ${sparse} ${id}`
        ).toBeLessThanOrEqual(8);
      }
    }
  }
});

it('GPULocationAllocation pipeline count does not grow with swaps or steps (rule 12) and gates its loops (rule 13)', () => {
  const build = (
    operation: GPULocationAllocationOperation,
    maximumSwapIterations: number | undefined,
    maximumFacilityCount = 2,
    sparse = false
  ) => {
    const graph = new GPUCommandGraph(createNullWebGPUDevice());
    let serial = 0;
    const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
      createTransientView(graph, `gate-${serial++}`, format, length);
    return new GPULocationAllocation({
      id: 'la',
      ...getCostInput(sparse, view),
      facilityCount: FACILITY_COUNT,
      operation,
      maximumFacilityCount,
      maximumSwapIterations,
      coverageDistance: operation === 'p-median' ? undefined : view('float32', 1),
      output: {
        selectedFacilities: view('uint32', FACILITY_COUNT),
        assignments: view('uint32', DEMAND_COUNT),
        objective: view('float32', 1),
        status: view('uint32', GPU_LOCATION_ALLOCATION_STATUS.length)
      }
    }).getCommandNodes(graph) as readonly {id: string; condition?: {mode: string}}[];
  };
  for (const [operation, sparse] of [
    ['p-median', false],
    ['maximum-coverage', false],
    ['p-median', true],
    ['maximum-coverage', true]
  ] as const) {
    const distinct = (swaps: number, steps: number) =>
      getGPUCommandNodePipelineKeys(build(operation, swaps, steps, sparse)).size;
    expect(distinct(3, 2), `${operation} sparse ${sparse}`).toBe(distinct(2, 2));
    expect(distinct(2, 3), `${operation} sparse ${sparse}`).toBe(distinct(2, 2));
    const nodes = build(operation, 2, 2, sparse);
    for (const stage of ['list', 'prepare', sparse ? 'removal' : 'totals', 'score', 'choose']) {
      expect(nodes.find(node => node.id === `la-swap-1-${stage}`)?.condition?.mode).toBe(
        'indirect'
      );
    }
    // The apply kernel writes the gate, so it stays ungated.
    expect(nodes.find(node => node.id === 'la-swap-1-apply')?.condition).toBeUndefined();
    const greedyGated = operation === 'maximum-coverage';
    for (const stage of ['score', 'update']) {
      expect(nodes.find(node => node.id === `la-${stage}-1`)?.condition !== undefined).toBe(
        greedyGated
      );
    }
    expect(nodes.find(node => node.id === 'la-choose-1')?.condition).toBeUndefined();
  }
  for (const sparse of [false, true]) {
    const coverage = getGPUCommandNodePipelineKeys(
      build('set-covering', undefined, 3, sparse)
    ).size;
    expect(getGPUCommandNodePipelineKeys(build('set-covering', undefined, 4, sparse)).size).toBe(
      coverage
    );
  }
});
