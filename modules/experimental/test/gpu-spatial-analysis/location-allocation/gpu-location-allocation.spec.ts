// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/test-utils';
import {expect, it} from 'vitest';
import {
  GPULocationAllocation,
  GPU_LOCATION_ALLOCATION_STATUS,
  type GPULocationAllocationOperation
} from '../../../src/gpu-spatial-analysis/location-allocation';
import {importGraphBuffer} from '../../../src/utils/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

async function runAllocation(
  operation: GPULocationAllocationOperation,
  costs: Float32Array,
  facilityCount: number,
  maximumFacilityCount: number,
  coverageDistance?: number
) {
  const device = await getWebGPUTestDevice();
  if (!device) return null;
  const demandCount = costs.length / facilityCount;
  const costsBuffer = createInputBuffer(device, costs);
  const coverageBuffer =
    coverageDistance === undefined
      ? undefined
      : createInputBuffer(device, Float32Array.from([coverageDistance]));
  const selectedBuffer = createOutputBuffer(device, facilityCount);
  const assignmentsBuffer = createOutputBuffer(device, demandCount);
  const assignedCostsBuffer = createOutputBuffer(device, demandCount);
  const objectiveBuffer = createOutputBuffer(device, 1);
  const statusBuffer = createOutputBuffer(device, GPU_LOCATION_ALLOCATION_STATUS.length);
  const graph = new GPUCommandGraph(device, {id: `location-allocation-${operation}`});
  graph.add(
    new GPULocationAllocation({
      costs: importGraphBuffer(graph, 'costs', costsBuffer, 'float32', costs.length),
      facilityCount,
      operation,
      maximumFacilityCount,
      coverageDistance: coverageBuffer
        ? importGraphBuffer(graph, 'coverage', coverageBuffer, 'float32', 1)
        : undefined,
      output: {
        selectedFacilities: importGraphBuffer(
          graph,
          'selected',
          selectedBuffer,
          'uint32',
          facilityCount
        ),
        assignments: importGraphBuffer(
          graph,
          'assignments',
          assignmentsBuffer,
          'uint32',
          demandCount
        ),
        assignedCosts: importGraphBuffer(
          graph,
          'assigned-costs',
          assignedCostsBuffer,
          'float32',
          demandCount
        ),
        objective: importGraphBuffer(graph, 'objective', objectiveBuffer, 'float32', 1),
        status: importGraphBuffer(
          graph,
          'status',
          statusBuffer,
          'uint32',
          GPU_LOCATION_ALLOCATION_STATUS.length
        )
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    selected: await readUint32(selectedBuffer, facilityCount),
    assignments: await readUint32(assignmentsBuffer, demandCount),
    assignedCosts: await readFloat32(assignedCostsBuffer, demandCount),
    objective: (await readFloat32(objectiveBuffer, 1))[0],
    status: await readUint32(statusBuffer, GPU_LOCATION_ALLOCATION_STATUS.length)
  };
  compiled.destroy();
  costsBuffer.destroy();
  coverageBuffer?.destroy();
  selectedBuffer.destroy();
  assignmentsBuffer.destroy();
  assignedCostsBuffer.destroy();
  objectiveBuffer.destroy();
  statusBuffer.destroy();
  return result;
}

it('GPULocationAllocation greedily minimizes p-median cost', async () => {
  const result = await runAllocation(
    'p-median',
    Float32Array.from([0, 10, 20, 10, 0, 10, 20, 10, 0]),
    3,
    1
  );
  if (!result) return;
  expect(result.selected).toEqual([0, 1, 0]);
  expect(result.assignments).toEqual([1, 1, 1]);
  expect(result.assignedCosts).toEqual([10, 0, 10]);
  expect(result.objective).toBe(20);
  expect(result.status).toEqual([1, 0, 1, 0]);
});

it('GPULocationAllocation reports maximum-coverage budget completion', async () => {
  const result = await runAllocation(
    'maximum-coverage',
    Float32Array.from([0, 10, 10, 0, 10, 10]),
    2,
    1,
    0
  );
  if (!result) return;
  expect(result.selected).toEqual([1, 0]);
  expect(result.objective).toBe(1);
  expect(result.status).toEqual([1, 2, 1, 0]);
});

it('GPULocationAllocation completes deterministic set covering', async () => {
  const result = await runAllocation('set-covering', Float32Array.from([0, 10, 10, 0]), 2, 2, 0);
  if (!result) return;
  expect(result.selected).toEqual([1, 1]);
  expect(result.assignments).toEqual([0, 1]);
  expect(result.objective).toBe(2);
  expect(result.status).toEqual([2, 0, 1, 0]);
});
