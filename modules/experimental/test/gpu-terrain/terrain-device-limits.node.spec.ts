// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {describe, expect, it} from 'vitest';
import {makeGPURasterMetadata} from '../../src/gpu-raster';
import {
  GPUSimpleLocalRelief,
  GPUTerrainSpikeRepair,
  GPUTerrainTopographicPosition,
  getGPUTerrainSpikeRepairMaximumCellCount,
  getGPUTerrainSummedAreaMaximumCellCount
} from '../../src/gpu-terrain';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';
import {createBand} from './terrain-test-utils';

/** A fake device whose storage bindings and buffers hold 1 GiB, like a large desktop GPU. */
function createLargeDevice(): Device {
  const device = createNullWebGPUDevice();
  Object.defineProperty(device, 'limits', {
    value: {
      ...device.limits,
      maxStorageBufferBindingSize: 2 ** 30,
      maxBufferSize: 2 ** 30
    }
  });
  return device;
}

const GRID = makeGPURasterMetadata({width: 4096, height: 4096});
const CELL_COUNT = 4096 * 4096;

function declareSpikeRepair(device: Device) {
  const graph = new GPUCommandGraph(device);
  new GPUTerrainSpikeRepair({
    grid: GRID,
    elevation: createBand(graph, 'elevation', CELL_COUNT),
    output: {
      values: createTransientView(graph, 'values', 'float32', CELL_COUNT),
      outputValidity: createTransientView(graph, 'validity', 'uint32', CELL_COUNT)
    }
  }).getCommandNodes(graph);
}

function declareTopographicPosition(device: Device) {
  const graph = new GPUCommandGraph(device);
  return new GPUTerrainTopographicPosition({
    grid: GRID,
    elevation: createBand(graph, 'elevation', CELL_COUNT),
    scales: [{radius: 5}],
    output: {outputValidity: createTransientView(graph, 'validity', 'uint32', CELL_COUNT)}
  }).getCommandNodes(graph);
}

function declareSimpleLocalRelief(device: Device) {
  const graph = new GPUCommandGraph(device);
  return new GPUSimpleLocalRelief({
    grid: GRID,
    maximumRadius: 10,
    elevation: createBand(graph, 'elevation', CELL_COUNT),
    parameters: createTransientView(graph, 'parameters', 'float32', 4),
    output: {relief: createTransientView(graph, 'relief', 'float32', CELL_COUNT)}
  })
    .getCommandNodes(graph)
    .map(node => node.id);
}

describe('terrain maxima follow the device binding limit', () => {
  it('derives the maxima from maxStorageBufferBindingSize', () => {
    const core = createNullWebGPUDevice();
    const large = createLargeDevice();
    expect(getGPUTerrainSpikeRepairMaximumCellCount(core)).toBe(4194304);
    expect(getGPUTerrainSpikeRepairMaximumCellCount(large)).toBe(2 ** 25);
    expect(getGPUTerrainSummedAreaMaximumCellCount(core)).toBe(11184810);
    expect(getGPUTerrainSummedAreaMaximumCellCount(large)).toBeGreaterThanOrEqual(CELL_COUNT);
  });

  it('GPUTerrainSpikeRepair accepts a 4096 x 4096 grid on a 1 GiB device only', () => {
    expect(() => declareSpikeRepair(createLargeDevice())).not.toThrow();
    expect(() => declareSpikeRepair(createNullWebGPUDevice())).toThrow(
      'terrain-spike-repair grid cells must be at most 4194304'
    );
  });

  it('GPUTerrainTopographicPosition accepts a 4096 x 4096 grid on a 1 GiB device only', () => {
    expect(declareTopographicPosition(createLargeDevice()).length).toBeGreaterThan(0);
    expect(() => declareTopographicPosition(createNullWebGPUDevice())).toThrow(
      'terrain-topographic-position grid cells must be at most 11184810'
    );
  });

  it('GPUSimpleLocalRelief takes the exact table on a 1 GiB device and the fallback on core', () => {
    const large = declareSimpleLocalRelief(createLargeDevice());
    const core = declareSimpleLocalRelief(createNullWebGPUDevice());
    expect(large.some(id => id.includes('summed-area'))).toBe(true);
    expect(core.some(id => id.includes('summed-area'))).toBe(false);
  });
});
