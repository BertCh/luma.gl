// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUTerrainFlow,
  getGPUTerrainFlowParameterValues,
  type GPUTerrainFlowProps
} from '../../../src/map-graphs/hydrology';
import {createNullWebGPUDevice} from '../map-graph-test-utils';

function createRecipe(overrides: Partial<GPUTerrainFlowProps> = {}) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const recipe = new GPUTerrainFlow({
    width: 6,
    height: 5,
    elevation: {
      id: 'elevation',
      format: 'float32',
      storage: {kind: 'buffer', values: createTransientView(graph, 'elevation', 'float32', 30)}
    },
    settings: createTransientView(graph, 'settings', 'float32', 8),
    accumulation: createTransientView(graph, 'accumulation', 'float32', 30),
    accumulationConverged: createTransientView(graph, 'converged', 'uint32', 1),
    maxAccumulationIterations: 3,
    ...overrides
  });
  return {device, graph, recipe};
}

it('GPUTerrainFlow validates flowRouting', () => {
  expect(() => createRecipe({flowRouting: 'dinf' as never})).toThrow(
    /flowRouting must be d8, d-infinity, mfd-freeman, or mfd-quinn/
  );
  for (const flowRouting of ['d8', 'd-infinity', 'mfd-freeman', 'mfd-quinn'] as const) {
    expect(() => createRecipe({flowRouting}).device.destroy()).not.toThrow();
  }
});

it('GPUTerrainFlow routed accumulation uses the D8 node layout and ids', () => {
  const ids = (flowRouting: GPUTerrainFlowProps['flowRouting']) => {
    const {device, graph, recipe} = createRecipe({flowRouting});
    const nodeIds = recipe.getCommandNodes(graph).map(node => node.id);
    device.destroy();
    return nodeIds;
  };
  const d8 = ids('d8');
  for (const routing of ['d-infinity', 'mfd-freeman', 'mfd-quinn'] as const) {
    const routed = ids(routing);
    expect(routed).toEqual(d8);
    expect(routed).toContain('terrain-flow-accumulate-init');
    expect(routed).toContain('terrain-flow-accumulate-round-2');
    expect(routed).toContain('terrain-flow-accumulate-finalize');
    expect(new Set(routed).size).toBe(routed.length);
  }
});

it('getGPUTerrainFlowParameterValues packs flowExponent in slot 6', () => {
  expect(getGPUTerrainFlowParameterValues({cellSize: [1, 1], flowExponent: 4})[6]).toBe(4);
  expect(getGPUTerrainFlowParameterValues({cellSize: [1, 1]})[6]).toBe(0);
  expect(
    Array.from(getGPUTerrainFlowParameterValues({cellSize: [2, 3], flowExponent: 1.1}))
  ).toEqual([2, 3, 0, 0, 0, Infinity, Math.fround(1.1), 0]);
});
