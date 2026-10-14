// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUTerrainFlow,
  getGPUTerrainFlowParameterValues,
  type GPUTerrainFlowProps
} from '../../../src/gpu-terrain/hydrology';
import {makeGPURasterBand, makeGPURasterMetadata} from '../../../src/gpu-raster';
import {WgslReflect} from 'wgsl_reflect';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {getRasterPipelines} from '../../gpu-raster/raster-pipeline-count';

const GRID = makeGPURasterMetadata({width: 6, height: 5});

function createContributor(overrides: Partial<GPUTerrainFlowProps> = {}, withRunoff = false) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUTerrainFlow({
    grid: GRID,
    elevation: makeGPURasterBand(createTransientView(graph, 'elevation', 'float32', 30), GRID, {
      id: 'elevation'
    }),
    parameters: createTransientView(graph, 'settings', 'float32', 8),
    output: {
      accumulation: createTransientView(graph, 'accumulation', 'float32', 30),
      status: {accumulationConverged: createTransientView(graph, 'converged', 'uint32', 1)}
    },
    maximumAccumulationIterations: 3,
    runoff: withRunoff ? createTransientView(graph, 'runoff', 'float32', 30) : undefined,
    ...overrides
  });
  return {device, graph, contributor};
}

it('GPUTerrainFlow validates flowRouting', () => {
  expect(() => createContributor({flowRouting: 'dinf' as never})).toThrow(
    /flowRouting must be d8, d-infinity, mfd-freeman, or mfd-quinn/
  );
  for (const flowRouting of ['d8', 'd-infinity', 'mfd-freeman', 'mfd-quinn'] as const) {
    expect(() => createContributor({flowRouting}).device.destroy()).not.toThrow();
  }
});

it('GPUTerrainFlow routed accumulation uses the D8 node layout plus one routing pass', () => {
  // `runoff` keeps D8 on the pull rounds, the layout the routed accumulation extends.
  const ids = (flowRouting: GPUTerrainFlowProps['flowRouting']) => {
    const {device, graph, contributor} = createContributor({flowRouting}, true);
    const nodeIds = contributor.getCommandNodes(graph).map(node => node.id);
    device.destroy();
    return nodeIds;
  };
  const d8 = ids('d8');
  for (const routing of ['d-infinity', 'mfd-freeman', 'mfd-quinn'] as const) {
    const routed = ids(routing);
    // The routing records are computed once per cell, right before the accumulation sentinels.
    const prepare = routed.indexOf('terrain-flow-routing-prepare');
    expect(routed[prepare + 1]).toBe('terrain-flow-accumulate-init');
    expect(routed.filter(id => id !== 'terrain-flow-routing-prepare')).toEqual(d8);
    expect(routed).toContain('terrain-flow-accumulate-init');
    expect(routed).toContain('terrain-flow-accumulate-round-2');
    expect(routed).toContain('terrain-flow-accumulate-finalize');
    expect(new Set(routed).size).toBe(routed.length);
  }
});

it('getGPUTerrainFlowParameterValues packs flowExponent in slot 6', () => {
  expect(getGPUTerrainFlowParameterValues({flowExponent: 4}, GRID)[6]).toBe(4);
  expect(getGPUTerrainFlowParameterValues({}, GRID)[6]).toBe(0);
  const grid = makeGPURasterMetadata({width: 6, height: 5, cellSize: [2, 3]});
  expect(Array.from(getGPUTerrainFlowParameterValues({flowExponent: 1.1}, grid))).toEqual([
    2,
    3,
    0,
    -15,
    0,
    Infinity,
    Math.fround(1.1),
    0
  ]);
});

it('review C (suspicion 5): routed accumulation rounds read routing records, not facets', () => {
  // D-infinity re-derived all eight facets (two atan2, a sqrt and a ground cell size each) of
  // every donor on every finalize attempt: about 64 facet evaluations per attempt and 200 per walk
  // step, against 16 loads for D8. The facets now run once per cell in routing-prepare.
  for (const flowRouting of ['d-infinity', 'mfd-freeman', 'mfd-quinn'] as const) {
    const {device, graph, contributor} = createContributor({flowRouting});
    const pipelines = getRasterPipelines(contributor.getCommandNodes(graph));
    device.destroy();
    const prepare = pipelines.find(pipeline => pipeline.id === 'terrain-flow-routing-prepare')!;
    const round = pipelines.find(pipeline => pipeline.id === 'terrain-flow-accumulate-round-0')!;
    expect(prepare.source).toMatch(/computeRouting\(index\)/);
    // The round binds the records read-only, so it must not even declare the writer.
    expect(round.source).not.toMatch(/computeRouting/);
    expect(round.source).toMatch(/let code = routing\[routingOffset \+ 2u \* cell\];/);
    expect(() => new WgslReflect(prepare.source)).not.toThrow();
    expect(() => new WgslReflect(round.source)).not.toThrow();
  }
});
