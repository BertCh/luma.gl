// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPU_TERRAIN_HYDROLOGIC_INDICES_PARAMETER_LENGTH,
  GPUTerrainHydrologicIndices,
  getGPUTerrainHydrologicIndicesParameterValues,
  type GPUTerrainHydrologicIndicesOutput,
  type GPUTerrainHydrologicIndicesProps
} from '../../../src/gpu-terrain/hydrology/gpu-terrain-hydrologic-indices';
import {makeGPURasterBand, makeGPURasterMetadata} from '../../../src/gpu-raster';
import {getRasterGridGeoreference} from '../../../src/analysis-core/raster-grid';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const GRID = makeGPURasterMetadata({width: 6, height: 5});

/** Props overrides whose `output` members replace the default `wetnessIndex` output. */
type Overrides = Partial<Omit<GPUTerrainHydrologicIndicesProps, 'output'>> & {
  output?: GPUTerrainHydrologicIndicesOutput;
};

function createFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let instance = 0;
  const create = ({output, ...overrides}: Overrides = {}) => {
    instance++;
    return new GPUTerrainHydrologicIndices({
      grid: GRID,
      elevation: makeGPURasterBand(
        createTransientView(graph, `elevation-${instance}`, 'float32', 30),
        GRID,
        {id: `elevation-${instance}`}
      ),
      accumulation: createTransientView(graph, `accumulation-${instance}`, 'float32', 30),
      parameters: createTransientView(graph, `settings-${instance}`, 'float32', 8),
      ...overrides,
      output: output ?? {
        wetnessIndex: createTransientView(graph, `wetness-${instance}`, 'float32', 30)
      }
    });
  };
  return {device, graph, create};
}

it('getGPUTerrainHydrologicIndicesParameterValues packs settings', () => {
  expect(GPU_TERRAIN_HYDROLOGIC_INDICES_PARAMETER_LENGTH).toBe(8);
  // Cell size and row edges come from the grid.
  const grid = makeGPURasterMetadata({width: 6, height: 5, cellSize: [2, 3], origin: [0, 20]});
  const {cellSize, northEdge, southEdge} = getRasterGridGeoreference(grid);
  expect([cellSize, northEdge, southEdge]).toEqual([[2, 3], 20, 5]);
  expect(
    Array.from(getGPUTerrainHydrologicIndicesParameterValues({minimumSlope: 0.125}, grid))
  ).toEqual([2, 3, 20, 5, 0.125, 0, 0, 0]);
  expect(getGPUTerrainHydrologicIndicesParameterValues({}, GRID)[4]).toBeCloseTo(0.001, 9);
  expect(() =>
    getGPUTerrainHydrologicIndicesParameterValues({}, GRID, new Float32Array(7))
  ).toThrow(/8 values/);
});

it('GPUTerrainHydrologicIndices prefixes node ids', () => {
  const {device, graph, create} = createFixture();
  const contributor = create({id: 'twi'});
  expect(contributor.getCommandNodes(graph).map(node => node.id)).toEqual([
    'twi-elevation',
    'twi-indices'
  ]);
  device.destroy();
});

it('GPUTerrainHydrologicIndices validates its props', () => {
  const {device, graph, create} = createFixture();
  const view = <Format extends 'float32' | 'uint32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, name, format, length);
  expect(() => create({grid: {...GRID, width: 0}})).toThrow(/dimensions/);
  expect(() => create({output: {}})).toThrow(/at least one output/);
  expect(() => create({parameters: view('s7', 'float32', 7)})).toThrow(/parameters/);
  expect(() => create({cellSizeMode: 'polar' as never})).toThrow(/cellSizeMode/);
  expect(() => create({accumulation: view('short', 'float32', 29)})).toThrow(/one value per cell/);
  expect(() => create({output: {slope: view('uint', 'uint32', 30) as never}})).toThrow(/float32/);
  const shared = view('shared', 'float32', 30);
  expect(() => create({output: {wetnessIndex: shared, slope: shared}})).toThrow(/share buffers/);
  const accumulation = view('input', 'float32', 30);
  expect(() => create({accumulation, output: {wetnessIndex: accumulation}})).toThrow(
    /share buffers/
  );
  const otherGraph = new GPUCommandGraph(device);
  expect(() =>
    create({
      output: {wetnessIndex: createTransientView(otherGraph, 'foreign', 'float32', 30)}
    }).getCommandNodes(graph)
  ).toThrow(/belong to the target graph/);
  device.destroy();
});
