// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUNetworkIsochrones,
  getGPUNetworkIsochroneParameterValues,
  GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH,
  type GPUNetworkIsochronesProps
} from '../../../src/gpu-network/network-isochrones';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUNetworkIsochronesProps> = {}
): GPUNetworkIsochronesProps {
  const view = <Format extends 'uint32' | 'float32' | 'float32x2' | 'uint32x2' | 'float32x4'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  return {
    offsets: view('uint32', 5),
    neighbors: view('uint32', 6),
    weights: view('float32', 6),
    nodePositions: view('float32x2', 4),
    costs: view('float32', 4),
    breaks: view('float32', 3),
    parameters: view('float32', GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH),
    raster: {
      width: 8,
      height: 8,
      output: {
        triangles: view('float32x2', 3 * 64),
        triangleBands: view('uint32', 64),
        count: view('uint32', 1),
        overflow: view('uint32', 1)
      }
    },
    ...overrides
  };
}

it('GPUNetworkIsochrones declares search, splat and isobands nodes', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'isochrones-node'});
  const plain = new GPUNetworkIsochrones(createProps(graph, {id: 'plain'})).getCommandNodes(graph);
  const withSearch = new GPUNetworkIsochrones(
    createProps(graph, {
      id: 'searching',
      sources: createTransientView(graph, 'search-sources', 'uint32', 2)
    })
  ).getCommandNodes(graph);
  expect(plain.length).toBeGreaterThanOrEqual(6);
  expect(withSearch.length).toBeGreaterThan(plain.length);
});

it('GPUNetworkIsochrones validates its properties', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'isochrones-validate'});
  const props = createProps(graph);
  expect(() => new GPUNetworkIsochrones({...props, raster: undefined})).toThrow(
    /raster or cellOutline/
  );
  expect(() => new GPUNetworkIsochrones({...props, offsets: props.weights as never})).toThrow();
  expect(() => new GPUNetworkIsochrones({...props, raster: {...props.raster!, width: 1}})).toThrow(
    /raster.width/
  );
  expect(
    () => new GPUNetworkIsochrones({...props, raster: {...props.raster!, maximumBufferPixels: 40}})
  ).toThrow(/maximumBufferPixels/);
  expect(() => new GPUNetworkIsochrones({...props, sourceCount: props.neighbors})).toThrow(
    /needs sources/
  );
});

it('getGPUNetworkIsochroneParameterValues packs and validates', () => {
  const values = getGPUNetworkIsochroneParameterValues({
    breakCount: 3,
    extent: [0, 1, 10, 21],
    bufferRadius: 2,
    walkCostPerUnit: 0.25
  });
  expect(Array.from(values)).toEqual([
    3,
    0,
    1,
    10,
    21,
    2,
    0.25,
    0,
    2,
    3.0e38 > 3.4e38 ? 0 : Math.fround(3.0e38),
    0,
    0
  ]);
  expect(() =>
    getGPUNetworkIsochroneParameterValues({breakCount: 1, extent: [0, 0, 0, 1]})
  ).toThrow(/extent/);
  expect(() =>
    getGPUNetworkIsochroneParameterValues({breakCount: -1, extent: [0, 0, 1, 1]})
  ).toThrow(/breakCount/);
});
