// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/gpgpu/gpu-core';
import {getResourceBuffer} from '../network-resource-buffers';
import {expect, it} from 'vitest';
import {
  GPUNetworkIsochrones,
  getGPUNetworkIsochronesParameterValues,
  GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH,
  type GPUNetworkIsochronesProps
} from '../../../src/gpu-network/network-isochrones';
import {
  createNullWebGPUDevice,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';

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
    topology: {
      offsets: view('uint32', 5),
      neighbors: view('uint32', 6),
      weights: view('float32', 6),
      nodePositions: view('float32x2', 4)
    },
    costs: view('float32', 4),
    costBreaks: view('float32', 3),
    parameters: view('float32', GPU_NETWORK_ISOCHRONES_PARAMETER_LENGTH),
    raster: {
      width: 8,
      height: 8,
      output: {
        triangles: view('float32x2', 3 * 64),
        triangleBands: view('uint32', 64),
        status: {count: view('uint32', 1), overflow: view('uint32', 1)}
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
      startNodes: createTransientView(graph, 'search-sources', 'uint32', 2)
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
  expect(
    () =>
      new GPUNetworkIsochrones({
        ...props,
        topology: {...props.topology, offsets: props.topology.weights as never}
      })
  ).toThrow(/topology.offsets/);
  expect(
    () =>
      new GPUNetworkIsochrones({
        ...props,
        topology: {...props.topology, nodePositions: undefined as never}
      })
  ).toThrow(/topology.nodePositions is required/);
  expect(
    () =>
      new GPUNetworkIsochrones({
        ...props,
        topology: {
          ...props.topology,
          weights: createTransientView(
            new GPUCommandGraph(createNullWebGPUDevice()),
            'short-weights',
            'float32',
            5
          )
        }
      })
  ).toThrow(/topology.weights length/);
  expect(() => new GPUNetworkIsochrones({...props, raster: {...props.raster!, width: 1}})).toThrow(
    /raster.width/
  );
  expect(
    () => new GPUNetworkIsochrones({...props, raster: {...props.raster!, maximumBufferPixels: 40}})
  ).toThrow(/maximumBufferPixels/);
  expect(() => new GPUNetworkIsochrones({...props, startCount: props.topology.neighbors})).toThrow(
    /needs startNodes/
  );
});

// Review B-05: the wrapped search's convergence was invisible, so a high-diameter network was
// silently truncated into smaller polygons.
it('GPUNetworkIsochrones forwards converged and localIterations to its search', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'isochrones-converged'});
  for (const withAssignments of [false, true]) {
    const converged = createTransientView(graph, `converged-${serial++}`, 'uint32', 1);
    const nodes = new GPUNetworkIsochrones(
      createProps(graph, {
        id: `search-${withAssignments}`,
        startNodes: createTransientView(graph, `sources-${serial++}`, 'uint32', 2),
        assignments: withAssignments
          ? createTransientView(graph, `assignments-${serial++}`, 'uint32', 4)
          : undefined,
        maximumIterations: 2,
        localIterations: 4,
        output: {status: {converged}}
      })
    ).getCommandNodes(graph);
    const writers = nodes.filter(node =>
      node.resources?.some(
        resource =>
          getResourceBuffer(resource) === converged && resource.usage === 'storage-read-write'
      )
    );
    expect(writers.length).toBeGreaterThan(0);
  }
  const props = createProps(graph);
  const flag = createTransientView(graph, `converged-${serial++}`, 'uint32', 1);
  expect(() => new GPUNetworkIsochrones({...props, output: {status: {converged: flag}}})).toThrow(
    /needs startNodes/
  );
  expect(() => new GPUNetworkIsochrones({...props, localIterations: 4})).toThrow(
    /needs startNodes/
  );
  const startNodes = createTransientView(graph, `sources-${serial++}`, 'uint32', 2);
  expect(
    () =>
      new GPUNetworkIsochrones({
        ...props,
        startNodes,
        output: {
          status: {converged: createTransientView(graph, `converged-${serial++}`, 'uint32', 2)}
        }
      })
  ).toThrow(/output\.status\.converged must contain exactly one row/);
  // Optional, as on GPUNetworkReachability and GPUNetworkServiceAreas.
  expect(() => new GPUNetworkIsochrones({...props, startNodes})).not.toThrow();
});

it('getGPUNetworkIsochronesParameterValues packs and validates', () => {
  const values = getGPUNetworkIsochronesParameterValues({
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
    getGPUNetworkIsochronesParameterValues({breakCount: 1, extent: [0, 0, 0, 1]})
  ).toThrow(/GPUNetworkIsochrones extent/);
  expect(() =>
    getGPUNetworkIsochronesParameterValues({breakCount: -1, extent: [0, 0, 1, 1]})
  ).toThrow(/GPUNetworkIsochrones breakCount/);
});

it('GPUNetworkIsochrones bakes a numeric startCount without a binding', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'isochrones-count'});
  const startNodes = createTransientView(graph, `sources-${serial++}`, 'uint32', 3);
  const getSeedSource = (startCount: number | ReturnType<typeof createTransientView>) => {
    const nodes = new GPUNetworkIsochrones(
      createProps(graph, {id: `count-${serial++}`, startNodes, startCount: startCount as never})
    ).getCommandNodes(graph);
    // Only the search kernel is parsed; the isobands kernels belong to another contributor.
    return getParsedKernelSources(nodes.filter(node => node.id.endsWith('-search-seed')));
  };
  const constant = getSeedSource(2);
  const seedSources = [...constant].filter(([id]) => id.endsWith('-search-seed'));
  expect(seedSources.length).toBeGreaterThan(0);
  for (const [, source] of seedSources) {
    expect(source).toContain('if (index >= 2u) { return; }');
    expect(source).not.toContain('sourceCount[');
  }
  expect(() => new GPUNetworkIsochrones(createProps(graph, {startNodes, startCount: 4}))).toThrow(
    /startCount must be an integer in \[0, 3\]/
  );
});
