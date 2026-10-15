// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/gpgpu/gpu-core';
import {afterEach, expect, it} from 'vitest';
import {
  GPUCostDistance,
  getGPURasterRelaxationContinuationStateLength
} from '../../../src/gpu-raster/cost-distance';
import {makeGPURasterBand, makeGPURasterMetadata} from '../../../src/gpu-raster';
import {
  createRasterGatedLoopNodes,
  createRasterIterationState,
  createRasterTiledRelaxation,
  createRasterTiledRelaxationNodes
} from '../../../src/gpu-raster/cost-distance/raster-relaxation';
import {createWGSLKernelNode} from '../../../src/utils/wgsl-kernel-nodes';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

const GRID = makeGPURasterMetadata({width: 40, height: 37});
const CELL_COUNT = GRID.width * GRID.height;
const MAXIMUM_ITERATIONS = 7;
const cleanups: (() => void)[] = [];

afterEach(() => {
  for (const cleanup of cleanups.splice(0).reverse()) cleanup();
});

function createFixture(recording = false) {
  const device = createNullWebGPUDevice();
  cleanups.push(() => device.destroy());
  Object.defineProperty(device, 'props', {
    value: {...device.props, _cacheShaders: false, _cachePipelines: false}
  });
  // NullDevice has no compute pipeline; compilation only needs a disposable pipeline handle.
  Object.defineProperty(device, 'createComputePipeline', {
    value: (props: {id?: string}) => ({id: props.id ?? '', hash: '', destroy: () => {}})
  });
  const graph = new GPUCommandGraph(device, {
    id: 'loop-recording',
    ...(recording ? {recording: {}} : {})
  });
  const props = {
    id: 'distance',
    grid: GRID,
    maximumIterations: MAXIMUM_ITERATIONS,
    sweepAfterIteration: 1,
    friction: makeGPURasterBand(
      createTransientView(graph, 'friction', 'float32', CELL_COUNT),
      GRID,
      {id: 'friction'}
    ),
    parameters: createTransientView(graph, 'settings', 'float32', 8),
    sources: createTransientView(graph, 'sources', 'uint32', 2),
    output: {
      costs: createTransientView(graph, 'costs', 'float32', CELL_COUNT),
      backLinks: createTransientView(graph, 'back-links', 'uint32', CELL_COUNT)
    }
  };
  return {graph, props};
}

function buildGraph(recording: boolean, stamps: 'omitted' | 'undefined' | 'present' = 'omitted') {
  const {graph, props} = createFixture(recording);
  const output = {
    ...props.output,
    ...(stamps === 'omitted'
      ? {}
      : {
          settledAt:
            stamps === 'present'
              ? createTransientView(graph, 'settled-at', 'uint32', CELL_COUNT)
              : undefined
        })
  };
  const nodes = new GPUCostDistance({...props, output}).getCommandNodes(graph);
  graph.add({getCommandNodes: () => nodes});
  const kernels = getGPUCommandNodeKernelSources(nodes);
  const compiled = graph.compile();
  cleanups.push(() => compiled.destroy());
  return {nodes, compiled, kernels};
}

it('recording annotates every round and group gate without changing scheduling or kernels', () => {
  const plain = buildGraph(false);
  const recorded = buildGraph(true);
  expect(recorded.compiled.stats.nodeOrder).toEqual(plain.compiled.stats.nodeOrder);
  expect(recorded.kernels).toEqual(plain.kernels);
  expect(plain.nodes.every(node => !('annotation' in node))).toBe(true);
  expect(plain.compiled.describe().graph.nodes.every(node => !('loop' in node))).toBe(true);
  const descriptions = recorded.compiled.describe().graph.nodes;
  for (const id of ['distance-relax', 'distance-tie-relax']) {
    const rounds = descriptions.filter(
      node => node.loop?.id === id && node.attrs?.['loopRole'] === 'round'
    );
    expect(rounds.map(node => node.loop)).toEqual(
      Array.from({length: MAXIMUM_ITERATIONS}, (_, iteration) => ({
        id,
        iteration,
        phase: 'gated',
        maximumIterations: MAXIMUM_ITERATIONS
      }))
    );
    const gates = descriptions.filter(
      node => node.loop?.id === id && node.attrs?.['loopRole'] === 'gate'
    );
    expect(gates.map(node => node.loop)).toEqual(
      [3, 6].map(iteration => ({
        id,
        iteration,
        phase: 'gated',
        maximumIterations: MAXIMUM_ITERATIONS
      }))
    );
    expect(
      descriptions.find(node => node.id === `${id}-gate-slots-write`)?.attrs?.['loopRole']
    ).toBe('setup');
    expect(rounds.every(node => node.source && node.op === 'GPUCostDistance')).toBe(true);
  }
});

it('gated loops preserve a round node own loop and other annotations', () => {
  const {graph} = createFixture(true);
  const loop = {id: 'nested', iteration: 2, phase: 'inner'};
  const nodes = createRasterGatedLoopNodes(graph, {
    id: 'outer',
    operation: 'TestLoop',
    maxIterations: 1,
    state: createRasterIterationState(graph, 'outer', 'TestLoop', 1),
    createRound: ({nodeId}) =>
      createWGSLKernelNode(graph, {
        id: nodeId,
        operation: 'TestLoop',
        bindings: [],
        invocationCount: 1,
        body: '',
        record: {loop, label: 'Nested round', attrs: {retained: true, loopRole: 'old'}}
      })
  });
  expect(nodes[1].annotation).toMatchObject({
    loop,
    label: 'Nested round',
    attrs: {retained: true, loopRole: 'round'}
  });
});

it('settledAt adds one reset and only instruments the main cost round kernels', () => {
  const plain = buildGraph(false);
  const explicitUndefined = buildGraph(false, 'undefined');
  const stamped = buildGraph(true, 'present');
  expect(explicitUndefined.compiled.stats.nodeOrder).toEqual(plain.compiled.stats.nodeOrder);
  expect(explicitUndefined.kernels).toEqual(plain.kernels);
  const order = [...plain.compiled.stats.nodeOrder];
  order.splice(order.indexOf('distance-relax-gate-slots-write'), 0, 'distance-settled-at-reset');
  expect(stamped.compiled.stats.nodeOrder).toEqual(order);
  expect(plain.kernels.every(kernel => !kernel.source.includes('settledAt'))).toBe(true);
  const originals = new Map(plain.kernels.map(kernel => [kernel.id, kernel]));
  for (const kernel of stamped.kernels) {
    if (/^distance-relax-\d+$/.test(kernel.id)) {
      expect(kernel.source).toContain(
        'settledAt[settledAtOffset + row * GRID_WIDTH + column] = iteration + 1u;'
      );
    } else if (kernel.id === 'distance-settled-at-reset') {
      expect(kernel.source).toContain('settledAt[settledAtOffset + index] = 0u;');
      expect(kernel.constants['INVOCATION_COUNT']).toBe(CELL_COUNT);
    } else {
      expect(kernel).toEqual(originals.get(kernel.id));
    }
  }
});

it('settledAt validates per-cell packed uint32 storage, distinctness and graph ownership', () => {
  const {graph, props} = createFixture();
  const settledAt = createTransientView(graph, 'settled-at', 'uint32', CELL_COUNT);
  const makeContributor = (view = settledAt) =>
    new GPUCostDistance({...props, output: {...props.output, settledAt: view}});
  expect(() =>
    makeContributor(createTransientView(graph, 'short', 'uint32', CELL_COUNT - 1))
  ).toThrow(/one value per cell/);
  expect(() =>
    makeContributor(
      graph.createDataView(settledAt.buffer, {format: 'uint32', length: 2, byteStride: 8})
    )
  ).toThrow(/packed/);
  // Deliberately give the runtime validator a view with the wrong format.
  expect(
    () =>
      new GPUCostDistance({
        ...props,
        output: {
          ...props.output,
          // @ts-expect-error settledAt requires uint32.
          settledAt: props.output.costs
        }
      })
  ).toThrow(/uint32/);
  expect(() => makeContributor(props.output.backLinks)).toThrow(/must not share buffers/);
  const friction = props.friction.storage;
  if (friction.kind === 'buffer') {
    expect(() =>
      makeContributor(
        graph.createDataView(friction.values.buffer, {format: 'uint32', length: CELL_COUNT})
      )
    ).toThrow(/must not share buffers/);
  }
  const foreign = createFixture().graph;
  expect(() =>
    makeContributor(createTransientView(foreign, 'foreign', 'uint32', CELL_COUNT)).getCommandNodes(
      graph
    )
  ).toThrow(/graph/i);
});

it('settledAt rejects continuation at the cost-distance and tiled-relaxation boundaries', () => {
  const {graph, props} = createFixture();
  const settledAt = createTransientView(graph, 'settled-at', 'uint32', CELL_COUNT);
  const continuation = {
    state: createTransientView(
      graph,
      'state',
      'uint32',
      getGPURasterRelaxationContinuationStateLength(GRID)
    ),
    values: createTransientView(graph, 'working', 'float32', CELL_COUNT)
  };
  expect(
    () =>
      new GPUCostDistance({...props, continuation, output: {costs: props.output.costs, settledAt}})
  ).toThrow(/continuation.*settledAt/);
  const relaxationProps = {
    id: 'tiled',
    operation: 'Test',
    width: GRID.width,
    height: GRID.height,
    cellSizeMode: 'uniform' as const,
    maxIterations: 4
  };
  const {relaxation} = createRasterTiledRelaxation(graph, {...relaxationProps, continuation}, true);
  expect(() =>
    createRasterTiledRelaxationNodes(graph, {
      ...relaxationProps,
      relaxation,
      settledAt,
      values: props.output.costs,
      auxiliary: props.output.costs,
      settings: props.parameters,
      declarations: ''
    })
  ).toThrow(/settledAt.*continuation/);
});
