// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, GPUMask, createTransientView} from '@luma.gl/gpgpu/gpu-core';
import {expect, test} from 'vitest';
import {
  captureGraphCommandNodes,
  getGraphViewChunks,
  importGraphBuffer,
  validateGraphViewsBelongToGraph,
  validateMapGraphCompactOutput,
  type GPUMapGraphRecipe
} from '../../src/map-graphs';
import {GPUPairwisePointDistance} from '../../src/geospatial';
import {createNullWebGPUDevice} from './map-graph-test-utils';

test('captureGraphCommandNodes records addToGraph-style nodes without scheduling them', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'capture'});
  const input = createTransientView(graph, 'input', 'uint32', 16);
  const output = createTransientView(graph, 'output', 'uint32', 16);
  const originalAddComputePass = graph.addComputePass;

  const nodes = captureGraphCommandNodes(graph, () => {
    graph.add(new GPUMask({id: 'captured-mask', inputs: [input], output, operation: 'not'}));
  });

  expect(nodes.map(node => [node.type, node.id])).toEqual([['compute', 'captured-mask']]);
  expect(graph.addComputePass, 'graph methods are restored').toBe(originalAddComputePass);
  expect(Object.hasOwn(graph, 'addComputePass')).toBe(false);
  device.destroy();
});

test('captureGraphCommandNodes bridges experimental addToGraph contributors', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'capture-geospatial'});
  const left = createTransientView(graph, 'left', 'float32x2', 4);
  const right = createTransientView(graph, 'right', 'float32x2', 4);
  const output = createTransientView(graph, 'distance', 'float32', 4);

  const nodes = captureGraphCommandNodes(graph, () => {
    new GPUPairwisePointDistance({id: 'distance', left, right, output}).addToGraph(graph);
  });

  expect(nodes.length).toBeGreaterThan(0);
  expect(nodes.every(node => node.id.startsWith('distance'))).toBe(true);
  device.destroy();
});

test('a recipe built on captureGraphCommandNodes composes through graph.add', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'recipe-composition'});
  const input = createTransientView(graph, 'input', 'uint32', 8);
  const output = createTransientView(graph, 'output', 'uint32', 8);
  const scheduled: string[] = [];
  const addComputePass = graph.addComputePass.bind(graph);
  graph.addComputePass = node => {
    scheduled.push(node.id);
    addComputePass(node);
  };

  class InvertRecipe implements GPUMapGraphRecipe {
    readonly id = 'invert';
    readonly recipe = 'invert-example';
    getCommandNodes<Parameters>(target: GPUCommandGraph<Parameters>) {
      return captureGraphCommandNodes(target, () => {
        target.add(new GPUMask({id: `${this.id}-mask`, inputs: [input], output, operation: 'not'}));
      });
    }
  }
  graph.add(new InvertRecipe());

  expect(scheduled).toEqual(['invert-mask']);
  device.destroy();
});

test('map-graph validation helpers enforce graph membership and compact output layout', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'validation'});
  const otherGraph = new GPUCommandGraph(device, {id: 'other'});
  const buffer = device.createBuffer({byteLength: 64, usage: Buffer.STORAGE | Buffer.COPY_SRC});
  const ids = importGraphBuffer(graph, 'ids', buffer, 'uint32');
  const count = createTransientView(graph, 'count', 'uint32', 1);
  const overflow = createTransientView(graph, 'overflow', 'uint32', 1);
  const foreign = createTransientView(otherGraph, 'foreign', 'uint32', 1);

  expect(ids.length, 'length defaults to whole rows').toBe(16);
  expect(getGraphViewChunks(ids)).toEqual([ids]);
  expect(() => validateGraphViewsBelongToGraph('recipe', graph, [ids, undefined])).not.toThrow();
  expect(() => validateGraphViewsBelongToGraph('recipe', graph, [foreign])).toThrow(/target graph/);
  expect(() => validateMapGraphCompactOutput('recipe', {ids, count, overflow})).not.toThrow();
  expect(() =>
    validateMapGraphCompactOutput('recipe', {
      ids,
      count: createTransientView(graph, 'empty-count', 'uint32', 0),
      overflow
    })
  ).toThrow(/count must contain one uint32 row/);
  buffer.destroy();
  device.destroy();
});
