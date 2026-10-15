// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/gpgpu/gpu-core';
import {GPUNetworkReachability} from '../../../src/gpu-network/network-reachability';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';

test('record metadata preserves node order and every generated kernel', () => {
  const device = createNullWebGPUDevice();
  const build = (recording: boolean) => {
    const graph = new GPUCommandGraph(device, {
      id: 'record-equality',
      ...(recording ? {recording: {}} : {})
    });
    const view = (id: string, length: number) => createTransientView(graph, id, 'uint32', length);
    const contributor = new GPUNetworkReachability({
      id: 'reach',
      maximumIterations: 4,
      maximumTieIterations: 3,
      localIterations: 1,
      topology: {
        offsets: view('offsets', 5),
        neighbors: view('neighbors', 4),
        weights: createTransientView(graph, 'weights', 'float32', 4)
      },
      startNodes: view('starts', 1),
      costBreaks: createTransientView(graph, 'thresholds', 'float32', 5),
      output: {
        costs: createTransientView(graph, 'costs', 'float32', 4),
        bands: view('bands', 4),
        bandCounts: view('counts', 5),
        predecessors: view('predecessors', 4),
        status: {converged: view('converged', 1), iterationCount: view('iterationCount', 1)}
      }
    });
    const nodes = contributor.getCommandNodes(graph);
    const kernels = getGPUCommandNodeKernelSources(nodes);
    expect(getParsedKernelSources(nodes).size).toBeGreaterThan(0);
    graph.add(
      nodes.map(node => {
        const {compile: _compile, compileAsync: _compileAsync, ...structure} = node;
        return {
          ...structure,
          compile: () => ({
            encode: () => {
              throw new Error('Structure-only graph cannot execute');
            }
          })
        };
      })
    );
    return {compiled: graph.compile(), kernels};
  };
  const production = build(false);
  const recording = build(true);
  try {
    expect(recording.compiled.stats.nodeOrder).toEqual(production.compiled.stats.nodeOrder);
    expect(recording.kernels).toEqual(production.kernels);
    const nodes = recording.compiled.describe().graph.nodes;
    expect(
      nodes.filter(node => /^reach-relax-\d+$/.test(node.id)).map(node => node.loop?.iteration)
    ).toEqual([0, 1, 2, 3]);
    expect(
      nodes.filter(node => /^reach-tie-level-\d+$/.test(node.id)).map(node => node.loop?.iteration)
    ).toEqual([0, 1, 2]);
    expect(production.compiled.describe().graph.nodes.every(node => node.loop === undefined)).toBe(
      true
    );
  } finally {
    production.compiled.destroy();
    recording.compiled.destroy();
    device.destroy();
  }
});
