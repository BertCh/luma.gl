// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import type {Device} from '@luma.gl/core';
import {
  GPUCommandGraph,
  createTransientView,
  type GPUCommandGraphComputeNode
} from '@luma.gl/gpgpu/gpu-core';
import {
  GPUAdjacencyMatrix,
  GPUNetworkAnalyticsColumns,
  GPUNetworkCentrality,
  GPUNetworkLineGraph,
  GPUNetworkStatistics,
  GPUNetworkSubgraphFilter,
  getGPUNetworkStatisticsLength
} from '../../../src/gpu-network/index';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

type ComputeNode = Omit<GPUCommandGraphComputeNode<void>, 'type'>;

// Null devices cannot create compute pipelines: keep the node structure, drop the compile callbacks,
// and remember the original nodes so their generated WGSL can still be compared.
class StructureGraph extends GPUCommandGraph {
  readonly originals: unknown[] = [];

  override addComputePass(node: ComputeNode): void {
    this.originals.push({...node, type: 'compute'});
    const {compile: _compile, compileAsync: _compileAsync, ...structure} = node;
    super.addComputePass({
      ...structure,
      compile: () => ({
        encode: () => {
          throw new Error('Structure-only graph cannot execute');
        }
      })
    });
  }
}

type Views = {
  uint: (id: string, length: number) => ReturnType<typeof createTransientView<'uint32'>>;
  float: (id: string, length: number) => ReturnType<typeof createTransientView<'float32'>>;
  point: (id: string, length: number) => ReturnType<typeof createTransientView<'float32x2'>>;
};

const BUILDERS: Record<string, (graph: StructureGraph, view: Views) => void> = {
  GPUNetworkAnalyticsColumns: (graph, view) => {
    graph.add(
      new GPUNetworkAnalyticsColumns({
        id: 'analytics',
        topology: {offsets: view.uint('offsets', 5), neighbors: view.uint('neighbors', 8)},
        degree: {
          output: view.uint('degree', 4),
          normalized: view.float('normalized', 4),
          extent: view.uint('extent', 2)
        }
      })
    );
  },
  GPUNetworkCentrality: (graph, view) => {
    graph.add(
      new GPUNetworkCentrality({
        id: 'centrality',
        topology: {
          offsets: view.uint('offsets', 5),
          neighbors: view.uint('neighbors', 8),
          weights: view.float('weights', 8)
        },
        reverseTopology: {
          offsets: view.uint('reverse-offsets', 5),
          neighbors: view.uint('reverse-neighbors', 8),
          weights: view.float('reverse-weights', 8)
        },
        pivotNodes: view.uint('pivots', 2),
        laneCount: 2,
        maximumIterations: 3,
        output: {
          closeness: view.float('closeness', 2),
          reachedCounts: view.uint('reached', 2),
          status: {converged: view.uint('converged', 1)}
        }
      })
    );
  },
  GPUNetworkStatistics: (graph, view) => {
    graph.add(
      new GPUNetworkStatistics({
        id: 'statistics',
        topology: {offsets: view.uint('offsets', 5), neighbors: view.uint('neighbors', 8)},
        directed: true,
        degreeBinCount: 8,
        maximumComponentIterations: 3,
        output: view.uint('summary', getGPUNetworkStatisticsLength(8))
      })
    );
  },
  GPUNetworkSubgraphFilter: (graph, view) => {
    graph.add(
      new GPUNetworkSubgraphFilter({
        id: 'filter',
        topology: {offsets: view.uint('offsets', 5), neighbors: view.uint('neighbors', 8)},
        directed: true,
        edgeColumns: [view.float('weights', 8)],
        parameters: view.float('parameters', 4),
        dropIsolated: true,
        output: {
          nodeMask: view.uint('node-mask', 4),
          edgeMask: view.uint('edge-mask', 8),
          counts: view.uint('counts', 4)
        }
      })
    );
  },
  GPUAdjacencyMatrix: (graph, view) => {
    graph.add(
      new GPUAdjacencyMatrix({
        id: 'matrix',
        topology: {offsets: view.uint('offsets', 5), neighbors: view.uint('neighbors', 8)},
        directed: true,
        order: view.uint('order', 4),
        resolution: 4,
        output: {counts: view.uint('counts', 16), maximumCount: view.uint('maximum', 1)}
      })
    );
  },
  GPUNetworkLineGraph: (graph, view) => {
    graph.add(
      new GPUNetworkLineGraph({
        id: 'line',
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
        topology: {
          offsets: view.uint('offsets', 5),
          neighbors: view.uint('neighbors', 8),
          weights: view.float('weights', 8),
          nodePositions: view.point('positions', 4)
        },
        parameters: view.float('parameters', 8),
        output: {
          topology: {
            offsets: view.uint('line-offsets', 9),
            neighbors: view.uint('line-neighbors', 16),
            weights: view.float('line-weights', 16)
          },
          arcCount: view.uint('arc-count', 1),
          status: {overflow: view.uint('overflow', 1)}
        }
      })
    );
  }
};

function build(device: Device, name: string, recording: boolean) {
  const graph = new StructureGraph(device, {
    id: `${name}-${recording}`,
    ...(recording ? {recording: {}} : {})
  });
  const view: Views = {
    uint: (id, length) => createTransientView(graph, id, 'uint32', length),
    float: (id, length) => createTransientView(graph, id, 'float32', length),
    point: (id, length) => createTransientView(graph, id, 'float32x2', length)
  };
  BUILDERS[name](graph, view);
  // The workload operation label of composed primitives is not part of the generated kernel.
  const kernels = getGPUCommandNodeKernelSources(graph.originals).map(
    ({operation: _operation, ...kernel}) => kernel
  );
  return {compiled: graph.compile(), kernels};
}

for (const name of Object.keys(BUILDERS))
  test(`${name}: record metadata preserves node order and every generated kernel`, () => {
    const device = createNullWebGPUDevice();
    const production = build(device, name, false);
    const recording = build(device, name, true);
    try {
      expect(recording.kernels.length).toBeGreaterThan(0);
      expect(recording.compiled.stats.nodeOrder).toEqual(production.compiled.stats.nodeOrder);
      expect(recording.kernels).toEqual(production.kernels);
      const formulas = recording.compiled
        .describe()
        .graph.nodes.filter(node => node.formula?.startsWith('gpu-network.'));
      expect(formulas.length).toBeGreaterThan(0);
      expect(
        production.compiled.describe().graph.nodes.every(node => node.formula === undefined)
      ).toBe(true);
    } finally {
      production.compiled.destroy();
      recording.compiled.destroy();
      device.destroy();
    }
  });
