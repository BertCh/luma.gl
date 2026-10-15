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
  GPUEdgeBundling,
  GPUFlowAggregation,
  GPUNetworkCostTable,
  GPUNetworkCostWeights,
  GPUNetworkKFunction
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

type Case = {
  build: (graph: StructureGraph, view: Views) => void;
  /** Formula ids the recording build must stamp on kernels. */
  formulas: string[];
  /** The recording build must stamp an unrolled loop coordinate. */
  loop?: string;
};

const CASES: Record<string, Case> = {
  GPUNetworkCostTable: {
    formulas: ['gpu-network.cost-table.admission'],
    build: (graph, view) => {
      graph.add(
        new GPUNetworkCostTable({
          id: 'table',
          topology: {
            offsets: view.uint('offsets', 5),
            neighbors: view.uint('neighbors', 8),
            weights: view.float('weights', 8)
          },
          costLimit: view.float('limit', 1),
          laneCount: 2,
          maximumIterations: 3,
          output: {
            offsets: view.uint('table-offsets', 5),
            targets: view.uint('table-targets', 16),
            costs: view.float('table-costs', 16)
          }
        })
      );
    }
  },
  GPUNetworkCostWeights: {
    formulas: ['gpu-network.cost-weights.decay'],
    build: (graph, view) => {
      graph.add(
        new GPUNetworkCostWeights({
          id: 'weights',
          costs: view.float('costs', 8),
          rowCount: 2,
          parameters: view.float('parameters', 4),
          output: {
            offsets: view.uint('weight-offsets', 3),
            neighbors: view.uint('weight-neighbors', 8),
            weights: view.float('weight-values', 8)
          }
        })
      );
    }
  },
  GPUNetworkKFunction: {
    formulas: ['gpu-network.k-function.k'],
    build: (graph, view) => {
      graph.add(
        new GPUNetworkKFunction({
          id: 'k',
          positions: view.point('events', 3),
          topology: {
            offsets: view.uint('offsets', 5),
            neighbors: view.uint('neighbors', 8),
            weights: view.float('weights', 8),
            nodePositions: view.point('positions', 4)
          },
          maximumDistance: view.float('maximum-distance', 1),
          networkLength: view.float('network-length', 1),
          parameters: view.uint('parameters', 4),
          bandCount: 4,
          rowsPerBlock: 3,
          maximumIterations: 2,
          output: {kValues: view.float('k-values', 4), pairCounts: view.uint('pair-counts', 4)}
        })
      );
    }
  },
  GPUFlowAggregation: {
    formulas: ['gpu-network.flow-aggregation.pair-sum'],
    build: (graph, view) => {
      graph.add(
        new GPUFlowAggregation({
          id: 'flows',
          zones: {kind: 'ids', zoneCount: 4},
          originZoneIds: view.uint('origins', 6),
          destinationZoneIds: view.uint('destinations', 6),
          weights: view.float('trips', 6),
          sumOrder: 'sorted',
          pairCapacity: 16,
          output: {
            ids: view.uint('keys', 4),
            flowWeights: view.float('flow-weights', 4),
            status: {count: view.uint('count', 1), overflow: view.uint('overflow', 1)}
          }
        })
      );
    }
  },
  GPUEdgeBundling: {
    formulas: [],
    loop: 'bundling-iteration',
    build: (graph, view) => {
      graph.add(
        new GPUEdgeBundling({
          id: 'bundling',
          nodePositions: view.point('positions', 3),
          sourceNodes: view.uint('sources', 2),
          targetNodes: view.uint('targets', 2),
          pointsPerEdge: 4,
          maximumIterations: 2,
          densityResolution: 8,
          output: {paths: view.point('paths', 8)}
        })
      );
    }
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
  CASES[name].build(graph, view);
  // The workload operation label of composed primitives is not part of the generated kernel.
  const kernels = getGPUCommandNodeKernelSources(graph.originals).map(
    ({operation: _operation, ...kernel}) => kernel
  );
  return {compiled: graph.compile(), kernels};
}

for (const name of Object.keys(CASES))
  test(`${name}: record metadata preserves node order and every generated kernel`, () => {
    const device = createNullWebGPUDevice();
    const production = build(device, name, false);
    const recording = build(device, name, true);
    try {
      expect(recording.kernels.length).toBeGreaterThan(0);
      expect(recording.compiled.stats.nodeOrder).toEqual(production.compiled.stats.nodeOrder);
      expect(recording.kernels).toEqual(production.kernels);
      const nodes = recording.compiled.describe().graph.nodes;
      for (const formula of CASES[name].formulas)
        expect(nodes.some(node => node.formula === formula)).toBe(true);
      if (CASES[name].loop)
        expect(nodes.some(node => node.loop?.id.endsWith(CASES[name].loop!))).toBe(true);
      expect(
        production.compiled.describe().graph.nodes.every(node => node.loop === undefined)
      ).toBe(true);
      expect(
        production.compiled.describe().graph.nodes.every(node => node.formula === undefined)
      ).toBe(true);
    } finally {
      production.compiled.destroy();
      recording.compiled.destroy();
      device.destroy();
    }
  });
