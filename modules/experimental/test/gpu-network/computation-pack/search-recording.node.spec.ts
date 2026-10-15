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
  GPUNetworkAccessibility,
  GPUNetworkIsochrones,
  GPUNetworkNeighborhood,
  GPUNetworkServiceAreas,
  GPUNetworkSnapping,
  addDriveTimeCatchmentRecipe
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
  GPUNetworkServiceAreas: (graph, view) => {
    graph.add(
      new GPUNetworkServiceAreas({
        id: 'areas',
        maximumIterations: 3,
        topology: {
          offsets: view.uint('offsets', 5),
          neighbors: view.uint('neighbors', 8),
          weights: view.float('weights', 8)
        },
        facilities: view.uint('facilities', 2),
        output: {
          assignments: view.uint('assignments', 4),
          costs: view.float('costs', 4),
          status: {converged: view.uint('converged', 1)}
        }
      })
    );
  },
  GPUNetworkNeighborhood: (graph, view) => {
    graph.add(
      new GPUNetworkNeighborhood({
        id: 'ego',
        maximumHops: 3,
        topology: {offsets: view.uint('offsets', 5), neighbors: view.uint('neighbors', 8)},
        startNodes: view.uint('starts', 1),
        hops: view.uint('hops', 1),
        output: {hopDistances: view.uint('hop-distances', 4)}
      })
    );
  },
  GPUNetworkIsochrones: (graph, view) => {
    graph.add(
      new GPUNetworkIsochrones({
        id: 'iso',
        maximumIterations: 3,
        topology: {
          offsets: view.uint('offsets', 5),
          neighbors: view.uint('neighbors', 8),
          weights: view.float('weights', 8),
          nodePositions: view.point('positions', 4)
        },
        costs: view.float('costs', 4),
        startNodes: view.uint('starts', 1),
        costBreaks: view.float('breaks', 3),
        parameters: view.float('parameters', 12),
        output: {status: {converged: view.uint('converged', 1)}},
        raster: {
          width: 8,
          height: 8,
          mode: 'min',
          output: {
            triangles: view.point('triangles', 3 * 64),
            triangleBands: view.uint('bands', 64),
            status: {count: view.uint('count', 1), overflow: view.uint('overflow', 1)}
          }
        }
      })
    );
  },
  GPUNetworkSnapping: (graph, view) => {
    graph.add(
      new GPUNetworkSnapping({
        id: 'snap',
        positions: view.point('points', 2),
        topology: {
          offsets: view.uint('offsets', 5),
          neighbors: view.uint('neighbors', 8),
          nodePositions: view.point('positions', 4)
        },
        output: {snappedEdges: view.uint('edges', 2), snapDistances: view.float('distances', 2)}
      })
    );
  },
  GPUNetworkAccessibility: (graph, view) => {
    graph.add(
      new GPUNetworkAccessibility({
        id: 'access',
        costs: view.float('costs', 8),
        opportunityWeights: view.float('weights', 2),
        parameters: view.float('parameters', 4),
        gravity: view.float('gravity', 4)
      })
    );
  },
  addDriveTimeCatchmentRecipe: (graph, view) => {
    addDriveTimeCatchmentRecipe(graph, {
      id: 'catchment',
      maximumIterations: 3,
      topology: {
        offsets: view.uint('offsets', 5),
        neighbors: view.uint('neighbors', 8),
        weights: view.float('weights', 8),
        nodePositions: view.point('positions', 4)
      },
      facilities: view.point('facilities', 2),
      demand: view.point('demand', 6),
      costBreaks: view.float('breaks', 3)
    });
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
